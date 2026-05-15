"""
trainer.py - training loop

Supports:
  - Standard single-task training (Section 5.3)
  - Multi-stage continual training (full, limited, class-incremental)
  - Warm-start pretraining + fine-tuning (Section 3.2)
  - Gradient clipping (ResNet-18 needs this per Appendix F.3)
  - Optional plasticity metric logging (expensive, off by default)
  - CSV output to results/ directory

Example on how to use:
    from src.trainer import Trainer, TrainerConfig
    cfg = TrainerConfig(arch="resnet18", dataset="cifar100",
                        activation="aid", act_kwargs={"p": 0.7},
                        num_classes=100, learning_rate=1e-3, device="cuda")
    trainer = Trainer(cfg)
    results = trainer.run_continual_full()
"""

import csv
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import build_model
from .metrics import compute_all_metrics


@dataclass
class TrainerConfig:
    # model
    arch: str = "resnet18"
    num_classes: int = 100

    # activation
    activation: str = "aid"
    act_kwargs: dict = field(default_factory=lambda: {"p": 0.7})
    dropout_p: Optional[float] = None  # only for standard Dropout baseline

    # dataset
    dataset: str = "cifar100"
    data_root: str = "data"

    # optimiser
    optimiser: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9        # SGD only
    grad_clip: Optional[float] = None  # 0.5 for ResNet-18

    # LR schedule for standard supervised learning
    lr_decay_epochs: list = field(default_factory=lambda: [100, 150])
    lr_decay_factor: float = 0.1

    # training duration
    epochs_per_stage: int = 100
    batch_size: int = 256
    num_stages: int = 10
    reset_optimiser: bool = True  # reset at each continual stage

    # logging
    log_dir: str = "results"
    run_name: str = ""
    compute_plasticity_metrics: bool = False  # slow - only enable for analysis
    metrics_interval: int = 10  # compute every N epochs if enabled
    save_checkpoints: bool = False
    checkpoint_dir: str = "checkpoints"

    # hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 13


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        _set_seed(cfg.seed)

        if not cfg.run_name:
            cfg.run_name = f"{cfg.arch}_{cfg.dataset}_{cfg.activation}_seed{cfg.seed}"

        os.makedirs(cfg.log_dir, exist_ok=True)
        self.log_path = os.path.join(cfg.log_dir, f"{cfg.run_name}.csv")
        self.device   = torch.device(cfg.device)

        self.model = build_model(
            cfg.arch, cfg.num_classes, cfg.activation, cfg.act_kwargs, cfg.dropout_p
        ).to(self.device)

        self._log_file   = open(self.log_path, "w", newline="")
        self._csv_writer = None

    # Public experiment runners

    def run_continual_full(self):
        from .data import continual_full_tasks
        tasks = list(continual_full_tasks(
            self.cfg.dataset, self.cfg.num_stages,
            self.cfg.batch_size, self.cfg.data_root, seed=self.cfg.seed,
        ))
        return self._run_stages(tasks, "continual_full")

    def run_continual_limited(self):
        from .data import continual_limited_tasks
        tasks = list(continual_limited_tasks(
            self.cfg.dataset, self.cfg.num_stages,
            self.cfg.batch_size, self.cfg.data_root, seed=self.cfg.seed,
        ))
        return self._run_stages(tasks, "continual_limited")

    def run_class_incremental(self):
        from .data import class_incremental_tasks
        tasks = list(class_incremental_tasks(
            self.cfg.dataset, self.cfg.num_stages,
            self.cfg.batch_size, self.cfg.data_root,
        ))
        return self._run_stages(tasks, "class_incremental")

    def run_trainability(self, train_loaders, test_loader):
        """
        Trainability experiment: each entry in train_loaders is one task.
        The model is never reset between tasks.
        """
        opt = self._make_opt()
        all_rows = []

        task_bar = tqdm(enumerate(train_loaders), total=len(train_loaders),
                        desc="Tasks", unit="task", dynamic_ncols=True)

        for task_idx, train_loader in task_bar:
            rows = self._train_stage(opt, train_loader, test_loader, task_idx)
            all_rows.extend(rows)
            task_bar.set_postfix(acc=f"{rows[-1]['test_acc']:.3f}")

        self._log_file.close()
        return all_rows

    def run_standard_supervised(self, train_loader, test_loader, total_epochs=200):
        """Standard classification with LR decay at lr_decay_epochs."""
        opt = self._make_opt()
        all_rows = []

        epoch_bar = tqdm(range(1, total_epochs + 1), desc="Epochs",
                         unit="epoch", dynamic_ncols=True)

        for epoch in epoch_bar:
            if epoch in self.cfg.lr_decay_epochs:
                for pg in opt.param_groups:
                    pg["lr"] *= self.cfg.lr_decay_factor
                tqdm.write(f"  LR decayed -> {opt.param_groups[0]['lr']:.2e}")

            train_loss, train_acc = self._train_epoch(opt, train_loader)
            test_acc = self._eval(test_loader)

            row = {"epoch": epoch, "train_loss": round(train_loss, 6),
                   "train_acc": round(train_acc, 6), "test_acc": round(test_acc, 6)}
            self._write_row(row)
            all_rows.append(row)

            epoch_bar.set_postfix(
                loss=f"{train_loss:.4f}",
                train=f"{train_acc:.3f}",
                test=f"{test_acc:.3f}",
            )

        self._log_file.close()
        return all_rows

    def run_warm_start(self, pretrain_loader, full_loader, test_loader,
                       pretrain_epochs=1000, finetune_epochs=100):
        """
        Warm-start experiment from Section 3.2:
        1. Pretrain on 10% of data for pretrainn_epochs
        2. Reset optimiser
        3. Fine-tune on full dataset for finetune_epochs
        """
        opt = self._make_opt()
        all_rows = []

        print("=== Pretraining ===")
        pretrain_bar = tqdm(range(1, pretrain_epochs + 1), desc="Pretrain",
                            unit="epoch", dynamic_ncols=True)
        for epoch in pretrain_bar:
            train_loss, train_acc = self._train_epoch(opt, pretrain_loader)
            test_acc = self._eval(test_loader)
            row = {"phase": "pretrain", "epoch": epoch,
                   "train_loss": round(train_loss, 6),
                   "train_acc": round(train_acc, 6), "test_acc": round(test_acc, 6)}
            self._write_row(row)
            all_rows.append(row)
            pretrain_bar.set_postfix(loss=f"{train_loss:.4f}", test=f"{test_acc:.3f}")

        opt = self._make_opt()  # reset optimiser before fine-tuning

        print("=== Fine-tuning ===")
        finetune_bar = tqdm(range(1, finetune_epochs + 1), desc="Fine-tune",
                            unit="epoch", dynamic_ncols=True)
        for epoch in finetune_bar:
            train_loss, train_acc = self._train_epoch(opt, full_loader)
            test_acc = self._eval(test_loader)
            row = {"phase": "finetune", "epoch": epoch,
                   "train_loss": round(train_loss, 6),
                   "train_acc": round(train_acc, 6), "test_acc": round(test_acc, 6)}
            self._write_row(row)
            all_rows.append(row)
            finetune_bar.set_postfix(loss=f"{train_loss:.4f}", test=f"{test_acc:.3f}")

        self._log_file.close()
        return all_rows

    # Internal helpers

    def _run_stages(self, tasks, setting):
        opt = self._make_opt()
        all_rows = []

        for stage, (train_loader, test_loader) in enumerate(tasks):
            if self.cfg.reset_optimiser and stage > 0:
                opt = self._make_opt()

            print(f"\n[{self.cfg.run_name}] Stage {stage+1}/{len(tasks)}")
            rows = self._train_stage(opt, train_loader, test_loader, stage)
            all_rows.extend(rows)
            print(f"  -> stage done | test acc: {rows[-1]['test_acc']:.4f}")

            if self.cfg.save_checkpoints:
                self._save_ckpt(stage)

        self._log_file.close()
        return all_rows

    def _train_stage(self, opt, train_loader, test_loader, stage):
        rows = []

        epoch_bar = tqdm(
            range(1, self.cfg.epochs_per_stage + 1),
            desc=f"  Stage {stage+1}",
            unit="epoch",
            leave=True,
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            train_loss, train_acc = self._train_epoch(opt, train_loader, epoch, stage)
            test_acc = self._eval(test_loader)

            global_epoch = stage * self.cfg.epochs_per_stage + epoch
            row = {
                "stage": stage,
                "epoch": epoch,
                "global_epoch": global_epoch,
                "train_loss": round(train_loss, 6),
                "train_acc":  round(train_acc, 6),
                "test_acc":   round(test_acc, 6),
            }

            if (self.cfg.compute_plasticity_metrics
                    and epoch % self.cfg.metrics_interval == 0):
                row.update(compute_all_metrics(self.model, test_loader, self.device))

            self._write_row(row)
            rows.append(row)

            # update the epoch bar with current metrics
            epoch_bar.set_postfix(
                loss=f"{train_loss:.4f}",
                train=f"{train_acc:.3f}",
                test=f"{test_acc:.3f}",
            )

        return rows

    def _train_epoch(self, opt, loader, epoch=None, stage=None):
        """Run one epoch of training, returning (mean_loss, train_accuracy)."""
        self.model.train()
        criterion  = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct    = 0
        total      = 0
        running_loss = 0.0

        # batch-level bar shows activity within the epoch
        # leave = False so it clears after each epoch and doesn't flood the terminal
        desc = f"    batches"
        batch_bar = tqdm(
            loader,
            desc=desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for x, y in batch_bar:
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad()
            logits = self.model(x)
            loss   = criterion(logits, y)
            loss.backward()

            if self.cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            opt.step()

            batch_loss  = loss.item()
            total_loss += batch_loss * x.size(0)
            correct    += logits.argmax(1).eq(y).sum().item()
            total      += x.size(0)

            # show a running loss on the batch bar so you can see it training
            running_loss = total_loss / total
            batch_bar.set_postfix(loss=f"{running_loss:.4f}")

        return total_loss / total, correct / total

    @torch.no_grad()
    def _eval(self, loader):
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            correct += self.model(x).argmax(1).eq(y).sum().item()
            total   += x.size(0)
        return correct / total if total > 0 else 0.0

    def _make_opt(self):
        params = self.model.parameters()
        if self.cfg.optimiser.lower() == "adam":
            return optim.Adam(params, lr=self.cfg.learning_rate,
                              weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimiser.lower() == "sgd":
            return optim.SGD(params, lr=self.cfg.learning_rate,
                             momentum=self.cfg.momentum,
                             weight_decay=self.cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimiser '{self.cfg.optimiser}'")

    def _write_row(self, row):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._log_file, fieldnames=list(row.keys()), extrasaction="ignore"
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._log_file.flush()

    def _save_ckpt(self, stage):
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.cfg.checkpoint_dir,
                            f"{self.cfg.run_name}_stage{stage}.pt")
        torch.save({"stage": stage, "model": self.model.state_dict(),
                    "config": self.cfg}, path)

# Seed helper (outside class so it can be imported standalone if needed

def _set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
