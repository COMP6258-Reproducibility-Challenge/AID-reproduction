"""
run_experiments.py - script to run all experiments.

Usage:
    python run_experiments.py --experiment all
    python run_experiments.py --experiment supervised --dataset cifar100
    python run_experiments.py --experiment trainability --quick
    python run_experiments.py --experiment continual_full --device cuda

The --quick flag is used for testing all the experiments quickly. It runs everything with very few epochs
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trainer import Trainer, TrainerConfig
from src.plotting import (
    plot_trainability,
    plot_continual_accuracy,
    plot_class_incremental_relative,
    plot_generalisation_gap,
    plot_learning_curves,
    print_supervised_table,
)

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
DATA_ROOT   = "data"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def fig(name):
    return os.path.join(FIGURES_DIR, f"{name}.png")


# Shared config builders


# architecture defaults for each dataset (matching paper Appendix F.3)
ARCH_CFG = {
    "cifar10":      {"arch": "cnn",      "num_classes": 10,  "lr": 1e-3, "clip": None},
    "cifar100":     {"arch": "resnet18",  "num_classes": 100, "lr": 1e-3, "clip": 0.5},
    "tinyimagenet": {"arch": "vgg16",     "num_classes": 200, "lr": 1e-3, "clip": None},
}

# the methods we compare across all experiments:
METHODS = {
    "vanilla":       {"activation": "relu"},
    "dropout":       {"activation": "relu", "dropout_p": 0.3},
    "aid":           {"activation": "aid",  "act_kwargs": {"p": 0.95}},
    "smooth_aid":    {"activation": "smooth_aid",
                      "act_kwargs": {"base_p": 0.95, "sharpness": 5.0}},
    "learnable_aid": {"activation": "learnable_aid",
                      "act_kwargs": {"init_p": 0.95}},
}

AID_P = {"cifar10": 0.95, "cifar100": 0.8, "tinyimagenet": 0.9}
DROPOUT_P = {"cifar10": 0.3, "cifar100": 0.1, "tinyimagenet": 0.1}
L2_LAMBDA = {"cifar10": 1e-2, "cifar100": 1e-5, "tinyimagenet": 1e-5}

def make_trainer(method_name, dataset, extra_cfg, quick, device, seed=13):
    """Build a Trainer for a given method and dataset."""
    ac = ARCH_CFG[dataset]
    return Trainer(TrainerConfig(
        arch=ac["arch"],
        num_classes=ac["num_classes"],
        dataset=dataset,
        data_root=DATA_ROOT,
        learning_rate=ac["lr"],
        grad_clip=ac["clip"],
        epochs_per_stage=5  if quick else 100,
        num_stages=2        if quick else 10,
        batch_size=32       if quick else 256,
        device=device,
        log_dir=RESULTS_DIR,
        run_name=f"{method_name}_{dataset}_seed{seed}",
        seed=seed,
        **extra_cfg,
    ))


# Experiment 1: trainability (permuted + random-label MNIST, Section 5.1.1)


def run_trainability(device, quick):
    from src.data import permuted_mnist_tasks, random_label_mnist_tasks

    print("\n=== Experiment 1: Trainability (MNIST) ===")

    num_tasks  = 5  if quick else 50
    batch_size = 512

    mnist_methods = {
        "vanilla":       {"activation": "relu"},
        "dropout":       {"activation": "relu", "dropout_p": 0.15},
        "aid":           {"activation": "aid",  "act_kwargs": {"p": 0.9}},
        "smooth_aid":    {"activation": "smooth_aid",
                          "act_kwargs": {"base_p": 0.9, "sharpness": 5.0}},
        "learnable_aid": {"activation": "learnable_aid",
                          "act_kwargs": {"init_p": 0.9}},
    }

    base_cfg = dict(
        arch="mlp", num_classes=10, dataset="cifar10",  # dataset unused for MLP
        data_root=DATA_ROOT, optimiser="adam", learning_rate=1e-3,
        epochs_per_stage=3 if quick else 100, batch_size=batch_size,
        device=device, log_dir=RESULTS_DIR,
    )

    for exp_name, task_fn, gen_kw in [
        ("PermutedMNIST",    permuted_mnist_tasks,    {"num_tasks": num_tasks}),
        ("RandomLabelMNIST", random_label_mnist_tasks, {"num_tasks": num_tasks}),
    ]:
        all_results = {}
        tasks = list(task_fn(root=DATA_ROOT, batch_size=batch_size, **gen_kw))

        for name, extra in mnist_methods.items():
            print(f"  {exp_name} | {name}")
            trainer = Trainer(TrainerConfig(
                run_name=f"{name}_{exp_name}", **base_cfg, **extra
            ))
            rows = trainer.run_trainability([t[0] for t in tasks], tasks[0][1])
            all_results[name] = rows

        plot_trainability(all_results, title=f"Trainability - {exp_name}",
                          save_path=fig(f"trainability_{exp_name}"))



# Experiment 2: continual full (Section 5.1.2)


def run_continual_full(dataset, device, quick):
    print(f"\n=== Experiment 2: Continual Full ({dataset}) ===")
    all_results = {}
    for name, extra in METHODS.items():
        print(f"\n--- {name} ---")
        trainer = make_trainer(name, dataset, extra, quick, device)
        all_results[name] = trainer.run_continual_full()

    total_ep = (2 if quick else 10) * (5 if quick else 100)
    plot_continual_accuracy(all_results, total_ep,
                            title=f"Continual Full - {dataset}",
                            save_path=fig(f"continual_full_{dataset}"))


# Experiment 3: continual limited (Section 5.1.2)

def run_continual_limited(dataset, device, quick):
    print(f"\n=== Experiment 3: Continual Limited ({dataset}) ===")
    all_results = {}
    for name, extra in METHODS.items():
        print(f"\n--- {name} ---")
        trainer = make_trainer(name, dataset, extra, quick, device)
        all_results[name] = trainer.run_continual_limited()

    total_ep = (2 if quick else 10) * (5 if quick else 100)
    plot_continual_accuracy(all_results, total_ep,
                            title=f"Continual Limited - {dataset}",
                            save_path=fig(f"continual_limited_{dataset}"))


# Experiment 4: class-incremental (Section 5.1.2)

def run_class_incremental(dataset, device, quick):
    print(f"\n=== Experiment 4: Class-Incremental ({dataset}) ===")

    # override stages - class-incremental uses 20 stages in the paper
    all_results = {}
    for name, extra in METHODS.items():
        print(f"\n--- {name} ---")
        ac = ARCH_CFG[dataset]
        trainer = Trainer(TrainerConfig(
            arch=ac["arch"], num_classes=ac["num_classes"],
            dataset=dataset, data_root=DATA_ROOT,
            learning_rate=ac["lr"], grad_clip=ac["clip"],
            epochs_per_stage=3  if quick else 100,
            num_stages=4        if quick else 20,
            batch_size=32       if quick else 256,
            device=device, log_dir=RESULTS_DIR,
            run_name=f"ci_{name}_{dataset}",
            **extra,
        ))
        all_results[name] = trainer.run_class_incremental()

    # use vanilla as the full-reset reference (best approximation without
    # running a dedicated full-reset baseline)
    full_reset_rows = all_results["vanilla"]
    plot_class_incremental_relative(
        all_results, full_reset_rows,
        title=f"Class-Incremental - {dataset}",
        save_path=fig(f"class_incremental_{dataset}"),
    )


# Experiment 5: standard supervised learning (Section 5.3, Table 1)

def run_supervised(dataset, device, quick, method):
    from src.data import _load_dataset
    from torch.utils.data import DataLoader
    print(f"\n=== Experiment 5: Standard Supervised ({dataset}) ===")
    ac           = ARCH_CFG[dataset]
    total_epochs = 10  if quick else 200
    batch_size   = 32  if quick else 256
    train_ds, test_ds = _load_dataset(dataset, DATA_ROOT, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=torch.cuda.is_available())
    aid_p = AID_P[dataset]
    dropout_p = DROPOUT_P[dataset]
    l2_lambda = L2_LAMBDA[dataset]
    supervised_methods = {
        "vanilla":  {"activation": "relu"},
        "dropout": {"activation": "relu", "dropout_p": dropout_p},
        "aid": {"activation": "aid",  "act_kwargs": {"p": aid_p}},
        "l2": {"activation": "relu", "weight_decay": l2_lambda},
        "smooth_aid": {"activation": "smooth_aid",
                          "act_kwargs": {"base_p": aid_p, "sharpness": 5.0}},
        "learnable_aid": {"activation": "learnable_aid",
                          "act_kwargs": {"init_p": aid_p}},
    }
    # filter to single method if requested (for slurm array parallelism)
    if method is not None:
        supervised_methods = {method: supervised_methods[method]}
    all_results = {}
    table_data  = {}
    ds_label    = {
        "cifar10": "CIFAR-10 (CNN)",
        "cifar100": "CIFAR-100 (ResNet-18)",
        "tinyimagenet": "TinyImageNet (VGG-16)",
    }[dataset]
    for name, extra in supervised_methods.items():
        print(f"  {name}")
        #if name == "learnable_aid" or name == "smooth_aid":
        #    continue
        trainer = Trainer(TrainerConfig(
            arch=ac["arch"], num_classes=ac["num_classes"],
            dataset=dataset, data_root=DATA_ROOT,
            learning_rate=ac["lr"], grad_clip=ac["clip"],
            optimiser="adam",
            lr_decay_epochs=[int(total_epochs * 0.5), int(total_epochs * 0.75)],
            device=device, log_dir=RESULTS_DIR,
            run_name=f"supervised_{name}_{dataset}",
            **extra,
        ))
        rows = trainer.run_standard_supervised(train_loader, test_loader, total_epochs)
        all_results[name] = rows
        table_data[name]  = {ds_label: rows[-1]["test_acc"]}
    plot_learning_curves(all_results, title=f"Supervised - {dataset}",
                         save_path=fig(f"supervised_{dataset}"))
    print("\n--- Table 1 (partial) ---")
    print_supervised_table(table_data, datasets=[ds_label])
# Experiment 6: warm-start generalisability (Section 3.2)

def run_warm_start(dataset, device, quick):
    from src.data import warm_start_loaders

    print(f"\n=== Experiment 6: Warm-Start ({dataset}) ===")

    batch_size       = 32  if quick else 256
    pretrain_epochs  = 5   if quick else 1000
    finetune_epochs  = 5   if quick else 100
    ac = ARCH_CFG[dataset]

    pretrain_loader, full_loader, test_loader = warm_start_loaders(
        dataset, root=DATA_ROOT, batch_size=batch_size
    )

    # only compare the three methods from Figure 2 (right)
    ws_methods = {
        "vanilla": {"activation": "relu"},
        "dropout": {"activation": "relu", "dropout_p": 0.3},
        "aid":     {"activation": "aid",  "act_kwargs": {"p": 0.8}},
    }

    gap_results = {}
    for name, extra in ws_methods.items():
        gap_results[name] = {}
        for variant in ("warm", "cold"):
            print(f"  {name} ({variant})")
            trainer = Trainer(TrainerConfig(
                arch=ac["arch"], num_classes=ac["num_classes"],
                dataset=dataset, data_root=DATA_ROOT,
                learning_rate=ac["lr"], grad_clip=ac["clip"],
                device=device, log_dir=RESULTS_DIR,
                run_name=f"ws_{name}_{variant}_{dataset}",
                **extra,
            ))
            if variant == "warm":
                rows = trainer.run_warm_start(
                    pretrain_loader, full_loader, test_loader,
                    pretrain_epochs, finetune_epochs,
                )
            else:
                # cold start -> train from scratch on full data
                rows = trainer.run_warm_start(
                    full_loader, full_loader, test_loader,
                    finetune_epochs, finetune_epochs,
                )
            gap_results[name][variant] = rows

    plot_generalisation_gap(gap_results, title=f"Generalisation Gap - {dataset}",
                            save_path=fig(f"warm_start_{dataset}"))


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", "-e", default="all",
                   choices=["all", "trainability", "continual_full",
                            "continual_limited", "class_incremental",
                            "supervised", "warm_start"])
    p.add_argument("--dataset", "-d", default="cifar100",
                   choices=["cifar10", "cifar100", "tinyimagenet"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true",
                   help="Minimal epochs/stages - use to check the code runs.")
    p.add_argument("--method", default=None,
                   choices=[None, "vanilla", "dropout", "aid", "l2", "smooth_aid", "learnable_aid"])
    return p.parse_args()


def main():
    args = parse_args()
    exp  = args.experiment

    print(f"Device: {args.device} | Dataset: {args.dataset} | Quick: {args.quick}")

    if exp in ("all", "trainability"):
        run_trainability(args.device, args.quick)
    if exp in ("all", "continual_full"):
        run_continual_full(args.dataset, args.device, args.quick)
    if exp in ("all", "continual_limited"):
        run_continual_limited(args.dataset, args.device, args.quick)
    if exp in ("all", "class_incremental"):
        run_class_incremental(args.dataset, args.device, args.quick)
    if exp in ("all", "supervised"):
        run_supervised(args.dataset, args.device, args.quick, args.method)
    if exp in ("all", "warm_start"):
        run_warm_start(args.dataset, args.device, args.quick)

    print(f"\nDone. Results -> {RESULTS_DIR}/  Figures -> {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
