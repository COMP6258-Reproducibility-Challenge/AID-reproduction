"""
data.py - dataset loading and continual learning task generators

Covers all experiments in the paper:
  - MNIST: permuted + random-label (trainability experiments)
  - CIFAR-10/100 + TinyImageNet: continual full/limited/class-incremental

Each task generator is a regular Python generator that yields (train_loader, test_loader) pairs, one per task/stage

TinyImageNet note: torchvision's ImageFolder needs the val set to be restructured into per-class subdirectories.
Call restructure_tiny_val() once before using the dataset (see download_tiny_imagenet.py)
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder


# Normalisation stats

_STATS = {
    "mnist":   ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100":((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tiny":    ((0.480,  0.448,  0.398),  (0.277,  0.269,  0.282)),
}


# Helpers

def _make_loader(dataset, batch_size, shuffle, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _load_mnist(root="data"):
    mean, std = _STATS["mnist"]
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train = MNIST(root, train=True,  download=True, transform=tf)
    test  = MNIST(root, train=False, download=True, transform=tf)
    return train, test


def _load_cifar10(root="data", augment=True):
    mean, std = _STATS["cifar10"]
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]) if augment else T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return (
        CIFAR10(root, train=True,  download=True, transform=train_tf),
        CIFAR10(root, train=False, download=True, transform=test_tf),
    )


def _load_cifar100(root="data", augment=True):
    mean, std = _STATS["cifar100"]
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]) if augment else T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return (
        CIFAR100(root, train=True,  download=True, transform=train_tf),
        CIFAR100(root, train=False, download=True, transform=test_tf),
    )


def _load_tiny_imagenet(root="data/tiny-imagenet-200", augment=True):
    """
    Load TinyImageNet from root
    The val set needs its images moved into per-class subdirectories for ImageFolder to work. We do this automatically on first load if needed
    """
    _maybe_restructure_val(root)

    mean, std = _STATS["tiny"]
    train_tf = T.Compose([
        T.RandomCrop(64, padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]) if augment else T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return (
        ImageFolder(os.path.join(root, "train"), transform=train_tf),
        ImageFolder(os.path.join(root, "val"),   transform=test_tf),
    )


def _maybe_restructure_val(root):
    """
    Move val images into per-class subdirectories if not already done
    Checks for val_annotations.txt to decide whether restructuring is needed.
    """
    import shutil
    val_dir  = Path(root) / "val"
    img_dir  = val_dir / "images"
    ann_file = val_dir / "val_annotations.txt"

    if not ann_file.exists():
        # already restructured - nothing to do
        return

    print("First-time setup: restructuring the val/ folder")
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_name, class_id = parts[0], parts[1]
            class_dir = val_dir / class_id
            class_dir.mkdir(exist_ok=True)
            src = img_dir / img_name
            dst = class_dir / img_name
            if src.exists():
                shutil.move(str(src), str(dst))

    if img_dir.exists() and not any(img_dir.iterdir()):
        img_dir.rmdir()
    print("Done. Val set is ready.")


def _load_dataset(name, root="data", augment=True):
    """Load a dataset by name and return (train_ds, test_ds)"""
    name = name.lower()
    if name == "cifar10":
        return _load_cifar10(root, augment)
    elif name == "cifar100":
        return _load_cifar100(root, augment)
    elif name in ("tinyimagenet", "tiny_imagenet", "tiny"):
        return _load_tiny_imagenet(os.path.join(root, "tiny-imagenet-200"), augment)
    else:
        raise ValueError(f"Unknown dataset '{name}'")


# Permuted MNIST

class PermutedMNIST(Dataset):
    """MNIST with a fixed pixel permutation applied to every image"""

    def __init__(self, base_dataset, permutation):
        self.base = base_dataset
        self.perm = permutation

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x.view(-1)[self.perm].view(1, 28, 28), y


def permuted_mnist_tasks(num_tasks=800, batch_size=512, root="data", seed=13):
    """
    Yield (train_loader, test_loader) for each permuted-MNIST task
    Each task applies a fresh random pixel permutation
    """
    rng = torch.Generator().manual_seed(seed)
    train_base, test_base = _load_mnist(root)

    for _ in range(num_tasks):
        perm = torch.randperm(784, generator=rng)
        yield (
            _make_loader(PermutedMNIST(train_base, perm), batch_size, shuffle=True),
            _make_loader(PermutedMNIST(test_base,  perm), batch_size, shuffle=False),
        )


# Random-Label MNIST

class RandomLabelMNIST(Dataset):
    """MNIST with randomly shuffled class labels"""

    def __init__(self, base_dataset, seed=0):
        self.base = base_dataset
        self.rng  = random.Random(seed)
        self._shuffle()

    def _shuffle(self):
        classes = list(range(10))
        shuffled = classes[:]
        self.rng.shuffle(shuffled)
        self.label_map = dict(zip(classes, shuffled))

    def reshuffle(self):
        """Call between tasks to get a new label assignment"""
        self._shuffle()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, self.label_map[int(y)]


def random_label_mnist_tasks(num_tasks=200, samples_per_task=1600,
                              batch_size=64, root="data", seed=13):
    """
    Yield (train_loader, test_loader) for each random-label MNIST task
    Labels are re-shuffled at the start of each task
    """
    train_base, test_base = _load_mnist(root)
    train_ds = RandomLabelMNIST(train_base, seed=seed)
    test_ds  = RandomLabelMNIST(test_base,  seed=seed + 1)

    for _ in range(num_tasks):
        train_ds.reshuffle()
        test_ds.reshuffle()

        idxs = list(range(len(train_ds)))
        random.shuffle(idxs)
        sub = Subset(train_ds, idxs[:samples_per_task])

        yield (
            _make_loader(sub,     batch_size, shuffle=True),
            _make_loader(test_ds, batch_size, shuffle=False),
        )


# Continual-full

def continual_full_tasks(dataset_name, num_stages=10, batch_size=256,
                         root="data", augment=True, seed=13):
    """
    Continual-full setting (Appendix F.3): training data accumulates over stages. 
    At stage k, the model sees all chunks 1 to k
    Yields (train_loader, test_loader) per stage
    """
    train_ds, test_ds = _load_dataset(dataset_name, root, augment)
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(len(train_ds)).tolist()
    chunk = len(idxs) // num_stages
    test_loader = _make_loader(test_ds, batch_size, shuffle=False)

    for stage in range(num_stages):
        seen = idxs[: (stage + 1) * chunk]
        yield _make_loader(Subset(train_ds, seen), batch_size, shuffle=True), test_loader


# Continual-limited

def continual_limited_tasks(dataset_name, num_stages=10, batch_size=256,
                             root="data", augment=True, seed=13):
    """
    Continual-limited setting (Appendix F.3): only the current chunk is available at each stage, no replay of past data.
    Yields (train_loader, test_loader) per stage
    """
    train_ds, test_ds = _load_dataset(dataset_name, root, augment)
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(len(train_ds)).tolist()
    chunk = len(idxs) // num_stages
    test_loader = _make_loader(test_ds, batch_size, shuffle=False)

    for stage in range(num_stages):
        current = idxs[stage * chunk : (stage + 1) * chunk]
        yield _make_loader(Subset(train_ds, current), batch_size, shuffle=True), test_loader


# Class-incremental

def class_incremental_tasks(dataset_name, num_stages=20, batch_size=256,
                             root="data", augment=True):
    """
    Class-incremental setting (Appendix F.3): classes are introduced in groups of (num_classes / num_stages). At stage k all data for classes
    0to k * group_size is available. Test is on seen classes only
    Yields (train_loader, test_loader) per stage
    """
    train_ds, test_ds = _load_dataset(dataset_name, root, augment)

    train_labels = np.array(train_ds.targets)
    test_labels  = np.array(test_ds.targets)
    num_classes  = len(np.unique(train_labels))
    group_size   = num_classes // num_stages

    for stage in range(num_stages):
        seen = list(range((stage + 1) * group_size))
        t_idx = np.where(np.isin(train_labels, seen))[0].tolist()
        v_idx = np.where(np.isin(test_labels,  seen))[0].tolist()
        yield (
            _make_loader(Subset(train_ds, t_idx), batch_size, shuffle=True),
            _make_loader(Subset(test_ds,  v_idx), batch_size, shuffle=False),
        )


# Warm-start

def warm_start_loaders(dataset_name, pretrain_fraction=0.1, batch_size=256,
                       root="data", augment=True, seed=13):
    """
    Returns (preetrain_loader, full_train_loader, test_loader) for the warm-start generalisability experiment from Section 3.2
    pretrain_loader covers pretrain_fraction of the training data.
    """
    train_ds, test_ds = _load_dataset(dataset_name, root, augment)
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(len(train_ds)).tolist()
    n_pre = int(len(idxs) * pretrain_fraction)

    pretrain_loader   = _make_loader(Subset(train_ds, idxs[:n_pre]), batch_size, True)
    full_train_loader = _make_loader(train_ds, batch_size, True)
    test_loader       = _make_loader(test_ds,  batch_size, False)
    return pretrain_loader, full_train_loader, test_loader
