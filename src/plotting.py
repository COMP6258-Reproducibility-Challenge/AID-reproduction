"""
plotting.py - plots all the figures

Figures we replicate:
  - Fig 2: trainability curves + warm-start generalisability
  - Fig 4: continual full/limited accuracy over epochs
  - Fig 5: class-incremental relative accuracy vs full reset
  - Fig 8: plasticity metrics (dormant ratio, sign entropy, eff. rank)
  - Fig 16: learning curves for standard supervised learning
  - Table 1: supervised learning accuracy summary

All functions take results dicts (method_name -> list of row dicts from Trainer) and save PNG files if a save_path is given
"""

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# Colours and labels to match the paper's style roughly

COLOURS = {
    "vanilla":      "#000000",
    "dropout":      "#e06c75",
    "aid":          "#56b4e9",
    "smooth_aid":   "#009e73",
    "learnable_aid":"#cc79a7",
    "full_reset":   "#d55e00",
    "l2":           "#f0e442",
    "l2_init":      "#0072b2",
    "crelu":        "#999999",
    "drelu":        "#e69f00",
    "fourier":      "#66c2a5",
    "cbp":          "#fc8d62",
    "snp":          "#8da0cb",
}

LABELS = {
    "vanilla":      "Vanilla",
    "dropout":      "Dropout",
    "aid":          "AID (Ours)",
    "smooth_aid":   "SmoothAID (Ext.)",
    "learnable_aid":"LearnableAID (Ext.)",
    "full_reset":   "Full Reset",
    "l2":           "L2",
    "crelu":        "CReLU",
    "drelu":        "DropReLU",
    "fourier":      "Fourier",
    "cbp":          "CBP",
    "snp":          "S&P",
}


def _c(name): return COLOURS.get(name.lower(), "#444444")
def _l(name): return LABELS.get(name.lower(), name)


plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def _save(fig, path):
    if path:
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")


# Fig 2 (left) - trainability accuracy per task

def plot_trainability(results, title="Trainability", save_path=None):
    """
    results: dict mapping method_name -> list of row dicts from Trainer
    Plots final test_acc per task for each method
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for method, rows in results.items():
        df = pd.DataFrame(rows)
        if "stage" in df.columns:
            xs = df.groupby("stage")["test_acc"].last().index.values + 1
            ys = df.groupby("stage")["test_acc"].last().values
        else:
            xs = np.arange(1, len(df) + 1)
            ys = df["test_acc"].values

        ax.plot(xs, ys, label=_l(method), color=_c(method), linewidth=1.5)

    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0.1, 1.05)
    ax.legend(loc="upper right", ncol=1, framealpha=0.7, fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# Fig 2 (right) - warm-start generalisability gap

def plot_generalisation_gap(results, title="Generalisation Gap", save_path=None):
    """
    results: dict mapping method_name -> {"warm": rows, "cold": rows}
    Plots cold_acc - warm_acc over fine-tuning epochs
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for method, variants in results.items():
        cold_df = pd.DataFrame(variants.get("cold", []))
        warm_df = pd.DataFrame(variants.get("warm", []))
        if cold_df.empty or warm_df.empty:
            continue

        # only look at fine-tuning phase
        if "phase" in cold_df.columns:
            cold_df = cold_df[cold_df["phase"] == "finetune"]
            warm_df = warm_df[warm_df["phase"] == "finetune"]

        n = min(len(cold_df), len(warm_df))
        gap    = (cold_df["test_acc"].values[:n] - warm_df["test_acc"].values[:n]) * 100
        epochs = cold_df["epoch"].values[:n] if "epoch" in cold_df else np.arange(1, n + 1)

        ax.plot(epochs, gap, label=_l(method), color=_c(method), linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Generalisation gap (%p)")
    ax.set_title(title)
    ax.legend(framealpha=0.7)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# Fig 4 - continual full/limited accuracy over global epochs

def plot_continual_accuracy(results, total_epochs, title="Continual Learning",
                            y_lim=(0.0, 0.8), save_path=None):
    """
    results: method_name -> list of row dicts with "global_epoch" and "test_acc".
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for method, rows in results.items():
        df = pd.DataFrame(rows)
        if "global_epoch" not in df.columns:
            df["global_epoch"] = np.arange(1, len(df) + 1)
        ax.plot(df["global_epoch"].values, df["test_acc"].values,
                label=_l(method), color=_c(method), linewidth=1.5)

    ax.set_xlim(0, total_epochs)
    ax.set_ylim(*y_lim)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.7, ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# Fig 5 - class-incremental relative accuracy vs full reset

def plot_class_incremental_relative(results, full_reset_rows,
                                    title="Class-Incremental vs Full Reset",
                                    save_path=None):
    """
    results: method_name -> list of row dicts with "stage" and "test_acc"
    Plots (method_acc - full_reset_acc) * 100 per stage
    """
    ref_acc = (pd.DataFrame(full_reset_rows)
               .groupby("stage")["test_acc"].last().values)

    fig, ax = plt.subplots(figsize=(5, 4))

    for method, rows in results.items():
        method_acc = (pd.DataFrame(rows)
                      .groupby("stage")["test_acc"].last().values)
        n        = min(len(method_acc), len(ref_acc))
        relative = (method_acc[:n] - ref_acc[:n]) * 100
        ax.plot(np.arange(1, n + 1), relative,
                label=_l(method), color=_c(method), linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Number of classes introduced")
    ax.set_ylabel("Relative accuracy (%p vs. full reset)")
    ax.set_title(title)
    ax.legend(framealpha=0.7, ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# Fig 8 - plasticity metrics over tasks

def plot_plasticity_metrics(results, title="Plasticity Metrics", save_path=None):
    """
    results: method_name -> list of row dicts with plasticity metric columns
    Plots dormant_ratio, sign_entropy, effective_rank side by side
    """
    metrics = {
        "dormant_ratio":  "Dormant Neuron Ratio",
        "sign_entropy":   "Avg. Sign Entropy",
        "effective_rank": "Effective Rank",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    for ax, (key, ylabel) in zip(axes, metrics.items()):
        for method, rows in results.items():
            df = pd.DataFrame(rows).dropna(subset=[key]) if key in pd.DataFrame(rows).columns else pd.DataFrame()
            if df.empty:
                continue
            if "stage" in df.columns:
                xs = df.groupby("stage")[key].mean().index.values + 1
                ys = df.groupby("stage")[key].mean().values
            else:
                xs = np.arange(1, len(df) + 1)
                ys = df[key].values
            ax.plot(xs, ys, label=_l(method), color=_c(method), linewidth=1.5)

        ax.set_xlabel("Task")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(framealpha=0.7, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# Fig 16 - learning curves (train + test) for standard supervised

def plot_learning_curves(results, title="Learning Curves", save_path=None):
    """
    Solid lines = test acc, dashed lines = train acc
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for method, rows in results.items():
        df     = pd.DataFrame(rows)
        epochs = df["epoch"].values if "epoch" in df else np.arange(1, len(df) + 1)
        c      = _c(method)

        if "test_acc" in df.columns:
            ax.plot(epochs, df["test_acc"].values, color=c, linestyle="-",
                    linewidth=1.5, label=f"{_l(method)} test")
        if "train_acc" in df.columns:
            ax.plot(epochs, df["train_acc"].values, color=c, linestyle="--",
                    linewidth=1.0, label=f"{_l(method)} train")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(framealpha=0.7, ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# Table 1 - supervised learning accuracy summary

def print_supervised_table(results, datasets=("CIFAR-10 (CNN)", "CIFAR-100 (ResNet-18)",
                                               "TinyImageNet (VGG-16)")):
    """
    results: method_name -> {dataset_name: acc or (mean, std)}
    Prints a formatted table and returns it as a string
    """
    col_w = 24
    header = "Method".ljust(16) + "".join(d.ljust(col_w) for d in datasets)
    sep    = "-" * len(header)
    lines  = [sep, header, sep]

    for method, ds_results in results.items():
        row = _l(method).ljust(16)
        for ds in datasets:
            val = ds_results.get(ds, float("nan"))
            if isinstance(val, (tuple, list)) and len(val) == 2:
                row += f"{val[0]:.3f} +/- {val[1]:.4f}".ljust(col_w)
            else:
                row += f"{float(val):.3f}".ljust(col_w)
        lines.append(row)

    lines.append(sep)
    out = "\n".join(lines)
    print(out)
    return out
