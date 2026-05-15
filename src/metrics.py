"""
metrics.py - all metrics from paper

Three metrics for measuring how much plasticity a network has lost:

  1. Dormant Neuron Ratio  - fraction of neurons with near-zero activation

  2. Average Unit Sign Entropy - how evenly a neuron fires for +/- inputs
     Near 0 -> diverse/healthy. Near 1 -> saturated in one direction

  3. Effective Rank (srank_delta) - rank of the penultimate-layer features
     Lower rank -> less representational capacity -> more plasticity loss

All functions use forward hooks to collect activations without modifying the model. 
Max_batches caps how many batches we run to keep things fast.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Helper: collects post-activation outputs from all hidden Linear/Conv layers

class _HookCollector:
    """Attach forward hooks to all Linear and Conv2d layers in a model"""

    def __init__(self, model, exclude_last=True):
        self.outputs = []
        self._handles = []

        # find all linear/conv layers
        targets = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
        if exclude_last and len(targets) > 1:
            targets = targets[:-1]  # skip the classification head

        for m in targets:
            self._handles.append(m.register_forward_hook(self._hook))

    def _hook(self, module, inp, out):
        self.outputs.append(out.detach().cpu())

    def clear(self):
        self.outputs.clear()

    def remove(self):
        for h in self._handles:
            h.remove()


# 1. Dormant Neuron Ratio

@torch.no_grad()
def dormant_neuron_ratio(model, loader, tau=0.0, device="cpu", max_batches=50):
    """
    Fraction of neurons with normalised activation score <= tau
    """
    model.eval()
    collector = _HookCollector(model)

    # run forward passes and collect all activations
    all_acts = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        collector.clear()
        model(x.to(device))
        all_acts.append([a.clone() for a in collector.outputs])

    collector.remove()

    if not all_acts:
        return float("nan")

    n_layers = len(all_acts[0])
    dormant = 0
    total   = 0

    for l in range(n_layers):
        # concatenate batches -> (N, C, ...)
        acts = torch.cat([snap[l] for snap in all_acts], dim=0)
        # flatten spatial dims -> (N, C)
        acts = acts.view(acts.size(0), acts.size(1), -1).mean(dim=-1)

        mean_abs = acts.abs().mean(dim=0) 
        layer_mean = mean_abs.mean()

        if layer_mean < 1e-8:
            # entire layer is dead
            dormant += mean_abs.numel()
        else:
            scores = mean_abs / layer_mean
            dormant += int((scores <= tau).sum().item())
        total += mean_abs.numel()

    return dormant / total if total > 0 else float("nan")


# 2. Average Unit Sign Entropy

@torch.no_grad()
def average_sign_entropy(model, loader, device="cpu", max_batches=50):
    """
    Mean |E[sgn(h(x))]| across all hidden units
    A unit with entropy near 1 is saturated (always fires same sign)
    A unit with entropy near 0 fires roughly equally for +/- inputs (healthy)
    We hook the pre-activation (output of Linear/Conv before activation)
    """
    model.eval()

    # we want pre-activations so we hook just before the activation
    targets = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    if len(targets) > 1:
        targets = targets[:-1]

    # separate per-layer buffers
    buffers = [[] for _ in targets]
    handles = []

    for i, m in enumerate(targets):
        def make_hook(idx):
            def hook(mod, inp, out):
                buffers[idx].append(out.detach().cpu())
            return hook
        handles.append(m.register_forward_hook(make_hook(i)))

    snapshots = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        for b in buffers:
            b.clear()
        model(x.to(device))
        snapshots.append([torch.cat(b, dim=0) for b in buffers])

    for h in handles:
        h.remove()

    if not snapshots:
        return float("nan")

    total_entropy = 0.0
    total_units   = 0

    for l in range(len(targets)):
        acts = torch.cat([s[l] for s in snapshots], dim=0)
        acts = acts.view(acts.size(0), acts.size(1), -1).mean(dim=-1)  
        mean_sign = acts.sign().float().mean(dim=0)  
        total_entropy += mean_sign.abs().sum().item()
        total_units   += mean_sign.numel()

    return total_entropy / total_units if total_units > 0 else float("nan")


# 3. Effective Rank (srank_delta)

@torch.no_grad()
def effective_rank(model, loader, delta=0.01, device="cpu", max_batches=50):
    model.eval()

    # hook the second-to-last linear layer
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linear_layers) < 2:
        return float("nan")

    buf = []
    handle = linear_layers[-2].register_forward_hook(
        lambda mod, inp, out: buf.append(out.detach().cpu())
    )

    feats = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        buf.clear()
        model(x.to(device))
        feats.append(buf[0].clone())

    handle.remove()

    if not feats:
        return float("nan")

    Phi = torch.cat(feats, dim=0)

    try:
        _, S, _ = torch.linalg.svd(Phi, full_matrices=False)
    except Exception:
        return float("nan")

    S = S.abs()
    total = S.sum().item()
    if total < 1e-8:
        return 0.0

    cumsum = S.cumsum(0) / total
    above_thresh = (cumsum >= (1.0 - delta)).nonzero(as_tuple=False)
    if above_thresh.numel() == 0:
        return float(S.numel())
    return float(above_thresh[0].item() + 1)



def compute_all_metrics(model, loader, device="cpu", max_batches=50):
    """Run all three metrics and return them as a dict."""
    return {
        "dormant_ratio":  dormant_neuron_ratio(model, loader, device=device, max_batches=max_batches),
        "sign_entropy":   average_sign_entropy(model, loader, device=device, max_batches=max_batches),
        "effective_rank": effective_rank(model, loader, device=device, max_batches=max_batches),
    }
