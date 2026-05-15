# COMP6258 Reproducibility Challenge

Code for:

> **Activation by Interval-wise Dropout: A Simple Way to Prevent Neural
> Networks from Plasticity Loss**
> Park et al., ICML 2025

---

## What the paper does

Neural networks suffer from "plasticity loss" - over time they lose the ability to adapt to new tasks or data distributions. This is a big problem in continual learning and reinforcement learning where the data distribution shifts during training.

The paper argues that standard Dropout fails to fix this because it just creates multiple subnetworks, each of which independently suffers from plasticity loss. Instead they propose AID (Activation by Interval-wise Dropout), which applies different dropout probabilities to positive and negative preactivations:

```
Training:  mask ~ Bernoulli(p)
           output = mask * ReLU(x) + (1-mask) * (-ReLU(-x))

Eval:      output = p*x      for x >= 0
           output = (1-p)*x  for x < 0
```

This regularises the network toward linear behaviour (Theorem 4.1), and linear networks don't suffer from plasticity loss.

---

## Our extensions

We added two variants on top of the paper's AID:

**SmoothAID** - Replaces the hard threshold at x=0 with a sigmoid so the dropout probability varies continuously. Motivation: The hard boundary could cause suboptimal gradients near zero.

```
p_eff(x) = base_p * sigmoid(sharpness * x)
```

**LearnableAID** - Makes p a learnable parameter (stored as raw_p, projected to (0,1) via sigmoid). The network decides how much to
regularise toward a linear network rather than us picking p manually. Uses a straight-through estimator so gradients flow back to raw_p.

---

## File structure

```
aid_replication/
|
+-- src/
|   +-- aid.py           AID, SmoothAID, LearnableAID + factory
|   +-- models.py        MLP, CNN, ResNet-18, VGG-16 (swappable activations)
|   +-- data.py          dataset loaders + continual task generators
|   +-- trainer.py       training loop + CSV logging
|   +-- metrics.py       dormant ratio, sign entropy, effective rank
|   +-- plotting.py      figure reproduction (Figs 2, 4, 5, 8, 16, Table 1)
|   +-- __init__.py
|
+-- tests/
|   +-- test_aid.py      39 unit tests
|
+-- run_experiments.py   main CLI script
+-- download_tiny_imagenet.py  dataset downloader with working mirrors
+-- requirements.txt
+-- README.md
```

---

## Setup

```bash
# create a venv
python -m venv venv
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

GPU support (match your CUDA version):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Datasets

CIFAR-10 and CIFAR-100 download automatically via torchvision.

For TinyImageNet, download the zip from Kaggle and extract it into the
`data/` folder:

```
https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet
```

After extracting, your directory should look like:
```
data/
  tiny-imagenet-200/
    train/
    val/
    ...
```

---

## Running experiments

Quick test to check everything runs (minimal epochs):
```bash
python run_experiments.py --quick --experiment all
```

Individual experiments:
```bash
# trainability - permuted + random-label MNIST (Section 5.1.1)
python run_experiments.py --experiment trainability

# continual-full - data accumulates over 10 stages (Section 5.1.2)
python run_experiments.py --experiment continual_full --dataset cifar100

# continual-limited - only current chunk available at each stage
python run_experiments.py --experiment continual_limited --dataset cifar100

# class-incremental - new classes introduced in groups
python run_experiments.py --experiment class_incremental --dataset cifar100

# standard supervised learning + Table 1 (Section 5.3)
python run_experiments.py --experiment supervised --dataset cifar100

# warm-start generalisability (Section 3.2)
python run_experiments.py --experiment warm_start --dataset cifar100
```

GPU:
```bash
python run_experiments.py --device cuda --experiment continual_full --dataset cifar100
```

All experiments write CSVs to `results/` and PNGs to `figures/`.

---

## Tests

```bash
python -m pytest tests/ -v
```

What the tests cover:

| Class | Tests |
|---|---|
| TestAID | shape, determinism, scaling at p=1/p=0.5, sign consistency, He variance |
| TestSmoothAID | shape, asymptotic behaviour for large +/- inputs |
| TestLearnableAID | raw_p is learnable, gradient flows, shape |
| TestMakeActivation | factory returns correct type, unknown name raises |
| TestModels | output shapes for all 4 architectures, end-to-end gradient flow |
| TestMetrics | dormant ratio/entropy/rank in valid ranges |
| TestTheorem41 | numerical check that E[L_AID] >= L_p (Theorem 4.1) |

---

## Hyperparameters

The paper's top hyperparameters (from Tables 3-10 in the appendix):

| Experiment | p (AID) | LR | Optimiser |
|---|---|---|---|
| Permuted MNIST | 0.99 | 1e-3 (Adam) / 3e-2 (SGD) | both |
| Random-Label MNIST | 0.9 | 1e-3 (Adam) / 3e-2 (SGD) | both |
| Continual Full CIFAR-10 | 0.9 | 1e-3 | Adam |
| Continual Full CIFAR-100 | 0.7 | 1e-3 | Adam |
| Continual Full TinyImageNet | 0.7 | 1e-4 | Adam |
| Standard Supervised | 0.8-0.95 | 1e-3 | Adam |
| Reinforcement Learning | 0.99-0.999 | 6.25e-4 | Adam |

Key things to note:
- ResNet-18 needs gradient clipping at 0.5 (Appendix F.3)
- Optimiser is reset at the start of each continual stage
- AID/DropReLU models use avg-pool instead of max-pool (Section F.3)

---

## Citation

```bibtex
@inproceedings{park2025aid,
  title     = {Activation by Interval-wise Dropout: A Simple Way to Prevent
               Neural Networks from Plasticity Loss},
  author    = {Park, Sangyeon and Han, Isaac and Oh, Seungwon and Kim, Kyung-Joong},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  year      = {2025},
  publisher = {PMLR}
}
```
