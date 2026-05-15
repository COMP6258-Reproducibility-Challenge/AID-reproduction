"""
aid.py - AID activation function and our two extensions

The paper defines AID as applying ReLU with prob p, and negative-ReLU with prob 1-p

Extensions we added for the project:
  - SmoothAID: replace the hard threshold at 0 with a sigmoid so the
    dropout probability varies continuously (no discontinuity at x=0)
  - LearnableAID: make p a learnable parameter so the network can
    self-regulate how much it regularises toward a linear network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AID(nn.Module):
    """
    AID from Algorithm 2 in the paper

    During training:
      - with prob p  -> apply ReLU  (keep positive, zero negative)
      - with prob 1-p -> apply neg-ReLU (zero positive, keep negative)

    During eval:
      - apply r_p(x): p*x for x>=0, (1-p)*x for x<0
      This is the deterministic expected output from training

    p=1.0 -> behaves like ReLU
    p=0.5 -> behaves like a 0.5x linear layer (strongest regularisation)
    """

    def __init__(self, p=0.9):
        super().__init__()
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, self.p)).bool()
            return torch.where(mask, F.relu(x), -F.relu(-x))
        else:
            return torch.where(x >= 0, self.p * x, (1.0 - self.p) * x)

    def extra_repr(self):
        return f"p={self.p}"


class SmoothAID(nn.Module):
    """
    Extension: Instead of a hard threshold at 0, the dropout probability is a sigmoid function of x so we get a smooth transition:

      p_eff(x) = base_p * sigmoid(sharpness*x)

    For large positive x -> p_eff ~= base_p (like the positive side of AID)
    For large negative x -> p_eff ~= 0 (like the negative side of AID)
    Near x=0 -> smooth interpolation between the two

    Motivation: The hard threshold in standard AID could cause bad gradients near x=0, so a continuous boundary might help stability
    """

    def __init__(self, base_p=0.9, sharpness=5.0):
        super().__init__()
        if not 0.0 < base_p <= 1.0:
            raise ValueError(f"base_p must be in (0, 1], got {base_p}")
        self.base_p = base_p
        self.sharpness = sharpness

    def forward(self, x):
        p_eff = self.base_p * torch.sigmoid(self.sharpness * x)
        if self.training:
            mask = torch.bernoulli(p_eff).bool()
            return torch.where(mask, F.relu(x), -F.relu(-x))
        else:
            return p_eff * F.relu(x) + (1.0 - p_eff) * (-F.relu(-x))

    def extra_repr(self):
        return f"base_p={self.base_p}, sharpness={self.sharpness}"


class LearnableAID(nn.Module):
    """
    Extension: Make p a learnable parameter so the network decides how much to regularise toward a linear network without manual tuning

    We store an unconstrained raw_p and project via sigmoid -> p in (0,1)
    p near 1.0 -> more like ReLU (less regularisation)
    p near 0.5 -> stronger linearisation (see Theorem 4.1 in the paper)

    The stochastic path is non-differentiable w.r.t. p, so we use a straight-through estimator. 
    We add a zero-valued differentiable term (det - det.detach()) so autograd can still reach raw_p
    """

    def __init__(self, init_p=0.9):
        super().__init__()
        if not 0.0 < init_p < 1.0:
            raise ValueError(f"init_p must be in (0, 1), got {init_p}")
        # inverse sigmoid so sigmoid(raw_p) = init_p at initialisation
        init_raw = torch.log(torch.tensor(init_p / (1.0 - init_p)))
        self.raw_p = nn.Parameter(init_raw)

    @property
    def p(self):
        return torch.sigmoid(self.raw_p)

    def forward(self, x):
        p = self.p
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, p.item())).bool()
            stoch = torch.where(mask, F.relu(x), -F.relu(-x))
            # straight-through estimator for gradient flow to raw_p
            det = torch.where(x >= 0, p * x, (1.0 - p) * x)
            return stoch + (det - det.detach())
        else:
            return torch.where(x >= 0, p * x, (1.0 - p) * x)

    def extra_repr(self):
        return f"p={self.p.item():.4f} (learnable)"


def make_activation(name, **kwargs):
    """swap activations by name in config dicts"""
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=False)
    elif name == "aid":
        return AID(**kwargs)
    elif name == "smooth_aid":
        return SmoothAID(**kwargs)
    elif name == "learnable_aid":
        return LearnableAID(**kwargs)
    else:
        raise ValueError(
            f"Unknown activation '{name}'. Options: relu, aid, smooth_aid, learnable_aid"
        )
