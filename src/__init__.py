"""
aid_replication - reproducibility challenge for Park et al., ICML 2025.
"Activation by Interval-wise Dropout: A Simple Way to Prevent Neural
Networks from Plasticity Loss"
"""

from .aid import AID, SmoothAID, LearnableAID, make_activation
from .models import build_model
from .trainer import Trainer, TrainerConfig

__all__ = [
    "AID", "SmoothAID", "LearnableAID", "make_activation",
    "build_model", "Trainer", "TrainerConfig",
]
