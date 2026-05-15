"""
tests/test_aid.py - unit tests

Run with: python -m pytest tests/ -v

Test coverage:
  - AID forward pass (shapes, determinism, scaling, He variance property)
  - SmoothAID (asymptotic behaviour, shape)
  - LearnableAID (p is learnable, gradient flows through raw_p)
  - make_activation factory
  - All four model architectures (output shapes, gradient flow)
  - Plasticity metrics (output ranges)
  - Theorem 4.1 numerical sanity check
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.aid import AID, SmoothAID, LearnableAID, make_activation
from src.models import build_model, SimpleMLP
from src.metrics import dormant_neuron_ratio, effective_rank, average_sign_entropy


# AID

class TestAID:

    def test_output_shape_train(self):
        aid = AID(p=0.9)
        aid.train()
        x = torch.randn(16, 64)
        assert aid(x).shape == x.shape

    def test_output_shape_eval(self):
        aid = AID(p=0.9)
        aid.eval()
        x = torch.randn(16, 64)
        assert aid(x).shape == x.shape

    def test_eval_is_deterministic(self):
        aid = AID(p=0.9)
        aid.eval()
        x = torch.randn(32)
        assert torch.allclose(aid(x), aid(x))

    def test_train_is_stochastic(self):
        aid = AID(p=0.9)
        aid.train()
        x = torch.randn(100)
        torch.manual_seed(1); out1 = aid(x)
        torch.manual_seed(2); out2 = aid(x)
        assert not torch.allclose(out1, out2)

    def test_positive_scaled_by_p_at_eval(self):
        p = 0.8
        aid = AID(p=p); aid.eval()
        x   = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(aid(x), p * x, atol=1e-6)

    def test_negative_scaled_by_one_minus_p_at_eval(self):
        p = 0.8
        aid = AID(p=p); aid.eval()
        x   = torch.tensor([-1.0, -2.0, -3.0])
        assert torch.allclose(aid(x), (1 - p) * x, atol=1e-6)

    def test_p1_is_relu(self):
        aid = AID(p=1.0); aid.eval()
        x   = torch.linspace(-3.0, 3.0, 50)
        assert torch.allclose(aid(x), nn.ReLU()(x), atol=1e-6)

    def test_p_half_is_half_linear(self):
        aid = AID(p=0.5); aid.eval()
        x   = torch.linspace(-3.0, 3.0, 50)
        assert torch.allclose(aid(x), 0.5 * x, atol=1e-6)

    def test_output_sign_matches_input_sign_during_training(self):
        # positive inputs -> non-negative output, negative inputs -> non-positive
        aid = AID(p=0.9); aid.train()
        x   = torch.linspace(-5.0, 5.0, 100)
        torch.manual_seed(0)
        for _ in range(20):
            out = aid(x)
            assert (out[x > 0] >= 0).all()
            assert (out[x < 0] <= 0).all()

    def test_zero_input_gives_zero(self):
        aid = AID(p=0.8)
        x   = torch.zeros(10)
        aid.train();  assert torch.allclose(aid(x), x)
        aid.eval();   assert torch.allclose(aid(x), x)

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError): AID(p=1.5)
        with pytest.raises(ValueError): AID(p=-0.1)

    def test_he_variance_property(self):
        """
        Appendix D shows E[(AID(y))^2] = 0.5 * Var(y) for y ~ N(0,1),
        same as ReLU. !!! Check this holds numerically
        """
        torch.manual_seed(7)
        y   = torch.randn(100_000)
        aid = AID(p=0.7); aid.train()
        out = aid(y)
        expected = 0.5 * y.var().item()
        actual   = (out ** 2).mean().item()
        assert abs(actual - expected) < 0.02, \
            f"He variance: expected ~{expected:.4f}, got {actual:.4f}"


# SmoothAID

class TestSmoothAID:

    def test_output_shape(self):
        s = SmoothAID(base_p=0.8); s.train()
        x = torch.randn(32)
        assert s(x).shape == x.shape

    def test_eval_deterministic(self):
        s = SmoothAID(); s.eval()
        x = torch.randn(32)
        assert torch.allclose(s(x), s(x))

    def test_large_positive_approaches_base_p_scaling(self):
        # p_eff -> base_p for large positive x, so output ~= base_p * x
        s = SmoothAID(base_p=0.8, sharpness=10.0); s.eval()
        x   = torch.tensor([10.0])
        out = s(x)
        assert abs(out.item() - 8.0) < 0.1

    def test_large_negative_approaches_full_negative(self):
        # p_eff -> 0 for large negative x, so output ~= (1-0)*x = x
        s = SmoothAID(base_p=0.8, sharpness=10.0); s.eval()
        x   = torch.tensor([-10.0])
        out = s(x)
        assert abs(out.item() - (-10.0)) < 0.2

    def test_invalid_base_p(self):
        with pytest.raises(ValueError): SmoothAID(base_p=1.5)


# LearnableAID

class TestLearnableAID:

    def test_raw_p_is_parameter(self):
        lrn = LearnableAID(init_p=0.9)
        assert "raw_p" in dict(lrn.named_parameters())

    def test_initial_p_correct(self):
        lrn = LearnableAID(init_p=0.85)
        assert abs(lrn.p.item() - 0.85) < 0.01

    def test_gradient_flows_to_raw_p(self):
        lrn = LearnableAID(init_p=0.9); lrn.train()
        x   = torch.randn(32)
        lrn(x).sum().backward()
        assert lrn.raw_p.grad is not None
        assert not torch.isnan(lrn.raw_p.grad)

    def test_output_shape(self):
        lrn = LearnableAID(); lrn.eval()
        x   = torch.randn(16)
        assert lrn(x).shape == x.shape

    def test_invalid_init_p(self):
        with pytest.raises(ValueError): LearnableAID(init_p=0.0)
        with pytest.raises(ValueError): LearnableAID(init_p=1.0)


# make_activation

class TestMakeActivation:

    @pytest.mark.parametrize("name", ["relu", "aid", "smooth_aid", "learnable_aid"])
    def test_returns_module(self, name):
        act = make_activation(name)
        assert isinstance(act, nn.Module)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            make_activation("mystery_activation_9000")


# Models

class TestModels:

    @pytest.mark.parametrize("act", ["relu", "aid"])
    def test_mlp_output_shape(self, act):
        model = build_model("mlp", 10, act, {"p": 0.9})
        assert model(torch.randn(4, 784)).shape == (4, 10)

    @pytest.mark.parametrize("act", ["relu", "aid"])
    def test_cnn_output_shape(self, act):
        model = build_model("cnn", 10, act, {"p": 0.9})
        assert model(torch.randn(4, 3, 32, 32)).shape == (4, 10)

    @pytest.mark.parametrize("act", ["relu", "aid"])
    def test_resnet18_output_shape(self, act):
        model = build_model("resnet18", 100, act, {"p": 0.7}); model.eval()
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 100)

    def test_vgg16_output_shape(self):
        model = build_model("vgg16", 200, "aid", {"p": 0.7}); model.eval()
        assert model(torch.randn(2, 3, 64, 64)).shape == (2, 200)

    def test_gradients_flow_through_aid_mlp(self):
        model = build_model("mlp", 10, "aid", {"p": 0.9}); model.train()
        loss  = nn.CrossEntropyLoss()(model(torch.randn(8, 784)),
                                      torch.randint(0, 10, (8,)))
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"

    def test_dropout_baseline_works(self):
        model = build_model("mlp", 10, "relu", dropout_p=0.3)
        assert model(torch.randn(4, 784)).shape == (4, 10)

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError):
            build_model("transformer_xl", 10)


# Plasticity metrics

class TestMetrics:

    def _make_setup(self):
        model  = SimpleMLP(784, 10, hidden_dim=64, num_layers=2); model.eval()
        x, y   = torch.randn(32, 784), torch.randint(0, 10, (32,))
        loader = DataLoader(TensorDataset(x, y), batch_size=16)
        return model, loader

    def test_dormant_ratio_in_range(self):
        m, l = self._make_setup()
        r    = dormant_neuron_ratio(m, l, max_batches=2)
        assert 0.0 <= r <= 1.0

    def test_effective_rank_positive(self):
        m, l = self._make_setup()
        r    = effective_rank(m, l, max_batches=2)
        assert r >= 1.0

    def test_sign_entropy_in_range(self):
        m, l = self._make_setup()
        e    = average_sign_entropy(m, l, max_batches=2)
        assert 0.0 <= e <= 1.0

    def test_dead_model_has_higher_dormant_ratio(self):
        # a model with all-zero weights should have more dormant neurons
        m_normal, loader = self._make_setup()
        m_dead,   _      = self._make_setup()
        with torch.no_grad():
            for p in m_dead.parameters():
                p.zero_()

        r_normal = dormant_neuron_ratio(m_normal, loader, max_batches=2)
        r_dead   = dormant_neuron_ratio(m_dead,   loader, max_batches=2)
        assert r_dead >= r_normal


# Theorem 4.1 numerical check

class TestTheorem41:

    def test_aid_loss_geq_leaky_relu_loss(self):
        """
        Theorem 4.1: E[L_AID] >= L_p + regularisation_term (>= 0).
        Check the inequality holds numerically with a small 2-layer network.
        """
        torch.manual_seed(42)
        n, d = 64, 32
        p    = 0.7
        W1   = torch.randn(d, d) * 0.1
        W2   = torch.randn(d, d) * 0.1
        x    = torch.randn(n, d)
        y    = torch.randn(n, d)

        # estimate E[L_AID] with Monte Carlo
        aid_losses = []
        for _ in range(500):
            pre  = W1 @ x.T  # (d, n)
            mask = (torch.rand_like(pre) < p)
            post = torch.where(mask, torch.relu(pre), -torch.relu(-pre))
            pred = (W2 @ post).T
            aid_losses.append(((pred - y) ** 2).mean().item())
        e_aid = sum(aid_losses) / len(aid_losses)

        # L_p with deterministic modified leaky ReLU
        pre    = W1 @ x.T
        post_p = torch.where(pre >= 0, p * pre, (1 - p) * pre)
        pred_p = (W2 @ post_p).T
        l_p    = ((pred_p - y) ** 2).mean().item()

        # allow small numerical slack
        assert e_aid >= l_p - 0.05, \
            f"Theorem 4.1 violated: E[L_AID]={e_aid:.4f} < L_p={l_p:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# Sam notes:
# fix tests for learnable aid - DONE
# fix tests for normal aid - DONE
# add better comments throughout all files - DONE