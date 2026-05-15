"""
models.py - all the models

Each model takes an activation name + kwargs so we can swap ReLU for AID (or our extensions) without changing the code

Models:
  SimpleMLP - 3-hidden-layer FC network for MNIST trainability experiments
  SimpleCNN - small conv net for CIFAR-10 (matches Appendix F.3)
  ResNet18 - standard ResNet-18 with stem removed for small images
  VGG16 - VGG-16 with BN for TinyImageNet

Note: for AID-based models we swap max-pool -> avg-pool as the paper recommended (Section F.3), 
since large positive values no longer necessarily indicate the most important features
"""

import torch
import torch.nn as nn
from .aid import make_activation


def _act(name, **kwargs):
    return make_activation(name, **kwargs)


# SimpleMLP

class SimpleMLP(nn.Module):
    """
    3-hidden-layer MLP with 2000 units per layer, matching Appendix F.2
    Used for permuted/random-label MNIST trainability experiments
    dropout_p is for the standard Dropout baseline only
    """

    def __init__(self, input_dim=784, num_classes=10, hidden_dim=2000,
                 num_layers=3, activation="relu", act_kwargs=None, dropout_p=None):
        super().__init__()
        act_kwargs = act_kwargs or {}

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_act(activation, **act_kwargs))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# SimpleCNN

class SimpleCNN(nn.Module):
    """
    Two 5x5 conv layers (16 channels each) + two FC layers (100 hidden units)
    Matches the CNN described in Appendix F.3 for CIFAR-10
    """
    def __init__(self, num_classes=10, activation="relu", act_kwargs=None, dropout_p=None):
        super().__init__()
        act_kwargs = act_kwargs or {}
        # use avg-pool for AID variants as recommended in the paper
        pool = nn.AvgPool2d(2, 2) if activation != "relu" else nn.MaxPool2d(2, 2)

        feature_layers = []
        feature_layers.append(nn.Conv2d(3, 16, kernel_size=5, padding=2))
        feature_layers.append(_act(activation, **act_kwargs))
        if dropout_p is not None:
            feature_layers.append(nn.Dropout(dropout_p))
        feature_layers.append(pool)

        feature_layers.append(nn.Conv2d(16, 16, kernel_size=5, padding=2))
        feature_layers.append(_act(activation, **act_kwargs))
        if dropout_p is not None:
            feature_layers.append(nn.Dropout(dropout_p))
        feature_layers.append(pool)

        self.features = nn.Sequential(*feature_layers)

        # after two 2x2 pools on 32x32 input -> 8x8 -> 16*64 = 1024 features
        classifier_layers = [
            nn.Linear(16 * 8 * 8, 100),
            _act(activation, **act_kwargs),
        ]
        if dropout_p is not None:
            classifier_layers.append(nn.Dropout(dropout_p))
        classifier_layers.append(nn.Linear(100, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ResNet-18

class BasicBlock(nn.Module):
    """Standard ResNet basic block (two 3x3 convs with a skip connection)."""
    def __init__(self, in_ch, out_ch, stride=1, activation="relu", act_kwargs=None, dropout_p=None):
        super().__init__()
        act_kwargs = act_kwargs or {}
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act1  = _act(activation, **act_kwargs)
        self.drop1 = nn.Dropout(dropout_p) if dropout_p is not None else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act2  = _act(activation, **act_kwargs)
        self.drop2 = nn.Dropout(dropout_p) if dropout_p is not None else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = self.bn2(self.conv2(out))
        out = self.act2(out + self.shortcut(x))
        out = self.drop2(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 with the large-kernel stem removed for small images (32x32)
    Matches the setup in Appendix F.3. Gradient clipping (0.5) should be applied in the trainer.
    """
    def __init__(self, num_classes=100, activation="relu", act_kwargs=None, dropout_p=None):
        super().__init__()
        act_kwargs = act_kwargs or {}
        self._act_name  = activation
        self._act_kw    = act_kwargs
        self._dropout_p = class AID(nn.Module):
    def __init__(self, p=0.9):
        super().__init__()
        self.p = float(p)

        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")

    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, self.p)).bool()
            return torch.where(mask, torch.relu(x), -torch.relu(-x))
        else:
            return torch.where(x >= 0, self.p * x, (1.0 - self.p) * x)
dropout_p

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            _act(activation, **act_kwargs),
            nn.Dropout(dropout_p) if dropout_p is not None else nn.Identity(),
        )
        self.layer1 = self._make_layer(64,  64,  2, stride=1)
        self.layer2 = self._make_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else nn.Identity()
        self.fc      = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        blocks = [BasicBlock(in_ch, out_ch, stride, self._act_name, self._act_kw, self._dropout_p)]
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock(out_ch, out_ch, 1, self._act_name, self._act_kw, self._dropout_p))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# VGG-16 with BN

# VGG-16 config- numbers are channel counts and 'M' is a pool layer
_VGG16_CFG = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M",
]


class VGG16(nn.Module):
    """
    VGG-16 with batch normalisation for TinyImageNet (64x64 input)
    Classifier head uses 4096 hidden units. Max-pool swapped for avg-pool when
    using AID variants. Dropout (if enabled) applied after every activation
    to match the placement of AID and provide a fair comparison.
    """
    def __init__(self, num_classes=200, activation="relu", act_kwargs=None, dropout_p=None):
        super().__init__()
        act_kwargs = act_kwargs or {}
        use_avg_pool = (activation != "relu")

        layers = []
        in_ch = 3
        for v in _VGG16_CFG:
            if v == "M":
                layers.append(nn.AvgPool2d(2, 2) if use_avg_pool else nn.MaxPool2d(2, 2))
            else:
                layers.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(v))
                layers.append(_act(activation, **act_kwargs))
                if dropout_p is not None:
                    layers.append(nn.Dropout(dropout_p))
                in_ch = v
        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        classifier_layers = [
            nn.Linear(512, 4096),
            _act(activation, **act_kwargs),
        ]
        if dropout_p is not None:
            classifier_layers.append(nn.Dropout(dropout_p))
        classifier_layers.append(nn.Linear(4096, 4096))
        classifier_layers.append(_act(activation, **act_kwargs))
        if dropout_p is not None:
            classifier_layers.append(nn.Dropout(dropout_p))
        classifier_layers.append(nn.Linear(4096, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)


def build_model(arch, num_classes, activation="relu", act_kwargs=None, dropout_p=None):
    """Build a model by name. arch can be: mlp, cnn, resnet18, vgg16."""
    kw = dict(num_classes=num_classes, activation=activation,
              act_kwargs=act_kwargs, dropout_p=dropout_p)
    arch = arch.lower()
    if arch == "mlp":
        return SimpleMLP(**kw)
    elif arch == "cnn":
        return SimpleCNN(**kw)
    elif arch == "resnet18":
        return ResNet18(**kw)
    elif arch == "vgg16":
        return VGG16(**kw)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Options: mlp, cnn, resnet18, vgg16")
