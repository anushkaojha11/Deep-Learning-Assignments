import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet-18.

    Implements the identity shortcut connection described in:
        He et al. (2016). Deep Residual Learning for Image Recognition.

    The forward pass computes:
        out = ReLU(F(x) + shortcut(x))

    where F(x) is two stacked 3x3 convolutions with batch normalisation,
    and shortcut(x) is either the identity (when dimensions match) or a
    1x1 conv to project x to the correct number of channels and spatial size.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : int
        Stride applied to the first conv and the shortcut projection.
        stride=2 halves the spatial dimensions (used between ResNet stages).
    """

    expansion = 1  # used by ResNet head to compute fc input size

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut: project only when spatial size or channel count changes
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # skip connection
        return self.relu(out)


class ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 inputs).

    The original ResNet-18 uses a 7x7 conv stem with stride 2 followed by
    a max pool, which reduces a 224x224 image to 56x56 before the residual
    stages. On 32x32 CIFAR images this would collapse spatial information too
    aggressively, so the stem is replaced with a single 3x3 conv (stride 1)
    without pooling — a standard adaptation used in the ResNet paper itself
    for CIFAR experiments.

    Architecture:
        Stem   : 3x3 Conv -> BN -> ReLU
        Layer 1: 2 x ResidualBlock(64  -> 64,  stride=1)
        Layer 2: 2 x ResidualBlock(64  -> 128, stride=2)
        Layer 3: 2 x ResidualBlock(128 -> 256, stride=2)
        Layer 4: 2 x ResidualBlock(256 -> 512, stride=2)
        Head   : AdaptiveAvgPool -> Flatten -> Linear(512, num_classes)

    Total blocks: 8 residual blocks = 16 weight layers + stem + fc = 18 layers

    Attributes
    ----------
    stem : Sequential
        Initial 3x3 conv suited for small CIFAR images
    layer1 - layer4 : Sequential
        Four residual stages with increasing channel width
    pool : AdaptiveAvgPool2d
        Global average pooling collapses spatial dims to 1x1
    fc : Linear
        Classification head
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64,  64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, n_blocks, stride):
        """Build a sequential stage of n_blocks residual blocks."""
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming initialisation for conv layers; standard init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)