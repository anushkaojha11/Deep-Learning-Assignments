import torch
import torch.nn as nn


class Inception(nn.Module):
    """
    Inception block for GoogLeNet.

    Four parallel branches are concatenated along the channel dimension:
        Branch 1 — 1x1 conv
        Branch 2 — 1x1 conv reduction -> 3x3 conv
        Branch 3 — 1x1 conv reduction -> two 3x3 convs (approximates 5x5)
        Branch 4 — 3x3 max pool -> 1x1 conv reduction

    Parameters
    ----------
    in_planes : int
        Number of input channels
    n1x1 : int
        Output channels of branch 1
    n3x3red : int
        Reduction channels before the 3x3 conv in branch 2
    n3x3 : int
        Output channels of the 3x3 conv in branch 2
    n5x5red : int
        Reduction channels before the two 3x3 convs in branch 3
    n5x5 : int
        Output channels of branch 3
    pool_planes : int
        Output channels of the 1x1 conv after pooling in branch 4
    """

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super().__init__()

        # Branch 1: 1x1 conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True),
        )

        # Branch 2: 1x1 reduction -> 3x3 conv
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True),
        )

        # Branch 3: 1x1 reduction -> 3x3 -> 3x3 (approximates original 5x5)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
        )

        # Branch 4: 3x3 max pool -> 1x1 reduction
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], dim=1)


class AuxClassifier(nn.Module):
    """
    Auxiliary classifier attached to intermediate inception blocks.

    As described in the GoogLeNet paper, two of these are added during
    training to combat vanishing gradients in the middle of the network.
    Each applies average pooling, a 1x1 conv, two FC layers, and dropout.

    The auxiliary loss is weighted at 0.3 and added to the main loss:
        L_total = L_main + 0.3 * L_aux1 + 0.3 * L_aux2

    Parameters
    ----------
    in_channels : int
        Number of input channels from the tapped inception block
    num_classes : int
        Number of output classes
    """

    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 1 * 1, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.pool(x)
        return self.body(x)


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) adapted for CIFAR-10 with 32x32 inputs.

    Follows the kuangliu implementation used in the course notebook.
    Uses a lightweight 3x3 conv stem suited for small images, with no
    auxiliary classifiers. Use this for the baseline CIFAR-10 comparison.

    Attributes
    ----------
    pre_layers : Sequential
        3x3 conv stem
    a3, b3 : Inception
        Stage 3 inception blocks
    maxpool : MaxPool2d
        Shared downsampling layer (used twice)
    a4 ... e4 : Inception
        Stage 4 inception blocks
    a5, b5 : Inception
        Stage 5 inception blocks
    avgpool : AdaptiveAvgPool2d
        Global average pooling to 1x1
    linear : Linear
        Final classifier head
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16,  32,  32)
        self.b3 = Inception(256, 128, 128, 192, 32,  96,  64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear  = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.maxpool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.linear(x)


class GoogLeNetAux(nn.Module):
    """
    GoogLeNet with the original ImageNet-style backbone and two auxiliary
    classifiers, for 224x224 inputs.

    The stem follows the paper exactly:
        7x7 conv (stride 2) -> MaxPool -> LRN
        1x1 conv -> 3x3 conv -> LRN -> MaxPool

    Two AuxClassifier branches are attached after inception blocks a4 and d4
    respectively. They are only active during training. At inference the model
    returns only the main logits.

    Loss weighting (to be applied in run.py):
        L_total = L_main + 0.3 * L_aux1 + 0.3 * L_aux2

    Attributes
    ----------
    conv1 : Sequential
        First conv stem block with LRN
    conv2 : Sequential
        Second conv stem block with LRN
    a3 ... b5 : Inception
        All inception blocks (same channel config as GoogLeNet)
    aux1 : AuxClassifier
        First auxiliary classifier (tapped after a4, 512 channels)
    aux2 : AuxClassifier
        Second auxiliary classifier (tapped after d4, 528 channels)
    avgpool : AdaptiveAvgPool2d
        Global average pooling to 1x1
    dropout : Dropout
        Dropout before the final linear layer (p=0.4 per the paper)
    linear : Linear
        Final classifier head
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Stem block 1: 7x7 conv -> pool -> LRN
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
        )

        # Stem block 2: 1x1 -> 3x3 conv -> LRN -> pool
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.a3 = Inception(192,  64,  96, 128, 16,  32,  32)
        self.b3 = Inception(256, 128, 128, 192, 32,  96,  64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        # Auxiliary classifiers tapped at a4 (512ch) and d4 (528ch)
        self.aux1 = AuxClassifier(in_channels=512, num_classes=num_classes)
        self.aux2 = AuxClassifier(in_channels=528, num_classes=num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.linear  = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)

        x = self.a4(x)
        aux1 = self.aux1(x) if self.training else None

        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        aux2 = self.aux2(x) if self.training else None

        x = self.e4(x)
        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)

        if self.training:
            return x, aux1, aux2
        return x