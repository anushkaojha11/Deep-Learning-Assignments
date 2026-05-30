import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet implementation for CIFAR-10.

    Supports training with and without Local Response Normalization (LRN)
    via the `lrn` flag. LRN is inserted after the first and second
    convolutional layers as described in the original paper:
        Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
        ImageNet classification with deep convolutional neural networks.

    Parameters from the paper:
        LRN size=5, alpha=1e-4, beta=0.75, k=2

    Attributes
    ----------
    lrn : bool
        Whether Local Response Normalisation is enabled
    features : Sequential
        Convolutional feature extraction layers
    avgpool : AdaptiveAvgPool2d
        Reduces any spatial size to 6x6 before the classifier
    classifier : Sequential
        Three fully-connected layers producing class logits
    """

    def __init__(self, num_classes: int = 10, lrn: bool = False) -> None:
        super().__init__()
        self.lrn = lrn

        def lrn_layer():
            return nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)

        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            *(([lrn_layer()]) if lrn else []),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            *(([lrn_layer()]) if lrn else []),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x