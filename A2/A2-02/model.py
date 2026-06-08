import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


#Building Block 

class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d -> BN -> ReLU blocks.
    Core building block of U-Net.
    padding=1 keeps spatial size the same (same-padding).
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


#U-Net from Scratch

class UNet(nn.Module):
    """
    U-Net built from scratch with random initialization.
    Encoder: DoubleConv + MaxPool
    Decoder: ConvTranspose2d + skip connection + DoubleConv
    """
    def __init__(self, in_channels=3, n_classes=3, features=[64, 128, 256, 512]):
        super().__init__()

        #Encoder
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        #Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        #Decoder
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))
            ch = f

        #Output
        self.output = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.output(x)


#U-Net with ResNet-18 Encoder + Skip Connections

class UNetResNet18(nn.Module):
    """
    U-Net with pretrained ResNet-18 encoder.
    Skip connections bring high-resolution features from encoder to decoder.

    ResNet-18 on 128x128 input:
      stem_conv (stride=2) → H/2,  64ch   [s0 skip]
      stem_pool (stride=2) → H/4
      layer1               → H/4,  64ch   [s1 skip]
      layer2 (stride=2)    → H/8,  128ch  [s2 skip]
      layer3 (stride=2)    → H/16, 256ch  [s3 skip]
      layer4 (stride=2)    → H/32, 512ch  [s4 skip]
    Decoder: 5 upsamples → back to H
    """
    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()

        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet  = models.resnet18(weights=weights)

        # encoder
        self.stem_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.stem_pool = resnet.maxpool
        self.enc1      = resnet.layer1
        self.enc2      = resnet.layer2
        self.enc3      = resnet.layer3
        self.enc4      = resnet.layer4

        # bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # decoder
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(512 + 512, 512)

        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(256 + 256, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(128 + 128, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)

        self.up0  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(32 + 64, 32)

        self.output = nn.Conv2d(32, n_classes, kernel_size=1)

    def _cat(self, x, skip):
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:])
        return torch.cat([skip, x], dim=1)

    def forward(self, x):
        s0 = self.stem_conv(x)
        sp = self.stem_pool(s0)
        s1 = self.enc1(sp)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        x = self.bottleneck(s4)

        x = self.up4(x); x = self._cat(x, s4); x = self.dec4(x)
        x = self.up3(x); x = self._cat(x, s3); x = self.dec3(x)
        x = self.up2(x); x = self._cat(x, s2); x = self.dec2(x)
        x = self.up1(x); x = self._cat(x, s1); x = self.dec1(x)
        x = self.up0(x); x = self._cat(x, s0); x = self.dec0(x)

        return self.output(x)


#U-Net with ResNet-18 Encoder — NO Skip Connections

class UNetResNet18NoSkip(nn.Module):
    """
    Same ResNet-18 encoder as UNetResNet18 but with skip connections REMOVED.
    Decoder must reconstruct spatial detail from bottleneck alone.
    Used for ablation study to measure skip connection contribution.
    """
    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()

        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet  = models.resnet18(weights=weights)

        # encoder — identical to UNetResNet18
        self.stem_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.stem_pool = resnet.maxpool
        self.enc1      = resnet.layer1
        self.enc2      = resnet.layer2
        self.enc3      = resnet.layer3
        self.enc4      = resnet.layer4

        # bottleneck — identical
        self.bottleneck = DoubleConv(512, 1024)

        # decoder — NO skip connections
        # input channels are halved at each stage since we don't concat skips
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(512, 512)       # no concat → 512 only

        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(256, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(128, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(64, 64)

        self.up0  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(32, 32)         # no concat → 32 only

        self.output = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]   # save original size
    
        s0 = self.stem_conv(x)
        sp = self.stem_pool(s0)
        s1 = self.enc1(sp)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
    
        x = self.bottleneck(s4)
    
        x = self.up4(x)
        if x.shape[2:] != s4.shape[2:]:
            x = F.interpolate(x, size=s4.shape[2:])
        x = self.dec4(x)
    
        x = self.up3(x)
        if x.shape[2:] != s3.shape[2:]:
            x = F.interpolate(x, size=s3.shape[2:])
        x = self.dec3(x)
    
        x = self.up2(x)
        if x.shape[2:] != s2.shape[2:]:
            x = F.interpolate(x, size=s2.shape[2:])
        x = self.dec2(x)
    
        x = self.up1(x)
        if x.shape[2:] != s1.shape[2:]:
            x = F.interpolate(x, size=s1.shape[2:])
        x = self.dec1(x)
    
        x = self.up0(x)
        if x.shape[2:] != s0.shape[2:]:
            x = F.interpolate(x, size=s0.shape[2:])
        x = self.dec0(x)
    
        # final resize to match input size exactly
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size)
    
        return self.output(x)


#Model Factory

def get_model(model_name, n_classes=3, pretrained=True):
    """
    Returns the requested model by name.
    Args:
        model_name : 'unet_scratch' | 'unet_resnet18' | 'unet_resnet18_no_skip'
        n_classes  : number of segmentation classes
        pretrained : use ImageNet pretrained weights for ResNet encoder
    """
    if model_name == 'unet_scratch':
        return UNet(n_classes=n_classes)

    elif model_name == 'unet_resnet18':
        return UNetResNet18(n_classes=n_classes, pretrained=pretrained)

    elif model_name == 'unet_resnet18_no_skip':
        return UNetResNet18NoSkip(n_classes=n_classes, pretrained=pretrained)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            "Choose from: unet_scratch, unet_resnet18, unet_resnet18_no_skip"
        )