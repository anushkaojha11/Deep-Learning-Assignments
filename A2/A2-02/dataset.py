import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms
import os


# ImageNet normalization stats
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Oxford Pet has 3 classes:
# 1 = Pet (foreground)
# 2 = Background
# 3 = Border/uncertain
# We subtract 1 to make them 0-indexed: 0=Pet, 1=Background, 2=Border
CLASS_NAMES  = ['Pet', 'Background', 'Border']
CLASS_COLORS = np.array([
    [255, 100, 100],   # Pet      — red
    [100, 100, 255],   # Background — blue
    [255, 255, 100],   # Border   — yellow
], dtype=np.uint8)
NUM_CLASSES = 3


class PetSegDataset(Dataset):
    """
    Oxford-IIIT Pet dataset for semantic segmentation.
    Returns (image_tensor, mask_tensor) pairs.

    image : float32 tensor [3, H, W] normalized with ImageNet stats
    mask  : long tensor   [H, W]    values in {0, 1, 2}
    """
    def __init__(self, split='trainval', img_size=128, data_dir='./data'):
        os.makedirs(data_dir, exist_ok=True)

        self.img_size = img_size
        self.base     = OxfordIIITPet(
            data_dir,
            split        = split,
            target_types = 'segmentation',
            download     = True,
        )

        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        self.mask_tf = transforms.Compose([
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]

        img  = self.img_tf(img)

        # mask values are 1,2,3 → subtract 1 → 0,1,2
        mask = (self.mask_tf(mask).squeeze(0).long() - 1).clamp(0, 2)

        return img, mask


def get_dataloaders(img_size=128, batch_size=16, data_dir='./data', num_workers=2):
    """
    Returns train and test DataLoaders for Oxford Pet dataset.
    """
    from torch.utils.data import DataLoader

    train_dataset = PetSegDataset(
        split    = 'trainval',
        img_size = img_size,
        data_dir = data_dir,
    )
    test_dataset = PetSegDataset(
        split    = 'test',
        img_size = img_size,
        data_dir = data_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    print(f'Train : {len(train_dataset)} samples ({len(train_loader)} batches)')
    print(f'Test  : {len(test_dataset)} samples ({len(test_loader)} batches)')

    return train_loader, test_loader


def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization.
    tensor: [3, H, W] normalized tensor
    Returns: [H, W, 3] numpy array in [0, 1]
    """
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    img  = torch.clamp(tensor * std + mean, 0, 1)
    return img.permute(1, 2, 0).numpy()