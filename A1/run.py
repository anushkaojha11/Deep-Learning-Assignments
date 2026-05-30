import argparse
import copy
import logging
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as transforms

from AlexNet import AlexNet
from GoogleNet import GoogLeNet, GoogLeNetAux
from ResNet import ResNet18


# ViT-Small (from scratch, for CIFAR-10 32x32)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)           # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)           # (B, embed_dim, n_patches)
        return x.transpose(1, 2)   # (B, n_patches, embed_dim)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class ViTSmall(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=128, depth=6, n_heads=4, n_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout    = nn.Dropout(dropout)
        self.blocks     = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln   = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.dropout(x + self.pos_embed)
        x   = self.blocks(x)
        x   = self.ln(x[:, 0])
        return self.head(x)
    
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Logging setup

def setup_logging(model_name: str) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"{model_name}.log")

    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# Data loading

def _subset(dataset, n: int):
    """Return a random subset of size n from a PyTorch dataset."""
    indices = torch.randperm(len(dataset))[:n].tolist()
    return torch.utils.data.Subset(dataset, indices)


def get_dataloaders(dataset: str, batch_size: int = 64,
                    img_size: int = 224, small: bool = True):
    """
    Build train / val / test DataLoaders for CIFAR-10.

    Parameters
    ----------
    img_size : int
        Target spatial size. Use 224 for AlexNet/GoogLeNetAux/pretrained,
        32 for GoogLeNet/ResNet18/ViTSmall.
    small : bool
        If True, subsample to 5000 train / 1000 val / 1000 test so training
        is fast enough for a MacBook without a discrete GPU.
    """
    if dataset != "cifar10":
        raise ValueError("Only cifar10 is supported.")

    if img_size == 224:
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])

    train_full = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=tf)
    test_full  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=tf)

    if small:
        train_full = _subset(train_full, 5000)
        test_full  = _subset(test_full,  1000)

    train_set, val_set = torch.utils.data.random_split(
        train_full, [int(len(train_full) * 0.8), len(train_full) - int(len(train_full) * 0.8)]
    )

    # num_workers=0 is safest on macOS to avoid multiprocessing issues
    loader_kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=False)

    return {
        "train": torch.utils.data.DataLoader(train_set, shuffle=True,  **loader_kwargs),
        "val":   torch.utils.data.DataLoader(val_set,   shuffle=False, **loader_kwargs),
        "test":  torch.utils.data.DataLoader(test_full, shuffle=False, **loader_kwargs),
    }


# Pretrained model builders

def build_pretrained_alexnet(device):
    model = tv_models.alexnet(weights=tv_models.AlexNet_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, 10)
    return model.to(device)


def build_pretrained_googlenet(device):
    model = tv_models.googlenet(
        weights=tv_models.GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=True)
    model.fc = nn.Linear(1024, 10)
    if model.aux1 is not None:
        model.aux1.fc2 = nn.Linear(1024, 10)
    if model.aux2 is not None:
        model.aux2.fc2 = nn.Linear(1024, 10)
    return model.to(device)


def build_pretrained_resnet18(device):
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 10)
    return model.to(device)


def build_pretrained_vit_b16(device):
    model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT)
    model.heads = nn.Linear(768, 10)
    return model.to(device)


# Model and optimiser registry

SCRATCH_MODELS = {
    "alexnet": {
        "factory": lambda: AlexNet(lrn=False),
        "img_size": 224,
        "optim": "sgd", "lr": 0.001, "momentum": 0.9,
        "epochs": 10,
    },
    "alexnet_lrn": {
        "factory": lambda: AlexNet(lrn=True),
        "img_size": 224,
        "optim": "sgd", "lr": 0.001, "momentum": 0.9,
        "epochs": 10,
    },
    "googlenet": {
        "factory": GoogLeNet,
        "img_size": 32,
        "optim": "adam", "lr": 0.01,
        "epochs": 25,
    },
    "googlenet_aux": {
        "factory": GoogLeNetAux,
        "img_size": 224,
        "optim": "adam", "lr": 0.001,
        "epochs": 25,
    },
    "resnet18": {
        "factory": ResNet18,
        "img_size": 32,
        "optim": "sgd", "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
        "epochs": 20,
    },
    "vit_small": {
        "factory": ViTSmall,
        "img_size": 32,
        "optim": "adam", "lr": 1e-3, "weight_decay": 1e-4,
        "epochs": 20,
    },
}

PRETRAINED_MODELS = {
    "alexnet_pretrained":   (build_pretrained_alexnet,   224),
    "googlenet_pretrained": (build_pretrained_googlenet, 224),
    "resnet18_pretrained":  (build_pretrained_resnet18,  224),
    "vit_b16_pretrained":   (build_pretrained_vit_b16,   224),
}


def build_scratch_model(name: str, device: torch.device):
    cfg       = SCRATCH_MODELS[name]
    model     = cfg["factory"]().to(device)
    opt_cls   = optim.SGD if cfg["optim"] == "sgd" else optim.Adam
    opt_kwargs = {"lr": cfg["lr"]}
    if "momentum"     in cfg: opt_kwargs["momentum"]     = cfg["momentum"]
    if "weight_decay" in cfg: opt_kwargs["weight_decay"] = cfg["weight_decay"]
    optimizer = opt_cls(model.parameters(), **opt_kwargs)
    return model, optimizer, cfg["epochs"], cfg["img_size"]

# Training loop

def train_model(model, dataloaders, criterion, optimizer,
                num_epochs, model_name, device, logger):
    os.makedirs("models", exist_ok=True)

    best_wts    = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    val_acc_history = []
    loss_history    = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        logger.info("-" * 40)

        for phase in ("train", "val"):
            model.train() if phase == "train" else model.eval()

            running_loss    = 0.0
            running_correct = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    # Handle auxiliary outputs (GoogLeNetAux returns a tuple)
                    if isinstance(outputs, tuple):
                        main_out  = outputs[0]
                        aux_losses = [criterion(o, labels)
                                      for o in outputs[1:] if o is not None]
                        loss = criterion(main_out, labels) + 0.3 * sum(aux_losses)
                        outputs = main_out
                    else:
                        loss = criterion(outputs, labels)

                    preds = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss    += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()

            epoch_loss = running_loss    / len(dataloaders[phase].dataset)
            epoch_acc  = running_correct / len(dataloaders[phase].dataset)

            logger.info(f"  {phase:5s}  loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

            if phase == "val":
                val_acc_history.append(epoch_acc)
                loss_history.append(epoch_loss)

                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_wts     = copy.deepcopy(model.state_dict())
                    save_path    = os.path.join("models", f"{model_name}_best.pth")
                    torch.save(best_wts, save_path)
                    logger.info(f"  --> Best val acc {best_val_acc:.4f} — saved to {save_path}")

        logger.info(f"  Time: {time.time() - t0:.1f}s")

    logger.info(f"\nBest val accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_wts)
    return model, val_acc_history, loss_history


# Two-stage fine-tuning for pretrained models

def finetune_pretrained(model, dataloaders, model_name, device, logger,
                        stage1_epochs=5, stage2_epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()

    # Identify the classification head
    if   hasattr(model, "fc"):             head = model.fc
    elif hasattr(model, "heads"):          head = model.heads
    elif hasattr(model, "classifier"):     head = model.classifier[6]
    else:
        raise AttributeError("Cannot find classifier head.")

    # Stage 1: freeze backbone, train head only
    logger.info(f"\n--- Stage 1: head only ({stage1_epochs} epochs) ---")
    for p in model.parameters():
        p.requires_grad = False
    head.requires_grad_(True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model, va1, ll1 = train_model(
        model, dataloaders, criterion, optimizer,
        stage1_epochs, f"{model_name}_stage1", device, logger)

    # Stage 2: unfreeze all, fine-tune with smaller lr
    logger.info(f"\n--- Stage 2: full fine-tune ({stage2_epochs} epochs) ---")
    for p in model.parameters():
        p.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)
    model, va2, ll2 = train_model(
        model, dataloaders, criterion, optimizer,
        stage2_epochs, f"{model_name}_stage2", device, logger)

    return model, va1 + va2, ll1 + ll2


# Evaluation

def test_model(model, test_loader, device, logger):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += labels.size(0)

    acc = correct / total
    logger.info(f"\nTest accuracy: {acc:.4f}  ({correct}/{total})")
    return acc


# Plotting

def plot_history(val_acc_history, loss_history, model_name):
    os.makedirs("plots", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(loss_history,    marker="o", label="val loss")
    ax1.set_title("Validation loss per epoch")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(val_acc_history, marker="o", label="val acc", color="green")
    ax2.set_title("Validation accuracy per epoch")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.suptitle(model_name)
    plt.tight_layout()

    path = os.path.join("plots", f"{model_name}_curves.png")
    plt.savefig(path)
    print(f"Curves saved to {path}")
    plt.close()


# Entry point

def main():
    parser = argparse.ArgumentParser(
        description="Train/test CNN and ViT models on CIFAR-10")

    parser.add_argument("--model",      required=True,
                        choices=list(SCRATCH_MODELS) + list(PRETRAINED_MODELS))
    parser.add_argument("--dataset",    default="cifar10", choices=["cifar10"])
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--test",       action="store_true")
    parser.add_argument("--weights",    type=str, default=None)
    parser.add_argument("--full_data",  action="store_true",
                        help="Use full CIFAR-10 (default: small subset for fast local runs)")

    args   = parser.parse_args()
    device = get_device()
    small  = not args.full_data

    model_name = f"{args.model}_{args.dataset}"
    logger     = setup_logging(model_name)

    logger.info(f"Device  : {device}")
    logger.info(f"Model   : {args.model}")
    logger.info(f"Dataset : {args.dataset}  (small={small})")

    #Training 
    if args.train:
        if args.model in PRETRAINED_MODELS:
            builder, img_size = PRETRAINED_MODELS[args.model]
            model = builder(device)
            dataloaders = get_dataloaders(
                args.dataset, args.batch_size, img_size=img_size, small=small)

            total_epochs  = args.epochs or 15
            stage1_epochs = max(1, total_epochs // 3)
            stage2_epochs = total_epochs - stage1_epochs

            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Params  : {n_params:,}")

            model, val_acc_history, loss_history = finetune_pretrained(
                model, dataloaders, model_name, device, logger,
                stage1_epochs=stage1_epochs, stage2_epochs=stage2_epochs)

        else:
            model, optimizer, default_epochs, img_size = build_scratch_model(
                args.model, device)
            dataloaders = get_dataloaders(
                args.dataset, args.batch_size, img_size=img_size, small=small)

            num_epochs = args.epochs or default_epochs
            n_params   = sum(p.numel() for p in model.parameters())
            logger.info(f"Params  : {n_params:,}")
            logger.info(f"Epochs  : {num_epochs}")

            criterion = nn.CrossEntropyLoss()
            model, val_acc_history, loss_history = train_model(
                model, dataloaders, criterion, optimizer,
                num_epochs, model_name, device, logger)

        plot_history(val_acc_history, loss_history, model_name)

    #Testing
    if args.test:
        if args.model in PRETRAINED_MODELS:
            builder, img_size = PRETRAINED_MODELS[args.model]
            model = builder(device)
        else:
            model, _, _, img_size = build_scratch_model(args.model, device)

        weights_path = args.weights or os.path.join(
            "models", f"{model_name}_best.pth")

        if not os.path.exists(weights_path):
            logger.error(f"Weights not found: {weights_path}")
            return

        model.load_state_dict(torch.load(weights_path, map_location=device))
        logger.info(f"Loaded weights from {weights_path}")

        dataloaders = get_dataloaders(
            args.dataset, args.batch_size,
            img_size=img_size if args.model in PRETRAINED_MODELS else
                     SCRATCH_MODELS[args.model]["img_size"],
            small=small)

        test_model(model, dataloaders["test"], device, logger)

    if not args.train and not args.test:
        parser.print_help()


if __name__ == "__main__":
    main()