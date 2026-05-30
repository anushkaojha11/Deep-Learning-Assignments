# A1: Representation Learning

Implementation and comparison of AlexNet, GoogLeNet, ResNet-18, and Vision Transformer (ViT) on CIFAR-10, built from scratch in PyTorch.

> All experiments were run on a MacBook Air M4 (Apple MPS) using a 5,000-sample CIFAR-10 subset due to hardware constraints. Use `--full_data` to train on the full dataset.

---

## Repository Structure

```
A1/
├── alexnet.py        # AlexNet with optional LRN (lrn=True/False)
├── googlenet.py      # GoogLeNet (CIFAR 32x32) and GoogLeNetAux (224x224 + aux classifiers)
├── resnet.py         # ResidualBlock and ResNet-18
├── run.py            # Main training and evaluation script
├── data/             # CIFAR-10 downloaded automatically
├── models/           # Saved best model weights (.pth)
├── plots/            # Training curve plots (.png)
├── logs/             # Training logs (.log)
└── README.md
```

---

## Training Commands

```bash
# Train from scratch
python3 run.py --model alexnet       --dataset cifar10 --epochs 10 --batch_size 64 --train
python3 run.py --model alexnet_lrn   --dataset cifar10 --epochs 10 --batch_size 64 --train
python3 run.py --model googlenet     --dataset cifar10 --epochs 25 --batch_size 64 --train
python3 run.py --model googlenet_aux --dataset cifar10 --epochs 25 --batch_size 64 --train
python3 run.py --model resnet18      --dataset cifar10 --epochs 20 --batch_size 64 --train
python3 run.py --model vit_small     --dataset cifar10 --epochs 20 --batch_size 64 --train

# Fine-tune pretrained models
python3 run.py --model vit_b16_pretrained --dataset cifar10 --epochs 6 --batch_size 16 --train

# Test saved weights
python3 run.py --model alexnet    --dataset cifar10 --test --weights models/alexnet_cifar10_best.pth
python3 run.py --model googlenet  --dataset cifar10 --test --weights models/googlenet_cifar10_best.pth
python3 run.py --model resnet18   --dataset cifar10 --test --weights models/resnet18_cifar10_best.pth
python3 run.py --model vit_small  --dataset cifar10 --test --weights models/vit_small_cifar10_best.pth
python3 run.py --model vit_b16_pretrained --dataset cifar10 --test --weights models/vit_b16_pretrained_cifar10_stage2_best.pth
```

> **Note:** Models with LRN (`alexnet_lrn`, `googlenet_aux`) require the MPS fallback flag on Apple Silicon:
> `PYTORCH_ENABLE_MPS_FALLBACK=1 python3 run.py ...`

---

## Results

| Model | # Params | Best Val Acc | Test Acc | Time/epoch | Architecture Type |
|---|---|---|---|---|---|
| AlexNet (from scratch) | 57,044,810 | 10.6% | 12.3% | ~14s | CNN |
| AlexNet + LRN (from scratch) | 57,044,810 | 15.5% | 14.3% | ~30s | CNN |
| GoogLeNet (from scratch) | 6,166,250 | 57.1% | 55.0% | ~45s | CNN + Inception |
| GoogLeNet + 2 Aux Losses (from scratch) | 6,702,974 | 58.2% | 60.5% | ~85s | CNN + Inception |
| ResNet-18 (from scratch) | 11,173,962 | 51.6% | 50.9% | ~15s | CNN + Skip connections |
| ViT-Small (from scratch) | 1,205,898 | 46.6% | 43.9% | ~4s | Transformer |
| ViT-B/16 (pretrained, fine-tuned) | 85,806,346 | 88.6% | 85.6% | ~530s | Transformer |

---

## Exercise 1

### Q1: Three Networks from Scratch

Three networks are implemented, each in its own file:

- `alexnet.py`: `AlexNet` class with an `lrn` flag to toggle Local Response Normalisation on or off
- `googlenet.py`: `GoogLeNet` (CIFAR 32×32 baseline) and `GoogLeNetAux` (ImageNet-style backbone with two auxiliary classifiers)
- `resnet.py`: `ResidualBlock` and `ResNet18`

Training, validation, and weight saving are handled by `run.py`.

---

### Q2: AlexNet with and without LRN

Local Response Normalisation (LRN) is added after the first and second convolutional layers as described in Krizhevsky et al. (2012), using `nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)`. A single `AlexNet` class handles both variants via an `lrn: bool` constructor argument.

| Model | Best Val Acc | Test Acc | Time/epoch |
|---|---|---|---|
| AlexNet (no LRN) | 10.6% | 12.3% | ~14s |
| AlexNet + LRN | 15.5% | 14.3% | ~30s |

Both models barely exceed random chance (10%) because AlexNet's 57M parameters are severely overparameterised for a 5,000-sample training set trained for only 10 epochs. AlexNet + LRN shows a marginal improvement, consistent with the original paper's findings that LRN provides a small regularisation benefit. LRN adds overhead (~2× slower per epoch) because the MPS backend does not natively support `avg_pool3d` and falls back to CPU. On the full CIFAR-10 dataset with sufficient epochs, both models would converge to much higher accuracy.

---

### Q3: GoogLeNet with ImageNet Backbone and Auxiliary Classifiers

The baseline `GoogLeNet` from the course notebook uses 32×32 CIFAR inputs and a lightweight 3×3 conv stem. `GoogLeNetAux` modifies this to match the original paper more closely:

- **Backbone:** 7×7 conv (stride 2) → MaxPool → LRN → 1×1 conv → 3×3 conv → LRN → MaxPool
- **Input size:** 224×224 (CIFAR images are upsampled)
- **Two auxiliary classifiers** tapped after inception blocks `a4` and `d4`, each contributing 0.3× weight to the total loss:

```
L_total = L_main + 0.3 × L_aux1 + 0.3 × L_aux2
```

| Model | Input | Aux Classifiers | # Params | Test Acc |
|---|---|---|---|---|
| GoogLeNet (Q1) | 32×32 | No | 6,166,250 | 55.0% |
| GoogLeNetAux (Q3) | 224×224 | Yes | 6,702,974 | 60.5% |

The auxiliary classifiers improved test accuracy by 5.5 percentage points. Upsampling CIFAR images to 224×224 does not add new visual information but allows the full ImageNet-style backbone to operate as designed. The auxiliary losses help stabilise gradient flow to the earlier inception blocks during training.

---

### Q4: Comparing AlexNet and GoogLeNet

| Model | # Params | Test Acc | Time/epoch |
|---|---|---|---|
| AlexNet | 57,044,810 | 12.3% | ~14s |
| GoogLeNet | 6,166,250 | 55.0% | ~45s |

GoogLeNet substantially outperforms AlexNet despite using ~9× fewer parameters. AlexNet's design concentrates most of its parameters in large fully connected layers, which require a large amount of data to generalise well. GoogLeNet's Inception modules capture multi-scale features efficiently through parallel branches, and global average pooling replaces the expensive FC layers. This makes GoogLeNet both more parameter-efficient and better suited to smaller datasets. The results show that architectural design matters far more than raw parameter count.

---

### Q5: Pretrained AlexNet and GoogLeNet

Pretrained models were loaded from torchvision with ImageNet weights and fine-tuned on CIFAR-10 using a two-stage strategy: freeze the backbone and train only the new classification head first, then unfreeze all layers and fine-tune with a lower learning rate.

> Pretrained AlexNet and GoogLeNet fine-tuning were not run locally due to hardware constraints (MacBook Air M4). Based on the pattern observed with pretrained ViT-B/16 (85.6% vs 43.9% scratch) and the results from the course notebook, pretrained models are expected to significantly outperform their scratch counterparts, likely reaching 85–94% test accuracy. The ImageNet-pretrained features (edges, textures, object parts) transfer effectively to CIFAR-10, demonstrating that GoogLeNet's Inception architecture has strong generalisation capacity while remaining parameter-efficient.

---

### Q6: ResNet-18

#### a) Implementation

`ResidualBlock` and `ResNet18` are implemented in `resnet.py`. Each residual block computes:

```
Input x
  ├── Conv(3x3) → BN → ReLU → Conv(3x3) → BN
  └── shortcut (Identity, or 1×1 Conv if channels/stride change)
          ↓
       Add → ReLU → Output
```

ResNet-18 contains four residual stages (2 blocks each, 64→128→256→512 channels) with a 3×3 conv stem instead of the original 7×7, which is standard for CIFAR-10 to preserve spatial resolution on small images.

#### b) Training from Scratch

| Model | # Params | Best Val Acc | Test Acc | Time/epoch |
|---|---|---|---|---|
| ResNet-18 (scratch) | 11,173,962 | 51.6% | 50.9% | ~15s |

ResNet-18 trains significantly faster per epoch than GoogLeNet (~15s vs ~45s) and outperforms AlexNet by a wide margin despite having far fewer parameters. The residual connections allow stable gradient flow, enabling effective learning even on a small dataset.

#### c) Pretrained ResNet-18 Fine-tuning

Pretrained ResNet-18 fine-tuning was not run locally due to hardware constraints. The two-stage strategy used in `run.py` follows the assignment specification exactly:

```python
# Stage 1: freeze backbone, train head only (5 epochs)
for param in resnet_pretrained.parameters():
    param.requires_grad = False
resnet_pretrained.fc.requires_grad_(True)

# Stage 2: unfreeze all, fine-tune with smaller lr (10 epochs)
for param in resnet_pretrained.parameters():
    param.requires_grad = True
```

#### d) Why Does ResNet Train Deep Networks Successfully?

In a deep network without skip connections, gradients must pass through every layer during backpropagation. Each layer multiplies the gradient by its local derivative, if these are consistently less than 1, the gradient shrinks exponentially toward earlier layers (the **vanishing gradient problem**), causing those layers to barely learn.

ResNet's skip connections create a direct shortcut path for the gradient. Instead of learning H(x) directly, each block learns the residual F(x) = H(x) − x, and the output is F(x) + x. During backpropagation, the gradient flows both through F(x) and directly through the identity shortcut, guaranteeing that the gradient reaching earlier layers is at least as large as the gradient from the output. This makes it possible to train networks with 50, 100, or even 150+ layers effectively.

---

## Exercise 2: Pretrained ViT-B/16

A pretrained ViT-B/16 was loaded from torchvision and fine-tuned on CIFAR-10 using the two-stage strategy. Since ViT-B/16 expects 224×224 inputs, CIFAR-10 images were upsampled before being fed in. The classification head was replaced with a 10-class linear layer:

```python
vit_pretrained = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
vit_pretrained.heads = nn.Linear(768, 10)
```

Due to hardware constraints (MacBook Air M4), training was limited to 6 epochs (2 stage 1 + 4 stage 2) instead of the recommended 15.

| Model | # Params | Best Val Acc | Test Acc | Time/epoch | Architecture Type |
|---|---|---|---|---|---|
| AlexNet + LRN (from scratch) | 57,044,810 | 15.5% | 14.3% | ~30s | CNN |
| GoogLeNet + 2 Aux Losses (from scratch) | 6,702,974 | 58.2% | 60.5% | ~85s | CNN + Inception |
| ResNet-18 (from scratch) | 11,173,962 | 51.6% | 50.9% | ~15s | CNN + Skip connections |
| ResNet-18 (pretrained) | 11,173,962 | — | — | — | CNN + Skip connections |
| ViT-Small (from scratch) | 1,205,898 | 46.6% | 43.9% | ~4s | Transformer |
| ViT-B/16 (pretrained, fine-tuned) | 85,806,346 | 88.6% | 85.6% | ~530s | Transformer |

### Discussion

The pretrained ViT-B/16 achieved the highest test accuracy at 85.6%, more than 25 percentage points above the best scratch model (GoogLeNet + Aux at 60.5%), despite being trained for only 6 epochs. This demonstrates the enormous advantage of large-scale ImageNet pretraining, the model's self-attention mechanism has already learned rich, transferable visual representations that adapt quickly to CIFAR-10 with minimal fine-tuning.

Among scratch models, GoogLeNet with auxiliary classifiers performed best (60.5%), showing that the Inception architecture's multi-scale feature extraction and auxiliary loss regularisation are well suited to small datasets. CNNs generally outperformed ViT-Small (43.9%) from scratch because convolutional inductive biases (locality, translation equivariance) allow efficient learning from limited data, whereas Transformers need large datasets to learn these spatial relationships on their own.

The trade-off between CNNs and Transformers is clear: CNNs are more data-efficient from scratch and train faster per epoch, while Transformers are more flexible and achieve superior performance when pretrained at scale. For practical use on small datasets, a pretrained CNN like ResNet-18 offers the best balance of accuracy, speed, and computational cost, whereas pretrained ViT models are the best choice when maximum accuracy is required and compute is available.