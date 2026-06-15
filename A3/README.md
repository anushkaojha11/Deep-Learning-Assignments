# A3: Self-Supervised Learning

This assignment implements and evaluates three families of Self-Supervised Learning (SSL) on CIFAR-10: contrastive (SimCLR), self-distillation (DINO), and masked reconstruction (MAE). All models are evaluated using **linear probing** — the encoder is frozen and only a single linear layer is trained on top with labels.

---

## How to Run

```bash
# Train
python3 run.py --model dino --epochs 50 --train
python3 run.py --model mae  --epochs 50 --train

# Linear evaluation (from saved checkpoint)
python3 run.py --model dino --weights saved/dino_nlocal4.pt        --evaluate --linear
python3 run.py --model mae  --weights saved/mae_mask75_encoder.pt  --evaluate --linear

# Ablations
python3 run.py --model dino --no-centering       --epochs 50 --train
python3 run.py --model dino --n-local 0          --epochs 50 --train
python3 run.py --model mae  --mask-ratio 0.25    --epochs 50 --train
python3 run.py --model mae  --mask-ratio 0.50    --epochs 50 --train
```

---

## Results

### DINO Ablations

| Setting | Linear Eval Acc | Time/epoch | Final Loss | Center Norm |
|---|---|---|---|---|
| Default (2 global + 4 local, with centering) | 69.52% | ~157s | 2.3964 | 36.15 |
| No centering | 37.88% | ~164s | 0.0000 | 241.65 |
| No local crops (n_local=0) | 63.67% | ~68s | 1.7650 | 35.36 |

### MAE Ablations

| Mask Ratio | Recon Loss | Linear Eval Acc | Time/epoch |
|---|---|---|---|
| 0.75 | 0.4810 | 49.29% | ~26s |
| 0.50 | 0.3486 | 46.44% | ~24s |
| 0.25 | 0.2661 | 44.55% | ~20s |

### Three-Way Comparison (SimCLR vs DINO vs MAE)

| Metric | SimCLR | DINO | MAE |
|---|---|---|---|
| Backbone | ResNet-18 | ViT-Tiny | ViT (custom) |
| Needs negative pairs? | Yes | No | No |
| Needs EMA teacher? | No | Yes | No |
| Linear Eval Accuracy | 64.83% | 69.52% | 49.29% |
| Training time/epoch | ~23s | ~157s | ~26s |
| t-SNE cluster quality (1–5) | 3 | 4 | 2 |
| Has interpretable attention maps? | No | Yes | No |

---

## Exercise Answers

### Exercise 1 — DINO Ablations

**1a) Center norm across training epochs**

The center norm grows steadily throughout training in the default (with centering) setting, stabilizing around 36 by epoch 50. This is expected — the center accumulates a running mean of teacher outputs and gradually shifts to cancel dominant dimensions, preventing collapse.

In the no-centering variant, the center norm explodes to 241 by epoch 50 (since the center buffer still exists but is never subtracted from teacher logits), and the loss collapses to exactly 0.0000 — a clear sign of mode collapse where the model outputs the same vector for every image.

**1b) Why removing centering causes collapse, and why removing local crops hurts**

**Removing centering causes collapse** because without subtracting the running mean from teacher logits, one output dimension can dominate the softmax distribution. The teacher consistently assigns near-zero probability mass to all but one dimension, so the student learns to trivially predict that single dominant dimension regardless of input — meaning all images map to the same representation. The loss reaches 0 not because the model learned anything, but because it found a degenerate shortcut. The no-centering run confirms this: loss = 0.0000 and linear eval drops from 69.52% to 37.88% (barely above random for a 10-class problem).

**Removing local crops hurts representation quality** because the multi-crop strategy is central to DINO's learning signal. Local crops force the student to predict global context (seen by the teacher) from only a small, partial view of the image. This creates a strong self-supervised objective — the model must understand what object it is looking at from a tiny patch. Without local crops, both student and teacher see the same scale of views, making the task easier and reducing the richness of the learned representations. The accuracy drop from 69.52% to 63.67% confirms this.

---

### Exercise 2 — MAE Masking Ablation

**Results:**

| Mask Ratio | Recon Loss | Linear Eval Acc |
|---|---|---|
| 0.75 | 0.4810 | 49.29% |
| 0.50 | 0.3486 | 46.44% |
| 0.25 | 0.2661 | 44.55% |

**Why low masking (0.25) produces worse representations despite lower reconstruction loss:**

At a low mask ratio like 0.25, only 25% of patches are hidden, so the model can reconstruct them by simply interpolating from the many visible neighboring patches — no global understanding is required. The reconstruction loss is low because the task is too easy, not because the encoder learned rich semantic features.

At 0.75 masking, the encoder sees only 25% of the image and must encode enough global semantic information to allow the decoder to reconstruct the remaining 75%. This forces the encoder to learn object-level structure, shape, and context rather than local texture patterns. The harder reconstruction task produces a higher loss but a far more useful representation, as confirmed by the 49.29% linear eval accuracy versus 44.55% at mask ratio 0.25.

This is a key insight of the MAE paper: reconstruction loss and representation quality are not the same thing. The loss measures pixel-level accuracy; the representation quality measures semantic content.

---

### Exercise 3 — Three-Way Comparison

**3a) Why MAE won for large-scale pre-training, and why DINO is still preferred for segmentation**

Two reasons MAE won for large-scale general pre-training:
1. **Scalability and simplicity:** MAE requires no contrastive pairs, no EMA teacher, no centering, and no careful augmentation design. It scales cleanly with model size and dataset size — larger ViTs trained longer on more data consistently improve. DINO requires careful tuning of temperature, momentum, and centering hyperparameters that become harder to manage at scale.
2. **Training efficiency:** MAE's encoder only processes the 25% visible patches during training, making each forward pass ~4× cheaper than processing the full image. This allows training much larger models within the same compute budget.

One reason DINO is still preferred for CV-only tasks like segmentation: DINO's [CLS] token attention maps exhibit emergent object segmentation — the model learns to localize foreground objects without any segmentation labels. This spatial awareness transfers directly to dense prediction tasks like segmentation and detection, whereas MAE's encoder produces patch-level features that require more adaptation for dense tasks.

**3b) Medical image segmentation with 500 labeled scans**

For a medical image segmentation system with only 500 labeled scans, **DINO** would be the best pre-training choice. Medical imaging tasks are inherently spatial — accurate segmentation requires precise localization of structures like tumors, organs, or lesions — and DINO's attention mechanism develops strong spatial and object-boundary awareness without any labels. The emergent foreground segmentation in DINO attention maps transfers directly to dense prediction tasks, meaning the frozen encoder already captures spatially meaningful features before any fine-tuning. With only 500 labeled scans, maximizing the quality of the pre-trained representation is critical, and DINO's linearly separable features (69.52% vs MAE's 49.29% in our experiments) mean that even a lightweight fine-tuning head can achieve strong segmentation performance. SimCLR is less suitable as it requires large batch sizes and lacks spatial awareness, while MAE produces strong global features but requires more labeled data to adapt effectively to dense prediction.

---

## Visualizations

### Loss Curves
- `plots/dino_nlocal4_loss.png` — DINO default training loss + center norm
- `plots/dino_nlocal4_nocentering_loss.png` — DINO no-centering (collapse visible)
- `plots/dino_nlocal0_loss.png` — DINO no local crops
- `plots/mae_mask75_loss.png` — MAE mask=0.75
- `plots/mae_mask50_loss.png` — MAE mask=0.50
- `plots/mae_mask25_loss.png` — MAE mask=0.25

### MAE Reconstruction
- `plots/mae_mask75_reconstruction.png` — Original / Masked (75%) / Reconstructed

### t-SNE
- `plots/dino_nlocal4_tsne.png` — DINO feature space
- `plots/mae_mask75_tsne.png` — MAE feature space

---

## Discussion

For a medical image segmentation project with limited labels (500 scans), DINO is the recommended pre-training approach. Medical segmentation demands precise spatial understanding — identifying organ boundaries, lesion extents, and tissue transitions — which aligns directly with DINO's emergent object localization from its [CLS] attention mechanism. Our experiments show DINO achieves 69.52% linear eval accuracy versus MAE's 49.29%, indicating its frozen encoder produces far more semantically structured and linearly separable representations. With only 500 labeled scans, the quality of the pre-trained encoder is critical since there is insufficient data to compensate for a weak initialization through extensive fine-tuning. While MAE excels at large-scale general pre-training due to its simplicity and scalability, its reconstruction-based objective does not naturally develop the spatial awareness that segmentation tasks require, making DINO the stronger choice for this constrained, spatially-demanding medical imaging scenario.