# Project 02 — StyleGAN2 Face Generation (FFHQ 1024×1024)

StyleGAN2-based unconditional face generator trained progressively on FFHQ.
**Generator: 34.10 M parameters** (ONNX ~35.5 M, under the 40 M submission limit).

---

## Architecture

| Component | Detail |
|---|---|
| Input | `z ~ N(0, I)`, dim = 512 |
| Mapping network | 12-layer EqualLinear MLP, `w_dim=640`, `lr_mul=0.01` |
| Synthesis | 9 blocks (4×4 → 1024×1024), ModulatedConv2d + SqueezeConnection |
| Channel schedule | `channel_base=131072`, `channel_max=512` → 512 ch up to 256×256 |
| Output | 3 × 1024 × 1024, tanh-clamped to [−1, 1] |
| Discriminator | Residual blocks + MinibatchStddev, same channel schedule |

### Channel widths per resolution

| Resolution | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|
| Channels | 512 | 512 | 512 | 512 | 512 | 512 | 512 | 256 | 128 |

### Synthesis block structure

Each block after the 4×4 base uses a **SqueezeConnection** (StyleGAN-Small, arXiv:2407.05527)
instead of the standard second ModulatedConv2d + ToRGB:

```
conv1 (ModulatedConv2d, upsample) → noise1 → act
└─ SqueezeConnection:
     squeeze  (EqualConv2d, c → c/8)
     noise    (NoiseInjection)          ← replaces conv2 stochasticity
     to_rgb   (ModulatedConv2d 1×1)    → rgb_current
     excite   (EqualConv2d, c/8 → c)
     project  (EqualConv2d, 2c → c)   → x_out (next block input)
```

This reduces per-block parameters versus conv2+ToRGB while alleviating the
1×1 projection bottleneck that limits high-resolution detail in standard StyleGAN2.

### Training losses

| Loss | Formula | Frequency |
|---|---|---|
| D logistic | `softplus(−D(real)) + softplus(D(fake))` | every step |
| R1 penalty | `γ/2 · E[‖∇D(real)‖²]` | every 16 D steps |
| G logistic | `softplus(−D(fake))` | every step |
| Path-length reg | `(‖J_w^T v‖ − EMA)²` | every 4 G steps |

---

## Progressive Training

```
256×256  →  512×512  →  1024×1024
 (stage 1)   (stage 2)    (stage 3)
```

Each stage loads the previous checkpoint. New high-resolution blocks are
randomly initialised; all existing blocks continue training (no freezing).

| Stage | Resolution | Batch | `lr_g` | `r1_gamma` |
|---|---|---|---|---|
| 1 | 256 | 16 | 1e-3 | 10.0 |
| 2 | 512 | 8 | 1e-3 | 15.0 |
| 3 | 1024 | 4 | 1e-3 | 50.0 |

---

## Project Structure

```
project02/
├── src/
│   ├── models/
│   │   ├── layers.py         # EqualLinear, ModulatedConv2d, SqueezeConnection, ToRGB, …
│   │   ├── generator.py      # MappingNetwork + SynthesisBlock* + StyleGAN2Generator
│   │   └── discriminator.py  # DiscBlock + MinibatchStddev + StyleGAN2Discriminator
│   ├── data/
│   │   └── dataset.py        # FFHQDataset (pre-split dir or flat indexed dir)
│   ├── training/
│   │   ├── losses.py         # d_logistic, d_r1, g_nonsaturating, g_path_length
│   │   └── trainer.py        # Trainer (AMP, lazy reg, grad clip, WandB, Drive backup)
│   └── utils/
│       ├── fid_score.py      # ValidFIDCache — precomputes real stats once
│       └── parallel_unzip.py # Fast parallel ZIP extraction (for FFHQ)
├── configs/
│   ├── train_256.yaml
│   ├── train_512.yaml
│   └── train_1024.yaml
├── main.ipynb                # Colab training notebook
├── submit_test.ipynb         # ONNX export + parameter count verification
└── pyproject.toml
```

---

## Setup

### Local development (uv)

```bash
git clone https://github.com/jyun-chae/skku-2-openai_pa2.git
cd skku-2-openai_pa2
uv sync
uv run python -c "from src.models.generator import StyleGAN2Generator; print('OK')"
```

### Colab training

Open `main.ipynb` and run cells top to bottom.
Required manual inputs:

```python
REPO_URL      = 'https://github.com/jyun-chae/skku-2-openai_pa2.git'
WANDB_API_KEY = 'YOUR_KEY'
DRIVE_DIR     = '/content/drive/MyDrive/project02'
```

Data zip files are expected in `{DRIVE_DIR}/data/`:
- Stage 1: `train_50k_256.zip`, `valid_10k_256.zip`
- Stage 2: `train_50k_512.zip`, `valid_10k_512.zip`
- Stage 3: `train_50k_1024.zip` (split chunks), `valid_10k_1024.zip`

---

## How to Train

### Stage 1 — 256×256 from scratch

```python
cfg = load_cfg('configs/train_256.yaml')
trainer = Trainer(cfg)
trainer.fit(train_loader, valid_loader, wandb_run=run)
```

### Stage 2 — 512×512 from 256 checkpoint

```python
cfg = load_cfg('configs/train_512.yaml')
trainer = Trainer(cfg)
state = torch.load('ckpt_256_final.pth')
trainer.G.load_from_lower_resolution(state['G'])
trainer.D.load_from_lower_resolution(state['D'])
trainer.fit(...)
```

### Stage 3 — 1024×1024 from 512 checkpoint

Same as Stage 2 with `train_1024.yaml` and the 512 checkpoint.

---

## Evaluation

FID is computed on the **validation set** (10 k images) against 10 k generated samples
using pytorch-fid's InceptionV3. Real statistics are precomputed once per session.

```python
fid_cache = ValidFIDCache(valid_loader, device)
fid = fid_cache.compute(G, n_gen=10000, batch_size=16)
print(f'FID: {fid:.2f}')
```

---

## ONNX Export & Parameter Check

```python
# From generator instance
onnx_params = G.export_onnx('generator_1024.onnx', batch_size=1)
# Prints ONNX initializer count and asserts < 40 M
```

For a standalone parameter audit before submission, run **`submit_test.ipynb`**:
- Creates a random-weight 1024 generator (no checkpoint needed)
- Exports to ONNX with `noise_mode="none"` (constant-fold eliminates noise ops)
- Reports PyTorch params vs ONNX initializer count side-by-side

| Count | Value |
|---|---|
| PyTorch `named_parameters()` | ~34.10 M |
| ONNX `graph.initializer` | ~35.5 M |
| 40 M limit margin | ~4.5 M |

> ONNX count exceeds PyTorch count by ~1.4 M because `do_constant_folding=True`
> bakes EqualLinear `weight × scale` products into the graph as additional initializers.

---

## Key Implementation Notes

### Numerical stability (AMP / fp16)

| Issue | Location | Fix |
|---|---|---|
| Demodulation overflow | `ModulatedConv2d` | Cast `weight²` sum to fp32 before `rsqrt` |
| R1 double-backward in fp16 | `trainer.d_step` | Run R1 under `autocast(enabled=False)` |
| PL `sqrt(0)` backward | `g_path_length_loss` | `(grad².sum() + 1e-8).sqrt()` |

### Lazy regularization

R1 runs every 16 D steps; PL runs every 4 G steps.
Optimizer LR and β₂ are pre-scaled by `interval / (interval + 1)` to maintain
effective learning dynamics identical to per-step regularization.

### Gradient clipping

`clip_grad_norm_(max_norm=1.0)` is applied after every G update (both main loss
and PL regularization step).

---

## Dependencies

```
torch >= 2.2.0
torchvision >= 0.17.0
wandb >= 0.17.0
pytorch-fid >= 0.3.0
onnx >= 1.16.0
pyyaml >= 6.0
pillow >= 10.0.0
scikit-image >= 0.21.0
```
