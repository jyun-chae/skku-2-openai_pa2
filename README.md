# Project 02 — StyleGAN2 Face Generation (FFHQ 1024×1024)

StyleGAN2-based unconditional face generator trained progressively on FFHQ.  
**Generator: 39.86 M parameters** (under the 40 M ONNX submission limit).

---

## Architecture

| Component | Detail |
|---|---|
| Input | `z ~ N(0, I)`, dim = 512 |
| Mapping network | 12-layer EqualLinear MLP, `w_dim=640`, `lr_mul=0.01` |
| Synthesis | Skip-RGB, 9 blocks (4×4 → 1024×1024), modulated + demodulated conv |
| Channel schedule | `channel_base=65536`, `channel_max=512` → 512 ch up to 128×128 |
| Output | 3 × 1024 × 1024, tanh-clamped to [-1, 1] |
| Discriminator | Residual, minibatch stddev, same channel schedule (not size-constrained) |

### Channel widths per resolution

| Resolution | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|
| Channels | 512 | 512 | 512 | 512 | 512 | 512 | 256 | 128 | 64 |

### Training losses

| Loss | Formula | Frequency |
|---|---|---|
| D logistic | `softplus(-D(real)) + softplus(D(fake))` | every step |
| R1 penalty | `γ/2 · E[‖∇D(real)‖²]` | every 16 D steps |
| G logistic | `softplus(-D(fake))` | every step |
| Path-length reg | `(‖J_w^T v‖ - EMA)²` | every 4 G steps |

---

## Progressive Training

```
256×256  →  512×512  →  1024×1024
  ↑ train      ↑ fine-tune   ↑ fine-tune
```

Each stage loads the previous checkpoint and extends the architecture with a new synthesis block (randomly initialised). Lower-resolution blocks are not frozen — all weights continue updating.

| Stage | Resolution | Batch | LR | Est. time (A100) |
|---|---|---|---|---|
| 1 | 256 | 32 | 2e-3 | ~6–8 h |
| 2 | 512 | 16 | 1e-3 | ~5–8 h |
| 3 | 1024 | 8 | 5e-4 | ~15–25 h |

---

## Project Structure

```
project02/
├── src/
│   ├── models/
│   │   ├── layers.py         # EqualLinear, ModulatedConv2d, NoiseInjection, ToRGB
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
├── parallel_unzip.py         # CLI entry-point (mirrors src/utils/parallel_unzip.py)
├── pyproject.toml
└── .python-version           # Python 3.11 (for uv)
```

---

## Setup

### Local development (uv)

```bash
git clone https://github.com/jyun-chae/skku-2-openai_pa2.git
cd skku-2-openai_pa2
uv sync          # creates .venv and installs all dependencies
uv run python -c "from src.models.generator import StyleGAN2Generator; print('OK')"
```

### Colab training

Open `main.ipynb` and run cells top to bottom.  
The only required manual inputs are:

```python
REPO_URL      = 'https://github.com/jyun-chae/skku-2-openai_pa2.git'
WANDB_API_KEY = 'YOUR_KEY'
DRIVE_DIR     = '/content/drive/MyDrive/project02'
```

Data zip files (`train_50k_256.zip`, `valid_10k_256.zip`) are expected in  
`{DRIVE_DIR}/data/`. Set `USE_DRIVE_IMAGES = True` on re-runs to skip unzip.

---

## How to Train

### Stage 1 — 256×256 from scratch

```python
# Notebook cell 6 (or equivalent script):
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
trainer.D.load_state_dict(state['D'], strict=False)
trainer.fit(...)
```

### Stage 3 — 1024×1024 from 512 checkpoint

Same as Stage 2, using `train_1024.yaml` and the 512 checkpoint.

---

## Evaluation

FID is computed on the **validation set** (10k images) against 10k generated samples using pytorch-fid's InceptionV3.  
Real statistics are precomputed once at the start of training (`ValidFIDCache`).

```python
fid_cache = ValidFIDCache(valid_loader, device)
fid = fid_cache.compute(G, n_gen=10000, batch_size=16)
print(f'FID: {fid:.2f}')
```

FID is logged to WandB every `fid_interval` steps (default: 2000 for 512/1024, 5000 for 256).

---

## WandB Logging

Every `log_interval` steps (default: 100):

| Key | Description |
|---|---|
| `D/loss` | Discriminator logistic loss |
| `G/loss` | Generator logistic loss |
| `D/real_score` | Mean D output for real images |
| `D/fake_score` | Mean D output for generated images |
| `D/score_gap` | `D(real) - D(fake)` — primary training health indicator |
| `D/r1_penalty` | R1 gradient penalty (logged on reg steps) |
| `G/pl_penalty` | Path-length penalty (logged on reg steps) |
| `lr_g` / `lr_d` | Effective learning rates (constant; logged for config tracking) |
| `samples` | 4×4 grid of generated images (every `sample_interval` steps) |
| `FID` | Validation FID score (every `fid_interval` steps) |

---

## ONNX Export

```python
onnx_params = G.export_onnx('generator_1024.onnx', batch_size=1)
# Prints parameter count and asserts < 40M
```

The export uses a wrapper that bakes `noise_mode="none"` into the graph for a fully deterministic trace. The ONNX parameter count is verified automatically against the 40M limit.

---

## Key Implementation Notes

### Numerical stability (AMP / fp16)

| Issue | Location | Fix |
|---|---|---|
| Demodulation overflow | `ModulatedConv2d` | Cast `weight²` sum to fp32 before `rsqrt` |
| R1 double-backward in fp16 | `trainer.d_step` | Run R1 under `autocast(enabled=False)`, direct `backward()` |
| PL `sqrt(0)` backward | `g_path_length_loss` | `(grad².sum() + 1e-8).sqrt()` |

### Gradient clipping

`clip_grad_norm_(G.parameters(), max_norm=1.0)` is applied after every G update (both main loss and PL regularization).

### Lazy regularization

R1 runs every 16 D steps; PL runs every 4 G steps.  
Optimizer LR and β₂ are pre-scaled by `interval / (interval + 1)` to keep effective learning dynamics identical to per-step regularization.

---

## Dependencies

```
torch >= 2.2.0 (CUDA 12.1)
torchvision >= 0.17.0
wandb >= 0.17.0
pytorch-fid >= 0.3.0
onnx >= 1.16.0
pyyaml >= 6.0
pillow >= 10.0.0
scikit-image >= 0.21.0
```

All dependencies are pinned in `uv.lock`.
