# Project 2 FFHQ StyleGAN

Self-contained PyTorch implementation for Project 2 image generation.  The
default model is a StyleGAN2-inspired GAN trained from noise to 1024x1024 RGB
faces.

## Requirements

```bash
pip install -r requirements.txt
```

Allowed project libraries are kept to PyTorch/TorchVision, Pillow, NumPy,
PyYAML, WandB, and ONNX tooling.  The code does not use HuggingFace libraries,
PyTorch Lightning, Accelerate, Albumentations, external model repositories, or
pretrained models.

## Data

Put the announced training split zip at:

```text
data/train_50k_1024.zip
```

Do not train on the valid or test split.  The zip should contain 1024x1024 JPG
or PNG images.

## Progressive Training

```bash
python train.py --config configs/stylegan_256.yaml
python train.py --config configs/stylegan_512.yaml --init-from runs/stylegan_256/final.pt
python train.py --config configs/stylegan_1024.yaml --init-from runs/stylegan_512/final.pt
```

On Colab, use `main.ipynb`. It mounts Google Drive, installs dependencies,
logs into W&B with an entered API key, and saves checkpoints under
`/content/drive/MyDrive/project02/runs/`.

For Colab, run training in shorter chunks and resume from `latest.pt`:

```bash
python train.py --config configs/stylegan_256.yaml --max-steps 2000
python train.py --config configs/stylegan_256.yaml --resume runs/stylegan_256/latest.pt --max-steps 2000
```

Every `--resume` invocation starts a fresh W&B run by default, so each resumed
chunk gets its own graphs. Pass `--resume-wandb-run` only if you intentionally
want to continue logging into the W&B run id saved in the checkpoint.

Each stage config saves both image-count checkpoints and step-count
checkpoints.  By default it writes:

- `latest.pt` every `ckpt_every_steps`
- `ckpt_step_XXXXXXXX.pt` every `ckpt_every_steps`
- `ckpt_XXXXXXXXX.pt` every `ckpt_every` images
- `final.pt` only when the configured `total_images` target is reached

You can override the step save interval from the command line:

```bash
python train.py --config configs/stylegan_256.yaml --max-steps 2000 --save-every-steps 250
```

The default stage configs are tuned for an A100-class Colab runtime with BF16:

```text
256: batch 64, workers 8
512: batch 24, workers 8
1024: batch 8, workers 6
```

On smaller Colab GPUs, lower the batch size from the config default:

```bash
python train.py --config configs/stylegan_256.yaml --batch-size 8 --num-workers 2 --max-steps 2000
```

The three configs instantiate the same full 1024 generator/discriminator and
train only the active prefix/suffix for the selected stage resolution.  This
keeps parameter names stable across 256 -> 512 -> 1024 training while leaving
newly activated high-resolution blocks at their saved initialization until
their stage begins.

The training loop uses a compact StyleGAN2-style recipe:

- zip-backed dataset loader
- horizontal flip by default; DiffAugment `color`/`translation` remains optional
- non-saturating logistic GAN loss
- lazy R1 regularization
- lazy path length regularization
- StyleGAN2-style style mixing
- EMA generator
- checkpoint/resume support
- WandB logging for losses, discriminator scores, score gaps, FID, image
  statistics, gradient norms, throughput, parameter counts, sample grids, and
  stage metadata

WandB is configured separately in each stage config:

```yaml
wandb:
  project: ffhqgen-student
  name: stylegan_256
  mode: online
```

Set `mode: offline` or `mode: disabled` if needed.

## Model

`src/model.py` defines the generator, discriminator, and EMA.

Generator:

- input: `(B, 512)` Gaussian noise
- mapping network: 8-layer MLP from `z` to `w`
- synthesis: learned 4x4 constant, per-layer `w+` styles, styled convolutions
  with AdaIN, per-layer noise injection, skip ToRGB outputs
- output: `(B, 3, 1024, 1024)` in `[-1, 1]`
- parameter count: about 39.87M at the 1024 stage, under the 40M hard threshold

Discriminator:

- StyleGAN2-like residual downsampling blocks
- spectral normalization
- minibatch standard deviation
- trained jointly with G

## Generate

```bash
python generate.py --ckpt runs/stylegan_1024/final.pt --out samples/grid.png --n 16
```

`generate.py` reads `meta.generator_config` from checkpoints saved by
`train.py`, so it reconstructs the trained architecture automatically. Use
`--truncation-psi 0.7` or another value below 1.0 to apply inference-only
truncation.

## Export ONNX

```bash
python export_onnx.py --ckpt runs/stylegan_1024/final.pt --out submission.onnx
```

The exported interface is:

```text
input  z      (B, 512), float32
output image  (B, 3, 1024, 1024), float32, range [-1, 1]
```

## Resume

```bash
python train.py --config configs/stylegan_1024.yaml \
  --resume runs/stylegan_1024/ckpt_000100000.pt
```
