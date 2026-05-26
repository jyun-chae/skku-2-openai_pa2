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

The three configs use matching low-resolution generator blocks, so each stage
can reuse the earlier stage's mapping network and synthesis blocks.  Newly
introduced high-resolution blocks start from random initialization.  The
discriminator is also partially loaded where tensor shapes match, but it is
expected to adapt more heavily at each new resolution.

The training loop keeps the original baseline flow where it is useful:

- zip-backed dataset loader
- DiffAugment `color,translation`
- non-saturating logistic GAN loss
- lazy R1 regularization
- EMA generator
- checkpoint/resume support
- WandB logging for losses, discriminator scores, image statistics, gradient
  norms, throughput, parameter counts, sample grids, and stage metadata

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
- mapping network: 4-layer MLP from `z` to `w`
- synthesis: learned 4x4 constant, styled convolutions with AdaIN, per-layer
  noise injection, skip ToRGB outputs
- output: `(B, 3, 1024, 1024)` in `[-1, 1]`
- parameter count: about 27.87M at the 1024 stage, under the 50M hard threshold

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
`train.py`, so it reconstructs the trained architecture automatically.

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
