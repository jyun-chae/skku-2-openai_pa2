# Project 02 — StyleGAN2 Face Generation (FFHQ 1024×1024)

StyleGAN2 기반 무조건부 얼굴 생성기. FFHQ 데이터셋으로 256 → 512 → 1024 해상도 순서의 **프로그레시브 트레이닝**으로 학습합니다.

- **Generator**: ~34.10M PyTorch 파라미터, ~35.5M ONNX initializer (50M 제출 한도 내)
- **입력**: `z ~ N(0, I)`, shape `(N, 512)`, dtype `float32`
- **출력**: `(N, 3, 1024, 1024)`, dtype `float32`, range `[-1, 1]`

---

## 아키텍처

### Generator

| 컴포넌트 | 상세 |
|---|---|
| 입력 | `z`, dim = 512 |
| Mapping Network | 12-layer EqualLinear MLP, `w_dim=640`, `lr_mul=0.01` |
| Synthesis | 9블록 (4×4 → 1024×1024), ModulatedConv2d + **SqueezeConnection** |
| Channel schedule | `channel_base=131072`, `channel_max=512` → 256×256까지 512채널 |
| 출력 | 3 × 1024 × 1024, tanh로 [-1, 1] 클램핑 |

#### 채널 폭 (해상도별)

| 해상도 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|
| 채널 | 512 | 512 | 512 | 512 | 512 | 512 | 512 | 256 | 128 |

### SqueezeConnection (StyleGAN-Small, arXiv:2407.05527)

기존 StyleGAN2의 `conv2 + ToRGB` 구조를 **SqueezeConnection**으로 교체합니다.
ToRGB 정보 병목을 완화하면서 파라미터를 줄입니다.

```
conv1 (ModulatedConv2d, upsample) → noise1 → act
└─ SqueezeConnection:
     squeeze  (EqualConv2d, c → c/8, k=3)
     noise    (NoiseInjection)          ← conv2 stochasticity 대체
     to_rgb   (ModulatedConv2d 1×1)    → rgb_current
     excite   (EqualConv2d, c/8 → c, k=3)
     project  (EqualConv2d, 2c → c, k=1) → x_out (다음 블록 입력)
```

### Discriminator

| 컴포넌트 | 상세 |
|---|---|
| FromRGB | EqualConv2d 1×1 + LeakyReLU |
| 블록 | DiscBlock (잔차 구조, avg pool downscale) × (log2(res) - 2)개 |
| MinibatchStddev | group_size=4, num_features=1 |
| 출력 헤드 | EqualConv2d 3×3 → Flatten → EqualLinear × 2 |

채널 스케줄은 Generator와 동일 (`channel_base=131072`, `channel_max=512`).

---

## 학습 손실 함수

| 손실 | 수식 | 주기 |
|---|---|---|
| D logistic | `softplus(−D(real)) + softplus(D(fake))` | 매 스텝 |
| R1 penalty (lazy) | `γ/2 · E[‖∇D(real)‖²]` | D 16스텝마다 |
| G logistic | `softplus(−D(fake))` | 매 스텝 |
| Path-length reg (lazy) | `(‖J_w^T v‖ − EMA)²` | G 4스텝마다 |

**Lazy regularization**: R1은 16 D스텝마다, PL은 4 G스텝마다 실행합니다.
옵티마이저 LR과 β₂는 `interval / (interval + 1)` 비율로 보정해 effective dynamics를 유지합니다.

### AMP / fp16 수치 안정성

| 문제 | 위치 | 해결책 |
|---|---|---|
| Demodulation overflow | `ModulatedConv2d` | weight² 합산을 fp32로 캐스팅 후 `rsqrt` |
| R1 double-backward in fp16 | `trainer.d_step` | `autocast(enabled=False)` 하에 R1 실행 |
| PL `sqrt(0)` backward | `g_path_length_loss` | `(grad².sum() + 1e-8).sqrt()` |

---

## 프로그레시브 트레이닝

```
256×256  →  512×512  →  1024×1024
 Stage 1     Stage 2     Stage 3
```

각 스테이지는 이전 체크포인트를 로드합니다.  
새 고해상도 블록은 랜덤 초기화되며, 기존 블록은 전체 학습됩니다(동결 없음).

### 스테이지별 학습 설정

| 스테이지 | 해상도 | Batch | `lr_g` | `lr_d` | `r1_gamma` | `total_kimg` |
|---|---|---|---|---|---|---|
| 1 | 256 | 16 | 1e-3 | 1e-3 | 10.0 | 25,000 |
| 2 | 512 | 8 | 1e-3 | 1e-3 | 15.0 | 10,000 |
| 3 | 1024 | 4 | 1e-3 | 5e-4 | 30.0 | 10,000 |

공통: `grad_clip=1.0`, `use_amp=true`, `pl_weight=2.0`, `pl_decay=0.01`, `pl_warmup_steps=500~1000`

---

## 프로젝트 구조

```
project02/
├── src/
│   ├── models/
│   │   ├── layers.py          # EqualLinear, EqualConv2d, ModulatedConv2d,
│   │   │                      # NoiseInjection, ToRGB, SqueezeConnection
│   │   ├── generator.py       # MappingNetwork, SynthesisBlock4/Block, StyleGAN2Generator
│   │   └── discriminator.py   # FromRGB, DiscBlock, MinibatchStddev, StyleGAN2Discriminator
│   ├── data/
│   │   └── dataset.py         # FFHQDataset (pre-split 또는 flat 디렉토리), build_dataloader
│   ├── training/
│   │   ├── losses.py          # d_logistic, d_r1, g_nonsaturating, g_path_length
│   │   └── trainer.py         # Trainer (AMP, lazy reg, grad clip, WandB, Drive 백업)
│   └── utils/
│       ├── fid_score.py       # ValidFIDCache — 실제 통계 1회 사전 계산
│       └── parallel_unzip.py  # 병렬 ZIP 압축 해제
├── configs/
│   ├── train_256.yaml         # Stage 1 설정
│   ├── train_512.yaml         # Stage 2 설정
│   └── train_1024.yaml        # Stage 3 설정
├── main.ipynb                 # Colab 학습 노트북 (전체 파이프라인)
├── submit_test.ipynb          # ONNX 파라미터 수 검증 노트북
├── export_onnx.py             # 학습된 체크포인트 → ONNX 변환 스크립트
├── verify_onnx.py             # onnxruntime으로 ONNX 입출력 검증
└── pyproject.toml
```

---

## 환경 설정

### 로컬 개발 (uv)

```bash
git clone https://github.com/jyun-chae/skku-2-openai_pa2.git
cd skku-2-openai_pa2
uv sync
uv run python -c "from src.models.generator import StyleGAN2Generator; print('OK')"
```

### Colab 학습

`main.ipynb`를 열고 위에서부터 순서대로 실행합니다.

**필수 설정값 (각 셀 상단 `USER CONFIG` 섹션)**:

```python
REPO_URL      = 'https://github.com/jyun-chae/skku-2-openai_pa2.git'
WANDB_API_KEY = 'YOUR_WANDB_API_KEY_HERE'
DRIVE_DIR     = '/content/drive/MyDrive/project02'
```

#### Colab 실행 흐름

1. 패키지 설치 + 레포 클론 (이미 클론된 경우 `git pull`)
2. Google Drive 마운트 → 체크포인트 백업 디렉토리 확인
3. 데이터 설정: Drive → `/content/` 복사 → chunk 병합 → ZIP 압축 해제
4. WandB 로그인
5. Stage 1 (256) 학습 → Stage 2 (512) → Stage 3 (1024)
6. FID 평가 (검증셋 10k vs 생성 10k)
7. 샘플 이미지 생성 (truncation=0.7)
8. ONNX 내보내기 → Drive 저장

---

## 데이터 준비

데이터는 Google Drive의 `project02/data/` 폴더에 청크 파일로 업로드합니다.

| 해상도 | train 청크 | valid 청크 |
|---|---|---|
| 256 | `train_50k_256.zip.000~001` (2개) | `valid_10k_256.zip` (단일) |
| 512 | `train_50k_512.zip.000~005` (6개) | `valid_10k_512.zip.000~001` (2개) |
| 1024 | `train_50k_1024.zip.000~019` (20개) | `valid_10k_1024.zip.000~003` (4개) |

`split_manifest.json`도 함께 업로드해야 각 청크의 크기 무결성 검증이 동작합니다.

---

## 학습 방법

### Stage 1 — 256×256 (처음부터 학습)

```python
cfg = load_cfg('configs/train_256.yaml')
trainer = Trainer(cfg)
trainer.fit(train_loader, valid_loader, wandb_run=run,
            drive_backup_dir=DRIVE_CKPT_DIR, ckpt_dir=CKPT_DIR)
```

### Stage 2 — 512×512 (256 체크포인트에서 파인튜닝)

```python
cfg = load_cfg('configs/train_512.yaml')
trainer = Trainer(cfg)
state = torch.load('ckpt_256_final.pth')
trainer.G.load_from_lower_resolution(state['G'])
trainer.D.load_from_lower_resolution(state['D'])
trainer.fit(...)
```

`load_from_lower_resolution`: 이전 해상도 체크포인트에서 shape이 맞는 가중치만 로드하고,
새 고해상도 블록(Generator: 추가 SynthesisBlock, Discriminator: from_rgb + blocks[0])은 랜덤 초기화됩니다.

### Stage 3 — 1024×1024 (512 체크포인트에서 파인튜닝)

Stage 2와 동일, `train_1024.yaml`과 512 체크포인트 사용.

### 체크포인트 복구 (Colab 세션 재시작 시)

```python
restore_from_drive(trainer, DRIVE_CKPT_DIR, CKPT_DIR, resolution=1024)
```

Drive에서 해당 해상도의 가장 최신 체크포인트를 로컬로 복사 후 로드합니다.
체크포인트가 없으면 `False`를 반환하고 이전 해상도 체크포인트에서 시작합니다.

---

## 평가 (FID)

검증셋 10k 이미지 vs 생성 이미지 10k 간 FID를 pytorch-fid의 InceptionV3(2048-d 피처)로 계산합니다.
실제 통계(μ, Σ)는 세션당 1회만 계산해 캐시합니다.

```python
fid_cache = ValidFIDCache(valid_loader, device)
fid = fid_cache.compute(G, n_gen=10000, batch_size=16)
print(f'FID: {fid:.2f}')
```

FID는 `fid_interval`마다(기본 2000스텝) 자동 계산되어 WandB에 로깅됩니다.

---

## ONNX 내보내기 및 검증

### 내보내기

```python
# Generator 인스턴스에서 직접 내보내기
onnx_params = G.export_onnx('generator_1024.onnx', batch_size=1)
```

- `noise_mode="none"` 고정 → 노이즈 텐서가 상수로 접혀서 결정론적 그래프 생성
- `do_constant_folding=True`, opset 17
- 내보내기 전 CPU로 이동 (CUDA에서 constant folding 호환 문제 방지)

학습된 체크포인트(`ckpt_1024_*.pth`)에서 직접 내보내려면:

```bash
python export_onnx.py
```

> **참고**: `export_onnx.py`는 기존 레거시 체크포인트(표준 StyleGAN2, `channel_base=65536`)용입니다.
> SqueezeConnection 아키텍처로 학습한 체크포인트는 `G.export_onnx()` 또는 `submit_test.ipynb`를 사용합니다.

### 검증

```bash
python verify_onnx.py
```

onnxruntime으로 배치 크기 1, 2에서 출력 shape `(B, 3, 1024, 1024)` 및 범위 `[-1, 1]`을 확인합니다.

### 제출 사양

| 항목 | 값 |
|---|---|
| 입력 텐서 이름 | `z` |
| 입력 shape | `(N, 512)`, `float32` |
| 출력 shape | `(N, 3, 1024, 1024)`, `float32` |
| 출력 범위 | `[-1, 1]` |
| PyTorch 파라미터 | ~34.10M |
| ONNX initializer 수 | ~35.5M |
| ONNX 파라미터 한도 | 50M |

> ONNX count가 PyTorch count보다 ~1.4M 많은 이유:
> `do_constant_folding=True`가 EqualLinear의 `weight × scale` 곱을 별도 initializer로 상수 접기하기 때문입니다.

파라미터 수를 독립적으로 검증하려면 `submit_test.ipynb`를 실행하세요.
(무작위 가중치로 1024 Generator 생성 → ONNX 내보내기 → PyTorch 파라미터 수 vs ONNX initializer 수 비교)

---

## 의존성

```
torch >= 2.2.0
torchvision >= 0.17.0
wandb >= 0.17.0
pytorch-fid >= 0.3.0
onnx >= 1.16.0
onnxruntime >= 1.18.0   # verify_onnx.py 실행 시
pyyaml >= 6.0
pillow >= 10.0.0
numpy >= 1.24.0
matplotlib >= 3.7.0
scikit-image >= 0.21.0
```

로컬: `uv sync` (uv.lock 기반 재현 가능한 환경)
Colab: `pip install wandb pytorch-fid pyyaml scikit-image onnx` (main.ipynb 1번 셀 자동 설치)

---

## 참고 논문

- [StyleGAN2](https://arxiv.org/abs/1912.04958) — Analyzing and Improving the Image Quality of StyleGAN
- [StyleGAN-Small](https://arxiv.org/abs/2407.05527) — SqueezeConnection 구조 출처
- [StyleGAN3](https://arxiv.org/abs/2106.12423) — Alias-Free Generative Adversarial Networks
- [StyleGAN-XL](https://arxiv.org/abs/2202.00273) — Scaling StyleGAN to Large Diverse Datasets
- [StyleSwin](https://arxiv.org/abs/2112.10762) — StyleSwin: Transformer-based GAN for High-resolution Image Generation
