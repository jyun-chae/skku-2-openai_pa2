"""FFHQ dataset loader.

Supports two directory layouts:

1. **Flat single directory** (all splits in one folder):
       <root>/00000.jpg … 69999.jpg
   Pass `split` to select the index range:
       train: 0 – 49999 | valid: 50000 – 59999 | test: 60000 – 69999

2. **Pre-split directories** (separate folder per split, 0-indexed):
       <train_root>/00000.jpg … 49999.jpg
       <valid_root>/00000.jpg …  9999.jpg
   Pass `split=None` to load ALL images in root (no index filtering).
   This is the mode used when train_50k_256.zip / valid_10k_256.zip are
   extracted into separate folders.

`aug` enables random horizontal flip (applied only for training).
Images are returned as float32 tensors in [-1, 1].
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


Split = Literal["train", "valid", "test"]

_SPLIT_RANGES: dict[str, tuple[int, int]] = {
    "train": (0, 50000),
    "valid": (50000, 60000),
    "test":  (60000, 70000),
}


class FFHQDataset(Dataset):
    """FFHQ face dataset.

    Args:
        root:       Directory containing extracted image files.
        split:      "train" | "valid" | "test" to filter by index range, or
                    None to load every image in root (use with pre-split dirs).
        resolution: Resize images to this square resolution.
        aug:        Apply random horizontal flip (train augmentation).
    """

    def __init__(
        self,
        root: str | Path,
        split: Optional[Split] = "train",
        resolution: int = 256,
        aug: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.resolution = resolution

        exts = {".png", ".jpg", ".jpeg"}
        all_files = sorted(
            f for f in self.root.iterdir()
            if f.suffix.lower() in exts
        )

        if split is None:
            # Pre-split directory: load everything, no index filter
            self.files = [f for f in all_files if _is_numeric(f)]
        else:
            lo, hi = _SPLIT_RANGES[split]
            self.files = [
                f for f in all_files
                if _is_numeric(f) and lo <= int(f.stem) < hi
            ]
            # Fallback: if nothing matched the range, the root is probably a
            # pre-split directory already (images numbered from 0).
            if len(self.files) == 0:
                self.files = [f for f in all_files if _is_numeric(f)]
                if self.files:
                    import warnings
                    warnings.warn(
                        f"No images matched split='{split}' index range "
                        f"({lo}–{hi-1}) in {self.root}. "
                        f"Loading all {len(self.files)} images instead "
                        "(assuming pre-split directory)."
                    )

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.root}.")

        is_train = (split == "train") or (split is None and aug)
        transform_list: list[object] = [
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.LANCZOS),
        ]
        if aug and is_train:
            transform_list.append(T.RandomHorizontalFlip())
        transform_list += [
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        self.transform = T.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


def _is_numeric(f: Path) -> bool:
    try:
        int(f.stem)
        return True
    except ValueError:
        return False


def build_dataloader(
    root: str | Path,
    split: Optional[Split],
    resolution: int,
    batch_size: int,
    num_workers: int = 4,
    aug: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    """Build a DataLoader for FFHQ.

    Pass split=None when root already contains only the images for one split
    (e.g. extracted from train_50k_256.zip into its own directory).
    """
    dataset = FFHQDataset(root, split, resolution, aug=aug)
    shuffle = aug and (split == "train" or split is None)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
        persistent_workers=(num_workers > 0 and persistent_workers),
    )
