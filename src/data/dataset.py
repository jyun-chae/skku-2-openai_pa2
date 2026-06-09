"""FFHQ dataset loader.

Two directory layouts are supported:
  flat dir   — pass split='train'/'valid'/'test' to filter by numeric filename range
               (train: 0–49999, valid: 50000–59999, test: 60000–69999)
  split dir  — pass split=None to load all images in root (train_50k_{res}.zip layout)

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
    """FFHQ face dataset with support for flat and pre-split directories.

    Args:
        root:       Directory containing extracted image files.
        split:      "train" | "valid" | "test" to filter by filename index, or
                    None to load every image (use with pre-split directories).
        resolution: Resize images to this square resolution.
        aug:        Apply random horizontal flip (training augmentation).
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
            self.files = [f for f in all_files if _is_numeric(f)]
        else:
            lo, hi = _SPLIT_RANGES[split]
            self.files = [
                f for f in all_files
                if _is_numeric(f) and lo <= int(f.stem) < hi
            ]
            if len(self.files) == 0:
                # Root is likely pre-split (filenames start from 0, not FFHQ global indices).
                self.files = [f for f in all_files if _is_numeric(f)]
                if self.files:
                    import warnings
                    warnings.warn(
                        f"No images matched split='{split}' range ({lo}–{hi-1}) in {self.root}. "
                        f"Loading all {len(self.files)} images (assuming pre-split directory)."
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
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [0,1] → [-1,1] to match tanh output
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
        drop_last=shuffle,  # keep batch size constant during training
        persistent_workers=(num_workers > 0 and persistent_workers),
    )
