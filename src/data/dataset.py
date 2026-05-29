"""FFHQ dataset loader.

Expected directory layout after extraction:
    <root>/
        00000.png  (or .jpg)
        00001.png
        ...
        69999.png

Split by index:
    train: 00000 – 49999   (50k)
    valid: 50000 – 59999   (10k)
    test:  60000 – 69999   (10k)

Images are returned as tensors in [-1, 1].
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

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
        split:      "train", "valid", or "test".
        resolution: Resize images to this square resolution.
        aug:        Apply horizontal flip augmentation (train only).
    """

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        resolution: int = 256,
        aug: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.resolution = resolution

        lo, hi = _SPLIT_RANGES[split]
        # Collect all image files and filter to the correct index range
        exts = {".png", ".jpg", ".jpeg"}
        all_files = sorted(
            f for f in self.root.iterdir()
            if f.suffix.lower() in exts
        )
        # Filter by numeric index in filename (handles both 00000.png and 00000.jpg)
        self.files: list[Path] = []
        for f in all_files:
            try:
                idx = int(f.stem)
            except ValueError:
                continue
            if lo <= idx < hi:
                self.files.append(f)

        if len(self.files) == 0:
            raise RuntimeError(
                f"No images found in {self.root} for split '{split}' "
                f"(expected indices {lo}–{hi - 1})."
            )

        transform_list: list[object] = [
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.LANCZOS),
        ]
        if aug and split == "train":
            transform_list.append(T.RandomHorizontalFlip())
        transform_list += [
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
        ]
        self.transform = T.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


def build_dataloader(
    root: str | Path,
    split: Split,
    resolution: int,
    batch_size: int,
    num_workers: int = 4,
    aug: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    dataset = FFHQDataset(root, split, resolution, aug=aug)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0 and persistent_workers),
    )
