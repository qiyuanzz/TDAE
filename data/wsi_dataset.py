from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class PatchCoordinateDataset(Dataset):
    """Dataset of patch coordinates for one WSI; image reading is handled by extractor scripts."""

    def __init__(self, coords: torch.Tensor, slide_path: str | Path, patch_size: int = 224) -> None:
        self.coords = coords.long()
        self.slide_path = Path(slide_path)
        self.patch_size = int(patch_size)

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        xy = self.coords[idx]
        return {"coord": xy, "slide_path": str(self.slide_path), "patch_size": self.patch_size}

