from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class EncoderSpec:
    name: str
    d_full: int
    model_path: str | None = None


ENCODER_SPECS = {
    "ctranspath": EncoderSpec("ctranspath", d_full=768, model_path="/mnt/Xsky/public_model_zoo/CHIEF/model_weight/CHIEF_CTransPath.pth"),
    "conch_v15": EncoderSpec("conch_v15", d_full=768, model_path="/mnt/Xsky/models/PFM/conchv1_5"),
    "uni2": EncoderSpec("uni2", d_full=1536, model_path="/mnt/Xsky/models/PFM/UNI2-h"),
    "virchow2": EncoderSpec("virchow2", d_full=1280, model_path="/mnt/Xsky/models/PFM/Virchow2"),
}


class IdentityEncoder(nn.Module):
    """Small deterministic encoder used for smoke tests and fixture generation."""

    def __init__(self, d_out: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(3, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.float().mean(dim=(-1, -2))
        return self.proj(pooled)


def get_encoder_spec(name: str) -> EncoderSpec:
    key = name.lower()
    if key not in ENCODER_SPECS:
        raise ValueError(f"Unsupported encoder {name}. Choices: {sorted(ENCODER_SPECS)}")
    return ENCODER_SPECS[key]


def build_encoder(name: str, mode: str = "full", truncate_layer: int = 3, smoke_dim: int = 16) -> nn.Module:
    """Build encoder by name.

    Real UNI2/Virchow/CONCH loading is site-specific; this project keeps extraction
    script hooks explicit and supports smoke-test identity encoding out of the box.
    """
    if name.lower() in {"identity", "smoke"}:
        return IdentityEncoder(d_out=smoke_dim)
    raise RuntimeError(
        "Direct encoder construction is not configured in this standalone project. "
        "Use pre-extracted features from pancancer or adapt build_encoder() with the local encoder loader."
    )


def encode_batches(
    encoder: nn.Module,
    batches,
    device: torch.device,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    encoder.eval().to(device)
    features = []
    with torch.no_grad():
        for batch in batches:
            x = batch.to(device)
            if transform is not None:
                x = transform(x)
            features.append(encoder(x).detach().cpu())
    return torch.cat(features, dim=0)
