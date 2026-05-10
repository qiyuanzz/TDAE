from __future__ import annotations

import torch.nn as nn

from .abmil import ABMIL
from .clam import CLAM
from .transmil import TransMIL


def build_aggregator(name: str, d_in: int, n_classes: int, d_hidden: int = 512) -> nn.Module:
    name = name.lower()
    if name == 'abmil':
        return ABMIL(d_in=d_in, d_hidden=d_hidden, n_classes=n_classes)
    if name == 'transmil':
        return TransMIL(d_in=d_in, n_classes=n_classes, d_hidden=d_hidden)
    if name == 'clam' or name == 'clam_sb':
        return CLAM(d_in=d_in, n_classes=n_classes)
    raise ValueError(f'Unsupported aggregator: {name}')
