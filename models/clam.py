from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attn_Net_Gated


class CLAM(nn.Module):
    """CLAM-SB bag classifier."""

    SIZES = {
        'small': (512, 256),
        'big': (512, 384),
    }

    def __init__(
        self,
        d_in: int,
        n_classes: int = 4,
        size: str = 'small',
        dropout: float = 0.25,
        gate: bool = True,
    ) -> None:
        super().__init__()
        d_hidden, d_attn = self.SIZES[size]
        fc_layers: list[nn.Module] = [nn.Linear(d_in, d_hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc_layers)
        if gate:
            self.attention_net = Attn_Net_Gated(L=d_hidden, D=d_attn, dropout=dropout, n_classes=1)
        else:
            self.attention_net = nn.Sequential(
                nn.Linear(d_hidden, d_attn),
                nn.Tanh(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(d_attn, 1),
            )
        self.classifier = nn.Linear(d_hidden, n_classes)
        self._gated = gate

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.dim() != 2:
            raise ValueError(f'CLAM expects features [N, d_in], got {tuple(features.shape)}')
        h = self.fc(features.float())
        if self._gated:
            attn_logits, h = self.attention_net(h)
        else:
            attn_logits = self.attention_net(h)
        attn = F.softmax(attn_logits, dim=0)
        bag = (attn * h).sum(dim=0)
        logits = self.classifier(bag)
        return logits, attn.squeeze(-1)
