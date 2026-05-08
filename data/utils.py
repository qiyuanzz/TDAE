from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch


def coords_to_grid(coords: torch.Tensor, patch_size: int = 224) -> torch.Tensor:
    """Convert level-0 pixel coordinates (x, y) to integer grid indices (row, col)."""
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must have shape [N, 2+] but got {tuple(coords.shape)}")
    xy = coords[:, :2].to(torch.long) // int(patch_size)
    return torch.stack((xy[:, 1], xy[:, 0]), dim=1)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {'.xls', '.xlsx'}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str], source: str | Path = 'table') -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f'{source} missing required columns: {missing}')


def normalize_slide_id(value: object) -> str:
    return str(value).strip()


def feature_bundle_paths(feature_root: str | Path, encoder_name: str, slide_id: str) -> dict[str, Path]:
    base = Path(feature_root) / encoder_name
    slide_id = normalize_slide_id(slide_id)
    return {
        'light': base / f'{slide_id}_light.pt',
        'medium': base / f'{slide_id}_medium.pt',
        'full': base / f'{slide_id}_full.pt',
        'coords': base / f'{slide_id}_coords.pt',
    }


def feature_triplet_paths(feature_root: str | Path, encoder_name: str, slide_id: str) -> dict[str, Path]:
    paths = feature_bundle_paths(feature_root, encoder_name, slide_id)
    return {'full': paths['full'], 'light': paths['light'], 'coords': paths['coords']}


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_feature_triplet(feature_root: str | Path, encoder_name: str, slide_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    paths = feature_triplet_paths(feature_root, encoder_name, slide_id)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing feature files for slide {slide_id}: {missing}')
    return (
        _load_tensor(paths['full']),
        _load_tensor(paths['light']),
        _load_tensor(paths['coords']).to(torch.long),
    )


def load_feature_bundle(
    feature_root: str | Path,
    encoder_name: str,
    slide_id: str,
    feature_keys: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    paths = feature_bundle_paths(feature_root, encoder_name, slide_id)
    keys = tuple(feature_keys) if feature_keys is not None else ('light', 'medium', 'full', 'coords')
    valid = {'light', 'medium', 'full', 'coords'}
    unknown = sorted(set(keys) - valid)
    if unknown:
        raise ValueError(f'Unknown feature keys for slide {slide_id}: {unknown}')
    required = {key: paths[key] for key in keys if key != 'medium'}
    if 'medium' in keys and not paths['medium'].exists():
        required['full'] = paths['full']
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing feature files for slide {slide_id}: {missing}')
    bundle: dict[str, torch.Tensor] = {}
    full: torch.Tensor | None = None
    if 'full' in keys or ('medium' in keys and not paths['medium'].exists()):
        full = _load_tensor(paths['full'])
        if 'full' in keys:
            bundle['full'] = full
    if 'light' in keys:
        bundle['light'] = _load_tensor(paths['light'])
    if 'medium' in keys:
        bundle['medium'] = _load_tensor(paths['medium']) if paths['medium'].exists() else full.clone()
    if 'coords' in keys:
        bundle['coords'] = _load_tensor(paths['coords']).to(torch.long)
    return bundle


def save_feature_triplet(
    feature_root: str | Path,
    encoder_name: str,
    slide_id: str,
    full_feats: torch.Tensor | None = None,
    light_feats: torch.Tensor | None = None,
    medium_feats: torch.Tensor | None = None,
    coords: torch.Tensor | None = None,
) -> dict[str, Path]:
    paths = feature_bundle_paths(feature_root, encoder_name, slide_id)
    paths['full'].parent.mkdir(parents=True, exist_ok=True)
    if full_feats is not None:
        torch.save(full_feats.cpu(), paths['full'])
    if light_feats is not None:
        torch.save(light_feats.cpu(), paths['light'])
    if medium_feats is not None:
        torch.save(medium_feats.cpu(), paths['medium'])
    if coords is not None:
        torch.save(coords.cpu().to(torch.long), paths['coords'])
    return paths
