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
        'full': base / 'features' / f'{slide_id}.pt',
        'coords': base / 'coords' / f'{slide_id}.pt',
    }


def feature_triplet_paths(feature_root: str | Path, encoder_name: str, slide_id: str) -> dict[str, Path]:
    paths = feature_bundle_paths(feature_root, encoder_name, slide_id)
    return {'full': paths['full'], 'coords': paths['coords']}


def _legacy_feature_bundle_paths(feature_root: str | Path, encoder_name: str, slide_id: str) -> dict[str, Path]:
    base = Path(feature_root) / encoder_name
    slide_id = normalize_slide_id(slide_id)
    return {
        'full': base / f'{slide_id}_full.pt',
        'coords': base / f'{slide_id}_coords.pt',
    }


def _feature_path_candidates(feature_root: str | Path, encoder_name: str, slide_id: str, key: str) -> list[Path]:
    canonical = feature_bundle_paths(feature_root, encoder_name, slide_id)[key]
    legacy = _legacy_feature_bundle_paths(feature_root, encoder_name, slide_id)[key]
    return [canonical] if canonical == legacy else [canonical, legacy]


def _resolve_feature_path(feature_root: str | Path, encoder_name: str, slide_id: str, key: str) -> Path:
    candidates = _feature_path_candidates(feature_root, encoder_name, slide_id, key)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_feature_triplet(feature_root: str | Path, encoder_name: str, slide_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    paths = {
        'full': _resolve_feature_path(feature_root, encoder_name, slide_id, 'full'),
        'coords': _resolve_feature_path(feature_root, encoder_name, slide_id, 'coords'),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing feature files for slide {slide_id}: {missing}')
    return (
        _load_tensor(paths['full']),
        _load_tensor(paths['coords']).to(torch.long),
    )


def load_feature_bundle(
    feature_root: str | Path,
    encoder_name: str,
    slide_id: str,
    feature_keys: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    keys = tuple(feature_keys) if feature_keys is not None else ('full', 'coords')
    valid = {'full', 'coords'}
    unknown = sorted(set(keys) - valid)
    if unknown:
        raise ValueError(f'Unknown feature keys for slide {slide_id}: {unknown}')
    paths = {key: _resolve_feature_path(feature_root, encoder_name, slide_id, key) for key in valid}
    required = {key: paths[key] for key in keys}
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing feature files for slide {slide_id}: {missing}')
    bundle: dict[str, torch.Tensor] = {}
    if 'full' in keys:
        bundle['full'] = _load_tensor(paths['full'])
    if 'coords' in keys:
        bundle['coords'] = _load_tensor(paths['coords']).to(torch.long)
    return bundle


def save_feature_triplet(
    feature_root: str | Path,
    encoder_name: str,
    slide_id: str,
    full_feats: torch.Tensor | None = None,
    coords: torch.Tensor | None = None,
) -> dict[str, Path]:
    paths = feature_bundle_paths(feature_root, encoder_name, slide_id)
    if full_feats is not None:
        paths['full'].parent.mkdir(parents=True, exist_ok=True)
        torch.save(full_feats.cpu(), paths['full'])
    if coords is not None:
        paths['coords'].parent.mkdir(parents=True, exist_ok=True)
        torch.save(coords.cpu().to(torch.long), paths['coords'])
    return paths
