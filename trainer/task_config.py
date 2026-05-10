from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER_ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURE_BANK_ROOT = Path('/mnt/Xsky/zqb/TCGA_WSI_feature_bank')


ENCODER_ALIASES = {
    'uni2': 'uni2h',
    'uni2h': 'uni2h',
    'uni_v2': 'uni2h',
    'conch_v15': 'conch_v15',
    'conchv1_5': 'conch_v15',
    'conch': 'conch_v15',
    'virchow2': 'virchow2',
    'ctranspath': 'ctranspath',
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def _config_root(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else PROJECT_ROOT
    candidates = [
        base / 'trainer' / 'configs',
        base / 'configs',
        TRAINER_ROOT / 'configs',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base / 'trainer' / 'configs'


def dataset_config_path(cohort: str, root: str | Path | None = None) -> Path:
    return _config_root(root) / 'dataset' / f'{cohort.lower()}.yaml'


def load_dataset_config(cohort: str, task: str | None = None, root: str | Path | None = None) -> dict[str, Any]:
    path = dataset_config_path(cohort, root=root)
    cfg = load_yaml(path)
    if task is None:
        return cfg
    task_cfg = cfg.get('tasks', {}).get(task, {})
    if not task_cfg:
        raise ValueError(f'{cohort} has no task config for {task!r}: {path}')
    if task_cfg.get('enabled') is False:
        raise ValueError(f'{cohort} does not enable task {task!r}; check {path}')
    merged = {key: value for key, value in cfg.items() if key != 'tasks'}
    merged.update(task_cfg)
    merged['task'] = task
    return merged


def encoder_config_path(encoder: str, root: str | Path | None = None) -> Path:
    key = ENCODER_ALIASES.get(str(encoder).lower(), str(encoder).lower())
    return _config_root(root) / 'encoder' / f'{key}.yaml'


def load_encoder_config(encoder: str, root: str | Path | None = None) -> dict[str, Any]:
    path = encoder_config_path(encoder, root=root)
    cfg = load_yaml(path)
    cfg.setdefault('encoder_config_name', path.stem)
    return cfg


def merge_config(default_cfg: dict[str, Any], dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = default_cfg.copy()
    merged.update({key: value for key, value in dataset_cfg.items() if value is not None})
    return merged


def resolve_feature_dir(cfg: dict[str, Any], cohort: str) -> str:
    cohort_name = str(cfg.get('cohort', cohort))
    feature_dir = cfg.get('feature_dir')
    if feature_dir:
        resolved = str(feature_dir).format(cohort=cohort)
        parts = {part.lower() for part in Path(resolved).parts}
        if str(cohort).lower() in parts or cohort_name.lower() in parts:
            return resolved
    feature_bank_root = Path(cfg.get('feature_bank_root', DEFAULT_FEATURE_BANK_ROOT))
    return str(feature_bank_root / cohort_name / 'features')
