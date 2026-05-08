from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import ensure_columns, load_feature_bundle, read_table


def _as_str_list(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None
    return [str(value) for value in values]


class FeatureDataset(Dataset):
    """One sample is one WSI with pre-extracted light/medium/full features."""

    def __init__(
        self,
        data_dir: str | Path,
        cohort_csv: str | Path,
        fold_csv: str | Path,
        encoder_name: str = 'uni2',
        split: str = 'train',
        task: str = 'classification',
        label_column: str = 'cancer_code',
        seed: int = 42,
        patch_drop_ratio: float = 0.0,
        patch_drop_seed: int | None = None,
        include_labels: Iterable[str] | None = None,
        label_aliases: dict[str, str] | None = None,
        feature_keys: Iterable[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.encoder_name = encoder_name
        self.task = str(task).lower()
        self.label_column = label_column
        self.patch_drop_ratio = float(patch_drop_ratio)
        if self.patch_drop_ratio < 0.0 or self.patch_drop_ratio >= 1.0:
            raise ValueError(f'patch_drop_ratio must be in [0, 1), got {self.patch_drop_ratio}')
        self.patch_drop_seed = int(seed if patch_drop_seed is None else patch_drop_seed)
        self.include_labels = _as_str_list(include_labels)
        self.label_aliases = {str(k): str(v) for k, v in (label_aliases or {}).items()}
        self.feature_keys = tuple(str(key) for key in feature_keys) if feature_keys is not None else ('light', 'medium', 'full', 'coords')
        self.generator = torch.Generator().manual_seed(seed)

        cohort = read_table(cohort_csv)
        folds = read_table(fold_csv)
        ensure_columns(cohort, ['case_submitter_id', 'slide_submitter_id'], 'cohort_csv')
        ensure_columns(folds, ['case_submitter_id', 'split'], 'fold_csv')

        fold_cases = set(folds['case_submitter_id'].astype(str))
        split_cases = set(folds.loc[folds['split'].astype(str) == split, 'case_submitter_id'].astype(str))
        fold_slides = cohort[cohort['case_submitter_id'].astype(str).isin(fold_cases)].copy()
        fold_slides = fold_slides.drop_duplicates('slide_submitter_id').reset_index(drop=True)

        if self.task == 'survival':
            ensure_columns(fold_slides, ['event', 'survival_days'], 'cohort_csv')
            fold_slides['event'] = pd.to_numeric(fold_slides['event'], errors='coerce')
            fold_slides['survival_days'] = pd.to_numeric(fold_slides['survival_days'], errors='coerce')
            fold_slides = fold_slides.dropna(subset=['event', 'survival_days'])
            fold_slides = fold_slides[fold_slides['survival_days'] > 0].copy()
            # NOTE: time_bin is no longer materialized in fold_csv. The trainer
            # fits cutpoints on the train fold uncensored cases at runtime
            # (MCAT/SurvPath convention) and discretizes train+val with the
            # same cutpoints to avoid leakage.
            self.class_names = ['risk']
            self.label_map: dict[str, int] = {}
            self.n_classes = 1
        elif self.task == 'classification':
            ensure_columns(fold_slides, [label_column], 'cohort_csv')
            fold_slides = fold_slides.dropna(subset=[label_column]).copy()
            fold_slides[label_column] = fold_slides[label_column].astype(str)
            if self.include_labels is not None:
                fold_slides = fold_slides[fold_slides[label_column].isin(self.include_labels)].copy()
            self.class_names = self._build_class_names(fold_slides[label_column])
            self.label_map = {name: idx for idx, name in enumerate(self.class_names)}
            self.n_classes = len(self.label_map)
            if self.n_classes < 2:
                raise ValueError(f'Classification task needs at least 2 classes after filtering; got {self.class_names}')
        else:
            raise ValueError(f'Unsupported task: {task}')

        slides = fold_slides[fold_slides['case_submitter_id'].astype(str).isin(split_cases)].copy()
        self.slides = slides.reset_index(drop=True)

    def _alias(self, raw_label: object) -> str:
        raw = str(raw_label)
        return self.label_aliases.get(raw, raw)

    def _build_class_names(self, labels: pd.Series) -> list[str]:
        if self.include_labels is not None:
            names = []
            present = set(labels.astype(str).tolist())
            for raw in self.include_labels:
                if raw in present:
                    names.append(self._alias(raw))
            return names
        return sorted({self._alias(label) for label in labels.astype(str).tolist()})

    def __len__(self) -> int:
        return len(self.slides)

    def _slide_seed(self, slide_id: str) -> int:
        payload = f'{self.patch_drop_seed}:{slide_id}'.encode('utf-8')
        return int.from_bytes(hashlib.sha1(payload).digest()[:8], 'little') % (2**63 - 1)

    def _sample_patches(self, bundle: dict[str, torch.Tensor], slide_id: str) -> dict[str, torch.Tensor]:
        n = int(bundle['full'].shape[0])
        keep_n = n
        if self.patch_drop_ratio > 0.0:
            keep_n = min(keep_n, max(1, int(math.ceil(n * (1.0 - self.patch_drop_ratio)))))
        if keep_n >= n:
            return bundle
        if self.patch_drop_ratio > 0.0:
            generator = torch.Generator().manual_seed(self._slide_seed(slide_id))
        else:
            generator = self.generator
        indices = torch.randperm(n, generator=generator)[:keep_n].sort().values
        return {key: value[indices] for key, value in bundle.items()}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.slides.iloc[idx]
        slide_id = str(row['slide_submitter_id'])
        bundle = self._sample_patches(load_feature_bundle(self.data_dir, self.encoder_name, slide_id, self.feature_keys), slide_id)

        raw_label = None
        label_name = None
        if self.task == 'classification':
            raw_label = str(row[self.label_column])
            label_name = self._alias(raw_label)
            label = torch.tensor(self.label_map[label_name], dtype=torch.long)
        else:
            event = float(row['event'])
            label = {
                'event': torch.tensor(event, dtype=torch.float32),
                'time': torch.tensor(float(row['survival_days']), dtype=torch.float32),
                'censorship': torch.tensor(1.0 - event, dtype=torch.float32),
            }

        sample = {
            'label': label,
            'label_name': label_name,
            'raw_label': raw_label,
            'slide_id': slide_id,
            'case_id': str(row['case_submitter_id']),
            'file_path': str(row['file_path']) if 'file_path' in row else '',
        }
        if 'full' in bundle:
            sample['full_feats'] = bundle['full'].float()
        if 'medium' in bundle:
            sample['medium_feats'] = bundle['medium'].float()
        if 'light' in bundle:
            sample['light_feats'] = bundle['light'].float()
        if 'coords' in bundle:
            sample['grid_indices'] = bundle['coords'].long()
        return sample
