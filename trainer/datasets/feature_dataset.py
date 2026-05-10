from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..io import ensure_columns, load_feature_bundle, read_table


def _as_str_list(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None
    return [str(value) for value in values]


def build_time_bin_cutpoints(labels: list[dict[str, torch.Tensor]], n_bins: int) -> torch.Tensor:
    if n_bins < 2:
        raise ValueError(f'n_bins must be >= 2 for survival time discretization, got {n_bins}')
    times = torch.stack([label['time'].detach().float().cpu().view(()) for label in labels])
    events = torch.stack([label['event'].detach().float().cpu().view(()) for label in labels])
    if times.numel() == 0:
        raise ValueError('Cannot build survival time bins from an empty cohort.')
    source_times = times[events > 0]
    if source_times.numel() == 0:
        source_times = times
    try:
        if source_times.numel() < n_bins:
            raise ValueError('Too few uncensored times for qcut.')
        _, bin_edges = pd.qcut(source_times.numpy(), q=n_bins, labels=False, retbins=True)
        cutpoints = torch.tensor(bin_edges[1:-1], dtype=torch.float32)
    except ValueError:
        quantiles = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float32)[1:-1]
        cutpoints = torch.quantile(source_times.float(), quantiles)
    if cutpoints.numel() != n_bins - 1:
        raise ValueError(f'Expected {n_bins - 1} cutpoints for {n_bins} bins, got {cutpoints.numel()}.')
    return cutpoints.float()


def discretize_survival_times(times: torch.Tensor, cutpoints: torch.Tensor, n_bins: int) -> torch.Tensor:
    if n_bins < 2:
        raise ValueError(f'n_bins must be >= 2 for survival time discretization, got {n_bins}')
    if cutpoints.numel() != n_bins - 1:
        raise ValueError(f'Expected {n_bins - 1} cutpoints for {n_bins} bins, got {cutpoints.numel()}')
    flat_times = times.float().view(-1)
    cutpoints = cutpoints.to(device=flat_times.device, dtype=flat_times.dtype)
    return torch.bucketize(flat_times, cutpoints, right=False).clamp(0, n_bins - 1).long()


class FeatureDataset(Dataset):
    """One sample is one WSI with pre-extracted full features and optional coords."""

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
        n_time_bins: int = 4,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.encoder_name = encoder_name
        self.task = str(task).lower()
        self.split = str(split) if split is not None else 'all'
        self.seed = int(seed)
        self.label_column = label_column
        self.patch_drop_ratio = float(patch_drop_ratio)
        if self.patch_drop_ratio < 0.0 or self.patch_drop_ratio >= 1.0:
            raise ValueError(f'patch_drop_ratio must be in [0, 1), got {self.patch_drop_ratio}')
        self.patch_drop_seed = int(seed if patch_drop_seed is None else patch_drop_seed)
        self.include_labels = _as_str_list(include_labels)
        self.label_aliases = {str(k): str(v) for k, v in (label_aliases or {}).items()}
        self.feature_keys = tuple(str(key) for key in feature_keys) if feature_keys is not None else ('full', 'coords')
        self.generator = torch.Generator().manual_seed(seed)
        self.n_time_bins = int(n_time_bins)
        self.time_bin_cutpoints: torch.Tensor | None = None

        cohort = read_table(cohort_csv)
        folds = read_table(fold_csv)
        ensure_columns(cohort, ['case_submitter_id', 'slide_submitter_id'], 'cohort_csv')
        ensure_columns(folds, ['case_submitter_id', 'split'], 'fold_csv')

        self.folds = folds.copy()
        fold_cases = set(folds['case_submitter_id'].astype(str))
        fold_slides = cohort[cohort['case_submitter_id'].astype(str).isin(fold_cases)].copy()
        fold_slides = fold_slides.drop_duplicates('slide_submitter_id').reset_index(drop=True)

        if self.task == 'survival':
            ensure_columns(fold_slides, ['event', 'survival_days'], 'cohort_csv')
            fold_slides['event'] = pd.to_numeric(fold_slides['event'], errors='coerce')
            fold_slides['survival_days'] = pd.to_numeric(fold_slides['survival_days'], errors='coerce')
            fold_slides = fold_slides.dropna(subset=['event', 'survival_days'])
            fold_slides = fold_slides[fold_slides['survival_days'] > 0].copy()
            fold_slides = self._attach_global_time_bins(fold_slides)
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

        self.all_slides = fold_slides.reset_index(drop=True)
        self.slides = self._slides_for_split(self.split)

    def _survival_case_labels(self, fold_slides: pd.DataFrame) -> list[dict[str, torch.Tensor]]:
        cases = fold_slides[['case_submitter_id', 'event', 'survival_days']].drop_duplicates('case_submitter_id')
        return [
            {
                'event': torch.tensor(float(row.event), dtype=torch.float32),
                'time': torch.tensor(float(row.survival_days), dtype=torch.float32),
            }
            for row in cases.itertuples(index=False)
        ]

    def _attach_global_time_bins(self, fold_slides: pd.DataFrame) -> pd.DataFrame:
        self.time_bin_cutpoints = build_time_bin_cutpoints(self._survival_case_labels(fold_slides), self.n_time_bins)
        cases = fold_slides[['case_submitter_id', 'survival_days']].drop_duplicates('case_submitter_id').copy()
        bins = discretize_survival_times(
            torch.tensor(cases['survival_days'].astype(float).tolist(), dtype=torch.float32),
            self.time_bin_cutpoints,
            self.n_time_bins,
        )
        cases['time_bin'] = bins.cpu().numpy().astype(int)
        return fold_slides.merge(cases[['case_submitter_id', 'time_bin']], on='case_submitter_id', how='left')

    def _slides_for_split(self, split: str | None) -> pd.DataFrame:
        if split is None or str(split).lower() in {'all', 'full'}:
            split_cases = set(self.folds['case_submitter_id'].astype(str))
        else:
            split_cases = set(
                self.folds.loc[self.folds['split'].astype(str) == str(split), 'case_submitter_id'].astype(str)
            )
        return self.all_slides[self.all_slides['case_submitter_id'].astype(str).isin(split_cases)].copy().reset_index(drop=True)

    def split_dataset(self, split: str) -> 'FeatureDataset':
        child = object.__new__(self.__class__)
        child.__dict__ = self.__dict__.copy()
        child.split = str(split)
        child.slides = self._slides_for_split(split)
        child.generator = torch.Generator().manual_seed(self.seed)
        return child

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
                'time_bin': torch.tensor(int(row['time_bin']), dtype=torch.long),
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
        if 'coords' in bundle:
            sample['grid_indices'] = bundle['coords'].long()
        return sample
