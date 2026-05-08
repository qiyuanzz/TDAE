from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .utils import ensure_columns, read_table


SURVIVAL_N_BINS = 4
SURVIVAL_BIN_EPS = 1e-6


def _as_str_list(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None
    return [str(value) for value in values]


def _alias(raw_label: object, label_aliases: dict[str, str] | None = None) -> str:
    raw = str(raw_label)
    return {str(k): str(v) for k, v in (label_aliases or {}).items()}.get(raw, raw)


def _assign_survival_time_bins(cases: pd.DataFrame, n_bins: int = SURVIVAL_N_BINS, eps: float = SURVIVAL_BIN_EPS) -> pd.DataFrame:
    """Assign MCAT/SurvPath-style discrete time bins.

    The bin edges are fit on uncensored/event cases, then all cases are assigned
    to those fixed edges with left-closed intervals.
    """
    uncensored = cases[cases['event'] > 0]
    source = uncensored if len(uncensored) >= n_bins else cases
    try:
        _, bin_edges = pd.qcut(source['survival_days'], q=n_bins, retbins=True, labels=False)
    except ValueError:
        _, bin_edges = pd.cut(source['survival_days'], bins=n_bins, retbins=True, labels=False, include_lowest=True)
    bin_edges = bin_edges.copy()
    bin_edges[0] = cases['survival_days'].min() - eps
    bin_edges[-1] = cases['survival_days'].max() + eps
    time_bins = pd.cut(cases['survival_days'], bins=bin_edges, labels=False, right=False, include_lowest=True)
    if time_bins.isna().any():
        raise ValueError('Failed to assign survival time bins for all cases.')
    cases = cases.copy()
    cases['time_bin'] = time_bins.astype(int)
    return cases


def _case_table(
    cohort: pd.DataFrame,
    label_column: str,
    task: str = 'classification',
    include_labels: Iterable[str] | None = None,
    label_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    ensure_columns(cohort, ['case_submitter_id'], 'cohort_csv')
    task = str(task).lower()
    if task == 'survival':
        ensure_columns(cohort, ['event', 'survival_days'], 'cohort_csv')
        cases = cohort[['case_submitter_id', 'event', 'survival_days']].drop_duplicates('case_submitter_id').copy()
        cases['event'] = pd.to_numeric(cases['event'], errors='coerce')
        cases['survival_days'] = pd.to_numeric(cases['survival_days'], errors='coerce')
        cases = cases.dropna(subset=['case_submitter_id', 'event', 'survival_days']).reset_index(drop=True)
        cases = cases[cases['survival_days'] > 0].copy().reset_index(drop=True)
        cases['case_submitter_id'] = cases['case_submitter_id'].astype(str)
        cases['event'] = cases['event'].astype(int)
        cases['censorship'] = (1 - cases['event']).astype(int)
        cases = _assign_survival_time_bins(cases)
    elif task == 'classification':
        ensure_columns(cohort, [label_column], 'cohort_csv')
        include = _as_str_list(include_labels)
        cases = cohort[['case_submitter_id', label_column]].drop_duplicates('case_submitter_id').copy()
        cases = cases.dropna(subset=['case_submitter_id', label_column]).reset_index(drop=True)
        cases['case_submitter_id'] = cases['case_submitter_id'].astype(str)
        cases[label_column] = cases[label_column].astype(str)
        if include is not None:
            cases = cases[cases[label_column].isin(include)].copy()
        cases['label'] = cases[label_column].map(lambda value: _alias(value, label_aliases))
    else:
        raise ValueError(f'Unsupported task: {task}')
    if len(cases) < 2:
        raise ValueError('Need at least two cases to generate splits after task filtering.')
    return cases


def _can_stratify(labels: pd.Series, n_splits: int) -> bool:
    counts = labels.value_counts()
    return len(counts) > 1 and int(counts.min()) >= n_splits


def _split_train_val(train_cases: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(train_cases) <= 2 or val_fraction <= 0:
        return train_cases.copy(), train_cases.iloc[0:0].copy()
    stratify = train_cases['label'] if train_cases['label'].value_counts().min() >= 2 else None
    train, val = train_test_split(
        train_cases,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def generate_case_splits(
    cohort_csv: str | Path,
    out_dir: str | Path,
    label_column: str = 'cancer_code',
    n_splits: int = 5,
    val_fraction: float = 0.1,
    seed: int = 42,
    task: str = 'classification',
    include_labels: Iterable[str] | None = None,
    label_aliases: dict[str, str] | None = None,
) -> list[Path]:
    """Create case-level fold CSVs.

    Survival uses standard 5-fold cross validation with train/val only:
    the held-out fold is the validation/evaluation fold, and there is no
    separate test split. Classification keeps the older train/val/test split
    for optional sanity experiments.
    """
    cohort = read_table(cohort_csv)
    cases = _case_table(cohort, label_column, task=task, include_labels=include_labels, label_aliases=label_aliases)
    n_splits = min(int(n_splits), len(cases))
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2 after accounting for case count.')

    if str(task).lower() == 'survival':
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_arg = None
    else:
        splitter = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            if _can_stratify(cases['label'], n_splits)
            else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        )
        split_arg = cases['label'] if isinstance(splitter, StratifiedKFold) else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for fold, (train_val_idx, test_idx) in enumerate(splitter.split(cases, split_arg)):
        train_val = cases.iloc[train_val_idx].reset_index(drop=True)
        test = cases.iloc[test_idx].reset_index(drop=True)
        if str(task).lower() == 'survival':
            train = train_val
            val = test
            split_frames = (('train', train), ('val', val))
        else:
            train, val = _split_train_val(train_val, val_fraction=val_fraction, seed=seed + fold)
            split_frames = (('train', train), ('val', val), ('test', test))
        pieces = []
        for split_name, frame in split_frames:
            if str(task).lower() == 'survival':
                part = frame[['case_submitter_id', 'time_bin', 'survival_days', 'event', 'censorship']].copy()
            else:
                part = frame[['case_submitter_id', 'label']].copy()
            part['split'] = split_name
            pieces.append(part)
        fold_df = pd.concat(pieces, ignore_index=True)
        if str(task).lower() == 'survival':
            columns = ['case_submitter_id', 'split', 'time_bin', 'survival_days', 'event', 'censorship']
        else:
            columns = ['case_submitter_id', 'split', 'label']
        fold_df = fold_df[columns].sort_values(['split', 'case_submitter_id'])
        path = out / f'fold_{fold}.csv'
        fold_df.to_csv(path, index=False)
        paths.append(path)
    return paths
