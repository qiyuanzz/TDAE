from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from trainer.splits import generate_case_splits
from trainer.task_config import load_dataset_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate case-level MIL folds.')
    parser.add_argument('--cohort', default=None, help='Load cohort/task settings from trainer/configs/dataset/{cohort}.yaml.')
    parser.add_argument('--task', default='classification', choices=['classification', 'survival'])
    parser.add_argument('--cohort_csv', default=None)
    parser.add_argument('--label_column', default=None)
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--include_labels', nargs='*', default=None)
    parser.add_argument('--label_aliases_json', default=None, help='JSON dict mapping raw labels to class names.')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = {}
    if args.cohort:
        cfg = load_dataset_config(args.cohort, args.task, root=root)
    cohort_csv = args.cohort_csv or cfg.get('cohort_csv')
    if not cohort_csv:
        raise ValueError('Provide --cohort or --cohort_csv.')
    label_column = args.label_column or cfg.get('label_column', 'cancer_code')
    out_dir = args.out_dir or str(root / 'metadata' / 'splits' / str(args.cohort).upper() / args.task)
    include_labels = args.include_labels if args.include_labels is not None else cfg.get('include_labels')
    label_aliases = json.loads(args.label_aliases_json) if args.label_aliases_json else cfg.get('label_aliases')
    paths = generate_case_splits(
        cohort_csv=cohort_csv,
        out_dir=out_dir,
        label_column=label_column,
        n_splits=args.n_splits,
        val_fraction=args.val_fraction,
        seed=args.seed,
        task=args.task,
        include_labels=include_labels,
        label_aliases=label_aliases,
    )
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
