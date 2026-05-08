from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from data.feature_dataset import FeatureDataset
from data.task_config import load_dataset_config, load_encoder_config, load_yaml, merge_config, resolve_feature_dir
from engine.evaluator import evaluate_model, measure_wall_time
from models.tdae import TDAE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate a trained 4-level TDAE checkpoint.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--cohort', default='GBMLGG')
    parser.add_argument('--task', default=None, choices=['classification', 'survival'])
    parser.add_argument('--encoder', default='uni2')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--measure_efficiency', action='store_true')
    parser.add_argument('--device', default='cuda')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    default_cfg = load_yaml(config_path)
    dataset_cfg = load_dataset_config(args.cohort, args.task or default_cfg.get('task', 'classification'), root=root)
    cfg = merge_config(default_cfg, dataset_cfg)
    cfg = merge_config(cfg, load_encoder_config(args.encoder, root=root))
    task = cfg.get('task', args.task or 'classification')
    cohort_csv = cfg.get('cohort_csv', str(root / 'data' / 'csvs' / f'TCGA_{args.cohort}.csv'))
    split_dir = Path(cfg.get('split_dir', root / 'data' / 'splits'))
    feature_dir = resolve_feature_dir(cfg, args.cohort)
    feature_encoder = cfg.get('encoder_name', args.encoder)
    fold_csv = split_dir / args.cohort / task / f'fold_{args.fold}.csv'
    if not fold_csv.exists():
        fold_csv = split_dir / args.cohort / f'fold_{args.fold}.csv'
    dataset = FeatureDataset(
        feature_dir,
        cohort_csv,
        fold_csv,
        feature_encoder,
        args.split,
        task,
        cfg.get('label_column', 'cancer_code'),
        include_labels=cfg.get('include_labels'),
        label_aliases=cfg.get('label_aliases'),
    )
    sample = dataset[0]
    model = TDAE(
        d_light=sample['light_feats'].shape[1],
        d_medium=sample['medium_feats'].shape[1],
        d_full=sample['full_feats'].shape[1],
        n_classes=dataset.n_classes,
        aggregator_type=cfg.get('aggregator', 'abmil'),
        gating_hidden=int(cfg.get('gating_hidden', 256)),
        pos_encoding_dim=int(cfg.get('pos_encoding_dim', 64)),
        d_gat_hidden=int(cfg.get('d_gat_hidden', 256)),
        gat_heads=int(cfg.get('gat_heads', 4)),
        gat_layers=int(cfg.get('gat_layers', 2)),
        k_neighbors=int(cfg.get('k_neighbors', 8)),
        flops_cost=cfg.get('flops_cost'),
        use_pyg_gat=cfg.get('use_pyg_gat'),
    )
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model.to(device)
    metrics = evaluate_model(model, dataset, task, device)
    if args.measure_efficiency:
        metrics.update(measure_wall_time(lambda: evaluate_model(model, dataset, task, device), repeats=3))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
