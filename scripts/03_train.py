from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from data.feature_dataset import FeatureDataset
from data.task_config import load_dataset_config, load_encoder_config, load_yaml, merge_config, resolve_feature_dir
from engine.trainer import TDAETrainer, TrainerConfig
from models.tdae import TDAE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train 4-level TDAE.')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--cohort', default='GBMLGG')
    parser.add_argument('--task', default=None, choices=['classification', 'survival'])
    parser.add_argument('--encoder', default='uni2')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--method', default=None, choices=['trainable_gating', 'tdae_auto', 'full_upper_bound', 'random_patch_drop'])
    parser.add_argument('--c_target', type=float, default=None)
    parser.add_argument('--patch_drop_ratio', type=float, default=None)
    parser.add_argument('--patch_drop_seed', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--bag_loss', choices=['cox', 'nll_surv'], default=None)
    parser.add_argument('--alpha_surv', type=float, default=None)
    parser.add_argument('--n_classes', type=int, default=None)
    parser.add_argument('--run_tag', default=None)
    parser.add_argument('--experiment_tag', default=None)
    parser.add_argument('--output_layout', choices=['seed', 'legacy'], default='seed')
    parser.add_argument('--torch_num_threads', type=int, default=None)
    return parser


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_threads(torch_num_threads: int | None) -> int | None:
    if torch_num_threads is None:
        env_threads = os.environ.get('TDAE_TORCH_NUM_THREADS') or os.environ.get('OMP_NUM_THREADS')
        torch_num_threads = int(env_threads) if env_threads else None
    if torch_num_threads is None or torch_num_threads <= 0:
        return None
    import torch

    torch.set_num_threads(int(torch_num_threads))
    torch.set_num_interop_threads(1)
    return int(torch_num_threads)


def safe_path_name(value: object) -> str:
    text = str(value).strip()
    allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-')
    safe = ''.join(char if char in allowed else '_' for char in text)
    return safe.strip('._') or 'unnamed'


def float_path_label(value: float) -> str:
    return safe_path_name(f'{float(value):.6g}')


def float_tag(value: float) -> str:
    text = f'{float(value):.0e}' if 0 < abs(float(value)) < 1e-3 else f'{float(value):.6g}'
    text = text.replace('e-0', 'e-').replace('e+0', 'e').replace('e+', 'e')
    return safe_path_name(text)


def default_experiment_tag(method: str, run_tag: str, seed: int, fold: int) -> str:
    if method in {'tdae_auto', 'full_upper_bound', 'random_patch_drop', 'trainable_gating'}:
        return safe_path_name(method)
    tag = str(run_tag)
    for suffix in (f'_seed{seed}_fold{fold}', f'_fold{fold}', f'_seed{seed}'):
        if tag.endswith(suffix):
            tag = tag[: -len(suffix)]
    return safe_path_name(tag)


def budget_label(method: str, run_tag: str, c_target: float, patch_drop_ratio: float) -> str:
    if method == 'tdae_auto':
        return f'c_{float_path_label(c_target)}'
    if method == 'random_patch_drop':
        return f'keep_{float_path_label(1.0 - float(patch_drop_ratio))}'
    return safe_path_name(run_tag)


def setting_label(
    method: str,
    run_tag: str,
    c_target: float,
    patch_drop_ratio: float,
    survival_batch_size: int,
    lr: float,
) -> str:
    return safe_path_name(
        f'wsi_TDAE_{budget_label(method, run_tag, c_target, patch_drop_ratio)}'
        f'_gc_{survival_batch_size}_lr_{float_tag(lr)}'
    )


def resolve_output_dir(
    result_root: Path,
    cohort: str,
    task: str,
    feature_encoder: str,
    method: str,
    run_tag: str,
    experiment_tag: str | None,
    seed: int,
    fold: int,
    c_target: float,
    patch_drop_ratio: float,
    survival_batch_size: int,
    lr: float,
    layout: str,
) -> tuple[Path, str, str]:
    base = result_root / 'checkpoints' / cohort / task / feature_encoder
    if layout == 'legacy':
        return base / run_tag / f'fold_{fold}', safe_path_name(run_tag), safe_path_name(run_tag)
    exp_tag = safe_path_name(experiment_tag) if experiment_tag else default_experiment_tag(method, run_tag, seed, fold)
    setting = setting_label(method, run_tag, c_target, patch_drop_ratio, survival_batch_size, lr)
    return base / exp_tag / setting / f'seed_{seed}' / f'fold_{fold}', exp_tag, setting


def copy_split_file(fold_csv: Path, out_dir: Path) -> Path:
    split_copy = out_dir / 'split.csv'
    if not split_copy.exists():
        shutil.copy2(fold_csv, split_copy)
    return split_copy


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    keys: list[str] = []
    for row in history:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = build_parser().parse_args()
    configured_threads = _configure_threads(args.torch_num_threads)
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    default_cfg = load_yaml(config_path)
    dataset_cfg = load_dataset_config(args.cohort, args.task or default_cfg.get('task', 'classification'), root=root)
    cfg = merge_config(default_cfg, dataset_cfg)
    cfg = merge_config(cfg, load_encoder_config(args.encoder, root=root))
    task = cfg.get('task', args.task or 'classification')
    method = args.method or cfg.get('method', 'trainable_gating')
    seed = int(args.seed if args.seed is not None else cfg.get('seed', 1))
    _set_seed(seed)
    c_target = float(args.c_target if args.c_target is not None else cfg.get('c_target', cfg.get('c_end', 0.3)))
    patch_drop_ratio = float(args.patch_drop_ratio if args.patch_drop_ratio is not None else cfg.get('patch_drop_ratio', 0.0))
    if patch_drop_ratio < 0.0 or patch_drop_ratio >= 1.0:
        raise ValueError(f'--patch_drop_ratio must be in [0, 1), got {patch_drop_ratio}')
    patch_drop_seed = int(args.patch_drop_seed if args.patch_drop_seed is not None else seed)
    selection_method = 'full_upper_bound' if method == 'random_patch_drop' else method
    cohort_csv = cfg.get('cohort_csv', str(root / 'data' / 'csvs' / f'TCGA_{args.cohort}.csv'))
    split_dir = Path(cfg.get('split_dir', root / 'data' / 'splits'))
    feature_dir = resolve_feature_dir(cfg, args.cohort)
    feature_encoder = cfg.get('encoder_name', args.encoder)
    result_dir = Path(cfg.get('result_dir', str(root / 'outputs')))
    fold_csv = split_dir / args.cohort / task / f'fold_{args.fold}.csv'
    if not fold_csv.exists():
        fold_csv = split_dir / args.cohort / f'fold_{args.fold}.csv'
    dataset_kwargs = {
        'include_labels': cfg.get('include_labels'),
        'label_aliases': cfg.get('label_aliases'),
        'patch_drop_ratio': patch_drop_ratio,
        'patch_drop_seed': patch_drop_seed,
        'seed': seed,
    }
    train_ds = FeatureDataset(feature_dir, cohort_csv, fold_csv, feature_encoder, 'train', task, cfg.get('label_column', 'cancer_code'), **dataset_kwargs)
    val_ds = FeatureDataset(feature_dir, cohort_csv, fold_csv, feature_encoder, 'val', task, cfg.get('label_column', 'cancer_code'), **dataset_kwargs)
    sample = train_ds[0]
    bag_loss = str(args.bag_loss if args.bag_loss is not None else cfg.get('bag_loss', 'nll_surv' if task == 'survival' else 'ce'))
    survival_n_classes = int(args.n_classes if args.n_classes is not None else cfg.get('n_classes', 4))
    model_n_classes = survival_n_classes if task == 'survival' and bag_loss == 'nll_surv' else train_ds.n_classes
    model = TDAE(
        d_light=sample['light_feats'].shape[1],
        d_medium=sample['medium_feats'].shape[1],
        d_full=sample['full_feats'].shape[1],
        n_classes=model_n_classes,
        aggregator_type=cfg.get('aggregator', 'abmil'),
        gating_hidden=int(cfg.get('gating_hidden', 256)),
        pos_encoding_dim=int(cfg.get('pos_encoding_dim', 64)),
        d_gat_hidden=int(cfg.get('d_gat_hidden', 256)),
        gat_heads=int(cfg.get('gat_heads', 4)),
        gat_layers=int(cfg.get('gat_layers', 2)),
        k_neighbors=int(cfg.get('k_neighbors', 8)),
        flops_cost=cfg.get('flops_cost'),
        use_pyg_gat=cfg.get('use_pyg_gat'),
        selection_method=selection_method,
        c_target=c_target,
        phase0_alpha=float(cfg.get('phase0_alpha', 1.0)),
        phase0_beta=float(cfg.get('phase0_beta', 1.0)),
        phase0_gamma=float(cfg.get('phase0_gamma', 1.0)),
        phase0_k_recon=int(cfg.get('phase0_k_recon', 8)),
    )
    trainer_cfg = TrainerConfig(
        method=method,
        warmup_epochs=int(cfg.get('warmup_epochs', 5)) if method == 'trainable_gating' else 0,
        joint_epochs=int(cfg.get('task_epochs', cfg.get('joint_epochs', 20))),
        lr_gating=float(cfg.get('lr_gating', 5e-4)),
        lr_aggregator=float(cfg.get('lr_task', cfg.get('lr_aggregator', 2e-4))),
        weight_decay=float(cfg.get('weight_decay', 1e-5)),
        lambda_budget=float(cfg.get('lambda_budget', 1.0)) if method == 'trainable_gating' else 0.0,
        mu_diversity=float(cfg.get('mu_diversity', 0.1)) if method == 'trainable_gating' else 0.0,
        tau_start=float(cfg.get('tau_start', 1.0)),
        tau_end=float(cfg.get('tau_end', 0.1)),
        c_start=float(cfg.get('c_start', cfg.get('r_start', 0.6))),
        c_end=float(cfg.get('c_end', cfg.get('r_end', 0.3))),
        c_target=c_target,
        eval_every=int(cfg.get('eval_every', 1)),
        task=task,
        survival_batch_size=int(cfg.get('gc', cfg.get('survival_batch_size', 32))),
        bag_loss=bag_loss,
        alpha_surv=float(args.alpha_surv if args.alpha_surv is not None else cfg.get('alpha_surv', 0.0)),
        n_classes=model_n_classes,
        device=args.device,
    )
    if args.run_tag:
        run_tag = args.run_tag
    elif method == 'tdae_auto':
        run_tag = f'{method}_c{c_target:g}'
    elif method == 'random_patch_drop':
        run_tag = f'random_patch_drop_drop{patch_drop_ratio:g}_seed{patch_drop_seed}'
    else:
        run_tag = method
    out_dir, experiment_tag, setting = resolve_output_dir(
        result_dir,
        args.cohort,
        task,
        feature_encoder,
        method,
        run_tag,
        args.experiment_tag,
        seed,
        args.fold,
        c_target,
        patch_drop_ratio,
        trainer_cfg.survival_batch_size,
        trainer_cfg.lr_aggregator,
        args.output_layout,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    split_copy = copy_split_file(fold_csv, out_dir)
    result = TDAETrainer(model, train_ds, val_ds, trainer_cfg).train(out_dir)
    history = result.get('history', [])
    history_csv = out_dir / 'history.csv'
    write_history(history_csv, history)
    summary = {
        'run_tag': run_tag,
        'experiment_tag': experiment_tag,
        'output_layout': args.output_layout,
        'setting': setting,
        'out_dir': str(out_dir),
        'checkpoint': result.get('checkpoint'),
        'history_csv': str(history_csv),
        'split_copy': str(split_copy),
        'cohort': args.cohort,
        'task': task,
        'encoder': args.encoder,
        'feature_encoder': feature_encoder,
        'method': method,
        'fold': args.fold,
        'seed': seed,
        'c_target': c_target,
        'patch_drop_ratio': patch_drop_ratio,
        'patch_drop_seed': patch_drop_seed,
        'bag_loss': bag_loss,
        'alpha_surv': trainer_cfg.alpha_surv,
        'n_classes': model_n_classes,
        'train_slides': len(train_ds),
        'val_slides': len(val_ds),
        'cohort_csv': str(cohort_csv),
        'fold_csv': str(fold_csv),
        'feature_dir': str(feature_dir),
        'torch_num_threads': configured_threads,
    }
    if history:
        summary['last_epoch'] = history[-1].get('epoch')
        summary['last_metrics'] = history[-1]
    summary_path = out_dir / 'summary.json'
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(result['checkpoint'])
    print(summary_path)


if __name__ == '__main__':
    main()
