from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from trainer.datasets.feature_dataset import FeatureDataset
from trainer.task_config import load_dataset_config, load_encoder_config, load_yaml, merge_config, resolve_feature_dir
from trainer.losses import NLLSurvLoss
from models.builder import build_aggregator


def _load_train_module():
    path = Path(__file__).with_name("step_04_train.py")
    spec = importlib.util.spec_from_file_location("train_mil_random_drop", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a full-feature MIL checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="trainer/configs/default.yaml")
    parser.add_argument("--cohort", default="GBMLGG")
    parser.add_argument("--task", default="survival", choices=["survival"])
    parser.add_argument("--encoder", default="uni2")
    parser.add_argument("--aggregator", choices=["abmil", "transmil", "clam"], default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--bag_loss", choices=["cox", "nll_surv"], default=None)
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--alpha_surv", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_mod = _load_train_module()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_yaml(config_path)
    cfg = merge_config(cfg, load_dataset_config(args.cohort, args.task, root=root))
    cfg = merge_config(cfg, load_encoder_config(args.encoder, root=root))

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    summary = checkpoint.get("summary", {}) if isinstance(checkpoint, dict) else {}
    aggregator_name = str(args.aggregator or summary.get("aggregator") or cfg.get("aggregator", "abmil")).lower()
    if aggregator_name not in {"abmil", "transmil", "clam"}:
        raise ValueError(f"Unsupported MIL aggregator: {aggregator_name}")

    feature_dir = resolve_feature_dir(cfg, args.cohort)
    feature_encoder = cfg.get("encoder_name", args.encoder)
    split_dir = Path(cfg.get("split_dir", root / "metadata" / "splits"))
    fold_csv = split_dir / args.cohort / args.task / f"fold_{args.fold}.csv"
    if not fold_csv.exists():
        fold_csv = split_dir / args.cohort / f"fold_{args.fold}.csv"

    bag_loss = str(args.bag_loss or summary.get("bag_loss") or cfg.get("bag_loss", "nll_surv"))
    n_classes = int(args.n_classes or summary.get("n_classes") or cfg.get("n_classes", 4))
    model_n_classes = n_classes if bag_loss == "nll_surv" else 1
    fold_dataset = FeatureDataset(
        feature_dir,
        cfg["cohort_csv"],
        fold_csv,
        feature_encoder,
        "all",
        args.task,
        cfg.get("label_column", "cancer_code"),
        feature_keys=("full",),
        n_time_bins=model_n_classes,
    )
    dataset = fold_dataset.split_dataset(args.split)
    sample = dataset[0]
    d_full = int(sample["full_feats"].shape[1])
    model = build_aggregator(
        aggregator_name,
        d_in=d_full,
        d_hidden=int(cfg.get("mil_hidden", 256)),
        n_classes=model_n_classes,
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model.to(device)
    alpha = float(args.alpha_surv if args.alpha_surv is not None else summary.get("alpha_surv", cfg.get("alpha_surv", 0.0)))
    loss_fn = NLLSurvLoss(alpha=alpha) if bag_loss == "nll_surv" else None
    metrics = train_mod.evaluate(model, dataset, device, bag_loss=bag_loss, loss_fn=loss_fn, n_bins=model_n_classes)
    metrics.update({"aggregator": aggregator_name, "checkpoint": str(args.checkpoint), "split": args.split})
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
