from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

import _bootstrap  # noqa: F401
from data.feature_dataset import FeatureDataset
from data.task_config import (
    load_dataset_config,
    load_encoder_config,
    load_yaml,
    merge_config,
    resolve_feature_dir,
)
from engine.losses import NLLSurvLoss, cox_partial_likelihood
from models.mil_aggregators import ABMIL
from utils.metrics import survival_c_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast pure ABMIL random patch-drop survival baseline.")
    parser.add_argument("--config", default="configs/brca_uni2_phase0.yaml")
    parser.add_argument("--cohort", default="BRCA")
    parser.add_argument("--task", default="survival", choices=["survival"])
    parser.add_argument("--encoder", default="uni2")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch_drop_ratio", type=float, default=0.0)
    parser.add_argument("--patch_drop_seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", "--max_epochs", dest="epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", "--reg", dest="weight_decay", type=float, default=1e-5)
    parser.add_argument("--opt", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--survival_batch_size", "--gc", dest="survival_batch_size", type=int, default=32)
    parser.add_argument("--bag_loss", choices=["cox", "nll_surv"], default="nll_surv")
    parser.add_argument("--alpha_surv", type=float, default=0.0)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--early_stopping_monitor", choices=["val_loss", "val_c_index"], default="val_loss")
    parser.add_argument("--permute_train_labels", action="store_true")
    parser.add_argument("--label_permutation_seed", type=int, default=None)
    parser.add_argument("--torch_num_threads", type=int, default=None)
    parser.add_argument("--no_cache_features", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--experiment_tag", default=None)
    parser.add_argument("--output_layout", choices=["seed", "legacy"], default="seed")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def case_index_groups(dataset) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = OrderedDict()
    if hasattr(dataset, "slides"):
        case_ids = dataset.slides["case_submitter_id"].astype(str).tolist()
    else:
        case_ids = [str(sample["case_id"]) for sample in dataset]
    for idx, case_id in enumerate(case_ids):
        groups.setdefault(case_id, []).append(idx)
    return groups


def cache_dataset(dataset: FeatureDataset, name: str) -> list[dict[str, Any]]:
    cached: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        cached.append(
            {
                "full_feats": sample["full_feats"].half().contiguous(),
                "label": sample["label"],
                "slide_id": sample["slide_id"],
                "case_id": sample["case_id"],
            }
        )
        if (idx + 1) % 100 == 0 or idx + 1 == len(dataset):
            print(f"[cache:{name}] {idx + 1}/{len(dataset)} slides", flush=True)
    return cached


def clone_label(label):
    if isinstance(label, dict):
        return {key: value.clone() for key, value in label.items()}
    return label.clone()


def permute_case_labels(cached: list[dict[str, Any]], seed: int) -> None:
    groups = case_index_groups(cached)
    case_ids = list(groups)
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(case_ids), generator=generator).tolist()
    source_labels = {
        case_id: clone_label(cached[groups[case_ids[source_pos]][0]]["label"])
        for case_id, source_pos in zip(case_ids, order)
    }
    for case_id, indices in groups.items():
        for idx in indices:
            cached[idx]["label"] = clone_label(source_labels[case_id])


def case_label_list(dataset, groups: dict[str, list[int]]) -> list[dict[str, torch.Tensor]]:
    return [dataset[indices[0]]["label"] for indices in groups.values() if indices]


def fold_case_label_list(*dataset_groups: tuple[Any, dict[str, list[int]]]) -> list[dict[str, torch.Tensor]]:
    labels_by_case: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
    for dataset, groups in dataset_groups:
        for case_id, indices in groups.items():
            if indices and case_id not in labels_by_case:
                labels_by_case[case_id] = dataset[indices[0]]["label"]
    return list(labels_by_case.values())


def build_time_bin_cutpoints(labels: list[dict[str, torch.Tensor]], n_bins: int) -> torch.Tensor:
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2 for nll_surv, got {n_bins}")
    times = torch.stack([label["time"].detach().float().cpu().view(()) for label in labels])
    events = torch.stack([label["event"].detach().float().cpu().view(()) for label in labels])
    if times.numel() == 0:
        raise ValueError("Cannot build survival time bins from an empty fold cohort.")
    source_times = times[events > 0]
    if source_times.numel() < n_bins:
        source_times = times
    if source_times.numel() < n_bins:
        raise ValueError(f"Need at least {n_bins} survival times to build {n_bins} qcut bins, got {source_times.numel()}.")
    try:
        _, bin_edges = pd.qcut(source_times.numpy(), q=n_bins, labels=False, retbins=True)
        cutpoints = torch.tensor(bin_edges[1:-1], dtype=torch.float32)
    except ValueError:
        quantiles = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float32)[1:-1]
        cutpoints = torch.quantile(source_times.float(), quantiles)
    if cutpoints.numel() != n_bins - 1:
        raise ValueError(f"Expected {n_bins - 1} cutpoints for {n_bins} bins, got {cutpoints.numel()}.")
    return cutpoints.float()


def discretize_survival_times(times: torch.Tensor, cutpoints: torch.Tensor, n_bins: int) -> torch.Tensor:
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2 for nll_surv, got {n_bins}")
    if cutpoints.numel() != n_bins - 1:
        raise ValueError(f"Expected {n_bins - 1} cutpoints for {n_bins} bins, got {cutpoints.numel()}")
    flat_times = times.float().view(-1)
    cutpoints = cutpoints.to(device=flat_times.device, dtype=flat_times.dtype)
    return torch.bucketize(flat_times, cutpoints, right=False).clamp(0, n_bins - 1).long()


def survival_logits_to_risk(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    return -survival.sum(dim=1)


def forward_case(model: ABMIL, dataset, indices: list[int], device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    case_feats = []
    label = None
    for idx in indices:
        sample = dataset[idx]
        case_feats.append(sample["full_feats"])
        if label is None:
            label = sample["label"]
    if label is None:
        raise ValueError("Cannot forward empty case.")
    feats = torch.cat(case_feats, dim=0).to(device=device, dtype=torch.float32)
    logits, _ = model(feats)
    return logits.flatten(), label


def cox_batch_loss(risks: list[torch.Tensor], labels: list[dict[str, torch.Tensor]], device: torch.device) -> torch.Tensor:
    risk_tensor = torch.cat([risk.flatten()[:1] for risk in risks], dim=0)
    target = {
        "event": torch.stack([label["event"].to(device).view(()) for label in labels]),
        "time": torch.stack([label["time"].to(device).view(()) for label in labels]),
    }
    return cox_partial_likelihood(risk_tensor, target)


def nll_batch_loss(
    logits: list[torch.Tensor],
    labels: list[dict[str, torch.Tensor]],
    device: torch.device,
    cutpoints: torch.Tensor | None,
    loss_fn: NLLSurvLoss,
    n_bins: int,
) -> torch.Tensor:
    logits_tensor = torch.stack([item.flatten() for item in logits], dim=0)
    times = torch.stack([label["time"].to(device).view(()) for label in labels])
    events = torch.stack([label["event"].to(device).view(()) for label in labels])
    if cutpoints is None:
        raise RuntimeError("nll_surv requires train-fold cutpoints.")
    time_bins = discretize_survival_times(times, cutpoints.to(device), n_bins)
    if all("censorship" in label for label in labels):
        censorships = torch.stack([label["censorship"].to(device).view(()) for label in labels]).long()
    else:
        censorships = (1.0 - events.float()).long()
    return loss_fn(logits_tensor, time_bins, censorships)


@torch.no_grad()
def evaluate(
    model: ABMIL,
    dataset,
    device: torch.device,
    bag_loss: str = "cox",
    cutpoints: torch.Tensor | None = None,
    loss_fn: NLLSurvLoss | None = None,
    n_bins: int = 4,
) -> dict[str, float]:
    model.eval()
    groups = case_index_groups(dataset)
    case_logits = []
    case_times = []
    case_events = []
    case_censorships = []
    n_slides = 0
    for indices in groups.values():
        logits, label = forward_case(model, dataset, indices, device)
        case_logits.append(logits.detach())
        case_times.append(label["time"].detach().cpu().view(()))
        case_events.append(label["event"].detach().cpu().view(()))
        if "censorship" in label:
            case_censorships.append(label["censorship"].detach().cpu().view(()))
        n_slides += len(indices)
    if not case_logits:
        return {"c_index": float("nan"), "val_loss": float("nan"), "n_cases": 0.0, "n_events": 0.0, "n_slides": 0.0}
    logits_tensor = torch.stack(case_logits, dim=0)
    times = torch.stack(case_times)
    events = torch.stack(case_events)
    if bag_loss == "nll_surv":
        if loss_fn is None:
            raise ValueError("nll_surv evaluation requires loss_fn.")
        risks = survival_logits_to_risk(logits_tensor).detach().cpu()
        if cutpoints is None:
            raise RuntimeError("nll_surv evaluation requires train-fold cutpoints.")
        time_bins = discretize_survival_times(times, cutpoints, n_bins)
        if len(case_censorships) == len(case_logits):
            censorships = torch.stack(case_censorships).long()
        else:
            censorships = (1.0 - events.float()).long()
        val_loss = loss_fn(logits_tensor.to(device), time_bins.to(device), censorships.to(device))
    else:
        risks = logits_tensor[:, 0].detach().cpu()
        val_loss = cox_partial_likelihood(risks, {"event": events, "time": times})
    return {
        "c_index": survival_c_index(risks, times, events),
        "val_loss": float(val_loss.detach().cpu()),
        "n_cases": float(len(case_logits)),
        "n_events": float(events.sum().item()),
        "n_slides": float(n_slides),
    }


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.10g}" if math.isfinite(value) else ""
    return value


def _summary_sort_key(path: Path) -> int:
    try:
        return int(path.parent.name.split("_")[-1])
    except (IndexError, ValueError):
        return 0


def write_seed_fold_metrics(seed_dir: str | Path) -> Path:
    """Write one seed-level CSV with fold rows plus mean/std rows."""
    seed_dir = Path(seed_dir)
    summary_paths = sorted(seed_dir.glob("fold_*/summary.json"), key=_summary_sort_key)
    summaries: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                summaries.append(json.load(handle))
        except (OSError, json.JSONDecodeError):
            continue

    out_csv = seed_dir / "seed_metrics.csv"
    fieldnames = [
        "row_type",
        "fold",
        "seed",
        "keep_ratio",
        "patch_drop_ratio",
        "best_c_index",
        "best_val_loss",
        "epochs_run",
        "best_epoch",
        "early_stopped",
        "stopped_epoch",
        "n_cases",
        "n_events",
        "n_slides",
        "out_dir",
    ]
    metric_fields = [
        "best_c_index",
        "best_val_loss",
        "epochs_run",
        "best_epoch",
        "n_cases",
        "n_events",
        "n_slides",
    ]
    rows: list[dict[str, Any]] = []
    for summary in sorted(summaries, key=lambda item: int(item.get("fold", 0))):
        rows.append(
            {
                "row_type": "fold",
                "fold": summary.get("fold", ""),
                "seed": summary.get("seed", ""),
                "keep_ratio": summary.get("keep_ratio", ""),
                "patch_drop_ratio": summary.get("patch_drop_ratio", ""),
                "best_c_index": summary.get("best_c_index", summary.get("c_index", "")),
                "best_val_loss": summary.get("best_val_loss", summary.get("val_loss", "")),
                "epochs_run": summary.get("epochs_run", summary.get("epochs", "")),
                "best_epoch": summary.get("best_epoch", ""),
                "early_stopped": summary.get("early_stopped", ""),
                "stopped_epoch": summary.get("stopped_epoch", ""),
                "n_cases": summary.get("n_cases", ""),
                "n_events": summary.get("n_events", ""),
                "n_slides": summary.get("n_slides", ""),
                "out_dir": summary.get("out_dir", str(seed_dir / f"fold_{summary.get('fold', '')}")),
            }
        )

    fold_rows = list(rows)
    if fold_rows:
        base = {
            "fold": "",
            "seed": fold_rows[0].get("seed", ""),
            "keep_ratio": fold_rows[0].get("keep_ratio", ""),
            "patch_drop_ratio": fold_rows[0].get("patch_drop_ratio", ""),
            "early_stopped": "",
            "stopped_epoch": "",
            "out_dir": "",
        }
        for row_type in ("mean", "std"):
            aggregate = {"row_type": row_type, **base}
            for field in metric_fields:
                values = [_as_float(row.get(field)) for row in fold_rows]
                values = [value for value in values if math.isfinite(value)]
                if not values:
                    aggregate[field] = ""
                elif row_type == "mean":
                    aggregate[field] = statistics.mean(values)
                else:
                    aggregate[field] = statistics.stdev(values) if len(values) > 1 else 0.0
            rows.append(aggregate)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_csv_value(row.get(field, "")) for field in fieldnames})
    return out_csv


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def monitor_score(metrics: dict[str, Any], monitor: str) -> float:
    if monitor == "val_loss":
        return float(metrics.get("val_loss", float("nan")))
    if monitor == "val_c_index":
        return float(metrics.get("c_index", float("nan")))
    raise ValueError(f"Unsupported early stopping monitor: {monitor}")


def monitor_improved(value: float, best_value: float, monitor: str, min_delta: float) -> bool:
    if not math.isfinite(value):
        return False
    if monitor == "val_loss":
        return value < best_value - min_delta
    return value > best_value + min_delta


def select_report_row(history: list[dict[str, Any]], monitor: str, use_early_stopping: bool) -> dict[str, Any]:
    if not history:
        raise ValueError("Cannot select a report row from empty history.")
    if not use_early_stopping:
        return history[-1].copy()
    if monitor == "val_loss":
        return min(history, key=lambda row: _as_float(row.get("val_loss"))).copy()
    if monitor == "val_c_index":
        return max(history, key=lambda row: _as_float(row.get("c_index"))).copy()
    raise ValueError(f"Unsupported early stopping monitor: {monitor}")


def safe_path_name(value: object) -> str:
    text = str(value).strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    safe = "".join(char if char in allowed else "_" for char in text)
    return safe.strip("._") or "unnamed"


def keep_label(keep_ratio: float) -> str:
    text = f"{float(keep_ratio):.6g}"
    return f"keep_{safe_path_name(text)}"


def float_tag(value: float) -> str:
    text = f"{float(value):.0e}" if 0 < abs(float(value)) < 1e-3 else f"{float(value):.6g}"
    text = text.replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
    return safe_path_name(text)


def setting_label(keep_ratio: float, survival_batch_size: int, lr: float) -> str:
    return safe_path_name(f"wsi_ABMIL_{keep_label(keep_ratio)}_gc_{survival_batch_size}_lr_{float_tag(lr)}")


def default_experiment_tag(run_tag: str, seed: int, fold: int) -> str:
    tag = str(run_tag)
    for suffix in (f"_seed{seed}_fold{fold}", f"_fold{fold}", f"_seed{seed}"):
        if tag.endswith(suffix):
            tag = tag[: -len(suffix)]
    if tag.startswith("abmil_random_keep"):
        return "abmil_random_keep"
    return safe_path_name(tag)


def resolve_output_dir(
    result_root: Path,
    cohort: str,
    task: str,
    feature_encoder: str,
    run_tag: str,
    experiment_tag: str | None,
    seed: int,
    fold: int,
    keep_ratio: float,
    survival_batch_size: int,
    lr: float,
    layout: str,
) -> tuple[Path, str, str]:
    base = result_root / "checkpoints" / cohort / task / feature_encoder
    if layout == "legacy":
        legacy_tag = safe_path_name(run_tag)
        return base / run_tag / f"fold_{fold}", legacy_tag, legacy_tag
    exp_tag = safe_path_name(experiment_tag) if experiment_tag else default_experiment_tag(run_tag, seed, fold)
    setting = setting_label(keep_ratio, survival_batch_size, lr)
    out_dir = base / exp_tag / setting / f"seed_{seed}" / f"fold_{fold}"
    return out_dir, exp_tag, setting


def copy_split_file(fold_csv: Path, out_dir: Path) -> Path:
    split_copy = out_dir / "split.csv"
    if not split_copy.exists():
        shutil.copy2(fold_csv, split_copy)
    return split_copy


def main() -> None:
    args = build_parser().parse_args()
    torch_num_threads = args.torch_num_threads
    if torch_num_threads is None:
        env_threads = os.environ.get("TDAE_TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
        torch_num_threads = int(env_threads) if env_threads else None
    if torch_num_threads is not None and torch_num_threads > 0:
        torch.set_num_threads(int(torch_num_threads))
        torch.set_num_interop_threads(1)
    if args.patch_drop_ratio < 0.0 or args.patch_drop_ratio >= 1.0:
        raise ValueError(f"patch_drop_ratio must be in [0, 1), got {args.patch_drop_ratio}")
    set_seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_yaml(config_path)
    cfg = merge_config(cfg, load_dataset_config(args.cohort, args.task, root=root))
    cfg = merge_config(cfg, load_encoder_config(args.encoder, root=root))

    feature_dir = resolve_feature_dir(cfg, args.cohort)
    feature_encoder = cfg.get("encoder_name", args.encoder)
    split_dir = Path(cfg.get("split_dir", root / "data" / "splits"))
    fold_csv = split_dir / args.cohort / args.task / f"fold_{args.fold}.csv"
    if not fold_csv.exists():
        fold_csv = split_dir / args.cohort / f"fold_{args.fold}.csv"

    dataset_kwargs = {
        "patch_drop_ratio": args.patch_drop_ratio,
        "patch_drop_seed": args.patch_drop_seed,
        "seed": args.seed,
        "feature_keys": ("full",),
    }
    train_ds = FeatureDataset(
        feature_dir,
        cfg["cohort_csv"],
        fold_csv,
        feature_encoder,
        "train",
        args.task,
        cfg.get("label_column", "cancer_code"),
        **dataset_kwargs,
    )
    val_ds = FeatureDataset(
        feature_dir,
        cfg["cohort_csv"],
        fold_csv,
        feature_encoder,
        "val",
        args.task,
        cfg.get("label_column", "cancer_code"),
        **dataset_kwargs,
    )

    if args.no_cache_features:
        if args.permute_train_labels:
            raise ValueError("--permute_train_labels requires cached features; omit --no_cache_features.")
        train_data = train_ds
        val_data = val_ds
    else:
        train_data = cache_dataset(train_ds, "train")
        val_data = cache_dataset(val_ds, "val")
        if args.permute_train_labels:
            perm_seed = int(args.label_permutation_seed if args.label_permutation_seed is not None else args.seed)
            permute_case_labels(train_data, perm_seed)

    train_groups = case_index_groups(train_data)
    train_case_ids = list(train_groups)
    val_groups = case_index_groups(val_data)
    if args.bag_loss == "nll_surv" and args.n_classes < 2:
        raise ValueError(f"--n_classes must be >= 2 for nll_surv, got {args.n_classes}")

    sample = train_data[0]
    d_full = int(sample["full_feats"].shape[1])
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model_n_classes = int(args.n_classes if args.bag_loss == "nll_surv" else 1)
    model = ABMIL(d_in=d_full, d_hidden=int(cfg.get("d_gat_hidden", 256)), n_classes=model_n_classes).to(device)

    epochs = int(args.epochs)
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    survival_batch_size = int(args.survival_batch_size)
    batch_size = int(args.batch_size)
    early_stopping_patience = int(args.early_stopping_patience)
    early_stopping_min_delta = float(args.early_stopping_min_delta)
    early_stopping_monitor = args.early_stopping_monitor
    loss_fn = NLLSurvLoss(alpha=float(args.alpha_surv)) if args.bag_loss == "nll_surv" else None
    # MCAT/SurvPath convention: fit cutpoints on the current fold's TRAIN
    # cases only (uncensored events), then apply the same cutpoints to val.
    # Never read time_bin from the fold csv even if a stale column is present.
    time_bin_cutpoints = None
    if args.bag_loss == "nll_surv":
        time_bin_cutpoints = build_time_bin_cutpoints(
            fold_case_label_list((train_data, train_groups)),
            model_n_classes,
        )
    optimizer_cls = torch.optim.AdamW if args.opt == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    keep_ratio = 1.0 - float(args.patch_drop_ratio)
    run_tag = args.run_tag or f"abmil_random_keep{keep_ratio:.2f}_seed{args.seed}_fold{args.fold}"
    out_dir, experiment_tag, setting = resolve_output_dir(
        Path(cfg.get("result_dir", root / "outputs")),
        args.cohort,
        args.task,
        feature_encoder,
        run_tag,
        args.experiment_tag,
        args.seed,
        args.fold,
        keep_ratio,
        survival_batch_size,
        lr,
        args.output_layout,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    split_copy = copy_split_file(fold_csv, out_dir)

    history: list[dict[str, Any]] = []
    print(
        json.dumps(
            {
                "run_tag": run_tag,
                "experiment_tag": experiment_tag,
                "setting": setting,
                "output_layout": args.output_layout,
                "out_dir": str(out_dir),
                "fold": args.fold,
                "keep_ratio": keep_ratio,
                "patch_drop_ratio": args.patch_drop_ratio,
                "train_cases": len(train_groups),
                "train_slides": len(train_data),
                "val_cases": len(val_groups),
                "val_slides": len(val_data),
                "cache_features": not args.no_cache_features,
                "device": str(device),
                "max_epochs": epochs,
                "batch_size": batch_size,
                "gc": survival_batch_size,
                "opt": args.opt,
                "lr": lr,
                "weight_decay": weight_decay,
                "bag_loss": args.bag_loss,
                "alpha_surv": float(args.alpha_surv),
                "n_classes": model_n_classes,
                "time_bin_cutpoints": None
                if time_bin_cutpoints is None
                else [float(value) for value in time_bin_cutpoints.detach().cpu().tolist()],
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "monitor": early_stopping_monitor,
                "permute_train_labels": args.permute_train_labels,
                "label_permutation_seed": args.label_permutation_seed if args.label_permutation_seed is not None else args.seed,
                "torch_num_threads": torch.get_num_threads(),
                "split_copy": str(split_copy),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    best_score = float("inf") if early_stopping_monitor == "val_loss" else -float("inf")
    best_epoch = -1
    best_row: dict[str, Any] | None = None
    best_state = cpu_state_dict(model)
    epochs_without_improvement = 0
    stopped_epoch = -1
    early_stopped = False

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_case_ids)).tolist()
        pending_logits: list[torch.Tensor] = []
        pending_labels: list[dict[str, torch.Tensor]] = []
        losses = []
        update_batch_size = max(1, survival_batch_size) if args.bag_loss == "nll_surv" else max(2, survival_batch_size)
        for case_pos in order:
            case_id = train_case_ids[case_pos]
            logits, label = forward_case(model, train_data, train_groups[case_id], device)
            pending_logits.append(logits)
            pending_labels.append(label)
            if len(pending_logits) >= update_batch_size:
                if args.bag_loss == "nll_surv":
                    assert loss_fn is not None
                    loss = nll_batch_loss(pending_logits, pending_labels, device, time_bin_cutpoints, loss_fn, model_n_classes)
                else:
                    loss = cox_batch_loss(pending_logits, pending_labels, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                pending_logits = []
                pending_labels = []
        if pending_logits and (args.bag_loss == "nll_surv" or len(pending_logits) > 1):
            if args.bag_loss == "nll_surv":
                assert loss_fn is not None
                loss = nll_batch_loss(pending_logits, pending_labels, device, time_bin_cutpoints, loss_fn, model_n_classes)
            else:
                loss = cox_batch_loss(pending_logits, pending_labels, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate(
            model,
            val_data,
            device,
            bag_loss=args.bag_loss,
            cutpoints=time_bin_cutpoints,
            loss_fn=loss_fn,
            n_bins=model_n_classes,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(sum(losses) / max(len(losses), 1)),
            "keep_ratio": keep_ratio,
            "patch_drop_ratio": float(args.patch_drop_ratio),
            **metrics,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        write_history(out_dir / "history.csv", history)

        score = monitor_score(metrics, early_stopping_monitor)
        improved = monitor_improved(score, best_score, early_stopping_monitor, early_stopping_min_delta)
        if improved:
            best_score = score
            best_epoch = epoch
            best_row = row.copy()
            best_state = cpu_state_dict(model)
            epochs_without_improvement = 0
            torch.save(
                {"model": best_state, "best_epoch": best_epoch, "best_metrics": best_row, "history": history},
                out_dir / "abmil_best.pt",
            )
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            stopped_epoch = epoch
            early_stopped = True
            print(
                json.dumps(
                    {
                        "event": "early_stopping",
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "selection_metric": early_stopping_monitor,
                        "best_monitor_value": best_score,
                        "best_c_index": None if best_row is None else best_row.get("c_index"),
                        "best_val_loss": None if best_row is None else best_row.get("val_loss"),
                        "patience": early_stopping_patience,
                        "min_delta": early_stopping_min_delta,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

    if not history:
        metrics = evaluate(
            model,
            val_data,
            device,
            bag_loss=args.bag_loss,
            cutpoints=time_bin_cutpoints,
            loss_fn=loss_fn,
            n_bins=model_n_classes,
        )
        history.append({"epoch": -1, "train_loss": float("nan"), **metrics})
    use_early_stopping = early_stopping_patience > 0
    monitor_best_row = select_report_row(history, early_stopping_monitor, use_early_stopping=True)
    report_row = select_report_row(history, early_stopping_monitor, use_early_stopping=use_early_stopping)
    if use_early_stopping:
        best_row = report_row.copy()
        best_epoch = int(best_row.get("epoch", -1))
        best_score = monitor_score(best_row, early_stopping_monitor)
        if best_state is None:
            best_state = cpu_state_dict(model)
    else:
        best_row = report_row.copy()
        best_epoch = int(best_row.get("epoch", -1))
        best_score = monitor_score(best_row, early_stopping_monitor)
        best_state = cpu_state_dict(model)
    torch.save(
        {"model": best_state, "best_epoch": best_epoch, "best_metrics": best_row, "history": history},
        out_dir / "abmil_best.pt",
    )

    metrics = report_row.copy()
    selected_c_index = float(metrics.get("c_index", float("nan")))
    selected_val_loss = float(metrics.get("val_loss", float("nan")))
    summary = {
        "run_tag": run_tag,
        "experiment_tag": experiment_tag,
        "setting": setting,
        "output_layout": args.output_layout,
        "out_dir": str(out_dir),
        "fold": args.fold,
        "seed": args.seed,
        "patch_drop_seed": args.patch_drop_seed,
        "keep_ratio": keep_ratio,
        "patch_drop_ratio": float(args.patch_drop_ratio),
        "epochs": len(history),
        "max_epochs": epochs,
        "epochs_run": len(history),
        "report_epoch": best_epoch,
        "report_c_index": selected_c_index,
        "report_val_loss": selected_val_loss,
        "best_epoch": best_epoch,
        "selection_metric": early_stopping_monitor if use_early_stopping else "last_epoch",
        "best_monitor_value": best_score,
        "best_c_index": selected_c_index,
        "best_val_loss": selected_val_loss,
        "monitor_best_epoch": int(monitor_best_row.get("epoch", -1)),
        "monitor_best_c_index": float(monitor_best_row.get("c_index", float("nan"))),
        "monitor_best_val_loss": float(monitor_best_row.get("val_loss", float("nan"))),
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "monitor": early_stopping_monitor,
        "permute_train_labels": args.permute_train_labels,
        "label_permutation_seed": args.label_permutation_seed if args.label_permutation_seed is not None else args.seed,
        "torch_num_threads": torch.get_num_threads(),
        "lr": lr,
        "weight_decay": weight_decay,
        "opt": args.opt,
        "batch_size": batch_size,
        "survival_batch_size": survival_batch_size,
        "bag_loss": args.bag_loss,
        "alpha_surv": float(args.alpha_surv),
        "n_classes": model_n_classes,
        "time_bin_cutpoints": None
        if time_bin_cutpoints is None
        else [float(value) for value in time_bin_cutpoints.detach().cpu().tolist()],
        "cache_features": not args.no_cache_features,
        "cohort_csv": str(cfg["cohort_csv"]),
        "fold_csv": str(fold_csv),
        "split_copy": str(split_copy),
        "feature_dir": str(feature_dir),
        **metrics,
    }
    torch.save({"model": model.state_dict(), "summary": summary, "history": history}, out_dir / "abmil_last.pt")
    torch.save({"model": best_state, "summary": summary, "history": history}, out_dir / "abmil_best.pt")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_seed_fold_metrics(out_dir.parent)
    print(str(out_dir / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
