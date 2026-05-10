from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import pandas as pd
import torch


TRIDENT_ENCODERS = {
    "uni2": "uni_v2",
    "uni2h": "uni_v2",
    "uni_v2": "uni_v2",
    "virchow2": "virchow2",
    "conch_v15": "conch_v15",
    "conchv1_5": "conch_v15",
    "ctranspath": "ctranspath",
}

DEFAULT_CKPTS = {
    "uni_v2": "/mnt/Xsky/models/PFM/UNI2-h/pytorch_model.bin",
    "virchow2": "/mnt/Xsky/models/PFM/Virchow2/pytorch_model.bin",
    "conch_v15": "/mnt/Xsky/models/PFM/conchv1_5/pytorch_model_vision.bin",
    "ctranspath": "/mnt/Xsky/public_model_zoo/CHIEF/model_weight/CHIEF_CTransPath.pth",
}


def _coords_dir_name(mag: float, patch_size: int, overlap: int) -> str:
    mag_str = f"{float(mag):g}"
    return f"{mag_str}x_{patch_size}px_{overlap}px_overlap"


def _slide_stem(row) -> str:
    if "file_path" in row and str(row["file_path"]) and str(row["file_path"]) != "nan":
        return Path(str(row["file_path"])).stem
    return Path(str(row["file_name"])).stem


def _write_wsi_list(cohort: pd.DataFrame, wsi_root: Path, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in cohort.drop_duplicates("slide_submitter_id").iterrows():
        file_path = Path(str(row["file_path"]))
        try:
            rel = file_path.relative_to(wsi_root)
        except ValueError:
            rel = Path(str(row.get("file_dir", ""))) / str(row["file_name"])
        rows.append({"wsi": rel.as_posix()})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wsi"])
        writer.writeheader()
        writer.writerows(rows)


def _coords_to_grid(coords: torch.Tensor, patch_size: int) -> torch.Tensor:
    if coords.numel() == 0:
        return coords.new_zeros((0, 2))
    xy = coords.long()
    return torch.stack((xy[:, 1] // patch_size, xy[:, 0] // patch_size), dim=1)


def _encoder_name(name: str) -> str:
    key = name.lower()
    if key not in TRIDENT_ENCODERS:
        raise ValueError(f"Unsupported encoder {name}. Choices: {sorted(TRIDENT_ENCODERS)}")
    return TRIDENT_ENCODERS[key]


def _feature_encoder_name(args: argparse.Namespace) -> str:
    return _encoder_name(args.encoder)


def _ckpt_path(trident_encoder: str, user_path: str | None) -> str | None:
    if user_path:
        return user_path
    path = DEFAULT_CKPTS.get(trident_encoder)
    if path and Path(path).exists():
        return path
    return None


def build_patch_encoder(args: argparse.Namespace):
    from trident.patch_encoder_models.load import encoder_factory

    trident_encoder = _encoder_name(args.encoder)
    ckpt = _ckpt_path(trident_encoder, args.patch_encoder_ckpt_path)
    base = encoder_factory(trident_encoder, weights_path=ckpt)
    return base, trident_encoder


def run_trident_extraction(args: argparse.Namespace, wsi_list: Path, coords_dir: str, enc_name: str, patch_encoder) -> None:
    from trident import Processor

    processor = Processor(
        job_dir=args.trident_job_dir,
        wsi_source=args.wsi_root,
        custom_list_of_wsis=str(wsi_list),
        skip_errors=args.skip_errors,
        max_workers=args.max_workers,
    )
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    processor.run_patch_feature_extraction_job(
        coords_dir=coords_dir,
        patch_encoder=patch_encoder,
        device=device,
        saveas="h5",
        batch_limit=args.batch_size,
    )


def _save_dtype(feats: torch.Tensor, dtype_name: str) -> torch.Tensor:
    if dtype_name == "float16":
        return feats.half()
    if dtype_name == "float32":
        return feats.float()
    raise ValueError(f"Unsupported save dtype: {dtype_name}")


def _materialize_for_save(feats: torch.Tensor, dtype_name: str) -> torch.Tensor:
    return _save_dtype(feats, dtype_name).detach().cpu().contiguous()


def _save_full_feature(feature_root: str, encoder: str, slide_id: str, feats: torch.Tensor, coords: torch.Tensor, dtype_name: str) -> None:
    out_dir = Path(feature_root) / encoder
    feat_path = out_dir / "features" / f"{slide_id}.pt"
    coord_path = out_dir / "coords" / f"{slide_id}.pt"
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    coord_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_materialize_for_save(feats, dtype_name), feat_path)
    torch.save(coords.detach().cpu().long().contiguous(), coord_path)


def normalize_outputs(args: argparse.Namespace, cohort: pd.DataFrame, coords_dir: str, enc_name: str, patch_encoder=None) -> list[str]:
    if args.mode != "full":
        raise ValueError("Only --mode full is supported.")
    feature_dir = Path(args.trident_job_dir) / coords_dir / f"features_{enc_name}"
    missing = []
    for _, row in cohort.drop_duplicates("slide_submitter_id").iterrows():
        slide_id = str(row["slide_submitter_id"])
        h5_path = feature_dir / f"{_slide_stem(row)}.h5"
        if not h5_path.exists():
            missing.append(str(h5_path))
            continue
        with h5py.File(h5_path, "r") as f:
            feats = torch.as_tensor(f["features"][:], dtype=torch.float32)
            coords = torch.as_tensor(f["coords"][:], dtype=torch.long)
        _save_full_feature(args.feature_root, args.output_encoder, slide_id, feats, _coords_to_grid(coords, args.patch_size), args.save_dtype)
        print(f"{slide_id}: normalized full {tuple(feats.shape)}")
    if missing:
        preview = "\n".join(missing[:5])
        if not getattr(args, "skip_errors", False):
            raise FileNotFoundError(f"Missing Trident feature files, first entries:\n{preview}")
        print(f"Skipped {len(missing)} missing Trident feature files, first entries:\n{preview}")
    return missing


def _prepend_trident_repo(trident_repo: str | None) -> None:
    if not trident_repo:
        return
    repo = Path(trident_repo)
    if not repo.exists():
        raise FileNotFoundError(f"Trident repo does not exist: {repo}")
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract full patch features via original Trident and normalize outputs.")
    parser.add_argument("--cohort_csv", required=True)
    parser.add_argument("--wsi_root", required=True)
    parser.add_argument("--trident_job_dir", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--encoder", default="uni2")
    parser.add_argument("--output_encoder", default="uni2")
    parser.add_argument("--mode", choices=["full"], default="full")
    parser.add_argument("--truncate_layer", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--mag", type=float, default=20.0)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_encoder_ckpt_path", default=None)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--save_dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--skip_errors", action="store_true")
    parser.add_argument("--normalize_only", action="store_true", help="Only convert existing Trident h5 files to project pt files.")
    parser.add_argument("--trident_repo", default="/mnt/Xsky/zqb/TRIDENT", help="Original Trident repository to import before any installed package.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _prepend_trident_repo(args.trident_repo)
    cohort = pd.read_csv(args.cohort_csv)
    for col in ("slide_submitter_id", "file_path", "file_name"):
        if col not in cohort.columns:
            raise ValueError(f"cohort_csv is missing required column: {col}")
    trident_job = Path(args.trident_job_dir)
    trident_job.mkdir(parents=True, exist_ok=True)
    coords_dir = _coords_dir_name(args.mag, args.patch_size, args.overlap)
    wsi_list = trident_job / "wsi_lists" / f"{Path(args.cohort_csv).stem}.csv"
    _write_wsi_list(cohort, Path(args.wsi_root), wsi_list)
    if args.normalize_only:
        patch_encoder = None
        enc_name = _feature_encoder_name(args)
    else:
        patch_encoder, enc_name = build_patch_encoder(args)
        run_trident_extraction(args, wsi_list, coords_dir, enc_name, patch_encoder)
    normalize_outputs(args, cohort, coords_dir, enc_name, patch_encoder)


if __name__ == "__main__":
    main()
