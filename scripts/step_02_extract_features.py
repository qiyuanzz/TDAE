from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from data.utils import ensure_columns, read_table, save_feature_triplet
from models.encoder_wrapper import build_encoder


def _pseudo_patches(coords: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            coords[:, 0].float() / max(float(coords[:, 0].max().item()), 1.0),
            coords[:, 1].float() / max(float(coords[:, 1].max().item()), 1.0),
            torch.ones(coords.shape[0]),
        ],
        dim=1,
    ).view(coords.shape[0], 3, 1, 1)


def _save_mode(feature_root: str, encoder: str, slide_id: str, mode: str, feats: torch.Tensor, coords: torch.Tensor) -> None:
    kwargs = {"coords": coords}
    if mode == "full":
        kwargs["full_feats"] = feats
    elif mode == "medium":
        kwargs["medium_feats"] = feats
    elif mode == "light":
        kwargs["light_feats"] = feats
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    save_feature_triplet(feature_root, encoder, slide_id, **kwargs)


def _apply_slide_shard(cohort, num_shards: int, shard_index: int):
    if int(num_shards) <= 1:
        return cohort
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f'shard_index must be in [0, {num_shards}), got {shard_index}')
    slide_ids = cohort.drop_duplicates("slide_submitter_id")["slide_submitter_id"].astype(str).tolist()
    selected = {slide_id for pos, slide_id in enumerate(slide_ids) if pos % num_shards == shard_index}
    sharded = cohort[cohort["slide_submitter_id"].astype(str).isin(selected)].copy()
    print(f"Using shard {shard_index}/{num_shards}: {len(selected)} slides")
    return sharded


def _materialize_sharded_cohort(args: argparse.Namespace, trident_job_dir: Path) -> str:
    if int(args.num_shards) <= 1:
        return args.cohort_csv
    cohort = read_table(args.cohort_csv)
    ensure_columns(cohort, ["slide_submitter_id"], "cohort_csv")
    cohort = _apply_slide_shard(cohort, args.num_shards, args.shard_index)
    out_csv = trident_job_dir / "wsi_lists" / "shards" / f"{Path(args.cohort_csv).stem}_shard{args.shard_index}of{args.num_shards}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out_csv, index=False)
    return str(out_csv)


def _run_smoke_or_local(args: argparse.Namespace) -> None:
    if args.mode == "all":
        raise ValueError("--mode all is only supported with --backend trident.")
    cohort = read_table(args.cohort_csv)
    ensure_columns(cohort, ["slide_submitter_id"], "cohort_csv")
    cohort = _apply_slide_shard(cohort, args.num_shards, args.shard_index)
    if args.mode == "light":
        smoke_dim = 8
        layer = args.truncate_layer
    elif args.mode == "medium":
        smoke_dim = 16
        layer = args.medium_layer
    else:
        smoke_dim = 16
        layer = args.medium_layer
    encoder_name = "identity" if args.smoke else args.encoder
    encoder = build_encoder(encoder_name, mode=args.mode, truncate_layer=layer, smoke_dim=smoke_dim)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    encoder.eval().to(device)
    for _, row in cohort.drop_duplicates("slide_submitter_id").iterrows():
        slide_id = str(row["slide_submitter_id"])
        patch_file = Path(args.patch_root) / "patches" / f"{slide_id}.pt"
        patch_data = torch.load(patch_file, map_location="cpu")
        coords = patch_data["grid_indices"].long()
        feats = []
        pseudo = _pseudo_patches(coords)
        with torch.no_grad():
            for start in range(0, pseudo.shape[0], args.batch_size):
                feats.append(encoder(pseudo[start:start + args.batch_size].to(device)).cpu())
        feats_t = torch.cat(feats, dim=0)
        _save_mode(args.feature_root, args.encoder, slide_id, args.mode, feats_t, coords)
        print(f"{slide_id}: saved {args.mode} {tuple(feats_t.shape)}")


def _run_trident(args: argparse.Namespace) -> None:
    trident_job_dir = Path(args.trident_job_dir) if args.trident_job_dir else Path(args.patch_root) / "trident"
    cohort_csv = _materialize_sharded_cohort(args, trident_job_dir)
    cmd = [
        args.trident_python,
        str(Path(__file__).with_name("trident_extract_level.py")),
        "--cohort_csv",
        cohort_csv,
        "--wsi_root",
        args.wsi_root,
        "--trident_job_dir",
        str(trident_job_dir),
        "--feature_root",
        args.feature_root,
        "--encoder",
        args.trident_encoder or args.encoder,
        "--tdae_encoder",
        args.encoder,
        "--mode",
        args.mode,
        "--truncate_layer",
        str(args.truncate_layer),
        "--medium_layer",
        str(args.medium_layer),
        "--patch_size",
        str(args.patch_size),
        "--mag",
        str(args.mag),
        "--overlap",
        str(args.overlap),
        "--gpu",
        str(args.gpu),
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--save_dtype",
        args.save_dtype,
    ]
    if args.light_pca_path:
        cmd.extend(["--light_pca_path", args.light_pca_path])
    if args.patch_encoder_ckpt_path:
        cmd.extend(["--patch_encoder_ckpt_path", args.patch_encoder_ckpt_path])
    if args.max_workers is not None:
        cmd.extend(["--max_workers", str(args.max_workers)])
    if args.skip_errors:
        cmd.append("--skip_errors")
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract one TDAE feature level from existing patch coords.")
    parser.add_argument("--encoder", default="uni2")
    parser.add_argument("--mode", default="full", choices=["light", "medium", "full", "all"])
    parser.add_argument("--truncate_layer", type=int, default=3)
    parser.add_argument("--medium_layer", type=int, default=12)
    parser.add_argument("--cohort_csv", required=True)
    parser.add_argument("--patch_root", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--light_pca_path", default=None, help="Optional PCA model for light feature compression.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Generate deterministic fixture features instead of loading a real encoder.")
    parser.add_argument("--backend", choices=["trident", "local"], default="trident")
    parser.add_argument("--wsi_root", default="/mnt/Archive/Dataset/GDC_DATA/GDC_DATA")
    parser.add_argument("--trident_python", default="/opt/conda/envs/trident/bin/python")
    parser.add_argument("--trident_job_dir", default=None)
    parser.add_argument("--trident_encoder", default=None, help="Override Trident encoder name, e.g. uni_v2.")
    parser.add_argument("--patch_encoder_ckpt_path", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mag", type=float, default=20.0)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--skip_errors", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke or args.backend == "local":
        _run_smoke_or_local(args)
    else:
        _run_trident(args)


if __name__ == "__main__":
    main()
