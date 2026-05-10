from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
import h5py
import torch

from trainer.io import coords_to_grid, ensure_columns, read_table


def simple_tissue_coords(wsi_path: str, patch_size: int, max_patches: int | None = None) -> torch.Tensor:
    try:
        import openslide
    except Exception as exc:
        raise RuntimeError("openslide-python is required for patch coordinate extraction.") from exc
    slide = openslide.OpenSlide(wsi_path)
    width, height = slide.dimensions
    coords = []
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            thumb = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB").resize((32, 32))
            tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(thumb.tobytes())).view(32, 32, 3).float()
            brightness = tensor.mean().item()
            saturation = (tensor.max(dim=2).values - tensor.min(dim=2).values).mean().item()
            if brightness < 245 and saturation > 8:
                coords.append((x, y))
                if max_patches and len(coords) >= max_patches:
                    return torch.tensor(coords, dtype=torch.long)
    return torch.tensor(coords, dtype=torch.long)


def _coords_dir_name(mag: float, patch_size: int, overlap: int) -> str:
    mag_str = f"{float(mag):g}"
    return f"{mag_str}x_{patch_size}px_{overlap}px_overlap"


def _slide_stem(row) -> str:
    if "file_path" in row and str(row["file_path"]) and str(row["file_path"]) != "nan":
        return Path(str(row["file_path"])).stem
    return Path(str(row["file_name"])).stem


def _write_trident_wsi_list(cohort, wsi_root: Path, out_csv: Path) -> None:
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


def _run_trident_task(args: argparse.Namespace, task: str, list_csv: Path, coords_dir: str) -> None:
    cmd = [
        args.trident_python,
        str(Path(args.trident_repo) / "run_batch_of_slides.py"),
        "--task",
        task,
        "--wsi_dir",
        args.wsi_root,
        "--custom_list_of_wsis",
        str(list_csv),
        "--job_dir",
        str(Path(args.output_root) / "trident"),
        "--gpu",
        str(args.gpu),
        "--mag",
        str(args.mag),
        "--patch_size",
        str(args.patch_size),
        "--overlap",
        str(args.overlap),
        "--coords_dir",
        coords_dir,
        "--segmenter",
        args.segmenter,
        "--batch_size",
        str(args.batch_size),
    ]
    if args.skip_errors:
        cmd.append("--skip_errors")
    if args.max_workers is not None:
        cmd.extend(["--max_workers", str(args.max_workers)])
    subprocess.run(cmd, check=True)


def _normalize_trident_coords(cohort, output_root: Path, coords_dir: str, patch_size: int, target_mpp: float, skip_missing: bool = False) -> list[str]:
    patch_out = output_root / "patches"
    patch_out.mkdir(parents=True, exist_ok=True)
    trident_patches = output_root / "trident" / coords_dir / "patches"
    missing = []
    for _, row in cohort.drop_duplicates("slide_submitter_id").iterrows():
        slide_id = str(row["slide_submitter_id"])
        h5_path = trident_patches / f"{_slide_stem(row)}_patches.h5"
        if not h5_path.exists():
            missing.append(str(h5_path))
            continue
        with h5py.File(h5_path, "r") as f:
            coords = torch.as_tensor(f["coords"][:], dtype=torch.long)
        grid = coords_to_grid(coords, patch_size=patch_size)
        torch.save(
            {"coords": coords, "grid_indices": grid, "patch_size": patch_size, "target_mpp": target_mpp},
            patch_out / f"{slide_id}.pt",
        )
        print(f"{slide_id}: normalized {coords.shape[0]} Trident coords")
    if missing:
        preview = "\n".join(missing[:5])
        if not skip_missing:
            raise FileNotFoundError(f"Missing Trident coordinate files, first entries:\n{preview}")
        print(f"Skipped {len(missing)} missing Trident coordinate files; first entries:\n{preview}")
    return missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract tissue patch coordinates from cohort WSI files.")
    parser.add_argument("--cohort_csv", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--target_mpp", type=float, default=0.5)
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument("--backend", choices=["trident", "simple"], default="trident")
    parser.add_argument("--wsi_root", default="/mnt/Archive/Dataset/GDC_DATA/GDC_DATA")
    parser.add_argument("--trident_python", default="/opt/conda/envs/trident/bin/python")
    parser.add_argument("--trident_repo", default="/mnt/Xsky/zqb/TRIDENT")
    parser.add_argument("--trident_tasks", nargs="+", default=["seg", "coords"], choices=["seg", "coords"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mag", type=float, default=20.0)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--segmenter", default="hest", choices=["hest", "grandqc", "otsu"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--skip_errors", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cohort = read_table(args.cohort_csv)
    ensure_columns(cohort, ["slide_submitter_id", "file_path", "file_name"], "cohort_csv")
    cohort = _apply_slide_shard(cohort, args.num_shards, args.shard_index)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.backend == "trident":
        coords_dir = _coords_dir_name(args.mag, args.patch_size, args.overlap)
        list_name = f"{Path(args.cohort_csv).stem}.csv" if int(args.num_shards) <= 1 else f"{Path(args.cohort_csv).stem}_shard{args.shard_index}of{args.num_shards}.csv"
        list_csv = output_root / "trident" / "wsi_lists" / list_name
        _write_trident_wsi_list(cohort, Path(args.wsi_root), list_csv)
        for task in args.trident_tasks:
            _run_trident_task(args, task, list_csv, coords_dir)
        _normalize_trident_coords(cohort, output_root, coords_dir, args.patch_size, args.target_mpp, skip_missing=args.skip_errors)
        return

    out = output_root / "patches"
    out.mkdir(parents=True, exist_ok=True)
    for _, row in cohort.drop_duplicates("slide_submitter_id").iterrows():
        slide_id = str(row["slide_submitter_id"])
        coords = simple_tissue_coords(str(row["file_path"]), patch_size=args.patch_size, max_patches=args.max_patches)
        grid = coords_to_grid(coords, patch_size=args.patch_size)
        torch.save({"coords": coords, "grid_indices": grid, "patch_size": args.patch_size, "target_mpp": args.target_mpp}, out / f"{slide_id}.pt")
        print(f"{slide_id}: {coords.shape[0]} patches")


if __name__ == "__main__":
    main()
