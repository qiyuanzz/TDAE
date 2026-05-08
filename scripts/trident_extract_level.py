from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import h5py
import pandas as pd
import torch
import torch.nn as nn


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
    trident_encoder = _encoder_name(args.encoder)
    if args.mode == "all":
        return f"{trident_encoder}_all_l{args.truncate_layer}_m{args.medium_layer}"
    if args.mode == "full":
        return trident_encoder
    layer = args.truncate_layer if args.mode == "light" else args.medium_layer
    return f"{trident_encoder}_{args.mode}_l{layer}"


def _ckpt_path(trident_encoder: str, user_path: str | None) -> str | None:
    if user_path:
        return user_path
    path = DEFAULT_CKPTS.get(trident_encoder)
    if path and Path(path).exists():
        return path
    return None


class ViTIntermediateModel(nn.Module):
    """Return CLS token after a selected transformer block for timm ViT-like encoders.

    The final ``model.norm`` LayerNorm is applied to the partial output so the
    intermediate feature lives in the same numerical space as Trident's full
    encoder output (which is post-norm via ``model.forward_features``). This
    keeps cosine distance / k-means / centroid statistics consistent across
    L1, L2, L3 features used by the Phase 0 allocator.
    """

    def __init__(self, model: nn.Module, layer: int) -> None:
        super().__init__()
        self.model = model
        self.layer = int(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.model
        if not all(hasattr(model, attr) for attr in ("patch_embed", "blocks")):
            raise RuntimeError("Intermediate extraction currently supports ViT-like encoders with patch_embed and blocks.")
        x = model.patch_embed(x)
        if hasattr(model, "_pos_embed"):
            x = model._pos_embed(x)
        else:
            if hasattr(model, "cls_token"):
                cls = model.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls, x), dim=1)
            if hasattr(model, "pos_embed"):
                x = x + model.pos_embed[:, : x.shape[1]]
        if hasattr(model, "patch_drop"):
            x = model.patch_drop(x)
        if hasattr(model, "norm_pre"):
            x = model.norm_pre(x)
        max_layer = min(self.layer, len(model.blocks))
        for block in model.blocks[:max_layer]:
            x = block(x)
        # Apply the encoder's final LayerNorm so intermediate CLS lives in the
        # same post-norm space as Trident's full feature output.
        if hasattr(model, "norm"):
            x = model.norm(x)
        if x.dim() == 3:
            return x[:, 0]
        if x.dim() == 4:
            return x.mean(dim=(1, 2))
        raise RuntimeError(f"Unsupported intermediate tensor shape: {tuple(x.shape)}")


class IntermediatePatchEncoder(nn.Module):
    def __init__(self, enc_name: str, model: nn.Module, transforms, precision: torch.dtype, embedding_dim: int | None = None) -> None:
        super().__init__()
        self.enc_name = enc_name
        self.model = model
        self.eval_transforms = transforms
        self.precision = precision
        if embedding_dim is not None:
            self.embedding_dim = int(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _cls_or_pool(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x[:, 0]
    if x.dim() == 4:
        return x.mean(dim=(1, 2))
    if x.dim() == 2:
        return x
    raise RuntimeError(f"Unsupported feature tensor shape: {tuple(x.shape)}")


class MultiLevelPatchEncoder(nn.Module):
    """Run the full ViT once and concatenate light/medium/full CLS features."""

    def __init__(self, enc_name: str, base, light_layer: int, medium_layer: int) -> None:
        super().__init__()
        self.enc_name = enc_name
        self.model = base.model
        self.eval_transforms = base.eval_transforms
        self.precision = getattr(base, "precision", torch.float32)
        self.light_layer = int(light_layer)
        self.medium_layer = int(medium_layer)
        self._captures: dict[int, torch.Tensor] = {}
        self.level_dims: tuple[int, int, int] | None = None
        if not hasattr(self.model, "blocks"):
            raise RuntimeError("Multi-level extraction currently supports ViT-like encoders with blocks.")
        n_blocks = len(self.model.blocks)
        for layer in (self.light_layer, self.medium_layer):
            if layer < 1 or layer > n_blocks:
                raise ValueError(f"Layer {layer} is outside encoder block range 1..{n_blocks}.")
            self.model.blocks[layer - 1].register_forward_hook(self._make_hook(layer))
        embedding_dim = getattr(base, "embedding_dim", None)
        if embedding_dim is not None:
            self.embedding_dim = int(embedding_dim) * 3

    def _make_hook(self, layer: int):
        # Apply the encoder's final LayerNorm to the captured intermediate
        # feature so light/medium are in the same post-norm space as the full
        # CLS token (Trident's default output is post-norm).
        norm = getattr(self.model, "norm", None)

        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if norm is not None:
                output = norm(output)
            self._captures[layer] = _cls_or_pool(output)

        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._captures = {}
        full = self.model(x)
        if isinstance(full, (tuple, list)):
            full = full[0]
        full = _cls_or_pool(full)
        light = self._captures.get(self.light_layer)
        medium = self._captures.get(self.medium_layer)
        if light is None or medium is None:
            raise RuntimeError("Failed to capture intermediate features during full forward.")
        self.level_dims = (int(light.shape[1]), int(medium.shape[1]), int(full.shape[1]))
        return torch.cat([light, medium, full], dim=1)


def build_patch_encoder(args: argparse.Namespace):
    from trident.patch_encoder_models.load import encoder_factory

    trident_encoder = _encoder_name(args.encoder)
    ckpt = _ckpt_path(trident_encoder, args.patch_encoder_ckpt_path)
    base = encoder_factory(trident_encoder, weights_path=ckpt)
    if args.mode == "all":
        enc_name = f"{trident_encoder}_all_l{args.truncate_layer}_m{args.medium_layer}"
        return MultiLevelPatchEncoder(enc_name, base, args.truncate_layer, args.medium_layer), enc_name
    if args.mode == "full":
        return base, trident_encoder
    layer = args.truncate_layer if args.mode == "light" else args.medium_layer
    model = ViTIntermediateModel(base.model, layer=layer)
    enc_name = f"{trident_encoder}_{args.mode}_l{layer}"
    wrapped = IntermediatePatchEncoder(
        enc_name=enc_name,
        model=model,
        transforms=base.eval_transforms,
        precision=getattr(base, "precision", torch.float32),
        embedding_dim=getattr(base, "embedding_dim", None),
    )
    return wrapped, enc_name


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


def _load_pca(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"light PCA path does not exist: {p}")
    try:
        import joblib

        return joblib.load(p)
    except Exception:
        with p.open("rb") as f:
            return pickle.load(f)


def _apply_light_pca(light: torch.Tensor, pca) -> torch.Tensor:
    if pca is None:
        return light
    device = light.device
    transformed = pca.transform(light.float().cpu().numpy())
    return torch.as_tensor(transformed, dtype=torch.float32, device=device)


def _split_multilevel_features(feats: torch.Tensor, patch_encoder) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dims = getattr(patch_encoder, "level_dims", None)
    if dims is not None and sum(int(v) for v in dims) == feats.shape[1]:
        d_light, d_medium, d_full = [int(v) for v in dims]
        light = feats[:, :d_light]
        medium = feats[:, d_light:d_light + d_medium]
        full = feats[:, d_light + d_medium:d_light + d_medium + d_full]
        return light, medium, full
    if feats.shape[1] % 3 != 0:
        raise ValueError(
            "Expected concatenated light/medium/full features. "
            f"Cannot infer level dims from shape {tuple(feats.shape)}; rerun extraction so hook metadata is available."
        )
    d = feats.shape[1] // 3
    return feats[:, :d], feats[:, d:2 * d], feats[:, 2 * d:]


def normalize_outputs(args: argparse.Namespace, cohort: pd.DataFrame, coords_dir: str, enc_name: str, patch_encoder=None) -> list[str]:
    feature_dir = Path(args.trident_job_dir) / coords_dir / f"features_{enc_name}"
    out_dir = Path(args.feature_root) / args.tdae_encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    light_pca = _load_pca(args.light_pca_path)
    for _, row in cohort.drop_duplicates("slide_submitter_id").iterrows():
        slide_id = str(row["slide_submitter_id"])
        h5_path = feature_dir / f"{_slide_stem(row)}.h5"
        if not h5_path.exists():
            missing.append(str(h5_path))
            continue
        with h5py.File(h5_path, "r") as f:
            feats = torch.as_tensor(f["features"][:], dtype=torch.float32)
            coords = torch.as_tensor(f["coords"][:], dtype=torch.long)
        if args.mode == "all":
            light, medium, full = _split_multilevel_features(feats, patch_encoder)
            light = _apply_light_pca(light, light_pca)
            torch.save(_materialize_for_save(light, args.save_dtype), out_dir / f"{slide_id}_light.pt")
            torch.save(_materialize_for_save(medium, args.save_dtype), out_dir / f"{slide_id}_medium.pt")
            torch.save(_materialize_for_save(full, args.save_dtype), out_dir / f"{slide_id}_full.pt")
            print(f"{slide_id}: normalized all light/medium/full {tuple(light.shape)} {tuple(medium.shape)} {tuple(full.shape)}")
        else:
            if args.mode == "light":
                feats = _apply_light_pca(feats, light_pca)
            torch.save(_materialize_for_save(feats, args.save_dtype), out_dir / f"{slide_id}_{args.mode}.pt")
            print(f"{slide_id}: normalized {args.mode} {tuple(feats.shape)}")
        torch.save(_coords_to_grid(coords, args.patch_size), out_dir / f"{slide_id}_coords.pt")
    if missing:
        preview = "\n".join(missing[:5])
        if not getattr(args, "skip_errors", False):
            raise FileNotFoundError(f"Missing Trident feature files, first entries:\n{preview}")
        print(f"Skipped {len(missing)} missing Trident feature files, first entries:\n{preview}")
    return missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract one TDAE feature level via Trident and normalize outputs.")
    parser.add_argument("--cohort_csv", required=True)
    parser.add_argument("--wsi_root", required=True)
    parser.add_argument("--trident_job_dir", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--encoder", default="uni2")
    parser.add_argument("--tdae_encoder", default="uni2")
    parser.add_argument("--mode", choices=["light", "medium", "full", "all"], required=True)
    parser.add_argument("--truncate_layer", type=int, default=3)
    parser.add_argument("--medium_layer", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--mag", type=float, default=20.0)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_encoder_ckpt_path", default=None)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--save_dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--light_pca_path", default=None, help="Optional joblib/pickle PCA fitted on light features.")
    parser.add_argument("--skip_errors", action="store_true")
    parser.add_argument("--normalize_only", action="store_true", help="Only convert existing Trident h5 files to TDAE pt files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
