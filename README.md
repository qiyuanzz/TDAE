# TDAE

Task-Driven Adaptive Encoding for efficient WSI feature extraction.

## Remote Defaults

- Project root: `/mnt/Xsky/zqb/TDAE`
- Training/evaluation Python: `/opt/conda/envs/trident/bin/python`
- Trident Python: `/opt/conda/envs/trident/bin/python`
- CSVs: `/mnt/Xsky/zqb/TDAE/data/csvs/TCGA_{BLCA,BRCA,GBMLGG,LUAD,UCEC}.csv`
- Feature bank: `/mnt/Xsky/zqb/TCGA_WSI_feature_bank/{cohort}/`
- Experiment results: `/mnt/Xsky/zqb/TDAE/outputs/`

## Current Method

The main model follows the updated plan: 4-level gating over `L0=skip`, `L1=light`, `L2=medium`, `L3=full`, then level-aware graph propagation and ABMIL aggregation. Training reads cached `{slide_id}_light.pt`, `{slide_id}_medium.pt`, `{slide_id}_full.pt`, and `{slide_id}_coords.pt`.

## Generate Splits

```bash
cd /mnt/Xsky/zqb/TDAE
/opt/conda/envs/trident/bin/python scripts/generate_splits.py --cohort GBMLGG --task classification
/opt/conda/envs/trident/bin/python scripts/generate_splits.py --cohort BRCA --task classification
/opt/conda/envs/trident/bin/python scripts/generate_splits.py --cohort UCEC --task classification
/opt/conda/envs/trident/bin/python scripts/generate_splits.py --cohort GBMLGG --task survival
```

## Feature Extraction

Trident is installed in the isolated environment:

```bash
/opt/conda/envs/trident/bin/trident --help
/opt/conda/envs/trident/bin/trident doctor
```

The TDAE scripts expect project-normalized feature names:

```text
/mnt/Xsky/zqb/TCGA_WSI_feature_bank/{cohort}/features/{encoder}/{slide_id}_light.pt
/mnt/Xsky/zqb/TCGA_WSI_feature_bank/{cohort}/features/{encoder}/{slide_id}_medium.pt
/mnt/Xsky/zqb/TCGA_WSI_feature_bank/{cohort}/features/{encoder}/{slide_id}_full.pt
/mnt/Xsky/zqb/TCGA_WSI_feature_bank/{cohort}/features/{encoder}/{slide_id}_coords.pt
```

Real Trident extraction is wrapped by the project scripts. When using one visible GPU, set `CUDA_VISIBLE_DEVICES` and pass `--gpu 0` to Trident:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/trident/bin/python scripts/00_extract_patches.py \
  --cohort_csv data/csvs/TCGA_GBMLGG.csv \
  --output_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/GBMLGG/preprocess \
  --backend trident \
  --wsi_root /mnt/Archive/Dataset/GDC_DATA/GDC_DATA \
  --patch_size 224 \
  --mag 20 \
  --gpu 0

CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/trident/bin/python scripts/01_extract_features.py \
  --backend trident \
  --mode all \
  --encoder uni2 \
  --cohort_csv data/csvs/TCGA_GBMLGG.csv \
  --patch_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/GBMLGG/preprocess \
  --feature_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/GBMLGG/features \
  --wsi_root /mnt/Archive/Dataset/GDC_DATA/GDC_DATA \
  --trident_python /opt/conda/envs/trident/bin/python \
  --gpu 0 \
  --batch_size 64 \
  --save_dtype float32
```

For plumbing checks without loading a real encoder:

```bash
/opt/conda/envs/trident/bin/python scripts/01_extract_features.py --smoke --mode light  --cohort_csv data/csvs/TCGA_GBMLGG.csv --patch_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/_smoke/patches --feature_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/_smoke/features
/opt/conda/envs/trident/bin/python scripts/01_extract_features.py --smoke --mode medium --cohort_csv data/csvs/TCGA_GBMLGG.csv --patch_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/_smoke/patches --feature_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/_smoke/features
/opt/conda/envs/trident/bin/python scripts/01_extract_features.py --smoke --mode full   --cohort_csv data/csvs/TCGA_GBMLGG.csv --patch_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/_smoke/patches --feature_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/_smoke/features
```

## Train And Evaluate

```bash
CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/trident/bin/python scripts/03_train.py \
  --config configs/default.yaml \
  --cohort GBMLGG \
  --task classification \
  --encoder uni2 \
  --fold 0 \
  --device cuda

CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/trident/bin/python scripts/04_evaluate.py \
  --checkpoint outputs/checkpoints/GBMLGG/classification/uni2/fold_0/tdae_last.pt \
  --config configs/default.yaml \
  --cohort GBMLGG \
  --task classification \
  --encoder uni2 \
  --fold 0 \
  --measure_efficiency \
  --device cuda
```

## Smoke Check

A CUDA smoke fixture uses `configs/smoke.yaml`, `data/splits/GBMLGG/classification/fold_99.csv`, and `/mnt/Xsky/zqb/TCGA_WSI_feature_bank/_smoke/features/uni2/`. It is only for plumbing verification.
