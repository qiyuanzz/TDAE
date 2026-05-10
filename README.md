# WSI MIL Baselines

Full-feature MIL baselines for WSI survival experiments.

## Remote Defaults

- Project root: `/mnt/Xsky/zqb/TDAE`
- Training/evaluation Python: `/opt/conda/envs/trident/bin/python`
- Trident Python: `/opt/conda/envs/trident/bin/python`
- CSVs: `/mnt/Xsky/zqb/TDAE/metadata/csvs/TCGA_{BLCA,BRCA,GBMLGG,LUAD,UCEC}.csv`
- Feature bank: `/mnt/Xsky/zqb/TCGA_WSI_feature_bank/{cohort}/`
- Experiment results: `/mnt/Xsky/zqb/TDAE/outputs/`

## Current Method

The active feature bank uses original Trident full patch features only. MIL baselines read cached full features from the `features/` folder and patch coordinates from the `coords/` folder. The training code keeps only three MIL aggregators: `abmil`, `transmil`, and `clam`. Random patch drop is still controlled by `--patch_drop_ratio`.

## Generate Splits

```bash
cd /mnt/Xsky/zqb/TDAE
/opt/conda/envs/trident/bin/python scripts/step_03_generate_splits.py --cohort GBMLGG --task classification
/opt/conda/envs/trident/bin/python scripts/step_03_generate_splits.py --cohort BRCA --task classification
/opt/conda/envs/trident/bin/python scripts/step_03_generate_splits.py --cohort UCEC --task classification
/opt/conda/envs/trident/bin/python scripts/step_03_generate_splits.py --cohort GBMLGG --task survival
```

## Feature Extraction

Trident is installed in the isolated environment:

```bash
/opt/conda/envs/trident/bin/trident --help
/opt/conda/envs/trident/bin/trident doctor
```

The project-normalized full-only feature layout is:

```text
/mnt/Xsky/zqb/TCGA_WSI_feature_bank/{cohort}/features/{encoder}/features/{slide_id}.pt
/mnt/Xsky/zqb/TCGA_WSI_feature_bank/{cohort}/features/{encoder}/coords/{slide_id}.pt
```

Real Trident extraction is wrapped by the project scripts. When using one visible GPU, set `CUDA_VISIBLE_DEVICES` and pass `--gpu 0` to Trident:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/trident/bin/python scripts/step_01_extract_patches.py \
  --cohort_csv metadata/csvs/TCGA_GBMLGG.csv \
  --output_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/GBMLGG/preprocess \
  --backend trident \
  --wsi_root /mnt/Archive/Dataset/GDC_DATA/GDC_DATA \
  --patch_size 224 \
  --mag 20 \
  --gpu 0

CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/trident/bin/python scripts/step_02_extract_features.py \
  --backend trident \
  --mode full \
  --encoder uni2 \
  --cohort_csv metadata/csvs/TCGA_GBMLGG.csv \
  --patch_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/GBMLGG/preprocess \
  --feature_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/GBMLGG/features \
  --wsi_root /mnt/Archive/Dataset/GDC_DATA/GDC_DATA \
  --trident_python /opt/conda/envs/trident/bin/python \
  --trident_repo /mnt/Xsky/zqb/TRIDENT \
  --gpu 0 \
  --batch_size 64 \
  --save_dtype float32
```

For plumbing checks without loading a real encoder:

```bash
/opt/conda/envs/trident/bin/python scripts/step_02_extract_features.py --smoke --mode full   --cohort_csv metadata/csvs/TCGA_GBMLGG.csv --patch_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/_smoke/patches --feature_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/_smoke/features
```

## Train And Evaluate

```bash
CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/trident/bin/python scripts/step_04_train.py \
  --config trainer/configs/default.yaml \
  --cohort GBMLGG \
  --task survival \
  --encoder uni2 \
  --aggregator abmil \
  --patch_drop_ratio 0.0 \
  --fold 0 \
  --device cuda

CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/trident/bin/python scripts/step_05_evaluate.py \
  --checkpoint outputs/checkpoints/GBMLGG/survival/uni2/abmil_random_keep/wsi_ABMIL_keep_1_gc_32_lr_0.0002/seed_1/fold_0/abmil_best.pt \
  --config trainer/configs/default.yaml \
  --cohort GBMLGG \
  --task survival \
  --encoder uni2 \
  --aggregator abmil \
  --fold 0 \
  --device cuda
```

## Smoke Check

A CUDA smoke fixture uses `trainer/configs/smoke.yaml`, `metadata/splits/GBMLGG/classification/fold_99.csv`, and `/mnt/Xsky/zqb/TCGA_WSI_feature_bank/_smoke/features/uni2/`. It is only for plumbing verification.
