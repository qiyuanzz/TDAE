# WSI MIL Baselines

Full-feature MIL baselines for TCGA WSI survival experiments.

## Project Layout

```text
trainer/   Training code, datasets, losses, metrics, tests, and YAML configs.
models/    MIL aggregators: ABMIL, TransMIL, and CLAM.
metadata/  Local cohort CSVs and fold split CSVs. Ignored by Git.
scripts/   Step scripts and run scripts.
outputs/   Local logs, checkpoints, histories, and summaries. Ignored by Git.
```

`metadata/` is intentionally local-only. It contains cohort tables and generated splits needed to run experiments on this machine, but it should not be uploaded to GitHub. The repository tracks the code and configs, not TCGA metadata copies or generated split files.

## Remote Defaults

- Project root: `/mnt/Xsky/zqb/TDAE`
- Training/evaluation Python: `/opt/conda/envs/trident/bin/python`
- Original Trident repo: `/mnt/Xsky/zqb/TRIDENT`
- WSI root: `/mnt/Archive/Dataset/GDC_DATA/GDC_DATA`
- Feature bank: `/mnt/Xsky/zqb/TCGA_WSI_feature_bank/{cohort}/`
- Local metadata: `/mnt/Xsky/zqb/TDAE/metadata/`
- Local experiment outputs: `/mnt/Xsky/zqb/TDAE/outputs/`

## Current Method

The active pipeline uses original Trident and stores only full patch features. There are no TDAE, light, or medium feature branches in the active MIL training path.

The normalized feature layout is:

```text
/mnt/Xsky/zqb/TCGA_WSI_feature_bank/{cohort}/features/{encoder}/features/{slide_id}.pt
/mnt/Xsky/zqb/TCGA_WSI_feature_bank/{cohort}/features/{encoder}/coords/{slide_id}.pt
```

MIL training currently supports only:

```text
abmil
transmil
clam
```

Random patch drop is retained and controlled by `--patch_drop_ratio`. Survival labels use global fold-level time discretization in `trainer/datasets/feature_dataset.py`: the fold dataset is built first, time bins are fit from event cases across the fold, then train/val dataset views share those same cutpoints.

## Patch And Feature Extraction

Run patch and full-feature extraction for BRCA, GBMLGG, and COADREAD on GPUs 4-7:

```bash
cd /mnt/Xsky/zqb/TDAE
GPUS="4 5 6 7" COHORTS="BRCA GBMLGG COADREAD" \
  bash scripts/run_patch_feature_all3.sh
```

Single-cohort extraction can also be run directly:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/trident/bin/python scripts/step_01_extract_patches.py \
  --cohort_csv metadata/csvs/TCGA_GBMLGG_survival_dx.csv \
  --output_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/GBMLGG/preprocess \
  --backend trident \
  --wsi_root /mnt/Archive/Dataset/GDC_DATA/GDC_DATA \
  --trident_python /opt/conda/envs/trident/bin/python \
  --trident_repo /mnt/Xsky/zqb/TRIDENT \
  --patch_size 224 \
  --mag 20 \
  --gpu 0

CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/trident/bin/python scripts/step_02_extract_features.py \
  --backend trident \
  --mode full \
  --encoder uni2 \
  --cohort_csv metadata/csvs/TCGA_GBMLGG_survival_dx.csv \
  --patch_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/GBMLGG/preprocess \
  --feature_root /mnt/Xsky/zqb/TCGA_WSI_feature_bank/GBMLGG/features \
  --wsi_root /mnt/Archive/Dataset/GDC_DATA/GDC_DATA \
  --trident_python /opt/conda/envs/trident/bin/python \
  --trident_repo /mnt/Xsky/zqb/TRIDENT \
  --gpu 0 \
  --batch_size 64 \
  --save_dtype float32
```

## Generate Splits

Splits are case-level five-fold survival splits and are written under `metadata/splits/{COHORT}/survival/`.

```bash
cd /mnt/Xsky/zqb/TDAE
/opt/conda/envs/trident/bin/python scripts/step_03_generate_splits.py \
  --cohort BRCA \
  --task survival \
  --n_splits 5 \
  --seed 1
```

## Train One Run

```bash
CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/trident/bin/python scripts/step_04_train.py \
  --config trainer/configs/default.yaml \
  --cohort GBMLGG \
  --task survival \
  --encoder uni2 \
  --aggregator abmil \
  --patch_drop_ratio 0.0 \
  --patch_drop_seed 1 \
  --seed 1 \
  --fold 0 \
  --epochs 20 \
  --device cuda
```

With `EARLY_STOPPING_PATIENCE=0`, the reported `summary.json` uses the final epoch. `abmil_best.pt` is kept for compatibility and is also the final epoch model in this setting.

## Current Patch-Drop Sweep

The active patch-drop sweep uses ABMIL only, cohorts `BRCA GBMLGG COADREAD`, seeds `1 2 3 4 5`, five folds, and eight keep ratios:

```text
keep: 1.00 0.90 0.70 0.50 0.30 0.10 0.05 0.01
drop: 0    0.10 0.30 0.50 0.70 0.90 0.95 0.99
```

Launch on GPUs 4-7:

```bash
cd /mnt/Xsky/zqb/TDAE
GPUS="4 5 6 7" COHORTS="BRCA GBMLGG COADREAD" \
  bash scripts/run_all3_abmil_patch_drop_keep_1_to_001.sh
```

The script first refreshes `metadata/csvs/TCGA_{COHORT}_survival_dx_extracted.csv`, regenerates case-level survival splits, then runs training. Logs are under `outputs/logs/abmil_patch_drop/`; checkpoints and summaries are under `outputs/checkpoints/{COHORT}/survival/uni2/`.

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/trident/bin/python scripts/step_05_evaluate.py \
  --checkpoint outputs/checkpoints/GBMLGG/survival/uni2/<experiment>/<setting>/seed_1/fold_0/abmil_best.pt \
  --config trainer/configs/default.yaml \
  --cohort GBMLGG \
  --task survival \
  --encoder uni2 \
  --aggregator abmil \
  --fold 0 \
  --device cuda
```

## Verification

```bash
cd /mnt/Xsky/zqb/TDAE
/opt/conda/envs/trident/bin/python -m compileall scripts trainer models -q
/opt/conda/envs/trident/bin/python -m pytest trainer/tests -q
```
