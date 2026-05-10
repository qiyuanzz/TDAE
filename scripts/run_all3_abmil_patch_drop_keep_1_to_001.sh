#!/usr/bin/env bash
set -euo pipefail

cd /mnt/Xsky/zqb/TDAE

PY=${PY:-/opt/conda/envs/trident/bin/python}

# User-requested training cohorts and GPUs. Default to GPU 4 and 5; override with GPUS="...".
COHORTS=(${COHORTS:-BRCA GBMLGG COADREAD})
GPUS_FIXED="${GPUS:-4 5}"

FOLDS_DEFAULT="${FOLDS:-0 1 2 3 4}"
SEEDS_DEFAULT="1 2 3 4 5"

# Patch-drop sweep is expressed as keep ratios from 1.00 down to 0.01.
# step_04_train.py receives patch_drop_ratio = 1 - keep.
KEEPS_DEFAULT="${KEEPS:-1.00 0.90 0.70 0.50 0.30 0.10 0.05 0.01}"
DROPS_DEFAULT="${DROPS:-0 0.10 0.30 0.50 0.70 0.90 0.95 0.99}"

SPLIT_SEED=${SPLIT_SEED:-1}
N_SPLITS=${N_SPLITS:-5}
EPOCHS_DEFAULT="${EPOCHS:-20}"
LR_DEFAULT="${LR:-2e-4}"
REG_DEFAULT="${REG:-1e-5}"
SURVIVAL_BATCH_SIZE_DEFAULT="${SURVIVAL_BATCH_SIZE:-32}"
CPU_THREADS_DEFAULT="${CPU_THREADS:-2}"
CACHE_MODE_DEFAULT="${CACHE_MODE:-auto}"
BAG_LOSS_DEFAULT="${BAG_LOSS:-nll_surv}"
N_CLASSES_DEFAULT="${N_CLASSES:-4}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

RUN_ROOT="outputs/logs/abmil_patch_drop"
RUN_ID="${RUN_ID:-abmil_patch_drop_keep_1_to_001_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$RUN_ROOT/$RUN_ID"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"
: > "$MASTER_LOG"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

prepare_extracted_csv() {
  local cohort="$1"
  "$PY" - "$cohort" <<'PY'
from pathlib import Path
import sys
import pandas as pd

cohort = sys.argv[1]
root = Path("/mnt/Xsky/zqb/TDAE")
feature_bank = Path("/mnt/Xsky/zqb/TCGA_WSI_feature_bank") / cohort
src = root / "metadata" / "csvs" / f"TCGA_{cohort}_survival_dx.csv"
out = root / "metadata" / "csvs" / f"TCGA_{cohort}_survival_dx_extracted.csv"
feat_dir = feature_bank / "features" / "uni2" / "features"
coord_dir = feature_bank / "features" / "uni2" / "coords"

if not src.exists():
    raise SystemExit(f"Missing cohort CSV: {src}")
if not feat_dir.exists() or not coord_dir.exists():
    raise SystemExit(f"{cohort}: feature dirs are not ready yet: {feat_dir} / {coord_dir}")

feature_ids = {path.stem for path in feat_dir.glob("*.pt")}
coord_ids = {path.stem for path in coord_dir.glob("*.pt")}
ready_ids = feature_ids & coord_ids
if not ready_ids:
    raise SystemExit(f"{cohort}: no paired feature/coord .pt files found yet.")

df = pd.read_csv(src)
if "slide_submitter_id" not in df.columns:
    raise SystemExit(f"{src} is missing slide_submitter_id")
before = df["slide_submitter_id"].astype(str).nunique()
out_df = df[df["slide_submitter_id"].astype(str).isin(ready_ids)].drop_duplicates("slide_submitter_id")
after = out_df["slide_submitter_id"].astype(str).nunique()
if after == 0:
    raise SystemExit(f"{cohort}: extracted CSV would be empty.")
out_df.to_csv(out, index=False)
print(f"{cohort}: wrote {out} slides={after} missing={before - after}")
PY
}

generate_splits() {
  local cohort="$1"
  "$PY" scripts/step_03_generate_splits.py \
    --cohort "$cohort" \
    --task survival \
    --n_splits "$N_SPLITS" \
    --seed "$SPLIT_SEED" \
    --out_dir "metadata/splits/${cohort}/survival"
}

config_for_cohort() {
  local cohort="$1"
  if [ "$cohort" = "BRCA" ]; then
    echo "trainer/configs/brca_uni2_mil.yaml"
  else
    echo "trainer/configs/default.yaml"
  fi
}

run_cohort() {
  local cohort="$1"
  local config
  config="$(config_for_cohort "$cohort")"
  local cohort_lower
  cohort_lower="$(printf '%s' "$cohort" | tr '[:upper:]' '[:lower:]')"
  local experiment_tag="${cohort_lower}_uni2_abmil_patch_drop_keep_1_to_001_seed1_5"

  log "prepare cohort=${cohort}"
  prepare_extracted_csv "$cohort" | tee -a "$MASTER_LOG"
  generate_splits "$cohort" | tee -a "$MASTER_LOG"
  if [ "$PREPARE_ONLY" = "1" ]; then
    log "prepare-only cohort=${cohort} done"
    return 0
  fi

  log "train cohort=${cohort} config=${config} experiment=${experiment_tag}"
  COHORT="$cohort" \
  CONFIG="$config" \
  TASK=survival \
  ENCODER=uni2 \
  AGGREGATOR=abmil \
  EXPERIMENT_TAG="$experiment_tag" \
  GPUS="$GPUS_FIXED" \
  FOLDS="$FOLDS_DEFAULT" \
  SEEDS="$SEEDS_DEFAULT" \
  KEEPS="$KEEPS_DEFAULT" \
  DROPS="$DROPS_DEFAULT" \
  EPOCHS="$EPOCHS_DEFAULT" \
  LR="$LR_DEFAULT" \
  REG="$REG_DEFAULT" \
  SURVIVAL_BATCH_SIZE="$SURVIVAL_BATCH_SIZE_DEFAULT" \
  CPU_THREADS="$CPU_THREADS_DEFAULT" \
  CACHE_MODE="$CACHE_MODE_DEFAULT" \
  BAG_LOSS="$BAG_LOSS_DEFAULT" \
  N_CLASSES="$N_CLASSES_DEFAULT" \
  bash scripts/run_abmil_random_keep_seed_layout.sh
  log "done cohort=${cohort} experiment=${experiment_tag}"
}

log "started run_id=${RUN_ID} cohorts=${COHORTS[*]} gpus=${GPUS_FIXED} seeds=${SEEDS_DEFAULT} keeps=${KEEPS_DEFAULT} prepare_only=${PREPARE_ONLY}"
for cohort in "${COHORTS[@]}"; do
  run_cohort "$cohort"
done
log "finished run_id=${RUN_ID}"
