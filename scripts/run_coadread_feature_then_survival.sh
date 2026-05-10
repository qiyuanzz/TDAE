#!/usr/bin/env bash
set -u

cd /mnt/Xsky/zqb/TDAE || exit 1

PY=${PY:-/opt/conda/envs/trident/bin/python}
TRIDENT_REPO=${TRIDENT_REPO:-/mnt/Xsky/zqb/TRIDENT}
GPUS=(${GPUS:-4 5 6 7})
SHARDS=(${SHARDS:-0 1 2 3})
NUM_SHARDS=${NUM_SHARDS:-4}
PATCH_MAX_WORKERS=${PATCH_MAX_WORKERS:-8}
FEATURE_MAX_WORKERS=${FEATURE_MAX_WORKERS:-4}
PATCH_BATCH_SIZE=${PATCH_BATCH_SIZE:-64}
FEATURE_BATCH_SIZE=${FEATURE_BATCH_SIZE:-64}

COADREAD_CSV=${COADREAD_CSV:-metadata/csvs/TCGA_COADREAD_survival_dx.csv}
COADREAD_EXTRACTED_CSV=${COADREAD_EXTRACTED_CSV:-metadata/csvs/TCGA_COADREAD_survival_dx_extracted.csv}
COADREAD_PATCH_ROOT=${COADREAD_PATCH_ROOT:-/mnt/Xsky/zqb/TCGA_WSI_feature_bank/COADREAD/preprocess}
COADREAD_FEATURE_ROOT=${COADREAD_FEATURE_ROOT:-/mnt/Xsky/zqb/TCGA_WSI_feature_bank/COADREAD/features}
WSI_ROOT=${WSI_ROOT:-/mnt/Archive/Dataset/GDC_DATA/GDC_DATA}
ENCODER=${ENCODER:-uni2}

SURVIVAL_EXPERIMENT_TAG=${SURVIVAL_EXPERIMENT_TAG:-abmil_random_keep}
KEEPS_VALUE=${KEEPS_VALUE:-"1.00 0.70 0.40 0.10 0.05 0.01"}
DROPS_VALUE=${DROPS_VALUE:-"0 0.30 0.60 0.90 0.95 0.99"}
SEEDS_VALUE=${SEEDS_VALUE:-"1 2 3 4 5"}
FOLDS_VALUE=${FOLDS_VALUE:-"0 1 2 3 4"}
SURVIVAL_BATCH_SIZE=${SURVIVAL_BATCH_SIZE:-32}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-50}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-5}
CPU_THREADS=${CPU_THREADS:-2}
CACHE_MODE=${CACHE_MODE:-auto}
CACHE_KEEP_THRESHOLD=${CACHE_KEEP_THRESHOLD:-0.05}

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-outputs/logs/coadread_feature_then_survival/${RUN_ID}}
mkdir -p "$LOG_ROOT" "$COADREAD_PATCH_ROOT" "$COADREAD_FEATURE_ROOT"
MASTER_LOG="$LOG_ROOT/master.log"
: > "$MASTER_LOG"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

run_parallel_stage() {
  local stage="$1"
  shift
  local status=0
  local pids=()
  local names=()
  local logs=()
  local idx=0

  for shard in "${SHARDS[@]}"; do
    local gpu="${GPUS[$idx]}"
    local shard_log="$LOG_ROOT/${stage}_shard${shard}.log"
    : > "$shard_log"
    log "launch stage=${stage} shard=${shard}/${NUM_SHARDS} gpu=${gpu} log=${shard_log}"
    CUDA_VISIBLE_DEVICES="$gpu" "$@" "$shard" >> "$shard_log" 2>&1 &
    pids+=("$!")
    names+=("${stage}_shard${shard}")
    logs+=("$shard_log")
    idx=$(((idx + 1) % ${#GPUS[@]}))
  done

  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "finished ${names[$i]} status=0"
    else
      local code=$?
      log "failed ${names[$i]} status=${code} log=${logs[$i]}"
      status=1
    fi
  done
  return "$status"
}

run_patch_shard() {
  local shard="$1"
  "$PY" scripts/step_01_extract_patches.py \
    --cohort_csv "$COADREAD_CSV" \
    --output_root "$COADREAD_PATCH_ROOT" \
    --backend trident \
    --wsi_root "$WSI_ROOT" \
    --patch_size 224 \
    --mag 20 \
    --gpu 0 \
    --batch_size "$PATCH_BATCH_SIZE" \
    --max_workers "$PATCH_MAX_WORKERS" \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$shard" \
    --skip_errors
}

run_feature_shard() {
  local shard="$1"
  "$PY" scripts/step_02_extract_features.py \
    --backend trident \
    --mode full \
    --encoder "$ENCODER" \
    --cohort_csv "$COADREAD_CSV" \
    --patch_root "$COADREAD_PATCH_ROOT" \
    --feature_root "$COADREAD_FEATURE_ROOT" \
    --wsi_root "$WSI_ROOT" \
    --trident_python "$PY" \
    --trident_repo "$TRIDENT_REPO" \
    --gpu 0 \
    --batch_size "$FEATURE_BATCH_SIZE" \
    --save_dtype float32 \
    --max_workers "$FEATURE_MAX_WORKERS" \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$shard" \
    --skip_errors
}

write_coadread_extracted_csv_and_splits() {
  "$PY" - <<'PY'
from pathlib import Path
import pandas as pd

cohort_csv = Path("metadata/csvs/TCGA_COADREAD_survival_dx.csv")
out_csv = Path("metadata/csvs/TCGA_COADREAD_survival_dx_extracted.csv")
feature_base = Path("/mnt/Xsky/zqb/TCGA_WSI_feature_bank/COADREAD/features/uni2")
df = pd.read_csv(cohort_csv)

def has_all_features(slide_id: object) -> bool:
    stem = str(slide_id)
    return (feature_base / "features" / f"{stem}.pt").exists() and (feature_base / "coords" / f"{stem}.pt").exists()

mask = df["slide_submitter_id"].map(has_all_features)
extracted = df.loc[mask].copy()
missing = df.loc[~mask, ["case_submitter_id", "slide_submitter_id", "file_name"]].copy()
out_csv.parent.mkdir(parents=True, exist_ok=True)
extracted.to_csv(out_csv, index=False)
missing.to_csv(Path("metadata/csvs/TCGA_COADREAD_survival_dx_missing_features.csv"), index=False)

config = Path("trainer/configs/dataset/coadread.yaml")
config.write_text(
    'cohort: "COADREAD"\n'
    'cohort_csv: "/mnt/Xsky/zqb/TDAE/metadata/csvs/TCGA_COADREAD_survival_dx_extracted.csv"\n'
    'tasks:\n'
    '  classification:\n'
    '    enabled: false\n'
    '    task: "classification"\n'
    '    label_column: "cancer_code"\n'
    '    include_labels:\n'
    '      - "COAD"\n'
    '      - "READ"\n'
    '    label_aliases:\n'
    '      COAD: "COAD"\n'
    '      READ: "READ"\n'
    '  survival:\n'
    '    enabled: true\n'
    '    task: "survival"\n'
    '    label_column: "cancer_code"\n',
    encoding="utf-8",
)
usable = extracted[pd.to_numeric(extracted["survival_days"], errors="coerce") > 0]
print(f"coadread_extracted_rows={len(extracted)} slides={extracted.slide_submitter_id.nunique()} cases={extracted.case_submitter_id.nunique()}")
print(f"coadread_missing_feature_rows={len(missing)}")
print(f"coadread_usable_survival_rows={len(usable)} cases={usable.case_submitter_id.nunique()}")
if len(usable) < 50:
    raise SystemExit("Too few usable COADREAD survival rows after feature extraction; refusing to launch survival.")
PY
  "$PY" scripts/step_03_generate_splits.py --cohort COADREAD --task survival --n_splits 5 --seed 42
}

run_survival_cohort() {
  local cohort="$1"
  local log_file="$LOG_ROOT/${cohort,,}_survival.log"
  local code=0
  log "launch survival cohort=${cohort} log=${log_file}"
  COHORT="$cohort" \
  TASK=survival \
  ENCODER="$ENCODER" \
  CONFIG=trainer/configs/brca_uni2_mil.yaml \
  EXPERIMENT_TAG="$SURVIVAL_EXPERIMENT_TAG" \
  GPUS="${GPUS[*]}" \
  FOLDS="$FOLDS_VALUE" \
  SEEDS="$SEEDS_VALUE" \
  KEEPS="$KEEPS_VALUE" \
  DROPS="$DROPS_VALUE" \
  EPOCHS="$EPOCHS" \
  SURVIVAL_BATCH_SIZE="$SURVIVAL_BATCH_SIZE" \
  LR="$LR" \
  EARLY_STOPPING_PATIENCE="$EARLY_STOPPING_PATIENCE" \
  EARLY_STOPPING_MONITOR=val_loss \
  CPU_THREADS="$CPU_THREADS" \
  CACHE_MODE="$CACHE_MODE" \
  CACHE_KEEP_THRESHOLD="$CACHE_KEEP_THRESHOLD" \
  bash scripts/run_abmil_random_keep_seed_layout.sh >> "$log_file" 2>&1
  code=$?
  log "finished survival cohort=${cohort} status=${code}"
  return "$code"
}

log "started pipeline run_id=${RUN_ID} gpus=${GPUS[*]} shards=${SHARDS[*]} coadread_csv=${COADREAD_CSV}"
log "stage=patch begin"
if ! run_parallel_stage patch run_patch_shard; then
  log "stage=patch failed"
  exit 1
fi

log "stage=feature begin"
if ! run_parallel_stage feature run_feature_shard; then
  log "stage=feature failed"
  exit 1
fi

log "stage=postprocess begin"
if ! write_coadread_extracted_csv_and_splits >> "$LOG_ROOT/postprocess.log" 2>&1; then
  log "stage=postprocess failed log=$LOG_ROOT/postprocess.log"
  exit 1
fi
cat "$LOG_ROOT/postprocess.log" >> "$MASTER_LOG"

log "stage=survival begin cohort=GBMLGG"
if ! run_survival_cohort GBMLGG; then
  log "stage=survival failed cohort=GBMLGG"
  exit 1
fi

log "stage=survival begin cohort=COADREAD"
if ! run_survival_cohort COADREAD; then
  log "stage=survival failed cohort=COADREAD"
  exit 1
fi

log "pipeline finished status=0"
