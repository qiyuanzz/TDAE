#!/usr/bin/env bash
set -u

cd /mnt/Xsky/zqb/TDAE || exit 1

PY=${PY:-/opt/conda/envs/trident/bin/python}
TRIDENT_REPO=${TRIDENT_REPO:-/mnt/Xsky/zqb/TRIDENT}
WSI_ROOT=${WSI_ROOT:-/mnt/Archive/Dataset/GDC_DATA/GDC_DATA}
ENCODER=${ENCODER:-uni2}

GPUS=(${GPUS:-4 5 6 7})
COHORTS=(${COHORTS:-GBMLGG BRCA COADREAD})
NUM_SHARDS=${NUM_SHARDS:-${#GPUS[@]}}

PATCH_BATCH_SIZE=${PATCH_BATCH_SIZE:-64}
FEATURE_BATCH_SIZE=${FEATURE_BATCH_SIZE:-64}
PATCH_MAX_WORKERS=${PATCH_MAX_WORKERS:-8}
FEATURE_MAX_WORKERS=${FEATURE_MAX_WORKERS:-4}
PATCH_SIZE=${PATCH_SIZE:-224}
MAG=${MAG:-20}
OVERLAP=${OVERLAP:-0}
SEGMENTER=${SEGMENTER:-hest}
SAVE_DTYPE=${SAVE_DTYPE:-float32}

RUN_PATCH=${RUN_PATCH:-1}
RUN_FEATURE=${RUN_FEATURE:-1}
SKIP_ERRORS=${SKIP_ERRORS:-1}

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-outputs/logs/patch_feature_all3/${RUN_ID}}
mkdir -p "$LOG_ROOT"
MASTER_LOG="$LOG_ROOT/master.log"
: > "$MASTER_LOG"

declare -A COHORT_CSV=(
  [GBMLGG]="metadata/csvs/TCGA_GBMLGG_survival_dx.csv"
  [BRCA]="metadata/csvs/TCGA_BRCA_survival_dx.csv"
  [COADREAD]="metadata/csvs/TCGA_COADREAD_survival_dx.csv"
)

declare -A PATCH_ROOT=(
  [GBMLGG]="/mnt/Xsky/zqb/TCGA_WSI_feature_bank/GBMLGG/preprocess"
  [BRCA]="/mnt/Xsky/zqb/TCGA_WSI_feature_bank/BRCA/preprocess"
  [COADREAD]="/mnt/Xsky/zqb/TCGA_WSI_feature_bank/COADREAD/preprocess"
)

declare -A FEATURE_ROOT=(
  [GBMLGG]="/mnt/Xsky/zqb/TCGA_WSI_feature_bank/GBMLGG/features"
  [BRCA]="/mnt/Xsky/zqb/TCGA_WSI_feature_bank/BRCA/features"
  [COADREAD]="/mnt/Xsky/zqb/TCGA_WSI_feature_bank/COADREAD/features"
)

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

skip_args() {
  if [ "$SKIP_ERRORS" = "1" ]; then
    printf '%s\n' "--skip_errors"
  fi
}

run_patch_shard() {
  local cohort="$1"
  local shard="$2"
  "$PY" scripts/step_01_extract_patches.py \
    --cohort_csv "${COHORT_CSV[$cohort]}" \
    --output_root "${PATCH_ROOT[$cohort]}" \
    --backend trident \
    --wsi_root "$WSI_ROOT" \
    --trident_python "$PY" \
    --trident_repo "$TRIDENT_REPO" \
    --trident_tasks seg coords \
    --gpu 0 \
    --mag "$MAG" \
    --patch_size "$PATCH_SIZE" \
    --overlap "$OVERLAP" \
    --segmenter "$SEGMENTER" \
    --batch_size "$PATCH_BATCH_SIZE" \
    --max_workers "$PATCH_MAX_WORKERS" \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$shard" \
    $(skip_args)
}

run_feature_shard() {
  local cohort="$1"
  local shard="$2"
  "$PY" scripts/step_02_extract_features.py \
    --backend trident \
    --mode full \
    --encoder "$ENCODER" \
    --cohort_csv "${COHORT_CSV[$cohort]}" \
    --patch_root "${PATCH_ROOT[$cohort]}" \
    --feature_root "${FEATURE_ROOT[$cohort]}" \
    --wsi_root "$WSI_ROOT" \
    --trident_python "$PY" \
    --trident_repo "$TRIDENT_REPO" \
    --gpu 0 \
    --mag "$MAG" \
    --patch_size "$PATCH_SIZE" \
    --overlap "$OVERLAP" \
    --batch_size "$FEATURE_BATCH_SIZE" \
    --save_dtype "$SAVE_DTYPE" \
    --max_workers "$FEATURE_MAX_WORKERS" \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$shard" \
    $(skip_args)
}

run_stage() {
  local cohort="$1"
  local stage="$2"
  local status=0
  local pids=()
  local names=()
  local logs=()

  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$((shard % ${#GPUS[@]}))]}"
    local shard_log="$LOG_ROOT/${cohort}_${stage}_shard${shard}.log"
    : > "$shard_log"
    log "launch cohort=$cohort stage=$stage shard=$shard/$NUM_SHARDS gpu=$gpu log=$shard_log"
    if [ "$stage" = "patch" ]; then
      (export CUDA_VISIBLE_DEVICES="$gpu"; run_patch_shard "$cohort" "$shard") >> "$shard_log" 2>&1 &
    else
      (export CUDA_VISIBLE_DEVICES="$gpu"; run_feature_shard "$cohort" "$shard") >> "$shard_log" 2>&1 &
    fi
    pids+=("$!")
    names+=("${cohort}_${stage}_shard${shard}")
    logs+=("$shard_log")
  done

  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "finished ${names[$i]} status=0"
    else
      local code=$?
      log "failed ${names[$i]} status=$code log=${logs[$i]}"
      status=1
    fi
  done
  return "$status"
}

summarize_cohort() {
  local cohort="$1"
  "$PY" - "$cohort" "${COHORT_CSV[$cohort]}" "${PATCH_ROOT[$cohort]}" "${FEATURE_ROOT[$cohort]}" "$ENCODER" "$LOG_ROOT" <<'PY'
import sys
from pathlib import Path
import pandas as pd

cohort, csv_path, patch_root, feature_root, encoder, log_root = sys.argv[1:]
df = pd.read_csv(csv_path)
slide_ids = [str(x) for x in df["slide_submitter_id"].drop_duplicates()]
patch_root = Path(patch_root)
feature_base = Path(feature_root) / encoder
missing_patch = []
missing_feature = []
for slide_id in slide_ids:
    if not (patch_root / "patches" / f"{slide_id}.pt").exists():
        missing_patch.append(slide_id)
    if not (feature_base / "features" / f"{slide_id}.pt").exists() or not (feature_base / "coords" / f"{slide_id}.pt").exists():
        missing_feature.append(slide_id)

log_root = Path(log_root)
pd.DataFrame({"slide_submitter_id": missing_patch}).to_csv(log_root / f"{cohort}_missing_patch.csv", index=False)
pd.DataFrame({"slide_submitter_id": missing_feature}).to_csv(log_root / f"{cohort}_missing_feature.csv", index=False)
print(
    f"{cohort}: total_slides={len(slide_ids)} "
    f"patch_done={len(slide_ids) - len(missing_patch)} missing_patch={len(missing_patch)} "
    f"feature_done={len(slide_ids) - len(missing_feature)} missing_feature={len(missing_feature)}"
)
PY
}

if [ "$NUM_SHARDS" -lt 1 ]; then
  echo "NUM_SHARDS must be >= 1" >&2
  exit 1
fi

log "started run_id=$RUN_ID cohorts=${COHORTS[*]} gpus=${GPUS[*]} num_shards=$NUM_SHARDS encoder=$ENCODER skip_errors=$SKIP_ERRORS"
overall=0

for cohort in "${COHORTS[@]}"; do
  if [ -z "${COHORT_CSV[$cohort]+x}" ]; then
    log "unknown cohort=$cohort; valid cohorts: ${!COHORT_CSV[*]}"
    overall=1
    break
  fi
  mkdir -p "${PATCH_ROOT[$cohort]}" "${FEATURE_ROOT[$cohort]}"
  log "cohort=$cohort begin csv=${COHORT_CSV[$cohort]} patch_root=${PATCH_ROOT[$cohort]} feature_root=${FEATURE_ROOT[$cohort]}"
  if [ "$RUN_PATCH" = "1" ]; then
    if ! run_stage "$cohort" patch; then
      overall=1
      log "cohort=$cohort patch failed"
      break
    fi
  else
    log "cohort=$cohort patch skipped RUN_PATCH=$RUN_PATCH"
  fi
  if [ "$RUN_FEATURE" = "1" ]; then
    if ! run_stage "$cohort" feature; then
      overall=1
      log "cohort=$cohort feature failed"
      break
    fi
  else
    log "cohort=$cohort feature skipped RUN_FEATURE=$RUN_FEATURE"
  fi
  summarize_cohort "$cohort" | tee -a "$MASTER_LOG"
done

log "finished status=$overall log_root=$LOG_ROOT"
exit "$overall"
