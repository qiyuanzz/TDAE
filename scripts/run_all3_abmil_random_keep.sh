#!/usr/bin/env bash
set -euo pipefail

cd /mnt/Xsky/zqb/TDAE

GPUS_DEFAULT="${GPUS:-4 5 6 7}"
FOLDS_DEFAULT="${FOLDS:-0 1 2 3 4}"
SEEDS_DEFAULT="${SEEDS:-1 2 3 4 5}"
KEEPS_DEFAULT="${KEEPS:-1.00 0.70 0.40 0.10 0.05 0.01}"
DROPS_DEFAULT="${DROPS:-0 0.30 0.60 0.90 0.95 0.99}"
EPOCHS_DEFAULT="${EPOCHS:-20}"
LR_DEFAULT="${LR:-2e-4}"
SURVIVAL_BATCH_SIZE_DEFAULT="${SURVIVAL_BATCH_SIZE:-32}"
CPU_THREADS_DEFAULT="${CPU_THREADS:-2}"
CACHE_MODE_DEFAULT="${CACHE_MODE:-auto}"

run_cohort() {
  local cohort="$1"
  local config="$2"
  local experiment_tag="$3"

  echo "[$(date)] start cohort=${cohort} config=${config} experiment=${experiment_tag}"
  COHORT="$cohort" \
  CONFIG="$config" \
  TASK=survival \
  ENCODER=uni2 \
  EXPERIMENT_TAG="$experiment_tag" \
  GPUS="$GPUS_DEFAULT" \
  FOLDS="$FOLDS_DEFAULT" \
  SEEDS="$SEEDS_DEFAULT" \
  KEEPS="$KEEPS_DEFAULT" \
  DROPS="$DROPS_DEFAULT" \
  EPOCHS="$EPOCHS_DEFAULT" \
  LR="$LR_DEFAULT" \
  SURVIVAL_BATCH_SIZE="$SURVIVAL_BATCH_SIZE_DEFAULT" \
  CPU_THREADS="$CPU_THREADS_DEFAULT" \
  CACHE_MODE="$CACHE_MODE_DEFAULT" \
  bash scripts/run_abmil_random_keep_seed_layout.sh
  echo "[$(date)] done cohort=${cohort} experiment=${experiment_tag}"
}

run_cohort BRCA configs/brca_uni2_phase0.yaml brca_uni2_abmil_random_keep_nll20_seedlayout_v2
run_cohort GBMLGG configs/default.yaml gbmlgg_uni2_abmil_random_keep_nll20_seedlayout_v2
run_cohort COADREAD configs/default.yaml coadread_uni2_abmil_random_keep_nll20_seedlayout_v2
