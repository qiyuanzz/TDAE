#!/usr/bin/env bash
set -u

cd /mnt/Xsky/zqb/TDAE || exit 1

PY=${PY:-/opt/conda/envs/trident/bin/python}
COHORT=${COHORT:-BRCA}
TASK=${TASK:-survival}
ENCODER=${ENCODER:-uni2}
CONFIG=${CONFIG:-trainer/configs/brca_uni2_mil.yaml}
AGGREGATOR=${AGGREGATOR:-abmil}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-brca_uni2_abmil_random_keep_seed_layout}

GPUS=(${GPUS:-4 5 6 7})
FOLDS=(${FOLDS:-0 1 2 3 4})
SEEDS=(${SEEDS:-1 2 3 4 5})
KEEPS=(${KEEPS:-1.00 0.70 0.40 0.10 0.05 0.01})
DROPS=(${DROPS:-0 0.30 0.60 0.90 0.95 0.99})

EPOCHS=${EPOCHS:-20}
SURVIVAL_BATCH_SIZE=${SURVIVAL_BATCH_SIZE:-32}
LR=${LR:-2e-4}
REG=${REG:-1e-5}
OPT=${OPT:-adam}
BATCH_SIZE=${BATCH_SIZE:-1}
BAG_LOSS=${BAG_LOSS:-nll_surv}
ALPHA_SURV=${ALPHA_SURV:-0.0}
N_CLASSES=${N_CLASSES:-4}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-0}
EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA:-0.0}
EARLY_STOPPING_MONITOR=${EARLY_STOPPING_MONITOR:-val_loss}
CPU_THREADS=${CPU_THREADS:-2}
CACHE_MODE=${CACHE_MODE:-auto}
CACHE_KEEP_THRESHOLD=${CACHE_KEEP_THRESHOLD:-0.05}

if [ "${#KEEPS[@]}" -ne "${#DROPS[@]}" ]; then
  echo "KEEPS and DROPS must have the same length." >&2
  exit 1
fi

BASE_DIR="outputs/checkpoints/${COHORT}/${TASK}/${ENCODER}/${EXPERIMENT_TAG}"
mkdir -p "$BASE_DIR"
MASTER_LOG="$BASE_DIR/master.log"
: > "$MASTER_LOG"

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export MIL_TORCH_NUM_THREADS="$CPU_THREADS"

keep_dir_name() {
  local keep="$1"
  python3 - "$keep" "$SURVIVAL_BATCH_SIZE" "$LR" "$AGGREGATOR" <<'PY'
import sys

keep = float(sys.argv[1])
gc = int(sys.argv[2])
lr = float(sys.argv[3])
aggregator = sys.argv[4].upper()

def float_tag(value):
    text = f"{value:.0e}" if 0 < abs(value) < 1e-3 else f"{value:.6g}"
    return text.replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")

print(f"wsi_{aggregator}_keep_{keep:.6g}_gc_{gc}_lr_{float_tag(lr)}")
PY
}

should_disable_cache() {
  local keep="$1"
  if [ "$CACHE_MODE" = "off" ]; then
    return 0
  fi
  if [ "$CACHE_MODE" = "on" ]; then
    return 1
  fi
  python3 - "$keep" "$CACHE_KEEP_THRESHOLD" <<'PY'
import sys
keep = float(sys.argv[1])
threshold = float(sys.argv[2])
raise SystemExit(0 if keep > threshold else 1)
PY
}

run_seed() {
  local seed="$1"
  local status=0
  local active_pids=()
  local active_names=()
  local active_logs=()
  local gpu_index=0
  local active_count=0

  for keep in "${KEEPS[@]}"; do
    local setting_dir
    setting_dir="$(keep_dir_name "$keep")"
    local setting_seed_dir="$BASE_DIR/$setting_dir/seed_${seed}"
    mkdir -p "$setting_seed_dir"
    : > "$setting_seed_dir/seed.log"
    echo "[$(date)] seed=$seed setting=$setting_dir started experiment=$EXPERIMENT_TAG folds=${FOLDS[*]} epochs=$EPOCHS lr=$LR reg=$REG opt=$OPT batch_size=$BATCH_SIZE gc=$SURVIVAL_BATCH_SIZE bag_loss=$BAG_LOSS alpha_surv=$ALPHA_SURV n_classes=$N_CLASSES early_stopping_patience=$EARLY_STOPPING_PATIENCE monitor=$EARLY_STOPPING_MONITOR cache_mode=$CACHE_MODE cache_keep_threshold=$CACHE_KEEP_THRESHOLD cpu_threads=$CPU_THREADS gpus=${GPUS[*]}" >> "$setting_seed_dir/seed.log"
  done

  wait_batch() {
    local batch_status=0
    local i pid name code log
    for i in "${!active_pids[@]}"; do
      pid="${active_pids[$i]}"
      name="${active_names[$i]}"
      log="${active_logs[$i]}"
      if wait "$pid"; then
        echo "[$(date)] finished name=$name pid=$pid status=0" >> "$log"
      else
        code=$?
        echo "[$(date)] failed name=$name pid=$pid status=$code" >> "$log"
        batch_status=1
      fi
    done
    active_pids=()
    active_names=()
    active_logs=()
    active_count=0
    gpu_index=0
    if [ "$batch_status" -ne 0 ]; then
      status=1
    fi
  }

  for fold in "${FOLDS[@]}"; do
    for idx in "${!KEEPS[@]}"; do
      local keep="${KEEPS[$idx]}"
      local drop="${DROPS[$idx]}"
      local gpu="${GPUS[$gpu_index]}"
      local keep_dir
      keep_dir="$(keep_dir_name "$keep")"
      local seed_dir="$BASE_DIR/$keep_dir/seed_${seed}"
      local seed_log="$seed_dir/seed.log"
      local fold_dir="$seed_dir/fold_${fold}"
      local summary_path="$fold_dir/summary.json"
      local run_tag="${AGGREGATOR}_random_keep${keep}_seed${seed}_fold${fold}"
      local cache_args=()
      if should_disable_cache "$keep"; then
        cache_args=(--no_cache_features)
      fi

      if [ -f "$summary_path" ]; then
        echo "[$(date)] skip existing name=$run_tag summary=$summary_path" >> "$seed_log"
        continue
      fi

      mkdir -p "$fold_dir"
      echo "[$(date)] launched name=$run_tag fold=$fold keep=$keep drop=$drop gpu=$gpu cache_args=${cache_args[*]:-cache_on}" >> "$seed_log"
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/step_04_train.py \
        --config "$CONFIG" \
        --cohort "$COHORT" \
        --task "$TASK" \
        --encoder "$ENCODER" \
        --aggregator "$AGGREGATOR" \
        --fold "$fold" \
        --patch_drop_ratio "$drop" \
        --patch_drop_seed "$seed" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --weight_decay "$REG" \
        --opt "$OPT" \
        --batch_size "$BATCH_SIZE" \
        --survival_batch_size "$SURVIVAL_BATCH_SIZE" \
        --bag_loss "$BAG_LOSS" \
        --alpha_surv "$ALPHA_SURV" \
        --n_classes "$N_CLASSES" \
        --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
        --early_stopping_min_delta "$EARLY_STOPPING_MIN_DELTA" \
        --early_stopping_monitor "$EARLY_STOPPING_MONITOR" \
        --torch_num_threads "$CPU_THREADS" \
        --experiment_tag "$EXPERIMENT_TAG" \
        --output_layout seed \
        --run_tag "$run_tag" \
        "${cache_args[@]}" >> "$seed_log" 2>&1 &

      local pid=$!
      active_pids+=("$pid")
      active_names+=("$run_tag")
      active_logs+=("$seed_log")
      active_count=$((active_count + 1))
      gpu_index=$(((gpu_index + 1) % ${#GPUS[@]}))
      if [ "$active_count" -eq "${#GPUS[@]}" ]; then
        wait_batch
      fi
    done
  done

  if [ "$active_count" -gt 0 ]; then
    wait_batch
  fi

  return "$status"
}

overall=0
echo "[$(date)] started experiment=$EXPERIMENT_TAG base_dir=$BASE_DIR seeds=${SEEDS[*]}" >> "$MASTER_LOG"
for seed in "${SEEDS[@]}"; do
  echo "[$(date)] seed=$seed started settings=${KEEPS[*]}" >> "$MASTER_LOG"
  if run_seed "$seed"; then
    echo "[$(date)] seed=$seed finished status=0" >> "$MASTER_LOG"
  else
    overall=1
    echo "[$(date)] seed=$seed finished status=1" >> "$MASTER_LOG"
    break
  fi
done
echo "[$(date)] finished experiment=$EXPERIMENT_TAG status=$overall" >> "$MASTER_LOG"
exit "$overall"
