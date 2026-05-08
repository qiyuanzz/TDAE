# TDAE Result Layout Standard

All new training runs should save outputs by experiment, setting, seed, and fold. The setting directory should combine the model name and key hyperparameters, so one setting contains the five-fold results for each seed.

## Directory Template

```text
outputs/checkpoints/<COHORT>/<TASK>/<ENCODER>/<EXPERIMENT_TAG>/
  master.log
  <SETTING>/
    seed_<SEED>/
      seed.log
      fold_0/
        summary.json
        history.csv
        split.csv
        <model>_last.pt
        <model>_best.pt    # when the entrypoint has model selection/early stopping
      fold_1/
      fold_2/
      fold_3/
      fold_4/
```

`<SETTING>` should be a compact signature such as:

```text
wsi_ABMIL_keep_0.01_gc_32_lr_1e-4
wsi_TDAE_c_0.3_gc_32_lr_1e-4
wsi_TDAE_full_upper_bound_gc_32_lr_1e-4
```

Use this level for model family, patch keep ratio or TDAE budget, effective survival batch/gradient-count setting, and learning rate. Put `fold_0` through `fold_4` under the setting/seed directory, not above it.

## Required Files

- `master.log`: one experiment-level launch log.
- `seed.log`: one log per setting and seed. Do not create per-GPU logs for normal runs.
- `summary.json`: run metadata and selected metrics for the fold and setting.
- `history.csv`: epoch-level training and validation history.
- `split.csv`: exact fold split copied from `data/splits`.
- model checkpoint files: last checkpoint is required; best checkpoint is required when the entrypoint has model selection or early stopping.

## Current Entrypoints

- `scripts/04_train_abmil_random_drop.py` defaults to `--output_layout seed` and saves patch-budget settings as `wsi_ABMIL_keep_<ratio>_gc_<batch>_lr_<lr>`.
- `scripts/03_train.py` defaults to `--output_layout seed` and saves TDAE settings as `wsi_TDAE_<budget-or-method>_gc_<batch>_lr_<lr>`.
- Pass `--output_layout legacy` only when reproducing old outputs.

## Resource Policy

Launchers should also keep resource settings explicit:

- use only GPUs `4,5,6,7` unless changed deliberately;
- keep `SURVIVAL_BATCH_SIZE >= 32` for survival runs;
- cap CPU threads with `CPU_THREADS` or `--torch_num_threads`;
- use adaptive feature caching for large patch ratios, because full-keep feature caching can consume hundreds of GB across parallel processes.
