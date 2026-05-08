from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path


DEFAULT_ABLATIONS = {
    "granularity": ["4level", "3level", "binary"],
    "propagation": ["gat", "weighted", "gcn", "none"],
    "gat_layers": [1, 2, 3],
    "gating_strategy": ["learned", "cosine_threshold", "random"],
    "loss_components": ["full", "no_budget", "no_diversity"],
    "annealing": ["linear", "cosine", "fixed"],
    "aggregator": ["abmil", "transmil", "clam"],
    "c_target": [0.2, 0.3, 0.4, 0.5],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize or run TDAE ablation commands.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--out_json", default="outputs/ablation_plan.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    keys = list(DEFAULT_ABLATIONS)
    rows = [dict(zip(keys, values)) for values in itertools.product(*(DEFAULT_ABLATIONS[key] for key in keys))]
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} ablation configs to {out}")
    if args.dry_run:
        return
    for row in rows:
        if row["gating_strategy"] != "learned":
            continue
        subprocess.run(["python", "scripts/03_train.py"], check=True)


if __name__ == "__main__":
    main()
