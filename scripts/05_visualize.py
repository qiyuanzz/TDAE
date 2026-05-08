from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from utils.visualization import save_gating_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Visualize a 4-level TDAE gating map from tensors.')
    parser.add_argument('--gate_values', '--gate_mask', dest='gate_values', required=True)
    parser.add_argument('--grid_indices', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--num_levels', type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gate = torch.load(args.gate_values, map_location='cpu')
    coords = torch.load(args.grid_indices, map_location='cpu')
    path = save_gating_map(gate, coords, Path(args.save_path), num_levels=args.num_levels)
    print(path)


if __name__ == '__main__':
    main()
