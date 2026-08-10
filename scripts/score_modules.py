#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from dufppi.model import score_modules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve()
    model_path = args.model or root / "data/model/l2_pipeline.joblib"
    input_path = args.input if args.input.is_absolute() else root / args.input
    output_path = args.output if args.output.is_absolute() else root / args.output
    modules = pd.read_csv(input_path, sep="\t", low_memory=False)
    scored = score_modules(joblib.load(model_path), modules)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
