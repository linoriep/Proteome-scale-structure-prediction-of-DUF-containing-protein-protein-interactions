#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dufppi.cohorts import build_cohorts, read_edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("data/model/l2_pipeline.joblib"))
    parser.add_argument("--output", type=Path, default=Path("results/cohort_candidates.tsv.gz"))
    args = parser.parse_args()
    result = build_cohorts(read_edges(args.edges), args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(result.groupby("cohort").size().to_string())


if __name__ == "__main__":
    main()
