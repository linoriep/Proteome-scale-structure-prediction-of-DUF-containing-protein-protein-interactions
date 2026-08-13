#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dufppi.predictions import collect_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("results/predictions"))
    args = parser.parse_args()
    result = collect_predictions(args.candidates, args.metrics)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_directory / "all_candidates.tsv.gz", sep="\t", index=False)
    result[result["candidate_status"].eq("success")].drop(columns="candidate_status").to_csv(
        args.output_directory / "all_successful_predictions.tsv.gz", sep="\t", index=False
    )
    print(result["candidate_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
