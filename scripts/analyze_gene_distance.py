#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dufppi.tables import gene_distance_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("data/predictions/all_successful_predictions.tsv.gz"))
    parser.add_argument("--locations", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions, sep="\t", low_memory=False)
    locations = pd.read_csv(args.locations, sep="\t", low_memory=False)
    distance, parse = gene_distance_tables(predictions, locations)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    distance.to_csv(args.output_directory / "paper_gene_distance_confidence_yield.tsv", sep="\t", index=False)
    parse.to_csv(args.output_directory / "paper_gene_distance_parse_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
