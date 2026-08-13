#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dufppi.tables import (
    confident_composition,
    family_coverage,
    multiplicity_bins,
    multiple_partner_thresholds,
    partner_composition,
    run_summary,
    score_yield,
    taxonomic_summary,
    taxa_recurrence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("data/predictions/all_successful_predictions.tsv.gz"))
    parser.add_argument("--annotations", type=Path, default=Path("data/predictions/duf_annotation_rows.tsv.gz"))
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions, sep="\t", low_memory=False)
    annotations = pd.read_csv(args.annotations, sep="\t", low_memory=False)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "paper_run_summary.tsv": run_summary(predictions, annotations),
        "paper_final_model_score_yield.tsv": score_yield(predictions),
        "paper_all_runs_taxa_recurrence_yield_to500.tsv": taxa_recurrence(predictions),
        "paper_partner_function_cohort_composition_liberal.tsv": partner_composition(predictions),
        "paper_partner_function_confident_composition.tsv": confident_composition(predictions),
        "paper_partner_function_duf_family_coverage.tsv": family_coverage(annotations),
        "paper_duf_liberal_partner_multiplicity_bins.tsv": multiplicity_bins(annotations),
        "paper_duf_multiple_partner_thresholds.tsv": multiple_partner_thresholds(annotations),
    }
    if args.taxonomy:
        taxonomy = pd.read_csv(args.taxonomy, sep="\t", low_memory=False)
        outputs["paper_taxonomic_domain_summary.tsv"] = taxonomic_summary(predictions, taxonomy)
    for filename, table in outputs.items():
        table.to_csv(args.output_directory / filename, sep="\t", index=False)
    print(f"Wrote {len(outputs)} tables to {args.output_directory}")


if __name__ == "__main__":
    main()
