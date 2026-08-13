#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from dufppi.model import FEATURES, load_training_data, make_pipeline, score_modules


DISPLAY = {
    "combined": "Combined", "neighborhood": "Neighborhood",
    "neighborhood_transferred": "Neighborhood transferred", "fusion": "Fusion",
    "cooccurence": "Phyletic co-occurrence", "homology": "Homology",
    "coexpression": "Coexpression", "coexpression_transferred": "Coexpression transferred",
    "experiments": "Experiments", "experiments_transferred": "Experiments transferred",
    "database": "Database", "database_transferred": "Database transferred",
    "textmining": "Text mining", "textmining_transferred": "Text mining transferred",
}


def channel_tables(modules: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = predictions[predictions["cohort"].eq("L2 model")].copy()
    predictions["module_key"] = predictions["duf_pfam_id"].astype(str) + " || " + predictions["partner_architecture"].fillna("NO_PFAM").astype(str)
    data = modules.merge(
        predictions[["module_key", "liberal_high_confidence", "strict_high_confidence"]],
        on="module_key", how="inner", validate="one_to_one",
    )
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1001]
    labels = ["0-99", "100-199", "200-299", "300-399", "400-499", "500-599", "600-699", "700-799", "800-899", "900-949", "950-1000"]
    outputs = []
    for confidence, count_name in [("liberal_high_confidence", "liberal"), ("strict_high_confidence", "strict")]:
        rows = []
        for feature in ["combined_score", *FEATURES[1:]]:
            source = "max_combined" if feature == "combined_score" else f"max_{feature}"
            channel = "combined" if feature == "combined_score" else feature
            groups = pd.cut(data[source], bins=bins, labels=labels, right=False)
            for label in labels:
                group = data[groups.eq(label)]
                count = int(group[confidence].sum())
                rows.append({"score_bin": label, "n": len(group), count_name: count,
                             f"{count_name}_fraction": count / len(group) if len(group) else np.nan,
                             "channel": DISPLAY[channel], "source_column": source})
        outputs.append(pd.DataFrame(rows))
    return outputs[0], outputs[1]


def class_weight_sensitivity(training: Path, modules: pd.DataFrame) -> pd.DataFrame:
    features, labels, _ = load_training_data(training)
    weights = [1.0, 3.0, 6.37, 12.0, 20.0]
    scores = {}; module_scores = {}; metrics = {}
    for weight in weights:
        pipeline = make_pipeline()
        pipeline.set_params(logistic_regression__class_weight={0: 1, 1: weight})
        oof, _, _ = cross_validate_with_pipeline(features, labels, pipeline)
        pipeline.fit(features, labels)
        scores[weight] = oof
        module_scores[weight] = score_modules(pipeline, modules)["l2_model_score"]
        metrics[weight] = (average_precision_score(labels, oof), roc_auc_score(labels, oof))
    reference = module_scores[12.0]
    top_n = max(1, int(np.ceil(len(reference) * 0.1)))
    reference_top = set(reference.nlargest(top_n).index)
    rows = []
    for weight in weights:
        current = module_scores[weight]
        eligible = current[modules["max_combined"].ge(900)]
        rows.append({
            "positive_class_weight": weight, "AP": round(metrics[weight][0], 4), "AUROC": round(metrics[weight][1], 4),
            "spearman_vs_weight12": round(float(spearmanr(current, reference).statistic), 4),
            "top10pct_overlap_vs_weight12": round(len(set(current.nlargest(top_n).index) & reference_top) / top_n, 4),
            "all_string_modules_score_ge_0p5": int(eligible.ge(0.5).sum()),
            "all_string_modules_score_ge_0p75": int(eligible.ge(0.75).sum()),
        })
    return pd.DataFrame(rows)


def cross_validate_with_pipeline(features: pd.DataFrame, labels: pd.Series, pipeline) -> tuple[np.ndarray, None, None]:
    from sklearn.model_selection import StratifiedKFold
    values = np.full(len(features), np.nan)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    for train, test in splitter.split(features, labels):
        fitted = clone(pipeline)
        fitted.fit(features.iloc[train], labels.iloc[train])
        values[test] = fitted.predict_proba(features.iloc[test])[:, 1]
    return values, None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-directory", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    root = args.release_root.resolve()
    modules = pd.read_csv(root / "data/candidates/all_organism_duf_modules.tsv.gz", sep="\t", low_memory=False)
    modules["module_key"] = modules["duf_pfam_id"].astype(str) + " || " + modules["partner_architecture"].fillna("NO_PFAM").astype(str)
    predictions = pd.read_csv(root / "data/predictions/all_successful_predictions.tsv.gz", sep="\t", low_memory=False)
    liberal, strict = channel_tables(modules, predictions)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    liberal.to_csv(args.output_directory / "paper_string_all_channels_score_bin_confidence.tsv", sep="\t", index=False)
    strict.to_csv(args.output_directory / "paper_string_all_channels_score_bin_strict_confidence.tsv", sep="\t", index=False)
    class_weight_sensitivity(root / "data/model/training_labels_channels.tsv.gz", modules).to_csv(
        args.output_directory / "paper_separate_channel_l2_class_weight_sensitivity.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    main()
