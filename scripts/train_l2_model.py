#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from dufppi.model import coefficient_table, cross_validate, fit_model, load_training_data, save_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.release_root.resolve()
    features, labels, names = load_training_data(root / "data/model/training_labels_channels.tsv.gz")
    oof, folds, pooled = cross_validate(features, labels)
    oof.insert(0, "prediction_name", names)
    oof.insert(1, "liberal_high_confidence", labels)
    oof = oof.rename(columns={"score": "l2_model_score"})
    oof.to_csv(root / "data/model/oof_predictions_reproduced.tsv.gz", sep="\t", index=False)
    folds.to_csv(root / "data/model/fold_metrics_reproduced.tsv", sep="\t", index=False)
    baseline_scores = {
        "L2 model": oof["l2_model_score"],
        "Combined STRING score": features["combined_score"],
        "Neighborhood score": features["neighborhood"],
        "Co-occurrence score": features["cooccurence"],
    }
    baseline_metrics = pd.DataFrame(
        [
            {
                "ranking": ranking,
                "average_precision": average_precision_score(labels, score),
                "auroc": roc_auc_score(labels, score),
            }
            for ranking, score in baseline_scores.items()
        ]
    )
    baseline_metrics.to_csv(
        root / "data/model/baseline_ranking_metrics_reproduced.tsv", sep="\t", index=False
    )
    pipeline = fit_model(features, labels)
    save_model(pipeline, root / "data/model/l2_pipeline_reproduced.joblib")
    coefficient_table(pipeline).to_csv(root / "data/model/fitted_coefficients_reproduced.tsv", sep="\t", index=False)
    (root / "data/model/validation_reproduced.json").write_text(json.dumps(pooled, indent=2) + "\n")
    print(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    main()
