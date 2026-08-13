from __future__ import annotations

from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "prediction_name",
    "interface_ipSAE",
    "interface_average_plddt",
    "model_average_plddt",
    "model_available",
    "interface_scored",
    "duf_domain_interacting",
]


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False)


def collect_predictions(candidates_path: str | Path, metrics_path: str | Path) -> pd.DataFrame:
    candidates = pd.read_csv(candidates_path, sep="\t", low_memory=False)
    metrics = pd.read_csv(metrics_path, sep="\t", low_memory=False)
    missing = [column for column in METRIC_COLUMNS if column not in metrics]
    if missing:
        raise ValueError(f"Missing prediction metric columns: {missing}")
    if metrics["prediction_name"].duplicated().any():
        raise ValueError("Prediction metrics contain duplicate prediction names")
    result = candidates.merge(metrics, on="prediction_name", how="left", validate="one_to_one")
    for column in ["model_available", "interface_scored", "duf_domain_interacting"]:
        result[column] = as_bool(result[column])
    interface_plddt = pd.to_numeric(result["interface_average_plddt"], errors="coerce")
    model_plddt = pd.to_numeric(result["model_average_plddt"], errors="coerce")
    plddt = interface_plddt.where(~result["cohort"].eq("L2 model"), model_plddt)
    ipsae = pd.to_numeric(result["interface_ipSAE"], errors="coerce")
    result["liberal_high_confidence"] = ipsae.ge(0.2) & plddt.ge(70)
    result["strict_high_confidence"] = ipsae.ge(0.6) & plddt.ge(70)
    result["outcome"] = "scored below threshold"
    result.loc[result["liberal_high_confidence"], "outcome"] = "liberal-only confidence"
    result.loc[result["strict_high_confidence"], "outcome"] = "strict high confidence"
    result.loc[result["model_available"] & ~result["interface_scored"], "outcome"] = "model present, no interface score"
    result.loc[~result["model_available"], "outcome"] = "technical failure"
    result["candidate_status"] = result["model_available"].map({True: "success", False: "technical failure"})
    return result
