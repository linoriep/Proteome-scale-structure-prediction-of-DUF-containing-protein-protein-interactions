from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from dufppi.model import FEATURES, score_modules


EDGE_COLUMNS = [
    "taxid",
    "duf_protein_id",
    "partner_protein_id",
    "duf_pfam_id",
    "duf_pfam_name",
    "duf_architecture",
    "partner_architecture",
    *FEATURES,
]


def read_edges(path: str | Path) -> pd.DataFrame:
    edges = pd.read_csv(path, sep="\t", low_memory=False)
    missing = [column for column in EDGE_COLUMNS if column not in edges]
    if missing:
        raise ValueError(f"Missing edge columns: {missing}")
    edges = edges.copy()
    edges["taxid"] = edges["taxid"].astype(str)
    for column in FEATURES:
        edges[column] = pd.to_numeric(edges[column], errors="coerce").fillna(0)
    edges["partner_architecture"] = edges["partner_architecture"].fillna("NO_PFAM")
    edges["duf_architecture"] = edges["duf_architecture"].fillna("NO_PFAM")
    edges["protein_pair_key"] = edges.apply(
        lambda row: " || ".join(sorted((str(row.duf_protein_id), str(row.partner_protein_id)))), axis=1
    )
    return edges


def _module_key(edges: pd.DataFrame, complete_architectures: bool = False) -> pd.Series:
    if complete_architectures:
        return edges["duf_architecture"].astype(str) + " || " + edges["partner_architecture"].astype(str)
    return edges["duf_pfam_id"].astype(str) + " || " + edges["partner_architecture"].astype(str)


def _representatives(edges: pd.DataFrame, sort_columns: list[str]) -> pd.DataFrame:
    ordered = edges.sort_values(
        ["module_key", *sort_columns, "protein_pair_key"],
        ascending=[True, *([False] * len(sort_columns)), True],
        kind="mergesort",
    )
    return ordered.drop_duplicates("module_key", keep="first")


def select_recurrent_cohort(
    edges: pd.DataFrame,
    combined_cutoff: int,
    minimum_taxa: int,
    cohort: str,
) -> pd.DataFrame:
    selected = edges[edges["combined_score"].ge(combined_cutoff)].copy()
    selected["module_key"] = _module_key(selected)
    taxa = selected.groupby("module_key")["taxid"].nunique().rename("n_taxa")
    selected = selected.join(taxa, on="module_key")
    selected = selected[selected["n_taxa"].ge(minimum_taxa)]
    selected["tie_break_score"] = selected[
        ["neighborhood", "fusion", "experiments", "database"]
    ].max(axis=1)
    result = _representatives(selected, ["combined_score", "tie_break_score"])
    result.insert(0, "cohort", cohort)
    return result.drop(columns="tie_break_score")


def select_fusion_cohort(edges: pd.DataFrame) -> pd.DataFrame:
    selected = edges[edges["fusion"].gt(1)].copy()
    selected["module_key"] = _module_key(selected, complete_architectures=True)
    taxa = selected.groupby("module_key")["taxid"].nunique().rename("n_taxa")
    selected = selected.join(taxa, on="module_key")
    selected = selected[selected["n_taxa"].ge(5)]
    result = _representatives(selected, ["fusion", "combined_score"])
    result.insert(0, "cohort", "Fusion-supported")
    return result


def aggregate_modules(edges: pd.DataFrame) -> pd.DataFrame:
    selected = edges[edges["combined_score"].ge(900)].copy()
    selected["module_key"] = _module_key(selected)
    grouped = selected.groupby("module_key", sort=True)
    modules = grouped.agg(
        duf_pfam_id=("duf_pfam_id", "first"),
        duf_pfam_name=("duf_pfam_name", "first"),
        partner_architecture=("partner_architecture", "first"),
        n_edges=("protein_pair_key", "nunique"),
        n_taxa=("taxid", "nunique"),
        n_duf_proteins=("duf_protein_id", "nunique"),
        n_partner_proteins=("partner_protein_id", "nunique"),
    ).reset_index()
    maxima = grouped[FEATURES].max().add_prefix("max_").reset_index()
    return modules.merge(maxima, on="module_key", validate="one_to_one")


def select_l2_cohort(edges: pd.DataFrame, model_path: str | Path, cutoff: float = 0.5) -> pd.DataFrame:
    modules = score_modules(joblib.load(model_path), aggregate_modules(edges))
    selected_modules = modules[modules["l2_model_score"].ge(cutoff)].copy()
    eligible = edges[edges["combined_score"].ge(900)].copy()
    eligible["module_key"] = _module_key(eligible)
    eligible = eligible.merge(
        selected_modules[["module_key", "n_taxa", "l2_model_score"]],
        on="module_key",
        how="inner",
        validate="many_to_one",
    )
    result = _representatives(eligible, ["combined_score"])
    result.insert(0, "cohort", "L2 model")
    return result


def build_cohorts(edges: pd.DataFrame, model_path: str | Path) -> pd.DataFrame:
    cohorts = [
        select_recurrent_cohort(edges, 900, 10, "STRING >=900, taxa >=10"),
        select_recurrent_cohort(edges, 700, 50, "STRING >=700, taxa >=50"),
        select_fusion_cohort(edges),
        select_l2_cohort(edges, model_path),
    ]
    result = pd.concat(cohorts, ignore_index=True, sort=False)
    result["model_score"] = result.get("l2_model_score")
    result["string_combined_score"] = result["combined_score"]
    return result
