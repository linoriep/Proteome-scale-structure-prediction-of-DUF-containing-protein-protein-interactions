from __future__ import annotations

import numpy as np
import pandas as pd


COHORT_ORDER = [
    "STRING >=900, taxa >=10",
    "STRING >=700, taxa >=50",
    "Fusion-supported",
    "L2 model",
]
CATEGORY_ORDER = [
    "metabolism / enzyme", "transport / membrane", "translation / RNA", "DNA / chromosome",
    "cell envelope / division", "regulation / signaling", "stress / defense",
    "unknown-function protein", "other / unclear",
]


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - half, centre + half


def run_summary(predictions: pd.DataFrame, annotations: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for cohort in COHORT_ORDER:
        group = predictions[predictions["cohort"].eq(cohort)]
        liberal = int(group["liberal_high_confidence"].sum())
        strict = int(group["strict_high_confidence"].sum())
        annotated = group if annotations is None else annotations[annotations["cohort"].eq(cohort)]
        interacting = annotated.groupby("prediction_name")["duf_domain_interacting"].any()
        liberal_low, liberal_high = wilson(liberal, len(group))
        rows.append(
            {
                "cohort": cohort,
                "predictions": len(group),
                "models_available": int(group["model_available"].sum()),
                "interface_scored": int(group["interface_scored"].sum()),
                "liberal_high_confidence": liberal,
                "liberal_fraction": liberal / len(group) if len(group) else np.nan,
                "liberal_ci_low": liberal_low,
                "liberal_ci_high": liberal_high,
                "strict_high_confidence": strict,
                "strict_fraction": strict / len(group) if len(group) else np.nan,
                "unique_duf_families": group["duf_pfam_id"].nunique(),
                "duf_families_with_liberal_member": annotated.loc[annotated["liberal_high_confidence"], "duf_pfam_id"].nunique(),
                "duf_families_with_strict_member": annotated.loc[annotated["strict_high_confidence"], "duf_pfam_id"].nunique(),
                "all_structures_with_interacting_duf": int(interacting.sum()),
                "liberal_structures_with_interacting_duf": int(interacting.reindex(group.loc[group["liberal_high_confidence"], "prediction_name"]).fillna(False).sum()),
                "strict_structures_with_interacting_duf": int(interacting.reindex(group.loc[group["strict_high_confidence"], "prediction_name"]).fillna(False).sum()),
                "unique_duf_proteins": group["duf_protein_id"].nunique(),
                "unique_partner_proteins": group["partner_protein_id"].nunique(),
                "unique_pairs": group["protein_pair_key"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def score_yield(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions[predictions["cohort"].eq("L2 model")].copy()
    bins = [0.5, 0.6, 0.7, 0.75, 0.8, np.inf]
    labels = ["[0.5, 0.6)", "[0.6, 0.7)", "[0.7, 0.75)", "[0.75, 0.8)", "[0.8, 1.001)"]
    data["score_bin"] = pd.cut(data["model_score"], bins=bins, labels=labels, right=False)
    rows = []
    for label in labels:
        group = data[data["score_bin"].eq(label)]
        liberal = int(group["liberal_high_confidence"].sum())
        strict = int(group["strict_high_confidence"].sum())
        liberal_ci = wilson(liberal, len(group))
        strict_ci = wilson(strict, len(group))
        rows.append({
            "score_bin": label, "n": len(group), "liberal": liberal, "strict": strict,
            "liberal_fraction": liberal / len(group) if len(group) else np.nan,
            "strict_fraction": strict / len(group) if len(group) else np.nan,
            "liberal_ci_low": liberal_ci[0], "liberal_ci_high": liberal_ci[1],
            "strict_ci_low": strict_ci[0], "strict_ci_high": strict_ci[1],
        })
    return pd.DataFrame(rows)


def taxa_recurrence(predictions: pd.DataFrame) -> pd.DataFrame:
    bins = [1, 5, 10, 20, 50, 100, 250, 500, np.inf]
    labels = ["1-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250-499", ">=500"]
    data = predictions.copy()
    data["taxa_bin"] = pd.cut(data["n_taxa"], bins=bins, labels=labels, right=False)
    rows = []
    for cohort in COHORT_ORDER:
        for label in labels:
            group = data[data["cohort"].eq(cohort) & data["taxa_bin"].eq(label)]
            liberal = int(group["liberal_high_confidence"].sum())
            strict = int(group["strict_high_confidence"].sum())
            rows.append({
                "cohort": cohort, "taxa_bin": label, "n": len(group), "liberal": liberal,
                "strict": strict, "duf_interacting": int(group["duf_domain_interacting"].sum()),
                "liberal_fraction": liberal / len(group) if len(group) else np.nan,
                "strict_fraction": strict / len(group) if len(group) else np.nan,
            })
    return pd.DataFrame(rows)


def partner_composition(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in COHORT_ORDER:
        group = predictions[predictions["cohort"].eq(cohort) & predictions["liberal_high_confidence"]]
        counts = group["partner_function_class"].value_counts(dropna=False)
        for category in sorted(CATEGORY_ORDER):
            count = int(counts.get(category, 0))
            rows.append({"cohort": cohort, "partner_function_class": category, "n_pairs": count,
                         "cohort_total": len(group), "pair_fraction": count / len(group)})
    return pd.DataFrame(rows)


def confident_composition(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, column in [("Liberal confidence", "liberal_high_confidence"), ("Strict confidence", "strict_high_confidence")]:
        for cohort in COHORT_ORDER:
            group = predictions[predictions["cohort"].eq(cohort) & predictions[column]]
            counts = group["partner_function_class"].value_counts()
            for category in CATEGORY_ORDER:
                count = int(counts.get(category, 0))
                rows.append({"confidence_subset": label, "cohort": cohort, "partner_function_class": category,
                             "n_pairs": count, "subset_total": len(group), "fraction_within_subset": count / len(group)})
    return pd.DataFrame(rows)


def family_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions[predictions["cohort"].eq("L2 model")]
    rows = []
    for category in CATEGORY_ORDER:
        group = predictions[predictions["partner_function_class"].eq(category)]
        rows.append({
            "partner_function_class": category,
            "candidate_duf_families": group["duf_pfam_id"].nunique(),
            "liberal_interacting_duf_families": group.loc[group["liberal_high_confidence"] & group["duf_domain_interacting"], "duf_pfam_id"].nunique(),
            "strict_interacting_duf_families": group.loc[group["strict_high_confidence"] & group["duf_domain_interacting"], "duf_pfam_id"].nunique(),
        })
    return pd.DataFrame(rows)


def multiplicity_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    labels = ["0", "1", "2", "3-4", "5-9", ">=10"]
    rows = []
    for cohort in COHORT_ORDER:
        group = predictions[predictions["cohort"].eq(cohort)]
        all_dufs = pd.Index(group["duf_protein_id"].dropna().unique())
        counts = group[group["liberal_high_confidence"]].groupby("duf_protein_id")["partner_protein_id"].nunique().reindex(all_dufs, fill_value=0)
        binned = pd.cut(counts, bins=[-1, 0, 1, 2, 4, 9, np.inf], labels=labels)
        for label in labels:
            rows.append({"cohort": cohort, "partner_bin": label, "n_duf_proteins": int(binned.eq(label).sum())})
    return pd.DataFrame(rows)


def multiple_partner_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in COHORT_ORDER:
        group = predictions[predictions["cohort"].eq(cohort)]
        counts = group.groupby("duf_protein_id")["partner_protein_id"].nunique()
        liberal = group[group["liberal_high_confidence"]].groupby("duf_protein_id")["partner_protein_id"].nunique()
        strict = group[group["strict_high_confidence"]].groupby("duf_protein_id")["partner_protein_id"].nunique()
        for threshold in [2, 5, 10]:
            rows.append({"cohort": cohort, "minimum_partners": threshold, "duf_proteins": len(counts),
                         "candidate_n": int(counts.ge(threshold).sum()), "liberal_n": int(liberal.ge(threshold).sum()),
                         "strict_n": int(strict.ge(threshold).sum())})
    return pd.DataFrame(rows)


def exclusion_report(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in COHORT_ORDER:
        group = candidates[candidates["cohort"].eq(cohort)]
        success = group[group["candidate_status"].eq("success")]
        rows.append({
            "cohort": cohort, "input_rows": len(group), "after_source_filter": len(group),
            "successful_structure_rows": len(success), "unique_protein_pairs": success["protein_pair_key"].nunique(),
            "source_or_viewer_rows_excluded": 0,
            "technical_failure_rows_excluded": int(group["candidate_status"].eq("technical failure").sum()),
            "duplicate_pair_annotation_rows": len(success) - success["protein_pair_key"].nunique(),
        })
    return pd.DataFrame(rows)


def taxonomic_summary(predictions: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    if not {"taxid", "domain"}.issubset(taxonomy.columns):
        raise ValueError("Taxonomy table requires taxid and domain columns")
    data = predictions[predictions["cohort"].eq("L2 model")].copy()
    data["taxid"] = data["taxid"].astype(str).str.replace(r"\.0$", "", regex=True)
    mapping = taxonomy.assign(taxid=taxonomy["taxid"].astype(str)).drop_duplicates("taxid")
    data = data.merge(mapping[["taxid", "domain"]], on="taxid", how="left", validate="many_to_one")
    if data["domain"].isna().any():
        raise ValueError(f"Missing taxonomy for {data.loc[data['domain'].isna(), 'taxid'].nunique()} taxids")
    rows = []
    for domain in ["Bacteria", "Eukaryotes", "Archaea"]:
        group = data[data["domain"].eq(domain)]
        liberal = int(group["liberal_high_confidence"].sum())
        strict = int(group["strict_high_confidence"].sum())
        liberal_ci = wilson(liberal, len(group)); strict_ci = wilson(strict, len(group))
        rows.append({
            "domain": domain, "n": len(group), "liberal": liberal, "strict": strict,
            "duf_families": group["duf_pfam_id"].nunique(), "pair_fraction": len(group) / len(data),
            "liberal_fraction": liberal / len(group), "strict_fraction": strict / len(group),
            "liberal_ci_low": liberal_ci[0], "liberal_ci_high": liberal_ci[1],
            "strict_ci_low": strict_ci[0], "strict_ci_high": strict_ci[1],
        })
    return pd.DataFrame(rows)


def gene_distance_tables(predictions: pd.DataFrame, locations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["protein_id", "replicon", "strand", "locus_index"]
    missing = [column for column in required if column not in locations]
    if missing:
        raise ValueError(f"Missing genome-location columns: {missing}")
    left = locations.add_prefix("duf_")
    right = locations.add_prefix("partner_")
    data = predictions.merge(left, left_on="duf_protein_id", right_on="duf_protein_id", how="left")
    data = data.merge(right, left_on="partner_protein_id", right_on="partner_protein_id", how="left")
    data["parse_status"] = "parsed"
    missing_location = data[["duf_replicon", "partner_replicon", "duf_locus_index", "partner_locus_index"]].isna().any(axis=1)
    data.loc[missing_location, "parse_status"] = "unparsed locus tag"
    different = data["duf_replicon"].ne(data["partner_replicon"]) & ~missing_location
    data.loc[different, "parse_status"] = "different contig/chromosome"
    data["distance"] = (pd.to_numeric(data["duf_locus_index"], errors="coerce") - pd.to_numeric(data["partner_locus_index"], errors="coerce")).abs()
    labels = ["1", "2", "3", "4", "5", "6-10", "11-20", ">20"]
    data["distance_bin"] = pd.cut(data["distance"], bins=[0, 1, 2, 3, 4, 5, 10, 20, np.inf], labels=labels, include_lowest=True)
    rows = []
    for cohort in COHORT_ORDER:
        for label in labels:
            group = data[data["cohort"].eq(cohort) & data["parse_status"].eq("parsed") & data["distance_bin"].eq(label)]
            rows.append({"cohort": cohort, "distance_bin": label, "n": len(group),
                         "liberal": int(group["liberal_high_confidence"].sum()), "strict": int(group["strict_high_confidence"].sum()),
                         "liberal_fraction": group["liberal_high_confidence"].mean(), "strict_fraction": group["strict_high_confidence"].mean()})
    parse = data.groupby(["cohort", "parse_status"], sort=False).size().rename("n").reset_index()
    parse["fraction"] = parse["n"] / parse.groupby("cohort")["n"].transform("sum")
    return pd.DataFrame(rows), parse
