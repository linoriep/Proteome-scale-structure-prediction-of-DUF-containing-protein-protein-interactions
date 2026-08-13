#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dufppi.model import FEATURES


def architecture(group: pd.DataFrame) -> str:
    ordered = group.sort_values(["start", "end", "pfam_id"])["pfam_id"].drop_duplicates()
    return ";".join(ordered) if len(ordered) else "NO_PFAM"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--string-links", type=Path, required=True)
    parser.add_argument("--pfam-hits", type=Path, required=True)
    parser.add_argument("--pfam-families", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/string_pfam_edges.tsv.gz"))
    args = parser.parse_args()
    links = pd.read_csv(args.string_links, sep="\t", low_memory=False)
    hits = pd.read_csv(args.pfam_hits, sep="\t", low_memory=False)
    families = pd.read_csv(args.pfam_families, sep="\t", low_memory=False)
    required_links = ["taxid", "protein1", "protein2", *FEATURES]
    required_hits = ["protein_id", "pfam_id", "start", "end"]
    required_families = ["pfam_id", "pfam_name", "is_duf"]
    for name, table, required in [("STRING links", links, required_links), ("Pfam hits", hits, required_hits), ("Pfam families", families, required_families)]:
        missing = [column for column in required if column not in table]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
    hits = hits.merge(families, on="pfam_id", how="left", validate="many_to_one")
    architectures = hits.groupby("protein_id", sort=False).apply(architecture).rename("architecture")
    dufs = hits[hits["is_duf"].astype(str).str.lower().isin(["true", "1"])]
    orientations = []
    for duf_side, partner_side in [("protein1", "protein2"), ("protein2", "protein1")]:
        oriented = links.merge(
            dufs[["protein_id", "pfam_id", "pfam_name"]], left_on=duf_side, right_on="protein_id", how="inner"
        )
        oriented["duf_protein_id"] = oriented[duf_side]
        oriented["partner_protein_id"] = oriented[partner_side]
        oriented["duf_pfam_id"] = oriented["pfam_id"]
        oriented["duf_pfam_name"] = oriented["pfam_name"]
        orientations.append(oriented)
    result = pd.concat(orientations, ignore_index=True)
    result = result.join(architectures.rename("duf_architecture"), on="duf_protein_id")
    result = result.join(architectures.rename("partner_architecture"), on="partner_protein_id")
    result[["duf_architecture", "partner_architecture"]] = result[["duf_architecture", "partner_architecture"]].fillna("NO_PFAM")
    columns = ["taxid", "duf_protein_id", "partner_protein_id", "duf_pfam_id", "duf_pfam_name", "duf_architecture", "partner_architecture", *FEATURES]
    result = result[columns].drop_duplicates()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(result):,} DUF-containing STRING edges")


if __name__ == "__main__":
    main()
