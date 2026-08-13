#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(); root = args.release_root.resolve()
    successful = pd.read_csv(root / "data/predictions/all_successful_predictions.tsv.gz", sep="\t", low_memory=False)
    expected = {"STRING >=900, taxa >=10": 4520, "STRING >=700, taxa >=50": 4017, "Fusion-supported": 2135, "L2 model": 12298}
    observed = successful.groupby("cohort").size().to_dict()
    assert observed == expected, (observed, expected)
    candidates = pd.read_csv(root / "data/predictions/all_candidates.tsv.gz", sep="\t", low_memory=False)
    assert len(candidates) == 23048, len(candidates)
    assert int(candidates.candidate_status.eq("technical failure").sum()) == 78
    oof = pd.read_csv(root / "data/model/oof_predictions.tsv.gz", sep="\t")
    labels = pd.to_numeric(oof.liberal_high_confidence).astype(int); scores = pd.to_numeric(oof.l2_model_score)
    assert abs(average_precision_score(labels, scores) - .3716) < 5e-4
    assert abs(roc_auc_score(labels, scores) - .8003) < 5e-4
    baseline = pd.read_csv(root / "data/model/baseline_ranking_metrics_reproduced.tsv", sep="\t").set_index("ranking")
    expected_baselines = {
        "L2 model": (.3716, .8003),
        "Combined STRING score": (.1419, .5245),
        "Neighborhood score": (.2440, .7373),
        "Co-occurrence score": (.3163, .7806),
    }
    for ranking, (expected_ap, expected_auroc) in expected_baselines.items():
        assert abs(baseline.at[ranking, "average_precision"] - expected_ap) < 5e-4
        assert abs(baseline.at[ranking, "auroc"] - expected_auroc) < 5e-4
    pfam_benchmark = pd.read_csv(root / "data/tables/paper_pfam38_2_retrospective_benchmark.tsv", sep="\t")
    assert len(pfam_benchmark) == 38
    assert pfam_benchmark.agreement_class.value_counts().to_dict() == {"specific": 15, "broader process": 15, "no clear relationship": 8}
    specific = pfam_benchmark[pfam_benchmark.agreement_class.eq("specific")]
    assert int(specific.liberal_pairs.gt(0).sum()) == 12
    assert int(specific.strict_pairs.gt(0).sum()) == 7
    pfam_supplement = pd.read_csv(root / "data/tables/paper_supplementary_pfam38_2_comparison.tsv", sep="\t")
    assert len(pfam_supplement) == 38
    external = pd.read_csv(root / "metadata/external_inputs.tsv", sep="\t")
    for relative in external["path"]:
        assert (root / relative).is_file(), relative
    table_provenance = pd.read_csv(root / "metadata/table_provenance.tsv", sep="\t")
    deposited_tables = {path.name for path in (root / "data/tables").glob("*.tsv")}
    assert deposited_tables == set(table_provenance["table"])
    structure_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".cif", ".pdb", ".bcif"}
    }
    assert structure_files == {
        "data/structures/duf4130_sam_cofolded_model.pdb",
        "data/structures/duf5819_menaquinone7_cofolded_model.pdb",
    }
    forbidden = (
        b"/" + b"nfs/", b"/" + b"homes/", b"/" + b"home/", b"/" + b"mnt/",
        b"de" + b"ulf", b"." + b"codex", b"current" + b"_release",
        b"from" + b"_deposit", b"DESKTOP" + b"-", b"file" + b"://",
    )
    for path in root.rglob("*"):
        if (
            ".git" in path.parts
            or "__pycache__" in path.parts
            or not path.is_file()
            or path.suffix.lower() in {".pyc", ".png", ".pdf", ".svg", ".joblib"}
        ):
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                assert not any(token in chunk for token in forbidden), path
    checksums = root / "metadata/checksums.sha256"
    checksum_rows = {}
    for line in checksums.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
        checksum_rows[relative] = digest
    manifest = pd.read_csv(root / "metadata/file_manifest.tsv", sep="\t")
    assert set(manifest.path) == set(checksum_rows)
    for row in manifest.itertuples(index=False):
        path = root / row.path
        assert path.stat().st_size == row.size_bytes, row.path
        assert row.sha256 == checksum_rows[row.path], row.path
    print(f"Validated {len(successful):,} successful predictions and {len(candidates):,} candidates")


if __name__ == "__main__":
    main()
