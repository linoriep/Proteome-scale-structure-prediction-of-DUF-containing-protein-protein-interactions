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
    assert len(candidates) == 23049, len(candidates)
    assert int(candidates.candidate_status.eq("technical failure").sum()) == 79
    oof = pd.read_csv(root / "data/model/oof_predictions.tsv.gz", sep="\t")
    labels = pd.to_numeric(oof.liberal_high_confidence).astype(int); scores = pd.to_numeric(oof.l2_model_score)
    assert abs(average_precision_score(labels, scores) - .3716) < 5e-4
    assert abs(roc_auc_score(labels, scores) - .8003) < 5e-4
    assert not any(path.suffix.lower() in {".cif", ".pdb", ".bcif"} for path in root.rglob("*") if path.is_file())
    forbidden = (b"/" + b"nfs/", b"/" + b"homes/")
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".png", ".pdf", ".svg", ".joblib"}:
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                assert not any(token in chunk for token in forbidden), path
    checksums = root / "metadata/checksums.sha256"
    for line in checksums.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
    print(f"Validated {len(successful):,} successful predictions and {len(candidates):,} candidates")


if __name__ == "__main__":
    main()
