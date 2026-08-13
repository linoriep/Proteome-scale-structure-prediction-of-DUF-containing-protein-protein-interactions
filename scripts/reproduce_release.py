#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(root: Path, script: str, *arguments: str) -> None:
    subprocess.run([sys.executable, str(root / "scripts" / script), *arguments], cwd=root, check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run(root, "train_l2_model.py")
    run(root, "build_paper_tables.py", "--output-directory", "results/tables")
    run(root, "analyze_model_sensitivity.py", "--output-directory", "results/tables")
    run(
        root,
        "compare_pfam_releases.py",
        "--output", "results/tables/paper_pfam38_2_retrospective_benchmark.tsv",
        "--supplementary-output", "results/tables/paper_supplementary_pfam38_2_comparison.tsv",
    )
    run(root, "generate_cohort_module_schematic.py")
    run(root, "generate_figures.py")
    run(root, "validate_release.py")


if __name__ == "__main__":
    main()
