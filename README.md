# DUF-containing protein interaction screen

This portable publication bundle contains the model implementation, source TSV
tables, fitted model, prediction metadata, figure inputs, and code used to
reproduce the final figures for the DUF-containing protein interaction study.
The manuscript and AlphaFold structure files are deposited separately.

## Contents

- `notebooks/0725_bioinformatics_advances_selected_final_figures.ipynb`: code-only final selected-figures notebook.
- `src/dufppi/`: reusable model and figure code.
- `scripts/`: command-line figure generation, model fitting/scoring, and validation.
- `data/predictions/`: successful prediction metadata and all candidate records.
- `data/model/`: STRING900/taxa10 training data, out-of-fold predictions, coefficients, and fitted L2 pipeline.
- `data/candidates/`: all-organism and L2-selected module tables.
- `data/tables/`: source TSVs for the main and supplementary analyses.
- `data/figure_sources/`: inputs for the DUF4130 and DUF5819 structure figure.
- `figures/`: generated PNG, PDF, and SVG outputs.
- `metadata/`: data dictionary, provenance, manifest, and SHA-256 checksums.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[notebooks]'
```

## Reproduce figures

```bash
python scripts/generate_figures.py --release-root .
```

Raster figures are written at 500 DPI. The notebook provides the same entry
point without embedding machine-specific paths or working-project commentary.

## Refit and apply the L2 model

```bash
python scripts/train_l2_model.py --release-root .
python scripts/score_modules.py \
  --release-root . \
  --input data/candidates/all_organism_duf_modules.tsv.gz \
  --output results/scored_modules.tsv.gz
```

The class-weighted L2 output is an enrichment/ranking score, not a calibrated
interaction probability.

## Validate

```bash
python scripts/validate_release.py --release-root .
```

Validation checks cohort counts, required columns, model metrics, relative
paths, structure-file exclusion, and checksums.

## Confidence definitions

Liberal confidence is interface ipSAE >=0.2 and pLDDT >=70. Strict confidence
is interface ipSAE >=0.6 and pLDDT >=70. The final L2 cohort uses the
whole-complex residue-weighted mean pLDDT; comparison cohorts retain their
established interface-average pLDDT convention.
