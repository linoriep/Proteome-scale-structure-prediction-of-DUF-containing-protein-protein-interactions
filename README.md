# Proteome-scale structure prediction of DUF-containing protein-protein interactions

This repository contains the analysis code, model files, source tables, and
figures associated with the manuscript. Predicted AlphaFold 3 structures are
distributed separately through the data repository cited in the manuscript.

## Contents

- `src/dufppi/`: model and figure code.
- `scripts/`: command-line figure generation, model fitting/scoring, and validation.
- `data/predictions/`: successful prediction metadata and all candidate records.
- `data/model/`: STRING >=900/taxa >=10 training data, out-of-fold predictions, coefficients, and fitted L2 pipeline.
- `data/candidates/`: all-organism and L2-selected module tables.
- `data/tables/`: source TSVs for the main and supplementary analyses.
- `data/figure_sources/`: inputs for the DUF4130 and DUF5819 structure figure.
- `data/structures/`: cofolded coordinate models for the two manuscript case studies.
- `figures/`: generated PNG, PDF, and SVG outputs.
- `metadata/`: data dictionary, provenance, manifest, and SHA-256 checksums.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Reproduce figures

```bash
python scripts/generate_figures.py --release-root .
```

## Reproduce the retrospective Pfam comparison

```bash
python scripts/compare_pfam_releases.py --release-root .
```

The script compares Pfam 38.0 DUF families in the final L2-model screen with
Pfam 38.2 and writes a source table and a 38-family
supplementary table. Family names, partner annotations, and confidence counts
are regenerated from the fixed Pfam releases and deposited prediction table;
the partner-context agreement classes are manually curated.

## Refit and apply the L2 model

```bash
python scripts/train_l2_model.py --release-root .
python scripts/score_modules.py \
  --release-root . \
  --input data/candidates/all_organism_duf_modules.tsv.gz \
  --output results/scored_modules.tsv.gz
```

## Data availability

The complete set of predicted structures and associated metadata is available from
[Zenodo](https://doi.org/10.5281/zenodo.21875362). This repository
contains the model implementation, code used for the analyses and figures, and
the two cofolded case-study models shown in Figure 4.

## Validate

```bash
python scripts/build_manifest.py --release-root .
python scripts/validate_release.py --release-root .
```

The first command rebuilds the relative-path file manifest and SHA-256
checksums. Validation checks cohort counts, model metrics, path portability,
the expected case-study structure files, and the release manifest.

## Confidence definitions

Liberal confidence is interface ipSAE >=0.2 and pLDDT >=70. Strict confidence
is interface ipSAE >=0.6 and pLDDT >=70. The final L2 cohort uses the
whole-complex residue-weighted mean pLDDT; comparison cohorts retain their
established interface-average pLDDT convention.
