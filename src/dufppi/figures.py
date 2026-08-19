from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

ORDER = ["STRING >=900, taxa >=10", "STRING >=700, taxa >=50", "Fusion-supported", "L2 model"]
SHORT = dict(zip(ORDER, ["STRING ≥900/taxa ≥10", "STRING ≥700/taxa ≥50", "Fusion-supported", "L2 model\nscore ≥0.50"]))
COLORS = dict(zip(ORDER, ["#2676B8", "#8B5A9E", "#D17B24", "#33865A"]))
LIBERAL = "#33865A"
STRICT = "#4E5964"
FUNCTION_ORDER = [
    "metabolism / enzyme", "transport / membrane", "translation / RNA", "DNA / chromosome",
    "cell envelope / division", "regulation / signaling", "stress / defense",
    "unknown-function protein", "other / unclear",
]
FUNCTION_COLORS = dict(zip(FUNCTION_ORDER, ["#2A6F97", "#D17B24", "#6A994E", "#8B5A9E", "#C94C4C", "#3A8D8F", "#B07A2A", "#6C757D", "#C7CCD1"]))


def configure() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 800, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overview(root: Path) -> None:
    tables = root / "data/tables"
    predictions = pd.read_csv(root / "data/predictions/all_successful_predictions.tsv.gz", sep="\t", low_memory=False)
    annotations = pd.read_csv(root / "data/predictions/duf_annotation_rows.tsv.gz", sep="\t", low_memory=False)
    for frame in (predictions, annotations):
        for column in ("liberal_high_confidence", "strict_high_confidence"):
            frame[column] = frame[column].astype(str).str.lower().eq("true")
    confidence = pd.read_csv(tables / "paper_figure1_confidence_with_afdb.tsv", sep="\t")
    confidence.loc[confidence.cohort.eq("STRING >=900, taxa >=10"), "label"] = "STRING ≥900/taxa ≥10"
    confidence.loc[confidence.cohort.eq("STRING >=700, taxa >=50"), "label"] = "STRING ≥700/taxa ≥50"
    confidence.loc[confidence.cohort.eq("L2 model"), "label"] = "L2 model\nscore ≥0.50"
    subset = predictions[predictions.cohort.eq("L2 model") & pd.to_numeric(predictions.model_score, errors="coerce").ge(0.75)]
    confidence = pd.concat([confidence, pd.DataFrame([{
        "cohort": "L2 model >=0.75", "label": "L2 model\nscore ≥0.75", "n": len(subset),
        "liberal": int(subset.liberal_high_confidence.sum()), "strict": int(subset.strict_high_confidence.sum()),
        "liberal_fraction": subset.liberal_high_confidence.mean(), "strict_fraction": subset.strict_high_confidence.mean(),
    }])], ignore_index=True)
    run = pd.read_csv(tables / "paper_run_summary.tsv", sep="\t")
    afdb = pd.read_csv(tables / "paper_cohort_summary_failure_excluded.tsv", sep="\t")
    afdb = afdb[afdb.cohort.eq("AFDB heterodimer DUF-containing")].iloc[0]
    families = pd.concat([
        pd.DataFrame([{"cohort": "AFDB", "unique_duf_families": afdb.unique_duf_families, "duf_families_with_liberal_member": afdb.duf_families_with_liberal_member, "duf_families_with_strict_member": afdb.duf_families_with_strict_member}]),
        run[["cohort", "unique_duf_families", "duf_families_with_liberal_member", "duf_families_with_strict_member"]],
    ], ignore_index=True)
    multiplicity = pd.read_csv(tables / "paper_duf_liberal_partner_multiplicity_bins.tsv", sep="\t")
    recurrence = pd.read_csv(tables / "paper_all_runs_taxa_recurrence_yield_to500.tsv", sep="\t")
    composition_data = pd.read_csv(tables / "paper_partner_function_cohort_composition_liberal.tsv", sep="\t")
    composition = composition_data.pivot(index="cohort", columns="partner_function_class", values="pair_fraction").reindex(index=ORDER, columns=FUNCTION_ORDER).fillna(0)
    domains = pd.read_csv(tables / "paper_taxonomic_domain_summary.tsv", sep="\t")
    score = pd.read_csv(tables / "paper_final_model_score_yield.tsv", sep="\t")

    fig = plt.figure(figsize=(15.6, 12.8))
    grid = fig.add_gridspec(3, 3, height_ratios=[1, .82, 1], hspace=.80, wspace=.34)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[2, 1]), fig.add_subplot(grid[0, 2])]
    function_ax = fig.add_subplot(grid[1, :])
    domain_ax = fig.add_subplot(grid[2, 0])
    score_ax = fig.add_subplot(grid[2, 2])

    x = np.arange(len(confidence)); width = .34
    axes[0].bar(x-width/2, confidence.liberal_fraction, width, color=LIBERAL, label="Liberal")
    axes[0].bar(x+width/2, confidence.strict_fraction, width, color=STRICT, label="Strict")
    axes[0].set_xticks(x, confidence.label, rotation=27, ha="right", fontsize=7); axes[0].set_ylabel("Fraction of predicted pairs"); axes[0].yaxis.set_major_formatter(PercentFormatter(1)); axes[0].set_title("A  Structural-confidence fraction", loc="left", fontweight="bold"); axes[0].legend(frameon=False)

    labels = families.cohort.map(SHORT).fillna(families.cohort)
    x = np.arange(len(families)); width = .25
    for offset, column, label, color in [(-width, "unique_duf_families", "Candidate", "#C8CDD2"), (0, "duf_families_with_liberal_member", "Liberal", LIBERAL), (width, "duf_families_with_strict_member", "Strict", STRICT)]:
        bars = axes[1].bar(x+offset, families[column], width, color=color, label=label)
        axes[1].bar_label(bars, rotation=90, padding=2, fontsize=6)
    axes[1].set_xticks(x, labels, rotation=28, ha="right", fontsize=7); axes[1].set_ylabel("Distinct DUF families"); axes[1].set_ylim(0, 3000); axes[1].set_title("B  DUF-family coverage", loc="left", fontweight="bold"); axes[1].legend(frameon=False, ncol=3, loc="upper center")

    bins = ["1-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250-499", ">=500"]
    x = np.arange(len(bins))
    for cohort in ORDER:
        part = recurrence[recurrence.cohort.eq(cohort)].set_index("taxa_bin").reindex(bins)
        axes[2].plot(x, part.liberal_fraction, "o-", color=COLORS[cohort], label=SHORT[cohort])
    axes[2].set_xticks(x, bins, rotation=25, ha="right"); axes[2].set_xlabel("Taxa per module"); axes[2].set_ylabel("Liberal-confidence fraction"); axes[2].set_ylim(0, recurrence.liberal_fraction.max() * 1.45); axes[2].yaxis.set_major_formatter(PercentFormatter(1)); axes[2].set_title("F  Yield by taxonomic recurrence", loc="left", fontweight="bold"); axes[2].legend(frameon=False, ncol=2, fontsize=6.5, loc="upper center")

    partner_bins = ["0", "1", "2", "3-4", "5-9", ">=10"]
    matrix = multiplicity.pivot(index="partner_bin", columns="cohort", values="n_duf_proteins").reindex(index=partner_bins, columns=ORDER).fillna(0)
    x = np.arange(len(partner_bins)); group_width = .78; bar_width = group_width / len(ORDER)
    for index, cohort in enumerate(ORDER):
        axes[3].bar(x-group_width/2+bar_width/2+index*bar_width, matrix[cohort], bar_width, color=COLORS[cohort], label=SHORT[cohort])
    axes[3].set_yscale("log"); axes[3].set_xticks(x, partner_bins); axes[3].set_xlabel("Liberal-confidence partners per DUF protein"); axes[3].set_ylabel("DUF-containing proteins (log scale)"); axes[3].set_title("C  DUF partner multiplicity", loc="left", fontweight="bold"); axes[3].legend(frameon=False, ncol=2, fontsize=6)

    left = np.zeros(len(composition))
    for category in FUNCTION_ORDER:
        values = composition[category].to_numpy(); function_ax.barh(np.arange(len(ORDER)), values, left=left, color=FUNCTION_COLORS[category], label=category); left += values
    function_ax.set_yticks(np.arange(len(ORDER)), [SHORT[item] for item in ORDER]); function_ax.invert_yaxis(); function_ax.set_xlim(0, 1); function_ax.xaxis.set_major_formatter(PercentFormatter(1)); function_ax.set_xlabel("Fraction of liberal-confidence complexes"); function_ax.set_title("D  Partner-function composition", loc="left", fontweight="bold"); function_ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.20))

    for ax, data, title, xlabel in [(domain_ax, domains, "E  Yield by taxonomic domain", "domain"), (score_ax, score, "G  Structural yield by L2 score", "score_bin")]:
        x = np.arange(len(data)); width = .34
        ax.bar(x-width/2, data.liberal_fraction, width, color=LIBERAL, label="Liberal"); ax.bar(x+width/2, data.strict_fraction, width, color=STRICT, label="Strict")
        names = [f"{getattr(row, xlabel)}\nn={int(row.n):,}" for row in data.itertuples()]
        rotation = 32 if xlabel == "score_bin" else 0
        font_size = 7.5 if xlabel == "score_bin" else 9
        ax.set_xticks(x, names, rotation=rotation, ha="right" if xlabel == "score_bin" else "center", fontsize=font_size); ax.set_ylabel("Fraction within group"); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_title(title, loc="left", fontweight="bold"); ax.legend(frameon=False)
    fig.tight_layout(rect=[0, .04, 1, 1])
    save(fig, root / "figures", "figure_2_cohort_overview")


def locus_distance(root: Path) -> None:
    data = pd.read_csv(root / "data/tables/paper_gene_distance_confidence_yield.tsv", sep="\t")
    bins = ["1", "2-5", "6-20", "21-100", "101-1000", ">1000"]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharey=True)
    for ax, cohort in zip(axes.ravel(), ORDER):
        cohort_bins = bins + (["different replicon"] if cohort == "L2 model" else [])
        part = data[data.cohort.eq(cohort)].set_index("distance_bin").reindex(cohort_bins).reset_index()
        x = np.arange(len(part)); width = .36
        ax.bar(x-width/2, part.liberal_fraction, width, color=LIBERAL, label="Liberal"); ax.bar(x+width/2, part.strict_fraction, width, color=STRICT, label="Strict")
        for index, row in part.iterrows():
            if row.n:
                ax.text(index-width/2, row.liberal_fraction+.008, f"n={int(row.liberal):,}/{int(row.n):,}", rotation=90, ha="center", va="bottom", fontsize=6)
                ax.text(index+width/2, row.strict_fraction+.008, f"n={int(row.strict):,}/{int(row.n):,}", rotation=90, ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x, [value.replace(" ", "\n") if value == "different replicon" else value for value in cohort_bins], rotation=28, ha="right"); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_title(SHORT[cohort], fontweight="bold"); ax.legend(frameon=False)
    axes[0, 0].set_ylabel("Fraction of successfully predicted pairs"); axes[1, 0].set_ylabel("Fraction of successfully predicted pairs")
    fig.suptitle("Structural-confidence yield by locus-tag distance", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .96])
    save(fig, root / "figures", "figure_3_locus_distance")


def case_studies(root: Path) -> None:
    panels = [
        ("A", "MiaB tRNA-modifying radical-SAM\nenzyme (PDB 7MJV)", "A_duf4130_sam_reference_7mjv.png"),
        ("B", "Predicted DUF4130-radical-SAM\ncomplex with SAM", "B_duf4130_af3_prediction_sam.png"),
        ("C", "RecJ DNA exonuclease\n(PDB 5F55)", "C_duf4130_dna_binding_reference_5f55.png"),
        ("D", "Human GGCX-protein S complex\n(PDB 9L54)", "D_duf5819_ggcx_reference.png"),
        ("E", "Predicted DUF5819-HTTM/DCC1-like\ncomplex with menaquinone-7", "E_duf5819_af3_prediction_vitamin_k.png"),
        ("F", "Close-up of the binding pocket in superposed\nPDB 9L54 and predicted DUF5819 complex", "F_duf5819_vitamin_k_pocket_closeup.png"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 8.0))
    for ax, (letter, title, filename) in zip(axes.ravel(), panels):
        heading_y = 1.12 if letter == "F" else 1.06
        image = plt.imread(root / "data/figure_sources" / filename)
        ax.imshow(image)
        if letter in "ABC":
            margin_x = image.shape[1] * .09
            margin_y = image.shape[0] * .09
            ax.set_xlim(-margin_x, image.shape[1] + margin_x)
            ax.set_ylim(image.shape[0] + margin_y, -margin_y)
        ax.axis("off"); ax.text(0, heading_y, letter, transform=ax.transAxes, va="top", fontweight="bold", fontsize=12); ax.text(.08, heading_y, title, transform=ax.transAxes, va="top", fontweight="bold", fontsize=9)
    fig.text(.01, .73, "DUF4130", rotation=90, fontweight="bold"); fig.text(.01, .27, "DUF5819", rotation=90, fontweight="bold"); fig.tight_layout(rect=[.02, 0, 1, 1])
    save(fig, root / "figures", "figure_4_case_studies")


def evidence_channels(root: Path) -> None:
    liberal = pd.read_csv(root / "data/tables/paper_string_all_channels_score_bin_confidence.tsv", sep="\t")
    strict = pd.read_csv(root / "data/tables/paper_string_all_channels_score_bin_strict_confidence.tsv", sep="\t")
    keys = ["channel", "score_bin", "source_column"]
    data = liberal.merge(
        strict[keys + ["n", "strict_fraction"]], on=keys, validate="one_to_one", suffixes=("", "_strict")
    )
    if not data["n"].equals(data["n_strict"]):
        raise ValueError("Liberal and strict channel tables use different candidate counts")
    channels = list(dict.fromkeys(data.channel))
    fig, axes = plt.subplots(4, 4, figsize=(15.4, 13)); axes = axes.ravel()
    for index, (ax, channel) in enumerate(zip(axes, channels)):
        part = data[data.channel.eq(channel)].copy(); x = np.arange(len(part))
        ax.bar(x, part.n, color="#8095A5"); ax.set_yscale("log"); ax.set_ylim(1, 20000); ax.set_yticks([10, 100, 1000, 10000]); ax.set_title(channel, loc="left", fontweight="bold"); ax.set_xticks(x[::2], part.score_bin.astype(str).iloc[::2], rotation=45, ha="right", fontsize=6)
        twin = ax.twinx()
        twin.plot(x, part.liberal_fraction, "o-", color=LIBERAL, ms=3)
        twin.plot(x, part.strict_fraction, "o-", color=STRICT, ms=3)
        twin.set_ylim(0, 1); twin.set_yticks([0, .5, 1]); twin.yaxis.set_major_formatter(PercentFormatter(1))
        if index == 0:
            ax.set_ylabel("Candidate count (log scale)")
            ax.set_xlabel("STRING combined score interval")
            twin.set_ylabel("Confidence fraction", rotation=270, labelpad=14)
    for ax in axes[len(channels):]: ax.axis("off")
    fig.legend(handles=[Patch(color="#8095A5", label="Candidate count"), Line2D([0], [0], color=LIBERAL, marker="o", label="Liberal confidence fraction"), Line2D([0], [0], color=STRICT, marker="o", label="Strict confidence fraction")], ncol=3, frameon=False, loc="upper center")
    fig.tight_layout(rect=[0, .02, 1, .96]); save(fig, root / "figures", "figure_S1_string_channels_confidence")


def generate_all(root: str | Path) -> None:
    root = Path(root).resolve(); configure(); overview(root); locus_distance(root); case_studies(root); evidence_channels(root)
