from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures"

COLORS = {
    "cohort_1": "#2676B8",
    "cohort_2": "#8B5A9E",
    "cohort_3": "#D17B24",
    "cohort_4": "#33865A",
    "duf": "#C94C4C",
    "pfam_a": "#2A6F97",
    "pfam_b": "#6A994E",
    "neutral": "#65727E",
    "light": "#E7EAED",
    "line": "#3F4850",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 800,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def arrow(ax, start, end, *, color=None, width=1.2, style="-") -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=width,
        linestyle=style,
        color=color or COLORS["line"],
        shrinkA=0,
        shrinkB=2,
    )
    ax.add_patch(patch)


def cohort_box(ax, x, y, color, number, lines, *, heading_size=8.4, body_size=7.6) -> None:
    width, height = 0.415, 0.148
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.007,rounding_size=0.008",
        facecolor="white",
        edgecolor=color,
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.add_patch(Rectangle((x, y), 0.058, height, facecolor=color, edgecolor="none"))
    ax.text(x + 0.029, y + height * 0.67, "COHORT", color="white", fontsize=4.8,
            fontweight="bold", ha="center", va="center")
    ax.text(x + 0.029, y + height * 0.39, str(number), color="white", fontsize=11,
            fontweight="bold", ha="center", va="center")
    ax.text(x + 0.072, y + height - 0.027, lines[0], color=color,
            fontweight="bold", ha="left", va="top", fontsize=heading_size)
    ax.text(x + 0.072, y + height - 0.058, "\n".join(lines[1:]),
            color="#222222", ha="left", va="top", fontsize=body_size, linespacing=1.15)


def protein(ax, x, y, domains, *, width=0.19, height=0.026) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor="#D9DEE2", edgecolor=COLORS["line"], linewidth=0.7,
        )
    )
    for start, span, color, label in domains:
        ax.add_patch(
            FancyBboxPatch(
                (x + start * width, y - 0.002), span * width, height + 0.004,
                boxstyle="round,pad=0.001,rounding_size=0.005",
                facecolor=color, edgecolor="white", linewidth=0.5,
            )
        )
        if label:
            ax.text(x + (start + span / 2) * width, y + height / 2, label,
                    color="white", ha="center", va="center", fontsize=6.4,
                    fontweight="bold")


def build_figure() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.1), gridspec_kw={"width_ratios": [1.03, 0.97]})
    left, right = axes
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0.10, 1)
        ax.axis("off")
    right.set_xlim(0, 0.90)

    left.text(0.055, 0.980, "Selection of the four prediction cohorts", fontsize=10,
              fontweight="bold", va="top")

    source = FancyBboxPatch(
        (0.13, 0.84), 0.72, 0.085,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="#F1F3F4", edgecolor=COLORS["neutral"], linewidth=1.0, zorder=3,
    )
    left.add_patch(source)
    left.text(0.49, 0.885, "STRING v12.0 associations involving DUF-containing proteins",
              ha="center", va="center", fontsize=8.2, fontweight="bold", zorder=4)
    left.text(0.49, 0.858, "Pfam 38.0 architectures assigned and recurrence counted across distinct taxa",
              ha="center", va="center", fontsize=7.2, color=COLORS["neutral"], zorder=4)

    cohort_box(left, 0.06, 0.605, COLORS["cohort_1"], 1,
               ["STRING ≥900/taxa ≥10", "Bacterial STRING interactions",
                "Combined score ≥900", "Modules containing pairs from ≥10 taxa", "Number of predictions: 4,523"],
               body_size=7.2)
    cohort_box(left, 0.54, 0.605, COLORS["cohort_2"], 2,
               ["STRING ≥700/taxa ≥50", "Bacterial STRING interactions",
                "Combined score ≥700", "Modules containing pairs from ≥50 taxa", "Number of predictions: 4,017"],
               body_size=7.2)
    cohort_box(left, 0.06, 0.382, COLORS["cohort_3"], 3,
               ["Fusion-supported", "Bacterial STRING interactions",
                "Fusion score >1", "Modules containing pairs from ≥5 taxa",
                "Number of predictions: 2,135"],
               body_size=7.1)
    cohort_box(left, 0.54, 0.382, COLORS["cohort_4"], 4,
               ["L2-model selected", "Logistic regression", "Trained on cohort 1",
                "All 12,535 STRING organisms", "Combined score ≥900, L2 score ≥0.5",
                "Number of predictions: 12,374"],
               heading_size=8.1, body_size=7.1)

    cohort_group = FancyBboxPatch(
        (0.025, 0.360), 0.955, 0.415,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="none", edgecolor="#A8B0B7", linewidth=0.9,
        linestyle=(0, (4, 3)), zorder=0,
    )
    left.add_patch(cohort_group)
    arrow(left, (0.49, 0.825), (0.49, 0.786), color=COLORS["line"], width=1.0)

    selection = FancyBboxPatch(
        (0.16, 0.155), 0.66, 0.10,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        facecolor="white", edgecolor=COLORS["neutral"], linewidth=1.0,
    )
    left.add_patch(selection)
    left.text(0.49, 0.220, "One protein pair selected per module",
              ha="center", va="center", fontsize=8.2, fontweight="bold")
    left.text(0.49, 0.183, "AlphaFold 3 complex prediction and AlphaJudge analysis",
              ha="center", va="center", fontsize=7.4, color=COLORS["neutral"])
    arrow(left, (0.49, 0.322), (0.49, 0.268), color=COLORS["line"], width=1.0)
    right.text(0.040, 0.980, "Definition of a module", fontsize=10,
               fontweight="bold", va="top")
    right.text(0.040, 0.925,
               "Pairs were grouped together when the DUF family and partner Pfam architecture were the same",
               fontsize=7.7, color=COLORS["neutral"], va="top")

    rows = [
        ("Taxon 1", 0.76, 0.170, 0.250, (0.18, 0.64), 0.4625, 0.275, (0.065, 0.40), (0.515, 0.42)),
        ("Taxon 1", 0.63, 0.170, 0.250, (0.08, 0.64), 0.4625, 0.275, (0.015, 0.40), (0.465, 0.42)),
        ("Taxon 2", 0.50, 0.170, 0.250, (0.18, 0.64), 0.4625, 0.275, (0.065, 0.40), (0.515, 0.42)),
    ]
    for taxon, y, duf_x, duf_width, duf_domain, partner_x, partner_width, pf_a, pf_b in rows:
        right.text(0.060, y + 0.012, taxon, ha="left", va="center", fontsize=7.2,
                   color=COLORS["neutral"])
        protein(right, duf_x, y, [(duf_domain[0], duf_domain[1], COLORS["duf"], "DUF-X")],
                width=duf_width, height=0.030)
        protein(right, partner_x, y,
                [(pf_a[0], pf_a[1], COLORS["pfam_a"], "PF-A"),
                 (pf_b[0], pf_b[1], COLORS["pfam_b"], "PF-B")], width=partner_width, height=0.030)

    right.text(0.295, 0.815, "DUF-containing protein", ha="center", va="center",
               fontsize=7.4, fontweight="bold")
    right.text(0.600, 0.815, "Interaction partner", ha="center", va="center",
               fontsize=7.4, fontweight="bold")

    module = FancyBboxPatch(
        (0.040, 0.43), 0.810, 0.430,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor="none", edgecolor="#A8B0B7", linewidth=0.9, linestyle=(0, (4, 3)),
    )
    right.add_patch(module)
    right.text(0.445, 0.445, "one module", ha="center", va="center",
               fontsize=7.3, color=COLORS["neutral"], fontweight="bold")

    arrow(right, (0.445, 0.41), (0.445, 0.335), color=COLORS["line"])
    right.text(0.445, 0.305, "Representative pair selected by cohort-specific ranking",
               ha="center", va="center", fontsize=8.2, fontweight="bold")
    right.text(0.445, 0.273,
               "Cohorts 1/2: highest combined STRING score   |   Cohort 3: highest fusion score",
               ha="center", va="center", fontsize=7.0, color=COLORS["neutral"])
    right.text(0.445, 0.246,
               "Cohort 4: highest combined STRING score among L2-selected pairs",
               ha="center", va="center", fontsize=7.0, color=COLORS["neutral"])
    protein(right, 0.17, 0.17, [(0.18, 0.64, COLORS["duf"], "DUF-X")], width=0.25, height=0.030)
    protein(right, 0.4625, 0.17,
            [(0.065, 0.40, COLORS["pfam_a"], "PF-A"),
             (0.515, 0.42, COLORS["pfam_b"], "PF-B")], width=0.275, height=0.030)

    fig.subplots_adjust(left=0.025, right=0.985, top=0.97, bottom=0.045, wspace=0.05)
    for ax, label in ((left, "A"), (right, "B")):
        position = ax.get_position()
        letter_y = position.y0 + 0.985 * position.height - 0.004
        ax.set_position([position.x0, position.y0 - 0.003, position.width, position.height])
        fig.text(position.x0, letter_y, label, fontsize=12, fontweight="bold", va="top")
    return fig


def main() -> None:
    configure()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"figure_1_cohort_module_schematic.{extension}",
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
