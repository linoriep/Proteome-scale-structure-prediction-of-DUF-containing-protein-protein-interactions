#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import re
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


PFAM_38_0 = "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam38.0/Pfam-A.clans.tsv.gz"
PFAM_38_2 = "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam38.2/Pfam-A.clans.tsv.gz"
DUF_PATTERN = re.compile(r"DUF[0-9]+")

# Manual interpretation of partner context. The script derives all names,
# annotations, and counts from the deposited data and fixed Pfam releases.
CURATION = {
    "PF01345": ("broader process", "Type IX secretion and cell-envelope partners are consistent with an envelope-associated context but do not define CLIPPER function."),
    "PF01732": ("specific", "Immunoglobulin-binding and immunoglobulin-blocking partners are consistent with a Mycoplasma immunoglobulin protease system."),
    "PF02594": ("no clear relationship", "The observed partners do not indicate a specific relationship to the PF1765-like family assignment."),
    "PF04685": ("specific", "Beta-glucosidase and GH116-containing partners are consistent with the GH116 catalytic-region assignment."),
    "PF06075": ("no clear relationship", "The single partner does not indicate a relationship to the CORD N-terminal assignment."),
    "PF06672": ("broader process", "Alpha-2-macroglobulin and YfaP-associated partners are consistent with a bacterial envelope or protease-defense system but not the specific MagC function."),
    "PF07849": ("broader process", "Formate dehydrogenase and oxidoreductase partners are consistent with a membrane respiratory-complex context without defining the helical membrane plugin role."),
    "PF09836": ("specific", "BufA1- and MNIO-associated partners are consistent with the HvfC/BufC role in bufferin precursor modification."),
    "PF09906": ("broader process", "Alpha-2-macroglobulin and MagC partners are consistent with the same bacterial envelope or protease-defense system without defining YfaP function."),
    "PF09979": ("broader process", "Phage structural and packaging partners indicate a phage-assembly context but do not specifically identify the gp24 scaffolding function."),
    "PF10022": ("no clear relationship", "Predominantly self-annotations and ABC-transporter partners do not indicate beta-D-glucuronate dehydratase function."),
    "PF10048": ("specific", "HvfC/BufC-associated partners are consistent with BufA1 as a bufferin metallophore precursor."),
    "PF10240": ("specific", "VPS37B, TSG101 and VPS28 partners place the domain in the ESCRT-I context associated with MABP/MVB12."),
    "PF10709": ("specific", "The YebZ partner places YebY in the named AZY copper-uptake system."),
    "PF11175": ("broader process", "Glycosyl hydrolase and carbohydrate-binding partners are consistent with carbohydrate processing without identifying the GH172 domain specifically."),
    "PF11816": ("specific", "USP1 and Fanconi-pathway partners are consistent with the WDR48/Bun107 ubiquitin-associated context."),
    "PF11863": ("broader process", "Dit-like and other phage-tail partners indicate a tail-assembly context but do not specifically identify the gp31 N-terminal region."),
    "PF11958": ("broader process", "An outer-membrane starch-binding partner is consistent with a surface-associated context without identifying BPSS1860 function."),
    "PF11984": ("specific", "Exosortase and TPR-rich partners are consistent with an extracellular-polysaccharide system associated with methanolan biosynthesis EpsI."),
    "PF12532": ("no clear relationship", "The diverse ABC and defense-associated partners do not indicate a specific relationship to LmuB."),
    "PF12996": ("broader process", "Cell-wall glycosyltransferase partners are consistent with an envelope-associated process without specifically identifying the spore protein YkvP."),
    "PF13835": ("broader process", "DNA-repair and defense-associated partners indicate an antiphage-defense context but do not specifically identify JetB."),
    "PF14092": ("no clear relationship", "Galactose-oxidase and Kelch-repeat partners suggest a beta-propeller context, but no clear relationship to BamH was identified."),
    "PF14298": ("specific", "TonB-dependent iron-uptake receptors are consistent with the xenosiderophore-binding XusB assignment."),
    "PF14332": ("broader process", "GGDEF and response-regulator partners are consistent with a bacterial signaling context but do not specifically identify the PatA N-terminal domain."),
    "PF16090": ("specific", "The Ataxin-1 partner is consistent with the known protein context of the capicua homolog Cic."),
    "PF16324": ("specific", "SusC(lev), SusD and beta-fructofuranosidase partners are consistent with a levan-utilization system."),
    "PF16332": ("broader process", "A heparinase partner is consistent with polysaccharide processing without establishing dermatan-sulfate epimerase activity."),
    "PF16472": ("broader process", "EGF-like and extracellular partners are consistent with an LRP-like architecture without identifying the beta-propeller's specific function."),
    "PF17270": ("no clear relationship", "No informative partner annotation was available for comparison with the 34 kDa antigenic-protein assignment."),
    "PF17384": ("specific", "Ribosomal-protein partners are consistent with the ribosome-assembly context of RimP."),
    "PF17825": ("no clear relationship", "The only partner repeats the SHOC1 protein assignment and provides no independent functional context."),
    "PF19528": ("specific", "A phage glycosyltransferase partner is consistent with the glycosyltransferase-domain assignment."),
    "PF19557": ("broader process", "DNA methylase and ATPase partners indicate a defense-system context but do not distinguish BrxC from PglY."),
    "PF24346": ("broader process", "Type IX secretion and CLIPPER-containing partners indicate the expected cell-surface context but do not specifically establish CLIPPER 2 function."),
    "PF25372": ("specific", "SKP1 and cyclin-associated partners are consistent with an F-box/LRR protein context."),
    "PF26062": ("specific", "A propionyl-CoA carboxylase carboxyltransferase partner is consistent with the PccX assignment."),
    "PF26607": ("no clear relationship", "The varied phage and defense-associated partners do not establish a relationship to the PLL-like beta-propeller assignment."),
}
def read_pfam(source: str) -> pd.DataFrame:
    path = Path(source)
    compressed = path.read_bytes() if path.exists() else urlopen(source).read()
    columns = ["pfam_id", "clan_id", "clan_name", "name", "description"]
    return pd.read_csv(
        io.BytesIO(gzip.decompress(compressed)), sep="\t", names=columns, dtype=str
    )


def contains_duf(frame: pd.DataFrame) -> pd.Series:
    text = frame["name"].fillna("") + " " + frame["description"].fillna("")
    return text.str.contains(DUF_PATTERN)


def compact_annotation(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).split(" FUNCTION:", 1)[0]
    return re.sub(r"\s+", " ", text).strip()


def unique_values(series: pd.Series, limit: int = 4) -> str:
    values: list[str] = []
    for value in series:
        compact = compact_annotation(value)
        if compact and compact.lower() not in {"deleted", "unknown protein", "hypothetical protein"} and compact not in values:
            values.append(compact)
        if len(values) == limit:
            break
    return " | ".join(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pfam38-0", default=PFAM_38_0)
    parser.add_argument("--pfam38-2", default=PFAM_38_2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--supplementary-output", type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve()
    output = args.output or root / "data/tables/paper_pfam38_2_retrospective_benchmark.tsv"
    supplementary_output = args.supplementary_output or root / "data/tables/paper_supplementary_pfam38_2_comparison.tsv"

    old = read_pfam(args.pfam38_0)
    new = read_pfam(args.pfam38_2)
    old_dufs = old[contains_duf(old)][["pfam_id", "name", "description"]]
    renamed = old_dufs.merge(
        new[~contains_duf(new)][["pfam_id", "name", "description"]],
        on="pfam_id",
        suffixes=("_38_0", "_38_2"),
    )

    predictions = pd.read_csv(
        root / "data/predictions/all_successful_predictions.tsv.gz", sep="\t", low_memory=False
    )
    predictions = predictions[predictions["cohort"].eq("L2 model")]
    overlap = renamed[renamed["pfam_id"].isin(predictions["duf_pfam_id"])].copy()
    observed = set(overlap["pfam_id"])
    if observed != set(CURATION):
        raise ValueError(
            f"Curated accessions do not match the release overlap; missing={sorted(observed - set(CURATION))}, "
            f"extra={sorted(set(CURATION) - observed)}"
        )

    rows = []
    for record in overlap.sort_values("pfam_id").itertuples(index=False):
        family = predictions[predictions["duf_pfam_id"].eq(record.pfam_id)].sort_values(
            ["strict_high_confidence", "liberal_high_confidence", "interface_ipSAE", "model_score"],
            ascending=False,
        )
        agreement, rationale = CURATION[record.pfam_id]
        rows.append(
            {
                "pfam_id": record.pfam_id,
                "pfam38_0_name": record.name_38_0,
                "pfam38_0_description": record.description_38_0,
                "pfam38_2_name": record.name_38_2,
                "pfam38_2_description": record.description_38_2,
                "agreement_class": agreement,
                "evidence_summary": rationale,
                "successful_pairs": len(family),
                "liberal_pairs": int(family["liberal_high_confidence"].sum()),
                "strict_pairs": int(family["strict_high_confidence"].sum()),
                "duf_interacting_pairs": int(family["duf_domain_interacting"].sum()),
                "representative_partner_annotations": unique_values(family["partner_annotation"]),
                "partner_annotation_sources": unique_values(family["partner_annotation_source"]),
                "representative_partner_architectures": unique_values(family["partner_architecture"]),
            }
        )

    result = pd.DataFrame(rows)
    expected = {"specific": 15, "broader process": 15, "no clear relationship": 8}
    observed_counts = result["agreement_class"].value_counts().to_dict()
    if observed_counts != expected:
        raise ValueError(f"Unexpected agreement counts: {observed_counts}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, sep="\t", index=False)
    supplementary = result[
        [
            "pfam_id",
            "pfam38_0_name",
            "pfam38_2_name",
            "pfam38_2_description",
            "agreement_class",
            "evidence_summary",
            "successful_pairs",
            "liberal_pairs",
            "strict_pairs",
        ]
    ].copy()
    supplementary_output.parent.mkdir(parents=True, exist_ok=True)
    supplementary.to_csv(supplementary_output, sep="\t", index=False)
    print(f"Wrote {len(result)} families to {output}")
    print(f"Wrote supplementary table to {supplementary_output}")
    print(", ".join(f"{key}: {value}" for key, value in expected.items()))


if __name__ == "__main__":
    main()
