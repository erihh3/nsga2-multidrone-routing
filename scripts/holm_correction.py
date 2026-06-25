"""Holm (step-down Bonferroni) multiple-comparison correction — read-only analysis.

PHASE A artifact. Recomputes corrected significance over the EXISTING n=30 per-test
Mann-Whitney p-values already in ``results/<instance>_metrics.json`` and reports the
significance-marker flips under three candidate Holm *families*. It RE-RUNS NOTHING
and WRITES NO results/paper file: it only reads each ``summary[metric]["p"]`` from
disk and emits a comparison (stdout + ``planning/HOLM_CORRECTION_RESULTS.md``). The
family choice and any paper integration are deliberately deferred to Phase B.

Holm needs a defined family (the set of hypotheses corrected together). There is no
canonical family for a *single* pairwise comparison (NSGA-II vs MOPSO) spread over a
metric x problem grid, so this script reports all three defensible candidates:
  - per-instance (m=7):  the 7 metrics within each instance   (matches Table III rows)
  - full-grid    (m=28): all 7 metrics x 4 instances as one family (most conservative)
  - per-metric   (m=4):  the 4 instances within each metric

Usage:  .venv/bin/python scripts/holm_correction.py
"""

from __future__ import annotations

import json
import os

from uav.evaluation.stats import ALPHA, holm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS = os.path.join(ROOT, "results")
PLANNING = os.path.join(ROOT, "planning")

# Table III layout. n_evals is a budget-parity check, not a reported metric -> excluded.
INSTANCES = ("eil51-k3", "berlin52-k3", "eil76-k3", "rat99-k3")
METRICS = ("nps", "spacing", "gd", "igd", "dm", "hv", "ct")
SHORT = {"eil51-k3": "eil51", "berlin52-k3": "berlin52",
         "eil76-k3": "eil76", "rat99-k3": "rat99"}


def load_pvalues() -> dict[str, dict[str, float]]:
    """Read summary[metric]['p'] for the 7 metrics x 4 instances from disk (read-only)."""
    pv: dict[str, dict[str, float]] = {}
    for inst in INSTANCES:
        with open(os.path.join(RESULTS, f"{inst}_metrics.json")) as fh:
            summary = json.load(fh)["summary"]
        pv[inst] = {m: float(summary[m]["p"]) for m in METRICS}
    return pv


def reject_raw(pv) -> dict[tuple[str, str], bool]:
    """Uncorrected significance: p < alpha (the convention currently in the paper)."""
    return {(i, m): pv[i][m] < ALPHA for i in INSTANCES for m in METRICS}


def reject_per_instance(pv) -> dict[tuple[str, str], bool]:
    """Holm within each instance over its 7 metrics (m=7)."""
    out: dict[tuple[str, str], bool] = {}
    for i in INSTANCES:
        rej, _ = holm([pv[i][m] for m in METRICS])
        out.update({(i, m): r for m, r in zip(METRICS, rej)})
    return out


def reject_full_grid(pv) -> dict[tuple[str, str], bool]:
    """Holm over all 28 metric x instance tests as one family (m=28)."""
    keys = [(i, m) for i in INSTANCES for m in METRICS]
    rej, _ = holm([pv[i][m] for (i, m) in keys])
    return dict(zip(keys, rej))


def reject_per_metric(pv) -> dict[tuple[str, str], bool]:
    """Holm within each metric over its 4 instances (m=4)."""
    out: dict[tuple[str, str], bool] = {}
    for m in METRICS:
        rej, _ = holm([pv[i][m] for i in INSTANCES])
        out.update({(i, m): r for i, r in zip(INSTANCES, rej)})
    return out


def grid_lines(title: str, rej: dict[tuple[str, str], bool]) -> list[str]:
    """Render a SIG/ns grid (metrics down rows, instances across columns)."""
    header = f"{'metric':<9}" + "".join(f"{SHORT[i]:>10}" for i in INSTANCES)
    lines = [f"### {title}", "```", header, "-" * len(header)]
    for m in METRICS:
        row = f"{m:<9}" + "".join(
            f"{('SIG' if rej[(i, m)] else 'ns'):>10}" for i in INSTANCES)
        lines.append(row)
    lines.append("```")
    return lines


def flips_vs_raw(raw, fam) -> list[tuple[str, str]]:
    """Cells that were significant uncorrected but are NOT under the family."""
    return [(i, m) for i in INSTANCES for m in METRICS
            if raw[(i, m)] and not fam[(i, m)]]


def main() -> None:
    pv = load_pvalues()
    raw = reject_raw(pv)
    families = {
        "per-instance (m=7)": reject_per_instance(pv),
        "full-grid (m=28)": reject_full_grid(pv),
        "per-metric (m=4)": reject_per_metric(pv),
    }

    out: list[str] = []
    out.append("# Holm correction — Phase A results (read-only, n=30, "
               f"alpha={ALPHA})")
    out.append("")
    out.append("Recomputed from `results/<instance>_metrics.json` "
               "(`summary[metric]['p']`). No results/paper files were modified.")
    out.append("")

    # Raw per-test p-values, for transparency / sanity-check against the prompt preview.
    hdr = f"{'metric':<9}" + "".join(f"{SHORT[i]:>13}" for i in INSTANCES)
    out.append("## Raw per-test Mann-Whitney p-values")
    out.append("```")
    out.append(hdr)
    out.append("-" * len(hdr))
    for m in METRICS:
        out.append(f"{m:<9}" + "".join(f"{pv[i][m]:>13.2e}" for i in INSTANCES))
    out.append("```")
    out.append("")

    out.append("## Significance grids")
    out += grid_lines("Uncorrected (RAW, p<alpha)", raw)
    out.append("")
    for name, fam in families.items():
        out += grid_lines(f"Holm {name}", fam)
        flips = flips_vs_raw(raw, fam)
        if flips:
            flipped = ", ".join(f"{SHORT[i]}/{m}" for i, m in flips)
            out.append(f"*Flips SIG->ns vs RAW ({len(flips)}): {flipped}*")
        else:
            out.append("*No flips vs RAW.*")
        out.append("")

    # Headline guard: GD and IGD must stay significant on all four under every family.
    out.append("## Headline check (GD/IGD significant on all four instances)")
    all_ok = True
    for name, fam in families.items():
        ok = all(fam[(i, mm)] for i in INSTANCES for mm in ("gd", "igd"))
        all_ok &= ok
        out.append(f"- Holm {name}: {'PASS' if ok else 'FAIL'}")
    out.append("")
    out.append(f"**Headline survives under all families: "
               f"{'YES' if all_ok else 'NO — INVESTIGATE'}**")
    out.append("")

    # Flip summary table for the Phase B discussion.
    out.append("## Flip summary")
    out.append("")
    out.append("| family | # flips | markers SIG->ns |")
    out.append("|---|---|---|")
    for name, fam in families.items():
        flips = flips_vs_raw(raw, fam)
        cells = ", ".join(f"{SHORT[i]} {m}" for i, m in flips) or "(none)"
        out.append(f"| {name} | {len(flips)} | {cells} |")
    out.append("")

    text = "\n".join(out)
    print(text)
    report = os.path.join(PLANNING, "HOLM_CORRECTION_RESULTS.md")
    with open(report, "w") as fh:
        fh.write(text + "\n")
    print(f"\nsaved: {os.path.relpath(report, ROOT)}")


if __name__ == "__main__":
    main()
