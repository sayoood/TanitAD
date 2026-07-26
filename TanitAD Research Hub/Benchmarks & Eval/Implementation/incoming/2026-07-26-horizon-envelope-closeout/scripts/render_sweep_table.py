#!/usr/bin/env python3
"""render_sweep_table.py — turn `ksweep_results.json` into the report's §4.2 table.

Mechanical, so the numbers in the prose cannot drift from the numbers in the JSON
(the transcription-drift defect `tools/registry_lint.py` CHECK 1 exists for).
Writes the table in place of the `<!-- KSWEEP_TABLE -->` marker.

600-episode cluster yields are PUBLISHED (`POD2_EVAL_HOST.md` §3.1, artifact
`…/2026-07-26-pod2-eval-host/artifacts/stratum_yield_600{,_fine}.json`) and are
read from that artifact rather than retyped.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
REPORT = HERE.parent / "HORIZON_ENVELOPE.md"
YIELD600 = [
    HERE.parents[1] / "2026-07-26-pod2-eval-host" / "artifacts" / "stratum_yield_600.json",
    HERE.parents[1] / "2026-07-26-pod2-eval-host" / "artifacts" / "stratum_yield_600_fine.json",
]
MARK = "<!-- KSWEEP_TABLE -->"


def load_yield600():
    out = {}
    for p in YIELD600:
        if not p.exists():
            continue
        for k, r in json.loads(p.read_text(encoding="utf-8"))["by_K"].items():
            out[int(k)] = {s: r["strata"][s]["n_episode_clusters"]
                           for s in ("overall", "junction", "longitudinal", "other")}
            out[int(k)]["n_windows"] = r["n_windows"]
    return out


def cls(v):
    """The E1a class name from the verdict string, short."""
    t = (v or "").strip().upper()
    if t.startswith("PARTIAL"):
        return "PARTIAL"
    if t.startswith("EXTRAPOLATION"):
        return "⛔ **EXTRAP**"
    if t.startswith("MEASUREMENT"):
        return "✅ MEASURE"
    return "?"


def main():
    res = json.loads((ART / "ksweep_results.json").read_text(encoding="utf-8"))
    y600 = load_yield600()
    Ks = sorted(int(k) for k, v in res["results"].items() if "_skipped" not in v)

    L = []
    L.append("**Overall stratum** — the co-primary as a gate would read it:\n")
    L.append("| K | s | win / **clusters** (40 eps) | **clusters @600** *(PUBLISHED)* | "
             "`CDR@1.75` [cluster-bootstrap CI95] | peak XTE m | **steps out of envelope** | "
             "**windows out of envelope** | OOD ratio *(ceiling 1.298888)* | c1 | c2 | **verdict** |")
    L.append("|---:|---:|---:|---:|---|---:|---:|---:|---:|:--:|:--:|---|")
    for K in Ks:
        b = res["results"][str(K)]
        o, oo = b["overall"], b["ood"]["overall"]
        c = o["corridor_departure_rate"]
        y = y600.get(K, {}).get("overall")
        L.append(
            f"| **{K}** | {K * 0.1:.1f} | {o['n_windows']} / **{o['n_episodes']}** | "
            f"{y if y is not None else '—'} | "
            f"**{c['mean']:.4f}** [{c['lo']:.4f}, {c['hi']:.4f}] | "
            f"{o['peak_xte_m']['mean']:.3f} | "
            f"{oo['EXTRAPOLATION_frac_steps_any']:.4f} | "
            f"**{oo['EXTRAPOLATION_frac_windows_any_step_out_of_envelope']:.4f}** | "
            f"{oo['ood_peak_ratio']['mean']:.4f} | "
            f"{'🔥' if oo['criterion_1_ratio_over_1p5']['fires'] else '⛔ VOID'} | "
            f"{'🔥' if oo['criterion_2_steps_outside_measured_envelope']['fires'] else '—'} | "
            f"{cls(oo['EXTRAPOLATION_VERDICT'])} |")

    L.append("\n**Junction stratum** — reported separately, always (`GATE_PROTOCOL` §0.4). "
             "⚠️ n is the binding number here, not the value:\n")
    L.append("| K | win / **clusters** (40 eps) | **clusters @600** *(PUBLISHED, HP-2 bar = 200)* | "
             "`CDR@1.75` [CI95] | **windows out of envelope** | **verdict** |")
    L.append("|---:|---:|---:|---|---:|---|")
    for K in Ks:
        b = res["results"][str(K)]
        j, jo = b.get("junction"), b["ood"].get("junction")
        y = y600.get(K, {}).get("junction")
        ybar = ("—" if y is None else
                (f"**{y}** ✅" if y >= 200 else f"⛔ **{y}**"))
        if not j:
            L.append(f"| **{K}** | too small | {ybar} | *NOT MEASURED* | — | — |")
            continue
        c = j["corridor_departure_rate"]
        L.append(
            f"| **{K}** | {j['n_windows']} / **{j['n_episodes']}** | {ybar} | "
            f"**{c['mean']:.4f}** [{c['lo']:.4f}, {c['hi']:.4f}] | "
            f"**{jo['EXTRAPOLATION_frac_windows_any_step_out_of_envelope']:.4f}** | "
            f"{cls(jo['EXTRAPOLATION_VERDICT'])} |")

    L.append("\n**longitudinal / other** — the non-junction strata, for contrast:\n")
    L.append("| K | longitudinal: CDR · winOUT · verdict | other: CDR · winOUT · verdict |")
    L.append("|---:|---|---|")
    for K in Ks:
        b = res["results"][str(K)]
        cells = []
        for s in ("longitudinal", "other"):
            x, xo = b.get(s), b["ood"].get(s)
            if not x:
                cells.append("*NOT MEASURED*")
                continue
            cells.append(
                f"{x['corridor_departure_rate']['mean']:.4f} · "
                f"{xo['EXTRAPOLATION_frac_windows_any_step_out_of_envelope']:.4f} · "
                f"{cls(xo['EXTRAPOLATION_VERDICT'])} *(n {x['n_windows']}/{x['n_episodes']})*")
        L.append(f"| **{K}** | {cells[0]} | {cells[1]} |")

    if "common_start_paired" in res:
        cs = res["common_start_paired"]
        L.append(f"\n**Common-start paired design** — {cs.get('n_common_windows')} IDENTICAL windows "
                 f"over {cs.get('n_common_episodes')} episodes at every K, "
                 f"`paired_episode_cluster_bootstrap` B = 2000, oriented "
                 f"`CDR(K) − CDR(K={cs.get('reference_K')})`:\n")
        key = f"paired_delta_vs_K{cs.get('reference_K')}"
        if key in cs:
            L.append("| K | overall Δ | junction Δ |")
            L.append("|---:|---|---|")
            for K in sorted(int(k) for k in cs[key]):
                d = cs[key][str(K)]
                def fmt(n):
                    x = d.get(n)
                    if not x:
                        return "*n too small*"
                    sep = "✅ **separated**" if x.get("separated") else "not separated"
                    return (f"**{x['delta']:+.4f}** [{x['lo']:+.4f}, {x['hi']:+.4f}] {sep}")
                L.append(f"| **{K}** | {fmt('overall')} | {fmt('junction')} |")

    L.append(f"\n*Rendered from `artifacts/ksweep_results.json` by `scripts/render_sweep_table.py` — "
             f"no number in this table was transcribed by hand. "
             f"`sup(ood_peak_ratio) = "
             f"{res['ratio_supremum_analysis']['sup_ratio']}`, so column `c1` is VOID at every row.*")

    table = "\n".join(L)
    txt = REPORT.read_text(encoding="utf-8")
    assert MARK in txt, "marker missing"
    REPORT.write_text(txt.replace(MARK, table), encoding="utf-8", newline="\n")
    print(table)
    print(f"\n-> wrote table into {REPORT.name}")


if __name__ == "__main__":
    main()
