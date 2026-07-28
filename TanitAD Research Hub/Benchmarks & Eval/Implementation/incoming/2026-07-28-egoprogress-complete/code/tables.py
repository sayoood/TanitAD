#!/usr/bin/env python3
"""Regenerate T1-T12 from ``raw/*.json``. The report's tables are GENERATED.

    python3 code/tables.py raw > artifacts/tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ⛔ Windows' console default is cp1252 and this report is full of ≤ / ⭐ / ⛔.
# Without this the tables die on the FIRST unicode header, half-written.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROBES =("stand_still", "v1_ego_half", "v1_ego_double", "oracle_lon_straight")


def load(d, name):
    p = Path(d) / f"{name}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))[name]


def ci(b, dp=4):
    if b is None:
        return "—"
    s = "SEP" if b["separated"] else "n.s."
    d = b.get("display_dp", dp)
    return f"{b['delta']:+.{d}f} [{b['lo']:+.{d}f}, {b['hi']:+.{d}f}] {s}"


def esc(x):
    """⛔ Term ids and cell tags CONTAIN `|` (`mean|hyp_w1`,
    `v4_blind|lon_shift(-2)`). Unescaped they silently split the markdown row
    and the table renders one column short with the numbers shifted — a
    rendering bug that looks like a data bug."""
    return str(x).replace("|", "\\|")


def emit(title, header, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(esc(h) for h in header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(esc(x) for x in r) + " |")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "raw"
    print("# Generated tables — `ego_progress`, both sides\n")
    print("*Regenerated from `raw/*.json` by `code/tables.py`. "
          "Nothing here is hand-typed.*")

    # ---------------- T1: the under-side density ---------------------------- #
    ud = load(d, "under_density")
    if ud:
        rows = []
        for n, v in sorted(ud["per_arm"].items(),
                           key=lambda kv: -kv[1]["frac_r_le_0"]):
            rows.append([("⛔ " if n == "v4_blind" else "") + n,
                         f"{v['frac_r_le_0']:.4f}",
                         f"{v['frac_r_lt_0_STRICT']:.4f}",
                         f"{v['frac_r_eq_0_EXACT']:.4f}",
                         f"{v['min_r']:.3f}",
                         f"{v.get('under_median_r', float('nan')):.3f}",
                         f"{v.get('under_frac_strictly_below_m0p10', float('nan')):.3f}",
                         f"{v['term_floor']:.4f}", f"{v['term_live_frac']:.4f}",
                         f"{v['mean_floor']:.4f}", f"{v['mean_live_frac']:.4f}"])
        emit("T1 — the UNDER side's density, per arm "
             "(`term` = published n_sub 1, `mean` = 20-step)",
             ["arm", "r≤0", "r<0 strict", "r=0 exact", "min r",
              "median r | r≤0", "frac < −0.10",
              "term floor", "term live", "mean floor", "mean live"], rows)

        demo = ud["_identical_score_demonstration"]
        f = demo["_fixed_row_set"]
        keys = [k for k in demo if k.startswith("lon_shift")]
        terms = sorted({k.split("@", 1)[1] for k in demo[keys[0]]
                        if k.startswith("mean_score_ON_THE_FIXED_ROWS@")})
        rows = []
        for k in keys:
            v = demo[k]
            rows.append([k,
                         f"{v['mean_ground_truth_along_err_m_ON_THE_FIXED_ROWS']:.3f}",
                         f"{v['mean_ratio_r_ON_THE_FIXED_ROWS']:.3f}"]
                        + [f"{v[f'mean_score_ON_THE_FIXED_ROWS@{t}']:.6f}"
                           for t in ("clamp_v1", "twosided_v2",
                                     "sqrtlin_w0p3333", "hyp_w1")])
        emit(f"T2 — ⛔ THE SAME {f['n_rows']} ROWS, PUSHED FURTHER AND FURTHER "
             f"BACKWARDS ({f['definition']})",
             ["injection", "ground-truth |s_plan−s_human| (m)", "mean ratio r",
              "clamp_v1", "twosided_v2", "sqrtlin_w0p3333", "hyp_w1"], rows)

    # ---------------- T3: control direction verification --------------------- #
    iu = load(d, "inject_under")
    if iu:
        rows = []
        for k, v in iu["direction_check"].items():
            rows.append([k, ci(v), "✅" if v["DEGRADES_GROUND_TRUTH"]
                         else "⛔ **REFUSED**"])
        emit("T3 — ⭐ EVERY CONTROL'S DIRECTION, VERIFIED AGAINST A "
             "METRIC-INDEPENDENT GROUND TRUTH (`|s_plan − s_human|`, metres)",
             ["cell", "Δ ground-truth error (m)", "admissible as a degradation?"],
             rows)

        rows = []
        for term, node in iu["terms"].items():
            rows.append([("⭐ " if term == "twosided_v2" else "") + term,
                         f"{node['n_correct']}/{node['n_verified_cells']}",
                         node["n_separated_WRONG_WAY"],
                         "✅" if node["ALL_CORRECT"] else "⛔"])
        emit("T4 — the UNDER-SIDE acceptance test, every term "
             "(verified cells only)",
             ["term", "correct", "separated WRONG WAY", "all correct"], rows)

        rows = []
        for term in ("clamp_v1", "twosided_v2", "mean|clamp_v1",
                     "mean|twosided_v2", "sqrtlin_w0p3333"):
            node = iu["terms"].get(term)
            if not node:
                continue
            r = [term]
            for cell in ("v4_blind|lon_shift(-2)", "v4_blind|lon_shift(-5)",
                         "v4_blind|lon_jitter(+2)"):
                r.append(ci(node["cells"][cell]["ego_progress"]))
            rows.append(r)
        emit("T5 — ⭐ THE DECISIVE SUBSTRATE `v4_blind` (31.78 % of its rows at "
             "r ≤ 0), cell by cell",
             ["term", "lon_shift(−2 m)", "lon_shift(−5 m)",
              "⭐ lon_jitter(σ=2 m) ZERO-MEAN"], rows)

        st = iu["reversing_stress_substrate"]
        rows = [["base (no injection)",
                 f"{st['base_mean_ground_truth_along_err_m']:.3f}",
                 f"{st['base_levels']['clamp_v1']:.6f}",
                 f"{st['base_levels']['twosided_v2']:.6f}", "—"]]
        for k, v in st["injected_levels"].items():
            rows.append([k, f"{v['mean_ground_truth_along_err_m']:.3f}",
                         f"{v['mean_score@clamp_v1']:.6f}",
                         f"{v['mean_score@twosided_v2']:.6f}",
                         ci(st["cells"][f"twosided_v2 @ {k}"], 6)])
        emit("T6 — ⛔⛔ THE REVERSING SUBSTRATE "
             f"({st['frac_r_le_0']:.4f} of rows at r ≤ 0, median r "
             f"{st['median_r']:.4f}) — SYNTHETIC, never votes",
             ["injection", "ground-truth error (m)", "mean clamp_v1",
              "mean twosided_v2", "paired Δ @twosided_v2"], rows)

    # ---------------- T7: the under-side fix, priced ------------------------- #
    uf = load(d, "under_fix")
    if uf:
        per = uf["mean_ego_progress_per_arm_at_each_p"]
        ps = list(per)
        rows = []
        for arm in ("stand_still", "v4_blind", "cv_holdv0",
                    "v1_tactical_follow", "v1_ego_half"):
            rows.append([("⛔ " if arm == "stand_still" else "") + arm]
                        + [f"{per[p][arm]:.6f}" for p in ps])
        emit("T7 — ⛔ THE UNDER-SIDE RANGE BUDGET, PRICED "
             "(`p` = the score a plan making zero progress receives)",
             ["arm"] + ps, rows)

    # ---------------- T8: the over side -------------------------------------- #
    io = load(d, "inject_over")
    if io:
        rows = []
        for k, v in io["direction_check"].items():
            if "jitter" not in k:
                continue
            rows.append([k, ci(v), "✅" if v["DEGRADES_GROUND_TRUTH"]
                         else "⛔ **REFUSED**"])
        emit("T8 — ⛔⛔ THE OVER-SIDE ZERO-MEAN CELLS, DIRECTION-VERIFIED "
             "(this is what E-2's failing cell rests on)",
             ["cell", "Δ ground-truth error (m)", "admissible?"], rows)

        rows = []
        for term, node in io["terms"].items():
            c = node["cells"]["v1_ego_double|lon_jitter(+2)"]
            rows.append([("⭐ " if term == "twosided_v2" else "") + term,
                         f"{node['n_correct']}/{node['n_total']}",
                         node["n_separated_WRONG_WAY"],
                         f"{node['over_side_floor_frac_max_scorable']:.4f}",
                         ci(c)])
        emit("T9 — the OVER-SIDE grid and ⛔ THE FAILING CELL "
             "(`v1_ego_double × lon_jitter(σ=2 m)`, zero-mean)",
             ["term", "correct", "WRONG WAY", "over-side floor max (scorable)",
              "the failing cell"], rows)

    # ---------------- T10: contamination ------------------------------------- #
    ct = load(d, "contamination")
    if ct:
        rows = []
        for term, node in ct["terms"].items():
            r = [term]
            for arm in ("cv_holdv0", "v1_tactical_follow"):
                p = node["purity"][arm]
                r.append(f"{p['max_abs_delta_pure_lateral']:.4f} vs "
                         f"{p['min_abs_delta_longitudinal']:.4f} "
                         f"({p['margin_x']:.1f}×)")
            r.append("✅" if node["PURE_BOTH_ARMS"] else "⛔ **FAILS**")
            rows.append(r)
        emit("T10 — the CONTAMINATION panel: max|Δ| pure-lateral vs min|Δ| "
             "longitudinal (lateral must move this axis LEAST)",
             ["term", "cv_holdv0", "v1_tactical_follow", "pure?"], rows)

    # ---------------- T11: repro gate ---------------------------------------- #
    rg = load(d, "repro_gate")
    if rg:
        rows = [["published `@clamp_v1` composites checked",
                 rg["n_published_composites_checked"]],
                ["**max \\|diff\\|**", f"**{rg['max_abs_diff']:.6f}**"],
                ["verdict", "✅ **PASS**" if rg["PASS"] else "⛔ FAIL"],
                ["published metric id", f"`{rg['_published_metric_id']}`"],
                ["default metric id", f"`{rg['_default_metric_id']}`"],
                ["every new term bit-identical to `clamp_v1` for r ≤ 1",
                 "✅ " + str(all(v["bit_identical_to_clamp_v1_on_every_r_le_1_row"]
                                 for v in rg["strict_refinement_bit_identity"]
                                 .values()))]]
        emit("T11 — ✅ THE REPRODUCTION GATE", ["check", "result"], rows)

    # ---------------- T12: the panel ----------------------------------------- #
    pn = load(d, "panel")
    if pn:
        rows = []
        for term, node in pn["terms"].items():
            g1 = node["guards"]["v4_oracle - v4_blind"]
            g2 = node["guards"]["v1_ego_half - v1_tactical_follow"]
            rows.append([term,
                         node["cv_holdv0_rank_among_realisable"],
                         "✅" if node["v1_tactical_family_below_every_REFC"]
                         else "⛔",
                         f"{node['ego_progress_min_live_frac_scorable']:.4f}",
                         "✅" if node["GATE_V2_ADMITS_ego_progress"] else "⛔",
                         ci(g1), ci(g2)])
        emit("T12 — the RANKING STATEMENT and the two INSTRUMENT GUARDS",
             ["term", "cv_holdv0 rank (realisable)", "v1 tactical < every REF-C",
              "ego_progress min live_frac", "gate v2",
              "guard `v4_oracle − v4_blind`",
              "guard `v1_ego_half − v1_tactical_follow`"], rows)
        rows = [[t, n] for t, n in pn["n_flipped"].items()]
        emit(f"T13 — paired contrasts that FLIP sign against "
             f"`{pn['_reference_term_for_flips']}` (out of "
             f"{len(pn['paired_contrasts'])})", ["term", "flipped"], rows)

    # ---------------- T14: lat_heading weight -------------------------------- #
    lw = load(d, "lat_heading_weight")
    if lw:
        rows = []
        for t, v in lw["terms"].items():
            rows.append([t, f"{v['pooled_sd_within']:.4f}",
                         f"{v['pooled_p05_p95_span']:.4f}",
                         f"{v['BETWEEN_ARM_sd_of_means']:.4f}",
                         f"{v['BETWEEN_ARM_span_of_means']:.4f}"])
        rows.append(["**ratio shipped/published (between-arm span)**", "", "",
                     "", f"**{lw['shipped_over_published_BETWEEN_ARM_span']:.4f}**"])
        rows.append(["**weight preserving the published influence**", "", "",
                     "",
                     f"**{lw['weight_that_PRESERVES_the_published_influence']}**"
                     " (today: 1.0)"])
        emit("T14 — ⚠️ `lat_heading`'s CONTROL_WEIGHTS entry, re-derived",
             ["term", "pooled sd", "p05–p95 span", "between-arm sd",
              "between-arm span"], rows)


if __name__ == "__main__":
    main()
