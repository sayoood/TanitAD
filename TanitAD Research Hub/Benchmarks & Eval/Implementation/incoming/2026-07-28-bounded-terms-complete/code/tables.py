#!/usr/bin/env python3
"""Regenerate T1-T9 from the raw JSON. The tables are GENERATED, not hand-typed.

    python3 code/tables.py raw > artifacts/tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(d, name):
    p = Path(d) / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8"))[name] if p.exists() else None


def row(*c):
    return "| " + " | ".join(str(x) for x in c) + " |"


def sep(n):
    return "|" + "|".join(["---"] * n) + "|"


def f(x, n=4):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def ci(b):
    if not b:
        return "—"
    s = "SEP" if b.get("separated") else "n.s."
    return f"{b['delta']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] {s}"


def main(d):
    out = []
    A = out.append
    A("# Generated tables — bounded-terms completion (2026-07-28)")
    A("")
    A("*Every number here is produced by `code/tables.py` from the JSON in "
      "`raw/`. Nothing is hand-typed.*")

    # ---- T1 density ------------------------------------------------------- #
    den = load(d, "density")
    if den:
        A("\n## T1 — `lat_heading`'s raw quantity `u = |dpsi| / PSI_TOL`, per arm")
        A("")
        A(row("arm", "n def", "u med", "u p90", "u p99", "u max", "frac>1",
              "floor TERM", "floor MEAN"))
        A(sep(9))
        for k in sorted(den["lat_heading_raw_u"]):
            v = den["lat_heading_raw_u"][k]
            if not v.get("u_median"):
                A(row(k, v["n_defined"], "—", "—", "—", "—", "—", "—", "—"))
                continue
            A(row(k, v["n_defined"], f(v["u_median"]), f(v["u_p90"]),
                  f(v["u_p99"]), f(v["u_max"]), f(v["frac_u_gt_1"]),
                  f(v["floor_frac_terminal_resolution"]),
                  f(v["floor_frac_mean_resolution"])))
        p = den["lat_heading_pooled_non_probe"]
        A(row("**POOLED (non-probe)**", "—", f"**{f(p['u_median'])}**",
              f(p["u_p90"]), f(p["u_p99"]), f(p["u_max"]),
              f"**{f(p['frac_u_gt_1'])}**", "—", "—"))

        A("\n## T2 — would WIDENING `PSI_TOL` fix it? Floor fraction of the "
          "published shape")
        A("")
        A(row("arm", "psi_tol 0.1", "0.2 (published)", "0.4", "0.8 (4x)"))
        A(sep(5))
        for k in sorted(den["lat_heading_raw_u"]):
            v = den["lat_heading_raw_u"][k].get("published_floor_frac_at_psi_tol")
            if v:
                A(row(k, f(v["0.1"]), f"**{f(v['0.2'])}**", f(v["0.4"]),
                      f(v["0.8"])))
        A("")
        A("⛔ " + den["_psi_tol_widening_refuted"])

        A("\n## T3 — `ego_progress@twosided_v2`: WHERE the floor comes from")
        A("")
        A(row("arm", "floor total", "from r<=0 (UNDER)", "from r>=2 (OVER)",
              "median r", "max r"))
        A(sep(6))
        for k in sorted(den["ego_progress_floor_decomposition"]):
            v = den["ego_progress_floor_decomposition"][k]
            A(row(k, f(v["floor_frac"]), f(v["floor_from_UNDER_r_le_0"]),
                  f(v["floor_from_OVER_r_ge_2"]), f(v["median_ratio"]),
                  f(v["max_ratio"], 2)))

    # ---- T4 the acceptance test ------------------------------------------- #
    inj = load(d, "injections_lat_heading")
    if inj:
        sel = inj.get("selection", {})
        sat = inj.get("saturation", {}).get("terms", {})
        A("\n## T4 — THE ACCEPTANCE TEST, every candidate term")
        A("")
        A(row("term", "resolution", "shape", "H0 purity", "H1 correct",
              "H2 min live_frac", "reward bias", "SURVIVES"))
        A(sep(8))
        for t in inj["terms"]:
            r = sel.get("per_term", {}).get(t, {})
            A(row(f"`{t}`", r.get("resolution"), r.get("shape"),
                  "✅" if r.get("H0_axis_purity") else "⛔",
                  f"{inj['terms'][t]['n_correct']}/{inj['terms'][t]['n_total']}",
                  f(sat.get(t, {}).get("min_live_frac_over_scorable_arms")),
                  f(r.get("reward_bias"), 3),
                  "⭐ **YES**" if r.get("survives") else "⛔ no"))
        A("")
        A(f"**SELECTED: `{sel.get('SELECTED')}`** — {sel.get('why')}")

        A("\n## T5 — the shipped term, cell by cell")
        A("")
        pick = sel.get("SELECTED")
        pub = "term_lin_q0"
        A(row("arm", "injection", "zero-mean", f"`{pub}` (PUBLISHED)",
              f"`{pick}` (SHIPPED)"))
        A(sep(5))
        for cell in inj["terms"][pub]["cells"]:
            a, i = cell.split("|")
            zm = inj["terms"][pub]["cells"][cell]["zero_mean_control"]
            A(row(a, f"`{i}`", "⭐ **yes**" if zm else "",
                  ci(inj["terms"][pub]["cells"][cell]["lat_heading"]),
                  ci(inj["terms"][pick]["cells"][cell]["lat_heading"])))

        A("\n## T6 — the CONTAMINATION panel (a translation must move it LEAST)")
        A("")
        A(row("arm", "control", f"`{pub}`", f"`{pick}`", "`mean_lin_q0`"))
        A(sep(5))
        for cell in inj["terms"][pub]["contamination_cells"]:
            a, i = cell.split("|")
            if a not in ("cv_holdv0", "v1_tactical_follow"):
                continue
            A(row(a, f"`{i}`",
                  ci(inj["terms"][pub]["contamination_cells"][cell]["lat_heading"]),
                  ci(inj["terms"][pick]["contamination_cells"][cell]["lat_heading"]),
                  ci(inj["terms"]["mean_lin_q0"]["contamination_cells"][cell][
                      "lat_heading"]) if "mean_lin_q0" in inj["terms"] else "—"))

        cr = inj.get("charge_rate", {})
        if cr:
            A("\n## T7 — C47's discriminator on THIS term's density "
              f"(panel median u = {f(cr.get('panel_u_median'))})")
            A("")
            A(row("shape", "|dg/du| @0", "@0.5", "@1", "@2", "@3",
                  "reward bias", "H1 (term / mean / mean1)"))
            A(sep(8))
            for s, v in cr["shapes"].items():
                got = "/".join(
                    str(inj["terms"].get(f"{r}_{s}", {}).get("n_correct", "—"))
                    for r in ("term", "mean", "mean1"))
                A(row(f"`{s}`", f(v["slope_at_u0"]), f(v["slope_at_u0p5"]),
                      f(v["slope_at_u1"]), f(v["slope_at_u2"]),
                      f(v["slope_at_u3"]),
                      f"**{f(v['reward_bias_near_perfect_over_median'], 3)}**",
                      got))
            pred = sel.get("C47_prediction", {})
            A("")
            A(f"Families by ascending reward bias: "
              f"{pred.get('families_by_ascending_reward_bias')} · "
              f"**pass rate monotone in reward bias: "
              f"{pred.get('pass_rate_is_monotone_in_reward_bias')}**")

    # ---- T8 the guard ----------------------------------------------------- #
    g = load(d, "guard")
    if g:
        A("\n## T8 — THE GUARD: what v1 admits and what v2 refuses")
        A("")
        A(row("", "gate v1 (published)", "gate v2 (shipped)"))
        A(sep(3))
        for tag in ("PUBLISHED", "FIXED"):
            p = g["panels"][tag]
            A(row(f"`lat_heading@{p['lat_heading_term']}` ({tag})",
                  "✅ ADMITTED" if p["admitted_v1"]["lat_heading"]
                  else "⛔ refused",
                  "✅ ADMITTED" if p["admitted_v2"]["lat_heading"]
                  else "⛔ **REFUSED**"))
        A("")
        A(row("check", "result"))
        A(sep(2))
        for k in ("OLD_GUARD_ADMITTED_THE_BROKEN_TERM",
                  "GUARD_REFUSES_THE_BROKEN_TERM",
                  "GUARD_ADMITS_THE_FIXED_TERM"):
            A(row(f"`{k}`", "✅ " + str(g[k])))
        A("")
        A(row("arm", "live_frac (published term)", "v1", "v2"))
        A(sep(4))
        lf = g["panels"]["PUBLISHED"]["per_arm_lat_heading_live_frac"]
        for a in sorted(lf):
            A(row(a, f(lf[a]),
                  str(g["panels"]["PUBLISHED"]["per_arm_lat_heading_v1"][a]),
                  str(g["panels"]["PUBLISHED"]["per_arm_lat_heading_v2"][a])))

    # ---- T9 repro + panel ------------------------------------------------- #
    rg = load(d, "repro_gate")
    if rg:
        A("\n## T9 — THE REPRODUCTION GATE")
        A("")
        A(row("check", "value"))
        A(sep(2))
        A(row("published `@clamp_v1` composites checked", rg["n_checked"]))
        A(row("**max |diff|**", f"**{rg['max_abs_diff']:.6f}**"))
        A(row("verdict", "✅ PASS" if rg["PASS"] else "⛔ FAIL"))
        A(row("published metric id, unchanged", f"`{rg['metric_id_must_be_unchanged']}`"))
        A(row("published `lat_heading` BIT-identical (20/20 arms)",
              "✅ " + str(rg["lat_heading_published_is_BIT_IDENTICAL"])))
        A(row("published axis id unchanged",
              "✅ " + str(rg["published_lat_heading_axis_id_unchanged"])))
        A(row("published suite id", f"`{rg['published_suite_id_unchanged']}`"))
        A(row("new suite id", f"`{rg['new_suite_id']}`"))

    pn = load(d, "panel")
    if pn:
        A("\n## T10 — the RANKING STATEMENT and the INSTRUMENT GUARDS")
        A("")
        A(row("statement", "value"))
        A(sep(2))
        for k, v in pn["RANKING_STATEMENT"].items():
            if k.startswith("_") or isinstance(v, list):
                continue
            A(row(f"`{k}`", v))
        A("")
        A(row("guard", "PSS", "CONTROL (published lat_heading)",
              "CONTROL (shipped)"))
        A(sep(4))
        for k, v in pn["INSTRUMENT_GUARDS"].items():
            if k.startswith("_"):
                continue
            A(row(f"`{k}`", ci(v["PSS"]), ci(v["CONTROL_published"]),
                  ci(v["CONTROL_new"])))
        A("")
        A("### CONTROL composite levels")
        A("")
        A(row("arm", "published `lat_heading`", "shipped `lat_heading`"))
        A(sep(3))
        lv = pn["CONTROL_composite"]
        for a in sorted(lv["levels_new_lat_heading"],
                        key=lambda x: -lv["levels_new_lat_heading"][x]):
            A(row(a, f(lv["levels_published_lat_heading"][a]),
                  f(lv["levels_new_lat_heading"][a])))

    # ---- T11 ego_progress ------------------------------------------------- #
    ip = load(d, "injections_ego_progress")
    if ip:
        A("\n## T11 — `ego_progress` over-travel acceptance test")
        A("")
        A(row("progress term", "correct", "over-side floor max (scorable)",
              "failing cells"))
        A(sep(4))
        for t, v in ip["terms"].items():
            bad = [k for k, c in v["cells"].items()
                   if not c["CORRECT_DIRECTION"]]
            A(row(f"`{t}`", f"{v['n_correct']}/{v['n_total']}",
                  f(v["over_side_floor_frac_max_scorable"]),
                  ", ".join(f"`{b}`" for b in bad) or "—"))
        A("")
        A("### the decisive cells, on the already-over-travelling substrate")
        A("")
        A(row("cell", *[f"`{t}`" for t in ip["terms"]]))
        A(sep(1 + len(ip["terms"])))
        for cell in ip["terms"]["twosided_v2"]["cells"]:
            if not cell.startswith("v1_ego_double"):
                continue
            A(row(f"`{cell}`",
                  *[ci(ip["terms"][t]["cells"][cell]) for t in ip["terms"]]))

    # ---- T12 audit -------------------------------------------------------- #
    au = load(d, "audit")
    if au:
        A("\n## T12 — the FULL bounded-term audit, on the two-sided `live_frac`")
        A("")
        A(row("term", "n_sub", "max floor", "max ceil", "min live_frac",
              "gate v1", "gate v2"))
        A(sep(7))
        for k, v in au["terms"].items():
            A(row(f"`{k}`", v["n_sub_per_row"], f(v["max_floor_scorable"]),
                  f(v["max_ceiling_scorable"]), f(v["min_live_frac_scorable"]),
                  "✅" if v["GATE_V1_ADMITS"] else "⛔",
                  "✅" if v["GATE_V2_ADMITS"] else "⛔"))

    cf = load(d, "comfort")
    if cf:
        A("\n## T13 — `comfort`, 100 % saturated by construction")
        A("")
        A(row("arm", "pass rate", "distinct values", "observed_range",
              "live_frac"))
        A(sep(5))
        for a in sorted(cf["per_arm"]):
            v = cf["per_arm"][a]
            A(row(a, f(v["mean_pass_rate"]), v["n_distinct_values"],
                  f(v["observed_range"]), f(v["live_frac"])))
        A("")
        A("**DECISION.** " + cf["DECISION"])

    print("\n".join(out))


if __name__ == "__main__":
    # the tables carry ⛔/⭐/⚠️ markers; a cp1252 console would refuse them.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                        # pragma: no cover
        pass
    main(sys.argv[1] if len(sys.argv) > 1 else "raw")
