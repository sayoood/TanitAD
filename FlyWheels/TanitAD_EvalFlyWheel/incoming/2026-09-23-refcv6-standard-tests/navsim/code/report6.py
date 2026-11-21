#!/usr/bin/env python3
"""Render the markdown tables for RESULT.md from the banked summaries (no computation beyond
formatting — every number is read from the JSON it names).

    python code/report6.py --warmup raw/summary_warmup_s1000.json [--navhard …] [--navtest …]
"""
from __future__ import annotations

import argparse
import json
import sys

SUB = ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC")


def _f(x, d=4):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x))


def two_stage(path: str) -> str:
    s = json.load(open(path, encoding="utf-8"))
    out = [f"*{s['_label']} · {s['split']} · `{path}`*", "",
           "| arm | device | S2-EPDMS-u | official two-stage EPDMS | stop frac | " + " | ".join(SUB) + " |",
           "|---|---|---|---|---|" + "---|" * len(SUB)]
    for arm, r in s["arms"].items():
        o = r.get("official_two_stage_EPDMS")
        o = _f(o) if isinstance(o, float) else "UNAVAILABLE (stage-1 stand-in)"
        iv = (s.get("intervals") or {}).get(arm)
        if isinstance(iv, dict) and iv.get("status") == "OK":
            o += f" [{iv['lo']}, {iv['hi']}]"
        sf = (r.get("stop_fraction") or {}).get("frac_endpoint_lt_1m")
        out.append(f"| {arm} | {r.get('device', 'UNSTATED')} | {_f(r['S2_EPDMS_u'].get('value'))} | {o} | {_f(sf, 3)} | "
                   + " | ".join(_f(r['S2_submetric_means'][k], 3) for k in SUB) + " |")
    out += ["", "| pair | ΔS2-EPDMS-u | Δ official | W/T/L | > seed floor | paired interval (logs) |",
            "|---|---|---|---|---|---|"]
    for k, v in s["pairs"].items():
        pi = (s.get("paired_intervals") or {}).get(k, {})
        piv = f"[{pi.get('lo')}, {pi.get('hi')}]" if pi.get("status") == "OK" else (
            pi.get("status") or "—")
        out.append(f"| {k} | {_f(v['S2_EPDMS_u_delta'])} | {_f(v.get('official_two_stage_EPDMS_delta'))} | "
                   f"{v['wins']}/{v['ties']}/{v['losses']} | {v.get('exceeds_seed_floor', '—')} | {piv} |")
    if "seed_floor_S2_EPDMS_u" in s:
        out.append(f"\nseed floor |S2(R6_A1) − S2(R6_A1_s1)| = {s['seed_floor_S2_EPDMS_u']:.4f}")
    return "\n".join(out)


def navtest(path: str) -> str:
    s = json.load(open(path, encoding="utf-8"))
    out = [f"*{s['_label']} · navtest · n = {s['n_tokens']} tokens / {s['n_logs']} logs · `{path}`*",
           "", "| arm | device | PDMS | NC | DAC | EP | TTC | C | zeroed | log-cluster CI |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, r in s["arms"].items():
        iv = r.get("interval", {})
        ci = f"[{iv.get('lo')}, {iv.get('hi')}]" if iv.get("status") == "OK" else iv.get("status")
        out.append(f"| {arm} | {r.get('device', 'UNSTATED')} | {r['PDMS']:.4f} | {r['NC']:.2f} | {r['DAC']:.2f} | {r['EP']:.2f} | "
                   f"{r['TTC']:.2f} | {r['C']:.2f} | {r['zeroed_by_NC_or_DAC']} | {ci} |")
    out += ["", "| pair | Δ ×100 | W/T/L | paired interval |", "|---|---|---|---|"]
    for k, v in s["pairs"].items():
        iv = v.get("interval", {})
        ci = f"[{iv.get('lo')}, {iv.get('hi')}]" if iv.get("status") == "OK" else iv.get("status")
        out.append(f"| {k} | {v['delta_x100']:+.4f} | {v['wins']}/{v['ties']}/{v['losses']} | {ci} |")
    return "\n".join(out)


def plans(path: str) -> str:
    """Label-free plan deltas vs R6_A1 (code/plan_deltas.py)."""
    s = json.load(open(path, encoding="utf-8"))
    out = [f"*plan deltas vs R6_A1 · `{path}` · seed floor (endpoint median) = "
           f"{_f(s.get('seed_floor_endpoint_median_m'), 3)} m*", "",
           "| arm | n | bit-identical plans | same anchor | 4 s endpoint abs Δ median / p90 / max (m) | > seed floor |",
           "|---|---|---|---|---|---|"]
    for k, v in s["vs_R6_A1"].items():
        if not v.get("n"):
            continue
        e = v["endpoint_4s_m"]
        out.append(f"| {k} | {v['n']} | {v['frac_bit_identical']:.3f} | {v['frac_same_selection']:.3f} | "
                   f"{e['median']:.3f} / {e['p90']:.3f} / {e['max']:.2f} | "
                   f"{v.get('endpoint_median_exceeds_seed_floor', '—')} |")
    return "\n".join(out)


def decomp(path: str) -> str:
    """The SPEC §5 ladder, compact: zero attribution (UNIQUE zeros), the single-term ceiling, and the
    per-command split vs STOP (code/decompose6.py)."""
    s = json.load(open(path, encoding="utf-8"))
    out = [f"*SPEC §5 ladder · {s['split']} · {s['arm']} · `{path}`*", ""]
    if "w7_ladder" in s:
        w = s["w7_ladder"]
        for st, za in w.get("zero_attribution", {}).items():
            if s.get("stage1_is_cv_standin") and st == "stage_1":
                continue
            terms = ", ".join(f"{k} {v['n_zeroed_scenes_where_ONLY_this_term_is_0']}/"
                              f"{v['n_zeroed_scenes_with_this_term_0']}" for k, v in za["per_term"].items())
            out.append(f"* {st}: {za['n_zero_score']}/{za['n']} scenes score 0 — unique/co-occurring "
                       f"zeros per multiplier: {terms}")
        for st, lv in w.get("lever_ranking_counterfactual", {}).items():
            if s.get("stage1_is_cv_standin") and st == "stage_1":
                continue
            pt = lv.get("per_term", {})
            if isinstance(pt, dict) and "status" not in pt:
                top = sorted(pt.items(), key=lambda kv: -kv[1]["gain"])[:3]
                out.append(f"* {st} single-term ceilings (scene mean {lv['actual_scene_mean']}): "
                           + "; ".join(f"{k} +{v['gain']:.4f}" for k, v in top))
        pc = s.get("per_command", {})
        for st, blk in pc.items():
            out += ["", f"| {st} command | n | arm | STOP | Δ vs STOP | W/T/L vs STOP | NC0 | DAC0 |",
                    "|---|---|---|---|---|---|---|---|"]
            for c, r in blk.items():
                vs = r["vs"].get("STOP_zero", {})
                if isinstance(vs, dict) and "wtl" in vs:
                    wtl = vs["wtl"]
                    out.append(f"| {c} | {r['n']} | {r['arm']:.4f} | {vs['other']:.4f} | {vs['delta']:+.4f} | "
                               f"{wtl['win']}/{wtl['tie']}/{wtl['loss']} | {r['arm_zero_rate']['NC']:.3f} | "
                               f"{r['arm_zero_rate']['DAC']:.3f} |")
    else:
        lad = s.get("ladder", {})
        if lad:
            za = lad.get("zero_attribution", {})
            out.append(f"* {lad.get('n_zero_score')}/{lad.get('n')} tokens score 0 — unique/co-occurring zeros: "
                       + ", ".join(f"{k} {v['n_zeroed_where_ONLY_this_term_is_0']}/{v['n_zeroed_with_this_term_0']}"
                                   for k, v in za.items()))
            stc = lad.get("single_term_ceiling", {})
            if isinstance(stc, dict) and "status" not in stc:
                out.append("* single-term ceilings (PDMS x100 gain; formula self-check max |d| "
                           f"{lad['formula_selfcheck']['max_abs_diff_vs_devkit_score']:.1e}): "
                           + "; ".join(f"{k} +{v['gain_x100']:.2f}" for k, v in list(stc.items())[:3]))
        for key in ("by_command", "by_speed_band"):
            out += ["", f"| {key} | n | arm PDMS | STOP | Δ vs STOP | CV | HUMAN | NC0 % | DAC0 % |",
                    "|---|---|---|---|---|---|---|---|---|"]
            for g, r in s.get(key, {}).items():
                out.append(f"| {g} | {r['n']} | {r['arm_PDMS_x100']:.2f} | {r['STOP_PDMS_x100']:.2f} | "
                           f"{r['minus_STOP_x100']:+.2f} | {r['CV_PDMS_x100']:.2f} | {r['HUMAN_PDMS_x100']:.2f} | "
                           f"{r['arm_NC0_pct']:.1f} | {r['arm_DAC0_pct']:.1f} |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", default="")
    ap.add_argument("--navhard", default="")
    ap.add_argument("--navtest", default="")
    ap.add_argument("--plans", default="")
    ap.add_argument("--decomp", nargs="*", default=[])
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    for p in (a.warmup, a.navhard):
        if p:
            print(two_stage(p) + "\n")
    if a.navtest:
        print(navtest(a.navtest) + "\n")
    if a.plans:
        print(plans(a.plans) + "\n")
    for d in a.decomp:
        print(decomp(d) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
