#!/usr/bin/env python3
"""Regenerate EVERY table in ``CLOSEDLOOP_CONTROL_SUITE.md`` from the raw JSON.

The tables in the report are generated, not hand-typed, so a transcription error
cannot enter a headline. Run from the deliverable directory::

    python3 code/tables.py raw > artifacts/tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Windows consoles default to cp1252; the tables carry unicode markers.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "raw")


def load(n):
    p = RAW / f"{n}.json"
    return json.loads(p.read_text(encoding="utf-8"))[n] if p.exists() else None


def fmt(v, dp=4):
    return "—" if v is None else (f"{v:.{dp}f}" if isinstance(v, float) else str(v))


def t_dynrange(dr):
    print("\n## T1 — DEMONSTRATED DYNAMIC RANGE (zero-bias reference arm "
          "`human_replay`)\n")
    print("| axis | control | zero-mean? | MDE down | MDE up | unit | "
          "both sides | monotone (beyond MDE) |")
    print("|---|---|:--:|--:|--:|---|:--:|:--:|")
    for k, c in dr["arms"]["human_replay"].items():
        ax, ctl = k.split("__")
        if not c.get("own_axis_control"):
            continue
        mono = " / ".join(
            str(c.get(f"monotone_beyond_mde_{t}")) for t in ("down", "up"))
        print(f"| `{ax}` | `{ctl}` | {'⭐ yes' if c['zero_mean_control'] else 'no'}"
              f" | {fmt(c['mde_down'], 3)} | {fmt(c['mde_up'], 3)} | {c['unit']}"
              f" | {c['both_directions_separate']} | {mono} |")

    print("\n## T2 — THE ADMISSION VERDICT\n")
    print("| axis | admissible | reason if refused |")
    print("|---|:--:|---|")
    for ax, v in dr["admission"].items():
        why = (v.get("REFUSED", "") or "").split(": ", 1)
        why = why[1][:220] if len(why) > 1 else ""
        print(f"| `{ax}` | {'✅' if v['admissible'] else '⛔'} | {why} |")

    print("\n## T3 — THE SAME LADDER ON THE REAL ARMS (the confound, measured)\n")
    print("| arm | axis | control | MDE down | MDE up | both | "
          "monotone full ladder (dn/up) |")
    print("|---|---|---|--:|--:|:--:|:--:|")
    for arm in dr["arms"]:
        if arm == "human_replay":
            continue
        for k, c in dr["arms"][arm].items():
            ax, ctl = k.split("__")
            if not c.get("own_axis_control"):
                continue
            print(f"| `{arm}` | `{ax}` | `{ctl}` | {fmt(c['mde_down'], 3)} | "
                  f"{fmt(c['mde_up'], 3)} | {c['both_directions_separate']} | "
                  f"{c['monotone_down']} / {c['monotone_up']} |")


def t_recovery(dr, ro):
    print("\n## T4 — ⛔ THE COMPOSITE REWARDS LATERAL DEGRADATION\n")
    print("| arm | injected lateral degradation | Δ`recovery` | Δ`ego_progress`"
          " | Δ`PSS@twosided_v2` | separated |")
    print("|---|---|---|---|---|:--:|")
    for arm, cells in ro["direction"].items():
        for ctl, blk in cells.items():
            p = blk["PSS_composite"]
            r = blk["recovery"]
            e = blk["ego_progress"]
            print(f"| `{arm}` | `{ctl}` | {r['delta']:+.4f} | {e['delta']:+.4f}"
                  f" | **{p['delta']:+.4f}** [{p['lo']:+.4f}, {p['hi']:+.4f}] |"
                  f" {'⛔ SEP' if p['separated'] else 'n.s.'} |")

    print("\n## T5 — ⛔ `recovery` IS FLOORED, AND THE FLOOR IS WHERE THE ARMS "
          "LIVE\n")
    print("| arm | `recovery` mean | defined | **frac at floor 0** | "
          "unclamped ratio median | p99 | max | frac ratio > 1 |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|")
    for n, v in sorted(ro["floor_saturation"].items(),
                       key=lambda t: -t[1]["frac_at_floor_le_0p001"]):
        print(f"| `{n}` | {v['mean']:.4f} | {v['defined_frac']:.4f} | "
              f"**{v['frac_at_floor_le_0p001']:.4f}** | "
              f"{v['unclamped_ratio_median']:.3f} | {v['unclamped_ratio_p99']:.3f}"
              f" | {v['unclamped_ratio_max']:.1f} | "
              f"{v['frac_ratio_gt_1_i_e_WORSE_THAN_DOING_NOTHING']:.4f} |")


def t_comfort(ca):
    print("\n## T6 — ⛔ THE `comfort` AUDIT: the HUMAN fails the same bounds\n")
    print(f"Limits (PROPOSED): `{ca['limits']}`\n")
    print("| what | all four clauses | a_lon | a_lat | **jerk** | yaw_rate |")
    print("|---|--:|--:|--:|--:|--:|")
    h = ca["human_reference"]
    c = h["pass_frac_per_clause"]
    print(f"| ⭐ **the HUMAN's own logged path** | **{h['comfort_mean_ALL_CLAUSES']:.4f}**"
          f" | {c['a_lon']:.4f} | {c['a_lat']:.4f} | **{c['jerk']:.4f}** | "
          f"{c['yaw_rate']:.4f} |")
    order = sorted(ca["arms"].items(),
                   key=lambda t: -t[1]["comfort_mean_ALL_CLAUSES"])
    for n, v in order:
        c = v["pass_frac_per_clause"]
        print(f"| `{n}` | {v['comfort_mean_ALL_CLAUSES']:.4f} | {c['a_lon']:.4f}"
              f" | {c['a_lat']:.4f} | {c['jerk']:.4f} | {c['yaw_rate']:.4f} |")


def t_cross(cs):
    print("\n## T7 — AXIS PURITY, MEASURED (response to the OTHER axis' control)\n")
    print(cs.get("_why_two_arms", ""), "\n")
    print("| arm | axis form | Δ under `lon_retime(0.5)` | Δ under its own "
          "control | contamination ÷ signal | own control dominates? |")
    print("|---|---|--:|--:|--:|:--:|")
    for _arm in [k for k in cs if not k.startswith("_")]:
        _t_cross_one(_arm, cs[_arm])


def _t_cross_one(arm, node):
    p = node["purity"]
    rows = [
        ("`lat_track` — widening corridor (**SHIPPED**)",
         p["lat_track_widening_corridor"]["d_under_lon_retime_0p5"],
         p["lat_track_widening_corridor"]["d_under_lat_shift_minus1m"],
         p["lat_track_widening_corridor"]["contamination_over_signal"],
         None),
        ("`lat_track` — FLAT tolerance (**REJECTED**)",
         p["lat_track_FLAT_tolerance_the_rejected_form"]["d_under_lon_retime_0p5"],
         p["lat_track_FLAT_tolerance_the_rejected_form"]["d_under_lat_shift_minus1m"],
         p["lat_track_FLAT_tolerance_the_rejected_form"]["contamination_over_signal"],
         None),
        ("`lon_track`",
         p["lon_track"]["d_under_lat_shift_minus1m"],
         p["lon_track"]["d_under_lon_retime_0p5"],
         p["lon_track"]["contamination_over_signal"],
         None),
        ("`lat_heading`",
         p["lat_heading"]["d_under_lon_retime_0p5"],
         p["lat_heading"]["d_under_yaw_bias_5deg"],
         p["lat_heading"]["contamination_over_signal"],
         None),
    ]
    for name, cross, own, ratio, _v in rows:
        verdict = "✅ yes" if ratio < 1.0 else "⛔ NO"
        print(f"| `{arm}` | {name} | {cross:+.4f} | {own:+.4f} | "
              f"**{ratio:.4f}** | {verdict} |")


def t_panel(ap):
    print("\n## T8 — THE ARM PANEL ON THE NEW AXES (panel-wide gate)\n")
    print(f"Gate admitted: `{sorted(ap['gate']['admitted'])}` · dropped: "
          f"`{sorted(ap['gate']['dropped'])}`\n")
    ranks = ap["ranking_per_axis"]
    arms = [r["arm"] for r in ranks["lon_track"]]
    print("| arm | `lon_track` | rk | `lat_track` | rk | `lat_heading` | rk | "
          "`ego_progress` | `recovery` |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    idx = {ax: {r["arm"]: (r["rank"], r["value"]) for r in ranks[ax]}
           for ax in ranks}
    for a in arms:
        cells = []
        for ax in ("lon_track", "lat_track", "lat_heading"):
            rk, v = idx[ax].get(a, (None, None))
            cells += [fmt(v), str(rk)]
        for ax in ("ego_progress", "recovery"):
            cells.append(fmt(idx[ax].get(a, (None, None))[1]))
        print(f"| `{a}` | " + " | ".join(cells) + " |")


def main():
    dr, ro = load("dynamic_range"), load("recovery_onesided")
    ca, cs, ap = load("comfort_audit"), load("cross_sensitivity"), load("axes_panel")
    print("# Generated tables — closed-loop control suite\n")
    print("Regenerate: `python3 code/tables.py raw > artifacts/tables.md`")
    if dr:
        t_dynrange(dr)
    if ro:
        t_recovery(dr, ro)
    if ca:
        t_comfort(ca)
    if cs:
        t_cross(cs)
    if ap:
        t_panel(ap)


if __name__ == "__main__":
    main()
