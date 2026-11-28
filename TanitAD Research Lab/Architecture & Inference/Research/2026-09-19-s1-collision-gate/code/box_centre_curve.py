"""How far is A8's box head from the 2 m bar, and what may be said about it? (0 GPU, banked rows.)

⛔ CLAUDE.md's fit discipline, applied literally: every exponent carries its FIT WINDOW, R² and n;
below R² 0.80 there is no quotable exponent at all (use the matched-step ratio instead); and
NOTHING is extrapolated more than 2x beyond the fitted range. If the 2x bound cannot reach 2 m,
the honest sentence is "the dev box cannot answer this, and here is the arithmetic".

Also answers two one-liners the reading needs:
 * does the stable matched count (36-39) mean the head matches the SAME agents? -> compare
   `box3d_n_matched` with `box3d_n_target`: `agent_slots.match_slots` matches min(n_target, n_query)
   pairs, so with 100 queries and <= 32 padded targets EVERY valid target is matched BY
   CONSTRUCTION, and the count measures LABEL DENSITY, not head quality;
 * is the centre error dominated by x or by y? -> `box3d_centre` is a SUMMED L1 over both axes
   (agent_slots.py:577) and the trainer logs no split, so this file states that it CANNOT answer
   it from banked rows, and the held-out probe answers it instead.

Usage: python box_centre_curve.py <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

A8 = r"C:\Users\Admin\tanitad-caches\a8-occupancy-5k-20260919\run\metrics.jsonl"
BAR_M = 2.0            # the matching threshold the read uses
R2_QUOTABLE = 0.80     # CLAUDE.md: below this there is no quotable exponent


def fit(steps, vals):
    """OLS of log(value) on log(step) -> (exponent, intercept, R², n, window)."""
    x, y = np.log(np.asarray(steps, float)), np.log(np.asarray(vals, float))
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"exponent": float(b), "intercept": float(a),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "n": int(len(x)), "window": [int(min(steps)), int(max(steps))]}


def step_for(f, target):
    return float(np.exp((np.log(target) - f["intercept"]) / f["exponent"]))


def main(argv=None) -> int:
    out_path = (sys.argv[1:] if argv is None else argv)[0]
    rows = [json.loads(l) for l in open(A8, encoding="utf-8") if l.strip()]
    tr = [r for r in rows if "box3d_centre" in r and "eval_traj" not in r and r.get("step")]
    steps = np.array([r["step"] for r in tr], float)
    centre = np.array([r["box3d_centre"] for r in tr], float)
    ok = centre > 0
    steps, centre = steps[ok], centre[ok]
    full = fit(steps, centre)
    half = fit(steps[steps >= 2500], centre[steps >= 2500])
    last = float(np.mean(centre[steps >= 4500]))
    rec = {"_what": "box3d_centre (summed L1 in metres over x,y for matched pairs) vs step",
           "_evidence_class": "MEASURED (ours; A8's own banked metrics.jsonl)", "_tier": "train-side",
           "bar_m": BAR_M, "n_rows": int(len(steps)),
           "last_500_steps_mean_centre_m": round(last, 3),
           "distance_to_bar_x": round(last / BAR_M, 2),
           "fit_full": full, "fit_last_half": half,
           "extrapolation_bound_steps": int(2 * max(steps)),
           "quotable": {}}
    for name, f in (("fit_full", full), ("fit_last_half", half)):
        q = {"r2": round(f["r2"], 4), "window": f["window"], "n": f["n"],
             "quotable_rate": bool(f["r2"] >= R2_QUOTABLE)}
        if f["r2"] >= R2_QUOTABLE:
            s = step_for(f, BAR_M)
            q["steps_to_2m_by_this_fit"] = round(s, 1)
            q["within_2x_bound"] = bool(s <= 2 * max(steps))
            q["reading"] = ("REACHES the bar inside the 2x bound" if s <= 2 * max(steps)
                            else "the bar is BEYOND the 2x extrapolation bound: the dev box "
                                 "cannot answer this, and this number is NOT quotable as a "
                                 "step count")
        else:
            q["reading"] = ("R2 below 0.80: NO quotable exponent. Use the matched-step ratio "
                            "instead, and do not convert this into a step count.")
        rec["quotable"][name] = q
    # ⭐ the prescribed fallback when no exponent is quotable: the MATCHED-STEP RATIO.
    # Compare equal-length step windows and ask what the observed decay rate would need.
    def blk(a, b):
        m = (steps >= a) & (steps < b)
        return float(np.mean(centre[m])) if m.any() else float("nan")
    w1, w2 = blk(1000, 3000), blk(3000, 5001)
    ratio = w2 / w1 if w1 > 0 else float("nan")
    need = np.log(BAR_M / w2) / np.log(ratio) if 0 < ratio < 1 and w2 > BAR_M else float("nan")
    rec["matched_step_ratio"] = {
        "window_a": [1000, 3000], "mean_a_m": round(w1, 3),
        "window_b": [3000, 5000], "mean_b_m": round(w2, 3),
        "ratio_per_2000_steps": round(ratio, 4),
        "implied_extra_2000_step_blocks_to_2m": (None if not np.isfinite(need)
                                                 else round(float(need), 2)),
        "implied_total_steps": (None if not np.isfinite(need)
                                else int(5000 + round(float(need) * 2000))),
        "within_2x_bound": (None if not np.isfinite(need)
                            else bool(5000 + need * 2000 <= 2 * max(steps))),
        "reading": ("⛔ A DIRECTION, NOT A RATE. With no quotable exponent (R2 0.41 / 0.09) this "
                    "ratio may say 'still falling' and roughly how fast between these two "
                    "windows; it may NOT be turned into a promised step count. Where the implied "
                    "total exceeds the 2x extrapolation bound, the honest sentence is that THIS "
                    "RUN CANNOT ANSWER IT.")}
    # the matched-count question, from the rows themselves
    mt = np.array([r.get("box3d_n_matched", np.nan) for r in tr], float)
    tg = np.array([r.get("box3d_n_target", np.nan) for r in tr], float)
    dr = np.array([r.get("box3d_n_dropped", np.nan) for r in tr], float)
    rec["matched_count"] = {
        "n_matched_mean": round(float(np.nanmean(mt)), 3),
        "n_target_mean": round(float(np.nanmean(tg)), 3),
        "n_dropped_mean": round(float(np.nanmean(dr)), 3),
        "matched_equals_target_share": round(float(np.nanmean(mt == tg)), 4),
        "reading": ("match_slots matches min(n_target, n_query) pairs; with 100 queries and at "
                    "most --agent-pad 32 targets per window EVERY valid target is matched BY "
                    "CONSTRUCTION. ⇒ a stable matched count measures LABEL DENSITY, not that the "
                    "head keeps matching the SAME agents, and it is not evidence about head "
                    "quality in either direction.")}
    rec["axis_split"] = {
        "available_from_banked_rows": False,
        "reading": ("box3d_centre is a SUMMED L1 over x and y (agent_slots.py:577) and the "
                    "trainer logs no per-axis split, so this question CANNOT be answered from "
                    "banked rows. The held-out probe answers it on predictions instead.")}
    Path(out_path).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("last-500 mean centre %.3f m = %.2fx the %.1f m bar | n rows %d"
          % (last, last / BAR_M, BAR_M, len(steps)))
    for k, v in rec["quotable"].items():
        print("%-14s R2 %.4f n=%d window %s -> %s" % (k, v["r2"], v["n"], v["window"],
                                                      v.get("steps_to_2m_by_this_fit", v["reading"][:60])))
    print("matched:", rec["matched_count"]["n_matched_mean"], "target:",
          rec["matched_count"]["n_target_mean"], "equal share:",
          rec["matched_count"]["matched_equals_target_share"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
