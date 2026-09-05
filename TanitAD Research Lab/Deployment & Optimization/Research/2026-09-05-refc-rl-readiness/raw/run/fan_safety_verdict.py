#!/usr/bin/env python3
"""Re-derive the PRIMARY endpoint's paired deltas for EVERY arm under ONE rule,
and select the committed exit of SPEC section 10.6.

⛔ WHY THIS EXISTS RATHER THAN A PATCH TO THE DRIVER'S OWN `paired_delta`.
`ctrl0` — whose weights did not change at all (`weights_changed: False`, every R*
readout delta at 1e-5..1e-8, none separated) — reported **40 of 57 fan-safety
metrics as SEPARATED**. Two distinct defects, both caught by the control doing
exactly what a control is for:

  (1) THE DEGENERATE ZERO CI. `sep = (lo > 0) == (hi > 0)`. When every paired
      per-episode delta is EXACTLY 0.0 (which is the normal case for a discrete
      rate under an update that changed nothing), lo = hi = 0.0 and the rule
      evaluates `False == False` -> **True**. "Nothing moved" is reported as
      "separated". This is the CLAUDE.md 2026-08-22 probe trap verbatim: a
      zero-width CI at exactly 0.0000 beating every noisy estimate.

  (2) SEPARATION WITHOUT AN EFFECT SIZE. `mass_rank_ttc_below` read
      delta = -8.2e-09 with CI [-1.9e-08, -4.3e-10] — a correct interval that
      excludes zero, around a change of eight BILLIONTHS of a probability mass.
      Statistically separated; physically nothing. A rate metric needs a minimum
      detectable effect or float noise is promoted to a finding.

⇒ The rule applied here, to every arm identically:

      separated  ==  CI excludes 0
                 AND NOT (every paired delta exactly 0)
                 AND |delta| >= MIN_EFFECT

MIN_EFFECT = 1e-4. Justification, stated rather than tuned: the smallest change
the fan can express is ONE candidate in ONE window. Over N = 128 anchors and the
readout's 120 windows that is 1/(128*120) = 6.5e-5, so 1e-4 is one conservative
step above the quantum of the measurement. Anything smaller cannot correspond to
a pruned candidate.

⛔ Computed from the BANKED per-window rows, never from the driver's own summary,
so all arms are read under one definition regardless of which driver revision
produced them. Estimator: paired EPISODE-cluster bootstrap (the house rule),
never `overlapping_holdout_se`.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

MIN_EFFECT = 1e-4
N_BOOT = 4000
SEED = 11
EXTRAS = ("fan_peak_g_mean", "sel_peak_g", "fan_v_mean_2s_spread")
FLAGS = ("contact", "ttc_below", "ttc_veto", "kamm_over", "envelope", "off_reach",
         "infeasible", "unsafe", "flagged")
LEAD_ONLY = ("contact", "ttc_below", "ttc_veto", "unsafe")
# the endpoint families of SPEC section 10.2, in the order the report prints them
ENDPOINT = {"(a) contact": ("contact",), "(b) ttc": ("ttc_below", "ttc_veto"),
            "(c) infeasible": ("kamm_over", "envelope", "off_reach", "infeasible"),
            "(d) mass": ()}


def paired(before_rows, after_rows, key, lead_only, n_boot=N_BOOT, seed=SEED):
    b = {r["wi"]: r for r in before_rows}
    a = {r["wi"]: r for r in after_rows}
    per_ep: dict[int, list] = {}
    n_win = 0
    for wi in sorted(set(b) & set(a)):
        if lead_only and not b[wi].get("has_lead"):
            continue
        if key not in b[wi] or key not in a[wi]:
            continue
        per_ep.setdefault(b[wi]["eid"], []).append(a[wi][key] - b[wi][key])
        n_win += 1
    if not per_ep:
        return None
    d = np.array([float(np.mean(v)) for v in per_ep.values()])
    all_zero = bool(np.all(d == 0.0))
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)])
    lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
    delta = float(d.mean())
    ci_excludes_zero = bool(lo > 0.0 or hi < 0.0)
    return {"delta": delta, "lo": lo, "hi": hi,
            "n_episodes": int(d.size), "n_windows": int(n_win),
            "all_zero": all_zero,
            "ci_excludes_zero": ci_excludes_zero,
            "abs_ge_min_effect": bool(abs(delta) >= MIN_EFFECT),
            "separated": bool(ci_excludes_zero and not all_zero
                              and abs(delta) >= MIN_EFFECT),
            "population": "lead windows" if lead_only else "all windows",
            "_estimator": "paired episode-cluster bootstrap",
            "_min_effect": MIN_EFFECT}


def arm_record(run_dir, arm):
    p_b = os.path.join(run_dir, arm, "readout_before.json")
    p_a = os.path.join(run_dir, arm, "readout_after.json")
    if not (os.path.exists(p_b) and os.path.exists(p_a)):
        return None
    with open(p_b, encoding="utf-8") as fh:
        before = json.load(fh)
    with open(p_a, encoding="utf-8") as fh:
        after = json.load(fh)
    rows_b, rows_a = before["per_window"], after["per_window"]
    keys = [k for k in rows_b[0]
            if any(k.endswith("_" + f) for f in FLAGS) or k in EXTRAS]
    out = {}
    for k in keys:
        lead_only = any(f in k for f in LEAD_ONLY)
        r = paired(rows_b, rows_a, k, lead_only)
        if r is not None:
            r["before"] = float(np.mean([x[k] for x in rows_b
                                         if (not lead_only) or x.get("has_lead")]))
            r["after"] = float(np.mean([x[k] for x in rows_a
                                        if (not lead_only) or x.get("has_lead")]))
            out[k] = r
    return {"arm": arm, "n_windows": before["n_windows"],
            "n_episodes": before["n_episodes"],
            "n_with_lead": before.get("n_with_lead"),
            "metrics": out}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    run_dir = argv[0] if argv else r"C:\Users\Admin\rl_rescope\run"
    out_path = argv[1] if len(argv) > 1 else os.path.join(run_dir, "fan_safety_verdict.json")
    arms = argv[2].split(",") if len(argv) > 2 else ["ctrl0", "ctrl_const", "rl", "reg_echo"]

    recs = {}
    for arm in arms:
        r = arm_record(run_dir, arm)
        if r is not None:
            recs[arm] = r
            sep = [k for k, v in r["metrics"].items() if v["separated"]]
            allz = sum(1 for v in r["metrics"].values() if v["all_zero"])
            print(f"[{arm}] {len(r['metrics'])} metrics · {allz} identically zero · "
                  f"SEPARATED {len(sep)}: {sep[:8]}")
        else:
            print(f"[{arm}] no readouts yet")

    rec = {"_what": "PRIMARY endpoint (fan safety) re-derived under ONE rule for every arm",
           "_evidence_class": "MEASURED (ours)",
           "_tier": "T0 readout on 120 fixed EVAL windows; the fan the model emits. "
                    "The four families at T1 are the SECONDARY endpoint.",
           "_rule": "separated = CI excludes 0 AND not all-zero AND |delta| >= "
                    f"{MIN_EFFECT} (one conservative step above 1/(128*120) = 6.5e-5, "
                    "the quantum of one candidate in one window)",
           "_n_boot": N_BOOT, "_seed": SEED,
           "arms": recs}

    # --- the validity gates of SPEC section 10.6 -------------------------------
    gates = {}
    if "ctrl0" in recs:
        moved = [k for k, v in recs["ctrl0"]["metrics"].items() if v["separated"]]
        gates["V1_ctrl0"] = {"moved": moved, "PASS": not moved}
    if "ctrl_const" in recs:
        moved = [k for k, v in recs["ctrl_const"]["metrics"].items() if v["separated"]]
        gates["V4_ctrl_const"] = {"moved": moved, "PASS": not moved}
    rec["gates"] = gates
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(f"[verdict] gates: {json.dumps({k: v['PASS'] for k, v in gates.items()})}")
    print(f"[verdict] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
