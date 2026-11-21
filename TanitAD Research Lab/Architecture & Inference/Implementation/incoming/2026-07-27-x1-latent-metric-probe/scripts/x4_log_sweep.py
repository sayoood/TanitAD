#!/usr/bin/env python3
"""X4 — the zero-GPU real-vs-imagined decode sweep across every committed arm.

Pre-registered in
``TanitAD Research Hub/Architecture & Inference/Research/2026-07-27-imagination-perception-manifold/``
(MANIFOLD_MISMATCH_RESEARCH.md §6, row X4):

    "Re-run §2.2's band analysis on ALL committed *_train_log.jsonl (v2, v3enc,
     expA-nodrop, and any v4 log that carries g_* keys) and on the four worktree
     logs under .claude/worktrees/.  0 GPU, ~5 CPU-min.
     REFUTES universality if any arm shows the ratio SHRINKING with training."

WHAT THE TWO QUANTITIES ARE (source-verified, ``stack/tanitad/models/metric_dynamics.py``
``grounding_losses`` + ``stack/tanitad/train/flagship_losses.py``):

  g_{lvl}_mid_de_m   MetricInverseDynamics[lvl] on REAL encoded latent pairs
                     -> relative ego-pose, metre endpoint error.  Weight 2.0.
  g_{lvl}_fwd_ade_m  StepDisplacementReadout[lvl] decoded on the predictor's
                     IMAGINATION rollout, SE(2)-accumulated, metre ADE.  Weight 1.0.

Both are logged from the SAME forward pass on the SAME training batch, so the
pair is a perfectly matched real-vs-imagined decode contrast.

AGGREGATION CAVEAT — carried on every ratio this script prints.
  *_mid_de_m is an ENDPOINT displacement error averaged over that level's
  horizons; *_fwd_ade_m is an ADE over waypoints 1..k.  For a monotonically
  growing error the endpoint statistic is the larger, typically by ~2x.  The
  ratio is therefore a BOUND OF THE RIGHT ORDER, not a clean effect size; the
  generously corrected value (ratio / 2) is printed alongside.  The TREND (how
  the ratio moves over training) is invariant to any fixed aggregation factor.

GATE: the five v1 bands committed in MANIFOLD_MISMATCH_RESEARCH.md §2.2 and the
str-level pair must reproduce to 1e-4 before anything new is quoted.

Usage:  python x4_log_sweep.py --out <artifacts_dir>
No GPU, no pod, stdlib only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                    "..", "..", ".."))

LEVELS = ("op", "tac", "str")

# The five pre-registered bands of MANIFOLD_MISMATCH_RESEARCH.md §2.2, [lo, hi).
BANDS = [("0-1k", 0, 1000), ("4-6k", 4000, 6000), ("9-11k", 9000, 11000),
         ("19-21k", 19000, 21000), ("28-30k", 28000, 30000)]

# ---- the reproduction gate: v1's committed numbers ------------------------ #
GATE = {  # band -> (op_mid, op_fwd, n_rows)
    "0-1k": (2.2486, 0.9541, 20),
    "4-6k": (1.4833, 0.2442, 40),
    "9-11k": (1.2821, 0.1034, 59),
    "19-21k": (1.1004, 0.0516, 40),
    "28-30k": (1.0129, 0.0304, 41),
}
GATE_STR = {"0-1k": (16.0284, 4.5320), "28-30k": (1.8735, 0.1606)}
GATE_NOSPEED = {"19-21k": (1.1333, 0.3543)}  # §2.3 attributing ablation
GATE_TOL = 1e-4


def find_logs():
    """Every committed *train_log*.jsonl in the repo, with an arm label.

    Absence-at-one-location rule: we glob THREE families (the curated
    taniteval/results/trainlogs mirror, the worktree run dirs, and the research
    hub's own raw/ dumps) rather than trusting the first.
    """
    pats = [
        "taniteval/results/trainlogs/*_train_log.jsonl",
        ".claude/worktrees/*/stack/experiments/*/train_log.jsonl",
        ".claude/worktrees/*/taniteval/results/trainlogs/*_train_log.jsonl",
        "TanitAD Research Hub/**/*train_log*.jsonl",
        "stack/experiments/*/train_log.jsonl",
    ]
    seen, out = set(), []
    for p in pats:
        for f in glob.glob(os.path.join(REPO, p), recursive=True):
            f = os.path.abspath(f)
            if f in seen:
                continue
            seen.add(f)
            rel = os.path.relpath(f, REPO).replace("\\", "/")
            parts = rel.split("/")
            if parts[-1].endswith("_train_log.jsonl"):
                arm = parts[-1][: -len("_train_log.jsonl")]
            else:
                arm = parts[-2]
            out.append((arm, rel, f))
    return sorted(out)


def load(fp):
    rows = []
    with open(fp, "r", encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue          # a truncated last line on a live run
    return rows


def band_stat(rows, lo, hi, key):
    v = [r[key] for r in rows
         if isinstance(r.get("step"), (int, float)) and lo <= r["step"] < hi
         and isinstance(r.get(key), (int, float))]
    if not v:
        return None
    return {"mean": statistics.fmean(v), "n": len(v),
            "median": statistics.median(v)}


def analyse_arm(arm, rel, rows):
    steps = [r["step"] for r in rows if isinstance(r.get("step"), (int, float))]
    rec = {"arm": arm, "log": rel, "n_rows": len(rows),
           "step_min": min(steps) if steps else None,
           "step_max": max(steps) if steps else None,
           "has_grounding": False, "levels": {}}
    gkeys = set()
    for r in rows[-5:] or rows:
        gkeys |= {k for k in r if k.startswith("g_")}
    rec["g_keys"] = sorted(gkeys)
    if not any(f"g_{l}_mid_de_m" in gkeys for l in LEVELS):
        return rec
    rec["has_grounding"] = True
    steps_span = (min(steps), max(steps)) if steps else None

    for lvl in LEVELS:
        mk, fk = f"g_{lvl}_mid_de_m", f"g_{lvl}_fwd_ade_m"
        if mk not in gkeys or fk not in gkeys:
            continue
        bands = {}
        for name, lo, hi in BANDS:
            m, f = band_stat(rows, lo, hi, mk), band_stat(rows, lo, hi, fk)
            if m is None or f is None:
                continue
            ratio = m["mean"] / f["mean"] if f["mean"] > 0 else None
            bands[name] = {
                "real_mid_de_m": round(m["mean"], 4),
                "imag_fwd_ade_m": round(f["mean"], 4),
                "ratio_raw": round(ratio, 3) if ratio else None,
                "ratio_agg_corrected_x0.5": round(ratio / 2.0, 3) if ratio else None,
                "n_rows": m["n"],
            }
        # trend = ratio at the LAST available band vs the FIRST available band
        got = [b for b in bands if bands[b]["ratio_raw"] is not None]
        trend = None
        if len(got) >= 2:
            first, last = got[0], got[-1]
            seq = [bands[b]["ratio_raw"] for b in got]
            trend = {
                "first_band": first, "last_band": last,
                "ratio_first": bands[first]["ratio_raw"],
                "ratio_last": bands[last]["ratio_raw"],
                "ratio_growth": round(bands[last]["ratio_raw"]
                                      / bands[first]["ratio_raw"], 3),
                "real_improvement_x": round(bands[first]["real_mid_de_m"]
                                            / bands[last]["real_mid_de_m"], 3),
                "imag_improvement_x": round(bands[first]["imag_fwd_ade_m"]
                                            / bands[last]["imag_fwd_ade_m"], 3),
                "ratio_sequence": seq,
                "monotone_increasing": all(b >= a for a, b in zip(seq, seq[1:])),
                "SHRINKS": bands[last]["ratio_raw"] < bands[first]["ratio_raw"],
            }
        # own-range trend: first vs last fifth of THIS arm's own step range, so a
        # short arm (expA-nodrop stops at 1999) is still evaluable against the
        # pre-registered falsifier instead of silently dropping out of it.
        own = None
        if steps_span:
            lo, hi = steps_span
            w = max(1.0, (hi - lo) / 5.0)
            a_m, a_f = band_stat(rows, lo, lo + w, mk), band_stat(rows, lo, lo + w, fk)
            b_m, b_f = band_stat(rows, hi - w, hi + 1, mk), band_stat(rows, hi - w, hi + 1, fk)
            if a_m and a_f and b_m and b_f and a_f["mean"] > 0 and b_f["mean"] > 0:
                r0, r1 = a_m["mean"] / a_f["mean"], b_m["mean"] / b_f["mean"]
                own = {"window": [lo, round(lo + w), round(hi - w), hi],
                       "ratio_first_fifth": round(r0, 3),
                       "ratio_last_fifth": round(r1, 3),
                       "ratio_growth": round(r1 / r0, 3),
                       "real_first": round(a_m["mean"], 4),
                       "real_last": round(b_m["mean"], 4),
                       "imag_first": round(a_f["mean"], 4),
                       "imag_last": round(b_f["mean"], 4),
                       "SHRINKS": r1 < r0}
        rec["levels"][lvl] = {"bands": bands, "trend": trend, "own_range": own}
    return rec


def run_gate(recs, errors):
    v1 = [r for r in recs if r["arm"] == "v1-speedjerk" and r["has_grounding"]]
    if not v1:
        errors.append("GATE: v1-speedjerk log not found")
        return {"passed": False, "checks": []}
    v1 = v1[0]
    checks, ok = [], True
    for band, (em, ef, en) in GATE.items():
        got = v1["levels"]["op"]["bands"].get(band)
        if got is None:
            checks.append({"band": band, "level": "op", "status": "MISSING"})
            ok = False
            continue
        dm = abs(got["real_mid_de_m"] - em)
        df = abs(got["imag_fwd_ade_m"] - ef)
        st = "PASS" if (dm <= GATE_TOL and df <= GATE_TOL) else "FAIL"
        ok &= st == "PASS"
        checks.append({"band": band, "level": "op", "status": st,
                       "expect_real": em, "got_real": got["real_mid_de_m"],
                       "expect_imag": ef, "got_imag": got["imag_fwd_ade_m"],
                       "expect_n": en, "got_n": got["n_rows"],
                       "n_matches": got["n_rows"] == en})
    for band, (em, ef) in GATE_STR.items():
        got = v1["levels"]["str"]["bands"].get(band)
        st = "PASS" if got and abs(got["real_mid_de_m"] - em) <= GATE_TOL \
            and abs(got["imag_fwd_ade_m"] - ef) <= GATE_TOL else "FAIL"
        ok &= st == "PASS"
        checks.append({"band": band, "level": "str", "status": st,
                       "expect_real": em, "got_real": got and got["real_mid_de_m"],
                       "expect_imag": ef, "got_imag": got and got["imag_fwd_ade_m"]})
    ns = [r for r in recs if r["arm"] == "nospeed-phase0" and r["has_grounding"]]
    if ns:
        for band, (em, ef) in GATE_NOSPEED.items():
            got = ns[0]["levels"]["op"]["bands"].get(band)
            st = "PASS" if got and abs(got["real_mid_de_m"] - em) <= GATE_TOL \
                and abs(got["imag_fwd_ade_m"] - ef) <= GATE_TOL else "FAIL"
            ok &= st == "PASS"
            checks.append({"band": band, "level": "op", "arm": "nospeed-phase0",
                           "status": st, "expect_real": em,
                           "got_real": got and got["real_mid_de_m"],
                           "expect_imag": ef,
                           "got_imag": got and got["imag_fwd_ade_m"]})
    return {"passed": bool(ok), "tol": GATE_TOL, "checks": checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "artifacts"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    logs = find_logs()
    recs, errors = [], []
    for arm, rel, fp in logs:
        rows = load(fp)
        if not rows:
            errors.append(f"EMPTY {rel}")
            continue
        recs.append(analyse_arm(arm, rel, rows))

    gate = run_gate(recs, errors)

    grounded = [r for r in recs if r["has_grounding"]]
    ungrounded = [r for r in recs if not r["has_grounding"]]
    shrink, own_shrink = [], []
    for r in grounded:
        for lvl, d in r["levels"].items():
            t = d.get("trend")
            if t and t["SHRINKS"]:
                shrink.append({"arm": r["arm"], "log": r["log"],
                               "level": lvl, **t})
            o = d.get("own_range")
            if o and o["SHRINKS"]:
                own_shrink.append({"arm": r["arm"], "log": r["log"],
                                   "level": lvl, **o})

    # ---- the attributing ablation, extended to ALL THREE LEVELS ----------- #
    # §2.3 of the source report ran this at `op` only. The two arms differ in
    # exactly three config fields (action_dim 2->3, aux_accel, jerk_weight); the
    # encoder never sees v0 (it is an ACTION into the predictor), so the real-pair
    # path is expected to be unmoved and the imagined path is not.
    attrib = {}
    v1r = next((r for r in grounded if r["arm"] == "v1-speedjerk"), None)
    nsr = next((r for r in grounded if r["arm"] == "nospeed-phase0"), None)
    if v1r and nsr:
        for lvl in LEVELS:
            b1 = v1r["levels"].get(lvl, {}).get("bands", {}).get("19-21k")
            b0 = nsr["levels"].get(lvl, {}).get("bands", {}).get("19-21k")
            if not b1 or not b0:
                continue
            attrib[lvl] = {
                "band": "19-21k (both arms alive)",
                "v1_speed_real": b1["real_mid_de_m"],
                "nospeed_real": b0["real_mid_de_m"],
                "real_delta_pct": round(100 * (b0["real_mid_de_m"]
                                               - b1["real_mid_de_m"])
                                        / b1["real_mid_de_m"], 2),
                "v1_speed_imag": b1["imag_fwd_ade_m"],
                "nospeed_imag": b0["imag_fwd_ade_m"],
                "imag_ratio_x": round(b0["imag_fwd_ade_m"]
                                      / b1["imag_fwd_ade_m"], 3),
                "gap_growth_v1": (v1r["levels"][lvl]["trend"] or {}).get(
                    "ratio_growth"),
                "gap_growth_nospeed": (nsr["levels"][lvl]["trend"] or {}).get(
                    "ratio_growth")}

    out = {
        "experiment": "X4 — real-vs-imagined decode band sweep, every committed arm",
        "attributing_ablation_all_levels": attrib,
        "preregistered_in": ("TanitAD Research Hub/Architecture & Inference/Research/"
                             "2026-07-27-imagination-perception-manifold/"
                             "MANIFOLD_MISMATCH_RESEARCH.md §6 row X4"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": "dev box (CPU only, no GPU, no pod)",
        "python": sys.version.split()[0],
        "aggregation_caveat": (
            "*_mid_de_m is an ENDPOINT error averaged over the level's horizons; "
            "*_fwd_ade_m is an ADE over waypoints 1..k. Ratios are a bound of the "
            "right order (generous correction = ratio/2, also reported). The TREND "
            "is invariant to any fixed aggregation factor."),
        "reproduction_gate": gate,
        "n_logs_scanned": len(logs),
        "n_with_grounding": len(grounded),
        "arms_without_grounding_instrument": [
            {"arm": r["arm"], "log": r["log"], "n_rows": r["n_rows"],
             "step_max": r["step_max"], "keys": sorted(
                 k for k in ("plan_ade", "oracle_ade", "wm", "total")
                 if any(k in x for x in [r.get("g_keys", [])]) ) or None}
            for r in ungrounded],
        "falsifier": {
            "statement": ("REFUTES universality if any arm shows the real/imagined "
                          "ratio SHRINKING with training"),
            "n_arm_level_pairs_shrinking_prereg_bands": len(shrink),
            "shrinking_prereg_bands": shrink,
            "n_arm_level_pairs_shrinking_own_range": len(own_shrink),
            "shrinking_own_range": own_shrink,
            "verdict": ("UNIVERSALITY REFUTED — see shrinking_* for where"
                        if (shrink or own_shrink)
                        else "UNIVERSAL across every instrumented arm"),
        },
        "arms": recs,
    }
    fp = os.path.join(a.out, "x4_log_sweep.json")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    # ---- console summary ---- #
    print(f"GATE: {'PASS' if gate['passed'] else 'FAIL'}")
    for c in gate["checks"]:
        print(f"  [{c['status']}] {c.get('arm','v1-speedjerk')} {c['level']} "
              f"{c['band']}: real {c.get('got_real')} (exp {c.get('expect_real')}) "
              f"imag {c.get('got_imag')} (exp {c.get('expect_imag')})"
              + (f" n={c.get('got_n')}/{c.get('expect_n')}"
                 if "got_n" in c else ""))
    print(f"\nlogs scanned {len(logs)}  with grounding {len(grounded)}  "
          f"without {len(ungrounded)}")
    for r in ungrounded:
        print(f"  NO INSTRUMENT: {r['arm']:<24} steps 0..{r['step_max']}  {r['log']}")
    print("\narm                       lvl  first band -> last band   ratio  growth")
    for r in grounded:
        for lvl, d in r["levels"].items():
            t = d.get("trend")
            if not t:
                o = d.get("own_range")
                if o:
                    print(f"  {r['arm']:<24} {lvl:<4} own-range "
                          f"{o['ratio_first_fifth']:>6.2f}x -> "
                          f"{o['ratio_last_fifth']:>6.2f}x  growth "
                          f"{o['ratio_growth']:>6.2f}x"
                          f"{'   *** SHRINKS ***' if o['SHRINKS'] else ''}")
                continue
            print(f"  {r['arm']:<24} {lvl:<4} {t['first_band']:>7} "
                  f"{t['ratio_first']:>7.2f}x -> {t['last_band']:>7} "
                  f"{t['ratio_last']:>7.2f}x   growth {t['ratio_growth']:>6.2f}x"
                  f"{'   *** SHRINKS ***' if t['SHRINKS'] else ''}"
                  f"   seq {[round(x,2) for x in t['ratio_sequence']]}")
    print(f"\nFALSIFIER: {out['falsifier']['verdict']}")
    if errors:
        print("errors:", errors)
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
