#!/usr/bin/env python3
"""Phase analysis for the Thor concurrency pilot: step time BEFORE / DURING / AFTER.

Two things make this non-trivial and both are handled here:

1. ``step_s`` is a CUMULATIVE MEAN (see stepwatch.py). Every number below is the
   first-differenced instantaneous rate ``r_inst``, never the raw field.

2. The trainer log carries **no timestamp**. Wall-clock is recovered exactly:
   ``elapsed_i = step_s_i * N_i`` is seconds since process start, and the file's
   mtime is the wall-clock of the LAST line, so
   ``t_i = mtime_last - (elapsed_last - elapsed_i)``.

A logged point spans the 50 steps BEFORE it, so a point is classified DURING only
if that whole window lies inside the load window. Points whose window straddles a
boundary are labelled TRANSITION and excluded from both groups — counting a
half-loaded window as "during" would dilute exactly the effect being measured.

Usage:
  analyze_phases.py <rows.jsonl> <mtime_epoch> <load_start_epoch> <load_end_epoch>
"""
from __future__ import annotations
import json
import statistics as st
import sys


def _q(v, f):
    s = sorted(v)
    i = f * (len(s) - 1)
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def main() -> int:
    rows = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
    mtime = float(sys.argv[2])
    t0 = float(sys.argv[3])
    t1 = float(sys.argv[4])

    rows.sort(key=lambda r: r["step"])
    # keep only the current process (N strictly increasing)
    cut = 0
    for i in range(1, len(rows)):
        if rows[i]["n"] <= rows[i - 1]["n"]:
            cut = i
    rows = rows[cut:]

    e_last = rows[-1]["step_s"] * rows[-1]["n"]
    for r in rows:
        r["elapsed"] = r["step_s"] * r["n"]
        r["t"] = mtime - (e_last - r["elapsed"])

    pts = []
    for a, b in zip(rows, rows[1:]):
        dn = b["n"] - a["n"]
        if dn <= 0:
            continue
        r = (b["elapsed"] - a["elapsed"]) / dn
        ws, we = a["t"], b["t"]            # the window this rate covers
        if we <= t0:
            ph = "BEFORE"
        elif ws >= t0 and we <= t1:
            ph = "DURING"
        elif ws >= t1:
            ph = "AFTER"
        else:
            ph = "TRANSITION"
        pts.append({"step": b["step"], "r": r, "phase": ph,
                    "gnorm": b["gnorm"], "loss": b["loss"],
                    "w_start": ws, "w_end": we})

    print(f"{'step':>7} {'r_inst':>9} {'phase':>11} {'gnorm':>10} {'loss':>8}")
    for p in pts[-40:]:
        print(f"{p['step']:>7} {p['r']:>9.4f} {p['phase']:>11} "
              f"{p['gnorm']:>10.2f} {p['loss']:>8.4f}")

    print("\n=== PHASE SUMMARY (instantaneous s/step) ===")
    base = None
    for ph in ("BEFORE", "TRANSITION", "DURING", "AFTER"):
        v = [p["r"] for p in pts if p["phase"] == ph]
        if not v:
            print(f"{ph:>11}: n=0")
            continue
        med = st.median(v)
        if ph == "BEFORE":
            base = med
        d = f"   delta_vs_BEFORE={med-base:+.4f} s ({100*(med-base)/base:+.2f} %)" \
            if base and ph != "BEFORE" else ""
        print(f"{ph:>11}: n={len(v):<4} median={med:.4f}  "
              f"IQR=[{_q(v,.25):.4f}, {_q(v,.75):.4f}]  "
              f"min={min(v):.4f}  max={max(v):.4f}{d}")

    b = [p["r"] for p in pts if p["phase"] == "BEFORE"]
    d = [p["r"] for p in pts if p["phase"] == "DURING"]
    if b and d:
        print("\n=== IS THE DIFFERENCE DISTINGUISHABLE FROM NORMAL VARIATION? ===")
        lo, hi = min(b), max(b)
        inside = sum(1 for x in d if lo <= x <= hi)
        print(f"BEFORE range [{lo:.4f}, {hi:.4f}] over n={len(b)}")
        print(f"DURING points inside that range: {inside}/{len(d)}")
        try:
            u = _mwu(b, d)
            print(f"Mann-Whitney U two-sided p = {u:.4f}  "
                  f"(n_before={len(b)}, n_during={len(d)})")
        except Exception as e:
            print(f"(U test unavailable: {e})")
    return 0


def _mwu(a, b):
    """Two-sided Mann-Whitney U with a normal approximation + tie correction.

    Deliberately assumption-light: step times are not normal and n is small, so a
    t-test would be the wrong instrument. Reported as a descriptive aid only —
    with n this size a null result is weak evidence of absence, and the writeup
    says so rather than claiming equivalence.
    """
    import math
    n1, n2 = len(a), len(b)
    allv = sorted([(x, 0) for x in a] + [(x, 1) for x in b])
    ranks = [0.0] * len(allv)
    i = 0
    ties = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        rk = (i + j) / 2.0 + 1.0
        t = j - i + 1
        ties += t ** 3 - t
        for k in range(i, j + 1):
            ranks[k] = rk
        i = j + 1
    r1 = sum(rk for rk, (_, g) in zip(ranks, allv) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    sd = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1))))
    if sd == 0:
        return 1.0
    z = (abs(u - mu) - 0.5) / sd
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2))))


if __name__ == "__main__":
    raise SystemExit(main())
