"""E-O2-A ON THE LIVE LOG — O2's unique content as a function of training step.

⛔ ZERO GPU. This is arithmetic on a training log that already exists.

THE IDENTITY, RE-ESTABLISHED FROM SOURCE (not inherited):

  ``o2_near_field_loss`` (stack/scripts/train_v6_staged.py:784-818)
      err  = (pred_cells - true_cells).abs().mean(dim=-1)      # [B, C]
      loss = (w * err).mean()                                  # -> o2_loss
      log    "o2_unweighted": err.mean()                       # -> O5's step-j term
  ``time_to_reach_weights`` (stack/tanitad/models/v6.py:325-345) with
  ``normalize=True`` divides by ``w.mean(dim=-1)``, i.e. **w is MEAN-1 over the
  CELL axis, per batch element**.

  => o2_loss - o2_unweighted
       = E_{b,c}[(w-1) * err]
       = E_b[ (1/C) SUM_c (w_bc - 1) err_bc ]
       = E_b[ Cov_c(w_b, err_b) ]                              EXACT, no residual.

  Both means are over the SAME axes (B and C), so the difference is exactly the
  per-batch-element cross-cell covariance, averaged over the batch. Nothing is
  approximated.

WHY THE RE-TAKE. The banked measurement (2026-08-17-O234-DESIGN-RESEARCH.md
2.1a, n=7, |Cov|/unwt 0.45-3.33 %, median 1.81 %, signs 4-/3+) came from
DRY-LADDER rows at steps 1-2, i.e. AT INITIALISATION, where the per-cell error
profile is near-uniform and Cov ~ 0 almost by construction. Its own caveat said
so and named this measurement as the settling one.

⚠️ TWO ROW SCHEMAS IN THIS LOG. Training rows carry loss/gnorm/o2_*/o5_*;
SPECTRUM rows are {"step": N, "spectrum": {...}} and have NO loss field. A
parser using .get("loss", 0) turns those into zeros and they read as a training
collapse. This script filters BY SCHEMA (presence of both o2 fields), never by
position, and prints the schema census so the filtering is auditable.

⚠️ ESTIMATOR NOTE, STATED SO IT IS NOT ASKED FOR. No episode-cluster bootstrap
is quoted here and that is deliberate: this is a SERIALLY-CORRELATED TIME SERIES
of training-batch statistics, not a per-window eval metric over episodes. The
paired episode-cluster bootstrap is the decision-grade estimator for eval
numbers over the 40 val episodes; applying it to training log rows would invent
an interval for a quantity that has no episode clustering. What IS reported is
the trajectory, the quantiles, the sign census and an OLS fit carrying its
window, R^2 and n (CLAUDE.md: never quote an exponent bare).

Usage:  python o2_live_trend.py <train_log.jsonl> [--out <json>]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys


def load(path: pathlib.Path):
    """Return (train_rows, schema_census). Filters BY SCHEMA, never by position."""
    train, census, bad = [], collections.Counter(), 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        census[_schema(d)] += 1
        # the schema test: BOTH o2 fields present. A spectrum row has neither.
        if "o2_loss" in d and "o2_unweighted" in d:
            train.append(d)
    return train, census, bad


def _schema(d: dict) -> str:
    if "o2_unweighted" in d:
        return "TRAIN"
    if "spectrum" in d:
        return "SPECTRUM (no loss field — a .get('loss', 0) parser reads these as 0)"
    if "run_start" in d:
        return "RUN_START"
    return "OTHER:" + ",".join(sorted(d)[:4])


def ols(xs, ys):
    """Plain least squares. Returns (slope, intercept, r2, n)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0, my, 0.0, n
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return b, a, r2, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--bin", type=int, default=1000, help="step bin width")
    a = ap.parse_args(argv)

    rows, census, bad = load(a.log)
    print("SCHEMA CENSUS (filtering is by schema, not position):")
    for k, v in census.most_common():
        print(f"  n={v:5d}  {k}")
    if bad:
        print(f"  n={bad:5d}  UNPARSEABLE")
    if not rows:
        print("no training rows with both o2 fields")
        return 1

    rows.sort(key=lambda d: d["step"])
    out = []
    for d in rows:
        cov = d["o2_loss"] - d["o2_unweighted"]
        out.append({
            "step": d["step"], "stage": d.get("stage"),
            "o2_loss": d["o2_loss"], "o2_unweighted": d["o2_unweighted"],
            "cov": cov, "rel_pct": abs(cov) / d["o2_unweighted"] * 100.0,
            "sign": "+" if cov > 0 else "-",
            "o2_w_min": d.get("o2_w_min"), "o2_w_max": d.get("o2_w_max"),
            "o5_loss": d.get("o5_loss"), "o5_k": d.get("o5_k"),
            "o2_at_step": d.get("o2_at_step"), "loss": d.get("loss"),
        })

    rel = [r["rel_pct"] for r in out]
    signs = "".join(r["sign"] for r in out)
    npos, nneg = signs.count("+"), signs.count("-")

    print(f"\nn = {len(out)} training rows · steps {out[0]['step']}–{out[-1]['step']}")
    print(f"|Cov|/unweighted:  min {min(rel):.2f}%  p25 {_q(rel,.25):.2f}%  "
          f"median {statistics.median(rel):.2f}%  p75 {_q(rel,.75):.2f}%  "
          f"max {max(rel):.2f}%")
    print(f"sign of Cov: {nneg} negative, {npos} positive")

    # --- trend, binned ---
    print(f"\nTREND ({a.bin}-step bins):")
    print(f"{'bin':>14} {'n':>4} {'median |Cov|/unwt':>18} {'min':>7} {'max':>7} "
          f"{'signs':>12} {'median o2_unwt':>15}")
    bins = collections.defaultdict(list)
    for r in out:
        bins[r["step"] // a.bin].append(r)
    binned = []
    for b in sorted(bins):
        rs = bins[b]
        rl = [r["rel_pct"] for r in rs]
        sg = "".join(r["sign"] for r in rs)
        rec = {"bin_lo": b * a.bin, "bin_hi": (b + 1) * a.bin - 1, "n": len(rs),
               "rel_pct_median": statistics.median(rl), "rel_pct_min": min(rl),
               "rel_pct_max": max(rl), "n_neg": sg.count("-"),
               "n_pos": sg.count("+"),
               "o2_unweighted_median": statistics.median(
                   [r["o2_unweighted"] for r in rs])}
        binned.append(rec)
        print(f"{rec['bin_lo']:6d}–{rec['bin_hi']:<7d} {rec['n']:>4} "
              f"{rec['rel_pct_median']:17.2f}% {rec['rel_pct_min']:6.2f}% "
              f"{rec['rel_pct_max']:6.2f}% {rec['n_neg']:>5}-/{rec['n_pos']:<5}+ "
              f"{rec['o2_unweighted_median']:15.5f}")

    # --- OLS on the full window, carrying window/R^2/n as CLAUDE.md requires ---
    xs = [r["step"] for r in out]
    slope, icept, r2, n = ols(xs, rel)
    fit = {"window": [out[0]["step"], out[-1]["step"]], "n": n,
           "slope_pct_per_1k_steps": slope * 1000.0, "intercept_pct": icept,
           "r2": r2,
           "_admissibility": ("CLAUDE.md: below R^2 0.80 there is no quotable "
                              "exponent/slope — use the matched-step ratio "
                              "instead. Reported here WITH its R^2 so the "
                              "reader can apply that rule, never bare.")}
    print(f"\nOLS |Cov|/unwt vs step over [{fit['window'][0]}, {fit['window'][1]}]: "
          f"slope {fit['slope_pct_per_1k_steps']:+.3f} %/1k steps · "
          f"R^2 {r2:.3f} · n {n}")
    print("  ^ quote ONLY with this window, R^2 and n (CLAUDE.md).")

    # matched-step ratio: the admissible statement when R^2 is low
    first_k = [r["rel_pct"] for r in out if r["step"] <= 1000]
    last_k = [r["rel_pct"] for r in out if r["step"] >= out[-1]["step"] - 1000]
    ratio = {"early_window": [out[0]["step"], 1000], "early_n": len(first_k),
             "early_median_pct": statistics.median(first_k),
             "late_window": [out[-1]["step"] - 1000, out[-1]["step"]],
             "late_n": len(last_k),
             "late_median_pct": statistics.median(last_k)}
    ratio["late_over_early"] = ratio["late_median_pct"] / ratio["early_median_pct"]
    print(f"\nMATCHED-STEP RATIO (the admissible statement when R^2 is low):")
    print(f"  early {ratio['early_window']} n={ratio['early_n']}: "
          f"median {ratio['early_median_pct']:.2f}%")
    print(f"  late  {ratio['late_window']} n={ratio['late_n']}: "
          f"median {ratio['late_median_pct']:.2f}%")
    print(f"  late/early = {ratio['late_over_early']:.2f}x")

    payload = {
        "_evidence_class": "MEASURED (ours; Thor live v6F-SW-30k train_log.jsonl, "
                           "md5 370e778b0b7f79917c94302337f142c1, pulled read-only)",
        "eval_tier": "T0-DIAGNOSTIC (a WM training-log diagnostic; NEVER a driving number)",
        "experiment": "E-O2-A (2026-08-17-O234-DESIGN-RESEARCH.md 6.3)",
        "identity": "o2_loss - o2_unweighted == E_b[Cov_c(w,err)] EXACTLY "
                    "(w mean-1 over cells, v6.py:343-344; both means over the "
                    "same B,C axes, train_v6_staged.py:813-817)",
        "estimator_note": "NO episode-cluster bootstrap: this is a serially "
                          "correlated training-log time series, not a per-window "
                          "eval metric over episodes. overlapping_holdout_se is "
                          "forbidden and was not used.",
        "schema_census": {k: v for k, v in census.items()},
        "n": len(out), "step_min": out[0]["step"], "step_max": out[-1]["step"],
        "rel_pct_min": min(rel), "rel_pct_p25": _q(rel, .25),
        "rel_pct_median": statistics.median(rel), "rel_pct_p75": _q(rel, .75),
        "rel_pct_max": max(rel),
        "n_negative": nneg, "n_positive": npos, "signs": signs,
        "binned": binned, "ols_fit": fit, "matched_step_ratio": ratio,
        "rows": out,
    }
    if a.out:
        a.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nwrote {a.out}")
    return 0


def _q(xs, p):
    s = sorted(xs)
    i = p * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


if __name__ == "__main__":
    sys.exit(main())
