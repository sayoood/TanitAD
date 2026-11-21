"""The PRE-REGISTERED seed-group check (`raw/PREREG_SEED_GROUP_CHECK_50400.md`, sha256 in
`raw/PREREG_SEED_GROUP_CHECK_50400.sha256`), run exactly as registered. Zero GPU.

On a 24-seed G0 artifact, for `eval_traj`:
  * the row-mean sd of seeds 0-7 and of seeds 8-23;
  * the one-sided F test var(8-23) / var(0-7), df (15, 7);
  * verdict: (A) p >= 0.05 -- the step-5,000 pattern does not recur, "an 8-sample sd taken as stable"
    is the class; (B) p < 0.01 in the same direction -- seed- or harness-structured; else INCONCLUSIVE.

The "row mean" is the mean over G0's 8 batches of that seed's per-batch `traj` (full precision); the
5-dp `eval_traj` row value is carried beside it as a rounding CONTROL and the verdict is computed on
both (they must agree, else the check reports the disagreement instead of a verdict).

⭐ KNOWN-VALUE CONTROL (a check must be able to read a value it did not produce): with `--control-5000`
the same code recomputes the step-5,000 numbers the pre-registration QUOTES (seeds 0-7 from
`raw/step5000/g0.json`, seeds 8-23 from `raw/g0diag_step5000/diag.json`): sd 0.00212 / 0.01308,
F = 38.1, one-sided p = 3.1e-5, Levene p = 0.018. If those do not reproduce, the step-50,400 verdict is
not reported (the instrument is wrong, not the data).

The verdict is a diagnostic of the instrument; it never changes a G0 verdict (prereg, last line).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from scipy import stats


def _row_means_g0(g0: dict, seeds) -> dict:
    out = {}
    for s in seeds:
        rec = g0["by_seed"][str(s)]
        pb = [float(b["traj"]) for b in rec["per_batch"]]
        out[s] = {"row_mean_full": statistics.fmean(pb), "row_5dp": float(rec["row"]["eval_traj"]),
                  "n_batches": len(pb)}
    return out


def _row_means_diag(diag: dict, seeds) -> dict:
    out = {}
    for s in seeds:
        rec = diag["arms"][f"seed{s}"]
        pb = [float(x) for x in rec["per_batch_traj"]]
        out[s] = {"row_mean_full": statistics.fmean(pb), "row_5dp": float(rec["row"]["eval_traj"]),
                  "n_batches": len(pb)}
    return out


def test(lo: list, hi: list) -> dict:
    v_lo, v_hi = statistics.variance(lo), statistics.variance(hi)
    F = v_hi / v_lo
    df1, df2 = len(hi) - 1, len(lo) - 1
    p_one = float(stats.f.sf(F, df1, df2))           # H1: var(8-23) > var(0-7)
    lev = stats.levene(hi, lo, center="median")      # Brown-Forsythe form (scipy default)
    if p_one >= 0.05:
        verdict = "A"
    elif p_one < 0.01 and F > 1:
        verdict = "B"
    else:
        verdict = "INCONCLUSIVE"
    return {"n_0_7": len(lo), "n_8_23": len(hi), "sd_0_7": v_lo ** 0.5, "sd_8_23": v_hi ** 0.5,
            "mean_0_7": statistics.fmean(lo), "mean_8_23": statistics.fmean(hi),
            "F": F, "df": [df1, df2], "p_one_sided": p_one,
            "levene_p": float(lev.pvalue), "levene_center": "median", "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", required=True, help="the 24-seed G0 artifact (raw/step50400/g0.json)")
    ap.add_argument("--control-5000-g0", default=None)
    ap.add_argument("--control-5000-diag", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"tool": "seed_group_check.py", "prereg": "raw/PREREG_SEED_GROUP_CHECK_50400.md",
           "term": "eval_traj", "estimator": "row mean over G0's 8 batches per inference seed; "
           "one-sided F test of sample variances, df (15, 7); Levene (median-centred) as the robust "
           "companion the prereg quotes"}
    ctl_ok = True
    if a.control_5000_g0 and a.control_5000_diag:
        g5 = json.load(open(a.control_5000_g0, encoding="utf-8"))
        d5 = json.load(open(a.control_5000_diag, encoding="utf-8"))
        r_lo = _row_means_g0(g5, range(8))
        r_hi = _row_means_diag(d5, range(8, 24))
        t5 = test([r_lo[s]["row_5dp"] for s in r_lo], [r_hi[s]["row_5dp"] for s in r_hi])
        t5f = test([r_lo[s]["row_mean_full"] for s in r_lo], [r_hi[s]["row_mean_full"] for s in r_hi])
        expect = {"sd_0_7": 0.00212, "sd_8_23": 0.01308, "F": 38.1, "p_one_sided": 3.1e-5,
                  "levene_p": 0.018}
        # literals from the prereg's table (written before this code); tolerance = the prereg's own
        # printed precision
        tol = {"sd_0_7": 5e-6, "sd_8_23": 5e-6, "F": 0.05, "p_one_sided": 0.05e-5, "levene_p": 5e-4}
        chk = {k: {"expected": v, "got_5dp": t5[k], "got_full": t5f[k],
                   "ok": (abs(t5[k] - v) <= tol[k]) or (abs(t5f[k] - v) <= tol[k])}
               for k, v in expect.items()}
        ctl_ok = all(c["ok"] for c in chk.values())
        out["control_step5000"] = {"reproduces_prereg_quotes": ctl_ok, "checks": chk,
                                   "test_5dp": t5, "test_full": t5f}
    g0 = json.load(open(a.g0, encoding="utf-8"))
    seeds = sorted(int(s) for s in g0["by_seed"])
    if seeds != list(range(24)):
        out["REFUSED"] = f"seeds {seeds} != 0..23"
    else:
        rm = _row_means_g0(g0, seeds)
        out["step"] = g0.get("step")
        out["ckpt_md5"] = g0.get("ckpt_md5")
        out["per_seed"] = rm
        out["rounding_control_max_abs"] = max(abs(v["row_mean_full"] - v["row_5dp"]) for v in rm.values())
        tf = test([rm[s]["row_mean_full"] for s in range(8)], [rm[s]["row_mean_full"] for s in range(8, 24)])
        t5 = test([rm[s]["row_5dp"] for s in range(8)], [rm[s]["row_5dp"] for s in range(8, 24)])
        out["test_full"] = tf
        out["test_5dp"] = t5
        if not ctl_ok:
            out["VERDICT"] = "NOT REPORTED -- the step-5,000 known-value control did not reproduce"
        elif tf["verdict"] != t5["verdict"]:
            out["VERDICT"] = f"DISAGREE full={tf['verdict']} 5dp={t5['verdict']}"
        else:
            out["VERDICT"] = tf["verdict"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: out.get(k) for k in ("VERDICT", "test_full", "test_5dp", "rounding_control_max_abs")},
                     indent=1, default=str))
    if "control_step5000" in out:
        print("control_step5000 reproduces:", out["control_step5000"]["reproduces_prereg_quotes"],
              json.dumps({k: (round(v["got_full"], 6), round(v["got_5dp"], 6), v["ok"])
                          for k, v in out["control_step5000"]["checks"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
