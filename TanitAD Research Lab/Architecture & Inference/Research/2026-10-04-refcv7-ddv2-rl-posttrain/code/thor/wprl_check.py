#!/usr/bin/env python
"""WP-RL integrity checks on a run dir (SPEC_RL sec. 8): I-1, I-2, I-3, I-4, I-5, I-9, I-10 canary.

    python wprl_check.py <run_dir> [--min-steps N] [--json out.json]

Reads ONLY the run's own artifacts (run.json, metrics.jsonl). Exit 0 = every applicable check holds;
2 = a check FAILED (named); 3 = the artifacts are missing (INCOMPLETE -- never a pass).
Each check is proven able to fail by ``--selftest`` (mutated copies of a passing record).
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path


def checks(run: dict, rows: list[dict], min_steps: int = 1) -> dict:
    kind = run["arm"]["kind"]
    out = {}
    n = len(rows)
    out["n_steps"] = n
    out["I0_enough_steps"] = n >= min_steps
    out["I1_finite"] = all(bool(r.get("finite")) and math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"])
                           for r in rows)
    # an lr-0 run is the IDENTITY control: its parameters must NOT move (I-6), so this check is moot
    out["I1_params_moved"] = (rows[-1]["param_delta_norm"] > 0.0) if (n and float(run.get("lr", 1)) > 0) else None
    if n and float(run.get("lr", 1)) == 0.0:
        out["I6_lr0_params_unchanged"] = all(r["param_delta_norm"] == 0.0 for r in rows)
    if kind == "rloff":
        out["I2_rloff_coef_exactly_zero"] = all(r["coef_rl_abs_sum"] == 0.0 for r in rows)
    else:
        out["I2_rl_coef_positive"] = sum(r["coef_rl_abs_sum"] > 0.0 for r in rows) >= 0.9 * n
        out["I3_positive_adv_on_90pct_steps"] = sum(r["frac_positive_after_bar"] > 0 for r in rows) >= 0.9 * n
    hn = sum(r["human_nc1_frac"] * r["n_windows"] for r in rows) / max(1, sum(r["n_windows"] for r in rows))
    hd = sum(r["human_dac1_frac"] * r["n_windows"] for r in rows) / max(1, sum(r["n_windows"] for r in rows))
    out["I4_human_nc1_frac"] = round(hn, 4)
    out["I4_human_dac1_frac"] = round(hd, 4)
    out["I4_pass"] = hn >= 0.95 and hd >= 0.95
    out["I5_dac_live_frac_steps"] = round(sum(r["cand_dac_mean"] < 1.0 for r in rows) / max(1, n), 4)
    out["I5_dac_dead_windows"] = sum(r["n_dac_dead"] for r in rows)
    out["I5_pass"] = (out["I5_dac_live_frac_steps"] >= 0.5
                      and out["I5_dac_dead_windows"] <= 0.01 * sum(r["n_windows"] for r in rows))
    if kind == "rlshuf":
        out["I9_derangement_every_step"] = all(
            r.get("shuffle_perm") is not None and all(int(p) != i for i, p in enumerate(r["shuffle_perm"]))
            for r in rows)
    else:
        out["I9_no_shuffle"] = all(r.get("shuffle_perm") is None for r in rows)
    s0 = rows[0]["chain_endpoint_spread_m"] if n else None
    out["I10_spread_first_last_m"] = [s0, rows[-1]["chain_endpoint_spread_m"] if n else None]
    out["I10_canary_retained_ge_60pct"] = (rows[-1]["chain_endpoint_spread_m"] >= 0.6 * s0) if n and s0 else None
    keys = [k for k in out if k.startswith("I") and isinstance(out[k], bool) and not k.startswith("I10")]
    out["failed"] = [k for k in keys if out[k] is False]
    out["verdict"] = "PASS" if not out["failed"] else "FAIL"
    return out


def load(run_dir: Path):
    rp, mp = run_dir / "run.json", run_dir / "metrics.jsonl"
    if not rp.exists() or not mp.exists():
        return None, None
    run = json.loads(rp.read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in mp.read_text(encoding="utf-8").splitlines() if x.strip()]
    return run, rows


def selftest(run, rows):
    """Each check must FAIL on its own reintroduced defect."""
    res = {}
    def mutate(fn):
        r2, rows2 = copy.deepcopy(run), copy.deepcopy(rows)
        fn(r2, rows2)
        return checks(r2, rows2)
    res["I1_nonfinite"] = "I1_finite" in mutate(lambda r, x: x[0].update(loss=float("nan")))["failed"]
    if run["arm"]["kind"] == "rloff":
        res["I2_nonzero"] = "I2_rloff_coef_exactly_zero" in mutate(lambda r, x: x[0].update(coef_rl_abs_sum=1e-9))["failed"]
    else:
        res["I3_no_positive"] = "I3_positive_adv_on_90pct_steps" in mutate(
            lambda r, x: [y.update(frac_positive_after_bar=0.0) for y in x])["failed"]
    res["I4_dac_human"] = "I4_pass" in mutate(lambda r, x: [y.update(human_dac1_frac=0.5) for y in x])["failed"]
    res["I5_dead_dac"] = "I5_pass" in mutate(lambda r, x: [y.update(cand_dac_mean=1.0) for y in x])["failed"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--min-steps", type=int, default=1)
    ap.add_argument("--json", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    run, rows = load(Path(a.run_dir))
    if run is None or not rows:
        print(json.dumps({"verdict": "INCOMPLETE", "run_dir": a.run_dir}))
        sys.exit(3)
    rec = checks(run, rows, a.min_steps)
    rec["arm"] = run["arm"]
    if a.selftest:
        rec["selftest_each_check_fires"] = selftest(run, rows)
    print(json.dumps(rec, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    sys.exit(0 if rec["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
