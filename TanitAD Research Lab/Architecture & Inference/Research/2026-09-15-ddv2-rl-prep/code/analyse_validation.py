"""PREREG_DDV2_RL_VALIDATION.md §5-§7, computed from the ARTIFACTS the chain wrote.

Writes one JSON (numbers + verdicts) and prints the tables. Clip ids never appear: held-out rows
are keyed by sha12, T1 comparisons are run through ``taniteval/tools/paired_openloop.py`` and only
its numeric records are kept.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

WT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.insert(0, os.path.join(WT, "taniteval"))
from taniteval.ci import episode_cluster_bootstrap, paired_episode_cluster_bootstrap  # noqa: E402

ARMS = ["base", "arm-rl-s0", "arm-norl-s0", "arm-rl-s1"]
TRAINED = [("arm-rl-s0", "rl"), ("arm-norl-s0", "norl"), ("arm-rl-s1", "rl")]
T0_METRICS = ["fan_pdms_mean", "fan_pdms_best", "fan_nc_fail_frac", "fan_endpoint_spread_m", "fan_minade_m",
              "sel_pdms", "sel_nc", "sel_ttc", "sel_ep", "sel_comfort", "sel_ade_m", "sel_fde_m", "human_pdms"]
N_BOOT, SEED = 2000, 0


def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


def training_block(runs):
    out = {}
    for name, kind in TRAINED:
        p = os.path.join(runs, name, "metrics.jsonl")
        if not os.path.exists(p):
            out[name] = {"status": "MISSING"}
            continue
        rows = load_jsonl(p)
        first, last = rows[:50], rows[-50:]
        keys = ["reward_mean", "reward_best_mean", "il_mean_m", "loss", "rl_part", "grad_norm",
                "frac_positive_after_bar", "frac_admitted_by_bar", "frac_constraint_fail", "cand_nc_mean",
                "cand_ttc_mean", "cand_ep_mean", "cand_comfort_mean", "x0_clamp_frac_final_a_lon",
                "x0_clamp_frac_final_a_lat", "input_clamp_frac_a_lat", "il_weight_mean"]
        blk = {"n_steps": len(rows), "first50": {k: mean([r[k] for r in first]) for k in keys},
               "last50": {k: mean([r[k] for r in last]) for k in keys},
               "param_delta_norm_end": rows[-1]["param_delta_norm"], "wall_s": rows[-1]["wall_s"]}
        blk["I1_finite_all"] = all(bool(r["finite"]) for r in rows)
        if kind == "norl":
            blk["I2_rl_part_exactly_zero_all"] = all(r["rl_part"] == 0.0 and r["rl_coef_abs_sum"] == 0.0 for r in rows)
        else:
            blk["I3_positive_step_frac"] = sum(1 for r in rows if r["frac_positive_after_bar"] > 0) / len(rows)
        blk["human_nc_eq_1_frac_pooled"] = mean([r["human_nc_eq_1_frac"] for r in rows])
        out[name] = blk
    pooled = [out[n]["human_nc_eq_1_frac_pooled"] for n, _ in TRAINED if "human_nc_eq_1_frac_pooled" in out[n]]
    out["I4_human_nc_pooled"] = mean(pooled)
    return out


def heldout_block(runs):
    data = {}
    for name in ARMS + ["base_repeat"]:
        p = os.path.join(runs, f"heldout_{name}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                data[name] = json.load(fh)["rows"]
    out = {"present": sorted(data)}
    if "base" not in data:
        return out
    key = [(r["sha12"], r["t0"]) for r in data["base"]]
    for name, rows in data.items():
        if [(r["sha12"], r["t0"]) for r in rows] != key:
            raise SystemExit(f"held-out rows of {name} are not aligned with base — pairing broken")
    eid = [r["sha12"] for r in data["base"]]
    out["n_windows"], out["n_clips"] = len(key), len(set(eid))
    if "base_repeat" in data:
        mism = sum(1 for a, b in zip(data["base"], data["base_repeat"])
                   for m in T0_METRICS if a[m] != b[m])
        out["I5_base_repeat_bitwise"] = (mism == 0)
        out["I5_mismatches"] = mism
    hp = {n: [r["human_pdms"] for r in data[n]] for n in data}
    out["C_human_pdms_identical_across_ckpts"] = all(hp[n] == hp["base"] for n in hp)
    levels = {}
    for n in ARMS:
        if n not in data:
            continue
        levels[n] = {}
        for m in T0_METRICS:
            v = np.array([r[m] for r in data[n]], dtype=np.float64)
            ci = episode_cluster_bootstrap(v, eid, n_boot=N_BOOT, seed=SEED)
            levels[n][m] = {"mean": float(v.mean()), "lo": ci.get("lo"), "hi": ci.get("hi")}
    out["levels"] = levels
    pairs = [("arm-rl-s0", "arm-norl-s0"), ("arm-rl-s1", "arm-norl-s0"), ("arm-rl-s0", "base"),
             ("arm-rl-s1", "base"), ("arm-norl-s0", "base"), ("arm-rl-s1", "arm-rl-s0")]
    deltas = {}
    for a, b in pairs:
        if a not in data or b not in data:
            continue
        deltas[f"{a} - {b}"] = {}
        for m in T0_METRICS:
            va = np.array([r[m] for r in data[a]], dtype=np.float64)
            vb = np.array([r[m] for r in data[b]], dtype=np.float64)
            res = paired_episode_cluster_bootstrap(va, vb, eid, n_boot=N_BOOT, seed=SEED)
            deltas[f"{a} - {b}"][m] = {"delta_exact": float(va.mean() - vb.mean()), "delta": res["delta"],
                                       "lo": res["lo"], "hi": res["hi"], "separated": res["separated"]}
    out["deltas"] = deltas
    verdict = None
    k0, k1 = "arm-rl-s0 - arm-norl-s0", "arm-rl-s1 - arm-norl-s0"
    if k0 in deltas and k1 in deltas:
        d0, d1 = deltas[k0]["fan_pdms_mean"], deltas[k1]["fan_pdms_mean"]
        # ⛔ decided on ci.py's UNROUNDED `separated` + the exact sign, never on the rendered
        # bounds: `_render_bounds` rounds, and a tiny positive lower bound can print as 0.0
        pass0 = bool(d0["separated"]) and d0["delta_exact"] > 0
        pass1 = bool(d1["separated"]) and d1["delta_exact"] > 0
        neg = (bool(d0["separated"]) and d0["delta_exact"] < 0) or (bool(d1["separated"]) and d1["delta_exact"] < 0)
        if pass0 and pass1:
            verdict = "SUCCESS"
        elif (pass0 or pass1) and not neg:
            verdict = "PARTIAL"
        else:
            verdict = "FAILURE"
    out["H1_verdict"] = verdict
    return out


def t1_families(p):
    with open(p, encoding="utf-8") as fh:
        r = json.load(fh)
    os_ = r["arms"]["os"]
    ff = os_["four_families"]
    pick = {"longitudinal": ["speed_mae_mps", "speed_bias_mps", "along_mae_m", "along_bias_m", "accel_mae_mps2"],
            "lateral": ["heading_mae_deg", "cross_mae_m", "cross_bias_m", "curvature_mae_1pm"]}
    fam = {k: {m: ff[k].get(m) for m in ms} for k, ms in pick.items()}
    def _num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    for k in ("tactical", "strategic"):
        v = ff.get(k)
        if not isinstance(v, dict):
            fam[k] = v
            continue
        flat = {"status": v.get("status")}
        for m, x in v.items():
            if _num(x):
                flat[m] = x
            elif isinstance(x, dict):                      # one level: e.g. lateral_decision.accuracy
                for mm, xx in x.items():
                    if _num(xx):
                        flat[f"{m}.{mm}"] = xx
        fam[k] = flat
    iv = (os_.get("intervals") or {}).get("metrics", {})       # {tier, n, estimator, metrics: {...}}
    fam["intervals"] = {m: {kk: iv[m].get(kk) for kk in ("mean", "lo", "hi", "n_windows", "n_episodes") if kk in iv[m]}
                        for m in iv if isinstance(iv[m], dict)}
    if not fam["intervals"]:
        raise SystemExit(f"{p}: no T1 intervals found under arms.os.intervals.metrics")
    return {"n_windows": r.get("n_windows"), "n_episodes": r.get("n_episodes"), "families": fam}


def run_paired(runs, a, b, a_dump=None, b_dump=None, tag=None):
    py = sys.executable
    tag = tag or f"{a}__vs__{b}"
    outp = os.path.join(runs, f"paired_{tag}.json")
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.pathsep.join([os.path.join(WT, "stack"), os.path.join(WT, "taniteval"), WT]))
    cmd = [py, os.path.join(WT, "taniteval", "tools", "paired_openloop.py"),
           "--a-dump", a_dump or os.path.join(runs, f"t1_{a}_dump"), "--a-name", a, "--a-arm", "os",
           "--b-dump", b_dump or os.path.join(runs, f"t1_{b}_dump"), "--b-name", b, "--b-arm", "os",
           "--floor", "ha0", "--n-boot", str(N_BOOT), "--seed", str(SEED), "--out", outp]
    p = subprocess.run(cmd, cwd=WT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    with open(os.path.join(runs, f"paired_{tag}.log"), "w", encoding="utf-8") as fh:
        fh.write(p.stdout + "\n" + p.stderr)
    if p.returncode != 0 or not os.path.exists(outp):
        return {"status": "FAILED", "returncode": p.returncode, "stderr_tail": p.stderr[-800:]}
    rows = []
    for line in p.stdout.splitlines():
        s = line.strip()
        if s.startswith(("ade_m", "fde_m", "LON_", "LAT_", "TAC_", "STR_")):
            parts = s.split()
            try:
                ci = s[s.index("["):s.index("]") + 1]
                rows.append({"metric": parts[0], "A_abs": float(parts[1]), "B_abs": float(parts[2]),
                             "floor": float(parts[3]), "B_minus_A": float(parts[6]), "ci": ci,
                             "sep": parts[-1]})
            except (ValueError, IndexError):
                continue
    n_line = [l for l in p.stdout.splitlines() if l.strip().startswith("n = ")]
    return {"status": "OK", "record": os.path.basename(outp), "n": n_line[0].strip() if n_line else None,
            "convention": "B_minus_A = (B - floor) - (A - floor); for error metrics POSITIVE means A (first arm) is better",
            "rows": rows}


def t1_block(runs):
    out = {"families": {}, "paired": {}}
    for n in ARMS:
        p = os.path.join(runs, f"t1_{n}.json")
        if os.path.exists(p):
            out["families"][n] = t1_families(p)
    have = set(out["families"])
    for a, b in [("arm-rl-s0", "base"), ("arm-rl-s1", "base"), ("arm-norl-s0", "base"),
                 ("arm-rl-s0", "arm-norl-s0"), ("arm-rl-s1", "arm-norl-s0")]:
        if a in have and b in have:
            out["paired"][f"{a} vs {b}"] = run_paired(runs, a, b)
    land = "C:/Users/Admin/refcv5cmp/out/refcv5-v2_dump"
    if "base" in have and os.path.isdir(land):
        out["paired"]["I6 base(tip, held-out roll) vs landing dump"] = run_paired(
            runs, "base", "landing", b_dump=land, tag="I6_base_vs_landing")
    h2 = None
    try:
        worse = []
        for s in ("arm-rl-s0 vs base", "arm-rl-s1 vs base"):
            row = [r for r in out["paired"][s]["rows"] if r["metric"] == "ade_m"][0]
            hi = float(row["ci"].strip("[]").split(",")[1])
            worse.append(hi < 0)
        h2 = "FAIL-HARM" if all(worse) else "no harm detected at this n"
    except (KeyError, IndexError, ValueError):
        h2 = None
    out["H2_verdict"] = h2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="C:/Users/Admin/tanitad-caches/ddv2rl-20260915")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-t1", action="store_true")
    a = ap.parse_args()
    rec = {"_prereg": "Project Steering/PREREG_DDV2_RL_VALIDATION.md", "_tiers": {"training": "T0",
           "heldout": "T0 (deployed sampler, proxy reward)", "t1": "T1 self-action OPEN loop"},
           "_n_boot": N_BOOT, "_seed": SEED}
    rec["training"] = training_block(a.runs)
    rec["heldout_T0"] = heldout_block(a.runs)
    if not a.skip_t1:
        rec["T1"] = t1_block(a.runs)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1)[:20000])


if __name__ == "__main__":
    main()
