"""Assert on a finished arm's ARTIFACTS (never on its exit code). Exit 0 = all integrity checks hold.

Checks (PREREG_DDV2_RL_VALIDATION.md §6): I1 every step present and finite, the parameters moved;
I2 (NORL) the policy-gradient part is exactly zero on every step; I3 (RL) a positive advantage
exists on >= 90 % of steps; the checkpoint and config were written. Prints what it measured.
"""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--steps", type=int, required=True)
ap.add_argument("--arm", choices=("rl", "norl"), required=True)
a = ap.parse_args()
bad = []
mpath = os.path.join(a.run_dir, "metrics.jsonl")
rows = []
if os.path.exists(mpath):
    with open(mpath, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
if len(rows) != a.steps:
    bad.append(f"metrics rows {len(rows)} != steps {a.steps}")
num_keys = ("loss", "rl_part", "il_mean_m", "grad_norm", "param_delta_norm", "reward_mean")
nonfinite = [r["step"] for r in rows if not all(math.isfinite(float(r[k])) for k in num_keys)]
if nonfinite:
    bad.append(f"non-finite at steps {nonfinite[:5]}")
if rows and not rows[-1]["param_delta_norm"] > 0:
    bad.append("parameters did not move")
if a.arm == "norl":
    # ⛔ DEVIATION D-1 (2026-09-15T21:44Z, decided BEFORE any held-out read). The pre-registered
    # form `rl_part == 0.0 and rl_coef_abs_sum == 0.0` FAILED on the first NORL run: `rl_part` is
    # LOGGED as float(li) - float(il_coef) * float(il_i), a float64 subtraction from a float32
    # sum, so its exact equality to 0 can never hold (MEASURED: 600/600 steps non-zero, max
    # 4.2e-8, <= 3.0e-8 of |loss|; RL-s0's median |rl_part| is 0.106). The load-bearing quantity
    # is the COEFFICIENT that multiplies the policy gradient; it must be EXACTLY zero. The logged
    # residual must sit inside float32 round-off. Both halves are checked.
    coef_nz = [r["step"] for r in rows if r["rl_coef_abs_sum"] != 0.0]
    if coef_nz:
        bad.append(f"NORL policy-gradient COEFFICIENT non-zero at steps {coef_nz[:5]}")
    roundoff = [r["step"] for r in rows if abs(r["rl_part"]) > 1e-6 * max(1.0, abs(r["loss"]))]
    if roundoff:
        bad.append(f"NORL logged rl_part above float32 round-off at steps {roundoff[:5]}")
else:
    pos = sum(1 for r in rows if r["frac_positive_after_bar"] > 0)
    if rows and pos / len(rows) < 0.90:
        bad.append(f"positive advantage on only {pos}/{len(rows)} steps")
for f, minsize in (("ckpt.pt", 10 ** 9), ("config.json", 1000), ("run.json", 100)):
    p = os.path.join(a.run_dir, f)
    if not os.path.exists(p) or os.path.getsize(p) < minsize:
        bad.append(f"{f} missing or too small")
summary = {"run_dir": os.path.basename(a.run_dir), "rows": len(rows),
           "first_reward": rows[0]["reward_mean"] if rows else None,
           "last_reward": rows[-1]["reward_mean"] if rows else None,
           "last_il_m": rows[-1]["il_mean_m"] if rows else None,
           "last_delta": rows[-1]["param_delta_norm"] if rows else None,
           "human_nc_eq_1_frac_pooled": (sum(r["human_nc_eq_1_frac"] for r in rows) / len(rows)) if rows else None,
           "problems": bad}
print(json.dumps(summary))
sys.exit(1 if bad else 0)
