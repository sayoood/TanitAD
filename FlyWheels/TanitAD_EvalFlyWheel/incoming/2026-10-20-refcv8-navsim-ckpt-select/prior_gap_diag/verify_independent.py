#!/usr/bin/env python3
"""INDEPENDENT re-derivation (pandas, different code path, literal expectations) of the headline numbers in PRIOR_GAP_DIAG.json.
Reads the raw scorer CSVs + inputs again and compares to the JSON.  Exit 0 iff every check matches.  POST-HOC / DESCRIPTIVE."""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

RUN = "C:/Users/Admin/refcv8_pf/navsim_ckpt/run"
PKG = "C:/Users/Admin/refcv8_pf/navsim_ckpt/prior_gap_diag"
SEL = RUN + "/navtrain_sel"
J = json.load(open(sys.argv[1] if len(sys.argv) > 1 else PKG + "/PRIOR_GAP_DIAG.json"))
inp = json.load(gzip.open(RUN + "/inputs/navtrain_sample_inputs.json.gz"))["tokens"]
meta = pd.DataFrame({"token": list(inp), "log": [inp[t]["log_name"] for t in inp],
                     "cmd": [int(np.argmax(inp[t]["ego_statuses"][-1]["driving_command"])) for t in inp],
                     "v0": [float(np.hypot(*inp[t]["ego_statuses"][-1]["ego_velocity"])) for t in inp]})


def rd(path):
    d = pd.read_csv(path)
    d = d[d["token"] != "average"][["token", "no_at_fault_collisions", "drivable_area_compliance", "score"]]
    return d.rename(columns={"score": "pdms", "no_at_fault_collisions": "nc", "drivable_area_compliance": "dac"})


rt = rd(SEL + "/scores/r8sel_ema48852_s0/r8sel_ema48852_s0.csv")
rt1 = rd(SEL + "/scores/r8sel_ema48852_s1/r8sel_ema48852_s1.csv")
pr = rd(SEL + "/scores/r8sel_ref_PRIOR_ha0p/r8sel_ref_PRIOR_ha0p.csv")
m = meta.merge(rt, on="token").merge(pr, on="token", suffixes=("_rt", "_pr")).merge(rt1.add_suffix("_rt1").rename(columns={"token_rt1": "token"}), on="token")
assert len(m) == 2089
m["gap"] = m.pdms_rt - m.pdms_pr
checks = []


def chk(name, got, want, tol=6e-4):
    ok = abs(got - want) <= tol
    checks.append((name, got, want, ok))


chk("total gap x100", 100 * m.gap.mean(), J["Q1_where"]["headline"]["RT_s0_minus_PRIOR"]["mean_x100"])
for c, nm in ((0, "left"), (1, "straight"), (2, "right")):
    chk(f"{nm} gap x100", 100 * m[m.cmd == c].gap.mean(), J["Q1_where"]["command"][nm]["mean_x100"])
chk("straight prior mean x100", 100 * m[m.cmd == 1].pdms_pr.mean(), J["Q1_where"]["command"]["straight"]["prior_mean_x100"])
chk("right rt mean x100", 100 * m[m.cmd == 2].pdms_rt.mean(), J["Q1_where"]["command"]["right"]["rt_s0_mean_x100"])
b = pd.cut(m.v0, [-1, 2, 8.33, 13.89, 1e9], right=False, labels=False)
for i, nm in enumerate(["[0,2)", "[2,8.33)", "[8.33,13.89)", ">=13.89"]):
    chk(f"band {nm} n", float((b == i).sum()), float(J["Q1_where"]["speed_band"][nm]["n"]), 0)
    chk(f"band {nm} gap x100", 100 * m[b == i].gap.mean(), J["Q1_where"]["speed_band"][nm]["mean_x100"])
# class A: the NC*DAC multiplier of RT is below the PRIOR's
mult_rt, mult_pr = m.nc_rt * m.dac_rt, m.nc_pr * m.dac_pr
A = mult_rt < mult_pr - 1e-9
chk("class A n", float(A.sum()), float(J["Q1_where"]["failure_class_exclusive"]["A_RT_lost_multiplier"]["n"]), 0)
chk("class A contribution x100", 100 * m[A].gap.sum() / len(m), J["Q1_where"]["failure_class_exclusive"]["A_RT_lost_multiplier"]["contrib_x100"])
Bc = mult_rt > mult_pr + 1e-9
chk("class B contribution x100", 100 * m[Bc].gap.sum() / len(m), J["Q1_where"]["failure_class_exclusive"]["B_RT_fixed_multiplier"]["contrib_x100"])
# hybrid gate: PRIOR on straight, RT elsewhere
hyb = np.where(m.cmd == 1, m.pdms_pr, m.pdms_rt)
chk("hybrid mean x100", 100 * hyb.mean(), [g for g in J["Q4a_gating"]["cell_gates"] if g["gate"].startswith("HYBRID")][0]["gated_mean_x100"])
chk("oracle best-of-two x100", 100 * np.maximum(m.pdms_rt, m.pdms_pr).mean(), J["Q4a_gating"]["bounds_x100"]["ORACLE_per_token_best_of_two_NOT_computable_at_inference"])
# literal known values (from the brief, NOT from the JSON)
chk("RT s0 mean x100 (brief 63.40)", 100 * m.pdms_rt.mean(), 63.40, 0.006)
chk("PRIOR mean x100 (brief 73.84)", 100 * m.pdms_pr.mean(), 73.84, 0.006)
chk("RT s1 mean x100 (brief 62.73)", 100 * m.pdms_rt1.mean(), 62.73, 0.006)
# a plain pandas log-cluster bootstrap of the headline gap, independent RNG -> must land near the JSON interval
logs = m.log.unique()
g = m.groupby("log").gap.agg(["sum", "count"])
rng = np.random.default_rng(12345)
bs = []
for _ in range(4000):
    pick = rng.choice(logs, len(logs))
    bs.append(100 * g.loc[pick, "sum"].sum() / g.loc[pick, "count"].sum())
lo, hi = np.percentile(bs, [2.5, 97.5])
J_ci = J["Q1_where"]["headline"]["RT_s0_minus_PRIOR"]["ci95_x100"]
chk("headline CI lo (independent RNG, 4000 draws)", lo, J_ci[0], 0.6)
chk("headline CI hi (independent RNG, 4000 draws)", hi, J_ci[1], 0.6)
bad = [c for c in checks if not c[3]]
for c in checks:
    print(("OK   " if c[3] else "FAIL ") + f"{c[0]:52s} got {c[1]:10.4f}  want {c[2]:10.4f}")
print("INDEPENDENT_VERIFY", "PASS" if not bad else f"FAIL({len(bad)})")
sys.exit(0 if not bad else 4)
