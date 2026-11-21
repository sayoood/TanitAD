"""REP s0 outlier diagnosis (DAG-ORC replicate floor): per-cell mean of T's constraint outputs (cons_pred) and the eval JSON's
longitudinal bias rows, for CTRL (DAG-OACC), REP and A at both seeds. T0 (box 2's standard eval, eval-141, ages cycle): the window list
is deterministic (eval grid + age = f(window index), stack/scripts/v8f_mini_eval.py:269 and :292-294), so the per-window difference of two
runs is on the same windows. Reads only banked artifacts; writes rep_s0_speedcells.json beside this script.
CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe rep_s0_speedcells.py <CONS_NAMES source file>"""
import hashlib
import json
import os
import sys
import numpy as np

src = open(sys.argv[1], encoding="utf-8").read()
ns = {}
exec(src.split("CI =")[0], ns)
NAMES = ns["CONS_NAMES"]
A = "E:/Projects/TanitAD-artifacts/"
R = {"CTRL_s0": A + "v8f-dagoacc/M2b_s0_dagoacc/eval_M2b-dagoacc_s0",
     "REP_s0": A + "v8f-dag-orc-oacc/M2b_s0_dagoaccorcrep/eval_M2b-dagoaccorcrep_s0",
     "A_s0": A + "v8f-dag-orc-oacc/M2b_s0_dagoaccorc/eval_M2b-dagoaccorc_s0",
     "CTRL_s1": A + "v8f-dagoacc/M2b_s1_dagoacc/eval_M2b-dagoacc_s1",
     "REP_s1": A + "v8f-dag-orc-oacc/M2b_s1_dagoaccorcrep/eval_M2b-dagoaccorcrep_s1",
     "A_s1": A + "v8f-dag-orc-oacc/M2b_s1_dagoaccorc/eval_M2b-dagoaccorc_s1"}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
Z = {k: np.load(v + ".npz") for k, v in R.items()}
J = {k: json.load(open(v + ".json")) for k, v in R.items()}
eid0 = Z["CTRL_s0"]["eid"]
assert all(np.array_equal(z["eid"], eid0) for z in Z.values()), "episode ids differ"
assert len(NAMES) == Z["CTRL_s0"]["cons_pred"].shape[1] == 26
SPEED = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
out = {"inputs": {k: {"npz": R[k] + ".npz", "npz_sha256": sha(R[k] + ".npz"), "json_sha256": sha(R[k] + ".json")} for k in R},
       "tier": "T0 (box 2's standard eval, eval-141, ages cycle): a diagnostic, never driving performance",
       "estimator": "per-window paired difference on the same windows; mean and episode-cluster bootstrap 95% CI (B=2000, seed 0)",
       "cells": {}, "longitudinal_rows": {}}
rng = np.random.default_rng(0)
ueid = np.unique(eid0)
idx = {e: np.where(eid0 == e)[0] for e in ueid}
def boot(d):
    m = np.isfinite(d)
    s = np.array([np.nansum(d[idx[e]]) for e in ueid]); n = np.array([np.isfinite(d[idx[e]]).sum() for e in ueid])
    bs = []
    for _ in range(2000):
        pick = rng.integers(0, len(ueid), len(ueid))
        bs.append(s[pick].sum() / max(n[pick].sum(), 1))
    return float(np.nanmean(d[m])), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
for s in ("0", "1"):
    c = Z["CTRL_s" + s]["cons_pred"]
    for arm in ("REP", "A"):
        x = Z[arm + "_s" + s]["cons_pred"]
        for i, n in enumerate(NAMES):
            if n in SPEED:
                out["cells"].setdefault(n, {})[f"{arm}_s{s}-CTRL_s{s}"] = boot(x[:, i] - c[:, i])
                out["cells"][n][f"CTRL_s{s}_mean"] = float(np.nanmean(c[:, i]))
def leaf(j, key):
    return j["raw"]["families"]["longitudinal"]["ci"]["components"][key]["mean"]
for key in ("speed_bias_mps", "speed_mae_mps", "along_bias_m", "along_final_bias_m"):
    out["longitudinal_rows"][key] = {k: leaf(J[k], key) for k in R}
out["ade6_t0"] = {k: J[k]["raw"]["traj"]["ade6_m"]["mean"] for k in R}
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rep_s0_speedcells.json"), "w", newline="\n").write(json.dumps(out, indent=1))
for n in SPEED:
    print(n, {k: tuple(round(v, 4) for v in t) if isinstance(t, tuple) else round(t, 4) for k, t in out["cells"][n].items()})
print(json.dumps(out["longitudinal_rows"], indent=0)); print(out["ade6_t0"])
