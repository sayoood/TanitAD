#!/usr/bin/env python3
"""D-SAFE-CAL-2 readout — the exit selected MECHANICALLY from PREREG_D_SAFE_CAL_2 §3.

The five seed-1 arms RAN on 2026-08-30 (`tanitad-data/rl-pilot/d2-*`, logs 08:51-09:12)
and were never analysed or written up — a pre-registered experiment stranded on one
disk (found 2026-09-05 by the RL-readiness audit). This script applies the committed
table in its committed ORDER: V1, V2 (VOID conditions override everything), then
exits 1-5, first match wins. No branch is read into a result.

Estimator: PAIRED episode-cluster bootstrap over the 15 val episodes, 4000 reps
(`per_episode` is stored in every readout; eval-mode readout is deterministic, so the
pairing is exact). V1's reference is the BANKED seed-0 no-proximity w=1 arm
(`raw/p_rc21_sweep/sweep_paired.json` → `s2-w1`): D must reproduce it within D's own
paired CI on R1 AND R3.

`R4_full` (§1b): the readout's R4 is scored with the DEFAULT spec, so the varied term
`proximity` is recovered from `dsafe2_prox_decomp.json` (before/after proximity means
per arm, computed post hoc from `ckpt_after.pt`, zero GPU) and reported alongside.
"""
import json
import pathlib
import sys

import numpy as np

O = pathlib.Path(r"C:/Users/Admin/tanitad-data/rl-pilot")
BANKED = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None   # sweep_paired.json
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else O / "dsafe2_result.json"
ARMS = {"A": "d2-A-w1-d2", "B": "d2-B-w10-d2", "C": "d2-C-w1-d5",
        "D": "d2-D-repro-d2", "REG": "d2-REG-hackable"}
KEYS = {"R1": "r", "R2": "cr", "R3": "ade", "R5": "viol", "R5arm": "viol_arm"}


def paired(before, after, key, reps=4000, seed=11):
    eps = sorted(set(before) & set(after))
    d = np.array([after[e][key] - before[e][key] for e in eps
                  if np.isfinite(after[e].get(key, np.nan))
                  and np.isfinite(before[e].get(key, np.nan))])
    if d.size == 0:
        return {"delta": float("nan"), "ci": [float("nan")] * 2, "n": 0, "sep": False}
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(reps)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci": [float(lo), float(hi)], "n": int(d.size),
            "sep": bool((lo > 0) == (hi > 0))}


res, logs = {}, {}
for label, sub in ARMS.items():
    b = json.loads((O / sub / "readout_before.json").read_text(encoding="utf-8"))
    a = json.loads((O / sub / "readout_after.json").read_text(encoding="utf-8"))
    cfg = json.loads((O / sub / "config.json").read_text(encoding="utf-8"))
    row = {"dir": sub, "seed": cfg.get("seed"), "w_anchor": cfg.get("w_anchor"),
           "steps": cfg.get("steps"),
           "reward_weights_in_force": cfg.get("reward_weights_in_force"),
           "n_ep": len(a["per_episode"]), "eval_mode": a.get("eval_mode")}
    for name, k in KEYS.items():
        row[name] = paired(b["per_episode"], a["per_episode"], k)
    row["R4_default_before"] = b["R4_component_means"]
    row["R4_default_after"] = a["R4_component_means"]
    row["R5_after_fixed"] = a["R5_gt_clearance_violation_frac"]["mean"]
    res[label] = row
prox = json.loads((O / "dsafe2_prox_decomp.json").read_text(encoding="utf-8"))
for label, sub in ARMS.items():
    res[label]["R4_full_proximity"] = prox.get(sub)

# ------------------------------------------------------------------- the exit
verdict, why = None, []
A, B, C, D, REG = (res[k] for k in ("A", "B", "C", "D", "REG"))

# V1 — reproduction test against the BANKED seed-0 no-proximity w=1 arm
v1 = {"rule": "D must MATCH the banked no-proximity arm within D's paired CI (R1 and R3)"}
if BANKED and BANKED.exists():
    bk = json.loads(BANKED.read_text(encoding="utf-8"))["s2-w1"]
    v1["banked_s2_w1"] = {k: bk[k] for k in ("R1", "R2", "R3")}
    inside = {k: bool(D[k]["ci"][0] <= bk[k]["delta"] <= D[k]["ci"][1]) for k in ("R1", "R3")}
    v1["banked_point_inside_D_ci"] = inside
    v1["PASS"] = all(inside.values())
else:
    v1["PASS"] = None
    v1["note"] = "sweep_paired.json not supplied — V1 UNEVALUATED"
if v1["PASS"] is False:
    verdict = "VOID (V1) — arm D DIVERGES from the banked no-proximity arm"
    why.append(f"banked s2-w1 dR1 {v1['banked_s2_w1']['R1']['delta']:+.4f} / dR3 "
               f"{v1['banked_s2_w1']['R3']['delta']:+.4f} vs D's paired CI "
               f"R1 {D['R1']['ci']} / R3 {D['R3']['ci']}: inside={v1['banked_point_inside_D_ci']}")
# V2 — the regression licence
v2 = {"rule": "REG must degrade with paired separation on R1 or R3",
      "R1_sep_neg": bool(REG["R1"]["sep"] and REG["R1"]["delta"] < 0),
      "R3_sep_pos": bool(REG["R3"]["sep"] and REG["R3"]["delta"] > 0)}
v2["PASS"] = v2["R1_sep_neg"] or v2["R3_sep_pos"]
if not v2["PASS"] and verdict is None:
    verdict = "VOID (V2) — the deliberate-regression control did not degrade with separation"
    why.append("the readout cannot see failure; no null in the table is a measurement")

prox_move_A = abs(A["R4_full_proximity"]["delta"]) if A.get("R4_full_proximity") else float("nan")
a_r2_falls = bool(A["R2"]["sep"] and A["R2"]["delta"] < 0)
c_null = not C["R2"]["sep"]
a_r5_rises = bool(A["R5"]["sep"] and A["R5"]["delta"] > 0)
a_r3_deg = bool(A["R3"]["sep"] and A["R3"]["delta"] > 0)
if verdict is None:
    if a_r2_falls and c_null:
        verdict = "EXIT 1 — THE THRESHOLD WAS THE BLOCKER"
        why.append("dR2 fell with paired separation in A while C reproduced the banked null")
    elif (not A["R2"]["sep"]) and (not C["R2"]["sep"]) and prox_move_A < 0.01:
        verdict = "EXIT 2 — THE THRESHOLD WAS NOT THE BLOCKER; THE RL LINE CLOSES ON THIS BASE MODEL"
        why.append(f"A and C both null on R2 (A {A['R2']['delta']*100:+.3f} pp {A['R2']['ci']}, "
                   f"C {C['R2']['delta']*100:+.3f} pp {C['R2']['ci']}) and R4_full proximity moved "
                   f"{prox_move_A:.4f} (< 0.01) in A")
    elif a_r2_falls and a_r5_rises:
        verdict = "EXIT 3 — REWARD HACKING"
        why.append("the scored fan improved while the driven path's violation rate rose on the frozen ruler")
    elif a_r3_deg and not A["R2"]["sep"]:
        verdict = "EXIT 4 — the opposition is not threshold-specific (closes as exit 2, conflict persists)"
        why.append(f"dR3 degraded with separation in A ({A['R3']['delta']:+.4f} m {A['R3']['ci']}) "
                   "while dR2 stayed flat")
    else:
        verdict = "EXIT 5 — NO COMMITTED BRANCH MATCHES; report as-is, decide nothing"

out = {"_what": "D-SAFE-CAL-2 mechanical readout (PREREG_D_SAFE_CAL_2 §3, committed order)",
       "_estimator": "paired episode-cluster bootstrap, 4000 reps, 15 val episodes; eval-mode "
                     "deterministic readout; NON-PARITY pilot corpus; T0 training-side",
       "_evidence_class": "MEASURED (ours; the banked seed-1 arm readouts of 2026-08-30)",
       "arms": res, "V1": v1, "V2": v2,
       "exit_inputs": {"A_R2_falls_sep": a_r2_falls, "C_R2_null": c_null,
                       "A_R5_rises_sep": a_r5_rises, "A_R3_degrades_sep": a_r3_deg,
                       "A_R4_full_proximity_abs_move": prox_move_A},
       "verdict": verdict, "why": why}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

print("=" * 100)
print(f"{'arm':<6}{'w':>5}{'d_safe/wprox':>14}{'dR1':>26}{'dR2 (pp)':>26}{'dR3 (m)':>26}")
print("=" * 100)
for label, r in res.items():
    def f(k, s=1.0):
        v = r[k]; m = "*" if v["sep"] else " "
        return f"{v['delta']*s:+.4f}{m}[{v['ci'][0]*s:+.4f},{v['ci'][1]*s:+.4f}]".rjust(26)
    w = r["reward_weights_in_force"] or {}
    print(f"{label:<6}{r['w_anchor']:>5}{str(w.get('proximity')):>14}{f('R1')}{f('R2', 100)}{f('R3')}")
print("  (* = paired 95% CI excludes zero)")
print(f"\nR4_full proximity Δ: " + ", ".join(f"{k} {r['R4_full_proximity']['delta']:+.4f}" for k, r in res.items()))
print(f"R5 @5.0 m fixed: " + ", ".join(f"{k} {r['R5']['delta']:+.4f}{'*' if r['R5']['sep'] else ''}" for k, r in res.items()))
print(f"\nV1 {v1}\nV2 {v2}\n\n  ⇒ {verdict}")
for w in why:
    print(f"     · {w}")
print(f"-> {OUT}")
