"""D-SAFE-CAL readout — the exit is selected MECHANICALLY from the committed table.

⛔ THE POINT OF THIS SCRIPT: the exit condition is chosen by code from the
pre-registered rules, not by me reading a table and deciding which branch it
"feels like". Every prior campaign exit in this line was selected the same way,
and it is the only defence against reading a favourable branch into an ambiguous
result.

Estimator: PAIRED episode-cluster bootstrap over the val episodes, 4000 reps,
same windows before/after (the readout is eval-mode deterministic, so the pairing
is exact). House rule: two arms on the same windows use the PAIRED version.
"""
import json
import pathlib
import sys

import numpy as np

O = pathlib.Path(r"C:/Users/Admin/tanitad-data/rl-pilot")
ARMS = {
    "A  w=1  d_safe=2": "dsafe-A-w1-d2",
    "B  w=10 d_safe=2": "dsafe-B-w10-d2",
    "C  w=1  d_safe=5": "dsafe-C-w1-d5",
    "D  floor (w_prox=0)": "dsafe-D-floor-d2",
    "REG deliberate-regression": "dsafe-REG-hackable",
}
KEYS = {"R1": "r", "R2": "cr", "R3": "ade", "R5": "viol", "R5arm": "viol_arm"}


def paired(before, after, key, reps=4000, seed=11):
    """Paired episode-cluster bootstrap of the per-episode delta."""
    eps = sorted(set(before) & set(after))
    d = np.array([after[e][key] - before[e][key] for e in eps
                  if np.isfinite(after[e].get(key, np.nan))
                  and np.isfinite(before[e].get(key, np.nan))])
    if d.size == 0:
        return float("nan"), (float("nan"), float("nan")), 0, False
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(reps)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), (float(lo), float(hi)), int(d.size), bool((lo > 0) == (hi > 0))


res = {}
for label, sub in ARMS.items():
    b_p, a_p = O / sub / "readout_before.json", O / sub / "readout_after.json"
    if not a_p.exists():
        print(f"  ⚠️ {label}: no readout_after — SKIPPED")
        continue
    b, a = json.loads(b_p.read_text()), json.loads(a_p.read_text())
    row = {"n_ep": len(a["per_episode"])}
    for name, k in KEYS.items():
        m, ci, n, sep = paired(b["per_episode"], a["per_episode"], k)
        row[name] = {"delta": m, "ci": ci, "n": n, "sep": sep}
    row["R4_before"] = b["R4_component_means"]
    row["R4_after"] = a["R4_component_means"]
    row["R5_after_fixed"] = a["R5_gt_clearance_violation_frac"]["mean"]
    row["R5_after_arm"] = a["R5_gt_clearance_violation_frac"].get(
        "mean_at_arm_threshold")
    res[label] = row

print("=" * 96)
print(f"{'arm':<27}{'dR1':>20}{'dR2 (pp)':>20}{'dR3 (m)':>20}")
print("=" * 96)
for label, r in res.items():
    def f(k, scale=1.0, w=20):
        v = r[k]
        s = "*" if v["sep"] else " "
        return f"{v['delta']*scale:+.4f}{s}[{v['ci'][0]*scale:+.3f},{v['ci'][1]*scale:+.3f}]".rjust(w)
    print(f"{label:<27}{f('R1')}{f('R2', 100)}{f('R3')}")
print("  (* = paired 95% CI excludes zero)")

print("\n" + "=" * 96)
print("R5 — the DRIVEN path's clearance violation (exit 3 reads the FIXED 5.0 m ruler)")
print("=" * 96)
print(f"{'arm':<27}{'dR5 @5.0m FIXED':>26}{'level after':>14}{'@arm thresh':>14}")
for label, r in res.items():
    v = r["R5"]
    s = "*" if v["sep"] else " "
    print(f"{label:<27}"
          f"{v['delta']:+.4f}{s}[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]".rjust(27)
          + f"{r['R5_after_fixed']:>14.4f}{(r['R5_after_arm'] or float('nan')):>14.4f}")

print("\n" + "=" * 96)
print("DECOMPOSITION of dR1 by component — the PRIMARY diagnostic, not the headline")
print("=" * 96)
comps = sorted({c for r in res.values() for c in r["R4_after"]})
print(f"{'arm':<27}" + "".join(f"{c[:9]:>11}" for c in comps))
for label, r in res.items():
    print(f"{label:<27}" + "".join(
        f"{r['R4_after'].get(c, 0.0) - r['R4_before'].get(c, 0.0):>+11.4f}"
        for c in comps))

# ---------------------------------------------------------------- the exit
print("\n" + "=" * 96)
print("EXIT — selected mechanically from PREREG_D_SAFE_CAL §4")
print("=" * 96)
A, C, D = res.get("A  w=1  d_safe=2"), res.get("C  w=1  d_safe=5"), res.get("D  floor (w_prox=0)")
REG = res.get("REG deliberate-regression")
verdict, why = None, []

# void conditions first — they override every other reading
if D and any(D[k]["sep"] for k in ("R1", "R2", "R3")):
    verdict = "VOID (exit 5)"
    why.append("arm D carries proximity at WEIGHT 0 and still moved a metric "
               "with paired separation. An inert term cannot change the "
               "objective ⇒ harness failure, not a result.")
if REG and not (REG["R1"]["sep"] or REG["R3"]["sep"]):
    verdict = verdict or "VOID (no regression licence)"
    why.append("the deliberate-regression control did NOT degrade with "
               "separation ⇒ the readout cannot see failure, so no null in this "
               "table is admissible as a measurement.")

if verdict is None:
    a_r2_falls = A["R2"]["sep"] and A["R2"]["delta"] < 0
    c_null = not C["R2"]["sep"]
    a_r5_rises = A["R5"]["sep"] and A["R5"]["delta"] > 0
    prox_move = abs(A["R4_after"].get("proximity", 0.0)
                    - A["R4_before"].get("proximity", 0.0))
    if a_r2_falls and a_r5_rises:
        verdict = "EXIT 3 — REWARD HACKING"
        why.append("fan collisions fell while the SELECTED path's violation rate "
                   "ROSE on the frozen ruler: the scored fan improved and what "
                   "would actually be driven got worse.")
    elif a_r2_falls and c_null:
        verdict = "EXIT 1 — THE THRESHOLD WAS THE BLOCKER"
        why.append("A's fan collision rate fell with paired separation while C "
                   "reproduced the banked null ⇒ recalibration is the difference.")
    elif A["R3"]["sep"] and A["R3"]["delta"] > 0 and not A["R2"]["sep"]:
        verdict = "EXIT 4 — opposition is not threshold-specific"
        why.append("sel-ADE degraded with separation while fan collisions stayed "
                   "flat: the barrier still fights the trust region, only later.")
    elif (not A["R2"]["sep"]) and (not C["R2"]["sep"]) and prox_move < 0.01:
        verdict = "EXIT 2 — THE THRESHOLD WAS NOT THE BLOCKER; THE RL LINE CLOSES"
        why.append(f"A and C both null on R2 and proximity moved {prox_move:.4f} "
                   "(< 0.01) ⇒ decoder-only RL on a 128-anchor fan has no "
                   "headroom on this task.")
    else:
        verdict = "NO COMMITTED BRANCH MATCHES — report as-is, decide nothing"
        why.append("the result does not satisfy any pre-registered exit; ⛔ do "
                   "NOT retro-fit a branch to it.")

print(f"\n  ⇒ {verdict}\n")
for w in why:
    print(f"     · {w}")
json.dump({"arms": {k: {kk: vv for kk, vv in v.items()} for k, v in res.items()},
           "verdict": verdict, "why": why},
          open(O / "dsafe_cal_result.json", "w"), indent=1, default=float)
print(f"\n-> {O}/dsafe_cal_result.json")
