"""AMENDMENT 2 readout — paired CIs + the per-component decomposition.

The amendment makes the DECOMPOSITION the primary diagnostic, not the headline
delta: "a headline delta without the decomposition is not an admissible readout
for this arm". This script therefore prints the decomposition first and decides
the pre-committed exit (A/B/C/D) mechanically from it.
"""
import json, sys
import numpy as np

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
ARMS = [("rerun-t1-w1-prox", "T1 w=1 +prox"), ("rerun-t2-w10-prox", "T2 w=10 +prox"),
        ("rerun-treg", "T-reg hackable")]
CTX = [("sweep-s0-w0", "ctx w=0 (prev)"), ("sweep-s2-w1", "prev w=1 no-prox"),
       ("sweep-s3-w10", "prev w=10 no-prox")]
rng = np.random.default_rng(7)


def paired(arm):
    b = json.load(open(f"{O}/{arm}/readout_before.json"))
    a = json.load(open(f"{O}/{arm}/readout_after.json"))
    eb, ea = b["per_episode"], a["per_episode"]
    eps = sorted(set(eb) & set(ea))
    out = {}
    for key, name in (("r", "R1"), ("cr", "R2"), ("ade", "R3")):
        d = np.array([ea[e][key] - eb[e][key] for e in eps])
        boot = np.array([d[rng.choice(len(eps), len(eps), True)].mean()
                         for _ in range(4000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = (float(d.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0))
    # ⚠️ The readout measures R1 with the DEFAULT spec for cross-arm
    # comparability, so R4_component_means covers only the default components.
    # Decompose over the INTERSECTION and report the gap explicitly rather than
    # KeyError-ing or silently dropping terms.
    w = json.load(open(f"{O}/{arm}/config.json"))["reward_weights"]
    meas = set(a["R4_component_means"]) & set(b["R4_component_means"])
    from tanitad.rl.rewards import DEFAULT_WEIGHTS as DW
    comp = {c: DW.get(c, w.get(c, 0.0)) *
               (a["R4_component_means"][c] - b["R4_component_means"][c])
            for c in sorted(meas)}
    comp["_not_in_R1"] = sorted(set(w) - meas)
    return out, comp, len(eps)


print("=" * 78)
print("PER-COMPONENT dR1 DECOMPOSITION (weighted) — the PRIMARY diagnostic")
print("=" * 78)
allc = {}
for arm, label in ARMS + CTX:
    try:
        _, comp, _ = paired(arm)
    except Exception as e:
        print(f"{label:<18} unavailable ({type(e).__name__})"); continue
    allc[label] = comp
keys = sorted({k for c in allc.values() for k in c if not k.startswith("_")})
print(f"{'arm':<18}" + "".join(f"{k[:9]:>11}" for k in keys) + f"{'SUM':>11}")
for label, comp in allc.items():
    vals = {k: v for k, v in comp.items() if not k.startswith("_")}
    print(f"{label:<18}" + "".join(f"{vals[k]:>+11.4f}" if k in vals else f"{'--':>11}"
                                   for k in keys) + f"{sum(vals.values()):>+11.4f}"
          + (f"   [not in R1: {','.join(comp['_not_in_R1'])}]"
             if comp.get("_not_in_R1") else ""))

print("\n" + "=" * 78)
print("PAIRED DELTAS (4000 reps, same clusters before/after; * = CI excludes 0)")
print("=" * 78)
res = {}
for arm, label in ARMS:
    try:
        o, comp, n = paired(arm)
    except Exception as e:
        print(f"{label:<18} unavailable"); continue
    res[label] = (o, comp)
    f = lambda k, s=1.0: (f"{o[k][0]*s:+.4f}[{o[k][1]*s:+.4f},{o[k][2]*s:+.4f}]"
                          + ("*" if o[k][3] else " "))
    print(f"{label:<18} n={n}  R1 {f('R1'):>26}  R2pp {f('R2',100):>26}  "
          f"R3 {f('R3'):>26}")

print("\n" + "=" * 78)
print("PRE-COMMITTED EXIT (AMENDMENT 2) — decided from the decomposition")
print("=" * 78)
for label in ("T1 w=1 +prox", "T2 w=10 +prox"):
    if label not in res:
        continue
    o, comp = res[label]
    pos = {k: v for k, v in comp.items() if not k.startswith("_") and v > 1e-9}
    tot = sum(v for k, v in comp.items() if not k.startswith("_"))
    feas = comp.get("feasibility", 0.0)
    prox = comp.get("proximity", 0.0)
    share = (feas / tot * 100) if abs(tot) > 1e-9 else float("nan")
    print(f"\n{label}:")
    print(f"   total dR1 {tot:+.4f} | feasibility {feas:+.4f} "
          f"({share:.0f}% of total) | proximity {prox:+.4f}")
    print(f"   positive components: {pos if pos else 'NONE'}")
    if abs(tot) < 0.01:
        print("   -> EXIT C: total dR1 ~ 0; the objective is flat inside the "
              "trust region. Lever is the FAN, not the scoring.")
    elif feas > 0 and feas >= 0.5 * abs(tot):
        print("   -> EXIT A: feasibility STILL dominates => make feasibility "
              "RELATIVE to the cold-start fan.")
    elif abs(prox) > abs(feas):
        print("   -> EXIT B: proximity outranks feasibility => go to the R2 "
              "question (does fan collision separate?).")
    else:
        print("   -> MIXED: report both, do not force a branch.")
json.dump({k: {"paired": v[0], "components": v[1]} for k, v in res.items()},
          open(f"{O}/rerun_paired.json", "w"), indent=1, default=str)
print(f"\n-> {O}/rerun_paired.json")
