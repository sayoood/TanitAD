"""PAIRED episode-cluster bootstrap over the P-RC21 anchor sweep.

The house rule: for two arms on the SAME windows use the PAIRED bootstrap,
never independent CIs. The first campaign could not do this — its readout kept
aggregates only. `per_episode` is now stored, so the delta each arm produced is
resampled over the SAME 15 episode clusters before and after.
"""
import json, numpy as np

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
ARMS = [("s0-w0", 0.0), ("s1-w0p1", 0.1), ("s2-w1", 1.0), ("s3-w10", 10.0),
        ("sreg", "REG")]
rng = np.random.default_rng(7)
rows = {}
for arm, w in ARMS:
    b = json.load(open(f"{O}/sweep-{arm}/readout_before.json"))["per_episode"]
    a = json.load(open(f"{O}/sweep-{arm}/readout_after.json"))["per_episode"]
    eps = sorted(set(b) & set(a))
    d = {m: np.array([a[e][m] - b[e][m] for e in eps]) for m in ("r", "cr", "ade")}
    out = {"n_ep": len(eps), "w": w}
    for key, name in (("r", "R1"), ("cr", "R2"), ("ade", "R3")):
        boot = np.array([d[key][rng.choice(len(eps), len(eps), True)].mean()
                         for _ in range(4000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = {"delta": float(d[key].mean()), "ci": [float(lo), float(hi)],
                     "sep": bool(lo > 0 or hi < 0)}
    rows[arm] = out

print(f"{'arm':<10}{'w':>6} | {'dR1 (paired)':>26} | {'dR2 collision pp':>28} | "
      f"{'dR3 ADE m':>26}")
for arm, _ in ARMS:
    r = rows[arm]
    f = lambda k, s=1.0, mark=True: (
        f"{r[k]['delta']*s:+.4f} [{r[k]['ci'][0]*s:+.4f},{r[k]['ci'][1]*s:+.4f}]"
        + ("*" if (mark and r[k]["sep"]) else " "))
    print(f"{arm:<10}{str(r['w']):>6} | {f('R1'):>26} | {f('R2',100):>28} | "
          f"{f('R3'):>26}")
print("\n* = paired 95% CI excludes zero (SEPARATED)")
json.dump(rows, open(f"{O}/sweep_paired.json", "w"), indent=1, default=str)

print("\n--- ANCHOR-STRENGTH CURVE (R3 drift vs trust-region weight) ---")
for arm, w in ARMS[:4]:
    r = rows[arm]
    print(f"  w_anchor {str(w):>5}  dADE {r['R3']['delta']:+.4f} m"
          f"  ({100*r['R3']['delta']/0.654:+.1f}% of the 0.654 m baseline)"
          f"  {'SEPARATED' if r['R3']['sep'] else 'not separated'}")
