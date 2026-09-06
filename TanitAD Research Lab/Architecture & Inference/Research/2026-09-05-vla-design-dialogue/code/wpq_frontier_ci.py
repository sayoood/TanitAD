"""WP-Q — intervals on the frontier, and the knee located.

⛔ WHAT WP-P LEFT UNDONE. Its table is nine point estimates on one set of 1,360
windows. The design conclusion drawn from it -- "the usable window is beta in
[1, 2]" -- rests on three comparisons that were never given an interval:

    1. is ADE(beta=2) really <= ADE(base)?           (0.6583 vs 0.6636)
    2. is ADE(beta=2) really worse than ADE(beta=1)? (0.6583 vs 0.6521)
    3. does the real-vs-cross gap really grow 1 -> 2? (+0.0252 -> +0.0742)

A window recommended to Stage A on three un-intervalled point estimates is a
guess wearing a table. Every one is a PAIRED contrast on the same windows, so the
episode-cluster bootstrap applies directly and costs nothing.

⭐ AND THE GRID IS REFINED WHERE THE ANSWER LIVES. WP-P jumped 2 -> 5 and ADE went
0.6583 -> 1.4864, a 2.3x collapse across one gap in the grid. The knee is somewhere
in between and the design needs to know where, so this sweeps 1.0 ... 4.0 finely.

⛔ CONTROLS, unchanged so the two runs stay comparable:
  * beta = 0 is the port-ablated base policy and every contrast is against it
  * beta = 1 must reproduce the shipped model EXACTLY (asserted)
  * the cross-episode arm is swept at every beta, averaged over draws
  * every interval is a PAIRED episode-cluster bootstrap over the 34 episodes

Tier T0. NON-PARITY pilot. Evidence class MEASURED (ours). Zero GPU.
"""
import io as _io
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "stack")
from tanitad.instruments.cot_influence import consistency, rerank_logits   # noqa: E402

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
d = torch.load(O + "/wpn_port_tensors.pt", map_location="cpu", weights_only=False)
S0, S1, ADE, EPS = d["S0"], d["S1"], d["ADE"], d["EPS"]
E = np.array(EPS)
UE = sorted(set(EPS))
epn = np.array([UE.index(e) for e in EPS])
N, K = S0.shape
energy = -(S1 - S0)
b = torch.arange(N)
sd_s0 = float(S0.std(dim=1).median())
sd_e = float(energy.std(dim=1).median())
assert torch.allclose(rerank_logits(S0, energy, 1.0), S1, atol=1e-5)
BETAS = [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
N_DRAW = 12
IX = {e: np.where(E == e)[0] for e in UE}
print("[wpq] n=%d K=%d episodes=%d | sd[S0] %.3f sd_a[E] %.3f" % (N, K, len(UE),
                                                                  sd_s0, sd_e))


def cross_idx(r):
    idx = np.empty(N, dtype=int)
    for i in range(N):
        j = int(r.integers(0, N))
        k = 0
        while epn[j] == epn[i] and k < 200:
            j = int(r.integers(0, N))
            k += 1
        idx[i] = j
    assert (epn[idx] != epn).all()
    return idx


r = np.random.default_rng(0)
CROSS = [torch.from_numpy(cross_idx(r)) for _ in range(N_DRAW)]


def sel_ade(en, beta):
    s = rerank_logits(S0, en, beta) if beta else S0
    return ADE[b, s.argmax(-1)]


def boot(x, y, iters=4000, seed=0):
    rr = np.random.default_rng(seed)
    dv = (x - y).numpy() if torch.is_tensor(x) else (x - y)
    out = []
    for _ in range(iters):
        pick = np.concatenate([IX[UE[i]] for i in rr.integers(0, len(UE), len(UE))])
        out.append(float(dv[pick].mean()))
    o = np.array(out)
    return float(o.mean()), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


base = sel_ade(energy, 0.0)
b1 = sel_ade(energy, 1.0)
rows = []
print("\n%6s%9s%10s%22s%10s%22s"
      % ("beta", "auth%", "ADE", "ADE - base  CI95", "CONrank", "gap(cross-real) CI95"))
for beta in BETAS:
    a_real = sel_ade(energy, beta)
    a_cross = torch.stack([sel_ade(energy[ix], beta) for ix in CROSS]).mean(0)
    m, lo, hi = boot(a_real, base)
    gm, glo, ghi = boot(a_cross, a_real)
    con = consistency(energy, (rerank_logits(S0, energy, beta) if beta
                               else S0).argmax(-1))
    rows.append({"beta": beta, "auth": 100 * beta * sd_e / sd_s0,
                 "ade": float(a_real.mean()), "d_base": m, "d_base_ci": [lo, hi],
                 "d_base_sep": bool(lo > 0 or hi < 0),
                 "con_rank": con.mean_rank, "gap": gm, "gap_ci": [glo, ghi],
                 "gap_sep": bool(glo > 0 or ghi < 0)})
    print("%6.2f%9.1f%10.4f  %+8.4f [%+.4f,%+.4f]%s%10.2f  %+8.4f [%+.4f,%+.4f]%s"
          % (beta, 100 * beta * sd_e / sd_s0, float(a_real.mean()), m, lo, hi,
             "*" if (lo > 0 or hi < 0) else " ", con.mean_rank, gm, glo, ghi,
             "*" if (glo > 0 or ghi < 0) else " "), flush=True)
print("  (* = interval excludes 0)   chance CON rank = %.1f" % ((K + 1) / 2.0))

print("\n=== THE THREE CLAIMS WP-P MADE WITHOUT INTERVALS ===")
r2 = sel_ade(energy, 2.0)
m, lo, hi = boot(r2, base)
print("1. ADE(beta=2) - ADE(base)     %+.4f [%+.4f,%+.4f]  %s"
      % (m, lo, hi, "SEPARATED" if (lo > 0 or hi < 0) else "overlaps 0"))
m2, lo2, hi2 = boot(r2, b1)
print("2. ADE(beta=2) - ADE(beta=1)   %+.4f [%+.4f,%+.4f]  %s"
      % (m2, lo2, hi2, "SEPARATED" if (lo2 > 0 or hi2 < 0) else "overlaps 0"))
g1 = torch.stack([sel_ade(energy[ix], 1.0) for ix in CROSS]).mean(0) - b1
g2 = torch.stack([sel_ade(energy[ix], 2.0) for ix in CROSS]).mean(0) - r2
m3, lo3, hi3 = boot(g2, g1)
print("3. gap(beta=2) - gap(beta=1)   %+.4f [%+.4f,%+.4f]  %s"
      % (m3, lo3, hi3, "SEPARATED" if (lo3 > 0 or hi3 < 0) else "overlaps 0"))

knee = next((x for x in rows if x["beta"] > 0 and x["d_base_sep"] and x["d_base"] > 0),
            None)
safe = [x for x in rows if x["beta"] > 0 and not (x["d_base_sep"] and x["d_base"] > 0)]
print("\n  KNEE (first beta whose ADE is SEPARATED WORSE than base): %s"
      % ("beta %.2f (auth %.1f %%, ADE %.4f)" % (knee["beta"], knee["auth"], knee["ade"])
         if knee else "none within the swept range"))
print("  SAFE band (ADE not separated-worse than base): beta %s"
      % ", ".join("%.2f" % x["beta"] for x in safe))
best_con = min(safe, key=lambda x: x["con_rank"]) if safe else None
print("  Best consistency INSIDE the safe band: %s"
      % ("beta %.2f -> CON rank %.2f (auth %.1f %%, ADE %.4f)"
         % (best_con["beta"], best_con["con_rank"], best_con["auth"], best_con["ade"])
         if best_con else "n/a"))
verdict = ("STAGE A BETA = %.2f -- the largest authority whose ADE is not separated-worse "
           "than the base policy, giving the best readable consistency in that band"
           % best_con["beta"] if best_con else "no safe band found")
print("\n  => %s" % verdict)

json.dump({"n": N, "K": K, "episodes": len(UE), "sd_s0": sd_s0, "sd_energy": sd_e,
           "n_cross_draws": N_DRAW, "rows": rows,
           "claim_checks": {"ade_b2_minus_base": [m, lo, hi],
                            "ade_b2_minus_b1": [m2, lo2, hi2],
                            "gap_b2_minus_gap_b1": [m3, lo3, hi3]},
           "knee": knee, "safe_betas": [x["beta"] for x in safe],
           "recommended_beta": best_con["beta"] if best_con else None,
           "verdict": verdict,
           "_tier": "T0; NON-PARITY pilot; paired episode-cluster bootstrap 4000 it; "
                    "zero GPU (closed-form on banked tensors)"},
          _io.open(O + "/wpq_frontier_ci.json", "w", encoding="utf-8"), indent=1)
print("\n-> wpq_frontier_ci.json")
