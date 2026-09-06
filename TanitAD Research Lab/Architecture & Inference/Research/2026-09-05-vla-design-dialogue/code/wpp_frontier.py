"""WP-P — the INFLUENCE / CONSISTENCY FRONTIER, measured on a real model.

⭐ THIS IS THE PI'S QUESTION, TURNED INTO A CURVE. The requirement was a CoT that
INFLUENCES behaviour while staying CONSISTENT with the selected trajectory. The
frontier literature's verdict on that pair is verbatim *"Neither design satisfies
both desiderata"* (arXiv 2606.12706) -- but it compares TWO DISCRETE MODELS. The
Energy Bridge has a CONTINUOUS DIAL, `beta`, so the trade-off can be traced rather
than argued about. Nobody has published that curve; this is the cheapest possible
version of it.

⭐ AND IT COSTS NO GPU. `S1 = S0 - beta * E` is closed-form in the banked tensors,
so a whole beta sweep is arithmetic on arrays already on disk.

⛔ THE CONSTRAINT THIS EXISTS TO QUANTIFY. WP-M measured REF-C's trained port at
sd_a[Delta] = 1.457 against sd[S0] = 17.907 -- about **8 %** of the scorer's scale
-- and at that ratio `consistency()` reads mean rank 64.40 of 128 against a chance
of 64.50, i.e. NOTHING. That is not a fact about REF-C; it is arithmetic: a small
additive prior cannot rank the chosen anchor first however much it knows. The
design consequence was stated but never quantified: **how large must beta be
before consistency is measurable at all, and what does the metric cost there?**

WHAT IS SWEPT, and what each column means:

    beta            the guidance weight; beta = 1 reproduces the shipped model
    scale%          beta * sd_a[Delta] / sd[S0] -- the port's authority over the
                    decision, the quantity the constraint is stated in
    INF flip / kl   influence: how much the decision moved from the base policy
    CON rank        consistency: where the chosen anchor sits under the energy
                    (1 = the energy's own favourite; 64.5 = chance at K=128)
    ADE             what the selection actually achieved
    ADE cross-ep    the SAME beta with a prior from a DIFFERENT EPISODE

⛔ THE CROSS-EPISODE ARM IS SWEPT TOO, AND THAT IS THE POINT. Influence and
consistency both rise trivially with beta -- a large enough weight makes ANY
energy dominate the decision and rank its own favourite first. What must NOT rise
trivially is the gap between a matched prior and a mismatched one. If real and
cross-episode converge as beta grows, the extra authority is buying obedience
rather than understanding, and the frontier is being climbed for nothing.

Tier T0. NON-PARITY pilot. Evidence class MEASURED (ours).
"""
import io as _io
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "stack")
from tanitad.instruments.cot_influence import (                          # noqa: E402
    consistency, influence, rerank_logits,
)

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
BANK = O + "/wpn_port_tensors.pt"
N_DRAW = 12
BETAS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]

d = torch.load(BANK, map_location="cpu", weights_only=False)
S0, S1, ADE, EPS = d["S0"], d["S1"], d["ADE"], d["EPS"]
E = np.array(EPS)
UE = sorted(set(EPS))
epn = np.array([UE.index(e) for e in EPS])
N, K = S0.shape
energy = -(S1 - S0)
sd_s0 = float(S0.std(dim=1).median())
sd_e = float(energy.std(dim=1).median())
b = torch.arange(N)
print("[wpp] n=%d  K=%d  episodes=%d" % (N, K, len(UE)))
print("[wpp] sd[S0] %.4f   sd_a[E] %.4f   shipped scale %.1f %%"
      % (sd_s0, sd_e, 100 * sd_e / sd_s0))
assert torch.allclose(rerank_logits(S0, energy, 1.0), S1, atol=1e-5), \
    "beta=1 must reproduce the shipped model, or the sweep is about another object"


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

print("\n%6s%9s%9s%10s%10s%9s%11s%12s"
      % ("beta", "scale%", "INF flip", "INF kl", "CON rank", "CON top1",
         "ADE", "ADE cross"))
rows = []
for beta in BETAS:
    s1 = rerank_logits(S0, energy, beta) if beta else S0.clone()
    sel = s1.argmax(-1)
    inf = influence(S0, s1)
    con = consistency(energy, sel)
    ade = float(ADE[b, sel].mean())
    ade_x = float(torch.stack(
        [ADE[b, (rerank_logits(S0, energy[ix], beta) if beta else S0).argmax(-1)]
         for ix in CROSS]).mean())
    scale = 100 * beta * sd_e / sd_s0
    rows.append({"beta": beta, "scale_pct": scale, "flip": inf.flip_rate,
                 "kl": inf.kl, "con_rank": con.mean_rank, "con_top1": con.top1_rate,
                 "ade": ade, "ade_cross": ade_x, "gap": ade_x - ade})
    print("%6.1f%9.1f%9.4f%10.4f%10.2f%9.4f%11.4f%12.4f"
          % (beta, scale, inf.flip_rate, inf.kl, con.mean_rank, con.top1_rate,
             ade, ade_x))

chance = (K + 1) / 2.0
print("\n  chance CON rank = %.1f  (K=%d).  Lower is more consistent." % (chance, K))

# where does consistency become measurable at all?
meas = next((x for x in rows if x["con_rank"] < 0.9 * chance), None)
print("\n  CONSISTENCY BECOMES MEASURABLE at beta = %s"
      % ("%.1f (scale %.1f %%, ADE %.4f m)"
         % (meas["beta"], meas["scale_pct"], meas["ade"]) if meas else
         "NEVER within the swept range"))
best = min(rows, key=lambda x: x["ade"])
print("  BEST ADE in the sweep: beta %.1f (scale %.1f %%) -> %.4f m  "
      "[shipped beta=1.0 -> %.4f m]"
      % (best["beta"], best["scale_pct"], best["ade"],
         [x for x in rows if x["beta"] == 1.0][0]["ade"]))

print("\n  THE CONTROL THAT DECIDES WHETHER THE FRONTIER IS WORTH CLIMBING:")
print("  real-vs-cross gap must GROW with beta; if it collapses, the extra")
print("  authority is buying obedience, not understanding.")
for x in rows:
    print("    beta %6.1f  gap %+.4f m" % (x["beta"], x["gap"]))
# !! MAX-OVER-A-NOISY-SEQUENCE IS NOT A TREND. Version 1 took max(gap) over the
# whole sweep and compared it to the gap at beta=1 -- which picks the largest
# positive FLUCTUATION. The gap oscillates in SIGN above beta=5 (+0.149, -0.097,
# +0.429, -0.150, +0.039), because by then ADE has degraded 2-11x and both arms
# are wrecked; a sign-oscillating series has no trend to extract. Same class as
# selecting a hyper-parameter on the scored split.
# => the trend is only read in the USABLE regime, where ADE is still <= base.
g1 = [x for x in rows if x["beta"] == 1.0][0]["gap"]
base_ade = [x for x in rows if x["beta"] == 0.0][0]["ade"]
usable = [x for x in rows if x["ade"] <= base_ade and x["beta"] > 0]
signs = {1 if x["gap"] > 0 else -1 for x in rows if x["beta"] >= 5.0}
gmax = max(usable, key=lambda x: x["gap"]) if usable else rows[0]
print("  usable regime (ADE <= base): beta " +
      ", ".join("%.1f" % x["beta"] for x in usable))
print("  gap sign above beta=5 oscillates: %s -> no trend is readable there"
      % ("YES" if len(signs) > 1 else "no"))
print("  gap at shipped beta=1: %+.4f m | largest gap %+.4f m at beta %.1f"
      % (g1, gmax["gap"], gmax["beta"]))
verdict = ("IN THE USABLE REGIME the gap GROWS with authority (%+.4f at beta 1 -> "
           "%+.4f at beta %.1f), but ADE collapses before consistency becomes good, "
           "so raising beta ALONE cannot deliver both -- the energy must get BETTER, "
           "not louder" % (g1, gmax["gap"], gmax["beta"])
           if usable and gmax["gap"] > g1 else
           "THE GAP DOES NOT GROW in the usable regime - extra authority buys "
           "obedience, not understanding")
print("\n  => %s" % verdict)

json.dump({"n": N, "K": K, "episodes": len(UE), "sd_s0": sd_s0, "sd_energy": sd_e,
           "shipped_scale_pct": 100 * sd_e / sd_s0, "chance_rank": chance,
           "n_cross_draws": N_DRAW, "rows": rows,
           "consistency_measurable_at": meas, "best_ade": best, "verdict": verdict,
           "_tier": "T0; NON-PARITY pilot; closed-form beta sweep on banked "
                    "tensors; cross-episode arm averaged over %d draws" % N_DRAW,
           "_note": "beta=1.0 reproduces the shipped refc-base-30k model exactly "
                    "(asserted); beta=0 is the port-ablated base policy."},
          _io.open(O + "/wpp_frontier.json", "w", encoding="utf-8"), indent=1)
print("\n-> wpp_frontier.json")
