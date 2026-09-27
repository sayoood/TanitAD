"""Per-slot optimal presence probability p* under each published presence/classification loss.

A slot whose features imply it is matched to a target with probability pi (the Bayes posterior
the head can reach) minimises the EXPECTED per-slot loss E[L] = pi*L(p,1) + (1-pi)*L(p,0).
We report p*(pi) and the pi at which p* crosses a decision gate, for:
  ours   : BCE, positive weight 1.0, negative weight NO_OBJECT_W = 0.1   (agent_slots.py:233, :864-873)
  bce    : unweighted BCE (the calibrated reference: p* = pi)
  detr   : softmax CE with eos_coef 0.1 on the no-object class, collapsed to object-vs-empty
           (detr.py empty_weight; identical per-slot optimum to `ours`)
  focal  : sigmoid focal loss, alpha 0.25, gamma 2 (Deformable DETR / DETR3D / PETR / BEVFormer /
           StreamPETR / UniAD configs)
  focal_a50: focal with alpha 0.5, gamma 2 (isolates the gamma effect)
Pure arithmetic: a 1e-6 grid over p in (0,1). No data, no model.
"""
import json, math, sys
import numpy as np

p = np.linspace(1e-6, 1 - 1e-6, 1_000_001)
lp, l1p = -np.log(p), -np.log(1 - p)

def loss_curves(name):
    if name == "ours":
        return lp, 0.1 * l1p
    if name == "bce":
        return lp, l1p
    if name == "detr":
        return lp, 0.1 * l1p
    if name == "focal":
        a, g = 0.25, 2.0
        return a * (1 - p) ** g * lp, (1 - a) * p ** g * l1p
    if name == "focal_a50":
        a, g = 0.5, 2.0
        return a * (1 - p) ** g * lp, (1 - a) * p ** g * l1p
    raise KeyError(name)

def p_star(name, pi):
    pos, neg = loss_curves(name)
    e = pi * pos + (1 - pi) * neg
    return float(p[int(np.argmin(e))])

def pi_for_gate(name, gate, lo=1e-4, hi=1 - 1e-4):
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if p_star(name, mid) < gate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

names = ["bce", "ours", "detr", "focal", "focal_a50"]
pis = [0.01, 0.02, 0.05, 0.091, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
out = {"method": __doc__.strip().splitlines()[0],
       "p_star": {n: {f"{pi:.3f}": round(p_star(n, pi), 4) for pi in pis} for n in names},
       "pi_at_gate": {n: {str(g): round(pi_for_gate(n, g), 4) for g in (0.2, 0.3, 0.35, 0.4, 0.5)}
                      for n in names},
       "closed_form_check_ours": {f"{pi:.3f}": round(pi / (pi + 0.1 * (1 - pi)), 4) for pi in pis}}
# exchangeable slots: K targets spread over 100 slots -> pi = K/100
out["exchangeable_100_slots"] = {n: {str(K): round(p_star(n, K / 100.0), 4)
                                     for K in (2.65, 4.2, 5, 10, 21, 24.5, 50, 80.3)} for n in names}
json.dump(out, sys.stdout, indent=1)
