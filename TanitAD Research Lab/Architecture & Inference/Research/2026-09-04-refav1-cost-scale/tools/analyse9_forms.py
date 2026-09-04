"""Which FORM of the goal term is commensurable with the penalties?

Per form, on the SAME rollouts:
  * the term's own dynamic range over the candidate box
  * the tipping W_KAPPA for a constant |kappa| = 0.1 candidate  (how far the
    shipped 0.05 has to move)
  * the tipping W_JERK for a candidate carrying iCEM's own median jerk
  * whether a REAL iCEM iteration-0 population is still 100 % excluded
"""
import json, sys
import numpy as np, torch
SC = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad"
sys.path.insert(0, r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\repo\stack")
from tanitad.refs.refa_v1_plan import colored_noise, PlanConfig, _clip
from tanitad.refs import refa_v1 as R

J = json.load(open(SC + r"\cost_forms_devbox.json"))
BN, rows, FORMS = J["box_names"], J["rows"], [f for f in J["forms"]]
N = len(rows)
print(f"n = {N} windows (20-episode dev-box slice, step {J['model']['step']}); "
      f"box = {len(BN)} candidates: {BN}")
A = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
kap, jerk = A("kap_raw"), A("jerk_raw")
i_cv = BN.index("cv"); iL, iR = BN.index("kap+0.1"), BN.index("kap-0.1")
kbar = kap[0, iL]
kidx = [i for i, n in enumerate(BN) if n.startswith("kap")]

pc = PlanConfig(horizon=10, dt=0.2, seed=0)
gen = torch.Generator().manual_seed(0)
S = _clip(torch.zeros(pc.horizon, 2) + colored_noise(
    (300, pc.horizon, 2), pc.beta, generator=gen), pc)
icem_jerk = ((S[:, 1:, 0] - S[:, :-1, 0]) / pc.dt).pow(2).mean(-1).numpy()
icem_kap = S[..., 1].pow(2).mean(-1).numpy()

print("\n" + "="*118)
print(f"{'form':9s} {'median over box':>16s} {'p99':>11s} {'max':>10s} | "
      f"{'goal(cv) med':>13s} {'kappa-box spread':>17s} | "
      f"{'tip W_KAPPA':>12s} {'shipped/tip':>12s} | {'iCEM excl. @med':>16s}")
print("="*118)
res = {}
for f in FORMS:
    G = A(f)
    gcv = G[:, i_cv]
    sub = [i_cv] + kidx
    spread = np.median(G[:, sub].max(1) - G[:, sub].min(1))
    # tipping W_KAPPA: best-signed constant |kappa|=0.1 vs cv, per window
    adv = np.minimum(G[:, iL], G[:, iR]) - gcv
    w = -adv / kbar
    good = w > 0
    wt = float(np.median(w[good])) if good.sum() else float("nan")
    pen = R.W_JERK * icem_jerk + R.W_KAPPA * icem_kap
    ex = float((pen >= np.median(gcv)).mean())
    print(f"{f:9s} {np.median(G):16.6g} {np.percentile(G,99):11.4g} "
          f"{G.max():10.4g} | {np.median(gcv):13.4g} {spread:17.6g} | "
          f"{wt:12.4g} {R.W_KAPPA/wt if wt==wt else float('nan'):12.4g} | "
          f"{ex*100:15.2f}%")
    res[f] = {"gcv": gcv, "G": G, "w_tip": wt}

print("\n" + "="*118)
print("WHAT WEIGHTS WOULD THE FORM NEED, so that a REAL iCEM candidate is not "
      "excluded a priori?")
print("="*118)
print(f"iCEM iteration-0: jerk_raw median {np.median(icem_jerk):.4g}, min "
      f"{icem_jerk.min():.4g};  kappa_raw median {np.median(icem_kap):.4g}")
print(f"{'form':9s} {'goal(cv) med':>13s} {'W_JERK cap':>13s} {'shipped/cap':>13s}"
      f"   (cap = goal(cv) / median iCEM jerk_raw)")
for f in FORMS:
    gcv = res[f]["gcv"]
    cap = np.median(gcv) / np.median(icem_jerk)
    print(f"{f:9s} {np.median(gcv):13.4g} {cap:13.4g} {R.W_JERK/cap:13.4g}")

print("\n" + "="*118)
print("DISCRIMINATION — does the form separate a CORRECT turn from a WRONG one?")
print("  (paired within window: the two signs of |kappa| = 0.1; larger |gap| is "
      "more discriminating)")
print("="*118)
for f in FORMS:
    G = res[f]["G"]
    gap = np.abs(G[:, iL] - G[:, iR])
    rel = gap / np.maximum(0.5 * (G[:, iL] + G[:, iR]), 1e-300)
    print(f"  {f:9s} |L-R| median {np.median(gap):12.6g}   RELATIVE to the "
          f"term's own level: {np.median(rel)*100:8.3f}%")
print("\n⭐ the RELATIVE column is the one that matters: it is the fraction of "
      "the term's own\n   magnitude that the L/R decision moves. A form whose "
      "decision is a small relative\n   perturbation of a large constant is the "
      "defect; a form whose decision IS the term\n   is what a planner cost "
      "should look like.")
