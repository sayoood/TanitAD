"""Conditioning of the centred forms, and WHY scale-normalisation alone fails.

`boxnorm`/`goalnorm` normalise the DENOMINATOR but leave the common mode in the
NUMERATOR: ||(x - ref) - (g - ref)|| == ||x - g||, so centring changes an L2
DISTANCE by exactly nothing. Only the DIRECTION comparison in centred space
(`ccos`/`cchord`) removes it. This script measures that decomposition, and the
price the centred forms pay.
"""
import json
import numpy as np
SC = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad"
J = json.load(open(SC + r"\cost_forms_devbox.json"))
BN, rows = J["box_names"], J["rows"]
N = len(rows)
A = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
RESP, GR, RN = A("_resp"), A("_goalresp")[:, 0], A("_refnorm")[:, 0]
i_cv = BN.index("cv"); iL, iR = BN.index("kap+0.1"), BN.index("kap-0.1")

print("="*92)
print("HOW MUCH ACTION-INDUCED CHANGE IS THERE TO NORMALISE BY?  (n=%d)" % N)
print("="*92)
print(f"  ||x_cv||                    median {np.median(RN):.4g}")
print(f"  ||x_i - x_cv|| over the box median {np.median(RESP):.4g}  "
      f"= {np.median(RESP / RN[:, None])*100:.3f} % of ||x_cv||")
print(f"  ||g   - x_cv||              median {np.median(GR):.4g}  "
      f"= {np.median(GR / RN)*100:.3f} % of ||x_cv||")
deg = GR < 1e-3 * RN
print(f"  windows where the GOAL carries essentially NO action-induced change")
print(f"     (||g - x_cv|| < 1e-3 ||x_cv||): {int(deg.sum())}/{N} "
      f"= {deg.mean()*100:.1f} %")

print("\n" + "="*92)
print("WHY SCALE-NORMALISATION ALONE DOES NOT WORK")
print("="*92)
print("  ||(x - x_cv) - (g - x_cv)|| == ||x - g||  EXACTLY, so centring changes")
print("  an L2 DISTANCE by nothing at all. That is why `boxnorm` / `goalnorm` /")
print("  `abs_l2` / `rel_l2` all read the SAME 0.811 % L/R decision share as each")
print("  other — they differ only by a per-window constant. The common mode lives")
print("  in the NUMERATOR and only a DIRECTION comparison removes it:")
for f in ("abs_l2", "rel_l2", "boxnorm", "goalnorm"):
    G = A(f)
    rel = np.abs(G[:, iL] - G[:, iR]) / (0.5 * (G[:, iL] + G[:, iR]))
    print(f"     {f:9s} L/R share {np.median(rel)*100:7.3f} %")
for f in ("cos", "chord", "ccos", "cchord"):
    G = A(f)
    rel = np.abs(G[:, iL] - G[:, iR]) / (0.5 * (G[:, iL] + G[:, iR]))
    print(f"     {f:9s} L/R share {np.median(rel)*100:7.3f} %"
          f"{'   <- DIRECTION in CENTRED space' if f.startswith('c') and f != 'chord' else ''}")

print("\n" + "="*92)
print("THE PRICE OF THE CENTRED FORMS — state it before recommending them")
print("="*92)
cc = A("ccos")
print(f"  `ccos`(cv) is EXACTLY {np.unique(np.round(cc[:, i_cv], 12))} on all "
      f"{N} windows — the do-nothing candidate scores the worst possible value")
print(f"  BY DEFINITION (its centred vector is the zero vector), not by "
      f"measurement.")
print(f"  On a HOLD goal the goal's own centred vector is near zero too, so the")
print(f"  comparison degenerates to noise; that stratum is "
      f"{deg.mean()*100:.1f} % here but 86.5 % of the")
print(f"  full 282-window grid by decoded token (LANE_KEEP 244/282).")
print(f"  ⇒ a centred form must carry an explicit HOLD branch (e.g. gate on")
print(f"    ||g - x_cv|| and fall back to a distance when the goal is 'hold'),")
print(f"    or it trades one degeneracy for another. THIS IS A PRE-REGISTRATION")
print(f"    ITEM, not something to ship from this document.")
