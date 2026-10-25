"""n1_counts_and_tickrow.py -- (a) reproduce the banked 210 / 129 counts; (b) re-pair them on the row T ACTUALLY READS (EXPLORATORY, MEASURED eval-141).

The banked s2c_routesilent.py takes nav_tok from join row rowN = off + t (the NOW row), but T reads its nav (and is scored against its labels)
at the T-tick row rowT = off + t - a_T (v8f_mini.py:run_t `i = last_t`, v8f_mini_data.py:item `rows[1]`; a_T in {0,1,2} cache steps = 0 / 0.2 / 0.4 s).
Also checks whether the 'route-silent' FOLLOW token really leaves the nav FEATURE vector silent (nav[3] = d_next/100, nav[4] = args_valid).
"""
import numpy as np
from nt_common import *

J, K, T, grid = load_inputs()
ci, tt, a_t, rowT = T["ci"], T["tt"], T["a_t"], T["rowT"]
N = len(ci)
rowN = J["off"][ci] + tt
has = T["has"]
G = T["gt_std"]
turn = np.isfinite(G[:, 1]) & has                       # banked definition: a turn-label cell present at the T-tick row

tokN = J["nav"][rowN][:, :3].argmax(1)                  # banked: NOW row
tokT = J["nav"][rowT][:, :3].argmax(1)                  # what T reads
print(f"windows {N}; with v9 row {int(has.sum())}; turn-label windows (turn_t_start present at rowT) {int(turn.sum())}")
print(f"a_T counts {np.bincount(a_t)}  (a_T=0 -> rowT == rowN)")
print(f"token at rowN != token at rowT: {int((tokN != tokT).sum())} windows  (of {int((a_t > 0).sum())} with a_T>0)")

# --- (a) the banked numbers
b210 = int((turn & (tokN == 0)).sum())
nav_turn_N = (tokN != 0) & has
b129 = int((nav_turn_N & ~np.isfinite(G[:, 1])).sum())
print(f"\n(a) BANKED pairing (nav at rowN, label at rowT): turn-label & nav FOLLOW = {b210}   nav TURN & no turn label = {b129}   total {b210 + b129}")
print(f"    nav TURN windows (rowN, with v9 row) {int(nav_turn_N.sum())}; of which with a turn label {int((nav_turn_N & np.isfinite(G[:, 1])).sum())}")

# --- (b) T-tick pairing (what the model reads)
nav_turn_T = (tokT != 0) & has
c210 = int((turn & (tokT == 0)).sum())
c129 = int((nav_turn_T & ~np.isfinite(G[:, 1])).sum())
print(f"\n(b) T-TICK pairing (nav and label both at rowT): turn-label & nav FOLLOW = {c210}   nav TURN & no turn label = {c129}   total {c210 + c129}")
print(f"    nav TURN windows (rowT) {int(nav_turn_T.sum())}; with a turn label {int((nav_turn_T & np.isfinite(G[:, 1])).sum())}")
sameN = turn & (tokN == 0)
sameT = turn & (tokT == 0)
print(f"    set overlap of the 210-type sets: both {int((sameN & sameT).sum())}  banked-only {int((sameN & ~sameT).sum())}  tick-only {int((~sameN & sameT).sum())}")
sN = nav_turn_N & ~np.isfinite(G[:, 1])
sT = nav_turn_T & ~np.isfinite(G[:, 1])
print(f"    set overlap of the 129-type sets: both {int((sN & sT).sum())}  banked-only {int((sN & ~sT).sum())}  tick-only {int((~sN & sT).sum())}")

# --- (c) is the 'route-silent' nav FEATURE vector silent?  nav[3] = d_next/100 clipped, nav[4] = args_valid
navT = J["nav"][rowT]
sil = turn & (tokT == 0)
print(f"\n(c) in the {int(sil.sum())} turn-label windows with nav token FOLLOW (T-tick row): nav args_valid=1 in {int((navT[sil, 4] == 1).sum())} "
      f"({(navT[sil, 4] == 1).mean():.1%}); d_next median (valid only) {np.median(navT[sil & (navT[:, 4] == 1), 3] * 100):.1f} m")
allf = has & (tokT == 0)
print(f"    all FOLLOW windows at rowT: {int(allf.sum())}; args_valid=1 in {int((navT[allf, 4] == 1).sum())} ({(navT[allf, 4] == 1).mean():.1%})")
# the label's own turn distance vs the nav distance, where both exist
both = sil & (navT[:, 4] == 1)
d_lab = G[both, 3] * T["sd"][3] + T["mu"][3]
d_nav = navT[both, 3] * 100
print(f"    in those with a valid nav distance: label turn_d_start_m median {np.median(d_lab):.1f}, nav d_next median {np.median(d_nav):.1f}; "
      f"nav d_next at clip ceiling 300 m: {int((d_nav >= 299.9).sum())}")
