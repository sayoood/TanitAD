"""a7_controls.py -- instrument controls for the cause attribution (EXPLORATORY, MEASURED eval-141).
(1) Suppression: in how many clips does turn-suppression ON change the re-run token vs OFF, and which of the two does the shipped join follow?
(2) CURVE windows: does the re-run (suppression per clip, baseline gate) reproduce the join's FOLLOW token at the nav row, and is the matched segment's vmin clear of the 8 m/s gate (margin)?
(3) The re-run's agreement with the join on the 339 themselves.
"""
import csv
import numpy as np
from nt_common import *

J, K, T, grid = load_inputs()
R = dict(np.load(HERE + "repro_eval141.npz", allow_pickle=True))
rows = list(csv.DictReader(open(HERE + "navturn_windows.csv", encoding="utf-8")))
off, nn = J["off"], J["n"]
tokJ = J["nav"][:, :3].argmax(1)
has = J["has"]
diffclips = []
for c in range(len(nn)):
    o, n = int(off[c]), int(nn[c])
    sl = slice(o, o + n)
    m = has[sl]
    if not m.any():
        continue
    d = (R["tok_on"][sl] != R["tok_off"][sl]) & m
    if d.any():
        diffclips.append((c, int(d.sum()), int((R["tok_on"][sl][d] == tokJ[sl][d]).sum()), int((R["tok_off"][sl][d] == tokJ[sl][d]).sum())))
print(f"(1) clips where suppression ON changes the re-run token on >=1 row: {len(diffclips)} of {int((nn > 0).sum())}")
for c, n, on, offc in diffclips:
    print(f"    clip {str(J['clip'][c])[:8]}: rows that differ {n}; join follows ON on {on}, OFF on {offc}")
cv = [r for r in rows if r["type"] == "A" and r["cause"] == "CURVE"]
rN = np.array([int(r["rowN"]) for r in cv])
print(f"\n(2) CURVE windows n={len(cv)}: re-run token at rowN (suppression per clip) == join token (FOLLOW) in {int((R['nav_token'][rN] == tokJ[rN]).sum())}")
vm = np.array([float(r["vmin"]) for r in cv])
print(f"    matched-segment vmin: min {vm.min():.2f}  p5 {np.percentile(vm, 5):.2f}  windows with vmin in (8.0, 8.3] {int(((vm > 8.0) & (vm <= 8.3)).sum())}  (margin to the gate; pose-vs-log speed agreement is 0.0013 m/s mean, |max| 0.022 per join_check)")
allr = [r for r in rows]
rN = np.array([int(r["rowN"]) for r in allr]); typ = np.array([r["type"] for r in allr])
for t_ in ("A", "B"):
    m = typ == t_
    print(f"(3) {t_} windows: re-run token == join token at rowN in {int((R['nav_token'][rN[m]] == tokJ[rN[m]]).sum())} of {int(m.sum())}")
