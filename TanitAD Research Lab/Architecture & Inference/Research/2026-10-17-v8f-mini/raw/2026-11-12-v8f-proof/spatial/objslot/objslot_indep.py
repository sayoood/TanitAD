# objslot_indep.py : independent recomputation of the LABEL headline of OBJSLOT stage 0 (own rule math; only the loaders of the agent's script are reused).
import sys, os, numpy as np
sys.path.insert(0, "F:/TanitAD-scratch/v8proof/spatial/objslot")
import objslot_s0 as O
D, lat = O.load_arm("P_s0")
eid, g0, fold, folds_b = O.arm_windows(D)
G, miss = O.load_label_frames(eid, g0)
V, _, _ = O.speed_and_states("P_s0", eid, g0)
print("G", G.shape, "V", V.shape, "miss", miss)
n = len(G); y = G[:, 14]; cp = G[:, 4]
pred1 = np.full(n, np.nan); pred2 = np.full(n, np.nan); use = np.zeros(n, bool)
for i in range(n):
    seg = G[i, 4:15]
    if not np.isfinite(seg).all() or np.max(np.abs(np.diff(seg))) > 3.0:
        continue
    fr = [4]
    j = 3
    while j >= 0 and np.isfinite(G[i, j]) and abs(G[i, j + 1] - G[i, j]) <= 3.0:
        fr.append(j); j -= 1
    if len(fr) < 3:
        continue
    fr = sorted(fr)
    c = np.polyfit(np.array(fr) * 0.2, G[i, fr], 1)[0]
    vL = V[i, 4] + c
    travel = sum(0.2 * 0.5 * (V[i, 4 + k] + V[i, 5 + k]) for k in range(10))
    pred1[i] = G[i, 4] + 2.0 * c
    pred2[i] = G[i, 4] + 2.0 * vL - travel
    use[i] = True
sse = lambda p: float(np.sum((p[use] - y[use]) ** 2))
print("n used", int(use.sum()), "episodes", len(set(eid[use])))
print("R1 skill %.4f  R2 skill %.4f" % (1 - sse(pred1) / sse(cp), 1 - sse(pred2) / sse(cp)))
r2 = lambda p: 1 - sse(p) / float(np.sum((y[use] - y[use].mean()) ** 2))
print("level R2: copy %.4f  R1 %.4f  R2 %.4f" % (r2(cp), r2(pred1), r2(pred2)))
print("rmse m: copy %.3f  R2 %.3f ; mean |gap change| %.3f m" % (np.sqrt(sse(cp) / use.sum()), np.sqrt(sse(pred2) / use.sum()), float(np.mean(np.abs(y[use] - cp[use])))))
