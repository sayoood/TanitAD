"""a4_sweep.py -- MUTATION test of the cause attribution: change ONE literal of the mirror's own v9 builder, re-run it on the eval-141 pose tracks, and count what happens to the 339 (EXPLORATORY, ESTIMATED where marked).
Levers: (G) the is_turn speed gate TURN_MAX_VMIN_MS (s2_geom_emit_v7.py:89, default 8.0 m/s) and (H) the nav announcement horizon nav_h_s (build_v9_labels.py:44, default 6.0 s; D_ann = max(30 m, nav_h_s * v_now)).
The turn-label cells do not depend on either lever (the dominant segment is picked without is_turn; build_v9_labels.py:819-826), so only the nav token moves; lat_cls_a (is_junc) also moves with G.
Reconstruction is exact where it matches the join (r1_repro.py); a window whose baseline re-run token differs from the join keeps the JOIN token under every lever (so changes in unreproducible tail windows are UNDER-counted).
"""
import sys
import numpy as np
from nt_common import *
import build_v9_labels as B
S2 = sys.modules["s2_geom_emit_v7"] if "s2_geom_emit_v7" in sys.modules else __import__("s2_geom_emit_v7")

J, K, T, grid = load_inputs()
R0 = dict(np.load(HERE + "repro_eval141.npz", allow_pickle=True))
P_all = clip_tracks(K)
off, nn = J["off"], J["n"]
NJ = int(nn.sum())
ci, tt, rowT, hasw = T["ci"], T["tt"], T["rowT"], T["has"]
rowN = off[ci] + tt
tokJ = J["nav"][:, :3].argmax(1)
G = T["gt_std"]
labT = np.isfinite(G[:, 1]) & hasw


def build_log(P, ns):
    n_raw = len(P)
    fin = np.where(np.isfinite(ns))[0]
    dt = float(np.median(np.diff(ns[fin]) / np.diff(fin)) / 2.0)
    g0 = float(ns[fin[0]] - 2 * fin[0] * dt)
    ts = g0 + np.arange(n_raw) * dt
    f = -ts[0] / (ts[1] - ts[0])
    x0 = P[0, 0] + f * (P[1, 0] - P[0, 0]); y0 = P[0, 1] + f * (P[1, 1] - P[0, 1])
    yw0 = P[0, 2] + f * (P[1, 2] - P[0, 2]); v0 = P[0, 3] + f * (P[1, 3] - P[0, 3])
    ts_a = np.concatenate([[0.0], ts]); X = np.concatenate([[x0], P[:, 0]]); Y = np.concatenate([[y0], P[:, 1]])
    YW = np.concatenate([[yw0], P[:, 2]]); V = np.concatenate([[v0], P[:, 3]])
    vx, vy = np.gradient(X, ts_a), np.gradient(Y, ts_a)
    rev = ((np.hypot(vx, vy) > 0.3) & (np.cos(np.arctan2(vy, vx) - YW) < 0.0)).astype(np.float64)
    return B.log_from_arrays(ts_a, X, Y, np.unwrap(YW), V, None, rev)


def run(gate, nav_h):
    S2.TURN_MAX_VMIN_MS = gate
    B.LITERALS["nav_h_s"] = nav_h
    tok = np.full(NJ, np.nan)
    latc = np.full(NJ, np.nan)
    for c in range(len(nn)):
        o, n = int(off[c]), int(nn[c])
        if not J["has"][o:o + n].any():
            continue
        ns = J["now_s"][o:o + n]
        log = build_log(P_all[c], ns)
        jj = np.where(np.isfinite(ns))[0]
        jj = jj[jj >= 1]
        sup = bool(R0["sup_on"][o + jj[0]])
        G_ = B.clip_geo(log, {"applied": True} if sup else None)
        F = B.clip_labels(G_, ns[jj], 2 * jj, None, None, None)
        tok[o + jj] = F["nav_token"]
        latc[o + jj] = F["lat_cls_a"]
    return tok, latc


base_tok, base_lat = run(8.0, 6.0)
ok_base = np.isfinite(base_tok) & (base_tok == tokJ)           # rows where the baseline re-run reproduces the join token
print(f"baseline re-run token == join token on {int(ok_base[J['has']].sum())} of {int(J['has'].sum())} join rows with a v9 row")


def adj(tok):
    out = tokJ.copy()
    m = ok_base & np.isfinite(tok)
    out[m] = tok[m].astype(int)
    return out


def counts(tok_rows, label):
    t = tok_rows[rowN]
    A = labT & (t == 0)
    Bm = hasw & (t != 0) & ~labT
    agree = labT & (t != 0)
    return A, Bm, agree


A0, B0, ag0 = counts(tokJ, "join")
print(f"join (banked): A {int(A0.sum())}  B {int(B0.sum())}  agree {int(ag0.sum())}   total disagree {int(A0.sum() + B0.sum())}")
cause = {}
import csv
for r in csv.DictReader(open(HERE + "navturn_windows.csv", encoding="utf-8")):
    cause[int(r["w"])] = (r["type"], r["cause"])
cA = np.array([cause.get(w, ("", ""))[1] if cause.get(w, ("", ""))[0] == "A" else "" for w in range(len(ci))], dtype=object)
cB = np.array([cause.get(w, ("", ""))[1] if cause.get(w, ("", ""))[0] == "B" else "" for w in range(len(ci))], dtype=object)

print("\n(G) is_turn SPEED GATE (m/s), nav horizon fixed at 6 s:")
print(f"{'gate':>8s} {'A':>4s} {'B':>4s} {'agree':>5s} {'sum':>4s} | fixed of A-CURVE(156) {'':s} fixed of A-total | NEW B (were agree/none) | lat_cls TURN windows (grid)")
for gate in (8.0, 10.0, 12.0, 15.0, 1e9):
    tok, lat = run(gate, 6.0)
    A, Bm, ag = counts(adj(tok), "")
    fixedC = int(((cA == "CURVE") & ~A).sum())
    fixedA = int((A0 & ~A).sum())
    newB = int((Bm & ~B0).sum())
    nturn = int(np.isin(lat[rowT][hasw & np.isfinite(lat[rowT])], [1, 2]).sum())
    print(f"{gate:8.0f} {int(A.sum()):4d} {int(Bm.sum()):4d} {int(ag.sum()):5d} {int(A.sum() + Bm.sum()):4d} | {fixedC:4d} of {int((cA == 'CURVE').sum())}            {fixedA:4d} of {int(A0.sum())}      | {newB:4d}                     | {nturn}")

print("\n(H) nav HORIZON nav_h_s (s): D_ann = max(30 m, nav_h_s * v_now), gate fixed at 8 m/s:")
print(f"{'nav_h_s':>8s} {'A':>4s} {'B':>4s} {'agree':>5s} {'sum':>4s} | fixed of A-HORIZON(16) | fixed of B-HORIZON(50)+BAND_EDGE(4) | NEW A (were agree/B)  NEW B")
for h in (6.0, 8.0, 10.0, 12.0):
    tok, lat = run(8.0, h)
    A, Bm, ag = counts(adj(tok), "")
    fixH = int(((cA == "HORIZON") & ~A).sum())
    fixB = int((np.isin(cB, ["HORIZON", "BAND_EDGE"]) & ~Bm).sum())
    print(f"{h:8.1f} {int(A.sum()):4d} {int(Bm.sum()):4d} {int(ag.sum()):5d} {int(A.sum() + Bm.sum()):4d} | {fixH:4d} of {int((cA == 'HORIZON').sum())}            | {fixB:4d} of {int(np.isin(cB, ['HORIZON', 'BAND_EDGE']).sum())}                        | newA {int((A & ~A0).sum())}  newB {int((Bm & ~B0).sum())}")

print("\n(G+H) both: gate 1e9 and nav_h_s 8:")
tok, lat = run(1e9, 8.0)
A, Bm, ag = counts(adj(tok), "")
print(f"  A {int(A.sum())}  B {int(Bm.sum())}  agree {int(ag.sum())}  sum {int(A.sum() + Bm.sum())}  (remaining A causes: {dict(zip(*np.unique(cA[A], return_counts=True)))};  remaining B causes: {dict(zip(*np.unique(cB[Bm], return_counts=True)))})")
S2.TURN_MAX_VMIN_MS = 8.0
B.LITERALS["nav_h_s"] = 6.0
