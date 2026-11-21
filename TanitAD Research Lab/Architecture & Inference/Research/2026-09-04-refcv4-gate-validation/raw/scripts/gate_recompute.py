"""INDEPENDENT recompute of the refcv4 anchor-vocabulary gate.

Gate (Master Mind, 2026-09-04): the NEW vocabulary's ORACLE-IN-VOCABULARY ADE
(0-2 s, held-out windows) must beat `ha` = 0.2996 m -- the hold-action control.

This runs on the EXACT surface the 0.2996 was measured on: the banked per-window
dump of the refcv3 step-40,284 open-loop suite, n = 4,823 windows / 141 episodes,
window-stride 5, K=4 instants 0.5/1.0/1.5/2.0 s, ego frame at t0, metres.

No model, no forward pass, no GPU.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "refcv3_40284_dump")

# ---------------------------------------------------------------- load dump
eps = sorted(f for f in os.listdir(DUMP) if f.startswith("ep") and f.endswith(".npz"))
G, HA, HA0, OS, ORS, V0, EPI = [], [], [], [], [], [], []
for i, f in enumerate(eps):
    z = np.load(os.path.join(DUMP, f))
    G.append(z["g"])
    HA.append(z["ha"])
    HA0.append(z["ha0"])
    OS.append(z["os"])
    ORS.append(z["oracle_sel"])
    V0.append(z["v0"])
    EPI.append(np.full(z["g"].shape[0], i, dtype=np.int64))
G = np.concatenate(G).astype(np.float64)      # [N,4,2]
HA = np.concatenate(HA).astype(np.float64)
HA0 = np.concatenate(HA0).astype(np.float64)
OS = np.concatenate(OS).astype(np.float64)
ORS = np.concatenate(ORS).astype(np.float64)
V0 = np.concatenate(V0).astype(np.float64)
EPI = np.concatenate(EPI)
N = G.shape[0]
print(f"[surface] windows={N} episodes={len(eps)} K={G.shape[1]} dims={G.shape[2]}")


def ade_per_window(pred):
    """mean over K instants of euclidean distance -> [N]"""
    return np.linalg.norm(pred - G, axis=-1).mean(axis=1)


def summarize(name, per_win):
    print(f"  {name:<26s} ADE = {per_win.mean():.4f}")
    return per_win.mean()


# ------------------------------------------------- CONTROL 1: reproduce panel
print("\n=== CONTROL 1 — reproduce the published panel from the dump ===")
print("   (if these do not match, my ADE definition is wrong and nothing below counts)")
exp = {"ha": 0.2996, "ha0": 0.6723, "os": 0.4419, "oracle_sel": 0.3668}
got = {}
for nm, arr in (("ha", HA), ("ha0", HA0), ("os", OS), ("oracle_sel", ORS)):
    got[nm] = ade_per_window(arr).mean()
    d = got[nm] - exp[nm]
    flag = "OK " if abs(d) < 5e-5 else "MISMATCH"
    print(f"  {flag} {nm:<12s} got {got[nm]:.6f}  published {exp[nm]:.4f}  d={d:+.6f}")

# ---------------------------- CONTROL 2: ha0 must be the constant-velocity ray
print("\n=== CONTROL 2 — frame/units check: ha0 == (v0*t, 0) in ego frame ===")
T = np.array([0.5, 1.0, 1.5, 2.0])
cv = np.zeros_like(HA0)
cv[:, :, 0] = V0[:, None] * T[None, :]
err = np.abs(HA0 - cv)
print(f"  max |ha0 - (v0*t,0)| = {err.max():.6e} m   "
      f"(mean {err.mean():.3e})  -> x is ALONG-TRACK, y is LATERAL, metres")

# --------------------------------------------------------------- the anchors
apath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "anchors_pod.pt")
A = torch.load(apath, map_location="cpu", weights_only=True)
if isinstance(A, dict):
    A = A.get("anchors", A)
A = np.asarray(A, dtype=np.float64)
print(f"\n[anchors] {os.path.basename(apath)} shape={A.shape}")
A4 = A[:, :4, :]                                    # slots 5,10,15,20 = 0.5..2.0 s

# --------------------------------- CONTROL 3: a no-information anchor set
print("\n=== CONTROL 3 — controls that must read a known value ===")
zero_ade = np.linalg.norm(G, axis=-1).mean(axis=1)
print(f"  zero-path (no information)   ADE = {zero_ade.mean():.4f}"
      f"   [must be large: the raw GT displacement]")


def oracle_in_vocab(anch4):
    """min over anchors of the 0-2 s ADE; returns (ade[N], along[N], lat[N], idx[N])"""
    d = np.linalg.norm(G[:, None, :, :] - anch4[None, :, :, :], axis=-1)  # [N,A,K]
    ade = d.mean(axis=2)                                                  # [N,A]
    idx = ade.argmin(axis=1)
    best = anch4[idx]                                                     # [N,K,2]
    resid = best - G
    return ade[np.arange(len(idx)), idx], np.abs(resid[..., 0]).mean(1), \
        np.abs(resid[..., 1]).mean(1), idx


print("\n=== THE GATE — oracle-in-vocabulary, raw (unclamped) vocabulary ===")
ov, ov_a, ov_l, ov_i = oracle_in_vocab(A4)
print(f"  oracle-in-vocab ADE   = {ov.mean():.4f} m")
print(f"    ALONG-TRACK MAE     = {ov_a.mean():.4f} m")
print(f"    LATERAL     MAE     = {ov_l.mean():.4f} m")
print(f"    distinct anchors hit= {len(np.unique(ov_i))} of {A4.shape[0]}")

# ------------------------------------------------ paired episode-cluster boot
def paired_boot(a, b, epi, n_boot=2000, seed=0):
    """paired episode-cluster bootstrap on (a-b); returns (delta, lo, hi)"""
    rng = np.random.default_rng(seed)
    uq = np.unique(epi)
    idx_by_ep = [np.where(epi == e)[0] for e in uq]
    d = a - b
    obs = d.mean()
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uq), len(uq))
        sel = np.concatenate([idx_by_ep[p] for p in pick])
        out[i] = d[sel].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return obs, lo, hi


ha_pw = ade_per_window(HA)
d, lo, hi = paired_boot(ov, ha_pw, EPI)
sep = "SEPARATED" if (lo > 0) == (hi > 0) else "not separated"
verdict = "PASS" if hi < 0 else ("FAIL" if lo > 0 else "INCONCLUSIVE")
print(f"\n  vs ha (0.2996):  delta = {d:+.4f} m  [{lo:+.4f}, {hi:+.4f}]  {sep}")
print(f"  GATE VERDICT: {verdict}  "
      f"(negative delta separated from 0 = the vocabulary beats hold-action)")

# ------------------------------------------- reach clamp: what the model sees
print("\n=== The DEPLOYED fan — after the reach clamp (a_max sweep) ===")
print("   clamp: an anchor survives if its mean speed over the horizon is within")
print("   v0 +/- a_max*horizon_s.  horizon_s = 6.0 (max(V3_HORIZONS)*0.1).")
HOR_S = 6.0
# mean speed of an anchor over the FULL 6 s horizon = |x at 6s| / 6  (path length proxy
# matching refc: it uses the terminal along-track displacement / horizon)
seg = np.diff(np.concatenate([np.zeros((A.shape[0], 1, 2)), A], axis=1), axis=1)
arc = np.linalg.norm(seg, axis=-1).sum(axis=1)          # total path length over 6 s
vmean_anchor = arc / HOR_S                              # [A]
for amax in (1.5, 2.0, 2.5):
    band = amax * HOR_S
    ok = np.abs(vmean_anchor[None, :] - V0[:, None]) <= band        # [N,A]
    kill = 1.0 - ok.mean()
    empty = (~ok.any(axis=1)).mean()
    dm = np.linalg.norm(G[:, None, :, :] - A4[None, :, :, :], axis=-1).mean(axis=2)
    dm = np.where(ok, dm, np.inf)
    cl = dm.min(axis=1)
    cl = np.where(np.isfinite(cl), cl, ov)              # empty windows fall back
    print(f"  a_max={amax:<4} band=+/-{band:4.1f} m/s  kill={kill*100:5.2f}%  "
          f"empty={empty*100:.2f}%  clamped-oracle ADE = {cl.mean():.4f} m  "
          f"(dADE {cl.mean()-ov.mean():+.5f})")

# ------------------------------------------------------------ Kamm circle
print("\n=== Kamm-circle feasibility of the vocabulary (dry road mu=0.9) ===")
DT = 0.1 * np.array([5, 10, 15, 20, 30, 40, 50, 60], dtype=np.float64)
P = np.concatenate([np.zeros((A.shape[0], 1, 2)), A], axis=1)        # [A,9,2]
tt = np.concatenate([[0.0], DT])
dt = np.diff(tt)
d_seg = np.diff(P, axis=1)                                            # [A,8,2]
ds = np.linalg.norm(d_seg, axis=-1)
v = ds / dt[None, :]                                                  # [A,8] segment speed
a_lon = np.diff(np.concatenate([v[:, :1], v], axis=1), axis=1) / dt[None, :]
th = np.arctan2(d_seg[..., 1], d_seg[..., 0])
dth = np.diff(np.concatenate([th[:, :1], th], axis=1), axis=1)
dth = (dth + np.pi) % (2 * np.pi) - np.pi
yaw = dth / dt[None, :]
a_lat = v * yaw
a_tot = np.hypot(a_lon, a_lat)
G_ACC = 9.81
mu = 0.9
viol = a_tot > mu * G_ACC
n_break = int(viol.any(axis=1).sum())
stalled_turn = int(((v < 0.5) & (np.abs(yaw) > 0.2)).any(axis=1).sum())
print(f"  anchors breaking a {mu} g circle: {n_break} of {A.shape[0]}"
      f"   peak |a| = {a_tot.max()/G_ACC:.2f} g")
print(f"  max |a_lon| = {np.abs(a_lon).max():.2f} m/s2   "
      f"max |a_lat| = {np.abs(a_lat).max():.2f} m/s2   max speed = {v.max():.2f} m/s")
print(f"  anchors that turn while stalled (v<0.5 m/s, |yaw|>0.2 rad/s): {stalled_turn}")

json.dump(
    {
        "surface": {"windows": int(N), "episodes": int(len(eps)),
                    "dump": "taniteval/results/refcv3-40284-openloop-dump.tar.gz"},
        "panel_reproduced": {k: float(v) for k, v in got.items()},
        "anchors_file": os.path.basename(apath),
        "oracle_in_vocab": {
            "ade": float(ov.mean()), "along": float(ov_a.mean()),
            "lat": float(ov_l.mean()), "distinct": int(len(np.unique(ov_i)))},
        "vs_ha": {"delta": float(d), "lo": float(lo), "hi": float(hi),
                  "verdict": verdict},
        "kamm": {"n_break": n_break, "peak_g": float(a_tot.max() / G_ACC),
                 "stalled_turn": stalled_turn},
    },
    open(os.path.join(HERE, "GATE_RECOMPUTE.json"), "w"), indent=1)
print("\nwrote GATE_RECOMPUTE.json")
