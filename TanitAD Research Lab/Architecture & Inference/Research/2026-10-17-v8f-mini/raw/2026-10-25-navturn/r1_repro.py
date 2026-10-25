"""r1_repro.py -- re-run the mirror's OWN v9 builder (build_v9_labels.clip_geo / clip_labels, unmodified) on the eval-141 pose tracks and compare with the shipped join.
EXPLORATORY, MEASURED. Purpose: get the per-window segment facts the join does not carry (is_turn, announced, d_next vs d_ann, in-progress, the dominant
segment's vmin ...) and VALIDATE the reconstruction against the join before any cause is read from it.

Reconstruction: the builder's log is the native ego log; here it is rebuilt from the v2ep pose track (x, y, yaw, v per camera frame k = 0..200) with the
clip clock ts(k) = g0 + k*dt recovered from the join's now_s (even k). A synthetic sample at ts = 0 (linear back-extrapolation) restores the builder's
t = 0 origin. Reversal (log.rev) is recomputed from the track. Per clip both turn-suppression OFF and ON are run; the join decides which one a clip used.
Output: repro_eval141.npz (per join row of the 141 clips: reproduced fields) + the agreement table on stdout.
"""
import numpy as np
from nt_common import *
import build_v9_labels as B

J, K, T, grid = load_inputs()
P_all = clip_tracks(K)
off, nn = J["off"], J["n"]
FIELDS = ("nav_token", "nav_args_valid", "nav_d_next_m", "nav_d_ann_m", "nav_in_progress", "nav_d_end_m", "nav_dyaw_next_deg",
          "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m", "turn_dyaw_deg", "turn_r_arc_m", "turn_vmin_ms", "turn_is_turn",
          "turn_in_progress", "curve", "lat_seg_found", "lat_theta_deg", "h_obs_s", "reversing", "lat_cls_a", "v_now_ms")
NJ = int(nn.sum())
R = {f: np.full(NJ, np.nan) for f in FIELDS}
R["sup_on"] = np.zeros(NJ, bool)
R["t_left"] = np.full(NJ, np.nan)      # seconds of reconstructed log left after NOW (the REAL log is longer: MEASURED below)
R["s_left"] = np.full(NJ, np.nan)      # metres of reconstructed arc left after NOW
R["s_now"] = np.full(NJ, np.nan)
R["tok_off"] = np.full(NJ, np.nan)     # reproduced nav token with turn-suppression OFF
R["tok_on"] = np.full(NJ, np.nan)      # ... and with it ON (clip-level A7 suppression, SPEC 5.1: announced = is_turn and not suppressed)
R["dann_off"] = np.full(NJ, np.nan)
clip_info = []
TURNS = {}


def build_log(P, ns):
    n_raw = len(P)
    fin = np.where(np.isfinite(ns))[0]
    dt = float(np.median(np.diff(ns[fin]) / np.diff(fin)) / 2.0)
    g0 = float(ns[fin[0]] - 2 * fin[0] * dt)
    ts = g0 + np.arange(n_raw) * dt
    # synthetic t = 0 sample by linear back-extrapolation over the first interval
    if ts[0] > 1e-9:
        f = -ts[0] / (ts[1] - ts[0])
        x0 = P[0, 0] + f * (P[1, 0] - P[0, 0]); y0 = P[0, 1] + f * (P[1, 1] - P[0, 1])
        yw0 = P[0, 2] + f * (P[1, 2] - P[0, 2]); v0 = P[0, 3] + f * (P[1, 3] - P[0, 3])
        ts_a = np.concatenate([[0.0], ts]); X = np.concatenate([[x0], P[:, 0]]); Y = np.concatenate([[y0], P[:, 1]])
        YW = np.concatenate([[yw0], P[:, 2]]); V = np.concatenate([[v0], P[:, 3]])
    else:
        ts_a, X, Y, YW, V = ts, P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    vx, vy = np.gradient(X, ts_a), np.gradient(Y, ts_a)
    rev = ((np.hypot(vx, vy) > 0.3) & (np.cos(np.arctan2(vy, vx) - YW) < 0.0)).astype(np.float64)
    return B.log_from_arrays(ts_a, X, Y, np.unwrap(YW), V, None, rev), g0, dt


for c in range(len(nn)):
    o, n = int(off[c]), int(nn[c])
    ns = J["now_s"][o:o + n]
    P = P_all[c]
    if not J["has"][o:o + n].any():
        clip_info.append((c, n, np.nan, np.nan, 0, 0, 0, False))
        continue
    log, g0, dt = build_log(P, ns)
    jj = np.where(np.isfinite(ns))[0]
    jj = jj[jj >= 1]
    k = 2 * jj
    now_k = ns[jj]
    best = None
    for sup in (False, True):
        G = B.clip_geo(log, {"applied": True} if sup else None)
        F = B.clip_labels(G, now_k, k, None, None, None)
        navtok = np.asarray(F["nav_token"], np.int64)
        R["tok_on" if sup else "tok_off"][o + jj] = navtok
        if not sup:
            R["dann_off"][o + jj] = np.asarray(F["nav_d_ann_m"], np.float64)
        has = J["has"][o + jj]
        score = int(((navtok == J["nav"][o + jj][:, :3].argmax(1)) & has).sum())
        if best is None or score > best[0]:
            best = (score, sup, F, G)
        if not any(u.suppressed for u in G.turns) and sup:
            pass
    score, sup, F, G = best
    # if suppression OFF and ON score the same, call it OFF (no suppressed turn changes the nav)
    for f in FIELDS:
        if f in F:
            R[f][o + jj] = np.asarray(F[f], np.float64)
    R["sup_on"][o + jj] = sup
    R["t_left"][o + jj] = log.t_end - now_k
    R["s_now"][o + jj] = B.s_at(G.tr, now_k)
    R["s_left"][o + jj] = G.s_max - R["s_now"][o + jj]
    TURNS[int(c)] = {"t_end": float(log.t_end), "s_max": float(G.s_max), "sup": bool(sup),
                     "turns": [{"t0": u.t_start, "t1": u.t_end, "s0": u.s_start, "s1": u.s_end, "dyaw": u.dyaw_deg, "R": u.r_arc_m,
                                "vmin": u.v_min_ms, "is_turn": u.is_turn, "sup": u.suppressed} for u in G.turns]}
    clip_info.append((c, n, dt, g0, len(G.turns), sum(u.is_turn for u in G.turns), sum(u.announced for u in G.turns), sup))

np.savez_compressed(HERE + "repro_eval141.npz", **R, off=off, n=nn)
import json as _json
_json.dump(TURNS, open(HERE + "repro_turns.json", "w"))

# ------------------------------------------------------------------ agreement with the shipped join (all join rows with a v9 row, then the 2,399 grid windows)
has = J["has"]
tokJ = J["nav"][:, :3].argmax(1)
ok = has & np.isfinite(R["nav_token"])
print(f"join rows with a v9 row {int(has.sum())}; reproduced rows {int(ok.sum())}; clips with suppression ON chosen {sum(1 for ci_ in clip_info if ci_[7])} of {len(clip_info)}")
tokR = R["nav_token"].copy()
agree = (tokR[ok] == tokJ[ok])
print(f"NAV TOKEN agree: {int(agree.sum())}/{int(ok.sum())} = {agree.mean():.4f}")
for t in (0, 1, 2):
    m = ok & (tokJ == t)
    print(f"   join token {t}: n {int(m.sum())}  reproduced same {int((tokR[m] == t).sum())}")
vj = J["nav"][:, 4] == 1
print(f"NAV args_valid agree: {int(((R['nav_args_valid'][ok] == 1) == vj[ok]).sum())}/{int(ok.sum())}")
m = ok & vj & (R["nav_args_valid"] == 1)
dd = np.abs(R["nav_d_next_m"][m] - J["nav_d"][m])
print(f"NAV d_next (m), both valid n={int(m.sum())}: median |diff| {np.median(dd):.3f}, p95 {np.percentile(dd, 95):.3f}, max {dd.max():.2f}; within 1 m: {(dd <= 1.0).mean():.4f}")
for nm, ix in (("turn_t_start_s", 1), ("turn_t_end_s", 2), ("turn_d_start_m", 3), ("turn_len_m", 4), ("turn_dyaw_deg", 5), ("turn_r_arc_m", 6), ("lat_theta_deg", 0)):
    jv = J["cons"][:, ix].astype(np.float64)
    pres_j, pres_r = np.isfinite(jv) & has, np.isfinite(R[nm]) & ok
    both = pres_j & pres_r
    d = np.abs(R[nm][both] - jv[both])
    tol = {"turn_t_start_s": 0.15, "turn_t_end_s": 0.15, "turn_d_start_m": 2.0, "turn_len_m": 3.0, "turn_dyaw_deg": 3.0, "turn_r_arc_m": 5.0, "lat_theta_deg": 3.0}[nm]
    print(f"{nm:15s} presence: join {int(pres_j.sum())} repro {int(pres_r[ok].sum())} both {int(both.sum())} join-only {int((pres_j & ~pres_r & ok).sum())} "
          f"repro-only {int((~pres_j & pres_r & ok).sum())} | value |diff| median {np.median(d) if both.any() else float('nan'):.3f} within tol({tol}) {(d <= tol).mean() if both.any() else float('nan'):.3f}")
latR = R["lat_cls_a"]
lj = J["lat_cls"].astype(np.float64)
agl = (latR[ok] == lj[ok])
print(f"lat_cls_a agree {int(agl.sum())}/{int(ok.sum())} = {agl.mean():.4f}")

# restricted to the 2,399 grid windows at the rows the banked analysis used (turn label at rowT, token at rowN)
ci, tt, rowT = T["ci"], T["tt"], T["rowT"]
rowN = off[ci] + tt
okg = T["has"] & np.isfinite(R["nav_token"][rowT])
print(f"\nGRID windows with v9 row {int(T['has'].sum())}: reproduced nav token at rowN agree "
      f"{int((R['nav_token'][rowN][okg] == tokJ[rowN][okg]).sum())}/{int(okg.sum())}; turn-label presence at rowT agree "
      f"{int((np.isfinite(R['turn_t_start_s'][rowT][okg]) == np.isfinite(J['cons'][rowT, 1][okg])).sum())}/{int(okg.sum())}")

# ------------------------------------------------------------------ the same agreement restricted to rows whose lookahead the reconstructed track fully covers
d_ann = np.maximum(30.0, 6.0 * np.maximum(R["v_now_ms"], 0.0))
interior = ok & (R["t_left"] >= 8.5) & (R["s_left"] >= d_ann + 5.0)
print(f"INTERIOR rows (>= 8.5 s and >= d_ann + 5 m of reconstructed track left): {int(interior.sum())} of {int(ok.sum())} ({interior.mean():.1%} of all rows)")
ai = (tokR[interior] == tokJ[interior])
print(f"  NAV TOKEN agree {int(ai.sum())}/{int(interior.sum())} = {ai.mean():.4f}")
for t in (0, 1, 2):
    m = interior & (tokJ == t)
    print(f"    join token {t}: n {int(m.sum())} reproduced same {int((tokR[m] == t).sum())}")
print(f"  NAV args_valid agree {int(((R['nav_args_valid'][interior] == 1) == vj[interior]).sum())}/{int(interior.sum())}")
pj = np.isfinite(J["cons"][:, 1].astype(np.float64))
pr = np.isfinite(R["turn_t_start_s"])
print(f"  turn-label PRESENCE agree {int((pj[interior] == pr[interior]).sum())}/{int(interior.sum())}  (join-only {int((pj & ~pr)[interior].sum())}, repro-only {int((~pj & pr)[interior].sum())})")
both = interior & pj & pr
for nm, ix, tol in (("turn_t_start_s", 1, 0.15), ("turn_t_end_s", 2, 0.15), ("turn_d_start_m", 3, 2.0), ("turn_dyaw_deg", 5, 3.0), ("turn_r_arc_m", 6, 5.0)):
    d = np.abs(R[nm][both] - J["cons"][both, ix].astype(np.float64))
    print(f"    {nm:15s} both n={int(both.sum())}: median|diff| {np.median(d):.3f} within tol {tol}: {(d <= tol).mean():.3f}")
agl = (latR[interior] == lj[interior])
print(f"  lat_cls_a agree {int(agl.sum())}/{int(interior.sum())} = {agl.mean():.4f}")
