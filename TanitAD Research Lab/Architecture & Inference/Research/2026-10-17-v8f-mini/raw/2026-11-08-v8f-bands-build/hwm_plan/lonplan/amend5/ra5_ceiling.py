"""AMEND_HWMPLAN_5 draft measurement: what can a level-1 cost read through RA express on 0-2 s acceleration? CPU, TRAIN windows only, zero training. No SCORE window is planned or scored.
usage: ra5_ceiling.py <s0|s1> <readout_K10_R35_s{k}.pt> <out.json>

DESIGN CONSTANTS, FIXED BEFORE THIS SCRIPT FIRST RAN (2026-11-17). None is tuned on its output; every number it writes is a TRAIN, in-sample feasibility record, never a result.
* RA       : the frozen readout of ACTRSP Amendment 1 (ridge on z_{t+10} - z_t -> mean scaled acceleration over the 10 transitions). sha256 pinned below; refused on mismatch.
             ra(D) = ((D - mu) / sd) @ W + ym. Raw m/s^2 = 2.0 x scaled (hwm_plan.ACC_SCALE).
* C-RA-FIT : the RA fit set rebuilt from the local TRAIN latents exactly as amend1_kstep.py builds it (frame table, K 10, finite actions); n_pairs must be 8,306 in 141 clips; the ridge refit at the
             frozen lambda must reproduce the frozen (mu, sd, W, ym). Split guard: 0 of the TRAIN clips may be a SCORE episode (eid list of the 994 SCORE windows, read for ids only).
* Windows  : LON-PLAN's TRAIN selection set (sel_<k>/selection_windows.npz, 564 windows in 141 TRAIN clips, 0 eval clips). Frames 0..10 of each window. All 564, batches of 64.
* The RA pass (the operation the amendment proposes for level 1): variable = level 1's 12 acceleration knots (scaled), yaw channel held at HOLD_c's yaw; one candidate, HOLD_c (S = 1);
             cost(a) = (ra(F^10(z0, a) - z0) - target)^2, F = R35's level-1 rollout (context 1); the library's st._optimize: AdamW lr 0.1 (level-1 lr), 90 iterations, patience 10, rel_tol 1e-4,
             weight decay 0, clipped to the level-1 prior box. The plan is knots 1-10 of the best iterate.
* Rows (0-2 s acc_mae, raw m/s^2, against the logged accelerations of knots 1-10):
    HOLD       : a0 held (raw).                          HOLD_c   : a0 held, clipped to the level-1 box.
    ORAC_MEAN  : the logged 2 s mean, held (what any 2 s mean target can express with a flat profile; an oracle, never a plan).
    RA_READ    : 2.0 * ra(z_10 - z_0), held: RA applied to the TRUE 2 s displacement (in-sample for RA), no world model, no planner.
    RA_ORACLE  : the RA pass with target = ra(z_10 - z_0) (the best target an upper level could hand level 1 at frame 10). Ceiling of the RA read.
    RA_PLANT   : the RA pass with target = ra(F^10(z0, logged scaled actions) - z0): inversion control; its 2 s mean must land near the logged 2 s mean.
    RA_NEG     : the RA pass with the oracle target mirrored about HOLD_c's own read (2 ra(F^10(z0, HOLD_c)) - target): deliberate regression, must read worse than RA_ORACLE.
    RA_SHUF    : the RA pass with W permuted (numpy RandomState(0)) on both sides, oracle target formed with the permuted W: shuffled-readout control.
    RA_ORACLE_S16 : RA_ORACLE with 16 candidates (candidate 0 = HOLD_c, others HOLD_c + prior_std_1 * N(0,1) on the acceleration channel, generator manual_seed(2**42 + batch)), argmin pick:
                    measures what a pick among candidates does under a one-dimensional cost.
* Beside   : gain g = (ra(F^10(z0, HOLD_c + 0.5 e_acc)) - ra(F^10(z0, HOLD_c - 0.5 e_acc))) / 1.0 per window (scaled per scaled); Pearson r of each row's 2 s mean departure from a0 against the logged one.
* Estimator: clip-cluster percentile bootstrap, B 2,000, seed 0, over the 141 TRAIN clips, on row - HOLD and on RA_NEG / RA_SHUF / RA_ORACLE_S16 - RA_ORACLE. Answers another draw of TRAIN clips with
             R35 and RA held fixed. RA was FIT on these clips, so every RA row is in-sample for RA and optimistic.
"""
import hashlib, json, os, sys, time
sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
CODE_ROOT = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory"
for _d in ("wmact", "plantest2", "plantest", "hwm_plan"):
    sys.path.insert(0, os.path.join(CODE_ROOT, _d))
import numpy as np
import torch
import hwm_plan as H, hwm_data as D, plantest_lib as P, plantest2_lib as Q

torch.set_num_threads(2)
K, RA_PATH, OUTJ = sys.argv[1], sys.argv[2], sys.argv[3]
TAG = f"R35_{K}"
RA_SHA = {"s0": "edab7dce7b57f9ae21d90e5cc317c914aed432751fde7a7d4e6cf6af0748a6cf", "s1": "a6d252aa405f624a7cea4129a7f7d65ba5c047045d88e89b340a8abb935bb495"}[K]
LP = "F:/TanitAD-scratch/actrsp/lonplan"
CKPT = f"C:/Users/Admin/t2v8f_pulls/ladder_{TAG}/ckpt_last.pt"
LAT = f"F:/TanitAD-scratch/hwm_plan/latents_train_{TAG}.pt"
WIN = f"{LP}/sel_{K}/selection_windows.npz"
SCOREWIN = f"{LP}/score_windows_s1/windows.npz"   # eid only (split guard); the 71 SCORE episodes are the same at both chassis
KS = 10
ACC = float(H.ACC_SCALE)
T0 = time.time()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


res = {"tag": TAG, "tier": "T0 world-model diagnostic; TRAIN, in-sample for R35 and for RA; feasibility record, never a result", "inputs": {}}
for nm, p in (("ckpt", CKPT), ("latents_train", LAT), ("selection_windows", WIN), ("readout", RA_PATH), ("score_windows_s1_eid_only", SCOREWIN), ("script", os.path.abspath(__file__))):
    res["inputs"][nm] = {"path": p, "sha256": sha(p)}
assert res["inputs"]["readout"]["sha256"] == RA_SHA, "RA sha256 mismatch"

# ------------------------------------------------------------------ model, priors, planner pieces (as l2img_probe / LON-PLAN)
m, cfg, ck = D.load_model(CKPT)
wm, st = m.wm, m.wm.stack
for p_ in wm.parameters():
    p_.requires_grad_(False)
st.eval()
Z, A = D.trainset_from_latents_pt(LAT, TAG)
D.fit_priors(st, wm, Z, A)
geom = H.Geometry.from_stack(st, wm, goal_steps=None, stitch="decode")
sol = H.SolverCfg(seed_mode="hold", lam=0.0, S=P.S_CAND, n_iters=P.N_ITERS, decode_lam=0.0, decode_iters=300, plan_ctx=1)
planner = H.HWMPlanner(wm, geom, sol)
lo1, hi1 = st.prior_lo_1, st.prior_hi_1
assert geom.n_op == 10 and geom.h1 == 12, (geom.n_op, geom.h1)
log("model + priors")

# ------------------------------------------------------------------ RA and C-RA-FIT
R = torch.load(RA_PATH, map_location="cpu", weights_only=False)
ra_np = R["RA"]
mu, sd, W, ym = (np.asarray(ra_np[k], np.float64) for k in ("mu", "sd", "W", "ym"))
LT = torch.load(LAT, map_location="cpu", weights_only=False)
Zw, Aw, okw = LT["Z1"][TAG].float(), wm.scale_actions(LT["A_raw"].float()), LT["ok"].bool()
tw, eidw = np.asarray(LT["t"]), list(LT["eid"])
clips = sorted(set(eidw)); cix = {c: i for i, c in enumerate(clips)}
tab = {}
for w_ in range(Zw.shape[0]):
    for j in range(Zw.shape[1]):
        if bool(okw[w_, j]):
            tab.setdefault((cix[eidw[w_]], int(tw[w_]) - (Zw.shape[1] - 1) + j), (Zw[w_, j], Aw[w_, j]))
ZT, ZK, AB = [], [], []
for ci in range(len(clips)):
    fr = sorted(f for (c, f) in tab if c == ci)
    pos = {int(f): i for i, f in enumerate(fr)}
    Zc = torch.stack([tab[(ci, f)][0] for f in fr]); Ac = torch.stack([tab[(ci, f)][1] for f in fr])
    for i, f in enumerate(fr):
        f = int(f)
        if all((f + q) in pos for q in range(KS + 1)):
            js = [pos[f + q] for q in range(KS)]
            if torch.isfinite(Ac[js]).all():
                ZT.append(Zc[i]); ZK.append(Zc[pos[f + KS]]); AB.append(Ac[js].mean(0))
Dr = (torch.stack(ZK) - torch.stack(ZT)).numpy().astype(np.float64)
Yf = torch.stack(AB).numpy().astype(np.float64)[:, :1]
mu2, sd2 = Dr.mean(0), Dr.std(0) + 1e-6
Xs = (Dr - mu2) / sd2; ym2 = Yf.mean(0)
lam = float(R["lambda"])
W2 = np.linalg.solve(Xs.T @ Xs + lam * len(Dr) * np.eye(Dr.shape[1]), Xs.T @ (Yf - ym2))
sw = np.load(SCOREWIN, allow_pickle=False)
score_eids = set(sw["eid"].tolist())
res["C_RA_FIT"] = {"n_pairs": int(len(Dr)), "n_clips": len(clips), "n_pairs_record": 8306, "lambda": lam,
                   "max_abs_dW": float(np.abs(W2 - W).max()), "max_abs_W": float(np.abs(W).max()), "max_abs_dmu": float(np.abs(mu2 - mu).max()),
                   "max_abs_dsd": float(np.abs(sd2 - sd).max()), "abs_dym": float(np.abs(ym2 - ym).max()),
                   "train_clips_in_score_episodes": len(set(clips) & score_eids), "n_score_episodes": len(score_eids), "n_score_windows": int(len(sw["eid"])),
                   "s_cal_swap_heldout_frozen": R.get("s_cal_swap_heldout")}
log("C_RA_FIT", res["C_RA_FIT"])

MU, SD, WT, YM = (torch.tensor(x, dtype=torch.float32) for x in (mu, sd, W, ym))
perm = np.random.RandomState(0).permutation(W.shape[0])
WSH = WT[torch.as_tensor(perm)]


def ra(Dz, Wv):
    return (((Dz - MU) / SD) @ Wv)[..., 0] + YM[0]


# ------------------------------------------------------------------ windows
w = np.load(WIN, allow_pickle=False)
WZ, WA, WE = torch.from_numpy(w["z"]).float(), np.asarray(w["A"], np.float64), w["eid"]
assert len(set(WE.tolist()) & score_eids) == 0
n = WZ.shape[0]
z0_all, z10_all = WZ[:, 0], WZ[:, KS]
L = WA[:, :KS, 0]                                          # logged raw acceleration, knots 1-10
a0 = WA[:, 0, 0]
hc_all = planner.hold_c(WA[:, 0])                          # (n, n_hold, 2) scaled, clipped
A_sc = wm.scale_actions(torch.from_numpy(WA[:, :KS]).float())


def roll10(z0, a):
    """z0 (B, D); a (B, S, H>=10, 2) scaled -> level-1 state after 10 actions (B, S, D)."""
    z0e = z0.unsqueeze(1).expand(a.shape[0], a.shape[1], -1)
    return st.rollout(1, z0e, a)[:, :, KS - 1]


def ra_pass(z0, hc, target, Wv, S=1, gen=None):
    """The RA pass: level 1's 12 acceleration knots from HOLD_c, yaw held at HOLD_c, cost (ra(F^10(z0,a) - z0) - target)^2."""
    b = z0.shape[0]
    base = hc[:, :geom.h1].unsqueeze(1).expand(b, S, geom.h1, 2).clone()
    if S > 1:
        noise = torch.randn(b, S, geom.h1, generator=gen)
        base[:, 1:, :, 0] = base[:, 1:, :, 0] + st.prior_std_1[0] * noise[:, 1:]
        base = torch.max(torch.min(base, hi1), lo1)
    yaw = hc[:, :geom.h1, 1:2].unsqueeze(1).expand(b, S, geom.h1, 1)

    def cost(a):
        a_eff = torch.cat([a[..., :1], yaw], -1)
        r = ra(roll10(z0, a_eff) - z0.unsqueeze(1), Wv)
        return (r - target.unsqueeze(1)) ** 2

    with torch.enable_grad():
        best_a, best_cost, init_cost, hist, nit = st._optimize(cost, base, lr=float(P.LR[1]), n_iters=P.N_ITERS, patience=P.PATIENCE, rel_tol=P.REL_TOL,
                                                               weight_decay=0.0, lo=lo1, hi=hi1, project=None)
    pick = best_cost.argmin(1)
    ar = torch.arange(b)
    ch = best_a[ar, pick]
    return ch[:, :KS, 0].detach().numpy().astype(np.float64) * ACC, best_cost[ar, pick].detach().numpy(), int(nit), float((ch[:, :, 1] - hc[:, :geom.h1, 1]).abs().max())


rows = {k: np.zeros((n, KS)) for k in ("HOLD", "HOLD_c", "ORAC_MEAN", "RA_READ", "RA_ORACLE", "RA_PLANT", "RA_NEG", "RA_SHUF", "RA_ORACLE_S16")}
aux = {k: [] for k in ("gain", "cost_ORACLE", "cost_PLANT", "cost_NEG", "cost_SHUF", "iters", "yaw_moved", "sec_S1")}
B = P.BATCH
for b0 in range(0, n, B):
    ii = slice(b0, min(n, b0 + B))
    z0, z10, hc = z0_all[ii], z10_all[ii], hc_all[ii]
    nb = z0.shape[0]
    with torch.no_grad():
        t_or = ra(z10 - z0, WT)
        t_sh = ra(z10 - z0, WSH)
        t_pl = ra(roll10(z0, A_sc[ii].unsqueeze(1))[:, 0] - z0, WT)
        r_h = ra(roll10(z0, hc[:, :geom.h1].unsqueeze(1))[:, 0] - z0, WT)
        hp, hm = hc[:, :geom.h1].clone(), hc[:, :geom.h1].clone()
        hp[:, :, 0] += 0.5; hm[:, :, 0] -= 0.5
        g = ra(roll10(z0, hp.unsqueeze(1))[:, 0] - z0, WT) - ra(roll10(z0, hm.unsqueeze(1))[:, 0] - z0, WT)
    aux["gain"] += g.tolist()
    rows["HOLD"][ii] = np.repeat(a0[ii, None], KS, 1)
    rows["HOLD_c"][ii] = hc[:, :KS, 0].numpy().astype(np.float64) * ACC
    rows["ORAC_MEAN"][ii] = np.repeat(L[ii].mean(1, keepdims=True), KS, 1)
    rows["RA_READ"][ii] = np.repeat((t_or.numpy().astype(np.float64) * ACC)[:, None], KS, 1)
    t1 = time.time()
    rows["RA_ORACLE"][ii], c, it, ym_ = ra_pass(z0, hc, t_or, WT)
    aux["sec_S1"].append((time.time() - t1) / nb); aux["cost_ORACLE"] += c.tolist(); aux["iters"].append(it); aux["yaw_moved"].append(ym_)
    rows["RA_PLANT"][ii], c, _, _ = ra_pass(z0, hc, t_pl, WT); aux["cost_PLANT"] += c.tolist()
    rows["RA_NEG"][ii], c, _, _ = ra_pass(z0, hc, 2 * r_h - t_or, WT); aux["cost_NEG"] += c.tolist()
    rows["RA_SHUF"][ii], c, _, _ = ra_pass(z0, hc, t_sh, WSH); aux["cost_SHUF"] += c.tolist()
    gen = torch.Generator().manual_seed(2 ** 42 + b0 // B)
    rows["RA_ORACLE_S16"][ii], _, _, _ = ra_pass(z0, hc, t_or, WT, S=P.S_CAND, gen=gen)
    log("batch", b0 // B, "s/window S1", round(aux["sec_S1"][-1], 3))

# ------------------------------------------------------------------ read
cl = np.array([{c: i for i, c in enumerate(sorted(set(WE.tolist())))}[e] for e in WE])
ncl = cl.max() + 1
rng = np.random.RandomState(0)
BS = [rng.randint(0, ncl, ncl) for _ in range(2000)]
cnt = np.bincount(cl, minlength=ncl)


def boot(d):
    s = np.bincount(cl, weights=d, minlength=ncl)
    pts = np.array([s[bb].sum() / cnt[bb].sum() for bb in BS])
    return [round(float(d.mean()), 4), [round(float(np.percentile(pts, 2.5)), 4), round(float(np.percentile(pts, 97.5)), 4)]]


mae = {k: np.abs(v - L).mean(1) for k, v in rows.items()}
dep_log = L.mean(1) - a0
out = {}
for k in rows:
    dep = rows[k].mean(1) - a0
    out[k] = {"acc_mae_0_2s": round(float(mae[k].mean()), 4), "minus_HOLD": boot(mae[k] - mae["HOLD"]),
              "r_departure_vs_logged": (round(float(np.corrcoef(dep, dep_log)[0, 1]), 4) if dep.std() > 0 else None),
              "mean_abs_departure_2s_mean": round(float(np.abs(dep).mean()), 4)}
for k in ("RA_NEG", "RA_SHUF", "RA_ORACLE_S16"):
    out[k]["minus_RA_ORACLE"] = boot(mae[k] - mae["RA_ORACLE"])
res["rows"] = out
res["RA_PLANT_recovery"] = {"mean_abs_planmean_minus_loggedmean": round(float(np.abs(rows["RA_PLANT"].mean(1) - L.mean(1)).mean()), 4),
                            "mean_abs_a0_minus_loggedmean": round(float(np.abs(a0 - L.mean(1)).mean()), 4)}
gq = np.percentile(aux["gain"], [10, 50, 90])
res["beside"] = {"gain_scaled_per_scaled_q10_q50_q90": [round(float(x), 4) for x in gq],
                 "best_cost_median": {k: round(float(np.median(aux[f"cost_{k}"])), 6) for k in ("ORACLE", "PLANT", "NEG", "SHUF")},
                 "iters_per_batch": aux["iters"], "yaw_max_abs_move": float(max(aux["yaw_moved"])), "sec_per_window_S1": round(float(np.mean(aux["sec_S1"])), 4),
                 "n_windows": int(n), "n_clips": int(ncl),
                 "r_a0_vs_logged_2s_mean": round(float(np.corrcoef(a0, L.mean(1))[0, 1]), 4),
                 "r_RA_READ_vs_logged_2s_mean": round(float(np.corrcoef(rows["RA_READ"][:, 0], L.mean(1))[0, 1]), 4)}
res["estimator"] = "clip-cluster percentile bootstrap, B 2000, seed 0, over the TRAIN clips; another draw of TRAIN clips, R35 and RA fixed; in-sample for RA"
res["wall_s"] = round(time.time() - T0, 1)
with open(OUTJ, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
log("WROTE", OUTJ)
print("ZZ RA5 DONE ZZ", flush=True)
