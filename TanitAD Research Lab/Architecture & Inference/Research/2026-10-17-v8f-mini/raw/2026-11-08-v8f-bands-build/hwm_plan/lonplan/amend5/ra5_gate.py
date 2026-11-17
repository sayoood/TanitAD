"""AMEND_HWMPLAN_5 draft measurement, part 3 (ADDED AFTER parts 1-2 showed RA_ORACLE above HOLD at every lambda, with R35's per-window gain along W spread from q10 0.01-0.04 to q90 0.51):
is the RA pass's excess concentrated in low-gain windows, and would a gain gate (the RA pass where g >= g_min, HOLD_c elsewhere) beat HOLD? g is computable at inference (z_now and HOLD_c only).
usage: ra5_gate.py <s0|s1> <readout_K10_R35_s{k}.pt> <out.json>
Constants fixed before it ran: RA_ORACLE as ra5_ceiling.py (lambda 0, S 1); gain tercile split; gates at the TRAIN quantiles q50, q75, q90 of g. TRAIN, in-sample for RA and R35 and for the gate quantiles: a feasibility record, never a selection or a result."""
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



G = np.zeros(n); PL = np.zeros((n, KS))
for b0 in range(0, n, P.BATCH):
    ii = slice(b0, min(n, b0 + P.BATCH))
    z0, z10, hc = z0_all[ii], z10_all[ii], hc_all[ii]
    with torch.no_grad():
        t_or = ra(z10 - z0, WT)
        hp, hm = hc[:, :geom.h1].clone(), hc[:, :geom.h1].clone()
        hp[:, :, 0] += 0.5; hm[:, :, 0] -= 0.5
        G[ii] = (ra(roll10(z0, hp.unsqueeze(1))[:, 0] - z0, WT) - ra(roll10(z0, hm.unsqueeze(1))[:, 0] - z0, WT)).numpy()
    PL[ii] = ra_pass(z0, hc, t_or, WT)[0]
    log("batch", b0 // P.BATCH)
hold = np.repeat(a0[:, None], KS, 1); holdc = hc_all[:, :KS, 0].numpy().astype(np.float64) * ACC
mh, mo = np.abs(hold - L).mean(1), np.abs(PL - L).mean(1)
cl = np.array([{c: i for i, c in enumerate(sorted(set(WE.tolist())))}[e] for e in WE]); ncl = cl.max() + 1
rng = np.random.RandomState(0); BS = [rng.randint(0, ncl, ncl) for _ in range(2000)]; cnt = np.bincount(cl, minlength=ncl)


def boot(d):
    s = np.bincount(cl, weights=d, minlength=ncl)
    pts = np.array([s[bb].sum() / cnt[bb].sum() for bb in BS])
    return [round(float(d.mean()), 4), [round(float(np.percentile(pts, 2.5)), 4), round(float(np.percentile(pts, 97.5)), 4)]]


t1, t2 = np.percentile(G, [100 / 3, 200 / 3])
terc = {}
for nm, msk in (("low", G < t1), ("mid", (G >= t1) & (G < t2)), ("high", G >= t2)):
    terc[nm] = {"n": int(msk.sum()), "g_range": [round(float(G[msk].min()), 4), round(float(G[msk].max()), 4)],
                "RA_ORACLE_minus_HOLD_mean": round(float((mo - mh)[msk].mean()), 4), "HOLD_acc_mae": round(float(mh[msk].mean()), 4)}
gates = {}
for q in (50, 75, 90):
    gm = np.percentile(G, q); use = G >= gm
    plan = np.where(use[:, None], PL, holdc)
    gates[f"q{q}"] = {"g_min": round(float(gm), 4), "share_gated_in": round(float(use.mean()), 4), "minus_HOLD": boot(np.abs(plan - L).mean(1) - mh)}
res["gain_terciles"] = terc; res["gates"] = gates; res["RA_ORACLE_minus_HOLD"] = boot(mo - mh)
res["wall_s"] = round(time.time() - T0, 1)
with open(OUTJ, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print("ZZ RA5GATE DONE ZZ", flush=True)
