"""AMEND_HWMPLAN_4 draft measurement: the level-2 image constraint (implementation I, "prim") on TRAIN windows. CPU, zero training, no SCORE window read.
usage: l2img_probe.py <s0|s1> <arms, comma-separated from off,prim,rs> [n_batches=1]

DESIGN CONSTANTS, FIXED BEFORE THIS SCRIPT FIRST RAN (2026-11-17). None is tuned on its output; its numbers are a feasibility record, never a result.
* off  : the code of record as LON-PLAN's A0 (seed-mode hold, lam 0, T3 2, plan-ctx 1, stitch decode, vocab off, constr 0). Nothing is installed.
* prim : level 2 optimises primitives p (B, S, h2*c2 = 30, 2) in the level-1 prior box; its macro is macro_enc(act_embed(p)) (the frozen encoders, as st.encode builds level-2 actions).
         candidate 0 = HOLD_c[:, :30]; candidates 1..S-1 = HOLD_c + prior_std_1 * N(0,1), clipped to the level-1 box, N drawn from a SEPARATE generator
         manual_seed(2**40 + seed*100003 + batch), so the planner's own generator is consumed exactly as under off (level 3 and level 1 draw the same numbers).
         lr = the level-1 lr (0.1: the variable lives in level-1 units); n_iters, patience, rel_tol, weight decay as the solver. The level-2 box is not applied (the image of the level-1 box replaces it).
         knots 11-30 = the chosen p[:, 10:30] itself (no decode).
* rs   : prim, plus level 1 plans against the level-2 rollout of a donor window: one cyclic shift d* per batch, the smallest d >= 1 with clip(i + d mod B) != clip(i) for every i (deliberate regression).
Windows: the LON-PLAN TRAIN selection set (sel_s{k}/selection_windows.npz, 564 windows, 141 TRAIN clips, 0 eval clips), batches of 64 in order, generator make_gen(0, batch), as the LON-PLAN lam_0 run.
"""
import hashlib, json, math, os, sys, time
sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
CODE_ROOT = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory"
for _d in ("wmact", "plantest2", "plantest", "hwm_plan"):
    sys.path.insert(0, os.path.join(CODE_ROOT, _d))
import numpy as np
import torch
import hwm_plan as H, hwm_data as D, plantest_lib as P, plantest2_lib as Q

torch.set_num_threads(2)
K = sys.argv[1]; ARMS = sys.argv[2].split(","); NB = int(sys.argv[3]) if len(sys.argv) > 3 else 1
TAG = f"R35_{K}"
LP = "F:/TanitAD-scratch/actrsp/lonplan"
OUT = f"{LP}/amend4"
CKPT = f"C:/Users/Admin/t2v8f_pulls/ladder_{TAG}/ckpt_last.pt"
LAT = f"F:/TanitAD-scratch/hwm_plan/latents_train_{TAG}.pt"
WIN = f"{LP}/sel_{K}/selection_windows.npz"
BANK = f"{LP}/sel_{K}/lam_0/plans_hwm_oracle_s0.npz"
SEED = 0
SMOKE = bool(os.environ.get("PROBE_SMOKE"))
if SMOKE:
    P.BATCH = 8


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


t0 = time.time()
m, cfg, ck = D.load_model(CKPT)
wm, st = m.wm, m.wm.stack
Z, A = D.trainset_from_latents_pt(LAT, TAG)
D.fit_priors(st, wm, Z, A)
geom = H.Geometry.from_stack(st, wm, goal_steps=None, stitch="decode")
sol = H.SolverCfg(seed_mode="hold", lam=0.0, S=P.S_CAND, n_iters=3 if SMOKE else P.N_ITERS, decode_lam=0.0, decode_iters=300, plan_ctx=1)
planner = H.HWMPlanner(wm, geom, sol)
lo1, hi1 = st.prior_lo_1, st.prior_hi_1
std_o = Q.prior_std_orig(st)
c2, h2 = geom.c2, geom.h2
NP = h2 * c2
mac2, mac3 = st.macro_encoders[0], st.macro_encoders[1]
assert geom.n_op == 10 and geom.n_traj == 30 and NP == 30 and geom.T3 == 2, geom

w = np.load(WIN, allow_pickle=False)
WZ, WA, WE = torch.from_numpy(w["z"]), w["A"], w["eid"]
goal = H.OracleGoal(WZ[:, geom.goal_frame], geom.goal_frame)
z_now, A0raw = WZ[:, 0], WA[:, 0]
hc_all = planner.hold_c(A0raw)


def M2(p):
    """p (..., 30, 2) scaled -> level-2 macros (..., 10, E2), exactly the grouping of st.encode / decode_macro."""
    e = st.act_embed(p)
    return mac2(e.reshape(*p.shape[:-2], p.shape[-2] // c2, c2, e.shape[-1]))


def M3(m2):
    """level-2 macros (..., 10, E2) -> level-3 macros (..., 2, E3)."""
    s3 = int(st.cfg[2].stride)
    return mac3(m2.reshape(*m2.shape[:-2], m2.shape[-2] // s3, s3, m2.shape[-1]))


class PrimShadow:
    """Instance shadow on st._optimize: the 2nd call of a hier plan (level 2) optimises primitives. Level 3 (1st call) and level 1 (3rd call) are the library's own."""

    def __init__(self, hc, seed, batch):
        self.hc, self.n, self.store = hc, 0, {}
        self.gen_p = torch.Generator().manual_seed(2 ** 40 + int(seed) * 100003 + int(batch))

    def __enter__(self):
        self.orig = st._optimize
        st.__dict__["_optimize"] = self.opt
        return self

    def __exit__(self, *e):
        st.__dict__.pop("_optimize", None)

    def opt(self, cost_fn, init, **kw):
        self.n += 1
        level = (3, 2, 1)[self.n - 1]
        if level != 2:
            return self.orig(cost_fn, init, **kw)
        b, S = init.shape[:2]
        sd = self.hc[:, :NP].to(init.dtype)
        noise = torch.randn(b, S, NP, 2, generator=self.gen_p, dtype=init.dtype)
        p0 = sd.unsqueeze(1) + st.prior_std_1.to(init.dtype) * noise
        p0[:, 0] = sd
        p0 = torch.max(torch.min(p0, hi1), lo1)
        kw2 = dict(kw); kw2.update(lr=float(P.LR[1]), lo=lo1, hi=hi1, project=None)
        best_p, best_cost, init_cost, hist, nit = self.orig(lambda p: cost_fn(M2(p)), p0, **kw2)
        with torch.no_grad():
            best_m = M2(best_p)
        self.store.update(best_p=best_p.detach(), best_cost=best_cost.detach())
        return best_m, best_cost, init_cost, hist, nit


class DonorRollout:
    """rs: the level-2 rollout that level 1 reads (the one call with 3-dim actions at level 2) is replaced by the donor rows."""

    def __init__(self, donor):
        self.donor = torch.as_tensor(donor)

    def __enter__(self):
        self.orig = st.rollout
        o, d = self.orig, self.donor

        def roll(level, z0, actions, *, context=1):
            r = o(level, z0, actions, context=context)
            return r[d] if (level == 2 and actions.dim() == 3) else r
        st.__dict__["rollout"] = roll
        return self

    def __exit__(self, *e):
        st.__dict__.pop("rollout", None)


def donor_shift(eids):
    B = len(eids)
    for d in range(1, B):
        if all(eids[(i + d) % B] != eids[i] for i in range(B)):
            return (np.arange(B) + d) % B, d
    raise RuntimeError("no clip-disjoint cyclic shift in this batch")


def gen_hash(g):
    return hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest()[:16]


def plan_arm(arm):
    out = {k: [] for k in ("plan_raw", "a1_raw", "m2", "m3", "c1", "c2", "c3", "it1", "it2", "it3", "resid", "p2_raw", "img_resid", "gen")}
    meta = {"donor_shift": []}
    t = time.time()
    for j in range(NB):
        ii = np.arange(j * P.BATCH, min(len(WE), (j + 1) * P.BATCH))
        hc = hc_all[ii]
        gen = P.make_gen(SEED, j)
        ctxs = []
        sh = None
        if arm in ("prim", "rs"):
            sh = PrimShadow(hc, SEED, j); ctxs.append(sh)
        if arm == "rs":
            dn, d = donor_shift(WE[ii]); meta["donor_shift"].append(int(d)); ctxs.append(DonorRollout(dn))
        for c in ctxs:
            c.__enter__()
        try:
            res = planner.plan_batch("hier", z_now[ii], goal(ii), hc, gen, goal_level=1, ii=ii)
        finally:
            for c in reversed(ctxs):
                c.__exit__()
        a1 = res.actions.detach()
        m2 = res.level_actions[2].detach()
        if sh is None:
            traj, resid = H.stitch(st, geom, a1, m2, hc, lo1, hi1, std_o, sol)
            out["resid"].append(resid.numpy())
        else:
            ar = torch.arange(len(ii))
            pick = sh.store["best_cost"].argmin(1)
            pc = sh.store["best_p"][ar, pick]                              # (B, 30, 2) the chosen level-2 primitives
            traj = torch.zeros(len(ii), geom.n_traj, 2)
            traj[:, :geom.n_op] = a1[:, :geom.n_op]
            traj[:, geom.n_op:] = pc[:, geom.n_op:geom.n_traj]
            with torch.no_grad():
                mm = M2(pc)
                sc = m2.pow(2).sum(-1).mean(-1).clamp_min(1e-12)
                ir = ((mm - m2).pow(2).sum(-1).mean(-1) / sc).sqrt()
            out["img_resid"].append(ir.numpy()); out["p2_raw"].append(H.to_raw(wm, pc.numpy().astype(np.float64)))
        out["plan_raw"].append(H.to_raw(wm, traj.numpy().astype(np.float64)))
        out["a1_raw"].append(H.to_raw(wm, a1.numpy().astype(np.float64)))
        out["m2"].append(m2.numpy()); out["m3"].append(res.level_actions[3].detach().numpy())
        for L in (1, 2, 3):
            out[f"c{L}"].append(res.costs[L].numpy()); out[f"it{L}"].append(np.full(len(ii), float(res.iters[L])))
        out["gen"].append(gen_hash(gen))
        print(arm, "batch", j, round(time.time() - t, 1), "s", flush=True)
    r = {k: (np.concatenate(v) if (v and k != "gen") else v) for k, v in out.items()}
    r["wall_s"] = time.time() - t
    r.update(meta)
    return r


res = {a: plan_arm(a) for a in ARMS}
n = NB * P.BATCH
n = min(n, len(WE))
Aw = WA[:n]


def q(x):
    x = np.asarray(x, np.float64).ravel()
    return dict(n=int(x.size), median=float(np.median(x)), q10=float(np.quantile(x, .1)), q90=float(np.quantile(x, .9)), max=float(x.max()), mean=float(x.mean()))


def band(plan, k0, k1, ch):
    return float(np.abs(plan[:, k0:k1, ch] - Aw[:, k0:k1, ch]).mean(1).mean())


R = dict(kind="AMEND4_L2IMG_PROBE", tier="T0 diagnostic on TRAIN windows (in-sample): a feasibility record, never a result", evidence="MEASURED", chassis=TAG, arms=ARMS, n_windows=n, n_batches=NB,
         inputs=dict(ckpt_sha256=sha(CKPT), train_latents_sha256=sha(LAT), windows_sha256=sha(WIN), banked_lam0_sha256=sha(BANK)), geometry=str(geom), solver=str(sol), threads=torch.get_num_threads())

# --- control: off reproduces LON-PLAN's lam_0 TRAIN plans (the same windows, batches, generator)
bk = np.load(BANK, allow_pickle=False)
if "off" in res:
    R["C_OFF_vs_banked_lam0"] = dict(plan_raw_maxabs=float(np.abs(res["off"]["plan_raw"] - bk["plan_raw"][:n]).max()), cost1_maxabs=float(np.abs(res["off"]["c1"] - bk["cost1"][:n]).max()),
                                     a1_maxabs=float(np.abs(res["off"]["a1_raw"] - bk["a1_raw"][:n]).max()), decode_resid_maxabs=float(np.abs(res["off"]["resid"] - bk["decode_resid"][:n]).max()))

# --- the TRAIN in-image macro cloud (all ok TRAIN 16-frame windows, as fit_priors), leave-own-clip-out nearest neighbour
Lt = P.load_latents(LAT, TAG)
with torch.no_grad():
    enc = st.encode(Lt["Z1"], wm.scale_actions(Lt["A"])[:, :-1])
cl2 = enc.act[1].reshape(-1, enc.act[1].shape[-1]); e2 = np.repeat(Lt["eid"], enc.act[1].shape[1])
cl3 = enc.act[2].reshape(-1, enc.act[2].shape[-1]); e3 = np.repeat(Lt["eid"], enc.act[2].shape[1])
box2 = ((enc.act[1] < st.prior_lo_2) | (enc.act[1] > st.prior_hi_2)).any(-1).float().mean().item()
box3 = ((enc.act[2] < st.prior_lo_3) | (enc.act[2] > st.prior_hi_3)).any(-1).float().mean().item()


def nn_loco(X, Xe, C, Ce):
    X = torch.as_tensor(X).reshape(-1, C.shape[-1]).float()
    d = torch.cdist(X, C)
    same = torch.as_tensor(np.asarray(Xe)[:, None] == np.asarray(Ce)[None, :])
    d[same] = math.inf
    return d.min(1).values.numpy()


with torch.no_grad():
    Ap = wm.scale_actions(torch.from_numpy(Aw).float())[:, :NP]                # logged TRAIN actions of the same windows, scaled
    log_m2 = M2(Ap); log_m3 = M3(log_m2)
    hold_m2 = M2(hc_all[:n, :NP]); hold_m3 = M3(hold_m2)
we = WE[:n]
ref2 = nn_loco(log_m2, np.repeat(we, 10), cl2, e2); ref3 = nn_loco(log_m3, np.repeat(we, 2), cl3, e3)
R["census_ref"] = dict(cloud2_n=int(len(cl2)), cloud3_n=int(len(cl3)), train_macros_outside_level2_box_frac=box2, train_macros_outside_level3_box_frac=box3,
                       logged_inimage_L2=q(ref2), logged_inimage_L3=q(ref3), hold_inimage_L2=q(nn_loco(hold_m2, np.repeat(we, 10), cl2, e2)), hold_inimage_L3=q(nn_loco(hold_m3, np.repeat(we, 2), cl3, e3)))
for a, r in res.items():
    d2 = nn_loco(r["m2"], np.repeat(we, 10), cl2, e2); d3 = nn_loco(r["m3"], np.repeat(we, 2), cl3, e3)
    d2b = nn_loco(r["m2"][:, 3:10], np.repeat(we, 7), cl2, e2)
    m2t = torch.as_tensor(r["m2"])
    rec = dict(wall_s=r["wall_s"], s_per_window=r["wall_s"] / n, gen_state=r["gen"],
               L2_nn=q(d2), L2_nn_ratio_median_vs_logged=float(np.median(d2) / np.median(ref2)), L2_knots11_30_nn_ratio=float(np.median(d2b) / np.median(ref2)),
               L2_frac_beyond_logged_q95=float((d2 > np.quantile(ref2, .95)).mean()), L2_outside_level2_box_frac=float(((m2t < st.prior_lo_2) | (m2t > st.prior_hi_2)).any(-1).float().mean()),
               L3_nn=q(d3), L3_nn_ratio_median_vs_logged=float(np.median(d3) / np.median(ref3)), L3_frac_beyond_logged_q95=float((d3 > np.quantile(ref3, .95)).mean()),
               cost=dict(c1=q(r["c1"]), c2=q(r["c2"]), c3=q(r["c3"])), iters=dict(it1=float(np.mean(r["it1"])), it2=float(np.mean(r["it2"])), it3=float(np.mean(r["it3"]))),
               train_in_sample=dict(acc_mae_0_2=band(r["plan_raw"], 0, 10, 0), yaw_mae_0_2=band(r["plan_raw"], 0, 10, 1), acc_mae_2_6=band(r["plan_raw"], 10, 30, 0), yaw_mae_2_6=band(r["plan_raw"], 10, 30, 1)))
    if len(r["resid"]):
        rec["decode_resid"] = q(r["resid"])
        rec["decode_bounds_any_frac"] = float((np.isclose(r["plan_raw"][:, 10:, :], H.to_raw(wm, lo1.numpy().reshape(1, 1, 2).astype(np.float64))) | np.isclose(r["plan_raw"][:, 10:, :], H.to_raw(wm, hi1.numpy().reshape(1, 1, 2).astype(np.float64)))).any(-1).any(-1).mean())
    if len(r["img_resid"]):
        rec["image_resid"] = q(r["img_resid"])
        rec["image_resid_max"] = float(np.max(r["img_resid"]))
        rec["knots11_30_equal_p"] = float(np.abs(r["plan_raw"][:, 10:30] - r["p2_raw"][:, 10:30]).max())
        rec["p2_as_knots1_10"] = dict(acc_mae_0_2=band(r["p2_raw"], 0, 10, 0), yaw_mae_0_2=band(r["p2_raw"], 0, 10, 1))
    if r.get("donor_shift"):
        rec["donor_shift"] = r["donor_shift"]
    R[a] = rec
hold = H.hold_plan(A0raw[:n], 30)
R["HOLD_train_in_sample"] = dict(acc_mae_0_2=band(hold, 0, 10, 0), yaw_mae_0_2=band(hold, 0, 10, 1), acc_mae_2_6=band(hold, 10, 30, 0), yaw_mae_2_6=band(hold, 10, 30, 1))
if "off" in res and "prim" in res:
    o, p = res["off"], res["prim"]
    R["C_L3ID_off_vs_prim"] = dict(m3_maxabs=float(np.abs(o["m3"] - p["m3"]).max()), c3_maxabs=float(np.abs(o["c3"] - p["c3"]).max()), gen_state_equal=o["gen"] == p["gen"],
                                   knots1_10_maxabs=float(np.abs(o["plan_raw"][:, :10] - p["plan_raw"][:, :10]).max()), m2_maxabs=float(np.abs(o["m2"] - p["m2"]).max()))
    R["paired_train_in_sample_prim_minus_off"] = dict(acc_mae_0_2=R["prim"]["train_in_sample"]["acc_mae_0_2"] - R["off"]["train_in_sample"]["acc_mae_0_2"],
                                                      yaw_mae_0_2=R["prim"]["train_in_sample"]["yaw_mae_0_2"] - R["off"]["train_in_sample"]["yaw_mae_0_2"],
                                                      acc_mae_2_6=R["prim"]["train_in_sample"]["acc_mae_2_6"] - R["off"]["train_in_sample"]["acc_mae_2_6"],
                                                      cost1_median_ratio=float(np.median(p["c1"]) / np.median(o["c1"])), cost2_median_ratio=float(np.median(p["c2"]) / np.median(o["c2"])))
if "rs" in res and "off" in res:
    R["sign_check_rs_minus_off_acc_0_2"] = R["rs"]["train_in_sample"]["acc_mae_0_2"] - R["off"]["train_in_sample"]["acc_mae_0_2"]
if "rs" in res and "prim" in res:
    R["sign_check_rs_minus_prim_acc_0_2"] = R["rs"]["train_in_sample"]["acc_mae_0_2"] - R["prim"]["train_in_sample"]["acc_mae_0_2"]
R["wall_total_s"] = time.time() - t0
R["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
fn = f"{OUT}/probe_{K}_{'_'.join(ARMS)}{'_SMOKE' if SMOKE else ''}.json"
with open(fn, "w", encoding="utf-8", newline="\n") as f:
    json.dump(R, f, indent=1, default=float)
print("WROTE", fn, round(R["wall_total_s"], 1), "s", flush=True)
print("ZZ PROBE DONE ZZ", flush=True)
