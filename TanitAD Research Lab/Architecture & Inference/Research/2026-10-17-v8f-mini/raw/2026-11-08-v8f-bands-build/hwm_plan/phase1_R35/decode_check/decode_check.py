"""Decode-vs-image diagnostic for HWM-PLAN phase 1 (R35). CPU. Imports code of record, edits nothing.
usage: decode_check.py <s0|s1> <nbatches_of_64_planned_windows>"""
import sys, os, json, time
sys.dont_write_bytecode = True
H_DIR = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/hwm_plan"
sys.path.insert(0, H_DIR)
import numpy as np, torch
import hwm_plan as H, hwm_data as D, plantest_lib as P, plantest2_lib as Q
from types import SimpleNamespace
torch.set_num_threads(2)
S = sys.argv[1]; NB = int(sys.argv[2]); T = f"R35_{S}"
B = "F:/TanitAD-scratch/hwm_plan"; OUT = f"{B}/decode_check"
O = f"{B}/phase1_R35/{S}"
t0 = time.time()
# --- build exactly as run_hwm.prepare (ckpt, TRAIN latents prior fit, geometry, solver defaults = phase1_R35.sh)
m, cfg, ck = D.load_model(f"C:/Users/Admin/t2v8f_pulls/ladder_{T}/ckpt_last.pt")
wm, st = m.wm, m.wm.stack
Z, A = D.trainset_from_latents_pt(f"{B}/latents_train_{T}.pt", T)
D.fit_priors(st, wm, Z, A)
geom = H.Geometry.from_stack(st, wm, goal_steps=None, stitch="decode")
cfg_s = H.SolverCfg(seed_mode="prior", lam=0.0, S=P.S_CAND, n_iters=P.N_ITERS, decode_lam=0.0, decode_iters=300, plan_ctx=1)
lo, hi = st.prior_lo_1, st.prior_hi_1
std = Q.prior_std_orig(st)
c2 = geom.c2
print("geom", geom, "lo", lo.flatten().tolist(), "hi", hi.flatten().tolist(), flush=True)
mac = st.macro_encoders[0]


def menc(p):  # p (b, n*c2, 2) scaled -> macro (b, n, E), exactly decode_macro enc
    b = p.shape[0]; n = p.shape[1] // c2
    e = st.act_embed(p)
    return mac(e.reshape(b, n, c2, e.shape[-1]))


def hold_of(a0raw, n):  # (b,2) raw -> (b, n, 2) scaled, clipped (hold_full)
    return H.hold_full(wm, a0raw, lo, hi, n)


def resid_of(tgt, p):
    with torch.no_grad():
        sc = tgt.pow(2).sum(-1).mean(-1).clamp_min(1e-12)
        return ((menc(p) - tgt).pow(2).sum(-1).mean(-1) / sc).sqrt()


def q(x):
    x = np.asarray(x, np.float64)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), q10=float(np.quantile(x, .1)), q90=float(np.quantile(x, .9)), min=float(x.min()), max=float(x.max()))


def bounds_hit(p, tol=1e-6):
    at = (p <= lo + tol) | (p >= hi - tol)
    return dict(frac_rows_any_elem_at_bound=float(at.flatten(-2).any(-1).float().mean()), frac_elems_at_bound=float(at.float().mean()))


def decode(tgt, ref, iters):
    c = H.SolverCfg(seed_mode="prior", lam=0.0, decode_lam=0.0, decode_iters=iters, decode_lr=0.1)
    out_p, out_r = [], []
    for s in range(0, tgt.shape[0], 512):
        p, r = H.decode_macro(st, tgt[s:s + 512], ref[s:s + 512], lo, hi, std, c)
        out_p.append(p); out_r.append(r)
    return torch.cat(out_p), torch.cat(out_r)


R = dict(tag=T, geom=str(geom))
# --- TRAIN primitive triples (ok windows only; load_latents already ok-filters)
Lt = P.load_latents(f"{B}/latents_train_{T}.pt", T)
A_sc = wm.scale_actions(Lt["A"])[:, :-1]            # (N, 15, 2) scaled, as fit_priors
eid_tr = Lt["eid"]
N = A_sc.shape[0]
a0 = Lt["A"][:, 0]
offs = [0, 3, 6, 9, 12]
trip = torch.stack([A_sc[:, k:k + 3] for k in offs], 1)           # (N, 5, 3, 2)
rng = np.random.RandomState(0)
ueid = np.unique(eid_tr); perm = rng.permutation(len(ueid)); held = set(ueid[perm[:len(ueid) // 5]])
is_held = np.array([e in held for e in eid_tr])
print("train windows", N, "episodes", len(ueid), "held-out eps", len(held), "held windows", int(is_held.sum()), flush=True)
with torch.no_grad():
    mac_all = menc(trip.reshape(N * 5, 3, 2)).reshape(N, 5, -1)  # in-image macros, (N,5,E)
E_dim = mac_all.shape[-1]
R["E_dim"] = int(E_dim); R["n_train_windows"] = int(N)
# --- STEP 1: in-image targets, n=1 triple, ref = window a0 HOLD (planner style)
sel = rng.permutation(N * 5)[:3000]
wi, oi = sel // 5, sel % 5
tgt = mac_all[wi, oi][:, None]
true_p = trip[wi, oi]
ref_hold = hold_of(a0[wi], 3)
p300, r300 = decode(tgt, ref_hold, 300)
R["step1_inimage_n1_holdA0_300it"] = dict(resid=q(r300.numpy()), bounds=bounds_hit(p300))
true_c = torch.max(torch.min(true_p, hi), lo)
c = H.SolverCfg(seed_mode="prior", lam=0.0, decode_lam=0.0, decode_iters=300)
pc, rc = H.decode_macro(st, tgt, true_c, lo, hi, std, c)
R["control_start_at_truth_300it"] = dict(resid=q(rc.numpy()), truth_outside_box_frac=float(((true_p < lo) | (true_p > hi)).any(-1).any(-1).float().mean()))
R["control_zero_iter_at_truth"] = q(resid_of(tgt, true_c).numpy())
g = torch.Generator().manual_seed(1)
rnd = torch.randn(3000, 1, E_dim, generator=g)
rnd = rnd / rnd.pow(2).sum(-1, keepdim=True).sqrt() * tgt.pow(2).sum(-1, keepdim=True).sqrt().mean()
_, rr = decode(rnd, ref_hold, 300)
R["regression_random_macros_inimage_scale_300it"] = q(rr.numpy())
rawA = Lt["A"]
pa = torch.stack([rawA[wi[i], max(int(oi[i]) * 3 - 1, 0)] for i in range(len(wi))])  # raw action just before the triple (a0 for the first)
p_loc, r_loc = decode(tgt, hold_of(pa, 3), 300)
R["step1_inimage_n1_prevaction_hold_300it"] = dict(resid=q(r_loc.numpy()), bounds=bounds_hit(p_loc))
wi5 = rng.permutation(N)[:600]
tgt5 = mac_all[wi5]; ref5 = hold_of(a0[wi5], 15)
p5, r5 = decode(tgt5, ref5, 300)
R["step1_inimage_n5chain_holdA0_300it"] = dict(resid=q(r5.numpy()), bounds=bounds_hit(p5))
p3k, r3k = decode(tgt, ref_hold, 3000)
R["step4_inimage_n1_holdA0_3000it"] = dict(resid=q(r3k.numpy()), bounds=bounds_hit(p3k))
p3k5, r3k5 = decode(tgt5, ref5, 3000)
R["step4_inimage_n5chain_holdA0_3000it"] = dict(resid=q(r3k5.numpy()), bounds=bounds_hit(p3k5))
print("step1 done", round(time.time() - t0), "s", {k: R[k]["resid"] if "resid" in R[k] else R[k] for k in ("step1_inimage_n1_holdA0_300it", "control_start_at_truth_300it")}, flush=True)

# --- STEP 2: planned macros. Re-run the phase-1 hwm_oracle seed-0 plan for NB batches, capture macro2 at stitch (code of record, hook only)
wd = np.load(f"{O}/windows.npz", allow_pickle=False)
ws = SimpleNamespace(z=torch.from_numpy(wd["z"]), A=wd["A"], eid=wd["eid"])
planner = H.HWMPlanner(wm, geom, cfg_s)
goal = H.OracleGoal(ws.z[:, geom.goal_frame], geom.goal_frame)
cap = []
orig_stitch = H.stitch


def cap_stitch(st_, geom_, a1, macro2, hold_c, *r, **k):
    cap.append(dict(macro2=macro2.detach().clone(), hc=hold_c.detach().clone()))
    return orig_stitch(st_, geom_, a1, macro2, hold_c, *r, **k)


H.stitch = cap_stitch
banked = np.load(f"{O}/plans_hwm_oracle_s0.npz")
res = []
for b in range(NB):
    idx = np.arange(b * P.BATCH, min(len(ws.eid), (b + 1) * P.BATCH))
    r = planner.plan_windows("hier", goal, ws.z[:, 0], ws.A[:, 0], idx=idx, seed=0, batch0=b)
    res.append(r); print("planned batch", b, round(time.time() - t0), "s", flush=True)
H.stitch = orig_stitch
idx_all = np.concatenate([r["idx"] for r in res])
resid_re = np.concatenate([r["decode_resid"] for r in res])
resid_bank = banked["decode_resid"][idx_all]
R["step2_repro"] = dict(n_windows=int(len(idx_all)), rerun_vs_banked_resid_maxabsdiff=float(np.abs(resid_re - resid_bank).max()), banked_median_same_windows=float(np.median(resid_bank)), rerun_median=float(np.median(resid_re)),
                        plan_raw_maxabsdiff=float(np.abs(np.concatenate([r["plan_raw"] for r in res]) - banked["plan_raw"][idx_all]).max()),
                        banked_all994=dict(median=float(np.median(banked["decode_resid"])), q10=float(np.quantile(banked["decode_resid"], .1)), q90=float(np.quantile(banked["decode_resid"], .9))))
j0 = geom.n_op // c2; j1 = int(np.ceil(geom.n_traj / c2))
PM = torch.cat([c_["macro2"][:, j0:j1] for c_ in cap])          # (n, 7, E)
HC = torch.cat([c_["hc"] for c_ in cap])
refP = HC[:, j0 * c2:j1 * c2]
pp, rp = decode(PM, refP, 300)
R["step2_planned_redecode_300it"] = dict(resid=q(rp.numpy()), bounds=bounds_hit(pp), maxabsdiff_vs_banked=float(np.abs(rp.numpy() - resid_bank).max()))
pp3, rp3 = decode(PM, refP, 3000)
R["step2_planned_redecode_3000it"] = dict(resid=q(rp3.numpy()), bounds=bounds_hit(pp3))
rndP = torch.randn(PM.shape, generator=g)
rndP = rndP / rndP.pow(2).sum(-1, keepdim=True).sqrt() * PM.pow(2).sum(-1, keepdim=True).sqrt()
_, rrP = decode(rndP, refP, 300)
R["regression_random_macros_planned_scale_300it"] = q(rrP.numpy())
# in-image chain of 7 macros, same windows, same positions, same HOLD ref: true logged eval actions
Ae = wm.scale_actions(torch.from_numpy(ws.A).float())[:, :-1]    # (994, 30, 2)
trip_e = torch.stack([Ae[:, 3 * j:3 * j + 3] for j in range(10)], 1)
with torch.no_grad():
    mac_e = menc(trip_e.reshape(-1, 3, 2)).reshape(len(ws.eid), 10, -1)
tg7 = mac_e[idx_all][:, j0:j1]
p7, r7 = decode(tg7, refP, 300)
R["step1b_inimage_eval_windows_same_positions_n7_300it"] = dict(resid=q(r7.numpy()), bounds=bounds_hit(p7), note="true logged eval actions of the SAME 6 s windows, triples j0..j1, ref = same HOLD_c as the planner")
p73, r73 = decode(tg7, refP, 3000)
R["step4b_inimage_eval_windows_n7_3000it"] = dict(resid=q(r73.numpy()), bounds=bounds_hit(p73))

# --- STEP 3: nearest-neighbour distance to the TRAIN in-image cloud (train episodes not held out)
cloud = mac_all[~is_held].reshape(-1, E_dim)
held_cloud = mac_all[is_held].reshape(-1, E_dim)


def nn(X, C):
    out = []
    for s in range(0, len(X), 1024):
        out.append(torch.cdist(X[s:s + 1024], C).min(1).values)
    return torch.cat(out)


Xp = PM.reshape(-1, E_dim)
d_pl = nn(Xp, cloud).numpy()
d_ho = nn(held_cloud, cloud).numpy()
Xe7 = tg7.reshape(-1, E_dim); d_ev7 = nn(Xe7, cloud).numpy()
qs = [.05, .1, .25, .5, .75, .9, .95, .99]
tab = lambda d: {str(k): float(np.quantile(d, k)) for k in qs}
nrm = lambda X: X.pow(2).sum(-1).sqrt().numpy()
R["step3_nn"] = dict(cloud_n=int(len(cloud)), planned=dict(n=int(len(d_pl)), qt=tab(d_pl), rel_to_norm_median=float(np.median(d_pl / nrm(Xp)))),
                     heldout_train_eps=dict(n=int(len(d_ho)), qt=tab(d_ho), rel_to_norm_median=float(np.median(d_ho / nrm(held_cloud)))),
                     eval_inimage_same_positions=dict(n=int(len(d_ev7)), qt=tab(d_ev7), rel_to_norm_median=float(np.median(d_ev7 / nrm(Xe7)))),
                     median_ratio_planned_over_heldout=float(np.median(d_pl) / np.median(d_ho)),
                     median_ratio_planned_over_evalinimage_samepos=float(np.median(d_pl) / np.median(d_ev7)),
                     quantile_ratio_planned_over_heldout={str(k): float(np.quantile(d_pl, k) / np.quantile(d_ho, k)) for k in qs},
                     norm=dict(cloud_median=float(np.median(nrm(cloud))), planned_median=float(np.median(nrm(Xp))), heldout_median=float(np.median(nrm(held_cloud)))),
                     frac_planned_farther_than_heldout_q95=float((d_pl > np.quantile(d_ho, .95)).mean()))
R["wall_s"] = time.time() - t0
json.dump(R, open(f"{OUT}/result_{S}.json", "w"), indent=1)
print(json.dumps(R, indent=1))
