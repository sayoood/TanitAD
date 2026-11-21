"""PREREG_V8F_PLANTEST3: the real-data side. Reuses plantest2_real.prepare2 (models, 1,976 windows, TRAIN prior, TUNE / SCORE split, HOLD_c) and adds the per-knot targets, the decoders, the gate, the SHRINK factor and one-unit planning.
Speed per knot = native_tracks[clip][2 (t - 15 + k), 3]; yaw rate per knot = the logged yaw rate A_raw[:, k, 1] (IMPL section 2). The decoder fit sees TUNE rows only."""
import json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "plantest2"))
import numpy as np
import torch
import plantest3_lib as L3
import plantest2_lib as Q
import plantest2_real as R2
import plantest_lib as P
import plantest_real as R

RUNS = f"{HERE}/runs"
SHRINK_JSON = f"{HERE}/../wmact/RESULT_WMACT_SHRINK.json"
C_EXPECT = {"M2b_s0": 0.49619369970619825, "M2b_s1": 0.525653520686631}
FIT_LOG = []                                # row-index sets handed to the decoder fitter (control 4)


def shrink_c(tag):
    d = json.load(open(SHRINK_JSON))
    c = float(d["rows"][f"{tag}/L1/k1"]["c_train"])
    assert c == C_EXPECT[tag], (tag, c)
    return c


def prepare3(tag):
    ctx = R2.prepare2(tag)
    ev = P.load_latents(R.EVAL, tag)
    assert len(ev["eid"]) == ctx.n and torch.equal(ev["Z1"][:, 0], ctx.z_now) and torch.equal(ev["Z1"][:, 12], ctx.z_fut)
    ctx.Z1 = ev["Z1"]                                      # (n, 16, D)
    ctx.z_path = ev["Z1"][:, 1:13].contiguous()            # encoded true-future latents at knots 1..12
    TRK = torch.load(R.TRACKS, weights_only=False)["tracks"]
    vk = np.full((ctx.n, 16), np.nan)
    for i, (e, t) in enumerate(zip(ev["eid"], ev["t"])):
        g = int(t) - 15
        r = TRK.get(str(e))
        if r is None or g < 0:
            continue
        r = np.asarray(r)
        for k in range(16):
            row = 2 * (g + k)
            if row < len(r):
                vk[i, k] = float(r[row, 3])
    ctx.vk = vk
    ctx.yk = ctx.A[:, :, 1].copy()
    assert np.array_equal(np.isnan(ctx.vk[:, 0]), np.isnan(ctx.v0)) and np.allclose(ctx.vk[:, 0][~np.isnan(ctx.v0)], ctx.v0[~np.isnan(ctx.v0)])
    ctx.shrink_c = shrink_c(tag)
    ctx.dec = None
    return ctx


def fit_rows(ctx, rows, with_l2=True):
    """Fit the decoder on the given rows only (logged, so control 4 can assert the rows)."""
    rows = np.asarray(rows)
    FIT_LOG.append(rows.copy())
    return L3.fit_decoder(ctx.st, ctx.Z1[rows], ctx.vk[rows], ctx.yk[rows], with_l2=with_l2)


def get_dec(ctx):
    """The planning decoder: all TUNE rows of this ctx (deterministic)."""
    if ctx.dec is None:
        ctx.dec = fit_rows(ctx, ctx.tune_idx)
    return ctx.dec


def gate(ctx):
    """The decoder gate (IMPL section 3): the gate decoder is fitted on the inner-fit TUNE episodes, scored on the inner-held-out TUNE episodes. Returns a json-able record."""
    st = ctx.st
    e = np.asarray(ctx.eid)
    fit_ids, held_ids = L3.inner_split(e[ctx.tune_idx])
    ifit = np.array([i for i in ctx.tune_idx if e[i] in fit_ids]); iheld = np.array([i for i in ctx.tune_idx if e[i] in held_ids])
    assert set(e[ifit]).isdisjoint(set(e[iheld])) and set(ifit) | set(iheld) == set(ctx.tune_idx)
    dec = fit_rows(ctx, ifit)
    rec = dict(tag=ctx.tag, n_tune_windows=int(len(ctx.tune_idx)), n_inner_fit_windows=int(len(ifit)), n_inner_held_windows=int(len(iheld)), n_inner_fit_episodes=len(fit_ids), n_inner_held_episodes=len(held_ids),
               ystd_speed=float(dec.ystd[0]), ystd_yaw=float(dec.ystd[1]), ridge=L3.RIDGE,
               tune_speed_samples_missing=int(np.isnan(ctx.vk[ctx.tune_idx]).sum()), tune_samples_total=int(ctx.vk[ctx.tune_idx].size))
    Zh, vh, yh = ctx.Z1[iheld], ctx.vk[iheld], ctx.yk[iheld]
    for lv in (1, 2):
        rs, ry, ns, ny = L3.heldout_r2(st, dec, lv, Zh, vh, yh)
        rec[f"enc_L{lv}"] = dict(r2_speed=rs, r2_yaw=ry, n_speed=ns, n_yaw=ny)
    rs, ry, ns, ny = L3.heldout_r2(st, dec, 1, Zh[:, 12:13], vh[:, 12:13], yh[:, 12:13])
    rec["enc_L1_k12"] = dict(r2_speed=rs, r2_yaw=ry, n_speed=ns, n_yaw=ny)
    # rollout at k = 12 under the logged actions (gate only: these actions reach no planner)
    with torch.no_grad():
        a_sc = ctx.wm.scale_actions(torch.as_tensor(ctx.A[iheld][:, :12]).float())
        z12 = st.rollout(1, ctx.Z1[iheld][:, 0], a_sc)[:, -1:]
    rs, ry, ns, ny = L3.heldout_r2(st, dec, 1, z12, vh[:, 12:13], yh[:, 12:13])
    rec["rollout_L1_k12_logged_actions"] = dict(r2_speed=rs, r2_yaw=ry, n_speed=ns, n_yaw=ny)
    rec["DECODER_WEAK_this_seed"] = bool(rec["enc_L1"]["r2_speed"] < L3.R2_SPEED_MIN or rec["enc_L1"]["r2_yaw"] < L3.R2_YAW_MIN)
    rec["gate_decoder_sha256"] = dec.digest()
    pdec = get_dec(ctx)
    rec["planning_decoder_sha256"] = pdec.digest()
    rec["planning_ystd"] = [float(pdec.ystd[0]), float(pdec.ystd[1])]
    return rec


def kin_unit(ctx, idx, donor=False):
    """KIN-CEILING plans (raw, n x 12 x 2) on the windows idx from the decoded true-future (or the donor's) latents at knots 0..12."""
    dec = get_dec(ctx)
    src = np.asarray(idx) if not donor else np.asarray(ctx.donor)[np.asarray(idx)]
    return L3.kin_plan(dec, ctx.Z1[src][:, :13])


def make_cost(ctx, arm, form):
    """arm: latent | end | path; form: plain | shrink."""
    c = ctx.shrink_c if form == "shrink" else None
    if arm == "latent":
        return None if c is None else L3.CostSpec("latent", c)
    return L3.CostSpec(arm, c, get_dec(ctx))


def unit3(ctx, kind, gname, idx, lam, arm, form, seed=0, collect=True):
    t = time.time()
    r = L3.plan_all3(ctx.st, kind, gname, ctx.z_now, ctx.z_fut, ctx.z_path, ctx.donor, R2.hold_c_of(ctx), idx=idx, seed=seed, collect=collect, lam=lam, seed_mode="hold", cost=make_cost(ctx, arm, form))
    r["plan_raw"] = R.to_raw(ctx, r.pop("actions"))
    r["idx"] = np.asarray(idx)
    r["wall_s"] = np.array(time.time() - t)
    return r


def tune_lambdas3(ctx, kind, arm, form, lams, tune_idx, seed=0):
    """TUNE grid under G-TRUE; reads only the windows in tune_idx. Returns {lam: {acc_mae, yaw_mae}}."""
    out = {}
    for lam in lams:
        r = unit3(ctx, kind, "true", tune_idx, lam, arm, form, seed=seed, collect=False)
        m = P.window_metrics(r["plan_raw"], ctx.A[tune_idx], ctx.v0[tune_idx])
        out[lam] = dict(acc_mae=float(m["acc_mae"].mean()), yaw_mae=float(m["yaw_mae"].mean()))
    return out
