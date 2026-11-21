"""PREREG_V8F_PT3M: FLAT1 DEC-END with a four-frame per-target reader in the cost, fitted on N1053. PLANTEST3 / PT3N code is imported; one injection point: R3.make_cost.
Usage: python run_pt3m.py <n_workers>. Every write goes under this folder (control 6). Tier T0, CPU only. The .cmd wrapper writes PT3M.done last."""
import hashlib, json, os, sys, time, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import pt3m_lib as PM
from pt3m_lib import DN, DL, L3, P, PT3, THEORY
import run_plantest3 as RP
import plantest3_real as R3

RP.HERE = HERE                                     # choose_from_runs writes tune_choice.json next to this file
RP.RUNS = f"{HERE}/runs"
RP.ARMS = ("end",)
RUNS = RP.RUNS
TAGS = RP.TAGS
N1053 = f"{PT3}/n1053"
DECN_RUNS = f"{THEORY}/decinfo/decn/runs"
_CTX = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_cost(ctx, arm, form):
    assert arm == "end"
    return PM.ReaderCost(ctx.shrink_c if form == "shrink" else None, ctx.reader, ctx.dec)


R3.make_cost = make_cost


def load_ridge(tag):
    d = torch.load(f"{N1053}/runs/decoder_{tag}.pt", weights_only=False)
    dec = L3.Decoder(d["W1"], d["b1"], d["W2"], d["b2"], d["ystd"], n_samples=d.get("n_samples"))
    assert dec.digest() == d["sha256"]
    return dec


def get_ctx(tag):
    if tag not in _CTX:
        _CTX.clear()
        torch.set_num_threads(1)                   # decinfo_lib sets 4 at import, for the reader fit in the main process; planning workers stay at 1, as PT3N
        ctx = R3.prepare3(tag)
        ctx.dec = load_ridge(tag)
        ctx.reader = PM.load_reader(f"{RUNS}/reader_{tag}.pt")
        _CTX[tag] = ctx
    return _CTX[tag]


def guard_mutation():
    bad = 0
    for args in ((["a"], ["a"], [], ["e"], ["t"]), (["a"], ["b"], [], ["t"], ["t"]), (["a"], ["t"], [], ["e"], ["t"]), (["a"], ["b"], ["x"], ["e"], ["t"])):
        try:
            DN.split_guard_n(*args)
        except AssertionError:
            bad += 1
    assert bad == 4, "control 4 mutation: the guard let a violation through"


def clip_order():
    c141, ce = DN.clip_lists()
    order = list(c141["clips"]) + list(ce["clips"])
    assert len(order) == 1000 and len(set(order)) == 1000, "control 5: clip list"
    assert DN.available_clips(order) == 1000, "control 5: shards"
    return order


def decn_row(tag, reader, tgt):
    rows = json.load(open(f"{DECN_RUNS}/R0_{tag}_N1053.json", encoding="utf-8"))["rows"]
    return [r["r2"] for r in rows if r["reader"] == reader and r["m"] == 4 and r["target"] == tgt][0]


def fit_reader(tag):
    """Fit the reader of one training seed; writes reader_<tag>.pt and pre_<tag>.json (the gate, controls 1 to 5)."""
    t0 = time.time()
    order = clip_order()
    T = DN.load_tune_npz()
    key = f"z1_{tag}"
    X_tr, ci_tr = DN.load_train_arrays(order, 1000, key)
    vs, ys = [], []
    for c in order:
        z = np.load(f"{DN.LAT}/{c}.npz")
        vs.append(z["v"]); ys.append(z["yaw"])
    D, X, ids_tr = DN.assemble(T, key, X_tr, ci_tr, order, 1000, (np.concatenate(vs), np.concatenate(ys)))
    ev = {f[:-3] for f in os.listdir(f"{DL.W.SF}/cache/eval/clips") if f.endswith(".pt")}
    DN.split_guard_n(set(map(str, D.eid[:D.nf])), set(map(str, D.eid[D.nf:])), DL.es_ids_of(set(map(str, D.eid[:D.nf]))), ev, ids_tr)
    assert not (set(map(str, D.eid[:D.nf])) & set(map(str, D.eid[D.nf:])))
    Xs = DL.stack_ch(X, PM.M)
    yS = {k: v[:, PM.M - 1:] for k, v in DL.targets(D).items()}
    fit_ix, held_ix = np.arange(D.nf), np.arange(D.nf, D.n)
    # speed: ridge
    xf, yf = DL._flat(Xs, yS["speed"], fit_ix)
    Ws, bs, sd_v = PM.ridge_weights(xf, yf)
    xh, yh = DL._flat(Xs, yS["speed"], held_ix)
    p = (torch.as_tensor(xh, dtype=torch.double) @ Ws + bs).numpy() * sd_v
    r_speed = DL.r2(yh, p)
    n_speed = len(yh)
    del xf, yf
    # yaw rate: MLP, learning rate from the early-stopping loss
    (xa, ya), (xv, yv), (xh, yh) = DL._parts(D, Xs, yS["yaw"], yS["yaw"], False)
    yh64 = yh.numpy().astype(np.float64)
    fits = {}
    for lr in PM.LRS:
        DL.wait_ram()
        m = PM.train_net(xa, ya, xv, yv, lr=lr)
        r = DL.r2(yh64, PM.mlp_predict(m, xh).astype(np.float64))
        fits[lr] = dict(m=m, r2=r)
        log(tag, "yaw MLP lr", lr, f"es loss {m['es_loss']:.5f} held r2 {r:+.3f} epochs {m['epochs']}")
    best_lr = min(PM.LRS, key=lambda l: fits[l]["m"]["es_loss"])
    m = fits[best_lr]["m"]
    sd_y = float(torch.cat([ya, yv]).numpy().astype(np.float64).std())
    rd = PM.Reader(Ws, bs, m["net"], m["mu"], m["sd"], m["ym"], m["ys"], [sd_v, sd_y])
    PM.save_reader(rd, f"{RUNS}/reader_{tag}.pt")
    # beside: shuffled targets at the chosen learning rate (no bar; DECN's shuffled row at 1e-3 covers the pipeline)
    ysh = DN.derange_gen(D, DL.targets(D)["yaw"], seed=101)[:, PM.M - 1:]
    (xa2, ya2), (xv2, yv2), _ = DL._parts(D, Xs, yS["yaw"], ysh, False)
    rshuf = DL.r2(yh64, PM.mlp_predict(PM.train_net(xa2, ya2, xv2, yv2, lr=best_lr), xh).astype(np.float64))
    del Xs, X, X_tr, xa, xa2, xv, xv2
    # the gate on the harness's own tensors
    ctx = R3.prepare3(tag)
    e = np.asarray(ctx.eid)
    fit_ids, held_ids = L3.inner_split(e[ctx.tune_idx])
    iheld = np.array([i for i in ctx.tune_idx if e[i] in held_ids])
    assert len(iheld) == 238
    Zh = ctx.Z1[iheld]
    Xh = torch.cat([PM.stack_knots(Zh, k).unsqueeze(1) for k in range(3, 16)], 1)     # (238, 13, 512)
    Y = rd.read_raw(Xh)
    def r2_valid(y, p):
        ok = np.isfinite(y)
        return DL.r2(y[ok], p[ok]), int(ok.sum())
    rs, ns = r2_valid(ctx.vk[iheld, 3:16].reshape(-1), Y[..., 0].reshape(-1))
    ry, ny = r2_valid(ctx.yk[iheld, 3:16].reshape(-1), Y[..., 1].reshape(-1))
    rs12, ns12 = r2_valid(ctx.vk[iheld, 12], Y[:, 9, 0]); ry12, ny12 = r2_valid(ctx.yk[iheld, 12], Y[:, 9, 1])
    with torch.no_grad():
        a_sc = ctx.wm.scale_actions(torch.as_tensor(ctx.A[iheld][:, :12]).float())
        roll = ctx.st.rollout(1, ctx.Z1[iheld][:, 0], a_sc)
    Yr = rd.read_raw(roll[:, PM.KNOT_IDX, :].flatten(-2))
    rsr, nsr = r2_valid(ctx.vk[iheld, 12], Yr[:, 0]); ryr, nyr = r2_valid(ctx.yk[iheld, 12], Yr[:, 1])
    nan = dict(r2_speed=float("nan"), r2_yaw=float("nan"), n_speed=0, n_yaw=0)
    rec = dict(tag=tag, fit_set="N1053, four-frame reader", n_inner_fit_episodes=int(D.n_fit_episodes), n_inner_held_episodes=int(D.n_held_episodes), enc_L1=dict(r2_speed=rs, r2_yaw=ry, n_speed=ns, n_yaw=ny), enc_L2=nan,
               enc_L1_k12=dict(r2_speed=rs12, r2_yaw=ry12, n_speed=ns12, n_yaw=ny12), rollout_L1_k12_logged_actions=dict(r2_speed=rsr, r2_yaw=ryr, n_speed=nsr, n_yaw=nyr),
               ystd=[sd_v, sd_y], lr_grid={str(l): dict(es_loss=fits[l]["m"]["es_loss"], r2_held=fits[l]["r2"], epochs=fits[l]["m"]["epochs"]) for l in PM.LRS}, lr_chosen=best_lr, yaw_shuffled_at_chosen_lr=rshuf, reader_sha256=rd.digest(),
               planning_decoder_sha256=rd.digest(), gate_decoder_sha256=rd.digest(), planning_ystd=[sd_v, sd_y])
    rec["DECODER_WEAK_this_seed"] = bool(rs < L3.R2_SPEED_MIN or ry < L3.R2_YAW_MIN)
    t_s, t_y = decn_row(tag, "ridge", "speed"), decn_row(tag, "mlp", "yaw")
    rec["control1_tie_decn"] = {"speed_ridge_m4": dict(mine=r_speed, decn=t_s, abs_diff=abs(r_speed - t_s), passed=bool(abs(r_speed - t_s) <= 0.01)),
                                "yaw_mlp_m4_lr1e-3": dict(mine=fits[1e-3]["r2"], decn=t_y, abs_diff=abs(fits[1e-3]["r2"] - t_y), passed=bool(abs(fits[1e-3]["r2"] - t_y) <= 0.01))}
    rec["harness_vs_fitset_held_read"] = dict(speed_abs_diff=abs(rs - r_speed), yaw_abs_diff=abs(ry - fits[best_lr]["r2"]))
    rec["wall_s"] = time.time() - t0
    json.dump(rec, open(f"{RUNS}/pre_{tag}.json", "w", encoding="utf-8"), indent=1)
    log("fit", tag, f"speed {rs:+.3f} yaw {ry:+.3f} lr {best_lr}", "tie", {k: v["passed"] for k, v in rec["control1_tie_decn"].items()}, f"{time.time() - t0:.0f}s")
    return all(v["passed"] for v in rec["control1_tie_decn"].values())


def controls_2_3(ctx):
    """Reachability and harness tie, from this runner, on real SCORE tensors (first two batches)."""
    st, idx = ctx.st, np.asarray(ctx.score_idx[:2 * P.BATCH])
    z0 = ctx.z_now[idx].detach(); goal = ctx.z_path[idx].detach(); gend = ctx.z_fut[idx].detach()
    hold = R3.R2.hold_c_of(ctx)[idx].detach()
    A = hold.unsqueeze(1).repeat(1, 4, 1, 1) + 0.05 * torch.randn(len(idx), 4, *hold.shape[1:], generator=torch.Generator().manual_seed(3))
    def cg(cost, dbl=False):
        a = A.clone().requires_grad_(True)
        roll = cost.roll(st, 1, z0.unsqueeze(1).expand(-1, 4, -1), a)
        g_ = goal if cost.mode == "path" else gend
        if dbl:                                    # control 3 compares two exact expressions, so float32 rounding of near-zero costs must not enter
            roll, g_ = roll.double(), g_.double()
        c = cost.top_cost(st, roll, g_, 1, 1)
        return c, torch.autograd.grad(c.sum(), a)[0]
    c1, g1 = cg(PM.ReaderCost(None, ctx.reader, ctx.dec))
    c0, g0 = cg(PM.ReaderCost(0.0, ctx.reader, ctx.dec))
    rec = dict(grad_abs_mean_by_channel=[float(x) for x in g1.abs().mean((0, 1, 2))], grad_abs_max_shrink0=float(g0.abs().max()))
    rec["control2_passed"] = bool(all(x > 1e-6 for x in rec["grad_abs_mean_by_channel"]) and rec["grad_abs_max_shrink0"] == 0.0)
    dd = L3.Decoder(ctx.dec.W1.double(), ctx.dec.b1.double(), ctx.dec.W2, ctx.dec.b2, ctx.dec.ystd)
    c_l, g_l = cg(PM.ReaderCost(None, PM.LinearK12(dd, ctx.z_now.shape[1]), dd), dbl=True)
    c_o, g_o = cg(L3.CostSpec("end", None, dd), dbl=True)
    rec["cost_rel_diff_max"] = float(((c_l - c_o).abs() / c_o.abs().clamp_min(1e-9)).max())
    rec["grad_norm_rel_diff"] = float((g_l - g_o).norm() / g_o.norm())
    rec["control3_passed"] = bool(rec["cost_rel_diff_max"] <= 1e-4 and rec["grad_norm_rel_diff"] <= 1e-3)
    return rec


def path(job):
    if job[0] == "kin":
        return f"{RUNS}/ceil_{job[1]}.npz"
    if job[0] == "ctl":
        return f"{RUNS}/ctl_{job[1]}.json"
    return RP.path(job)


def work(job):
    if os.path.exists(path(job)):
        return job, 0.0, "skip"
    try:
        RP.wait_ram()
        ctx = get_ctx(job[1])
        t = time.time()
        if job[0] == "ctl":
            rec = controls_2_3(ctx)
            json.dump(rec, open(path(job), "w", encoding="utf-8"), indent=1)
            return job, time.time() - t, "ok" if rec["control2_passed"] and rec["control3_passed"] else "ERR controls 2/3 " + json.dumps(rec)
        if job[0] == "kin":
            sc = np.asarray(ctx.score_idx)
            hold = P.hold_plan(ctx.A[sc])
            for donor in (False, True):
                src = np.asarray(ctx.donor)[sc] if donor else sc
                np.savez(f"{RUNS}/kin_{job[1]}_{'rand' if donor else 'true'}.npz", plan_raw=PM.kin_plan_m4(ctx.reader, ctx.Z1[src][:, :13], hold), idx=sc)
            v0, w0 = ctx.vk[sc, 0], ctx.A[sc, 0, 1]
            y12 = ctx.reader.read_raw(PM.stack_knots(ctx.Z1[sc], 12))
            np.savez(path(job), oracle=PM.end_plan(v0, ctx.vk[sc, 12], w0, ctx.yk[sc, 12]), read=PM.end_plan(v0, y12[:, 0], w0, y12[:, 1]), idx=sc)
            return job, time.time() - t, "ok"
        if job[0] == "tune":
            _, _, arm, form, lam = job
            r = R3.unit3(ctx, "flat1", "true", ctx.tune_idx, lam, arm, form, seed=0, collect=False)
            m = P.window_metrics(r["plan_raw"], ctx.A[ctx.tune_idx], ctx.v0[ctx.tune_idx])
            r["acc_mae_mean"], r["yaw_mae_mean"] = np.array(m["acc_mae"].mean()), np.array(m["yaw_mae"].mean())
        else:
            _, _, kind, arm, form, goal, lam, pseed = job
            r = R3.unit3(ctx, kind, goal, ctx.score_idx, lam, arm, form, seed=pseed, collect=True)
        tmp = path(job) + ".tmp.npz"
        np.savez(tmp, **r)
        os.replace(tmp, path(job))
        return job, time.time() - t, "ok"
    except Exception:
        return job, 0.0, "ERR " + traceback.format_exc()[-800:]


def run_stage(pool, J, label):
    bad = 0
    for job, dt, st in pool.imap_unordered(work, J, chunksize=1):
        log(label, job, round(dt, 1), "s", st[:900])
        bad += st.startswith("ERR")
    return bad


def shas():
    return {p: hashlib.sha256(open(f"{p}/tune_choice.json", "rb").read()).hexdigest() for p in (PT3, N1053)}


if __name__ == "__main__":
    import multiprocessing as mp
    n = int(sys.argv[1])
    os.makedirs(RUNS, exist_ok=True)
    guard_mutation()
    base = shas()
    status = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"))
    ok = True
    for t in TAGS:
        if os.path.exists(f"{RUNS}/pre_{t}.json"):
            rec = json.load(open(f"{RUNS}/pre_{t}.json", encoding="utf-8"))
            ok &= all(v["passed"] for v in rec["control1_tie_decn"].values())
        else:
            RP.wait_ram()
            ok &= fit_reader(t)
    if not ok:
        log("control 1 failed: no planning")
        json.dump(dict(status, failed="control 1"), open(f"{HERE}/run_pt3m.status.json", "w", encoding="utf-8"))
        sys.exit(2)
    errs = 0
    with mp.get_context("spawn").Pool(n, maxtasksperchild=4) as pool:
        errs += run_stage(pool, [("ctl", t) for t in TAGS] + [("kin", t) for t in TAGS], "PRE")
        if errs == 0:
            ea = run_stage(pool, RP.jobs_a(), "A")
            errs += ea
            if ea == 0:
                choice, tab = RP.choose_from_runs()
                log("CHOICE", json.dumps(choice))
                B1, _, B3 = RP.jobs_b(choice, tab)
                errs += run_stage(pool, B1, "B1")
                errs += run_stage(pool, [j for j in B3 if j[2] == "flat1"], "B3")
    status.update(errors=errs, finished=time.strftime("%Y-%m-%d %H:%M:%S"), tune_choice_shas_unchanged=bool(shas() == base))
    json.dump(status, open(f"{HERE}/run_pt3m.status.json", "w", encoding="utf-8"), indent=1)
    assert status["tune_choice_shas_unchanged"], "control 6"
    sys.exit(1 if errs else 0)
