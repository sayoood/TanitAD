"""AMEND_HWMPLAN_5 section 8.2 controls. Pure checks (importable; each returns a dict with `ok`), the toy linear world (C-GRAD, C-PLANT (i)), and the per-chassis runner.
EVERY check has a deliberate-regression arm that must read RED (ok False): a control PASSES only when its green arm reads ok AND its regression arm reads RED.
The runner plans only: TRAIN windows (any arm; 16- and 64-window smokes) and, for the identity checks, base files read from disk (C-OFF5 copies, C-READBACK re-plans batch 0 under the BASE's own solver).
No SCORE window gets an RA pass here. usage:
  python amend5_controls.py run --chassis s0|s1 [--bases A0 P P23] [--cstrat]    writes <ROOT>/controls/controls_<k>.json
  python amend5_controls.py merge                                                  reads both, writes <ROOT>/controls/CONTROLS.json, exit 0 only if every control and every regression arm read as required
Tier T0. CPU only, OMP_NUM_THREADS=2."""
import argparse, json, os, shutil, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amend5_stage as ST
import ra_l1 as R
import lonplan_common as C
import numpy as np
import torch
import hwm_plan as H
import plantest_lib as P

ACC = R.ACC
KEYS_OFF = ("plan_raw", "a1_raw", "cost1", "iters1", "idx", "decode_resid", "l2_prim_raw")


# ---------------------------------------------------------------------------------------------- the toy linear world (C-GRAD, C-PLANT (i), the unit tests)
class ToyStack:
    """z_{t+1} = z_t + B a_t (a (..., H, 2) scaled), a LINEAR readout downstream, and the LIBRARY's own optimiser (HJEPAStack._optimize, an unbound function that reads no attribute of self)."""
    from tanitad.models.hjepa import HJEPAStack as _S
    _optimize = _S._optimize

    def __init__(self, B):
        self.B = B

    def rollout(self, level, z0, a, context=1):
        return z0.unsqueeze(-2) + torch.cumsum(a @ self.B.T, -2)


def toy_ctx(D=8, seed=0, lo=-3.0, hi=3.0, h1=12, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(D, 2, generator=g, dtype=dtype) * 0.3
    W = torch.randn(D, 1, generator=g, dtype=dtype)
    perm = np.random.RandomState(0).permutation(D)
    ra = R.Readout(mu=torch.zeros(D, dtype=dtype), sd=torch.ones(D, dtype=dtype), W=W, ym=torch.zeros(1, dtype=dtype), Wshuf=W[torch.as_tensor(perm)], perm=perm, sha256="toy", lam=0.0, path="toy")
    lo1, hi1 = torch.tensor([lo, lo], dtype=dtype), torch.tensor([hi, hi], dtype=dtype)

    def hold_c(A0):
        a = torch.as_tensor(np.asarray(A0), dtype=dtype)
        hc = torch.stack([a[:, 0] / ACC, a[:, 1]], -1).unsqueeze(1).repeat(1, 14, 1)
        return torch.max(torch.min(hc, hi1), lo1)

    return R.Ctx(st=ToyStack(B), h1=h1, lo1=lo1, hi1=hi1, ra=ra, hold_c=hold_c, ch1_scale=1.0, tag="toy")


def toy_world(ctx, n=20, seed=1, offset=0.4):
    """windows (z, A, eid) and a base plan file (plan_raw, a1_raw, l2_prim_raw, cost1, ...) over the toy world; a1 / l2 acceleration knots sit OFFSET above the held a0 so the pass has to move."""
    g = torch.Generator().manual_seed(seed)
    D = ctx.st.B.shape[0]
    z = torch.randn(n, 31, D, generator=g, dtype=torch.float64)
    A = np.zeros((n, 31, 2)); A[:, :, 0] = (torch.randn(n, 31, generator=g, dtype=torch.float64).numpy() * 0.3 + 0.2) * ACC
    A[:, :, 1] = torch.randn(n, 31, generator=g, dtype=torch.float64).numpy() * 0.02
    eid = np.array([f"e{i // 4}" for i in range(n)])
    a0 = A[:, 0, 0]
    a1 = np.zeros((n, 12, 2)); a1[:, :, 0] = a0[:, None] + offset * ACC * (1 + 0.3 * torch.randn(n, 12, generator=g, dtype=torch.float64).numpy()); a1[:, :, 1] = A[:, 0, 1][:, None]
    l2 = np.zeros((n, 12, 2)); l2[:, :, 0] = a0[:, None] + 0.5 * offset * ACC; l2[:, :, 1] = a1[:, :, 1]
    plan = np.zeros((n, 30, 2)); plan[:, :12] = a1; plan[:, 12:, 0] = a0[:, None] + 0.1; plan[:, 12:, 1] = a1[:, :1, 1]
    base = dict(plan_raw=plan, a1_raw=a1, l2_prim_raw=l2, cost1=np.abs(np.random.RandomState(0).randn(n)), iters1=np.full(n, 90.0), idx=np.arange(n), decode_resid=np.zeros(n))
    return dict(z=z.numpy(), A=A, eid=eid), base


# ---------------------------------------------------------------------------------------------- pure checks
def check_off_equal(new, ref, keys=KEYS_OFF):
    """C-OFF5: every data key of the base is bit-equal in the RA-OFF output (np.array_equal; max abs difference reported)."""
    d, ok = {}, True
    for k in keys:
        if k not in ref:
            continue
        if k not in new:
            d[k] = dict(equal=False, maxabs=float("nan"), missing=True); ok = False; continue
        eq = bool(np.array_equal(np.asarray(new[k]), np.asarray(ref[k])[:len(new[k])]))
        d[k] = dict(equal=eq, maxabs=float(np.abs(np.asarray(new[k], np.float64) - np.asarray(ref[k], np.float64)[:len(new[k])]).max()))
        ok = ok and eq
    return dict(ok=ok, keys=d)


def check_yaw_band(new, base, ks=R.KS):
    """C-YAW-ID / C-BAND-ID: yaw knots 1-30 and acceleration knots 11-30 of plan_raw equal the base's exactly (max |delta| = 0.0), and a1_raw / cost1 untouched."""
    p, q = np.asarray(new["plan_raw"], np.float64), np.asarray(base["plan_raw"], np.float64)
    d = dict(yaw_1_30=float(np.abs(p[:, :, 1] - q[:, :, 1]).max()), acc_11_30=float(np.abs(p[:, ks:, 0] - q[:, ks:, 0]).max()),
             a1_raw=float(np.abs(np.asarray(new["a1_raw"], np.float64) - np.asarray(base["a1_raw"], np.float64)).max()), cost1=float(np.abs(np.asarray(new["cost1"]) - np.asarray(base["cost1"])).max()),
             acc_1_10_moved=float(np.abs(p[:, :ks, 0] - q[:, :ks, 0]).max()))
    d["ok"] = bool(d["yaw_1_30"] == 0.0 and d["acc_11_30"] == 0.0 and d["a1_raw"] == 0.0 and d["cost1"] == 0.0)
    return d


def grad_of(st, ra_fn, z0, acc0, yaw, target, lo1, hi1, read_after=R.KS):
    cost, base = R.ra_pass(st, lo1, hi1, ra_fn, z0, acc0, yaw, target, read_after=read_after, return_cost_fn=True)
    a = base.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(cost(a).sum(), a)
    return g                                                                # (B, 1, h1, 2)


def check_grad(g, ks=R.KS):
    """C-GRAD: non-zero gradient on acceleration knots 1-10 (every knot, every window), EXACTLY 0.0 on knots 11-12 and on the yaw channel."""
    acc, yaw = g[:, 0, :, 0], g[:, 0, :, 1]
    d = dict(min_abs_knots_1_10=float(acc[:, :ks].abs().min()), max_abs_knots_11_12=float(acc[:, ks:].abs().max()), max_abs_yaw=float(yaw.abs().max()))
    d["ok"] = bool(d["min_abs_knots_1_10"] > 0.0 and d["max_abs_knots_11_12"] == 0.0 and d["max_abs_yaw"] == 0.0)
    return d


def check_plant_toy(ctx, world, seed=3, cost_bar=1e-4, mean_tol=2e-3, mutate=None):
    """C-PLANT (i), the inversion on the linear toy. A target formed from a known source acceleration sequence; the pass (the library's optimiser, S = 1, from HOLD_c) must reach the target read
    (cost <= cost_bar, the real-model bar) and, with a uniform per-knot response, its plan's mean must equal the source's mean (mean_tol). Analytic identities at 1e-6: the cost is 0 at the source's actions,
    and the closed-form flat solution reaches the source mean. mutate 'own_actions': the target is formed from the plan's OWN start (a pass that cannot move) -> the plan stays at HOLD_c -> RED."""
    ctx_, (win, base) = ctx, world
    st, ra = ctx_.st, ctx_.ra
    z0 = torch.from_numpy(win["z"][:, 0]).to(ra.W.dtype)
    hc = ctx_.hold_c(win["A"][:, 0])[:, :ctx_.h1]
    yaw = hc[:, :, 1]
    src = torch.from_numpy(base["a1_raw"][:, :, 0] / ACC).to(ra.W.dtype)
    ra_fn = lambda Dz: ra.read(Dz)
    with torch.no_grad():
        tgt = ra_fn(R.roll(st, z0, torch.stack([src if mutate != "own_actions" else hc[:, :, 0], yaw], -1).unsqueeze(1))[:, 0] - z0)
        tgt_src = ra_fn(R.roll(st, z0, torch.stack([src, yaw], -1).unsqueeze(1))[:, 0] - z0)
    acc, cst, nit, hit = R.ra_pass(st, ctx_.lo1, ctx_.hi1, ra_fn, z0, hc[:, :, 0], yaw, tgt)
    with torch.no_grad():
        rd = ra_fn(R.roll(st, z0, torch.stack([acc, yaw], -1).unsqueeze(1))[:, 0] - z0)
        c_at_src = ((ra_fn(R.roll(st, z0, torch.stack([src, yaw], -1).unsqueeze(1))[:, 0] - z0) - tgt_src) ** 2).abs().max()
        g = (ra_fn(R.roll(st, z0, torch.stack([hc[:, :, 0] + 1.0, yaw], -1).unsqueeze(1))[:, 0] - z0) - ra_fn(R.roll(st, z0, torch.stack([hc[:, :, 0], yaw], -1).unsqueeze(1))[:, 0] - z0))     # read gain for a +1 shift of every knot
        flat_shift = (tgt_src - ra_fn(R.roll(st, z0, torch.stack([hc[:, :, 0], yaw], -1).unsqueeze(1))[:, 0] - z0)) / g
        closed = hc[:, :R.KS, 0] + flat_shift[:, None]
        closed_read = ra_fn(R.roll(st, z0, torch.stack([torch.cat([closed, hc[:, R.KS:, 0]], 1), yaw], -1).unsqueeze(1))[:, 0] - z0)
    d = dict(max_cost=float(cst.max()), median_cost=float(cst.median()), max_read_err_vs_target=float((rd - tgt_src).abs().max()),
             plan_mean_minus_source_mean_max=float((acc[:, :R.KS].mean(1) - src[:, :R.KS].mean(1)).abs().max()), iterations=int(nit),
             analytic_cost_at_source_actions=float(c_at_src), analytic_closed_form_read_err=float((closed_read - tgt_src).abs().max()))
    d["ok"] = bool(d["max_cost"] <= cost_bar and d["plan_mean_minus_source_mean_max"] <= mean_tol and d["analytic_cost_at_source_actions"] <= 1e-6 and d["analytic_closed_form_read_err"] <= 1e-6)
    return d


def check_plant_real(costs, bar=1e-4):
    """C-PLANT (ii), real model: the pass reaches its target read: median final cost <= 1e-4 (scaled^2). `costs`: the TRAIN-stage ra_cost array of an RA arm."""
    c = np.asarray(costs, np.float64)
    d = dict(n=int(len(c)), median_cost=float(np.median(c)), q90_cost=float(np.quantile(c, 0.9)), max_cost=float(c.max()), bar=bar)
    d["ok"] = bool(d["median_cost"] <= bar)
    return d


def check_shuf_sign(row, separated=True):
    """C-SHUF: RA-SHUF - X on 0-2 s acc_mae. SCORE: separated-worse (the paired CI lies wholly above 0). TRAIN stage sign check (separated=False): the point difference is > 0. row: dict(delta, lo, hi)."""
    return dict(delta=row["delta"], lo=row["lo"], hi=row["hi"], separated=separated, ok=bool((row["lo"] > 0) if separated else (row["delta"] > 0)))


def check_mean_zero(hc_acc_raw, lo_raw, hi_raw):
    """C-MEAN, the identity: MEAN-src at a zero source departure (source = HOLD_c's own acceleration) equals HOLD_c exactly. The guard: a shift taken from the logged accelerations is refused."""
    out = R.mean_assembly(hc_acc_raw.copy(), hc_acc_raw, lo_raw, hi_raw)
    return dict(equal=bool(np.array_equal(out, hc_acc_raw)), maxabs=float(np.abs(out - hc_acc_raw).max()), ok=bool(np.array_equal(out, hc_acc_raw)))


def check_mean_guard(src_acc_raw, hc_acc_raw, lo_raw, hi_raw, logged_acc_raw, mutate=False):
    """The guard of C-MEAN: the shift handed in (override) must be the one derived from the source file. green: the derived shift passes; mutation (`logged`): a shift from the logged accelerations must RAISE.
    Returns ok True iff the call completed without a refusal."""
    sh = (logged_acc_raw.mean(1, keepdims=True) - hc_acc_raw.mean(1, keepdims=True)) if mutate else (src_acc_raw.mean(1, keepdims=True) - hc_acc_raw.mean(1, keepdims=True))
    try:
        R.mean_assembly(src_acc_raw, hc_acc_raw, lo_raw, hi_raw, shift_override=sh)
        return dict(ok=True, refused=False)
    except R.RefusedRun as e:
        return dict(ok=False, refused=True, why=str(e))


def check_l4(argv, declared):
    """C-L4-RA (argv half): True iff the wrapper accepts the argv against the declared arm. A mismatch, or an unnamed --ra-l1, must be refused."""
    try:
        R.check_declared(argv, declared)
        return dict(ok=True, refused=False)
    except R.RefusedRun as e:
        return dict(ok=False, refused=True, why=str(e))


def check_l4_built(ctx, base, win, mode, src, declared, mutate=None):
    """C-L4-RA (built half): the executed pass equals the declared one (source key, readout sha, mirror, perm)."""
    built = {}
    R.plan_arrays(ctx, base, win, mode, src, "none", mutate=mutate, built=built)
    try:
        R.check_built(built, declared)
        return dict(ok=True, refused=False, built=built)
    except R.RefusedRun as e:
        return dict(ok=False, refused=True, why=str(e))


def check_fresh_out_pre(tmp_dir, preplant):
    """C-L4-RA: a folder holding a pre-planted plans_*.npz must be refused (ok False)."""
    os.makedirs(tmp_dir, exist_ok=True)
    if preplant:
        np.savez(os.path.join(tmp_dir, "plans_hwm_oracle_s0.npz"), plan_raw=np.zeros(1))
    try:
        R.check_fresh_out(tmp_dir)
        return dict(ok=True, refused=False)
    except R.RefusedRun as e:
        return dict(ok=False, refused=True, why=str(e))


def check_ra_sha(path, k, flip_byte=False, tmp_dir=None):
    """C-RA-SHA: the readout's sha256 equals section 2.1's value; one changed byte refuses (ok False)."""
    p = path
    if flip_byte:
        raw = bytearray(open(path, "rb").read()); raw[len(raw) // 2] ^= 1
        p = os.path.join(tmp_dir, "readout_flip.pt"); open(p, "wb").write(bytes(raw))
    try:
        r = R.load_readout(p, k)
        return dict(ok=True, refused=False, sha256=r.sha256)
    except R.RefusedRun as e:
        return dict(ok=False, refused=True, why=str(e)[:160])


def check_in(record, strict_keys=None):
    """C-IN: every input the read uses is sha256-equal to its record. record: {name: (path, want_sha_or_prefix)}."""
    d, ok = {}, True
    for name, (p, want) in record.items():
        got = C.sha256_file(p) if os.path.exists(p) else None
        eq = bool(got is not None and got.startswith(want))
        d[name] = dict(path=p, sha256=got, want=want, equal=eq)
        ok = ok and eq
    return dict(ok=ok, inputs=d)


def check_train_iso(train_eids, score_eids, windows_path, expect_path):
    """C-TRAIN-ISO: no SCORE window is read by a TRAIN step: the TRAIN windows are LON-PLAN's selection set (path equal) and none of their clips is a SCORE episode."""
    inter = sorted(set(map(str, train_eids)) & set(map(str, score_eids)))
    return dict(n_train_windows=int(len(train_eids)), n_overlap_with_score_episodes=len(inter), path_equal=bool(os.path.abspath(windows_path) == os.path.abspath(expect_path)), ok=bool(not inter and os.path.abspath(windows_path) == os.path.abspath(expect_path)))


def verdict(green, regression):
    """A control PASSES when its green arm reads ok and its regression arm reads RED (ok False)."""
    return dict(green=green, regression=regression, passed=bool(green["ok"] and not regression["ok"]))


# ---------------------------------------------------------------------------------------------- the per-chassis runner
def _slice(d, n):
    return {k: (v[:n] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) > n else v) for k, v in d.items()}


def run_chassis(k, bases=("A0",), cstrat=False, root=None, log=print):
    t0 = time.time()
    root = root or ST.ROOT
    os.makedirs(f"{root}/controls", exist_ok=True)
    tmp = f"{root}/controls/tmp_{k}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    c = ST.CH[k]
    out = {"chassis": k, "controls": {}}
    ctl = out["controls"]
    ctx = R.load_ctx(k)
    log("ctx loaded", round(time.time() - t0, 1))
    wz = np.load(c["win"], allow_pickle=False)
    score_eids = sorted(set(wz["eid"].tolist()))
    # ---- C-RA-SHA
    rp = ST.readout_path(k)
    ctl["C-RA-SHA"] = verdict(check_ra_sha(rp, k), check_ra_sha(rp, k, flip_byte=True, tmp_dir=tmp))
    # ---- C-RA-FIT (+ both mutations)
    LT = torch.load(c["lat"], map_location="cpu", weights_only=False)
    rd = torch.load(rp, map_location="cpu", weights_only=False)
    ra_np = {x: np.asarray(rd["RA"][x], np.float64) for x in ("mu", "sd", "W", "ym")}
    fit = R.rebuild_ra_fit(LT, ctx.wm.scale_actions, float(rd["lambda"]))
    green = R.check_ra_fit(fit, ra_np, score_eids)
    disp = (wz["z"][0, R.KS].astype(np.float64) - wz["z"][0, 0].astype(np.float64))[None]
    y1 = ctx.wm.scale_actions(torch.from_numpy(wz["A"][0, :R.KS].astype(np.float64)).float()).numpy().astype(np.float64).mean(0)[:1][None]
    fit_plus = R.refit_ridge(np.vstack([fit["Dr"], disp]), np.vstack([fit["Yf"], y1]), float(rd["lambda"]), fit["clips"])
    reg_a = R.check_ra_fit(fit_plus, ra_np, score_eids)
    LT2 = dict(LT); e2 = list(LT["eid"]); e2[0] = score_eids[0]; LT2["eid"] = e2
    fit_pl = R.rebuild_ra_fit(LT2, ctx.wm.scale_actions, float(rd["lambda"]))
    reg_b = R.check_ra_fit(fit_pl, ra_np, score_eids)
    ctl["C-RA-FIT"] = verdict(green, reg_a)
    ctl["C-RA-FIT"]["regression_planted_eid"] = reg_b
    ctl["C-RA-FIT"]["passed"] = bool(ctl["C-RA-FIT"]["passed"] and not reg_b["ok"])
    del LT, LT2
    log("C-RA-FIT", green["n_pairs"], green["n_clips"], "max dW", green["max_abs_dW"])
    # ---- TRAIN smoke arrays (16 and 64 windows of LON-PLAN's selection set, A0's TRAIN plans)
    tw = np.load(ST.windows_of("train", k), allow_pickle=False)
    twin = {x: tw[x] for x in ("z", "A", "eid")}
    with np.load(ST.base_plan("train", "A0", k, 0), allow_pickle=False) as z:
        tb = {x: z[x] for x in z.files}
    n16, n64 = 16, 64
    win16, b16 = _slice(twin, n16), _slice(tb, n16)
    win64, b64 = _slice(twin, n64), _slice(tb, n64)
    # ---- C-TRAIN-ISO
    ctl["C-TRAIN-ISO"] = verdict(check_train_iso(twin["eid"], score_eids, ST.windows_of("train", k), f"{ST.L0}/sel_{k}/selection_windows.npz"),
                                 check_train_iso(np.concatenate([twin["eid"][:5], [score_eids[0]]]), score_eids, ST.windows_of("train", k), f"{ST.L0}/sel_{k}/selection_windows.npz"))
    # ---- C-GRAD on the toy stack and on the 16-window TRAIN smoke
    tctx = toy_ctx()
    twld, tbase = toy_world(tctx)
    tz = torch.from_numpy(twld["z"][:, 0]); thc = tctx.hold_c(twld["A"][:, 0])[:, :12]
    tsrc = torch.from_numpy(tbase["a1_raw"][:, :, 0] / ACC)
    tt = tctx.ra.read(R.roll(tctx.st, tz, torch.stack([tsrc, thc[:, :, 1]], -1).unsqueeze(1))[:, 0] - tz)
    g_toy = grad_of(tctx.st, tctx.ra.read, tz, thc[:, :, 0], thc[:, :, 1], tt, tctx.lo1, tctx.hi1)
    g_toy_bad = grad_of(tctx.st, tctx.ra.read, tz, thc[:, :, 0], thc[:, :, 1], tt, tctx.lo1, tctx.hi1, read_after=12)
    z16 = torch.from_numpy(win16["z"]).float(); hc16 = ctx.hold_c(win16["A"][:, 0])[:, :12]
    yb16 = torch.from_numpy(b16["a1_raw"][:, :12, 1] / ctx.ch1_scale).float(); sa16 = torch.from_numpy(b16["a1_raw"][:, :12, 0] / ACC).float()
    with torch.no_grad():
        t16 = ctx.ra.read(R.roll(ctx.st, z16[:, 0], torch.stack([sa16, yb16], -1).unsqueeze(1))[:, 0] - z16[:, 0])
    g_real = grad_of(ctx.st, ctx.ra.read, z16[:, 0], hc16[:, :, 0], yb16, t16, ctx.lo1, ctx.hi1)
    g_real_bad = grad_of(ctx.st, ctx.ra.read, z16[:, 0], hc16[:, :, 0], yb16, t16, ctx.lo1, ctx.hi1, read_after=12)
    ctl["C-GRAD"] = verdict(check_grad(g_toy), check_grad(g_toy_bad))
    ctl["C-GRAD"]["real_16_windows"] = verdict(check_grad(g_real), check_grad(g_real_bad))
    ctl["C-GRAD"]["passed"] = bool(ctl["C-GRAD"]["passed"] and ctl["C-GRAD"]["real_16_windows"]["passed"])
    # ---- C-PLANT (i)
    ctl["C-PLANT-i"] = verdict(check_plant_toy(tctx, (twld, tbase)), check_plant_toy(tctx, (twld, tbase), mutate="own_actions"))
    # ---- C-PLANT (ii): the pass reaches its target read on 64 TRAIN windows; regression = a pass with a 1-iteration budget
    new64, _ = R.plan_arrays(ctx, b64, win64, "l1", "l1")
    bad64, _ = R.plan_arrays(ctx, b64, win64, "l1", "l1", mutate="n_iters1")
    ctl["C-PLANT-ii-smoke"] = verdict(check_plant_real(new64["ra_cost"]), check_plant_real(bad64["ra_cost"]))
    # ---- C-YAW-ID / C-BAND-ID on the 64-window TRAIN smoke (all five pass modes), regressions: write12, write_yaw
    chk = {}
    for mode, src in (("l1", "l1"), ("neg", "l1"), ("shuf", "l1"), ("oracle", "none")):
        nw, _ = R.plan_arrays(ctx, b16, win16, mode, src)
        chk[mode] = check_yaw_band(nw, b16)
    green_all = dict(ok=all(v["ok"] for v in chk.values()), per_mode=chk)
    w12, _ = R.plan_arrays(ctx, b16, win16, "l1", "l1", mutate="write12")
    wy, _ = R.plan_arrays(ctx, b16, win16, "l1", "l1", mutate="write_yaw")
    r12, ry = check_yaw_band(w12, b16), check_yaw_band(wy, b16)
    ctl["C-YAW-ID"] = verdict(green_all, dict(ok=bool(r12["ok"] or ry["ok"]), write12=r12, write_yaw=ry))
    ctl["C-YAW-ID"]["regression_each_red"] = bool((not r12["ok"]) and (not ry["ok"]))
    ctl["C-YAW-ID"]["passed"] = bool(ctl["C-YAW-ID"]["passed"] and ctl["C-YAW-ID"]["regression_each_red"])
    # ---- C-OFF5: RA-OFF copy of every base file on disk (no pass), through the wrapper's file path; regression = a wrapper that runs the pass while the flag reads off (16 TRAIN windows)
    off_res = {}
    for base in bases:
        for stage in ("train", "score"):
            for s in ST.seeds_of(stage, "OFF") if stage == "train" else [0, 1]:
                bp = ST.base_plan(stage, base, k, s)
                if not os.path.exists(bp):
                    off_res[f"{stage}/{base}_s{s}"] = dict(skipped="no such base file")
                    continue
                with np.load(bp, allow_pickle=False) as z:
                    basearr = {x: z[x] for x in z.files}
                ww = np.load(ST.windows_of(stage, k), allow_pickle=False)
                new, built = R.plan_arrays(ctx, basearr, {x: ww[x] for x in ("z", "A")}, "off", "none", "none")
                off_res[f"{stage}/{base}_s{s}"] = check_off_equal(new, basearr)
                off_res[f"{stage}/{base}_s{s}"]["n"] = int(len(basearr["plan_raw"]))
    sol_off = R.run_arm(ST.build("train", "OFF", "A0", k, out=f"{tmp}/off_wrapper")["argv"], ST.build("train", "OFF", "A0", k)["declared"], ctx=ctx, windows_src=ST.windows_of("train", k),
                        windows_sha=json.load(open(f"{ST.L0}/sel_{k}/lam_0/SOLVER.json", encoding="utf-8"))["windows_sha256"])
    with np.load(f"{tmp}/off_wrapper/plans_hwm_oracle_s0.npz", allow_pickle=False) as z:
        wrapped = {x: z[x] for x in z.files}
    off_res["wrapper_file_train_A0_s0"] = check_off_equal(wrapped, tb)
    green = dict(ok=all(v.get("ok", True) for v in off_res.values()), files=off_res)
    mut, _ = R.plan_arrays(ctx, b16, win16, "off", "none", "none", mutate="off_runs_pass")
    ctl["C-OFF5"] = verdict(green, check_off_equal(mut, b16))
    # ---- C-MEAN
    hc_raw = ctx.hold_c(win64["A"][:, 0])[:, :R.KS, 0].numpy().astype(np.float64) * ACC
    lo_raw, hi_raw = float(ctx.lo1[0]) * ACC, float(ctx.hi1[0]) * ACC
    src64 = b64["a1_raw"][:, :R.KS, 0]
    logged = win64["A"][:, :R.KS, 0]
    mz = check_mean_zero(hc_raw, lo_raw, hi_raw)
    mg_g, mg_r = check_mean_guard(src64, hc_raw, lo_raw, hi_raw, logged), check_mean_guard(src64, hc_raw, lo_raw, hi_raw, logged, mutate=True)
    ctl["C-MEAN"] = verdict(dict(ok=bool(mz["ok"] and mg_g["ok"]), zero=mz, guard=mg_g), dict(ok=mg_r["ok"], guard_mutation=mg_r))
    # ---- C-L4-RA
    dec = ST.build("train", "RAL1", "A0", k)
    argv_ok, dec_ok = dec["argv"], dec["declared"]
    argv_l2 = [t for t in argv_ok]; argv_l2[argv_l2.index("--ra-l1") + 1] = "l2"
    argv_unnamed = [t for i, t in enumerate(argv_ok) if t != "--ra-l1" and not (i > 0 and argv_ok[i - 1] == "--ra-l1")]
    bl = check_l4_built(ctx, b16, win16, "l1", "l1", dec_ok)
    bl_bad = check_l4_built(ctx, dict(b16, l2_prim_raw=b16["a1_raw"] + 0.1), win16, "l1", "l1", dec_ok, mutate="wrong_src")
    pre_ok, pre_bad = check_fresh_out_pre(f"{tmp}/fresh_ok", False), check_fresh_out_pre(f"{tmp}/fresh_bad", True)
    l4g = dict(ok=bool(check_l4(argv_ok, dec_ok)["ok"] and bl["ok"] and pre_ok["ok"]), argv=check_l4(argv_ok, dec_ok), built=bl, fresh=pre_ok)
    l4r = dict(argv_mismatch=check_l4(argv_l2, dec_ok), argv_unnamed=check_l4(argv_unnamed, dec_ok), built_mismatch=bl_bad, preplanted=pre_bad)
    ctl["C-L4-RA"] = dict(green=l4g, regression=l4r, passed=bool(l4g["ok"] and not any(v["ok"] for v in l4r.values())))
    # ---- C-IN
    rec = {"ckpt": (c["ck"], c["cksha"]), "windows": (c["win"], c["wsha"]), "train_latents": (c["lat"], c["latsha"]), "readout": (rp, ST.RA_SHA[k]), "spec": (ST.SPEC, ST.SPEC_SHA)}
    for base in bases:
        for s in (0, 1):
            bp = ST.base_plan("score", base, k, s)
            if os.path.exists(bp):
                want = json.load(open(os.path.join(os.path.dirname(bp), "SOLVER.json"), encoding="utf-8"))["plans"][os.path.basename(bp)]["sha256"]
                rec[f"base_{base}_s{s}"] = (bp, want)
    bad_rec = dict(rec); bad_rec["readout"] = (rp, "0" * 64)
    ctl["C-IN"] = verdict(check_in(rec), check_in(bad_rec))
    cs = json.load(open(f"{ST.L0}/score/A0_{k}/SOLVER.json", encoding="utf-8"))["code_sha256"]
    now = C.code_of_record_sha256()
    ctl["C-IN"]["code_of_record_equal_A0_solver"] = bool(all(now[f] == cs[f] for f in cs) and all(now[f] == v and cs[f] == v for f, v in ST.CODE_SHA_FULL.items()))
    ctl["C-IN"]["passed"] = bool(ctl["C-IN"]["passed"] and ctl["C-IN"]["code_of_record_equal_A0_solver"])
    # ---- C-READBACK of every base file used (batch 0 from its SOLVER.json, bit-equal) + the regression arm
    rb = {}
    for base in bases:
        for s in (0, 1):
            bp = ST.base_plan("score", base, k, s)
            if os.path.exists(bp):
                rb[f"{base}_s{s}"] = R.readback_base(base, ST.base_dir("score", base, k), s)
                log("readback", base, s, rb[f"{base}_s{s}"]["equal"], round(time.time() - t0, 1))
    reg = R.readback_base("A0", ST.base_dir("score", "A0", k), 0, lam_override=10.0)
    ctl["C-READBACK"] = verdict(dict(ok=bool(rb) and all(v["equal"] for v in rb.values()), files=rb), dict(ok=bool(reg["equal"]), lam_override=reg))
    # ---- C-STRAT (code path is chassis independent: s0 only)
    if cstrat:
        import subprocess
        wd = f"{tmp}/cstrat"
        r = subprocess.run([sys.executable, f"{ST.LONPLAN_BUILD}/lonplan_cstrat.py", "--windows", ST.base_dir("score", "A0", "s0"), "--a", ST.base_plan("score", "A0", "s0", 0), "--b", ST.base_plan("score", "A0", "s0", 1), "--workdir", wd, "--n-boot", "200"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ))
        ctl["C-STRAT"] = dict(passed=bool(r.returncode == 0), rc=r.returncode, tail=r.stdout[-1200:], note="lonplan_cstrat.py: green = every non-STRATEGIC cell bit-equal to the code of record and the STRATEGIC cells computed; regression arm = the record's eval module reads PAUSED and is flagged; B = 200 here")
    out["wall_s"] = round(time.time() - t0, 1)
    out["red"] = sorted(n for n, v in ctl.items() if not v["passed"])
    out["ok"] = not out["red"]
    shutil.rmtree(tmp, ignore_errors=True)
    json.dump(out, open(f"{root}/controls/controls_{k}.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    return out


def merge(root=None):
    root = root or ST.ROOT
    parts = {k: json.load(open(f"{root}/controls/controls_{k}.json", encoding="utf-8")) for k in ("s0", "s1")}
    red = sorted({f"{k}:{n}" for k, p in parts.items() for n in p["red"]})
    res = dict(ok=not red, red=red, chassis={k: dict(ok=p["ok"], wall_s=p["wall_s"], passed={n: v["passed"] for n, v in p["controls"].items()}) for k, p in parts.items()})
    json.dump(res, open(f"{root}/controls/CONTROLS.json", "w", encoding="utf-8", newline="\n"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--chassis", required=True); r.add_argument("--bases", nargs="+", default=["A0"]); r.add_argument("--cstrat", action="store_true")
    sp.add_parser("merge")
    a = ap.parse_args()
    if a.cmd == "run":
        o = run_chassis(a.chassis, tuple(a.bases), a.cstrat)
        print("RED:", o["red"] or "none", "wall", o["wall_s"])
        sys.exit(0)
    m = merge()
    print(json.dumps(m, indent=1))
    sys.exit(0 if m["ok"] else 1)


if __name__ == "__main__":
    main()
