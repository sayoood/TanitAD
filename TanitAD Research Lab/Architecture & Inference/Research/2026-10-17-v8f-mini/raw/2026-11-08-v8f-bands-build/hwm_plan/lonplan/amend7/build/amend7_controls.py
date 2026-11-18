"""AMEND_HWMPLAN_7 section 8 controls. Each control has a deliberate-regression arm that must read RED: a control PASSES only when its green arm reads ok AND its regression arm reads RED.
  python amend7_controls.py fit --chassis k    8.1 (FIT stage): C-PAIRS7, C-ISO7, C-CURVE7, K-control, C-FOLD7, C-SEL7, C-CONST7, C-ZERO7, C-WM7, PLANT, SHUF-T  -> controls/fit_<k>.json   (needs the crossfit outputs)
  python amend7_controls.py run --chassis k    8.2 (SCORE stage): AMEND_5's controls on the amend7 folders + C-GRAD7, C-PLANT7, C-RA7-SHA        -> controls/controls_<k>.json (needs RA7_FIT_<k>.json and both readouts)
  python amend7_controls.py merge --which fit|run   controls/CONTROLS_FIT.json | controls/CONTROLS.json; exit 0 only if every control and every regression arm reads as required
Tier T0. CPU only, OMP_NUM_THREADS=2."""
import amend7_env  # noqa: F401  (first)
import argparse, copy, json, os, shutil, sys, time
sys.dont_write_bytecode = True
import numpy as np
import torch
from amend7_env import ROOT, ST, A6, R, readout_path7, sha_B
import amend5_controls as K5
import fit_ra6 as F
import mlp7 as M
import pairs7 as PR
import fit7 as FT
import ra7_l1

verdict = K5.verdict
ACC = R.ACC


# ---------------------------------------------------------------------------------------------- 8.1 FIT stage
def fit_controls(k, log=print):
    t0 = time.time()
    T = FT.table(k)
    y, fo, oc, part = T["y"], T["fold"], T["oc"], T["part"]
    n_clips = int(oc.max() + 1)
    L = lambda n: np.load(f"{FT.FITD}/{n}_{k}.npz")
    J = lambda n: json.load(open(f"{FT.FITD}/{n}_{k}.json", encoding="utf-8"))
    A = {n: L(n) for n in ("RA", "MLPA", "MLPB", "KCTL", "SHUFT", "ZERO", "CONST", "PLANT")}
    rec = {n: J(n) for n in ("RA", "MLPA", "MLPB", "KCTL", "SHUFT", "ZERO", "CONST", "PLANT")}
    pr = json.load(open(f"{ROOT}/PAIRS7_{k}.json", encoding="utf-8"))
    ctl = {}
    tmp = f"{ROOT}/controls/tmp_fit_{k}"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    # ---- C-PAIRS7: the build asserted the table equals fit_ra6.pairs bit for bit (recorded); the table's sha256 is re-verified on load; every chunk's sha256 now; the counts
    Jc = json.load(open(f"{A6}/lat6/CHUNKS.json", encoding="utf-8"))
    g = dict(M.check_chunk_files(Jc, f"{A6}/lat6"))
    g["build_record_ok"] = bool(pr["C_PAIRS7"]["ok"])
    g["counts_ok"] = bool(len(y) == PR.N_PAIRS and n_clips == PR.N_CLIPS and pr["n"] == PR.N_PAIRS)
    g["ok"] = bool(g["ok"] and g["build_record_ok"] and g["counts_ok"])
    ch0 = Jc["chunks"][0]
    raw = bytearray(open(f"{A6}/lat6/{ch0['file']}", "rb").read()); raw[len(raw) // 2] ^= 1
    open(f"{tmp}/{ch0['file']}", "wb").write(bytes(raw))
    ctl["C-PAIRS7"] = verdict(g, M.check_chunk_files(dict(chunks=[ch0]), tmp))
    # ---- C-ISO7
    c141 = json.load(open(PR.C141, encoding="utf-8"))["clips"]
    c6 = json.load(open(PR.CLIPS6, encoding="utf-8"))["clips"]
    import hashlib
    weids = sorted(set(np.load(ST.CH[k]["win"], allow_pickle=False)["eid"].tolist()))
    order = list(c141) + list(c6[:PR.N_NEW])
    gi = M.check_iso(order, c141, c6, weids)
    gi["order_sha_equal_record"] = bool(hashlib.sha256(json.dumps(order).encode()).hexdigest() == pr["clip_order_sha256"])
    gi["ok"] = bool(gi["ok"] and gi["order_sha_equal_record"])
    bad = list(order); bad[5] = weids[0]
    ctl["C-ISO7"] = verdict(gi, M.check_iso(bad, c141, c6, weids))
    # ---- C-CURVE7: RA's cross-fit reproduces the 1,000 row of FIT_CURVE; regression: the fold map from the clips sorted by id
    ref = json.load(open(f"{A6}/readout/FIT_CURVE_{k}.json", encoding="utf-8"))["1000"]
    mine = M.curve_row(A["RA"]["pred"], y, oc, n_clips, 1000, 0)
    fo_bad = F.fold_map(n_clips, T["ep"])
    bad_res = M.crossfit(M.ridge_learner(), "D", T["z0"], T["z10"], y, fo_bad, oc)
    ctl["C-CURVE7"] = verdict(M.check_curve(mine, ref), M.check_curve(M.curve_row(bad_res["pred"], y, oc, n_clips, 1000, 0), ref))
    ctl["C-CURVE7"]["green"]["row"] = mine
    # ---- K-control: KCTL within 1e-4 of RA; regression: the targets taken from each pair's swap partner
    lams = [f["lam"] for f in rec["RA"]["folds"]]
    mis = M.crossfit(M.lind_learner(lams), "z0z1", T["z0"], T["z10"], y[part], fo, oc)
    ctl["K-CONTROL"] = verdict(M.check_k(A["KCTL"]["pred"], A["RA"]["pred"], y), M.check_k(mis["pred"], A["RA"]["pred"], y))
    # ---- C-FOLD7: every arm's recorded folds; regression: fold f's rows added to its own fit
    allf = {n: rec[n]["folds"] for n in ("RA", "MLPA", "MLPB", "KCTL", "SHUFT", "ZERO", "CONST")}
    allf["PLANT_MLP"], allf["PLANT_RA"] = rec["PLANT"]["folds_MLP"], rec["PLANT"]["folds_RA"]
    gf = {n: M.check_fold(v, n_clips) for n, v in allf.items()}
    mut = M.crossfit(M.ridge_learner(), "D", T["z0"], T["z10"], y, fo, oc, mutate="test_in_train")
    ctl["C-FOLD7"] = verdict(dict(ok=all(v["ok"] for v in gf.values()), per_arm=gf), M.check_fold(mut["folds"], n_clips))
    # ---- C-SEL7: every selection fit's validation clips lie inside its outer fold's training clips, argmin; regression: scored on the outer test fold / a non-argmin choice
    sels = {n: M.check_sel(rec[n]["folds"]) for n in ("MLPA", "MLPB", "SHUFT", "ZERO")}
    sels["PLANT_MLP"] = M.check_sel(rec["PLANT"]["folds_MLP"])
    badrec = copy.deepcopy(rec["MLPA"]["folds"])
    badrec[0]["selection"]["fits"][0]["val_clips"] = list(badrec[0]["test_clips"])
    badrec2 = copy.deepcopy(rec["MLPA"]["folds"])
    badrec2[1]["selection"]["chosen"] = [999, 0.5]
    rs, rs2 = M.check_sel(badrec), M.check_sel(badrec2)
    ctl["C-SEL7"] = verdict(dict(ok=all(v["ok"] for v in sels.values()), per_arm=sels), dict(ok=bool(rs["ok"] or rs2["ok"]), scored_on_test=rs, not_argmin=rs2))
    ctl["C-SEL7"]["regression_each_red"] = bool(not rs["ok"] and not rs2["ok"])
    ctl["C-SEL7"]["passed"] = bool(ctl["C-SEL7"]["passed"] and ctl["C-SEL7"]["regression_each_red"])
    # ---- C-CONST7
    ctl["C-CONST7"] = verdict(M.check_const(A["CONST"]["pred"], y, fo), M.check_const(np.full(len(y), y.mean()), y, fo))
    # ---- C-ZERO7: ZERO's dR is exactly 0.0; regression: ZERO read with the real z10 (its fold models read [z0, F])
    models = torch.load(f"{FT.FITD}/ZERO_{k}_models.pt", map_location="cpu", weights_only=False)
    dR_bad = np.zeros(len(y))
    for f in range(5):
        te = fo == f
        dR_bad[te] = M.predict(models[f], M.build_input("z0z1", T["z0"][te], T["F_true"][te])) - M.predict(models[f], M.build_input("z0z1", T["z0"][te], T["F_alt"][te]))
    ctl["C-ZERO7"] = verdict(M.check_zero(A["ZERO"]["dR"]), M.check_zero(dR_bad))
    # ---- C-WM7: r_wm(RA) CI above 0 and the roll equals the 12-knot rollout; regression: F_true and F_alt exchanged
    x = A["RA"]["dR"]
    bt = M.Boot(oc, n_clips, dict(y=y, yy=y * y, x=x, xx=x * x, xy=x * y, nx=-x, nxx=x * x, nxy=-x * y))
    pt, lo, hi = bt.run(lambda t: np.array([M.t_corr(t, "x", "xx", "xy"), M.t_corr(t, "nx", "nxx", "nxy")]), FT.B_BOOT, 0)
    wm = PR._model(k)
    idx = np.arange(0, len(y), len(y) // 64)[:64]
    z0t, at = torch.from_numpy(T["z0"][idx]), torch.from_numpy(T["act"][idx])
    with torch.no_grad():                                                   # float32 kernels differ by ~1e-6 between a 10-knot and a 12-knot call shape; the causality check runs in float64 (exact)
        roll32 = M.check_roll_causal(lambda z, a: R.roll(wm.stack, z, a), lambda z, a: wm.stack.rollout(1, z.unsqueeze(1), a), z0t, at, tol=1e-4)
        st64 = wm.stack.double()
        roll_ok = M.check_roll_causal(lambda z, a: R.roll(st64, z, a), lambda z, a: st64.rollout(1, z.unsqueeze(1), a), z0t.double(), at.double())
    roll_ok["float32_max_abs"] = roll32["max_abs"]; roll_ok["float32_within_1e-4"] = roll32["ok"]
    gw = M.check_wm_sign([lo[0], hi[0]])
    gw.update(roll=roll_ok, r_wm_RA=float(pt[0]), ok=bool(gw["ok"] and roll_ok["ok"]))
    ctl["C-WM7"] = verdict(gw, M.check_wm_sign([lo[1], hi[1]]))
    # ---- PLANT: P1, P2 for MLP-A; regression P3: KCTL on y_plant must miss P1
    P = A["PLANT"]; yp = P["y_plant"]
    btp = M.Boot(oc, n_clips, dict(y=yp, yy=yp * yp, e2_M=(P["pred_MLP"] - yp) ** 2, e2_R=(P["pred_RA"] - yp) ** 2))
    pp, plo, phi = btp.run(lambda t: np.array([M.t_r2(t, "e2_M") - M.t_r2(t, "e2_R")]), FT.B_BOOT, 0)
    dr2 = dict(point=float(pp[0]), lo=float(plo[0]), hi=float(phi[0]))
    r2o = rec["PLANT"]["r2_oracle"]
    gp = M.check_plant(M.r2(P["pred_MLP"], yp), r2o, dr2, M.M_K[k])
    ctl["PLANT"] = verdict(gp, M.check_plant_p1(M.r2(P["pred_KCTL"], yp), r2o))
    # ---- SHUF-T
    ctl["SHUF-T"] = verdict(M.check_shuf(M.r2(A["SHUFT"]["pred"], A["SHUFT"]["y_used"])), M.check_shuf(M.r2(A["MLPA"]["pred"], y)))
    shutil.rmtree(tmp, ignore_errors=True)
    out = dict(chassis=k, controls=ctl, wall_s=round(time.time() - t0, 1), tier="T0", evidence="MEASURED")
    out["red"] = sorted(n for n, v in ctl.items() if not v["passed"]); out["ok"] = not out["red"]
    FT.jdump(out, f"{ROOT}/controls/fit_{k}.json")
    return out


# ---------------------------------------------------------------------------------------------- 8.2 SCORE stage
def toy_ctx7(D=8, seed=0, dtype=torch.float64):
    """The toy linear world of AMEND_5's controls with a fixed random readout of RA7's shape (MLP on [z0, z0 + D])."""
    ctx = K5.toy_ctx(D, seed, dtype=dtype)
    torch.manual_seed(seed + 11)
    net = M.make_net(2 * D, 16)
    ctx.ra = ra7_l1.MLPReadout(net, torch.zeros(2 * D), torch.ones(2 * D), "toy", "toy", dtype)
    return ctx


def grad_of7(st, ra, z0, acc0, yaw, target, lo1, hi1, read_after=R.KS, detach=False):
    ra_fn = lambda Dz: ra.read(Dz.detach() if detach else Dz, z0)
    cost, base = R.ra_pass(st, lo1, hi1, ra_fn, z0, acc0, yaw, target, read_after=read_after, return_cost_fn=True)
    a = base.clone().requires_grad_(True)
    try:
        (g,) = torch.autograd.grad(cost(a).sum(), a, allow_unused=True)
    except RuntimeError:
        g = None
    return torch.zeros_like(a) if g is None else g


def check_plant_toy7(ctx, world, mutate=None, cost_bar=1e-6, move_bar=1e-3):
    """C-PLANT7 (i): a toy linear world, a fixed random RA7-shaped readout, a target formed from a known action sequence (the base's level-1 knots). The pass (the library's optimiser, S = 1, from HOLD_c) must reach the
    target read (final cost <= cost_bar) and move off HOLD_c; the analytic identity (the cost is 0 at the source's actions) holds at 1e-6. mutate 'own_actions': the target is formed from HOLD_c's own actions -> the pass cannot move -> RED."""
    st, ra = ctx.st, ctx.ra
    win, base = world
    dt = ra.dtype
    z0 = torch.from_numpy(win["z"][:, 0]).to(dt)
    hc = ctx.hold_c(win["A"][:, 0])[:, :ctx.h1]
    yaw = hc[:, :, 1]
    src = torch.from_numpy(base["a1_raw"][:, :, 0] / ACC).to(dt)
    ra_fn = lambda Dz: ra.read(Dz, z0)
    with torch.no_grad():
        tgt_src = ra_fn(R.roll(st, z0, torch.stack([src, yaw], -1).unsqueeze(1))[:, 0] - z0)
        tgt = ra_fn(R.roll(st, z0, torch.stack([hc[:, :, 0] if mutate == "own_actions" else src, yaw], -1).unsqueeze(1))[:, 0] - z0)
    acc, cst, nit, hit = R.ra_pass(st, ctx.lo1, ctx.hi1, ra_fn, z0, hc[:, :, 0], yaw, tgt)
    with torch.no_grad():
        rd = ra_fn(R.roll(st, z0, torch.stack([acc, yaw], -1).unsqueeze(1))[:, 0] - z0)
        c_at_src = ((ra_fn(R.roll(st, z0, torch.stack([src, yaw], -1).unsqueeze(1))[:, 0] - z0) - tgt_src) ** 2).abs().max()
    moved = float((acc[:, :R.KS] - hc[:, :R.KS, 0]).abs().max())
    d = dict(max_cost=float(cst.max()), max_read_err_vs_target=float((rd - tgt).abs().max()), moved_off_hold_c=moved, analytic_cost_at_source_actions=float(c_at_src), iterations=int(nit))
    d["ok"] = bool(d["max_cost"] <= cost_bar and moved > move_bar and d["analytic_cost_at_source_actions"] <= 1e-6)
    return d


def check_ra7_sha(path, k, sha, flip=False, tmp=None):
    p = path
    if flip:
        raw = bytearray(open(path, "rb").read()); raw[len(raw) // 2] ^= 1
        p = os.path.join(tmp, "readout_flip.pt"); open(p, "wb").write(bytes(raw))
    try:
        r = ra7_l1.load_readout7(p, k, sha)
        return dict(ok=True, refused=False, sha256=r.sha256)
    except R.RefusedRun as e:
        return dict(ok=False, refused=True, why=str(e)[:160])


def run_chassis7(k, cstrat=False, root=None, log=print):
    t0 = time.time()
    root = root or ST.ROOT
    os.makedirs(f"{root}/controls", exist_ok=True)
    tmp = f"{root}/controls/tmp_{k}"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp, exist_ok=True)
    c = ST.CH[k]
    out = {"chassis": k, "controls": {}}
    ctl = out["controls"]
    J7 = json.load(open(f"{root}/RA7_FIT_{k}.json", encoding="utf-8"))
    ctx = R.load_ctx(k)
    log("ctx loaded", round(time.time() - t0, 1))
    wz = np.load(c["win"], allow_pickle=False)
    score_eids = sorted(set(wz["eid"].tolist()))
    pa, pb = readout_path7(k), readout_path7(k, "_B")
    # ---- C-RA7-SHA (readouts A and B; one changed byte refuses)
    ga, gb = check_ra7_sha(pa, k, J7["sha256"]), check_ra7_sha(pb, k, J7["sha256_B"])
    ctl["C-RA7-SHA"] = verdict(dict(ok=bool(ga["ok"] and gb["ok"]), A=ga, B=gb), check_ra7_sha(pa, k, J7["sha256"], flip=True, tmp=tmp))
    # ---- TRAIN smoke arrays
    tw = np.load(ST.windows_of("train", k), allow_pickle=False)
    twin = {x: tw[x] for x in ("z", "A", "eid")}
    with np.load(ST.base_plan("train", "A0", k, 0), allow_pickle=False) as z:
        tb = {x: z[x] for x in z.files}
    win16, b16 = K5._slice(twin, 16), K5._slice(tb, 16)
    win64, b64 = K5._slice(twin, 64), K5._slice(tb, 64)
    ctl["C-TRAIN-ISO"] = verdict(K5.check_train_iso(twin["eid"], score_eids, ST.windows_of("train", k), f"{ST.L0}/sel_{k}/selection_windows.npz"),
                                 K5.check_train_iso(np.concatenate([twin["eid"][:5], [score_eids[0]]]), score_eids, ST.windows_of("train", k), f"{ST.L0}/sel_{k}/selection_windows.npz"))
    # ---- C-GRAD7: toy stack and the 16-window TRAIN smoke; regressions: RA7 detached in the cost, RA7 reading after 12 actions
    tctx = toy_ctx7()
    twld, tbase = K5.toy_world(tctx)
    dt = tctx.ra.dtype
    tz = torch.from_numpy(twld["z"][:, 0]).to(dt); thc = tctx.hold_c(twld["A"][:, 0])[:, :12]
    tsrc = torch.from_numpy(tbase["a1_raw"][:, :, 0] / ACC).to(dt)
    with torch.no_grad():
        tt = tctx.ra.read(R.roll(tctx.st, tz, torch.stack([tsrc, thc[:, :, 1]], -1).unsqueeze(1))[:, 0] - tz, tz)
    G = lambda **kw: grad_of7(tctx.st, tctx.ra, tz, thc[:, :, 0], thc[:, :, 1], tt, tctx.lo1, tctx.hi1, **kw)
    ctl["C-GRAD7"] = verdict(K5.check_grad(G()), dict(ok=bool(K5.check_grad(G(detach=True))["ok"] or K5.check_grad(G(read_after=12))["ok"]), detached=K5.check_grad(G(detach=True)), after12=K5.check_grad(G(read_after=12))))
    ctl["C-GRAD7"]["regression_each_red"] = bool(not K5.check_grad(G(detach=True))["ok"] and not K5.check_grad(G(read_after=12))["ok"])
    z16 = torch.from_numpy(win16["z"]).float(); hc16 = ctx.hold_c(win16["A"][:, 0])[:, :12]
    yb16 = torch.from_numpy(b16["a1_raw"][:, :12, 1] / ctx.ch1_scale).float(); sa16 = torch.from_numpy(b16["a1_raw"][:, :12, 0] / ACC).float()
    with torch.no_grad():
        t16 = ctx.ra.read(R.roll(ctx.st, z16[:, 0], torch.stack([sa16, yb16], -1).unsqueeze(1))[:, 0] - z16[:, 0], z16[:, 0])
    G16 = lambda **kw: grad_of7(ctx.st, ctx.ra, z16[:, 0], hc16[:, :, 0], yb16, t16, ctx.lo1, ctx.hi1, **kw)
    r16 = verdict(K5.check_grad(G16()), dict(ok=bool(K5.check_grad(G16(detach=True))["ok"] or K5.check_grad(G16(read_after=12))["ok"])))
    ctl["C-GRAD7"]["real_16_windows"] = r16
    ctl["C-GRAD7"]["passed"] = bool(ctl["C-GRAD7"]["passed"] and ctl["C-GRAD7"]["regression_each_red"] and r16["passed"])
    # ---- C-PLANT7 (i) toy, (ii) the TRAIN smoke's median final cost (regression: a 1-iteration budget)
    ctl["C-PLANT7-i"] = verdict(check_plant_toy7(tctx, (twld, tbase)), check_plant_toy7(tctx, (twld, tbase), mutate="own_actions"))
    new64, _ = R.plan_arrays(ctx, b64, win64, "l1", "l1")
    bad64, _ = R.plan_arrays(ctx, b64, win64, "l1", "l1", mutate="n_iters1")
    ctl["C-PLANT7-ii-smoke"] = verdict(K5.check_plant_real(new64["ra_cost"]), K5.check_plant_real(bad64["ra_cost"]))
    # ---- C-YAW-ID / C-BAND-ID on the TRAIN smoke (all pass modes), regressions: write12, write_yaw
    chk = {}
    for mode, src in (("l1", "l1"), ("neg", "l1"), ("shuf", "l1"), ("oracle", "none")):
        nw, _ = R.plan_arrays(ctx, b16, win16, mode, src)
        chk[mode] = K5.check_yaw_band(nw, b16)
    green_all = dict(ok=all(v["ok"] for v in chk.values()), per_mode=chk)
    w12, _ = R.plan_arrays(ctx, b16, win16, "l1", "l1", mutate="write12")
    wy, _ = R.plan_arrays(ctx, b16, win16, "l1", "l1", mutate="write_yaw")
    r12, ry = K5.check_yaw_band(w12, b16), K5.check_yaw_band(wy, b16)
    ctl["C-YAW-ID"] = verdict(green_all, dict(ok=bool(r12["ok"] or ry["ok"]), write12=r12, write_yaw=ry))
    ctl["C-YAW-ID"]["regression_each_red"] = bool((not r12["ok"]) and (not ry["ok"]))
    ctl["C-YAW-ID"]["passed"] = bool(ctl["C-YAW-ID"]["passed"] and ctl["C-YAW-ID"]["regression_each_red"])
    # ---- C-OFF5
    off_res = {}
    for stage in ("train", "score"):
        for s in ([0] if stage == "train" else [0, 1]):
            bp = ST.base_plan(stage, "A0", k, s)
            if not os.path.exists(bp):
                off_res[f"{stage}/A0_s{s}"] = dict(skipped="no such base file"); continue
            with np.load(bp, allow_pickle=False) as z:
                basearr = {x: z[x] for x in z.files}
            ww = np.load(ST.windows_of(stage, k), allow_pickle=False)
            new, built = R.plan_arrays(ctx, basearr, {x: ww[x] for x in ("z", "A")}, "off", "none", "none")
            off_res[f"{stage}/A0_s{s}"] = K5.check_off_equal(new, basearr)
            off_res[f"{stage}/A0_s{s}"]["n"] = int(len(basearr["plan_raw"]))
    R.run_arm(ST.build("train", "OFF", "A0", k, out=f"{tmp}/off_wrapper")["argv"], ST.build("train", "OFF", "A0", k)["declared"], ctx=ctx, windows_src=ST.windows_of("train", k),
              windows_sha=json.load(open(f"{ST.L0}/sel_{k}/lam_0/SOLVER.json", encoding="utf-8"))["windows_sha256"])
    with np.load(f"{tmp}/off_wrapper/plans_hwm_oracle_s0.npz", allow_pickle=False) as z:
        wrapped = {x: z[x] for x in z.files}
    off_res["wrapper_file_train_A0_s0"] = K5.check_off_equal(wrapped, tb)
    green = dict(ok=all(v.get("ok", True) for v in off_res.values()), files=off_res)
    mut, _ = R.plan_arrays(ctx, b16, win16, "off", "none", "none", mutate="off_runs_pass")
    ctl["C-OFF5"] = verdict(green, K5.check_off_equal(mut, b16))
    # ---- C-MEAN
    hc_raw = ctx.hold_c(win64["A"][:, 0])[:, :R.KS, 0].numpy().astype(np.float64) * ACC
    lo_raw, hi_raw = float(ctx.lo1[0]) * ACC, float(ctx.hi1[0]) * ACC
    src64, logged = b64["a1_raw"][:, :R.KS, 0], win64["A"][:, :R.KS, 0]
    mz = K5.check_mean_zero(hc_raw, lo_raw, hi_raw)
    mg_g, mg_r = K5.check_mean_guard(src64, hc_raw, lo_raw, hi_raw, logged), K5.check_mean_guard(src64, hc_raw, lo_raw, hi_raw, logged, mutate=True)
    ctl["C-MEAN"] = verdict(dict(ok=bool(mz["ok"] and mg_g["ok"]), zero=mz, guard=mg_g), dict(ok=mg_r["ok"], guard_mutation=mg_r))
    # ---- C-L4-RA (argv -> built; the built readout kind is `mlp` and its sha256 the frozen one)
    dec = ST.build("train", "RAL1", "A0", k)
    argv_ok, dec_ok = dec["argv"], dec["declared"]
    argv_l2 = list(argv_ok); argv_l2[argv_l2.index("--ra-l1") + 1] = "l2"
    argv_unnamed = [t for i, t in enumerate(argv_ok) if t != "--ra-l1" and not (i > 0 and argv_ok[i - 1] == "--ra-l1")]
    bl = K5.check_l4_built(ctx, b16, win16, "l1", "l1", dec_ok)
    bl["kind_mlp"] = bool(bl.get("built", {}).get("kind") == "mlp" and bl.get("built", {}).get("readout_sha256") == J7["sha256"])
    bl["ok"] = bool(bl["ok"] and bl["kind_mlp"])
    bl_bad = K5.check_l4_built(ctx, dict(b16, l2_prim_raw=b16["a1_raw"] + 0.1), win16, "l1", "l1", dec_ok, mutate="wrong_src")
    pre_ok, pre_bad = K5.check_fresh_out_pre(f"{tmp}/fresh_ok", False), K5.check_fresh_out_pre(f"{tmp}/fresh_bad", True)
    l4g = dict(ok=bool(K5.check_l4(argv_ok, dec_ok)["ok"] and bl["ok"] and pre_ok["ok"]), argv=K5.check_l4(argv_ok, dec_ok), built=bl, fresh=pre_ok)
    l4r = dict(argv_mismatch=K5.check_l4(argv_l2, dec_ok), argv_unnamed=K5.check_l4(argv_unnamed, dec_ok), built_mismatch=bl_bad, preplanted=pre_bad)
    ctl["C-L4-RA"] = dict(green=l4g, regression=l4r, passed=bool(l4g["ok"] and not any(v["ok"] for v in l4r.values())))
    # ---- C-IN
    rec = {"ckpt": (c["ck"], c["cksha"]), "windows": (c["win"], c["wsha"]), "train_latents": (c["lat"], c["latsha"]), "readout_A": (pa, J7["sha256"]), "readout_B": (pb, J7["sha256_B"]), "spec": (ST.SPEC, ST.SPEC_SHA)}
    for s in (0, 1):
        bp = ST.base_plan("score", "A0", k, s)
        if os.path.exists(bp):
            rec[f"base_A0_s{s}"] = (bp, json.load(open(os.path.join(os.path.dirname(bp), "SOLVER.json"), encoding="utf-8"))["plans"][os.path.basename(bp)]["sha256"])
    for nm, fld in (("MEANL1", "MEANL1"), ("RAL1_RA6", "RAL1"), ("ORACLE_RA6", "ORACLE")):
        for s in (0, 1):
            d = f"{A6}/score/{fld}_A0_{k}"
            p = f"{d}/plans_hwm_oracle_s{s}.npz"
            if os.path.exists(p):
                rec[f"amend6_{nm}_s{s}"] = (p, json.load(open(f"{d}/SOLVER_RA.json", encoding="utf-8"))["plans"][os.path.basename(p)]["sha256"])
    bad_rec = dict(rec); bad_rec["readout_A"] = (pa, "0" * 64)
    ctl["C-IN"] = verdict(K5.check_in(rec), K5.check_in(bad_rec))
    cs = json.load(open(f"{ST.L0}/score/A0_{k}/SOLVER.json", encoding="utf-8"))["code_sha256"]
    now = K5.C.code_of_record_sha256()
    ctl["C-IN"]["code_of_record_equal_A0_solver"] = bool(all(now[f] == cs[f] for f in cs) and all(now[f] == v and cs[f] == v for f, v in ST.CODE_SHA_FULL.items()))
    ctl["C-IN"]["passed"] = bool(ctl["C-IN"]["passed"] and ctl["C-IN"]["code_of_record_equal_A0_solver"])
    # ---- C-READBACK of the base files + the regression arm
    rb = {}
    for s in (0, 1):
        bp = ST.base_plan("score", "A0", k, s)
        if os.path.exists(bp):
            rb[f"A0_s{s}"] = R.readback_base("A0", ST.base_dir("score", "A0", k), s)
            log("readback", s, rb[f"A0_s{s}"]["equal"], round(time.time() - t0, 1))
    reg = R.readback_base("A0", ST.base_dir("score", "A0", k), 0, lam_override=10.0)
    ctl["C-READBACK"] = verdict(dict(ok=bool(rb) and all(v["equal"] for v in rb.values()), files=rb), dict(ok=bool(reg["equal"]), lam_override=reg))
    # ---- C-STRAT (chassis independent: s0 only)
    if cstrat:
        import subprocess
        wd = f"{tmp}/cstrat"
        r = subprocess.run([sys.executable, f"{ST.LONPLAN_BUILD}/lonplan_cstrat.py", "--windows", ST.base_dir("score", "A0", "s0"), "--a", ST.base_plan("score", "A0", "s0", 0), "--b", ST.base_plan("score", "A0", "s0", 1), "--workdir", wd, "--n-boot", "200"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ))
        ctl["C-STRAT"] = dict(passed=bool(r.returncode == 0), rc=r.returncode, tail=r.stdout[-1200:], note="lonplan_cstrat.py: B = 200 here")
    out["wall_s"] = round(time.time() - t0, 1)
    out["red"] = sorted(n for n, v in ctl.items() if not v["passed"])
    out["ok"] = not out["red"]
    shutil.rmtree(tmp, ignore_errors=True)
    FT.jdump(out, f"{root}/controls/controls_{k}.json")
    return out


def merge(which, root=None):
    root = root or ROOT
    pat = "fit_{k}" if which == "fit" else "controls_{k}"
    parts = {k: json.load(open(f"{root}/controls/{pat.format(k=k)}.json", encoding="utf-8")) for k in ("s0", "s1")}
    red = sorted({f"{k}:{n}" for k, p in parts.items() for n in p["red"]})
    res = dict(ok=not red, red=red, which=which, chassis={k: dict(ok=p["ok"], wall_s=p["wall_s"], passed={n: v["passed"] for n, v in p["controls"].items()}) for k, p in parts.items()})
    FT.jdump(res, f"{root}/controls/{'CONTROLS_FIT' if which == 'fit' else 'CONTROLS'}.json")
    return res


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit"); f.add_argument("--chassis", required=True)
    r = sp.add_parser("run"); r.add_argument("--chassis", required=True); r.add_argument("--cstrat", action="store_true")
    m = sp.add_parser("merge"); m.add_argument("--which", choices=["fit", "run"], required=True)
    a = ap.parse_args()
    if a.cmd == "fit":
        o = fit_controls(a.chassis); print("RED:", o["red"] or "none", "wall", o["wall_s"]); sys.exit(0)
    if a.cmd == "run":
        o = run_chassis7(a.chassis, cstrat=(a.cstrat or a.chassis == "s0")); print("RED:", o["red"] or "none", "wall", o["wall_s"]); sys.exit(0)
    res = merge(a.which); print(json.dumps(res, indent=1)); sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
