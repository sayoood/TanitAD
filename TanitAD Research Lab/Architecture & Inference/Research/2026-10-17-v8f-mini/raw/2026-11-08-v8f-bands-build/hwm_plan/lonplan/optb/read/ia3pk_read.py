"""IA-3P-K reader (AMEND_OPTB section 3, mechanism read) and the C10' control. Tier T0 (world-model diagnostic, never driving performance). CPU only.

What it measures, per checkpoint (one chassis k, one arm):
  1. RA's FORM (ridge on D = z_{t+10} - z_t -> mean acceleration over the 10 transitions, WM units; lambda chosen on inner clip-grouped folds) is CROSS-FITTED on the 141 EVAL clips with IA-3's folds
     (lr_lib.fold_of: sha256(clip)[:8] % 2). Every number a clip contributes comes from the probe fitted on the OTHER fold's clips.
  2. s_swap, the transition-only calibration slope, on REAL pairs: y = z_t + D_t' (t' another start of the SAME clip >= 5 frames away, RandomState(0)),
     s_swap = Cov(R(real) - R(swap), abar - abar') / Var(abar - abar'). C10' = its CI wholly above 0.
  3. The arm's response: two contrasts +/- delta (WM units, default 0.4203 = the term's one-sd delta) on the acceleration channel over all K = 10 steps, one batched rollout (context 1) of the
     ARM's own WM from every valid anchor; read = R(z_K^+) - R(z_K^-); IA-3P-K gain G = mean(read) / (2 delta s_swap).  (1.0 = the response is the size of a real acceleration's.)
  4. Beside (TARGET-READ, no bar): the same response read through the FROZEN train-fit RA probe (readout_K10_R35_s{k}.pt) and its s_a (0.20076 / 0.20323): the term's own cycle gain on the eval clips.
Estimator (every interval): episode-cluster bootstrap over the 141 eval clips, B 2,000, seed 0 (taniteval.ci's draw). s_swap and the gain are recomputed from the SAME resample (numerator and
calibration move together). It answers ANOTHER DRAW OF EPISODES with the checkpoint and the fitted probe form held fixed; it does NOT answer another training run (that is the floor F_T,k from the
three FT-NULL runs) or another inference run.
The real pairs (z_t, z_{t+10}) do not depend on the arm: adapter and wm.proj are frozen, so z1 is the same for every arm of a chassis. The reader asserts wm.proj of the checkpoint is bit-equal to
the R35 reference's and refuses otherwise (the probe would be applied in the wrong space).
Usage:  python ia3pk_read.py read --ckpt <dir or ckpt_last.pt> --chassis 0|1 --tag FTRSP_s0 --out <json>   |   python ia3pk_read.py mech --dir <dir with the ten ia3pk_<tag>.json> --out <json>
"""
import argparse, dataclasses, hashlib, json, os, sys, time
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.environ.get("OPTB_TREE", "F:/TanitAD-scratch/actrsp/optb/tree")
FRAMES = os.environ.get("OPTB_FRAMES", "F:/TanitAD-scratch/ladder_read/cache/frames_s{k}.npz")
PROBE = os.environ.get("OPTB_PROBE", "F:/TanitAD-scratch/actrsp/amend1/out/readout_K10_R35_s{k}.pt")
R35 = os.environ.get("OPTB_R35", "C:/Users/Admin/t2v8f_pulls/ladder_R35_s{k}/ckpt_last.pt")
KS, B_BOOT, DELTA_WM = 10, 2000, 0.4203
LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
STAMP = dict(tier="T0", evidence="MEASURED", estimator="episode-cluster bootstrap over the 141 eval clips, B 2000, seed 0 (taniteval.ci draw); s_swap and the gain recomputed from the same resample",
             answers="another draw of EPISODES, checkpoint and probe form held fixed; NOT another training run (floor F_T,k) and NOT another inference run")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------- IA-3's folds and draws (copies of lr_lib.fold_of, imag1b_lib.boot_weights)
def fold_of(e):
    return int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2


def boot_weights(n, n_boot=B_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.arange(n)
    W = np.zeros((n_boot, n))
    for b in range(n_boot):
        pick = rng.choice(uniq, size=n, replace=True)
        W[b] = np.bincount(np.searchsorted(uniq, pick), minlength=n)
    return W


# ---------------------------------------------------------------------------------------------------------------- ridge (amend1_kstep.py, line for line)
def ridge_fit(X, Yy, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    ym = Yy.mean(0)
    W = np.linalg.solve(Xs.T @ Xs + lam * len(X) * np.eye(X.shape[1]), Xs.T @ (Yy - ym))
    return mu, sd, W, ym


def ridge_pred(m, X):
    return ((X - m[0]) / m[1]) @ m[2] + m[3]


def ridge_lam(X, Yy, fo):
    best, bl = np.inf, None
    for lam in LAMS:
        err = sum(((ridge_pred(ridge_fit(X[fo != g], Yy[fo != g], lam), X[fo == g]) - Yy[fo == g]) ** 2).sum() for g in np.unique(fo))
        if err < best:
            best, bl = err, lam
    return bl


# ---------------------------------------------------------------------------------------------------------------- pairs from the cached eval frames
def pick_partners(ep, fr):
    """swap partner of each anchor: another start of the SAME clip >= 5 frames away (RandomState(0), anchors in order); itself if none."""
    rng = np.random.RandomState(0)
    part = np.empty(len(ep), dtype=int)
    groups = {}
    for i, e in enumerate(ep):
        groups.setdefault(int(e), []).append(i)
    for e, g in groups.items():
        g = np.array(g)
        for i in g:
            cand = g[np.abs(fr[g] - fr[i]) >= 5]
            part[i] = rng.choice(cand) if len(cand) else i
    return part


def build_pairs(frames, z1_of, scale_actions, K=KS):
    """frames: {clips, T, zmean, act, ok} (ladder_read cache). z1_of(zmean[T,d]) -> z1 [T,D] (float32 torch); scale_actions(act[T,2]) -> WM units [T,2].
    Returns the arrays of every valid anchor: all K+1 frames ok and all K actions finite. (clip index, frame, z_t, z_{t+K}, abar [mean of the K actions], aseq [K,2], partner)."""
    import torch
    clips, T = [str(c) for c in frames["clips"]], [int(t) for t in frames["T"]]
    cut = np.cumsum(T)[:-1]
    zm, act, ok = np.split(frames["zmean"], cut), np.split(frames["act"], cut), np.split(frames["ok"], cut)
    EP, FR, ZT, ZK, AS = [], [], [], [], []
    for ci in range(len(clips)):
        z1 = z1_of(torch.as_tensor(zm[ci]).float()).float()
        a = scale_actions(torch.as_tensor(act[ci]).float()).float()
        okc = torch.as_tensor(ok[ci]).bool()
        for f in range(0, T[ci] - K):
            if bool(okc[f:f + K + 1].all()) and bool(torch.isfinite(a[f:f + K]).all()):
                EP.append(ci); FR.append(f); ZT.append(z1[f]); ZK.append(z1[f + K]); AS.append(a[f:f + K])
    P = {"ep": np.array(EP), "fr": np.array(FR), "zt": torch.stack(ZT), "zk": torch.stack(ZK), "aseq": torch.stack(AS), "clips": clips}
    P["abar"] = P["aseq"].mean(1).numpy().astype(np.float64)
    part = pick_partners(P["ep"], P["fr"])
    P["part"] = part
    return P


def crossfit_ra(P):
    """RA's form on the eval clips, folds = lr_lib.fold_of. Returns per-fold models, R_real and R_swap on every anchor (from the OTHER fold's probe), and the held-out R^2."""
    Zt, Zk = P["zt"].numpy().astype(np.float64), P["zk"].numpy().astype(np.float64)
    y = P["abar"][:, 0:1]
    ep = P["ep"]
    fold = np.array([fold_of(c) for c in P["clips"]])[ep]
    Dr = Zk - Zt
    Dsw = Zt + (Zk[P["part"]] - Zt[P["part"]]) - Zt                   # = D of the partner step grafted on z_t (z_t cancels: RA reads the difference)
    models, lams = {}, {}
    Rr, Rs = np.zeros(len(ep)), np.zeros(len(ep))
    rng = np.random.RandomState(0)
    inner_of_clip = rng.permutation(len(P["clips"])) % 5
    for f in (0, 1):
        tr, te = fold != f, fold == f
        lam = ridge_lam(Dr[tr], y[tr], inner_of_clip[ep][tr])
        m = ridge_fit(Dr[tr], y[tr], lam)
        models[f], lams[f] = m, lam
        Rr[te] = ridge_pred(m, Dr[te])[:, 0]
        Rs[te] = ridge_pred(m, Dsw[te])[:, 0]
    r2 = float(1 - ((Rr - y[:, 0]) ** 2).mean() / ((y[:, 0] - y[:, 0].mean()) ** 2).mean())
    return {"models": models, "lams": lams, "R_real": Rr, "R_swap": Rs, "fold": fold, "r2": r2}


# ---------------------------------------------------------------------------------------------------------------- the response: two contrasts, one batched rollout
def contrast_read(roll_fn, P, readers, delta=DELTA_WM, chunk=2048):
    """roll_fn(z0 [N,D] torch, a [N,K,2] torch) -> z_K [N,D] torch (context 1, the ARM's WM). readers: callable(zk_minus_z0 float64 [N,D], idx) -> read [N].
    Returns per anchor: z_K^+ and z_K^- reads (float64). Acceleration channel only (channel 0)."""
    import torch
    n = len(P["ep"])
    out = {}
    for sign, nm in ((+1.0, "plus"), (-1.0, "minus")):
        a = P["aseq"].clone()
        a[..., 0] += sign * delta
        zk = []
        for s in range(0, n, chunk):
            with torch.no_grad():
                zk.append(roll_fn(P["zt"][s:s + chunk], a[s:s + chunk]).float())
        out[nm] = (torch.cat(zk).numpy().astype(np.float64) - P["zt"].numpy().astype(np.float64))
    return out


def moments_by_clip(ep, vals, n_clips):
    """per-clip [n, S1.. ] sums for a list of per-anchor value arrays: returns [n_clips, 1 + len(vals)] with column 0 = count."""
    M = np.zeros((n_clips, 1 + len(vals)))
    M[:, 0] = np.bincount(ep, minlength=n_clips)
    for j, v in enumerate(vals):
        M[:, 1 + j] = np.bincount(ep, weights=v, minlength=n_clips)
    return M


def s_swap_from_moments(Ms, W):
    """Ms columns: n, Su, Sd, Sud, Sdd (u = R(real)-R(swap), d = abar-abar'). W [B, n_clips] (or [1, n_clips] ones). Population Cov/Var of the resample."""
    S = W @ Ms
    n = S[:, 0]
    cov = S[:, 3] / n - (S[:, 1] / n) * (S[:, 2] / n)
    var = S[:, 4] / n - (S[:, 2] / n) ** 2
    return cov / var


def percentiles(point, boot):
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)]}


def read_core(P, roll_fn, delta=DELTA_WM, frozen_probe=None, W=None):
    """Model-agnostic core (the tests call it with synthetic worlds). Returns the result dict + the per-clip moment arrays (for paired arm differences)."""
    n_clips = len(P["clips"])
    W = boot_weights(n_clips) if W is None else W
    ones = np.ones((1, n_clips))
    cf = crossfit_ra(P)
    d = P["abar"][:, 0] - P["abar"][P["part"], 0]
    u = cf["R_real"] - cf["R_swap"]
    Ms = moments_by_clip(P["ep"], [u, d, u * d, d * d], n_clips)
    s_pt = float(s_swap_from_moments(Ms, ones)[0])
    s_bt = s_swap_from_moments(Ms, W)
    # response through the cross-fitted probe of each anchor's own fold
    rd = contrast_read(roll_fn, P, None, delta)
    def apply(models_by_fold):
        r = {}
        for nm in ("plus", "minus"):
            v = np.zeros(len(P["ep"]))
            for f in (0, 1):
                m = cf["fold"] == f
                v[m] = ridge_pred(models_by_fold[f], rd[nm][m])[:, 0]
            r[nm] = v
        return r["plus"] - r["minus"]
    dR = apply(cf["models"])
    Mg = moments_by_clip(P["ep"], [dR], n_clips)
    mean_bt = (W @ Mg)[:, 1] / (W @ Mg)[:, 0]
    mean_pt = float(Mg[:, 1].sum() / Mg[:, 0].sum())
    g_pt, g_bt = mean_pt / (2 * delta * s_pt), mean_bt / (2 * delta * s_bt)
    res = {"n_anchors": int(len(P["ep"])), "n_clips": n_clips, "delta_wm": delta, "ra_crossfit_r2_heldout": cf["r2"], "ra_lambda_by_fold": {str(k): v for k, v in cf["lams"].items()},
           "s_swap": percentiles(s_pt, s_bt), "c10p_s_swap_ci_above_0": bool(np.nanpercentile(s_bt, 2.5) > 0),
           "ia3pk_gain_L1_2s_lon": percentiles(g_pt, g_bt), "mean_read_diff_plus_minus": mean_pt, "estimator": STAMP}
    if frozen_probe is not None:
        mA, s_a = frozen_probe
        rr = {nm: ridge_pred(mA, rd[nm])[:, 0] for nm in ("plus", "minus")}
        Mf = moments_by_clip(P["ep"], [rr["plus"] - rr["minus"]], n_clips)
        Sf = W @ Mf
        pt = Mf[:, 1].sum() / Mf[:, 0].sum() / (2 * delta * s_a)
        res["target_read_cycle_gain_frozen_probe"] = {**percentiles(pt, Sf[:, 1] / Sf[:, 0] / (2 * delta * s_a)), "s_a_frozen": s_a,
                                                      "note": "beside, no bar: the term's own cycle gain G_aa on the EVAL clips through the train-fit frozen RA"}
    return res, {"Ms": Ms, "Mg": Mg, "ep_n": Mg[:, 0]}


# ---------------------------------------------------------------------------------------------------------------- model loading (the tree = blob-identical to 4c396aff's stack)
def load_ckpt(path):
    import torch
    sys.path.insert(0, os.path.join(TREE, "stack"))
    from tanitad.models import v8f_mini as M
    p = path if path.endswith(".pt") else os.path.join(path, "ckpt_last.pt")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    names = {f.name for f in dataclasses.fields(M.V8FMiniConfig)}
    cfg = M.V8FMiniConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items() if k in names})
    torch.manual_seed(0)
    wm = M.WMAux(cfg)
    wm.load_state_dict({k[3:]: v for k, v in ck["model"].items() if k.startswith("wm.")}, strict=True)
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    h = hashlib.sha256()
    for k in sorted(k for k in ck["model"] if k.startswith("adapter.") or k == "pos"):
        t = ck["model"][k].detach().cpu().contiguous()
        h.update(k.encode() + str(tuple(t.shape)).encode() + str(t.dtype).encode() + t.numpy().tobytes())
    proj = hashlib.sha256()
    for k in sorted(k for k in ck["model"] if k.startswith("wm.proj")):
        t = ck["model"][k].detach().cpu().contiguous()
        proj.update(k.encode() + t.numpy().tobytes())
    return wm, {"adapter_sha": h.hexdigest(), "proj_sha": proj.hexdigest(), "path": p, "bytes": os.path.getsize(p), "step": int(ck.get("step", -1))}, M


def read_checkpoint(ckpt, chassis, tag, out, delta=DELTA_WM, smoke_frac=None):
    import torch
    t0 = time.time()
    fr_np = np.load(FRAMES.format(k=chassis), allow_pickle=False)
    frames = {k: fr_np[k] for k in ("clips", "T", "zmean", "act", "ok")}
    wm, info, M = load_ckpt(ckpt)
    ref_wm, ref_info, _ = load_ckpt(R35.format(k=chassis))
    if str(fr_np["adapter_sha"]) != info["adapter_sha"]:
        raise RuntimeError(f"REFUSED: adapter of {ckpt} differs from the cached frames' adapter ({info['adapter_sha'][:12]} vs {str(fr_np['adapter_sha'])[:12]})")
    if info["proj_sha"] != ref_info["proj_sha"]:
        raise RuntimeError("REFUSED: wm.proj differs from R35's: the frozen probe would be applied in another latent space")
    if smoke_frac:
        keep = int(len(frames["T"]) * smoke_frac)
        cut = int(np.sum(frames["T"][:keep]))
        frames = {"clips": frames["clips"][:keep], "T": frames["T"][:keep], "zmean": frames["zmean"][:cut], "act": frames["act"][:cut], "ok": frames["ok"][:cut]}
    P = build_pairs(frames, lambda x: wm.z1_from_mean(x[None])[0], lambda a: wm.scale_actions(a[None])[0])
    log("pairs", len(P["ep"]), "anchors", len(P["clips"]), "clips")
    st = wm.stack
    roll = lambda z0, a: st.rollout(1, z0, a, context=1)[:, -1]
    pr = torch.load(PROBE.format(k=chassis), map_location="cpu", weights_only=False)
    mA = (pr["RA"]["mu"], pr["RA"]["sd"], pr["RA"]["W"], pr["RA"]["ym"])
    res, mom = read_core(P, roll, delta, frozen_probe=(mA, float(pr["s_cal_swap_heldout"][0])))
    res.update({"tag": tag, "chassis": chassis, "ckpt": info, "r35_ref_proj_sha_equal": True, "wall_s": round(time.time() - t0, 1), "smoke_frac": smoke_frac})
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    np.savez(out[:-5] + "_moments.npz", **mom)
    log("done", tag, "s_swap", res["s_swap"], "gain", res["ia3pk_gain_L1_2s_lon"], "wall", res["wall_s"])
    return res


# ---------------------------------------------------------------------------------------------------------------- MECH-MOVED over the ten arms
PAIRS = {0: [("FTRSP_s0", "FTNULL_s0"), ("FTRSP_s2", "FTNULL_s2")], 1: [("FTRSP_s1", "FTNULL_s1"), ("FTRSP_s3", "FTNULL_s3")]}
NULLS = {0: ["FTNULL_s0", "FTNULL_s2", "FTNULL_s4"], 1: ["FTNULL_s1", "FTNULL_s3", "FTNULL_s5"]}


def mech_moved(R):
    """R: {tag: result dict with ia3pk_gain_L1_2s_lon}. MECH-MOVED iff at BOTH chassis and BOTH lever seeds: gain(FT-RSP) > gain(same-seed FT-NULL), CIs disjoint (RSP lo > NULL hi), and the gap
    exceeds F_T,k = max - min of the three FT-NULL point gains of that chassis. A missing arm = NOT-COMPUTABLE (never a pass)."""
    rows, ok_all, missing = [], True, []
    for k in (0, 1):
        if not all(t in R for t in NULLS[k]):
            missing += [t for t in NULLS[k] if t not in R]
        pts = [R[t]["ia3pk_gain_L1_2s_lon"]["point"] for t in NULLS[k] if t in R]
        floor = (max(pts) - min(pts)) if len(pts) == 3 else None
        for rsp, nul in PAIRS[k]:
            if rsp not in R or nul not in R or floor is None:
                missing += [t for t in (rsp, nul) if t not in R]
                ok_all = False
                rows.append({"chassis": k, "pair": [rsp, nul], "status": "NOT-COMPUTABLE"})
                continue
            a, b = R[rsp]["ia3pk_gain_L1_2s_lon"], R[nul]["ia3pk_gain_L1_2s_lon"]
            gap = a["point"] - b["point"]
            c = {"above": gap > 0, "ci_disjoint": a["ci95"][0] > b["ci95"][1], "gap_above_floor": gap > floor}
            row = {"chassis": k, "pair": [rsp, nul], "gain_rsp": a, "gain_null": b, "gap": gap, "F_T_k": floor, **c, "pass": all(c.values())}
            ok_all &= row["pass"]
            rows.append(row)
    label = "NOT-COMPUTABLE" if missing else ("MECH-MOVED" if ok_all else "NOT-MECH-MOVED")
    return {"label": label, "missing": sorted(set(missing)), "rows": rows, "rule": "AMEND_OPTB section 3 (mechanism read)", "estimator": STAMP}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--ckpt", required=True); r.add_argument("--chassis", type=int, required=True); r.add_argument("--tag", required=True); r.add_argument("--out", required=True)
    r.add_argument("--delta", type=float, default=DELTA_WM); r.add_argument("--smoke-frac", type=float, default=None)
    m = sub.add_parser("mech")
    m.add_argument("--dir", required=True); m.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "read":
        if os.path.exists(a.out):
            sys.exit(f"REFUSED: {a.out} exists")
        read_checkpoint(a.ckpt, a.chassis, a.tag, a.out, a.delta, a.smoke_frac)
    else:
        R = {}
        for k in (0, 1):
            for t in NULLS[k] + [x for p in PAIRS[k] for x in p[:1]]:
                p = os.path.join(a.dir, f"ia3pk_{t}.json")
                if os.path.exists(p):
                    R[t] = json.load(open(p, encoding="utf-8"))
        json.dump(mech_moved(R), open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
