"""AMEND_HWMPLAN_7: the pure functions of the RA7 fit stage. No file I/O. Numpy + torch (CPU).
Sections of the card: 2.1 the net, 2.3 selection and seeds, 3 arms / R2 / r_wm / planted / LinearD / bootstrap, 4 gate G7-FIT, 8.1 the pure FIT-stage controls.
Tier T0. Every function here is exercised by test_amend7.py on synthetic data."""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
A6B = os.path.dirname(os.path.dirname(HERE)) + "/amend6/build"
if A6B not in sys.path:
    sys.path.append(A6B)
import numpy as np
import torch
import fit_ra6 as F          # AMEND_6's ridge, lambda choice, fold map (imported, never edited)

WIDTH, LR, BATCH = 256, 1e-3, 256
GRID = [(e, wd) for e in (2, 6, 12, 24) for wd in (1.0, 0.1, 0.01)]          # card 2.3: epochs ascending outer, weight decay descending inner; a tie goes to the earlier config
S0_A, S0_B = 0, 500
SEED_FINAL_SEL, SEED_FINAL_REFIT = 8, 9
ABAR = 0.0026808911934494972                                                     # card 3.3: amend1's a_mean_wm[0] (scaled)
M_K = {"s0": 0.0238, "s1": 0.0246}                                               # card section 4: upper half-width of RA's own 1,000-clip R2 interval
SEED_PERM_T, SEED_PLANT, SEED_PLANT_EPS = 20261120, 20261118, 20261119


# ---------------------------------------------------------------------------------------------- 2.1 the net
def make_net(d_in, width=WIDTH):
    return torch.nn.Sequential(torch.nn.Linear(d_in, width), torch.nn.GELU(), torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, 1))


def batch_order(n, seed, epoch):
    """The seed sets the batch order as well as the initialisation (card 2.1): RandomState(1000 * seed + epoch)."""
    return np.random.RandomState(1000 * seed + epoch).permutation(n)


def fit_snapshots(X, y, epochs, wd, seed, width=None, batch=BATCH, lr=LR):
    """RA7's fit. X (n, d) float64, y (n,) or (n, 1). AdamW lr 1e-3 default betas, weight decay wd, batch 256, MSE on the unscaled y, float32, no early stopping.
    One run to max(epochs) with a snapshot at each requested epoch count; a run to e epochs is bit-equal to the snapshot at e because the batch order of epoch e' depends only on (seed, e').
    -> {e: model}. Standardisation: the fit rows' mean and population sd + 1e-6."""
    width = WIDTH if width is None else width
    epochs = sorted(set(int(e) for e in epochs))
    torch.manual_seed(int(seed))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor(np.asarray(y, np.float64).reshape(-1, 1), dtype=torch.float32)
    net = make_net(X.shape[1], width)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=wd)
    out = {}
    for e in range(epochs[-1]):
        idx = batch_order(len(X), seed, e)
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            loss = ((net(Xs[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (e + 1) in epochs:
            out[e + 1] = dict(kind="mlp", state={k: v.detach().clone() for k, v in net.state_dict().items()}, mu=mu, sd=sd, width=width, epochs=e + 1, wd=wd, seed=int(seed))
    return out


def predict(m, X):
    """X (n, d_in) float64 -> (n,) float64, by the model kind."""
    kind = m["kind"]
    if kind == "ridge":
        return F.ridge_pred(m["m"], X)[:, 0]
    if kind == "mlp":
        net = make_net(X.shape[1], m["width"]); net.load_state_dict(m["state"]); net.eval()
        with torch.no_grad():
            return net(torch.tensor((X - m["mu"]) / m["sd"], dtype=torch.float32)).numpy().astype(np.float64)[:, 0]
    if kind == "lind":
        d = X.shape[1] // 2
        return ((X[:, d:] - X[:, :d] - m["mu"]) / m["sd"]) @ m["w"] + m["b"]
    if kind == "const":
        return np.full(len(X), m["c"])
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------- 3.5 the K-control model
def fit_lind(X, y, lam, max_iter=2000):
    """LinearD: input [z0, z10] -> D = z10 - z0, standardised by the training rows' mean and population sd + 1e-6, Linear(128, 1).
    Loss mean((p - y)^2) + lam ||w||^2, bias unpenalised; full-batch L-BFGS, float64, strong Wolfe, <= 2000 iterations, gradient tolerance 1e-12."""
    d = X.shape[1] // 2
    D = X[:, d:] - X[:, :d]
    mu, sd = D.mean(0), D.std(0) + 1e-6
    Ds = torch.tensor((D - mu) / sd, dtype=torch.float64)
    yt = torch.tensor(np.asarray(y, np.float64).reshape(-1, 1))
    w = torch.zeros(d, 1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, tolerance_grad=1e-12, tolerance_change=1e-30, history_size=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((Ds @ w + b - yt) ** 2).mean() + lam * (w ** 2).sum()
        loss.backward()
        return loss
    for _ in range(10):
        opt.step(closure)
        g = max(float(w.grad.abs().max()), float(b.grad.abs().max()))
        if g <= 1e-12:
            break
    return dict(kind="lind", mu=mu, sd=sd, w=w.detach().numpy()[:, 0].copy(), b=float(b.detach()), lam=float(lam))


# ---------------------------------------------------------------------------------------------- 2.3 selection (inner folds only)
def select_config(X, y, fo, oc, seed_base, grid=None, width=None, val_override=None):
    """The outer training rows X (folds != f), their fold labels fo and order-clip index oc. For each config the inner error = sum over the folds g present in fo of the squared error on fold g of the model fit on
    the folds not in {f, g} (here: fo != g within X), all fits of one slot at seed seed_base + g. The config with the least inner error; a tie goes to the earlier config.
    val_override (the C-SEL7 regression arm only): (Xv, yv, clips_v) replaces every validation set (selection scored on the OUTER TEST fold).
    -> dict(chosen=(epochs, wd), errors=[{epochs, wd, err}], fits=[{g, seed, val_clips, train_folds}])."""
    grid = GRID if grid is None else grid
    width = WIDTH if width is None else width
    folds = sorted(set(int(g) for g in np.unique(fo)))
    errs = {c: 0.0 for c in grid}
    fits = []
    wds = list(dict.fromkeys(wd for _, wd in grid))
    for g in folds:
        tr, va = fo != g, fo == g
        if val_override is not None:
            Xv, yv, cv = val_override
        else:
            Xv, yv, cv = X[va], np.asarray(y).reshape(-1)[va], oc[va]
        for wd in wds:
            eps = sorted({e for e, w in grid if w == wd})
            ms = fit_snapshots(X[tr], np.asarray(y).reshape(-1)[tr], eps, wd, seed_base + g, width)
            for e in eps:
                errs[(e, wd)] += float(((predict(ms[e], Xv) - yv) ** 2).sum())
        fits.append(dict(g=int(g), seed=int(seed_base + g), val_clips=sorted(set(int(c) for c in cv)), train_folds=[int(h) for h in folds if h != g]))
    best = None
    for c in grid:
        if best is None or errs[c] < errs[best]:
            best = c
    return dict(chosen=[int(best[0]), float(best[1])], errors=[dict(epochs=int(e), wd=float(w), err=errs[(e, w)]) for e, w in grid], fits=fits)


def mlp_learner(S0, grid=None, width=None, mutate=None):
    """-> learn(X, y, fo, oc, f, test=None): select (card 2.3) on the training rows, refit on all of them at seed S0 + 10 f + 9. f = 8 is the final readout's selection (inner folds over all 1,000 clips)."""
    def learn(X, y, fo, oc, f, test=None):
        g_, w_ = (GRID if grid is None else grid), (WIDTH if width is None else width)
        base = S0 + 10 * f
        vo = test if (mutate == "select_on_test" and test is not None) else None
        sel = select_config(X, y, fo, oc, base, g_, w_, val_override=vo)
        e, wd = sel["chosen"]
        m = fit_snapshots(X, np.asarray(y).reshape(-1), [e], wd, base + 9, w_)[e]
        return m, dict(selection=sel, refit_seed=int(base + 9), chosen=sel["chosen"])
    return learn


def ridge_learner():
    def learn(X, y, fo, oc, f, test=None):
        lam = F.choose_lam(X, np.asarray(y).reshape(-1, 1), fo)
        return dict(kind="ridge", m=F.ridge_fit(X, np.asarray(y).reshape(-1, 1), lam), lam=lam), dict(lam=float(lam))
    return learn


def lind_learner(lams):
    def learn(X, y, fo, oc, f, test=None):
        return fit_lind(X, y, lams[f]), dict(lam=float(lams[f]))
    return learn


def const_learner():
    def learn(X, y, fo, oc, f, test=None):
        return dict(kind="const", c=float(np.asarray(y).mean())), dict(c=float(np.asarray(y).mean()))
    return learn


# ---------------------------------------------------------------------------------------------- inputs
def build_input(kind, z0, z1):
    """kind D: z1 - z0 (RA, ridge); z0z1: [z0, z1] (RA7, KCTL); z0z0: [z0, z0] (ZERO: z10 replaced by z0, so any z1 is ignored)."""
    z0 = np.asarray(z0, np.float64); z1 = np.asarray(z1, np.float64)
    if kind == "D":
        return z1 - z0
    if kind == "z0z1":
        return np.concatenate([z0, z1], 1)
    if kind == "z0z0":
        return np.concatenate([z0, z0], 1)
    raise ValueError(kind)


def crossfit(learn, kind, z0, z10, y, fo, oc, Ftrue=None, Falt=None, Zswap=None, mutate=None):
    """The 5-fold clip-grouped cross-fit with the fold map `fo` (card 2.2). Each pair is predicted by the model that did not see its clip.
    -> dict(pred, dR, rswap, models, folds): dR = R(z0, F_true) - R(z0, F_alt) by the fold model (card 3.3), rswap = R(z0, Zswap) (the swap end state), folds = records for C-FOLD7 / C-SEL7.
    mutate: 'test_in_train' (C-FOLD7 regression arm: fold f's rows are added to its own fit)."""
    n = len(y)
    X = build_input(kind, z0, z10)
    pred = np.zeros(n); dR = np.zeros(n) if Ftrue is not None else None; rswap = np.zeros(n) if Zswap is not None else None
    models, recs = [], []
    for f in range(5):
        tr, te = fo != f, fo == f
        trx = np.ones(n, bool) if mutate == "test_in_train" else tr
        m, rec = learn(X[trx], np.asarray(y)[trx], fo[trx], oc[trx], f, test=(X[te], np.asarray(y)[te], oc[te]))
        pred[te] = predict(m, X[te])
        if Ftrue is not None:
            dR[te] = predict(m, build_input(kind, z0[te], Ftrue[te])) - predict(m, build_input(kind, z0[te], Falt[te]))
        if Zswap is not None:
            rswap[te] = predict(m, build_input(kind, z0[te], Zswap[te]))
        models.append(m)
        recs.append(dict(f=f, train_clips=sorted(set(int(c) for c in oc[trx])), test_clips=sorted(set(int(c) for c in oc[te])), **rec))
    return dict(pred=pred, dR=dR, rswap=rswap, models=models, folds=recs)


# ---------------------------------------------------------------------------------------------- metrics and the estimator (3.2, 3.3, 3.6)
def r2(p, y):
    """R2 = 1 - mean((p - y)^2) / mean((y - ybar)^2) over the pooled held-out predictions (fit_ra6.py:166)."""
    return float(1 - ((p - y) ** 2).mean() / ((y - y.mean()) ** 2).mean())


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


class Boot:
    """Paired clip-cluster percentile bootstrap through per-clip sums. A replicate draws RandomState(seed).randint(0, n_clips, n_clips) clips (with replacement) and every statistic is computed on that one
    resample (card 3.6). Statistics are functions of the totals dict T = sums over the resampled clips; equal to the row-concatenation of fit_ra6.curve_pure up to float summation order (~1e-15)."""
    def __init__(self, oc, n_clips, cols):
        self.names = list(cols) + ["n"]
        self.n_clips = n_clips
        self.M = np.zeros((n_clips, len(self.names)))
        for j, nm in enumerate(cols):
            self.M[:, j] = np.bincount(oc, weights=np.asarray(cols[nm], np.float64), minlength=n_clips)
        self.M[:, -1] = np.bincount(oc, minlength=n_clips)

    def totals(self, w):
        return dict(zip(self.names, w @ self.M))

    def run(self, fn, B, seed=0):
        pt = np.asarray(fn(self.totals(np.ones(self.n_clips))), float)
        r = np.random.RandomState(seed)
        bs = np.array([np.asarray(fn(self.totals(np.bincount(r.randint(0, self.n_clips, self.n_clips), minlength=self.n_clips).astype(float))), float) for _ in range(B)])
        lo, hi = np.percentile(bs, [2.5, 97.5], 0)
        return pt, lo, hi


def t_r2(T, e2):
    n = T["n"]
    return 1 - (T[e2] / n) / (T["yy"] / n - (T["y"] / n) ** 2)


def t_corr(T, x, xx, xy):
    n = T["n"]
    mx, my = T[x] / n, T["y"] / n
    return (T[xy] / n - mx * my) / np.sqrt((T[xx] / n - mx ** 2) * (T["yy"] / n - my ** 2))


def curve_row(pred, y, oc, n_clips, B=1000, seed=0):
    """RA's FIT-CURVE row at one total (r2, r and their clip-cluster CIs) for C-CURVE7: the estimator of fit_ra6.curve_pure (B 1000, seed 0)."""
    bt = Boot(oc, n_clips, dict(y=y, yy=y * y, e2=(pred - y) ** 2, p=pred, pp=pred * pred, py=pred * y))
    pt, lo, hi = bt.run(lambda T: np.array([t_r2(T, "e2"), t_corr(T, "p", "pp", "py")]), B, seed)
    return dict(r2=float(pt[0]), r2_ci=[float(lo[0]), float(hi[0])], r=float(pt[1]), r_ci=[float(lo[1]), float(hi[1])])


# ---------------------------------------------------------------------------------------------- 3.4 the planted target, the shuffled target
def _std(x):
    return (x - x.mean()) / x.std()


def _resid(x, cols):
    A = np.column_stack([np.ones(len(x))] + list(cols))
    return x - A @ np.linalg.lstsq(A, x, rcond=None)[0]


def planted(z0, D):
    """card 3.4. -> y_plant (n,), r2_oracle, f (n,)."""
    n, d = D.shape
    z0s = (z0 - z0.mean(0)) / z0.std(0); Ds = (D - D.mean(0)) / D.std(0)
    rng = np.random.RandomState(SEED_PLANT)
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    v1 = rng.standard_normal(d); v1 /= np.linalg.norm(v1)
    v2 = rng.standard_normal(d); v2 /= np.linalg.norm(v2)
    t1 = _std(Ds @ v1)
    t2 = _std(_resid(Ds @ v2, [t1]))
    s = _std(_resid(z0s @ u, [t1, t2]))
    t1c, t2c, sc = (np.clip(x, -3, 3) for x in (t1, t2, s))
    f = t1c * (1 + np.tanh(sc)) + 0.5 * (t2c ** 2 - 1)
    g = _std(f)
    eps = np.random.RandomState(SEED_PLANT_EPS).standard_normal(n)
    yp = g + eps
    return yp, float(1 - (eps ** 2).mean() / yp.var()), f


def shuffled(y):
    return np.asarray(y)[np.random.RandomState(SEED_PERM_T).permutation(len(y))]


# ---------------------------------------------------------------------------------------------- 3.3 the swap partner (amend1_kstep.py:71-80, on this table's frames)
def swap_partner(oc_sorted_idx, FR):
    """oc_sorted_idx: the clip index in the sorted-clip order of fit_ra6.pairs (rows are grouped by it, ascending); FR the frames. For each pair a partner of the SAME clip >= 5 frames away, RandomState(0).choice;
    a clip with no such partner maps to itself."""
    n = len(FR)
    rng = np.random.RandomState(0)
    part = np.empty(n, dtype=int)
    groups = {}
    for i, e in enumerate(oc_sorted_idx):
        groups.setdefault(int(e), []).append(i)
    for e, g in groups.items():
        g = np.array(g)
        for i in g:
            cand = g[np.abs(FR[g] - FR[i]) >= 5]
            part[i] = rng.choice(cand) if len(cand) else i
    return part


# ---------------------------------------------------------------------------------------------- 4 the gate
def chassis_gate(a, M, ft_r2, ft_r):
    """a: dict with dR2_A, dR2_B, dr_A, dr_B, each dict(point, lo, hi). M: the chassis' M_k; ft_r2, ft_r: F_T,k of R2 and of r_wm.
    (a) the R2 part, (b) the pass-contrast part. -> dict(a, b, label)."""
    A, B, rA, rB = a["dR2_A"], a["dR2_B"], a["dr_A"], a["dr_B"]
    pa = bool(A["lo"] > 0 and A["point"] > max(ft_r2, M) and B["lo"] > 0)
    pb = bool(rA["lo"] > 0 and rA["point"] > ft_r and rB["lo"] > 0)
    return dict(a=pa, b=pb, label=("R7-FIT-PASS" if (pa and pb) else "R7-STATE-ONLY" if pa else "R7-FIT-FAIL"))


FIT_ORDER = ("R7-FIT-PASS", "R7-STATE-ONLY", "R7-FIT-FAIL")


def fit_outcome(per, controls_ok=True):
    """per: {chassis: chassis_gate output}. Both chassis required; one chassis only is SEEDS-DISAGREE and the less favourable label applies. TRANS-ONLY qualifies R7-FIT-FAIL when (b) holds at both chassis.
    A RED control or a regression arm not RED is R7-FIT-INVALID (nothing is read)."""
    if not controls_ok:
        return dict(outcome="R7-FIT-INVALID", label="R7-FIT-INVALID", seeds_disagree=False, trans_only=False, text="R7-FIT-INVALID")
    ls = [v["label"] for v in per.values()]
    dis = len(set(ls)) > 1
    eff = max(ls, key=FIT_ORDER.index)
    trans = bool(eff == "R7-FIT-FAIL" and all(v["b"] for v in per.values()))
    text = eff + (" TRANS-ONLY" if trans else "") + (" SEEDS-DISAGREE" if dis else "")
    return dict(outcome=eff, label=eff, seeds_disagree=dis, trans_only=trans, text=text)


# ---------------------------------------------------------------------------------------------- 8.1 the pure FIT-stage controls (each returns dict(ok=...); the regression arm must read ok False)
def check_pairs(ext, ref, n_expected=None, n_clips_expected=None):
    """C-PAIRS7: the extended table equals fit_ra6.pairs bit for bit (D, y, clip index); counts as expected."""
    d = dict(n=int(len(ref["Yf"])), D_equal=bool(np.array_equal(ext["D"], ref["Dr"])), y_equal=bool(np.array_equal(ext["y"][:, None], ref["Yf"])), ep_equal=bool(np.array_equal(ext["ep"], ref["EP"])))
    d["counts_ok"] = bool((n_expected is None or d["n"] == n_expected) and (n_clips_expected is None or len(ref["clips"]) == n_clips_expected))
    d["ok"] = bool(d["D_equal"] and d["y_equal"] and d["ep_equal"] and d["counts_ok"])
    return d


def check_chunk_files(J, lat_dir):
    import hashlib
    bad = [ch["file"] for ch in J["chunks"] if not os.path.exists(f"{lat_dir}/{ch['file']}") or hashlib.sha256(open(f"{lat_dir}/{ch['file']}", "rb").read()).hexdigest() != ch["sha256"]]
    return dict(ok=not bad, n_chunks=len(J["chunks"]), bad=bad)


def check_iso(fit_clips, c141, clips6, score_eids, n_new=859):
    """C-ISO7: the fit clips (those present in the merged latents) are exactly clips141 + clips6[:859], with no repeat; none is a SCORE episode."""
    want = list(c141) + list(clips6[:n_new])
    inter = sorted(set(fit_clips) & set(score_eids))
    return dict(ok=bool(len(fit_clips) == len(want) and set(fit_clips) == set(want) and len(set(fit_clips)) == len(want) and not inter), n=len(fit_clips), n_score_overlap=len(inter))


def check_curve(mine, ref, tol=1e-9):
    """C-CURVE7: r2, r and both CIs within tol of the FIT_CURVE row."""
    d = dict(r2=abs(mine["r2"] - ref["r2"]), r=abs(mine["r"] - ref["r"]), r2_ci=float(np.abs(np.array(mine["r2_ci"]) - np.array(ref["r2_ci"])).max()), r_ci=float(np.abs(np.array(mine["r_ci"]) - np.array(ref["r_ci"])).max()))
    return dict(ok=bool(max(d.values()) <= tol), max_abs_diff=float(max(d.values())), parts=d, tol=tol)


def check_k(p_k, p_ra, y, tol=1e-4):
    """K-control: KCTL's pooled R2 within 1e-4 of RA's and max |p_KCTL - p_RA| <= 1e-4 (scaled units)."""
    d = dict(dr2=abs(r2(p_k, y) - r2(p_ra, y)), max_abs_dp=float(np.abs(p_k - p_ra).max()))
    return dict(ok=bool(d["dr2"] <= tol and d["max_abs_dp"] <= tol), **d, tol=tol)


def check_fold(folds, n_clips):
    """C-FOLD7: for every outer fold the recorded training clips are disjoint from the fold's test clips and the two together are all clips."""
    bad = []
    for r in folds:
        tr, te = set(r["train_clips"]), set(r["test_clips"])
        if tr & te or (tr | te) != set(range(n_clips)):
            bad.append(r["f"])
    return dict(ok=not bad, bad_folds=bad)


def check_sel(folds, selection_key="selection"):
    """C-SEL7: every selection fit's validation clips lie inside its outer fold's training clips (and none in the test clips); the chosen config is the argmin of the recorded inner errors (a tie to the earlier config)."""
    bad = []
    for r in folds:
        sel = r.get(selection_key)
        if sel is None:
            continue
        tr, te = set(r["train_clips"]), set(r["test_clips"])
        for ft in sel["fits"]:
            v = set(ft["val_clips"])
            if not v <= tr or v & te:
                bad.append((r["f"], ft["g"], "val outside training clips"))
        best = None
        for e in sel["errors"]:
            if best is None or e["err"] < best["err"]:
                best = e
        if [best["epochs"], best["wd"]] != list(sel["chosen"]):
            bad.append((r["f"], "chosen is not the argmin"))
    return dict(ok=not bad, bad=bad[:5])


def const_closed_form(y, fo):
    """CONST's pooled R2 in closed form: each fold predicted by the mean of the other folds' rows."""
    yv = np.asarray(y, np.float64)
    sse = 0.0
    for f in range(5):
        c = yv[fo != f].mean()
        sse += ((yv[fo == f] - c) ** 2).sum()
    return float(1 - (sse / len(yv)) / ((yv - yv.mean()) ** 2).mean())


def check_const(pred, y, fo, tol=1e-12):
    cf = const_closed_form(y, fo)
    return dict(ok=bool(abs(r2(pred, y) - cf) <= tol), r2=r2(pred, y), closed_form=cf)


def check_zero(dR):
    return dict(ok=bool(np.all(np.asarray(dR) == 0.0)), max_abs=float(np.abs(dR).max()))


def check_wm_sign(r_ci):
    """C-WM7 sign part: r_wm(RA)'s CI lies wholly above 0."""
    return dict(ok=bool(r_ci[0] > 0), ci=list(map(float, r_ci)))


def check_roll_causal(roll_fn, rollout12_fn, z0, act, tol=1e-6):
    """C-WM7 roll part: the state after the 10-action `roll` equals the state after 10 steps of a 12-knot rollout (arbitrary knots 11-12). roll_fn(z0, a (B,1,10,2)) -> (B,1,D); rollout12_fn(z0, a (B,1,12,2)) -> (B,1,12,D)."""
    a12 = torch.cat([act, torch.randn(act.shape[0], 2, 2, dtype=act.dtype)], 1).unsqueeze(1)
    full = rollout12_fn(z0, a12)[:, :, 9]
    short = roll_fn(z0, act.unsqueeze(1))
    d = float((full - short).abs().max())
    return dict(ok=bool(d <= tol), max_abs=d, tol=tol)


def check_plant(r2_mlp, r2_oracle, dr2, M, p1_frac=0.70):
    """PLANT P1: MLP-A's R2 on y_plant >= 0.70 R2_oracle. P2: the paired dR2 (MLP-A - RA) CI wholly above 0 and its point above M_k. dr2 = dict(point, lo, hi)."""
    p1 = bool(r2_mlp >= p1_frac * r2_oracle)
    p2 = bool(dr2["lo"] > 0 and dr2["point"] > M)
    return dict(ok=bool(p1 and p2), P1=p1, P2=p2, r2_mlp=float(r2_mlp), r2_oracle=float(r2_oracle), dr2=dr2)


def check_plant_p1(r2_x, r2_oracle, p1_frac=0.70):
    """P3 (the regression arm): KCTL on y_plant must miss P1; this returns ok True iff it MEETS P1 (so the regression arm reads RED when KCTL misses)."""
    return dict(ok=bool(r2_x >= p1_frac * r2_oracle), r2=float(r2_x))


def check_shuf(r2_shuf):
    return dict(ok=bool(r2_shuf <= 0.0), r2=float(r2_shuf))
