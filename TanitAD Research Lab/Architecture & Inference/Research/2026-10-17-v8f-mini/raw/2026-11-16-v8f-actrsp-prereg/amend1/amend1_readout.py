"""ACTRSP amendment 1 (B-CAL triggered): what the level-1 readout of the action-response term actually reads.

TRAIN latents only (latents_train_R35_s{0,1}.pt, 141 TRAIN clips, the files B-CAL used); no eval window is read; nothing here is fit on a scored window.
Code of record for the model classes: a13f6f16 (stack/tanitad/models/hjepa.py blob 0cfd7081, v8f_mini.py blob f15c28c9).

Part 0  controls: (a) the trained head's R2 on the 31,575 WINDOW transitions reproduces PROBE_ACTSENS_RESULT.json partA idm_L1 (0.4894 / 0.6368 on s0,
        0.5139 / 0.6285 on s1); (b) latents and actions of one (clip, frame) are identical in every window that holds it (else the de-duplication is wrong).
Part A  the TRAINED readout I = stack.idm_heads[0] on the UNIQUE real transitions (z_t, z_t+1): its output with z_t+1 replaced by a counterfactual y while z_t
        is held fixed. Counterfactuals: STAY y = z_t; WM_MEAN y = F(z_t, a_mean) (the WM's own one-step prediction, context 1, under the TRAIN mean action);
        WM_TRUE y = F(z_t, a_t); TIME y = z_t + (z_s+1 - z_s), s another transition of the SAME clip at least 5 frames away (a grafted, shuffled-in-time
        step); DONOR y = z_t + (z'_t+1 - z'_t), the nearest z'_t of ANOTHER clip (grafted); DONOR_RAW y = z'_t+1.
        Per channel: s_full = Cov(I(z_t, z_t+1), a)/Var(a) (the card's slope, pooled), s_tr = Cov(I(z_t, z_t+1) - I(z_t, y), a)/Var(a) (the part of s_full
        that runs through the transition: s_full = s_state + s_tr exactly), s_state = Cov(I(z_t, y), a)/Var(a), R2 of I(z_t, y) (skill left without the
        transition), share_moved = E[(I - I_cf)^2]/Var(I), and for TIME / DONOR the matched slope s_match = Cov(dI, a - a_cf)/Var(a - a_cf) and the
        two-regressor fit I_cf ~ b_own * a + b_follow * a_cf.
Part C  the term at R35's end (TRAIN clips): anchors every 10 frames, K = 10, the four +/- delta contrasts, context 1, read by I exactly as hjepa.response_loss
        reads them; G with the card's slope s_full and with the candidate slopes; the loss mean (G - I)^2.
Part D  readouts that might see the transition, cross-fitted with folds grouped by CLIP (5 folds), on horizon h in {1, 2, 5, 10}: target = mean action
        over the h transitions; features S = z_t, P = [z_t, z_t+h], D = z_t+h - z_t, N = [z_t, z_t+h - F^h(z_t, a_mean)]; ridge (lambda chosen on inner
        clip-grouped folds) for S/P/D/N, an MLP (the IDM head's shape) for S/P. Incremental skill dR2 = R2(P) - R2(S) with a PAIRED clip-cluster bootstrap.
Estimator: clip(episode)-cluster percentile bootstrap, B = 1000, seed 0, over the 141 TRAIN clips. It answers ANOTHER DRAW OF CLIPS with the trained model
(and, in Part D, the fitted probes' held-out predictions) held fixed; it does not answer another training run.
Usage: python amend1_readout.py <k in 0|1> <out.json>
"""
import dataclasses, json, sys, time
import numpy as np
import torch

torch.set_num_threads(8)
sys.path.insert(0, "/root/actrsp_amend/stack")
from tanitad.models import v8f_mini as M  # noqa: E402

K_CH = int(sys.argv[1]); OUT = sys.argv[2]
TAG = f"R35_s{K_CH}"
T0 = time.time()
B_BOOT = 1000
CH = ("acc", "yaw_rate")


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def cfg_from_ckpt(ck):
    names = {f.name for f in dataclasses.fields(M.V8FMiniConfig)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items() if k in names}
    return M.V8FMiniConfig(**kw)


ck = torch.load(f"/root/actrsp_amend/data/ckpt_{TAG}.pt", map_location="cpu", weights_only=False)
cfg = cfg_from_ckpt(ck)
torch.manual_seed(0)
wm = M.WMAux(cfg)
wm.load_state_dict({k[3:]: v for k, v in ck["model"].items() if k.startswith("wm.")}, strict=True)
wm.eval()
st = wm.stack
head = st.idm_heads[0]
for p in wm.parameters():
    p.requires_grad_(False)
del ck

L = torch.load(f"/root/actrsp_amend/data/latents_train_{TAG}.pt", map_location="cpu", weights_only=False)
Zw = L["Z1"][TAG].float()                       # [N, 16, 128]
Aw = wm.scale_actions(L["A_raw"].float())       # [N, 16, 2]  WM units; A[j] is the action of the transition j -> j+1
okw = L["ok"].bool()
tw = np.asarray(L["t"]); eidw = list(L["eid"])
N, W = Zw.shape[:2]
res = {"tag": TAG, "file": f"latents_train_{TAG}.pt", "code": "a13f6f16 (hjepa 0cfd7081, v8f_mini f15c28c9)", "estimator": "clip-cluster percentile bootstrap, B 1000, seed 0, 141 TRAIN clips; answers another draw of clips, model held fixed", "parts": {}}

# ---------------- Part 0a: the window-set control (must reproduce PROBE_ACTSENS partA)
with torch.no_grad():
    vw = okw[:, :-1] & okw[:, 1:]
    ipw = head(Zw[:, :-1], Zw[:, 1:])[vw]
    aw = Aw[:, :-1][vw]
    r2w = (1 - (ipw - aw).pow(2).mean(0) / aw.var(0, unbiased=False)).tolist()
res["parts"]["control_window_r2"] = {"n": int(vw.sum()), "r2": dict(zip(CH, r2w)), "known": {"R35_s0": [0.4894, 0.6368], "R35_s1": [0.5139, 0.6285]}[TAG]}
log("control window R2", r2w)

# ---------------- Part 0b: per-clip frame tables (de-duplicated) + identity check
clips = sorted(set(eidw)); cix = {c: i for i, c in enumerate(clips)}
tab = {}           # (clip_i, frame) -> (z, a)
maxdz = maxda = 0.0
for w in range(N):
    ci = cix[eidw[w]]
    for j in range(W):
        if not bool(okw[w, j]):
            continue
        f = int(tw[w]) - (W - 1) + j
        z, a = Zw[w, j], Aw[w, j]
        if (ci, f) in tab:
            z0, a0 = tab[(ci, f)]
            maxdz = max(maxdz, float((z - z0).abs().max()))
            if torch.isfinite(a).all() and torch.isfinite(a0).all():
                maxda = max(maxda, float((a - a0).abs().max()))
        else:
            tab[(ci, f)] = (z, a)
res["parts"]["control_dedup"] = {"unique_frames": len(tab), "max_abs_dz_same_frame": maxdz, "max_abs_da_same_frame": maxda}
log("unique frames", len(tab), "max dz", maxdz, "max da", maxda)
seq = {}           # clip_i -> (frames sorted, Z [n,128], A [n,2])
for ci in range(len(clips)):
    fr = sorted(f for (c, f) in tab if c == ci)
    seq[ci] = (np.array(fr), torch.stack([tab[(ci, f)][0] for f in fr]), torch.stack([tab[(ci, f)][1] for f in fr]))


def pairs(h):
    """(clip, z_t, z_t+h, mean action over t..t+h-1, frame) for every t with frames t..t+h present and finite actions."""
    out_c, zt, zh, ah, idx = [], [], [], [], []
    for ci, (fr, Z, A) in seq.items():
        pos = {int(f): i for i, f in enumerate(fr)}
        for i, f in enumerate(fr):
            if all((int(f) + m) in pos for m in range(h + 1)):
                js = [pos[int(f) + m] for m in range(h)]
                am = A[js]
                if not torch.isfinite(am).all():
                    continue
                out_c.append(ci); zt.append(Z[i]); zh.append(Z[pos[int(f) + h]]); ah.append(am.mean(0)); idx.append((ci, i))
    return np.array(out_c), torch.stack(zt), torch.stack(zh), torch.stack(ah), idx


ep1, zt, zn, a1, idx1 = pairs(1)
n1 = len(ep1)
amean = a1.mean(0)
res["parts"]["n_unique_transitions"] = n1
res["parts"]["a_mean_wm"] = amean.tolist()
res["parts"]["a_sd_wm"] = a1.std(0, unbiased=False).tolist()
log("unique transitions", n1, "a_mean", amean.tolist())

byep = {}
for i, e in enumerate(ep1):
    byep.setdefault(int(e), []).append(i)
eps = np.array(sorted(byep))
byep_arr = [np.array(byep[e]) for e in eps]


def boot(fn, groups, B=B_BOOT, seed=0):
    """fn(idx) -> 1-D array; clip-cluster percentile bootstrap. Returns point, lo, hi."""
    pt = np.asarray(fn(np.concatenate(groups)), dtype=float)
    rng = np.random.RandomState(seed)
    bs = np.empty((B, pt.size))
    for b in range(B):
        pick = rng.randint(0, len(groups), len(groups))
        bs[b] = fn(np.concatenate([groups[g] for g in pick]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return pt, lo, hi


def cov(x, y):
    return ((x - x.mean(0)) * (y - y.mean(0))).mean(0)


def r2(p, y):
    return 1 - ((p - y) ** 2).mean(0) / ((y - y.mean(0)) ** 2).mean(0)


def pack(pt, lo, hi, names):
    return {nm: {"point": float(pt[i]), "ci95": [float(lo[i]), float(hi[i])]} for i, nm in enumerate(names)}


# ---------------- Part A: the trained readout with z_t+1 replaced
with torch.no_grad():
    I_real = head(zt, zn).numpy().astype(np.float64)
    roll1 = lambda z0, a: st.rollout(1, z0, a[:, None, :], context=1)[:, 0]
    y_wm_mean = roll1(zt, amean.expand(n1, 2).contiguous())
    y_wm_true = roll1(zt, a1)
    # TIME: another transition of the same clip, >= 5 frames away (fixed RNG)
    rng = np.random.RandomState(0)
    t_src = np.empty(n1, dtype=int)
    frame1 = np.array([int(seq[ci][0][i]) for (ci, i) in idx1])
    for e, g in zip(eps, byep_arr):
        for i in g:
            cand = g[np.abs(frame1[g] - frame1[i]) >= 5]
            t_src[i] = rng.choice(cand) if len(cand) else i
    y_time = zt + (zn[t_src] - zt[t_src])
    a_time = a1[t_src]
    # DONOR: nearest z_t of another clip (Euclidean), grafted step and raw next state
    d_src = np.empty(n1, dtype=int)
    epT = torch.as_tensor(ep1)
    for s in range(0, n1, 2048):
        d = torch.cdist(zt[s:s + 2048], zt)
        d[epT[s:s + 2048][:, None] == epT[None, :]] = float("inf")
        d_src[s:s + 2048] = d.argmin(1).numpy()
    y_donor = zt + (zn[d_src] - zt[d_src])
    y_donor_raw = zn[d_src]
    a_donor = a1[d_src]
    nn_dist = (zt - zt[d_src]).norm(dim=1)
    step_norm = (zn - zt).norm(dim=1)
    CF = {"STAY": (zt, None), "WM_MEAN": (y_wm_mean, None), "WM_TRUE": (y_wm_true, None), "TIME": (y_time, a_time), "DONOR": (y_donor, a_donor), "DONOR_RAW": (y_donor_raw, a_donor)}
    Icf = {k: head(zt, y).numpy().astype(np.float64) for k, (y, _) in CF.items()}
A1 = a1.numpy().astype(np.float64)
res["parts"]["geometry"] = {"real_step_norm_mean": float(step_norm.mean()), "donor_nn_dist_mean": float(nn_dist.mean()),
                            "donor_nn_dist_over_step_norm": float(nn_dist.mean() / step_norm.mean()),
                            "wm_mean_pred_err_norm": float((y_wm_mean - zn).norm(dim=1).mean()), "wm_true_pred_err_norm": float((y_wm_true - zn).norm(dim=1).mean()),
                            "wm_true_minus_wm_mean_norm": float((y_wm_true - y_wm_mean).norm(dim=1).mean()),
                            "donor_action_corr": [float(np.corrcoef(A1[:, c], A1[d_src, c])[0, 1]) for c in range(2)],
                            "time_action_corr": [float(np.corrcoef(A1[:, c], A1[t_src, c])[0, 1]) for c in range(2)]}
log("geometry", res["parts"]["geometry"])


def fn_real(ii):
    I, a = I_real[ii], A1[ii]
    s = cov(I, a) / a.var(0)
    return np.concatenate([s, r2(I, a)])


pt, lo, hi = boot(fn_real, byep_arr)
partA = {"real": pack(pt, lo, hi, ["s_full_acc", "s_full_yaw", "r2_acc", "r2_yaw"])}
for k, (y, acf) in CF.items():
    Ic = Icf[k]
    Acf = None if acf is None else acf.numpy().astype(np.float64)

    def fn(ii, Ic=Ic, Acf=Acf):
        I, a, c = I_real[ii], A1[ii], Ic[ii]
        va = a.var(0)
        s_full = cov(I, a) / va
        s_state = cov(c, a) / va
        s_tr = cov(I - c, a) / va
        out = [s_full, s_state, s_tr, s_tr / s_full, r2(c, a), ((I - c) ** 2).mean(0) / I.var(0)]
        if Acf is not None:
            ac = Acf[ii]
            da = a - ac
            out.append(cov(I - c, da) / da.var(0))
            bo, bf = [], []
            for ch in range(2):
                X = np.stack([a[:, ch] - a[:, ch].mean(), ac[:, ch] - ac[:, ch].mean()], 1)
                bco = np.linalg.lstsq(X, c[:, ch] - c[:, ch].mean(), rcond=None)[0]
                bo.append(bco[0]); bf.append(bco[1])
            out += [np.array(bo), np.array(bf)]
        return np.concatenate(out)
    names = [f"{m}_{c}" for m in ("s_full", "s_state", "s_tr", "s_tr_over_s_full", "r2_cf", "share_moved") for c in ("acc", "yaw")]
    if acf is not None:
        names += [f"{m}_{c}" for m in ("s_match", "b_own", "b_follow") for c in ("acc", "yaw")]
    pt, lo, hi = boot(fn, byep_arr)
    partA[k] = pack(pt, lo, hi, names)
    log("A", k, {n: round(float(v), 4) for n, v in zip(names, pt)})
res["parts"]["A_trained_readout"] = partA
json.dump(res, open(OUT, "w"), indent=1)

# ---------------- Part C: the term on R35 at its end, TRAIN clips (anchors every 10 frames, K 10)
KS, DLT = 10, torch.tensor(M.RSP_DELTA)
anc_z, anc_a, anc_ep = [], [], []
for ci, (fr, Z, A) in seq.items():
    pos = {int(f): i for i, f in enumerate(fr)}
    f0 = int(fr[0])
    for f in range(f0, int(fr[-1]) - KS + 1, KS):
        if all((f + m) in pos for m in range(KS + 1)):
            js = [pos[f + m] for m in range(KS)]
            if torch.isfinite(A[js]).all():
                anc_z.append(Z[pos[f]]); anc_a.append(A[js]); anc_ep.append(ci)
anc_z, anc_a, anc_ep = torch.stack(anc_z), torch.stack(anc_a), np.array(anc_ep)
with torch.no_grad():
    e = torch.zeros(4, 2); e[0, 0], e[1, 0], e[2, 1], e[3, 1] = 1, -1, 1, -1
    a4 = anc_a[None] + (e * DLT)[:, None, None, :]                       # (4, n, K, 2)
    z0x4 = anc_z[None].expand(4, -1, -1)
    zh = st.rollout(1, z0x4, a4, context=1)                             # (4, n, K, D)
    prev = torch.cat([z0x4[..., None, :], zh[..., :-1, :]], dim=-2)
    rd = head(prev, zh).numpy().astype(np.float64)                      # (4, n, K, 2)
num_lon = (rd[0] - rd[1]) / (2 * float(DLT[0]))                         # (n, K, 2): read change per unit acc command
num_lat = (rd[2] - rd[3]) / (2 * float(DLT[1]))
# per-step numerators at h = 1 and over all K
s_full_pt = np.array([partA["real"]["s_full_acc"]["point"], partA["real"]["s_full_yaw"]["point"]])
s_tr_wm = np.array([partA["WM_MEAN"]["s_tr_acc"]["point"], partA["WM_MEAN"]["s_tr_yaw"]["point"]])
s_tr_stay = np.array([partA["STAY"]["s_tr_acc"]["point"], partA["STAY"]["s_tr_yaw"]["point"]])
byanc = {}
for i, ep in enumerate(anc_ep):
    byanc.setdefault(int(ep), []).append(i)
anc_groups = [np.array(v) for v in byanc.values()]


def fnC(ii):
    nl, nt = num_lon[ii], num_lat[ii]
    return np.array([nl[..., 0].mean(), nt[..., 1].mean(), nl[..., 1].mean(), nt[..., 0].mean(), nl[:, 0, 0].mean(), nt[:, 0, 1].mean()])


pt, lo, hi = boot(fnC, anc_groups)
partC = {"n_anchors": int(len(anc_ep)), "n_clips": len(anc_groups),
         "numerator_per_unit_command": pack(pt, lo, hi, ["acc_on_acc_allK", "yaw_on_yaw_allK", "yaw_read_on_acc_cmd_allK", "acc_read_on_yaw_cmd_allK", "acc_on_acc_h1", "yaw_on_yaw_h1"])}
for nm, s in (("card_s_full", s_full_pt), ("s_tr_WM_MEAN", s_tr_wm), ("s_tr_STAY", s_tr_stay)):
    ss = np.where(s > 0.1, s, 0.1)
    g = np.stack([num_lon / ss, num_lat / ss], -1)                      # (n, K, c', c)
    loss = ((g - np.eye(2)) ** 2).mean()
    partC[nm] = {"slope_used_floored": ss.tolist(), "slope_raw": s.tolist(), "G_aa": float(g[..., 0, 0].mean()), "G_ww": float(g[..., 1, 1].mean()),
                 "G_aw": float(g[..., 0, 1].mean()), "G_wa": float(g[..., 1, 0].mean()), "rsp_loss": float(loss)}
res["parts"]["C_term_at_R35_end"] = partC
log("C", json.dumps(partC)[:1500])
json.dump(res, open(OUT, "w"), indent=1)


# ---------------- Part D: readouts that might see the transition (clip-grouped cross-fit)
def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd; ym = Y.mean(0)
    Wt = np.linalg.solve(Xs.T @ Xs + lam * len(X) * np.eye(X.shape[1]), Xs.T @ (Y - ym))
    return mu, sd, Wt, ym


def ridge_pred(m, X):
    mu, sd, Wt, ym = m
    return ((X - mu) / sd) @ Wt + ym


LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def ridge_cv(X, Y, fold):
    P = np.zeros_like(Y); lam_ch = []
    for f in range(5):
        tr, te = fold != f, fold == f
        inner = fold[tr]
        ufo = np.unique(inner)
        best, bl = np.inf, None
        for lam in LAMS:
            err = 0.0
            for g in ufo:
                m = ridge_fit(X[tr][inner != g], Y[tr][inner != g], lam)
                err += ((ridge_pred(m, X[tr][inner == g]) - Y[tr][inner == g]) ** 2).sum()
            if err < best:
                best, bl = err, lam
        lam_ch.append(bl)
        P[te] = ridge_pred(ridge_fit(X[tr], Y[tr], bl), X[te])
    return P, lam_ch


def mlp_cv(X, Y, fold, ep=12):
    P = np.zeros_like(Y)
    for f in range(5):
        tr, te = fold != f, fold == f
        torch.manual_seed(f)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs = torch.tensor((X - mu) / sd, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 2))
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
        idx = np.where(tr)[0]
        for e_ in range(ep):
            np.random.RandomState(e_).shuffle(idx)
            for i in range(0, len(idx), 256):
                b = idx[i:i + 256]
                loss = ((net(Xs[b]) - Yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            P[te] = net(Xs[te]).numpy()
    return P


fold_of_ep = np.random.RandomState(0).permutation(len(clips)) % 5
partD = {}
for h in (1, 2, 5, 10):
    eph, zth, zhh, ahh, _ = pairs(h)
    with torch.no_grad():
        zf = st.rollout(1, zth, amean.expand(len(eph), h, 2).contiguous(), context=1)[:, -1]
    Zt, Zh, Zf, Y = zth.numpy().astype(np.float64), zhh.numpy().astype(np.float64), zf.numpy().astype(np.float64), ahh.numpy().astype(np.float64)
    fold = fold_of_ep[eph]
    feats = {"S": Zt, "P": np.concatenate([Zt, Zh], 1), "D": Zh - Zt, "N": np.concatenate([Zt, Zh - Zf], 1)}
    preds, lams = {}, {}
    for nm, X in feats.items():
        preds[f"ridge_{nm}"], lams[nm] = ridge_cv(X, Y, fold)
    for nm in ("S", "P"):
        preds[f"mlp_{nm}"] = mlp_cv(feats[nm], Y, fold)
    gh = {}
    for i, e_ in enumerate(eph):
        gh.setdefault(int(e_), []).append(i)
    groups = [np.array(v) for v in gh.values()]
    keys = list(preds)

    def fnD(ii):
        y = Y[ii]
        r = {k: r2(preds[k][ii], y) for k in keys}
        return np.concatenate([r[k] for k in keys] + [r["ridge_P"] - r["ridge_S"], r["ridge_N"] - r["ridge_S"], r["mlp_P"] - r["mlp_S"]])
    names = [f"{k}_{c}" for k in keys for c in ("acc", "yaw")] + [f"{k}_{c}" for k in ("dR2_ridge_P_minus_S", "dR2_ridge_N_minus_S", "dR2_mlp_P_minus_S") for c in ("acc", "yaw")]
    pt, lo, hi = boot(fnD, groups)
    # the linear pair probe's transition-only slope (held-out predictions, WM_MEAN counterfactual) -- the calibration slope it would carry as a readout
    Pcf, _ = None, None
    Xcf = np.concatenate([Zt, Zf], 1)
    Pcf = np.zeros_like(Y)
    for f in range(5):
        tr, te = fold != f, fold == f
        m = ridge_fit(feats["P"][tr], Y[tr], lams["P"][f])
        Pcf[te] = ridge_pred(m, Xcf[te])

    def fnS(ii):
        y = Y[ii]; dP = preds["ridge_P"][ii] - Pcf[ii]
        return np.concatenate([cov(preds["ridge_P"][ii], y) / y.var(0), cov(dP, y) / y.var(0)])
    pt2, lo2, hi2 = boot(fnS, groups)
    partD[f"h{h}"] = {"n_pairs": int(len(eph)), "n_clips": len(groups), "lambda_chosen": {k: [float(x) for x in v] for k, v in lams.items()},
                      "r2": pack(pt, lo, hi, names), "ridge_P_slopes": pack(pt2, lo2, hi2, ["s_full_acc", "s_full_yaw", "s_tr_WM_MEAN_acc", "s_tr_WM_MEAN_yaw"])}
    log("D", h, {n: round(float(v), 4) for n, v in zip(names, pt)}, "slopes", pt2.round(4).tolist())
    res["parts"]["D_readouts_clip_grouped"] = partD
    json.dump(res, open(OUT, "w"), indent=1)
res["wall_s"] = time.time() - T0
json.dump(res, open(OUT, "w"), indent=1)
log("ZZ AMEND1 DONE ZZ")
