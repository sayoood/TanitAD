"""FULLRES probe (PREREG_V8F_PROOF_ADD9.md, frozen card). Tier T0: frozen-feature probes on logged frames, a perception diagnostic, never driving performance.
Zero training of any model; only probes are fitted. Reads the reduced tokens written by fullres_encode.py.
    python fullres_probe.py run     full read (GPU, behind the shared dev-box GPU lock for every CUDA section)
    python fullres_probe.py smoke   mechanics only (<= 10 episodes of feat/progress.json, 2 folds, 2 epochs); writes SMOKE_OK.txt and nothing else
Definitions and numbered deviations R1..: FULL_DEFS.md (same folder). Exit code 2 = a control missed its bar (results still written).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # ruleshield_read (label loader) would setdefault -1 at import
import argparse, gc, hashlib, json, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
V8 = "F:/TanitAD-scratch/v8proof"
for _p in (V8 + "/res", V8 + "/cue", V8 + "/rule", HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import resprobe as RP          # noqa: E402  numpy-only at import
import cueprobe as CP          # noqa: E402  numpy-only at import (torch / tanitad only inside rs())

FEAT, KERN, STAGES = HERE + "/feat", HERE + "/kern", HERE + "/stages"
KERN_IN_RAM = True      # R17: Gram matrices stay in memory (about 3.3 GB); F: is at the 20 GB floor of another session
HI_DEFAULT = "C:/Users/Admin/v8proof_tmp/full/HI_r128.i8"
CARD = V8 + "/PREREG_V8F_PROOF_ADD9.md"
RES_BANKED = V8 + "/res/RESULT_RESPROBE.json"
CUE_FRAMES = V8 + "/cue/cache/frames.npz"
BANK_FRAMES = V8 + "/res/cache/frames.npz"
R = 128
WHITE = 6.0 / 127.0
TNAMES = ["GAP", "NEAR", "COUNT", "LEADP", "STOPFLAG", "STOPDIST"]
KIND = {"GAP": "r2", "NEAR": "r2", "COUNT": "r2", "LEADP": "auc", "STOPFLAG": "auc", "STOPDIST": "mae"}
BINCOL = [KIND[t] == "auc" for t in TNAMES]
SETS = ["L40", "L160", "U640", "C640", "H1664"]
NTOK = {"L40": 40, "L160": 160, "U640": 640, "C640": 640, "H1664": 1664}
B_BOOT, N_SEEDS, MAX_EPOCHS, N_FOLDS = 2000, 3, 40, 5
STOP_NEAR_M = 40.0
TIER = "T0 (frozen-feature probe on logged frames): a perception diagnostic, never driving performance"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ===================================================================================== pure layer (unit-tested)
def pool_tokens(x, name):
    """x [b, Tfull, C] (LO: 640 = 16x40 row-major; HI: 1664 = 26x64 row-major) -> [b, T, C] for the feature set ``name``."""
    b, _, C = x.shape
    if name == "L40":
        return x.float().reshape(b, 4, 4, 10, 4, C).mean(dim=(2, 4)).reshape(b, 40, C)
    if name == "L160":
        return x.float().reshape(b, 8, 2, 20, 2, C).mean(dim=(2, 4)).reshape(b, 160, C)
    if name == "U640":
        assert x.shape[1] == 640
        return x
    if name == "C640":
        return x.reshape(b, 26, 64, C)[:, 5:21, 12:52].reshape(b, 640, C)
    if name == "H1664":
        assert x.shape[1] == 1664
        return x
    raise ValueError(name)


def map_rows(clips, g, clip_pos, row0, T):
    """Memmap row of (clip, g) = row0[clip] + g; ok = clip present in the index and 0 <= g < T[clip]. Returns (rows, ok)."""
    n = len(clips)
    rows, ok = np.zeros(n, np.int64), np.zeros(n, bool)
    for i in range(n):
        j = clip_pos.get(str(clips[i]))
        if j is not None and 0 <= int(g[i]) < int(T[j]):
            rows[i], ok[i] = int(row0[j]) + int(g[i]), True
    return rows, ok


def masked_loss(out, y, m, bincols):
    """Mean over targets of the per-target mean loss on the rows where that target is observed (m True); masked rows contribute nothing."""
    ls = []
    for t in range(out.shape[1]):
        mt = m[:, t]
        if bool(mt.any()):
            ls.append(F.binary_cross_entropy_with_logits(out[mt, t], y[mt, t]) if bincols[t] else F.mse_loss(out[mt, t], y[mt, t]))
    return torch.stack(ls).mean() if ls else out.sum() * 0.0


class AttnProbe(nn.Module):
    """P3: per-token MLP cin -> 64 (GELU); per target a linear score and a linear value per token; prediction = sum_tokens softmax(score) * value + linear(E)."""

    def __init__(self, cin, nt, ne, d=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cin, d), nn.GELU())
        self.score, self.value, self.elin = nn.Linear(d, nt), nn.Linear(d, nt), nn.Linear(ne, nt)

    def forward(self, x, e):
        h = self.mlp(x)
        w = torch.softmax(self.score(h), dim=1)
        return (w * self.value(h)).sum(1) + self.elin(e)


def p3_train_predict(get_x, cin, E, Y, M, bincols, tr, te, epi, seed, epochs, dev, d=64, bs=256, patience=5):
    """One multi-task P3 fit. Returns (pred [len(te), nt] (logits for binary, original units for regression), epochs used)."""
    nt = Y.shape[1]
    tr = tr[M[tr].any(1)]
    iv = RP.inner_split(epi[tr])
    fit_idx, val_idx = tr[~iv], tr[iv]
    ym, ys = np.zeros(nt), np.ones(nt)
    for t in range(nt):
        if not bincols[t]:
            v = Y[tr][M[tr, t], t]
            if len(v):
                ym[t], ys[t] = float(v.mean()), float(v.std() + 1e-8)
    Yn = np.where(M, (np.nan_to_num(Y) - ym) / ys, 0.0)
    Em, Es = E[tr].mean(0), E[tr].std(0) + 1e-9
    Yt = torch.tensor(Yn, dtype=torch.float32, device=dev)
    Mt = torch.tensor(M, device=dev)
    Et = torch.tensor((E - Em) / Es, dtype=torch.float32, device=dev)
    torch.manual_seed(seed)
    model = AttnProbe(cin, nt, E.shape[1], d).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    fit_t, val_t = torch.tensor(fit_idx), torch.tensor(val_idx)

    def outputs(idx_t):
        o = []
        with torch.no_grad():
            for ib in idx_t.split(512):
                ib = ib.to(dev)
                o.append(model(get_x(ib), Et[ib]))
        return torch.cat(o) if o else torch.zeros(0, nt, device=dev)

    best, best_sd, bad, used = float("inf"), None, 0, 0
    for ep in range(epochs):
        model.train()
        for ib in fit_t[torch.randperm(len(fit_t), generator=g)].split(bs):
            ib = ib.to(dev)
            loss = masked_loss(model(get_x(ib), Et[ib]), Yt[ib], Mt[ib], bincols)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        vi = val_t.to(dev)
        v = float(masked_loss(outputs(val_t), Yt[vi], Mt[vi], bincols))
        used = ep + 1
        if v < best - 1e-6:
            best, bad = v, 0
            best_sd = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_sd)
    model.eval()
    o = outputs(torch.tensor(te)).cpu().numpy().astype(np.float64)
    for t in range(nt):
        if not bincols[t]:
            o[:, t] = o[:, t] * ys[t] + ym[t]
    return o, used


def metric_value(kind, y, p):
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return float("nan")
    if kind == "r2":
        return RP.r2_pooled(y, p)
    if kind == "auc":
        return RP.auc_pooled(y, p)
    return -float(np.median(np.abs(y[m] - p[m])))       # "mae": higher is better (negative median absolute error)


class Boot:
    """Paired episode-cluster bootstrap, B draws, seed 0; the SAME draw for every pair."""

    def __init__(self, epi, n_ep, y_stop_mask, B=B_BOOT):
        self.epi, self.n_ep, self.B = epi, n_ep, B
        self.C = RP.boot_counts(n_ep, B)
        self.stop = y_stop_mask
        self.draws = CP.make_idx_draws(epi[y_stop_mask], B) if y_stop_mask.any() else None

    def vec(self, kind, y, p):
        if kind == "r2":
            return RP.boot_r2(RP.ep_stats_r2(self.epi, y, p, self.n_ep), self.C), RP.r2_pooled(y, p)
        return RP.boot_auc(self.epi, y, p, self.C), RP.auc_pooled(y, p)

    def pair(self, kind, y, pa, pb, mask=None):
        """a minus b; positive = a better (for 'mae' the median absolute error of b minus that of a)."""
        if kind == "mae":
            s = self.stop & np.isfinite(y)
            if mask is not None:
                s = s & mask
            ya, a, b = y[self.stop], pa[self.stop], pb[self.stop]
            keep = np.ones(len(ya), bool) if mask is None else mask[self.stop]
            r = CP.boot_median_gain(self.draws, np.abs(b - ya), np.abs(a - ya)) if keep.all() else \
                CP.boot_median_gain(CP.make_idx_draws(self.epi[self.stop][keep], self.B), np.abs(b - ya)[keep], np.abs(a - ya)[keep])
            return {"diff": r["gain"], "ci95": [r["lo"], r["hi"]], "separated_pos": bool(r["lo"] > 0), "separated_neg": bool(r["hi"] < 0), "n": int(r["n_windows"]),
                    "note": "median|err| of b minus median|err| of a, metres"}
        yy = y if mask is None else np.where(mask, y, np.nan)
        va, pa_ = self.vec(kind, yy, pa)
        vb, pb_ = self.vec(kind, yy, pb)
        row = RP.diff_row(va, vb, pa_, pb_)
        row["n"] = int((np.isfinite(yy) & np.isfinite(pa) & np.isfinite(pb)).sum())
        return row


def decide_fine_over_coarse(p3, p1):
    """Card rule: P3 decides; a 'fine beats coarse' claim needs P3 separated above 0 and P1 not separated in the opposite direction."""
    return bool(p3["separated_pos"] and not p1["separated_neg"])


# ===================================================================================== data table
def get_stop_labels(cache=True):
    """STOPFLAG / has / STOPDIST per cue window (plan seed 0), through cueprobe's own loaders (heavy: torch + tanitad tree; run with the cue job's PYTHONPATH)."""
    path = KERN + "/stoplabels.npz"
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    Z = np.load(CUE_FRAMES)
    win = Z["win"]
    assert len(win) == 2399
    x = CP.rs().load(0)
    for k in ("ci", "ws", "stop", "has", "plan", "or_xy"):
        x[k] = x[k][win]
    assert np.array_equal(x["ci"], Z["ci"]) and np.array_equal(x["ws"], Z["ws"]), "cue windows differ from the dump windows"
    stop, has = x["stop"].astype(bool), x["has"].astype(bool)
    d = np.full(len(win), np.nan)
    for i in np.flatnonzero(stop):
        v = CP.cue_d0(x["plan"][i], x["or_xy"][i])
        d[i] = np.nan if v is None else v
    assert np.isfinite(d[stop]).all()
    out = {"stop": stop, "has": has, "dist": d}
    if cache:
        os.makedirs(KERN, exist_ok=True)
        np.savez(path, **out)
    return out


def build_table(idx, bank, cue, lab, restrict=None, nf=N_FOLDS):
    clips = [str(c) for c in idx["clips"]]
    cpos = {c: i for i, c in enumerate(clips)}
    keep = np.isfinite(bank["E"]).all(1)
    beid, bg, Eb = bank["beid"].astype(str)[keep], bank["bg"][keep].astype(np.int64), bank["E"][keep]
    yb = {k: bank[k][keep].astype(np.float64) for k in ("lead_gap", "nearest", "agents30", "lead_present")}
    if restrict is not None:
        rs_ = set(restrict)
        keep_b = np.array([c in rs_ for c in beid])
        beid, bg, Eb, yb = beid[keep_b], bg[keep_b], Eb[keep_b], {k: v[keep_b] for k, v in yb.items()}
    ufold = RP.make_folds(beid, k=nf)
    fold_of = {c: int(f) for c, f in zip(beid, ufold)}
    wclip, wws = cue["clip"].astype(str), cue["ws"].astype(np.int64)
    wE = cue["E"]
    if restrict is not None:
        kw = np.array([c in set(restrict) for c in wclip])
    else:
        kw = np.ones(len(wclip), bool)
    rows_b, ok_b = map_rows(beid, bg, cpos, idx["row0"], idx["T"])
    rows_w, ok_w = map_rows(wclip, wws, cpos, idx["row0"], idx["T"])
    ok_w = ok_w & kw
    miss = sorted({c for c in wclip[ok_w] if c not in fold_of})
    for i, c in enumerate(miss):
        fold_of[c] = i % nf
    nb, nw = int(ok_b.sum()), int(ok_w.sum())
    N = nb + nw
    clip = np.concatenate([beid[ok_b], wclip[ok_w]])
    g = np.concatenate([bg[ok_b], wws[ok_w]])
    Y = np.full((N, 6), np.nan)
    Y[:nb, 0], Y[:nb, 1], Y[:nb, 2], Y[:nb, 3] = yb["lead_gap"][ok_b], yb["nearest"][ok_b], yb["agents30"][ok_b], yb["lead_present"][ok_b]
    stop, has, dist = lab["stop"][ok_w], lab["has"][ok_w], lab["dist"][ok_w]
    Y[nb:, 4] = np.where(has, stop.astype(np.float64), np.nan)
    Y[nb:, 5] = np.where(stop, dist, np.nan)
    ucl = sorted(set(clip))
    epi = np.array([ucl.index(c) for c in clip])
    E = np.concatenate([Eb[ok_b], wE[ok_w]])
    assert np.isfinite(E).all()
    return {"N": N, "nb": nb, "nw": nw, "clip": clip, "g": g, "mrow": np.concatenate([rows_b[ok_b], rows_w[ok_w]]), "E": E, "Y": Y, "M": np.isfinite(Y), "epi": epi,
            "n_ep": len(ucl), "folds": np.array([fold_of[c] for c in clip], np.int64), "nf": nf,
            "dropped": {"bank_not_in_index_or_out_of_range": int((~ok_b).sum()), "windows_dropped": int((~ok_w).sum()), "clips_added_to_fold_map": len(miss)},
            "is_window": np.arange(N) >= nb}


# ===================================================================================== token sets on the GPU
class TokSet:
    def __init__(self, name, X, scale):
        self.name, self.X, self.scale, self.n = name, X, scale, X.shape[0]

    def get(self, idx, space="white", plant=None):
        x = self.X[idx].float() * (self.scale if space == "true" else WHITE)
        if plant is not None:
            x = plant_add(x, idx, plant, self.name, space)
        return x


def plant_add(x, idx, plant, name, space):
    vec = plant["vec_white"] * (plant["sigma"] if space == "true" else 1.0)
    f = plant["flag"][idx].float()[:, None]
    pos = plant["pos"][idx]
    ar = torch.arange(x.shape[0], device=x.device)
    if name == "U640":
        x[ar, pos] += vec[None, :] * f
    elif name == "L40":
        p40 = (pos // 160) * 10 + (pos % 40) // 4
        x[ar, p40] += (vec[None, :] / 16.0) * f
    else:
        raise ValueError(name)
    return x


def build_tokset(name, rows, lo, hi, dev, scale, blk=256):
    src = hi if name in ("C640", "H1664") else lo
    dt = torch.float16 if name in ("L40", "L160") else torch.int8
    X = torch.empty((len(rows), NTOK[name], R), dtype=dt, device=dev)
    for s in range(0, len(rows), blk):
        r = rows[s:s + blk]
        xb = torch.from_numpy(np.ascontiguousarray(src[r])).to(dev)
        X[s:s + len(r)] = pool_tokens(xb, name).to(dt)
    return TokSet(name, X, scale)


def gram_gpu(ts, plant=None, blk=384):
    """Linear kernel on the flattened TRUE PCA coordinates (q * scale), row blocks, float32 on the GPU; returns float32 [n, n] on the CPU."""
    n = ts.n
    K = np.zeros((n, n), np.float32)
    for i0 in range(0, n, blk):
        i1 = min(n, i0 + blk)
        ai = ts.get(torch.arange(i0, i1, device=ts.X.device), "true", plant).reshape(i1 - i0, -1)
        for j0 in range(i0, n, blk):
            j1 = min(n, j0 + blk)
            aj = ai if j0 == i0 else ts.get(torch.arange(j0, j1, device=ts.X.device), "true", plant).reshape(j1 - j0, -1)
            kb = (ai @ aj.T).cpu().numpy()
            K[i0:i1, j0:j1] = kb
            if j0 != i0:
                K[j0:j1, i0:i1] = kb.T
    return K


# ===================================================================================== probes
def p1_cv(K, E, Y, M, epi, folds, nf, tcols):
    """RESPROBE P1 by import: kernel PCA rank 256 on the training folds (RP.kpca_fit), +E block (RP.e_block), kernel ridge (RP.fit_p1). Returns {col: oof [N]}."""
    N = len(E)
    oof = {t: np.full(N, np.nan) for t in tcols}
    for f in range(nf):
        te, tr = np.flatnonzero(folds == f), np.flatnonzero(folds != f)
        proj, _ = RP.kpca_fit(K, tr, rank=min(RP.RANK, max(8, len(tr) // 3)))
        Str, Ste = proj(tr), proj(te)
        Eb = RP.e_block(E, tr, float(Str.var(0).mean()))
        Ftr, Fte = np.concatenate([Str, Eb[tr]], 1), np.concatenate([Ste, Eb[te]], 1)
        for t in tcols:
            mtr, mte = M[tr, t], M[te, t]
            if mtr.sum() < 8 or not mte.any():
                continue
            oof[t][te[mte]] = RP.fit_p1(Ftr[mtr], Y[tr, t][mtr], epi[tr][mtr])(Fte[mte])
    return oof


def fit_logit(Z, y, l2=1e-3):
    X = torch.tensor(Z, dtype=torch.float32)
    Yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=100, line_search_fn="strong_wolfe")

    def clo():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(X @ w + b, Yt) + l2 * (w * w).sum()
        l.backward()
        return l
    opt.step(clo)
    w0, b0 = w.detach().numpy().astype(np.float64), float(b.detach())
    return lambda Zn: Zn @ w0 + b0


def e_baseline(E, Y, M, epi, folds, nf):
    """Ego-only baseline per target: the better (pooled metric) of RESPROBE's ego-only probes (P1 ridge, P2 MLP) and a linear / logistic fit on E (addendum 1 rule)."""
    N = len(E)
    ff = []
    for f in range(nf):
        te, tr = np.flatnonzero(folds == f), np.flatnonzero(folds != f)
        mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-9
        ff.append(((E[tr] - mu) / sd, (E[te] - mu) / sd, tr, te))
    best = np.full((N, 6), np.nan)
    detail = {}
    for t, nm in enumerate(TNAMES):
        valid, binary = M[:, t], BINCOL[t]
        if valid.sum() < 20:
            continue
        cand = {"P1": RP.run_probe_cv(ff, Y[:, t], epi, folds, "P1", binary, valid), "P2": RP.run_probe_cv(ff, Y[:, t], epi, folds, "P2", binary, valid)}
        lin = np.full(N, np.nan)
        for (Ftr, Fte, tr, te) in ff:
            mtr, mte = valid[tr], valid[te]
            if binary:
                lin[te[mte]] = fit_logit(Ftr[mtr], Y[tr, t][mtr])(Fte[mte])
            else:
                A = np.concatenate([Ftr[mtr], np.ones((mtr.sum(), 1))], 1)
                w = np.linalg.lstsq(A, Y[tr, t][mtr], rcond=None)[0]
                lin[te[mte]] = np.concatenate([Fte[mte], np.ones((mte.sum(), 1))], 1) @ w
        cand["linear" if not binary else "logistic"] = lin
        sc = {k: metric_value(KIND[nm], Y[:, t], v) for k, v in cand.items()}
        pick = max(sc, key=lambda k: (-1e9 if not np.isfinite(sc[k]) else sc[k]))
        best[:, t] = cand[pick]
        detail[nm] = {"scores": sc, "chosen": pick}
    return best, detail


def per_fold(kind, y, p, folds, nf):
    return [metric_value(kind, np.where(folds == f, y, np.nan), p) for f in range(nf)]


# ===================================================================================== P3 over a feature set
def cfg_id(tbl, epochs, tag):
    return f"{tag}|N{tbl['N']}|nf{tbl['nf']}|ep{epochs}|seeds{N_SEEDS}|v1"


def p3_cv(prefix, get_x, cin, tbl, Y, M, bincols, epochs, dev, rows_ok=None, seeds=N_SEEDS):
    """5 folds x seeds P3 fits (stage checkpoints per fit); returns oof [N, nt] = test predictions averaged over the seeds."""
    N, nt = tbl["N"], Y.shape[1]
    E, epi, folds = tbl["E"], tbl["epi"], tbl["folds"]
    ok = np.ones(N, bool) if rows_ok is None else rows_ok
    Mx = M & ok[:, None]
    oof = np.full((N, nt), np.nan)
    cid = cfg_id(tbl, epochs, prefix)
    for f in range(tbl["nf"]):
        te = np.flatnonzero((folds == f) & ok)
        tr = np.flatnonzero((folds != f) & ok)
        acc = []
        for s in range(seeds):
            sp = f"{STAGES}/{prefix}_f{f}_s{s}.npz"
            if os.path.exists(sp):
                z = np.load(sp)
                if str(z["cfg"]) == cid and len(z["pred"]) == len(te):
                    acc.append(z["pred"])
                    continue
            t0 = time.time()
            pred, used = p3_train_predict(get_x, cin, E, Y, Mx, bincols, tr, te, epi, seed=1000 * f + s, epochs=epochs, dev=dev)
            assert np.isfinite(pred).all(), "non-finite P3 prediction"
            np.savez(sp, pred=pred, cfg=cid, epochs=used)
            acc.append(pred)
            log("P3", prefix, "fold", f, "seed", s, "epochs", used, f"{time.time() - t0:.0f}s")
        oof[te] = np.mean(acc, 0)
    oof[~Mx] = np.nan
    return oof


# ===================================================================================== evaluation
def evaluate(tbl, OOF, Eb, edetail, boot):
    Y, M, folds, nf = tbl["Y"], tbl["M"], tbl["folds"], tbl["nf"]
    res = {"metrics": {}, "pairs": {}}
    names = ["E"] + SETS
    for probe in ("P3", "P1"):
        res["metrics"][probe] = {}
        for s in names:
            P = Eb if s == "E" else OOF[probe].get(s)
            if P is None:
                continue
            res["metrics"][probe][s] = {}
            for t, nm in enumerate(TNAMES):
                y = np.where(M[:, t], Y[:, t], np.nan)
                p = np.where(M[:, t], P[:, t], np.nan)
                if np.isfinite(p).sum() < 3:
                    continue
                d = {"value": metric_value(KIND[nm], y, p), "n": int(np.isfinite(p).sum()), "per_fold": per_fold(KIND[nm], y, p, folds, nf)}
                if nm == "STOPDIST":
                    d["median_abs_err_m"] = -d["value"]
                    d["r2"] = RP.r2_pooled(y, p)
                res["metrics"][probe][s][nm] = d
    pairs = [(s, "E") for s in SETS] + [(s, "L40") for s in SETS if s != "L40"] + [("U640", "L160"), ("H1664", "U640")]
    for probe in ("P3", "P1"):
        res["pairs"][probe] = {}
        for t, nm in enumerate(TNAMES):
            y = np.where(M[:, t], Y[:, t], np.nan)
            res["pairs"][probe][nm] = {}
            for a, b in pairs:
                pa = Eb if a == "E" else OOF[probe].get(a)
                pb = Eb if b == "E" else OOF[probe].get(b)
                if pa is None or pb is None or np.isfinite(pa[:, t]).sum() < 3 or np.isfinite(pb[:, t]).sum() < 3:
                    continue
                res["pairs"][probe][nm][f"{a} minus {b}"] = boot.pair(KIND[nm], y, np.where(M[:, t], pa[:, t], np.nan), np.where(M[:, t], pb[:, t], np.nan))
    # P4c: STOPFLAG gain of C640+E over E on stop windows with rest distance < 40 m vs the others
    t = TNAMES.index("STOPFLAG")
    dist = tbl["Y"][:, 5]
    stop = tbl["is_window"] & (tbl["Y"][:, 4] == 1)
    near, farm = stop & (dist < STOP_NEAR_M), stop & ~(dist < STOP_NEAR_M)
    mA = M[:, t] & (~stop | near)
    mB = M[:, t] & (~stop | farm)
    p4c = {"n_stop_under_40m": int(near.sum()), "n_stop_ge_40m": int(farm.sum())}
    for probe in ("P3", "P1"):
        pa = OOF[probe]["C640"][:, t]
        yA, yB = np.where(mA, Y[:, t], np.nan), np.where(mB, Y[:, t], np.nan)
        vcA, pcA = boot.vec("auc", yA, pa)
        veA, peA = boot.vec("auc", yA, Eb[:, t])
        vcB, pcB = boot.vec("auc", yB, pa)
        veB, peB = boot.vec("auc", yB, Eb[:, t])
        gA, gB = vcA - veA, vcB - veB
        p4c[probe] = {"gain_under_40m": float(pcA - peA), "gain_ge_40m": float(pcB - peB), "gain_under_minus_ge": float((pcA - peA) - (pcB - peB)),
                      "ci95_of_that_difference": RP.ci(gA - gB), "gain_under_ci95": RP.ci(gA), "gain_ge_ci95": RP.ci(gB)}
    res["P4c"] = p4c
    return res


def outcomes(res):
    P3, P1 = res["pairs"]["P3"], res["pairs"]["P1"]
    out = {}
    g3, g1 = P3["GAP"], P1["GAP"]
    k = "U640 minus L160"
    out["P4a_full"] = {"P3": g3[k], "P1": g1[k], "holds": decide_fine_over_coarse(g3[k], g1[k])}
    k = "H1664 minus U640"
    out["P4a_res"] = {"P3": g3[k], "P1": g1[k], "holds": decide_fine_over_coarse(g3[k], g1[k])}
    out["GAP_pooling"] = "UNPOOL-HELPS" if out["P4a_full"]["holds"] else "UNPOOL-FLAT"
    out["GAP_resolution"] = "RES-HELPS" if out["P4a_res"]["holds"] else "RES-FLAT"
    mets = res["metrics"]["P3"]
    bestfine = max(("U640", "H1664"), key=lambda s: mets[s]["GAP"]["value"])
    kk = f"{bestfine} minus L40"
    r3, r1 = g3[kk], g1[kk]
    big = r3["diff"] >= 0.10 and r3["separated_pos"] and not r1["separated_neg"]
    neither = all(not (g3[f"{s} minus L40"]["separated_pos"] or g3[f"{s} minus L40"]["separated_neg"]) for s in ("U640", "H1664"))
    out["GAP_limit"] = {"label": "INTERFACE-LIMITED" if big else ("ENCODER-LIMITED" if neither else "INTERFACE-PARTIAL"), "best_fine_set": bestfine, "P3": r3, "P1": r1}
    fs = P3["STOPFLAG"]
    found = [s for s in ("U640", "C640", "H1664") if fs[f"{s} minus E"]["separated_pos"]]
    sd = P3["STOPDIST"]
    bestd = min(("U640", "C640", "H1664"), key=lambda s: -mets[s]["STOPDIST"]["value"] if "STOPDIST" in mets[s] else 1e9)
    dist_found = bool(sd[f"{bestd} minus E"]["separated_pos"])
    names = (["CUE-FOUND"] if found else []) + (["DIST-FOUND"] if dist_found else [])
    out["stop_cue"] = {"labels": names or ["CUE-ABSENT-FULLRES"], "flag_sets_separated_above_E": found, "best_dist_set": bestd, "dist_P3": sd[f"{bestd} minus E"],
                       "P4b_C640": {"P3": fs["C640 minus E"], "P1": P1["STOPFLAG"]["C640 minus E"]}, "P4b_L40": {"P3": fs["L40 minus E"], "P1": P1["STOPFLAG"]["L40 minus E"]},
                       "P4b_holds": bool(fs["C640 minus E"]["separated_pos"] and not fs["L40 minus E"]["separated_pos"])}
    out["temporal_trigger"] = bool(out["GAP_limit"]["label"] == "ENCODER-LIMITED" or out["GAP_pooling"] == "UNPOOL-FLAT")
    return out


# ===================================================================================== controls
def controls_f1_f2_f3(K40c, bank, P1_L40, tbl):
    out = {}
    keep = np.isfinite(bank["E"]).all(1)
    idx_keep = np.flatnonzero(keep)
    beid, E = bank["beid"].astype(str)[keep], bank["E"][keep]
    Kb = np.asarray(K40c[np.ix_(idx_keep, idx_keep)])
    uep = sorted(set(beid))
    epi = np.array([uep.index(e) for e in beid])
    folds = RP.make_folds(beid)
    yg = bank["lead_gap"][keep].astype(np.float64)
    Yb = yg[:, None]
    oof = p1_cv(Kb, E, Yb, np.isfinite(Yb), epi, folds, N_FOLDS, [0])[0]
    mine = RP.r2_pooled(yg, oof)
    banked = json.load(open(RES_BANKED, encoding="utf-8"))["metric"]
    out["f1"] = {"mine_GAP_R2_V40+E": mine, "banked": banked["GAP|V40+E|P1"], "abs_diff": abs(mine - banked["GAP|V40+E|P1"]), "bar": 1e-6, "pass": bool(abs(mine - banked["GAP|V40+E|P1"]) <= 1e-6)}
    return out, banked


def f2_read():
    p = FEAT + "/F2_RECIPE.json"
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"missing": True}


# ===================================================================================== run
def to_py(o):
    return CP.to_py(o)


def load_inputs(smoke):
    idx = np.load(FEAT + "/index.npz")
    bank = dict(np.load(BANK_FRAMES))
    cue = dict(np.load(CUE_FRAMES))
    restrict = None
    if smoke:
        done = json.load(open(FEAT + "/progress.json", encoding="utf-8"))
        restrict = done[:139]
    lab = get_stop_labels(cache=not smoke)
    tbl = build_table(idx, bank, cue, lab, restrict=restrict, nf=2 if smoke else N_FOLDS)
    return idx, bank, tbl


def open_maps(idx):
    done = FEAT + "/ENCODE_DONE.json"
    hi = HI_DEFAULT
    if os.path.exists(done):
        hi = json.load(open(done, encoding="utf-8")).get("hi", HI_DEFAULT)
    N = int(idx["N"])
    lo = np.memmap(FEAT + "/LO_r128.i8", dtype=np.int8, mode="r", shape=(N, 640, R))
    hm = np.memmap(hi, dtype=np.int8, mode="r", shape=(N, 1664, R))
    return lo, hm


def make_plant(N, ts_u640, dev, seed=5):
    rng = np.random.default_rng(seed)
    flag = np.zeros(N, bool)
    flag[rng.permutation(N)[:N // 2]] = True
    pos = rng.integers(0, 640, N)
    v = rng.standard_normal(R)
    samp = torch.tensor(np.sort(rng.choice(N, min(N, 512), replace=False)), device=dev)
    med = float(ts_u640.get(samp, "white").norm(dim=-1).flatten().median())
    v = v / np.linalg.norm(v) * med
    sigma = ts_u640.scale * 127.0 / 6.0
    return {"flag": torch.tensor(flag, device=dev), "pos": torch.tensor(pos, device=dev), "vec_white": torch.tensor(v, dtype=torch.float32, device=dev), "sigma": sigma,
            "median_token_norm": med, "flag_np": flag}


def gpu_stage(tbl, lo, hm, P, dev, epochs, smoke, Yperm, OOF, f4, f5, KS):
    """Every CUDA section of the main read, behind the shared GPU lock: token sets, Grams, P3 for all sets, f4 and f5 fits."""
    from fullres_encode import take_lock, release
    N = tbl["N"]
    Y, M = tbl["Y"], tbl["M"]
    lock = take_lock("v8proof-fullres-probe")
    plant = None
    try:
        scale = torch.tensor(P["scale"], dtype=torch.float32, device=dev)
        for s in ["U640", "L40", "L160", "C640", "H1664"]:
            t0 = time.time()
            ts = build_tokset(s, tbl["mrow"], lo, hm, dev, scale)
            assert torch.isfinite(ts.get(torch.arange(min(N, 8), device=dev))).all()
            log("set", s, tuple(ts.X.shape), ts.X.dtype, f"{time.time() - t0:.0f}s")
            kp = None if (smoke or KERN_IN_RAM) else f"{KERN}/K_{s}.npy"
            if kp is None or not os.path.exists(kp):
                K = gram_gpu(ts)
                if kp:
                    np.save(kp, K)
                else:
                    KS[s] = K
                del K
            gx = lambda ib, ts=ts: ts.get(ib)
            OOF["P3"][s] = p3_cv(f"P3_{s}", gx, R, tbl, Y, M, BINCOL, epochs, dev)
            if s in ("L40", "U640"):
                f4[s] = p3_cv(f"F4_{s}", gx, R, tbl, Yperm, M, BINCOL, epochs, dev)
            if s == "U640":
                plant = make_plant(N, ts, dev)
            if s in ("L40", "U640"):
                Y1, M1 = plant["flag_np"].astype(np.float64)[:, None], np.ones((N, 1), bool)
                f5["P3_" + s] = p3_cv("F5_" + s, lambda ib, ts=ts: ts.get(ib, "white", plant), R, tbl, Y1, M1, [True], epochs, dev)[:, 0]
            if s == "U640":
                kp5 = None if (smoke or KERN_IN_RAM) else f"{KERN}/K_U640_plant.npy"
                if kp5 is None or not os.path.exists(kp5):
                    K = gram_gpu(ts, plant)
                    if kp5:
                        np.save(kp5, K)
                    else:
                        KS["plant"] = K
                    del K
            del ts, gx      # gx holds the token set through its default argument (R16)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        from fullres_encode import used_mib
        gc.collect()
        torch.cuda.empty_cache()
        log("gpu stage end: GPU used MiB", used_mib())
        release(lock)
    return plant


def cmd_run(a, smoke=False):
    t00 = time.time()
    os.makedirs(KERN, exist_ok=True)
    os.makedirs(STAGES, exist_ok=True)
    if not smoke:
        assert os.path.exists(FEAT + "/ENCODE_DONE.json"), "encode not finished"
    idx, bank, tbl = load_inputs(smoke)
    N, nf = tbl["N"], tbl["nf"]
    epochs = 2 if smoke else MAX_EPOCHS
    log("table", {k: tbl[k] for k in ("N", "nb", "nw", "n_ep", "dropped")})
    if not smoke:
        log("n per target", {t: int(tbl["M"][:, i].sum()) for i, t in enumerate(TNAMES)})
    P = np.load(FEAT + "/pca.npz")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GPU required"
    lo, hm = open_maps(idx)
    Y, M, E = tbl["Y"], tbl["M"], tbl["E"]
    boot = Boot(tbl["epi"], tbl["n_ep"], np.isfinite(Y[:, 5]), B=200 if smoke else B_BOOT)
    # ---- stage A (CPU): controls f1, f2 first, then the ego baselines
    ctl = {}
    t0 = time.time()
    if not smoke:
        c, _ = controls_f1_f2_f3(np.load(V8 + "/res/cache/K40.npy", mmap_mode="r"), bank, None, tbl)
        ctl.update(c)
        ctl["f2"] = f2_read()
        log("CONTROL f1", ctl["f1"], "f2", ctl["f2"])
    Eb, edetail = e_baseline(E, Y, M, tbl["epi"], tbl["folds"], nf)
    log("ego baselines done", f"{time.time() - t0:.0f}s", "" if smoke else {k: v["chosen"] for k, v in edetail.items()})
    # ---- stage B (GPU, lock held): Grams, P3 for every set, f4, f5
    rng = np.random.default_rng(4)
    Yperm = Y.copy()
    for t in range(6):
        for f in range(nf):
            ii = np.flatnonzero((tbl["folds"] == f) & M[:, t])
            Yperm[ii, t] = Y[rng.permutation(ii), t]
    OOF, f4, f5, KS = {"P3": {}, "P1": {}}, {}, {}, {}
    plant = gpu_stage(tbl, lo, hm, P, dev, epochs, smoke, Yperm, OOF, f4, f5, KS)
    # ---- stage C (CPU): P1 on every Gram (f3 reads L40), f5 P1
    for s in SETS:
        t0 = time.time()
        K = KS[s] if (smoke or KERN_IN_RAM) else np.load(f"{KERN}/K_{s}.npy")
        d = p1_cv(K, E, Y, M, tbl["epi"], tbl["folds"], nf, list(range(6)))
        OOF["P1"][s] = np.stack([d[t] for t in range(6)], 1)
        log("P1", s, f"{time.time() - t0:.0f}s")
        del K
    Y1, M1 = plant["flag_np"].astype(np.float64)[:, None], np.ones((N, 1), bool)
    K = KS["plant"] if (smoke or KERN_IN_RAM) else np.load(f"{KERN}/K_U640_plant.npy")
    f5["P1_U640"] = p1_cv(K, E, Y1, M1, tbl["epi"], tbl["folds"], nf, [0])[0]
    del K
    if smoke:
        for pr in ("P3", "P1"):
            for s in SETS:
                arr = OOF[pr][s]
                obs = arr[np.isfinite(arr)]
                assert obs.size > 0, (pr, s)
                log("smoke finite-check", pr, s, "shape", arr.shape, "n finite", int(obs.size), "all finite values finite: True")
        for k, v in list(f5.items()):
            assert np.isfinite(v).sum() > 0, k
            log("smoke f5 path", k, "shape", v.shape)
        for s, v in f4.items():
            assert np.isfinite(v).sum() > 0, s
            log("smoke f4 path", s, "shape", v.shape)
        run_temporal(tbl, lo, hm, P, dev, epochs, smoke=True)
        with open(HERE + "/SMOKE_OK.txt", "w", encoding="utf-8") as fh:
            fh.write("smoke ok " + time.strftime("%Y-%m-%dT%H:%M:%S") + " N=%d folds=%d\n" % (N, nf))
        log("ZZ SMOKE-OK")
        return 0
    res = {"meta": {"card": "PREREG_V8F_PROOF_ADD9.md", "tier": TIER, "evidence_class": "MEASURED", "smoke": False}}
    ev = evaluate(tbl, OOF, Eb, edetail, boot)
    res.update(ev)
    res["outcomes"] = outcomes(ev)
    res["E_baseline"] = edetail
    res["config"] = {"B": B_BOOT, "seeds": N_SEEDS, "max_epochs": MAX_EPOCHS, "folds": nf, "rank": RP.RANK, "reduction": "PCA r128 int8 6 sigma",
                     "table": {k: tbl[k] for k in ("N", "nb", "nw", "n_ep", "dropped")},
                     "n_per_target": {t: int(M[:, i].sum()) for i, t in enumerate(TNAMES)},
                     "n_episodes_per_target": {t: int(len(set(tbl["epi"][M[:, i]]))) for i, t in enumerate(TNAMES)},
                     "sha256": {"card": sha256(CARD), "index.npz": sha256(FEAT + "/index.npz"), "pca.npz": sha256(FEAT + "/pca.npz")}}
    mt = res["metrics"]["P1"]["L40"]
    bk = json.load(open(RES_BANKED, encoding="utf-8"))["metric"]
    ctl["f3"] = {"GAP_R2_L40+E": mt["GAP"]["value"], "banked_GAP_V40+E": bk["GAP|V40+E|P1"], "GAP_diff": mt["GAP"]["value"] - bk["GAP|V40+E|P1"], "GAP_bar": 0.05,
                 "LEADP_AUROC_L40+E": mt["LEADP"]["value"], "banked_LEADP_V40+E": bk["LEADP|V40+E|P1"], "LEADP_diff": mt["LEADP"]["value"] - bk["LEADP|V40+E|P1"], "LEADP_bar": 0.03}
    ctl["f3"]["pass"] = bool(abs(ctl["f3"]["GAP_diff"]) <= 0.05 and abs(ctl["f3"]["LEADP_diff"]) <= 0.03)
    f4r, f4pass = {}, True
    for s in ("L40", "U640"):
        r = {}
        for t, nm in enumerate(TNAMES):
            y, p = np.where(M[:, t], Yperm[:, t], np.nan), np.where(M[:, t], f4[s][:, t], np.nan)
            if np.isfinite(p).sum() < 3:
                continue
            if KIND[nm] == "auc":
                pf = per_fold("auc", y, p, tbl["folds"], nf)
                r[nm] = {"mean_per_fold_AUROC": float(np.nanmean(pf)), "per_fold": pf, "pass": bool(abs(np.nanmean(pf) - 0.5) <= 0.06)}
            else:
                r[nm] = {"R2": RP.r2_pooled(y, p), "pass": bool(RP.r2_pooled(y, p) <= 0.02)}
            f4pass &= r[nm]["pass"]
        f4r[s] = r
    ctl["f4"] = {"sets": f4r, "pass": bool(f4pass)}
    pl = plant["flag_np"].astype(np.float64)
    a_u, a_l, a_p1 = RP.auc_pooled(pl, f5["P3_U640"]), RP.auc_pooled(pl, f5["P3_L40"]), RP.auc_pooled(pl, f5["P1_U640"])
    ctl["f5"] = {"P3_U640_AUROC": a_u, "bar": 0.95, "P1_U640_AUROC_beside": a_p1, "P3_L40_AUROC_beside": a_l, "median_token_norm_white": plant["median_token_norm"], "pass": bool(a_u >= 0.95)}
    res["controls"] = ctl
    res["controls_pass"] = bool(ctl["f1"]["pass"] and ctl["f3"]["pass"] and ctl["f4"]["pass"] and ctl["f5"]["pass"])
    log("CONTROLS", json.dumps(to_py({k: v.get("pass") for k, v in ctl.items() if isinstance(v, dict) and "pass" in v})), json.dumps(to_py(ctl["f5"])))
    if not ctl["f3"]["pass"]:
        res["outcomes"]["INVALID"] = "INVALID-AS-DESIGNED: f3 missed its bar; the read is rerun with r = 256 (card)"
    if res["outcomes"]["temporal_trigger"]:
        res["temporal"] = run_temporal(tbl, lo, hm, P, dev, epochs, OOF=OOF, boot=boot)
    else:
        res["temporal"] = {"run": False, "reason": "GAP outcome is neither ENCODER-LIMITED nor UNPOOL-FLAT"}
    res["meta"]["runtime_s"] = time.time() - t00
    np.savez(HERE + "/OOF_FULLRES.npz", **{f"{p}_{s}": OOF[p][s] for p in OOF for s in OOF[p]}, E=Eb, Y=Y, M=M, epi=tbl["epi"], folds=tbl["folds"], is_window=tbl["is_window"])
    json.dump(to_py(res), open(HERE + "/RESULT_FULLRES.json", "w", encoding="utf-8"), indent=1)
    open(HERE + "/VERDICT_FULLRES.md", "w", encoding="utf-8").write(verdict_md(to_py(res)))
    log("WROTE RESULT_FULLRES.json VERDICT_FULLRES.md controls_pass", res["controls_pass"])
    return 0 if res["controls_pass"] else 2


def run_temporal(tbl, lo, hm, P, dev, epochs, smoke=False, OOF=None, boot=None):
    """P3 on U640 of frame g with frame g-5 (1,280 tokens + a frame-bit channel); rows with g < 5 dropped; same targets and folds. CUDA behind the lock."""
    from fullres_encode import take_lock, release
    N = tbl["N"]
    ok = tbl["g"] >= 5
    prev = np.where(ok, tbl["mrow"] - 5, tbl["mrow"])
    from fullres_encode import used_mib
    t_w = time.time()
    while used_mib() >= 3500 and time.time() - t_w < 1800:      # R16: bounded wait, never an open-ended one
        time.sleep(30)
    if used_mib() >= 3500:
        log("temporal not run: GPU used MiB", used_mib(), "not below the idle bar after 1800 s")
        return None if smoke else {"run": False, "reason": "named blocker: GPU memory not below the idle bar of the shared lock (3,500 MiB) within 1,800 s"}
    lock = take_lock("v8proof-fullres-probe-temporal")
    try:
        scale = torch.tensor(P["scale"], dtype=torch.float32, device=dev)
        ts0 = build_tokset("U640", tbl["mrow"], lo, hm, dev, scale)
        ts1 = build_tokset("U640", prev, lo, hm, dev, scale)

        def gx(ib):
            a, b = ts0.get(ib), ts1.get(ib)
            z, o = torch.zeros(a.shape[0], a.shape[1], 1, device=a.device), torch.ones(a.shape[0], a.shape[1], 1, device=a.device)
            return torch.cat([torch.cat([a, z], 2), torch.cat([b, o], 2)], 1)
        oof = p3_cv("TEMP_U640", gx, R + 1, tbl, tbl["Y"], tbl["M"], BINCOL, epochs, dev, rows_ok=ok)
        del ts0, ts1, gx
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        release(lock)
    if smoke:
        assert np.isfinite(oof).sum() > 0
        log("temporal smoke: shape", oof.shape, "rows kept", int(ok.sum()), "of", N)
        return None
    res = {"run": True, "n_rows_common": int(ok.sum()), "n_dropped_g_lt_5": int((~ok).sum()), "pairs": {}}
    base = OOF["P3"]["U640"]
    for t, nm in enumerate(TNAMES):
        m = tbl["M"][:, t] & ok
        y = np.where(m, tbl["Y"][:, t], np.nan)
        pa, pb = np.where(m, oof[:, t], np.nan), np.where(m, base[:, t], np.nan)
        if np.isfinite(pa).sum() < 3:
            continue
        res["pairs"][nm] = boot.pair(KIND[nm], y, pa, pb)
        res["pairs"][nm]["value_temporal"], res["pairs"][nm]["value_U640"] = metric_value(KIND[nm], y, pa), metric_value(KIND[nm], y, pb)
    return res


# ===================================================================================== verdict
def fmt_pair(r):
    return f"{r['diff']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]{' *' if r['separated_pos'] or r['separated_neg'] else ''} (n {r.get('n', '')})"


def verdict_md(res):
    c, o, m = res["controls"], res["outcomes"], res["metrics"]
    L = ["# VERDICT_FULLRES (PREREG_V8F_PROOF_ADD9, H-V8F-PROOF-4 / theorem T5, H-V8F-PROOF-8)", "",
         "Evidence class MEASURED (ours; RESULT_FULLRES.json in this folder). Tier T0: a perception diagnostic, never driving performance. Zero training of any model; only probes are fitted. Paired episode-cluster bootstrap B 2,000, seed 0, on cross-fitted predictions. A separated interval answers 'another draw of episodes'; probe initialisation variance is handled by 3 seeds (P3).",
         "Metric families: LONGITUDINAL = the GAP, STOPDIST and STOPFLAG probes. LATERAL and TACTICAL: not measured by this read (a frozen-token perception probe has no planner, no lateral trajectory and no manoeuvre decision to score). STRATEGIC: PAUSED (PI decision queue item 25), not a pass, not omitted.", "",
         "## Controls (evaluated first)", f"* all controls pass: **{res['controls_pass']}**" + ("" if res["controls_pass"] else "  (a control missed its bar: see below; f3 failing = INVALID-AS-DESIGNED)"),
         f"* f1 harness: mine {c['f1']['mine_GAP_R2_V40+E']:.8f} vs banked {c['f1']['banked']:.8f}, diff {c['f1']['abs_diff']:.2e} (bar 1e-6): **{c['f1']['pass']}**",
         f"* f2 recipe (reported): {json.dumps(c['f2'])}",
         f"* f3 pooled level: GAP R2 L40+E {c['f3']['GAP_R2_L40+E']:.4f} vs banked V40+E {c['f3']['banked_GAP_V40+E']:.4f} (diff {c['f3']['GAP_diff']:+.4f}, bar 0.05); LEADP AUROC {c['f3']['LEADP_AUROC_L40+E']:.4f} vs {c['f3']['banked_LEADP_V40+E']:.4f} (diff {c['f3']['LEADP_diff']:+.4f}, bar 0.03): **{c['f3']['pass']}**"]
    for s, r in c["f4"]["sets"].items():
        L.append(f"* f4 shuffle, P3 {s}: " + "; ".join(f"{k} " + (f"R2 {v['R2']:+.4f}" if 'R2' in v else f"mean per-fold AUROC {v['mean_per_fold_AUROC']:.3f}") + (" ok" if v["pass"] else " MISS") for k, v in r.items()))
    L.append(f"* f4 overall: **{c['f4']['pass']}**")
    f5 = c["f5"]
    L.append(f"* f5 planted object: P3 U640 AUROC {f5['P3_U640_AUROC']:.4f} (bar 0.95): **{f5['pass']}**; beside: P1 U640 {f5['P1_U640_AUROC_beside']:.4f}, P3 L40 {f5['P3_L40_AUROC_beside']:.4f} (theorem T5 predicts the pooled set loses a one-token object)")
    if "INVALID" in o:
        L.append(f"* **{o['INVALID']}**")
    L += ["", "## Outcomes (names fixed by the card; P3 decides, P1 beside)", f"* GAP, pooling: **{o['GAP_pooling']}** (U640+E minus L160+E: P3 {fmt_pair(o['P4a_full']['P3'])}; P1 {fmt_pair(o['P4a_full']['P1'])})",
          f"* GAP, resolution: **{o['GAP_resolution']}** (H1664+E minus U640+E: P3 {fmt_pair(o['P4a_res']['P3'])}; P1 {fmt_pair(o['P4a_res']['P1'])})",
          f"* GAP, the limit: **{o['GAP_limit']['label']}** (best fine set {o['GAP_limit']['best_fine_set']} minus L40+E: P3 {fmt_pair(o['GAP_limit']['P3'])}; P1 {fmt_pair(o['GAP_limit']['P1'])})",
          f"* Stop cue: **{' + '.join(o['stop_cue']['labels'])}** (STOPFLAG sets separated above E: {o['stop_cue']['flag_sets_separated_above_E']}; best STOPDIST set {o['stop_cue']['best_dist_set']}: {fmt_pair(o['stop_cue']['dist_P3'])})",
          f"* P4b: C640+E minus E P3 {fmt_pair(o['stop_cue']['P4b_C640']['P3'])}; L40+E minus E P3 {fmt_pair(o['stop_cue']['P4b_L40']['P3'])}; holds: {o['stop_cue']['P4b_holds']}"]
    p4 = res["P4c"]
    L.append(f"* P4c (STOPFLAG gain of C640+E over E, no outcome name depends on it; n stop windows under 40 m {p4['n_stop_under_40m']}, 40 m or more {p4['n_stop_ge_40m']}): P3 gain {p4['P3']['gain_under_40m']:+.4f} vs {p4['P3']['gain_ge_40m']:+.4f}, difference {p4['P3']['gain_under_minus_ge']:+.4f} CI {p4['P3']['ci95_of_that_difference']}; P1 {p4['P1']['gain_under_40m']:+.4f} vs {p4['P1']['gain_ge_40m']:+.4f}")
    L += ["", "NEAR, COUNT, LEADP are printed beside and decide nothing. '*' = interval excludes 0. Positive = first set better (STOPDIST: lower median absolute error).", ""]
    for nm in TNAMES:
        L += [f"## {nm}" + (" (median |error| m, lower better; pair difference = median|err|(b) - median|err|(a))" if nm == "STOPDIST" else ""), "", "| set | P3 | P1 |" + (" P3 median |err| m | P1 median |err| m |" if nm == "STOPDIST" else ""), "|---|---|---|" + ("---|---|" if nm == "STOPDIST" else "")]
        for s in ["E"] + SETS:
            a, b = m["P3"].get(s, {}).get(nm), m["P1"].get(s, {}).get(nm)
            if a is None and b is None:
                continue
            fa = lambda d: "-" if d is None else (f"{d['r2']:.3f}" if nm == "STOPDIST" else f"{d['value']:.4f}")
            row = f"| {s}{'' if s == 'E' else '+E'} | {fa(a)} | {fa(b)} |"
            if nm == "STOPDIST":
                row += f" {'-' if a is None else format(a['median_abs_err_m'], '.3f')} | {'-' if b is None else format(b['median_abs_err_m'], '.3f')} |"
            L.append(row)
        L += ["", "| pair (a minus b) | P3 | P1 |", "|---|---|---|"]
        for k, r3 in res["pairs"]["P3"].get(nm, {}).items():
            r1 = res["pairs"]["P1"].get(nm, {}).get(k)
            L.append(f"| {k} | {fmt_pair(r3)} | {'-' if r1 is None else fmt_pair(r1)} |")
        L.append("")
    t = res["temporal"]
    if t.get("run"):
        L += ["## TEMPORAL (frame g with frame g-5, P3; zero training; committed next lever of the card)", f"Common rows n {t['n_rows_common']} (dropped g < 5: {t['n_dropped_g_lt_5']}).", ""]
        for nm, r in t["pairs"].items():
            L.append(f"* {nm}: TEMPORAL minus U640 {fmt_pair(r)}; values {r['value_temporal']:.4f} vs {r['value_U640']:.4f}")
    else:
        L.append(f"## TEMPORAL: not run ({t.get('reason')})")
    L += ["", f"Config and n per target: {json.dumps(res['config']['n_per_target'])}; dropped: {json.dumps(res['config']['table']['dropped'])}; sha256 of card, index.npz and pca.npz are in RESULT_FULLRES.json.",
          "Reading choices R1.. are in FULL_DEFS.md. A null under P1 and P3 is not a null about a trained reader on the training clips (card limits).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "smoke"])
    a = ap.parse_args()
    sys.exit(cmd_run(a, smoke=(a.cmd == "smoke")))
