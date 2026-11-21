#!/usr/bin/env python
"""MOTIONPROBE - do cached frozen DINOv3-L token grids, stacked over frames, carry ego motion?

Pre-registration: ../PREREG_MOTIONPROBE.md (sha in ../FROZEN_SHA.txt). CPU only, threads <= 8.
Stages (resumable, each leaves a marker in the scratch dir):
  sample -> ingest -> bases -> project -> ridge -> mlp -> grid2 -> report
Run:  python -I motion_probe.py --run-dir <probe dir> --scratch <scratch dir>
Self test on synthetic tokens:  add  --synth planted|null  (no real token is read).
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
import argparse
import copy
import datetime
import gc
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(8)

NFOLD = 5
NCELL4, NCELL8, DIM = 40, 160, 1024
BAND = (3.0, 8.0)
G_GRID = [10 ** e for e in (-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5)]
FS_REAL = ["F1mean", "F1", "F4", "D3", "D3f", "F1d"]
FS_K2 = ["F4_K2", "D3_K2"]
FS_K3 = ["F4_K3", "D3_K3"]
FS_RIDGE = FS_REAL + FS_K2 + FS_K3
FS_MLP = ["F1mean", "F1", "F1d", "F4", "D3", "F4_K2", "D3_K2", "F4_K3", "D3_K3"]
K1_BOUNDS = (-0.05, 0.03)
RSS_LIMIT_GB = 9.0

LOGF = None


def log(msg):
    line = "[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if LOGF is not None:
        with open(LOGF, "a", newline="\n") as f:
            f.write(line + "\n")


def rss_gb():
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:
        return float("nan")


def guard_ram(tag):
    r = rss_gb()
    log("rss %.2f GB @ %s" % (r, tag))
    if r > RSS_LIMIT_GB:
        raise MemoryError("rss %.2f GB > %.1f GB at %s" % (r, RSS_LIMIT_GB, tag))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def jdump(obj, path):
    with open(path, "w", newline="\n") as f:
        json.dump(clean(obj), f, indent=1)
        f.write("\n")


def jload(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- synthetic data (plumbing check)
def make_synth(cache_dir, variant, n=40, seed=0):
    """Synthetic clips in the real cache format. planted: tokens = scene + position * U, so the speed lives ONLY
    in the frame-to-frame difference (position itself is a random offset plus a running sum). null: no motion."""
    cache_dir = Path(cache_dir)
    (cache_dir / "clips").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    U4 = rng.standard_normal((NCELL4, DIM)).astype(np.float32) * 0.004
    U8 = rng.standard_normal((NCELL8, DIM)).astype(np.float32) * 0.004
    rows = []
    for i in range(n):
        T = 101
        v = np.empty(T, np.float32)
        v[0] = rng.uniform(2, 12)
        for j in range(1, T):
            v[j] = np.clip(v[j - 1] + rng.normal(0, 0.3), 0.5, 14)
        pos = rng.uniform(0, 600) + np.cumsum(v) * 0.2  # large random offset: position alone says little about speed
        S4 = rng.standard_normal((NCELL4, DIM)).astype(np.float32) * 0.2
        S8 = rng.standard_normal((NCELL8, DIM)).astype(np.float32) * 0.2
        x4 = S4[None] + rng.standard_normal((T, NCELL4, DIM)).astype(np.float32) * 0.02
        x8 = S8[None] + rng.standard_normal((T, NCELL8, DIM)).astype(np.float32) * 0.02
        if variant == "planted":
            x4 += pos[:, None, None].astype(np.float32) * U4[None]
            x8 += pos[:, None, None].astype(np.float32) * U8[None]
        name = "syn_%s_%03d" % (variant, i)
        d = {
            "pool4x10": torch.from_numpy(x4).to(torch.float8_e4m3fn),
            "pool8x20": torch.from_numpy(x8).to(torch.float8_e4m3fn),
            "v": torch.from_numpy(v),
            "a": torch.zeros(T),
        }
        torch.save(d, cache_dir / "clips" / (name + ".pt"))
        rows.append({"clip": name, "file": "clips/%s.pt" % name, "n_frames": T})
    with open(cache_dir / "manifest.jsonl", "w", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ----------------------------------------------------------------------------- stage: sample
def stage_sample(a, S):
    rows = [json.loads(l) for l in open(Path(a.cache) / "manifest.jsonl")]
    rng = np.random.default_rng(a.seed)
    n = min(a.n_clips, len(rows))
    idx = np.sort(rng.choice(len(rows), size=n, replace=False))
    sel = [rows[i] for i in idx]
    ev = {json.loads(l)["clip"] for l in open(a.eval_manifest)}
    inter = {r["clip"] for r in sel} & ev
    assert not inter, "eval-141 clips in the sample: %s" % sorted(inter)[:5]
    perm = np.random.default_rng(a.seed).permutation(n)
    fold = np.zeros(n, int)
    rank_in_fold = np.zeros(n, int)
    cnt = [0] * NFOLD
    for r, i in enumerate(perm):
        fold[i] = r % NFOLD
        rank_in_fold[i] = cnt[fold[i]]
        cnt[fold[i]] += 1
    per8 = a.n_clips8 // NFOLD
    for i, r in enumerate(sel):
        r["_fold"] = int(fold[i])
        r["_in8"] = bool(rank_in_fold[i] < per8)
    jdump({"sel": sel, "n_eval_ids": len(ev), "eval_intersection": 0}, S / "sample.json")
    log("sample: %d clips, folds %s, in8 %d, eval-141 disjoint (checked vs %d ids)"
        % (n, np.bincount(fold, minlength=NFOLD).tolist(), sum(r["_in8"] for r in sel), len(ev)))


# ----------------------------------------------------------------------------- stage: ingest
def stage_ingest(a, S):
    sel = jload(S / "sample.json")["sel"]
    order = sorted(range(len(sel)), key=lambda i: (sel[i]["_fold"], i))
    cache = Path(a.cache)
    f4 = open(S / "x4.u8", "wb")
    f8 = open(S / "x8.u8", "wb")
    clips, dropped, vlist, mlist = [], [], [], []
    off = off8 = 0
    cur_fold = None
    acc = None

    def new_acc():
        return {
            "Sf": torch.zeros(NCELL4, DIM, DIM), "sxf": torch.zeros(NCELL4, DIM), "nf": 0,
            "Sd": torch.zeros(NCELL4, DIM, DIM), "sxd": torch.zeros(NCELL4, DIM), "nd": 0,
            "S8": torch.zeros(DIM, DIM, dtype=torch.float64), "sx8": torch.zeros(DIM, dtype=torch.float64), "n8": 0,
        }

    def flush(k, ac):
        torch.save(ac, S / ("cov_%d.pt" % k))
        log("saved cov fold %d (nf %d nd %d n8 %d)" % (k, ac["nf"], ac["nd"], ac["n8"]))

    t0 = time.time()
    for n_done, i in enumerate(order):
        r = sel[i]
        if r["_fold"] != cur_fold:
            if acc is not None:
                flush(cur_fold, acc)
            cur_fold = r["_fold"]
            acc = new_acc()
        try:
            d = torch.load(cache / r["file"], weights_only=True)
            x4 = d["pool4x10"]
            T = x4.shape[0]
            assert tuple(x4.shape) == (T, NCELL4, DIM) and x4.dtype == torch.float8_e4m3fn
            assert T == r["n_frames"], "T %d != manifest %d" % (T, r["n_frames"])
            v = d["v"].to(torch.float32).numpy()
            assert v.shape == (T,)
            x8 = None
            if r["_in8"]:
                x8 = d["pool8x20"]
                assert tuple(x8.shape) == (T, NCELL8, DIM) and x8.dtype == torch.float8_e4m3fn
        except Exception as e:  # dropped clips are logged, never replaced
            log("DROP %s: %r" % (r["clip"], e))
            dropped.append({"clip": r["clip"], "why": repr(e)})
            continue
        f4.write(x4.contiguous().view(torch.uint8).numpy().tobytes())
        xf = x4.to(torch.float32).permute(1, 0, 2).contiguous()  # [40,T,1024]
        acc["Sf"] += torch.bmm(xf.transpose(1, 2), xf)
        acc["sxf"] += xf.sum(1)
        acc["nf"] += T
        dx = xf[:, 1:] - xf[:, :-1]
        acc["Sd"] += torch.bmm(dx.transpose(1, 2), dx)
        acc["sxd"] += dx.sum(1)
        acc["nd"] += T - 1
        mlist.append(xf.mean(0).numpy())  # mean over the 40 cells -> [T,1024]
        vlist.append(v)
        e = {"clip": r["clip"], "fold": r["_fold"], "T": int(T), "off": off, "in8": bool(r["_in8"]), "off8": -1,
             "file": r["file"]}
        if x8 is not None:
            f8.write(x8.contiguous().view(torch.uint8).numpy().tobytes())
            xs = x8[::2, ::2, :].to(torch.float32).reshape(-1, DIM)
            acc["S8"] += (xs.T @ xs).double()
            acc["sx8"] += xs.sum(0).double()
            acc["n8"] += xs.shape[0]
            e["off8"] = off8
            off8 += T
        off += T
        clips.append(e)
        del d, x4, x8, xf, dx
        if (n_done + 1) % 50 == 0:
            log("ingest %d/%d  %.0fs" % (n_done + 1, len(order), time.time() - t0))
            guard_ram("ingest")
    flush(cur_fold, acc)
    f4.close()
    f8.close()
    np.save(S / "v_all.npy", np.concatenate(vlist))
    np.save(S / "mean4.npy", np.concatenate(mlist).astype(np.float32))
    names = [c["clip"] for c in clips]
    jdump({"clips": clips, "dropped": dropped, "sum_T": off, "sum_T8": off8,
           "clip_list_sha256": hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()}, S / "meta.json")
    log("ingest done: %d clips kept, %d dropped, rows %d, rows8 %d, %.0fs" % (len(clips), len(dropped), off, off8,
                                                                         time.time() - t0))


# ----------------------------------------------------------------------------- stage: bases + projection
def top_eig(C, k):
    ev, V = torch.linalg.eigh(C)
    tot = ev.clamp_min(0).sum(-1)
    top = V[..., -k:].flip(-1).contiguous()
    kept = ev[..., -k:].clamp_min(0).sum(-1) / tot
    return top, kept


def stage_bases(a, S):
    t0 = time.time()
    tot = None
    # accumulate totals one file at a time (low RAM)
    keys = ["Sf", "sxf", "nf", "Sd", "sxd", "nd", "S8", "sx8", "n8"]
    for k in range(NFOLD):
        c = torch.load(S / ("cov_%d.pt" % k), weights_only=True)
        if tot is None:
            tot = {kk: (c[kk].clone() if torch.is_tensor(c[kk]) else c[kk]) for kk in keys}
        else:
            for kk in keys:
                tot[kk] = tot[kk] + c[kk]
        del c
    Pf, muf, Pd, P8, mu8 = [], [], [], [], []
    kept_f, kept_d, kept_8 = [], [], []
    for k in range(NFOLD):
        c = torch.load(S / ("cov_%d.pt" % k), weights_only=True)
        n = tot["nf"] - c["nf"]
        mu = (tot["sxf"] - c["sxf"]) / n
        C = (tot["Sf"] - c["Sf"]) / n - mu[:, :, None] * mu[:, None, :]
        P, kp = top_eig(C, a.pca_dim)
        Pf.append(P)
        muf.append(mu)
        kept_f.append(kp.mean().item())
        n = tot["nd"] - c["nd"]
        mu = (tot["sxd"] - c["sxd"]) / n
        C = (tot["Sd"] - c["Sd"]) / n - mu[:, :, None] * mu[:, None, :]
        P, kp = top_eig(C, a.pca_dim)
        Pd.append(P)
        kept_d.append(kp.mean().item())
        n = tot["n8"] - c["n8"]
        mu = (tot["sx8"] - c["sx8"]) / n
        C = (tot["S8"] - c["S8"]) / n - torch.outer(mu, mu)
        P, kp = top_eig(C, a.pca_dim8)
        P8.append(P.float())
        mu8.append(mu.float())
        kept_8.append(kp.item())
        del c, C
        log("bases fold %d: frame-PCA keeps %.3f, diff-PCA keeps %.3f, pooled 8x20 PCA keeps %.3f  %.0fs"
            % (k, kept_f[-1], kept_d[-1], kept_8[-1], time.time() - t0))
    torch.save({"Pf": torch.stack(Pf), "muf": torch.stack(muf), "Pd": torch.stack(Pd), "P8": torch.stack(P8),
                "mu8": torch.stack(mu8), "kept_f": kept_f, "kept_d": kept_d, "kept_8": kept_8}, S / "bases.pt")
    del tot
    gc.collect()


def stage_project(a, S):
    t0 = time.time()
    meta = jload(S / "meta.json")
    B = torch.load(S / "bases.pt", weights_only=True)
    kd = NCELL4 * a.pca_dim
    k8 = NCELL8 * a.pca_dim8
    fz = [open(S / ("zf_%d.f32" % k), "wb") for k in range(NFOLD)]
    fd = [open(S / ("zd_%d.f32" % k), "wb") for k in range(NFOLD)]
    f8 = [open(S / ("z8_%d.f32" % k), "wb") for k in range(NFOLD)]
    for n_done, c in enumerate(meta["clips"]):
        T = c["T"]
        raw = np.fromfile(S / "x4.u8", dtype=np.uint8, count=T * NCELL4 * DIM, offset=c["off"] * NCELL4 * DIM)
        x = torch.from_numpy(raw).view(torch.float8_e4m3fn).reshape(T, NCELL4, DIM).to(torch.float32)
        xt = x.permute(1, 0, 2).contiguous()  # [40,T,1024]
        for k in range(NFOLD):
            zf = torch.bmm(xt - B["muf"][k][:, None, :], B["Pf"][k])  # [40,T,32]
            zd = torch.bmm(xt, B["Pd"][k])
            fz[k].write(zf.permute(1, 0, 2).reshape(T, kd).contiguous().numpy().tobytes())
            fd[k].write(zd.permute(1, 0, 2).reshape(T, kd).contiguous().numpy().tobytes())
        if c["in8"]:
            raw8 = np.fromfile(S / "x8.u8", dtype=np.uint8, count=T * NCELL8 * DIM, offset=c["off8"] * NCELL8 * DIM)
            x8 = torch.from_numpy(raw8).view(torch.float8_e4m3fn).reshape(T, NCELL8, DIM).to(torch.float32)
            for k in range(NFOLD):
                z8 = (x8 - B["mu8"][k]) @ B["P8"][k]  # [T,160,8]
                f8[k].write(z8.reshape(T, k8).contiguous().numpy().tobytes())
            del raw8, x8
        if (n_done + 1) % 100 == 0:
            log("project %d/%d  %.0fs" % (n_done + 1, len(meta["clips"]), time.time() - t0))
    for f in fz + fd + f8:
        f.close()
    log("project done %.0fs" % (time.time() - t0))


# ----------------------------------------------------------------------------- windows and feature sets
class Windows:
    def __init__(self, a, S):
        meta = jload(S / "meta.json")
        self.meta = meta
        self.clips = meta["clips"]
        v_all = np.load(S / "v_all.npy")
        ci, tt = [], []
        for i, c in enumerate(self.clips):
            T = c["T"]
            v = v_all[c["off"]:c["off"] + T]
            t = np.arange(3, T, 2)
            ok = np.isfinite(v[t]) & np.isfinite(v[t - 3])
            ci.append(np.full(ok.sum(), i))
            tt.append(t[ok])
        self.ci = np.concatenate(ci)
        self.t = np.concatenate(tt)
        off = np.array([c["off"] for c in self.clips])
        Tn = np.array([c["T"] for c in self.clips])
        off8 = np.array([c["off8"] for c in self.clips])
        fold = np.array([c["fold"] for c in self.clips])
        self.row = off[self.ci] + self.t
        self.fold = fold[self.ci]
        self.in8 = np.array([c["in8"] for c in self.clips])[self.ci]
        self.row8 = np.where(self.in8, off8[self.ci] + self.t, -1)
        self.v = v_all[self.row].astype(np.float64)
        self.dv = self.v - v_all[self.row - 3].astype(np.float64)
        n = len(self.ci)
        # donor frames for K2: cyclic next clip of the same fold
        donor = np.zeros(len(self.clips), int)
        for f in range(NFOLD):
            ids = [i for i, c in enumerate(self.clips) if c["fold"] == f]
            for j, i in enumerate(ids):
                donor[i] = ids[(j + 1) % len(ids)]
        d = donor[self.ci]
        dt = np.minimum(self.t[:, None] - 3 + np.arange(3)[None], Tn[d][:, None] - 1)
        self.R_real = self.row[:, None] - np.array([3, 2, 1, 0])[None]
        self.R_k2 = np.concatenate([off[d][:, None] + dt, self.row[:, None]], 1)
        perm = np.argsort(np.random.default_rng(a.seed + 7).random((n, 4)), axis=1)
        self.R_k3 = (self.row - 3)[:, None] + perm
        self.band = (self.v >= BAND[0]) & (self.v <= BAND[1])
        self.n = n

    def R_for(self, fs):
        if fs.endswith("_K2"):
            return self.R_k2
        if fs.endswith("_K3"):
            return self.R_k3
        return self.R_real


def build_X(fs, idx, W, Zf, Zd, mean4):
    R = W.R_for(fs)[idx]
    base = fs.split("_")[0]
    if base == "F1mean":
        return mean4[R[:, 3]]
    if base == "F1":
        return Zf[R[:, 3]]
    if base == "F1d":  # sensitivity only: frame t read through the DIFFERENCE basis
        return Zd[R[:, 3]]
    if base == "F4":
        return np.concatenate([Zf[R[:, j]] for j in range(4)], 1)
    Z = Zd if base == "D3" else Zf
    return np.concatenate([Z[R[:, 3]] - Z[R[:, 2]], Z[R[:, 2]] - Z[R[:, 1]], Z[R[:, 1]] - Z[R[:, 0]]], 1)


# ----------------------------------------------------------------------------- readouts
def r2(y, p):
    if len(y) < 10:
        return float("nan")
    sst = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - p) ** 2).sum() / sst) if sst > 0 else float("nan")


def cell_metrics(y, p, band):
    out = {}
    for nm, m in (("all", np.ones(len(y), bool)), ("band", band)):
        yy, pp = y[m], p[m]
        out["r2_" + nm] = r2(yy, pp)
        out["rmse_" + nm] = float(np.sqrt(((yy - pp) ** 2).mean())) if len(yy) else float("nan")
        out["n_" + nm] = int(len(yy))
    return out


def standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr.sub_(mu).div_(sd)
    Xte.sub_(mu).div_(sd)


def ridge_fit_predict(Xtr, Ytr4, Xte, val):
    """Ytr4 columns: v, dv, v_perm, dv_perm (K1: globally permuted labels). Alpha chosen PER COLUMN on the inner
    split (val mask), exactly as for any target."""
    standardize(Xtr, Xte)
    n, d = Xtr.shape
    ym = Ytr4.mean(0, keepdim=True)
    Yc = Ytr4 - ym
    valt = torch.from_numpy(val)
    Xv = Xtr[valt]
    nin = n - int(val.sum())
    XtX = Xtr.T @ Xtr
    XtY = Xtr.T @ Yc
    A = XtX - Xv.T @ Xv
    B = XtY - Xv.T @ Yc[valt]
    ev, V = torch.linalg.eigh(A)
    ev = ev.clamp_min(0)
    XvV = Xv @ V
    VtB = V.T @ B
    Yv = Yc[valt]
    sse = []
    for g in G_GRID:
        coef = VtB / (ev[:, None] + g * nin)
        sse.append(((XvV @ coef - Yv) ** 2).sum(0))
    sse = torch.stack(sse)  # [G,4]
    gi = sse.argmin(0).tolist()
    del A, V, XvV, ev
    W = torch.zeros(d, 4)
    eye = torch.eye(d)
    for gidx in sorted(set(gi)):
        cols = [j for j in range(4) if gi[j] == gidx]
        Lc = torch.linalg.cholesky(XtX + (G_GRID[gidx] * n) * eye)
        W[:, cols] = torch.cholesky_solve(XtY[:, cols], Lc)
    pred = Xte @ W + ym
    return pred.numpy(), [G_GRID[i] for i in gi], [int(i in (0, len(G_GRID) - 1)) for i in gi]


def mlp_fit_predict(Xtr, Ytr2, Xte, val, seed):
    standardize(Xtr, Xte)
    ym, ys = Ytr2.mean(0, keepdim=True), Ytr2.std(0, keepdim=True)
    Y = (Ytr2 - ym) / ys
    tr_idx = torch.from_numpy(np.where(~val)[0])
    va_idx = torch.from_numpy(np.where(val)[0])
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.ReLU(),
                        nn.Dropout(0.1), nn.Linear(256, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad, ep_best = float("inf"), None, 0, 0
    for ep in range(25):
        net.train()
        order = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        for s in range(0, len(order), 256):
            b = order[s:s + 256]
            opt.zero_grad()
            loss = ((net(Xtr[b]) - Y[b]) ** 2).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(((net(Xtr[va_idx]) - Y[va_idx]) ** 2).mean())
        if vl < best - 1e-5:
            best, best_state, bad, ep_best = vl, copy.deepcopy(net.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p = net(Xte) * ys + ym
    return p.numpy(), ep_best, ep + 1


def inner_val_mask(W, train_idx, seed, k):
    cl = np.unique(W.ci[train_idx])
    rng = np.random.default_rng(seed + 100 + k)
    vc = set(rng.choice(cl, size=max(1, int(round(0.2 * len(cl)))), replace=False).tolist())
    return np.array([c in vc for c in W.ci[train_idx]])


def k1_targets(W, a):
    """K1 (amended, PREREG amendment 1): (v, dv) rows permuted ONCE across ALL windows before any split."""
    y = np.stack([W.v, W.dv], 1)
    return y[np.random.default_rng(a.seed + 200).permutation(W.n)]


def load_fold(S, k, nrows, kd):
    Zf = np.fromfile(S / ("zf_%d.f32" % k), dtype=np.float32).reshape(nrows, kd)
    Zd = np.fromfile(S / ("zd_%d.f32" % k), dtype=np.float32).reshape(nrows, kd)
    return Zf, Zd


def run_readout(a, S, readout, fs_list, out_json, k1_mlp_sets=("F4",)):
    meta = jload(S / "meta.json")
    W = Windows(a, S)
    mean4 = np.load(S / "mean4.npy")
    kd = NCELL4 * a.pca_dim
    cells = jload(out_json) if Path(out_json).exists() else {}
    t0 = time.time()
    for k in range(NFOLD):
        todo = [fs for fs in fs_list if str(k) not in cells.get(fs, {})]
        if not todo:
            continue
        Zf, Zd = load_fold(S, k, meta["sum_T"], kd)
        tr = np.where(W.fold != k)[0]
        te = np.where(W.fold == k)[0]
        val = inner_val_mask(W, tr, a.seed, k)
        y = np.stack([W.v, W.dv], 1)
        yk = k1_targets(W, a)
        ytr = torch.from_numpy(y[tr]).float()
        ytr4 = torch.cat([ytr, torch.from_numpy(yk[tr]).float()], 1)
        for fs in todo:
            ts = time.time()
            Xtr = torch.from_numpy(np.ascontiguousarray(build_X(fs, tr, W, Zf, Zd, mean4))).float()
            Xte = torch.from_numpy(np.ascontiguousarray(build_X(fs, te, W, Zf, Zd, mean4))).float()
            rec = {"n_all": int(len(te)), "n_band": int(W.band[te].sum()), "n_train": int(len(tr)), "dim": int(Xtr.shape[1])}
            band = W.band[te]
            if readout == "ridge":
                pred, gsel, edge = ridge_fit_predict(Xtr, ytr4, Xte, val)
                rec["v"] = cell_metrics(W.v[te], pred[:, 0], band)
                rec["dv"] = cell_metrics(W.dv[te], pred[:, 1], band)
                rec["k1_v"] = cell_metrics(yk[te, 0], pred[:, 2], band)
                rec["k1_dv"] = cell_metrics(yk[te, 1], pred[:, 3], band)
                rec["alpha_g"] = {"v": gsel[0], "dv": gsel[1], "k1_v": gsel[2], "k1_dv": gsel[3]}
                rec["alpha_edge"] = {"v": edge[0], "dv": edge[1], "k1_v": edge[2], "k1_dv": edge[3]}
            else:
                pred, ep_best, ep_run = mlp_fit_predict(Xtr, ytr, Xte, val, a.seed + k)
                rec["v"] = cell_metrics(W.v[te], pred[:, 0], band)
                rec["dv"] = cell_metrics(W.dv[te], pred[:, 1], band)
                rec["epochs"] = {"best": ep_best, "run": ep_run}
                if fs in k1_mlp_sets:
                    Xtr2 = torch.from_numpy(np.ascontiguousarray(build_X(fs, tr, W, Zf, Zd, mean4))).float()
                    Xte2 = torch.from_numpy(np.ascontiguousarray(build_X(fs, te, W, Zf, Zd, mean4))).float()
                    p1, _, _ = mlp_fit_predict(Xtr2, ytr4[:, 2:4].contiguous(), Xte2, val, a.seed + k)
                    rec["k1_v"] = cell_metrics(yk[te, 0], p1[:, 0], band)
                    rec["k1_dv"] = cell_metrics(yk[te, 1], p1[:, 1], band)
                    del Xtr2, Xte2
            cells.setdefault(fs, {})[str(k)] = rec
            jdump(cells, out_json)
            log("%s fold %d %-7s dim %5d  v band %.3f all %.3f | dv band %.3f all %.3f | %.0fs"
                % (readout, k, fs, rec["dim"], rec["v"]["r2_band"], rec["v"]["r2_all"], rec["dv"]["r2_band"],
                   rec["dv"]["r2_all"], time.time() - ts))
            del Xtr, Xte
            gc.collect()
        del Zf, Zd
        guard_ram("%s fold %d" % (readout, k))
    log("%s stage done %.0fs" % (readout, time.time() - t0))


def stage_grid2(a, S, out_json):
    meta = jload(S / "meta.json")
    W = Windows(a, S)
    kd = NCELL4 * a.pca_dim
    k8 = NCELL8 * a.pca_dim8
    sub = np.where(W.in8)[0]
    cells = jload(out_json) if Path(out_json).exists() else {}
    for k in range(NFOLD):
        todo = [fs for fs in ("F1_8", "D3_8", "F1_4s", "D3f_4s") if str(k) not in cells.get(fs, {})]
        if not todo:
            continue
        Zf, _ = load_fold(S, k, meta["sum_T"], kd)
        Z8 = np.fromfile(S / ("z8_%d.f32" % k), dtype=np.float32).reshape(meta["sum_T8"], k8)
        tr = sub[W.fold[sub] != k]
        te = sub[W.fold[sub] == k]
        val = inner_val_mask(W, tr, a.seed, k)
        y = np.stack([W.v, W.dv], 1)
        yk = k1_targets(W, a)
        ytr = torch.from_numpy(y[tr]).float()
        ytr4 = torch.cat([ytr, torch.from_numpy(yk[tr]).float()], 1)
        for fs in todo:
            Z, rows = (Z8, W.row8) if fs.endswith("_8") else (Zf, W.row)
            def feat(idx):
                r = rows[idx]
                if fs.startswith("F1"):
                    return Z[r]
                return np.concatenate([Z[r] - Z[r - 1], Z[r - 1] - Z[r - 2], Z[r - 2] - Z[r - 3]], 1)
            Xtr = torch.from_numpy(np.ascontiguousarray(feat(tr))).float()
            Xte = torch.from_numpy(np.ascontiguousarray(feat(te))).float()
            pred, gsel, edge = ridge_fit_predict(Xtr, ytr4, Xte, val)
            band = W.band[te]
            rec = {"n_all": int(len(te)), "n_band": int(band.sum()), "n_train": int(len(tr)), "dim": int(Xtr.shape[1]),
                   "v": cell_metrics(W.v[te], pred[:, 0], band), "dv": cell_metrics(W.dv[te], pred[:, 1], band),
                   "k1_v": cell_metrics(yk[te, 0], pred[:, 2], band), "k1_dv": cell_metrics(yk[te, 1], pred[:, 3], band),
                   "alpha_g": {"v": gsel[0], "dv": gsel[1], "k1_v": gsel[2], "k1_dv": gsel[3]}}
            cells.setdefault(fs, {})[str(k)] = rec
            jdump(cells, out_json)
            log("grid2 fold %d %-7s v band %.3f all %.3f | dv band %.3f all %.3f"
                % (k, fs, rec["v"]["r2_band"], rec["v"]["r2_all"], rec["dv"]["r2_band"], rec["dv"]["r2_all"]))
            del Xtr, Xte
            gc.collect()
        del Zf, Z8


# ----------------------------------------------------------------------------- aggregation, verdict, report
def vals(C, ro, fs, tgt, key):
    if ro not in C or fs not in C[ro] or len(C[ro][fs]) < NFOLD:
        return None
    return [C[ro][fs][str(k)][tgt][key] for k in range(NFOLD)]


def ms(v):
    if v is None:
        return (float("nan"), float("nan"))
    a = np.array([np.nan if x is None else x for x in v], float)
    return (float(np.nanmean(a)), float(np.nanstd(a, ddof=1)))


def best_ro(C, fs, tgt, subset="band"):
    best = None
    for ro in C:
        v = vals(C, ro, fs, tgt, "r2_" + subset)
        if v is None:
            continue
        m = ms(v)[0]
        if best is None or m > best[1]:
            best = (ro, m, v)
    return best


def k1_check(C):
    bad = []
    for ro in C:
        for fs in C[ro]:
            for tgt in ("k1_v", "k1_dv"):
                if len(C[ro][fs]) < NFOLD or tgt not in C[ro][fs]["0"]:
                    continue
                m = ms(vals(C, ro, fs, tgt, "r2_all"))[0]
                if not (K1_BOUNDS[0] <= m <= K1_BOUNDS[1]):
                    bad.append({"readout": ro, "fs": fs, "target": tgt, "mean_r2_all": m})
    return bad


def verdict_for(C, tgt, bad_k1):
    f1 = best_ro(C, "F1", tgt)
    cand = {s: best_ro(C, s, tgt) for s in ("F4", "D3")}
    cand = {s: c for s, c in cand.items() if c is not None}
    if f1 is None or not cand:
        return {"target": tgt, "verdict": "INCOMPLETE"}
    s_star = max(cand, key=lambda s: cand[s][1])
    g = cand[s_star][1] - f1[1]
    k2 = best_ro(C, s_star + "_K2", tgt)
    k3 = best_ro(C, s_star + "_K3", tgt)
    gk2 = (k2[1] - f1[1]) if k2 else float("nan")
    gk3 = (k3[1] - f1[1]) if k3 else float("nan")
    if bad_k1:
        v = "INVALID"
    elif g < 0.05:
        v = "ABSENT"
    elif g >= 0.15 and gk3 <= 0.5 * g and gk2 <= max(0.02, 0.10 * g):
        v = "PRESENT"
    else:
        v = "WEAK"
    # fold-paired gain of s* over F1 (best readouts), same test windows
    pg = [a - b for a, b in zip(cand[s_star][2], f1[2])]
    # supplementary, post-hoc, informational only: the same gains on all-windows R2 (far less noisy than the band)
    f1a, sa = best_ro(C, "F1", tgt, "all"), best_ro(C, s_star, tgt, "all")
    k2a, k3a = best_ro(C, s_star + "_K2", tgt, "all"), best_ro(C, s_star + "_K3", tgt, "all")
    supp = {"gain_all": (sa[1] - f1a[1]) if (sa and f1a) else None,
            "gain_K2_all": (k2a[1] - f1a[1]) if (k2a and f1a) else None,
            "gain_K3_all": (k3a[1] - f1a[1]) if (k3a and f1a) else None,
            "R_F1_all": f1a[1] if f1a else None, "R_sstar_all": sa[1] if sa else None}
    d3f = best_ro(C, "D3f", tgt)
    f1d = best_ro(C, "F1d", tgt)
    d3b = cand.get("D3")
    f4b = cand.get("F4")
    return {"target": tgt, "verdict": v, "s_star": s_star, "readout_s_star": cand[s_star][0],
            "readout_F1": f1[0], "R1_band": f1[1], "R_sstar_band": cand[s_star][1], "gain": g,
            "gain_K2": gk2, "gain_K3": gk3, "K2_readout": k2[0] if k2 else None, "K3_readout": k3[0] if k3 else None,
            "K2_band_r2": k2[1] if k2 else None, "K3_band_r2": k3[1] if k3 else None,
            "paired_gain_by_fold": pg, "paired_gain_mean_sd": ms(pg), "supplementary_all_windows": supp,
            "D3f_gain_vs_F1_sensitivity": (d3f[1] - f1[1]) if d3f else None,
            "F1d_vs_F1_sensitivity": (f1d[1] - f1[1]) if f1d else None,
            "D3_minus_F1d_sensitivity": (d3b[1] - f1d[1]) if (f1d and d3b) else None,
            "F4_minus_F1d_sensitivity": (f4b[1] - f1d[1]) if (f1d and f4b) else None,
            "k1_violations": bad_k1,
            "rule": {"PRESENT": "g>=0.15 and gK3<=0.5g and gK2<=max(0.02,0.10g)", "ABSENT": "g<0.05",
                     "WEAK": "otherwise"}}


IMPLIC = {
    "PRESENT": "The cached token grids carry motion that a single frame does not; a cheap motion adapter that reads "
               "stacked or differenced cached tokens is viable as a perception change.",
    "WEAK": "Something is readable but it is not shown to be motion-borne, or it is below the committed bar. The cheapest "
            "discriminator is a stronger readout (unreduced 1024-d differences, temporal conv) to see whether the PCA "
            "bottleneck hides signal, or a pixel-difference CNN control on the same windows.",
    "ABSENT": "Under per-cell PCA(32) plus ridge/MLP, stacking or differencing cached tokens adds less than 0.05 band R2 "
              "over frame t alone; motion has to come from pixels (a small CNN on frame differences, fused with the "
              "frozen tokens) or from encoder adaptation.",
    "INVALID": "A K1 control left its known range: the pipeline has a defect and no verdict is issued.",
    "INCOMPLETE": "Not all cells were available.",
}


def fmt(m, s, nd=3):
    return "n/a" if (m is None or (isinstance(m, float) and math.isnan(m))) else ("%+.*f +/- %.*f" % (nd, m, nd, s))


def table(C, tgt, key, fs_list, ros=("ridge", "mlp"), nd=3):
    lines = ["| feature set | dim | n windows (all folds) | " + " | ".join(ros) + " |", "|---|---|---|" + "---|" * len(ros)]
    for fs in fs_list:
        row = None
        for ro in ros:
            if ro in C and fs in C[ro] and len(C[ro][fs]) == NFOLD:
                row = C[ro][fs]
                break
        if row is None:
            continue
        nk = "n_band" if key.endswith("band") else "n_all"
        n = sum(row[str(k)][nk] for k in range(NFOLD))
        cells = []
        for ro in ros:
            v = vals(C, ro, fs, tgt, key)
            cells.append("n/a" if v is None else fmt(*ms(v), nd))
        lines.append("| %s | %d | %d | %s |" % (fs, row["0"]["dim"], n, " | ".join(cells)))
    return "\n".join(lines)


def build_report(a, run, S, C, G, verd, meta, extra):
    L = []
    L.append("# RESULT - MOTIONPROBE: do cached frozen DINOv3-L token grids, stacked over frames, carry ego motion?\n")
    L.append("Evidence class: MEASURED (ours). Pre-registration: `PREREG_MOTIONPROBE.md` (sha256 in `FROZEN_SHA.txt`); "
             "numbers: `raw/MOTIONPROBE.json` (cells), `raw/cells_ridge.json`, `raw/cells_mlp.json`, `raw/cells_grid2.json`, "
             "`raw/VERDICT.json`. Tier: not a T1 driving number (T0-class diagnostic of the cached "
             "perception features; no driving model involved).\n")
    L.append("Estimator: fold mean +/- SD (ddof 1) over 5 folds BY CLIP. It answers \"would another draw of CLIPS say "
             "this?\" only. It does not answer \"would another training run say this?\" (ridge is deterministic; the MLP "
             "has one seed per fold) nor \"would another inference run say this?\" (nothing stochastic at inference). "
             "Controls K1/K2/K3 are the guard against reading a spurious gain as motion.\n")
    L.append("## Verdict (committed rule, PREREG section 9)\n")
    L.append("| target | verdict | s* | readout (s*) | readout (F1) | R2 band F1 | R2 band s* | gain g | gain K2 | gain K3 | "
             "paired gain by fold |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in ("v", "dv"):
        d = verd[t]
        if d["verdict"] in ("INCOMPLETE",):
            L.append("| %s | INCOMPLETE | | | | | | | | | |" % t)
            continue
        L.append("| %s | **%s** | %s | %s | %s | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f | %s |"
                 % (t, d["verdict"], d["s_star"], d["readout_s_star"], d["readout_F1"], d["R1_band"], d["R_sstar_band"],
                    d["gain"], d["gain_K2"], d["gain_K3"], ", ".join("%+.3f" % x for x in d["paired_gain_by_fold"])))
    L.append("")
    for t in ("v", "dv"):
        d = verd[t]
        L.append("* `%s`: %s %s" % (t, d["verdict"], IMPLIC.get(d["verdict"], "")))
    L.append("")
    L.append("Noise context (post-hoc, informational, cannot change the verdict). The band is selected on v, so band R2 is "
             "noisy between folds and the best-of-readouts selection leans upward:\n")
    L.append("| target | paired band gain mean +/- SD over folds | all-windows R2 F1 | all-windows R2 s* | gain (all windows) | "
             "gain K2 (all) | gain K3 (all) |\n|---|---|---|---|---|---|---|")
    for t in ("v", "dv"):
        d = verd[t]
        if d["verdict"] == "INCOMPLETE":
            continue
        sp = d["supplementary_all_windows"]
        nz = lambda x: float("nan") if x is None else x
        L.append("| %s | %+.3f +/- %.3f | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f |"
                 % (t, d["paired_gain_mean_sd"][0], d["paired_gain_mean_sd"][1], nz(sp["R_F1_all"]),
                    nz(sp["R_sstar_all"]), nz(sp["gain_all"]), nz(sp["gain_K2_all"]), nz(sp["gain_K3_all"])))
    L.append("")
    if verd["v"].get("D3f_gain_vs_F1_sensitivity") is not None:
        L.append("Sensitivity (cannot change the verdict): D3f (differences projected with the frame basis) best-readout "
                 "band R2 minus F1 = %+.3f on v, %+.3f on dv; K2/K3 were not run for D3f."
                 % (verd["v"]["D3f_gain_vs_F1_sensitivity"], verd["dv"].get("D3f_gain_vs_F1_sensitivity") or float("nan")))
    if verd["v"].get("F1d_vs_F1_sensitivity") is not None:
        gaps = []
        for t in ("v", "dv"):
            q = verd[t]
            gaps.append("%s: F1d-F1 %+.3f, D3-F1d %+.3f, F4-F1d %+.3f" % (
                t, q.get("F1d_vs_F1_sensitivity") or float("nan"), q.get("D3_minus_F1d_sensitivity") or float("nan"),
                q.get("F4_minus_F1d_sensitivity") or float("nan")))
        L.append("\nSensitivity F1d (frame t read through the difference basis; cannot change the verdict), best-readout band "
                 "R2 gaps: " + "; ".join(gaps) + ". If D3's gain over F1 is matched by F1d over F1, the gain is basis content, "
                 "not motion.")
    if verd["v"].get("k1_violations"):
        L.append("\nK1 VIOLATIONS: %s" % json.dumps(clean(verd["v"]["k1_violations"])))
    L.append("\n## Sample\n")
    L.append("%d clips kept (%d dropped), clip-list sha256 `%s`, 5 folds by clip, windows total %d (band 3-8 m/s: %d). "
             "eval-141 disjointness asserted against `%s` (0 shared ids). pool8x20 subset: %d clips."
             % (len(meta["clips"]), len(meta["dropped"]), meta["clip_list_sha256"], extra["n_windows"],
                extra["n_band"], a.eval_manifest, sum(c["in8"] for c in meta["clips"])))
    if "kept" in extra:
        L.append("\nPCA retained variance (mean over cells and folds): frame basis %.3f, difference basis %.3f, pooled "
                 "8x20 basis %.3f." % extra["kept"])
    for tgt, nm in (("v", "v(t), current speed"), ("dv", "dv = v(t)-v(t-3)")):
        for key, kn in (("r2_band", "band 3-8 m/s"), ("r2_all", "all windows")):
            L.append("\n## R2, target %s, %s (mean +/- SD over folds)\n" % (nm, kn))
            L.append(table(C, tgt, key, FS_RIDGE))
    for tgt in ("v", "dv"):
        L.append("\n## RMSE (m/s), band 3-8 m/s, target %s\n" % tgt)
        L.append(table(C, tgt, "rmse_band", FS_RIDGE, nd=2))
    L.append("\n## Controls\n")
    L.append("K1 (amendment 1: (v, dv) permuted once across ALL windows before the split, standard pipeline incl. its own "
             "alpha choice, scored on the permuted test targets; known value: all-windows R2 in [%.2f, %.2f]):\n"
             % K1_BOUNDS)
    L.append("| readout | feature set | target | mean all-windows R2 |\n|---|---|---|---|")
    for ro in C:
        for fs in C[ro]:
            if len(C[ro][fs]) < NFOLD:
                continue
            for tgt in ("k1_v", "k1_dv"):
                if tgt in C[ro][fs]["0"]:
                    L.append("| %s | %s | %s | %+.3f |" % (ro, fs, tgt, ms(vals(C, ro, fs, tgt, "r2_all"))[0]))
    L.append("\nK2 (frames t-3..t-1 from another clip of the same fold) and K3 (frame order shuffled per window) rows are "
             "the `_K2` and `_K3` rows of the R2 tables above; their gains over F1 are in the verdict table.")
    if G:
        L.append("\n## Second grid pool8x20 (supplementary, ridge, 300-clip subset; cannot produce PRESENT)\n")
        L.append("F1_8 / D3_8 use a 160-cell pooled PCA basis, 8 dims per cell; F1_4s / D3f_4s are the matched 4x10 "
                 "references on the same clips.\n")
        G2 = {"ridge": G}
        for tgt in ("v", "dv"):
            L.append("\nTarget %s, band R2:\n" % tgt)
            L.append(table(G2, tgt, "r2_band", ["F1_8", "D3_8", "F1_4s", "D3f_4s"], ros=("ridge",)))
            L.append("\nTarget %s, all-windows R2:\n" % tgt)
            L.append(table(G2, tgt, "r2_all", ["F1_8", "D3_8", "F1_4s", "D3f_4s"], ros=("ridge",)))
        for tgt in ("v", "dv"):
            f1 = ms(vals(G2, "ridge", "F1_8", tgt, "r2_band"))[0]
            d3 = ms(vals(G2, "ridge", "D3_8", tgt, "r2_band"))[0]
            f1s = ms(vals(G2, "ridge", "F1_4s", tgt, "r2_band"))[0]
            d3s = ms(vals(G2, "ridge", "D3f_4s", tgt, "r2_band"))[0]
            L.append("\n%s: 8x20 gain D3_8 - F1_8 = %+.3f; 4x10 matched gain D3f_4s - F1_4s = %+.3f (band R2)."
                     % (tgt, d3 - f1, d3s - f1s))
    L.append("\n## Changes after the freeze (all made before any real-data fit)\n")
    L.append("* PREREG amendment 2 (K2 'vanishes' operationalised as gK2 <= max(0.02, 0.10 g) instead of gK2 <= 0.02): in the "
             "synthetic plumbing check a planted gain of +2.27 band R2 was labelled WEAK because a no-motion control sat "
             "+0.074 above F1, which is the scale of the band-R2 noise between two no-signal models (their SD across folds was "
             "0.3-0.6). Initial texts: `PREREG_MOTIONPROBE_v0.md`, `PREREG_MOTIONPROBE_v1.md`.\n"
             "* PREREG amendment 1 (K1 redefined, original text kept in `PREREG_MOTIONPROBE_v0.md`; see `FROZEN_SHA.txt`): the "
             "synthetic plumbing check showed that a K1 that reuses the real target's alpha reads strongly negative whenever the "
             "real signal is strong (low alpha overfits permuted noise), which is an overfit, not a pipeline defect. The "
             "amended K1 permutes the labels globally before the split and lets every column choose its own alpha.\n"
             "* `F1d` (frame t projected with the difference basis, 1,280 dims) was added as a SENSITIVITY row for ridge and "
             "MLP; it is not in the decision rule.\n"
             "* A plumbing check on synthetic tokens (planted displacement and null) lives in `selftest/`; it reads no "
             "real token.")
    L.append("\n## Reading guide and limits\n")
    L.append("* The band is selected on v itself, so band R2 on v is harsh (small SST); read it with the RMSE table.\n"
             "* The probe is a statement about THIS reduction (per-cell PCA to 32 dims) and THESE readouts (ridge, "
             "2 x 256 MLP). It is a floor on what a trained adapter over the full 1024-d tokens could extract, not "
             "a ceiling.\n"
             "* Alpha grid edges hit are recorded per cell in `raw/cells_ridge.json` (`alpha_edge`).\n"
             "* Windows overlap within a clip (every 2nd frame); folds are by clip, so the SD reflects clip draws only.\n"
             "* Code sha256 and PREREG sha256 are recorded in `raw/MOTIONPROBE.json`.")
    return "\n".join(L) + "\n"


def stage_report(a, run, S):
    raw = run / "raw"
    C = {}
    for ro in ("ridge", "mlp"):
        p = raw / ("cells_%s.json" % ro)
        if p.exists():
            C[ro] = jload(p)
    G = jload(raw / "cells_grid2.json") if (raw / "cells_grid2.json").exists() else None
    meta = jload(S / "meta.json")
    bad = k1_check(C)
    verd = {t: verdict_for(C, t, bad) for t in ("v", "dv")}
    W = Windows(a, S)
    B = torch.load(S / "bases.pt", weights_only=True)
    extra = {"n_windows": int(W.n), "n_band": int(W.band.sum()),
             "kept": (float(np.mean(B["kept_f"])), float(np.mean(B["kept_d"])), float(np.mean(B["kept_8"])))}
    prereg = run / "PREREG_MOTIONPROBE.md"
    jdump({"cells": C, "grid2": G, "verdict": verd, "k1_violations": bad,
           "clip_list_sha256": meta["clip_list_sha256"], "clips": [c["clip"] for c in meta["clips"]],
           "dropped": meta["dropped"], "n_windows": extra["n_windows"], "n_band": extra["n_band"],
           "pca_kept": {"frame": B["kept_f"], "diff": B["kept_d"], "pooled8x20": B["kept_8"]},
           "code_sha256": sha256_file(Path(__file__)),
           "prereg_sha256": sha256_file(prereg) if prereg.exists() else None,
           "args": vars(a), "torch": torch.__version__, "numpy": np.__version__},
          raw / "MOTIONPROBE.json")
    jdump(verd, raw / "VERDICT.json")
    (run / "RESULT.md").write_text(build_report(a, run, S, C, G, verd, meta, extra), encoding="utf-8", newline="\n")
    log("report written; verdicts v=%s dv=%s" % (verd["v"]["verdict"], verd["dv"]["verdict"]))


def write_manifest(run):
    lines = ["# path | bytes | sha256 (paths relative to %s)" % run.as_posix()]
    for p in sorted(run.rglob("*")):
        if p.is_file() and p.name not in ("MANIFEST.txt", "PROBE.done", "PROBE.fail"):
            lines.append("%s | %d | %s" % (p.relative_to(run).as_posix(), p.stat().st_size, sha256_file(p)))
    (run / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------- driver
def done(S, name):
    return (S / ("stage_%s.done" % name)).exists()


def mark(S, name):
    (S / ("stage_%s.done" % name)).write_text("ok\n")


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--cache", default="F:/TanitAD-scratch/speedread/cache/train")
    ap.add_argument("--eval-manifest", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/manifest.jsonl")
    ap.add_argument("--n-clips", type=int, default=600)
    ap.add_argument("--n-clips8", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--pca-dim8", type=int, default=8)
    ap.add_argument("--synth", default=None, choices=[None, "planted", "null"])
    ap.add_argument("--synth-n", type=int, default=40)
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--skip-grid2", action="store_true")
    a = ap.parse_args()
    run, S = Path(a.run_dir), Path(a.scratch)
    (run / "raw").mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(parents=True, exist_ok=True)
    S.mkdir(parents=True, exist_ok=True)
    LOGF = run / "logs" / "run.log"
    try:
        log("start; args %s" % json.dumps(vars(a)))
        if a.synth:
            a.cache = str(S / "synth_cache")
            if not (Path(a.cache) / "manifest.jsonl").exists():
                make_synth(a.cache, a.synth, n=a.synth_n)
            log("synthetic cache '%s' at %s" % (a.synth, a.cache))
        stages = [("sample", stage_sample), ("ingest", stage_ingest), ("bases", stage_bases), ("project", stage_project)]
        for nm, fn in stages:
            if not done(S, nm):
                t0 = time.time()
                fn(a, S)
                mark(S, nm)
                log("stage %s ok %.0fs" % (nm, time.time() - t0))
                guard_ram(nm)
        raw = run / "raw"
        run_readout(a, S, "ridge", FS_RIDGE, raw / "cells_ridge.json")
        if not a.skip_mlp:
            run_readout(a, S, "mlp", FS_MLP, raw / "cells_mlp.json")
        if not a.skip_grid2:
            stage_grid2(a, S, raw / "cells_grid2.json")
        stage_report(a, run, S)
        log("manifest")
        write_manifest(run)
        (raw / "PROBE.done").write_text("done %s\n" % datetime.datetime.now().isoformat(timespec="seconds"))
    except BaseException:
        tb = traceback.format_exc()
        log("FAIL\n" + tb)
        (run / "raw" / "PROBE.fail").write_text(tb)
        raise


if __name__ == "__main__":
    main()
