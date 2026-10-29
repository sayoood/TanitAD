"""p4_lib.py -- shared layer of trigger probe 4 (PREREG_FROZEN.md): does a SPATIAL readout (attention / max pooling over tokens) recover the go/stay trigger from the frozen DINOv3 per-token PCA-16 features of
probe 3, where the flat readouts did not?  Reads probes 1 and 3 READ-ONLY; writes only under trigger_probe4/ (small) and F:/TanitAD-scratch/trigger_probe4_scratch/ (big).
EVIDENCE CLASS of everything computed with it: MEASURED (artifacts under raw/).  ESTIMATOR: clip-cluster bootstrap B = 2000 seed 0 on the MEAN OF PER-FOLD AUCs (tp_lib, byte-identical to probes 1 and 3).
It answers "would another draw of EPISODES say this?"; the readout-seed spread (3 seeds per arm) is the training-variance companion (H-ESTIM-SEED-1).  CUDA_VISIBLE_DEVICES=-1."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
_THR = os.environ.get("P4_THREADS", "6")
assert 1 <= int(_THR) <= 6, "PREREG 7: torch / BLAS threads <= 6"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = _THR
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW = f"{HERE}/raw"
SCR = "F:/TanitAD-scratch/trigger_probe4_scratch"
P3 = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe3"
P3RAW = f"{P3}/raw"
P3SCR = "F:/TanitAD-scratch/trigger_probe3_scratch"
OLD = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe"
OLDRAW = f"{OLD}/raw"
WINDOWS_SHA = "a9a57b32ffac558929b8f57e49a979826b88d17be17e5e9d7f9347e5875783ec"
TP_LIB_SHA = "44b8797740af5cb01dfbdb37d35f4caba80783393c932536127824ca98ee8932"
sys.path.insert(0, HERE)
import tp_lib as L  # noqa: E402  (byte-identical copy of trigger_probe3/tp_lib.py; sha asserted in the tests and in run_p4)

POPS = ("pstop", "pmove")
POP_INDEX = {"pstop": 0, "pmove": 1}
GRID_HW = {"G8": (8, 20), "G4": (4, 10)}
N_TOK = {"G8": 160, "G4": 40}
K_TOK = 16
N_FOLD = 5
P3_MARKERS = ("CHAIN_c2band.done", "CHAIN_mlpctl.done", "CHAIN_evalmlp.done", "CHAIN_mutate.done", "CHAIN_pf.done")

# ------------------------------------------------------------------------------------------------ the FROZEN configuration (PREREG_FROZEN.md 3; tests may override a COPY, the real run asserts these values)
CFG = dict(lr=1e-3, wd=1e-2, batch=256, max_epochs=40, patience=5, val_group_mod=5, n_query=4, d_key=32, d_tok=16, d_hidden_max=32, seeds=(0, 1, 2),
           n_boot=2000, boot_seed=0, pos_std=1.0, q_std=1.0, eval_bs=1024)
ARMS = ("G8att", "G4att", "G8max", "G4max")                 # priority order: the two decision arms first
ARM_GRID = {"G8att": "G8", "G4att": "G4", "G8max": "G8", "G4max": "G4"}
ARM_KIND = {"G8att": "att", "G4att": "att", "G8max": "max", "G4max": "max"}


# ------------------------------------------------------------------------------------------------ resources
def free_mb() -> int:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def require_ram(min_mb=4000, what="heavy load", max_wait_s=14400, poll_s=30):
    """RAM free >= 4 GB before every heavy load (brief).  Polls up to ``max_wait_s``, then exits: a step never starts below the floor."""
    if min_mb <= 0:
        return 0
    t0 = time.time()
    while True:
        fm = free_mb()
        if fm >= min_mb:
            return fm
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"RAM GUARD: free {fm} MB < {min_mb} MB before {what}")
        time.sleep(poll_s)


def sha256_file(p, blk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(blk), b""):
            h.update(b)
    return h.hexdigest()


def p3_markers_present():
    return {m: os.path.exists(f"{P3RAW}/{m}") for m in P3_MARKERS}


# ------------------------------------------------------------------------------------------------ data (identical to probe 3's g_lib.load_pop / banked; copied, not imported, so probe 3's module state stays untouched)
def load_pop(pop):
    A = np.load(f"{OLDRAW}/windows_train.npz", allow_pickle=False)
    idx = np.load(f"{OLDRAW}/sample_{pop}.npz")["idx"]
    clip = A["clip_i"][idx].astype(np.int64)
    sha = A["episode_sha12"][clip]
    d = {"idx": idx, "y": A["go"][idx].astype(int), "clip": clip, "sha": sha, "outer": L.outer_fold(sha), "inner": L.inner_fold(sha), "E": A["E"][idx].astype(np.float64), "v0": A["v0"][idx], "t": A["t"][idx]}
    for k in ("TL_ANY", "TL_RED", "TL_GREEN", "TL_YELLOW", "NO_TL", "lead_now", "no_lead", "lead_depart"):
        d[k] = A[k][idx]
    return d


def banked_comparators(pop):
    """E and V+E (probe 1, linear and MLP) and probe 3's linear G4+E / G8+E, all as banked out-of-fold scores.  Row order is asserted by the caller against load_pop (y, clip)."""
    ov = dict(np.load(f"{OLDRAW}/oof_{pop}_V_linear.npz"))
    om = dict(np.load(f"{OLDRAW}/oof_{pop}_V_mlp.npz"))
    out = {"E": ov["E"], "V+E": ov["V+E"], "V+E(mlp)": om["V+E"], "E(mlp)": om["E"], "y": ov["y"], "clip": ov["clip"]}
    p3 = f"{P3RAW}/oofg_{pop}_linear.npz"
    if os.path.exists(p3):
        z = np.load(p3)
        for k in ("G4+E", "G8+E"):
            if k in z.files:
                out[f"P3:{k}"] = z[k]
    return out


def layer_shape_ok(grid, shape, n=12000):
    """The grid entry point: a S file for ``grid`` MUST be [6, n, N_TOK[grid] * 16].  Asking G8 with the S4 file (640 wide) raises: reachability (PREREG 6)."""
    want = (N_FOLD + 1, n, N_TOK[grid] * K_TOK)
    if tuple(shape) != want:
        raise AssertionError(f"grid {grid}: expected S file shape {want}, got {tuple(shape)}")
    return True


def open_tokens(grid, path, n=12000):
    S = np.load(path, mmap_mode="r")
    layer_shape_ok(grid, S.shape, n)
    assert S.dtype == np.float32, S.dtype
    return S


_WIN_OK = []


def real_data(pop):
    """The complete input bundle of one population (asserted)."""
    if not _WIN_OK:
        assert sha256_file(f"{OLDRAW}/windows_train.npz") == WINDOWS_SHA, "windows_train.npz differs from probe 1's recorded sha256"
        _WIN_OK.append(True)
    d = load_pop(pop)
    bk = banked_comparators(pop)
    assert np.array_equal(bk["y"], d["y"]) and np.array_equal(bk["clip"], d["clip"]), "row order differs from probe 1's banked OOF"
    S = {"G8": open_tokens("G8", f"{P3SCR}/S8_{pop}.npy", len(d["y"])), "G4": open_tokens("G4", f"{P3SCR}/S4_{pop}.npy", len(d["y"]))}
    fp = json.load(open(f"{P3RAW}/pcaG_{pop}.json"))["fingerprints"]
    for grid, key in (("G8", "S8_fold5_sha256_first1000"), ("G4", "S4_fold5_sha256_first1000")):
        h = hashlib.sha256(np.ascontiguousarray(S[grid][5, :1000]).tobytes()).hexdigest()
        assert h == fp[key], (grid, "S file differs from the PCA fingerprint recorded by probe 3", h, fp[key])
    return {"pop": pop, "d": d, "S": S, "bk": bk}


# ------------------------------------------------------------------------------------------------ models
def _torch():
    import torch
    torch.set_num_threads(int(_THR))
    return torch


def make_model(arch, hw, cfg=CFG, uniform=False):
    """arch 'att' = R-ATT, 'max' = R-MAX.  ``uniform`` is the MUTATION of the attention path (softmax replaced by 1/T): used only by the mutation test."""
    torch = _torch()
    nn = torch.nn
    d, dk, nq, n_e = cfg["d_tok"], cfg["d_key"], cfg["n_query"], 8

    class RAtt(nn.Module):
        def __init__(self):
            super().__init__()
            R, C = hw
            self.R, self.C = R, C
            self.row = nn.Parameter(torch.randn(R, d) * cfg["pos_std"])             # learned SEPARABLE 2-D position embedding: row + col, 16-d
            self.col = nn.Parameter(torch.randn(C, d) * cfg["pos_std"])
            self.ln = nn.LayerNorm(d)
            self.k = nn.Linear(d, dk)
            self.v = nn.Linear(d, dk)
            self.q = nn.Parameter(torch.randn(nq, dk) * cfg["q_std"])               # 4 learned queries
            self.head = nn.Linear(nq * dk + n_e, 1)                                  # 4 x 32 = 128 pooled + 8 E -> 1
            self.uniform = uniform

        def forward(self, x, e, return_attn=False):
            B, T, _ = x.shape
            pos = (self.row[:, None, :] + self.col[None, :, :]).reshape(T, d)
            h = self.ln(x + pos)
            K, V = self.k(h), self.v(h)
            s = torch.einsum("qd,btd->bqt", self.q, K) / math.sqrt(dk)
            a = torch.full_like(s, 1.0 / T) if self.uniform else torch.softmax(s, dim=-1)
            pooled = torch.einsum("bqt,btd->bqd", a, V).reshape(B, nq * dk)
            out = self.head(torch.cat([pooled, e], 1)).squeeze(1)
            return (out, a) if return_attn else out

    class RMax(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Linear(d, cfg["d_hidden_max"])
            self.head = nn.Linear(cfg["d_hidden_max"] + n_e, 1)

        def forward(self, x, e, return_attn=False):
            h = torch.relu(self.tok(x)).max(dim=1).values
            out = self.head(torch.cat([h, e], 1)).squeeze(1)
            return (out, None) if return_attn else out

    return RAtt() if arch == "att" else RMax()


# ------------------------------------------------------------------------------------------------ training
def inner_val_mask(sha_int, seed, fold, mod=5):
    """The inner 20 % early-stopping split, GROUPED BY CLIP (a function of the clip's episode sha only): val iff (sha // 15 + seed + fold) % 5 == 0.  (sha % 5 is the outer fold, (sha // 5) % 3 the probe-1 inner
    fold; (sha // 15) % 5 is the next independent mixed-radix digit, so the split is independent of both.)"""
    return ((np.asarray(sha_int, object) // 15 + seed + fold) % mod == 0).astype(bool)


def sha_ints(sha12s):
    return np.array([int(s, 16) for s in sha12s], dtype=object)


def tok_standardiser(Xtr):
    """Per-dim mean / std over all tokens of the given rows, SHARED across token positions (the PCA is shared too).  [16], [16] float32."""
    n, T, d = Xtr.shape
    m = np.zeros(d, np.float64)
    s2 = np.zeros(d, np.float64)
    for a in range(0, n, 2000):
        b = np.asarray(Xtr[a:a + 2000], np.float64)
        m += b.sum((0, 1))
        s2 += (b ** 2).sum((0, 1))
    m /= n * T
    sd = np.sqrt(np.maximum(s2 / (n * T) - m ** 2, 1e-12))
    return m.astype(np.float32), sd.astype(np.float32)


def train_one(arch, hw, Xfit, Efit, yfit, Xval, Eval, yval, seed, cfg=CFG, uniform=False, log=None):
    """One readout fit: AdamW (lr 1e-3, wd 1e-2), batch 256, <= 40 epochs, BCE, early stopping on the inner val BCE (patience 5, best weights restored).  Inputs are float32 numpy, already standardised.
    Returns (model, info)."""
    torch = _torch()
    torch.manual_seed(int(seed))
    model = make_model(arch, hw, cfg, uniform=uniform)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lossf = torch.nn.BCEWithLogitsLoss()
    Xf, Ef, yf = torch.from_numpy(Xfit), torch.from_numpy(Efit), torch.from_numpy(np.asarray(yfit, np.float32))
    Xv, Ev, yv = torch.from_numpy(Xval), torch.from_numpy(Eval), torch.from_numpy(np.asarray(yval, np.float32))
    g = torch.Generator().manual_seed(int(seed) + 12345)
    n = len(Xf)
    best, best_ep, bad, best_state = float("inf"), -1, 0, None
    hist = []
    for ep in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g)
        for a in range(0, n, cfg["batch"]):
            ix = perm[a:a + cfg["batch"]]
            opt.zero_grad()
            loss = lossf(model(Xf[ix], Ef[ix]), yf[ix])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tot = 0.0
            for a in range(0, len(Xv), cfg["eval_bs"]):
                tot += float(lossf(model(Xv[a:a + cfg["eval_bs"]], Ev[a:a + cfg["eval_bs"]]), yv[a:a + cfg["eval_bs"]])) * len(yv[a:a + cfg["eval_bs"]])
            lv = tot / len(yv)
        hist.append(lv)
        if lv < best - 1e-9:
            best, best_ep, bad = lv, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs_run": len(hist), "best_epoch": best_ep, "best_val_bce": best, "n_fit": int(n), "n_val": int(len(yv))}


def predict(model, X, E, bs=1024, want_attn=False):
    """Logits [n] (and attention [n, nq, T] float16 if asked and the model has it) on standardised float32 inputs."""
    torch = _torch()
    outs, atts = [], []
    with torch.no_grad():
        for a in range(0, len(X), bs):
            xb, eb = torch.from_numpy(X[a:a + bs]), torch.from_numpy(E[a:a + bs])
            if want_attn:
                o, at = model(xb, eb, return_attn=True)
                if at is not None:
                    atts.append(at.numpy().astype(np.float16))
            else:
                o = model(xb, eb)
            outs.append(o.numpy().astype(np.float64))
    o = np.concatenate(outs)
    return (o, np.concatenate(atts) if atts else None) if want_attn else o


def fold_view(S, f):
    """The fold-proper projection: outer fold f (0..4) uses S[f] (the PCA fitted on the outer-train windows of f); S[5] is the all-rows fit (eval-141 / maps only).  Asserted from probe 3's pca_g.py."""
    assert 0 <= f <= 5
    return S[f]


def cv_oof_nn(Sget, E, y, outer, shas, arch, hw, seed, cfg=CFG, collect_attn=False, uniform=False, folds=range(N_FOLD)):
    """Pooled out-of-fold logits of one (arm, readout seed): per outer fold f, the readout is fitted on the outer-train rows of S[f] with E appended, early-stopped on the inner clip-grouped 20 % split, scored on the
    held-out fold.  Token standardisation and E standardisation are fitted on the outer-train rows.  Returns (oof [N], attn [N, nq, T] float16 or None, info list)."""
    N, T = len(y), hw[0] * hw[1]
    oof = np.full(N, np.nan)
    att = np.zeros((N, cfg["n_query"], T), np.float16) if (collect_attn and arch == "att") else None
    sh = sha_ints(shas)
    info = []
    for f in folds:
        t0 = time.time()
        X = np.asarray(Sget(f), np.float32).reshape(N, T, K_TOK)
        te = outer == f
        tr = ~te
        mu, sd = tok_standardiser(X[tr])
        X = (X - mu) / sd
        em, es = E[tr].mean(0), E[tr].std(0) + 1e-9
        Es = ((E - em) / es).astype(np.float32)
        val = tr & inner_val_mask(sh, seed, f, cfg["val_group_mod"])
        fit = tr & ~val
        yv = np.asarray(y)[val]
        assert 0 < yv.sum() < len(yv), "inner val split has a single class"
        model, inf = train_one(arch, hw, X[fit], Es[fit], np.asarray(y)[fit], X[val], Es[val], yv, seed * 100 + f, cfg, uniform=uniform)
        if att is not None:
            o, a = predict(model, X[te], Es[te], want_attn=True)
            att[te] = a
        else:
            o = predict(model, X[te], Es[te])
        oof[te] = o
        inf.update({"fold": int(f), "seconds": time.time() - t0})
        info.append(inf)
    return oof, att, info


# ------------------------------------------------------------------------------------------------ controls
def shuffle_within_folds(y, outer, seed):
    """C1: labels permuted WITHIN each outer fold (clip-folds), so each fold keeps its exact prevalence (the pooled-AUC intercept bias of probe 3's C2 cannot arise)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    out = y.copy()
    for f in np.unique(outer):
        ix = np.flatnonzero(outer == f)
        out[ix] = y[rng.permutation(ix)]
    return out


BAND = (0.47, 0.53)


def c1_verdict(rec_fold):
    """rec_fold = {auc, lo, hi}: passes iff the CI covers 0.5 AND the point estimate sits in the band."""
    return bool(rec_fold["lo"] <= 0.5 <= rec_fold["hi"] and BAND[0] <= rec_fold["auc"] <= BAND[1])


def c2_verdict(seed2_fold, band_aucs=None):
    """seed 2 covers 0.5 -> pass; else the 5-draw mean (seed 2 + seeds 3..6) must lie in the band; fewer than 5 draws -> None (pending)."""
    if seed2_fold["lo"] <= 0.5 <= seed2_fold["hi"]:
        return True, "seed 2 CI covers 0.5"
    if band_aucs is None or len(band_aucs) < 5:
        return None, "seed 2 CI misses 0.5; seeds 3-6 pending"
    m = float(np.mean(band_aucs))
    return bool(BAND[0] <= m <= BAND[1]), f"5-draw mean {m:.4f} (band {BAND})"


# ------------------------------------------------------------------------------------------------ decision rule (PREREG 4)
def seed_spread(aucs):
    return float(max(aucs) - min(aucs))


def decide(P):
    """P[pop] = {"A": dAUC(G8att+E - V+E) dict | None, "B": dAUC(G8att+E - G4att+E) dict | None, "spread": {"G8att": s, "G4att": s} | None, "ctrl": {"G8att": bool|None, "G4att": bool|None, "pair": bool|None}}.
    Each dAUC dict has delta, lo, hi.  Returns {"verdict", "cells", "helps": {...}}.  Committed as written in PREREG_FROZEN.md section 4."""
    cells = {}
    for pop in POPS:
        p = P.get(pop) or {}
        for cell, need, sp_arms in (("A", ("G8att",), ("G8att",)), ("B", ("G8att", "G4att", "pair"), ("G8att", "G4att"))):
            c = p.get(cell)
            ctrl = p.get("ctrl") or {}
            sp = p.get("spread") or {}
            key = f"{pop}:{cell}"
            if c is None or any(a not in sp for a in sp_arms):
                cells[key] = {"status": "INCOMPLETE", "reason": "dAUC or seed spread missing", "helps": False, "below_0.01": None}
                continue
            flags = [ctrl.get(a) for a in need]
            if any(fl is None for fl in flags):
                status, reason = "INCOMPLETE", "controls not (yet) run: " + ",".join(a for a, fl in zip(need, flags) if fl is None)
            elif not all(flags):
                status, reason = "CONTROL-FAILED", "controls FAILED: " + ",".join(a for a, fl in zip(need, flags) if not fl)
            else:
                status, reason = "OK", ""
            thr_seed = 2.0 * max(sp[a] for a in sp_arms)
            helps = bool(c["delta"] >= 0.03 and c["lo"] > 0 and c["delta"] > thr_seed)
            cells[key] = {"status": status, "reason": reason, "delta": c["delta"], "lo": c["lo"], "hi": c["hi"], "seed_spread_max": max(sp[a] for a in sp_arms), "twice_spread": thr_seed,
                          "helps": helps and status == "OK", "helps_if_controls_ok": helps, "below_0.01": bool(c["delta"] < 0.01)}
    spatial = [k for k, v in cells.items() if k.endswith(":A") and v["helps"]]
    resol = [k for k, v in cells.items() if k.endswith(":B") and v["helps"]]
    if spatial or resol:
        parts = []
        if spatial:
            parts.append("SPATIAL READOUT HELPS (" + ",".join(spatial) + ")")
        if resol:
            parts.append("RESOLUTION HELPS UNDER A SPATIAL READOUT (" + ",".join(resol) + ")")
        verdict = "; ".join(parts)
    elif len(cells) == 4 and all(v["status"] == "OK" and v["below_0.01"] for v in cells.values()):
        verdict = "DOES NOT HELP"
    elif any(v["status"] == "INCOMPLETE" for v in cells.values()):
        verdict = "INCOMPLETE"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "cells": cells, "spatial_helps_cells": spatial, "resolution_helps_cells": resol}


# ------------------------------------------------------------------------------------------------ attention descriptors (descriptive only; mean softmax is NOT an attribution)
def attn_map(att, mask, hw):
    """Mean over the masked windows and over queries of the softmax weights -> [R, C] float64; None if the mask is empty."""
    if mask.sum() == 0:
        return None
    return np.asarray(att[mask], np.float64).mean((0, 1)).reshape(hw)


def attn_descriptors(m):
    """Share of mass in the lower half, normalised entropy, max token share of a map [R, C]."""
    R, C = m.shape
    p = m.reshape(-1) / m.sum()
    ent = float(-(p * np.log(p + 1e-30)).sum() / math.log(len(p)))
    return {"lower_half_share": float(m[R // 2:].sum() / m.sum()), "norm_entropy": ent, "max_token_share": float(p.max()), "argmax_rc": [int(np.argmax(p) // C), int(np.argmax(p) % C)]}


# ------------------------------------------------------------------------------------------------ grids and projections (copied from trigger_probe3/g_lib.py; equality asserted in the tests, in a subprocess)
def g4_from_g8(x):
    """x [n, 160, C] -> the exact 2x2 average [n, 40, C] float32, order (((a+b)+c)+d)*0.25 over (2r,2c),(2r,2c+1),(2r+1,2c),(2r+1,2c+1)."""
    x = np.asarray(x, np.float32)
    n, T, C = x.shape
    assert T == 160, T
    g = x.reshape(n, 8, 20, C)
    a, b, c, d = g[:, 0::2, 0::2], g[:, 0::2, 1::2], g[:, 1::2, 0::2], g[:, 1::2, 1::2]
    return ((((a + b) + c) + d) * np.float32(0.25)).reshape(n, 40, C)


def project_tokens(x, pcas):
    """x [m, T, 1024] float32 -> [len(pcas), m, T*16] (token-major: token t, component j at column 16 t + j), one centred projection per PCA in ``pcas``."""
    m, T, D = x.shape
    Ccat = np.concatenate([p["comp"] for p in pcas], 0)
    P = x.reshape(m * T, D) @ Ccat.T
    k = pcas[0]["comp"].shape[0]
    P = P.reshape(m, T, len(pcas), k)
    out = np.empty((len(pcas), m, T * k), np.float32)
    for i, p in enumerate(pcas):
        off = (p["mu"] @ p["comp"].T).astype(np.float32)
        out[i] = (P[:, :, i, :] - off).reshape(m, T * k)
    return out


# ------------------------------------------------------------------------------------------------ arms and bank helpers
def rec(o, y, clip, outer, counts):
    return {"pooled": L.auc_ci(o, y, clip, counts=counts), "fold": L.auc_ci_fold(o, y, clip, outer, counts=counts)}


def paired(a, b, y, clip, outer, counts):
    return {"fold": L.dauc_ci_fold(a, b, y, clip, outer, counts=counts), "pooled": L.dauc_ci(a, b, y, clip, counts=counts)}


def fold_mean_auc(score, y, outer):
    return L.FoldWAuc(score, y, outer).auc()
