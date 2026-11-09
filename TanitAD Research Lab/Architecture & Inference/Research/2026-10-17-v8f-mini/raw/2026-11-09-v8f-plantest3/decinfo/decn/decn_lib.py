"""PREREG_V8F_DECN library: chunked ridge, fit-set assembly across N, the unit reader, digests. Tier T0, CPU. See IMPL_CHOICES_DECN.md."""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
DI = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, DI)
import numpy as np
import torch

import decinfo_lib as DL

SCRATCH = "F:/TanitAD-scratch/decn_scratch/run1"
LAT = f"{SCRATCH}/lat"
TUNE_NPZ = f"{SCRATCH}/tune_inputs.npz"
RUNS = os.path.join(HERE, "runs")
THEORY = DL.THEORY
CAP_N53, CAP_BIG = 120, 60
CHUNK = 20000


# ------------------------------------------------------------------------------------------------ ridge, chunked (same closed form as plantest3_lib.ridge_fit)
def ridge_pred_chunked(xf, yf, xh, ridge=DL.RIDGE, chunk=CHUNK):
    sd = float(yf.std())
    y = torch.as_tensor(yf / sd, dtype=torch.double)
    n, d = xf.shape
    G = torch.zeros(d, d, dtype=torch.double); bv = torch.zeros(d, dtype=torch.double); sx = torch.zeros(d, dtype=torch.double)
    for i in range(0, n, chunk):
        xc = torch.as_tensor(xf[i:i + chunk], dtype=torch.double)
        G += xc.T @ xc; bv += xc.T @ y[i:i + chunk]; sx += xc.sum(0)
    mu, ym = sx / n, y.mean()
    Gc = G - n * torch.outer(mu, mu) + ridge * n * torch.eye(d, dtype=torch.double)
    bc = bv - n * mu * ym
    Wt = torch.linalg.solve(Gc, bc)
    b = ym - mu @ Wt
    out = np.empty(len(xh))
    for i in range(0, len(xh), chunk):
        out[i:i + chunk] = (torch.as_tensor(xh[i:i + chunk], dtype=torch.double) @ Wt + b).numpy() * sd
    return out


def ridge_read_c(D, Xs, y, y_fit_use):
    xf, yf = DL._flat(Xs, y_fit_use, np.arange(D.nf))
    xh, yh = DL._flat(Xs, y, np.arange(D.nf, D.n))
    return DL.r2(yh, ridge_pred_chunked(xf, yf, xh)), len(yf), len(yh)


def planted_check_c(Xf, Xh, seed=0):
    yf, yh = DL.planted_target(Xf, Xh, seed)
    a = DL.r2(yh, ridge_pred_chunked(Xf, yf, Xh))
    z = DL.r2(yh, ridge_pred_chunked(np.zeros_like(Xf), yf, np.zeros_like(Xh)))
    return a, z


def mlp_read_c(D, Xs, y, y_fit_use, epochs, seed=0):
    (xa, ya), (xv, yv), (xh, yh) = DL._parts(D, Xs, y, y_fit_use, False)
    p, ep = DL.DM.train(xa, ya, xv, yv, xh, seed=seed, epochs=epochs)
    return DL.r2(yh.numpy().astype(np.float64), p.astype(np.float64)), dict(n_train=len(ya), n_es=len(yv), n_held=len(yh), epochs=ep, cap=epochs, hit_cap=bool(ep >= epochs))


# ------------------------------------------------------------------------------------------------ digests / guards (control 5)
def held_digest(vk_h, yk_h, eid_h):
    h = hashlib.sha256()
    h.update(np.nan_to_num(np.asarray(vk_h, np.float64), nan=-9e9).tobytes())
    h.update(np.ascontiguousarray(np.asarray(yk_h, np.float64)).tobytes())
    h.update("\n".join(map(str, eid_h)).encode())
    return h.hexdigest()


def split_guard_n(fit_ids, held_ids, es_ids, eval_clips, train_ids):
    """Control 5 for a fit set that holds train clips: pairwise disjoint, no eval clip among the additions, early-stopping inside fit."""
    fit, held, tr, ev = set(map(str, fit_ids)), set(map(str, held_ids)), set(map(str, train_ids)), set(map(str, eval_clips))
    assert not (fit & held), "fit reads a held-out episode"
    assert not (tr & ev), "an eval clip is among the train additions"
    assert not (tr & held), "a train addition is a held-out episode"
    assert set(map(str, es_ids)) <= fit, "early-stopping episodes must be fit episodes"
    return True


# ------------------------------------------------------------------------------------------------ data
def clip_lists():
    c141 = json.load(open(f"{THEORY}/wmact/train_stream/clips141.json", encoding="utf-8"))
    ce = json.load(open(os.path.join(HERE, "clips_extra.json"), encoding="utf-8"))
    return c141, ce


def load_tune_npz():
    z = np.load(TUNE_NPZ, allow_pickle=False)
    return {k: z[k] for k in z.files}


def available_clips(order):
    """Number of leading clips of `order` whose shard exists (contiguous)."""
    k = 0
    for c in order:
        if os.path.exists(f"{LAT}/{c}.npz"):
            k += 1
        else:
            break
    return k


def load_train_arrays(order, k, key):
    """Concatenate shard arrays `key` of the first k clips: returns (X (w,16,C), clip index per window)."""
    xs, ci = [], []
    for i, c in enumerate(order[:k]):
        z = np.load(f"{LAT}/{c}.npz")
        a = z[key]
        xs.append(a); ci += [i] * len(a)
    return np.concatenate(xs), np.array(ci)


class FitSet:
    pass


def assemble(T, key, X_tr, ci_tr, order, k_clips, tr_targets):
    """Fit windows = TUNE fit (T) + first k_clips train clips; held-out windows = TUNE held. Returns a duck-typed D and X."""
    nf0 = int(T["nf"]); n0 = len(T["eid"])
    sel = ci_tr < k_clips
    v_tr, y_tr = tr_targets[0][sel], tr_targets[1][sel]
    eid_tr = np.array([order[i] for i in ci_tr[sel]], dtype="U36")
    D = FitSet()
    D.nf = nf0 + int(sel.sum()); D.n = D.nf + (n0 - nf0)
    D.eid = np.concatenate([T["eid"][:nf0], eid_tr, T["eid"][nf0:]])
    D.vk = np.concatenate([T["vk"][:nf0], v_tr, T["vk"][nf0:]])
    D.yk = np.concatenate([T["yk"][:nf0], y_tr, T["yk"][nf0:]])
    ids_tr = set(order[:k_clips])
    es_set = DL.es_ids_of(ids_tr)
    es_tr = np.array([e in es_set for e in eid_tr], bool)
    D.es = np.concatenate([T["es"][:nf0], es_tr, np.zeros(n0 - nf0, bool)])
    X = np.concatenate([T[key][:nf0], X_tr[sel], T[key][nf0:]]).astype(np.float32)
    D.n_fit_episodes = int(T["fit_eps"]) + k_clips
    D.n_held_episodes = int(T["held_eps"])
    D.held_digest = held_digest(D.vk[D.nf:], D.yk[D.nf:], D.eid[D.nf:])
    return D, X, ids_tr


def derange_gen(D, y, seed):
    ys = y.copy()
    ys[:D.nf] = DL.derange_tracks(y[:D.nf], D.eid[:D.nf], seed)
    return ys


def read_unit(D, X, rung, tag, Nlabel, log, ms=(1, 4)):
    rows, planted = [], []
    ys_all = DL.targets(D)
    cap = CAP_N53 if Nlabel == "N53" else CAP_BIG
    for m in ms:
        Xs = DL.stack_ch(X, m)
        Xf = Xs[:D.nf].reshape(D.nf * Xs.shape[1], -1); Xh = Xs[D.nf:].reshape((D.n - D.nf) * Xs.shape[1], -1)
        a, z = planted_check_c(Xf, Xh)
        planted.append(dict(rung=rung, tag=tag, N=Nlabel, m=m, r2_signal=a, r2_zeroed=z, passed=bool(a >= 0.80 and z <= 0.02), n_features=int(Xf.shape[1])))
        log(rung, tag, Nlabel, f"m={m} planted {a:.3f} zeroed {z:.3f}")
        del Xf, Xh
        for ti, tgt in enumerate(("speed", "yaw")):
            y = ys_all[tgt]
            ysh = derange_gen(D, y, seed=100 + ti)
            yS, ySh = y[:, m - 1:], ysh[:, m - 1:]
            r, nf_, nh_ = ridge_read_c(D, Xs, yS, yS)
            rs, _, _ = ridge_read_c(D, Xs, yS, ySh)
            rows.append(dict(rung=rung, tag=tag, N=Nlabel, reader="ridge", m=m, target=tgt, r2=r, r2_shuffled=rs, n_fit=nf_, n_held=nh_,
                             n_fit_windows=int(D.nf), n_fit_episodes=D.n_fit_episodes, n_held_episodes=D.n_held_episodes))
            log(rung, tag, Nlabel, "ridge", m, tgt, f"{r:+.3f} shuf {rs:+.3f}")
            r, e = mlp_read_c(D, Xs, yS, yS, cap)
            rs, es_ = mlp_read_c(D, Xs, yS, ySh, cap)
            rows.append(dict(rung=rung, tag=tag, N=Nlabel, reader="mlp", m=m, target=tgt, r2=r, r2_shuffled=rs, n_fit=e["n_train"] + e["n_es"], n_held=e["n_held"],
                             n_fit_windows=int(D.nf), n_fit_episodes=D.n_fit_episodes, n_held_episodes=D.n_held_episodes, detail=e, detail_shuffled=es_))
            log(rung, tag, Nlabel, "mlp", m, tgt, f"{r:+.3f} shuf {rs:+.3f} ep {e['epochs']}/{cap}")
    return rows, planted
