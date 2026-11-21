"""PREREG_V8F_DECINFO library: windows, split, targets, loaders, readers (ridge / MLP / conv), controls. Tier T0, CPU, readouts only. See IMPL_CHOICES_DECINFO.md."""
import ctypes
import dataclasses
import glob
import hashlib
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.abspath(os.path.join(HERE, ".."))
for _s in ("plantest", "plantest2", "plantest3", "wmact"):
    sys.path.insert(0, os.path.join(THEORY, _s))
import numpy as np
import torch
import torch.nn as nn

import wmact_lib as W                      # asserts tanitad.__file__ under WMACT_CODE_ROOT
import plantest_lib as P
import plantest2_lib as Q
import plantest3_lib as L3
import diag_mlp as DM                      # train() unchanged
import diag_multiframe as DMF              # TRACKS, EVAL paths

torch.set_num_threads(4)
TREE = "F:/TanitAD-scratch/plantest_tree"
assert os.path.abspath(W.tanitad.__file__).replace("\\", "/").lower().startswith(TREE.lower()), W.tanitad.__file__
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU only: set CUDA_VISIBLE_DEVICES=-1"

EVAL = DMF.EVAL
TRACKS = DMF.TRACKS
PRE = os.path.join(THEORY, "plantest3", "runs", "pre_{tag}.json")
M2B = W.CKPT                                 # M2b_s0, M2b_s1
LADDER_GLOB = "C:/Users/Admin/t2v8f_pulls/ladder_*/ckpt_last.pt"
FROZEN_LADDER = ("C0_s0", "R35_s0", "R35H_s0", "S35_s0")
BAR = {"speed": 0.70, "yaw": 0.50}
RIDGE = L3.RIDGE
MIN_FREE_GB = 6.0
PCA_K = 32


# ------------------------------------------------------------------------------------------ RAM guard
def free_gb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS(); m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def wait_ram(min_gb=MIN_FREE_GB, poll=15, beat=60):
    t0 = last = time.time()
    while free_gb() < min_gb:
        if time.time() - last >= beat:
            print(time.strftime("%H:%M:%S"), f"RAM guard: free {free_gb():.1f} GB < {min_gb} GB, waiting {time.time() - t0:.0f} s", flush=True)
            last = time.time()
        time.sleep(poll)


# ------------------------------------------------------------------------------------------ split + guard (control 3)
def es_ids_of(fit_ids):
    return {e for e in fit_ids if hashlib.sha256(f"plantest3-mlp-es|{e}".encode()).digest()[0] < 51}


def split_guard(fit_eids, held_eids, es_eids, score_eids, tune_eids=None):
    """Control 3: pairwise-disjoint episode sets; the early-stopping part is inside the fit part; fit and held-out inside TUNE; nothing from SCORE."""
    fit_eids, held_eids, es_eids, score_eids = (set(map(str, s)) for s in (fit_eids, held_eids, es_eids, score_eids))
    assert not (fit_eids & held_eids), "fit reads a held-out episode"
    assert es_eids <= fit_eids, "early-stopping episodes must be fit episodes"
    assert not ((fit_eids | held_eids) & score_eids), "a SCORE episode is read"
    if tune_eids is not None:
        assert (fit_eids | held_eids) <= set(map(str, tune_eids)), "an episode outside TUNE is read"
    return True


def build_split(eid):
    """eid: episode id per window of the 1,976 selected windows. Returns dict of window positions (into the selection)."""
    tune, score = Q.split_indices(eid)
    fit_ids, held_ids = L3.inner_split(eid[tune])
    es = es_ids_of(fit_ids)
    split_guard(fit_ids, held_ids, es, set(eid[score]), set(eid[tune]))
    fit = np.array([i for i in tune if eid[i] in fit_ids])
    held = np.array([i for i in tune if eid[i] in held_ids])
    return dict(tune=tune, score=score, fit=fit, held=held, fit_ids=fit_ids, held_ids=held_ids, es_ids=es)


@dataclasses.dataclass
class Data:
    feats: np.ndarray        # (n, 16, 40, 1024) float16, fit windows first then held-out
    act: np.ndarray          # (n, 16, 2) raw
    eid: np.ndarray          # (n,)
    vk: np.ndarray           # (n, 16) speed, NaN missing
    yk: np.ndarray           # (n, 16) yaw rate
    nf: int                  # first nf windows are fit
    es: np.ndarray           # bool (n,) fit window belongs to an early-stopping episode
    pos: np.ndarray          # selection positions of the windows (into the 1,976)
    ds_idx: np.ndarray       # indices into the 2,399 dataset windows
    fit_eps: int
    held_eps: int
    z1_ref: dict             # tag -> stored Z1 of wmact/latents.pt for these windows (M2b)

    @property
    def n(self):
        return len(self.eid)


def load_tune(read_feats=True):
    """Open ONLY the TUNE windows (asserted). feats are read from the dataset item by item."""
    L = torch.load(EVAL, weights_only=False)
    okall = L["ok"].numpy().all(1)
    sel = np.nonzero(okall)[0]
    eid_all = np.array(L["eid"])
    eid = eid_all[sel].astype("U36")
    sp = build_split(eid)
    order = np.concatenate([sp["fit"], sp["held"]])
    assert set(order.tolist()) == set(sp["tune"].tolist()) and not (set(order.tolist()) & set(sp["score"].tolist()))
    ds_idx = sel[order]
    TRK = torch.load(TRACKS, weights_only=False)["tracks"]
    ds = wins = None
    if read_feats:
        ds, wins, eid_ds = W.open_windows(16)
        assert (eid_ds == eid_all).all()
    n = len(order)
    feats = np.zeros((n, 16, 40, 1024), np.float16) if read_feats else None
    act = np.zeros((n, 16, 2), np.float32)
    vk = np.full((n, 16), np.nan)
    tt = np.asarray(L["t"])
    assert not (set(ds_idx.tolist()) & set(sel[sp["score"]].tolist())), "a SCORE window would be read"
    for j, di in enumerate(ds_idx):
        if read_feats:
            it = ds.item(wins[di][0], wins[di][1])
            assert bool(np.all(it["frame_ok"])) and int(it["t"]) == int(tt[di])
            feats[j] = it["feats"]
            act[j] = it["act"]
        e = str(eid_all[di]); g = int(tt[di]) - 15
        r = TRK.get(e)
        if r is not None and g >= 0:
            r = np.asarray(r)
            for k in range(16):
                if 2 * (g + k) < len(r):
                    vk[j, k] = float(r[2 * (g + k), 3])
    if not read_feats:                                            # stored raw actions (identical to the item's 'act'; checked in the full path)
        act = L["A_raw"][ds_idx].float().numpy()
    else:
        assert np.array_equal(act, L["A_raw"][ds_idx].float().numpy())
    eid_o = eid[order]
    nf = len(sp["fit"])
    es = np.array([(e in sp["es_ids"]) for e in eid_o]) & (np.arange(n) < nf)
    z1_ref = {t: L["Z1"][t][ds_idx].float() for t in L["Z1"]}
    return Data(feats, act, eid_o, vk, act[:, :, 1].astype(np.float64), nf, es, order, ds_idx, len(sp["fit_ids"]), len(sp["held_ids"]), z1_ref), sp


# ------------------------------------------------------------------------------------------ models
def load_any(path):
    """z1-capable loader (IMPL section 2): adapter, pos and wm.proj must load; dropped tensors must be wm.* outside wm.proj."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    fields = {f.name for f in dataclasses.fields(W.M.V8FMiniConfig)}
    cfg = W.M.V8FMiniConfig(**{k: v for k, v in ck["cfg"].items() if k in fields})
    model = W.M.V8FMini(cfg, W.M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * W.M.K30)))
    sd = model.state_dict()
    use = {k: v for k, v in ck["model"].items() if k in sd and sd[k].shape == v.shape}
    dropped = [k for k in ck["model"] if k not in use]
    need = [k for k in sd if k.startswith("adapter.") or k == "pos" or k.startswith("wm.proj")]
    assert need and all(k in use for k in need), [k for k in need if k not in use]
    assert all((k.startswith("wm.") and not k.startswith("wm.proj")) or k == "hacc_w" for k in dropped), [k for k in dropped if not (k.startswith("wm.") or k == "hacc_w") or k.startswith("wm.proj")]
    model.load_state_dict(use, strict=False)
    return model.eval(), dict(path=path, dropped=len(dropped), step=ck.get("step"), cfg_hmax=ck["cfg"].get("hmax"))


def ladder_models():
    out = {}
    for p in sorted(glob.glob(LADDER_GLOB)):
        tag = os.path.basename(os.path.dirname(p))[len("ladder_"):]
        out[tag] = p
    return out


@torch.no_grad()
def adapt_chunks(model, feats, bs=32):
    """Yield (start, adapt [b, 16, 40, 256] float32) in chunks."""
    for s in range(0, len(feats), bs):
        f = torch.from_numpy(feats[s:s + bs]).float()
        yield s, model.adapter(f) + model.pos[None, None]


@torch.no_grad()
def vector_inputs(model, feats, bs=32):
    """z1 [n,16,128] and pooled token [n,16,256] (two separate calls on the same adapter tokens, control 5)."""
    z1, pooled = [], []
    for _, a in adapt_chunks(model, feats, bs):
        pooled.append(a.mean(2))
        z1.append(model.wm.z1(a))
    return torch.cat(z1), torch.cat(pooled)


def fit_pca_stream(chunks, d, nfit_windows, k=PCA_K):
    """chunks yields float tensors [b, 16, 40, d]; only the first nfit_windows windows (fit episodes) enter."""
    s = np.zeros(d); S = np.zeros((d, d)); n = 0
    seen = 0
    for x in chunks:
        take = min(len(x), nfit_windows - seen)
        if take <= 0:
            break
        t = x[:take].reshape(-1, d).double()
        s += t.sum(0).numpy(); S += (t.T @ t).numpy(); n += t.shape[0]; seen += take
    assert seen == nfit_windows
    mu = s / n
    C = S / n - np.outer(mu, mu)
    w, V = np.linalg.eigh(C)
    comps = V[:, ::-1][:, :k].copy()
    return mu, comps


def project_stream(chunks, mu, comps, n):
    """chunks -> scores (n, 16, k, 40) float32 (channel axis before position)."""
    out = np.zeros((n, 16, comps.shape[1], 40), np.float32)
    mu_t, c_t = torch.from_numpy(mu).float(), torch.from_numpy(comps).float()
    s = 0
    for x in chunks:
        sc = (x - mu_t) @ c_t                                  # [b,16,40,k]
        out[s:s + len(x)] = sc.permute(0, 1, 3, 2).numpy()
        s += len(x)
    assert s == n
    return out


def adapter_grid(model, D):
    mu, comps = fit_pca_stream((a for _, a in adapt_chunks(model, D.feats)), 256, D.nf)
    return project_stream((a for _, a in adapt_chunks(model, D.feats)), mu, comps, D.n), (mu, comps)


def encoder_grid(D, bs=32):
    ch = lambda: (torch.from_numpy(D.feats[s:s + bs]).float() for s in range(0, D.n, bs))
    mu, comps = fit_pca_stream(ch(), 1024, D.nf)
    return project_stream(ch(), mu, comps, D.n), (mu, comps)


# ------------------------------------------------------------------------------------------ samples
def stack_ch(X, m):
    """X (n, 16, C, ...) -> (n, 16-m+1, m*C, ...): frames t-m+1..t concatenated along the channel axis (axis 2), for t = m-1..15."""
    return np.concatenate([X[:, m - 1 - j: X.shape[1] - j] for j in range(m)], axis=2)


def make_xy(X, y, m):
    """X (n,16,C[,40]) numpy; y (n,16). Returns Xs (n, T, ...), ys (n, T)."""
    return stack_ch(X, m), y[:, m - 1:]


def r2(y, p):
    return float(L3.r2_cols(y[:, None], p[:, None])[0])


def targets(D):
    return {"speed": D.vk, "yaw": D.yk}


def derange_tracks(y_fit, eid_fit, seed):
    """Control 1: permute whole target tracks among fit windows so that no window keeps a track of its own episode."""
    rng = np.random.default_rng(seed)
    n = len(y_fit)
    perm = rng.permutation(n)
    for _ in range(2000):
        bad = np.nonzero(eid_fit[perm] == eid_fit)[0]
        if len(bad) == 0:
            break
        for i in bad:
            j = int(rng.integers(n))
            if eid_fit[perm[j]] != eid_fit[i] and eid_fit[perm[i]] != eid_fit[j]:
                perm[i], perm[j] = perm[j], perm[i]
    assert (eid_fit[perm] != eid_fit).all()
    return y_fit[perm]


# ------------------------------------------------------------------------------------------ readers
def _flat(Xs, ys, ix):
    x = Xs[ix].reshape(len(ix) * Xs.shape[1], -1)
    y = ys[ix].reshape(-1)
    ok = np.isfinite(y) & np.isfinite(x).all(1)
    return x[ok], y[ok]


def ridge_pred(xf, yf, xh):
    sd = yf.std()
    W_, b = L3.ridge_fit(xf, yf / sd, RIDGE)
    xh = torch.as_tensor(xh, dtype=torch.double)
    return (xh @ W_ + b).numpy() * sd


def ridge_read(D, Xs, y, y_fit_use):
    """Fit on the fit windows with targets y_fit_use (true or shuffled), score on the held windows against y. Returns (r2, n_fit, n_held)."""
    xf, yf = _flat(Xs, y_fit_use, np.arange(D.nf))
    xh, yh = _flat(Xs, y, np.arange(D.nf, D.n))
    return r2(yh, ridge_pred(xf, yf, xh)), len(yf), len(yh)


class ConvNet(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32 * 40, 1))

    def forward(self, x):
        return self.head(self.c(x).flatten(1))[:, 0]


def conv_train(xa, ya, xv, yv, xh, seed, epochs=80, bs=256, patience=10):
    torch.manual_seed(seed)
    mu = xa.mean((0, 2, 3), keepdim=True); sd = xa.std((0, 2, 3), keepdim=True) + 1e-6
    ym, ys = ya.mean(), ya.std() + 1e-9
    xa, xv, xh = (xa - mu) / sd, (xv - mu) / sd, (xh - mu) / sd
    t = (ya - ym) / ys
    net = ConvNet(xa.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, bad = 1e9, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xa), generator=g)
        for i in range(0, len(xa), bs):
            j = perm[i:i + bs]
            loss = ((net(xa[j]) - t[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(xv) * ys + ym - yv) ** 2).mean())
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: b.clone() for k, b in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return (net(xh) * ys + ym).numpy(), ep + 1


def _parts(D, Xs, y, y_fit_use, grid):
    """(train, es, held) tensors. train/es from fit windows split by D.es; held from held windows scored against y."""
    fit_ix = np.arange(D.nf)
    tr = fit_ix[~D.es[:D.nf]]; va = fit_ix[D.es[:D.nf]]; he = np.arange(D.nf, D.n)
    assert not (set(tr) & set(he)) and not (set(va) & set(he)) and not (set(tr) & set(va))
    def part(ix, yy):
        x, yv = _flat(Xs, yy, ix)
        x = torch.from_numpy(x).float()
        if grid:
            x = x.reshape(len(x), -1, 4, 10)
        return x, torch.from_numpy(yv).float()
    return part(tr, y_fit_use), part(va, y_fit_use), part(he, y)


def mlp_read(D, Xs, y, y_fit_use, seed=0):
    (xa, ya), (xv, yv), (xh, yh) = _parts(D, Xs, y, y_fit_use, False)
    p, ep = DM.train(xa, ya, xv, yv, xh, seed=seed)
    return r2(yh.numpy().astype(np.float64), p.astype(np.float64)), dict(n_train=len(ya), n_es=len(yv), n_held=len(yh), epochs=ep)


def conv_read(D, Xs, y, y_fit_use, seed):
    (xa, ya), (xv, yv), (xh, yh) = _parts(D, Xs, y, y_fit_use, True)
    p, ep = conv_train(xa, ya, xv, yv, xh, seed)
    return r2(yh.numpy().astype(np.float64), p.astype(np.float64)), dict(n_train=len(ya), n_es=len(yv), n_held=len(yh), epochs=ep)


# ------------------------------------------------------------------------------------------ control 2: planted signal
def planted_target(Xf, Xh, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(Xf.shape[1])
    pf, ph = Xf.astype(np.float64) @ w, Xh.astype(np.float64) @ w
    sd = pf.std()
    return pf / sd + noise * rng.standard_normal(len(pf)), ph / sd + noise * rng.standard_normal(len(ph))


def planted_check(Xf, Xh, seed=0):
    """Returns (r2 with the signal, r2 with zeroed features). Pass bars: >= 0.80 and <= 0.02."""
    yf, yh = planted_target(Xf, Xh, seed)
    a = r2(yh, ridge_pred(Xf, yf, Xh))
    z = r2(yh, ridge_pred(np.zeros_like(Xf), yf, np.zeros_like(Xh)))
    return a, z
