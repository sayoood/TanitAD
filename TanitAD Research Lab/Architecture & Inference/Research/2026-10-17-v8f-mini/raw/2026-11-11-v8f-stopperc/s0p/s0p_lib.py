"""s0p_lib.py -- shared pieces of the S0-P probe pipeline (card PREREG_V8F_STOPPERC.md section 4 "S0-P" and section 8; AMENDMENT_1_S0.md).

Pure numpy parts (no torch, no tanitad): the sampler, the E layout, the npz writer, the AUROC.  The torch/tanitad parts (tree set-up, lean store) import lazily.

E layout (18 floats), as T reads them at the NOW frame (the last history slot of the batch):
    E[0:8]   p10f last row         (kinematic features from the native pose track, kin_mode yaw)
    E[8:13]  nav last row          (5)
    E[13:17] limit                 (4)  -- AMENDMENT 1: the CAUSAL bucket (past speed only), never the logged sidecar
    E[17]    limit_ok              (1)  -- AMENDMENT 1: the causal bucket's ok flag
E-limit = E[:, :13].
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

SEED = 20261111
N_POS, N_NEG = 4000, 12000
E_DIM, E_NO_LIMIT = 18, 13
GOAL_STOP_BIT = 6               # GOAL22 index of STOP_POINT
CAUSAL_STEPS = 20               # v8f_t1_eval.CAUSAL_STEPS (asserted against the tree at run time)
FIELDS = ("E", "v40", "v160", "mean", "y", "clip", "ws", "eid")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------ the sampler (card section 8, S0-P)
def draw_sample(clip, ws, pos, neg, seed=SEED, n_pos=N_POS, n_neg=N_NEG):
    """Pool arrays (one entry per goal-observed window): clip (str), ws (int), pos / neg (bool, disjoint).

    Card: seed numpy.random.default_rng(20261111); positives sorted by (clip id, window start), `n_pos` drawn without replacement; `n_neg` negatives drawn the same way from the
    goal-observed non-positive windows.  Reading fixed here (the card does not say): ONE generator, positives drawn first, then negatives; clip id = the clip-id STRING, sorted
    lexicographically; each draw is a `rng.choice(n_pool, k, replace=False)` over the sorted pool; the returned indices are sorted ascending (pool order).
    -> (idx_pos, idx_neg): indices into the ARGUMENTS."""
    clip, ws, pos, neg = np.asarray(clip), np.asarray(ws, np.int64), np.asarray(pos, bool), np.asarray(neg, bool)
    assert not (pos & neg).any(), "positive and negative pools overlap"
    rng = np.random.default_rng(seed)
    out = []
    for mask, k in ((pos, n_pos), (neg, n_neg)):
        ix = np.where(mask)[0]
        order = np.lexsort((ws[ix], clip[ix]))          # primary key clip id, secondary window start
        ix = ix[order]
        if len(ix) < k:
            raise ValueError(f"pool has {len(ix)} windows, {k} requested")
        pick = rng.choice(len(ix), size=k, replace=False)
        out.append(np.sort(ix[pick]))
    return out[0], out[1]


# ------------------------------------------------------------------------------------------------------------ E
def e_from_item_arrays(p10f_last, nav_last, limit, limit_ok):
    """[N,8] [N,5] [N,4] [N] -> E [N,18] float32 (the layout above)."""
    p10f_last, nav_last, limit = (np.asarray(x, np.float32) for x in (p10f_last, nav_last, limit))
    lo = np.asarray(limit_ok, np.float32).reshape(-1, 1)
    assert p10f_last.shape[1] == 8 and nav_last.shape[1] == 5 and limit.shape[1] == 4
    return np.concatenate([p10f_last, nav_last, limit, lo], 1).astype(np.float32)


# ------------------------------------------------------------------------------------------------------------ npz writer / reader
def save_npz(path, E, v40, v160, mean, y, clip, ws, eid, meta, ci=None):
    """v40 / v160: uint8 views of float8_e4m3fn [N,40,1024] / [N,160,1024]; mean fp16 [N,1024]."""
    assert E.shape[1] == E_DIM and v40.dtype == np.uint8 and v160.dtype == np.uint8 and mean.dtype == np.float16
    n = len(E)
    assert all(len(a) == n for a in (v40, v160, mean, y, clip, ws, eid))
    extra = {} if ci is None else {"ci": np.asarray(ci, np.int64)}      # eval side: the index into the eval join (STOPDEC's `clip_index`)
    np.savez(path, **extra, E=E.astype(np.float32), v40=v40, v160=v160, mean=mean, y=np.asarray(y, np.uint8), clip=np.asarray(clip).astype("U40"),
             ws=np.asarray(ws, np.int64), eid=np.asarray(eid, np.int64), meta=np.array(json.dumps(meta, sort_keys=True)))
    return sha256_file(path)


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def decode_fp8(u8):
    """uint8 bytes of float8_e4m3fn -> float32 (torch)."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(u8)).view(torch.float8_e4m3fn).to(torch.float32).numpy()


# ------------------------------------------------------------------------------------------------------------ AUROC (Mann-Whitney, ties one half; identical rule to stopcue_read.auroc)
def auroc(score, label):
    from scipy.stats import rankdata
    s, y = np.asarray(score, float), np.asarray(label, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s, method="average")
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


# ------------------------------------------------------------------------------------------------------------ the read tree
def setup_tree(tree):
    """Put <tree>/stack and <tree>/taniteval first on sys.path and assert tanitad comes from there.  `tree` is the repo root of the code of record (dev box: the STOPBAL read tree; box 2: its repo)."""
    tree = os.path.abspath(tree).replace("\\", "/")
    for p in (tree + "/taniteval", tree + "/stack"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tanitad
    f = os.path.abspath(tanitad.__file__).replace("\\", "/")
    assert f.startswith(tree), f"tanitad imported from {f}, not from {tree}"
    return tree


def load_eval_module(tree):
    """stack/scripts/v8f_t1_eval.py loaded from its file (the module `E` of the STOPDEC / STOPORC instruments)."""
    path = tree + "/stack/scripts/v8f_t1_eval.py"
    spec = importlib.util.spec_from_file_location("s0p_v8f_t1_eval", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["s0p_v8f_t1_eval"] = m
    spec.loader.exec_module(m)
    assert m.CAUSAL_STEPS == CAUSAL_STEPS, (m.CAUSAL_STEPS, CAUSAL_STEPS)
    return m


class LazyFeats:
    """Stands in for a clip's feature grid inside ClipStore.get: `get` only calls `.to(float16).numpy()`, `.shape[0]` on it, and the window item indexes rows.  Serving zeros of
    the right row count keeps ClipStore.get's code path (hence p10f, nav, v, pose) byte-for-byte the trainer's while never reading the grid (the real rows are mmap-read separately)."""

    def __init__(self, T):
        self.shape = (T, 1, 1)

    def to(self, *_a, **_k):
        return self

    def numpy(self):
        return _ZeroRows(self.shape[0])


class _ZeroRows:
    def __init__(self, T):
        self.shape = (T, 1, 1)

    def __getitem__(self, idx):
        return np.zeros((len(idx), 1, 1), np.float16)


class MmapProxyLoad:
    """Context manager: torch.load(path, ...) -> the clip dict with weights_only+mmap, with the grid replaced by LazyFeats and the real mmap'd object kept in `.last` (clip path -> object)."""

    def __init__(self, grid="pool4x10"):
        self.grid, self.last = grid, {}

    def __enter__(self):
        import torch
        self._orig = torch.load
        outer = self

        def load(path, *a, **k):
            o = outer._orig(path, map_location="cpu", weights_only=True, mmap=True)
            outer.last[str(path)] = o
            T = o["pose_grid"].shape[0]
            view = {kk: o[kk] for kk in o.keys() if kk not in (outer.grid, "pool8x20", "mean")}
            view[outer.grid] = LazyFeats(T)
            return view
        torch.load = load
        return self

    def __exit__(self, *exc):
        import torch
        torch.load = self._orig
        self.last.clear()
        return False
