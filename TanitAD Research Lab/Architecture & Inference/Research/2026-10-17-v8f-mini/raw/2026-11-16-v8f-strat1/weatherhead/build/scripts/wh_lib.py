"""wh_lib.py -- PREREG_V8F_WEATHERHEAD form (a): the shared library of the fit, the checker and the reader (card sections 4.2, 4.4, 4.5).

Evidence class of everything this file produces when RUN: MEASURED (ours).  Nothing here is a result; a SMOKE run is marked SMOKE and never quoted.
The tree of record is ``WH_TREE`` (default ``<work>/tree``): ``tanitad`` must import from it (asserted) -- never from G: or E:.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BUILD = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(BUILD)
TREE = os.path.abspath(os.environ.get("WH_TREE", WORK + "/tree"))
for _p in (TREE + "/taniteval", TREE + "/stack"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from tanitad.models import v8f_mini as M  # noqa: E402

if not os.path.abspath(M.__file__).replace("\\", "/").lower().startswith(TREE.replace("\\", "/").lower()):
    raise RuntimeError(f"tanitad imports from {M.__file__}, not from the tree of record {TREE}")

READ = os.environ.get("STRAT1_READ", "F:/TanitAD-scratch/strat1/read")
INP = READ + "/inputs"
CK = READ + "/ckpts"
TRAIN_CACHE = os.environ.get("WH_TRAIN_CACHE", "F:/TanitAD-scratch/speedread/cache/train")
EVAL_CACHE = os.environ.get("WH_EVAL_CACHE", "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval")
BASE_COMMIT = "a4357e79"
GPU_LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"

#: card 4.5 pins (full sha256 where the card gives a prefix, the prefix is checked against the bank manifest)
PINS = {"strat_labels_table.npz": "8055a321df6db1725ea17abde2999d48bcb68d43f1a480a69ed42dd0cf184eff",
        "eval_grid_windows.json": "659c3b65e364ea5de87a4e17d076eba5645df95cdcf39d8abc08caab1b2d594c",
        "join_eval141.npz": "f0ea722f59c1a4d8ba81ce7f9b204dd3a28950fc1c16778f63c2962786e5206a"}
CKPT_PREFIX = {"S1_s0": "6c2af475", "S1_s1": "e51b756f", "C0_s0": "ad91be31", "C0_s1": "17c751a0"}
READER_PINS = {"strat1_ctx_read.py": "1b9377ba", "taniteval/taniteval/ci.py": "528276c5"}
E_GRID = (4, 8, 16)
NW = 8                                    # NOW frames per train clip
BS, LR, WD, WARM = 256, 1e-3, 0.01, 100
COMMIT_MARGIN = 5.5e9                     # card 4.5 / ruling 6: free commit must exceed the feature size plus this
FOK = 4                                   # frames per window (t-3 .. t)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def pinned(name):
    p = f"{INP}/{name}"
    got = sha256_file(p)
    if got != PINS[name]:
        raise RuntimeError(f"pinned input {name}: sha256 {got[:12]} != {PINS[name][:12]}")
    return p


# ------------------------------------------------------------------------------------------------ memory (ruling 6, P-COMMIT)
def commit_headroom() -> float:
    """Free commit charge in bytes (GlobalMemoryStatusEx ullAvailPageFile); +inf off Windows.  This is the number the dev-box OOMs read (memory rule devbox-commit-limit-oom)."""
    if os.name != "nt":
        return float("inf")

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    s = MS()
    s.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return float(s.ullAvailPageFile)


def feature_bytes(n_windows: int) -> int:
    return int(n_windows) * FOK * 40 * 1024            # float8: 1 byte per element


def choose_source(n_windows: int, mode: str = "auto", headroom=None) -> dict:
    """P-COMMIT.  'auto': in-RAM iff free commit >= feature bytes + 5.5 GB, else stream (memory-mapped clip files).  Returns {'mode', 'feature_bytes', 'commit_free', 'needed'}."""
    need = feature_bytes(n_windows) + COMMIT_MARGIN
    free = commit_headroom() if headroom is None else float(headroom)
    if mode == "auto":
        m = "ram" if free >= need else "stream"
    elif mode in ("ram", "stream"):
        m = mode
    else:
        raise ValueError(mode)
    return {"mode": m, "requested": mode, "feature_bytes": feature_bytes(n_windows), "commit_free": free, "needed": need}


# ------------------------------------------------------------------------------------------------ rows and labels
def load_table(path=None):
    return dict(np.load(path or pinned("strat_labels_table.npz")))


def train_rowset(tab, clips=None):
    """The probe's window set (card 4.2): per train clip, NOW frames t = unique(round(linspace(3, n_frames - 1, 8))); row = train_off[clip] + t (the STRAT1 convention).  -> dict of arrays ci, t, row."""
    ci, t, row = [], [], []
    for c in (range(len(tab["train_n"])) if clips is None else clips):
        n, off = int(tab["train_n"][c]), int(tab["train_off"][c])
        if n < FOK:
            continue
        for tt in np.unique(np.round(np.linspace(3, n - 1, NW)).astype(int)):
            ci.append(c)
            t.append(int(tt))
            row.append(off + int(tt))
    return {"ci": np.asarray(ci, np.int64), "t": np.asarray(t, np.int64), "row": np.asarray(row, np.int64)}


def fit_labels(tab, rows, shift=0):
    """y [N, 5] int64 in CTX_VARS order at train-table rows ``rows`` (+ ``shift``, the G-CLOCK regression arm): -1 where flag != 0, the variable's valid bit is False, or illumination is twilight (CTX_TWILIGHT_ID).
    Written independently of ``strat1_ctx_read.label_rows`` so that G-CLOCK compares two implementations."""
    r = np.clip(np.asarray(rows, np.int64) + int(shift), 0, len(tab["train_flag"]) - 1)
    fl = np.asarray(tab["train_flag"])[r]
    y = np.full((len(r), len(M.CTX_VARS)), -1, np.int64)
    for k, v in enumerate(M.CTX_VARS):
        ids = np.asarray(tab[f"train_{v}_id"], np.int64)[r]
        ok = np.asarray(tab[f"train_{v}_valid"], bool)[r] & (fl == 0)
        if v == "illumination":
            ok &= ids != M.CTX_TWILIGHT_ID
        y[ok, k] = ids[ok]
    return y


def permute_labels_between_clips(y, ci, seed=0):
    """R1: the rows of train clip c take the labels of clip perm[c] (a seed-0 permutation of the clips), rows by position modulo; all five variables together."""
    uc = np.unique(ci)
    pc = np.random.RandomState(seed).permutation(len(uc))
    rows_of = {int(c): np.where(ci == c)[0] for c in uc}
    out = y.copy()
    for i, c in enumerate(uc):
        dst, src = rows_of[int(c)], rows_of[int(uc[pc[i]])]
        out[dst] = y[src[np.arange(len(dst)) % len(src)]]
    return out


# ------------------------------------------------------------------------------------------------ feature sources
def clip_file(cache_dir, clip):
    return f"{cache_dir}/clips/{clip}.pt"


def read_frames(path, ts, offset=0):
    """[len(ts), 4, 40, 1024] float8 from one clip file (memory-mapped): frames t-3+offset .. t+offset.  ``offset`` is the deliberate-regression arm (a window shift)."""
    d = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    x = d["pool4x10"]
    n = x.shape[0]
    st = [min(max(int(t) - 3 + offset, 0), n - FOK) for t in ts]            # offset 0 never clamps (a window at t >= 3 ends at t <= n - 1); the clamp only keeps the shifted regression arm in range
    return torch.stack([x[a:a + FOK] for a in st]).clone()


def clip_frames(path):
    return int(torch.load(path, map_location="cpu", mmap=True, weights_only=False)["pool4x10"].shape[0])


class StreamSource:
    """Features read per batch from the memory-mapped clip files (P-COMMIT fallback): no feature tensor is held."""
    kind = "stream"

    def __init__(self, cache_dir, clips, rs, offset=0, threads=8):
        self.dir, self.clips, self.rs, self.offset, self.threads = cache_dir, [str(c) for c in clips], rs, offset, threads
        self.n = len(rs["ci"])

    def gather(self, idx):
        idx = np.asarray(idx, np.int64)
        out = torch.empty((len(idx), FOK, 40, 1024), dtype=torch.float8_e4m3fn)
        by = {}
        for k, i in enumerate(idx):
            by.setdefault(int(self.rs["ci"][i]), []).append(k)

        def work(c):
            ks = by[c]
            x = read_frames(clip_file(self.dir, self.clips[c]), [self.rs["t"][idx[k]] for k in ks], self.offset)
            return ks, x
        with ThreadPoolExecutor(self.threads) as ex:
            for ks, x in ex.map(work, list(by)):
                out[torch.as_tensor(ks)] = x
        return out


class RamSource:
    """All features held in one float8 tensor (about 5.9 GB for the full train window set)."""
    kind = "ram"

    def __init__(self, cache_dir, clips, rs, offset=0, threads=8, log=None):
        self.n = len(rs["ci"])
        self.feats = torch.empty((self.n, FOK, 40, 1024), dtype=torch.float8_e4m3fn)
        by = {}
        for i, c in enumerate(rs["ci"]):
            by.setdefault(int(c), []).append(i)

        def work(c):
            ks = by[c]
            return ks, read_frames(clip_file(cache_dir, str(clips[c])), [rs["t"][k] for k in ks], offset)
        t0 = time.time()
        with ThreadPoolExecutor(threads) as ex:
            for j, (ks, x) in enumerate(ex.map(work, list(by))):
                self.feats[torch.as_tensor(ks)] = x
                if log and j % 500 == 0:
                    log(f"  read {j} of {len(by)} clips {time.time() - t0:.0f}s")

    def gather(self, idx):
        return self.feats[torch.as_tensor(np.asarray(idx, np.int64))]


def make_source(kind, cache_dir, clips, rs, offset=0, log=None):
    return (RamSource(cache_dir, clips, rs, offset, log=log) if kind == "ram" else StreamSource(cache_dir, clips, rs, offset))


def drop_mismatching_clips(cache_dir, clips, tab, rs, threads=8):
    """A clip whose file frame count != train_n would shift every row after it: its windows are dropped (the probe's rule) and COUNTED.  -> (rs', n_skipped_clips)"""
    cs = np.unique(rs["ci"])

    def n_of(c):
        return int(c), clip_frames(clip_file(cache_dir, str(clips[c])))
    with ThreadPoolExecutor(threads) as ex:
        bad = {c for c, n in ex.map(n_of, cs) if n != int(tab["train_n"][c])}
    if not bad:
        return rs, 0
    keep = ~np.isin(rs["ci"], list(bad))
    return {k: v[keep] for k, v in rs.items()}, len(bad)


# ------------------------------------------------------------------------------------------------ the model of an arm
def p10_zero():
    return M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30))


def ckpt_sha_problem(tag):
    """-> reason or None: the dev-box ckpt of ``tag`` against its BANK_MANIFEST.json and the card's pin prefix."""
    p = f"{CK}/{tag}/ckpt_last.pt"
    man = json.load(open(f"{CK}/{tag}/BANK_MANIFEST.json", encoding="utf-8"))
    want = man["files"]["ckpt_last.pt"]["sha256"]
    if not want.startswith(CKPT_PREFIX[tag]):
        return f"{tag}: manifest sha {want[:8]} != the card pin {CKPT_PREFIX[tag]}"
    if sha256_file(p) != want:
        return f"{tag}: ckpt bytes differ from the bank manifest"
    return None


def load_pool_model(ck, adapter="s1", adapter_seed=0, cls=None):
    """The frozen driving model (``ck`` = the loaded S1 / C0 checkpoint dict) with the pooled head BUILT: every S1 tensor loaded, the 55 head tensors fresh.  ``adapter``: s1 = as banked;
    random | feat = a fresh LayerNorm(1024) -> Linear(1024, 256) (nn default init), ``pos`` / ``time`` ~ N(0, 0.02^2), init seed ``adapter_seed`` (random = frozen: RA arms; feat = trained with the head: FEAT arms)."""
    cfg = M.V8FMiniConfig(**{**ck["cfg"], "ctx_pool": "on"})
    model = (cls or M.V8FMini)(cfg, p10_zero())
    res = model.load_state_dict(ck["model"], strict=False)
    if res.unexpected_keys or not (len(res.missing_keys) == 55 and all(k.startswith("ctx_pool_head.") for k in res.missing_keys)):
        raise RuntimeError(f"state dict mismatch: missing {res.missing_keys[:3]} unexpected {res.unexpected_keys[:3]}")
    if adapter in ("random", "feat"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(adapter_seed)
            model.adapter = nn.Sequential(nn.LayerNorm(cfg.d_feat), nn.Linear(cfg.d_feat, cfg.d))
            g = torch.Generator().manual_seed(adapter_seed)
            model.pos.data = torch.randn(cfg.n_tok, cfg.d, generator=g) * 0.02
            model.time.data = torch.randn(cfg.hist, cfg.d, generator=g) * 0.02
    elif adapter != "s1":
        raise ValueError(adapter)
    return model


def reinit_head(model, seed):
    """a fresh head under a forked RNG (the global stream is untouched), on the model's device"""
    dev = next(model.parameters()).device
    model.ctx_pool_head = M.build_ctx_pool_heads(model.cfg.d, seed).to(dev)


def head_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith("ctx_pool_head.")}


def save_head_ckpt(path, model, ck, meta):
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = {k: v for k, v in ck.items() if k not in ("model", "opt", "cfg")}
    out["cfg"] = {**ck["cfg"], "ctx_pool": "on"}
    out["model"] = sd
    out["wh"] = meta
    torch.save(out, path)
    return sha256_file(path)


# ------------------------------------------------------------------------------------------------ fit
def window_batch(feats):
    return {"feats": feats, "frame_ok": torch.ones(feats.shape[0], feats.shape[1], dtype=torch.bool, device=feats.device)}


def fit_step(model, feats, y, train_adapter=False):
    """One evaluation of the FIT loss (card 4.2): logits from ``ctx_pool_fit_logits`` (adapter + scene under no_grad), the per-variable masked CE ``T["ctx_pool/<var>"]`` and their sum.  -> (loss, T)"""
    b = window_batch(feats)
    logits = model.ctx_pool_fit_logits(b, train_adapter)
    T = M.ctx_pool_terms(logits, y, b["frame_ok"][:, -1])
    loss = T["ctx_pool"]
    return loss, T


def live_check(model, feats, y, step=None, train_adapter=False):
    """G-LIVE / reachability (card 4.4 item 2) on ONE real batch with requires_grad True on EVERY model parameter: backward of the fit loss must give a non-zero gradient on each of the five sub-heads and EXACTLY 0.0 (or None)
    on every other parameter (the control that proves the probe can read zero).  -> a record with ``pass``."""
    step = step or fit_step
    for p in model.parameters():
        p.requires_grad_(True)
        p.grad = None
    model.eval()
    loss, T = step(model, feats, y, train_adapter)
    loss.backward()
    pool = {v: 0.0 for v in M.CTX_VARS}
    other, nz = 0.0, []
    for name, p in model.named_parameters():
        g = None if p.grad is None else float(p.grad.abs().sum())
        if name.startswith("ctx_pool_head."):
            pool[name.split(".")[1]] += g or 0.0
        elif train_adapter and name.startswith(("adapter.", "pos", "time")):
            continue
        elif g:
            other += g
            nz.append(name)
    for p in model.parameters():
        p.grad = None
    ok = all(v > 0.0 for v in pool.values()) and other == 0.0
    return {"pass": bool(ok), "subhead_grad_abs_sum": pool, "other_grad_abs_sum": other, "other_nonzero_params": nz[:5], "loss": float(loss.detach()), "terms": {k: float(v.detach()) for k, v in T.items()}}


def expected_state_names(model, train_adapter):
    names = {n for n, _ in model.named_parameters() if n.startswith("ctx_pool_head.")}
    if train_adapter:
        names |= {n for n, _ in model.named_parameters() if n.startswith(("adapter.", "pos", "time"))}
    return names


def adam_census(model, opt, train_adapter=False):
    """Card 4.4 item 3 (the logic of probe_zerograd3.py): Adam allocates its state lazily, so a registered parameter WITHOUT an entry never received a gradient.  Optimizer state must exist for exactly the head tensors
    (55) -- plus the adapter, ``pos`` and ``time`` for a FEAT fit -- and for no other parameter."""
    have = {n for n, p in model.named_parameters() if p in opt.state and "exp_avg" in opt.state[p]}
    want = expected_state_names(model, train_adapter)
    n_reg = sum(1 for _ in model.named_parameters())
    return {"pass": have == want, "n_registered": n_reg, "n_with_state": len(have), "n_expected": len(want), "missing": sorted(want - have)[:5], "extra": sorted(have - want)[:5]}


def lr_lambda(steps, warm=WARM):
    return lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, steps - 1))))


def prefetched(src, chunks, ahead=3):
    """yield (chunk, features) in ORDER with the next ``ahead`` gathers running in a background thread (hides the file reads of a streamed source behind the GPU step; the order and the bytes are unchanged)."""
    from collections import deque
    ex = ThreadPoolExecutor(1)
    q = deque()
    try:
        it = iter(chunks)
        for _ in range(ahead):
            c = next(it, None)
            if c is not None:
                q.append((c, ex.submit(src.gather, c)))
        while q:
            c, f = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append((nxt, ex.submit(src.gather, nxt)))
            yield c, f.result()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def fit_head(model, src, y, idx, epochs, seed, dev, train_adapter=False, log=print, bs=BS, step=None):
    """Fit the five sub-heads on the windows ``idx`` for ``epochs`` (AdamW lr 1e-3, wd 0.01, 100 warm-up steps, cosine to 0, batch 256, fp32): the head is re-initialised under a forked RNG seeded ``seed`` and the batch order is
    ``np.random.RandomState(seed)``.  The optimiser is given EVERY model parameter (requires_grad True) so that the Adam-state census is a real test.  -> record with the census and the per-epoch loss."""
    step = step or fit_step
    model.to(dev)
    reinit_head(model, seed)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(list(model.parameters()), lr=LR, weight_decay=WD)
    idx = np.asarray(idx, np.int64)
    n = len(idx)
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps))
    rng = np.random.RandomState(seed)
    yt = torch.as_tensor(y)
    model.eval()
    hist = []
    for ep in range(epochs):
        perm = idx[rng.permutation(n)]
        tot, t0 = 0.0, time.time()
        for ix, feats in prefetched(src, [perm[i:i + bs] for i in range(0, n, bs)]):
            feats = feats.to(dev)
            loss, _ = step(model, feats, yt[torch.as_tensor(ix)].to(dev), train_adapter)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(ix)
        hist.append(tot / n)
        log(f"    ep {ep + 1}/{epochs} loss {tot / n:.4f} ({time.time() - t0:.0f}s)")
    return {"epochs": epochs, "seed": seed, "n_windows": int(n), "steps": int(steps), "loss_per_epoch": hist, "census": adam_census(model, opt, train_adapter)}


@torch.no_grad()
def predict(model, src, idx, dev, bs=BS):
    """ctx_pool logits [n, 41] (numpy) on windows ``idx`` of ``src`` -- the FIT path (``ctx_pool_fit_logits``)."""
    model.eval()
    out = []
    idx = np.asarray(idx, np.int64)
    for i in range(0, len(idx), bs):
        feats = src.gather(idx[i:i + bs]).to(dev)
        out.append(model.ctx_pool_fit_logits(window_batch(feats)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, M.CTX_DIM), np.float32)


def margin_per_var(logits, y):
    """held-out accuracy minus the majority-class accuracy, per variable (rows with a label for that variable); None where no row."""
    out, o = {}, 0
    for k, (v, n) in enumerate(zip(M.CTX_VARS, M.CTX_N)):
        m = y[:, k] >= 0
        if not m.any():
            out[v] = None
        else:
            yy, pp = y[m, k], logits[m][:, o:o + n].argmax(1)
            out[v] = float((pp == yy).mean() - (yy == np.bincount(yy, minlength=n).argmax()).mean())
        o += n
    return out


def heldout_split(ci, seed=0):
    """train-only 80 / 20 split by a seed-0 clip permutation (the probe's rule) -> (fit_idx, held_idx)"""
    uc = np.unique(ci)
    perm = np.random.RandomState(seed).permutation(len(uc))
    held = uc[perm[: len(uc) // 5]]
    m = np.isin(ci, held)
    return np.where(~m)[0], np.where(m)[0]


def pick_epochs(scores, grid=E_GRID):
    """per variable the E of the largest held-out margin, ties to the smaller E; a variable with no scorable held-out row gets the smallest E (recorded)."""
    out = {}
    for v in M.CTX_VARS:
        vals = {E: scores[E][v] for E in grid if scores[E][v] is not None}
        out[v] = min(grid) if not vals else max(grid, key=lambda E: (vals.get(E, -9.0), -E))
    return out


def select_epochs(model, src, y, ci, dev, log=print, grid=E_GRID, seed=0, train_adapter=False, fit=None):
    """Card 4.2: all five sub-heads fitted on the 80 % at each E of ``grid``; E_v = argmax of the held-out (accuracy - majority accuracy).  TRAIN windows only.  -> (E_v, scores, records)"""
    fit = fit or fit_head
    fi, hi = heldout_split(ci)
    scores, recs = {}, {}
    for E in grid:
        log(f"  epoch selection: E={E} on {len(fi)} windows, held-out {len(hi)}")
        recs[E] = fit(model, src, y, fi, E, seed, dev, train_adapter, log)
        scores[E] = margin_per_var(predict(model, src, hi, dev), y[hi])
        log(f"  E={E} held-out margins {scores[E]}")
    return pick_epochs(scores, grid), {str(E): s for E, s in scores.items()}, {str(E): r for E, r in recs.items()}


def fit_final(model, src, y, E_v, seed, dev, train_adapter=False, log=print, fit=None):
    """The final head: fitted on all train windows once at each DISTINCT chosen E; sub-head v is taken from the fit at E_v (legitimate: the sub-heads are independent).  Loads the composed head into ``model``.
    -> {'recs': {E: record}, 'live': ...}"""
    fit = fit or fit_head
    idx = np.arange(len(y))
    subs, recs = {}, {}
    for E in sorted(set(E_v.values())):
        log(f"  final fit at E={E} on {len(idx)} windows (variables: {[v for v in M.CTX_VARS if E_v[v] == E]})")
        recs[E] = fit(model, src, y, idx, E, seed, dev, train_adapter, log)
        subs[E] = head_state(model)
    comp = {}
    for v in M.CTX_VARS:
        comp.update({k: t for k, t in subs[E_v[v]].items() if k.startswith(f"ctx_pool_head.{v}.")})
    model.load_state_dict(comp, strict=False)
    return {"recs": {str(E): r for E, r in recs.items()}}


# ------------------------------------------------------------------------------------------------ identity checks
def iso_a(s1_sd, head_sd, new_prefix="ctx_pool_head."):
    """T-ISO (a): every key of the S1 checkpoint is present in the head checkpoint with a byte-identical tensor (sha256 per tensor); the only new keys start with ``ctx_pool_head.``.  -> problems"""
    bad = []
    for k, v in s1_sd.items():
        if k not in head_sd:
            bad.append(f"{k} missing from the head checkpoint")
        elif tensor_sha(v) != tensor_sha(head_sd[k]):
            bad.append(f"{k} differs")
    bad += [f"new key {k} outside {new_prefix}" for k in head_sd if k not in s1_sd and not k.startswith(new_prefix)]
    return bad


def eval_windows(grid_path, order):
    """(ci, t) of the eval grid in the evaluators' order"""
    g = json.load(open(grid_path, encoding="utf-8"))
    if list(g["order"]) != [str(c) for c in order]:
        raise RuntimeError("eval grid episode order differs from the join's")
    ci, t = [], []
    for i, key in enumerate(g["order"]):
        for tt in g["windows"][key]:
            ci.append(i)
            t.append(int(tt))
    return {"ci": np.asarray(ci, np.int64), "t": np.asarray(t, np.int64)}


def disjointness(tab, grid_path, train_dir, eval_clips):
    """card 4.5: table train vs eval clips; fit clip files vs the 141 eval clips; fit clip files vs the grid's order.  Each must read 0."""
    train_clips = [str(c) for c in tab["train_clip"]]
    g = json.load(open(grid_path, encoding="utf-8"))
    files = {f[:-3] for f in os.listdir(f"{train_dir}/clips") if f.endswith(".pt")}
    return {"table_train_vs_eval": len(set(train_clips) & set(eval_clips)), "train_cache_files_vs_eval": len(files & set(eval_clips)), "train_cache_files_vs_grid_order": len(files & set(g["order"]))}


def cache_provenance(cache_dir):
    """P-CACHE (beside, no bar): the cache's own encoder record where one exists, else 'not recorded'; manifest md5 count."""
    mj = f"{cache_dir}/meta.json"
    rec = {"encoder_module_sha256": "not recorded", "pipeline": "not recorded"}
    if os.path.exists(mj):
        j = json.load(open(mj, encoding="utf-8"))
        rec = {"encoder_module_sha256": j.get("encoder_module_sha256", "not recorded"), "pipeline": j.get("pipeline", "not recorded"), "model": j.get("model")}
    rec["n_manifest_rows"] = sum(1 for _ in open(f"{cache_dir}/manifest.jsonl", encoding="utf-8")) if os.path.exists(f"{cache_dir}/manifest.jsonl") else None
    return rec


def p_cache_match(local_manifest, pod_manifest):
    """content_md5 equal count of the dev-box train manifest against the pod's (a file the MM pulls).  Printed, no bar depends on it.  -> {'n_equal', 'n_local', 'pod_manifest'}"""
    if not pod_manifest or not os.path.exists(pod_manifest):
        return {"n_equal": None, "n_local": None, "pod_manifest": "not provided (a file the MM pulls; no pod job)"}
    loc = {j["clip"]: j["content_md5"] for j in map(json.loads, open(local_manifest, encoding="utf-8"))}
    pod = {j["clip"]: j["content_md5"] for j in map(json.loads, open(pod_manifest, encoding="utf-8"))}
    return {"n_equal": int(sum(loc[c] == pod.get(c) for c in loc)), "n_local": len(loc), "pod_manifest": pod_manifest}


# ------------------------------------------------------------------------------------------------ the shared dev-box GPU lock (ruling 7)
def gpu_used_mib():
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, encoding="utf-8", timeout=30)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:                                                          # noqa: BLE001
        return float("inf")


def gpu_lock_acquire(job, timeout_s=0, poll_s=30, log=print):
    """Take ``devbox_gpu.lock`` (an exclusive create; never overwritten, never forced) and wait for the GPU to read under 3,500 MiB used.  -> True / False (lock held by another job, or the GPU is busy, at the timeout)."""
    t0 = time.time()
    rec = json.dumps({"job": job, "pid": os.getpid(), "host": os.environ.get("COMPUTERNAME", "?"), "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    while True:
        try:
            fd = os.open(GPU_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, rec.encode("utf-8"))
            os.close(fd)
            break
        except FileExistsError:
            log(f"GPU lock held: {open(GPU_LOCK, encoding='utf-8').read()[:200]}")
            if time.time() - t0 >= timeout_s:
                return False
            time.sleep(poll_s)
    while gpu_used_mib() >= 3500:
        if time.time() - t0 >= timeout_s:
            gpu_lock_release(job)
            return False
        time.sleep(poll_s)
    return True


def gpu_lock_release(job):
    """delete the lock iff it is OURS (same job name and pid)"""
    try:
        j = json.load(open(GPU_LOCK, encoding="utf-8"))
        if j.get("job") == job and j.get("pid") == os.getpid():
            os.remove(GPU_LOCK)
            return True
    except Exception:                                                          # noqa: BLE001
        pass
    return False
