"""attn_extract.py -- the instrumented forward of v8F-mini (CPU only, read-only on every input).

WHAT IT DOES
  Wraps `layer.multihead_attn` of every decoder layer of the three Readers (`read_s`, `read_t`, `read_o`) so that the cross-attention
  weights (need_weights=True, average_attn_weights=False -> [B, heads, n_q, S]) are CAPTURED while the forward result is passed through
  unchanged.  Every layer and every head is kept.  The memory of a Reader is  mem = [cond tokens ; scene tokens]  with
      S cond = (ego, nav, lim)                         3 tokens
      T cond = (ego, nav, lim, ck)                     4 tokens
      O cond = (ego, nav, lim, ck, code, age)          6 tokens
      scene  = hist(4) frames x n_tok(40), index  h*40 + tok,  h = 0 OLDEST ... h = 3 = the Reader's tick frame; tok = r*10 + c
  (v8f_mini.py: Reader.forward / V8FMini._scene / run_s / run_t / run_o).  Padded history frames (frame_ok False) receive weight 0.

TOKEN GEOMETRY (measured from the cache builder, not assumed)
  The DINOv3-L/16 patch field of a 256x640 cylindrical frame is 16x40 patches; `pull_encode_pooled.py:259` pools it with
  `reshape(T, gh, 16/gh, gw, 40/gw, D).mean((2, 4))` -> for pool4x10 each token (r, c) is the mean of the 4x4 block of 16-px patches
  rows 4r..4r+3, cols 4c..4c+3, i.e. the PIXEL RECTANGLE y in [64 r, 64 r + 64), x in [64 c, 64 c + 64) of the 256x640 frame; token = r*10 + c.
  Cache step j is native frame 2j of the v2ep episode (`dinov3_fp8_encode_ship.py:117`: `range(0, len(lens), 2)`).

USAGE   (python, never -I: this file puts TREE/stack and the taniteval copy on sys.path itself)
  from attn_extract import *
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SCRATCH = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/viz"
TREE, WORK, OUT = f"{SCRATCH}/tree", f"{SCRATCH}/work", f"{SCRATCH}/out"
for _p in (f"{TREE}/stack", f"{WORK}/taniteval_b2"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

LEVELS = ("S", "T", "O")
READER = {"S": "read_s", "T": "read_t", "O": "read_o"}
COND_NAMES = {"S": ("ego", "nav", "lim"),
              "T": ("ego", "nav", "lim", "ck"),
              "O": ("ego", "nav", "lim", "ck", "code", "age")}
ALL_COND = ("ego", "nav", "lim", "ck", "code", "age")
HIST, NTOK, GH, GW = 4, 40, 4, 10
PKG = f"{TREE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
STRATA_NPZ = f"{PKG}/raw/2026-10-25-kind/window_strata_eval141.npz"
INPUTS = f"{WORK}/inputs"
CACHE = f"{WORK}/cache"
CKPT = {"M2b_s0": f"{WORK}/ckpt/M2b_s0.pt", "M2b_s1": f"{WORK}/ckpt/M2b_s1.pt"}
P10 = f"{WORK}/ckpt/p10_prior.pt"


# ------------------------------------------------------------------------------------------------ model + data
def load_model(name: str):
    ck = torch.load(CKPT[name], map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    return model.eval(), cfg


class EvalWorld:
    """The eval's own loader (v8f_mini_eval.cmd_run, lines 233-300): same grid, join, X, ClipStore and window dataset.
    Window i of the 2,399 is (ci, t, a_t, a_s) with `--ages cycle`:  a_t = i % 3,  a_s = a_t + (i // 3) % 5  (then clamped by `item`)."""

    def __init__(self, cfg, lru: int = 8, cache: str | None = None):
        cache = cache or CACHE
        self.grid = json.load(open(f"{INPUTS}/eval_grid_windows.json"))
        self.J = D.load_join(f"{INPUTS}/eval_join.npz") if os.path.exists(f"{INPUTS}/eval_join.npz") else D.load_join(f"{INPUTS}/join_eval141.npz")
        self.X = D.eval_context(self.grid, f"{INPUTS}/eval141_kin.npz", self.J, kin_mode=cfg.kin_mode)
        self.tracks = D.load_native_tracks(f"{INPUTS}/native_tracks.pt")
        self.store = D.ClipStore(cache, "eval", lru=lru, grid="pool4x10", kin_mode=cfg.kin_mode, tracks=self.tracks)
        self.wins = D.eval_grid_windows(f"{INPUTS}/eval_grid_windows.json", self.J)
        self.ds = D.V8FMiniWindows(self.store, self.J, self.wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t,
                                   max_age_s=cfg.max_age_s, window_start=cfg.window_start)
        self.clips = list(self.J["clip"].tolist())
        self.N = len(self.wins)
        self.win_clip = np.array([self.clips[w[0]] for w in self.wins])

    def item(self, i: int):
        ci, t, _, _ = self.wins[i]
        a_t = i % 3
        a_s = a_t + (i // 3) % 5
        return self.ds.item(ci, t, a_t, a_s)

    def batch(self, idx):
        return D.collate([self.item(int(i)) for i in idx])

    def clip_windows(self, clip: str):
        return np.nonzero(self.win_clip == clip)[0]


# ------------------------------------------------------------------------------------------------ the tap
def explicit_attn_weights(mha, query, key, key_padding_mask):
    """The cross-attention weights [B, H, n_q, S] recomputed from the module's own projections (nn.MultiheadAttention, batch_first, packed in_proj):
    softmax(Q K^T / sqrt(dh) + padding).  Used so the REAL forward runs untouched (need_weights=False, the SDPA path)."""
    d, H = mha.embed_dim, mha.num_heads
    dh = d // H
    W, bias = mha.in_proj_weight, mha.in_proj_bias
    q = torch.nn.functional.linear(query, W[:d], bias[:d])
    k = torch.nn.functional.linear(key, W[d:2 * d], bias[d:2 * d])
    B, nq, S = q.shape[0], q.shape[1], k.shape[1]
    q = q.reshape(B, nq, H, dh).transpose(1, 2)
    k = k.reshape(B, S, H, dh).transpose(1, 2)
    sc = (q @ k.transpose(-1, -2)) / (dh ** 0.5)
    if key_padding_mask is not None:
        sc = sc.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    return torch.softmax(sc, dim=-1)


class AttnTap:
    """Context manager. `tap.w[(level, layer)]` = float32 [B, heads, n_q, S] after a forward.
    mode='exact' (DEFAULT): the original `multihead_attn.forward` is called UNCHANGED (need_weights=False -> the forward result is bit-identical to the
                            unhooked model) and the weights are recomputed beside it from the same query/key/projection (`explicit_attn_weights`).
    mode='need_weights':    the original is called with need_weights=True, average_attn_weights=False (torch's own weights; a different kernel, so
                            the forward differs from the unhooked one by float32 round-off, <= 1 ulp)."""

    def __init__(self, model, mode: str = "exact"):
        assert mode in ("exact", "need_weights")
        self.model, self.mode, self.w, self._orig, self.on = model, mode, {}, [], True

    def __enter__(self):
        for lv in LEVELS:
            rd = getattr(self.model, READER[lv])
            for li, layer in enumerate(rd.dec.layers):
                mha = layer.multihead_attn
                orig = mha.forward

                def wrapped(*a, _orig=orig, _key=(lv, li), _mha=mha, **k):
                    if not self.on:
                        return _orig(*a, **k)
                    if self.mode == "exact":
                        out = _orig(*a, **k)
                        self.w[_key] = explicit_attn_weights(_mha, a[0], a[1], k.get("key_padding_mask")).detach().float()
                        return out
                    k["need_weights"], k["average_attn_weights"] = True, False
                    out, wts = _orig(*a, **k)
                    self.w[_key] = wts.detach().float()
                    return out, wts

                mha.forward = wrapped
                self._orig.append((mha, orig))
        return self

    def __exit__(self, *e):
        for mha, orig in self._orig:
            mha.forward = orig
        self._orig = []


def forward_tapped(model, batch, code_override=None, mode: str = "exact"):
    """Hooked inference forward. Returns (out, {(level, layer): [B, H, Q, S]})."""
    with torch.no_grad(), AttnTap(model, mode) as tap:
        out = model.forward_infer(batch) if code_override is None else model.forward_diag(batch, code_override)
    return out, dict(tap.w)


def forward_plain(model, batch, code_override=None):
    with torch.no_grad():
        return model.forward_infer(batch) if code_override is None else model.forward_diag(batch, code_override)


# ------------------------------------------------------------------------------------------------ decomposition of a weight tensor
def split_weights(w: torch.Tensor, level: str, agg: str = "mean_all"):
    """w [B, H, Q, S] of one (level, layer) -> dict(cond [B, n_cond], scene [B, HIST, NTOK]) averaged over heads and queries.
    The row sums to 1 over cond + scene (padded frames carry 0)."""
    nc = len(COND_NAMES[level])
    a = w.mean(dim=(1, 2))                                    # [B, S]  mean over heads and queries
    assert a.shape[1] == nc + HIST * NTOK, (a.shape, level)
    return {"cond": a[:, :nc], "scene": a[:, nc:].reshape(-1, HIST, NTOK)}


def level_maps(attn: dict, level: str, layers: str = "mean"):
    """Aggregate over layers. layers='mean' (the PLOTTED aggregation: mean over layers, heads and queries) | 'last' | int."""
    keys = sorted(k for k in attn if k[0] == level)
    if layers == "mean":
        ws = [split_weights(attn[k], level) for k in keys]
        return {"cond": torch.stack([x["cond"] for x in ws]).mean(0), "scene": torch.stack([x["scene"] for x in ws]).mean(0)}
    if layers == "last":
        return split_weights(attn[keys[-1]], level)
    return split_weights(attn[keys[int(layers)]], level)


def token_rect(tok: int):
    """pixel rectangle (x0, y0, x1, y1) in the 256x640 frame of pool4x10 token `tok` (row-major r*10 + c)."""
    r, c = divmod(int(tok), GW)
    return (64 * c, 64 * r, 64 * c + 64, 64 * r + 64)


# ------------------------------------------------------------------------------------------------ per-window ADE (the eval's own formula)
def ade6_of(traj30: np.ndarray, gt: np.ndarray):
    """traj_metrics: d = ||pred - gt||, ade6 = mean over the 30 steps (v8f_mini_eval.py:67-69)."""
    return np.linalg.norm(traj30.astype(np.float64) - gt, axis=-1).mean(1)
