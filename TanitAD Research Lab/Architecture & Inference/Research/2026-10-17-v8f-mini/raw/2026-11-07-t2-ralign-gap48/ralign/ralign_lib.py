"""RALIGN library: one definition of the fold rule, the pair loader, the per-cell OLS and the map file (PREREG_T2_RALIGN.md, sha256 a6c84433...). CPU only, torch + numpy.

Shared by `ralign_fit.py`, `ralign_gate.py`, `ralign_policy.py` (shipped to box 2 in the overlay) and the tests, so the fold rule and the map arithmetic exist in ONE place.

Card sections used here:
  2  folds: scene s -> fold A if the first hex digit of sha256(scene id) is even, else B (`fold_of`; `digit=1` is the replicate of section 6.1). The map applied to a scene is the OTHER fold's.
  3  the map: per (position p, channel c) an affine x^[p,c] = a[p,c] * r[p,c] + b[p,c], OLS on the fold's (render r, real x) pairs. A cell with fewer than 200 pairs -> NOT EVALUABLE.
Conventions the card leaves open (each recorded in the fit result, none changes the family):
  * a cell whose render tokens have zero variance (OLS undefined) gets a := 0, b := mean(x) (the intercept-only OLS) and is COUNTED in `n_degenerate`;
  * non-finite pairs are dropped per cell before counting;
  * the map output is rounded to fp16 (the dtype the policy stores its scene tokens in, `V8FMiniT2Policy.push_tokens`); an output that is not finite in fp16 raises RALIGN_NONFINITE.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

CARD_SHA256 = "a6c84433f19285c39cb92503273851fed78858c2ff9db63ea5c75d84d49eb306"
SHAPE = (40, 1024)
MIN_PAIRS = 200
RISE_MIN, LEVEL_MIN = 0.05, 0.90
FOLDS = ("A", "B")
DEGENERATE_REL = 1e-12


def sha256_file(path, n: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------ folds
def fold_of(scene_id: str, digit: int = 0) -> str:
    """Card section 2: fold A if hex digit `digit` (0 = the first; 1 = the replicate of section 6.1) of sha256(utf-8 scene id) is even, B if odd."""
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError(f"RALIGN_BAD_SCENE_ID {scene_id!r}")
    if digit not in (0, 1):
        raise ValueError(f"RALIGN_BAD_FOLD_DIGIT {digit!r}: 0 (the card) or 1 (the section 6.1 replicate)")
    return "A" if int(hashlib.sha256(scene_id.encode("utf-8")).hexdigest()[digit], 16) % 2 == 0 else "B"


def other_fold(fold: str) -> str:
    if fold not in FOLDS:
        raise ValueError(f"RALIGN_BAD_FOLD {fold!r}")
    return "B" if fold == "A" else "A"


# ------------------------------------------------------------------------------------------------------------------------------ pair files (GAP48 format, configurable)
def load_pair_file(path, scene_key="scene", render_key="render", real_key="real", shape=SHAPE):
    """One GAP48 token file (`tokens/<scene8>.pt`, a torch dict) -> (scene id, render [n,P,C] fp32, real [n,P,C] fp32). The key names are arguments because the GAP48 format is read
    from its builder's code (`t2v8f_gap48.py`: keys scene, ticks, cam_frames, render, real, cosine, rel_l2, grad_ncc_pixels, token_dtype), not from a file. Refuses by name."""
    o = torch.load(path, map_location="cpu", weights_only=True)
    for k in (scene_key, render_key, real_key):
        if k not in o:
            raise ValueError(f"RALIGN_PAIR_KEY_MISSING {k!r} in {os.path.basename(str(path))} (keys: {sorted(o)})")
    sid, R, X = o[scene_key], o[render_key], o[real_key]
    if not isinstance(sid, str) or not sid:
        raise ValueError(f"RALIGN_PAIR_SCENE_ID in {os.path.basename(str(path))}: {sid!r}")
    if not (torch.is_tensor(R) and torch.is_tensor(X)) or R.shape != X.shape or R.ndim != 3 or tuple(R.shape[1:]) != tuple(shape) or R.shape[0] < 1:
        raise ValueError(f"RALIGN_PAIR_SHAPE in {os.path.basename(str(path))}: render {tuple(getattr(R, 'shape', ()))} real {tuple(getattr(X, 'shape', ()))}, want [n, {shape[0]}, {shape[1]}]")
    return sid, R.float(), X.float()


# ------------------------------------------------------------------------------------------------------------------------------ OLS
class Stats:
    """Sufficient statistics of the per-cell OLS of x on r, float64, accumulated scene by scene (memory = 6 x [P, C])."""

    def __init__(self, shape=SHAPE):
        z = lambda: torch.zeros(shape, dtype=torch.float64)      # noqa: E731
        self.n, self.sr, self.sx, self.srr, self.srx = z(), z(), z(), z(), z()
        self.n_pairs = 0

    def add(self, R: torch.Tensor, X: torch.Tensor):
        """R, X: [n, P, C]. Per cell, a pair with a non-finite r or x is dropped."""
        ok = torch.isfinite(R) & torch.isfinite(X)
        r, x = torch.where(ok, R, torch.zeros_like(R)).double(), torch.where(ok, X, torch.zeros_like(X)).double()
        self.n += ok.sum(0).double()
        self.sr += r.sum(0)
        self.sx += x.sum(0)
        self.srr += (r * r).sum(0)
        self.srx += (r * x).sum(0)
        self.n_pairs += int(R.shape[0])


def solve(st: Stats, min_pairs: int = MIN_PAIRS):
    """-> (a, b float32 [P, C], info). Raises ValueError('RALIGN_NOT_EVALUABLE ...') when any cell has fewer than `min_pairs` pairs."""
    n = st.n
    low = int((n < min_pairs).sum())
    if low:
        raise ValueError(f"RALIGN_NOT_EVALUABLE {low} of {n.numel()} cells have fewer than {min_pairs} pairs (min count {int(n.min())}, pairs read {st.n_pairs})")
    mr, mx = st.sr / n, st.sx / n
    var = st.srr - st.sr * st.sr / n
    cov = st.srx - st.sr * st.sx / n
    deg = var <= DEGENERATE_REL * st.srr.clamp_min(1e-300)
    a = torch.where(deg, torch.zeros_like(var), cov / torch.where(deg, torch.ones_like(var), var))
    b = mx - a * mr
    info = {"n_cells": int(n.numel()), "min_count": int(n.min()), "max_count": int(n.max()), "n_pairs": st.n_pairs, "n_degenerate": int(deg.sum()),
            "a_min": float(a.min()), "a_max": float(a.max()), "a_median": float(a.median()), "b_abs_max": float(b.abs().max())}
    return a.float().contiguous(), b.float().contiguous(), info


# ------------------------------------------------------------------------------------------------------------------------------ the map file
def tensor_sha256(a: torch.Tensor, b: torch.Tensor) -> str:
    """Digest of the map's CONTENT (dtype, shape, bytes of a then b), independent of the container."""
    h = hashlib.sha256()
    for t in (a, b):
        t = t.detach().cpu().contiguous()
        assert t.dtype == torch.float32
        h.update(f"{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def save_map(path, a, b, meta: dict) -> dict:
    """Writes {'a', 'b' fp32 [P, C], 'meta'} and returns {'file_sha256', 'tensor_sha256'}."""
    a, b = a.float().contiguous(), b.float().contiguous()
    torch.save({"a": a, "b": b, "meta": meta}, path)
    return {"file_sha256": sha256_file(path), "tensor_sha256": tensor_sha256(a, b)}


class TokenMap:
    """x^ = a * r + b, per cell, fp32 arithmetic, fp16 output. `apply` is the ONE function the T2 policy and the gate both call."""

    def __init__(self, a, b, meta=None, tensor_sha=None, file_sha=None):
        a, b = torch.as_tensor(a), torch.as_tensor(b)
        if a.dtype != torch.float32 or b.dtype != torch.float32 or a.shape != b.shape or a.ndim != 2:
            raise ValueError(f"RALIGN_MAP_FORMAT a {a.dtype} {tuple(a.shape)}, b {b.dtype} {tuple(b.shape)}: want two fp32 [P, C] tensors")
        self.a, self.b, self.meta = a.contiguous(), b.contiguous(), dict(meta or {})
        self.tensor_sha256 = tensor_sha or tensor_sha256(self.a, self.b)
        self.file_sha256 = file_sha
        self.fold = self.meta.get("fold")
        self.n_applied = 0

    def apply(self, tok) -> torch.Tensor:
        """Scene tokens [..., P, C] (fp8 or fp16 or fp32) -> fp16 of a * r + b (fp32 arithmetic). A signed zero survives a := 1, b := -0.0 (see `identity_pair`)."""
        x = torch.as_tensor(tok).to(torch.float32)
        if tuple(x.shape[-2:]) != tuple(self.a.shape):
            raise ValueError(f"RALIGN_TOKEN_SHAPE {tuple(x.shape)} against a map of {tuple(self.a.shape)}")
        y = (self.a * x + self.b).to(torch.float16)
        if not bool(torch.isfinite(y).all()):
            raise RuntimeError("RALIGN_NONFINITE: the mapped tokens are not finite in fp16 (a cell with a very large slope?)")
        self.n_applied += 1
        return y

    @property
    def is_identity(self) -> bool:
        return bool(torch.all(self.a == 1.0)) and bool(torch.all(self.b == 0.0))


def load_map(path, expect_file_sha=None, expect_tensor_sha=None, shape=SHAPE) -> TokenMap:
    """Reads a map file, re-hashes it (by CONTENT) and refuses by name on a differing byte."""
    fs = sha256_file(path)
    if expect_file_sha and fs != expect_file_sha:
        raise ValueError(f"RALIGN_MAP_HASH_MISMATCH {os.path.basename(str(path))}: file sha256 {fs}, expected {expect_file_sha}")
    o = torch.load(path, map_location="cpu", weights_only=True)
    if set(o) != {"a", "b", "meta"}:
        raise ValueError(f"RALIGN_MAP_FORMAT {os.path.basename(str(path))}: keys {sorted(o)}")
    m = TokenMap(o["a"], o["b"], o["meta"], file_sha=fs)
    if tuple(m.a.shape) != tuple(shape):
        raise ValueError(f"RALIGN_MAP_SHAPE {tuple(m.a.shape)}, expected {tuple(shape)}")
    if expect_tensor_sha and m.tensor_sha256 != expect_tensor_sha:
        raise ValueError(f"RALIGN_MAP_TENSOR_MISMATCH {os.path.basename(str(path))}: {m.tensor_sha256}, expected {expect_tensor_sha}")
    if not (bool(torch.isfinite(m.a).all()) and bool(torch.isfinite(m.b).all())):
        raise ValueError(f"RALIGN_MAP_NONFINITE {os.path.basename(str(path))}")
    return m


def identity_pair(shape=SHAPE):
    """a := 1, b := 0 for the G-IDMAP gate. b is written as -0.0: x + (-0.0) == x for EVERY x including -0.0 (a +0.0 would turn a stored -0.0 token into +0.0, a bit change
    that a bit-for-bit gate would see although the value is equal)."""
    return torch.ones(shape, dtype=torch.float32), torch.full(shape, -0.0, dtype=torch.float32)


def jdump(o, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(o, f, indent=1, sort_keys=False)
        f.write("\n")


def percentile_summary(x) -> dict:
    x = np.asarray(x, np.float64)
    return {"n": int(x.size), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)), "min": float(x.min()), "max": float(x.max())}
