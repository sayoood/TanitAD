#!/usr/bin/env python3
"""L2 Step B, piece 1: P2's scorer with the EP head row REPLACED by a given L2 head file.  Everything else (trunk context, set decoder, the other five head rows) stays P2's.

WHAT IS REPLACED (MEASURED from the checkpoint, l2lib.py docstring): P2's scorer ends in `score_head = nn.Linear(256, 6)`; row 2 is the EP head (component order NC, DAC, EP, TTC, C, DDC).  A head file is
the full `[6, 256]` weight + `[6]` bias the L2 fit wrote (`final_head_s{0,1}.npz`); ONLY row 2 is taken from it, and the file's five other rows must equal the checkpoint's bit for bit or the load is REFUSED
(a head file that changed another head is a different arm).

THE PATH IS L1'S, NOT A COPY: the scoring closure is `real_scorer.make_score_fn` (Step A's own step 6), with `real_scorer.load_scorer` wrapped for the duration of one call so that the head is installed
into the very `ns` that closure scores with.  Nothing of L1 is re-implemented; the only difference between this scorer and Step A's is the EP row.

FAILURE MODES THIS FILE REFUSES (each is a test in test_stepB.py and a check in g_reach_stepb.py):
  * the file's sha256 is not the expected one                      -> HeadRefused   (weights + sha256 checked at load)
  * wrong shape / dtype / non-finite values                        -> HeadRefused
  * the file's non-EP rows differ from the checkpoint's            -> HeadRefused
  * the install did not take (read-back of the installed row)      -> HeadRefused
Lines tagged `@INSTALL` / `@READBACK` are the ones g_reach_stepb.py's mutation arms delete (the "built, tested, unreachable from its caller" class, CLAUDE.md): with `@INSTALL` gone the read-back
raises; with BOTH gone only the CALLER-LEVEL observation (the EP logits the search actually receives) can see that the head never arrived.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
from dataclasses import dataclass, field

import numpy as np

EP = 2                       # row of the EP head in score_head
N_ROWS = 6
D_TOKEN = 256                # width of the score_head input token


class HeadRefused(RuntimeError):
    """A head file was refused at load.  Never caught and continued: a refused head means NO search."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


@dataclass(frozen=True)
class Head:
    """a verified L2 head file: weight [6,256] f32, bias [6] f32, and the sha256 of the file's bytes"""
    W: np.ndarray
    b: np.ndarray
    sha256: str
    path: str
    meta: dict = field(default_factory=dict)


def read_head_file(path: str, expected_sha256: str) -> Head:
    """Read a head file.  `expected_sha256` is REQUIRED (no default, no 'None means skip'): the file's byte sha256 must equal it, else HeadRefused."""
    if not expected_sha256 or len(str(expected_sha256)) != 64:
        raise HeadRefused(f"an expected sha256 (64 hex chars) is required to load {path!r}; got {expected_sha256!r}")
    if not os.path.isfile(path):
        raise HeadRefused(f"head file not found: {path!r}")
    got = sha256_file(path)
    if got != str(expected_sha256).lower():
        raise HeadRefused(f"head file sha256 mismatch for {path!r}: file {got} != expected {expected_sha256}")
    z = np.load(path, allow_pickle=False)
    if "weight" not in z.files or "bias" not in z.files:
        raise HeadRefused(f"{path!r} has no weight/bias arrays (has {z.files})")
    W, b = np.asarray(z["weight"]), np.asarray(z["bias"])
    if W.shape != (N_ROWS, D_TOKEN) or b.shape != (N_ROWS,):
        raise HeadRefused(f"head shapes {W.shape} / {b.shape} != ({N_ROWS}, {D_TOKEN}) / ({N_ROWS},)")
    if W.dtype != np.float32 or b.dtype != np.float32:
        raise HeadRefused(f"head dtypes {W.dtype} / {b.dtype} are not float32")
    if not (np.isfinite(W).all() and np.isfinite(b).all()):
        raise HeadRefused("head file has non-finite values")
    meta = {k: (z[k].item() if z[k].shape == () else None) for k in z.files if k not in ("weight", "bias")}
    return Head(W=W.copy(), b=b.copy(), sha256=got, path=os.path.abspath(path), meta=meta)


def install_ep_head(ns, head: Head) -> dict:
    """Replace row EP of `ns.score_head` with the head file's, verify, and return the provenance record.  Every other row stays the checkpoint's; a head file that differs from it elsewhere is refused."""
    import torch
    lin = ns.score_head
    W0 = lin.weight.detach().clone()
    b0 = lin.bias.detach().clone()
    if tuple(W0.shape) != head.W.shape:
        raise HeadRefused(f"the scorer's score_head weight {tuple(W0.shape)} != the head file's {head.W.shape}")
    others = [r for r in range(N_ROWS) if r != EP]
    if not (np.array_equal(W0.numpy()[others], head.W[others]) and np.array_equal(b0.numpy()[others], head.b[others])):
        raise HeadRefused("the head file's non-EP rows differ from the checkpoint's: only the EP row may change")
    wep = torch.from_numpy(np.array(head.W[EP], copy=True))
    bep = torch.from_numpy(np.array(head.b[EP:EP + 1], copy=True))[0]
    with torch.no_grad():
        lin.weight[EP] = wep                                                                       # @INSTALL
        lin.bias[EP] = bep                                                                         # @INSTALL
    if not (torch.equal(lin.weight[EP].detach(), wep) and torch.equal(lin.bias[EP].detach(), bep)): raise HeadRefused("install read-back failed: the scorer's EP row is not the head file's")  # @READBACK
    W1, b1 = lin.weight.detach(), lin.bias.detach()
    return {"head_path": head.path, "head_sha256": head.sha256,
            "installed_ep_row_sha256": sha256_bytes(W1[EP].numpy().tobytes() + b1[EP:EP + 1].numpy().tobytes()),
            "p2_ep_row_sha256": sha256_bytes(W0[EP].numpy().tobytes() + b0[EP:EP + 1].numpy().tobytes()),
            "ep_row_equals_p2": bool(torch.equal(W1[EP], W0[EP]) and torch.equal(b1[EP], b0[EP])),
            "ep_row_l2_delta_vs_p2": float(torch.linalg.vector_norm(W1[EP] - W0[EP])),
            "ep_bias_delta_vs_p2": float(b1[EP] - b0[EP]),
            "other_rows_equal_p2": bool(torch.equal(W1[others], W0[others]) and torch.equal(b1[others], b0[others]))}


@contextlib.contextmanager
def _scorer_with_head(scorer_mod, head: Head, sink: dict):
    """for the duration of one `make_score_fn` call, `scorer_mod.load_scorer` returns the SAME `ns` with the head installed (and the provenance goes to `sink`)"""
    orig = scorer_mod.load_scorer

    def patched(*a, **k):
        ns, info = orig(*a, **k)
        rec = install_ep_head(ns, head)
        sink.update(rec)
        info.update(rec)
        return ns, info

    scorer_mod.load_scorer = patched
    try:
        yield
    finally:
        scorer_mod.load_scorer = orig


def build_score_fn(scorer_mod, ckpt: str, ctx_path: str, tokens, head: Head | None, refe_dir: str | None = None, device: str = "cpu", autocast: str = "none", ctx_shape=None):
    """`score_fn(set [K,20,3]) -> logits [K,6]` for the token last passed to `.bind(n)`.  `head=None` is P2 AS IS (the reference); a Head is P2 with its EP row replaced.
    `.info` carries the provenance of what was built, including the installed row's sha256."""
    sink: dict = {}
    if head is None:
        fn = scorer_mod.make_score_fn(ckpt, ctx_path, tokens, device=device, refe_dir=refe_dir, autocast=autocast, ctx_shape=ctx_shape)
        fn.info = {**fn.info, "head": None}
        return fn
    with _scorer_with_head(scorer_mod, head, sink):
        fn = scorer_mod.make_score_fn(ckpt, ctx_path, tokens, device=device, refe_dir=refe_dir, autocast=autocast, ctx_shape=ctx_shape)
    if not sink:
        raise HeadRefused("the head installer never ran (load_scorer was not reached through make_score_fn): refusing a scorer that may be plain P2")
    fn.info = {**fn.info, "head": sink}
    return fn
