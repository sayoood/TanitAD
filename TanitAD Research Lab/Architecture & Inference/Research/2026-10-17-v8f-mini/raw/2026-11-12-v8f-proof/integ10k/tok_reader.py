#!/usr/bin/env python
"""tok_reader.py -- card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE), part 2: the lead reader LR16 (the token probe of card ADD43, grid pool4x10, per-token width 16,
five-frame causal history) read from cached encoder tokens ONLY, and the held-out rule of the card.

Reader forward (the TokenProbe of tok_probe_gr.py, written here without the tree so that the rollout wrapper needs no model code of the probe's tree):
  x = layer_norm(tokens[..., 1024]) -> Linear(1024,128) -> GELU -> Linear(128,W) per token -> flatten (40*W per frame)
  -> head.frame Linear(40*W, 64) per frame -> concat of the 5 frame embeddings -> head.mlp Linear(5*64,128) -> GELU -> Linear(128,3)
  outputs (presence logit, gap/50 m, closing/10 m/s). Presence = sigmoid(logit), gap = 50 * o1 (metres). The closing speed is not used by this card.

Held-out rule (card section 2): a clip is held out if int(first 8 hex digits of sha256(clip id), 16) is divisible by 10; map FIT clips are those divisible by 20, map CHECK clips the other held-out ones.

  python tok_reader.py extract --reader CKPT --cache CACHE_POD --join join_train.npz --out OUT.npz [--device cpu]
Tier T0 for every number of this module (a read on logged tokens); never driving performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HIST, N_TOK, D_FEAT = 5, 40, 1024
GAP_SCALE_M = 50.0
GAP_MAX_M = 60.0


# ---------------------------------------------------------------- the held-out rule
def clip_hash_value(clip_id):
    return int(hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest()[:8], 16)


def is_heldout(clip_id):
    return clip_hash_value(clip_id) % 10 == 0


def is_fit(clip_id):
    """map fit clips: held out AND hash value divisible by 20 (divisible by 20 implies by 10)."""
    return clip_hash_value(clip_id) % 20 == 0


def is_check(clip_id):
    return is_heldout(clip_id) and not is_fit(clip_id)


def split_clips(clip_ids):
    ids = [str(c) for c in clip_ids]
    return {"train": [c for c in ids if not is_heldout(c)], "fit": [c for c in ids if is_fit(c)], "check": [c for c in ids if is_check(c)]}


# ---------------------------------------------------------------- the reader
def _erf(x):
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


def _gelu(x):
    try:
        import torch
        return torch.nn.functional.gelu(torch.from_numpy(np.ascontiguousarray(x, np.float32))).numpy()
    except Exception:
        return (0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))).astype(np.float32)


def _ln(x):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5)


class TokReader:
    """weights: dict of float32 numpy arrays with the keys of the probe's state_dict: tok.0.weight tok.0.bias tok.2.weight tok.2.bias head.frame.* head.mlp.0.* head.mlp.2.*"""
    KEYS = ("tok.0.weight", "tok.0.bias", "tok.2.weight", "tok.2.bias", "head.frame.weight", "head.frame.bias", "head.mlp.0.weight", "head.mlp.0.bias", "head.mlp.2.weight", "head.mlp.2.bias")

    def __init__(self, weights, hist=HIST, n_tok=N_TOK, meta=None):
        for k in self.KEYS:
            if k not in weights:
                raise SystemExit("REFUSED: reader weights lack " + k)
        self.w = {k: np.asarray(weights[k], np.float32) for k in self.KEYS}
        self.hist, self.n_tok, self.meta = int(hist), int(n_tok), dict(meta or {})
        self.W = self.w["tok.2.weight"].shape[0]
        if self.w["head.frame.weight"].shape[1] != self.n_tok * self.W or self.w["head.mlp.0.weight"].shape[1] != self.hist * self.w["head.frame.weight"].shape[0]:
            raise SystemExit("REFUSED: reader weight shapes do not fit n_tok %d, W %d, hist %d" % (self.n_tok, self.W, self.hist))

    @classmethod
    def from_ckpt(cls, path):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("kind") != "tokprobe":
            raise SystemExit("REFUSED: %s is not a tokprobe checkpoint" % path)
        if ck.get("input_norm") != "ln" or ck.get("grid", "pool4x10") != "pool4x10":
            raise SystemExit("REFUSED: the reader of this card is grid pool4x10 with input norm ln")
        sd = {k: v.detach().cpu().numpy() for k, v in ck["state_dict"].items()}
        return cls(sd, hist=ck["hist"], n_tok=ck["n_tok"], meta={"arm": ck.get("arm"), "step": ck.get("step"), "W": ck.get("W"), "seed": ck.get("seed")})

    def frame_embed(self, tokens):
        """tokens [..., n_tok, 1024] -> frame embeddings [..., 64] (per-frame part of the head)."""
        x = _ln(np.asarray(tokens, np.float32))
        h = _gelu(x @ self.w["tok.0.weight"].T + self.w["tok.0.bias"])
        z = h @ self.w["tok.2.weight"].T + self.w["tok.2.bias"]
        z = z.reshape(z.shape[:-2] + (self.n_tok * self.W,))
        return z @ self.w["head.frame.weight"].T + self.w["head.frame.bias"]

    def head_from_embeds(self, E):
        """E [B, hist, 64] -> (presence prob [B], gap_m [B])."""
        w = E.reshape(E.shape[0], -1)
        h = _gelu(w @ self.w["head.mlp.0.weight"].T + self.w["head.mlp.0.bias"])
        o = h @ self.w["head.mlp.2.weight"].T + self.w["head.mlp.2.bias"]
        return (1.0 / (1.0 + np.exp(-o[:, 0].astype(np.float64)))).astype(np.float32), (o[:, 1] * GAP_SCALE_M).astype(np.float32)

    def read(self, tokens):
        """tokens [B, hist, n_tok, 1024] (cache steps j-4 .. j, oldest first; numpy or torch) -> (presence prob [B], gap_m [B]). Reads nothing but the tokens."""
        if hasattr(tokens, "detach"):
            tokens = tokens.detach().float().cpu().numpy()
        t = np.asarray(tokens, np.float32)
        if t.ndim != 4 or t.shape[1] != self.hist or t.shape[2] != self.n_tok or t.shape[3] != D_FEAT:
            raise SystemExit("REFUSED: reader input %s, expected [B, %d, %d, %d]" % (list(t.shape), self.hist, self.n_tok, D_FEAT))
        return self.head_from_embeds(self.frame_embed(t))


def read_clip(reader, tokens, chunk=256):
    """All cache steps of one clip: tokens [T, n_tok, 1024] -> p [T], gap_m [T], ok [T] (a full history of 5 real frames t-4..t exists, so ok is False for t < 4).
    The reading at step t uses frames t-4..t only."""
    T = int(tokens.shape[0])
    E = np.empty((T, reader.w["head.frame.weight"].shape[0]), np.float32)
    for i in range(0, T, chunk):
        E[i:i + chunk] = reader.frame_embed(np.asarray(tokens[i:i + chunk], np.float32))
    p = np.full(T, np.nan, np.float32)
    g = np.full(T, np.nan, np.float32)
    ok = np.zeros(T, bool)
    if T >= reader.hist:
        idx = np.arange(reader.hist - 1, T)
        win = np.stack([E[idx - (reader.hist - 1 - k)] for k in range(reader.hist)], 1)
        p[idx], g[idx] = reader.head_from_embeds(win)
        ok[idx] = True
    return p, g, ok


# ---------------------------------------------------------------- extract (held-out clips of the corpus training cache)
def label_rows(join, ci, t):
    """label presence / gap at cache step t of clip row ci of the join (rule of LabelSource: valid == 1, finite, 0 <= gap <= 60)."""
    off, n = np.asarray(join["off"], np.int64), np.asarray(join["n"], np.int64)
    r = off[ci] + t
    inb = (t >= 0) & (t < n[ci])
    gap = np.asarray(join["lead_gap"], np.float64)
    val = np.asarray(join["lead_valid"])
    g = np.where(inb, gap[np.where(inb, r, 0)], np.nan)
    pres = inb & (val[np.where(inb, r, 0)] == 1) & np.isfinite(g) & (g >= 0) & (g <= GAP_MAX_M)
    return pres, np.where(pres, g, np.nan)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_extract(a):
    import torch
    reader = TokReader.from_ckpt(a.reader)
    J = np.load(a.join, allow_pickle=True)
    jclip = [str(c) for c in J["clip"]]
    row = {c: i for i, c in enumerate(jclip)}
    man = [json.loads(l)["clip"] for l in open(os.path.join(a.cache, "train", "manifest.jsonl"), encoding="utf-8") if l.strip()]
    held = [c for c in man if is_heldout(c)]
    cols = {k: [] for k in ("ci", "t", "p", "g", "lab_pres", "lab_gap", "fit")}
    for n, c in enumerate(held):
        d = torch.load(os.path.join(a.cache, "train", "clips", c + ".pt"), map_location="cpu")
        x = d["pool4x10"]
        p, g, ok = read_clip(reader, x.float().numpy())
        t = np.arange(len(p))
        ci = np.full(len(p), row[c], np.int64)
        lp, lg = label_rows(J, ci, t)
        sel = ok
        for k, v in (("ci", np.full(sel.sum(), n)), ("t", t[sel]), ("p", p[sel]), ("g", g[sel]), ("lab_pres", lp[sel]), ("lab_gap", lg[sel]), ("fit", np.full(sel.sum(), is_fit(c)))):
            cols[k].append(v)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    with open(a.out, "wb") as fh:
        np.savez(fh, clips=np.array(held), **out)
    rec = {"reader": a.reader, "reader_sha256": sha256_file(a.reader), "out": a.out, "out_sha256": sha256_file(a.out), "n_heldout_clips": len(held), "n_fit_clips": int(sum(is_fit(c) for c in held)),
           "n_rows": int(len(out["t"])), "join": a.join, "tier": "T0"}
    json.dump(rec, open(a.out + ".json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("extract", json.dumps({k: rec[k] for k in ("n_heldout_clips", "n_fit_clips", "n_rows")}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--reader", required=True)
    e.add_argument("--cache", required=True)
    e.add_argument("--join", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_extract(a)


if __name__ == "__main__":
    sys.exit(main())
