#!/usr/bin/env python
"""closest.py -- card PREREG_V8F_PROOF_ADD79 (ESTCLOSE10K): the saved closing-speed estimator, numpy only (no torch). The ONE definition of the file format and of the forward;
the save tool writes it, the study wrapper and the reader apply it.

Input of an estimate at a look at frame j of a clip: the reader's reads on the five-frame windows that end at the frames j-H+1 .. j, oldest first, per frame
(gap read before any map in m, presence p in 0..1, closing output times 10 m/s): X [H, 3]. It needs the frames j-H-3 .. j of the clip (H + reader history - 1 frames).

File (npz, no pickle):
  arm   "L" (ridge) or "N" (mean of small networks)        H  history in frames        seed  reader seed (-1 for a planted file)
  mu, sd [3H] float64: input normalisation, z = (x - mu) / sd
  L:  w [3H] float64, b float64                      estimate = z @ w + b
  N:  n_nets; per net i: W1_i [64, 3H], b1_i [64], W2_i [64, 64], b2_i [64], W3_i [1, 64], b3_i [1] (float32)
      estimate = mean over nets of (relu(relu(z32 @ W1.T + b1) @ W2.T + b2) @ W3.T + b3), z32 = z as float32 (the forward of the fit tool, closeread_fit.py train_mlp)
  meta_json  a JSON string (card, source dump hashes, fit record)

  python closest.py --selftest
"""
from __future__ import annotations

import json
import sys

import numpy as np

READER_HIST = 5


class ClosingEstimator:
    def __init__(self, d):
        self.arm = str(d["arm"])
        self.H = int(d["H"])
        self.seed = int(d["seed"]) if "seed" in d else -1
        self.mu = np.asarray(d["mu"], np.float64)
        self.sd = np.asarray(d["sd"], np.float64)
        n = 3 * self.H
        if self.arm not in ("L", "N") or self.mu.shape != (n,) or self.sd.shape != (n,) or not np.all(self.sd > 0):
            raise ValueError("estimator file: arm, mu or sd not as the format has them")
        if self.arm == "L":
            self.w = np.asarray(d["w"], np.float64)
            self.b = float(d["b"])
            if self.w.shape != (n,):
                raise ValueError("estimator file: w has shape %s, expected (%d,)" % (self.w.shape, n))
            self.nets = []
        else:
            k = int(d["n_nets"])
            if k < 1:
                raise ValueError("estimator file: no net")
            self.nets = [tuple(np.asarray(d["%s_%d" % (nm, i)], np.float32) for nm in ("W1", "b1", "W2", "b2", "W3", "b3")) for i in range(k)]
            for W1, b1, W2, b2, W3, b3 in self.nets:
                if W1.shape[1] != n or W2.shape[1] != W1.shape[0] or W3.shape != (1, W2.shape[0]):
                    raise ValueError("estimator file: net shapes not as the format has them")
        self.meta = json.loads(str(d["meta_json"])) if "meta_json" in d else {}

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls({k: z[k] for k in z.files})

    def need_frames(self, reader_hist=READER_HIST):
        """frames of the clip an estimate at frame j needs: j - need_frames + 1 .. j"""
        return self.H + reader_hist - 1

    def apply(self, X):
        """X [n, H, 3] or [n, 3H] (oldest frame first; gap read, presence, closing output) -> estimate [n] float64 in m/s. Non-finite input rows give NaN."""
        X = np.asarray(X, np.float64)
        X = X.reshape(X.shape[0], -1) if X.ndim != 1 else X.reshape(1, -1)
        if X.shape[1] != 3 * self.H:
            raise ValueError("input has %d columns, the estimator needs %d" % (X.shape[1], 3 * self.H))
        ok = np.isfinite(X).all(1)
        out = np.full(X.shape[0], np.nan)
        if not ok.any():
            return out
        Z = (X[ok] - self.mu) / self.sd
        if self.arm == "L":
            out[ok] = Z @ self.w + self.b
            return out
        z = Z.astype(np.float32)
        preds = []
        for W1, b1, W2, b2, W3, b3 in self.nets:
            h = np.maximum(z @ W1.T + b1, 0.0)
            h = np.maximum(h @ W2.T + b2, 0.0)
            preds.append((h @ W3.T + b3)[:, 0].astype(np.float64))
        out[ok] = np.mean(preds, 0)
        return out


def save_linear(path, H, mu, sd, w, b, seed=-1, meta=None):
    np.savez(path, arm=np.asarray("L"), H=np.int64(H), seed=np.int64(seed), mu=np.asarray(mu, np.float64), sd=np.asarray(sd, np.float64), w=np.asarray(w, np.float64), b=np.float64(b),
             meta_json=np.asarray(json.dumps(meta or {})))


def save_nets(path, H, mu, sd, nets, seed=-1, meta=None):
    """nets: list of (W1, b1, W2, b2, W3, b3) arrays (torch layout: weight [out, in])."""
    d = {"arm": np.asarray("N"), "H": np.int64(H), "seed": np.int64(seed), "mu": np.asarray(mu, np.float64), "sd": np.asarray(sd, np.float64), "n_nets": np.int64(len(nets)),
         "meta_json": np.asarray(json.dumps(meta or {}))}
    for i, net in enumerate(nets):
        for nm, a in zip(("W1", "b1", "W2", "b2", "W3", "b3"), net):
            d["%s_%d" % (nm, i)] = np.asarray(a, np.float32)
    np.savez(path, **d)


def save_planted(path, kind, H, value=0.0):
    """planted estimators of control e1: 'zero' -> 0.0; 'const' -> value; 'reader_closing' -> the closing output of the last frame (column 3H-1 of the input)."""
    n = 3 * H
    w = np.zeros(n)
    b = 0.0
    if kind == "const":
        b = float(value)
    elif kind == "reader_closing":
        w[n - 1] = 1.0
    elif kind != "zero":
        raise ValueError("kind " + kind)
    save_linear(path, H, np.zeros(n), np.ones(n), w, b, seed=-1, meta={"planted": kind, "value": float(value)})


def selftest():
    import os
    import tempfile
    fails = []

    def chk(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)
    rng = np.random.default_rng(0)
    d = tempfile.mkdtemp(prefix="closest_selftest_")
    H = 10
    X = rng.normal(0, 1, (50, H, 3))
    for kind, want in (("zero", np.zeros(50)), ("const", np.full(50, 2.0)), ("reader_closing", X[:, -1, 2])):
        p = os.path.join(d, kind + ".npz")
        save_planted(p, kind, H, 2.0)
        chk("planted %s" % kind, np.allclose(ClosingEstimator.load(p).apply(X), want, atol=1e-12))
    mu, sd, w = rng.normal(0, 1, 3 * H), rng.uniform(0.5, 2, 3 * H), rng.normal(0, 1, 3 * H)
    p = os.path.join(d, "lin.npz")
    save_linear(p, H, mu, sd, w, 0.3, seed=1, meta={"a": 1})
    E = ClosingEstimator.load(p)
    chk("linear forward and meta", np.allclose(E.apply(X), ((X.reshape(50, -1) - mu) / sd) @ w + 0.3) and E.meta == {"a": 1} and E.seed == 1 and E.need_frames() == H + 4)
    nets = [(rng.normal(0, 0.3, (64, 3 * H)), rng.normal(0, 0.1, 64), rng.normal(0, 0.3, (64, 64)), rng.normal(0, 0.1, 64), rng.normal(0, 0.3, (1, 64)), rng.normal(0, 0.1, 1)) for _ in range(3)]
    p = os.path.join(d, "net.npz")
    save_nets(p, H, mu, sd, nets, seed=0)
    E = ClosingEstimator.load(p)
    z = ((X.reshape(50, -1) - mu) / sd).astype(np.float32)
    ref = np.mean([(np.maximum(np.maximum(z @ a.astype(np.float32).T + b.astype(np.float32), 0) @ c.astype(np.float32).T + e.astype(np.float32), 0) @ f.astype(np.float32).T + g.astype(np.float32))[:, 0].astype(np.float64)
                   for a, b, c, e, f, g in nets], 0)
    chk("network forward, mean of three nets", np.allclose(E.apply(X), ref, atol=1e-6))
    Xn = X.copy()
    Xn[3, 2, 0] = np.nan
    out = E.apply(Xn)
    chk("a non-finite input row gives NaN and no other row changes", np.isnan(out[3]) and np.allclose(np.delete(out, 3), np.delete(ref, 3), atol=1e-6))
    try:
        E.apply(np.zeros((2, 3 * H + 1)))
        chk("a wrong input width is refused", False)
    except ValueError:
        chk("a wrong input width is refused", True)
    print("ZZ CLOSEST-SELFTEST rc %d" % (0 if not fails else 1))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else 0)
