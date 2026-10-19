"""Synthetic data for the CPU tests and the false-fire simulations of SH5 (nothing here touches the real corpus).

A token has an ASYMMETRIC road: a left boundary at lateral offset +wl and a right one at -wr (m).  Each of its sets has 64 proposals whose key-time lateral offsets are y_k(t); the signed clearance of a
point is min(wl - y, wr + y), the DAC label is 1 iff every key clearance is >= 0.  P2's stand-in logit z_L knows the proposals but only a GENERIC symmetric width (3.2 m) plus noise, so it ranks the
proposals of a set by |y| - and a within-set ranking by |y| is WRONG whenever wl != wr (a symmetric width alone would leave the within-set order untouched: only the threshold moves, which a
within-set AUC cannot see; that was the first draft of this world and it made the ctx useless by construction).  ctx (when `info`) carries wl and wr in channels 0 and 1 at every position (plus noise), i.e. the
information the branch must read; without it the ctx is pure noise (the NULL).  A token's ctx from ANOTHER token (S-shuf) carries the wrong asymmetry.
"""
from __future__ import annotations

import numpy as np

import sh5_lib as L


def make_synth(T: int = 40, per_log: int = 2, sets_per_tok=(1, 2), P: int = 1280, d_ctx: int = 40, info: bool = True, seed: int = 0, zl_noise: float = 1.0, ctx_noise: float = 0.05,
               ctx_leak_label: bool = False, w_lo: float = 1.0, w_hi: float = 5.0):
    rng = np.random.default_rng(seed)
    wl_tok = rng.uniform(w_lo, w_hi, T)
    wr_tok = rng.uniform(w_lo, w_hi, T)
    tok_log = np.arange(T) // per_log
    n_logs = int(tok_log.max()) + 1
    rows = []
    for t in range(T):
        for _ in range(int(rng.integers(sets_per_tok[0], sets_per_tok[1] + 1))):
            rows.append(t)
    token_idx = np.array(rows, np.int64)
    N = len(token_idx)
    kt = np.array(L.KEY_TIMES_S)
    v = rng.uniform(3.0, 12.0, (N, 64, 1))
    o = rng.normal(0.0, 1.4, (N, 64, 1))
    dr = rng.normal(0.0, 0.6, (N, 64, 1))
    xs = v * kt
    ys = o + dr * kt
    yaw = np.arctan2(dr, v) * np.ones_like(xs)
    pts = np.stack([xs, ys, yaw], axis=-1).astype(np.float32)                    # [N,64,9,3]
    wl, wr = wl_tok[token_idx][:, None, None], wr_tok[token_idx][:, None, None]
    sc = np.minimum(wl - ys, wr + ys).astype(np.float32)                          # signed clearance at the 9 key times
    tgt = np.minimum(np.abs(sc), 10.0).astype(np.float32)                        # the 9 key-time |clearance| (sign-clean)
    y = (sc.min(-1) >= 0.0).astype(np.float32)
    zl = 2.0 * (3.2 - np.abs(ys).max(-1)) + rng.normal(0.0, zl_noise, (N, 64))
    z6 = rng.normal(0.0, 1.0, (N, 64, 6)).astype(np.float32)
    z6[..., L.DAC_COL] = zl.astype(np.float32)
    lab = np.ones((N, 64, 6), np.float32)
    lab[..., L.DAC_COL] = y
    spd = v[..., 0]
    q = np.quantile(spd, [0.25, 0.5, 0.75])
    code = (1 + (spd > q[0]).astype(int) + (spd > q[1]) + (spd > q[2])).astype(np.uint8)
    log_idx = tok_log[token_idx].astype(np.int64)
    fold = (log_idx % L.FOLDS).astype(np.int64)
    ctx = rng.normal(0.0, 1.0, (T, P, d_ctx)).astype(np.float32)
    if info:
        ctx[:, :, 0] = (wl_tok / 4.0)[:, None] + rng.normal(0.0, ctx_noise, (T, P))
        ctx[:, :, 1] = (wr_tok / 4.0)[:, None] + rng.normal(0.0, ctx_noise, (T, P))
    if ctx_leak_label:                                                          # an INTENTIONAL leak (RED control): ctx carries the label of set 0 of the token
        for t in range(T):
            ctx[t, :, 2] = y[np.flatnonzero(token_idx == t)[0]].mean()
    D = {"N": N, "T": T, "n_logs": n_logs, "token": np.array([f"t{t:03d}" for t in range(T)]), "token_idx": token_idx, "fold": fold, "log_idx": log_idx, "code": code, "z6": z6, "lab": lab,
         "ep_s0": rng.normal(0, 1, (N, 64)).astype(np.float32), "ep_s1": rng.normal(0, 1, (N, 64)).astype(np.float32), "pts": pts, "tgt": tgt, "sc_key": sc,
         "tok_fold": np.array([fold[np.flatnonzero(token_idx == t)[0]] for t in range(T)], np.int64), "tok_sets": [np.flatnonzero(token_idx == t) for t in range(T)]}
    return D, ctx
