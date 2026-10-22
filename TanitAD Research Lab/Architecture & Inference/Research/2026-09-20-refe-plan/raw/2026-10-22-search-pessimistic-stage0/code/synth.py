#!/usr/bin/env python3
"""Synthetic worlds for the Stage 0 tests and the power simulation.  The GEOMETRY (24 logs, tokens per log, Step A tokens per log) is the real table's (non-label counts, geometry_real.json);
features and labels are synthetic.  Labels come from `stage0.plant_labels`, i.e. from the SAME generator the C-II-b control uses on the real features, so a planted gap of strength `kappa`
means the same thing in the tests, in the power simulation and in the real read's positive control."""
from __future__ import annotations

import json
import os

import numpy as np

import stage0 as S0

HERE = os.path.dirname(os.path.abspath(__file__))
GEOMETRY = os.path.join(HERE, "geometry_real.json")


def load_geometry(path: str = GEOMETRY) -> dict:
    return json.load(open(path))["logs"]


def make_world(seed: int = 0, kappa: float = 0.0, D: int = 256, geometry: dict | None = None, frac: float = 1.0, K: int = 64, Q: int = 96,
               rho_c: float = 1.0, rho_s: float = 0.5, sig_logit: float = 2.0, noise_ep: float = 0.05, sharp: float = S0.PLANT_SHARP, b_hi: float = 5.0, het: float = 0.0) -> S0.World:
    """frac < 1 thins every log (fast tests) but keeps the 24 logs / 21 sessions / Step A structure."""
    geo = geometry or load_geometry()
    rng = np.random.default_rng(seed)
    logs, toks, sa_rows = [], [], []
    for lg, v in sorted(geo.items()):
        n = max(int(round(v["n_tokens"] * frac)), 4)
        nsa = min(max(int(round(v["n_stepA"] * frac)), 1), n)
        base = len(logs)
        logs += [lg] * n
        toks += [f"{lg[-20:]}_{i:04d}" for i in range(n)]
        sa_rows += list(range(base, base + nsa))
    N = len(logs)
    logs, toks = np.array(logs), np.array(toks)
    sess = np.array([S0.session_of(x) for x in logs])
    sess_u = sorted(set(sess.tolist()))
    c = rng.standard_normal((N, 1, D)) * rho_c
    ms = {s: rng.standard_normal(D) * rho_s for s in sess_u}
    m = np.stack([ms[s] for s in sess])[:, None, :]
    X = (c + m + rng.standard_normal((N, K, D))).astype(np.float32)
    var = 1.0 + rho_c ** 2 + rho_s ** 2
    G = rng.standard_normal((6, D))
    W6 = G / np.linalg.norm(G, axis=1, keepdims=True) * (sig_logit / np.sqrt(var))
    b6 = np.array([b_hi, b_hi, 0.0, b_hi - 0.5, b_hi - 0.5, b_hi]) + rng.standard_normal(6) * 0.3     # most real proposals pass NC / DAC / TTC / C: pdms is high and regret is small
    sa_row = np.array(sa_rows, dtype=np.int64)
    Ns = len(sa_row)
    slots = np.stack([rng.permutation(K)[:16] for _ in range(Ns)])
    anchor = np.repeat(slots, Q // 16, axis=1).astype(np.int64)
    Xp = (np.take_along_axis(X[sa_row], anchor[..., None], 1) + 0.5 * rng.standard_normal((Ns, Q, D))).astype(np.float32)
    dummy = S0.Labels(np.ones((N, K, 6)), np.ones((N, K)), np.ones((N, K), bool), np.ones((Ns, Q, 6)), np.ones((Ns, Q)), np.ones((Ns, Q), bool))
    w = S0.World(toks, logs, X, dummy, sa_row, Xp, anchor, W6, b6, meta={"kind": "synthetic", "seed": seed, "kappa": kappa, "D": D, "frac": frac})
    sharp_s = sharp * np.exp(rng.uniform(-het, het, w.S)) if het else sharp                       # het > 0: sessions differ in how predictable their labels are (a nuisance, not a gap)
    return w.with_labels(S0.plant_labels(w, kappa=kappa, seed=seed + 101, noise_ep=noise_ep, sharp=sharp_s))


def make_files(root: str, seed: int = 0, kappa: float = 0.0, D: int = 32, frac: float = 0.25) -> dict:
    """Write a SYNTHETIC copy of the real input files in the real layouts (so the loader, the sha256 gate bypass and the CLI path are exercised end to end).  Returns the world."""
    w = make_world(seed, kappa, D, frac=frac)
    L = w.labels
    os.makedirs(os.path.join(root, "p2_heldout_table_full_2026-10-18"), exist_ok=True)
    os.makedirs(os.path.join(root, "l2_offprop"), exist_ok=True)
    os.makedirs(os.path.join(root, "l1_stepA_run", "bundle"), exist_ok=True)
    logits = (w.X.astype(np.float64) @ w.W6.T + w.b6).astype(np.float32)
    np.savez(os.path.join(root, S0.INPUTS["table"][0]), token=w.token, log=w.log, chunk=np.zeros(w.N, np.int32), fingerprint=w.token, pdms=L.pdms, sub=L.sub, valid=L.valid, logits=logits,
             proposals=np.zeros((w.N, w.K, 8, 3), np.float32), pick=np.zeros(w.N, np.int64), sub_names=np.array(["a"] * 6), pdms_at_pick=np.zeros(w.N))
    np.savez(os.path.join(root, S0.INPUTS["prehead"][0]), token=w.token, log=w.log, prehead_features=w.X.astype(np.float16), what=np.array("synthetic"))
    np.savez(os.path.join(root, S0.INPUTS["head"][0]), weight=w.W6.astype(np.float32), bias=w.b6.astype(np.float32), ckpt=np.array("synthetic"))
    np.savez(os.path.join(root, S0.INPUTS["bundle"][0]), token=w.token[w.sa_row], log=w.log[w.sa_row], traj_raw=np.zeros((w.Ns, w.K, 20, 3), np.float32), logits=logits[w.sa_row],
             pick=np.zeros(w.Ns, np.int64), sub=L.sub[w.sa_row], pdms=L.pdms[w.sa_row], valid=L.valid[w.sa_row])
    np.savez(os.path.join(root, S0.INPUTS["perts"][0]), anchor=w.anchor.astype(np.int32))
    np.savez(os.path.join(root, S0.INPUTS["labels"][0]), sub=L.p_sub, pdms=L.p_pdms)
    np.savez(os.path.join(root, S0.INPUTS["feats"][0]), infan_tok=w.X[w.sa_row], pert_tok=w.Xp)
    return w
