#!/usr/bin/env python3
"""EP-1 shared helpers. CPU only. Tables are read in place, never modified.

Table schema (both the fit table and the navtest table): token, log, pdms [N,64], sub [N,64,6] (nc, dac, ep, ttc, c, ddc),
logits [N,64,6] (float32), proposals [N,64,8,3] (x, y, heading at 0.5 s steps), pick [N].
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from zero_tail import Boot, B_BOOT, SEED  # noqa: E402,F401  (vendored byte-identical, sha256 149c3e97...)

FIT = "D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz"
FIT_SHA = "8eacc19b35b12aee9b0ab047716c901399a711d22ddd3c25cd890ecf2f02004b"
NAVTEST = "D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz"
NAVTEST_SHA_PREFIX = "33441082b7044c95"
NC, DAC, EP, TTC, C, DDC = range(6)
W = np.array([5.0, 5.0, 2.0])  # EP, TTC, C ; sum 12


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def load(path, expect_prefix=None):
    sha = sha256_file(path)
    if expect_prefix and not sha.startswith(expect_prefix):
        raise SystemExit(f"table sha256 mismatch: {sha}")
    z = np.load(path, allow_pickle=True)
    t = {k: z[k] for k in z.files}
    t["sha256"] = sha
    logs = np.unique(t["log"])
    t["log_names"] = logs
    t["log_idx"] = np.searchsorted(logs, t["log"])
    t["n_logs"] = len(logs)
    return t


def pdms_formula(sub):
    """NC * DAC * (5 EP + 5 TTC + 2 C) / 12, DDC at weight 0 (MEASURED to reproduce the table)."""
    return sub[..., NC] * sub[..., DAC] * (5 * sub[..., EP] + 5 * sub[..., TTC] + 2 * sub[..., C]) / 12.0


def score_terms(p):
    """p = sigmoid(logits) [..., 6] -> (pNC*pDAC, W) with the planner's navsim_v1 rule."""
    w = (5 * p[..., EP] + 5 * p[..., TTC] + 2 * p[..., C]) / 12.0
    return p[..., NC] * p[..., DAC], w


def planner_score(logits):
    p = sigmoid(logits)
    g, w = score_terms(p)
    return g * w


def geometry(proposals):
    """end distance L [N,64] (m), path length P [N,64] (m), mean speed v=L/4 s, final lateral offset |y|."""
    xy = proposals[..., :2].astype(np.float64)
    L = np.linalg.norm(xy[..., -1, :], axis=-1)
    seg = np.diff(np.concatenate([np.zeros_like(xy[..., :1, :]), xy], axis=-2), axis=-2)
    P = np.linalg.norm(seg, axis=-1).sum(-1)
    return L, P


def pick_ref(t):
    return np.argmax(planner_score(t["logits"]), axis=1)


def gates(t, name):
    """Instrument gates on a table: PDMS formula reproduces; planner rule reproduces `pick`; pdms at pick matches."""
    f = pdms_formula(t["sub"])
    g = {"name": name, "n_tokens": int(len(t["token"])), "n_logs": int(t["n_logs"]), "table_sha256": t["sha256"]}
    g["pdms_formula_max_abs_diff"] = float(np.abs(f - t["pdms"]).max())
    f_bad = t["sub"].copy()
    f_bad_pdms = f_bad[..., NC] * f_bad[..., DAC] * (5 * f_bad[..., EP] + 5 * f_bad[..., TTC] + 2 * f_bad[..., C] + 0 * f_bad[..., DDC]) / 12.0 * f_bad[..., DDC]
    g["control_ddc_in_product_max_abs_diff"] = float(np.abs(f_bad_pdms - t["pdms"]).max())
    pk = pick_ref(t)
    g["pick_reproduced"] = int((pk == t["pick"]).sum())
    g["pdms_at_pick_mean_x100"] = float(t["pdms"][np.arange(len(pk)), pk].mean() * 100)
    return g


def paired(boot, a_tok, b_tok, scale=100.0):
    """Paired difference of two per-token vectors (a - b), mean over tokens, log-cluster bootstrap, same draws."""
    d = np.asarray(a_tok, float) - np.asarray(b_tok, float)
    return boot.ratio(d, np.ones_like(d), scale)


def level(boot, a_tok, scale=100.0):
    a = np.asarray(a_tok, float)
    return boot.ratio(a, np.ones_like(a), scale)


def at(arr, idx):
    """arr [N,64,...] at per-token proposal index idx [N]."""
    return arr[np.arange(len(idx)), idx]
