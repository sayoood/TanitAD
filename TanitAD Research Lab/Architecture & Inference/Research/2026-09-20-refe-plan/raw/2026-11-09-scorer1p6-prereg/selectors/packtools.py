#!/usr/bin/env python3
"""packtools.py -- builders of fit_pack.npz-format packs for TESTS and the SMOKE only. The real pack comes from the pod label job (labels/p6_labels_tool.py cmd_pack).

  make_synthetic_pack(path, ...)   a pack of the exact format, with learnable structure (truth depends on geometry; logits are noisy views of it)
  make_standin_pack(dump, table, path)   the 1,058 real navtest tokens of dump_p6.npz + table_p6.npz in the same format: SMOKE ONLY, no number from it is a result
"""
from __future__ import annotations

import numpy as np

PDM_NAMES = ["nc", "dac", "ep", "ttc", "c", "ddc", "pdms"]


def _save(path, log, token, xy, yaw, logits, pdm, has, v0=None, heldout=None):
    n = len(log)
    kw = dict(log=np.array(log), token=np.array(token), step=np.zeros(n, int), rank=np.zeros(n, int), ckpt_step=np.zeros(n, int), traj_xy=xy.astype(np.float32),
              yaw=yaw.astype(np.float32), logits=logits.astype(np.float32), pdm=pdm.astype(np.float32), pdm_names=np.array(PDM_NAMES), has_pdm=has,
              is_heldout=np.zeros(n, bool) if heldout is None else heldout, logit_space=np.array("raw head output, bf16 autocast, as P6 emits it"))
    if v0 is not None:
        kw["v0"] = np.asarray(v0, np.float32)
    np.savez(path, **kw)


def make_synthetic_pack(path, n_logs=24, sets_per_log=20, seed=0, with_v0=True, token_prefix="syn", uneven=True):
    rng = np.random.default_rng(seed)
    log, token, XY, YAW, LG, PD, V0 = [], [], [], [], [], [], []
    for li in range(n_logs):
        ns = int(sets_per_log * (0.5 + rng.random())) if uneven else sets_per_log
        for si in range(max(ns, 2)):
            v0 = 3 + 12 * rng.random()
            sp = v0 * (0.6 + 0.8 * rng.random(64))              # per-proposal speed
            lat = rng.normal(0, 1.5, 64)                        # lateral offset at the end
            t = (np.arange(1, 9) / 8.0)[None, :]
            x = np.cumsum(np.repeat((sp * 0.5)[:, None], 8, 1), axis=1)
            y = lat[:, None] * t ** 2
            yaw = np.arctan2(np.gradient(y, axis=1), np.maximum(np.gradient(x, axis=1), 1e-3))
            ratio = sp / v0
            nc = (rng.random(64) > 0.15 * (ratio > 1.15)).astype(float)
            dac = (np.abs(lat) < 2.5).astype(float)
            ttc = (ratio < 1.2).astype(float) * (rng.random(64) > 0.05)
            ep = np.clip(sp / (1.4 * v0), 0, 1)
            c = np.ones(64)
            ddc = np.ones(64)
            pdms = nc * dac * (5 * ep + 5 * ttc + 2 * c) / 12
            truth = np.stack([nc, dac, ep, ttc, c, ddc], -1)
            tp = np.clip(0.8 * truth + 0.1, 0.02, 0.98)
            logits = np.log(tp / (1 - tp)) + rng.normal(0, 0.9, truth.shape)
            log.append(f"{token_prefix}_log{li:03d}")
            token.append(f"{token_prefix}{li:03d}{si:03d}{seed:02d}")
            XY.append(np.stack([x, y], -1))
            YAW.append(yaw)
            LG.append(logits)
            PD.append(np.concatenate([truth, pdms[:, None]], -1))
            V0.append(v0)
    n = len(log)
    _save(path, log, token, np.array(XY), np.array(YAW), np.array(LG), np.array(PD), np.ones(n, bool), np.array(V0) if with_v0 else None)
    return n


def make_standin_pack(dump_npz, table_npz, path):
    """read-only on both inputs. dump: props8 [N,64,8,3], logits, ego (v0 = ego[:,6]), log; table: pdms [N,64], sub [N,64,6]"""
    Z, T = np.load(dump_npz, allow_pickle=False), np.load(table_npz, allow_pickle=False)
    assert [str(t) for t in Z["token"]] == [str(t) for t in T["token"]]
    P8 = Z["props8"]
    pdm = np.concatenate([T["sub"], T["pdms"][..., None]], -1)
    _save(path, [str(x) for x in Z["log"]], [str(x) for x in Z["token"]], P8[..., :2], P8[..., 2], Z["logits"], pdm, np.ones(len(P8), bool), Z["ego"][:, 6])
    return len(P8)
