"""Synthetic search-bank fixture: random but schema-exact stand-ins for a bundle.npz, a merged search npz and a labels npz (the files the adapter reads).  PLANTED data only: it is used by the tests and by the
simulation's PC3 (shortcut-equals-estimator) control, so that no real candidate statistic is ever computed to check plumbing.  The values mean nothing; the SHAPES, dtypes, key names and invariants are real."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epadm_searchbank as SB                                                                # noqa: E402


def make_fixture(out_dir: str, n_tokens: int = 60, n_logs: int = 6, seed: int = 0, moved_share: float = 0.3) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    P = SB.pickrule()
    rng = np.random.default_rng(seed)
    logs = np.array([f"2021.01.0{(i % n_logs) + 1}.00.00.00_veh-{i % n_logs}_00001_00100" for i in range(n_tokens)])
    tokens = np.array([f"{(i * 2654435761 + seed) % (1 << 64):016x}" for i in range(n_tokens)])
    ep_y = rng.beta(2, 2, (n_tokens, 64))
    logits = rng.normal(0.0, 1.0, (n_tokens, 64, 6))
    logits[..., 2] = np.log(np.clip(ep_y, 1e-3, 1 - 1e-3) / (1 - np.clip(ep_y, 1e-3, 1 - 1e-3))) * 0.8 + rng.normal(0, 0.8, (n_tokens, 64))
    logits[..., 0] += 2.5
    logits[..., 1] += 2.5
    logits = logits.astype(np.float32)
    lab = np.zeros((n_tokens, 64, 6))
    lab[..., 0] = (rng.random((n_tokens, 64)) < 0.96)
    lab[..., 1] = (rng.random((n_tokens, 64)) < 0.97)
    lab[..., 2] = ep_y
    lab[..., 3] = (rng.random((n_tokens, 64)) < 0.9)
    lab[..., 4] = (rng.random((n_tokens, 64)) < 0.95)
    lab[..., 5] = rng.beta(5, 1, (n_tokens, 64))
    pick = P.rule_pick(logits)
    ar = np.arange(n_tokens)
    moved = rng.random((n_tokens, 3)) < moved_share
    pl = logits[ar, pick].astype(np.float64) + rng.normal(0.0, 1e-3, (n_tokens, 6))            # the CPU re-score of the pick: a hair off the banked logits
    po = np.repeat(pl[:, None, :], 3, axis=1).copy()
    for k in range(3):
        bump = rng.uniform(0.15, 0.6, (n_tokens, 6))
        po[:, k][moved[:, k]] = (pl + bump)[moved[:, k]]                                         # all heads up => the frozen aggregate is higher than the pick's
    out_lab = np.repeat(lab[ar, pick][:, None, :], 3, axis=1).copy()
    for k in range(3):
        fresh = lab[ar, pick] + 0.0
        fresh[:, 2] = np.clip(fresh[:, 2] + rng.normal(0.0, 0.15, n_tokens), 0, 1)
        fresh[:, 5] = np.clip(fresh[:, 5] + rng.normal(0.0, 0.05, n_tokens), 0, 1)
        out_lab[:, k][moved[:, k]] = fresh[moved[:, k]]
    bpath = os.path.join(out_dir, "bundle.npz")
    np.savez(bpath, token=tokens, log=logs, traj_raw=np.zeros((n_tokens, 64, 20, 3), np.float32), logits=logits, pick=pick.astype(np.int64), sub=lab, pdms=np.zeros((n_tokens, 64)), valid=np.ones((n_tokens, 64), bool))
    tsha = SB.token_list_sha256(tokens)
    meta = {"arm": "P2HEAD", "head_sha256": None, "ckpt_sha256": SB.PIN_CKPT_SHA, "config_sha256": SB.PIN_CONFIG_SHA, "bundle_sha256": SB.sha256_file(bpath),
            "data_clause": {"carve_sha256": tsha, "bundle_tokens_sha256": tsha, "n_tokens": n_tokens, "sealed_default": True}, "n_tokens": n_tokens}
    spath = os.path.join(out_dir, "search.npz")
    np.savez(spath, token=tokens, bundle_index=np.arange(n_tokens), pick=pick.astype(np.int64), seeds=np.array(SB.PIN_SEEDS, np.int64), exec=np.zeros((n_tokens, 3, 20, 3), np.float32), moved=moved,
             pred_logit_out=po, pred_logit_pick=pl, meta=np.array(json.dumps(meta)))
    lpath = os.path.join(out_dir, "labels.npz")
    np.savez(lpath, sub=out_lab, pdms=np.zeros((n_tokens, 3)), id_sub=lab[ar, pick], id_pdms=np.zeros(n_tokens), sub_names=np.array(SB.LABEL_ORDER), token=tokens, bundle_index=np.arange(n_tokens),
             search_sha256=np.array(SB.sha256_file(spath)), identity=np.array("{}"))
    return {"bundle": bpath, "search": spath, "labels": lpath, "token_sha256": tsha, "dir": out_dir}


def rewrite(path: str, **changes) -> None:
    """Rewrite an npz with some arrays replaced (a defect planted into a fixture file)."""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    z.close()
    d.update(changes)
    np.savez(path, **d)


def refresh_labels_search_sha(fx: dict) -> None:
    """After editing the search file, re-point the labels at it (so only the planted defect, not the sha link, is under test)."""
    rewrite(fx["labels"], search_sha256=np.array(SB.sha256_file(fx["search"])))


def refresh_meta(fx: dict, **meta_changes) -> None:
    z = np.load(fx["search"], allow_pickle=False)
    m = json.loads(str(z["meta"]))
    m.update(meta_changes)
    rewrite(fx["search"], meta=np.array(json.dumps(m)))
    refresh_labels_search_sha(fx)


def build_default(fx: dict, out_dir: str) -> dict:
    return SB.build(fx["bundle"], fx["search"], fx["labels"], out_dir, {"token_sha256": fx["token_sha256"]})
