#!/usr/bin/env python3
"""Shared I/O + estimator for the RL-S1 O2(i) calibration read (PREREG_RL_S1.md section 11).  CPU only, zero training.

Reads ONLY banked artifacts, each sha256-pinned below:
  * score dumps of the HF private repos Sayood/tanitad-refe-rls1-{c0,r0,r0b} (final/scores_<tag>_{A,B}.npz; logit space, [N sets, 64, 7]);
  * RL-val labels, HF private Sayood/tanitad-refe-rls1 data/rlfan_{pdm,terms}_val.tar.gz (sha256 = MANIFEST_rlfan_sha256.txt).
The selection rule is the deployed one for the 7-output base: navsim_v1_lane = NC*DAC*(5EP+5TTC+2C)/12 * p_lane over sigmoid(logits)
(refe/planner.py:552-557, eval/rl_val_eval.py select(), refe/scorer_finetune.py sel_score).  Ties -> lowest index (np.argmax, = RV.select).
Estimator: paired log-cluster bootstrap, eval/proxy_eval.boot (10,000, seed 20260927): "another draw of LOGS" only.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PK, "refe"))
sys.path.insert(0, os.path.join(PK, "eval"))
import proxy_eval as PE  # noqa: E402
import rl_reward as RR  # noqa: E402
import rl_val_eval as RV  # noqa: E402

TOL = RR.SAFE_TOL
DL_DEFAULT = r"C:/Users/Admin/rls1b_dl"          # hf_hub_download local_dir layout: <dl>/{c0,r0,r0b}/final/...
X_DEFAULT = r"C:/Users/Admin/rls1_calib_x"       # fresh extraction of the two RL-val label tarballs (byte-equal to the earlier one)
SUBS = ("PDMS", "NC", "DAC", "EP", "TTC", "C", "DDC")
COL = {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3, "C": 4, "DDC": 5, "LANE": 6}

# sha256 of every input (HF LFS sha256 == local, verified 2026-10-15 by hf_get_verify.py; tarballs == MANIFEST_rlfan_sha256.txt)
PIN = {
    "c0/final/scores_step0_base.npz": "6332fcc6175805349dd84766d6e38eac79f3ab194588b541755c1cf6281fef63",
    "c0/final/scores_update250_A.npz": "5afec357353c67810a9b8c547bcb1f103fef378e7351bfebc3edba2e5443ca55",
    "c0/final/scores_update500_A.npz": "8500f6d506a39dcab44762265a0e5ac4d11e57e66d3636dc4c81e812ba200fbd",
    "c0/final/scores_update750_A.npz": "41e452233900155c506236b19301879bf266eb7ac6abf54b505528b5e20fe440",
    "c0/final/scores_update1000_A.npz": "5a524381fa8075dd5d74ca390525dada829935770403d2f7fb8ea431cebdb9f5",
    "r0/final/scores_update250_A.npz": "f0c886212603471f970fbc0d0036fb6e0fb95e43ea4bb5ffbee7c3f6098a6007",
    "r0/final/scores_update250_B.npz": "4a9a4b94ec4d7740710ccd9e449a1f7ef4add452937d3eb9ac4589726111df05",
    "r0/final/scores_update500_A.npz": "e43ff0d11a7dec1db0b5dc7de52501d30076fdca6eafabd7c5a6832c6c8cab4c",
    "r0/final/scores_update500_B.npz": "f5ae251182c19736532af7d4b59657a1e771f857d82eac62d577cd2889160384",
    "r0/final/scores_update750_B.npz": "0f486d42c5625eca9c44c4858936c31f682491ad65fb3a010b9be16cf9ed0195",
    "r0/final/scores_update1000_A.npz": "c5986680bd076987e2a9dd39b6368b3df8eaeb32f5afa6d74e243d0e2886ed2f",
    "r0/final/scores_update1000_B.npz": "65dcd60e12744717ec535cbbbc07d4aa6ef4f52a8be1438baf98c1ac232aea46",
    "r0b/final/scores_step0_base.npz": "6332fcc6175805349dd84766d6e38eac79f3ab194588b541755c1cf6281fef63",
    "r0b/final/scores_update250_A.npz": "b968da74ae50130784f01e49a993b47c20ea656ce129a0e8eb2c2312311ed8f3",
    "r0b/final/scores_update250_B.npz": "41374b8e85b244d15c44c374ac78e3d57386a1f96a379a06b00682f038a244e8",
    "r0b/final/scores_update500_A.npz": "7f5e56e398f1e2034822b20421d99e1fb6e0662ee1665199beddc84a7a559b4c",
    "r0b/final/scores_update500_B.npz": "74910863de3a7557afd6daa8c2f655c95a1724c757dfec9346aeec1f02691277",
    "data/data/rlfan_pdm_val.tar.gz": "3dd6cba33cb1395f6c18b0204299a235384a160ea1456395f132de82fefbaae7",
    "data/data/rlfan_terms_val.tar.gz": "6f12b90fa14abc86effa78643d78a72aff8b1145ec6bd66d84290b9d2fa3fce7",
}
REPOS = {"c0": "Sayood/tanitad-refe-rls1-c0", "r0": "Sayood/tanitad-refe-rls1-r0", "r0b": "Sayood/tanitad-refe-rls1-r0b",
         "data": "Sayood/tanitad-refe-rls1"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(dl, rels):
    out = {}
    for rel in rels:
        got = sha(os.path.join(dl, rel))
        if got != PIN[rel]:
            raise SystemExit(f"sha256 mismatch {rel}: {got} != {PIN[rel]}")
        out[rel] = got
    return out


def load_val(x):
    """-> keys (sorted), tg [N, 64, 6], comps dict, stats.  Join = pdm ∩ terms with equal ckpt_step (attrib_rls1b.load_side)."""
    pdm, st_p = RV.load_pdm(os.path.join(x, "pdm_val", "pdm_val"))
    terms, st_t = RV.load_terms(os.path.join(x, "terms_val", "terms_val"))
    keys = sorted(k for k in pdm if k in terms and terms[k]["ckpt"] == pdm[k]["ckpt"])
    tg = np.stack([pdm[k]["tg"] for k in keys])
    ck = {k: pdm[k]["ckpt"] for k in keys}
    s = RR.split_components(tg)
    s["pdms"] = RR.pdms_v1(s["nc"], s["dac"], s["ep"], s["ttc"], s["c"])
    s["safe"] = RR.safe_mask(s["nc"], s["dac"], s["ttc"])
    return keys, tg, ck, s, {"pdm": st_p, "terms": st_t, "joined": len(keys)}


def load_scores_aligned(path, keys, ck):
    """RV.load_scores semantics (key_of, duplicate = error, logit space, ckpt_step must equal the label's), one read per column."""
    with np.load(path, allow_pickle=False) as z:
        if "space" in z.files and str(z["space"]) != "logit":
            raise SystemExit(f"{path}: not logit space")
        lg, tk, st, rk, cs = z["log"], z["token"], z["step"], z["rank"], z["ckpt_step"]
        sc = np.asarray(z["scores"], np.float64)
    idx = {}
    for i in range(len(tk)):
        k = RV.key_of((lg[i], tk[i], st[i], rk[i]))
        if k in idx:
            raise SystemExit(f"{path}: duplicate key {k}")
        idx[k] = i
    miss = [k for k in keys if k not in idx or int(cs[idx[k]]) != ck[k]]
    if miss:
        raise SystemExit(f"{path}: {len(miss)} sets missing or stale")
    out = sc[[idx[k] for k in keys]]
    if out.shape[1:] != (64, 7):
        raise SystemExit(f"{path}: shape {out.shape}")
    return out


def sel_v1_lane(p):
    return p[..., 0] * p[..., 1] * (5.0 * p[..., 2] + 5.0 * p[..., 3] + 2.0 * p[..., 4]) / 12.0 * p[..., 6]


def picks_from_probs(p):
    return sel_v1_lane(p).argmax(-1)


def picks_from_logits(z):
    return picks_from_probs(RV.sigmoid(z))


def sub_at(s, pick):
    ar = np.arange(len(pick))
    return {"PDMS": s["pdms"][ar, pick], "NC": s["nc"][ar, pick], "DAC": s["dac"][ar, pick], "EP": s["ep"][ar, pick],
            "TTC": s["ttc"][ar, pick], "C": s["c"][ar, pick], "DDC": s["ddc"][ar, pick]}


def boot_delta(a, b, keys, scale=100.0):
    """paired log-cluster bootstrap of mean(a - b) x scale; keys carry the log in position 0."""
    d = {RV.kstr(k): scale * (float(a[n]) - float(b[n])) for n, k in enumerate(keys)}
    tl = {RV.kstr(k): k[0] for k in keys}
    mu, lo, hi = PE.boot(d, tl)
    return {"mean": mu, "ci95": [lo, hi], "separated": bool(lo > 0 or hi < 0)}


def delta_table(s, pick_x, pick_c, keys):
    """every sub-score delta (x100 = 'points') of pick_x minus pick_c, each with its paired log-cluster CI, plus pick-changed / unsafe counts."""
    ax, ac = sub_at(s, pick_x), sub_at(s, pick_c)
    out = {q: boot_delta(ax[q], ac[q], keys) for q in SUBS}
    out["means_x100"] = {"x": {q: float(100 * ax[q].mean()) for q in SUBS}, "c": {q: float(100 * ac[q].mean()) for q in SUBS}}
    out["picks_changed"] = int((pick_x != pick_c).sum())
    ar = np.arange(len(pick_x))
    out["unsafe_picks"] = {"x": int((~s["safe"][ar, pick_x]).sum()), "c": int((~s["safe"][ar, pick_c]).sum())}
    return out


def g5(tab, bound=0.5):
    """the registered g5 read: POINT deltas (x100) of TTC and NC may not fall by more than `bound` versus the control (rl_val_eval G5_SUBSCORE_FALL_MAX)."""
    return {"TTC_fall_ok": bool(tab["TTC"]["mean"] >= -bound), "NC_fall_ok": bool(tab["NC"]["mean"] >= -bound), "bound": bound}
