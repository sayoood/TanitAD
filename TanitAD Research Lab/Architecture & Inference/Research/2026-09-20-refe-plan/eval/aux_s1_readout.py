#!/usr/bin/env python3
"""S1 readout harness of the auxiliary drivable-area head (eval/PREREG_AUX_DRIVABLE_DRAFT.md, raw/2026-10-13-aux-head/PREREG_BLANKS_PROPOSED.md section 5).

CPU only, no model, no trunk: it READS banked outputs. Two readers, one estimator.

PERCEPTION READS (on the head itself; an N arm has no head, so these are per head arm and are paired H2-sK vs H1-sK)
  P1  held-out mask IoU in the bands [0,20) [20,40) [40,72) m (LONGITUDINAL x of the 2 m cell centre in the rear-axle frame; 72 m = the label's x1), pooled
      over cells, for all frames and split by "the frame's token has a u400 DAC = 0" vs "none". K2: IoU in [0,40) m, all frames, < 0.80  => K2 fires.
  P2  SDF MAE (m) over the cells whose TRUE block-mean SDF lies within +-3 m of the boundary.
  P3  AUC of the minimum predicted SDF along each proposal's footprint corners (40 steps x 4 corners, 0.1 s grid) for NAVSIM DAC = 1 vs 0, pooled over the
      64 proposals x the C3 tokens, against the scorer's own DAC AUC on the SAME proposals. Decision: head AUC >= max(0.90, scorer AUC + 0.05) => H2 is
      evaluated; head AUC < scorer AUC => H2 is killed; in between is H2_PROVISIONAL (S1 ruling R3; the harness's old label was UNCOMMITTED_BAND).
  AMENDMENT A2 (PREREG_AUX_DRIVABLE_S1.md): the PAIRED FLOOR. `--pred-init ARM=<head_pred npz of that arm's INITIAL, never-trained head>` (same tokens, same ctx): the
      init head's pooled P3 AUC and the paired difference (trained - init) through the SAME log resamples as head-minus-scorer (same weights, same seed). H2_EVALUATED and
      H2_PROVISIONAL additionally need the (trained - init) 95 % LOWER bound > 0, else P3_AT_FLOOR; an arm with no init prediction reads FLOOR_UNREAD (never a pass);
      H2_KILLED is unchanged. The init head's P1 / P2 are reported beside P1 (no decision). The init npz must cover EXACTLY the trained npz's keys (FloorRefused).
SELECTION READS (from `--dump-scores`, per arm paired with the N run of the SAME seed: H2-sK minus N-sK, H1-sK minus N-sK)
  pick PDMS and the six sub-scores (E1 / DAC), the held-out analogue of E2 (DAC-zero share of the picks = -DAC delta), zero picks (E3 analogue), via
  rl_val_eval's own functions; the seed floor = the largest |delta| between two N seeds (read only when >= 3 N seeds exist, else the verdict is capped).

ESTIMATOR: paired LOG-CLUSTER bootstrap, 10,000 resamples, 95 %, seed 20260927 = proxy_eval.boot (the estimator of eval/stage3_full.py). Means and means of
differences go THROUGH proxy_eval.boot / rl_val_eval.read_pair. A RATIO of sums (IoU, MAE) or a pooled AUC cannot be a mean, so those use the SAME resampling
scheme (same generator, same seed, same draw order => the same log resamples for every arm and read); the test file pins that this scheme IS proxy_eval.boot where
both apply. It answers "would another draw of LOGS say this?" ONLY: not another training run (that is the N-vs-N seed floor), not another inference run.

TIER: T0-like. A head's perception on teacher-forced frames and a scorer's selection on banked held-out sets; never driving performance. The prereg's E1 / E2 /
E3 on the full navtest (needs the planner seam applying the feed) and the four families are NOT computed here (`not_computed` in the output).

INPUT CONTRACT the trainer does NOT yet satisfy (see raw/2026-10-13-aux-s1-readout/PATCH_head_pred_dump.diff): scorer_finetune.py writes the head's
per-frame outputs nowhere (only AuxEvalAcc aggregates and aux_drv.pt). The reader needs, per head arm, head_pred_<tag>.npz:
    log, token  str [N]      logit float [N, 40, 32]      sdf_m float [N, 40, 32] (metres)      optional step int [N] (rows with step != 0 are dropped and counted)
GUARDS: the label store must be the pinned train or navtest store (sha256 + key count); every evaluated key must be disjoint (key AND log) from the keys the
head was trained on (LeakRefused); an N arm cannot be a perception arm; a head arm without an N of its seed is UNPAIRED, never paired with another seed.

    python eval/aux_s1_readout.py perception --pred H2-s0=head_pred_update400.npz --pred-init H2-s0=head_pred_init_seed0.npz --labels-set navtest --dac-csv <u400 csv> --out p.json
    python eval/aux_s1_readout.py selection  --scores N-s0=.../scores_update400_B.npz H2-s0=... --pdm <heldout_pdm dir> --rule navsim_v1 --out s.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "refe"))
import proxy_eval as PE  # noqa: E402  -- the estimator (boot) the programme uses
import rl_val_eval as RV  # noqa: E402  -- selection reads: eval_set / read_pair / read_one / summarise
import rl_reward  # noqa: E402
import aux_drivable_head as AUX  # noqa: E402  -- the head's own target definition and its bilinear SDF lookup
import navsim_dac as ND  # noqa: E402  -- NAVSIM's 8 -> 41 pose interpolation

TOOL = "aux_s1_readout/1"
DUP_LOGIT_ATOL, DUP_SDF_ATOL_M = 1.0, 2.0      # A8: a duplicated t0 frame may differ by bf16 noise; more than one logit or one 2 m cell is a different frame
N_BOOT, SEED = 10000, 20260927
BANDS = ((0.0, 20.0), (20.0, 40.0), (40.0, 72.0))
BAND_NAMES = ("0-20", "20-40", "40-72")
K2_BAND = (0.0, 40.0)
K2_IOU_MIN = 0.80
P2_HALF_BAND_M = 3.0
P3_AUC_FLOOR, P3_AUC_MARGIN = 0.90, 0.05
SPLITS = ("all", "dac0", "dac_ok")
CELL_M = AUX.CELL_M
NCX = int(round((AUX.X1 - AUX.X0) / CELL_M))          # 40
NCY = int(round((AUX.Y1 - AUX.Y0) / CELL_M))          # 32
ART = "D:/Projects/TanitAD-artifacts/aux_drivable"
LABEL_PINS = {                                         # (store_sha256, stored_keys): the two stores, nothing else is readable
    "train": ("4a17db673b1548ba35724ab848a7dc551e4b0bc1e49c57ae93e6357454bf287a", 103139),
    "navtest": ("985a3bc77f3ab7a96b0dceace48e73b80e5b69b70710958635060939bce0d7c1", 12146),
}
LABEL_DIRS = {"train": f"{ART}/label_store", "navtest": f"{ART}/label_store_navtest"}
ARM_RE = re.compile(r"^(N|H1|H2)-s(\d+)$")
NOT_COMPUTED = ["E1 / E2 / E3 on the full navtest (PDMS, DAC-zero share, zeroed share): the planner / seam does not apply H2's feed (PREREG_BLANKS_PROPOSED section 5)",
                "the four metric families on navtest (same reason); STRATEGIC is PAUSED (PI 2026-09-19, PI_DECISION_QUEUE item 25)",
                "inference variance: REFe scoring is deterministic, ASSERTED by a second pass of the same checkpoint, not measured here"]


class LeakRefused(RuntimeError):
    """An evaluated frame belongs to the data the head was trained on (key or log overlap)."""


class FloorRefused(RuntimeError):
    """The initial-head prediction (the A2 floor) does not cover exactly the trained arm's (log, token) keys."""


# ================================================================================================================ geometry
def cell_x_centres() -> np.ndarray:
    """Forward distance (m, rear-axle frame) of each of the 40 cell columns [ix]."""
    return AUX.X0 + (np.arange(NCX) + 0.5) * CELL_M


def band_cells(lo: float, hi: float) -> np.ndarray:
    """bool [40, 32]: the cells whose centre has lo <= x < hi (the band varies along the FIRST axis, ix)."""
    xc = cell_x_centres()
    return np.broadcast_to(((xc >= lo) & (xc < hi))[:, None], (NCX, NCY)).copy()


# ================================================================================================================ bootstrap
def boot_weights(n_groups: int, n: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Multiplicity of each log in each resample, [n, G]. The draw is proxy_eval.boot's exactly: default_rng(seed), one integers(0, G, G) per resample, in
    order, groups = logs sorted by name (the caller's order)."""
    rng = np.random.default_rng(seed)
    W = np.empty((n, n_groups), np.float64)
    for b in range(n):
        W[b] = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
    return W


def _ci(reps):
    ok = np.isfinite(reps)
    if not ok.any():
        return [float("nan"), float("nan")], int((~ok).sum())
    return [float(np.percentile(reps[ok], 2.5)), float(np.percentile(reps[ok], 97.5))], int((~ok).sum())


def ratio_read(num_l: np.ndarray, den_l: np.ndarray, W: np.ndarray) -> dict:
    """ratio of per-log sums (IoU, MAE) with its log-cluster bootstrap."""
    d = float(den_l.sum())
    est = float(num_l.sum()) / d if d > 0 else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = (W @ num_l) / (W @ den_l)
    ci, n_nan = _ci(reps)
    return {"est": est, "ci95": ci, "n_den": d, "n_nan_resamples": n_nan}


def diff_read(num_a, den_a, num_b, den_b, W: np.ndarray) -> dict:
    """A - B, both ratios of sums, on the SAME resamples (paired)."""
    Wb = W
    da, db = float(den_a.sum()), float(den_b.sum())
    est = (float(num_a.sum()) / da - float(num_b.sum()) / db) if (da > 0 and db > 0) else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = (W @ num_a) / (W @ den_a) - (Wb @ num_b) / (Wb @ den_b)
    ci, n_nan = _ci(reps)
    return {"est": est, "ci95": ci, "n_nan_resamples": n_nan}


def by_log(logs, *arrays):
    """per-frame arrays -> per-log sums, logs sorted by name (PE.boot's group order). -> (sorted logs, [sums])"""
    ul = sorted(set(map(str, logs)))
    pos = {l: i for i, l in enumerate(ul)}
    inv = np.asarray([pos[str(l)] for l in logs], np.int64)
    return ul, [np.bincount(inv, weights=np.asarray(a, np.float64), minlength=len(ul)) for a in arrays]


# ================================================================================================================ AUC
def set_auc(score, passes):
    """within-set AUC: P(score of a pass > score of a fail), ties 1/2; None if one class only."""
    s = np.asarray(score, np.float64)
    p = np.asarray(passes, bool)
    pos, neg = s[p], s[~p]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def auc_cluster_stats(score, passes, cluster, n_groups):
    """pooled AUC decomposed over log pairs: S[a, b] = #(pass in a ranked above fail in b) + 1/2 ties, Nm[a, b] = n_pass_a * n_fail_b.
    A resample with log multiplicities w has pooled AUC (w S w) / (w Nm w) -- exact, no per-resample sort."""
    score = np.asarray(score, np.float64)
    passes = np.asarray(passes, bool)
    cluster = np.asarray(cluster, np.int64)
    pos = [np.sort(score[(cluster == g) & passes]) for g in range(n_groups)]
    neg = [np.sort(score[(cluster == g) & ~passes]) for g in range(n_groups)]
    S = np.zeros((n_groups, n_groups))
    Nm = np.zeros((n_groups, n_groups))
    for b in range(n_groups):
        if len(neg[b]) == 0:
            continue
        for a in range(n_groups):
            if len(pos[a]) == 0:
                continue
            lt = np.searchsorted(neg[b], pos[a], side="left").sum()
            le = np.searchsorted(neg[b], pos[a], side="right").sum()
            S[a, b] = lt + 0.5 * (le - lt)
            Nm[a, b] = len(pos[a]) * len(neg[b])
    return S, Nm


def auc_value(S, Nm, w=None):
    if w is None:
        w = np.ones(S.shape[0])
    d = float(w @ Nm @ w)
    return float(w @ S @ w) / d if d > 0 else float("nan")


def auc_read(S, Nm, W):
    est = auc_value(S, Nm)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = ((W @ S) * W).sum(1) / ((W @ Nm) * W).sum(1)
    ci, n_nan = _ci(reps)
    return {"est": est, "ci95": ci, "n_pairs": float(Nm.sum()), "n_nan_resamples": n_nan}


def auc_diff_read(Sa, Na, Sb, Nb, W):
    """AUC_a - AUC_b on the same rows (paired) and the same resamples."""
    est = auc_value(Sa, Na) - auc_value(Sb, Nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = ((W @ Sa) * W).sum(1) / ((W @ Na) * W).sum(1) - ((W @ Sb) * W).sum(1) / ((W @ Nb) * W).sum(1)
    ci, n_nan = _ci(reps)
    return {"est": est, "ci95": ci, "n_nan_resamples": n_nan}


# ================================================================================================================ inputs
def load_head_pred(path: str) -> dict:
    """head_pred_<tag>.npz -> {keys [(log, token)], logit [N,40,32] f32, sdf [N,40,32] f32, dropped_step_nonzero, duplicate_keys_identical,
    duplicate_keys_within_tol, duplicate_max_abs_diff [logit, sdf_m]}.  A duplicated key keeps its FIRST row (A8)."""
    z = np.load(path, allow_pickle=False)
    for k in ("log", "token", "logit", "sdf_m"):
        if k not in z.files:
            raise ValueError(f"{path}: missing array {k!r} (contract: log, token, logit, sdf_m)")
    log, tok = [str(x) for x in z["log"]], [str(x) for x in z["token"]]
    lg, sd = np.asarray(z["logit"], np.float32), np.asarray(z["sdf_m"], np.float32)
    if lg.shape != (len(log), NCX, NCY) or sd.shape != lg.shape or len(tok) != len(log):
        raise ValueError(f"{path}: need logit/sdf_m [N,{NCX},{NCY}] for N={len(log)}, got {lg.shape} / {sd.shape}")
    step = np.asarray(z["step"], np.int64) if "step" in z.files else np.zeros(len(log), np.int64)
    keep = step == 0                                     # the label is the frame at t0 of the token (SPEC 3.1): a later on-policy step is another frame
    seen, rows = {}, []
    dup_identical, dup_close, dmax_lg, dmax_sd = 0, 0, 0.0, 0.0
    for i in np.flatnonzero(keep):
        k = (log[i], tok[i])
        if k in seen:
            j = seen[k]
            if not (np.array_equal(lg[i], lg[j]) and np.array_equal(sd[i], sd[j])):
                # A8: the held-out bank holds two ranks of one t0 frame, and the FIRST copy is kept.  bf16 batch composition moves an output by ulps
                # (MEASURED 2026-10-15: one frame per arm, in the dump's final partial batch, <= 0.031 logit / 0.094 m); beyond one logit or one cell it is
                # a DIFFERENT frame under the same key, and that is refused.
                d_lg, d_sd = float(np.abs(lg[i] - lg[j]).max()), float(np.abs(sd[i] - sd[j]).max())
                if d_lg > DUP_LOGIT_ATOL or d_sd > DUP_SDF_ATOL_M:
                    raise ValueError(f"{path}: key {k} appears twice with DIFFERENT head outputs (max |d| logit {d_lg:.4g}, sdf {d_sd:.4g} m)")
                dup_close += 1
                dmax_lg, dmax_sd = max(dmax_lg, d_lg), max(dmax_sd, d_sd)
                continue
            dup_identical += 1
            continue
        seen[k] = i
        rows.append(i)
    rows = np.asarray(rows, np.int64)
    return {"keys": [(log[i], tok[i]) for i in rows], "logit": lg[rows], "sdf": sd[rows], "dropped_step_nonzero": int((~keep).sum()),
            "duplicate_keys_identical": dup_identical, "duplicate_keys_within_tol": dup_close, "duplicate_max_abs_diff": [dmax_lg, dmax_sd]}


def align_init_pred(trained: dict, init: dict) -> dict:
    """A2 guard: the init head's prediction must cover EXACTLY the trained arm's (log, token) keys -- neither fewer (a floor on other frames is not a floor) nor more.
    Returns the init prediction re-ordered to the trained arm's row order (so P1 / P3 are read row for row on the same frames)."""
    tk, ik = trained["keys"], init["keys"]
    st, si = set(tk), set(ik)
    if st != si:
        miss, extra = sorted(st - si), sorted(si - st)
        raise FloorRefused(f"the initial-head prediction must cover exactly the trained arm's keys: {len(miss)} trained keys are missing from it (first {miss[:1]}), "
                           f"it has {len(extra)} keys the trained arm lacks (first {extra[:1]})")
    pos = {k: i for i, k in enumerate(ik)}
    rows = np.asarray([pos[k] for k in tk], np.int64)
    return {"keys": list(tk), "logit": init["logit"][rows], "sdf": init["sdf"][rows], "dropped_step_nonzero": init["dropped_step_nonzero"],
            "duplicate_keys_identical": init["duplicate_keys_identical"], "duplicate_keys_within_tol": init.get("duplicate_keys_within_tol", 0),
            "duplicate_max_abs_diff": init.get("duplicate_max_abs_diff", [0.0, 0.0])}


def open_labels(label_set: str, store: str | None = None):
    sha, n = LABEL_PINS[label_set]
    return AUX.AuxLabelSource(store or LABEL_DIRS[label_set], pinned_sha256=sha, pinned_keys=n)


def load_dac_zero_tokens(csv_path: str) -> set:
    """tokens of a NAVSIM per-token csv whose drivable_area_compliance == 0 (valid rows; the 'average' row is skipped)."""
    out = set()
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("token") in (None, "", "average"):
                continue
            if float(r["drivable_area_compliance"]) == 0.0:
                out.add(r["token"])
    return out


def assert_disjoint(eval_keys, train_keys, heldout_logs=None):
    """Refuse an evaluation that touches the head's training data. eval_keys / train_keys: iterables of (log, token). heldout_logs: when the eval frames come from the
    TRAIN store's held-out logs, every evaluated log must be a held-out log and no held-out log may hold a training key."""
    ek, tk = set(eval_keys), set(train_keys)
    key_ov = ek & tk
    log_ov = {k[0] for k in ek} & {k[0] for k in tk}
    if key_ov or log_ov:
        raise LeakRefused(f"{len(key_ov)} evaluated (log, token) keys and {len(log_ov)} evaluated logs are in the training keys (first: "
                          f"{sorted(key_ov)[:1] or sorted(log_ov)[:1]})")
    if heldout_logs is not None:
        outside = {k[0] for k in ek} - set(heldout_logs)
        if outside:
            raise LeakRefused(f"{len(outside)} evaluated logs are not held-out logs (first: {sorted(outside)[0]})")


def train_keys_for(label_set: str, train_source, heldout_logs=None):
    """The keys the head was trained on. navtest read: EVERY train-store key. held-out read from the train store: the train-store keys of the logs that are not held out."""
    keys = train_source.all_keys()
    if label_set == "navtest":
        return keys
    if heldout_logs is None:
        raise ValueError("a held-out read from the train store needs --heldout-logs")
    return [k for k in keys if k[0] not in set(heldout_logs)]


# ================================================================================================================ P1 / P2 per-frame statistics
def frame_stats(pred: dict, source, chunk: int = 256) -> dict:
    """Per-frame sufficient statistics of one head arm against a label store. Arrays are [N] (frames in pred order)."""
    n = len(pred["keys"])
    bands = list(BANDS) + [K2_BAND, None]                  # None = every cell (the trainer's aux_eval pooling, a cross-check)
    out = {"valid": np.zeros(n, bool)}
    for b in range(len(bands)):
        out[f"inter{b}"], out[f"union{b}"], out[f"cells{b}"] = (np.zeros(n) for _ in range(3))
    out["p2_abs"], out["p2_n"], out["all_abs"], out["all_n"] = (np.zeros(n) for _ in range(4))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        lab, valid = source.labels(pred["keys"][s:e])
        t = AUX.build_targets(lab, valid)
        tm, kn = t.mask_frac.numpy() >= 0.5, t.known.numpy()
        pm = pred["logit"][s:e] > 0
        true_sdf = t.sdf_m.numpy().astype(np.float64)
        out["valid"][s:e] = valid.numpy()
        for b, band in enumerate(bands):
            k = kn & (band_cells(*band)[None] if band is not None else True)
            out[f"inter{b}"][s:e] = (tm & pm & k).sum((1, 2))
            out[f"union{b}"][s:e] = ((tm | pm) & k).sum((1, 2))
            out[f"cells{b}"][s:e] = k.sum((1, 2))
        err = np.abs(pred["sdf"][s:e].astype(np.float64) - true_sdf)
        near = kn & (np.abs(true_sdf) <= P2_HALF_BAND_M)
        out["p2_abs"][s:e], out["p2_n"][s:e] = (err * near).sum((1, 2)), near.sum((1, 2))
        out["all_abs"][s:e], out["all_n"][s:e] = (err * kn).sum((1, 2)), kn.sum((1, 2))
    return out


def split_masks(keys, dac_zero_tokens):
    toks = np.asarray([k[1] for k in keys])
    if dac_zero_tokens is None:
        return {"all": np.ones(len(keys), bool)}
    z = np.asarray([t in dac_zero_tokens for t in toks], bool)
    return {"all": np.ones(len(keys), bool), "dac0": z, "dac_ok": ~z}


def perception_reads(pred: dict, st: dict, dac_zero_tokens=None, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """P1 (bands x splits), P2 and K2 for one head arm. Frames without a label are excluded and counted."""
    keys = pred["keys"]
    sm = split_masks(keys, dac_zero_tokens)
    use = st["valid"]
    logs_all = [k[0] for k in keys]
    ul = sorted(set(l for l, u in zip(logs_all, use) if u))
    W = boot_weights(len(ul), n_boot, seed)
    names = list(BAND_NAMES) + ["0-40", "all-cells"]
    res = {"n_frames": len(keys), "n_labelled": int(use.sum()), "n_unlabelled": int((~use).sum()), "n_logs": len(ul), "p1": {}}
    for sname, msk in sm.items():
        sel = use & msk
        res["p1"][sname] = {"n_frames": int(sel.sum())}
        for b, nm in enumerate(names):
            lg = [l for l, s in zip(logs_all, sel) if s]
            if not lg:
                res["p1"][sname][nm] = {"est": float("nan"), "ci95": [float("nan")] * 2, "n_cells": 0.0}
                continue
            ul_s, (inter, union, cells) = by_log(lg, st[f"inter{b}"][sel], st[f"union{b}"][sel], st[f"cells{b}"][sel])
            num, den = _align(ul, ul_s, inter), _align(ul, ul_s, union)
            r = ratio_read(num, den, W)
            r["n_cells"] = float(cells.sum())
            r["n_union"] = r.pop("n_den")
            res["p1"][sname][nm] = r
    lg = [l for l, s in zip(logs_all, use) if s]
    if lg:
        ul_s, (pa, pn, aa, an) = by_log(lg, st["p2_abs"][use], st["p2_n"][use], st["all_abs"][use], st["all_n"][use])
        p2 = ratio_read(_align(ul, ul_s, pa), _align(ul, ul_s, pn), W)
        p2["n_cells"] = p2.pop("n_den")
        res["p2"] = {"near_boundary_mae_m": p2, "half_band_m": P2_HALF_BAND_M,
                     "overall_mae_m": {k: v for k, v in ratio_read(_align(ul, ul_s, aa), _align(ul, ul_s, an), W).items() if k != "n_den"}}
    else:
        res["p2"] = None
    res["k2"] = k2_decision(res["p1"]["all"]["0-40"])
    return res


def _align(ul_full, ul_part, vec):
    """per-log vector over ul_part -> over ul_full (zeros for a log absent from the part)."""
    pos = {l: i for i, l in enumerate(ul_part)}
    return np.asarray([vec[pos[l]] if l in pos else 0.0 for l in ul_full], np.float64)


def k2_decision(read_0_40: dict) -> dict:
    iou = read_0_40.get("est")
    if iou is None or not math.isfinite(iou):
        return {"decision": "K2_UNDEFINED", "iou_0_40": iou, "bar": K2_IOU_MIN, "note": "no labelled cell in [0, 40) m"}
    fires = iou < K2_IOU_MIN
    return {"decision": "K2_FIRES" if fires else "K2_CLEARED", "iou_0_40": iou, "ci95": read_0_40["ci95"], "bar": K2_IOU_MIN,
            "note": "decided on the point estimate over ALL labelled frames, as the prereg words it; the interval is context"}


def head_fit_diff(pred_a: dict, st_a: dict, pred_b: dict, st_b: dict, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """arm A minus arm B head fit (all frames; same frames required), paired on the same log resamples: P1 bands and P2."""
    if pred_a["keys"] != pred_b["keys"]:
        raise ValueError("head_fit_diff needs the SAME frames in the SAME order for both arms")
    use = st_a["valid"] & st_b["valid"]
    logs = [k[0] for k, u in zip(pred_a["keys"], use) if u]
    ul = sorted(set(logs))
    W = boot_weights(len(ul), n_boot, seed)
    out = {}
    for b, nm in enumerate(list(BAND_NAMES) + ["0-40", "all-cells"]):
        _u, (ia, ua, ib, ub) = by_log(logs, st_a[f"inter{b}"][use], st_a[f"union{b}"][use], st_b[f"inter{b}"][use], st_b[f"union{b}"][use])
        out[f"iou_{nm}"] = diff_read(ia, ua, ib, ub, W)
    _u, (pa, na, pb, nb) = by_log(logs, st_a["p2_abs"][use], st_a["p2_n"][use], st_b["p2_abs"][use], st_b["p2_n"][use])
    out["near_boundary_mae_m"] = diff_read(pa, na, pb, nb, W)
    return out


# ================================================================================================================ P3
def corner_min_sdf(sdf_pred: np.ndarray, props8: np.ndarray) -> tuple:
    """sdf_pred [N,40,32] (m), props8 [N,K,8,3] NAVSIM-representation rear-axle poses at 0.5..4.0 s in the ego frame
    -> (min over the 40 steps x 4 corners of the predicted SDF [N,K], share of sampled corners outside the head's extent [N,K]).
    Steps 0.1 .. 4.0 s (t = 0 excluded, as AUX.feed_scalars), bilinear lookup AUX.sample_pred_sdf; an out-of-extent corner reads +6 m (unknown)."""
    import torch
    N, K = props8.shape[:2]
    rel = ND.at_10hz(np.asarray(props8, np.float64))[..., 1:, :]                 # [N,K,40,3]
    corners = AUX.FL.footprint_corners(torch.from_numpy(rel).float())            # [N,K,40,4,2]
    pts = corners.reshape(N, K * 160, 2)
    s, inb = AUX.sample_pred_sdf(torch.from_numpy(np.asarray(sdf_pred, np.float32)), pts)
    s, inb = s.reshape(N, K, 160), inb.reshape(N, K, 160)
    return s.min(-1).values.numpy().astype(np.float64), (1.0 - inb.float().mean(-1)).numpy().astype(np.float64)


def p3_decision(auc_head: float, auc_scorer: float, floor_lo: float | None = None) -> dict:
    """The P3 decision (S1 prereg 3 R2/R3, Amendment A2). `floor_lo` = the 95 % LOWER bound of the paired pooled-AUC difference (trained head - its initial head);
    None = the arm's init prediction was not read. The 0.90 / scorer + 0.05 thresholds and 'head < scorer => killed' are decided on the point estimates, unchanged;
    the floor is an ADDED condition on the two outcomes that would promote H2."""
    thr = max(P3_AUC_FLOOR, auc_scorer + P3_AUC_MARGIN)
    if auc_head >= thr:
        base = "H2_EVALUATED"
    elif auc_head < auc_scorer:
        base = "H2_KILLED"
    else:
        base = "H2_PROVISIONAL"                          # R3: between the scorer's AUC and the threshold (the harness's old label was UNCOMMITTED_BAND)
    d = base
    if base != "H2_KILLED":
        if floor_lo is None:
            d = "FLOOR_UNREAD"
        elif not (floor_lo > 0.0):                       # also true for NaN: a floor that cannot be read is not cleared
            d = "P3_AT_FLOOR"
    return {"decision": d, "decision_without_floor": base, "auc_head": auc_head, "auc_scorer": auc_scorer, "evaluate_threshold": thr, "floor_lower_bound": floor_lo,
            "note": "thresholds decided on the point estimates; H2_EVALUATED / H2_PROVISIONAL additionally need the paired (trained - init) 95 % lower bound > 0 (A2.1), "
                    "else P3_AT_FLOOR; no init prediction => FLOOR_UNREAD, never a pass; H2_KILLED is unchanged"}


def p3_read(head_sdf: np.ndarray, props8: np.ndarray, dac: np.ndarray, valid: np.ndarray, scorer_dac_logit: np.ndarray, tok_logs, n_boot: int = N_BOOT,
            seed: int = SEED, init_sdf: np.ndarray | None = None) -> dict:
    """head_sdf [N,40,32]; props8 [N,K,8,3]; dac [N,K] in {0,1}; valid [N,K] bool; scorer_dac_logit [N,K]; tok_logs: log of each of the N tokens.
    init_sdf [N,40,32] (optional): the SAME arm's head at initialisation on the SAME tokens (A2.1 floor); without it the decision reads FLOOR_UNREAD."""
    if init_sdf is not None and np.shape(init_sdf) != np.shape(head_sdf):
        raise ValueError(f"init_sdf {np.shape(init_sdf)} must have the trained head's shape {np.shape(head_sdf)} (same tokens, same order)")
    dac = np.asarray(dac, np.float64)
    if not np.isin(dac[valid], (0.0, 1.0)).all():
        raise ValueError("DAC labels must be exactly 0.0 or 1.0 on the valid proposals")
    head_score, oob = corner_min_sdf(head_sdf, props8)
    N, K = dac.shape
    ul = sorted(set(map(str, tok_logs)))
    pos = {l: i for i, l in enumerate(ul)}
    cl = np.repeat(np.asarray([pos[str(l)] for l in tok_logs], np.int64)[:, None], K, axis=1)
    m = np.asarray(valid, bool)
    passes = dac[m] == 1.0
    Sh, Nh = auc_cluster_stats(head_score[m], passes, cl[m], len(ul))
    Ss, Ns = auc_cluster_stats(np.asarray(scorer_dac_logit, np.float64)[m], passes, cl[m], len(ul))
    W = boot_weights(len(ul), n_boot, seed)
    head, scorer, diff = auc_read(Sh, Nh, W), auc_read(Ss, Ns, W), auc_diff_read(Sh, Nh, Ss, Ns, W)
    floor, floor_lo = {"status": "FLOOR_UNREAD", "reason": "no initial-head prediction was given for this arm (--pred-init)"}, None
    if init_sdf is not None:                                                    # A2.1: the SAME proposals, tokens, clusters, weights and seed as the reads above
        init_score, init_oob = corner_min_sdf(init_sdf, props8)
        Si, Ni = auc_cluster_stats(init_score[m], passes, cl[m], len(ul))
        floor_diff = auc_diff_read(Sh, Nh, Si, Ni, W)
        floor = {"status": "READ", "init": auc_read(Si, Ni, W), "trained_minus_init": floor_diff, "init_minus_scorer": auc_diff_read(Si, Ni, Ss, Ns, W),
                 "init_corner_oob_share_mean": float(init_oob[m].mean()) if m.any() else None,
                 "estimator": f"paired log-cluster bootstrap, {n_boot} resamples, seed {seed}: the SAME resamples as head_minus_scorer"}
        floor_lo = floor_diff["ci95"][0]
    # mean within-set AUC (SUPPLEMENTARY: the prereg's words are 'over the 64 proposals x the C3 tokens', i.e. the pooled read above)
    wh, ws, tl = {}, {}, {}
    for i in range(N):
        a_h, a_s = set_auc(head_score[i][valid[i]], dac[i][valid[i]] == 1.0), set_auc(np.asarray(scorer_dac_logit)[i][valid[i]], dac[i][valid[i]] == 1.0)
        if a_h is not None and a_s is not None:
            wh[str(i)], ws[str(i)], tl[str(i)] = a_h, a_s, str(tok_logs[i])
    within = None
    if wh:
        mh, lh, hh = PE.boot(wh, tl, n=n_boot, seed=seed)
        ms, ls, hs = PE.boot(ws, tl, n=n_boot, seed=seed)
        md, ld, hd = PE.boot({k: wh[k] - ws[k] for k in wh}, tl, n=n_boot, seed=seed)
        within = {"head": {"est": mh, "ci95": [lh, hh]}, "scorer": {"est": ms, "ci95": [ls, hs]}, "head_minus_scorer": {"est": md, "ci95": [ld, hd]},
                  "n_sets_with_both_classes": len(wh), "note": "mean of per-token AUCs; supplementary, NOT the committed read"}
    return {"pooled": {"head": head, "scorer": scorer, "head_minus_scorer": diff, "n_proposals": int(m.sum()), "n_pass": int(passes.sum()),
                       "n_fail": int((~passes).sum()), "n_tokens": N, "n_logs": len(ul)},
            "within_set": within, "head_corner_oob_share_mean": float(oob[m].mean()) if m.any() else None, "floor": floor,
            "decision": p3_decision(head["est"], scorer["est"], floor_lo), "frames_note": "failure-enriched set (529 u400-zeroed + 529 random): a DIAGNOSTIC, never a rate"}


# ================================================================================================================ pairing by seed + selection
def parse_arm(name: str) -> tuple:
    m = ARM_RE.match(name)
    if not m:
        raise ValueError(f"arm name {name!r} is not N-s<k> / H1-s<k> / H2-s<k>")
    return m.group(1), int(m.group(2))


def pair_with_n(names) -> dict:
    """{head arm: the N arm of the SAME seed, or None}. An N arm is never in the keys; a head arm is never paired with another seed."""
    parsed = {n: parse_arm(n) for n in names}
    n_by_seed = {s: n for n, (c, s) in parsed.items() if c == "N"}
    return {n: n_by_seed.get(s) for n, (c, s) in sorted(parsed.items()) if c != "N"}


def eligible_sets(arm_scores: dict, pdm: dict) -> list:
    """sets scored by every arm that have a PDM record from the SAME checkpoint (a stale record is dropped from EVERY arm, rl_val_eval's rule)."""
    common = set.intersection(*[set(a) for a in arm_scores.values()])
    ks = []
    for k in sorted(common):
        p = pdm.get(k)
        if p is not None and {a[k]["ckpt"] for a in arm_scores.values()} == {p["ckpt"]}:
            ks.append(k)
    return ks


def selection_pair(head_scores: dict, n_scores: dict, pdm: dict, rule: str, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """head arm MINUS the N arm of its seed on the same held-out sets: PDMS + six sub-scores + R through rl_val_eval.read_pair; plus the held-out analogues of
    E2 (DAC-zero share of the picks) and E3 (zero picks). Self-pair and swap are re-checked on the real inputs."""
    ks = eligible_sets({"head": head_scores, "n": n_scores}, pdm)
    if not ks:
        raise ValueError("no eligible set (no common key with a PDM record of the same checkpoint)")
    cfg = rl_reward.RewardCfg(lam_d=0.0, w_prog=0.5, guard=True, use_k=False, use_l=False)

    def per(sc):
        return {k: RV.eval_set(sc[k], {"tg": pdm[k]["tg"], "terms": None, "L": None}, rule, cfg) for k in ks}

    ph, pn = per(head_scores), per(n_scores)
    fwd = RV.read_pair(ph, pn, n_boot, seed)
    rev = RV.read_pair(pn, ph, n_boot, seed)
    d = fwd["quantities"]["DAC"]
    e2 = {"mean_pp": 0.0 - d["mean"], "ci95_pp": [-d["ci95"][1], -d["ci95"][0]], "note": "DAC-zero share of the picks, head minus N, in percentage points = -(DAC delta x100); held-out analogue of E2, T0-like"}
    return {"n_sets": len(ks), "n_logs": len({k[0] for k in ks}), "paired_vs_n": fwd, "e2_dac_zero_share_delta": e2,
            "e3_zero_picks": RV.read_one(ph, pn, ks, "zero", n_boot, seed),
            "arms": {"head": RV.summarise(ph), "n": RV.summarise(pn)},
            "selfcheck": {"selfpair": RV.check_selfpair(RV.read_pair(pn, pn, n_boot, seed)), "swap": RV.check_swap(fwd, rev)}}


def seed_floor(pair_means: dict, n_seeds: int) -> dict:
    """pair_means: {quantity: [delta between two N seeds, ...]} -> the floor per quantity (largest |delta|). Measured only with >= 3 N seeds (3 pairs)."""
    floor = {q: max(abs(x) for x in v) for q, v in pair_means.items() if v}
    return {"floor": floor, "n_n_seeds": n_seeds, "measured": n_seeds >= 3}


def classify(read: dict, floor: dict | None, q: str) -> str:
    lo, hi = read["ci95"]
    if hi < 0:
        base = "adverse"
    elif lo > 0:
        base = "separated"
    else:
        base = "not_separated"
    if base == "separated":
        if floor is None or not floor["measured"]:
            return "separated, single seed, not yet a lever claim"
        return "separated and above the seed floor" if lo > floor["floor"].get(q, float("inf")) else "separated but within the seed floor"
    return base


def selection_reads(scores: dict, pdm: dict, rule: str, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """scores: {arm name: rl_val_eval.load_scores dict}. Every head arm is paired with the N arm of its seed; the N-vs-N deltas give the floor."""
    pairs = pair_with_n(scores)
    out = {"rule": rule, "pairs": {}, "unpaired_head_arms": sorted(h for h, n in pairs.items() if n is None)}
    n_names = sorted(n for n in scores if parse_arm(n)[0] == "N")
    pm: dict = {}
    nn = {}
    for i, a in enumerate(n_names):
        for b in n_names[i + 1:]:
            r = selection_pair(scores[b], scores[a], pdm, rule, n_boot, seed)
            nn[f"{b}-minus-{a}"] = {q: r["paired_vs_n"]["quantities"][q]["mean"] for q in ("PDMS", "DAC", "EP", "TTC", "NC")}
            for q, v in nn[f"{b}-minus-{a}"].items():
                pm.setdefault(q, []).append(v)
    out["seed_floor"] = dict(seed_floor(pm, len(n_names)), n_vs_n_deltas=nn)
    for h, n in pairs.items():
        if n is None:
            continue
        r = selection_pair(scores[h], scores[n], pdm, rule, n_boot, seed)
        r["classification"] = {q: classify(r["paired_vs_n"]["quantities"][q], out["seed_floor"], q) for q in ("PDMS", "DAC")}
        out["pairs"][f"{h}-minus-{n}"] = r
    return out


# ================================================================================================================ CLI
def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cmd_perception(a) -> int:
    arms = {}
    for spec in a.pred:
        name, _, path = spec.partition("=")
        cls, _seed = parse_arm(name)
        if cls == "N":
            print(f"REFUSED: {name} is a no-head arm; it has no perception read", flush=True)
            return 4
        arms[name] = path
    init_paths = {}
    for spec in (a.pred_init or []):
        name, _, path = spec.partition("=")
        cls, _seed = parse_arm(name)
        if cls == "N" or name not in arms:
            print(f"REFUSED: --pred-init {name}: {'a no-head arm has no floor' if cls == 'N' else 'no --pred arm of that name (a floor belongs to a trained arm)'}", flush=True)
            return 4
        init_paths[name] = path
    src = open_labels(a.labels_set, a.labels)
    heldout = [l.strip() for l in open(a.heldout_logs, encoding="utf-8") if l.strip()] if a.heldout_logs else None
    train_src = open_labels("train", a.train_store) if a.labels_set == "navtest" or a.heldout_logs else None
    dz = load_dac_zero_tokens(a.dac_csv) if a.dac_csv else None
    res = {"tool": TOOL, "tier": "T0-like: head perception on teacher-forced frames, never driving performance", "labels": {
        "set": a.labels_set, "store_sha256": src.manifest["store_sha256"], "frames": src.manifest["counts"]["stored_keys"]},
        "estimator": f"paired log-cluster bootstrap, {a.n_boot} resamples, 95 %, seed {SEED}; proxy_eval.boot's draw scheme. Answers 'another draw of LOGS' only",
        "dac_zero_tokens": None if dz is None else len(dz), "arms": {}, "not_computed": NOT_COMPUTED}
    preds, stats, inits = {}, {}, {}
    for name, path in sorted(arms.items()):
        p = load_head_pred(path)
        if train_src is not None:
            try:
                assert_disjoint(p["keys"], train_keys_for(a.labels_set, train_src, heldout), heldout if a.labels_set == "train" else None)
            except LeakRefused as e:
                print(f"ZZAUXS1_REFUSED {name}: {e}", flush=True)
                return 3
        elif a.labels_set == "train":
            print("REFUSED: a held-out read from the train store needs --heldout-logs", flush=True)
            return 4
        st = frame_stats(p, src)
        preds[name], stats[name] = p, st
        r = perception_reads(p, st, dz, a.n_boot)
        r["input"] = {"pred_sha256": sha256_file(path), "dropped_step_nonzero": p["dropped_step_nonzero"], "duplicate_keys_identical": p["duplicate_keys_identical"],
                      "duplicate_keys_within_tol": p.get("duplicate_keys_within_tol", 0), "duplicate_max_abs_diff": p.get("duplicate_max_abs_diff", [0.0, 0.0])}
        if name in init_paths:                       # A2: the arm's own head at initialisation, on exactly the same keys
            try:
                ip = align_init_pred(p, load_head_pred(init_paths[name]))
            except FloorRefused as e:
                print(f"ZZAUXS1_REFUSED {name}: {e}", flush=True)
                return 3
            ist = frame_stats(ip, src)
            fl = perception_reads(ip, ist, dz, a.n_boot)
            inits[name] = ip
            r["floor"] = {"status": "READ", "init_pred_sha256": sha256_file(init_paths[name]), "p1": fl["p1"], "p2": fl["p2"],
                          "trained_minus_init": head_fit_diff(p, st, ip, ist, a.n_boot),
                          "note": "A2.2: the initial head's P1 / P2 on the same frames, reported beside P1; no decision. K2's 0.80 is unchanged"}
        else:
            r["floor"] = {"status": "FLOOR_UNREAD", "reason": "no --pred-init for this arm"}
        res["arms"][name] = r
    if a.c3_table and a.c3_tokens:
        tab, toks = np.load(a.c3_table, allow_pickle=False), json.load(open(a.c3_tokens, encoding="utf-8"))
        names = [str(x) for x in tab["sub_names"]]
        di = next(i for i, n in enumerate(names) if "drivable" in n.lower() or n.upper() == "DAC")
        tok_list = [str(t) for t in tab["token"]]
        for name, p in preds.items():
            idx = {k[1]: i for i, k in enumerate(p["keys"])}
            miss = [t for t in tok_list if t not in idx]
            if miss:
                res["arms"][name]["p3"] = {"status": "NOT_COMPUTED", "reason": f"{len(miss)} of {len(tok_list)} C3 tokens are not in this arm's head predictions"}
                continue
            sel = np.asarray([idx[t] for t in tok_list])
            res["arms"][name]["p3"] = p3_read(p["sdf"][sel], tab["proposals"], tab["sub"][..., di], tab["valid"], tab["logits"][..., 1],
                                              [toks["token_log"][t] for t in tok_list], a.n_boot, init_sdf=inits[name]["sdf"][sel] if name in inits else None)
    else:
        for name in res["arms"]:
            res["arms"][name]["p3"] = {"status": "NOT_COMPUTED", "reason": "no --c3-table / --c3-tokens given"}
    res["head_fit_pairs"] = {}
    for h2 in sorted(n for n in arms if parse_arm(n)[0] == "H2"):
        h1 = f"H1-s{parse_arm(h2)[1]}"
        if h1 in preds:
            res["head_fit_pairs"][f"{h2}-minus-{h1}"] = head_fit_diff(preds[h2], stats[h2], preds[h1], stats[h1], a.n_boot)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=_json)
    print(f"ZZAUXS1_PERCEPTION_OK arms {sorted(res['arms'])} floor_read {sorted(inits)} floor_unread {sorted(set(res['arms']) - set(inits))}", flush=True)
    return 0


def cmd_selection(a) -> int:
    scores, hashes = {}, {}
    for spec in a.scores:
        name, _, path = spec.partition("=")
        parse_arm(name)
        scores[name] = RV.load_scores(path)
        hashes[name] = sha256_file(path)
    pdm, pst = RV.load_pdm(a.pdm)
    try:
        res = selection_reads(scores, pdm, a.rule, a.n_boot)
    except ValueError as e:
        print(f"ZZAUXS1_REFUSED {e}", flush=True)
        return 3
    res.update({"tool": TOOL, "tier": "T0-like: banked held-out sets, the scorer's pick looked up in frozen PDM labels; a selection read, never driving performance",
                "estimator": f"paired log-cluster bootstrap, {a.n_boot} resamples, 95 %, seed {SEED}; rl_val_eval.read_pair -> proxy_eval.boot. Answers 'another draw of LOGS' only",
                "inputs": {"scores_sha256": hashes, "pdm_dir": os.path.basename(os.path.normpath(a.pdm)), "pdm_stats": pst}, "not_computed": NOT_COMPUTED})
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=_json)
    ok = all(v["selfcheck"]["selfpair"]["ok"] and v["selfcheck"]["swap"]["ok"] for v in res["pairs"].values())
    print(f"ZZAUXS1_SELECTION_{'OK' if ok else 'SELFCHECK_FAILED'} pairs {sorted(res['pairs'])} unpaired {res['unpaired_head_arms']}", flush=True)
    return 0 if ok else 5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("perception")
    p.add_argument("--pred", nargs="+", required=True, help="ARM=head_pred.npz (ARM = H1-s<k> | H2-s<k>)")
    p.add_argument("--pred-init", nargs="*", default=None, help="ARM=head_pred.npz of that arm's INITIAL (never-trained) head on the SAME tokens (A2 floor); an arm without one reads FLOOR_UNREAD")
    p.add_argument("--labels-set", choices=sorted(LABEL_PINS), required=True)
    p.add_argument("--labels", default=None, help="label store dir (default: the pinned D: artifact)")
    p.add_argument("--train-store", default=None)
    p.add_argument("--heldout-logs", default=None, help="file, one log per line (required for --labels-set train)")
    p.add_argument("--dac-csv", default=None, help="u400's per-token NAVSIM csv: enables the DAC-zero / none split")
    p.add_argument("--c3-table", default=None)
    p.add_argument("--c3-tokens", default=None)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_perception)
    p = sub.add_parser("selection")
    p.add_argument("--scores", nargs="+", required=True, help="ARM=scores_<tag>_B.npz (ARM = N-s<k> | H1-s<k> | H2-s<k>)")
    p.add_argument("--pdm", required=True)
    p.add_argument("--rule", required=True, choices=RV.RULES)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_selection)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
