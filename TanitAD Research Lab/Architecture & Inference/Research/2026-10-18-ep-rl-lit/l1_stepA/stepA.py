#!/usr/bin/env python3
"""L1 Step A: the OFF-PROPOSAL ADMISSION TEST of REFe's own scorer (PREREG_L1_SEARCH_ADMISSION.md sections 2-4, sha256 a76f62d8...).

Question (prereg section 0): is the scorer accurate OFF the proposal set, i.e. may a search climb it?  Nothing here searches.
The whole module is numpy-only and CPU-testable on synthetic tensors; the real model enters only through `score_fn` (a set -> logits callable), so the
GPU is behind a flag (`real_scorer.py --device cuda`, taken only through `launch_stepA.py`'s lock).

PIPELINE (each phase is a sub-command; every phase hashes its inputs into its output)
    sample    400 tokens stratified by log, seed 20260927, BEFORE any label is read          -> sample.json
    perturb   per token: 8 top-by-scorer + 8 random anchors x (4 speed scales + 2 lateral offsets) -> perts.npz  (executed plans + scorer inputs)
    score     scorer over each perturbed FULL 64-set (the anchor slot replaced)              -> scores.npz   (gates G2a / G2b / G2c live here)
    analyse   labels (CPU PDM labeller, `label_perturbations.py`) + scores -> A1..A5, log-cluster bootstrap, outcome O1/O2/O3 -> RESULT.json

DEGREES OF FREEDOM THE PREREG LEFT OPEN are fixed HERE and listed in PREREG_L1_AMENDMENT_1_PROPOSED.md (C1..C9); every one is recorded in RESULT.json.

Component order of a logit row (`train.ScorerBank.COMPONENTS`, planner.py:549): 0 NC, 1 DAC, 2 EP, 3 TTC, 4 C, 5 DDC.  The label tables use (nc, dac, ep, ttc, c, ddc).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

PREREG_SHA256 = "a76f62d8043e4cc9d6c73b35cf2884f27027a1e01bcd3deec658380cc58bd801"
SEED = 20260927
N_SETS = 400
N_BOOT = 10_000
SPEED_SCALES = (0.85, 0.92, 1.08, 1.15)
LAT_OFFSETS = (-0.3, 0.3)
V1_W = (5.0, 5.0, 2.0)                        # planner.py:520, rule navsim_v1: NC x DAC x (5 EP + 5 TTC + 2 C) / 12
N_TOP, N_RAND = 8, 8
THR_TRUE_PAIR = 0.05                          # a pair enters a concordance only if the TRUE value differs by >= this (prereg A1/A2)
THR_AGG_RISE = 0.02                           # "the scorer aggregate rises by >= 0.02" (A4/A5)
THR_PDMS_FALL = 0.10                          # "the true PDMS FALLS by >= 0.10" (A4)
GATE_A1, GATE_A4, GATE_A5 = 0.80, 0.10, 0.05
LAT_PROFILE = "ramp"                          # C3: offset ramps 0 -> d along the plan (sin^2); "const" is the alternative recorded in the amendment
TACTICAL_BEARING_RAD = 0.245                  # nav_command.GOAL_RULE_BEARING_RAD, applied to the plan's end point (HYPOTHESIS: a proxy for the manoeuvre class)
SUBS = ("nc", "dac", "ep", "ttc", "c", "ddc")
KIND_LON, KIND_LAT = 0, 1
T_STEPS = 20                                  # REFe's native plan: 20 poses at 0.2 s


# ============================================================================================ small helpers
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def require_lock_for_gpu(device) -> None:
    """GPU use only under launch_stepA.py (the dev-box lock): the launcher exports L1_GPU_LOCKED=1 to its child.  A bare `--device cuda` is refused (exit 4)."""
    if str(device).startswith("cuda") and os.environ.get("L1_GPU_LOCKED") != "1":
        raise SystemExit(4)


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _seed_for(token: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}|{token}".encode()).hexdigest()[:8], 16)


# ============================================================================================ G0: sampling + provenance
def sample_stratified(tokens, logs, n: int, seed: int = SEED):
    """n tokens stratified by log (C1): every log gets 1, the rest is allocated proportionally to (size - 1) by largest remainder (ties by log name), the draw inside a log
    is `default_rng(seed).choice(..., replace=False)` over the logs in sorted order.  Returns the tokens sorted.  Reads NO label."""
    tokens, logs = [str(t) for t in tokens], [str(l) for l in logs]
    if len(tokens) != len(logs) or len(set(tokens)) != len(tokens):
        raise ValueError("tokens must be unique and aligned with logs")
    by = {}
    for t, l in zip(tokens, logs):
        by.setdefault(l, []).append(t)
    names = sorted(by)
    if not len(names) <= n <= len(tokens):
        raise ValueError(f"n={n} must lie in [{len(names)} logs, {len(tokens)} tokens]")
    for l in names:
        by[l] = sorted(by[l])
    spare = n - len(names)
    room = {l: len(by[l]) - 1 for l in names}
    tot = sum(room.values())
    ideal = {l: (spare * room[l] / tot if tot else 0.0) for l in names}
    k = {l: 1 + int(math.floor(ideal[l])) for l in names}
    rest = n - sum(k.values())
    for l in sorted(names, key=lambda x: (-(ideal[x] - math.floor(ideal[x])), x)):
        if rest <= 0:
            break
        if k[l] < len(by[l]):
            k[l] += 1
            rest -= 1
    rng = np.random.default_rng(seed)
    out = []
    for l in names:
        pick = rng.choice(len(by[l]), size=k[l], replace=False)
        out += [by[l][int(i)] for i in sorted(pick)]
    if len(out) != n:
        raise AssertionError(f"allocation produced {len(out)} != {n}")
    return sorted(out)


def g0_check(fit_tokens, fit_logs, train_tokens, train_logs) -> dict:
    """G0 provenance (prereg section 2): the fit sets share NO LOG with the training manifest (the prereg's criterion); the token overlap is reported beside it."""
    ft, fl, tt, tl = set(map(str, fit_tokens)), set(map(str, fit_logs)), set(map(str, train_tokens)), set(map(str, train_logs))
    lo, to = sorted(fl & tl), sorted(ft & tt)
    return {"passed": len(lo) == 0, "logs_overlap": len(lo), "tokens_overlap": len(to), "n_fit_tokens": len(ft), "n_fit_logs": len(fl),
            "overlap_log_examples": lo[:3], "overlap_token_examples": to[:3]}


# ============================================================================================ the scorer's aggregate (planner.py:537, rule navsim_v1)
def aggregate(logits) -> np.ndarray:
    """NC x DAC x (5 EP + 5 TTC + 2 C) / 12 over sigmoid probabilities.  [..., 6] -> [...].  Equality with the LANDED function is pinned by test_stepA.py."""
    p = sigmoid(logits)
    w = np.asarray(V1_W, dtype=np.float64)
    return p[..., 0] * p[..., 1] * (p[..., 2] * w[0] + p[..., 3] * w[1] + p[..., 4] * w[2]) / w.sum()


# ============================================================================================ perturbation geometry (executed-plan space)
def repair_last_heading_np(traj):
    """planner.repair_last_heading (A7): the executed plan takes the previous pose's heading at the last pose."""
    out = np.array(traj, copy=True)
    out[..., -1, 2] = out[..., -2, 2]
    return out


def _path(traj):
    """[T,3] executed plan -> (P [T+1,3] with the ego origin prepended, S [T+1] cumulative arclength)."""
    P = np.concatenate([np.zeros((1, 3), dtype=np.float64), np.asarray(traj, dtype=np.float64)], 0)
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    return P, np.concatenate([[0.0], np.cumsum(seg)])


def scale_speed(traj_exec, s: float):
    """Speed-profile scale (C2): the SAME path, travelled s times as fast: pose i moves to arclength s * S_i along the original polyline (extrapolated along the last
    non-degenerate segment when s > 1).  Yaw is interpolated along the path and HELD beyond its end.  s == 1 returns the plan unchanged (bit for bit)."""
    P, S = _path(traj_exec)
    if S[-1] < 1e-6:
        return np.array(traj_exec, copy=True)
    seg = np.diff(S)
    last = int(np.nonzero(seg > 1e-12)[0][-1])
    out = np.empty((P.shape[0] - 1, 3), dtype=np.float64)
    for i in range(1, P.shape[0]):
        q = s * S[i]
        j = min(max(int(np.searchsorted(S, q, side="right")) - 1, 0), P.shape[0] - 2)
        if q > S[-1]:
            j = last
        L = S[j + 1] - S[j]
        f = 0.0 if L <= 1e-12 else (q - S[j]) / L
        out[i - 1, :2] = P[j, :2] + f * (P[j + 1, :2] - P[j, :2])
        out[i - 1, 2] = P[j, 2] + min(f, 1.0) * _wrap(P[j + 1, 2] - P[j, 2])
    return out


def lateral_offset(traj_exec, d: float, profile: str = LAT_PROFILE):
    """Lateral offset d metres (+ = left, the ego frame is x forward / y left; planner.py:184), applied along each pose's own left normal.  profile 'ramp' (C3):
    weight sin^2(pi/2 * i/T), 0 at the ego and 1 at the horizon; 'const': the full offset from the first pose.  Yaw is untouched (the heading change is < d*pi/2/path_length)."""
    P = np.asarray(traj_exec, dtype=np.float64)
    T = P.shape[0]
    i = np.arange(1, T + 1, dtype=np.float64)
    w = np.sin(0.5 * math.pi * i / T) ** 2 if profile == "ramp" else np.ones(T)
    out = np.array(P, copy=True)
    out[:, 0] = P[:, 0] + w * d * (-np.sin(P[:, 2]))
    out[:, 1] = P[:, 1] + w * d * np.cos(P[:, 2])
    return out


def make_perturbation(raw_anchor, kind: int, param: float, profile: str = LAT_PROFILE):
    """-> (exec_pert [T,3] float32: what the PDM labeller scores, scorer_in [T,3] float32: what the scorer reads).
    The scorer reads the model's RAW trajectory, whose last heading is corrupted (A7); a perturbation keeps the anchor's raw last heading so the scorer sees the SAME
    corruption it was trained on (C4).  Identity (s = 1, d = 0) therefore returns the raw anchor exactly."""
    raw = np.asarray(raw_anchor, dtype=np.float32)
    ex = repair_last_heading_np(raw).astype(np.float64)
    pert = scale_speed(ex, param) if kind == KIND_LON else lateral_offset(ex, param, profile)
    pert = repair_last_heading_np(pert)
    sc_in = np.array(pert, copy=True)
    sc_in[-1, 2] = float(raw[-1, 2])
    return pert.astype(np.float32), sc_in.astype(np.float32)


def perturbation_grid(speed_scales=SPEED_SCALES, lat_offsets=LAT_OFFSETS):
    return [(KIND_LON, float(s)) for s in speed_scales] + [(KIND_LAT, float(d)) for d in lat_offsets]


def command_class(exec_plan) -> int:
    """Tactical proxy: nav_command.goal_to_command's bearing rule on the plan's end point: 0 LEFT, 1 STRAIGHT, 2 RIGHT."""
    x, y = float(exec_plan[-1, 0]), float(exec_plan[-1, 1])
    ang = math.atan2(y, max(x, 1e-6))
    return 0 if ang > TACTICAL_BEARING_RAD else (2 if ang < -TACTICAL_BEARING_RAD else 1)


# ============================================================================================ anchors and full-set scoring
def choose_anchors(logits, token: str, seed: int = SEED, n_top: int = N_TOP, n_rand: int = N_RAND):
    """the scorer's top-`n_top` by aggregate (stable by index) + `n_rand` random OTHER proposals, so the anchor choice is not the scorer's own (prereg section 3).
    Returns [(index, 'top'|'rand')]; the random draw is a pure function of (seed, token)."""
    agg = aggregate(np.asarray(logits, dtype=np.float64))
    order = np.argsort(-agg, kind="stable")
    top = [int(i) for i in order[:n_top]]
    rest = [i for i in range(len(agg)) if i not in set(top)]
    rng = np.random.default_rng(_seed_for(token, seed))
    rnd = sorted(int(rest[i]) for i in rng.choice(len(rest), size=n_rand, replace=False))
    return [(i, "top") for i in top] + [(i, "rand") for i in rnd]


def score_replaced(score_fn, fan_raw, slot: int, scorer_in):
    """Score the COMPLETE set with proposal `slot` replaced by `scorer_in` (set-conditional scorer: model.py:699).  Returns the [K,6] logits of the whole set."""
    s = np.array(fan_raw, copy=True)
    s[slot] = scorer_in                                                    # <- the perturbation line (G3's mutation deletes it)
    return np.asarray(score_fn(s), dtype=np.float64)


# ============================================================================================ gates on the scoring path
def g2_identity(score_fn, fan_raw, slot: int) -> dict:
    """G2a: the unperturbed fan scored THROUGH the perturbation code path (identity perturbation in `slot`) equals the plain set score, bit for bit."""
    plain = np.asarray(score_fn(np.array(fan_raw, copy=True)), dtype=np.float64)
    ex, sc_in = make_perturbation(fan_raw[slot], KIND_LON, 1.0)
    via = score_replaced(score_fn, fan_raw, slot, sc_in)
    d = float(np.max(np.abs(plain - via)))
    return {"passed": bool(d == 0.0), "max_abs_diff": d, "input_identical": bool(np.array_equal(sc_in, np.asarray(fan_raw[slot], dtype=np.float32)))}


def g2_set_dependence(score_fn, fan_raw, drop: int) -> dict:
    """G2c: dropping 1 of 64 proposals must change the OTHER 63 scores (the set self-attention premise).  `premise_true` False => the caveat of prereg section 5 is withdrawn."""
    full = np.asarray(score_fn(np.array(fan_raw, copy=True)), dtype=np.float64)
    keep = [i for i in range(fan_raw.shape[0]) if i != drop]
    part = np.asarray(score_fn(np.array(fan_raw[keep], copy=True)), dtype=np.float64)
    d = float(np.max(np.abs(full[keep] - part)))
    return {"premise_true": bool(d > 0.0), "max_abs_change_on_other_63": d}


class Recorder:
    """wraps a score_fn and keeps every set it was asked to score: the spy of G3."""

    def __init__(self, fn):
        self.fn, self.sets = fn, []

    def __call__(self, s):
        self.sets.append(np.array(s, copy=True))
        return self.fn(s)


def g3_reachability(score_fn, fan_raw, slot: int, kind: int, param: float, profile: str = LAT_PROFILE) -> dict:
    """G3: the perturbation reaches `score_trajectories`: the set the scorer received differs from the fan in EXACTLY the anchor slot, and that slot is the scorer input
    of `make_perturbation` for the requested (kind, param)."""
    rec = Recorder(score_fn)
    _, sc_in = make_perturbation(fan_raw[slot], kind, param, profile)
    score_replaced(rec, fan_raw, slot, sc_in)
    got = rec.sets[-1]
    diff = [i for i in range(got.shape[0]) if not np.array_equal(got[i], np.asarray(fan_raw[i], dtype=np.float32))]
    ok = diff == [slot] and np.array_equal(got[slot], sc_in) and (param == 1.0 and kind == KIND_LON or not np.array_equal(got[slot], np.asarray(fan_raw[slot], np.float32)))
    return {"passed": bool(ok), "slots_changed": diff, "expected_slot": slot}


def g1_check(labelled, banked, tol: float = 1e-6) -> dict:
    """G1 label identity: the CPU labeller reproduces the banked PDM sub-scores of the in-fan proposals.  labelled/banked: [..., 6] (+ pdms via 'pdms' key if dicts)."""
    a, b = np.asarray(labelled, dtype=np.float64), np.asarray(banked, dtype=np.float64)
    d = float(np.max(np.abs(a - b)))
    return {"passed": bool(d <= tol and np.isfinite(d)), "max_abs_diff": d, "tol": tol}


# ============================================================================================ the pairs and the four-family admission metrics
def concordance_counts(s_a, s_p, t_a, t_p, thr: float = THR_TRUE_PAIR):
    """pairs whose TRUE value differs by >= thr: (concordant weight, n).  A scorer tie counts 0.5.  s_* are scorer values (monotone in the head), t_* the true values."""
    s_a, s_p, t_a, t_p = (np.asarray(x, dtype=np.float64) for x in (s_a, s_p, t_a, t_p))
    m = np.abs(t_p - t_a) >= thr - 1e-12
    ds, dt = np.sign(s_p - s_a), np.sign(t_p - t_a)
    conc = np.where(ds == 0, 0.5, (ds == dt).astype(np.float64))
    return conc * m, m.astype(np.float64)


def admission_terms(P: dict) -> dict:
    """per-pair (num, den) for every admission quantity.  P: dict of aligned arrays a_logit/p_logit [n,6], a_sub/p_sub [n,6] (nc,dac,ep,ttc,c,ddc), a_pdms/p_pdms [n]."""
    al, pl = np.asarray(P["a_logit"], np.float64), np.asarray(P["p_logit"], np.float64)
    asub, psub = np.asarray(P["a_sub"], np.float64), np.asarray(P["p_sub"], np.float64)
    apd, ppd = np.asarray(P["a_pdms"], np.float64), np.asarray(P["p_pdms"], np.float64)
    ag, pg = aggregate(al), aggregate(pl)
    rise = (pg - ag) >= THR_AGG_RISE - 1e-12
    fall = (ppd - apd) <= -THR_PDMS_FALL + 1e-12
    flip = ((asub[:, 0] == 1) & (psub[:, 0] == 0)) | ((asub[:, 1] == 1) & (psub[:, 1] == 0))     # true NC or DAC goes 1 -> 0
    T = {"A1": concordance_counts(al[:, 2], pl[:, 2], asub[:, 2], psub[:, 2]),
         "A2": concordance_counts(al[:, 3], pl[:, 3], asub[:, 3], psub[:, 3]),
         "A3": concordance_counts(ag, pg, apd, ppd),
         "A4": ((rise & fall).astype(np.float64), rise.astype(np.float64)),
         "A5": ((rise & flip).astype(np.float64), rise.astype(np.float64)),          # C6 primary reading: of the moves the scorer endorses, the share that breach NC/DAC
         "A5b": ((rise & flip).astype(np.float64), flip.astype(np.float64))}         # C6 alternative (descriptive): of the NC/DAC breaches, the share the scorer endorses
    return T


def per_log_sums(terms: dict, log_idx, n_logs: int, mask=None) -> dict:
    out = {}
    m = np.ones_like(np.asarray(log_idx), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    for k, (num, den) in terms.items():
        out[k] = (np.bincount(np.asarray(log_idx)[m], weights=np.asarray(num)[m], minlength=n_logs),
                  np.bincount(np.asarray(log_idx)[m], weights=np.asarray(den)[m], minlength=n_logs))
    return out


def boot_indices(n_logs: int, B: int = N_BOOT, seed: int = SEED):
    """the SAME resample draws for every quantity (paired log-cluster bootstrap, 10,000 resamples, seed 20260927)"""
    return np.random.default_rng(seed).integers(0, n_logs, size=(B, n_logs))


def ratio_ci(num_by_log, den_by_log, idx, alpha: float = 0.05) -> dict:
    """ratio of sums, resampling WHOLE LOGS.  It answers 'would another draw of LOGS say this' and nothing else (not another training seed, not another search seed)."""
    num, den = np.asarray(num_by_log, np.float64), np.asarray(den_by_log, np.float64)
    pt = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
    n, d = num[idx].sum(1), den[idx].sum(1)
    ok = d > 0
    r = n[ok] / d[ok]
    if r.size == 0:
        return {"point": pt, "lo": float("nan"), "hi": float("nan"), "n_valid_resamples": 0, "den": float(den.sum())}
    return {"point": pt, "lo": float(np.quantile(r, alpha / 2)), "hi": float(np.quantile(r, 1 - alpha / 2)), "n_valid_resamples": int(ok.sum()), "den": float(den.sum())}


def classify(a1: dict, a4: dict, a5: dict, mode: str = "interval_aware") -> str:
    """prereg section 4 -> 'O1' | 'O2' | 'O3'.  The prereg writes O2 by POINT estimates and O3 by INTERVALS, so a straddling gate can satisfy both (C7).
    interval_aware (proposed): O2 if a gate FAILS DECISIVELY (A1 hi < 0.80, A4 lo > 0.10, A5 point > 0.05); else O3 if the A1 or A4 interval contains its gate; else O1.
    literal: O2 on any point failure first, then O3, then O1."""
    s1 = a1["lo"] <= GATE_A1 <= a1["hi"]
    s4 = a4["lo"] <= GATE_A4 <= a4["hi"]
    point_fail = (a1["point"] < GATE_A1) or (a4["point"] > GATE_A4) or (a5["point"] > GATE_A5)
    if mode == "literal":
        return "O2" if point_fail else ("O3" if (s1 or s4) else "O1")
    decisive = (a1["hi"] < GATE_A1) or (a4["lo"] > GATE_A4) or (a5["point"] > GATE_A5)
    if decisive:
        return "O2"
    if s1 or s4:
        return "O3"
    return "O1"


# ============================================================================================ the run
def build_jobs(logits, token, seed=SEED, grid=None):
    grid = perturbation_grid() if grid is None else grid
    anchors = choose_anchors(logits, token, seed)
    return [(a, role, kind, par) for a, role in anchors for kind, par in grid]


def make_perts(bundle: dict, seed=SEED, grid=None, profile=LAT_PROFILE) -> dict:
    """phase `perturb`: bundle {token, log, traj_raw [N,64,20,3], logits [N,64,6]} -> exec/scorer-in plans and the job table (pure numpy, no scorer)."""
    N = len(bundle["token"])
    jobs0 = build_jobs(bundle["logits"][0], str(bundle["token"][0]), seed, grid)
    Q = len(jobs0)
    ex = np.zeros((N, Q, T_STEPS, 3), np.float32)
    si = np.zeros((N, Q, T_STEPS, 3), np.float32)
    anchor, kind, param, top = np.zeros((N, Q), np.int32), np.zeros((N, Q), np.int8), np.zeros((N, Q), np.float64), np.zeros((N, Q), bool)
    for n in range(N):
        for q, (a, role, k, p) in enumerate(build_jobs(bundle["logits"][n], str(bundle["token"][n]), seed, grid)):
            ex[n, q], si[n, q] = make_perturbation(bundle["traj_raw"][n, a], k, p, profile)
            anchor[n, q], kind[n, q], param[n, q], top[n, q] = a, k, p, role == "top"
    return {"exec": ex, "scorer_in": si, "anchor": anchor, "kind": kind, "param": param, "is_top": top, "profile": np.array(profile), "seed": np.array(seed)}


def run_scores(score_fn, bundle: dict, perts: dict, gate_tokens: int = 20) -> dict:
    """phase `score`: every perturbed set is scored COMPLETE, the anchor slot replaced.  Returns p_logit [N,Q,6], a_logit [N,64,6] and the gates."""
    N, Q = perts["anchor"].shape
    p_logit = np.zeros((N, Q, 6), np.float64)
    a_logit = np.zeros((N, 64, 6), np.float64)
    gates = {"G2a": [], "G2c": [], "G3": []}
    bind = getattr(score_fn, "bind", None)                                # a real scorer holds ONE token's visual_ctx at a time (real_scorer.py)
    for n in range(N):
        if bind is not None:
            bind(n)
        fan = np.asarray(bundle["traj_raw"][n], np.float32)
        a_logit[n] = np.asarray(score_fn(np.array(fan, copy=True)), np.float64)
        for q in range(Q):
            p_logit[n, q] = score_replaced(score_fn, fan, int(perts["anchor"][n, q]), perts["scorer_in"][n, q])[int(perts["anchor"][n, q])]
        if n < gate_tokens:
            slot = int(perts["anchor"][n, 0])
            gates["G2a"].append(g2_identity(score_fn, fan, slot))
            gates["G2c"].append(g2_set_dependence(score_fn, fan, slot))
            gates["G3"].append(g3_reachability(score_fn, fan, slot, int(perts["kind"][n, 0]), float(perts["param"][n, 0]), str(perts["profile"])))
    return {"p_logit": p_logit, "a_logit": a_logit, "gates": gates}


def assemble_pairs(bundle: dict, perts: dict, scores: dict, labels: dict, log_names) -> tuple[dict, np.ndarray, np.ndarray]:
    """flatten (token, job) into pairs.  Tokens with a non-finite perturbation label, or an invalid anchor label, are dropped (counted by the caller)."""
    N, Q = perts["anchor"].shape
    li = {l: i for i, l in enumerate(log_names)}
    cols = {k: [] for k in ("a_logit", "p_logit", "a_sub", "p_sub", "a_pdms", "p_pdms", "kind", "tact", "tok")}
    log_idx = []
    for n in range(N):
        if not (np.isfinite(labels["sub"][n]).all() and np.isfinite(labels["pdms"][n]).all()):
            continue
        for q in range(Q):
            a = int(perts["anchor"][n, q])
            if not bool(bundle["valid"][n, a]):
                continue
            cols["a_logit"].append(scores["a_logit"][n, a])
            cols["p_logit"].append(scores["p_logit"][n, q])
            cols["a_sub"].append(bundle["sub"][n, a])
            cols["p_sub"].append(labels["sub"][n, q])
            cols["a_pdms"].append(bundle["pdms"][n, a])
            cols["p_pdms"].append(labels["pdms"][n, q])
            cols["kind"].append(int(perts["kind"][n, q]))
            cols["tact"].append(command_class(perts["exec"][n, q]) != command_class(repair_last_heading_np(np.asarray(bundle["traj_raw"][n, a], np.float64))))
            cols["tok"].append(n)
            log_idx.append(li[str(bundle["log"][n])])
    P = {k: np.asarray(v) for k, v in cols.items()}
    return P, np.asarray(log_idx, dtype=np.int64), np.asarray(P["kind"])


def analyse(P: dict, log_idx, n_logs: int, B: int = N_BOOT, seed: int = SEED, mode: str = "interval_aware") -> dict:
    """phase `analyse`: A1..A5 with the paired log-cluster bootstrap, per kind (never pooled across the families the prereg separates), tactical share, outcome."""
    terms = admission_terms(P)
    idx = boot_indices(n_logs, B, seed)
    kinds = np.asarray(P["kind"])
    res = {"n_pairs": int(len(kinds)), "n_pairs_lon": int((kinds == KIND_LON).sum()), "n_pairs_lat": int((kinds == KIND_LAT).sum()), "n_logs": int(n_logs), "bootstrap": {"B": B, "seed": seed, "unit": "log",
           "answers": "another draw of LOGS only; one training seed (H-ESTIM-SEED-1); no search was run"}}
    for tag, m in (("all", None), ("lon", kinds == KIND_LON), ("lat", kinds == KIND_LAT)):
        s = per_log_sums(terms, log_idx, n_logs, m)
        res[tag] = {k: ratio_ci(v[0], v[1], idx) for k, v in s.items()}
    res["tactical"] = {"share_changing_command_class": float(np.mean(P["tact"])) if len(kinds) else float("nan"), "n": int(len(kinds)), "proxy": "bearing rule on the plan end point (HYPOTHESIS)"}
    res["strategic"] = "PAUSED (PI ruling 2026-09-19, ITEM 25); reported as PAUSED, not as a pass"
    res["outcome"] = classify(res["all"]["A1"], res["all"]["A4"], res["all"]["A5"], mode)
    res["outcome_mode"] = mode
    res["outcome_if_literal"] = classify(res["all"]["A1"], res["all"]["A4"], res["all"]["A5"], "literal")
    res["gates"] = {"A1_ge": GATE_A1, "A4_le": GATE_A4, "A5_le": GATE_A5}
    return res


# ============================================================================================ I/O + CLI
def load_npz(p):
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def cmd_sample(a):
    t = load_npz(a.table)
    toks = sample_stratified(t["token"], t["log"], a.n, a.seed)
    out = {"n": len(toks), "seed": a.seed, "table_sha256": sha256_file(a.table), "prereg_sha256": PREREG_SHA256, "tokens": toks}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print("ZZSAMPLE_OK", len(toks))
    return 0


def cmd_perturb(a):
    b = load_npz(a.bundle)
    grid = perturbation_grid(a.speed_scales, a.lat_offsets)
    pe = make_perts(b, a.seed, grid, a.lat_profile)
    np.savez(a.out, **pe, bundle_sha256=np.array(sha256_file(a.bundle)))
    print("ZZPERTURB_OK", pe["anchor"].shape)
    return 0


def cmd_score(a):
    require_lock_for_gpu(a.device)
    b = load_npz(a.bundle)
    pe = load_npz(a.perts)
    if a.scorer == "real":
        import real_scorer
        sf = real_scorer.make_score_fn(a.ckpt, a.ctx, np.asarray(b["token"]), device=a.device, refe_dir=a.refe_dir, autocast=a.autocast)
    else:
        raise SystemExit(f"unknown --scorer {a.scorer!r}")
    sc = run_scores(sf, b, pe, a.gate_tokens)
    np.savez(a.out, p_logit=sc["p_logit"], a_logit=sc["a_logit"], perts_sha256=np.array(sha256_file(a.perts)))
    g2b = {"device": a.device, "autocast": a.autocast, "max_abs_diff_vs_bundle_logits": float(np.max(np.abs(sc["a_logit"] - np.asarray(b["logits"], np.float64)))),
           "pick_agreement": float(np.mean(np.argmax(aggregate(sc["a_logit"]), 1) == np.asarray(b["pick"]))),
           "bit_exact_expected": bool(a.device != "cpu" and a.autocast == "bf16"), "gating": False,
           "note": "the bundle logits come from the GPU planner forward; a CPU fp32 score is not bit-comparable (amendment C9). Anchor and perturbation are BOTH scored by this path, so every pair is like for like."}
    sc["gates"]["G2b_cross_device"] = [g2b]
    json.dump(sc["gates"], open(a.out + ".gates.json", "w"), indent=1)
    bad = [k for k, v in sc["gates"].items() if k not in ("G2c", "G2b_cross_device") and not all(x["passed"] for x in v)]
    print("ZZSCORE_" + ("FAIL " + ",".join(bad) if bad else "OK"))
    return 1 if bad else 0


def cmd_analyse(a):
    b, pe, sc, lab = load_npz(a.bundle), load_npz(a.perts), load_npz(a.scores), load_npz(a.labels)
    names = sorted(set(str(x) for x in b["log"]))
    P, li, _ = assemble_pairs(b, pe, sc, lab, names)
    res = analyse(P, li, len(names), a.boot, a.seed, a.mode)
    res["inputs"] = {k: sha256_file(getattr(a, k)) for k in ("bundle", "perts", "scores", "labels")}
    res["prereg_sha256"] = PREREG_SHA256
    res["choices"] = {"lat_profile": str(pe["profile"]), "speed_scales": sorted({float(x) for x in pe["param"][pe["kind"] == KIND_LON]}), "lat_offsets": sorted({float(x) for x in pe["param"][pe["kind"] == KIND_LAT]})}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print("ZZANALYSE_OK", res["outcome"], {k: round(res["all"][k]["point"], 4) for k in ("A1", "A2", "A3", "A4", "A5")})
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sample")
    s.add_argument("--table", required=True)
    s.add_argument("--n", type=int, default=N_SETS)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_sample)
    p = sp.add_parser("perturb")
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--speed-scales", type=float, nargs="+", default=list(SPEED_SCALES))
    p.add_argument("--lat-offsets", type=float, nargs="+", default=list(LAT_OFFSETS))
    p.add_argument("--lat-profile", choices=("ramp", "const"), default=LAT_PROFILE)
    p.set_defaults(fn=cmd_perturb)
    c = sp.add_parser("score")
    c.add_argument("--bundle", required=True)
    c.add_argument("--perts", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--scorer", default="real")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--ctx", required=True)
    c.add_argument("--device", default="cpu", help="cuda only under launch_stepA.py's lock")
    c.add_argument("--autocast", default="none", choices=("none", "bf16"))
    c.add_argument("--refe-dir", default=None)
    c.add_argument("--gate-tokens", type=int, default=20)
    c.set_defaults(fn=cmd_score)
    n = sp.add_parser("analyse")
    for k in ("bundle", "perts", "scores", "labels", "out"):
        n.add_argument(f"--{k}", required=True)
    n.add_argument("--boot", type=int, default=N_BOOT)
    n.add_argument("--seed", type=int, default=SEED)
    n.add_argument("--mode", choices=("interval_aware", "literal"), default="interval_aware")
    n.set_defaults(fn=cmd_analyse)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
