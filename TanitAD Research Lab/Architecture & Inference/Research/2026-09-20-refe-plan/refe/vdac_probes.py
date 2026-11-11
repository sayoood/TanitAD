#!/usr/bin/env python3
"""VDAC launch blockers (PREREG_VDAC_P6 card section 7), as PURE functions so a CPU test can exercise them on synthetic data and the trainer's preflight can call the very same code on the real model / real frames.

G19  head -> (ego, command) DISJOINTNESS.  The drivable head reads ONE tensor (visual_ctx, a function of the images only).  The check is a measurement, not a reading of the code:
     autograd of the head's output with respect to the ego tensor and the command tensor must be None or EXACTLY 0.0.  A CONTROL in the same call (the model's own trajectory output, which DOES
     depend on ego and command) must read non-zero, so a zero from a broken probe cannot pass.  The deliberate regression (REFE_AUX_MUTATE_EGO=1 in scorer_finetune.py) routes the model's ego token
     into the head's context and must turn this RED.

G20  label ALIGNMENT.  On >= 64 real frames, P6's rank-0 target points must lie inside the stored drivable label at the banked rate (the store's own QC read 99.9989 %; the bar is 0.95, card and
     SPEC_P7).  Two deliberate mutants are scored in the same call and EACH must read BELOW the bar: the label flipped in y, and the lookup shifted by one frame.  A check whose mutants also pass
     cannot see a misaligned label, so it is not accepted.

judge_g19 / judge_g20 re-judge the RAW FIELDS (gate_checks and the launch gate call them); the gate's own `ok` is a claim, never the evidence.
"""
from __future__ import annotations

import math

import numpy as np
import torch

# the label geometry (aux_label_store / aux_drivable_head: rear-axle frame, x forward, y left; index [ix, iy]; 0.5 m cells; + inside)
X0, Y0, RES, NX, NY = -8.0, -32.0, 0.5, 160, 128
BAR = 0.95
MIN_FRAMES = 64
MUTATE_EGO_ENV = "REFE_AUX_MUTATE_EGO"


# ====================================================== G19 ======================================================
def _read(g):
    """a gradient read: None stays None, a tensor becomes its max |.| as a python float (nan stays nan)"""
    if g is None:
        return None
    return float(g.detach().abs().max().item()) if g.numel() else 0.0


def disjoint_probe(run, ego, goal, *, mutate=()):
    """`run(ego, goal) -> (head_scalar, control_scalar)`: ONE forward that returns a scalar built from the head's output and a scalar built from an output that depends on ego and goal.
    ego / goal are re-created here as fresh float leaves that require grad.  -> a dict of RAW FIELDS (`ok` is for humans; judge_g19 re-judges)."""
    e = ego.detach().clone().float().requires_grad_(True)
    g = goal.detach().clone().float().requires_grad_(True)
    head_s, ctl_s = run(e, g)
    hg = torch.autograd.grad(head_s, [e, g], retain_graph=True, allow_unused=True)
    cg = torch.autograd.grad(ctl_s, [e, g], retain_graph=True, allow_unused=True)
    f = {"inputs": {"ego": list(e.shape), "command": list(g.shape)},
         "head_grad_ego": _read(hg[0]), "head_grad_command": _read(hg[1]),
         "control_grad_ego": _read(cg[0]), "control_grad_command": _read(cg[1]),
         "mutate": list(mutate)}
    f["ok"] = not judge_g19(f)
    return f


def judge_g19(f) -> list:
    """[] = proved.  The head's gradients w.r.t. ego and command are None or exactly 0.0; the control reads finite and non-zero on both; no regression switch was set."""
    if not isinstance(f, dict):
        return ["G19 is not a dict"]
    r = []
    if f.get("mutate"):
        r.append(f"G19 was produced under regression switch(es) {f['mutate']}: a regression arm can never count")
    for k in ("ego", "command"):
        v = f.get("head_grad_" + k, "absent")
        if v == "absent":
            r.append(f"G19: no head_grad_{k} field")
        elif v is not None and not (isinstance(v, (int, float)) and v == 0.0):
            r.append(f"G19: the drivable head's output has gradient {v} w.r.t. the {k} tensor (must be None or exactly 0.0): the head is NOT disjoint from {k}")
    for k in ("ego", "command"):
        c = f.get("control_grad_" + k)
        if not (isinstance(c, (int, float)) and math.isfinite(c) and c > 0.0):
            r.append(f"G19: the control (the model's own trajectory output) reads {c!r} w.r.t. {k}, not finite and non-zero: the probe is not proved able to see a path")
    if not f.get("inputs"):
        r.append("G19: no input shapes recorded")
    return r


# ====================================================== G20 ======================================================
def points_inside(label_i8, xy):
    """(inside, in_extent) point counts of ONE frame: label int8 [160, 128] (+ inside), xy [n, 2+] ego-frame metres.  The same rule as aux_label_store.target_probe."""
    xy = np.asarray(xy, dtype=np.float64)[:, :2]
    ix = np.rint((xy[:, 0] - X0) / RES - 0.5).astype(int)
    iy = np.rint((xy[:, 1] - Y0) / RES - 0.5).astype(int)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    lab = np.asarray(label_i8)
    return int((lab[ix[ok], iy[ok]] > 0).sum()), int(ok.sum())


def pooled_fraction(labels, xys):
    """pooled over points: sum(inside) / sum(in extent) over the frames (labels: list of [160, 128]; xys: list of [n, 2+])"""
    ins = tot = 0
    for lab, xy in zip(labels, xys):
        a, b = points_inside(lab, xy)
        ins += a
        tot += b
    return (ins / tot) if tot else float("nan"), tot


def alignment_probe(labels, xys, labels_shifted, *, bar=BAR, min_frames=MIN_FRAMES):
    """labels[i] is the label the training lookup returns for frame i, xys[i] its rank-0 target points, labels_shifted[i] the label of the NEXT frame (the lookup shifted by one frame).
    -> RAW FIELDS: the real fraction, the y-flip mutant's and the one-frame-shift mutant's, n frames and points."""
    n = len(labels)
    real, npts = pooled_fraction(labels, xys)
    flipped, _ = pooled_fraction([np.asarray(l)[:, ::-1] for l in labels], xys)
    shifted, _ = pooled_fraction(labels_shifted, xys)
    f = {"n_frames": n, "n_points": npts, "bar": bar, "min_frames": min_frames,
         "frac_real": real, "frac_yflip": flipped, "frac_shift1": shifted}
    f["ok"] = not judge_g20(f)
    return f


def judge_g20(f) -> list:
    """[] = proved: >= 64 frames; the real lookup reads >= the bar; BOTH mutants read BELOW the bar (a check that cannot fail on a flipped or shifted label is not a check)."""
    if not isinstance(f, dict):
        return ["G20 is not a dict"]
    r = []
    bar = f.get("bar")
    if bar is None or bar < BAR:
        r.append(f"G20: the bar {bar!r} is below the card's {BAR}")
        bar = BAR
    n = f.get("n_frames")
    if not (isinstance(n, int) and n >= MIN_FRAMES):
        r.append(f"G20: {n!r} frames, need >= {MIN_FRAMES}")
    for k in ("frac_real", "frac_yflip", "frac_shift1"):
        v = f.get(k)
        if not (isinstance(v, (int, float)) and math.isfinite(v)):
            r.append(f"G20: {k} is {v!r}")
    if r:
        return r
    if f["frac_real"] < bar:
        r.append(f"G20: only {f['frac_real']:.4f} of the rank-0 target points lie inside the label (bar {bar}): the label is misaligned with the targets")
    for k, what in (("frac_yflip", "the y-flipped label"), ("frac_shift1", "the one-frame-shifted lookup")):
        if f[k] >= bar:
            r.append(f"G20: the MUTANT {what} also reads {f[k]:.4f} >= {bar}: this check cannot see that misalignment")
    return r


def align_frames(rows, idx, src, n_max=256):
    """Pick real frames from the training index: rank-0 bank rows that carry a target path, whose label is stored AND whose successor frame (next `step` in the same log) has a stored label too
    (so the one-frame-shift mutant has something to read).  `src.labels([(log, token)]) -> (int8 [n, 160, 128], valid bool [n])` is the TRUE (unshuffled) store.  Deterministic: an even stride
    over the sorted candidates.  -> (labels, xys, labels_shifted) lists of equal length <= n_max."""
    by_log: dict = {}
    for i in idx:
        r = rows[i]
        if int(r.get("rank", 0) or 0) != 0 or not r.get("traj"):
            continue
        by_log.setdefault(r.get("log_name", ""), {}).setdefault(r.get("token", ""), (r.get("step", 0), i))
    pairs = []
    for log in sorted(by_log):
        d = by_log[log]
        toks = sorted(d, key=lambda t: (d[t][0], t))
        pairs += [((log, a), (log, b), d[a][1]) for a, b in zip(toks, toks[1:])]
    if not pairs:
        return [], [], []
    stride = max(1, len(pairs) // (n_max * 3))
    pairs = pairs[::stride][: n_max * 3]
    lab, v1 = src.labels([p[0] for p in pairs])
    lab2, v2 = src.labels([p[1] for p in pairs])
    L, X, S = [], [], []
    for k, (_k0, _k1, i) in enumerate(pairs):
        if bool(v1[k]) and bool(v2[k]):
            L.append(np.asarray(lab[k]))
            S.append(np.asarray(lab2[k]))
            X.append(np.asarray(rows[i]["traj"], dtype=np.float64)[:20, :2])
        if len(L) >= n_max:
            break
    return L, X, S


def alignment_gate(rows, idx, src, *, n_max=256):
    """G20 on the real training index: the raw fields of alignment_probe over `align_frames`."""
    L, X, S = align_frames(rows, idx, src, n_max=n_max)
    f = alignment_probe(L, X, S)
    f["source"] = "rank-0 rows of the training index; labels from the unshuffled store"
    ks = [(rows[i].get("log_name", ""), rows[i].get("token", "")) for i in idx]
    f["bank"] = {"sets": len(ks), "distinct_logs": len({k[0] for k in ks}), "distinct_frames": len(set(ks))}       # the head draws its frames from P6's bank: what that pool is (printed by the chain before training)
    return f
