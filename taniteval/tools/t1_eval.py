#!/usr/bin/env python3
"""T1 — the ACTION-CLOSED-LOOP eval, promoted from the 2026-08-06 ad-hoc pair.

⛔ TIER DOCTRINE (``Project Steering/EVAL_DOCTRINE.md``, BINDING 2026-08-09).
**T1 = the predictor consumes the decoder/planner's OWN actions**; perception
context is fixed at t0. T1 is the PRIMARY offline tier — a capability claim
("drives", "handles") requires T1 or better. **T0 = teacher-forced** (the
predictor consumes the RECORDED future actions) and supports only
prediction/attribution claims, ⛔ never "driving performance". This tool rolls
T1 by default; the T0 teacher-forced arm exists behind ``--with-t0-open-loop``
and every number it emits is stamped ``"tier": "T0"``.

PROVENANCE — what this promotes, and the reproduction gate
----------------------------------------------------------
Faithful parameterised port of the working T1 implementation that produced
MODEL_REGISTRY §1.12 (open-loop lateral skill is an action echo; S-curve
reproduction 97.9 % -> ~5 % closed-loop, hold-action 0.0 %):

* ``TanitAD Research Hub/Architecture & Inference/Implementation/incoming/
  2026-08-06-v1-defect-triage/tools/closed_loop_dump.py``  (the GPU roll+dump)
* ``…/tools/analyze_cl.py`` + ``…/tools/s_curve_dump.py``  (the analysis)
* ``…/results/closed_loop_analysis.json``                   (the §1.12 numbers)

The byte-close gate: run pod-side with the §1.12 checkpoint/heads/corpus and
``--analyze-only`` over the original ``dump_cl`` directory; the
``legacy_epmean_row`` / ``legacy_paired`` / ``s_curve_masked_legacy`` blocks
must reproduce ``closed_loop_analysis.json``. Their arithmetic is ported
VERBATIM (including mean-of-episode-means and the per-call ``default_rng(0)``)
and is labelled reproduction-only — the HEADLINE rows here are full-set pooled
means with episode-cluster bootstrap intervals, per the estimator rule.
(The JSON's ``o6_replan_accel_jump_gtframe`` field was produced by a later
one-off probe, not by ``analyze_cl.py``; it is intentionally not ported.)

WHAT EVERY NUMBER CARRIES
-------------------------
``tier`` (T0/T1), ``estimator`` (name), ``n`` — block-level, mechanical, and a
family whose inputs are missing is reported ``UNAVAILABLE`` with the reason and
its n (the binding four-families rule), never silently dropped.

SEPARABILITY
------------
The GPU pass (pod) and the analysis (any box) are separable: the roll writes
per-episode npz dumps (schema below), and ``--analyze-only <dump-dir>``
consumes an existing dump — including the ORIGINAL §1.12 ``dump_cl`` (its arm
keys ``o16/o6/c16/c6/h16`` carry default tier stamps).

E1.4 ADAPTER — THE v5f / v5.8f STACK (2026-08-11, ADDITIVE)
-----------------------------------------------------------
The §1.12 path reads a RAW epcache (``--corpus <dir of ep_*.pt>``) and a
SEPARATE ``UnicycleStepReadout`` checkpoint (``--head``). The w120 flagship
corpora are v2 compressed caches (``*.v2ep.pt``) and the v5f/v5.8f checkpoints
carry their OWN metric decoder inside the grounding module, so two additive
inputs exist:

``--v2-val-cache DIR [DIR…]``  the v2 corpus, read through the SAME providers
    ``scripts/eval_flagship_v4.py`` and ``scripts/w7_roll_rerank.py`` use
    (``build_v2_val_episodes`` -> ``tanitad.data.v2_dataset.build_v2_providers``,
    IMPORTED — no fourth decode path), wrapped by :class:`V2RawEp` into the
    exact ``(feats, poses, actions)`` surface the roll already consumes. The
    geometry flags (``--frame-h/-w/--frame-hfov/--projection/--v2-subframe``)
    are the eval-side rig-clean seam, resolved by the trainer's OWN
    ``resolve_v2_frames`` via ``eval_flagship_v4.resolve_eval_frames``.

``--grounding-readout``  decode with the checkpoint's own
    ``grounding.step['op']`` (:class:`StepDisplacementReadout`) instead of a
    ``--head`` ckpt. T0/hold arms go through
    ``metric_dynamics.decode_transitions(step_readout, rollout_transitions(…), k)``
    — byte-the-same call convention as ``scripts/w7_roll_rerank.py`` and
    ``scripts/stage_a_probes.py``, i.e. the canary decode
    (``train_flagship_v4.py:584-586``). ⭐ Consequence worth stating: with
    ``--with-t0-open-loop`` the ``ol`` arm IS the WM canary quantity, so its
    dense ADE is an in-run cross-check against the run's banked canary.

⛔ **Nothing on the legacy path changed.** ``--corpus`` + ``--head`` still runs
:func:`run_rollout` — the byte-close-validated §1.12 roll — untouched, and the
adapter lives in :func:`run_rollout_ext`. The duplication of the window loop is
DELIBERATE: parameterising the validated loop would put the byte-close gate at
risk for a cosmetic gain. ``stack/tests/test_t1_v2_adapter.py`` pins that
separation (source-hash guard + dispatch predicate).

⚠️ THE ONE PIECE OF NEW MATH, stated so it can be attacked: the T1 closed loop
needs the decoder's emission expressed in the corpus ACTION contract, and the
grounding readout emits a Δpose, not controls. :func:`implied_controls` inverts
``physicalai.signals_at`` (physicalai.py:596-641) exactly as :func:`roll_closed`
already does for the unicycle head — speed from the step displacement (the same
definition :func:`controls` uses), accel from its finite difference, yaw rate
from ``dyaw``, and ``steer = atan(wheelbase * yaw_rate / max(v, 0.3))``. The
``ol``/``ha`` arms need none of it (they consume recorded/held actions), which
is why the T0-vs-T1 triplet is the right thing to run in one pass.

DUMP SCHEMA (per-episode ``ep*.npz``)
-------------------------------------
    g    [N,K,2]  GT ego-frame waypoints            (required)
    cl   [N,K,2]  closed-loop arm (T1)               — this tool's default arm
    ol   [N,K,2]  open-loop teacher-forced arm (T0)  — only with --with-t0-open-loop
    ha   [N,K,2]  hold-action control (T1 battery)   — only with --with-hold-action
    ws   [N]      window origin frame indices        (provenance, optional)
    <arm>_fan_err [N,C], <arm>_sel_idx [N], <arm>_fan_scores [N,C] (optional)
                  a fan+selector surface -> the taniteval.selgap hook fires.
Any other non-suffixed key is treated as an additional arm and MUST carry a
tier (``--tiers name=T0|T1``); an unstamped arm is a hard error, because an
un-tiered number is exactly what the doctrine forbids.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

# path bootstrap, same convention as the other taniteval tools/tests: derive
# the package parent and the stack from THIS file's location so the CLI runs
# from any cwd with no preset PYTHONPATH (tests invoke it as a subprocess).
_HERE = os.path.dirname(os.path.abspath(__file__))    # <repo>/taniteval/tools
_TE_PARENT = os.path.dirname(_HERE)                    # <repo>/taniteval
_REPO = os.path.dirname(_TE_PARENT)                    # <repo>
for _pth in (os.path.join(_REPO, "stack"), _TE_PARENT):
    if os.path.isdir(_pth) and _pth not in sys.path:
        sys.path.insert(0, _pth)

# --- the §1.12 grid constants (closed_loop_dump.py), overridable by CLI ----- #
DT = 0.1          # latent tick, s
K = 20            # rollout horizon, latent steps (2 s)
W = 8             # context window, frames
WHEELBASE = 2.9   # legacy wheelbase of the corpus action contract (signals_at)
EPISODES_DEFAULT = 40   # §1.12 took the first 40 episodes of the corpus

# --- the analyze_cl.py / s_curve_dump.py analysis constants ----------------- #
THR = 0.03        # rad — S-lobe threshold (~1.7 deg per half-window)
NEAR = 3          # steps — "near" accel horizon for event/response metrics
EVENT_ACC = 1.0   # m/s^2 — |GT near-accel| above this is a decel/accel event
LAG_MAX = 20      # windows — xcorr search range (stride-1 => 1 window = DT s)
LAG_MIN_CORR = 0.2
STATIONARY_DS = 0.05   # m — analyze_cl's fixed step-displacement gate
                       # (⚠️ deliberately NOT dt-scaled: faithful port)

#: tier stamps for the known arm keys — §1.12 legacy names and this tool's.
#: hold-action consumes NO recorded future (it holds the last OBSERVED action),
#: so it sits in the T1 battery as the control arm, per EVAL_DOCTRINE rule 3.
DEFAULT_TIERS = {
    "cl": "T1", "c16": "T1", "c6": "T1",
    "ha": "T1", "h16": "T1",
    "ol": "T0", "o16": "T0", "o6": "T0",
}
_TIER_NOTE = {
    "T0": "teacher-forced — prediction quality only, NEVER driving performance",
    "T1": "action-closed loop (imagination) — the PRIMARY offline tier",
}
_FAN_SUFFIXES = ("_fan_err", "_sel_idx", "_fan_scores")

_ESTIMATOR_NOTE = (
    "point estimates are FULL-SET pooled means over windows; intervals are the "
    "episode-cluster bootstrap (taniteval.ci). ⛔ overlapping_holdout_se is NOT "
    "used anywhere: it biases the POINT ESTIMATE, not only the interval. The "
    "legacy_* blocks reproduce analyze_cl.py's mean-of-episode-means arithmetic "
    "for the §1.12 byte-close gate ONLY and are not quotable as new results.")


def _p(*a):
    print(*a, flush=True)


# ============================================================================ #
# Pure-numpy analysis primitives — VERBATIM ports (CPU-testable)               #
# ============================================================================ #
def controls(path, dt=DT):
    """Per-window kinematics from a waypoint path — port of ``analyze_cl.controls``.

    ``path`` [N, K, >=2] ego-frame waypoints (origin prepended internally).
    Returns ``(speed [N,K], accel [N,K-1], net_yaw_masked [N], dheading [N,K-1],
    ok [N,K-1])`` where ``ok`` masks steps whose displacement is below the fixed
    ``STATIONARY_DS`` gate on either side. ``net_yaw_masked`` sums the MASKED
    dheading (stationary steps contribute 0); ``dheading`` is returned RAW —
    the unmasked S-definition reads it directly, the masked one applies ``ok``.
    """
    p = np.concatenate([np.zeros((path.shape[0], 1, 2)), path[..., :2]], 1)
    d = p[:, 1:] - p[:, :-1]
    ds = np.sqrt((d ** 2).sum(-1) + 1e-12)
    sp = ds / dt
    acc = (sp[:, 1:] - sp[:, :-1]) / dt
    h = np.arctan2(d[..., 1], d[..., 0])
    dh = (h[:, 1:] - h[:, :-1] + math.pi) % (2 * math.pi) - math.pi
    ok = (ds[:, 1:] > STATIONARY_DS) & (ds[:, :-1] > STATIONARY_DS)
    return sp, acc, np.where(ok, dh, 0.0).sum(1), dh, ok


def s_halves(dh, ok=None):
    """First/second half-window net headings ``(h1 [N], h2 [N])``.

    ``ok=None`` -> the UNMASKED definition (analyze_cl.py inline: raw dh).
    ``ok`` given -> the STATIONARY-MASKED definition (s_curve_dump.halves —
    the BANKED §1.12 headline: 0.9785 open / 0.0538 closed / 0.0 hold).
    The split index is ``(K-1)//2`` = 9 for K=20, matching both originals.
    """
    if ok is not None:
        dh = np.where(ok, dh, 0.0)
    split = dh.shape[1] // 2   # 19 -> 9, matching dh[:, :9] / dh[:, 9:]
    return dh[:, :split].sum(1), dh[:, split:].sum(1)


def s_windows(h1, h2, thr=THR):
    """GT S-window mask: opposite-sign half-lobes, both above ``thr``."""
    return (np.sign(h1) != np.sign(h2)) & (np.abs(h1) > thr) & (np.abs(h2) > thr)


def s_hits(p1, p2, g1, g2, thr=THR):
    """Arm reproduces the S: matches BOTH lobe signs with magnitude > thr/2."""
    return ((np.sign(p1) == np.sign(g1)) & (np.sign(p2) == np.sign(g2))
            & (np.abs(p1) > thr / 2) & (np.abs(p2) > thr / 2))


def xcorr_lag(a_p, a_g, dt=DT, max_lag=LAG_MAX, min_corr=LAG_MIN_CORR):
    """Lag (s) of the arm's near-accel behind GT's, per episode — analyze_cl port.

    ``a_p``/``a_g`` [n_windows] near-accel per stride-1 window (1 window = dt s).
    Returns the argmax-of-xcorr lag in seconds, or None when the normalised
    peak never exceeds ``min_corr`` (no attributable accel signal).
    Positive = the arm RESPONDS LATE.
    """
    x, y = a_p - a_p.mean(), a_g - a_g.mean()
    den = math.sqrt((x ** 2).sum() * (y ** 2).sum())
    if den <= 1e-9:
        return None
    # identical to analyze_cl's argmax over l in [-max_lag, max_lag] (first
    # occurrence wins on ties), PLUS an overlap guard for episodes with fewer
    # windows than the search range — the original never met one (>=170
    # windows/episode on the §1.12 grid) and would broadcast-crash there.
    n = len(y)
    best_l, best_c = None, -float("inf")
    for l in range(-max_lag, max_lag + 1):
        if abs(l) >= n:
            continue
        c = (float((x[l:] * y[:n - l]).sum()) if l >= 0
             else float((x[:l] * y[-l:]).sum())) / den
        if c > best_c:
            best_c, best_l = c, l
    if best_l is None or best_c <= min_corr:
        return None
    return best_l * dt


def fam_row(P, G, dt=DT):
    """Per-episode scalar row — VERBATIM port of ``analyze_cl.fam_row``.

    ⛔ Reproduction-only arithmetic: pooled over the EPISODE's windows, then
    (by the caller) averaged over episodes -> mean-of-episode-means. The
    headline rows use the full-set pooled path instead; this exists so the
    §1.12 numbers in ``closed_loop_analysis.json`` reproduce byte-close.
    """
    sp_p, ac_p, ny_p, _, _ = controls(P, dt)
    sp_g, ac_g, ny_g, _, _ = controls(G, dt)
    jerk = (ac_p[:, 1:] - ac_p[:, :-1]) / dt
    return {"ade_m": float(np.linalg.norm(P[..., :2] - G[..., :2], axis=-1).mean()),
            "speed_bias_mps": float((sp_p - sp_g).mean()),
            "speed_mae_mps": float(np.abs(sp_p - sp_g).mean()),
            "accel_rms_mps2": float(np.sqrt((ac_p ** 2).mean())),
            "jerk_rms_mps3": float(np.sqrt((jerk ** 2).mean())),
            "net_yaw_err_rad": float(np.abs(ny_p - ny_g).mean())}


def legacy_paired_boot(per_a, per_b, keys, n=2000, seed=0):
    """VERBATIM port of ``analyze_cl.boot`` — episode-mean bootstrap on B−A.

    ⛔ Reproduction-only (fresh ``default_rng(seed)`` per call, resamples the
    per-EPISODE fam_row means). The decision-grade paired interval is
    ``taniteval.ci.paired_episode_cluster_bootstrap`` on per-WINDOW values,
    emitted beside this under ``paired_decision_grade``.
    """
    rng = np.random.default_rng(seed)
    E = len(per_a)
    out = {}
    for k in keys:
        d = np.array([e[k] for e in per_b]) - np.array([e[k] for e in per_a])
        dr = [float(d[rng.integers(0, E, E)].mean()) for _ in range(n)]
        lo, hi = np.percentile(dr, [2.5, 97.5])
        out[k] = {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi),
                  "separated": bool(lo > 0 or hi < 0)}
    return out


def _unavailable(reason, tier, n=0):
    """The binding shape for a family/metric whose inputs are missing."""
    return {"status": "UNAVAILABLE", "reason": reason, "n": int(n),
            "tier": tier, "estimator": "n/a — inputs missing (WORK ITEM, not a pass)"}


def resolve_tiers(arm_keys, overrides=None):
    """Map every arm to a tier stamp; an unstamped arm is a HARD ERROR."""
    tiers = dict(DEFAULT_TIERS)
    tiers.update(overrides or {})
    missing = [a for a in arm_keys if tiers.get(a) not in ("T0", "T1")]
    if missing:
        raise ValueError(
            f"arms {sorted(set(missing))} carry no T0/T1 tier stamp. Every "
            f"emitted number must carry its tier (EVAL_DOCTRINE.md); pass "
            f"--tiers name=T0|T1. Known defaults: {sorted(DEFAULT_TIERS)}")
    return {a: tiers[a] for a in arm_keys}


# ============================================================================ #
# The analysis (CPU; consumes a dump directory)                                #
# ============================================================================ #
def analyze(dump_files, *, tiers=None, gt_key="g", n_boot=2000, seed=0,
            dt=DT, lead=None, byte_check=None, paired=None):
    """Score a T1 dump: four families + S-rate + lag + response + selgap hooks.

    ``dump_files``: ordered per-episode npz paths (schema in module docstring).
    ``byte_check``: optional ``(arm, ref_files, ref_key)`` — max |Δ| of ``arm``
    vs a reference dump's ``ref_key`` (the §1.12 ``_o16_vs_banked_b_max_abs``
    gate, parameterised). ``paired``: list of ``(a, b, name)`` arm pairs; the
    default reproduces the §1.12 trio when those arms are present, else
    ``(ol, cl)`` when both exist. ``lead``: optional lead block dict (see
    ``tools/build_lead_block.py``) row-aligned to the dump's window grid.
    """
    files = [str(f) for f in dump_files]
    if not files:
        raise ValueError("analyze() got no dump files")
    with np.load(files[0]) as d0:
        keys = list(d0.files)
    if gt_key not in keys:
        raise ValueError(f"dump has no GT key {gt_key!r}; keys: {keys}")
    arms = [k for k in keys if k not in (gt_key, "ws")
            and not k.endswith(_FAN_SUFFIXES)]
    if not arms:
        raise ValueError(f"dump has no arm keys beside {gt_key!r}/'ws': {keys}")
    tier = resolve_tiers(arms, tiers)

    # ---- pass 1: per-episode accumulation (pure numpy) --------------------- #
    per_ep = {a: [] for a in arms}                      # legacy fam_row rows
    ev_gt = {"decel": [], "accel": []}
    ev = {a: {"decel": [], "accel": []} for a in arms}
    lag = {a: [] for a in arms}
    lag_eid = {a: [] for a in arms}
    s_m = {a: [] for a in arms}          # masked (BANKED) hit lists
    s_u = {a: [] for a in arms}          # unmasked (analyze_cl inline)
    s_eid_m, s_eid_u = [], []
    P_cat = {a: [] for a in arms}
    fan = {a: {"err": [], "idx": [], "sc": [], "has_sc": False} for a in arms}
    G_cat, eid_w, byte_max = [], [], []

    ref_files = None
    if byte_check is not None:
        bc_arm, ref_files, ref_key = byte_check
        if len(ref_files) != len(files):
            raise ValueError(f"byte-check ref dump has {len(ref_files)} episodes "
                             f"for {len(files)} — different grids, refusing")

    for fi, f in enumerate(files):
        d = np.load(f)
        eid = os.path.splitext(os.path.basename(f))[0]
        G = d[gt_key][..., :2].astype(np.float64)
        n_w = G.shape[0]
        G_cat.append(G)
        eid_w += [eid] * n_w

        _, ac_g, _, dh_g, ok_g = controls(G, dt)
        a_g = ac_g[:, :NEAR].mean(1)
        m_de, m_ac = a_g < -EVENT_ACC, a_g > EVENT_ACC
        ev_gt["decel"] += list(a_g[m_de])
        ev_gt["accel"] += list(a_g[m_ac])
        g1u, g2u = s_halves(dh_g)                       # unmasked
        is_s_u = s_windows(g1u, g2u)
        g1m, g2m = s_halves(dh_g, ok_g)                 # masked (banked)
        is_s_m = s_windows(g1m, g2m)
        s_eid_u += [eid] * int(is_s_u.sum())
        s_eid_m += [eid] * int(is_s_m.sum())

        if ref_files is not None:
            with np.load(ref_files[fi]) as dr:
                ref = dr[ref_key]
            mine = d[bc_arm]
            if ref.shape[:2] != mine.shape[:2]:
                raise ValueError(f"byte-check shape mismatch on {f}: "
                                 f"{mine.shape} vs ref {ref.shape}")
            byte_max.append(float(np.abs(
                mine[..., :min(mine.shape[-1], ref.shape[-1])]
                - ref[..., :min(mine.shape[-1], ref.shape[-1])]).max()))

        for a in arms:
            P = d[a][..., :2].astype(np.float64)
            if P.shape != G.shape:
                raise ValueError(f"{f}: arm {a!r} shape {P.shape} != GT {G.shape}")
            P_cat[a].append(P)
            per_ep[a].append(fam_row(P, G, dt))
            _, ac_p, _, dh_p, ok_p = controls(P, dt)
            a_p = ac_p[:, :NEAR].mean(1)
            ev[a]["decel"] += list(a_p[m_de])
            ev[a]["accel"] += list(a_p[m_ac])
            lg = xcorr_lag(a_p, a_g, dt)
            if lg is not None:
                lag[a].append(lg)
                lag_eid[a].append(eid)
            p1u, p2u = s_halves(dh_p)
            s_u[a] += list(s_hits(p1u, p2u, g1u, g2u)[is_s_u])
            p1m, p2m = s_halves(dh_p, ok_p)
            s_m[a] += list(s_hits(p1m, p2m, g1m, g2m)[is_s_m])
            if f"{a}_fan_err" in d.files and f"{a}_sel_idx" in d.files:
                fan[a]["err"].append(d[f"{a}_fan_err"].astype(np.float64))
                fan[a]["idx"].append(d[f"{a}_sel_idx"].astype(np.int64))
                if f"{a}_fan_scores" in d.files:
                    fan[a]["sc"].append(d[f"{a}_fan_scores"].astype(np.float64))
                    fan[a]["has_sc"] = True
        d.close()

    # ---- pass 2: pooled scoring (torch/taniteval machinery, still CPU) ----- #
    import torch

    from taniteval import ci as _ci
    from taniteval import four_families as ff
    from taniteval import selgap as _selgap

    G_all = np.concatenate(G_cat)
    N, Kh = int(G_all.shape[0]), int(G_all.shape[1])
    gt_t = torch.as_tensor(G_all).float()
    wp_idx = sorted({max(0, int(round(Kh * q)) - 1) for q in (.25, .5, .75, 1.0)})
    n_ep = len(files)

    if lead is not None:
        n_lead = int(np.asarray(lead["leads"]).shape[0])
        if n_lead != N:
            # ⛔ POSITIONAL JOIN (same refusal as eval_four_families.py): a
            # mismatched lead block scores this dump against another grid's
            # traffic — a plausible number, not an error.
            raise ValueError(f"lead block has {n_lead} rows for {N} windows — "
                             f"different window grids; rebuild, do NOT truncate")

    out = {
        "tool": "taniteval/tools/t1_eval.py",
        "n_episodes": n_ep,
        "n_windows": N,
        "horizon_steps": Kh,
        "dt_s": dt,
        "gt_key": gt_key,
        "arm_keys": arms,
        "tiers": tier,
        "_tier_doctrine": ("EVAL_DOCTRINE.md 2026-08-09 — T1 (action-closed "
                           "loop) is the PRIMARY tier; T0 (teacher-forced) "
                           "supports only prediction/attribution claims and is "
                           "NEVER quotable as driving performance."),
        "_estimator": _ESTIMATOR_NOTE,
        "_binding": ("Sayed 2026-08-02 — LONGITUDINAL + LATERAL + TACTICAL + "
                     "STRATEGIC in ADDITION to ADE, per-family, never pooled. "
                     "A family reported UNAVAILABLE is a WORK ITEM, not a pass."),
        "arms": {},
    }
    if byte_check is not None:
        out["byte_check"] = {"arm": byte_check[0], "ref_key": byte_check[2],
                             "max_abs": max(byte_max) if byte_max else None,
                             "n_episodes": len(byte_max)}

    gt_de = float(np.mean(ev_gt["decel"])) if ev_gt["decel"] else None
    gt_ac = float(np.mean(ev_gt["accel"])) if ev_gt["accel"] else None
    comps_w = {}                                        # per-window, for paired

    for a in arms:
        t = tier[a]
        P_all = np.concatenate(P_cat[a])
        pred_t = torch.as_tensor(P_all).float()
        win = {"pred_dense": pred_t, "gt_dense": gt_t,
               "pred": pred_t[:, wp_idx], "gt": gt_t[:, wp_idx],
               "wp_steps": [i + 1 for i in wp_idx],
               "dense_steps": list(range(1, Kh + 1)),
               "dt_s": dt, "eid": eid_w}
        if lead is not None:
            win["lead"] = {"leads": np.asarray(lead["leads"]),
                           "lead_lens": np.asarray(lead["lead_lens"]),
                           "speeds": np.asarray(lead["speeds"]),
                           "state": np.asarray(lead["state"]),
                           "eid": list(lead["eid"]), "n_boot": n_boot}

        # -- FOUR FAMILIES: taniteval machinery, not a copy ------------------ #
        # TACTICAL/STRATEGIC have no inputs in a T1 dump (no decision heads are
        # traversed) -> all_families reports them UNAVAILABLE with reason + n,
        # which is the binding shape. Stamp tier/estimator on every family.
        fam = ff.all_families(win)
        for fk in ("longitudinal", "lateral", "tactical", "strategic"):
            fam[fk]["tier"] = t
            fam[fk].setdefault("n", fam[fk].get("n_windows", fam[fk].get("n", N)))
            fam[fk].setdefault(
                "estimator", "full_set pooled mean over windows (point); "
                             "intervals under 'intervals'")

        # -- per-window components -> ONE episode-cluster resampling --------- #
        # ⛔ computed with four_families' OWN _seq_geometry, not a re-derivation
        # (same rule as tools/eval_four_families.py).
        P_g = ff._seq_geometry(pred_t, dt)
        G_g = ff._seq_geometry(gt_t, dt)
        ade_w = torch.linalg.norm(pred_t - gt_t, dim=-1).mean(1).numpy()
        fde_w = torch.linalg.norm(pred_t[:, -1] - gt_t[:, -1], dim=-1).numpy()
        sp_w = (P_g["speed"] - G_g["speed"]).abs().mean(1).numpy()
        al_w = (P_g["along"] - G_g["along"]).abs().mean(1).numpy()
        ct_w = (P_g["cross"] - G_g["cross"]).abs().mean(1).numpy()
        comps_w[a] = {"ade_m": ade_w, "speed_mae_mps": sp_w}
        boots = _ci.bootstrap_metrics(
            {"ade_dense_m": (ade_w, "mean"),
             "fde_last_m": (fde_w, "mean"),
             "LON_speed_mae_mps": (sp_w, "mean"),
             "LON_along_mae_m": (al_w, "mean"),
             "LAT_cross_mae_m": (ct_w, "mean")},
            eid_w, n_boot=n_boot, seed=seed)
        both = (P_g["valid"] & G_g["valid"])
        dh = P_g["heading"] - G_g["heading"]
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        nvalid = both.sum(1)
        head_w = torch.where(nvalid > 0,
                             (dh.abs() * both).sum(1) / nvalid.clamp_min(1),
                             torch.nan).numpy() * 180.0 / np.pi
        keep = ~np.isnan(head_w)
        if keep.any():
            boots["LAT_heading_mae_deg"] = _ci.episode_cluster_bootstrap(
                head_w[keep], [e for e, k in zip(eid_w, keep) if k],
                n_boot=n_boot, seed=seed)
            boots["LAT_heading_mae_deg"]["n_windows_dropped_no_valid_step"] = \
                int((~keep).sum())

        # -- S-CURVE reproduction (the T1 standard battery) ------------------ #
        s_blk = {"tier": t,
                 "definition": (f"S = GT half-window net headings opposite "
                                f"signs, both > |{THR}| rad; hit = arm matches "
                                f"BOTH lobe signs (mag > thr/2). 'masked' "
                                f"zeroes stationary steps (ds <= "
                                f"{STATIONARY_DS} m) — the BANKED §1.12 "
                                f"definition; 'unmasked' is analyze_cl.py's "
                                f"inline variant.")}
        for name, hits, eids in (("masked", s_m[a], s_eid_m),
                                 ("unmasked", s_u[a], s_eid_u)):
            if not hits:
                s_blk[name] = _unavailable(
                    "no S-windows in GT on this corpus (n_s = 0) — the S-rate "
                    "has no denominator here", t)
                continue
            hv = np.asarray(hits, dtype=np.float64)
            s_blk[name] = {
                "rate": round(float(hv.mean()), 4),
                "n_s_windows": int(hv.size),
                "n": int(hv.size),
                "tier": t,
                "estimator": "episode_cluster_bootstrap over per-S-window "
                             "hit indicators",
                "ci": _ci.episode_cluster_bootstrap(hv, eids, n_boot=n_boot,
                                                    seed=seed)}

        # -- LAG (response latency, analyze_cl port) ------------------------- #
        if lag[a]:
            lv = np.asarray(lag[a], dtype=np.float64)
            lag_blk = {
                "lag_accel_s_mean": round(float(lv.mean()), 4),
                "n": int(lv.size),
                "n_episodes_with_signal": int(lv.size),
                "n_episodes_total": n_ep,
                "tier": t,
                "estimator": ("per-episode xcorr-peak lag (near-accel, "
                              f"|lag| <= {LAG_MAX} windows, min corr "
                              f"{LAG_MIN_CORR}); mean + episode-cluster "
                              "bootstrap over episodes"),
                "ci": _ci.episode_cluster_bootstrap(lv, lag_eid[a],
                                                    n_boot=n_boot, seed=seed)}
        else:
            lag_blk = _unavailable(
                f"no episode's near-accel xcorr peak exceeded {LAG_MIN_CORR} — "
                f"no attributable accel signal to lag against", t)

        # -- EVENT RESPONSE ratios (analyze_cl port) ------------------------- #
        resp = {"tier": t,
                "estimator": ("pooled event-conditioned near-accel mean, "
                              "arm / GT (analyze_cl port; events = GT "
                              f"|near-accel| > {EVENT_ACC} m/s^2)")}
        for kind, gt_mean in (("decel", gt_de), ("accel", gt_ac)):
            n_e = len(ev_gt[kind])
            if gt_mean is None or n_e == 0:
                resp[kind] = _unavailable(
                    f"no GT {kind} events (|near-accel| > {EVENT_ACC} m/s^2) "
                    f"in this corpus", t)
            else:
                resp[kind] = {
                    "response_ratio": round(float(np.mean(ev[a][kind]))
                                            / gt_mean, 4),
                    "gt_mean_near_accel_mps2": round(gt_mean, 4),
                    "n_events": n_e, "n": n_e, "tier": t,
                    "estimator": resp["estimator"]}

        # -- SEL_GAP hook (taniteval.selgap) when a fan surface is present --- #
        if fan[a]["err"]:
            fe = np.concatenate(fan[a]["err"])
            si = np.concatenate(fan[a]["idx"])
            sc = np.concatenate(fan[a]["sc"]) if fan[a]["has_sc"] else None
            if fe.shape[0] != N:
                sel_blk = _unavailable(
                    f"{a}_fan_err has {fe.shape[0]} rows for {N} windows — "
                    f"fan surface is on a different grid; refusing the join", t,
                    n=fe.shape[0])
            else:
                sel_blk = _selgap.selgap(fe, si, eid_w, n_boot=n_boot,
                                         seed=seed, scores=sc,
                                         level=f"operative_{t.lower()}_action_fan")
                sel_blk["tier"] = t
                sel_blk["estimator"] = ("taniteval.selgap (episode-cluster "
                                        "bootstrap on the per-window gap)")
                sel_blk["n"] = sel_blk["n_windows"]
        else:
            sel_blk = _unavailable(
                f"no fan+selector surface in the dump (keys {a}_fan_err / "
                f"{a}_sel_idx absent) — this dump's arm commits to one action "
                f"per step, so there is no fan to gap. Emitting a fan from the "
                f"roll is a WORK ITEM (see taniteval/taniteval/selgap.py)", t)

        # -- legacy §1.12 reproduction row (analyze_cl arithmetic) ----------- #
        legacy = {k: round(float(np.mean([r[k] for r in per_ep[a]])), 4)
                  for k in per_ep[a][0]}
        if gt_de is not None:
            legacy["decel_response_ratio"] = round(
                float(np.mean(ev[a]["decel"])) / gt_de, 4)
        if gt_ac is not None:
            legacy["accel_response_ratio"] = round(
                float(np.mean(ev[a]["accel"])) / gt_ac, 4)
        legacy["lag_accel_s_mean"] = (round(float(np.mean(lag[a])), 4)
                                      if lag[a] else None)
        legacy["s_reproduction_rate"] = (round(float(np.mean(s_u[a])), 4)
                                         if s_u[a] else None)
        legacy["_reproduction_of"] = (
            "analyze_cl.py per-arm row — mean-of-EPISODE-means, unmasked "
            "S-def. §1.12 byte-close gate ONLY; the quotable row is "
            "'four_families' + 'intervals' (full-set pooled).")

        out["arms"][a] = {
            "tier": t,
            "tier_note": _TIER_NOTE[t] + (
                " — hold-action control baseline" if a in ("ha", "h16") else ""),
            "four_families": fam,
            "intervals": {"tier": t, "n": N,
                          "estimator": "episode_cluster_bootstrap",
                          "metrics": boots},
            "s_curve": s_blk,
            "lag": lag_blk,
            "response": resp,
            "sel_gap": sel_blk,
            "legacy_epmean_row": legacy,
        }

    # ---- the banked masked-S legacy block (closed_loop_analysis.json) ------ #
    out["s_curve_masked_legacy"] = {
        "n_s": len(s_eid_m),
        **{a: (round(float(np.mean(s_m[a])), 4) if s_m[a] else None)
           for a in arms},
        "_note": "stationary-masked dh, matching s_curve_dump.py's banked "
                 "definition — §1.12 reproduction block",
    }

    # ---- paired contrasts --------------------------------------------------- #
    if paired is None:
        if {"o16", "o6", "c16", "c6"} <= set(arms):     # the §1.12 trio
            paired = [("o16", "o6", "paired_run6_minus_v16_open"),
                      ("o16", "c16", "paired_closed_minus_open_v16"),
                      ("o6", "c6", "paired_closed_minus_open_run6")]
        elif {"ol", "cl"} <= set(arms):
            paired = [("ol", "cl", "paired_closed_minus_open")]
        else:
            paired = []
    out["paired_legacy"] = {}
    out["paired_decision_grade"] = {}
    for pa, pb, name in paired:
        # exact §1.12 key-sets: the o16-vs-o6 pair carried 4 keys, the two
        # closed-vs-open pairs 2. Any other (generic) pair gets all 4.
        keys4 = (("ade_m", "speed_mae_mps")
                 if name.startswith("paired_closed_minus_open_")
                 else ("ade_m", "speed_mae_mps", "jerk_rms_mps3",
                       "net_yaw_err_rad"))
        out["paired_legacy"][name] = legacy_paired_boot(
            per_ep[pa], per_ep[pb], keys4, n=2000, seed=0)
        out["paired_legacy"][name]["_reproduction_of"] = (
            "analyze_cl.boot — episode-mean bootstrap, fresh rng(0) per call; "
            "§1.12 byte-close gate only")
        dec = {"tier": f"{tier[pb]} minus {tier[pa]}", "n": N,
               "estimator": "paired_episode_cluster_bootstrap",
               "direction": f"{pb} - {pa}"}
        for mk in ("ade_m", "speed_mae_mps"):
            dec[mk] = _ci.paired_episode_cluster_bootstrap(
                comps_w[pb][mk], comps_w[pa][mk], eid_w,
                n_boot=n_boot, seed=seed)
        out["paired_decision_grade"][name] = dec
    return out


# ============================================================================ #
# The GPU roll (pod-side) — faithful port of closed_loop_dump.py               #
# ============================================================================ #
def decode_open(head, trans, v0, dt=DT):
    """Decode a unicycle head over pre-rolled ``(z_prev, z_hat)`` transitions.

    Port of ``closed_loop_dump.decode_open``. Tier depends on how ``trans`` was
    rolled: TRUE future actions -> **T0 teacher-forced**; held t0 action -> the
    T1 hold-action control. ⚠️ pod-verification pending — this port has been
    CPU-import-tested only; the byte-close gate vs the §1.12 dump runs pod-side.
    """
    import torch
    from tanitad.models.metric_dynamics import accumulate_se2
    v = v0.clone()
    ap = torch.zeros_like(v)
    yp = torch.zeros_like(v)
    rows = []
    for zp, zh in trans:
        aj, yj = head(zp, zh, v, ap, yp)
        rows.append(torch.stack([v * dt, torch.zeros_like(v), yj * dt], -1))
        v = (v + aj * dt).clamp_min(0.0)
        ap, yp = aj, yj
    return accumulate_se2(torch.stack(rows, 1))


def roll_closed(model, head, states, awE, v0, ego, k=K, dt=DT,
                wheelbase=WHEELBASE):
    """T1: each predictor step is conditioned on the action the head JUST CHOSE.

    Port of ``closed_loop_dump.roll_closed`` — the corpus action contract
    (``physicalai.signals_at``): ``steer = atan(wheelbase * kappa)`` with
    ``kappa = yaw_rate / max(v, 0.3)``, accel direct; the appended ego channel
    holds v0 (t0 speed) throughout, matching the open-loop eval convention, so
    the ONLY change vs the banked eval is where future actions come from.
    ⚠️ pod-verification pending — CPU-import-tested only; exact §1.12
    reproduction requires the pod checkpoint + corpus.
    """
    import torch
    from tanitad.models.metric_dynamics import accumulate_se2
    win_s, win_a = states, awE
    v = v0.clone()
    ap = torch.zeros_like(v)
    yp = torch.zeros_like(v)
    rows = []
    for j in range(k):
        z_hat = model.predictor(win_s, win_a)[1]
        aj, yj = head(win_s[:, -1], z_hat, v, ap, yp)
        rows.append(torch.stack([v * dt, torch.zeros_like(v), yj * dt], -1))
        v = (v + aj * dt).clamp_min(0.0)
        ap, yp = aj, yj
        if j < k - 1:
            kappa = yj / v.clamp_min(0.3)
            steer = torch.atan(wheelbase * kappa)
            a_next = torch.stack([steer, aj], -1)
            a_next = torch.cat([a_next, ego], -1)    # ego channel: v0 held
            win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], 1)
            win_a = torch.cat([win_a[:, 1:], a_next.unsqueeze(1)], 1)
    return accumulate_se2(torch.stack(rows, 1))


# ============================================================================ #
# E1.4 ADAPTER — v2 caches + the checkpoint's OWN grounding decoder            #
# ⛔ EVERYTHING BELOW IS ADDITIVE. The §1.12 roll (:func:`run_rollout`), its    #
# decode helpers (:func:`decode_open`, :func:`roll_closed`) and the whole      #
# analysis are untouched; the adapter never calls into them except by their    #
# public signatures. Pinned by stack/tests/test_t1_v2_adapter.py.              #
# ============================================================================ #
def implied_controls(dpose, v_prev, dt=DT, wheelbase=WHEELBASE):
    """Δpose ``[B, 3]`` (dx, dy, dyaw) + carried speed ``[B]`` -> the controls
    the corpus action contract would have recorded for that step.

    Returns ``(v [B], accel [B], yaw_rate [B], steer [B])`` where

        v        = ||(dx, dy)|| / dt      the step's realised speed
        accel    = (v - v_prev) / dt      its finite difference
        yaw_rate = dyaw / dt
        steer    = atan(wheelbase * yaw_rate / max(v, 0.3))

    ⭐ WHY THIS EXACT FORM (it is an inversion, not an invention):
    * ``physicalai.signals_at`` (physicalai.py:596-641) mints
      ``steer = atan(wheelbase * curvature)`` and ``poses[:, 3] = v``, and
      :func:`roll_closed` already closes the unicycle head's loop with
      ``kappa = yaw_rate / max(v, 0.3)`` — the SAME clamp, reused verbatim so
      the two closed loops differ only in where the motion came from;
    * speed as ``||Δ|| / dt`` is the definition :func:`controls` (the VERBATIM
      ``analyze_cl`` port that SCORES this dump) already uses, so the loop feeds
      back the same kinematics the analysis reads out. Taking ``dx`` alone would
      silently drop the free decoder's lateral component — and that component is
      exactly the defect ``UnicycleStepReadout`` exists to remove, so it must not
      be hidden by the feedback path.
    ⚠️ ``accel`` at step 0 is measured against the OBSERVED ``v0``, the same
    anchor :func:`roll_closed` starts its integrator from.
    """
    import torch
    v = torch.linalg.vector_norm(dpose[..., :2].float(), dim=-1) / dt
    accel = (v - v_prev.float()) / dt
    yaw_rate = dpose[..., 2].float() / dt
    steer = torch.atan(wheelbase * (yaw_rate / v.clamp_min(0.3)))
    return v, accel, yaw_rate, steer


def roll_closed_grounding(model, step_readout, states, awE, v0, ego, k=K,
                          dt=DT, wheelbase=WHEELBASE):
    """T1 for a checkpoint whose decoder is ``grounding.step['op']``.

    The twin of :func:`roll_closed`: each predictor step is conditioned on the
    action implied by the Δpose the readout JUST emitted (:func:`implied_controls`),
    the appended ego channel holds v0 throughout (the trunk's own rollout
    contract — the canary and ``w7_roll_rerank`` hold it constant too), and the
    waypoints are ``accumulate_se2`` of the emitted Δposes, i.e. exactly what
    ``decode_transitions`` would return on the same transitions.

    ⚠️ pod-verification pending on this box (no CUDA, no checkpoint): the
    action-feedback arithmetic and the ``[B, k, 2]`` contract are CPU-tested
    against a stub predictor/readout in ``stack/tests/test_t1_v2_adapter.py``.
    """
    import torch
    from tanitad.models.metric_dynamics import accumulate_se2
    win_s, win_a = states, awE
    v_prev = v0.float().clone()
    rows = []
    for j in range(k):
        z_hat = model.predictor(win_s, win_a)[1]
        d = step_readout(win_s[:, -1], z_hat).float()          # [B, 3]
        rows.append(d)
        if j < k - 1:
            v, accel, _yr, steer = implied_controls(d, v_prev, dt=dt,
                                                    wheelbase=wheelbase)
            a_next = torch.stack([steer, accel], -1)           # [B, 2]
            a_next = torch.cat([a_next, ego.float()], -1)      # ego channel: v0
            win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], 1)
            win_a = torch.cat([win_a[:, 1:],
                               a_next.unsqueeze(1).to(win_a.dtype)], 1)
            v_prev = v
    return accumulate_se2(torch.stack(rows, 1))


def decode_open_grounding(step_readout, trans, k):
    """Decode pre-rolled transitions with the grounding readout -> ``[B, k, 2]``.

    A one-line delegation to ``metric_dynamics.decode_transitions`` — the SAME
    call ``w7_roll_rerank.mini_eval`` makes (``decode_transitions(grounding.
    step['op'], trans, roll_k)``, w7_roll_rerank.py:649) and the pinned twin of
    the canary's ``rollout_decode`` (train_flagship_v4.py:584-586). Tier depends
    on what ``trans`` was rolled under: recorded future actions -> **T0**; the
    held t0 action -> the T1 hold-action control.
    """
    from tanitad.models.metric_dynamics import decode_transitions
    return decode_transitions(step_readout, trans, k)[0]


class V2RawEp:
    """``taniteval.data.RawEp``-shaped view over ONE v2 provider.

    ``RawEp`` (taniteval/data.py:218-225) is the surface the roll consumes:
    ``.feats`` uint8 ``[T, C, H, W]`` sliceable on dim 0, ``.actions`` / ``.poses``
    float32 ``[T, 2]`` / ``[T, 4]``. A :class:`tanitad.data.v2_dataset.LazyV2Episode`
    already exposes exactly that under the names ``.frames`` / ``.actions`` /
    ``.poses``, so this is a RENAME, not a decode: the JPEG/PNG decode stays in
    the provider's own ``_V2FramesProxy`` (partial, LRU-bounded), which is the
    module the trainers read through.
    """

    __slots__ = ("feats", "actions", "poses", "episode_id", "clip_index")

    def __init__(self, provider, clip_index: int):
        self.feats = provider.frames                # proxy; dim-0 slice decodes
        self.actions = provider.actions.float()
        self.poses = provider.poses.float()
        self.episode_id = int(getattr(provider, "episode_id", clip_index))
        self.clip_index = int(clip_index)


def uses_ext_rollout(a) -> bool:
    """``True`` when the roll must take the E1.4 adapter path.

    ⛔ The legacy ``--corpus`` + ``--head`` combination NEVER returns ``True``.
    That is the whole mechanism keeping the byte-close-validated
    :func:`run_rollout` reachable and unchanged."""
    return bool(getattr(a, "v2_val_cache", None)
                or getattr(a, "grounding_readout", False))


def _stack_scripts_on_path():
    """Put ``<repo>/stack/scripts`` on ``sys.path`` — LAZILY and only here.

    ``eval_flagship_v4`` / ``train_flagship_v4`` live in ``stack/scripts`` and
    are plain modules, not a package. Inserting that directory at IMPORT time
    would change the module-resolution surface of the legacy path too (a
    ``stack/scripts/*.py`` could shadow a stdlib/taniteval name), so it happens
    inside the adapter and nowhere else."""
    p = os.path.join(_REPO, "stack", "scripts")
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
    return p


def resolve_ext_frames(a):
    """``(cfg, cache_frame, model_frame)`` for the adapter path.

    Delegates to ``eval_flagship_v4.resolve_eval_frames`` -> the TRAINER's own
    ``resolve_v2_frames``, so this tool cannot resolve ``--v2-subframe`` a
    second, different way. With no geometry flags it returns the canonical
    256x256 frame and nothing moves."""
    _stack_scripts_on_path()
    from eval_flagship_v4 import _eval_cfg, resolve_eval_frames
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(a, cfg, label="t1_eval")
    return cfg, cache_frame, model_frame


def ext_episode_sources(a, *, cache_frame, model_frame):
    """``(loaders, provenance)`` — one zero-arg loader per episode.

    Each loader returns an object carrying ``.feats`` / ``.poses`` / ``.actions``
    (a :class:`V2RawEp` for a v2 cache, ``taniteval.data.RawEp`` for a raw
    epcache). Loading stays per-episode and lazy, exactly as the §1.12 loop does,
    so a 40-episode roll never materialises 40 decoded clips."""
    if getattr(a, "v2_val_cache", None):
        _stack_scripts_on_path()
        from eval_flagship_v4 import build_v2_val_episodes
        providers, prov = build_v2_val_episodes(
            a, cache_frame=cache_frame, train_frame=model_frame)
        n_avail = len(providers)
        if a.episodes:
            providers = providers[:a.episodes]
        loaders = [(lambda p=p, i=i: V2RawEp(p, i))
                   for i, p in enumerate(providers)]
        prov = dict(prov)
        prov.update({"corpus_format": "v2-compressed",
                     "dirs": list(a.v2_val_cache) if isinstance(
                         a.v2_val_cache, (list, tuple)) else [a.v2_val_cache],
                     "episode_ids": [int(p.episode_id) for p in providers],
                     "n_episodes": len(providers), "n_available": n_avail})
        return loaders, prov
    from taniteval.data import load_frames
    files = sorted(glob.glob(os.path.join(a.corpus, "ep_*.pt")))
    if not files:
        sys.exit(f"no ep_*.pt under {a.corpus}")
    n_avail = len(files)
    if a.episodes:
        files = files[:a.episodes]
    return ([(lambda f=f: load_frames([f])[0]) for f in files],
            {"corpus_format": "raw-epcache", "dirs": [a.corpus],
             "files": files, "n_episodes": len(files), "n_available": n_avail})


def load_ext_trunk(a, model_frame):
    """MODE-A load (model + grounding) through ``eval_flagship_v4.load_v1_from_ck``.

    IMPORTED, not reimplemented — the same loader ``stage_a_probes`` uses for
    v1/v4/v5f checkpoints, so the encoder is built at ``model_frame`` and the
    state dict loads STRICT. Returns ``(world, grounding, base_step)``."""
    import torch
    _stack_scripts_on_path()
    from eval_flagship_v4 import load_v1_from_ck
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck or "grounding" not in ck:
        sys.exit(f"[t1] {a.ckpt} has no 'model'+'grounding' keys — the adapter "
                 f"path needs a flagship trunk checkpoint (v1/v4/v5f shape). "
                 f"Keys: {sorted(ck)[:12] if isinstance(ck, dict) else type(ck)}")
    world, grounding, base_step = load_v1_from_ck(ck, a.device,
                                                 frame=model_frame)
    del ck
    return world, grounding, base_step


def run_rollout_ext(a):
    """The E1.4 roll: v2 (or raw) corpus x grounding (or unicycle-head) decode.

    Writes the SAME per-episode dump schema :func:`run_rollout` writes, so
    :func:`analyze` — and therefore every number's estimator and tier stamp — is
    reached unchanged. Returns a provenance dict for the output record.

    ⚠️ pod-verification pending: this box has no CUDA, no checkpoint and no v2
    corpus. What IS verified here (``stack/tests/test_t1_v2_adapter.py``): the
    ``(feats, poses, actions)`` adapter contract, the closed-loop feedback
    arithmetic against a stub trunk, the decode call convention against
    ``w7_roll_rerank``'s imported ``decode_transitions``, and the dump schema.
    """
    import torch

    # ⛔ tanitad FIRST, deliberately. ``taniteval.rollout`` calls
    # ``stack_guard.ensure_stack_on_path()``, which can put a DIFFERENT stack
    # tree at the front of sys.path when this host carries two (the
    # STALE_IMPORT_GUARD failure: a plausible wrong number instead of an
    # error). Binding ``tanitad`` from the tree this tool was launched from
    # before that runs makes the resolution deterministic — first import wins.
    _stack_scripts_on_path()
    # UnicycleStepReadout is needed ONLY by the legacy --head path. Importing it
    # eagerly makes an older pod-side metric_dynamics.py (which predates the
    # class) fail the ENTIRE tool, including --grounding-readout, which uses the
    # checkpoint's own decoder and never touches it. Bound lazily below.
    from tanitad.models.metric_dynamics import (gt_ego_waypoints,
                                                rollout_transitions)
    from taniteval.rollout import SPEED_SCALE

    t0 = time.time()
    cfg, cache_frame, model_frame = resolve_ext_frames(a)
    world, grounding, base_step = load_ext_trunk(a, model_frame)
    model = world.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    state_dim = int(getattr(model, "state_dim", 0) or a.head_state_dim)
    _p(f"[model] loaded STRICT in {time.time()-t0:.1f}s  step={base_step}  "
       f"state_dim={state_dim}  frame={model_frame.height}x{model_frame.width}")

    if a.grounding_readout:
        step_readout = grounding.step["op"].eval()
        for p in step_readout.parameters():
            p.requires_grad_(False)
        head = None
        dec = {"kind": "grounding.step['op'] (StepDisplacementReadout)",
               "source": "the checkpoint's OWN metric decoder",
               "open_decode": "metric_dynamics.decode_transitions — the "
                              "w7_roll_rerank / canary call convention",
               "closed_feedback": "implied_controls (physicalai.signals_at "
                                  "inverse; same clamp as roll_closed)"}
    else:
        from tanitad.models.metric_dynamics import UnicycleStepReadout
        ck = torch.load(a.head, map_location="cpu", weights_only=False)
        head = UnicycleStepReadout(state_dim, hidden=a.head_hidden,
                                   speed_input=bool(a.head_speed_input),
                                   predict_delta=bool(a.head_predict_delta)
                                   ).to(a.device)
        head.load_state_dict(ck["head"])
        head.eval()
        step_readout = None
        dec = {"kind": "UnicycleStepReadout (--head ckpt)", "ckpt": a.head,
               "hidden": a.head_hidden,
               "speed_input": bool(a.head_speed_input),
               "predict_delta": bool(a.head_predict_delta)}
    _p(f"[decoder] {dec['kind']}")

    loaders, corpus_prov = ext_episode_sources(a, cache_frame=cache_frame,
                                               model_frame=model_frame)
    _p(f"[corpus] {corpus_prov['corpus_format']} — episodes "
       f"{corpus_prov['n_episodes']} of {corpus_prov['n_available']} "
       f"(§1.12 grid default: first {EPISODES_DEFAULT})")
    os.makedirs(a.dump_dir, exist_ok=True)

    dev = a.device
    dev_type = torch.device(dev).type
    amp_on = (dev_type == "cuda") and bool(a.grounding_readout) \
        and not a.no_amp
    k, w, dt, stride = a.horizon_k, a.window, a.dt, max(1, int(a.window_stride))
    arm_keys = ["cl"] + (["ol"] if a.with_t0_open_loop else []) \
        + (["ha"] if a.with_hold_action else [])
    _p(f"[grid] window={w} k={k} dt={dt} stride={stride} chunk={a.chunk} "
       f"amp={amp_on} arms={arm_keys}")
    n_win = 0
    t0 = time.time()
    for fi, load_ep in enumerate(loaders):
        ep = load_ep()
        poses = ep.poses.float()
        T = min(int(ep.feats.shape[0]), int(poses.shape[0]),
                int(ep.actions.shape[0]))
        starts = list(range(0, T - w - k, stride))
        acc = {kk: [] for kk in ["g"] + arm_keys}
        lastl = []
        with torch.no_grad():
            for i0 in range(0, len(starts), a.chunk):
                ch = starts[i0:i0 + a.chunk]
                last = torch.tensor([s + w - 1 for s in ch])
                fw = torch.stack([torch.as_tensor(ep.feats[s:s + w])
                                  for s in ch]).to(dev).float().div_(255.0)
                aw = torch.stack([ep.actions[s:s + w]
                                  for s in ch]).to(dev).float()
                pl = poses[last].to(dev)
                ego = pl[:, 3:4] / SPEED_SCALE
                awE = torch.cat([aw, ego[:, None].expand(-1, aw.shape[1], -1)],
                                -1)
                v0 = pl[:, 3]
                with torch.autocast(dev_type, dtype=torch.bfloat16,
                                    enabled=amp_on):
                    states = model.encode_window(fw)
                    aw3 = awE.to(states.dtype)
                    if a.grounding_readout:
                        cl = roll_closed_grounding(model, step_readout, states,
                                                   aw3, v0, ego, k=k, dt=dt,
                                                   wheelbase=a.wheelbase)
                    else:
                        cl = roll_closed(model, head, states, awE, v0, ego,
                                         k=k, dt=dt, wheelbase=a.wheelbase)
                    acc["cl"].append(cl.float().cpu().numpy())
                    if a.with_t0_open_loop:
                        fa = torch.stack([ep.actions[s + w:s + w + k]
                                          for s in ch]).to(dev).float()
                        faE = torch.cat(
                            [fa, ego[:, None].expand(-1, fa.shape[1], -1)], -1)
                        tr = rollout_transitions(model.predictor, states, aw3,
                                                 faE.to(states.dtype), k)
                        ol = (decode_open_grounding(step_readout, tr, k)
                              if a.grounding_readout
                              else decode_open(head, tr, v0, dt=dt))
                        acc["ol"].append(ol.float().cpu().numpy())
                    if a.with_hold_action:
                        faH = aw3[:, -1:, :].expand(-1, k, -1)
                        trH = rollout_transitions(model.predictor, states, aw3,
                                                  faH, k)
                        ha = (decode_open_grounding(step_readout, trH, k)
                              if a.grounding_readout
                              else decode_open(head, trH, v0, dt=dt))
                        acc["ha"].append(ha.float().cpu().numpy())
                fp = torch.stack([poses[s + w:s + w + k]
                                  for s in ch]).to(dev)
                acc["g"].append(gt_ego_waypoints(pl, fp, list(range(1, k + 1)))
                                .float().cpu().numpy())
                lastl += [int(x) for x in last]
        np.savez_compressed(
            os.path.join(a.dump_dir, f"ep{fi:03d}.npz"),
            **{kk: np.concatenate(v).astype(np.float32)
               for kk, v in acc.items()},
            ws=np.array(lastl))
        n_win += len(lastl)
        _p(f"  [{fi+1}/{len(loaders)}] {len(lastl)} windows  "
           f"{time.time()-t0:.0f}s")
    _p("T1_DUMP_DONE")
    return {"path": "run_rollout_ext (E1.4 adapter)",
            "decoder": dec, "corpus": corpus_prov,
            "base_step": base_step, "state_dim": state_dim,
            "cache_frame": cache_frame.to_dict(),
            "model_frame": model_frame.to_dict(),
            "v2_subframe": getattr(a, "v2_subframe", None),
            "grid": {"window": w, "horizon_k": k, "dt_s": dt,
                     "window_stride": stride, "n_windows": n_win,
                     "episodes": len(loaders)},
            "amp_bf16": amp_on,
            "arms": arm_keys,
            "canary_equivalent_arm": ("ol — decode_transitions over the "
                                      "recorded-action roll IS the WM canary "
                                      "quantity (train_flagship_v4.py:584-586)"
                                      if (a.grounding_readout
                                          and a.with_t0_open_loop) else None),
            "wallclock_s": round(time.time() - t0, 1)}


def run_rollout(a):
    """Roll the checkpoint over the corpus and write the per-episode dump.

    Faithful parameterisation of ``closed_loop_dump.py``'s main loop: stride-1
    window grid, chunked batches, first ``--episodes`` files of the sorted
    corpus (§1.12 took 40). The T1 closed-loop arm ('cl') always rolls; the T0
    teacher-forced arm ('ol') and the hold-action control ('ha') are opt-in.
    ⚠️ pod-verification pending — needs CUDA + the checkpoint; this box has
    neither. The import/arg surface and the dump schema are CPU-tested.
    """
    import torch

    from taniteval import loaders
    from taniteval.data import load_frames
    from taniteval.rollout import SPEED_SCALE
    from tanitad.models.metric_dynamics import (UnicycleStepReadout,
                                                gt_ego_waypoints,
                                                rollout_transitions)

    entry = {"arch": "flagship-worldmodel-v2" if a.run_config
             else "flagship-worldmodel",
             "ckpt": a.ckpt, "run_config": a.run_config,
             "speed_input": bool(a.speed_input)}
    t0 = time.time()
    h = loaders.load(entry, device=a.device)
    model = h["model"].eval()
    for p in model.parameters():
        p.requires_grad_(False)
    state_dim = int(h.get("state_dim") or a.head_state_dim)
    _p(f"[model] loaded STRICT in {time.time()-t0:.1f}s  arch={entry['arch']}"
       f"  step={h.get('step')}  state_dim={state_dim}")

    ck = torch.load(a.head, map_location="cpu", weights_only=False)
    head = UnicycleStepReadout(state_dim, hidden=a.head_hidden,
                               speed_input=bool(a.head_speed_input),
                               predict_delta=bool(a.head_predict_delta)
                               ).to(a.device)
    head.load_state_dict(ck["head"])
    head.eval()
    _p(f"[head] {a.head}  hidden={a.head_hidden}  "
       f"speed_input={a.head_speed_input}  predict_delta={a.head_predict_delta}")

    files = sorted(glob.glob(os.path.join(a.corpus, "ep_*.pt")))
    if not files:
        sys.exit(f"no ep_*.pt under {a.corpus}")
    n_avail = len(files)
    if a.episodes:
        files = files[:a.episodes]
    _p(f"[corpus] {a.corpus}  episodes {len(files)} of {n_avail} "
       f"(§1.12 grid default: first {EPISODES_DEFAULT})")
    os.makedirs(a.dump_dir, exist_ok=True)

    k, w, dt = a.horizon_k, a.window, a.dt
    arm_keys = ["cl"] + (["ol"] if a.with_t0_open_loop else []) \
        + (["ha"] if a.with_hold_action else [])
    t0 = time.time()
    for fi, f in enumerate(files):
        ep = load_frames([f])[0]
        poses = ep.poses.float()
        T = min(ep.feats.shape[0], poses.shape[0], ep.actions.shape[0])
        starts = list(range(0, T - w - k))          # stride-1, the §1.12 grid
        acc = {kk: [] for kk in ["g"] + arm_keys}
        lastl = []
        with torch.no_grad():
            for i0 in range(0, len(starts), a.chunk):
                ch = starts[i0:i0 + a.chunk]
                last = torch.tensor([s + w - 1 for s in ch])
                fw = torch.stack([torch.as_tensor(ep.feats[s:s + w])
                                  for s in ch]).to(a.device).float().div_(255.0)
                aw = torch.stack([ep.actions[s:s + w]
                                  for s in ch]).to(a.device).float()
                pl = poses[last].to(a.device)
                ego = pl[:, 3:4] / SPEED_SCALE
                awE = torch.cat([aw, ego[:, None].expand(-1, aw.shape[1], -1)],
                                -1)
                states = model.encode_window(fw)
                v0 = pl[:, 3]
                acc["cl"].append(roll_closed(model, head, states, awE, v0, ego,
                                             k=k, dt=dt,
                                             wheelbase=a.wheelbase
                                             ).float().cpu().numpy())
                if a.with_t0_open_loop:
                    fa = torch.stack([ep.actions[s + w:s + w + k]
                                      for s in ch]).to(a.device).float()
                    faE = torch.cat([fa, ego[:, None].expand(-1, fa.shape[1],
                                                             -1)], -1)
                    tr = rollout_transitions(model.predictor, states, awE,
                                             faE, k)
                    acc["ol"].append(decode_open(head, tr, v0, dt=dt)
                                     .float().cpu().numpy())
                if a.with_hold_action:
                    faH = awE[:, -1:, :].expand(-1, k, -1)
                    trH = rollout_transitions(model.predictor, states, awE,
                                              faH, k)
                    acc["ha"].append(decode_open(head, trH, v0, dt=dt)
                                     .float().cpu().numpy())
                fp = torch.stack([poses[s + w:s + w + k]
                                  for s in ch]).to(a.device)
                acc["g"].append(gt_ego_waypoints(pl, fp, list(range(1, k + 1)))
                                .cpu().numpy())
                lastl += [int(x) for x in last]
        np.savez_compressed(
            os.path.join(a.dump_dir, f"ep{fi:03d}.npz"),
            **{kk: np.concatenate(v).astype(np.float32)
               for kk, v in acc.items()},
            ws=np.array(lastl))
        _p(f"  [{fi+1}/{len(files)}] {time.time()-t0:.0f}s")
    _p("T1_DUMP_DONE")
    return files


# ============================================================================ #
# CLI                                                                          #
# ============================================================================ #
def _parse_tiers(s):
    out = {}
    for part in filter(None, (s or "").split(",")):
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="T1 action-closed-loop eval (EVAL_DOCTRINE.md tier T1 — "
                    "the PRIMARY offline eval). Promoted from the §1.12 "
                    "closed_loop_dump.py/analyze_cl.py pair.")
    # -- eval_four_families-style arm/corpus surface ------------------------- #
    ap.add_argument("--ckpt", help="flagship checkpoint (rollout mode)")
    ap.add_argument("--run-config", default=None,
                    help="the run's config.json -> model rebuilt from the "
                         "RUN'S OWN cfg and loaded STRICT")
    ap.add_argument("--arm", required=True, help="label for the output record")
    ap.add_argument("--corpus", default=None, help="dir of ep_*.pt episodes")
    # -- E1.4 adapter: the v2 corpus + the checkpoint's own decoder ----------- #
    ap.add_argument("--v2-val-cache", default=None, nargs="+",
                    help="v2 compressed VAL split dir(s) of *.v2ep.pt — the "
                         "w120 flagship corpus format. Exactly one of this and "
                         "--corpus, never both (two CORPUS FORMATS, not two "
                         "sources to mix). Selects run_rollout_ext.")
    ap.add_argument("--v2-lru", type=int, default=64,
                    help="v2 payload LRU per cache dir (measure YOUR cache's "
                         "MB/clip before raising it — 33 MB/clip on the w120 "
                         "PNG caches)")
    ap.add_argument("--v2-subframe", default=None, metavar="HxW",
                    help="centred sub-frame the model reads (e.g. 176x624) — "
                         "MUST match the run; cross-checked vs the cache")
    ap.add_argument("--require-parity", action="store_true")
    ap.add_argument("--grounding-readout", action="store_true",
                    help="decode with the CHECKPOINT'S OWN grounding.step['op'] "
                         "(StepDisplacementReadout) instead of a --head ckpt — "
                         "the v5f/v5.8f decoder. Selects run_rollout_ext.")
    ap.add_argument("--window-stride", type=int, default=1,
                    help="window grid stride on the adapter path; 1 = the "
                         "§1.12 stride-1 grid (default). Recorded in the "
                         "output either way.")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable bf16 autocast on the adapter path. AMP is ON "
                         "by default for --grounding-readout on cuda because "
                         "that is the regime the canary/W7/stage_a numbers this "
                         "arm is compared against were produced in.")
    ap.add_argument("--out", required=True, help="JSON output FILE (not a dir)")
    ap.add_argument("--episodes", type=int, default=EPISODES_DEFAULT,
                    help=f"first N episodes of the sorted corpus; the §1.12 "
                         f"grid took {EPISODES_DEFAULT}. 0 = ALL (recorded in "
                         f"the output either way).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--speed-input", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    # -- frame/geometry (the §1.12 grid, parameterised) ---------------------- #
    ap.add_argument("--window", type=int, default=W)
    ap.add_argument("--horizon-k", type=int, default=K)
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--wheelbase", type=float, default=WHEELBASE,
                    help="signals_at legacy wheelbase for steer=atan(L*kappa)")
    ap.add_argument("--chunk", type=int, default=16)
    # -- the unicycle head ---------------------------------------------------- #
    ap.add_argument("--head", help="UnicycleStepReadout ckpt (key 'head')")
    ap.add_argument("--head-hidden", type=int, default=512)
    ap.add_argument("--head-state-dim", type=int, default=2048,
                    help="fallback when the loader reports no state_dim")
    ap.add_argument("--head-speed-input", action="store_true",
                    help="§1.12 heads were speed_input=False")
    ap.add_argument("--head-predict-delta", action="store_true",
                    help="§1.12 heads were predict_delta=False")
    # -- arms ----------------------------------------------------------------- #
    ap.add_argument("--with-t0-open-loop", action="store_true",
                    help="ALSO roll the T0 teacher-forced arm ('ol'). ⛔ OFF by "
                         "default: T0 consumes the RECORDED future actions and "
                         "is quotable only as prediction quality, NEVER as "
                         "driving performance (EVAL_DOCTRINE.md).")
    ap.add_argument("--with-hold-action", action="store_true",
                    help="also roll the hold-t0-action control arm ('ha', T1 "
                         "battery)")
    ap.add_argument("--tiers", default="",
                    help="tier overrides/additions, e.g. 'myarm=T1,ref=T0'. "
                         "Every arm in the dump must resolve to a tier.")
    # -- dump / analyze separability ------------------------------------------ #
    ap.add_argument("--dump-dir", default=None,
                    help="where the roll writes per-episode npz (rollout mode)")
    ap.add_argument("--analyze-only", default=None, metavar="DUMP_DIR",
                    help="consume an existing dump (e.g. the §1.12 dump_cl) "
                         "with NO model/GPU — the pod pass and the analysis "
                         "are separable")
    ap.add_argument("--dump-only", action="store_true",
                    help="roll + dump, skip the analysis (pod-side half)")
    # -- optional attachments -------------------------------------------------- #
    ap.add_argument("--lead", default=None,
                    help="lead block (tools/build_lead_block.py) row-aligned "
                         "to THIS dump's stride-1 grid; without it the "
                         "distance-keeping half of LONGITUDINAL is "
                         "UNAVAILABLE — a WORK ITEM, not a pass")
    ap.add_argument("--byte-check-dump", default=None,
                    help="reference dump dir for the §1.12 byte-close gate")
    ap.add_argument("--byte-check-arm", default="o16")
    ap.add_argument("--byte-check-key", default="b",
                    help="key in the reference dump ('b' = the banked v1.6 "
                         "open-loop arm in the v16 dump)")
    # the geometry flags come from tanitad.geometry itself — never re-spelled
    # here, because "a geometry can never be spelled two different ways" is the
    # whole point of that module. Absent stack -> the flags are absent, and the
    # adapter path refuses below with the import error rather than guessing.
    geom_err = None
    try:
        from tanitad.geometry import add_geometry_args
        add_geometry_args(ap)          # --frame-h/-w/--frame-hfov/--projection
    except Exception as ex:            # noqa: BLE001 (no stack on this host)
        geom_err = f"{type(ex).__name__}: {ex}"
    a = ap.parse_args(argv)

    if os.path.isdir(a.out):
        sys.exit(f"--out must be a FILE, got a directory: {a.out}")
    ext = uses_ext_rollout(a)
    if a.analyze_only is None:
        if a.corpus and a.v2_val_cache:
            sys.exit("[t1] --corpus (raw epcache) and --v2-val-cache (v2 "
                     "compressed) are two CORPUS FORMATS, not two sources to "
                     "mix. Pass exactly one.")
        if a.head and a.grounding_readout:
            sys.exit("[t1] --head (a separate UnicycleStepReadout ckpt) and "
                     "--grounding-readout (the checkpoint's OWN "
                     "grounding.step['op']) are two DECODERS. Pass exactly "
                     "one — scoring one arm's decoder under the other's name "
                     "is unattributable.")
        req = [(a.ckpt, "--ckpt"), (a.dump_dir, "--dump-dir"),
               (a.corpus or a.v2_val_cache, "--corpus or --v2-val-cache"),
               (a.head or a.grounding_readout, "--head or --grounding-readout")]
        for r, name in req:
            if not r:
                sys.exit(f"rollout mode needs {name} (or use --analyze-only)")
        if ext and geom_err:
            sys.exit(f"[t1] the adapter path (--v2-val-cache / "
                     f"--grounding-readout) needs the tanitad stack on "
                     f"PYTHONPATH, but importing tanitad.geometry failed: "
                     f"{geom_err}")

    rollout_prov = None
    if a.analyze_only:
        dump_dir = a.analyze_only
    else:
        if ext:
            rollout_prov = run_rollout_ext(a)
        else:
            run_rollout(a)
        dump_dir = a.dump_dir
        if a.dump_only:
            _p(f"[dump-only] dump at {dump_dir}; analyze later with "
               f"--analyze-only {dump_dir}")
            return

    dump_files = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    if not dump_files:
        sys.exit(f"no ep*.npz under {dump_dir}")

    lead = None
    if a.lead:
        import torch
        lead = torch.load(a.lead, map_location="cpu", weights_only=False)

    byte_check = None
    if a.byte_check_dump:
        ref_files = sorted(glob.glob(os.path.join(a.byte_check_dump,
                                                  "ep*.npz")))
        byte_check = (a.byte_check_arm, ref_files, a.byte_check_key)

    rec = analyze(dump_files, tiers=_parse_tiers(a.tiers),
                  n_boot=a.n_boot, dt=a.dt, lead=lead, byte_check=byte_check)
    corpus_dir = a.corpus or (a.v2_val_cache[0] if a.v2_val_cache else None)
    rec.update({"arm": a.arm, "ckpt": a.ckpt, "run_config": a.run_config,
                "corpus": a.corpus,
                "v2_val_cache": a.v2_val_cache,
                "corpus_key": (os.path.basename(str(corpus_dir).rstrip("/"))
                               if corpus_dir else None),
                "episodes_flag": a.episodes, "dump_dir": dump_dir,
                "lead_block": a.lead,
                "decoder": ("grounding.step['op']" if a.grounding_readout
                            else ("UnicycleStepReadout(--head)" if a.head
                                  else None)),
                "rollout_provenance": rollout_prov,
                "rollout_path": (
                    "n/a — analyze-only (no roll in this invocation)"
                    if a.analyze_only else
                    ("run_rollout_ext (E1.4 adapter)" if ext
                     else "run_rollout (§1.12 byte-close path)")),
                "mode": "analyze-only" if a.analyze_only else "rollout+analyze"})
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    _p(f"[out] {a.out}")
    for arm, blk in rec["arms"].items():
        s = blk["s_curve"].get("masked", {})
        _p(f"  {arm:6s} tier={blk['tier']}  "
           f"S-rate(masked)={s.get('rate', 'UNAVAILABLE')}  "
           f"lag={blk['lag'].get('lag_accel_s_mean', 'UNAVAILABLE')}  "
           f"families_unavailable="
           f"{blk['four_families']['_families_unavailable']}")


if __name__ == "__main__":
    main()
