"""D1-D3 gate runner with instrument-doctrine gating (backlog #1, WP6/Arch).

WHY THIS EXISTS
---------------
Phase 0 Plan §4 defines falsifiable gates D1-D9. D1-D3 are the *decode* gates:

    D1  encoder state decodable          frozen-probe ADE@1s < 0.5 m (BEV)
                                          / < 1.0 m (camera); I2, I3 pass; vs global-pool
    D2  imagination usable for selection  direction acc > 0.7; imag-rel < 0.8;
                                          I1 ~ 1.0 first; vs persistence
    D3  trajectory decode from imag       imagined-ADE@2s <= 1.5x oracle-ADE@2s;
                                          probe_real vs probe_imag (A3)

The **program rule** (Phase 0 Plan §4, DECISIONS D-004): *no architecture change
may be motivated by a gate that has not passed its instrument rows.* This module
encodes that rule mechanically. Every gate first assembles its I1-I4 instrument
rows (protocol §6: instruments FIRST) and a gate can only be marked ``passed`` if
it is first ``admissible`` (instruments clear their bars). A gate whose
instruments fail is reported ``BLOCKED``, never ``FAIL`` — the number is not a
claim, the harness is not trusted.

WHAT D1-D3 ARE AND ARE NOT (research finding, 2026-07-14 note)
--------------------------------------------------------------
The large-scale JEPA-WM planning ablation (arXiv 2512.24497, "What Drives Success
in Physical Planning with JEPWMs?") reports that models which unroll faithfully do
NOT necessarily plan well: *decode/probe quality does not reliably predict planning
success*. Therefore D1-D3 are **necessary, not sufficient** instrument gates on the
representation; the closed-loop gates D4-D6 remain the arbiters of the hierarchy
edge. This module labels D1-D3 accordingly and never lets a passing decode gate be
read as a driving-competence claim. (G-AI1 instrument doctrine.)

SEAM WITH BENCHMARKS & EVAL (Thursday)
--------------------------------------
ADE/FDE here are the *standard* recognizable metrics (D-007) and are owned by the
gate wiring. The *custom* TanitAD metrics (LAL/TMS/OKRI/CNCE/LOPS) are Thursday's
suite: each ``run_d*`` accepts ``extra_metrics={name: callable}`` merged into the
gate's ``metrics`` block, so the custom suite plugs in without touching this file.

This module is standalone. ``tanitad`` must be importable (editable stack install);
it composes the *existing* I1-I4 primitives from ``tanitad.instruments.checks`` and
the frozen ``RidgeProbe`` from ``tanitad.models.readout`` — it does not reimplement
them. Proposed target on integration: ``stack/tanitad/eval/gates.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor

from tanitad.instruments.checks import (
    i2_batch_consistency,
    i3_episode_split,
    i4_imag_relative,
    i7_task_identity,
)
from tanitad.models.readout import RidgeProbe

# --------------------------------------------------------------------------- #
# Thresholds (Phase 0 Plan §4; D2 redefined per D-017/A13). Named constants.  #
# --------------------------------------------------------------------------- #
D1_ADE_MAX = {"bev": 0.5, "camera": 1.0}   # metres, ADE@1s
D2_DIR_ACC_MIN = 0.7                        # calibrated OR forward-dynamics (P4)
D3_RATIO_MAX = 1.5                          # imagined-ADE@2s / oracle-ADE@2s
I1_FLOOR = 0.9                              # oracle-decode R^2 sanity floor
I2_TOL = 1e-4
I4_FLOOR = 1.0                              # persistence bar (D3 multi-step context)
# D-017/A13: for D2, imag-rel is a DIAGNOSTIC metric, never an admissibility
# row — control was measured usable at imag-rel 1.27; action DISCRIMINATION in
# decoded-state space is what bounds control. D3 keeps its I4 row (multi-step
# decode is a different claim).


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class I2Input:
    """Everything I2 (batch-consistency, D-004) needs: the encoder + real frames."""
    encode_fn: Callable[[Tensor], Tensor]
    frames: Tensor
    batch_size: int = 32


@dataclass
class GateReport:
    gate: str
    claim: str
    admissible: bool                 # instrument rows clear their bars -> a claim is allowed
    passed: bool                     # admissible AND metric beats threshold
    instruments: list[dict]          # I-rows FIRST (protocol §6)
    metrics: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    ablation: dict = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    verdict: str = ""

    def to_dict(self) -> dict:
        return {
            "gate": self.gate,
            "claim": self.claim,
            "instruments": self.instruments,     # FIRST — a run without these does not exist (D-004)
            "admissible": self.admissible,
            "passed": self.passed,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "ablation": self.ablation,
            "blockers": self.blockers,
            "verdict": self.verdict,
        }

    @property
    def status(self) -> str:
        if not self.admissible:
            return "BLOCKED"
        return "PASS" if self.passed else "FAIL"


# --------------------------------------------------------------------------- #
# Standard metric primitives                                                   #
# --------------------------------------------------------------------------- #
def _as_traj(xy: Tensor) -> Tensor:
    """[N, 2] -> [N, 1, 2]; [N, H, 2] unchanged."""
    if xy.dim() == 2:
        return xy.unsqueeze(1)
    assert xy.dim() == 3 and xy.shape[-1] == 2, f"bad traj shape {tuple(xy.shape)}"
    return xy


def ade_fde(pred_xy: Tensor, true_xy: Tensor) -> tuple[float, float]:
    """Average / final displacement error (metres). Accepts [N,2] or [N,H,2]."""
    p, t = _as_traj(pred_xy), _as_traj(true_xy)
    d = (p - t).norm(dim=-1)                       # [N, H]
    return float(d.mean()), float(d[:, -1].mean())


def split_by_episode(episode_ids: Sequence[int], val_frac: float = 0.2,
                     seed: int = 0) -> tuple[list[int], list[int]]:
    """I3-correct sample split: episodes are disjoint between train and val.

    Returns (train sample indices, val sample indices). Built on the shared
    ``i3_episode_split`` primitive so route/day leakage cannot creep back in.
    """
    uniq = sorted(set(int(e) for e in episode_ids))
    assert len(uniq) >= 2, "need >=2 distinct episodes for an I3 split"
    _train_ep, val_ep = i3_episode_split(uniq, val_frac=val_frac, seed=seed)
    val_set = set(val_ep)
    train_idx = [i for i, e in enumerate(episode_ids) if int(e) not in val_set]
    val_idx = [i for i, e in enumerate(episode_ids) if int(e) in val_set]
    return train_idx, val_idx


# --------------------------------------------------------------------------- #
# Instrument rows                                                              #
# --------------------------------------------------------------------------- #
def _i1_row(fit_r2: float) -> dict:
    return {"row": "I1", "name": "oracle-decode (probe fit R^2)",
            "value": fit_r2, "pass": bool(fit_r2 >= I1_FLOOR), "floor": I1_FLOOR}


def _i2_row(i2: I2Input | None) -> dict:
    if i2 is None:
        return {"row": "I2", "name": "batch-consistency", "value": None,
                "pass": False, "note": "NOT SUPPLIED — D-004 mandates I2"}
    ok, rel = i2_batch_consistency(i2.encode_fn, i2.frames, i2.batch_size, tol=I2_TOL)
    return {"row": "I2", "name": "batch-consistency", "value": rel,
            "pass": bool(ok), "tol": I2_TOL,
            "note": "batch-1 vs batched encoding max rel deviation"}


def _i3_row(episode_ids: Sequence[int], train_idx, val_idx) -> dict:
    tr = {int(episode_ids[i]) for i in train_idx}
    va = {int(episode_ids[i]) for i in val_idx}
    overlap = len(tr & va)
    return {"row": "I3", "name": "episode-level split", "value": overlap,
            "pass": bool(overlap == 0 and len(va) > 0),
            "note": f"{len(tr)} train / {len(va)} val episodes, overlap={overlap}"}


def _i4_row(z_pred: Tensor, z_true: Tensor, z_prev: Tensor, ceil: float) -> dict:
    rel = i4_imag_relative(z_pred, z_true, z_prev)
    return {"row": "I4", "name": "imag-relative (vs persistence)", "value": rel,
            "pass": bool(rel < ceil), "ceil": ceil,
            "note": "||z_hat-z||/||z-z_prev||; < ceil beats persistence"}


def _i7_row(fit_meta: Mapping | None, run_meta: Mapping | None) -> dict | None:
    """I7 task-identity (D-017): probe-fit corpus and eval stream must be the
    same task (same fingerprint). Only emitted when metadata is supplied."""
    if fit_meta is None and run_meta is None:
        return None
    ok, bad = i7_task_identity(dict(fit_meta or {}), dict(run_meta or {}))
    return {"row": "I7", "name": "task-identity fingerprint", "value": bad,
            "pass": bool(ok),
            "note": "fit vs eval corpus fingerprint mismatch keys" if bad
            else "fit and eval fingerprints identical"}


def _admissible(rows: list[dict]) -> tuple[bool, list[str]]:
    blockers = [f"{r['row']}({r.get('value')})" for r in rows if not r["pass"]]
    return (len(blockers) == 0), blockers


def _fit_probe(feats: Tensor, targets: Tensor, alpha: float) -> tuple[RidgeProbe, float]:
    probe = RidgeProbe(alpha=alpha).fit(feats, targets)
    return probe, probe.r2(feats, targets)          # fit R^2 == I1 sanity


# --------------------------------------------------------------------------- #
# D1 — encoder state decodable                                                 #
# --------------------------------------------------------------------------- #
def run_d1(states: Tensor, targets_xy: Tensor, episode_ids: Sequence[int],
           unit: str = "camera", i2: I2Input | None = None,
           pooled_states: Tensor | None = None, alpha: float = 1e-3,
           val_frac: float = 0.2, seed: int = 0,
           extra_metrics: Mapping[str, Callable] | None = None) -> GateReport:
    """D1: can a frozen probe read metric ego position out of the encoder state?

    ``states`` [N, S] (grid readout), ``targets_xy`` [N,2] or [N,H,2] ego
    waypoints. ``pooled_states`` [N, Sp] runs the *vs global-pool* ablation (A7).
    """
    assert unit in D1_ADE_MAX, f"unit must be one of {list(D1_ADE_MAX)}"
    tr, va = split_by_episode(episode_ids, val_frac, seed)
    probe, fit_r2 = _fit_probe(states[tr], _as_traj(targets_xy)[tr].flatten(1), alpha)
    pred_val = probe.predict(states[va]).reshape(len(va), -1, 2)
    ade, fde = ade_fde(pred_val, _as_traj(targets_xy)[va])

    rows = [_i1_row(fit_r2), _i2_row(i2), _i3_row(episode_ids, tr, va)]
    admissible, blockers = _admissible(rows)
    thr = D1_ADE_MAX[unit]
    passed = admissible and ade < thr

    ablation: dict = {}
    if pooled_states is not None:
        pool_probe, _ = _fit_probe(pooled_states[tr], _as_traj(targets_xy)[tr].flatten(1), alpha)
        pool_pred = pool_probe.predict(pooled_states[va]).reshape(len(va), -1, 2)
        pool_ade, _ = ade_fde(pool_pred, _as_traj(targets_xy)[va])
        ablation = {"global_pool_ade": pool_ade, "grid_ade": ade,
                    "grid_beats_pool": bool(ade <= pool_ade),
                    "note": "A7: spatial grid readout should decode >= global pooling"}

    metrics = {"ade@1s": ade, "fde@1s": fde}
    if extra_metrics:
        metrics.update({k: fn(pred_val, _as_traj(targets_xy)[va]) for k, fn in extra_metrics.items()})

    verdict = (f"BLOCKED: instruments failed {blockers}" if not admissible else
               f"{'PASS' if passed else 'FAIL'}: ADE@1s={ade:.3f} m vs <{thr} ({unit}). "
               f"Necessary-not-sufficient (2512.24497): closed-loop D4-D6 arbitrate.")
    return GateReport("D1", "encoder state decodable", admissible, passed,
                      rows, metrics, {"ade@1s_max": thr, "unit": unit}, ablation,
                      blockers, verdict)


# --------------------------------------------------------------------------- #
# D2 — imagination usable for selection                                        #
# --------------------------------------------------------------------------- #
def run_d2(z_prev: Tensor, z_true: Tensor, z_imag: Tensor, disp_true_xy: Tensor,
           episode_ids: Sequence[int], i2: I2Input | None = None,
           disp_persist_xy: Tensor | None = None,
           actions: Tensor | None = None, prev_state: Tensor | None = None,
           fit_meta: Mapping | None = None, run_meta: Mapping | None = None,
           alpha: float = 1e-3, val_frac: float = 0.2, seed: int = 0,
           extra_metrics: Mapping[str, Callable] | None = None) -> GateReport:
    """D2 (redefined per D-017/A13): can candidate maneuvers be RANKED — via the
    calibrated imagination probe (P1) OR the forward-dynamics probe (P4)?

    ``z_imag`` [N, S] imagined next-state, ``z_true`` [N, S] real next-state,
    ``z_prev`` [N, S] current state, ``disp_true_xy`` [N, 2] true ego displacement.
    P1: probe calibrated on imagined latents (A3). P4 (when ``actions`` [N, A] and
    ``prev_state`` [N, P] — a LOW-D decoded/proprioceptive state, e.g. (v, yaw) —
    are supplied): ridge [prev_state ⊕ action] → displacement, NO imagination in
    the loop; the strongest+cheapest readout in the ALPS egocentric run (0.76).
    Gate passes if EITHER path clears D2_DIR_ACC_MIN. imag-rel is reported as a
    DIAGNOSTIC metric — it does not gate admissibility (A13: usable at 1.27).
    """
    tr, va = split_by_episode(episode_ids, val_frac, seed)
    truth = disp_true_xy[va]

    # P1 — calibrated imagination probe (A3)
    probe, fit_r2 = _fit_probe(z_imag[tr], disp_true_xy[tr], alpha)
    pred = probe.predict(z_imag[va])                      # [n_val, 2]
    cos = torch.cosine_similarity(pred, truth, dim=-1)
    dir_acc = float((cos > 0).float().mean())

    # P4 — forward-dynamics probe in low-D decoded-state space (D-017)
    p4_dir_acc, p4_fit_r2 = None, None
    if actions is not None and prev_state is not None:
        fd_feats = torch.cat([prev_state, actions], dim=-1)
        fd_probe, p4_fit_r2 = _fit_probe(fd_feats[tr], disp_true_xy[tr], alpha)
        fd_pred = fd_probe.predict(fd_feats[va])
        fd_cos = torch.cosine_similarity(fd_pred, truth, dim=-1)
        p4_dir_acc = float((fd_cos > 0).float().mean())

    # baselines (direction persistence is a reference)
    if disp_persist_xy is not None:
        base_cos = torch.cosine_similarity(disp_persist_xy[va], truth, dim=-1)
        base_dir_acc = float((base_cos > 0).float().mean())
    else:
        base_dir_acc = 0.5

    # I1 applies to the ACTIVE readout path: the gate passes via P1 OR P4, so
    # harness sanity is satisfied if EITHER probe family fits (the claimed path
    # must be the sane one; both fit-R^2s are reported in metrics).
    i1_fit = max(fit_r2, p4_fit_r2 if p4_fit_r2 is not None else 0.0)
    rows = [_i1_row(i1_fit), _i2_row(i2), _i3_row(episode_ids, tr, va)]
    i7 = _i7_row(fit_meta, run_meta)
    if i7 is not None:
        rows.append(i7)
    admissible, blockers = _admissible(rows)
    p1_ok = fit_r2 >= I1_FLOOR and dir_acc > D2_DIR_ACC_MIN
    p4_ok = (p4_fit_r2 is not None and p4_fit_r2 >= I1_FLOOR
             and (p4_dir_acc or 0.0) > D2_DIR_ACC_MIN)
    passed = admissible and (p1_ok or p4_ok)

    best_acc = max(dir_acc if fit_r2 >= I1_FLOOR else 0.0,
                   (p4_dir_acc or 0.0) if (p4_fit_r2 or 0.0) >= I1_FLOOR else 0.0)
    metrics = {"direction_acc": dir_acc, "mean_cos": float(cos.mean()),
               "p1_fit_r2": fit_r2,
               "p4_forward_dynamics_dir_acc": p4_dir_acc,
               "p4_fit_r2": p4_fit_r2,
               "imag_rel_diagnostic": i4_imag_relative(z_imag, z_true, z_prev),
               "baseline_persistence_dir_acc": base_dir_acc,
               "beats_persistence": bool(best_acc > base_dir_acc)}
    if extra_metrics:
        metrics.update({k: fn(pred, truth) for k, fn in extra_metrics.items()})

    p4_txt = f", P4={p4_dir_acc:.3f}" if p4_dir_acc is not None else ""
    verdict = (f"BLOCKED: instruments failed {blockers}" if not admissible else
               f"{'PASS' if passed else 'FAIL'}: dir-acc P1={dir_acc:.3f}{p4_txt} "
               f"vs >{D2_DIR_ACC_MIN} (persistence {base_dir_acc:.3f}); "
               f"imag-rel={metrics['imag_rel_diagnostic']:.2f} is diagnostic (D-017).")
    return GateReport("D2", "imagination usable for selection (P1 or P4)",
                      admissible, passed, rows, metrics,
                      {"dir_acc_min": D2_DIR_ACC_MIN,
                       "paths": "P1 calibrated OR P4 forward-dynamics"},
                      {}, blockers, verdict)


# --------------------------------------------------------------------------- #
# D3 — trajectory decode from imagination                                      #
# --------------------------------------------------------------------------- #
def run_d3(z_prev: Tensor, z_true_future: Tensor, z_imag_future: Tensor,
           targets_xy: Tensor, episode_ids: Sequence[int],
           i2: I2Input | None = None, alpha: float = 1e-3,
           val_frac: float = 0.2, seed: int = 0,
           extra_metrics: Mapping[str, Callable] | None = None) -> GateReport:
    """D3: imagined-decode ADE@2s must be within 1.5x the oracle-decode ADE@2s.

    oracle-decode := probe_real fitted+evaluated on REAL future latents (the best
    the probe family can do). imagined-decode := probe_imag (A3-calibrated on
    imagined latents) on the model's imagined future latents. Ablation reports the
    A3 gap: probe_real-on-imag (mis-calibrated) vs probe_imag-on-imag.
    """
    tr, va = split_by_episode(episode_ids, val_frac, seed)
    tgt = _as_traj(targets_xy)
    flat = lambda x: x.flatten(1)

    probe_real, fit_r2 = _fit_probe(z_true_future[tr], flat(tgt[tr]), alpha)
    probe_imag, _ = _fit_probe(z_imag_future[tr], flat(tgt[tr]), alpha)

    def _ade(probe, feats):
        pred = probe.predict(feats[va]).reshape(len(va), -1, 2)
        return ade_fde(pred, tgt[va])[0]

    oracle_ade = _ade(probe_real, z_true_future)
    imag_ade = _ade(probe_imag, z_imag_future)
    miscal_ade = _ade(probe_real, z_imag_future)           # A3 ablation: wrong probe on imag
    ratio = imag_ade / max(oracle_ade, 1e-8)

    rows = [_i1_row(fit_r2), _i2_row(i2), _i3_row(episode_ids, tr, va),
            _i4_row(z_imag_future, z_true_future, z_prev, ceil=I4_FLOOR)]
    admissible, blockers = _admissible(rows)
    passed = admissible and ratio <= D3_RATIO_MAX

    ablation = {"probe_imag_on_imag_ade": imag_ade,
                "probe_real_on_imag_ade": miscal_ade,
                "a3_calibration_helps": bool(imag_ade <= miscal_ade),
                "note": "A3: probe calibrated on imagined latents should decode imag >= a real-fit probe"}
    metrics = {"oracle_decode_ade@2s": oracle_ade, "imagined_ade@2s": imag_ade,
               "ratio": ratio}
    if extra_metrics:
        va_pred = probe_imag.predict(z_imag_future[va]).reshape(len(va), -1, 2)
        metrics.update({k: fn(va_pred, tgt[va]) for k, fn in extra_metrics.items()})

    verdict = (f"BLOCKED: instruments failed {blockers}" if not admissible else
               f"{'PASS' if passed else 'FAIL'}: imagined/oracle ADE@2s ratio={ratio:.3f} "
               f"vs <={D3_RATIO_MAX}.")
    return GateReport("D3", "trajectory decode from imagination", admissible, passed,
                      rows, metrics, {"ratio_max": D3_RATIO_MAX}, ablation,
                      blockers, verdict)


# --------------------------------------------------------------------------- #
# Assembly — protocol §6 metrics.json (instruments FIRST)                      #
# --------------------------------------------------------------------------- #
def gates_metrics_json(exp_id: str, git_hash: str,
                       reports: Sequence[GateReport],
                       extra: Mapping | None = None) -> dict:
    """Assemble a protocol-§6 metrics dict. Instrument rows are emitted FIRST and
    the summary marks each gate PASS / FAIL / BLOCKED. A gate that is BLOCKED
    contributes no claim — that is the whole point of D-004."""
    return {
        "exp_id": exp_id,
        "git_hash": git_hash,
        "instruments": {r.gate: r.instruments for r in reports},   # FIRST
        "gates": [r.to_dict() for r in reports],
        "summary": {r.gate: r.status for r in reports},
        "doctrine": ("D1-D3 are decode/instrument gates: necessary, not sufficient "
                     "(arXiv 2512.24497). Closed-loop D4-D6 arbitrate the driving claim."),
        **(dict(extra) if extra else {}),
    }


def encode_states(world, frames: Tensor) -> Tensor:
    """Convenience: [B, C, H, W] frames -> compact readout states via a WorldModel."""
    with torch.no_grad():
        return world.encode(frames)
