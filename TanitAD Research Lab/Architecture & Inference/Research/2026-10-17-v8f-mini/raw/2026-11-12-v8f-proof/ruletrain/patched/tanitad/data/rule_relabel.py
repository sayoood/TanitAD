"""rule_relabel.py -- RULETRAIN (card PREREG_V8F_PROOF_ADD32): the label-fed distance rule applied to the TRAINING target of the executed plan.

``--rule-relabel on`` replaces, per training sample whose CURRENT frame has a label lead, the 30-knot target ``batch["gt"]`` (the logged future in the ego frame at the sample's frame; the one array the loss
reads for ``ade`` / ``fde`` / ``acc`` / ``launch_along``) by ``dist_rule.distance_rule(gt, v0, present, gap, gap, v_L, floor_lifts=False)`` (cap-only form of card ADD32 / note N1), with

  v0    = d["p10f"][t][0]                    RAW from the clip store (the batch's ``p10f[:, -1, 0]`` has NaN as 0)
  v_log = d["v"][t]                          the cache speed row of the current frame, RAW (the wrapper's ``logged_v[j]``; the batch's ``wm_speed`` has NaN as 0)
  A non-finite v0 or v_log excludes the row (it keeps its target and is not counted), as the wrapper's ``src_bad`` makes the tick inactive.
  gap   = label gap [m] at the current frame, closing = label closing speed [m/s]   (``lead_sup_label_arrays`` rows ``off[ci] + t``; the wrapper's ``LabelSource.read``)
  v_L   = max(0, v_log + closing)            g0 = gap (on a logged state a_l = a_m = 0, so the wrapper's g0 = g_src + a_l - a_m = g_src)

Samples without a label lead (no presence, gap or closing label NaN), without a GT future (``gt_ok`` False) or without a real current frame (``frame_ok[:, -1]`` False) keep their target bit for bit.
Nothing here is read at inference; the labels are privileged and used for the training target only. The module is numpy + torch only.
"""
from __future__ import annotations

import numpy as np
import torch

from tanitad.data import dist_rule as R
from tanitad.data import v8f_mini_data as D

#: the rule's ``m`` (executed steps): it only feeds the ``n_sat_exec`` log of ``distance_rule`` and does not change a plan
RULE_M = 10


class RuleRelabel:
    """The trainer's stage object. ``mode`` is ``off`` or ``on``; ``on`` computes the label arrays once from the join (``D.lead_sup_label_arrays``)."""

    def __init__(self, mode: str, J: dict | None = None, store=None):
        if mode not in ("off", "on"):
            raise ValueError(f"rule_relabel must be off or on, got {mode!r}")
        self.mode = mode
        self.on = mode == "on"
        if self.on and store is None:
            raise ValueError("rule_relabel on needs the clip store: the logged speeds are read RAW (the batch tensors carry NaN as 0)")
        self.store, self.J = store, J
        self.labels = D.lead_sup_label_arrays(J) if self.on else None

    def label_rows(self, ci, t):
        """labels of the LOGGED frame (clip index ``ci``, cache step ``t``) -> (pres, gap, clos) float32 arrays."""
        LS = self.labels
        rows = LS["off"][np.asarray(ci, np.int64)] + np.asarray(t, np.int64)
        return LS["pres"][rows], LS["gap"][rows], LS["clos"][rows]

    def raw_speeds(self, ci, t):
        """(v0, v_logged) RAW from the clip store: ``d["p10f"][t][0]`` and ``d["v"][t]`` (NaN stays NaN; the batch tensors have it as 0.0, which would read as a stopped ego)."""
        v0, vl = np.full(len(ci), np.nan), np.full(len(ci), np.nan)
        for b, (c, tt) in enumerate(zip(np.asarray(ci, np.int64), np.asarray(t, np.int64))):
            d = self.store.get(str(self.J["clip"][int(c)]))
            v0[b], vl[b] = float(d["p10f"][int(tt)][0]), float(d["v"][int(tt)])
        return v0, vl

    def stage(self, batch: dict) -> tuple:
        """COLLATED batch (CPU, logged) -> (gt_new [B, 30, 2] float32 tensor, info dict). Must run BEFORE ego.apply / dag.apply / rcv.apply (a swapped sample's ci / t is the own one)."""
        ci, t = batch["ci"].cpu().numpy(), batch["t"].cpu().numpy()
        pres, gap, clos = self.label_rows(ci, t)
        v0, vl = self.raw_speeds(ci, t)       # a row whose logged speed (or v0) is not finite keeps its target and is not counted (the rollout wrapper: src_bad -> inactive)
        return relabel_gt(batch["gt"], batch["gt_ok"], batch["frame_ok"][:, -1], v0, vl, pres, gap, clos)


def relabel_gt(gt, gt_ok, frame_ok_last, v0, v_logged, pres, gap, clos, mask_override=None) -> tuple:
    """The relabel on explicit inputs (tensors or arrays). Returns (gt_new float32 tensor, info). ``mask_override`` (bool [B]) AND-s an extra mask (control: an all-False mask must return ``gt`` itself)."""
    gt_t = torch.as_tensor(gt)
    g = gt_t.detach().cpu().numpy()
    f = lambda x: np.asarray(torch.as_tensor(x).detach().cpu().numpy() if not isinstance(x, np.ndarray) else x, np.float64)      # noqa: E731
    v0, v_logged, pres, gap, clos = f(v0), f(v_logged), f(pres), f(gap), f(clos)
    ok = np.asarray(torch.as_tensor(gt_ok).detach().cpu().numpy(), bool) & np.asarray(torch.as_tensor(frame_ok_last).detach().cpu().numpy(), bool)
    with np.errstate(invalid="ignore"):
        mask = ok & (pres == 1.0) & np.isfinite(gap) & np.isfinite(clos) & np.isfinite(v0) & np.isfinite(v_logged)
    if mask_override is not None:
        mask &= np.asarray(mask_override, bool)
    info = {"n_rows": int(len(g)), "n_relabel": int(mask.sum()), "n_changed": 0, "mask": mask}
    if not mask.any():
        return gt_t, info
    idx = np.nonzero(mask)[0]
    v_L = np.maximum(0.0, v_logged[idx] + clos[idx])
    out = R.distance_rule(g[idx], v0[idx], np.ones(len(idx), bool), gap[idx], gap[idx], v_L, m=RULE_M, floor_lifts=False)
    new = np.array(g, copy=True)
    new[idx] = out.plan
    info["n_changed"] = int(out.changed.sum())
    info["changed"] = np.zeros(len(g), bool)
    info["changed"][idx] = out.changed
    return torch.as_tensor(new, dtype=gt_t.dtype), info
