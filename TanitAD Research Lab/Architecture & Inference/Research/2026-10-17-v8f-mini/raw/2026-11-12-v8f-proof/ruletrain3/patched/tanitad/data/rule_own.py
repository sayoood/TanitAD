"""rule_own.py -- RULEOWN (card PREREG_V8F_PROOF_ADD34 section 2; stage 2 of note PREREG_V8F_PROOF_ADD32_N1): two trainer levers that sit next to ``rule_relabel.py``.

``--w-rule-own W`` (default 0.0 = off): for every training sample with a label lead (the mask and the inputs of ``--rule-relabel``) the model's OWN predicted 30-knot trajectory ``out["traj30"]`` is capped by
``dist_rule.distance_rule(plan, v0, present, gap, gap, v_L, m=10, floor_lifts=False)`` (plan = the prediction, DETACHED; no gradient through the rule). The per-sample term is the MEAN over the 30 knots of the squared
Euclidean distance (m^2) between the predicted knot and its cap knot; the batch term is the mean over the masked samples (exactly 0.0 when there are none); the loss gets ``W * term``.
A plan that already keeps the rule equals its cap bit for bit, so term and gradient are exactly 0 there. Nothing is read at inference.

``--rule-relabel-weight W`` (default 1.0, needs ``--rule-relabel on`` when != 1): the samples whose target the relabel CHANGED carry the weight W in the trajectory loss terms that read ``batch["gt"]``
(ade, fde, and acc / launch_along when those are on). Not renormalised: a weight of 4 means those samples count four times (the denominator stays the GT-valid count). Done as an EXTRA loss
``(W - 1) * (the same terms restricted to the changed samples)`` added in the trainer, so the model file and the unweighted terms are untouched.

The module is numpy + torch only. The model code (``v8f_mini.py``) is not changed by either lever.
"""
from __future__ import annotations

import numpy as np
import torch

from tanitad.data import dist_rule as R
from tanitad.data import rule_relabel as RRL
from tanitad.models import v8f_mini as M

#: the rule's ``m`` (executed steps): it only feeds the ``n_sat_exec`` log of ``distance_rule`` and does not change a plan (same as rule_relabel.RULE_M)
RULE_M = RRL.RULE_M

#: the forms of the own-plan term (``--rule-own-form``): ``sq`` = mean over the knots of the SQUARED distance (m^2, the card's form), ``l1`` = mean over the knots of the distance (m)
FORMS = ("sq", "l1")


class RuleOwn:
    """The trainer's stage object for both levers. ``w_own`` = --w-rule-own, ``w_weight`` = --rule-relabel-weight. Off (0.0 and 1.0) builds nothing.
    The label arrays and the raw speeds come from a ``RuleRelabel`` object (``share``: the trainer's own when ``--rule-relabel on``, else one built here from the join)."""

    def __init__(self, w_own: float, w_weight: float, J: dict | None = None, store=None, share: "RRL.RuleRelabel | None" = None, form: str = "sq"):
        self.w_own, self.w_weight = float(w_own), float(w_weight)
        if form not in FORMS:
            raise ValueError(f"rule_own_form must be one of {FORMS}, got {form!r}")
        self.form = form
        if form != "sq" and not self.w_own > 0.0:
            raise ValueError(f"rule_own_form {form} needs w_rule_own > 0 (got {w_own!r})")
        if not np.isfinite(self.w_own) or self.w_own < 0.0:
            raise ValueError(f"w_rule_own must be finite and >= 0, got {w_own!r}")
        if not np.isfinite(self.w_weight) or self.w_weight < 0.0:
            raise ValueError(f"rule_relabel_weight must be finite and >= 0, got {w_weight!r}")
        self.own_on = self.w_own > 0.0
        self.weight_on = self.w_weight != 1.0
        self.src = None
        if self.own_on:
            self.src = share if (share is not None and share.on) else RRL.RuleRelabel("on", J, store)

    # ---- flag A: the term
    def stage(self, batch: dict) -> dict:
        """COLLATED batch (CPU, logged) -> numpy inputs of the term. Must run BEFORE ego.apply / dag.apply / rcv.apply (as RuleRelabel.stage; the combinations are refused anyway)."""
        ci, t = batch["ci"].cpu().numpy(), batch["t"].cpu().numpy()
        pres, gap, clos = self.src.label_rows(ci, t)
        v0, vl = self.src.raw_speeds(ci, t)
        return own_inputs(batch["gt_ok"], batch["frame_ok"][:, -1], v0, vl, pres, gap, clos)

    def built_levers(self) -> dict:
        return {"w_rule_own": self.w_own, "rule_relabel_weight": self.w_weight}


def own_inputs(gt_ok, frame_ok_last, v0, v_logged, pres, gap, clos) -> dict:
    """The mask and the rule inputs, the SAME as ``rule_relabel.relabel_gt``: label presence 1, finite gap and closing label, GT-valid, real current frame, finite raw v0 and logged speed.
    v_L = max(0, v_logged + closing); g0 = gap."""
    f = lambda x: np.asarray(torch.as_tensor(x).detach().cpu().numpy() if not isinstance(x, np.ndarray) else x, np.float64)      # noqa: E731
    v0, v_logged, pres, gap, clos = f(v0), f(v_logged), f(pres), f(gap), f(clos)
    ok = np.asarray(torch.as_tensor(gt_ok).detach().cpu().numpy(), bool) & np.asarray(torch.as_tensor(frame_ok_last).detach().cpu().numpy(), bool)
    with np.errstate(invalid="ignore"):
        mask = ok & (pres == 1.0) & np.isfinite(gap) & np.isfinite(clos) & np.isfinite(v0) & np.isfinite(v_logged)
        v_L = np.maximum(0.0, v_logged + clos)
    return {"mask": mask, "v0": v0, "gap": gap, "v_L": v_L, "n_rows": int(len(mask))}


def cap_of_plan(plan_np: np.ndarray, v0, gap, v_L) -> np.ndarray:
    """The cap of a batch of plans [n, 30, 2] (any float in, float64 out, the rule called on float64 plan and inputs): ``distance_rule(..., m=10, floor_lifts=False).plan``. Inactive / unchanged windows come back bit for bit."""
    n = len(plan_np)
    f64 = lambda x: np.asarray(x, np.float64)      # noqa: E731
    out = R.distance_rule(f64(plan_np), f64(v0), np.ones(n, bool), f64(gap), f64(gap), f64(v_L), m=RULE_M, floor_lifts=False)      # the rule runs in float64 (idempotent to 1e-5); a plan that keeps the rule comes back equal
    return np.asarray(out.plan, np.float64)


def knot_distance(diff: torch.Tensor) -> torch.Tensor:
    """Euclidean norm over the last axis, in the units of ``diff``, with an exact zero value AND an exact zero gradient where the norm is 0 (a plain sqrt has an infinite slope there and gives NaN):
    the sqrt is taken on a safe input of 1.0 at the knots with a zero squared norm and the result is selected away from them. No epsilon: a knot that is not on its cap keeps the exact norm."""
    sq = (diff ** 2).sum(-1)
    pos = sq > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, sq, torch.ones_like(sq))), torch.zeros_like(sq))


def own_term(traj30: torch.Tensor, inp: dict, form: str = "sq") -> tuple:
    """(term, info): ``term`` = mean over the masked samples of (mean over the 30 knots of the squared Euclidean distance between ``traj30`` and its cap). The cap is computed from the DETACHED plan.
    No masked sample -> a plain zero tensor with no graph (exactly 0.0). info: n_rows, n_lead (masked samples), n_active (samples whose per-sample term is above 0)."""
    traj30 = traj30.float()
    mask = inp["mask"]
    n = int(mask.sum())
    info = {"n_rows": int(inp["n_rows"]), "n_lead": n, "n_active": 0}
    if n == 0:
        return traj30.new_zeros(()), info
    sel = np.nonzero(mask)[0]
    idx = torch.as_tensor(sel, device=traj30.device)
    pl = traj30.index_select(0, idx)
    plan = pl.detach().double().cpu().numpy()
    cap = torch.as_tensor(cap_of_plan(plan, inp["v0"][sel], inp["gap"][sel], inp["v_L"][sel]), dtype=pl.dtype, device=pl.device)
    if form == "l1":
        per = knot_distance(pl - cap).mean(1)                     # [n] mean over the knots of the distance (m); exact 0 value and gradient at a knot on its cap
    elif form == "sq":
        per = ((pl - cap) ** 2).sum(-1).mean(1)                   # [n] mean over the knots of the squared distance (m^2)
    else:
        raise ValueError(f"rule_own_form must be one of {FORMS}, got {form!r}")
    info["n_active"] = int((per.detach() > 0).sum())
    return per.mean(), info


# ---- flag B: the weight on the samples the relabel changed
def weight_extra(traj30: torch.Tensor, gt: torch.Tensor, gt_ok, changed, W: float, w_traj: float, w_acc: float = 0.0, launch=None, launch_w: float = 0.0) -> torch.Tensor:
    """The EXTRA loss of weight W on the CHANGED samples: ``(W - 1) x`` the ade / fde terms of ``v8f_mini.traj_terms`` (and ``accel_term`` when ``w_acc > 0``, ``launch_along_term`` when ``launch_w > 0``) restricted to those
    samples, with the SAME denominators (the GT-valid count; not renormalised) and the SAME total weights as ``V8FMini.loss`` (``w_traj * (ade / 3 + 0.5 * fde / 6)``, ``w_acc * acc``, ``w_traj * launch_w * launch_along / 3``).
    ``gt`` is the target the loss receives (the relabelled one); ``changed`` [B] bool. total_with_weight = total + this."""
    traj30 = traj30.float()
    gok = torch.as_tensor(gt_ok).to(traj30.device).float()
    u = (float(W) - 1.0) * torch.as_tensor(changed).to(traj30.device).float()
    den = gok.sum().clamp_min(1.0)
    gt = gt.to(traj30.device).float()
    d = torch.sqrt(((traj30 - gt) ** 2).sum(-1) + 1e-9)
    ade_x = (d.mean(1) * gok * u).sum() / den
    fde_x = (d[:, -1] * gok * u).sum() / den
    extra = w_traj * (ade_x / 3.0 + 0.5 * fde_x / 6.0)
    if w_acc > 0.0:
        d2 = M.second_diff_02(traj30) - M.second_diff_02(gt)
        extra = extra + w_acc * (((d2 ** 2).sum(-1).mean(1)) * gok * u).sum() / den
    if launch_w > 0.0 and launch is not None:
        lo, hi = M.LAUNCH_KNOT_LO - 1, M.LAUNCH_KNOT_HI
        dx = traj30[:, lo:hi, 0] - gt[:, lo:hi, 0]
        per = torch.sqrt(dx ** 2 + 1e-9).sum(1) / float(traj30.shape[1])
        extra = extra + w_traj * launch_w * ((per * gok * torch.as_tensor(launch).to(traj30.device).float() * u).sum() / den) / 3.0
    return extra


def changed_vector(rr_info: dict, n_rows: int) -> np.ndarray:
    """[B] bool: the samples whose target the relabel changed (``RuleRelabel.stage`` info; the key is absent when no sample was relabelled)."""
    c = rr_info.get("changed")
    return np.zeros(n_rows, bool) if c is None else np.asarray(c, bool)


def add_own_term(T: dict, traj30: torch.Tensor, inp: dict, W: float, form: str = "sq") -> dict:
    """THE line that applies flag A: ``T["total"] = T["total"] + W * term`` (the trainer loop and the G-LIVE probe both call this; a deleted add line is caught by the loop guard ZZRULE-OWN-NOT-APPLIEDZZ).
    No new key is put into ``T`` (the train log keys of a flag-on arm come from the returned info). Returns info: n_rows, n_lead, n_active, term (float)."""
    term, info = own_term(traj30, inp, form)
    T["total"] = T["total"] + W * term
    info["term"] = float(term.detach())
    return info


def add_weight_extra(T: dict, traj30: torch.Tensor, gt: torch.Tensor, gt_ok, changed, W: float, w_traj: float, w_acc: float = 0.0, launch=None, launch_w: float = 0.0) -> None:
    """THE line that applies flag B: ``T["total"] = T["total"] + weight_extra(...)`` (a deleted line is caught by the loop guard ZZRULE-WEIGHT-NOT-APPLIEDZZ)."""
    T["total"] = T["total"] + weight_extra(traj30, gt, gt_ok, changed, W, w_traj, w_acc, launch, launch_w)
