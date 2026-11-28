#!/usr/bin/env python3
"""``E-DDA-2b`` — the runnable driver for the coarse-to-fine sub-metric selector.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
``tanitad.refs.refc_selector`` (the model), ``refc_selector_targets`` (the
rule-based targets) and ``refc_selector_aug`` (the fan augmentation and the
foreign bank) are three libraries with no caller. This script is the caller:
it runs the whole stage-II path end to end, on CPU, with the controls the
programme requires, and writes a JSON a panel can read.

⛔ **IT MAKES NO CAPABILITY CLAIM AND CANNOT.** On ``--source synthetic`` the
fan is an analytic rig built here; the only admissible readings from it are
(a) the instrument runs, (b) the **controls read their known values exactly**,
and (c) the arms are separable. A driving claim needs a banked refcv4 fan
(``--source dump``), the ``obstacle.offline`` replay join, T1 and the four
families — none of which this script invents.

THE FOUR ARMS, AND WHY EACH ONE EXISTS
--------------------------------------
``--arm selector``      the hypothesis: the learned coarse-to-fine selector.
``--arm const``         ⛔ THE NO-INFORMATION CONTROL. Every candidate gets the
                        SAME score. Its ``rank_auc`` must be **exactly 0.5000**
                        and its tie-group ``AP`` must be **exactly the base
                        rate** — both analytic, both checked as literals. An
                        arm that cannot make its control read a known value has
                        no scale on which to read the hypothesis.
``--arm raw_floor``     ⛔ THE RAW-INPUT FLOOR. A ridge probe on the RAW
                        candidate coordinates (``d = 2 * n_steps``). *A learned
                        representation that does not beat raw input has added
                        nothing* — the 2026-08-22 probe-panel rule, which caught
                        a latent scoring identically to raw pixels.
``--arm progress_only`` ⛔ THE DELIBERATE REGRESSION, as pre-registered in
                        ``H-DDA-7``. Only the progress head is supervised and
                        only progress enters the composition. It must NOT
                        improve the contact readout; if it does, the contact
                        readout is blind and the panel is VOID.

⛔ TIES ARE THE MEASUREMENT, NOT AN EDGE CASE
---------------------------------------------
:func:`average_precision` sums by TIE GROUP. A naive per-sample AP scores a
CONSTANT arm **above** its own base rate (MEASURED elsewhere in this programme:
+22.8 % on one rig, +0.881 % on another) and has already forced a retraction.
With tie-group summation a constant scorer lands on the base rate exactly, which
is what makes the control readable at all. Same for :func:`rank_auc`, where a
tie contributes 0.5.

⛔ EVERY KNOB IS DERIVED FROM ARGPARSE, NEVER FROM A HAND LIST
--------------------------------------------------------------
``--wp-index`` shipped with **3 of 6 knobs parsed, stamped and inert**, and it
was caught only because a test derived its knob list from the parser. So:
:func:`term_specs` walks ``_knob_group(parser)._group_actions`` and emits one
row per action. Adding a flag to that group without wiring it produces a row
whose ``needs`` is unmet, and ``stack/tests/test_refc_selector_e2b.py`` asserts
the parser's dest set and the stamp's dest set are EQUAL — so a new flag cannot
be forgotten in either direction.

Evidence class: the DiffusionDriveV2 mechanisms are ``PUBLISHED-CODE``
(``hustvl/DiffusionDriveV2@1cd12a1``, banked in-repo and sha256-verified); every
number this script prints is ``MEASURED`` (ours, this run) and carries its ``n``
and its ``d``.

Usage (dev box; the mirror recipe, PYTHONIOENCODING is required)::

    cd C:/Users/Admin/tanitad-wt/stack
    PYTHONPATH="C:/Users/Admin/tanitad-wt/stack;C:/Users/Admin/tanitad-wt/taniteval" \\
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 python scripts/refc_selector_e2b.py \\
        --arm selector --steps 150 --out /tmp/e2b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import torch
from torch import Tensor

try:                                    # pragma: no cover - console encoding
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                       # pragma: no cover
    pass

from tanitad import effective_weights as EW
from tanitad.refs import refc_selector as S
from tanitad.refs import refc_selector_aug as AUG
from tanitad.refs import refc_selector_targets as T

TAG = "e2b"
KNOB_GROUP = "E-DDA-2b knobs"

#: ⛔ The analytic no-information value of :func:`rank_auc`. A scorer that
#: cannot order anything ranks a better candidate above a worse one exactly half
#: the time. Written as a LITERAL so a change to the metric cannot quietly move
#: the bar it is read against.
RANK_AUC_NO_INFORMATION: float = 0.5


# ===========================================================================
# metrics — ties handled by GROUP, and the no-information value is analytic
# ===========================================================================
def rank_auc(score: Tensor, quality: Tensor) -> tuple[float, int]:
    """P(score orders a better candidate above a worse one), ties = 0.5.

    ``score`` and ``quality`` are ``[B, N]``. Pairs are formed WITHIN a window.
    Returns ``(auc, n_pairs)``.

    ⭐ A CONSTANT ``score`` makes every pair a tie and therefore reads
    **exactly** :data:`RANK_AUC_NO_INFORMATION`. That is the property the
    ``const`` arm is for, and it is analytic — it does not depend on the data.
    """
    if score.shape != quality.shape or score.dim() != 2:
        raise ValueError(f"score {tuple(score.shape)} and quality "
                         f"{tuple(quality.shape)} must both be [B, N]")
    s_i, s_j = score.unsqueeze(2), score.unsqueeze(1)          # [B, N, N]
    q_i, q_j = quality.unsqueeze(2), quality.unsqueeze(1)
    valid = q_i > q_j
    n = int(valid.sum())
    if n == 0:
        return float("nan"), 0
    win = (s_i > s_j).to(score.dtype) + 0.5 * (s_i == s_j).to(score.dtype)
    return float((win * valid.to(score.dtype)).sum() / n), n


def average_precision(score: Tensor, label: Tensor) -> tuple[float, int, float]:
    """AP with **tie-group summation**. ``[N]`` score, ``[N]`` binary label.

    Returns ``(ap, n, base_rate)``.

    ⛔ THE TIE RULE IS THE WHOLE POINT. Candidates sharing a score share a rank
    position; the precision is evaluated ONCE at the end of the group and
    credited to every positive in it. Sorting ties by array index instead lets a
    CONSTANT scorer beat its own base rate — MEASURED in this programme at
    +22.8 % on one rig and +0.881 % on another, and it forced a retraction.

    ⭐ Analytic consequence, which is why the ``const`` arm can be read: with one
    tie group of size N holding P positives, precision at the group's end is
    ``P/N`` and ``AP = (P/N)*P / P = P/N`` — **exactly the base rate**.
    """
    if score.dim() != 1 or label.shape != score.shape:
        raise ValueError(f"score {tuple(score.shape)} and label "
                         f"{tuple(label.shape)} must both be [N]")
    n = int(score.numel())
    y = (label > 0.5).to(torch.float64)
    pos = float(y.sum())
    base = pos / n if n else float("nan")
    if pos == 0 or n == 0:
        return float("nan"), n, base

    order = torch.argsort(score.to(torch.float64), descending=True, stable=True)
    s_sorted = score.to(torch.float64)[order]
    y_sorted = y[order]

    ap, cum_tp, cum_n, i = 0.0, 0.0, 0, 0
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        tp_g = float(y_sorted[i:j].sum())
        cum_tp += tp_g
        cum_n += (j - i)
        ap += (cum_tp / cum_n) * tp_g            # one precision per TIE GROUP
        i = j
    return ap / pos, n, base


def _ap_naive(score: Tensor, label: Tensor) -> float:
    """⛔ THE WRONG ONE, kept so the test can show it is wrong. Breaks ties by
    array order — the form that scores a constant arm above its base rate."""
    n = int(score.numel())
    y = (label > 0.5).to(torch.float64)
    pos = float(y.sum())
    if pos == 0:
        return float("nan")
    order = torch.argsort(score.to(torch.float64), descending=True, stable=True)
    y_s = y[order]
    cum = torch.cumsum(y_s, 0)
    prec = cum / torch.arange(1, n + 1, dtype=torch.float64)
    return float((prec * y_s).sum() / pos)


# ===========================================================================
# the analytic rig
# ===========================================================================
@dataclass(frozen=True)
class RigSpec:
    """A fan of constant-curvature, constant-acceleration arcs.

    ⭐ ANALYTIC BY CONSTRUCTION: candidate ``(k, a)`` is an arc of curvature
    exactly ``k`` at initial speed ``v0`` with acceleration exactly ``a``, so
    the curvature the target code recovers has a reference value that the target
    code never computed. ``--self-check`` reads exactly that.

    ⛔ ``n_curv`` MUST BE ODD so the grid contains ``kappa = 0`` EXACTLY. That arm
    is the only assertion in the self-check with **no discretisation error**, and
    it is therefore the one that can be written as a literal.

    ⛔ THE ACCELERATION GRID MAY NOT LET A CANDIDATE REVERSE. MEASURED
    2026-09-10, and it is the reason this constraint is enforced rather than
    documented: with ``a = -2.0``, ``v0 = 10``, ``T = 6 s`` the speed crosses
    zero at ``t = 5 s``, the path turns around, ``ds -> 0``, and the recovered
    curvature reads **-1.2862 against a built -0.0600** — a 21x error on
    **16 of 32** candidates. It looked exactly like a broken curvature
    instrument; the instrument was fine and the FAN was unphysical.
    """
    n_windows: int = 24
    n_curv: int = 9                      # ODD -> the grid contains kappa = 0
    n_acc: int = 4
    n_steps: int = 12
    dt: float = 0.5
    v0: float = 10.0
    seed: int = 0
    kappa_max: float = 0.04
    #: the slowest a candidate may end at, as a fraction of ``v0``. Sets the
    #: most negative admissible acceleration analytically:
    #: ``a_min = (frac - 1) * v0 / T``.
    end_speed_frac: float = 0.4
    accel_max: float = 2.0

    @property
    def horizon_s(self) -> float:
        return float(self.n_steps * self.dt)

    @property
    def accel_min(self) -> float:
        return (self.end_speed_frac - 1.0) * self.v0 / self.horizon_s

    @property
    def n_cand(self) -> int:
        return self.n_curv * self.n_acc


def build_arcs(spec: RigSpec) -> tuple[Tensor, Tensor, Tensor]:
    """``(cand [B, N, S, 2], kappa [N], accel [N])`` — arcs with known curvature."""
    if spec.n_curv % 2 == 0:
        raise ValueError(
            f"n_curv must be ODD so the grid contains kappa = 0 exactly, got "
            f"{spec.n_curv}. The zero arm is the only self-check assertion "
            "with no discretisation error.")
    kap = torch.linspace(-spec.kappa_max, spec.kappa_max, spec.n_curv)
    acc = torch.linspace(spec.accel_min, spec.accel_max, spec.n_acc)
    kk = kap.repeat_interleave(spec.n_acc)                    # [N]
    aa = acc.repeat(spec.n_curv)                              # [N]
    t = torch.arange(1, spec.n_steps + 1, dtype=torch.float32) * spec.dt
    v_end = spec.v0 + acc * spec.horizon_s
    if float(v_end.min()) <= 0.0:
        raise ValueError(
            f"accel grid {acc.tolist()} lets a candidate reach speed "
            f"{float(v_end.min()):.3f} m/s within {spec.horizon_s} s — the path "
            "REVERSES and its recovered curvature is meaningless (MEASURED: "
            "-1.2862 against a built -0.0600 on 16/32 candidates). Raise "
            "end_speed_frac.")
    s = spec.v0 * t.unsqueeze(0) + 0.5 * aa.unsqueeze(1) * t.unsqueeze(0) ** 2
    th = kk.unsqueeze(1) * s                                  # heading = kappa * s
    # exact arc integration; the k -> 0 limit is the straight line
    small = kk.abs() < 1e-9
    kk_safe = torch.where(small, torch.ones_like(kk), kk).unsqueeze(1)
    x = torch.where(small.unsqueeze(1), s, torch.sin(th) / kk_safe)
    y = torch.where(small.unsqueeze(1), torch.zeros_like(s),
                    (1.0 - torch.cos(th)) / kk_safe)
    cand = torch.stack([x, y], dim=-1).unsqueeze(0)           # [1, N, S, 2]
    return cand.repeat(spec.n_windows, 1, 1, 1), kk, aa


def build_rig(spec: RigSpec) -> dict:
    """The fan, the scene, and the per-window facts the TARGETS read."""
    g = torch.Generator().manual_seed(spec.seed)
    cand, kap, acc = build_arcs(spec)
    b, n, s, _ = cand.shape

    # a lead vehicle straight ahead at a per-window gap and speed
    gap = 8.0 + 24.0 * torch.rand((b, 1), generator=g)
    v_lead = 4.0 + 10.0 * torch.rand((b, 1), generator=g)
    t = torch.arange(1, s + 1, dtype=torch.float32) * spec.dt
    lead_x = gap + v_lead * t.unsqueeze(0)
    lead = torch.stack([lead_x, torch.zeros_like(lead_x)], dim=-1)   # [B, S, 2]

    v0 = torch.full((b,), spec.v0)
    ctx = {"dt": spec.dt, "v0": v0, "lead_path": lead}

    # ⚠️ THE SCENE THE SELECTOR SEES. It is the SAME environment the targets
    # read (a lead track and the ego speed), which is legitimate — the real
    # model sees the scene — but on this RIG it makes the task easy, so no
    # capability claim may be read off it. Stamped in the output as `rig`.
    scene = torch.cat([
        (lead / 40.0).reshape(b, s * 2),
        (gap / 40.0), (v_lead / 20.0), (v0.reshape(b, 1) / 20.0),
    ], dim=1)                                                  # [B, scene_dim]
    return {"cand": cand, "ctx": ctx, "scene": scene.unsqueeze(1),
            "kappa": kap, "accel": acc, "spec": spec}


def rig_self_check(rig: dict) -> dict:
    """⭐ THE ANALYTIC CROSS-CHECK: does the target code recover the curvature we
    BUILT? The reference ``1/R = kappa`` was written down before the code ran.

    Three statements, in decreasing strength — and only the first is EXACT:

    1. ``straight_arm_kappa_max`` — the ``kappa = 0`` arm. A chord estimator on a
       straight line is exactly 0 whatever the step size, so this one has **no
       discretisation error** and a literal belongs on it. It is also what a
       sign flip or an axis swap breaks first (the ``WP-A`` mirrored-azimuth
       class).
    2. ``sign_agreement`` / ``rank_agreement`` — exact and discretisation-free:
       the recovered curvature must carry the built sign, and must be monotone
       in the built value.
    3. ``kappa_max_abs_err`` — ⚠️ NOT exact. ``rewards.kinematics`` is a chord
       estimator over ``dt`` steps, and at ``kappa * ds ~ 0.2 rad`` a few percent
       is the geometry, not a bug. Quoted with ``kappa_ds_rad`` so a reader can
       tell the two apart instead of reading the residual as an error.
    """
    from tanitad.rl import rewards as R
    spec: RigSpec = rig["spec"]
    kin = R.kinematics(rig["cand"], spec.dt)
    got = kin.kappa.mean(dim=-1)[0]                            # [N]
    want = rig["kappa"]
    err = (got - want).abs()
    zero = want.abs() < 1e-9
    nz = ~zero
    ds = float(kin.arc_len[0].max()) / max(spec.n_steps, 1)
    # ⚠️ MONOTONICITY IS ACROSS THE CURVATURE GRID, NOT ACROSS THE FLAT
    # CANDIDATE INDEX. The fan cycles the ACCELERATION inside each curvature
    # block, so consecutive flat entries share a built kappa and differ only by
    # their step length — comparing them read `False` and looked like a broken
    # instrument (MEASURED 2026-09-10). Compare the per-kappa means.
    uniq = torch.unique(want)
    blocks = torch.stack([got[want == k].mean() for k in uniq])
    order_ok = bool((blocks[1:] - blocks[:-1]).gt(0.0).all()) \
        if blocks.numel() > 1 else True
    return {
        "kappa_max_abs_err": float(err.max()),
        "kappa_mean_abs_err": float(err.mean()),
        "kappa_rel_err_max": float((err[nz] / want[nz].abs()).max()),
        "straight_arm_present": bool(zero.any()),
        "straight_arm_kappa_max": (float(got[zero].abs().max())
                                   if bool(zero.any()) else float("nan")),
        "sign_agreement": float((torch.sign(got[nz]) ==
                                 torch.sign(want[nz])).float().mean()),
        "rank_agreement_monotone": order_ok,
        "kappa_ds_rad": float(want.abs().max()) * ds,
        "min_end_speed_mps": float(kin.speed[0].min()),
        "n_candidates": int(want.numel()),
        "_note": ("statements 1 and 2 are exact; kappa_max_abs_err carries the "
                  "chord discretisation at kappa_ds_rad and is not a defect"),
    }


# ===========================================================================
# arms
# ===========================================================================
def _quality_and_labels(cand: Tensor, ctx: dict, tcfg: T.TargetConfig
                        ) -> tuple[Tensor, Tensor, dict]:
    """The rule-based composite the selector is asked to rank by, and the binary
    no-contact label AP is read on. Both come from ``refc_selector_targets`` —
    environment predicates only; the ego's logged future is structurally absent.
    """
    out = T.selector_targets(cand, ctx, tcfg)
    tg = out["targets"]
    logits = {}
    for h in S.HEAD_NAMES:
        v = tg.get(h)
        if v is None:
            v = torch.full(cand.shape[:2], 0.5, device=cand.device)
        logits[h] = torch.logit(v.clamp(1e-4, 1 - 1e-4))
    q = S.compose_score(logits)
    return q, tg["nc"], out


def _fit_ridge(x: Tensor, y: Tensor, lam: float) -> Tensor:
    xtx = x.T @ x + lam * torch.eye(x.shape[1], dtype=x.dtype)
    return torch.linalg.solve(xtx, x.T @ y)


def run_raw_floor(cand: Tensor, quality: Tensor, fit: Tensor, sco: Tensor,
                  lambdas: tuple[float, ...]) -> tuple[Tensor, dict]:
    """⛔ THE RAW-INPUT FLOOR: ridge on the RAW candidate coordinates.

    ⚠️ ``lam`` is chosen on a slice of the FIT split ONLY. Selecting it on the
    scored split is the 2026-08-22 failure that picked ``lam=1e6``, collapsed the
    ridge to the constant predictor, and made the latent AND the control both
    read ``+0.0000`` — an underpowered panel that looked like an absence.
    """
    b, n, s, _ = cand.shape
    x_all = cand.reshape(b, n, s * 2).to(torch.float64)
    x_all = torch.cat([x_all, torch.ones(b, n, 1, dtype=torch.float64)], dim=-1)
    y_all = quality.to(torch.float64)
    d = int(x_all.shape[-1])

    n_in = max(int(0.75 * fit.numel()), 1)
    inner_fit, inner_val = fit[:n_in], fit[n_in:]
    if inner_val.numel() == 0:
        inner_val = inner_fit
    xf = x_all[inner_fit].reshape(-1, d)
    yf = y_all[inner_fit].reshape(-1)
    best, best_lam = float("inf"), lambdas[0]
    for lam in lambdas:
        w = _fit_ridge(xf, yf, lam)
        pred = x_all[inner_val] @ w
        mse = float(((pred - y_all[inner_val]) ** 2).mean())
        if mse < best:
            best, best_lam = mse, lam
    w = _fit_ridge(x_all[fit].reshape(-1, d), y_all[fit].reshape(-1), best_lam)
    return (x_all[sco] @ w).to(cand.dtype), {
        "lambda": best_lam, "inner_val_mse": best, "d": d,
        "n_fit_rows": int(inner_fit.numel() * n),
        "lambda_grid": list(lambdas),
        "_note": "lambda selected on a slice of the FIT split only",
    }


def run_selector(rig: dict, args, tcfg: T.TargetConfig, acfg: AUG.AugmentConfig,
                 fit: Tensor, sco: Tensor, progress_only: bool) -> dict:
    torch.manual_seed(args.seed)
    cand, ctx, scene = rig["cand"], rig["ctx"], rig["scene"]
    spec: RigSpec = rig["spec"]
    scfg = S.SelectorConfig(
        d=args.sel_d, n_heads=args.sel_heads, coarse_layers=args.coarse_layers,
        fine_layers=args.fine_layers, top_k=args.top_k,
        self_attn=(args.self_attn == "on"), margin=args.margin,
        margin_pairs=args.margin_pairs, w_compliance=args.w_compliance,
        w_tactical=args.w_tactical)
    model = S.SubMetricSelector(scene_dim=int(scene.shape[-1]),
                                n_steps=spec.n_steps, cfg=scfg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    bank = None
    if acfg.foreign_frac > 0.0:
        # ⭐ OUR "FOREIGN VOCABULARY": arcs the arm's own fan does not contain.
        # V2 mixes 1 % of the GTRS 16,384 bank; our analogue is a bank the
        # generator did not emit — here a denser, wider arc grid.
        fspec = RigSpec(n_windows=1, n_curv=args.foreign_bank_curv,
                        n_acc=args.foreign_bank_acc, n_steps=spec.n_steps,
                        dt=spec.dt, v0=spec.v0, seed=spec.seed + 991,
                        kappa_max=args.foreign_bank_kappa_max)
        bank = build_arcs(fspec)[0][0]                        # [G, S, 2]

    gen = torch.Generator().manual_seed(args.seed + 1)
    hist = []
    for step in range(int(args.steps)):
        idx = fit[torch.randint(fit.numel(), (min(args.batch, fit.numel()),),
                                generator=gen)]
        c = cand[idx]
        a = AUG.augment_fan(c, acfg, bank=bank, training=True, generator=gen)
        cb = a["cand"]
        sub = {"dt": spec.dt, "v0": ctx["v0"][idx],
               "lead_path": ctx["lead_path"][idx]}
        tg = T.selector_targets(cb, sub, tcfg)
        targets, masks = dict(tg["targets"]), dict(tg["masks"])
        if progress_only:
            targets = {"progress": targets["progress"]}
            masks = {"progress": masks["progress"]}
        out = model(cb, scene[idx])
        res = S.selector_losses(out, targets, masks, cfg=scfg, generator=gen)
        opt.zero_grad(); res["loss"].backward(); opt.step()
        if step % max(int(args.steps) // 10, 1) == 0:
            hist.append({"step": step, "loss": float(res["loss"].detach()),
                         "n_candidates": res["n_candidates"],
                         "n_scored": res["n_scored"]})

    model.eval()
    with torch.no_grad():
        if args.eval_aug:
            # ⚠️ V2 DOES augment at test (`_model_sel.py:1444`, n_aug=3,
            # U(0.1, 0.3)) -- but it mixes NO vocabulary there. `training=False`
            # reproduces both halves. ⛔ The scored fan then differs from the
            # other arms', so an --eval-aug arm is comparable only to another
            # --eval-aug arm; the run record carries the flag.
            ae = AUG.augment_fan(cand[sco], acfg, bank=bank, training=False,
                                 generator=gen)
            out = model(ae["cand"], scene[sco])
            # ⛔ THE GUARD, ACTUALLY CALLED. It is reachable exactly when an
            # operator turns off the released train-only default, which is the
            # only way a foreign candidate can reach a pick.
            AUG.assert_pick_emittable(out["sel_idx_selector"], ae["origin"],
                                      where="refc_selector_e2b eval pick")
            eval_origin = AUG.origin_summary(ae["origin"])
            eval_cand = ae["cand"]
        else:
            out = model(cand[sco], scene[sco])
            eval_cand = cand[sco]
            eval_origin = AUG.origin_summary(
                torch.zeros(cand[sco].shape[:2], dtype=torch.long))
    return {"score": out["score"], "hist": hist, "eval_origin": eval_origin,
            "eval_cand": eval_cand,
            "n_params": sum(p.numel() for p in model.parameters()),
            "selector_cfg": scfg.as_dict(),
            "augment_cfg": acfg.as_dict(),
            "foreign_bank_size": (0 if bank is None else int(bank.shape[0]))}


# ===========================================================================
# the knobs — ONE source of truth, and the parser is built from it
# ===========================================================================
def _knob_group(parser: argparse.ArgumentParser):
    for g in parser._action_groups:
        if g.title == KNOB_GROUP:
            return g
    raise RuntimeError(f"parser has no {KNOB_GROUP!r} group")


def knob_dests(parser: argparse.ArgumentParser) -> list[str]:
    """⭐ THE KNOB LIST, DERIVED FROM ARGPARSE. Never a hand-written tuple."""
    return [a.dest for a in _knob_group(parser)._group_actions]


def _encode(v) -> float:
    """A knob's numeric encoding for the effective-weight table.

    numbers -> themselves; bools -> 1/0; strings -> 1.0 when the knob names a
    non-default mode. Documented because an encoded value in a table someone
    reads as a weight is the scope error this programme keeps paying for; the
    LITERAL value of every knob is carried beside the table in ``values``.
    """
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return 1.0 if v not in (None, "", "none", "off") else 0.0


#: Knobs that only mean anything when a selector is built. Named here so the
#: gate is one list rather than a condition repeated per branch.
_SELECTOR_ONLY: frozenset[str] = frozenset({
    "sel_d", "sel_heads", "coarse_layers", "fine_layers", "self_attn",
    "top_k", "margin", "margin_pairs", "w_compliance", "w_tactical",
    "lr", "steps", "batch",
})

#: Knobs that only mean anything when the fan is augmented.
_AUGMENT_ONLY: frozenset[str] = frozenset({
    "n_aug", "std_min", "std_max", "std_per_row", "foreign_frac",
    "foreign_at_eval", "foreign_bank_curv", "foreign_bank_acc",
    "foreign_bank_kappa_max", "eval_aug",
})


def term_specs(parser: argparse.ArgumentParser, args,
               acfg: AUG.AugmentConfig) -> list[EW.TermSpec]:
    """One row per argparse knob, with the STRUCTURAL preconditions attached.

    ⛔ ``needs`` is where an inert knob becomes visible: ``--foreign-frac`` with
    no bank, ``--top-k`` above the fan (the prune never fires), ``--margin`` with
    ``--margin-pairs 0``, ``--n-aug`` on an arm that does not augment.
    """
    group = _knob_group(parser)
    rows: list[EW.TermSpec] = []
    n_cand = args.n_curv * args.n_acc
    augmenting = args.arm in ("selector", "progress_only")
    for a in group._group_actions:
        v = getattr(args, a.dest)
        flag = a.option_strings[-1] if a.option_strings else a.dest
        declared = _encode(a.default)
        eff = _encode(v)
        layer = needs = None
        if a.dest == "foreign_at_eval" and v:
            needs = (bool(args.eval_aug) and args.foreign_frac > 0.0,
                     "--foreign-at-eval only reaches a pick when the SCORED "
                     "fan is augmented (--eval-aug) and a bank exists "
                     "(--foreign-frac > 0); otherwise eval scores the native "
                     "fan and this knob is INERT")
        elif a.dest == "eval_aug" and v and args.n_aug == 0:
            needs = (False,
                     "--eval-aug with --n-aug 0 augments the scored fan with "
                     "ZERO extra copies -- parsed, stamped and INERT")
        elif a.dest.startswith("foreign_") and a.dest != "foreign_frac" and                 args.foreign_frac == 0.0:
            layer = "--foreign-frac 0.0 (the bank is OFF)"
            eff = 0.0
        elif a.dest == "foreign_frac" and v > 0.0:
            keep = int(args.foreign_bank_curv * args.foreign_bank_acc * v)
            needs = (keep > 0,
                     f"foreign_frac {v} over a bank of "
                     f"{args.foreign_bank_curv * args.foreign_bank_acc} keeps "
                     f"{keep} trajectories -- the knob would be INERT")
        elif a.dest == "top_k":
            fan = n_cand * (1 + args.n_aug)
            needs = (v < fan,
                     f"top_k {v} >= the augmented fan {fan}: the coarse prune "
                     f"never fires and the coarse/fine split is inert")
        elif a.dest == "margin":
            needs = (args.margin_pairs > 0,
                     "--margin-pairs 0 removes the ranking term entirely, so "
                     "--margin is parsed, stamped and inert")
        elif a.dest in _AUGMENT_ONLY and not augmenting:
            layer = f"arm={args.arm} does not augment"
            eff = 0.0
        elif a.dest in _AUGMENT_ONLY and args.n_aug == 0 and                 args.foreign_frac == 0.0 and a.dest not in ("n_aug",
                                                            "foreign_frac"):
            layer = "augmentation OFF (n_aug 0, foreign_frac 0)"
            eff = 0.0
        elif a.dest in ("w_compliance", "w_tactical") and v > 0.0:
            needs = (False,
                     "the compliance/tactical targets need nav_cmd and the "
                     "v7.2 (lat, lon) labels; this rig supplies neither")
        if a.dest in _SELECTOR_ONLY and args.arm in ("const", "raw_floor"):
            layer = f"arm={args.arm} builds no selector"
            eff = 0.0
            needs = None
        rows.append(EW.TermSpec(term=a.dest, flag=flag, dest=a.dest,
                                declared=declared, requested=_encode(v),
                                effective=eff, layer=layer, needs=needs))
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E-DDA-2b sub-metric selector driver")
    p.add_argument("--arm", default="selector",
                   choices=("selector", "const", "raw_floor", "progress_only"))
    p.add_argument("--source", default="synthetic", choices=("synthetic", "dump"))
    p.add_argument("--fan-dump", default=None,
                   help="a banked refcv4 fan (.pt) — the real E-DDA-2b surface")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-check", action="store_true",
                   help="run the analytic curvature cross-check and exit")
    p.add_argument("--allow-discarded-weights", action="store_true")

    g = p.add_argument_group(KNOB_GROUP)
    # rig geometry
    g.add_argument("--n-windows", type=int, default=24)
    g.add_argument("--n-curv", type=int, default=9)
    g.add_argument("--n-acc", type=int, default=4)
    g.add_argument("--n-steps", type=int, default=12)
    g.add_argument("--dt", type=float, default=0.5)
    g.add_argument("--v0", type=float, default=10.0)
    # augmentation (piece 5) — OFF at the defaults
    g.add_argument("--n-aug", type=int, default=0)
    g.add_argument("--std-min", type=float, default=0.1)
    g.add_argument("--std-max", type=float, default=0.2)
    g.add_argument("--std-per-row", action="store_true")
    g.add_argument("--foreign-frac", type=float, default=0.0)
    g.add_argument("--foreign-at-eval", action="store_true",
                   help="⛔ mix the bank at eval too — the released code does "
                        "NOT; a foreign pick is not emittable. Needs "
                        "--eval-aug, or the eval fan is native and this is "
                        "INERT.")
    g.add_argument("--eval-aug", action="store_true",
                   help="augment the SCORED fan as well (V2 does: n_aug=3, "
                        "U(0.1,0.3) at test). Off by default so every arm "
                        "scores the same native fan and stays comparable.")
    # ⚠️ ODD, for the same reason the fan's grid is odd. The bank is genuinely
    # FOREIGN: its curvature range is wider than the fan's, so its members are
    # trajectories the generator does not emit -- which is the whole point of
    # V2's GTRS slice, and the reason a foreign pick must not reach an emission.
    g.add_argument("--foreign-bank-curv", type=int, default=33)
    g.add_argument("--foreign-bank-acc", type=int, default=8)
    g.add_argument("--foreign-bank-kappa-max", type=float, default=0.08)
    # selector (piece 4)
    g.add_argument("--sel-d", type=int, default=64)
    g.add_argument("--sel-heads", type=int, default=4)
    g.add_argument("--coarse-layers", type=int, default=1)
    g.add_argument("--fine-layers", type=int, default=3)
    g.add_argument("--top-k", type=int, default=32)
    # ⛔ ONE ACTION PER DEST, NOT A --flag/--no-flag PAIR. MEASURED 2026-09-10:
    # `effective_weights.explicit_dests` gives each ACTION its own sentinel but
    # keys the returned map by DEST, so a dest served by two actions has its
    # first sentinel compared against the second and reads EXPLICIT on every
    # launch -- which then trips the DISCARDED refusal on a flag nobody typed.
    # A `choices` flag has one action and is read correctly.
    g.add_argument("--self-attn", choices=("on", "off"), default="on")
    g.add_argument("--margin", type=float, default=0.05)
    g.add_argument("--margin-pairs", type=int, default=2)
    g.add_argument("--w-compliance", type=float, default=0.0)
    g.add_argument("--w-tactical", type=float, default=0.0)
    # optimisation
    g.add_argument("--steps", type=int, default=120)
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--lr", type=float, default=3e-3)
    g.add_argument("--fit-frac", type=float, default=0.5)
    return p


# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    argv_used = list(sys.argv[1:] if argv is None else argv)

    if args.source == "dump":
        # ⛔ NAMED, NOT APPROXIMATED. Inventing a fan here would produce a
        # confident number about an object that does not exist.
        print(f"[{TAG}] --source dump needs a banked refcv4 fan and the "
              f"obstacle.offline replay join; neither exists on this box. "
              f"BLOCKED, not approximated.")
        return 3

    spec = RigSpec(n_windows=args.n_windows, n_curv=args.n_curv,
                   n_acc=args.n_acc, n_steps=args.n_steps, dt=args.dt,
                   v0=args.v0, seed=args.seed)
    rig = build_rig(spec)

    if args.self_check:
        chk = rig_self_check(rig)
        print(f"[{TAG}] ANALYTIC CROSS-CHECK — curvature built vs recovered")
        for k, v in chk.items():
            print(f"[{TAG}]   {k}: {v}")
        return 0

    acfg = AUG.AugmentConfig(
        n_aug=args.n_aug, std_min=args.std_min, std_max=args.std_max,
        std_per_row=bool(args.std_per_row), foreign_frac=args.foreign_frac,
        foreign_train_only=not bool(args.foreign_at_eval))
    tcfg = T.TargetConfig(dt=args.dt)

    # --- the effective-weight table, derived from argparse -----------------
    specs = term_specs(parser, args, acfg)
    explicit = EW.explicit_dests(parser, argv_used)
    rows = EW.classify_all(specs, explicit)
    print(EW.render_table(rows, where=f"refc_selector_e2b --arm {args.arm}",
                          tag=TAG))
    bad = EW.refusals(rows, where=f"--arm {args.arm}")
    if bad and not args.allow_discarded_weights:
        for b in bad:
            print(f"[{TAG}] REFUSED: {b}")
        return 2
    stamp = EW.stamp(rows, where=f"refc_selector_e2b --arm {args.arm}",
                     explicit_source=("argv" if explicit is not None
                                      else EW.SRC_UNAVAILABLE),
                     acknowledged=bool(args.allow_discarded_weights))
    stamp["knob_dests"] = knob_dests(parser)
    stamp["values"] = {d: getattr(args, d) for d in knob_dests(parser)}
    stamp["_encoding"] = ("numbers as-is; bools 1/0; strings 1.0 when the knob "
                          "names a non-default mode. The LITERAL value of every "
                          "knob is in `values`.")

    # --- splits: fit and scored are DISJOINT -------------------------------
    b = spec.n_windows
    perm = torch.randperm(b, generator=torch.Generator().manual_seed(args.seed))
    n_fit = max(int(args.fit_frac * b), 1)
    fit, sco = perm[:n_fit], perm[n_fit:]
    if sco.numel() == 0:
        sco = fit

    quality, nc_label, tinfo = _quality_and_labels(rig["cand"], rig["ctx"], tcfg)
    q_sco, nc_sco = quality[sco], nc_label[sco]

    arm: dict = {"arm": args.arm}
    eval_cand = rig["cand"][sco]
    if args.arm == "const":
        score = torch.zeros_like(q_sco)
        arm["d"] = 0
        arm["_what"] = "every candidate scored identically"
    elif args.arm == "raw_floor":
        score, info = run_raw_floor(rig["cand"], quality, fit, sco,
                                    (1e-4, 1e-2, 1.0, 1e2, 1e4))
        arm.update(info)
    else:
        r = run_selector(rig, args, tcfg, acfg, fit, sco,
                         progress_only=(args.arm == "progress_only"))
        score = r.pop("score")
        arm["eval_fan_origin"] = r.pop("eval_origin")
        eval_cand = r.pop("eval_cand")
        if eval_cand.shape[1] != rig["cand"].shape[1]:
            # ⛔ THE SCORED FAN CHANGED, SO THE TARGETS MUST BE RECOMPUTED ON
            # IT. Reusing the native fan's quality against an augmented score
            # is a shape error that numpy-style broadcasting would sometimes
            # SILENCE rather than raise; rank_auc refuses it explicitly.
            sub = {"dt": spec.dt, "v0": rig["ctx"]["v0"][sco],
                   "lead_path": rig["ctx"]["lead_path"][sco]}
            q_sco, nc_sco, _ = _quality_and_labels(eval_cand, sub, tcfg)
        arm.update(r)
        arm["d"] = args.sel_d

    auc, n_pairs = rank_auc(score, q_sco)
    ap, n_ap, base = average_precision(score.reshape(-1), nc_sco.reshape(-1))
    pick = score.argmax(dim=1)
    best = q_sco.argmax(dim=1)
    regret = float((q_sco.gather(1, best[:, None])
                    - q_sco.gather(1, pick[:, None])).mean())

    result = {
        "_tier": "T0 rig — NOT a driving claim (see the module docstring)",
        "_evidence_class": "MEASURED (ours, this run)",
        "arm": arm,
        "n": {"n_windows_total": int(b), "n_fit_windows": int(fit.numel()),
              "n_scored_windows": int(sco.numel()),
              "n_candidates_per_window": int(spec.n_cand),
              "n_scored_candidates": int(q_sco.numel()),
              "n_rank_pairs": int(n_pairs), "n_ap": int(n_ap),
              "d": arm.get("d")},
        "metrics": {
            "rank_auc": auc,
            "rank_auc_no_information": RANK_AUC_NO_INFORMATION,
            "ap_no_contact_tiegroup": ap,
            "ap_base_rate": base,
            "ap_naive_ties_by_index": _ap_naive(score.reshape(-1),
                                                nc_sco.reshape(-1)),
            "pick_regret_composite": regret,
        },
        "target_coverage": {k: v for k, v in tinfo["coverage"].items()
                            if k != "_note"},
        "target_provenance": tinfo["provenance"],
        "effective_weights": stamp,
        "rig": {"spec": spec.__dict__,
                "self_check": rig_self_check(rig),
                "_caveat": "the scene tokens encode the same lead track the "
                           "targets read, so this rig is easy BY DESIGN; only "
                           "the controls' known values and arm separability "
                           "are readable here"},
        "argv": argv_used,
    }

    print(f"[{TAG}] arm={args.arm}  n_scored_candidates="
          f"{result['n']['n_scored_candidates']}  d={arm.get('d')}")
    print(f"[{TAG}] rank_auc={auc:.6f}  (no-information = "
          f"{RANK_AUC_NO_INFORMATION})")
    print(f"[{TAG}] AP(tie-group)={ap:.6f}  base_rate={base:.6f}  "
          f"AP(naive)={result['metrics']['ap_naive_ties_by_index']:.6f}")
    print(f"[{TAG}] pick_regret_composite={regret:.6f}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"e2b_{args.arm}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[{TAG}] wrote {path}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
