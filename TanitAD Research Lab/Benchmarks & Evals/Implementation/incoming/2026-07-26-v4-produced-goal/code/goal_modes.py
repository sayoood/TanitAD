"""goal_modes.py — WHERE THE EVALUATED MODEL'S GOAL COMES FROM, as a switch.

``goal_provenance.py`` is the *disclosure* ("this number was fed an oracle").
This module is the *fix*: an explicit, recorded ``goal_mode`` that lets an
evaluator feed the flagship-v4 head a goal the model **produced from its own
observations** instead of one minted from the ego's own future poses.

It is the v4 port of the pattern already landed for REF-C
(``taniteval/refc_eval.py``'s ``nav_mode`` / ``resolve_nav``, 2026-07-26).

WHAT IS ACTUALLY PRIVILEGED — MEASURED, and NARROWER than previously written
---------------------------------------------------------------------------
``GATE_PROTOCOL.md`` §0.8 and ``goal_provenance.py`` both name **four** oracle
fields — ``route``, ``route_graded``, ``vt_band``, ``vt_speed``. Reading the
code that actually assembles them (``train_flagship_v4._goal_inputs``, the
single function both the trainer and ``eval_flagship_v4`` call) shows it is
**three**::

    if cfg.cond_vtarget:
        kw["vt_band"]  = batch.get("vt_band", ...)     # ORACLE  (future speed)
        kw["vt_speed"] = v0                            # <-- NOT the batch field
    if cfg.cond_route:
        kw["route"]        = batch.get("route", ...)        # ORACLE (future pose)
        kw["route_graded"] = batch.get("route_graded", ...)  # ORACLE (future pose)

``vt_speed`` is overwritten with ``v0 = pose_last[:, 3]`` — the **last observed
speed** — and the batch's future-derived ``vt_speed`` never reaches the head, in
training or in eval. So ``vt_speed`` is an observation, not a goal oracle, and
the selection penalty it drives (``FlagshipV15Head.select``:
``-|v_term - clamp(vt_speed, v0 +- reach)|``) reduces to a *hold-v0* term.

The three genuinely future-derived channels are:

===============  ==================================================  ===========
field            minted by                                            horizon
===============  ==================================================  ===========
``route``        ``refb_labels.route_from_future_v3(poses, last)``     <=25 s fwd
``route_graded`` same call, ``tanh(mean_curv / CURV_TURN_PER_M)``      <=25 s fwd
``vt_band``      ``lake.vtarget.vtarget_v2`` -> ``vtarget_band``       10-20 s fwd
===============  ==================================================  ===========

THE MODES
---------
``oracle`` (**default — do not change it silently**)
    Calls ``train_flagship_v4._goal_inputs`` verbatim. This is the historical
    path; every published v4 MODE-B number is one of these, and keeping it
    bit-identical is what lets the record be *corrected* rather than *replaced*.

``produced``
    Two passes over ONE encode of the observation window:

      pass 1  ``goal_head(states[:, -1])`` -> the four strategic goal scalars
              ``(ttm, curv_3s, curv_5s, tspeed_5s)`` in
              ``v4_labels.STRAT_SCALAR_NAMES`` order. ``states`` is
              ``world.encode_window(frames)`` — frames only. No future, no
              label, no batch goal field is read.
      pass 2  those scalars are mapped into the head's goal channels and the
              planner head is run on them.

    The mapping, and its honest limits:

    * ``route_graded <- tanh(curv_5s_pred / refb_labels.CURV_TURN_PER_M)``.
      This is the label's OWN formula: ``route_graded`` is defined as
      ``tanh(mean_curv / CURV_TURN_PER_M)`` where ``mean_curv = net_dyaw / arc``,
      and ``curv_5s`` is *the same functional quantity* (``v4_labels
      .mean_curvature`` = net-heading-change / arc-length) — the head is trained
      to regress it. ⚠️ The **horizon differs**: the label integrates over the
      adaptive route horizon (up to ``NAV_HORIZON_STEPS`` = 25 s), the produced
      one over exactly 5 s. Same units, same sign convention (+ = left),
      different window. This is an approximation and is stamped as one.
    * ``route <- sign/threshold on that produced graded value`` at
      ``|g| >= tanh(1)`` (i.e. ``|mean_curv| >= CURV_TURN_PER_M``, the label's
      own junction-curvature unit): ``+`` -> ``ROUTE_LEFT``, ``-`` ->
      ``ROUTE_RIGHT``, else ``ROUTE_STRAIGHT``. ⚠️ The label's 3-class rule is
      NOT this: it thresholds ``peak_kappa`` with a transience gate. The head
      produces no ``peak_kappa``, so the discrete class is a **derived
      approximation of a differently-defined label**, not a reproduction of it.
      :func:`agreement` measures exactly how far apart they land.
    * ``vt_band <- vocab.vtarget_band(tspeed_5s_pred)``. ⚠️ Same caveat, larger:
      the label's ``vt_band`` comes from ``vtarget_v2`` = the **85th percentile
      of future speed over 10-20 s, dropping steps braking harder than
      1.5 m/s²**, while ``tspeed_5s`` is the smoothed speed at exactly 5 s
      ahead. Same units and the same banding function, a different statistic.
    * ``vt_speed <- v0``, unchanged — it was never an oracle (see above).

``neutral``
    No goal information at all: the head's LEARNED "no goal given" rows. The
    cheapest control that makes the oracle-vs-produced gap readable — it answers
    "is the produced goal worth more than nothing?", which a two-arm comparison
    cannot. Which row is used depends on what the run actually trained:
    ``goal_dropout > 0`` trains the DROPPED rows (``ROUTE_DROPPED`` = 4,
    ``VT_DROPPED`` = 23), so those are used; with ``goal_dropout == 0`` those
    rows are at their init values and would inject noise, so the sentinels the
    LABELER really emits are used instead (``ROUTE_UNKNOWN`` = 3 and the
    ``vt_valid == False`` fill, which is also 23). The choice is recorded.

NO CHANNEL IS EVER INVENTED. If a checkpoint has no ``goal_head``, ``produced``
does not fabricate a substitute: it refuses unless ``--goal-fallback`` is passed,
and the fallback it then uses (hold-last-observed / neutral rows) is named in the
provenance block and in ``method``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

GOAL_MODES = ("oracle", "produced", "neutral")

#: ``FlagshipV15Head`` embedding geometry (models/flagship_v15.py:71-81).
N_VTARGET_BANDS = 23        # real bands 0..22
VT_DROPPED = 23             # the learned "no vtarget given" row
N_ROUTE_CLASSES = 4         # left/straight/right + the v2.1 UNKNOWN sentinel
ROUTE_DROPPED = 4           # the learned "no route given" row


# ---------------------------------------------------------------------------
# pass 1 — the model's own goal, from the observation window only
# ---------------------------------------------------------------------------
@torch.no_grad()
def produce_goal_scalars(goal_head, states: torch.Tensor) -> torch.Tensor:
    """``states`` [B, W, S] (``world.encode_window(frames)``) -> [B, 4] scalars.

    Reads the LAST readout state, which is what ``train_flagship_v4
    .v4_loss_step`` trains the head on (``goal_head(states[:, -1])``). Nothing
    but the encoded observation window enters."""
    return goal_head(states[:, -1]).float()


def scalars_to_goal(scalars: torch.Tensor, v0: torch.Tensor) -> dict:
    """[B, 4] produced ``(ttm, curv_3s, curv_5s, tspeed_5s)`` -> the head's goal
    channels. Pure; no batch, no poses, no future. See the module docstring for
    each mapping and its stated approximation."""
    import refb_labels as rl
    from tanitad.lake.vocab import VTARGET_TOKENS, vtarget_band

    dev = scalars.device
    curv_5s = scalars[:, 2]
    tspeed = scalars[:, 3]

    # route_graded: the LABEL'S OWN formula, at the head's 5 s horizon.
    graded = torch.tanh(curv_5s / rl.CURV_TURN_PER_M)

    # route class: |mean_curv| >= CURV_TURN_PER_M  <=>  |graded| >= tanh(1).
    thr = float(torch.tanh(torch.tensor(1.0)))
    route = torch.full_like(graded, float(rl.ROUTE_STRAIGHT))
    route = torch.where(graded >= thr, torch.full_like(graded,
                                                       float(rl.ROUTE_LEFT)),
                        route)
    route = torch.where(graded <= -thr, torch.full_like(graded,
                                                        float(rl.ROUTE_RIGHT)),
                        route)
    route = route.long()

    # vt_band: the same banding function the labeler uses, on the produced speed.
    toks = list(VTARGET_TOKENS)
    band = torch.tensor([toks.index(vtarget_band(float(s)))
                         for s in tspeed.detach().cpu()],
                        dtype=torch.long, device=dev)

    return {"route": route, "route_graded": graded.to(v0.dtype),
            "vt_band": band, "vt_speed": v0,
            "_produced_thr": thr}


def neutral_goal(v0: torch.Tensor, goal_dropout: float) -> tuple[dict, str]:
    """The head's learned "no goal given" state -> (kwargs, which_row_and_why)."""
    b = v0.shape[0]
    dev = v0.device
    if goal_dropout and goal_dropout > 0:
        r, vb = ROUTE_DROPPED, VT_DROPPED
        note = (f"DROPPED rows (route={r}, vt_band={vb}) — goal_dropout="
                f"{goal_dropout} > 0, so these rows were trained as an explicit "
                "'no goal given' state")
    else:
        import refb_labels as rl
        r, vb = int(rl.ROUTE_UNKNOWN), VT_DROPPED
        note = (f"LABELER sentinels (route=ROUTE_UNKNOWN={r}, vt_band={vb}) — "
                f"goal_dropout={goal_dropout}, so the DROPPED rows are at their "
                "init values and feeding them would inject untrained noise; "
                "these are the rows the labeler itself emits on unjudgeable "
                "windows, so they are trained")
    return ({"route": torch.full((b,), r, dtype=torch.long, device=dev),
             "route_graded": torch.zeros(b, dtype=v0.dtype, device=dev),
             "vt_band": torch.full((b,), vb, dtype=torch.long, device=dev),
             "vt_speed": v0}, note)


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------
def resolve_goal(mode: str, *, head, batch: dict, v0: torch.Tensor,
                 states: torch.Tensor, goal_head=None,
                 allow_fallback: bool = False) -> tuple[dict, dict]:
    """-> (kwargs for ``head(...)``, per-batch provenance record).

    ``oracle`` delegates to ``train_flagship_v4._goal_inputs`` **verbatim** — the
    historical path is not reimplemented here, so it cannot drift from it."""
    assert mode in GOAL_MODES, f"goal_mode must be one of {GOAL_MODES}, got {mode!r}"
    cfg = head.cfg
    from train_flagship_v4 import _goal_inputs

    if mode == "oracle":
        return _goal_inputs(cfg, batch, v0), {"mode": "oracle", "fallback": None}

    if mode == "neutral":
        kw, note = neutral_goal(v0, float(getattr(cfg, "goal_dropout", 0.0) or 0.0))
        return _filter(cfg, kw), {"mode": "neutral", "fallback": None,
                                  "neutral_rows": note}

    # ---- produced ---------------------------------------------------------
    if goal_head is None:
        if not allow_fallback:
            raise SystemExit(
                "[goal-mode] --goal-mode produced, but this checkpoint carries "
                "NO 'goal_head'. There is no model-side producer for route / "
                "route_graded / vt_band on it, so a produced-goal number cannot "
                "be measured. Re-run with --goal-fallback to fall back to the "
                "NEUTRAL rows (recorded as such, and NOT a produced-goal "
                "number), or use --goal-mode oracle/neutral explicitly.")
        kw, note = neutral_goal(v0, float(getattr(cfg, "goal_dropout", 0.0) or 0.0))
        return _filter(cfg, kw), {
            "mode": "produced", "fallback": "neutral",
            "neutral_rows": note,
            "_read": "NO goal_head on this checkpoint -> this is NOT a "
                     "produced-goal number; it is the neutral control."}

    sc = produce_goal_scalars(goal_head, states)
    kw = scalars_to_goal(sc, v0)
    thr = kw.pop("_produced_thr")
    return _filter(cfg, kw), {"mode": "produced", "fallback": None,
                              "route_threshold_abs_graded": thr,
                              "scalars": sc.cpu()}


def _filter(cfg, kw: dict) -> dict:
    """Drop channels the head is not conditioned on, so the kwargs match
    ``_goal_inputs``' shape exactly for the same config."""
    out = {}
    if getattr(cfg, "cond_vtarget", False):
        out["vt_band"] = kw["vt_band"]
        out["vt_speed"] = kw["vt_speed"]
    if getattr(cfg, "cond_route", False):
        out["route"] = kw["route"]
        out["route_graded"] = kw["route_graded"]
    return out


# ---------------------------------------------------------------------------
# how good IS the produced goal? (so the gap is readable, not just reported)
# ---------------------------------------------------------------------------
class GoalAgreement:
    """Accumulates produced-vs-oracle agreement over the eval pass.

    The oracle values are used ONLY as a reference here — they are never fed to
    the model in ``produced`` mode. Without this, an oracle-vs-produced gap
    cannot be attributed: a large gap from a produced goal that is pure noise
    means something different from a large gap from an accurate one."""

    def __init__(self):
        self.route_ok = self.route_n = 0
        self.vt_exact = self.vt_within1 = self.vt_n = 0
        self._sc_pred: list[torch.Tensor] = []
        self._sc_true: list[torch.Tensor] = []
        self._sc_mask: list[torch.Tensor] = []
        self.route_conf = torch.zeros(5, 5, dtype=torch.long)   # oracle x produced

    def update(self, produced: dict, batch: dict, scalars: torch.Tensor | None):
        if "route" in produced and "route" in batch:
            o = batch["route"].detach().cpu().long()
            p = produced["route"].detach().cpu().long()
            self.route_n += o.numel()
            self.route_ok += int((o == p).sum())
            for a, b in zip(o.tolist(), p.tolist()):
                if 0 <= a < 5 and 0 <= b < 5:
                    self.route_conf[a, b] += 1
        if "vt_band" in produced and "vt_band" in batch:
            o = batch["vt_band"].detach().cpu().long()
            p = produced["vt_band"].detach().cpu().long()
            self.vt_n += o.numel()
            self.vt_exact += int((o == p).sum())
            self.vt_within1 += int(((o - p).abs() <= 1).sum())
        if scalars is not None and "strat_scalars" in batch:
            self._sc_pred.append(scalars.detach().cpu())
            self._sc_true.append(batch["strat_scalars"].detach().cpu().float())
            self._sc_mask.append(batch["strat_scalar_mask"].detach().cpu().bool())

    def report(self) -> dict:
        from v4_labels import STRAT_SCALAR_NAMES
        out: dict = {
            "_read": ("produced-vs-ORACLE agreement. The oracle values are a "
                      "REFERENCE here and were never fed to the model in "
                      "produced mode. Low agreement means the produced goal is "
                      "weak, which is part of what an oracle-vs-produced gap "
                      "measures — it does not make the gap wrong."),
            "route_exact_agreement": (round(self.route_ok / self.route_n, 4)
                                      if self.route_n else None),
            "route_n": self.route_n,
            "route_confusion_oracle_rows_x_produced_cols":
                self.route_conf.tolist(),
            "vt_band_exact_agreement": (round(self.vt_exact / self.vt_n, 4)
                                        if self.vt_n else None),
            "vt_band_within_1_agreement": (round(self.vt_within1 / self.vt_n, 4)
                                           if self.vt_n else None),
            "vt_band_n": self.vt_n,
        }
        if self._sc_pred:
            P = torch.cat(self._sc_pred)
            T = torch.cat(self._sc_true)
            M = torch.cat(self._sc_mask)
            per = {}
            for i, name in enumerate(STRAT_SCALAR_NAMES):
                m = M[:, i]
                n = int(m.sum())
                if n < 2:
                    per[name] = {"n": n, "r2": None, "note": "too few valid"}
                    continue
                p, t = P[m, i].double(), T[m, i].double()
                ss_res = float(((t - p) ** 2).sum())
                ss_tot = float(((t - t.mean()) ** 2).sum())
                per[name] = {
                    "n": n,
                    "r2": round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
                    "rmse": round(float(((t - p) ** 2).mean().sqrt()), 4),
                    "pred_mean": round(float(p.mean()), 4),
                    "true_mean": round(float(t.mean()), 4)}
            out["goal_scalar_regression_vs_oracle_label"] = per
            out["goal_scalar_note"] = (
                "R2 of the model's OWN goal_head against the kinematic label it "
                "was trained on. This is the quality of the produced goal, "
                "measured on the same windows the ADE is measured on.")
        return out


# ---------------------------------------------------------------------------
# the provenance block that goes into the result JSON
# ---------------------------------------------------------------------------
def provenance(mode: str, *, cfg, fallback=None, extra: dict | None = None) -> dict:
    """The ``goal_provenance``-shaped record for a given mode. Always stamped —
    ``oracle`` included — so no artifact can be read without knowing which it is."""
    import goal_provenance as gp

    src = {"oracle": "oracle_gt_future",
           "produced": "produced_from_vision",
           "neutral": "dropped"}[mode]
    if mode == "produced" and fallback == "neutral":
        src = "dropped"
    fields = [f for f, on in (("vt_band", getattr(cfg, "cond_vtarget", False)),
                              ("route", getattr(cfg, "cond_route", False)),
                              ("route_graded", getattr(cfg, "cond_route", False)))
              if on]
    block = gp.disclose("eval_flagship_v4", goal_source=src, fields=fields,
                        quiet=(mode != "oracle"))
    block["goal_mode"] = mode
    block["goal_mode_fallback"] = fallback
    block["oracle_fields_fed"] = fields if mode == "oracle" else []
    block["vt_speed_note"] = (
        "vt_speed is NOT in the oracle field list: `_goal_inputs` overwrites it "
        "with v0 = pose_last[:,3], the LAST OBSERVED speed, in training and in "
        "eval alike. GATE_PROTOCOL.md 0.8 and goal_provenance.py list it as an "
        "oracle field; that is a documentation error, corrected here against the "
        "code (train_flagship_v4.py `_goal_inputs`).")
    block["mode_semantics"] = {
        "oracle": "route/route_graded/vt_band minted from the ego's own FUTURE "
                  "poses. Historical path; upper bound, not deployable.",
        "produced": "route/route_graded/vt_band derived from the model's own "
                    "goal_head run on the encoded observation window. No future, "
                    "no label. THE DEPLOYABLE PATH.",
        "neutral": "the head's learned no-goal-given rows. The control that "
                   "makes the oracle-vs-produced gap readable.",
    }
    if extra:
        block.update(extra)
    return block
