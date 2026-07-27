"""C2 — select a candidate by proximity to ONE world-model reference roll-out.

THE RULE
--------
Roll the scoring world model forward **once** per window under the *observed*
action, zero-order-held, decode it to metres, and pick the fan candidate whose
mean-over-horizon L2 distance to that single reference trajectory is smallest::

    ref        = rollout_decode(predictor, states, win_actions, None, sr, K)
    cost[b, n] = mean_k || fan[b, n, k] - ref[b, k] ||_2
    pick[b]    = argmin_n cost[b, n]

**One extra roll-out per window — not one per candidate.** The per-candidate
roll (:func:`tanitad.models.flagship_v15.imagine_candidates`, rule "A1" in the
E-V5-1 stream) costs N=256 times more and is measured *worse*; this module is
deliberately the cheap rule.

There is **no gate, no threshold and no fitter here.** Same precedent as
:mod:`tanitad.models.readout_selection`: the module ships a rule, and
``test_wm_reference_select.py`` inspects every public signature to enforce that
nothing in it can consume ground truth.

WHAT IS MEASURED — read this before changing :data:`WM_REFERENCE_SELECT_DEFAULT`
-------------------------------------------------------------------------------
881 canonical val windows / 40 episode clusters, paired episode-cluster
bootstrap (``taniteval.ci``, B=2000, unit = episode), applied to **100 % of
windows** (``selected_frac`` 1.000). Primary artifacts:
``…/incoming/2026-07-26-v5-imagination-selection/raw/v5_{v4,v1}_windows_reduced.pt``;
re-verified independently 2026-07-28 in
``…/incoming/2026-07-28-c2-ship/raw/c2_reverify.json``.

=========================================  ==========  ==============================  ====
scoring world model                        ade_0_2s    paired Δ vs the as-trained pick  sep
=========================================  ==========  ==============================  ====
— (as-trained selector, the baseline)      0.8563      —                               —
**v1** (``flagship-speedjerk-30k``)        **0.5645**  **-0.2918** [-0.4233, -0.1598]  yes
⛔ **v4's OWN world model (self-scoring)**  1.0653      **+0.2090** [+0.0550, +0.3642]  yes
=========================================  ==========  ==============================  ====

⛔ **THE RULE IS NOT SAFE ON EVERY ARM. It is separated-WORSE when an arm scores
its own fan** — the same code, the same fan, the same windows, a different
scoring world model. That is why :data:`WM_REFERENCE_SELECT_DEFAULT` is
``False`` and why :func:`resolve_scorer_tag` refuses to self-score by omission.
The sign of this rule is a property of the *scorer*, not of the rule.

WHAT IT DOES NOT DO
-------------------
* It does **not** improve the fan. It re-selects inside a frozen proposal set,
  so ``oracle_in_fan`` is unchanged by construction and the win is capped by it.
* It is **not** a bound on trajectory quality. v1's world model scoring *its own*
  2 s output reaches 0.4271; projected onto v4's 256-anchor fan the same scorer
  reaches 0.5645 — a **0.1374 m quantisation tax** paid to the fan.
* It carries **no gate**. A conditional variant must report its firing rate:
  see :func:`selection_telemetry`, which always emits ``selected_frac``.
* It is **not** validated in combination with the reachability clamp
  (``V15Config.sel_reach_clamp``). ``keep=`` exists so a caller can try it; the
  measured rows above have ``keep=None``.
"""
from __future__ import annotations

import torch
from torch import Tensor

from tanitad.models.flagship_v15 import assert_candidate_axis
from tanitad.models.metric_dynamics import rollout_decode

__all__ = [
    "WM_REFERENCE_SELECT_DEFAULT", "SELF_SCORING", "MEASURED_ARMS",
    "wm_reference_rollout", "wm_reference_cost", "select_by_wm_reference",
    "selection_telemetry", "resolve_scorer_tag",
]

#: ⛔ **OFF, and the reason is measured, not precautionary.** See the table
#: above: with an arm scoring its own fan the identical rule is separated-WORSE
#: by +0.2090 m. A default that silently changed a selector would make every
#: committed number of every arm unreproducible — the exact failure the
#: ``vision_rank`` default caused for v4. Turning this on is a per-run,
#: per-scorer decision made at the command line.
WM_REFERENCE_SELECT_DEFAULT = False

#: The sentinel a caller must pass to score a fan with the arm's own world
#: model. Self-scoring is the configuration MEASURED to be worse, so it may
#: never be reached by leaving an argument unset.
SELF_SCORING = "self"

#: MEASURED (ours) · tier CONFIRMED · unit = episode, 881 windows / 40 clusters.
#: Kept in code, not only in prose, so ``test_wm_reference_select.py`` can pin
#: the default to the evidence instead of to an opinion.
MEASURED_ARMS: dict[str, dict] = {
    "v1_scores_v4_fan": {
        "scorer": "flagship-speedjerk-30k (v1)", "fan": "flagship-v4-fromscratch-30k",
        "ade_0_2s": 0.5645, "as_trained_ade_0_2s": 0.8563,
        "paired_delta": -0.2918, "lo": -0.4233, "hi": -0.1598,
        "separated": True, "selected_frac": 1.0, "better": True,
    },
    "v4_scores_its_own_fan": {
        "scorer": "flagship-v4-fromscratch-30k (self)", "fan": "flagship-v4-fromscratch-30k",
        "ade_0_2s": 1.0653, "as_trained_ade_0_2s": 0.8563,
        "paired_delta": +0.2090, "lo": +0.0550, "hi": +0.3642,
        "separated": True, "selected_frac": 1.0, "better": False,
    },
}


def resolve_scorer_tag(scorer: str | None) -> str:
    """Refuse to self-score by omission. Returns the tag to stamp on the run.

    ``None`` is not "use my own world model" — it is a missing decision, and the
    missing decision resolves to the arm that was MEASURED separated-worse. A
    caller who genuinely wants the self-scoring diagnostic passes
    :data:`SELF_SCORING` explicitly and it is recorded in the telemetry.
    """
    if scorer is None:
        raise ValueError(
            "wm_reference_select: no scoring world model was named. This rule's "
            "SIGN depends on the scorer — MEASURED +0.2090 m WORSE when an arm "
            "scores its own fan and -0.2918 m better under v1's world model "
            f"(see MEASURED_ARMS). Pass a checkpoint, or '{SELF_SCORING}' to "
            "ask for the self-scoring diagnostic on purpose.")
    return str(scorer)


@torch.no_grad()
def wm_reference_rollout(predictor, states: Tensor, actions: Tensor,
                         step_readout, k: int) -> Tensor:
    """The ONE roll-out. ``states`` [B, W, S], ``actions`` [B, W, A] -> [B, k, 2].

    ``future_actions=None`` is the whole point: the reference is what the world
    model believes happens if the *observed* action is held, so it needs no
    plan, no candidate and no ground truth. It is the same call the deployed
    grounded rollout makes, with the future-action argument omitted — which is
    also why this is **not** ``wm_fidelity_ade_2s``: that metric hands the world
    model the TRUE future actions (``taniteval/rollout.py``,
    ``actions_source="expert_future"``) and is not a planning bar.
    """
    wp, _ = rollout_decode(predictor, states, actions, None, step_readout, k)
    return wp


def wm_reference_cost(fan: Tensor, ref: Tensor,
                      horizons: tuple[int, ...] | None = None) -> Tensor:
    """``fan`` [B, N, K, 2] · ``ref`` [B, R, 2] -> cost [B, N]. Lower is better.

    ``horizons`` is required when the fan is emitted at a SPARSE set of lead
    times (e.g. ``(5, 10, 15, 20)``) while the reference roll is dense 1..R: the
    reference is then read at ``horizons`` so the two are compared at the same
    lead times. Passing a fan and a reference of different lengths without
    ``horizons`` raises rather than broadcasting something meaningless.
    """
    if fan.dim() != 4 or fan.shape[-1] != 2:
        raise ValueError(f"fan must be [B, N, K, 2], got {tuple(fan.shape)}")
    if ref.dim() != 3 or ref.shape[-1] != 2:
        raise ValueError(f"ref must be [B, R, 2], got {tuple(ref.shape)}")
    if ref.shape[0] != fan.shape[0]:
        raise ValueError(f"batch mismatch: fan {fan.shape[0]} vs ref {ref.shape[0]}")
    r = ref
    if horizons is not None:
        idx = torch.as_tensor([h - 1 for h in horizons], device=ref.device)
        if int(idx.max()) >= ref.shape[1]:
            raise ValueError(
                f"horizons {tuple(horizons)} need {int(idx.max()) + 1} reference "
                f"steps but the roll is only {ref.shape[1]} long")
        r = ref.index_select(1, idx)
    if r.shape[1] != fan.shape[2]:
        raise ValueError(
            f"fan horizon axis {fan.shape[2]} != reference {r.shape[1]}. Pass "
            "`horizons` to read the reference at the fan's own lead times; a "
            "silent broadcast here would compare different points in time.")
    return (fan - r[:, None].to(fan.dtype)).norm(dim=-1).mean(dim=-1)


def selection_telemetry(cost: Tensor, idx: Tensor, *,
                        baseline_idx: Tensor | None = None,
                        scorer: str = "?") -> dict:
    """Degeneracy checks that must travel WITH the number.

    Two pure-noise gates in the E-CP-1 stream would have been written up as
    separated wins without a ``selected_frac`` column, so every row this module
    emits carries one — 1.000 here, because the shipped rule is unconditional
    and a conditional variant must not be able to hide its firing rate.

    ``cost_span`` and ``n_constant_rows`` catch the other degeneracy: a cost that
    does not vary across candidates cannot rank them, whatever its mean.
    """
    srt = cost.sort(dim=1).values
    tele = {
        "rule": "C2_wm_reference_proximity",
        "scorer": scorer,
        "selected_frac": 1.0,
        "n_windows": int(cost.shape[0]),
        "n_candidates": int(cost.shape[1]),
        "n_tied_argmin": int((srt[:, 0] == srt[:, 1]).sum()) if cost.shape[1] > 1 else 0,
        "n_constant_cost_rows": int((cost.std(dim=1) < 1e-12).sum()),
        "cost_span_mean": float((srt[:, -1] - srt[:, 0]).mean()),
        "n_distinct_picks": int(idx.unique().numel()),
    }
    if baseline_idx is not None:
        tele["frac_pick_equals_baseline"] = float((idx == baseline_idx).float().mean())
    return tele


def select_by_wm_reference(fan: Tensor, ref: Tensor, *,
                           horizons: tuple[int, ...] | None = None,
                           keep: Tensor | None = None,
                           baseline_idx: Tensor | None = None,
                           scorer: str = "?") -> tuple[Tensor, Tensor, dict]:
    """The rule. Returns ``(idx [B], cost [B, N], telemetry)``.

    ``keep`` [B, N] bool restricts the argmin to the surviving candidates (e.g.
    :meth:`FlagshipV15Head.reachability_mask`). ⚠️ **UNMEASURED in combination**
    — every row in :data:`MEASURED_ARMS` was produced with ``keep=None``. A row
    whose survivor set is empty keeps its whole fan rather than returning no
    plan, matching :meth:`FlagshipV15Head.select`.
    """
    cost = wm_reference_cost(fan, ref, horizons=horizons)
    # a cost that is constant along the candidate axis ranks nothing; this is the
    # same guard that would have made the E-V5-1 imagination negative fail loudly.
    assert_candidate_axis(cost, fan.shape[1], name="wm_reference_cost")
    rank = cost
    if keep is not None:
        dead = ~keep.any(dim=1)
        keep = keep | dead[:, None]
        rank = cost.masked_fill(~keep, float("inf"))
    idx = rank.argmin(dim=1)
    tele = selection_telemetry(cost, idx, baseline_idx=baseline_idx, scorer=scorer)
    tele["keep_applied"] = keep is not None
    if keep is not None:
        tele["keep_frac"] = float(keep.to(cost.dtype).mean())
        tele["_unmeasured"] = ("keep= was NOT applied in any MEASURED_ARMS row; "
                               "this configuration has no interval behind it")
    return idx, cost, tele
