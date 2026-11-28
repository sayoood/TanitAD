"""v5.8f — the frontier ASSEMBLY: frozen v5f 30k trunk+head + W4 unicycle fan
+ W4b rescorer selection (V58F_FUSION.md §2/§4; PI directive 2026-08-09).

WHAT THIS MODULE IS. The COMPOSITION layer only — no new trainable parts, no
re-derivations. Every component is a proven artifact loaded FROZEN:

  * v5f 30k trunk + ``FlagshipV4Head``   oracle ADE 0.1975 @30k (registry
    §1.8); loaded through ``eval_flagship_v4.load_v4_from_ck`` — REUSED.
  * W4 ``UnicycleEmission`` fan          PASSED both pre-registered gates
    (``w4_gate.json``: oracle 0.1077, selected-candidate accel MAE 0.774,
    census-violation frac 0.0 — registry §1.13). The fan is per-candidate
    (a, kappa) control sequences integrated by the unicycle rollout —
    feasible BY CONSTRUCTION.
  * W4b ``W4bRescorer`` (optional)       selector recalibration on the frozen
    fan (PREREG_W4B_SELECTOR.md, both outcomes bound in advance).

SELECTION (``select_rule``). ⛔ The assembly DEFAULT is decided by the W4b
GATE at assembly time (:func:`select_rule_from_gate`), never hardcoded here —
the constructor has no default value for ``select_rule`` on purpose.

  * ``"rescorer-argmax"``        the G1 branch (selected ADE <= 0.45 m on the
                                 881 grid): recalibration suffices — deploy
                                 the rescorer's argmax.
  * ``"rescorer-top8-kincost"``  the G2 branch: the rescorer demotes to a
                                 top-8 PRUNER and the pick inside the
                                 shortlist is minimum kinematic cost
                                 ``mean|a| + 0.5*mean|jerk|`` computed FROM
                                 THE CONTROLS. ⚠️ W1 REFUTED this cost family
                                 on the OLD fan (-16.7 %, registry §1.13) —
                                 but that fan was 97.6 %-infeasible jitter,
                                 so a waypoint-space cost ranked jitter, not
                                 manoeuvre quality. On the W4 unicycle fan
                                 the controls ARE the kinematics (violations
                                 0.0), the cost is exact in control space,
                                 and the refutation does not transfer. Saying
                                 so here is what makes the rule admissible.
  * ``"frozen-argmax"``          the frozen v5f selector's own pick on the
                                 new fan — the W4 configuration (selected
                                 0.7933, near-uninformed; kept as the
                                 measured control arm).

TIER DISCIPLINE. This module is mechanism, not measurement: any NUMBER quoted
about the assembly comes from ``stack/scripts/eval_v58f.py`` with its tier
stamp (T0) and estimator notes, never from ad-hoc calls here.

GOAL ADMISSIBILITY (binding, 2026-08-03): whatever ``goal_kw`` a caller feeds
:meth:`V58F.plan` must NOT carry the situation classifier's output in any
form; the goal path and the situation path stay information-disjoint at
inference. This wrapper cannot check that — the caller's provenance must.

Scripts reuse: ``stack/scripts`` is not an importable package from tanitad at
module scope, so the proven pieces (``OffsetFeatureTap``, ``UnicycleEmission``
loader, ``frozen_forward``, ``load_v4_from_ck``) are imported LAZILY through a
sys.path insertion — the ``dynamics_encoder.py:286`` precedent — inside the
functions that need them. Where an import can genuinely be unavailable the
code mirrors the source and SAYS SO (``_OffsetTapMirror``; :meth:`V58F.plan`'s
docstring), per the assembly brief.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

__all__ = [
    "SELECT_RULES", "TOP_PRUNE_K", "KINCOST_JERK_WEIGHT", "DT",
    "kinematic_cost", "select_candidate", "accel_mae_from_controls",
    "select_rule_from_gate", "V58F", "load_w4b_rescorer", "load_v58f",
]

#: the three deployable selection rules (module docstring for gate semantics).
SELECT_RULES = ("rescorer-argmax", "rescorer-top8-kincost", "frozen-argmax")
#: G2 shortlist size — the prereg's "top-K pruner" K; its bound metric is the
#: top-8 oracle (<= 0.15 makes the pruner role viable, PREREG_W4B_SELECTOR.md).
TOP_PRUNE_K = 8
#: kinematic-cost jerk weight: cost = mean|a| + 0.5 * mean|jerk| (the task-
#: binding form; jerk = da/dt from the CONTROLS, not from waypoints).
KINCOST_JERK_WEIGHT = 0.5
#: 10 Hz dense-horizon tick. Re-declared, not imported — scripts/ is not a
#: package reachable at tanitad module scope (tactical.py:74-76 precedent);
#: source of truth train_v58f_unicycle_head.DT, cross-asserted in load_v58f().
DT = 0.1


def _ensure_scripts() -> None:
    """Make ``stack/scripts`` importable (dynamics_encoder.py:286 precedent).
    tanitad/models/v58f.py -> parents[2] == the stack root."""
    sp = str(Path(__file__).resolve().parents[2] / "scripts")
    if sp not in sys.path:
        sys.path.insert(0, sp)


# ============================================================================
# pure selection helpers (CPU-testable — tests/test_v58f.py)
# ============================================================================
def kinematic_cost(a_ctl: Tensor, dt: float = DT,
                   jerk_weight: float = KINCOST_JERK_WEIGHT) -> Tensor:
    """Per-candidate kinematic cost FROM THE CONTROLS:
    ``mean|a| + 0.5*mean|jerk|``, ``a_ctl [..., K] -> [...]``.

    jerk_k = (a_{k+1} - a_k) / dt over the K-1 control transitions; K == 1 has
    no jerk term. ⚠️ Control-space on purpose: on the unicycle fan the
    controls are the true kinematics (waypoint-derived accel == a up to the
    v >= 0 clamp), which is why the W1 refutation — a WAYPOINT-space cost on
    the 97.6 %-infeasible old fan ranking jitter — does not transfer here."""
    if a_ctl.ndim < 1 or a_ctl.shape[-1] < 1:
        raise ValueError(f"a_ctl needs a control axis, got {tuple(a_ctl.shape)}")
    a = a_ctl.float()
    cost = a.abs().mean(dim=-1)
    if a.shape[-1] >= 2:
        jerk = torch.diff(a, dim=-1) / dt
        cost = cost + jerk_weight * jerk.abs().mean(dim=-1)
    return cost


def select_candidate(select_rule: str, *, frozen_sel_idx: Tensor | None = None,
                     scores: Tensor | None = None, a_ctl: Tensor | None = None,
                     k_prune: int = TOP_PRUNE_K,
                     dt: float = DT) -> tuple[Tensor, dict]:
    """The one selection switch of the assembly. Returns ``(sel_idx [B], aux)``.

    * ``frozen-argmax``: ``frozen_sel_idx`` (the head's own ``out["sel_idx"]``
      — its full ``select()``, incl. vt-keep logic; NOT a recomputed argmax).
    * ``rescorer-argmax``: ``scores.argmax(dim=1)`` over the W4b logits.
    * ``rescorer-top8-kincost``: shortlist = top ``k_prune`` by ``scores``,
      pick = min :func:`kinematic_cost` (from ``a_ctl``) INSIDE the shortlist.
      ``aux`` carries ``shortlist [B, k]`` and the full ``kincost [B, N]``.
    """
    if select_rule not in SELECT_RULES:
        raise ValueError(f"select_rule must be one of {SELECT_RULES}, got "
                         f"{select_rule!r}")
    if select_rule == "frozen-argmax":
        if frozen_sel_idx is None:
            raise ValueError("frozen-argmax needs frozen_sel_idx (the head's "
                             "own out['sel_idx'])")
        return frozen_sel_idx, {"rule": select_rule}
    if scores is None:
        raise ValueError(f"{select_rule} needs the W4b rescorer scores "
                         f"[B, N]; got None")
    if select_rule == "rescorer-argmax":
        return scores.argmax(dim=1), {"rule": select_rule}
    # rescorer-top8-kincost
    if a_ctl is None:
        raise ValueError("rescorer-top8-kincost needs a_ctl [B, N, K] — the "
                         "kinematic cost is computed FROM THE CONTROLS")
    kk = min(int(k_prune), scores.shape[1])
    top = scores.topk(kk, dim=1).indices                       # [B, kk]
    cost = kinematic_cost(a_ctl, dt=dt)                        # [B, N]
    pick = cost.gather(1, top).argmin(dim=1)                   # [B]
    sel = top.gather(1, pick[:, None]).squeeze(1)              # [B]
    return sel, {"rule": select_rule, "shortlist": top, "kincost": cost}


def accel_mae_from_controls(a_sel: Tensor, tgt: Tensor,
                            dt: float = DT) -> float:
    """Accel MAE of the SELECTED candidate COMPUTED FROM ITS CONTROLS, vs the
    target's waypoint-derived accel profile: ``mean |a_sel[:, :K-1] -
    accel_wp(tgt)|`` in m/s².

    ``a_sel [B, K]`` commanded accels; ``tgt [B, K, 2]`` target waypoints (ego
    origin implicit, the W4 geometry). The GT side is
    ``train_v58f_unicycle_head.speeds_and_accels`` (IMPORTED — the exact
    instrument the W4 gate's 0.774 was read with). The control side is exact
    for the unicycle fan: waypoint-derived accel of an integrated candidate
    equals its commanded a up to the v >= 0 clamp — that exactness is what
    "computed from controls" buys over finite-differencing the prediction.
    Slot alignment: control a_k moves v between finite-difference speed slots
    k and k+1, so ``a_sel[:, :K-1]`` lines up with the K-1 waypoint-derived
    accels one-for-one."""
    if a_sel.ndim != 2 or tgt.ndim != 3 or tgt.shape[-1] != 2:
        raise ValueError(f"need a_sel [B, K] and tgt [B, K, 2], got "
                         f"{tuple(a_sel.shape)} / {tuple(tgt.shape)}")
    if a_sel.shape[0] != tgt.shape[0] or a_sel.shape[1] != tgt.shape[1]:
        raise ValueError(f"B/K mismatch: {tuple(a_sel.shape)} vs "
                         f"{tuple(tgt.shape)}")
    if a_sel.shape[1] < 2:
        raise ValueError("accel needs K >= 2 horizons")
    _ensure_scripts()
    from train_v58f_unicycle_head import speeds_and_accels
    _, acc_gt = speeds_and_accels(tgt.float(), dt)             # [B, K-1]
    return float((a_sel.float()[:, :-1] - acc_gt).abs().mean())


def select_rule_from_gate(gate) -> tuple[str, dict]:
    """⭐ THE ASSEMBLY-TIME DEFAULT: read the W4b gate record (the
    ``build_w4b_gate`` schema of ``w4b_gate.json``) and return
    ``(select_rule, provenance record)`` per the pre-registered branches:

      * G1 pass (selected ADE <= 0.45)  -> ``"rescorer-argmax"``
      * G2 (G1 fail)                    -> ``"rescorer-top8-kincost"``
        (the prereg's "selector demotes to top-K pruner"; the kinematic-cost
        pick inside the shortlist is the assembly's interim mechanism until
        W7's WM-roll re-rank exists).

    ``gate`` is the dict itself or a path to the JSON. When G2 engages with
    ``pruner_viable`` False (top-8 oracle > 0.15) the record carries a warning
    — the prereg calls the pruner role NOT viable then, so the number must be
    quoted with that caveat and W7 stays the primary fix."""
    src = "dict"
    if isinstance(gate, (str, Path)):
        src = str(gate)
        gate = json.loads(Path(gate).read_text())
    try:
        g1 = gate["gate_G1_recalibration_suffices"]
        g2 = gate["gate_G2_recalibration_insufficient"]
        g1_pass = bool(g1["pass"])
    except (KeyError, TypeError) as ex:
        raise ValueError(
            f"not a W4b gate record (build_w4b_gate schema) — missing {ex} "
            f"in {src}") from ex
    rule = "rescorer-argmax" if g1_pass else "rescorer-top8-kincost"
    rec = {
        "select_rule": rule,
        "g1_pass": g1_pass,
        "selected_ade": g1.get("selected_ade"),
        "threshold_m": g1.get("threshold_m"),
        "top8_oracle": g2.get("top8_oracle"),
        "pruner_viable": g2.get("pruner_viable"),
        "variant": gate.get("variant"),
        "source": src,
        "decision": ("G1 PASS -> recalibrated selector argmax"
                     if g1_pass else
                     "G2 -> rescorer demotes to top-8 pruner + kinematic-"
                     "cost pick (W7 WM-roll re-rank remains the primary "
                     "selection fix)"),
    }
    if not g1_pass and g2.get("pruner_viable") is False:
        rec["warning"] = (
            "G2 engaged but top-8 oracle > 0.15 — the prereg calls the pruner "
            "role NOT viable at this coverage; quote any top8-kincost number "
            "with this caveat and treat W7 as the load-bearing fix")
    return rule, rec


# ============================================================================
# the mirror (used ONLY when scripts/ is unimportable — and it says so)
# ============================================================================
class _OffsetTapMirror:
    """VERBATIM mirror of ``train_v58f_unicycle_head.OffsetFeatureTap``
    (lines 185-210): capture the per-candidate query ``q`` [B, N, d] feeding
    ``AnchoredDiffusionDecoder.offset_head`` via a forward PRE-hook; the LAST
    call per forward is the final denoise pass whose offset produced the
    emitted fan. Exists only for environments where ``stack/scripts`` is not
    importable; :class:`V58F` records which implementation it used
    (``tap_source``) so a mirrored run is visible in provenance, per the
    import-it-if-importable-else-mirror-and-say-so brief."""

    def __init__(self, offset_head: nn.Module):
        self._buf: list[Tensor] = []
        self._h = offset_head.register_forward_pre_hook(
            lambda _m, args: self._buf.append(args[0]))

    def clear(self) -> None:
        self._buf.clear()

    def last(self) -> Tensor:
        if not self._buf:
            raise RuntimeError("OffsetFeatureTap(mirror): no decoder pass "
                               "captured — was head.forward run after clear()?")
        return self._buf[-1]

    def n_calls(self) -> int:
        return len(self._buf)

    def remove(self) -> None:
        self._h.remove()


# ============================================================================
# the assembly
# ============================================================================
class V58F(nn.Module):
    """v5.8f: (world, grounding, head, emission, rescorer|None) -> plan.

    An INFERENCE composition: construction freezes every parameter and puts
    all modules in eval — training any part of v5.8f happens in that part's
    own trainer (train_flagship_v4 / train_v58f_unicycle_head /
    train_w4b_selector), never through this wrapper. ``grounding`` is carried
    for the T1/canary extensions (EVAL_DOCTRINE.md); :meth:`plan` itself does
    not consume it.

    ``select_rule`` is REQUIRED — the default is a W4b-gate decision made at
    assembly time (:func:`select_rule_from_gate`), never a hardcoded value.
    ``w4_cond`` is the W4 emission's conditioning mode ('feature' = the
    offset-head query captured by the tap; 'anchor' = the flattened detached
    ``anchor_traj`` — the projection-to-manifold fallback), read from the W4
    checkpoint by :func:`load_v58f`.
    """

    def __init__(self, world, grounding, head, emission, rescorer=None, *,
                 select_rule: str, w4_cond: str = "feature", probes=None,
                 amp_on: bool = False, k_prune: int = TOP_PRUNE_K,
                 dt: float = DT):
        super().__init__()
        if select_rule not in SELECT_RULES:
            raise ValueError(
                f"select_rule must be one of {SELECT_RULES}, got "
                f"{select_rule!r} — and the DEFAULT is decided by the W4b "
                f"gate at assembly time (select_rule_from_gate), never "
                f"hardcoded")
        if select_rule.startswith("rescorer") and rescorer is None:
            raise ValueError(
                f"select_rule={select_rule!r} is a rescorer rule: the W4b "
                f"rescorer is REQUIRED (load_v58f: pass w4b_ckpt=)")
        if w4_cond not in ("feature", "anchor"):
            raise ValueError(f"w4_cond must be 'feature' or 'anchor', got "
                             f"{w4_cond!r}")
        horizons = tuple(head.cfg.horizons)
        if horizons != tuple(range(1, len(horizons) + 1)):
            raise ValueError(
                f"head horizons {horizons} are not contiguous 1..K @10 Hz — "
                f"the unicycle fan is defined on the dense tick only (the W4 "
                f"contract)")
        self.world, self.grounding = world, grounding
        self.head, self.emission, self.rescorer = head, emission, rescorer
        self.select_rule, self.w4_cond = select_rule, w4_cond
        self.probes = probes
        self.amp_on = bool(amp_on)
        self.k_prune, self.dt = int(k_prune), float(dt)
        self.horizons = horizons
        # inference composition: frozen + eval, wholesale.
        self.requires_grad_(False)
        self.eval()
        _ensure_scripts()
        try:
            from train_v58f_unicycle_head import OffsetFeatureTap
            tap_cls, self.tap_source = OffsetFeatureTap, \
                "train_v58f_unicycle_head.OffsetFeatureTap (imported)"
        except ImportError:                              # pragma: no cover
            tap_cls, self.tap_source = _OffsetTapMirror, \
                "_OffsetTapMirror (scripts/ unimportable here — MIRRORED, " \
                "and saying so per the assembly brief)"
        self.tap = tap_cls(head.decoder.offset_head)

    # ------------------------------------------------------------------ API --
    @torch.no_grad()
    def plan(self, frames: Tensor, v0: Tensor, goal_kw: dict | None = None,
             lambda_plan: float = 1.0, tgt: Tensor | None = None) -> dict:
        """Deployment-shaped plan: ``frames`` (an encode_window input), ``v0``
        [B] -> the selection dict (see :meth:`_select_and_pack`).

        ⭐ PLUMBING PROVENANCE — this MIRRORS ``frozen_forward``
        (train_v58f_unicycle_head.py:341-364, the function
        train_w4b_selector.py imports) rather than importing it, and says so:
        ``frozen_forward`` REQUIRES a labelled batch (it mints the GT target
        and the goal/imagination kwargs from batch fields), while this entry
        point serves inputs where no labelled batch exists — so the
        goal/imagination kwargs are the CALLER's ``goal_kw`` and the target is
        optional. The pattern is otherwise identical step for step: encode ->
        tap.clear -> head forward under the same bf16 autocast -> conditioning
        feature read OUTSIDE autocast in float32 (the W4 gate numerics).
        :meth:`plan_batch` is the labelled-batch twin that IMPORTS
        ``frozen_forward`` — the eval path uses it, so measurement plumbing
        cannot drift from the W4/W4b contract.

        ⚠️ ``goal_kw`` admissibility is the caller's contract (2026-08-03
        binding rule): it must not carry the situation classifier's output.
        ``tgt`` [B, K, 2] (optional) enables the accel-MAE telemetry."""
        goal_kw = dict(goal_kw or {})
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=self.amp_on):
            st = self.world.encode_window(frames)
            self.tap.clear()
            out = self.head(st, v0, lambda_plan=lambda_plan, **goal_kw)
        return self._select_and_pack(out, v0, tgt=tgt)

    @torch.no_grad()
    def plan_batch(self, batch: dict, device) -> dict:
        """Labelled-batch plan — IMPORTS ``frozen_forward`` (the exact
        plumbing train_w4b_selector trains and gates through: same autocast,
        same tap discipline, same cond branch, same GT minting), so the eval
        numbers and the banked W4/W4b gate numbers live on one code path.
        Returns the :meth:`plan` dict plus ``"tgt"`` (dense GT waypoints).
        Raises ImportError pointing at :meth:`plan` (the stated mirror) if
        ``stack/scripts`` is genuinely unreachable."""
        _ensure_scripts()
        try:
            from train_v58f_unicycle_head import frozen_forward
        except ImportError as ex:                        # pragma: no cover
            raise ImportError(
                "plan_batch needs stack/scripts on sys.path (it imports "
                "frozen_forward); for deployment-shaped inputs use plan(), "
                "the stated mirror") from ex
        out, emis_feat, v0, tgt = frozen_forward(
            self.world, self.head, self.tap, batch, device,
            probes=self.probes, amp_on=self.amp_on, cond=self.w4_cond)
        return self._select_and_pack(out, v0, tgt=tgt, emis_feat=emis_feat)

    def close(self) -> None:
        """Remove the offset-head forward hook (symmetry with the trainers'
        ``tap.remove()``)."""
        self.tap.remove()

    # ---------------------------------------------------------- internals --
    @torch.no_grad()
    def _select_and_pack(self, out: dict, v0: Tensor, *,
                         tgt: Tensor | None = None,
                         emis_feat: Tensor | None = None) -> dict:
        """fan + (a, kappa) + selection + telemetry, shared by both entries.

        Emission, rescoring and selection run OUTSIDE autocast in float32 end
        to end — the numerics every W4/W4b gate number was read at."""
        q = self.tap.last().detach().float()                     # [B, N, d]
        if emis_feat is None:
            emis_feat = (q if self.w4_cond == "feature"
                         else out["anchor_traj"].detach().float().flatten(2))
        a_ctl, kappa, fan = self.emission(emis_feat, v0)
        a_ctl, kappa, fan = a_ctl.float(), kappa.float(), fan.float()
        frozen_idx = out["sel_idx"]
        if self.rescorer is not None:
            scores = self.rescorer(q, a_ctl, kappa).float()
            scores_source = ("w4b_rescorer"
                             f"[{getattr(self.rescorer, 'variant', '?')}]")
        else:
            rl = out.get("refined_logits")
            scores = None if rl is None else rl.detach().float()
            scores_source = ("frozen_refined_logits" if scores is not None
                             else None)
        sel_idx, aux = select_candidate(
            self.select_rule, frozen_sel_idx=frozen_idx, scores=scores,
            a_ctl=a_ctl, k_prune=self.k_prune, dt=self.dt)
        ar = torch.arange(fan.shape[0], device=fan.device)
        traj = fan[ar, sel_idx]
        a_sel, kap_sel = a_ctl[ar, sel_idx], kappa[ar, sel_idx]
        tele = {
            "select_rule": self.select_rule,
            "scores_source": scores_source,
            "n_candidates": int(fan.shape[1]),
            "sel_matches_frozen_frac":
                float((sel_idx == frozen_idx).float().mean()),
            "accel_mean_abs_selected_ms2": float(a_sel.abs().mean()),
            "jerk_mean_abs_selected_ms3":
                (float((torch.diff(a_sel, dim=-1) / self.dt).abs().mean())
                 if a_sel.shape[-1] >= 2 else 0.0),
            "kappa_mean_abs_selected_1pm": float(kap_sel.abs().mean()),
            "kincost_selected_mean":
                float(kinematic_cost(a_sel, dt=self.dt).mean()),
        }
        if tgt is not None:
            tele["accel_mae_selected_from_controls_ms2"] = \
                accel_mae_from_controls(a_sel, tgt, dt=self.dt)
            tele["_accel_mae_note"] = (
                "selected candidate's COMMANDED accels (controls) vs the "
                "target's waypoint-derived profile (speeds_and_accels "
                "geometry) — exact for the unicycle fan up to the v>=0 clamp")
        else:
            tele["accel_mae_selected_from_controls_ms2"] = None
            tele["_accel_mae_note"] = ("no target supplied (deployment call) "
                                       "— control-space magnitudes above are "
                                       "the available telemetry")
        if "shortlist" in aux:
            tele["shortlist_k"] = int(aux["shortlist"].shape[1])
        res = {"fan": fan, "controls": {"a": a_ctl, "kappa": kappa},
               "sel_idx": sel_idx, "traj": traj, "scores": scores,
               "telemetry": tele, "head_out": out, "v0": v0}
        if tgt is not None:
            res["tgt"] = tgt
        if "kincost" in aux:
            res["kincost"], res["shortlist"] = aux["kincost"], aux["shortlist"]
        return res


# ============================================================================
# loaders (POD-SIDE for the real artifacts; arg validation is CPU-tested)
# ============================================================================
def load_w4b_rescorer(path, device, *, k_expected: int,
                      offset_in_features: int):
    """Load + FREEZE a trained ``w4b_rescorer.pt`` (the train_w4b_selector
    save format: rescorer/variant/feat_dim/in_dim/k/step/...), cross-checked
    against the loaded head so a mismatched checkpoint fails loudly."""
    _ensure_scripts()
    from train_w4b_selector import W4bRescorer
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("rescorer", "variant", "feat_dim", "k"):
        if not (isinstance(ck, dict) and key in ck):
            raise ValueError(f"[v58f] w4b_ckpt {path} has no '{key}' key — "
                             f"not a train_w4b_selector w4b_rescorer.pt")
    if int(ck["k"]) != int(k_expected):
        raise ValueError(f"[v58f] W4b rescorer K={ck['k']} != head horizons "
                         f"K={k_expected}")
    if int(ck["feat_dim"]) != int(offset_in_features):
        raise ValueError(f"[v58f] W4b rescorer feat_dim={ck['feat_dim']} != "
                         f"offset_head.in_features={offset_in_features} — "
                         f"wrong trunk for this rescorer")
    rescorer = W4bRescorer(feat_dim=int(ck["feat_dim"]), k=int(ck["k"]),
                           variant=str(ck["variant"])).to(device)
    if "in_dim" in ck and int(ck["in_dim"]) != rescorer.in_dim:
        raise ValueError(f"[v58f] W4b in_dim mismatch: ckpt {ck['in_dim']} "
                         f"vs rebuilt {rescorer.in_dim}")
    rescorer.load_state_dict(ck["rescorer"])
    rescorer.eval()
    rescorer.requires_grad_(False)
    meta = {"variant": str(ck["variant"]), "w4b_step": ck.get("step"),
            "w4b_base_ckpt": ck.get("base_ckpt"),
            "w4b_w4_ckpt": ck.get("w4_ckpt"), "in_dim": rescorer.in_dim,
            "feat_dim": int(ck["feat_dim"]), "k": int(ck["k"])}
    return rescorer, meta


def load_v58f(v5f_ckpt, w4_ckpt, w4b_ckpt=None, *, select_rule: str,
              device="cuda", frame=None, head_config=None, anchors_dense=None,
              probe_vocab=None, amp_on: bool | None = None):
    """Assemble a :class:`V58F` from the three checkpoints. Returns
    ``(model, provenance)``.

    * v5f trunk+head via ``eval_flagship_v4.load_v4_from_ck`` (REUSED — the
      same STRICT loader every banked v5f number went through; it also
      freezes world/grounding/head). ``head_config`` / ``probe_vocab``
      default to siblings of the checkpoint, ``anchors_dense`` should be
      passed explicitly (the loader itself warns when absent).
    * W4 emission via ``train_w4b_selector.load_w4_emission`` (REUSED; frozen,
      K- and feat_dim-cross-checked; its ``cond_mode`` decides ``w4_cond``).
    * W4b rescorer via :func:`load_w4b_rescorer` — REQUIRED for the rescorer
      rules, REFUSED for ``frozen-argmax`` (it would be silently ignored —
      the ``eval_flagship_v4`` ``--c2-scorer`` refusal pattern).

    Argument validation runs BEFORE any file I/O so a mis-assembled call fails
    on the arguments, not 3 GB of checkpoint later."""
    if select_rule not in SELECT_RULES:
        raise ValueError(
            f"select_rule must be one of {SELECT_RULES}, got {select_rule!r}"
            f" — resolve 'from-gate' with select_rule_from_gate(w4b_gate) "
            f"BEFORE calling the loader")
    needs_rescorer = select_rule.startswith("rescorer")
    if needs_rescorer and not w4b_ckpt:
        raise ValueError(
            f"select_rule={select_rule!r} is a rescorer rule: w4b_ckpt (the "
            f"trained w4b_rescorer.pt) is REQUIRED")
    if (not needs_rescorer) and w4b_ckpt:
        raise ValueError(
            "w4b_ckpt given with select_rule='frozen-argmax' — it would be "
            "silently ignored; drop it or pick a rescorer rule (the "
            "eval_flagship_v4 --c2-scorer refusal pattern)")
    _ensure_scripts()
    from eval_flagship_v4 import load_v4_from_ck
    from train_v58f_unicycle_head import DT as _W4_DT
    from train_w4b_selector import load_w4_emission
    if abs(float(_W4_DT) - DT) > 1e-12:                  # pragma: no cover
        raise RuntimeError(f"DT drifted: v58f {DT} vs W4 head {_W4_DT}")
    device = str(device)
    if device == "cuda" and not torch.cuda.is_available():
        print("[v58f] WARNING: cuda unavailable, falling back to cpu",
              flush=True)
        device = "cpu"
    if amp_on is None:
        amp_on = device == "cuda"

    print(f"[v58f] loading v5f checkpoint {v5f_ckpt} ...", flush=True)
    ck = torch.load(v5f_ckpt, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "head" in ck):
        raise ValueError("[v58f] v5f_ckpt has no 'head' key — v5.8f composes "
                         "the v4 planner head's fan; a plain trunk has no fan")
    head_cfg_path = str(head_config or Path(v5f_ckpt).parent / "config.json")
    world, grounding, head, base_step, hcfg, goal_head = load_v4_from_ck(
        ck, device,
        head_config_path=(head_cfg_path if Path(head_cfg_path).exists()
                          else None),
        anchors_dense_path=anchors_dense, frame=frame)
    del ck
    horizons = tuple(head.cfg.horizons)
    K = len(horizons)

    probes = None
    if getattr(head.cfg, "cond_imagination", False):
        pv = Path(probe_vocab or (Path(v5f_ckpt).parent / "probe_vocab.pt"))
        if not pv.exists():
            raise ValueError(f"[v58f] cond_imagination head but no {pv} — a "
                             f"silent skip would run a head minus 32 inputs")
        probes = torch.load(pv, map_location=device)
        print(f"[v58f] imagination probes: {tuple(probes.shape)}", flush=True)

    feat_dim_q = int(head.decoder.offset_head.in_features)
    emission, w4_cond, w4_meta = load_w4_emission(
        w4_ckpt, device, k_expected=K, offset_in_features=feat_dim_q)
    print(f"[v58f] W4 emission loaded: cond={w4_cond} "
          f"feat_dim={w4_meta['feat_dim']} K={K} — FROZEN", flush=True)

    rescorer, w4b_meta = None, None
    if needs_rescorer:
        rescorer, w4b_meta = load_w4b_rescorer(
            w4b_ckpt, device, k_expected=K, offset_in_features=feat_dim_q)
        print(f"[v58f] W4b rescorer loaded: variant={w4b_meta['variant']} "
              f"in_dim={w4b_meta['in_dim']} (step {w4b_meta['w4b_step']}) — "
              f"FROZEN", flush=True)

    model = V58F(world, grounding, head, emission, rescorer,
                 select_rule=select_rule, w4_cond=w4_cond, probes=probes,
                 amp_on=bool(amp_on))
    assert not any(p.requires_grad for p in model.parameters()), \
        "v5.8f is an inference assembly — everything frozen"
    prov = {
        "assembly": "v5.8f (V58F_FUSION.md §2/§4)",
        "v5f_ckpt": str(v5f_ckpt), "base_step": base_step,
        "w4_ckpt": str(w4_ckpt), "w4_meta": w4_meta,
        "w4b_ckpt": (str(w4b_ckpt) if w4b_ckpt else None),
        "w4b_meta": w4b_meta,
        "select_rule": select_rule, "w4_cond": w4_cond,
        "horizons_K": K, "feat_dim_q": feat_dim_q,
        "goal_head_present": bool(goal_head is not None),
        "tap_source": model.tap_source,
        "device": device, "amp_on": bool(amp_on),
        "frozen": ("world+grounding+head via load_v4_from_ck; emission via "
                   "load_w4_emission; rescorer via load_w4b_rescorer; V58F "
                   "ctor re-freezes wholesale"),
    }
    print(f"[v58f] assembled: select_rule={select_rule} K={K} "
          f"n_candidates={head.decoder.anchors.shape[0]} base_step={base_step}",
          flush=True)
    return model, prov
