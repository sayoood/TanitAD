#!/usr/bin/env python3
"""PRICE the ``vt_band`` options — run the REAL mid-run held-out gate under each.

⛔ CHANGES NO DEFAULT. Imports ``tanitad.train.heldout_goal`` (which is inert)
and drives ``tanitad.train.heldout_gate.HeldoutGate`` unmodified.

WHAT IT PRODUCES, per option
----------------------------
1. **does the gate run at all** — the shipped ``crash_today`` is the RED baseline;
2. **the probe value** ``pseudosim_composite_PSS_recovery_progress@twosided_v2``
   with a **paired episode-cluster bootstrap** (``taniteval.ci``, B=2000,
   unit = held-out episode) against a chosen reference option;
3. **a lateral / longitudinal decomposition** from ``pseudosim._cross_and_along``
   — cross-track endpoint error (LATERAL) and along-track travel (LONGITUDINAL),
   plus the ``recovery`` / ``ego_progress`` sub-scores those feed;
4. **does the early-stop still FIRE** — a deliberately degraded arm is injected
   and the gate's own ``observe`` must reach ``stop = True``.

⚠️ THE DEGRADATION IS DIRECTION-CHECKED, NOT ASSUMED. A sibling stream degraded
a planner by SLOWING IT DOWN and the composite went UP (+0.1698), because on the
gate's ``dlat = 0`` grid ``xt_hold = s_along*|tan(dpsi)|`` vanishes with
along-track travel and ``recovery`` becomes NaN by construction. The degradation
here is therefore a pure LATERAL corruption that PRESERVES along-track travel,
and ``--degrade-selfcheck`` asserts the composite actually went DOWN.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# paths                                                                        #
# --------------------------------------------------------------------------- #
def _wire_paths(stack: Path):
    repo = stack.parent
    for p in (str(stack), str(stack / "scripts"), str(repo / "taniteval")):
        if p not in sys.path:
            sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
# the degradation — LATERAL ONLY, so it cannot be a NaN artefact               #
# --------------------------------------------------------------------------- #
class LateralDegradedPlanner:
    """⛔ THE DEGRADATION THAT FAILED ITS OWN DIRECTION CHECK — kept as evidence.

    ``y_t -> y_t + drift * (t/T)^2`` in the plan's own ego frame: a smooth
    cross-track excursion leaving ``x`` (along-track) bit-identical. It LOOKS
    surgical — ``ego_progress`` and ``xt_hold`` both read ``x[:, -1]``, so the
    whole effect should land on ``recovery``'s numerator ``xt_end``.

    ⚠️ MEASURED, 2 episodes, ``flagship-v4-fromscratch-15k``: it made the arm
    BETTER. ``recovery`` 0.0291 -> 0.2364 and the composite 0.6228 -> 0.6903.
    The reason is that a CONSTANT-SIGN drift is a CORRECTION for a
    systematically-biased planner: the model's signed cross-track error is
    predominantly one-sided, so +3 m re-centres it on roughly half the grid
    (``dyaw`` is symmetric ``(-8, 0, +8)``, so a fixed sign helps one wing and
    hurts the other).

    ⇒ this is the SAME class as the sibling stream's +0.1698 slow-plan artefact:
    a degradation whose direction was assumed rather than checked. It is retained
    behind ``--degrade-mode lateral_drift`` so the failure stays reproducible, and
    the DEFAULT is :class:`RandomSelectionDegradedPlanner`.
    """

    def __init__(self, inner, drift_m: float = 3.0, **_):
        self.inner, self.drift = inner, float(drift_m)
        self.provenance = {**getattr(inner, "provenance", {}),
                           "DEGRADED": f"lateral drift {drift_m} m (quadratic in t), "
                                       f"along-track bit-identical. ⚠️ THIS "
                                       f"DEGRADATION FAILED ITS DIRECTION CHECK"}
        self.horizons = getattr(inner, "horizons", ())

    def traj(self, frames, v0, goal=None):
        tj = self.inner.traj(frames, v0, goal).clone()
        T = tj.shape[1]
        w = (torch.arange(1, T + 1, dtype=tj.dtype) / T) ** 2
        tj[..., 1] = tj[..., 1] + self.drift * w[None, :]
        return tj


class RandomSelectionDegradedPlanner:
    """⭐ THE DEGRADATION USED. The head's RANKED pick becomes a UNIFORM RANDOM
    candidate from the SAME fan.

    Why this one and not a trajectory edit:

    * it is the failure the gate was built for — ``heldout_gate``'s own docstring
      says *"Selection is the thing that regressed on v4, so a probe that
      bypasses the selector cannot see the defect it exists to catch"*;
    * its direction cannot be an artefact of the scoring geometry: the candidate
      set is unchanged, only the choice within it is, so it cannot slow the plan
      systematically (the NaN trap) and cannot re-centre a lateral bias (the trap
      :class:`LateralDegradedPlanner` fell into);
    * it is option-INDEPENDENT — the same corruption is applied under every
      option, so "does the stop fire" is attributable to the option's baseline
      rather than to a differently-sized insult.

    ``--degrade-selfcheck`` still asserts the composite went DOWN. A degradation
    is not trusted here because it sounds bad; it is trusted because it measured
    worse.
    """

    def __init__(self, inner, seed: int = 0, **_):
        self.inner = inner
        self.head = inner.head
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.provenance = {**getattr(inner, "provenance", {}),
                           "DEGRADED": "SELECTION randomised — uniform pick from "
                                       "the head's own fan (the v4 regression's "
                                       "own failure mode)"}
        self.horizons = getattr(inner, "horizons", ())

    def traj(self, frames, v0, goal=None):
        head, orig = self.head, self.head.select

        def patched(fan, refined_logits, vt_speed, vt_keep, v0_):
            traj, idx, score, tele = orig(fan, refined_logits, vt_speed,
                                          vt_keep, v0_)
            n = fan.shape[1]
            ridx = torch.randint(0, n, (fan.shape[0],), generator=self.gen)
            ridx = ridx.to(fan.device)
            rtraj = fan[torch.arange(fan.shape[0], device=fan.device), ridx]
            tele = {**tele, "DEGRADED_random_selection": True}
            return rtraj, ridx, score, tele

        head.select = patched
        try:
            return self.inner.traj(frames, v0, goal)
        finally:
            head.select = orig            # restore the bound method


DEGRADERS = {"random_select": RandomSelectionDegradedPlanner,
             "lateral_drift": LateralDegradedPlanner}


# --------------------------------------------------------------------------- #
# build the real stack                                                         #
# --------------------------------------------------------------------------- #
def build_stack(a):
    """world + head + goal_head, built EXACTLY as ``train()`` builds them.

    ⚠️ Deliberately reuses ``train_flagship_v4``'s own ``flagship4b_config`` /
    ``resolve_v2_frames`` / ``v4_config`` rather than re-declaring geometry: a
    driver that built a slightly different head would price the driver, not the
    options."""
    import dataclasses

    import train_flagship_v4 as T
    from tanitad.config import flagship4b_config
    from tanitad.models.flagship_v4 import FlagshipV4Head, v4_config
    from tanitad.models.fourbrain import WorldModel
    from tanitad.models.strategic_goal import GoalScalarConfig, GoalScalarHead

    cfg = flagship4b_config()
    cache_frame, frame = T.resolve_v2_frames(a, cfg, label="vtband_probe")
    a._frames = (cache_frame, frame)          # single source for load_val_episodes
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)

    world = WorldModel(cfg).to(a.device)
    hcfg = v4_config()
    hcfg.state_dim = world.state_dim
    hcfg.cond_imagination = False
    hcfg.window = cfg.predictor.window

    # ⚠️ HISTORICAL CHECKPOINTS PREDATE vision_rank=16. The factorised readers
    # were a raw-2048 Linear before that lever landed, so a v4 checkpoint trained
    # earlier carries lat/lon/dist heads of shape [128, 2048]. Build the head to
    # MATCH THE CHECKPOINT rather than forcing the current default and loading
    # strict=False — that would leave the three factorised heads at random init
    # and silently randomise the ranking every option is measured through.
    # ⚠️ vision_rank feeds only the factorised RANKING grafts; the vt_band
    # decision lives in condition()/vt_keep, so this is orthogonal to the options
    # — but it is recorded, not assumed.
    _peek = ({} if a.from_scratch
             else torch.load(a.ckpt, map_location="cpu", weights_only=False))
    _lat = _peek.get("head", {}).get("lat_head.0.weight")
    ckpt_d_vis = None if _lat is None else int(_lat.shape[1])
    if ckpt_d_vis is not None and ckpt_d_vis != hcfg.vision_rank:
        if ckpt_d_vis == hcfg.state_dim:
            hcfg.allow_raw_vision = True
            hcfg.vision_rank_reason = (
                "loading a PRE-vision_rank v4 checkpoint whose factorised heads "
                "are raw-2048; building the head to match it is the only way to "
                "load its trained weights. Recorded in the artifact.")
        hcfg.vision_rank = ckpt_d_vis
        print(f"[build] ⚠️ checkpoint predates vision_rank=16 — building the "
              f"factorised heads at d_vis={ckpt_d_vis} to match it", flush=True)

    head = FlagshipV4Head(hcfg).to(a.device)
    if a.anchors_dense:
        anc = torch.load(a.anchors_dense, weights_only=False)
        head.load_anchors((anc["anchors"] if isinstance(anc, dict) else anc)
                          .to(a.device))

    if a.from_scratch:
        # ⛔ STRUCTURAL LEG ONLY. Random init answers "does the gate RUN on the
        # real v5 config and the real v5 caches" — nothing else. Every probe
        # VALUE from this path is arithmetic on random weights and is marked
        # unquotable in the artifact: with an untrained head all 24 vtarget rows
        # are i.i.d. N(0, 0.02) and sel_gate is exactly 0, so band0 vs
        # VT_DROPPED CANNOT differ by anything but initialisation noise. Pricing
        # the options here would be the C13 shape.
        print("[build] ⛔ FROM-SCRATCH: structural leg only — probe VALUES from "
              "this run are NOT quotable (untrained conditioning rows).",
              flush=True)
        # ⭐ A goal_head IS built, at random init — because the real v5 run builds
        # one from step 0 (`--strategic` != off). Omitting it would make the
        # `produced` option report "cannot run" for a reason that is false on the
        # actual launch config.
        gh = GoalScalarHead(GoalScalarConfig(in_dim=world.state_dim))
        gh = gh.to(a.device).eval()
        world.eval(); head.eval()
        return world, head, gh, cfg, {"step": None, "_from_scratch": True}

    ck = _peek
    miss_w = world.load_state_dict(ck["model"], strict=False)
    miss_h = head.load_state_dict(ck["head"], strict=False)
    goal_head = None
    if ck.get("goal_head") is not None:
        goal_head = GoalScalarHead(GoalScalarConfig(in_dim=world.state_dim))
        goal_head.load_state_dict(ck["goal_head"], strict=False)
        goal_head = goal_head.to(a.device).eval()

    # ⛔ FAIL LOUD on a checkpoint that does not fit: a head loaded with
    # strict=False and 40 missing tensors is a RANDOM head wearing a step number,
    # and every option would then be priced on noise (the C13 shape).
    bad = [(len(miss_w.missing_keys), len(miss_w.unexpected_keys)),
           (len(miss_h.missing_keys), len(miss_h.unexpected_keys))]
    print(f"[build] ckpt step={ck.get('step')} goal_head={goal_head is not None} "
          f"world(missing,unexpected)={bad[0]} head(missing,unexpected)={bad[1]}",
          flush=True)
    # ⚠️ Refuse on missing PARAMETERS only. A missing non-persistent BUFFER
    # (e.g. vision_rank_proj.basis_loaded, a 0-param flag on the raw path) is not
    # a randomised weight and must not block the run — but a missing Parameter
    # is exactly the "randomly-initialised head wearing a step number" case.
    params_h = {n for n, _ in head.named_parameters()}
    missing_params = [k for k in miss_h.missing_keys if k in params_h]
    if missing_params:
        raise SystemExit(
            f"[build] REFUSING: the head is missing {len(missing_params)} trained "
            f"PARAMETERS from this checkpoint ({missing_params[:6]}). Those would "
            f"stay at random init and every probe below would price "
            f"initialisation noise instead of a trained conditioning channel.")
    if miss_h.missing_keys:
        print(f"[build] missing non-parameter buffers (benign): "
              f"{miss_h.missing_keys}", flush=True)
    # the conditioning tensors the whole decision rests on MUST be trained
    for must in ("vtarget_emb.weight", "route_emb.weight", "vt_gate", "rt_gate",
                 "sel_gate"):
        if must not in ck["head"]:
            raise SystemExit(
                f"[build] REFUSING: '{must}' is absent from this checkpoint, so "
                f"the goal-conditioning channel this decision is about was never "
                f"trained here.")
    print(f"[build] goal_dropout={hcfg.goal_dropout} "
          f"cond_vtarget={hcfg.cond_vtarget} cond_route={hcfg.cond_route} "
          f"horizons={len(hcfg.horizons)} frame={frame} cache_frame={cache_frame}",
          flush=True)
    world.eval(); head.eval()
    return world, head, goal_head, cfg, ck


# --------------------------------------------------------------------------- #
# one probe                                                                    #
# --------------------------------------------------------------------------- #
def probe_once(option, world, head, goal_head, episodes, *, device, gcfg,
               degrade=None, degrade_mode="random_select", degrade_seed=0,
               frames_of=None):
    """-> (per-window composite, eid, extras) under ``option``. Uses the SHIPPED
    ``HeldoutGate._composite_of``, so the primary is the gate's own primary."""
    from tanitad.train import heldout_goal as HGoal
    from tanitad.train.heldout_gate import HeldoutGate
    from taniteval import pseudosim as ps

    fn = HGoal.make_goal_kwargs_fn(option, head.cfg, goal_head=goal_head)
    planner = HGoal.StatesAwareSurfacePlanner(
        world, head, device=device, amp=gcfg.amp, goal_kwargs_fn=fn, option=option)
    if degrade is not None:
        planner = DEGRADERS[degrade_mode](planner, drift_m=degrade,
                                          seed=degrade_seed)

    grid = gcfg.resolved_grid()
    # ⭐ frame= is the TRAIN frame the held-out pixels were built at. The trainer
    # passes it (`HeldoutGateConfig(..., frame=frame)`), `probe` forwards it, and
    # `pseudo_evaluate` REFUSES a non-default frame without it rather than
    # re-rendering through the deployed 256x256 pinhole intrinsics. Omitting it
    # here made the v5 leg read as "the gate cannot run" for a reason that is
    # false on the real launch config — a harness defect, not a v5 one.
    pw = ps.pseudo_evaluate(planner, episodes, grid, device=device,
                            stride=gcfg.stride, horizon=gcfg.horizon,
                            batch=gcfg.batch, frames_of=frames_of, verbose=False,
                            frame=gcfg.frame)
    if pw.get("_empty"):
        raise SystemExit(f"[{option}] pseudo_evaluate produced no windows")

    g = HeldoutGate(gcfg)
    value, eid, node = g._composite_of(pw)
    sc = ps.score_windows(pw, progress_term=gcfg.progress_term)
    x, y, ref_x, ref_y = ps._cross_and_along(pw)

    fin = np.isfinite(value)
    extras = {
        "n_windows_total": int(value.size),
        "n_windows_finite": int(fin.sum()),
        "n_episodes": int(len(set(eid))),
        "planner_calls": int(pw.get("planner_calls", 0)),
        "components_admitted": g._pinned_admitted,
        "grid": pw.get("grid"),
        # --- LATERAL / LONGITUDINAL decomposition -------------------------- #
        "lateral": {
            "cross_track_end_m_mean": _m(sc["cross_track_end_m"]),
            "cross_track_end_m_p95": _p95(sc["cross_track_end_m"]),
            "cross_track_hold_matched_m_mean": _m(sc["cross_track_hold_matched_m"]),
            "recovery_mean": _m(sc["recovery"]),
            "recovery_n_finite": int(np.isfinite(sc["recovery"]).sum()),
            "_read": "recovery = 1 - xt_end/xt_hold; the LATERAL axis of the composite",
        },
        "longitudinal": {
            "along_track_end_m_mean": _m(sc["along_track_end_m"]),
            "along_track_end_m_p95": _p95(sc["along_track_end_m"]),
            "ego_progress_mean": _m(sc["ego_progress"]),
            "ego_progress_raw_ratio_mean": _m(sc["ego_progress_raw_ratio"]),
            "ego_progress_n_finite": int(np.isfinite(sc["ego_progress"]).sum()),
            "human_along_track_m_mean": _m(
                np.sqrt((ref_x[:, -1] - ref_x[:, 0]).numpy() ** 2
                        + (ref_y[:, -1] - ref_y[:, 0]).numpy() ** 2)),
            "_read": "ego_progress = plan along-track / human along-track; the "
                     "LONGITUDINAL axis of the composite",
        },
        "comfort_mean": _m(sc["comfort"]),
        "surface_provenance": planner.provenance,
    }
    return value, eid, extras, sc


def _m(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return None if a.size == 0 else round(float(a.mean()), 6)


def _p95(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return None if a.size == 0 else round(float(np.percentile(a, 95)), 6)


# --------------------------------------------------------------------------- #
# the paired interval                                                          #
# --------------------------------------------------------------------------- #
def paired(a_val, b_val, eid, n_boot, seed):
    """PAIRED episode-cluster bootstrap on the jointly-finite windows.

    ⛔ NEVER ``overlapping_holdout_se`` — CLAUDE.md: it is not a jackknife, it
    biases the point estimate bidirectionally, and on paired deltas it has been
    measured at up to x-4.15 INCLUDING A SIGN FLIP."""
    from taniteval import ci as cci
    a, b = np.asarray(a_val, float), np.asarray(b_val, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return {"error": "fewer than 2 jointly-finite windows"}
    out = cci.paired_episode_cluster_bootstrap(
        a[m], b[m], list(np.asarray(eid)[m]), n_boot=n_boot, seed=seed)
    out["n_jointly_finite"] = int(m.sum())
    return out


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stack", default="/workspace/v5gate/stack")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--from-scratch", action="store_true",
                    help="build the v5 head at random init — the STRUCTURAL leg "
                         "('does the gate run on the real config/caches'). Probe "
                         "VALUES from this path are not quotable.")
    ap.add_argument("--anchors-dense", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--options", default="crash_today,zeros_naive,band0,dropped,produced")
    ap.add_argument("--reference", default="dropped",
                    help="the option every paired delta is taken AGAINST")
    # data
    ap.add_argument("--val-cache", default="")
    ap.add_argument("--v2-val-cache", default="")
    ap.add_argument("--v2-subframe", default=None, metavar="HxW")
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--require-parity", action="store_true")
    # ⭐ geometry via the SHARED helper, so the defaults here are byte-identical
    # to train_flagship_v4's and a frame can never be spelled two ways.
    _wire_paths(Path(ap.parse_known_args(argv)[0].stack).resolve())
    from tanitad.geometry import add_geometry_args
    add_geometry_args(ap)
    # gate
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    # degradation
    ap.add_argument("--degrade-m", type=float, default=3.0)
    ap.add_argument("--degrade-mode", default="random_select",
                    choices=sorted(DEGRADERS))
    ap.add_argument("--degrade-seed", type=int, default=0)
    ap.add_argument("--degrade-selfcheck", action="store_true", default=True)
    a = ap.parse_args(argv)

    from tanitad.train import heldout_goal as HGoal
    from tanitad.train.heldout_gate import (PRIMARY_NAME, HeldoutGate,
                                            HeldoutGateConfig)

    world, head, goal_head, cfg, ck = build_stack(a)
    # ⭐ the gate config carries the TRAIN frame, exactly as train() builds it.
    gcfg = HeldoutGateConfig(every=1, episodes=a.episodes, stride=a.stride,
                             horizon=a.horizon, batch=a.batch, n_boot=a.n_boot,
                             seed=a.seed, amp=a.amp, frame=a._frames[1])
    episodes, frames_of = load_val_episodes(a, cfg)
    episodes = episodes[:a.episodes]
    print(f"[data] {len(episodes)} held-out episodes "
          f"(the gate's val_eps[:{a.episodes}])", flush=True)

    res = {
        "_what": "vt_band decision — the mid-run held-out gate under each option",
        "primary": PRIMARY_NAME,
        "estimator": f"paired episode-cluster bootstrap (B={a.n_boot}, "
                     f"unit = held-out episode); NEVER overlapping_holdout_se",
        "ckpt": a.ckpt or "(from-scratch)", "ckpt_step": ck.get("step"),
        "FROM_SCRATCH_values_not_quotable": bool(a.from_scratch),
        "head_cfg": {"cond_vtarget": head.cfg.cond_vtarget,
                     "cond_route": head.cfg.cond_route,
                     "goal_dropout": head.cfg.goal_dropout,
                     "sel_gate_trained": float(head.sel_gate.detach()),
                     "vt_gate_trained": float(head.vt_gate.detach()),
                     "rt_gate_trained": float(head.rt_gate.detach()),
                     "n_horizons": len(head.cfg.horizons)},
        "index_zero_means": HGoal.describe_index_zero(),
        "oracle": HGoal.oracle_availability(),
        "reference_option": a.reference,
        "train_frame": str(a._frames[1]),
        "options": {}, "degradation": {}, "early_stop": {},
    }

    vals, eids, sc_by = {}, {}, {}
    for opt in a.options.split(","):
        opt = opt.strip()
        if not opt:
            continue
        entry = {"meaning": HGoal.OPTION_MEANING.get(opt, "?")}
        try:
            v, eid, extras, sc = probe_once(
                opt, world, head, goal_head, episodes, device=a.device,
                gcfg=gcfg, frames_of=frames_of)
            vals[opt], eids[opt], sc_by[opt] = v, eid, sc
            entry.update({"gate_runs": True,
                          "probe_value": round(float(np.nanmean(v)), 6),
                          **extras})
            print(f"  [{opt:12s}] RUNS  primary={entry['probe_value']:.6f}  "
                  f"n={extras['n_windows_finite']}/{extras['n_windows_total']}",
                  flush=True)
        except Exception as exc:                              # noqa: BLE001
            entry.update({"gate_runs": False,
                          "error_type": type(exc).__name__,
                          "error": str(exc)[:400]})
            print(f"  [{opt:12s}] ⛔ DOES NOT RUN: "
                  f"{type(exc).__name__}: {str(exc)[:120]}", flush=True)
        res["options"][opt] = entry

    # ---- paired deltas vs the reference ----------------------------------- #
    ref = a.reference
    if ref in vals:
        for opt, v in vals.items():
            if opt == ref:
                continue
            if eids[opt] != eids[ref]:
                res["options"][opt]["paired_vs_reference"] = {
                    "error": "window sets differ — a paired test would be wrong"}
                continue
            res["options"][opt]["paired_vs_reference"] = paired(
                v, vals[ref], eids[ref], a.n_boot, a.seed)
            d = res["options"][opt]["paired_vs_reference"]
            print(f"  [{opt:12s}] vs {ref}: delta={d.get('delta')} "
                  f"[{d.get('lo')}, {d.get('hi')}] separated={d.get('separated')}",
                  flush=True)

    # ---- does the early-stop still FIRE under each option? ---------------- #
    for opt in list(vals):
        print(f"  [{opt:12s}] degradation + early-stop ...", flush=True)
        try:
            dv, deid, dextras, dsc = probe_once(
                opt, world, head, goal_head, episodes, device=a.device,
                gcfg=gcfg, degrade=a.degrade_m, degrade_mode=a.degrade_mode,
                degrade_seed=a.degrade_seed, frames_of=frames_of)
        except Exception as exc:                              # noqa: BLE001
            res["degradation"][opt] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        base_pt = float(np.nanmean(vals[opt]))
        deg_pt = float(np.nanmean(dv))
        pd = paired(dv, vals[opt], eids[opt], a.n_boot, a.seed)
        direction_ok = deg_pt < base_pt
        res["degradation"][opt] = {
            "mode": a.degrade_mode, "drift_m": a.degrade_m,
            "baseline_primary": round(base_pt, 6),
            "degraded_primary": round(deg_pt, 6),
            "paired_delta_degraded_minus_baseline": pd,
            "DIRECTION_OK_composite_went_DOWN": direction_ok,
            "along_track_m_mean_baseline":
                res["options"][opt]["longitudinal"]["along_track_end_m_mean"],
            "along_track_m_mean_degraded":
                dextras["longitudinal"]["along_track_end_m_mean"],
            "ego_progress_baseline":
                res["options"][opt]["longitudinal"]["ego_progress_mean"],
            "ego_progress_degraded": dextras["longitudinal"]["ego_progress_mean"],
            "_slow_plan_artefact_check": (
                "if along_track collapsed AND recovery_n_finite fell, the "
                "composite move is the NaN artefact, not a defect"),
            "recovery_mean_baseline": res["options"][opt]["lateral"]["recovery_mean"],
            "recovery_mean_degraded": dextras["lateral"]["recovery_mean"],
            "recovery_n_finite_baseline":
                res["options"][opt]["lateral"]["recovery_n_finite"],
            "recovery_n_finite_degraded": dextras["lateral"]["recovery_n_finite"],
        }
        if a.degrade_selfcheck and not direction_ok:
            res["degradation"][opt]["⛔ SELFCHECK_FAILED"] = (
                "the composite went UP under a degradation — this reproduces the "
                "sibling stream's +0.1698 NaN artefact and the number below is "
                "NOT a measure of the gate's sensitivity")

        # ---- drive the SHIPPED stop rule -------------------------------- #
        g = HeldoutGate(HeldoutGateConfig(
            every=1, episodes=a.episodes, stride=a.stride, horizon=a.horizon,
            batch=a.batch, n_boot=a.n_boot, seed=a.seed, patience=2, amp=a.amp,
            frame=a._frames[1]))
        hist = []
        try:
            hist.append(g.observe(0, vals[opt], eids[opt]))     # incumbent
            hist.append(g.observe(1, dv, deid))                 # challenger 1
            hist.append(g.observe(2, dv, deid))                 # challenger 2
            res["early_stop"][opt] = {
                "STOPS_the_degraded_arm": bool(g.stop),
                "worse_streak": int(g.worse_streak),
                "stop_reason": g.stop_reason,
                "separated_worse_per_probe": [h.get("separated_worse")
                                              for h in hist],
                "_rule": "incumbent at probe 0, the degraded arm at probes 1 and "
                         "2; patience=2, so a working gate must reach stop=True",
            }
            print(f"  [{opt:12s}] early-stop fires: {g.stop} "
                  f"(streak {g.worse_streak})", flush=True)
        except Exception as exc:                              # noqa: BLE001
            res["early_stop"][opt] = {"error": f"{type(exc).__name__}: {exc}"}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=str),
                           encoding="utf-8")
    print(f"[out] {a.out}", flush=True)
    return res


def load_val_episodes(a, cfg):
    """-> (episodes, frames_of). Mirrors ``train()``'s two data paths.

    ⭐ The gate is handed ``val_eps[:n]`` — a FIXED prefix of the val split, the
    same episodes every probe. That is what makes the paired bootstrap valid, and
    it is reproduced here verbatim."""
    if a.v2_val_cache:
        from tanitad.data.v2_dataset import build_v2_providers
        # ⚠️ frames come from build_stack's SINGLE resolve_v2_frames call —
        # resolve_v2_frames applies the frame to cfg, so calling it twice would
        # re-apply it.
        cache_frame, frame = a._frames
        slice_frame = None if frame == cache_frame else frame
        eps = build_v2_providers(a.v2_val_cache, lru_size=a.v2_lru,
                                 frame=slice_frame, verbose=True)
        if not eps:
            raise SystemExit(f"no *.v2ep.pt under {a.v2_val_cache}")
        return eps, None
    from tanitad.data.mixing import load_episode
    eps = [load_episode(str(p), mmap=True)
           for p in sorted(Path(a.val_cache).glob("ep_*.pt"))]
    if not eps:
        raise SystemExit(f"no ep_*.pt under {a.val_cache}")
    return eps, None


if __name__ == "__main__":
    main()
