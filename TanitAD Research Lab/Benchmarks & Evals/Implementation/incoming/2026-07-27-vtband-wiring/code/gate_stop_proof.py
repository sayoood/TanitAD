#!/usr/bin/env python3
"""⭐ DOES THE WIRED EARLY-STOP STILL FIRE? — driven through the SHIPPED gate.

WHAT MAKES THIS DIFFERENT FROM THE DECISION STREAM'S PROBE
-----------------------------------------------------------
``vtband_probe.py`` measured the OPTIONS through its own
``StatesAwareSurfacePlanner`` harness, because at that time nothing in the
shipped path could express them. This driver measures the **WIRING**: every
probe goes through :meth:`tanitad.train.heldout_gate.HeldoutGate.probe` with
**no ``goal_kwargs_fn`` argument at all** — exactly the shape
``train_flagship_v4``'s call site uses — so the gate must build its own goal from
``HeldoutGateConfig.goal_option`` or fail.

⇒ if this driver produces a stop, the code a real v5 run executes produces a stop.

THE DEGRADATION, AND WHY IT IS NOT ASSUMED
-------------------------------------------
The head's ranked pick is replaced by a **uniform random candidate from its own
fan** (``head.select`` is monkeypatched, so the degradation lives in the MODEL
and the whole planner path stays shipped code). Chosen because it is the failure
the gate exists to catch — ``heldout_gate``'s docstring: *"Selection is the thing
that regressed on v4."*

⚠️ **TWO EARLIER DEGRADATIONS IN THIS PROGRAM MOVED THE WRONG WAY**, both for
structural reasons, so direction is MEASURED here and never assumed:

* a constant-sign **lateral drift** made an arm BETTER (recovery 0.029 -> 0.236):
  a one-sided drift *re-centres* a systematically-biased planner;
* a **slowdown** raised the composite **+0.1698**: a barely-moving plan has
  ``s_along -> 0``, so ``xt_hold -> 0`` on the gate's ``dlat = 0`` grid and
  ``recovery`` is **NaN by construction**.

``--selfcheck`` therefore REFUSES to report a stop unless all three hold:

1. the composite went DOWN;
2. the paired episode-cluster CI on (degraded - clean) is SEPARATED and negative;
3. the mechanism is not the NaN artefact — ``recovery``'s finite count must not
   collapse, and along-track travel must not have shrunk toward zero.

⛔ Never ``overlapping_holdout_se``. The interval is the paired episode-cluster
bootstrap, unit = held-out episode.

⛔ Launches nothing, trains nothing, writes no checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _wire(stack: Path, code: Path):
    for p in (str(stack), str(stack / "scripts"), str(stack.parent / "taniteval"),
              str(code)):
        if p not in sys.path:
            sys.path.insert(0, p)


def patch_select_to_random(head, seed: int):
    """-> restore(). Replace the head's RANKED pick with a uniform fan draw.

    Patching the MODEL rather than wrapping the planner is deliberate: it keeps
    ``DeployableSurfacePlanner`` and ``HeldoutGate.probe`` byte-identical to what
    a run executes, so a stop measured here is not a property of this driver."""
    orig = head.select
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def patched(fan, refined_logits, vt_speed, vt_keep, v0_):
        traj, idx, score, tele = orig(fan, refined_logits, vt_speed, vt_keep, v0_)
        n = fan.shape[1]
        ridx = torch.randint(0, n, (fan.shape[0],), generator=gen).to(fan.device)
        rtraj = fan[torch.arange(fan.shape[0], device=fan.device), ridx]
        return rtraj, ridx, score, {**tele, "DEGRADED_random_selection": True}

    head.select = patched

    def restore():
        head.select = orig
    return restore


def _m(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return None if a.size == 0 else round(float(a.mean()), 6)


def mechanism(world, head, episodes, gcfg, device, goal_head):
    """The STRUCTURAL half of the direction check — lateral/longitudinal, so a
    'worse' composite can be attributed rather than believed."""
    from taniteval import pseudosim as ps
    from tanitad.train import heldout_goal as HGoal
    from tanitad.train.heldout_gate import DeployableSurfacePlanner

    fn = HGoal.make_goal_kwargs_fn(gcfg.goal_option, head.cfg, goal_head=goal_head)
    planner = DeployableSurfacePlanner(world, head, device=device, amp=gcfg.amp,
                                       goal_kwargs_fn=fn,
                                       goal_option=gcfg.goal_option)
    pw = ps.pseudo_evaluate(planner, episodes, gcfg.resolved_grid(), device=device,
                            stride=gcfg.stride, horizon=gcfg.horizon,
                            batch=gcfg.batch, verbose=False, frame=gcfg.frame)
    sc = ps.score_windows(pw, progress_term=gcfg.progress_term)
    return {
        "LON_along_track_end_m_mean": _m(sc["along_track_end_m"]),
        "LON_ego_progress_mean": _m(sc["ego_progress"]),
        "LON_ego_progress_n_finite": int(np.isfinite(sc["ego_progress"]).sum()),
        "LAT_cross_track_end_m_mean": _m(sc["cross_track_end_m"]),
        "LAT_recovery_mean": _m(sc["recovery"]),
        "LAT_recovery_n_finite": int(np.isfinite(sc["recovery"]).sum()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stack", default="/workspace/v5gate/stack")
    ap.add_argument("--vtband-code", default="/workspace/vtband/code",
                    help="dir holding vtband_probe.py (build_stack/load_val_episodes "
                         "are reused so the head is built EXACTLY as train() does)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anchors-dense", default="")
    ap.add_argument("--val-cache", default="")
    ap.add_argument("--v2-val-cache", default="")
    ap.add_argument("--v2-subframe", default=None)
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--require-parity", action="store_true")
    ap.add_argument("--frame-h", type=int, default=None)
    ap.add_argument("--frame-w", type=int, default=None)
    ap.add_argument("--frame-hfov", type=float, default=None)
    ap.add_argument("--projection", default=None)
    ap.add_argument("--goal", default=None,
                    help="--heldout-goal equivalent; default = the shipped "
                         "GOAL_OPTION_DEFAULT, i.e. what a v5 run would use")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--degrade-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.from_scratch = False
    a.options = ""

    _wire(Path(a.stack), Path(a.vtband_code))
    import vtband_probe as VP
    from tanitad.train.heldout_gate import (GOAL_OPTION_DEFAULT,
                                            GOAL_OPTION_PROVENANCE, PRIMARY_NAME,
                                            HeldoutGate, HeldoutGateConfig)

    goal = a.goal or GOAL_OPTION_DEFAULT
    world, head, goal_head, cfg, ck = VP.build_stack(a)
    gcfg = HeldoutGateConfig(every=1, episodes=a.episodes, stride=a.stride,
                             horizon=a.horizon, batch=a.batch, n_boot=a.n_boot,
                             seed=a.seed, amp=a.amp, patience=a.patience,
                             frame=a._frames[1], goal_option=goal)
    episodes, _ = VP.load_val_episodes(a, cfg)
    episodes = episodes[:a.episodes]

    gate = HeldoutGate(gcfg)
    res = {
        "_what": "does the WIRED mid-run early-stop still FIRE? — every probe "
                 "driven through the SHIPPED HeldoutGate.probe with NO "
                 "goal_kwargs_fn argument, i.e. the trainer's own call shape",
        "primary": PRIMARY_NAME,
        "estimator": f"paired episode-cluster bootstrap (B={a.n_boot}, unit = "
                     f"held-out episode); NEVER overlapping_holdout_se",
        "goal_option": goal,
        "goal_option_provenance": GOAL_OPTION_PROVENANCE,
        "ckpt": a.ckpt, "ckpt_step": ck.get("step"),
        "head_cfg": {"cond_vtarget": head.cfg.cond_vtarget,
                     "cond_route": head.cfg.cond_route,
                     "goal_dropout": head.cfg.goal_dropout,
                     "sel_gate_trained": float(head.sel_gate.detach()),
                     "vt_gate_trained": float(head.vt_gate.detach()),
                     "rt_gate_trained": float(head.rt_gate.detach())},
        "train_frame": str(a._frames[1]),
        "n_episodes_probed": len(episodes),
        "patience": a.patience,
        "degradation": {
            "mode": "random_selection",
            "what": "head.select's RANKED pick replaced by a uniform random "
                    "candidate from the SAME fan — the v4 regression's own "
                    "failure mode; the candidate set is unchanged",
            "injected_at_probes": [1, 2],
            "seed": a.degrade_seed,
        },
        "probes": [], "direction_check": {}, "verdict": {},
    }

    # ---- probe 0: the CLEAN incumbent ------------------------------------- #
    print(f"[probe 0] clean incumbent, goal={goal!r} ...", flush=True)
    r0 = gate.probe(0, world, head, episodes, device=a.device, goal_head=goal_head)
    res["probes"].append(_slim(r0))
    print(f"  primary={r0['primary_value']} n={r0['n_windows']}/"
          f"{r0['n_episodes']} eps", flush=True)
    mech_clean = mechanism(world, head, episodes, gcfg, a.device, goal_head)

    # ---- probes 1..patience: the DEGRADED arm ----------------------------- #
    restore = patch_select_to_random(head, a.degrade_seed)
    try:
        mech_deg = mechanism(world, head, episodes, gcfg, a.device, goal_head)
        for i in range(1, a.patience + 1):
            print(f"[probe {i}] DEGRADED (random selection) ...", flush=True)
            r = gate.probe(i, world, head, episodes, device=a.device,
                           goal_head=goal_head)
            res["probes"].append(_slim(r))
            p = r.get("paired") or {}
            print(f"  primary={r['primary_value']} delta={p.get('delta')} "
                  f"[{p.get('lo')}, {p.get('hi')}] sep={p.get('separated')} "
                  f"streak={r.get('worse_streak')} stop={r.get('stop')}",
                  flush=True)
    finally:
        restore()

    # ---- THE DIRECTION CHECK, before any claim ---------------------------- #
    last = res["probes"][-1]
    paired = last.get("paired") or {}
    base_pt, deg_pt = res["probes"][0]["primary_value"], last["primary_value"]
    went_down = deg_pt < base_pt
    sep_neg = bool(paired.get("separated") and (paired.get("delta") or 0) < 0)
    rec_kept = (mech_deg["LAT_recovery_n_finite"]
                >= 0.9 * mech_clean["LAT_recovery_n_finite"])
    not_slowplan = (mech_deg["LON_along_track_end_m_mean"]
                    >= 0.5 * mech_clean["LON_along_track_end_m_mean"])
    res["direction_check"] = {
        "clean": mech_clean, "degraded": mech_deg,
        "composite_went_DOWN": bool(went_down),
        "paired_separated_and_negative": sep_neg,
        "recovery_finite_count_preserved": bool(rec_kept),
        "not_the_slow_plan_NaN_artefact": bool(not_slowplan),
        "_why": ("two earlier degradations in this program moved the WRONG way "
                 "for structural reasons (a constant-sign lateral drift "
                 "RE-CENTRES a one-sidedly-biased planner; a slowdown NaNs "
                 "`recovery` and RAISES the composite). Direction is measured, "
                 "never assumed."),
    }
    ok = went_down and sep_neg and rec_kept and not_slowplan
    res["direction_check"]["DIRECTION_OK"] = bool(ok)

    res["verdict"] = {
        "STOPS": bool(gate.stop),
        "worse_streak": int(gate.worse_streak),
        "stop_reason": gate.stop_reason,
        "DIRECTION_OK": bool(ok),
        "ADMISSIBLE": bool(gate.stop and ok),
        "_read": ("ADMISSIBLE requires BOTH: the gate stopped, AND the "
                  "degradation is verified to have made the arm worse. A stop "
                  "on a degradation whose direction was not checked proves "
                  "nothing about the gate."),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=str),
                           encoding="utf-8")
    print(json.dumps(res["verdict"], indent=2), flush=True)
    print(f"[out] {a.out}", flush=True)
    return 0 if res["verdict"]["ADMISSIBLE"] else 3


def _slim(rec):
    """The probe record minus the per-window arrays (which carry no clip ids but
    are large and are not the evidence)."""
    out = {k: v for k, v in rec.items() if k != "pseudosim"}
    ps = rec.get("pseudosim") or {}
    out["pseudosim"] = {k: ps.get(k) for k in
                        ("goal_option", "goal_option_provenance", "grid",
                         "components_admitted", "estimator", "warp", "surface",
                         "planner_calls", "envelope_proof")}
    return out


if __name__ == "__main__":
    sys.exit(main())
