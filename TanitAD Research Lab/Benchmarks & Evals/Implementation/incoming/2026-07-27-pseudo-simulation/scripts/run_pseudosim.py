#!/usr/bin/env python3
"""Run the PSEUDO-SIMULATION protocol on the flagship-v4 30 k arm + two controls.

WHAT THIS PRODUCES
------------------
The first TanitAD closed-loop-class number that is a **MEASUREMENT**: every
evaluated state's deviation IS its grid point, so ``0 %`` of evaluations lie
outside the MEASURED envelope, asserted by
``taniteval.pseudosim.assert_grid_in_envelope`` **before any checkpoint is
loaded**.

THE THREE ARMS, AND WHY THESE THREE
-----------------------------------
``v4_oracle``  the gate arm. ``taniteval.clhorizon.V4Planner``, i.e. byte-for-byte
               the forward pass ``eval_flagship_v4.collect_planner`` runs, so a
               pseudo-simulation number cannot drift from the open-loop primary.
               Goal provenance ORACLE (stamped; an upper bound, not deployable).
``v4_blind``   ⭐ **THE DECISIVE CONTROL.** The identical checkpoint, identical
               oracle goal, on a DESTROYED observation (``dead_black``). The
               heading perturbation is visible ONLY in the image, so an arm that
               cannot see it cannot recover from it. **If the protocol does not
               separate ``v4_oracle`` from ``v4_blind``, it measures nothing** —
               and this run says so rather than reporting a score.
``cv_holdv0``  constant velocity, zero steering: the program's standing
               non-learned baseline. Map-free, deterministic, zero-parameter.

⚠️ TRAFFIC MODE is recorded in every emitted node: ``log_replay_nonreactive``.
Our AlpaSim ``trafficsim`` is disabled (``skip: true``) ⇒ literal replay, so
EVERY published TanitAD closed-loop number ran against non-reactive traffic.

⛔ Collision / TTC are NOT emitted: the val cache has no agent cuboids and the
cached ``episode_id`` collides across clips. A constant is not substituted.

Estimator: ``taniteval.ci.episode_cluster_bootstrap`` B=2000, unit = val
episode, PAIRED between arms on identical (anchor, grid point) rows.
``overlapping_holdout_se`` appears nowhere.

Host: the EVAL pod only. Writes to ``/workspace`` (``/root`` is 99 % full and
SILENTLY TRUNCATES). No training pod is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

for p in ("/root/v4eval/stack", "/root/v4eval/stack/scripts",
          "/workspace/_pseudosim"):
    if p not in sys.path:
        sys.path.insert(0, p)

import goal_modes                                                  # noqa: E402
from eval_flagship_v4 import load_v4_from_ck                       # noqa: E402
from tanitad.data.mixing import load_episode                       # noqa: E402

from taniteval import ci as _ci                                    # noqa: E402
from taniteval import clhorizon as CH                              # noqa: E402
from taniteval import pseudosim as PS                              # noqa: E402


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class CVPlanner:
    """Constant velocity, zero steering. The non-learned control."""
    name = "cv_holdv0"

    def __init__(self, horizon=20):
        self.horizon = int(horizon)

    def traj(self, fw, v0, goal):
        t = torch.arange(1, self.horizon + 1, dtype=torch.float32) * CH.DT
        x = v0.detach().float().cpu()[:, None] * t[None]
        return torch.stack([x, torch.zeros_like(x)], -1)


class BlindWrapper:
    """The same planner on a DESTROYED observation (``dead_black``).

    The ONLY difference from the sighted arm is the image: the checkpoint, the
    goal, the state input ``v0`` and the grid are identical. So a gap between
    them is attributable to *seeing the perturbation* and to nothing else."""

    def __init__(self, inner):
        self.inner = inner

    def traj(self, fw, v0, goal):
        return self.inner.traj(torch.zeros_like(fw), v0, goal)


def _paired(a, b, eid):
    """Paired episode-cluster bootstrap on rows that are finite in BOTH arms."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2 or len(set(np.asarray(eid)[m])) < 2:
        return None
    return _ci.paired_episode_cluster_bootstrap(a[m], b[m],
                                               list(np.asarray(eid)[m]),
                                               n_boot=2000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anchors-dense", required=True)
    ap.add_argument("--head-config", default=None)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--goal-mode", default="oracle")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--perwindow-out", default="")
    a = ap.parse_args()

    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    grid = PS.default_grid()

    # ⭐ THE ASSERTION, BEFORE ANY CHECKPOINT IS LOADED. A bad grid costs 0 GPU s.
    proof = PS.assert_grid_in_envelope(grid)
    print(f"[pseudosim] envelope proof: "
          f"frac_steps_any={proof['EXTRAPOLATION_frac_steps_any']} "
          f"frac_windows={proof['EXTRAPOLATION_frac_windows_any_step_out_of_envelope']} "
          f"verdict={proof['EXTRAPOLATION_VERDICT']!r}", flush=True)
    print(f"[pseudosim] grid {grid.n_points} points: "
          f"dyaw={list(grid.dyaw_deg)} dlon={list(grid.dlon_steps)} "
          f"dlat={list(grid.dlat_m)} (lateral REFUSED)", flush=True)

    eps_files = sorted(Path(a.val_dir).glob("ep_*.pt"))[:a.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in eps_files]
    Ts = [int(e.poses.shape[0]) for e in episodes]
    print(f"[pseudosim] {len(episodes)} val episodes, T in "
          f"[{min(Ts)},{max(Ts)}], dev {device}", flush=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    world, grounding, head, step, hcfg, goal_head = load_v4_from_ck(
        ck, device,
        head_config_path=(a.head_config or Path(a.ckpt).parent / "config.json"),
        anchors_dense_path=a.anchors_dense)
    del ck
    v4 = CH.V4Planner(world, head, goal_head, a.goal_mode, goal_modes)

    t0 = time.time()
    goals = CH._v4_goal_cache(episodes, stack_paths=("/root/v4eval/stack/scripts",))
    for i in range(len(episodes)):
        goals._build(i)
    print(f"[pseudosim] goal '{a.goal_mode}' minted for {goals.n_total} indices "
          f"in {time.time() - t0:.0f}s; refusals {goals.n_fail}", flush=True)

    arms = [("v4_oracle", v4, goals),
            ("v4_blind", BlindWrapper(v4), goals),
            ("cv_holdv0", CVPlanner(a.horizon), None)]

    res = {
        "_experiment": "PSEUDO-SIMULATION -- a bounded pre-generated perturbation "
                       "grid replaces sequential rollout",
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
        "_protocol": PS.PROTOCOL,
        "traffic_mode": PS.TRAFFIC_MODE_LOG_REPLAY,
        "traffic_mode_note": PS.TRAFFIC_MODE_NOTE,
        "envelope_proof": proof,
        "grid": grid.describe(),
        "ckpt": a.ckpt, "ckpt_md5": _md5(a.ckpt), "ckpt_step": int(step),
        "anchors_dense": a.anchors_dense, "anchors_md5": _md5(a.anchors_dense),
        "val_dir": a.val_dir, "n_episodes": len(episodes),
        "stride": a.stride, "horizon_steps": a.horizon,
        "horizon_s": round(a.horizon * CH.DT, 2),
        "goal_mode": a.goal_mode,
        "goal_provenance_warning": (
            "ORACLE goal: route / route_graded / vt_band are minted from the "
            "ego's own FUTURE poses. Every v4 number on this surface is an "
            "UPPER BOUND, not a deployable number." if a.goal_mode == "oracle"
            else None),
        "collision_and_ttc": {"emitted": False,
                              "reason": PS.COLLISION_UNAVAILABLE_REASON},
        "arms": {},
    }

    def bank():
        Path(a.out).write_text(json.dumps(res, indent=2, default=str),
                               encoding="utf-8")

    per_arm_scores, per_arm_pw, per_arm_comp = {}, {}, {}
    for name, planner, g in arms:
        t1 = time.time()
        pw = PS.pseudo_evaluate(planner, episodes, grid, device=device,
                                stride=a.stride, horizon=a.horizon,
                                goals=g, batch=a.batch, verbose=False)
        sc = PS.score_windows(pw)
        per_arm_pw[name] = pw
        per_arm_scores[name] = {k: sc[k] for k in
                                ("ego_progress", "recovery", "comfort")}
        per_arm_scores[name]["no_collision"] = None
        per_arm_scores[name]["ttc"] = None
        print(f"[{name}] {pw['planner_calls']} planner calls, "
              f"{pw['rollout_steps_executed']} rollout steps, "
              f"{time.time() - t1:.0f}s", flush=True)

    for name, _, _ in arms:
        node = PS.emit(per_arm_pw[name], arm=name, n_boot=2000,
                       by_arm_scores=per_arm_scores)
        per_arm_comp[name] = node.pop("_per_window_composite", None)
        pwd = node.pop("_per_window")
        res["arms"][name] = node
        c = node.get("composite", {})
        ci = (c.get("ci") or {}) if isinstance(c, dict) else {}
        print(f"[{name}] PSS={ci.get('mean')} [{ci.get('lo')},{ci.get('hi')}] "
              f"n_win={ci.get('n_windows')} n_ep={ci.get('n_episodes')}",
              flush=True)
        if a.perwindow_out:
            np.savez_compressed(
                str(Path(a.perwindow_out).with_suffix("")) + f"_{name}.npz",
                **{k: v for k, v in pwd.items()})
        bank()

    # -------- paired arm-vs-arm, on identical (anchor, grid point) rows ----- #
    eid = list(per_arm_pw["v4_oracle"]["eid"])
    pairs = [("v4_oracle", "v4_blind"), ("v4_oracle", "cv_holdv0"),
             ("v4_blind", "cv_holdv0")]
    res["paired"] = {}
    for x, y in pairs:
        blk = {"_estimator": "taniteval.ci.paired_episode_cluster_bootstrap "
                             "(B=2000, unit = val episode)",
               "_refused_estimator": "overlapping_holdout_se"}
        for comp in ("ego_progress", "recovery", "comfort"):
            blk[comp] = _paired(per_arm_scores[x][comp],
                                per_arm_scores[y][comp], eid)
        if per_arm_comp.get(x) is not None and per_arm_comp.get(y) is not None:
            blk["PSS_recovery_progress"] = _paired(per_arm_comp[x],
                                                   per_arm_comp[y], eid)
        res["paired"][f"{x}__minus__{y}"] = blk
        d = (blk.get("PSS_recovery_progress") or {})
        print(f"[paired] {x} - {y}: PSS delta={d.get('delta')} "
              f"[{d.get('lo')},{d.get('hi')}] separated={d.get('separated')}",
              flush=True)
        bank()

    # -------- the instrument's own validity gate --------------------------- #
    blk = res["paired"].get("v4_oracle__minus__v4_blind", {})
    sight = blk.get("PSS_recovery_progress") or {}
    res["INSTRUMENT_VALIDITY"] = {
        "question": "does the protocol separate an arm that CAN see the "
                    "perturbation from the IDENTICAL arm that cannot? Only the "
                    "image differs: same checkpoint, same oracle goal, same v0, "
                    "same grid, same windows.",
        "sighted_minus_blind_PSS": sight,
        "sighted_minus_blind_ego_progress": blk.get("ego_progress"),
        "sighted_minus_blind_recovery": blk.get("recovery"),
        "separated": sight.get("separated"),
        "reading": ("If this is NOT separated, the protocol has no sensitivity "
                    "to the perturbation it applies and no arm score from it is "
                    "admissible. That is the C13 gate on the instrument itself."),
        "_why_PSS_and_not_recovery_alone": (
            "`recovery` is UNDEFINED for a plan that does not move (its "
            "denominator is the plan's own along-track distance), so a blind "
            "arm that stalls is excluded from it rather than scored 1.0. The "
            "composite is therefore the correct gate quantity."),
    }
    res["_recovery_defined_fraction"] = {
        name: round(float(np.isfinite(per_arm_scores[name]["recovery"]).mean()), 6)
        for name, _, _ in arms}
    res["_ego_progress_mean"] = {
        name: round(float(np.nanmean(per_arm_scores[name]["ego_progress"])), 6)
        for name, _, _ in arms}
    bank()
    print(json.dumps(res["INSTRUMENT_VALIDITY"], indent=2, default=str))
    print(f"[pseudosim] wrote {a.out}")
    print("PSEUDOSIM_DONE", flush=True)


if __name__ == "__main__":
    main()
