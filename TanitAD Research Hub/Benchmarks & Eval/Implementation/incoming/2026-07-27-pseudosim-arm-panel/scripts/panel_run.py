#!/usr/bin/env python3
"""Score ONE arm on the PSEUDO-SIMULATION grid and bank its per-window dump.

WHY ONE ARM PER PROCESS
-----------------------
The 2026-07-27 driver (``run_pseudosim.py``) scored three arms in one process.
That is fine for three arms that share a checkpoint; it is wrong for a PANEL,
because a panel needs (a) arms whose checkpoints do not fit in memory together,
(b) incremental banking so a killed agent still yields value, and (c) an
arithmetic-only recombination path so the paired table can be rebuilt with NO
GPU. This script therefore does exactly one arm and writes the FULL
``pseudo_evaluate`` record (``traj``/``ref_path``/``ref_yaw``/keys), from which
``panel_combine.py`` re-derives every score by calling
``taniteval.pseudosim.score_windows`` / ``emit`` VERBATIM.

⚠️ NOTHING IN taniteval/pseudosim.py IS REIMPLEMENTED HERE. The grid, the
envelope assertion, the composite and the estimator are imported. This file adds
only PLANNER ADAPTERS: the ``planner.traj(fw, v0, goal) -> [b, H, 2]`` interface
for arms the original driver did not cover.

THE ADAPTERS, AND WHAT EACH ONE IS
----------------------------------
``v4``          ``taniteval.clhorizon.V4Planner`` — unchanged, the reproduction arm.
``flagship``    ⭐ the 4-brain STATE-ONLY plan step: ``encode_window`` ->
                ``strategic_policy(states, nav)['ctx']`` ->
                ``tactical_policy(states, ctx)['waypoints']`` -> densify.
                This is ``taniteval/closedloop.py:317-318`` verbatim, i.e. the
                deploy path. It takes **no future actions** — which is why v1 and
                the no-speed control ARE scoreable here, contrary to the claim
                that the only v1 plan step is ``rollout_decode(..., fa, ...)``.
                (That claim is true of ``taniteval/rollout.py``, whose own
                ``honest_metric_name`` is ``wm_fidelity_ade_2s``.)
``refc``        ``model(fw, nav_cmd, v0, steps)`` -> the argmax-confidence anchor
                trajectory read at WP_STEPS, densified. ``nav_mode`` is resolved
                by ``taniteval.refc_eval.resolve_nav`` — not reimplemented.
``cv``          constant velocity, zero steering (verbatim from run_pseudosim.py).
``blind``       wrapper: the SAME planner on a zeroed image. The heading
                perturbation is visible ONLY in the image.
``still``       ⚠️ THE ADVERSARY: a planner that does not move. Pre-registered
                gate G5 — ``recovery`` must be NaN for it, never 1.0.

DENSIFICATION. The flagship tactical head and REF-C both emit 4 waypoints at
5/10/15/20 steps. ``taniteval.closedloop.densify_plan`` piecewise-linearly
interpolates them onto the 20-step 10 Hz grid. **This cannot change either
admissible component**: both ``ego_progress`` and ``recovery`` read only index
-1, and interpolation through the knots reproduces the 20-step knot EXACTLY.
``--assert-densify-endpoint`` checks that identity numerically at run time.

Host: pod2 only. Writes to /workspace.
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

for p in ("/root/TanitAD/stack", "/root/TanitAD/stack/scripts", "/root/taniteval"):
    if p not in sys.path:
        sys.path.insert(0, p)

from taniteval import clhorizon as CH          # noqa: E402
from taniteval import pseudosim as PS          # noqa: E402
from taniteval.closedloop import densify_plan  # noqa: E402

WP_STEPS = (5, 10, 15, 20)


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# =========================================================================== #
# planner adapters                                                            #
# =========================================================================== #
class CVPlanner:
    """Constant velocity, zero steering. VERBATIM from run_pseudosim.py."""
    kind = "cv_holdv0"

    def __init__(self, horizon=20):
        self.horizon = int(horizon)

    def traj(self, fw, v0, goal):
        t = torch.arange(1, self.horizon + 1, dtype=torch.float32) * CH.DT
        x = v0.detach().float().cpu()[:, None] * t[None]
        return torch.stack([x, torch.zeros_like(x)], -1)


class StandStillPlanner:
    """⚠️ THE ADVERSARY (pre-registered gate G5). Emits the zero path.

    The FIRST version of `recovery` scored this shape ABOVE a sighted planner
    (+0.597) because a plan that barely moves has a small cross-track error. The
    progress-matched denominator made it NaN instead. This arm is run so the fix
    is re-verified ON THIS PANEL rather than inherited from a smoke test."""
    kind = "stand_still"

    def __init__(self, horizon=20):
        self.horizon = int(horizon)

    def traj(self, fw, v0, goal):
        b = fw.shape[0]
        return torch.zeros(b, self.horizon, 2, dtype=torch.float32)


class BlindWrapper:
    """The same planner on a DESTROYED observation. VERBATIM from run_pseudosim.py."""

    def __init__(self, inner):
        self.inner = inner

    def traj(self, fw, v0, goal):
        return self.inner.traj(torch.zeros_like(fw), v0, goal)


class FlagshipTacticalPlanner:
    """⭐ The 4-brain STATE-ONLY plan step — v1 / no-speed / any flagship trunk.

    ``taniteval/closedloop.py:317-318`` verbatim:

        ctx = model.strategic_policy(states, nav)["ctx"]
        wp  = model.tactical_policy(states, ctx)["waypoints"]

    No future actions, no expert control sequence, no oracle trajectory — the
    deploy path. ``nav`` is a NAV_COMMANDS index: 0 = follow (the historical
    constant), or the GT command from ``refb_labels.nav_command`` when
    ``goal`` carries one (the ORACLE control, an upper bound).
    """
    kind = "flagship_tactical"

    def __init__(self, model, horizon=20):
        self.model, self.horizon = model, int(horizon)
        assert getattr(model, "strategic_policy", None) is not None, \
            "no strategic_policy on this checkpoint"
        assert getattr(model, "tactical_policy", None) is not None, \
            "no tactical_policy on this checkpoint"
        # v2 lever 1 (ego -> planners) would need [v0, yr0]; v1 does not have it.
        self.ego_input = (model.strategic_policy.ego_emb is not None
                          or model.tactical_policy.ego_emb is not None)

    @torch.no_grad()
    def traj(self, fw, v0, goal):
        st = self.model.encode_window(fw)
        b = st.shape[0]
        if goal is not None and "nav" in goal:
            nav = goal["nav"].to(st.device).long()
        else:
            nav = torch.zeros(b, dtype=torch.long, device=st.device)
        ctx = self.model.strategic_policy(st, nav)["ctx"]
        wp = self.model.tactical_policy(st, ctx)["waypoints"]
        return densify_plan({int(k): wp[k] for k in wp}, self.horizon).cpu()


class RefCPlanner:
    """REF-C anchored-diffusion decode -> densified 10 Hz path.

    The nav command is resolved by ``taniteval.refc_eval.resolve_nav`` (not
    reimplemented). ``produced`` reads the model's OWN route head (image-only,
    no future); ``oracle`` uses the GT route and is an upper bound;
    ``follow_constant`` is the 07-21 C6 confound and is only run when asked.
    """
    kind = "refc"

    def __init__(self, model, nav_mode="produced", horizon=20, steps=None):
        from taniteval import refc_eval as RE
        self.RE = RE
        self.model, self.nav_mode, self.horizon = model, nav_mode, int(horizon)
        assert not getattr(model.cfg, "refc1", False), "refc1 ckpt is not time-waypoint comparable"
        assert tuple(model.cfg.trajectory.horizons) == WP_STEPS, \
            f"REF-C horizons {tuple(model.cfg.trajectory.horizons)} != {WP_STEPS}"
        self.steps = (model.cfg.decoder.diffusion_steps if steps is None
                      else int(steps))
        self.nav_note = None
        self.nav_hist = torch.zeros(4, dtype=torch.long)

    @torch.no_grad()
    def traj(self, fw, v0, goal):
        if self.nav_mode == "oracle":
            assert goal is not None and "nav" in goal, "oracle nav needs a goal cache"
            nav_cmd = goal["nav"].to(fw.device).long()
            self.nav_note = ("GT route from the ego's OWN FUTURE poses -- AN "
                             "ORACLE, upper bound only")
        else:
            nav_cmd, self.nav_note = self.RE.resolve_nav(
                self.model, fw, v0, self.steps, self.nav_mode)
        if nav_cmd is not None:
            self.nav_hist += torch.bincount(nav_cmd.cpu(), minlength=4)
        else:
            self.nav_hist[0] += fw.shape[0]
        out = self.model(fw, nav_cmd=nav_cmd, v0=v0, steps=self.steps)
        wp = {int(k): out["waypoints"][k] for k in WP_STEPS}
        return densify_plan(wp, self.horizon).cpu()


# =========================================================================== #
# goal caches                                                                 #
# =========================================================================== #
class NavGoalCache:
    """ORACLE nav command per (episode, index), from ``refb_labels.nav_command``.

    Same interface as ``clhorizon._v4_goal_cache``: ``get(ep_i, last_ix, device)``.
    ⚠️ future-derived => an ORACLE => upper bound only, stamped on every node."""

    def __init__(self, episodes):
        import refb_labels as rl
        self.rl, self.eps = rl, episodes
        self._c, self.n_total, self.n_fail = {}, 0, 0

    def _build(self, e_i):
        poses = torch.as_tensor(self.eps[e_i].poses, dtype=torch.float32)
        T = int(poses.shape[0])
        nav = np.zeros(T, dtype=np.int64)
        for t in range(T):
            self.n_total += 1
            try:
                c, ok = self.rl.nav_command(poses, t)
                nav[t] = int(c)
                if not ok:
                    self.n_fail += 1
            except Exception:
                self.n_fail += 1
        self._c[e_i] = nav
        return nav

    def get(self, e_i, last_ix, device):
        nav = self._c.get(e_i)
        if nav is None:
            nav = self._build(e_i)
        li = np.clip(np.asarray(last_ix, dtype=np.int64), 0, len(nav) - 1)
        return {"nav": torch.as_tensor(nav[li], device=device)}


# =========================================================================== #
# main                                                                        #
# =========================================================================== #
def _save_pw(pw, path):
    """Persist the FULL pseudo_evaluate record — the no-GPU recompute path."""
    np.savez_compressed(
        path,
        traj=pw["traj"].numpy().astype(np.float32),
        ref_path=pw["ref_path"].numpy().astype(np.float32),
        ref_yaw=pw["ref_yaw"].numpy().astype(np.float32),
        v0=pw["v0"].numpy().astype(np.float32),
        pt_dlat=pw["pt_dlat"].numpy().astype(np.float32),
        pt_dyaw=pw["pt_dyaw"].numpy().astype(np.float32),
        pt_dlon=pw["pt_dlon"].numpy().astype(np.float32),
        anchor=pw["anchor"].numpy().astype(np.int64),
        ep_i=pw["ep_i"].numpy().astype(np.int64),
        eid=np.asarray(pw["eid"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="result key, e.g. v1_tactical_oracle")
    ap.add_argument("--kind", required=True,
                    choices=["v4", "flagship", "refc", "cv", "still"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--head-config", default=None)
    ap.add_argument("--anchors-dense", default="")
    ap.add_argument("--refc-preset", default="base")
    ap.add_argument("--goal", default="none",
                    choices=["none", "v4_oracle", "nav_oracle", "nav_follow"])
    ap.add_argument("--refc-nav", default="produced",
                    choices=["produced", "oracle", "follow_constant"])
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--speed-input", type=int, default=1)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = PS.default_grid()

    # --- G3: THE ASSERTION, before any checkpoint is loaded ----------------- #
    proof = PS.assert_grid_in_envelope(grid)
    print(f"[panel:{a.arm}] envelope proof frac_steps_any="
          f"{proof['EXTRAPOLATION_frac_steps_any']} frac_windows="
          f"{proof['EXTRAPOLATION_frac_windows_any_step_out_of_envelope']} "
          f"verdict={proof['EXTRAPOLATION_VERDICT']!r}", flush=True)

    # --- G4: the DELIBERATELY FAILING input, exercised on THIS host --------- #
    from taniteval.ood import ENV_YAW_MAX
    g4 = {"edge_value_accepted": None, "just_outside_raises": None}
    try:
        PS.assert_grid_in_envelope(PS.GridSpec(dyaw_deg=(ENV_YAW_MAX,)))
        g4["edge_value_accepted"] = True
    except PS.EnvelopeViolation:
        g4["edge_value_accepted"] = False
    try:
        PS.assert_grid_in_envelope(PS.GridSpec(dyaw_deg=(ENV_YAW_MAX * 1.0001,)))
        g4["just_outside_raises"] = False
    except PS.EnvelopeViolation:
        g4["just_outside_raises"] = True
    g4["G4_PASS"] = bool(g4["edge_value_accepted"] and g4["just_outside_raises"])
    print(f"[panel:{a.arm}] G4 falsifier: {g4}", flush=True)
    # and the lateral refusal, also exercised here
    try:
        PS.GridSpec(dlat_m=(1.0,))
        g4["lateral_refused"] = False
    except PS.LateralAxisRefused:
        g4["lateral_refused"] = True

    from tanitad.data.mixing import load_episode
    eps_files = sorted(Path(a.val_dir).glob("ep_*.pt"))[:a.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in eps_files]
    Ts = [int(e.poses.shape[0]) for e in episodes]
    print(f"[panel:{a.arm}] {len(episodes)} val episodes, T in "
          f"[{min(Ts)},{max(Ts)}], dev {device}", flush=True)

    meta = {"arm": a.arm, "kind": a.kind, "goal": a.goal, "blind": bool(a.blind),
            "envelope_proof": proof, "G4_falsifier_exercised": g4,
            "grid": grid.describe(), "stride": a.stride, "horizon": a.horizon,
            "n_episodes": len(episodes), "val_dir": a.val_dir,
            "traffic_mode": PS.TRAFFIC_MODE_LOG_REPLAY,
            "traffic_mode_note": PS.TRAFFIC_MODE_NOTE,
            "protocol": PS.PROTOCOL, "host": "pod2",
            "python": sys.version.split()[0], "torch": torch.__version__}

    goals = None
    t0 = time.time()
    # ---------------------------------------------------------------- arms -- #
    if a.kind == "cv":
        planner = CVPlanner(a.horizon)
    elif a.kind == "still":
        planner = StandStillPlanner(a.horizon)
    elif a.kind == "v4":
        import goal_modes
        from eval_flagship_v4 import load_v4_from_ck
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        world, grounding, head, step, hcfg, goal_head = load_v4_from_ck(
            ck, device,
            head_config_path=(a.head_config or Path(a.ckpt).parent / "config.json"),
            anchors_dense_path=a.anchors_dense)
        del ck
        planner = CH.V4Planner(world, head, goal_head, "oracle", goal_modes)
        meta.update(ckpt=a.ckpt, ckpt_md5=_md5(a.ckpt), ckpt_step=int(step),
                    anchors_dense=a.anchors_dense, anchors_md5=_md5(a.anchors_dense))
        goals = CH._v4_goal_cache(episodes,
                                  stack_paths=("/root/TanitAD/stack/scripts",))
        for i in range(len(episodes)):
            goals._build(i)
        meta["goal_provenance"] = "ORACLE (route/route_graded/vt_band from the ego's OWN FUTURE)"
        meta["goal_labeler_refusals"] = int(goals.n_fail)
    elif a.kind == "flagship":
        from taniteval.loaders import load as load_arm
        L = load_arm({"arch": "flagship-worldmodel", "ckpt": a.ckpt,
                      "speed_input": bool(a.speed_input)}, device)
        planner = FlagshipTacticalPlanner(L["model"], a.horizon)
        meta.update(ckpt=a.ckpt, ckpt_md5=_md5(a.ckpt), ckpt_step=L.get("step"),
                    speed_input=bool(a.speed_input),
                    ego_input_on_planners=bool(planner.ego_input),
                    plan_surface=("strategic_policy -> tactical_policy -> "
                                  "waypoints (STATE-ONLY; no future actions)"))
    elif a.kind == "refc":
        from taniteval.loaders import load as load_arm
        L = load_arm({"arch": "refc", "ckpt": a.ckpt,
                      "config_preset": a.refc_preset}, device)
        planner = RefCPlanner(L["model"], a.refc_nav, a.horizon)
        meta.update(ckpt=a.ckpt, ckpt_md5=_md5(a.ckpt), ckpt_step=L.get("step"),
                    refc_preset=a.refc_preset, refc_nav_mode=a.refc_nav,
                    refc_denoise_steps=planner.steps,
                    refc_n_anchors=int(L["model"].cfg.anchors.n_anchors))
    print(f"[panel:{a.arm}] planner ready in {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------- goals ---- #
    if a.goal == "nav_oracle":
        goals = NavGoalCache(episodes)
        for i in range(len(episodes)):
            goals._build(i)
        meta["goal_provenance"] = ("ORACLE nav command (refb_labels.nav_command "
                                   "over the ego's OWN FUTURE heading)")
        meta["goal_labeler_unjudgeable"] = int(goals.n_fail)
    elif a.goal == "nav_follow":
        meta["goal_provenance"] = ("CONSTANT `follow` (nav_cmd = 0) -- the "
                                   "deployable/historical path, and the 07-21 "
                                   "C6 shape: the route input is NOT exercised")
    elif a.goal == "none":
        meta.setdefault("goal_provenance", "none (arm takes no goal)")

    if a.blind:
        planner = BlindWrapper(planner)
        meta["blind_note"] = ("image ZEROED; checkpoint, goal, v0 and grid are "
                              "IDENTICAL to the sighted arm. The heading "
                              "perturbation is visible ONLY in the image.")

    # ---- densify endpoint identity (the adapter cannot move the metric) ---- #
    if a.kind in ("flagship", "refc"):
        probe = {5: torch.tensor([[1.0, 0.1]]), 10: torch.tensor([[2.0, 0.3]]),
                 15: torch.tensor([[3.0, 0.6]]), 20: torch.tensor([[4.0, 1.0]])}
        d = densify_plan(probe, a.horizon)
        err = float((d[:, -1] - probe[20]).abs().max())
        meta["densify_endpoint_max_err"] = err
        assert err == 0.0, f"densify moved the 2 s endpoint by {err}"

    # -------------------------------------------------------------- run ----- #
    t1 = time.time()
    pw = PS.pseudo_evaluate(planner, episodes, grid, device=device,
                            stride=a.stride, horizon=a.horizon, goals=goals,
                            batch=a.batch, verbose=False)
    meta["planner_calls"] = int(pw["planner_calls"])
    meta["rollout_steps_executed"] = int(pw["rollout_steps_executed"])
    meta["wallclock_s"] = round(time.time() - t1, 1)
    meta["n_rows"] = int(pw["traj"].shape[0])
    meta["traj_shape"] = list(pw["traj"].shape)
    if a.kind == "refc":
        inner = planner.inner if a.blind else planner
        meta["refc_fed_command_hist"] = {n: int(c) for n, c in zip(
            ("follow", "left", "right", "straight"), inner.nav_hist.tolist()) if c}
        meta["refc_nav_note"] = inner.nav_note

    _save_pw(pw, str(out_dir / f"pw_{a.arm}.npz"))
    # single-arm node (ranges here are SINGLE-ARM; the panel recomputes them
    # with by_arm in panel_combine.py -- that is the adjudicating form)
    node = PS.emit(pw, arm=a.arm, n_boot=2000)
    node.pop("_per_window", None)
    node.pop("_per_window_composite", None)
    node["_meta"] = meta
    (out_dir / f"arm_{a.arm}.json").write_text(
        json.dumps(node, indent=2, default=str), encoding="utf-8")
    c = node.get("composite", {})
    ci = (c.get("ci") or {}) if isinstance(c, dict) else {}
    print(f"[panel:{a.arm}] PSS={ci.get('mean')} [{ci.get('lo')},{ci.get('hi')}] "
          f"n_win={ci.get('n_windows')} n_ep={ci.get('n_episodes')} "
          f"{meta['wallclock_s']}s", flush=True)
    print("PANEL_ARM_DONE " + a.arm, flush=True)


if __name__ == "__main__":
    main()
