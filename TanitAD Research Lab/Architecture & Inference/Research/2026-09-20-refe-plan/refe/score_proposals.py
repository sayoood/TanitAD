"""Stage 2b -- per-proposal PDM targets for REFe's scorer, using THEIR released calculators.

THIS ESTIMATE HAS BEEN WRONG TWICE AND THIS FILE IS THE THIRD, VERIFIED ANSWER.
  v1  "their feature builder already emits most of the ScenarioData arrays" -- from a NAME GREP
      that reported 14/14. A name in a file is not an array produced. WRONG.
  v2  "only 3 of 14 required fields come from the map builder; budget it as real work" -- from
      running `_empty_map_arrays`. That measured the MAP helper, which was the wrong object.
  v3  MEASURED, and what this file is built on: `DriveRLNuPlanFeatureBuilder.build(current_input,
      initialization) -> ScenarioData` (feature_builder.py:79, constructing at :226) already
      returns a COMPLETE ScenarioData from a nuPlan planner input. There is nothing to re-derive.

WHAT THE CALCULATORS ACTUALLY NEED. `BaseRewardCalculator.forward(scenario_data, log_scenario_data,
rewards_and_infos)` reads agent geometry OUT OF the ScenarioData it is given
(`get_recent_agent_polygons()`), so it scores whatever state that object holds -- it does NOT take a
candidate trajectory argument. To score one of REFe's 64 proposals we therefore APPEND that
proposal's poses to the ego track and re-run the calculators.

  agent_positions_all is [N, A, T, 2] and AGENT 0 IS THE EGO (scenario_data.py:670,
  `ego_pos = self.agent_positions_all[:, 0, -1, :]`).
  The polygons are CACHED, so any mutation must be followed by `clean_polygon_cache()` or the
  calculators silently score the previous state. That cache is the single most likely way to get a
  plausible-looking but wrong target bank, which is why `score_proposal` always clears it.

SCOPE / HONESTY
  * this produces the six components per proposal; REFe's scorer is trained with BCE against them.
  * it does NOT simulate other agents reacting to the proposal -- the calculators score the
    proposal against the recorded scene, which is what a per-proposal PDM target is.
  * ⛔ it is NOT yet wired into training. `train.py` still uses a placeholder scorer target, and the
    closed-loop consequence (arbitrary proposal selection) stands until it is.

Usage: python score_proposals.py --selftest
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# the six components REFe's scorer predicts, in a fixed order that the training target must match
# ⛔ THE CANDIDATE'S OWN SAMPLE SPACING, AND IT IS NOT THE ENGINE'S FRAME INTERVAL.
# The first version of the velocity derivation divided by `frame_time_interval` (0.1 s), because
# that is the engine's simulation step. The CANDIDATE's poses are not on that grid: the trajectory
# bank is 20 samples at STRIDE 2 over a 10 Hz history, i.e. 0.2 s apart, which is also the policy's
# own 5 Hz query cadence. Dividing a 0.2 s displacement by 0.1 s makes every derived velocity
# EXACTLY 2x too large -- MEASURED before the fix: the teacher's injected horizon ran 5.77 -> 19.50
# m/s with a mean of 12.393 on a scenario whose real speed is ~13 m/s and whose final speed is not
# 19.5. Nothing errors; comfort, jerk and time-to-collision simply score a car going twice as fast.
# ⭐ Same family as the STRIDE bug that halved the target horizon: a rate that is correct for one
# object quoted for another. It MUST match the cadence `build_targets.py` banks at; if that changes,
# this changes with it.
TRAJ_DT = 0.2

# The shortest prefix that has a well-defined jerk. Two differences are needed for acceleration and
# three poses for jerk, so anything below 3 is a boundary artefact rather than a measurement.
MIN_PREFIX = 3

# ⛔ COMFORT NEEDS A LONGER PREFIX THAN EVERYTHING ELSE, AND MIN_PREFIX=3 ONLY MOVED THE ARTEFACT.
# `Comfort` switches estimator at a history length of 15 -- i.e. k >= 10 on our grid -- and below
# that it is reading a boundary, not the candidate. MEASURED by the third review: banked with
# MIN_PREFIX=3 the four lateral candidates read 0.5000, IDENTICAL to the deliberate violator;
# restricted to the valid prefixes they read 0.8333, identical to the teacher, and only `jerky`
# separates at 0.6111. The earlier fix removed the k=1 artefact and left the k=3..9 one.
KEY_MIN_PREFIX = {"comfort": 10}

# ⛔ A LATCH ARMED ON AN INADMISSIBLE PREFIX DEFEATS THE MIN-PREFIX FILTER ENTIRELY.
# Filtering comfort's VALUES to k >= 10 is not enough: its repeat-offender latch persists through
# `carry`, so a violation recorded at k=3 -- a prefix the code itself declares inadmissible --
# is still latched when k reaches 10 and drags the banked value down. MEASURED: 4 of 11 candidates,
# the entire lateral family, had comfort pulled 1.0000 -> 0.7500, exactly the calculator's 0.75
# penalty. ⇒ a calculator's state must not START accumulating before its values are admissible.
LATCH_OWNER = {"_comfort_triggered": "comfort",
               "_nuplan_ttc_triggered": "ttc", "_nuplan_ttc_collided_matrix": "ttc"}


def _min_prefix_for(key: str) -> int:
    head = key.split(".", 1)[0]
    return KEY_MIN_PREFIX.get(head, MIN_PREFIX)

# Leaves that ACCUMULATE over the horizon and must therefore be read at its end rather than
# minimised over prefixes.
CUMULATIVE_KEYS = ("progress.advance_m",)

# Leaves that exist but must NEVER be read as a signal -- see the comment at the harvest.
NON_AUTHORITATIVE = frozenset({"collision.NuPlanCollision.reward", "ttc.NuPlanTTC.info"})

# How many rows of `agent_positions_all` history span one TRAJ_DT. The feature builder resamples
# the 10 Hz simulation history to `history_sample_interval = 0.2 s` over `history_steps = 5`, so
# one row IS one TRAJ_DT and the answer is 1 -- not the 2 you get by assuming the history kept the
# simulator's 10 Hz. Verified from DriveRLNuPlanFeatureBuilderConfig.
HISTORY_STEP_ROWS = 1

COMPONENTS = ("off_road", "collision", "ttc", "comfort", "goal_reaching", "center_line")


def build_calculators(engine_config):
    """Instantiate the six released reward calculators."""
    from driverl.env.engine.reward_calculator.off_road import OffRoad
    from driverl.env.engine.reward_calculator.collision.nuplan_collision import NuPlanCollision
    from driverl.env.engine.reward_calculator.nuplan_ttc import NuPlanTTC
    from driverl.env.engine.reward_calculator.comfort import Comfort
    from driverl.env.engine.reward_calculator.goal_reaching import GoalReaching
    from driverl.env.engine.reward_calculator.center_line import CenterLine
    return {
        "off_road": OffRoad(engine_config),
        "collision": NuPlanCollision(engine_config),
        "ttc": NuPlanTTC(engine_config),
        "comfort": Comfort(engine_config),
        "goal_reaching": GoalReaching(engine_config),
        "center_line": CenterLine(engine_config),
    }


def enrich_lane_graph(sd, map_api, anchor, route_roadblock_ids,
                      max_center_segments: int = 1024, max_boundary_segments: int = 1024):
    """Fill the lane-GRAPH arrays the nuPlan feature path never produces.

    MEASURED: `feature_builder._build_map_features` returns only 8 keys and none of them is
    `lanes_centers_groups` / `_ids` / `_next_groups`. The devkit's lower-level
    `driverl_runtime_map_features._build_map_arrays` DOES compute them, and it is callable offline:
    on a real frame it returns 1,019 nonzero groups, 1,024 ids and 4,095 next-group links over 96
    centre segments.

    ⛔ THE LAST SENTENCE OF THIS DOCSTRING USED TO CLAIM THAT WITHOUT THEM THE CALCULATORS SEE AN
    EMPTY GRAPH AND RETURN NO-EVENT DEFAULTS -- "which is exactly the inert scorer". THAT IS
    REFUTED. MEASURED twice: `scorer_gate.py --no-enrich` produces output BYTE-IDENTICAL to the
    enriched run, on all 28 signals. The lane graph is NOT load-bearing for these six calculators,
    and the inert scorer had three other causes (a reader collapsing nested dicts, endpoint-only
    scoring, and a control that never left the road).
    ⇒ This function is kept because the arrays are genuinely absent from a builder-produced
    `ScenarioData` and a future consumer may need them -- at a MEASURED cost of 28 ms, 0.2 % of a
    frame. It is not kept because it fixed anything. Do not cite it as a fix.
    """
    from nuplan.planning.script import driverl_runtime_map_features as M
    arrays, stats = M._build_map_arrays(
        map_api=map_api, anchor=anchor, total_frames=1,
        sampled_frames=[anchor], map_query_frames=[anchor], tl_by_token={},
        route_roadblock_ids=list(route_roadblock_ids),
        radius_m=M.DEFAULT_MAP_RADIUS_M,
        max_center_segments=max_center_segments, max_boundary_segments=max_boundary_segments)
    import numpy as np
    for field, key in (("lanes_centers_groups", "lanes_centers_groups"),
                       ("lanes_centers_ids", "lanes_centers_ids"),
                       ("lanes_centers_next_groups", "lanes_centers_next_groups")):
        v = arrays.get(key)
        if v is None:
            continue
        t = torch.as_tensor(np.asarray(v))
        if t.dim() >= 1:
            t = t.unsqueeze(0)                     # add the batch dim ScenarioData expects
        setattr(sd, field, t)
    return sd, stats


def inject_proposal(sd, traj_xy: torch.Tensor, yaw: torch.Tensor | None = None,
                    log_sd=None, horizon: int | None = None):
    """Append a candidate ego trajectory to a COPY of `sd` and invalidate the polygon cache.

    ⛔ FRAME. `traj_xy` is [T, 2] in the **EGO-CENTRIC** frame, NOT world coordinates.
    MEASURED by building a ScenarioData offline from a simulation log: the ego (agent 0) last
    position reads **[0.0, 0.0]** while the true world pose at that step is
    [664433.28, 3998238.98]. Feeding world coordinates here would place the candidate ~660 km from
    the scene; every calculator would return a confident, meaningless number rather than an error.
    ⭐ This happens to be exactly the frame `build_targets.py` already stores its trajectories in,
    so the target bank and the scorer agree without conversion -- but state the frame, never assume
    it. A correct formula in the wrong frame reads exactly like an answer.
    """
    out = copy.deepcopy(sd)
    T = traj_xy.shape[0]
    pos = out.agent_positions_all                       # [N, A, T0, 2]
    N, A, _, _ = pos.shape
    add = pos[:, :, -1:, :].repeat(1, 1, T, 1)          # hold every NPC at its last pose
    add[:, 0, :, :] = traj_xy.to(pos.dtype).to(pos.device)   # agent 0 = ego <- the proposal
    out.agent_positions_all = torch.cat([pos, add], dim=2)
    if yaw is not None and getattr(out, "agent_orientation_all", None) is not None:
        o = out.agent_orientation_all
        addo = o[:, :, -1:].repeat(1, 1, T)
        addo[:, 0, :] = yaw.to(o.dtype).to(o.device)
        out.agent_orientation_all = torch.cat([o, addo], dim=2)
    for name in ("npc_mask_all", "agent_size_all", "agent_type_all", "agent_velocity_all"):
        t = getattr(out, name, None)
        if t is None or t.numel() == 0 or t.dim() < 3:
            continue
        setattr(out, name, torch.cat([t, t[:, :, -1:].repeat(
            *( [1, 1, T] + [1] * (t.dim() - 3) ))], dim=2))
    # ⛔ AND THE BACKGROUND MUST FOLLOW THE LOG, NOT FREEZE.
    # The loop above holds every NPC at its last pose for the whole horizon. MEASURED by the third
    # review: the ego moves 6.340 m while all 127 NPCs move EXACTLY 0.000000 m. The paper's own
    # protocol is "rolled out with ALL BACKGROUND ACTORS FOLLOWING THE DRIVING LOG", and a frozen
    # background MANUFACTURES collisions -- a candidate that merely goes where a car has already
    # left still hits it. Splicing the log's real futures removes up to 16.7 pp of collisions and
    # adds none, selectively on the lateral and slow candidates.
    # ⭐ The futures are already in hand: `log_sd.agent_positions_all` carries (1, 128, 25, 2).
    if log_sd is not None and log_sd is not sd:
        _splice_npc_futures(out, log_sd, horizon if horizon else T, k=T)
    # ⛔ AND THE EGO'S VELOCITY MUST FOLLOW THE CANDIDATE, OR COMFORT CANNOT RANK ANYTHING.
    # The loop above HOLDS every appended step at the last recorded value, which is right for the
    # NPCs -- they do not react -- and wrong for the ego, whose whole trajectory we just replaced.
    # Consequence, MEASURED by the 2026-09-20 conformance review: `comfort` was identical across
    # every candidate of every frame (it varies in 0.0 % of 620 frames), because the comfort
    # calculator reads accelerations and jerk out of `agent_velocity_all` and every candidate was
    # handed the SAME velocities. A sixth of the scoring head was therefore supervised on a
    # constant, which cannot move the argmax.
    # ⇒ derive the ego's FULL kinematics from the poses we injected -- see
    #   _derive_ego_kinematics, which also covers acceleration, jerk and yaw rate,
    #   because Comfort does not read velocity for its verdict.
    _derive_ego_kinematics(out, T)
    # MANDATORY: the polygons are cached. Without this the calculators score the PREVIOUS state and
    # return a completely plausible, completely wrong target.
    if hasattr(out, "clean_polygon_cache"):
        out.clean_polygon_cache()
    # `agent_nearest_indices` is DERIVED, not built by the nuPlan feature builder, and the collision
    # and TTC calculators index it as [:, :, -1, :] -- which is why they raised
    # "IndexError: too many indices for tensor of dimension 1" on a builder-produced object.
    # ScenarioData exposes the derivation publicly, so recompute it for the injected state.
    if hasattr(out, "update_nearest_neighbors"):
        try:
            out.update_nearest_neighbors()
        except Exception:
            pass
    return out


def _splice_npc_futures(out, log_sd, T: int, k: int | None = None) -> None:
    """Give the NPCs the futures the driving log recorded, instead of freezing them.

    ⛔ TWO DEFECTS IN THE FIRST VERSION, BOTH MEASURED, AND TOGETHER THEY WERE WORSE THAN THE
    FROZEN BACKGROUND THEY REPLACED.

    1. IT WROTE POSITIONS AND MASKS BUT NOT SIZES. An agent that appears in the log's FUTURE but
       not in the 5-row history became `valid=True`, at a real position, carrying a **zero-area
       polygon** -- and DriveRL's intersection test fires on a degenerate box. Three-arm control on
       the TEACHER'S OWN REALISED PATH, which cannot collide: no splice -> collision 0.0000;
       splice without sizes -> **1.0000** (hit polygon extent 0.00 x 0.00 m at 8.83 m); splice with
       sizes -> 0.0000. Corpus signature: the expert's own path recorded a collision in 36.6 % of
       frames, and `stopped` was the ONLY candidate clean on both collision and TTC -- i.e. two of
       six PDM components had turned into a stop-detector.
    2. IT INDEXED BY TAIL INSTEAD OF BY PREFIX. `-n:` always takes the END of the log's horizon, so
       when scoring a k-step prefix the NPCs sat k steps AHEAD of the ego -- MEASURED +3.40 s at
       k=3. Only the full-horizon prefix was ever aligned, so 9 of 10 evaluations at the default
       stride compared the candidate against a future that had not happened yet.
    ⇒ `k` is the prefix length being scored; the log's future is read from ITS START, not its end.
    """
    lp = getattr(log_sd, "agent_positions_all", None)
    op = getattr(out, "agent_positions_all", None)
    if not (torch.is_tensor(lp) and torch.is_tensor(op)) or lp.dim() < 4 or op.dim() < 4:
        return
    n = T if k is None else min(k, T)
    n = min(n, op.shape[2])
    if n <= 0 or lp.shape[1] < 2 or op.shape[1] < 2:
        return
    a = min(lp.shape[1], op.shape[1])
    # the log's future begins where its history ends; take the FIRST n of it, not the last n
    lstart = lp.shape[2] - T
    if lstart < 0:
        lstart, n = 0, min(n, lp.shape[2])
    src = slice(lstart, lstart + n)
    op[:, 1:a, -n:, :] = lp[:, 1:a, src, :].to(op.dtype)
    # ⛔ SIZES AND TYPES TRAVEL WITH THE POSITIONS. Without them a spliced agent is a point.
    for name in ("agent_orientation_all", "agent_velocity_all", "npc_mask_all",
                 "agent_size_all", "agent_type_all"):
        lt, ot = getattr(log_sd, name, None), getattr(out, name, None)
        if (torch.is_tensor(lt) and torch.is_tensor(ot) and lt.dim() == ot.dim()
                and lt.dim() >= 3 and lt.shape[2] >= lstart + n and ot.shape[2] >= n):
            try:
                ot[:, 1:a, -n:] = lt[:, 1:a, src].to(ot.dtype)
            except Exception:
                pass
    # any agent still carrying a degenerate box must not be treated as present
    sz = getattr(out, "agent_size_all", None)
    mk = getattr(out, "npc_mask_all", None)
    if torch.is_tensor(sz) and torch.is_tensor(mk) and sz.dim() >= 4 and mk.dim() >= 3:
        try:
            degenerate = (sz[:, 1:a, -n:, 0] <= 1e-6) | (sz[:, 1:a, -n:, 1] <= 1e-6)
            mk[:, 1:a, -n:] = mk[:, 1:a, -n:] & (~degenerate)
        except Exception:
            pass
    if hasattr(out, "clean_polygon_cache"):
        out.clean_polygon_cache()


def _ext_time(t, T):
    """Extend a [N, A, T0, ...] tensor to T0+T by repeating its last step."""
    if t is None or not torch.is_tensor(t) or t.numel() == 0 or t.dim() < 3 or t.shape[2] == 0:
        return t
    return torch.cat([t, t[:, :, -1:].repeat(*([1, 1, T] + [1] * (t.dim() - 3)))], dim=2)


def _derive_ego_kinematics(out, T: int) -> None:
    """Give the ego the DYNAMICS of the candidate it was just handed.

    ⛔ WHY THE FIRST VERSION OF THIS WAS NOT ENOUGH. `inject_proposal` appends poses and then held
    every other per-step tensor at its last recorded value. That is right for the NPCs, which do not
    react, and wrong for the ego, whose entire trajectory was just replaced. Deriving the VELOCITY
    alone still left `comfort` constant across every candidate -- MEASURED 1.0000 for all ten,
    including a deliberately jerky sawtooth -- because `Comfort.data_preprocessing` does not read
    velocity for its verdict. It reads:
        agent_acceleration_state_all · agent_jerk_lat_all · agent_jerk_long_all
        agent_steering_state_all     · agent_yaw_rate_all · agent_velocity_all
    and five of those six were still the ego's last LOGGED values, identical for every candidate.
    ⭐ The lesson is the one this file keeps relearning: find what the consumer READS before
    deciding what to write. Two rounds were spent on velocity because it was the obvious tensor.
    ⚠️ Longitudinal and lateral are taken in the EGO'S OWN HEADING at each step, not in the world
    frame, because that is what the comfort thresholds are defined against.
    """
    pos = out.agent_positions_all
    if pos is None or pos.dim() < 4 or pos.shape[2] < T + 1:
        return
    dt = TRAJ_DT
    seg = pos[:, 0, -(T + 1):, :]                                  # [N, T+1, 2]
    vel = (seg[:, 1:, :] - seg[:, :-1, :]) / dt                    # [N, T, 2]
    # ⛔ SEED THE DERIVATIVE FROM THE EGO'S REAL LAST VELOCITY -- DO NOT ZERO IT.
    # The first version set acc[0] = 0 and then took jerk[1] = (acc[1] - acc[0]) / dt, which turns
    # an ordinary acceleration a into a SPURIOUS jerk of a/dt = 5a. Against their 4.13 m/s^3
    # longitudinal-jerk threshold that manufactures a comfort violation for EVERY candidate, and it
    # did: comfort read exactly 0.5000 for all ten, INCLUDING a stationary ego, which cannot be
    # uncomfortable. The round, identical value across unlike candidates is what gave it away.
    # ⭐ This is precisely the failure their own comment warns about -- "the first frame after
    # init_steps is the first simulated frame, so dropping the anchor would make its jerk/rates
    # spuriously zero" -- with the sign reversed: dropping the anchor made ours spuriously HUGE.
    # ⛔ INDEX RELATIVE TO THE ANCHOR, NOT TO THE END OF THE TENSOR.
    # My previous attempt read `pos[:, 0, -1]` and `pos[:, 0, -3]` to build the seed -- but by that
    # point `agent_positions_all` HAS ALREADY BEEN EXTENDED with the candidate, so those indices
    # are the last and third-from-last CANDIDATE poses, not history. The seed became the
    # candidate's own end-of-horizon speed (~19.5 m/s for the teacher), and the first acceleration
    # read **-81.32 m/s^2** while `stopped` -- whose candidate poses are all zero -- came out
    # PERFECTLY COMFORTABLE at 0.00. ⭐ The inversion is what exposed it: a car that stops dead
    # cannot be the comfortable one, and a metric that says so is measuring the wrong thing.
    # The anchor is at -(T+1); one TRAJ_DT earlier is `back` history steps before that.
    # ⛔ ONE HISTORY STEP IS ALREADY 0.2 s -- I ASSUMED 10 Hz AND IT IS 5 Hz.
    # `DriveRLNuPlanFeatureBuilderConfig.history_sample_interval` is **0.2**, verified from the
    # dataclass, with `history_steps = 5`. The SIMULATION history is 10 Hz; the FEATURE BUILDER
    # resamples it to 0.2 s. Stepping back 2 rows therefore spans 0.4 s while dividing by 0.2 s,
    # which inflates the seed and left the teacher's first acceleration at -10.43 m/s^2 against an
    # interior of +1.42. ⭐ Same family as the STRIDE bug and the frame-interval bug earlier today:
    # a rate that is correct for one object quoted for another. Read the rate from the object that
    # owns it instead of inferring it.
    back = HISTORY_STEP_ROWS
    if pos.shape[2] >= T + 1 + back:
        anchor = pos[:, 0, -(T + 1), :]
        earlier = pos[:, 0, -(T + 1 + back), :]
        v_prev = ((anchor - earlier) / TRAJ_DT).unsqueeze(1)
    else:
        v_prev = vel[:, :1, :]
    vel_ext = torch.cat([v_prev.reshape(vel.shape[0], 1, 2).to(vel.dtype), vel], dim=1)
    acc = (vel_ext[:, 1:, :] - vel_ext[:, :-1, :]) / dt            # [N, T], genuine at index 0
    acc_ext = torch.cat([acc[:, :1, :], acc], dim=1)
    jrk = (acc_ext[:, 1:, :] - acc_ext[:, :-1, :]) / dt            # jerk[0] = 0 by construction

    ori = getattr(out, "agent_orientation_all", None)
    if ori is not None and ori.dim() >= 3 and ori.shape[2] >= T:
        yaw = ori[:, 0, -T:]
    else:
        yaw = torch.atan2(vel[..., 1], vel[..., 0])
    cs, sn = torch.cos(yaw), torch.sin(yaw)
    a_long = acc[..., 0] * cs + acc[..., 1] * sn
    a_lat = -acc[..., 0] * sn + acc[..., 1] * cs
    j_long = jrk[..., 0] * cs + jrk[..., 1] * sn
    j_lat = -jrk[..., 0] * sn + jrk[..., 1] * cs
    # ⛔ SEED THE YAW RATE TOO. Acceleration is seeded from the anchor, and yaw was NOT -- its first
    # sample was forced to zero, planting exactly the kind of seam the acceleration seed exists to
    # remove. MEASURED by the third review: excluding that one seam, yaw-rate correlation against
    # the log rises 0.925 -> 0.987 and the maximum error falls 4.9x. Inconsistency between two
    # derivatives of the same trajectory is not a small thing; it is one of them being wrong.
    dyaw = torch.zeros_like(yaw)
    if ori is not None and ori.dim() >= 3 and ori.shape[2] >= T + 1:
        y_prev = ori[:, 0, -(T + 1)]
        dyaw[:, 0] = yaw[:, 0] - y_prev
    dyaw[:, 1:] = yaw[:, 1:] - yaw[:, :-1]
    dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi              # wrap before dividing
    yaw_rate = dyaw / dt

    target_len = pos.shape[2]          # positions were already extended to T0 + T

    def put(name, value, vec=False):
        t = getattr(out, name, None)
        if t is None or not torch.is_tensor(t) or t.numel() == 0 or t.dim() < 3:
            return
        # ⛔ EXTEND BY T, DO NOT TRUNCATE TO T. MEASURED shapes: positions and velocity arrive
        # [N, 128, 5, 2] and the acceleration / jerk / yaw-rate tensors [N, 128, 5], i.e. only the
        # 5 HISTORY steps -- `inject_proposal`'s loop never touched them. Growing them to exactly T
        # and writing all T would DESTROY the history, and Comfort explicitly keeps the final
        # logged frame as its DERIVATIVE ANCHOR ("dropping the anchor would make its jerk/rates
        # spuriously zero"). Grow to the SAME length the positions already have and write only the
        # appended tail, so the anchor survives and the derived values sit where the candidate is.
        if t.shape[2] < target_len:
            t = _ext_time(t, target_len - t.shape[2])
        v = value.to(t.dtype)
        try:
            if vec and t.dim() >= 4:
                t[:, 0, -T:, :v.shape[-1]] = v
            elif not vec and t.dim() == 3:
                t[:, 0, -T:] = v
            else:
                return
        except Exception:
            return
        setattr(out, name, t)

    put("agent_velocity_all", vel, vec=True)
    put("agent_acceleration_state_all", a_long)
    put("agent_jerk_long_all", j_long)
    put("agent_jerk_lat_all", j_lat)
    put("agent_yaw_rate_all", yaw_rate)
    # ⛔ AND EXTEND THE ONE TENSOR THAT HAS NO DERIVED VALUE, OR IT STAYS SHORT.
    # `Comfort.data_preprocessing` reads SIX tensors. Five now carry derived values; the sixth,
    # `agent_steering_state_all`, has none to derive -- a candidate is a path, not a steering
    # command -- and `inject_proposal`'s hold-loop never listed it. So it remained 5 rows long
    # while every other array grew to 25, and the comfort score is the MINIMUM over all its
    # component gates: a single mismatched gate pins the result to `comfort_weight` (0.5) no matter
    # how good the rest are. ⭐ That is exactly the symptom -- teacher and a dead-stop both 0.5000
    # while every quantity I could measure said one was comfortable and the other was not.
    st = getattr(out, "agent_steering_state_all", None)
    if torch.is_tensor(st) and st.dim() >= 3 and st.shape[2] < target_len:
        out.agent_steering_state_all = _ext_time(st, target_len - st.shape[2])


def _ego_scalar(v):
    """Pull the EGO's value out of whatever a calculator wrote.

    ⛔ The calculators emit PER-AGENT tensors of shape [N, A] over all 128 agents, and AGENT 0 IS
    THE EGO. Averaging across agents answers "how are all 128 doing", which is not the question and
    is dominated by NPCs that the candidate trajectory cannot affect -- so it would look almost
    identical for a good and a terrible proposal. MEASURED: my first version did exactly that.
    """
    # MEASURED: each calculator writes a NESTED dict, e.g.
    #   shared["OffRoad"]   = {"reward": tensor[N,A], "info": ..., "occ_hit": ...}
    #   shared["CrossLane"] = {"reward": ..., "lane_change_info": ..., ...}
    # so the scalar lives one level deeper than the calculator's own key. My first version read
    # info["reward"] at the TOP level (nothing there) and my second passed the dict straight to a
    # tensor path (NaN). Both printed as "nan", which looks exactly like a failing calculator.
    if isinstance(v, dict):
        v = v.get("reward", next(iter(v.values()), None))
    if v is None:
        return float("nan")
    if not torch.is_tensor(v):
        return float(v) if isinstance(v, (int, float)) else float("nan")
    f = v.float()
    if f.dim() >= 2:
        e = f[0, 0]                    # batch 0, agent 0 = ego
    elif f.dim() == 1 and f.numel() > 0:
        e = f[0]
    else:
        e = f
    # MEASURED: some calculators write a per-agent VECTOR (e.g. [N, A, 2]), so f[0, 0] is itself a
    # tensor and float() raises "only one element tensors can be converted". Reduce to its norm --
    # which is the right scalar for an offset vector -- rather than silently taking component 0.
    e = e.reshape(-1)
    if e.numel() == 0:
        return float("nan")
    return float(e[0]) if e.numel() == 1 else float(torch.linalg.norm(e))


def _leaves(v, prefix: str):
    """Every leaf of a calculator's nested dict, not just its `reward`.

    ⛔ THIS IS WHY THE SCORER READ AS INERT FOR AN ENTIRE AFTERNOON. Each calculator writes
    `{"reward": tensor, "info": tensor, ...}`. The `reward` field is the WEIGHTED, CLAMPED training
    signal and it is flat across candidates by design -- a 25 m veer and the teacher's own path both
    land on the same saturated value. The `info` field is the raw event/offset channel and it is
    where the discrimination lives: MEASURED on one frame, `CenterLine.info` reads 0.0691 for the
    teacher path and 0.3211 for the veer, while `CenterLine.reward` reads 0.7500 for both.
    ⭐ So "0 of 9 signals differ" was a statement about MY READOUT, not about their calculators.
    Same root-cause class as the two earlier `_ego_scalar` bugs in this file -- reading the wrong
    level of a nested dict -- and it is the third time it has bitten here. Emit every leaf.
    """
    if isinstance(v, dict):
        for k, sub in v.items():
            yield from _leaves(sub, f"{prefix}.{k}")
    else:
        yield prefix, v


# ORDER IS LOAD-BEARING, not cosmetic. MEASURED: CenterLine raises
#   "CenterLine expects OffRoad to run before it; ensure reward calculator order includes OffRoad
#    first."
# so the calculators SHARE one rewards_and_infos dict and must run in this sequence.
CALC_ORDER = ("off_road", "collision", "ttc", "comfort", "goal_reaching", "center_line")


def route_arc_length(sd, step: int = -1) -> float:
    """Arc length along the ROUTE POLYLINE -- the only curve long enough to measure progress on.

    ⛔ THE FIRST TWO VERSIONS BOTH USED `lanes_centers_points`, AND THAT ARRAY CANNOT EXPRESS
    PROGRESS. Each of its entries is a TWO-POINT SEGMENT, so an "arc length" along one of them
    saturates at that segment's own length. MEASURED by the fix-verification review: the total
    available arc was **9.75 m** and the advance capped at **1.068 m** while the candidate actually
    travelled **24.78 m** -- roughly 4 % of real progress, and every candidate beyond a metre or so
    looked identical. Picking the nearest segment at each step (version 1) made it worse still by
    letting the origin jump; pinning one segment (version 2) fixed the jumping and kept the cap.
    ⭐ `route_points_array` is a **100-point polyline** of the route the teacher was given. That is
    the curve PDM's ego-progress term is defined against, it is long enough to contain the whole
    horizon, and it is the same route the goal augmentation varies -- so progress, the navigation
    command and the teacher trajectory all refer to one object, which is what the paper requires.
    """
    from driverl.env.engine.reward_calculator.center_line import CenterLine
    rp = getattr(sd, "route_points_array", None)
    if rp is None or not torch.is_tensor(rp) or rp.numel() == 0 or rp.shape[-2] < 2:
        return float("nan")
    route = rp[..., :2].reshape(rp.shape[0], 1, -1, 2)          # [N, 1, P, 2] ONE long baseline
    mask = torch.ones(route.shape[0], 1, dtype=torch.bool, device=route.device)
    # ⭐ EGO ONLY under the ego view (2026-09-24): only [0, 0] is read below, and the progress is a
    # per-agent projection reduced over route SEGMENTS, never over agents. MEASURED 2.69 ms/call x
    # 2 calls per prefix = 13.4 % of a frame with all 128 slots; REFE_SCORER_EGO_VIEW=0 keeps them.
    pos = (sd.agent_positions_all[:, _EGO_AGENT:_EGO_AGENT + 1, step, :] if EGO_VIEW
           else sd.agent_positions_all[:, :, step, :])
    try:
        _idx, arc = CenterLine.calculate_baseline_progress(pos, route, mask)
    except Exception:
        return float("nan")
    return float(arc.reshape(arc.shape[0], -1)[0, 0])


# ⛔ THE DIRECTION-VIOLATION DETECTOR IS STATEFUL ACROSS CALLS, AND A PER-CANDIDATE RIG BREAKS IT.
# `CenterLine.compute_driving_direction_violation` stores `_direction_progress_last`,
# `_direction_baseline_last` and `_direction_progress_buffer` ON THE ScenarioData, and accumulates
# backward progress over a 1.0 s window. On a FIRST call `last_progress` is seeded to the current
# progress, so the delta is ZERO BY CONSTRUCTION and nothing can fire. Our rollout deep-copies a
# fresh object for every prefix, so every call was a first call -- which is why `OffRoad.info` took
# only {0, 1} across the entire bank and driving-direction compliance carried no supervision.
# ⭐ It is not that their detector never fires on our data; it is that an offline rig which starts
# from a clean object each step can never let it accumulate. Carrying these three attributes from
# one prefix to the next reproduces the engine loop's continuity without simulating anything.
# ⛔ ALL SEVEN, NOT THREE. Enumerated from source across the six calculators: carrying only the
# three direction attributes left FOUR latches un-threaded, and two of them belong to components we
# actually consume -- `_nuplan_ttc_triggered` / `_nuplan_ttc_collided_matrix` (time-to-collision)
# and `_comfort_triggered` (comfort's repeat-offender latch). A latch that is reset on every prefix
# can never latch, so those components were being scored as if each step were the first.
CARRY_STATE = ("_direction_progress_last", "_direction_baseline_last",
               "_direction_progress_buffer", "_cross_lane_deadband_counter",
               "_comfort_triggered", "_nuplan_ttc_triggered", "_nuplan_ttc_collided_matrix")

# ⭐ EGO VIEW -- A DATA-PREP SPEED LEVER (2026-09-24), EXACT BY CONSTRUCTION AND A/B-VERIFIED.
# MEASURED on a real navtrain frame (dev box, one core): of ~71 ms per scored prefix, CenterLine
# took 42.7 ms and OffRoad 17.8 ms -- 85 % -- because both build dense [N, A, S] point-to-segment
# tensors over ALL 128 agent slots x ~1,024 map segments (28 slots were valid, ONE was controlled),
# while this scorer reads only agent 0 (`_ego_scalar`). Everything they compute for agents 1..127
# is discarded.
# WHY IT IS EXACT, from source (driverl reward_calculator, read 2026-09-24):
#   * both calculators are PER-AGENT: every reduction runs over segments / polygon points / time /
#     top-k, never over the agent axis (`_narrow_road_right_preferance` reduces a
#     [num_controlled, K, M-1] tensor, and is disabled in the teacher config anyway);
#   * the only cross-calculator reads are CenterLine reading OffRoad's and CrossLane's entries,
#     elementwise -- and the two run TOGETHER on the view; collision / ttc / comfort /
#     goal_reaching read nothing another calculator wrote, and keep the FULL object (they are the
#     pairwise ones, and they need every NPC);
#   * neither reads `log_scenario_data`.
# The view SLICES only fields enumerated below as agent-axis, SHARES the map fields, and returns
# None -- the caller then runs the full object -- on any non-empty field it cannot classify:
# `lanes_centers_lines` is [N, 8, 128, 2], so a shape rule would have sliced a map tensor.
# REFE_SCORER_EGO_VIEW=0 restores the full-object path; `diag_scorer_ego_view.py` proves the two
# write byte-identical rows and that a wrong-agent mutation goes RED.
EGO_VIEW = os.environ.get("REFE_SCORER_EGO_VIEW", "1") != "0"
_EGO_AGENT = 1 if os.environ.get("REFE_SCORER_EGO_VIEW_MUTATE") == "agent1" else 0   # diag only
EGO_VIEW_CALCS = frozenset({"off_road", "center_line"})
EGO_VIEW_CARRY = frozenset({"_direction_progress_last", "_direction_baseline_last",
                            "_direction_progress_buffer", "_cross_lane_deadband_counter"})
_EGO_VIEW_AGENT_AXIS = frozenset({
    "npc_id_array", "agent_positions_all", "agent_velocity_all", "agent_size_all",
    "agent_orientation_all", "agent_type_all", "npc_mask_all",
    "agent_acceleration_state_all", "agent_acceleration_control_all",
    "agent_steering_state_all", "agent_steering_control_all", "agent_jerk_lat_all",
    "agent_jerk_long_all", "agent_yaw_rate_all", "agent_goal_reached_all", "goal_stage",
    "agent_not_blind_mask", "goal_positions", "frame_drop_mask_all"})
_EGO_VIEW_SHARED = frozenset({
    "lanes_points", "lanes_points_mask", "lanes_centers_points", "lanes_centers_mask",
    "route_points_array", "route_points_mask", "occupancy_grid", "lanes_centers_groups",
    "lanes_centers_ids", "lanes_centers_next_groups", "lanes_centers_lines",
    "lanes_centers_lines_mask", "lanes_centers_tl_states", "lanes_centers_tl_masks",
    "lanes_tl_states", "lanes_tl_masks", "route_lane_connector_polygon_vertices",
    "route_lane_connector_polygon_offsets", "route_lane_connector_bboxes",
    "route_lane_connector_tl_status_codes", "route_lane_connector_ids",
    "route_lane_connector_batch_indices", "route_lane_connector_world_offsets",
    "route_lane_connector_data_available", "occ_surface_points_all"})
_EGO_VIEW_RESET = ("_polygons_now", "_polygons_last", "_valid_mask")
# AGENT-INDEX / agent-x-agent fields: slicing their first agent axis would leave indices and a
# second axis pointing at agents the view no longer holds. Neither view calculator reads them
# (grep of driverl, 2026-09-24: `agent_nearest_indices` is read only by NuPlanCollision and
# NuPlanTTC -- which keep the full object -- the other three only by the policy agent), so the
# view carries them EMPTY: a future read would fail loudly instead of indexing the wrong agent.
# ⛔ MEASURED on the first A/B: `inject_proposal` fills `agent_nearest_indices` on every prefix, so
# without this line the view fell back on 1,980/1,980 prefixes and the A/B tested nothing.
_EGO_VIEW_EMPTY = frozenset({"agent_nearest_indices", "agent_visible_mask", "debug_weight",
                             "advantage_filtering_mask"})
EGO_VIEW_STATS = {"view": 0, "fallback": 0, "fallback_field": None}


def _ego_view(cand):
    """A SHALLOW copy of `cand` holding only agent 0, or None when a field cannot be classified.

    Slices are VIEWS (no copy) and the map tensors are the same objects as `cand`'s; neither
    calculator writes into an input tensor in place (every `clamp_` / masked assignment in them
    acts on a tensor they created), and the A/B diag compares every banked row byte for byte.
    """
    import dataclasses
    a = cand.agent_positions_all.shape[1]
    sl = slice(_EGO_AGENT, _EGO_AGENT + 1)
    v = copy.copy(cand)
    for f in dataclasses.fields(cand):
        t = getattr(cand, f.name, None)
        if not torch.is_tensor(t) or t.numel() == 0 or f.name in _EGO_VIEW_RESET:
            continue
        if f.name in _EGO_VIEW_AGENT_AXIS:
            if t.dim() < 2 or t.shape[1] != a:
                EGO_VIEW_STATS["fallback"] += 1
                EGO_VIEW_STATS["fallback_field"] = f.name
                return None
            setattr(v, f.name, t[:, sl])
        elif f.name in _EGO_VIEW_EMPTY:
            setattr(v, f.name, t.new_empty(0))
        elif f.name in _EGO_VIEW_SHARED or f.name in CARRY_STATE:
            continue
        else:                                   # a non-empty field nobody classified: never guess
            EGO_VIEW_STATS["fallback"] += 1
            EGO_VIEW_STATS["fallback_field"] = f.name
            return None
    for name in _EGO_VIEW_RESET:                # the polygon cache must be the VIEW's own
        setattr(v, name, None)
    m = cand._agent_control_manager
    if m is not None:
        m2 = copy.copy(m)
        m2._control_types = m._control_types[:, sl]
        m2.max_agents = 1
        v._agent_control_manager = m2
    r = cand._randomized_features
    if r is not None:
        r2 = copy.copy(r)
        r2.values = {k: (x[:, sl] if torch.is_tensor(x) and x.dim() >= 2 and x.shape[1] == a else x)
                     for k, x in r.values.items()}
        v._randomized_features = r2
    EGO_VIEW_STATS["view"] += 1
    return v


def score_proposal(sd, log_sd, calculators, traj_xy, yaw=None, carry: dict | None = None,
                   horizon: int | None = None) -> dict:
    """Run the six calculators over one candidate and return their written scalars.

    ⛔ TWO THINGS THAT LOOK LIKE FAILURES BUT ARE MY BUGS, both MEASURED 2026-09-20:
      1. the calculators do NOT write a key called "reward". OffRoad writes `OffRoad` and
         `CrossLane`. Reading info["reward"] returned None for every calculator and printed as
         "n/a" -- which looks exactly like "the calculator failed" and is not.
      2. they share ONE dict and have a required ORDER (CenterLine asserts OffRoad ran first), so a
         fresh dict per calculator breaks the chain.
    """
    cand = inject_proposal(sd, traj_xy, yaw, log_sd=log_sd, horizon=horizon)
    if carry:                                   # continuity for the stateful direction detector
        for _k, _v in carry.items():
            setattr(cand, _k, _v)
    shared: dict = {}
    out: dict = {}
    # PROGRESS, in metres of arc length advanced along the nearest centre line over the candidate's
    # horizon. Normalising it into the paper's bounded EP needs a reference, which only the caller
    # has (we use the teacher's own advance), so the RAW advance is emitted here and the ratio is
    # formed in build_scorer_targets.
    a0 = route_arc_length(sd, -1)
    a1 = route_arc_length(cand, -1)
    out["progress.advance_m"] = (a1 - a0) if (a0 == a0 and a1 == a1) else float("nan")
    # built AFTER the carry is applied, so the view inherits this prefix's threaded state
    view = _ego_view(cand) if EGO_VIEW else None
    for name in CALC_ORDER:
        calc = calculators.get(name)
        if calc is None:
            continue
        try:
            calc.forward(view if (view is not None and name in EGO_VIEW_CALCS) else cand,
                         log_sd, shared)
        except Exception as exc:                          # record, never silently zero
            out[f"{name}.error"] = f"{type(exc).__name__}: {exc}"[:110]
    # ⛔ HARVEST AT THE END, NOT PER CALCULATOR -- A LATER CALCULATOR MUTATES AN EARLIER ONE'S
    # ENTRY. The previous version read `set(shared) - before` after each `forward`, i.e. only the
    # keys THAT calculator newly created. But `CenterLine` writes the wrong-way categories
    # **into `rewards_and_infos["OffRoad"]["info"]`**, an entry `OffRoad` had already created, so
    # the key was not "new" and the mutation was never re-read.
    # MEASURED consequence: `OffRoad.info` took only {0, 1} across the entire bank, categories 3
    # and 4 never appeared, and driving-direction compliance carried ZERO supervision -- which read
    # as "their detector never fires on our data" when in fact our reader never looked again.
    # ⭐ The shared dict is the calculators' contract with each other; the final state is the answer
    # and any snapshot taken mid-chain is a different question.
    owner = {"OffRoad": "off_road", "CrossLane": "off_road", "NuPlanCollision": "collision",
             "NuPlanTTC": "ttc", "Comfort": "comfort", "GoalReaching": "goal_reaching",
             "CenterLine": "center_line", "CurbClearance": "center_line",
             "Overspeed": "center_line"}
    for k, v in shared.items():
        for leaf, val in _leaves(v, f"{owner.get(k, 'other')}.{k}"):
            # ⛔ TWO LEAVES CANNOT BE READ AS SIGNALS AND ARE RENAMED SO NOBODY TRIES.
            #   collision.NuPlanCollision.reward -- a SIGN-DEPENDENT value aggregated with `min`.
            #     "Worst" is only `min` while the sign convention holds, and nothing enforces it;
            #     the authoritative collision signal is the `.info` EVENT, which we already use.
            #   ttc.NuPlanTTC.info -- a NORM over a PADDED stack. It banks as a constant 6.7082
            #     that looks like a time-to-collision in seconds and is not one.
            # Emitting them under `raw.` keeps the diagnostic value and makes it impossible to
            # mistake either for a component. Deleting them would hide a number someone may need.
            out[("raw." + leaf) if leaf in NON_AUTHORITATIVE else leaf] = _ego_scalar(val)
    if carry is not None:                       # hand the accumulated state to the next prefix
        for _k in CARRY_STATE:
            # the direction / deadband latches live on whichever object their calculator ran on
            _v = getattr(view if (view is not None and _k in EGO_VIEW_CARRY) else cand, _k, None)
            if _v is not None:
                carry[_k] = _v
    # DRIVING-DIRECTION COMPLIANCE, which we were not reading at all. The DDC detector exists and
    # writes into `OffRoad.info` as CATEGORIES 3 (wrong way) and 4 (severe wrong way) -- see
    # `off_road.py`: 0 none / 1 off-road / 2 solid-line crossing / 3 wrong-way / 4 severe. Our
    # consumer clipped that category to [0, 1], which collapses 2, 3 and 4 onto 1 and threw the
    # distinction away. Emit DDC as its own leaf so the category is never clipped again.
    # ⛔ `OffRoad.info` IS NOMINAL (0 none / 1 off-road / 2 solid-line / 3 wrong-way / 4 severe),
    # AND AGGREGATING A CATEGORY WITH `max` ASKS THE WRONG QUESTION. MEASURED by the third review:
    # 100 of 1,066 rows sit at category 3 or 4, and because `max` returns 4 rather than 1 the
    # drivable-area consumer -- which tests `cat == 1` -- read them as CLEAN. A larger code is not
    # a worse code; they are different events.
    # ⇒ derive the two BOOLEAN questions here, per prefix, and let the rollout aggregate booleans.
    # "Was it ever off-road" and "was it ever wrong-way" are both well-defined under `max`;
    # "what was the largest category" is not a question anyone wants answered.
    info = out.get("off_road.OffRoad.info")
    if info is not None and not isinstance(info, str) and info == info:
        cat = int(round(float(info)))
        out["dac.violation"] = 1.0 if cat == 1 else 0.0
        out["ddc.violation"] = 1.0 if cat in (3, 4) else 0.0
    return out


def score_proposal_rollout(sd, log_sd, calculators, traj_xy, yaw=None,
                           stride: int = 1) -> dict:
    """Score a proposal STEP BY STEP over its horizon and aggregate. This is the correct semantics.

    ⛔ WHY ENDPOINT-ONLY SCORING IS WRONG, MEASURED 2026-09-20 with an analytic target.
    `extract_recent_agent_polygons` reads ONLY timesteps -1 and -2, and
    `OffRoad.detect_cross_line_knn` is a per-step CROSSING detector: it fires when the ego footprint
    straddles a curb in that one transition. A proposal that crosses a curb at step 6 and ends 8 m
    beyond it therefore scores CLEAN at its endpoint -- the violation happened in a step nobody
    looked at. MEASURED on one frame: a trajectory aimed through the nearest curb reads
    `OffRoad.info` 1.0 when it STOPS on the curb and 0.0 when it continues 8 m past it.
    In the RL engine these calculators are called once per simulation step, which is exactly what
    this function reproduces offline.

    ⭐ THE CONTROL THAT PROVES IT: the '8 m past the curb' candidate MUST read off-road here and
    MUST read clean under `score_proposal`. If both agree, this function is not doing its job.

    COST: T calculator passes per proposal instead of 1 (T=20 => 20x). With 64 proposals that is
    1,280 passes per frame; use `stride` to subsample the horizon when building a large bank, and
    state the stride with any number produced from it.
    """
    T = traj_xy.shape[0]
    # ⛔ START AT 3, NOT 1. A prefix of one pose has no acceleration and a prefix of two has no
    # jerk, so the short prefixes are DEGENERATE -- their comfort verdict is a boundary artefact of
    # the derivative, not a property of the candidate. MEASURED by the fix-verification review:
    # at k=1 EVERY candidate reads comfort 0.5, so a `min` over prefixes returned 0.5 for all ten
    # and the component could not rank anything -- even though the full-horizon values are correct
    # and well separated (teacher 1.0000, stopped 0.5000). The aggregation, not the calculator, was
    # eating the signal.
    steps = [k for k in range(MIN_PREFIX, T + 1, max(int(stride), 1))]
    if not steps:
        steps = [T]
    if steps[-1] != T:
        steps.append(T)
    per_step = []
    carry: dict = {}                            # threads the direction detector's state forward
    for k in steps:
        # drop any latch whose owner is not yet admissible at this prefix, so it cannot arm early
        for _lk, _own in LATCH_OWNER.items():
            if k < KEY_MIN_PREFIX.get(_own, MIN_PREFIX):
                carry.pop(_lk, None)
        per_step.append(score_proposal(sd, log_sd, calculators, traj_xy[:k],
                                       None if yaw is None else yaw[:k], carry=carry,
                                       horizon=T))
    out: dict = {}
    keys = set().union(*[set(d) for d in per_step]) if per_step else set()
    for key in keys:
        kmin = _min_prefix_for(key)
        vals = [d[key] for k, d in zip(steps, per_step)
                if key in d and not isinstance(d[key], str) and k >= kmin]
        if not vals:            # never reached a valid prefix -- say so rather than inventing one
            vals = [d[key] for d in per_step if key in d and not isinstance(d[key], str)][-1:]
        if not vals:
            out[key] = next(d[key] for d in per_step if key in d)
            continue
        finite = [v for v in vals if v == v]
        if not finite:
            out[key] = float("nan")
            continue
        # an EVENT is "did it ever happen"; a reward is "the worst it ever got".
        if key.endswith((".info", ".occ_hit", ".lane_change_info",
                         ".solid_lane_change_info", ".stage_reached", "ddc.violation")):
            out[key] = max(finite)                 # an EVENT: did it ever happen
        elif key in CUMULATIVE_KEYS:
            # ⛔ PROGRESS IS CUMULATIVE AND MUST BE READ AT THE FULL HORIZON, NOT MINIMISED.
            # A `min` over prefixes returns the SHORTEST prefix's advance, which is close to zero
            # for every candidate. MEASURED by the review: the banked value was the 0.2 s prefix,
            # so progress saturated at ~4 % of the distance actually travelled and could not
            # distinguish a candidate that drove 25 m from one that drove 1 m.
            out[key] = vals[-1]
        else:
            out[key] = min(finite)                 # a QUALITY: the worst it ever got
        out[key + "@last"] = vals[-1]
    out["_n_steps_scored"] = float(len(steps))
    return out


def selftest() -> int:
    """Prove the pieces exist and connect, without pretending to produce a real target bank."""
    print("Stage 2b self-test: do the released calculators connect to a nuPlan ScenarioData?\n")
    ok = True
    try:
        from driverl.nuplan.feature_builder import DriveRLNuPlanFeatureBuilder  # noqa: F401
        print("  feature builder import                    OK")
    except Exception as exc:
        print(f"  feature builder import                    FAIL {type(exc).__name__}: {exc}")
        ok = False
    try:
        from driverl.datatypes.scenario_data import ScenarioData  # noqa: F401
        print("  ScenarioData import                       OK")
        import inspect
        src = inspect.getsource(ScenarioData)
        print(f"  agent 0 is the ego                        "
              f"{'CONFIRMED' if 'agent_positions_all[:, 0' in src else 'NOT CONFIRMED'}")
        print(f"  polygon cache exists (must be cleared)    "
              f"{'YES' if 'clean_polygon_cache' in src else 'NO'}")
    except Exception as exc:
        print(f"  ScenarioData import                       FAIL {type(exc).__name__}: {exc}")
        ok = False
    names = []
    for mod, cls in (("off_road", "OffRoad"), ("collision.nuplan_collision", "NuPlanCollision"),
                     ("nuplan_ttc", "NuPlanTTC"), ("comfort", "Comfort"),
                     ("goal_reaching", "GoalReaching"), ("center_line", "CenterLine")):
        try:
            m = __import__(f"driverl.env.engine.reward_calculator.{mod}", fromlist=[cls])
            getattr(m, cls)
            names.append(cls)
        except Exception as exc:
            print(f"  calculator {cls:16s}               FAIL {type(exc).__name__}: {exc}")
            ok = False
    print(f"  six calculators importable                {len(names)}/6  {names}")
    print(f"\n  components, in the order the scorer target must use: {COMPONENTS}")
    print("\n" + ("SCORER_SELFTEST_OK" if ok else "SCORER_SELFTEST_FAILED"))
    print("\n  NOT DONE YET: wiring these targets into train.py. Until that happens the scorer")
    print("  trains against a placeholder and proposal selection at inference stays arbitrary.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else selftest())
