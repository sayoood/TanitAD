"""The paper's scorer labels: NAVSIM v1.1 PDM targets for every proposal, computed from a nuPlan scenario (SFT-4).

DriveZero (2609.06055) Sec. 3: "These components are supervised by their corresponding PDM targets [17]" (RETRACTION_LOG
R30: our scorer had learned NC/EP/TTC/DDC from the teacher's calculators instead). This module produces, per proposal, the
six components exactly as NAVSIM's harness scores a submitted plan:

  metric cache   NAVSIM's OWN MetricCacheProcessor steps (navsim @ 3e8291b, metric_cache_processor.py:63-235), run on the
                 scenario WITHOUT writing a file: PDM-Closed (its route correction included) -> reference trajectory,
                 centreline, on-route lanes, drivable-area map; ground-truth detections at 0.5 s over 5 s interpolated to
                 0.1 s -> PDMObservation
  scoring        NAVSIM's OWN PDMSimulator + PDMScorer on [PDM-Closed, proposal_1..M] (41 steps, 0.1 s)
  per proposal   NC, DAC, TTC, C (per-trajectory in the scorer), DDC (computed, weight 0 in v1), and EP / PDMS REPRODUCED
                 PAIRWISE -- each proposal normalised against PDM-Closed alone, as pdm_score.py scores [pdm, pred]
  plans          the EXECUTED plans: REFe's 20 poses -> planner.repair_last_heading (A7) -> the seam's to_navsim (8 poses)

Validated against NAVSIM's real navtest metric cache by eval/validate_pdm_targets.py before any label is used.
Runs in an environment that has the `navsim` package (v1.1) and nuplan-devkit on the path.
"""
from __future__ import annotations

import numpy as np

_S: dict = {}


def _stub_cache_metadata_entry():
    """DriveRL's nuplan-devkit fork (the pod's teacher venv) ships without `nuplan.planning.training`. NAVSIM's
    metric_cache_processor imports ONE name from it, `CacheMetadataEntry`, used only as the return type of the file-writing
    `compute_metric_cache` that this module never calls. Register a minimal stand-in ONLY when the real module is absent."""
    import importlib.util
    import sys
    import types
    try:
        if importlib.util.find_spec("nuplan.planning.training.experiments.cache_metadata_entry") is not None:
            return False
    except ModuleNotFoundError:
        pass
    for name in ("nuplan.planning.training", "nuplan.planning.training.experiments"):
        sys.modules.setdefault(name, types.ModuleType(name))
    m = types.ModuleType("nuplan.planning.training.experiments.cache_metadata_entry")

    class CacheMetadataEntry:                          # noqa: D401 -- never instantiated on this path
        def __init__(self, file_name):
            self.file_name = file_name
    m.CacheMetadataEntry = CacheMetadataEntry
    sys.modules[m.__name__] = m
    return True


def _alias_ml_planner_transform_utils():
    """The same fork lacks `nuplan.planning.simulation.planner.ml_planner.transform_utils`, which navsim/evaluate/pdm_score.py
    imports (_get_fixed_timesteps, _se2_vel_acc_to_ego_state). refe/navsim_pdm/nuplan_transform_utils.py vendors exactly
    those functions VERBATIM (validated: navsim_dac's simulated states == NAVSIM's on 25,600 proposals). Alias it in ONLY
    when the real module is absent."""
    import importlib.util
    import os
    import sys
    import types
    try:
        if importlib.util.find_spec("nuplan.planning.simulation.planner.ml_planner.transform_utils") is not None:
            return False
    except ModuleNotFoundError:
        pass
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from navsim_pdm import nuplan_transform_utils as V
    sys.modules.setdefault("nuplan.planning.simulation.planner.ml_planner", types.ModuleType("nuplan.planning.simulation.planner.ml_planner"))
    sys.modules["nuplan.planning.simulation.planner.ml_planner.transform_utils"] = V
    return True


def _setup():
    if _S:
        return _S
    _S["stubbed_cache_metadata_entry"] = _stub_cache_metadata_entry()
    _S["aliased_transform_utils"] = _alias_ml_planner_transform_utils()
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    samp = TrajectorySampling(num_poses=40, interval_length=0.1)
    _S.update(proc=MetricCacheProcessor(cache_path=None, force_feature_computation=True), samp=samp,
              samp8=TrajectorySampling(num_poses=8, interval_length=0.5), sim=PDMSimulator(samp),
              scorer=PDMScorer(samp, PDMScorerConfig()))
    return _S


def metric_cache_parts(scenario):
    """NAVSIM's MetricCacheProcessor.compute_metric_cache, minus the file: (pdm_traj, ego, obs, centerline, lanes, dam)."""
    S = _setup()
    proc = S["proc"]
    planner_input, planner_initialization = proc._get_planner_inputs(scenario)
    proc._pdm_closed.initialize(planner_initialization)
    pdm_traj = proc._pdm_closed.compute_planner_trajectory(planner_input)
    obs = proc._interpolate_gt_observation(scenario)
    return (pdm_traj, scenario.initial_ego_state, obs, proc._pdm_closed._centerline,
            list(proc._pdm_closed._route_lane_dict.keys()), proc._pdm_closed._drivable_area_map)


def score_plans(parts, plans8: np.ndarray) -> dict:
    """plans8 [M, 8, 3] NAVSIM-frame poses (ego frame, 2 Hz) -> per-plan PDM targets (arrays of length M)."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    S = _setup()
    pdm_traj, ego, obs, centerline, lanes, dam = parts
    states = [get_trajectory_as_array(pdm_traj, S["samp"], ego.time_point)]
    for p in np.asarray(plans8, dtype=np.float32):
        it = transform_trajectory(Trajectory(p, S["samp8"]), ego)
        states.append(get_trajectory_as_array(it, S["samp"], ego.time_point))
    sim = S["sim"].simulate_proposals(np.stack(states, 0), ego)
    sc = S["scorer"]
    sc.score_proposals(sim, obs, centerline, lanes, dam)
    mult = sc._multi_metrics.prod(axis=0)
    raw = sc._progress_raw * mult
    ttc = sc._weighted_metrics[WeightedMetricIndex.TTC]
    com = sc._weighted_metrics[WeightedMetricIndex.COMFORTABLE]
    n = sim.shape[0]
    ep = np.zeros(n)
    pdms = np.zeros(n)
    for j in range(n):
        m = max(raw[0], raw[j])
        ep[j] = raw[j] / m if m > 5.0 else (1.0 if mult[j] != 0 else 0.0)
        pdms[j] = mult[j] * (5 * ep[j] + 5 * ttc[j] + 2 * com[j]) / 12.0
    sl = slice(1, n)                                   # drop PDM-Closed (index 0)
    return {"nc": sc._multi_metrics[MultiMetricIndex.NO_COLLISION][sl].copy(),
            "dac": sc._multi_metrics[MultiMetricIndex.DRIVABLE_AREA][sl].copy(),
            "ep": ep[sl], "ttc": np.asarray(ttc)[sl].copy(), "c": np.asarray(com)[sl].copy(),
            "ddc": np.asarray(sc._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION])[sl].copy(),
            "pdms": pdms[sl]}


def executed_plans8(props20: np.ndarray, repair: bool = True) -> np.ndarray:
    """REFe's [M, 20, 3] (5 Hz) -> the executed plans on NAVSIM's [M, 8, 3] grid (A7 last-heading repair, then to_navsim)."""
    import navsim_dac as ND
    P = np.asarray(props20, dtype=np.float64)
    if repair:
        P = P.copy()
        P[..., -1, 2] = P[..., -2, 2]                  # planner.repair_last_heading, identically
    return np.stack([ND.to_navsim(p) for p in P])
