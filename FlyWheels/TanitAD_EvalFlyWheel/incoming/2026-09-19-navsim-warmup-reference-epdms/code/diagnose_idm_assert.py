#!/usr/bin/env python3
"""Reproduce and DIAGNOSE the navhard failure:
`AssertionError: Agent's baseline does not intersect the agent itself`
(navsim_idm_agent_manager.py:84), which poisoned 166/5,462 stage-2 rows and then killed the
whole run in the aggregator (scene_aggregator.py:62 -> run_pdm_score.py:382 -> TypeError).

For one failing token it re-runs the official scoring path and, when the assertion fires, walks
`self.agents` and reports, for each agent whose own token is missing from its own buffered path's
intersection set: the raw vs simplified own-polygon, the distance from the agent box to its
path-to-go, the agent width, and whether the RAW (unsimplified) map would have contained it.
That distinguishes "the devkit's in-loop simplify(1e-5) eroded the geometry" from
"the agent was snapped away from its own baseline" (navsim idm_snap_threshold = 3.0 m).

Usage: python diagnose_idm_assert.py <token> [<token> ...] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CACHE = Path("C:/Users/Admin/navsim-crun/exp/metric_cache_navhard_two_stage")
SCENES = Path("C:/Users/Admin/navsim-crun/data/openscene/navhard_two_stage/synthetic_scene_pickles")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import navsim_win  # noqa: E402  (the loader separator patch lives here)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens", nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from shapely.geometry.base import CAP_STYLE
    from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
    from navsim.common.dataclasses import SensorConfig, Scene
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.observation.navsim_idm import navsim_idm_agent_manager as M
    from navsim.agents.constant_velocity_agent import ConstantVelocityAgent

    MetricCacheLoader._load_metric_cache_paths = navsim_win._load_metric_cache_paths_patched
    cache = MetricCacheLoader(CACHE)

    with initialize_config_module(config_module="navsim.planning.script.config.pdm_scoring", version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=[
            "train_test_split=navhard_two_stage", "experiment_name=e1_diag", "output_dir=C:/Users/Admin/navsim-crun/exp/e1/diag"])
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    policy = instantiate(cfg.traffic_agents_policy.reactive, simulator.proposal_sampling)

    findings = {}
    orig = M.NavsimIDMAgentManager.propagate_agents

    def probe(self, ego_state, tspan, iteration, open_loop_detections, radius, traffic_light_status=None):
        try:
            return orig(self, ego_state, tspan, iteration, open_loop_detections, radius, traffic_light_status)
        except AssertionError as e:
            rows = []
            for token, agent in self.agents.items():
                if not (agent.is_active(iteration) and agent.has_valid_path()):
                    continue
                path = path_to_linestring(agent.get_path_to_go())
                buf = path.buffer(agent.width / 2, cap_style=CAP_STYLE.flat)
                inter = self.agent_occupancy.intersects(buf)
                if inter.contains(token):
                    continue
                own = self.agent_occupancy.get(token)
                rows.append({
                    "agent_token": token,
                    "agent_in_occupancy_map": self.agent_occupancy.contains(token),
                    "own_geom_is_empty": bool(own is None or own.is_empty),
                    "own_geom_area": (None if own is None else round(float(own.area), 6)),
                    "own_geom_valid": (None if own is None else bool(own.is_valid)),
                    "agent_width_m": round(float(agent.width), 3),
                    "path_to_go_length_m": round(float(path.length), 3),
                    "dist_own_box_to_path_m": (None if own is None else round(float(own.distance(path)), 3)),
                    "dist_own_box_to_buffered_path_m": (None if own is None else round(float(own.distance(buf)), 3)),
                    "buffer_half_width_m": round(float(agent.width / 2), 3),
                    "agent_pos": [round(float(v), 3) for v in agent.to_se2().point.array],
                    "path_start": [round(float(v), 3) for v in path.coords[0]],
                    "ego_pos": [round(float(v), 3) for v in ego_state.center.point.array],
                    "dist_agent_to_ego_m": round(float(((agent.to_se2().point.array - ego_state.center.point.array) ** 2).sum() ** 0.5), 2),
                    "n_agents": len(self.agents), "occupancy_size": int(self.agent_occupancy.size),
                    "iteration": int(iteration),
                })
            findings.setdefault("_assert_events", []).append({"error": str(e), "offenders": rows})
            raise

    M.NavsimIDMAgentManager.propagate_agents = probe
    agent = ConstantVelocityAgent()
    agent.initialize()

    for tok in a.tokens:
        mc = cache.get_from_token(tok)
        scene = Scene.load_from_disk(_find_scene(tok), None, SensorConfig.build_no_sensors())
        traj = agent.compute_trajectory(scene.get_agent_input())
        findings[tok] = {"scene_type": str(mc.scene_type), "log": mc.log_name}
        try:
            pdm_score(metric_cache=mc, model_trajectory=traj, future_sampling=simulator.proposal_sampling,
                      simulator=simulator, scorer=scorer, traffic_agents_policy=policy)
            findings[tok]["result"] = "SCORED OK (no assertion)"
        except AssertionError as e:
            findings[tok]["result"] = f"AssertionError: {e}"
            findings[tok]["events"] = findings.pop("_assert_events", [])
    a.out.write_text(json.dumps(findings, indent=1, default=str), encoding="utf-8")
    print(json.dumps(findings, indent=1, default=str)[:4000])
    return 0


def _find_scene(token: str) -> Path:
    """synthetic pickles are named by SCENE token; the scorer keys by INITIAL token, so scan once."""
    import pickle, pathlib
    global _INDEX
    try:
        _INDEX
    except NameError:
        class U(pickle.Unpickler):
            def find_class(self, m, n):
                return pathlib.PurePosixPath if (m == "pathlib" and n == "PosixPath") else super().find_class(m, n)
        _INDEX = {}
        for p in SCENES.iterdir():
            md = U(open(p, "rb")).load()["scene_metadata"]
            _INDEX[md["initial_token"]] = p
    return _INDEX[token]


if __name__ == "__main__":
    raise SystemExit(main())
