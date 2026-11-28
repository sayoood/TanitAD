#!/usr/bin/env python3
"""Score named navhard tokens through the official `pdm_score` path and dump their rows.

Run it BEFORE the IDM patch and AFTER it; the two runs are the patch's evidence:
  * tokens that previously raised must SCORE after the patch  (the fix works)
  * tokens that already scored must be BIT-IDENTICAL after it (the patch is inert elsewhere)
A patch without the second half is a change of unknown blast radius.

Usage: python verify_idm_patch.py --out <json> --tokens A B C ...
"""
from __future__ import annotations

import argparse
import json
import pickle
import pathlib
import sys
from pathlib import Path

CACHE = Path("C:/Users/Admin/navsim-crun/exp/metric_cache_navhard_two_stage")
SCENES = Path("C:/Users/Admin/navsim-crun/data/openscene/navhard_two_stage/synthetic_scene_pickles")
COLS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
        "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
        "history_comfort", "multiplicative_metrics_prod", "pdm_score"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import navsim_win  # noqa: E402

_INDEX = None


def scene_path(token: str) -> Path:
    global _INDEX
    if _INDEX is None:
        class U(pickle.Unpickler):
            def find_class(self, m, n):
                return pathlib.PurePosixPath if (m == "pathlib" and n == "PosixPath") else super().find_class(m, n)
        _INDEX = {}
        for p in SCENES.iterdir():
            _INDEX[U(open(p, "rb")).load()["scene_metadata"]["initial_token"]] = p
    return _INDEX[token]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.agents.constant_velocity_agent import ConstantVelocityAgent

    MetricCacheLoader._load_metric_cache_paths = navsim_win._load_metric_cache_paths_patched
    cache = MetricCacheLoader(CACHE)
    with initialize_config_module(config_module="navsim.planning.script.config.pdm_scoring", version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=[
            "train_test_split=navhard_two_stage", "experiment_name=e1_patchcheck",
            "output_dir=C:/Users/Admin/navsim-crun/exp/e1/patchcheck"])
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    agent = ConstantVelocityAgent()
    agent.initialize()
    import navsim.planning.simulation.observation.navsim_idm.navsim_idm_agent_manager as M
    patched = "E1 2026-09-20" in Path(M.__file__).read_text(encoding="utf-8", errors="replace")

    rows = {}
    for tok in a.tokens:
        mc = cache.get_from_token(tok)
        scene = Scene.load_from_disk(scene_path(tok), None, SensorConfig.build_no_sensors())
        traj = agent.compute_trajectory(scene.get_agent_input())
        policy = instantiate(cfg.traffic_agents_policy.reactive, simulator.proposal_sampling)  # fresh per token
        try:
            df, _ = pdm_score(metric_cache=mc, model_trajectory=traj, future_sampling=simulator.proposal_sampling,
                              simulator=simulator, scorer=scorer, traffic_agents_policy=policy)
            rows[tok] = {"status": "SCORED", **{c: (None if c not in df.columns else float(df[c].iloc[0])) for c in COLS}}
        except Exception as e:                                                # noqa: BLE001
            rows[tok] = {"status": f"{type(e).__name__}: {e}"}
    res = {"idm_patch_applied": patched, "module": M.__file__, "n_tokens": len(a.tokens), "rows": rows}
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"idm_patch_applied": patched, **{k: v["status"] for k, v in rows.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
