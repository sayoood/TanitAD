"""NAVSIM **v1.1** agents for W3 (NAVSIM VENV). Scored by the OFFICIAL, UNMODIFIED v1.1
``run_pdm_score.py`` through Hydra overrides only, e.g.::

    agent=constant_velocity_agent  agent._target_=w3_agents_v1.StopAgent
    agent=constant_velocity_agent  agent._target_=w3_agents_v1.SeamAgentV1 \
        +agent.seam_file=<seam.npz> +agent.call_log=<calls.jsonl>

* ``StopAgent`` — the STOP floor: 8 × ``(0, 0, 0)`` (x fwd, y left, heading), i.e. "stay at
  the t0 rear-axle pose". It reads NOTHING (no ego status, no scene); the plan is then tracked
  by the devkit's LQR + bicycle model like any other.
* ``SeamAgentV1`` — the v1.1 port of E2's ``tanitad_seam_agent.TanitADSeamAgent``: a LOOKUP
  TABLE keyed by the SCORER token ``scene.scene_metadata.initial_token`` (v1.1
  ``dataclasses.py:422-427``: the frame ``num_history_frames-1`` token, the same key the runner
  iterates), refusing any token it does not hold and any AgentInput whose fingerprint differs
  from the one the export recorded (E2's ``fingerprint``, imported verbatim below). v1.1
  differences from E2's v2 class: ``AbstractAgent.__init__`` takes only ``requires_scene``;
  no stage-1 stand-in exists (navtest is single-stage: every token is an original log frame).
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory

_DEF_SAMPLING = TrajectorySampling(time_horizon=4, interval_length=0.5)


def fingerprint(ego_statuses) -> str:
    """VERBATIM copy of E2's ``export_agent_inputs.fingerprint`` / ``tanitad_seam_agent.fingerprint``
    (``…/2026-09-19-navsim-refcv4b-bridge/code/tanitad_seam_agent.py``); pinned equal by
    ``tests/test_w3_agents.py``."""
    h = hashlib.sha1()
    for es in ego_statuses:
        for arr in (es.ego_pose, es.ego_velocity, es.ego_acceleration, es.driving_command):
            h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes())
    return h.hexdigest()


class StopAgent(AbstractAgent):
    """The all-zero plan. requires_scene=False: it is handed an AgentInput and ignores it."""

    requires_scene = False

    def __init__(self, trajectory_sampling: TrajectorySampling = _DEF_SAMPLING):
        super().__init__(requires_scene=False)
        self._trajectory_sampling = trajectory_sampling

    def name(self) -> str:
        return "StopAgent"

    def initialize(self) -> None:
        pass

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        poses = np.zeros((self._trajectory_sampling.num_poses, 3), dtype=np.float32)
        return Trajectory(poses, self._trajectory_sampling)


class SeamAgentV1(AbstractAgent):
    """Lookup of precomputed poses by scorer token, fingerprint-checked. Never a model."""

    requires_scene = True

    def __init__(self, trajectory_sampling: TrajectorySampling = _DEF_SAMPLING,
                 seam_file: str = "", call_log: str = ""):
        super().__init__(requires_scene=True)
        self._trajectory_sampling = trajectory_sampling
        self._seam_file = seam_file
        self._call_log = call_log
        self._rows: dict = {}
        self._fp: dict = {}
        self._meta: dict = {}

    def name(self) -> str:
        return "SeamAgentV1"

    def initialize(self) -> None:
        if not self._seam_file or not os.path.exists(self._seam_file):
            raise FileNotFoundError(f"seam file missing: {self._seam_file!r}")
        z = np.load(self._seam_file, allow_pickle=False)
        toks = [str(x) for x in z["token"]]
        fps = [str(x) for x in z["fingerprint"]]
        poses = np.asarray(z["poses"], dtype=np.float32)
        ts = [float(x) for x in z["sampling"]]
        if (int(ts[0]), ts[1]) != (self._trajectory_sampling.num_poses,
                                   self._trajectory_sampling.interval_length):
            raise ValueError(f"seam sampling {ts} != agent sampling {self._trajectory_sampling}")
        if len(set(toks)) != len(toks):
            raise ValueError("seam carries duplicate tokens")
        if poses.shape != (len(toks), self._trajectory_sampling.num_poses, 3):
            raise ValueError(f"seam poses {poses.shape} != ({len(toks)}, "
                             f"{self._trajectory_sampling.num_poses}, 3)")
        if not np.isfinite(poses).all():
            raise ValueError("non-finite seam poses")
        for tok, fp, p in zip(toks, fps, poses):
            self._fp[tok] = fp
            self._rows[tok] = p
        self._meta = {"n_rows": len(self._rows),
                      "arm": str(z["arm"]) if "arm" in z.files else "?"}
        self._log(event="initialize", **self._meta, seam_file=self._seam_file)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def _log(self, **rec) -> None:
        if not self._call_log:
            return
        rec["t"] = round(time.time(), 3)
        with open(self._call_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        tok = scene.scene_metadata.initial_token                  # ⛔ the ONLY scene field read
        want = self._fp.get(tok)
        if want is None:
            self._log(event="call", token=tok, source="UNKNOWN_TOKEN")
            raise KeyError(f"token {tok} not in the seam — refusing to invent a trajectory")
        got = fingerprint(agent_input.ego_statuses)
        if got != want:
            self._log(event="call", token=tok, source="FINGERPRINT_MISMATCH")
            raise ValueError(f"token {tok}: AgentInput fingerprint {got} != exported {want}")
        self._log(event="call", token=tok, source="seam")
        return Trajectory(self._rows[tok].copy(), self._trajectory_sampling)
