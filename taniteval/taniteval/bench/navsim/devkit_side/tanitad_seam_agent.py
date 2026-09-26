"""NavSim-side seam agent (NAVSIM VENV) — a LOOKUP TABLE, not a model.

Scored by the OFFICIAL, UNMODIFIED ``run_pdm_score.py`` through Hydra overrides
only (no devkit file is edited):

    agent=constant_velocity_agent
    agent._target_=tanitad_seam_agent.TanitADSeamAgent
    +agent.seam_file=<raw/seam_<arm>.npz>  +agent.call_log=<raw/...jsonl>

⛔ WHAT IT READS — AND WHY IT NEEDS ``requires_scene``. The row it returns is
keyed by the SCORER TOKEN, ``scene.scene_metadata.initial_token`` — the same
key the devkit's scorer loop iterates (``SceneLoader.synthetic_scenes`` is keyed
by ``initial_token``; stage-1 ``initial_token`` is the frame-3 token), and the
same object ``HumanAgent`` receives. An AgentInput-only key was tried first and
REFUSED BY MEASUREMENT: 8 groups / 19 tokens of distinct synthetic renders carry
a byte-identical 4-frame ego history (``export_agent_inputs.log``, rc=3), so a
lookup on ``agent_input`` would hand one render another's plan. From the Scene it
reads ``scene_metadata.initial_token`` and NOTHING ELSE (``tests/test_seam_agent``
feeds a Scene whose frames are poisoned and requires an identical output).

⛔ WHAT IT CHECKS. The fingerprint of the ``agent_input`` the scorer actually
hands it must equal the fingerprint the export recorded for that token — i.e.
the declared values in the seam manifest are provably the values the devkit
delivered. A mismatch RAISES.

⛔ STAGE-1 STAND-IN (declared, never silent). The camera arms have NO stage-1
frames on this box (0/192 original jpgs). The seam lists those tokens
explicitly as ``cv_standin``; for them — and ONLY them — this agent delegates to
the DEVKIT's own ``ConstantVelocityAgent`` so the official script can run end to
end. Any other unknown token RAISES (the scorer records an invalid row and the
count guard in ``score_arm.py`` refuses the run).
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory


def fingerprint(ego_statuses) -> str:
    """VERBATIM copy of export_agent_inputs.fingerprint."""
    h = hashlib.sha1()
    for es in ego_statuses:
        for arr in (es.ego_pose, es.ego_velocity, es.ego_acceleration, es.driving_command):
            h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes())
    return h.hexdigest()


class TanitADSeamAgent(AbstractAgent):
    requires_scene = True

    def __init__(self, trajectory_sampling: TrajectorySampling = TrajectorySampling(
                 time_horizon=4, interval_length=0.5), seam_file: str = "",
                 call_log: str = ""):
        super().__init__(trajectory_sampling, requires_scene=True)
        self._seam_file = seam_file
        self._call_log = call_log
        self._rows: dict = {}
        self._fp: dict = {}
        self._standin: set = set()
        self._cv = ConstantVelocityAgent(trajectory_sampling)
        self._meta: dict = {}

    def name(self) -> str:
        return "TanitADSeamAgent"

    def initialize(self) -> None:
        if not self._seam_file or not os.path.exists(self._seam_file):
            raise FileNotFoundError(f"seam file missing: {self._seam_file!r}")
        z = np.load(self._seam_file, allow_pickle=False)
        toks = [str(x) for x in z["token"]]
        fps = [str(x) for x in z["fingerprint"]]
        src = [str(x) for x in z["source"]]
        poses = np.asarray(z["poses"], dtype=np.float32)
        ts = [float(x) for x in z["sampling"]]            # (num_poses, interval_length)
        if (int(ts[0]), ts[1]) != (self._trajectory_sampling.num_poses,
                                   self._trajectory_sampling.interval_length):
            raise ValueError(f"seam sampling {ts} != agent sampling "
                             f"{self._trajectory_sampling}")
        if len(set(toks)) != len(toks):
            raise ValueError("seam carries duplicate tokens")
        for tok, fp, s, p in zip(toks, fps, src, poses):
            self._fp[tok] = fp
            if s == "cv_standin":
                self._standin.add(tok)
            elif s in ("refcv4b", "precomputed"):
                if not np.isfinite(p).all():
                    raise ValueError(f"non-finite seam poses for {tok}")
                self._rows[tok] = p
            else:
                raise ValueError(f"unknown seam source {s!r}")
        self._meta = {"n_rows": len(self._rows), "n_standin": len(self._standin),
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
        tok = scene.scene_metadata.initial_token          # ⛔ the ONLY field read
        want = self._fp.get(tok)
        got = fingerprint(agent_input.ego_statuses)
        if want is None:
            self._log(event="call", token=tok, source="UNKNOWN_TOKEN")
            raise KeyError(f"token {tok} not in the seam — refusing to invent a trajectory")
        if got != want:
            self._log(event="call", token=tok, source="FINGERPRINT_MISMATCH")
            raise ValueError(f"token {tok}: the AgentInput handed by the scorer does not "
                             f"match the exported one ({got} != {want})")
        if tok in self._rows:
            self._log(event="call", token=tok, source="seam")
            return Trajectory(self._rows[tok].copy(), self._trajectory_sampling)
        if tok in self._standin:
            self._log(event="call", token=tok, source="cv_standin")
            return self._cv.compute_trajectory(agent_input)
        raise KeyError(f"token {tok} has no row and is not a declared stand-in")
