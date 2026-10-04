"""CANDIDATE (not default) v1.1 seam agent for the Thor backend: W3's ``SeamAgentV1`` with the same
reference-checked fingerprint fallback as ``tanitad_seam_agent_ulp.TanitADSeamAgentULP`` (rationale,
measurements and the rule are in that docstring and in ``ulp_guard.py``). Byte-identical behaviour
when the fingerprint matches; on a mismatch it accepts the row only if the dev-box reference
re-hashes to the seam fingerprint AND Thor's AgentInput is within ``atol`` of it; every acceptance is
logged ``source: "seam", fp_mode: "reference"``.

    agent=constant_velocity_agent agent._target_=w3_agents_v1_ulp.SeamAgentV1ULP \
        +agent.seam_file=<npz> +agent.call_log=<jsonl> +agent.ulp_reference=<json> +agent.atol=1e-9
"""
from __future__ import annotations

import json

from navsim.common.dataclasses import Trajectory
from w3_agents_v1 import SeamAgentV1, fingerprint

import ulp_guard as U


class SeamAgentV1ULP(SeamAgentV1):
    def __init__(self, *a, ulp_reference: str = "", atol: float = 1e-9, **k):
        super().__init__(*a, **k)
        self._ulp_reference = ulp_reference
        self._atol = float(atol)
        self._ref: dict = {}

    def name(self) -> str:
        return "SeamAgentV1ULP"

    def initialize(self) -> None:
        super().initialize()
        if self._ulp_reference:
            self._ref = json.load(open(self._ulp_reference, encoding="utf-8"))["tokens"]
        self._log(event="reference_mode", reference=self._ulp_reference, n_reference_tokens=len(self._ref),
                  atol=self._atol)

    def compute_trajectory(self, agent_input, scene):
        tok = scene.scene_metadata.initial_token
        want = self._fp.get(tok)
        if want is None or tok not in self._ref:
            return super().compute_trajectory(agent_input, scene)
        got = fingerprint(agent_input.ego_statuses)
        if got == want:
            return super().compute_trajectory(agent_input, scene)
        ref = self._ref[tok]
        if U.fingerprint_from_hex(ref) != want:
            self._log(event="call", token=tok, source="FINGERPRINT_MISMATCH", fp_mode="reference",
                      why="reference does not re-hash to the seam fingerprint")
            raise ValueError(f"token {tok}: reference does not reproduce the exported fingerprint")
        dabs = U.max_abs_vs_reference(agent_input.ego_statuses, ref)
        if not dabs <= self._atol:
            self._log(event="call", token=tok, source="FINGERPRINT_MISMATCH", fp_mode="reference", max_abs=dabs)
            raise ValueError(f"token {tok}: AgentInput differs from the exported one by {dabs} > {self._atol}")
        self._log(event="call", token=tok, source="seam", fp_mode="reference", max_abs=dabs,
                  max_ulp=U.max_ulp_vs_reference(agent_input.ego_statuses, ref), fp_thor=got, fp_seam=want)
        return Trajectory(self._rows[tok].copy(), self._trajectory_sampling)
