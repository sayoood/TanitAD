"""CANDIDATE seam agent for the Thor backend (navsim v2 / navhard two-stage): E2's ``TanitADSeamAgent``
with a reference-checked fingerprint fallback.  NOT a default until the Master Mind accepts it.

WHY (MEASURED 2026-10-04): on Thor the AgentInput the devkit builds differs from the dev-box one in
the last bits for 481 / 5,912 navhard tokens (``raw/fp_audit_navhard_all.json``; all of them
synthetic stage-two scenes) -- ``np.cos(theta)`` and ``arctan2(sin, cos)`` round differently in the
Windows CRT and in glibc/aarch64 (``raw/diag_agent_input_{devbox,thor}.json``), max |delta| 3.6e-15
(``raw/ulp_reference_check_navhard.json``). E2's fingerprint is SHA-1 over the float64 BYTES, so
those tokens are refused, rows go invalid and the two-stage aggregation dies. For a seam arm the
AgentInput is NOT a scoring input (the agent returns the precomputed plan); the fingerprint proves
the row belongs to the token.

RULE: fingerprint equal -> E2's class, byte-identical behaviour. Fingerprint different -> accept ONLY
if the dev-box reference for that token (exact float64 hex, ``make_ulp_reference.py`` on the dev box)
(a) RE-HASHES to the seam's recorded fingerprint -- i.e. it IS the exported input -- and (b) every
element of Thor's AgentInput is within ``atol`` (default 1e-9; chosen after the 3.6e-15
measurement, see ``ulp_guard.max_abs_vs_reference``) of it. Otherwise raise exactly like E2. Each
acceptance is logged ``source: "seam", fp_mode: "reference"`` with the measured max |delta| and ulp,
so the count guard holds and every waiver is auditable.
"""
from __future__ import annotations

import json

from tanitad_seam_agent import TanitADSeamAgent, fingerprint

import ulp_guard as U


class TanitADSeamAgentULP(TanitADSeamAgent):
    def __init__(self, *a, ulp_reference: str = "", atol: float = 1e-9, **k):
        super().__init__(*a, **k)
        self._ulp_reference = ulp_reference
        self._atol = float(atol)
        self._ref: dict = {}

    def name(self) -> str:
        return "TanitADSeamAgentULP"

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
            return super().compute_trajectory(agent_input, scene)       # E2's exact behaviour
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
        if tok not in self._rows:
            return super().compute_trajectory(agent_input, scene)       # (stand-in path: unchanged)
        self._log(event="call", token=tok, source="seam", fp_mode="reference", max_abs=dabs,
                  max_ulp=U.max_ulp_vs_reference(agent_input.ego_statuses, ref), fp_thor=got, fp_seam=want)
        from navsim.common.dataclasses import Trajectory
        return Trajectory(self._rows[tok].copy(), self._trajectory_sampling)
