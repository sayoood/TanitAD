"""ULP-bounded AgentInput identity check (pure numpy; no navsim import, so v1.1 and v2 agents share it).

Used by the Thor seam-agent variants ONLY when E2/W3's byte fingerprint does not match. The reference
holds the dev-box AgentInput as exact float64 hex (written by ``make_ulp_reference.py`` /
``diag_agent_input.py`` on the dev box); it is admissible only if it RE-HASHES to the fingerprint the
seam recorded -- i.e. it is provably the exported input -- and Thor's values must then lie within
``max_ulp`` of it element by element. Anything else refuses, exactly like the byte check.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np

FIELDS = ("agent_ego_pose", "agent_ego_velocity", "agent_ego_acceleration", "agent_driving_command")


def _bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", float(x)))[0]


def ulp_distance(a: float, b: float) -> int:
    ia, ib = _bits(a), _bits(b)
    if (ia < 0) != (ib < 0):
        return abs(ia) + abs(ib)
    return abs(ia - ib)


def fingerprint_from_hex(ref: dict) -> str:
    h = hashlib.sha1()
    for i in range(len(ref["agent_ego_pose"])):
        for k in FIELDS:
            arr = np.array([float.fromhex(v) for v in ref[k][i]], dtype=np.float64)
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def max_abs_vs_reference(ego_statuses, ref: dict) -> float:
    """Largest |thor - reference| over every AgentInput element (inf on a shape mismatch).

    WHY ABSOLUTE, NOT ULP (MEASURED 2026-10-04, ``raw/ulp_reference_check_navhard.json``, 481 navhard
    tokens): the ulp distance explodes for values near zero (max 4,194,304 ulp) while the absolute
    difference never exceeds 3.6e-15 (m / rad). The guard's job is token IDENTITY -- two different
    scenes differ by centimetres -- so the bound is absolute. It was chosen AFTER that measurement
    (atol 1e-9 = ~2.8e5 x the observed max), and is reported per accepted token."""
    worst = 0.0
    if len(ego_statuses) != len(ref["agent_ego_pose"]):
        return float("inf")
    for i, es in enumerate(ego_statuses):
        for k, arr in zip(FIELDS, (es.ego_pose, es.ego_velocity, es.ego_acceleration, es.driving_command)):
            mine = np.asarray(arr, dtype=np.float64).ravel()
            theirs = np.array([float.fromhex(v) for v in ref[k][i]], dtype=np.float64)
            if mine.shape != theirs.shape or not np.all(np.isfinite(mine)):
                return float("inf")
            if mine.size:
                worst = max(worst, float(np.max(np.abs(mine - theirs))))
    return worst


def max_ulp_vs_reference(ego_statuses, ref: dict) -> int:
    worst = 0
    if len(ego_statuses) != len(ref["agent_ego_pose"]):
        return 1 << 62
    for i, es in enumerate(ego_statuses):
        for k, arr in zip(FIELDS, (es.ego_pose, es.ego_velocity, es.ego_acceleration, es.driving_command)):
            mine = np.asarray(arr, dtype=np.float64).ravel()
            theirs = [float.fromhex(v) for v in ref[k][i]]
            if len(mine) != len(theirs):
                return 1 << 62
            for x, y in zip(mine, theirs):
                worst = max(worst, ulp_distance(x, y))
    return worst
