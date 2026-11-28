"""Floor / control SEAMS for the NavSim-side lookup agent — PROMOTED from E2
``code/make_seam_stop.py``, ``make_seam_cv.py`` (K4) and ``make_seam_echo.py`` (EvalFlyWheel,
2026-09-19). The array layout is E2's, unchanged, because the promoted
``devkit_side/tanitad_seam_agent.py`` reads exactly it:

    token [N] str · fingerprint [N] str · source [N] ('precomputed'|'refcv4b'|'cv_standin')
    poses [N, 8, 3] float32 (x fwd, y left, heading; 0.5 s grid) · knots [N, 8, 2] float32
    sampling [8, 0.5] · arm str

* STOP  — the all-zero plan for EVERY token. Reads no input, so it runs stage 1 too and its
  OFFICIAL two-stage EPDMS exists (E2: 0.3009 on warmup, above CV's 0.1854 — the STOP floor is
  MANDATORY on every NavSim row, BUILD_PLAN.md §1).
* ECHO  — ``ha0_ext``: constant measured a0 AND curvature k0, integrated by the programme's own
  ``tanitad.refs.refc_v3.kinematic_goal_extrapolation`` (IMPORTED, never re-derived) from exactly
  the declared t0 ego inputs of the model arm. Added whenever a checkpoint is scored.
* CVSEAM — (K4 control) the devkit CV agent's own poses through the seam; scoring it must
  reproduce the official ``constant_velocity_agent`` CSV exactly (seam transparency).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _save(path: Path, toks, doc, poses, arm: str, source="precomputed"):
    toks = list(toks)
    np.savez(path, token=np.asarray(toks),
             fingerprint=np.asarray([doc["tokens"][t]["fingerprint"] for t in toks]),
             source=np.asarray([source] * len(toks)), poses=np.asarray(poses, dtype=np.float32),
             knots=np.full((len(toks), 8, 2), np.nan, np.float32),
             sampling=np.asarray([8, 0.5]), arm=np.asarray(arm))
    return path


def make_stop_seam(doc: dict, path: Path, arm: str = "STOP") -> Path:
    """E2 make_seam_stop.main: zeros (N, 8, 3) float32 for every token, sorted."""
    toks = sorted(doc["tokens"])
    return _save(path, toks, doc, np.zeros((len(toks), 8, 3), np.float32), arm)


def make_cv_seam(doc: dict, path: Path, arm: str = "CVSEAM") -> Path:
    """E2 make_seam_cv.main (K4): the devkit CV agent's exported poses for every token."""
    toks = sorted(doc["tokens"])
    if any("cv_poses" not in doc["tokens"][t] for t in toks):
        raise ValueError("export has no cv_poses — re-run export_agent_inputs.py")
    poses = np.asarray([doc["tokens"][t]["cv_poses"] for t in toks], dtype=np.float32)
    return _save(path, toks, doc, poses, arm)


def make_echo_seam(doc: dict, path: Path, arm: str = "ECHO") -> tuple:
    """E2 make_seam_echo.main: ha0_ext from A3's declared t0 ego block (ego_velocity, ego_acceleration).
    Needs the TanitAD venv (torch + tanitad). -> (path, manifest dict)."""
    import torch
    from tanitad.refs.refc_v3 import kinematic_goal_extrapolation
    from . import bridge as B
    toks = sorted(doc["tokens"])
    rows, poses = [], []
    for t in toks:
        r = doc["tokens"][t]
        decl = B.declare(r["ego_statuses"], "A3_ego_nocmd")      # ego[t0] only, no command
        eb = B.ego_block(decl)
        v0, a0, _, k0, _ = eb["ego_state"]
        out = kinematic_goal_extrapolation(torch.tensor([v0], dtype=torch.float64),
                                           torch.tensor([a0], dtype=torch.float64),
                                           torch.tensor([k0], dtype=torch.float64),
                                           B.NAVSIM_T_S)[0].numpy()        # [8, 4]
        poses.append(out[:, :3].astype(np.float32))
        rows.append({"token": t, "stage": r["stage"], "v0": v0, "a0": a0, "k0": k0})
    _save(path, toks, doc, np.stack(poses), arm)
    man = {"arm": arm, "status": "floor (ECHO of the model arm's own declared ego inputs)",
           "declared_inputs": ["ego_velocity[t0]", "ego_acceleration[t0]"],
           "implementation": "tanitad.refs.refc_v3.kinematic_goal_extrapolation",
           "promoted_from": "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/code/make_seam_echo.py",
           "n": len(toks), "rows": rows}
    Path(str(path) + ".manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return path, man
