"""``plans/<arm>.npz`` + ``scenes.json`` — the inputs W5's FAILURE GALLERY needs (the part the PI
asked for by name: camera projection + metric BEV inset + text overlay).

Shapes are W5's (`taniteval/taniteval/benchreport/adapt.py:452-460` and `:162-174`), so the gallery
reads a W1 run exactly as it reads W5's own fixture:

    plans/<arm>.npz   token [N] · poses [N, 8, 3] float32 (x fwd, y left, heading) · source [N]
                      ('devkit_agent' | 'precomputed' | 'refcv4b' | 'cv_standin') ·
                      sampling [num_poses, interval_s] · arm · seam_file
    scenes.json       {source, what, synthetic_scene_pickles, frame_bank, maps_root,
                       tokens: {token: {stage, log, v0, command, command_onehot, scene_token,
                                        map_name, pickle, frame_type, num_future_frames}}}

⭐ THE POSES COME FROM THE SCORER'S OWN HOOK, not from the seam: the promoted wrapper records
``agent_poses`` for EVERY ``pdm_score`` call (E1's observation hook), so what is drawn is what the
official scorer executed — for a devkit agent (CV) as well as for a seam arm. The seam's ``source``
column is used only to LABEL each row (so ``cv_standin`` rows stay excluded from the gallery).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _poses_from_hooks(hooks: list) -> dict:
    out = {}
    for c in hooks:
        t, p = c.get("token"), c.get("agent_poses")
        if t and p is not None:
            out[str(t)] = np.asarray(p, dtype=np.float32)
    return out


def _sampling_from_hooks(hooks: list, default=(8, 0.5)) -> tuple:
    for c in hooks:
        s = c.get("agent_sampling")
        if s:
            return (int(s[0]), float(s[1]))
    return default


def write_plans(out_path: Path, *, arm: str, hooks: list, seam: Path | None) -> dict:
    """-> a record for the run manifest, or a refusal (never a silent absence)."""
    poses = _poses_from_hooks(hooks)
    if not poses:
        return {"status": "UNAVAILABLE", "n": 0,
                "reason": (f"no agent poses recorded for {arm}: the wrapper's pdm_score hook wrote no calls "
                           "(a process-pool run does not reach it — score with worker=sequential)")}
    src = {}
    seam_name = None
    if seam is not None and Path(seam).exists():
        z = np.load(seam, allow_pickle=False)
        src = {str(t): str(s) for t, s in zip(z["token"], z["source"])}
        seam_name = Path(seam).name
    toks = sorted(poses)
    arr = np.stack([poses[t] for t in toks]).astype(np.float32)
    source = np.asarray([src.get(t, "devkit_agent") for t in toks])
    n_poses, dt = _sampling_from_hooks(hooks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, token=np.asarray(toks), poses=arr, source=source,
                        sampling=np.asarray([n_poses, dt]), arm=np.asarray(arm),
                        seam_file=np.asarray(seam_name or "devkit agent (poses from the scorer's pdm_score hook)"))
    return {"status": "OK", "n": len(toks), "path": out_path.name, "shape": list(arr.shape),
            "sampling": [n_poses, dt], "n_cv_standin": int(sum(1 for s in source if s == "cv_standin")),
            "poses_from": "the scorer's own pdm_score hook (what the official runner executed)"}


def _drop_top_level_nulls(d: dict) -> list:
    """Remove top-level keys whose value is ``None`` and return their names.

    ⛔ THE CLASS, not the instance: ``{"k": None}.get("k", "")`` returns **None**, not ``""`` --
    a dict default fires only on an ABSENT key, never on a present-but-null one. So emitting a null
    into a shared artifact hands every downstream ``.get(k, default)`` a value its author
    specifically wrote code to avoid. Absence is the honest encoding of "not declared".

    ⚠ TOP LEVEL ONLY, on purpose: ``tokens`` legitimately carries per-token nulls (``pickle``,
    ``map_name`` …) that W5 reads with ``meta.get(...) or <fallback>``, which IS null-safe. Dropping
    those would change the token schema to fix a problem that does not exist there.
    """
    dropped = sorted(k for k, v in d.items() if v is None)
    for k in dropped:
        del d[k]
    return dropped


def write_scenes(out_path: Path, doc: dict, *, prof, frame_bank: str | None = None) -> dict:
    """``scenes.json`` from the devkit AgentInput export — W5's ``_scene_meta`` fields, same names."""
    meta = {}
    for t, r in doc["tokens"].items():
        ego = r["ego_statuses"][-1]
        vx, vy = ego["ego_velocity"]
        cmd = ego.get("driving_command") or []
        meta[t] = {"stage": int(r["stage"]), "log": r["log_name"], "v0": math.hypot(vx, vy),
                   "command_onehot": list(cmd),
                   "command": (["left", "straight", "right", "unknown"][cmd.index(max(cmd))]
                               if cmd and max(cmd) > 0 else "none"),
                   "scene_token": r.get("scene_token"), "map_name": r.get("map_name"),
                   "pickle": r.get("pickle"), "frame_type": r.get("frame_type"),
                   "num_future_frames": r.get("num_future_frames")}
    doc_out = {
        "source": "the run's own devkit AgentInput export (raw/export_record.json names the file + sha256)",
        "what": "per scorer token: stage, log, |v0| (m/s), NavSim command at t0, scene/pickle ids",
        "synthetic_scene_pickles": str(prof.syn_scenes).replace("\\", "/"),
        "frame_bank": frame_bank,
        "maps_root": "C:/Users/Admin/navsim-crun/data/maps",
        "split": prof.name, "protocol": prof.protocol,
        "tokens": {t: meta[t] for t in sorted(meta)}}
    dropped = _drop_top_level_nulls(doc_out)
    if dropped:
        doc_out["_omitted_because_null"] = {
            "keys": dropped,
            "why": ("a key PRESENT with a null value defeats every consumer's `.get(key, default)` -- "
                    "the default fires only when the key is ABSENT. MEASURED 2026-09-20: navhard has no "
                    "entry in model_arms.DEFAULT_BANKS, so `frame_bank` was written as null, and W5's "
                    "`Path(run.scenes.get('frame_bank', ''))` raised TypeError: Path(None) -- taking down "
                    "the WHOLE report of a COMPLETE run whose scores were fine. Warmup was unaffected only "
                    "because it HAS a default bank, which is why this survived every warmup run."),
            "read_it_as": "absent == not declared for this run; see the *_declared flags in gallery_inputs",
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc_out, indent=1), encoding="utf-8")
    return {"status": "OK", "n_tokens": len(meta), "path": out_path.name,
            "frame_bank_declared": bool(frame_bank), "omitted_null_keys": dropped}
