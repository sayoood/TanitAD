"""L2D (yaak-ai/L2D, LeRobot) -> TanitAD episode-contract mapping SPEC + parsers.

This is the *ingest spec* for the Phase-1 L2D loader recommended in
`2026-07-11-semantic-strategic-label-dataset-survey.md`.  It is deliberately dependency-light
(numpy only) and does NOT decode video — the heavy loader (Cosmos-mirror + D-016 focal) is a
separate Phase-1 package.  What ships here and is TESTED offline:

  1. FRONT_CAMERA_KEY        — which of L2D's 6 surround cams is the front proxy.
  2. map_continuous_action() — L2D action.continuous[3] -> our contract [steer, accel].
  3. nav_command_class()     — parse an L2D task.instruction into a coarse strategic class
                               (the label set the REF-B strategic head trains on).
  4. label_entropy()         — effective-#classes of a command distribution (the metric that
                               quantifies comma2k19's `follow`-starvation vs L2D).
  5. build_contract_row()    — assemble ONE contract-shaped step dict from an L2D row.

Measured L2D schema (probe_l2d_taxonomy.py, real HF bytes 2026-07-11):
  action.continuous dim=3, action.discrete dim=2, waypoints dim=10, 6 surround cams @1080x1920,
  4,219 distinct task.instructions.  See l2d_taxonomy_result.json.
"""
from __future__ import annotations
import re
import numpy as np

# L2D has no dedicated narrow front; `front_left` is the forward-center-left camera = front proxy.
# (Full loader canonicalizes it to F_REF=266 via D-016 focal_crop_resize with the fixed KIA rig.)
FRONT_CAMERA_KEY = "observation.images.front_left"

# L2D action.continuous is [steer, accel, brake]-style (3-dim). Our contract is [steer, accel].
# accel/brake collapse to a single signed longitudinal channel: accel - brake.
STEER_IDX, ACCEL_IDX, BRAKE_IDX = 0, 1, 2


def map_continuous_action(a3) -> np.ndarray:
    """L2D action.continuous[3] -> contract [steer, accel_signed].

    accel_signed = throttle/accel - brake so a single channel carries longitudinal intent,
    matching comma2k19's derived [steer, accel]. Sign/unit calibration against a real decode is
    pinned as a loader TODO; this fixes the CHANNEL algebra so the contract shape is stable.
    """
    a3 = np.asarray(a3, dtype=np.float32).reshape(-1)
    if a3.shape[0] != 3:
        raise ValueError(f"L2D action.continuous must be dim-3, got {a3.shape[0]}")
    steer = a3[STEER_IDX]
    accel = a3[ACCEL_IDX] - a3[BRAKE_IDX]
    return np.asarray([steer, accel], dtype=np.float32)


# --- strategic-label taxonomy: instruction string -> coarse class -----------------------------
# Order matters: more specific maneuvers are matched before the generic `follow`.
_CLASS_PATTERNS = [
    ("u_turn",       r"\bu-?turn\b"),
    ("roundabout",   r"\broundabout\b"),
    ("turn_left",    r"\bturn left\b|\bbear left\b|\bkeep left\b"),
    ("turn_right",   r"\bturn right\b|\bbear right\b|\bkeep right\b"),
    ("lane_change",  r"\blane change\b|\bchange lane"),
    ("reverse",      r"\breverse\b"),
    ("merge_exit",   r"\bexit\b|\bmerge\b|\btake the\b.*\bexit\b"),
    ("follow",       r"\bgo straight\b|\bcontinue\b|\bfollow\b"),
]
_CLASS_NAMES = [c for c, _ in _CLASS_PATTERNS] + ["other"]


def nav_command_class(instruction: str) -> str:
    """Coarse strategic class for an L2D task.instruction (the REF-B strategic-head target).

    A compositional instruction ("go straight ... and turn right ...") is labeled by its FIRST
    non-`follow` maneuver if one is present, else `follow` — i.e. the decisive strategic action.
    """
    s = (instruction or "").lower()
    # roundabout is a dominant maneuver primitive ("exit the roundabout" is a roundabout action,
    # not a generic highway exit) -> override.
    if re.search(r"\broundabout\b", s):
        return "roundabout"
    # otherwise the decisive strategic action = the EARLIEST-occurring non-follow maneuver in the
    # string (what the ego must do next): "go straight ... and turn right" -> turn_right, but
    # "reverse out ... and turn left" -> reverse.
    best_name, best_pos = None, len(s) + 1
    for name, pat in _CLASS_PATTERNS:
        if name == "follow":
            continue
        m = re.search(pat, s)
        if m and m.start() < best_pos:
            best_name, best_pos = name, m.start()
    if best_name is not None:
        return best_name
    if re.search(r"\bgo straight\b|\bcontinue\b|\bfollow\b", s):
        return "follow"
    return "other"


def label_entropy(classes) -> float:
    """Effective number of classes = exp(Shannon entropy) of a class distribution.

    ~1.0 for a `follow`-only stream (comma2k19's starvation), rising toward the class count as
    the command distribution diversifies (L2D). This is the metric §4.4 of the survey uses to
    quantify the REF-B supervision gap.
    """
    vals, counts = np.unique(np.asarray(list(classes)), return_counts=True)
    p = counts / counts.sum()
    ent = float(-(p * np.log(p)).sum())
    return float(np.exp(ent))


def build_contract_row(row: dict) -> dict:
    """Assemble one contract-shaped step dict from an L2D parquet row (no image decode here).

    Returns the action + strategic label + route channel in our conventions; the loader adds the
    focal-canonicalized front-cam stack. Raises loudly on a schema mismatch (fail-loud ingest).
    """
    if "action.continuous" not in row:
        raise ValueError("row missing action.continuous — not an L2D step row")
    action = map_continuous_action(row["action.continuous"])
    instr = row.get("task.instructions")
    if isinstance(instr, dict):  # tasks.jsonl packs {'task_index':..,'__index_level_0__': str}
        instr = instr.get("__index_level_0__", "")
    wp = np.asarray(row.get("observation.state.waypoints", []), dtype=np.float32).reshape(-1)
    return {
        "action": action,                       # [steer, accel] contract
        "nav_class": nav_command_class(instr),  # strategic-head target
        "instruction": instr,
        "waypoints": wp,                         # dim-10 route channel
        "front_camera_key": FRONT_CAMERA_KEY,
    }
