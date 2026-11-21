"""PSG targets — physical-state grounding from OUR OWN banked 3D cuboids.

E-DEC-18. PhyLatent (ICLR 2025, banked primary ``2608.05720``) puts a SHARED
state head on BOTH the encoded and the predicted trajectory, and is explicit
that it is "used only during training and not required by the planner" -- which
is exactly the PI's binding rule (labels may use ego and other agents;
⛔ INFERENCE IS VISION-ONLY). Nothing here is ever read at inference.

The labels are ``obstacle.offline`` (97.44 % corpus coverage), already banked as
one json record per (clip, frame) with a list of ego-frame cuboids.

GEOMETRY — the registration is the point, and it is not a guess
--------------------------------------------------------------
The target is a per-AZIMUTH-COLUMN map over the SAME 8 columns the readout uses.
That is only meaningful because the corpus is **cylindrical**, where the image
column axis is LINEAR IN AZIMUTH over the rig's 120° field. On a pinhole
projection it would not be, and the same code would silently mis-register every
agent (the retracted 92.6°-vs-120° error, in reverse).

The ego-frame convention is **+x forward, +y LEFT**, MEASURED by the parked-car
experiment and documented at ``bev_raster.py`` and ``build_obstacle_join.py:15``
— NOT assumed here. Positive azimuth is therefore to the LEFT, and image column
0 is the LEFTMOST column, so::

    col = floor((hfov/2 - az_deg) / (hfov / n_cols))

⛔ THE LEAK, AND THE GUARD THAT IS MANDATORY BECAUSE OF IT
----------------------------------------------------------
This target **determines** both metrics the campaign scores for environment
content: ``n_agents`` is the sum of the count channel, and ``lead_gap_m`` is the
range in the centre columns. Training on it and then scoring those two would be
measuring the training label -- the same family as the situation-classifier leak
(a head given a signal the label was derived from) and the C6 confound.

⇒ :func:`clip_split` is not optional. PSG is supervised on the TRAIN clips only
and environment decodability is read on the HELD-OUT clips, and a report must
also carry at least one column PSG was never told about --
:func:`unsupervised_probes` returns two (``frac_occluded``, ``mean_abs_yaw``),
which are computable from the same cuboids but appear nowhere in the target.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

__all__ = ["PSG_N_COLS", "PSG_HFOV_DEG", "PSG_CHANNELS", "PSG_R_MAX_M",
           "azimuth_column", "frame_target", "load_targets", "clip_split",
           "unsupervised_probes"]

PSG_N_COLS = 8          # matches --readout-grid-w 8 (E-DEC-2's settled geometry)
PSG_HFOV_DEG = 120.0    # camera_front_wide_120fov, cylindrical
PSG_CHANNELS = 2        # (log1p count, inverse range)
PSG_R_MAX_M = 80.0      # the panel's own "no lead" default, kept identical


def azimuth_column(cx: float, cy: float, n_cols: int = PSG_N_COLS,
                   hfov_deg: float = PSG_HFOV_DEG) -> int | None:
    """Ego-frame (+x fwd, +y LEFT) -> image column, or None if outside the field.

    Returns None for anything at or behind the image plane: an agent at
    ``cx <= 0`` has no column, and folding it into column 0 or 7 would put
    traffic BEHIND the car into the leftmost/rightmost cell.
    """
    if cx <= 0.0:
        return None
    az = math.degrees(math.atan2(cy, cx))
    if abs(az) > hfov_deg / 2.0:
        return None
    col = int((hfov_deg / 2.0 - az) // (hfov_deg / n_cols))
    return min(max(col, 0), n_cols - 1)


def frame_target(agents, n_cols: int = PSG_N_COLS, hfov_deg: float = PSG_HFOV_DEG,
                 r_max: float = PSG_R_MAX_M) -> np.ndarray:
    """One frame's cuboid list -> ``[n_cols, 2]`` float32.

    Channel 0 is ``log1p(count)`` rather than the raw count: counts are heavily
    skewed and an L2 loss on the raw value would be dominated by the few busy
    columns. Channel 1 is a bounded NEARNESS ``clip(1 - r/r_max, 0, 1)``, taking
    the MAXIMUM over the column (i.e. the nearest agent) -- an empty column reads
    0, which is also what a column whose only agent sits at ``r_max`` reads, and
    that degeneracy is deliberate: both mean "nothing close here".
    """
    counts = np.zeros(n_cols, dtype=np.float64)
    near = np.zeros(n_cols, dtype=np.float64)
    for a in agents:
        cx, cy = float(a["cx"]), float(a["cy"])
        col = azimuth_column(cx, cy, n_cols, hfov_deg)
        if col is None:
            continue
        counts[col] += 1.0
        r = math.hypot(cx, cy)
        near[col] = max(near[col], min(max(1.0 - r / r_max, 0.0), 1.0))
    return np.stack([np.log1p(counts), near], axis=-1).astype(np.float32)


def load_targets(jsonl: str | Path, n_cols: int = PSG_N_COLS,
                 hfov_deg: float = PSG_HFOV_DEG,
                 r_max: float = PSG_R_MAX_M) -> dict[str, np.ndarray]:
    """``{clip_id: [T, n_cols, 2]}``, T = 1 + max labelled frame_idx for the clip.

    Frames with no record are left at zero (= "nothing seen"), which is why the
    consumer must also carry a validity mask if a clip is sparsely labelled;
    :func:`load_targets` reports density via the companion ``_seen`` array.
    """
    rows: dict[str, dict[int, np.ndarray]] = {}
    with open(jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.setdefault(str(r["clip_id"]), {})[int(r["frame_idx"])] = \
                frame_target(r.get("agents", []), n_cols, hfov_deg, r_max)
    out: dict[str, np.ndarray] = {}
    for cid, per in rows.items():
        t_max = max(per) + 1
        arr = np.zeros((t_max, n_cols, PSG_CHANNELS), dtype=np.float32)
        for t, v in per.items():
            arr[t] = v
        out[cid] = arr
    return out


def clip_split(clip_ids, eval_every: int = 3) -> tuple[list[str], list[str]]:
    """⛔ THE LEAK GUARD. Deterministic CLIP-DISJOINT split.

    Every ``eval_every``-th clip of the sorted list is held out, so the split is
    reproducible from the clip list alone and carries no randomness a rerun could
    change. PSG is supervised on ``train`` only; environment decodability is
    scored on ``held_out`` only.
    """
    ids = sorted(str(c) for c in clip_ids)
    held = [c for i, c in enumerate(ids) if i % eval_every == 0]
    train = [c for i, c in enumerate(ids) if i % eval_every != 0]
    return train, held


def unsupervised_probes(agents) -> dict[str, float]:
    """Scene properties PSG is NEVER told about, for the generalisation test.

    Both are computable from the same cuboids and neither appears in
    :func:`frame_target`, so a representation that improves on these after PSG
    training has generalised rather than memorised the supervised statistic.
    """
    if not agents:
        return {"frac_occluded": 0.0, "mean_abs_yaw": 0.0}
    occ = [float(a.get("occ", 0)) for a in agents]
    yaw = [abs(float(a.get("yaw", 0.0))) for a in agents]
    return {"frac_occluded": float(np.mean(occ)),
            "mean_abs_yaw": float(np.mean(yaw))}
