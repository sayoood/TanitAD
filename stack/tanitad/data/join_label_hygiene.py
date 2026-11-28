"""refcv8 WP-C fixes 3 and 4 -- two label defects of the agent join, MASKED at read time, OPT-IN.

Source: ``TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D3_raw_plausibility`` (RESULT.md
DEFECT D-2 and the MINOR "Boxes" row; ``raw/c3b_agents_detail.json``, script ``scripts/c3b_agents_detail.py``).

* **D-2 -- the EGO as an agent box.** 18 TRAIN clips, 693 frames (0 EVAL): a box whose centre lies in x in (-1, 4) m,
  |y| < 1 m (rig frame, rear-axle origin) -- 84.27 % of them within 0.5 m of (1.4, 0) with the ego's own footprint
  (~4.6 x 2.0 m). It teaches "a car at the ego position" and is a permanent collision for an oracle no-collision
  scorer. FIX: remove those rows from the listed (clip, frame) records, BEFORE anything is derived from them, so the 2-D
  rows, the classes, the track ids, the rates and the occupancy raster all lose the SAME row (they are aligned by
  position, and a mask applied to one of them would silently shift the others).
* **Track-id switches.** Displacement > 5 m (vehicles) / 2 m (persons, strollers) / 3 m (others) per ~0.1 s in the WORLD
  frame on 0.028 % of track steps (6,410 frames, 887 clips; 91 % single-track events, up to 276 m): the join re-uses an
  id for another object, and ``agent_slots.track_rates_from_join`` -- an unclipped central difference by track id --
  turns the jump into a ~10^2..10^3 m/s relative-rate target. FIX: MASK the rate row (``rates_mask`` False and the value
  zeroed -- "a missing rate is MASKED, never zero-filled", the reader's own rule) at the two records the jump
  contaminates: a jump between frames f-1 and f enters the central difference of BOTH record f-1 (neighbours f-2, f) and
  record f (neighbours f-1, f+1), and no other. Frames in which EVERY track moves by the same vector (the D3
  ``pose_glitch_like`` class, 853 frames: ``median_all_disp >= 0.5 * max_jump``) are a POSE defect, not an id switch --
  the join's boxes are in the ego frame, so their ego-frame rates are smooth -- and are NOT listed.

WHY A FRAME LIST AS DATA, NOT A RULE AT READ TIME. The jump test needs the ego's world poses (the join carries none), and
the ego test is only meaningful in the frames D3 audited. The list is produced offline by
``stack/scripts/mine_join_label_defects.py`` from the SAME join and manifest, and is KEYED BY sha12 of the clip id
(``clip_key``) -- no raw clip id is stored anywhere.

DEFAULT: nothing is masked. ``JoinFileReader(..., defect_masks=None)`` is the pre-change reader, byte for byte
(``stack/tests/test_refcv8_join_label_hygiene.py`` pins the arrays of a default read against the unmodified tip reader).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

__all__ = ["SCHEMA", "EGO_X_RANGE_M", "EGO_ABS_Y_M", "JUMP_M", "POSE_GLITCH_FRAC", "VEHICLE_CLASSES", "PERSON_CLASSES",
           "clip_key", "track_key", "class_kind", "jump_threshold_m", "is_ego_footprint", "JoinDefectMasks"]

SCHEMA = "tanitad.join_label_defects/1"

#: D3 ``c3b_agents_detail.py`` rule (1), STRICT inequalities, rig frame (+x forward, +y left, rear-axle origin).
EGO_X_RANGE_M: tuple[float, float] = (-1.0, 4.0)
EGO_ABS_Y_M: float = 1.0
VEHICLE_CLASSES = frozenset({"automobile", "heavy_truck", "trailer", "bus", "other_vehicle", "train_or_tram_car"})
PERSON_CLASSES = frozenset({"person", "stroller"})
#: D3 ``c3b`` JUMP: world displacement per ~0.1 s above which a same-id step is a jump, by class kind
#: (v vehicles, p person/stroller, o everything else -- ``rider`` included).
JUMP_M: dict = {"v": 5.0, "p": 2.0, "o": 3.0}
#: a frame with ``median_all_disp >= POSE_GLITCH_FRAC * max_jump`` moves EVERY track: a pose defect, not an id switch.
POSE_GLITCH_FRAC: float = 0.5


def clip_key(clip_id) -> str:
    """The sha12 every banked artifact of the programme uses for a clip (``sha256(clip_id)[:12]``)."""
    return hashlib.sha256(str(clip_id).encode()).hexdigest()[:12]


def track_key(track_id) -> str:
    """sha12 of a track id string (the list never stores a track id itself)."""
    return hashlib.sha256(str(track_id).encode()).hexdigest()[:12]


def class_kind(label_class) -> str:
    c = str(label_class)
    return "v" if c in VEHICLE_CLASSES else ("p" if c in PERSON_CLASSES else "o")


def jump_threshold_m(label_class) -> float:
    return float(JUMP_M[class_kind(label_class)])


def is_ego_footprint(cx, cy):
    """D3 rule (1): centre in x in (-1, 4) m and |y| < 1 m. Scalars -> bool, arrays -> bool array."""
    cx = np.asarray(cx, dtype=np.float64)
    cy = np.asarray(cy, dtype=np.float64)
    m = (cx > EGO_X_RANGE_M[0]) & (cx < EGO_X_RANGE_M[1]) & (np.abs(cy) < EGO_ABS_Y_M)
    return bool(m) if m.ndim == 0 else m


class JoinDefectMasks:
    """The two frame lists of ``tanitad.join_label_defects/1`` plus the two read-time operations.

    ``ego_frames``: ``{clip sha12: {frame_idx}}``; ``track_events``: ``{clip sha12: {(frame_idx, track sha12)}}`` where
    an event ``(f, t)`` means "track ``t`` jumped between frame f-1 and frame f". Counters (``n_ego_rows_removed``,
    ``n_rate_rows_masked``) accumulate over the lifetime of the object so a reader can stamp them."""

    def __init__(self, ego_frames: dict, track_events: dict, stamp: dict | None = None):
        self.ego_frames = {str(k): frozenset(int(f) for f in v) for k, v in ego_frames.items()}
        self.track_events = {str(k): frozenset((int(f), str(t)) for f, t in v) for k, v in track_events.items()}
        # record -> tracks whose rate row that record's central difference corrupts: BOTH f-1 and f (module docstring)
        rec: dict = {}
        for ck, evs in self.track_events.items():
            for f, tk in evs:
                for r in (f - 1, f):
                    rec.setdefault((ck, r), set()).add(tk)
        self._rate_records = {k: frozenset(v) for k, v in rec.items()}
        self.stamp = dict(stamp or {})
        self.n_ego_rows_removed = 0
        self.n_ego_frames_hit = 0
        self.n_rate_rows_masked = 0
        self.n_rate_records_hit = 0

    # ----------------------------------------------------------------------------------------- loading -- #
    @classmethod
    def load(cls, path) -> "JoinDefectMasks":
        """Read a ``tanitad.join_label_defects/1`` file. REFUSES another schema, a missing section, and a file that
        carries a raw-looking clip id (a 36-char UUID key) -- the list is sha12-keyed by contract."""
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"join defect masks file {str(p)!r} does not exist")
        raw = p.read_bytes()
        d = json.loads(raw.decode("utf-8"))
        if d.get("schema") != SCHEMA:
            raise ValueError(f"{p.name}: schema {d.get('schema')!r} != {SCHEMA!r}")
        for sec in ("ego_footprint", "track_jumps"):
            if not isinstance(d.get(sec), dict):
                raise ValueError(f"{p.name}: no `{sec}` object")
        ego = d["ego_footprint"].get("frames")
        ev = d["track_jumps"].get("events")
        if not isinstance(ego, dict) or not isinstance(ev, dict):
            raise ValueError(f"{p.name}: `ego_footprint.frames` / `track_jumps.events` must be objects keyed by clip sha12")
        for k in list(ego) + list(ev):
            if len(k) != 12 or any(ch not in "0123456789abcdef" for ch in k):
                raise ValueError(f"{p.name}: key {k[:20]!r} is not a sha12 -- the list is sha12-keyed, never by raw clip id")
        stamp = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "schema": SCHEMA,
                 "n_ego_frames": int(sum(len(v) for v in ego.values())), "n_ego_clips": len(ego),
                 "n_track_events": int(sum(len(v) for v in ev.values())), "n_track_event_clips": len(ev),
                 "provenance": d.get("provenance")}
        return cls({k: v for k, v in ego.items()}, {k: [tuple(e) for e in v] for k, v in ev.items()}, stamp)

    # --------------------------------------------------------------------------------------- operations -- #
    def has_ego_frame(self, clip_id, frame_idx: int) -> bool:
        s = self.ego_frames.get(clip_key(clip_id))
        return bool(s) and int(frame_idx) in s

    def strip_ego_boxes(self, clip_id, frame_idx: int, agents):
        """``agents``: the join record's ``agents`` list of dicts. Returns ``(agents', n_removed)``.

        Only in a LISTED (clip, frame) are the rows with :func:`is_ego_footprint` centres removed; a legitimate box in
        the same frame stays. An unlisted frame returns the SAME list object (no copy, no change)."""
        if not self.has_ego_frame(clip_id, frame_idx):
            return agents, 0
        keep = [d for d in agents if not is_ego_footprint(float(d["cx"]), float(d["cy"]))]
        n = len(agents) - len(keep)
        if n:
            self.n_ego_rows_removed += n
            self.n_ego_frames_hit += 1
        return keep, n

    def rate_mask(self, clip_id, frame_idx: int, agents) -> np.ndarray:
        """``[A] bool``: True where that row's rate target is corrupted by a track-id switch. ``agents`` is the record's
        (already ego-stripped) agent list, in row order."""
        n = len(agents)
        out = np.zeros(n, dtype=bool)
        if not n:
            return out
        tks = self._rate_records.get((clip_key(clip_id), int(frame_idx)))
        if not tks:
            return out
        for i, d in enumerate(agents):
            if track_key(d.get("track_id", "")) in tks:
                out[i] = True
        k = int(out.sum())
        if k:
            self.n_rate_rows_masked += k
            self.n_rate_records_hit += 1
        return out

    def apply_rate_mask(self, clip_id, frame_idx: int, agents, rates: np.ndarray, mask: np.ndarray):
        """Mask (and zero) the corrupted rate rows of ``rates [A, 3]`` / ``mask [A]``; returns new arrays, inputs
        untouched. A row that was already unobserved stays unobserved."""
        bad = self.rate_mask(clip_id, frame_idx, agents)
        if not bad.any():
            return rates, mask
        r2 = np.array(rates, copy=True)
        m2 = np.array(mask, copy=True)
        r2[bad] = 0.0
        m2[bad] = False
        return r2, m2

    def stats(self) -> dict:
        """What a trainer stamps into config.json."""
        return {**{k: v for k, v in self.stamp.items() if k != "provenance"},
                "n_ego_rows_removed": int(self.n_ego_rows_removed), "n_ego_frames_hit": int(self.n_ego_frames_hit),
                "n_rate_rows_masked": int(self.n_rate_rows_masked), "n_rate_records_hit": int(self.n_rate_records_hit)}
