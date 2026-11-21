"""Alpamayo's STRUCTURED analyses — typed manoeuvres with explicit durations.

⚠️ **THIS DOES NOT REPLACE `meta_action` — I CLAIMED IT WOULD, THEN MEASURED.**
Written first as "the primary CoT extractor"; the measurement says otherwise and
the claim is corrected here rather than left standing. On lateral agreement with
geometry over our tactical band (n=1,670 clips at our 8.0 s anchor):

| source | agreement | chance | lift |
|---|---|---|---|
| `meta_action` lateral axis | **69.9 %** | 41.6 % | **x1.68** |
| structured band tokens | 55.2 % | 47.7 % | x1.16 |

`meta_action` wins because the structured segments are dominated by
`keep lane`, which raises the chance baseline to 47.7 % and leaves little room
above it.

⭐ **SO WHAT IS THIS SOURCE FOR?** Not the three axes `meta_action` already
covers — for everything it CANNOT express: **timing** (which band a manoeuvre
falls in), **typed objects** (EVADE / TAKE_EXIT / YIELD / STOP / CREEP as
distinct tokens rather than a `Steer Right`), and the **explicit clean
negative** below. Use `meta_action` for the axes and this for the vocabulary.

`auto_labeling.ego_vehicle_motion_analysis` is not prose. It is a numbered list
of typed segments with time spans:

    1. type: keep lane,
       duration: 0-4 seconds,
       motion: keep lane and decelerate while approaching a red traffic light.
    2. type: stop,
       duration: 4-8 seconds,
       motion: stop at the stop line and wait for the red light.

MEASURED 2026-08-23: present on 2,943 of 4,729 clips and **97.1 % of those parse
into segments** (2,857). 1,746 clips give 2 segments, 845 give 3, 54 give 4.
The `type:` vocabulary (121 distinct) maps almost one-to-one onto v7 — `nudge to
the left` is EVADE_IN_CORRIDOR, `split to the right` is TAKE_EXIT_R,
`stop/yield` is YIELD, `adapt speed` is ADAPT_SPEED_FOR_CURVE.

`critical_components_analysis` is the same shape and names the OBJECT that
forces the behaviour (265 distinct, parsed on 2,888):

    none 853 · lead vehicle 800 · straight traffic light 255 ·
    oncoming vehicle 105 · right-hand road curvature 79 · stop sign 29 ·
    roundabout yield sign 35 · yield sign 20

⭐ **`type: none` (853 clips) is an EXPLICIT CLEAN NEGATIVE** — Alpamayo saying
"nothing is critical here". Regex over prose can never produce that; absence of
a match is not a claim of absence. This is the first source in the programme
that distinguishes *"no obstacle"* from *"no statement about obstacles"*.

## ⚠️ THE TIME BASE — GET THIS WRONG AND EVERY BAND IS SHIFTED 2.9 s

Segment durations are relative to **Alpamayo's t0 = 5.1 s**, not our s2 anchor
at 8.0 s. Converting to our anchor-relative time:

| Alpamayo span | absolute clip time | relative to OUR 8.0 s anchor |
|---|---|---|
| 0–4 s | 5.1–9.1 s | **−2.9 … +1.1 s** (mostly the PAST) |
| 4–8 s | 9.1–13.1 s | **+1.1 … +5.1 s** (our tactical band) |
| 0–2 s | 5.1–7.1 s | −2.9 … −0.9 s (entirely past) |

So the segment that describes OUR tactical horizon is usually the SECOND one,
and a naive reading of segment 1 as "what happens next" describes a moment that
has already passed by our anchor. `segments_for_band()` does this conversion;
nothing else should do it by hand.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import alpamayo_records as AR

#: Numbered segment: "1. type: X, duration: A-B seconds, motion: ..."
_SEG = re.compile(
    r"(\d+)\.\s*type:\s*([^,\n]+),\s*duration:\s*([0-9.]+)\s*-\s*([0-9.]+)\s*"
    r"seconds?,\s*motion:\s*([^\n]*(?:\n(?!\s*\d+\.)[^\n]*)*)", re.I)
_COMP = re.compile(r"type:\s*([^,\n]+),\s*why it is critical:\s*(.+)", re.I | re.S)

#: Alpamayo segment `type:` -> (lateral token, longitudinal token) in v7 terms.
#: Only the head of the phrase is matched, longest-first, so "keep lane and
#: accelerate" resolves on both axes rather than falling through to the default.
_LAT_RULES: tuple[tuple[str, str], ...] = (
    ("nudge to the left", "EVADE_IN_CORRIDOR"),
    ("nudge to the right", "EVADE_IN_CORRIDOR"),
    ("split to the right", "TAKE_EXIT_R"),
    ("split to the left", "TAKE_EXIT_L"),
    ("turn right", "TURN_R"),
    ("turn left", "TURN_L"),
    ("merge", "MERGE"),
    ("change lane to the left", "LANE_CHANGE_L"),
    ("change lane to the right", "LANE_CHANGE_R"),
    ("keep lane", "FOLLOW_LANE"),
    ("keep distance", "FOLLOW_LANE"),
)
_LON_RULES: tuple[tuple[str, str], ...] = (
    ("stop and wait", "STOP"),
    ("stop/yield", "YIELD"),
    ("stop", "STOP"),
    ("slow down", "BRAKE_TO"),
    ("adapt speed", "ADAPT_SPEED_FOR_CURVE"),
    ("creep forward", "CREEP"),
    ("resume speed", "ACCELERATE"),
    ("accelerate", "ACCELERATE"),
    ("keep distance", "FOLLOW_LEAD"),
    ("wait behind the lead vehicle", "FOLLOW_LEAD"),
)

#: `critical_components_analysis` type -> the grounding kind it should carry.
_COMP_KIND: tuple[tuple[str, str], ...] = (
    ("traffic light", "traffic_light"),
    ("lead vehicle", "vehicle"),
    ("oncoming", "vehicle"),
    ("cross-traffic", "vehicle"),
    ("pedestrian", "vru"),
    ("cyclist", "vru"),
    ("bicycle", "vru"),
)


@dataclass(frozen=True)
class MotionSegment:
    """One typed segment of Alpamayo's motion analysis."""

    index: int
    raw_type: str
    t0_alpamayo: float        # seconds from Alpamayo's own anchor
    t1_alpamayo: float
    motion: str

    @property
    def t0_rel(self) -> float:
        """Start, relative to OUR s2 anchor (negative = already past)."""
        from .egomotion_source import RAW_T0_S
        return self.t0_alpamayo + AR.ALPAMAYO_T0_S - RAW_T0_S

    @property
    def t1_rel(self) -> float:
        from .egomotion_source import RAW_T0_S
        return self.t1_alpamayo + AR.ALPAMAYO_T0_S - RAW_T0_S

    def tokens(self) -> tuple[str | None, str | None]:
        """(lateral token, longitudinal token) in v7 terms, either may be None."""
        t = self.raw_type.strip().lower()
        lat = next((v for k, v in _LAT_RULES if k in t), None)
        lon = next((v for k, v in _LON_RULES if k in t), None)
        # The `motion:` sentence carries the axis the `type:` omits often
        # enough to be worth reading — "keep lane and decelerate" types as
        # `keep lane` but states the longitudinal action in the motion text.
        if lon is None:
            m = self.motion.lower()
            lon = next((v for k, v in _LON_RULES if k in m), None)
        return lat, lon

    @property
    def lateral_side(self) -> str | None:
        """``left`` | ``right`` | ``straight``, or None when the segment makes no
        lateral claim — read from Alpamayo's OWN segment type.

        ⛔ WHY THIS EXISTS (PI 2026-08-30: *"leverage more the cot and reasoning
        of Alpamayo"*). `_LAT_RULES` maps BOTH ``nudge to the left`` AND ``nudge
        to the right`` onto the single unsided token ``EVADE_IN_CORRIDOR``, so
        **the direction Alpamayo explicitly stated was discarded** — on 130
        segments (91 left / 39 right). Every other sided type keeps its side
        (`turn right`->TURN_R, `split to the right`->TAKE_EXIT_R,
        `change lane to the left`->LANE_CHANGE_L); nudge was the one that lost it.

        ⚠️ The v7 vocabulary is FROZEN, so this does NOT mint EVADE_IN_CORRIDOR_L/R.
        The token is unchanged; the side travels beside it, the same way
        `EVADE_IN_CORRIDOR` already carries an ``obstacle_class`` argument
        (`cot_tokens_v7.py:276`).
        """
        t = self.raw_type.strip().lower()
        m = re.search(r"\b(?:nudge|shift|move over|split|merge|turn|change lane)\b"
                      r"[^.,;]*?\bto the (left|right)\b", t)
        if m:
            return m.group(1)
        lat, _ = self.tokens()
        if lat:
            if lat.endswith("_L"):
                return "left"
            if lat.endswith("_R"):
                return "right"
            return "straight"
        return None


@dataclass(frozen=True)
class CriticalComponent:
    """What Alpamayo says forces the behaviour, and why."""

    raw_type: str
    why: str

    @property
    def is_none(self) -> bool:
        """⭐ An EXPLICIT statement that nothing is critical (853 clips)."""
        return self.raw_type.strip().lower() in ("none", "no critical component",
                                                 "n/a", "-")

    @property
    def kind(self) -> str | None:
        """The grounding kind a box would have to show to corroborate this."""
        t = self.raw_type.lower()
        return next((v for k, v in _COMP_KIND if k in t), None)


def motion_segments(clip_id: str) -> list[MotionSegment]:
    """Parsed segments, in order. Empty when absent or unparseable."""
    c = AR.get(clip_id)
    if not c or not c.motion_analysis:
        return []
    out = []
    for idx, ty, a, b, motion in _SEG.findall(c.motion_analysis):
        try:
            t0, t1 = float(a), float(b)
        except ValueError:
            continue
        out.append(MotionSegment(int(idx), ty.strip(), t0, t1,
                                 " ".join(motion.split())))
    return out


#: Each entry of a NUMBERED components list: "1. Type: X, Why it is critical: Y"
_COMP_ALL = re.compile(
    r"type:\s*([^,\n]+?)[,\n]\s*why it is critical:\s*"
    r"(.*?)(?=\n\s*\d+\.\s*type:|\Z)", re.I | re.S)


def critical_components(clip_id: str) -> list[CriticalComponent]:
    """EVERY critical component, not just the first.

    ⛔ MEASURED 2026-08-27 (PI review of `59b57590`): `components_analysis` is a
    NUMBERED LIST on **197 clips** (2-6 entries), and only the first was read.
    On that clip we kept `Lane divider (dashed center line)` and discarded
    `Parked vehicles along the right curb`, `Traffic lights ahead` and
    `Lead vehicle far ahead` — three entries that map to tokens, thrown away in
    favour of a road marking.

    Same family as reading only the first match of a repeated structure; the
    single-entry `critical_component()` below is kept for callers that genuinely
    want the primary one, and now documents that it is a CHOICE.
    """
    c = AR.get(clip_id)
    if not c or not c.components_analysis:
        return []
    return [CriticalComponent(m.group(1).strip(), " ".join(m.group(2).split()))
            for m in _COMP_ALL.finditer(c.components_analysis)]


def critical_component(clip_id: str) -> CriticalComponent | None:
    """The FIRST critical component. See `critical_components` for all of them.

    ⚠️ On a numbered list the first entry is not necessarily the most
    important — on `59b57590` it is a lane marking while entries 2-4 are the
    parked vehicles, the traffic lights and the lead vehicle.
    """
    all_ = critical_components(clip_id)
    if all_:
        return all_[0]
    c = AR.get(clip_id)
    if not c or not c.components_analysis:
        return None
    m = _COMP.search(c.components_analysis)
    if not m:
        return None
    return CriticalComponent(m.group(1).strip(), " ".join(m.group(2).split()))


def segments_for_band(clip_id: str, lo_rel: float, hi_rel: float,
                      *, min_overlap_s: float = 0.5) -> list[MotionSegment]:
    """Segments overlapping [lo_rel, hi_rel] in OUR anchor-relative seconds.

    ⚠️ Use this rather than indexing segments by position. MEASURED: the
    2.9 s anchor offset puts Alpamayo's FIRST segment (0-4 s) mostly in our
    past (-2.9 … +1.1 s), so "segment 1 is what happens next" is wrong on the
    majority of clips. `min_overlap_s` refuses a segment that only grazes the
    band, which otherwise lets a 0.1 s sliver decide a tactical label.
    """
    out = []
    for s in motion_segments(clip_id):
        ov = min(s.t1_rel, hi_rel) - max(s.t0_rel, lo_rel)
        if ov >= min_overlap_s:
            out.append(s)
    return out


def band_tokens(clip_id: str, lo_rel: float, hi_rel: float) -> dict:
    """What Alpamayo claims happens in a band, as v7 tokens plus provenance."""
    segs = segments_for_band(clip_id, lo_rel, hi_rel)
    lat: list[str] = []
    lon: list[str] = []
    for s in segs:
        a, b = s.tokens()
        if a and a not in lat:
            lat.append(a)
        if b and b not in lon:
            lon.append(b)
    comp = critical_component(clip_id)
    # ⭐ The banded lateral SIDE — Alpamayo's reasoning as a second, time-aligned
    # lateral channel, independent of its `meta_action` label. MEASURED
    # 2026-08-30 over 2,654 clips: the two channels agree on 64 % of clips, and
    # where they AGREE the claim matches geometry 74.7 % of the time against a
    # 65 % baseline; where they CONFLICT it is a coin flip (meta right 459,
    # reasoning right 404). So concordance is a usable confidence signal and
    # neither channel dominates the other.
    sides = {s.lateral_side for s in segs} - {None}
    directional = sides - {"straight"}
    if len(directional) == 1:
        band_side = directional.pop()
    elif not directional and sides:
        band_side = "straight"
    else:
        band_side = None                    # silent on a mixed or empty band
    return {
        "lateral": lat,
        "lateral_side": band_side,
        "lateral_sides_seen": sorted(sides),
        "longitudinal": lon,
        "n_segments": len(segs),
        "spans": [(round(s.t0_rel, 1), round(s.t1_rel, 1)) for s in segs],
        "raw_types": [s.raw_type for s in segs],
        "critical": None if comp is None else comp.raw_type,
        "critical_is_none": bool(comp and comp.is_none),
        "provenance": "alpamayo-structured",
    }


def coverage() -> dict:
    """How much of the corpus this source can actually speak for."""
    d = AR._load()
    n_motion = n_parsed = n_comp = n_none = 0
    for cid, c in d.items():
        if c.motion_analysis:
            n_motion += 1
            if motion_segments(cid):
                n_parsed += 1
        comp = critical_component(cid)
        if comp:
            n_comp += 1
            if comp.is_none:
                n_none += 1
    return {"clips": len(d), "with_motion_analysis": n_motion,
            "parsed_segments": n_parsed, "with_component": n_comp,
            "component_is_none": n_none}
