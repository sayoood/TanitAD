"""CoT term extraction for the v7 tactical goals that geometry cannot see.

Every result carries `provenance="vlm-cot"` and stays `disputed` — the CoT is a
generative model's claim (temperature 0.6, ONE draw per clip), measured at
3 correct / 2 wrong on visually checkable statements (RETRACTION_LOG C136/C139).

⚠️ WHAT IS AND IS NOT AVAILABLE. Alpamayo's raw records carry
`answer|box|cot|cot_auto_labeling|meta_action|raw_outputs`, but only
**`cot`, `lane`, `lateral`, `longitudinal`** were exported per clip.
`meta_action` is NOT reachable locally — using it needs the source parquet. All
extraction here is therefore from the CoT sentence alone.

⭐ MEASURED YIELDS over the 4,729-clip corpus (this is what each token can
actually be populated from):

    yield                400  8.5 %   "yield due to pedestrians in the crosswalk"
    parked               419  8.9 %   -> EVADE (static obstacle)
    pedestrian           278  5.9 %
    traffic light        638 13.5 %   green 422 / red 186 / yellow 28
    oncoming             142  3.0 %   clearance-dominant, see REACT_ON_ONCOMING
    gap                   68  1.4 %   "create a gap in the adjacent lane"
    cyclist               64  1.4 %
    slower traffic        56  1.2 %   -> OVERTAKE (moving obstacle)
    merge                 45  1.0 %
    speed limit           41  0.9 %
    ramp                  26  0.5 %
    exit                  17  0.4 %
    overtake (explicit)   13  0.3 %
    open door              0  0.0 %   ⚠️ the PI's door case does NOT occur
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

# --- traffic light ----------------------------------------------------------
_LIGHT = re.compile(r"\btraffic light|\bstop light|\bsignal\b")
_LIGHT_COLOUR = (("RED", re.compile(r"\bred\b")),
                 ("YELLOW", re.compile(r"\byellow\b|\bamber\b")),
                 ("GREEN", re.compile(r"\bgreen\b")))
# --- oncoming: PI — react on it, do not require "wait" ----------------------
_ONCOMING = re.compile(r"\boncoming\b")
# --- yield: 400 clips, the largest semantic class after the light -----------
_YIELD = re.compile(r"\byield")
_YIELD_SIGN = re.compile(r"\byield sign\b")
# --- merge / gap ------------------------------------------------------------
_MERGE = re.compile(r"\bmerg(?:e|es|ing)\b")
_GAP = re.compile(r"\b(?:create|usable|find)\s+(?:a\s+)?gap\b|\bgap\b.*\blane\b")
# --- exit / ramp: PI — take it from terms, not geometry ---------------------
_EXIT = re.compile(r"\bexit\b|\boff-?ramp\b|\bramp\b|\bsplit to the (left|right)\b")
_EXIT_SIDE = (("RIGHT", re.compile(r"\bright\b")), ("LEFT", re.compile(r"\bleft\b")))
# --- overtake vs evade: the PI's distinction --------------------------------
#: OVERTAKE = passing a SLOWER MOVING vehicle in front.
_OVERTAKE = re.compile(
    r"\bovertak(?:e|ing|es)\b"
    r"|\b(?:pass(?:ing|es)?|lane change)\b[^.]*\bslow(?:er|-moving)\b"
    r"|\bpass(?:ing|es)?\s+(?:the|a|an)\s+(?!parked)"
    r"(?:car|vehicle|truck|bus|van)\s+ahead\b")
#: EVADE = a lateral manoeuvre around a STATIC obstacle or a VRU.
_EVADE_VERB = re.compile(r"\bnudge\b|\bincrease clearance\b|\bshift\b|\bmove over\b")
_EVADE_OBJ = (("PARKED", re.compile(r"\bparked\b")),
              ("CYCLIST", re.compile(r"\bcyclist|\bbicycle|\bbike\b")),
              ("PEDESTRIAN", re.compile(r"\bpedestrian")),
              ("DOOR", re.compile(r"\b(?:open |car )?door\b")),
              ("ONCOMING", re.compile(r"\boncoming\b")))
_SPEED_LIMIT = re.compile(r"\bspeed limit\b")


@dataclass
class CotTokens:
    traffic_light: str | None = None     # RED|YELLOW|GREEN|UNKNOWN
    oncoming: bool = False
    yield_: str | None = None            # SIGN | HAZARD
    merge: bool = False
    gap: bool = False
    exit_side: str | None = None         # LEFT | RIGHT | UNKNOWN
    overtake: bool = False
    evade_obj: str | None = None         # PARKED|CYCLIST|PEDESTRIAN|DOOR|ONCOMING
    speed_limit: bool = False
    evidence: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def extract(cot: str | None) -> CotTokens:
    if not cot or not cot.strip():
        return CotTokens()
    t = cot.lower()
    c = CotTokens(evidence=cot)

    if _LIGHT.search(t):
        c.traffic_light = "UNKNOWN"
        for name, pat in _LIGHT_COLOUR:
            if pat.search(t):
                c.traffic_light = name
                break
    c.oncoming = bool(_ONCOMING.search(t))
    if _YIELD.search(t):
        c.yield_ = "SIGN" if _YIELD_SIGN.search(t) else "HAZARD"
    c.merge = bool(_MERGE.search(t))
    c.gap = bool(_GAP.search(t))
    if _EXIT.search(t):
        c.exit_side = "UNKNOWN"
        for name, pat in _EXIT_SIDE:
            if pat.search(t):
                c.exit_side = name
                break
    # ⚠️ ORDER MATTERS. Overtake is tested FIRST because "pass the parked car"
    # and "overtake the slower truck" share the verb; only the OBJECT separates
    # them, and mis-ordering files 419 parked-car evasions as overtakes.
    c.overtake = bool(_OVERTAKE.search(t))
    if not c.overtake and _EVADE_VERB.search(t):
        for name, pat in _EVADE_OBJ:
            if pat.search(t):
                c.evade_obj = name
                break
    c.speed_limit = bool(_SPEED_LIMIT.search(t))
    return c


def goals_from_cot(cot: str | None) -> dict[str, dict]:
    """v7 tactical goal tokens this CoT supports, with their args.

    ⛔ Every token here is in `TACTICAL_GOAL_NEEDS_PERCEPTION`: a geometry
    emitter must never produce them, and a consumer must treat them as
    `disputed` until grounded in image space.
    """
    c = extract(cot)
    out: dict[str, dict] = {}
    if c.traffic_light:
        tok = ("TRAFFIC_LIGHT_REACT" if c.traffic_light == "UNKNOWN"
               else f"TRAFFIC_LIGHT_REACT_{c.traffic_light}")
        out[tok] = {"state": c.traffic_light.lower()}
    if c.oncoming:
        out["REACT_ON_ONCOMING"] = {"oncoming_slot": None}
    if c.yield_:
        out["YIELD"] = {"reason": c.yield_.lower()}
    if c.merge:
        out["MERGE"] = {"agent_slot": None}
    if c.gap:
        out["GAP_TARGET"] = {"agent_slot": None, "time_gap_s": None}
    if c.exit_side:
        side = c.exit_side if c.exit_side != "UNKNOWN" else "RIGHT"
        out[f"TAKE_EXIT_{side[0]}"] = {"side": side.lower()}
    if c.overtake:
        out["OVERTAKE_VEHICLE"] = {"agent_slot": None}
    if c.evade_obj:
        out["EVADE_IN_CORRIDOR"] = {"obstacle_class": c.evade_obj.lower()}
    return out
