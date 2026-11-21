"""CoT term extraction for the v7 tactical goals that geometry cannot see.

Every result carries `provenance="vlm-cot"` and stays `disputed` — the CoT is a
generative model's claim (temperature 0.6, ONE draw per clip), measured at
3 correct / 2 wrong on visually checkable statements (RETRACTION_LOG C136/C139).

⛔ RETRACTED 2026-08-23 (C142). The sentence *"`meta_action` is NOT reachable
locally — using it needs the source parquet"* was WRONG and is kept here only
so it cannot come back: the source parquet IS local (`alpamayo_records.RECORDS`,
md5 `9f13474723b880eec7fcc09a7be478d8`) and `alpamayo_records.py` reads all
FIVE tasks — `meta_action`, `trajectory`, `auto_labeling`, `vqa`,
`grounding_via_vqa` — over 23,644 rows / 4,729 clips.

⛔⛔ THE YIELD TABLE BELOW IS SCOPED TO ONE TEXT FIELD, AND ITS SCOPE WAS
NOT STATED — WHICH MADE ITS SPEED-LIMIT ROW READ 5.3x SMALLER THAN THE CORPUS.
MEASURED 2026-09-06 (n = 4,729 clips, `records.parquet` md5
`9f13474723b880eec7fcc09a7be478d8`): every row below is the **`cot` field of
the `meta_action` task ALONE** — the two are the same text, `AR.get(cid).cot`.
Across all five tasks the same phrases occur far more often:
`speed limit` 41 ⇒ **219**, `pedestrian` 278 ⇒ 2,115, `traffic light` 636
⇒ 2,032, `yield` 400 ⇒ 688. ⇒ **Read every row as "of the `meta_action` CoT
sentence", never as "of everything we hold".** Same family as the `df` / Thor
`free` / `step_s` traps: a true measurement quoted outside its scope.

⚠️ AND 219 IS THE COUNT OF CLIPS WHERE ALPAMAYO *STATES* THE PHRASE. A
further 40 clips carry it only in the QUESTION PUT TO THE MODEL (the sampled
VQA bank); 219 stated + 16 question-only = the **235** a sibling report
published as the corpus figure. **Being ASKED about a speed limit is not the
corpus stating one**, so 219 is the number a label pipeline can act on.

⭐ MEASURED YIELDS, **`meta_action` CoT sentence**, n = 4,729 clips (this is
what each token can be populated from FROM THIS FIELD):

    yield                400  8.5 %   "yield due to pedestrians in the crosswalk"
    parked               419  8.9 %   -> EVADE (static obstacle)
    pedestrian           278  5.9 %
    traffic light        638 13.5 %   green 422 / red 186 / yellow 28
    oncoming             142  3.0 %   clearance-dominant, see REACT_ON_ONCOMING
    gap                   68  1.4 %   "create a gap in the adjacent lane"
    cyclist               64  1.4 %
    slower traffic        56  1.2 %   -> OVERTAKE (moving obstacle)
    merge                 45  1.0 %
    speed limit           41  0.9 %   ⛔ THIS FIELD ONLY. Corpus-wide 219
                                      STATED (4.63 %) / 235 stated-or-asked;
                                      **57 carry a VALUE** — see
                                      `speed_limit_reading`.
    ramp                  26  0.5 %
    exit                  17  0.4 %
    overtake (explicit)   13  0.3 %
    open door              0  0.0 %   ⚠️ the PI's door case does NOT occur
                                      in this field (3 clips across all five
                                      tasks, so it is rare, not absent)
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from . import cot_negation as NEG

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
#: ⛔ A MERGE THAT IS A NAMED RISK, NOT AN EVENT. MEASURED 2026-08-27 on
#: `59b57590`, which the PI questioned: the ONLY "merg" in the whole text is
#:
#:   "…create potential door-opening/merge hazards; keeping extra clearance
#:    can motivate moving left."
#:
#: — a hypothetical hazard CLASS inside a compound noun, in a sentence
#: explaining why parked cars matter. There is no merge in the scene. Negation
#: scoping cannot catch this: "potential" is not a negation, it is IRREALIS.
#: ⇒ refuse `merge` when it heads a hazard/risk noun phrase or is bound into a
#: slashed compound. 3 of 129 MERGE emissions (2.3 %) are this shape.
_MERGE_HAZARD = re.compile(
    r"\bmerg\w*[-/\s]*(?:hazard|risk|conflict)s?\b"
    r"|\b(?:hazard|risk|conflict)s?\s+(?:of|from)\s+merg\w*"
    r"|[/-]merg\w*", re.I)
_MERGE = re.compile(r"\bmerg(?:e|es|ing)\b")

#: ⭐ THE LANE CHANGE THE PIPELINE NEVER READ. MEASURED 2026-08-27: **174 clips
#: state a lane change in plain language and NOT ONE produced a token** (101
#: left, 73 right) — `LANE_CHANGE_L`/`LANE_CHANGE_R` are in the frozen
#: vocabulary and simply had no extractor. `59b57590` opens with "Change lanes
#: to the left due to the right lane being constrained by parked vehicles" and
#: emitted a phantom MERGE instead of the manoeuvre it actually describes.
_LANE_CHANGE = (
    ("L", re.compile(r"\bchange\s+lanes?\s+to\s+the\s+left\b"
                     r"|\blane\s+change\s+to\s+the\s+left\b"
                     r"|\bmove\s+(?:in)?to\s+the\s+left\s+lane\b"
                     r"|\bmerge\s+(?:in)?to\s+the\s+left\s+lane\b", re.I)),
    ("R", re.compile(r"\bchange\s+lanes?\s+to\s+the\s+right\b"
                     r"|\blane\s+change\s+to\s+the\s+right\b"
                     r"|\bmove\s+(?:in)?to\s+the\s+right\s+lane\b"
                     r"|\bmerge\s+(?:in)?to\s+the\s+right\s+lane\b", re.I)),
)
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
#: ⭐ A STOPPED VEHICLE IS A STATIC OBSTACLE, SO PASSING IT IS **EVADE**, NOT
#: OVERTAKE — the PI's distinction turns on whether the object MOVES.
#: MEASURED 2026-08-24 on `d452ea24`: the CoT reads "Nudge left to pass the
#: stopped bus in the same lane", geometry shows `NUDGE_L`, and the clip was
#: labelled `FOLLOW_LANE` and nothing else — because `bus` was absent from this
#: list, and the OVERTAKE pattern demands "…ahead" immediately after the noun,
#: which "the stopped bus in the same lane" does not supply. Listed FIRST so a
#: stopped vehicle cannot fall through to a weaker class.
_EVADE_OBJ = (("STOPPED_VEHICLE", re.compile(
                  r"\bstopped\s+(?:bus|truck|van|car|vehicle|lorry)\b"
                  r"|\b(?:bus|truck|van|car|vehicle)\s+(?:is\s+)?stopped\b"
                  r"|\bstationary\s+(?:bus|truck|van|car|vehicle)\b"
                  r"|\bdouble-?parked\b")),
              ("PARKED", re.compile(r"\bparked\b")),
              ("CYCLIST", re.compile(r"\bcyclist|\bbicycle|\bbike\b")),
              ("PEDESTRIAN", re.compile(r"\bpedestrian")),
              ("DOOR", re.compile(r"\b(?:open |car )?door\b")),
              ("ONCOMING", re.compile(r"\boncoming\b")))
#: ⛔⛔ A DEAD BOOLEAN THAT THREW AWAY FREE DATA. Until 2026-09-06
#: `CotTokens.speed_limit` was a `bool`, this pattern had NO CAPTURING GROUP,
#: and `goals_from_cot` emitted nothing for it — so every posted-limit VALUE
#: the CoT contains was matched and discarded at the regex. MEASURED over the
#: 4,729 clips of `Sayood/tanitad-alpamayo2-augmentation`: **57 clips state a
#: value read off a sign**, unhedged and un-negated.
#: ⚠️ The boolean's own semantics are UNCHANGED (presence, on the
#: negation-stripped text), because every banked label was built with them.
#: The value is captured ALONGSIDE it, never instead of it.
_SPEED_LIMIT = re.compile(r"\bspeed limit\b")

#: ⛔ A VALUE ASSERTED AS READ OFF A SIGN / GANTRY. Same definition as
#: `—/2026-09-06-speed-limit-source/code/classify_limits.py`, so the two
#: agree clip-for-clip; the harness in this package asserts that they do.
_SPEED_LIMIT_READ = re.compile(
    r"(?:speed[\s\-]?limit\s+sign|limit\s+sign|gantry\s+signs?|sign)\s*"
    r"(?:[a-z,\s]{0,40}?)(?:indicat\w+|display\w*|read\w*|shows?|of|is|posted|:)?\s*"
    r"(\d{1,3})\s*(km/?h|kph|mph)?"
    r"|(?:a|an)\s+(\d{1,3})\s*(km/?h|kph|mph)?\s*speed[\s\-]?limit\s+sign"
    r"|speed[\s\-]?limit\s+(?:of|is|at)\s+(\d{1,3})\s*(km/?h|kph|mph)?", re.I)

#: ⚠️ HEDGING marks a LANGUAGE PRIOR, not a reading. Admitting
#: *"residential areas typically have 25-30 mph"* would be the `road_class`
#: circularity in a new costume, so it is classified and EXCLUDED, not dropped
#: silently.
_SPEED_LIMIT_HEDGE = re.compile(
    r"\btypically\b|\busually\b|\bgenerally\b|\boften\b|\blikely\b|\bprobabl\w+"
    r"|\bcould\s+(?:range|be)\b|\bwould\s+(?:be|likely)\b|\bapproximately\b"
    r"|\bestimate\w*\b|\bassum\w+|\bif\s+not\b|\bdepending\s+on\s+local\b"
    r"|\bnot\s+visible\b|\bnot\s+observed\b|\bexact\s+speed", re.I)

_SPEED_LIMIT_NEG = re.compile(
    r"\b(?:no|not)\b[^.]{0,40}?speed[\s\-]?limit"
    r"|speed[\s\-]?limit[^.]{0,40}?\bnot\s+(?:visible|observed|present|shown)",
    re.I)

#: ⛔ 1 mph = 0.44704 m/s exactly; 1 km/h = 1/3.6 m/s exactly.
#: MEASURED: 70 km/h = 19.44 m/s, 70 mph = 31.29 m/s — a **1.61x** spread,
#: which is why a unit-less reading may NEVER be converted.
_SPEED_LIMIT_MS = {"kph": 1.0 / 3.6, "mph": 0.44704}


def _norm_speed_unit(u: str | None) -> str | None:
    """`km/h`|`kmh`|`kph` -> `kph`; `mph` -> `mph`; anything else -> None."""
    if not u:
        return None
    u = u.strip().lower().replace("/", "")
    if u in ("kmh", "kph"):
        return "kph"
    return "mph" if u == "mph" else None


@dataclass
class SpeedLimitReading:
    """One posted-speed-limit claim, with the unit it was stated in.

    ⛔ **A NUMBER WITHOUT ITS UNIT IS INADMISSIBLE**, so `unit_missing` is
    carried explicitly and `value_ms` is **None** whenever it is True. MEASURED
    2026-09-06 over the 4,729-clip corpus: **18 of the 57 clip-level readings
    (31.6 %) state a bare number** with no unit at all. Guessing km/h for them
    would be a 1.61x error on a third of the set — the same class as the
    `anchors.pt` control-units trap, where a correct formula under the wrong
    unit produced 396 g of lateral acceleration and looked like an answer.

    `state` is one of:
      `READ`    a value asserted as read off a sign, unhedged and un-negated
      `HEDGED`  a language prior ("residential areas typically have 25-30 mph")
      `NEGATED` an explicit absence ("the speed limit sign is not visible")
      `MENTION` the phrase occurs but carries no value and is neither of those

    ⚠️ ONLY `READ` carries a value. The other three carry the state and
    nothing else, so a consumer cannot mistake a prior for a perception.
    """

    state: str
    value: int | None = None
    unit: str | None = None          # "kph" | "mph" | None
    unit_missing: bool = False
    value_ms: float | None = None
    text: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def speed_limit_reading(text: str | None) -> SpeedLimitReading | None:
    """The posted speed limit this text claims, or None if it claims none.

    ⭐ Takes RAW text, not negation-stripped text: the negation and hedging
    are part of the CLASSIFICATION here, not noise to be removed before it.

    ⛔ This function does not decide anything. It captures and labels. The
    extract-vs-supply decision is the PI's and is OPEN, and these labels are
    ego-coupled (median ratio limit/ego **1.01**, grounded **0 of 57**), so no
    token is emitted and nothing reaches inference. See
    `—/Research/2026-09-06-speed-limit-source/RESULT.md`.
    """
    if not text:
        return None
    low = text.lower()
    if "speed limit" not in low and "speed-limit" not in low:
        return None
    neg = bool(_SPEED_LIMIT_NEG.search(text))
    hed = bool(_SPEED_LIMIT_HEDGE.search(text))
    m = _SPEED_LIMIT_READ.search(text)
    val = unit = None
    span = ""
    if m:
        span = m.group(0)[:160]
        digits = [g for g in m.groups() if g and g.isdigit()]
        units = [g for g in m.groups() if g and not g.isdigit()]
        if digits:
            val = int(digits[0])
        if units:
            unit = _norm_speed_unit(units[0])
    # ⛔ PRECEDENCE IS LOAD-BEARING and matches the sibling's classifier: a
    # blob that is BOTH hedged and negated is HEDGED, and a READ requires a
    # value that is neither.
    if val is not None and not hed and not neg:
        state = "READ"
    elif hed:
        state = "HEDGED"
    elif neg:
        state = "NEGATED"
    else:
        state = "MENTION"
    if state != "READ":
        return SpeedLimitReading(state=state, text=span)
    missing = unit is None
    return SpeedLimitReading(
        state=state, value=val, unit=unit, unit_missing=missing,
        value_ms=(None if missing else round(val * _SPEED_LIMIT_MS[unit], 4)),
        text=span)

# --- corridor offset: a HELD in-lane bias, PI-designed extraction 2026-08-28 --
#: ⭐ TWO PATTERN CLASSES WITH OPPOSITE SIGN RULES — the subtlety that makes
#: this extractable at all. DIRECT phrases state the EGO's own direction
#: ("keep right", "position left", "slight left offset"); OBJECT-SIDE phrases
#: place the OBSTACLE ("clearance to the van ON THE RIGHT", "pedestrians near
#: the LEFT curb") and the ego offsets to the OPPOSITE side. Reading the second
#: class as the first silently inverts ~2/3 of the corpus signal.
#:
#: MEASURED 2026-08-28 over 4,729 clips (negation-stripped text):
#:   * 906 clips claim an offset — 723 left / 164 right / 19 conflicted
#:     (excluded). The 82 % LEFT skew is itself content validation: obstacles
#:     overwhelmingly sit on the RIGHT in right-hand traffic.
#:   * clips carrying BOTH classes agree on the side **89 %** (78/88).
#:   * geometry sign test: correct direction, p = 0.14 — CANNOT RULE, and
#:     structurally never could: the arc fit that removes road curvature also
#:     absorbs any HELD offset. That is precisely why this token needed a
#:     non-geometric route (see D-DATA-GTAC-b).
_OFFSET_DIRECT = tuple(re.compile(x) for x in (
    r"\b(?:keep|stay|keeping|staying)\s+(?:to\s+the\s+|slightly\s+)?(left|right)\b",
    r"\bposition(?:ed|ing)?\s+(?:to\s+the\s+|to(?:ward)?s?\s+the\s+|slightly\s+)?(left|right)\b",
    r"\b(?:slight|small)\s+(left|right)\s+offset\b",
    r"\boffset(?:ting)?\s+(?:slightly\s+)?(?:to\s+the\s+)?(left|right)\b",
    r"\bshift(?:ing|ed)?\s+(?:slightly\s+)?(?:to\s+the\s+)?(left|right)\b",
    r"\bbias(?:ed)?\s+(?:to(?:ward)?s?\s+the\s+)?(left|right)\b",
))
_OFFSET_OBJSIDE = tuple(re.compile(x) for x in (
    r"\bclearance\s+(?:to|from)\s+[^.]{0,60}?\bon\s+the\s+(left|right)\b",
    r"\b(?:parked|stopped|stationary)\s+(?:car|van|truck|vehicle|bus)s?\b[^.]{0,50}?\bon\s+the\s+(left|right)\b",
    r"\b(?:pedestrian|cyclist|worker)s?\b[^.]{0,60}?\b(?:on|near|along)\s+the\s+(left|right)\b",
    r"\b(left|right)\s+(?:curb|edge|shoulder)\b",
))
_OFFSET_OPP = {"left": "right", "right": "left"}


def offset_side(t: str) -> str | None:
    """Resolved ego-offset side, or None (no claim, or conflicted votes).

    Majority vote across both classes; a tie is AMBIGUITY and returns None —
    when the source cannot say which side, the honest output is neither
    (the same rule that removed the TAKE_EXIT_L+R forbidden pair).
    """
    votes: dict[str, int] = {}
    for pat in _OFFSET_DIRECT:
        for m in pat.finditer(t):
            votes[m.group(1)] = votes.get(m.group(1), 0) + 1
    for pat in _OFFSET_OBJSIDE:
        for m in pat.finditer(t):
            side = _OFFSET_OPP[m.group(1)]
            votes[side] = votes.get(side, 0) + 1
    if not votes:
        return None
    top = max(votes, key=votes.get)
    if sum(v for k, v in votes.items() if k != top) >= votes[top]:
        return None
    return top


@dataclass
class CotTokens:
    traffic_light: str | None = None     # RED|YELLOW|GREEN|UNKNOWN
    oncoming: bool = False
    yield_: str | None = None            # SIGN | HAZARD
    merge: bool = False
    lane_change: str | None = None     # L | R
    gap: bool = False
    exit_side: str | None = None         # LEFT | RIGHT | UNKNOWN
    overtake: bool = False
    evade_obj: str | None = None         # PARKED|CYCLIST|PEDESTRIAN|DOOR|ONCOMING
    speed_limit: bool = False
    #: ⛔ THE VALUE THE OLD REGEX THREW AWAY. `speed_limit` above stays a
    #: presence flag with its original semantics; these five carry what it
    #: discarded. `speed_limit_ms` is **None whenever the unit is missing** —
    #: a number without its unit may not be converted, at any cost.
    speed_limit_state: str | None = None   # READ|HEDGED|NEGATED|MENTION
    speed_limit_value: int | None = None
    speed_limit_unit: str | None = None    # "kph" | "mph" | None
    speed_limit_unit_missing: bool = False
    speed_limit_ms: float | None = None
    evidence: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def extract(cot: str | None) -> CotTokens:
    if not cot or not cot.strip():
        return CotTokens()
    # ⛔⛔ NEGATED TERMS ARE NOT PRESENT TERMS. MEASURED 2026-08-24:
    # `TRAFFIC_LIGHT_REACT` fired on 60.4 % of the clips where Alpamayo states
    # nothing is critical, against 0.1 % where it names a component — a 408x
    # inversion, caused entirely by matching inside sentences of the form
    # "...with no lead vehicle pedestrians cyclists traffic lights or
    # obstacles...". Blanking the negated spans BEFORE matching is the fix;
    # offsets are preserved so `evidence` still refers to the original text.
    t = NEG.strip_negated(cot.lower())
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
    # a lane change is a stronger, more specific claim than "merge" and is
    # tested FIRST — "merge into the left lane" is a lane change, not a merge.
    for side, pat in _LANE_CHANGE:
        if pat.search(t):
            c.lane_change = side
            break
    # ⛔ refuse a `merge` that names a HAZARD CLASS rather than an event
    c.merge = bool(_MERGE.search(t)) and not _MERGE_HAZARD.search(t)
    if c.lane_change:
        c.merge = False
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
    # ⭐ Read from the RAW `cot`, not the negation-stripped `t`: the reading
    # classifier does its own negation and hedging, and stripping first would
    # hide the very state it is there to record.
    r = speed_limit_reading(cot)
    if r is not None:
        c.speed_limit_state = r.state
        c.speed_limit_value = r.value
        c.speed_limit_unit = r.unit
        c.speed_limit_unit_missing = r.unit_missing
        c.speed_limit_ms = r.value_ms
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
    if c.lane_change:
        out[f"LANE_CHANGE_{c.lane_change}"] = {"side": c.lane_change.lower()}
    if c.merge:
        out["MERGE"] = {"agent_slot": None}
    if c.gap:
        out["GAP_TARGET"] = {"agent_slot": None, "time_gap_s": None}
    if c.exit_side and c.exit_side != "UNKNOWN":
        # ⛔ NO SIDE STATED -> NO TOKEN. The old branch DEFAULTED unknown to
        # RIGHT — a coin flip written into ground truth. MEASURED 2026-08-28:
        # 9 of 183 exit clips, and one of them reads "continue straight PAST
        # the service-area exit" — not an exit-taking at all. The same
        # honest-output rule as offset ties and the both-sides exit conflict:
        # when the source cannot say which side, emit neither.
        out[f"TAKE_EXIT_{c.exit_side[0]}"] = {"side": c.exit_side.lower()}
    if c.overtake:
        out["OVERTAKE_VEHICLE"] = {"agent_slot": None}
    if c.evade_obj:
        out["EVADE_IN_CORRIDOR"] = {"obstacle_class": c.evade_obj.lower()}
    # ⭐ PI 2026-08-28: CORRIDOR_OFFSET carries EXACTLY one constraint — the
    # side. No magnitude: no source can state one, and inventing it would be
    # the uncalibrated-threshold defect this token was declared unreachable
    # for. A held offset is invisible to arc-removed geometry, so this is a
    # perception claim like the rest of this module's output: disputed until
    # corroborated, never geometry-verified.
    side = offset_side(NEG.strip_negated((cot or "").lower()))
    if side:
        out["CORRIDOR_OFFSET"] = {"side": side}
    # ⛔ NO SPEED-LIMIT TOKEN IS EMITTED, DELIBERATELY. The value is now
    # CAPTURED on `CotTokens` (and lands in the banked label record), but it
    # does not become a goal and it does not reach inference. Three reasons,
    # all MEASURED over the 4,729-clip corpus:
    #   1. coverage is 57 clips (1.21 %) — not a training set;
    #   2. GROUNDED SHARE IS 0/57 — the `grounding_via_vqa` box census has no
    #      sign class at all and 0 of 4,728 questions mention a sign, so not one
    #      reading is corroborable in image space;
    #   3. the readings are EGO-COUPLED (median ratio limit/ego 1.01), so a
    #      supplied channel would be the nav-echo defect and a fitted head could
    #      score well by predicting `v0`.
    # ⇒ EXTRACT-vs-SUPPLY IS AN OPEN PI DECISION. Capturing the data is the
    # whole scope; `—/Research/2026-09-06-speed-limit-source/RESULT.md` Sec 5.
    return out
