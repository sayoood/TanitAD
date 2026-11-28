"""Semantic extraction from Alpamayo-Super chain-of-thought text.

⭐ THE DIVISION OF LABOUR THIS ENCODES (measured, 2026-08-23). Alpamayo's
categorical axes and its own CoT were tested against ego kinematics on the 16
covered clips of the validation sample:

  * its LONGITUDINAL axis agrees with our ego dv on only **9/16 = 56.2 %**, with
    **4/16 outright decel-vs-accelerate contradictions**;
  * the disagreement is **NOT a temporal-alignment artefact** — I hypothesised
    Alpamayo described a different window and TESTED it at six anchors
    (clip start, 2-4 s, 4-6 s, our 7.8 s anchor, 10-12 s, whole clip). Our
    anchor scored BEST at 56.2 %; no offset improved it. Hypothesis REFUTED,
    so the axis is genuinely unreliable against ego ground truth;
  * but its CoT names **red lights, yellow lights, stop signs, yield signs,
    speed bumps, crosswalks, pedestrians, cyclists and lead vehicles** — every
    one of which is invisible to ego poses.

⇒ **USE ALPAMAYO FOR SEMANTICS (why), NEVER FOR KINEMATICS (what).** Ego
geometry is authoritative for what the vehicle did; the CoT supplies the reason.
This is exactly what separates a red-light stop from a traffic jam, which
kinematics alone cannot do reliably (PI, 2026-08-23).

⛔ ADMISSIBILITY. These tags are LABEL-SIDE ONLY, derived from an offline VLM
pass. They are never an inference-time input (vision-only rule, PI 2026-08-03),
and they may not enter a goal signal that must stay disjoint from the situation
classifier (PI 2026-08-03).

⚠️ These are KEYWORD tags over generated text, not a parser. The CoT is model
output and can be wrong or self-contradictory — `3a0165bd` says "slow down due
to the speed bump" while its own longitudinal axis reads "Gentle Acceleration".
A tag therefore states *what the VLM claimed*, never *what was there*. Anything
consuming it must carry that distinction.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# Ordered: the FIRST matching family wins, so a "red light" beats a bare
# "lead vehicle" mention when both appear in one sentence.
_PATTERNS: list[tuple[str, str]] = [
    ("TRAFFIC_LIGHT_RED", r"\bred (?:traffic )?light|\blight (?:is |turns? )?red"),
    ("TRAFFIC_LIGHT_YELLOW", r"\byellow (?:traffic )?light|\bamber\b"),
    ("TRAFFIC_LIGHT_GREEN", r"\bgreen (?:traffic )?light|\blight (?:is |turns? )?green"),
    ("STOP_SIGN", r"\bstop sign\b"),
    ("YIELD_SIGN", r"\byield sign\b|\bgive way\b"),
    ("PEDESTRIAN", r"\bpedestrian|\bcrosswalk|\bcrossing\b"),
    ("CYCLIST", r"\bcyclist|\bbicycle|\bbike\b"),
    ("LEAD_VEHICLE", r"\blead vehicle|\bvehicle ahead|\bcar ahead|\bsafe distance"),
    ("MERGE_GAP", r"\bgap\b.*\b(?:lane|vehicles)|\bmerge\b"),
    ("SPEED_BUMP", r"\bspeed bump|\bhump\b"),
    ("PARKED", r"\bparked\b"),
    ("INTERSECTION", r"\bintersection|\bjunction\b"),
    # ⭐ ROUNDABOUT is its own referent, not a flavour of INTERSECTION: the
    # vocabulary has no ROUNDABOUT token and wide rotations are currently
    # absorbed into TURN_LEFT/TURN_RIGHT (confirmed visually on `d5a38fdd`,
    # a 177 deg circulation). Tagging it separately is what lets that gap be
    # COUNTED rather than silently folded away.
    ("ROUNDABOUT", r"\broundabout\b|\btraffic circle\b"),
    # THE VLM NAMES ROAD CURVATURE EXPLICITLY ("adapt speed for the right
    # curve ahead", 301 clips). That is an INDEPENDENT witness for the
    # ROAD_BEND vs JUNCTION_TURN split in `ego_manoeuvre`, derived from
    # PIXELS rather than poses -- so it can CONTROL the curvature classifier
    # rather than merely agree with it.
    ("CURVE_AHEAD", r"\bcurve ahead|\b(?:right|left)(?:-hand)? curve\b"),
    ("CLEAR_ROAD", r"\broad ahead is clear|\blane ahead is clear\b"),
]

# What a stop at each referent SHOULD be called strategically. This is the
# mapping the PI asked for: a light/sign stop is a CONTROLLED stop and belongs
# in a different strategic class from queueing behind traffic.
_STOP_REASON = {
    "TRAFFIC_LIGHT_RED": "light",
    "TRAFFIC_LIGHT_YELLOW": "light",
    "TRAFFIC_LIGHT_GREEN": "light",
    "STOP_SIGN": "sign",
    "YIELD_SIGN": "sign",
    "PEDESTRIAN": "hazard",
    "CYCLIST": "hazard",
    "SPEED_BUMP": "hazard",
    "LEAD_VEHICLE": "queue",
    "MERGE_GAP": "queue",
}

_NEGATED = re.compile(r"\bno\s+(?:traffic\s+)?light|\bwithout\b")


@dataclass
class Semantics:
    referents: tuple[str, ...]      # every family the CoT mentions
    stop_reason: str | None         # light | sign | queue | hazard | None
    controlled: bool | None         # True = light/sign, False = queue/hazard
    cot: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["referents"] = list(self.referents)
        return d


def extract(cot: str | None) -> Semantics:
    """Tag one chain-of-thought sentence. Empty input yields empty tags."""
    if not cot or not cot.strip():
        return Semantics(referents=(), stop_reason=None, controlled=None,
                         cot="")
    text = cot.lower()
    hits = [name for name, pat in _PATTERNS
            if re.search(pat, text) and not _NEGATED.search(text)]
    reason = next((_STOP_REASON[h] for h in hits if h in _STOP_REASON), None)
    controlled = None if reason is None else reason in ("light", "sign")
    return Semantics(referents=tuple(hits), stop_reason=reason,
                     controlled=controlled, cot=cot)


def reconcile(stop_type_kinematic: str, sem: Semantics) -> tuple[str, str]:
    """Combine the kinematic stop shape with the VLM's stated reason.

    Returns ``(resolved_stop_type, provenance)``. The rules are deliberately
    conservative — geometry wins on WHETHER a stop happened, semantics wins on
    WHY, and a conflict is surfaced rather than silently resolved.
    """
    if stop_type_kinematic == "NONE":
        return "NONE", "kinematic"
    if sem.stop_reason is None:
        return stop_type_kinematic, "kinematic-only (no VLM reason)"
    if sem.stop_reason in ("light", "sign"):
        if stop_type_kinematic == "QUEUE":
            # repetition says jam, the VLM says signal — do not overwrite,
            # report the conflict so a human or a second engine can settle it.
            return "QUEUE", f"CONFLICT: kinematics=QUEUE vs VLM={sem.stop_reason}"
        return "CONTROLLED", f"VLM {sem.stop_reason}"
    if sem.stop_reason == "queue":
        if stop_type_kinematic == "CONTROLLED":
            return "QUEUE", "VLM lead-vehicle/merge overrides single-stop shape"
        return "QUEUE", f"VLM {sem.stop_reason}"
    return stop_type_kinematic, f"VLM {sem.stop_reason} (hazard keeps shape)"


# ===========================================================================
# VERB -> TACTICAL TOKEN  (PI idea, 2026-08-23: "build term search in the
# pipeline on the Alpamayo CoT and leverage these as additional labels")
# ===========================================================================
# ⭐ WHY THIS MATTERS MORE THAN IT LOOKS. HIERARCHY_VOCABULARY.md lists SIX
# tactical tokens as blocked because they need NON-EGO inputs the corpus does
# not ship: GAP_TARGET, YIELD_AT, WAIT_FOR_ONCOMING, EVADE_IN_CORRIDOR,
# TRAFFIC_LIGHT_REACT and SPEED_BAND. The CoT names precisely those referents.
# MEASURED over all 4,729 augmented clips:
#     "nudge left" 608 + "nudge right" 184 = 792  -> EVADE_IN_CORRIDOR
#     "yield" 156                               -> YIELD_AT
#     light referents 629 (red 175/yellow 26/green 428) -> TRAFFIC_LIGHT_REACT
#     lead-vehicle 985 + merge-gap 68           -> GAP_TARGET
# ⇒ the CoT is not a nice-to-have corroboration channel; it is the only
# in-hand SOURCE for tokens that are otherwise unemittable.
#
# ⚠️ EVERY token proposed here is a CLAIM BY A GENERATIVE MODEL about a scene,
# not a measurement of it. Each carries `provenance="vlm-cot"` and a verbatim
# `evidence` string so a reviewer can always see the sentence it came from.
# Per HIERARCHY_VOCABULARY's fusion gate, an ungrounded semantic claim is
# `disputed` by default — these have no image-space grounding (no bbox, no
# frame index), so they may supervise a goal/interpretation head but must NOT
# be promoted to trusted perception without a grounding pass.

_VERB_PATTERNS: list[tuple[str, str]] = [
    ("NUDGE_L", r"^nudge left|\bnudge left\b"),
    ("NUDGE_R", r"^nudge right|\bnudge right\b"),
    ("YIELD", r"^yield\b|\byield to\b"),
    ("STOP", r"^stop\b"),
    ("RESUME", r"^resume speed|^accelerate"),
    ("SLOW", r"^slow down|^decelerate"),
    ("KEEP", r"^keep\b|^maintain\b|^adapt speed|^match speed"),
    ("TURN_L", r"^turn left|\bturn left at\b"),
    ("TURN_R", r"^turn right|\bturn right at\b"),
]

_EVADE_OBSTACLE = ("CYCLIST", "PEDESTRIAN", "PARKED")


@dataclass
class TokenProposal:
    token: str
    args: dict
    provenance: str
    evidence: str
    disputed: bool          # True until image-space grounding exists

    def as_dict(self) -> dict:
        return asdict(self)


def verb_of(cot: str | None) -> str | None:
    """The leading action verb the CoT names, as a coarse class."""
    if not cot:
        return None
    text = cot.strip().lower()
    for name, pat in _VERB_PATTERNS:
        if re.search(pat, text):
            return name
    return None


def propose_tokens(cot: str | None) -> list[TokenProposal]:
    """Map one CoT sentence onto tactical vocabulary tokens.

    Returns every token the sentence supports — a CoT can carry more than one
    (e.g. "slow down due to the red light at the intersection" supports both
    TRAFFIC_LIGHT_REACT and STOP_POINT). Unmappable sentences return [] and
    SHOULD BE LOGGED by the caller, never silently dropped: vocabulary
    completeness is measured as coverage of the phrase distribution.
    """
    sem = extract(cot)
    if not cot:
        return []
    verb, out = verb_of(cot), []
    refs = set(sem.referents)

    # EVADE_IN_CORRIDOR — a lateral offset around an obstacle, bounded by the
    # corridor. NOT a lane change (the vocabulary is explicit about that).
    if verb in ("NUDGE_L", "NUDGE_R"):
        obstacle = next((r for r in _EVADE_OBSTACLE if r in refs), None)
        out.append(TokenProposal(
            "EVADE_IN_CORRIDOR",
            {"direction": "left" if verb == "NUDGE_L" else "right",
             "obstacle_class": obstacle},
            "vlm-cot", cot, disputed=True))

    if verb == "YIELD" or "YIELD_SIGN" in refs:
        out.append(TokenProposal(
            "YIELD_AT", {"gap_class": "MERGE_GAP" if "MERGE_GAP" in refs else None},
            "vlm-cot", cot, disputed=True))

    state = ("red" if "TRAFFIC_LIGHT_RED" in refs else
             "yellow" if "TRAFFIC_LIGHT_YELLOW" in refs else
             "green" if "TRAFFIC_LIGHT_GREEN" in refs else None)
    if state:
        out.append(TokenProposal(
            "TRAFFIC_LIGHT_REACT", {"state": state}, "vlm-cot", cot,
            disputed=True))

    if refs & {"LEAD_VEHICLE", "MERGE_GAP"}:
        out.append(TokenProposal(
            "GAP_TARGET",
            {"agent_class": "lead" if "LEAD_VEHICLE" in refs else "merge"},
            "vlm-cot", cot, disputed=True))

    if verb == "STOP" or sem.stop_reason in ("light", "sign"):
        out.append(TokenProposal(
            "STOP_POINT", {"reason": sem.stop_reason or "hazard"},
            "vlm-cot", cot, disputed=True))
    return out


# ===========================================================================
# JUNCTION CORROBORATION  (PI idea, 2026-08-23)
# ===========================================================================
# ⭐ THE IDEA: "whenever you detect turning / exit etc as terms IN COMBINATION
# with the word intersection, roundabout ... " use it to support the strategic
# route goal. The COMBINATION is what carries information — a turn verb alone
# is a kinematic claim (and Alpamayo's kinematic axes are at chance), while a
# turn verb TOGETHER WITH a junction referent is a claim about the ROAD
# TOPOLOGY, which ego poses cannot see at all.
#
# ⛔ AND IT IS CORROBORATION, NEVER AN OVERRIDE. I have already been burned
# once using this exact source to move a threshold (C138): the CoT for
# `00d05901` asserted "turn right at the intersection since the traffic light
# is green" on a clip whose frames show a RURAL FOREST ROAD with neither. So
# this function REPORTS agreement; it never decides. Geometry decides whether a
# turn happened; this says whether the VLM independently saw a junction there.
#
# Where it is genuinely useful is the ambiguous-radius band, where curvature
# alone cannot separate a wide junction turn from a road bend — and there it
# raises CONFIDENCE, it does not flip the class.

_TURN_VERB = re.compile(
    r"\bturn(?:ing|s)?\b|\bexit(?:ing|s)?\b|\bmerge?\b|\bturn off\b")
_JUNCTION_REF = ("INTERSECTION", "ROUNDABOUT")


@dataclass
class JunctionCorroboration:
    has_turn_verb: bool
    has_junction_referent: bool
    corroborates: bool          # BOTH present -> the VLM saw a junction manoeuvre
    direction: str | None       # left | right | None, when the CoT names one
    evidence: str

    def as_dict(self) -> dict:
        return asdict(self)


def junction_corroboration(cot: str | None) -> JunctionCorroboration:
    """Does the CoT independently place a TURN at a JUNCTION?"""
    if not cot:
        return JunctionCorroboration(False, False, False, None, "")
    text = cot.lower()
    verb = bool(_TURN_VERB.search(text))
    refs = set(extract(cot).referents)
    junc = bool(refs & set(_JUNCTION_REF))
    direction = ("left" if re.search(r"\bleft\b", text) else
                 "right" if re.search(r"\bright\b", text) else None)
    return JunctionCorroboration(
        has_turn_verb=verb, has_junction_referent=junc,
        corroborates=(verb and junc), direction=direction, evidence=cot)
