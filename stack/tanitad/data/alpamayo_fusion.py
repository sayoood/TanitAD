"""Fuse the Alpamayo augmentation with ego geometry — calibrated, not assumed.

⭐ WHAT THIS MODULE IS FOR. Geometry knows WHAT the ego did; Alpamayo knows WHY
and, it turns out, HOW HARD. This is the layer that combines them, and every
constant in it is MEASURED on the 968 clips where both sources exist — not
chosen.

## The longitudinal calibration (MEASURED 2026-08-23, n=968)

`dv` is the speed change over Alpamayo's own 6 s horizon from its own 5.1 s
anchor. The axis is monotone, well separated, and magnitude-bearing:

| phrase | n | mean dv | v0 | interpretation |
|---|---|---|---|---|
| Strong Deceleration | 61 | **−5.88** | 10.64 | hard brake, stays down (v+6s 4.76) |
| Gentle Deceleration | 312 | −1.73 | 11.85 | dips to 9.56, partly recovers |
| Maintain Speed | 286 | +0.03 | 13.09 | flat — and the FASTEST class |
| Gentle Acceleration | 222 | +1.93 | 8.64 | |
| Strong Acceleration | 43 | +6.00 | 2.69 | 48.8 % launch from below 1 m/s |
| Stop | 48 | +1.96 | **1.11** | ⚠️ see below |

`accel` vs `decel` separation is **Cohen's d = 1.59**. Strong is ≈3× gentle on
both signs. This is a usable magnitude scale, not just a sign.

⚠️ **`Stop` IS A STATE, NOT AN ACTION — and reading it as an action inverts it.**
Its mean dv is **+1.96 m/s**, which looks like the axis is broken. It is not:
`v0 = 1.11 m/s` and **66.7 % of Stop clips are already below 1 m/s at t0**, with
`vmin = 0.10`. Alpamayo is saying *"the ego is at a standstill"*, and what
follows in the horizon is the LAUNCH. Mapping `Stop → decelerating` (which I did)
puts 48 clips on exactly the wrong side of the longitudinal axis.

That single mis-mapping is most of why the naive 4-way agreement read 30.7 %
against a 26.5 % chance baseline — a weak number produced by a strong signal
passed through a lossy collapse. Same family as the `heldout`-vs-`full_set`
trap: the estimator, not the quantity.

## The lateral axis

65.7 % vs 37.0 % shuffled, **p < 0.0001**, n = 920 — strongly informative, and
the axis I previously called "at chance" from n = 16 (RETRACTION_LOG C142).

## What is NOT dense enough to label from

`vqa` is a SAMPLED bank: ~5 questions per clip drawn from a large pool, so any
individual question covers ~0.3 % of the corpus (e.g. *"is there a dedicated
left-turn phase"* appears 15 times in 4,729 clips). It is excellent for
spot-checks and for the visual report; it **cannot** systematically populate a
label, and treating it as a dense feature would fabricate coverage.

Dense enough to build on: `meta_action` 100 %, `cot` 100 %,
`chain_of_causation` 100 %, `boxes` 93.3 %.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import alpamayo_records as AR

#: MEASURED mean dv (m/s) over the 6 s Alpamayo horizon, by raw phrase.
#: Used to CHECK a geometric longitudinal class, never to replace the geometry.
LON_PHRASE_DV: dict[str, float] = {
    "Strong Deceleration": -5.88,
    "Gentle Deceleration": -1.73,
    "Maintain Speed": +0.03,
    "Gentle Acceleration": +1.93,
    "Strong Acceleration": +6.00,
}
#: `Stop` is excluded above on purpose: it describes v0, not dv.
LON_STATE_PHRASES = ("Stop",)

#: A phrase and a measured dv disagree if they fall on opposite sides of this
#: band. Set to the CRUISE half-width already used by the guard, so the two
#: instruments cannot contradict each other by construction.
DV_NEUTRAL_MS = 1.0

#: Below this the ego is "at rest" for the purpose of reading `Stop`.
V_REST_MS = 1.0


@dataclass
class LonFusion:
    """What the two sources jointly say about the longitudinal axis."""

    phrase: str | None          # Alpamayo's raw phrase
    dv_expected: float | None   # calibrated dv for that phrase
    dv_measured: float          # what the ego actually did
    at_rest: bool               # v0 < V_REST_MS
    agree: bool | None          # None when the phrase carries no dv claim
    strength: str | None        # "gentle" | "strong" | None
    note: str = ""

    @property
    def usable(self) -> bool:
        """Is this a corroborated longitudinal reading?"""
        return self.agree is True


def fuse_longitudinal(clip_id: str, v0: float, dv_measured: float) -> LonFusion:
    """Cross-check the measured speed change against Alpamayo's claim.

    ⚠️ Geometry stays authoritative. This returns AGREEMENT, so a disagreement
    becomes a QA item rather than silently overwriting a measured quantity with
    a generative model's phrase.
    """
    c = AR.get(clip_id)
    phrase = (c.meta_action.get("longitudinal") if c and c.meta_action else None)
    at_rest = v0 < V_REST_MS

    if not phrase:
        return LonFusion(None, None, dv_measured, at_rest, None, None,
                         "no meta_action")

    if phrase in LON_STATE_PHRASES:
        # `Stop` claims v0 ~ 0, NOT dv < 0. Score it on the right quantity.
        ok = at_rest
        return LonFusion(phrase, None, dv_measured, at_rest, ok, None,
                         "state claim (v0), not a dv claim; "
                         f"v0={v0:.2f} m/s {'confirms' if ok else 'REFUTES'}")

    exp = LON_PHRASE_DV.get(phrase)
    if exp is None:
        return LonFusion(phrase, None, dv_measured, at_rest, None, None,
                         "phrase not in the calibration table")

    # Agreement = same side of the neutral band, or both inside it.
    def side(x: float) -> int:
        return 0 if abs(x) <= DV_NEUTRAL_MS else (1 if x > 0 else -1)

    ok = side(exp) == side(dv_measured)
    strength = ("strong" if "Strong" in phrase else
                "gentle" if "Gentle" in phrase else None)
    return LonFusion(phrase, exp, dv_measured, at_rest, ok, strength,
                     f"expected ~{exp:+.1f}, measured {dv_measured:+.1f} m/s")


#: The ONLY admissible geometry sides. A caller passes one of these, never a
#: token to be pattern-matched.
SIDES = ("left", "right", "straight")


def side_of(lateral_class: str) -> str:
    """Map a geometry lateral CLASS to a side, by SUFFIX not by substring.

    ⛔⛔ THIS FUNCTION EXISTS BECAUSE `"_L" in "FOLLOW_LANE"` IS TRUE.
    MEASURED 2026-08-23: the fusion layer tested `"_L" in token`, so every
    straight-driving clip — the majority of the corpus — was read as a LEFT
    turn, and the lateral agreement it reported collapsed from the true
    **69.9 %** to **24.9 %**. The number looked like a finding about Alpamayo.
    It was a substring collision in our own code.

    Same family as the polling monitor whose filter matched its own echoed
    command: a pattern that accidentally matches a longer string, read as a
    result. ⇒ **Match token SUFFIXES explicitly; never substring-test a
    vocabulary token.**
    """
    t = lateral_class.upper()
    if t.endswith("_L") or t.endswith("_LEFT"):
        return "left"
    if t.endswith("_R") or t.endswith("_RIGHT"):
        return "right"
    return "straight"


def fuse_lateral(clip_id: str, geom_side: str) -> tuple[bool | None, str | None]:
    """(agrees, alpamayo_side). Geometry decides; this corroborates.

    Per C142 the axis is informative (69.9 % vs 41.6 % chance, lift x1.68) —
    but 69.9 % is not 90 %, so it is a second opinion, never an override.

    ``geom_side`` MUST already be one of `SIDES`; pass `side_of(cls)` to
    convert a class. Passing a raw token is refused rather than silently
    substring-matched.
    """
    if geom_side not in SIDES:
        raise ValueError(
            f"geom_side must be one of {SIDES}, got {geom_side!r} — "
            f"use side_of() to convert a lateral CLASS, and never pass a "
            f"vocabulary token (\"_L\" in \"FOLLOW_LANE\" is True)")
    c = AR.get(clip_id)
    a = c.lateral if c else None
    if a is None:
        return None, None
    return (a == geom_side), a


#: Which box classes ground which CoT-derived token.
_GROUNDS: dict[str, str] = {
    "TRAFFIC_LIGHT_REACT": "traffic_light",
    "TRAFFIC_LIGHT_REACT_RED": "traffic_light",
    "TRAFFIC_LIGHT_REACT_YELLOW": "traffic_light",
    "TRAFFIC_LIGHT_REACT_GREEN": "traffic_light",
    "OVERTAKE_VEHICLE": "vehicle",
    "REACT_ON_ONCOMING": "vehicle",
    "GAP_TARGET": "vehicle",
    "MERGE": "vehicle",
}
#: EVADE grounds on the obstacle class its own arg names.
_EVADE_GROUND = {"cyclist": "vru", "pedestrian": "vru", "door": "vehicle",
                 "parked": "vehicle", "oncoming": "vehicle",
                 "stopped_vehicle": "vehicle"}


def ground_tokens(clip_id: str, tokens: dict[str, dict]) -> dict[str, dict]:
    """Promote CoT tokens out of ``disputed`` where a BOX supports them.

    ⭐ THIS IS THE FUSION GATE THE v7 DESIGN REQUIRED AND COULD NOT SATISFY.
    A CoT sentence is a generative claim, measured at 3 correct / 2 wrong on
    visually checkable statements (C136/C139), so every CoT token carried
    ``disputed=True``. A 2D box is image-space perception. 93.3 % of clips have
    boxes — Car 2,292 · Pedestrian 1,079 · traffic light 889 — so most claims
    can now be checked rather than believed.

    Each token gains ``grounded`` (bool) and ``grounding`` (what settled it).
    A token whose box class is ABSENT keeps ``disputed=True`` and is marked
    ``contradicted`` when the corpus had boxes for that clip yet none of the
    right kind — that is a stronger negative than "no boxes at all", and the
    two must not be conflated.
    """
    c = AR.get(clip_id)
    out: dict[str, dict] = {}
    has_any = bool(c and c.boxes)
    for tok, args in tokens.items():
        a = dict(args)
        kind = _GROUNDS.get(tok)
        if kind is None and tok == "EVADE_IN_CORRIDOR":
            kind = _EVADE_GROUND.get(str(a.get("obstacle_class", "")).lower())
        if kind is None or c is None:
            a["grounded"] = False
            a["grounding"] = "no box class defined for this token"
        elif c.grounded(kind):
            a["grounded"] = True
            a["disputed"] = False
            a["grounding"] = f"box:{kind}"
        else:
            # ⛔ NOT "contradicted". RETRACTED 2026-08-23: I wrote a
            # `contradicted` state for "clip has boxes, none of kind X" and it
            # is INVALID REASONING. There is exactly ONE grounding question per
            # clip, sampled from a pool — the pedestrian question is asked on
            # only 1,165 of 4,729 clips, and **3,246 clips with boxes were
            # never asked about pedestrians at all**. A missing box of kind X
            # almost always means the question about X was not asked.
            #
            # ⇒ **Grounding can CONFIRM, never REFUTE.** The honest state is
            # "not checked", and it must not read as evidence against.
            asked = (c.box_question or "")
            a["grounded"] = False
            a["disputed"] = True
            a["grounding"] = (
                f"not checked — the one grounding question on this clip was "
                f"{asked[:60]!r}" if has_any else "no grounding question on this clip")
        out[tok] = a
    return out


_SIDE_L = __import__("re").compile(r"\bleft\b", __import__("re").I)
_SIDE_R = __import__("re").compile(r"\bright\b", __import__("re").I)


def _stated_side(text: str | None) -> str | None:
    """`L` / `R` if the text names exactly one side, else None."""
    if not text:
        return None
    left, right = bool(_SIDE_L.search(text)), bool(_SIDE_R.search(text))
    return "L" if left and not right else "R" if right and not left else None


def cot_conflict(clip_id: str) -> dict:
    """Do `cot` and `chain_of_causation` contradict each other?

    ⛔⛔ THEY ARE TWO INDEPENDENT GENERATIVE DRAWS AND THEY DISAGREE. MEASURED
    2026-08-27 (PI review of `d94365be`):

        cot   : "Nudge LEFT due to the PARKED CAR on the right."
        chain : "Nudge RIGHT due to the ONCOMING VEHICLE"

    Opposite directions, different objects, same clip. Over the corpus, of the
    557 clips where both fields name exactly one side, **49 (8.8 %) CONFLICT** —
    "lead vehicle turning left" vs "turning right", "lane change to the right"
    vs "change lanes left".

    ⚠️ `cot_text()` concatenates all four fields, so the extractor was taking
    claims from BOTH SIDES of a contradiction and emitting them together. The
    +77 % token-yield gain that concatenation bought is therefore partly this.
    `chain_of_causation` alone contributes 93 `REACT_ON_ONCOMING` tokens that
    `cot` does not.

    ⇒ On a stated-side conflict the PRIMARY field (`cot`) wins and the conflict
    is recorded, rather than both being believed.
    """
    c = AR.get(clip_id)
    if not c:
        return {"conflict": False}
    a, b = _stated_side(c.cot), _stated_side(c.chain_of_causation)
    return {"conflict": bool(a and b and a != b),
            "cot_side": a, "chain_side": b,
            "cot": c.cot, "chain_of_causation": c.chain_of_causation}


def corroborate(clip_id: str, token: str, args: dict) -> dict:
    """A SECOND, clearly-separated evidence tier for a CoT token.

    ⚠️ CORRECTS A NUMBER I PUBLISHED. I reported box grounding at "~13 % of CoT
    tokens" and called it a ceiling. That was the WRONG DENOMINATOR — it counted
    every CoT token, including those with no checkable object at all (a `YIELD`
    whose reason is a sign has nothing a box could confirm). Over the tokens
    that DO name a checkable object kind, box grounding is **41.1 %**
    (755/1,835).

    ⭐ And a second source exists. `critical_components_analysis` names the
    object that forces the behaviour, and it comes from a DIFFERENT Alpamayo
    task (`auto_labeling`) than the boxes (`grounding_via_vqa`). It corroborates
    **50.2 %**, and either source reaches **70.1 %**.

    ⛔ THEY ARE NOT THE SAME EVIDENCE CLASS AND ARE NEVER MERGED INTO ONE FLAG.
    A box is IMAGE-SPACE PERCEPTION; a named component is a SECOND TEXT CLAIM by
    the same model family. `grounded` keeps meaning box-only, and this returns a
    separate `corroboration` tier so a consumer can require perception where
    perception is what it needs.
    """
    from . import alpamayo_structured as AST
    KIND = {"TRAFFIC_LIGHT_REACT": "traffic_light",
            "TRAFFIC_LIGHT_REACT_RED": "traffic_light",
            "TRAFFIC_LIGHT_REACT_GREEN": "traffic_light",
            "TRAFFIC_LIGHT_REACT_YELLOW": "traffic_light",
            "OVERTAKE_VEHICLE": "vehicle", "REACT_ON_ONCOMING": "vehicle",
            "GAP_TARGET": "vehicle", "MERGE": "vehicle"}
    kind = KIND.get(token) or _EVADE_GROUND.get(
        str(args.get("obstacle_class", "")).lower())
    if not kind:
        return {"corroboration": "not_checkable",
                "reason": "no object kind a box or component could confirm"}
    by_box = bool(args.get("grounded"))
    kinds = {c.kind for c in AST.critical_components(clip_id) if c.kind}
    by_comp = kind in kinds
    tier = ("both" if by_box and by_comp else
            "box" if by_box else "component" if by_comp else "none")
    return {"corroboration": tier, "object_kind": kind,
            "by_box_perception": by_box, "by_named_component": by_comp}


_TURN_WORD = __import__("re").compile(
    r"\bturn(?:s|ing)?\s+(?:to\s+the\s+)?(left|right)\b", __import__("re").I)
_TURN_CTX = __import__("re").compile(
    r"\b(?:intersection|junction|t-junction|crossroads|corner|turn(?:s|ing)?)\b",
    __import__("re").I)
_PASS_CTX = __import__("re").compile(
    r"\b(?:parked|stopped|stationary|double-?parked)\b"
    r"[^.]{0,60}?\b(?:car|vehicle|van|truck|bus)s?\b"
    r"|\bnudge\b|\bpass(?:ing|es)?\s+the\s+parked\b|\bpull(?:ing)?\s+out\b",
    __import__("re").I)


def turn_corroboration(clip_id: str, side: str, t_start_rel_s: float) -> dict:
    """Is a GEOMETRIC turn a junction turn, or an obstacle-pass in disguise?

    ⛔⛔ WHY THIS EXISTS (PI, 2026-08-29, clip `d8f80c0f`): at walking speed a
    pull-out around a parked car produces a tight, slow, large-dyaw manoeuvre
    that is KINEMATICALLY IDENTICAL to a junction turn — `d8f80c0f` scored
    -42.7 deg at R=12.2 m and 1.6-4.6 m/s, passed every `is_turn` gate, and was
    published as YIELD_FOR_TURN_R + TURN_R. The zoomed frames show no junction
    at all: a parked yellow car at the left edge, and the ego straightening
    onto the SAME street. Geometry cannot tell these apart; the TEXT can.

    Three states, and the asymmetry is the design:

      confirmed      — an explicit turn CLAIM agrees: a motion segment typed
                       "turn <side>", or a side-matched "turn left/right" in
                       the text, or junction context words.
      contested      — NO turn claim anywhere AND the text actively describes
                       an obstacle-pass (parked/stopped vehicle, nudge,
                       pull-out). Only then is the turn re-read as obstacle
                       geometry.
      uncorroborated — silence. ⛔ SILENCE IS NOT CONTRADICTION: `95d2c361` is
                       a real, visually confirmed -90 deg junction turn whose
                       CoT discusses only the lead vehicle. Uncorroborated
                       turns are KEPT — geometry stays authoritative — and
                       merely labelled so.

    ⚠️ `meta_action` steer direction deliberately does NOT confirm: steering
    around a parked car is also a "Sharp Steer Right", so it cannot separate
    the two cases (`d8f80c0f`'s meta says exactly that).

    ⚠️ The text describes Alpamayo's window, which ends ~+5.1 s relative to our
    anchor. A manoeuvre starting later is OUTSIDE what the text can speak
    about, so it can be confirmed but never contested by it.
    """
    from . import alpamayo_structured as AST
    from . import cot_negation as NEG
    txt = NEG.strip_negated(cot_text(clip_id).lower())
    want = side.lower()

    seg_turn = any(
        ("turn left" in seg.raw_type.lower() and want == "left")
        or ("turn right" in seg.raw_type.lower() and want == "right")
        for seg in AST.motion_segments(clip_id))
    m = _TURN_WORD.search(txt)
    word_turn = bool(m and m.group(1) == want)
    ctx_turn = bool(_TURN_CTX.search(txt))
    if seg_turn or word_turn or ctx_turn:
        ev = ("segment" if seg_turn else "turn-word" if word_turn else "junction-context")
        return {"state": "confirmed", "evidence": ev}

    beyond_text = t_start_rel_s > (AR.ALPAMAYO_T0_S - 8.0) + 8.0  # +5.1 s rel
    if not beyond_text and _PASS_CTX.search(txt):
        return {"state": "contested",
                "evidence": "obstacle-pass context, no turn claim anywhere"}
    return {"state": "uncorroborated",
            "evidence": "text silent on the lateral manoeuvre"}


def cot_text(clip_id: str) -> str:
    """The richest CoT text available for term extraction.

    MEASURED: `chain_of_causation` differs from `cot` on 821 of 968 clips and
    is LONGER on 510 — so extracting from `cot` alone discarded text on more
    than half the corpus. Both are concatenated: a term found in either counts,
    and neither is authoritative on its own.

    ⛔ EXCEPT ON A DIRECTION CONFLICT. When `cot` and `chain_of_causation` name
    OPPOSITE sides (49 clips, 8.8 % of those stating a side), concatenating them
    feeds the extractor both halves of a contradiction — `d94365be` says "nudge
    LEFT due to the parked car" and "nudge RIGHT due to the oncoming vehicle".
    There the PRIMARY field wins and the secondary is dropped, because believing
    both is strictly worse than believing one. See `cot_conflict`.
    """
    c = AR.get(clip_id)
    if not c:
        return ""
    chain = c.chain_of_causation or ""
    if cot_conflict(clip_id)["conflict"]:
        chain = ""              # the primary field wins; the conflict is logged
    parts = [c.cot or "", chain,
             c.components_analysis or "", c.motion_analysis or ""]
    seen: list[str] = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.append(p)
    return "  ".join(seen)
