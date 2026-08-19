#!/usr/bin/env python3
"""Qwen3.5-9B (thinking mode) prompts for the tactical/strategic label gaps.

⭐ TWO CALLS, ONE AXIS EACH — never one call for both. Asking for the lateral
and longitudinal decision in a single response reintroduces, *in the prompt*,
exactly the mixing the factored vocabulary exists to remove (D-TAC1 measured the
mixed 5-way head at 0.7581 acc / 0.5313 macro-recall with `accelerate` NEVER
predicted, vs 0.9348 / 0.8290 factored).

  CALL A — the longitudinal REASON  (which of our six, and WHAT is it responding to)
  CALL B — the RELATIVE LANE TARGET (left / current / right) + exit-ahead

⛔ SAMPLING. Qwen3.5 runs in thinking mode BY DEFAULT (model card: *"Qwen3.5
models operate in thinking mode by default, generating thinking content
signified by <think>…</think>"*). Its card recommends `presence_penalty=1.5`
for general thinking tasks — **that is wrong here**: a presence penalty pushes
away from tokens already produced, and across thousands of forced-choice labels
drawn from six recurring tokens it biases against the FREQUENT classes.
`LANE_KEEP` alone is 85.33 % of the lateral axis; distorting that distribution
is the one thing a labeller must not do. ⇒ we use the card's *precise* profile
and, decisively, **constrain the verdict by logit masking** so sampling can move
the reasoning but never the label.

⛔ THE ECHO PASS. Every clip is labelled TWICE — with and without the ego
kinematics block — because agreement between a VLM that was shown ego numbers
and an ego-derived label is an ECHO, not corroboration (the review sheet's own
`B_TWO_OF_THREE` caveat). The delta between passes MEASURES the echo instead of
assuming it away.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from tanitad.lake.tac_str_labels import ABSTAIN
from tanitad.models.v6 import TACTICAL_GOAL_TOKENS, TACTICAL_LON_ACTIONS

__all__ = ["SAMPLING", "LON_VERDICTS", "LANE_VERDICTS", "build_lon_prompt",
           "build_lane_prompt", "parse_verdict", "allowed_verdict_tokens"]

#: The card's PRECISE profile, not its general-task one. See module docstring.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0,
            "max_context": 131072}

LON_VERDICTS: tuple[str, ...] = (*TACTICAL_LON_ACTIONS, ABSTAIN)
LANE_VERDICTS: tuple[str, ...] = ("LEFT", "CURRENT", "RIGHT", ABSTAIN)

_LON_GLOSS = {
    "FOLLOW": "maintaining a gap to a specific lead vehicle",
    "CRUISE": "free-flow; no object is governing the speed",
    "YIELD_MERGE": "giving way to a specific merging or crossing agent",
    "BRAKE_TO": "decelerating toward a specific point (stop line, light, obstacle)",
    "CREEP": "very slow advance with restricted visibility or a blocked path",
    "HOLD": "stationary and remaining stationary",
}

_COMMON = """You are labelling driving video for a research dataset. You see a clip
spanning t=-1.0s to t=+6.0s. t=0 is the decision moment; the band 2-6s is the
window being labelled. THE FUTURE IS SHOWN ON PURPOSE — use it to resolve what
actually happened rather than guessing intent.

Answer in two parts:
  1) EVIDENCE - what you actually see. Be specific and concrete.
  2) VERDICT  - exactly one token from the allowed list, on its own final line,
                formatted as: VERDICT: <TOKEN>

⛔ ABSTAIN is a real answer and carries no penalty. Use it whenever the evidence
does not decide the question - occlusion, darkness, no visible markings, an
ambiguous scene. A guess is worse than an abstention for this dataset."""


@dataclass(frozen=True)
class ClipContext:
    """Everything non-visual the prompt may carry. ``ego`` is withheld on the
    echo-control pass."""

    lead_gap_m: float | None = None
    lead_time_gap_s: float | None = None
    lead_closing_ms: float | None = None
    v0_ms: float | None = None
    v_end_ms: float | None = None
    alpamayo_magnitude: str | None = None
    alpamayo_cot: str | None = None

    def block(self, *, with_ego: bool) -> str:
        out = []
        if self.lead_gap_m is not None:
            # The obstacle join makes "is there a lead" a GIVEN rather than a
            # perception question the VLM might get wrong.
            out.append(f"- Lead vehicle (from our detector, treat as fact): gap "
                       f"{self.lead_gap_m:.1f} m"
                       + (f", time-gap {self.lead_time_gap_s:.1f} s"
                          if self.lead_time_gap_s is not None else "")
                       + (f", closing {self.lead_closing_ms:+.1f} m/s"
                          if self.lead_closing_ms is not None else ""))
        else:
            out.append("- Lead vehicle (from our detector, treat as fact): NONE "
                       "within range")
        if with_ego and self.v0_ms is not None:
            out.append(f"- Ego speed: {self.v0_ms:.1f} m/s at t=0"
                       + (f" -> {self.v_end_ms:.1f} m/s at t=+6s"
                          if self.v_end_ms is not None else ""))
        if self.alpamayo_magnitude:
            # ⚠️ Labelled as a PRIOR, never as the answer. Presenting another
            # model's opinion as ground truth imports its errors as anchors.
            out.append(f"- A different model's opinion (a PRIOR ONLY, it may be "
                       f"wrong, do not defer to it): motion described as "
                       f"\"{self.alpamayo_magnitude}\""
                       + (f"; its stated reason: \"{self.alpamayo_cot}\""
                          if self.alpamayo_cot else ""))
        return "\n".join(out)


def build_lon_prompt(ctx: ClipContext, *, with_ego: bool = True) -> str:
    """CALL A — the longitudinal REASON. Referent before verdict."""
    opts = "\n".join(f"  {t:<12} = {_LON_GLOSS[t]}" for t in TACTICAL_LON_ACTIONS)
    return f"""{_COMMON}

CONTEXT
{ctx.block(with_ego=with_ego)}

QUESTION - what is the ego's LONGITUDINAL behaviour governed by, over 2-6s?

Allowed verdicts:
{opts}
  {ABSTAIN:<12} = the evidence does not decide it

EVIDENCE - answer these before the verdict:
  a) Name the ONE object or road feature the ego is responding to (e.g. "the
     white van ahead in our lane", "the stop line at the junction", "nothing -
     open road"). If you cannot name it, the answer is {ABSTAIN}.
  b) Say what in the video tells you the ego is responding to it.

⛔ Do NOT answer with how fast the ego is going or how hard it brakes. The
magnitude is already known. The question is WHY - the same deceleration is
consistent with several of these tokens, and only the referent separates them.

VERDICT: <one token>"""


def build_lane_prompt(ctx: ClipContext, *, with_ego: bool = True) -> str:
    """CALL B — the RELATIVE lane target, and the exit predicate.

    ⭐ The third evidence question is the one that matters: the geometric gate
    this replaces died because **15 of 19 firings had a lateral offset FULLY
    EXPLAINED by constant-curvature road following**. Ego geometry cannot tell a
    lane change from a curving road; a viewer watching the MARKINGS can. Asking
    for that discrimination explicitly is the point of the whole call.
    """
    return f"""{_COMMON}

CONTEXT
{ctx.block(with_ego=with_ego)}

QUESTION - relative to the lane the ego occupies at t=0, which lane does it
occupy at the END of the clip?

Allowed verdicts:
  LEFT     = one lane to the left of the t=0 lane
  CURRENT  = the same lane
  RIGHT    = one lane to the right of the t=0 lane
  {ABSTAIN}  = markings not visible, or the answer is not determinable

EVIDENCE - answer these before the verdict:
  a) At t=0, which lane markings bound the ego? Describe them (solid/dashed,
     left/right, colour if visible).
  b) At the end of the clip, which markings bound the ego?
  c) ⭐ CRITICAL - did the ego CROSS a lane marking, or did the ROAD CURVE
     while the ego stayed between the same two markings? These look different:
     a crossing shows a marking passing under the vehicle. Say which you saw.
  d) Is there an exit, off-ramp or slip road ahead? If yes, on which side?

If your answer to (c) is "the road curved", the verdict is CURRENT.

VERDICT: <one token>"""


def build_sign_prompt(ctx: ClipContext, *, with_ego: bool = True) -> str:
    """CALL C — ⭐ the DIRECTION SIGN, which is what un-gates `ROUTE_TO`.

    ⛔ WHY THIS CALL EXISTS AT ALL. `ROUTE_TO` was gated because it names a
    navigation DESTINATION and *"no destination exists anywhere in the corpus;
    a VLM asked for one would invent it"* (G1 CLOSED at 0/31). The PI's
    observation dissolves that: **a direction sign puts the destination IN THE
    PIXELS**, and the future path says which of the signed branches the ego
    actually took. The label is then two observations, not an invention.

    ⛔ AND THE GUARD THAT MUST TRAVEL WITH IT — MEASURED, in this corpus.
    `v6.py` records that the sign detector's two HIGHEST-confidence detections
    were a **dashboard `30` roundel (0.927)** and a **hoarding (0.778)**, both
    *above* true signs — so *"a confidence threshold removes the harmless errors
    and KEEPS the harmful ones"*. The roundel is the **EGO SPEEDOMETER**: a
    sign read off the dashboard is *"an ego echo arriving through the vision
    channel, which a vision-only admissibility audit does not watch"*. ⇒ Q(a)
    below asks explicitly whether the sign is OUTSIDE the vehicle, and an
    interior sign is a hard ABSTAIN. This is not defensive boilerplate; it is
    the one failure mode this corpus has already demonstrated.
    """
    return f"""{_COMMON}

CONTEXT
{ctx.block(with_ego=with_ego)}

QUESTION - is the ego following a signed route, and along which branch?

Allowed verdicts:
  LEFT      = a direction sign is legible and the ego followed its LEFT branch
  STRAIGHT  = ... its STRAIGHT-ON branch
  RIGHT     = ... its RIGHT branch
  {ABSTAIN}   = no legible direction sign, or the ego did not follow one

EVIDENCE - answer these before the verdict:
  a) ⛔ FIRST: is the sign OUTSIDE the vehicle, mounted on a post, gantry or
     wall? Anything on the dashboard, windscreen or inside the cabin is NOT a
     road sign - if that is all you see, the verdict is {ABSTAIN}.
  b) Read the sign. Quote the destination text and the arrow direction for each
     branch it names (e.g. "Zentrum <-, A9 ^").
  c) Which branch did the ego actually take? Say what in the later frames shows
     it.
  d) Does (c) match one of the branches in (b)? If the ego went a way the sign
     did not name, the verdict is {ABSTAIN}.

⛔ Do NOT infer a destination from road shape, lane markings or traffic. If you
cannot READ it on a sign, it is not evidence for this question.

VERDICT: <one token>"""


SIGN_VERDICTS: tuple[str, ...] = ("LEFT", "STRAIGHT", "RIGHT", ABSTAIN)


def allowed_verdict_tokens(kind: str) -> tuple[str, ...]:
    if kind == "lon":
        return LON_VERDICTS
    if kind == "lane":
        return LANE_VERDICTS
    if kind == "sign":
        return SIGN_VERDICTS
    raise ValueError(f"kind must be 'lon', 'lane' or 'sign', got {kind!r}")


_VERDICT_RE = re.compile(r"VERDICT:\s*([A-Z_]+)\s*$", re.M)


def parse_verdict(text: str, kind: str) -> tuple[str, str]:
    """-> (verdict, thinking). ⛔ Strict: an unparseable or out-of-vocabulary
    answer becomes ABSTAIN with a reason, NEVER a nearest-match guess — a
    labeller that silently repairs its own output is a labeller you cannot audit.
    """
    allowed = allowed_verdict_tokens(kind)
    think = ""
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    if m:
        think = m.group(1).strip()
        text = text[m.end():]
    hits = _VERDICT_RE.findall(text)
    if not hits:
        return ABSTAIN, think
    tok = hits[-1].strip().upper()
    return (tok if tok in allowed else ABSTAIN), think


def main(argv=None) -> int:
    """Print both prompts for eyeballing before any GPU is spent."""
    ctx = ClipContext(lead_gap_m=18.4, lead_time_gap_s=1.6, lead_closing_ms=-1.2,
                      v0_ms=11.5, v_end_ms=7.0,
                      alpamayo_magnitude="Gentle Deceleration",
                      alpamayo_cot="Slow down due to the lead vehicle ahead.")
    print("=" * 78, "\nCALL A — longitudinal reason (with ego)\n", "=" * 78)
    print(build_lon_prompt(ctx))
    print("\n", "=" * 78, "\nCALL B — relative lane target (ECHO CONTROL: no ego)\n", "=" * 78)
    print(build_lane_prompt(ctx, with_ego=False))
    print("\nsampling:", json.dumps(SAMPLING))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
