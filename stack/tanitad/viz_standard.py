"""THE VIZ STANDARD, as an enforced contract rather than a convention.

The standing standard (binding) is that every trajectory view shows, TOGETHER:

    1. camera      -- GT + predicted path projected into the front image
    2. bev         -- the same paths in metres, top-down (calibration-free)
    3. tactical    -- the model's DECODED manoeuvre
    4. strategic   -- the model's route / goal decision
    5. ade         -- the scalar error

with a BEV-only fallback when the camera calibration is unrecoverable.

Until now that was a docstring claim plus a README table, and it failed in
exactly the way an unchecked claim fails: ``tanitad/resim/README.md`` asserted
that the SPA's strategic row was ``route_logits`` (a model OUTPUT) while the
code shipped ``nav_cmd`` -- a route command DERIVED FROM THE EPISODE'S OWN
FUTURE POSES and FED TO the model (``tanitad/replay/arms.py:526``, annotated
"strategic input" at ``tanitad/replay/engine.py:220``). Thirty-nine green tests
missed it because each asserted that a value was PRESENT and LABELLED, and none
asserted WHAT IT MEANT. That is the nav-echo class: flagship v1's route head
was an exact bijection of the nav we fed it (369/369) and scored 1.0000 -- an
echo of its own input read as skill.

This module is the per-family reporting discipline applied to pixels:

* a rendered frame declares FIVE element records; ``check_frame`` refuses to
  let a renderer proceed with one silently missing;
* ``source`` is MANDATORY when an element is present -- one declared line makes
  an input-masquerading-as-an-output visible at a glance;
* ``kind`` is MANDATORY and is RENDERED -- a ``given_input`` / ``gt_label``
  value draws with a privileged marker, because a GT-derived route on screen is
  optimistic by construction and the frame must say so;
* ``reason`` is MANDATORY when unavailable AND IS DRAWN ON THE FRAME, never
  only in a sidecar. An arm with no policy brain renders
  ``tactical: unavailable -- arm 'refa' has no tactical head``, never a blank;
* ``PREDICTION_ELEMENTS`` (``tactical`` / ``strategic``) REFUSE a privileged
  kind outright. The section-3.1 defect is not merely detectable here, it is
  unconstructible: putting the GT-derived nav command in the strategic slot
  raises.

``conditioned_on`` extends the design one step further, and is the
admissibility check the programme already binds to in prose: *"for any goal
signal, ask -- could this have been computed from the situation classifier's
output?"*. REF-B's ``route_logits`` IS a model output with its own auxiliary
CE, but ``StrategicHead`` is FiLM-conditioned on ``nav_emb(nav_cmd)``
(``tanitad/refs/refb.py:295-305``), so the prediction is computed UNDER the
GT-derived input. That is not a reason to hide it; it is a reason to declare
it, and the HUD prints ``(cond. nav_cmd)`` beside the value.

Deliberately stdlib-only: the three renderer families that must consume it are
PIL (``taniteval/taniteval/corpus_overlay.py``), JS/canvas (the TanitResim SPA,
via its Python exporter) and cv2 (``stack/experiments/alpasim-gsplat/
overlay_video.py``) -- and the cv2 one runs on the Jetson. Importing this
module must never pull torch, numpy or PIL.

    from tanitad.viz_standard import VizElement, check_frame, hud_lines

    els = check_frame([
        VizElement.present("camera", "fan", source="to_image_plane(wp, cam)",
                           kind="model_output"),
        VizElement.present("bev", "fan", source="ArmOutput.waypoints [m]",
                           kind="model_output"),
        VizElement.present("tactical", "turn_left",
                           source="maneuver_probs.argmax(-1)",
                           kind="model_output"),
        VizElement.unavailable("strategic", "arm 'main' has no route head"),
        VizElement.present("ade", "0.42 m", source="mean L2", kind="derived"),
    ], where="main @ ep0/step3")
    for line in hud_lines(els):
        draw(line)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

__all__ = [
    "STANDARD_ELEMENTS", "AUX_ELEMENTS", "KNOWN_ELEMENTS",
    "PREDICTION_ELEMENTS", "KINDS", "PRIVILEGED_KINDS", "STATES",
    "VizElement", "VizStandardError", "check_frame", "hud_lines",
    "to_json", "from_json",
]


# The five elements every trajectory view must declare. A renderer may declare
# more (see AUX_ELEMENTS) but never fewer.
STANDARD_ELEMENTS: tuple[str, ...] = (
    "camera", "bev", "tactical", "strategic", "ade")

# Optional, declared the same way when shown. `strategic_input` is where a
# GT-derived navigator command belongs -- NOT in `strategic`.
AUX_ELEMENTS: tuple[str, ...] = ("strategic_input", "tactical_gt")

KNOWN_ELEMENTS: tuple[str, ...] = STANDARD_ELEMENTS + AUX_ELEMENTS

# Slots that answer "what did the MODEL decide?". A privileged kind in one of
# these is the nav-echo defect and is refused by check_frame.
PREDICTION_ELEMENTS: tuple[str, ...] = ("tactical", "strategic")

STATES: tuple[str, ...] = ("present", "unavailable")

KINDS: tuple[str, ...] = ("model_output", "given_input", "gt_label", "derived")

# Kinds that are NOT the model's own decision. They render with a marker.
PRIVILEGED_KINDS: tuple[str, ...] = ("given_input", "gt_label")

_MARKER = {"given_input": "GIVEN INPUT, not a prediction",
           "gt_label": "GROUND TRUTH label, not a prediction"}


class VizStandardError(RuntimeError):
    """A frame violated the viz standard. Raised instead of rendering.

    Fail loud by design: a silently missing element is exactly how a viewer
    concludes "the model has no strategic output" or, worse, reads an input as
    one.
    """


def _blank(s: object) -> bool:
    return s is None or (isinstance(s, str) and not s.strip())


@dataclass(frozen=True)
class VizElement:
    """One declared element of one rendered frame (or of one episode).

    element        one of KNOWN_ELEMENTS
    state          "present" | "unavailable"
    value          the displayed value (present only) -- already formatted
    source         REQUIRED when present: the exact expression the value came
                   from, e.g. "strategic_policy.route_logits.argmax(-1)" or
                   "refb_labels.nav_command(episode.poses, t)"
    kind           REQUIRED when present: model_output | given_input |
                   gt_label | derived
    reason         REQUIRED when unavailable; it is DRAWN on the frame
    n              frames / windows the record applies to (optional)
    conditioned_on privileged inputs the value was computed UNDER. Empty for a
                   clean model output; ("nav_cmd",) for a route head that is
                   FiLM-conditioned on the GT-derived navigator command.
    """

    element: str
    state: str
    value: str | None = None
    source: str | None = None
    kind: str | None = None
    reason: str | None = None
    n: int | None = None
    conditioned_on: tuple[str, ...] = field(default=())

    # -- construction ------------------------------------------------------
    @classmethod
    def present(cls, element: str, value: object, *, source: str, kind: str,
                n: int | None = None,
                conditioned_on: Sequence[str] = ()) -> "VizElement":
        return cls(element=element, state="present",
                   value=None if value is None else str(value),
                   source=source, kind=kind, n=n,
                   conditioned_on=tuple(conditioned_on))

    @classmethod
    def unavailable(cls, element: str, reason: str,
                    n: int | None = None) -> "VizElement":
        return cls(element=element, state="unavailable", reason=reason, n=n)

    def __post_init__(self) -> None:
        if self.element not in KNOWN_ELEMENTS:
            raise VizStandardError(
                f"unknown viz element {self.element!r} -- known: "
                f"{', '.join(KNOWN_ELEMENTS)}")
        if self.state not in STATES:
            raise VizStandardError(
                f"{self.element}: state must be one of {STATES}, "
                f"got {self.state!r}")
        if self.state == "present":
            if _blank(self.source):
                raise VizStandardError(
                    f"{self.element}: a PRESENT element must declare its "
                    f"`source` -- the exact expression its value came from. "
                    f"An undeclared source is how a ground-truth-derived input "
                    f"passed for a model output.")
            if self.kind not in KINDS:
                raise VizStandardError(
                    f"{self.element}: a PRESENT element must declare a `kind` "
                    f"in {KINDS}, got {self.kind!r}")
            if not _blank(self.reason):
                raise VizStandardError(
                    f"{self.element}: a PRESENT element must not carry a "
                    f"`reason` (reasons explain absence)")
        else:
            if _blank(self.reason):
                raise VizStandardError(
                    f"{self.element}: an UNAVAILABLE element must declare a "
                    f"`reason`, and it is drawn on the frame -- a blank slot "
                    f"is indistinguishable from an element nobody implemented")
            if not (_blank(self.source) and self.kind is None
                    and _blank(self.value)):
                raise VizStandardError(
                    f"{self.element}: an UNAVAILABLE element carries no "
                    f"value/source/kind")
        if any(_blank(c) for c in self.conditioned_on):
            raise VizStandardError(
                f"{self.element}: `conditioned_on` entries must be non-empty")

    # -- reading -----------------------------------------------------------
    @property
    def privileged(self) -> bool:
        """True when the value is NOT the model's own decision."""
        return self.kind in PRIVILEGED_KINDS

    def marker(self) -> str:
        """The on-frame provenance marker; "" for a clean model output."""
        if self.state != "present":
            return ""
        bits = []
        if self.privileged:
            bits.append(_MARKER[self.kind])
        if self.conditioned_on:
            bits.append("cond. " + ", ".join(self.conditioned_on))
        return "  [" + " | ".join(bits) + "]" if bits else ""

    def hud_text(self) -> str:
        """The line a renderer draws. Never blank, in either state."""
        if self.state == "unavailable":
            return f"{self.element}: unavailable -- {self.reason}"
        shown = self.value if not _blank(self.value) else "shown"
        return f"{self.element}: {shown}{self.marker()}"

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return {"element": self.element, "state": self.state,
                "value": self.value, "source": self.source, "kind": self.kind,
                "reason": self.reason, "n": self.n,
                "conditioned_on": list(self.conditioned_on)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "VizElement":
        return cls(element=d["element"], state=d["state"],
                   value=d.get("value"), source=d.get("source"),
                   kind=d.get("kind"), reason=d.get("reason"), n=d.get("n"),
                   conditioned_on=tuple(d.get("conditioned_on") or ()))

    def with_n(self, n: int) -> "VizElement":
        return replace(self, n=n)


def check_frame(elements: Iterable[VizElement], *,
                where: str) -> tuple[VizElement, ...]:
    """Validate one frame's declarations. Return them, or raise.

    ``where`` names the frame in the error (arm / episode / step) -- a contract
    violation must be locatable without a debugger.

    Refuses, in order:
      1. a duplicate declaration of the same element;
      2. any of STANDARD_ELEMENTS missing -- an element may be *declared*
         unavailable, never silently absent;
      3. a PREDICTION slot filled by a privileged kind (given_input /
         gt_label). That is the nav-echo defect: an input rendered where a
         reader can only read it as the model's decision.
    """
    els = tuple(elements)
    seen: dict[str, VizElement] = {}
    for e in els:
        if not isinstance(e, VizElement):
            raise VizStandardError(
                f"{where}: expected VizElement records, got {type(e).__name__}")
        if e.element in seen:
            raise VizStandardError(
                f"{where}: element {e.element!r} declared twice")
        seen[e.element] = e

    missing = [e for e in STANDARD_ELEMENTS if e not in seen]
    if missing:
        raise VizStandardError(
            f"{where}: THE VIZ STANDARD requires all of "
            f"{', '.join(STANDARD_ELEMENTS)} to be declared; missing "
            f"{', '.join(missing)}. An element that cannot be computed is "
            f"declared UNAVAILABLE with a reason -- it is never omitted.")

    for name in PREDICTION_ELEMENTS:
        e = seen[name]
        if e.state == "present" and e.privileged:
            raise VizStandardError(
                f"{where}: {name!r} is a PREDICTION slot but was filled with "
                f"kind={e.kind!r} (source: {e.source!r}). A ground-truth-"
                f"derived input rendered as a prediction is the nav-echo "
                f"defect -- a value the model was GIVEN, read as a value the "
                f"model DECIDED. Put it in its own slot ({name}_input / "
                f"{name}_gt) and leave {name!r} unavailable with a reason.")
    return els


def hud_lines(elements: Iterable[VizElement]) -> list[str]:
    """One drawable line per element, in KNOWN_ELEMENTS order."""
    els = list(elements)
    order = {name: i for i, name in enumerate(KNOWN_ELEMENTS)}
    els.sort(key=lambda e: order.get(e.element, len(order)))
    return [e.hud_text() for e in els]


def to_json(elements: Iterable[VizElement]) -> list[dict]:
    """JSON-safe rows for a session bundle / render sidecar."""
    return [e.to_dict() for e in elements]


def from_json(rows: Iterable[Mapping]) -> tuple[VizElement, ...]:
    return tuple(VizElement.from_dict(r) for r in rows)
