"""WHICH forward channels an RL rollout may be fed — declared BY THE SEAM THAT OWNS EACH.

⛔⛔ WHAT THIS MODULE IS NOT: it is not the place the reasons live. It defines the
SHAPE of an exclusion and the union that reads them, and nothing else. Every reason,
every piece of evidence and every route back lives in the seam module that owns the
channel, because a reason moved away from its seam is a reason that rots: the seam
changes, the central list does not, and the exclusion silently starts covering
something else. That is the exact failure ``refc_adapter.FORWARD_KEYS`` already had
twice, one level up.

WHY AN EXCLUSION NEEDS MORE THAN A NAME
=======================================
``tanitad.rl.refc_adapter.forward_kwargs`` builds its kwargs by iterating
``FORWARD_KEYS``, so a channel absent from that tuple is **never passed at all** — no
error, no warning. Two channels drifted out of it silently (``agent_gt``, then
``gp_point``/``gp_valid``). The fix was to DERIVE the requirement from the live
signature and let a seam declare the channels that must be withheld.

⚠️ But a flat tuple of names — which is what
``goal_point.DIAGNOSTIC_ONLY_FORWARD_KWARGS`` was — destroys the only thing a reader
needs. The five channels currently withheld are withheld for **three different
reasons with three different routes back**:

    ``gp_point`` / ``gp_valid``   NEVER admissible. The goal point IS the label.
    ``nav_args``                  not admissible **AS BUILT** — the block ships a
                                  ``time_s`` slot no nav system can supply. A
                                  distance-only mode unblocks it.
    ``v_max_ms`` / ``v_max_valid`` not admissible **FROM THIS CORPUS** — the PI ruled
                                  the channel legitimate in principle, but the value
                                  we have is read off the ego's realised future. A map
                                  source unblocks it.

A list of five names states none of that, and after one personnel change the two
temporary exclusions are indistinguishable from the permanent one. **An exclusion
with no stated route back reads as permanent**, and two of these three are not.

⇒ :class:`ChannelExclusion` therefore REFUSES to be constructed without a reason, an
unblock condition and an evidence class. A declaration that cannot say why is not a
declaration, it is a suppression with better manners.

HOW A SEAM DECLARES
===================
Export ``FORWARD_EXCLUSIONS`` — a tuple of :class:`ChannelExclusion` — from the module
that owns the channel, and add that module to :data:`SEAM_MODULES` here. The list of
seam MODULES is central (it must be, or an unimported seam would silently stop
excluding and the adapter would plumb a label); the CONTENT of each declaration is
not, and must never be moved here.

Tier: none. ⛔ This is a WIRING-CONTRACT surface, not an eval surface — no model
produces a trajectory anywhere in this module, so an eval tier or a four-family
metric table would be a category error. Evidence class of the claims above:
PUBLISHED-CODE (the seam modules' own citations) and MEASURED (quoted per record).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

__all__ = ["ChannelExclusion", "ExclusionDeclarationError", "SEAM_MODULES",
           "forward_exclusions", "excluded_channels", "exclusion_by_channel",
           "format_exclusions"]


class ExclusionDeclarationError(ValueError):
    """A seam's exclusion declaration is malformed, or two seams disagree."""


#: The shortest text that can carry a reason / a route back. Not a style rule:
#: ``reason="label"`` and ``unblock="tbd"`` are exactly the placeholders that make an
#: exclusion unreviewable, and the point of this module is that they cannot be typed.
_MIN_REASON = 40
_MIN_UNBLOCK = 20
_MIN_EVIDENCE = 8

#: Placeholders a reviewer would have to chase. Refused by name so the refusal can say
#: what is wrong instead of only that the string is short.
_PLACEHOLDERS = frozenset({"tbd", "todo", "n/a", "na", "none", "-", "?", "unknown",
                           "see above", "fixme", "xxx"})


#: What each required field is FOR, quoted verbatim in the refusals below so the
#: message says what to write rather than only that something is missing.
_WHY = {
    "owner": "which seam made the call",
    "reason": "where an RL rollout would get the value and why that source is "
              "not admissible",
    "unblock": "what would make the channel admissible (or why nothing could)",
    "evidence": "the evidence class and its citation",
}


@dataclass(frozen=True)
class ChannelExclusion:
    """One forward channel an RL rollout must NOT be fed, and why.

    ``channel``   the forward kwarg name, exactly as ``RefCV3Model.forward`` spells it.
    ``owner``     the seam that owns the channel and made this call.
    ``reason``    WHERE an RL rollout would have to get the value, and why that source
                  is not admissible. ⛔ Not "it is a label" — the reader must be able
                  to check the claim without asking anyone.
    ``unblock``   WHAT WOULD MAKE IT ADMISSIBLE. For a permanent exclusion this states
                  why no route exists; it may not be empty either way, because "no
                  route" and "nobody wrote one down" must not look the same.
    ``evidence``  the evidence class and its citation, per the operating standard.
    ``permanent`` True only when no change to the code or the corpus could make the
                  channel admissible. ⚠️ Default False: temporary is the humbler
                  default, and a permanent call should have to be typed.
    """

    channel: str
    owner: str
    reason: str
    unblock: str
    evidence: str
    permanent: bool = False
    #: free-form, for a seam that wants to point at a package or a ticket
    refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not (isinstance(self.channel, str) and self.channel.isidentifier()):
            raise ExclusionDeclarationError(
                f"channel must be the forward kwarg's identifier, got "
                f"{self.channel!r}. A name that is not an identifier can never "
                f"match a signature parameter, so the exclusion would cover "
                f"nothing while looking like it covers something.")
        for name, value, floor in (("owner", self.owner, 3),
                                   ("reason", self.reason, _MIN_REASON),
                                   ("unblock", self.unblock, _MIN_UNBLOCK),
                                   ("evidence", self.evidence, _MIN_EVIDENCE)):
            if not isinstance(value, str):
                raise ExclusionDeclarationError(
                    f"{self.channel}: {name} must be a string, got {type(value)!r}")
            text = value.strip()
            if not text:
                raise ExclusionDeclarationError(
                    f"{self.channel}: {name} is EMPTY. An exclusion that cannot say "
                    f"{_WHY[name]} is indistinguishable from a channel somebody "
                    f"forgot to plumb -- which is the defect this whole surface "
                    f"exists to make impossible.")
            if text.lower().strip(".") in _PLACEHOLDERS:
                raise ExclusionDeclarationError(
                    f"{self.channel}: {name} is the placeholder {text!r}. Write "
                    f"{_WHY[name]}, or do not exclude the channel.")
            if len(text) < floor:
                raise ExclusionDeclarationError(
                    f"{self.channel}: {name} is {len(text)} chars ({text!r}); at "
                    f"least {floor} are needed to state {_WHY[name]}. This floor is "
                    f"not a style rule -- a one-word reason cannot be reviewed, and "
                    f"an unreviewable exclusion is a suppression.")

    # -- convenience, so callers do not re-derive the same two shapes ---------
    @property
    def summary(self) -> str:
        route = "PERMANENT" if self.permanent else "TEMPORARY"
        return f"{self.channel} [{route}, owner={self.owner}]: {self.reason}"

    def to_dict(self) -> dict:
        """The shape a run record / preflight artifact banks."""
        return {"channel": self.channel, "owner": self.owner,
                "reason": self.reason, "unblock": self.unblock,
                "evidence": self.evidence, "permanent": bool(self.permanent),
                "refs": list(self.refs)}


#: ⛔ THE MODULES THAT DECLARE. Central by necessity, not by preference: a registry
#: populated as a side effect of importing a seam would silently hold NOTHING if the
#: seam were not imported, and the adapter would then plumb a label with no guard
#: firing. So the list of PARTICIPANTS is here and the CONTENT is not, and
#: :func:`forward_exclusions` fails loud when a listed module does not declare.
#:
#: ⭐ Adding a seam here is the cheap half. The expensive half — the reason, the
#: route back and the evidence — belongs in the seam's own module, next to the code
#: that creates the channel.
SEAM_MODULES: tuple[str, ...] = (
    "tanitad.refs.goal_point",          # gp_point / gp_valid   (E15, S7)
    "tanitad.models.nav_conditioning",  # nav_args              (E13b)
    "tanitad.refs.max_speed_input",     # v_max_ms / v_max_valid (E16)
)


def forward_exclusions(modules: tuple[str, ...] | None = None
                       ) -> tuple[ChannelExclusion, ...]:
    """The UNION of every seam's declaration, in seam order.

    ⛔ Raises rather than skipping. A seam that cannot be imported, or that is listed
    and does not declare, is a HOLE in the contract, and a hole that returns an empty
    tuple looks exactly like "nothing is excluded" — the vacuity the guards downstream
    are built to refuse. Failing here is the only reading that cannot be mistaken for
    a clean bill of health.
    """
    mods = SEAM_MODULES if modules is None else tuple(modules)
    out: list[ChannelExclusion] = []
    seen: dict[str, ChannelExclusion] = {}
    for dotted in mods:
        try:
            mod = importlib.import_module(dotted)
        except Exception as exc:                       # noqa: BLE001 — re-raised
            raise ExclusionDeclarationError(
                f"seam module {dotted!r} is listed in SEAM_MODULES but could not be "
                f"imported ({type(exc).__name__}: {exc}). Refusing rather than "
                f"skipping: a skipped seam contributes NO exclusions, which reads "
                f"downstream as 'this channel is fine to plumb'.") from exc
        decls = getattr(mod, "FORWARD_EXCLUSIONS", None)
        if decls is None:
            raise ExclusionDeclarationError(
                f"seam module {dotted!r} is listed in SEAM_MODULES but exports no "
                f"FORWARD_EXCLUSIONS. Either declare there, or remove it from the "
                f"list -- a listed seam that declares nothing is a guard that has "
                f"quietly stopped guarding.")
        for d in decls:
            if not isinstance(d, ChannelExclusion):
                raise ExclusionDeclarationError(
                    f"{dotted} declared {d!r}, which is not a ChannelExclusion. The "
                    f"type is what enforces the reason and the unblock condition; a "
                    f"bare string would bypass both.")
            prev = seen.get(d.channel)
            if prev is not None and prev != d:
                raise ExclusionDeclarationError(
                    f"two seams declare {d.channel!r} differently: {prev.owner} says "
                    f"{prev.reason!r}, {d.owner} says {d.reason!r}. One channel, one "
                    f"owner -- resolve the ownership rather than letting the union "
                    f"pick whichever import ran first.")
            if prev is None:
                seen[d.channel] = d
                out.append(d)
    return tuple(out)


def excluded_channels(modules: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Just the names — for the set arithmetic the adapter and the guard do."""
    return tuple(d.channel for d in forward_exclusions(modules))


def exclusion_by_channel(modules: tuple[str, ...] | None = None
                         ) -> dict[str, ChannelExclusion]:
    """Name -> declaration, so a refusal can quote the reason at the reader."""
    return {d.channel: d for d in forward_exclusions(modules)}


def format_exclusions(decls=None) -> str:
    """A human-readable block for a refusal message or a preflight banner."""
    decls = forward_exclusions() if decls is None else tuple(decls)
    lines = []
    for d in decls:
        route = "PERMANENT" if d.permanent else "TEMPORARY"
        lines.append(f"  - {d.channel}  [{route}]  owner={d.owner}")
        lines.append(f"      reason : {d.reason}")
        lines.append(f"      unblock: {d.unblock}")
        lines.append(f"      evidence: {d.evidence}")
    return "\n".join(lines)
