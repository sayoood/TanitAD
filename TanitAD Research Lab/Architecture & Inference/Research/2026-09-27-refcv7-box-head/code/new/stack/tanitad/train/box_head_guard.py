"""The refcv7 BOX-HEAD launch requirement (SPEC_REFCV7 §14 A9, §15 A10): the A9 levers must be ON in argv AND BUILT.

G-DVB (``declared_vs_built``) proves argv == built for every flag; it cannot know that a refcv7 launch must SET these
flags -- an argv that simply omits them trains the pre-A9 head and every G-DVB check is green. This module is the
SPEC literal, the way ``declared_vs_built.REFCV7_REQUIRED_ON`` is for the selection mechanisms: the launch gate calls
:func:`check_refcv7_box_required` and a non-empty list refuses the launch.

LITERALS (A9): focal presence, prior 0.01, per-layer supervision, VIS-1 with a sidecar, 300 queries on BOTH slot heads
(the box3d decoder and the planner's learned agent head).
"""
from __future__ import annotations

from typing import Any

__all__ = ["REFCV7_BOX_REQUIRED", "REFCV7_N_QUERIES", "check_refcv7_box_required"]

#: argv dest -> the value a refcv7 launch must carry (SPEC_REFCV7 §14 A9 R1-R3). LITERALS, never read from the code.
REFCV7_BOX_REQUIRED: dict[str, Any] = {
    "slot_presence_loss": "focal",
    "slot_presence_prior": 0.01,
    "slot_deep_supervision": True,
    "slot_vis1": True,
}
#: A9 R4: both slot heads build 300 queries.
REFCV7_N_QUERIES: int = 300


def _mm(lever, declared, built, where, why):
    from tanitad.train.declared_vs_built import Mismatch
    return Mismatch(lever, declared, built, where, why)


def check_refcv7_box_required(model, args) -> list:
    """-> a Mismatch per A9 lever that is OFF in argv, or not BUILT on either slot head ([] == a refcv7 box build)."""
    out = []
    for dest, want in REFCV7_BOX_REQUIRED.items():
        got = getattr(args, dest, None)
        ok = (abs(float(got) - float(want)) <= 1e-12) if isinstance(want, float) and got is not None else got == want
        if not ok:
            out.append(_mm("--" + dest.replace("_", "-"), want, got, "argv", "SPEC_REFCV7 §14 (A9)"))
    if not getattr(args, "vis1_sidecar", None):
        out.append(_mm("--vis1-sidecar", "a sidecar path", None, "argv", "A9 R3: VIS-1 is precomputed"))
    core = getattr(model, "core", model)
    ah = getattr(core, "agent_head", None)
    br = getattr(model, "_perception", None)
    bd = getattr(br, "box_dec", None) if br is not None else None
    heads = [("agent", ah, getattr(getattr(core, "cfg", None), "agents", None)), ("box3d", bd, getattr(br, "cfg", None))]
    for name, dec, lcfg in heads:
        if dec is None or not hasattr(dec, "deep_supervision"):
            out.append(_mm(f"{name} slot head", "BUILT (learned)", "not built", name,
                           "A9 applies to BOTH slot heads (box3d + the planner's agent head)"))
            continue
        if int(getattr(dec, "n_queries", -1)) != REFCV7_N_QUERIES:
            out.append(_mm("n_queries", REFCV7_N_QUERIES, getattr(dec, "n_queries", None), f"{name} decoder",
                           "A9 R4: 300 queries"))
        if not bool(dec.deep_supervision):
            out.append(_mm("--slot-deep-supervision", True, False, f"{name}.deep_supervision", "A9 R2"))
        if abs(float(getattr(dec, "presence_prior", -1.0)) - 0.01) > 1e-12:
            out.append(_mm("--slot-presence-prior", 0.01, getattr(dec, "presence_prior", None),
                           f"{name}.presence_prior", "A9 R1"))
        if str(getattr(lcfg, "presence_loss", "")) != "focal":
            out.append(_mm("--slot-presence-loss", "focal", getattr(lcfg, "presence_loss", None),
                           f"{name} loss config", "A9 R1"))
        if not bool(getattr(lcfg, "vis1", False)):
            out.append(_mm("--slot-vis1", True, getattr(lcfg, "vis1", None), f"{name} loss config", "A9 R3"))
    if not bool(getattr(model, "_vis1", False)):
        out.append(_mm("--slot-vis1", True, getattr(model, "_vis1", None), "model._vis1", "A9 R3"))
    return out
