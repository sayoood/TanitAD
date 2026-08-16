"""X2 — THE PRODUCER SIDE: bank the 60-step plan so `seam_probe.py` can run.

⛔ WHY THIS MODULE EXISTS. F-16's seam probe (`taniteval/taniteval/seam.py`,
`taniteval/tools/seam_probe.py`) is complete, self-tested and validated — and
has produced **ZERO real-arm numbers**, because the emitted plan exists only
inside `V6Stack.emit`'s return value and the trainer saves only checkpoints
(`SEAM_INSTRUMENT.md` §8, three independent probes: no `torch.save` of
`emit()`, no `*plan*.pt` on disk, and the one banked fan — REF-C's — is 4
waypoints entirely inside the operative band, so it has no 2 s boundary to
test). A built instrument that nothing feeds is the same class of stranding
the operating standard exists to prevent. **This is the six lines
`SEAM_INSTRUMENT.md` §8 specifies, made into a function with its refusals.**

⛔ IT PRODUCES A DUMP AND NOTHING ELSE. No loss, no gradient, no metric, no
verdict. `seam_probe.py` remains the only thing that scores a seam, and the
one thing that must never happen is a seam-repair term appearing because this
banked something.

ZERO EXTRA GPU
--------------
The intended call site already HAS the plan: `V6Stack.forward` puts
`emit()`'s output at `out["plan"]`, so banking it is `.detach().cpu()` plus a
`torch.save`. Nothing is re-run, no extra forward, no extra rollout. On the
training path it is called at the CHECKPOINT-SAVE boundary (not per step) and
is DEFAULT-OFF — `--dump-seam-plan` unset means this module is never imported.

⚠️ RUN IT ON S-T OR LATER, NOT S-W
----------------------------------
In S-W the planner is absent and the emission head sits at its zero-init, so
every control is exactly (0, 0) and the probe correctly returns
**DEGENERATE** — the right answer, and not a pass. :func:`plan_is_degenerate`
computes that here so a caller can refuse to bank a dump that cannot answer
the question, rather than discovering it after the fact. The live v6F S-W run
is exactly this case.

⚠️ THE TIER IS NOT OPTIONAL. `EVAL_DOCTRINE.md` forbids an un-tiered number
and `seam_probe` hard-refuses a dump with no tier. It is a REQUIRED argument
here rather than a defaulted one: a default would silently stamp somebody
else's tier onto a dump, which is worse than refusing. The continuity blocks
consume the emitted plan alone (no GT, no recorded future actions), so they
are tier-invariant and `T1` is correct for them; the BAND-ERROR block compares
against `gt`, so it inherits whatever is declared here — pass the tier the
`gt` came from, and pass no `gt` at all if there is none.
"""
from __future__ import annotations

__all__ = [
    "SEAM_DUMP_KEYS", "SeamDumpError", "plan_is_degenerate",
    "seam_dump_from_plan", "save_seam_dump",
]

#: Exactly the schema `seam_probe.load_dump` consumes (its module docstring is
#: the law; this tuple is the producer's mirror of the REQUIRED half).
SEAM_DUMP_KEYS = ("eid", "controls", "waypoints", "sel", "tier", "arm",
                  "plan_steps", "dt", "op_band_s", "tac_band_s")
#: Below this, every control in the dump is indistinguishable from the
#: zero-init emission head — i.e. S-W. Chosen as an absolute on (a, kappa),
#: which are both O(1) in their own units after the `tanh` envelopes.
DEGENERATE_ABS = 1e-8


class SeamDumpError(ValueError):
    """A refusal to BANK a plan. Its own type so a caller can tell "this plan
    cannot answer the seam question" apart from an I/O error."""


def plan_is_degenerate(controls, *, atol: float = DEGENERATE_ABS) -> bool:
    """Is every control identically zero (the S-W zero-init emission head)?

    ⛔ This is a PROPERTY OF THE ARM, not a defect in the dump — and it is why
    it is reported rather than raised by default. A DEGENERATE verdict from
    the probe is the correct answer for S-W; banking such a dump is only wrong
    if somebody then quotes it as a seam result."""
    return bool(float(controls.detach().abs().max()) <= atol) \
        if hasattr(controls, "detach") else \
        bool(abs(controls).max() <= atol)


def _cpu(x):
    return x.detach().to("cpu", copy=True)


def seam_dump_from_plan(plan: dict, *, eids, tier: str, arm: str,
                        gt=None, dt: float = 0.1,
                        op_band_s=(0.0, 2.0), tac_band_s=(2.0, 6.0),
                        allow_degenerate: bool = False) -> dict:
    """`V6Stack.emit`'s output -> the `seam_probe` dump dict (CPU tensors).

    ``plan`` is ``out["plan"]`` from the forward (or ``stack.emit(...)``'s
    return): ``controls``/``waypoints`` [B, N, T, 2], optional ``sel_score``
    [B, N]. ``eids`` is one EPISODE id per row — the probe's CI resamples by
    episode because windows inside one episode are strongly dependent, so a
    per-WINDOW counter here would silently understate every interval.

    ``gt`` (optional) is the [B, T, 2] ego-frame future used by the band-error
    block; omit it and that block is simply absent rather than fabricated.

    ⛔ REFUSES rather than banks: a missing control/waypoint pair, a fan with
    no selector, a length mismatch between ``eids`` and the batch, and (unless
    ``allow_degenerate``) an all-zero plan."""
    import torch                                              # noqa: PLC0415

    for k in ("controls", "waypoints"):
        if k not in plan:
            raise SeamDumpError(
                f"plan has no {k!r} — `seam.control_channels` needs BOTH "
                f"controls and waypoints because they fail differently (a "
                f"control-space stitch keeps position continuous). Bank the "
                f"pair or bank nothing.")
    controls, waypoints = plan["controls"], plan["waypoints"]
    if controls.ndim not in (3, 4) or controls.shape[-1] != 2:
        raise SeamDumpError(f"controls must be [N, T, 2] or [N, C, T, 2], got "
                            f"{tuple(controls.shape)}")
    if waypoints.shape[:-2] != controls.shape[:-2]:
        raise SeamDumpError(
            f"controls {tuple(controls.shape)} and waypoints "
            f"{tuple(waypoints.shape)} disagree on the batch/fan axes — one "
            f"of them is not the plan the other belongs to.")
    n = controls.shape[0]
    eid = torch.as_tensor(list(eids)).reshape(-1)
    if eid.numel() != n:
        raise SeamDumpError(
            f"{eid.numel()} eids for {n} rows. The eid is the CI's RESAMPLING "
            f"UNIT (episode-cluster bootstrap); a wrong-length or per-window "
            f"eid does not fail loudly, it silently narrows every interval.")
    is_fan = controls.ndim == 4
    sel = plan.get("sel_score")
    if sel is not None:
        sel = _cpu(sel).argmax(-1).reshape(-1).long()
    elif is_fan:
        # ⛔ NOT a silent zeros() fallback. With a fan and no selector the
        # probe would score candidate 0 — a trajectory the planner never
        # proposed — and F-16 asks specifically about the EMITTED WINNER.
        raise SeamDumpError(
            f"the plan is a fan [N, {controls.shape[1]}, T, 2] but carries no "
            f"'sel_score', so there is no emitted winner to probe. Run "
            f"seam_probe with --candidate all if scoring every candidate is "
            f"really what is wanted, but do not bank a guessed winner.")
    else:
        sel = torch.zeros(n, dtype=torch.long)
    if not allow_degenerate and plan_is_degenerate(controls):
        raise SeamDumpError(
            "every control in this plan is EXACTLY zero — this is the S-W "
            "zero-init emission head, and the probe will (correctly) return "
            "DEGENERATE. Bank an S-T-or-later checkpoint, or pass "
            "allow_degenerate=True if the degenerate dump is itself the "
            "artifact being kept.")
    t_steps = int(controls.shape[-2])
    out = {
        "eid": eid.long(),
        "controls": _cpu(controls),
        "waypoints": _cpu(waypoints),
        "sel": sel,
        "tier": str(tier),
        "arm": str(arm),
        "plan_steps": t_steps,
        "dt": float(dt),
        "op_band_s": [float(op_band_s[0]), float(op_band_s[1])],
        "tac_band_s": [float(tac_band_s[0]), float(tac_band_s[1])],
        "_evidence_class": "MEASURED (ours; the emitted plan of this arm)",
        "_read": "X2 seam dump. Score it with taniteval/tools/seam_probe.py "
                 "--dump <this file>; this file carries NO verdict.",
    }
    if gt is not None:
        gt_t = _cpu(gt)
        if gt_t.shape[0] != n or gt_t.shape[-2] != t_steps:
            raise SeamDumpError(
                f"gt {tuple(gt_t.shape)} does not match the plan "
                f"[{n}, ..., {t_steps}, 2] — a misaligned GT would produce a "
                f"band-error block that looks fine and is measuring the wrong "
                f"rows.")
        out["gt"] = gt_t
    return out


def save_seam_dump(dump: dict, path) -> str:
    """`torch.save` the dump, after re-checking the REQUIRED keys are present.

    The re-check is not belt-and-braces: callers assemble dumps by hand in
    scripts, and a dump missing `tier` is refused by `seam_probe` only AFTER
    the expensive part has already run (the `t1_eval` analysis-time-refusal
    trap). Failing here costs milliseconds."""
    import torch                                              # noqa: PLC0415
    from pathlib import Path                                  # noqa: PLC0415

    missing = [k for k in SEAM_DUMP_KEYS if k not in dump]
    if missing:
        raise SeamDumpError(f"seam dump is missing {missing} — refusing to "
                            f"write an artifact seam_probe would reject.")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dump, p)
    return str(p)
