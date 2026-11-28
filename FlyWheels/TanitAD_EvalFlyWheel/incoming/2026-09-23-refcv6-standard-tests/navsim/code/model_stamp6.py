"""The model-as-trained stamp every refcv6 number carries — STEP-AWARE (Master Mind, 2026-09-26).

PI ruling on the A16 defects, 2026-09-26 13:30 Berlin: "Stop now, resume with fixes". refcv6-r101-s0
stopped at the step-34,500 checkpoint (ckpt.pt md5 3fbbde7470914b8bd55176e60fe399d3) and resumed at
34,500 on commit 82c2331; from there the F3 per-stage cascade loss trains (first row: step 34,550) and
tactical labels are read on the true clip clock. Sources: GOALS_AND_CLAIMS D-REFCV6-F3-WHITELIST and
D-REFCV6-LABEL-CLOCK (landed 9d16c441); evidence class INHERITED (the Master Mind's audit, not
re-verified by this stream).

⛔ A pre-switch vs post-switch comparison mixes training time with the fix: never attribute its
difference to the fix alone (``switch_warning``).
"""
from __future__ import annotations

SWITCH_STEP = 34500
PRE = ('"F3 detach-only, F4 on the last layer only; tactical labels ~0.37 s early" (steps <= 34,500: '
       'the F3 per-stage cascade loss never ran, decoder stages 0-2 frozen at init; Master Mind audit '
       '2026-09-26, GOALS_AND_CLAIMS D-REFCV6-F3-WHITELIST / D-REFCV6-LABEL-CLOCK; INHERITED)')
POST = ('"hybrid: F3 cascade loss + true label clock from step 34,500" (the run stopped at 34,500 and '
        'resumed on commit 82c2331 with both fixes; steps <= 34,500 trained without them; PI ruling '
        '2026-09-26; INHERITED)')


def stamp(step) -> str:
    try:
        s = int(step)
    except (TypeError, ValueError):
        return "UNKNOWN step — cannot place it against the 34,500 fix switch"
    return PRE if s <= SWITCH_STEP else POST


def switch_warning(step_a, step_b):
    """None unless the two steps straddle the fix switch."""
    try:
        a, b = int(step_a), int(step_b)
    except (TypeError, ValueError):
        return "UNKNOWN step(s): cannot tell whether this comparison straddles the 34,500 fix switch"
    if (a <= SWITCH_STEP) != (b <= SWITCH_STEP):
        return ("PRE- vs POST-SWITCH: this difference mixes further training with the step-34,500 fixes "
                "(F3 cascade loss, true label clock) — never attribute it to the fixes alone")
    return None
