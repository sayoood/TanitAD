"""The model-as-trained stamp every refcv7 NavSim number carries (same interface as the refcv6
suite's ``model_stamp6``: ``stamp(step)``, ``switch_warning(a, b)``).

refcv7-r101-s0 was launched 2026-09-28 ~00:03 Berlin (``gate_exec_1.json`` / ``gate_verify_1.json`` in
the run dir) with FIX-1..5, NEW-1 (``--residual-prior ha0_ext_pose``), NEW-2 at 10 cm under A6/A7
(``--map-hires on``, ``--bev-source map_hires_pool``, 100 m x +-30 m), the A9 box head (300 queries,
``learned_ref``), the three selection mechanisms ON (A2) and the strategic layer OFF -- the run's
``config.json`` md5 ``e6512a01b9c70f0e4a4dac581621984a`` as pulled 2026-09-28 05:11 Berlin. No mid-run
recipe change is recorded at the time of writing (INHERITED: the run record, not re-verified beyond
the config md5). ⛔ The runner re-reads Thor's ``config.json`` md5 at every milestone; a DIFFERENT md5
is a recipe change and every comparison straddling it is flagged (``SWITCHES`` then gains the step).
"""
from __future__ import annotations

CONFIG_MD5_AT_LAUNCH = "e6512a01b9c70f0e4a4dac581621984a"
#: steps at which the recipe changed mid-run (none recorded)
SWITCHES: tuple = ()
AS_LAUNCHED = ('"refcv7-r101-s0 as launched" (FIX-1..5; NEW-1 ha0_ext_pose; NEW-2 10 cm map, A6/A7 '
               'pooled BEV; A9 box head 300 queries learned_ref; A2 selection mechanisms ON; strategic '
               'OFF; config.json md5 e6512a01...; no mid-run recipe change recorded -- INHERITED)')


def stamp(step) -> str:
    try:
        int(step)
    except (TypeError, ValueError):
        return "UNKNOWN step -- cannot place it against the run's recipe record"
    return AS_LAUNCHED


def switch_warning(step_a, step_b):
    """None unless the two steps straddle a recorded recipe change."""
    try:
        a, b = int(step_a), int(step_b)
    except (TypeError, ValueError):
        return "UNKNOWN step(s): cannot tell whether this comparison straddles a recipe change"
    for s in SWITCHES:
        if (a <= s) != (b <= s):
            return f"straddles the recipe change at step {s}: never attribute the difference to it alone"
    return None
