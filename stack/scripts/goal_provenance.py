"""Where did the evaluated model's GOAL come from? — one disclosure, three callers.

WHY THIS FILE EXISTS
--------------------
``V4_FLAGSHIP_DESIGN.md:558-560`` states the rule::

    No leaderboard number may come from a GT-derived plan or a GT-derived goal.

Three evaluators violate it today, and none of them says so in its own output:

* ``eval_flagship_v15.py`` — ``route``/``route_graded``/``vt_band``/``vt_speed``
  read straight off the GT label file.
* ``eval_flagship_v16.py`` — identical feed. **v1.6's published 0.4375 is one of
  these numbers.**
* ``eval_flagship_v4.py`` MODE B — ``_goal_inputs(head.cfg, b, v0)`` reads the
  same fields off a ``FlagshipV4Dataset`` batch, which mints them **per window
  from the episode's full future poses**.

Every one of those quantities is derived from **the ego's own future**. A model
told where it is going to end up is not solving the same problem as a deployed
model, so the number is an **upper bound**, not a deployable result.

WHAT THIS MODULE DOES — AND DELIBERATELY DOES NOT DO
----------------------------------------------------
It **discloses in place**. It does not change any number, any input, or any
code path. The published values stay exactly as they are so the registry owner
can correct the record from a stable base rather than chasing a moving one
(PI direction, 2026-07-26). :func:`disclose` returns a block to embed in the
result JSON and prints a banner that a human running the script cannot miss.

Removing the oracle is a separate, larger change: it needs a *produced* goal
path (a strategic head that infers the route from vision), which is exactly what
PC1 is for. Until that lands, the honest form is to report **both** — oracle-goal
and produced-goal — with the estimator named, per ``HPP0_CONFOUND_AUDIT.md``
PC1 item #5.
"""
from __future__ import annotations

#: Goal-source tags. Only ``produced`` is deployable.
GOAL_SOURCES = ("oracle_gt_future", "produced_from_vision", "constant",
                "dropped")

_ORACLE_FIELDS = {
    "eval_flagship_v15": ("route", "route_graded", "vt_band", "vt_speed"),
    "eval_flagship_v16": ("route", "route_graded", "vt_band", "vt_speed"),
    "eval_flagship_v4": ("route", "route_graded", "vt_band", "vt_speed",
                         "strat_scalars"),
}

#: Published numbers known to have been produced with a goal oracle. Kept HERE,
#: next to the mechanism, so the list cannot drift away from the code that
#: causes it. `MODEL_REGISTRY.md` is the authority for the values themselves;
#: this is the authority for *how they were fed*.
AFFECTED_PUBLISHED_NUMBERS = (
    {"arm": "flagship v1.6", "value_m": 0.4375, "metric": "ade@2s",
     "evaluator": "eval_flagship_v16.py:135-143",
     "note": "the headline v1.6 number"},
    {"arm": "flagship v1.5", "value_m": None, "metric": "ade@2s",
     "evaluator": "eval_flagship_v15.py:92-103",
     "note": "every v1.5 anchored-fan number"},
    {"arm": "flagship v4 MODE B", "value_m": None, "metric": "ade@2s",
     "evaluator": "eval_flagship_v4.py:322 (_goal_inputs)",
     "note": "every v4 MODE-B number, including the 30k gate's primary"},
)


def disclose(script: str, *, goal_source: str = "oracle_gt_future",
             fields=None, quiet: bool = False) -> dict:
    """Return the provenance block for a result JSON; print the banner.

    ``script`` is the evaluator's module name (e.g. ``"eval_flagship_v16"``).
    Never raises: a disclosure that can break an eval would be turned off."""
    assert goal_source in GOAL_SOURCES, goal_source
    fields = tuple(fields) if fields else _ORACLE_FIELDS.get(script, ())
    is_oracle = goal_source == "oracle_gt_future"
    block = {
        "goal_source": goal_source,
        "goal_fields": list(fields),
        "is_oracle": is_oracle,
        "deployable": not is_oracle,
        "rule": ("V4_FLAGSHIP_DESIGN.md:558-560 — no leaderboard number may "
                 "come from a GT-derived plan or a GT-derived goal"),
        "spec": ("TanitAD Research Hub/Architecture & Inference/Implementation/"
                 "incoming/2026-07-25-hpp0-confound-audit/"
                 "HPP0_CONFOUND_AUDIT.md §1.4, PC1 item #5"),
        "_read": (
            "the listed fields are minted from the EGO'S OWN FUTURE POSES. The "
            "model is told where it ends up, so this is an UPPER BOUND on a "
            "deployed number, not the deployed number. It is NOT comparable to "
            "an arm evaluated with a constant command (REF-C's historical "
            "`nav_cmd=None`) nor to one evaluated with a produced goal."
            if is_oracle else
            "the goal was produced by the model from its observations; no "
            "future-derived quantity entered the evaluated forward pass."),
        "affected_published_numbers": [dict(x) for x in
                                       AFFECTED_PUBLISHED_NUMBERS],
        "disclosure_only": ("this record changes NO number and NO code path. "
                            "Removing the oracle requires a produced-goal head "
                            "(PC1) and a re-run; the registry owner corrects "
                            "the published values, not this script."),
    }
    if is_oracle and not quiet:
        print("=" * 78, flush=True)
        print(f"[goal-oracle] {script}: THIS EVALUATION FEEDS A GOAL ORACLE.",
              flush=True)
        print(f"[goal-oracle]   fields from the ego's own future: "
              f"{', '.join(fields)}", flush=True)
        print("[goal-oracle]   -> the resulting ADE is an UPPER BOUND, not a "
              "deployable number,", flush=True)
        print("[goal-oracle]      and violates V4_FLAGSHIP_DESIGN.md:558-560 "
              "as a leaderboard value.", flush=True)
        print("[goal-oracle]   -> known affected published numbers: v1.5, "
              "v1.6 (0.4375), v4 MODE B.", flush=True)
        print("=" * 78, flush=True)
    return block
