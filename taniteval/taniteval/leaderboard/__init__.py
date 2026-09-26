"""TanitEval leaderboard generator (EvalFlyWheel W4).

``python -m taniteval.leaderboard build`` (run from ``<repo>/taniteval``, like ``taniteval.runner``)
regenerates the marker-delimited section of ``Benchmarks & Eval/LEADERBOARD.md``
(``<!-- BENCH:BEGIN -->`` … ``<!-- BENCH:END -->``) and ``Benchmarks & Eval/leaderboard.html`` from
files on disk — nothing else on the page is touched.

Inputs (all read at build time, never copied into code):
  * ``products/P7-TanitEval/benchmarks/published_results.json`` — external numbers, each re-read from a
    banked primary (library key + table + page);
  * ``taniteval/results/bench/<benchmark>/<split>/<run_id>/summary.json`` — our runs (W1 contract);
    until W1 lands, the legacy adapters in ``leaderboard_sources.json`` read E1/E2's banked outputs;
  * registry-anchored raw eval JSON, addressed by JSON pointer (``leaderboard_sources.json``);
  * ``Project Steering/MODEL_REGISTRY.md`` — every internal row's anchor must be present;
  * the E4 currency audit JSON.

Binding rules the generator ENFORCES (it refuses to render, it does not just document):
  * one protocol per table — a row whose protocol tag differs from its table's raises;
  * a NavSim run row needs its STOP and CV floors, or it is refused;
  * a warmup row never carries an interval (``EPDMS_v2_warmup_two_stage`` → UNAVAILABLE);
  * a published number sits in a comparison cell only if ``comparison_admissible`` and its
    #151 side is known; pre-#151 values render in their own table, never beside post-fix ones;
  * every internal row carries tier + loop + n + the paired estimator + its floors.

Determinism: sorted inputs, fixed float formatting, no wall-clock; the section ends with a digest of
the canonical data model, so an unchanged input set rebuilds byte-identically.
"""
from .build import build, render_section  # noqa: F401
