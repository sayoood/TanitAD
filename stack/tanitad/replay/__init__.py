"""Open-loop replay, regression-test & visualization harness.

Replays identical cached episodes through N architecture arms (main world
model, REF-A frozen-DINO, REF-B E2E) and emits per-timestep records that feed
(a) the aggregation/regression machinery in :mod:`tanitad.replay.stats` and
(b) the rerun visualization schema in :mod:`tanitad.replay.rr_log`.

Entry point: ``stack/scripts/replay_app.py``. Design notes, rerun verdict and
roadmap: ``tanitad/replay/README.md``.
"""

from tanitad.replay.engine import (DT, WAYPOINT_STEPS, ArmOutput,
                                   ReplayEngine, ReplayEpisode,
                                   TimestepRecord, WindowBatch, WindowRef,
                                   load_corpora, split_fit_replay)

__all__ = [
    "DT", "WAYPOINT_STEPS", "ArmOutput", "ReplayEngine", "ReplayEpisode",
    "TimestepRecord", "WindowBatch", "WindowRef", "load_corpora",
    "split_fit_replay",
]
