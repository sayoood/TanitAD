"""PREREG_V8F_T1KIN_CAUSALBUCKET: the ``--limit causal`` / ``causal_all`` bucket of ``v8f_t1_eval`` reads the past only, follows the sidecar's
containing-window rule, and leaves ``--limit logged`` / ``off`` untouched (C-L1, C-L2 and the rule; C-L3 and reachability run on real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import v8f_t1_eval as T1  # noqa: E402
from tanitad.refs.refcv6_max_speed import SPEED_MAX_STEPS_MS_V6  # noqa: E402


def _bin(oh):
    return int(np.argmax(oh))


def test_modes_are_declared():
    assert T1.LIMIT_MODES == ("logged", "off", "causal", "causal_all")
    assert T1.CAUSAL_STEPS == 20                                    # 4 s of 0.2 s rows, fixed by the PREREG


@pytest.mark.parametrize("steps", [20, None])
@pytest.mark.parametrize("poison", [np.nan, 99.0, 1e6])
def test_c_l2_rows_after_j_are_never_read(steps, poison):
    """C-L2: whatever sits after row j, the bucket at j is the same; a leak would turn 99 m/s into the 120 bucket."""
    rng = np.random.default_rng(0)
    v = rng.uniform(0.0, 7.0, 200)                                  # all inside the 30 km/h window
    for j in (0, 5, 20, 21, 57, 150):
        base, ok = T1.causal_bucket(v.copy(), j, steps)
        w = v.copy()
        w[j + 1:] = poison
        got, ok2 = T1.causal_bucket(w, j, steps)
        assert ok and ok2 and _bin(base) == 0 and (got == base).all()


def test_c_l2_the_next_row_alone_cannot_move_it():
    """The deliberate-leak mutant (reading row j + 1) must fail exactly here."""
    v = np.full(60, 5.0)
    v[31] = 40.0
    oh, ok = T1.causal_bucket(v, 30, 20)
    assert ok and _bin(oh) == 0
    oh, ok = T1.causal_bucket(v, 31, 20)                            # at j = 31 the row is the past and must count
    assert ok and _bin(oh) == 3


def test_the_window_is_the_last_4_s_and_causal_all_is_the_whole_past():
    v = np.full(100, 5.0)
    v[10] = 20.0                                                    # 72 km/h -> the 100 window
    assert _bin(T1.causal_bucket(v, 30, 20)[0]) == 2                # row 10 = j - 20: inside [j - 20, j]
    assert _bin(T1.causal_bucket(v, 31, 20)[0]) == 0                # row 10 = j - 21: outside
    assert _bin(T1.causal_bucket(v, 99, None)[0]) == 2              # causal_all keeps it
    assert _bin(T1.causal_bucket(v, 5, 20)[0]) == 0                 # the slice is clipped at the clip start, never wraps


def test_the_containing_window_rule_is_the_sidecars():
    for i, s in enumerate(SPEED_MAX_STEPS_MS_V6):                   # exactly on a step -> that step; just above -> the next
        assert _bin(T1.causal_bucket(np.array([s]), 0, 20)[0]) == i
        assert _bin(T1.causal_bucket(np.array([s + 1e-3]), 0, 20)[0]) == min(i + 1, 3)
    assert _bin(T1.causal_bucket(np.array([0.0]), 0, 20)[0]) == 0   # standstill -> 30 km/h
    assert _bin(T1.causal_bucket(np.array([-0.2]), 0, 20)[0]) == 0  # a tiny negative speed estimate is a standstill, not a refusal
    oh, ok = T1.causal_bucket(np.array([300.0]), 0, 20)
    assert ok and _bin(oh) == 3 and oh.sum() == 1.0 and oh.dtype == np.float32


def test_no_finite_past_is_the_off_format():
    oh, ok = T1.causal_bucket(np.full(30, np.nan), 25, 20)
    assert ok is False and (oh == 0).all() and oh.shape == (4,)
    oh, ok = T1.causal_bucket(np.array([np.nan, np.nan, 3.0]), 1, 20)
    assert ok is False


def test_build_item_counts_the_models_ticks_only():
    """The stats seam: build_item adds to ``stats`` only when handed one (rollout hands it to the model planner, not the floor)."""
    class _DS:
        hmax = 3

        def item(self, ci, j, a_t, a_s):
            return {"limit": np.array([0, 1, 0, 0], np.float32), "limit_ok": True, "feats": None, "frame_ok": None}

    class _P:
        d = None

    class _St:
        ci, j, T = 0, 10, 50
        d = {"v": np.r_[np.full(11, 4.0), np.full(39, np.nan)]}
        nav = np.zeros((50, 5), np.float32)

    stats = {}
    it = T1.build_item(_DS(), _P(), _St(), 0, 0, 0, "logged", "causal", stats)
    assert _bin(it["limit"]) == 0 and it["limit_ok"] is True
    assert stats == {"ticks": 1, "confusion_logged_x_causal": [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}
    it = T1.build_item(_DS(), _P(), _St(), 0, 0, 0, "logged", "logged", stats)
    assert _bin(it["limit"]) == 1 and stats["ticks"] == 1           # logged: the item's own bucket, nothing counted
    it = T1.build_item(_DS(), _P(), _St(), 0, 0, 0, "logged", "causal", None)
    assert stats["ticks"] == 1                                      # no stats dict: nothing counted
