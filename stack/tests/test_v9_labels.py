"""v9 reader contract: lookups by (sid, k) / dataset window / trainer clock, NO mutable module state (the D1 F2
defect), pickle round-trip as a DataLoader worker would see it, decoding of goals / partial labels / inputs.

A tiny synthetic release in the builder's on-disk format is written to tmp_path; no corpus data is needed.
"""
from __future__ import annotations

import hashlib
import json
import pickle

import numpy as np
import pytest

from tanitad.data import v9_labels as V


def _write_release(tmp_path, split, sid, n=10, g0=0.113, dt=0.1007, token=1, goal_y=0b110, goal_w=0b111):
    k = np.arange(n) + 2
    rows = {
        "k": k.astype(np.int16), "now_s": g0 + k * dt,
        "lat_cls_a": np.full(n, 1, np.int16), "lat_cls_b": np.full(n, 1, np.int16),
        "lat_allowed_a": np.full(n, 0b10, np.uint8), "lat_allowed_b": np.full(n, 0b10, np.uint8),
        "lat_v7id_a": np.full(n, 6, np.int16), "lat_v7id_b": np.full(n, 6, np.int16),
        "lat_allowed_v7_a": np.full(n, 1 << 6, np.uint8), "lat_allowed_v7_b": np.full(n, 1 << 6, np.uint8),
        "lon_cls": np.full(n, -100, np.int16), "lon_allowed": np.full(n, (1 << 6) | (1 << 3), np.uint8),
        "lon_v7id": np.full(n, -100, np.int16), "lon_allowed_v7": np.full(n, (1 << 1) | (1 << 0), np.uint8),
        "goal_y": np.full(n, goal_y, np.uint32), "goal_w": np.full(n, goal_w, np.uint32),
        "nav_token": np.full(n, float(token)), "nav_side_next": np.full(n, 1.0), "nav_d_next_m": np.full(n, 42.0),
        "nav_d_end_m": np.full(n, 70.0), "nav_dyaw_next_deg": np.full(n, 88.0), "nav_args_valid": np.full(n, 1.0),
        "nav_lookahead_m": np.full(n, 900.0), "nav_t_next_s": np.full(n, 3.0), "nav_token_ttime": np.full(n, 1.0),
        "rc_A80_x": np.full(n, 70.0), "rc_A80_y": np.full(n, 20.0), "rc_A80_psi": np.full(n, 45.0),
        "rc_A80_valid": np.full(n, 1.0)}
    for v in ("A30", "A50", "B"):
        for c in ("x", "y", "psi"):
            rows[f"rc_{v}_{c}"] = np.full(n, np.nan)
        rows[f"rc_{v}_valid"] = np.zeros(n)
    clips = {"sid": np.array([sid], np.int64), "sha12": np.array(["abcdef012345"]), "row0": np.array([0]),
             "n_rows": np.array([n]), "k0": np.array([2])}
    p = tmp_path / f"v9_labels_{split}.npz"
    np.savez_compressed(p, **{"row__" + k_: v for k_, v in rows.items()}, **{"clip__" + k_: v for k_, v in clips.items()})
    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    (tmp_path / f"v9_labels_{split}.manifest.json").write_text(json.dumps({"schema": V.SCHEMA, "split": split, "npz_md5": md5}))
    return str(p), md5


def _module_state():
    """Every non-callable, non-module global of the reader, frozen to a comparable form."""
    out = {}
    for k, v in vars(V).items():
        if k.startswith("__") or callable(v) or type(v).__name__ == "module":
            continue
        out[k] = repr(v)
    return out


def test_lookup_by_frame_window_and_clock(tmp_path):
    p, _ = _write_release(tmp_path, "train", sid=12345)
    rel = V.load_v9_release(p)
    assert V.row_for_window(rel, 12345, 0) == V.row_index(rel, 12345, 9)        # NOW of window t = 0 is k = 9
    r = V.row_index(rel, 12345, 5)
    assert V.row_for_now(rel, 12345, 5, 0.113 + 5 * 0.1007) == r
    with pytest.raises(V.V9LabelError):
        V.row_for_now(rel, 12345, 5, 0.113 + 5 * 0.1007 + 0.05)                  # trainer clock disagrees: refuse
    with pytest.raises(V.V9LabelError):
        V.row_index(rel, 99, 5)                                                  # unknown clip
    with pytest.raises(V.V9LabelError):
        V.row_index(rel, 12345, 50)                                              # outside the clip


def test_loading_eval_after_train_leaves_train_and_the_module_unchanged(tmp_path):
    """THE D1 F2 GUARD: the eval load must not change what the train object (or any module global) says."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    pa, _ = _write_release(tmp_path / "a", "train", sid=1, goal_w=0b111)
    pb, _ = _write_release(tmp_path / "b", "eval139", sid=2, goal_w=0b001)
    before_mod = _module_state()
    tr = V.load_v9_release(pa)
    snap = {k: v.copy() for k, v in tr.rows.items()}
    t_before = V.window_targets(tr, 0)
    ev = V.load_v9_release(pb)
    assert ev.split == "eval139" and tr.split == "train"
    for k, v in tr.rows.items():
        assert np.array_equal(v, snap[k], equal_nan=v.dtype.kind == "f")
    t_after = V.window_targets(tr, 0)
    assert np.array_equal(t_before["goal_w"], t_after["goal_w"])               # the train policy is untouched
    assert _module_state() == before_mod                                       # nothing global moved


def test_the_module_state_guard_goes_red_on_a_reintroduced_global(tmp_path, monkeypatch):
    """Mutation arm: a loader that stores per-load state in a module global must be caught by _module_state()."""
    pa, _ = _write_release(tmp_path, "train", sid=1)
    before = _module_state()
    real = V.load_v9_release

    def leaky(path, *a, **kw):
        rel = real(path, *a, **kw)
        V._LAST_SPLIT = rel.split                                                # the v7_labels defect, reintroduced
        return rel
    monkeypatch.setattr(V, "load_v9_release", leaky)
    V.load_v9_release(pa)
    assert _module_state() != before                                           # RED, as required
    monkeypatch.delattr(V, "_LAST_SPLIT")


def test_pickle_round_trip_like_a_dataloader_worker(tmp_path):
    p, _ = _write_release(tmp_path, "train", sid=7)
    rel = V.load_v9_release(p)
    back = pickle.loads(pickle.dumps(rel))
    assert back.md5 == rel.md5 and back.split == rel.split and dict(back.clips) == dict(rel.clips)
    for k, v in rel.rows.items():
        assert np.array_equal(back.rows[k], v, equal_nan=v.dtype.kind == "f")
        assert not back.rows[k].flags.writeable                                 # still read-only in the worker
    assert V.window_targets(back, 3)["lat"] == V.window_targets(rel, 3)["lat"]


def test_md5_and_schema_refusals(tmp_path):
    p, md5 = _write_release(tmp_path, "train", sid=7)
    with pytest.raises(V.V9LabelError):
        V.load_v9_release(p, expect_md5="0" * 32)
    assert V.load_v9_release(p, expect_md5=md5).md5 == md5


def test_decoding_goals_partial_labels_and_inputs(tmp_path):
    p, _ = _write_release(tmp_path, "train", sid=7, goal_y=0b110, goal_w=0b111)
    rel = V.load_v9_release(p)
    T = V.window_targets(rel, 0, ids="v7")
    assert T["lat"] == V.LAT7.index("TURN_L") and T["lat_allowed"][V.LAT7.index("TURN_L")]
    assert T["lon"] == -100 and T["lon_allowed"].sum() == 2                    # partial label {CRUISE, FOLLOW}
    assert list(T["goal_y"][:3]) == [0.0, 1.0, 1.0] and list(T["goal_w"][:3]) == [1.0, 1.0, 1.0]
    T9 = V.window_targets(rel, 0, ids="v9")
    assert V.LAT9[T9["lat"]] == "TURN_L"
    I = V.window_inputs(rel, 0, rc_variant="A80")
    assert I["nav_token"] == 1 and I["nav_args_valid"] == 1 and I["rc_valid"] == 1
    assert "nav_t_next_s" not in I and "nav_token_ttime" not in I             # future-speed fields never inputs
    J = V.window_inputs(rel, 0, rc_variant="A30")
    assert J["rc_valid"] == 0 and np.all(J["rc"] == 0)                         # invalid RC -> zeros + mask, never NaN
    with pytest.raises(ValueError):
        V.window_inputs(rel, 0, rc_variant="A40")
