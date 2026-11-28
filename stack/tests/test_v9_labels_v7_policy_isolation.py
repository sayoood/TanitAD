"""The v7_labels goal-negative policy belongs to the LOADED SPLIT, not to the module (WP-A fix, 2026-10-04).

refcv7 (D1 F2, refcv8 data audit): ``train()`` loaded the TRAIN blob, then the EVAL blob, then forked the DataLoader
workers. ``v7_labels._MEASURED_GEOMETRY_TOKENS`` was module state refilled by each load, so the workers supervised
the train windows with the EVAL blob's policy — ``LANE_CHANGE_L`` (CoT-backed in train, absent from the eval CoT set)
became a "geometry" token and was trained as a negative on 100 % of tactical windows.

These tests build two tiny blobs that reproduce exactly that asymmetry and assert:
  1. train-then-eval loading leaves the TRAIN labels' targets and policy unchanged;
  2. a pickle round-trip (what a spawned DataLoader worker receives) carries the train policy even when the parent
     loaded the eval blob in between;
  3. MUTATION: re-routing the policy lookup through the module global (the historical defect) makes test 1 go RED.
"""
from __future__ import annotations

import gzip
import json
import pickle

import pytest

from tanitad.data import v7_labels as v7l


def _rec(cid, goals):
    return {"schema_version": v7l.EXPECTED_SCHEMA, "vocab": v7l.EXPECTED_VOCAB, "clip_id": cid,
            "a_tac": {"lat": "LANE_KEEP", "lon": "CRUISE"}, "a_str": {"token": "HOLD_MAIN_ROAD"},
            "g_str": {"token": "FOLLOW_ROUTE"},
            "g_tac": {"anchor": {"goal_x_m": 50.0, "goal_y_m": 0.0, "t_reach_s": 6.0}, "goals": goals},
            "bands": {"operative_s": [0.0, 2.0], "tactical_s": [2.0, 6.0], "strategic_s": [8.0, 30.0]},
            "t0_s": 8.0, "horizon": {}, "nav_command": {"token": "NAV_FOLLOW_ROAD", "provenance": "ego-future"}}


def _blob(path, recs):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return str(path)


@pytest.fixture()
def blobs(tmp_path):
    geo = {"FOLLOW_LANE": {"provenance": "geometry"}, "SPEED_BAND": {"provenance": "geometry"}}
    # TRAIN: one clip's caption asserts LANE_CHANGE_L -> the token is vlm-cot-backed -> its absence is IGNORED
    train = _blob(tmp_path / "train.jsonl.gz", [
        _rec("clip-train-a", {**geo, "LANE_CHANGE_L": {"provenance": "vlm-cot"}}),
        _rec("clip-train-b", dict(geo))])
    # EVAL: no clip mentions LANE_CHANGE_L -> by the set arithmetic it becomes a GEOMETRY token (absence = negative)
    evl = _blob(tmp_path / "eval.jsonl.gz", [_rec("clip-eval-a", dict(geo))])
    return train, evl


def _lc_weight(label):
    _, w = v7l.tactical_goal_targets(label, label.t0_s, negatives="measured")
    return w[v7l.TAC_GOAL_TOKENS.index("LANE_CHANGE_L")]


def test_the_two_blobs_really_disagree_on_lane_change_l(blobs):
    """Control: the fixture must reproduce the D1 F2 asymmetry, or the isolation test below proves nothing."""
    train, evl = blobs
    tr, mtr = v7l.load_v7_labels(train)
    ev, mev = v7l.load_v7_labels(evl)
    assert "LANE_CHANGE_L" not in mtr.goal_geometry_tokens
    assert "LANE_CHANGE_L" in mev.goal_geometry_tokens


def test_train_policy_survives_the_eval_load(blobs):
    train, evl = blobs
    tr, mtr = v7l.load_v7_labels(train)
    b = next(x for x in tr if x.clip_id == "clip-train-b")       # no LANE_CHANGE_L positive on this clip
    before = _lc_weight(b)
    assert before == v7l.IGNORE_W                                  # train policy: caption silence is IGNORE
    v7l.load_v7_labels(evl)                                        # what train() did before forking the workers
    assert _lc_weight(b) == before                                 # ⛔ was a weight-1 NEGATIVE before the fix
    assert b.goal_geometry_tokens == frozenset(mtr.goal_geometry_tokens)
    assert "goal_geometry_tokens" in mtr.to_dict()                 # the config states the policy actually used


def test_a_pickled_worker_copy_keeps_the_train_policy(blobs):
    train, evl = blobs
    tr, _ = v7l.load_v7_labels(train)
    em = v7l.TacGoalEmitter(tr, {0: "clip-train-b"}, negatives="measured")
    payload = pickle.dumps((tr, em))                               # the dataset as a spawned worker receives it
    v7l.load_v7_labels(evl)                                        # the parent loads eval in between
    tr2, em2 = pickle.loads(payload)
    b2 = next(x for x in tr2 if x.clip_id == "clip-train-b")
    assert _lc_weight(b2) == v7l.IGNORE_W
    y, w = em2([0], [8.0])
    assert float(w[0, v7l.TAC_GOAL_TOKENS.index("LANE_CHANGE_L")]) == v7l.IGNORE_W
    assert "LANE_CHANGE_L" not in em2.provenance()["supervised_negative_tokens"]


def test_census_and_cot_backed_tokens_read_the_split_not_the_last_load(blobs):
    train, evl = blobs
    tr, _ = v7l.load_v7_labels(train)
    v7l.load_v7_labels(evl)
    assert "LANE_CHANGE_L" in v7l.cot_backed_tokens(tr)            # the train split's own measurement
    assert v7l.goal_supervision_census(tr)["LANE_CHANGE_L"]["supervised_negative"] is False


def test_mixing_two_policies_is_refused(blobs):
    train, evl = blobs
    tr, _ = v7l.load_v7_labels(train)
    ev, _ = v7l.load_v7_labels(evl)
    with pytest.raises(ValueError):
        v7l.goal_supervision_census(list(tr) + list(ev))


def test_MUTATION_reading_the_module_global_goes_red(blobs, monkeypatch):
    """Reintroduce the historical defect (policy read from the last-load module global): the survival check must
    FAIL. A guard that cannot fail is not a guard."""
    train, evl = blobs
    monkeypatch.setattr(v7l, "_geometry_tokens_for", lambda label: v7l._MEASURED_GEOMETRY_TOKENS)
    tr, _ = v7l.load_v7_labels(train)
    b = next(x for x in tr if x.clip_id == "clip-train-b")
    before = _lc_weight(b)
    v7l.load_v7_labels(evl)
    assert _lc_weight(b) != before                                 # RED: the eval load moved the train targets
    assert _lc_weight(b) == 1.0                                    # ... into exactly the D1 F2 negative
