"""refcv8 WP-B X4 (D1 F2) -- the TRAINER-SIDE pin of WP-A's ``v7_labels`` fix (INTEGRATION.md sec. 6; MM ruling
2026-10-04: "your trainer-side test pins that the trainer reads it that way").

The defect (D1): ``v7_labels`` kept the goal-negative policy in MODULE globals; the trainer loads the TRAIN blob, then
the EVAL blob, then forks the DataLoader workers -- so every train window was computed under the EVAL policy and
``LANE_CHANGE_L`` was a weight-1 NEGATIVE on 100 % of in-band train windows. WP-A's fix (base 86f0c46e -> abb1f64c)
makes the policy TRAVEL on the label objects (``V7Label.goal_geometry_tokens``, ``LabelManifest.goal_geometry_tokens``).

Pinned here, on the REAL v8 blobs (the ones refcv7 trained on), in the trainer's own load order:
1. the two splits' manifests differ by EXACTLY ``LANE_CHANGE_L`` (the precondition, measured, not assumed);
2. a train emitter's policy == the TRAIN manifest's, after the EVAL load;
3. the trainer installs NO scope when the policy travels (``r8train.v7_scope_for``);
4. the per-window ``(y, w)`` of a train clip, in process AND in a pickled ``V3Dataset`` (= a DataLoader worker), is
   the TRAIN policy's: ``LANE_CHANGE_L`` is CoT-backed in TRAIN -> weight 0, not a negative;
5. DELIBERATE REGRESSION: re-introduce the module-global read -> the same check goes RED;
6. the fallback ``V7PolicyScope`` (for an unfixed module) still isolates a module-state policy.
"""
from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

TRAIN = Path("D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz")      # md5 b45377a1... (refcv7's train blob)
EVAL = Path("D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz")   # md5 eefc38d1...
_REAL = pytest.mark.skipif(not (TRAIN.exists() and EVAL.exists()),
                           reason="the real v8 label blobs are not on this machine")


@pytest.fixture(scope="module")
def setup():
    import _refcv8_rig as R
    T = R.trainer()
    v7l = T.v7l
    from tanitad.train import refcv8_train as RT
    tr, man_tr = v7l.load_v7_labels(str(TRAIN), allow_oracle_nav=True)
    scope_tr = RT.v7_scope_for(v7l)                    # what train() installs right after the TRAIN load
    ev, man_ev = v7l.load_v7_labels(str(EVAL), allow_oracle_nav=True)
    assert man_tr.md5 == "b45377a1f25263b5c0f3d318c126b1ac" and man_ev.md5 == "eefc38d1453bd1c73802d44d45affced"
    lab = next(l for l in tr if "LANE_CHANGE_L" not in (l.tac_goal_meta or {}))
    cfg = T.v3.refc_v3_smoke_config(True)
    eps = T._synth_episodes(1, cfg.core, seed=0, min_frames=140)
    eps[0].episode_id = 424242                       # an integer stable id, as the real cache carries
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    ds.v7_by_sid = {int(eps[0].episode_id): lab}
    ds._v7_scope = scope_tr
    ds.legacy_label_clock = True                       # tests only: NOW = (t + w - 1) * 0.1 s
    ds.v7_dt = 0.1
    ds.tac_goal_targets = True
    w = int(cfg.core.window)
    i_band = next(i for i, (e, t) in enumerate(ds.index)
                  if abs((t + w - 1) * 0.1 - float(lab.t0_s)) <= 0.5)
    return T, v7l, RT, ds, (tr, man_tr, ev, man_ev), i_band


def _lc_weight(T, ds, i) -> float:
    k = list(T.v7l.TAC_GOAL_TOKENS).index("LANE_CHANGE_L")
    return float(ds[i]["tac_goal_w"][k])


def _train_policy_holds(T, ds, i) -> bool:
    """The check: LANE_CHANGE_L is CoT-backed in TRAIN, so the TRAIN policy gives it NO evidence (w = 0); the EVAL
    policy (loaded last) would make it a weight-1 negative. In process AND in a pickled worker."""
    worker = pickle.loads(pickle.dumps(ds))
    return _lc_weight(T, ds, i) == 0.0 and _lc_weight(T, worker, i) == 0.0


@_REAL
def test_the_two_splits_differ_by_exactly_LANE_CHANGE_L_and_the_policy_travels(setup) -> None:
    T, v7l, RT, ds, (tr, man_tr, ev, man_ev), i = setup
    assert sorted(set(man_tr.goal_geometry_tokens) ^ set(man_ev.goal_geometry_tokens)) == ["LANE_CHANGE_L"]
    assert {l.goal_geometry_tokens for l in tr} == {frozenset(man_tr.goal_geometry_tokens)}
    em = v7l.TacGoalEmitter(tr[:3], {j: l.clip_id for j, l in enumerate(tr[:3])})
    assert em.geometry_tokens == frozenset(man_tr.goal_geometry_tokens)          # after the EVAL load
    assert "goal_geometry_tokens" in man_tr.to_dict()                            # config.json states the policy


@_REAL
def test_the_trainer_installs_no_scope_when_the_policy_travels(setup) -> None:
    T, v7l, RT, ds, _, i = setup
    assert RT.v7_policy_travels(v7l) is True
    assert RT.v7_scope_for(v7l) is None and ds._v7_scope is None


@_REAL
def test_the_train_targets_follow_the_TRAIN_policy_in_process_and_in_a_worker(setup) -> None:
    T, v7l, RT, ds, _, i = setup
    assert _train_policy_holds(T, ds, i)


@_REAL
def test_DELIBERATE_REGRESSION_reading_the_module_global_goes_RED(setup, monkeypatch) -> None:
    """The unfixed behaviour: the policy read from the module state (the EVAL blob's, loaded last)."""
    T, v7l, RT, ds, _, i = setup
    monkeypatch.setattr(v7l, "_geometry_tokens_for", lambda label: v7l._MEASURED_GEOMETRY_TOKENS)
    assert not _train_policy_holds(T, ds, i)


def test_the_fallback_scope_isolates_a_module_state_policy() -> None:
    """For an UNFIXED module, ``V7PolicyScope`` snapshots at load and re-applies around each use (a fake module)."""
    from tanitad.train import refcv8_train as RT
    mod = types.ModuleType("_fake_v7l_for_scope")
    mod._MEASURED_GEOMETRY_TOKENS = frozenset({"A", "LANE_CHANGE_L"})
    mod._MEASURED_COT_TOKENS = frozenset({"B"})
    sys.modules[mod.__name__] = mod
    try:
        assert RT.v7_policy_travels(mod) is False
        sc = RT.v7_scope_for(mod)
        mod._MEASURED_GEOMETRY_TOKENS = frozenset({"A"})                       # the EVAL load overwrites
        with sc.applied():
            assert mod._MEASURED_GEOMETRY_TOKENS == frozenset({"A", "LANE_CHANGE_L"})
        assert mod._MEASURED_GEOMETRY_TOKENS == frozenset({"A"})                # restored after use
    finally:
        sys.modules.pop(mod.__name__, None)
