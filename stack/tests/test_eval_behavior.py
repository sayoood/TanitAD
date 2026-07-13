"""Tests for scripts/eval_behavior.py (behavioral-quality eval).

CPU synthetic, no real data / no checkpoint:
  - GT maneuver labelling correct on hand-built turn/brake/straight/accel paths;
  - route-intent labelling correct + valid-mask honoured;
  - confusion-matrix + per-class P/R/F1 + macro-F1 + balanced-accuracy math
    correct by hand;
  - the vectorized imagine-and-select rollout is PINNED equal to
    fourbrain.TacticalSelector (reuse-by-equivalence, not reimplementation);
  - the classifier probe recovers a linearly-separable synthetic signal;
  - collect -> maneuver probe -> strategic probe -> imagine-and-select runs
    end-to-end through a real (untrained) smoke WorldModel + toy episodes.

Poses are built in float64 so the kinematic-threshold assertions are exact.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_behavior as eb  # noqa: E402
import refb_labels as rl  # noqa: E402

from tanitad.config import smoke_config  # noqa: E402
from tanitad.data.toy_driving import generate_episode  # noqa: E402
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from tanitad.models.fourbrain import (Maneuver, TacticalSelector,  # noqa: E402
                                      WorldModel)
from tanitad.models.readout import RidgeProbe  # noqa: E402


# --------------------------------------------------------------------------- #
# synthetic pose builders (poses [T,4] = x, y, yaw, v)                         #
# --------------------------------------------------------------------------- #
def _poses(T, yaw_fn, v_fn):
    t = torch.arange(T, dtype=torch.float64)
    yaw = torch.tensor([yaw_fn(int(i)) for i in t], dtype=torch.float64)
    v = torch.tensor([v_fn(int(i)) for i in t], dtype=torch.float64)
    # xy irrelevant to the maneuver label (kinematics from yaw/v only)
    xy = torch.zeros(T, 2, dtype=torch.float64)
    return torch.cat([xy, yaw[:, None], v[:, None]], dim=1)


# --------------------------------------------------------------------------- #
# GT maneuver labelling                                                        #
# --------------------------------------------------------------------------- #
def test_gt_maneuver_straight_is_lane_keep():
    poses = _poses(40, lambda i: 0.02, lambda i: 20.0)   # tiny yaw, const speed
    last = torch.tensor([0, 5, 10])
    assert torch.equal(eb.gt_maneuver(poses, last),
                       torch.full((3,), rl.LANE_KEEP))


def test_gt_maneuver_turn_left_and_right():
    # +0.02 rad/step * 20 = 0.4 rad > 0.15 threshold -> turn_left (CCW +)
    left = _poses(40, lambda i: 0.02 * i, lambda i: 20.0)
    assert int(eb.gt_maneuver(left, torch.tensor([0]))[0]) == rl.TURN_LEFT
    right = _poses(40, lambda i: -0.02 * i, lambda i: 20.0)
    assert int(eb.gt_maneuver(right, torch.tensor([0]))[0]) == rl.TURN_RIGHT


def test_gt_maneuver_brake_and_accel():
    # dv over 20 steps = -0.2*20 = -4 m/s < -1 -> brake_stop
    brake = _poses(40, lambda i: 0.0, lambda i: 20.0 - 0.2 * i)
    assert int(eb.gt_maneuver(brake, torch.tensor([0]))[0]) == rl.BRAKE_STOP
    # dv = +0.2*20 = +4 m/s > +1, straight -> accelerate
    accel = _poses(40, lambda i: 0.0, lambda i: 5.0 + 0.2 * i)
    assert int(eb.gt_maneuver(accel, torch.tensor([0]))[0]) == rl.ACCELERATE


def test_gt_maneuver_turn_overrides_brake():
    # braking INTO a turn is a TURN (priority order) — refb doctrine
    poses = _poses(40, lambda i: 0.02 * i, lambda i: 20.0 - 0.2 * i)
    assert int(eb.gt_maneuver(poses, torch.tensor([0]))[0]) == rl.TURN_LEFT


# --------------------------------------------------------------------------- #
# route intent                                                                 #
# --------------------------------------------------------------------------- #
def test_route_intent_classes_and_valid_mask():
    T = 160
    thr = math.radians(45.0)
    # net heading +0.01/step * 100 = 1.0 rad > 45deg -> route_left
    left = _poses(T, lambda i: 0.01 * i, lambda i: 20.0)
    cls, valid = eb.route_intent(left, torch.tensor([0]), T, thr)
    assert int(cls[0]) == 0 and bool(valid[0]) is True          # route_left
    straight = _poses(T, lambda i: 0.0, lambda i: 20.0)
    cls, _ = eb.route_intent(straight, torch.tensor([0]), T, thr)
    assert int(cls[0]) == 1                                       # route_straight
    right = _poses(T, lambda i: -0.01 * i, lambda i: 20.0)
    cls, _ = eb.route_intent(right, torch.tensor([0]), T, thr)
    assert int(cls[0]) == 2                                       # route_right
    # a window with < ROUTE_MIN future is route-invalid
    _, valid_late = eb.route_intent(straight, torch.tensor([T - 10]), T, thr)
    assert bool(valid_late[0]) is False


# --------------------------------------------------------------------------- #
# confusion-matrix + macro-F1 math (hand-verified)                             #
# --------------------------------------------------------------------------- #
def test_confusion_matrix_and_metrics_by_hand():
    y_true = torch.tensor([0, 0, 0, 1, 1, 2])
    y_pred = torch.tensor([0, 0, 1, 1, 2, 2])
    cm = eb.confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    p, r, f1, sup = eb.per_class_prf(cm)
    assert torch.allclose(p, torch.tensor([1.0, 0.5, 0.5], dtype=torch.float64))
    assert torch.allclose(r, torch.tensor([2 / 3, 0.5, 1.0], dtype=torch.float64))
    assert torch.allclose(f1, torch.tensor([0.8, 0.5, 2 / 3], dtype=torch.float64),
                          atol=1e-6)
    assert torch.equal(sup, torch.tensor([3, 2, 1], dtype=torch.float64))
    assert eb.macro_f1(cm) == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert eb.balanced_accuracy(cm) == pytest.approx((2 / 3 + 0.5 + 1.0) / 3)
    assert eb.accuracy(cm) == pytest.approx(4 / 6)


def test_macro_f1_ignores_absent_gt_class():
    # class 2 never appears in GT -> excluded from macro averaging
    y_true = torch.tensor([0, 0, 1, 1])
    y_pred = torch.tensor([0, 0, 1, 1])
    cm = eb.confusion_matrix(y_true, y_pred, 3)
    assert eb.macro_f1(cm) == pytest.approx(1.0)
    assert eb.balanced_accuracy(cm) == pytest.approx(1.0)


def test_class_balance_counts():
    labels = torch.tensor([0, 0, 0, 4, 1])
    b = eb.class_balance(labels, eb.N_MAN, eb.MANEUVER_CLASSES)
    assert b["n"] == 5
    assert b["counts"]["lane_keep"] == 3 and b["counts"]["brake_stop"] == 1
    assert b["frac"]["lane_keep"] == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# imagine-and-select rollout PINNED to fourbrain.TacticalSelector              #
# --------------------------------------------------------------------------- #
def test_roll_operative_matches_tactical_selector():
    torch.manual_seed(0)
    world = WorldModel(smoke_config()).eval()
    W = world.predictor.cfg.window
    S = world.state_dim
    A = world.predictor.cfg.action_dim
    states = torch.randn(1, W, S)
    past = torch.randn(1, W, A)
    # calibrated probe: latent -> 2-D ego (RidgeProbe, as the selector expects)
    probe = RidgeProbe(alpha=1.0).fit(torch.randn(64, S), torch.randn(64, 2))
    subgoal = torch.tensor([1.3, -0.7])
    cw = 0.05
    K = 3
    maneuvers = [Maneuver(f"m{j}", torch.randn(K, A)) for j in range(4)]
    ts = TacticalSelector(world, probe, comfort_weight=cw)
    with strict_numerics():
        idx, scores = ts.select(states, past, maneuvers, subgoal)
        # independent recompute via the vectorized rollout used by the eval
        my = []
        for m in maneuvers:
            z_end = eb.roll_operative(world, states, past, m.actions)
            xy = probe.predict(z_end)
            dist = (xy[:, :2] - subgoal).norm(dim=-1)
            comfort = m.actions.pow(2).mean()
            my.append(dist + cw * comfort)
        my = torch.stack(my, dim=1).squeeze(0)
    assert torch.allclose(my, scores, atol=1e-5)
    assert int(my.argmin()) == int(idx)


def test_build_primitives_classes():
    prims, klass = eb.build_primitives(0.15, 1.0, eb.SELECT_H, "cpu")
    assert prims.shape == (9, eb.SELECT_H, 2)
    # the zero-steer, zero-accel primitive is lane_keep
    zero = [i for i in range(9) if prims[i, 0, 0] == 0 and prims[i, 0, 1] == 0]
    assert len(zero) == 1 and int(klass[zero[0]]) == rl.LANE_KEEP
    # a +steer primitive is turn_left regardless of accel (turn priority)
    left = [i for i in range(9) if prims[i, 0, 0] > 0]
    assert all(int(klass[i]) == rl.TURN_LEFT for i in left)


# --------------------------------------------------------------------------- #
# classifier probe recovers a linearly separable signal                        #
# --------------------------------------------------------------------------- #
def test_fit_classifier_recovers_separable_signal():
    torch.manual_seed(0)
    n, f, c = 600, 8, 3
    centers = torch.randn(c, f) * 5.0
    y = torch.randint(0, c, (n,))
    X = centers[y] + torch.randn(n, f) * 0.3
    tr = torch.arange(0, n, 2)
    va = torch.arange(1, n, 2)
    pred, tr_acc = eb.fit_classifier(X[tr], y[tr], X[va], c, kind="linear",
                                     epochs=150, seed=0)
    acc = float((pred == y[va]).float().mean())
    assert acc > 0.9 and tr_acc > 0.9


# --------------------------------------------------------------------------- #
# end-to-end pipeline through a real (untrained) smoke WorldModel              #
# --------------------------------------------------------------------------- #
def test_pipeline_end_to_end_cpu():
    torch.manual_seed(0)
    world = WorldModel(smoke_config()).eval()
    window = world.predictor.cfg.window
    # long-enough episodes for the 2 s maneuver horizon + a route window
    eps = [generate_episode(i, steps=90, size=64) for i in range(4)]
    corpora = ["comma2k19", "comma2k19", "physicalai", "physicalai"]
    with strict_numerics():
        data = eb.collect(world, eps, corpora, "cpu", window,
                          math.radians(45.0), math.radians(20.0),
                          stride=6, batch=4, keep_states=True)
        assert data["man"].numel() > 0
        assert data["encoder_state"].shape[0] == data["man"].shape[0]
        assert data["has_tactical"] is False          # smoke_config has no tac
        seeds = [0, 1]
        man = eb.maneuver_probe_eval(data, seeds, 0.5, "cpu", epochs=20)
        strat = eb.strategic_probe_eval(data, seeds, 0.5, "cpu", epochs=20,
                                        turn_deg=(45.0, 20.0))
        sel = eb.imagine_and_select_eval(
            world, data, "cpu", seeds, 0.5, 0.15, 1.0, 0.01, max_windows=64)
        worst = eb.worst_k_errors(data, "cpu", 0.5, 0, 20, "encoder_state", k=5)

    # maneuver probe well-formed on the present latent sources
    assert "encoder_state" in man and "operative_k4" in man
    for scope, cell in man["encoder_state"].items():
        if "linear" in cell and "confusion_matrix" in cell["linear"]:
            cm = cell["linear"]["confusion_matrix"]
            assert len(cm) == eb.N_MAN and len(cm[0]) == eb.N_MAN
            assert 0.0 <= cell["linear"]["balanced_accuracy"] <= 1.0
    # strategic proxy present with the honest gap verdict
    assert "no intrinsic decodable code" in strat["verdict"].lower()
    assert "strict" in strat and "relaxed" in strat
    # imagine-and-select produced a confusion matrix + the goal-conditioned note
    assert "overall" in sel and "confusion_matrix" in sel
    assert "goal-conditioned" in sel["note"].lower()
    cm = torch.tensor(sel["confusion_matrix"])
    assert int(cm.sum()) == sel["n_windows"]
    # worst-K entries carry episode/step locators
    if not worst.get("skipped"):
        for w in worst["worst"]:
            assert {"corpus", "episode_id", "step", "gt_maneuver",
                    "pred_maneuver"} <= set(w)


def test_flagship4b_config_collect_and_probe_cpu():
    """The behavior eval runs on a flagship4b (smoke) WorldModel — the arch that
    the new --config flagship4b targets. base250cam's strict load would fail on
    a real flagship (policy keys + rebalanced depths), so --config must swap the
    factory; here we prove the flagship4b arch flows through collect + probe and
    that its tactical_pred turns has_tactical True."""
    from tanitad.config import flagship4b_smoke_config
    torch.manual_seed(0)
    world = WorldModel(flagship4b_smoke_config()).eval()
    window = world.predictor.cfg.window
    eps = [generate_episode(i, steps=100, size=64) for i in range(4)]
    corpora = ["comma2k19", "comma2k19", "physicalai", "physicalai"]
    with strict_numerics():
        data = eb.collect(world, eps, corpora, "cpu", window,
                          math.radians(45.0), math.radians(20.0),
                          stride=6, batch=4, keep_states=True)
        assert data["has_tactical"] is True        # flagship4b has tactical_pred
        man = eb.maneuver_probe_eval(data, [0, 1], 0.5, "cpu", epochs=10)
    cell = man["encoder_state"]["_all"]["linear"]
    assert 0.0 <= cell["seed_mean_std"]["balanced_accuracy"][0] <= 1.0
