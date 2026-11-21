"""Standalone tests for the D1-D3 gate runner.

Runs with no simulator and no trained model. ``tanitad`` must be importable
(editable stack install). Two families of tests:

  * controlled-linear tests exercise the PASS path with data where the ridge
    probe is (near) exact, so thresholds are genuinely met;
  * doctrine tests exercise the BLOCKED path — the whole reason the runner exists
    is that a failing instrument row must forbid the claim (D-004);
  * one end-to-end test runs the real ``WorldModel(smoke_config)`` through the
    runner to prove the composition executes and is well-formed.

    pytest "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-14-gate-runner-d1-d3/tests" -q
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tanitad_gates as g  # noqa: E402
from tanitad.config import smoke_config  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402


torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _linear_states(n=60, s=16, seed=0):
    """States plus a linear target map, so a ridge probe is (near) exact."""
    gen = torch.Generator().manual_seed(seed)
    states = torch.randn(n, s, generator=gen)
    w = torch.randn(s, 2, generator=gen)
    targets = states @ w
    episode_ids = [i // 6 for i in range(n)]          # 10 episodes, 6 frames each
    return states, targets, episode_ids


def _good_i2(states):
    """A batch-independent encode_fn passes I2; frames just need to exist."""
    frames = states.reshape(states.shape[0], 1, 4, 4)
    return g.I2Input(encode_fn=lambda x: x.flatten(1), frames=frames, batch_size=32)


def _bad_i2(states):
    """A batch-DEPENDENT encode_fn (adds the batch mean) fails I2 — a banned
    batch-statistic layer, the exact ALPS-4B BatchNorm incident (D-004)."""
    frames = states.reshape(states.shape[0], 1, 4, 4)
    fn = lambda x: x.flatten(1) + x.flatten(1).mean(dim=0, keepdim=True)
    return g.I2Input(encode_fn=fn, frames=frames, batch_size=32)


# --------------------------------------------------------------------------- #
# standard primitives                                                          #
# --------------------------------------------------------------------------- #
def test_ade_fde_shapes_and_values():
    pred = torch.zeros(4, 3, 2)
    true = torch.ones(4, 3, 2)                        # each point off by sqrt(2)
    ade, fde = g.ade_fde(pred, true)
    assert ade == pytest.approx(2 ** 0.5, rel=1e-5)
    assert fde == pytest.approx(2 ** 0.5, rel=1e-5)
    # 2-D input is treated as a single waypoint
    ade2, fde2 = g.ade_fde(torch.zeros(4, 2), torch.ones(4, 2))
    assert ade2 == pytest.approx(fde2)


def test_split_by_episode_is_disjoint():
    episode_ids = [i // 5 for i in range(50)]         # 10 episodes
    tr, va = g.split_by_episode(episode_ids, val_frac=0.3, seed=1)
    tr_ep = {episode_ids[i] for i in tr}
    va_ep = {episode_ids[i] for i in va}
    assert tr_ep.isdisjoint(va_ep)
    assert len(va_ep) > 0 and set(tr) | set(va) == set(range(50))


# --------------------------------------------------------------------------- #
# D1                                                                           #
# --------------------------------------------------------------------------- #
def test_d1_pass_and_grid_beats_pool():
    states, targets, eps = _linear_states()
    # A7: global pooling destroys spatial LAYOUT. Model that by collapsing every
    # cell to its mean (repeated) -> the per-cell info the target needs is gone.
    pooled = states.mean(dim=1, keepdim=True).expand(-1, states.shape[1]).contiguous()
    r = g.run_d1(states, targets, eps, unit="camera", i2=_good_i2(states),
                 pooled_states=pooled)
    assert r.admissible and r.passed and r.status == "PASS"
    assert r.instruments[0]["row"] == "I1"            # instruments FIRST
    assert r.metrics["ade@1s"] < g.D1_ADE_MAX["camera"]
    assert r.ablation["grid_beats_pool"] is True


def test_d1_blocked_when_i2_fails():
    states, targets, eps = _linear_states()
    r = g.run_d1(states, targets, eps, unit="camera", i2=_bad_i2(states))
    assert r.admissible is False and r.passed is False and r.status == "BLOCKED"
    assert any(b.startswith("I2") for b in r.blockers)
    assert "BLOCKED" in r.verdict


def test_d1_blocked_when_i2_missing():
    states, targets, eps = _linear_states()
    r = g.run_d1(states, targets, eps, unit="camera", i2=None)
    assert r.status == "BLOCKED" and any(b.startswith("I2") for b in r.blockers)


def test_d1_bev_threshold_is_stricter_than_camera():
    assert g.D1_ADE_MAX["bev"] < g.D1_ADE_MAX["camera"]


# --------------------------------------------------------------------------- #
# D2                                                                           #
# --------------------------------------------------------------------------- #
def test_d2_pass_when_imag_tracks_truth():
    states, disp, eps = _linear_states()
    z_prev = states
    z_true = states + 0.05 * torch.randn_like(states)   # small step
    z_imag = z_true + 0.01 * torch.randn_like(states)   # good predictor -> imag-rel << 0.8
    r = g.run_d2(z_prev, z_true, z_imag, disp, eps, i2=_good_i2(states))
    assert r.admissible and r.passed and r.status == "PASS"
    assert r.metrics["direction_acc"] > g.D2_DIR_ACC_MIN
    i4 = [row for row in r.instruments if row["row"] == "I4"][0]
    assert i4["value"] < g.D2_IMAG_REL_MAX


def test_d2_blocked_when_imagination_worse_than_persistence():
    states, disp, eps = _linear_states()
    z_prev = states
    z_true = states + 0.05 * torch.randn_like(states)
    z_imag = z_true + 5.0 * torch.randn_like(states)    # garbage predictor -> imag-rel >> 0.8
    r = g.run_d2(z_prev, z_true, z_imag, disp, eps, i2=_good_i2(states))
    assert r.status == "BLOCKED" and any(b.startswith("I4") for b in r.blockers)


# --------------------------------------------------------------------------- #
# D3                                                                           #
# --------------------------------------------------------------------------- #
def test_d3_pass_ratio_within_bound():
    states, tgt, eps = _linear_states()
    z_prev = states
    z_true_future = states + 0.05 * torch.randn_like(states)
    z_imag_future = z_true_future + 0.01 * torch.randn_like(states)
    r = g.run_d3(z_prev, z_true_future, z_imag_future, tgt, eps, i2=_good_i2(states))
    assert r.admissible and r.passed
    assert r.metrics["ratio"] <= g.D3_RATIO_MAX
    assert "a3_calibration_helps" in r.ablation


def test_d3_reports_oracle_and_imagined_ade():
    states, tgt, eps = _linear_states()
    r = g.run_d3(states, states + 0.05 * torch.randn_like(states),
                 states + 0.05 * torch.randn_like(states), tgt, eps, i2=_good_i2(states))
    assert r.metrics["oracle_decode_ade@2s"] >= 0.0
    assert r.metrics["imagined_ade@2s"] >= 0.0


# --------------------------------------------------------------------------- #
# assembly + doctrine                                                          #
# --------------------------------------------------------------------------- #
def test_metrics_json_instruments_first_and_summary():
    states, targets, eps = _linear_states()
    d1 = g.run_d1(states, targets, eps, unit="camera", i2=_good_i2(states))
    d1_blocked = g.run_d1(states, targets, eps, unit="camera", i2=None)
    out = g.gates_metrics_json("p0-test", "deadbeef", [d1])
    keys = list(out.keys())
    assert keys.index("instruments") < keys.index("gates")   # instruments FIRST
    assert out["summary"]["D1"] == "PASS"
    assert g.gates_metrics_json("x", "y", [d1_blocked])["summary"]["D1"] == "BLOCKED"
    assert "doctrine" in out


def test_extra_metrics_hook_is_merged():
    states, targets, eps = _linear_states()
    r = g.run_d1(states, targets, eps, unit="camera", i2=_good_i2(states),
                 extra_metrics={"lops_stub": lambda pred, true: 0.42})
    assert r.metrics["lops_stub"] == 0.42


# --------------------------------------------------------------------------- #
# end-to-end: real WorldModel through the runner                               #
# --------------------------------------------------------------------------- #
def test_smoke_worldmodel_path_is_wellformed():
    cfg = smoke_config()
    world = WorldModel(cfg)
    world.eval()
    n, c, s = 40, cfg.encoder.in_channels, cfg.encoder.image_size
    frames = torch.randn(n, c, s, s)
    states = g.encode_states(world, frames)          # [N, S]
    targets = torch.randn(n, 2)
    eps = [i // 4 for i in range(n)]
    i2 = g.I2Input(encode_fn=lambda x: g.encode_states(world, x), frames=frames)
    r = g.run_d1(states, targets, eps, unit="camera", i2=i2)
    # A random encoder will not PASS the metric; the point is a well-formed,
    # instruments-first report whose I2 (batch-free norms) genuinely holds.
    assert r.instruments[0]["row"] == "I1"
    i2row = [row for row in r.instruments if row["row"] == "I2"][0]
    assert i2row["pass"] is True                     # smoke encoder is batch-consistent
    assert r.status in {"PASS", "FAIL"}              # admissible either way (not BLOCKED)
