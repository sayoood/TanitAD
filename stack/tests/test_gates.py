"""D1-D3 gate runner tests (integrated from the 2026-07-14 intake package,
reworked per D-017: P4 path, imag-rel demoted to diagnostic, I7 wiring).

Families: controlled-linear PASS paths; doctrine BLOCKED paths (a failing
instrument row forbids the claim, D-004); the A13 case (control usable despite
imag-rel > 1); end-to-end through a real WorldModel.
"""

import pytest
import torch

import tanitad.eval.gates as g
from tanitad.config import smoke_config
from tanitad.models.fourbrain import WorldModel

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _linear_states(n=60, s=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    states = torch.randn(n, s, generator=gen)
    w = torch.randn(s, 2, generator=gen)
    targets = states @ w
    episode_ids = [i // 6 for i in range(n)]          # 10 episodes, 6 frames each
    return states, targets, episode_ids


def _good_i2(states):
    frames = states.reshape(states.shape[0], 1, 4, 4)
    return g.I2Input(encode_fn=lambda x: x.flatten(1), frames=frames, batch_size=32)


def _bad_i2(states):
    """Batch-DEPENDENT encode_fn — the ALPS-4B BatchNorm incident (D-004)."""
    frames = states.reshape(states.shape[0], 1, 4, 4)
    fn = lambda x: x.flatten(1) + x.flatten(1).mean(dim=0, keepdim=True)
    return g.I2Input(encode_fn=fn, frames=frames, batch_size=32)


# --------------------------------------------------------------------------- #
# standard primitives                                                          #
# --------------------------------------------------------------------------- #
def test_ade_fde_shapes_and_values():
    ade, fde = g.ade_fde(torch.zeros(4, 3, 2), torch.ones(4, 3, 2))
    assert ade == pytest.approx(2 ** 0.5, rel=1e-5)
    assert fde == pytest.approx(2 ** 0.5, rel=1e-5)
    ade2, fde2 = g.ade_fde(torch.zeros(4, 2), torch.ones(4, 2))
    assert ade2 == pytest.approx(fde2)


def test_split_by_episode_is_disjoint():
    episode_ids = [i // 5 for i in range(50)]
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
    assert r.status == "BLOCKED" and any(b.startswith("I2") for b in r.blockers)


def test_d1_blocked_when_i2_missing():
    states, targets, eps = _linear_states()
    r = g.run_d1(states, targets, eps, unit="camera", i2=None)
    assert r.status == "BLOCKED" and any(b.startswith("I2") for b in r.blockers)


def test_d1_bev_threshold_is_stricter_than_camera():
    assert g.D1_ADE_MAX["bev"] < g.D1_ADE_MAX["camera"]


# --------------------------------------------------------------------------- #
# D2 (D-017 semantics)                                                         #
# --------------------------------------------------------------------------- #
def test_d2_pass_when_imag_tracks_truth():
    states, disp, eps = _linear_states()
    z_prev = states
    z_true = states + 0.05 * torch.randn_like(states)
    z_imag = z_true + 0.01 * torch.randn_like(states)
    r = g.run_d2(z_prev, z_true, z_imag, disp, eps, i2=_good_i2(states))
    assert r.admissible and r.passed and r.status == "PASS"
    assert r.metrics["direction_acc"] > g.D2_DIR_ACC_MIN
    # D-017: imag-rel is a diagnostic METRIC, not an instrument row
    assert "imag_rel_diagnostic" in r.metrics
    assert not any(row["row"] == "I4" for row in r.instruments)


def test_d2_a13_usable_despite_high_imag_rel():
    """The A13 case: raw imagination far off-manifold (imag-rel >> 1) yet the
    action contrast decodes — the gate must be assessable and PASS."""
    states, disp, eps = _linear_states()
    z_prev = states
    z_true = states + 0.05 * torch.randn_like(states)
    z_imag = z_true + 5.0                              # large SYSTEMATIC offset
    r = g.run_d2(z_prev, z_true, z_imag, disp, eps, i2=_good_i2(states))
    assert r.metrics["imag_rel_diagnostic"] > 1.0      # imagination "worse than persistence"
    assert r.status == "PASS"                          # ...but control is usable (A13)


def test_d2_garbage_predictor_does_not_pass():
    states, disp, eps = _linear_states()
    z_prev = states
    z_true = states + 0.05 * torch.randn_like(states)
    z_imag = 5.0 * torch.randn_like(states)            # pure noise imagination
    r = g.run_d2(z_prev, z_true, z_imag, disp, eps, i2=_good_i2(states))
    assert not r.passed                                # FAIL or BLOCKED(I1), never PASS


def test_d2_p4_forward_dynamics_path():
    """P4 (D-017): [low-D state ⊕ action] -> displacement, no imagination."""
    n = 60
    gen = torch.Generator().manual_seed(3)
    prev_state = torch.randn(n, 3, generator=gen)      # e.g. (v, yaw, kappa)
    actions = torch.randn(n, 2, generator=gen)
    w = torch.randn(5, 2, generator=gen)
    disp = torch.cat([prev_state, actions], dim=-1) @ w
    states, _, eps = _linear_states(n=n)
    z_prev = states
    z_true = states + 0.05 * torch.randn_like(states)
    z_imag = z_true + 0.01 * torch.randn_like(states)
    r = g.run_d2(z_prev, z_true, z_imag, disp, eps, i2=_good_i2(states),
                 actions=actions, prev_state=prev_state)
    assert r.metrics["p4_forward_dynamics_dir_acc"] is not None
    assert r.metrics["p4_forward_dynamics_dir_acc"] > g.D2_DIR_ACC_MIN
    assert r.passed


def test_d2_i7_mismatch_blocks():
    states, disp, eps = _linear_states()
    z_prev = states
    z_true = states + 0.05 * torch.randn_like(states)
    z_imag = z_true + 0.01 * torch.randn_like(states)
    r = g.run_d2(z_prev, z_true, z_imag, disp, eps, i2=_good_i2(states),
                 fit_meta={"f_eff_px": 266.0}, run_meta={"f_eff_px": 554.0})
    assert r.status == "BLOCKED" and any(b.startswith("I7") for b in r.blockers)


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
                 states + 0.05 * torch.randn_like(states), tgt, eps,
                 i2=_good_i2(states))
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
    assert keys.index("instruments") < keys.index("gates")
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
    states = g.encode_states(world, frames)
    targets = torch.randn(n, 2)
    eps = [i // 4 for i in range(n)]
    i2 = g.I2Input(encode_fn=lambda x: g.encode_states(world, x), frames=frames)
    r = g.run_d1(states, targets, eps, unit="camera", i2=i2)
    assert r.instruments[0]["row"] == "I1"
    i2row = [row for row in r.instruments if row["row"] == "I2"][0]
    assert i2row["pass"] is True
    assert r.status in {"PASS", "FAIL"}
