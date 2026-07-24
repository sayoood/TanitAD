"""flagship v4 P1 (model core) + P2 (λ_plan seam) tests — tanitad/models/flagship_v4.py.

Pins the four structural changes v4 adds to the v1.5 head (V4_FLAGSHIP_DESIGN §6/§7,
build plan §15 P1/P2), each with the failure it guards against:

(a) BUDGET — the dense operative decoder is exactly 8,559,785 (design §3.1), the
    factorised heads + grafts stay under the 811,543 bound, and param_breakdown sums.
(b) ATTRIBUTABILITY — the three factorised grafts are zero-init, so the ranked
    selection path is BIT-IDENTICAL to the graft-free baseline at step 0. This is
    what lets ``--lat/lon/dist-weight 0`` reproduce the baseline (§16); v3enc
    "believed it had controls too" and could not attribute its own failure.
(c) NORM CLAMP — a graft that swamps the base score is a second selector, not a
    prior (the F3 / ROUTE-seam failure, fired at 2.80x). The clamp rescales in-graph
    at 1.0x and FAILS LOUD at 1.5x; the KILL secondary seam_norm_ratio_max<=1.0
    reads the post-clamp ratio.
(d) P5b NULL ROW — a dropped v0 becomes a LEARNED embedding row, never a zero-fill
    (0.0 m/s is in-distribution "stationary"; the measured v3enc root cause). The
    null row differs from measurement(0), and v1.5's default stays byte-identical.
(e) λ_plan SEAM (P2/O-20) — a GRADIENT scale at the trunk->planner boundary, not a
    loss weight: λ=1.0 is a strict no-op; λ=0.0 lets the heads train at full rate on
    a trunk they cannot move (the LP regime that Phase A reproduces byte-identically).
"""

from __future__ import annotations

import pytest
import torch

from tanitad.models.flagship_v4 import (DENSE_HORIZONS, N_DIST, N_LAT, N_LON,
                                        FlagshipV4Head, V4Config, param_breakdown,
                                        tactical_config, v4_config)
from tanitad.models.flagship_v15 import FlagshipV15Head, V15Config, v15_losses
from tanitad.models.metric_dynamics import grad_scale


def _small() -> V4Config:
    """A CPU-sized v4 head (same structure, shrunk widths) for shape/grad tests."""
    from tanitad.refs.refc import DecoderConfig
    cfg = V4Config()
    cfg.state_dim = 64
    cfg.readout_grid = 4
    cfg.d_cell = 4
    cfg.window = 4
    cfg.horizons = (1, 2, 3, 4)
    cfg.imag_read = (1, 2)
    cfg.n_anchors = 12
    cfg.d_token = 16
    cfg.d_meas = 8
    cfg.n_probes = 2
    cfg.factor_hidden = 8
    cfg.decoder = DecoderConfig(d=16, n_heads=2, layers=2, ff_mult=2,
                                aux_hidden=16, diffusion_steps=2, noise_std=0.1)
    return cfg


def _batch(cfg: V4Config, b: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "states": torch.randn(b, cfg.window, cfg.state_dim, generator=g),
        "v0": torch.rand(b, generator=g) * 20 + 3,
        "imagined": torch.randn(b, cfg.n_probes * len(cfg.imag_read),
                                cfg.state_dim, generator=g),
        "vt_band": torch.randint(0, 23, (b,), generator=g),
        "route": torch.randint(0, 4, (b,), generator=g),
        "route_graded": torch.randn(b, generator=g),
        "traj_tgt": torch.randn(b, len(cfg.horizons), 2, generator=g),
    }


def _run(head, b, **kw):
    return head(b["states"], b["v0"], imagined=b["imagined"], vt_band=b["vt_band"],
                route=b["route"], route_graded=b["route_graded"], vt_speed=b["v0"],
                **kw)


# --------------------------------------------------------------- (a) budget --
def test_param_budget_matches_the_design_and_sums():
    head = FlagshipV4Head(v4_config())
    pb = param_breakdown(head)
    assert pb["total"] == sum(p.numel() for p in head.parameters())
    # the dense operative decoder — the exact §3.1 measured figure (d384x4L, 20 steps)
    assert pb["decoder"] == 8_559_785, pb["decoder"]
    # factorised heads + grafts under the design's 811,543 bound; grafts ~5 k
    assert pb["factor_heads"] + pb["factor_grafts"] <= 811_543
    assert 4_000 <= pb["factor_grafts"] <= 8_000, pb["factor_grafts"]
    # P5b null row is present and small
    assert pb["ego_null"] == head.cfg.d_meas
    assert len(DENSE_HORIZONS) == 20                     # the smoothness precondition


def test_dense_plan_is_twenty_steps():
    head = FlagshipV4Head(v4_config()).eval()
    b = _batch(v4_config())
    # a 4-point head admits ONE third difference; the emitted plan must be dense
    out = head(b["states"][:, :head.cfg.window], b["v0"],
               imagined=torch.randn(3, head.cfg.n_probes * len(head.cfg.imag_read),
                                    head.cfg.state_dim),
               vt_band=b["vt_band"], route=b["route"],
               route_graded=b["route_graded"], vt_speed=b["v0"])
    assert out["traj"].shape[1] == 20


def test_tactical_instance_is_the_same_class_at_coarse_horizons():
    tac = FlagshipV4Head(tactical_config())
    assert tac.cfg.horizons == tuple(range(5, 51, 5))
    assert isinstance(tac, FlagshipV4Head)


# ------------------------------------------- (b) attributability / byte-id ---
def test_grafts_are_zero_init():
    head = FlagshipV4Head(_small())
    for name, g in (("lat", head.lat_to_anchor), ("lon", head.lon_to_anchor),
                    ("dist", head.dist_to_anchor)):
        assert int(torch.count_nonzero(g.weight)) == 0, name
        assert g.bias is None                    # bias=False, mirrors maneuver_to_anchor
    assert (head.lat_to_anchor.out_features, head.lat_to_anchor.in_features) \
        == (head.cfg.n_anchors, N_LAT)
    assert head.lon_to_anchor.in_features == N_LON
    assert head.dist_to_anchor.in_features == N_DIST


def test_zero_init_grafts_leave_the_ranked_score_bit_identical():
    """The core attributability claim: with zero-init grafts the selection path is
    bit-identical to the graft-free decoder output, so a v4-vs-baseline diff is
    caused by the graft alone, not by everything v4 changed at once."""
    cfg = _small()
    head = FlagshipV4Head(cfg).eval()
    b = _batch(cfg)
    # re-derive the graft-free ranked score exactly as forward builds it (eval =
    # deterministic: no ego/goal dropout, no denoise noise)
    tokens = head.build_tokens(grad_scale(b["states"], 1.0), b["imagined"])
    m, _, _ = head.condition(b["v0"], b["vt_band"], b["route"], b["route_graded"])
    dec = head.decoder(tokens, m, steps=cfg.decoder.diffusion_steps)
    out = _run(head, b)
    assert torch.equal(out["refined_logits"], dec["refined_logits"])
    assert out["telemetry"]["seam_norm_ratio_preclamp_max"] == 0.0


def test_a_trained_graft_actually_moves_the_ranking():
    """The flip side: once a graft is non-zero it MUST change the ranked score —
    else the seam is decorative (the failure §6.2 discipline 4 guards against).
    Seeded + high seam_fail so it is order-independent and isolates the effect."""
    torch.manual_seed(0)
    cfg = _small()
    cfg.seam_fail = 100.0                         # this test is about EFFECT, not the clamp
    head = FlagshipV4Head(cfg).eval()
    b = _batch(cfg)
    base = _run(head, b)["refined_logits"].clone()
    with torch.no_grad():
        head.lon_to_anchor.weight.fill_(0.1)     # deterministic nonzero graft
    moved = _run(head, b)["refined_logits"]
    assert not torch.allclose(base, moved)


# ------------------------------------------------------- (c) the norm clamp --
def test_seam_clamp_fails_loud_above_the_fail_ratio():
    cfg = _small()
    head = FlagshipV4Head(cfg).eval()
    b = _batch(cfg)
    with torch.no_grad():                        # make a graft swamp the base score
        head.lat_to_anchor.weight.fill_(50.0)
    with pytest.raises(RuntimeError, match="fail-loud"):
        _run(head, b)


def test_seam_clamp_rescales_in_graph_below_the_fail_ratio():
    """A graft that exceeds seam_clamp but stays under seam_fail is rescaled
    in-graph so its EFFECTIVE ratio never exceeds seam_clamp, and it does NOT
    raise. Deterministic (fixed fill, high seam_fail) so it exercises only the
    rescale path regardless of test order / global RNG state."""
    torch.manual_seed(0)                         # deterministic head init (order-independent)
    cfg = _small()
    cfg.seam_clamp = 0.05                         # low, so a modest graft trips it
    cfg.seam_fail = 1.0e6                         # effectively off: this test is the RESCALE path
    head = FlagshipV4Head(cfg).eval()
    b = _batch(cfg)
    with torch.no_grad():
        head.lat_to_anchor.weight.fill_(0.5)     # a nonzero graft above seam_clamp
    out = _run(head, b)
    pre = out["telemetry"]["seam_norm_ratio_preclamp_max"]
    eff = out["telemetry"]["seam_norm_ratio_max"]
    assert pre > cfg.seam_clamp, pre             # the graft did exceed the clamp
    assert eff <= cfg.seam_clamp + 1e-6, eff     # ... and was rescaled in-graph


# --------------------------------------------------------- (d) P5b null row --
def test_null_row_differs_from_the_zero_fill():
    """P5b: with ego_null_row a dropped v0 yields the LEARNED null embedding, which
    must NOT equal measurement(0) (the zero-fill the v3enc root cause used). Goal
    seams are switched off so ``m`` is the pure ego path and the comparison is fair
    within one head (same measurement MLP)."""
    torch.manual_seed(0)
    cfg = _small()
    cfg.ego_dropout = 1.0                         # drop every sample -> deterministic
    cfg.cond_vtarget = cfg.cond_route = False     # isolate the ego/measurement path
    head = FlagshipV4Head(cfg).train()            # ego_null_row True (V4Config)
    b = _batch(cfg)
    m_null, _, _ = head.condition(b["v0"], None, None, None)
    # every dropped row is the learned null embedding, independent of v0
    assert torch.allclose(m_null, head.ego_null[None].expand_as(m_null))
    # ... and it is NOT the zero-fill image measurement(0) — that is the whole point
    m_zero = head.measurement(torch.zeros(b["v0"].shape[0], 1))
    assert not torch.allclose(m_null, m_zero)


def test_zero_fill_path_is_the_measurement_of_zero():
    """The legacy (ego_null_row False) path must reproduce the exact v3enc-era
    behaviour — measurement(0) — so ``--ego-zero-fill`` is a faithful ablation."""
    torch.manual_seed(1)
    cfg = _small()
    cfg.ego_dropout = 1.0
    cfg.ego_null_row = False
    cfg.cond_vtarget = cfg.cond_route = False
    head = FlagshipV4Head(cfg).train()
    b = _batch(cfg)
    m_zero, _, _ = head.condition(b["v0"], None, None, None)
    assert torch.allclose(m_zero, head.measurement(torch.zeros(b["v0"].shape[0], 1)))


def test_v15_default_stays_byte_identical_and_checkpoint_compatible():
    """V15Config() must keep ego_null_row False so a trained v1.5 checkpoint (with
    no ego_null key) still loads and the model is unchanged."""
    assert V15Config().ego_null_row is False
    head = FlagshipV15Head(V15Config())
    assert not hasattr(head, "ego_null")
    assert "ego_null" not in dict(head.named_parameters())


def test_null_row_receives_gradient_when_a_sample_is_dropped():
    cfg = _small()
    cfg.ego_dropout = 1.0                         # guarantee a drop -> null row used
    cfg.goal_dropout = 0.0
    head = FlagshipV4Head(cfg).train()
    b = _batch(cfg)
    out = _run(head, b)
    v15_losses(out, head.decoder.anchors, b["traj_tgt"])["loss"].backward()
    assert head.ego_null.grad is not None and float(head.ego_null.grad.abs().sum()) > 0


# ------------------------------------------------------- (e) the λ_plan seam --
def test_lambda_plan_one_is_a_strict_noop():
    cfg = _small()
    head = FlagshipV4Head(cfg).eval()
    b = _batch(cfg)
    a = _run(head, b, lambda_plan=1.0)
    z = _run(head, b, lambda_plan=1.0)
    assert torch.equal(a["refined_logits"], z["refined_logits"])
    # grad_scale(x, 1.0) short-circuits to x itself (a strict no-op)
    x = torch.randn(4, 3)
    assert grad_scale(x, 1.0) is x


def test_lambda_plan_zero_stops_trunk_gradient_but_heads_still_train():
    """The LP regime (Phase A): λ_plan=0 forward-identical, but the planner loss
    pushes ZERO gradient into the trunk state while the head params still learn."""
    cfg = _small()
    cfg.ego_dropout = 0.0                         # deterministic seam test
    cfg.goal_dropout = 0.0
    head = FlagshipV4Head(cfg).train()
    b = _batch(cfg)

    def trunk_and_head_grad(lam):
        states = b["states"].clone().requires_grad_(True)
        head.zero_grad(set_to_none=True)
        out = head(states, b["v0"], imagined=b["imagined"], vt_band=b["vt_band"],
                   route=b["route"], route_graded=b["route_graded"],
                   vt_speed=b["v0"], lambda_plan=lam)
        v15_losses(out, head.decoder.anchors, b["traj_tgt"])["loss"].backward()
        trunk = 0.0 if states.grad is None else float(states.grad.abs().sum())
        headg = sum(float(p.grad.abs().sum()) for p in head.decoder.parameters()
                    if p.grad is not None)
        return trunk, headg

    trunk0, head0 = trunk_and_head_grad(0.0)
    trunk1, head1 = trunk_and_head_grad(1.0)
    assert trunk0 == 0.0, "λ_plan=0 must not push gradient into the trunk state"
    assert head0 > 0.0, "the planner heads must still train at λ_plan=0 (LP regime)"
    assert trunk1 > 0.0, "λ_plan=1 must let the planner gradient reach the trunk"


def test_lambda_plan_forward_value_is_invariant_to_lambda():
    """Only the backward changes with λ; the forward (and the emitted plan) is
    bit-exact for every λ — grad_scale is straight-through."""
    cfg = _small()
    head = FlagshipV4Head(cfg).eval()
    b = _batch(cfg)
    ref = _run(head, b, lambda_plan=1.0)["traj"]
    for lam in (0.0, 0.25, 0.5):
        assert torch.equal(_run(head, b, lambda_plan=lam)["traj"], ref), lam
