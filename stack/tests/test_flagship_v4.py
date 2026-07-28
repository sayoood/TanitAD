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


# ------------------------------------- (b2) E-H2 prior strength (λ, τ) ------
def _trained_grafts(head, seed: int = 7) -> None:
    """Give the three zero-init grafts a deterministic NON-zero weight, so the
    (λ, τ) knobs have something to scale. Zero-init grafts make every λ identical
    and would let a broken knob pass — the failure this helper exists to avoid."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for lin in (head.lat_to_anchor, head.lon_to_anchor, head.dist_to_anchor):
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * 0.05)


def test_lambda_one_tau_one_is_bit_identical_to_the_shipped_graft_path():
    """⛔ THE GATE for the whole (λ, τ) sweep: at the defaults the exposed knobs
    must reproduce the SHIPPED arithmetic bit-for-bit, or every swept cell is
    measured against a baseline that is not the deployed model.

    The reference is not another config — it is the pre-change expression
    ``W(log_softmax(logits)) summed, norm-clamped``, re-implemented here so the
    assertion cannot pass by comparing the new code to itself.
    """
    cfg = _small()
    cfg.seam_fail = 100.0                         # this test is about EQUALITY
    head = FlagshipV4Head(cfg).eval()
    _trained_grafts(head)                         # non-zero, else λ is unobservable
    b = _batch(cfg)
    out = _run(head, b)

    # --- the SHIPPED expression, verbatim, with no λ and no τ ----------------
    tokens = head.build_tokens(grad_scale(b["states"], 1.0), b["imagined"])
    m, _, _ = head.condition(b["v0"], b["vt_band"], b["route"], b["route_graded"])
    dec = head.decoder(tokens, m, steps=cfg.decoder.diffusion_steps)
    refined = dec["refined_logits"]
    lsm = torch.log_softmax
    lat_l, lon_l, dist_l = out["lat_logits"], out["lon_logits"], out["dist_logits"]
    graft = (head.lat_to_anchor(lsm(lat_l, dim=-1))
             + head.lon_to_anchor(lsm(lon_l, dim=-1))
             + head.dist_to_anchor(lsm(dist_l, dim=-1)))
    base = refined.norm(dim=-1).clamp_min(1e-9)
    ratio = graft.norm(dim=-1) / base
    scale = cfg.seam_clamp / ratio.clamp_min(cfg.seam_clamp)
    shipped = refined + graft * scale[:, None]

    assert cfg.graft_lambda == 1.0 and cfg.graft_tau == 1.0        # the defaults
    assert torch.equal(out["refined_logits"], shipped), (
        "λ=1, τ=1 is NOT bit-identical to the shipped graft path — every swept "
        "cell would be measured against the wrong baseline")
    assert torch.equal(out["sel_idx"],
                       out["sel_score"].argmax(dim=1))             # flat argmax


def test_graft_lambda_zero_removes_the_prior_exactly():
    """λ=0 must reproduce the PRE-GRAFT score bit-for-bit — the `F_base_only` arm
    becomes a config, not a re-implementation."""
    cfg = _small()
    cfg.seam_fail = 100.0
    head = FlagshipV4Head(cfg).eval()
    _trained_grafts(head)
    b = _batch(cfg)
    tokens = head.build_tokens(grad_scale(b["states"], 1.0), b["imagined"])
    m, _, _ = head.condition(b["v0"], b["vt_band"], b["route"], b["route_graded"])
    pre = head.decoder(tokens, m, steps=cfg.decoder.diffusion_steps)["refined_logits"]
    head.cfg.graft_lambda = 0.0
    out0 = _run(head, b)
    assert torch.equal(out0["refined_logits"], pre)
    assert out0["telemetry"]["seam_norm_ratio_preclamp_max"] == 0.0
    assert out0["telemetry"]["seam_clamp_bound_frac"] == 0.0


def test_graft_tau_to_zero_makes_the_class_posterior_one_hot():
    """τ→0 is MAXIMALLY HARD commitment WITHOUT truncating the candidate set —
    the axis `q` could never give us.

    Two properties are pinned, and the second is the one that is easy to get
    wrong: because the graft consumes ``log_softmax`` (not ``softmax``), τ→0 does
    NOT converge to "the argmax class's weight column". The non-argmax entries go
    to −(gap)/τ, so the graft's DIRECTION converges while its MAGNITUDE diverges
    like 1/τ. That divergence is precisely why the norm clamp decides the τ axis,
    and why every swept cell must report its pre-clamp ratio.
    """
    cfg = _small()
    cfg.seam_clamp, cfg.seam_fail = 1.0e9, 1.0e12  # isolate sharpness, not the clamp
    head = FlagshipV4Head(cfg).eval()
    _trained_grafts(head)
    b = _batch(cfg)
    tokens = head.build_tokens(grad_scale(b["states"], 1.0), b["imagined"])
    m, _, _ = head.condition(b["v0"], b["vt_band"], b["route"], b["route_graded"])
    pre = head.decoder(tokens, m, steps=cfg.decoder.diffusion_steps)["refined_logits"]

    def graft_at(tau):
        head.cfg.graft_tau = tau
        o = _run(head, b)
        return o, o["refined_logits"] - pre

    o_hi, _ = graft_at(1.0)
    # (1) the class posterior itself is one-hot at small τ
    for k in ("lat", "lon", "dist"):
        p = torch.softmax(o_hi[f"{k}_logits"].detach() / 1.0e-4, dim=-1)
        assert float(p.max(dim=-1).values.min()) > 1.0 - 1e-6, k

    # (2) the graft DIRECTION converges as τ→0 while its magnitude diverges ~1/τ
    _, g3 = graft_at(1.0e-3)
    _, g4 = graft_at(1.0e-4)
    cos = torch.nn.functional.cosine_similarity(g3, g4, dim=-1)
    assert float(cos.min()) > 0.999, float(cos.min())
    grow = (g4.norm(dim=-1) / g3.norm(dim=-1).clamp_min(1e-30))
    assert 5.0 < float(grow.min()) and float(grow.max()) < 20.0, float(grow.mean())

    # ... and the candidate set is STILL the full vocabulary: nothing was masked
    o4 = _run(head, b)
    assert torch.isfinite(o4["sel_score"]).all()
    assert o4["sel_score"].shape[1] == cfg.n_anchors


def test_the_reachability_clamp_is_OFF_on_v4_until_it_is_measured_there():
    """v1.5 turns the clamp on because it was MEASURED free on REF-C-XL's fan
    (72.08 % removed, oracle 100 % intact, paired Δ 0.0000). v4's own fan
    geometry was never dumped, so the property is UNMEASURED there — and v4
    carries the no-truncation invariant below. Inheritance must not smuggle a
    default across a surface it was not measured on."""
    from tanitad.models.flagship_v15 import V15Config
    assert V15Config().sel_reach_clamp is True
    assert V4Config().sel_reach_clamp is False
    assert tactical_config().sel_reach_clamp is False


def test_flipping_the_clamp_ON_V4_cannot_perturb_any_LOSS_TERM(capsys):
    """⭐ v5 prep §1.1 — the half of the v4 flip that IS measurable today.

    The v4 flip is gated on a sibling's zero-change measurement over v4's OWN
    emitted fan (paired Δ ADE, the 72.08 %/oracle-100 % property), which needs a
    fan dump that does not exist. That gate is about whether the clamp is FREE.

    A separate question is whether the flip is CODE-SAFE — whether turning it on
    can perturb the supervised path, i.e. change training rather than only the
    emitted pick. That is measurable right now on v4's own head, and it is
    measured here: with the clamp ON and OFF the head must return a BIT-IDENTICAL
    ``sel_score`` (the tensor ``v15_losses`` cross-entropies) and bit-identical
    loss terms, because the mask is applied at the ARGMAX ONLY.

    Δ is proven == 0, not asserted. This de-risks the flip to exactly one
    remaining question, the sibling's.
    """
    cfg = _small()
    head = FlagshipV4Head(cfg).eval()
    b = _batch(cfg)

    head.cfg.sel_reach_clamp = True
    on = _run(head, b)
    head.cfg.sel_reach_clamp = False
    off = _run(head, b)

    d = (on["sel_score"] - off["sel_score"]).abs().max().item()
    assert d == 0.0, f"the clamp perturbed the supervised score by {d}"
    assert torch.equal(on["sel_score"], off["sel_score"])
    assert torch.isfinite(on["sel_score"]).all(), "no -inf may reach the CE"

    la = v15_losses(on, head.decoder.anchors, b["traj_tgt"])
    lb = v15_losses(off, head.decoder.anchors, b["traj_tgt"])
    for k in ("cls", "cls_refined", "traj", "loss"):
        assert float((la[k] - lb[k]).abs()) == 0.0, f"{k} moved under the clamp"

    # and the clamp really was live (a guard that did nothing proves nothing)
    assert "reach_frac_candidates_clipped" in on["telemetry"]
    assert "reach_frac_candidates_clipped" not in off["telemetry"]


def test_the_selector_never_truncates_the_candidate_set():
    """⛔ `q` MUST NOT EXIST in the deployment path. Every candidate stays
    rankable at every (λ, τ): no masking, no -inf, no top-k. This is the guard
    that stops `H_graft(q)` — a MEASUREMENT arm that cost +0.21…+5.82 m — from
    ever creeping into the deployed selector."""
    cfg = _small()
    assert cfg.sel_reach_clamp is False, (
        "this guard is about SCORE-BASED truncation; a physical reachability "
        "band is a different object and is measured separately. If it is ever "
        "enabled on v4, this test must be re-derived, not silently relaxed.")
    cfg.seam_clamp, cfg.seam_fail = 1.0e6, 1.0e9
    head = FlagshipV4Head(cfg).eval()
    _trained_grafts(head)
    b = _batch(cfg)
    for lam, tau in ((0.0, 1.0), (1.0, 1.0), (8.0, 1.0), (1.0, 0.1), (8.0, 0.1)):
        head.cfg.graft_lambda, head.cfg.graft_tau = lam, tau
        out = _run(head, b)
        s = out["sel_score"]
        assert s.shape[1] == cfg.n_anchors, (lam, tau)
        assert torch.isfinite(s).all(), (lam, tau)
        # every candidate is reachable: no anchor is pinned at -inf/NaN, and the
        # emitted pick is the flat argmax over ALL of them
        assert torch.equal(out["sel_idx"], s.argmax(dim=1)), (lam, tau)


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
def test_seam_clamp_fails_loud_on_SUSTAINED_saturation():
    """The guard still kills a genuine runaway — but only a SUSTAINED one.

    Redesigned 2026-07-28 (C51): the trigger is a POPULATION condition over TIME
    (mean ratio > seam_fail AND bound_frac > seam_fail_frac, held for
    seam_fail_patience consecutive steps), not `ratio.max()` on one batch.
    """
    cfg = _small()
    cfg.seam_fail_patience = 3                   # short, so the test is fast
    head = FlagshipV4Head(cfg).eval()
    b = _batch(cfg)
    with torch.no_grad():                        # make a graft swamp the base score
        head.lat_to_anchor.weight.fill_(50.0)

    # it must NOT fire before patience is reached ...
    for i in range(cfg.seam_fail_patience - 1):
        out = _run(head, b)
        assert out["telemetry"]["seam_sat_steps"] == i + 1
    # ... and must fire once it is
    with pytest.raises(RuntimeError, match="SATURATED"):
        _run(head, b)


def test_a_MINORITY_of_saturated_samples_can_NEVER_kill_the_run():
    """⭐ THE C51 REGRESSION PIN — the defect that cost the PI's geometry
    validation both wide arms.

    The old rule fired on `ratio.max()`, so ONE sample out of 64 could end a
    multi-GPU-hour run. The fix makes the batch FRACTION a REQUIRED CONJUNCT, so
    no minority can trigger a kill however extreme it is.

    ⚠️ Note on what is and is not driven here: the per-sample ratio is a function
    of the decoder's own output, so a *literal* one-of-N outlier cannot be dialled
    in from the outside. What IS decisive — and is what the fix turns on — is that
    the fraction gate is a real conjunct rather than decoration. So this drives a
    batch with an ENORMOUS mean ratio (~10^3, far past `seam_fail`) while the
    fraction requirement is unreachable, and asserts the guard stays silent and
    the counter never accumulates. If the fraction gate were dropped or ORed, this
    test fires immediately.
    """
    cfg = _small()
    cfg.seam_fail_frac = 1.01                    # unreachable: no batch is >100 %
    cfg.seam_fail_patience = 3
    head = FlagshipV4Head(cfg).eval()
    b = _batch(cfg)
    with torch.no_grad():
        head.lat_to_anchor.weight.fill_(50.0)    # mean ratio far above seam_fail

    for _ in range(cfg.seam_fail_patience + 5):  # well past the old kill point
        t = _run(head, b)["telemetry"]           # must NOT raise
        assert t["seam_norm_ratio_preclamp_mean"] > cfg.seam_fail, (
            "fixture is vacuous — the mean must exceed seam_fail, or this test "
            "proves nothing about the conjunct")
        assert t["seam_sat_steps"] == 0, (
            "the counter accumulated while the POPULATION condition was unmet — "
            "the fraction gate is not a real conjunct, i.e. C51 has returned")


def test_the_saturation_counter_RESETS_on_a_healthy_step():
    """A transient spike must never accumulate into a kill across gaps."""
    cfg = _small()
    cfg.seam_fail_patience = 3
    head = FlagshipV4Head(cfg).eval()
    b = _batch(cfg)

    with torch.no_grad():
        head.lat_to_anchor.weight.fill_(50.0)    # saturating
    assert _run(head, b)["telemetry"]["seam_sat_steps"] == 1
    assert _run(head, b)["telemetry"]["seam_sat_steps"] == 2

    with torch.no_grad():
        head.lat_to_anchor.weight.zero_()        # healthy again
    assert _run(head, b)["telemetry"]["seam_sat_steps"] == 0, "counter did not reset"

    with torch.no_grad():
        head.lat_to_anchor.weight.fill_(50.0)    # saturating again
    # must start counting from 1, NOT resume at 3 and fire immediately
    assert _run(head, b)["telemetry"]["seam_sat_steps"] == 1


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


def test_seam_fail_is_a_pure_guard_and_changes_no_computed_value():
    """⭐ The claim that rescued the PI's geometry validation without re-running
    its control arm.

    MEASURED 2026-07-28: the small validation lost BOTH wide arms to this guard
    (B_wide pre-clamp 1.760, C_v5 1.511, both ~step 350 at λ_plan 0.833) while
    the 51.4° control ran clean to 1500. Because arm A never raised, its
    pre-clamp max never crossed 1.5 at ANY forward pass — so raising the
    threshold is provably a NO-OP for arm A, the A-vs-B contrast stays matched,
    and A does not need re-running (3 h 47 m of A40 saved).

    That argument rests entirely on ``seam_fail`` appearing ONLY in the raise
    (``flagship_v4.py:233``) and never in a computed value — ``seam_clamp`` is
    what shapes the graft. This test is that argument, executable: two heads
    identical but for ``seam_fail``, both above the trip, must agree bit-exactly;
    and the threshold must still control whether it raises at all.

    ⚠️ It is NOT a claim the guard is worthless — it is a claim the guard is a
    REPORTING threshold wearing a kill switch's clothes.
    """
    FILL = 50.0                                   # as the saturation test above

    def _built(seam_fail: float, patience: int = 1):
        torch.manual_seed(0)                      # identical init...
        cfg = _small()
        cfg.seam_fail = seam_fail
        cfg.seam_fail_patience = patience         # 1 => decide on this single step
        head = FlagshipV4Head(cfg).eval()
        with torch.no_grad():
            head.lat_to_anchor.weight.fill_(FILL)
        torch.manual_seed(1)                      # ...and an identical batch
        return head, _batch(cfg)

    # (1) measure the pre-clamp MEAN ratio with the guard effectively off.
    #     (C51: the trigger is the MEAN over the batch, no longer the max.)
    head_off, b_off = _built(1.0e9)
    torch.manual_seed(2)                          # the decoder is STOCHASTIC
    out_off = _run(head_off, b_off)
    r = out_off["telemetry"]["seam_norm_ratio_preclamp_mean"]
    assert r > _small().seam_fail, (
        f"the fixture must exceed the SHIPPED default {_small().seam_fail} for "
        f"this test to be decision-relevant (got {r})")
    assert out_off["telemetry"]["seam_clamp_bound_frac"] > _small().seam_fail_frac, (
        "the fixture must also saturate the POPULATION, or part (3) would be "
        "testing the wrong branch of the new guard")

    # (2) a threshold just above the observed ratio: no raise, and every tensor
    #     must be bit-identical to the guard-off run
    head_ok, b_ok = _built(r * 1.01)
    torch.manual_seed(2)
    out_ok = _run(head_ok, b_ok)
    moved = [k for k, v in out_off.items()
             if isinstance(v, torch.Tensor) and not torch.equal(v, out_ok[k])]
    assert not moved, (
        f"output(s) {moved} MOVED when ONLY seam_fail changed — seam_fail is not "
        f"a pure guard, and the 'arm A needs no re-run' argument is void.")

    # (3) just below it: the one and only thing the threshold changes
    head_bad, b_bad = _built(r * 0.99)
    torch.manual_seed(2)
    with pytest.raises(RuntimeError, match="SATURATED"):
        _run(head_bad, b_bad)


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
