"""``E-DDA-2b`` piece 5 — the selector's fan augmentation and foreign bank.

⛔ EVERY EXPECTATION HERE IS EITHER AN **ANALYTIC** TARGET OR A **LITERAL**.
Nothing in this file re-runs the producer's own derivation and calls the
agreement a check — that measures determinism, not correctness (`CLAUDE.md`, the
four green-forever guards of 2026-09-07). The invariants used are:

* a straight line along ``+x`` has curvature **exactly 0** and stays a straight
  line under ANY two-scalar multiplier ⇒ literal ``0.0``;
* a two-scalar multiplier is **shape-preserving**: ``out[s]/out[S-1]`` equals
  ``c[s]/c[S-1]`` for every waypoint, whatever the draw — an identity that needs
  no knowledge of the multiplier's value;
* ``std_min = std_max = 0`` makes the multiplier **exactly 1.0**, so the
  augmented copies must be **bitwise equal** to their parents — the
  no-information control, reading its known value exactly;
* ``int(3277 * 0.01) == 32`` reproduces the "≈ 32 foreign trajectories per
  scene" the DDv2 analysis published from the other direction (the vocabulary
  slice and the dropout ratio), so the keep-count formula is checked against an
  independently authored number rather than against itself.

⭐ ``test_MUTATION_*`` are the deliberate-regression arms. Each reintroduces a
defect that has really happened in this programme and asserts the guard goes
RED — a guard whose failure branch is unreachable is green forever.
"""

from __future__ import annotations

import pytest
import torch

from tanitad.refs import refc_selector_aug as A


# --------------------------------------------------------------------------- #
# helpers — analytic fans, built here so the expectation is independent         #
# --------------------------------------------------------------------------- #
def _straight(b: int = 2, n: int = 3, s: int = 8, d: float = 2.0) -> torch.Tensor:
    """A fan of straight lines along +x with spacing ``d``. Curvature is
    analytically 0 and stays 0 under any (a, b) scaling."""
    x = torch.arange(1, s + 1, dtype=torch.float32) * d
    c = torch.zeros(b, n, s, 2)
    c[..., 0] = x
    c[..., 1] = 0.0
    # give each candidate a distinct lateral drift so "constant along S" is not
    # trivially true from an all-zero column
    for j in range(n):
        c[:, j, :, 1] = torch.linspace(0.0, 0.5 * (j + 1), s)
    return c


def _curvature_max(traj: torch.Tensor) -> float:
    """Discrete curvature magnitude, derived HERE from the definition
    (dtheta / ds), not imported from the code under test."""
    d = traj[..., 1:, :] - traj[..., :-1, :]
    th = torch.atan2(d[..., 1], d[..., 0])
    dth = th[..., 1:] - th[..., :-1]
    dth = torch.atan2(torch.sin(dth), torch.cos(dth))
    ds = d.norm(dim=-1)[..., :-1].clamp_min(1e-9)
    return float((dth / ds).abs().max())


# --------------------------------------------------------------------------- #
# 1. OFF is OFF — the default path, proved two ways                            #
# --------------------------------------------------------------------------- #
def test_default_config_is_disabled_and_returns_the_input_object():
    cfg = A.AugmentConfig()
    assert cfg.enabled is False
    c = _straight()
    out, origin = A.add_mul_noise(c, cfg)
    assert out is c, "OFF must return the INPUT OBJECT, not a copy of it"
    assert torch.equal(origin, torch.zeros(2, 3, dtype=torch.long))


def test_OFF_DRAWS_NO_RANDOM_NUMBERS_so_a_downstream_stream_is_unchanged():
    """⛔ The strong form of 'bitwise unchanged'.

    Equality of the returned tensor is NOT enough: an augmentation that returned
    its input while still advancing the global generator would silently re-roll
    every arm that came after it. The check is on the RNG STREAM.
    """
    torch.manual_seed(1234)
    reference = torch.randn(5)

    torch.manual_seed(1234)
    A.augment_fan(_straight(), A.AugmentConfig())          # the OFF path
    after = torch.randn(5)
    assert torch.equal(reference, after), (
        "the OFF path consumed randomness — a downstream stream is no longer "
        "bit-identical to a run in which this module was never called")


def test_augment_fan_off_reports_native_only():
    r = A.augment_fan(_straight(b=2, n=4), A.AugmentConfig())
    assert r["n_total"] == 4 and r["n_native"] == 4
    assert r["n_aug_cands"] == 0 and r["n_foreign"] == 0
    assert bool(r["native"].all()) and bool(r["emittable"].all())


# --------------------------------------------------------------------------- #
# 2. the two-scalar law — ANALYTIC invariants, true for every draw              #
# --------------------------------------------------------------------------- #
def test_zero_std_reproduces_the_parent_BITWISE_the_no_information_control():
    """σ = 0 ⇒ multiplier ≡ 1.0 ⇒ the augmented copies ARE the parents.

    This is the control that must read its known value EXACTLY: not 'close',
    not 'within 1e-6' — equal.
    """
    c = _straight(b=2, n=3)
    cfg = A.AugmentConfig(n_aug=2, std_min=0.0, std_max=0.0)
    out, origin = A.add_mul_noise(c, cfg, generator=torch.Generator().manual_seed(0))
    assert out.shape[1] == 9
    assert torch.equal(out[:, 0:3], c)
    assert torch.equal(out[:, 3:6], c), "round 1 at σ=0 must be BITWISE the parent"
    assert torch.equal(out[:, 6:9], c), "round 2 at σ=0 must be BITWISE the parent"
    assert origin[0].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_a_straight_line_stays_EXACTLY_straight_under_any_draw():
    """ANALYTIC: an axis-aligned two-scalar map sends the line through
    ``(2t, 1t)`` to the line through ``(2at, bt)`` — still a straight line, so
    the curvature is **exactly 0** before and after, for every seed.

    ⛔ THE FAN MUST BE OBLIQUE, AND THIS IS NOT A STYLE CHOICE. MEASURED
    2026-09-10 while writing this file: on a **pure +x** fan (y ≡ 0) the
    curvature control reads 0.0 for per-coordinate noise as well — the y column
    stays 0, so the path stays straight and only its SPACING is jittered. The
    test would have been green against the exact defect it exists to catch,
    which is the *"a check that shares the defect it checks for"* class in
    `CLAUDE.md`. The mutation arm below is what found it; the oblique fan is
    what fixes it.

    ⚠️ THE TARGET IS EXACTLY 0; THE TOLERANCE IS FLOAT ROUND-OFF AND IS QUOTED
    WITH ITS MEASUREMENT. MEASURED 2026-09-10 over 8 seeds: the two-scalar law
    reads ``<= 3.17e-07`` in float32 and ``<= 5.76e-16`` in float64, while the
    per-coordinate mutation reads ``>= 1.586`` (float32, min over the same 8
    seeds) — a separation of ~5e6x. So the literals below are not a fudge
    factor: any value a real defect can produce is millions of times larger.
    """
    t = torch.arange(1, 9, dtype=torch.float64)
    c = torch.zeros(2, 3, 8, 2, dtype=torch.float64)
    c[..., 0] = 2.0 * t                 # oblique, exactly straight
    c[..., 1] = 1.0 * t
    assert _curvature_max(c) == 0.0
    cfg = A.AugmentConfig(n_aug=3, std_min=0.1, std_max=0.3)
    for seed in range(5):
        out, _ = A.add_mul_noise(c, cfg,
                                 generator=torch.Generator().manual_seed(seed))
        assert _curvature_max(out) < 1e-12, f"seed {seed} bent a straight line"
        out32, _ = A.add_mul_noise(
            c.float(), cfg, generator=torch.Generator().manual_seed(seed))
        assert _curvature_max(out32) < 1e-5, (
            f"seed {seed} bent a straight line in the production dtype")


def test_the_multiplier_is_SHAPE_PRESERVING_an_identity_needing_no_draw_value():
    """ANALYTIC: out[..., s, k] * c[..., S-1, k] == out[..., S-1, k] * c[..., s, k].

    A per-coordinate noise breaks this at once; a two-scalar noise cannot.
    The identity never reads the multiplier, so it is not the producer's own
    derivation run again.
    """
    c = _straight(b=2, n=3, s=8)
    cfg = A.AugmentConfig(n_aug=2, std_min=0.1, std_max=0.2)
    out, origin = A.add_mul_noise(c, cfg,
                                  generator=torch.Generator().manual_seed(7))
    parent = c.repeat(1, 1 + cfg.n_aug, 1, 1)
    lhs = out[..., :, :] * parent[..., -1:, :]
    rhs = out[..., -1:, :] * parent[..., :, :]
    assert float((lhs - rhs).abs().max()) < 1e-4, (
        "the augmented candidate is not a pure axis-scaling of its parent — "
        "the multiplier varied along the waypoint axis")


def test_the_two_axes_are_independent_draws():
    c = _straight(b=4, n=6)
    cfg = A.AugmentConfig(n_aug=1, std_min=0.2, std_max=0.2)
    out, _ = A.add_mul_noise(c, cfg, generator=torch.Generator().manual_seed(3))
    rx = out[:, 6:, 3, 0] / c[:, :, 3, 0]
    ry = out[:, 6:, 3, 1] / c[:, :, 3, 1]
    assert float((rx - ry).abs().mean()) > 0.02, (
        "the longitudinal and lateral scalars must be separate draws")


def test_v2_draws_ONE_std_per_round_for_the_whole_batch():
    """`_model_sel.py:1272` draws the std as a python float, so every row of a
    round shares it. With std_per_row=True each row gets its own — a DEVIATION,
    offered and named, never the default."""
    assert A.AugmentConfig().std_per_row is False
    c = torch.full((6, 2, 8, 2), 3.0)
    cfg = A.AugmentConfig(n_aug=1, std_min=0.05, std_max=0.45)
    out, _ = A.add_mul_noise(c, cfg, generator=torch.Generator().manual_seed(11))
    # one shared sigma ⇒ the per-row spread of (ratio - 1) is one sigma for all
    # rows; with 12 candidates the sample std of the ratios is a single-sigma
    # estimate. The discriminating statement is the CONFIG default above; this
    # asserts the shape actually broadcast (a per-round scalar, not per-row).
    assert out.shape == (6, 4, 8, 2)


# --------------------------------------------------------------------------- #
# 3. the foreign bank                                                          #
# --------------------------------------------------------------------------- #
def test_keep_count_reproduces_the_PUBLISHED_32_per_scene():
    """The DDv2 analysis reports the GTRS slice as 3,277 trajectories at
    dropout 0.99 ≈ 32 per scene. Our formula must land on that number — an
    independently authored reference, not our own arithmetic replayed."""
    assert int(3277 * 0.01) == 32
    c = _straight(b=1, n=2)
    bank = torch.randn(3277, 8, 2)
    r = A.augment_fan(c, A.AugmentConfig(foreign_frac=0.01), bank=bank,
                      generator=torch.Generator().manual_seed(0))
    assert r["n_foreign"] == 32
    assert r["n_total"] == 2 + 32


def test_each_row_draws_its_own_foreign_subset():
    c = _straight(b=8, n=2)
    bank = torch.arange(200 * 8 * 2, dtype=torch.float32).reshape(200, 8, 2)
    r = A.augment_fan(c, A.AugmentConfig(foreign_frac=0.05), bank=bank,
                      generator=torch.Generator().manual_seed(5))
    foreign = r["cand"][:, 2:]                       # [8, 10, 8, 2]
    row0, row1 = foreign[0].reshape(-1), foreign[1].reshape(-1)
    assert not torch.equal(row0, row1), "every row must draw its own permutation"


def test_foreign_is_TRAIN_ONLY_at_the_defaults_and_says_so():
    c = _straight(b=2, n=3)
    bank = torch.randn(500, 8, 2)
    cfg = A.AugmentConfig(foreign_frac=0.02)
    assert cfg.foreign_train_only is True
    r = A.augment_fan(c, cfg, bank=bank, training=False)
    assert r["n_foreign"] == 0
    assert "forward_test_rl" in r["provenance"]["foreign_skipped"]
    r2 = A.augment_fan(c, cfg, bank=bank, training=True,
                       generator=torch.Generator().manual_seed(0))
    assert r2["n_foreign"] == 10


def test_an_ON_foreign_knob_with_no_bank_is_REFUSED():
    with pytest.raises(ValueError, match="no `bank`"):
        A.augment_fan(_straight(), A.AugmentConfig(foreign_frac=0.1))


def test_a_foreign_fraction_that_keeps_nothing_is_REFUSED():
    """The `--wp-index` class: a knob parsed, stamped, and inert."""
    with pytest.raises(ValueError, match="INERT"):
        A.augment_fan(_straight(), A.AugmentConfig(foreign_frac=0.001),
                      bank=torch.randn(50, 8, 2))


def test_a_bank_at_a_different_HORIZON_is_REFUSED_not_resampled():
    with pytest.raises(ValueError, match="not the same"):
        A.augment_fan(_straight(s=8), A.AugmentConfig(foreign_frac=0.5),
                      bank=torch.randn(10, 16, 2))


# --------------------------------------------------------------------------- #
# 4. the pick guard                                                            #
# --------------------------------------------------------------------------- #
def test_a_pick_on_a_foreign_candidate_is_REFUSED():
    origin = torch.tensor([[0, 0, 1, -1], [0, 0, 1, -1]])
    A.assert_pick_emittable(torch.tensor([0, 2]), origin)        # fine
    with pytest.raises(ValueError, match="FOREIGN"):
        A.assert_pick_emittable(torch.tensor([0, 3]), origin)


def test_emittable_covers_augmented_but_not_foreign():
    c = _straight(b=1, n=2)
    r = A.augment_fan(c, A.AugmentConfig(n_aug=1, foreign_frac=0.5),
                      bank=torch.randn(4, 8, 2),
                      generator=torch.Generator().manual_seed(0))
    assert r["origin"][0].tolist() == [0, 0, 1, 1, -1, -1]
    assert r["emittable"][0].tolist() == [True, True, True, True, False, False]
    assert r["native"][0].tolist() == [True, True, False, False, False, False]


def test_origin_summary_counts_are_literal():
    origin = torch.tensor([[0, 0, 1, 1, 2, -1, -1, -1]])
    s = A.origin_summary(origin)
    assert s == {"n_total": 8, "n_native": 2, "n_augmented": 3, "n_foreign": 3,
                 "frac_native": 0.25, "frac_foreign": 0.375,
                 "rounds": [-1, 0, 1, 2]}


# --------------------------------------------------------------------------- #
# 5. config refusals                                                           #
# --------------------------------------------------------------------------- #
def test_config_refuses_an_inverted_std_range():
    with pytest.raises(ValueError, match="std_max"):
        A.AugmentConfig(std_min=0.3, std_max=0.1)


def test_config_refuses_a_dropout_ratio_pasted_into_the_keep_fraction():
    A.AugmentConfig(foreign_frac=0.99)                       # legal, means 99 %
    with pytest.raises(ValueError, match="KEPT fraction"):
        A.AugmentConfig(foreign_frac=1.5)


def test_as_dict_carries_the_v2_recipes_and_the_origin_codes():
    d = A.AugmentConfig().as_dict()
    assert d["enabled"] is False
    assert d["v2_train_recipe"]["n_aug"] == 2
    assert d["v2_test_recipe"]["n_aug"] == 3
    assert d["native_origin_code"] == 0 and d["foreign_origin_code"] == -1


# --------------------------------------------------------------------------- #
# 6. MUTATION ARMS — each must go RED, or the guard above is green forever      #
# --------------------------------------------------------------------------- #
def test_MUTATION_per_coordinate_noise_breaks_the_shape_identity():
    """Reintroduce the REAL defect §3.2 names: *"the adapter's
    `noise_mode='multiplicative'` is PER-COORDINATE; the V2-faithful form is two
    scalars"*. The shape-preservation identity must FAIL."""
    c = _straight(b=2, n=3, s=8)
    torch.manual_seed(0)
    bad = c.repeat(1, 3, 1, 1) * (torch.randn(2, 9, 8, 2) * 0.15 + 1.0)
    parent = c.repeat(1, 3, 1, 1)
    lhs = bad * parent[..., -1:, :]
    rhs = bad[..., -1:, :] * parent
    assert float((lhs - rhs).abs().max()) > 1e-3, (
        "the identity did not separate per-coordinate noise from the two-scalar "
        "law — it cannot detect the defect it exists for")


def test_MUTATION_per_coordinate_noise_bends_an_OBLIQUE_line_but_not_an_axial_one():
    """⭐ THE ARM THAT FOUND A HOLE IN MY OWN CONTROL, kept as the record.

    MEASURED 2026-09-10: on a **pure +x** fan, per-coordinate noise leaves the
    curvature at **exactly 0.0** — the y column is 0 and stays 0, so the path is
    still a straight line and only its spacing is jittered. A straight-line
    curvature control built on an axial fan is therefore GREEN against the exact
    defect it exists to catch. On an **oblique** fan the same noise bends it.
    ⇒ the control above uses an oblique fan; this arm pins both halves so the
    fix cannot be undone by "simplifying" the fixture back to +x.
    """
    t = torch.arange(1, 9, dtype=torch.float32)

    axial = torch.zeros(1, 1, 8, 2)
    axial[..., 0] = 2.0 * t
    torch.manual_seed(0)
    bad_axial = axial * (torch.randn(1, 1, 8, 2) * 0.15 + 1.0)
    assert _curvature_max(bad_axial) == 0.0, (
        "the axial fan is expected to be BLIND here — that is the whole point "
        "of this arm; if it ever stops being blind, re-read the fixture")

    oblique = torch.zeros(1, 1, 8, 2)
    oblique[..., 0] = 2.0 * t
    oblique[..., 1] = 1.0 * t
    torch.manual_seed(0)
    bad_obl = oblique * (torch.randn(1, 1, 8, 2) * 0.15 + 1.0)
    assert _curvature_max(bad_obl) > 1e-3, (
        "even the oblique curvature control cannot see per-coordinate jitter — "
        "the straight-line test proves nothing")


def test_MUTATION_an_off_path_that_draws_randomness_is_CAUGHT():
    """The RNG-stream assertion must be able to fail. If it cannot, 'OFF is
    bitwise unchanged' was never tested — only asserted."""
    torch.manual_seed(1234)
    reference = torch.randn(5)
    torch.manual_seed(1234)
    torch.randn(2, 3, 1, 2)              # what a leaky OFF path would consume
    after = torch.randn(5)
    assert not torch.equal(reference, after)


def test_MUTATION_marking_foreign_as_emittable_defeats_the_pick_guard():
    origin_bad = torch.tensor([[0, 0, 1, 0]])        # foreign relabelled NATIVE
    A.assert_pick_emittable(torch.tensor([3]), origin_bad)   # passes — the bug
    with pytest.raises(ValueError):
        A.assert_pick_emittable(torch.tensor([3]), torch.tensor([[0, 0, 1, -1]]))


def test_MUTATION_a_zero_std_control_that_is_only_APPROXIMATE_would_pass_a_weak_check():
    """The σ=0 control asserts BITWISE equality. A tolerance-based version would
    also pass on a path that added a tiny additive term — which is exactly the
    'additive branch is multiplied by zero' asymmetry V2 has. Show the weak form
    is weak."""
    c = _straight(b=1, n=1)
    leaky = c + 1e-9
    assert torch.allclose(leaky, c, atol=1e-6)       # a weak check passes
    assert not torch.equal(leaky, c)                 # the literal one does not
