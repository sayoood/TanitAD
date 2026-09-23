"""The four MEASURED perception-supervision defects of 2026-09-22, and their fixes.

Source of the defects: ``TanitAD Research Lab/Architecture & Inference/Research/
2026-09-22-refcv6-review/PERCEPTION_DATA_REVIEW.md`` (D-1 .. D-4). Source of the fixes and
of every measured number quoted below: ``…/2026-09-23-refcv6-fixes/PERCEPTION_FIXES.md``.

⭐ **EVERY EXPECTATION HERE IS A LITERAL OR ANALYTIC**, never an expression over the code
under test. Three boxes whose visibility is decided before any code runs; a query budget
whose outcome is countable by hand; a lift mask with a hand-placed number of valid cells; a
weighted cross-entropy recomputed in plain Python. Re-running a producer's own derivation
and finding agreement measures determinism, not correctness -- the failure this file's
subject matter is made of.

⛔ The deliberate-regression arms live in ``…/2026-09-23-refcv6-fixes/code/
mutate_perception_fixes.py``: each reintroduces the exact historical defect by WHOLE-LINE
equality and requires the matching test below to go RED. A mutation that does not apply
ABORTS as INVALID rather than reporting a pass.

⛔ CPU only.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import numpy as np                                                       # noqa: E402

from tanitad.data.bev_raster import GRID_DEFAULT                         # noqa: E402
from tanitad.models import agent_slots as A                              # noqa: E402
from tanitad.models import refcv6_perception_branch as PB                # noqa: E402
from tanitad.models.agent_slots import (                                 # noqa: E402
    AgentSlotDecoder, SLOT_WIDTH, match_slots, targets_from_join,
)
from tanitad.models.bev_encoder import map_soft_ce                       # noqa: E402
from tanitad.models.box3d_head import (                                  # noqa: E402
    SLOT3D_WIDTH, box3d_set_loss, zh_targets,
)

# --------------------------------------------------------------------------- #
# the corpus figures these fixes were sized against. LINE-STAMPED, ALWAYS.
# v7-B1 = physicalai-b1-w120-256x640cyl, 4,566 clips / 875,657 frames /
# 28,958,699 boxes. ⛔ NOT the parity line -- `refc_agents.py` quoted parity's
# 41.06 %/15.67 % to justify a default on a B1 arm and that citation is now fixed.
# --------------------------------------------------------------------------- #
B1_N_BOXES = 28_958_699
B1_N_IN_VOCAB = 28_929_493
B1_N_IN_VOCAB_VISIBLE = 4_737_114          # MEASURED 2026-09-23, this session
B1_DIGEST_RAW = "c3937558f59299e7"
B1_DIGEST_VISIBLE = "c9dbc340acf7d27e"


def _pred(n_q: int, seed: int = 0, three_d: bool = True) -> dict:
    """A decoder output of the right shape. Its CONTENT never carries an expectation --
    every assertion below is about WHICH TARGETS are supervised, not about the loss value.
    """
    torch.manual_seed(seed)
    dec = AgentSlotDecoder(d_memory=16, n_memory=8, n_queries=n_q, d_model=32,
                           depth=1, n_heads=4, enforce_band=False)
    out = dec(torch.randn(1, 8, 16))
    if three_d:
        out = {**out, "cz": torch.zeros(1, n_q), "h": torch.ones(1, n_q) * 1.6}
    return out


def _targets(rows) -> dict:
    """``rows`` = [(cx, cy, l, w, yaw)] -> a 3-D target dict with all-False zh_mask."""
    a = np.zeros((len(rows), 6), dtype=np.float64)
    for i, (cx, cy, l, w, yaw) in enumerate(rows):
        a[i] = (cx, cy, yaw, l, w, 0.0)
    t = targets_from_join(a, classes=["automobile"] * len(rows))
    return zh_targets(t)


# =========================================================================== #
# D-1 -- the BOX loss applied NO visibility filter
# =========================================================================== #
def test_D1_three_boxes_whose_visibility_is_known_before_any_code_runs():
    """⭐ THE ANALYTIC ARM, and it is the same three boxes the review's P1 probe used.

    A at ``(+20, 0)``  -- 0 deg azimuth, 20 m ahead: VISIBLE.
    B at ``(-20, 0)``  -- 180 deg azimuth, BEHIND THE EGO: the camera cannot see it.
    C at ``(+120, 0)`` -- 0 deg azimuth but 2x the decode box's 60 m: the head cannot
                          EXPRESS it, so training against it is a loss reducible only by
                          being wrong somewhere reachable.

    MEASURED before the fix (``raw/p1_box_path_filter.json``): refcv6 ``n_target 3``
    against the v6 seam's ``3 -> visible 1``. The literals below are 1 / 2 / 3.
    """
    tgt = _targets([(20.0, 0.0, 4.5, 1.9, 0.0),
                    (-20.0, 0.0, 4.5, 1.9, 0.0),
                    (120.0, 0.0, 4.5, 1.9, 0.0)])
    pred = _pred(12)

    on = box3d_set_loss(pred, tgt, visible_filter=True)
    assert on["n"]["target_prefilter"] == 3
    assert on["n"]["target_visible"] == 1
    assert on["n"]["dropped_not_visible"] == 2
    assert on["n"]["target"] == 1
    assert on["n"]["matched"] == 1

    # ⛔ THE DELIBERATE REGRESSION, by name -- the state of this path until 2026-09-23.
    off = box3d_set_loss(pred, tgt, visible_filter=False)
    assert off["n"]["target_prefilter"] == 3
    assert off["n"]["target_visible"] == 3
    assert off["n"]["dropped_not_visible"] == 0
    assert off["n"]["target"] == 3
    assert off["n"]["matched"] == 3


def test_D1_THE_DEFAULT_IS_FILTERED_because_the_trainer_never_names_it():
    """⛔⛔ THE TEST THE MUTATION HARNESS DEMANDED, AND IT IS THE ONE THAT MATTERS.

    Every other D-1 assertion passes ``visible_filter=`` explicitly, so every one of them
    stays GREEN when the DEFAULT is flipped back to ``False`` -- arm ``M1`` escaped exactly
    that way on its first run. The trainer calls ``box3d_loss_row(_s3, _t3, ...)`` and
    names no filter, so the default IS the behaviour of every refcv6 arm. Asserted with no
    keyword at all, on both entry points.
    """
    tgt = _targets([(20.0, 0.0, 4.5, 1.9, 0.0),
                    (-20.0, 0.0, 4.5, 1.9, 0.0),
                    (120.0, 0.0, 4.5, 1.9, 0.0)])
    pred = _pred(12)

    default = box3d_set_loss(pred, tgt)                  # ⛔ no keyword
    assert default["_visible_filter"] is True
    assert default["n"]["target_visible"] == 1
    assert default["n"]["dropped_not_visible"] == 2

    row = PB.box3d_loss_row(pred, tgt)                   # ⛔ no keyword, trainer's entry
    assert row["box3d_n_target_prefilter"] == 3.0
    assert row["box3d_n_target_visible"] == 1.0
    assert row["box3d_n_dropped_not_visible"] == 2.0
    # ⛔ AND THE ARM TRAVELS IN THE ROW. `dropped_not_visible == 0` means "filter off" or
    # "nothing to drop"; only the flag beside it tells them apart.
    assert row["box3d_visible_filter"] == 1.0
    # and the same row with the regression named must differ, or the flag reaches nothing
    off = PB.box3d_loss_row(pred, tgt, visible_filter=False)
    assert off["box3d_n_target_visible"] == 3.0
    assert off["box3d_visible_filter"] == 0.0


def test_D1_the_surviving_target_is_A_not_merely_one_of_them():
    """⛔ A COUNT OF 1 IS NOT PROOF IT KEPT THE RIGHT ONE. Filtering to any single box
    would satisfy the test above; this one names the survivor by its coordinates."""
    tgt = _targets([(20.0, 0.0, 4.5, 1.9, 0.0),
                    (-20.0, 0.0, 4.5, 1.9, 0.0),
                    (120.0, 0.0, 4.5, 1.9, 0.0)])
    pred = _pred(12)
    from tanitad.refs.refc_agents import visible_target_filter
    kept = visible_target_filter(tgt)["valid"][0]
    assert kept.tolist() == [True, False, False]
    m = match_slots(pred, visible_target_filter(tgt))
    assert m["cols"][0].tolist() == [0]          # index of A, a literal


def test_D1_the_boundary_is_60_DEGREES_and_60_METRES_exactly():
    """⭐ AN ANALYTIC TARGET AT THE EDGE. At ``cx = 10`` the 60 deg ray is at
    ``cy = 10*tan(60 deg) = 17.3205 m``; the decode box's own ``y_half`` is 16.0 m, so the
    BOX is the binding cut there and a point at ``cy = 17.0`` must fail on range even
    though it is inside the field. Two adjacent points, opposite verdicts."""
    assert float(GRID_DEFAULT.x_fwd_m) == 60.0
    assert float(GRID_DEFAULT.y_half_m) == 16.0
    from tanitad.refs.refc_agents import (FOV_HALF_ANGLE_RAD,
                                          filter_targets_to_visible,
                                          visible_target_filter)
    assert math.degrees(FOV_HALF_ANGLE_RAD) == pytest.approx(60.0, abs=1e-9)

    rows = [(10.0, 15.9, 4.5, 1.9, 0.0),     # in field (58.1 deg) and in box
            (10.0, 17.0, 4.5, 1.9, 0.0),     # in field (59.5 deg), OUTSIDE |cy|<=16
            (10.0, 17.4, 4.5, 1.9, 0.0),     # 60.1 deg -- outside the field itself
            (59.9, 0.0, 4.5, 1.9, 0.0),      # inside 60 m
            (60.1, 0.0, 4.5, 1.9, 0.0)]      # outside 60 m
    tgt = _targets(rows)
    assert visible_target_filter(tgt)["valid"][0].tolist() == \
        [True, False, False, True, False]
    # the FIELD cut alone (no box) keeps the first two -- which is how we know the second
    # row failed on RANGE and not on azimuth
    assert filter_targets_to_visible(tgt)["valid"][0].tolist() == \
        [True, True, False, True, True]


def test_D1_a_supplied_match_with_the_filter_on_is_REFUSED():
    """⛔ THE SILENT PATH IS CLOSED. A match built over unfiltered targets would re-admit
    exactly the boxes the filter removed, and `n` would then describe a different set from
    the loss -- a scope error wearing a green test."""
    tgt = _targets([(20.0, 0.0, 4.5, 1.9, 0.0), (-20.0, 0.0, 4.5, 1.9, 0.0)])
    pred = _pred(12)
    m = match_slots(pred, tgt)
    with pytest.raises(ValueError) as e:
        box3d_set_loss(pred, tgt, match=m, visible_filter=True)
    assert "match" in str(e.value).lower()
    # the same call with the filter named OFF is legal -- the control that proves the
    # refusal is about the PAIRING and not about `match=` itself
    assert box3d_set_loss(pred, tgt, match=m, visible_filter=False)["n"]["target"] == 2


def test_D1_the_zh_mask_follows_the_filtered_validity():
    """A filtered-out row must not carry a 3-D target into the z/h loop."""
    rows = [(20.0, 0.0, 4.5, 1.9, 0.0), (-20.0, 0.0, 4.5, 1.9, 0.0)]
    t = targets_from_join(np.array([[cx, cy, yaw, l, w, 0.0]
                                    for cx, cy, l, w, yaw in rows]),
                          classes=["automobile"] * 2)
    t3 = zh_targets(t, torch.ones(1, 2) * 0.8, torch.ones(1, 2) * 1.6)
    assert int(t3["zh_mask"].sum()) == 2                       # both, before the filter
    pred = _pred(12)
    assert box3d_set_loss(pred, t3, visible_filter=True)["n"]["z"] == 1
    assert box3d_set_loss(pred, t3, visible_filter=False)["n"]["z"] == 2


# =========================================================================== #
# D-2 -- the query budget spent half its slots behind the ego
# =========================================================================== #
def test_D2_the_budget_now_runs_over_the_IN_FIELD_set():
    """⭐ ANALYTIC, AND IT IS THE WHOLE MECHANISM IN FIVE BOXES.

    Two queries. Three targets: two BEHIND the ego at 5 m and 6 m, one AHEAD at 40 m.
    ``match_slots`` keeps the NEAREST ``n_queries`` by ``sqrt(cx^2+cy^2)``.

      * unfiltered -> {5 m behind, 6 m behind}; the 40 m car ahead is DROPPED, and the
        head receives ZERO in-field supervision on this frame;
      * filtered first -> the two behind-ego boxes are gone before the budget applies, so
        the 40 m car is the only target and it IS supervised.

    MEASURED at scale on v7-B1 (``raw/p6_query_budget_bias.json``): 49.145 % of 4,559,200
    query slots on over-budget frames went to ``cx < 0`` boxes, and 625,379 in-field boxes
    -- 27.18 % of what was available -- were destroyed by ordering alone.
    """
    tgt = _targets([(-5.0, 0.0, 4.5, 1.9, 0.0),      # 0: behind, 5 m
                    (-6.0, 0.0, 4.5, 1.9, 0.0),      # 1: behind, 6 m
                    (40.0, 0.0, 4.5, 1.9, 0.0)])     # 2: ahead, 40 m
    pred = _pred(2)

    off = box3d_set_loss(pred, tgt, visible_filter=False)
    assert off["n"]["target"] == 3 and off["n"]["dropped"] == 1
    assert off["n"]["matched"] == 2

    on = box3d_set_loss(pred, tgt, visible_filter=True)
    assert on["n"]["target_prefilter"] == 3
    assert on["n"]["target"] == 1                 # only the 40 m car survives the filter
    assert on["n"]["dropped"] == 0                # ...and it FITS the budget
    assert on["n"]["matched"] == 1

    from tanitad.refs.refc_agents import visible_target_filter
    # ⛔ WHICH targets, not how many. `cols` is in Hungarian assignment order, so it is
    # sorted here -- the SET is the claim, the permutation is the matcher's business.
    assert sorted(match_slots(pred,
                              visible_target_filter(tgt))["cols"][0].tolist()) == [2]
    assert sorted(match_slots(pred, tgt)["cols"][0].tolist()) == [0, 1]   # the defect


def test_D2_ordering_is_unchanged_when_everything_is_already_visible():
    """⛔ THE DISCRIMINATING CONTROL. If the fix also changed the budget POLICY, this
    would move -- and the finding would no longer be attributable to the filter. Three
    in-field targets at 10/20/30 m with two queries must still keep the nearest two,
    filter on or off, identically."""
    tgt = _targets([(10.0, 0.0, 4.5, 1.9, 0.0),
                    (20.0, 0.0, 4.5, 1.9, 0.0),
                    (30.0, 0.0, 4.5, 1.9, 0.0)])
    pred = _pred(2)
    on = box3d_set_loss(pred, tgt, visible_filter=True)
    off = box3d_set_loss(pred, tgt, visible_filter=False)
    assert on["n"]["target"] == off["n"]["target"] == 3
    assert on["n"]["dropped"] == off["n"]["dropped"] == 1
    assert torch.equal(on["total"].detach(), off["total"].detach())


# =========================================================================== #
# D-3 -- the MAP head's `seen` is a CLIP-LIFETIME mask
# =========================================================================== #
def _map_case(n_valid: int):
    """4x4 grid, 9 classes, every cell ``seen``; exactly ``n_valid`` cells lift-valid.

    The logits are PERFECT on the lift-valid cells and MAXIMALLY WRONG on the rest, so the
    two arms are separated by construction rather than by a threshold someone chose.
    """
    C, X, Y = 9, 4, 4
    frac = torch.zeros(1, C, X, Y)
    frac[:, 0] = 1.0                                   # label = class 0 everywhere
    logits = torch.full((1, C, X, Y), -10.0)
    logits[:, 1] = 10.0                                # wrong everywhere...
    seen = torch.ones(1, X, Y, dtype=torch.bool)
    valid4 = torch.zeros(1, 3, X, Y, dtype=torch.bool)   # [B, Z, X, Y], 3 heights
    flat = valid4.view(1, 3, -1)
    flat[0, 0, :n_valid] = True                        # ...except on the valid cells
    lv = PB.map_valid_from_lift(valid4)
    logits[:, 0][lv] = 10.0
    logits[:, 1][lv] = -10.0
    return logits, frac, seen, valid4, lv


def test_D3_the_lift_mask_narrows_the_supervised_cells_to_a_counted_number():
    """⭐ LITERALS: 16 cells seen, 6 lift-valid, 10 unobserved. Nothing derived."""
    logits, frac, seen, valid4, lv = _map_case(6)
    assert int(seen.sum()) == 16
    assert int(lv.sum()) == 6

    fixed = PB.map_loss_row(logits, frac, seen, lift_valid=lv, with_metrics=True)
    assert fixed["n_map_cells"] == 6.0
    assert fixed["n_map_cells_seen"] == 16.0
    assert fixed["n_map_cells_unobserved"] == 10.0

    # ⛔ THE DELIBERATE REGRESSION: the same call without the mask -- what every arm
    # before 2026-09-23 computed.
    broken = PB.map_loss_row(logits, frac, seen, with_metrics=True)
    assert broken["n_map_cells"] == 16.0
    assert broken["n_map_cells_unobserved"] == 0.0

    # the head is RIGHT on every cell it could have seen and WRONG on the other ten
    assert fixed["map_acc"] == 1.0
    assert broken["map_acc"] == pytest.approx(6.0 / 16.0, abs=1e-9)
    assert float(fixed["loss"]) < 1e-3
    assert float(broken["loss"]) > 1.0


def test_D3_the_mask_is_the_SAME_predicate_the_lift_substitutes_on():
    """⭐ NOT A SECOND SPELLING. ``BEVLift.forward`` adds its learned ``unobserved``
    embedding exactly where ``~valid.any(dim=1)``; the loss must drop exactly those cells
    or the head is asked to name a class from a constant. Checked against the module's own
    output, on a real ``BEVLift``, not against a re-derivation of the predicate."""
    from tanitad.models.bev_lift import BEVLift
    torch.manual_seed(0)
    lift = BEVLift(d_in=4, d_out=3, n_heights=2, feat_hw=(2, 3))
    B, Z, X, Y = 1, 2, 4, 5
    grid = torch.zeros(B, Z, X, Y, 2)
    valid = torch.zeros(B, Z, X, Y, dtype=torch.bool)
    valid[0, 0, 1, 1] = True
    valid[0, 1, 2, 3] = True
    out = lift(torch.zeros(B, 4, 2, 3), grid, valid)     # zero features -> proj bias only
    # a cell carries the `unobserved` vector iff the lift called it unobserved
    got = ~torch.isclose(out, lift.proj.bias.view(1, -1, 1, 1).expand_as(out),
                         atol=1e-6).all(dim=1)
    assert torch.equal(got, ~PB.map_valid_from_lift(valid))
    assert int(PB.map_valid_from_lift(valid).sum()) == 2


def test_D3_a_float_or_misshaped_lift_mask_is_REFUSED():
    logits, frac, seen, _, lv = _map_case(6)
    with pytest.raises(ValueError):
        PB.map_loss_row(logits, frac, seen, lift_valid=lv.float())
    with pytest.raises(ValueError):
        PB.map_loss_row(logits, frac, seen, lift_valid=lv[:, :2])
    with pytest.raises(ValueError):
        PB.map_valid_from_lift(lv)                        # [B,X,Y], not [B,Z,X,Y]


def test_D3_591_is_the_analytic_out_of_field_count_at_60_degrees():
    """⭐ THE REVIEW'S OWN ANALYTIC ARM, recomputed here from the grid rather than quoted.

    A cell whose azimuth exceeds 60 deg is outside the rig's only camera at EVERY instant.
    MEASURED 2026-09-22 (``raw/p4_map_gt_noncausal.json``): the analytic count over the
    120x64 grid is **590**, the census found **590**, and **90.088 %** of them carry a
    ``seen`` label. This reproduces the 590 from the grid spec -- and it is the number
    that makes ``seen`` a clip-lifetime mask rather than a camera one.
    """
    g = GRID_DEFAULT
    nx, ny = g.shape
    assert (nx, ny) == (120, 64)
    xs = (np.arange(nx) + 0.5) * float(g.cell_m)
    ys = (np.arange(ny) + 0.5 - ny / 2.0) * float(g.cell_m)
    az = np.abs(np.arctan2(np.abs(ys)[None, :], xs[:, None]))
    assert int((az > math.radians(60.0)).sum()) == 590
    assert nx * ny == 7680


# =========================================================================== #
# D-4 -- the class-weight guard, and the vector the fix invalidates
# =========================================================================== #
def test_D4_the_corpus_line_is_DERIVED_FROM_THE_JOIN_not_from_the_weight_key():
    """⛔ TWO INDEPENDENT SOURCES, WHICH IS THE ENTIRE FIX. MEASURED 2026-09-22
    (``raw/p5_cls_weight_guard.json``): with the expectation and the artifact selected by
    one key, ``guard_blocks_the_operator_error`` read **false** -- a B1 arm launched with
    ``--agent-cls-weight train2400`` loaded the PARITY vector and nothing refused."""
    assert A.corpus_line_for_join(
        "D:/x/a40-rescue/b1_train_plus_eval_agents.jsonl.xz") == A.CORPUS_LINE_B1
    assert A.corpus_line_for_join(
        "/data/joins/train2400_agents.jsonl.xz") == A.CORPUS_LINE_PARITY
    assert A.corpus_line_for_join("/data/joins/something_else.jsonl.xz") is None
    assert A.corpus_line_for_join(None) is None
    with pytest.raises(SystemExit):
        A.corpus_line_for_join("/data/joins/something_else.jsonl.xz", strict=True)


def test_D4_the_OPERATOR_ERROR_that_previously_loaded_is_now_REFUSED():
    """⛔ THE ARM THAT MEASURED ``LOADED`` IN P5, RERUN. A B1 arm asking for the
    train2400 vector: the expectation now comes from the join it actually trains."""
    b1_join = "D:/x/a40-rescue/b1_train_plus_eval_agents.jsonl.xz"
    with pytest.raises(SystemExit) as e:
        A.load_cls_class_weight(A.CLS_WEIGHTS_TRAIN2400,
                                expect_corpus_line=A.corpus_line_for_join(b1_join))
    assert "corpus line" in str(e.value)
    # ⭐ THE SAME-BREATH CONTROL: the CORRECT pairing must still load, or the "refusal"
    # is only a broken loader.
    vec, stamp = A.load_cls_class_weight(
        A.CLS_WEIGHTS_B1, expect_corpus_line=A.corpus_line_for_join(b1_join))
    assert stamp["corpus_line"] == A.CORPUS_LINE_B1
    # ...and the mirror-image error on a parity arm
    par_join = "/data/joins/train2400_agents.jsonl.xz"
    with pytest.raises(SystemExit):
        A.load_cls_class_weight(A.CLS_WEIGHTS_B1,
                                expect_corpus_line=A.corpus_line_for_join(par_join))


def test_D4_the_RAW_vector_is_UNMOVED_by_this_session():
    """⛔ BIT-IDENTITY FOR EVERY EXISTING ARM. The artifact gained a second vector and a
    declared population; the one that was already there must digest to the same 16 hex
    characters it did before, or a banked arm silently changed."""
    vec, stamp = A.load_cls_class_weight(A.CLS_WEIGHTS_B1,
                                         expect_corpus_line=A.CORPUS_LINE_B1)
    assert stamp["digest"] == B1_DIGEST_RAW
    assert stamp["target_population"] == A.TARGET_POPULATION_RAW
    assert float(stamp["imbalance_majority_to_rarest"]) == 1825.7
    assert stamp["counts"]["automobile"] == 21_515_941
    assert sum(stamp["counts"].values()) == B1_N_IN_VOCAB
    par, pstamp = A.load_cls_class_weight(A.CLS_WEIGHTS_TRAIN2400,
                                          expect_corpus_line=A.CORPUS_LINE_PARITY)
    assert A.cls_weight_digest(par) == "bde3aa19dfd0e59a"


def test_D4_the_VISIBLE_vector_is_the_measured_post_filter_census():
    """⛔ LITERALS FROM THE 2026-09-23 PRODUCER RUN over the whole v7-B1 join, which
    reproduced the reviewer's INDEPENDENT p6 counts class for class and reproduced the
    banked RAW digest exactly (``_control_reconstruction_max_abs_dev`` 4.96e-7).

    The per-class shift is the point: the ``cls`` term's frequencies are NOT the raw
    join's once the loss only sees what the camera sees."""
    vec, stamp = A.load_cls_class_weight(
        A.CLS_WEIGHTS_B1, expect_corpus_line=A.CORPUS_LINE_B1,
        target_population=A.TARGET_POPULATION_VISIBLE)
    assert stamp["digest"] == B1_DIGEST_VISIBLE
    assert stamp["target_population"] == A.TARGET_POPULATION_VISIBLE
    assert stamp["counts"]["automobile"] == 3_219_256
    assert stamp["counts"]["other_vehicle"] == 8_988
    assert stamp["counts"]["stroller"] == 11_054
    assert sum(stamp["counts"].values()) == B1_N_IN_VOCAB_VISIBLE
    assert float(stamp["imbalance_majority_to_rarest"]) == 1298.6
    assert abs(float(vec.mean()) - 1.0) < 1e-6
    # the shift, per class, against the raw vector -- LITERALS from the review's table
    raw, _ = A.load_cls_class_weight(A.CLS_WEIGHTS_B1,
                                     expect_corpus_line=A.CORPUS_LINE_B1)
    i = {c: k for k, c in enumerate(A.AGENT_CLASSES)}
    assert float(vec[i["other_vehicle"]] / raw[i["other_vehicle"]]) == \
        pytest.approx(1.746, abs=0.001)
    assert float(vec[i["heavy_truck"]] / raw[i["heavy_truck"]]) == \
        pytest.approx(1.729, abs=0.001)
    assert float(vec[i["stroller"]] / raw[i["stroller"]]) == \
        pytest.approx(0.663, abs=0.001)
    assert A.cls_weight_digest(vec) != A.cls_weight_digest(raw)


def test_D4_a_FILTERED_arm_cannot_load_an_artifact_with_no_visible_vector():
    """⛔ (1) AND (4) LAND TOGETHER OR NOT AT ALL. The parity artifact has no
    visible-population census (a NAMED blocker, declared in the file), so a parity arm
    that switches the filter on is REFUSED rather than served the raw vector."""
    with pytest.raises(SystemExit) as e:
        A.load_cls_class_weight(A.CLS_WEIGHTS_TRAIN2400,
                                expect_corpus_line=A.CORPUS_LINE_PARITY,
                                target_population=A.TARGET_POPULATION_VISIBLE)
    msg = str(e.value)
    assert "visible" in msg or "in_field_and_decode_box" in msg
    with pytest.raises(SystemExit):
        A.load_cls_class_weight(A.CLS_WEIGHTS_B1, target_population="whatever")


def test_D4_the_digests_are_COMPUTED_from_the_vectors_on_disk():
    """⛔ NOT A HAND-WRITTEN ATTESTATION. Both stated digests must reproduce from the two
    weight blocks by ``cls_weight_digest``, which is the ONE recipe and is what
    ``load_cls_class_weight`` verifies. The historical failure was a digest no code could
    recompute (`RETR-2026-09-22-SELF-ATTESTING-DIGEST`)."""
    import json
    import pathlib
    p = (pathlib.Path(A.__file__).resolve().parent.parent / "data" / A.CLS_WEIGHTS_B1)
    art = json.loads(p.read_text(encoding="utf-8"))
    for wk, dk in (("weights_inv_freq_mean1", "_self_digest_sha256_of_weights"),
                   ("weights_inv_freq_mean1_visible",
                    "_self_digest_sha256_of_weights_visible")):
        got = A.cls_weight_digest([art[wk][c] for c in A.AGENT_CLASSES], A.AGENT_CLASSES)
        assert got == art[dk], (wk, got, art[dk])
    assert art["_source"]["n_boxes_total"] == B1_N_BOXES
    assert art["_producer"].startswith("stack/scripts/build_cls_weight_artifact.py")
    assert art["_SMOKE"] is False
    assert art["_control_reconstruction_max_abs_dev_from_previous_raw"] < 1e-6


# =========================================================================== #
# the cheap extra -- map class weighting, with the identity control
# =========================================================================== #
def test_map_class_weight_at_ONES_is_BIT_IDENTICAL_to_passing_nothing():
    """⭐ THE IDENTITY CONTROL. The denominator follows the weights (``slot_set_loss``'s
    rule), so a uniform vector must reproduce the unweighted loss EXACTLY -- not
    approximately. If it did not, turning weighting on would move the term's scale against
    every sibling loss as well as its per-class emphasis: two changes in a one-variable
    arm."""
    torch.manual_seed(0)
    logits = torch.randn(2, 9, 5, 4, dtype=torch.float64)
    frac = torch.rand(2, 9, 5, 4, dtype=torch.float64)
    frac = frac / frac.sum(dim=1, keepdim=True)
    seen = torch.rand(2, 5, 4) > 0.3
    a = map_soft_ce(logits, frac, seen)
    b = map_soft_ce(logits, frac, seen, class_weight=torch.ones(9, dtype=torch.float64))
    assert float(a["loss"]) == pytest.approx(float(b["loss"]), rel=0, abs=1e-12)


def test_map_class_weight_matches_a_HAND_COMPUTED_weighted_cross_entropy():
    """⭐ AN INDEPENDENTLY AUTHORED REFERENCE. Two seen cells, three classes, every number
    written out and recombined in plain Python -- never by calling the function again."""
    lg = [[0.0, 1.0, 2.0], [1.0, -1.0, 0.5]]
    p = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    w = [3.0, 1.0, 0.5]
    logits = torch.tensor(lg, dtype=torch.float64).T.reshape(1, 3, 1, 2)
    frac = torch.tensor(p, dtype=torch.float64).T.reshape(1, 3, 1, 2)
    seen = torch.ones(1, 1, 2, dtype=torch.bool)

    num = den = 0.0
    for cell in range(2):
        z = lg[cell]
        lse = math.log(sum(math.exp(v) for v in z))
        for c in range(3):
            num += w[c] * p[cell][c] * (lse - z[c])
            den += w[c] * p[cell][c]
    expect = num / den

    got = float(map_soft_ce(logits, frac, seen,
                            class_weight=torch.tensor(w, dtype=torch.float64))["loss"])
    assert got == pytest.approx(expect, abs=1e-12)
    # ⛔ AND IT MUST DIFFER FROM THE UNWEIGHTED FORM, or the weight reached nothing
    assert got != pytest.approx(float(map_soft_ce(logits, frac, seen)["loss"]), abs=1e-6)


def test_map_class_weight_refuses_a_negative_or_misshaped_vector():
    logits = torch.zeros(1, 9, 2, 2)
    frac = torch.zeros(1, 9, 2, 2)
    frac[:, 0] = 1.0
    seen = torch.ones(1, 2, 2, dtype=torch.bool)
    with pytest.raises(ValueError):
        map_soft_ce(logits, frac, seen, class_weight=torch.ones(8))
    with pytest.raises(ValueError):
        map_soft_ce(logits, frac, seen, class_weight=torch.tensor([-1.0] + [1.0] * 8))


# =========================================================================== #
# the branch emits the fix's input, and nothing else moved
# =========================================================================== #
def test_the_branch_emits_map_valid_and_it_is_the_lift_predicate():
    from tanitad.models.trunk_shapes import TrunkSpec, frame_for_width
    cfg = PB.PerceptionBranchConfig(w_map=1.0, w_box3d=0.0, enforce_param_band=False)
    br = PB.PerceptionBranch(cfg, d_image=8, image_hw=(4, 8))
    B, Z = 2, len(cfg.heights_m)
    X, Y = GRID_DEFAULT.shape
    grid = torch.zeros(B, Z, X, Y, 2)
    valid = torch.zeros(B, Z, X, Y, dtype=torch.bool)
    valid[0, 0, :10] = True
    out = br(torch.randn(B, 8, 4, 8), grid, valid)
    assert "map_valid" in out
    assert out["map_valid"].dtype == torch.bool
    assert tuple(out["map_valid"].shape) == (B, X, Y)
    assert torch.equal(out["map_valid"], valid.any(dim=1))
    assert int(out["map_valid"].sum()) == 10 * Y
    assert int(out["map_valid"][1].sum()) == 0          # the second element saw nothing


def test_SLOT_WIDTHS_ARE_UNMOVED_BY_THESE_FIXES():
    """⛔ THE SAME-BREATH CONTROL FOR THE WHOLE FILE: none of these fixes touches the
    emitted channel layout, so a checkpoint's head columns still mean what they meant."""
    assert SLOT_WIDTH == 21
    assert SLOT3D_WIDTH == 23
    assert A.N_QUERIES_DEFAULT == 100
    assert len(A.AGENT_CLASSES) == 10
