"""The camera-field audit of `bev_raster`'s OTHER consumers (2026-08-16).

The P8 v6 port fixed `train_p8_occupancy.py` and explicitly deferred
`lf0_bev_lead.py` and `p8_bev_reel.py`. These tests pin what that audit
MEASURED, so the findings cannot rot the way a docstring does:

  * LF0 scores a CORRIDOR, not the grid — its exposure is ~10x smaller than the
    grid-wide number, and at the frame it actually ran on it is ZERO;
  * `--min-row 2` (added to skip the ego's own footprint, for an unrelated
    reason) is exactly the guard that buys that zero — on a LEGACY pinhole
    frame the same default still leaves 8 corridor cells unanswerable;
  * masking never changes an already-in-field read, and DOES change an
    out-of-field one (otherwise the mask would be decorative);
  * the reel's shading is opt-out and the unshaded path stays byte-identical.

CPU-only: pure geometry and pure numpy compositing. No corpus, no checkpoint,
no GPU, no join file.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_STACK = os.path.dirname(_HERE)
for _p in (_STACK, os.path.join(_STACK, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data.bev_raster import (GRID_DEFAULT, fov_mask,  # noqa: E402
                                     fov_row_floor)
from lf0_bev_lead import (CORRIDOR_M, HEADLINE_CORRIDOR,  # noqa: E402
                          corridor_cols, corridor_fov_census, read_lead_range)
import p8_bev_reel as reel  # noqa: E402

#: The three real frames, as half-angles in radians. Values are the MEASURED
#: ones from `raw/bev_consumer_geometry.json` (which derives the sub-frame
#: through `calib.centred_subframe`, never by typing "117").
HALF = {
    "v6f_cyl120": math.radians(60.0),
    "v5f_sub_cyl117": math.radians(58.5),
    "legacy_pinhole": math.radians(25.697),
}
LF0_MIN_ROW_DEFAULT = 2


def _cols(w):
    return corridor_cols(GRID_DEFAULT.shape[1], GRID_DEFAULT.y_half_m,
                         GRID_DEFAULT.cell_m, w)


def _census(half, min_row=LF0_MIN_ROW_DEFAULT):
    return corridor_fov_census(fov_mask(GRID_DEFAULT, half),
                               {w: _cols(w) for w in CORRIDOR_M},
                               GRID_DEFAULT.cell_m, min_row)


# ---------------------------------------------------------------- LF0 ------ #
@pytest.mark.parametrize("frame", ["v6f_cyl120", "v5f_sub_cyl117"])
@pytest.mark.parametrize("w", CORRIDOR_M)
def test_lf0_corridor_is_fully_in_field_on_every_cylindrical_frame(frame, w):
    """⭐ THE HEADLINE: the banked LF0 verdict CANNOT move.

    `lf0_chain.sh:93-94` ran it at `--frame-h 256 --frame-w 640 --frame-hfov
    120 --projection cylindrical --v2-subframe 176x624`, i.e. the 117 deg
    sub-frame, with the default `--min-row 2`. Every cell it scanned was inside
    the camera's horizontal field, so the masked and unmasked reads are
    identical BY CONSTRUCTION.
    """
    c = _census(HALF[frame])[str(w)]
    assert c["out_of_fov_cells"] == 0
    assert c["read_can_change"] is False


def test_lf0_headline_corridor_scans_708_cells():
    """A pinned denominator: 6 columns x 118 rows (120 - min_row 2)."""
    c = _census(HALF["v5f_sub_cyl117"])[str(HEADLINE_CORRIDOR)]
    assert c["n_cols"] == 6
    assert c["scanned_cells"] == 708


def test_the_min_row_guard_is_what_buys_the_zero_not_luck():
    """`--min-row 2` exists to skip the ego's own footprint — an unrelated
    reason. Without it the same corridor on the same frame is NOT clean, so the
    zero above is a property of the CONFIGURATION, not of the geometry alone."""
    with_guard = _census(HALF["v5f_sub_cyl117"], min_row=2)
    without = _census(HALF["v5f_sub_cyl117"], min_row=0)
    assert with_guard[str(HEADLINE_CORRIDOR)]["out_of_fov_cells"] == 0
    assert without[str(HEADLINE_CORRIDOR)]["out_of_fov_cells"] == 6
    assert without[str(HEADLINE_CORRIDOR)]["read_can_change"] is True


def test_legacy_pinhole_frame_would_NOT_be_clean():
    """⛔ The forward-looking hazard. On the legacy square frame the SAME LF0
    default leaves 8 corridor cells unanswerable, out to x = 2.25 m — and
    because the reader returns the NEAREST hit, a spurious cell there reads as
    'lead at 1.25 m', which is the worst possible failure for a gap probe."""
    c = _census(HALF["legacy_pinhole"])[str(HEADLINE_CORRIDOR)]
    assert c["out_of_fov_cells"] == 8
    assert c["read_can_change"] is True
    assert c["out_of_fov_max_x_m"] == pytest.approx(2.25)


def test_corridor_exposure_is_an_order_of_magnitude_below_the_grid():
    """⛔ THE DENOMINATOR RULE. Quoting the grid-wide fraction against LF0
    overstates its exposure by >10x — the mirror of the error this audit
    exists to catch."""
    half = HALF["v5f_sub_cyl117"]
    grid_frac = float((~fov_mask(GRID_DEFAULT, half)).mean())
    corridor_frac = _census(half, min_row=0)[str(HEADLINE_CORRIDOR)][
        "out_of_fov_frac"]
    assert grid_frac == pytest.approx(626 / 7680, abs=1e-9)
    assert corridor_frac < grid_frac / 9.0


def test_fov_row_floor_agrees_with_the_corridor_census():
    """The helper and the census must not be two implementations of one fact."""
    for name, half in HALF.items():
        for w in CORRIDOR_M:
            floor = fov_row_floor(GRID_DEFAULT, half, _cols(w))
            c = corridor_fov_census(fov_mask(GRID_DEFAULT, half),
                                    {w: _cols(w)}, GRID_DEFAULT.cell_m,
                                    floor)[str(w)]
            assert c["out_of_fov_cells"] == 0, (name, w, floor)


def test_fov_row_floor_defaults_to_the_whole_width():
    """With no `cols` it must reproduce fov_census's first_fully_visible_row."""
    assert fov_row_floor(GRID_DEFAULT, HALF["v6f_cyl120"]) == 18
    assert fov_row_floor(GRID_DEFAULT, HALF["legacy_pinhole"]) == 65


def test_fov_row_floor_is_none_when_no_row_is_ever_clean():
    """A 1 deg field never covers a +-16 m grid; the honest answer is None, not
    a row index nobody can use."""
    assert fov_row_floor(GRID_DEFAULT, math.radians(0.5)) is None


# ------------------------------------------------- read_lead_range mask ---- #
def _raster_with_hit(row, col):
    r = np.zeros(GRID_DEFAULT.shape, dtype=np.float32)
    r[row, col] = 1.0
    return r


def test_mask_none_is_byte_identical_to_the_incumbent():
    cols = _cols(HEADLINE_CORRIDOR)
    r = _raster_with_hit(40, int(cols[2]))
    kw = dict(tau=0.7, cols=cols, cell_m=GRID_DEFAULT.cell_m, min_row=2)
    assert read_lead_range(r, **kw) == pytest.approx(20.25)
    assert read_lead_range(r, fov=None, **kw) == pytest.approx(20.25)


def test_masking_does_NOT_change_an_in_field_read():
    """The mask must be inert where it has no cells to remove — otherwise the
    'the banked verdict cannot move' claim above is not safe."""
    cols = _cols(HEADLINE_CORRIDOR)
    r = _raster_with_hit(40, int(cols[2]))
    kw = dict(tau=0.7, cols=cols, cell_m=GRID_DEFAULT.cell_m, min_row=2)
    for half in HALF.values():
        assert read_lead_range(r, fov=fov_mask(GRID_DEFAULT, half),
                               **kw) == pytest.approx(20.25)


def test_masking_DOES_change_an_out_of_field_read():
    """...and it must not be decorative either. On the legacy frame a hit at
    row 2 / |y| = 1.25 m is outside the camera; unmasked it reads 1.25 m
    ('lead almost touching'), masked it is correctly censored to NaN."""
    cols = _cols(HEADLINE_CORRIDOR)
    r = _raster_with_hit(2, int(cols[0]))               # y = -1.25 m, x = 1.25
    kw = dict(tau=0.7, cols=cols, cell_m=GRID_DEFAULT.cell_m, min_row=2)
    assert read_lead_range(r, **kw) == pytest.approx(1.25)
    masked = read_lead_range(r, fov=fov_mask(GRID_DEFAULT,
                                             HALF["legacy_pinhole"]), **kw)
    assert math.isnan(masked)


def test_masking_a_near_false_positive_recovers_the_true_far_lead():
    """The reader returns the NEAREST hit, so one out-of-field cell does not add
    noise — it SHORTENS the read. Masking must restore the real lead."""
    cols = _cols(HEADLINE_CORRIDOR)
    r = _raster_with_hit(2, int(cols[0]))
    r[40, int(cols[2])] = 1.0                           # the true lead, 20.25 m
    kw = dict(tau=0.7, cols=cols, cell_m=GRID_DEFAULT.cell_m, min_row=2)
    assert read_lead_range(r, **kw) == pytest.approx(1.25)
    assert read_lead_range(r, fov=fov_mask(GRID_DEFAULT,
                                           HALF["legacy_pinhole"]),
                           **kw) == pytest.approx(20.25)


def test_read_lead_range_refuses_a_wrong_shaped_mask():
    """A broadcastable-but-wrong mask would silently score a grid nobody
    specified — the same refusal class as `assert_raster_shape`."""
    cols = _cols(HEADLINE_CORRIDOR)
    r = _raster_with_hit(40, int(cols[2]))
    with pytest.raises(ValueError, match="refusing to mask the wrong cells"):
        read_lead_range(r, tau=0.7, cols=cols, cell_m=GRID_DEFAULT.cell_m,
                        fov=np.ones((GRID_DEFAULT.shape[0], 1), dtype=bool))


def test_census_reports_the_deepest_unanswerable_row_not_just_a_count():
    """A count alone cannot tell a reader whether the exposure is in the band
    that matters; the max-x is what makes it actionable."""
    c = _census(HALF["legacy_pinhole"], min_row=0)[str(HEADLINE_CORRIDOR)]
    assert c["out_of_fov_max_x_m"] == pytest.approx(2.25)
    clean = _census(HALF["v6f_cyl120"])[str(HEADLINE_CORRIDOR)]
    assert clean["out_of_fov_max_x_m"] is None


# --------------------------------------------------------------- reel ------ #
@pytest.fixture()
def occ():
    a = np.zeros(GRID_DEFAULT.shape, dtype=np.float32)
    a[30:34, 30:34] = 1.0
    return a


def test_reel_panes_are_byte_identical_without_a_mask(occ):
    """`fov=None` must reproduce every banked still exactly."""
    assert np.array_equal(reel.raster_to_rgb(occ, reel.BELIEF),
                          reel.raster_to_rgb(occ, reel.BELIEF, fov=None))
    assert np.array_equal(reel.overlay_rgb(occ, occ),
                          reel.overlay_rgb(occ, occ, fov=None))


def test_reel_shades_the_cells_outside_the_field(occ):
    """8.151 % of every pane was drawn as scene; it must now be visibly not."""
    m = fov_mask(GRID_DEFAULT, HALF["v5f_sub_cyl117"])
    img = reel.raster_to_rgb(occ, reel.BELIEF, fov=m)
    shaded = np.flipud(img)[~m]
    assert (shaded == np.array(reel.NOFOV, dtype=np.uint8)).all(axis=-1).any()
    assert int((~m).sum()) == 626


def test_reel_marks_belief_on_an_unobservable_cell_in_its_own_colour():
    """A decoder firing where no camera looked is not a detection, and must not
    share a colour with one."""
    m = fov_mask(GRID_DEFAULT, HALF["legacy_pinhole"])
    a = np.zeros(GRID_DEFAULT.shape, dtype=np.float32)
    bad = np.argwhere(~m)[0]
    a[bad[0], bad[1]] = 1.0
    img = np.flipud(reel.raster_to_rgb(a, reel.BELIEF, fov=m))
    assert tuple(img[bad[0], bad[1]]) == reel.NOFOV_HIT
    assert tuple(img[bad[0], bad[1]]) != reel.BELIEF


def test_reel_overlay_keeps_ground_truth_visible_outside_the_field():
    """⚠️ A labelled agent outside the camera is a REAL agent. The shading says
    the belief had no evidence there; it must not delete the truth."""
    m = fov_mask(GRID_DEFAULT, HALF["legacy_pinhole"])
    bad = np.argwhere(~m)[0]
    gt = np.zeros(GRID_DEFAULT.shape, dtype=np.float32)
    gt[bad[0], bad[1]] = 1.0
    img = np.flipud(reel.overlay_rgb(np.zeros_like(gt), gt, fov=m))
    assert tuple(img[bad[0], bad[1]]) == reel.TRUTH


def test_reel_iou_pair_matches_the_incumbent_inline_computation(occ):
    """The extracted helper must not move the captioned number."""
    gt = np.zeros(GRID_DEFAULT.shape, dtype=np.float32)
    gt[32:36, 32:36] = 1.0
    inter = float(((occ > .5) & (gt > .5)).sum())
    union = float(((occ > .5) | (gt > .5)).sum())
    assert reel.iou_pair(occ, gt) == pytest.approx(inter / union)


def test_reel_iou_pair_empty_union_is_nan_not_zero(occ):
    """An empty union is 'not computable', never 'perfectly wrong'."""
    z = np.zeros(GRID_DEFAULT.shape, dtype=np.float32)
    assert math.isnan(reel.iou_pair(z, z))


def test_reel_in_field_iou_can_differ_from_the_all_cells_iou():
    """If it could not, reporting both would be theatre."""
    m = fov_mask(GRID_DEFAULT, HALF["legacy_pinhole"])
    bad = np.argwhere(~m)[0]
    pred = np.zeros(GRID_DEFAULT.shape, dtype=np.float32)
    pred[bad[0], bad[1]] = 1.0
    gt = np.zeros(GRID_DEFAULT.shape, dtype=np.float32)
    gt[bad[0], bad[1]] = 1.0
    assert reel.iou_pair(pred, gt) == pytest.approx(1.0)
    assert math.isnan(reel.iou_pair(pred, gt, fov=m))


def test_reel_compose_frame_accepts_a_mask_and_keeps_its_shape(occ):
    m = fov_mask(GRID_DEFAULT, HALF["v6f_cyl120"])
    cam = np.zeros((64, 128, 3), dtype=np.uint8)
    plain = reel.compose_frame(cam, occ, occ, "c", "s")
    shaded = reel.compose_frame(cam, occ, occ, "c", "s", fov=m)
    assert plain.shape == shaded.shape
    assert not np.array_equal(plain, shaded)


# ------------------------------------ Paper/figures/make_lf0_bev_panels ---- #
def _figmod():
    """Load the paper-figure generator (it lives outside `stack/`)."""
    import importlib.util
    p = os.path.join(os.path.dirname(_STACK), "Paper", "figures",
                     "make_lf0_bev_panels.py")
    if not os.path.exists(p):
        pytest.skip(f"partial checkout — {p} absent (pods get stack/ only)")
    spec = importlib.util.spec_from_file_location("_mlbp", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_figure_imports_the_grid_instead_of_typing_it():
    """⛔ THE THIRD CONSUMER a two-file audit would have missed — and the one
    with a PUBLISHED artifact (Paper/figures/lf0_bev_panels.svg). It restated
    `NX, NY = 120, 64` inline beside a live GRID_DEFAULT; a geometry fact
    restated inline is the same rot class as the 'N of 36' count."""
    m = _figmod()
    assert (m.NX, m.NY) == GRID_DEFAULT.shape == (120, 64)


def test_figure_refuses_to_guess_a_frame_it_was_not_given():
    """⚠️ The published render recorded NO frame. Shading a guessed one would be
    worse than not shading: 117° is 8.151 % of the grid, the legacy 51.4°
    pinhole is 27.682 %, so a wrong guess shades 3x too much and reads as a
    measurement."""
    assert _figmod().nofov_spans(None) == ()


@pytest.mark.parametrize("hfov,cells", [(117.0, 626), (120.0, 590),
                                        (51.393999, 2126)])
def test_figure_shading_covers_exactly_the_out_of_field_cells(hfov, cells):
    """The span rectangles must tile the mask exactly — no over- or under-draw."""
    spans = _figmod().nofov_spans(hfov)
    assert sum((r1 - r0 + 1) * (c1 - c0 + 1)
               for r0, r1, c0, c1 in spans) == cells


def test_figure_main_never_writes_outside_its_output_dir(tmp_path):
    """⛔ REGRESSION GUARD, from a real mistake made during this audit: calling
    `main()` with synthetic data OVERWROTE the published, git-tracked SVG. It
    was recovered byte-identically only because it was committed. `LF0_FIG_OUT`
    now redirects it, and this test both exercises the renderer end to end and
    pins that the redirect works."""
    m = _figmod()
    src = tmp_path / "panels_compact.json"
    src.write_text(json.dumps({
        "cols": list(range(29, 35)), "cell_m": 0.5, "true_m": [18.25],
        "win": [7], "hfov_deg": 117.0,
        "panels": [{"gt_hits": [[36, 31]], "gt_n": 33,
                    "enc_hits": [[80, 10]], "enc_n": 40,
                    "pred_hits": [[90, 55]], "pred_n": 68}]}), encoding="utf-8")
    published = os.path.join(os.path.dirname(_STACK), "Paper", "figures",
                             "lf0_bev_panels.svg")
    before = (open(published, "rb").read() if os.path.exists(published)
              else None)
    old = {k: os.environ.get(k) for k in ("LF0_PANELS", "LF0_FIG_OUT")}
    try:
        os.environ["LF0_PANELS"] = str(src)
        os.environ["LF0_FIG_OUT"] = str(tmp_path)
        assert m.main() == 0
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    assert (tmp_path / "lf0_bev_panels.svg").exists()
    if before is not None:                       # the published one is untouched
        assert open(published, "rb").read() == before
    svg = (tmp_path / "lf0_bev_panels.svg").read_text(encoding="utf-8")
    assert m.C_NOFOV in svg, "the frame was given; the shading must appear"


def test_figure_caveats_the_missing_frame_when_none_is_recorded(tmp_path):
    """With no frame the render must SAY the counts include unshaded
    out-of-field cells, rather than quietly looking complete."""
    m = _figmod()
    src = tmp_path / "panels_compact.json"
    src.write_text(json.dumps({
        "cols": list(range(29, 35)), "cell_m": 0.5, "true_m": [18.25],
        "win": [7],
        "panels": [{"gt_hits": [[36, 31]], "gt_n": 33, "enc_hits": [[80, 10]],
                    "enc_n": 40, "pred_hits": [[90, 55]], "pred_n": 68}]}),
        encoding="utf-8")
    old = {k: os.environ.get(k) for k in ("LF0_PANELS", "LF0_FIG_OUT")}
    try:
        os.environ["LF0_PANELS"] = str(src)
        os.environ["LF0_FIG_OUT"] = str(tmp_path)
        assert m.main() == 0
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    svg = (tmp_path / "lf0_bev_panels.svg").read_text(encoding="utf-8")
    assert "records NO camera frame" in svg
    assert m.C_NOFOV not in svg, "nothing may be shaded without a frame"


# ------------------------------------------------- the banked-frame fix ---- #
def test_the_banked_p8_frame_is_recoverable_from_the_launch_chains():
    """⛔ P8 port escalation #2: `p8_gate_attempt2.json` records NO frame, so
    its out-of-field fraction was called unrecoverable. It is recoverable — not
    from a pod-side `train_log.jsonl` (that host is gone) but from the launch
    chains banked IN THIS REPO, which name the exact flags. This test pins the
    provenance so the recovery cannot be lost again."""
    root = os.path.dirname(_STACK)
    chain = os.path.join(root, "TanitAD Research Hub", "Architecture & "
                         "Inference", "Implementation", "incoming",
                         "2026-08-11-ops-bundle", "p8c_chain.sh")
    assert os.path.exists(chain), chain
    txt = open(chain, encoding="utf-8").read()
    assert "--frame-h 256 --frame-w 640 --frame-hfov 120" in txt
    assert "--projection cylindrical" in txt
    assert "--v2-subframe 176x624" in txt
    assert "flagship-v5f-w120-30k/ckpt_30k_final.pt" in txt
    assert "--out /workspace/experiments/p8-occupancy-c" in txt
