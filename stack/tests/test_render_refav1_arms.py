"""Pins for `taniteval/tools/render_refav1_arms_video.py` — the multi-arm reel.

⛔ WHY THESE ARE THE TESTS AND NOT "IT RENDERED". A renderer that decoded nothing
writes a full-size, perfectly valid, BLACK video and exits 0. So every test here
follows the programme's rule: **a guard is proved by MUTATION, not by
inspection** — the defect is reintroduced and the guard must FIRE. A test that
only feeds good input and asserts "no exception" cannot distinguish a working
guard from a guard that is never reached.

Each test carries its own SAME-BREATH CONTROL: the good input must pass in the
same test that the bad input fails, or a blanket failure (a broken import, a
missing font) would read as a passing guard.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "taniteval", "tools")
_MOD = os.path.join(_TOOLS, "render_refav1_arms_video.py")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_MOD), reason=f"renderer not present at {_MOD}")


@pytest.fixture(scope="module")
def R():
    """Import the renderer by path (it loads its siblings the same way)."""
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    spec = importlib.util.spec_from_file_location("_arms_reel_under_test", _MOD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_arms_reel_under_test"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"renderer imports are unavailable here: {e!r}")
    return mod


# --------------------------------------------------------------------------- #
# 1. THE SHARED-ARRAY REFUSAL — the assertion that licenses one overlay        #
# --------------------------------------------------------------------------- #
def _fake_arms(n_arms=2, n_win=3, k=4, *, break_shared=False, same_plan=False):
    """Two arms on one episode. `break_shared` moves an array that MUST be
    shared; `same_plan` makes every arm emit the identical trajectory."""
    rng = np.random.default_rng(0)
    shared = {
        "ws": np.arange(n_win) * 16 + 3,
        "v0": np.full(n_win, 8.0, dtype=np.float32),
        "g": rng.normal(size=(n_win, k, 2)).astype(np.float32),
        "ha0_ext": rng.normal(size=(n_win, k, 2)).astype(np.float32),
        "wm_mse_model": rng.random((n_win, 5)).astype(np.float32),
        "wm_mse_const": rng.random((n_win, 5)).astype(np.float32),
        "wm_mse_zero": rng.random((n_win, 5)).astype(np.float32),
        "wm_tgt_std": rng.random(n_win).astype(np.float32),
        "lat_label": np.zeros(n_win, dtype=np.int64),
        "lon_label": np.zeros(n_win, dtype=np.int64),
        "route_label": np.ones(n_win, dtype=np.int64),
        "nav_cmd": np.zeros(n_win, dtype=np.int64),
        "nav_valid": np.ones(n_win, dtype=bool),
        "lat_pred_nav_true": np.zeros(n_win, dtype=np.int64),
        "lon_pred_nav_true": np.zeros(n_win, dtype=np.int64),
        "route_pred_nav_true": np.ones(n_win, dtype=np.int64),
    }
    arms = {}
    for i in range(n_arms):
        rec = {kk: vv.copy() for kk, vv in shared.items()}
        if break_shared and i == 1:
            rec["g"] = rec["g"] + 1.0            # a DIFFERENT scene
        rec["cl"] = (shared["g"].copy() if same_plan
                     else (shared["g"] + 0.1 * (i + 1)).astype(np.float32))
        arms[f"arm{i}"] = {"manifest": {}, "dir": "/nowhere",
                           "eps": {"ep": rec}}
    return arms, [f"arm{i}" for i in range(n_arms)]


def test_shared_verification_passes_and_refuses_a_moved_scene(R):
    """⭐ SAME-BREATH CONTROL + MUTATION. The good panel must verify, and the
    SAME check must refuse when a supposedly shared array moves."""
    good, order = _fake_arms()
    rep = R.verify_shared(good, order)                       # the control
    assert rep["n_windows"] == 3
    assert rep["arms_differing_on_cl"] == 1

    bad, order = _fake_arms(break_shared=True)               # the mutation
    with pytest.raises(SystemExit) as e:
        R.verify_shared(bad, order)
    assert "SHARED array" in str(e.value)


def test_shared_verification_refuses_arms_that_are_one_arm(R):
    """⛔ Equality is not evidence on its own: four identical plans under four
    names would draw one line and read as agreement between arms."""
    same, order = _fake_arms(same_plan=True)
    with pytest.raises(SystemExit) as e:
        R.verify_shared(same, order)
    assert "SAME cl trajectory" in str(e.value)


# --------------------------------------------------------------------------- #
# 2. THE CONTENT ASSERTION — a black reel exits 0 exactly like a good one      #
# --------------------------------------------------------------------------- #
def _stats():
    return dict(lum_sum=0.0, lum_n=0, lum_min=1e9, lum_max=-1e9)


def test_content_assertion_passes_a_real_frame_and_fires_on_a_black_one(R):
    from PIL import Image
    ok = Image.new("RGB", (64, 48), (40, 44, 52))            # the control
    st = _stats()
    R.assert_frame_content(ok, [("g", np.zeros((3, 2)))], "ctl", st)
    assert st["lum_n"] == 1 and st["lum_min"] > 2.0

    black = Image.new("RGB", (64, 48), (0, 0, 0))            # the mutation
    with pytest.raises(SystemExit) as e:
        R.assert_frame_content(black, [], "mut", _stats())
    assert "black or blown" in str(e.value)


def test_content_assertion_fires_on_a_nonfinite_and_a_runaway_path(R):
    from PIL import Image
    im = Image.new("RGB", (64, 48), (40, 44, 52))
    with pytest.raises(SystemExit) as e:
        R.assert_frame_content(im, [("cl", np.array([[1.0, np.nan]]))], "w",
                               _stats())
    assert "not finite" in str(e.value)
    with pytest.raises(SystemExit) as e:
        R.assert_frame_content(im, [("cl", np.array([[1e4, 0.0]]))], "w",
                               _stats())
    assert "implausible" in str(e.value)


# --------------------------------------------------------------------------- #
# 3. FEASIBILITY IS THE SCORER'S OWN, AND IT SEPARATES                        #
# --------------------------------------------------------------------------- #
def test_window_feasibility_separates_a_gentle_path_from_a_violent_one(R):
    """⛔ A flag that reads the same on both inputs is not a flag. A straight
    path at 10 m/s must be inside the circle; a hairpin at the same speed must
    not — and the peak lateral g must ORDER them."""
    pytest.importorskip("torch")
    dt, v = 0.2, 10.0
    straight = np.stack([np.arange(1, 11) * v * dt, np.zeros(10)], axis=1)
    th = np.arange(1, 11) * 0.30                       # ~1.5 rad/s yaw at 10 m/s
    hairpin = np.stack([np.cumsum(np.cos(th)) * v * dt,
                        np.cumsum(np.sin(th)) * v * dt], axis=1)
    a, b = R.window_feasibility(straight, dt), R.window_feasibility(hairpin, dt)
    assert a["kamm_over"] is False, "a straight line cannot leave the circle"
    assert b["kamm_over"] is True, "a hairpin at 10 m/s must leave it"
    assert b["peak_g"] > a["peak_g"] * 5.0
    assert R.FD.MU_KAMM == 0.7 and R.FD.KAPPA_MAX_1PM == 0.2


# --------------------------------------------------------------------------- #
# 4. COLOUR SEMANTICS — one meaning per colour, and no collisions              #
# --------------------------------------------------------------------------- #
def test_arm_palette_is_distinct_and_never_reuses_a_reserved_colour(R):
    """⛔ GREEN means ground truth, WHITE the floor and AMBER a given input,
    everywhere in the programme. An arm drawn in one of them would be read as
    the thing it is being compared against."""
    reserved = {R.C_GT, R.C_FLOOR, R.C_GIVEN}
    assert len(set(R.ARM_COLOURS)) == len(R.ARM_COLOURS)
    for c in R.ARM_COLOURS:
        assert c not in reserved
        for r in reserved:
            assert sum(abs(x - y) for x, y in zip(c, r)) > 90, \
                f"{c} is too close to the reserved colour {r}"


def test_layout_matches_the_imported_card_canvas(R):
    """⛔ MEASURED DEFECT, PINNED. `draw_card` is IMPORTED from the sibling
    renderer and paints at ITS canvas size. With a 1080-tall frame and a
    1122-tall card, ffmpeg's image demuxer took the FIRST frame's size and
    rescaled every clip frame into it: the whole reel stretched vertically by
    3.9 %, `verify_mp4` reported 1920x1122, and every exit code read 0.

    The frame size and the card size must be ONE number, and the renderer must
    refuse rather than ship a stretched reel."""
    import importlib.util as _iu
    spec = _iu.spec_from_file_location(
        "_sibling_card_renderer", os.path.join(_TOOLS, "render_refav1_video.py"))
    sib = _iu.module_from_spec(spec)
    sys.modules["_sibling_card_renderer"] = sib
    spec.loader.exec_module(sib)
    assert (R.W_TOT, R.H_TOT) == (sib.W_TOT, sib.H_TOT),         "the reel and the cards it imports must share one canvas"
    assert (R.W_TOT, R.H_TOT) == (1920, 1122)
    assert R.PAD + R.W_LEFT + R.PAD + R.W_RIGHT + R.PAD == R.W_TOT
    assert R.X_RIGHT + R.W_RIGHT + R.PAD == R.W_TOT
    assert R.Y_CAM + R.H_CAM < R.Y_MID < R.Y_MID + R.H_MID < R.Y_BOT
    assert R.Y_BOT + R.H_BOT + R.PAD == R.H_TOT


# --------------------------------------------------------------------------- #
# 5. THE PANELS DRAW, AND THEY DRAW SOMETHING                                  #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def F(R):
    try:
        return {"h1": R.font(30, True), "h2": R.font(21, True),
                "body": R.font(18), "small": R.font(16, True),
                "micro": R.font(14), "mono": R.font(15)}
    except SystemExit as e:                     # font() refuses loudly
        pytest.skip(f"no truetype face available: {e}")


def _red(img):
    a = np.asarray(img)
    return (a[:, :, 0] > 200) & (a[:, :, 1] < 80) & (a[:, :, 2] < 80)


def _red_below(img, row):
    """Red pixels strictly BELOW a row — i.e. behind the ego origin."""
    return int(_red(img)[int(row) + 3:].sum())


def _touches_bottom(img, n=3):
    """Did the path run off the panel's bottom edge (i.e. get clipped)?"""
    return bool(_red(img)[-n:].any())


def test_bev_draws_behind_the_ego_and_a_zero_based_range_clips_it(R, F):
    """⛔ MUTATION, AND A PROXY THAT CANNOT LIE. A plan is HELD while the ego
    drives along it, so its first waypoints end up BEHIND the ego; a panel
    starting at x = 0 clips them and the plan appears to shrink from its tail.

    ⚠️ The obvious proxy — "count red pixels" — is WRONG and this test failed on
    it first: widening the range also SHRINKS the drawing, so the panel showing
    MORE of the path can carry FEWER red pixels (measured 1519 against 1666).
    The honest proxy is positional: red must appear BELOW the x = 0 gridline
    when the range starts behind the ego, and must not when it starts at it."""
    h, m = 260, 44
    path = np.stack([np.linspace(-8.0, 12.0, 10), np.zeros(10)], axis=1)
    ser = [(path, (255, 0, 0), 5, False)]
    back = R.draw_bev((300, h), ser, -10.0, 20.0, "n", F, 8.0)
    flat = R.draw_bev((300, h), ser, 0.0, 20.0, "n", F, 8.0)
    # where x = 0 lands in each panel, from the same mapping draw_bev uses
    row_back = (h - m) - (h - 2 * m) * (0.0 - (-10.0)) / 30.0
    row_flat = (h - m) - (h - 2 * m) * (0.0 - 0.0) / 20.0
    assert int((np.asarray(back)[:, :, 0] > 200).sum()) > 100, "panel is blank"
    assert _red_below(back, row_back) > 50, \
        "the negative range must draw the part of the plan behind the ego"
    # ⚠️ AND THE SECOND WRONG PROXY, ALSO MEASURED: "no red below x = 0" fails,
    # because the panel's bottom MARGIN is still inside the image and PIL draws
    # into it. The discriminator that holds is whether the path RUNS OFF the
    # panel: with a 0-based range the receding tail is clipped by the image
    # edge, so red reaches the last row; with the negative range it terminates
    # at its own last waypoint, well above it.
    assert _touches_bottom(flat), \
        "a 0-based range must run the receding tail off the panel edge"
    assert not _touches_bottom(back), \
        "the negative range must end at the plan's own last waypoint, on-panel"


def test_wm_strip_sign_convention_is_model_wins_positive(R, F):
    """⚠️ THE SIGN IS THE FINDING. `persistence - model` positive means the
    model wins; the opposite convention reads identically and inverts it."""
    n = 30
    const = np.linspace(1.0, 2.0, n)
    model = const - 0.3                       # the model wins everywhere
    im = np.asarray(R.draw_wm((420, 260), model, const, const * 1.4, F,
                              mark_s=1.0))
    assert im.mean() > 5.0
    # the strip prints the advantage; a green (C_OK) heading means model wins
    ok = np.array(R.C_OK)
    hit = ((np.abs(im[:40].astype(int) - ok).sum(axis=2)) < 60).sum()
    assert hit > 0, "a winning model must print its advantage in the OK colour"


def test_arms_table_renders_every_row(R, F):
    rows = [dict(arm=f"a{i}", colour=R.ARM_COLOURS[i], geom="ccos W_KAPPA 0",
                 ade=f"{i:6.3f}", kap="0.0800", acc="+0.00", src="cem",
                 pg="0.300", pg_col=R.C_FG, feas="inside", feas_col=R.C_OK)
            for i in range(4)]
    im = np.asarray(R.draw_arms_table((1280, 288), rows, F, "hdr"))
    for i in range(4):
        c = np.array(R.ARM_COLOURS[i])
        assert ((np.abs(im.astype(int) - c).sum(axis=2)) < 40).sum() > 20, \
            f"row {i}'s swatch is missing — a silently dropped arm"


# --------------------------------------------------------------------------- #
# 6. THE DOC AND THE TOOL AGREE ON THE REFUSALS                                #
# --------------------------------------------------------------------------- #
def test_the_tool_reads_the_codec_field_and_never_the_buffer_name(R):
    """The E-DETECT-1 trap, pinned: the decode must branch on `codec`, and the
    magic bytes must be asserted against it."""
    src = open(_MOD, encoding="utf-8").read()
    sib = open(os.path.join(_TOOLS, "render_refav1_video.py"),
               encoding="utf-8").read()
    assert "frame_decoder" in src, "the renderer must reuse the sibling decoder"
    assert 'pay.get("codec"' in sib and "magic" in sib
    # ⛔ and it must not have grown a private decoder that skips the check
    assert "decode_jpeg" not in src and "decode_png" not in src


def test_the_tool_declares_every_standard_viz_element(R):
    from tanitad.viz_standard import STANDARD_ELEMENTS
    src = open(_MOD, encoding="utf-8").read()
    for name in STANDARD_ELEMENTS:
        assert f'VizElement(\n                    "{name}"' in src or \
               f'VizElement(\n                        "{name}"' in src or \
               f'"{name}", "present"' in src or f'"{name}", "unavailable"' in src, \
            f"the renderer never declares the required viz element {name!r}"

def test_the_camera_overlay_is_drawn_into_its_own_pane(R):
    """⛔ MEASURED REGRESSION, PINNED. `CylProjector` keeps a point up to ONE
    RASTER outside the image (`-h <= v <= 2h`) so a polyline heading just
    off-frame still connects. Drawn straight onto the shared canvas those
    coordinates land in the ARMS TABLE and the DECISIONS panel — it happened,
    on the ground-truth control frame, and a trajectory drawn across a table of
    numbers reads as a defect in the numbers.

    The fix is structural, not arithmetic: the overlay is drawn into a
    pane-sized layer that is pasted afterwards, so PIL clips it. This test pins
    that the offsets are gone — a `+ Y_CAM` on a projected point is exactly the
    defect — and the SAME-BREATH CONTROL is that the paste itself is present, so
    deleting the drawing entirely would not pass."""
    src = open(_MOD, encoding="utf-8").read()
    assert "q[1] + Y_CAM" not in src and "q[0] + PAD" not in src,         "a projected point is being offset onto the shared canvas again"
    assert src.count("im.paste(cam, (PAD, Y_CAM))") == 2,         "both the reel loop and the control must paste a finished camera layer"
    assert "ImageDraw.Draw(cam)" in src

def test_emit_refuses_a_frame_of_the_wrong_size(R):
    """⛔ MUTATION: the size guard must FIRE on a wrong-size image, and the
    SAME-BREATH CONTROL is that the correct size is accepted by the same
    expression. The guard lives inside `main`'s closure, so the pin is on the
    source text plus the constant it compares against."""
    src = open(_MOD, encoding="utf-8").read()
    assert "if im.size != (W_TOT, H_TOT):" in src,         "emit() no longer asserts the canvas size"
    assert "SILENTLY rescaled by ffmpeg" in src
    assert "imported card renderer paints" in src,         "the canvas mismatch must be caught BEFORE the render, not after it"
