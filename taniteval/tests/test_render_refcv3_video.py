"""``tools/render_refcv3_video.py`` — the claim guards and the camera model, pinned.

⛔ WHY A VIDEO NEEDS TESTS AT ALL. An orange line tracking a green line down a real
road is the most over-claimable artefact this programme produces, and the two ways
this particular reel could lie are both silent:

1. **Drawing `a_star`.** It is the anchor nearest the GROUND TRUTH
   (``refc_v3_train.py:460``). It would look *better* than the deployed selection and
   nothing on screen would say so. The deployed path is ``out["traj"]``, the model's
   own ``sel_score_v3`` choice, and these tests pin that the drawn path comes from
   there and that the oracle is always labelled where it is drawn.
2. **Projecting with the pinhole formula.** The B1 cache is **cylindrical**; the
   pinhole formula produces a plausible-looking overlay that is wrong, and it is
   wrong by a large enough margin to move the drawn path off the lane at the image
   edge. :func:`test_the_projector_inverts_calibs_own_cylindrical_rays` round-trips
   the projector against ``tanitad.data.calib.cylindrical_rays`` — the forward model
   the cache was BUILT with — so a swapped formula fails numerically, not stylistically.
   :func:`test_a_pinhole_projector_would_be_caught` is the control that proves the
   round trip has the power to catch it.

⚠️ ``read_text(encoding="utf-8")`` everywhere. The sibling
``test_render_openloop_video.py`` reads without it and therefore FAILS on any cp1252
box the moment the tool it reads contains a non-Latin-1 glyph — which every tool in
this directory does. That is an environment-shaped false failure, and repeating it
here would make this file's result depend on the machine's locale.
"""
from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "render_refcv3_video.py"
HELPER = Path(__file__).resolve().parents[1] / "tools" / "pai_extrinsics_table.py"
DOC = Path(__file__).resolve().parents[1] / "tools" / "RENDER_REFCV3_VIDEO.md"

#: The B1 EVAL cache's OWN frame, as every `*.v2ep.pt` in it records it. Not a
#: constant this renderer chooses — it is read from the payload at render time, and
#: it is repeated here so a test failure names the geometry it was written against.
B1_FRAME = {"height": 256, "width": 640, "f_ref": 305.5774907364391,
            "projection": "cylindrical"}


def _src() -> str:
    return TOOL.read_text(encoding="utf-8")


def _mod():
    """Import the tool by path. Skips if the heavy chain (torch / the trainer /
    `refcv3_arm`) is unavailable, so the SOURCE-level guards below still run on a
    box that cannot import the stack."""
    try:
        spec = importlib.util.spec_from_file_location("_render_refcv3_video", TOOL)
        m = importlib.util.module_from_spec(spec)
        sys.modules["_render_refcv3_video"] = m
        spec.loader.exec_module(m)
        return m
    except Exception as ex:                                  # pragma: no cover
        pytest.skip(f"render_refcv3_video is not importable here: {ex!r}")


# --------------------------------------------------------------------------- #
# the tool exists, parses, and its README travels with it                      #
# --------------------------------------------------------------------------- #
def test_the_tool_and_its_helper_and_its_readme_all_exist_and_parse():
    assert TOOL.exists() and HELPER.exists() and DOC.exists()
    ast.parse(_src())
    ast.parse(HELPER.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# ⛔ the selection is the model's own, never the oracle                         #
# --------------------------------------------------------------------------- #
def test_the_drawn_path_is_out_traj_and_never_a_star():
    """``sel`` — the array every panel draws — is assigned from ``out["traj"]``
    exactly once, and no assignment to it mentions ``a_star`` or ``anchor_traj``."""
    tree = ast.parse(_src())
    assigns = [n for n in ast.walk(tree)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "sel" for t in n.targets)]
    assert assigns, "no assignment to `sel` — the drawn path must be a named array"
    srcs = [ast.unparse(n.value) for n in assigns]
    assert any("out['traj']" in s or 'out["traj"]' in s for s in srcs), srcs
    for s in srcs:
        assert "a_star" not in s, (
            f"the drawn path is assigned from {s!r}, which mentions a_star — the "
            f"GT-NEAREST anchor. That is an oracle and rendering it as the model's "
            f"choice is a fabricated result.")


def test_the_oracle_is_opt_in_and_is_labelled_wherever_it_is_drawn():
    s = _src()
    assert "--with-oracle" in s
    # the array actually drawn for the oracle is gated on the flag
    assert "oracle = fan[a_star] if a.with_oracle else None" in s
    # and the word ORACLE reaches the FRAME, not only this file's comments
    assert "a_star ORACLE" in s, "the selection strip must label the oracle column"
    assert "ORACLE ceiling, NOT the model's choice" in s, (
        "the legend drawn on every frame must say the violet path is not the "
        "model's choice")


def test_the_sidecar_says_what_was_drawn_and_what_it_is_not():
    s = _src()
    assert "what_is_drawn" in s and "what_this_is_not" in s
    assert "nav_is_an_oracle" in s, (
        "the nav token is an ego-future oracle that will not exist at deployment; "
        "the sidecar must say so beside any number taken from this render")


# --------------------------------------------------------------------------- #
# ⛔ the camera model                                                           #
# --------------------------------------------------------------------------- #
def _identity_extrinsic(height_m: float = 1.4):
    """A camera-basis extrinsic: camera +x right, +y down, +z forward, mounted at
    ``height_m`` above the vehicle origin with no pitch, roll or yaw.

    Built as the quaternion of R = [[0,0,1],[-1,0,0],[0,-1,0]] (columns = the camera
    axes expressed in the vehicle frame), so the test exercises the SAME
    quaternion -> R path the real extrinsics take."""
    R = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-6:                                    # not the case for this R
        raise AssertionError("degenerate quaternion branch")
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return {"qx": x, "qy": y, "qz": z, "qw": w,
            "x": 0.0, "y": 0.0, "z": float(height_m), "source": "test fixture"}


def test_an_unknown_projection_is_refused_not_approximated():
    m = _mod()
    with pytest.raises(m.ProjectionRefused):
        m.CylProjector({**B1_FRAME, "projection": "equirectangular"}, None)


def test_without_extrinsics_the_camera_overlay_is_disabled_and_says_so():
    m = _mod()
    p = m.CylProjector(B1_FRAME, None)
    assert p.enabled is False
    assert p([[10.0, 0.0]]) == []
    assert "DISABLED" in p.label()


def test_the_projector_inverts_calibs_own_cylindrical_rays():
    """⭐ THE ROUND TRIP, against the forward model the CACHE WAS BUILT WITH.

    ``calib.cylindrical_rays`` maps each output pixel to its camera-frame ray. This
    test walks that map forwards — pixel -> ray -> the ground point that ray hits —
    and requires the projector to return the ORIGINAL pixel. Nothing here re-states
    the projector's own formula, so a wrong formula cannot pass by agreeing with
    itself."""
    m = _mod()
    torch = pytest.importorskip("torch")
    from tanitad.data.calib import CanonicalFrame, cylindrical_rays

    frame = CanonicalFrame(height=B1_FRAME["height"], width=B1_FRAME["width"],
                           f_ref=B1_FRAME["f_ref"], projection="cylindrical")
    x, y, z = cylindrical_rays(frame)                       # [H, W] each
    extr = _identity_extrinsic(1.4)
    proj = m.CylProjector(B1_FRAME, extr)
    R, t = proj.R, proj.t

    checked = 0
    for v in (150, 170, 200, 230):                          # rows BELOW the horizon
        for u in (20, 160, 320, 480, 620):                  # incl. both edges
            d_cam = np.array([float(x[v, u]), float(y[v, u]), float(z[v, u])])
            d_veh = R @ d_cam
            assert d_veh[2] < 0, "a below-horizon ray must point DOWN in the vehicle frame"
            s = -t[2] / d_veh[2]
            p = t + s * d_veh                               # the ground intersection
            assert abs(p[2]) < 1e-9
            got = proj([[p[0], p[1]]], up=1)
            assert got and got[0] is not None, (u, v)
            assert got[0][0] == pytest.approx(u, abs=1e-4), (u, v, got[0])
            assert got[0][1] == pytest.approx(v, abs=1e-4), (u, v, got[0])
            checked += 1
    assert checked == 20


def test_a_pinhole_projector_would_be_caught():
    """⛔ THE CONTROL THAT GIVES THE ROUND TRIP ITS POWER.

    A round trip that a WRONG formula would also pass proves nothing. Here the
    pinhole column ``u = cx + f·x/z`` is computed on the same ground points and must
    differ from the cylindrical answer by **metres of apparent lateral offset** near
    the frame edge — so the previous test can actually fail if the formula is
    swapped. (It agrees at the image centre by construction: both projections fix the
    boresight, which is exactly why a centre-only eyeball check would miss this.)"""
    m = _mod()
    extr = _identity_extrinsic(1.4)
    cyl = m.CylProjector(B1_FRAME, extr)
    pin = m.CylProjector({**B1_FRAME, "projection": "pinhole"}, extr)
    ground = [[12.0, 0.0], [12.0, 3.0], [12.0, 9.0], [12.0, 16.0]]
    dc = cyl(ground, up=1)
    dp = pin(ground, up=1)
    du = [abs(a[0] - b[0]) for a, b in zip(dc, dp) if a and b]
    assert len(du) == 4
    assert du[0] < 1e-6, "the two projections must agree on the boresight"
    assert du[-1] > 40.0, (
        f"pinhole vs cylindrical differ by only {du[-1]:.2f} px at the edge — the "
        f"round-trip test would not catch a swapped formula")


def test_the_camera_height_is_per_clip_and_is_never_a_constant():
    """⚠️ MEASURED over 40 PhysicalAI clips: 1.245-1.607 m, 37 distinct values. The
    three constants circulating in this repo (1.22 / 1.43 / 1.5) are all wrong as a
    constant, and 1.22 is below the observed minimum. The renderer must take the
    height from the clip, never from a literal."""
    s = _src()
    assert "extr" in s and "--extrinsics" in s
    for bad in ("CAM_H = 1.22", "CAM_H = 1.43", "CAM_H = 1.5",
                "cam_h = 1.22", "cam_h = 1.43", "cam_h = 1.5"):
        assert bad not in s, f"{bad!r} is a hard-coded camera height"
    assert 'float(extr["z"])' in s, (
        "the camera height must be read from the clip's own extrinsic")


def test_the_horizon_is_drawn_so_the_frame_carries_its_own_falsifier():
    s = _src()
    assert "horizon predicted from the clip's" in s, (
        "the predicted horizon row is the only on-frame evidence that the "
        "projection is right; without it the overlay must be taken on trust")


# --------------------------------------------------------------------------- #
# ⛔ the panels                                                                 #
# --------------------------------------------------------------------------- #
def test_the_five_standard_viz_elements_are_all_declared():
    """`viz_standard.check_frame` REFUSES a frame missing any of the five, so this
    pins that the renderer actually calls it rather than merely importing it."""
    s = _src()
    assert "check_frame(els, where=" in s
    for el in ("camera", "bev", "tactical", "strategic", "ade",
               "strategic_input", "tactical_gt"):
        assert f'"{el}"' in s, el


def test_the_fed_nav_token_cannot_be_rendered_as_the_strategic_prediction():
    """⛔ THE NAV-ECHO DEFECT, UNCONSTRUCTIBLE. Flagship v1's route head scored
    1.0000 as an exact bijection of the nav it was fed. Putting the fed token in the
    `strategic` slot must RAISE, not merely look wrong."""
    from tanitad.viz_standard import VizElement, VizStandardError, check_frame
    ok = [
        VizElement.present("camera", "x", source="s", kind="derived"),
        VizElement.present("bev", "x", source="s", kind="model_output"),
        VizElement.present("tactical", "x", source="s", kind="model_output"),
        VizElement.present("strategic", "x", source="s", kind="model_output"),
        VizElement.present("ade", "0.1 m", source="s", kind="derived"),
    ]
    check_frame(ok, where="control")
    bad = list(ok)
    bad[3] = VizElement.present("strategic", "right",
                                source="the fed v7.2 nav_command",
                                kind="given_input")
    with pytest.raises(VizStandardError):
        check_frame(bad, where="nav-echo")


def test_the_tactical_factors_are_never_collapsed_into_one_row():
    """⛔ `COLLAPSE_TABLE` makes a turn absorb the longitudinal decision entirely, so
    a single 5-way chart cannot show a braking-into-a-turn error AT ALL. Two separate
    `draw_factor` calls, on the two separate heads."""
    tree = ast.parse(_src())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "draw_factor"]
    assert len(calls) == 2, f"expected exactly 2 factored rows, got {len(calls)}"
    args = " ".join(ast.unparse(c) for c in calls)
    assert "lat_p" in args and "lon_p" in args
    assert "maneuver_logits" not in _src(), (
        "the 5-way head must not be rendered here — it is the defect the factored "
        "head exists to remove")


def test_the_fan_and_the_reachability_mask_are_both_drawn():
    s = _src()
    assert 'out["anchor_traj"]' in s, "the fan is the model's output space"
    assert 'reach_keep' in s and "C_DEAD" in s, (
        "anchors killed by the reachability guard must be DRAWN in a distinct "
        "colour, not omitted — an excluded anchor is evidence about the guard")


def test_the_selection_profile_is_on_the_frame_because_degeneracy_is_the_failure():
    """REFCV3_ARM.md §4: a random-init model selected ONE anchor on 42/42 windows
    while the trivial profile read 0.0000. A constant selection is refcv3's
    characteristic degeneracy and the strip is what makes it visible per frame."""
    s = _src()
    assert "SELECTION SURFACE" in s
    assert "killed by reach_keep" in s


# --------------------------------------------------------------------------- #
# ⛔ the guards that stop an expensive render from being wasted or misread      #
# --------------------------------------------------------------------------- #
def test_the_encoder_is_resolved_before_any_frame_is_rendered():
    s = _src()
    assert s.index("Refusing to render frames that could not be encoded") < \
        s.index("canvas.save("), (
        "a missing ffmpeg found at the encode step throws away every rendered frame")


def test_out_refuses_a_directory_before_the_expensive_work():
    s = _src()
    assert s.index("--out must be a FILE") < s.index("arm.load_model")


def test_a_dropped_projection_point_is_never_bridged():
    """A polyline drawn THROUGH a point the projection refused is a line the model
    did not predict, crossing a region the projector said it could not express."""
    m = _mod()

    class _D:
        def __init__(self):
            self.runs = []

        def line(self, pts, **kw):
            self.runs.append(list(pts))

    d = _D()
    m.polyline(d, [(0, 0), (1, 1), None, (5, 5), (6, 6)], (1, 2, 3), 2)
    assert d.runs == [[(0, 0), (1, 1)], [(5, 5), (6, 6)]]
    d2 = _D()
    m.polyline(d2, [(0, 0), None, (5, 5)], (1, 2, 3), 2)
    assert d2.runs == [], "a one-point run is not a line"


def test_densify_prepends_the_ego_origin_and_keeps_the_endpoint():
    """The smooth camera curve is the model's EIGHT emitted slots interpolated; the
    frame says so and the slot markers are drawn on top. This pins that the
    interpolation neither invents an endpoint nor drops the origin."""
    m = _mod()
    path = np.array([[5.0, 0.0], [10.0, 1.0], [15.0, 3.0]])
    out = m.densify(path, n=33)
    assert out.shape == (33, 2)
    assert out[0] == pytest.approx([0.0, 0.0])
    assert out[-1] == pytest.approx(path[-1])


def test_the_font_refuses_rather_than_falling_back_to_the_bitmap_default():
    """PIL's default face is 11 px and is illegible at 1920 px, and it fails by
    looking merely ugly rather than by raising — which is how it ships."""
    s = _src()
    assert "load_default" not in s
    assert "refuses rather" in s or "refuses rather than" in s


def test_the_clip_selection_and_stride_are_written_into_the_sidecar():
    """⛔ A hand-picked reel must never be quotable as a representative one."""
    s = _src()
    assert "clip_selection" in s and "window_stride" in s


def test_expect_step_exists_because_the_final_checkpoint_has_no_unique_name():
    """⛔ MEASURED at source: `refc_v3_train.py:103` sets
    ``MILESTONES = (5000, 15000, 20000, 30000)`` and the only milestone write is
    ``ckpt_{step}.pt`` gated on ``step in MILESTONES`` (:1197-1199). There is no
    ``ckpt_40284_FINAL.pt`` and nothing will write one — the FINAL checkpoint is
    the ROLLING ``ckpt.pt`` (:1090/:1192), whose NAME never changes while its
    CONTENTS do. A reel rendered from a stale copy of it is indistinguishable
    from the real one, so the tool checks the step BY CONTENT and refuses BEFORE
    the GPU is spent."""
    tr = Path(__file__).resolve().parents[2] / "stack" / "scripts" / "refc_v3_train.py"
    if tr.exists():
        t = tr.read_text(encoding="utf-8")
        assert "MILESTONES = (5000, 15000, 20000, 30000)" in t
        assert "_FINAL.pt" not in t, (
            "the trainer now writes a *_FINAL.pt — update this test AND "
            "RENDER_REFCV3_VIDEO.md §6.3, which says it never does")
    s = _src()
    assert "--expect-step" in s
    assert "STEP MISMATCH" in s
    assert s.index("STEP MISMATCH") < s.index("canvas.save("), (
        "the step check must refuse BEFORE any frame is rendered")
    doc = DOC.read_text(encoding="utf-8")
    assert "ckpt.pt" in doc and "--expect-step" in doc, (
        "the README's re-render recipe must name the file that actually exists")


# --------------------------------------------------------------------------- #
# the nav caption: the WIRING and the BEHAVIOUR, and the step that bounds it   #
# --------------------------------------------------------------------------- #
def test_the_nav_caption_states_the_behaviour_not_only_the_wiring():
    """⛔ THE NAV-ECHO DEFECT, READ BACKWARDS. Saying only that "both heads are
    conditioned on the nav token (E13)" is true of the WIRING and credits an oracle
    input for a decision it does not drive: the route head is nav-INSENSITIVE BY
    CONSTRUCTION — ``route_logits = route_head(pooled)`` and ``pooled`` is the
    encoder's output over ``frames`` alone, while ``nav_cmd`` reaches only the
    SIBLING ``measurement(...)`` branch. Both facts go on the frame, or neither."""
    s = _src()
    assert "nav-INSENSITIVE BY CONSTRUCTION" in s, (
        "the drawn caption must say the route head does not use the token, not "
        "merely that it is wired to it")
    assert "reads pooled VISION" in s
    assert "nav_is_an_oracle" in s and "SIBLING" in s, (
        "the sidecar guard must carry the structural argument too")


def test_the_nav_caption_carries_the_step_the_number_was_measured_at():
    """⛔ A NUMBER FROM ANOTHER CHECKPOINT, PRINTED BARE, READS AS MEASURED HERE.
    ``paired_true_minus_shuffled_accuracy = 0.0 [0.0, 0.0]`` (paired episode-cluster
    bootstrap, n = 3 622 windows / 128 episodes) was measured at step **30 000**
    (``taniteval/results/refcv3-30k-openloop-20260903-2004.json``). This reel is step
    40 284. The step must travel with the number on the frame AND in the sidecar —
    the architectural argument is what carries across checkpoints, not the run."""
    s = _src()
    i = s.index("+0.0000 true−shuffled")
    assert "MEASURED @ step " in s[i:i + 200], (
        "the drawn caption prints +0.0000 without saying which step it was "
        "measured at — on a 40 284 frame that reads as measured here")
    assert "MEASURED AT STEP 30000" in s, "the sidecar must carry the same bound"
    assert "refcv3-30k-openloop-20260903-2004.json" in s, (
        "the sidecar must name the artifact the number came from")


def test_an_over_long_nav_caption_refuses_instead_of_being_trimmed():
    """⛔ MEASURED 2026-09-04: the rewritten caption wrapped to SEVEN lines under a
    ``[:6]`` slice and the line the slice ate was *"MEASURED @ step 30 000"* — the
    qualifier the rewrite existed to add. A silent truncation does not shorten a
    caption, it DELETES THE BOUND ON THE CLAIM, and the frame still looks finished.
    The cap must refuse, with the dropped line quoted."""
    s = _src()
    assert "NAV_CAP" in s
    tail = s[s.index("NAV_CAP = "):]
    assert "sys.exit(" in tail[:2000], (
        "the caption cap must refuse rather than slice; a bare [:N] on this "
        "string is how the bound disappears")
    assert "nav_lines[-1]!r" in tail[:2000], (
        "the refusal must QUOTE the line it would have dropped, or the operator "
        "cannot tell what the frame stopped saying")


def test_the_route_head_insensitivity_is_never_called_a_defect():
    """⚠️ THE PRIMARY ARTIFACT ARGUES THE REVERSE. On the 1 736 changed-nav windows
    the route head follows the LABEL at 0.7414 and the WRONG fed nav at 0.2224 —
    vision-derived route skill, the opposite of flagship v1's 1.0000 echo. And the
    TACTICAL heads DO move with nav, so the two families are never quoted as one."""
    s = _src()
    assert "0.7414" in s and "0.2224" in s, (
        "the sidecar must carry the changed-nav evidence that makes this a "
        "positive finding, not an indictment")
    assert "TACTICAL heads are a different" in s, (
        "route-head insensitivity must not be generalised to the tactical heads, "
        "which are separated on the longitudinal factor")
