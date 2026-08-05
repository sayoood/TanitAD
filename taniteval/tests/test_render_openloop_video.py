"""``tools/render_openloop_video.py`` — the claim guards, pinned.

⛔ WHY A VIDEO NEEDS TESTS AT ALL. An orange line tracking a green line is the most
over-claimable artefact this programme produces: it looks like the car driving. It
is not. The rollout decodes the **expert's true future actions** — ``collect``'s own
PC2 record says ``actions_source="expert_future"``, ``pc2_pass`` False by
construction — so it is world-model fidelity, and open loop on top of that means the
ego never leaves the log. These tests pin that both facts reach the *frame*, not just
a README nobody re-reads beside the file.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "render_openloop_video.py"


def _src():
    return TOOL.read_text()


def test_the_tool_exists_and_parses():
    assert TOOL.exists()
    ast.parse(_src())


def test_every_frame_carries_the_fidelity_and_open_loop_labels():
    """⛔ The banner is drawn per frame, so a clip cut out of the reel keeps the
    caveat. A README beside the mp4 does not travel with the mp4."""
    s = _src()
    banner = s[s.index('"OPEN LOOP — ego follows'):]
    banner = banner[:400]
    assert "EXPERT'S TRUE FUTURE ACTIONS" in banner
    assert "NOT autonomous driving" in banner
    assert "NOT a hierarchy result" in banner


def test_the_selection_is_named_in_the_banner_not_only_the_sidecar():
    """A cherry-picked reel must not be quotable as a representative one."""
    s = _src()
    assert "{select}" in s, "the banner must interpolate the selection mode"
    assert '"⛔_not_a_selection_claim"' in s


def test_there_is_no_silent_selection_fallback():
    """An earlier version offered --select worst/best and silently rendered
    'spread' for both — a reel labelled 'worst' that was nothing of the kind."""
    tree = ast.parse(_src())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_argument"):
            continue
        args = [a for a in node.args if isinstance(a, ast.Constant)]
        if args and args[0].value == "--select":
            kw = {k.arg: k.value for k in node.keywords}
            choices = [c.value for c in kw["choices"].elts]
            assert set(choices) == {"spread", "first"}, (
                f"--select offers {choices}; every one of them must be "
                f"IMPLEMENTED. Ranked reels go through --episode-list so the "
                f"ranking's provenance is explicit.")
            return
    pytest.fail("--select not found")


def test_explicit_episode_list_is_range_checked():
    s = _src()
    assert "out-of-range indices" in s, (
        "an out-of-range index would silently shift the reel onto other clips")


def test_the_encoder_is_resolved_before_any_frame_is_rendered():
    """MEASURED on pod4: ffmpeg is absent and apt has no package lists. Finding
    that at the encode step throws away every rendered frame."""
    s = _src()
    assert s.index("ffmpeg = _ffmpeg()") < s.index("for rank, ei in enumerate(idx)")


def test_the_bev_is_declared_calibration_independent():
    """The camera panel depends on a projection model; the BEV does not. That is
    why it is a full panel here and not a 152 px inset — if they disagree, the
    projection is wrong and the BEV is the one to believe."""
    s = _src()
    assert "calibration-independent" in s


def test_out_refuses_a_directory_before_the_expensive_work():
    s = _src()
    assert s.index("os.path.isdir(a.out)") < s.index("loaders.load(")


def test_frames_come_from_RawEp_feats_not_a_frames_attribute():
    """``data.RawEp`` exposes the stack as ``.feats``; ``.frames`` raises. Caught
    at runtime once — pinned so it is caught at test time next."""
    from taniteval.data import RawEp
    assert hasattr(RawEp, "__init__")
    assert "ep.feats[t, -3:]" in _src()
    assert "ep.frames[t" not in _src()


# --- the drawing primitives, on synthetic paths ----------------------------- #

def _fonts():
    from taniteval.flagship_overlay import _font
    return {"tiny": _font(10), "sub": _font(12)}


def _cols():
    from taniteval.flagship_overlay import COL_GT, COL_PRED, HUD_DIM
    return {"gt": COL_GT, "pred": COL_PRED, "dim": HUD_DIM}


def _tool_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_rov", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bev_puts_a_left_turn_on_the_LEFT():
    """A sign flip here would draw every turn mirrored and the video would look
    plausible — the failure mode a shape test cannot catch."""
    m = _tool_module()
    gt = np.stack([np.linspace(1, 20, 20), np.linspace(0, 8, 20)], 1)   # +y = left
    im = m.draw_bev_large((420, 442), gt, gt, 20.0, 10.0, None, _cols(), _fonts())
    a = np.asarray(im)
    g = np.asarray(_cols()["gt"])
    hit = (np.abs(a.astype(int) - g).sum(-1) < 40)
    xs = np.nonzero(hit.any(0))[0]
    assert xs.size, "the GT path was not drawn at all"
    # the path ends far left of centre; its drawn extent must reach left of centre
    assert xs.min() < 420 // 2, "a +y (left) path was drawn to the RIGHT of centre"


def test_bev_renders_without_a_past_track():
    m = _tool_module()
    gt = np.stack([np.linspace(1, 20, 20), np.zeros(20)], 1)
    m.draw_bev_large((420, 442), gt, gt, 20.0, 10.0, None, _cols(), _fonts())


def test_trace_handles_an_empty_history():
    """Frame 0 of every episode has no history; a crash there kills the reel."""
    m = _tool_module()
    m.draw_trace((420, 60), [], _cols(), _fonts())


def test_cli_help_works_without_a_gpu():
    r = subprocess.run([sys.executable, str(TOOL), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert "--episode-list" in r.stdout and "--corpus" in r.stdout
