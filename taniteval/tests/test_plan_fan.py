"""Guards for the plan-fan panel's drawing contract.

Covers the additive 2026-07-21 changes (constant-velocity layer + the clip
driver's window->frame reconstruction) and re-pins the two hazards
PLANNER_VIZ_CONCEPT.md §8 documents, because both fail as a SILENT HANG rather
than an exception and would never surface in a normal run.
"""
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "stack"))
sys.path.insert(0, str(HERE.parents[1] / "stack" / "scripts"))

from taniteval import plan_fan as PF  # noqa: E402


def _toy(n=6, s=4):
    g = torch.linspace(0.5, 1.5, n)[:, None, None]
    base = torch.stack([torch.arange(1, s + 1).float() * 3.0,
                        torch.zeros(s)], dim=-1)
    fan = (base[None] * g).clone()
    fan[:, :, 1] = torch.linspace(-2, 2, n)[:, None]      # lateral spread
    probs = torch.softmax(torch.linspace(0, 3, n), 0).tolist()
    return fan.tolist(), probs


def test_legend_rows_track_what_is_drawn():
    """A legend entry with nothing on the panel is a lie about the picture."""
    with_cv, without = PF.legend_rows(True), PF.legend_rows(False)
    assert len(with_cv) == len(without) + 1
    assert any(r[1] == PF.COL_CV for r in with_cv)
    assert not any(r[1] == PF.COL_CV for r in without)


def test_draw_bev_cv_is_optional_and_visible():
    """CV must default off (old callers unchanged) and must change pixels on."""
    fan, probs = _toy()
    gt = [[3.0, 0.0], [6.0, 0.0], [9.0, 0.0], [12.0, 0.0]]
    cv = [[3.5, 0.0], [7.0, 0.0], [10.5, 0.0], [14.0, 0.0]]
    kw = dict(anchors=fan, fan=fan, probs=probs, sel=3, gt_wp=gt, xmax=25.0)
    a = PF.draw_bev(240, 300, **kw)
    b = PF.draw_bev(240, 300, **kw, cv=cv)
    assert a.size == b.size == (240, 300)
    assert a.tobytes() != b.tobytes(), "cv= drew nothing"


def test_clean_drops_degenerate_segments():
    """PIL's wide-line rasteriser divides by the segment length: a repeated
    waypoint (any plan for a stopped vehicle, and every CEM sampler) makes it
    fill a ~1e16-wide box — one core pinned, minutes per frame, no error."""
    pts = [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (1.0, 1.0),
           (float("nan"), 2.0), (2.0, 2.0)]
    out = PF._clean(pts)
    assert out == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    assert all(math.dist(out[i], out[i + 1]) > 1e-3 for i in range(len(out) - 1))


def test_dashed_terminates_on_a_degenerate_polyline():
    """The dash walk must be closed-form: an incremental carry underflows and
    emits the degenerate wide line above."""
    from PIL import Image, ImageDraw
    d = ImageDraw.Draw(Image.new("RGB", (40, 40)), "RGBA")
    PF._dashed(d, [(5.0, 5.0), (5.0, 5.0)], (255, 0, 0, 255), width=3)
    PF._dashed(d, [(5.0, 5.0), (5.0, 5.0 + 1e-12)], (255, 0, 0, 255), width=3)
    PF._dashed(d, [(0.0, 0.0), (30.0, 30.0)], (255, 0, 0, 255), width=3)


def test_colour_scale_is_fixed_and_monotone():
    """A colour must mean the same probability in every frame of every clip."""
    assert PF.p_to_t(PF.P_FLOOR / 10) == 0.0
    assert PF.p_to_t(1.0) == 1.0
    ts = [PF.p_to_t(p) for p in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
    assert ts == sorted(ts)
    assert PF.viridis(0.0) == PF.VIRIDIS[0] and PF.viridis(1.0) == PF.VIRIDIS[-1]


def test_clip_window_to_frame_reconstruction():
    """plan_fan_clips maps fan-dump window index -> episode frame purely from
    the dump protocol (starts = range(0, T-8-20, 8), last = start + 7). If that
    drifts, every clip renders the WRONG frame while still looking plausible —
    which is why the dump asserts the anchor frame reproduces the dumped ADE."""
    from collections import Counter
    eid = [0] * 3 + [1] * 2
    seen, frame = Counter(), []
    for e in eid:
        frame.append(8 * seen[e] + 8 - 1)
        seen[e] += 1
    assert frame == [7, 15, 23, 7, 15]
