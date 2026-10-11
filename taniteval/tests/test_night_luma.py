"""Tests for ``taniteval.night_luma`` (SPEC_REFCV8_DRAFT section 8.10 X10 NIGHT-LUMA).

Synthetic frames of KNOWN luma; every expected value is a literal worked out by hand
(a 90/0/0 frame is exactly 30.0 under the channel mean; a frame of half-60 / half-0
columns is exactly 30.0; and so on). Nothing here reads the corpus.

The deliberate-regression arms re-introduce one defect each and require the SAME
literal check to go RED:
  * the black bottom strip is averaged in      (a rig-B clip reads 40 -> 28 and flips to "night")
  * Rec.601 weights instead of the channel mean  (R-only 90 reads 26.91, not 30.0)
  * ``<=`` at the threshold                      (a clip at exactly 30.0 becomes "night")
  * the median across frames instead of the mean (0/20/100 reads 20, not 40)
  * D3's original strip rule on a black frame    (the whole frame is swallowed -> no rows left)

pytest is not installed on every pod; runs standalone too:
  python taniteval/tests/test_night_luma.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)              # <repo>/taniteval
if _PKG_PARENT not in sys.path:
    sys.path.append(_PKG_PARENT)

from taniteval import night_luma as nl   # noqa: E402


def _flat(v, h=10, w=8):
    return np.full((h, w, 3), v, dtype=np.uint8)


def _stripped(v, rows=7, strip=3, w=8):
    f = np.zeros((rows + strip, w, 3), dtype=np.uint8)
    f[:rows] = v
    return f


def _red_only(v=90, h=6, w=6):
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[..., 0] = v
    return f


def _half_columns(v, h=6, w=8):
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:, ::2, :] = v
    return f


def _check_literals():
    """The registered literals, one assertion per defect the mutations re-introduce."""
    # the channel mean: (90 + 0 + 0) / 3 = 30.0 exactly -> NOT night (strict <)
    assert nl.clip_mean_luma(_red_only(90)) == 30.0
    assert nl.is_night_luma(nl.clip_mean_luma(_red_only(90))) is False
    # half the columns at 60 -> exactly 30.0;  at 58 -> exactly 29.0 -> night
    assert nl.clip_mean_luma(_half_columns(60)) == 30.0
    assert nl.clip_mean_luma(_half_columns(58)) == 29.0
    assert nl.is_night_luma(nl.clip_mean_luma(_half_columns(58))) is True
    # the strip: 7 rows at 40 over 3 black rows -> 40.0 (day), not 40 * 7/10 = 28.0 (night)
    assert nl.bottom_strip_rows(_stripped(40)) == 3
    assert nl.clip_mean_luma(_stripped(40)[None]) == 40.0
    assert nl.is_night_luma(nl.clip_mean_luma(_stripped(40)[None])) is False
    # mean over frames of the per-frame mean: 0, 20, 100 -> 40.0 (a median would read 20.0)
    clip = np.stack([_flat(0), _flat(20), _flat(100)])
    assert nl.clip_mean_luma(clip) == 40.0
    # a fully black clip reads 0.0 and is NIGHT (D3's strip rule would swallow the whole frame)
    assert nl.bottom_strip_rows(_flat(0)) == 0
    assert nl.clip_mean_luma(np.stack([_flat(0)] * 3)) == 0.0
    assert nl.is_night_luma(0.0) is True
    # the strict threshold itself
    assert nl.is_night_luma(30.0) is False and nl.is_night_luma(29.999) is True
    return True


# --------------------------------------------------------------------------- #
# the definition                                                               #
# --------------------------------------------------------------------------- #
def test_registered_literals():
    assert _check_literals()
    assert nl.NIGHT_LUMA_THRESHOLD == 30.0


def test_channel_mean_is_the_registered_luma_and_rec601_is_not():
    f = _red_only(90)
    assert nl.frame_mean_luma(f) == 30.0
    # Rec.601: 0.299 * 90 = 26.91 -- a different convention that would flip this clip to "night"
    assert abs(nl.frame_mean_luma(f, weights=nl.REC601) - 26.91) < 1e-9
    assert nl.is_night_luma(nl.frame_mean_luma(f, weights=nl.REC601)) is True
    # grey is the same under both conventions: 0.299 + 0.587 + 0.114 = 1
    assert abs(nl.frame_mean_luma(_flat(50), weights=nl.REC601) - 50.0) < 1e-9
    assert nl.frame_mean_luma(_flat(50)) == 50.0
    try:
        nl.frame_mean_luma(f, weights="hsv")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown luma convention must be refused")


def test_bottom_strip_is_excluded_and_detected_on_frame_zero():
    s = _stripped(40)                                       # 7 rows of 40 over 3 zero rows
    assert nl.frame_mean_luma(s, strip_rows=0) == 28.0      # strip averaged in: reads dark
    assert nl.frame_mean_luma(s, strip_rows=3) == 40.0
    assert nl.clip_mean_luma(s[None], strip_rows="none") == 28.0
    assert nl.clip_mean_luma(s[None], strip_rows=3) == 40.0
    assert nl.clip_mean_luma(s[None]) == 40.0               # default "first"
    # only ALL-ZERO rows count: a bottom row of value 1 is image, not strip
    t = _stripped(40)
    t[-1] = 1
    assert nl.bottom_strip_rows(t) == 0
    # an interior zero row is not a bottom strip
    u = _flat(40)
    u[4] = 0
    assert nl.bottom_strip_rows(u) == 0
    # D3 detects on frame 0 and applies it to every frame; per_frame re-detects
    f0, f1 = _flat(40), _stripped(40)
    clip = np.stack([f0, f1])
    assert nl.clip_mean_luma(clip, strip_rows="first") == 34.0       # (40 + 28) / 2
    assert nl.clip_mean_luma(clip, strip_rows="per_frame") == 40.0   # (40 + 40) / 2
    for bad in ("bottom", 1.5):
        try:
            nl.clip_mean_luma(clip, strip_rows=bad)
        except ValueError:
            continue
        raise AssertionError(f"strip_rows={bad!r} must be refused")
    try:
        nl.frame_mean_luma(_flat(10, h=4), strip_rows=4)
    except ValueError:
        pass
    else:
        raise AssertionError("a strip that leaves no rows must be refused")


def test_clip_aggregation_is_the_mean_over_frames():
    clip = np.stack([_flat(0), _flat(20), _flat(100)])
    assert nl.clip_mean_luma(clip) == 40.0                  # median 20, min 0 -- neither
    assert nl.is_night_luma(nl.clip_mean_luma(clip)) is False
    one = nl.clip_mean_luma(_flat(77))                      # a single [H,W,3] frame is a 1-frame clip
    assert one == 77.0


def test_black_clip_is_the_darkest_not_a_nan():
    z = np.stack([_flat(0)] * 4)
    assert nl.clip_mean_luma(z) == 0.0
    assert nl.is_night_luma(nl.clip_mean_luma(z)) is True


def test_layouts_agree_and_refuse_bad_shapes():
    hwc = np.stack([_red_only(90), _red_only(90)])                      # [T,H,W,3]
    chw = np.moveaxis(hwc, -1, 1)                                       # [T,3,H,W]
    assert nl.clip_mean_luma(hwc) == 30.0
    assert nl.clip_mean_luma(chw, layout="CHW") == 30.0
    assert nl.clip_mean_luma(chw[0], layout="CHW") == 30.0              # one [3,H,W] frame
    for fn in (lambda: nl.clip_mean_luma(np.zeros((4, 4, 4, 4), np.uint8)),     # 4 channels
               lambda: nl.clip_mean_luma(np.zeros((0, 4, 4, 3), np.uint8)),     # no frames
               lambda: nl.clip_mean_luma(hwc, layout="WHC")):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("a malformed frame stack must be refused")


def test_d3_controls_read_their_known_values():
    """The D3 ``controls()`` through THIS function: all-zero -> 0 (dark), all-255 -> 255, a
    left-to-right gradient of known mean -> that mean."""
    z = np.zeros((416, 1024, 3), np.uint8)
    w = np.full((416, 1024, 3), 255, np.uint8)
    g = np.tile(np.linspace(0, 255, 1024).astype(np.uint8)[None, :, None], (416, 1, 3))
    assert nl.clip_mean_luma(z[None]) == 0.0 and nl.is_night_luma(0.0)
    assert nl.clip_mean_luma(w[None]) == 255.0 and not nl.is_night_luma(255.0)
    assert abs(nl.clip_mean_luma(g[None]) - 127.5) < 1.0
    # the bottom-strip control: 384 white rows over 32 zero rows
    sw = np.concatenate([w[:384], z[:32]], 0)
    assert nl.bottom_strip_rows(sw) == 32
    assert nl.clip_mean_luma(sw[None]) == 255.0


def test_is_night_luma_threshold_and_nan():
    assert nl.is_night_luma(12.6) is True                   # D3's darkest "day" clip
    assert nl.is_night_luma(91.3) is False                  # D3's brightest "night" clip
    assert nl.is_night_luma(45.0, threshold=50.0) is True
    for bad in (float("nan"), float("inf")):
        try:
            nl.is_night_luma(bad)
        except ValueError:
            continue
        raise AssertionError("a non-finite luma must not be labelled")


# --------------------------------------------------------------------------- #
# the census and the window mask (report-only)                                  #
# --------------------------------------------------------------------------- #
def test_census_literals():
    ids = ["c0", "c1", "c2", "c3", "c4"]
    luma = [10.0, 20.0, 40.0, 100.0, 200.0]
    clock = [True, False, True, False, True]
    r = nl.night_luma_census(ids, luma, clock)
    assert r["n_clips"] == 5 and r["n_dark"] == 2 and r["share_dark"] == 0.4, r
    assert r["n_clock_night"] == 3 and r["share_clock_night"] == 0.6, r
    assert r["crosstab_clips"] == {"clock_night_and_dark": 1, "clock_night_and_not_dark": 2,
                                   "clock_day_and_dark": 1, "clock_day_and_not_dark": 1}, r
    assert r["report_only"] is True and "< 30" in r["rule"], r
    assert "crosstab_clips" not in nl.night_luma_census(ids, luma)       # clock label optional
    for fn in (lambda: nl.night_luma_census(["a", "a"], [1.0, 2.0]),     # duplicate clip id
               lambda: nl.night_luma_census(["a", "b"], [1.0, float("nan")]),
               lambda: nl.night_luma_census(["a", "b"], [1.0]),
               lambda: nl.night_luma_census(["a", "b"], [1.0, 2.0], [True])):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("a malformed census must be refused")


def test_window_mask_carries_window_n_and_clip_n():
    wins = ["A", "A", "B", "C", "C", "C"]
    r = nl.stratum_mask(wins, {"A": 10.0, "B": 50.0, "C": 25.0})
    assert r["mask"].tolist() == [True, True, False, True, True, True], r
    assert r["n_windows"] == 5 and r["n_clips"] == 2 and r["clips"] == ["A", "C"], r
    assert r["n_windows_total"] == 6 and r["n_clips_total"] == 3, r
    try:
        nl.stratum_mask(["A", "Z"], {"A": 10.0})
    except ValueError as e:
        assert "no luma" in str(e)
    else:
        raise AssertionError("a window whose clip has no luma must raise, not read as day")


# --------------------------------------------------------------------------- #
# deliberate regressions                                                        #
# --------------------------------------------------------------------------- #
def _expect_red(patch):
    restore = patch(nl)
    try:
        try:
            _check_literals()
        except (AssertionError, ValueError):
            return
        raise AssertionError("the deliberate regression stayed GREEN -- the literals cannot see it")
    finally:
        restore()


def test_mutation_strip_averaged_in_goes_red():
    def patch(m):
        orig = m.bottom_strip_rows
        m.bottom_strip_rows = lambda frame: 0
        return lambda: setattr(m, "bottom_strip_rows", orig)
    _expect_red(patch)


def test_mutation_rec601_default_goes_red():
    def patch(m):
        orig = m._luma_plane
        m._luma_plane = lambda a, weights: orig(a, m.REC601)
        return lambda: setattr(m, "_luma_plane", orig)
    _expect_red(patch)


def test_mutation_le_at_threshold_goes_red():
    def patch(m):
        orig = m.is_night_luma
        m.is_night_luma = lambda mean_luma, threshold=30.0: bool(float(mean_luma) <= float(threshold))
        return lambda: setattr(m, "is_night_luma", orig)
    _expect_red(patch)


def test_mutation_median_across_frames_goes_red():
    def patch(m):
        orig = m.clip_mean_luma

        def med(frames, *, strip_rows="first", weights=m.RGB_MEAN, layout="HWC"):
            a = m._hwc(frames, layout)
            s = m.bottom_strip_rows(a[0]) if strip_rows == "first" else 0
            return float(np.median([m.frame_mean_luma(f[None], strip_rows=s, weights=weights) for f in a]))
        m.clip_mean_luma = med
        return lambda: setattr(m, "clip_mean_luma", orig)
    _expect_red(patch)


def test_mutation_d3_strip_rule_on_a_black_frame_goes_red():
    def patch(m):
        orig = m.bottom_strip_rows

        def d3(frame):
            a = np.asarray(frame)
            zero_row = a.reshape(a.shape[0], -1).max(axis=1) == 0
            n = 0
            for z in zero_row[::-1]:               # D3 strip_rows: no all-zero-frame special case
                if not z:
                    break
                n += 1
            return n
        m.bottom_strip_rows = d3
        return lambda: setattr(m, "bottom_strip_rows", orig)
    _expect_red(patch)


if __name__ == "__main__":                                  # pragma: no cover
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for k, v in fns:
        try:
            v()
            print("PASS", k)
        except Exception as e:
            bad += 1
            print("FAIL", k, type(e).__name__, e)
    sys.exit(1 if bad else 0)
