"""Guards for the ALIGNMENT GATE in `render_quality.py` (P3 of the render re-baseline).

The failure being prevented is `R-2026-08-03-k`: for weeks every absolute grad-NCC /
MAE / PSNR on NuRec scene `00040136` was scored against a reference video **6 frames too
early**, the numbers looked plausible, and the harness's own negative control PASSED on
every frame — because ``wrong_frames_for()`` enforces ``MIN_WRONG_GAP = 40`` and is
therefore blind to a small index error *by construction*.

So these tests do not check that the gate passes.  They check that it **FAILS**, loudly,
on exactly the input that shipped:

* ``test_gate_fails_on_a_six_frame_misalignment`` — the real defect.
* ``test_gate_failure_message_names_the_corrected_offset`` — a gate that stops the run
  but does not say what to do next gets disabled by the next person under time pressure.
* ``test_gate_writes_its_evidence_even_when_it_fails`` — a refusal with no artifact is
  unciteable, and C10 in this programme is "the evaluator does not implement its own
  pre-registration".

Plus the index arithmetic the whole fix rests on: ``load_refs`` decodes video frame
``f + ref_offset`` and returns it under **rig** key ``f``.  If that mapping is wrong or
silently identity, everything above is decoration.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

_EXP = Path(__file__).resolve().parents[1] / "experiments" / "alpasim-gsplat"
sys.path.insert(0, str(_EXP))

import render_quality as rq  # noqa: E402


# --------------------------------------------------------------------------------- #
# a renderer stub: "render of rig frame f" is a token that matches ref token f        #
# --------------------------------------------------------------------------------- #
class _StubRenderer:
    """Renders frame f as the scalar f. `grad_ncc` below scores 1 - |a - b| / 100."""

    def __init__(self, truth_shift: int = 0):
        self.truth_shift = truth_shift
        self.n_render = 0

    def gt_cam_to_nre(self, f):
        return float(f)

    def frame_timestamps_us(self, f):
        return (0.0, float(f) * 33333.0)

    def render(self, c2n, actor_time_us=None, cam_to_nre_end=None):
        self.n_render += 1
        return (np.float32(c2n + self.truth_shift), None, 1.0)


_FLAT = 1000.0   # render tokens >= this score on a near-flat curve (see below)


def _fake_grad_ncc(a, b):
    """Peaked, symmetric, and maximal when the render token equals the ref token.

    A token >= `_FLAT` scores on the SAME peak position but with a slope 30x shallower,
    which reproduces the real phenomenon this gate has to survive: on a stationary
    segment every neighbouring reference frame is nearly identical, so the curve has an
    interior argmax whose prominence is below the noise floor.  MEASURED on `7c72937c`
    frame 60: the whole +-10 curve spans 0.3994-0.4041.
    """
    a = float(a)
    if a >= _FLAT:
        return float(0.40 - 0.001 * abs((a - _FLAT) - float(b)))
    return float(0.40 - 0.03 * abs(a - float(b)))


@pytest.fixture(autouse=True)
def _patch_metric(monkeypatch):
    monkeypatch.setattr(rq, "grad_ncc", _fake_grad_ncc)


def _refs_for(frames, k, residual=0):
    """refs[rig f] carries token (f + residual): residual 0 == correctly aligned."""
    return {f + d: np.float32(f + d + residual)
            for f in frames for d in range(-k - 1, k + 2)}


# --------------------------------------------------------------------------------- #
# the gate                                                                           #
# --------------------------------------------------------------------------------- #
def test_gate_passes_when_the_reference_is_aligned(tmp_path):
    frames = [0, 60, 150, 300, 450]
    res = rq.assert_reference_aligned(_StubRenderer(), _refs_for(frames, 3), frames,
                                      ref_offset=6, k=3, out_dir=tmp_path)
    assert res["pass"] is True
    assert set(res["residual_argmax_by_frame"].values()) == {0}
    assert res["residual_offset_bootstrap"]["point"] == 0
    assert (tmp_path / "alignment_gate.json").exists()


def test_gate_fails_on_a_six_frame_misalignment(tmp_path):
    """THE defect that shipped: reference index 6 frames too early."""
    frames = [0, 60, 150, 300, 450]
    with pytest.raises(SystemExit) as ei:
        rq.assert_reference_aligned(_StubRenderer(), _refs_for(frames, 3, residual=-6),
                                    frames, ref_offset=0, k=3, out_dir=tmp_path)
    assert "ALIGNMENT GATE FAILED" in str(ei.value)
    assert "R-2026-08-03-k" in str(ei.value)


@pytest.mark.parametrize("residual", [-3, -2, -1, 1, 2, 3])
def test_gate_fails_on_every_nonzero_residual_inside_the_window(tmp_path, residual):
    """One frame off is enough. The gate is not a tolerance band."""
    frames = [0, 60, 150]
    with pytest.raises(SystemExit):
        rq.assert_reference_aligned(_StubRenderer(),
                                    _refs_for(frames, 3, residual=residual),
                                    frames, ref_offset=0, k=3, out_dir=tmp_path)


def test_gate_failure_message_names_the_corrected_offset(tmp_path):
    """A gate that halts without saying what to run next gets disabled, not obeyed."""
    frames = [0, 60, 150]
    with pytest.raises(SystemExit) as ei:
        rq.assert_reference_aligned(_StubRenderer(), _refs_for(frames, 3, residual=-2),
                                    frames, ref_offset=4, k=3, out_dir=tmp_path)
    # applied +4, residual +2 => the correct offset is +6
    assert "+6" in str(ei.value)


def test_gate_writes_its_evidence_even_when_it_fails(tmp_path):
    import json
    frames = [0, 60, 150]
    with pytest.raises(SystemExit):
        rq.assert_reference_aligned(_StubRenderer(), _refs_for(frames, 3, residual=-2),
                                    frames, ref_offset=0, k=3, out_dir=tmp_path)
    g = json.loads((tmp_path / "alignment_gate.json").read_text())
    assert g["pass"] is False
    assert set(g["residual_argmax_by_frame"].values()) == {2}
    assert g["ref_offset_applied"] == 0
    assert g["per_frame"]["0"]["gain_vs_offset0"] > 0


def test_gate_refuses_to_name_a_correction_it_cannot_see(tmp_path):
    """A 6-frame residual is outside a +-3 scan. The gate must still FAIL, and must say
    it cannot name the correction rather than reporting its own window edge as the
    answer — that is the ">= +3" failure that produced this whole retraction."""
    frames = [0, 60, 150]
    with pytest.raises(SystemExit) as ei:
        rq.assert_reference_aligned(_StubRenderer(), _refs_for(frames, 3, residual=-6),
                                    frames, ref_offset=0, k=3, out_dir=tmp_path)
    msg = str(ei.value)
    assert "ALIGNMENT GATE FAILED" in msg
    assert "BEYOND" in msg and "--align-k" in msg
    assert "correct offset is" not in msg      # ⛔ never guesses from a boundary

    # widen the window and the same input becomes answerable
    with pytest.raises(SystemExit) as ei2:
        rq.assert_reference_aligned(_StubRenderer(), _refs_for(frames, 10, residual=-6),
                                    frames, ref_offset=0, k=10, out_dir=tmp_path)
    assert "correct offset is +6" in str(ei2.value)


def test_gate_skips_frames_whose_neighbourhood_is_truncated(tmp_path):
    """MEASURED false alarm: rig frame 0 has no `f-k` neighbours, so its argmax sits at a
    truncated edge, the adjudicator correctly refuses on `boundary`, and the gate read
    that as a misalignment at a CORRECT offset. The gate must skip it and say so."""
    frames = [0, 60, 150, 300, 450]
    refs = _refs_for(frames, 3)
    for d in (-3, -2, -1):                     # frame 0's left neighbours do not exist
        refs.pop(d, None)
    res = rq.assert_reference_aligned(_StubRenderer(), refs, frames, ref_offset=6, k=3,
                                      n_probe=3, out_dir=tmp_path)
    assert res["pass"] is True
    assert 0 not in res["probe_frames"]
    assert res["frames_skipped_incomplete_window"] == [0]


def test_gate_refuses_to_run_rather_than_silently_skip_itself(tmp_path):
    """If NO frame has a complete window the gate must halt, not pass vacuously."""
    frames = [0]
    refs = {0: np.float32(0), 1: np.float32(1)}
    with pytest.raises(SystemExit) as ei:
        rq.assert_reference_aligned(_StubRenderer(), refs, frames, ref_offset=6, k=3,
                                    out_dir=tmp_path)
    assert "CANNOT RUN" in str(ei.value)


def test_gate_ignores_an_uninformative_frame_instead_of_failing_on_it(tmp_path):
    """MEASURED false alarm: `7c72937c` frame 60 sits in a STATIONARY segment; its whole
    +-10 curve spans 0.3994-0.4041 and its argmax landed at -6 at a CORRECT offset. A
    flat curve carries no alignment information — it is uninformative, not misaligned,
    and a gate that fails on it is a gate that gets switched off."""
    frames = [60, 300, 450]
    refs = _refs_for(frames, 3)

    class _FlatOnOneFrame(_StubRenderer):
        def render(self, c2n, actor_time_us=None, cam_to_nre_end=None):
            # frame 60 scores on a near-flat curve: interior argmax, no prominence
            return (np.float32(c2n if c2n != 60 else _FLAT + c2n), None, 1.0)

    res = rq.assert_reference_aligned(_FlatOnOneFrame(), refs, frames, ref_offset=6,
                                      k=3, n_probe=3, out_dir=tmp_path)
    assert res["pass"] is True
    assert 60 not in res["informative_frames"]
    assert "60" in res["uninformative_frames"]


def test_gate_still_fails_a_genuine_off_by_one(tmp_path):
    """The tolerance introduced for sub-frame preference must NOT swallow a real
    one-frame index error: a true +1 moves the bootstrap mass wholesale."""
    frames = [60, 300, 450]
    with pytest.raises(SystemExit) as ei:
        rq.assert_reference_aligned(_StubRenderer(), _refs_for(frames, 3, residual=-1),
                                    frames, ref_offset=5, k=3, out_dir=tmp_path)
    assert "correct offset is +6" in str(ei.value)


def test_gate_cannot_certify_when_no_frame_is_informative(tmp_path):
    """'cannot certify' must be its own outcome, distinct from 'aligned'."""
    frames = [60, 300, 450]

    class _AllFlat(_StubRenderer):
        def render(self, c2n, actor_time_us=None, cam_to_nre_end=None):
            return (np.float32(_FLAT + c2n), None, 1.0)

    with pytest.raises(SystemExit) as ei:
        rq.assert_reference_aligned(_AllFlat(), _refs_for(frames, 3), frames,
                                    ref_offset=6, k=3, out_dir=tmp_path)
    assert "CANNOT CERTIFY" in str(ei.value)
    assert "NOT 'aligned'" in str(ei.value)


def test_gate_sees_what_the_min_wrong_gap_control_cannot(tmp_path):
    """The two controls answer different questions — this is the whole retraction.

    ``wrong_frames_for`` excludes everything within 40 frames, so a 6-frame error can
    never appear among its candidates.  Asserting that here keeps the reason the gate
    exists attached to the gate."""
    assert rq.MIN_WRONG_GAP == 40
    cands = rq.wrong_frames_for(150, 599)
    assert all(abs(c - 150) >= 40 for c in cands)
    assert not any(abs(c - 150) <= 10 for c in cands)   # the hard negatives are absent


# --------------------------------------------------------------------------------- #
# the index arithmetic the fix rests on                                              #
# --------------------------------------------------------------------------------- #
class _FakeCap:
    """Sequential mp4 whose frame i is a 1x1x3 image carrying the value i."""

    def __init__(self, n=700):
        self.i, self.n = 0, n

    def read(self):
        if self.i >= self.n:
            return False, None
        img = np.full((1, 1, 3), self.i % 256, np.uint8)
        self.i += 1
        return True, img

    def get(self, _):
        return float(self.n)

    def release(self):
        pass


def _install_fake_cv2(monkeypatch, n=700):
    m = types.ModuleType("cv2")
    m.VideoCapture = lambda _p: _FakeCap(n)
    m.CAP_PROP_FRAME_COUNT = 7
    m.INTER_AREA = 3
    m.resize = lambda img, wh, interpolation=None: img
    monkeypatch.setitem(sys.modules, "cv2", m)


def test_load_refs_maps_rig_index_to_video_index_plus_offset(monkeypatch):
    _install_fake_cv2(monkeypatch)
    refs = rq.load_refs("x.mp4", [0, 60, 150], (1, 1), ref_offset=6)
    assert sorted(refs) == [0, 60, 150]                     # keys stay in RIG space
    assert [int(refs[f][0, 0, 0]) for f in (0, 60, 150)] == [6, 66, 156]


def test_load_refs_offset_zero_is_the_identity(monkeypatch):
    """The default must not silently shift anything — 62 other call sites rely on it."""
    _install_fake_cv2(monkeypatch)
    refs = rq.load_refs("x.mp4", [0, 60, 150], (1, 1))
    assert [int(refs[f][0, 0, 0]) for f in (0, 60, 150)] == [0, 60, 150]


def test_load_refs_reports_a_frame_it_could_not_decode(monkeypatch):
    """Running off the end must be visible as a missing key, not a silent short dict
    that the caller then averages over."""
    _install_fake_cv2(monkeypatch, n=100)
    refs = rq.load_refs("x.mp4", [0, 60, 150], (1, 1), ref_offset=6)
    assert sorted(refs) == [0, 60]
    assert 150 not in refs
