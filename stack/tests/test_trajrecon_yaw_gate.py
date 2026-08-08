"""The lane-yaw credibility gate must not judge a measurement against a default.

WHY THIS EXISTS
---------------
MEASURED 2026-08-08 on the `14-19-54` recording. `lane_calib` measured a mount
yaw of **-7.01 deg** with a half-split spread of **0.20 deg** — its own stability
gate is 0.6 deg, so the measurement was comfortably stable — and the pipeline
threw it away with:

    yaw declined: -7.01 deg is 7.0 deg from the FOE, not credible

There was no FOE. `camera.py`'s FOE fit had already failed
("FOE fit produced no usable flow - kept nominal"), which leaves `cam.yaw` at the
**nominal 0.0**. The gate at `lane_calib.py:243` compares against `cam.yaw`, so
with no FOE it degenerates into *"reject any mount yaw beyond 4 deg of dead
ahead"* — and it rejects **hardest exactly when the mount is most crooked**,
which is precisely when the correction is worth having.

The cost was not subtle. An independent VP fit over 154 straight frames / 6689
segments gives -6.05 deg [95% CI -6.28, -5.83], confirming the pipeline's own
number. Shipping 0.0 instead put **4.9 m of lateral error at 40 m** into the
rendered overlay — outside the ego lane from ~15 m onward.

Three layers made it quiet, and the third is the one to remember:

  1. the gate rejected a good measurement against a placeholder;
  2. the warning *called* the placeholder "the FOE", so the log looked like a
     real disagreement between two estimates;
  3. `lane_calib` returns `yaw_deg if yaw_ok else None`, so `pipeline.py:494`'s
     `if res.yaw_deg is not None` was False and the one diagnostic that spelled
     out the damage — `"... (-7.01 deg, 4.89 m at 40 m)"` — **never printed.**
     The louder the error, the quieter the log.

These are source-level checks on purpose: they need neither the trajrecon extras
nor torch, so they run everywhere the rest of the suite runs.
"""

from __future__ import annotations

import pathlib
import re

PKG = pathlib.Path(__file__).resolve().parents[1] / "tanitad" / "data" / "trajrecon"
PIPELINE = (PKG / "pipeline.py").read_text(encoding="utf-8")
CAMERA = (PKG / "camera.py").read_text(encoding="utf-8")
LANE = (PKG / "lane_calib.py").read_text(encoding="utf-8")


def test_caller_passes_the_gate_threshold_rather_than_taking_the_default():
    """Upstream let `max_yaw_correction_deg` default to 4.0 unconditionally."""
    assert "max_yaw_correction_deg=" in PIPELINE, (
        "pipeline.py no longer sets max_yaw_correction_deg — the lane-VP yaw is "
        "back to being gated against cam.yaw even when no FOE was measured")


def test_the_threshold_is_conditioned_on_whether_a_foe_actually_exists():
    """The whole point: the gate is only meaningful with a real FOE to agree with."""
    assert re.search(r'foe_measured\s*=\s*"extrinsics"\s+in\s+cam\.source', PIPELINE), (
        "the FOE-measured test is gone; without it the caller cannot know "
        "whether cam.yaw is a measurement or a nominal placeholder")
    assert re.search(r"max_yaw_correction_deg\s*=\s*\(?\s*4\.0\s+if\s+foe_measured\s+else\s+15\.0",
                     PIPELINE), (
        "the conditional threshold is gone — with no FOE the bound must be a "
        "plausibility limit on mount yaw, not an agreement limit against 0.0")


def test_the_foe_measured_key_still_means_what_we_think():
    """Guard the detection itself.

    `"extrinsics" in cam.source` is only a valid "the FOE succeeded" test while
    `camera.py` writes that key **exclusively** in the success branch. If upstream
    ever sets it on a failure path too, the fix above silently stops working —
    so pin the invariant rather than trusting it.
    """
    success = re.findall(r'source\["extrinsics"\]\s*=', CAMERA)
    failures = re.findall(r'source\["extrinsics_note"\]\s*=', CAMERA)
    assert len(success) == 1, (
        f'camera.py writes source["extrinsics"] {len(success)} times; the '
        f'FOE-measured test assumes exactly one (the success branch)')
    assert len(failures) >= 3, (
        f'camera.py writes source["extrinsics_note"] only {len(failures)} times; '
        f'the nominal-fallback paths appear to have changed')


def test_the_plausibility_bound_matches_cameras_own_constant():
    """15 deg is not arbitrary — camera.py:405 uses it to sanity-check the FOE.

    Two different numbers for "a dashcam mount is never more crooked than this"
    would drift apart; this asserts they are still the same number.
    """
    assert "np.deg2rad(15)" in CAMERA, (
        "camera.py's mount-yaw plausibility constant changed; the 15.0 passed "
        "from pipeline.py was chosen to match it and must be updated together")


def test_the_gate_being_guarded_still_exists():
    """If upstream fixes this properly, this whole file should be revisited.

    A guard for a defect that no longer exists is rot — it should fail loudly
    rather than pass vacuously forever.
    """
    assert "max_yaw_correction_deg" in LANE and "not credible" in LANE, (
        "lane_calib's credibility gate has changed shape upstream — re-derive "
        "whether the pipeline.py workaround is still the right fix")


def test_declined_yaw_still_collapses_to_none_so_the_diagnostic_is_suppressed():
    """Pin layer 3, the one that hid the error.

    `pipeline.py` only logs the metres-of-error line when `res.yaw_deg is not
    None`, and `lane_calib` nulls it on decline. So a rejected yaw prints no
    magnitude. This test documents that coupling; if either side changes, the
    suppression may be gone (good) and this note should be revisited.
    """
    assert "yaw_deg if yaw_ok else None" in LANE
    assert "if res.yaw_deg is not None:" in PIPELINE
