"""Ground-truth ego-trajectory reconstruction from smartphone dashcam recordings.

Turns a Sensor Logger (Android) export -- camera MP4 plus IMU / GNSS / orientation
CSVs -- into, for every video frame, the vehicle's past and future trajectory in
the ego frame, plus a bird's-eye view and an in-image projection. The pipeline is::

    load -> QC gates -> time sync -> vehicle frame -> trajectory -> validation -> render

The end-to-end driver is :mod:`.pipeline`, which consumes a folder of Sensor
Logger ``.zip`` archives directly (no manual extraction) and is resumable via a
registry keyed on an input fingerprint::

    python -m tanitad.data.trajrecon.pipeline --input-dir <zips> --output-dir <out>

Conventions
-----------
**Ego frame is FLU** (ROS / ISO 8855): ``x`` forward, ``y`` left, ``z`` up, origin
at the vehicle at the reference instant. ``ego_trajectory`` returns ``t`` running
from ``-t_past`` to ``+t_future`` with ``x = y = 0`` at ``t = 0``. **Camera frame
is OpenCV** (``x`` right, ``y`` down, ``z`` forward). **Session timebase** is
``seconds_elapsed``, seconds since ``Metadata.csv``'s recording epoch.

Why imports here are lazy
-------------------------
These modules need ``opencv-python``, ``scipy``, ``pandas`` and ``matplotlib``
(the ``trajrecon`` extra), and ``pipeline``/``timesync`` additionally shell out to
**ffmpeg AND ffprobe on PATH** -- neither of which is a pip dependency. Importing
them eagerly would make a bare ``import tanitad.data`` fail for every consumer
that has none of that installed, so submodules resolve on first attribute access
instead. This mirrors the lazy ``av`` import in :mod:`tanitad.data.comma2k19`.

Provenance and evidence class
-----------------------------
This package was copied from the upstream ``trajlib`` project. The module bodies
were originally unmodified; **that is no longer true and the exceptions are
listed here**, because a silently-drifted "verbatim" copy is worse than an
honestly-annotated one.

**Deliberate divergences from upstream** (each with the measurement that forced it):

* ``camera.py`` — ``estimate_foe``'s rotation gate is fed from the **gyro**
  (``_gyro_rot_on_frame_pairs``) rather than from ``timesync.image_angular_rate``.
  That function is a TIMING instrument: its own docstring says the yaw amplitude
  is "only approximate" because forward translation leaks into ``tx``, and the
  leak scales with SPEED. MEASURED 2026-08-08 at 70–84 km/h: camera |yaw| p50
  **5.42 deg/s** vs gyro **1.46 deg/s**, so all three axes passed the 2.01 deg/s
  gate on **0.4 %** of frames — below ``estimate_foe``'s own 10-frame floor, and
  it returned ``None``. With the gyro the same clip yields **425,093 flow
  vectors at 86 % inliers**, and the camera verdict goes **DEGRADED → OK**.
* ``pipeline.py`` — ``self_calibrate`` now passes ``max_yaw_correction_deg`` to
  ``lane_calib.estimate`` explicitly. Upstream always used the default 4.0, which
  gates the lane-VP yaw against ``cam.yaw``. When the FOE fit fails, ``cam.yaw``
  is the **nominal 0.0**, so that gate degenerates into "reject any mount yaw
  beyond 4 deg of dead ahead" — rejecting hardest exactly when the mount is most
  crooked. MEASURED 2026-08-08 on ``14-19-54``: a stable −7.01 deg (half-split
  spread 0.20 deg) was discarded as "7.0 deg from the FOE" that did not exist,
  and an independent VP fit over 154 straight frames / 6689 segments gives
  −6.05 deg [95 % CI −6.28, −5.83]. Shipping 0.0 put **4.9 m of lateral error at
  40 m** into the overlay — outside the ego lane from ~15 m onward.

Accuracy figures quoted in ``README.md`` (hold-out RMS 2.23 m position, 1.27 m/s
speed, 0.84 deg heading on the 2025-08-11 session) are **INHERITED** from that
project's own documentation. They have NOT been reproduced in this repo and must
not be cited as a TanitAD result. *(Our own 2026-08-08 hold-out — 0.70 m /
0.08 m/s / 0.53 deg — is in
``TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-08-trajrecon-render/``.
It is a different recording and route, so it supersedes rather than beats them.)*

⚠️ ``README.md`` and the code DISAGREE on the default camera height: the README's
parameter table says 1.25 m, its own limitations section says 1.17 m, and
``pipeline.py``'s ``--cam-height`` default is **1.17**. The code is the source of
truth. Camera height scales the entire ground projection linearly, so this is not
cosmetic -- but note ``--plane-calib`` is on by default and measures height from
the road-plane homography, making the operator value a fallback rather than the
operating assumption.

⚠️ ``--lane-width`` defaults to **3.65 m (US)**. It is the external metric that
lets ``scale_calib`` separate focal length from camera height, so on a German
recording it must be set to 3.50 or the ground-projection scale carries the error.
"""

from __future__ import annotations

import importlib
import sys

# ---------------------------------------------------------------------------
# The ``trajlib`` alias — without it NOTHING in this package runs.
# ---------------------------------------------------------------------------
# The 24 upstream modules were landed BYTE-EXACT, so 7 of them still import each
# other under the upstream distribution name: ``from trajlib import timesync``,
# ``from trajlib.camera import calibrate_camera``, and so on (pipeline.py:53-62,
# render_video.py:25-34, run_demo.py:23-24, and 4 more). Nothing in this repo
# provides ``trajlib``, so the documented entry point
#
#     python -m tanitad.data.trajrecon.pipeline ...
#
# died at the first import until this alias existed.
#
# It went unnoticed because `stack/tests/test_trajrecon_*.py` read the sources as
# BYTES -- they check for control characters and valid UTF-8, and never import a
# single module. Byte-integrity was proven; runnability never was. The static
# check in `test_trajrecon_imports.py` is the guard that actually covers it.
#
# Rewriting the imports was the alternative and was rejected: byte-exactness is
# the property that makes "verified against upstream" mean anything here, and it
# is what `test_trajrecon_integrity.py` exists to protect. Aliasing keeps the
# upstream bodies untouched and puts the adaptation in this file, which is ours.
#
# `setdefault`, not assignment: if a real upstream `trajlib` is installed it wins,
# and we never shadow it.
sys.modules.setdefault("trajlib", sys.modules[__name__])

__all__ = [
    "accel_source", "camera", "contract", "diagnose", "frame_folder", "frames",
    "geo", "ground_calib", "io_sensorlogger", "lane_calib", "pipeline",
    "plane_calib", "quality", "render_video", "run_demo", "scale_calib",
    "steering", "timesync", "trajectory", "validate", "vehicle_frame", "viz",
    "vp_calib",
]


def __getattr__(name: str):
    """Resolve submodules on first access (see "Why imports here are lazy")."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
