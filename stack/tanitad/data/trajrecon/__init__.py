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
This package was copied verbatim from the upstream ``trajlib`` project; the
module bodies are unmodified. Accuracy figures quoted in ``README.md`` (hold-out
RMS 2.23 m position, 1.27 m/s speed, 0.84 deg heading on the 2025-08-11 session)
are **INHERITED** from that project's own documentation. They have NOT been
reproduced in this repo and must not be cited as a TanitAD result until
re-measured here with :mod:`.validate`.

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

__all__ = [
    "accel_source", "camera", "diagnose", "frame_folder", "frames", "geo",
    "ground_calib", "io_sensorlogger", "lane_calib", "pipeline", "plane_calib",
    "quality", "render_video", "scale_calib", "steering", "timesync",
    "trajectory", "validate", "viz", "vp_calib",
]


def __getattr__(name: str):
    """Resolve submodules on first access (see "Why imports here are lazy")."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
