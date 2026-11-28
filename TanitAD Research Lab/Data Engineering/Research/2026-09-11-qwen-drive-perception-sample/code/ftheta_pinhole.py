"""f-theta (PhysicalAI) -> rectilinear pinhole rectification, RECTANGULAR output.

WHY THIS FILE EXISTS
--------------------
Qwen-Drive-1.0's BEV perception head builds its lift-splat projection from a
3x3 PINHOLE intrinsic matrix K (``qwen_drive_perception.geometry.build_lidar2img``).
Every PhysicalAI-AV camera is an **f-theta fisheye** (``fw_poly_0..4``), which has
no 3x3 K at all. Feeding a nominal pinhole focal to an f-theta frame is a
MEASURED past error of this programme (``stack/tanitad/data/calib.py:305``).

``stack/tanitad/data/calib.py::ftheta_undistort_grid`` already does this, but only
for a SQUARE output centred on the optical axis. Qwen-Drive wants 896x512, and the
PhysicalAI front-wide has two rigs whose principal point sits at cy~543 (rig A) or
cy~755 (rig B) on a 1080-tall sensor -- so a symmetric fan centred on the optical
axis runs off the bottom of the sensor for rig B and ``padding_mode="border"``
would REPLICATE edge pixels into what looks like real road.

This module therefore (a) supports a rectangular output, (b) lets the output
principal point (u0, v0) float so the fan stays on the sensor, and (c) returns an
explicit OBSERVED MASK plus the observed fraction, so unobserved pixels are
recorded rather than silently invented.

⛔ The mask is the point. A rectifier that cannot say which pixels it made up is
the "probe that reports the wrong scope" trap in optics costume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FTheta:
    """r(theta) = sum_i poly[i] * theta**i, in NATIVE px from (cx, cy)."""

    poly: tuple[float, ...]
    cx: float
    cy: float
    width: int
    height: int

    def r_of_theta(self, theta):
        r = np.zeros_like(np.asarray(theta, dtype=np.float64))
        for c in reversed(self.poly):
            r = r * theta + c
        return r

    def theta_of_r(self, r_target: float, hi: float = 1.75) -> float:
        """Inverse by bisection; the map is monotone across a fisheye field."""
        lo, high = 0.0, hi
        for _ in range(90):
            mid = 0.5 * (lo + high)
            if float(self.r_of_theta(np.array(mid))) < r_target:
                lo = mid
            else:
                high = mid
        return 0.5 * (lo + high)

    @property
    def paraxial_focal(self) -> float:
        return float(self.poly[1])


def max_theta_on_sensor(intr: FTheta) -> dict:
    """Largest incidence angle still on the sensor, per direction from (cx, cy)."""
    return {
        "left": intr.theta_of_r(intr.cx),
        "right": intr.theta_of_r(intr.width - 1 - intr.cx),
        "up": intr.theta_of_r(intr.cy),
        "down": intr.theta_of_r(intr.height - 1 - intr.cy),
    }


def choose_pinhole(intr: FTheta, out_w: int, out_h: int, hfov_deg: float) -> dict:
    """Pick f and the output principal point (u0, v0).

    f comes from the requested horizontal FOV. (u0, v0) are then placed so the
    output window lands inside the observed field wherever that is possible --
    this is what keeps rig B's low principal point from fanning off the sensor.
    A pinhole K may have any principal point, and ``build_lidar2img`` consumes an
    arbitrary 3x3, so an off-centre (u0, v0) is fully admissible.
    """
    f = (out_w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    lim = max_theta_on_sensor(intr)
    # available half-extents in OUTPUT pixels
    av = {k: f * np.tan(min(v, np.deg2rad(88.0))) for k, v in lim.items()}

    def place(size, before, after):
        """Position the output principal point to MAXIMISE observed area.

        Two regimes, and the second one is the reason this is not just a clip:
        * ``before + after >= size`` -- the window fits somewhere, so centre it and
          slide only as far as needed to keep both edges on the sensor.
        * otherwise the observed field is SMALLER than the window, so some
          overflow is unavoidable. Splitting it in proportion to the available
          field beats butting the fan against one edge: forcing the top to fit
          exactly just pushes the whole deficit out of the bottom, which MEASURED
          -1.5 pp of observed area on the 70 deg rear cameras.
        """
        if before + after >= size:
            return float(np.clip(size / 2.0, size - after, before))
        return float(size * before / max(before + after, 1e-9))

    u0 = place(out_w, av["left"], av["right"])
    v0 = place(out_h, av["up"], av["down"])
    return {
        "f": float(f),
        "u0": float(u0),
        "v0": float(v0),
        "hfov_deg": float(hfov_deg),
        "vfov_deg": float(np.rad2deg(np.arctan(v0 / f) + np.arctan((out_h - v0) / f))),
        "max_theta_deg": {k: float(np.rad2deg(v)) for k, v in lim.items()},
    }


def rectify_map(intr: FTheta, out_w: int, out_h: int, f: float, u0: float, v0: float):
    """Return (map_x, map_y, valid) sampling the native f-theta frame.

    Output pixel (u, v) -> ray ((u-u0)/f, (v-v0)/f, 1) -> theta -> native radius.
    ``valid`` is False wherever the ray leaves the physical sensor.
    """
    us, vs = np.meshgrid(np.arange(out_w, dtype=np.float64),
                         np.arange(out_h, dtype=np.float64))
    x = (us - u0) / f
    y = (vs - v0) / f
    rho = np.sqrt(x * x + y * y)                       # rectilinear radius (unit z)
    theta = np.arctan(rho)                             # incidence angle
    r = intr.r_of_theta(theta)                         # native fisheye radius [px]
    scale = np.where(rho > 1e-9, r / np.maximum(rho, 1e-9), 0.0)
    map_x = intr.cx + x * scale
    map_y = intr.cy + y * scale
    valid = ((map_x >= 0) & (map_x <= intr.width - 1)
             & (map_y >= 0) & (map_y <= intr.height - 1))
    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def remap_bilinear(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray,
                   valid: np.ndarray) -> np.ndarray:
    """Bilinear sample ``img`` [H,W,3] uint8; unobserved pixels become black."""
    h, w = img.shape[:2]
    xs = np.clip(map_x, 0, w - 1)
    ys = np.clip(map_y, 0, h - 1)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (xs - x0)[..., None]
    wy = (ys - y0)[..., None]
    im = img.astype(np.float32)
    top = im[y0, x0] * (1 - wx) + im[y0, x1] * wx
    bot = im[y1, x0] * (1 - wx) + im[y1, x1] * wx
    out = top * (1 - wy) + bot * wy
    out[~valid] = 0.0
    return np.clip(out, 0, 255).astype(np.uint8)


def build_K(f: float, u0: float, v0: float) -> np.ndarray:
    return np.array([[f, 0.0, u0], [0.0, f, v0], [0.0, 0.0, 1.0]], dtype=np.float32)


def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Rotation matrix from (x, y, z, w). PhysicalAI extrinsics are CAMERA->RIG."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
