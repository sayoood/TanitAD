"""Geodetic helpers: WGS-84 lat/lon/alt <-> local ENU tangent plane.

Uses the exact ellipsoidal transform (geodetic -> ECEF -> ENU) rather than the
flat-earth approximation, so that position error stays sub-millimetre over the
few-kilometre extent of a driving session.
"""
from __future__ import annotations

import numpy as np

# WGS-84
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    alt = np.asarray(alt_m, dtype=float)
    sl, cl = np.sin(lat), np.cos(lat)
    N = _A / np.sqrt(1.0 - _E2 * sl * sl)
    x = (N + alt) * cl * np.cos(lon)
    y = (N + alt) * cl * np.sin(lon)
    z = (N * (1.0 - _E2) + alt) * sl
    return np.stack([x, y, z], axis=-1)


def ecef_to_enu_matrix(lat0_deg, lon0_deg):
    """Rotation taking an ECEF *difference* vector into local ENU."""
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    sla, cla = np.sin(lat0), np.cos(lat0)
    slo, clo = np.sin(lon0), np.cos(lon0)
    return np.array([
        [-slo,        clo,       0.0],
        [-sla * clo, -sla * slo, cla],
        [cla * clo,   cla * slo, sla],
    ])


class LocalENU:
    """Local East-North-Up frame anchored at a reference geodetic point."""

    def __init__(self, lat0_deg: float, lon0_deg: float, alt0_m: float = 0.0):
        self.lat0 = float(lat0_deg)
        self.lon0 = float(lon0_deg)
        self.alt0 = float(alt0_m)
        self._r0 = geodetic_to_ecef(self.lat0, self.lon0, self.alt0)
        self._R = ecef_to_enu_matrix(self.lat0, self.lon0)

    def forward(self, lat_deg, lon_deg, alt_m=None):
        """lat/lon/alt -> (E, N, U) in metres. Returns (n, 3)."""
        lat_deg = np.atleast_1d(np.asarray(lat_deg, dtype=float))
        lon_deg = np.atleast_1d(np.asarray(lon_deg, dtype=float))
        if alt_m is None:
            alt_m = np.full_like(lat_deg, self.alt0)
        alt_m = np.atleast_1d(np.asarray(alt_m, dtype=float))
        d = geodetic_to_ecef(lat_deg, lon_deg, alt_m) - self._r0
        return d @ self._R.T

    def inverse(self, enu):
        """(E, N, U) -> (lat, lon, alt). Iterative (Bowring) ECEF->geodetic."""
        enu = np.atleast_2d(np.asarray(enu, dtype=float))
        r = enu @ self._R + self._r0
        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        lon = np.arctan2(y, x)
        p = np.hypot(x, y)
        lat = np.arctan2(z, p * (1.0 - _E2))
        for _ in range(6):
            sl = np.sin(lat)
            N = _A / np.sqrt(1.0 - _E2 * sl * sl)
            alt = p / np.cos(lat) - N
            lat = np.arctan2(z, p * (1.0 - _E2 * N / (N + alt)))
        sl = np.sin(lat)
        N = _A / np.sqrt(1.0 - _E2 * sl * sl)
        alt = p / np.cos(lat) - N
        return np.rad2deg(lat), np.rad2deg(lon), alt


def bearing_to_math_angle(bearing_deg):
    """GPS bearing (deg clockwise from North) -> math angle (rad CCW from East)."""
    return np.deg2rad(90.0 - np.asarray(bearing_deg, dtype=float))


def wrap_pi(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
