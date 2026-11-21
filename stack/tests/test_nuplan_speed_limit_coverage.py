"""Analytic targets for the GeoPackage length parser in nuplan_speed_limit_coverage.py.

The first run of that script reported 0.0 km of lanes per city because nuPlan stores
EPSG:4326 degrees and the parser summed them as metres. These expectations are LITERALS
derived by hand, not expressions over the code under test.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nuplan_speed_limit_coverage as M  # noqa: E402


def _gp_linestring(points, srs=4326):
    # GeoPackage header: magic 'GP', version 0, flags (little-endian, no envelope), srs_id
    header = b"GP" + bytes([0, 0b00000001]) + struct.pack("<i", srs)
    wkb = struct.pack("<BII", 1, 2, len(points)) + b"".join(struct.pack("<dd", *p) for p in points)
    return header + wkb


def test_cartesian_345_triangle_is_exactly_5():
    assert abs(M._gpkg_linestring_length(_gp_linestring([(0, 0), (3, 4)], srs=0)) - 5.0) < 1e-12


def test_geographic_north_step_is_about_110_6_m():
    # 0.001 deg of latitude at 40 N ~ 110.574 m (independent reference: WGS-84 meridian arc ~110.5-111.0 km/deg)
    ln = M._gpkg_linestring_length(_gp_linestring([(-80.0, 40.0), (-80.0, 40.001)]), geographic=True)
    assert 110.0 < ln < 111.2


def test_geographic_east_step_shrinks_with_cos_latitude():
    # 0.001 deg of longitude at 60 N is half of that at the equator: ~55.66 m
    ln = M._gpkg_linestring_length(_gp_linestring([(10.0, 60.0), (10.001, 60.0)]), geographic=True)
    assert 55.3 < ln < 56.0


def test_regression_degrees_read_as_metres_is_detected():
    # the historical defect: without the geographic flag a 110 m lane reads as 0.001 "m"
    ln = M._gpkg_linestring_length(_gp_linestring([(-80.0, 40.0), (-80.0, 40.001)]), geographic=False)
    assert ln < 0.01
