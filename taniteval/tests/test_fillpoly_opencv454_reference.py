"""``cv_fill_poly`` is pixel-exact with REAL ``cv2.fillPoly`` 4.5.4 — checked against a banked reference.

⭐ WHY THIS EXISTS. Every nuScenes collision grid is drawn by ``cv_fill_poly``, a port of OpenCV 4.5.4's
``drawing.cpp`` (the main venv has no OpenCV). Its only parity test skipped on this box for its whole life
(*"NOT RUN: OpenCV not installed in this venv"*) — a guard that had never run. Installing OpenCV into the
main venv was ruled out (``CLAUDE.md``: a package install once dragged torch to a wheel the driver could not
run), so a THROWAWAY env drew the reference once:

* ``D:/venvs/opencv-ref-454`` — Python 3.9.25, ``opencv-python-headless`` 4.5.4.60, numpy 1.26.4, installed
  OFFLINE (``--no-index --no-deps``); both wheels' sha256 match PyPI's published digests; torch not importable.
  PI approval relayed by the Master Mind: *"you can download opencv on D:"*.
* 4.5.4 exactly, not the latest (5.0.0.93): a raster is only a reference if it comes from the routine the
  port claims to reproduce. The renderer ASSERTED ``cv2.__version__ == "4.5.4"`` before drawing anything.
* the VERTICES were generated in THIS venv and banked, so no test depends on numpy 1.26 and 2.x producing
  the same random stream.

MEASURED 2026-09-26: **520 / 520 cases pixel-exact, 0 differing pixels** — the old test's 200 random quads,
300 rotated agent boxes through the occupancy builders' own pixel transform, and 20 adversarial shapes.

⚠️ Only the BINARY ``.npz`` is byte-hashed. Text files are converted LF↔CRLF on checkout here, so hashing
``polygons.json`` would fail on another machine for a reason that has nothing to do with the reference.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TE = os.path.dirname(_HERE)
_REPO = os.path.dirname(_TE)
for _p in (os.path.join(_REPO, "stack"), _TE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from adapters import nuscenes_planning as NP              # noqa: E402

FIX = os.path.join(_HERE, "fixtures", "fillpoly_opencv454")
NPZ = os.path.join(FIX, "fillpoly_cv2_454_reference.npz")
RECEIPT = os.path.join(FIX, "RECEIPT.json")

#: MEASURED literals from the render — written out, never computed from the code under test
NPZ_SHA256 = "08e0d3e0f384a5060eb1ac2e11e8428369b458a2636f6edf108a4a86262de5ca"
GROUPS = {"quads": (200, (40, 40), 91489), "boxes": (300, (200, 200), 16976), "adversarial": (20, (64, 64), 15178)}
WHEEL_SHA_PREFIX = {"opencv_python_headless-4.5.4.60-cp39-cp39-win_amd64.whl": "a1f9d41c6afe86fd",
                    "numpy-1.26.4-cp39-cp39-win_amd64.whl": "3373d5d70a5fe74a"}


@pytest.fixture(scope="module")
def ref():
    return np.load(NPZ, allow_pickle=False)


def _cases(ref, g):
    polys, nv, ras = ref[f"{g}_polys"], ref[f"{g}_nverts"], ref[f"{g}_rasters"]
    for i in range(len(nv)):
        yield i, [tuple(v) for v in polys[i, :nv[i]].tolist()], ras[i]


# ------------------------------------------------------------------ the reference is what it claims

def test_the_reference_is_the_banked_bytes():
    h = hashlib.sha256(open(NPZ, "rb").read()).hexdigest()
    assert h == NPZ_SHA256, "the reference rasters changed — a silently replaced reference tests nothing"


def test_the_reference_proves_its_own_version():
    r = json.load(open(RECEIPT, encoding="utf-8"))
    assert r["cv2_version"] == "4.5.4"
    assert "OpenCV 4.5.4" in r["cv2_build_info_head"][0]
    assert r["npz_sha256"] == NPZ_SHA256
    for name, prefix in WHEEL_SHA_PREFIX.items():
        assert r["wheels_sha256_recomputed"][name].startswith(prefix), name
    assert all(not m["cv2_raised"] for m in r["groups"].values())


@pytest.mark.parametrize("g", sorted(GROUPS))
def test_group_shapes_and_content_are_the_measured_literals(ref, g):
    """⛔ Guards against an all-zero reference: a blank raster matches a blank port output trivially."""
    n, hw, px = GROUPS[g]
    assert len(ref[f"{g}_nverts"]) == n
    assert tuple(ref[f"{g}_hw"]) == hw
    assert int(ref[f"{g}_rasters"].sum()) == px
    assert bool(ref[f"{g}_ok"].all())


# ------------------------------------------------------------------ the parity itself

@pytest.mark.parametrize("g", sorted(GROUPS))
def test_port_is_pixel_exact_with_cv2_454(ref, g):
    bad = []
    for i, poly, want in _cases(ref, g):
        got = NP.cv_fill_poly(np.zeros(want.shape, np.uint8), poly, 1)
        if not np.array_equal(got, want):
            bad.append((i, int((got != want).sum()), poly))
    assert not bad, f"{len(bad)} {g} cases differ from cv2 4.5.4; first: {bad[:3]}"


# ------------------------------------------------------------------ the comparison CAN fail

def test_MUTATION_a_shifted_polygon_is_caught(ref):
    """Drawing every quad one pixel to the right must disagree with the reference on (almost) all
    non-empty cases. If it did not, the comparison would have no power to see a real defect."""
    caught = nonempty = 0
    for _i, poly, want in _cases(ref, "quads"):
        if not want.any():
            continue
        nonempty += 1
        got = NP.cv_fill_poly(np.zeros(want.shape, np.uint8), [(x + 1, y) for x, y in poly], 1)
        caught += not np.array_equal(got, want)
    assert nonempty > 150 and caught / nonempty > 0.95, (caught, nonempty)


def test_MUTATION_a_single_flipped_reference_pixel_is_caught(ref):
    """One pixel of difference must be enough to fail — the check is exact, not tolerant."""
    i, poly, want = next(c for c in _cases(ref, "boxes") if c[2].any())
    bent = want.copy()
    r, c = np.argwhere(bent)[0]
    bent[r, c] ^= 1
    got = NP.cv_fill_poly(np.zeros(want.shape, np.uint8), poly, 1)
    assert np.array_equal(got, want) and not np.array_equal(got, bent)
