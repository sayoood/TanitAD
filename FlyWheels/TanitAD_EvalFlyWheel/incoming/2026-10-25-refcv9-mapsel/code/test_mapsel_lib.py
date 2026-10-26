"""test_mapsel_lib.py -- ANALYTIC-target tests of mapsel_lib (expectations are LITERALS derived by hand, never re-computed from the code under test) + a deliberate-regression arm.
Run: PYTHONPATH=<tree>/stack python -X utf8 test_mapsel_lib.py   (prints UNITPASS)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mapsel_lib as L  # noqa: E402

torch.set_num_threads(2)
X_SLOTS = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def straight(y=0.0):
    p = torch.zeros(1, 8, 2)
    p[0, :, 0] = torch.tensor(X_SLOTS)
    p[0, :, 1] = y
    return p


def road(code=1):
    c = np.full((L.H, L.W), code, np.uint8)
    return c


def put(c, x0, x1, y0, y1, code):
    c[int(round(x0 / 0.1)):int(round(x1 / 0.1)), int(round((y0 + 30.0) / 0.1)):int(round((y1 + 30.0) / 0.1))] = code
    return c


def run_all():
    g = L.Geom(straight())
    # --- 0. geometry constants == pdm_proxy (the refcv8 critic's own box)
    try:
        from tanitad.rl import pdm_proxy as pp
        assert (pp.PROXY.ego_length, pp.PROXY.ego_width, pp.PROXY.rear_axle_to_center) == (L.EGO_L, L.EGO_W, L.REAR_TO_CTR), "box constants drifted from pdm_proxy.PROXY"
        pinned = True
    except ImportError:
        pinned = False
    # --- 1. sample count: s from 5.0 to 60.0 at 0.25 m = 221 samples + the always-valid final column
    assert int(g.n_samples[0]) == 222, g.n_samples
    # --- 2. an all-drivable raster: nothing off
    off, edge, lane = g.flags(L.masks_gt(road(1)))
    assert not off[0] and not edge[0] and not lane[0]
    # --- 3. drivable: a sidewalk band across the road at x in [30, 31): rear corners at (x - 1.127) .. front (x + 4.049) sweep it -> OFF
    c = put(road(1), 30.0, 31.0, -30.0, 30.0, 7)
    assert g.flags(L.masks_gt(c))[0][0]
    # --- 4. the final slot is x = 60: the front corners end at 64.049.  A band at [64.5, 65) is NOT reached; a band at [63.9, 64.1) IS (front edge 64.049 inside)
    assert not g.flags(L.masks_gt(put(road(1), 64.5, 65.0, -30.0, 30.0, 7)))[0][0]
    assert g.flags(L.masks_gt(put(road(1), 63.9, 64.1, -30.0, 30.0, 7)))[0][0]
    # --- 5. a band ONLY to the left of the body (y in [1.4, 2.0)): the left corners are at y = +1.1485, so no drivable violation
    assert not g.flags(L.masks_gt(put(road(1), 0, 100, 1.4, 2.0, 7)))[0][0]
    #       ... but a band at y in [1.0, 1.3) contains the left corners (1.1485) -> OFF
    assert g.flags(L.masks_gt(put(road(1), 0, 100, 1.0, 1.3, 7)))[0][0]
    # --- 6. edge margin 0.3 m: an edge strip at y in [1.4, 1.5) is within 0.3 m of the left corners (1.1485 + 0.3 = 1.4485 >= 1.4) -> contact
    assert g.flags(L.masks_gt(put(road(1), 0, 100, 1.4, 1.5, 5)))[1][0]
    #       an edge strip at y in [1.6, 1.7) is NOT (dilated lower edge 1.6 - 0.3 = 1.3 > 1.1485 + 0.0 ... corners at 1.1485 < 1.3)
    assert not g.flags(L.masks_gt(put(road(1), 0, 100, 1.6, 1.7, 5)))[1][0]
    # --- 7. lane crossing uses the CENTRE line only: a thin lane strip across at x in [30, 30.1) is crossed (band +-0.2 m = 0.5 m wide >= the 0.25 m step) ...
    assert g.flags(L.masks_gt(put(road(1), 30.0, 30.1, -30.0, 30.0, 2)))[2][0]
    #       ... a lane line PARALLEL at y = +3.5 is not crossed ...
    assert not g.flags(L.masks_gt(put(road(1), 0, 100, 3.5, 3.6, 2)))[2][0]
    #       ... and one exactly under the centre line (y in [-0.1, 0.1)) is
    assert g.flags(L.masks_gt(put(road(1), 0, 100, -0.1, 0.1, 2)))[2][0]
    # --- 8. AMENDMENT 1: NOT SEEN (255) AND nocls (0) are UNKNOWN: neither raises off_drv; a SIDEWALK (7) band does; an EDGE (5) band is NOT a T1 violation (it is a T2 concept)
    assert not g.flags(L.masks_gt(put(road(1), 30.0, 31.0, -30.0, 30.0, 255)))[0][0]
    assert not g.flags(L.masks_gt(put(road(1), 30.0, 31.0, -30.0, 30.0, 0)))[0][0]
    assert not g.flags(L.masks_gt(put(road(1), 30.0, 31.0, -30.0, 30.0, 5)))[0][0]
    assert g.flags(L.masks_gt(put(road(1), 30.0, 31.0, -30.0, 30.0, 7)))[0][0]
    # --- 8b. COVERAGE: an all-nocls raster -> nocls share exactly 1.0 / unseen 0.0; all not-seen -> 0.0 / 1.0; road -> 0 / 0; left half nocls (cols >= 300 <=> y >= 0): FL, RL and the centre (y = 0 -> col 300)
    #       are on it, RR and FR (y = -1.1485) are not -> exactly 3 of 5 = 0.6 (every sample is in range: the path starts at x = 5, the rear corners at 5 - 1.127 > 0)
    cv = lambda c: tuple(float(x[0]) for x in g.coverage(L.masks_gt(c)))   # noqa: E731
    assert cv(road(0)) == (1.0, 0.0) and cv(road(255)) == (0.0, 1.0) and cv(road(1)) == (0.0, 0.0)
    half = road(1); half[:, 300:] = 0
    assert abs(cv(half)[0] - 0.6) < 1e-6, cv(half)
    # --- 8c. OWN mirror: P^(sidewalk) >= 128 on a band raises the flag; 127 does not
    pw = np.zeros((L.H, L.W), np.uint8); cd = np.full((L.H, L.W), 1, np.uint8)
    pw[300:310, :] = 128
    assert g.flags(L.masks_own(pw, cd))[0][0]
    pw[300:310, :] = 127
    assert not g.flags(L.masks_own(pw, cd))[0][0]
    pm = g.own_cont(pw, np.zeros_like(pw))[0]
    assert abs(float(pm[0]) - 127 / 255.0) < 1e-6, pm
    # --- 9. a plan entirely beyond the raster (x > 100 m for every sample incl. footprint) is UNCHECKED
    far = straight(); far[0, :, 0] = torch.tensor([120., 130, 140, 150, 160, 170, 180, 190])
    assert not L.Geom(far).flags(L.masks_gt(road(7)))[0][0]
    # --- 10. a STALLED plan (all slots at the origin): one sample (+ the final column); rear corners at x = -1.127 are outside the raster (unchecked); the front ones (4.049, +-1.1485) are inside
    z = torch.zeros(1, 8, 2)
    gz = L.Geom(z)
    assert int(gz.n_samples[0]) == 2, gz.n_samples
    assert gz.flags(L.masks_gt(road(7)))[0][0]                               # front corners are in range and on a sidewalk
    assert not gz.flags(L.masks_gt(put(road(1), 3.0, 4.0, -30.0, 30.0, 7)))[0][0]   # a band at x in [3, 4) holds neither the centre (1.461) nor the front corners (4.049)
    # --- 11. a left turn of radius 40 m (arc), headings follow the chords: the inner (left) corners sit closer to the circle centre.  Slots at arc length v*t, v = 8 m/s
    R, v = 40.0, 8.0
    ang = np.array(L.SLOT_T) * v / R
    arc = torch.tensor(np.stack([R * np.sin(ang), R * (1 - np.cos(ang))], -1), dtype=torch.float32)[None]
    ga = L.Geom(arc)
    # a drivable disc of radius 44 around the centre (0, 40) EXCLUDES nothing of the path (outer front corner radius ~ 40 + 1.15 + small): all drivable -> not off
    X = (np.arange(L.H) + 0.5)[:, None] * 0.1 * np.ones((1, L.W))
    Y = ((np.arange(L.W) + 0.5)[None, :] * 0.1 - 30.0) * np.ones((L.H, 1))
    rr = np.hypot(X - 0.0, Y - 40.0)
    disc = np.where(rr <= 43.0, 1, 7).astype(np.uint8)                      # radius 43 > 40 + 1.15 + chord sag
    assert not ga.flags(L.masks_gt(disc))[0][0]
    disc2 = np.where(rr <= 41.0, 1, 7).astype(np.uint8)                     # radius 41: the OUTER corners (40 + 1.1485 = 41.15) leave it -> off
    assert ga.flags(L.masks_gt(disc2))[0][0]
    # --- 12. tier_pick on a planted window: 5 candidates, scores 5 > 4 > 3 > 2 > 1; cand 0 off, cand 1 edge, cand 2 lane, cand 3 clean, cand 4 clean-but-ineligible
    sc = np.array([5., 4, 3, 2, 1]); el = np.array([1, 1, 1, 1, 0], bool)
    off_ = np.array([1, 0, 0, 0, 0], bool); ed_ = np.array([0, 1, 0, 0, 0], bool); ln_ = np.array([0, 0, 1, 0, 0], bool)
    assert L.tier_pick(sc, el, off_, ed_, ln_, 0)[0] == 0
    assert L.tier_pick(sc, el, off_, ed_, ln_, 1)[0] == 1
    assert L.tier_pick(sc, el, off_, ed_, ln_, 2)[0] == 2
    assert L.tier_pick(sc, el, off_, ed_, ln_, 3)[0] == 3
    assert L.tier_pick(sc, el, off_, ed_, ln_, 3, lc_window=True)[0] == 2        # the lane-change exception skips T3
    # fallbacks: every eligible candidate off -> the filter is skipped (fallback flag set), the native pick stays
    p, fb, n = L.tier_pick(sc, el, np.ones(5, bool), ed_, ln_, 1)
    assert p == 0 and fb == [1, 0, 0] and n == 4
    # --- 13. the eligible() replica: reach drops cand 0; ceil absent (-1); emit absent
    s2 = np.array([[5., 4, 3]]); r = np.array([[0, 1, 1]], np.int8); a = np.full((1, 3), -1, np.int8)
    assert L.eligible(s2, r, a, a)[0].tolist() == [False, True, True]
    #       a dead row (reach drops everything) falls back to the finite set, never an empty one
    assert L.eligible(s2, np.zeros((1, 3), np.int8), a, a)[0].tolist() == [False, False, False]
    # --- 14. dilate equals a brute-force square dilation
    m = np.zeros((20, 30), bool); m[5, 7] = True; m[15, 29] = True
    d = L.dilate(torch.tensor(m), 2).numpy()
    bf = np.zeros_like(m)
    for i, j in zip(*np.nonzero(m)):
        bf[max(0, i - 2):i + 3, max(0, j - 2):j + 3] = True
    assert (d == bf).all()
    return pinned


def main():
    pinned = run_all()
    # --- DELIBERATE REGRESSION: a footprint whose centre offset is wrong (0 instead of 1.461) must make a hand-derived expectation FAIL
    saved = L.REAR_TO_CTR
    L.REAR_TO_CTR = 0.0
    try:
        try:
            run_all()
            raise SystemExit("REGRESSION ARM NOT DETECTED: the tests passed with REAR_TO_CTR = 0")
        except AssertionError:
            pass
    finally:
        L.REAR_TO_CTR = saved
    saved = L.BAD_CODES
    L.BAD_CODES = (0, 5, 7)                                                  # the SEALED (pre-amendment) class set: nocls / edge as violations must be detected
    try:
        try:
            run_all()
            raise SystemExit("REGRESSION ARM NOT DETECTED: the tests passed with BAD_CODES = (0, 5, 7)")
        except AssertionError:
            pass
    finally:
        L.BAD_CODES = saved
    saved = L.STEP_M
    L.STEP_M = 8.0                                                           # a path sampled every 8 m skips a 0.5 m lane band
    try:
        try:
            run_all()
            raise SystemExit("REGRESSION ARM NOT DETECTED: the tests passed with STEP_M = 8")
        except AssertionError:
            pass
    finally:
        L.STEP_M = saved
    print(f"UNITPASS pdm_proxy_constants_pinned={pinned} regression_arms_detected=3")


if __name__ == "__main__":
    main()
