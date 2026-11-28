"""Worst-case straight-line bowing sweep + per-patch locality, for the
cylindrical frame declared at calib.py:1244-1248.

The question the review must answer: a ViT with 2D positional encodings sees
the image as a grid of 16x16 patches. Two things could hurt it:
  (a) GLOBAL: a world straight line becomes a curve -> a linear/edge prior
      learned at one place does not transfer to another.
  (b) LOCAL: within ONE 16-px patch, is the mapping still ~affine? If yes, the
      patch-level feature extractor is unaffected and only the *arrangement*
      of patches changes -- which is exactly what a positional encoding can
      absorb.
Both are computed here from the code's own formulas. No data needed.
"""
import math

F = 305.5774907364391
H, W = 256, 640
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0


def cyl(X, Y, Z):
    phi = math.atan2(X, Z)
    rho = math.hypot(X, Z)
    if rho < 1e-9:
        return None
    return CX + F * phi, CY + F * Y / rho


def pin(X, Y, Z):
    if Z <= 1e-6:
        return None
    return CX + F * X / Z, CY + F * Y / Z


def inside(p):
    return p is not None and 0 <= p[0] < W and 0 <= p[1] < H


def sagitta(pts):
    if len(pts) < 3:
        return None
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return 0.0
    return max(abs((px - x0) * dy - (py - y0) * dx) / L for px, py in pts)


def sample_line(P0, D, ts, proj):
    out = []
    for t in ts:
        p = proj(P0[0] + D[0] * t, P0[1] + D[1] * t, P0[2] + D[2] * t)
        if inside(p):
            out.append(p)
    return out


CAMH = 1.5
ts_fwd = [0.5 * (1.09 ** i) for i in range(120)]          # 0.5 .. ~1e4 m
ts_lat = [-40 + 0.25 * i for i in range(321)]             # -40 .. +40 m

CASES = [
    # (name, point-on-line, direction, parameter samples)
    ("lane marking, ego lane edge (X=1.75)", (1.75, CAMH, 0.0), (0, 0, 1), ts_fwd),
    ("lane marking, next lane (X=3.5)",      (3.5, CAMH, 0.0), (0, 0, 1), ts_fwd),
    ("lane marking, 2 lanes over (X=7)",     (7.0, CAMH, 0.0), (0, 0, 1), ts_fwd),
    ("kerb / barrier top, 0.9 m high, X=5",  (5.0, CAMH - 0.9, 0.0), (0, 0, 1), ts_fwd),
    ("building roofline 8 m high, X=12",     (12.0, CAMH - 8.0, 0.0), (0, 0, 1), ts_fwd),
    ("STOP LINE across road, Z=15",          (0.0, CAMH, 15.0), (1, 0, 0), ts_lat),
    ("STOP LINE across road, Z=8",           (0.0, CAMH, 8.0), (1, 0, 0), ts_lat),
    ("STOP LINE across road, Z=30",          (0.0, CAMH, 30.0), (1, 0, 0), ts_lat),
    ("overhead gantry 5 m, Z=30",            (0.0, CAMH - 5.0, 30.0), (1, 0, 0), ts_lat),
    ("overhead gantry 5 m, Z=15",            (0.0, CAMH - 5.0, 15.0), (1, 0, 0), ts_lat),
    ("cross-street kerb, Z=20, 45 deg",      (0.0, CAMH, 20.0),
     (math.cos(math.pi / 4), 0, math.sin(math.pi / 4)), ts_lat),
    ("HORIZON (elev 0)",                     (0.0, 0.0, 1.0), (1, 0, 0),
     [-1e5 + 1e3 * i for i in range(201)]),
    ("vertical pole at X=3, Z=12",           (3.0, 0.0, 12.0), (0, 1, 0),
     [-6 + 0.05 * i for i in range(241)]),
]

print("=" * 92)
print("WORST-CASE STRAIGHT-LINE BOWING, cylindrical vs pinhole (same f_ref)")
print("Sagitta = max perpendicular deviation from the chord, over the IN-FRAME part.")
print("=" * 92)
print(f"{'world line':<40} {'n px':>5} {'span px':>8} {'CYL sag':>9} "
      f"{'PIN sag':>9} {'CYL % of H':>11}")
worst = (0, "")
for name, P0, D, ts in CASES:
    c = sample_line(P0, D, ts, cyl)
    p = sample_line(P0, D, ts, pin)
    sc = sagitta(c)
    sp = sagitta(p)
    if sc is None:
        print(f"{name:<40} {'--- not visible in frame ---':>45}")
        continue
    span = math.hypot(c[-1][0] - c[0][0], c[-1][1] - c[0][1])
    if sc > worst[0]:
        worst = (sc, name)
    sps = f"{sp:9.2f}" if sp is not None else "      n/a"
    print(f"{name:<40} {len(c):>5} {span:>8.1f} {sc:>9.2f} {sps} "
          f"{sc/H*100:>10.2f}%")
print()
print(f"WORST cylindrical sagitta over all cases: {worst[0]:.2f} px "
      f"({worst[0]/H*100:.2f}% of frame height)  -- {worst[1]}")

print()
print("=" * 92)
print("LOCAL (PER-PATCH) DEPARTURE FROM AFFINE  -- does one 16x16 patch care?")
print("=" * 92)
print("For a small world patch imaged at azimuth phi, compare the cylindrical")
print("Jacobian to the best local affine map. The cylindrical map u = f*phi,")
print("v = f*Y/rho is SMOOTH; its second-order term over a 16-px patch is the")
print("relevant quantity. Reported: max |curvature| * (patch/2)^2, in px.\n")
# d^2u/dphi^2 = 0 exactly (u is LINEAR in azimuth) -> no horizontal 2nd-order term.
# The vertical term: v = f * tan(elev). Over a patch spanning dphi = 16/f rad,
# elevation of a FIXED WORLD POINT does not change, so the only curvature comes
# from the scene, not the projection. Quantify via a worst-case ground line.
patch = 16
dphi = patch / F
print(f"  one 16-px patch spans {math.degrees(dphi):.3f} deg of azimuth "
      f"({patch} px at {math.degrees(1/F)*60:.2f} arcmin/px)")
print("  d(u)/d(phi) = f  EXACTLY, and d^2(u)/d(phi)^2 = 0 EXACTLY:")
print("  the horizontal mapping is PERFECTLY LINEAR in azimuth -> zero")
print("  horizontal second-order distortion anywhere in the frame.")
print("  (Pinhole, by contrast: d^2/dphi^2 (f tan phi) = 2 f sec^2 phi tan phi,")
for az in (0, 20, 40, 55):
    a = math.radians(az)
    curv = 2 * F / math.cos(a) ** 2 * math.tan(a)
    print(f"     at {az:>2} deg = {curv:9.1f} px/rad^2 -> "
          f"{0.5*curv*(dphi/2)**2:6.3f} px over half a patch)")
print()
print("  Worst-case LOCAL sagitta of a world straight line WITHIN one patch:")
for name, P0, D, ts in CASES:
    c = sample_line(P0, D, ts, cyl)
    if not c or len(c) < 5:
        continue
    # slide a 16-px window along the polyline and take the worst local sagitta
    best = 0.0
    for i in range(len(c)):
        win = [c[i]]
        for j in range(i + 1, len(c)):
            if math.hypot(c[j][0] - c[i][0], c[j][1] - c[i][1]) > patch:
                break
            win.append(c[j])
        if len(win) >= 3:
            best = max(best, sagitta(win))
    print(f"    {name:<40} {best:6.3f} px within a 16-px patch")
