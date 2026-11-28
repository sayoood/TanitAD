"""MEASURED-BY-COMPUTATION: what TanitAD's cylindrical projection does to
scene geometry, derived from the EXACT formulas in stack/tanitad/data/calib.py.

calib.py:906-935  cylindrical_rays: ray = (sin(phi), y_n, cos(phi))
                  phi = (u - (W-1)/2)/f_ref ;  y_n = (v - (H-1)/2)/f_ref
calib.py:149-165  hfov = 2*(W/2)/f_ref (linear in azimuth); vfov = 2*atan((H/2)/f_ref)
calib.py:1244-1248 PHYSICALAI_WIDE120_256x640 f_ref = 305.5774907364391

Camera convention: +x right, +y DOWN, +z boresight.
No data needed — these are definitional. Cross-checked against the code's own
hfov_deg/vfov_deg properties.
"""
import math

F = 305.5774907364391
H, W = 256, 640
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0
CAM_H = 1.5  # m, assumed camera height above ground (ESTIMATED, sensitivity below)


def cyl_project(X, Y, Z):
    """world/camera-frame point (x right, y DOWN, z forward) -> cylindrical (u, v) px.
    Inverse of cylindrical_rays."""
    if Z <= 0 and X == 0:
        return None
    phi = math.atan2(X, Z)          # ray = (sin phi, y_n, cos phi)
    rho = math.hypot(X, Z)          # sqrt(sin^2+cos^2)=1 scaling => y_n = Y/rho
    y_n = Y / rho
    return CX + F * phi, CY + F * y_n


def pin_project(X, Y, Z):
    """Same point in a PINHOLE frame of the same f_ref (the field's convention)."""
    if Z <= 1e-9:
        return None
    return CX + F * X / Z, CY + F * Y / Z


def sagitta(pts):
    """Max perpendicular deviation (px) of a point list from the chord joining
    its endpoints. This is 'how curved is the line that should be straight'."""
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return 0.0
    return max(abs((px - x0) * dy - (py - y0) * dx) / L for px, py in pts)


print("=" * 78)
print("1. FRAME GEOMETRY (definitional, calib.py:149-165)")
print("=" * 78)
hfov = math.degrees(2 * (W / 2) / F)
vfov = math.degrees(2 * math.atan((H / 2) / F))
print(f"  {H}x{W} cylindrical, f_ref={F}")
print(f"  HFOV {hfov:.4f} deg   VFOV {vfov:.4f} deg")
print(f"  horizontal angular resolution: {hfov/W*60:.3f} arcmin/px "
      f"({hfov/W:.5f} deg/px)  -- UNIFORM across the width (that is the point)")
print(f"  vertical  angular resolution at centre: "
      f"{math.degrees(math.atan((0.5)/F))*60:.3f} arcmin/px")
print(f"  patch16 -> {H//16}x{W//16} = {(H//16)*(W//16)} tokens, "
      f"{hfov/(W//16):.3f} deg per token column")

print()
print("=" * 78)
print("2. STRAIGHT-LINE BOWING -- lane markings, the thing pinhole preserves")
print("=" * 78)
print("A ground-plane line parallel to the ego heading at lateral offset X.")
print("In PINHOLE it is EXACTLY straight (both u,v linear in 1/Z).")
print("In CYLINDRICAL it bows. Sagitta = max deviation from the chord, in px.\n")
print(f"{'lat X (m)':>10} {'Z range (m)':>14} {'cyl sagitta px':>15} "
      f"{'pin sagitta px':>15} {'cyl sag / frame H':>18}")
Zs = [1000, 500, 200, 100, 70, 50, 35, 25, 18, 13, 10, 8, 6, 5, 4, 3.5, 3]
for Xlat in (0.0, 1.75, 3.5, 5.25, 7.0, 10.5):
    cyl_pts, pin_pts = [], []
    for Z in Zs:
        c = cyl_project(Xlat, CAM_H, Z)
        p = pin_project(Xlat, CAM_H, Z)
        # keep only what is inside the frame
        if c and 0 <= c[0] < W and 0 <= c[1] < H:
            cyl_pts.append(c)
        if p and 0 <= p[0] < W and 0 <= p[1] < H:
            pin_pts.append(p)
    if len(cyl_pts) < 3:
        continue
    sc = sagitta(cyl_pts)
    sp = sagitta(pin_pts) if len(pin_pts) >= 3 else float("nan")
    print(f"{Xlat:>10.2f} {min(Zs):>6.0f}-{max(Zs):<7.0f} {sc:>15.2f} "
          f"{sp:>15.2f} {sc/H*100:>17.2f}%")

print()
print("  Cross-check: the pinhole column must be ~0 (float noise) by construction.")

print()
print("=" * 78)
print("3. APPEARANCE SHEAR -- does a car LOOK the same at different azimuths?")
print("=" * 78)
print("A 4.5 m x 1.8 m x 1.5 m box (a car), rear face perpendicular to the ego")
print("heading, placed at constant RANGE 25 m, swept across azimuth.")
print("Reported: apparent width/height of its rear face, and the SKEW (how much")
print("the two vertical edges differ in projected height).\n")
RANGE = 25.0
CARW, CARH = 1.8, 1.5
print(f"{'azimuth':>9} {'u centre':>9} {'w px':>8} {'h px':>8} {'w/h':>7} "
      f"{'edge-h skew %':>14} {'vs boresight w':>15}")
ref = None
for az_deg in (0, 10, 20, 30, 40, 50, 59):
    az = math.radians(az_deg)
    # box centre at range RANGE, azimuth az; rear face spans +-CARW/2 PERPENDICULAR
    # to the ego heading (the world-aligned case, i.e. a car in a parallel lane)
    cxw, czw = RANGE * math.sin(az), RANGE * math.cos(az)
    corners = []
    for sgn in (-1, 1):
        Xc = cxw + sgn * CARW / 2.0     # lateral extent is world-x aligned
        Zc = czw
        top = cyl_project(Xc, CAM_H - CARH, Zc)
        bot = cyl_project(Xc, CAM_H, Zc)
        corners.append((top, bot))
    (tl, bl), (tr, br) = corners
    wpx = abs(tr[0] - tl[0])
    hl = abs(bl[1] - tl[1])
    hr = abs(br[1] - tr[1])
    hpx = (hl + hr) / 2.0
    skew = abs(hr - hl) / hpx * 100.0
    uc = (tl[0] + tr[0]) / 2.0
    if ref is None:
        ref = wpx
    print(f"{az_deg:>7} deg {uc:>9.1f} {wpx:>8.2f} {hpx:>8.2f} {wpx/hpx:>7.3f} "
          f"{skew:>13.2f}% {wpx/ref*100:>14.1f}%")

print()
print("  Same object, PINHOLE frame of the same f_ref, for comparison:")
print(f"{'azimuth':>9} {'u centre':>9} {'w px':>8} {'h px':>8} {'w/h':>7} "
      f"{'vs boresight w':>15}")
refp = None
for az_deg in (0, 10, 20, 30, 40, 45):
    az = math.radians(az_deg)
    cxw, czw = RANGE * math.sin(az), RANGE * math.cos(az)
    pts = []
    for sgn in (-1, 1):
        Xc = cxw + sgn * CARW / 2.0
        pts.append((pin_project(Xc, CAM_H - CARH, czw),
                    pin_project(Xc, CAM_H, czw)))
    (tl, bl), (tr, br) = pts
    wpx = abs(tr[0] - tl[0])
    hpx = (abs(bl[1] - tl[1]) + abs(br[1] - tr[1])) / 2.0
    if refp is None:
        refp = wpx
    print(f"{az_deg:>7} deg {(tl[0]+tr[0])/2:>9.1f} {wpx:>8.2f} {hpx:>8.2f} "
          f"{wpx/hpx:>7.3f} {wpx/refp*100:>14.1f}%")

print()
print("=" * 78)
print("4. OBJECT PIXEL SIZE vs RANGE -- the small-object question")
print("=" * 78)
print("Apparent size on the boresight (best case; uniform in azimuth for cyl).\n")
objs = [("traffic light housing", 0.30, 0.90),
        ("traffic light LAMP (state)", 0.20, 0.20),
        ("pedestrian", 0.50, 1.70),
        ("cyclist", 0.60, 1.70),
        ("car (rear face)", 1.80, 1.50),
        ("lane marking width", 0.15, 0.15),
        ("speed-limit sign face", 0.60, 0.60)]
print(f"{'object':>28} {'size m':>9} " + "".join(f"{z:>8.0f}m" for z in
      (15, 25, 40, 60, 80, 100)))
for name, wm, hm in objs:
    row = f"{name:>28} {wm:>5.2f}x{hm:<3.2f}"
    for Z in (15, 25, 40, 60, 80, 100):
        wpx = F * (2 * math.atan(wm / 2 / Z))     # cyl: u is linear in azimuth
        row += f"{wpx:>9.1f}"
    print(row + "   px WIDE")
print()
print("  (heights are within 2% of these for objects near the boresight row)")
print("  Rule of thumb from the detection literature: <10 px is 'small', "
      "<5 px is at/below\n  the floor for most detectors; a traffic-light STATE "
      "needs the lamp resolved.")

print()
print("=" * 78)
print("5. TOKEN / READOUT ANGULAR FOOTPRINT")
print("=" * 78)
for g, gw in ((4, 4), (4, 8), (4, 10)):
    az = hfov / gw
    el = vfov / g
    # metric width of one azimuth bin at range
    print(f"  readout {g}x{gw}: {az:.2f} deg azimuth/bin, {el:.2f} deg elev/bin"
          f"   -> bin width at 10 m = {2*10*math.tan(math.radians(az/2)):.2f} m,"
          f" at 30 m = {2*30*math.tan(math.radians(az/2)):.2f} m")
print()
print("  For comparison a nuScenes-convention BEV grid of 0.5 m cells at 30 m")
print(f"  range subtends {math.degrees(2*math.atan(0.25/30)):.3f} deg per cell.")

print()
print("=" * 78)
print("6. SENSITIVITY -- camera height assumption")
print("=" * 78)
for ch in (1.2, 1.5, 1.8, 2.1):
    pts = []
    for Z in Zs:
        c = cyl_project(3.5, ch, Z)
        if c and 0 <= c[0] < W and 0 <= c[1] < H:
            pts.append(c)
    print(f"  cam height {ch} m -> lane-line sagitta at X=3.5 m: "
          f"{sagitta(pts):.2f} px ({sagitta(pts)/H*100:.2f}% of frame H)")
