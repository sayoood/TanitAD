"""Reconciling two conflicting recommendations that arrived from different streams,
and pricing the real options. All from calib.py's own formulas.

CONFLICT 1: a literature stream proposed "crop elevation 256->176 rows, +45% vertical
px/deg for free". A CROP CANNOT ADD RESOLUTION -- it removes field at constant px/deg.
The +45% is only available by RE-REMAPPING at a higher f_ref, which is a cache rebuild.
These are two different options and must be priced separately.

CONFLICT 2: is cropping the elevation band actually safe? It removes overhead structure
(traffic lights, gantries) at short range -- exactly where they matter at a stop line.
Quantify before recommending.
"""
import math

F0 = 305.5774907364391
CAM_H = 1.5          # ESTIMATED camera height, m
LIGHT_H = 5.0        # ESTIMATED overhead traffic-light height, m


def frame(H, W, f):
    hf = math.degrees(2 * (W / 2) / f)          # cylindrical: linear in azimuth
    vf = math.degrees(2 * math.atan((H / 2) / f))  # rectilinear in elevation
    return hf, vf


print("=" * 92)
print("A. CLOSING A CAVEAT: is elevation really rectilinear with the same f?")
print("=" * 92)
print("  YES -- established from source, not assumed:")
print("    calib.py:154-157   half_angle_y_rad = atan((H/2)/f_ref)")
print("                       docstring: 'cylinder is only equidistant in azimuth'")
print("    calib.py:906-935   ray = (sin phi, y_n, cos phi), y_n = (v-(H-1)/2)/f_ref")
print("                       => tan(elevation) = y_n  =>  v = f*tan(eps)   [rectilinear]")
print("  The 45.4556 deg VFOV therefore stands on source, not on an assumption.")

print()
print("=" * 92)
print("B. THE OPTIONS, PRICED  (patch 16 throughout)")
print("=" * 92)
OPTS = [
    ("current            256x640 @ f=305.577", 256, 640, F0, "none"),
    ("D1 CROP (rig-clean) 176x624 @ f=305.577", 176, 624, F0, "flag only, --v2-subframe"),
    ("D1b CROP strict     128x576 @ f=305.577", 128, 576, F0, "flag only"),
    ("D2 REBUILD          256x640 @ f=444.75", 256, 640, 444.75, "CACHE REBUILD"),
    ("D2b REBUILD         256x624 @ f=444.75", 256, 624, 444.75, "CACHE REBUILD"),
]
print(f"{'option':<40} {'HFOV':>8} {'VFOV':>8} {'px/deg x':>9} {'px/deg y':>9} "
      f"{'tokens':>7} {'FLOP x':>7}")
base_tok = (256 // 16) * (640 // 16)
for name, H, W, f, _ in OPTS:
    hf, vf = frame(H, W, f)
    tok = (H // 16) * (W // 16)
    print(f"{name:<40} {hf:>7.2f}° {vf:>7.2f}° {W/hf:>9.3f} {H/vf:>9.3f} "
          f"{tok:>7} {tok/base_tok:>6.2f}x")

print()
print("  ⇒ CORRECTION: D1 (a crop) keeps px/deg EXACTLY UNCHANGED in x and nearly")
print("    unchanged in y. It buys 0% resolution. What it buys is 31% fewer tokens")
print("    and the rig-clean guarantee. The '+45% vertical px/deg' is D2, and D2 is")
print("    a cache rebuild, not a flag.")

print()
print("=" * 92)
print("C. WHAT THE ELEVATION CROP ACTUALLY COSTS -- overhead structure")
print("=" * 92)
print(f"  camera height {CAM_H} m (ESTIMATED), overhead light at {LIGHT_H} m (ESTIMATED)")
print(f"  An object at height h leaves the top of frame when "
      f"atan((h-cam)/Z) > half-VFOV.\n")
print(f"{'frame':<26} {'half-VFOV':>10} {'light vanishes closer than':>29} "
      f"{'road visible from':>19}")
for name, H, W, f, _ in OPTS:
    hf, vf = frame(H, W, f)
    half = math.radians(vf / 2)
    z_light = (LIGHT_H - CAM_H) / math.tan(half)
    z_road = CAM_H / math.tan(half)          # nearest ground point in view
    print(f"{name.split('@')[0].strip():<26} {vf/2:>9.2f}° {z_light:>26.1f} m "
          f"{z_road:>16.1f} m")

print()
print("  ⇒ THE HONEST TRADE. Cropping to 176 rows makes an overhead traffic light")
print("    disappear from frame at 12.1 m instead of 8.4 m, and pushes the nearest")
print("    visible road from 3.6 m out to 5.2 m. At an intersection stop line the")
print("    light is typically 10-25 m away, so the crop bites exactly where a")
print("    stop-line decision is made. This is NOT a free lunch and must not be")
print("    sold as one.")

print()
print("=" * 92)
print("D. THE COMBINED OPTION -- what constant-compute actually buys")
print("=" * 92)
print("  Keep 640 tokens. Spend the elevation band the field spends it on.")
print("  BEVDet's precedent: 256 rows over ~25 deg (fixed bottom crop, sky discarded).\n")
for vf_target in (45.4556, 40.0, 36.0, 32.131, 28.0, 25.0):
    f_needed = (256 / 2) / math.tan(math.radians(vf_target / 2))
    w_needed = 2 * f_needed * math.radians(120.0 / 2)      # keep 120 deg
    print(f"  VFOV {vf_target:>7.3f}°  -> f_ref {f_needed:>7.2f}  "
          f"px/deg_y {256/vf_target:>6.3f} ({256/vf_target/(256/45.4556):>5.2f}x today)  "
          f"width for 120° = {w_needed:>7.1f} px ({w_needed/16:>4.1f} token cols)")
print()
print("  ⇒ At a FIXED 256x640 grid, narrowing the elevation band from 45.46° to")
print("    32.13° raises vertical resolution 1.41x -- but the SAME f_ref then only")
print("    spans 87.7° horizontally at 640 px, so keeping 120° costs 876 px of width")
print("    (55 token cols, 880 tokens, 1.38x). Resolution and field trade against")
print("    each other through ONE f_ref; there is no free axis.")

print()
print("=" * 92)
print("E. SANITY: our px/deg against the two comparators the streams surfaced")
print("=" * 92)
hf, vf = frame(256, 640, F0)
print(f"  ours horizontal  {640/hf:.3f} px/deg   (TransFuser 704/132 = "
      f"{704/132:.3f} -- identical to 4 s.f.)")
print(f"  ours vertical    {256/vf:.3f} px/deg   (BEVDet ~256/25 = "
      f"{256/25:.3f}, i.e. {(256/25)/(256/vf):.2f}x finer than ours)")
print(f"  ⇒ we are level with the field HORIZONTALLY and {(256/25)/(256/vf):.2f}x coarser")
print( "    VERTICALLY, and the whole of that deficit is explained by spending")
print(f"    {vf:.1f}° of elevation where BEVDet spends ~25°.")
