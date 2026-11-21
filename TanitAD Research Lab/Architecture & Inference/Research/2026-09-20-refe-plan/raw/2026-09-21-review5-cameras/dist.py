"""How wrong is an IDEAL-PINHOLE unprojection on nuPlan's DISTORTED images?
distortion MEASURED from the camera table: [k1,k2,p1,p2,k3] = [-0.3561,0.1725,-0.0021,0.0005,-0.0523]
OpenCV model: x_d = x_n*(1+k1 r^2 + k2 r^4 + k3 r^6) (+ tangential, small here)."""
import math
import numpy as np
from scipy.optimize import brentq
k1,k2,p1,p2,k3 = -0.3561,0.1725,-0.0021,0.0005,-0.0523
fx=fy=1545.0; cx,cy=960.0,560.0; W,H=1920,1080
def f(r): return 1 + k1*r**2 + k2*r**4 + k3*r**6
def undistort_r(rd):
    return brentq(lambda ru: ru*f(ru)-rd, 0, 3.0)
print("radial distortion factor f(r) over the image radius:")
for ru in (0.0,0.2,0.4,0.6,0.8,1.0):
    print(f"   r_undist={ru:.2f} -> f={f(ru):.4f}  r_dist={ru*f(ru):.4f}")
print()
print(f"{'image point':>22s} {'r_observed':>11s} {'angle ASSUMED':>14s} {'angle TRUE':>11s} {'error':>8s} {'lateral err @60m':>17s}")
pts = [("centre",960,560),("h-edge u=0",0,560),("h-edge u=1919",1919,560),
       ("v-edge v=0",960,0),("corner (0,0)",0,0),("corner (1919,1079)",1919,1079),
       ("mid-radius",240,560)]
maxang=0
for nm,u,v in pts:
    xd,yd=(u-cx)/fx,(v-cy)/fy
    rd=math.hypot(xd,yd)
    ru=undistort_r(rd) if rd>0 else 0.0
    a_ass=math.degrees(math.atan(rd)); a_true=math.degrees(math.atan(ru))
    err=a_true-a_ass; maxang=max(maxang,abs(err))
    lat=60*math.tan(math.radians(a_true))-60*math.tan(math.radians(a_ass))
    print(f"{nm:>22s} {rd:11.4f} {a_ass:13.2f}d {a_true:10.2f}d {err:+7.2f}d {lat:+16.2f} m")
# true horizontal half-FOV
xd_edge=(0-cx)/fx; yd=0.0
ru=undistort_r(abs(xd_edge))
print()
print(f"HFOV from the intrinsics alone (ideal pinhole): {2*math.degrees(math.atan(abs(xd_edge))):.2f} deg")
print(f"HFOV with the DB's own distortion applied     : {2*math.degrees(math.atan(ru)):.2f} deg")
print(f"  => the pinhole HFOV UNDERSTATES the real one by {2*math.degrees(math.atan(ru))-2*math.degrees(math.atan(abs(xd_edge))):.2f} deg")
print(f"  max angular ray error over the sampled points: {maxang:.2f} deg")
