import math
import numpy as np
k1,k2,p1,p2,k3 = -0.3561,0.1725,-0.0021,0.0005,-0.0523     # MEASURED, nuPlan camera.distortion
fx=fy=1545.0; cx,cy=960.0,560.0; W,H=1920,1080
def f(r): return 1 + k1*r**2 + k2*r**4 + k3*r**6
ru = np.linspace(0,2.0,200001); rd = ru*f(ru)
imax = int(np.argmax(rd)); print(f"r_d(r_u) is non-monotonic; peak r_d={rd[imax]:.4f} at r_u={ru[imax]:.4f} -> invert on [0,{ru[imax]:.3f}]")
def undist(x):
    return float(np.interp(x, rd[:imax+1], ru[:imax+1]))
print()
print(f"{'image point':>22s} {'r_obs':>8s} {'assumed ang':>12s} {'true ang':>9s} {'err':>7s} {'lateral err @60m':>17s}")
pts=[("centre",960,560),("h-edge u=0",0,560),("v-edge v=0",960,0),("corner (0,0)",0,0),
     ("corner (1919,1079)",1919,1079),("quarter (480,560)",480,560)]
mx=0
for nm,u,v in pts:
    xd,yd=(u-cx)/fx,(v-cy)/fy; r=math.hypot(xd,yd)
    r_u = undist(r)
    a1=math.degrees(math.atan(r)); a2=math.degrees(math.atan(r_u)); e=a2-a1; mx=max(mx,abs(e))
    lat=60*(math.tan(math.radians(a2))-math.tan(math.radians(a1)))
    print(f"{nm:>22s} {r:8.4f} {a1:11.2f}d {a2:8.2f}d {e:+6.2f}d {lat:+16.2f} m")
xe=abs((0-cx)/fx)
print()
print(f"HFOV, ideal pinhole (what the coverage table used): {2*math.degrees(math.atan(xe)):.2f} deg")
print(f"HFOV, with the DB's Caltech distortion            : {2*math.degrees(math.atan(undist(xe))):.2f} deg")
print(f"max angular ray error over the sampled points     : {mx:.2f} deg")
print()
# CROSS-CAMERA CONSISTENCY: a world point in the F0/L0 overlap, seen by both
import json
cal=json.load(open('D:/Projects/TanitAD/data/nuplan_cam_calib.json'))
def q2m(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
Pw=np.array([25.0, 12.0, 0.0])      # a point ~25.6 deg to the left at 27.7 m -- inside BOTH F0 and L0
print(f"a world point at ego {Pw} (bearing {math.degrees(math.atan2(Pw[1],Pw[0])):.1f} deg) is inside BOTH F0 and L0:")
for ch in ('CAM_F0','CAM_L0'):
    R=q2m(cal[ch]['q']); t=np.array(cal[ch]['t'])
    pc = R.T@(Pw-t)                       # ego -> camera
    xn,yn,z = pc[0]/pc[2], pc[1]/pc[2], pc[2]
    r=math.hypot(xn,yn); s=f(r)
    u,v = fx*xn*s+cx, fy*yn*s+cy          # the DISTORTED pixel the sensor really writes
    # what REFe's ideal-pinhole frustum reconstructs from that pixel, at the TRUE depth
    xr,yr = (u-cx)/fx, (v-cy)/fy
    p_rec_cam = np.array([xr*z, yr*z, z])
    p_rec_ego = R@p_rec_cam + t
    print(f"   {ch}: true depth {z:6.2f} m, observed pixel ({u:7.1f},{v:7.1f})  "
          f"REFe reconstructs ego {np.round(p_rec_ego,2)}  error {np.linalg.norm(p_rec_ego-Pw):.2f} m")
