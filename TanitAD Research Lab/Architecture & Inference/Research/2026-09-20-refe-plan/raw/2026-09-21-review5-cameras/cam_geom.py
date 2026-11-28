import json, math
import numpy as np
d = json.load(open('D:/Projects/TanitAD/data/nuplan_cam_calib.json'))
def q2m_wxyz(q):
    w,x,y,z = q
    return np.array([
      [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
      [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
      [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
print("channels:", sorted(d.keys()))
print()
print(f"{'cam':8s} {'|q|':>6s} {'t (x,y,z) ego':>34s} {'optical axis z_cam->ego':>30s} {'yaw deg':>9s} {'pitch deg':>9s} {'x_cam->ego (right)':>28s}")
rows={}
for ch in sorted(d):
    q = d[ch]['q']; t = np.array(d[ch]['t'])
    R = q2m_wxyz(q)
    ax = R @ np.array([0.,0.,1.])   # camera optical axis (z fwd) in ego
    rt = R @ np.array([1.,0.,0.])   # camera right (x) in ego
    up = R @ np.array([0.,-1.,0.])  # camera up (-y) in ego
    yaw = math.degrees(math.atan2(ax[1], ax[0]))
    pitch = math.degrees(math.asin(max(-1,min(1,ax[2]))))
    rows[ch]=(yaw,ax,up,t)
    print(f"{ch:8s} {np.linalg.norm(q):6.4f} {str(np.round(t,3)):>34s} {str(np.round(ax,3)):>30s} {yaw:9.2f} {pitch:9.2f} {str(np.round(rt,3)):>28s}  up={np.round(up,3)}")
print()
# orthonormality / det check
for ch in sorted(d):
    R=q2m_wxyz(d[ch]['q'])
    print(f"  {ch}: det={np.linalg.det(R):+.6f}  orth_err={np.abs(R@R.T-np.eye(3)).max():.2e}")
