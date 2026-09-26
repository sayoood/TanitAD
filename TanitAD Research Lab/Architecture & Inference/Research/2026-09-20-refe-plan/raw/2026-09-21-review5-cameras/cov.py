import json, math, itertools
import numpy as np
d = json.load(open('D:/Projects/TanitAD/data/nuplan_cam_calib.json'))
def q2m(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
yaw={}
for ch in d:
    ax=q2m(d[ch]['q'])@np.array([0.,0.,1.])
    yaw[ch]=math.degrees(math.atan2(ax[1],ax[0]))
K=d['CAM_F0']['K']; fx=K[0][0]; cx=K[0][2]
hfov=2*math.degrees(math.atan(cx/fx))
print(f"fx={fx} cx={cx}  ->  HFOV = 2*atan(cx/fx) = {hfov:.3f} deg  (pinhole; nuPlan K is a pinhole matrix)")
print(f"  half-FOV {hfov/2:.3f} deg\n")
grid=np.arange(-180,180,0.05)
def covered(chs):
    m=np.zeros_like(grid,dtype=bool)
    for ch in chs:
        dd=(grid-yaw[ch]+180)%360-180
        m |= np.abs(dd)<=hfov/2
    return m
best=[]
for combo in itertools.combinations(sorted(d),4):
    m=covered(combo)
    frac=m.mean()
    # largest contiguous gap
    un=~m
    # circular run length
    gaps=[];run=0
    ext=np.concatenate([un,un])
    mx=0;run=0
    for v in ext:
        run = run+1 if v else 0
        mx=max(mx,run)
    mx=min(mx,len(grid))
    best.append((frac, mx*0.05, combo))
best.sort(key=lambda r:(-r[0], r[1]))
print(f"{'rank':>4s} {'coverage':>9s} {'max gap':>8s}  cameras (yaw deg)")
for i,(f,g,c) in enumerate(best[:10],1):
    print(f"{i:4d} {100*f:8.2f}% {g:7.1f}d  {', '.join(f'{ch}({yaw[ch]:+.0f})' for ch in c)}")
print("  ...")
ours=('CAM_F0','CAM_L0','CAM_R0','CAM_B0')
rank=[i for i,(f,g,c) in enumerate(best,1) if set(c)==set(ours)][0]
f,g,c=[r for r in best if set(r[2])==set(ours)][0]
print(f"\nOURS  CAM_F0/L0/R0/B0 : coverage {100*f:.2f}%  max contiguous gap {g:.1f} deg   RANK {rank} of {len(best)}")
for alt in [('CAM_F0','CAM_L1','CAM_R1','CAM_B0'),('CAM_F0','CAM_L2','CAM_R2','CAM_B0'),('CAM_F0','CAM_L1','CAM_R1','CAM_L2')]:
    f2,g2,c2=[r for r in best if set(r[2])==set(alt)][0]
    rk=[i for i,(a,b,cc) in enumerate(best,1) if set(cc)==set(alt)][0]
    print(f"ALT   {'/'.join(x[4:] for x in alt):18s}: coverage {100*f2:.2f}%  max gap {g2:.1f} deg   rank {rk}")
m=covered(ours)
un=~m
# list gap intervals for ours
ivs=[];s=None
for i,v in enumerate(un):
    if v and s is None: s=grid[i]
    if not v and s is not None: ivs.append((s,grid[i])); s=None
if s is not None: ivs.append((s,grid[-1]))
print("\nOURS uncovered azimuth intervals (ego yaw, 0=forward, + = left):")
for a,b in ivs: print(f"   [{a:+7.1f} , {b:+7.1f}]  width {b-a:6.1f} deg")
print(f"   total uncovered {100*un.mean():.2f} % of 360")
