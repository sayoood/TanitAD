import sys, numpy as np
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import bev_calib as BC, lag_scale as LS, run_real as RR

recs = RR.load_records(RR.RUN)
anchors = RR.build_anchors(recs, RR.RUN, 3, 1.5, 0.15, 6000)
grid = BC.BevGrid(x_range=(6.0,40.0), y_range=(-8.0,8.0), cell=0.06)
P = dict(RR.NOMINAL); P.update(yaw=np.deg2rad(-7.01), height=1.03, fx=1478.3)
P["pitch"] = LS.pitch_for_horizon(P, 523.4)
print(f"horizon {LS.horizon_row(P):.1f}  f*h assumed {1478.3*1.03:.1f}")

obs, poses, fno = anchors[0]
for k in (1, 2, 4):
    D = float(np.hypot(poses[k][0]-poses[0][0], poses[k][1]-poses[0][1]))
    A = LS.highpass(LS.own_frame_bev(obs[0], P, grid), grid.cell)
    B = LS.highpass(LS.own_frame_bev(obs[k], P, grid, psi=poses[k][2]-poses[0][2]), grid.cell)
    nx = A.shape[0]
    i_hi = min(nx//2, int(round(16.0/grid.cell)))
    prof = LS._ncc_profile(A, B, 0, i_hi, int(round(0.8/grid.cell)))
    lags = np.arange(0, i_hi+1)*grid.cell
    print(f"\n=== D = {D:.2f} m   (s=1 would put the peak at {D:.2f} m) ===")
    st = max(1,int(round(0.4/grid.cell)))
    for a in range(0, len(prof), st):
        m = ""
        if abs(lags[a]-D) < 0.25: m = "  <== s=1"
        if a == int(np.argmax(prof))//st*st: m += "  <== ARGMAX"
        print(f"  lag {lags[a]:6.2f}  r {prof[a]:+.4f} {'#'*int(max(0,prof[a])*70)}{m}")
    print(f"  argmax {lags[np.argmax(prof)]:.2f} m -> s {lags[np.argmax(prof)]/D:.3f}")
