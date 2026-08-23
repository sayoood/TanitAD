"""H2 crux: per-camera frustum membership of obstacle.offline tracks vs the MODEL's front crop."""
import pandas as pd, numpy as np, zipfile, io, glob, math, os, json, sys

DR = r"C:\Users\Admin\tanitad-data\physicalai"
CAMS = ["camera_front_wide_120fov","camera_cross_left_120fov","camera_cross_right_120fov",
        "camera_front_tele_30fov","camera_rear_left_70fov","camera_rear_right_70fov","camera_rear_tele_30fov"]
F_REF, SIZE = 266.0, 256
CANON_HALF = math.atan((SIZE/2)/F_REF)          # 25.697 deg -> the model's retained half-angle

def poly_r(poly, th):
    r = np.zeros_like(np.asarray(th, dtype=float))
    for c in reversed(poly): r = r*th + c
    return r
def theta_of_r(poly, rt, hi=2.2):
    lo, high = 0.0, hi
    for _ in range(60):
        mid = .5*(lo+high)
        if float(poly_r(poly, mid)) < rt: lo = mid
        else: high = mid
    return .5*(lo+high)
def q2R(qx,qy,qz,qw):
    n=math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw); qx,qy,qz,qw=qx/n,qy/n,qz/n,qw/n
    return np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                     [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                     [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])

# ---- calibration (all local chunks) ----
CI = pd.concat([pd.read_parquet(f) for f in glob.glob(DR+r"\calibration\camera_intrinsics\*.parquet")]).reset_index()
SE = pd.concat([pd.read_parquet(f) for f in glob.glob(DR+r"\calibration\sensor_extrinsics\*.parquet")]).reset_index()
CI = CI.set_index(["clip_id","camera_name"]); SE = SE.set_index(["clip_id","sensor_name"])
calib_clips = set(CI.index.get_level_values(0))

def clip_rig(clip):
    """-> {cam: (Rt, t, poly, cx, cy, W, H, th_max, c_half)} ; Rt maps rig->cam."""
    out={}
    for cam in CAMS:
        try: i = CI.loc[(clip,cam)]; e = SE.loc[(clip,cam)]
        except KeyError: return None
        poly=(float(i.fw_poly_0),float(i.fw_poly_1),float(i.fw_poly_2),float(i.fw_poly_3),float(i.fw_poly_4))
        R=q2R(float(e.qx),float(e.qy),float(e.qz),float(e.qw))
        t=np.array([float(e.x),float(e.y),float(e.z)])
        rx=min(float(i.cx), float(i.width)-float(i.cx)); ry=min(float(i.cy), float(i.height)-float(i.cy))
        out[cam]=dict(Rt=R.T, t=t, poly=poly, cx=float(i.cx), cy=float(i.cy),
                      W=float(i.width), H=float(i.height),
                      th_h=theta_of_r(poly,rx), th_v=theta_of_r(poly,ry),
                      c_half=float(poly_r(poly, CANON_HALF)))   # model-crop half-side in native px
    return out

def project(P_rig, K):
    """P_rig [N,3] -> (u,v,theta) in that camera."""
    Pc = (P_rig - K["t"]) @ K["Rt"].T
    x,y,z = Pc[:,0],Pc[:,1],Pc[:,2]
    rho = np.hypot(x,y); th = np.arctan2(rho,z)
    r = poly_r(K["poly"], th)
    s = np.where(rho>1e-9, r/np.maximum(rho,1e-9), 0.0)
    return K["cx"]+x*s, K["cy"]+y*s, th

def in_frame(u,v,th,K):
    return (u>=0)&(u<K["W"])&(v>=0)&(v<K["H"])&(th<max(K["th_h"],K["th_v"])*1.02)

def in_model_crop(u,v,K):
    """The 256x256 canonical crop actually fed to the encoder: side 2*c_half centered on (cx,cy)."""
    return (np.abs(u-K["cx"])<=K["c_half"])&(np.abs(v-K["cy"])<=K["c_half"])

def resample_tracks(df, grid_us, max_gap_us=500_000):
    """Per-track linear resample onto grid with an explicit max-gap guard (the documented trap)."""
    recs=[]
    for tid, g in df.groupby("track_id", sort=False):
        g=g.sort_values("timestamp_us")
        ts=g.timestamp_us.to_numpy(np.int64)
        if len(ts)<2: continue
        ok = (grid_us>=ts[0]-max_gap_us)&(grid_us<=ts[-1]+max_gap_us)
        gg = grid_us[ok]
        if not len(gg): continue
        idx=np.searchsorted(ts,gg); idx=np.clip(idx,1,len(ts)-1)
        gap=np.minimum(np.abs(gg-ts[idx-1]),np.abs(ts[idx]-gg))
        keep=gap<=max_gap_us
        if not keep.any(): continue
        gg=gg[keep]
        cx=np.interp(gg,ts,g.center_x.to_numpy(float))
        cy=np.interp(gg,ts,g.center_y.to_numpy(float))
        cz=np.interp(gg,ts,g.center_z.to_numpy(float))
        sx=np.interp(gg,ts,g.size_x.to_numpy(float))
        recs.append(pd.DataFrame(dict(t_us=gg,track_id=tid,X=cx,Y=cy,Z=cz,L=sx,
                                      cls=g.label_class.iloc[0])))
    return pd.concat(recs) if recs else None

def run_chunk(zpath, limit=None):
    z=zipfile.ZipFile(zpath); names=[n for n in z.namelist() if n.endswith(".parquet")]
    rows=[]
    for k,n in enumerate(names):
        if limit and k>=limit: break
        clip=n.split("/")[-1].split(".")[0]
        if clip not in calib_clips: continue
        K=clip_rig(clip)
        if K is None: continue
        df=pd.read_parquet(io.BytesIO(z.read(n)))
        if not len(df): continue
        t0,t1=int(df.timestamp_us.min()),int(df.timestamp_us.max())
        grid=np.arange(t0,t1,100_000,dtype=np.int64)          # 10 Hz
        if len(grid)<50: continue
        R=resample_tracks(df,grid)
        if R is None or not len(R): continue
        P=R[["X","Y","Z"]].to_numpy(float)
        vis={}
        for cam in CAMS:
            u,v,th=project(P,K[cam]); vis[cam]=in_frame(u,v,th,K[cam])
            if cam=="camera_front_wide_120fov":
                R["in_model"]=in_model_crop(u,v,K[cam])&vis[cam]
        for cam in CAMS: R["v_"+cam]=vis[cam]
        R["rng"]=np.linalg.norm(P[:,:2],axis=1)
        R["az"]=np.degrees(np.arctan2(P[:,1],P[:,0]))
        R["clip_id"]=clip
        rows.append(R)
    return pd.concat(rows) if rows else None

if __name__=="__main__":
    chunks=sorted(glob.glob(DR+r"\labels\obstacle.offline\*.zip"))
    nch=int(sys.argv[1]) if len(sys.argv)>1 else 3
    lim=int(sys.argv[2]) if len(sys.argv)>2 else 40
    allr=[]
    for zp in chunks[:nch]:
        r=run_chunk(zp,limit=lim)
        if r is not None:
            r["chunk"]=os.path.basename(zp).split("_")[-1].split(".")[0]
            allr.append(r); print(f"[{os.path.basename(zp)}] clips={r.clip_id.nunique()} rows={len(r)}",flush=True)
    A=pd.concat(allr)
    A.to_parquet(r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\vis.parquet")
    print("\n=== TOTAL: clips %d  agent-frames %d ==="%(A.clip_id.nunique(),len(A)))
