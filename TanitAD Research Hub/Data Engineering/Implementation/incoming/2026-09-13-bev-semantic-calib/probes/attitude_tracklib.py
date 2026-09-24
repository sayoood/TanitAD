import sys, json, numpy as np, cv2
S="/tmp/claude-0/-home-user-TanitAD/d367c501-690c-51f0-8672-b9ccb83dd3ca/scratchpad"
sys.path.insert(0,S)
B="/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib"
sys.path.insert(0,B+"/probes"); sys.path.insert(0,B)
import lane_vp_wholeline as V, run_real as RR, bev_calib as BC, lag_scale as LS
recs={int(r["frame"]):r for r in RR.load_records(RR.RUN)}
have={int(p.stem) for p in (RR.RUN/"frames").glob("*.jpg")}
FX=V.FX; Hm=V.Hm; LAT=-0.260

def P_of(yaw,hz):
    return V.params(yaw,LAT,hz)

def yaw_for_u(u_target,hz):
    lo,hi=-12.0,0.0
    def u_of(y):
        uv=BC.project_ground(np.array([[1e5,0.0]]),P_of(y,hz)); return float(uv[0,0])
    ulo,uhi=u_of(lo),u_of(hi)
    for _ in range(50):
        m=0.5*(lo+hi); um=u_of(m)
        if (um-u_target)*(ulo-u_target)<=0: hi,uhi=m,um
        else: lo,ulo=m,um
    return 0.5*(lo+hi)


def smooth(tq, ts, vs, half=3.0, max_half=8.0, need=3):
    out=[]
    for x in tq:
        h=half
        while True:
            m=np.abs(ts-x)<=h
            if m.sum()>=need or h>=max_half: break
            h+=1.0
        out.append(np.median(vs[m]) if m.sum() else np.nan)
    out=np.array(out)
    ok=np.isfinite(out)
    out=np.interp(tq,tq[ok],out[ok])
    k=np.exp(-0.5*(np.arange(-6,7)/3.0)**2); k/=k.sum()          # ~1.5 s gaussian
    pad=np.pad(out,6,mode="edge"); return np.convolve(pad,k,"valid")


def lat40(f):
    r=recs[f]; px=np.asarray(r["x"],float); py=np.asarray(r["y"],float); m=px>0
    return abs(float(np.interp(40.,px[m],py[m]))) if m.sum()>2 else 9.

def angle(F,yaw,hz):
    img=cv2.imread(str(RR.RUN/"frames"/f"{F:06d}.jpg"),cv2.IMREAD_COLOR)
    P=P_of(yaw,hz); rib=V.ribbon_uv(P,recs[F],np.arange(7.,60.,1.0))
    fl,fr,_=V.lane_lines(img,rib)
    if not fl or not fr: return None
    (al,bl,nl),(ar,br,nr)=fl,fr
    if nl<6 or nr<3 or not (-3.5<bl<-0.6 and 0.6<br<3.5): return None
    vvp=(ar-al)/(bl-br); uvp=al+bl*vvp
    if not (440<vvp<520) or not (380<(ar+br*720)-(al+bl*720)<750): return None
    cu=rib[:,2,0]; cvv=rib[:,2,1]; far=cvv<np.percentile(cvv,60)
    k=np.polyfit(cvv[far],cu[far],1)
    return float(np.rad2deg(np.arctan((np.polyval(k,vvp)-uvp)/FX)))

def paint_w(F,yaw,hz):
    g=cv2.imread(str(RR.RUN/"frames"/f"{F:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
    P=P_of(yaw,hz); LON=float(P["longitudinal"]); ws=[]
    for x in (8.,10.,12.,15.,20.):
        uv=BC.project_ground(np.array([[x+LON,0.0]]),P); c0=float(uv[0,0]); v=float(uv[0,1]); vi=int(round(v))
        if not (0<=vi<g.shape[0]) or v-hz<=1: continue
        mpp=Hm/(v-hz); lo=max(0,int(c0-3.6/mpp)); hi=min(g.shape[1],int(c0-0.9/mpp))
        if hi-lo<30: continue
        seg=g[vi,lo:hi].astype(np.float32); base=float(np.median(seg))
        if seg.max()-base<45: continue
        idx=np.flatnonzero(seg>base+0.5*(seg.max()-base))
        brk=np.flatnonzero(np.diff(idx)>1); st=np.r_[0,brk+1]; en=np.r_[brk,len(idx)-1]
        kk=int(np.argmax(idx[en]-idx[st])); wpx=idx[en[kk]]-idx[st[kk]]+1
        if 1<=wpx<=80: ws.append(wpx*mpp)
    return float(np.median(ws)) if ws else None

