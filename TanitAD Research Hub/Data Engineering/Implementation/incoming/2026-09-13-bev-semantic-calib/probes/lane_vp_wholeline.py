"""Lane vanishing point vs ribbon vanishing point, per frame, from WHOLE lines.

Every per-row ridge detector in this investigation has failed (seam, foliage,
dashed-line wander, nearest-ridge). A lane line is a long straight feature, so fit
it as one: Hough segments, grouped by side, each side fitted as a single line
u = a + b*v through ALL its segment endpoints (length-weighted). The two lines meet
at the lane VP. The ribbon's own direction meets the horizon at the path VP.
Delta-u between them, over f, is the angle between the drawn path and the lane.
No height, no metric scale, no horizon-row accuracy needed for the ANGLE.
"""
import sys, pathlib, numpy as np, cv2
B="/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib"
sys.path.insert(0,B+"/probes"); sys.path.insert(0,B)
import run_real as RR, bev_calib as BC, lag_scale as LS
FX=1533.0; Hm=1.586; HZ=463.0; HW=1.855/2

def params(yaw=-6.40, lat=-0.260, hz=HZ):
    P=dict(RR.NOMINAL); P.update(fx=FX,height=Hm,lateral=lat,yaw=np.deg2rad(yaw))
    P["pitch"]=LS.pitch_for_horizon(P,hz); return P

def ribbon_uv(P, rec, xs):
    px=np.asarray(rec["x"],float); py=np.asarray(rec["y"],float); m=px>0
    LON=float(P["longitudinal"]); out=[]
    for x in xs:
        xc=x+LON
        if xc>px[m].max(): break
        yc=float(np.interp(xc,px[m],py[m]))
        uv=BC.project_ground(np.array([[xc,yc+HW],[xc,yc-HW],[xc,yc]],float),P)
        if np.isfinite(uv).all(): out.append(uv)
    return np.asarray(out)          # (n,3,2): left edge, right edge, centre

def lane_lines(img, rib, v_lo=500, v_hi=820):
    """Fit the left and right lane boundaries as single image lines u=a+b*v."""
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # paint is BRIGHT and thin: top-hat isolates it from asphalt shading
    th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_RECT,(31,1)))
    _,bw=cv2.threshold(th,40,255,cv2.THRESH_BINARY)
    bw[:v_lo]=0; bw[v_hi:]=0
    segs=cv2.HoughLinesP(bw,1,np.pi/360,40,minLineLength=35,maxLineGap=25)
    if segs is None: return None,None,bw
    # ribbon centre as a function of row, to split left/right
    cu=rib[:,2,0]; cv_=rib[:,2,1]; o=np.argsort(cv_)
    def centre_at(v): return float(np.interp(v,cv_[o],cu[o]))
    L=[];R=[]
    for s in segs[:,0]:
        u1,v1,u2,v2=map(float,s)
        if abs(v2-v1)<8: continue                      # near-horizontal: not a lane line
        b=(u2-u1)/(v2-v1)                              # du/dv
        vm=0.5*(v1+v2); um=0.5*(u1+u2)
        side = um < centre_at(vm)
        # a left boundary leans right going UP (du/dv<0); a right one leans left
        if side and b<-0.2: L.append((u1,v1,u2,v2))
        elif (not side) and b>0.2: R.append((u1,v1,u2,v2))
    def fit(S, pick):
        if len(S)<2: return None
        # keep the segments NEAREST the ribbon on that side (the ego lane's own
        # boundary), judged at a common reference row, then fit one line.
        vr=720.0
        ext=[(s,(s[0]+(vr-s[1])*(s[2]-s[0])/(s[3]-s[1]))) for s in S]
        ref=centre_at(vr)
        ext=[(s,u) for s,u in ext if (u<ref if pick=="L" else u>ref)]
        if not ext: return None
        near=max(ext,key=lambda t:t[1]) if pick=="L" else min(ext,key=lambda t:t[1])
        u0=near[1]
        keep=[s for s,u in ext if abs(u-u0)<45]         # same line as the nearest one
        V=[];U=[];W=[]
        for (u1,v1,u2,v2) in keep:
            ln=np.hypot(u2-u1,v2-v1)
            V+= [v1,v2]; U+=[u1,u2]; W+=[ln,ln]
        A=np.stack([np.ones(len(V)),V],1)*np.sqrt(np.array(W))[:,None]
        a,b=np.linalg.lstsq(A,np.array(U)*np.sqrt(np.array(W)),rcond=None)[0]
        return float(a),float(b),len(keep)
    return fit(L,"L"),fit(R,"R"),bw

def analyse(F, yaw=-6.40, lat=-0.260, draw=None):
    recs={int(r["frame"]):r for r in RR.load_records(RR.RUN)}
    img=cv2.imread(str(RR.RUN/"frames"/f"{F:06d}.jpg"),cv2.IMREAD_COLOR)
    P=params(yaw,lat); rib=ribbon_uv(P,recs[F],np.arange(7.,60.,1.0))
    fl,fr,bw=lane_lines(img,rib)
    res=dict(frame=F)
    if fl and fr:
        (al,bl,nl),(ar,br,nr)=fl,fr
        vvp=(ar-al)/(bl-br); uvp=al+bl*vvp            # lane VP
        res.update(lane_vp=(uvp,vvp),nl=nl,nr=nr)
        # ribbon centre-line direction extended to the lane-VP row
        cu=rib[:,2,0]; cv_=rib[:,2,1]
        far=cv_<np.percentile(cv_,60)                 # far half of the ribbon
        if far.sum()>=3:
            k=np.polyfit(cv_[far],cu[far],1)
            u_path=float(np.polyval(k,vvp))
            res["path_u_at_vp"]=u_path
            res["delta_u"]=u_path-uvp
            res["angle_deg"]=float(np.rad2deg(np.arctan((u_path-uvp)/FX)))
        # lane centre vs ribbon centre at two rows, in lane-width units (scale-free)
        for vr in (700.0, 560.0):
            uL=al+bl*vr; uR=ar+br*vr
            o=np.argsort(cv_); uc=float(np.interp(vr,cv_[o],cu[o]))
            res[f"offset_frac_{int(vr)}"]=(uc-0.5*(uL+uR))/(uR-uL)   # + = ribbon RIGHT
    if draw is not None:
        vis=img.copy()
        for pts,col in ((rib[:,0],(0,140,255)),(rib[:,1],(0,140,255)),(rib[:,2],(0,140,255))):
            cv2.polylines(vis,[pts.astype(np.int32)],False,col,3,cv2.LINE_AA)
        for f_,col in ((fl,(255,0,255)),(fr,(255,0,255))):
            if f_:
                a,b,_=f_
                cv2.line(vis,(int(a+b*820),820),(int(a+b*470),470),col,4,cv2.LINE_AA)
        if "lane_vp" in res:
            cv2.circle(vis,(int(res["lane_vp"][0]),int(res["lane_vp"][1])),12,(255,0,255),3)
        if "path_u_at_vp" in res:
            cv2.circle(vis,(int(res["path_u_at_vp"]),int(res["lane_vp"][1])),12,(0,140,255),3)
        cv2.rectangle(vis,(0,0),(1920,120),(0,0,0),-1)
        cv2.putText(vis,f"frame {F}  yaw {yaw:+.2f}  MAGENTA = lane lines fitted as whole lines",
                    (20,45),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,255),2)
        txt=(f"ORANGE = drawn ribbon.  path-vs-lane angle {res.get('angle_deg',float('nan')):+.2f} deg,"
             f"  offset @row700 {res.get('offset_frac_700',float('nan')):+.3f} lane widths")
        cv2.putText(vis,txt,(20,95),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,140,255),2)
        cv2.imwrite(draw,vis)
    return res

if __name__=="__main__":
    F=int(sys.argv[1]); out=sys.argv[2]
    r=analyse(F,draw=out)
    for k,v in r.items(): print(f"  {k}: {v}")
