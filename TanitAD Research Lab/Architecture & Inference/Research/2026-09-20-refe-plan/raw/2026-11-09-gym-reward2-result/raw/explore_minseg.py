import glob, os, numpy as np, math
CL={}
for sub in ("cl_fast","cl"):
    for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), sub, "cl_*.npz")):
        z=np.load(f); CL[int(os.path.basename(f)[3:-4])]=(z["pts"],z["s"],float(z["s0"]),float(z["limit"]))
allds=np.concatenate([np.diff(v[1]) for v in CL.values()])
print("segments",len(allds),"zero",(allds<1e-9).sum(),"<1e-3",(allds<1e-3).sum(),"1e-3..0.05",((allds>=1e-3)&(allds<0.05)).sum(),"0.05..0.2",((allds>=0.05)&(allds<0.2)).sum(),">=0.2",(allds>=0.2).sum())
print("pct ds quantiles",np.quantile(allds,[0.001,0.01,0.05,0.5,0.99,1]))
def v5(pts,s,s0,ahead=30,a=2.0):
    m=(s>=s0)&(s<=s0+ahead); h=pts[m,2]
    dh=np.abs(np.diff(np.unwrap(h))); ds=np.maximum(np.diff(s[m]),1e-3); k=np.max(dh/ds); return k
def vB(pts,s,s0,minseg,ahead=30):
    m=(s>=s0)&(s<=s0+ahead); h=np.unwrap(pts[m,2]); ss=s[m]
    keep=[0]
    for i in range(1,len(ss)):
        if ss[i]-ss[keep[-1]]>=minseg: keep.append(i)
    keep=np.array(keep)
    if len(keep)<3: return 0.0
    return float(np.max(np.abs(np.diff(h[keep]))/np.diff(ss[keep])))
for minseg in (0.01,0.05,0.1):
    ks=[];k5=[]
    for idx,(pts,s,s0,L) in CL.items():
        for q in np.arange(s0,min(s[-1],s0+60),1.0):
            ks.append(vB(pts,s,q,minseg)); k5.append(v5(pts,s,q))
    ks=np.array(ks);k5=np.array(k5)
    print(minseg,"B max k",ks.max(),"q99",np.quantile(ks,.99),"frac k>2",(ks>2).mean(),"v5 frac k>2",(k5>2).mean(), "v5 max",k5.max(), "n k_B>0.5",(ks>0.5).sum(), "of",len(ks))
