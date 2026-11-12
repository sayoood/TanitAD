import json, glob, numpy as np, collections
rows=[]
for f in ["tf2/rows/tf2_s0.jsonl","tf2/rows/tf2_s1.jsonl","tf2/rows/tf2_s2.jsonl","tf2/rows/tf2_s2r0.jsonl","tf2/rows/tf2_s2r1.jsonl","tf2b/rows/tf2b_s0.jsonl","tf2b/rows/tf2b_s1.jsonl"]:
    for ln in open(f,encoding="utf-8"):
        try: r=json.loads(ln)
        except: continue
        if "error" in r or "Tc20s_tracked" not in r: continue
        r["_src"]="tf2b" if "tf2b" in f else "tf2"
        rows.append(r)
d={}
for r in rows: d[(r["log"],r["token"])]=r
rows=list(d.values())
print("rows",len(rows), collections.Counter(r["_src"] for r in rows))
arms=["T_tracked","Tc20s_tracked","H_tracked","T_transparent","T_sub20_tracked"]
for a in arms:
    ep=np.array([r[a]["ep"] for r in rows]); mult=np.array([r[a]["mult"] for r in rows]); pd=np.array([r[a]["pdms"] for r in rows])
    alive=mult>0
    print(f"{a:18s} PDMS {100*pd.mean():.2f} EP {100*ep.mean():.2f} zero-mult {100*(~alive).mean():.2f}% EP_alive {100*ep[alive].mean():.2f} EP<1 alive {100*(ep[alive]<0.9999).mean():.1f}% EPloss_from_zero {100*(1-ep)[~alive].sum()/len(ep):.2f} EPloss_alive {100*(1-ep)[alive].sum()/len(ep):.2f}")
v0=np.array([r["info"]["v0"] for r in rows])
bands=[(0,0.5),(0.5,3),(3,6),(6,9),(9,12),(12,40)]
print("band n | EP alive Tc20s | H | Ttransp | loss share Tc20s (EP pts of total)")
tc=np.array([r["Tc20s_tracked"]["ep"] for r in rows]); tcm=np.array([r["Tc20s_tracked"]["mult"] for r in rows])
h=np.array([r["H_tracked"]["ep"] for r in rows]); tt=np.array([r["T_transparent"]["ep"] for r in rows])
for lo,hi in bands:
    m=(v0>=lo)&(v0<hi); a=m&(tcm>0)
    print(f"[{lo},{hi}) n={m.sum():4d} EP_tc {100*tc[a].mean():.2f} H {100*h[a].mean():.2f} Tt {100*tt[a].mean():.2f} loss_alive {100*(1-tc[a]).sum()/len(tc):.2f} loss_zero {100*(1-tc[m&(tcm==0)]).sum()/len(tc):.2f}")
