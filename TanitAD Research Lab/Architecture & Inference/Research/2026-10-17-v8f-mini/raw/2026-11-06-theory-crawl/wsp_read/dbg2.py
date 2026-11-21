import json, glob, numpy as np
for root in ("E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage2/pull_box2_run2/pull2/rollouts","C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/rollouts"):
  for d in sorted(glob.glob(root+"/*/objects/T2/P10")):
    try: recs=[json.loads(l) for l in open(d+"/records.jsonl")]
    except Exception: continue
    for r in recs:
        if r["scene"]==2 and r["f0"]==83:
            tr=np.array(r["track"]); print(d, r["f0"], r["v0"], tr.shape)
            print("track v every 0.5s:", np.round(tr[::5,3],1))
            print("track x:", np.round(tr[::10,0],1))
            for k,e in enumerate(r["extra"][:3]):
                p=np.array(r["plans"][k]); sp=np.hypot(*np.diff(np.vstack([[0,0],p]),axis=0).T)/0.2
                print(" k",k,"step",e["step"],"v0_ms",round(e["v0_ms"],2),"plan speeds",np.round(sp[::2],1), "p10_prior speeds", np.round(np.hypot(*np.diff(np.vstack([[0,0],np.array(e["p10_prior"])]),axis=0).T)[::2]/0.2,1))
            print("g (logged) speeds", np.round(np.hypot(*np.diff(np.vstack([[0,0],np.array(r["g"])]),axis=0).T)[::2]/0.2,1))
