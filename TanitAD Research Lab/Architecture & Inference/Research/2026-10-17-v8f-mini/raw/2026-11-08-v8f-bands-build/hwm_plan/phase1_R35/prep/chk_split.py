# second route: recompute the split from first principles (sha256 rank), compare to the ladder harness's RESULT meta and to the subset
import hashlib, json, numpy as np
d=np.load("F:/TanitAD-scratch/hwm_plan/phase1_R35/s0/windows_all.npz",allow_pickle=False)
s=np.load("F:/TanitAD-scratch/hwm_plan/phase1_R35/s0/windows.npz",allow_pickle=False)
eid=[str(x) for x in d["eid"]]; now=d["now"]
uniq=sorted(set(eid)); key=sorted(uniq,key=lambda e:(hashlib.sha256(f"plantest2-v1|{e}".encode()).hexdigest(),e))
score=set(key[len(key)//2:])
exp=[i for i,e in enumerate(eid) if e in score]
got=set(zip([str(x) for x in s["eid"]],s["now"].tolist()))
expset=set((eid[i],int(now[i])) for i in exp)
print("distinct eps",len(uniq),"score eps expected",len(score),"subset eps",len(set(str(x) for x in s["eid"])))
print("expected windows",len(expset),"subset windows",len(got),"sets equal",expset==got)
R=json.load(open("E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/ladder_plantest/RESULT_LADDER_PLANTEST.json"))
c=R["R35"]["c1"]; print("ladder RESULT R35 c1 n_windows_score",c["n_windows_score"],"episodes",c["n_episodes_score"])
