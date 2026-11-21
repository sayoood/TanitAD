import json, sys, numpy as np
rows=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8") if l.strip()]
ok=[r for r in rows if "cderiv_maxdev_m" in r]
d=np.array([r["cderiv_maxdev_m"] for r in ok])
print(json.dumps({"rows":len(rows),"errors":sum("error" in r for r in rows),"n":len(d),"max":float(d.max()),"median":float(np.median(d)),"n_le_1e9":int((d<=1e-9).sum()),"n_eq_0":int((d==0).sum()),
 "label_ego":{k:sum(r.get("label_ego")==k for r in rows) for k in ("db","cache","none")},"mean_t_s":float(np.mean([r["t_s"] for r in ok]))}))
