import csv,sys,json
a,b,out=sys.argv[1:4]
cols=["no_at_fault_collisions","drivable_area_compliance","driving_direction_compliance","ego_progress","time_to_collision_within_bound","comfort","score"]
def L(p): return {r["token"]:r for r in csv.DictReader(open(p,encoding="utf-8")) if r["token"]!="average"}
A,B=L(a),L(b)
res={"n_a":len(A),"n_b":len(B),"missing_in_a":len(set(B)-set(A)),"missing_in_b":len(set(A)-set(B)),"tol":1e-6,"per_metric":{}}
for c in cols:
    ds=[abs(float(A[t][c])-float(B[t][c])) for t in set(A)&set(B)]
    res["per_metric"][c]={"max_abs_diff":max(ds),"n_over_1e-6":sum(d>1e-6 for d in ds)}
res["PASS"]=all(v["n_over_1e-6"]==0 for v in res["per_metric"].values()) and not res["missing_in_a"] and not res["missing_in_b"]
json.dump(res,open(out,"w"),indent=1); print(json.dumps(res,indent=1))
