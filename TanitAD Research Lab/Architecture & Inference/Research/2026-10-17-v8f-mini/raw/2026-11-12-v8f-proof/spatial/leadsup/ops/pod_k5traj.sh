cd /workspace/v8proof/runs || exit 1
/workspace/venv/bin/python - <<'PY'
import json, os, glob
def find(o, key, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key and isinstance(v, str): out.append(v)
            find(v, key, out)
    elif isinstance(o, list):
        for v in o: find(v, key, out)
specs = [("leadsup_a","LS_s0"),("leadsup_a","LS_s1"),("leadsup_b","LSD_s0"),("leadsup_b","LSD_s1"),("leadsup_c","LSH_s0"),("leadsup_c","LSH_s1")]
bins = [(0,100),(100,300),(300,1000),(1000,4000),(4000,9000),(9000,14000),(14000,18164),(18164,99999)]
for q,a in specs:
    rows = [json.loads(l) for l in open(os.path.join(q,a,"train_log.jsonl")) if l.strip().startswith("{")]
    rows = [r for r in rows if "lead_sup" in r]
    if a == "LS_s0": print("KEYS", sorted(k for k in rows[0] if k.startswith("lead")), "p_gt_first/last", rows[0].get("p_gt"), rows[-1].get("p_gt"))
    ks = [k for k in ("lead_sup","lead_sup_bce","lead_sup_gap","lead_sup_clos","lead_sup_n_gap","lead_sup_n_clos","lead_sup_n","wm","p_gt","lr") if k in rows[0]]
    for lo,hi in bins:
        rr = [r for r in rows if lo < r["step"] <= hi]
        if not rr: continue
        print("TRAJ", a, "%5d-%5d n=%3d " % (lo,min(hi,19164),len(rr)) + " ".join("%s=%.4g" % (k.replace("lead_sup","L"), sum(r[k] for r in rr)/len(rr)) for k in ks))
    t1 = glob.glob(os.path.join(q,"t1",a,"t1v8f_%s_*.json" % a)); rec = json.load(open(t1[0])); out=[]; find(rec,"ckpt",out)
    ck = os.path.abspath(os.path.join(q,a,"ckpt_last.pt"))
    print("K6", a, "t1_ckpt_paths", sorted(set(out)), "equal", all(os.path.abspath(p)==ck for p in out) and bool(out), "ckpt_older", os.path.getmtime(ck) <= os.path.getmtime(t1[0]))
PY
