import json,sys
d=json.load(open(sys.argv[1]));out=[d["counts_line"],"checks: "+", ".join(d["checks"]),"control: "+json.dumps({k:d["control"].get(k) for k in ("rc","n_tests","green","seconds")})]
if d.get("control_after"): out.append("control_after: "+json.dumps({k:d["control_after"].get(k) for k in ("rc","n_tests","green","seconds")}))
for m in d["mutations"]: out.append("%s %s rc=%s failed=%s expected_all_red=%s"%(m["id"],m["observed"],m["rc"],[f.split("::")[-1] for f in m["failed"]],m["expected_all_red"]))
open(sys.argv[2],"w").write("\n".join(out)+"\n")
