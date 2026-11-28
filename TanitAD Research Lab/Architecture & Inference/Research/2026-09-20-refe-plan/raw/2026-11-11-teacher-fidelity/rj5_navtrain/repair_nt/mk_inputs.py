"""RJ5 navtrain labels -> REPAIR-D derive inputs (n4_nt_build.py derive --route navtrain).
plans20 = the P5-R label `p5r` (what a student is trained on); traj20 = the RJ5 roll `traj` (20 poses) from the merged RJ5 rows.
NOT tc20s: tc20s is the tracker-conditioned target; MEASURED 2026-11-13 smoke (4 tokens): C-DERIV max|R(tc20s) - p5r| 0.60-0.94 m, max|R(traj) - p5r| 0.0."""
import gzip, json, sys
import numpy as np
lab_src, rj5_src, out_npz, out_rolls = sys.argv[1:5]
traj = {}
with gzip.open(rj5_src, "rt", encoding="utf-8") as f:
    for ln in f:
        r = json.loads(ln)
        if isinstance(r.get("traj"), list) and len(r["traj"]) == 20:
            traj[r["token"]] = r["traj"]
toks, logs, plans, n_all, n_nolab, n_notraj = [], [], [], 0, 0, 0
with gzip.open(lab_src, "rt", encoding="utf-8") as f, open(out_rolls, "w", encoding="utf-8", newline="\n") as g:
    for ln in f:
        r = json.loads(ln); n_all += 1
        p = r.get("p5r")
        if not (isinstance(p, list) and len(p) == 20):
            n_nolab += 1; continue
        if r["token"] not in traj:
            n_notraj += 1; continue
        lg = r.get("log") or r.get("log_name")
        toks.append(r["token"]); logs.append(lg); plans.append(p)
        g.write(json.dumps({"token": r["token"], "log": lg, "traj20": traj[r["token"]]}) + "\n")
assert len(set(toks)) == len(toks), "duplicate tokens"
np.savez(out_npz, tokens=np.array(toks), logs=np.array(logs), plans20=np.asarray(plans, float))
print(json.dumps({"label_rows": n_all, "kept": len(toks), "no_p5r": n_nolab, "no_rj5_traj": n_notraj, "logs": len(set(logs))}))
