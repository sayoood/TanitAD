import json, glob, gzip, os
o = gzip.open(os.environ["OUT"], "wt", encoding="utf-8")
for ch in ("run_local", "run_fetch_A", "run_fetch_B"):
    for fp in sorted(glob.glob(f"F:/tanitad_scratch/rj5nt/{ch}/rows/rj5_*ctl*.jsonl")):
        for line in open(fp, encoding="utf-8"):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("arm") == "R0" and "traj" in r:
                o.write(json.dumps({"ch": ch, "log_name": r["log_name"], "token": r["token"], "goal": r["goal"], "origin_world": r["origin_world"], "traj": r["traj"], "bank_dmax": r.get("bank_dmax")}) + "\n")
o.close()
