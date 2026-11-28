import glob, json, statistics as st, collections
C = "/workspace/ops/bank3c/controls/timing_prod"
au = [json.loads(l) for fp in glob.glob(C + "/shards/s*/audit.jsonl") for l in open(fp) if l.strip()]
rs = [r["rollout_s"] for r in au if r["rollout_s"] is not None]
t0, t1 = float(open(C + "/t0").read()), float(open(C + "/t1").read())
ca0, ca1 = int(open(C + "/cpuacct_before").read()), int(open(C + "/cpuacct_after").read())
th0 = dict(l.split() for l in open(C + "/throttle_before")); th1 = dict(l.split() for l in open(C + "/throttle_after"))
wall = t1 - t0; cpu = (ca1 - ca0) / 1e9
reasons = collections.Counter((r["drop_reason"] or "kept").split(":")[0] for r in au)
byk = collections.Counter(r["k"] for r in au)
out = {"procs": 11, "shards_of_720": 11, "limit_per_shard": 40, "twins_audited": len(au), "rolled": len(rs), "by_outcome": dict(reasons), "by_k": dict(byk), "wall_s": round(wall, 1),
  "cpu_s_total": round(cpu, 1), "mean_cores_used": round(cpu / wall, 2), "rollout_s_mean": round(st.mean(rs), 2), "rollout_s_median": round(st.median(rs), 2),
  "cpu_s_per_audited_twin_all_in": round(cpu / len(au), 2), "twins_per_s_wall": round(len(au) / wall, 3), "nr_throttled_delta": int(th1["nr_throttled"]) - int(th0["nr_throttled"])}
print(json.dumps(out)); json.dump(out, open(C + "/timing_summary.json", "w"), indent=1)
