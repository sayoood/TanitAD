import glob, json, statistics as st
C = "/workspace/ops/bank3c/controls/calib"
au = [json.loads(l) for fp in glob.glob(C + "/shards/s*/audit.jsonl") for l in open(fp) if l.strip()]
rs = [r["rollout_s"] for r in au if r["rollout_s"] is not None]
t0, t1 = float(open(C + "/t0").read()), float(open(C + "/t1").read())
ca0, ca1 = int(open(C + "/cpuacct_before").read()), int(open(C + "/cpuacct_after").read())
th0 = dict(l.split() for l in open(C + "/throttle_before")); th1 = dict(l.split() for l in open(C + "/throttle_after"))
wall = t1 - t0; cpu = (ca1 - ca0) / 1e9
print(json.dumps({"procs": 11, "twins_audited": len(au), "rolled": len(rs), "wall_s": round(wall, 1), "cpu_s_total": round(cpu, 1), "mean_cpu_cores_used": round(cpu / wall, 2),
  "rollout_s_mean": round(st.mean(rs), 2), "rollout_s_median": round(st.median(rs), 2), "sum_rollout_s": round(sum(rs), 1),
  "cpu_s_per_audited_twin_all_in": round(cpu / len(au), 2), "cpu_s_per_rolled_twin_all_in": round(cpu / len(rs), 2),
  "nr_throttled_delta": int(th1["nr_throttled"]) - int(th0["nr_throttled"]), "throttled_s_delta": round((int(th1["throttled_time"]) - int(th0["throttled_time"])) / 1e9, 1)}))
