"""CPU/IO diagnostic (read-only on /workspace): cold read latency of the online labeller's parts files, sampled the way training samples them
(random rows of the training bank -> online_labeller.parts_path), split into raw NFS read time and lzma+unpickle CPU time."""
import json, random, sys, time, os, lzma, pickle
sys.path.insert(0, sys.argv[1])            # the shipped refe dir (online_labeller.parts_path)
import online_labeller as OL
bank, mc, n, seed = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rows = []
for nm in ("targets_rank0.jsonl", "targets_rank1.jsonl"):
    with open(os.path.join(bank, nm), "rb") as fh:
        for ln in fh:
            rows.append(ln)
rng = random.Random(seed)
pick = [json.loads(rows[i]) for i in rng.sample(range(len(rows)), n)]
rd, cpu, sz, miss = [], [], [], 0
for r in pick:
    p = OL.parts_path(mc, r["log_name"], r["token"], r.get("aug"))
    t0 = time.perf_counter()
    try:
        with open(p, "rb") as fh:
            b = fh.read()
    except FileNotFoundError:
        miss += 1
        continue
    t1 = time.perf_counter()
    c0 = time.process_time()
    pickle.loads(lzma.decompress(b))
    cpu.append(time.process_time() - c0)
    rd.append(t1 - t0); sz.append(len(b))
def q(xs, p):
    xs = sorted(xs); return round(xs[min(len(xs) - 1, int(p * len(xs)))], 4)
out = {"n": len(rd), "missing": miss, "bytes_median": q(sz, .5),
       "read_s": {"p50": q(rd, .5), "p90": q(rd, .9), "p99": q(rd, .99), "max": round(max(rd), 4), "mean": round(sum(rd) / len(rd), 4), "n_over_0.1s": sum(x > 0.1 for x in rd), "n_over_0.3s": sum(x > 0.3 for x in rd)},
       "decode_cpu_s": {"p50": q(cpu, .5), "p90": q(cpu, .9), "max": round(max(cpu), 4)}}
print("ZZNFS_PARTS " + json.dumps(out))
