#!/bin/bash
# kin_cache_precompute.sh [SEED ...] -- REVISION 2 (Amendment 1 section 3). The label-cache PRECOMPUTE of each RK seed's bank, ON THE POD, CPU ONLY, OUTSIDE the GPU lock. One job at a time (seeds in order), nice 19, OMP/MKL 2 threads, CUDA hidden.
# For each seed K: the trainer's own `--selfcheck-only` (via the pod's rh_selfcheck.py) on the RK_sK argv with `--device cpu` (the ONLY difference from the launch argv; rk_argv.py cpu-argv builds it in memory from RL_sK, writes nothing under runs/),
# at the code of record (tree_rk, commit 1a8b11a1). The argv names `--dagger-label-cache $R/kin_cache/s<K>_kin_v1.npz`; that path is absent, so the labeller runs once and the trainer writes the cache atomically.
# Writes ONLY under $R/kin_cache/: s<K>_kin_v1.npz (the cache), s<K>_selfcheck_cpu.json (the selfcheck.json of that run: the gate record's census / sha check), s<K>_argv_cpu.json, s<K>_selfcheck_result.json, s<K>_precompute.log,
# s<K>_precompute.json (sha256, size, wall seconds, census from kin_status_counts); the scratch dir $KC/sc_s<K> is deleted at the end of the seed (logged to $KC/deletions.log). Never touches the GPU, /root/gpu_queue.lock, /root/N1DW_READY or another process.
# Guards: the container's total_rss (cgroup, never free) must be < 40 GB before each start (waits up to 30 min, then stops and reports); the child is killed BY PID if the container reaches 40 GB, its own RSS passes 14 GB, or it runs > 90 min.
# An existing cache file is never overwritten: that seed is refused (rename it by hand). Last line of $KC/precompute.log: ZZ RK-CACHE-PRECOMPUTE DONE ZZ or ZZ RK-CACHE-PRECOMPUTE STOP ZZ <why>.
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/rk_env.sh
export_py
SEEDS=${*:-0 1}
mkdir -p $KC
PL=$KC/precompute.log
say() { echo "$(date -u +%FT%TZ) $*" >> $PL; }
stop() { say "ZZ RK-CACHE-PRECOMPUTE STOP ZZ $1"; exit 1; }
rss() { awk '/^total_rss /{printf "%d", $2}' /sys/fs/cgroup/memory/memory.stat; }
LIM=42949672960
[ -d $TREE ] || stop "no $TREE"
$PY $HERE/rk_argv.py cpu-argv --seeds $SEEDS --out-dir $KC >> $PL 2>&1 || stop "rk_argv.py cpu-argv refused"
for s in $SEEDS; do
  C=$KC/s${s}_kin_v1.npz
  [ ! -e $C ] && [ ! -e $C.part ] || stop "$C or $C.part exists: refusing (never overwritten; rename by hand)"
  n=0; until [ "$(rss)" -lt $LIM ]; do n=$((n+1)); [ $n -gt 180 ] && stop "container total_rss $(rss) B >= 40 GB for 30 min before seed $s"; sleep 10; done
  say "seed $s: start; container total_rss $(( $(rss) / 1048576 )) MB"
  t0=$(date +%s)
  ( export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; RH_TREE=$TREE exec nice -n 19 $PY $R/rh_selfcheck.py $KC/s${s}_argv_cpu.json $KC/sc_s$s $KC/s${s}_selfcheck_result.json > $KC/s${s}_precompute.log 2>&1 ) &
  P=$!; last=0
  while kill -0 $P 2>/dev/null; do
    sleep 10
    el=$(( $(date +%s) - t0 )); pr=$(ps -o rss= -p $P 2>/dev/null | tr -d ' '); cr=$(rss)
    if [ $((el - last)) -ge 300 ]; then last=$el; say "seed $s ${el}s rss $((${pr:-0}/1024)) MB container $((cr/1048576)) MB: $(tail -n 1 $KC/s${s}_precompute.log | cut -c1-140)"; fi
    if [ "${pr:-0}" -gt 14680064 ] || [ $el -gt 5400 ] || [ "${cr:-0}" -ge $LIM ]; then
      kill -TERM $P 2>/dev/null; sleep 5; kill -KILL $P 2>/dev/null
      stop "seed $s stopped by the guard at ${el}s (rss ${pr:-0} KB, container ${cr} B); last log line: $(tail -n 1 $KC/s${s}_precompute.log | cut -c1-200)"
    fi
  done
  wait $P; rc=$?
  wall=$(( $(date +%s) - t0 ))
  say "seed $s: rc=$rc wall ${wall}s: $(tail -n 1 $KC/s${s}_precompute.log | cut -c1-300)"
  [ -s $KC/sc_s$s/selfcheck.json ] && [ -s $C ] || stop "seed $s wrote no selfcheck.json / no cache (rc=$rc)"
  cp $KC/sc_s$s/selfcheck.json $KC/s${s}_selfcheck_cpu.json
  $PY - $KC $s $wall >> $PL <<'PYE' || stop "seed $s: record"
import hashlib, json, os, sys
KC, s, wall = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = f"{KC}/s{s}_kin_v1.npz"
h = hashlib.sha256()
with open(c, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
sc = json.load(open(f"{KC}/s{s}_selfcheck_cpu.json"))


def find(o, k):
    if isinstance(o, dict):
        for a, v in o.items():
            if a == k:
                yield v
            yield from find(v, k)
    elif isinstance(o, list):
        for v in o:
            yield from find(v, k)


counts = [v for v in find(sc.get("dagger_probe") or {}, "kin_status_counts") if isinstance(v, dict)]
n_ok = sum(counts[0].values()) if counts else None
share = counts[0].get("kin_emitted", 0) / n_ok if counts and n_ok else None
rec = {"seed": int(s), "cache": c, "sha256": h.hexdigest(), "size_bytes": os.path.getsize(c), "wall_s": wall, "kin_status_counts": counts[0] if counts else None, "n_ok": n_ok, "relabelled_share_of_ok": share,
       "bar": 0.5, "levers_cache_sha256": sorted(set(map(str, find(sc.get("levers", {}), "dagger_label_cache_sha256")))), "declared_vs_built": sc.get("declared_vs_built")}
json.dump(rec, open(f"{KC}/s{s}_precompute.json", "w"), indent=1)
print("precompute", json.dumps({k: rec[k] for k in ("seed", "sha256", "size_bytes", "wall_s", "n_ok", "relabelled_share_of_ok")}))
PYE
  du -sk $KC/sc_s$s | cut -f1 | { read k; echo "$(date -u +%FT%TZ) $KC/sc_s$s ${k} KB (scratch selfcheck dir of seed $s; selfcheck.json kept as s${s}_selfcheck_cpu.json) deleted" >> $KC/deletions.log; }
  rm -rf "${KC:?}/sc_s$s"
  say "seed $s: done"
done
say "ZZ RK-CACHE-PRECOMPUTE DONE ZZ"
