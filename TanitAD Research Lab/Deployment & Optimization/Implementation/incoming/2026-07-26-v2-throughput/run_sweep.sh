#!/usr/bin/env bash
# One PROCESS PER CONFIG.
#
# WHY: in the first pass all configs ran in a single process and every config
# after the first OOM reported ~42 GiB already "in use" -- a caught
# torch.cuda.OutOfMemoryError keeps a traceback that references the frames
# holding `frames`/`fut`/activations, so `del model; empty_cache()` cannot
# reclaim them. Every post-OOM result was therefore an artifact of the leak,
# not a property of the config. A fresh process guarantees a clean allocator,
# so an OOM verdict here is a REAL verdict.
set -u

BENCH=/root/v2bench/bench_v2_throughput.py
OUTDIR=/root/v2bench/out
STACK=/root/v2bench/stack
TIMED=${TIMED:-10}
WARM=${WARM:-2}
POOL=${POOL:-2}

mkdir -p "$OUTDIR"
CONFIGS="gc_on_16x4 gc_on_32x2 gc_on_64x1 gc_off_16x4 gc_off_32x2 gc_off_64x1"

for c in $CONFIGS; do
    echo "=============================================================="
    echo "[runner] $c  (fresh process)  $(date -u +%H:%M:%SZ)"
    PYTHONPATH="$STACK" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python3 "$BENCH" --out "$OUTDIR/$c.json" \
        --timed-steps "$TIMED" --warmup-steps "$WARM" --pool-batches "$POOL" \
        --only "$c" 2>&1 | grep -vE '^\[env\]|^\[cfg\]'
    echo "[runner] $c exit=${PIPESTATUS[0]}"
    sleep 3          # let the driver fully release the context
done

echo "=============================================================="
echo "[runner] merging"
python3 - "$OUTDIR" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
order = ["gc_on_16x4", "gc_on_32x2", "gc_on_64x1",
         "gc_off_16x4", "gc_off_32x2", "gc_off_64x1"]
merged, meta = [], None
for name in order:
    p = d / f"{name}.json"
    if not p.exists():
        merged.append({"config": name, "status": "NO_OUTPUT"})
        continue
    j = json.loads(p.read_text())
    meta = meta or j.get("meta")
    merged.extend(j.get("configs", []))
out = {"meta": meta, "configs": merged}
(d.parent / "results.json").write_text(json.dumps(out, indent=2))
print(f"merged {len(merged)} configs -> {d.parent/'results.json'}")
PY
echo "[runner] DONE"
