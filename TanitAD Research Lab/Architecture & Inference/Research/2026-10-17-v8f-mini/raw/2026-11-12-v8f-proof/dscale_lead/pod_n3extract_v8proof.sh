#!/usr/bin/env bash
# pod_n3extract_v8proof.sh -- lead's check of note N3 of card ADD45 on the v8proof pod, CPU only: the tool's extraction of the evaluation states through the evaluation
# view (built in the N3 trial from this pod's manifest REVERSED) gives the same arrays as the extraction from this pod's own evaluation cache. It waits for one
# checkpoint of the running card ADD60 and only reads it; it writes into a folder of its own. Not a result of any card.
# Start: nohup setsid bash pod_n3extract_v8proof.sh > N3EXTRACT.txt 2>&1 < /dev/null &      Marker: $N/N3EXTRACT_DONE.txt
set -u
N=/workspace/v8proof/dscale_lead_n3; OLD=/workspace/v8proof/dscale_lead; DATA=/workspace/v8proof/data; PY=/workspace/venv/bin/python
CK=$OLD/half1/out/runC/${ARM:-TK16_s0}/ckpt_last.pt
k=0; while [ ! -f "$CK" ]; do k=$((k+1)); [ "$k" -gt 360 ] && { echo "ZZ N3EXTRACT STOP no checkpoint after 3 h" | tee "$N/N3EXTRACT_DONE.txt"; exit 3; }; sleep 30; done
sleep 20
echo "checkpoint $(sha256sum "$CK" | cut -c1-16) at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat); free=$(( (mm - an) / 1073741824 )); echo "container free ${free} GB"
[ "$free" -ge 32 ] || { echo "ZZ N3EXTRACT STOP memory ${free} GB" | tee "$N/N3EXTRACT_DONE.txt"; exit 3; }
T=$(ls -d "$N"/trial_* | tail -n 1); V=$T/out/eval_view
W=$N/extract_$(date -u +%H%M%S); mkdir -p "$W"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
for tag in view own; do
  c=$V; [ "$tag" = own ] && c=$DATA/cache_pod
  nice -n 10 "$PY" "$N/tok_probe_u.py" extract --grid pool4x10 --tree "$OLD/ship_b/tree" --ckpt "$CK" --arm "${ARM:-TK16_s0}" --cache "$c" --states-dir "$W/states_$tag" --splits eval --eval-range 0:141 --device cpu --threads 3 > "$W/extract_$tag.out" 2>&1
  echo "extract $tag rc $? :: $(tail -n 1 "$W/extract_$tag.out" | cut -c1-220)"
done
"$PY" - "$W" "$V/eval/manifest.jsonl" "$T/revcache/eval/manifest.jsonl" <<'EOF'
import glob, hashlib, json, os, sys
import numpy as np
w = sys.argv[1]
ids = lambda p: [json.loads(l)["clip"] for l in open(p, encoding="utf-8") if l.strip()]
print("view manifest order == reversed source order:", ids(sys.argv[2]) == ids(sys.argv[3]), "(must be False)")
files = sorted(os.path.relpath(p, w + "/states_own") for p in glob.glob(w + "/states_own/**/*.npz", recursive=True))
print("npz files:", len(files))
ok = bool(files)
for f in files:
    a, b = np.load(w + "/states_own/" + f, allow_pickle=False), np.load(w + "/states_view/" + f, allow_pickle=False)
    same = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    ok = ok and same
    big = max(a.files, key=lambda k: a[k].size)
    print(f, "arrays", len(a.files), "equal", same, "largest", big, tuple(a[big].shape), hashlib.sha256(np.ascontiguousarray(a[big]).tobytes()).hexdigest()[:16],
          "clips in sha12 order:", ("clips" in a.files and [str(x) for x in a["clips"]] == sorted([str(x) for x in a["clips"]], key=lambda c: hashlib.sha256(c.encode()).hexdigest()[:12])))
print("ZZ N3EXTRACT states through the view equal the states from the pod's own cache:", ok)
EOF
ls "$W" | tr '\n' ' '; echo
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') done $W" > "$N/N3EXTRACT_DONE.txt"
