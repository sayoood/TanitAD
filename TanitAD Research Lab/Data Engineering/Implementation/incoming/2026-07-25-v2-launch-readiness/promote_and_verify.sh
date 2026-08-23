#!/bin/bash
# Promote the transferred pod3 shard into the live v2 corpus dir on pod1, then
# prove the union IS the designed 9,000-clip corpus before anything trains on it.
#
#   ssh tanitad-pod 'bash -s' < promote_and_verify.sh
#
# Safe to re-run: promotion is skipped once staging is empty, and every check
# below is read-only. Exits non-zero on ANY failed check — do not launch if it does.
set -u
D=/workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d
S=/workspace/data/physicalai_v2/_incoming_pod3
EXPECT_KEY=4b7eeeac222d
rc=0

echo "=== 1. staging completeness ==="
n_stage=$(ls -1 $S/*.v2ep.pt 2>/dev/null | wc -l)
echo "staged = $n_stage (expect 4047)"
if [ "$n_stage" -eq 4047 ]; then
  echo "=== 2. promote (same filesystem -> instant rename, no extra space) ==="
  mv $S/*.v2ep.pt $D/ && rmdir $S && echo "promoted OK"
elif [ "$n_stage" -eq 0 ] && [ ! -d "$S" ]; then
  echo "staging already promoted — continuing to verification"
else
  echo "REFUSING to promote: staging holds $n_stage of 4047."
  echo "Re-run transfer_v2.sh (it is resumable) until this reads 4047."
  exit 1
fi

echo "=== 3. drop the stale manifest (pre-MANIFEST_VERSION 2, and the file set changed) ==="
rm -f $D/_v2manifest.pt && echo "removed _v2manifest.pt (loader rebuilds it)"

echo "=== 4. corpus completeness + key ==="
python3 - "$D" "$EXPECT_KEY" <<'PY'
import hashlib, json, os, sys
D, expect = sys.argv[1], sys.argv[2]
ids = sorted(f[:-len(".v2ep.pt")] for f in os.listdir(D) if f.endswith(".v2ep.pt"))
TARGET = [0.45, 0.14, 0.14, 0.13, 0.14] + [0.10, 0.52, 0.38]   # TMAN + TSPD
key = hashlib.sha1(json.dumps({"ids": ids, "target": TARGET, "k": 9000},
                              sort_keys=True).encode()).hexdigest()[:12]
ok = True
print(f"clips        = {len(ids)} (expect 9000)")
print(f"unique ids   = {len(set(ids))}")
print(f"corpus key   = {key} (expect {expect})")
if len(ids) != 9000:          print("FAIL: clip count"); ok = False
if len(set(ids)) != len(ids): print("FAIL: duplicate clip ids"); ok = False
if key != expect:             print("FAIL: corpus key mismatch"); ok = False
print("CORPUS_KEY_OK" if ok else "CORPUS_CHECK_FAILED")
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && rc=1

echo "=== 5. providers + collision-free episode ids over the full union ==="
cd /workspace/TanitAD/stack || exit 1
PYTHONPATH=/workspace/TanitAD/stack python3 - <<'PY'
import sys, time
from tanitad.data.v2_dataset import load_or_build_manifest
D = "/workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d"
t0 = time.time()
man = load_or_build_manifest(D, verbose=False)
raw = [int(x) for x in man["episode_id"]]
uid = [int(x) for x in man["episode_uid"]]
print(f"manifest v{man['version']} built in {time.time()-t0:.1f}s over {len(man['files'])} clips")
print(f"distinct RAW episode_id = {len(set(raw))}  (collisions {len(raw)-len(set(raw))})")
print(f"distinct STABLE ids     = {len(set(uid))}  (collisions {len(uid)-len(set(uid))})")
ok = len(man["files"]) == 9000 and len(set(uid)) == 9000
print("PROVIDERS_OK" if ok else "PROVIDERS_CHECK_FAILED")
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && rc=1

echo "=== 6. disk headroom (local NVMe — df IS valid on pod1) ==="
df -B1 --output=avail /workspace | tail -1 | awk '{printf "free = %.2f GiB\n", $1/1073741824}'

echo
[ $rc -eq 0 ] && echo "ALL CHECKS PASSED — safe to launch (see V2_LAUNCH_READINESS.md §5)" \
             || echo "ONE OR MORE CHECKS FAILED — DO NOT LAUNCH"
exit $rc
