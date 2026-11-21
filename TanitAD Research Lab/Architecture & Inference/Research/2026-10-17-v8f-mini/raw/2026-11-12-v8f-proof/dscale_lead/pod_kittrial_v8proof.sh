#!/usr/bin/env bash
# pod_kittrial_v8proof.sh -- lead's trial of the kit script itself (stages precheck and cpucheck, run for real, CPU only) on the v8proof pod with stand-in data:
# the corpus cache as the "union", the first half of its manifest as the "corpus list", the stand-in lead file, the shipped tree. Not a result of the card.
# This pod has cgroup v2, so the kit's memory floor (cgroup v1 files) is set to 0 here and the memory is checked by this script instead.
# Usage: bash pod_kittrial_v8proof.sh > KITTRIAL.txt 2>&1
set -u
D=/workspace/v8proof/dscale_lead; DATA=/workspace/v8proof/data
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat); free=$(( (mm - an) / 1073741824 )); echo "container free ${free} GB"
[ "$free" -ge 20 ] || { echo "ZZ KITTRIAL STOP memory"; exit 3; }
W=$D/kittrial_$(date -u +%H%M%S); mkdir -p "$W"
/workspace/venv/bin/python - "$DATA/cache_pod/train/manifest.jsonl" "$W" <<'EOF'
import json, sys
ids = list(dict.fromkeys(json.loads(l)["clip"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()))
open(sys.argv[2] + "/half.txt", "w", encoding="utf-8", newline="\n").write("\n".join(ids[: len(ids) // 2]) + "\n")
print("manifest clips", len(ids), "half", len(ids) // 2)
EOF
LEAD=$D/standin/lead_train.npz
export KIT=$D TREE=$D/ship_b/tree OUT=$W/out PY=/workspace/venv/bin/python
export CACHE=$DATA/cache_pod EVAL_CACHE=$DATA/cache_pod
export V9_UNION=$DATA/v9/v9_labels_train.npz V9_MD5=83e30bd3791f0767707eade5f4671759 LEAD_UNION=$LEAD LEAD_CORPUS=$LEAD SPEED_UNION=$DATA/v9/refcv6_speed_max_v8_train.jsonl JOIN_UNION=$DATA/inputs/join_train.npz
export BSET=$W/half.txt P10=$DATA/inputs/p10_prior.pt TRACKS=$DATA/inputs/native_tracks.pt
export SHA_LEAD_UNION=$(sha256sum "$LEAD" | cut -c1-8) SHA_LEAD_CORPUS=$(sha256sum "$LEAD" | cut -c1-8) SHA_BSET=$(sha256sum "$BSET" | cut -c1-8)
export EXPECT_UNION=4572 EXPECT_CORPUS=2286 MIN_FREE_GB=0 THREADS=3
for st in precheck cpucheck; do
  bash "$D/pod_dscale_lead_run.sh" "$st" > "$W/$st.out" 2>&1; echo "stage $st rc $?"
  grep -v "run: \|run under" "$W/$st.out" | cut -c1-260 | tail -n 14
done
ls "$W/out" "$W/out/precheck" "$W/out/markers" 2>/dev/null | tr '\n' ' '; echo
echo "no GPU stage is run here; the run stage under DRY=1 prints this many commands under the lock:"
DRY=1 bash "$D/pod_dscale_lead_run.sh" run 2>&1 | grep -c "flock"
echo "ZZ KITTRIAL DONE $W"
