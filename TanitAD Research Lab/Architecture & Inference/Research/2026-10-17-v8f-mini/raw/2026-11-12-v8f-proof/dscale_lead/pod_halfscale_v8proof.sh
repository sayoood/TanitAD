#!/usr/bin/env bash
# pod_halfscale_v8proof.sh -- card PREREG_V8F_PROOF_ADD60 (HALFSCALE-LEAD) on the v8proof pod (the pod of this fork): the kit of card ADD45, unchanged, with the
# study's corpus cache as the kit's "union" and the first half of its training manifest as the kit's "corpus list"; then the reader of card ADD45 with the corpus count 2,286.
# It runs beside the rollout queue: one GPU process of its own at a time (the kit's lock is a lock file of this folder). This pod has cgroup v2, so the kit's own memory
# floor (cgroup v1 files) is set to 0 and this script checks the memory: 20 GB free before every stage, and after the smoke the lowest free value seen during the smoke
# must stay above 30 GB (the rollout queue waits under 28 GB), else the script stops before the run stage.
# Start: nohup setsid bash pod_halfscale_v8proof.sh > half1_stdout.txt 2>&1 < /dev/null &      Markers: $H/HALFSCALE_DONE.txt or $H/HALFSCALE_STOP.txt. Nothing is deleted.
set -u
D=/workspace/v8proof/dscale_lead; DATA=/workspace/v8proof/data; H=$D/${HALF_DIR:-half1}
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
freegb() { local mm an; mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat); echo $(( (mm - an) / 1073741824 )); }
[ -e "$H" ] && { echo "REFUSED: $H exists"; exit 3; }
mkdir -p "$H"
say() { echo "[$(ts)] halfscale: $*" | tee -a "$H/halfscale.log"; }
stop() { say "STOP $*"; echo "$(ts) $*" > "$H/HALFSCALE_STOP.txt"; exit 3; }
/workspace/venv/bin/python - "$DATA/cache_pod/train/manifest.jsonl" "$H" <<'EOF'
import json, sys
ids = list(dict.fromkeys(json.loads(l)["clip"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()))
open(sys.argv[2] + "/half.txt", "w", encoding="utf-8", newline="\n").write("\n".join(ids[: len(ids) // 2]) + "\n")
print("manifest clips", len(ids), "half", len(ids) // 2)
EOF
say "half.txt sha256 $(sha256sum "$H/half.txt" | cut -c1-64) lines $(wc -l < "$H/half.txt")"
[ "$(sha256sum "$H/half.txt" | cut -c1-64)" = "${HALF_SHA:?the sha256 of half.txt recorded at the freeze}" ] || stop "HALF-LIST-SHA"
LEAD=$D/standin/lead_train.npz
export KIT=$D TREE=$D/ship_b/tree OUT=$H/out PY=/workspace/venv/bin/python
export CACHE=$DATA/cache_pod EVAL_CACHE=$DATA/cache_pod
export V9_UNION=$DATA/v9/v9_labels_train.npz V9_MD5=83e30bd3791f0767707eade5f4671759 LEAD_UNION=$LEAD LEAD_CORPUS=$LEAD SPEED_UNION=$DATA/v9/refcv6_speed_max_v8_train.jsonl JOIN_UNION=$DATA/inputs/join_train.npz
export BSET=$H/half.txt P10=$DATA/inputs/p10_prior.pt TRACKS=$DATA/inputs/native_tracks.pt
export SHA_LEAD_UNION=$(sha256sum "$LEAD" | cut -c1-8) SHA_LEAD_CORPUS=$(sha256sum "$LEAD" | cut -c1-8) SHA_BSET=$(sha256sum "$BSET" | cut -c1-8)
export EXPECT_UNION=4572 EXPECT_CORPUS=2286 MIN_FREE_GB=0 THREADS=4 LOCK=$H/gpu.lock
say "kit script sha256 $(sha256sum "$D/pod_dscale_lead_run.sh" | cut -c1-16) tool $(sha256sum "$D/tok_probe_u.py" | cut -c1-16) reader $(sha256sum "$D/dscale_lead_read.py" | cut -c1-16)"
for st in precheck cpucheck smoke run extract checklog; do
  f=$(freegb); say "stage $st starts; container free ${f} GB"
  [ "$f" -ge 20 ] || stop "MEMORY before $st: ${f} GB"
  ( lo=999; while :; do g=$(freegb); [ "$g" -lt "$lo" ] && { lo=$g; echo "$lo" > "$H/minfree_$st.txt"; }; sleep 20; done ) & mon=$!
  bash "$D/pod_dscale_lead_run.sh" "$st" > "$H/$st.out" 2>&1; rc=$?
  kill "$mon" 2>/dev/null; wait "$mon" 2>/dev/null
  say "stage $st rc $rc; lowest free memory seen $(cat "$H/minfree_$st.txt" 2>/dev/null) GB"
  [ $rc -eq 0 ] || stop "STAGE $st rc $rc: $(tail -n 2 "$H/$st.out" | cut -c1-200 | tr '\n' ' ')"
  if [ "$st" = smoke ]; then
    say "smoke estimate: $(cat "$OUT/SMOKE_ESTIMATE.json" 2>/dev/null)"
    lo=$(cat "$H/minfree_smoke.txt" 2>/dev/null || echo 0); [ "$lo" -gt 30 ] || stop "MEMORY during the smoke: lowest free ${lo} GB (the rollout queue waits under 28 GB)"
  fi
done
cd "$D" || stop "CD"
[ -e "$D/F:" ] || ln -s "/workspace/v8proof/finegrid/rk/F:" "$D/F:"
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/venv/bin/python "$D/dscale_lead_read.py" --in-dir "$OUT/states" --run-dir "$OUT" --out-dir "$H/read" --expect-corpus-clips 2286 --done-marker "$OUT/DSCALE_LEAD_DONE.txt" > "$H/read.out" 2>&1; rc=$?
say "reader rc $rc (0 names, 2 a control failed, 3 an input is missing)"
ls "$H/read" 2>/dev/null | tr '\n' ' ' | tee -a "$H/halfscale.log"; echo
echo "$(ts) reader rc $rc" > "$H/HALFSCALE_DONE.txt"
say "HALFSCALE_DONE"
