#!/bin/bash
# stageA.sh -- pod-side, DETACHED. Rebuilds the v8F-mini training token cache on this pod (the previous pod's proven route):
#   verify inputs -> fetch DINOv3-L weights (sha pinned) -> v2ep_pod stage -> byte gate -> v2ep run (CPU frames) || cache_pod (GPU encode, trailing)
#   -> final verification (4,572 train + 141 eval) -> STAGEA_REPORT.json -> STAGEA_DONE.txt.
# Fail closed: any failure writes ONE line to STAGEA_STOP.txt and exits. No address, port or pod id in this file; the HF token is read in place by python.
set -u
W=/workspace/v8proof
A=$W/stageA
HR=$W/hierrep
DATA=$W/data
PW=$DATA/podwork
PY=$W/../venv/bin/python
PY=/workspace/venv/bin/python
WORKERS=${STAGEA_WORKERS:-5}
mkdir -p "$HR" "$PW" "$DATA/cache_pod" "$DATA/inputs" "$DATA/v9" "$W/hf"
rm -f "$HR/STAGEA_DONE.txt" "$HR/STAGEA_STOP.txt"
date +%s > "$HR/stageA_start.epoch"
exec 9>"$HR/stageA.lock"; flock -n 9 || { echo "STAGEA already running" ; exit 1; }
say()  { echo "[stageA $(date -u +%FT%TZ)] $*"; }
stop() { say "ZZ STAGEA STOP ZZ $*"; echo "$*" > "$HR/STAGEA_STOP.txt"; exit 1; }

export HF_HUB_DISABLE_XET=1 HF_HUB_CACHE=$W/hf PYTHONUNBUFFERED=1
export TANITAD_STACK=$W/tree/stack
export PYTHONPATH=$W/tree/stack:$W/tree/stack/scripts:$W/tree/taniteval
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

# ---- 1 inputs (shipped from the dev box, sha256 pinned)
say "step 1: verify shipped inputs"
chk() { got=$(sha256sum "$1" 2>/dev/null | cut -d' ' -f1); [ "$got" = "$2" ] || stop "input $1 sha256 ${got:-missing} != pin $2"; }
chk $DATA/inputs/join_train.npz     978f41bf4f1e4c8b75bdfad31d39ee184b59634feb5da902157d306595ac3124
chk $DATA/inputs/native_tracks.pt   1db431f112b1bd7c426073c43c7df5fc75b2a032736c1e2020f02ffb118669e7
chk $DATA/inputs/p10_prior.pt       14a062a91078266a1b3226af91bf52b23b5238d84fe7e633323d0b0bea2703bf
for f in $A/ops/train4572.txt $A/ops/clips141.txt $A/ops/banked_hashes.json $A/ops/intrinsics_dev.csv $A/code/v2ep_pod.py $A/code/cache_pod.py $A/code/pull_encode_pooled.py; do
  [ -s "$f" ] || stop "missing file $f"
done
[ "$(wc -l < $A/ops/train4572.txt)" = 4572 ] || stop "train4572.txt is not 4,572 lines"
[ -d $W/tree/stack/tanitad ] || stop "code tree missing at $W/tree"
nvidia-smi -L > /dev/null 2>&1 || stop "no GPU visible"
say "inputs ok"

# ---- 2 weights
say "step 2: fetch DINOv3-L weights"
$PY $A/code/stageA_fetch.py > $HR/fetch_dinov3.log 2>&1 || stop "DINOv3 fetch or sha check failed ($HR/fetch_dinov3.log)"
export DINO_SAFETENSORS=$(cat $HR/dino_safetensors.path)
say "weights ok $(cut -c1-60 <<< "$DINO_SAFETENSORS")"

# ---- 3 stage (timestamps, egomotion, intrinsics from HF) and the byte gate
cd $A/code || stop "cd code"
say "step 3: v2ep_pod stage"
$PY -u v2ep_pod.py stage >> $HR/v2ep_stage.log 2>&1 || stop "v2ep stage failed (tail: $(tail -n 2 $HR/v2ep_stage.log | tr '\n' '|' | sed 's/hf_[A-Za-z0-9]*/X/g' | cut -c1-240))"
say "step 4: byte gate on 6 banked clips"
WAIVE=${STAGEA_WAIVE_GATE:-}
if $PY -u v2ep_pod.py gate >> $HR/v2ep_gate.log 2>&1; then
  say "gate pass"
elif [ -n "$WAIVE" ] && [ -f $PW/gate_result.json ]; then
  say "byte gate FAILED, WAIVED by the lead: $WAIVE"
else
  stop "byte gate FAILED (0/3 banked hashes, as on the previous pod; see $HR/v2ep_gate.log, $PW/gate_result.json). Needs a lead decision; to continue: STAGEA_WAIVE_GATE=<reason> nohup setsid bash $A/stageA.sh > $HR/stageA.log 2>&1 < /dev/null &"
fi
RUNEXTRA=()
[ -n "$WAIVE" ] && RUNEXTRA=(--waive-gate "$WAIVE")

# ---- 4 frames (CPU) and trailing encode (GPU)
say "step 5: v2ep run (workers $WORKERS) and cache_pod"
( $PY -u v2ep_pod.py run --workers $WORKERS "${RUNEXTRA[@]}" >>$HR/v2ep_run.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "ZZ V2EP DONE ZZ" >> $HR/v2ep_run.log; else echo "ZZ V2EP FAIL ZZ rc=$rc" >> $HR/v2ep_run.log; fi
  exit $rc ) &
PB=$!
( HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -u cache_pod.py --out $DATA/cache_pod --threads 2 --poll 60 >> $HR/cache_pod.log 2>&1 ) &
PC=$!
echo "$PB $PC" > $HR/stageA_children.pids
say "children: frames build pid $PB, cache encoder pid $PC"
wait $PB; rb=$?
wait $PC; rc=$?
say "frames rc=$rb encode rc=$rc"
[ $rb -eq 0 ] || stop "v2ep run failed rc=$rb ($HR/v2ep_run.log)"
[ $rc -eq 0 ] || stop "cache_pod failed rc=$rc ($HR/cache_pod.log)"
grep -a -q "ZZ CACHEPOD DONE ZZ" $HR/cache_pod.log || stop "cache_pod exited 0 without its DONE marker"

# ---- 5 verify
say "step 6: final verification"
$PY $A/code/stageA_final.py > $HR/stageA_final.log 2>&1 || stop "final verification failed: $(tail -n 3 $HR/stageA_final.log | tr '\n' ' ' | cut -c1-300)"

# ---- 6 delete re-creatable scratch (frames + staged sources: HF is the source; literal paths; logged)
for d in $PW/v2ep $PW/_hf $PW/stage; do
  if [ -d "$d" ]; then sz=$(du -sk "$d" | cut -f1); rm -rf "$d"; echo "deleted $d ${sz}KB" >> $HR/deleted.log; fi
done
echo "ZZ STAGEA DONE ZZ cache $DATA/cache_pod train 4572 eval 141" > $HR/STAGEA_DONE.txt
say "ZZ STAGEA DONE ZZ"
exit 0
