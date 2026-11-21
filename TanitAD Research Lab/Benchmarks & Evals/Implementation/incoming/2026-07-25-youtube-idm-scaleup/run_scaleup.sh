#!/usr/bin/env bash
# YouTube-IDM SCALE-UP driver (pod3). Footprint-bounded, resumable, banks as it goes.
#   loop: harvest a BATCH of clips -> dd-check disk -> pseudo_label (encode + DELETE
#         frames) -> repeat until TARGET clips or the candidate pool is exhausted
#   then: downstream decision read at >=SEEDS seeds -> results JSON -> DONE sentinel
# Reuses the PROVEN pilot pseudo_label.py + run_youtube_pilot_downstream.py verbatim;
# only harvest is the non-CC scale-up variant. Peak on-disk imagery ~= one BATCH
# (~20 GB), well under the ~60 GB budget; latents (non-imagery) + pointers persist.
#
# Launch detached:
#   ssh tanitad-pod3 'PYTHONPATH=/workspace/TanitAD/stack nohup bash \
#     /workspace/tmp/yt_scaleup/scripts/run_scaleup.sh > /workspace/tmp/yt_scaleup/run.log 2>&1 &'
set -u
export PYTHONPATH=/workspace/TanitAD/stack
V=/workspace/venv/bin/python
WORK=/workspace/tmp/yt_scaleup
S=$WORK/scripts
R=$WORK/results
CKPT=/workspace/tmp/idm/ckpt.pt
mkdir -p "$R" "$WORK/clips" "$WORK/latents"

TARGET=${TARGET:-800}       # decision-grade scale (pilot named ~300+; go beyond)
BATCH=${BATCH:-140}         # clips per harvest cycle (~20 GB peak, footprint-bounded)
SEEDS=${SEEDS:-4}           # >=4 = decision-grade (pilot was 3)
GEO=${GEO:-$WORK/geocalib_intrinsics.json}   # used iff present, else fixed-HFOV
DISK_MIN_GB=${DISK_MIN_GB:-8}

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

ddcheck(){   # real MooseFS-quota check (df lies on pod3); STOP if we cannot write
  if ! dd if=/dev/zero of="$WORK/_ddprobe" bs=1M count=$((DISK_MIN_GB*1024)) oflag=direct >/dev/null 2>&1; then
    log "DDCHECK FAILED — cannot write ${DISK_MIN_GB}GB to $WORK (quota?). STOP to avoid bricking pod."
    truncate -s 0 "$WORK/_ddprobe" 2>/dev/null; unlink "$WORK/_ddprobe" 2>/dev/null
    return 1
  fi
  truncate -s 0 "$WORK/_ddprobe" 2>/dev/null; unlink "$WORK/_ddprobe" 2>/dev/null
  return 0
}

harvested_total(){ $V - <<'PY'
import json,os
p="/workspace/tmp/yt_scaleup/harvest_state.json"
print(json.load(open(p))["n_clips"] if os.path.exists(p) else 0)
PY
}

log "SCALEUP START target=$TARGET batch=$BATCH seeds=$SEEDS geo=${GEO}"
[ -f "$CKPT" ] || { log "FATAL: encoder ckpt $CKPT missing"; exit 2; }
GEOARG=""; [ -f "$GEO" ] && GEOARG="--geocalib-json $GEO" && log "GeoCalib intrinsics FOUND -> per-video geometry" || log "GeoCalib absent -> fixed-HFOV fallback (re-runnable later)"

stall=0
while : ; do
  HN=$(harvested_total)
  if [ "$HN" -ge "$TARGET" ]; then log "target reached: $HN >= $TARGET"; break; fi
  ddcheck || break
  CAP=$(( HN + BATCH )); [ "$CAP" -gt "$TARGET" ] && CAP=$TARGET
  log "HARVEST cycle: have $HN, cap -> $CAP"
  $V "$S/harvest_scaleup.py" --work "$WORK" \
     --queries-file "$S/queries_noncc.txt" --channels-file "$S/channels.txt" \
     --max-clips "$CAP" --per-video-clips 30 --clip-frames 250 --max-videos 60 \
     --per-query 40 --allow-noncc $GEOARG 2>&1 | tail -40
  log "PSEUDO_LABEL cycle (encode + delete frames)"
  $V "$S/pseudo_label.py" --ckpt "$CKPT" \
     --clips-dir "$WORK/clips" --latents-dir "$WORK/latents" \
     --out "$R/pseudo_labels.json" 2>&1 | tail -15
  HN2=$(harvested_total)
  LAT=$(ls "$WORK/latents" 2>/dev/null | wc -l)
  log "post-cycle: harvested=$HN2 latents=$LAT"
  cp -f "$WORK/manifest.json" "$R/harvest_manifest.json" 2>/dev/null
  if [ "$HN2" -le "$HN" ]; then
    stall=$((stall+1)); log "no new clips this cycle (stall $stall/2 — candidate pool thinning)"
    [ "$stall" -ge 2 ] && { log "candidate pool exhausted at $HN2 clips — proceeding with what we have"; break; }
  else stall=0; fi
done

# final drain (any clips harvested but not yet encoded) then bank
$V "$S/pseudo_label.py" --ckpt "$CKPT" --clips-dir "$WORK/clips" \
   --latents-dir "$WORK/latents" --out "$R/pseudo_labels.json" 2>&1 | tail -8
LAT=$(ls "$WORK/latents" 2>/dev/null | wc -l)
log "HARVEST+LABEL COMPLETE: $LAT clip-latents. Starting downstream (seeds=$SEEDS)."
echo "HARVEST_LABEL_DONE latents=$LAT" > "$R/STAGE_HARVEST_DONE"

# ---- P4 decision read: FLOOR vs PSEUDO_YT on parity-val, >=4 seeds, clip-bootstrap CI ----
$V "$S/run_youtube_pilot_downstream.py" \
   --flagship-ckpt "$CKPT" --yt-latents "$WORK/latents" \
   --work "$R" --out "$R/results_scaleup_downstream.json" \
   --seeds "$SEEDS" --pt-epochs 25 --ft-epochs 60 --with-parity-ref 2>&1 | tail -60

if [ -f "$R/results_scaleup_downstream.json" ]; then
  log "DOWNSTREAM DONE -> $R/results_scaleup_downstream.json"
  echo "SCALEUP_ALL_DONE latents=$LAT $(date -u +%FT%TZ)" > "$R/DONE"
else
  log "DOWNSTREAM did not produce results JSON — see run.log"
  echo "DOWNSTREAM_FAILED $(date -u +%FT%TZ)" > "$R/DONE"
fi
log "SCALEUP DRIVER EXIT"
