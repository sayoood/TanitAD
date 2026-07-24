#!/usr/bin/env bash
# YouTube-IDM SCALE-UP — PARALLEL driver (pod3 has 96 cores; single-process used ~5).
# W bounded harvest workers over DISJOINT query slices, each in its own --work dir
# (so the tested harvest_scaleup.py / pseudo_label.py run UNCHANGED — no clip_id races).
# Round-based to bound footprint (~W*ROUND_CLIPS*146MB peak). After each round, each
# worker's new clips are encoded to latents + frames DELETED, and latents are merged
# into one namespace. Then the P4 read runs at >=SEEDS seeds. "bounded worker pool"
# per the brief (no sub-agent spawning). Banks incrementally; DONE sentinel at the end.
#
# Launch detached:
#   ssh tanitad-pod3 'PYTHONPATH=/workspace/TanitAD/stack W=8 TARGET=600 SEEDS=4 \
#     setsid nohup bash /workspace/tmp/yt_scaleup/scripts/run_scaleup_parallel.sh \
#     > /workspace/tmp/yt_scaleup/run.log 2>&1 < /dev/null &'
set -u
export PYTHONPATH=/workspace/TanitAD/stack
V=/workspace/venv/bin/python
WORK=/workspace/tmp/yt_scaleup
S=$WORK/scripts
R=$WORK/results
MERGED=$WORK/latents            # merged (global) latent namespace for the P4 read
CKPT=/workspace/tmp/idm/ckpt.pt
mkdir -p "$R" "$MERGED"

W=${W:-8}                       # bounded worker pool
TARGET=${TARGET:-600}           # decision-grade total clips
ROUND_CLIPS=${ROUND_CLIPS:-25}  # per-worker clips per round (peak ~W*25*146MB ~= 29GB)
SEEDS=${SEEDS:-4}
GEO=${GEO:-$WORK/geocalib_intrinsics.json}
DISK_MIN_GB=${DISK_MIN_GB:-8}
PVC=${PVC:-30}                  # per-video-clips (long drives)
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

ddcheck(){
  if ! dd if=/dev/zero of="$WORK/_ddprobe" bs=1M count=$((DISK_MIN_GB*1024)) oflag=direct >/dev/null 2>&1; then
    log "DDCHECK FAILED — cannot write ${DISK_MIN_GB}GB (quota?). STOP."; truncate -s 0 "$WORK/_ddprobe" 2>/dev/null; unlink "$WORK/_ddprobe" 2>/dev/null; return 1; fi
  truncate -s 0 "$WORK/_ddprobe" 2>/dev/null; unlink "$WORK/_ddprobe" 2>/dev/null; return 0; }

merged_count(){ ls "$MERGED" 2>/dev/null | grep -c '^yt_' ; }

log "PARALLEL SCALEUP START W=$W target=$TARGET round_clips=$ROUND_CLIPS seeds=$SEEDS"
[ -f "$CKPT" ] || { log "FATAL: encoder ckpt $CKPT missing"; exit 2; }
GEOARG=""; [ -f "$GEO" ] && { GEOARG="--geocalib-json $GEO"; log "GeoCalib FOUND -> per-video geometry"; } || log "GeoCalib absent -> fixed-HFOV fallback (re-runnable later)"

# split queries round-robin into W per-worker files
$V - "$S/queries_noncc.txt" "$W" "$S" <<'PY'
import sys
src, W, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
qs=[l.strip() for l in open(src) if l.strip() and not l.startswith('#')]
for i in range(W):
    open(f"{out}/wq{i}.txt","w").write("\n".join(qs[i::W])+"\n")
print(f"split {len(qs)} queries into {W} worker files")
PY

round=0; stall=0
while : ; do
  HAVE=$(merged_count)
  [ "$HAVE" -ge "$TARGET" ] && { log "target reached: $HAVE >= $TARGET"; break; }
  ddcheck || break
  round=$((round+1)); CAP=$((round*ROUND_CLIPS))
  log "ROUND $round: merged have $HAVE, per-worker cap -> $CAP; launching $W harvest workers"
  pids=""
  for i in $(seq 0 $((W-1))); do
    mkdir -p "$WORK/w$i"
    $V "$S/harvest_scaleup.py" --work "$WORK/w$i" \
       --queries-file "$S/wq$i.txt" --channels-file "$S/channels.txt" \
       --max-clips "$CAP" --per-video-clips "$PVC" --clip-frames 250 \
       --max-videos 60 --per-query 40 --allow-noncc $GEOARG \
       > "$WORK/w$i/harvest.log" 2>&1 &
    pids="$pids $!"
  done
  log "waiting on workers:$pids"
  wait $pids
  log "round $round harvest done; encoding + merging"
  ddcheck || log "WARN dd low after harvest"
  for i in $(seq 0 $((W-1))); do
    $V "$S/pseudo_label.py" --ckpt "$CKPT" \
       --clips-dir "$WORK/w$i/clips" --latents-dir "$WORK/w$i/latents" \
       --out "$R/pseudo_labels_w$i.json" > "$WORK/w$i/label.log" 2>&1
    # merge worker i's new latents into the global namespace (unique names), then drop worker copies
    for f in "$WORK/w$i"/latents/yt_*.pt; do
      [ -e "$f" ] || continue
      n=$(basename "$f" | sed -E 's/yt_0*([0-9]+)\.pt/\1/')
      gid=$(( i*100000 + n ))
      mv -f "$f" "$MERGED/yt_$(printf '%07d' "$gid").pt"
    done
  done
  HAVE2=$(merged_count)
  log "ROUND $round complete: merged latents = $HAVE2"
  # bank a combined harvest manifest snapshot
  $V - "$WORK" "$W" "$R" <<'PY'
import json,sys,glob,os
work,W,R=sys.argv[1],int(sys.argv[2]),sys.argv[3]
agg={"experiment":"youtube_idm_scaleup_parallel","workers":W,"per_worker":[],
     "n_clips_total":0,"videos_tried_total":0,"license_distribution":{}}
for i in range(W):
    p=f"{work}/w{i}/manifest.json"
    if os.path.exists(p):
        m=json.load(open(p)); agg["per_worker"].append({"w":i,"n_clips":m.get("n_clips"),
            "videos_tried":m.get("videos_tried"),"rejects":m.get("rejects"),
            "licenses":m.get("license_distribution"),"geometry":m.get("geometry")})
        agg["n_clips_total"]+=m.get("n_clips",0); agg["videos_tried_total"]+=m.get("videos_tried",0)
        for k,v in (m.get("license_distribution") or {}).items():
            agg["license_distribution"][k]=agg["license_distribution"].get(k,0)+v
agg["merged_latents"]=len(glob.glob(f"{work}/latents/yt_*.pt"))
json.dump(agg,open(f"{R}/harvest_manifest.json","w"),indent=2)
print("manifest merged_latents",agg["merged_latents"],"n_clips_total",agg["n_clips_total"])
PY
  if [ "$HAVE2" -le "$HAVE" ]; then stall=$((stall+1)); log "no new clips (stall $stall/2)";
    [ "$stall" -ge 2 ] && { log "pool exhausted at $HAVE2 — proceeding"; break; }
  else stall=0; fi
done

LAT=$(merged_count)
log "HARVEST+LABEL COMPLETE: $LAT merged clip-latents -> downstream (seeds=$SEEDS)"
echo "HARVEST_LABEL_DONE latents=$LAT $(date -u +%FT%TZ)" > "$R/STAGE_HARVEST_DONE"

$V "$S/run_youtube_pilot_downstream.py" \
   --flagship-ckpt "$CKPT" --yt-latents "$MERGED" \
   --work "$R" --out "$R/results_scaleup_downstream.json" \
   --seeds "$SEEDS" --pt-epochs 25 --ft-epochs 60 --with-parity-ref 2>&1 | tail -60

if [ -f "$R/results_scaleup_downstream.json" ]; then
  log "DOWNSTREAM DONE -> $R/results_scaleup_downstream.json"
  echo "SCALEUP_ALL_DONE latents=$LAT $(date -u +%FT%TZ)" > "$R/DONE"
else
  log "DOWNSTREAM produced no results JSON — see run.log"; echo "DOWNSTREAM_FAILED $(date -u +%FT%TZ)" > "$R/DONE"
fi
log "PARALLEL SCALEUP DRIVER EXIT"
