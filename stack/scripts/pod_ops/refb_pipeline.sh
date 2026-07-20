#!/usr/bin/env bash
# REF-B Phase-0 pipeline supervisor (pod1) -- chained, resumable, detached.
#   egomotion gap-fill -> camera fetch -> build (TRAIN-ONLY) -> parity gate -> REF-B (auto-resume)
# Survives SSH disconnect via setsid+nohup. Resumable across restarts via log
# sentinels (each stage is skipped if already complete). Sequential single
# Python process throughout (respects the ~62GB cgroup cap; no worker pool).
#
# PARITY IS GATING: REF-B is launched ONLY if the build produced the exact pod2
# train key with zero per-clip-intrinsics fallbacks, zero disk-full errors, and
# the disk safety-valve never fired. Otherwise it writes PARITY_FAILED and stops.
set -u
ROOT=/workspace/data/physicalai_phase0
STACK=/workspace/TanitAD/stack
OUT=/workspace/experiments/refb-phase0-30k
EXPECT_KEY=physicalai-train-e438721ae894
HB=$ROOT/pipeline.heartbeat
STAGEF=$ROOT/pipeline.stage
PLOG=$ROOT/pipeline.log
mkdir -p "$ROOT" "$OUT"
export HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9]+' /workspace/TanitAD/Keys.txt | head -1)"
export TANITAD_PHYSICALAI_ROOT="$ROOT"
export PAI_BUILD_SPLITS=train      # honored by the local build_pai_cache.py edit
cd "$STACK"

log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$PLOG"; }
setstage(){ echo "$1" > "$STAGEF"; log "STAGE=$1"; }
freeg(){ df -BG /workspace 2>/dev/null | tail -1 | awk '{print $(NF-2)}' | tr -dc '0-9'; }

# ---- background heartbeat every 30s: timestamp + stage + free disk + refb step
(
  while true; do
    echo "$(date -u +%FT%TZ) stage=$(cat "$STAGEF" 2>/dev/null) free_gib=$(freeg) refb_step=$(grep -o '"step": [0-9]*' "$OUT/train.log" 2>/dev/null | tail -1 | grep -o '[0-9]*')" > "$HB"
    sleep 30
  done
) & HBPID=$!

# ---- background disk safety-valve: never let /workspace wedge at 100%.
# If free < 12 GiB, reap raw mp4s (largest first) until free > 25 GiB, and DROP
# a marker so the parity gate fails (guard firing under train-only means our disk
# estimate was wrong -> the built set may be pod1-trimmed -> needs human review).
(
  while true; do
    f=$(freeg)
    if [ -n "$f" ] && [ "$f" -lt 12 ]; then
      touch "$ROOT/DISK_GUARD_FIRED"
      log "disk-guard: free=${f}GiB<12 -> reaping mp4s (PARITY marker dropped)"
      find "$ROOT/r0/camera_front_wide" -name '*.mp4' -printf '%s %p\n' 2>/dev/null \
        | sort -rn | awk '{print $2}' | while read -r m; do
            rm -f "$m"
            g=$(freeg); [ -n "$g" ] && [ "$g" -gt 25 ] && break
          done
    fi
    sleep 25
  done
) & GUARDPID=$!
trap 'kill $HBPID $GUARDPID 2>/dev/null' EXIT

# ===================== STAGE 1: EGOMOTION (parity-safe gap-fill) =============
if ! grep -q 'EGO_ALL_PRESENT' "$ROOT/ego.log" 2>/dev/null; then
  setstage EGO; n=0
  until grep -q 'EGO_ALL_PRESENT' "$ROOT/ego.log" 2>/dev/null; do
    n=$((n+1)); log "ego attempt $n"
    python /workspace/fetch_ego.py >> "$ROOT/ego.log" 2>&1
    grep -q 'EGO_ALL_PRESENT' "$ROOT/ego.log" && break
    [ $n -ge 12 ] && { setstage EGO_FAILED; log "ego FAILED after $n attempts"; exit 11; }
    sleep 20
  done
fi
setstage EGO_DONE

# ===================== STAGE 2: CAMERA (mp4s + timestamps) ===================
if ! grep -q 'camera fetch done' "$ROOT/fetch.log" 2>/dev/null; then
  setstage CAMERA; n=0
  until grep -q 'camera fetch done' "$ROOT/fetch.log" 2>/dev/null; do
    n=$((n+1)); log "camera attempt $n"
    python scripts/physicalai_r0.py fetch-camera --max-chunks 250 >> "$ROOT/fetch.log" 2>&1
    grep -q 'camera fetch done' "$ROOT/fetch.log" && break
    [ $n -ge 60 ] && { setstage CAMERA_FAILED; log "camera FAILED after $n attempts"; exit 12; }
    log "camera incomplete; retry (resumes, skips complete chunks) in 30s"; sleep 30
  done
fi
setstage CAMERA_DONE

# ===================== STAGE 3: BUILD (train split only) =====================
if ! grep -q 'PAI_CACHE_DONE' "$ROOT/build.log" 2>/dev/null; then
  setstage BUILD; n=0
  until grep -q 'PAI_CACHE_DONE' "$ROOT/build.log" 2>/dev/null; do
    n=$((n+1)); log "build attempt $n (PAI_BUILD_SPLITS=$PAI_BUILD_SPLITS)"
    python scripts/build_pai_cache.py --root "$ROOT" >> "$ROOT/build.log" 2>&1
    grep -q 'PAI_CACHE_DONE' "$ROOT/build.log" && break
    [ $n -ge 40 ] && { setstage BUILD_FAILED; log "build FAILED after $n attempts"; exit 13; }
    log "build incomplete; retry (resumes per-source) in 30s"; sleep 30
  done
fi
setstage BUILD_DONE

# ===================== STAGE 4: PARITY GATE (hard, canonical skipset) ========
# Requires: key e438721ae894, zero intrinsics-fallback, zero disk-full, disk
# guard never fired, AND the canonical strict-parity skipset match (2376 built /
# 24 skips / sha256==f09e44db... via /workspace/parity_skipset.sh if present,
# else a standard serialization). Anything short -> PARITY_HOLD, no REF-B launch.
setstage GATE
rm -f "$ROOT/PARITY_OK" "$ROOT/PARITY_HOLD"
KEYDIR="$ROOT/_epcache/$EXPECT_KEY"
FB=$(grep -c 'no per-clip intrinsics' "$ROOT/build.log" 2>/dev/null | tr -dc '0-9'); FB=${FB:-0}
DF=$(grep -c 'No space left on device' "$ROOT/build.log" 2>/dev/null | tr -dc '0-9'); DF=${DF:-0}
GUARD=no; [ -f "$ROOT/DISK_GUARD_FIRED" ] && GUARD=yes
# (1) verify strict parity as-built
python /workspace/compute_skipset.py > "$ROOT/skipset.out" 2>&1; SKIPRC=$?
# (2) if pod1 DECODED the 24 corrupt clips (2400/0-skip, like pod3 REF-A), do the
#     Sayed-authorized reconciliation to pod2's canonical 2376 (subset-safe), then
#     re-verify. The hash re-check below means a wrong drop can only HOLD, not launch.
if [ "$SKIPRC" -ne 0 ]; then
  log "gate: as-built parity failed (rc=$SKIPRC) -> attempting canonical reconciliation"
  python /workspace/reconcile_to_pod2.py > "$ROOT/reconcile.out" 2>&1; RRC=$?
  log "reconcile rc=$RRC: $(tail -1 "$ROOT/reconcile.out" 2>/dev/null)"
  if [ "$RRC" -eq 0 ]; then
    python /workspace/compute_skipset.py > "$ROOT/skipset.out" 2>&1; SKIPRC=$?
  fi
fi
SKIPV=$(grep '^VERDICT' "$ROOT/skipset.out" | head -1)
NEP=$(ls "$KEYDIR"/ep_*.pt 2>/dev/null | wc -l | tr -dc '0-9')
{ echo "$SKIPV"; echo "--- skipset ---"; cat "$ROOT/skipset.out"; [ -f "$ROOT/reconcile.out" ] && { echo "--- reconcile ---"; cat "$ROOT/reconcile.out"; }; } > "$ROOT/PARITY_PROOF.txt"
BASE_OK=no
[ -d "$KEYDIR" ] && [ "$FB" -eq 0 ] && [ "$DF" -eq 0 ] && [ "$GUARD" = no ] && BASE_OK=yes
HDR="base_ok=$BASE_OK skiprc=$SKIPRC $SKIPV key=$EXPECT_KEY train_ep=${NEP:-0} fallback=$FB diskfull=$DF guard=$GUARD"
if [ "$BASE_OK" = yes ] && [ "$SKIPRC" -eq 0 ]; then
  { echo "PARITY_OK $HDR $(date -u +%FT%TZ)"; echo "---"; cat "$ROOT/PARITY_PROOF.txt"; } > "$ROOT/PARITY_OK"
  log "PARITY_OK $HDR"
else
  { echo "PARITY_HOLD $HDR $(date -u +%FT%TZ)"; echo "---"; cat "$ROOT/PARITY_PROOF.txt"; } > "$ROOT/PARITY_HOLD"
  setstage PARITY_HOLD
  log "PARITY_HOLD $HDR -> NOT launching REF-B (human review required)"
  exit 20
fi

# ===================== STAGE 5: REF-B (auto-resume from ckpt.pt) =============
setstage REFB; n=0
until [ -f "$OUT/metrics.json" ] || grep -q '"done": true' "$OUT/train.log" 2>/dev/null; do
  n=$((n+1)); log "refb launch/resume attempt $n"
  python scripts/refb_train.py --data-root "$ROOT/_epcache" --out "$OUT" \
    --steps 30000 --grad-checkpoint --save-every 500 >> "$OUT/train.log" 2>&1
  rc=$?; log "refb exited rc=$rc"
  { [ -f "$OUT/metrics.json" ] || grep -q '"done": true' "$OUT/train.log" 2>/dev/null; } && break
  [ $n -ge 400 ] && { setstage REFB_ABORT; log "refb too many restarts"; exit 30; }
  sleep 10
done
setstage REFB_DONE
log "pipeline complete"
