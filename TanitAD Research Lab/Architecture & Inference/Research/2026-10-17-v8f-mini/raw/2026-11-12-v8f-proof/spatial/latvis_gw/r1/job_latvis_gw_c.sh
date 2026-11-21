#!/usr/bin/env bash
# C: variant (run folder on C: because F: is at its free-space floor; markers carry the run tag $1; the arms are listed after the prep).
# LATVIS on the trained states of SPATIAL stage 1 (PREREG_V8F_PROOF_ADD12 section 4). Start from Git bash, in the background:
#   bash /f/TanitAD-scratch/v8proof/spatial/latvis_gw/job_latvis_gw.sh > /dev/null 2>&1 &
# Steps: wait for free commit >= 10 GB; arms with a matching sha256; tree check; frames (copy of the baseline's if it exists, else prep on the CPU);
# the read (collect + predicted latents on the CPU, decoders on the GPU behind the shared lock, taken inside latvis_gw.py by fullres_encode.take_lock:
# same lock file, same wait-for-idle rule, never forced); LATVIS_GW_DONE_$RUN.txt or LATVIS_GW_STOP_$RUN.txt. Deletes nothing.
HERE=/c/Users/Admin/v8proof_c/latvis_gw
RUN=${1:-r1}
BASEDIR=/f/TanitAD-scratch/v8proof/latvis
CK=F:/TanitAD-scratch/v8proof/spatial/ckpt
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$HERE/latvis_gw_job.log
cd "$HERE" || exit 1
unset CUDA_VISIBLE_DEVICES LATVIS_NOLOCK
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
stop() { echo "ZZ LATVIS-GW STOP $*" | tee -a "$LOG" > "$HERE/LATVIS_GW_STOP_$RUN.txt"; echo "ZZ LATVIS-GW STOP $*"; exit 1; }

# 1. free commit >= 10 GB (as job_latvis.cmd)
WAITS=0
while true; do
  FREE=$(powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" | tr -d '\r\n ')
  case "$FREE" in ''|*[!0-9]*) FREE=0;; esac
  [ "$FREE" -ge 10 ] && break
  WAITS=$((WAITS+1))
  [ "$WAITS" -ge 240 ] && stop "commit-wait-timeout free=${FREE}GB"
  say "wait $WAITS free $FREE GB"
  sleep 120
done
say "start free $FREE GB $(date)"

# 3. tree against the pod pins
"$PY" latvis_gw.py check-tree >> "$LOG" 2>&1 || stop "tree-hash-mismatch see latvis_gw_job.log"

# 4. frames: reuse the baseline's artefacts when they exist, else decode (CPU, resumable, about 35 min)
if [ -f "$BASEDIR/frames64.u8" ] && [ -f "$BASEDIR/prep_done.json" ] && [ ! -f "$HERE/frames64.u8" ]; then
  cp "$BASEDIR/frames64.u8" "$HERE/frames64.u8" && cp "$BASEDIR/prep_done.json" "$HERE/prep_done_gw.json" && say "frames copied from the baseline read"
fi
"$PY" latvis_gw.py prep >> "$LOG" 2>&1 || stop "prep-failed see latvis_gw_job.log"

# 2. arms whose files exist with a matching sha256 (others are skipped and named)
ARGS=()
READ=""
for A in G_s0 G_s1 W_s0 W_s1; do
  D=$CK/$A
  if [ ! -f "$D/ckpt_last.pt" ] || [ ! -f "$D/ckpt_last.sha256" ]; then say "SKIP $A: checkpoint or sha256 file absent"; continue; fi
  WANT=$(awk '{print tolower($1)}' "$D/ckpt_last.sha256" | head -n 1)
  GOT=$(sha256sum "$D/ckpt_last.pt" | awk '{print tolower($1)}')
  if [ "$WANT" != "$GOT" ]; then say "SKIP $A: sha256 mismatch (file $WANT, bytes $GOT)"; continue; fi
  case "$A" in G_*) KIND=grid;; *) KIND=mean;; esac
  ARGS+=(--arm "$A,$D/ckpt_last.pt,640,$KIND")
  READ="$READ $A"
done
[ -z "$READ" ] && stop "no-arm-with-matching-sha256"
say "arms read:$READ"

# 5. the read (the frame identity check against the baseline sheet runs before the GPU lock is taken)
"$PY" latvis_gw.py run "${ARGS[@]}" >> "$LOG" 2>&1
RC=$?
if [ "$RC" = "2" ]; then stop "control-missed rc 2 see VERDICT_LATVIS_GW.md and latvis_gw_job.log"; fi
if [ "$RC" != "0" ]; then stop "run-failed rc $RC see latvis_gw_job.log"; fi
echo "ZZ LATVIS-GW DONE arms:$READ $(date)" | tee -a "$LOG" > "$HERE/LATVIS_GW_DONE_$RUN.txt"
echo "ZZ LATVIS-GW DONE"
exit 0
