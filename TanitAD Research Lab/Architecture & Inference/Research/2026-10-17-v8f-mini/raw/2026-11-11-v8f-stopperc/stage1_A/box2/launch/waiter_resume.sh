#!/bin/bash
# waiter_resume.sh (MM 2026-11-13 ~00:20): RESUMES the STOPPERC stage-1 chain after its disk-low stop at the W smoke (2026-11-13 00:07:49,
# "ZZ STOPPERC-STOP wsmoke disk-low rc=6 :: box 2 D: free 2 GB < 5 GB"; ship done). Same PI go ("Yes, run arm A", 2026-11-12 ~20:30), same pass check as
# waiter.sh. It deletes nothing: it waits until box 2's D: has >= 5.5 GB free (ssh probe every 10 min), the busy marker is absent and the pass file still
# covers the package, then runs the chain with STOPPERC_RESUME=1 (the chain moves its STOP.txt aside and skips ship). Only a STOP whose head is a disk-low
# or box2-busy stop is resumed (at most 6 tries); any other STOP is left for the MM. Deadline 2026-11-13 14:00. Sentinels: WAITER_RESUME_END.txt / _STOP.txt.
set -u
D=C:/Users/Admin/stopperc_launch
LOG=$D/waiter_resume.log
BUSY=C:/Users/Admin/box2_ship/BOX2_BUSY.txt
PASS=C:/Users/Admin/stopperc_launch/MM_DRYRUN_OK.json   # written by the MM after reviewing the agent's C:/Users/Admin/stopperc_dry/DRYRUN_PASS.json (same schema)
SLOW=C:/Users/Admin/slowwm_state
ST=C:/Users/Admin/stopperc_state
TH=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory
PKG=$TH/stopperc/stage1_A/box2
BAL=$TH/stopbal/box2
PYV=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
DEADLINE=$(date -d '2026-11-13 22:00' +%s)
NEED_MB=5120   # the chain's own run gate: floor(D: free / 1 GiB) >= 5 (MM 2026-11-13 ~10:00: D: sat at 5593 MB for 9.5 h, 39 MB under the 5.5 GB margin this waiter added; each run is pulled and cleaned off box 2, ESTIMATED <= 1.5 GB)
GPU_BAR=3000   # PI 2026-11-13 ~14:20 ("Raise to 3 GB"): run stage 1 beside the desktop on box 2; passed to the chain as STOPBAL_GPU_IDLE_MIB
lg() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
last=""
note() { [ "$1" != "$last" ] && lg "$1"; last="$1"; }
pass_ok() {
  "$PYV" -I - "$PASS" "$PKG/stopperc_chain.sh" "$PKG/stopperc_check.py" "$PKG/argvs_stopperc.txt" "$PKG/make_argvs_stopperc.py" "$PKG/jobs/stopperc_ship_job.cmd" "$PKG/jobs/stopperc_stage_job.cmd" \
    "$BAL/stages/stopbal_common.sh" "$BAL/stopbal_check.py" "$BAL/stopbal_ship_tree.py" "$BAL/stopbal_refresh.py" << 'PYEOF'
import hashlib, json, os, sys
p = sys.argv[1]
if not os.path.exists(p): print("no pass file"); sys.exit(1)
j = json.load(open(p, encoding="utf-8"))
if j.get("result") != "PASS" or j.get("failed", 0) != 0: print("pass file does not read PASS"); sys.exit(1)
have = {str(v).lower() for v in (j.get("sha256") or {}).values()}
miss = [f for f in sys.argv[2:] if hashlib.sha256(open(f, "rb").read()).hexdigest() not in have]
if miss: print("files changed since the dry run: " + " ".join(os.path.basename(m) for m in miss)); sys.exit(1)
print("pass ok"); sys.exit(0)
PYEOF
}
lg "resume waiter start pid $$"
tries=0
b2_gpu_busy() {   # empty when the chain's own GPU gate would pass: no python.exe on box 2 and GPU used < $GPU_BAR MiB (the chain reads the same value as STOPBAL_GPU_IDLE_MIB)
  local n m; n="$(timeout 120 "$(command -v ssh)" tanitad-box2 'tasklist /FI "IMAGENAME eq python.exe" /NH' 2>/dev/null | grep -ci 'python.exe')"; n="${n:-0}"
  m="$(timeout 120 "$(command -v ssh)" tanitad-box2 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits' 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | head -1)"
  if [ "$n" = 0 ] && [ -n "$m" ] && [ "$m" -lt "$GPU_BAR" ]; then return 0; fi; echo "python.exe ${n}, GPU ${m:-?} MiB used, bar $GPU_BAR"; }
b2_free_mb() { timeout 120 "$(command -v ssh)" tanitad-box2 "powershell -NoProfile -Command [math]::Floor((Get-PSDrive D).Free/1MB)" 2>/dev/null | tr -d '' | grep -E '^[0-9]+$' | tail -1; }
while :; do
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "ZZ STOPPERC-WAITER-RESUME-STOP deadline 2026-11-13 22:00 passed; last state: $last" > "$D/WAITER_RESUME_STOP.txt"; lg "deadline passed"; exit 3; }
  [ -e "$ST/DONE.txt" ] && { lg "chain DONE present: nothing to resume"; exit 0; }
  if [ -e "$ST/STOP.txt" ] && ! grep -Eq '^ZZ STOPPERC-STOP [a-zA-Z0-9_-]+ (disk-low|box2-busy)' "$ST/STOP.txt"; then echo "ZZ STOPPERC-WAITER-RESUME-STOP a non-resumable STOP: $(head -c 300 "$ST/STOP.txt")" > "$D/WAITER_RESUME_STOP.txt"; lg "non-resumable STOP: left for the MM"; exit 4; fi
  msg="$(pass_ok)"; if [ $? -ne 0 ]; then note "waiting: dry run ($msg)"; sleep 600; continue; fi
  if [ -e "$BUSY" ]; then note "waiting: box 2 busy ($(head -c 80 "$BUSY"))"; sleep 600; continue; fi
  f="$(b2_free_mb)"; if [ -z "$f" ] || [ "$f" -lt "$NEED_MB" ]; then note "waiting: box 2 D: free ${f:-?} MB < $NEED_MB MB"; sleep 600; continue; fi
  g="$(b2_gpu_busy)"; if [ -n "$g" ]; then note "waiting: box 2 GPU not idle ($g)"; sleep 600; continue; fi   # mirrors the chain gate (MM 11-13 ~10:05) so a desktop-held GPU spends no try
  tries=$((tries+1)); [ "$tries" -le 6 ] || { echo "ZZ STOPPERC-WAITER-RESUME-STOP 6 resume tries used" > "$D/WAITER_RESUME_STOP.txt"; exit 5; }
  lg "resume try $tries: box 2 D: free $f MB, box 2 free, dry run ok; STOPPERC_RESUME=1, GPU bar $GPU_BAR MiB"
  STOPPERC_SSH="$(command -v ssh)" STOPPERC_SFTP="$(command -v sftp)" STOPPERC_RESUME=1 STOPBAL_GPU_IDLE_MIB="$GPU_BAR" bash "$PKG/stopperc_chain.sh" >> "$D/chain_console.log" 2>&1; rc=$?
  lg "chain exited rc=$rc"
  if [ "$rc" -eq 0 ] || [ -e "$ST/DONE.txt" ]; then echo "ZZ STOPPERC-WAITER-RESUME-END chain rc=$rc $(date '+%F %T')" > "$D/WAITER_RESUME_END.txt"; exit 0; fi
  if [ -e "$ST/STOP.txt" ] && grep -Eq '^ZZ STOPPERC-STOP [a-zA-Z0-9_-]+ (disk-low|box2-busy)' "$ST/STOP.txt"; then lg "resumable stop again: $(head -c 160 "$ST/STOP.txt")"; sleep 600; continue; fi
  echo "ZZ STOPPERC-WAITER-RESUME-END chain rc=$rc $(date '+%F %T')" > "$D/WAITER_RESUME_END.txt"; exit 0
done
