#!/bin/bash
# resume_e3.sh [DRY|REAL] -- MM 4a3d05, 2026-11-23. RESUMES E3 (arm K2DC02) after the early-D1 pause (PAUSED.json, supervisor exit 7) AND after D1's own chain ended. The pause is a ONE-SHOT latch:
# the supervisor started again on the SAME manifest ($R/runs.d/r9e3_K2DC02.env, written by the chain) sees PAUSED.json and RESUMES from the latest checkpoint instead of pausing again (sup_refcv7.sh:207-208).
# It runs under the Thor GPU lock (waits up to 2 h, never forces it) and refuses a lock holder's GPU (compute app). Detached on Thor; log $R/logs/resume_e3_<ts>.out.
# Guards: NOPAUSED (PAUSED.json present, no done summary), NOMANIFEST, TREEDRIFT (launch_tree.commit == the manifest's COMMIT: a D1 restart on a NEWER tip moves the E3 tree aside and installs the new one at
# the same path, so the E3 token would no longer MATCH its tree: MM decision, see HANDOFF), CHAINLIVE (no gate3 chain with cwd $R: D1's chain must have ended), GPULOCK (DRY reads it, REAL waits for it).
# DRY only READS Thor. Ends ZZRESUMEE3-DRYOKZZ / ZZRESUMEE3-STARTEDZZ / ZZRESUMEE3-STOP-<why>ZZ.
export PATH="/usr/bin:/bin:$PATH"
MODE=${1:-DRY}; case "$MODE" in DRY|REAL) ;; *) echo "usage: resume_e3.sh [DRY|REAL]"; exit 2;; esac
R=/home/nvidia/r8ns/r9e3; ARM=r9e3_K2DC02; OUTD=$R/runs/r9e3-K2DC02; LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock
S=${R9K_LOG:-C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad}; OUT=$S/RESUME_E3.txt
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT"; }
th() { timeout 90 ssh -n -o ConnectTimeout=20 tanitad-thor "$1" 2>/dev/null | tr -d '\r'; }
FAILS=0
bad() { say "ZZRESUMEE3-STOP-$1ZZ $2"; if [ "$MODE" = DRY ]; then FAILS=$((FAILS+1)); else exit "${3:-5}"; fi; }
[ -n "$(th "test -e $OUTD/PAUSED.json && echo y")" ] || bad NOPAUSED "no $OUTD/PAUSED.json: E3 has not paused (still training, or never armed)"
[ -z "$(th "grep -l '\"done\"[[:space:]]*:[[:space:]]*true' $OUTD/summary.json 2>/dev/null")" ] || bad ALREADYDONE "$OUTD/summary.json is done: nothing to resume"
[ -n "$(th "test -s $R/runs.d/$ARM.env && echo y")" ] || bad NOMANIFEST "no $R/runs.d/$ARM.env"
MC=$(th "sed -n 's/^export COMMIT=//p' $R/runs.d/$ARM.env"); LC=$(th "cat $R/launch_tree.commit")
[ -n "$MC" ] && [ "$MC" = "$LC" ] || bad TREEDRIFT "the manifest's COMMIT (${MC:0:12}) != $R/launch_tree.commit (${LC:0:12}): a D1 restart installed a different tree; E3's token binds the old one. MM decision"
NCH=$(th "n=0; for p in \$(ps -eo pid,args | awk '/[k]it\\/r9gate3_chain.sh/ {print \$1}'); do [ \"\$(readlink /proc/\$p/cwd)\" = $R ] && n=\$((n+1)); done; echo \$n")
[ "$NCH" = 0 ] || bad CHAINLIVE "$NCH gate3 chain process(es) with cwd $R: D1's chain has not ended; resume only after ZZR9E1CHAIN-ALLDONEZZ for D1"
[ "$(th "tail -n 1 $R/gate3/CHAIN_STATUS.txt | grep -c ' ALLDONE '")" = 1 ] || bad D1NOTDONE "the last CHAIN_STATUS line is not ALLDONE (D1's chain: $(th "tail -n 1 $R/gate3/CHAIN_STATUS.txt" | cut -c1-120))"
if [ "$MODE" = DRY ]; then say "ZZRESUMEE3-DRYOKZZ $FAILS failing guard(s); the lock is $([ -n "$(th "flock -n $LOCKF true && echo FREE")" ] && echo free || echo held); nothing started"; exit 0; fi
TS=$(date -u +%Y%m%dT%H%M%SZ)
QO=$(timeout 90 ssh -n -o ConnectTimeout=20 tanitad-thor "cd $R && setsid nohup bash -c 'exec 9>$LOCKF; flock -w 7200 9 || { echo LOCK-TIMEOUT; exit 7; }; n=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c \"[0-9]\"); [ \"\$n\" = 0 ] || { echo GPU-NOT-EMPTY \$n; exit 7; }; bash $R/launch_tree/stack/ops/sup_refcv7.sh $ARM $R/runs.d' > logs/resume_e3_$TS.out 2>&1 < /dev/null & sleep 3; echo started" 2>&1 | tr -d '\r')
echo "$QO" | grep -q started || bad START "resume did not start: $QO" 8
say "ZZRESUMEE3-STARTEDZZ supervisor $ARM on its manifest under the Thor GPU lock (log $R/logs/resume_e3_$TS.out; run dir $OUTD)"
