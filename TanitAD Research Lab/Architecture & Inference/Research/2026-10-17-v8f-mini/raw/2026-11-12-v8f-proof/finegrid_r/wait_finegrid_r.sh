#!/bin/bash
# wait_finegrid_r.sh -- dev box (script polling, no model turns). Card PREREG_V8F_PROOF_ADD43 with note N1.
# 1. polls the pod every 300 s for the end marker of pod_finegrid_r_run.sh (FINEGRID_R_DONE.txt or FINEGRID_R_STOP.txt; bound 12 h);
# 2. on DONE starts the reader ON THE POD (finegrid/rk/run_pod_read_r.sh; nohup setsid; its own marker refuses a second start; one CPU job) and polls for READ_R_DONE.txt (bound 4 h);
# 3. pulls the reader's small outputs and the run's small records into finegrid_r/out_pod/ and compares sha256 on both sides.
# It starts nothing on the pod GPU and deletes nothing. Last line: ZZ FINEGRID-R-WAIT <state>.
set -u
cd /f/TanitAD-scratch/v8proof/finegrid_r || exit 9
P=/c/Users/Admin/v8pod/pssh.sh
C=/c/Users/Admin/v8pod/pcat.sh
RUN=/workspace/v8proof/runs/finegrid_r
RK=/workspace/v8proof/finegrid/rk
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
mkdir -p out_pod
n=0; st=
while :; do
  s=$(pod 60 "[ -e $RUN/FINEGRID_R_DONE.txt ] && echo DONE; [ -e $RUN/FINEGRID_R_STOP.txt ] && echo STOP; tail -n 1 /workspace/v8proof/runs/finegrid_r.log 2>/dev/null | cut -c1-200; echo MARK")
  if echo "$s" | grep -q '^STOP$'; then st=STOP; break; fi
  if echo "$s" | grep -q '^DONE$'; then st=DONE; break; fi
  n=$((n+1)); [ $((n % 6)) = 1 ] && echo "[$(ts)] poll $n: $(echo "$s" | grep -v '^MARK$' | tail -n 1 | cut -c1-200)"
  [ "$n" -gt 144 ] && { echo "ZZ FINEGRID-R-WAIT TIMEOUT $(ts)"; exit 8; }
  sleep 300
done
echo "[$(ts)] run ended on the pod: $st"
pod 60 "tail -n 6 /workspace/v8proof/runs/finegrid_r.log | cut -c1-260; cat $RUN/FINEGRID_R_STOP.txt 2>/dev/null | cut -c1-300; cat $RUN/R1_SAMPLER.json 2>/dev/null | cut -c1-400; du -sm /workspace/v8proof/finegrid_r/states 2>/dev/null"
if [ "$st" != DONE ]; then echo "ZZ FINEGRID-R-WAIT STOP: the run wrote its STOP marker; no read ($(ts))"; exit 0; fi
pod 60 "cd $RK && [ -f run_pod_read_r.sh ] && [ -f 'F:/TanitAD-scratch/v8proof/finegrid_r/finegrid_r_read.py' ] && [ -f 'F:/TanitAD-scratch/v8proof/finegrid_r/controls_devbox_r.json' ] && echo FILES-OK" | grep -q FILES-OK || { echo "ZZ FINEGRID-R-WAIT STOP: the reader files are not in $RK ($(ts))"; exit 3; }
timeout 60 bash "$P" "cd $RK && nohup setsid bash run_pod_read_r.sh > pod_read_r.nohup 2>&1 < /dev/null & sleep 1; echo started" > /dev/null 2>&1
echo "[$(ts)] reader started on the pod"
n=0
while :; do
  s=$(pod 60 "cat $RK/READ_R_DONE.txt 2>/dev/null; echo MARK")
  echo "$s" | grep -q '^rc=' && break
  n=$((n+1)); [ "$n" -gt 48 ] && { echo "ZZ FINEGRID-R-WAIT STOP: the reader has not ended in 4 h ($(ts))"; exit 8; }
  sleep 300
done
echo "[$(ts)] reader ended: $(echo "$s" | grep '^rc=' | cut -c1-80)"
pod 60 "tail -n 4 $RK/pod_read_r.nohup | cut -c1-240"
for pr in "$RK/out_read_r/RESULT_FINEGRID_R.json:RESULT_FINEGRID_R.json" "$RK/out_read_r/VERDICT_FINEGRID_R.md:VERDICT_FINEGRID_R.md" "$RK/read_r.out:read_r.out" "$RUN/R1_SAMPLER.json:R1_SAMPLER.json" \
          "$RUN/G2_GRIDS.json:G2_GRIDS.json" "$RUN/train_A/TRAIN_DONE.json:train_A_TRAIN_DONE.json" "$RUN/train_B/TRAIN_DONE.json:train_B_TRAIN_DONE.json" "$RUN/train_A/r1_batches.json:train_A_r1_batches.json" \
          "$RUN/train_B/r1_batches.json:train_B_r1_batches.json" "$RUN/p3_A/TRAIN_DONE.json:p3_A_TRAIN_DONE.json" "$RUN/p3_B/TRAIN_DONE.json:p3_B_TRAIN_DONE.json" "/workspace/v8proof/runs/finegrid_r.log:finegrid_r.log" \
          "/workspace/v8proof/finegrid_r/checks_pod.log:checks_pod.log" "/workspace/v8proof/finegrid_r/controls_devbox_r.json:controls_devbox_r_pod.json"; do
  r=${pr%%:*}; l=${pr##*:}
  rs=$(pod 60 "sha256sum '$r' 2>/dev/null | cut -c1-64")
  [ -n "$rs" ] || { echo "[$(ts)] not on the pod: $r"; continue; }
  timeout 300 bash "$C" "$r" "out_pod/$l" < /dev/null > /dev/null 2>&1
  [ "$(sha256sum "out_pod/$l" 2>/dev/null | cut -c1-64)" = "$rs" ] && echo "[$(ts)] pulled $l sha256 ${rs:0:16}" || echo "[$(ts)] ZZ SHA MISMATCH $l"
done
head -n 30 out_pod/VERDICT_FINEGRID_R.md 2>/dev/null | cut -c1-400
echo "ZZ FINEGRID-R-WAIT DONE $(ts)"
