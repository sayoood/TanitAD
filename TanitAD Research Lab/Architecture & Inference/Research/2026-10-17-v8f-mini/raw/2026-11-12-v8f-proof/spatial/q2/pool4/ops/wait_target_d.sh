#!/usr/bin/env bash
# wait_target_d.sh : card PREREG_V8F_PROOF_ADD19, second seed of PF (pod chain target_d, queued behind chain imag_a; the lever that notes
# ADD18_N2 section 6 and ADD18_N3 section 5 commit). Dev-box side, script polling, no model turns. It trains nothing and stops nothing.
#   1. wait for the chain to end (DONE or STOP marker), up to 30 h;
#   2. wait until the pod has 24 GB of memory free (the state extraction is one short CPU job at idle priority), up to 3 h, else nothing is read;
#   3. pull the checkpoint and the T1-kin dump, extract and pull the level-1 states, run the lead read of the card against the base arm P_s1
#      (the same scripts and arguments as PF_s0: read_t_arm.sh PF_s0 cell OOF_RB_P_s0.npz P_s0);
#   4. family and headway reads of the pair PF minus P on seeds 0 and 1 (cost check of the card).
# Deletes nothing. Last line: ZZ TARGET-D READS DONE or ZZ TARGET-D READS NOT RUN.
cd /f/TanitAD-scratch/v8proof || exit 1
export CK_ROOT=/c/Users/Admin/v8proof_c/ckpt PULL_ROOT=/c/Users/Admin/v8proof_c/pull RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
P=/c/Users/Admin/v8pod/pssh.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
H=/f/TanitAD-scratch/v8proof
RB=C:/Users/Admin/v8proof_c/ctxroll/rb
ROOTS="F:/TanitAD-scratch/v8proof/hiergap/pull,C:/Users/Admin/v8proof_c/pull"
RW=F:/TanitAD-scratch/v8proof/spatial/q2/pool4/read_pf2
R=$H/spatial/q2/pool4/read_pf2
ts() { date -u +%H:%M:%SZ; }
mkdir -p "$R"
k=0
while :; do
  k=$((k+1))
  s=$(timeout 60 bash "$P" "cd /workspace/v8proof/chains/target_d 2>/dev/null && ls target_d_DONE.txt target_d_STOP.txt 2>/dev/null | tr '\n' ' '" 2>/dev/null | tr -d '\r')
  case "$s" in
    *STOP*) echo "[$(ts)] chain target_d wrote its STOP marker"; timeout 60 bash "$P" "tail -n 5 /workspace/v8proof/chains/target_d/chain.log | cut -c1-240" 2>/dev/null | sed '/hf_/d'; echo "ZZ TARGET-D READS NOT RUN"; exit 2;;
    *DONE*) break;;
  esac
  [ "$k" -gt 360 ] && { echo "[$(ts)] chain target_d not ended after 30 h"; echo "ZZ TARGET-D READS NOT RUN"; exit 3; }
  sleep 300
done
echo "[$(ts)] chain target_d done"
k=0
while :; do
  f=$(timeout 60 bash "$P" "m=\$(cat /sys/fs/cgroup/memory.max); a=\$(grep '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); echo \$(( (m - a) / 1073741824 ))" 2>/dev/null | tr -cd "0-9")
  [ -n "$f" ] && [ "$f" -ge 24 ] && break
  k=$((k+1))
  [ "$k" -gt 36 ] && { echo "[$(ts)] pod memory free ${f:-unread} GB, below 24 GB after 3 h"; echo "ZZ TARGET-D READS NOT RUN"; exit 4; }
  sleep 300
done
echo "[$(ts)] pod memory free $f GB; PF_s1 start"
bash spatial/q3/wait_pull_arm.sh target_d PF_s1 60 || { echo "[$(ts)] pull rc=$?"; echo "ZZ TARGET-D READS NOT RUN"; exit 5; }
GW_QUEUE=target_d GW_TREE=tree_target GW_ST=/c/Users/Admin/v8proof_c/s1states bash spatial/s1/gw_read_arm_q.sh PF_s1 10 || echo "[$(ts)] states rc=$?"
bash spatial/q2/read_t_arm.sh PF_s1 cell $RB/OOF_RB_P_s1.npz P_s1 60 || echo "[$(ts)] lead read rc=$?"
( cd $H/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "PF:P" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$RW/hw_pf2" > "$R/hw_pf2.log" 2>&1 ); echo "[$(ts)] headway rc $? : $(tail -n 1 "$R/hw_pf2.log" | cut -c1-200)"
( cd $H/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "PF:P" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$RW/q3_pf2" > "$R/q3_pf2.log" 2>&1 ); echo "[$(ts)] families rc $? : $(tail -n 1 "$R/q3_pf2.log" | cut -c1-200)"
echo "ZZ TARGET-D READS DONE $(ts)"
