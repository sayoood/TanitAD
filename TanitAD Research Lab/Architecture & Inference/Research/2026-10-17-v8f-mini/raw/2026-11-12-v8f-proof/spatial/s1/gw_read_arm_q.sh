#!/usr/bin/env bash
# gw_read_arm_q.sh <ARM> [max_hours]  -- the same script as gw_read_arm.sh for any pod queue: env GW_QUEUE (default spatial) and GW_TREE (default tree_spatial);
# arms P_s<N> of CELLPRED (card ADD16) are read as grid states of width 640 with GW_QUEUE=cellpred GW_TREE=tree_cellpred.
# gw_read_arm.sh <ARM> [max_hours]  -- dev-box side, script polling (no model turns): the trained-state reads of note N8 for ONE arm of SPATIAL stage 1
# (G_s0, W_s0, G_s1, W_s1). Same tools, same splits and same commands as the HIER baseline seeds (pod/pull_arms.sh + spatial/s1/auto_reads.sh);
# the only differences are the tree (tree_spatial, which holds the patched model), the run folder and the expected state width and kind.
#   1. wait for "ARM DONE <ARM>" in the pod's spatial chain log, and for the three-arm speed window (speed3) to be over;
#   2. extract the level-1 states on the pod (CPU, idle priority): train 600:1800 and eval 0:141;
#   3. pull both files binary-clean and compare sha256 with the pod's record;
#   4. run s1_eval_read.py (eval bank, out-of-sample) and s1_read.py (train probe set, in-sample), one read at a time on the dev box (lock folder);
#   5. report the size of the pulled train states (a copy; nothing is deleted by this script).
# This script chooses nothing: tools and rules are fixed (S1_DEFS.md, note N8). The lead reads the controls block of every verdict first.
A=${1:?arm}; MAXH=${2:-30}
Q=${GW_QUEUE:-spatial}; TREE=${GW_TREE:-tree_spatial}
case "$A" in G_s[0-9]|P_s[0-9]|PR_s[0-9]|RR_s[0-9]|RES_s[0-9]|ROLL_s[0-9]|PE_s[0-9]|PF_s[0-9]|PEF_s[0-9]) KIND=grid;; W_s[0-9]) KIND=mean;; *) echo "GW-FAIL $A: not an arm of stage 1"; exit 2;; esac
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
W=/f/TanitAD-scratch/v8proof/spatial; S1=$W/s1; ST=${GW_ST:-$W/s1states}
RR=/workspace/v8proof/runs/$Q; PB=/workspace/v8proof/probe; CH=/workspace/v8proof/chains
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
LOCK=$S1/gw_read.lockdir
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
ts() { date '+%Y-%m-%d %H:%M:%S'; }
sha() { sha256sum "$1" | cut -c1-64; }
nap() { timeout $(( $1 + 20 )) ping -n "$1" 127.0.0.1 > /dev/null 2>&1; }
free_gb() { powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" 2>/dev/null | tr -d '\r'; }
over() { [ $(( ($(date +%s) - t0) / 3600 )) -ge "$MAXH" ]; }
t0=$(date +%s)
mkdir -p "$ST"; cd "$S1" || exit 1

# 1. the arm is done, no stop marker, and the speed window is over
while true; do
  st=$(timeout 60 bash "$P" "grep -c -- 'ARM DONE $A\$' $CH/$Q/chain.log; ls $CH/$Q/${Q}_STOP.txt 2>/dev/null; grep -c -E 'SPEED3_DONE|TIMEOUT' $CH/speed3_DAG_s1.txt 2>/dev/null || echo 0" 2>/dev/null | tr -d '\r')
  n_done=$(echo "$st" | sed -n '1p'); n_sp=$(echo "$st" | tail -n 1)
  case "$st" in *STOP*) echo "[$(ts)] GW-FAIL $A: the spatial queue wrote a STOP marker"; exit 4;; esac
  if [ "${n_done:-0}" = "1" ] && [ "${n_sp:-0}" != "0" ]; then break; fi
  if over; then echo "[$(ts)] GW-FAIL $A: not done after $MAXH h (done=$n_done speed_window=$n_sp)"; exit 3; fi
  nap 600
done
echo "[$(ts)] $A: arm done on the pod, speed window over"

# 2. extraction on the pod, idempotent (the .npz.json record is written last)
for att in 1 2 3; do
  timeout 3000 bash "$P" "mkdir -p $PB/states && cd $PB && for sp in 'train 600 1800' 'eval 0 141'; do set -- \$sp; [ -f states/${A}_\$1.npz.json ] || nice -n 19 ionice -c3 /workspace/venv/bin/python extract_states.py --tree /workspace/v8proof/$TREE --ckpt $RR/$A/ckpt_last.pt --cache /workspace/v8proof/data/cache_pod --split \$1 --lo \$2 --hi \$3 --out states/${A}_\$1.npz > states/${A}_\$1.log 2>&1; done" > /dev/null 2>&1
  n=$(timeout 60 bash "$P" "ls $PB/states | grep -c -- '^${A}_.*npz.json\$'" 2>/dev/null | tr -d '\r' | tail -n 1)
  [ "${n:-0}" = "2" ] && break
  echo "[$(ts)] $A: extraction attempt $att left $n of 2 records"; nap 120
done
if [ "${n:-0}" != "2" ]; then
  echo "[$(ts)] GW-FAIL $A: extraction did not finish; last lines of the pod logs:"
  timeout 60 bash "$P" "tail -n 6 $PB/states/${A}_train.log $PB/states/${A}_eval.log 2>/dev/null | cut -c1-300" 2>/dev/null | sed '/hf_/d'
  exit 5
fi

# 3. pull and verify
ok=0
for att in 1 2 3; do
  ok=1
  for sp in train eval; do
    timeout 900 bash "$C" "$PB/states/${A}_$sp.npz" "$ST/${A}_$sp.npz"; timeout 120 bash "$C" "$PB/states/${A}_$sp.npz.json" "$ST/${A}_$sp.npz.json"
    want=$(grep -o '"out_sha256": "[0-9a-f]*"' "$ST/${A}_$sp.npz.json" | cut -d'"' -f4)
    [ -n "$want" ] && [ "$(sha "$ST/${A}_$sp.npz")" = "$want" ] || ok=0
  done
  [ $ok = 1 ] && break
  echo "[$(ts)] $A: pulled state files not verified (attempt $att)"; nap 60
done
if [ $ok != 1 ]; then echo "[$(ts)] GW-FAIL $A: state files not verified after 3 pulls"; exit 6; fi
ck=$(grep -o '"ckpt_sha256": "[0-9a-f]*"' "$ST/${A}_train.npz.json" | cut -d'"' -f4)
echo "[$(ts)] $A: states pulled and verified (ckpt ${ck:0:16}); $(ls -la "$ST/${A}_train.npz" "$ST/${A}_eval.npz" | awk '{printf "%s bytes ", $5}')"

# 4. the two reads, one arm at a time on this box
until mkdir "$LOCK" 2> /dev/null; do if over; then echo "[$(ts)] GW-FAIL $A: read lock not free after $MAXH h"; exit 7; fi; nap 120; done
trap 'rmdir "$LOCK" 2> /dev/null' EXIT
until f=$(free_gb); [ "${f:-0}" -ge 6 ]; do echo "[$(ts)] $A: free commit ${f} GB, waiting"; if over; then echo "[$(ts)] GW-FAIL $A: memory"; exit 8; fi; nap 300; done
"$PY" s1_eval_read.py run --states "$ST/${A}_eval.npz" --tag "$A" --expect-dim 640 --ckpt-sha256 "$ck" > "auto_s1eval_$A.log" 2>&1; r1=$?
echo "[$(ts)] $A: eval-bank read rc $r1"
"$PY" s1_read.py run --states "$ST/${A}_train.npz" --tag "$A" --expect-dim 640 --expect-state "$KIND" --ckpt-sha256 "$ck" > "auto_s1_$A.log" 2>&1; r2=$?
echo "[$(ts)] $A: train-probe read rc $r2"
if [ $r1 != 0 ] || [ $r2 != 0 ] || [ ! -f "RESULT_S1_$A.json" ] || [ ! -f "RESULT_S1EVAL_$A.json" ]; then
  echo "[$(ts)] GW-FAIL $A: a read did not finish (rc $r1 / $r2); last log lines:"; tail -n 5 "auto_s1eval_$A.log" "auto_s1_$A.log" | cut -c1-300
  exit 9
fi

# 5. the pulled train states are a copy of a file the pod can rebuild from the checkpoint: report the size
sz=$(ls -la "$ST/${A}_train.npz" | awk '{print $5}')
echo "[$(ts)] kept $ST/${A}_train.npz ($sz bytes; a copy, the pod can rebuild it from the checkpoint)"
echo "[$(ts)] GW READ DONE $A"
