#!/usr/bin/env bash
# read_k.sh <ARM> : dev-box side (ARM = RLFT or RLAm). Card PREREG_V8F_PROOF_ADD40, notes N4 and N5. After the run lists of both bases exist (chain_k7.sh 0 and 1):
# per base the control reads from the pulled logs (plumb_read4.py; replay_read.py if the reader's dump of the checkpoint is on the dev box), the T1-kin pulls of the
# eight runs per base (Q=imagplan), the headway and family reads of the pairs, then read_k.py (controls first; names only if every control passed).
# Starts nothing on the pod GPU. Deletes nothing. Last line: ZZ READ-K <ARM> DONE.
A=${1:?arm}
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
PULL=/c/Users/Admin/v8proof_c/pull
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull"
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
cd $H || exit 1
ts() { date -u +%H:%M:%SZ; }
R=$H/imagplan/read_k/$A; RW=$W/imagplan/read_k/$A; mkdir -p "$R"
tg=$(echo "$A" | tr 'A-Z' 'a-z')
for k in 0 1; do
  B=${A}_s$k
  "$PY" imagplan/plumb_read4.py --pull imagplan/pull --arm $B --sfx x --out imagplan/read_plumb/PLUMB4_$B.json > "$R/plumb4_$B.log" 2>&1; echo "[$(ts)] plumb4 $B rc $?"
  if [ -f imaghead/dump/DUMP_$B.npz ] && [ -f imagplan/pull/IPix_$B/p45_IPix_$B.npz ]; then
    "$PY" imagplan/replay_read.py --p45 imagplan/pull/IPix_$B/p45_IPix_$B.npz --imag-log imagplan/pull/IPix_$B/imag_log_IPix_$B.npz --dump imaghead/dump/DUMP_$B.npz --grid imagplan/pull/eval_grid_windows.json --out imagplan/read_replay/REPLAY_$B.json > "$R/replay_$B.log" 2>&1; echo "[$(ts)] replay $B rc $?"
  else echo "[$(ts)] replay $B: the reader's dump or the p4/p5 log is not on the dev box"; fi
  for t in IPo IPi IPk IPkid1 IPkid3 IPks IPc IPl; do
    r=${t}x_$B
    [ -f "$PULL/$r/PULLED.txt" ] || Q=imagplan bash spatial/q3/pull_spatial_t1.sh "$r" 2>&1 | tail -n 1 | cut -c1-160
  done
done
n() { echo "${1}x_$A"; }
PAIRS="$(n IPl):$(n IPo),$(n IPk):$(n IPo),$(n IPk):$(n IPc),$(n IPk):$(n IPl),$(n IPc):$(n IPo),$(n IPks):$(n IPk),$(n IPk):$(n IPi),$(n IPi):$(n IPo)"
( cd $H/headway && "$PY" headway_read.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$RW/hw" > "$R/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd $H/spatial/q3 && "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$RW/q3" > "$R/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" imagplan/read_k.py --arm $A --sfx x --dir imagplan --read-dir "imagplan/read_k/$A" --seq-outs imagplan/deploy_k_${tg}0k2.out imagplan/deploy_k_${tg}1k2.out imagplan/cmp_subset_${A}_s0.out imagplan/cmp_subset_${A}_s1.out --out "imagplan/read_k/$A/READ_K_$A.json" 2>&1 | cut -c1-420
echo "ZZ READ-K $A DONE $(ts)"
