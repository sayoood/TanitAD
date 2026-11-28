#!/usr/bin/env bash
# read_k8.sh <ARM> <B1> <B2> : dev-box side of card PREREG_V8F_PROOF_ADD40 note N6 (ARM = RLFT or RLAm; bases 2 3). The form of read_k.sh with three changes:
#   * the two bases are arguments (read_k8.py --bases; made from read_k.py by make_read_k8.py, two edits);
#   * the headway and family reads use the reader versions r2 (row H-V8F-PROOF-84: same values, the STRATEGIC entry is written as NOT COMPUTED with reason and n);
#   * the T1-kin pulls of the tool runs go to F: (imagplan/pull_t1), since C: is at its floor.
# After chain_k8.sh has ended for the arm at both bases. Starts nothing on the pod GPU. Deletes nothing. Last line: ZZ READ-K8 <ARM> DONE.
A=${1:?arm}; B1=${2:?base}; B2=${3:?base}
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
PULL=/f/TanitAD-scratch/v8proof/imagplan/pull_t1
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull,$W/imagplan/pull_t1"
mkdir -p "$PULL"
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
cd $H || exit 1
ts() { date -u +%H:%M:%SZ; }
R=$H/imagplan/read_k8/$A; RW=$W/imagplan/read_k8/$A; mkdir -p "$R"
tg=$(echo "$A" | tr 'A-Z' 'a-z')
[ -f imagplan/read_k8.py ] || "$PY" imagplan/make_read_k8.py
for k in $B1 $B2; do
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
( cd $H/headway && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds $B1,$B2 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$RW/hw" > "$R/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd $H/spatial/q3 && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds $B1,$B2 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$RW/q3" > "$R/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" imagplan/read_k8.py --arm $A --sfx x --bases $B1,$B2 --dir imagplan --read-dir "imagplan/read_k8/$A" --seq-outs imagplan/deploy_k_${tg}${B1}k2.out imagplan/deploy_k_${tg}${B2}k2.out imagplan/cmp_subset_${A}_s$B1.out imagplan/cmp_subset_${A}_s$B2.out --out "imagplan/read_k8/$A/READ_K8_$A.json" > "$R/read_k8.log" 2>&1; echo "[$(ts)] read_k8 rc $?"
tail -n 30 "$R/read_k8.log" | cut -c1-330
echo "ZZ READ-K8 $A DONE $(ts)"
