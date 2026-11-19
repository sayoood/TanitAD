#!/usr/bin/env bash
# read_i9.sh <ARM> <B1> <B2> : dev-box side of card PREREG_V8F_PROOF_ADD40, source cand-imag on an arm that meets control p5b at both bases (note N6 section 4;
# ARM = RLA3, bases 0 1). The form of read_k8.sh for the full imagined source: reader read_i9.py (names of section 4 of the card as written), control p3 in the
# subset form through pod_cmp_subset_i.sh, plumbing controls through plumb_read3.py WITH the log of the cand-cv run (written to PLUMB3cv_*.json; the PLUMB3_*.json
# files of row H-V8F-PROOF-91 are not touched). Every Python job runs under the capped launcher (ops/pycap.sh, one job at a time).
# After chain_k9.sh has ended. Starts nothing on the pod GPU. Deletes nothing. Last line: ZZ READ-I9 <ARM> DONE.
A=${1:?arm}; B1=${2:?base}; B2=${3:?base}
PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh
P=/c/Users/Admin/v8pod/pssh.sh
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
PULL=/f/TanitAD-scratch/v8proof/imagplan/pull_t1
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull,$W/imagplan/pull_t1"
mkdir -p "$PULL"
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export PYCAP_MB=${PYCAP_MB:-2000}
cd $H || exit 1
ts() { date -u +%H:%M:%SZ; }
R=$H/imagplan/read_i9/$A; RW=$W/imagplan/read_i9/$A; mkdir -p "$R"
tg=$(echo "$A" | tr 'A-Z' 'a-z')
for k in $B1 $B2; do
  B=${A}_s$k
  timeout 240 bash "$P" "bash -s -- $B" < imagplan/pod_cmp_subset_i.sh 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "imagplan/cmp_subset_i_$B.out"; cut -c1-200 "imagplan/cmp_subset_i_$B.out"
  cv=()
  [ -f imagplan/pull/IPcx_$B/imag_log_IPcx_$B.npz ] && cv=(--cv-log imagplan/pull/IPcx_$B/imag_log_IPcx_$B.npz)
  "$PY" imagplan/plumb_read3.py --p45 imagplan/pull/IPix_$B/p45_IPix_$B.npz --imag-log imagplan/pull/IPix_$B/imag_log_IPix_$B.npz ${cv[@]+"${cv[@]}"} --responds yes --out imagplan/read_plumb/PLUMB3cv_$B.json > "$R/plumb3cv_$B.log" 2>&1; echo "[$(ts)] plumb3 (with the cv log: ${#cv[@]} args) $B rc $?"
  if [ -f imaghead/dump/DUMP_$B.npz ] && [ -f imagplan/pull/IPix_$B/p45_IPix_$B.npz ]; then
    "$PY" imagplan/replay_read.py --p45 imagplan/pull/IPix_$B/p45_IPix_$B.npz --imag-log imagplan/pull/IPix_$B/imag_log_IPix_$B.npz --dump imaghead/dump/DUMP_$B.npz --grid imagplan/pull/eval_grid_windows.json --out imagplan/read_replay/REPLAY_$B.json > "$R/replay_$B.log" 2>&1; echo "[$(ts)] replay $B rc $?"
  else echo "[$(ts)] replay $B: the reader's dump or the p4/p5 log is not on the dev box"; fi
  for t in IPo IPi IPid1 IPid3 IPs IPc IPl IPk; do
    r=${t}x_$B
    [ -f "$PULL/$r/PULLED.txt" ] || Q=imagplan bash spatial/q3/pull_spatial_t1.sh "$r" 2>&1 | tail -n 1 | cut -c1-160
  done
done
n() { echo "${1}x_$A"; }
PAIRS="$(n IPl):$(n IPo),$(n IPi):$(n IPo),$(n IPi):$(n IPc),$(n IPi):$(n IPl),$(n IPc):$(n IPo),$(n IPs):$(n IPi),$(n IPk):$(n IPi),$(n IPk):$(n IPo),$(n IPk):$(n IPc)"
( cd $H/headway && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds $B1,$B2 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$RW/hw" > "$R/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd $H/spatial/q3 && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds $B1,$B2 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$RW/q3" > "$R/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" imagplan/read_i9.py --arm $A --sfx x --bases $B1,$B2 --dir imagplan --read-dir "imagplan/read_i9/$A" --seq-outs imagplan/deploy_k_${tg}s${B1}i3.out imagplan/deploy_k_${tg}s${B2}i3.out imagplan/cmp_subset_i_${A}_s$B1.out imagplan/cmp_subset_i_${A}_s$B2.out --out "imagplan/read_i9/$A/READ_I9_$A.json" > "$R/read_i9.log" 2>&1; echo "[$(ts)] read_i9 rc $?"
tail -n 30 "$R/read_i9.log" | cut -c1-330
echo "ZZ READ-I9 $A DONE $(ts)"
