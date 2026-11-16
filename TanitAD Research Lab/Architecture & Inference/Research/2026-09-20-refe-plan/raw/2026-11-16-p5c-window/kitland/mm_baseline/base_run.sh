#!/bin/bash
# MM 2026-11-16: BASELINE for the 8 kit-tree failures. t_base = the kit tree with the kit's 22 refe files reverted to tip 1c48ee8f
# (13 modified files restored from the tip, 9 new files removed). Same runner settings as one.sh. Writes base_run/summary.txt, ends ZZKLBASE_END.
R=/workspace/data/kitland
PKN="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
set -u
rm -rf "/workspace/data/kitland/t_base"
cp -a "$R/t" "$R/t_base" || { echo ZZKLBASE_FAIL cp; exit 1; }
cd "$R/t_base/$PKN/refe" || exit 99
rm -rf "/tmp/refe_tip"; tar xf "$R/refe_tip.tar" --no-same-owner -C /tmp/ || { echo ZZKLBASE_FAIL tar; exit 1; }
cp /tmp/refe_tip/* . || { echo ZZKLBASE_FAIL cp_tip; exit 1; }
rm -rf "/tmp/refe_tip"
cmp -s launch_gate.py "$R/t/$PKN/refe/launch_gate.py" && { echo ZZKLBASE_FAIL launch_gate_not_reverted; exit 1; }
for f in traj_label_table.py test_traj_label_table.py test_p5c_profile.py p5c_gate.py p5c_postgate.py p5c_pod.sh test_p5c_live.py test_p5c_postgate.py test_p5c_chain.py; do rm -f "./$f"; done
OUT="$R/base_run"; mkdir -p "$OUT"; : > "$OUT/summary.txt"
grep -c "P5C_BASE_SHA256" launch_gate.py >> "$OUT/summary.txt"
for S in test_hnscreen_chain.py test_p2r_chain.py test_rls1_chain.py test_s1aux_chain.py test_s1z_preflight.py test_s1zsm_preflight.py test_sft4r_chain.py test_vdac_stage0_forward.py; do
  TT="$R/tmp/base_$S"; mkdir -p "$TT"
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONPATH="$R/t_base/stack" TMP="$TT" TEMP="$TT" TMPDIR="$TT" \
    timeout 1500 nice -n 10 /workspace/venv-teacher/bin/python "$S" > "$OUT/$S.log" 2>&1 < /dev/null
  echo "$S rc=$? last=$(grep -a -E 'FAIL|REFUS|Error|PASS' "$OUT/$S.log" | tail -1 | cut -c1-160)" >> "$OUT/summary.txt"
  rm -rf "${TT:?}"
done
echo ZZKLBASE_END >> "$OUT/summary.txt"
