#!/usr/bin/env bash
# MOTION lead read of one arm on the CPU (card ADD18 + note N1; reads I and V, the one-step files for S, the three rollouts of ADD17), without the pixel job.
#   bash read_m_arm.sh <ARM> <cell|flat> <OOF npz of the comparison arm> <name of the comparison arm> [max_minutes]
#   e.g. read_m_arm.sh PR_s0 cell C:/Users/Admin/v8proof_c/ctxroll/rb/OOF_RB_P_s0.npz P_s0
#   Steps as read_p_arm.sh; read folder C:/Users/Admin/v8proof_c/latvis_m (tree with the MOTION files). Serial, CPU, deletes nothing.
ARM=$1; KIND=$2; GO=$3; BASE=$4; MAXMIN=${5:-600}
W=/f/TanitAD-scratch/v8proof; PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
C=C:/Users/Admin/v8proof_c; LP=$C/latvis_m; LG=$C/latvis_gw; OUT=$C/ctxroll; ST=$C/s1states
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8
mkdir -p "$OUT/rb" "$W/spatial/q2/ctx" "$W/spatial/q2/out_m"
N=0
while [ ! -f "$C/ckpt/$ARM/ckpt_last.sha256" ] || [ ! -f "$ST/${ARM}_eval.npz" ]; do
  N=$((N+1)); [ "$N" -gt "$MAXMIN" ] && { echo "ZZ READP STOP $ARM wait-timeout"; exit 1; }
  ping -n 61 127.0.0.1 > /dev/null
done
cd "$LP" || exit 1
if [ ! -f "$LP/latents/$ARM.pt" ]; then
  "$PY" collect_m.py "$ARM" "$C/ckpt/$ARM/ckpt_last.pt" "$KIND" > "$LP/collect_$ARM.log" 2>&1 || { echo "ZZ READP STOP $ARM collect"; tail -n 3 "$LP/collect_$ARM.log"; exit 1; }
fi
cd "$W/spatial/q2" || exit 1
"$PY" ctx_roll.py --arm "$ARM" --ckpt "$C/ckpt/$ARM/ckpt_last.pt" --tree "$LP/tree" --states-pt "$LP/latents/$ARM.pt" --map-latents "$LG/LATENTS_GW_G_s0.npz" --map-states-pt "$LG/latents/G_s0.pt" --emit-frame0 --out-dir "$OUT" > "$OUT/ctx_roll_$ARM.log" 2>&1 || { echo "ZZ READP STOP $ARM ctx_roll"; tail -n 3 "$OUT/ctx_roll_$ARM.log"; exit 1; }
"$PY" rb_trained.py run --arm "$ARM" --states "$ST/${ARM}_eval.npz" --latents "$OUT/LATENTS_F0_$ARM.npz" --out-dir "$OUT/rb" --expect-dim 640 > "$OUT/rb/run_$ARM.log" 2>&1 || { echo "ZZ READP STOP $ARM rb frame0"; tail -n 3 "$OUT/rb/run_$ARM.log"; exit 1; }
"$PY" rb_trained.py pair --a "$OUT/rb/OOF_RB_$ARM.npz" --b "$GO" --out "$OUT/rb/PAIR_RB_${ARM}_minus_$BASE.md" > /dev/null 2>&1 || { echo "ZZ READP STOP $ARM pair base"; exit 1; }
"$PY" rb_change.py run --oof "$OUT/rb/OOF_RB_$ARM.npz" --oof "$GO" --pair "$ARM:$BASE" --out "$OUT/rb/CHANGE_RB_${ARM}_$BASE.md" > /dev/null 2>&1 || { echo "ZZ READP STOP $ARM change form"; exit 1; }
cp "$OUT/rb/RESULT_RB_$ARM.json" "$OUT/rb/VERDICT_RB_$ARM.md" "$OUT/rb/PAIR_RB_${ARM}_minus_$BASE.md" "$OUT/rb/CHANGE_RB_${ARM}_$BASE.md" "$OUT/rb/CHANGE_RB_${ARM}_$BASE.json" "$W/spatial/q2/out_m/"
echo "ZZ READP C1 DONE $ARM $(date -u +%H:%M:%SZ)"
CK_ROOT=$C/ckpt LAT_DIR=$LP ST_DIR=$ST OUT=$OUT TREE=$LP/tree bash "$W/spatial/q2/run_ctxroll.sh" "$ARM" || { echo "ZZ READP STOP $ARM ctx variants"; exit 1; }
echo "ZZ READP ALL DONE $ARM $(date -u +%H:%M:%SZ)"
