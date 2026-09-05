#!/usr/bin/env bash
# refav1 A3 -- the GOAL-CONDITIONED LATERAL COST -- at T1, on Thor.
#
# WHY THOR AND WHY NOW: the dev box is at its measured 2-arm ceiling
# (6737/8188 MiB) and a sibling queue owns the next free slot there.
# `queueTHOR.sh` has exhausted its 7-arm plan and sits in `wait`, so these two
# slots are free and nothing will contend for them.
#
# ⛔ CROSS-RIG PAIRING IS INADMISSIBLE (queueTHOR.sh's own note: CEM on a
# different GPU is not bit-reproducible). These arms are paired against the
# THOR-LOCAL `T_wk15` -- the same ckpt, labels, episodes, window grid and
# `--cost-weights` triple -- which is already banked at
# /home/nvidia/refav1_lon/out/rec_T_wk15.json, together with its seed replicate
# `T_wk15_s1`. So the pair AND its inference-seed floor are both in-rig.
#
# ⭐ ONE VARIABLE against T_wk15: `--w-kappa-by-goal`. The scalar W_KAPPA is
# unchanged at 15.11245 and, because goal_source is `tactical_imagined` on
# 100 % of these windows, it governs nothing -- every window is priced by the
# map.
set -u
R=/home/nvidia/refav1_ctg
LON=/home/nvidia/refav1_lon
OUT=$R/out
PY=/home/nvidia/venvs/tanitad-edge/bin/python
export PYTHONPATH=$R/code/stack:$R/code/taniteval
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
LBL=/home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz
MAXJ=${MAXJ:-2}

common=(--ckpt $LON/ckpt/ckpt.pt
        --config $LON/ckpt/config.json
        --cache $LON/p4/fp8 --episodes $LON/p4/eps --labels "$LBL" --nav "$LBL"
        --device cuda --episodes-n 0 --window-stride 16 --no-navshuf
        --no-lead-block --cost-metric ccos
        --cost-weights 0.0,15.11245,64.29715042415070)

run () {
  local tag="$1"; shift
  rm -rf "$OUT/dump_$tag"
  echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ" >> "$OUT/queue.log"
  ( "$PY" "$R/code/taniteval/tools/refav1_arm.py" "${common[@]}" "$@" \
      --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
      --arm "refav1-21109-thorctg-${tag}" >> "$OUT/${tag}.log" 2>&1
    echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ" >> "$OUT/queue.log" ) &
  while [ "$(jobs -p | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
}

# PRIORITY ORDER -- a killed queue still yields the arm that answers the PI.
# 1 THE ARM: charge curvature on LANE_KEEP goals, free it on TURN goals.
run G_gkappa      --plan-seed 0 --w-kappa-by-goal 15.11245,0.0
# 2 THE DELIBERATE-REGRESSION ARM: the INVERTED cost. Must be WORSE on
#   turn_left recall AND on ADE, or the conditioning is inert and (1) is noise.
run G_gkappa_inv  --plan-seed 0 --w-kappa-by-goal 0.0,15.11245
wait
echo "ZZQUEUECTG-DONE-$(date -u +%FT%TZ)ZZ" >> "$OUT/queue.log"
