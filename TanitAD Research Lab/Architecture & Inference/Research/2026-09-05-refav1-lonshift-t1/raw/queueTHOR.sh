#!/usr/bin/env bash
# refav1 LONGITUDINAL arms at T1, on Thor.
#
# WHY A THOR-LOCAL BASELINE: the ckpt/labels/episodes/code are md5-identical to
# the dev-box rig, but CEM on a different GPU is not bit-reproducible, so a
# cross-rig pairing is INADMISSIBLE. `T_wk15` is arm 1 and every lever is paired
# against IT, window-for-window, inside one rig. Thor's T_wk15 vs the dev box's
# wk15 is then a RIG control, never a lever result.
#
# CONCURRENCY GATE: `jobs -p` counts THIS script's own children. It never greps
# the process table, so it cannot self-match the way `pgrep -f` does (and the
# artifact being counted is the ARM, not a python process -- one arm is several).
set -u
R=/home/nvidia/refav1_lon
OUT=$R/out
PY=/home/nvidia/venvs/tanitad-edge/bin/python
export PYTHONPATH=$R/code/stack:$R/code/taniteval
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
LBL=/home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz
MAXJ=${MAXJ:-3}

common=(--ckpt $R/ckpt/ckpt.pt
        --config $R/ckpt/config.json
        --cache $R/p4/fp8 --episodes $R/p4/eps --labels "$LBL" --nav "$LBL"
        --device cuda --episodes-n 0 --window-stride 16 --no-navshuf
        --no-lead-block --cost-metric ccos
        --cost-weights 0.0,15.11245,64.29715042415070)

run () {
  local tag="$1"; shift
  rm -rf "$OUT/dump_$tag"
  echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ" >> "$OUT/queue.log"
  ( "$PY" "$R/code/taniteval/tools/refav1_arm.py" "${common[@]}" "$@" \
      --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
      --arm "refav1-21109-thor-${tag}" >> "$OUT/${tag}.log" 2>&1
    echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ" >> "$OUT/queue.log" ) &
  while [ "$(jobs -p | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
}

# PRIORITY ORDER -- a killed queue still yields value.
run T_wk15                                                  # 1 in-rig baseline (REQUIRED by all pairs)
run T_lonshift     --plan-seed 0 --a-sustain-mode a0_shift  # 2 D2, the headline
run T_lonshift_s1  --plan-seed 1 --a-sustain-mode a0_shift  # 3 the seed floor ON the lever arm (MANDATORY)
run T_lonvocab     --plan-seed 0 --a-sustain-mode a0        # 4 D1, for ATTRIBUTION
run T_wk15_s1      --plan-seed 1                            # 5 the seed floor on the BASELINE
run T_lonseam      --plan-seed 0 --jerk-seam a0             # 6 P4 successor: the cost lever alone
run T_loncomb      --plan-seed 0 --a-sustain-mode a0_shift --jerk-seam a0   # 7 D2 + seam
wait
echo "ZZQUEUETHOR-DONE-$(date -u +%FT%TZ)ZZ" >> "$OUT/queue.log"
