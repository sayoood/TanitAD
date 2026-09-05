#!/bin/bash
# D-REFAV1-CCOS-EVAL — the refav1 open-loop re-evaluation chain on Thor (tanitad-thor-wifi).
#
# SAME windows as the banked `cos` read (`/home/nvidia/refav1_evalrun/full_dump`: v7.2 EVAL split,
# 141 clips, --window-stride 40 -> 282 windows / 141 episode clusters, K = 10 @ 0.2 s, --no-navshuf,
# --action-units kappa = the legacy contract every banked refav1 number was produced under), SAME
# checkpoint (`refav1-b1-v72-ep3-speed/ckpt.pt`, step 21,109 — NOT `…-1ep-21109`, whose log stops at
# step 1,000), SAME shipped iCEM config (seed 0). Only the goal term's FORM and the declared weight
# triple change, per arm:
#   smoke   ccos @ shipped weights, 1 episode      -> proves the flag reaches plan() (verify-gate) before 2.8 h is spent
#   arm1    ccos @ shipped weights (NAIVE flip)     -> the form change PLUS its implicit re-weight
#   arm2    ccos @ compensated weights              -> the form change with the goal:penalty balance held where cos left it
#           (the triple is read from arm2_weights.txt WHEN ARM 2 STARTS, so it can be set from the
#            282-window probe while arm 1 runs; if the file is absent the arm is SKIPPED, loudly)
#   arm3    chord @ shipped weights                 -> the deliberate-regression form on the full grid
#                                                     (monotone-equivalent to cos: MUST stay flat)
# Each arm banks per-episode npz files as it goes, so a killed chain still yields a partial dump that
# `--analyze-only` can read with zero GPU.
R=/home/nvidia/refav1_ccos
cd $R/repo || exit 1
export PYTHONPATH=$R/repo/stack:$R/repo/taniteval OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8
PY=/home/nvidia/venvs/tanitad-train/bin/python
CK=/home/nvidia/experiments/refav1-b1-v72-ep3-speed
COMMON="--ckpt $CK/ckpt.pt --config $CK/config.json --cache /home/nvidia/data/refav1-fp8-eval \
 --episodes /home/nvidia/data/physicalai-b1-w120-256x640cyl \
 --labels /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz \
 --nav /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz \
 --lead-block $R/b1_eval_lead_block.npz --device cuda --window-stride 40 --no-navshuf"

stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
# ⛔ verify-gate by CONTENT before anything is launched (a stale tree would score `cos` under the name `ccos`)
n=$(grep -c '"ccos"' $R/repo/stack/tanitad/refs/refa_v1.py); [ "$n" -ge 3 ] || { stamp "STALE STACK: ccos hits=$n"; exit 9; }
grep -q "cost_weights" $R/repo/taniteval/tools/refav1_arm.py || { stamp "STALE ARM"; exit 9; }
grep -q '"ha0_ext": "T1"' $R/repo/taniteval/tools/t1_eval.py || { stamp "STALE t1_eval (ha0_ext tier)"; exit 9; }

run_arm() {  # name metric [weights]
  local name=$1 metric=$2 w=$3 extra=""
  [ -n "$w" ] && extra="--cost-weights $w"
  stamp "ARM $name start metric=$metric weights=${w:-shipped}"
  $PY taniteval/tools/refav1_arm.py $COMMON --episodes-n ${EPN:-0} --cost-metric $metric $extra \
     --dump-dir $R/dump_$name --out $R/rec_$name.json --arm refav1-21109-$name > $R/$name.log 2>&1
  local rc=$?
  stamp "ARM $name exit=$rc"
  echo "$rc" > $R/$name.EXIT
  return $rc
}

rm -f $R/chain.DONE
stamp "chain start (pid $$)"
EPN=1 run_arm smoke ccos "" || { stamp "SMOKE FAILED — refusing to spend 2.8 h"; exit 2; }
grep -q "cost\] metric=ccos" $R/smoke.log || { stamp "smoke log lacks the cost banner"; exit 2; }
EPN=0 run_arm ccos_naive ccos ""
if [ -f $R/arm2_weights.txt ]; then
  W=$(tr -d ' \r\n' < $R/arm2_weights.txt)
  EPN=0 run_arm ccos_comp ccos "$W"
else
  stamp "ARM ccos_comp SKIPPED: $R/arm2_weights.txt absent"
fi
EPN=0 run_arm chord_shipped chord ""
stamp "chain end"
touch $R/chain.DONE
