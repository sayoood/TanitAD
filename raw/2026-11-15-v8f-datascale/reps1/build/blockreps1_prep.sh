#!/bin/bash
# blockreps1_prep.sh -- DSCALE10K REPS1: everything the stage-A gate tokens of UE_s1 and UM_s1 need (pod). FORK of block3/block3_prep.sh (UE arm) and blockum/blockum_prep.sh (UM arm). Idempotent: a step whose
# output exists is kept. Runs INSIDE blockreps1.sh (under /root/gpu_queue.lock; GPU steps = two reach probes about 15 s each and the UM trainer's own --selfcheck-only, a few minutes) or by hand under the same lock.
# `cd $D` is the FIRST command. Steps (UE_s1 on $D/tree = cdbff971, UM_s1 on $D/tree_um = 648d4ca3, as their seed-0 twins):
#   0 gen_argv_reps1.py                     argv/{UE_s1,UM_s1}.A.json + blockreps1/PAIRING_PROOF.json   (= the s0 argvs with --seed 1 and --out only; refuses otherwise)
#   1 dry UE_s1 (CPU)                       dry/UE_s1.A.dry.json     (join_union.npz sha256 before / after EQUAL)
#   2 init_reps1.py ue (CPU)                gate/init_ue_s1.json
#   3 reach probe UE_s1 (GPU)               gate/probe2/UE_s1.A.json (gate/reach_probe2.py, unchanged)
#   4 assemble_evidence_reps1_ue.py         gate/evidence_UE_s1.json
#   5 chk_arm_reps1.sh mint UE_s1 A, verify_token_reps1.py UE_s1 A
#   6 dry UM_s1 (CPU)                       dry/UM_s1.A.dry.json     (join_union.npz sha256 before / after EQUAL)
#   7 init_reps1.py um (CPU)                gate/init_um_s1.json
#   8 reach_probe_um_s1.py (GPU)            gate/probe2/UM_s1.A.json (tenk_ref = probe2/UE_s1.A.json)
#   9 trainer --selfcheck-only (GPU)        runs/UM_s1/selfcheck_pre/ (tac_src_probe on the BUILT model, real union data)
#  10 assemble_evidence_reps1_um.py         gate/evidence_UM_s1.json
#  11 chk_arm_reps1.sh mint UM_s1 A, verify_token_reps1.py UM_s1 A
# gate records reused unchanged (INHERITED from seed 0, same trees): init.json, init_s1.json, clock, eval, ckptproc, resume, resume_um, suite, suite_um, cachecheck, stageb, sess.
# Ends with `ZZ REPS1-PREP DONE ZZ` or `ZZ REPS1-PREP FAIL ZZ <why>` (last line of the output).
D=${DS_ROOT:-/root/dscale10k}
cd $D || { echo "ZZ REPS1-PREP FAIL ZZ cd $D"; exit 1; }
set -u
G=$D/gate; PY=${DS_PY:-/root/venv/bin/python}; RP=${DS_RP:-$D/blockreps1}; TREE_U=$D/tree; TREE_M=$D/tree_um
fail() { echo "ZZ REPS1-PREP FAIL ZZ $1"; exit 1; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1 PYTHONSAFEPATH=1
pp() { echo "$1/stack:$1/stack/scripts:$1:$RP/study:$D/study"; }
gpu_idle() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt 3500 ]; }
wait_idle() { n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle before $1"; sleep 10; done; }
for f in gen_argv_reps1.py init_reps1.py reach_probe_um_s1.py assemble_evidence_reps1_ue.py assemble_evidence_reps1_um.py chk_arm_reps1.sh verify_token_reps1.py um_checks_reps1.py gate_items_um.py study/dscale10k_check.py; do [ -s $RP/$f ] || fail "missing $RP/$f"; done
for f in gate/suite.json gate/suite_um.json gate/init.json gate/init_s1.json gate/resume_um.json gate/reach_probe2.py dry/U_s0.A.dry.json dry/B_s0.A.dry.json mix/bset_clips.txt; do [ -s $D/$f ] || fail "missing $D/$f"; done
[ -s $G/tokens/UE_s1.A/gate_token.json ] && [ -s $G/tokens/UM_s1.A/gate_token.json ] && { echo "tokens of UE_s1 and UM_s1 stage A exist (kept)"; echo "ZZ REPS1-PREP DONE ZZ"; exit 0; }
$PY $RP/gen_argv_reps1.py > $G/gen_argv_reps1.log 2>&1 || fail "gen_argv_reps1: $(tail -c 300 $G/gen_argv_reps1.log)"
dry() {   # dry ARM TREE
  if [ -s dry/$1.A.dry.json ]; then echo "$(date -u +%T) dry $1 exists (kept)"; return 0; fi
  echo "$(date -u +%T) dry $1"
  h0=$(sha256sum merged/join_union.npz | cut -d' ' -f1)
  PYTHONPATH=$(pp $2) CUDA_VISIBLE_DEVICES= $PY $RP/study/dscale10k_check.py dry --argv argv/$1.A.json --root $2 --out dry/$1.A.dry.json > dry/$1.A.dry.log 2>&1 || fail "dry $1: $(tail -c 300 dry/$1.A.dry.log)"
  h1=$(sha256sum merged/join_union.npz | cut -d' ' -f1)
  [ "$h0" = "$h1" ] || fail "merged/join_union.npz changed under the $1 dry build (--fit-join): $h0 -> $h1"
}
# ---- UE_s1
if [ ! -s $G/tokens/UE_s1.A/gate_token.json ]; then
  dry UE_s1 $TREE_U
  if [ ! -s $G/init_ue_s1.json ]; then PYTHONPATH=$(pp $TREE_U) CUDA_VISIBLE_DEVICES= $PY $RP/init_reps1.py ue > $G/init_ue_s1.log 2>&1 || fail "init_reps1 ue: $(tail -c 400 $G/init_ue_s1.log)"; fi
  echo "$(date -u +%T) init_ue_s1: $(tail -n 1 $G/init_ue_s1.log | cut -c1-200)"
  mkdir -p $G/probe2
  if [ ! -s $G/probe2/UE_s1.A.json ]; then
    wait_idle "the UE_s1 reach probe"
    PYTHONPATH=$(pp $TREE_U) $PY $G/reach_probe2.py $D/argv/UE_s1.A.json $G/probe2/UE_s1.A.json $D/merged/v9_labels_union.manifest.json --reach > $G/probe2/UE_s1.A.log 2>&1 || fail "reach probe UE_s1: $(tail -c 400 $G/probe2/UE_s1.A.log)"
  fi
  echo "$(date -u +%T) reach probe UE_s1: $(tail -n 1 $G/probe2/UE_s1.A.log | cut -c1-200)"
  PYTHONPATH=$(pp $TREE_U) $PY $RP/assemble_evidence_reps1_ue.py UE_s1 > $G/assemble_ue_s1.log 2>&1 || fail "assemble_evidence_reps1_ue: $(tail -c 400 $G/assemble_ue_s1.log)"
  echo "$(date -u +%T) evidence UE_s1 assembled"
  bash $RP/chk_arm_reps1.sh mint UE_s1 A > $G/mint_UE_s1.A.log 2>&1 || fail "mint UE_s1 A refused: $(tail -c 700 $G/mint_UE_s1.A.log | tr '\n' ' ')"
  echo "$(date -u +%T) $(tail -n 1 $G/mint_UE_s1.A.log | cut -c1-200)"
  PYTHONPATH=$(pp $TREE_U) $PY $RP/verify_token_reps1.py UE_s1 A > $G/verify_UE_s1.A.log 2>&1 || fail "verify_token UE_s1 A: $(tail -c 300 $G/verify_UE_s1.A.log)"
fi
# ---- UM_s1
if [ ! -s $G/tokens/UM_s1.A/gate_token.json ]; then
  dry UM_s1 $TREE_M
  if [ ! -s $G/init_um_s1.json ]; then PYTHONPATH=$(pp $TREE_M) CUDA_VISIBLE_DEVICES= $PY $RP/init_reps1.py um > $G/init_um_s1.log 2>&1 || fail "init_reps1 um: $(tail -c 400 $G/init_um_s1.log)"; fi
  echo "$(date -u +%T) init_um_s1: $(tail -n 1 $G/init_um_s1.log | cut -c1-200)"
  mkdir -p $G/probe2 $D/runs/UM_s1
  if [ ! -s $G/probe2/UM_s1.A.json ]; then
    wait_idle "the UM_s1 reach probe"
    PYTHONPATH=$(pp $TREE_M) $PY $RP/reach_probe_um_s1.py $D/argv/UM_s1.A.json $G/probe2/UM_s1.A.json $D/merged/v9_labels_union.manifest.json --reach > $G/probe2/UM_s1.A.log 2>&1 || fail "reach probe UM_s1: $(tail -c 400 $G/probe2/UM_s1.A.log)"
  fi
  echo "$(date -u +%T) reach probe UM_s1: $(tail -n 1 $G/probe2/UM_s1.A.log | cut -c1-240)"
  if [ ! -s $D/runs/UM_s1/selfcheck_pre/selfcheck.json ]; then
    wait_idle "the UM_s1 selfcheck"
    $PY - $D/argv/UM_s1.A.json $D/runs/UM_s1/selfcheck_pre_argv.json $D/runs/UM_s1/selfcheck_pre <<'E' || fail "selfcheck argv"
import json, sys
t = [x for x in json.load(open(sys.argv[1]))["argv"] if x != "--fit-join"]      # the join file exists (the dry build wrote it); this run must not rewrite it
t[t.index("--out") + 1] = sys.argv[3]
json.dump({"argv": t + ["--selfcheck-only"]}, open(sys.argv[2], "w"))
E
    ( cd $TREE_M && PYTHONPATH=$(pp $TREE_M) $PY $TREE_M/stack/scripts/v8f_mini_train.py $($PY -c "import json;print(' '.join(json.load(open('$D/runs/UM_s1/selfcheck_pre_argv.json'))['argv']))") > $D/runs/UM_s1/selfcheck_pre.log 2>&1 )
    [ -s $D/runs/UM_s1/selfcheck_pre/selfcheck.json ] || fail "selfcheck UM_s1 wrote no selfcheck.json: $(tail -c 400 $D/runs/UM_s1/selfcheck_pre.log | tr '\n' ' ')"
  fi
  $PY - $D/runs/UM_s1/selfcheck_pre/selfcheck.json <<'E' || fail "the pre-launch selfcheck has no tac_src_probe"
import json, sys
sc = json.load(open(sys.argv[1]))
assert sc.get("tac_src_probe") and sc.get("tac_src"), list(sc)
print("selfcheck_pre tac_src_probe", json.dumps(sc["tac_src_probe"])[:300])
E
  PYTHONPATH=$(pp $TREE_M) $PY $RP/assemble_evidence_reps1_um.py UM_s1 > $G/assemble_um_s1.log 2>&1 || fail "assemble_evidence_reps1_um: $(tail -c 400 $G/assemble_um_s1.log)"
  echo "$(date -u +%T) evidence UM_s1 assembled"
  bash $RP/chk_arm_reps1.sh mint UM_s1 A > $G/mint_UM_s1.A.log 2>&1 || fail "mint UM_s1 A refused: $(tail -c 700 $G/mint_UM_s1.A.log | tr '\n' ' ')"
  echo "$(date -u +%T) $(tail -n 1 $G/mint_UM_s1.A.log | cut -c1-200)"
  PYTHONPATH=$(pp $TREE_M) $PY $RP/verify_token_reps1.py UM_s1 A > $G/verify_UM_s1.A.log 2>&1 || fail "verify_token UM_s1 A: $(tail -c 300 $G/verify_UM_s1.A.log)"
fi
echo "ZZ REPS1-PREP DONE ZZ"
