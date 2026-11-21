#!/bin/bash
# The read of record's own inputs (motion_c/t2v8f/audit/read_record_args.sh, STAGE4_LAUNCH.md section 2 + the three exclusions flags of HANDOFF_P), then the pair options you give.
# Every path is spelled E:/ or C:/ (never /e/ or $PWD: a Git Bash $PWD breaks the `tar::member` argument), and the interpreter is the venv Python (the system Python has no numpy).
# usage: run_pair.sh <out-dir> --a-arm NAME [...] --b-arm NAME [...] [--identity-record R]
#   identity:   run_pair.sh $S/logs/identity_run --a-arm C0C --b-arm P10 --identity-record $T/stage4/score/DOMINANCE_T2.json
#   MC - C0C:   (first EXCLUSIONS5 per HANDOFF_S5.md: python -I $T/code/scene_exclusions.py --result $P/STAGE5_RESULT.json --pull $P --sources $T/stage4/SCENE_SOURCES_2p3.json --out $T/stage5/MC/EXCLUSIONS5.json, P=C:/Users/Admin/t2v8f_pulls/stage5_MC/pull5_MC)
#               run_pair.sh $T/stage5/MC/pair_MC_C0C --a-arm MC --a-stage 5 --a-root $P/rollouts --a-exclusions $T/stage5/MC/EXCLUSIONS5.json #                 --a-overlay "$T/stage5/MC/t2v8f_stage5_overlay_MC.tar::pkg/MANIFEST5.json" --a-gate-log $T/logs/pytest_full_Q.log --b-arm C0C
#   BPC - MC:   run_pair.sh $T/stage5/BPC/pair_BPC_MC --a-arm BPC --a-stage 5 --a-root $PB/rollouts --a-exclusions $T/stage5/BPC/EXCLUSIONS5.json #                 --a-overlay "$T/stage5/BPC/t2v8f_stage5_overlay_BPC.tar::pkg/MANIFEST5.json" --a-gate-log <green log> #                 --b-arm MC --b-stage 5 --b-root $P/rollouts --b-exclusions $T/stage5/MC/EXCLUSIONS5.json #                 --b-overlay "$T/stage5/MC/t2v8f_stage5_overlay_MC.tar::pkg/MANIFEST5.json" --b-gate-log <green log>        (PB=C:/Users/Admin/t2v8f_pulls/stage5_BPC/pull5_BPC)
#   Also read each arm against P10 as the pinned dominance read does (code/read_t2_dominance.py --arm ...), and C0C - MC / MC - BPC are the same calls with the sides swapped.
#   --a-result / --b-result default to STAGE5_RESULT.json beside the rollouts dir. A green gate log must be newer than tests/test_t2v8f_gates.py and tests/test_t2v8f_e5.py (G2 b).
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f
OUT="$1"; shift 1
exec env CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I $T/score_pair/read_t2_pair.py \
  --root2 $T/stage2/pull_box2_run2/pull2/rollouts \
  --root3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/rollouts \
  --root4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/rollouts \
  --sources2 $T/stage2/SCENE_SOURCES.json \
  --sources3 $T/stage3/SCENE_SOURCES3.json \
  --g1 $T/C0C_ID_RESULT.json \
  --result1 $T/stage1/pull_box2/pull/STAGE1_RESULT.json \
  --manifest1 "$T/stage1/t2v8f_stage1.tar::pkg/MANIFEST.json" \
  --result2 $T/stage2/pull_box2_run2/pull2/STAGE2_RESULT.json \
  --result3 C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/STAGE3_RESULT.json \
  --result4 C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/STAGE4_RESULT.json \
  --exclusions2 $T/stage2/EXCLUSIONS2.json \
  --exclusions3 $T/stage3/EXCLUSIONS3.json \
  --exclusions4 $T/stage4/EXCLUSIONS4.json \
  --out-dir "$OUT" "$@"
