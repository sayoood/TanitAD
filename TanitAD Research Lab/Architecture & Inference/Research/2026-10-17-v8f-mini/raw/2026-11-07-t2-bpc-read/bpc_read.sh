#!/bin/bash
# The BPC stage-5 read (PREREG_T2_ATTRIB), in order: pull check, R-SCN exclusions WITH the cell logs, the pinned read (pin 4) of BPC, the pair BPC - MC, the RT verdict, and C0C - BPC beside.
set -u
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f; P=C:/Users/Admin/t2v8f_pulls/stage5_BPC/pull5_BPC; O=$T/stage5/BPC; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PM=C:/Users/Admin/t2v8f_pulls/stage5_MC/pull5_MC; GL=$T/logs/pytest_full_Q.log; OV="$O/t2v8f_stage5_overlay_BPC.tar::pkg/MANIFEST5.json"; OVM="$T/stage5/MC/t2v8f_stage5_overlay_MC.tar::pkg/MANIFEST5.json"
[ -f $P/STAGE5_RESULT.json ] || { echo "ZZ BPCREAD-NO-RESULT $P ZZ"; exit 2; }
echo "local files=$(find $P -type f | wc -l) bytes=$(find $P -type f -printf '%s\n' | awk '{s+=$1} END{print s}')"
export CUDA_VISIBLE_DEVICES=-1
[ -e $O/EXCLUSIONS5.json ] && { echo "ZZ EXCLUSIONS5 EXISTS, not overwritten ZZ"; exit 3; }
$PY -I $T/code/scene_exclusions.py --result $P/STAGE5_RESULT.json --pull $P --log deciding=$P/rollouts_deciding.log --log beside=$P/rollouts_beside.log --sources $T/stage4/SCENE_SOURCES_2p3.json --out $O/EXCLUSIONS5.json > $O/exclusions5.out 2>&1 || { echo "ZZ BPCREAD-EXCL-FAIL ZZ"; tail -5 $O/exclusions5.out; exit 4; }
tail -3 $O/exclusions5.out
bash $T/audit/read_record_args.sh $T/code/read_t2_dominance.py $O/score_dom --arm BPC --arm-stage 5 --arm-root $P/rollouts --arm-exclusions $O/EXCLUSIONS5.json --arm-overlay "$OV" --arm-gate-log $GL > $O/score_dom.out 2>&1 || { echo "ZZ BPCREAD-DOM-FAIL ZZ"; tail -5 $O/score_dom.out; exit 5; }
tail -1 $O/score_dom.out
A=(--a-arm BPC --a-stage 5 --a-root $P/rollouts --a-exclusions $O/EXCLUSIONS5.json --a-overlay "$OV" --a-gate-log $GL)
B=(--b-arm MC --b-stage 5 --b-root $PM/rollouts --b-exclusions $T/stage5/MC/EXCLUSIONS5.json --b-overlay "$OVM" --b-gate-log $GL)
bash $T/score_pair/run_pair.sh $O/pair_BPC_MC "${A[@]}" "${B[@]}" > $O/pair_BPC_MC.out 2>&1 || { echo "ZZ BPCREAD-PAIR-FAIL ZZ"; tail -5 $O/pair_BPC_MC.out; exit 6; }
$PY -I $O/rt_verdict.py $O/pair_BPC_MC/PAIR_T2.json $T/stage5/MC/score_dom/DOMINANCE_T2.json $O/RT_VERDICT.json
bash $T/score_pair/run_pair.sh $O/pair_C0C_BPC --a-arm C0C --b-arm BPC --b-stage 5 --b-root $P/rollouts --b-exclusions $O/EXCLUSIONS5.json --b-overlay "$OV" --b-gate-log $GL > $O/pair_C0C_BPC.out 2>&1 || echo "ZZ beside pair C0C-BPC failed (beside only) ZZ"
echo "ZZ BPCREAD-DONE ZZ"
