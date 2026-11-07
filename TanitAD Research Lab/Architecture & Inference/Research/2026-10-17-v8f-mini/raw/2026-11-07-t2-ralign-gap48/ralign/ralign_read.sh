#!/bin/bash
# The C0C-RA stage-5 read (PREREG_T2_RALIGN.md section 5), in the order the BPC read ran (stage5/BPC/bpc_read.sh): pull check, R-SCN exclusions WITH the cell logs, the pinned read (pin 4) of C0C-RA,
# then the pair read C0C-RA - C0C. NOT RUN by its author (RALIGN prep agent); the Master Mind runs it after `stage5_chain.sh C0C-RA` has pulled.
# usage: bash ralign_read.sh [RUN]            RUN defaults to C0C-RA; GL = a green full-suite pytest log newer than tests/test_t2v8f_gates.py and test_t2v8f_e5.py (set it)
set -u
RUN="${1:-C0C-RA}"
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
P=C:/Users/Admin/t2v8f_pulls/stage5_$RUN/pull5_$RUN; O=$T/stage5/$RUN; GL="${GL:-$T/logs/pytest_full_Q.log}"
OV="$O/t2v8f_stage5_overlay_$RUN.tar::pkg/MANIFEST5.json"
[ -f "$P/STAGE5_RESULT.json" ] || { echo "ZZ RA-READ-NO-RESULT $P ZZ"; exit 2; }
[ -f "$GL" ] || { echo "ZZ RA-READ-NO-GATE-LOG $GL (set GL to a green full-suite log) ZZ"; exit 2; }
export CUDA_VISIBLE_DEVICES=-1
echo "local files=$(find "$P" -type f | wc -l)"
[ -e "$O/EXCLUSIONS5.json" ] && { echo "ZZ EXCLUSIONS5 EXISTS, not overwritten ZZ"; exit 3; }
$PY -I $T/code/scene_exclusions.py --result "$P/STAGE5_RESULT.json" --pull "$P" --log deciding="$P/rollouts_deciding.log" --log beside="$P/rollouts_beside.log" --sources $T/stage4/SCENE_SOURCES_2p3.json --out "$O/EXCLUSIONS5.json" > "$O/exclusions5.out" 2>&1 || { echo "ZZ RA-READ-EXCL-FAIL ZZ"; tail -5 "$O/exclusions5.out"; exit 4; }
tail -3 "$O/exclusions5.out"
# the pinned read (pin 4): arm C0C-RA at stage 5 (the arm name carries a hyphen; the reader's ARM_NAME_RE allows it and appends _s0 / _s1 itself)
bash $T/audit/read_record_args.sh $T/code/read_t2_dominance.py "$O/score_dom" --arm "$RUN" --arm-stage 5 --arm-root "$P/rollouts" --arm-exclusions "$O/EXCLUSIONS5.json" --arm-overlay "$OV" --arm-gate-log "$GL" > "$O/score_dom.out" 2>&1 || { echo "ZZ RA-READ-DOM-FAIL ZZ"; tail -5 "$O/score_dom.out"; exit 5; }
tail -1 "$O/score_dom.out"
# the pair read C0C-RA - C0C (read_t2_pair.py via score_pair/run_pair.sh; its identity check is the Master Mind's, run before this)
bash $T/score_pair/run_pair.sh "$O/pair_${RUN}_C0C" --a-arm "$RUN" --a-stage 5 --a-root "$P/rollouts" --a-exclusions "$O/EXCLUSIONS5.json" --a-overlay "$OV" --a-gate-log "$GL" --b-arm C0C > "$O/pair_${RUN}_C0C.out" 2>&1 || { echo "ZZ RA-READ-PAIR-FAIL ZZ"; tail -5 "$O/pair_${RUN}_C0C.out"; exit 6; }
echo "ZZ RA-READ-DONE ZZ"
