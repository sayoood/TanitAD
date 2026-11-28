#!/bin/bash
# The C0C-RE stage-5 read (PREREG_T2_RENC.md sections 5 and 6), in the order of stage5/BPC/bpc_read.sh and ralign/ralign_read.sh:
#   1 pull check  2 R-SCN exclusions WITH the cell logs  3 the pinned read (pin 4) of C0C-RE  4 the pair C0C-RE - C0C  5 beside, if the BLIND pull exists: C0C-RE-BLIND - C0C-RE  6 c0cre_outcome.py (section 6).
# NOT RUN by its author. Run it once renc/c0cre/logs/c0cre_chain.out shows "ZZ STAGE5 PULLED MK_DONE C0C-RE ->" (the deciding pull). It is CPU only and does not touch box 2 or any busy marker.
# usage: bash c0cre_read.sh            env (all optional; the defaults are the real run):
#   GL                  a green full-suite pytest log newer than tests/test_t2v8f_gates.py and test_t2v8f_e5.py (G2 b); default $T/logs/pytest_full_Q.log, the one the BPC read used
#   C0CRE_RUN           run name = pull and package folder (default C0C-RE; the identity rehearsal sets C0C-RE_ID). The arm read is C0C-RE (C0CRE_ARM overrides it for a rehearsal only).
#   C0CRE_PULLS         root of the pulls (default C:/Users/Admin/t2v8f_pulls)
#   C0CRE_OUT           where outputs go (default the package folder $T/stage5/$C0CRE_RUN); a rehearsal points it at a scratch folder
#   C0CRE_BLIND=0       skip the beside pair;  C0CRE_BLIND_ONLY=1  steps 5 and 6 only, on an existing deciding read (see below);  C0CRE_BLIND_OUT  where the BLIND exclusions go (default $T/stage5/C0C-RE-BLIND)
#   C0CRE_DRYRUN=1      print the heavy commands instead of running them (wiring check)
# Refuses (named, nonzero) if the result file is missing (2), if EXCLUSIONS5.json already exists (3: never overwritten), or a step crashes (4 exclusions, 5 pinned read, 6 pair: no JSON written). A failed gate inside a
# pinned read is NOT a refusal here: the reader still writes its JSON with outcome GATE-FAIL (the pair reader exits 2) and c0cre_outcome.py reports NOT-READ. The BLIND beside never blocks the outcome.
set -u
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ARM="${C0CRE_ARM:-C0C-RE}"; BARM=C0C-RE-BLIND; RUN="${C0CRE_RUN:-$ARM}"; BRUN="$BARM"      # C0CRE_ARM is for a rehearsal on another arm's complete pull (BPC); the real read leaves it unset
PULLS="${C0CRE_PULLS:-C:/Users/Admin/t2v8f_pulls}"
P=$PULLS/stage5_$RUN/pull5_$RUN; PK=$T/stage5/$RUN; O="${C0CRE_OUT:-$PK}"
PB=$PULLS/stage5_$BRUN/pull5_$BRUN; PKB=$T/stage5/$BRUN; OB="${C0CRE_BLIND_OUT:-$PKB}"
GL="${GL:-$T/logs/pytest_full_Q.log}"; CHAIN=$T/renc/c0cre/logs/c0cre_chain.out
OV="$PK/t2v8f_stage5_overlay_$RUN.tar::pkg/MANIFEST5.json"; OVB="$PKB/t2v8f_stage5_overlay_$BRUN.tar::pkg/MANIFEST5.json"
C0C_REF=$T/stage4/score_q_identity/DOMINANCE_T2.json         # the pin-4 read of C0C against P10: the reference for "no new separated-worse row" (c0cre_outcome.py cross-checks it against the pair)
x() { if [ "${C0CRE_DRYRUN:-0}" = 1 ]; then echo "DRY: $*"; else "$@"; fi; }
have() { [ "${C0CRE_DRYRUN:-0}" = 1 ] || [ -s "$1" ]; }      # a reader that gates out still WRITES its JSON (outcome GATE-FAIL, exit 2): that is not a refusal; a missing JSON (a crash) is
[ -f "$P/STAGE5_RESULT.json" ] || { echo "ZZ C0CRE-READ-NO-RESULT $P ZZ"; exit 2; }
[ -f "$GL" ] || { echo "ZZ C0CRE-READ-NO-GATE-LOG $GL (set GL to a green full-suite log) ZZ"; exit 2; }
[ -f "$C0C_REF" ] || { echo "ZZ C0CRE-READ-NO-C0C-REFERENCE $C0C_REF ZZ"; exit 2; }
# ---- 1. pull check (counts as bpc_read.sh; the markers and the chain line are information, the pinned gates decide)
echo "local files=$(find "$P" -type f | wc -l) bytes=$(find "$P" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')"
for m in STAGE5.done STAGE5.fail; do [ -e "$P/$m" ] && echo "marker present: $m"; done
[ -e "$P/STAGE5.fail" ] && echo "ZZ WARNING marker STAGE5.fail is in the pull; the pinned gates decide whether any row can be read ZZ"
[ -f "$CHAIN" ] && grep -E "STAGE5 PULLED .* $RUN ->|STAGE5-|C0CRE-" "$CHAIN" | tail -3
export CUDA_VISIBLE_DEVICES=-1
mkdir -p "$O"
BONLY="${C0CRE_BLIND_ONLY:-0}"      # 1 = the deciding read is already in $O: run ONLY steps 5 and 6 (the BLIND pull landed later); the old RESULT_T2_RENC.* are moved aside, never deleted
if [ "$BONLY" = 1 ]; then
  [ -s "$O/EXCLUSIONS5.json" ] && [ -s "$O/score_dom/DOMINANCE_T2.json" ] && [ -s "$O/pair_${ARM}_C0C/PAIR_T2.json" ] || { echo "ZZ C0CRE-BLIND-ONLY needs the deciding read in $O (EXCLUSIONS5.json, score_dom, pair_${ARM}_C0C) ZZ"; exit 2; }
  echo "blind-only: reusing the deciding read in $O"
else
[ -e "$O/EXCLUSIONS5.json" ] && { echo "ZZ EXCLUSIONS5 EXISTS, not overwritten ZZ"; exit 3; }
# the read's own inputs, recorded (the pinned reads also record their argv in their JSON)
{ echo "c0cre_read.sh $(date '+%F %T')"; echo "RUN=$RUN ARM=$ARM P=$P O=$O GL=$GL OV=$OV C0C_REF=$C0C_REF"
  for f in code/read_t2_dominance.py score_pair/read_t2_pair.py code/scene_exclusions.py stage5/C0C-RE/c0cre_outcome.py; do echo "$f sha256 $($PY -I -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$T/$f" | tr -d '\r')"; done
  echo "GL sha256 $($PY -I -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$GL" | tr -d '\r')"; } > "$O/READ_ARGS.txt"
# ---- 2. R-SCN exclusions, WITH the cell logs of both passes
x $PY -I $T/code/scene_exclusions.py --result "$P/STAGE5_RESULT.json" --pull "$P" --log deciding="$P/rollouts_deciding.log" --log beside="$P/rollouts_beside.log" --sources $T/stage4/SCENE_SOURCES_2p3.json --out "$O/EXCLUSIONS5.json" > "$O/exclusions5.out" 2>&1 || { echo "ZZ C0CRE-READ-EXCL-FAIL ZZ"; tail -5 "$O/exclusions5.out"; exit 4; }
tail -3 "$O/exclusions5.out"
# ---- 3. the pinned read (pin 4) of C0C-RE at stage 5 (the arm name carries a hyphen; the reader's ARM_NAME_RE allows it and appends _s0 / _s1 itself)
x bash $T/audit/read_record_args.sh $T/code/read_t2_dominance.py "$O/score_dom" --arm $ARM --arm-stage 5 --arm-root "$P/rollouts" --arm-exclusions "$O/EXCLUSIONS5.json" --arm-overlay "$OV" --arm-gate-log "$GL" > "$O/score_dom.out" 2>&1
have "$O/score_dom/DOMINANCE_T2.json" || { echo "ZZ C0CRE-READ-DOM-FAIL (no DOMINANCE_T2.json) ZZ"; tail -5 "$O/score_dom.out"; exit 5; }
tail -1 "$O/score_dom.out"
# ---- 4. the pair C0C-RE - C0C (C0C is the stage-4 default: no --b-stage, no --b-root; the reader pairs seed against seed over the same starts)
x bash $T/score_pair/run_pair.sh "$O/pair_${ARM}_C0C" --a-arm $ARM --a-stage 5 --a-root "$P/rollouts" --a-exclusions "$O/EXCLUSIONS5.json" --a-overlay "$OV" --a-gate-log "$GL" --b-arm C0C > "$O/pair_${ARM}_C0C.out" 2>&1
have "$O/pair_${ARM}_C0C/PAIR_T2.json" || { echo "ZZ C0CRE-READ-PAIR-FAIL (no PAIR_T2.json) ZZ"; tail -5 "$O/pair_${ARM}_C0C.out"; exit 6; }
tail -1 "$O/pair_${ARM}_C0C.out"
fi
# ---- 5. beside, deciding nothing: C0C-RE-BLIND - C0C-RE (only if the BLIND pull has landed; its own exclusions, never overwritten)
BLINDPAIR=""
if [ "${C0CRE_BLIND:-1}" != 0 ] && [ -f "$PB/STAGE5_RESULT.json" ]; then
  mkdir -p "$OB"
  if [ -e "$OB/EXCLUSIONS5.json" ]; then echo "ZZ beside: $OB/EXCLUSIONS5.json EXISTS, not overwritten; the BLIND beside is skipped ZZ"
  else
    x $PY -I $T/code/scene_exclusions.py --result "$PB/STAGE5_RESULT.json" --pull "$PB" --log deciding="$PB/rollouts_deciding.log" --log beside="$PB/rollouts_beside.log" --sources $T/stage4/SCENE_SOURCES_2p3.json --out "$OB/EXCLUSIONS5.json" > "$OB/exclusions5.out" 2>&1
    if have "$OB/EXCLUSIONS5.json"; then
      x bash $T/score_pair/run_pair.sh "$O/pair_${BARM}_${ARM}" --a-arm $BARM --a-stage 5 --a-root "$PB/rollouts" --a-exclusions "$OB/EXCLUSIONS5.json" --a-overlay "$OVB" --a-gate-log "$GL"            --b-arm $ARM --b-stage 5 --b-root "$P/rollouts" --b-exclusions "$O/EXCLUSIONS5.json" --b-overlay "$OV" --b-gate-log "$GL" > "$O/pair_${BARM}_${ARM}.out" 2>&1
      if have "$O/pair_${BARM}_${ARM}/PAIR_T2.json"; then tail -1 "$O/pair_${BARM}_${ARM}.out"; BLINDPAIR="$O/pair_${BARM}_${ARM}/PAIR_T2.json"
      else echo "ZZ beside pair $BARM - $ARM wrote no PAIR_T2.json (beside only; the outcome does not wait for it) ZZ"; fi
    else echo "ZZ beside: exclusions of the BLIND pull failed (beside only; the outcome does not wait for it) ZZ"; tail -3 "$OB/exclusions5.out"; fi
  fi
else
  echo "beside skipped: no BLIND pull at $PB (C0CRE_BLIND=${C0CRE_BLIND:-1}); run again with C0CRE_BLIND_ONLY=1 once it has landed (steps 2-4 are not repeated)"
fi
# ---- 6. section 6, applied mechanically: RESULT_T2_RENC.json and the .md generated from it
BARGS=(); [ -n "$BLINDPAIR" ] && BARGS=(--blind-pair "$BLINDPAIR")
if [ "$BONLY" = 1 ]; then for e in json md; do [ -e "$O/RESULT_T2_RENC.$e" ] && x mv "$O/RESULT_T2_RENC.$e" "$O/RESULT_T2_RENC.before_blind.$e"; done; fi
x $PY -I $T/stage5/C0C-RE/c0cre_outcome.py --dom "$O/score_dom/DOMINANCE_T2.json" --pair "$O/pair_${ARM}_C0C/PAIR_T2.json" --arm $ARM --c0c-dom "$C0C_REF" "${BARGS[@]}" --out-dir "$O" | tee "$O/c0cre_outcome.out"
rc=${PIPESTATUS[0]}
echo "ZZ C0CRE-READ-DONE outcome-rc=$rc (0 = outcome applied, 2 = NOT-READ, 3 = INPUT-INVALID) ZZ"
exit $rc
