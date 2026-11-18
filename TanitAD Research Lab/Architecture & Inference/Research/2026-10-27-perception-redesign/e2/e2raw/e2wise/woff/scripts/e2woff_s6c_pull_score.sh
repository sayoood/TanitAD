#!/bin/bash
# DERIVED by k2dc/make_woff_kit.py from the PLAN-RESET script of the same role: WIRE-OFF = K2DC's tail with its 6 box_gates and the
# map_chan_proj weight/bias at 0, K0's tail unchanged (model_wireoff.pt, wire_off.py), pre-registered in
# e2/e2raw/PREREG_E2WISE_2026-11-17_ADDENDUM4.md (landed 81d409d9, before any WIRE-OFF number).
# e2raw_s6c_pull_score.sh [DRY] -- MM 4a3d05, 2026-11-16: E2-WIRE-OFF STAGE B copy (2026-11-17) of e2_s6c_pull_score_k2dc.sh (OUT, D, LOG changed only). The DEV-BOX half of the E2 read: after Thor's r9e2_roll.sh stopped at
# ZZR9E2ROLL-AWAIT-S6CZZ (exit 5), pull <OUT>/read + E2_{K2,K0}_mapsel to C:, verify every file by sha256 against Thor, export the scoring code from the mirror tip
# (raw bytes, autocrlf off; r9e2_read + r9_e2_verdict.py + taniteval + the mapsel / l2 code + DEV_HOLD_SPLIT.json are unchanged cf653ca4bb47..tip except the env file),
# then run r9e2_s6c_resume.sh (s6c_dev x4 -> r9e2_inputs.py -> r9_e2_verdict.py). EXIT = the resume's: 0 GO | 1 NO-GO | 2 REFUSED | 3 abort | 4 refused to start;
# 6 = this wrapper refused (no AWAIT marker, disk, pull mismatch). DRY: no pull; checks the marker state, exports the code, runs the resume with R9E2_DRY=1.
export PATH="/usr/bin:/bin:$PATH"
MODE=${1:-}
OUT=/home/nvidia/r8ns/r9e2/runs/E2WISE_WOFF/stageB
D=/c/Users/Admin/r9e2p/woff/s6c; GD=C:/Users/Admin/tanitad-push/.git; BR=agent/arch-inf-20260803
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/AUTO_E2S6C_E2WISE_WOFF_B.txt
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
th() { timeout 120 ssh -n -o ConnectTimeout=20 tanitad-thor "$1" 2>/dev/null | tr -d '\r'; }
die() { say "ZZE2S6C-$1ZZ $2"; exit 6; }
say "START (pid $$) mode ${MODE:-REAL}"
AW=$(th "grep -c 'AWAIT-S6C' $OUT/ROLL_STATUS.txt")
if [ "$MODE" != DRY ]; then
  [ "${AW:-0}" -ge 1 ] 2>/dev/null || die NOAWAIT "no AWAIT-S6C in $OUT/ROLL_STATUS.txt: $(th "tail -2 $OUT/ROLL_STATUS.txt" | tr '\n' ' ' | cut -c1-200)"
  [ -e "$D/pull" ] && die EXISTS "$D/pull already exists (a previous pull): MM"
  NEED=$(th "du -sm $OUT/read $OUT/E2_K2_mapsel $OUT/E2_K0_mapsel | awk '{s+=\$1} END {print s}'")
  mkdir -p "$D" || die MKDIR "$D"   # MM 11-16: $D is a junction to F:/refcv9_e3/r9e2_read_K2DC; measure the drive that holds it, not /c
  FREE=$(df -BM "$D" | awk 'NR==2 {gsub("M","",$4); print $4}')
  [ -n "$NEED" ] && [ "$FREE" -gt $((NEED + 3000)) ] 2>/dev/null || die CDISK "$D drive free ${FREE} MB, pull needs ${NEED} MB + 3 GB margin"
  say "AWAIT-S6C present; pulling ${NEED} MB to $D (C: free ${FREE} MB)"
  mkdir -p "$D/pull" || die MKDIR "$D/pull"
  th "cd $OUT && find read E2_K2_mapsel E2_K0_mapsel -type f -print0 | sort -z | xargs -0 sha256sum" > "$D/thor.sha256"
  [ -s "$D/thor.sha256" ] || die THORSHA "no sha256 list from Thor"
  timeout 3600 ssh -n -o ConnectTimeout=20 tanitad-thor "cd $OUT && tar -cf - read E2_K2_mapsel E2_K0_mapsel" | tar --force-local -xf - -C "$D/pull" || die PULL "tar stream failed"
  (cd "$D/pull" && sha256sum -c --quiet "$D/thor.sha256") > "$D/sha_check.txt" 2>&1 || die PULLSHA "sha256 mismatch: $(head -3 "$D/sha_check.txt" | tr '\n' ' ')"
  say "pull verified: $(wc -l < "$D/thor.sha256") files sha256-equal"
else
  say "DRY: AWAIT-S6C count on Thor = ${AW:-none} (expected 0 before the read); no pull"
  mkdir -p "$D" || die MKDIR "$D"
fi
T=$(git --git-dir=$GD ls-remote origin $BR | cut -c1-40)
git --git-dir=$GD cat-file -e "$T^{commit}" || { git --git-dir=$GD fetch -q origin $BR; git --git-dir=$GD cat-file -e "$T^{commit}" || die TIP "tip $T not in the mirror"; }
INC="FlyWheels/TanitAD_EvalFlyWheel/incoming"; SPL="TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign/e0/DEV_HOLD_SPLIT.json"
C="$D/code_${T:0:12}"
if [ ! -d "$C" ]; then
  mkdir -p "$C" && git --git-dir=$GD -c core.autocrlf=false archive $T stack/ops/r9e2_read stack/scripts/r9_e2_verdict.py stack/scripts/r9_e1_eval.py stack/tanitad taniteval \
    "$INC/2026-10-25-refcv9-mapsel" "$INC/2026-10-25-refcv9-l2" "$SPL" | tar --force-local -x -C "$C" || die EXPORT "git archive $T"
  BAD=0
  while read -r mode typ blob path; do [ "$(git --git-dir=$GD hash-object --no-filters "$C/$path")" = "$blob" ] || { say "blob mismatch $path"; BAD=1; }; done \
    < <(git --git-dir=$GD ls-tree -r $T stack/ops/r9e2_read stack/scripts/r9_e2_verdict.py "$INC/2026-10-25-refcv9-mapsel/code")
  [ "$BAD" = 0 ] || die EXPORT "blob mismatch"
fi
W() { cygpath -w "$1"; }
export R9E2_RD="$(W "$D/pull/read")" R9E2_MS_ROOT="$(W "$D/pull")" R9E2_SPLIT="$(W "$C/$SPL")" R9E2_MAPSEL_CODE="$(W "$C/$INC/2026-10-25-refcv9-mapsel/code")"
export R9E2_OPS="$(W "$C/stack/ops/r9e2_read")" R9E2_VERDICT="$(W "$C/stack/scripts/r9_e2_verdict.py")" R9E2_L2_CODE="$(W "$C/$INC/2026-10-25-refcv9-l2/code")" R9E2_PY="$PY"
export PYTHONPATH="$(W "$C/stack");$(W "$C/taniteval")" PYTHONIOENCODING=utf-8
[ "$MODE" = DRY ] && export R9E2_DRY=1
say "code $C (tip $T); running r9e2_s6c_resume.sh${R9E2_DRY:+ (DRY)}"
bash "$C/stack/ops/r9e2_read/r9e2_s6c_resume.sh" > "$D/s6c_resume${R9E2_DRY:+_dry}.log" 2>&1 < /dev/null; RC=$?
tail -4 "$D/s6c_resume${R9E2_DRY:+_dry}.log" | cut -c1-240 | tee -a "$LOG"
say "ZZE2S6C-END rc $RC (0 GO / 1 NO-GO / 2 REFUSED / 3 abort / 4 refused) log $D/s6c_resume${R9E2_DRY:+_dry}.log"
exit $RC
