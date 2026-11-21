#!/bin/bash
# k2dc02tail_s6c_pull_score.sh [DRY] -- DEV-BOX half of the K2DC02-TAIL read (PREREG_K2DC02TAIL_2026-11-20 + AMENDMENT1). Build agent for MM 4a3d05, 2026-11-20.
# Derived from e2tail_s6c_pull_score.sh and E2-WISE wire-off's wrapper, NOT from the K2DC wrapper: this one exports stack/scripts (incl. r9_e1_eval.py) + stack/tanitad, which the box-paired step needs.
# REFUSES (exit 6, ZZK2DC02T-S6C-ABORT-NOAWAITZZ) until Thor's stage B reached AWAIT-S6C. It does not wait: poll with watch_k2dc02tail.sh or re-run.
# After AWAIT-S6C it
#   1 pulls   $W/stageB/{read (minus dump_K0 = a header-only stub), E2_K2_mapsel, E2_K2_pf}, $W/cmp/E2_K0_pf, $W/cmp_k2dc/E2_K2_pf  to $D/pull, each file sha256-verified against a Thor-side list
#             (stageB/E2_K2_pf = K2DC02t's own planner dumps; the dose rule needs it though the kickoff list omitted it)
#   2 assembles the K0 slot from E2-TAIL's dev-box pull ($E2TSRC: read/dump_K0, E2_K0_mapsel) by COPY to C:, sha256-checked against that pull's thor.sha256 (F: files are never moved or changed)
#   3 exports the scoring code at the PINNED commit (blob-verified, every file)
#   4 runs r9e2_s6c_resume.sh: s6c_dev x4 (K2DC02t and K0t, seeds 0 and 1) -> box-paired -> INPUTS -> r9_e2_verdict.py (the E2 verdict K2DC02t vs K0t, exactly as E2-TAIL)
# Markers (stdout and $D/logs/s6c.log): ZZK2DC02T-S6C-DONEZZ (resume rc 0 GO or 1 NO-GO) | ZZK2DC02T-S6C-ABORT-<why>ZZ.   EXIT: 0 GO | 1 NO-GO | 6 refused/aborted here | 3 resume abort | 2 resume REFUSED | 4 resume refused to start.
# DRY: every guard incl. the AWAIT check (still refuses without it), the code export, the local K0-slot sha check; NO pull, NO python. Ends ZZK2DC02T-S6C-DRY-OKZZ.
# Disk: ~ (Thor sizes) + 2.8 GB for the K0 slot on C:; refuses unless C: keeps >= 6 GB free afterwards.
MODE=${1:-}
. "$(dirname "$0")/k2dc02tail_common.sh"
LOG=$D/logs/s6c.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
die() { say "ZZK2DC02T-S6C-ABORT-$1ZZ $2"; exit 6; }
OUT=$W/stageB; P=$D/pull; MS=$P/stageB; RD=$MS/read
say "START (pid $$) mode ${MODE:-REAL} tip ${TIP:0:12}"
[ -z "$MODE" ] || [ "$MODE" = DRY ] || die USAGE "usage: k2dc02tail_s6c_pull_score.sh [DRY]"
[ "$(th "echo up")" = up ] || die NOSSH "cannot reach tanitad-thor"
[ "$(th "[ -e $OUT/ROLL_ABORTED ] && echo Y")" = Y ] && die ROLLABORTED "$OUT/ROLL_ABORTED exists: $(th "head -c 200 $OUT/ROLL_ABORTED" | tr '\n' ' ')"
AW=$(th "grep -c 'AWAIT-S6C' $OUT/ROLL_STATUS.txt")
[ "${AW:-0}" -ge 1 ] 2>/dev/null || die NOAWAIT "no AWAIT-S6C in $OUT/ROLL_STATUS.txt (stage B not done): $(th "tail -1 $OUT/ROLL_STATUS.txt" | cut -c1-200)"
[ "$(th "grep -c ' DONE ' $W/CMP_STATUS.txt")" -ge 1 ] 2>/dev/null || die NOCMP "no DONE in $W/CMP_STATUS.txt: the comparator re-roll did not finish"
for p in $W/cmp/E2_K0_pf $W/cmp_k2dc/E2_K2_pf $OUT/E2_K2_pf $OUT/E2_K2_mapsel $OUT/read; do [ "$(th "[ -d $p ] && echo Y")" = Y ] || die THORIN "missing on Thor: $p"; done
[ -s "$E2TSRC/thor.sha256" ] && [ -d "$E2TSRC/pull/read/dump_K0" ] && [ -d "$E2TSRC/pull/E2_K0_mapsel" ] || die NOK0SRC "E2-TAIL dev-box pull incomplete under $E2TSRC"
[ "$MODE" = DRY ] || { [ -e "$P" ] && die EXISTS "$P exists (a previous pull): MM decision, by literal path"; }
export_code; rc=$?; [ $rc = 0 ] || die EXPORT "export_code rc $rc (11 tip not in mirror, 12 half-made $C, 13 archive, 14-16 blob check)"
say "code $C: $(cat "$C/.verified")"
# the K0 slot's source integrity (E2-TAIL's own thor.sha256): checked on the COPY in REAL mode, on the F: source in DRY mode
grep -E '^[0-9a-f]{64}  (read/dump_K0/|E2_K0_mapsel/)' "$E2TSRC/thor.sha256" > "$D/sha/e2tail_k0slot.sha256"
[ -s "$D/sha/e2tail_k0slot.sha256" ] || die NOK0LIST "no K0-slot lines in $E2TSRC/thor.sha256"
if [ "$MODE" = DRY ]; then
  K0N=$(wc -l < "$D/sha/e2tail_k0slot.sha256"); say "DRY: AWAIT present, guards pass, K0-slot list $K0N files; no pull"
  say "ZZK2DC02T-S6C-DRY-OKZZ"; exit 0
fi
NEED=$(th "cd $W && du -sm stageB/read stageB/E2_K2_mapsel stageB/E2_K2_pf cmp/E2_K0_pf cmp_k2dc/E2_K2_pf | awk '{s+=\$1} END {print s}'")
FREE=$(df -BM "$D" | awk 'NR==2 {gsub("M","",$4); print $4}')
[ -n "$NEED" ] && [ "$FREE" -gt $((NEED + 2800 + 6144)) ] 2>/dev/null || die CDISK "$D drive free ${FREE} MB; pull ${NEED} MB + K0 slot 2800 MB + 6144 MB floor"
mkdir -p "$MS" "$P/cmp" "$P/cmp_k2dc" || die MKDIR "$P"
say "pulling ~${NEED} MB (C: free ${FREE} MB)"
# 1 pulls. pull TAG SRCDIR DSTDIR "<find excludes>" "<tar excludes>" entries...
pull() {
  local tag=$1 src=$2 dst=$3 fx=$4 tx=$5; shift 5
  TH_TO=3600 th "cd $src && find $* -type f $fx -print0 | sort -z | xargs -0 sha256sum" > "$D/sha/thor_$tag.sha256"
  [ -s "$D/sha/thor_$tag.sha256" ] || die THORSHA "no sha256 list for $tag"
  timeout 3600 ssh -n -o ConnectTimeout=20 tanitad-thor "cd $src && tar -cf - $tx $*" | tar --force-local -xf - -C "$dst" || die PULL "tar stream failed ($tag)"
  (cd "$dst" && sha256sum -c --quiet "$D/sha/thor_$tag.sha256") > "$D/sha/check_$tag.txt" 2>&1 || die PULLSHA "$tag sha256 mismatch: $(head -3 "$D/sha/check_$tag.txt" | tr '\n' ' ')"
  local nx; nx=$(cd "$dst" && find "$@" -type f | wc -l)
  [ "$nx" = "$(wc -l < "$D/sha/thor_$tag.sha256")" ] || die PULLCOUNT "$tag: $nx files on disk vs $(wc -l < "$D/sha/thor_$tag.sha256") listed"
  say "pulled $tag: $nx files sha256-equal to Thor"
}
pull stageB "$OUT" "$MS" "! -path 'read/dump_K0/*'" "--exclude=read/dump_K0" read E2_K2_mapsel E2_K2_pf
pull cmp_k0 "$W/cmp" "$P/cmp" "" "" E2_K0_pf
pull cmp_k2dc "$W/cmp_k2dc" "$P/cmp_k2dc" "" "" E2_K2_pf
# 2 the K0 slot: COPY from E2-TAIL's pull, then verify the copy against E2-TAIL's thor.sha256
cp -r "$E2TSRC/pull/read/dump_K0" "$RD/dump_K0" && cp -r "$E2TSRC/pull/E2_K0_mapsel" "$MS/E2_K0_mapsel" || die K0COPY "cp from $E2TSRC/pull"
(cd "$MS" && sha256sum -c --quiet "$D/sha/e2tail_k0slot.sha256") > "$D/sha/check_k0slot.txt" 2>&1 || die K0SHA "K0 slot differs from E2-TAIL's thor.sha256: $(head -3 "$D/sha/check_k0slot.txt" | tr '\n' ' ')"
say "K0 slot copied from $E2TSRC and sha256-equal to E2-TAIL's thor.sha256 ($(wc -l < "$D/sha/e2tail_k0slot.sha256") files)"
# Thor's seeded K0 traces and K0 fitted EVAL must be those same E2-TAIL files
for s in 0 1; do
  [ "$(th "sha256sum $OUT/E2_K0_mapsel/trace_s$s.npz" | cut -d' ' -f1)" = "$(sha256sum "$MS/E2_K0_mapsel/trace_s$s.npz" | cut -d' ' -f1)" ] || die K0SEED "Thor's seeded E2_K0_mapsel/trace_s$s.npz != E2-TAIL's"
done
[ "$(git --git-dir=$GD hash-object --no-filters "$RD/EVAL_K0.json")" = "$(git --git-dir=$GD rev-parse "$TIP:$E2P/e2/e2raw/e2tail/stageA/EVAL_K0tail_fitthr.json")" ] || die K0EVAL "pulled read/EVAL_K0.json != the banked E2-TAIL EVAL_K0tail_fitthr.json at $TIP"
say "K0 slot seeds verified (traces == E2-TAIL's, EVAL_K0.json == banked EVAL_K0tail_fitthr.json)"
# tail guard (PREREG sec. 2.1): the K2DC02t DEV dump records the tail file's step 6000
"$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if int(d.get('step',-1))==6000 and d.get('ckpt_md5') else 1)" "$(W2 "$RD/dump_K2/dump.json")" || die TAILGUARD "K2DC02t dump_K2 does not record step 6000 + ckpt_md5"
# 4 the resume (code from the pinned export; same environment as E2-TAIL's wrapper)
INCW="$C/$INC"
export R9E2_RD="$(W2 "$RD")" R9E2_MS_ROOT="$(W2 "$MS")" R9E2_SPLIT="$(W2 "$C/$SPL")" R9E2_MAPSEL_CODE="$(W2 "$INCW/2026-10-25-refcv9-mapsel/code")"
export R9E2_OPS="$(W2 "$C/stack/ops/r9e2_read")" R9E2_VERDICT="$(W2 "$C/stack/scripts/r9_e2_verdict.py")" R9E2_L2_CODE="$(W2 "$INCW/2026-10-25-refcv9-l2/code")" R9E2_PY="$PY"
export PYTHONPATH="$(W2 "$C/stack");$(W2 "$C/taniteval")" PYTHONIOENCODING=utf-8
gate s6c || die COMMITLOW "commit free < 5632 MB for 8 h; inputs are pulled, rerun after deleting nothing (the pull is kept)"
say "running r9e2_s6c_resume.sh"
bash "$C/stack/ops/r9e2_read/r9e2_s6c_resume.sh" > "$D/s6c_resume.log" 2>&1 < /dev/null; RC=$?
tail -4 "$D/s6c_resume.log" | cut -c1-240 | tee -a "$LOG"
case "$RC" in 0|1) say "ZZK2DC02T-S6C-DONEZZ rc $RC (0 GO / 1 NO-GO) log $D/s6c_resume.log"; exit "$RC";;
  *) say "ZZK2DC02T-S6C-ABORT-RESUMEZZ rc $RC (2 REFUSED / 3 abort / 4 refused) log $D/s6c_resume.log"; exit "$RC";; esac
