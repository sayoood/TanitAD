#!/bin/bash
# Runs the real lane-manifest block of files/scorer1_chain.sh against a throwaway directory.  Each case runs in its own subshell; refuse() prints and exits 1.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
CH=$HERE/files/scorer1_chain.sh
T=$(mktemp -d)
BL=$T/block.sh
sed -n '/^# >>> lane manifest/,/^# <<< lane manifest/p' "$CH" > "$BL"
[ -s "$BL" ] || { echo "ZZTEST_FAIL no block"; exit 1; }
fails=0
expect() {   # expect <ok|refuse> <name> <block file> <commands>
  local want=$1 name=$2 blk=$3 cmd=$4 out rc
  out=$( ( refuse() { echo "REFUSED: $*"; exit 1; }; D=$T/d; ROOT=$T/r; ARGS="--x 1 --teacher-lane-train $T/d/train_teacher_lane --y 2"; . "$blk"; eval "$cmd" ) 2>&1 ); rc=$?
  if { [ "$want" = ok ] && [ $rc -eq 0 ]; } || { [ "$want" = refuse ] && [ $rc -ne 0 ] && [[ "$out" == REFUSED:* ]]; }; then echo "PASS $name"; else echo "FAIL $name rc=$rc out=$(echo "$out" | cut -c1-160)"; fails=$((fails + 1)); fi
}
reset() { rm -rf "${T:?}/d" "${T:?}/r"; mkdir -p "$T/d/train_teacher_lane" "$T/r"; printf 'a\n' > "$T/d/train_teacher_lane/one.jsonl"; printf 'b\n' > "$T/d/train_teacher_lane/two.jsonl"; }
LD=$T/d/train_teacher_lane
reset; expect ok     "pin then check on the same files"   "$BL" 'lane_pin; lane_check'
reset; expect refuse "check before any pin"               "$BL" 'lane_check'
reset; expect refuse "one byte changed"                   "$BL" "lane_pin; printf 'c\n' > $LD/one.jsonl; lane_check"
reset; expect refuse "file added"                         "$BL" "lane_pin; printf 'x\n' > $LD/three.jsonl; lane_check"
reset; expect refuse "file removed"                       "$BL" "lane_pin; rm $LD/two.jsonl; lane_check"
reset; expect refuse "same bytes, renamed file"           "$BL" "lane_pin; mv $LD/two.jsonl $LD/zwei.jsonl; lane_check"
reset; expect refuse "second pin does not re-bless"       "$BL" "lane_pin; printf 'c\n' > $LD/one.jsonl; lane_pin; lane_check"
reset; rm "$LD"/*; expect refuse "empty directory"        "$BL" 'lane_pin'
reset; expect refuse "argv without the flag"              "$BL" 'lane_pin; ARGS="--x 1"; lane_check'
reset; expect refuse "argv with another directory"        "$BL" "lane_pin; ARGS=\"--teacher-lane-train $T/other\"; lane_check"
# deliberate regression: without the cmp line, a changed file must NOT be refused; this proves the case above is read by that line
grep -v 'cmp -s' "$BL" > "$T/mut.sh"
reset; expect ok     "MUTANT (cmp removed) misses the change" "$T/mut.sh" "lane_pin; printf 'c\n' > $LD/one.jsonl; lane_check"
for k in lane_pin 'lane_check; log "ZZSCORER1_LANE_DRAW_OK'; do grep -qF "$k" "$CH" || { echo "FAIL call site missing: $k"; fails=$((fails + 1)); }; done
[ "$(grep -c '^lane_pin$' "$CH")" = 1 ] || { echo "FAIL lane_pin call count"; fails=$((fails + 1)); }
bash -n "$CH" && echo "PASS bash -n" || { echo "FAIL bash -n"; fails=$((fails + 1)); }
rm -rf "${T:?}"
[ $fails -eq 0 ] && echo ZZTEST_OK || { echo "ZZTEST_FAIL $fails"; exit 1; }
