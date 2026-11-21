#!/bin/bash
# POD-SIDE, launched DETACHED by post_merge_to_p6gate.sh.   (c'2) of PREREG_3C.md = HANDOFF.md step 6, command for command:
#   cprime_check.py sample  ->  12 x reroll_twins.py run --arm held (the probe's "held, 12 s" arm, 4 s, CPU)  ->  cprime_check.py analyze
# Env in: ENV3C (the 3-C env file) F OUT IN EXPECT_SHA.   Writes ONLY under $F/c2.   Markers: PID, C2_FINISHED (always last), SHARD_RCS, ANALYZE_RC.
set -u
export PYTHONDONTWRITEBYTECODE=1                  # no __pycache__ written into the 3-C or P6 code trees
source "$ENV3C"                                   # PKG CODE3C PY run3c ...
C2=$F/c2
mkdir -p "$C2"
echo $$ > "$C2/PID"
rm -f "$C2/C2_FINISHED" "$C2/ANALYZE_RC" "$C2/SAMPLE_FAILED"
cur=$(sha256sum "$OUT/targets_rank1.jsonl" | cut -d' ' -f1)
if [ "$cur" != "$EXPECT_SHA" ]; then echo "merged bank changed under the script: $cur != $EXPECT_SHA" > "$C2/SAMPLE_FAILED"; touch "$C2/C2_FINISHED"; exit 3; fi
echo "$cur" > "$C2/merged_sha.txt"
if [ ! -f "$C2/sample.json" ]; then
  python3 "$CODE3C/cprime_check.py" sample --merged-dir "$OUT" --bank-in "$IN" --out "$C2" > "$C2/sample.out" 2>&1 \
    || { echo "cprime_check sample rc $?" > "$C2/SAMPLE_FAILED"; touch "$C2/C2_FINISHED"; exit 4; }
fi
cd "$PKG/refe" || { echo "no $PKG/refe" > "$C2/SAMPLE_FAILED"; touch "$C2/C2_FINISHED"; exit 5; }
NSH=12
pids=()
for ((i=0;i<NSH;i++)); do
  RES=""; [ -d "$C2/rr_held12/s$i" ] && RES="--resume"            # a second attempt resumes the shards the first one left; the first attempt is HANDOFF step 6 verbatim
  run3c $PY "$CODE3C/reroll_twins.py" run --bank "$C2/bank_c2" --out "$C2/rr_held12/s$i" --arm held --device cpu --log-shard $i/$NSH \
        --code-dir "$CODE3C" --refe-dir "$PKG/refe" --pkg-code-dir "$PKG/code" $RES > "$C2/l$i.log" 2>&1 &
  pids+=($!)
done
rcs=""
for p in "${pids[@]}"; do wait "$p"; rcs="$rcs $?"; done
echo "$rcs" > "$C2/SHARD_RCS"
python3 "$CODE3C/cprime_check.py" analyze --merged-dir "$OUT" --rr-held12 "$C2/rr_held12/s*/targets_reroll.jsonl" --out "$C2/cprime2.json" > "$C2/analyze.out" 2>&1
echo $? > "$C2/ANALYZE_RC"
touch "$C2/C2_FINISHED"
