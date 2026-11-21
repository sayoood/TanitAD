#!/bin/bash
# PREREG_V8F_HACC_MC (Project Steering/PREREG_V8F_HACC_MC.md @ 8d4b936d): the whole eval matrix as ONE detached chain on the dev-box CPU (ZERO training).
#
#   usage:  bash hmc_chain.sh <TOOL_TREE>        TOOL_TREE = a repo export whose stack/scripts/v8f_t1_eval.py has BOTH `--limit causal` AND `--dump-ticks` (forward slashes: C:/Users/Admin/xxx)
#
#   0  guards: the tool has --dump-ticks AND the causal bucket (grep, each with a same-breath control; then the tool's own --help) -- REFUSED otherwise;
#      sha256-pin the TOOL TREE (the tool + every .py it can import), the baked checkpoints, BAKE_REPORT_MC.json and the tables; inputs md5
#   1  C-7: the frozen suite of the TOOL TREE (one pytest process), read by the reader from logs/suite.out
#   2  two lanes (s0, s1), 2 python processes at a time, OMP/MKL 8 threads each (the thread count the banked dumps were made with):
#        per seed, in this order:  MC-H m10, STEEP m10, MC-F m10, MC-H m30, STEEP m30, MC-F m30
#        each = v8f_t1_eval.py run --replan-steps {10|30} --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks  (default --batch 32: do not change it)
#   3  the reader's CONTROLS-ONLY pass (no outcome row is opened) -> run/READ_CONTROLS.json
# Markers in logs/chain.out:   ZZHMC-DONE <n runs with a json (12)>   |   ZZHMC-STOP <reason>      (STOP also touches logs/STOP; the other lane stops at its next step)
# Resume: re-run the same command; finished runs (json + ticks + 141 episode dumps) are skipped, the pins are re-asserted.  A changed TOOL TREE stops the chain (delete run/PINS.txt only on purpose).
# HMC_SMOKE=1: plumbing test only -- 24 windows, outputs under smoke/chain, tags SMOKE_*, marker ZZHMC-SMOKE-DONE, the causal-bucket guard waived (uses --limit logged when the tool lacks it).
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
H=/c/Users/Admin/hmc
SMOKE=${HMC_SMOKE:-0}
if [ "$SMOKE" = 1 ]; then OUT=${HMC_OUT:-$H/smoke/chain}; LOG=${HMC_LOG:-$H/smoke/chain/logs}; PFX=SMOKE_; MAXW="--max-windows 24"; else OUT=${HMC_OUT:-/e/Projects/TanitAD-artifacts/v8f-mini-hacc-mc/run}; LOG=${HMC_LOG:-$H/logs}; PFX=""; MAXW=""; fi      # run output (~2 GB incl. 141-file dump dirs) on E: -- C: has < 1 GB free
PT=/e/Projects/TanitAD-artifacts/v8f-mini-hacc-mc/pt
mkdir -p "$LOG" "$OUT" "$OUT/ticks" "$H/pt" "$PT"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG/chain.out"; }
stop() { say "STOP: $*"; echo "ZZHMC-STOP $*" >> "$LOG/chain.out"; touch "$LOG/STOP"; exit 1; }
[ $# -ge 1 ] || { echo "usage: hmc_chain.sh <TOOL_TREE>"; echo "ZZHMC-STOP no TOOL_TREE argument" >> "$LOG/chain.out"; exit 1; }
TTu=$(cygpath -u "$1"); TTm=$(cygpath -m "$1")
TOOL="$TTu/stack/scripts/v8f_t1_eval.py"
[ -f "$TOOL" ] || stop "no tool at $TOOL"
rm -f "$LOG/STOP"

# ------------------------------------------------------------------------------------------------ 0a  the flag guards (grep, with same-breath controls)
c_pos=$(grep -c -e '--replan-steps' "$TOOL"); c_neg=$(grep -c -e '--zzz-no-such-flag-zzz' "$TOOL")
[ "$c_pos" -ge 1 ] && [ "$c_neg" -eq 0 ] || stop "grep control broken on $TOOL (positive control --replan-steps = $c_pos, must be >= 1; negative control = $c_neg, must be 0)"
n_dump=$(grep -c -e '--dump-ticks' "$TOOL")
n_caus=$(grep -cE 'LIMIT_MODES *= *\(.*"causal"' "$TOOL")
say "guards: --dump-ticks grep=$n_dump, causal-in-LIMIT_MODES grep=$n_caus (controls: --replan-steps=$c_pos, no-such-flag=$c_neg) in $TTm"
[ "$n_dump" -ge 1 ] || stop "REFUSED: $TTm has no --dump-ticks (grep $n_dump; the control read $c_pos, so the grep works)"
if [ "$SMOKE" = 1 ] && [ "$n_caus" -eq 0 ]; then LIMIT=logged; say "SMOKE: the tool has no causal bucket; using --limit logged (plumbing test only)"; else
  [ "$n_caus" -ge 1 ] || stop "REFUSED: $TTm has no causal bucket (LIMIT_MODES grep $n_caus; the control read $c_pos, so the grep works)"
  LIMIT=causal
fi
if [ "${HMC_DRY:-0}" = 1 ]; then say "DRY: grep guards passed (limit=$LIMIT)"; echo "ZZHMC-DRY-OK" >> "$LOG/chain.out"; exit 0; fi

PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
m() { cygpath -m "$1"; }
Hm=$(m "$H"); OUTm=$(m "$OUT"); LOGm=$(m "$LOG")
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TTu/stack")
export PYTHONPATH="$TTm/stack;$TTm/taniteval;$TTm/stack/scripts"
HELP=$("$PY" "$TTm/stack/scripts/v8f_t1_eval.py" run --help 2>&1 < /dev/null)
HELP1=$(echo "$HELP" | tr -s '[:space:]' ' ')
echo "$HELP1" | grep -q -e '--dump-ticks' || stop "REFUSED: the tool's own --help lists no --dump-ticks"
if [ "$LIMIT" = causal ]; then echo "$HELP1" | grep -qE '\{logged,[^}]*causal' || stop "REFUSED: the tool's own --help lists no causal choice for --limit"; fi
say "tool --help agrees (limit=$LIMIT)"

# ------------------------------------------------------------------------------------------------ 0b  pins
tree_digest() {   # one sha256 over every .py the tool can import (stack/tanitad, taniteval) + the tool
  ( cd "$TTu" && { find stack/tanitad taniteval -name '*.py' -type f; echo stack/scripts/v8f_t1_eval.py; } | LC_ALL=C sort -u | tr '\n' '\0' | xargs -0 sha256sum | sha256sum | cut -c1-64 )
}
keyfiles="stack/scripts/v8f_t1_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py taniteval/taniteval/floors_kin.py taniteval/taniteval/four_families.py taniteval/taniteval/ci.py taniteval/tools/t1_eval.py"
BR=$H/BAKE_REPORT_MC.json
if [ ! -e "$OUT/PINS.txt" ]; then
  { echo "tree=$TTm"; echo "digest=$(tree_digest)"; for f in $keyfiles; do echo "file=$f $(sha256sum < "$TTu/$f" | cut -c1-64)"; done
    echo "bake_report=$(sha256sum < "$BR" | cut -c1-64)"; for f in "$H"/tables/table_*.json; do echo "table=$(basename "$f") $(sha256sum < "$f" | cut -c1-64)"; done; } > "$OUT/PINS.txt"
  say "pins written: tree digest $(grep ^digest= "$OUT/PINS.txt" | cut -c8-19), tool $(grep 'file=stack/scripts' "$OUT/PINS.txt" | cut -d' ' -f2 | cut -c1-12)"
fi
[ "$(grep ^tree= "$OUT/PINS.txt" | cut -c6-)" = "$TTm" ] || stop "PINS.txt belongs to another TOOL_TREE ($(grep ^tree= "$OUT/PINS.txt")); delete it deliberately to re-pin"
pin_check() {
  [ "$(grep ^digest= "$OUT/PINS.txt" | cut -c8-)" = "$(tree_digest)" ] || stop "the tool tree changed ($1)"
  [ "$(grep ^bake_report= "$OUT/PINS.txt" | cut -c13-)" = "$(sha256sum < "$BR" | cut -c1-64)" ] || stop "BAKE_REPORT_MC.json changed ($1)"
  for f in "$H"/tables/table_*.json; do [ "$(grep "^table=$(basename "$f") " "$OUT/PINS.txt" | cut -d' ' -f2)" = "$(sha256sum < "$f" | cut -c1-64)" ] || stop "$(basename "$f") changed ($1)"; done
}
pin_check "start"
SC=/c/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
SF=$SC/sfresh; SFm=$(m "$SF")
GRID=$(m "$H/../hlat/inputs/eval_grid_windows.json"); JOIN=$(m "$H/../hlat/inputs/join_eval141.npz"); TRACKS=$SFm/inputs/native_tracks.pt
md5chk() { [ "$(md5sum < "$1" | cut -c1-32)" = "$2" ] || stop "md5 $1"; }
md5chk "$GRID" 34ed8ac56365eeff72e42581bde2f5c4
md5chk "$JOIN" 532fcd644c0e0835e847b2f1cee011ad
md5chk "$TRACKS" 189f4e68913c0163912a509203963f8d
[ "$(ls "$SF/cache/eval/clips" | wc -l)" = 141 ] || stop "eval cache is not 141 clips"
# every baked checkpoint is the one BAKE_REPORT_MC.json recorded (sha256), and the shipped HACC checkpoints are the pinned ones
"$PY" -c "
import json
r = json.load(open('$(m "$BR")'))
for s in ('s0', 's1'):
    for a, v in r['ckpts'][s]['arms'].items():
        print(v['out'], v['sha256'])
    print(r['ckpts'][s]['source_ckpt'], r['ckpts'][s]['source_sha256'])
" | tr -d '\r' > "$OUT/ckpt_pins.txt" || stop "cannot read BAKE_REPORT_MC.json"
[ "$(wc -l < "$OUT/ckpt_pins.txt")" = 8 ] || stop "BAKE_REPORT_MC.json does not list 6 baked + 2 source checkpoints"
while read -r p sha; do [ "$(sha256sum < "$p" | cut -c1-64)" = "$sha" ] || stop "checkpoint sha256 $p"; done < "$OUT/ckpt_pins.txt"
[ "$(grep -c '"C-4_as_written_PASS": true' "$BR")" = 1 ] || say "NOTE: BAKE_REPORT_MC.json records C-4 (as written) = FALSE: the reader will REFUSE unless a PREREG amendment is supplied (see HANDOFF)"
say "inputs OK (grid, join, tracks, eval cache, 6 baked + 2 source checkpoints, report and tables pinned)"

# ------------------------------------------------------------------------------------------------ 1  C-7 the suite
if [ "$SMOKE" != 1 ] && [ ! -e "$LOG/suite.ok" ]; then
  say "== C-7 suite of the tool tree"
  files=""; for f in "$TTu"/stack/tests/test_v8f_t1*.py "$TTu"/stack/tests/test_v8f_hacc_infer.py "$TTu"/stack/tests/test_v8f_hlat_gate.py "$TTu"/stack/tests/test_v8f_hlat.py "$TTu"/stack/tests/test_v8f_mini.py "$TTu"/stack/tests/test_v8f_oacc.py; do [ -f "$f" ] && files="$files tests/$(basename "$f")"; done
  say "suite files:$files"
  ( cd "$TTu/stack" && "$PY" -m pytest -q -p no:cacheprovider --basetemp "$(m "$PT")/suite" $files ) > "$LOG/suite.out" 2>&1 < /dev/null
  last=$(tail -1 "$LOG/suite.out")
  { echo "$last" | grep -qE '[0-9]+ passed' && ! echo "$last" | grep -qE 'failed|error'; } || stop "C-7: suite not green: $last"
  echo "$last" > "$LOG/suite.ok"
  say "C-7 suite: $last"
  pin_check "after suite"
fi

# ------------------------------------------------------------------------------------------------ 2  the matrix
EV="--eval-cache $SFm/cache --eval-grid $GRID --eval-join $JOIN"
nostop() { [ -e "$LOG/STOP" ] && exit 1; return 0; }
t1run() {   # t1run <seed> <arm tag> <m>
  local s=$1 a=$2 mm=$3
  local tag="${PFX}${a}_s${s}_m${mm}" ck="E:/Projects/TanitAD-artifacts/v8f-mini-hacc-mc/M2b_s${s}_hmc_${a}.pt"
  local want=141; [ "$SMOKE" = 1 ] && want=24
  if [ -f "$OUT/t1v8f_$tag.json" ] && [ -f "$OUT/ticks/ticks_$tag.npz" ] && [ "$(ls "$OUT/dump_$tag" 2>/dev/null | wc -l)" -ge "$([ "$SMOKE" = 1 ] && echo 1 || echo 141)" ]; then say "skip $tag (complete)"; return 0; fi
  [ -n "$OUT" ] && [ -n "$tag" ] && rm -rf "$OUT/dump_$tag" "$OUT/ticks/ticks_$tag.npz" "$OUT/t1v8f_$tag.json"        # our own incomplete output of an earlier attempt
  nostop; pin_check "before $tag"
  local t0=$(date +%s)
  say "start $tag (ckpt $(basename "$ck"), limit $LIMIT, m $mm)"
  "$PY" "$TTm/stack/scripts/v8f_t1_eval.py" run --ckpt "$ck" --tag "$tag" --tracks "$TRACKS" $EV --out-dir "$OUTm" \
     --replan-steps "$mm" --nav deadreckon --scene logged --schedule periodic --limit "$LIMIT" --dump-ticks "$OUTm/ticks" --device cpu $MAXW \
     > "$LOG/$tag.out" 2>&1 < /dev/null
  local rc=$?
  say "end $tag rc=$rc wall=$(( $(date +%s) - t0 ))s"
  pin_check "after $tag"
  { [ $rc -eq 0 ] && [ -f "$OUT/t1v8f_$tag.json" ] && [ -f "$OUT/ticks/ticks_$tag.npz" ]; } || stop "run $tag failed (rc $rc): $(tail -2 "$LOG/$tag.out" | tr '\n' ' ')"
}
lane() {
  local s=$1
  for spec in "MCH 10" "STEEP 10" "MCF 10" "MCH 30" "STEEP 30" "MCF 30"; do set -- $spec; nostop; t1run "$s" "$1" "$2"; done
  say "lane s$s done"
}
say "== matrix: 3 arms x 2 seeds x {m10, m30} (limit $LIMIT$([ "$SMOKE" = 1 ] && echo ', SMOKE 24 windows'))"
lane 0 & P0=$!
lane 1 & P1=$!
wait $P0 $P1
[ -e "$LOG/STOP" ] && { say "a lane stopped: no DONE"; exit 1; }
pin_check "end"
n=$(ls "$OUT"/t1v8f_${PFX}*.json 2>/dev/null | wc -l)

# ------------------------------------------------------------------------------------------------ 3  controls-only read (no outcome row is opened)
if [ "$SMOKE" != 1 ]; then
  "$PY" "$Hm/hmc_read.py" "$TTm" --controls-only --run-dir "$OUTm" > "$LOG/read_controls.out" 2>&1 < /dev/null
  say "controls-only read rc=$?: $(tail -1 "$LOG/read_controls.out" | cut -c1-300)"
fi
say "runs with a json: $n / 12"
if [ "$SMOKE" = 1 ]; then echo "ZZHMC-SMOKE-DONE $n" >> "$LOG/chain.out"; else echo "ZZHMC-DONE $n" >> "$LOG/chain.out"; fi
