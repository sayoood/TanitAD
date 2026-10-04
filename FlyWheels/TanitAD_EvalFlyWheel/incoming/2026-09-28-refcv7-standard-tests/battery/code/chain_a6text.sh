#!/bin/sh
# refcv7 step 50,400 four-family battery on the REGISTERED A6 TEXT verdict (Master Mind ruling 2026-10-04).
# Stages = chain_milestone.sh 1 + 3 + 4: T1 rolls (seed 0 every arm, seed 1 os-only), panels, four families,
# bars; perception pass + scoring; RESULT + sanitized bank. NO refcv6@38k baseline stage (MM: skip for now ->
# BAR-R7-2 reads PENDING). G0 is NOT re-run: the banked coded-FAIL g0.json goes in with --g0-json and the
# corrected-judge text verdict with --g0-text-verdict, which run_battery_r7.g0_text_override RE-VERIFIES on
# content (fail closed). Separate TAG: the coded-FAIL artifacts in battery/step50400 + raw/step50400 stay as is.
# Host memory: wait_commit (>= 6 GB free commit) before the lock, and the battery's gpu_gate (>= 6 GB free RAM)
# before every GPU stage; both fail closed. Assert on ARTIFACTS, never on exit codes; markers ZZ...ZZ.
STEP=50400
TAG=step50400_a6text
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery
PKGW='D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery'
CH=/d/refcv7_eval_kit/chain
CHW='D:/refcv7_eval_kit/chain'
KITW='D:/refcv7_eval_kit'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
EV7='C:/Users/Admin/ev7'
PP="$EV7/stack;$EV7/taniteval"
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
LOG=$CH/chain_$TAG.log
OUT=/d/refcv7_eval_kit/battery/$TAG
OUTW="D:/refcv7_eval_kit/battery/$TAG"
BANK="$PKGW/raw/$TAG"
WIN='D:/refcv7_eval_kit/windows/s2_windows.json'
CK="$KITW/ckpt/ckpt_$STEP.pt"
G0W='D:/refcv7_eval_kit/battery/step50400/g0.json'
TXT="$PKGW/raw/step50400/g0_A6_text.json"
RULING="Master Mind ruling 2026-10-04: the REGISTERED A6 text is the record (A6 item 7 keeps A2's low-support rule; hashed 2026-10-04T08:17:13+02:00, before any step-50,400 number). The coded judge omitted A6 from the A2 low-support tuple; the corrected judge applies the same criterion. Both records carried: G0-A6 as registered (text) PASS; G0-A6 as coded FAIL."
mkdir -p $OUT $CH
echo "ZZA6TCHAINSTART $(date +%FT%T%z) pid $$" >> $LOG
[ -s "$TXT" ] || { echo "ZZA6TNOTEXT $(date +%FT%T%z)" >> $LOG; exit 2; }
PYTHONPATH="$PP" $PY "$PKGW/code/wait_commit.py" --min-gb 6 --max-wait-s 43200 --out "$CHW/wait_commit_$TAG.json" >> $LOG 2>&1
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open(r'$CHW/wait_commit_$TAG.json'))['met'] else 1)"; then
  echo "ZZA6TNOCOMMIT $(date +%FT%T%z)" >> $LOG; exit 3
fi
JOB="refcv7-a6text-$TAG"
PYTHONPATH="$PP" $PY "$PKGW/code/gpu_lock.py" acquire --job $JOB --pid $$ --max-wait-s 43200 --out "$CHW/lock_$TAG.json" >> $LOG 2>&1
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open(r'$CHW/lock_$TAG.json'))['ok'] else 1)"; then
  echo "ZZA6TNOLOCK $(date +%FT%T%z)" >> $LOG; exit 3
fi
release() { PYTHONPATH="$PP" $PY "$PKGW/code/gpu_lock.py" release --job $JOB --pid $$ >> $LOG 2>&1; }
STOPW="$CHW/gpu_watch_$TAG.stop"
rm -f "$CH/gpu_watch_$TAG.stop"
PYTHONPATH="$PP" $PY "$PKGW/code/gpu_watch.py" --out "$CHW/gpu_watch_$TAG.jsonl" --stop-file "$STOPW" >> $LOG 2>&1 &
stopwatch() { : > "$CH/gpu_watch_$TAG.stop"; }
marker() { PYTHONPATH="$PP" $PY "$PKGW/code/chain_marker.py" "$1" --chain $JOB --pid $$ >> $LOG 2>&1; }
marker create
gyield() { PYTHONPATH="$PP" $PY "$PKGW/code/gpu_yield.py" --job $JOB --pid $$ --stage "$1" --log "$CHW/gpu_yields.jsonl" >> $LOG 2>&1; }
trap 'stopwatch; release; marker remove' EXIT
echo "ZZA6TLOCKED $(date +%FT%T%z)" >> $LOG
bank() { PYTHONPATH="$PP" $PY "$PKGW/code/bank_tag.py" "$OUTW" "$BANK" --with-dumps >> $LOG 2>&1; }
finish() {
  PYTHONPATH="$PP" $PY "$PKGW/code/result_r7.py" --tag $TAG >> $LOG 2>&1
  [ -s $OUT/RESULT_$TAG.json ] && echo "ZZA6TRESULT $(date +%FT%T%z)" >> $LOG || echo "ZZA6TRESULTNOJSON" >> $LOG
  bank
  [ -s "$PKG/raw/$TAG/RESULT_$TAG.json" ] && echo "ZZA6TBANKED $(date +%FT%T%z)" >> $LOG || echo "ZZA6TBANKFAIL" >> $LOG
}
battery() {
  PYTHONPATH="$PP" $PY "$PKGW/code/run_battery_r7.py" --ckpt "$CK" --config "$KITW/ckpt/config.json" \
      --g0-json "$G0W" --g0-text-verdict "$TXT" --g0-text-ruling "$RULING" --skip-roll --tag $TAG --lock-held \
      --yield-job $JOB --yield-pid $$ --yield-log "$CHW/gpu_yields.jsonl" >> $CH/battery_$TAG.log 2>&1
}
nbars() { $PY -c "import json; s=json.load(open(r'$OUTW/battery_summary.json',encoding='utf-8')); print(len(s.get('bars') or []))" 2>/dev/null; }
ovr() { $PY -c "import json; s=json.load(open(r'$OUTW/battery_summary.json',encoding='utf-8')); print('OK' if ((s.get('stages') or {}).get('g0') or {}).get('text_override') else 'NO')" 2>/dev/null; }
battery
if [ "$(nbars)" != "3" ] && [ "$(ovr)" = "OK" ]; then
  echo "ZZA6TBATTERYRETRY $(date +%FT%T%z)" >> $LOG
  battery
fi
if [ "$(ovr)" != "OK" ]; then
  echo "ZZA6TOVERRIDEREFUSED $(date +%FT%T%z) -- see battery_$TAG.log; no battery number" >> $LOG
  finish; exit 4
fi
echo "ZZA6TBATTERYDONE bars=$(nbars) $(date +%FT%T%z)" >> $LOG
PYTHONPATH="$PP" CUDA_VISIBLE_DEVICES=-1 $PY "$PKGW/code/void_gates_map.py" --g0 "$G0W" --out "$OUTW/void_gates_map.json" >> $LOG 2>&1
P6=""
[ -s /d/refcv7_eval_kit/baseline_refcv6_38k/perc_refcv6_38k.pkl ] && P6="--refcv6-pkl D:/refcv7_eval_kit/baseline_refcv6_38k/perc_refcv6_38k.pkl"
PR=""
[ -s /d/refcv7_eval_kit/prior/prior_train300.npz ] && PR="--prior-npz D:/refcv7_eval_kit/prior/prior_train300.npz"
perc7() {
  # the host-memory guard before this GPU stage too (fail closed: no pass if commit never reaches 6 GB)
  PYTHONPATH="$PP" $PY "$PKGW/code/wait_commit.py" --min-gb 6 --max-wait-s 21600 --out "$CHW/wait_commit_${TAG}_perc.json" >> $LOG 2>&1
  if ! $PY -c "import json,sys; sys.exit(0 if json.load(open(r'$CHW/wait_commit_${TAG}_perc.json'))['met'] else 1)"; then
    echo "ZZA6TPERCNOCOMMIT $(date +%FT%T%z)" >> $LOG; return 0
  fi
  PYTHONPATH="$PP" $PY "$PKGW/code/perc_pass_r7.py" --ckpt "$CK" --config "$KITW/ckpt/config.json" \
      --windows-json "$WIN" --out "$KITW/perc/$TAG/perc" $P6 $PR >> $CH/perc7_$TAG.log 2>&1
}
[ -s /d/refcv7_eval_kit/perc/$TAG/perc.json ] || perc7
[ -s /d/refcv7_eval_kit/perc/$TAG/perc.json ] || { echo "ZZA6TPERC7RETRY $(date +%FT%T%z)" >> $LOG; perc7; }
gyield perc_refcv7
if [ -s /d/refcv7_eval_kit/perc/$TAG/perc.json ]; then
  echo "ZZA6TPERC7OK $(date +%FT%T%z)" >> $LOG
  PYTHONPATH="$PP" $PY "$PKGW/code/perc_score.py" --prefix "$KITW/perc/$TAG/perc" --out "$OUTW/perc_score.json" --milestone > $CH/perc_score_$TAG.log 2>&1
  [ -s $OUT/perc_score.json ] && echo "ZZA6TPERCSCOREOK $(date +%FT%T%z)" >> $LOG || echo "ZZA6TPERCSCORENOJSON" >> $LOG
else
  echo "ZZA6TPERC7NOJSON $(date +%FT%T%z)" >> $LOG
fi
NC=$(grep -c '"COLLISION": true' "$CH/gpu_watch_$TAG.jsonl" 2>/dev/null)
echo "ZZA6TCOLLISIONROWS ${NC:-0}" >> $LOG
cp "$CH/gpu_watch_$TAG.jsonl" $OUT/gpu_watch.jsonl 2>/dev/null
finish
echo "ZZA6TCHAINEND $(date +%FT%T%z)" >> $LOG
