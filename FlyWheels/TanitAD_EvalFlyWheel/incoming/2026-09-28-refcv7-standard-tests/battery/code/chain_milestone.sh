#!/bin/sh
# refcv7 MILESTONE CHAIN (called by waiter_milestone.sh once D:/refcv7_eval_kit/ckpt/ckpt_<N>.pt is
# pulled and md5/step-verified). Dev box only. Runs, holding the dev-box GPU lock for the whole chain
# (the milestone battery has PRIORITY over NavSim, brief), every GPU stage as a child process:
#   1. run_battery_r7.py: G0 (8 seeds + M1/M2/M4 + wrapper) -> STOP unless PASS; T1 rolls at inference
#      seeds 0 and 1 on S2; panels; four families; BAR-R7-1/2/3 (R7-2 needs refcv6@38k's S2 dumps).
#   2. the baselines it needs, if the pre-stage has not banked them: refcv6@38k perception dump
#      (82c2331 tree) and refcv6@38k S2 rolls (the refcv6 package's own run_battery.py with its G0);
#      then the refcv7 battery's panels are re-read (--skip-roll) so BAR-R7-2 is evaluated.
#   3. perc_pass_r7.py (map 10 cm + box, refcv7 vs refcv6@38k vs positional prior, same windows),
#      perc_score.py -> BAR-M7-1..4, BAR-B7-1/2.
#   4. result_r7.py -> RESULT_step<N>.json + RESULT_SECTION_step<N>.md in the package.
# Assert on ARTIFACTS (the JSON each stage writes), never on an exit code; markers are ZZ...ZZ.
STEP="${1:-5000}"
TAG="step$STEP"
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery
PKGW='D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery'
R6PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
CH=/d/refcv7_eval_kit/chain
CHW='D:/refcv7_eval_kit/chain'
KITW='D:/refcv7_eval_kit'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
EV7='C:/Users/Admin/ev7'
EV6N='C:/Users/Admin/ev6_82c2331'
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
LOG=$CH/chain_$STEP.log
OUT=/d/refcv7_eval_kit/battery/$TAG          # WORK dir: raw clip ids live here, never landed
OUTW="D:/refcv7_eval_kit/battery/$TAG"
BANK="$PKGW/raw/$TAG"                         # the SANITIZED copy (sha12 only) that lands
WIN='D:/refcv7_eval_kit/windows/s2_windows.json'
[ -s /d/refcv7_eval_kit/windows/s2_windows.json ] || PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/make_s2_windows.py" "$WIN" >> $LOG 2>&1
mkdir -p $OUT $CH
echo "ZZCHAINSTART${STEP}ZZ $(date +%FT%T%z) pid $$" >> $LOG
CK="$KITW/ckpt/ckpt_$STEP.pt"
M=$($PY -c "import json; print(json.load(open(r'$CHW/pull_${STEP}_OK.json',encoding='utf-8'))['metrics_copy'])")
echo "ckpt $CK metrics $M" >> $LOG
# ---- host COMMIT first, the lock second (SPEC AMENDMENT A5 item 5) -------------------------- #
# MEASURED 2026-10-04 00:33: 2.0 GB free commit of 50 GB on the shared box; a G0 died at its first read
# with MemoryError. Waiting for memory BEFORE taking the lock avoids holding the card while another
# stream's process (which holds the memory) waits for the card.
PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/wait_commit.py" --min-gb 6 --max-wait-s 43200 \
    --out "$CHW/wait_commit_$STEP.json" >> $LOG 2>&1
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open(r'$CHW/wait_commit_$STEP.json'))['met'] else 1)"; then
  echo "ZZCHAINNOCOMMIT${STEP}ZZ $(date +%FT%T%z)" >> $LOG; exit 3
fi
echo "ZZCOMMITOK${STEP}ZZ $(date +%FT%T%z)" >> $LOG
# ---- the GPU lock, held for the whole chain ------------------------------------------------ #
JOB="refcv7-milestone-$TAG"
PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/gpu_lock.py" acquire --job $JOB --pid $$ \
    --max-wait-s 43200 --out "$CHW/lock_$STEP.json" >> $LOG 2>&1
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open(r'$CHW/lock_$STEP.json'))['ok'] else 1)"; then
  echo "ZZCHAINNOLOCK${STEP}ZZ $(date +%FT%T%z)" >> $LOG; exit 3
fi
release() { PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/gpu_lock.py" release --job $JOB --pid $$ >> $LOG 2>&1; }
# the sanitized bank (sha12 only; bank_tag.py refuses on any UUID hit; decisions/ stay on the dev box)
bank() { PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/bank_tag.py" "$OUTW" "$BANK" --with-dumps >> $LOG 2>&1; }
# the RESULT + sanitized bank + RESULT.md section + a NEW landing heading (both exit paths)
finish() {
  PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/result_r7.py" --tag $TAG >> $LOG 2>&1
  [ -s $OUT/RESULT_$TAG.json ] && echo "ZZRESULT${STEP}ZZ $(date +%FT%T%z)" >> $LOG || echo "ZZRESULTNOJSON${STEP}ZZ" >> $LOG
  bank
  [ -s "$PKG/raw/$TAG/RESULT_$TAG.json" ] && echo "ZZBANKED${STEP}ZZ $(date +%FT%T%z)" >> $LOG || echo "ZZBANKFAIL${STEP}ZZ" >> $LOG
  # the package RESULT.md gets the section; LANDING_READY.txt gets a NEW heading for this tag's files
  if [ -s "$PKG/raw/$TAG/RESULT_SECTION_$TAG.md" ]; then
    { echo ""; cat "$PKG/raw/$TAG/RESULT_SECTION_$TAG.md"; } >> "$PKG/RESULT.md"
  fi
  PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/make_landing.py" --append "raw/$TAG" RESULT.md     --heading "## $(date +%F) REFCV7-BATTERY $TAG (appended by chain_milestone.sh; sanitized, sha12 ids)" >> $LOG 2>&1
  echo "ZZLANDINGAPPENDED${STEP}ZZ $(date +%FT%T%z)" >> $LOG
}
# the collision watch (Master Mind 2026-09-28): logs any python GPU app that is not this battery's
STOPW="$CHW/gpu_watch_$STEP.stop"
rm -f /d/refcv7_eval_kit/chain/gpu_watch_$STEP.stop
PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/gpu_watch.py" --out "$CHW/gpu_watch_$STEP.jsonl" \
    --stop-file "$STOPW" >> $LOG 2>&1 &
stopwatch() { : > /d/refcv7_eval_kit/chain/gpu_watch_$STEP.stop; }
# the chain-active marker (scheduling rule, README): created now, removed on EVERY exit path, only if ours
marker() { PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/chain_marker.py" "$1" --chain $JOB --pid $$ >> $LOG 2>&1; }
marker create
# the evening yield after every GPU stage that ends after 20:00 (scheduling rule, README)
gyield() { PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/gpu_yield.py" --job $JOB --pid $$ \
    --stage "$1" --log "$CHW/gpu_yields.jsonl" >> $LOG 2>&1; }
trap 'stopwatch; release; marker remove' EXIT
echo "ZZLOCKED${STEP}ZZ $(date +%FT%T%z)" >> $LOG
# ---- 1. the refcv7 T1 battery (G0 first; it STOPS unless G0 = PASS) ------------------------- #
# RESUMABLE (Master Mind 2026-09-28): a G0 artifact for THIS checkpoint md5 is reused; --skip-roll
# reuses only COMPLETE seed dumps (manifest + roll record) and rolls the rest; one retry per stage, so a
# collision or a crash costs one seed, not the run.
battery() {
  G0ARG="--metrics $M"
  if [ -s $OUT/g0.json ] && $PY -c "import json,sys,hashlib
g=json.load(open(r'$OUTW/g0.json',encoding='utf-8'))
h=hashlib.md5(open(r'$CK','rb').read()).hexdigest()
sys.exit(0 if g.get('ckpt_md5')==h and (g.get('verdict') or {}).get('G0') else 1)"; then
    G0ARG="--g0-json $OUTW/g0.json"
  fi
  PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/run_battery_r7.py" --ckpt "$CK" \
      --config "$KITW/ckpt/config.json" $G0ARG --skip-roll --tag $TAG --lock-held \
      --g0-seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
      --yield-job $JOB --yield-pid $$ --yield-log "$CHW/gpu_yields.jsonl" >> $CH/battery_$STEP.log 2>&1
}
battery
NB=$($PY -c "import json; s=json.load(open(r'$OUTW/battery_summary.json',encoding='utf-8')); print(len(s.get('bars') or []))" 2>/dev/null)
G0T=$($PY -c "import json; s=json.load(open(r'$OUTW/battery_summary.json',encoding='utf-8')); print(((s.get('stages') or {}).get('g0') or {}).get('G0','NONE'))" 2>/dev/null)
if [ "$NB" != "3" ] && { [ "$G0T" = "PASS" ] || [ "$G0T" = "NO ARTIFACT" ] || [ "$G0T" = "NONE" ]; }; then
  echo "ZZBATTERYRETRY${STEP}ZZ $(date +%FT%T%z) (bars=$NB after the first attempt)" >> $LOG
  battery
fi
G0=$($PY -c "import json; s=json.load(open(r'$OUTW/battery_summary.json',encoding='utf-8')); print(((s.get('stages') or {}).get('g0') or {}).get('G0','NONE'))" 2>/dev/null)
echo "ZZG0${STEP}ZZ $G0 $(date +%FT%T%z)" >> $LOG
if [ "$G0" != "PASS" ]; then
  echo "ZZCHAINSTOPG0${STEP}ZZ G0=$G0 -- no battery number exists for this checkpoint (SPEC §2)" >> $LOG
  finish
  exit 4
fi
BARS=$($PY -c "import json; s=json.load(open(r'$OUTW/battery_summary.json',encoding='utf-8')); print(len(s.get('bars') or []))")
echo "ZZBATTERYDONE${STEP}ZZ bars=$BARS $(date +%FT%T%z)" >> $LOG
# ---- 2. baselines (pre-stage normally banks them; run here only if missing) ---------------- #
B6=/d/refcv7_eval_kit/baseline_refcv6_38k
# SPEC A4: a banked baseline is REUSED only if its stamp re-verifies; otherwise it is set aside and re-rolled
if [ -s $B6/BASELINE_STAMP.json ]; then
  PYTHONPATH="$EV7/stack;$EV7/taniteval" CUDA_VISIBLE_DEVICES=-1 $PY "$PKGW/code/baseline_stamp.py" verify >> $LOG 2>&1
  if ! $PY -c "import json,sys; sys.exit(0 if json.load(open(r'D:/refcv7_eval_kit/baseline_refcv6_38k/BASELINE_VERIFY.json'))['reuse_admissible'] else 1)"; then
    TS=$(date +%Y%m%dT%H%M%S)
    echo "ZZB6STAMPMISMATCH${STEP}ZZ $TS -- set aside, re-rolled" >> $LOG
    mv $B6/refcv6_step38000 $B6/refcv6_step38000.stale_$TS 2>/dev/null
    mv $B6/perc_refcv6_38k.pkl $B6/perc_refcv6_38k.pkl.stale_$TS 2>/dev/null
    mv $B6/perc_refcv6_38k.json $B6/perc_refcv6_38k.json.stale_$TS 2>/dev/null
  else
    echo "ZZB6STAMPOK${STEP}ZZ (reused, SPEC A4)" >> $LOG
  fi
fi
if [ ! -s $B6/perc_refcv6_38k.json ]; then
  echo "ZZBASEPERC6START${STEP}ZZ $(date +%FT%T%z)" >> $LOG
  REFCV6_REPO="$EV6N" PYTHONPATH="$EV6N/stack;$EV6N/taniteval" $PY "$PKGW/code/perc_dump_refcv6.py" \
    --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step38000_stopped.pt --config D:/refcv6_eval_kit/ckpt_final/config.json \
    --windows-json "$WIN" --out "D:/refcv7_eval_kit/baseline_refcv6_38k/perc_refcv6_38k.npz" \
    >> $CH/perc6_$STEP.log 2>&1
  if [ ! -s $B6/perc_refcv6_38k.json ]; then
    echo "ZZBASEPERC6RETRY${STEP}ZZ $(date +%FT%T%z)" >> $LOG
    REFCV6_REPO="$EV6N" PYTHONPATH="$EV6N/stack;$EV6N/taniteval" $PY "$PKGW/code/perc_dump_refcv6.py" \
      --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step38000_stopped.pt --config D:/refcv6_eval_kit/ckpt_final/config.json \
      --windows-json "$WIN" --out "D:/refcv7_eval_kit/baseline_refcv6_38k/perc_refcv6_38k.npz" \
      >> $CH/perc6_$STEP.log 2>&1
  fi
fi
[ -s $B6/perc_refcv6_38k.json ] && echo "ZZBASEPERC6OK${STEP}ZZ" >> $LOG || echo "ZZBASEPERC6MISSING${STEP}ZZ" >> $LOG
gyield perc_refcv6_38k
# ---- 3. perception: map 10 cm + box, three arms on the SAME windows ------------------------ #
# VOID gates 5 (/3 == /2 inside the old window; the refcv6 hook literal) and 6 (perception seed-
# invariant in this checkpoint's G0) -- CPU, before the pass that relies on them
PYTHONPATH="$EV7/stack;$EV7/taniteval" CUDA_VISIBLE_DEVICES=-1 $PY "$PKGW/code/void_gates_map.py" \
    --g0 "$OUTW/g0.json" --out "$OUTW/void_gates_map.json" >> $LOG 2>&1
[ -s $OUT/void_gates_map.json ] && echo "ZZVOIDMAP${STEP}ZZ $(date +%FT%T%z)" >> $LOG || echo "ZZVOIDMAPNOJSON${STEP}ZZ" >> $LOG
P6=""
[ -s $B6/perc_refcv6_38k.pkl ] && P6="--refcv6-pkl D:/refcv7_eval_kit/baseline_refcv6_38k/perc_refcv6_38k.pkl"
PR=""
[ -s /d/refcv7_eval_kit/prior/prior_train300.npz ] && PR="--prior-npz D:/refcv7_eval_kit/prior/prior_train300.npz"
perc7() {
  PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/perc_pass_r7.py" --ckpt "$CK" --config "$KITW/ckpt/config.json" \
      --windows-json "$WIN" --out "$KITW/perc/$TAG/perc" $P6 $PR >> $CH/perc7_$STEP.log 2>&1
}
[ -s /d/refcv7_eval_kit/perc/$TAG/perc.json ] || perc7
[ -s /d/refcv7_eval_kit/perc/$TAG/perc.json ] || { echo "ZZPERC7RETRY${STEP}ZZ $(date +%FT%T%z)" >> $LOG; perc7; }
gyield perc_refcv7
if [ -s /d/refcv7_eval_kit/perc/$TAG/perc.json ]; then
  echo "ZZPERC7OK${STEP}ZZ $(date +%FT%T%z)" >> $LOG
  PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/perc_score.py" --prefix "$KITW/perc/$TAG/perc" \
      --out "$OUTW/perc_score.json" --milestone > $CH/perc_score_$STEP.log 2>&1
  [ -s $OUT/perc_score.json ] && echo "ZZPERCSCOREOK${STEP}ZZ $(date +%FT%T%z)" >> $LOG || echo "ZZPERCSCORENOJSON${STEP}ZZ" >> $LOG
else
  echo "ZZPERC7NOJSON${STEP}ZZ $(date +%FT%T%z)" >> $LOG
fi
# an interim RESULT + sanitized bank now (the refcv6 rolls below can take hours); finish() redoes both
PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/result_r7.py" --tag $TAG >> $LOG 2>&1
bank
echo "ZZINTERIMRESULT${STEP}ZZ $(date +%FT%T%z)" >> $LOG
# ---- 2b. refcv6@38k S2 rolls for BAR-R7-2 if the pre-stage has not banked them (the long one, last) #
if [ ! -s $B6/refcv6_step38000/dump_s1/manifest.json ]; then
  echo "ZZBASEROLL6START${STEP}ZZ $(date +%FT%T%z)" >> $LOG
  sh $PKG/code/baseline_refcv6_38k.sh --lock-held $JOB $$ >> $LOG 2>&1
  if [ -s $B6/refcv6_step38000/dump_s1/manifest.json ]; then
    PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/run_battery_r7.py" --ckpt "$CK" \
        --config "$KITW/ckpt/config.json" --g0-json "$OUTW/g0.json" --skip-roll --tag $TAG --lock-held \
        > $CH/battery_${STEP}_rebars.log 2>&1
    echo "ZZREBARS${STEP}ZZ $(date +%FT%T%z)" >> $LOG
  fi
fi
# ---- 4. the RESULT (JSON + a short section), from the artifacts, then the sanitized bank --- #
NC=$(grep -c '"COLLISION": true' /d/refcv7_eval_kit/chain/gpu_watch_$STEP.jsonl 2>/dev/null)
echo "ZZCOLLISIONROWS${STEP}ZZ ${NC:-0}" >> $LOG
cp /d/refcv7_eval_kit/chain/gpu_watch_$STEP.jsonl $OUT/gpu_watch.jsonl 2>/dev/null
finish
echo "ZZCHAINEND${STEP}ZZ $(date +%FT%T%z)" >> $LOG
