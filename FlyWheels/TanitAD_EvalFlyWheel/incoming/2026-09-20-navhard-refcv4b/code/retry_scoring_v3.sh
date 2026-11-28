#!/usr/bin/env bash
# W7 — RETRY v3 (2026-09-21): v2 + `--reuse-echo "$ECHO_SRC"`. Written as a NEW FILE while v2 was live:
# bash reads a running script lazily by byte offset, so editing v2 in place could make the live shell
# execute garbage (CLAUDE.md, the supervisor-script rule). ⭐ WHY: on attempt 2 (f99b29) ECHO PASSED
# 5,912/5,912 in 4,663 s. If A1 then dies, v2 would re-score ECHO from zero — 78 more minutes of
# exposure to the same RAM killer. ECHO is a deterministic, checkpoint-independent seam (a freshly
# rebuilt one is BYTE-EQUAL to f99b29's, 5,912 tokens), so it is adopted under the floor identity check.
# W7 — RAM-GATED RETRY of the navhard scoring. v2, 2026-09-21.
#
# ⛔⛔ v1 DECLARED SUCCESS ON AN ALL-FAILED RUN, AND THIS VERSION EXISTS BECAUSE OF IT. v1's line 73
# was `[ -f "$RD/summary.json" ]` — EXISTENCE. The suite writes summary.json for a FAILED run too, so
# v1 logged `BENCH_STATUS=FAILED` and then `SUCCESS` on the very next line, wrote
# SUCCESSFUL_RUN_DIR.txt pointing at the all-FAILED run d3c2b2, and exited with 5 of 6 attempts
# unused. It was the finisher's self-trap from the evening before, surviving in the sibling script —
# the one place I had not propagated the fix. (That marker is kept, renamed, as
# raw/FALSE_SUCCESS_d3c2b2.txt: it is evidence, and anything that read it would inherit a total
# failure as the good run.)
# ⇒ success is now judged by ONE shared judge on CONTENT (code/judge_run.py: every required arm
#   `status: OK`, counts printed), mutation-tested against the REAL d3c2b2 summary (FAILED) and a
#   fabricated all-OK copy (SUCCESS) — raw/judge_mutation_test.json.
#
# ⭐ AND IT NO LONGER RE-SCORES THE FLOORS. Both previous attempts died at ~80 % of their FIRST scored
# arm (STOP 4,915/5,912; CV 4,294/5,462 stage-two) on E1_RAM_GUARD_ABORT with 948–1,750 MB available
# while OTHER sessions' jobs held the box. CV + STOP were already scored COMPLETELY on the identical
# tokens in 06e257. `--reuse-floors` adopts them after an identity check (devkit sha, patch set, metric
# cache, agent-input export, token set BY VALUE with stage labels, STOP seam byte-equal) that REFUSES
# on any mismatch — probed on the real artifacts, 6/6 mutations RED (raw/floor_reuse_probe.json).
# Only ECHO + A1 are scored: the exposure window is halved. `--reuse-seams` keeps A1's inference at 0 s.
#
# ⛔ The RAM guard floor stays 3,000 / 2,000 MB — it prevents a kernel OOM. Fix the run, not the gate.
# ⛔ Never reads $? through a pipe; every status comes from a FILE or a literal JUDGE= line.
set -u
REPO="D:/Projects/TanitAD"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PKG="$REPO/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-20-navhard-refcv4b"
SEAMS="${SEAMS:-$REPO/taniteval/results/bench/navsim_v2/navhard_two_stage/20260920T144140Z-navsim_v2-refcv4b_b1_v72_40k-b11027/raw/model}"
FLOORS="${FLOORS:-$REPO/taniteval/results/bench/navsim_v2/navhard_two_stage/20260920T082848Z-navsim_v2-none-06e257}"
BANK="C:/Users/Admin/tanitad-caches/navsim-frames-navhard-20260920"
CKPT="D:/Projects/TanitAD-artifacts/refcv4b_final/ckpt_40284_FINAL.pt"
MD5="99b573e8277d94a5e3bfbf630cb4d751"
LOG="$PKG/raw/retry_v3.log"
NEED_GB="${NEED_GB:-6.0}"
NEED_N="${NEED_N:-5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"
BACKOFF_S="${BACKOFF_S:-1200}"
REQUIRE="CV,STOP,ECHO,A1"

say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }
avail_gb() { "$PY" -c "import psutil;print('%.2f'%(psutil.virtual_memory().available/2**30))" 2>/dev/null || echo 0; }

wait_for_headroom() {
  local ok=0
  while [ "$ok" -lt "$NEED_N" ]; do
    local g; g=$(avail_gb)
    if [ "$("$PY" -c "print(1 if float('$g')>=float('$NEED_GB') else 0)")" = "1" ]; then
      ok=$((ok+1))
    else
      [ "$ok" -gt 0 ] && say "headroom lost at ${g} GB (had $ok/$NEED_N) — restarting the count"
      ok=0
    fi
    [ "$ok" -lt "$NEED_N" ] && sleep 60
  done
  say "headroom SUSTAINED: $NEED_N consecutive samples >= ${NEED_GB} GB (now $(avail_gb) GB)"
}

say "retry v3 armed (reuse-echo ${ECHO_SRC:-<none>}): reuse-seams $SEAMS · reuse-floors $FLOORS · require $REQUIRE"
[ -f "$SEAMS/A1.npz" ] || { say "REFUSED: no banked A1 seam at $SEAMS/A1.npz"; exit 2; }
[ -f "$FLOORS/summary.json" ] || { say "REFUSED: no floors run at $FLOORS"; exit 2; }

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  say "attempt $attempt/$MAX_ATTEMPTS — waiting for sustained headroom (>= ${NEED_GB} GB x $NEED_N)"
  wait_for_headroom
  OUT="$PKG/raw/retry_v3_attempt${attempt}.log"
  say "attempt $attempt: launching"
  # ⛔ cwd MUST be taniteval/: from the repo root `taniteval` resolves to the OUTER namespace dir and
  # `-m taniteval.bench` dies with "No module named taniteval.bench" (v1's tombstone step did exactly that).
  ( cd "$REPO/taniteval" && env PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=16 \
      TANITEVAL_STACK_OVERRIDE="$REPO/stack" "$PY" -m taniteval.bench navsim_v2 \
      --ckpt "$CKPT" --split navhard_two_stage --arms A1 --device auto \
      --registry-key refcv4b-b1-v72-40k --ckpt-md5 "$MD5" \
      --frame-bank "$BANK" --cpu-threads 16 \
      --reuse-seams "$SEAMS" --reuse-floors "$FLOORS" ${ECHO_SRC:+--reuse-echo "$ECHO_SRC"} > "$OUT" 2>&1 )
  RD=$(grep '^BENCH_RUN_DIR=' "$OUT" | tail -1 | cut -d= -f2-)
  ST=$(grep '^BENCH_STATUS=' "$OUT" | tail -1 | cut -d= -f2-)
  say "attempt $attempt: BENCH_STATUS=${ST:-<absent>} dir=${RD:-<absent>}"
  if [ -n "$RD" ]; then
    "$PY" "$PKG/code/judge_run.py" "$RD" --require "$REQUIRE" > "$PKG/raw/retry_v3_attempt${attempt}.judge.txt" 2>&1
    J=$(grep '^JUDGE=' "$PKG/raw/retry_v3_attempt${attempt}.judge.txt" | tail -1)
    say "attempt $attempt: $J — $(head -1 "$PKG/raw/retry_v3_attempt${attempt}.judge.txt" | cut -c1-240)"
    if [ "$J" = "JUDGE=SUCCESS" ]; then
      echo "$RD" > "$PKG/raw/SUCCESSFUL_RUN_DIR.txt"
      say "SUCCESS (judged on CONTENT): every required arm is status OK in $RD/summary.json"
      exit 0
    fi
  fi
  say "attempt $attempt FAILED (content judge); backing off ${BACKOFF_S}s"
  attempt=$((attempt+1))
  [ "$attempt" -le "$MAX_ATTEMPTS" ] && sleep "$BACKOFF_S"
done
say "GIVING UP after $MAX_ATTEMPTS attempts"
exit 1
