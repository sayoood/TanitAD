#!/bin/bash
# PI "start" 2026-09-20: pull nuPlan v1.1 TEST (95.9 GB) then VAL (97.0 GB) archives to D:, resumable.
# Sizes are the S3 Content-Length values measured by HEAD on 2026-09-20; a download only counts as
# done when the byte count matches EXACTLY. Val's download overlaps test's extraction (network vs disk).
# Data lives on D: ONLY. Markers: DOWNLOAD_DONE|DOWNLOAD_FAIL <split>, SPLIT_DONE|SPLIT_FAIL <split>, PULL_ALL_DONE.
set -u
DATA="D:/Projects/TanitAD/data/nuplan"
BASE="https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
VE="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/code/verify_extract_split.py"
mkdir -p "$DATA"

pull() {  # pull <split> <expected_bytes>
  local split="$1" expect="$2" zp="$DATA/nuplan-v1.1_$1.zip" rc sz
  echo "$(date +%T) [$split] curl start (resume from $(stat -c %s "$zp" 2>/dev/null || echo 0) bytes) -> $zp"
  # -C - resumes; --retry with --retry-all-errors re-resumes after drops; speed guard aborts a stalled
  # transfer (<20 KB/s for 120 s) so the retry kicks in. --ssl-no-revoke is the dev-box TLS rule.
  curl --ssl-no-revoke -L -C - --retry 30 --retry-delay 20 --retry-all-errors \
       --speed-time 120 --speed-limit 20000 -sS -o "$zp" "$BASE/nuplan-v1.1_$split.zip" 2>>"$DATA/curl_$split.err"
  rc=$?; sz=$(stat -c %s "$zp" 2>/dev/null || echo 0)
  echo "$(date +%T) [$split] curl rc=$rc size=$sz expect=$expect"
  if [ "$sz" = "$expect" ]; then echo "DOWNLOAD_DONE $split"; return 0; fi
  echo "DOWNLOAD_FAIL $split (rc=$rc size=$sz)"; return 1
}

TPID=""
if pull test 95919476643; then
  "$PY" "$VE" test "$DATA/nuplan-v1.1_test.zip" 95919476643 > "$DATA/extract_test.log" 2>&1 &
  TPID=$!
  echo "$(date +%T) [test] extraction running in background (pid $TPID) while val downloads"
fi
if pull val 96957279835; then
  "$PY" "$VE" val "$DATA/nuplan-v1.1_val.zip" 96957279835 2>&1
fi
if [ -n "$TPID" ]; then wait "$TPID"; echo "--- test extraction log ---"; cat "$DATA/extract_test.log"; fi
echo "$(date +%T) PULL_ALL_DONE"
