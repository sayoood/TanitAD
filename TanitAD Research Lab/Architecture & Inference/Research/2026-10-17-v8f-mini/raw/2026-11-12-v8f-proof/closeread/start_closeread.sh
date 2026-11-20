#!/usr/bin/env bash
# start_closeread.sh -- pod: the self-test of the fit tool in the pod environment, then the run of card ADD77 (pod_closeread_run.sh). No delete, no kill.
# A self-test that does not end with its PASS line writes SELFTEST_FAIL.txt and nothing of the card runs.
set -u
W=/workspace/v8proof/closeread
PY=/workspace/venv/bin/python
cd "$W" || exit 2
[ -e selftest_pod.log ] && { echo "REFUSED: selftest_pod.log exists (a start was made)"; exit 2; }
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 timeout 3600 "$PY" tools/closeread_fit.py --selftest > selftest_pod.log 2>&1
if tail -n 1 selftest_pod.log | grep -q "ZZ CLOSEREAD rc 0 controls_all_pass True"; then
  bash "$W/pod_closeread_run.sh" > run_closeread.nohup 2>&1
else
  date -u +%Y-%m-%dT%H:%M:%SZ > SELFTEST_FAIL.txt
fi
