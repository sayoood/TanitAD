#!/usr/bin/env bash
# AR-3 detached driver v2. Sequential; ONE pytest process of ours at a time, started only when <= 1 other pytest runs (<= 2 python.exe lines).
# Deselects tests/test_v8f_mini.py::test_five_hand_checked_real_windows everywhere: it opens a file under C:/Users/Admin/qland, which the brief forbids touching.
SP=$1; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
DESEL="--deselect tests/test_v8f_mini.py::test_five_hand_checked_real_windows"
pcount() { powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | ForEach-Object { \$_.CommandLine }" | grep -c "pytest"; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $SP/raw/driver.log; }
waitp() { for i in $(seq 1 480); do c=$(pcount); [ "${c:-99}" -le 2 ] && return 0; sleep 30; done; return 1; }
say "driver2 start; waiting for the live4 gate to end (RAM safety: ~3 GB free)"
for i in $(seq 1 720); do grep -q '^ZZRC' $SP/raw/selfcheck_L_live4.log 2>/dev/null && break; sleep 30; done
say "live4 gate: $(grep '^ZZRC' $SP/raw/selfcheck_L_live4.log)"
export CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
# ---- P1: LoRA / v8f-mini test files = the G-DVB / G-LIVE / G-CKPT / G-EVAL mechanisms (work tree with the package code dirs)
waitp || { say "P1 not started: pytest count stayed > 2 lines"; exit 3; }
say "P1 start (python.exe pytest lines at start: $(pcount))"
cd C:/Users/Admin/ar3_work/stack
PYTHONPATH="C:/Users/Admin/ar3_work/stack;C:/Users/Admin/ar3_work/taniteval" $PY -m pytest -q -p no:cacheprovider --basetemp=C:/Users/Admin/ar3_work/bt1/p1 -rfEsx $DESEL \
  --junitxml=$SP/raw/pytest_lora_subset.xml tests/test_v8f_lora.py tests/test_v8f_lora_shape.py tests/test_v8f_lora_device.py tests/test_v8f_lora_ar2.py tests/test_v8f_mini.py > $SP/raw/pytest_lora_subset.log 2>&1
say "P1 end rc=$?"
# ---- G-SUITE (plain pytest on the CLEAN tree = git archive of the tip: stack, taniteval, tools; same three suites the gate runs)
for S in stack taniteval tools; do
  waitp || { say "G-SUITE $S not started: pytest count stayed > 2 lines"; exit 3; }
  say "G-SUITE $S start (python.exe pytest lines at start: $(pcount))"
  cd C:/Users/Admin/ar3_gt/$S
  D=""; [ "$S" = stack ] && D="$DESEL"
  PYTHONPATH="C:/Users/Admin/ar3_gt/stack;C:/Users/Admin/ar3_gt/taniteval" $PY -m pytest -q -p no:cacheprovider --basetemp=C:/Users/Admin/ar3_work/bt2/$S -rfEs $D \
    --junitxml=$SP/raw/gsuite_$S.xml tests > $SP/raw/gsuite_$S.log 2>&1
  say "G-SUITE $S end rc=$?"
done
echo "ZZDRIVER-DONE" >> $SP/raw/driver.log
