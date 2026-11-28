#!/bin/sh
# LAST in the queue: once nothing is reading the frozen clone any more, sync it to repo HEAD
# and re-run the RL suite. A sibling stream changed `rewards.py::_collision` to a swept-segment
# test WHILE this panel was running (commit 9765634); the clone was deliberately frozen so s0
# and s1 stayed comparable, which means the veto tests added here have NOT yet been run against
# the current `_collision`. Running them after the last consumer finishes is the only ordering
# that is both comparable and integrated.
set -u
REPO="/c/Users/Admin/refcv4b_repo"
G="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
while ! grep -q "ZZFINAL-COMPLETE\|ZZFINAL-ABORT" /c/Users/Admin/veto_run/final_chain.log 2>/dev/null; do sleep 30; done
echo "ZZSUITE-START-$(date -u +%H:%M:%S)Z-ZZ"
for f in stack/tanitad/rl/rewards.py; do
  i=0
  while [ $i -lt 10 ]; do
    i=$((i+1))
    cp "$G/$f" "$REPO/$f" 2>/dev/null
    a=$(md5sum "$G/$f" 2>/dev/null | cut -d' ' -f1); b=$(md5sum "$REPO/$f" 2>/dev/null | cut -d' ' -f1)
    if [ ${#a} -eq 32 ] && [ "$a" = "$b" ]; then echo "ZZSYNC-OK-$f-$a-ZZ"; break; fi
    sleep 6
  done
done
cd "$REPO/stack" && PYTHONPATH="$REPO/stack" OMP_NUM_THREADS=6 "$PY" -m pytest -q tests/test_rl_*.py tests/test_fan_safety.py tests/test_collision_swept_segment.py 2>&1 | tail -6
echo "ZZSUITE-DONE-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
