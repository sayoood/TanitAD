#!/usr/bin/env bash
# Dev box: G-SUITE end to end on the TIP (ab436ee; candidate == baseline, the gate's own files are
# not in any commit yet) -- clean git-archive trees, pytest over stack/taniteval/tools, the
# skip-not-a-pass rule, and the tip's guard_mutation_audit. Behind the brief's 8 GB RAM floor.
set -u
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
L=C:/Users/Admin/lg0926
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' PYTHONIOENCODING=utf-8
T=ab436eefa89f34eee139e092c595839e785eab7c
until grep -q "ZZUNIT-DONE" "$L/g/unit_cab/pytest_launch_gate.log" 2>/dev/null; do sleep 60; done
echo "ZZQ-suite-$(date -u +%H%M%S)ZZ"
"$PY" "$L/cab/stack/scripts/launch_gate.py" run --profile refcv6 --checks G-SUITE \
  --tree "$L/tab" --commit "$T" --argv-file "$L/inputs/argv_refcv6_r101_s0.json" \
  --out-dir "$L/g/suite_ab436ee" --git-dir C:/Users/Admin/tanitad-push/.git --baseline "$T" \
  --suite-work C:/lgs --cpu-only --omp 4 --min-free-gb 8 --ram-wait-s 21600 \
  --key-file "$L/keys/rehearsal.key" > "$L/g/suite_ab436ee.out" 2>&1
echo "ZZQ-done-$(date -u +%H%M%S)ZZ"
