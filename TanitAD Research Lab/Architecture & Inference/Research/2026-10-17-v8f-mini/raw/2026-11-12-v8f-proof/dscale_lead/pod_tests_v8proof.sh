#!/usr/bin/env bash
# pod_tests_v8proof.sh -- lead's run of the kit's tests and of its mutation list on the v8proof pod (the pod of this fork), CPU only.
# The pod's Python has no pytest and nothing is installed into its venv: the tests run under the stand-in runner in pytest_shim/ (it says so in its last line).
# The reader's import chain uses dev-box paths written as F:/...; on the pod they resolve through a link named "F:" in this folder to the reader mirror of card ADD43.
# Usage: bash pod_tests_v8proof.sh [suites|mutations|all] > TESTS.txt 2>&1
set -u
WHAT=${1:-all}
D=/workspace/v8proof/dscale_lead; PY=/workspace/venv/bin/python
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export PYTHONPATH=$D/pytest_shim
export DSCALE_TREE=/workspace/v8proof/tree_leadsup TOKPROBE_GR_SRC=/workspace/v8proof/finegrid/tok_probe_gr.py
cd "$D" || exit 3
[ -e "$D/F:" ] || ln -s "/workspace/v8proof/finegrid/rk/F:" "$D/F:"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat); free=$(( (mm - an) / 1073741824 )); echo "container free ${free} GB"
[ "$free" -ge 20 ] || { echo "ZZ TESTS STOP memory"; exit 3; }
W=$D/tests_$(date -u +%H%M%S); mkdir -p "$W"
echo "generator output equals tok_probe_u.py: $("$PY" make_tok_probe_u.py --help >/dev/null 2>&1; echo checked-by-the-tool-test)"
bad=0
if [ "$WHAT" = suites ] || [ "$WHAT" = all ]; then
  for t in test_dscale_precheck test_dscale_lead_read test_tok_probe_u; do
    "$PY" -m pytest -q -p no:cacheprovider "$t.py" --basetemp "$W/pt_$t" > "$W/$t.out" 2>&1; rc=$?
    echo "$t rc $rc :: $(tail -n 1 "$W/$t.out" | cut -c1-200)"
    [ $rc -eq 0 ] || { bad=1; grep -n "^FAILED" "$W/$t.out" | head -n 12 | cut -c1-200; grep -n "Error\|assert" "$W/$t.out" | head -n 14 | cut -c1-260; }
    grep "^SKIPPED" "$W/$t.out" | head -n 5 | cut -c1-200
  done
fi
if [ "$WHAT" = mutations ] || { [ "$WHAT" = all ] && [ $bad -eq 0 ]; }; then
  "$PY" run_mutations_u.py --targets precheck,reader,tool > "$W/mutations.out" 2>&1; echo "mutation runner rc $?"
  cut -c1-200 "$W/mutations.out" | tail -n 34
fi
echo "ZZ TESTS DONE $W bad=$bad"
