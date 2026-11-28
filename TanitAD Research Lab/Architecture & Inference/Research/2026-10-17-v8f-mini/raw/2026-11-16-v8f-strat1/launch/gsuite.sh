#!/bin/bash
# G-SUITE on the pod: runs stack/tests of ONE tree in N shards (by test file, size-balanced), junit xml per shard.
# usage: gsuite.sh <tree: base|new> <nshards>
set -u
TREE=$1; N=${2:-8}
R=/root/strat1/$TREE; J=/root/strat1/junit/$TREE; T=/root/strat1/tmp/$TREE
mkdir -p "$J" "$T"
cd "$R" || exit 2
export PYTHONPATH="$R/stack:$R" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
/root/venv/bin/python - <<EOF || { echo "ZZ GSUITE $TREE FAIL ZZ import"; exit 3; }
import tanitad, os
assert os.path.realpath(tanitad.__file__).startswith("$R/stack/"), tanitad.__file__
print("tanitad at", tanitad.__file__)
EOF
# size-balanced assignment of test files to shards
/root/venv/bin/python - <<EOF
import glob, os
fs = sorted(glob.glob("$R/stack/tests/test_*.py"), key=os.path.getsize, reverse=True)
load = [0]*$N; sh = [[] for _ in range($N)]
for f in fs:
    k = load.index(min(load)); sh[k].append(f); load[k] += os.path.getsize(f) + 20000
for k, l in enumerate(sh):
    open("$J/shard%d.files" % k, "w").write("\n".join(l) + "\n")
print(len(fs), "files in", $N, "shards")
EOF
pids=()
for k in $(seq 0 $((N-1))); do
  (
    mkdir -p "$T/s$k"
    TMPDIR="$T/s$k" TMP="$T/s$k" TEMP="$T/s$k" timeout 4h /root/venv/bin/python -m pytest -q -p no:cacheprovider \
      --basetemp="$T/s$k/bt" --junitxml="$J/shard$k.xml" $(cat "$J/shard$k.files" | tr '\n' ' ') > "$J/shard$k.log" 2>&1
    echo "rc=$?" >> "$J/shard$k.log"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "ZZ GSUITE $TREE DONE ZZ"
