#!/usr/bin/env bash
set -u
cd ~
PY=~/venvs/tanitad-edge/bin/python
$PY -c 'import ast; ast.parse(open("/home/nvidia/thor_b1b_fastpath_probe.py").read()); print("SYNTAX OK")' || exit 1
setsid env OMP_NUM_THREADS=6 PYTHONPATH=/usr/lib/python3.12/dist-packages \
    $PY ~/thor_b1b_fastpath_probe.py > ~/thor_b1b.log 2>&1 < /dev/null &
sleep 2
echo "relaunched pid $!"
