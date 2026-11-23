#!/bin/bash
# Supplementary (environmental): test_p2r_chain.py does os.utime(ckpt, ns=(1, 1.7e18)); E: (exFAT) cannot store a 1970 atime (WinError 87).  Same test, same tree, with os.utime's atime clamped to 1980-01-01 by a runpy shim.  NOT a gating run.
set -u
SHIP=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/ship; O=$SHIP/fs/raw/2026-11-23-fs-gate-1b; X=$SHIP/fs_scratch/y; Q=$X/q
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; LK=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/pytest.lockdir
cd "$X/refe" || exit 2; WT="$(cygpath -w $SHIP/fs_scratch/tmp)"; PP="$(cygpath -w $X/refe);$(cygpath -w $Q/stack);$(cygpath -w $Q)"
"$PY" "$O/wait_commit.py" 5.5 > /dev/null 2>&1
until mkdir "$LK" 2>/dev/null; do sleep 15; done
TMP="$WT" TEMP="$WT" TMPDIR="$WT" PYTHONPATH="$PP" CUDA_VISIBLE_DEVICES=-1 timeout 3600 "$PY" "$O/utime_shim.py" test_p2r_chain.py > "$SHIP/fs_scratch/logs/t_p2r_shim.log" 2>&1; rc=$?
rmdir "$LK"
last="$(tail -n 1 "$SHIP/fs_scratch/logs/t_p2r_shim.log" | tr -d '\r' | cut -c1-140)"
printf 'test_p2r_chain.py\t(utime-shim; supplementary)\t%s\t%s\n' "$rc" "$last" | tee -a "$O/tests_supplementary.tsv"
