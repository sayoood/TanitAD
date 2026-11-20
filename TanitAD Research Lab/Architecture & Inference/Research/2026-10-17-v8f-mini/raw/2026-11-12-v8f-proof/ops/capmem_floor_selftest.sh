#!/usr/bin/env bash
# capmem_floor_selftest.sh -- known-value self-test of ops/capmem.py with the shared dev-box floor (CAPMEM_FLOOR_MB, default 5500).
# Five arms, each with the value it must read; last line: ZZ CAPMEM-FLOOR-SELFTEST PASS | FAIL <arms>
#   a1 small job, limit 300, default floor            -> rc 0, stdout "small ok 6553600"
#   a2 big job (700 MB), limit 300                    -> rc not 0 (the cap acts), no "big ok"
#   a3 small job, floor planted above any free memory -> rc 95, stderr names the planted floor
#   a4 small job, floor 0                             -> rc 0 (the floor is the only reason a3 stopped)
#   a5 small job, floor 0, reserve planted too large  -> rc 95 (the older gate still acts by itself)
set -u
O=/f/TanitAD-scratch/v8proof/ops; PYX=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONDONTWRITEBYTECODE=1 CAPMEM_LOG=F:/TanitAD-scratch/v8proof/ops/capmem_floor_selftest_peaks.log
# one BLAS thread: with the default thread count the import of numpy alone asks for more than 300 MB of thread buffers (first run of this script, 2026-11-20: a1 and a4 ended rc 1 on that)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bad=""
run() { "$PYX" F:/TanitAD-scratch/v8proof/ops/capmem.py "$@"; }
o=$(run 300 "$O/capmem_selftest.py" small 2> "$O/fs_a1.err"); rc=$?
[ "$rc" = 0 ] && [ "$(echo "$o" | tr -d '\r')" = "small ok 6553600" ] && echo "a1 ok" || { echo "a1 BAD rc $rc out $o"; bad="$bad a1"; }
o=$(run 300 "$O/capmem_selftest.py" big 2> "$O/fs_a2.err"); rc=$?
[ "$rc" != 0 ] && ! echo "$o" | grep -q 'big ok' && echo "a2 ok (rc $rc, $(grep -c MemoryError "$O/fs_a2.err") MemoryError line)" || { echo "a2 BAD rc $rc out $o"; bad="$bad a2"; }
o=$(CAPMEM_FLOOR_MB=99999999 CAPMEM_WAIT_S=35 run 300 "$O/capmem_selftest.py" small 2> "$O/fs_a3.err"); rc=$?
[ "$rc" = 95 ] && grep -q 'ZZ CAPMEM STOP' "$O/fs_a3.err" && grep -q '99999999' "$O/fs_a3.err" && [ -z "$o" ] && echo "a3 ok (rc 95)" || { echo "a3 BAD rc $rc out $o"; bad="$bad a3"; }
o=$(CAPMEM_FLOOR_MB=0 run 300 "$O/capmem_selftest.py" small 2> "$O/fs_a4.err"); rc=$?
[ "$rc" = 0 ] && [ "$(echo "$o" | tr -d '\r')" = "small ok 6553600" ] && echo "a4 ok" || { echo "a4 BAD rc $rc out $o"; bad="$bad a4"; }
o=$(CAPMEM_FLOOR_MB=0 CAPMEM_RESERVE_MB=99999999 CAPMEM_WAIT_S=35 run 300 "$O/capmem_selftest.py" small 2> "$O/fs_a5.err"); rc=$?
[ "$rc" = 95 ] && grep -q 'ZZ CAPMEM STOP' "$O/fs_a5.err" && [ -z "$o" ] && echo "a5 ok (rc 95)" || { echo "a5 BAD rc $rc out $o"; bad="$bad a5"; }
[ -z "$bad" ] && echo "ZZ CAPMEM-FLOOR-SELFTEST PASS" || echo "ZZ CAPMEM-FLOOR-SELFTEST FAIL$bad"
