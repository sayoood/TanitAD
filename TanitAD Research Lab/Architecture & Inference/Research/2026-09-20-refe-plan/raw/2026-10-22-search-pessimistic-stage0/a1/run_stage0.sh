#!/usr/bin/env bash
# SEARCH-PESSIMISTIC Stage 0, AMENDMENT A1: the REAL read, in one command (git-bash twin of run_stage0.cmd).  CPU only.  Writes raw/RESULT_STAGE0_A1.json.
set -u
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
PY="${TANITAD_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/PREREG_STAGE0_A1.md" ] || { echo "ABORT: no sealed PREREG_STAGE0_A1.md next to this script"; exit 2; }
[ -f "$HERE/PREREG_STAGE0_A1.sha256" ] || { echo "ABORT: no PREREG_STAGE0_A1.sha256 next to this script"; exit 2; }
SEAL="$(awk 'NR==1{print $1}' "$HERE/PREREG_STAGE0_A1.sha256")"
mkdir -p "$HERE/raw"
"$PY" -m pytest "$HERE/code/test_stage0.py" -q --no-header -p no:cacheprovider > "$HERE/raw/pytest_before_read_A1.txt" 2>&1 || { echo "ABORT: the test suite is red, see raw/pytest_before_read_A1.txt"; exit 3; }
"$PY" "$HERE/code/stage0.py" run --prereg "$HERE/PREREG_STAGE0_A1.md" --prereg-sha256 "$SEAL" --out "$HERE/raw/RESULT_STAGE0_A1.json"
