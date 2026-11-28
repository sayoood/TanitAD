#!/usr/bin/env bash
# podread_stratroll.sh -- pod-side full read of card PREREG_V8F_PROOF_ADD66 (STRATROLL10K). CPU only (CUDA_VISIBLE_DEVICES=-1). Starts no rollout, no training, writes no tick file.
# Usage: bash podread_stratroll.sh [cand] [suffix]
#   cand   : pass --with-cand (the four runs of card ADD65 are admitted only if /workspace/v8proof/podread/out/read_cand/READ_CAND10K.json has controls_all_pass true)
#   suffix : output folder suffix, out/read_stratroll<suffix>/ (default empty). The script REFUSES to overwrite an existing READ_STRATROLL10K.json: give a new suffix.
# Needs: $P/root/integ10k/strat_roll_read.py (pushed by the lead). Output: $P/out/read_stratroll<suffix>/READ_STRATROLL10K.json and $P/podread_stratroll.out (this script's log).
# Last line: ZZ PODREAD-STRATROLL DONE or ZZ PODREAD-STRATROLL STOP <why>.
set -u
P=/workspace/v8proof/podread; W=/workspace/v8proof/integ10k; PY=/workspace/venv/bin/python
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-STRATROLL STOP $*"; exit 3; }
CAND=""; SUF=""
for x in "$@"; do if [ "$x" = "cand" ]; then CAND="--with-cand"; else SUF="$x"; fi; done
OUT=$P/out/read_stratroll$SUF
[ -f "$OUT/READ_STRATROLL10K.json" ] && { echo "ZZ PODREAD-STRATROLL STOP OUTPUT-EXISTS $OUT/READ_STRATROLL10K.json (give a new suffix)"; exit 3; }
trap 'tail -n 130 "$P/podread_stratroll.out" | cut -c1-430' EXIT
{
[ -f "$P/root/integ10k/strat_roll_read.py" ] || stop "NO-READER-FILE"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"
[ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
mkdir -p "$OUT"
"$PY" "$P/root/integ10k/strat_roll_read.py" --w "$W" --out "$OUT" $CAND > "$OUT/read_stratroll10k.log" 2>&1; rc=$?
echo "[$(ts)] reader rc $rc"
tail -n 120 "$OUT/read_stratroll10k.log" | cut -c1-430
[ "$rc" = 0 ] || stop "READER-RC $rc"
echo "ZZ PODREAD-STRATROLL DONE $(ts)"
} > "$P/podread_stratroll.out" 2>&1
