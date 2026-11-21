#!/bin/bash
# pod_n6_read.sh -- note N6 of card ADD45 on the pod of this fork, CPU only: (1) self-test of the wrapper dscale_lead_read_n6.py, (2) known value: the HALF run of card ADD60 read through
# the wrapper equals its read under note N5 leaf for leaf apart from the added fields, (3) only if both hold: the read of record of the union run of card ADD45 into a new folder read_n6.
# No rm, no mv, no kill. Never reads twice into one folder. Log: /workspace/v8proof/dscale_lead_add45/n6_read.log; last line ZZ POD-N6 DONE rc <rc> | STOP <why>
set -u
K=/workspace/v8proof/dscale_lead; PY=/workspace/venv/bin/python
HALF=/workspace/v8proof/dscale_lead/half1
H=/workspace/v8proof/dscale_lead_add45; OUT=$H/out2
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "ZZ POD-N6 STOP $*"; exit 3; }
for f in dscale_lead_read.py dscale_lead_read_n4.py dscale_lead_read_n5.py dscale_lead_read_n6.py n6_kv_compare.py; do [ -f "$K/$f" ] || stop "MISSING $f"; done
[ -e "$H/read_n6" ] && stop "EXISTS $H/read_n6"
[ -e "$HALF/read_n6_kv" ] && stop "EXISTS $HALF/read_n6_kv"
[ -f "$HALF/read_n5/RESULT_DSCALE_LEAD.json" ] || stop "NO-HALF-READ-N5"
[ -f "$OUT/DSCALE_LEAD_DONE.txt" ] || stop "NO-DONE-MARKER"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"; [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
sha256sum "$K/dscale_lead_read_n6.py" "$K/dscale_lead_read_n5.py" "$K/dscale_lead_read_n4.py" "$K/dscale_lead_read.py" "$K/n6_kv_compare.py" | cut -c1-16,65-
cd "$K" || stop "NO-KIT-DIR"
"$PY" dscale_lead_read_n6.py --selftest > "$H/n6_selftest.out" 2>&1; st=$?
tail -n 1 "$H/n6_selftest.out" | cut -c1-80
[ "$st" = 0 ] || stop "SELFTEST rc $st"
"$PY" dscale_lead_read_n6.py --k2-tol 1e-3 --in-dir "$HALF/out/states" --run-dir "$HALF/out" --out-dir "$HALF/read_n6_kv" --expect-corpus-clips 2286 --done-marker "$HALF/out/DSCALE_LEAD_DONE.txt" > "$HALF/read_n6_kv.out" 2>&1
echo "[$(ts)] known-value read rc $?"
"$PY" n6_kv_compare.py "$HALF/read_n5/RESULT_DSCALE_LEAD.json" "$HALF/read_n6_kv/RESULT_DSCALE_LEAD.json" > "$H/n6_kv.out" 2>&1; kv=$?
cat "$H/n6_kv.out" | cut -c1-200
[ "$kv" = 0 ] || stop "KNOWN-VALUE-DIFFERS"
"$PY" dscale_lead_read_n6.py --k2-tol 1e-3 --in-dir "$OUT/states" --run-dir "$OUT" --out-dir "$H/read_n6" --expect-corpus-clips 4512 --done-marker "$OUT/DSCALE_LEAD_DONE.txt" > "$H/read_n6.out" 2>&1
rc=$?
echo "$rc" > "$H/read_n6.rc"
tail -n 1 "$H/read_n6.out" | cut -c1-120
echo "ZZ POD-N6 DONE rc $rc $(ts)"
