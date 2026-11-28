#!/bin/bash
# Card ADD45 (DSCALE-LEAD), union run: read of record under notes N4 and N5 (reader stage only, tolerance of control k2 1e-3 through the wrappers around the unchanged reader).
# The GPU stages ran on the pod of the other session; their output folder came back to the dev box as dscale_lead/back/add45_out2_gpu_nockpt.tgz (out2 without the checkpoints).
# This script puts that archive on the pod of this fork (hash compare), unpacks it into a new folder, runs the reader there on the CPU and pulls its three small files.
# No rm, no mv, no kill. Never reads twice into one folder. Last line: ZZ READ-N5-ADD45 DONE | STOP <reason>
set -u
PSSH=/c/Users/Admin/v8pod/pssh.sh
TGZ=/f/TanitAD-scratch/v8proof/dscale_lead/back/add45_out2_gpu_nockpt.tgz
WANT=0a896f60972931d66be96ef53b35306e9cf3ffba407ebc7b2c675edf2f54e082
H=/workspace/v8proof/dscale_lead_add45
OUT=$H/out2
L=/f/TanitAD-scratch/v8proof/dscale_lead/add45_union
mkdir -p "$L"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
pod() { timeout "${2:-120}" bash "$PSSH" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
echo "[$(ts)] read_n5_add45 start"
[ "$(sha256sum "$TGZ" | cut -c1-64)" = "$WANT" ] || { echo "ZZ READ-N5-ADD45 STOP local archive hash differs"; exit 2; }
if [ "$(pod "test -e $H/read_n5 && echo yes" | tail -n 1)" = "yes" ]; then echo "ZZ READ-N5-ADD45 STOP $H/read_n5 exists already"; exit 2; fi
if [ "$(pod "test -f $OUT/DSCALE_LEAD_DONE.txt && echo yes" | tail -n 1)" != "yes" ]; then
  pod "mkdir -p $H" 60 > /dev/null
  timeout 3000 bash "$PSSH" "cat > $H/add45_out2_gpu_nockpt.tgz" < "$TGZ"
  R=$(pod "sha256sum $H/add45_out2_gpu_nockpt.tgz | cut -c1-64" 300 | tail -n 1)
  echo "[$(ts)] archive on the pod sha ${R:0:16}"
  [ "$R" = "$WANT" ] || { echo "ZZ READ-N5-ADD45 STOP archive hash on the pod differs"; exit 2; }
  pod "cd $H && tar -xzf add45_out2_gpu_nockpt.tgz && ls out2/states | tr '\n' ' '" 900
fi
if [ "$(pod "test -f $OUT/DSCALE_LEAD_DONE.txt && echo yes" | tail -n 1)" != "yes" ]; then echo "ZZ READ-N5-ADD45 STOP no done marker of the run after the unpack"; exit 2; fi
FREE=$(pod 'M=$(cat /sys/fs/cgroup/memory.max); A=$(awk "/^anon /{print \$2}" /sys/fs/cgroup/memory.stat); echo $(( (M-A)/1073741824 ))' | tail -n 1)
echo "[$(ts)] pod free memory GB: $FREE"
case "$FREE" in ''|*[!0-9]*) echo "ZZ READ-N5-ADD45 STOP memory probe gave no number"; exit 2;; esac
if [ "$FREE" -lt 20 ]; then echo "ZZ READ-N5-ADD45 STOP less than 20 GB free"; exit 2; fi
pod "sha256sum /workspace/v8proof/dscale_lead/dscale_lead_read_n5.py /workspace/v8proof/dscale_lead/dscale_lead_read_n4.py /workspace/v8proof/dscale_lead/dscale_lead_read.py | cut -c1-16,65-"
pod "cd /workspace/v8proof/dscale_lead && (nohup setsid bash -c 'CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/venv/bin/python /workspace/v8proof/dscale_lead/dscale_lead_read_n5.py --k2-tol 1e-3 --in-dir $OUT/states --run-dir $OUT --out-dir $H/read_n5 --expect-corpus-clips 4512 --done-marker $OUT/DSCALE_LEAD_DONE.txt > $H/read_n5.out 2>&1; echo \$? > $H/read_n5.rc' > /dev/null 2>&1 < /dev/null &); echo started" 60
echo "[$(ts)] reader started"
RC=
for i in $(seq 1 240); do
  RC=$(pod "cat $H/read_n5.rc 2>/dev/null" | tail -n 1)
  [ -n "$RC" ] && break
  sleep 30
done
echo "[$(ts)] reader exit code: ${RC:-none}"
if [ -z "${RC:-}" ]; then echo "ZZ READ-N5-ADD45 STOP no exit code after 2 h"; exit 2; fi
BAD=0
pull() { # pod path, local name
  bash "$PSSH" "cat $1" 2>/dev/null | sed '/hf_[A-Za-z0-9]\{20,\}/d' | tr -d '\r' > "$L/$2"
  R=$(pod "sha256sum $1" | awk '{print $1}' | tail -n 1)
  S=$(sha256sum "$L/$2" | awk '{print $1}')
  if [ "$R" = "$S" ]; then echo "pulled $2 sha ${S:0:16} same"; else echo "pulled $2 sha DIFFERS pod ${R:0:16} local ${S:0:16}"; BAD=1; fi
}
pull "$H/read_n5.out" read_n5.out
if [ "$RC" = "0" ] || [ "$RC" = "2" ]; then
  pull "$H/read_n5/RESULT_DSCALE_LEAD.json" read_n5_RESULT_DSCALE_LEAD.json
  pull "$H/read_n5/VERDICT_DSCALE_LEAD.md" read_n5_VERDICT_DSCALE_LEAD.md
fi
if [ "$BAD" != "0" ]; then echo "ZZ READ-N5-ADD45 STOP a pulled file differs"; exit 2; fi
if [ "$RC" != "0" ]; then echo "ZZ READ-N5-ADD45 STOP reader exit code $RC"; exit 2; fi
echo "ZZ READ-N5-ADD45 DONE"
