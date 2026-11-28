#!/bin/bash
# Card PREREG_V8F_PROOF_ADD64: read of record of one fraction run under note N4 of
# card ADD45 (reader stage only, tolerance of control k2 1e-3 through the wrapper
# around the unchanged reader), then pull of its three small files to the dev box.
# The frozen reader's own output of the run (folder read/) is left as it is.
# usage: FRAC_DIR=quarter1 FRAC_N=1143 bash read_n4_frac.sh
# last line: ZZ READ-N4-FRAC <dir> DONE | STOP <reason>
set -u
FRAC_DIR="${FRAC_DIR:?}"
FRAC_N="${FRAC_N:?}"
PSSH=/c/Users/Admin/v8pod/pssh.sh
H=/workspace/v8proof/dscale_lead_frac/$FRAC_DIR
OUT=$H/out
L=/f/TanitAD-scratch/v8proof/dscale_lead/fracscale/$FRAC_DIR
mkdir -p "$L"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
pod() { timeout "${2:-120}" bash "$PSSH" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }

echo "[$(ts)] read_n4_frac $FRAC_DIR expect $FRAC_N clips"
# the run must have ended and written its states
if [ "$(pod "test -f $OUT/DSCALE_LEAD_DONE.txt && echo yes" | tail -n 1)" != "yes" ]; then
  echo "ZZ READ-N4-FRAC $FRAC_DIR STOP no done marker of the run"; exit 2
fi
# memory rule of the pod: 20 GB free before adding a process
FREE=$(pod 'M=$(cat /sys/fs/cgroup/memory.max); A=$(awk "/^anon /{print \$2}" /sys/fs/cgroup/memory.stat); echo $(( (M-A)/1073741824 ))' | tail -n 1)
echo "[$(ts)] pod free memory GB: $FREE"
case "$FREE" in ''|*[!0-9]*) echo "ZZ READ-N4-FRAC $FRAC_DIR STOP memory probe gave no number"; exit 2;; esac
if [ "$FREE" -lt 20 ]; then echo "ZZ READ-N4-FRAC $FRAC_DIR STOP less than 20 GB free"; exit 2; fi
# never read twice into the same folder
if [ "$(pod "test -e $H/read_n4 && echo yes" | tail -n 1)" = "yes" ]; then
  echo "[$(ts)] $H/read_n4 exists already: pull only"
else
  pod "cd /workspace/v8proof/dscale_lead && nohup setsid bash -c 'CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /workspace/venv/bin/python /workspace/v8proof/dscale_lead/dscale_lead_read_n4.py --k2-tol 1e-3 --in-dir $OUT/states --run-dir $OUT --out-dir $H/read_n4 --expect-corpus-clips $FRAC_N --done-marker $OUT/DSCALE_LEAD_DONE.txt > $H/read_n4.out 2>&1; echo \$? > $H/read_n4.rc' > /dev/null 2>&1 < /dev/null & sleep 2; echo started" > /dev/null
  echo "[$(ts)] reader started"
fi
# wait for the exit code file (the script polls, not a model turn)
for i in $(seq 1 240); do
  RC=$(pod "cat $H/read_n4.rc 2>/dev/null" | tail -n 1)
  [ -n "$RC" ] && break
  sleep 30
done
echo "[$(ts)] reader exit code: ${RC:-none}"
if [ -z "${RC:-}" ]; then echo "ZZ READ-N4-FRAC $FRAC_DIR STOP no exit code after 2 h"; exit 2; fi
pull() { # pod path, local name
  bash "$PSSH" "cat $1" 2>/dev/null | sed '/hf_[A-Za-z0-9]\{20,\}/d' | tr -d '\r' > "$L/$2"
  R=$(pod "sha256sum $1" | awk '{print $1}' | tail -n 1)
  S=$(sha256sum "$L/$2" | awk '{print $1}')
  if [ "$R" = "$S" ]; then echo "pulled $2 sha ${S:0:16} same"; else echo "pulled $2 sha DIFFERS pod ${R:0:16} local ${S:0:16}"; BAD=1; fi
}
BAD=0
pull "$H/read_n4.out" read_n4.out
if [ "$RC" = "0" ] || [ "$RC" = "2" ]; then
  pull "$H/read_n4/RESULT_DSCALE_LEAD.json" read_n4_RESULT_DSCALE_LEAD.json
  pull "$H/read_n4/VERDICT_DSCALE_LEAD.md" read_n4_VERDICT_DSCALE_LEAD.md
fi
if [ "$BAD" != "0" ]; then echo "ZZ READ-N4-FRAC $FRAC_DIR STOP a pulled file differs"; exit 2; fi
if [ "$RC" != "0" ]; then echo "ZZ READ-N4-FRAC $FRAC_DIR STOP reader exit code $RC"; exit 2; fi
echo "ZZ READ-N4-FRAC $FRAC_DIR DONE"
