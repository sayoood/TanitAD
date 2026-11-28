#!/usr/bin/env bash
# wait_halfscale.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD60, bash only (no Python on the dev box). Waits for the pod marker HALFSCALE_DONE.txt (or _STOP) in
# /workspace/v8proof/dscale_lead/<HALF_DIR>, then brings the small result files back as text into dscale_lead/halfscale/<HALF_DIR>/. Last line: ZZ WAIT-HALFSCALE DONE or STOP.
HD=${HALF_DIR:-half1}
L=/f/TanitAD-scratch/v8proof/dscale_lead/halfscale/$HD; P=/c/Users/Admin/v8pod/pssh.sh
R=/workspace/v8proof/dscale_lead/$HD
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_[A-Za-z0-9]\{20,\}/d' | tr -d '\r'; }
mkdir -p "$L"
k=0; st=
while :; do
  s=$(pod "ls $R/HALFSCALE_DONE.txt $R/HALFSCALE_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) st=STOP; break;; *DONE*) st=DONE; break;; esac
  k=$((k+1)); [ "$k" -gt 1440 ] && { echo "ZZ WAIT-HALFSCALE STOP not ended after 12 h"; exit 3; }
  sleep 30
done
echo "[$(ts)] pod marker: $st $(pod "cat $R/HALFSCALE_$st.txt | cut -c1-240")"
pull() { # remote file, local name
  pod "cat $R/$1" > "$L/$2"
  a=$(pod "sha256sum $R/$1 | cut -c1-64"); b=$(sha256sum "$L/$2" | cut -c1-64)
  [ "$a" = "$b" ] && echo "[$(ts)] pulled $2 sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $2"
}
for f in halfscale.log half.txt precheck.out cpucheck.out smoke.out run.out extract.out checklog.out read.out; do pod "[ -f $R/$f ] && echo yes" | grep -q yes && pull "$f" "$f"; done
for f in out/precheck/PRECHECK.json out/precheck/PRECHECK_PLANTED.json out/precheck/CLOCK.json out/precheck/CLOCK_PLANTED.json out/precheck/EVALSET.json out/precheck/EVALSET_PLANTED.json out/SMOKE_ESTIMATE.json out/DSCALE_LEAD_DONE.txt out/DSCALE_LEAD_STOP.txt out/dscale_lead.log; do
  pod "[ -f $R/$f ] && echo yes" | grep -q yes && pull "$f" "$(basename $f)"
done
for f in $(pod "ls $R/read 2>/dev/null | tr '\n' ' '"); do pull "read/$f" "read_$f"; done
for f in $(pod "ls $R/out/checklog 2>/dev/null | tr '\n' ' '"); do pull "out/checklog/$f" "checklog_$f"; done
echo "ZZ WAIT-HALFSCALE $st $(ts)"
