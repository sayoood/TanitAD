#!/usr/bin/env bash
# wait_fracscale.sh -- dev-box waiter of card PREREG_V8F_PROOF_ADD64, bash only (no Python on the dev box). For FRAC_DIR (quarter1 or eighth1) it waits for the pod
# marker FRACSCALE_DONE.txt (or _STOP) in /workspace/v8proof/dscale_lead_frac/<FRAC_DIR>, then brings the small result files back as text into
# dscale_lead/fracscale/<FRAC_DIR>/. Last line: ZZ WAIT-FRACSCALE <FRAC_DIR> DONE or STOP. The waiter of card ADD60 (wait_halfscale.sh) with other names.
HD=${FRAC_DIR:?FRAC_DIR}
L=/f/TanitAD-scratch/v8proof/dscale_lead/fracscale/$HD; P=/c/Users/Admin/v8pod/pssh.sh
R=/workspace/v8proof/dscale_lead_frac/$HD
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "${PODT:-60}" bash "$P" "$1" 2>/dev/null | sed '/hf_[A-Za-z0-9]\{20,\}/d' | tr -d '\r'; }
mkdir -p "$L"
k=0; st=
while :; do
  s=$(pod "ls $R/FRACSCALE_DONE.txt $R/FRACSCALE_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '\n' ' '")
  case "$s" in *STOP*) st=STOP; break;; *DONE*) st=DONE; break;; esac
  k=$((k+1)); [ "$k" -gt 960 ] && { echo "ZZ WAIT-FRACSCALE $HD STOP not ended after 16 h"; exit 3; }
  sleep 60
done
echo "[$(ts)] pod marker: $st $(pod "cat $R/FRACSCALE_$st.txt | cut -c1-240")"
pull() { # remote file, local name
  pod "cat $R/$1" > "$L/$2"
  a=$(pod "sha256sum $R/$1 | cut -c1-64"); b=$(sha256sum "$L/$2" | cut -c1-64)
  [ "$a" = "$b" ] && echo "[$(ts)] pulled $2 sha256 ${a:0:16}" || echo "[$(ts)] PULL-MISMATCH $2"
}
for f in fracscale.log frac.txt precheck.out cpucheck.out smoke.out run.out extract.out checklog.out read.out; do pod "[ -f $R/$f ] && echo yes" | grep -q yes && pull "$f" "$f"; done
for f in out/precheck/PRECHECK.json out/precheck/PRECHECK_PLANTED.json out/precheck/CLOCK.json out/precheck/CLOCK_PLANTED.json out/precheck/EVALSET.json out/precheck/EVALSET_PLANTED.json out/SMOKE_ESTIMATE.json out/DSCALE_LEAD_DONE.txt out/DSCALE_LEAD_STOP.txt out/dscale_lead.log; do
  pod "[ -f $R/$f ] && echo yes" | grep -q yes && pull "$f" "$(basename $f)"
done
for f in $(pod "ls $R/read 2>/dev/null | tr '\n' ' '"); do pull "read/$f" "read_$f"; done
for f in $(pod "ls $R/out/checklog 2>/dev/null | tr '\n' ' '"); do pull "out/checklog/$f" "checklog_$f"; done
echo "ZZ WAIT-FRACSCALE $HD $st $(ts)"
