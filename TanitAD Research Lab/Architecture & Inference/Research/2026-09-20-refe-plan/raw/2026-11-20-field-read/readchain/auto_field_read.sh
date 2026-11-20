#!/bin/bash
# MM 2026-11-20: run the F-vs-C read chain without MM latency.
#   wait for C's seam chain (markers_c/chain.DONE; chain.FAILED -> stop) -> detached WMI launch of run_field_read_chain.cmd pre-C (the C-vs-C rehearsal)
#   -> wait for TERMINAL_pre-C.txt -> PREC_DONE: detached launch of `full` (it polls up to 48 h for F's chain.DONE itself); anything else: stop, say why.
# Prints exactly one ZZAUTOR_* line per terminal state.  usage: auto_field_read.sh
set -u
FW=E:/Projects/TanitAD-artifacts/field_read; W=E:/Projects/TanitAD-artifacts/field_verdict
HOLD='E:\Projects\TanitAD-artifacts\scorer_gym_20261107\field_read\readchain'
ts() { date -u +%FT%H:%MZ; }
launch() {   # <mode>
  powershell -NoProfile -Command "\$hold = '$HOLD'; \$cmd = 'cmd.exe /c \"\"' + \$hold + '\run_field_read_chain.cmd\" $1\"'; \$si = (New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }); \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = \$cmd; CurrentDirectory = \$hold; ProcessStartupInformation = \$si }; 'launched: ReturnValue=' + \$r.ReturnValue + ' ProcessId=' + \$r.ProcessId"
}
for i in $(seq 1 720); do
  [ -e $FW/markers_c/chain.DONE ] && break
  [ -e $FW/markers_c/chain.FAILED ] && { echo "ZZAUTOR_C_FAILED $(ts) $(cat $FW/markers_c/chain.FAILED)"; exit 0; }
  sleep 60
done
[ -e $FW/markers_c/chain.DONE ] || { echo "ZZAUTOR_C_TIMEOUT $(ts)"; exit 0; }
gE=$(df -k /e | tail -n 1 | awk '{print int($4/1048576)}')
[ "$gE" -ge 4 ] || { echo "ZZAUTOR_BLOCKED_DISK $(ts) E: ${gE} GB (< 4)"; exit 0; }
[ -e "$W/TERMINAL_pre-C.txt" ] && { echo "ZZAUTOR_REFUSED $(ts) $W/TERMINAL_pre-C.txt already exists: $(cat $W/TERMINAL_pre-C.txt)"; exit 0; }
echo "C seam chain DONE $(ts); launching pre-C: $(launch pre-C)"
for i in $(seq 1 240); do [ -s "$W/TERMINAL_pre-C.txt" ] && break; sleep 60; done
t="$(cat "$W/TERMINAL_pre-C.txt" 2>/dev/null)"
case "$t" in
  *ZZFIELDREAD_PREC_DONE*) ;;
  "") echo "ZZAUTOR_PREC_TIMEOUT $(ts) no TERMINAL_pre-C.txt after 4 h; tail: $(tail -n 2 $W/CHAIN.log 2>/dev/null | tr '\n' ' ' | cut -c1-200)"; exit 0 ;;
  *) echo "ZZAUTOR_PREC_NOT_DONE $(ts) $t"; exit 0 ;;
esac
grep -h "ZZFIELD_READ\|E1 " "$W"/out*/*.txt 2>/dev/null | head -n 3
# PREC_ONLY=1 (MM 2026-11-20): stop after the rehearsal; the MM launches `full` once field_read v3 (the corrected TTC / headway / time-gap
# polarity) is in FR_DIR, so the full chain's INSTRUMENTS.sha256 records the instrument it really runs.
[ "${PREC_ONLY:-0}" = 1 ] && { echo "ZZAUTOR_PREC_PASSED_STOP $(ts) (PREC_ONLY: full NOT launched)"; exit 0; }
echo "pre-C PASSED $(ts); launching full: $(launch full)"
sleep 120
if grep -q "chain start mode=full" "$W/CHAIN.log" 2>/dev/null; then echo "ZZAUTOR_FULL_LAUNCHED $(ts) heartbeat: $(tail -n 1 $W/CHAIN.log | cut -c1-200)"; else echo "ZZAUTOR_FULL_NO_HEARTBEAT $(ts)"; fi
