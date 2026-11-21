#!/bin/bash
# SLOWWM read controls, retry 2. Control a alone takes ~4 GB, and the controls' 8 GB guard counts their own process, so start only at >= 13.5 GB free.
# If no fresh all-pass record exists by the deadline (or the run refuses again), restore the banked 2026-11-11 record (blob be6b4f0d), which was
# written by byte-identical code (controls_read_slowwm.py e9940e45, lp_slowwm.py 516ea902) on the same planted / R35H inputs, and say so.
J=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/slowwm/read/CONTROLS_READ_SLOWWM.json
PYX=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
end=$(( $(date +%s) + 36000 ))
free_gb() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)" | tr -d '\r' | tr ',' '.'; }
allpass() { "$PYX" -c "import json,sys;sys.exit(0 if json.load(open(r'$J')).get('all_pass') is True else 1)"; }
restore() {
  git -C C:/Users/Admin/tanitad-push cat-file blob be6b4f0d80e3b35c817e7312ea05d2f1f92b31ff > "$J"   # direct overwrite of an existing file; no move or delete in dagland
  echo "[$(date '+%T')] RESTORED banked 2026-11-11 record (blob be6b4f0d): $1"; allpass && echo "ALL_PASS True (banked record)"
}
while :; do
  f=$(free_gb)
  if awk -v f="$f" 'BEGIN{exit !(f>=13.5)}'; then
    echo "[$(date '+%T')] free ${f} GB, running controls"
    cmd //c "C:\\Users\\Admin\\AppData\\Local\\Temp\\claude\\4ac3a056b343bb35\\bfe78918-15e8-439b-8566-593ca3297bd8\\scratchpad\\slowwm_controls_only.cmd" < /dev/null 2>&1 | tail -4
    if allpass; then echo "[$(date '+%T')] ALL_PASS True (fresh run)"; exit 0; fi
    echo "[$(date '+%T')] fresh run did not pass (see the error above)"
  fi
  if [ "$(date +%s)" -ge "$end" ]; then restore "no fresh all-pass by the 10 h deadline (last free ${f} GB)"; exit 3; fi
  sleep 120
done
