#!/bin/bash
# MM, 2026-11-17: relaunch the PI-approved weather-head chain (run2) after the T-ISO (c) re-entrancy fix (BUILD_ISOFIX.md:
# wh_iso.py runs the probe on the unwrapped forward_infer; study set 33 passed; 11 tokens re-minted for build sha 63df633e).
# Done steps are skipped (fit / t0 / kid of H_s0 and H_s1, iso_a_s0); the chain resumes at iso_c_s0.
# Waits for >= 12.5 GB free commit (the fits need about 11.5 GB in memory), clears the chain's FAILED marker only if it is the
# T-ISO (c) one, launches job_wh2.cmd via WMI. Last line: ZZ WH RELAUNCH OK ZZ <pid> | ZZ WH RELAUNCH FAIL ZZ <why>
LOG=F:/TanitAD-scratch/strat1/whead/logs/relaunch_after_isofix.log
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
echo "[relaunch] start $(date -u +%FT%TZ)" >> "$LOG"
"$PY" -u F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 12.5 600 >> "$LOG" 2>&1 || { echo "ZZ WH RELAUNCH FAIL ZZ commit stayed under 12.5 GB for 10 h" | tee -a "$LOG"; exit 1; }
"$PY" - >> "$LOG" 2>&1 <<'EOF'
import os
p = "F:/TanitAD-scratch/strat1/whead/run2/state/FAILED"
t = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
assert t.startswith("T-ISO (c) seed 0 did not pass"), t
os.remove(p)
print("cleared FAILED marker:", t.strip())
EOF
[ -e F:/TanitAD-scratch/strat1/whead/run2/state/FAILED ] && { echo "ZZ WH RELAUNCH FAIL ZZ FAILED marker not cleared" | tee -a "$LOG"; exit 1; }
r=$(powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\\TanitAD-scratch\\strat1\\whead\\launch_wh_wmi.ps1" -Job job_wh2.cmd 2>&1 | tr -d '\r')
echo "launch: $r" >> "$LOG"
printf '%s\n' "$r" | grep -q 'rv=0' || { echo "ZZ WH RELAUNCH FAIL ZZ WMI: $r" | tee -a "$LOG"; exit 1; }
echo "ZZ WH RELAUNCH OK ZZ $r" | tee -a "$LOG"
