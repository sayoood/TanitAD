#!/usr/bin/env bash
# RECOVER-NEXT addendum row D (RAPR as frozen, 2 runs): wait for box 2's lock, take it atomically, MOVE ASIDE the previous chain's
# CHAIN_RNEXT.done (never delete), ship, verify bytes, start telemetry -> guard -> the two-run chain (rnext_chain_rapr.cmd). RUN ONLY AFTER THE PI'S BOX-2 GO. Steps are HANDOFF_LAUNCH.md's, unchanged except the chain file. Fail closed: any failed step before the
# chain starts releases OUR lock (moved aside, never deleted) and exits non-zero. Prints ZZRAPR2-LAUNCHED on success.
set -u
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"; SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
SRC=/e/Projects/TanitAD-artifacts/dagland/recover_next/box2_files
LOCK=/c/Users/Admin/box2_ship/BOX2_BUSY.txt
WAIT_MAX=${RNEXT4_WAIT_MAX:-21600}     # 6 h
ts() { date '+%m-%d %H:%M:%S'; }
release() { mv "$LOCK" "/c/Users/Admin/box2_ship/BOX2_BUSY.rapr2_launch_failed_$(date +%Y%m%d_%H%M%S).txt"; echo "[$(ts)] lock released (moved aside)"; }
die() { echo "[$(ts)] ZZRAPR2-ABORT $*"; [ "${TELE:-0}" = 1 ] && $SSH "powershell -NoProfile -Command \"Set-Content -Path D:/tanitad/logs/telemetry.stop -Value stop\""; [ "${HAVE_LOCK:-0}" = 1 ] && release; exit 2; }

# 0. wait for the lock, then take it atomically (another owner may win the race: then keep waiting)
t0=$(date +%s); HAVE_LOCK=0
while :; do
  if [ ! -e "$LOCK" ]; then
    if ( set -o noclobber; echo "RECOVER-NEXT RAPR two-run chain (D:\\tanitad\\rnext\\rnext_chain_rapr.cmd), owner: Master Mind bfe78918, PI go ${RAPR2_PI_GO:?set RAPR2_PI_GO to the PI go time}, since $(date '+%m-%d %H:%M'), ETA ~4 h. Done when D:\\tanitad\\runs\\v8f_mini_full\\rnext\\CHAIN_RNEXT.done exists. Do NOT launch box 2 jobs while this file exists; moved aside after the pull." > "$LOCK" ) 2>/dev/null; then HAVE_LOCK=1; break; fi
  fi
  [ $(( $(date +%s) - t0 )) -gt "$WAIT_MAX" ] && { echo "[$(ts)] ZZRAPR2-WAIT-TIMEOUT lock still held:"; head -3 "$LOCK"; exit 5; }
  sleep 60
done
echo "[$(ts)] lock taken"

# 1. read-only preconditions: no python on box 2, D: free >= 8 GB (the chain itself refuses too); previous chains' state shown
pre=$($SSH "powershell -NoProfile -Command \"@(Get-Process python -ErrorAction SilentlyContinue).Count; [int]((Get-PSDrive D).Free/1GB); Test-Path D:/tanitad/runs/v8f_mini_full/rnext/CHAIN_RNEXT.done\"" | tr -d '\r') || die "ssh preconditions failed"
echo "[$(ts)] box2: python=$(echo "$pre" | sed -n 1p) Dfree_GB=$(echo "$pre" | sed -n 2p) old_CHAIN_RNEXT.done=$(echo "$pre" | sed -n 3p)"
[ "$(echo "$pre" | sed -n 1p)" = "0" ] || die "a python process is running on box 2"
[ "$(echo "$pre" | sed -n 2p)" -ge 8 ] 2>/dev/null || die "box 2 D: below 8 GB"
if [ "$(echo "$pre" | sed -n 3p)" = "True" ]; then
  st=$(date +%Y%m%d_%H%M%S)
  $SSH "powershell -NoProfile -Command \"Move-Item D:/tanitad/runs/v8f_mini_full/rnext/CHAIN_RNEXT.done D:/tanitad/runs/v8f_mini_full/rnext/CHAIN_RNEXT.rtl_rp01_$st.done\"" || die "moving the old CHAIN_RNEXT.done aside failed"
  echo "[$(ts)] old CHAIN_RNEXT.done moved aside to CHAIN_RNEXT.rtl_rp01_$st.done"
fi

# 2. ship (new folder; only the named files, never __pycache__)
$SSH "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force D:/tanitad/rnext/argv | Out-Null\"" || die "mkdir"
for f in rnext_check.py rnext_chain.cmd rnext_chain_rtl_rp01.cmd rnext_chain_rapr.cmd rnext_run.cmd rnext_gate_only.cmd rnext_guard.ps1 rnext_run_guard.cmd rnext_run_telemetry.cmd; do $SCP $SRC/$f tanitad-box2:D:/tanitad/rnext/$f || die "scp $f"; done
for t in RAPR_0 RAPR_1 RTL_0 RTL_1 RP01_0 RP01_1; do $SCP $SRC/argv/$t.train.txt tanitad-box2:D:/tanitad/rnext/argv/$t.train.txt || die "scp argv $t"; done

# 3. verify the shipped bytes
W=$(mktemp -d)
( cd $SRC && sha256sum rnext_* argv/*.train.txt | sed 's/ \*/  /; s# argv/# #' ) | sort > $W/dev.txt
$SSH "powershell -NoProfile -Command -" <<'PS' | tr -d '\r' | sort > $W/box2.txt
Get-ChildItem D:\tanitad\rnext -Recurse -File | ForEach-Object { (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower() + '  ' + $_.Name }
PS
diff $W/dev.txt $W/box2.txt || die "shipped bytes differ"
echo "[$(ts)] shipped bytes identical ($(wc -l < $W/dev.txt) files)"

# 4. telemetry first (must be FRESH), then the guard, then the chain
L() { $SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\tanitad\\launch.ps1 -Job $1 -Log $2"; }
L 'D:\tanitad\rnext\rnext_run_telemetry.cmd' 'D:\tanitad\logs\rnext_telemetry.log' || die "telemetry launch"; TELE=1; sleep 60
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" | tr -d '\r'; }
t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || die "no fresh telemetry (the guard would be blind): '$t1' / '$t2'"
echo "[$(ts)] telemetry fresh: $t2"
L 'D:\tanitad\rnext\rnext_run_guard.cmd' 'D:\tanitad\logs\rnext_guard.out' || die "guard launch"
sleep 20
L 'D:\tanitad\rnext\rnext_chain_rapr.cmd' 'D:\tanitad\logs\rnext_chain_rapr.log' || die "chain launch"
sleep 90
log=$($SSH "powershell -NoProfile -Command \"Get-Content D:\\tanitad\\logs\\rnext_chain_rapr.log -Tail 12\"" | tr -d '\r')
echo "$log"
echo "$log" | grep -q "ZZRNEXT-STOP" && { echo "[$(ts)] ZZRAPR2-CHAIN-STOPPED-AT-START (lock kept for the MM to read)"; exit 3; }
echo "$log" | grep -q "RNEXT CHAIN START" || { echo "[$(ts)] ZZRAPR2-NO-START-LINE (lock kept for the MM to read)"; exit 4; }
echo "[$(ts)] ZZRAPR2-LAUNCHED"
