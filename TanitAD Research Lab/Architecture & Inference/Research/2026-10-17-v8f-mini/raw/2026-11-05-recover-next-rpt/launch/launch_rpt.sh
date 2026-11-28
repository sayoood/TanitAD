#!/usr/bin/env bash
# RECOVER-NEXT addendum 4 (RPT = R's frozen argv with --recover-dev-shape pertick, 2 runs): wait for box 2's lock, take it atomically, MOVE ASIDE the previous chain's
# CHAIN_RNEXT.done (never delete), ship, verify bytes, SHIP THE THREE STACK FILES (trainer, v8f_recover, v8f_mini_data: sha256 pinned, box 2's old files moved aside, re-hashed on box 2), start telemetry -> guard -> the two-run chain (rnext_chain_rpt.cmd). RUN ONLY AFTER THE PI'S BOX-2 GO. Steps are HANDOFF_LAUNCH.md's, unchanged except the chain file and the stack ship. Fail closed: any failed step before the
# chain starts releases OUR lock (moved aside, never deleted) and exits non-zero. Prints ZZRPT-LAUNCHED on success.
set -u
: "${RPT_PI_GO:?set RPT_PI_GO to the PI go time (HH:MM): this launcher refuses to run without the go of the PI}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"; SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
SRC=/e/Projects/TanitAD-artifacts/dagland/recover_next/box2_files
LOCK=/c/Users/Admin/box2_ship/BOX2_BUSY.txt
WAIT_MAX=${RNEXT4_WAIT_MAX:-21600}     # 6 h
ts() { date '+%m-%d %H:%M:%S'; }
release() { mv "$LOCK" "/c/Users/Admin/box2_ship/BOX2_BUSY.rpt_launch_failed_$(date +%Y%m%d_%H%M%S).txt"; echo "[$(ts)] lock released (moved aside)"; }
die() { echo "[$(ts)] ZZRPT-ABORT $*"; [ "${TELE:-0}" = 1 ] && $SSH "powershell -NoProfile -Command \"Set-Content -Path D:/tanitad/logs/telemetry.stop -Value stop\""; [ "${HAVE_LOCK:-0}" = 1 ] && release; exit 2; }

# 0. wait for the lock, then take it atomically (another owner may win the race: then keep waiting)
t0=$(date +%s); HAVE_LOCK=0
while :; do
  if [ ! -e "$LOCK" ]; then
    if ( set -o noclobber; echo "RECOVER-NEXT RPT two-run chain (D:\\tanitad\\rnext\\rnext_chain_rpt.cmd), owner: Master Mind bfe78918, PI go ${RPT_PI_GO:?set RPT_PI_GO to the PI go time}, since $(date '+%m-%d %H:%M'), ETA ~4 h. Done when D:\\tanitad\\runs\\v8f_mini_full\\rnext\\CHAIN_RNEXT.done exists. Do NOT launch box 2 jobs while this file exists; moved aside after the pull." > "$LOCK" ) 2>/dev/null; then HAVE_LOCK=1; break; fi
  fi
  [ $(( $(date +%s) - t0 )) -gt "$WAIT_MAX" ] && { echo "[$(ts)] ZZRPT-WAIT-TIMEOUT lock still held:"; head -3 "$LOCK"; exit 5; }
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
  $SSH "powershell -NoProfile -Command \"Move-Item D:/tanitad/runs/v8f_mini_full/rnext/CHAIN_RNEXT.done D:/tanitad/runs/v8f_mini_full/rnext/CHAIN_RNEXT.raprp01_$st.done\"" || die "moving the old CHAIN_RNEXT.done aside failed"
  echo "[$(ts)] old CHAIN_RNEXT.done moved aside to CHAIN_RNEXT.raprp01_$st.done"
fi

# 2. ship (new folder; only the named files, never __pycache__)
$SSH "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force D:/tanitad/rnext/argv | Out-Null\"" || die "mkdir"
for f in rnext_check.py rnext_chain.cmd rnext_chain_rtl_rp01.cmd rnext_chain_rapr.cmd rnext_chain_raprp01.cmd rnext_chain_rpt.cmd rnext_run.cmd rnext_gate_only.cmd rnext_guard.ps1 rnext_run_guard.cmd rnext_run_telemetry.cmd; do $SCP $SRC/$f tanitad-box2:D:/tanitad/rnext/$f || die "scp $f"; done
for t in RAPR_0 RAPR_1 RTL_0 RTL_1 RP01_0 RP01_1 RAPRP01_0 RAPRP01_1 RPT_0 RPT_1; do $SCP $SRC/argv/$t.train.txt tanitad-box2:D:/tanitad/rnext/argv/$t.train.txt || die "scp argv $t"; done

# 3. verify the shipped bytes
W=$(mktemp -d)
( cd $SRC && sha256sum rnext_* argv/*.train.txt | sed 's/ \*/  /; s# argv/# #' ) | sort > $W/dev.txt
$SSH "powershell -NoProfile -Command -" <<'PS' | tr -d '\r' | sort > $W/box2.txt
Get-ChildItem D:\tanitad\rnext -Recurse -File | ForEach-Object { (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower() + '  ' + $_.Name }
PS
diff $W/dev.txt $W/box2.txt || die "shipped bytes differ"
echo "[$(ts)] shipped bytes identical ($(wc -l < $W/dev.txt) files)"

# ---- STACK-SHIP-BEGIN (RPT: the new trainer lever --recover-dev-shape, its library and the data module's ClipStore.get_kin: three files over box 2's repo, before the telemetry)
# Local sha256 of each source against the PINNED value first (nothing on box 2 is touched on a mismatch). Then per file: box 2's current file is MOVED ASIDE to <name>.pre_rpt_<ts> (never deleted), the new one is scp'd, and re-hashed ON box 2;
# any failure puts every file shipped so far back (the new one aside as <name>.rpt_failed_<ts>, the old one back in place) and dies, which releases our lock.
REPO2=D:/tanitad/repo; STAMP=$(date +%Y%m%d_%H%M%S); STK_DONE=()
STK=(
  "stack/scripts/v8f_mini_train.py|/e/Projects/TanitAD-artifacts/dagland/pertick/stack_patch/stack/scripts/v8f_mini_train.py|bda468fe01b25e901df9115218fcc125ce4234225aa3ea43613e03df3742660e"
  "stack/tanitad/data/v8f_recover.py|/e/Projects/TanitAD-artifacts/dagland/pertick/stack_patch/stack/tanitad/data/v8f_recover.py|0257ccb93251827fe5f1b3d7195999b5836145e8615cfd0d0a599262b8834b4c"
  "stack/tanitad/data/v8f_mini_data.py|$SRC/../rpt_stack/stack/tanitad/data/v8f_mini_data.py|209da7a60765ba81268ee794aba187e21b0a6c73a49b9e98c86e2d876581ece6"
)
restore_stack() { local e rel; for e in "${STK_DONE[@]:-}"; do [ -n "$e" ] || continue; rel=${e%%|*}; $SSH "powershell -NoProfile -Command \"if (Test-Path $REPO2/$rel) { Move-Item $REPO2/$rel $REPO2/$rel.rpt_failed_$STAMP }; if (Test-Path $REPO2/$rel.pre_rpt_$STAMP) { Move-Item $REPO2/$rel.pre_rpt_$STAMP $REPO2/$rel }\"" || echo "[$(ts)] WARNING restore of $rel failed: see $REPO2/$rel.pre_rpt_$STAMP"; done; }
for e in "${STK[@]}"; do
  IFS='|' read -r rel src want <<< "$e"
  [ -f "$src" ] || die "stack ship: source $src missing"
  got=$(sha256sum "$src" | cut -c1-64); [ "$got" = "$want" ] || die "stack ship: $src sha256 $got != pinned $want (re-pin with mk_rpt_pins.py, then regenerate this launcher)"
done
for e in "${STK[@]}"; do
  IFS='|' read -r rel src want <<< "$e"
  $SSH "powershell -NoProfile -Command \"if (Test-Path $REPO2/$rel) { Move-Item $REPO2/$rel $REPO2/$rel.pre_rpt_$STAMP }\"" || { restore_stack; die "moving box 2's $rel aside failed"; }
  STK_DONE+=("$rel|$src|$want")
  $SCP "$src" "tanitad-box2:$REPO2/$rel" || { restore_stack; die "scp $rel"; }
  have=$($SSH "powershell -NoProfile -Command \"(Get-FileHash $REPO2/$rel -Algorithm SHA256).Hash.ToLower()\"" | tr -d '\r') || { restore_stack; die "re-hash of $rel on box 2 failed"; }
  [ "$have" = "$want" ] || { restore_stack; die "stack ship: $rel on box 2 hashes '$have', expected $want"; }
  echo "[$(ts)] shipped $rel (${want:0:12}), old file at $rel.pre_rpt_$STAMP"
done
# ---- STACK-SHIP-END

# 4. telemetry first (must be FRESH), then the guard, then the chain
L() { $SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\tanitad\\launch.ps1 -Job $1 -Log $2"; }
L 'D:\tanitad\rnext\rnext_run_telemetry.cmd' 'D:\tanitad\logs\rnext_telemetry.log' || die "telemetry launch"; TELE=1; sleep 60
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" | tr -d '\r'; }
t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || die "no fresh telemetry (the guard would be blind): '$t1' / '$t2'"
echo "[$(ts)] telemetry fresh: $t2"
L 'D:\tanitad\rnext\rnext_run_guard.cmd' 'D:\tanitad\logs\rnext_guard.out' || die "guard launch"
sleep 20
L 'D:\tanitad\rnext\rnext_chain_rpt.cmd' 'D:\tanitad\logs\rnext_chain_rpt.log' || die "chain launch"
sleep 90
log=$($SSH "powershell -NoProfile -Command \"Get-Content D:\\tanitad\\logs\\rnext_chain_rpt.log -Tail 12\"" | tr -d '\r')
echo "$log"
echo "$log" | grep -q "ZZRNEXT-STOP" && { echo "[$(ts)] ZZRPT-CHAIN-STOPPED-AT-START (lock kept for the MM to read)"; exit 3; }
echo "$log" | grep -q "RNEXT CHAIN START" || { echo "[$(ts)] ZZRPT-NO-START-LINE (lock kept for the MM to read)"; exit 4; }
echo "[$(ts)] ZZRPT-LAUNCHED"
