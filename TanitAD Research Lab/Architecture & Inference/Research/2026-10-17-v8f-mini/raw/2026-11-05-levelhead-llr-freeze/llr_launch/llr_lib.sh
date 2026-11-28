#!/usr/bin/env bash
# LEVELHEAD arm LLR: the launch library, SOURCED by launch_wleadprobe.sh (MODE=wprobe) and launch_llr.sh (MODE=chain). A clone of the arm-CLR launcher (clr_launch/clr_lib.sh, a clone of the arm-CL and RPT launchers) with the same safety:
#   refuses without the PI's go (LLR_PI_GO) -> local refusal (llr_local_check.py: frozen W, the LLR freeze file, pins, argv audit, landed stack, the frozen A2 card + ruling) -> the BOX2_BUSY lock (refuse if held; LLR_WAIT_MAX=<s> opts into waiting) -> read-only preconditions
#   -> ship + sha256 re-hash on box 2 -> the four stack files (old ones moved aside, re-hashed on box 2) -> telemetry (must be FRESH) -> guard -> the job (SAFE MODE inside the job's .cmd files).
# Fail closed: any failed step before the job starts releases OUR lock (moved aside, never deleted) and exits non-zero. Nothing here runs without LLR_PI_GO; ssh/scp are only ever called through $SSH / $SCP.
# Own folders only: D:\tanitad\llr (files), D:\tanitad\runs\v8f_mini_full\llr (markers), gate_llr (gate outputs). The folders of the earlier arms and of the RPT job are never touched, and no path of theirs may appear in this package (tests/test_llr_launch.py pins that).
# Test hooks (dev box only; production sets none): LLR_SSH, LLR_SCP (whole commands), LLR_PY, LLR_SRC, LLR_LOCK, LLR_WAIT_MAX, LLR_NAP (sleep seconds). A WARNING line names any that is set.
: "${LLR_PI_GO:?set LLR_PI_GO to the PI go time (HH:MM): this script refuses to touch box 2 without the go of the PI}"
LLDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SSH=${LLR_SSH:-"ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"}
SCP=${LLR_SCP:-"scp -q -o BatchMode=yes -o ConnectTimeout=20"}
PY=${LLR_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}
SRC=${LLR_SRC:-$LLDIR/box2_files}
LOCK=${LLR_LOCK:-/c/Users/Admin/box2_ship/BOX2_BUSY.txt}
WAIT_MAX=${LLR_WAIT_MAX:-0}
H2=D:/tanitad/llr; R2=D:/tanitad/runs/v8f_mini_full/llr
ts() { date '+%m-%d %H:%M:%S'; }
nap() { sleep "${LLR_NAP:-$1}"; }
for _h in LLR_SSH LLR_SCP LLR_PY LLR_SRC LLR_LOCK LLR_WAIT_MAX LLR_NAP; do [ -n "${!_h:-}" ] && echo "[$(ts)] WARNING test hook $_h is set: $_h=${!_h}"; done

release() { mv "$LOCK" "$(dirname "$LOCK")/BOX2_BUSY.llr_launch_failed_$(date +%Y%m%d_%H%M%S).txt"; echo "[$(ts)] lock released (moved aside)"; }
die() { echo "[$(ts)] ZZLLR-ABORT $*"; [ "${TELE:-0}" = 1 ] && $SSH "powershell -NoProfile -Command \"Set-Content -Path D:/tanitad/logs/telemetry.stop -Value stop\""; [ "${HAVE_LOCK:-0}" = 1 ] && release; exit 2; }

llr_launch() {
  case "$MODE" in
    wprobe) JOB=llr_wprobe.cmd; JLOG=llr_wprobe.log; GUARDCMD=llr_run_guard_wprobe.cmd; ASIDE="WLPROBE.done WLPROBE.FAILED WLPROBE.STOPPED"; START="LLR WPROBE START"; ETA="~20 min"; DONE="WLPROBE.done / WLPROBE.FAILED"; WHAT="W_lead probe (LLR_0's argv with --selfcheck-only --w-lead 1 and the frozen wiring)";;
    chain)  JOB=llr_chain.cmd;  JLOG=llr_chain.log;  GUARDCMD=llr_run_guard_chain.cmd;  ASIDE="CHAIN_LLR.done CHAIN_LLR.STOPPED"; START="LLR CHAIN START"; ETA="~2.5 h"; DONE="CHAIN_LLR.done"; WHAT="two-run chain (LLR_0, LLR_1: CLR's argv + --lead-head on --w-lead WL --aux-cond WIRING, 3,000 steps each)";;
    *) echo "MODE must be wprobe or chain"; exit 64;;
  esac
  # -1. local refusal (nothing remote, no lock): the frozen W, the LLR freeze file (wiring; W_lead for the chain), the pins, the argv audit, the landed stack, the frozen A2 card and ruling
  $PY -I "$LLDIR/llr_local_check.py" "$MODE" | tr -d '\r'; [ "${PIPESTATUS[0]}" = 0 ] || { echo "[$(ts)] ZZLLR-REFUSED by the local check (nothing contacted, no lock taken)"; exit 6; }
  # 0. the lock: refuse when held (LLR_WAIT_MAX > 0 waits like the RPT launcher), then take it atomically
  t0=$(date +%s); HAVE_LOCK=0
  while :; do
    if [ ! -e "$LOCK" ]; then
      if ( set -o noclobber; echo "LEVELHEAD LLR $WHAT (D:\\tanitad\\llr\\$JOB), owner: Master Mind, PI go ${LLR_PI_GO}, since $(date '+%m-%d %H:%M'), ETA $ETA. Done when D:\\tanitad\\runs\\v8f_mini_full\\llr\\$DONE exists. Do NOT launch box 2 jobs while this file exists; moved aside after the pull." > "$LOCK" ) 2>/dev/null; then HAVE_LOCK=1; break; fi
    fi
    [ $(( $(date +%s) - t0 )) -ge "$WAIT_MAX" ] && { echo "[$(ts)] ZZLLR-LOCK-HELD box 2 is busy (LLR_WAIT_MAX=$WAIT_MAX); the lock says:"; head -3 "$LOCK"; exit 5; }
    sleep 60
  done
  echo "[$(ts)] lock taken"

  # 1. read-only preconditions: no python on box 2, D: free >= 8 GB, the marker folder's names (old markers of THIS job are moved aside, a guard stop is read by a human)
  pre=$($SSH "powershell -NoProfile -Command \"@(Get-Process python -ErrorAction SilentlyContinue).Count; [int]((Get-PSDrive D).Free/1GB); if (Test-Path $R2) { Get-ChildItem $R2 -Name }; exit 0\"" | tr -d '\r') || die "ssh preconditions failed"
  echo "[$(ts)] box2: python=$(echo "$pre" | sed -n 1p) Dfree_GB=$(echo "$pre" | sed -n 2p) markers: $(echo "$pre" | sed -n '3,$p' | tr '\n' ' ')"
  [ "$(echo "$pre" | sed -n 1p)" = "0" ] || die "a python process is running on box 2"
  [ "$(echo "$pre" | sed -n 2p)" -ge 8 ] 2>/dev/null || die "box 2 D: below 8 GB"
  st=$(date +%Y%m%d_%H%M%S)
  for nm in $(echo "$pre" | sed -n '3,$p'); do
    [ "$nm" = "GUARD_STOP.txt" ] && die "a GUARD_STOP.txt from an earlier job is in $R2: read it (heat stop) and move it aside by hand before launching"
    for m in $ASIDE; do
      if [ "$nm" = "$m" ]; then $SSH "powershell -NoProfile -Command \"Move-Item $R2/$nm $R2/$nm.prev_$st\"" || die "moving the old $nm aside failed"; echo "[$(ts)] old $nm moved aside to $nm.prev_$st"; fi
    done
  done

  # 2. ship (new folder D:\tanitad\llr; only the generated files, never __pycache__)
  FILES=$($PY -I "$LLDIR/llr_local_check.py" ship-files | tr -d '\r'); [ -n "$FILES" ] || die "no files to ship"
  $SSH "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force $H2/argv | Out-Null\"" || die "mkdir"
  for f in $FILES; do $SCP "$SRC/$f" "tanitad-box2:$H2/$f" || die "scp $f"; done

  # 3. verify the shipped bytes (the whole folder: names and sha256)
  TMPD=$(mktemp -d)
  ( cd "$SRC" && sha256sum $FILES | sed 's/ \*/  /; s#  .*/#  #' ) | sort > "$TMPD/dev.txt"
  $SSH "powershell -NoProfile -Command -" <<'PS' | tr -d '\r' | sort > "$TMPD/box2.txt"
Get-ChildItem D:\tanitad\llr -Recurse -File | ForEach-Object { (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower() + '  ' + $_.Name }
PS
  diff "$TMPD/dev.txt" "$TMPD/box2.txt" || die "shipped bytes differ"
  echo "[$(ts)] shipped bytes identical ($(wc -l < "$TMPD/dev.txt") files)"

  # ---- STACK-SHIP-BEGIN (the four stack files over box 2's repo: the lead-head trainer, model and data module + v8f_recover.py, all four equal the pins)
  # Local sha256 of each source against the PINNED value first (nothing on box 2 is touched on a mismatch). Then per file: box 2's current file is MOVED ASIDE to <name>.pre_llr_<ts> (never deleted), the new one is scp'd and re-hashed ON box 2;
  # any failure puts every file shipped so far back (the new one aside as <name>.llr_failed_<ts>, the old one back in place) and dies, which releases our lock.
  REPO2=D:/tanitad/repo; STAMP=$(date +%Y%m%d_%H%M%S); STK_DONE=()
  mapfile -t STK < <($PY -I "$LLDIR/llr_local_check.py" stack-rows | tr -d '\r'); [ "${#STK[@]}" = 4 ] || die "stack-rows did not give four rows"
  restore_stack() { local e rel; for e in "${STK_DONE[@]:-}"; do [ -n "$e" ] || continue; rel=${e%%|*}; $SSH "powershell -NoProfile -Command \"if (Test-Path $REPO2/$rel) { Move-Item $REPO2/$rel $REPO2/$rel.llr_failed_$STAMP }; if (Test-Path $REPO2/$rel.pre_llr_$STAMP) { Move-Item $REPO2/$rel.pre_llr_$STAMP $REPO2/$rel }\"" || echo "[$(ts)] WARNING restore of $rel failed: see $REPO2/$rel.pre_llr_$STAMP"; done; }
  for e in "${STK[@]}"; do
    IFS='|' read -r rel src want <<< "$e"
    [ -f "$src" ] || die "stack ship: source $src missing"
    got=$(sha256sum "$src" | cut -c1-64); [ "$got" = "$want" ] || die "stack ship: $src sha256 $got != pinned $want (regenerate with mk_llr_package.py)"
  done
  for e in "${STK[@]}"; do
    IFS='|' read -r rel src want <<< "$e"
    $SSH "powershell -NoProfile -Command \"if (Test-Path $REPO2/$rel) { Move-Item $REPO2/$rel $REPO2/$rel.pre_llr_$STAMP }\"" || { restore_stack; die "moving box 2's $rel aside failed"; }
    STK_DONE+=("$rel|$src|$want")
    $SCP "$src" "tanitad-box2:$REPO2/$rel" || { restore_stack; die "scp $rel"; }
    have=$($SSH "powershell -NoProfile -Command \"(Get-FileHash $REPO2/$rel -Algorithm SHA256).Hash.ToLower()\"" | tr -d '\r') || { restore_stack; die "re-hash of $rel on box 2 failed"; }
    [ "$have" = "$want" ] || { restore_stack; die "stack ship: $rel on box 2 hashes '$have', expected $want"; }
    echo "[$(ts)] shipped $rel (${want:0:12}), old file at $rel.pre_llr_$STAMP"
  done
  # ---- STACK-SHIP-END

  # 4. telemetry first (must be FRESH), then the guard, then the job
  L() { $SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\tanitad\\launch.ps1 -Job $1 -Log $2"; }
  L 'D:\tanitad\llr\llr_run_telemetry.cmd' 'D:\tanitad\logs\llr_telemetry.log' || die "telemetry launch"; TELE=1; nap 60
  tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" | tr -d '\r'; }
  t1=$(tl); nap 35; t2=$(tl)
  [[ $t2 == 20* && $t2 != "$t1" ]] || die "no fresh telemetry (the guard would be blind): '$t1' / '$t2'"
  echo "[$(ts)] telemetry fresh: $t2"
  L "D:\\tanitad\\llr\\$GUARDCMD" 'D:\tanitad\logs\llr_guard.out' || die "guard launch"
  nap 20
  L "D:\\tanitad\\llr\\$JOB" "D:\\tanitad\\logs\\$JLOG" || die "job launch"
  nap 90
  log=$($SSH "powershell -NoProfile -Command \"Get-Content D:\\tanitad\\logs\\$JLOG -Tail 12\"" | tr -d '\r')
  echo "$log"
  echo "$log" | grep -q "ZZLLR-STOP" && { echo "[$(ts)] ZZLLR-JOB-STOPPED-AT-START (lock kept for the MM to read)"; exit 3; }
  echo "$log" | grep -q "$START" || { echo "[$(ts)] ZZLLR-NO-START-LINE (lock kept for the MM to read)"; exit 4; }
  echo "[$(ts)] ZZLLR-LAUNCHED $MODE"
}
