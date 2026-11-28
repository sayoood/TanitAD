#!/usr/bin/env bash
# LEVELHEAD arm CL: the launch library, SOURCED by launch_wprobe.sh (MODE=wprobe) and launch_cl.sh (MODE=chain). It is a clone of the RPT launcher (recover_next/launch_rpt.sh) with the same safety:
#   refuses without the PI's go (CL_PI_GO) -> local refusal (cl_local_check.py: frozen W, pins, argv audit) -> the BOX2_BUSY lock (refuse if held; CL_WAIT_MAX=<s> opts into waiting) -> read-only preconditions
#   -> ship + sha256 re-hash on box 2 -> the four stack files (old ones moved aside, re-hashed on box 2) -> telemetry (must be FRESH) -> guard -> the job (SAFE MODE inside the job's .cmd files).
# Fail closed: any failed step before the job starts releases OUR lock (moved aside, never deleted) and exits non-zero. Nothing here runs without CL_PI_GO; ssh/scp are only ever called through $SSH / $SCP.
# Test hooks (dev box only; production sets none): CL_SSH, CL_SCP (whole commands), CL_PY, CL_SRC, CL_LOCK, CL_WAIT_MAX, CL_NAP (sleep seconds). A WARNING line names any that is set.
: "${CL_PI_GO:?set CL_PI_GO to the PI go time (HH:MM): this script refuses to touch box 2 without the go of the PI}"
CLDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SSH=${CL_SSH:-"ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"}
SCP=${CL_SCP:-"scp -q -o BatchMode=yes -o ConnectTimeout=20"}
PY=${CL_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}
SRC=${CL_SRC:-$CLDIR/box2_files}
LOCK=${CL_LOCK:-/c/Users/Admin/box2_ship/BOX2_BUSY.txt}
WAIT_MAX=${CL_WAIT_MAX:-0}
H2=D:/tanitad/cl; R2=D:/tanitad/runs/v8f_mini_full/cl
ts() { date '+%m-%d %H:%M:%S'; }
nap() { sleep "${CL_NAP:-$1}"; }
for _h in CL_SSH CL_SCP CL_PY CL_SRC CL_LOCK CL_WAIT_MAX CL_NAP; do [ -n "${!_h:-}" ] && echo "[$(ts)] WARNING test hook $_h is set: $_h=${!_h}"; done

release() { mv "$LOCK" "$(dirname "$LOCK")/BOX2_BUSY.cl_launch_failed_$(date +%Y%m%d_%H%M%S).txt"; echo "[$(ts)] lock released (moved aside)"; }
die() { echo "[$(ts)] ZZCL-ABORT $*"; [ "${TELE:-0}" = 1 ] && $SSH "powershell -NoProfile -Command \"Set-Content -Path D:/tanitad/logs/telemetry.stop -Value stop\""; [ "${HAVE_LOCK:-0}" = 1 ] && release; exit 2; }

cl_launch() {
  case "$MODE" in
    wprobe) JOB=cl_wprobe.cmd; JLOG=cl_wprobe.log; GUARDCMD=cl_run_guard_wprobe.cmd; ASIDE="WPROBE.done WPROBE.FAILED WPROBE.STOPPED"; START="CL WPROBE START"; ETA="~20 min"; DONE="WPROBE.done / WPROBE.FAILED"; WHAT="W probe (CL_0's argv with --selfcheck-only --level-head on --w-level 1)";;
    chain)  JOB=cl_chain.cmd;  JLOG=cl_chain.log;  GUARDCMD=cl_run_guard_chain.cmd;  ASIDE="CHAIN_CL.done CHAIN_CL.STOPPED"; START="CL CHAIN START"; ETA="~2 h"; DONE="CHAIN_CL.done"; WHAT="two-run chain (CL_0, CL_1: C0's argv + --level-head on --w-level W, 3,000 steps each)";;
    *) echo "MODE must be wprobe or chain"; exit 64;;
  esac
  # -1. local refusal (nothing remote, no lock): the frozen W, the pins, the argv audit
  $PY -I "$CLDIR/cl_local_check.py" "$MODE" | tr -d '\r'; [ "${PIPESTATUS[0]}" = 0 ] || { echo "[$(ts)] ZZCL-REFUSED by the local check (nothing contacted, no lock taken)"; exit 6; }
  # 0. the lock: refuse when held (CL_WAIT_MAX > 0 waits like the RPT launcher), then take it atomically
  t0=$(date +%s); HAVE_LOCK=0
  while :; do
    if [ ! -e "$LOCK" ]; then
      if ( set -o noclobber; echo "LEVELHEAD CL $WHAT (D:\\tanitad\\cl\\$JOB), owner: Master Mind, PI go ${CL_PI_GO}, since $(date '+%m-%d %H:%M'), ETA $ETA. Done when D:\\tanitad\\runs\\v8f_mini_full\\cl\\$DONE exists. Do NOT launch box 2 jobs while this file exists; moved aside after the pull." > "$LOCK" ) 2>/dev/null; then HAVE_LOCK=1; break; fi
    fi
    [ $(( $(date +%s) - t0 )) -ge "$WAIT_MAX" ] && { echo "[$(ts)] ZZCL-LOCK-HELD box 2 is busy (CL_WAIT_MAX=$WAIT_MAX); the lock says:"; head -3 "$LOCK"; exit 5; }
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

  # 2. ship (new folder D:\tanitad\cl; only the generated files, never __pycache__)
  FILES=$($PY -I "$CLDIR/cl_local_check.py" ship-files | tr -d '\r'); [ -n "$FILES" ] || die "no files to ship"
  $SSH "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force $H2/argv | Out-Null\"" || die "mkdir"
  for f in $FILES; do $SCP "$SRC/$f" "tanitad-box2:$H2/$f" || die "scp $f"; done

  # 3. verify the shipped bytes (the whole folder: names and sha256)
  TMPD=$(mktemp -d)
  ( cd "$SRC" && sha256sum $FILES | sed 's/ \*/  /; s#  .*/#  #' ) | sort > "$TMPD/dev.txt"
  $SSH "powershell -NoProfile -Command -" <<'PS' | tr -d '\r' | sort > "$TMPD/box2.txt"
Get-ChildItem D:\tanitad\cl -Recurse -File | ForEach-Object { (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower() + '  ' + $_.Name }
PS
  diff "$TMPD/dev.txt" "$TMPD/box2.txt" || die "shipped bytes differ"
  echo "[$(ts)] shipped bytes identical ($(wc -l < "$TMPD/dev.txt") files)"

  # ---- STACK-SHIP-BEGIN (the four stack files over box 2's repo: the new trainer and model, and the two unchanged files as the RPT launcher shipped them)
  # Local sha256 of each source against the PINNED value first (nothing on box 2 is touched on a mismatch). Then per file: box 2's current file is MOVED ASIDE to <name>.pre_cl_<ts> (never deleted), the new one is scp'd and re-hashed ON box 2;
  # any failure puts every file shipped so far back (the new one aside as <name>.cl_failed_<ts>, the old one back in place) and dies, which releases our lock.
  REPO2=D:/tanitad/repo; STAMP=$(date +%Y%m%d_%H%M%S); STK_DONE=()
  mapfile -t STK < <($PY -I "$CLDIR/cl_local_check.py" stack-rows | tr -d '\r'); [ "${#STK[@]}" = 4 ] || die "stack-rows did not give four rows"
  restore_stack() { local e rel; for e in "${STK_DONE[@]:-}"; do [ -n "$e" ] || continue; rel=${e%%|*}; $SSH "powershell -NoProfile -Command \"if (Test-Path $REPO2/$rel) { Move-Item $REPO2/$rel $REPO2/$rel.cl_failed_$STAMP }; if (Test-Path $REPO2/$rel.pre_cl_$STAMP) { Move-Item $REPO2/$rel.pre_cl_$STAMP $REPO2/$rel }\"" || echo "[$(ts)] WARNING restore of $rel failed: see $REPO2/$rel.pre_cl_$STAMP"; done; }
  for e in "${STK[@]}"; do
    IFS='|' read -r rel src want <<< "$e"
    [ -f "$src" ] || die "stack ship: source $src missing"
    got=$(sha256sum "$src" | cut -c1-64); [ "$got" = "$want" ] || die "stack ship: $src sha256 $got != pinned $want (regenerate with mk_cl_package.py)"
  done
  for e in "${STK[@]}"; do
    IFS='|' read -r rel src want <<< "$e"
    $SSH "powershell -NoProfile -Command \"if (Test-Path $REPO2/$rel) { Move-Item $REPO2/$rel $REPO2/$rel.pre_cl_$STAMP }\"" || { restore_stack; die "moving box 2's $rel aside failed"; }
    STK_DONE+=("$rel|$src|$want")
    $SCP "$src" "tanitad-box2:$REPO2/$rel" || { restore_stack; die "scp $rel"; }
    have=$($SSH "powershell -NoProfile -Command \"(Get-FileHash $REPO2/$rel -Algorithm SHA256).Hash.ToLower()\"" | tr -d '\r') || { restore_stack; die "re-hash of $rel on box 2 failed"; }
    [ "$have" = "$want" ] || { restore_stack; die "stack ship: $rel on box 2 hashes '$have', expected $want"; }
    echo "[$(ts)] shipped $rel (${want:0:12}), old file at $rel.pre_cl_$STAMP"
  done
  # ---- STACK-SHIP-END

  # 4. telemetry first (must be FRESH), then the guard, then the job
  L() { $SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\tanitad\\launch.ps1 -Job $1 -Log $2"; }
  L 'D:\tanitad\cl\cl_run_telemetry.cmd' 'D:\tanitad\logs\cl_telemetry.log' || die "telemetry launch"; TELE=1; nap 60
  tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" | tr -d '\r'; }
  t1=$(tl); nap 35; t2=$(tl)
  [[ $t2 == 20* && $t2 != "$t1" ]] || die "no fresh telemetry (the guard would be blind): '$t1' / '$t2'"
  echo "[$(ts)] telemetry fresh: $t2"
  L "D:\\tanitad\\cl\\$GUARDCMD" 'D:\tanitad\logs\cl_guard.out' || die "guard launch"
  nap 20
  L "D:\\tanitad\\cl\\$JOB" "D:\\tanitad\\logs\\$JLOG" || die "job launch"
  nap 90
  log=$($SSH "powershell -NoProfile -Command \"Get-Content D:\\tanitad\\logs\\$JLOG -Tail 12\"" | tr -d '\r')
  echo "$log"
  echo "$log" | grep -q "ZZCL-STOP" && { echo "[$(ts)] ZZCL-JOB-STOPPED-AT-START (lock kept for the MM to read)"; exit 3; }
  echo "$log" | grep -q "$START" || { echo "[$(ts)] ZZCL-NO-START-LINE (lock kept for the MM to read)"; exit 4; }
  echo "[$(ts)] ZZCL-LAUNCHED $MODE"
}
