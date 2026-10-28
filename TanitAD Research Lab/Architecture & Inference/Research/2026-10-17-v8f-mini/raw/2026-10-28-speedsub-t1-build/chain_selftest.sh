#!/usr/bin/env bash
# chain_selftest.sh -- the chain's own control flow, exercised for real with a STUB wrapper that starts no model and rolls out nothing.
# The stub writes the files a finished rollout leaves (empty dump npz files, ticks, json, record), so run_speedsub_t1.sh's real code runs: render-argv, the done records, the resume/skip rule, the
# move-aside of an incomplete directory, the lock, the stop on a failed rollout, and the STOP at the smoke check (the stub's records are empty, so the reader's --smoke-check must refuse them).
#
#   bash chain_selftest.sh [<work dir>]        prints PASS/FAIL per check and ZZSST1-SELFTEST-DONE n/n
shopt -u patsub_replacement 2>/dev/null || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYV="${SST1_PYTHON:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
SRC_CODE="${SST1_CODE:-$HERE/stack/scripts}"
W="${1:-F:/TanitAD-scratch/speedsub_t1/selftest/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$W/code" "$W/out"
cp "$SRC_CODE"/*.py "$W/code/"
cat > "$W/code/v8f_t1_speedsub.py" <<'STUB'
"""STUB wrapper of the chain self-test: parses like the real argv, rolls out nothing, writes the files a finished rollout leaves."""
import os, sys
a = sys.argv[1:]
def val(flag):
    return a[a.index(flag) + 1]
if os.environ.get("STUB_FAIL") and os.environ["STUB_FAIL"] in val("--tag"):
    sys.exit(3)
tag, out, ticks = val("--tag"), val("--out-dir"), val("--dump-ticks")
os.makedirs(f"{out}/dump_{tag}", exist_ok=True)
os.makedirs(ticks, exist_ok=True)
for i in range(1 if "--max-windows" in a else 141):
    open(f"{out}/dump_{tag}/ep{i:04d}.npz", "wb").close()
for p in (f"{out}/t1v8f_{tag}.json", f"{ticks}/ticks_{tag}.npz", f"{ticks}/speedsub_{tag}.npz"):
    open(p, "wb").close()
print("stub rollout", tag)
STUB
export SST1_CODE="$W/code" SST1_OUT="$W/out" SST1_READER="${SST1_READER:-$HERE/read/speedsub_t1_read.py}"
N=0; OK=0
check() { N=$((N+1)); if [ "$2" = 0 ]; then OK=$((OK+1)); echo "PASS $1"; else echo "FAIL $1"; fi; }
chain() { bash "$HERE/run_speedsub_t1.sh" "$@" > "$W/last.out" 2>&1; echo $? > "$W/last.rc"; }
has() { grep -q -- "$1" "$W/last.out"; echo $?; }

# 1. the first run: stages fit, pin, static pass; the smoke rollouts are made by the stub; the reader's --smoke-check refuses the empty records -> STOP smoke, rc 1
chain
check "1a first run stops at the smoke check (rc 1)" "$([ "$(cat "$W/last.rc")" = 1 ] && echo 0 || echo 1)"
check "1b marker ZZSST1-STOP smoke" "$(has 'ZZSST1-STOP smoke')"
check "1c the pin stage wrote pins.json" "$([ -f "$W/out/pins.json" ] && echo 0 || echo 1)"
check "1d both smoke rollouts have a done record" "$([ -f "$W/out/SMOKE/CTRL_s0/identity/done_SST1_SMOKE_CTRL_s0_identity.txt" ] && [ -f "$W/out/SMOKE/CTRL_s0/egoc/done_SST1_SMOKE_CTRL_s0_egoc.txt" ] && echo 0 || echo 1)"
check "1e no decision rollout was started after the stop" "$([ ! -d "$W/out/CTRL_s0" ] && echo 0 || echo 1)"
check "1f the lock is released on exit" "$([ ! -d "$W/out/.chain.lock" ] && echo 0 || echo 1)"

# 2. resume: a finished rollout is skipped
chain
check "2a rerun skips the finished smoke rollouts" "$(has 'skip SST1_SMOKE_CTRL_s0_identity')"
check "2b rerun stops at the same place" "$(has 'ZZSST1-STOP smoke')"

# 3. an incomplete directory (no done record) is moved aside, never deleted
rm -f "$W/out/SMOKE/CTRL_s0/egoc/done_SST1_SMOKE_CTRL_s0_egoc.txt"
chain
check "3a an incomplete rollout directory is moved aside" "$(has 'moved an incomplete')"
check "3b the moved-aside copy exists" "$(ls -d "$W"/out/SMOKE/CTRL_s0/egoc.incomplete.* >/dev/null 2>&1 && echo 0 || echo 1)"

# 4. a rollout that exits non-zero stops the chain at once, naming it
rm -rf "$W/out/SMOKE/CTRL_s0/identity"
STUB_FAIL=SMOKE_CTRL_s0_identity chain
check "4a a failed rollout stops the chain" "$(has 'ZZSST1-STOP SST1_SMOKE_CTRL_s0_identity the rollout exited 3')"
check "4b and its done record was NOT written" "$([ ! -f "$W/out/SMOKE/CTRL_s0/identity/done_SST1_SMOKE_CTRL_s0_identity.txt" ] && echo 0 || echo 1)"

# 5. an existing lock stops a second chain
mkdir "$W/out/.chain.lock"
chain
check "5a a held lock stops a second chain" "$(has 'ZZSST1-STOP init')"
rmdir "$W/out/.chain.lock"

# 6. one stage on its own; pins that differ from the files stop the pin stage
echo "tampered" >> "$W/code/v8f_speedsub_t1_fit.py"
chain --stage pin
check "6a pins that do not match the files stop the pin stage" "$(has 'ZZSST1-STOP pin')"

echo "ZZSST1-SELFTEST-DONE $OK/$N  (work dir $W)"
[ "$OK" = "$N" ]
