#!/usr/bin/env bash
# Dry run of the SLOWWM chain's wait and refusal paths over a stub transport that reaches nothing (ssh_stub.sh exits 1, no box exists; a poison ssh/sftp/scp is first on PATH and must never run).
# Covers: the OBD seed-3 wait (RUNNING waits to timeout rc 12, DONE proceeds, busy STOP waits, non-busy STOP proceeds, the shared busy marker holds the chain), SLOWWM_DRY refusals (test hook without DRY; stub without DRY; real client with DRY),
# commit refusals, a stage failure writing STOP and releasing the marker. NOT covered here (needs the full fake box of WMPOOL's harness): check3-fail stop, one refused arm, pull sha mismatch rc 7 (see README, UNVERIFIED).
# Usage: bash test_slowwm_dryrun.sh -> prints "DRYRUN ok=<n> failed=<m>"
set -u
TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PKG="$(cd "$TESTS/.." && pwd)"; PKG="$(cygpath -m "$PKG" 2>/dev/null || echo "$PKG")"
PYV="${SLOWWM_TEST_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
T="${SLOWWM_TEST_TMP:-F:/TanitAD-scratch/slowwm_box2_build/slowwm_dry_run_$$}"
case "$T" in */slowwm_dry_run_*) ;; *) echo "refusing an unrecognised scratch dir $T" >&2; exit 2;; esac
mkdir -p "$T/poison"
for n in ssh sftp scp; do printf '#!/usr/bin/env bash\necho BREACH-%s >> "%s/BREACH"\nexit 99\n' "$n" "$T" > "$T/poison/$n"; chmod +x "$T/poison/$n"; done
export PATH="$T/poison:$PATH"
OK=0; BAD=0
ok() { if eval "$2"; then OK=$((OK+1)); else BAD=$((BAD+1)); echo "FAIL: $1"; fi; }
STUB="$(cygpath -m "$TESTS/ssh_stub.sh")"; C40="$(printf 'a%.0s' $(seq 1 40))"
setup() {   # setup <name>: a fresh state, busy marker and predecessor folder; the stubs; DRY on
  export SLOWWM_STATE="$T/st_$1" SLOWWM_BUSY_DEV="$T/busy_$1.txt" SLOWWM_ASIDE_DEV="$T/aside_$1" SLOWWM_PRED_STATE="$T/pred_$1" SLOWWM_PY="$PYV" SLOWWM_COMMIT="$C40"
  export SLOWWM_SSH="$STUB" SLOWWM_SFTP="$STUB" SLOWWM_DRY=1 SLOWWM_WAIT_POLLS=0 SLOWWM_WAIT_POLL_S=0
  unset SLOWWM_RESUME SLOWWM_PRED_DONE_PREFIX SLOWWM_TEST_COUNTS SLOWWM_TEST_BASIS SLOWWM_FAKE_FREE_GB; mkdir -p "$SLOWWM_PRED_STATE"
}
chain() { bash "$PKG/slowwm_chain.sh" > "$T/out_$1.txt" 2>&1; echo $?; }
stoptxt() { cat "$SLOWWM_STATE/STOP.txt" 2>/dev/null; }

echo "== 1 wait: OBD seed-3 chain RUNNING (no DONE, no STOP) -> waits, then times out rc 12"
setup w1; rc="$(chain w1)"
ok "rc 12" '[ "$rc" -eq 12 ]'; ok "STOP names predecessor-not-ended and rc=12" 'stoptxt | grep -q "predecessor-not-ended" && stoptxt | grep -q "rc=12"'
ok "the chain never reached ship" '! grep -q "start ship" "$SLOWWM_STATE/chain.log"'; ok "no busy marker was taken" '[ ! -e "$SLOWWM_BUSY_DEV" ]'
echo "== 2 wait: DONE.txt (ZZ OBD3-DONE) -> proceeds (the stub box then fails the ship stage: rc 11, STOP, marker released)"
setup w2; echo "ZZ OBD3-DONE 2026-11-11 00:00:00" > "$SLOWWM_PRED_STATE/DONE.txt"; rc="$(chain w2)"
ok "the chain saw the predecessor ended (DONE) and started ship" 'grep -q "ended (DONE)" "$SLOWWM_STATE/chain.log" && grep -q "start ship" "$SLOWWM_STATE/chain.log"'
ok "the stub failure stops the chain with a STOP line at stage ship (rc 11)" '[ "$rc" -eq 11 ] && stoptxt | grep -q "stop\|STOP" && stoptxt | grep -q "ship"'
ok "the busy marker was released after the stop" '[ ! -e "$SLOWWM_BUSY_DEV" ] && ls "$SLOWWM_ASIDE_DEV"/BOX2_BUSY.slowwm.*.txt > /dev/null 2>&1'
ok "no sentinel text in chain.log" '! grep -q "ZZ SLOWWM" "$SLOWWM_STATE/chain.log"'
echo "== 3 wait: a DONE.txt without the OBD3 prefix is not DONE"
setup w3; echo "something else" > "$SLOWWM_PRED_STATE/DONE.txt"; rc="$(chain w3)"; ok "waits (rc 12)" '[ "$rc" -eq 12 ]'
echo "== 4 wait: a busy STOP (head ends rc=12) keeps waiting"
setup w4; echo "ZZ OBD3-STOP start box2-busy rc=12 :: detail rc=4" > "$SLOWWM_PRED_STATE/STOP.txt"; rc="$(chain w4)"
ok "rc 12 and last state STOP-BUSY" '[ "$rc" -eq 12 ] && stoptxt | grep -q "STOP-BUSY"'
echo "== 5 wait: a non-busy STOP (rc=4) counts as ended -> proceeds"
setup w5; echo "ZZ OBD3-STOP arms arms-refused rc=4" > "$SLOWWM_PRED_STATE/STOP.txt"; rc="$(chain w5)"
ok "ended (STOP) and ship started" 'grep -q "ended (STOP)" "$SLOWWM_STATE/chain.log" && grep -q "start ship" "$SLOWWM_STATE/chain.log"'
echo "== 6 wait: DONE but the shared busy marker exists -> keeps waiting, the marker is untouched"
setup w6; echo "ZZ OBD3-DONE x" > "$SLOWWM_PRED_STATE/DONE.txt"; echo "another chain" > "$SLOWWM_BUSY_DEV"; rc="$(chain w6)"
ok "rc 12, ship never started, marker content unchanged" '[ "$rc" -eq 12 ] && ! grep -q "start ship" "$SLOWWM_STATE/chain.log" && [ "$(cat "$SLOWWM_BUSY_DEV")" = "another chain" ]'
echo "== 7 SLOWWM_DRY refusals"
setup d1; export SLOWWM_DRY=0 SLOWWM_FAKE_FREE_GB=100; rc="$(chain d1)"; ok "a test hook without SLOWWM_DRY=1 is refused (rc 8)" '[ "$rc" -eq 8 ] && stoptxt | grep -q "test-hook-in-real-run"'
setup d2; unset SLOWWM_DRY; rc="$(chain d2)"; ok "a stub transport without SLOWWM_DRY is refused (rc 16)" '[ "$rc" -eq 16 ] && stoptxt | grep -q "transport-not-fail-closed"'
setup d3; export SLOWWM_SSH="$(command -v ssh 2>/dev/null || echo "$T/poison/ssh")" SLOWWM_SFTP="$(command -v sftp 2>/dev/null || echo "$T/poison/sftp")"; export SLOWWM_SSH="$(cygpath -m "$SLOWWM_SSH")" SLOWWM_SFTP="$(cygpath -m "$SLOWWM_SFTP")"
rc="$(chain d3)"; ok "test mode with a real-client path is refused (rc 16)" '[ "$rc" -eq 16 ]'
setup d4; unset SLOWWM_SSH; rc="$(chain d4)"; ok "an unset SLOWWM_SSH is refused (rc 16)" '[ "$rc" -eq 16 ]'
setup d5; export SLOWWM_SSH="ssh"; rc="$(chain d5)"; ok "a relative SLOWWM_SSH is refused (rc 16)" '[ "$rc" -eq 16 ]'
echo "== 8 commit refusals"
setup c1; export SLOWWM_COMMIT="abc123"; rc="$(chain c1)"; ok "a short commit is refused (rc 8)" '[ "$rc" -eq 8 ]'
setup c2; unset SLOWWM_COMMIT; rc="$(chain c2)"; ok "no commit is refused (rc 8)" '[ "$rc" -eq 8 ]'
echo "== 9 a standing STOP refuses a rerun without SLOWWM_RESUME"
setup r1; echo "ZZ SLOWWM-STOP x y rc=5" > "$SLOWWM_STATE/STOP.txt" 2>/dev/null || { mkdir -p "$SLOWWM_STATE"; echo "ZZ SLOWWM-STOP x y rc=5" > "$SLOWWM_STATE/STOP.txt"; }; rc="$(chain r1)"; ok "rc 2" '[ "$rc" -eq 2 ]'
echo "== 10 relaunch behind the STOPBAL second dose (SLOWWM_PRED_STATE + SLOWWM_PRED_DONE_PREFIX): it WAITS for that chain"
dose2() { setup "$1"; export SLOWWM_PRED_DONE_PREFIX="ZZ STOPBAL2-DONE"; }
dose2 p1; rc="$(chain p1)"; ok "dose 2 still running (no DONE, no STOP): waits, times out rc 12, never reached ship" '[ "$rc" -eq 12 ] && stoptxt | grep -q "predecessor-not-ended" && stoptxt | grep -q "RUNNING" && ! grep -q "start ship" "$SLOWWM_STATE/chain.log"'
dose2 p2; echo "ZZ STOPBAL2-DONE 2026-11-11 23:00:00" > "$SLOWWM_PRED_STATE/DONE.txt"; rc="$(chain p2)"; ok "dose 2 DONE (ZZ STOPBAL2-DONE) releases it: ended (DONE), ship started" 'grep -q "ended (DONE)" "$SLOWWM_STATE/chain.log" && grep -q "start ship" "$SLOWWM_STATE/chain.log"'
dose2 p3; echo "ZZ OBD3-DONE 2026-11-11 23:00:00" > "$SLOWWM_PRED_STATE/DONE.txt"; rc="$(chain p3)"; ok "a DONE with the OBD3 prefix does not release a chain waiting for STOPBAL2 (rc 12, no ship)" '[ "$rc" -eq 12 ] && ! grep -q "start ship" "$SLOWWM_STATE/chain.log"'
dose2 p4; echo "ZZ STOPBAL2-STOP start box2-busy rc=12 :: x rc=4" > "$SLOWWM_PRED_STATE/STOP.txt"; rc="$(chain p4)"; ok "a busy STOP (head ends rc=12) of dose 2 keeps it waiting" '[ "$rc" -eq 12 ] && stoptxt | grep -q "STOP-BUSY" && ! grep -q "start ship" "$SLOWWM_STATE/chain.log"'
dose2 p5; echo "ZZ STOPBAL2-DONE x" > "$SLOWWM_PRED_STATE/DONE.txt"; echo "stopbal dose 2" > "$SLOWWM_BUSY_DEV"; rc="$(chain p5)"; ok "dose 2 DONE but the shared busy marker held: keeps waiting, marker untouched" '[ "$rc" -eq 12 ] && ! grep -q "start ship" "$SLOWWM_STATE/chain.log" && [ "$(cat "$SLOWWM_BUSY_DEV")" = "stopbal dose 2" ]'
dose2 p6; echo "ZZ STOPBAL2-STOP arms arms-refused rc=4" > "$SLOWWM_PRED_STATE/STOP.txt"; rc="$(chain p6)"; ok "a non-busy STOP of dose 2 counts as ended" 'grep -q "ended (STOP)" "$SLOWWM_STATE/chain.log"'
unset SLOWWM_PRED_DONE_PREFIX
echo "== 11 relaunch after the basis STOP (SLOWWM_RESUME=1): own STOP renamed (never deleted); the amended checker is shipped again; a ship done for this checker is skipped"
CKSHA="$(sha256sum "$PKG/slowwm_check.py" | cut -d' ' -f1)"
resume_setup() { setup "$1"; export SLOWWM_RESUME=1; mkdir -p "$SLOWWM_STATE/done"; echo "ZZ SLOWWM-STOP basis gate-or-pin-refused rc=4 :: basis: check3-refused" > "$SLOWWM_STATE/STOP.txt"; echo "ZZ OBD3-DONE x" > "$SLOWWM_PRED_STATE/DONE.txt"; }
resume_setup s1; echo "ship 2026-11-11 19:00:00" > "$SLOWWM_STATE/done/ship.done"; rc="$(chain s1)"
ok "an old ship marker (no checker sha) is shipped again; the old STOP is renamed, not deleted" 'grep -q "ship was done for another checker" "$SLOWWM_STATE/chain.log" && grep -q "start ship" "$SLOWWM_STATE/chain.log" && ls "$SLOWWM_STATE"/STOP.2*.txt > /dev/null 2>&1 && grep -q "check3-refused" "$SLOWWM_STATE"/STOP.2*.txt'
resume_setup s2; echo "ship 2026-11-11 19:00:00 checker=$(printf '0%.0s' $(seq 1 64))" > "$SLOWWM_STATE/done/ship.done"; rc="$(chain s2)"
ok "a ship marker for ANOTHER checker is shipped again" 'grep -q "ship was done for another checker" "$SLOWWM_STATE/chain.log" && grep -q "start ship" "$SLOWWM_STATE/chain.log"'
resume_setup s3; echo "ship 2026-11-11 19:00:00 checker=$CKSHA" > "$SLOWWM_STATE/done/ship.done"; rc="$(chain s3)"
ok "a ship marker for THIS checker is skipped and the basis stage starts (the stub box then stops it at stage basis)" 'grep -q "skip ship (done)" "$SLOWWM_STATE/chain.log" && grep -q "start basis" "$SLOWWM_STATE/chain.log" && ! grep -q "start ship" "$SLOWWM_STATE/chain.log" && stoptxt | grep -q "basis"'
resume_setup s4; unset SLOWWM_RESUME; rc="$(chain s4)"; ok "without SLOWWM_RESUME a standing STOP still refuses (rc 2) and is left in place" '[ "$rc" -eq 2 ] && grep -q "check3-refused" "$SLOWWM_STATE/STOP.txt"'
unset SLOWWM_RESUME
echo "== 12 hygiene"
ok "the poison transports never ran" '[ ! -e "$T/BREACH" ]'
ok "the package names no real-box path outside D:/tanitad and no G: path" '! grep -rq "G:/Meine\|G:\\\\Meine" "$PKG"/*.sh "$PKG"/stages "$PKG"/jobs "$PKG"/*.py'
ok "stage scripts parse (bash -n)" 'for f in "$PKG"/slowwm_chain.sh "$PKG"/stages/*.sh; do bash -n "$f" || exit 1; done'
echo "DRYRUN ok=$OK failed=$BAD"
[ "$BAD" -eq 0 ]
