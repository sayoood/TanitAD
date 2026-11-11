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
  unset SLOWWM_RESUME SLOWWM_TEST_COUNTS SLOWWM_TEST_BASIS SLOWWM_FAKE_FREE_GB; mkdir -p "$SLOWWM_PRED_STATE"
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
echo "== 10 hygiene"
ok "the poison transports never ran" '[ ! -e "$T/BREACH" ]'
ok "the package names no real-box path outside D:/tanitad and no G: path" '! grep -rq "G:/Meine\|G:\\\\Meine" "$PKG"/*.sh "$PKG"/stages "$PKG"/jobs "$PKG"/*.py'
ok "stage scripts parse (bash -n)" 'for f in "$PKG"/slowwm_chain.sh "$PKG"/stages/*.sh; do bash -n "$f" || exit 1; done'
echo "DRYRUN ok=$OK failed=$BAD"
[ "$BAD" -eq 0 ]
