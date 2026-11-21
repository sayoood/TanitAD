#!/usr/bin/env bash
# Dry run of the S0-P chain over a FAKE box 2 (fake_ssh.sh / fake_sftp.sh: a folder standing in for D:\; S0P_DRY=1; a poison ssh/sftp/scp is first on PATH and must never run).  Nothing real is reached.
# Covers: the SLOWWM wait (RUNNING / DONE / busy STOP / non-busy STOP / wrong prefix / busy marker held), transport + test-hook refusals, the full happy path (ship, pre-flight, dry run, 4 shards, pull + verify, box-2 copy deleted,
# busy marker released BEFORE the fit, fit stub args), disk-low (box and dev), a failed shard + resume (only that shard relaunched), a hung job never launched twice, pull retry / permanent corruption (box copy kept), a refused dry run,
# resume with everything pulled (no box contact), a stale own busy marker, job / launch.cmd rendering, hygiene.  NOT covered: the real extractor on box 2, real ssh / sftp / launch.ps1 behaviour, real timing (see README_CHAIN.md).
# Usage: bash test_s0p_chain_dryrun.sh -> prints "DRYRUN ok=<n> failed=<m>"
set -u
TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; CH="$(cd "$TESTS/.." && pwd)"; CH="$(cygpath -m "$CH" 2>/dev/null || echo "$CH")"; TESTS="$(cygpath -m "$TESTS")"
PYV="${S0P_TEST_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"; export S0P_TEST_PY="$PYV"
T="${S0P_TEST_TMP:-C:/Users/Admin/s0p_test/chain_dry_$$}"
case "$T" in */chain_dry_*) ;; *) echo "refusing an unrecognised scratch dir $T" >&2; exit 2;; esac
mkdir -p "$T/poison"
for n in ssh sftp scp; do printf '#!/usr/bin/env bash\necho BREACH-%s >> "%s/BREACH"\nexit 99\n' "$n" "$T" > "$T/poison/$n"; chmod +x "$T/poison/$n"; done
export PATH="$T/poison:$PATH"
OK=0; BAD=0
ok() { if eval "$2"; then OK=$((OK+1)); else BAD=$((BAD+1)); echo "FAIL: $1"; fi; }
SSHS="$TESTS/fake_ssh.sh"; SFTPS="$TESTS/fake_sftp.sh"
setup() {   # setup <name>: a fresh state, pull dir, work dir, fake box, busy marker and predecessor folder; stubs; DRY on
  local n="$1"
  export S0P_STATE="$T/st_$n" S0P_PULL="$T/pull_$n" S0P_WORK="$T/work_$n" S0P_BUSY_DEV="$T/busy_$n.txt" S0P_ASIDE_DEV="$T/aside_$n" S0P_PRED_STATE="$T/pred_$n" S0P_FAKE_BOX="$T/box_$n" S0P_RESULT="$T/st_$n/RESULT_S0P.json" S0P_EVAL="$T/eval_dummy.npz"
  export S0P_PY="$PYV" S0P_SSH="$SSHS" S0P_SFTP="$SFTPS" S0P_DRY=1 S0P_FIT_SCRIPT="$TESTS/fit_stub.py" S0P_K=4 S0P_EXPECT_NPOS=20 S0P_EXPECT_NNEG=60 S0P_SHARD_BYTES=1000000 S0P_FAKE_C_FREE_MB=50000
  export S0P_WAIT_POLLS=0 S0P_WAIT_POLL_S=0 S0P_POLL_S=0 S0P_POLLS_DRY=2 S0P_POLLS_SHARD=2
  unset S0P_RESUME; mkdir -p "$S0P_PRED_STATE" "$S0P_FAKE_BOX/ctl"; echo 20 > "$S0P_FAKE_BOX/ctl/npos"; echo 60 > "$S0P_FAKE_BOX/ctl/nneg"
  : > "$S0P_FAKE_BOX/ssh.log"; : > "$T/eval_dummy.npz"
  mkdir -p "$S0P_FAKE_BOX/D/tanitad/repo/stack/scripts" "$S0P_FAKE_BOX/D/tanitad/venv/Scripts" "$S0P_FAKE_BOX/D/tanitad/runs/v8f_mini_full" "$S0P_FAKE_BOX/D/tanitad/cache_full/train/clips" "$S0P_FAKE_BOX/D/tanitad/cache_full/eval"
  : > "$S0P_FAKE_BOX/D/tanitad/repo/stack/scripts/v8f_t1_eval.py"; : > "$S0P_FAKE_BOX/D/tanitad/venv/Scripts/python.exe"; : > "$S0P_FAKE_BOX/D/tanitad/runs/v8f_mini_full/join_train.npz"; : > "$S0P_FAKE_BOX/D/tanitad/cache_full/native_tracks.pt"
  : > "$S0P_FAKE_BOX/D/tanitad/cache_full/eval/manifest.jsonl"; : > "$S0P_FAKE_BOX/D/tanitad/launch.ps1"
}
slowdone() { echo "ZZ SLOWWM-DONE 2026-11-12 00:00:00" > "$S0P_PRED_STATE/DONE.txt"; }
chain() { bash "$CH/s0p_chain.sh" > "$T/out_$1.txt" 2>&1; echo $?; }
stoptxt() { cat "$S0P_STATE/STOP.txt" 2>/dev/null; }
nlaunch() { grep -c "$1" "$S0P_FAKE_BOX/launches.log" 2> /dev/null; }
box() { echo "$S0P_FAKE_BOX/D/tanitad/stopperc/$1"; }

echo "== 1 wait: SLOWWM RUNNING -> waits to timeout rc 12, no box contact, no busy marker"
setup w1; rc="$(chain w1)"
ok "rc 12, STOP predecessor-not-ended RUNNING" '[ "$rc" -eq 12 ] && stoptxt | grep -q "predecessor-not-ended" && stoptxt | grep -q "RUNNING" && stoptxt | grep -q "rc=12"'
ok "no ssh call, no busy marker" '[ ! -s "$S0P_FAKE_BOX/ssh.log" ] && [ ! -e "$S0P_BUSY_DEV" ]'
echo "== 2 wait: a wrong DONE prefix is not DONE; a busy STOP (head ends rc=12) keeps waiting; a STOP with a detail ending rc=4 after the head does not count as busy"
setup w2; echo "ZZ OBD3-DONE x" > "$S0P_PRED_STATE/DONE.txt"; rc="$(chain w2)"; ok "wrong prefix waits (rc 12)" '[ "$rc" -eq 12 ] && stoptxt | grep -q RUNNING'
setup w3; echo "ZZ SLOWWM-STOP start box2-busy rc=12 :: detail rc=4" > "$S0P_PRED_STATE/STOP.txt"; rc="$(chain w3)"; ok "busy STOP waits" '[ "$rc" -eq 12 ] && stoptxt | grep -q "STOP-BUSY"'
setup w4; echo "ZZ SLOWWM-STOP basis gate-or-pin-refused rc=4 :: x rc=12" > "$S0P_PRED_STATE/STOP.txt"; rc="$(chain w4)"; ok "non-busy STOP counts as ended" 'grep -q "ended (STOP)" "$S0P_STATE/chain.log"'
setup w5; slowdone; echo "another chain" > "$S0P_BUSY_DEV"; rc="$(chain w5)"
ok "DONE but the shared busy marker is held: waits, rc 12, marker untouched, no box contact" '[ "$rc" -eq 12 ] && [ "$(cat "$S0P_BUSY_DEV")" = "another chain" ] && [ ! -s "$S0P_FAKE_BOX/ssh.log" ]'

echo "== 3 the full happy path"
setup h; slowdone; rc="$(chain h)"
ok "rc 0, DONE.txt starts ZZ S0P-DONE with result and sha256" '[ "$rc" -eq 0 ] && head -c 14 "$S0P_STATE/DONE.txt" | grep -q "^ZZ S0P-DONE" && grep -q "sha256=$(sha256sum "$S0P_RESULT" | cut -d" " -f1)" "$S0P_STATE/DONE.txt"'
ok "no STOP.txt, no sentinel text in chain.log" '[ ! -e "$S0P_STATE/STOP.txt" ] && ! grep -q "ZZ S0P" "$S0P_STATE/chain.log"'
ok "all four shards pulled and verified; no partial dir" 'for i in 0 1 2 3; do [ -f "$S0P_PULL/shard_$i/MANIFEST.sha256" ] && [ -e "$S0P_STATE/done/shard_$i.done" ] || exit 1; done; ! ls -d "$S0P_PULL"/.partial_* 2> /dev/null'
ok "the box-2 shard copies were deleted, leftovers removed; scripts shipped" '! ls -d "$(box s0p_out)"/shard_* 2> /dev/null && [ ! -e "$(box s0p_out)" ] && [ -f "$(box s0p)/s0p_extract_train.py" ] && cmp -s "$(box s0p)/s0p_lib.py" "$CH/../s0p_lib.py"'
ok "deletions.log: 4 shards + the dry-run sample copy, path and bytes" '[ "$(grep -c "deleted box-2" "$S0P_STATE/deletions.log")" -eq 5 ] && grep -q "shard_2 (" "$S0P_STATE/deletions.log"'
ok "one dry run + exactly four shard launches" '[ "$(nlaunch "^dry")" -eq 1 ] && [ "$(nlaunch "^shard")" -eq 4 ] && [ "$(wc -l < "$S0P_FAKE_BOX/launches.log")" -eq 5 ]'
ok "the dry run ran BEFORE shard 0" '[ "$(head -n 1 "$S0P_FAKE_BOX/launches.log" | cut -d" " -f1)" = dry ]'
ok "the busy marker was released into busy_aside before the fit; the fit saw it absent" 'ls "$S0P_ASIDE_DEV"/BOX2_BUSY.s0p.*.txt > /dev/null 2>&1 && [ ! -e "$S0P_BUSY_DEV" ] && grep -q "\"busy_exists\": false" "$S0P_RESULT"'
ok "fit: CPU only, 2 threads, trained on the 4 shards (n 80), eval read-only path as given, result under the state dir" 'grep -q "\"cuda\": \"-1\"" "$S0P_RESULT" && grep -q "\"omp\": \"2\"" "$S0P_RESULT" && grep -q "\"n_train\": 80" "$S0P_RESULT" && grep -q "\"shards\": 4" "$S0P_RESULT" && grep -q "$S0P_EVAL" "$S0P_RESULT" && grep -q "$S0P_PULL" "$S0P_RESULT"'
ok "every box-2 WRITE (mkdir / delete / put) stays under D:\\tanitad\\stopperc; no unknown command" '! grep -E "New-Item|Remove-Item|rmdir" "$S0P_FAKE_BOX/ssh.log" | grep -vi "tanitad.stopperc" | grep -q . && ! grep -E "^put " "$S0P_FAKE_BOX/sftp.log" | grep -v "/D:/tanitad/stopperc/" | grep -q . && [ ! -e "$S0P_FAKE_BOX/unknown.log" ]'
echo "== 3b rerun with everything pulled, fit not yet done: no box contact, no wait, no claim"
rm -f "$S0P_STATE/DONE.txt" "$S0P_STATE/done/fit.done" "$S0P_RESULT"; : > "$S0P_FAKE_BOX/ssh.log"; rm -rf "$S0P_ASIDE_DEV"; rc="$(chain h2)"
ok "rc 0 with an empty ssh log and no busy marker taken" '[ "$rc" -eq 0 ] && [ ! -s "$S0P_FAKE_BOX/ssh.log" ] && [ ! -e "$S0P_BUSY_DEV" ] && [ -e "$S0P_STATE/DONE.txt" ]'

echo "== 4 refusals: transport and test hooks"
setup d1; export S0P_DRY=0; rc="$(chain d1)"; ok "a test hook without S0P_DRY=1 is refused (rc 8)" '[ "$rc" -eq 8 ] && stoptxt | grep -q "test-hook-in-real-run"'
setup d2; unset S0P_DRY S0P_FAKE_C_FREE_MB S0P_FIT_SCRIPT S0P_EXPECT_NPOS S0P_EXPECT_NNEG S0P_SHARD_BYTES S0P_K; rc="$(chain d2)"; ok "a stub transport without S0P_DRY is refused (rc 16)" '[ "$rc" -eq 16 ] && stoptxt | grep -q "transport-not-fail-closed"'
setup d3; export S0P_SSH="$(cygpath -m "$(command -v ssh 2>/dev/null || echo "$T/poison/ssh")")" S0P_SFTP="$(cygpath -m "$(command -v sftp 2>/dev/null || echo "$T/poison/sftp")")"; rc="$(chain d3)"; ok "test mode with a real-client path is refused (rc 16)" '[ "$rc" -eq 16 ]'
setup d4; unset S0P_SSH; rc="$(chain d4)"; ok "an unset S0P_SSH is refused (rc 16)" '[ "$rc" -eq 16 ]'
setup d5; export S0P_SSH="ssh"; rc="$(chain d5)"; ok "a relative S0P_SSH is refused (rc 16)" '[ "$rc" -eq 16 ]'
setup d6; mkdir -p "$S0P_STATE"; echo "ZZ S0P-STOP x y rc=5" > "$S0P_STATE/STOP.txt"; slowdone; rc="$(chain d6)"; ok "a standing STOP refuses a rerun without S0P_RESUME (rc 2), left in place" '[ "$rc" -eq 2 ] && grep -q "x y rc=5" "$S0P_STATE/STOP.txt"'
setup d7; export S0P_STATE="$T/slow_state_clash"; export S0P_PRED_STATE="$S0P_STATE"; rc="$(chain d7)"; ok "a state dir equal to the predecessor's is refused (rc 2)" '[ "$rc" -eq 2 ]'

echo "== 5 dry run refused / disk"
setup r1; slowdone; : > "$S0P_FAKE_BOX/ctl/dry_bad"; rc="$(chain r1)"
ok "a dry run without the banked-pool line stops the chain (rc 4), no shard launched, marker released" '[ "$rc" -eq 4 ] && stoptxt | grep -q "dry-run-refused" && [ "$(nlaunch "^shard")" -eq 0 ] && [ ! -e "$S0P_BUSY_DEV" ]'
setup k1; slowdone; echo 100000 > "$S0P_FAKE_BOX/ctl/free_bytes"; rc="$(chain k1)"
ok "box-2 D: free < shard + 1 GB -> STOP disk-low rc 6 at shard 0, no shard job, marker released" '[ "$rc" -eq 6 ] && stoptxt | grep -q "disk-low" && stoptxt | grep -q "box 2 D:" && [ "$(nlaunch "^shard")" -eq 0 ] && [ ! -e "$S0P_BUSY_DEV" ]'
setup k2; slowdone; export S0P_FAKE_C_FREE_MB=100; rc="$(chain k2)"; ok "dev C: below the 8 GB floor -> STOP disk-low rc 6, no shard job" '[ "$rc" -eq 6 ] && stoptxt | grep -q "dev C:" && [ "$(nlaunch "^shard")" -eq 0 ]'

echo "== 6 a failed shard job stops the chain; S0P_RESUME=1 relaunches ONLY that shard"
setup f1; slowdone; : > "$S0P_FAKE_BOX/ctl/fail_SH1"; rc="$(chain f1)"
ok "rc 5, STOP at shard1, shard 0 pulled, shard 1 not, marker released, no DONE" '[ "$rc" -eq 5 ] && stoptxt | grep -q "shard1" && [ -e "$S0P_STATE/done/shard_0.done" ] && [ ! -e "$S0P_STATE/done/shard_1.done" ] && [ ! -e "$S0P_BUSY_DEV" ] && [ ! -e "$S0P_STATE/DONE.txt" ]'
export S0P_RESUME=1; rc="$(chain f2)"
ok "resumed: rc 0, the old STOP renamed (not deleted), shard 0 launched once in total, shard 1 twice" '[ "$rc" -eq 0 ] && ls "$S0P_STATE"/STOP.2*.txt > /dev/null 2>&1 && [ "$(grep -c "^shard 0 " "$S0P_FAKE_BOX/launches.log")" -eq 1 ] && [ "$(grep -c "^shard 1 " "$S0P_FAKE_BOX/launches.log")" -eq 2 ] && grep -q "skip shard 0" "$S0P_STATE/chain.log"'
unset S0P_RESUME

echo "== 7 a job that is still running is WAITED on and never launched a second time"
setup j1; slowdone; : > "$S0P_FAKE_BOX/ctl/hang_SH2"; rc="$(chain j1)"
ok "timeout rc 11 at shard2, one launch of shard 2, its RUN marker on the box" '[ "$rc" -eq 11 ] && stoptxt | grep -q "shard2" && [ "$(grep -c "^shard 2 " "$S0P_FAKE_BOX/launches.log")" -eq 1 ] && [ -e "$(box s0p_jobs)/RUN_SH2.txt" ]'
export S0P_RESUME=1; rc="$(chain j2)"
ok "resume while it still hangs: no second launch of shard 2, waits again (rc 11)" '[ "$rc" -eq 11 ] && [ "$(grep -c "^shard 2 " "$S0P_FAKE_BOX/launches.log")" -eq 1 ] && grep -q "waiting for its marker instead of launching another" "$T/out_j2.txt"'
GID2="$(head -n 1 "$(box s0p_jobs)/RUN_SH2.txt" | tr -d '\r ')"
"$PYV" -I "$TESTS/fake_shard.py" --out "$(box s0p_out)/shard_2" --i 2 --k 4 --n-total 80 --n-pos 20 --sample-sha "$(printf 'sample%.0s' 1 | sha256sum | cut -d' ' -f1)" > /dev/null; echo done > "$(box s0p_jobs)/SH2_$GID2.done"
rm -f "$S0P_FAKE_BOX/ctl/hang_SH2"; rc="$(chain j3)"
ok "the job finishes meanwhile: resume pulls it WITHOUT launching again, chain completes" '[ "$rc" -eq 0 ] && [ "$(grep -c "^shard 2 " "$S0P_FAKE_BOX/launches.log")" -eq 1 ] && [ -e "$S0P_STATE/done/shard_2.done" ] && [ -e "$S0P_STATE/DONE.txt" ]'
unset S0P_RESUME

echo "== 8 pull verification"
setup p1; slowdone; : > "$S0P_FAKE_BOX/ctl/corrupt_once_v40.npy"; rc="$(chain p1)"
ok "one corrupted transfer is retried and the chain completes (v40.npy fetched twice for shard 0)" '[ "$rc" -eq 0 ] && [ "$(grep -c "^get -r .*shard_0/v40.npy" "$S0P_FAKE_BOX/sftp.log")" -eq 2 ]'
setup p2; slowdone; : > "$S0P_FAKE_BOX/ctl/corrupt_always_E.npy"; rc="$(chain p2)"
ok "a permanently corrupted transfer: rc 7 pull-sha-mismatch, 3 attempts, shard 0 NOT marked pulled, its BOX copy NOT deleted, marker released" '[ "$rc" -eq 7 ] && stoptxt | grep -q "pull-sha-mismatch" && [ "$(grep -c "^get -r .*shard_0/E.npy" "$S0P_FAKE_BOX/sftp.log")" -eq 3 ] && [ ! -e "$S0P_STATE/done/shard_0.done" ] && [ -f "$(box s0p_out)/shard_0/E.npy" ] && [ ! -e "$S0P_BUSY_DEV" ]'

echo "== 9 a stale OWN busy marker (killed chain) is moved aside under S0P_RESUME=1 only if its pid is gone"
setup s1; slowdone; echo "s0p chain 99999 2026-11-12 01:00:00" > "$S0P_BUSY_DEV"; export S0P_RESUME=1; rc="$(chain s1)"
ok "moved aside (S0P_RESUME=1, pid gone), chain completes" '[ "$rc" -eq 0 ] && ls "$S0P_ASIDE_DEV"/BOX2_BUSY.s0p_stale.*.txt > /dev/null 2>&1'
setup s2; slowdone; echo "s0p chain 99999 2026-11-12 01:00:00" > "$S0P_BUSY_DEV"; rc="$(chain s2)"; ok "without S0P_RESUME the stale marker is left alone and the chain waits (rc 12)" '[ "$rc" -eq 12 ] && [ -e "$S0P_BUSY_DEV" ]'
unset S0P_RESUME

echo "== 10 job and launch.cmd rendering"
JD="$T/render"; mkdir -p "$JD"
"$PYV" -I "$CH/s0p_chain_tool.py" mkjob --mode shard --i 2 --k 4 --gid 20261112_010203 --out "$JD/sh2.cmd"; "$PYV" -I "$CH/s0p_chain_tool.py" mkjob --mode dry --gid 20261112_010203 --out "$JD/dry.cmd"
ok "shard job: CRLF, --shard 2/4, venv python, CPU env, markers, writes only under D:\\tanitad\\stopperc" '[ "$(grep -c $'"'"'\r$'"'"' "$JD/sh2.cmd")" -eq "$(wc -l < "$JD/sh2.cmd")" ] && grep -q -- "--shard 2/4" "$JD/sh2.cmd" && grep -qF "D:\\tanitad\\venv\\Scripts\\python.exe" "$JD/sh2.cmd" && grep -q "CUDA_VISIBLE_DEVICES=-1" "$JD/sh2.cmd" && grep -q "OMP_NUM_THREADS=2" "$JD/sh2.cmd" && grep -qF "SH2_20261112_010203.done" "$JD/sh2.cmd" && grep -q "rem S0PJOB mode=shard i=2 k=4 gid=20261112_010203" "$JD/sh2.cmd" && ! grep -q -- "--dry-run" "$JD/sh2.cmd"'
ok "dry job has --dry-run and no --shard; no control character (the printf mangling of backslashes) in either" 'grep -q -- "--dry-run" "$JD/dry.cmd" && ! grep -q -- "--shard" "$JD/dry.cmd" && ! LC_ALL=C grep -q $'"'"'[\x01-\x08\x0b\x0c\x0e-\x1f]'"'"' "$JD/sh2.cmd" "$JD/dry.cmd"'
"$PYV" -I "$CH/s0p_chain_tool.py" mklaunch --chain "$CH/s0p_chain.sh" --console "C:/Users/Admin/s0p_state/chain.console.log" --ssh "C:/Program Files/Git/usr/bin/ssh.exe" --sftp "C:/Program Files/Git/usr/bin/sftp.exe" --out "$JD/launch.cmd"
ok "launch.cmd: CRLF, runs Git bash -l on the chain with the console log, the real client paths set, no control chars" '[ "$(grep -c $'"'"'\r$'"'"' "$JD/launch.cmd")" -eq "$(wc -l < "$JD/launch.cmd")" ] && grep -qF "\"C:\\Program Files\\Git\\bin\\bash.exe\" -l " "$JD/launch.cmd" && grep -qF "s0p_chain.sh\" > \"C:\Users\Admin\s0p_state\chain.console.log\" 2>&1" "$JD/launch.cmd" && grep -qF "set \"S0P_SSH=C:\\Program Files\\Git\\usr\\bin\\ssh.exe\"" "$JD/launch.cmd" && ! LC_ALL=C grep -q $'"'"'[\x01-\x08\x0b\x0c\x0e-\x1f]'"'"' "$JD/launch.cmd"'
echo "== 11 hygiene"
ok "the poison transports never ran" '[ ! -e "$T/BREACH" ]'
ok "chain scripts parse (bash -n)" 'for f in "$CH"/s0p_chain.sh "$CH"/s0p_common.sh "$TESTS"/fake_ssh.sh "$TESTS"/fake_sftp.sh; do bash -n "$f" || exit 1; done'
ok "no G: path, no F: write path in the chain sources" '! grep -rq "G:/Meine\|G:\\\\Meine" "$CH"/*.sh "$CH"/*.py && ! grep -nE "(>|mkdir -p|rm -rf?|mv|cp) +\"?F:" "$CH"/s0p_chain.sh "$CH"/s0p_common.sh | grep -q .'
echo "DRYRUN ok=$OK failed=$BAD"
[ "$BAD" -eq 0 ]
