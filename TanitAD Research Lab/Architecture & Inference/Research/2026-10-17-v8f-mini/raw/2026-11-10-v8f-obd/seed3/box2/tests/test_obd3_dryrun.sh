#!/usr/bin/env bash
# Dry run of the OBD seed-3 box-2 package over stubs. NOTHING here can reach box 2: OBD3_SSH / OBD3_SFTP name stub scripts by absolute path (OBD3_DRY=1 makes the chain refuse any file without the stub marker),
# a poison ssh/sftp/scp is put FIRST on PATH and must never run, and the "box" is a folder. The REAL job .cmd files run on Windows cmd.exe against it; only the trainer is a stand-in (the OBD chain's
# tests/plant_trainer.py, shipped through a throw-away git repo, so the ship, tree-binding and token code run for real). CPU only, no GPU, no torch.   Usage: bash test_obd3_dryrun.sh  -> prints "DRYRUN ok=<n> failed=<m>"
set -u
TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PKG="$(cd "$TESTS/.." && pwd)"; PKG="$(cygpath -m "$PKG" 2>/dev/null || echo "$PKG")"
ORIG="$(cd "$TESTS/../../../box2" && pwd)"; ORIG="$(cygpath -m "$ORIG" 2>/dev/null || echo "$ORIG")"          # theory/obd/box2: the OBD chain's package, never written
PYV="${OBD3_TEST_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
T="${OBD3_TEST_TMP:-F:/TanitAD-scratch/obd_seed3_build/obd3_dry_run_$$}"
case "$T" in */obd3_dry_run_*) ;; *) echo "refusing an unrecognised scratch dir $T" >&2; exit 2;; esac
rm -rf "${T:?}"; mkdir -p "$T/poison"
for n in ssh sftp scp; do printf '#!/usr/bin/env bash\necho BREACH-%s >> "%s/BREACH"\nexit 99\n' "$n" "$T" > "$T/poison/$n"; chmod +x "$T/poison/$n"; done
printf '#!/usr/bin/env bash\necho BREACH-realclient >> "%s/BREACH"\nexit 99\n' "$T" > "$T/real_client.sh"      # stands in for a real ssh/sftp (no stub marker); harmless
export PATH="$T/poison:$PATH"
export CUDA_VISIBLE_DEVICES=-1
OK=0; BAD=0
ok() { if eval "$2"; then OK=$((OK+1)); else BAD=$((BAD+1)); echo "FAIL: $1"; fi; }
SSHS="$(cygpath -m "$TESTS/ssh_stub.sh")"; SFTPS="$(cygpath -m "$TESTS/sftp_stub.sh")"; REALC="$(cygpath -m "$T/real_client.sh")"
treehash() { ( cd "$1" && find . -type f ! -path '*/__pycache__/*' | sort | xargs sha256sum | sha256sum | cut -d' ' -f1 ); }
ORIG_H0="$(treehash "$ORIG")"
COMMIT="$("$PYV" -W ignore "$TESTS/dry_helpers.py" mkrepo "$T/repo")"
ok "throw-away repo has a 40-hex commit" '[ ${#COMMIT} -eq 40 ]'
TAGS="OBD-CTL_s10 OBD-BAND_s10 OBD-CTL_s11 OBD-BAND_s11"
newbox() {   # newbox <name> [gpu MiB]  -> exports the dry-run environment for box <name>; the STOPBAL chain has ended (done marker) unless the test says otherwise
  B="$T/box_$1"; "$PYV" -W ignore "$TESTS/dry_helpers.py" mkbox "$B" "$PYV" "${2:-0}" "$COMMIT" > /dev/null
  mkdir -p "$B/D/tanitad/other/runs" "$B/D/tanitad/repo/stack"; echo "other chain file" > "$B/D/tanitad/other/runs/R1.txt"; echo "shared repo file" > "$B/D/tanitad/repo/stack/x.py"
  export OBD3_DRY=1 OBD3_SSH="$SSHS" OBD3_SFTP="$SFTPS" FAKE_BOX="$(cygpath -m "$B")" FAKE_PY="$PYV" OBD3_PY="$PYV" OBD3_STATE="$T/st_$1" OBD3_PULLS="$T/pulls_$1"
  export OBD3_BUSY_DEV="$T/busy_$1.txt" OBD3_ASIDE_DEV="$T/aside_$1" OBD3_B2_WIN_ROOT="$(cygpath -w "$B/D/tanitad")" OBD3_POLL_S=0 OBD3_REPO="$T/repo" OBD3_EXPORT_LIST="$T/repo/export_list.txt"
  export OBD3_COMMIT="$COMMIT" OBD3_NEED_C_MB=1 OBD3_POLLS_RUN=5 OBD3_POLLS_SHIP=5 OBD3_POLLS_CLEAN=5 OBD3_WAIT_POLLS=0 OBD3_WAIT_POLL_S=0
  export OBD3_STOPBAL_STATE="$T/stopbal_$1"; mkdir -p "$OBD3_STOPBAL_STATE"; echo "ZZ STOPBAL-DONE 2026-11-11 00:00:00 commit=x" > "$OBD3_STOPBAL_STATE/DONE.txt"
  unset OBD3_RESUME OBD3_CODE_COMMIT FAKE_CORRUPT_PUT FAKE_CORRUPT_GET
}
control() { printf '%s' "$1" > "$B/plant_control.json"; }
calls() { cat "$B/transport.log" 2>/dev/null | wc -l | tr -d ' '; }
stoptxt() { cat "$OBD3_STATE/STOP.txt" 2>/dev/null; }
# the box-2 desktop apps hold about 2.4 GB of VRAM: the default idle threshold (3500 MiB) must pass at 2400
echo "== 1 normal flow: wait check, ship (tree absent: fallback ship, then proof), four runs, banks, cleanups; GPU at 2400 MiB passes the default idle threshold"
newbox normal 2400; control '{"sps": 1.0, "startup_s": 30}'
OH0="$(treehash "$B/D/tanitad/other")"; RH0="$(treehash "$B/D/tanitad/repo")"
T0=$(date +%s); bash "$PKG/obd3_chain.sh" > "$T/normal.out" 2>&1; rc=$?; T1=$(date +%s); echo "   (scenario 1 took $((T1-T0)) s)"
ok "chain exits 0" '[ $rc -eq 0 ]'
ok "DONE sentinel written, no stop" '[ -e "$OBD3_STATE/DONE.txt" ] && [ ! -e "$OBD3_STATE/STOP.txt" ] && head -c 13 "$OBD3_STATE/DONE.txt" | grep -q "^ZZ OBD3-DONE"'
ok "sentinel text only in its own file (chain.log carries none)" '! grep -q "ZZ OBD3" "$OBD3_STATE/chain.log" && ! grep -q "ZZ STOPBAL" "$OBD3_STATE/chain.log"'
ok "run order in chain.log: CTL_s10, BAND_s10, CTL_s11, BAND_s11" '[ "$(grep -o "done OBD-[A-Za-z0-9_-]*" "$OBD3_STATE/chain.log" | tr "\n" " ")" = "done OBD-CTL_s10 done OBD-BAND_s10 done OBD-CTL_s11 done OBD-BAND_s11 " ]'
ok "no NULL arm, no pilot, no parity anywhere in the state or the transport" '! ls "$OBD3_STATE" "$OBD3_STATE/done" "$OBD3_STATE/runs" | grep -qiE "null|pilot|parity" && ! grep -qiE "NULL_|PILOT|parity" "$B/transport.log"'
ok "the tree was not on the box: the fallback shipped it (tar put) and SHIPCHECK then proved it" 'grep -q "obd3/ship/obd_tree.tar" "$B/transport.log" && [ -s "$OBD3_STATE/ship/$COMMIT/TREE_PROVEN.json" ] && grep -q "\"ok\": true" "$OBD3_STATE/ship/$COMMIT/TREE_PROVEN.json"'
ok "the tree sits at the OBD path (same path as the OBD chain) with TIP_OBD.txt = the commit" '[ -s "$B/D/tanitad/obd/repo/stack/scripts/v8f_mini_train.py" ] && [ "$(cat "$B/D/tanitad/obd/repo/TIP_OBD.txt" | tr -d "\r\n")" = "$COMMIT" ] && [ -s "$B/D/tanitad/obd/tree_manifest.json" ]'
ok "no tree was written to a package-owned path (obd3\\repo does not exist)" '[ ! -e "$B/D/tanitad/obd3/repo" ]'
for t in $TAGS; do
  BD="$OBD3_PULLS/obd_$t"
  ok "$t banked with BANK.done and the six evidence files" '[ -e "$BD/BANK.done" ] && [ -s "$BD/ckpt_last.pt" ] && [ -s "$BD/summary.json" ] && [ -s "$BD/train_log.jsonl" ] && [ -s "$BD/gate_token.json" ] && [ -s "$BD/selfcheck.json" ] && [ -s "$BD/RUN.done" ]'
  ok "$t banked checkpoint equals the box copy (sha256)" '[ "$(sha256sum "$BD/ckpt_last.pt" | cut -d" " -f1)" = "$(sha256sum "$B/D/tanitad/runs/v8f_obd/$t/ckpt_last.pt" | cut -d" " -f1)" ]'
  ok "$t token binds the commit and the argv" '"$PYV" -I "$PKG/obd3_check.py" verify-token --token "$BD/gate_token.json" --argv-file "$BD/argv.txt" --tag $t --commit "$COMMIT" > /dev/null'
  ok "$t RUN.done carries the moved identity (S, T and O moved)" 'grep -q "\"moved_identity\"" "$BD/RUN.done" && grep -q "\"S\"" "$BD/RUN.done" && grep -q "\"O\"" "$BD/RUN.done"'
  ok "$t gate dir on the box deleted (obd3\\gate_out), the checkpoint kept" '[ ! -e "$B/D/tanitad/obd3/gate_out/$t" ] && [ -s "$B/D/tanitad/runs/v8f_obd/$t/ckpt_last.pt" ]'
  ok "$t cleanup logged (path, bytes) in the bank" 'grep -q "\"bytes\"" "$BD/CLEANUP.json"'
  ok "$t run folder is under obd3\\runs (own state)" '[ -s "$B/D/tanitad/obd3/runs/$t/RUN.done" ] && [ ! -e "$B/D/tanitad/obd/runs/$t" ]'
done
ok "the four tokens bind four distinct argvs" '[ "$(cat "$OBD3_PULLS"/obd_*/gate_token.json | grep -o "\"argv_sha256\": \"[0-9a-f]*\"" | sort -u | wc -l | tr -d " ")" -eq 4 ]'
ok "each banked argv names its seed and out dir (10, 10, 11, 11)" '[ "$(for t in $TAGS; do tr " " "\n" < "$OBD3_PULLS/obd_$t/argv.txt" | sed -n "/^--seed$/{n;p}"; done | tr "\n" " ")" = "10 10 11 11 " ] && grep -q "v8f_obd.OBD-BAND_s11" "$OBD3_PULLS/obd_OBD-BAND_s11/argv.txt"'
ok "the checkpoints carry the training seeds (10, 10, 11, 11) and the init seeds' DAgger index" '"$PYV" -I -c "import pickle,sys;g=lambda t:pickle.load(open(sys.argv[1]+\"/obd_\"+t+\"/ckpt_last.pt\",\"rb\"));d={t:g(t) for t in sys.argv[2:]};assert [d[t][\"seed\"] for t in sys.argv[2:]]==[10,10,11,11] and [d[t][\"dagger\"][\"dagger_seed\"] for t in sys.argv[2:]]==[0,0,1,1]" "$OBD3_PULLS" $TAGS'
ok "the box saw only stub traffic and no unknown command" '[ "$(calls)" -gt 60 ] && ! grep -q "UNKNOWN COMMAND" "$B/transport.log"'
ok "disk was read before every stage (>= 5 reads: ship, four runs)" '[ "$(grep -c "Get-PSDrive D" "$B/transport.log")" -ge 5 ]'
ok "the GPU was read before every run (4 reads) and python.exe counted" '[ "$(grep -c "^ssh nvidia-smi" "$B/transport.log")" -eq 4 ] && [ "$(grep -c "^ssh tasklist" "$B/transport.log")" -eq 4 ]'
ok "isolation: another chain folder on the box is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/other")" = "$OH0" ]'
ok "isolation: the shared D:\\tanitad\\repo is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/repo")" = "$RH0" ]'
ok "the box wrote no bytecode into the shared OBD tree" '[ -z "$(find "$B/D/tanitad/obd/repo" -name "*.pyc")" ]'
ok "busy marker handed on" '[ ! -e "$OBD3_BUSY_DEV" ] && ls "$OBD3_ASIDE_DEV" | grep -q "BOX2_BUSY.obd3"'
ok "no breach through PATH" '[ ! -e "$T/BREACH" ]'
ok "a second chain run in the same state skips finished stages" 'bash "$PKG/obd3_chain.sh" 2>&1 | tail -3 | grep -q "all runs banked"'

echo "== 2 the tree is already on the box (shipped by an earlier chain): SHIPCHECK proves it, ship is skipped, the tree stays byte-identical"
NB="$B"; TH0="$(treehash "$NB/D/tanitad/obd")"; TREEH0="$(treehash "$NB/D/tanitad/obd/repo")"
export OBD3_STATE="$T/st_proven" OBD3_PULLS="$T/pulls_proven"; L0="$(calls)"
bash "$PKG/obd3_chain.sh" > "$T/proven.out" 2>&1; rc=$?
ok "chain exits 0 on the second state dir" '[ $rc -eq 0 ] && head -c 13 "$OBD3_STATE/DONE.txt" | grep -q "^ZZ OBD3-DONE"'
ok "ship was proven, not shipped: TREE_PROVEN.json exists, no tar and no manifest put after the first call count" '[ -s "$OBD3_STATE/ship/$COMMIT/TREE_PROVEN.json" ] && ! tail -n +$((L0+1)) "$B/transport.log" | grep -q "obd_tree" && ! tail -n +$((L0+1)) "$B/transport.log" | grep -q "obd_refresh"'
ok "no tar was built on the dev box for the proven path" '[ ! -e "$OBD3_STATE/ship/$COMMIT/obd_tree.tar" ]'
ok "the code tree is byte-identical (proof is read only)" '[ "$(treehash "$B/D/tanitad/obd/repo")" = "$TREEH0" ]'
ok "four banks again, in the second pulls folder" '[ "$(ls "$OBD3_PULLS"/*/BANK.done | wc -l | tr -d " ")" -eq 4 ]'
ok "the SHIPCHECK job log names the proof and the members" 'grep -q "OBD3-SHIPCHECK-OK" "$B/D/tanitad/obd3/ship/shipcheck.log"'
echo "== 2b the shipcheck is not fooled: a changed tree file, a wrong TIP, a missing manifest -> NOT proven, the fallback re-ships (and the changed file is put back, the old one copied aside)"
newbox tamper 0; mkdir -p "$B/D/tanitad"; cp -r "$NB/D/tanitad/obd" "$B/D/tanitad/obd"
printf 'tampered\n' >> "$B/D/tanitad/obd/repo/stack/tanitad/models/__init__.py"
bash "$PKG/obd3_chain.sh" > "$T/tamper.out" 2>&1; rc=$?
ok "tampered tree: chain still ends 0 (re-shipped, then proven)" '[ $rc -eq 0 ] && grep -q "obd3/ship/obd_tree.tar" "$B/transport.log"'
ok "the tampered file is back to the commit's bytes and the old one is copied aside" '[ "$(sha256sum "$B/D/tanitad/obd/repo/stack/tanitad/models/__init__.py" | cut -d" " -f1)" = "$(sha256sum "$T/repo/stack/tanitad/models/__init__.py" | cut -d" " -f1)" ] && [ -s "$B/D/tanitad/obd/repo_aside_obd_${COMMIT:0:12}/stack/tanitad/models/__init__.py" ]'
newbox tipwrong 0; cp -r "$NB/D/tanitad/obd" "$B/D/tanitad/obd"; echo "0000000000000000000000000000000000000000" > "$B/D/tanitad/obd/repo/TIP_OBD.txt"
bash "$PKG/obd3_chain.sh" > "$T/tipwrong.out" 2>&1; rc=$?
ok "wrong TIP_OBD.txt: not proven, re-shipped, then proven" '[ $rc -eq 0 ] && grep -q "obd3/ship/obd_tree.tar" "$B/transport.log" && [ "$(tr -d "\r\n" < "$B/D/tanitad/obd/repo/TIP_OBD.txt")" = "$COMMIT" ]'
echo "== 2c a commit that is not the box tree's: the shipcheck refuses and the fallback cannot make it so"
newbox othercommit 0; OTHER="$("$PYV" -I -c "print('1'*40)")"; export OBD3_COMMIT="$OTHER"
bash "$PKG/obd3_chain.sh" > "$T/other.out" 2>&1; rc=$?
ok "a commit the repo does not hold: the ship stage refuses (rc 8 or 9), no run starts" '{ [ $rc -eq 8 ] || [ $rc -eq 9 ]; } && [ ! -e "$OBD3_STATE/done/OBD-CTL_s10.done" ] && ! ls "$OBD3_STATE/done" | grep -q OBD-'

echo "== 3 the STOPBAL wait (DONE proceeds; a busy stop keeps waiting; a non-busy stop proceeds; a held busy marker keeps waiting)"
newbox w1 0; rm -f "$OBD3_STOPBAL_STATE/DONE.txt"
bash "$PKG/obd3_chain.sh" > "$T/w1.out" 2>&1; rc=$?
ok "no marker (STOPBAL still running): the chain waits, gives up with rc 12, touches nothing on the box" '[ $rc -eq 12 ] && stoptxt | grep -q "predecessor-not-ended" && [ "$(calls)" -eq 0 ] && ! [ -e "$OBD3_BUSY_DEV" ]'
newbox w2 0; rm -f "$OBD3_STOPBAL_STATE/DONE.txt"; echo "ZZ STOPBAL-STOP ship box2-busy-or-gpu-not-idle rc=12 :: detail text with rc=5 in the tail" > "$OBD3_STOPBAL_STATE/STOP.txt"
bash "$PKG/obd3_chain.sh" > "$T/w2.out" 2>&1; rc=$?
ok "a busy STOP (head ends in rc=12, detail mentions rc=5): keeps waiting, rc 12, box untouched" '[ $rc -eq 12 ] && stoptxt | grep -q "last state STOP-BUSY" && [ "$(calls)" -eq 0 ]'
newbox w3 0; echo "claimed by someone" > "$OBD3_BUSY_DEV"
bash "$PKG/obd3_chain.sh" > "$T/w3.out" 2>&1; rc=$?
ok "DONE but the shared busy marker exists: keeps waiting, rc 12, the marker is left alone" '[ $rc -eq 12 ] && [ "$(cat "$OBD3_BUSY_DEV" | tr -d "\r\n")" = "claimed by someone" ] && [ "$(calls)" -eq 0 ]'
newbox w4 0; rm -f "$OBD3_STOPBAL_STATE/DONE.txt"; echo "ZZ STOPBAL-STOP run:x stage-crashed rc=5 :: some detail" > "$OBD3_STOPBAL_STATE/STOP.txt"; printf '1' > "$B/py_count.txt"
bash "$PKG/obd3_chain.sh" > "$T/w4.out" 2>&1; rc=$?
ok "a non-busy STOP (rc=5): STOPBAL ended, the chain proceeds (and stops at its own python-running check, rc 12)" 'grep -q "the STOPBAL chain ended (STOP)" "$OBD3_STATE/chain.log" && [ $rc -eq 12 ] && stoptxt | grep -q "box2-busy-or-gpu-not-idle" && [ "$(calls)" -gt 0 ]'
newbox w5 0; rm -f "$OBD3_STOPBAL_STATE/DONE.txt"; echo "ZZ STOPBAL-STOP ship box2-busy-or-gpu-not-idle rc=12" > "$OBD3_STOPBAL_STATE/STOP.txt"; printf '1' > "$B/py_count.txt"
export OBD3_WAIT_POLLS=30 OBD3_WAIT_POLL_S=1
( sleep 4; echo "ZZ STOPBAL-DONE 2026-11-11 00:00:00 commit=x" > "$OBD3_STOPBAL_STATE/DONE.txt"; rm -f "$OBD3_STOPBAL_STATE/STOP.txt" ) &
bash "$PKG/obd3_chain.sh" > "$T/w5.out" 2>&1; rc=$?; wait
ok "a busy STOP, then DONE appears: the chain waited (log) and then proceeded" 'grep -q "waiting for the STOPBAL chain to end" "$OBD3_STATE/chain.log" && grep -q "the STOPBAL chain ended (DONE)" "$OBD3_STATE/chain.log" && [ "$(calls)" -gt 0 ]'
ok "chain.log carries no sentinel prefix of either chain" '! grep -q "ZZ " "$OBD3_STATE/chain.log"'
export OBD3_WAIT_POLLS=0 OBD3_WAIT_POLL_S=0

echo "== 4 one arm refused at its gate (a BUILT lever planted wrong): the chain goes on, ends STOP (never DONE), the other three are banked"
newbox refuse 0; control '{"modes": {"OBD-BAND_s10": "bad_params"}}'
bash "$PKG/obd3_chain.sh" > "$T/refuse.out" 2>&1; rc=$?
ok "chain exits 4" '[ $rc -eq 4 ]'
ok "ARM_REFUSED file for BAND_s10 only, with the sentinel prefix in its own file" '[ -e "$OBD3_STATE/ARM_REFUSED_OBD-BAND_s10.txt" ] && head -c 24 "$OBD3_STATE/ARM_REFUSED_OBD-BAND_s10.txt" | grep -q "^ZZ OBD3-ARM-REFUSED" && [ "$(ls "$OBD3_STATE" | grep -c ARM_REFUSED)" -eq 1 ]'
ok "the other three are banked and done; the refused one is neither" '( for t in OBD-CTL_s10 OBD-CTL_s11 OBD-BAND_s11; do [ -e "$OBD3_PULLS/obd_$t/BANK.done" ] && [ -e "$OBD3_STATE/done/$t.done" ] || exit 1; done; [ ! -e "$OBD3_STATE/done/OBD-BAND_s10.done" ] && [ ! -e "$OBD3_PULLS/obd_OBD-BAND_s10" ] )'
ok "the chain ends with STOP.txt (arms-refused), never DONE.txt" 'stoptxt | grep -q "^ZZ OBD3-STOP runs arms-refused: OBD-BAND_s10 rc=4" && [ ! -e "$OBD3_STATE/DONE.txt" ]'
ok "the refused arm trained nothing (no checkpoint on the box)" '[ ! -e "$B/D/tanitad/runs/v8f_obd/OBD-BAND_s10/ckpt_last.pt" ]'
ok "busy marker handed on" '[ ! -e "$OBD3_BUSY_DEV" ]'

echo "== 5 a pull whose sha256 differs from box 2's stops the chain (rc 7); nothing after it starts and the box keeps the data (no cleanup before the bank is proven)"
newbox pull 0; export FAKE_CORRUPT_GET=ckpt_last.pt
bash "$PKG/obd3_chain.sh" > "$T/pull.out" 2>&1; rc=$?
ok "chain exits 7 with a pull-sha-mismatch STOP" '[ $rc -eq 7 ] && stoptxt | grep -q "^ZZ OBD3-STOP OBD-CTL_s10 pull-sha-mismatch rc=7"'
ok "no later arm started: no BANK.done, no done marker, no stage job for BAND_s10" '[ ! -e "$OBD3_PULLS/obd_OBD-CTL_s10/BANK.done" ] && ! ls "$OBD3_STATE/done" | grep -q "OBD-" && [ ! -e "$B/D/tanitad/obd3/runs/OBD-BAND_s10" ]'
ok "the trained checkpoint and the run folder are still on the box (cleanup never ran)" '[ -s "$B/D/tanitad/runs/v8f_obd/OBD-CTL_s10/ckpt_last.pt" ] && [ -e "$B/D/tanitad/obd3/gate_out/OBD-CTL_s10" ] && [ ! -e "$B/D/tanitad/obd3/runs/OBD-CTL_s10/CLEANUP.json" ]'
ok "the corrupt pull was attempted three times" '[ "$(grep -c "pull of .*ckpt_last.pt attempt" "$T/pull.out")" -eq 3 ]'
export OBD3_STATE="$T/st_pull2"; unset FAKE_CORRUPT_GET
ok "(control) the same box without the corruption banks the arm: the stop was the pull" 'bash "$PKG/obd3_chain.sh" > "$T/pull2.out" 2>&1; [ $? -eq 0 ] && [ -e "$OBD3_PULLS/obd_OBD-CTL_s10/BANK.done" ]'

echo "== 6 failures after the gate stop the chain (no next arm); disk, GPU and python gates"
newbox crash 0; control '{"train_modes": {"OBD-CTL_s10": "crash"}}'
bash "$PKG/obd3_chain.sh" > "$T/crash.out" 2>&1; rc=$?
ok "a crashed training stops the chain (rc 5) before the next arm" '[ $rc -eq 5 ] && stoptxt | grep -q "job-or-transfer-failed" && [ ! -e "$B/D/tanitad/obd3/runs/OBD-BAND_s10" ]'
newbox post 0; control '{"train_modes": {"OBD-CTL_s10": "wrong_seed"}}'
bash "$PKG/obd3_chain.sh" > "$T/post.out" 2>&1; rc=$?
ok "a checkpoint with the wrong seed fails the post check (rc 5), nothing is banked" '[ $rc -eq 5 ] && stoptxt | grep -q "post check failed" && [ ! -e "$OBD3_PULLS/obd_OBD-CTL_s10/BANK.done" ]'
newbox disk 0; printf '3' > "$B/free_gb.txt"
bash "$PKG/obd3_chain.sh" > "$T/disk.out" 2>&1; rc=$?
ok "disk below the run floor: STOP disk-low (rc 6), no job started for a run" '[ $rc -eq 6 ] && stoptxt | grep -q "disk-low" && [ ! -e "$B/D/tanitad/obd3/runs/OBD-CTL_s10/stage.cmd" ]'
newbox gpu 3600
bash "$PKG/obd3_chain.sh" > "$T/gpu.out" 2>&1; rc=$?
ok "GPU at 3600 MiB (>= the 3500 default): STOP rc 12, no run job" '[ $rc -eq 12 ] && stoptxt | grep -q "box2-busy-or-gpu-not-idle" && [ ! -e "$B/D/tanitad/obd3/runs/OBD-CTL_s10/stage.cmd" ]'
newbox gpu2 3600; export OBD3_GPU_IDLE_MIB=4000
bash "$PKG/obd3_chain.sh" > "$T/gpu2.out" 2>&1; rc=$?
ok "OBD3_GPU_IDLE_MIB=4000 lets the same 3600 MiB pass" '[ $rc -eq 0 ]'
unset OBD3_GPU_IDLE_MIB
newbox py 0; printf '2' > "$B/py_count.txt"
bash "$PKG/obd3_chain.sh" > "$T/py.out" 2>&1; rc=$?
ok "another python.exe running: STOP rc 12" '[ $rc -eq 12 ] && stoptxt | grep -q "box2-busy-or-gpu-not-idle"'
newbox resume 0; printf '2' > "$B/py_count.txt"; bash "$PKG/obd3_chain.sh" > /dev/null 2>&1; printf '0' > "$B/py_count.txt"
bash "$PKG/obd3_chain.sh" > "$T/resume0.out" 2>&1; rc=$?
ok "a leftover STOP.txt refuses a plain rerun (rc 2); OBD3_RESUME=1 moves it aside and the chain completes" '[ $rc -eq 2 ] && OBD3_RESUME=1 bash "$PKG/obd3_chain.sh" > "$T/resume1.out" 2>&1 && [ -e "$OBD3_STATE/DONE.txt" ] && ls "$OBD3_STATE" | grep -q "^STOP\\."'

echo "== 7 commit, tag and package refusals; the transport variables fail closed"
newbox ref 0
( export OBD3_COMMIT=; bash "$PKG/obd3_chain.sh" > /dev/null 2>&1; echo $? > "$T/rc_nocommit" ); ok "no commit: rc 8" '[ "$(cat "$T/rc_nocommit")" -eq 8 ]'
rm -f "$OBD3_STATE/STOP.txt"; ( export OBD3_COMMIT=ABCDEF; bash "$PKG/obd3_chain.sh" > /dev/null 2>&1; echo $? > "$T/rc_hex" ); ok "a non-hex commit: rc 8" '[ "$(cat "$T/rc_hex")" -eq 8 ]'
rm -f "$OBD3_STATE/STOP.txt"; ( export OBD3_COMMIT=abc123; bash "$PKG/obd3_chain.sh" > /dev/null 2>&1; echo $? > "$T/rc_short" ); ok "a short commit: rc 8" '[ "$(cat "$T/rc_short")" -eq 8 ]'
rm -f "$OBD3_STATE/STOP.txt"; ( export OBD3_CODE_COMMIT="$COMMIT"; unset OBD3_DRY; bash "$PKG/obd3_chain.sh" > /dev/null 2>&1; echo $? > "$T/rc_hook" ); ok "the code-of-record test hook without OBD3_DRY: rc 8" '[ "$(cat "$T/rc_hook")" -eq 8 ] && stoptxt | grep -q "test-hook-in-real-run"'
rm -f "$OBD3_STATE/STOP.txt"; ( export OBD3_SSH="$REALC" OBD3_SFTP="$REALC"; unset OBD3_DRY; bash "$PKG/obd3_chain.sh" > /dev/null 2>&1; echo $? > "$T/rc_notcor" ); ok "a real (non-stub) client with a commit that is not the code of record: rc 8 before any call" '[ "$(cat "$T/rc_notcor")" -eq 8 ] && stoptxt | grep -q "commit-not-the-code-of-record" && [ ! -e "$T/BREACH" ]'
rm -f "$OBD3_STATE/STOP.txt"; ( export OBD3_SSH=; bash "$PKG/obd3_chain.sh" > /dev/null 2>&1; echo $? > "$T/rc_unset" ); ok "OBD3_SSH empty: rc 16" '[ "$(cat "$T/rc_unset")" -eq 16 ]'
rm -f "$OBD3_STATE/STOP.txt"; ( export OBD3_SFTP="sftp"; bash "$PKG/obd3_chain.sh" > /dev/null 2>&1; echo $? > "$T/rc_rel" ); ok "a relative OBD3_SFTP: rc 16" '[ "$(cat "$T/rc_rel")" -eq 16 ]'
rm -f "$OBD3_STATE/STOP.txt"; ( export OBD3_SSH="$T/none.sh"; bash "$PKG/obd3_chain.sh" > /dev/null 2>&1; echo $? > "$T/rc_abs" ); ok "an absent OBD3_SSH file: rc 16" '[ "$(cat "$T/rc_abs")" -eq 16 ]'
rm -f "$OBD3_STATE/STOP.txt"; ( export OBD3_SSH="$REALC"; bash "$PKG/obd3_chain.sh" > /dev/null 2>&1; echo $? > "$T/rc_real" ); ok "OBD3_DRY=1 (test mode) with a client that carries no stub marker: rc 16" '[ "$(cat "$T/rc_real")" -eq 16 ]'
rm -f "$OBD3_STATE/STOP.txt"; ( unset OBD3_DRY; bash "$PKG/obd3_chain.sh" > /dev/null 2>&1; echo $? > "$T/rc_nodry" ); ok "a stub transport without OBD3_DRY=1: rc 16" '[ "$(cat "$T/rc_nodry")" -eq 16 ]'
ok "none of these refusals touched the box" '[ "$(calls)" -eq 0 ]'
ok "an unknown tag is refused by the run stage (rc 23) and by the checker (rc 2); an old OBD tag too" 'bash "$PKG/stages/obd3_run.sh" OBD-NULL_s0 > /dev/null 2>&1; [ $? -eq 23 ] && "$PYV" -I "$PKG/obd3_check.py" argv --tag OBD-CTL_s0 > /dev/null; [ $? -eq 2 ]'
ok "the checker has no pilot, no permit, no project, no parity" '( for c in project permit parity; do "$PYV" -I "$PKG/obd3_check.py" $c > /dev/null; [ $? -eq 2 ] || exit 1; done; "$PYV" -I "$PKG/obd3_check.py" argv --tag OBD-CTL_s10 --pilot > /dev/null; [ $? -eq 2 ] )'
newbox pkg 0; ok "an incomplete package (copy without the frozen argv file) is refused by the chain (rc 8)" 'mkdir -p "$T/pkgcopy"; cp -r "$PKG/stages" "$PKG/jobs" "$PKG/obd3_check.py" "$PKG/make_argvs_seed3.py" "$PKG/obd3_chain.sh" "$T/pkgcopy/"; bash "$T/pkgcopy/obd3_chain.sh" > /dev/null 2>&1; [ $? -eq 8 ]'

echo "== 8 hygiene"
ok "no sentinel string is printed by any script outside its own file (grep over the package scripts: only the chain's own echo-to-file lines)" '[ "$(grep -h "ZZ OBD3" "$PKG"/obd3_chain.sh "$PKG"/stages/*.sh "$PKG"/jobs/*.cmd | grep -v "^#" | grep -v "> \"\$STATE/" | wc -l | tr -d " ")" -eq 0 ]'
ok "no package file uses a Windows reserved device name" '! find "$PKG" -type f | grep -iE "/(aux|con|prn|nul)(\\.|$)"'
ok "no subprocess text=True without encoding= in the python files" '! grep -n "text=True" "$PKG"/*.py "$PKG"/tests/*.py | grep -v encoding'
ok "the OBD chain package (theory/obd/box2) is byte-identical after all scenarios" '[ "$(treehash "$ORIG")" = "$ORIG_H0" ]'
ok "no breach through PATH, whole run" '[ ! -e "$T/BREACH" ]'
echo "DRYRUN ok=$OK failed=$BAD"
[ "$BAD" -eq 0 ]
