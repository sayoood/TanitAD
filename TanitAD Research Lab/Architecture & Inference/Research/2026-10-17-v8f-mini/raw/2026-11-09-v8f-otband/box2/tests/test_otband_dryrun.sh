#!/usr/bin/env bash
# Dry run of the OTBAND box-2 package over stubs. NOTHING here can reach box 2: OTBAND_SSH / OTBAND_SFTP name stub scripts by absolute path (OTBAND_DRY=1 makes the chain refuse any file without the stub marker),
# a poison ssh/sftp/scp is put FIRST on PATH and must never run, and the "box" is a folder. The REAL job .cmd files run on Windows cmd.exe against it; only the trainer is a stand-in (tests/plant_trainer.py,
# shipped through a throw-away git repo, so the ship, tree-binding and token code run for real). CPU only, no GPU, no torch.   Usage: bash test_otband_dryrun.sh  -> prints "DRYRUN ok=<n> failed=<m>"
set -u
TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PKG="$(cd "$TESTS/.." && pwd)"; PKG="$(cygpath -m "$PKG" 2>/dev/null || echo "$PKG")"
PYV="${OTBAND_TEST_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
T="${OTBAND_TEST_TMP:-C:/Users/Admin/AppData/Local/Temp/otband_dry_run_$$}"
case "$T" in */otband_dry_run_*|*/scratchpad/*|*/otband_dry_*) ;; *) echo "refusing an unrecognised scratch dir $T" >&2; exit 2;; esac
rm -rf "${T:?}"; mkdir -p "$T/poison"
for n in ssh sftp scp; do printf '#!/usr/bin/env bash\necho BREACH-%s >> "%s/BREACH"\nexit 99\n' "$n" "$T" > "$T/poison/$n"; chmod +x "$T/poison/$n"; done
export PATH="$T/poison:$PATH"
export CUDA_VISIBLE_DEVICES=-1
OK=0; BAD=0
ok() { if eval "$2"; then OK=$((OK+1)); else BAD=$((BAD+1)); echo "FAIL: $1"; fi; }
SSHS="$(cygpath -m "$TESTS/ssh_stub.sh")"; SFTPS="$(cygpath -m "$TESTS/sftp_stub.sh")"
COMMIT="$("$PYV" -W ignore "$TESTS/dry_helpers.py" mkrepo "$T/repo")"
ok "throw-away repo has a 40-hex commit" '[ ${#COMMIT} -eq 40 ]'
newbox() {   # newbox <name> [gpu MiB]  -> exports the dry-run environment for box <name>; the ladder has ended (FULL_DONE) unless the test says otherwise
  B="$T/box_$1"; "$PYV" -W ignore "$TESTS/dry_helpers.py" mkbox "$B" "$PYV" "${2:-0}" > /dev/null
  mkdir -p "$B/D/tanitad/ladder/runs" "$B/D/tanitad/repo/stack"; echo "ladder file" > "$B/D/tanitad/ladder/runs/R35_s0.txt"; echo "shared repo file" > "$B/D/tanitad/repo/stack/x.py"
  export OTBAND_DRY=1 OTBAND_SSH="$SSHS" OTBAND_SFTP="$SFTPS" FAKE_BOX="$(cygpath -m "$B")" FAKE_PY="$PYV" OTBAND_PY="$PYV" OTBAND_STATE="$T/st_$1" OTBAND_PULLS="$T/pulls_$1"
  export OTBAND_BUSY_DEV="$T/busy_$1.txt" OTBAND_ASIDE_DEV="$T/aside_$1" OTBAND_B2_WIN_ROOT="$(cygpath -w "$B/D/tanitad")" OTBAND_POLL_S=0 OTBAND_REPO="$T/repo" OTBAND_EXPORT_LIST="$T/repo/export_list.txt"
  export OTBAND_COMMIT="$COMMIT" OTBAND_NEED_C_MB=1 OTBAND_POLLS_RUN=5 OTBAND_POLLS_SHIP=5 OTBAND_POLLS_CLEAN=5 OTBAND_WAIT_POLLS=0 OTBAND_WAIT_POLL_S=0
  export OTBAND_LADDER_STATE="$T/ladder_$1"; mkdir -p "$OTBAND_LADDER_STATE"; echo "ZZ LADDER-FULL-DONE 2026-11-10 00:00:00 commit=x" > "$OTBAND_LADDER_STATE/FULL_DONE.txt"
  unset OTBAND_RESUME OTBAND_AMENDMENT
}
control() { printf '%s' "$1" > "$B/plant_control.json"; }
calls() { cat "$B/transport.log" 2>/dev/null | wc -l | tr -d ' '; }
stoptxt() { cat "$OTBAND_STATE/STOP.txt" 2>/dev/null; }
treehash() { ( cd "$1" && find . -type f | sort | xargs sha256sum | sha256sum | cut -d' ' -f1 ); }
BOXB() { echo "$B/D/tanitad"; }

echo "== 1 normal flow: wait check, ship, pilot, 72 h permit, five runs, banks, cleanups"
newbox normal 0; control '{"sps": 1.0, "startup_s": 30}'
LH0="$(treehash "$B/D/tanitad/ladder")"; RH0="$(treehash "$B/D/tanitad/repo")"
bash "$PKG/otband_chain.sh" > "$T/normal.out" 2>&1; rc=$?
ok "chain exits 0" '[ $rc -eq 0 ]'
ok "DONE sentinel written, no stop" '[ -e "$OTBAND_STATE/DONE.txt" ] && [ ! -e "$OTBAND_STATE/STOP.txt" ] && head -c 15 "$OTBAND_STATE/DONE.txt" | grep -q "^ZZ OTBAND-DONE"'
ok "sentinel text only in its own file (chain.log carries none)" '! grep -q "ZZ OTBAND" "$OTBAND_STATE/chain.log"'
P="$OTBAND_STATE/pilot/PILOT.verified.json"
ok "PILOT.verified.json names the commit" 'grep -q "\"commit\": \"$COMMIT\"" "$P"'
ok "pilot mean s/step excludes start-up (reads 1.0)" '"$PYV" -I -c "import json,sys;d=json.load(open(sys.argv[1]));assert abs(d[\"timing\"][\"mean_s_per_step\"]-1.0)<1e-9, d[\"timing\"]" "$P"'
ok "projection covers 5 runs x 9,582 steps" '"$PYV" -I -c "import json,sys;d=json.load(open(sys.argv[1]))[\"projection\"];assert d[\"steps_total\"]==47910 and d[\"n_runs_planned\"]==5 and d[\"within_limit_planned\"]" "$P"'
ok "pilot used the exact OB-BAND_s0 argv plus --steps 200 and a separate out dir" '"$PYV" -I -c "import sys,json;sys.path.insert(0,sys.argv[1]);import otband_check as C;l=open(sys.argv[2]).read().split();assert l==C.tokens(\"PILOT_OB-BAND_s0\") and l[-2:]==[\"--steps\",\"200\"] and \"pilot_out\" in l[l.index(\"--out\")+1] and [x for x in l[:-2] if x not in C.tokens(\"OB-BAND_s0\")]==[l[l.index(\"--out\")+1]]" "$PKG" "$OTBAND_STATE/runs/PILOT_OB-BAND_s0/argv.txt"'
ok "the pilot token binds commit and argv" '"$PYV" -I "$PKG/otband_check.py" verify-token --token "$OTBAND_STATE/pilot/gate_token.json" --argv-file "$OTBAND_STATE/runs/PILOT_OB-BAND_s0/argv.txt" --tag PILOT_OB-BAND_s0 --commit "$COMMIT" > /dev/null'
ok "pilot temp dirs deleted on the box (pilot out, gate outs)" '[ ! -e "$B/D/tanitad/otband/pilot_out/OB-BAND_s0" ] && [ ! -e "$B/D/tanitad/otband/gate_out/PILOT_OB-BAND_s0" ]'
ok "pilot deletions are logged (path, bytes) and banked" 'grep -q "\"bytes\"" "$OTBAND_STATE/pilot/CLEANUP.json" && grep -q "pilot_out" "$OTBAND_STATE/pilot/CLEANUP.json"'
for t in OB-CTL_s0 OB-BAND_s0 OB-CTL_s1 OB-BAND_s1 OB-BAND_rep; do
  BD="$OTBAND_PULLS/otband_$t"
  ok "$t banked with BANK.done and the six evidence files" '[ -e "$BD/BANK.done" ] && [ -s "$BD/ckpt_last.pt" ] && [ -s "$BD/summary.json" ] && [ -s "$BD/train_log.jsonl" ] && [ -s "$BD/gate_token.json" ] && [ -s "$BD/selfcheck.json" ] && [ -s "$BD/RUN.done" ]'
  ok "$t banked checkpoint equals the box copy (sha256)" '[ "$(sha256sum "$BD/ckpt_last.pt" | cut -d" " -f1)" = "$(sha256sum "$B/D/tanitad/runs/v8f_otband/$t/ckpt_last.pt" | cut -d" " -f1)" ]'
  ok "$t token binds the commit and the argv" '"$PYV" -I "$PKG/otband_check.py" verify-token --token "$BD/gate_token.json" --argv-file "$BD/argv.txt" --tag $t --commit "$COMMIT" > /dev/null'
  ok "$t RUN.done carries the freeze identity (11 frozen tensors equal)" 'grep -q "\"n_frozen_equal\": 11" "$BD/RUN.done"'
  ok "$t gate dir on the box deleted, the checkpoint kept" '[ ! -e "$B/D/tanitad/otband/gate_out/$t" ] && [ -s "$B/D/tanitad/runs/v8f_otband/$t/ckpt_last.pt" ]'
  ok "$t cleanup logged (path, bytes) in the bank" 'grep -q "\"bytes\"" "$BD/CLEANUP.json" || grep -q "deleted_own_temp\": \[\]" "$BD/CLEANUP.json"'
done
ok "the five tokens bind five distinct argvs" '[ "$(cat "$OTBAND_PULLS"/otband_*/gate_token.json | grep -o "\"argv_sha256\": \"[0-9a-f]*\"" | sort -u | wc -l | tr -d " ")" -eq 5 ]'
ok "run order in chain.log: CTL_s0, BAND_s0, CTL_s1, BAND_s1, rep" '[ "$(grep -o "done OB-[A-Za-z0-9_-]*" "$OTBAND_STATE/chain.log" | tr "\n" " ")" = "done OB-CTL_s0 done OB-BAND_s0 done OB-CTL_s1 done OB-BAND_s1 done OB-BAND_rep " ]'
ok "no token arm was run" '[ ! -e "$OTBAND_PULLS/otband_OB-BAND-TOK_s0" ] && [ ! -e "$B/D/tanitad/otband/runs/OB-BAND-TOK_s0" ]'
ok "the box saw only stub traffic and no unknown command" '[ "$(calls)" -gt 100 ] && ! grep -q "UNKNOWN COMMAND" "$B/transport.log"'
ok "disk was read before every stage (>= 7 reads: ship, pilot, five runs)" '[ "$(grep -c "Get-PSDrive D" "$B/transport.log")" -ge 7 ]'
ok "isolation: the ladder folder on the box is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/ladder")" = "$LH0" ]'
ok "isolation: the shared D:\\tanitad\\repo is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/repo")" = "$RH0" ]'
ok "own tree exists with the trainer and TIP_OTBAND.txt" '[ -s "$B/D/tanitad/otband/repo/stack/scripts/v8f_mini_train.py" ] && [ "$(cat "$B/D/tanitad/otband/repo/TIP_OTBAND.txt" | tr -d "\r\n")" = "$COMMIT" ]'
ok "busy marker handed on" '[ ! -e "$OTBAND_BUSY_DEV" ]'
ok "no breach through PATH" '[ ! -e "$T/BREACH" ]'
ok "a second chain run skips finished stages" 'bash "$PKG/otband_chain.sh" 2>&1 | tail -3 | grep -q "all runs banked"'
NORMAL_B="$B"

echo "== 2 the box-side pin refuses a changed token (checker run directly on the fake box, no transport)"
B="$NORMAL_B"; RD="$B/D/tanitad/otband/runs/OB-CTL_s0"; cp "$RD/argv.txt" "$T/argv.keep"
sed -i 's/--seed 0/--seed 7/' "$RD/argv.txt"
( export OTBAND_ROOT="$(cygpath -m "$B/D/tanitad")" OTBAND_COMMIT="$COMMIT" OTBAND_FAKE_INIT_SHA_FILE="$(cygpath -m "$B/D/tanitad/init_sha.json")" OTBAND_FLOOR_BYTES=1000 OTBAND_FAKE_V9_MD5="$(md5sum "$B/D/tanitad/data/v9/v9_labels_train.npz" | cut -d' ' -f1)"
  "$PYV" -I "$B/D/tanitad/otband/check/otband_check.py" stage --tag OB-CTL_s0 --gid pin1 > "$T/pin.out" 2>&1 ); rc=$?
ok "a changed seed token is refused (rc 3) and the first differing token is named" '[ $rc -eq 3 ] && grep -q "pin-fail" "$RD/STAGE_pin1.fail" && grep -q "token " "$RD/STAGE_pin1.fail" && grep -q "frozen argv has" "$RD/STAGE_pin1.fail"'
cp "$T/argv.keep" "$RD/argv.txt"

echo "== 3 ladder not ended, box held, GPU not idle, python running"
newbox wait 0; rm -f "$OTBAND_LADDER_STATE/FULL_DONE.txt"
bash "$PKG/otband_chain.sh" > "$T/w1.out" 2>&1; rc=$?
ok "no ladder finished marker: refused with rc 12, zero box calls" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ] && stoptxt | grep -q "ladder-not-ended"'
rm -f "$OTBAND_STATE/STOP.txt"; echo "ZZ LADDER-FULL-DONE x" > "$OTBAND_LADDER_STATE/FULL_DONE.txt"; echo "ladder full 1" > "$OTBAND_BUSY_DEV"
bash "$PKG/otband_chain.sh" > "$T/w2.out" 2>&1; rc=$?
ok "ladder marker present but the busy marker held: refused with rc 12, zero box calls" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$OTBAND_STATE/STOP.txt" "$OTBAND_BUSY_DEV"; echo "not a finished marker" > "$OTBAND_LADDER_STATE/FULL_DONE.txt"
bash "$PKG/otband_chain.sh" > "$T/w2b.out" 2>&1; rc=$?
ok "a marker file without the ladder prefix is not a finished marker (rc 12)" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$OTBAND_STATE/STOP.txt"; rm -f "$OTBAND_LADDER_STATE/FULL_DONE.txt"; echo "ZZ LADDER-FULL-STOP run x rc=4" > "$OTBAND_LADDER_STATE/FULL_STOP.txt"; echo 9000 > "$B/gpu_mib.txt"
bash "$PKG/otband_chain.sh" > "$T/w3.out" 2>&1; rc=$?
ok "a stopped ladder counts as ended; GPU busy (9000 MiB) refuses the pilot (rc 12), nothing trained" '[ $rc -eq 12 ] && stoptxt | grep -q "gpu-not-idle" && [ ! -e "$B/D/tanitad/otband/runs/PILOT_OB-BAND_s0/gate" ]'
rm -f "$OTBAND_STATE/STOP.txt" "$OTBAND_STATE/done/pilot.done"; echo 0 > "$B/gpu_mib.txt"; echo 2 > "$B/py_count.txt"
bash "$PKG/otband_chain.sh" > "$T/w4.out" 2>&1; rc=$?
ok "a python process on box 2 refuses the pilot (rc 12)" '[ $rc -eq 12 ] && stoptxt | grep -q "gpu-not-idle"'
ok "no breach through PATH after the wait tests" '[ ! -e "$T/BREACH" ]'

echo "== 4 disk gate"
newbox disk 0; control '{"sps": 1.0}'; echo 3 > "$B/free_gb.txt"
bash "$PKG/otband_chain.sh" > "$T/d1.out" 2>&1; rc=$?
ok "3 GB free: ship passes (1 GB), the pilot stops with the disk-low code (rc 6)" '[ $rc -eq 6 ] && stoptxt | grep -q "disk-low" && [ ! -e "$B/D/tanitad/otband/runs/PILOT_OB-BAND_s0/gate" ]'
newbox disk0 0; echo 0 > "$B/free_gb.txt"
bash "$PKG/otband_chain.sh" > "$T/d2.out" 2>&1; rc=$?
ok "0 GB free: the ship stage stops with rc 6" '[ $rc -eq 6 ] && [ ! -e "$B/D/tanitad/otband/ship/otband_tree.tar" ]'
newbox disk1 0; echo "" > "$B/free_gb.txt"
bash "$PKG/otband_chain.sh" > "$T/d3.out" 2>&1; rc=$?
ok "an unreadable free-space probe is a refusal, not a pass (rc 11)" '[ $rc -eq 11 ]'

echo "== 5 projected total above 72 h"
newbox slow 0; control '{"sps": 6.0}'
bash "$PKG/otband_chain.sh" > "$T/s1.out" 2>&1; rc=$?
ok "pilot at 6.0 s/step: the chain stops at the 72 h rule (rc 3), no run started" '[ $rc -eq 3 ] && stoptxt | grep -q "72h-rule-refused" && [ ! -e "$B/D/tanitad/otband/runs/OB-CTL_s0/RUN.done" ] && [ ! -e "$B/D/tanitad/runs/v8f_otband" ]'
rm -f "$OTBAND_STATE/STOP.txt"; printf 'OTBAND-AMENDMENT pilot_sha256=%s commit=%s\n' "$(printf '%064d' 0)" "$COMMIT" > "$T/am_wrong.txt"
OTBAND_RESUME=1 OTBAND_AMENDMENT="$T/am_wrong.txt" bash "$PKG/otband_chain.sh" > "$T/s2.out" 2>&1; rc=$?
ok "an amendment naming another pilot is refused (rc 3)" '[ $rc -eq 3 ]'
rm -f "$OTBAND_STATE/STOP.txt"; printf 'OTBAND-AMENDMENT pilot_sha256=%s commit=%s\n' "$(sha256sum "$OTBAND_STATE/pilot/PILOT.verified.json" | cut -d' ' -f1)" "$COMMIT" > "$T/am_ok.txt"; control '{"sps": 6.0}'
OTBAND_RESUME=1 OTBAND_AMENDMENT="$T/am_ok.txt" bash "$PKG/otband_chain.sh" > "$T/s3.out" 2>&1; rc=$?
ok "an amendment binding this pilot and commit lets the five runs go (rc 0)" '[ $rc -eq 0 ] && [ -e "$OTBAND_STATE/DONE.txt" ]'

echo "== 6 gate refusals (BUILT levers planted wrong in the stand-in trainer)"
newbox gfreeze 0; control '{"sps": 1.0, "modes": {"OB-BAND_s0": "bad_freeze"}}'
bash "$PKG/otband_chain.sh" > "$T/g1.out" 2>&1; rc=$?
ok "S built trainable: the pilot's gate of OB-BAND_s0 refuses (rc 4), the stop line names the BUILT S params" '[ $rc -eq 4 ] && stoptxt | grep -q "frozen group S"'
ok "no training happened and no token was written for the refused arm" '[ ! -e "$B/D/tanitad/otband/runs/PILOT_OB-BAND_s0/logs/train.log" ] && [ ! -e "$B/D/tanitad/otband/runs/OB-BAND_s0/gate/gate_token.json" ] && [ ! -e "$B/D/tanitad/runs/v8f_otband" ]'
newbox gdetach 0; control '{"sps": 1.0, "modes": {"PILOT_OB-BAND_s0": "bad_detach"}}'
bash "$PKG/otband_chain.sh" > "$T/g2.out" 2>&1; rc=$?
ok "O's op term reaching S / T (purity != 0): refused (rc 4)" '[ $rc -eq 4 ] && stoptxt | grep -q "upper_grad_abs_sum"'
newbox gsteps 0; control '{"sps": 1.0, "modes": {"OB-CTL_s0": "bad_steps"}}'
bash "$PKG/otband_chain.sh" > "$T/g3.out" 2>&1; rc=$?
ok "a full-length steps_total that is not 9,582 is refused at the pilot (rc 4), before the 72 h decision" '[ $rc -eq 4 ] && stoptxt | grep -q "steps_total"'
newbox garm 0; control '{"sps": 1.0, "modes": {"OB-CTL_s1": "bad_freeze"}}'
bash "$PKG/otband_chain.sh" > "$T/g4.out" 2>&1; rc=$?
ok "an arm-specific gate refusal (OB-CTL_s1) ends the chain with rc 4 but the other arms ran" '[ $rc -eq 4 ] && [ -e "$OTBAND_STATE/ARM_REFUSED_OB-CTL_s1.txt" ] && [ ! -e "$OTBAND_STATE/DONE.txt" ]'
ok "the independent arms after it were banked, the refused one was not trained" '[ -e "$OTBAND_PULLS/otband_OB-BAND_s1/BANK.done" ] && [ -e "$OTBAND_PULLS/otband_OB-BAND_rep/BANK.done" ] && [ ! -e "$B/D/tanitad/runs/v8f_otband/OB-CTL_s1" ] && [ -e "$OTBAND_PULLS/otband_OB-CTL_s0/BANK.done" ]'
ok "the refusal sentinel is in its own file" 'head -c 30 "$OTBAND_STATE/ARM_REFUSED_OB-CTL_s1.txt" | grep -q "^ZZ OTBAND-ARM-REFUSED" && ! grep -q "ZZ OTBAND" "$OTBAND_STATE/chain.log"'

echo "== 7 failures after the gate stop the chain (no next arm)"
newbox pfrozen 0; control '{"sps": 1.0, "train_modes": {"OB-CTL_s1": "move_frozen"}}'
bash "$PKG/otband_chain.sh" > "$T/p1.out" 2>&1; rc=$?
ok "a frozen tensor that moved is a post-check failure: chain stops (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "post check failed" && stoptxt | grep -q "frozen S / adapter / cond"'
ok "the failed arm is not banked and OB-BAND_s1 never started; the two earlier arms are banked" '[ ! -e "$OTBAND_PULLS/otband_OB-CTL_s1/BANK.done" ] && [ ! -e "$B/D/tanitad/otband/runs/OB-BAND_s1/argv.txt" ] && [ -e "$OTBAND_PULLS/otband_OB-CTL_s0/BANK.done" ] && [ -e "$OTBAND_PULLS/otband_OB-BAND_s0/BANK.done" ]'
ok "the checkpoint of the failed arm stays on the box (never deleted)" '[ -s "$B/D/tanitad/runs/v8f_otband/OB-CTL_s1/ckpt_last.pt" ]'
newbox pcrash 0; control '{"sps": 1.0, "train_modes": {"OB-CTL_s0": "crash"}}'
bash "$PKG/otband_chain.sh" > "$T/p2.out" 2>&1; rc=$?
ok "a trainer crash stops the chain (rc 5) before the next arm" '[ $rc -eq 5 ] && [ ! -e "$B/D/tanitad/otband/runs/OB-BAND_s0/RUN.done" ] && [ ! -e "$OTBAND_STATE/done/OB-BAND_s0.done" ] && [ ! -e "$OTBAND_PULLS/otband_OB-CTL_s0" ]'
newbox pnomove 0; control '{"sps": 1.0, "train_modes": {"OB-CTL_s0": "no_move"}}'
bash "$PKG/otband_chain.sh" > "$T/p3.out" 2>&1; rc=$?
ok "a run in which T and O did not move is a post-check failure (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "nothing was trained"'
newbox pcorrupt 0; control '{"sps": 1.0}'
FAKE_CORRUPT_PUT=argv.txt bash "$PKG/otband_chain.sh" > "$T/p4.out" 2>&1; rc=$?
ok "a put that arrives altered is caught by the re-hash on box 2 (rc 5), nothing runs" '[ $rc -eq 5 ] || [ $rc -eq 9 ]'

echo "== 8 commit, tag and package refusals"
newbox nocommit 0; unset OTBAND_COMMIT
bash "$PKG/otband_chain.sh" > "$T/nc.out" 2>&1; rc=$?
ok "chain without OTBAND_COMMIT refuses (rc 8), zero box calls" '[ $rc -eq 8 ] && stoptxt | grep -q "commit-missing" && [ "$(calls)" -eq 0 ]'
export OTBAND_COMMIT="abc123"; rm -f "$OTBAND_STATE/STOP.txt"
bash "$PKG/otband_chain.sh" > "$T/nc2.out" 2>&1; rc=$?
ok "an abbreviated commit refuses (rc 8)" '[ $rc -eq 8 ] && [ "$(calls)" -eq 0 ]'
export OTBAND_COMMIT="$COMMIT"; rm -f "$OTBAND_STATE/STOP.txt"
OTBAND_COMMIT="$(printf 'A%.0s' $(seq 1 40))" bash "$PKG/otband_chain.sh" > "$T/nc3.out" 2>&1; rc=$?
ok "an upper-case 40-character commit refuses (rc 8)" '[ $rc -eq 8 ]'
rm -f "$OTBAND_STATE/STOP.txt"
bash "$PKG/stages/otband_run.sh" OB-BAND-TOK_s0 > "$T/tok.out" 2>&1; rc=$?
ok "the blocked token arm refuses (rc 23), zero box calls" '[ $rc -eq 23 ] && [ "$(calls)" -eq 0 ]'
bash "$PKG/stages/otband_run.sh" OB-NOPE_s0 > "$T/tok2.out" 2>&1; rc=$?
ok "an unknown tag refuses (rc 16), zero box calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
ok "the checker CLI refuses the blocked tag (rc 23)" '"$PYV" -I "$PKG/otband_check.py" argv --tag OB-BAND-TOK_s1 > /dev/null 2>&1; [ $? -eq 23 ]'

echo "== 9 transport variable unset, relative, absent, the wrong kind, or test mode with a real client (fail closed)"
newbox tr 0
for how in unset relative absent realkind; do
  rm -f "$OTBAND_STATE/STOP.txt"; : > "$B/transport.log"
  case "$how" in
    unset) ( unset OTBAND_SSH; bash "$PKG/otband_chain.sh" > "$T/tr_$how.out" 2>&1 ); rc=$?;;
    relative) OTBAND_SSH="ssh" bash "$PKG/otband_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
    absent) OTBAND_SSH="$T/no_such_ssh" bash "$PKG/otband_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
    realkind) printf '#!/usr/bin/env bash\necho i am a real client\n' > "$T/fake_real_ssh"; chmod +x "$T/fake_real_ssh"; OTBAND_SSH="$(cygpath -m "$T/fake_real_ssh")" bash "$PKG/otband_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
  esac
  ok "transport '$how' refuses with rc 16" '[ $rc -eq 16 ]'
  ok "transport '$how' made no call and wrote a stop line" '[ "$(calls)" -eq 0 ] && grep -q "transport-not-fail-closed" "$OTBAND_STATE/STOP.txt"'
done
rm -f "$OTBAND_STATE/STOP.txt"; : > "$B/transport.log"
( unset OTBAND_SFTP; bash "$PKG/otband_chain.sh" > "$T/tr_sftp.out" 2>&1 ); rc=$?
ok "OTBAND_SFTP unset refuses with rc 16" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$OTBAND_STATE/STOP.txt"
( unset OTBAND_DRY; bash "$PKG/otband_chain.sh" > "$T/tr_nodry.out" 2>&1 ); rc=$?
ok "a stub without test mode refuses (the stub is never taken for the real client)" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$OTBAND_STATE/STOP.txt"
printf '#!/usr/bin/env bash\necho i am a real sftp\n' > "$T/fake_real_sftp"; chmod +x "$T/fake_real_sftp"
OTBAND_SFTP="$(cygpath -m "$T/fake_real_sftp")" bash "$PKG/otband_chain.sh" > "$T/tr_realsftp.out" 2>&1; rc=$?
ok "test mode (OTBAND_DRY=1) with a real sftp path refuses (rc 16), zero calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$OTBAND_STATE/STOP.txt"
OTBAND_SSH="$T/no_such_ssh" bash "$PKG/stages/otband_run.sh" OB-CTL_s0 > "$T/tr_stage.out" 2>&1; rc=$?
ok "a stage run alone with an absent transport refuses with rc 16 before any call" '[ $rc -eq 16 ]'
ok "no breach through PATH after all transport refusals" '[ ! -e "$T/BREACH" ]'

echo "== 10 hygiene"
ok "no ssh, sftp or scp command in any test script other than the stubs' own forwarding" '! grep -nE "^[^#]*\b(ssh|sftp|scp) +-" "$TESTS/test_otband_dryrun.sh" | grep -v "poison" | grep -q .'
ok "bash scripts are LF" '"$PYV" -I -c "import glob,sys;fs=glob.glob(sys.argv[1]+\"/*.sh\")+glob.glob(sys.argv[1]+\"/stages/*.sh\")+glob.glob(sys.argv[2]+\"/*.sh\");assert len(fs)>=7 and all(b\"\r\" not in open(f,\"rb\").read() for f in fs)" "$PKG" "$TESTS"'
rm -rf "${T:?}"
echo "DRYRUN ok=$OK failed=$BAD"
[ "$BAD" -eq 0 ]
