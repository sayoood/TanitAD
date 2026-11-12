#!/usr/bin/env bash
# Dry run of the STOPPERC stage-1 chain (stopperc_chain.sh) over stubs. NOTHING here can reach box 2: STOPPERC_SSH / STOPPERC_SFTP name stub scripts by absolute path (STOPPERC_DRY=1 makes the sourced STOPBAL common script refuse any
# file without the stub marker), a poison ssh/sftp/scp is put FIRST on PATH and must never run, and the "box" is a folder. The REAL job .cmd files run on Windows cmd.exe against it, the REAL checkers (stopperc_check.py over the pinned
# stopbal_check.py), the REAL ship/refresh code and the REAL W script blob (committed into a throw-away repo so its sha256 matches the pin); only the trainer is a stand-in (tests/plant_trainer.py). CPU only, no GPU.
# Every path the chain writes is overridden with its STOPPERC_* variable into the scratch dir; the real busy marker, stopperc_state, t2v8f_pulls and slowwm_state are never named.
# Usage: bash test_stopperc_dryrun.sh  -> prints "DRYRUN ok=<n> failed=<m>"
set -u
TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PKG="$(cd "$TESTS/.." && pwd)"; PKG="$(cygpath -m "$PKG" 2>/dev/null || echo "$PKG")"; TESTS="$(cygpath -m "$TESTS")"
BALDIR="$(cd "$PKG/../../../stopbal/box2" && pwd)"; BALDIR="$(cygpath -m "$BALDIR")"
PYV="${STOPPERC_TEST_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
T="${STOPPERC_TEST_TMP:-F:/TanitAD-scratch/stopperc_dry/run_$$}"
case "$T" in F:/TanitAD-scratch/stopperc_dry/run_*) ;; *) echo "refusing an unrecognised scratch dir $T" >&2; exit 2;; esac
mkdir -p "$T/poison" || exit 2
for n in ssh sftp scp; do printf '#!/usr/bin/env bash\necho BREACH-%s >> "%s/BREACH"\nexit 99\n' "$n" "$T" > "$T/poison/$n"; chmod +x "$T/poison/$n"; done
export PATH="$T/poison:$PATH"
export CUDA_VISIBLE_DEVICES=-1
OK=0; BAD=0
ok() { if eval "$2"; then OK=$((OK+1)); else BAD=$((BAD+1)); echo "FAIL: $1"; fi; }
SSHS="$TESTS/stopperc_ssh_stub.sh"; SFTPS="$TESTS/stopperc_sftp_stub.sh"
REAL_BUSY="C:/Users/Admin/box2_ship/BOX2_BUSY.txt"; REAL_BUSY_SHA0="$(sha256sum "$REAL_BUSY" 2> /dev/null | cut -d' ' -f1)"
WSHA="064baba6e7385ee1df475e9e3206eb5cf74360bf6b3c8d3637823ba27bd4de63"
COMMIT="$("$PYV" -W ignore "$TESTS/dry_helpers.py" mkrepo "$T/repo" good)"
COMMIT_BADW="$("$PYV" -W ignore "$TESTS/dry_helpers.py" mkrepo "$T/repo_badw" badw)"
ok "throw-away repos have 40-hex commits that differ" '[ ${#COMMIT} -eq 40 ] && [ ${#COMMIT_BADW} -eq 40 ] && [ "$COMMIT" != "$COMMIT_BADW" ]'
ok "the repo's W script is the pinned blob; the bad repo's is not" '[ "$(git -c safe.directory=* -C "$T/repo" show $COMMIT:stack/scripts/stop_aux_w_from_smoke.py | sha256sum | cut -d" " -f1)" = "$WSHA" ] && [ "$(git -c safe.directory=* -C "$T/repo_badw" show $COMMIT_BADW:stack/scripts/stop_aux_w_from_smoke.py | sha256sum | cut -d" " -f1)" != "$WSHA" ]'
newbox() {   # newbox <name> [gpu MiB] [bal: equal|differs|none]  -> exports the dry-run environment for box <name>; every path lives under $T
  B="$T/box_$1"; "$PYV" -W ignore "$TESTS/dry_helpers.py" mkbox "$B" "$PYV" "${2:-0}" > /dev/null
  mkdir -p "$B/D/tanitad/stopbal/runs" "$B/D/tanitad/repo/stack" "$B/D/tanitad/slowwm"; echo "stopbal file" > "$B/D/tanitad/stopbal/runs/R1.txt"; echo "shared repo file" > "$B/D/tanitad/repo/stack/x.py"; echo "slowwm file" > "$B/D/tanitad/slowwm/S.txt"
  export STOPPERC_DRY=1 STOPPERC_SSH="$SSHS" STOPPERC_SFTP="$SFTPS" FAKE_BOX="$(cygpath -m "$B")" FAKE_PY="$PYV" STOPPERC_PY="$PYV" STOPPERC_STATE="$T/st_$1" STOPPERC_PULLS="$T/pulls_$1"
  export STOPPERC_BUSY_DEV="$T/busy_$1.txt" STOPPERC_ASIDE_DEV="$T/aside_$1" STOPBAL_B2_WIN_ROOT="$(cygpath -w "$B/D/tanitad")" STOPBAL_POLL_S=0 STOPPERC_REPO="$T/repo" STOPPERC_EXPORT_LIST="$T/repo/export_list.txt"
  export STOPPERC_TEST_COMMIT="$COMMIT" STOPPERC_NEED_C_MB=1 STOPPERC_POLLS_RUN=5 STOPPERC_POLLS_SHIP=5 STOPPERC_POLLS_CLEAN=5 STOPPERC_POLLS_WSMOKE=5 STOPBAL_ROOT="$(cygpath -m "$B/D/tanitad")"
  export STOPPERC_BAL_NULL_CKPT="$T/bal_$1.pt"; unset STOPPERC_RESUME STOPPERC_TEST_BAL_SHA STOPPERC_TABLE_DEV STOPPERC_BAL_DIR FAKE_CORRUPT_PUT
  case "${3:-equal}" in none) ;; *) export STOPPERC_TEST_BAL_SHA="$("$PYV" -W ignore "$TESTS/dry_helpers.py" mkbal "$STOPPERC_BAL_NULL_CKPT" "${3:-equal}")";; esac
}
control() { printf '%s' "$1" > "$B/plant_control.json"; }
calls() { cat "$B/transport.log" 2>/dev/null | wc -l | tr -d ' '; }
stoptxt() { cat "$STOPPERC_STATE/STOP.txt" 2>/dev/null; }
treehash() { ( cd "$1" && find . -type f | sort | xargs sha256sum | sha256sum | cut -d' ' -f1 ); }
runchain() { bash "$PKG/stopperc_chain.sh" > "$T/$1.out" 2>&1; rc=$?; }
ARUNS="NULL_s0 NULL_rep2 A_s0 A_s1"
ALL_PATHS_IN_T() { local v; for v in STOPPERC_STATE STOPPERC_PULLS STOPPERC_BUSY_DEV STOPPERC_ASIDE_DEV STOPPERC_REPO STOPPERC_BAL_NULL_CKPT FAKE_BOX; do case "${!v}" in F:/TanitAD-scratch/stopperc_dry/run_*) ;; *) return 1;; esac; done; return 0; }

echo "== 1 normal flow: ship, W smoke, NULL_s0, reuse EQUAL, NULL_rep2, A_s0, A_s1, DONE"
newbox normal 0 equal; control '{"sps": 1.0, "startup_s": 30}'
ok "every overridden path of this dry run is under the scratch dir (none is a real state/busy/pulls path)" 'ALL_PATHS_IN_T'
SH0="$(treehash "$B/D/tanitad/stopbal")"; RH0="$(treehash "$B/D/tanitad/repo")"; WH0="$(treehash "$B/D/tanitad/slowwm")"
ARGVSHA0="$(sha256sum "$PKG/argvs_stopperc.txt" | cut -d' ' -f1)"
runchain normal
ok "chain exits 0" '[ $rc -eq 0 ]'
ok "DONE sentinel written, no stop, prefix ZZ STOPPERC-DONE" '[ -e "$STOPPERC_STATE/DONE.txt" ] && [ ! -e "$STOPPERC_STATE/STOP.txt" ] && head -c 20 "$STOPPERC_STATE/DONE.txt" | grep -q "^ZZ STOPPERC-DONE"'
ok "DONE lists exactly the four runs" 'head -1 "$STOPPERC_STATE/DONE.txt" | grep -q "runs=NULL_s0 NULL_rep2 A_s0 A_s1$"'
ok "sentinel text only in its own file (chain.log carries none)" '! grep -q "ZZ STOPPERC" "$STOPPERC_STATE/chain.log"'
ok "the chain used the stand-in commit only through the DRY hook: chain.log and the tokens name it" 'grep -q "\"commit\": \"$COMMIT\"" "$STOPPERC_PULLS/stopperc_A_s0/gate_token.json"'
ok "ship: SHIP_RESULT ok for the commit, TIP_STOPBAL.txt on the box = commit" '"$PYV" -I -c "import json,sys;r=json.load(open(sys.argv[1]));assert r[\"ok\"] is True and r[\"commit\"]==sys.argv[2] and not r[\"verify_failures\"]" "$B/D/tanitad/stopperc/ship/SHIP_RESULT.json" "$COMMIT" && [ "$(tr -d "\r\n" < "$B/D/tanitad/stopperc/repo/TIP_STOPBAL.txt")" = "$COMMIT" ]'
ok "ship: the package files on the box equal the package (stopperc_check, stopbal_check, argvs) by sha256" '[ "$(sha256sum "$B/D/tanitad/stopperc/check/stopperc_check.py" | cut -c1-64)" = "$(sha256sum "$PKG/stopperc_check.py" | cut -c1-64)" ] && [ "$(sha256sum "$B/D/tanitad/stopperc/check/stopbal_check.py" | cut -c1-64)" = "$(sha256sum "$BALDIR/stopbal_check.py" | cut -c1-64)" ] && [ "$(sha256sum "$B/D/tanitad/stopperc/check/argvs_stopperc.txt" | cut -c1-64)" = "$ARGVSHA0" ]'
ok "ship: the camera table on the box hashes to the pinned table sha256" '[ "$(sha256sum "$B/D/tanitad/stopperc/data/stopaux_extr_join4572.json" | cut -c1-64)" = "77b6e8dc63d07d90548d09182fd5b8073e81079e224e8ce672dda11d2c3ae361" ]'
ok "W script read from the repo at the commit is the pinned blob" '[ "$(sha256sum "$STOPPERC_STATE/wscript/stop_aux_w_from_smoke.py" | cut -c1-64)" = "$WSHA" ]'
W="$(cat "$STOPPERC_STATE/W.txt" 2>/dev/null | tr -d '\r\n')"
ok "W.txt is a 4-decimal number and equals goal / stop_aux of the smoke probe (2.3456 / 0.6789 -> 3.4550)" '[ "$W" = "3.4550" ]'
ok "the W smoke ran before the first run (chain.log order)" '[ "$(grep -n "start wsmoke" "$STOPPERC_STATE/chain.log" | cut -d: -f1)" -lt "$(grep -n "start NULL_s0" "$STOPPERC_STATE/chain.log" | cut -d: -f1)" ]'
ok "run order in chain.log: NULL_s0, NULL_rep2, A_s0, A_s1" '[ "$(grep -o "done [A-Za-z0-9_]*$" "$STOPPERC_STATE/chain.log" | tr "\n" " ")" = "done NULL_s0 done NULL_rep2 done A_s0 done A_s1 " ]'
for t in $ARUNS; do
  BD="$STOPPERC_PULLS/stopperc_$t"
  ok "$t banked with BANK.done, BANK_MANIFEST.json and the evidence files" '[ -e "$BD/BANK.done" ] && [ -s "$BD/BANK_MANIFEST.json" ] && [ -s "$BD/ckpt_last.pt" ] && [ -s "$BD/summary.json" ] && [ -s "$BD/train_log.jsonl" ] && [ -s "$BD/gate_token.json" ] && [ -s "$BD/selfcheck.json" ] && [ -s "$BD/RUN.done" ] && [ -s "$BD/argv.txt" ]'
  ok "$t banked checkpoint equals the box copy (sha256) and BANK_MANIFEST carries it" '[ "$(sha256sum "$BD/ckpt_last.pt" | cut -c1-64)" = "$(sha256sum "$B/D/tanitad/runs/v8f_stopperc/$t/ckpt_last.pt" | cut -c1-64)" ] && grep -q "$(sha256sum "$BD/ckpt_last.pt" | cut -c1-64)" "$BD/BANK_MANIFEST.json"'
  ok "$t token binds the commit and the argv (verify-token with W)" 'STOPPERC_W="$W" "$PYV" -I "$PKG/stopperc_check.py" verify-token --token "$BD/gate_token.json" --argv-file "$BD/argv.txt" --tag $t --commit "$COMMIT" > /dev/null'
  ok "$t RUN.done carries the moved identity (S, T and O moved)" 'grep -q "\"moved_identity\"" "$BD/RUN.done" && grep -q "\"S\"" "$BD/RUN.done" && grep -q "\"O\"" "$BD/RUN.done"'
  ok "$t gate dir on the box deleted, the checkpoint kept, cleanup logged" '[ ! -e "$B/D/tanitad/stopperc/gate_out/$t" ] && [ -s "$B/D/tanitad/runs/v8f_stopperc/$t/ckpt_last.pt" ] && [ -s "$BD/CLEANUP.json" ] && grep -q "\"bytes\"" "$BD/CLEANUP.json"'
done
ok "NULL argvs carry no stop flag; A argvs carry exactly --stop-aux on --w-stop-aux W --stop-aux-extr <table>, W filled" '"$PYV" -I -c "
import sys
R, W = sys.argv[1], sys.argv[2]
L = {t: open(f\"{R}/stopperc_{t}/argv.txt\").read().split() for t in (\"NULL_s0\", \"NULL_rep2\", \"A_s0\", \"A_s1\")}
T = r\"D:\\tanitad\\stopperc\\data\\stopaux_extr_join4572.json\"
for t in (\"NULL_s0\", \"NULL_rep2\"):
    assert not [x for x in L[t] if x.startswith(\"--stop\") or x.startswith(\"--w-stop\")], t
for t in (\"A_s0\", \"A_s1\"):
    assert L[t][-6:] == [\"--stop-aux\", \"on\", \"--w-stop-aux\", W, \"--stop-aux-extr\", T], (t, L[t][-6:])
    assert len([x for x in L[t] if x.startswith(\"--stop\") or x.startswith(\"--w-stop\")]) == 3 and \"W_PLACEHOLDER\" not in L[t]
n, a = L[\"NULL_s0\"], L[\"A_s0\"][:-6]
d = [i for i in range(len(n)) if n[i] != a[i]]
assert len(n) == len(a) and d == [n.index(\"--out\") + 1], d
" "$STOPPERC_PULLS" "$W"'
ok "the frozen argv file is unchanged on disk (it keeps W_PLACEHOLDER, so its sha stays pinned)" '[ "$(sha256sum "$PKG/argvs_stopperc.txt" | cut -d" " -f1)" = "$ARGVSHA0" ] && [ "$(grep -c W_PLACEHOLDER "$PKG/argvs_stopperc.txt")" -eq 2 ]'
ok "the A tokens bind W (gate_token.json w_stop_aux = W.txt) and the camera table sha256" '"$PYV" -I -c "import json,sys;[ (lambda d:(d[\"w_stop_aux\"]==sys.argv[2] and d[\"stop_aux_table_sha256\"]==\"77b6e8dc63d07d90548d09182fd5b8073e81079e224e8ce672dda11d2c3ae361\") or sys.exit(1))(json.load(open(sys.argv[1]+\"/stopperc_\"+t+\"/gate_token.json\"))) for t in (\"A_s0\",\"A_s1\")]" "$STOPPERC_PULLS" "$W"'
ok "A checkpoints carry the 8 stop head tensors and cfg stop_aux on / W; NULL checkpoints carry none" '"$PYV" -I -c "
import sys, torch
P, W = sys.argv[1], float(sys.argv[2])
g = lambda t: torch.load(P + \"/stopperc_\" + t + \"/ckpt_last.pt\", map_location=\"cpu\", weights_only=False)
for t in (\"A_s0\", \"A_s1\"):
    c = g(t); assert c[\"cfg\"][\"stop_aux\"] == \"on\" and abs(c[\"cfg\"][\"w_stop_aux\"] - W) < 1e-9 and len([k for k in c[\"model\"] if k.startswith(\"stop_\")]) == 8, t
for t in (\"NULL_s0\", \"NULL_rep2\"):
    c = g(t); assert not [k for k in c[\"cfg\"] if k.startswith((\"stop_aux\", \"w_stop_aux\"))] and not [k for k in c[\"model\"] if k.startswith(\"stop_\")], t
" "$STOPPERC_PULLS" "$W"'
ok "the four tokens bind four distinct argvs" '[ "$(cat "$STOPPERC_PULLS"/stopperc_*/gate_token.json | grep -o "\"argv_sha256\": \"[0-9a-f]*\"" | sort -u | wc -l | tr -d " ")" -eq 4 ]'
ok "NULL_REUSE.json: computed, tensors equal although the file shas differ (argv_sha embedded): EQUAL; the conditional pair did not run" 'grep -q "\"status\": \"computed\"" "$STOPPERC_STATE/NULL_REUSE.json" && grep -q "\"null_reuse\": \"EQUAL\"" "$STOPPERC_STATE/NULL_REUSE.json" && grep -q "\"tensor_equal\": true" "$STOPPERC_STATE/NULL_REUSE.json" && [ ! -e "$STOPPERC_STATE/NULL_REUSE_DIFFERS" ] && [ ! -e "$STOPPERC_PULLS/stopperc_NULL_s1" ] && [ ! -e "$STOPPERC_PULLS/stopperc_NULL_rep" ]'
ok "the reuse check ran after NULL_s0 and before NULL_rep2 started" '[ "$(grep -n "NULL reuse:" "$STOPPERC_STATE/chain.log" | cut -d: -f1)" -lt "$(grep -n "start NULL_rep2" "$STOPPERC_STATE/chain.log" | cut -d: -f1)" ]'
ok "the box saw only stub traffic and no unknown command" '[ "$(calls)" -gt 100 ] && ! grep -q "UNKNOWN COMMAND" "$B/transport.log"'
ok "disk was read before every stage (>= 6 reads: ship, W smoke, four runs)" '[ "$(grep -c "Get-PSDrive D" "$B/transport.log")" -ge 6 ]'
ok "isolation: the STOPBAL tree, the shared repo and the SLOWWM folder on the box are byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/stopbal")" = "$SH0" ] && [ "$(treehash "$B/D/tanitad/repo")" = "$RH0" ] && [ "$(treehash "$B/D/tanitad/slowwm")" = "$WH0" ]'
ok "busy marker released: moved aside, not deleted, and the aside names the chain" '[ ! -e "$STOPPERC_BUSY_DEV" ] && [ "$(ls "$STOPPERC_ASIDE_DEV" | wc -l | tr -d " ")" -eq 1 ] && head -c 40 "$STOPPERC_ASIDE_DEV"/BOX2_BUSY.stopperc.* | grep -q "^stopperc chain"'
ok "no breach through PATH" '[ ! -e "$T/BREACH" ]'
ok "a second chain run skips every finished stage and says so" 'bash "$PKG/stopperc_chain.sh" 2>&1 | tail -3 | grep -q "all runs banked" && [ "$(grep -c "start ship" "$STOPPERC_STATE/chain.log")" -eq 1 ] && [ "$(grep -c "start NULL_s0" "$STOPPERC_STATE/chain.log")" -eq 1 ]'
NORMAL_B="$B"; NORMAL_PULLS="$STOPPERC_PULLS"; NORMAL_STATE="$STOPPERC_STATE"

echo "== 1b the box-side pin refuses a changed token and a changed W (checker run directly on the fake box, no transport)"
B="$NORMAL_B"; RD="$B/D/tanitad/stopperc/runs/A_s1"; cp "$RD/argv.txt" "$T/argv.keep"
boxenv() { export STOPBAL_ROOT="$(cygpath -m "$B/D/tanitad")" STOPBAL_COMMIT="$COMMIT" STOPBAL_FAKE_INIT_SHA_FILE="$(cygpath -m "$B/D/tanitad/init_sha.json")" STOPBAL_FAKE_BANK_SHA_FILE="$(cygpath -m "$B/D/tanitad/bank_sha.json")" STOPBAL_FLOOR_BYTES=1000 STOPBAL_FAKE_V9_MD5="$(md5sum "$B/D/tanitad/data/v9/v9_labels_train.npz" | cut -d' ' -f1)"; }
sed -i 's/--seed 1/--seed 7/' "$RD/argv.txt"
( boxenv; STOPPERC_W="$W" "$PYV" -I "$B/D/tanitad/stopperc/check/stopperc_check.py" stage --tag A_s1 --gid pin1 > "$T/pin.out" 2>&1 ); rc=$?
ok "a changed seed token of an A argv is refused (rc 3) and the first differing token is named" '[ $rc -eq 3 ] && grep -q "pin-fail" "$RD/STAGE_pin1.fail" && grep -q "frozen argv has" "$RD/STAGE_pin1.fail"'
cp "$T/argv.keep" "$RD/argv.txt"
( boxenv; STOPPERC_W="9.9999" "$PYV" -I "$B/D/tanitad/stopperc/check/stopperc_check.py" stage --tag A_s1 --gid pin2 > "$T/pin2.out" 2>&1 ); rc=$?
ok "a W that is not the argv's W (box env says 9.9999, argv says $W) is refused at the pin (rc 3)" '[ $rc -eq 3 ] && grep -q "pin-fail" "$RD/STAGE_pin2.fail"'
( boxenv; STOPPERC_W="0.0000" "$PYV" -I "$PKG/stopperc_check.py" argv --tag A_s1 > "$T/w0.out" 2>&1 ); rc=$?
ok "W 0.0000 and W unfilled are refused by the argv command (non-zero)" '[ $rc -ne 0 ] && ( boxenv; env -u STOPPERC_W "$PYV" -I "$PKG/stopperc_check.py" argv --tag A_s1 > "$T/w1.out" 2>&1; [ $? -eq 23 ] )'
echo "changed bank" > "$B/D/tanitad/runs/v8f_mini_full/dagger_bank/s1/bank.json"
( boxenv; STOPPERC_W="$W" "$PYV" -I "$B/D/tanitad/stopperc/check/stopperc_check.py" stage --tag A_s1 --gid bank1 > "$T/bank.out" 2>&1 ); rc=$?
ok "a DAgger bank whose sha256 differs is refused at the box-side preflight (rc 3)" '[ $rc -eq 3 ] && grep -q "preflight-fail" "$RD/STAGE_bank1.fail" && grep -q "DAgger bank s1" "$RD/STAGE_bank1.fail"'

echo "== 2 reuse DIFFERS appends NULL_s1 and NULL_rep and runs them; reuse UNKNOWN stops (rc 13)"
newbox differs 0 differs; control '{"sps": 1.0}'
runchain d1
ok "chain exits 0, DONE lists the four runs then NULL_s1 and NULL_rep" '[ $rc -eq 0 ] && head -1 "$STOPPERC_STATE/DONE.txt" | grep -q "runs=NULL_s0 NULL_rep2 A_s0 A_s1 NULL_s1 NULL_rep$"'
ok "NULL_REUSE.json says DIFFERS (the tensors differ) and the marker file exists" 'grep -q "\"null_reuse\": \"DIFFERS\"" "$STOPPERC_STATE/NULL_REUSE.json" && grep -q "\"tensor_equal\": false" "$STOPPERC_STATE/NULL_REUSE.json" && [ -e "$STOPPERC_STATE/NULL_REUSE_DIFFERS" ]'
ok "NULL_s1 and NULL_rep ran and are banked with tokens (seed 1 / init 1 and seed 2 / init 0), run order is the card's with the pair last" 'for t in NULL_s1 NULL_rep; do [ -e "$STOPPERC_PULLS/stopperc_$t/BANK.done" ] || exit 1; done; grep -q -- "--seed 1 " "$STOPPERC_PULLS/stopperc_NULL_s1/argv.txt" && grep -q -- "--seed 2 " "$STOPPERC_PULLS/stopperc_NULL_rep/argv.txt" && [ "$(grep -o "done [A-Za-z0-9_]*$" "$STOPPERC_STATE/chain.log" | tr "\n" " ")" = "done NULL_s0 done NULL_rep2 done A_s0 done A_s1 done NULL_s1 done NULL_rep " ]'
ok "the two conditional NULL argvs carry no stop flag" '! grep -q -- "--stop-aux\|--w-stop-aux" "$STOPPERC_PULLS/stopperc_NULL_s1/argv.txt" "$STOPPERC_PULLS/stopperc_NULL_rep/argv.txt"'
newbox unknown 0 equal; control '{"sps": 1.0}'; unset STOPPERC_TEST_BAL_SHA
runchain u1
ok "a BAL checkpoint that does not hash to the pinned 5169ea95 (the hook withheld) is UNKNOWN: the chain stops with rc 13 and names it" '[ $rc -eq 13 ] && grep -q "\"null_reuse\": \"UNKNOWN\"" "$STOPPERC_STATE/NULL_REUSE.json" && stoptxt | grep -q "^ZZ STOPPERC-STOP reuse null-reuse-unknown"'
ok "NULL_s0 was banked, NULL_rep2 never started, no run was appended" '[ -e "$STOPPERC_PULLS/stopperc_NULL_s0/BANK.done" ] && [ ! -e "$B/D/tanitad/stopperc/runs/NULL_rep2/argv.txt" ] && [ ! -e "$STOPPERC_STATE/NULL_REUSE_DIFFERS" ] && [ ! -e "$STOPPERC_STATE/DONE.txt" ]'
newbox unknown2 0 none; control '{"sps": 1.0}'
runchain u2
ok "no BAL checkpoint file at all is UNKNOWN too: rc 13, nothing appended" '[ $rc -eq 13 ] && grep -q "\"null_reuse\": \"UNKNOWN\"" "$STOPPERC_STATE/NULL_REUSE.json" && grep -q "a checkpoint is missing" "$STOPPERC_STATE/NULL_REUSE.json"'

echo "== 3 W smoke refusals: probe_n_pos 0 (STOP rc 14, no run starts), W script refuses, W rounds to 0.0000, the pinned W script differs"
newbox npos 0; control '{"n_pos": {"W_SMOKE": 0}}'
runchain n1
ok "probe_n_pos 0 in the smoke selfcheck: STOP rc 14, the stop line names probe_n_pos and that the sampler is not changed" '[ $rc -eq 14 ] && stoptxt | grep -q "^ZZ STOPPERC-STOP wsmoke" && stoptxt | grep -q "probe_n_pos" && stoptxt | grep -q "sampler"'
ok "no run started: no NULL_s0 folder on the box, no bank, no W.txt, no wsmoke.done, busy marker released" '[ ! -e "$B/D/tanitad/stopperc/runs/NULL_s0/argv.txt" ] && [ ! -e "$B/D/tanitad/runs/v8f_stopperc" ] && [ ! -e "$STOPPERC_PULLS/stopperc_NULL_s0" ] && [ ! -e "$STOPPERC_STATE/W.txt" ] && [ ! -e "$STOPPERC_STATE/done/wsmoke.done" ] && [ ! -e "$STOPPERC_BUSY_DEV" ]'
newbox ncrash 0; control '{"selfcheck_crash": {"W_SMOKE": 1}}'
runchain n2
ok "a trainer that raises on the smoke (selfcheck.json never written) also stops with rc 14 and names the probe cause" '[ $rc -eq 14 ] && stoptxt | grep -q "probe_n_pos" && [ ! -e "$STOPPERC_STATE/W.txt" ]'
newbox ngoal 0; control '{"goal": {"W_SMOKE": 0}}'
runchain n3
ok "a probe whose goal term is 0 makes the W script refuse: STOP rc 14, no W.txt" '[ $rc -eq 14 ] && [ ! -e "$STOPPERC_STATE/W.txt" ] && [ ! -e "$B/D/tanitad/runs/v8f_stopperc" ]'
newbox nzero 0; control '{"goal": {"W_SMOKE": 0.00001}}'
runchain n4
ok "a W that rounds to 0.0000 stops at the W smoke (rc 14), BEFORE two NULL runs are spent" '[ $rc -eq 14 ] && [ ! -e "$B/D/tanitad/runs/v8f_stopperc" ] && [ ! -e "$STOPPERC_STATE/done/wsmoke.done" ]'
newbox badw 0; export STOPPERC_REPO="$T/repo_badw" STOPPERC_EXPORT_LIST="$T/repo_badw/export_list.txt" STOPPERC_TEST_COMMIT="$COMMIT_BADW"; control '{}'
runchain n5
ok "a W script that is not the pinned blob (committed with an appended line) is refused at the W smoke: rc 8, no W.txt, no run" '[ $rc -eq 8 ] && stoptxt | grep -q "^ZZ STOPPERC-STOP wsmoke" && stoptxt | grep -q "pinned" && [ ! -e "$STOPPERC_STATE/W.txt" ] && [ ! -e "$B/D/tanitad/runs/v8f_stopperc" ]'

echo "== 4 busy marker held by another owner"
newbox busy 0; control '{}'; printf 'slowwm chain 4242 2026-11-12 21:00:00' > "$STOPPERC_BUSY_DEV"; BUSYSHA="$(sha256sum "$STOPPERC_BUSY_DEV" | cut -d' ' -f1)"
runchain b1
ok "a marker held by another owner: refused with rc 12, zero box calls, nothing shipped, STOP names box2-busy" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ] && stoptxt | grep -q "box2-busy" && [ ! -e "$STOPPERC_STATE/ship" ] && [ ! -e "$B/D/tanitad/stopperc/ship" ]'
ok "the other owner's marker is byte-identical afterwards and still in place (not moved aside)" '[ "$(sha256sum "$STOPPERC_BUSY_DEV" | cut -d" " -f1)" = "$BUSYSHA" ] && [ ! -e "$STOPPERC_ASIDE_DEV"/BOX2_BUSY.stopperc.* ]'
rm -f "$STOPPERC_STATE/STOP.txt"; runchain b2
ok "a second attempt while the marker is still held is refused again (rc 12), marker unchanged" '[ $rc -eq 12 ] && [ "$(sha256sum "$STOPPERC_BUSY_DEV" | cut -d" " -f1)" = "$BUSYSHA" ] && [ "$(calls)" -eq 0 ]'
rm -f "$STOPPERC_STATE/STOP.txt" "$STOPPERC_BUSY_DEV"; echo 9000 > "$B/gpu_mib.txt"
runchain b3
ok "box GPU busy (9000 MiB) with the marker free: the W smoke refuses (rc 12), nothing trained, the chain releases its own marker" '[ $rc -eq 12 ] && stoptxt | grep -q "box2-busy-or-gpu-not-idle" && [ ! -e "$B/D/tanitad/runs/v8f_stopperc" ] && [ ! -e "$STOPPERC_BUSY_DEV" ]'
rm -f "$STOPPERC_STATE/STOP.txt"; echo 0 > "$B/gpu_mib.txt"; echo 2 > "$B/py_count.txt"; rm -f "$STOPPERC_STATE/done/wsmoke.done"
runchain b4
ok "a python process on box 2 refuses the W smoke (rc 12)" '[ $rc -eq 12 ] && stoptxt | grep -q "box2-busy-or-gpu-not-idle"'

echo "== 5 gate refusal on A_s0 stops the chain; STOP.txt refuses a rerun; STOPPERC_RESUME=1 resumes from done/ markers"
newbox gate 0; control '{"sps": 1.0, "modes": {"A_s0": "w_not_built"}}'
runchain g1
ok "a planted wrong lever (BUILT w_stop_aux 1.0, argv W $W) on A_s0 is refused at the gate (rc 4); the stop line names the lever" '[ $rc -eq 4 ] && stoptxt | grep -q "^ZZ STOPPERC-STOP A_s0 gate-or-pin-refused" && stoptxt | grep -q "w_stop_aux"'
ok "the earlier runs (W smoke, NULL_s0, NULL_rep2) were banked; A_s0 was never trained and A_s1 never started" '[ -e "$STOPPERC_PULLS/stopperc_NULL_s0/BANK.done" ] && [ -e "$STOPPERC_PULLS/stopperc_NULL_rep2/BANK.done" ] && [ ! -e "$B/D/tanitad/runs/v8f_stopperc/A_s0" ] && [ ! -e "$B/D/tanitad/stopperc/runs/A_s1/argv.txt" ] && [ ! -e "$STOPPERC_PULLS/stopperc_A_s0" ] && [ ! -e "$STOPPERC_STATE/DONE.txt" ]'
ok "no token was written for the refused arm; the busy marker was released" '[ ! -e "$B/D/tanitad/stopperc/runs/A_s0/gate/gate_token.json" ] && [ ! -e "$STOPPERC_BUSY_DEV" ]'
STOPSHA="$(sha256sum "$STOPPERC_STATE/STOP.txt" | cut -d' ' -f1)"; C0="$(calls)"
runchain g2
ok "STOP.txt present refuses a plain rerun (rc 2) without a single box call; STOP.txt is untouched" '[ $rc -eq 2 ] && [ "$(calls)" -eq "$C0" ] && [ "$(sha256sum "$STOPPERC_STATE/STOP.txt" | cut -d" " -f1)" = "$STOPSHA" ] && grep -q "REFUSED" "$T/g2.out"'
for m in stop_off_built no_head_params dead_stop_head bad_table_sha; do
  control "{\"sps\": 1.0, \"modes\": {\"A_s0\": \"$m\"}}"; STOPPERC_RESUME=1 runchain "g_$m"
  ok "RESUME with the planted defect '$m' on A_s0: refused at the gate again (rc 4), nothing of A trained" '[ $rc -eq 4 ] && stoptxt | grep -q "^ZZ STOPPERC-STOP A_s0 gate-or-pin-refused" && [ ! -e "$B/D/tanitad/runs/v8f_stopperc/A_s0" ]'
done
ok "each RESUME archived the previous STOP.txt (STOP.<timestamp>.txt files exist)" '[ "$(ls "$STOPPERC_STATE"/STOP.2*.txt 2>/dev/null | wc -l | tr -d " ")" -ge 4 ]'
control '{"sps": 1.0, "train_modes": {"A_s0": "cfg_stop_lost"}}'; STOPPERC_RESUME=1 runchain g_cfg
ok "an A_s0 checkpoint whose cfg lost the declared W is a post-check failure (rc 5); the checkpoint stays on the box, no bank" '[ $rc -eq 5 ] && stoptxt | grep -q "post-fail" && stoptxt | grep -q "w_stop_aux" && [ -s "$B/D/tanitad/runs/v8f_stopperc/A_s0/ckpt_last.pt" ] && [ ! -e "$STOPPERC_PULLS/stopperc_A_s0/BANK.done" ]'
control '{"sps": 1.0, "train_modes": {"A_s0": "head_missing"}}'; STOPPERC_RESUME=1 runchain g_head
ok "an A_s0 checkpoint without the 8 head tensors is a post-check failure (rc 5) naming them" '[ $rc -eq 5 ] && stoptxt | grep -q "head tensors"'
control '{"sps": 1.0}'; STOPPERC_RESUME=1 runchain g3
ok "defect removed + STOPPERC_RESUME=1: the chain resumes and finishes (rc 0), DONE lists the four runs" '[ $rc -eq 0 ] && head -1 "$STOPPERC_STATE/DONE.txt" | grep -q "runs=NULL_s0 NULL_rep2 A_s0 A_s1$" && [ ! -e "$STOPPERC_STATE/STOP.txt" ]'
ok "the resume did NOT redo ship, the W smoke, NULL_s0 or NULL_rep2 (start lines: once each) and did start A_s0 more than once" '[ "$(grep -c "start ship" "$STOPPERC_STATE/chain.log")" -eq 1 ] && [ "$(grep -c "start wsmoke" "$STOPPERC_STATE/chain.log")" -eq 1 ] && [ "$(grep -c "start NULL_s0" "$STOPPERC_STATE/chain.log")" -eq 1 ] && [ "$(grep -c "start NULL_rep2" "$STOPPERC_STATE/chain.log")" -eq 1 ] && [ "$(grep -c "start A_s0" "$STOPPERC_STATE/chain.log")" -ge 2 ]'
ok "A_s0 and A_s1 are banked with BANK.done after the resume" '[ -e "$STOPPERC_PULLS/stopperc_A_s0/BANK.done" ] && [ -e "$STOPPERC_PULLS/stopperc_A_s1/BANK.done" ]'
newbox gnull 0; control '{"sps": 1.0, "modes": {"NULL_s0": "null_has_stop"}}'
runchain gn1
ok "an OFF arm (NULL_s0) whose selfcheck carries a stop_* key is refused at the gate (rc 4): the NULL must be the pre-change run" '[ $rc -eq 4 ] && stoptxt | grep -q "^ZZ STOPPERC-STOP NULL_s0 gate-or-pin-refused" && stoptxt | grep -q "stop keys" && [ ! -e "$STOPPERC_PULLS/stopperc_NULL_s0" ]'
newbox gbase 0; control '{"sps": 1.0, "modes": {"NULL_s0": "bad_params"}}'
runchain gn2
ok "the BASE gate still runs on the NULL arm (a frozen S, params 0) -> refused (rc 4), STOP names the group" '[ $rc -eq 4 ] && stoptxt | grep -q "group S"'
newbox pnull 0; control '{"sps": 1.0, "train_modes": {"NULL_rep2": "null_cfg_stop"}}'
runchain gn3
ok "an OFF arm's checkpoint that carries a stop-aux cfg entry is a post-check failure (rc 5); NULL_s0 banked, A_s0 never started" '[ $rc -eq 5 ] && stoptxt | grep -q "off arm" && [ -e "$STOPPERC_PULLS/stopperc_NULL_s0/BANK.done" ] && [ ! -e "$B/D/tanitad/stopperc/runs/A_s0/argv.txt" ]'
newbox crash 0; control '{"sps": 1.0, "train_modes": {"NULL_s0": "crash"}}'
runchain c1
ok "a trainer crash on NULL_s0 stops the chain (rc 5) before NULL_rep2 and before any reuse check" '[ $rc -eq 5 ] && [ ! -e "$B/D/tanitad/stopperc/runs/NULL_rep2/argv.txt" ] && [ ! -e "$STOPPERC_STATE/NULL_REUSE.json" ] && [ ! -e "$STOPPERC_PULLS/stopperc_NULL_s0" ]'
newbox frozen 0; control '{"sps": 1.0, "train_modes": {"A_s1": "S_frozen"}}'
runchain c2
ok "an S that never moved on A_s1 is a post-check failure (rc 5); A_s0 is banked, no DONE" '[ $rc -eq 5 ] && stoptxt | grep -q "group S was not trained" && [ -e "$STOPPERC_PULLS/stopperc_A_s0/BANK.done" ] && [ ! -e "$STOPPERC_PULLS/stopperc_A_s1/BANK.done" ] && [ ! -e "$STOPPERC_STATE/DONE.txt" ]'
newbox nowd 0; control '{}'; mkdir -p "$STOPPERC_STATE/done"; for f in ship wsmoke NULL_s0 NULL_rep2; do echo x > "$STOPPERC_STATE/done/$f.done"; done; echo '{"null_reuse": "EQUAL"}' > "$STOPPERC_STATE/NULL_REUSE.json"
runchain nw
ok "arm A without a W.txt (the W smoke never ran): the chain refuses before any box call (rc 8), the A argv is never rendered with a placeholder" '[ $rc -eq 8 ] && stoptxt | grep -q "no W" && [ "$(calls)" -eq 0 ]'

echo "== 6 ship, disk and transport fail-closed"
newbox corrupt 0; control '{}'
FAKE_CORRUPT_PUT=argvs_stopperc.txt runchain k1
ok "a put that arrives altered is caught by the re-hash on box 2 (rc 5), the busy marker is released, nothing runs" '[ $rc -eq 5 ] && [ ! -e "$B/D/tanitad/runs/v8f_stopperc" ] && [ ! -e "$STOPPERC_BUSY_DEV" ] && [ ! -e "$STOPPERC_STATE/done/ship.done" ]'
newbox disk 0; control '{}'; echo 3 > "$B/free_gb.txt"
runchain k2
ok "3 GB free: ship passes (1 GB), the W smoke stops with the disk-low code (rc 6)" '[ $rc -eq 6 ] && stoptxt | grep -q "disk-low" && [ ! -e "$B/D/tanitad/runs/v8f_stopperc" ]'
newbox disk0 0; echo 0 > "$B/free_gb.txt"
runchain k3
ok "0 GB free: the ship stage stops with rc 6" '[ $rc -eq 6 ] && [ ! -e "$B/D/tanitad/stopperc/ship/stopbal_tree.tar" ]'
newbox disk1 0; echo "" > "$B/free_gb.txt"
runchain k4
ok "an unreadable free-space probe is a refusal, not a pass (rc 11)" '[ $rc -eq 11 ]'
newbox nocommit 0; unset STOPPERC_TEST_COMMIT
runchain k5
ok "without the test hook the chain's commit is the code of record 3e264bbf (the throw-away repo cannot supply it): the ship stage refuses (rc 8) and the ship folder is keyed by that sha" '[ $rc -eq 8 ] && ls "$STOPPERC_STATE/ship" | grep -q "^3e264bbf3fa2f6f17bea448de09aa972faaf7725$" && [ ! -e "$B/D/tanitad/stopperc/ship/stopbal_tree.tar" ]'
ok "the real ship code ACCEPTS the code of record 3e264bbf from the real mirror (read only) and its manifest names it and the stop-aux module" '"$PYV" -I "$BALDIR/stopbal_ship_tree.py" build --repo "C:/Users/Admin/tanitad-push" --commit 3e264bbf3fa2f6f17bea448de09aa972faaf7725 --export-list "E:/Projects/TanitAD-artifacts/dagland/recover_ft_launch/export_list.txt" --out-dir "$T/realship_ok" > "$T/realship_ok.out" 2>&1 && grep -q "\"commit\": \"3e264bbf3fa2f6f17bea448de09aa972faaf7725\"" "$T/realship_ok/stopbal_tree_manifest.json" && grep -q "v8f_stopaux.py" "$T/realship_ok/stopbal_tree_manifest.json"'
ok "the W script at the real code of record (read from the mirror) hashes to the chain's pin" '[ "$(git -C C:/Users/Admin/tanitad-push show 3e264bbf3fa2f6f17bea448de09aa972faaf7725:stack/scripts/stop_aux_w_from_smoke.py | sha256sum | cut -d" " -f1)" = "$WSHA" ] && grep -q "$WSHA" "$PKG/stopperc_chain.sh"'
newbox tr 0
for how in unset relative absent realkind; do
  rm -f "$STOPPERC_STATE/STOP.txt"; : > "$B/transport.log"
  case "$how" in
    unset) ( unset STOPPERC_SSH; bash "$PKG/stopperc_chain.sh" > "$T/tr_$how.out" 2>&1 ); rc=$?;;
    relative) STOPPERC_SSH="ssh" bash "$PKG/stopperc_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
    absent) STOPPERC_SSH="$T/no_such_ssh" bash "$PKG/stopperc_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
    realkind) printf '#!/usr/bin/env bash\necho i am a real client\n' > "$T/fake_real_ssh"; chmod +x "$T/fake_real_ssh"; STOPPERC_SSH="$(cygpath -m "$T/fake_real_ssh")" bash "$PKG/stopperc_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
  esac
  ok "transport '$how' refuses with rc 16, makes no call, and does NOT leave box 2 marked busy" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ] && [ ! -e "$STOPPERC_BUSY_DEV" ]'
done
rm -f "$STOPPERC_STATE/STOP.txt"; : > "$B/transport.log"
( unset STOPPERC_SFTP; bash "$PKG/stopperc_chain.sh" > "$T/tr_sftp.out" 2>&1 ); rc=$?
ok "STOPPERC_SFTP unset refuses with rc 16, zero calls, marker released" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ] && [ ! -e "$STOPPERC_BUSY_DEV" ]'
( unset STOPPERC_DRY; bash "$PKG/stopperc_chain.sh" > "$T/tr_nodry.out" 2>&1 ); rc=$?
ok "a stub WITHOUT test mode refuses (rc 16): the stub is never taken for the real client; and the test hook does not take effect without DRY" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ] && grep -q "stub but STOPBAL_DRY is not 1" "$T/tr_nodry.out"'
printf '#!/usr/bin/env bash\necho i am a real sftp\n' > "$T/fake_real_sftp"; chmod +x "$T/fake_real_sftp"
STOPPERC_SFTP="$(cygpath -m "$T/fake_real_sftp")" bash "$PKG/stopperc_chain.sh" > "$T/tr_realsftp.out" 2>&1; rc=$?
ok "test mode (STOPPERC_DRY=1) with a real sftp path refuses (rc 16), zero calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ] && [ ! -e "$STOPPERC_BUSY_DEV" ]'
ok "no breach through PATH after all transport refusals" '[ ! -e "$T/BREACH" ]'

echo "== 7 early refusals: camera table, pinned base checker, one real sha not equal"
newbox tab 0
cp "C:/Users/Admin/stopaux_extr/stopaux_extr_join4572.json" "$T/table_bad.json"; printf ' ' >> "$T/table_bad.json"
STOPPERC_TABLE_DEV="$T/table_bad.json" runchain e1
ok "a camera table that does not hash to the pinned sha256 is refused (rc 8), zero box calls, busy marker never claimed" '[ $rc -eq 8 ] && [ "$(calls)" -eq 0 ] && stoptxt | grep -q "camera table" && [ ! -e "$STOPPERC_BUSY_DEV" ]'
rm -f "$STOPPERC_STATE/STOP.txt"; mkdir -p "$T/baldir/stages"; cp "$BALDIR/stopbal_ship_tree.py" "$BALDIR/stopbal_refresh.py" "$T/baldir/"; cp "$BALDIR/stages/stopbal_common.sh" "$T/baldir/stages/"; { cat "$BALDIR/stopbal_check.py"; echo "# changed"; } > "$T/baldir/stopbal_check.py"
STOPPERC_BAL_DIR="$T/baldir" runchain e2
ok "a STOPBAL base checker that differs from the pinned one (appended comment) is refused (rc 8), zero box calls" '[ $rc -eq 8 ] && [ "$(calls)" -eq 0 ] && stoptxt | grep -q "differs from the pinned base" && [ ! -e "$STOPPERC_BUSY_DEV" ]'
rm -f "$STOPPERC_STATE/STOP.txt"; rm "$T/baldir/stopbal_refresh.py"
STOPPERC_BAL_DIR="$T/baldir" runchain e3
ok "a missing STOPBAL package file is refused (rc 8) before anything else" '[ $rc -eq 8 ] && stoptxt | grep -q "stopbal-package-incomplete" && [ "$(calls)" -eq 0 ]'
ok "the real busy marker was never touched (hash unchanged, or absent before and after)" '[ "$(sha256sum "$REAL_BUSY" 2> /dev/null | cut -d" " -f1)" = "$REAL_BUSY_SHA0" ]'
ok "no breach through PATH after everything" '[ ! -e "$T/BREACH" ]'

echo "== 8 hygiene"
ok "no ssh, sftp or scp command in this test script other than the stubs' own forwarding and the poison" '! grep -nE "^[^#]*\b(ssh|sftp|scp) +-" "$TESTS/test_stopperc_dryrun.sh" | grep -v "poison" | grep -q .'
ok "bash scripts and job files are LF (no CR byte)" '"$PYV" -I -c "import glob,sys;fs=glob.glob(sys.argv[1]+\"/*.sh\")+glob.glob(sys.argv[2]+\"/*.sh\")+glob.glob(sys.argv[1]+\"/jobs/*.cmd\")+glob.glob(sys.argv[2]+\"/*.py\");assert len(fs)>=9, fs;bad=[f for f in fs if b\"\r\" in open(f,\"rb\").read()];assert not bad, bad" "$PKG" "$TESTS"'
ok "the frozen argv file and every pinned file are unchanged by the dry run (argvs sha256 as at the start)" '[ "$(sha256sum "$PKG/argvs_stopperc.txt" | cut -d" " -f1)" = "$ARGVSHA0" ]'
echo "DRYRUN ok=$OK failed=$BAD"
[ "$BAD" -eq 0 ]
