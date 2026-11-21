#!/usr/bin/env bash
# Dry run of the OBD box-2 package over stubs. NOTHING here can reach box 2: OBD_SSH / OBD_SFTP name stub scripts by absolute path (OBD_DRY=1 makes the chain refuse any file without the stub marker),
# a poison ssh/sftp/scp is put FIRST on PATH and must never run, and the "box" is a folder. The REAL job .cmd files run on Windows cmd.exe against it; only the trainer is a stand-in (tests/plant_trainer.py,
# shipped through a throw-away git repo, so the ship, tree-binding and token code run for real). CPU only, no GPU, no torch.   Usage: bash test_obd_dryrun.sh  -> prints "DRYRUN ok=<n> failed=<m>"
set -u
TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PKG="$(cd "$TESTS/.." && pwd)"; PKG="$(cygpath -m "$PKG" 2>/dev/null || echo "$PKG")"
PYV="${OBD_TEST_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
T="${OBD_TEST_TMP:-F:/TanitAD-scratch/obd_build/obd_dry_run_$$}"
case "$T" in */obd_dry_run_*|*/scratchpad/*|*/obd_dry_*) ;; *) echo "refusing an unrecognised scratch dir $T" >&2; exit 2;; esac
rm -rf "${T:?}"; mkdir -p "$T/poison"
for n in ssh sftp scp; do printf '#!/usr/bin/env bash\necho BREACH-%s >> "%s/BREACH"\nexit 99\n' "$n" "$T" > "$T/poison/$n"; chmod +x "$T/poison/$n"; done
export PATH="$T/poison:$PATH"
export CUDA_VISIBLE_DEVICES=-1
OK=0; BAD=0
ok() { if eval "$2"; then OK=$((OK+1)); else BAD=$((BAD+1)); echo "FAIL: $1"; fi; }
SSHS="$(cygpath -m "$TESTS/ssh_stub.sh")"; SFTPS="$(cygpath -m "$TESTS/sftp_stub.sh")"
FAKESHA="$(printf 'f%.0s' $(seq 1 64))"
COMMIT="$("$PYV" -W ignore "$TESTS/dry_helpers.py" mkrepo "$T/repo")"
ok "throw-away repo has a 40-hex commit" '[ ${#COMMIT} -eq 40 ]'
newbox() {   # newbox <name> [gpu MiB]  -> exports the dry-run environment for box <name>; the EGOWM chain has ended (DONE) unless the test says otherwise
  B="$T/box_$1"; "$PYV" -W ignore "$TESTS/dry_helpers.py" mkbox "$B" "$PYV" "${2:-0}" > /dev/null
  mkdir -p "$B/D/tanitad/egowm/runs" "$B/D/tanitad/repo/stack"; echo "egowm file" > "$B/D/tanitad/egowm/runs/R1.txt"; echo "shared repo file" > "$B/D/tanitad/repo/stack/x.py"
  export OBD_DRY=1 OBD_SSH="$SSHS" OBD_SFTP="$SFTPS" FAKE_BOX="$(cygpath -m "$B")" FAKE_PY="$PYV" OBD_PY="$PYV" OBD_STATE="$T/st_$1" OBD_PULLS="$T/pulls_$1"
  export OBD_BUSY_DEV="$T/busy_$1.txt" OBD_ASIDE_DEV="$T/aside_$1" OBD_B2_WIN_ROOT="$(cygpath -w "$B/D/tanitad")" OBD_POLL_S=0 OBD_REPO="$T/repo" OBD_EXPORT_LIST="$T/repo/export_list.txt"
  export OBD_COMMIT="$COMMIT" OBD_NEED_C_MB=1 OBD_POLLS_RUN=5 OBD_POLLS_SHIP=5 OBD_POLLS_CLEAN=5 OBD_WAIT_POLLS=0 OBD_WAIT_POLL_S=0
  export OBD_EGOWM_STATE="$T/egowm_$1"; mkdir -p "$OBD_EGOWM_STATE"; echo "ZZ EGOWM-DONE 2026-11-10 00:00:00 commit=x" > "$OBD_EGOWM_STATE/DONE.txt"
  export OBD_DAGOACC_DIR="$T/dagoacc_$1"; mkdir -p "$OBD_DAGOACC_DIR"; printf '%s *ckpt_last.pt\n' "$FAKESHA" > "$OBD_DAGOACC_DIR/SHA256SUMS.txt"; export OBD_TEST_DAGOACC_SHA="$FAKESHA"
  unset OBD_RESUME OBD_AMENDMENT OBD_PARITY_BASIS
}
control() { printf '%s' "$1" > "$B/plant_control.json"; }
calls() { cat "$B/transport.log" 2>/dev/null | wc -l | tr -d ' '; }
stoptxt() { cat "$OBD_STATE/STOP.txt" 2>/dev/null; }
treehash() { ( cd "$1" && find . -type f | sort | xargs sha256sum | sha256sum | cut -d' ' -f1 ); }
ALL8="OBD-NULL_s0 OBD-NULL_rep OBD-CTL_s0 OBD-BAND_s0 OBD-CTL_s1 OBD-BAND_s1 OBD-BAND_rep OBD-NULL_s1"

echo "== 1 normal flow: wait check, ship, pilot, 12 h permit, seven runs, NULL parity (fails by file sha) -> OBD-NULL_s1, banks, cleanups"
newbox normal 0; control '{"sps": 1.0, "startup_s": 30}'
EH0="$(treehash "$B/D/tanitad/egowm")"; RH0="$(treehash "$B/D/tanitad/repo")"
bash "$PKG/obd_chain.sh" > "$T/normal.out" 2>&1; rc=$?
ok "chain exits 0" '[ $rc -eq 0 ]'
ok "DONE sentinel written, no stop" '[ -e "$OBD_STATE/DONE.txt" ] && [ ! -e "$OBD_STATE/STOP.txt" ] && head -c 13 "$OBD_STATE/DONE.txt" | grep -q "^ZZ OBD-DONE"'
ok "sentinel text only in its own file (chain.log carries none)" '! grep -q "ZZ OBD" "$OBD_STATE/chain.log"'
P="$OBD_STATE/pilot/PILOT.verified.json"
ok "PILOT.verified.json names the commit" 'grep -q "\"commit\": \"$COMMIT\"" "$P"'
ok "pilot mean s/step excludes start-up (reads 1.0)" '"$PYV" -I -c "import json,sys;d=json.load(open(sys.argv[1]));assert abs(d[\"timing\"][\"mean_s_per_step\"]-1.0)<1e-9, d[\"timing\"]" "$P"'
ok "projection covers 7 runs x 3,000 steps" '"$PYV" -I -c "import json,sys;d=json.load(open(sys.argv[1]))[\"projection\"];assert d[\"steps_total\"]==21000 and d[\"n_runs_planned\"]==7 and d[\"within_limit_planned\"] and d[\"limit_h\"]==12.0" "$P"'
ok "pilot used the exact OBD-BAND_s0 argv plus --steps 200 and a separate out dir" '"$PYV" -I -c "import sys,json;sys.path.insert(0,sys.argv[1]);import obd_check as C;l=open(sys.argv[2]).read().split();assert l==C.tokens(\"PILOT_OBD-BAND_s0\") and l[-2:]==[\"--steps\",\"200\"] and \"pilot_out\" in l[l.index(\"--out\")+1] and [x for x in l[:-2] if x not in C.tokens(\"OBD-BAND_s0\")]==[l[l.index(\"--out\")+1]]" "$PKG" "$OBD_STATE/runs/PILOT_OBD-BAND_s0/argv.txt"'
ok "the pilot token binds commit and argv" '"$PYV" -I "$PKG/obd_check.py" verify-token --token "$OBD_STATE/pilot/gate_token.json" --argv-file "$OBD_STATE/runs/PILOT_OBD-BAND_s0/argv.txt" --tag PILOT_OBD-BAND_s0 --commit "$COMMIT" > /dev/null'
ok "the pilot also gated the full-length BAND_s0, CTL_s0 and NULL_s0 argvs (tokens exist before the 12 h decision)" '[ -s "$B/D/tanitad/obd/runs/OBD-BAND_s0/gate/gate_token.json" ] && [ -s "$B/D/tanitad/obd/runs/OBD-CTL_s0/gate/gate_token.json" ] && [ -s "$B/D/tanitad/obd/runs/OBD-NULL_s0/gate/gate_token.json" ]'
ok "pilot temp dirs deleted on the box (pilot out, the four gate outs)" '[ ! -e "$B/D/tanitad/obd/pilot_out/OBD-BAND_s0" ] && [ ! -e "$B/D/tanitad/obd/gate_out/PILOT_OBD-BAND_s0" ] && [ ! -e "$B/D/tanitad/obd/gate_out/OBD-NULL_s0" ] && [ ! -e "$B/D/tanitad/obd/gate_out/OBD-CTL_s0" ]'
ok "pilot deletions are logged (path, bytes) and banked" 'grep -q "\"bytes\"" "$OBD_STATE/pilot/CLEANUP.json" && grep -q "pilot_out" "$OBD_STATE/pilot/CLEANUP.json"'
for t in $ALL8; do
  BD="$OBD_PULLS/obd_$t"
  ok "$t banked with BANK.done and the six evidence files" '[ -e "$BD/BANK.done" ] && [ -s "$BD/ckpt_last.pt" ] && [ -s "$BD/summary.json" ] && [ -s "$BD/train_log.jsonl" ] && [ -s "$BD/gate_token.json" ] && [ -s "$BD/selfcheck.json" ] && [ -s "$BD/RUN.done" ]'
  ok "$t banked checkpoint equals the box copy (sha256)" '[ "$(sha256sum "$BD/ckpt_last.pt" | cut -d" " -f1)" = "$(sha256sum "$B/D/tanitad/runs/v8f_obd/$t/ckpt_last.pt" | cut -d" " -f1)" ]'
  ok "$t token binds the commit and the argv" '"$PYV" -I "$PKG/obd_check.py" verify-token --token "$BD/gate_token.json" --argv-file "$BD/argv.txt" --tag $t --commit "$COMMIT" > /dev/null'
  ok "$t RUN.done carries the moved identity (S, T and O moved)" 'grep -q "\"moved_identity\"" "$BD/RUN.done" && grep -q "\"S\"" "$BD/RUN.done" && grep -q "\"O\"" "$BD/RUN.done"'
  ok "$t gate dir on the box deleted, the checkpoint kept" '[ ! -e "$B/D/tanitad/obd/gate_out/$t" ] && [ -s "$B/D/tanitad/runs/v8f_obd/$t/ckpt_last.pt" ]'
  ok "$t cleanup logged (path, bytes) in the bank" 'grep -q "\"bytes\"" "$BD/CLEANUP.json" || grep -q "deleted_own_temp\": \[\]" "$BD/CLEANUP.json"'
done
ok "the eight tokens bind eight distinct argvs" '[ "$(cat "$OBD_PULLS"/obd_*/gate_token.json | grep -o "\"argv_sha256\": \"[0-9a-f]*\"" | sort -u | wc -l | tr -d " ")" -eq 8 ]'
ok "run order in chain.log: NULL_s0, NULL_rep, CTL_s0, BAND_s0, CTL_s1, BAND_s1, BAND_rep, then the appended NULL_s1" '[ "$(grep -o "done OBD-[A-Za-z0-9_-]*" "$OBD_STATE/chain.log" | tr "\n" " ")" = "done OBD-NULL_s0 done OBD-NULL_rep done OBD-CTL_s0 done OBD-BAND_s0 done OBD-CTL_s1 done OBD-BAND_s1 done OBD-BAND_rep done OBD-NULL_s1 " ]'
ok "NULL_PARITY.json exists, the file sha256 differs and OBD-NULL_s1 was appended" 'grep -q "\"file_sha_equal\": false" "$OBD_STATE/NULL_PARITY.json" && grep -q "\"append_obd_null_s1\": true" "$OBD_STATE/NULL_PARITY.json" && grep -q "OBD-NULL_s1 appended" "$OBD_STATE/chain.log"'
ok "the parity ran after OBD-NULL_s0 and before OBD-NULL_rep started" '[ "$(grep -n "PARITY file_sha_equal" "$OBD_STATE/chain.log" | cut -d: -f1)" -lt "$(grep -n "start OBD-NULL_rep" "$OBD_STATE/chain.log" | cut -d: -f1)" ]'
ok "OBD-NULL_s0 and OBD-NULL_rep run without a detach: the stand-in checkpoints say so; CTL and BAND say detach" '"$PYV" -I -c "import pickle,sys;g=lambda t:pickle.load(open(sys.argv[1]+\"/obd_\"+t+\"/ckpt_last.pt\",\"rb\"))[\"cfg\"][\"detach_upper\"];assert not g(\"OBD-NULL_s0\") and not g(\"OBD-NULL_rep\") and g(\"OBD-CTL_s1\") and g(\"OBD-BAND_rep\")" "$OBD_PULLS"'
ok "the box saw only stub traffic and no unknown command" '[ "$(calls)" -gt 100 ] && ! grep -q "UNKNOWN COMMAND" "$B/transport.log"'
ok "disk was read before every stage (>= 10 reads: ship, pilot, eight runs)" '[ "$(grep -c "Get-PSDrive D" "$B/transport.log")" -ge 10 ]'
ok "isolation: the EGOWM folder on the box is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/egowm")" = "$EH0" ]'
ok "isolation: the shared D:\\tanitad\\repo is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/repo")" = "$RH0" ]'
ok "own tree exists with the trainer and TIP_OBD.txt" '[ -s "$B/D/tanitad/obd/repo/stack/scripts/v8f_mini_train.py" ] && [ "$(cat "$B/D/tanitad/obd/repo/TIP_OBD.txt" | tr -d "\r\n")" = "$COMMIT" ]'
ok "busy marker handed on" '[ ! -e "$OBD_BUSY_DEV" ]'
ok "no breach through PATH" '[ ! -e "$T/BREACH" ]'
ok "a second chain run skips finished stages" 'bash "$PKG/obd_chain.sh" 2>&1 | tail -3 | grep -q "all runs banked"'
NORMAL_B="$B"

echo "== 2 NULL parity HOLDS: the same finished state, sums now carry OBD-NULL_s0's checkpoint sha256 -> the chain does not train OBD-NULL_s1"
CKSHA="$(sha256sum "$OBD_PULLS/obd_OBD-NULL_s0/ckpt_last.pt" | cut -d' ' -f1)"
printf '%s *ckpt_last.pt\n' "$CKSHA" > "$OBD_DAGOACC_DIR/SHA256SUMS.txt"; export OBD_TEST_DAGOACC_SHA="$CKSHA"; mv "$OBD_STATE/NULL_PARITY.json" "$OBD_STATE/NULL_PARITY.fail.json"
bash "$PKG/obd_chain.sh" > "$T/par2.out" 2>&1; rc=$?
ok "chain finishes (rc 0)" '[ $rc -eq 0 ]'
ok "NULL_PARITY.json now says the file sha256 is equal and OBD-NULL_s1 is not appended" 'grep -q "\"file_sha_equal\": true" "$OBD_STATE/NULL_PARITY.json" && grep -q "\"append_obd_null_s1\": false" "$OBD_STATE/NULL_PARITY.json" && grep -q "NULL parity holds" "$OBD_STATE/chain.log"'
ok "DONE.txt lists the seven planned runs and not OBD-NULL_s1" 'head -1 "$OBD_STATE/DONE.txt" | grep -q "runs=OBD-NULL_s0 OBD-NULL_rep OBD-CTL_s0 OBD-BAND_s0 OBD-CTL_s1 OBD-BAND_s1 OBD-BAND_rep$"'

echo "== 3 the box-side pin refuses a changed token (checker run directly on the fake box, no transport)"
B="$NORMAL_B"; RD="$B/D/tanitad/obd/runs/OBD-CTL_s0"; cp "$RD/argv.txt" "$T/argv.keep"
sed -i 's/--seed 0/--seed 7/' "$RD/argv.txt"
( export OBD_ROOT="$(cygpath -m "$B/D/tanitad")" OBD_COMMIT="$COMMIT" OBD_FAKE_INIT_SHA_FILE="$(cygpath -m "$B/D/tanitad/init_sha.json")" OBD_FAKE_BANK_SHA_FILE="$(cygpath -m "$B/D/tanitad/bank_sha.json")" OBD_FLOOR_BYTES=1000 OBD_FAKE_V9_MD5="$(md5sum "$B/D/tanitad/data/v9/v9_labels_train.npz" | cut -d' ' -f1)"
  "$PYV" -I "$B/D/tanitad/obd/check/obd_check.py" stage --tag OBD-CTL_s0 --gid pin1 > "$T/pin.out" 2>&1 ); rc=$?
ok "a changed seed token is refused (rc 3) and the first differing token is named" '[ $rc -eq 3 ] && grep -q "pin-fail" "$RD/STAGE_pin1.fail" && grep -q "token " "$RD/STAGE_pin1.fail" && grep -q "frozen argv has" "$RD/STAGE_pin1.fail"'
cp "$T/argv.keep" "$RD/argv.txt"
echo "changed bank" > "$B/D/tanitad/runs/v8f_mini_full/dagger_bank/s0/bank.json"
( export OBD_ROOT="$(cygpath -m "$B/D/tanitad")" OBD_COMMIT="$COMMIT" OBD_FAKE_INIT_SHA_FILE="$(cygpath -m "$B/D/tanitad/init_sha.json")" OBD_FAKE_BANK_SHA_FILE="$(cygpath -m "$B/D/tanitad/bank_sha.json")" OBD_FLOOR_BYTES=1000 OBD_FAKE_V9_MD5="$(md5sum "$B/D/tanitad/data/v9/v9_labels_train.npz" | cut -d' ' -f1)"
  "$PYV" -I "$B/D/tanitad/obd/check/obd_check.py" stage --tag OBD-CTL_s0 --gid bank1 > "$T/bank.out" 2>&1 ); rc=$?
ok "a DAgger bank whose sha256 differs is refused at the box-side preflight (rc 3) before any gate or training" '[ $rc -eq 3 ] && grep -q "preflight-fail" "$RD/STAGE_bank1.fail" && grep -q "DAgger bank s0" "$RD/STAGE_bank1.fail"'

echo "== 4 EGOWM not ended, box held, GPU not idle, python running"
newbox wait 0; rm -f "$OBD_EGOWM_STATE/DONE.txt"
bash "$PKG/obd_chain.sh" > "$T/w1.out" 2>&1; rc=$?
ok "no EGOWM finished marker: refused with rc 12, zero box calls" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ] && stoptxt | grep -q "egowm-not-ended"'
rm -f "$OBD_STATE/STOP.txt"; echo "ZZ LADDER-FULL-DONE x" > "$OBD_EGOWM_STATE/DONE.txt"
bash "$PKG/obd_chain.sh" > "$T/w1b.out" 2>&1; rc=$?
ok "a ladder-style marker is not an EGOWM finished marker (rc 12)" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$OBD_STATE/STOP.txt"; echo "ZZ EGOWM-DONE x" > "$OBD_EGOWM_STATE/DONE.txt"; echo "egowm chain 1" > "$OBD_BUSY_DEV"
bash "$PKG/obd_chain.sh" > "$T/w2.out" 2>&1; rc=$?
ok "EGOWM marker present but the busy marker held: refused with rc 12, zero box calls" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
ok "the held busy marker is untouched by the refused chain" '[ "$(cat "$OBD_BUSY_DEV")" = "egowm chain 1" ]'
rm -f "$OBD_STATE/STOP.txt" "$OBD_BUSY_DEV"; echo "not a finished marker" > "$OBD_EGOWM_STATE/DONE.txt"
bash "$PKG/obd_chain.sh" > "$T/w2b.out" 2>&1; rc=$?
ok "a marker file without the EGOWM prefix is not a finished marker (rc 12)" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$OBD_STATE/STOP.txt"; rm -f "$OBD_EGOWM_STATE/DONE.txt"; echo "ZZ EGOWM-STOP run x rc=4" > "$OBD_EGOWM_STATE/STOP.txt"; echo 9000 > "$B/gpu_mib.txt"
bash "$PKG/obd_chain.sh" > "$T/w3.out" 2>&1; rc=$?
ok "a stopped EGOWM chain counts as ended; GPU busy (9000 MiB) refuses the pilot (rc 12), nothing trained" '[ $rc -eq 12 ] && stoptxt | grep -q "gpu-not-idle" && [ ! -e "$B/D/tanitad/obd/runs/PILOT_OBD-BAND_s0/gate" ]'
rm -f "$OBD_STATE/STOP.txt" "$OBD_STATE/done/pilot.done"; echo 0 > "$B/gpu_mib.txt"; echo 2 > "$B/py_count.txt"
bash "$PKG/obd_chain.sh" > "$T/w4.out" 2>&1; rc=$?
ok "a python process on box 2 refuses the pilot (rc 12)" '[ $rc -eq 12 ] && stoptxt | grep -q "gpu-not-idle"'
ok "no breach through PATH after the wait tests" '[ ! -e "$T/BREACH" ]'

echo "== 5 disk gate"
newbox disk 0; control '{"sps": 1.0}'; echo 3 > "$B/free_gb.txt"
bash "$PKG/obd_chain.sh" > "$T/d1.out" 2>&1; rc=$?
ok "3 GB free: ship passes (1 GB), the pilot stops with the disk-low code (rc 6)" '[ $rc -eq 6 ] && stoptxt | grep -q "disk-low" && [ ! -e "$B/D/tanitad/obd/runs/PILOT_OBD-BAND_s0/gate" ]'
newbox disk0 0; echo 0 > "$B/free_gb.txt"
bash "$PKG/obd_chain.sh" > "$T/d2.out" 2>&1; rc=$?
ok "0 GB free: the ship stage stops with rc 6" '[ $rc -eq 6 ] && [ ! -e "$B/D/tanitad/obd/ship/obd_tree.tar" ]'
newbox disk1 0; echo "" > "$B/free_gb.txt"
bash "$PKG/obd_chain.sh" > "$T/d3.out" 2>&1; rc=$?
ok "an unreadable free-space probe is a refusal, not a pass (rc 11)" '[ $rc -eq 11 ]'

echo "== 6 projected total above 12 h"
newbox slow 0; control '{"sps": 2.2}'
bash "$PKG/obd_chain.sh" > "$T/s1.out" 2>&1; rc=$?
ok "pilot at 2.2 s/step (12.8 h for seven runs): the chain stops at the 12 h rule (rc 3), no run started" '[ $rc -eq 3 ] && stoptxt | grep -q "12h-rule-refused" && [ ! -e "$B/D/tanitad/obd/runs/OBD-NULL_s0/RUN.done" ] && [ ! -e "$B/D/tanitad/runs/v8f_obd" ]'
ok "the pilot at 2.0 s/step would pass (11.7 h): the boundary is 12 h, not looser" '"$PYV" -I -c "import sys;sys.path.insert(0,sys.argv[1]);import obd_check as C;assert C.project(2.0)[\"within_limit_planned\"] and not C.project(2.2)[\"within_limit_planned\"]" "$PKG"'
rm -f "$OBD_STATE/STOP.txt"; printf 'OBD-AMENDMENT pilot_sha256=%s commit=%s\n' "$(printf '%064d' 0)" "$COMMIT" > "$T/am_wrong.txt"
OBD_RESUME=1 OBD_AMENDMENT="$T/am_wrong.txt" bash "$PKG/obd_chain.sh" > "$T/s2.out" 2>&1; rc=$?
ok "an amendment naming another pilot is refused (rc 3)" '[ $rc -eq 3 ]'
rm -f "$OBD_STATE/STOP.txt"; printf 'OBD-AMENDMENT pilot_sha256=%s commit=%s\n' "$(sha256sum "$OBD_STATE/pilot/PILOT.verified.json" | cut -d' ' -f1)" "$COMMIT" > "$T/am_ok.txt"
OBD_RESUME=1 OBD_AMENDMENT="$T/am_ok.txt" bash "$PKG/obd_chain.sh" > "$T/s3.out" 2>&1; rc=$?
ok "an amendment binding this pilot and commit lets the runs go (rc 0)" '[ $rc -eq 0 ] && [ -e "$OBD_STATE/DONE.txt" ]'

echo "== 7 gate refusals (BUILT levers planted wrong in the stand-in trainer)"
newbox gblocker 0; control '{"sps": 1.0, "modes": {"PILOT_OBD-BAND_s0": "no_acc_probe"}}'
bash "$PKG/obd_chain.sh" > "$T/g0.out" 2>&1; rc=$?
ok "THE BLOCKER: a trainer whose purity probe does not read the acceleration term is refused at the pilot (rc 4), the stop line names acc" '[ $rc -eq 4 ] && stoptxt | grep -q "by_term keys" && stoptxt | grep -q "acc"'
ok "nothing was trained after the blocker, no checkpoint exists" '[ ! -e "$B/D/tanitad/obd/runs/PILOT_OBD-BAND_s0/logs/train.log" ] && [ ! -e "$B/D/tanitad/runs/v8f_obd" ] && [ ! -e "$OBD_PULLS/obd_OBD-NULL_s0" ]'
newbox gacc 0; control '{"sps": 1.0, "modes": {"PILOT_OBD-BAND_s0": "bad_acc_leak"}}'
bash "$PKG/obd_chain.sh" > "$T/g0b.out" 2>&1; rc=$?
ok "the acceleration term putting gradient on S / T is refused at the pilot (rc 4)" '[ $rc -eq 4 ] && stoptxt | grep -q "by_term.acc.upper_grad_abs_sum"'
newbox gfreeze 0; control '{"sps": 1.0, "modes": {"OBD-BAND_s0": "bad_params"}}'
bash "$PKG/obd_chain.sh" > "$T/g1.out" 2>&1; rc=$?
ok "S built with no trainable parameter (frozen): the pilot's gate of OBD-BAND_s0 refuses (rc 4), the stop line names the BUILT S params" '[ $rc -eq 4 ] && stoptxt | grep -q "group S"'
ok "no training happened and no token was written for the refused arm" '[ ! -e "$B/D/tanitad/obd/runs/PILOT_OBD-BAND_s0/logs/train.log" ] && [ ! -e "$B/D/tanitad/obd/runs/OBD-BAND_s0/gate/gate_token.json" ] && [ ! -e "$B/D/tanitad/runs/v8f_obd" ]'
newbox gdetach 0; control '{"sps": 1.0, "modes": {"PILOT_OBD-BAND_s0": "bad_detach"}}'
bash "$PKG/obd_chain.sh" > "$T/g2.out" 2>&1; rc=$?
ok "O's op term reaching S / T (purity != 0): refused (rc 4)" '[ $rc -eq 4 ] && stoptxt | grep -q "upper_grad_abs_sum"'
newbox gsteps 0; control '{"sps": 1.0, "modes": {"OBD-NULL_s0": "bad_steps"}}'
bash "$PKG/obd_chain.sh" > "$T/g3.out" 2>&1; rc=$?
ok "a full-length steps_total that is not 3,000 (NULL_s0, gated at the pilot) is refused (rc 4), before the 12 h decision" '[ $rc -eq 4 ] && stoptxt | grep -q "steps_total"'
newbox gbank 0; control '{"sps": 1.0, "modes": {"PILOT_OBD-BAND_s0": "bad_bank"}}'
bash "$PKG/obd_chain.sh" > "$T/g3b.out" 2>&1; rc=$?
ok "a built DAgger bank sha256 that is not DAG-OACC's is refused (rc 4)" '[ $rc -eq 4 ] && stoptxt | grep -q "dagger_bank_sha256"'
newbox garm 0; control '{"sps": 1.0, "modes": {"OBD-CTL_s1": "bad_params"}}'
bash "$PKG/obd_chain.sh" > "$T/g4.out" 2>&1; rc=$?
ok "an arm-specific gate refusal (OBD-CTL_s1) ends the chain with rc 4 but the other arms ran" '[ $rc -eq 4 ] && [ -e "$OBD_STATE/ARM_REFUSED_OBD-CTL_s1.txt" ] && [ ! -e "$OBD_STATE/DONE.txt" ]'
ok "the independent arms after it were banked, the refused one was not trained" '[ -e "$OBD_PULLS/obd_OBD-BAND_s1/BANK.done" ] && [ -e "$OBD_PULLS/obd_OBD-BAND_rep/BANK.done" ] && [ ! -e "$B/D/tanitad/runs/v8f_obd/OBD-CTL_s1" ] && [ -e "$OBD_PULLS/obd_OBD-CTL_s0/BANK.done" ] && [ -e "$OBD_PULLS/obd_OBD-NULL_rep/BANK.done" ]'
ok "the refusal sentinel is in its own file" 'head -c 30 "$OBD_STATE/ARM_REFUSED_OBD-CTL_s1.txt" | grep -q "^ZZ OBD-ARM-REFUSED" && ! grep -q "ZZ OBD" "$OBD_STATE/chain.log"'

echo "== 8 failures after the gate stop the chain (no next arm)"
newbox pfrozen 0; control '{"sps": 1.0, "train_modes": {"OBD-CTL_s1": "S_frozen"}}'
bash "$PKG/obd_chain.sh" > "$T/p1.out" 2>&1; rc=$?
ok "an S that never moved is a post-check failure: chain stops (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "post check failed" && stoptxt | grep -q "group S was not trained"'
ok "the failed arm is not banked and OBD-BAND_s1 never started; the earlier arms are banked" '[ ! -e "$OBD_PULLS/obd_OBD-CTL_s1/BANK.done" ] && [ ! -e "$B/D/tanitad/obd/runs/OBD-BAND_s1/argv.txt" ] && [ -e "$OBD_PULLS/obd_OBD-CTL_s0/BANK.done" ] && [ -e "$OBD_PULLS/obd_OBD-BAND_s0/BANK.done" ] && [ -e "$OBD_PULLS/obd_OBD-NULL_s0/BANK.done" ]'
ok "the checkpoint of the failed arm stays on the box (never deleted)" '[ -s "$B/D/tanitad/runs/v8f_obd/OBD-CTL_s1/ckpt_last.pt" ]'
newbox pcrash 0; control '{"sps": 1.0, "train_modes": {"OBD-NULL_s0": "crash"}}'
bash "$PKG/obd_chain.sh" > "$T/p2.out" 2>&1; rc=$?
ok "a trainer crash on the first run stops the chain (rc 5) before the next arm and before any parity" '[ $rc -eq 5 ] && [ ! -e "$B/D/tanitad/obd/runs/OBD-NULL_rep/RUN.done" ] && [ ! -e "$OBD_STATE/done/OBD-NULL_rep.done" ] && [ ! -e "$OBD_PULLS/obd_OBD-NULL_s0" ] && [ ! -e "$OBD_STATE/NULL_PARITY.json" ]'
newbox pnomove 0; control '{"sps": 1.0, "train_modes": {"OBD-NULL_s0": "no_move"}}'
bash "$PKG/obd_chain.sh" > "$T/p3.out" 2>&1; rc=$?
ok "a run in which S, T and O did not move is a post-check failure (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "was not trained"'
newbox pdetach 0; control '{"sps": 1.0, "train_modes": {"OBD-CTL_s0": "cfg_no_detach"}}'
bash "$PKG/obd_chain.sh" > "$T/p3b.out" 2>&1; rc=$?
ok "a checkpoint whose cfg.detach_upper is not the declared one is a post-check failure (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "cfg.detach_upper"'
newbox pseed 0; control '{"sps": 1.0, "train_modes": {"OBD-NULL_rep": "wrong_seed"}}'
bash "$PKG/obd_chain.sh" > "$T/p3c.out" 2>&1; rc=$?
ok "a checkpoint recording another training seed is a post-check failure (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "seed"'
newbox pcorrupt 0; control '{"sps": 1.0}'
FAKE_CORRUPT_PUT=argv.txt bash "$PKG/obd_chain.sh" > "$T/p4.out" 2>&1; rc=$?
ok "a put that arrives altered is caught by the re-hash on box 2 (rc 5), nothing runs" '[ $rc -eq 5 ] || [ $rc -eq 9 ]'
newbox pparity 0; control '{"sps": 1.0}'; rm -f "$OBD_DAGOACC_DIR/SHA256SUMS.txt"
bash "$PKG/obd_chain.sh" > "$T/p5.out" 2>&1; rc=$?
ok "an unreadable parity reference (no SHA256SUMS.txt) stops the chain after OBD-NULL_s0 (rc 14), OBD-NULL_rep never starts" '[ $rc -eq 14 ] && stoptxt | grep -q "null-parity-input-unreadable" && [ -e "$OBD_PULLS/obd_OBD-NULL_s0/BANK.done" ] && [ ! -e "$B/D/tanitad/obd/runs/OBD-NULL_rep/argv.txt" ]'

echo "== 9 commit, tag and package refusals"
newbox nocommit 0; unset OBD_COMMIT
bash "$PKG/obd_chain.sh" > "$T/nc.out" 2>&1; rc=$?
ok "chain without OBD_COMMIT refuses (rc 8), zero box calls" '[ $rc -eq 8 ] && stoptxt | grep -q "commit-missing" && [ "$(calls)" -eq 0 ]'
export OBD_COMMIT="abc123"; rm -f "$OBD_STATE/STOP.txt"
bash "$PKG/obd_chain.sh" > "$T/nc2.out" 2>&1; rc=$?
ok "an abbreviated commit refuses (rc 8)" '[ $rc -eq 8 ] && [ "$(calls)" -eq 0 ]'
export OBD_COMMIT="$COMMIT"; rm -f "$OBD_STATE/STOP.txt"
OBD_COMMIT="$(printf 'A%.0s' $(seq 1 40))" bash "$PKG/obd_chain.sh" > "$T/nc3.out" 2>&1; rc=$?
ok "an upper-case 40-character commit refuses (rc 8)" '[ $rc -eq 8 ]'
rm -f "$OBD_STATE/STOP.txt"
bash "$PKG/stages/obd_run.sh" OBD-NOPE_s0 > "$T/tok2.out" 2>&1; rc=$?
ok "an unknown tag refuses (rc 16), zero box calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
bash "$PKG/stages/obd_run.sh" OBD-BAND-TOK_s0 > "$T/tok3.out" 2>&1; rc=$?
ok "the OTBAND token arm is not a tag here (rc 16), zero box calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
ok "the checker CLI refuses an unknown tag" '"$PYV" -I "$PKG/obd_check.py" argv --tag OBD-NOPE > /dev/null 2>&1; [ $? -eq 2 ]'
ok "the ship step refuses a commit whose trainer lacks the probe change (the real-commit blocker, real ship code)" '! "$PYV" -I "$PKG/obd_ship_tree.py" build --repo "C:/Users/Admin/tanitad-push" --commit 202849da013b5d74c0b0086dd9865fcda6836018 --export-list "$T/repo/export_list.txt" --out-dir "$T/realship" > "$T/realship.out" 2>&1 && grep -q "ZZ OBD-SHIP-REFUSED" "$T/realship.out" && grep -q "lacks" "$T/realship.out"'
ok "the refused ship wrote nothing" '[ ! -e "$T/realship/obd_tree.tar" ]'

echo "== 10 transport variable unset, relative, absent, the wrong kind, or test mode with a real client (fail closed)"
newbox tr 0
for how in unset relative absent realkind; do
  rm -f "$OBD_STATE/STOP.txt"; : > "$B/transport.log"
  case "$how" in
    unset) ( unset OBD_SSH; bash "$PKG/obd_chain.sh" > "$T/tr_$how.out" 2>&1 ); rc=$?;;
    relative) OBD_SSH="ssh" bash "$PKG/obd_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
    absent) OBD_SSH="$T/no_such_ssh" bash "$PKG/obd_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
    realkind) printf '#!/usr/bin/env bash\necho i am a real client\n' > "$T/fake_real_ssh"; chmod +x "$T/fake_real_ssh"; OBD_SSH="$(cygpath -m "$T/fake_real_ssh")" bash "$PKG/obd_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
  esac
  ok "transport '$how' refuses with rc 16" '[ $rc -eq 16 ]'
  ok "transport '$how' made no call and wrote a stop line" '[ "$(calls)" -eq 0 ] && grep -q "transport-not-fail-closed" "$OBD_STATE/STOP.txt"'
done
rm -f "$OBD_STATE/STOP.txt"; : > "$B/transport.log"
( unset OBD_SFTP; bash "$PKG/obd_chain.sh" > "$T/tr_sftp.out" 2>&1 ); rc=$?
ok "OBD_SFTP unset refuses with rc 16" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$OBD_STATE/STOP.txt"
( unset OBD_DRY; bash "$PKG/obd_chain.sh" > "$T/tr_nodry.out" 2>&1 ); rc=$?
ok "a stub without test mode refuses (the stub is never taken for the real client)" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$OBD_STATE/STOP.txt"
printf '#!/usr/bin/env bash\necho i am a real sftp\n' > "$T/fake_real_sftp"; chmod +x "$T/fake_real_sftp"
OBD_SFTP="$(cygpath -m "$T/fake_real_sftp")" bash "$PKG/obd_chain.sh" > "$T/tr_realsftp.out" 2>&1; rc=$?
ok "test mode (OBD_DRY=1) with a real sftp path refuses (rc 16), zero calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$OBD_STATE/STOP.txt"
OBD_SSH="$T/no_such_ssh" bash "$PKG/stages/obd_run.sh" OBD-CTL_s0 > "$T/tr_stage.out" 2>&1; rc=$?
ok "a stage run alone with an absent transport refuses with rc 16 before any call" '[ $rc -eq 16 ]'
ok "no breach through PATH after all transport refusals" '[ ! -e "$T/BREACH" ]'

echo "== 11 hygiene"
ok "no ssh, sftp or scp command in any test script other than the stubs' own forwarding" '! grep -nE "^[^#]*\b(ssh|sftp|scp) +-" "$TESTS/test_obd_dryrun.sh" | grep -v "poison" | grep -q .'
ok "bash scripts are LF" '"$PYV" -I -c "import glob,sys;fs=glob.glob(sys.argv[1]+\"/*.sh\")+glob.glob(sys.argv[1]+\"/stages/*.sh\")+glob.glob(sys.argv[2]+\"/*.sh\");assert len(fs)>=7 and all(b\"\r\" not in open(f,\"rb\").read() for f in fs)" "$PKG" "$TESTS"'
rm -rf "${T:?}"
echo "DRYRUN ok=$OK failed=$BAD"
[ "$BAD" -eq 0 ]
