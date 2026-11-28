#!/usr/bin/env bash
# Dry run of the STOPBAL box-2 package over stubs. NOTHING here can reach box 2: STOPBAL_SSH / STOPBAL_SFTP name stub scripts by absolute path (STOPBAL_DRY=1 makes the chain refuse any file without the stub marker),
# a poison ssh/sftp/scp is put FIRST on PATH and must never run, and the "box" is a folder. The REAL job .cmd files run on Windows cmd.exe against it; only the trainer is a stand-in (tests/plant_trainer.py,
# shipped through a throw-away git repo, so the ship, tree-binding and token code run for real). CPU only, no GPU, no torch.   Usage: bash test_stopbal_dryrun.sh  -> prints "DRYRUN ok=<n> failed=<m>"
set -u
TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PKG="$(cd "$TESTS/.." && pwd)"; PKG="$(cygpath -m "$PKG" 2>/dev/null || echo "$PKG")"
PYV="${STOPBAL_TEST_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
T="${STOPBAL_TEST_TMP:-F:/TanitAD-scratch/stopbal_build/stopbal_dry_run_$$}"
case "$T" in */stopbal_dry_run_*|*/scratchpad/*|*/stopbal_dry_*) ;; *) echo "refusing an unrecognised scratch dir $T" >&2; exit 2;; esac
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
newbox() {   # newbox <name> [gpu MiB]  -> exports the dry-run environment for box <name>; the OBD chain has ended (DONE) unless the test says otherwise
  B="$T/box_$1"; "$PYV" -W ignore "$TESTS/dry_helpers.py" mkbox "$B" "$PYV" "${2:-0}" > /dev/null
  mkdir -p "$B/D/tanitad/obd/runs" "$B/D/tanitad/repo/stack"; echo "obd file" > "$B/D/tanitad/obd/runs/R1.txt"; echo "shared repo file" > "$B/D/tanitad/repo/stack/x.py"
  export STOPBAL_DRY=1 STOPBAL_SSH="$SSHS" STOPBAL_SFTP="$SFTPS" FAKE_BOX="$(cygpath -m "$B")" FAKE_PY="$PYV" STOPBAL_PY="$PYV" STOPBAL_STATE="$T/st_$1" STOPBAL_PULLS="$T/pulls_$1"
  export STOPBAL_BUSY_DEV="$T/busy_$1.txt" STOPBAL_ASIDE_DEV="$T/aside_$1" STOPBAL_B2_WIN_ROOT="$(cygpath -w "$B/D/tanitad")" STOPBAL_POLL_S=0 STOPBAL_REPO="$T/repo" STOPBAL_EXPORT_LIST="$T/repo/export_list.txt"
  export STOPBAL_COMMIT="$COMMIT" STOPBAL_NEED_C_MB=1 STOPBAL_POLLS_RUN=5 STOPBAL_POLLS_SHIP=5 STOPBAL_POLLS_CLEAN=5 STOPBAL_WAIT_POLLS=0 STOPBAL_WAIT_POLL_S=0
  export STOPBAL_OBD_STATE="$T/obd_$1"; mkdir -p "$STOPBAL_OBD_STATE"; echo "ZZ OBD-DONE 2026-11-10 00:00:00 commit=x" > "$STOPBAL_OBD_STATE/DONE.txt"
  export STOPBAL_DAGOACC_DIR="$T/dagoacc_$1"; mkdir -p "$STOPBAL_DAGOACC_DIR"; printf '%s *ckpt_last.pt\n' "$FAKESHA" > "$STOPBAL_DAGOACC_DIR/SHA256SUMS.txt"; export STOPBAL_TEST_DAGOACC_SHA="$FAKESHA"
  unset STOPBAL_RESUME STOPBAL_AMENDMENT STOPBAL_PARITY_BASIS
}
control() { printf '%s' "$1" > "$B/plant_control.json"; }
calls() { cat "$B/transport.log" 2>/dev/null | wc -l | tr -d ' '; }
stoptxt() { cat "$STOPBAL_STATE/STOP.txt" 2>/dev/null; }
treehash() { ( cd "$1" && find . -type f | sort | xargs sha256sum | sha256sum | cut -d' ' -f1 ); }
ALL7="BAL-NULL_s0 BAL-NULL_rep BAL-G_s0 BAL-L_s0 BAL-NULL_s1 BAL-G_s1 BAL-L_s1"

echo "== 1 normal flow: wait check, ship, pilot, 12 h permit, seven runs in the card's order, NULL parity (tensor basis, no DAG-OACC checkpoint here: reported not-computed, never a stop), banks, cleanups"
newbox normal 0; control '{"sps": 1.0, "startup_s": 30}'
EH0="$(treehash "$B/D/tanitad/obd")"; RH0="$(treehash "$B/D/tanitad/repo")"
bash "$PKG/stopbal_chain.sh" > "$T/normal.out" 2>&1; rc=$?
ok "chain exits 0" '[ $rc -eq 0 ]'
ok "DONE sentinel written, no stop" '[ -e "$STOPBAL_STATE/DONE.txt" ] && [ ! -e "$STOPBAL_STATE/STOP.txt" ] && head -c 20 "$STOPBAL_STATE/DONE.txt" | grep -q "^ZZ STOPBAL-DONE"'
ok "sentinel text only in its own file (chain.log carries none)" '! grep -q "ZZ STOPBAL" "$STOPBAL_STATE/chain.log"'
P="$STOPBAL_STATE/pilot/PILOT.verified.json"
ok "PILOT.verified.json names the commit" 'grep -q "\"commit\": \"$COMMIT\"" "$P"'
ok "pilot mean s/step excludes start-up (reads 1.0)" '"$PYV" -I -c "import json,sys;d=json.load(open(sys.argv[1]));assert abs(d[\"timing\"][\"mean_s_per_step\"]-1.0)<1e-9, d[\"timing\"]" "$P"'
ok "projection covers 7 runs x 3,000 steps" '"$PYV" -I -c "import json,sys;d=json.load(open(sys.argv[1]))[\"projection\"];assert d[\"steps_total\"]==21000 and d[\"n_runs_planned\"]==7 and d[\"within_limit_planned\"] and d[\"limit_h\"]==12.0" "$P"'
ok "pilot used the exact BAL-G_s0 argv plus --steps 200 and a separate out dir" '"$PYV" -I -c "import sys,json;sys.path.insert(0,sys.argv[1]);import stopbal_check as C;l=open(sys.argv[2]).read().split();assert l==C.tokens(\"PILOT_BAL-G_s0\") and l[-2:]==[\"--steps\",\"200\"] and \"pilot_out\" in l[l.index(\"--out\")+1] and \"--goal-pos-weight-stop\" in l and [x for x in l[:-2] if x not in C.tokens(\"BAL-G_s0\")]==[l[l.index(\"--out\")+1]]" "$PKG" "$STOPBAL_STATE/runs/PILOT_BAL-G_s0/argv.txt"'
ok "the pilot token binds commit and argv" '"$PYV" -I "$PKG/stopbal_check.py" verify-token --token "$STOPBAL_STATE/pilot/gate_token.json" --argv-file "$STOPBAL_STATE/runs/PILOT_BAL-G_s0/argv.txt" --tag PILOT_BAL-G_s0 --commit "$COMMIT" > /dev/null'
ok "the pilot also gated the full-length G_s0, L_s0 and NULL_s0 argvs (tokens exist before the 12 h decision)" '[ -s "$B/D/tanitad/stopbal/runs/BAL-G_s0/gate/gate_token.json" ] && [ -s "$B/D/tanitad/stopbal/runs/BAL-L_s0/gate/gate_token.json" ] && [ -s "$B/D/tanitad/stopbal/runs/BAL-NULL_s0/gate/gate_token.json" ]'
ok "pilot temp dirs deleted on the box (pilot out, the four gate outs)" '[ ! -e "$B/D/tanitad/stopbal/pilot_out/BAL-G_s0" ] && [ ! -e "$B/D/tanitad/stopbal/gate_out/PILOT_BAL-G_s0" ] && [ ! -e "$B/D/tanitad/stopbal/gate_out/BAL-NULL_s0" ] && [ ! -e "$B/D/tanitad/stopbal/gate_out/BAL-L_s0" ] && [ ! -e "$B/D/tanitad/stopbal/gate_out/BAL-G_s0" ]'
ok "pilot deletions are logged (path, bytes) and banked" 'grep -q "\"bytes\"" "$STOPBAL_STATE/pilot/CLEANUP.json" && grep -q "pilot_out" "$STOPBAL_STATE/pilot/CLEANUP.json"'
for t in $ALL7; do
  BD="$STOPBAL_PULLS/stopbal_$t"
  ok "$t banked with BANK.done and the six evidence files" '[ -e "$BD/BANK.done" ] && [ -s "$BD/ckpt_last.pt" ] && [ -s "$BD/summary.json" ] && [ -s "$BD/train_log.jsonl" ] && [ -s "$BD/gate_token.json" ] && [ -s "$BD/selfcheck.json" ] && [ -s "$BD/RUN.done" ]'
  ok "$t banked checkpoint equals the box copy (sha256)" '[ "$(sha256sum "$BD/ckpt_last.pt" | cut -d" " -f1)" = "$(sha256sum "$B/D/tanitad/runs/v8f_stopbal/$t/ckpt_last.pt" | cut -d" " -f1)" ]'
  ok "$t token binds the commit and the argv" '"$PYV" -I "$PKG/stopbal_check.py" verify-token --token "$BD/gate_token.json" --argv-file "$BD/argv.txt" --tag $t --commit "$COMMIT" > /dev/null'
  ok "$t RUN.done carries the moved identity (S, T and O moved)" 'grep -q "\"moved_identity\"" "$BD/RUN.done" && grep -q "\"S\"" "$BD/RUN.done" && grep -q "\"O\"" "$BD/RUN.done"'
  ok "$t gate dir on the box deleted, the checkpoint kept" '[ ! -e "$B/D/tanitad/stopbal/gate_out/$t" ] && [ -s "$B/D/tanitad/runs/v8f_stopbal/$t/ckpt_last.pt" ]'
  ok "$t cleanup logged (path, bytes) in the bank" 'grep -q "\"bytes\"" "$BD/CLEANUP.json" || grep -q "deleted_own_temp\": \[\]" "$BD/CLEANUP.json"'
done
ok "the seven tokens bind seven distinct argvs" '[ "$(cat "$STOPBAL_PULLS"/stopbal_*/gate_token.json | grep -o "\"argv_sha256\": \"[0-9a-f]*\"" | sort -u | wc -l | tr -d " ")" -eq 7 ]'
ok "run order in chain.log is the card's: NULL_s0, NULL_rep, G_s0, L_s0, NULL_s1, G_s1, L_s1" '[ "$(grep -o "done BAL-[A-Za-z0-9_-]*" "$STOPBAL_STATE/chain.log" | tr "\n" " ")" = "done BAL-NULL_s0 done BAL-NULL_rep done BAL-G_s0 done BAL-L_s0 done BAL-NULL_s1 done BAL-G_s1 done BAL-L_s1 " ]'
ok "NULL_PARITY.json exists: not-computed (tensor basis, no DAG-OACC checkpoint on this dev box), and the chain still finished (a parity failure is never a stop)" 'grep -q "\"status\": \"not-computed\"" "$STOPBAL_STATE/NULL_PARITY.json" && grep -q "\"null_parity\": \"UNKNOWN\"" "$STOPBAL_STATE/NULL_PARITY.json" && grep -q "NULL parity (reported, never a stop; rc 2)" "$STOPBAL_STATE/chain.log" && [ -e "$STOPBAL_STATE/DONE.txt" ]'
ok "no run was added or skipped by the parity result (DONE lists exactly the seven)" 'head -1 "$STOPBAL_STATE/DONE.txt" | grep -q "runs=BAL-NULL_s0 BAL-NULL_rep BAL-G_s0 BAL-L_s0 BAL-NULL_s1 BAL-G_s1 BAL-L_s1$"'
ok "the parity ran after BAL-NULL_s0 and before BAL-NULL_rep started" '[ "$(grep -n "NULL parity (reported" "$STOPBAL_STATE/chain.log" | cut -d: -f1)" -lt "$(grep -n "start BAL-NULL_rep" "$STOPBAL_STATE/chain.log" | cut -d: -f1)" ]'
ok "the banked checkpoints carry the arm's own stop weights (NULL 1.0 / 1.0, G 4.5616 / 1.0, L 1.0 / 4.6991) and no detach" '"$PYV" -I -c "import pickle,sys;g=lambda t:pickle.load(open(sys.argv[1]+\"/stopbal_\"+t+\"/ckpt_last.pt\",\"rb\"))[\"cfg\"];w=lambda t:(g(t)[\"goal_pos_weight_stop\"],g(t)[\"lon_class_weight_stop\"]);assert w(\"BAL-NULL_s0\")==(1.0,1.0) and w(\"BAL-NULL_rep\")==(1.0,1.0) and w(\"BAL-NULL_s1\")==(1.0,1.0) and w(\"BAL-G_s0\")==(4.5616,1.0) and w(\"BAL-G_s1\")==(4.5616,1.0) and w(\"BAL-L_s0\")==(1.0,4.6991) and w(\"BAL-L_s1\")==(1.0,4.6991) and not any(g(t)[\"detach_upper\"] for t in sys.argv[2:])" "$STOPBAL_PULLS" BAL-NULL_s0 BAL-G_s0 BAL-L_s1'
ok "the box saw only stub traffic and no unknown command" '[ "$(calls)" -gt 100 ] && ! grep -q "UNKNOWN COMMAND" "$B/transport.log"'
ok "disk was read before every stage (>= 9 reads: ship, pilot, seven runs)" '[ "$(grep -c "Get-PSDrive D" "$B/transport.log")" -ge 9 ]'
ok "isolation: the OBD folder on the box is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/obd")" = "$EH0" ]'
ok "isolation: the shared D:\\tanitad\\repo is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/repo")" = "$RH0" ]'
ok "own tree exists with the trainer and TIP_STOPBAL.txt" '[ -s "$B/D/tanitad/stopbal/repo/stack/scripts/v8f_mini_train.py" ] && [ "$(cat "$B/D/tanitad/stopbal/repo/TIP_STOPBAL.txt" | tr -d "\r\n")" = "$COMMIT" ]'
ok "busy marker handed on" '[ ! -e "$STOPBAL_BUSY_DEV" ]'
ok "no breach through PATH" '[ ! -e "$T/BREACH" ]'
ok "a second chain run skips finished stages" 'bash "$PKG/stopbal_chain.sh" 2>&1 | tail -3 | grep -q "all runs banked"'
NORMAL_B="$B"

echo "== 2 parity, basis file: the stand-in checkpoint is not DAG-OACC's, so DIFFERS is reported and NOTHING is appended or stopped; then the sums carry BAL-NULL_s0's sha256 -> EQUAL"
mv "$STOPBAL_STATE/NULL_PARITY.json" "$STOPBAL_STATE/NULL_PARITY.nc.json"
STOPBAL_PARITY_BASIS=file bash "$PKG/stopbal_chain.sh" > "$T/par1.out" 2>&1; rc=$?
ok "chain finishes (rc 0)" '[ $rc -eq 0 ]'
ok "NULL_PARITY.json says computed, file sha256 not equal, DIFFERS; there is no append key" 'grep -q "\"status\": \"computed\"" "$STOPBAL_STATE/NULL_PARITY.json" && grep -q "\"file_sha_equal\": false" "$STOPBAL_STATE/NULL_PARITY.json" && grep -q "\"null_parity\": \"DIFFERS\"" "$STOPBAL_STATE/NULL_PARITY.json" && ! grep -q "append" "$STOPBAL_STATE/NULL_PARITY.json"'
ok "DONE.txt still lists exactly the seven planned runs" 'head -1 "$STOPBAL_STATE/DONE.txt" | grep -q "runs=BAL-NULL_s0 BAL-NULL_rep BAL-G_s0 BAL-L_s0 BAL-NULL_s1 BAL-G_s1 BAL-L_s1$"'
CKSHA="$(sha256sum "$STOPBAL_PULLS/stopbal_BAL-NULL_s0/ckpt_last.pt" | cut -d' ' -f1)"
printf '%s *ckpt_last.pt\n' "$CKSHA" > "$STOPBAL_DAGOACC_DIR/SHA256SUMS.txt"; export STOPBAL_TEST_DAGOACC_SHA="$CKSHA"; mv "$STOPBAL_STATE/NULL_PARITY.json" "$STOPBAL_STATE/NULL_PARITY.diff.json"
STOPBAL_PARITY_BASIS=file bash "$PKG/stopbal_chain.sh" > "$T/par2.out" 2>&1; rc=$?
ok "chain finishes (rc 0)" '[ $rc -eq 0 ]'
ok "NULL_PARITY.json now says the file sha256 is equal: EQUAL" 'grep -q "\"file_sha_equal\": true" "$STOPBAL_STATE/NULL_PARITY.json" && grep -q "\"null_parity\": \"EQUAL\"" "$STOPBAL_STATE/NULL_PARITY.json"'
unset STOPBAL_TEST_DAGOACC_SHA

echo "== 3 the box-side pin refuses a changed token (checker run directly on the fake box, no transport)"
B="$NORMAL_B"; RD="$B/D/tanitad/stopbal/runs/BAL-L_s0"; cp "$RD/argv.txt" "$T/argv.keep"
sed -i 's/--seed 0/--seed 7/' "$RD/argv.txt"
( export STOPBAL_ROOT="$(cygpath -m "$B/D/tanitad")" STOPBAL_COMMIT="$COMMIT" STOPBAL_FAKE_INIT_SHA_FILE="$(cygpath -m "$B/D/tanitad/init_sha.json")" STOPBAL_FAKE_BANK_SHA_FILE="$(cygpath -m "$B/D/tanitad/bank_sha.json")" STOPBAL_FLOOR_BYTES=1000 STOPBAL_FAKE_V9_MD5="$(md5sum "$B/D/tanitad/data/v9/v9_labels_train.npz" | cut -d' ' -f1)"
  "$PYV" -I "$B/D/tanitad/stopbal/check/stopbal_check.py" stage --tag BAL-L_s0 --gid pin1 > "$T/pin.out" 2>&1 ); rc=$?
ok "a changed seed token is refused (rc 3) and the first differing token is named" '[ $rc -eq 3 ] && grep -q "pin-fail" "$RD/STAGE_pin1.fail" && grep -q "token " "$RD/STAGE_pin1.fail" && grep -q "frozen argv has" "$RD/STAGE_pin1.fail"'
cp "$T/argv.keep" "$RD/argv.txt"
echo "changed bank" > "$B/D/tanitad/runs/v8f_mini_full/dagger_bank/s0/bank.json"
( export STOPBAL_ROOT="$(cygpath -m "$B/D/tanitad")" STOPBAL_COMMIT="$COMMIT" STOPBAL_FAKE_INIT_SHA_FILE="$(cygpath -m "$B/D/tanitad/init_sha.json")" STOPBAL_FAKE_BANK_SHA_FILE="$(cygpath -m "$B/D/tanitad/bank_sha.json")" STOPBAL_FLOOR_BYTES=1000 STOPBAL_FAKE_V9_MD5="$(md5sum "$B/D/tanitad/data/v9/v9_labels_train.npz" | cut -d' ' -f1)"
  "$PYV" -I "$B/D/tanitad/stopbal/check/stopbal_check.py" stage --tag BAL-L_s0 --gid bank1 > "$T/bank.out" 2>&1 ); rc=$?
ok "a DAgger bank whose sha256 differs is refused at the box-side preflight (rc 3) before any gate or training" '[ $rc -eq 3 ] && grep -q "preflight-fail" "$RD/STAGE_bank1.fail" && grep -q "DAgger bank s0" "$RD/STAGE_bank1.fail"'

echo "== 4 OBD not ended, box held, GPU not idle, python running"
newbox wait 0; rm -f "$STOPBAL_OBD_STATE/DONE.txt"
bash "$PKG/stopbal_chain.sh" > "$T/w1.out" 2>&1; rc=$?
ok "no OBD finished marker: refused with rc 12, zero box calls" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ] && stoptxt | grep -q "obd-not-ended"'
rm -f "$STOPBAL_STATE/STOP.txt"; echo "ZZ LADDER-FULL-DONE x" > "$STOPBAL_OBD_STATE/DONE.txt"
bash "$PKG/stopbal_chain.sh" > "$T/w1b.out" 2>&1; rc=$?
ok "a ladder-style marker is not an OBD finished marker (rc 12)" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$STOPBAL_STATE/STOP.txt"; echo "ZZ OBD-DONE x" > "$STOPBAL_OBD_STATE/DONE.txt"; echo "obd chain 1" > "$STOPBAL_BUSY_DEV"
bash "$PKG/stopbal_chain.sh" > "$T/w2.out" 2>&1; rc=$?
ok "OBD marker present but the busy marker held: refused with rc 12, zero box calls" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
ok "the held busy marker is untouched by the refused chain" '[ "$(cat "$STOPBAL_BUSY_DEV")" = "obd chain 1" ]'
rm -f "$STOPBAL_STATE/STOP.txt" "$STOPBAL_BUSY_DEV"; echo "not a finished marker" > "$STOPBAL_OBD_STATE/DONE.txt"
bash "$PKG/stopbal_chain.sh" > "$T/w2b.out" 2>&1; rc=$?
ok "a marker file without the OBD prefix is not a finished marker (rc 12)" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$STOPBAL_STATE/STOP.txt"; rm -f "$STOPBAL_OBD_STATE/DONE.txt"; echo "ZZ OBD-STOP run x rc=4" > "$STOPBAL_OBD_STATE/STOP.txt"; echo 9000 > "$B/gpu_mib.txt"
bash "$PKG/stopbal_chain.sh" > "$T/w3.out" 2>&1; rc=$?
ok "a stopped OBD chain counts as ended; GPU busy (9000 MiB) refuses the pilot (rc 12), nothing trained" '[ $rc -eq 12 ] && stoptxt | grep -q "gpu-not-idle" && [ ! -e "$B/D/tanitad/stopbal/runs/PILOT_BAL-G_s0/gate" ]'
rm -f "$STOPBAL_STATE/STOP.txt" "$STOPBAL_STATE/done/pilot.done"; echo 0 > "$B/gpu_mib.txt"; echo 2 > "$B/py_count.txt"
bash "$PKG/stopbal_chain.sh" > "$T/w4.out" 2>&1; rc=$?
ok "a python process on box 2 refuses the pilot (rc 12)" '[ $rc -eq 12 ] && stoptxt | grep -q "gpu-not-idle"'
ok "no breach through PATH after the wait tests" '[ ! -e "$T/BREACH" ]'

echo "== 5 disk gate"
newbox disk 0; control '{"sps": 1.0}'; echo 3 > "$B/free_gb.txt"
bash "$PKG/stopbal_chain.sh" > "$T/d1.out" 2>&1; rc=$?
ok "3 GB free: ship passes (1 GB), the pilot stops with the disk-low code (rc 6)" '[ $rc -eq 6 ] && stoptxt | grep -q "disk-low" && [ ! -e "$B/D/tanitad/stopbal/runs/PILOT_BAL-G_s0/gate" ]'
newbox disk0 0; echo 0 > "$B/free_gb.txt"
bash "$PKG/stopbal_chain.sh" > "$T/d2.out" 2>&1; rc=$?
ok "0 GB free: the ship stage stops with rc 6" '[ $rc -eq 6 ] && [ ! -e "$B/D/tanitad/stopbal/ship/stopbal_tree.tar" ]'
newbox disk1 0; echo "" > "$B/free_gb.txt"
bash "$PKG/stopbal_chain.sh" > "$T/d3.out" 2>&1; rc=$?
ok "an unreadable free-space probe is a refusal, not a pass (rc 11)" '[ $rc -eq 11 ]'

echo "== 6 projected total above 12 h"
newbox slow 0; control '{"sps": 2.2}'
bash "$PKG/stopbal_chain.sh" > "$T/s1.out" 2>&1; rc=$?
ok "pilot at 2.2 s/step (12.8 h for seven runs): the chain stops at the 12 h rule (rc 3), no run started" '[ $rc -eq 3 ] && stoptxt | grep -q "12h-rule-refused" && [ ! -e "$B/D/tanitad/stopbal/runs/BAL-NULL_s0/RUN.done" ] && [ ! -e "$B/D/tanitad/runs/v8f_stopbal" ]'
ok "the pilot at 2.0 s/step would pass (11.7 h): the boundary is 12 h, not looser" '"$PYV" -I -c "import sys;sys.path.insert(0,sys.argv[1]);import stopbal_check as C;assert C.project(2.0)[\"within_limit_planned\"] and not C.project(2.2)[\"within_limit_planned\"]" "$PKG"'
rm -f "$STOPBAL_STATE/STOP.txt"; printf 'STOPBAL-AMENDMENT pilot_sha256=%s commit=%s\n' "$(printf '%064d' 0)" "$COMMIT" > "$T/am_wrong.txt"
STOPBAL_RESUME=1 STOPBAL_AMENDMENT="$T/am_wrong.txt" bash "$PKG/stopbal_chain.sh" > "$T/s2.out" 2>&1; rc=$?
ok "an amendment naming another pilot is refused (rc 3)" '[ $rc -eq 3 ]'
rm -f "$STOPBAL_STATE/STOP.txt"; printf 'STOPBAL-AMENDMENT pilot_sha256=%s commit=%s\n' "$(sha256sum "$STOPBAL_STATE/pilot/PILOT.verified.json" | cut -d' ' -f1)" "$COMMIT" > "$T/am_ok.txt"
STOPBAL_RESUME=1 STOPBAL_AMENDMENT="$T/am_ok.txt" bash "$PKG/stopbal_chain.sh" > "$T/s3.out" 2>&1; rc=$?
ok "an amendment binding this pilot and commit lets the runs go (rc 0)" '[ $rc -eq 0 ] && [ -e "$STOPBAL_STATE/DONE.txt" ]'

echo "== 7 gate refusals (BUILT levers planted wrong in the stand-in trainer)"
newbox glever 0; control '{"sps": 1.0, "modes": {"PILOT_BAL-G_s0": "lever_not_built"}}'
bash "$PKG/stopbal_chain.sh" > "$T/g0.out" 2>&1; rc=$?
ok "THE POINT OF THE CARD'S CONTROL 1: a trainer whose BUILT loss reads 1.0 for its own lever is refused at the pilot (rc 4); the stop line names the lever, the value read and the one wanted" '[ $rc -eq 4 ] && stoptxt | grep -q "BUILT lever goal_pos_weight_stop = 1.0" && stoptxt | grep -q "4.5616"'
ok "nothing was trained after it, no checkpoint exists" '[ ! -e "$B/D/tanitad/stopbal/runs/PILOT_BAL-G_s0/logs/train.log" ] && [ ! -e "$B/D/tanitad/runs/v8f_stopbal" ] && [ ! -e "$STOPBAL_PULLS/stopbal_BAL-NULL_s0" ]'
newbox gdead 0; control '{"sps": 1.0, "modes": {"PILOT_BAL-G_s0": "dead_goal_head"}}'
bash "$PKG/stopbal_chain.sh" > "$T/g0b.out" 2>&1; rc=$?
ok "T's goal head receiving no gradient (G-LIVE) is refused at the pilot (rc 4) and the stop line says so" '[ $rc -eq 4 ] && stoptxt | grep -q "goal / lon head"'
newbox gfreeze 0; control '{"sps": 1.0, "modes": {"BAL-L_s0": "bad_params"}}'
bash "$PKG/stopbal_chain.sh" > "$T/g1.out" 2>&1; rc=$?
ok "S built with no trainable parameter (frozen): the pilot's gate of BAL-L_s0 refuses (rc 4), the stop line names the BUILT S params" '[ $rc -eq 4 ] && stoptxt | grep -q "group S"'
ok "no training happened and no token was written for the refused arm" '[ ! -e "$B/D/tanitad/stopbal/runs/PILOT_BAL-G_s0/logs/train.log" ] && [ ! -e "$B/D/tanitad/stopbal/runs/BAL-L_s0/gate/gate_token.json" ] && [ ! -e "$B/D/tanitad/runs/v8f_stopbal" ]'
newbox gsteps 0; control '{"sps": 1.0, "modes": {"BAL-NULL_s0": "bad_steps"}}'
bash "$PKG/stopbal_chain.sh" > "$T/g3.out" 2>&1; rc=$?
ok "a full-length steps_total that is not 3,000 (NULL_s0, gated at the pilot) is refused (rc 4), before the 12 h decision" '[ $rc -eq 4 ] && stoptxt | grep -q "steps_total"'
newbox gbank 0; control '{"sps": 1.0, "modes": {"PILOT_BAL-G_s0": "bad_bank"}}'
bash "$PKG/stopbal_chain.sh" > "$T/g3b.out" 2>&1; rc=$?
ok "a built DAgger bank sha256 that is not DAG-OACC's is refused (rc 4)" '[ $rc -eq 4 ] && stoptxt | grep -q "dagger_bank_sha256"'
newbox garm 0; control '{"sps": 1.0, "modes": {"BAL-G_s1": "lever_not_built"}}'
bash "$PKG/stopbal_chain.sh" > "$T/g4.out" 2>&1; rc=$?
ok "an arm-specific gate refusal (BAL-G_s1) ends the chain with rc 4 but the other arms ran" '[ $rc -eq 4 ] && [ -e "$STOPBAL_STATE/ARM_REFUSED_BAL-G_s1.txt" ] && [ ! -e "$STOPBAL_STATE/DONE.txt" ]'
ok "the independent arm after it (L_s1) was banked, the refused one was not trained, the earlier six were banked" '[ -e "$STOPBAL_PULLS/stopbal_BAL-L_s1/BANK.done" ] && [ ! -e "$B/D/tanitad/runs/v8f_stopbal/BAL-G_s1" ] && [ -e "$STOPBAL_PULLS/stopbal_BAL-G_s0/BANK.done" ] && [ -e "$STOPBAL_PULLS/stopbal_BAL-L_s0/BANK.done" ] && [ -e "$STOPBAL_PULLS/stopbal_BAL-NULL_s1/BANK.done" ] && [ -e "$STOPBAL_PULLS/stopbal_BAL-NULL_rep/BANK.done" ]'
ok "the refusal sentinel is in its own file" 'head -c 30 "$STOPBAL_STATE/ARM_REFUSED_BAL-G_s1.txt" | grep -q "^ZZ STOPBAL-ARM-REFUSED" && ! grep -q "ZZ STOPBAL" "$STOPBAL_STATE/chain.log"'

echo "== 8 failures after the gate stop the chain (no next arm)"
newbox pfrozen 0; control '{"sps": 1.0, "train_modes": {"BAL-G_s1": "S_frozen"}}'
bash "$PKG/stopbal_chain.sh" > "$T/p1.out" 2>&1; rc=$?
ok "an S that never moved is a post-check failure: chain stops (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "post check failed" && stoptxt | grep -q "group S was not trained"'
ok "the failed arm is not banked and BAL-L_s1 never started; the earlier arms are banked" '[ ! -e "$STOPBAL_PULLS/stopbal_BAL-G_s1/BANK.done" ] && [ ! -e "$B/D/tanitad/stopbal/runs/BAL-L_s1/argv.txt" ] && [ -e "$STOPBAL_PULLS/stopbal_BAL-G_s0/BANK.done" ] && [ -e "$STOPBAL_PULLS/stopbal_BAL-L_s0/BANK.done" ] && [ -e "$STOPBAL_PULLS/stopbal_BAL-NULL_s0/BANK.done" ]'
ok "the checkpoint of the failed arm stays on the box (never deleted)" '[ -s "$B/D/tanitad/runs/v8f_stopbal/BAL-G_s1/ckpt_last.pt" ]'
newbox pcrash 0; control '{"sps": 1.0, "train_modes": {"BAL-NULL_s0": "crash"}}'
bash "$PKG/stopbal_chain.sh" > "$T/p2.out" 2>&1; rc=$?
ok "a trainer crash on the first run stops the chain (rc 5) before the next arm and before any parity" '[ $rc -eq 5 ] && [ ! -e "$B/D/tanitad/stopbal/runs/BAL-NULL_rep/RUN.done" ] && [ ! -e "$STOPBAL_STATE/done/BAL-NULL_rep.done" ] && [ ! -e "$STOPBAL_PULLS/stopbal_BAL-NULL_s0" ] && [ ! -e "$STOPBAL_STATE/NULL_PARITY.json" ]'
newbox pnomove 0; control '{"sps": 1.0, "train_modes": {"BAL-NULL_s0": "no_move"}}'
bash "$PKG/stopbal_chain.sh" > "$T/p3.out" 2>&1; rc=$?
ok "a run in which S, T and O did not move is a post-check failure (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "was not trained"'
newbox pweight 0; control '{"sps": 1.0, "train_modes": {"BAL-G_s0": "cfg_weight_lost"}}'
bash "$PKG/stopbal_chain.sh" > "$T/p3b.out" 2>&1; rc=$?
ok "a checkpoint whose config does not carry the declared stop weight is a post-check failure (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "cfg.goal_pos_weight_stop"'
newbox pseed 0; control '{"sps": 1.0, "train_modes": {"BAL-NULL_rep": "wrong_seed"}}'
bash "$PKG/stopbal_chain.sh" > "$T/p3c.out" 2>&1; rc=$?
ok "a checkpoint recording another training seed is a post-check failure (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "seed"'
newbox pcorrupt 0; control '{"sps": 1.0}'
FAKE_CORRUPT_PUT=argv.txt bash "$PKG/stopbal_chain.sh" > "$T/p4.out" 2>&1; rc=$?
ok "a put that arrives altered is caught by the re-hash on box 2 (rc 5), nothing runs" '[ $rc -eq 5 ] || [ $rc -eq 9 ]'
newbox pparity 0; control '{"sps": 1.0}'; rm -f "$STOPBAL_DAGOACC_DIR/SHA256SUMS.txt"
bash "$PKG/stopbal_chain.sh" > "$T/p5.out" 2>&1; rc=$?
ok "an unreadable parity reference (no SHA256SUMS.txt) is NOT a stop: the chain finishes (rc 0), all seven runs are banked, NULL_PARITY.json says not-computed" '[ $rc -eq 0 ] && [ -e "$STOPBAL_STATE/DONE.txt" ] && [ -e "$STOPBAL_PULLS/stopbal_BAL-L_s1/BANK.done" ] && grep -q "\"status\": \"not-computed\"" "$STOPBAL_STATE/NULL_PARITY.json"'

echo "== 9 commit, tag and package refusals"
newbox nocommit 0; unset STOPBAL_COMMIT
bash "$PKG/stopbal_chain.sh" > "$T/nc.out" 2>&1; rc=$?
ok "chain without STOPBAL_COMMIT refuses (rc 8), zero box calls" '[ $rc -eq 8 ] && stoptxt | grep -q "commit-missing" && [ "$(calls)" -eq 0 ]'
export STOPBAL_COMMIT="abc123"; rm -f "$STOPBAL_STATE/STOP.txt"
bash "$PKG/stopbal_chain.sh" > "$T/nc2.out" 2>&1; rc=$?
ok "an abbreviated commit refuses (rc 8)" '[ $rc -eq 8 ] && [ "$(calls)" -eq 0 ]'
export STOPBAL_COMMIT="$COMMIT"; rm -f "$STOPBAL_STATE/STOP.txt"
STOPBAL_COMMIT="$(printf 'A%.0s' $(seq 1 40))" bash "$PKG/stopbal_chain.sh" > "$T/nc3.out" 2>&1; rc=$?
ok "an upper-case 40-character commit refuses (rc 8)" '[ $rc -eq 8 ]'
rm -f "$STOPBAL_STATE/STOP.txt"
bash "$PKG/stages/stopbal_run.sh" STOPBAL-NOPE_s0 > "$T/tok2.out" 2>&1; rc=$?
ok "an unknown tag refuses (rc 16), zero box calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
bash "$PKG/stages/stopbal_run.sh" BAL-XX_s0 > "$T/tok3.out" 2>&1; rc=$?
ok "an OBD-package tag (OBD-NULL_s0) is not a tag here (rc 16), zero box calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ] && { bash "$PKG/stages/stopbal_run.sh" OBD-NULL_s0 > "$T/tok4.out" 2>&1; [ $? -eq 16 ]; } && [ "$(calls)" -eq 0 ]'
ok "the checker CLI refuses an unknown tag" '"$PYV" -I "$PKG/stopbal_check.py" argv --tag STOPBAL-NOPE > /dev/null 2>&1; [ $? -eq 2 ]'
ok "the ship step refuses the PARENT 5514a684 (its trainer lacks both new flags): real ship code, real mirror" '! "$PYV" -I "$PKG/stopbal_ship_tree.py" build --repo "C:/Users/Admin/tanitad-push" --commit 5514a6840b57ad115dcadeb7e264b10f5c1d0081 --export-list "$T/repo/export_list.txt" --out-dir "$T/realship" > "$T/realship.out" 2>&1 && grep -q "ZZ STOPBAL-SHIP-REFUSED" "$T/realship.out" && grep -q "goal-pos-weight-stop" "$T/realship.out"'
ok "the refused ship wrote nothing" '[ ! -e "$T/realship/stopbal_tree.tar" ]'
ok "the ship step ACCEPTS the code of record 2bca1c52 (real ship code, real mirror) and its manifest names it" '"$PYV" -I "$PKG/stopbal_ship_tree.py" build --repo "C:/Users/Admin/tanitad-push" --commit 2bca1c52abbc2346d7380d33ae78b324cbda9df7 --export-list "$T/repo/export_list.txt" --out-dir "$T/realship_ok" > "$T/realship_ok.out" 2>&1 && grep -q "\"commit\": \"2bca1c52abbc2346d7380d33ae78b324cbda9df7\"" "$T/realship_ok/stopbal_tree_manifest.json" && grep -q "stack/scripts/v8f_mini_train.py" "$T/realship_ok/stopbal_tree_manifest.json"'

echo "== 10 transport variable unset, relative, absent, the wrong kind, or test mode with a real client (fail closed)"
newbox tr 0
for how in unset relative absent realkind; do
  rm -f "$STOPBAL_STATE/STOP.txt"; : > "$B/transport.log"
  case "$how" in
    unset) ( unset STOPBAL_SSH; bash "$PKG/stopbal_chain.sh" > "$T/tr_$how.out" 2>&1 ); rc=$?;;
    relative) STOPBAL_SSH="ssh" bash "$PKG/stopbal_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
    absent) STOPBAL_SSH="$T/no_such_ssh" bash "$PKG/stopbal_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
    realkind) printf '#!/usr/bin/env bash\necho i am a real client\n' > "$T/fake_real_ssh"; chmod +x "$T/fake_real_ssh"; STOPBAL_SSH="$(cygpath -m "$T/fake_real_ssh")" bash "$PKG/stopbal_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
  esac
  ok "transport '$how' refuses with rc 16" '[ $rc -eq 16 ]'
  ok "transport '$how' made no call and wrote a stop line" '[ "$(calls)" -eq 0 ] && grep -q "transport-not-fail-closed" "$STOPBAL_STATE/STOP.txt"'
done
rm -f "$STOPBAL_STATE/STOP.txt"; : > "$B/transport.log"
( unset STOPBAL_SFTP; bash "$PKG/stopbal_chain.sh" > "$T/tr_sftp.out" 2>&1 ); rc=$?
ok "STOPBAL_SFTP unset refuses with rc 16" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$STOPBAL_STATE/STOP.txt"
( unset STOPBAL_DRY; bash "$PKG/stopbal_chain.sh" > "$T/tr_nodry.out" 2>&1 ); rc=$?
ok "a stub without test mode refuses (the stub is never taken for the real client)" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$STOPBAL_STATE/STOP.txt"
printf '#!/usr/bin/env bash\necho i am a real sftp\n' > "$T/fake_real_sftp"; chmod +x "$T/fake_real_sftp"
STOPBAL_SFTP="$(cygpath -m "$T/fake_real_sftp")" bash "$PKG/stopbal_chain.sh" > "$T/tr_realsftp.out" 2>&1; rc=$?
ok "test mode (STOPBAL_DRY=1) with a real sftp path refuses (rc 16), zero calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$STOPBAL_STATE/STOP.txt"
STOPBAL_SSH="$T/no_such_ssh" bash "$PKG/stages/stopbal_run.sh" BAL-L_s0 > "$T/tr_stage.out" 2>&1; rc=$?
ok "a stage run alone with an absent transport refuses with rc 16 before any call" '[ $rc -eq 16 ]'
ok "no breach through PATH after all transport refusals" '[ ! -e "$T/BREACH" ]'

echo "== 11 hygiene"
ok "no ssh, sftp or scp command in any test script other than the stubs' own forwarding" '! grep -nE "^[^#]*\b(ssh|sftp|scp) +-" "$TESTS/test_stopbal_dryrun.sh" | grep -v "poison" | grep -q .'
ok "bash scripts are LF" '"$PYV" -I -c "import glob,sys;fs=glob.glob(sys.argv[1]+\"/*.sh\")+glob.glob(sys.argv[1]+\"/stages/*.sh\")+glob.glob(sys.argv[2]+\"/*.sh\");assert len(fs)>=7 and all(b\"\r\" not in open(f,\"rb\").read() for f in fs)" "$PKG" "$TESTS"'
rm -rf "${T:?}"
echo "DRYRUN ok=$OK failed=$BAD"
[ "$BAD" -eq 0 ]
