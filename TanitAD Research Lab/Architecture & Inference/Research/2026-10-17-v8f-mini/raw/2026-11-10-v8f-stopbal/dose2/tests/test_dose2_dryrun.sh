#!/usr/bin/env bash
# Dry run of the STOPBAL second-dose chain (dose2_chain.sh) over the first dose's stubs. NOTHING here can reach box 2: STOPBAL_SSH / STOPBAL_SFTP name the first dose's stub scripts by absolute path
# (STOPBAL_DRY=1 makes the chain refuse any file without the stub marker), a poison ssh/sftp/scp is put FIRST on PATH and must never run, and the "box" is a folder. The REAL job .cmd files run on Windows cmd.exe
# against it; only the trainer is a stand-in (box2/tests/plant_trainer.py, shipped through a throw-away git repo). CPU only. The first dose's helpers, stubs and stand-in are reached by path, not edited.
# Scratch only under F:/TanitAD-scratch/dose2_build/ (deleted at the end by literal path).   Usage: bash test_dose2_dryrun.sh  -> prints "DRYRUN2 ok=<n> failed=<m>"
set -u
TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PKG="$(cd "$TESTS/.." && pwd)"; PKG="$(cygpath -m "$PKG" 2>/dev/null || echo "$PKG")"
BOX2="$(cd "$TESTS/../../box2" && pwd)"; BOX2="$(cygpath -m "$BOX2" 2>/dev/null || echo "$BOX2")"; BT="$BOX2/tests"
PYV="${STOPBAL_TEST_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
T="${STOPBAL_TEST_TMP:-F:/TanitAD-scratch/dose2_build/dose2_dry_run_$$}"
case "$T" in F:/TanitAD-scratch/dose2_build/dose2_dry_run_*) ;; *) echo "refusing an unrecognised scratch dir $T" >&2; exit 2;; esac
[ ! -e "$T" ] || { echo "scratch dir $T already exists; remove it first" >&2; exit 2; }
mkdir -p "$T/poison"
for n in ssh sftp scp; do printf '#!/usr/bin/env bash\necho BREACH-%s >> "%s/BREACH"\nexit 99\n' "$n" "$T" > "$T/poison/$n"; chmod +x "$T/poison/$n"; done
export PATH="$T/poison:$PATH"
export CUDA_VISIBLE_DEVICES=-1
OK=0; BAD=0
ok() { if eval "$2"; then OK=$((OK+1)); else BAD=$((BAD+1)); echo "FAIL: $1"; fi; }
SSHS="$(cygpath -m "$BT/ssh_stub.sh")"; SFTPS="$(cygpath -m "$BT/sftp_stub.sh")"
FAKESHA="$(printf 'f%.0s' $(seq 1 64))"
COMMIT="$("$PYV" -W ignore "$BT/dry_helpers.py" mkrepo "$T/repo")"
ok "throw-away repo has a 40-hex commit" '[ ${#COMMIT} -eq 40 ]'
plant_null_banks() {   # plant_null_banks <pulls dir>: three whole fake NULL banks (what the first dose banked, in miniature)
  "$PYV" -I -c "
import hashlib, json, os, sys
pulls = sys.argv[1]
for t in ('BAL-NULL_s0', 'BAL-NULL_s1', 'BAL-NULL_rep'):
    d = os.path.join(pulls, 'stopbal_' + t); os.makedirs(d, exist_ok=True)
    ck = ('ckpt of ' + t).encode(); open(os.path.join(d, 'ckpt_last.pt'), 'wb').write(ck); sh = hashlib.sha256(ck).hexdigest()
    json.dump({'tag': t, 'files': {'ckpt_last.pt': {'bytes': len(ck), 'sha256': sh}}}, open(os.path.join(d, 'BANK_MANIFEST.json'), 'w'))
    json.dump({'tag': t, 'evidence': {'ckpt_sha256': sh}}, open(os.path.join(d, 'RUN.done'), 'w'))
    json.dump({'levers': {'goal_pos_weight_stop': 1.0, 'lon_class_weight_stop': 1.0}}, open(os.path.join(d, 'selfcheck.json'), 'w'))
    open(os.path.join(d, 'BANK.done'), 'w').write('verified against box 2 by sha256\n')
" "$1"
}
newbox() {   # newbox <name> [gpu MiB]: the dry-run environment for box <name>; the SLOWWM chain has ended (DONE) unless the test says otherwise; the three NULL banks are whole
  B="$T/box_$1"; "$PYV" -W ignore "$BT/dry_helpers.py" mkbox "$B" "$PYV" "${2:-0}" > /dev/null
  mkdir -p "$B/D/tanitad/slowwm/runs" "$B/D/tanitad/repo/stack"; echo "slowwm file" > "$B/D/tanitad/slowwm/runs/R1.txt"; echo "shared repo file" > "$B/D/tanitad/repo/stack/x.py"
  export STOPBAL_DRY=1 STOPBAL_SSH="$SSHS" STOPBAL_SFTP="$SFTPS" FAKE_BOX="$(cygpath -m "$B")" FAKE_PY="$PYV" STOPBAL_PY="$PYV" STOPBAL_STATE="$T/st_$1" STOPBAL_PULLS="$T/pulls_$1"
  export STOPBAL_BUSY_DEV="$T/busy_$1.txt" STOPBAL_ASIDE_DEV="$T/aside_$1" STOPBAL_B2_WIN_ROOT="$(cygpath -w "$B/D/tanitad")" STOPBAL_POLL_S=0 STOPBAL_REPO="$T/repo" STOPBAL_EXPORT_LIST="$T/repo/export_list.txt"
  export STOPBAL_COMMIT="$COMMIT" STOPBAL_NEED_C_MB=1 STOPBAL_POLLS_RUN=5 STOPBAL_POLLS_SHIP=5 STOPBAL_POLLS_CLEAN=5 STOPBAL_WAIT_POLLS=0 STOPBAL_WAIT_POLL_S=0
  export STOPBAL_PRED_STATE="$T/slowwm_$1"; mkdir -p "$STOPBAL_PRED_STATE"; echo "ZZ SLOWWM-DONE 2026-11-11 00:00:00 commit=x" > "$STOPBAL_PRED_STATE/DONE.txt"
  plant_null_banks "$STOPBAL_PULLS"
  unset STOPBAL_RESUME STOPBAL_AMENDMENT STOPBAL_BASE
}
control() { printf '%s' "$1" > "$B/plant_control.json"; }
calls() { cat "$B/transport.log" 2>/dev/null | wc -l | tr -d ' '; }
stoptxt() { cat "$STOPBAL_STATE/STOP.txt" 2>/dev/null; }
treehash() { ( cd "$1" && find . -type f | sort | xargs sha256sum | sha256sum | cut -d' ' -f1 ); }
ALL4="BAL-G2_s0 BAL-L2_s0 BAL-G2_s1 BAL-L2_s1"

echo "== 1 normal flow: SLOWWM ended, ship, pilot, 12 h permit, four runs in the card's order, banks, cleanups, no parity step"
newbox normal 0; control '{"sps": 1.0, "startup_s": 30}'
SH0="$(treehash "$B/D/tanitad/slowwm")"; RH0="$(treehash "$B/D/tanitad/repo")"; NB_REF="$(for t in BAL-NULL_s0 BAL-NULL_s1 BAL-NULL_rep; do treehash "$STOPBAL_PULLS/stopbal_$t"; done | sha256sum)"
bash "$PKG/dose2_chain.sh" > "$T/normal.out" 2>&1; rc=$?
ok "chain exits 0" '[ $rc -eq 0 ]'
ok "DONE sentinel written (ZZ STOPBAL2-DONE), no stop" '[ -e "$STOPBAL_STATE/DONE.txt" ] && [ ! -e "$STOPBAL_STATE/STOP.txt" ] && head -c 22 "$STOPBAL_STATE/DONE.txt" | grep -q "^ZZ STOPBAL2-DONE"'
ok "sentinel text only in its own file (chain.log carries none)" '! grep -q "ZZ STOPBAL" "$STOPBAL_STATE/chain.log"'
ok "the chain log says the SLOWWM chain ended (DONE) and waited for nothing" 'grep -q "the SLOWWM chain ended (DONE)" "$STOPBAL_STATE/chain.log" && ! grep -q "waiting for the SLOWWM chain" "$STOPBAL_STATE/chain.log"'
P="$STOPBAL_STATE/pilot/PILOT.verified.json"
ok "PILOT.verified.json names the commit" 'grep -q "\"commit\": \"$COMMIT\"" "$P"'
ok "pilot mean s/step excludes start-up (reads 1.0)" '"$PYV" -I -c "import json,sys;d=json.load(open(sys.argv[1]));assert abs(d[\"timing\"][\"mean_s_per_step\"]-1.0)<1e-9, d[\"timing\"]" "$P"'
ok "projection covers 4 runs x 3,000 steps = 12,000" '"$PYV" -I -c "import json,sys;d=json.load(open(sys.argv[1]))[\"projection\"];assert d[\"steps_total\"]==12000 and d[\"n_runs_planned\"]==4 and d[\"within_limit_planned\"] and d[\"limit_h\"]==12.0, d" "$P"'
ok "pilot used the exact BAL-G2_s0 argv plus --steps 200 and a separate out dir" '"$PYV" -I -c "import sys,json;sys.path.insert(0,sys.argv[1]);import stopbal_check as C;l=open(sys.argv[2]).read().split();assert l==C.tokens(\"PILOT_BAL-G2_s0\") and l[-2:]==[\"--steps\",\"200\"] and \"pilot_out\" in l[l.index(\"--out\")+1] and \"20.81\" in l and [x for x in l[:-2] if x not in C.tokens(\"BAL-G2_s0\")]==[l[l.index(\"--out\")+1]]" "$PKG" "$STOPBAL_STATE/runs/PILOT_BAL-G2_s0/argv.txt"'
ok "the pilot token binds commit and argv" '"$PYV" -I "$PKG/stopbal_check.py" verify-token --token "$STOPBAL_STATE/pilot/gate_token.json" --argv-file "$STOPBAL_STATE/runs/PILOT_BAL-G2_s0/argv.txt" --tag PILOT_BAL-G2_s0 --commit "$COMMIT" > /dev/null'
ok "the pilot also gated the full-length G2_s0 and L2_s0 argvs (tokens exist before the 12 h decision)" '[ -s "$B/D/tanitad/stopbal/runs/BAL-G2_s0/gate/gate_token.json" ] && [ -s "$B/D/tanitad/stopbal/runs/BAL-L2_s0/gate/gate_token.json" ]'
ok "pilot temp dirs deleted on the box (pilot out, the gate outs)" '[ ! -e "$B/D/tanitad/stopbal/pilot_out/BAL-G2_s0" ] && [ ! -e "$B/D/tanitad/stopbal/gate_out/PILOT_BAL-G2_s0" ] && [ ! -e "$B/D/tanitad/stopbal/gate_out/BAL-L2_s0" ] && [ ! -e "$B/D/tanitad/stopbal/gate_out/BAL-G2_s0" ]'
ok "pilot deletions are logged (path, bytes) and banked" 'grep -q "\"bytes\"" "$STOPBAL_STATE/pilot/CLEANUP.json" && grep -q "pilot_out" "$STOPBAL_STATE/pilot/CLEANUP.json"'
for t in $ALL4; do
  BD="$STOPBAL_PULLS/stopbal_$t"
  ok "$t banked with BANK.done and the six evidence files" '[ -e "$BD/BANK.done" ] && [ -s "$BD/ckpt_last.pt" ] && [ -s "$BD/summary.json" ] && [ -s "$BD/train_log.jsonl" ] && [ -s "$BD/gate_token.json" ] && [ -s "$BD/selfcheck.json" ] && [ -s "$BD/RUN.done" ]'
  ok "$t banked checkpoint equals the box copy (sha256)" '[ "$(sha256sum "$BD/ckpt_last.pt" | cut -d" " -f1)" = "$(sha256sum "$B/D/tanitad/runs/v8f_stopbal/$t/ckpt_last.pt" | cut -d" " -f1)" ]'
  ok "$t token binds the commit and the argv" '"$PYV" -I "$PKG/stopbal_check.py" verify-token --token "$BD/gate_token.json" --argv-file "$BD/argv.txt" --tag $t --commit "$COMMIT" > /dev/null'
  ok "$t RUN.done carries the moved identity (S, T and O moved)" 'grep -q "\"moved_identity\"" "$BD/RUN.done" && grep -q "\"S\"" "$BD/RUN.done" && grep -q "\"O\"" "$BD/RUN.done"'
  ok "$t gate dir on the box deleted, the checkpoint kept" '[ ! -e "$B/D/tanitad/stopbal/gate_out/$t" ] && [ -s "$B/D/tanitad/runs/v8f_stopbal/$t/ckpt_last.pt" ]'
  ok "$t cleanup logged (path, bytes) in the bank" 'grep -q "\"bytes\"" "$BD/CLEANUP.json" || grep -q "deleted_own_temp\": \[\]" "$BD/CLEANUP.json"'
done
ok "the four tokens bind four distinct argvs" '[ "$(cat "$STOPBAL_PULLS"/stopbal_BAL-G2_*/gate_token.json "$STOPBAL_PULLS"/stopbal_BAL-L2_*/gate_token.json | grep -o "\"argv_sha256\": \"[0-9a-f]*\"" | sort -u | wc -l | tr -d " ")" -eq 4 ]'
ok "run order in chain.log is the card's: G2_s0, L2_s0, G2_s1, L2_s1" '[ "$(grep -o "done BAL-[A-Za-z0-9_-]*" "$STOPBAL_STATE/chain.log" | tr "\n" " ")" = "done BAL-G2_s0 done BAL-L2_s0 done BAL-G2_s1 done BAL-L2_s1 " ]'
ok "DONE lists exactly the four planned runs" 'head -1 "$STOPBAL_STATE/DONE.txt" | grep -q "runs=BAL-G2_s0 BAL-L2_s0 BAL-G2_s1 BAL-L2_s1$"'
ok "no parity step ran and no NULL arm was trained (three NULL bank folders, none on the box)" '[ ! -e "$STOPBAL_STATE/NULL_PARITY.json" ] && ! grep -q "parity" "$STOPBAL_STATE/chain.log" && [ ! -e "$B/D/tanitad/runs/v8f_stopbal/BAL-NULL_s0" ] && [ "$(ls "$STOPBAL_PULLS" | grep -c NULL)" -eq 3 ]'
ok "the three NULL banks are byte-identical afterwards (never touched)" '[ "$(for t in BAL-NULL_s0 BAL-NULL_s1 BAL-NULL_rep; do treehash "$STOPBAL_PULLS/stopbal_$t"; done | sha256sum)" = "$NB_REF" ]'
ok "the banked checkpoints carry the arm's own stop weights (G2 20.81 / 1.0, L2 1.0 / 22.08) and no detach" '"$PYV" -I -c "import pickle,sys;g=lambda t:pickle.load(open(sys.argv[1]+\"/stopbal_\"+t+\"/ckpt_last.pt\",\"rb\"))[\"cfg\"];w=lambda t:(g(t)[\"goal_pos_weight_stop\"],g(t)[\"lon_class_weight_stop\"]);assert w(\"BAL-G2_s0\")==(20.81,1.0) and w(\"BAL-G2_s1\")==(20.81,1.0) and w(\"BAL-L2_s0\")==(1.0,22.08) and w(\"BAL-L2_s1\")==(1.0,22.08) and not any(g(t)[\"detach_upper\"] for t in sys.argv[2:])" "$STOPBAL_PULLS" BAL-G2_s0 BAL-L2_s1'
ok "the box saw only stub traffic and no unknown command" '[ "$(calls)" -gt 60 ] && ! grep -q "UNKNOWN COMMAND" "$B/transport.log"'
ok "disk was read before every stage (>= 6 reads: ship, pilot, four runs)" '[ "$(grep -c "Get-PSDrive D" "$B/transport.log")" -ge 6 ]'
ok "isolation: the SLOWWM folder on the box is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/slowwm")" = "$SH0" ]'
ok "isolation: the shared D:\\tanitad\\repo is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/repo")" = "$RH0" ]'
ok "own tree exists with the trainer and TIP_STOPBAL.txt" '[ -s "$B/D/tanitad/stopbal/repo/stack/scripts/v8f_mini_train.py" ] && [ "$(cat "$B/D/tanitad/stopbal/repo/TIP_STOPBAL.txt" | tr -d "\r\n")" = "$COMMIT" ]'
ok "box 2 got the dose-2 checker, the first dose's checker as stopbal_check_base.py, and the dose-2 argv file (each equal to the package's)" '[ "$(sha256sum "$B/D/tanitad/stopbal/check/stopbal_check.py" | cut -d" " -f1)" = "$(sha256sum "$PKG/stopbal_check.py" | cut -d" " -f1)" ] && [ "$(sha256sum "$B/D/tanitad/stopbal/check/stopbal_check_base.py" | cut -d" " -f1)" = "$(sha256sum "$BOX2/stopbal_check.py" | cut -d" " -f1)" ] && [ "$(sha256sum "$B/D/tanitad/stopbal/check/argvs_dose2.json" | cut -d" " -f1)" = "$(sha256sum "$PKG/frozen/argvs_dose2.json" | cut -d" " -f1)" ]'
ok "busy marker handed on" '[ ! -e "$STOPBAL_BUSY_DEV" ]'
ok "no breach through PATH" '[ ! -e "$T/BREACH" ]'
ok "a second chain run skips finished stages" 'bash "$PKG/dose2_chain.sh" 2>&1 | tail -3 | grep -q "all runs banked"'
NORMAL_B="$B"

echo "== 2 the wait-for-SLOWWM gate: waiting (nothing may reach the box) and released"
newbox wait 0; rm -f "$STOPBAL_PRED_STATE/DONE.txt"
bash "$PKG/dose2_chain.sh" > "$T/w1.out" 2>&1; rc=$?
ok "no SLOWWM marker at all: refused with rc 12, zero box calls, the stop line names the wait" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ] && stoptxt | grep -q "predecessor-not-ended" && stoptxt | grep -q "ZZ STOPBAL2-STOP"'
rm -f "$STOPBAL_STATE/STOP.txt"; echo "ZZ OBD-DONE 2026-11-10 x" > "$STOPBAL_PRED_STATE/DONE.txt"
bash "$PKG/dose2_chain.sh" > "$T/w1b.out" 2>&1; rc=$?
ok "another chain's DONE marker (OBD) is not SLOWWM's: keeps waiting (rc 12), zero calls" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$STOPBAL_STATE/STOP.txt"; echo "not a finished marker" > "$STOPBAL_PRED_STATE/DONE.txt"
bash "$PKG/dose2_chain.sh" > "$T/w1c.out" 2>&1; rc=$?
ok "a DONE file without the SLOWWM prefix is not a finished marker (rc 12, zero calls)" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$STOPBAL_STATE/STOP.txt" "$STOPBAL_PRED_STATE/DONE.txt"; echo "ZZ SLOWWM-STOP run x rc=12 :: the box was busy" > "$STOPBAL_PRED_STATE/STOP.txt"
bash "$PKG/dose2_chain.sh" > "$T/w2.out" 2>&1; rc=$?
ok "a SLOWWM STOP whose head ends in rc=12 (busy: it never ran its stage) keeps waiting (rc 12), zero calls" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ] && stoptxt | grep -q "predecessor-not-ended" && stoptxt | grep -q "last state STOP-BUSY"'
rm -f "$STOPBAL_STATE/STOP.txt"; echo "ZZ SLOWWM-STOP run x rc=12" > "$STOPBAL_PRED_STATE/STOP.txt"
bash "$PKG/dose2_chain.sh" > "$T/w2b.out" 2>&1; rc=$?
ok "the same with no detail text: keeps waiting" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$STOPBAL_STATE/STOP.txt"; echo "ZZ SLOWWM-DONE x" > "$STOPBAL_PRED_STATE/DONE.txt"; echo "slowwm chain 1" > "$STOPBAL_BUSY_DEV"
bash "$PKG/dose2_chain.sh" > "$T/w3.out" 2>&1; rc=$?
ok "SLOWWM DONE but the busy marker held by another chain: refused with rc 12, zero box calls" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
ok "the held busy marker is untouched by the refused chain" '[ "$(cat "$STOPBAL_BUSY_DEV")" = "slowwm chain 1" ]'
rm -f "$STOPBAL_STATE/STOP.txt" "$STOPBAL_BUSY_DEV" "$STOPBAL_PRED_STATE/DONE.txt"; echo "ZZ SLOWWM-STOP run x rc=4 :: detail that mentions rc=12 later" > "$STOPBAL_PRED_STATE/STOP.txt"; echo 9000 > "$B/gpu_mib.txt"
bash "$PKG/dose2_chain.sh" > "$T/w4.out" 2>&1; rc=$?
ok "a SLOWWM STOP with any other head (rc=4; an rc=12 only in the detail) RELEASES the chain: it reached the box and the busy GPU (9000 MiB) refused the pilot (rc 12)" '[ $rc -eq 12 ] && [ "$(calls)" -gt 0 ] && stoptxt | grep -q "gpu-not-idle" && grep -q "the SLOWWM chain ended (STOP)" "$STOPBAL_STATE/chain.log"'
ok "nothing was trained after the GPU refusal" '[ ! -e "$B/D/tanitad/stopbal/runs/PILOT_BAL-G2_s0/gate" ] && [ ! -e "$B/D/tanitad/runs/v8f_stopbal" ]'
newbox rel 0; control '{"sps": 1.0}'; rm -f "$STOPBAL_PRED_STATE/DONE.txt"
( sleep 4; echo "ZZ SLOWWM-DONE 2026-11-11 x" > "$STOPBAL_PRED_STATE/DONE.txt" ) &
STOPBAL_WAIT_POLLS=40 STOPBAL_WAIT_POLL_S=1 bash "$PKG/dose2_chain.sh" > "$T/rel.out" 2>&1; rc=$?
wait
ok "waiting, then released: the chain polled while SLOWWM ran (no box call during the wait), then the DONE marker released it and all four runs banked (rc 0)" '[ $rc -eq 0 ] && grep -q "waiting for the SLOWWM chain to end (poll 1, state RUNNING)" "$STOPBAL_STATE/chain.log" && grep -q "the SLOWWM chain ended (DONE)" "$STOPBAL_STATE/chain.log" && [ -e "$STOPBAL_STATE/DONE.txt" ] && [ "$(grep -n "waiting for" "$STOPBAL_STATE/chain.log" | tail -1 | cut -d: -f1)" -lt "$(grep -n "start ship" "$STOPBAL_STATE/chain.log" | cut -d: -f1)" ]'
ok "no breach through PATH after the wait tests" '[ ! -e "$T/BREACH" ]'

echo "== 3 the state-dir collision guard (refused with rc 2 before anything is created or any call is made)"
newbox coll 0
for other in "C:/Users/Admin/stopbal_state" "C:/Users/Admin/slowwm_state" "$STOPBAL_PRED_STATE"; do
  : > "$B/transport.log"
  STOPBAL_STATE="$other" bash "$PKG/dose2_chain.sh" > "$T/coll.out" 2>&1; rc=$?
  ok "STOPBAL_STATE=$other collides: refused with rc 2 and the message names the collision, zero calls" '[ $rc -eq 2 ] && grep -q "collides with" "$T/coll.out" && [ "$(calls)" -eq 0 ]'
done
ok "the guard created nothing in the colliding predecessor dir (no done/, no chain.log, no STOP)" '[ ! -e "$STOPBAL_PRED_STATE/done" ] && [ ! -e "$STOPBAL_PRED_STATE/chain.log" ] && [ ! -e "$STOPBAL_PRED_STATE/STOP.txt" ]'
ok "the default state dir of this chain is none of those three" '[ "C:/Users/Admin/stopbal_dose2_state" != "C:/Users/Admin/stopbal_state" ] && grep -q "stopbal_dose2_state" "$PKG/dose2_chain.sh"'

echo "== 4 the box-side pin refuses a changed token (checker run on the fake box, no transport)"
B="$NORMAL_B"; RD="$B/D/tanitad/stopbal/runs/BAL-L2_s0"; cp "$RD/argv.txt" "$T/argv.keep"
sed -i 's/--seed 0/--seed 7/' "$RD/argv.txt"
( export STOPBAL_ROOT="$(cygpath -m "$B/D/tanitad")" STOPBAL_COMMIT="$COMMIT" STOPBAL_FAKE_INIT_SHA_FILE="$(cygpath -m "$B/D/tanitad/init_sha.json")" STOPBAL_FAKE_BANK_SHA_FILE="$(cygpath -m "$B/D/tanitad/bank_sha.json")" STOPBAL_FLOOR_BYTES=1000 STOPBAL_FAKE_V9_MD5="$(md5sum "$B/D/tanitad/data/v9/v9_labels_train.npz" | cut -d' ' -f1)"
  "$PYV" -I "$B/D/tanitad/stopbal/check/stopbal_check.py" stage --tag BAL-L2_s0 --gid pin1 > "$T/pin.out" 2>&1 ); rc=$?
ok "RED wrong argv: a changed seed token is refused (rc 3) and the first differing token is named" '[ $rc -eq 3 ] && grep -q "pin-fail" "$RD/STAGE_pin1.fail" && grep -q "token " "$RD/STAGE_pin1.fail" && grep -q "frozen argv has" "$RD/STAGE_pin1.fail"'
cp "$T/argv.keep" "$RD/argv.txt"
sed -i 's/--lon-class-weight-stop 22.08/--lon-class-weight-stop 4.6991/' "$RD/argv.txt"
( export STOPBAL_ROOT="$(cygpath -m "$B/D/tanitad")" STOPBAL_COMMIT="$COMMIT" STOPBAL_FAKE_INIT_SHA_FILE="$(cygpath -m "$B/D/tanitad/init_sha.json")" STOPBAL_FAKE_BANK_SHA_FILE="$(cygpath -m "$B/D/tanitad/bank_sha.json")" STOPBAL_FLOOR_BYTES=1000 STOPBAL_FAKE_V9_MD5="$(md5sum "$B/D/tanitad/data/v9/v9_labels_train.npz" | cut -d' ' -f1)"
  "$PYV" -I "$B/D/tanitad/stopbal/check/stopbal_check.py" stage --tag BAL-L2_s0 --gid pin2 > "$T/pin2.out" 2>&1 ); rc=$?
ok "RED wrong weight: the first dose's 4.6991 in the L2 argv is refused at the box-side pin (rc 3), the token named, no training" '[ $rc -eq 3 ] && grep -q "pin-fail" "$RD/STAGE_pin2.fail" && grep -q "4.6991" "$RD/STAGE_pin2.fail" && grep -q "22.08" "$RD/STAGE_pin2.fail"'
cp "$T/argv.keep" "$RD/argv.txt"
cp "$B/D/tanitad/stopbal/check/argvs_dose2.json" "$T/argvs.keep"
sed -i 's/20.81/20.82/' "$B/D/tanitad/stopbal/check/argvs_dose2.json"
( export STOPBAL_ROOT="$(cygpath -m "$B/D/tanitad")" STOPBAL_COMMIT="$COMMIT" STOPBAL_FAKE_INIT_SHA_FILE="$(cygpath -m "$B/D/tanitad/init_sha.json")" STOPBAL_FAKE_BANK_SHA_FILE="$(cygpath -m "$B/D/tanitad/bank_sha.json")" STOPBAL_FLOOR_BYTES=1000 STOPBAL_FAKE_V9_MD5="$(md5sum "$B/D/tanitad/data/v9/v9_labels_train.npz" | cut -d' ' -f1)"
  "$PYV" -I "$B/D/tanitad/stopbal/check/stopbal_check.py" argv --tag BAL-G2_s0 > "$T/pin3.out" 2>&1 ); rc=$?
ok "RED a changed frozen argv file on the box is refused by its sha256 pin (the checker exits non-zero and says REFUSED)" '[ $rc -ne 0 ] && grep -q "REFUSED" "$T/pin3.out"'
cp "$T/argvs.keep" "$B/D/tanitad/stopbal/check/argvs_dose2.json"
echo "changed bank" > "$B/D/tanitad/runs/v8f_mini_full/dagger_bank/s0/bank.json"
( export STOPBAL_ROOT="$(cygpath -m "$B/D/tanitad")" STOPBAL_COMMIT="$COMMIT" STOPBAL_FAKE_INIT_SHA_FILE="$(cygpath -m "$B/D/tanitad/init_sha.json")" STOPBAL_FAKE_BANK_SHA_FILE="$(cygpath -m "$B/D/tanitad/bank_sha.json")" STOPBAL_FLOOR_BYTES=1000 STOPBAL_FAKE_V9_MD5="$(md5sum "$B/D/tanitad/data/v9/v9_labels_train.npz" | cut -d' ' -f1)"
  "$PYV" -I "$B/D/tanitad/stopbal/check/stopbal_check.py" stage --tag BAL-L2_s0 --gid bank1 > "$T/bank.out" 2>&1 ); rc=$?
ok "a DAgger bank whose sha256 differs is refused at the box-side preflight (rc 3) before any gate or training" '[ $rc -eq 3 ] && grep -q "preflight-fail" "$RD/STAGE_bank1.fail" && grep -q "DAgger bank s0" "$RD/STAGE_bank1.fail"'

echo "== 5 red arms through the chain: wrong weight, wrong commit, a missing banked NULL arm"
newbox glever 0; control '{"sps": 1.0, "modes": {"PILOT_BAL-G2_s0": "lever_not_built"}}'
bash "$PKG/dose2_chain.sh" > "$T/g0.out" 2>&1; rc=$?
ok "RED wrong weight: a trainer whose BUILT loss reads 1.0 for G2's lever is refused at the pilot (rc 4); the stop line names the lever, the value read and 20.81" '[ $rc -eq 4 ] && stoptxt | grep -q "BUILT lever goal_pos_weight_stop = 1.0" && stoptxt | grep -q "20.81"'
ok "nothing was trained after it, no checkpoint exists, no bank was written" '[ ! -e "$B/D/tanitad/stopbal/runs/PILOT_BAL-G2_s0/logs/train.log" ] && [ ! -e "$B/D/tanitad/runs/v8f_stopbal" ] && [ ! -e "$STOPBAL_PULLS/stopbal_BAL-G2_s0" ]'
newbox llever 0; control '{"sps": 1.0, "modes": {"BAL-L2_s0": "lever_not_built"}}'
bash "$PKG/dose2_chain.sh" > "$T/g0b.out" 2>&1; rc=$?
ok "RED wrong weight on the other arm: BAL-L2_s0 (gated at the pilot) reading 1.0 for the lon lever is refused (rc 4) and names 22.08" '[ $rc -eq 4 ] && stoptxt | grep -q "lon_class_weight_stop" && stoptxt | grep -q "22.08"'
newbox pweight 0; control '{"sps": 1.0, "train_modes": {"BAL-L2_s0": "cfg_weight_lost"}}'
bash "$PKG/dose2_chain.sh" > "$T/g0c.out" 2>&1; rc=$?
ok "RED wrong weight after training: a BAL-L2_s0 checkpoint whose config lost the declared weight is a post-check failure (rc 5), the arm is not banked, BAL-G2_s0 is" '[ $rc -eq 5 ] && stoptxt | grep -q "cfg.lon_class_weight_stop" && [ ! -e "$STOPBAL_PULLS/stopbal_BAL-L2_s0/BANK.done" ] && [ -e "$STOPBAL_PULLS/stopbal_BAL-G2_s0/BANK.done" ]'
newbox garm 0; control '{"sps": 1.0, "modes": {"BAL-L2_s1": "lever_not_built"}}'
bash "$PKG/dose2_chain.sh" > "$T/g4.out" 2>&1; rc=$?
ok "an arm-specific gate refusal (BAL-L2_s1) ends the chain with rc 4 but the other three ran and banked; the sentinel is in its own file" '[ $rc -eq 4 ] && [ -e "$STOPBAL_STATE/ARM_REFUSED_BAL-L2_s1.txt" ] && [ ! -e "$STOPBAL_STATE/DONE.txt" ] && [ -e "$STOPBAL_PULLS/stopbal_BAL-G2_s0/BANK.done" ] && [ -e "$STOPBAL_PULLS/stopbal_BAL-L2_s0/BANK.done" ] && [ -e "$STOPBAL_PULLS/stopbal_BAL-G2_s1/BANK.done" ] && [ ! -e "$B/D/tanitad/runs/v8f_stopbal/BAL-L2_s1" ] && head -c 30 "$STOPBAL_STATE/ARM_REFUSED_BAL-L2_s1.txt" | grep -q "^ZZ STOPBAL2-ARM-REFUSED" && ! grep -q "ZZ STOPBAL" "$STOPBAL_STATE/chain.log"'
# wrong commit: a repo whose trainer lacks the needles (the parent of the code of record in miniature)
cp -r "$T/repo" "$T/repo_old"
printf '# an old trainer without the stop-weight flags\n' > "$T/repo_old/stack/scripts/v8f_mini_train.py"
git -c safe.directory='*' -C "$T/repo_old" add -A > /dev/null 2>&1
git -c safe.directory='*' -c user.name=dry -c user.email=dry@example.invalid -C "$T/repo_old" commit -q -m old > /dev/null 2>&1
OLD="$(git -c safe.directory='*' -C "$T/repo_old" rev-parse HEAD)"
newbox wcommit 0; control '{"sps": 1.0}'
STOPBAL_REPO="$T/repo_old" STOPBAL_COMMIT="$OLD" bash "$PKG/dose2_chain.sh" > "$T/wc.out" 2>&1; rc=$?
ok "RED wrong commit: a commit whose trainer lacks the stop-weight flags is refused at the ship stage (stage rc 16, chain rc 8), the stop line names the missing needles" '[ $rc -eq 8 ] && stoptxt | grep -q "package-commit-or-transport" && stoptxt | grep -q "STOPBAL-SHIP-REFUSED" && stoptxt | grep -q "goal-pos-weight-stop"'
ok "no trainer reached the box, nothing ran, no bank" '[ ! -e "$B/D/tanitad/stopbal/repo/stack/scripts/v8f_mini_train.py" ] && [ ! -e "$B/D/tanitad/runs/v8f_stopbal" ] && [ ! -e "$STOPBAL_PULLS/stopbal_BAL-G2_s0" ]'
newbox wcommit2 0
STOPBAL_COMMIT="$(printf 'a%.0s' $(seq 1 40))" bash "$PKG/dose2_chain.sh" > "$T/wc2.out" 2>&1; rc=$?
ok "RED wrong commit: a well-formed sha that is not in the repository is refused at the ship stage (chain rc 8), nothing ran" '[ $rc -eq 8 ] && [ ! -e "$B/D/tanitad/runs/v8f_stopbal" ]'
for bad in "" "abc123" "$(printf 'A%.0s' $(seq 1 40))"; do
  newbox wcommit3 0; rm -f "$STOPBAL_STATE/STOP.txt"; : > "$B/transport.log"
  STOPBAL_COMMIT="$bad" bash "$PKG/dose2_chain.sh" > "$T/wc3.out" 2>&1; rc=$?
  ok "a missing / abbreviated / upper-case commit ('${bad:0:8}') refuses (rc 8), zero box calls" '[ $rc -eq 8 ] && stoptxt | grep -q "commit-missing" && [ "$(calls)" -eq 0 ]'
done
if [ -d "C:/Users/Admin/tanitad-push" ]; then
  ok "RED wrong commit, real mirror: the PARENT 5514a684 (no stop-weight flags) is refused by the ship tree builder the dose-2 ship stage uses" '! "$PYV" -I "$BOX2/stopbal_ship_tree.py" build --repo "C:/Users/Admin/tanitad-push" --commit 5514a6840b57ad115dcadeb7e264b10f5c1d0081 --export-list "$T/repo/export_list.txt" --out-dir "$T/realship" > "$T/realship.out" 2>&1 && grep -q "ZZ STOPBAL-SHIP-REFUSED" "$T/realship.out" && [ ! -e "$T/realship/stopbal_tree.tar" ]'
  ok "the code of record 715ef8fc is ACCEPTED by the same builder and its manifest names it" '"$PYV" -I "$BOX2/stopbal_ship_tree.py" build --repo "C:/Users/Admin/tanitad-push" --commit 715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf --export-list "$T/repo/export_list.txt" --out-dir "$T/realship_ok" > "$T/realship_ok.out" 2>&1 && grep -q "\"commit\": \"715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf\"" "$T/realship_ok/stopbal_tree_manifest.json"'
fi
# a missing banked NULL arm, three ways; each refuses with rc 8 BEFORE any box call
for how in folder ckpt_changed lever bankdone manifest; do
  newbox "nb_$how" 0; : > "$B/transport.log"; rm -f "$STOPBAL_STATE/STOP.txt"
  case "$how" in
    folder) rm -rf "${STOPBAL_PULLS:?}/stopbal_BAL-NULL_rep";;
    ckpt_changed) printf 'tampered' >> "$STOPBAL_PULLS/stopbal_BAL-NULL_s1/ckpt_last.pt";;
    lever) sed -i 's/"goal_pos_weight_stop": 1.0/"goal_pos_weight_stop": 4.5616/' "$STOPBAL_PULLS/stopbal_BAL-NULL_s0/selfcheck.json";;
    bankdone) rm -f "$STOPBAL_PULLS/stopbal_BAL-NULL_s0/BANK.done";;
    manifest) rm -f "$STOPBAL_PULLS/stopbal_BAL-NULL_s1/BANK_MANIFEST.json";;
  esac
  bash "$PKG/dose2_chain.sh" > "$T/nb_$how.out" 2>&1; rc=$?
  ok "RED missing banked NULL arm ($how): the chain refuses with rc 8 naming the bank, zero box calls, nothing trained" '[ $rc -eq 8 ] && stoptxt | grep -q "null-bank-missing-or-changed" && stoptxt | grep -q "BAL-NULL" && [ "$(calls)" -eq 0 ] && [ ! -e "$B/D/tanitad/stopbal/ship" ]'
done
ok "no breach through PATH after the red arms" '[ ! -e "$T/BREACH" ]'

echo "== 6 12 h rule, disk gate"
newbox slow 0; control '{"sps": 3.7}'
bash "$PKG/dose2_chain.sh" > "$T/s1.out" 2>&1; rc=$?
ok "pilot at 3.7 s/step (12.3 h for four runs): the chain stops at the 12 h rule (rc 3), no run started" '[ $rc -eq 3 ] && stoptxt | grep -q "12h-rule-refused" && [ ! -e "$B/D/tanitad/runs/v8f_stopbal" ]'
ok "3.5 s/step (11.7 h) would pass and 3.7 would not: the boundary is 12 h" '"$PYV" -I -c "import sys;sys.path.insert(0,sys.argv[1]);import stopbal_check as C;assert C.project(3.5)[\"within_limit_planned\"] and not C.project(3.7)[\"within_limit_planned\"] and C.project(0.8)[\"projected_total_h_planned\"]<3.0" "$PKG"'
newbox disk 0; control '{"sps": 1.0}'; echo 3 > "$B/free_gb.txt"
bash "$PKG/dose2_chain.sh" > "$T/d1.out" 2>&1; rc=$?
ok "3 GB free: ship passes (1 GB), the pilot stops with the disk-low code (rc 6)" '[ $rc -eq 6 ] && stoptxt | grep -q "disk-low" && [ ! -e "$B/D/tanitad/stopbal/runs/PILOT_BAL-G2_s0/gate" ]'

echo "== 7 transport fail-closed"
newbox tr 0
for how in unset relative absent; do
  rm -f "$STOPBAL_STATE/STOP.txt"; : > "$B/transport.log"
  case "$how" in
    unset) ( unset STOPBAL_SSH; bash "$PKG/dose2_chain.sh" > "$T/tr_$how.out" 2>&1 ); rc=$?;;
    relative) STOPBAL_SSH="ssh" bash "$PKG/dose2_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
    absent) STOPBAL_SSH="$T/no_such_ssh" bash "$PKG/dose2_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
  esac
  ok "transport '$how' refuses with rc 16, no call, a stop line" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ] && grep -q "transport-not-fail-closed" "$STOPBAL_STATE/STOP.txt"'
done
rm -f "$STOPBAL_STATE/STOP.txt"
( unset STOPBAL_DRY; bash "$PKG/dose2_chain.sh" > "$T/tr_nodry.out" 2>&1 ); rc=$?
ok "a stub without test mode refuses (rc 16, zero calls)" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$STOPBAL_STATE/STOP.txt"
bash "$PKG/stages/dose2_run.sh" BAL-L_s0 > "$T/tag1.out" 2>&1; rc=$?
ok "a first-dose tag (BAL-L_s0) is not a tag here (rc 16), zero box calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
bash "$PKG/stages/dose2_run.sh" BAL-NULL_s0 > "$T/tag2.out" 2>&1; rc=$?
ok "a NULL tag is not a tag here: no NULL arm is retrained (rc 16), zero box calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'

echo "== 8 hygiene"
ok "no ssh, sftp or scp command in this script other than the poison stubs" '! grep -nE "^[^#]*\b(ssh|sftp|scp) +-" "$TESTS/test_dose2_dryrun.sh" | grep -v "poison" | grep -q .'
ok "bash scripts of the package are LF" '"$PYV" -I -c "import glob,sys;fs=glob.glob(sys.argv[1]+\"/*.sh\")+glob.glob(sys.argv[1]+\"/stages/*.sh\")+glob.glob(sys.argv[1]+\"/tests/*.sh\");assert len(fs)>=5 and all(b\"\r\" not in open(f,\"rb\").read() for f in fs)" "$PKG"'
rm -rf "${T:?}"     # its own scratch dir, checked against the pattern at the top; the dry-run box folders are the only thing under it
echo "DRYRUN2 ok=$OK failed=$BAD"
[ "$BAD" -eq 0 ]
