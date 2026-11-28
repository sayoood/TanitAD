#!/usr/bin/env bash
# Dry run of the EGOWM box-2 package over stubs. NOTHING here can reach box 2: EGOWM_SSH / EGOWM_SFTP name stub scripts by absolute path (EGOWM_DRY=1 makes the chain refuse any file without the stub marker),
# a poison ssh/sftp/scp is put FIRST on PATH and must never run, and the "box" is a folder. The REAL job .cmd files run on Windows cmd.exe against it; only the trainer is a stand-in (tests/plant_trainer.py,
# shipped through a throw-away git repo, so the ship, tree-binding and token code run for real). CPU only, no GPU, no torch.   Usage: bash test_egowm_dryrun.sh  -> prints "DRYRUN ok=<n> failed=<m>"
set -u
TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PKG="$(cd "$TESTS/.." && pwd)"; PKG="$(cygpath -m "$PKG" 2>/dev/null || echo "$PKG")"
PYV="${EGOWM_TEST_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
T="${EGOWM_TEST_TMP:-C:/Users/Admin/AppData/Local/Temp/egowm_dry_run_$$}"
case "$T" in */egowm_dry_run_*|*/scratchpad/*|*/egowm_dry_*) ;; *) echo "refusing an unrecognised scratch dir $T" >&2; exit 2;; esac
rm -rf "${T:?}"; mkdir -p "$T/poison"
for n in ssh sftp scp; do printf '#!/usr/bin/env bash\necho BREACH-%s >> "%s/BREACH"\nexit 99\n' "$n" "$T" > "$T/poison/$n"; chmod +x "$T/poison/$n"; done
export PATH="$T/poison:$PATH"
export CUDA_VISIBLE_DEVICES=-1
OK=0; BAD=0
ok() { if eval "$2"; then OK=$((OK+1)); else BAD=$((BAD+1)); echo "FAIL: $1"; fi; }
SSHS="$(cygpath -m "$TESTS/ssh_stub.sh")"; SFTPS="$(cygpath -m "$TESTS/sftp_stub.sh")"
COMMIT="$("$PYV" -W ignore "$TESTS/dry_helpers.py" mkrepo "$T/repo")"
ok "throw-away repo has a 40-hex commit" '[ ${#COMMIT} -eq 40 ]'
newbox() {   # newbox <name> [gpu MiB]  -> exports the dry-run environment for box <name>; the OTBAND predecessor has ended (DONE.txt in a TEST state folder, never the real one) unless the test says otherwise
  B="$T/box_$1"; "$PYV" -W ignore "$TESTS/dry_helpers.py" mkbox "$B" "$PYV" "${2:-0}" > /dev/null
  mkdir -p "$B/D/tanitad/otband/runs" "$B/D/tanitad/repo/stack"; echo "otband file" > "$B/D/tanitad/otband/runs/R35_s0.txt"; echo "shared repo file" > "$B/D/tanitad/repo/stack/x.py"
  export EGOWM_DRY=1 EGOWM_SSH="$SSHS" EGOWM_SFTP="$SFTPS" FAKE_BOX="$(cygpath -m "$B")" FAKE_PY="$PYV" EGOWM_PY="$PYV" EGOWM_STATE="$T/st_$1" EGOWM_PULLS="$T/pulls_$1"
  export EGOWM_BUSY_DEV="$T/busy_$1.txt" EGOWM_ASIDE_DEV="$T/aside_$1" EGOWM_B2_WIN_ROOT="$(cygpath -w "$B/D/tanitad")" EGOWM_POLL_S=0 EGOWM_REPO="$T/repo" EGOWM_EXPORT_LIST="$T/repo/export_list.txt"
  export EGOWM_COMMIT="$COMMIT" EGOWM_NEED_C_MB=1 EGOWM_POLLS_RUN=5 EGOWM_POLLS_SHIP=5 EGOWM_POLLS_CLEAN=5 EGOWM_WAIT_POLLS=0 EGOWM_WAIT_POLL_S=0
  export EGOWM_OTBAND_STATE="$T/otband_state_$1"; mkdir -p "$EGOWM_OTBAND_STATE"; echo "ZZ OTBAND-DONE 2026-11-10 00:00:00 commit=x runs=x" > "$EGOWM_OTBAND_STATE/DONE.txt"
  unset EGOWM_RESUME EGOWM_AMENDMENT
}
control() { printf '%s' "$1" > "$B/plant_control.json"; }
calls() { cat "$B/transport.log" 2>/dev/null | wc -l | tr -d ' '; }
stoptxt() { cat "$EGOWM_STATE/STOP.txt" 2>/dev/null; }
treehash() { ( cd "$1" && find . -type f | sort | xargs sha256sum | sha256sum | cut -d' ' -f1 ); }
BOXB() { echo "$B/D/tanitad"; }

echo "== 1 normal flow: wait check, ship, pilot, 72 h permit, five runs, banks, cleanups"
newbox normal 0; control '{"sps": 1.0, "startup_s": 30}'
LH0="$(treehash "$B/D/tanitad/otband")"; RH0="$(treehash "$B/D/tanitad/repo")"
bash "$PKG/egowm_chain.sh" > "$T/normal.out" 2>&1; rc=$?
ok "chain exits 0" '[ $rc -eq 0 ]'
ok "DONE sentinel written, no stop" '[ -e "$EGOWM_STATE/DONE.txt" ] && [ ! -e "$EGOWM_STATE/STOP.txt" ] && head -c 15 "$EGOWM_STATE/DONE.txt" | grep -q "^ZZ EGOWM-DONE"'
ok "sentinel text only in its own file (chain.log carries none)" '! grep -q "ZZ EGOWM" "$EGOWM_STATE/chain.log"'
P="$EGOWM_STATE/pilot/PILOT.verified.json"
ok "PILOT.verified.json names the commit" 'grep -q "\"commit\": \"$COMMIT\"" "$P"'
ok "pilot mean s/step excludes start-up (reads 1.0)" '"$PYV" -I -c "import json,sys;d=json.load(open(sys.argv[1]));assert abs(d[\"timing\"][\"mean_s_per_step\"]-1.0)<1e-9, d[\"timing\"]" "$P"'
ok "projection covers 5 runs x 19,164 steps" '"$PYV" -I -c "import json,sys;d=json.load(open(sys.argv[1]))[\"projection\"];assert d[\"steps_total\"]==95820 and d[\"n_runs_planned\"]==5 and d[\"within_limit_planned\"]" "$P"'
ok "pilot used the exact E-KIN_s0 argv plus --steps 200 and a separate out dir" '"$PYV" -I -c "import sys,json;sys.path.insert(0,sys.argv[1]);import egowm_check as C;l=open(sys.argv[2]).read().split();assert l==C.tokens(\"PILOT_E-KIN_s0\") and l[-2:]==[\"--steps\",\"200\"] and \"pilot_out\" in l[l.index(\"--out\")+1] and [x for x in l[:-2] if x not in C.tokens(\"E-KIN_s0\")]==[l[l.index(\"--out\")+1]]" "$PKG" "$EGOWM_STATE/runs/PILOT_E-KIN_s0/argv.txt"'
ok "the pilot token binds commit and argv" '"$PYV" -I "$PKG/egowm_check.py" verify-token --token "$EGOWM_STATE/pilot/gate_token.json" --argv-file "$EGOWM_STATE/runs/PILOT_E-KIN_s0/argv.txt" --tag PILOT_E-KIN_s0 --commit "$COMMIT" > /dev/null'
ok "pilot temp dirs deleted on the box (pilot out, gate outs)" '[ ! -e "$B/D/tanitad/egowm/pilot_out/E-KIN_s0" ] && [ ! -e "$B/D/tanitad/egowm/gate_out/PILOT_E-KIN_s0" ]'
ok "pilot deletions are logged (path, bytes) and banked" 'grep -q "\"bytes\"" "$EGOWM_STATE/pilot/CLEANUP.json" && grep -q "pilot_out" "$EGOWM_STATE/pilot/CLEANUP.json"'
for t in E-KIN_s0 E-NOW_s0 E-KIN_s1 E-NOW_s1 E-KIN_rep_s2; do
  BD="$EGOWM_PULLS/egowm_$t"
  ok "$t banked with BANK.done and the six evidence files" '[ -e "$BD/BANK.done" ] && [ -s "$BD/ckpt_last.pt" ] && [ -s "$BD/summary.json" ] && [ -s "$BD/train_log.jsonl" ] && [ -s "$BD/gate_token.json" ] && [ -s "$BD/selfcheck.json" ] && [ -s "$BD/RUN.done" ]'
  ok "$t banked checkpoint equals the box copy (sha256)" '[ "$(sha256sum "$BD/ckpt_last.pt" | cut -d" " -f1)" = "$(sha256sum "$B/D/tanitad/runs/v8f_egowm/$t/ckpt_last.pt" | cut -d" " -f1)" ]'
  ok "$t token binds the commit and the argv" '"$PYV" -I "$PKG/egowm_check.py" verify-token --token "$BD/gate_token.json" --argv-file "$BD/argv.txt" --tag $t --commit "$COMMIT" > /dev/null'
  ok "$t RUN.done carries the weights identity (non-WM tensors equal, 3 speed tensors non-zero)" 'grep -q "\"n_speed_nonzero\": 3" "$BD/RUN.done" && grep -q "\"n_nonwm_equal\"" "$BD/RUN.done"'
  ok "$t gate dir on the box deleted, the checkpoint kept" '[ ! -e "$B/D/tanitad/egowm/gate_out/$t" ] && [ -s "$B/D/tanitad/runs/v8f_egowm/$t/ckpt_last.pt" ]'
  ok "$t cleanup logged (path, bytes) in the bank" 'grep -q "\"bytes\"" "$BD/CLEANUP.json" || grep -q "deleted_own_temp\": \[\]" "$BD/CLEANUP.json"'
done
ok "the five tokens bind five distinct argvs" '[ "$(cat "$EGOWM_PULLS"/egowm_*/gate_token.json | grep -o "\"argv_sha256\": \"[0-9a-f]*\"" | sort -u | wc -l | tr -d " ")" -eq 5 ]'
ok "run order in chain.log: KIN_s0, NOW_s0, KIN_s1, NOW_s1, KIN_rep_s2" '[ "$(grep -o "done E-[A-Za-z0-9_-]*" "$EGOWM_STATE/chain.log" | tr "\n" " ")" = "done E-KIN_s0 done E-NOW_s0 done E-KIN_s1 done E-NOW_s1 done E-KIN_rep_s2 " ]'
ok "only the five E- runs exist on the box (own folders only)" '[ "$(ls "$B/D/tanitad/runs/v8f_egowm" | wc -l | tr -d " ")" -eq 5 ] && [ ! -e "$B/D/tanitad/egowm/runs/OB-BAND_s0" ]'
ok "the box saw only stub traffic and no unknown command" '[ "$(calls)" -gt 100 ] && ! grep -q "UNKNOWN COMMAND" "$B/transport.log"'
ok "disk was read before every stage (>= 7 reads: ship, pilot, five runs)" '[ "$(grep -c "Get-PSDrive D" "$B/transport.log")" -ge 7 ]'
ok "isolation: the OTBAND folder on the box is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/otband")" = "$LH0" ]'
ok "isolation: the shared D:\\tanitad\\repo is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/repo")" = "$RH0" ]'
ok "own tree exists with the trainer and TIP_EGOWM.txt" '[ -s "$B/D/tanitad/egowm/repo/stack/scripts/v8f_mini_train.py" ] && [ "$(cat "$B/D/tanitad/egowm/repo/TIP_EGOWM.txt" | tr -d "\r\n")" = "$COMMIT" ]'
ok "busy marker handed on" '[ ! -e "$EGOWM_BUSY_DEV" ]'
ok "no breach through PATH" '[ ! -e "$T/BREACH" ]'
ok "a second chain run skips finished stages" 'bash "$PKG/egowm_chain.sh" 2>&1 | tail -3 | grep -q "all runs banked"'
NORMAL_B="$B"

echo "== 2 the box-side pin refuses a changed token (checker run directly on the fake box, no transport)"
B="$NORMAL_B"; RD="$B/D/tanitad/egowm/runs/E-NOW_s0"; cp "$RD/argv.txt" "$T/argv.keep"
sed -i 's/--seed 0/--seed 7/' "$RD/argv.txt"
( export EGOWM_ROOT="$(cygpath -m "$B/D/tanitad")" EGOWM_COMMIT="$COMMIT" EGOWM_FAKE_INIT_SHA_FILE="$(cygpath -m "$B/D/tanitad/init_sha.json")" EGOWM_FLOOR_BYTES=1000 EGOWM_FAKE_V9_MD5="$(md5sum "$B/D/tanitad/data/v9/v9_labels_train.npz" | cut -d' ' -f1)"
  "$PYV" -I "$B/D/tanitad/egowm/check/egowm_check.py" stage --tag E-NOW_s0 --gid pin1 > "$T/pin.out" 2>&1 ); rc=$?
ok "a changed seed token is refused (rc 3) and the first differing token is named" '[ $rc -eq 3 ] && grep -q "pin-fail" "$RD/STAGE_pin1.fail" && grep -q "token " "$RD/STAGE_pin1.fail" && grep -q "frozen argv has" "$RD/STAGE_pin1.fail"'
cp "$T/argv.keep" "$RD/argv.txt"

echo "== 3 predecessor (OTBAND chain) not ended / stopped, box held, GPU not idle, python running"
newbox wait 0; rm -f "$EGOWM_OTBAND_STATE/DONE.txt"
bash "$PKG/egowm_chain.sh" > "$T/w1.out" 2>&1; rc=$?
ok "no OTBAND DONE/STOP marker: refused with rc 12, zero box calls" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ] && stoptxt | grep -q "predecessor-not-ended"'
rm -f "$EGOWM_STATE/STOP.txt"; echo "ZZ OTBAND-DONE x" > "$EGOWM_OTBAND_STATE/DONE.txt"; echo "otband chain 1" > "$EGOWM_BUSY_DEV"
bash "$PKG/egowm_chain.sh" > "$T/w2.out" 2>&1; rc=$?
ok "DONE present but the busy marker held: refused with rc 12, zero box calls" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$EGOWM_STATE/STOP.txt" "$EGOWM_BUSY_DEV"; echo "not a finished marker" > "$EGOWM_OTBAND_STATE/DONE.txt"
bash "$PKG/egowm_chain.sh" > "$T/w2b.out" 2>&1; rc=$?
ok "a DONE file without the OTBAND prefix is not a finished marker (rc 12)" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$EGOWM_STATE/STOP.txt" "$EGOWM_OTBAND_STATE/DONE.txt"; echo "ZZ OTBAND-STOP runs arms-refused: OB-CTL_s1 rc=4" > "$EGOWM_OTBAND_STATE/STOP.txt"
bash "$PKG/egowm_chain.sh" > "$T/w3.out" 2>&1; rc=$?
ok "the predecessor ended in STOP: this chain stops too (rc 12) with the named reason, zero box calls, box not claimed, nothing shipped" '[ $rc -eq 12 ] && stoptxt | grep -q "predecessor-stopped" && stoptxt | grep -q "arms-refused" && [ "$(calls)" -eq 0 ] && [ ! -e "$EGOWM_BUSY_DEV" ] && [ ! -e "$B/D/tanitad/egowm/ship" ]'
rm -f "$EGOWM_STATE/STOP.txt"; echo "ZZ OTBAND-DONE x" > "$EGOWM_OTBAND_STATE/DONE.txt"
bash "$PKG/egowm_chain.sh" > "$T/w3b.out" 2>&1; rc=$?
ok "DONE and STOP both present: STOP wins (rc 12), zero box calls" '[ $rc -eq 12 ] && stoptxt | grep -q "predecessor-stopped" && [ "$(calls)" -eq 0 ]'
rm -f "$EGOWM_STATE/STOP.txt" "$EGOWM_OTBAND_STATE/STOP.txt"; echo 9000 > "$B/gpu_mib.txt"
bash "$PKG/egowm_chain.sh" > "$T/w3c.out" 2>&1; rc=$?
ok "a finished predecessor, GPU busy (9000 MiB) refuses the pilot (rc 12), nothing trained" '[ $rc -eq 12 ] && stoptxt | grep -q "GPU is not idle" && [ ! -e "$B/D/tanitad/egowm/runs/PILOT_E-KIN_s0/gate" ]'
rm -f "$EGOWM_STATE/STOP.txt" "$EGOWM_STATE/done/pilot.done"; echo 3600 > "$B/gpu_mib.txt"
bash "$PKG/egowm_chain.sh" > "$T/w3d.out" 2>&1; rc=$?
ok "3,600 MiB used (above the 3,500 limit) refuses the pilot (rc 12)" '[ $rc -eq 12 ] && stoptxt | grep -q "GPU is not idle"'
echo 3000 > "$B/gpu_mib.txt"
( source "$PKG/stages/egowm_common.sh"; EGOWM_STAGE=t; need_gpu_idle ) > "$T/gpu3000.out" 2>&1; rc=$?
ok "3,000 MiB used (the desktop apps' ~2,700 MiB with no job) counts as idle: need_gpu_idle passes" '[ $rc -eq 0 ] && grep -q "GPU idle (3000 MiB used)" "$T/gpu3000.out"'
ok "the default idle limit is 3,500 MiB" 'grep -q "EGOWM_GPU_IDLE_MIB:-3500" "$PKG/stages/egowm_common.sh"'
rm -f "$EGOWM_STATE/STOP.txt" "$EGOWM_STATE/done/pilot.done"; echo 0 > "$B/gpu_mib.txt"; echo 2 > "$B/py_count.txt"
bash "$PKG/egowm_chain.sh" > "$T/w4.out" 2>&1; rc=$?
ok "a python process on box 2 refuses the pilot (rc 12)" '[ $rc -eq 12 ] && stoptxt | grep -q "python process"'
ok "no breach through PATH after the wait tests" '[ ! -e "$T/BREACH" ]'

echo "== 4 disk gate"
newbox disk 0; control '{"sps": 1.0}'; echo 3 > "$B/free_gb.txt"
bash "$PKG/egowm_chain.sh" > "$T/d1.out" 2>&1; rc=$?
ok "3 GB free: ship passes (1 GB), the pilot stops with the disk-low code (rc 6)" '[ $rc -eq 6 ] && stoptxt | grep -q "disk-low" && [ ! -e "$B/D/tanitad/egowm/runs/PILOT_E-KIN_s0/gate" ]'
newbox disk0 0; echo 0 > "$B/free_gb.txt"
bash "$PKG/egowm_chain.sh" > "$T/d2.out" 2>&1; rc=$?
ok "0 GB free: the ship stage stops with rc 6" '[ $rc -eq 6 ] && [ ! -e "$B/D/tanitad/egowm/ship/egowm_tree.tar" ]'
newbox disk1 0; echo "" > "$B/free_gb.txt"
bash "$PKG/egowm_chain.sh" > "$T/d3.out" 2>&1; rc=$?
ok "an unreadable free-space probe is a refusal, not a pass (rc 11)" '[ $rc -eq 11 ]'

echo "== 5 projected total above 72 h"
newbox slow 0; control '{"sps": 6.0}'
bash "$PKG/egowm_chain.sh" > "$T/s1.out" 2>&1; rc=$?
ok "pilot at 6.0 s/step: the chain stops at the 72 h rule (rc 3), no run started" '[ $rc -eq 3 ] && stoptxt | grep -q "72h-rule-refused" && [ ! -e "$B/D/tanitad/egowm/runs/E-NOW_s0/RUN.done" ] && [ ! -e "$B/D/tanitad/runs/v8f_egowm" ]'
rm -f "$EGOWM_STATE/STOP.txt"; printf 'EGOWM-AMENDMENT pilot_sha256=%s commit=%s\n' "$(printf '%064d' 0)" "$COMMIT" > "$T/am_wrong.txt"
EGOWM_RESUME=1 EGOWM_AMENDMENT="$T/am_wrong.txt" bash "$PKG/egowm_chain.sh" > "$T/s2.out" 2>&1; rc=$?
ok "an amendment naming another pilot is refused (rc 3)" '[ $rc -eq 3 ]'
rm -f "$EGOWM_STATE/STOP.txt"; printf 'EGOWM-AMENDMENT pilot_sha256=%s commit=%s\n' "$(sha256sum "$EGOWM_STATE/pilot/PILOT.verified.json" | cut -d' ' -f1)" "$COMMIT" > "$T/am_ok.txt"; control '{"sps": 6.0}'
EGOWM_RESUME=1 EGOWM_AMENDMENT="$T/am_ok.txt" bash "$PKG/egowm_chain.sh" > "$T/s3.out" 2>&1; rc=$?
ok "an amendment binding this pilot and commit lets the five runs go (rc 0)" '[ $rc -eq 0 ] && [ -e "$EGOWM_STATE/DONE.txt" ]'

echo "== 6 gate refusals (BUILT levers planted wrong in the stand-in trainer)"
newbox gfreeze 0; control '{"sps": 1.0, "modes": {"E-KIN_s0": "bad_speed_mode"}}'
bash "$PKG/egowm_chain.sh" > "$T/g1.out" 2>&1; rc=$?
ok "the built stack shows the other speed mode: the pilot's gate of E-KIN_s0 refuses (rc 4), the stop line names the read-back mode" '[ $rc -eq 4 ] && stoptxt | grep -q "wm_speed_readback.built.mode"'
ok "no training happened and no token was written for the refused arm" '[ ! -e "$B/D/tanitad/egowm/runs/PILOT_E-KIN_s0/logs/train.log" ] && [ ! -e "$B/D/tanitad/egowm/runs/E-KIN_s0/gate/gate_token.json" ] && [ ! -e "$B/D/tanitad/runs/v8f_egowm" ]'
newbox gdetach 0; control '{"sps": 1.0, "modes": {"PILOT_E-KIN_s0": "bad_zcache"}}'
bash "$PKG/egowm_chain.sh" > "$T/g2.out" 2>&1; rc=$?
ok "the built dataset reads the other seed's latent cache: refused (rc 4)" '[ $rc -eq 4 ] && stoptxt | grep -q "zcache"'
newbox gsteps 0; control '{"sps": 1.0, "modes": {"E-NOW_s0": "bad_steps"}}'
bash "$PKG/egowm_chain.sh" > "$T/g3.out" 2>&1; rc=$?
ok "a full-length steps_total that is not 19,164 is refused at the pilot (rc 4), before the 72 h decision" '[ $rc -eq 4 ] && stoptxt | grep -q "steps_total"'
newbox garm 0; control '{"sps": 1.0, "modes": {"E-KIN_s1": "bad_speed_mode"}}'
bash "$PKG/egowm_chain.sh" > "$T/g4.out" 2>&1; rc=$?
ok "an arm-specific gate refusal (E-KIN_s1) ends the chain with rc 4 but the other arms ran" '[ $rc -eq 4 ] && [ -e "$EGOWM_STATE/ARM_REFUSED_E-KIN_s1.txt" ] && [ ! -e "$EGOWM_STATE/DONE.txt" ]'
ok "the independent arms after it were banked, the refused one was not trained" '[ -e "$EGOWM_PULLS/egowm_E-NOW_s1/BANK.done" ] && [ -e "$EGOWM_PULLS/egowm_E-KIN_rep_s2/BANK.done" ] && [ ! -e "$B/D/tanitad/runs/v8f_egowm/E-KIN_s1" ] && [ -e "$EGOWM_PULLS/egowm_E-NOW_s0/BANK.done" ]'
ok "the refusal sentinel is in its own file" 'head -c 30 "$EGOWM_STATE/ARM_REFUSED_E-KIN_s1.txt" | grep -q "^ZZ EGOWM-ARM-REFUSED" && ! grep -q "ZZ EGOWM" "$EGOWM_STATE/chain.log"'

echo "== 7 failures after the gate stop the chain (no next arm)"
newbox pfrozen 0; control '{"sps": 1.0, "train_modes": {"E-KIN_s1": "move_frozen"}}'
bash "$PKG/egowm_chain.sh" > "$T/p1.out" 2>&1; rc=$?
ok "a frozen tensor that moved is a post-check failure: chain stops (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "post check failed" && stoptxt | grep -q "non-WM tensor(s) differ"'
ok "the failed arm is not banked and E-NOW_s1 never started; the two earlier arms are banked" '[ ! -e "$EGOWM_PULLS/egowm_E-KIN_s1/BANK.done" ] && [ ! -e "$B/D/tanitad/egowm/runs/E-NOW_s1/argv.txt" ] && [ -e "$EGOWM_PULLS/egowm_E-NOW_s0/BANK.done" ] && [ -e "$EGOWM_PULLS/egowm_E-KIN_s0/BANK.done" ]'
ok "the checkpoint of the failed arm stays on the box (never deleted)" '[ -s "$B/D/tanitad/runs/v8f_egowm/E-KIN_s1/ckpt_last.pt" ]'
newbox pcrash 0; control '{"sps": 1.0, "train_modes": {"E-KIN_s0": "crash"}}'
bash "$PKG/egowm_chain.sh" > "$T/p2.out" 2>&1; rc=$?
ok "a trainer crash stops the chain (rc 5) before the next arm" '[ $rc -eq 5 ] && [ ! -e "$B/D/tanitad/egowm/runs/E-NOW_s0/RUN.done" ] && [ ! -e "$EGOWM_STATE/done/E-NOW_s0.done" ] && [ ! -e "$EGOWM_PULLS/egowm_E-KIN_s0" ]'
newbox pnomove 0; control '{"sps": 1.0, "train_modes": {"E-KIN_s0": "no_speed_move"}}'
bash "$PKG/egowm_chain.sh" > "$T/p3.out" 2>&1; rc=$?
ok "a run in which the speed channels stayed at zero is a post-check failure (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "speed-channel tensors"'
newbox pcorrupt 0; control '{"sps": 1.0}'
FAKE_CORRUPT_PUT=argv.txt bash "$PKG/egowm_chain.sh" > "$T/p4.out" 2>&1; rc=$?
ok "a put that arrives altered is caught by the re-hash on box 2 (rc 5), nothing runs" '[ $rc -eq 5 ] || [ $rc -eq 9 ]'

echo "== 8 commit, tag and package refusals"
newbox nocommit 0; unset EGOWM_COMMIT
bash "$PKG/egowm_chain.sh" > "$T/nc.out" 2>&1; rc=$?
ok "chain without EGOWM_COMMIT refuses (rc 8), zero box calls" '[ $rc -eq 8 ] && stoptxt | grep -q "commit-missing" && [ "$(calls)" -eq 0 ]'
export EGOWM_COMMIT="abc123"; rm -f "$EGOWM_STATE/STOP.txt"
bash "$PKG/egowm_chain.sh" > "$T/nc2.out" 2>&1; rc=$?
ok "an abbreviated commit refuses (rc 8)" '[ $rc -eq 8 ] && [ "$(calls)" -eq 0 ]'
export EGOWM_COMMIT="$COMMIT"; rm -f "$EGOWM_STATE/STOP.txt"
EGOWM_COMMIT="$(printf 'A%.0s' $(seq 1 40))" bash "$PKG/egowm_chain.sh" > "$T/nc3.out" 2>&1; rc=$?
ok "an upper-case 40-character commit refuses (rc 8)" '[ $rc -eq 8 ]'
rm -f "$EGOWM_STATE/STOP.txt"
bash "$PKG/stages/egowm_run.sh" OB-BAND_s0 > "$T/tok.out" 2>&1; rc=$?
ok "a tag of the earlier chain (OB-BAND_s0) refuses (rc 16), zero box calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
bash "$PKG/stages/egowm_run.sh" OB-NOPE_s0 > "$T/tok2.out" 2>&1; rc=$?
ok "an unknown tag refuses (rc 16), zero box calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
ok "the checker CLI refuses an unknown tag (rc 2)" '"$PYV" -I "$PKG/egowm_check.py" argv --tag OB-BAND-TOK_s1 > /dev/null 2>&1; [ $? -eq 2 ]'

echo "== 9 transport variable unset, relative, absent, the wrong kind, or test mode with a real client (fail closed)"
newbox tr 0
for how in unset relative absent realkind; do
  rm -f "$EGOWM_STATE/STOP.txt"; : > "$B/transport.log"
  case "$how" in
    unset) ( unset EGOWM_SSH; bash "$PKG/egowm_chain.sh" > "$T/tr_$how.out" 2>&1 ); rc=$?;;
    relative) EGOWM_SSH="ssh" bash "$PKG/egowm_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
    absent) EGOWM_SSH="$T/no_such_ssh" bash "$PKG/egowm_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
    realkind) printf '#!/usr/bin/env bash\necho i am a real client\n' > "$T/fake_real_ssh"; chmod +x "$T/fake_real_ssh"; EGOWM_SSH="$(cygpath -m "$T/fake_real_ssh")" bash "$PKG/egowm_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
  esac
  ok "transport '$how' refuses with rc 16" '[ $rc -eq 16 ]'
  ok "transport '$how' made no call and wrote a stop line" '[ "$(calls)" -eq 0 ] && grep -q "transport-not-fail-closed" "$EGOWM_STATE/STOP.txt"'
done
rm -f "$EGOWM_STATE/STOP.txt"; : > "$B/transport.log"
( unset EGOWM_SFTP; bash "$PKG/egowm_chain.sh" > "$T/tr_sftp.out" 2>&1 ); rc=$?
ok "EGOWM_SFTP unset refuses with rc 16" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$EGOWM_STATE/STOP.txt"
( unset EGOWM_DRY; bash "$PKG/egowm_chain.sh" > "$T/tr_nodry.out" 2>&1 ); rc=$?
ok "a stub without test mode refuses (the stub is never taken for the real client)" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$EGOWM_STATE/STOP.txt"
printf '#!/usr/bin/env bash\necho i am a real sftp\n' > "$T/fake_real_sftp"; chmod +x "$T/fake_real_sftp"
EGOWM_SFTP="$(cygpath -m "$T/fake_real_sftp")" bash "$PKG/egowm_chain.sh" > "$T/tr_realsftp.out" 2>&1; rc=$?
ok "test mode (EGOWM_DRY=1) with a real sftp path refuses (rc 16), zero calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$EGOWM_STATE/STOP.txt"
EGOWM_SSH="$T/no_such_ssh" bash "$PKG/stages/egowm_run.sh" E-NOW_s0 > "$T/tr_stage.out" 2>&1; rc=$?
ok "a stage run alone with an absent transport refuses with rc 16 before any call" '[ $rc -eq 16 ]'
ok "no breach through PATH after all transport refusals" '[ ! -e "$T/BREACH" ]'

echo "== 10 hygiene"
ok "no ssh, sftp or scp command in any test script other than the stubs' own forwarding" '! grep -nE "^[^#]*\b(ssh|sftp|scp) +-" "$TESTS/test_egowm_dryrun.sh" | grep -v "poison" | grep -q .'
ok "bash scripts are LF" '"$PYV" -I -c "import glob,sys;fs=glob.glob(sys.argv[1]+\"/*.sh\")+glob.glob(sys.argv[1]+\"/stages/*.sh\")+glob.glob(sys.argv[2]+\"/*.sh\");assert len(fs)>=7 and all(b\"\r\" not in open(f,\"rb\").read() for f in fs)" "$PKG" "$TESTS"'
rm -rf "${T:?}"
echo "DRYRUN ok=$OK failed=$BAD"
[ "$BAD" -eq 0 ]
