#!/usr/bin/env bash
# Dry run of the WMPOOL box-2 package over stubs. NOTHING here can reach box 2: WMPOOL_SSH / WMPOOL_SFTP name stub scripts by absolute path (WMPOOL_DRY=1 makes the chain refuse any file without the stub marker),
# a poison ssh/sftp/scp is put FIRST on PATH and must never run, and the "box" is a folder. The REAL job .cmd files run on Windows cmd.exe against it; only the trainer (tests/plant_trainer.py, shipped through a throw-away
# git repo, so the ship, tree-binding and token code run for real) and the cell builder (tests/plant_zcells.py, named by WMPOOL_TEST_ZCELLS) are stand-ins. CPU only, no GPU, no torch, no model.
# Usage: bash test_wmpool_dryrun.sh  -> prints "DRYRUN ok=<n> failed=<m>"
set -u
TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PKG="$(cd "$TESTS/.." && pwd)"; PKG="$(cygpath -m "$PKG" 2>/dev/null || echo "$PKG")"
PYV="${WMPOOL_TEST_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
T="${WMPOOL_TEST_TMP:-F:/TanitAD-scratch/wmpool_box2_dry/wmpool_dry_run_$$}"
case "$T" in */wmpool_dry_run_*|*/scratchpad/*|*/wmpool_dry_*) ;; *) echo "refusing an unrecognised scratch dir $T" >&2; exit 2;; esac
rm -rf "${T:?}"; mkdir -p "$T/poison"
for n in ssh sftp scp; do printf '#!/usr/bin/env bash\necho BREACH-%s >> "%s/BREACH"\nexit 99\n' "$n" "$T" > "$T/poison/$n"; chmod +x "$T/poison/$n"; done
export PATH="$T/poison:$PATH"
export CUDA_VISIBLE_DEVICES=-1
OK=0; BAD=0
ok() { if eval "$2"; then OK=$((OK+1)); else BAD=$((BAD+1)); echo "FAIL: $1"; fi; }
SSHS="$(cygpath -m "$TESTS/ssh_stub.sh")"; SFTPS="$(cygpath -m "$TESTS/sftp_stub.sh")"; PZC="$(cygpath -m "$TESTS/plant_zcells.py")"
COMMIT="$("$PYV" -W ignore "$TESTS/dry_helpers.py" mkrepo "$T/repo")"
ok "throw-away repo has a 40-hex commit" '[ ${#COMMIT} -eq 40 ]'
newbox() {   # newbox <name> [gpu MiB]  -> exports the dry-run environment for box <name>; the STOPBAL predecessor has ended (DONE.txt in a TEST state folder, never the real one) unless the test says otherwise
  B="$T/box_$1"; "$PYV" -W ignore "$TESTS/dry_helpers.py" mkbox "$B" "$PYV" "${2:-0}" > /dev/null
  mkdir -p "$B/D/tanitad/stopbal/runs" "$B/D/tanitad/egowm/runs" "$B/D/tanitad/repo/stack"; echo "stopbal file" > "$B/D/tanitad/stopbal/runs/BAL-G_s0.txt"; echo "egowm file" > "$B/D/tanitad/egowm/runs/E-KIN_s0.txt"; echo "shared repo file" > "$B/D/tanitad/repo/stack/x.py"
  export WMPOOL_DRY=1 WMPOOL_SSH="$SSHS" WMPOOL_SFTP="$SFTPS" FAKE_BOX="$(cygpath -m "$B")" FAKE_PY="$PYV" WMPOOL_PY="$PYV" WMPOOL_STATE="$T/st_$1" WMPOOL_PULLS="$T/pulls_$1"
  export WMPOOL_BUSY_DEV="$T/busy_$1.txt" WMPOOL_ASIDE_DEV="$T/aside_$1" WMPOOL_B2_WIN_ROOT="$(cygpath -w "$B/D/tanitad")" WMPOOL_POLL_S=0 WMPOOL_REPO="$T/repo" WMPOOL_EXPORT_LIST="$T/repo/export_list.txt"
  export WMPOOL_COMMIT="$COMMIT" WMPOOL_NEED_C_MB=1 WMPOOL_NEED_C_MB_CELLS=1 WMPOOL_POLLS_RUN=5 WMPOOL_POLLS_SHIP=5 WMPOOL_POLLS_CLEAN=5 WMPOOL_POLLS_CELLS=5 WMPOOL_WAIT_POLLS=0 WMPOOL_WAIT_POLL_S=0
  export WMPOOL_TEST_ZCELLS="$PZC" WMPOOL_TEST_CELL_COUNTS="train=6,eval=3" WMPOOL_FAKE_FREE_GB=100
  export WMPOOL_STOPBAL_STATE="$T/stopbal_state_$1"; mkdir -p "$WMPOOL_STOPBAL_STATE"; echo "ZZ STOPBAL-DONE 2026-11-11 00:00:00 commit=x runs=x" > "$WMPOOL_STOPBAL_STATE/DONE.txt"
  unset WMPOOL_RESUME WMPOOL_AMENDMENT WMPOOL_GPU_IDLE_MIB FAKE_CORRUPT_GET FAKE_CORRUPT_PUT
}
control() { printf '%s' "$1" > "$B/plant_control.json"; }
calls() { cat "$B/transport.log" 2>/dev/null | wc -l | tr -d ' '; }
stoptxt() { cat "$WMPOOL_STATE/STOP.txt" 2>/dev/null; }
treehash() { ( cd "$1" && find . -type f | sort | xargs sha256sum | sha256sum | cut -d' ' -f1 ); }

echo "== 1 normal flow: wait check, ship, cells, pilot, 72 h permit, three runs, banks, cleanups"
newbox normal 0; control '{"sps": 1.0, "startup_s": 30}'
SH0="$(treehash "$B/D/tanitad/stopbal")"; EH0="$(treehash "$B/D/tanitad/egowm")"; RH0="$(treehash "$B/D/tanitad/repo")"
bash "$PKG/wmpool_chain.sh" > "$T/normal.out" 2>&1; rc=$?
ok "chain exits 0" '[ $rc -eq 0 ]'
ok "DONE sentinel written, no stop" '[ -e "$WMPOOL_STATE/DONE.txt" ] && [ ! -e "$WMPOOL_STATE/STOP.txt" ] && head -c 15 "$WMPOOL_STATE/DONE.txt" | grep -q "^ZZ WMPOOL-DONE"'
ok "sentinel text only in its own file (chain.log carries none)" '! grep -q "ZZ WMPOOL" "$WMPOOL_STATE/chain.log"'
ok "stage order in chain.log: ship, cells, pilot, P-ATT_s0, P-ATT_s1, P-ATT_rep_s2" '[ "$(grep -o "done [A-Za-z0-9_-]*" "$WMPOOL_STATE/chain.log" | tr "\n" " ")" = "done ship done cells done pilot done P-ATT_s0 done P-ATT_s1 done P-ATT_rep_s2 " ]'
ok "the chain saw the predecessor ended before it took the box" 'grep -q "the STOPBAL chain ended (DONE)" "$WMPOOL_STATE/chain.log"'
CB="$B/D/tanitad/wmpool/cells"
ok "the cells job wrote CELLS.json and its done marker on the box and the dev box banked CELLS.json + ZCELLS_CHECK3.json" '[ -s "$CB/CELLS.json" ] && ls "$CB"/CELLS_*.done > /dev/null 2>&1 && [ -s "$WMPOOL_STATE/cells/CELLS.json" ] && [ -s "$WMPOOL_STATE/cells/ZCELLS_CHECK3.json" ]'
ok "the dev box re-judged the pulled cells record" '"$PYV" -I "$PKG/wmpool_check.py" verify-cells --cells-json "$WMPOOL_STATE/cells/CELLS.json" --check3-json "$WMPOOL_STATE/cells/ZCELLS_CHECK3.json" --commit "$COMMIT" > /dev/null'
ok "the cell cache exists for both seeds with a manifest per split and the planted counts (6 train, 3 eval)" '[ "$(ls "$B/D/tanitad/zcells/M2b_s0/train" | grep -c npz)" -eq 6 ] && [ "$(ls "$B/D/tanitad/zcells/M2b_s1/eval" | grep -c npz)" -eq 3 ] && [ -s "$B/D/tanitad/zcells/M2b_s1/train/CELL_MANIFEST.json" ] && [ -s "$B/D/tanitad/zcells/M2b_s0/eval/CELL_MANIFEST.json" ]'
ok "the cell cache is NOT deleted by any cleanup (it is the input of every run)" '[ "$(ls "$B/D/tanitad/zcells/M2b_s0/train" | grep -c npz)" -eq 6 ] && ! grep -rq zcells "$WMPOOL_PULLS"/wmpool_*/CLEANUP.json "$WMPOOL_STATE/pilot/CLEANUP.json"'
ok "the shipped cell code on the box equals the package bytes (zcells.py, zcache.py, clips8.json)" 'for f in zcells.py zcache.py clips8.json; do [ "$(sha256sum "$B/D/tanitad/wmpool/cellcode/$f" | cut -d" " -f1)" = "$(sha256sum "$PKG/frozen/$f" | cut -d" " -f1)" ] || exit 1; done'
P="$WMPOOL_STATE/pilot/PILOT.verified.json"
ok "PILOT.verified.json names the commit" 'grep -q "\"commit\": \"$COMMIT\"" "$P"'
ok "pilot mean s/step excludes start-up (reads 1.0)" '"$PYV" -I -c "import json,sys;d=json.load(open(sys.argv[1]));assert abs(d[\"timing\"][\"mean_s_per_step\"]-1.0)<1e-9, d[\"timing\"]" "$P"'
ok "projection covers 3 runs x 19,164 steps" '"$PYV" -I -c "import json,sys;d=json.load(open(sys.argv[1]))[\"projection\"];assert d[\"steps_total\"]==57492 and d[\"n_runs_planned\"]==3 and d[\"within_limit_planned\"]" "$P"'
ok "pilot used the exact P-ATT_s0 argv plus --steps 200 and a separate out dir" '"$PYV" -I -c "import sys;sys.path.insert(0,sys.argv[1]);import wmpool_check as C;l=open(sys.argv[2]).read().split();assert l==C.tokens(\"PILOT_P-ATT_s0\") and l[-2:]==[\"--steps\",\"200\"] and \"pilot_out\" in l[l.index(\"--out\")+1]" "$PKG" "$WMPOOL_STATE/runs/PILOT_P-ATT_s0/argv.txt"'
ok "the pilot token binds commit and argv" '"$PYV" -I "$PKG/wmpool_check.py" verify-token --token "$WMPOOL_STATE/pilot/gate_token.json" --argv-file "$WMPOOL_STATE/runs/PILOT_P-ATT_s0/argv.txt" --tag PILOT_P-ATT_s0 --commit "$COMMIT" > /dev/null'
ok "the pilot gated all three full-length argvs (a token for each on the box before any run)" 'for t in P-ATT_s0 P-ATT_s1 P-ATT_rep_s2; do [ -s "$B/D/tanitad/wmpool/runs/$t/gate/gate_token.json" ] || exit 1; done'
ok "pilot temp dirs deleted on the box (pilot out, gate outs of the pilot and of the three gated argvs)" '[ ! -e "$B/D/tanitad/wmpool/pilot_out/P-ATT_s0" ] && [ ! -e "$B/D/tanitad/wmpool/gate_out/PILOT_P-ATT_s0" ] && [ ! -e "$B/D/tanitad/wmpool/gate_out/P-ATT_s1" ] && [ ! -e "$B/D/tanitad/wmpool/gate_out/P-ATT_rep_s2" ]'
ok "pilot deletions are logged (path, bytes) and banked" 'grep -q "\"bytes\"" "$WMPOOL_STATE/pilot/CLEANUP.json" && grep -q "pilot_out" "$WMPOOL_STATE/pilot/CLEANUP.json"'
for t in P-ATT_s0 P-ATT_s1 P-ATT_rep_s2; do
  BD="$WMPOOL_PULLS/wmpool_$t"
  ok "$t banked with BANK.done and the six evidence files" '[ -e "$BD/BANK.done" ] && [ -s "$BD/ckpt_last.pt" ] && [ -s "$BD/summary.json" ] && [ -s "$BD/train_log.jsonl" ] && [ -s "$BD/gate_token.json" ] && [ -s "$BD/selfcheck.json" ] && [ -s "$BD/RUN.done" ]'
  ok "$t banked checkpoint equals the box copy (sha256)" '[ "$(sha256sum "$BD/ckpt_last.pt" | cut -d" " -f1)" = "$(sha256sum "$B/D/tanitad/runs/v8f_wmpool/$t/ckpt_last.pt" | cut -d" " -f1)" ]'
  ok "$t token binds the commit and the argv" '"$PYV" -I "$PKG/wmpool_check.py" verify-token --token "$BD/gate_token.json" --argv-file "$BD/argv.txt" --tag $t --commit "$COMMIT" > /dev/null'
  ok "$t RUN.done carries the weights identity (non-WM tensors equal, the pool query non-zero)" 'grep -q "\"n_pool_q_nonzero\": 1" "$BD/RUN.done" && grep -q "\"n_nonwm_equal\"" "$BD/RUN.done"'
  ok "$t selfcheck banked carries the pool read-back (levers.wm_pool_readback, att25)" 'grep -q "wm_pool_readback" "$BD/selfcheck.json" && grep -q "\"att25\"" "$BD/selfcheck.json"'
  ok "$t gate dir on the box deleted, the checkpoint kept" '[ ! -e "$B/D/tanitad/wmpool/gate_out/$t" ] && [ -s "$B/D/tanitad/runs/v8f_wmpool/$t/ckpt_last.pt" ]'
  ok "$t cleanup logged (path, bytes) in the bank" 'grep -q "\"bytes\"" "$BD/CLEANUP.json" || grep -q "deleted_own_temp\": \[\]" "$BD/CLEANUP.json"'
done
ok "the three tokens bind three distinct argvs" '[ "$(cat "$WMPOOL_PULLS"/wmpool_*/gate_token.json | grep -o "\"argv_sha256\": \"[0-9a-f]*\"" | sort -u | wc -l | tr -d " ")" -eq 3 ]'
ok "only the three P-ATT runs exist on the box (own folders only)" '[ "$(ls "$B/D/tanitad/runs/v8f_wmpool" | wc -l | tr -d " ")" -eq 3 ]'
ok "the box saw only stub traffic and no unknown command" '[ "$(calls)" -gt 100 ] && ! grep -q "UNKNOWN COMMAND" "$B/transport.log"'
ok "disk was read before every stage (>= 6 reads: ship, cells, pilot, three runs)" '[ "$(grep -c "Get-PSDrive D" "$B/transport.log")" -ge 6 ]'
ok "isolation: the STOPBAL folder on the box is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/stopbal")" = "$SH0" ]'
ok "isolation: the EGOWM folder on the box is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/egowm")" = "$EH0" ]'
ok "isolation: the shared D:\\tanitad\\repo is byte-identical afterwards" '[ "$(treehash "$B/D/tanitad/repo")" = "$RH0" ]'
ok "own tree exists with the trainer and TIP_WMPOOL.txt" '[ -s "$B/D/tanitad/wmpool/repo/stack/scripts/v8f_mini_train.py" ] && [ "$(cat "$B/D/tanitad/wmpool/repo/TIP_WMPOOL.txt" | tr -d "\r\n")" = "$COMMIT" ]'
ok "busy marker handed on" '[ ! -e "$WMPOOL_BUSY_DEV" ]'
ok "no breach through PATH" '[ ! -e "$T/BREACH" ]'
ok "a second chain run skips finished stages (ship, cells, pilot, runs)" 'bash "$PKG/wmpool_chain.sh" > "$T/normal2.out" 2>&1; grep -q "skip cells (done)" "$WMPOOL_STATE/chain.log" && grep -q "all runs banked" "$T/normal2.out"'
NORMAL_B="$B"

echo "== 2 the box-side pin refuses a changed token (checker run directly on the fake box, no transport)"
B="$NORMAL_B"; RD="$B/D/tanitad/wmpool/runs/P-ATT_s1"; cp "$RD/argv.txt" "$T/argv.keep"
sed -i 's/--seed 1/--seed 7/' "$RD/argv.txt"
( export WMPOOL_ROOT="$(cygpath -m "$B/D/tanitad")" WMPOOL_COMMIT="$COMMIT" WMPOOL_FAKE_INIT_SHA_FILE="$(cygpath -m "$B/D/tanitad/init_sha.json")" WMPOOL_FLOOR_BYTES=1000 WMPOOL_FAKE_V9_MD5="$(md5sum "$B/D/tanitad/data/v9/v9_labels_train.npz" | cut -d' ' -f1)"
  "$PYV" -I "$B/D/tanitad/wmpool/check/wmpool_check.py" stage --tag P-ATT_s1 --gid pin1 > "$T/pin.out" 2>&1 ); rc=$?
ok "a changed seed token is refused (rc 3) and the first differing token is named" '[ $rc -eq 3 ] && grep -q "pin-fail" "$RD/STAGE_pin1.fail" && grep -q "token " "$RD/STAGE_pin1.fail" && grep -q "frozen argv has" "$RD/STAGE_pin1.fail"'
cp "$T/argv.keep" "$RD/argv.txt"

echo "== 3 the STOPBAL wait: not ended, busy marker, bad marker, STOP busy (waits), STOP not busy (goes on), GPU limits"
newbox wait 0; rm -f "$WMPOOL_STOPBAL_STATE/DONE.txt"
bash "$PKG/wmpool_chain.sh" > "$T/w1.out" 2>&1; rc=$?
ok "no STOPBAL DONE/STOP marker: refused with rc 12, zero box calls" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ] && stoptxt | grep -q "predecessor-not-ended"'
rm -f "$WMPOOL_STATE/STOP.txt"; echo "ZZ STOPBAL-DONE x" > "$WMPOOL_STOPBAL_STATE/DONE.txt"; echo "stopbal chain 1" > "$WMPOOL_BUSY_DEV"
bash "$PKG/wmpool_chain.sh" > "$T/w2.out" 2>&1; rc=$?
ok "DONE present but the busy marker held: refused with rc 12, zero box calls" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$WMPOOL_STATE/STOP.txt" "$WMPOOL_BUSY_DEV"; echo "not a finished marker" > "$WMPOOL_STOPBAL_STATE/DONE.txt"
bash "$PKG/wmpool_chain.sh" > "$T/w2b.out" 2>&1; rc=$?
ok "a DONE file without the STOPBAL prefix is not a finished marker (rc 12)" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$WMPOOL_STATE/STOP.txt" "$WMPOOL_STOPBAL_STATE/DONE.txt"; echo "ZZ OBD-DONE 2026-11-10" > "$WMPOOL_STOPBAL_STATE/DONE.txt"
bash "$PKG/wmpool_chain.sh" > "$T/w2c.out" 2>&1; rc=$?
ok "another chain's DONE marker (ZZ OBD-DONE) in the STOPBAL folder is not STOPBAL's (rc 12)" '[ $rc -eq 12 ] && [ "$(calls)" -eq 0 ]'
rm -f "$WMPOOL_STATE/STOP.txt" "$WMPOOL_STOPBAL_STATE/DONE.txt"
echo "ZZ STOPBAL-STOP pilot box2-busy-or-gpu-not-idle rc=12 :: run:PILOT_BAL-G_s0: box 2 GPU is not idle: 2471 MiB used >= 1500" > "$WMPOOL_STOPBAL_STATE/STOP.txt"
bash "$PKG/wmpool_chain.sh" > "$T/w3.out" 2>&1; rc=$?
ok "STOPBAL stopped on a busy/GPU stop (the real line of 2026-11-11): this chain keeps waiting and then refuses with rc 12 (it may be resumed), zero box calls, box not claimed" '[ $rc -eq 12 ] && stoptxt | grep -q "predecessor-not-ended" && stoptxt | grep -q "STOP-BUSY" && [ "$(calls)" -eq 0 ] && [ ! -e "$WMPOOL_BUSY_DEV" ]'
rm -f "$WMPOOL_STATE/STOP.txt"; echo "ZZ STOPBAL-STOP start obd-not-ended (no ZZ OBD-DONE in x) rc=12" > "$WMPOOL_STOPBAL_STATE/STOP.txt"
bash "$PKG/wmpool_chain.sh" > "$T/w3a.out" 2>&1; rc=$?
ok "the early-refusal form of a busy stop (rc=12 at the end of the line) also keeps waiting" '[ $rc -eq 12 ] && stoptxt | grep -q "STOP-BUSY" && [ "$(calls)" -eq 0 ]'
rm -f "$WMPOOL_STATE/STOP.txt"; echo 9000 > "$B/gpu_mib.txt"; echo "ZZ STOPBAL-STOP runs arms-refused: BAL-NULL_s0 rc=4" > "$WMPOOL_STOPBAL_STATE/STOP.txt"
bash "$PKG/wmpool_chain.sh" > "$T/w3b.out" 2>&1; rc=$?
ok "STOPBAL ended in a STOP that is not a busy stop (arms-refused rc=4): this chain goes on, claims the box and is refused only at the GPU check (rc 12)" 'grep -q "the STOPBAL chain ended (STOP)" "$WMPOOL_STATE/chain.log" && [ $rc -eq 12 ] && stoptxt | grep -q "GPU is not idle"'
rm -f "$WMPOOL_STATE/STOP.txt"; rm -rf "$WMPOOL_STATE/done"; echo "ZZ STOPBAL-STOP train crash rc=5 :: run: a log line that says rc=12 inside the detail" > "$WMPOOL_STOPBAL_STATE/STOP.txt"
bash "$PKG/wmpool_chain.sh" > "$T/w3c.out" 2>&1; rc=$?
ok "an rc=12 inside the DETAIL of a non-busy stop does not read as a busy stop (the detail is cut off at ' :: ')" 'grep -q "the STOPBAL chain ended (STOP)" "$WMPOOL_STATE/chain.log" && [ $rc -eq 12 ] && stoptxt | grep -q "GPU is not idle"'
rm -f "$WMPOOL_STATE/STOP.txt" "$WMPOOL_STOPBAL_STATE/STOP.txt"; rm -rf "$WMPOOL_STATE/done"; echo "ZZ STOPBAL-DONE x" > "$WMPOOL_STOPBAL_STATE/DONE.txt"; echo 9000 > "$B/gpu_mib.txt"
bash "$PKG/wmpool_chain.sh" > "$T/w3d.out" 2>&1; rc=$?
ok "a finished predecessor, GPU busy (9000 MiB) refuses at the first GPU check (rc 12), nothing trained" '[ $rc -eq 12 ] && stoptxt | grep -q "GPU is not idle" && [ ! -e "$B/D/tanitad/wmpool/runs/PILOT_P-ATT_s0/gate" ]'
rm -f "$WMPOOL_STATE/STOP.txt"; rm -rf "$WMPOOL_STATE/done"; echo 3600 > "$B/gpu_mib.txt"
bash "$PKG/wmpool_chain.sh" > "$T/w3e.out" 2>&1; rc=$?
ok "3,600 MiB used (above the default 3,500 limit) is refused (rc 12)" '[ $rc -eq 12 ] && stoptxt | grep -q "GPU is not idle: 3600 MiB used >= 3500"'
rm -f "$WMPOOL_STATE/STOP.txt"; rm -rf "$WMPOOL_STATE/done"; echo 2471 > "$B/gpu_mib.txt"
WMPOOL_GPU_IDLE_MIB=1500 bash "$PKG/wmpool_chain.sh" > "$T/w3f.out" 2>&1; rc=$?
ok "WMPOOL_GPU_IDLE_MIB=1500 (the limit that stopped two chains): the PI's desktop apps at 2,471 MiB refuse (rc 12)" '[ $rc -eq 12 ] && stoptxt | grep -q "2471 MiB used >= 1500"'
echo 3000 > "$B/gpu_mib.txt"
( source "$PKG/stages/wmpool_common.sh"; WMPOOL_STAGE=t; need_gpu_idle ) > "$T/gpu3000.out" 2>&1; rc=$?
ok "3,000 MiB used passes with the default limit (the desktop apps' ~2,400-2,700 MiB with no job)" '[ $rc -eq 0 ] && grep -q "GPU idle (3000 MiB used)" "$T/gpu3000.out"'
( export WMPOOL_GPU_IDLE_MIB=5000; echo 4800 > "$B/gpu_mib.txt"; source "$PKG/stages/wmpool_common.sh"; WMPOOL_STAGE=t; need_gpu_idle ) > "$T/gpu5000.out" 2>&1; rc=$?
ok "WMPOOL_GPU_IDLE_MIB=5000 raises the limit: 4,800 MiB passes" '[ $rc -eq 0 ] && grep -q "GPU idle (4800 MiB used)" "$T/gpu5000.out"'
ok "the default idle limit is 3,500 MiB" 'grep -q "WMPOOL_GPU_IDLE_MIB:-3500" "$PKG/stages/wmpool_common.sh"'
rm -f "$WMPOOL_STATE/STOP.txt"; rm -rf "$WMPOOL_STATE/done"; echo 0 > "$B/gpu_mib.txt"; echo 2 > "$B/py_count.txt"
bash "$PKG/wmpool_chain.sh" > "$T/w4.out" 2>&1; rc=$?
ok "a python process on box 2 refuses (rc 12) whatever the GPU memory reads" '[ $rc -eq 12 ] && stoptxt | grep -q "python process"'
ok "no breach through PATH after the wait tests" '[ ! -e "$T/BREACH" ]'

echo "== 4 disk gates (dev-box reads and the box-side cells check)"
newbox disk 0; control '{"sps": 1.0}'; echo 3 > "$B/free_gb.txt"
bash "$PKG/wmpool_chain.sh" > "$T/d1.out" 2>&1; rc=$?
ok "3 GB free: ship passes (1 GB), the cells stage stops with the disk-low code (rc 6) before it builds anything" '[ $rc -eq 6 ] && stoptxt | grep -q "disk-low" && [ ! -e "$B/D/tanitad/zcells" ]'
newbox disk9 0; echo 5 > "$B/free_gb.txt"
bash "$PKG/wmpool_chain.sh" > "$T/d1b.out" 2>&1; rc=$?
ok "5 GB free (< cache + 1 GB, asked as 6): the cells stage stops with rc 6, no cache dir" '[ $rc -eq 6 ] && stoptxt | grep -q "disk-low" && [ ! -e "$B/D/tanitad/zcells" ]'
newbox disk0 0; echo 0 > "$B/free_gb.txt"
bash "$PKG/wmpool_chain.sh" > "$T/d2.out" 2>&1; rc=$?
ok "0 GB free: the ship stage stops with rc 6" '[ $rc -eq 6 ] && [ ! -e "$B/D/tanitad/wmpool/ship/wmpool_tree.tar" ]'
newbox disk1 0; echo "" > "$B/free_gb.txt"
bash "$PKG/wmpool_chain.sh" > "$T/d3.out" 2>&1; rc=$?
ok "an unreadable free-space probe is a refusal, not a pass (rc 11)" '[ $rc -eq 11 ]'
newbox diskbox 0; control '{"sps": 1.0}'; export WMPOOL_FAKE_FREE_GB=5.5
bash "$PKG/wmpool_chain.sh" > "$T/d4.out" 2>&1; rc=$?
ok "the box-side check (5.5 GB < cache + 1 GB) fails the cells job with disk-low (rc 6) and nothing is built" '[ $rc -eq 6 ] && stoptxt | grep -q "disk-low" && [ ! -e "$B/D/tanitad/zcells" ]'
ok "the pilot never started after the refused cells stage" '[ ! -e "$B/D/tanitad/wmpool/runs/PILOT_P-ATT_s0" ]'

echo "== 5 projected total above 72 h"
newbox slow 0; control '{"sps": 6.0}'
bash "$PKG/wmpool_chain.sh" > "$T/s1.out" 2>&1; rc=$?
ok "pilot at 6.0 s/step: the chain stops at the 72 h rule (rc 3), no run started" '[ $rc -eq 3 ] && stoptxt | grep -q "72h-rule-refused" && [ ! -e "$B/D/tanitad/wmpool/runs/P-ATT_s0/RUN.done" ] && [ ! -e "$B/D/tanitad/runs/v8f_wmpool" ]'
ok "the cells stage had already passed (cache kept)" '[ -e "$WMPOOL_STATE/done/cells.done" ] && [ -d "$B/D/tanitad/zcells/M2b_s0/train" ]'
rm -f "$WMPOOL_STATE/STOP.txt"; printf 'WMPOOL-AMENDMENT pilot_sha256=%s commit=%s\n' "$(printf '%064d' 0)" "$COMMIT" > "$T/am_wrong.txt"
WMPOOL_RESUME=1 WMPOOL_AMENDMENT="$T/am_wrong.txt" bash "$PKG/wmpool_chain.sh" > "$T/s2.out" 2>&1; rc=$?
ok "an amendment naming another pilot is refused (rc 3)" '[ $rc -eq 3 ]'
rm -f "$WMPOOL_STATE/STOP.txt"; printf 'WMPOOL-AMENDMENT pilot_sha256=%s commit=%s\n' "$(sha256sum "$WMPOOL_STATE/pilot/PILOT.verified.json" | cut -d' ' -f1)" "$COMMIT" > "$T/am_ok.txt"
WMPOOL_RESUME=1 WMPOOL_AMENDMENT="$T/am_ok.txt" bash "$PKG/wmpool_chain.sh" > "$T/s3.out" 2>&1; rc=$?
ok "an amendment binding this pilot and commit lets the three runs go (rc 0)" '[ $rc -eq 0 ] && [ -e "$WMPOOL_STATE/DONE.txt" ]'

echo "== 6 cells-stage refusals (a deliberate-regression arm per refusal; nothing is trained, the pilot never starts)"
for cm in check3_fail wrongseed_passes stored_off short_counts same_adapter tmp_left no_manifest; do
  newbox "c_$cm" 0; control "{\"sps\": 1.0, \"cells_mode\": \"$cm\"}"
  bash "$PKG/wmpool_chain.sh" > "$T/c_$cm.out" 2>&1; rc=$?
  ok "cells '$cm': the chain refuses (rc 4: gate/check refusal), no pilot, no gate, no training, STOP names the cells stage" '[ $rc -eq 4 ] && stoptxt | grep -q "cells" && [ ! -e "$WMPOOL_STATE/done/cells.done" ] && [ ! -e "$B/D/tanitad/wmpool/runs/PILOT_P-ATT_s0" ] && [ ! -e "$B/D/tanitad/runs/v8f_wmpool" ] && [ ! -e "$WMPOOL_STATE/DONE.txt" ]'
done
newbox c_crash 0; control '{"sps": 1.0, "cells_mode": "build_crash"}'
bash "$PKG/wmpool_chain.sh" > "$T/c_crash.out" 2>&1; rc=$?
ok "a crashed cell build stops the chain (rc 5), nothing is trained" '[ $rc -eq 5 ] && stoptxt | grep -q "cells job failed" && [ ! -e "$B/D/tanitad/wmpool/runs/PILOT_P-ATT_s0" ]'
ok "the real zcells.py is what the ship stage puts on the box (sha-checked on both ends) unless the dry-run hook names a stand-in" 'grep -q "put_verified \"\$PKG/frozen/\$f\" \"\$B2_ROOT/cellcode/\$f\"" "$PKG/stages/wmpool_ship.sh"'
newbox c_alter 0; control '{"sps": 1.0}'
FAKE_CORRUPT_PUT=zcells.py bash "$PKG/wmpool_chain.sh" > "$T/c_alter.out" 2>&1; rc=$?
ok "a zcells.py that arrives altered is caught by the re-hash after the put (rc 5), the cells job never runs" '[ $rc -eq 5 ] && [ ! -e "$B/D/tanitad/zcells" ]'
newbox c_pin 0; control '{"sps": 1.0}'
( export WMPOOL_ROOT="$(cygpath -m "$B/D/tanitad")" WMPOOL_COMMIT="$COMMIT"; unset WMPOOL_TEST_ZCELLS WMPOOL_DRY; mkdir -p "$B/D/tanitad/wmpool/cellcode"; echo "tampered" > "$B/D/tanitad/wmpool/cellcode/zcells.py"; cp "$PKG/frozen/zcache.py" "$PKG/frozen/clips8.json" "$B/D/tanitad/wmpool/cellcode/"
  "$PYV" -I "$PKG/wmpool_check.py" cells --gid pin > "$T/c_pin.out" 2>&1 ); rc=$?
ok "on the box the cells job refuses a zcells.py whose bytes differ from the pinned sha256 (outside a dry run), before it builds" '[ $rc -ne 0 ] && grep -q "differs from the pinned sha256" "$B/D/tanitad/wmpool/cells/CELLS_pin.fail" && [ ! -e "$B/D/tanitad/zcells" ]'

echo "== 7 gate refusals at the pilot (BUILT levers planted wrong in the stand-in trainer; the pilot gates all three argvs)"
for spec in "PILOT_P-ATT_s0:bad_pool_mode:wm_pool_readback.built.mode" "P-ATT_s0:query_not_zero:q_zero_at_build" "P-ATT_s1:not_uniform:uniform_at_build" "P-ATT_rep_s2:no_attention:attends_when_set" "P-ATT_s0:bad_cell_map:cell_map_ok" \
            "P-ATT_s1:no_pool_readback:wm_pool_readback is missing" "P-ATT_s0:no_pool_lever:wm_pool" "P-ATT_s1:bad_cell_shape:dataset_zcell_shape" "P-ATT_rep_s2:cache_problem:cache_problems" "P-ATT_s0:bad_steps:steps_total" \
            "P-ATT_s1:bad_zcache:cell cache" "P-ATT_s0:mean_cache:cell cache"; do
  tg="${spec%%:*}"; r_="${spec#*:}"; md="${r_%%:*}"; nm="${r_#*:}"
  newbox "g_${tg}_$md" 0; control "{\"sps\": 1.0, \"modes\": {\"$tg\": \"$md\"}}"
  bash "$PKG/wmpool_chain.sh" > "$T/g_$md.out" 2>&1; rc=$?
  ok "planted '$md' on the gate of $tg: the pilot job refuses (rc 4) and the stop line names '$nm'" '[ $rc -eq 4 ] && stoptxt | grep -q "$nm"'
  ok "'$md': no training happened, no run out dir, the token of the refused argv does not exist" '[ ! -e "$B/D/tanitad/wmpool/runs/PILOT_P-ATT_s0/logs/train.log" ] && [ ! -e "$B/D/tanitad/runs/v8f_wmpool" ] && [ ! -e "$B/D/tanitad/wmpool/runs/$tg/gate/gate_token.json" ]'
done
echo "-- an arm refused at its OWN gate after the pilot (ARM_REFUSED, the chain goes on)"
newbox arm 0; control '{"sps": 1.0}'
bash "$PKG/wmpool_chain.sh" > "$T/arm1.out" 2>&1
rm -f "$WMPOOL_STATE/done/P-ATT_rep_s2.done" "$WMPOOL_STATE/DONE.txt" "$B/D/tanitad/wmpool/runs/P-ATT_rep_s2/gate/gate_token.json"; rm -rf "$WMPOOL_PULLS/wmpool_P-ATT_rep_s2" "$B/D/tanitad/runs/v8f_wmpool/P-ATT_rep_s2"
control '{"sps": 1.0, "modes": {"P-ATT_rep_s2": "bad_pool_mode"}}'
bash "$PKG/wmpool_chain.sh" > "$T/arm2.out" 2>&1; rc=$?
ok "the refused arm ends the chain with rc 4 and no DONE; ARM_REFUSED_P-ATT_rep_s2.txt is its own file; the other two arms stay banked" '[ $rc -eq 4 ] && [ -e "$WMPOOL_STATE/ARM_REFUSED_P-ATT_rep_s2.txt" ] && [ ! -e "$WMPOOL_STATE/DONE.txt" ] && head -c 30 "$WMPOOL_STATE/ARM_REFUSED_P-ATT_rep_s2.txt" | grep -q "^ZZ WMPOOL-ARM-REFUSED" && [ -e "$WMPOOL_PULLS/wmpool_P-ATT_s0/BANK.done" ] && [ -e "$WMPOOL_PULLS/wmpool_P-ATT_s1/BANK.done" ] && [ ! -e "$B/D/tanitad/runs/v8f_wmpool/P-ATT_rep_s2" ]'
ok "the arm refusal is a STOP (never DONE) and the sentinel text is not in chain.log" 'stoptxt | grep -q "arms-refused: P-ATT_rep_s2" && ! grep -q "ZZ WMPOOL" "$WMPOOL_STATE/chain.log"'

echo "== 8 failures after the gate stop the chain (no next arm)"
newbox pfrozen 0; control '{"sps": 1.0, "train_modes": {"P-ATT_s1": "move_frozen"}}'
bash "$PKG/wmpool_chain.sh" > "$T/p1.out" 2>&1; rc=$?
ok "a frozen tensor that moved is a post-check failure: chain stops (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "post check failed" && stoptxt | grep -q "non-WM tensor(s) differ"'
ok "the failed arm is not banked and P-ATT_rep_s2 never started; the first arm is banked" '[ ! -e "$WMPOOL_PULLS/wmpool_P-ATT_s1/BANK.done" ] && [ ! -e "$B/D/tanitad/wmpool/runs/P-ATT_rep_s2/RUN.done" ] && [ -e "$WMPOOL_PULLS/wmpool_P-ATT_s0/BANK.done" ]'
ok "the checkpoint of the failed arm stays on the box (never deleted)" '[ -s "$B/D/tanitad/runs/v8f_wmpool/P-ATT_s1/ckpt_last.pt" ]'
newbox pcrash 0; control '{"sps": 1.0, "train_modes": {"P-ATT_s0": "crash"}}'
bash "$PKG/wmpool_chain.sh" > "$T/p2.out" 2>&1; rc=$?
ok "a trainer crash stops the chain (rc 5) before the next arm" '[ $rc -eq 5 ] && [ ! -e "$B/D/tanitad/wmpool/runs/P-ATT_s1/RUN.done" ] && [ ! -e "$WMPOOL_STATE/done/P-ATT_s1.done" ] && [ ! -e "$WMPOOL_PULLS/wmpool_P-ATT_s0" ]'
newbox pnomove 0; control '{"sps": 1.0, "train_modes": {"P-ATT_s0": "no_pool_move"}}'
bash "$PKG/wmpool_chain.sh" > "$T/p3.out" 2>&1; rc=$?
ok "a run in which the pool query stayed at zero is a post-check failure (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "pool-query tensors"'
newbox pnoq 0; control '{"sps": 1.0, "train_modes": {"P-ATT_s0": "no_pool_q"}}'
bash "$PKG/wmpool_chain.sh" > "$T/p3b.out" 2>&1; rc=$?
ok "a checkpoint with no pool query at all is a post-check failure (rc 5)" '[ $rc -eq 5 ] && stoptxt | grep -q "pool-query tensors"'
newbox pcorrupt 0; control '{"sps": 1.0}'
FAKE_CORRUPT_PUT=argv.txt bash "$PKG/wmpool_chain.sh" > "$T/p4.out" 2>&1; rc=$?
ok "a put that arrives altered is caught by the re-hash on box 2 (rc 5), nothing runs" '[ $rc -eq 5 ] || [ $rc -eq 9 ]'

echo "== 9 bank sha mismatch must go red (a pull that arrives altered)"
newbox bankbad 0; control '{"sps": 1.0}'
FAKE_CORRUPT_GET=ckpt_last.pt bash "$PKG/wmpool_chain.sh" > "$T/b1.out" 2>&1; rc=$?
ok "a checkpoint pulled altered fails the sha256 check against box 2's own (3 attempts): chain stops with rc 7, run NOT banked" '[ $rc -eq 7 ] && stoptxt | grep -q "pull-sha-mismatch" && [ ! -e "$WMPOOL_PULLS/wmpool_P-ATT_s0/BANK.done" ] && [ ! -e "$WMPOOL_PULLS/wmpool_P-ATT_s0/BANK_MANIFEST.json" ]'
ok "after the failed bank NOTHING is cleaned on the box: no BANK_VERIFIED proof, no CLEANUP, the checkpoint is still there" '[ ! -e "$B/D/tanitad/wmpool/runs/P-ATT_s0/CLEANUP.json" ] && ! ls "$B/D/tanitad/wmpool/runs/P-ATT_s0"/BANK_VERIFIED_* > /dev/null 2>&1 && [ -s "$B/D/tanitad/runs/v8f_wmpool/P-ATT_s0/ckpt_last.pt" ]'
ok "the next arm did not start" '[ ! -e "$B/D/tanitad/wmpool/runs/P-ATT_s1/RUN.done" ]'
newbox bankbad2 0; control '{"sps": 1.0}'
FAKE_CORRUPT_GET=CELLS.json bash "$PKG/wmpool_chain.sh" > "$T/b2.out" 2>&1; rc=$?
ok "a CELLS.json pulled altered stops the chain (rc 7) before the pilot" '[ $rc -eq 7 ] && [ ! -e "$B/D/tanitad/wmpool/runs/PILOT_P-ATT_s0" ]'
newbox bankbad3 0; control '{"sps": 1.0}'
FAKE_CORRUPT_GET=gate_token.json bash "$PKG/wmpool_chain.sh" > "$T/b3.out" 2>&1; rc=$?
ok "a pilot file (gate_token.json) pulled altered stops the chain (rc 7) before any run" '[ $rc -eq 7 ] && [ ! -e "$B/D/tanitad/runs/v8f_wmpool" ]'

echo "== 10 commit, tag and package refusals"
newbox nocommit 0; unset WMPOOL_COMMIT
bash "$PKG/wmpool_chain.sh" > "$T/nc.out" 2>&1; rc=$?
ok "chain without WMPOOL_COMMIT refuses (rc 8), zero box calls" '[ $rc -eq 8 ] && stoptxt | grep -q "commit-missing" && [ "$(calls)" -eq 0 ]'
export WMPOOL_COMMIT="abc123"; rm -f "$WMPOOL_STATE/STOP.txt"
bash "$PKG/wmpool_chain.sh" > "$T/nc2.out" 2>&1; rc=$?
ok "an abbreviated commit refuses (rc 8)" '[ $rc -eq 8 ] && [ "$(calls)" -eq 0 ]'
export WMPOOL_COMMIT="$COMMIT"; rm -f "$WMPOOL_STATE/STOP.txt"
WMPOOL_COMMIT="$(printf 'A%.0s' $(seq 1 40))" bash "$PKG/wmpool_chain.sh" > "$T/nc3.out" 2>&1; rc=$?
ok "an upper-case 40-character commit refuses (rc 8)" '[ $rc -eq 8 ]'
rm -f "$WMPOOL_STATE/STOP.txt"
for tg in E-KIN_s0 E-NOW_s1 BAL-G_s0 R35H_s0; do
  bash "$PKG/stages/wmpool_run.sh" $tg > "$T/tok_$tg.out" 2>&1; rc=$?
  ok "a tag of an earlier chain ($tg) refuses (rc 16), zero box calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
done
ok "the checker CLI refuses an unknown tag (rc 2)" '"$PYV" -I "$PKG/wmpool_check.py" argv --tag E-KIN_s0 > /dev/null 2>&1; [ $? -eq 2 ]'
ok "the checker CLI prints the frozen argv of a WMPOOL tag" '"$PYV" -I "$PKG/wmpool_check.py" argv --tag P-ATT_rep_s2 | grep -q -- "--wm-pool att25"'
( unset WMPOOL_DRY; bash "$PKG/wmpool_chain.sh" > "$T/hook1.out" 2>&1 ); rc=$?
ok "test hooks (WMPOOL_TEST_ZCELLS / _CELL_COUNTS / _FAKE_FREE_GB) set in a launch without WMPOOL_DRY=1 are refused (rc 8), zero box calls" '[ $rc -eq 8 ] && stoptxt | grep -q "test-hook-in-real-run" && [ "$(calls)" -eq 0 ]'
rm -f "$WMPOOL_STATE/STOP.txt"

echo "== 11 transport variable unset, relative, absent, the wrong kind, or test mode with a real client (fail closed)"
newbox tr 0
for how in unset relative absent realkind; do
  rm -f "$WMPOOL_STATE/STOP.txt"; : > "$B/transport.log"
  case "$how" in
    unset) ( unset WMPOOL_SSH; bash "$PKG/wmpool_chain.sh" > "$T/tr_$how.out" 2>&1 ); rc=$?;;
    relative) WMPOOL_SSH="ssh" bash "$PKG/wmpool_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
    absent) WMPOOL_SSH="$T/no_such_ssh" bash "$PKG/wmpool_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
    realkind) printf '#!/usr/bin/env bash\necho i am a real client\n' > "$T/fake_real_ssh"; chmod +x "$T/fake_real_ssh"; WMPOOL_SSH="$(cygpath -m "$T/fake_real_ssh")" bash "$PKG/wmpool_chain.sh" > "$T/tr_$how.out" 2>&1; rc=$?;;
  esac
  ok "transport '$how' refuses with rc 16" '[ $rc -eq 16 ]'
  ok "transport '$how' made no call and wrote a stop line" '[ "$(calls)" -eq 0 ] && grep -q "transport-not-fail-closed" "$WMPOOL_STATE/STOP.txt"'
done
rm -f "$WMPOOL_STATE/STOP.txt"; : > "$B/transport.log"
( unset WMPOOL_SFTP; bash "$PKG/wmpool_chain.sh" > "$T/tr_sftp.out" 2>&1 ); rc=$?
ok "WMPOOL_SFTP unset refuses with rc 16" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$WMPOOL_STATE/STOP.txt"
( unset WMPOOL_DRY WMPOOL_TEST_ZCELLS WMPOOL_TEST_CELL_COUNTS WMPOOL_FAKE_FREE_GB; bash "$PKG/wmpool_chain.sh" > "$T/tr_nodry.out" 2>&1 ); rc=$?
ok "a stub without test mode refuses (the stub is never taken for the real client)" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$WMPOOL_STATE/STOP.txt"
printf '#!/usr/bin/env bash\necho i am a real sftp\n' > "$T/fake_real_sftp"; chmod +x "$T/fake_real_sftp"
WMPOOL_SFTP="$(cygpath -m "$T/fake_real_sftp")" bash "$PKG/wmpool_chain.sh" > "$T/tr_realsftp.out" 2>&1; rc=$?
ok "test mode (WMPOOL_DRY=1) with a real sftp path refuses (rc 16), zero calls" '[ $rc -eq 16 ] && [ "$(calls)" -eq 0 ]'
rm -f "$WMPOOL_STATE/STOP.txt"
WMPOOL_SSH="$T/no_such_ssh" bash "$PKG/stages/wmpool_run.sh" P-ATT_s0 > "$T/tr_stage.out" 2>&1; rc=$?
ok "a stage run alone with an absent transport refuses with rc 16 before any call" '[ $rc -eq 16 ]'
WMPOOL_SSH="$T/no_such_ssh" bash "$PKG/stages/wmpool_cells.sh" > "$T/tr_cells.out" 2>&1; rc=$?
ok "the cells stage alone with an absent transport refuses with rc 16 before any call" '[ $rc -eq 16 ]'
ok "no breach through PATH after all transport refusals" '[ ! -e "$T/BREACH" ]'

echo "== 12 hygiene"
ok "no ssh, sftp or scp command in any test script other than the stubs' own forwarding" '! grep -nE "^[^#]*\b(ssh|sftp|scp) +-" "$TESTS/test_wmpool_dryrun.sh" | grep -v "poison" | grep -q .'
ok "bash scripts are LF" '"$PYV" -I -c "import glob,sys;fs=glob.glob(sys.argv[1]+\"/*.sh\")+glob.glob(sys.argv[1]+\"/stages/*.sh\")+glob.glob(sys.argv[2]+\"/*.sh\");assert len(fs)>=8 and all(b\"\r\" not in open(f,\"rb\").read() for f in fs)" "$PKG" "$TESTS"'
rm -rf "${T:?}"
echo "DRYRUN ok=$OK failed=$BAD"
[ "$BAD" -eq 0 ]
