#!/usr/bin/env bash
# F-vs-C READ CHAIN (MM 2026-11-20). Turns the two field seams into the verdict (RUNBOOK_FIELD_READ.md steps 1-6), unattended and resumable.
#   field_read_chain.sh pre-C   everything that needs only C: rows (C + P7 base), families C:base, tactical, strategic, rawdac C (+check, +RED once),
#                               then field_read.py --f C --c C --dry-run "CTRL-ID: C against itself" (must print NOT PROVEN, E1 0.0000 [0.0000, 0.0000]).
#   field_read_chain.sh full    waits (poll 10 min, up to 48 h) for markers_c/chain.DONE AND markers_f/chain.DONE, writes the seam ledger on first sight,
#                               then steps 1-6: rows (F, C one pass), families F:C, tactical, strategic, rawdac F + C, checks, the read -> VERDICT.txt, field_read.json.
# Never trains, never commits/pushes/stages, never touches GPU, pod, box 2, Thor. CPU only, <= 2 CPU processes, commit guard >= 6 before every CPU stage.
# ENV (defaults = the real read):
#   FR_DIR     instrument dir holding field_read.py, field_rawdac.py, RUNBOOK_FIELD_READ.md (default: the v2 hold copy; point at the landed export later)
#   FR_WORK    work dir (markers/, logs/, out/ ...)               default E:/Projects/TanitAD-artifacts/field_verdict
#   FR_DRY=1   DRY RUN on the P5-C stand-ins (F=C1, C=C0): no waiting, every output stamped "DRY RUN - NOT A FIELD VERDICT"
#   FR_TIP     the scoring harness export                          default F:/TanitAD_p5cread_tip (751f3ae7, read-only)
# Terminal states print EXACTLY ONE line: ZZFIELDREAD_PREC_DONE | ZZFIELDREAD_DONE | ZZFIELDREAD_INVALID <control> | ZZFIELDREAD_FAILED <stage> | ZZFIELDREAD_TIMEOUT.
# A failed control is ZZFIELDREAD_INVALID, never a verdict.
set -u
MODE="${1:?mode: pre-C | full}"
case "$MODE" in pre-C|full) ;; *) echo "bad mode $MODE" >&2; exit 2;; esac
HOLD=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_read/readchain
RC=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain
NX=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next
PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
FR_DIR="${FR_DIR:-E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_read/read/land/$PKR/eval/field}"
TIP="${FR_TIP:-F:/TanitAD_p5cread_tip}"; PK="$TIP/$PKR"; EV="$PK/eval"
DRY="${FR_DRY:-0}"
W="${FR_WORK:-E:/Projects/TanitAD-artifacts/field_verdict}"
PY_T=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PY_N=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
NAV="F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
DATA=E:/Projects/TanitAD/data/refe_navtest
FW=E:/Projects/TanitAD-artifacts/field_read
C3="${FR_C3:-$HOLD/tokens_c3.json}"
P7SEAM="$DATA/seams/stage2_p7_full.npz"; P7DUMP="$DATA/seams/stage2_p7_full.proposals.npz"
if [ "$DRY" = 1 ]; then
  FN=C1; CN=C0; SP=p5c; PT="$NX/propscore"; WAITMARK=0; STAMP="DRY RUN - NOT A FIELD VERDICT"
else
  FN=F; CN=C; SP=p7field; PT="$FW/propscore"; WAITMARK=1; STAMP=""
fi
lc() { echo "$1" | tr 'A-Z' 'a-z'; }
seam() { echo "$DATA/seams/stage2_${SP}_$(lc "$1").npz"; }
dump() { echo "$DATA/seams/stage2_${SP}_$(lc "$1").proposals.npz"; }
csv()  { echo "$DATA/score/refe_stage2_${SP}_$(lc "$1")/refe_stage2_${SP}_$(lc "$1").csv"; }
tbl()  { echo "$PT/full_${1}_s*of2.jsonl"; }
M="$W/markers"; L="$W/logs"; O="$W/out"; PL="$W/plans"; PRE="$W/pre"
mkdir -p "$M" "$L" "$O" "$PL" "$PRE" "$W/tmp" "$W/rawdac" 2>/dev/null
[ "$DRY" = 1 ] && printf '%s\n' "$STAMP" > "$W/DRY_RUN_NOT_A_FIELD_VERDICT.txt"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $MODE $*" >> "$W/CHAIN.log"; echo "[$(ts)] $MODE $*"; }
TERMINAL=0
terminal() {   # exactly one ZZFIELDREAD_* line per chain run
  [ "$TERMINAL" = 1 ] && return; TERMINAL=1
  local l="ZZFIELDREAD_$*"; [ "$DRY" = 1 ] && l="$l [$STAMP]"
  echo "$l" >> "$W/CHAIN.log"; echo "$l"; printf '%s\n' "$l" > "$W/TERMINAL_$MODE.txt"
}
fail() { printf 'stage=%s rc=%s time=%s log=%s\n' "$1" "$2" "$(ts)" "$L/$1.log" > "$M/chain.FAILED"; terminal "FAILED $1 rc=$2"; exit 1; }
invalid() { printf 'control=%s time=%s\n' "$1" "$(ts)" > "$M/chain.INVALID"; terminal "INVALID $1"; exit 3; }
trap 'terminal "FAILED signal"; exit 143' INT TERM
if [ -f "$M/chain.FAILED" ]; then n=1; while [ -f "$M/chain.FAILED.attempt_$n" ]; do n=$((n+1)); done; mv "$M/chain.FAILED" "$M/chain.FAILED.attempt_$n"; say "resuming (attempt $n kept)"; fi
commit_ok() { local g; g=$("$PY_T" "$RC/tools/commit_guard.py" | tr -d '\r\n '); case "$g" in ''|*[!0-9]*) return 1;; esac; [ "$g" -ge 6 ]; }
wait_commit() { local i=0; while ! commit_ok; do i=$((i+1)); [ $i -gt 60 ] && return 1; sleep 60; done; return 0; }
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$TIP/stack;$TIP/taniteval;$EV;$EV/p5c"
export TANITAD_REPO="$TIP" REFE_EVAL_DIR="$EV" P5C_REFE_DIR="$PK/refe" TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim REFE_NAVTEST_DIR="$DATA"
export FIELD_READ_P5C_DIR="$EV/p5c" FIELD_READ_DIR="$FR_DIR"
export TMP="$W/tmp" TEMP="$W/tmp" PYTHONIOENCODING=utf-8 PYTHONUTF8=1
run() {   # run STAGE cmd... : marker-skipping, commit-guarded, logged
  local st="$1"; shift
  [ -f "$M/$st.DONE" ] && { say "skip $st (DONE)"; return 0; }
  wait_commit || fail "$st" 90
  say "RUN $st"; local t0; t0=$(date +%s)
  "$@" > "$L/$st.log" 2>&1 < /dev/null; local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  say "  $st rc $rc after $dt s"
  [ $rc -eq 0 ] || fail "$st" $rc
  printf '%s rc=0 secs=%s\n' "$(ts)" "$dt" > "$M/$st.DONE"
}
say "chain start mode=$MODE dry=$DRY work=$W FR_DIR=$FR_DIR tip=$(cut -c1-8 "$TIP/EXPORT_SHA.txt" 2>/dev/null)"
for f in field_read.py field_rawdac.py RUNBOOK_FIELD_READ.md; do [ -f "$FR_DIR/$f" ] || { say "REFUSING: $FR_DIR/$f missing"; fail preflight 2; }; done
[ -d "$EV/p5c" ] && [ -f "$C3" ] || { say "REFUSING: tip export or C3 tokens missing"; fail preflight 2; }
g=$("$PY_T" -c "import shutil;print(int(shutil.disk_usage('E:/').free//(1<<30)))" | tr -d '\r\n '); [ "${g:-0}" -ge 3 ] || { say "REFUSING: E: ${g} GB free (< 3)"; fail preflight 2; }
# instrument fingerprints (recorded in the chain's output)
( cd "$FR_DIR" && "$PY_T" -c "
import hashlib,sys
for f in ('field_read.py','field_rawdac.py','RUNBOOK_FIELD_READ.md'):
    print(hashlib.sha256(open(f,'rb').read()).hexdigest(), f)
" ) > "$W/INSTRUMENTS.sha256"; "$PY_T" -c "
import hashlib,glob
for f in sorted(glob.glob('$EV/p5c/p5c_*.py'))+['$EV/instruments/strategic_family.py','$C3']:
    if 'test_' in f: continue
    print(hashlib.sha256(open(f,'rb').read()).hexdigest(), f)
" >> "$W/INSTRUMENTS.sha256"
say "instrument sha256 -> $W/INSTRUMENTS.sha256"

# ----------------------------------------------------------------------------------------------- stage bodies
rows_glob() { echo "$1/pass1/rows_s*.jsonl"; }
FRL=("$PY_T" "$FR_DIR/field_read.py")
rawdac_score() {   # arm  (2 shards at once = <= 2 CPU processes)
  local N="$1" pids=() s rc=0
  for s in 0 1; do
    PYTHONPATH="$NAV;$PK/refe" "$PY_N" "$FR_DIR/field_rawdac.py" score --arm "$N" --proposals "$(dump "$N")" --out "$W/rawdac/rawdac_${N}_s${s}of2.jsonl" --shard $s/2 --tokens "$C3" \
      > "$L/rawdac_${N}_s$s.log" 2>&1 < /dev/null &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || rc=$?; done
  return $rc
}
rawdac_check() {   # arm
  local N="$1"; PYTHONPATH="$NAV;$PK/refe" "$PY_N" "$FR_DIR/field_rawdac.py" check --rows "$W/rawdac/rawdac_${N}_s*of2.jsonl" --proposals "$(dump "$N")" --table "$(tbl "$N")" --out "$W/rawdac/ctrl_${N}.json"
}
rawdac_red() {
  local N="$1"; PYTHONPATH="$NAV;$PK/refe" "$PY_N" "$FR_DIR/field_rawdac.py" score --arm "$N" --red --limit 150 --proposals "$(dump "$N")" --out "$W/rawdac/rawdac_${N}_RED.jsonl" --tokens "$C3"
}
rawdac_red_check() {
  local N="$1"; PYTHONPATH="$NAV;$PK/refe" "$PY_N" "$FR_DIR/field_rawdac.py" check --rows "$W/rawdac/rawdac_${N}_s*of2.jsonl" --proposals "$(dump "$N")" --table "$(tbl "$N")" --red-rows "$W/rawdac/rawdac_${N}_RED.jsonl" --out "$W/rawdac/ctrl_${N}_with_red.json"
}
rawdac_arm() {   # arm: score, check, (RED once) ; a failed CTRL-SIM/PICK/RED = INVALID
  local N="$1"
  run "rawdac_$N" rawdac_score "$N" || exit 1
  if [ ! -f "$M/rawdac_check_$N.DONE" ]; then
    wait_commit || fail "rawdac_check_$N" 90
    rawdac_check "$N" > "$L/rawdac_check_$N.log" 2>&1 < /dev/null; grep -q "ZZRAWDAC_CHECK PASS" "$L/rawdac_check_$N.log" || invalid "CTRL-RAWDAC-SIM/PICK ($N)"
    printf '%s\n' "$(ts)" > "$M/rawdac_check_$N.DONE"
  fi
  if [ "$N" = "$CN" ]; then
    run "rawdac_red_$N" rawdac_red "$N" || exit 1
    if [ ! -f "$M/rawdac_red_check_$N.DONE" ]; then
      wait_commit || fail "rawdac_red_check_$N" 90
      rawdac_red_check "$N" > "$L/rawdac_red_check_$N.log" 2>&1 < /dev/null
      grep -q "PASS (the control drops)" "$L/rawdac_red_check_$N.log" && grep -q "ZZRAWDAC_CHECK PASS" "$L/rawdac_red_check_$N.log" || invalid "RED-RAWDAC ($N)"
      printf '%s\n' "$(ts)" > "$M/rawdac_red_check_$N.DONE"
    fi
  fi
}
# explicit read-argument builders (arrays, so the PK path with '&' never goes through string substitution)
read_args() {   # FARM CARM ROWSGLOB PLANSDIR FAMCI STRAT TACT LEDGER OUT TEXT [extra...]
  local f="$1" c="$2" rows="$3" plans="$4" famci="$5" strat="$6" tact="$7" ledger="$8" out="$9" txt="${10}"
  RA=(--rows "$rows" --f "$f" --c "$c"
      --seam-f "$(seam "$f")" --dump-f "$(dump "$f")" --seam-c "$(seam "$c")" --dump-c "$(dump "$c")"
      --csv-f "$(csv "$f")" --csv-c "$(csv "$c")" --seam-ledger "$ledger" --convert-report "$plans/convert_report.json"
      --proptable-f "$(tbl "$f")" --proptable-c "$(tbl "$c")"
      --rawdac-f "$W/rawdac/rawdac_${f}_s*of2.jsonl" --rawdac-c "$W/rawdac/rawdac_${c}_s*of2.jsonl" --c3-tokens "$C3"
      --strategic "$strat" --tactical "$tact" --out "$out" --text "$txt")
  [ -n "$famci" ] && RA+=(--fam-ci "$famci")
}
control_failures() {   # json -> names of failed controls (empty when all pass)
  "$PY_T" -c "
import json,sys
j=json.load(open(sys.argv[1],encoding='utf-8'))
print(' '.join(sorted({c.get('control','?') for c in j.get('controls',[]) if not c.get('pass',True)})))" "$1" | tr -d '\r'
}
do_read() {   # stage-name, then the same args as read_args, plus optional extra field_read flags in EXTRA
  local st="$1"; shift
  [ -f "$M/$st.DONE" ] && { say "skip $st (DONE)"; return 0; }
  read_args "$@"; wait_commit || fail "$st" 90
  say "RUN $st"; local t0; t0=$(date +%s)
  "${FRL[@]}" "${RA[@]}" "${EXTRA[@]}" > "$L/$st.log" 2>&1 < /dev/null; local rc=$?
  say "  $st rc $rc after $(( $(date +%s) - t0 )) s"
  [ $rc -eq 0 ] || fail "$st" $rc
  local out="${@: 9:1}"; local bad; bad=$(control_failures "$out"); [ -z "$bad" ] || invalid "$bad"
  printf '%s rc=0\n' "$(ts)" > "$M/$st.DONE"
}
selfread_check() {   # the CTRL-ID read must say NOT PROVEN and E1 0.0000 [0.0000, 0.0000]
  grep -q "NOT PROVEN" "$1" && grep -q "E1 0.0000 \[0.0000, 0.0000\]" "$1"
}
mk_ledger() {   # first sight: sha256 of every seam and dump, in field_read.py's ledger format
  local out="$1"; shift
  [ -f "$out" ] && return 0
  "$PY_T" - "$out" "$@" <<'PYEOF' || return 1
import hashlib, json, sys
out = sys.argv[1]; arms = {}
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()
for spec in sys.argv[2:]:
    n, s, d = spec.split('|')
    arms[n] = {"seam": s, "seam_sha256": sha(s), "dump": d, "dump_sha256": sha(d)}
json.dump({"arms": arms}, open(out, 'w'), indent=1)
PYEOF
}

# ----------------------------------------------------------------------------------------------- mode pre-C
pre_c() {
  [ -s "$(seam "$CN")" ] && [ -s "$(dump "$CN")" ] && [ -s "$(csv "$CN")" ] || { say "REFUSING: arm $CN seam/dump/csv missing"; fail preflight 2; }
  ls $(tbl "$CN") > /dev/null 2>&1 || { say "REFUSING: arm $CN 64-proposal table missing"; fail preflight 2; }
  [ -s "$P7SEAM" ] || { say "REFUSING: P7 base seam $P7SEAM missing"; fail preflight 2; }
  mk_ledger "$PRE/seam_ledger_pre.json" "$CN|$(seam "$CN")|$(dump "$CN")" || fail ledger 1
  run pc_convert "$PY_T" "$EV/p5c/p5c_read_prep.py" convert --arm "$CN=$(seam "$CN"),$(dump "$CN")" "base=$P7SEAM,$P7DUMP" --outdir "$PRE/plans"
  run pc_score "$PY_T" "$EV/p5c/p5c_read_prep.py" score --plans-dir "$PRE/plans" --out-dir "$PRE/pass1" --procs 2 --python "$PY_N"
  run pc_families "$PY_T" "$EV/p5c/p5c_families.py" --rows "$PRE/pass1/rows_s*.jsonl" --arms "$CN" base --pairs "$CN:base" --out-dir "$PRE/families" --python "$PY_N" --repo "$TIP"
  run pc_tactical "$PY_T" "$EV/p5c/p5c_tactical.py" --seam "C1=$(seam "$CN")" "C0=$(seam "$CN")" "base=$P7SEAM" --dump "C1=$(dump "$CN")" "C0=$(dump "$CN")" "base=$P7DUMP" --out "$PRE/tactical.json"
  run pc_strategic "$PY_T" "$EV/instruments/strategic_family.py" read --arm "$CN=$(seam "$CN")" --base "base=$P7SEAM" --out "$PRE/strategic_family.json"
  rawdac_arm "$CN"
  EXTRA=(--fam-labels base "$CN" --dry-run "CTRL-ID: $CN against itself")
  do_read pc_selfread "$CN" "$CN" "$PRE/pass1/rows_s*.jsonl" "$PRE/plans" "$PRE/families/famci_${CN}_vs_base.json" "$PRE/strategic_family.json" "$PRE/tactical.json" \
    "$PRE/seam_ledger_pre.json" "$W/ctrl_id_${CN}_selfread.json" "$W/ctrl_id_${CN}_selfread.txt"
  selfread_check "$L/pc_selfread.log" || invalid "CTRL-ID ($CN self-read is not NOT PROVEN / E1 0.0000)"
  local line; line=$(grep -m1 "^ZZFIELD_READ" "$L/pc_selfread.log" | tr -d '\r')
  printf '%s\n' "$(ts)" > "$M/pre_C.DONE"
  terminal "PREC_DONE CTRL-ID: ${line#ZZFIELD_READ }"
}

# ----------------------------------------------------------------------------------------------- mode full
full() {
  if [ "$WAITMARK" = 1 ]; then
    local i=0
    while [ ! -f "$FW/markers_c/chain.DONE" ] || [ ! -f "$FW/markers_f/chain.DONE" ]; do
      i=$((i+1)); [ $i -gt 288 ] && { terminal "TIMEOUT waited 48 h for both chain.DONE markers"; exit 4; }   # 288 x 10 min
      [ $((i % 6)) = 1 ] && say "waiting for markers (c: $([ -f $FW/markers_c/chain.DONE ] && echo yes || echo no), f: $([ -f $FW/markers_f/chain.DONE ] && echo yes || echo no))"
      sleep 600
    done
  fi
  for a in "$FN" "$CN"; do
    [ -s "$(seam "$a")" ] && [ -s "$(dump "$a")" ] && [ -s "$(csv "$a")" ] || { say "REFUSING: arm $a seam/dump/csv missing"; fail preflight 2; }
    ls $(tbl "$a") > /dev/null 2>&1 || { say "REFUSING: arm $a 64-proposal table missing"; fail preflight 2; }
  done
  [ -s "$P7SEAM" ] && [ -s "$P7DUMP" ] || { say "REFUSING: P7 base seam/dump missing"; fail preflight 2; }
  LEDGER="$O/seam_ledger.json"
  mk_ledger "$LEDGER" "$FN|$(seam "$FN")|$(dump "$FN")" "$CN|$(seam "$CN")|$(dump "$CN")" || fail ledger 1
  say "seam ledger (first sight) $LEDGER"
  run convert "$PY_T" "$EV/p5c/p5c_read_prep.py" convert --arm "$FN=$(seam "$FN"),$(dump "$FN")" "$CN=$(seam "$CN"),$(dump "$CN")" --outdir "$PL"
  run score "$PY_T" "$EV/p5c/p5c_read_prep.py" score --plans-dir "$PL" --out-dir "$O/pass1" --procs 2 --python "$PY_N"
  run families "$PY_T" "$EV/p5c/p5c_families.py" --rows "$O/pass1/rows_s*.jsonl" --arms "$FN" "$CN" --pairs "$FN:$CN" --out-dir "$O/families" --python "$PY_N" --repo "$TIP"
  run tactical "$PY_T" "$EV/p5c/p5c_tactical.py" --seam "C1=$(seam "$FN")" "C0=$(seam "$CN")" "base=$P7SEAM" --dump "C1=$(dump "$FN")" "C0=$(dump "$CN")" "base=$P7DUMP" --out "$O/tactical.json"
  run strategic "$PY_T" "$EV/instruments/strategic_family.py" read --arm "$FN=$(seam "$FN")" --base "$CN=$(seam "$CN")" --out "$O/strategic_family_${FN}_${CN}.json"
  rawdac_arm "$CN"
  rawdac_arm "$FN"
  FAMCI="$O/families/famci_${FN}_vs_${CN}.json"
  # step 6 first (controls before the verdict): CTRL-ID on C, unless pre-C already did it
  if [ ! -f "$M/pre_C.DONE" ]; then
    EXTRA=(--fam-labels "$CN" "$FN" --dry-run "CTRL-ID: $CN against itself")
    do_read selfread "$CN" "$CN" "$O/pass1/rows_s*.jsonl" "$PL" "$FAMCI" "$O/strategic_family_${FN}_${CN}.json" "$O/tactical.json" "$LEDGER" "$W/ctrl_id_${CN}_selfread.json" "$W/ctrl_id_${CN}_selfread.txt"
    selfread_check "$L/selfread.log" || invalid "CTRL-ID ($CN self-read)"
  fi
  # step 5: THE READ (a dry run carries its stamp through --dry-run; a real read has none)
  EXTRA=(--fam-labels "$CN" "$FN"); [ "$DRY" = 1 ] && EXTRA+=(--dry-run "C1 vs C0 as stand-ins for F vs C")
  do_read read "$FN" "$CN" "$O/pass1/rows_s*.jsonl" "$PL" "$FAMCI" "$O/strategic_family_${FN}_${CN}.json" "$O/tactical.json" "$LEDGER" "$W/field_read.json" "$W/VERDICT.txt"
  local line; line=$(grep -m1 "^ZZFIELD_READ" "$L/read.log" | tr -d '\r')
  printf '%s\n%s\n' "$(ts)" "$line" > "$M/chain.DONE"
  terminal "DONE ${line#ZZFIELD_READ }"
}
if [ "$MODE" = pre-C ]; then pre_c; else full; fi
