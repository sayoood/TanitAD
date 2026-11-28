#!/usr/bin/env bash
# MM 2026-11-23: the SCORER-FT Stage 1 read (T vs C), run UNATTENDED once T's seam lands. RUNBOOK_SCORERFT_READ.md sections 1-5, card
# PK/raw/2026-11-20-scorer-ft/PREREG_SCORER_FT.md (sha256 adc1b2ae...). Nothing here chooses a bar or an outcome: scorerft_read.py does.
#   wait C's chain.DONE (now) and T's proptable.DONE -> preflight (card hash, both seam reports 12,146 rows / 0 misses, module paths under the tip)
#   -> section 2: convert (T, C, base = C1's own seam) + ONE transp_score pass -> section 3: families T:C, tactical, strategic
#      (these run while T's CTRL-DET full-seam re-run is still on the GPU: --procs 1 then, so the box carries at most 2 heavy processes;
#       the shard count changes only which process scores a token, and the read's CTRL-T re-checks every token against the harness CSV)
#   -> wait T's chain.DONE (CTRL-DET's rep seam) -> section 5: scorerft_read.py with every control flag -> one ZZSFTREAD_* line in CHAIN.log.
# Resumable: one marker per step in $M. CPU only. commit_guard >= 6 GB before every step (as sft_seam_chain.sh v3, 240-min give-up).
# Tip = C:/Users/Admin/swp/sftread_tip = git archive of 429086ac (stack, taniteval minus results, PK/eval, PK/refe, + the card); its eval/ is
# byte-equal to the tree whose test_scorerft_read.py read 40/40 on 2026-11-23 (diff -rq 0 lines).
# usage: sft_read_s1.sh   (launch: run_sft_read_s1.cmd, detached)
set -u
W=E:/Projects/TanitAD-artifacts/scorerft_read; O=$W/read_out; MT=$W/markers_sftt; MC=$W/markers_sftc; M=$W/markers_read1; L=$W/logs_read1
TIP=C:/Users/Admin/swp/sftread_tip
PK="$TIP/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"; EV="$PK/eval"
CARD="$PK/raw/2026-11-20-scorer-ft/PREREG_SCORER_FT.md"; CARD_SHA=adc1b2ae8d3cc9b3dfed00a20a3c39197428cda8492be34af3bb9ed24010ec91
W0REF=f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4
PY_T=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PY_N=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
RC=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain
DATA=E:/Projects/TanitAD/data/refe_navtest; SE=$DATA/seams; SC=$DATA/score; PS=$W/propscore
ST=$SE/stage2_scorerft_t.npz; DT=$SE/stage2_scorerft_t.proposals.npz; CT=$SC/refe_stage2_scorerft_t/refe_stage2_scorerft_t.csv
SCC=$SE/stage2_scorerft_c.npz; DCC=$SE/stage2_scorerft_c.proposals.npz; CCC=$SC/refe_stage2_scorerft_c/refe_stage2_scorerft_c.csv
SB=$SE/stage2_p5c_c1.npz; DB=$SE/stage2_p5c_c1.proposals.npz
DET=$SE/stage2_scorerft_t_rep.npz
mkdir -p "$M" "$L" "$O" "$W/tmp_read"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] READ1 $*" >> "$W/CHAIN.log"; }
fail() { printf 'stage=%s rc=%s time=%s\n' "$1" "$2" "$(ts)" > "$M/read.FAILED"; say "ZZSFTREAD_FAIL $1 (rc $2) log $L/$1.log"; exit 1; }
[ -f "$M/read.FAILED" ] && { n=1; while [ -f "$M/read.FAILED.attempt_$n" ]; do n=$((n+1)); done; mv "$M/read.FAILED" "$M/read.FAILED.attempt_$n"; say "resuming (attempt $n kept)"; }
commit_ok() { local g; g=$("$PY_T" "$RC/tools/commit_guard.py" | tr -d '\r\n '); case "$g" in ''|*[!0-9]*) return 1;; esac; [ "$g" -ge 6 ]; }
wait_commit() { local i=0; while ! commit_ok; do i=$((i+1)); [ $i -gt 240 ] && return 1; sleep 60; done; return 0; }
waitfor() {   # <marker file> <max polls of 300 s>; gives up early if T's seam chain FAILED
  local f="$1" n="$2" i=0
  while [ ! -f "$f" ]; do
    [ -f "$MT/chain.FAILED" ] && fail "t_seam_chain_failed" 3
    i=$((i+1)); [ $i -gt "$n" ] && fail "timeout_$(basename "$f")" 4
    sleep 300
  done
}
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$TIP/stack;$TIP/taniteval;$EV;$EV/p5c;$EV/field;$EV/scorerft"
export TANITAD_REPO="$TIP" REFE_EVAL_DIR="$EV" P5C_REFE_DIR="$PK/refe" TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim REFE_NAVTEST_DIR="$DATA"
export TMP="$W/tmp_read" TEMP="$W/tmp_read" PYTHONIOENCODING=utf-8 PYTHONUTF8=1
run() {
  local st="$1"; shift
  [ -f "$M/$st.DONE" ] && { say "skip $st (DONE)"; return 0; }
  wait_commit || fail "$st" 90
  say "RUN $st"; local t0; t0=$(date +%s)
  "$@" > "$L/$st.log" 2>&1 < /dev/null; local rc=$?
  say "  $st rc $rc after $(( $(date +%s) - t0 )) s"
  [ $rc -eq 0 ] || fail "$st" $rc
  printf '%s rc=0\n' "$(ts)" > "$M/$st.DONE"
}
say "start tip=$(cut -c1-8 "$TIP/EXPORT_SHA.txt") pid $$"
[ -f "$MC/chain.DONE" ] || fail c_seam_not_done 2
waitfor "$MT/proptable.DONE" 432
# ---------------------------------------------------------------- section 1: preflight
[ "$(sha256sum "$CARD" | cut -d' ' -f1)" = "$CARD_SHA" ] || fail card_hash 2
for f in "$ST" "$DT" "$CT" "$SCC" "$DCC" "$CCC" "$SB" "$DB" "$O/argv_t.json" "$O/argv_c.json" "$O/w0_c.json"; do [ -s "$f" ] || { say "missing input $f"; fail inputs 2; }; done
"$PY_T" - "$SE/stage2_scorerft_t.report.json" "$SE/stage2_scorerft_c.report.json" > "$L/preflight.log" 2>&1 <<'PYEOF' || fail preflight_reports 2
import json, sys
for p in sys.argv[1:]:
    r = json.load(open(p))
    assert r["rows"] == 12146 and r["tokens_asked"] == 12146 and r["misses"] == 0, (p, r["rows"], r["misses"])
    print("ok", p, r["rows"], r["misses"])
PYEOF
"$PY_T" - "$TIP" >> "$L/preflight.log" 2>&1 <<'PYEOF' || fail preflight_modules 2
import os, sys
tip = os.path.normcase(os.path.abspath(sys.argv[1]))
import tanitad, taniteval, p5c_read, p5c_read_prep, p5c_families, p5c_tactical, field_read, scorerft_read
for m in (tanitad, taniteval, p5c_read, p5c_read_prep, p5c_families, p5c_tactical, field_read, scorerft_read):
    f = os.path.normcase(os.path.abspath(m.__file__))
    assert f.startswith(tip), f"module {m.__name__} imported from {f}, not from the tip export {tip}"
print("module paths ok (all under the tip export)")
PYEOF
say "preflight ok: card adc1b2ae, seams T/C 12146 rows 0 misses, modules under the tip"
# ---------------------------------------------------------------- section 2: plans + ONE transp_score pass (T, C, base = C1's own seam, + human)
run convert "$PY_T" "$EV/p5c/p5c_read_prep.py" convert --arm "T=$ST,$DT" "C=$SCC,$DCC" "base=$SB,$DB" --outdir "$O/plans"
NP=2; [ -f "$MT/chain.DONE" ] || NP=1
[ -f "$M/score.DONE" ] || say "score pass with --procs $NP (T's CTRL-DET $( [ "$NP" = 1 ] && echo 'still running' || echo done))"
run score "$PY_T" "$EV/p5c/p5c_read_prep.py" score --plans-dir "$O/plans" --out-dir "$O/pass1" --procs "$NP" --python "$PY_N"
# ---------------------------------------------------------------- section 3: families, tactical, strategic
run families "$PY_T" "$EV/p5c/p5c_families.py" --rows "$O/pass1/rows_s*.jsonl" --arms T C base --pairs T:C --out-dir "$O/families" --python "$PY_N" --repo "$TIP"
run tactical "$PY_T" "$EV/p5c/p5c_tactical.py" --seam "C1=$ST" "C0=$SCC" "base=$SB" --dump "C1=$DT" "C0=$DCC" "base=$DB" --out "$O/tactical.json"
run strategic "$PY_T" "$EV/instruments/strategic_family.py" read --arm "T=$ST" --base "C=$SCC" --out "$O/strategic_family_T_C.json"
say "sections 2-3 done; waiting for T's CTRL-DET (chain.DONE)"
# ---------------------------------------------------------------- section 5: the read (needs CTRL-DET's rep seam)
waitfor "$MT/chain.DONE" 216
[ -s "$DET" ] || fail det_seam_missing 5
run read "$PY_T" "$EV/scorerft/scorerft_read.py" --rows "$O/pass1/rows_s*.jsonl" --t T --c C \
  --seam-t "$ST" --dump-t "$DT" --csv-t "$CT" --proptable-t "$PS/full_SFTT_s*of2.jsonl" --argv-t "$O/argv_t.json" \
  --seam-c "$SCC" --dump-c "$DCC" --csv-c "$CCC" --proptable-c "$PS/full_SFTC_s*of2.jsonl" --argv-c "$O/argv_c.json" \
  --w0-digest-c "$O/w0_c.json" --w0-ref "$W0REF" --det-seam-t "$DET" \
  --seam-ledger "$O/seam_ledger.json" --convert-report "$O/plans/convert_report.json" \
  --fam-ci "$O/families/famci_T_vs_C.json" --tactical "$O/tactical.json" --strategic "$O/strategic_family_T_C.json" \
  --out "$O/read_stage1.json" --text "$O/report_stage1.txt" --verdict-line "$O/VERDICT_stage1.txt"
say "ZZSFTREAD_DONE $(tr -d '\r\n' < "$O/VERDICT_stage1.txt" | cut -c1-400)"
printf '%s\n' "$(ts)" > "$M/read.DONE"
