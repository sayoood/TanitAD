#!/usr/bin/env bash
# MM 2026-11-24: the SCORER-FT arm-P-vs-C read (card PREREG_SCORERFT_PICK.md), run UNATTENDED once P's seam chain and the pod artifacts have landed.
# RUNBOOK_SCORERFT_READ.md Part B (B1 preflight 1-6, B2-B3 plans + ONE transp_score pass + families / tactical / strategic, B4 the read), mirroring sft_read_s1.sh (v2).
# Nothing here chooses a bar or an outcome: scorerft_read.py --card pick does. STAGED, NOT STARTED (P has not trained yet).
#   waits : P's seam chain  $W/markers_sftp/chain.DONE  (fails fast on markers_sftp/chain.FAILED)
#           the pod artifacts  $OP/pod/{config_p,summary,metrics_p,weight_rule_rank_p,w0_p,argv_p,clip_probe}.json* , gated on $OP/pod/PULLED.txt (the MM writes it LAST)
#   steps (marker each, resumable): tip_export -> pins (card/train/C/Stage-1/clip pins read FROM the tip's scorerft_read.py) -> preflight (seam reports 12146/0, P chain-start
#           tip=751f3ae7, F: harness tip) -> pytest_read + pytest_clip (WHOLE files, under the pytest lock dir) -> ctrl3b (ctrl_s0 + ctrl_order on the real C / T logs)
#           -> repro (Stage 1 re-run with the modified tool into $OP/repro, field-by-field equality) -> convert -> score (ONE pass, --procs 2) -> families (P:C; labels checked)
#           -> tactical -> strategic -> read (every flag) -> one ZZSFTREADP_* line in CHAIN.log -> cleanup of this script's OWN scratch (tip export, tmp).
# Tip = git archive of the commit in $W/readp_tip.txt (if present; the MM writes a later commit there once CLIP_PROBE_SHA256 / PICK_CODE_TAG are filled) else b0264acb
#   (stack, taniteval minus results, PK/eval, PK/refe, the PICK card, pod_config_c.json, raw/2026-11-20-p5c-next/{head_auc,s_rerank}.py so that test_scorerft_read.py runs head_auc parity, not skips it) at the SHORT C: path below (MAX_PATH; never E: exFAT, ~18x bloat). The export runs at read time
#   (re-creatable from the mirror; ~190 MB raw) and is deleted at the end of a successful read.
# CPU only. commit_guard >= 6 GB before every step (1200-min give-up). CUDA_VISIBLE_DEVICES=-1, <= 2 CPU processes (score --procs 2).
# PK path contains '&': it is only ever used inside double quotes, never through sed text or ${a//x/$y}.
# usage: sft_read_p.sh        (launch: run_sft_read_p.cmd, detached)        env READP_REHEARSE=1 = DRY rehearsal on Stage 1 stand-ins (see the end of this header)
# REHEARSAL (READP_REHEARSE=1): P := the Stage 1 T arm, no waits, heavy steps (convert / score / families / tactical / strategic) replaced by Stage 1's own outputs, the read gets
#   --dry-run (every output stamped DRY RUN), work dir read_p_rehearsal. It exists to prove that every flag reaches the tool; it is never a verdict.
set -u
W=E:/Projects/TanitAD-artifacts/scorerft_read
REH="${READP_REHEARSE:-0}"
if [ "$REH" = 1 ]; then OP=$W/read_p_rehearsal; M=$W/markers_readp_rehearsal; L=$W/logs_readp_rehearsal; LOG=$W/CHAIN_readp_rehearsal.log
else OP=$W/read_p; M=$W/markers_readp; L=$W/logs_readp; LOG=$W/CHAIN.log; fi
O1=$W/read_out; MP=$W/markers_sftp; MC=$W/markers_sftc; POD=$OP/pod
TIP=C:/Users/Admin/swp/sftreadp_tip; MIRROR=C:/Users/Admin/tanitad-push
TIPSHA=b0264acb3426ef43d40e63a75cd040ee21101d09
[ -s "$W/readp_tip.txt" ] && TIPSHA=$(tr -d '\r\n ' < "$W/readp_tip.txt")
PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PK="$TIP/$PKR"; EV="$PK/eval"
CARD="$PK/raw/2026-11-24-scorerft-pick/PREREG_SCORERFT_PICK.md"
POD_CFG_C="$PK/raw/2026-11-24-scorerft-read/pod_config_c.json"
W0REF=f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4
PY_T=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PY_N=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
RC=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain
LOCKD=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/pytest.lockdir
EVID=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/o3/card/evidence
MET_C=$EVID/metrics_scorerft_c.jsonl; MET_T=$EVID/metrics_scorerft_t.jsonl
DATA=E:/Projects/TanitAD/data/refe_navtest; SE=$DATA/seams; SC=$DATA/score; PS=$W/propscore
# C (reused from Stage 1) and base (C1's own seam)
SCC=$SE/stage2_scorerft_c.npz; DCC=$SE/stage2_scorerft_c.proposals.npz; CCC=$SC/refe_stage2_scorerft_c/refe_stage2_scorerft_c.csv; RPC=$SE/stage2_scorerft_c.report.json
SB=$SE/stage2_p5c_c1.npz; DB=$SE/stage2_p5c_c1.proposals.npz
# P (made by sft_seam_chain.sh SFTP scorerft/p scorerft_p)
SP=$SE/stage2_scorerft_p.npz; DP=$SE/stage2_scorerft_p.proposals.npz; CP=$SC/refe_stage2_scorerft_p/refe_stage2_scorerft_p.csv; RPP=$SE/stage2_scorerft_p.report.json
DETP=$SE/stage2_scorerft_p_rep.npz; TABP="$PS/full_SFTP_s*of2.jsonl"; TABC="$PS/full_SFTC_s*of2.jsonl"
PARM=P; ROWS="$OP/pass1/rows_s*.jsonl"; FAMCI="$OP/families/famci_P_vs_C.json"; TACT="$OP/tactical.json"; STRAT="$OP/strategic_family_P_C.json"; CONVREP="$OP/plans/convert_report.json"
CHAINLOG=$LOG
if [ "$REH" = 1 ]; then   # stand-ins: P := T; the heavy outputs are Stage 1's own
  SP=$SE/stage2_scorerft_t.npz; DP=$SE/stage2_scorerft_t.proposals.npz; CP=$SC/refe_stage2_scorerft_t/refe_stage2_scorerft_t.csv; RPP=$SE/stage2_scorerft_t.report.json
  DETP=$SE/stage2_scorerft_t_rep.npz; TABP="$PS/full_SFTT_s*of2.jsonl"; PARM=T
  ROWS="$O1/pass1/rows_s*.jsonl"; FAMCI="$O1/families/famci_T_vs_C.json"; TACT="$O1/tactical.json"; STRAT="$O1/strategic_family_T_C.json"; CONVREP="$O1/plans/convert_report.json"
  CHAINLOG=$OP/chain_standin.log
fi
mkdir -p "$M" "$L" "$OP" "$POD" "$W/tmp_readp"
if [ "$REH" = 1 ]; then cp "$W/CHAIN.log" "$CHAINLOG"; echo "[rehearsal] SFTP chain start tip=751f3ae7 prefix=scorerft/p name=scorerft_p (synthetic stand-in line)" >> "$CHAINLOG"; fi
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] READP $*" >> "$LOG"; }
HAVE_LOCK=0
release_lock() { if [ "$HAVE_LOCK" = 1 ]; then rmdir "$LOCKD" 2>/dev/null; HAVE_LOCK=0; fi; }
trap 'release_lock' EXIT
trap 'say "signal received"; release_lock; exit 143' INT TERM
fail() { printf 'stage=%s rc=%s time=%s\n' "$1" "$2" "$(ts)" > "$M/read.FAILED"; say "ZZSFTREADP_FAIL $1 (rc $2) log $L/$1.log"; release_lock; exit 1; }
[ -f "$M/read.FAILED" ] && { n=1; while [ -f "$M/read.FAILED.attempt_$n" ]; do n=$((n+1)); done; mv "$M/read.FAILED" "$M/read.FAILED.attempt_$n"; say "resuming (attempt $n kept)"; }
commit_ok() { local g; g=$("$PY_T" "$RC/tools/commit_guard.py" | tr -d '\r\n '); case "$g" in ''|*[!0-9]*) return 1;; esac; [ "$g" -ge 6 ]; }
wait_commit() { local i=0; while ! commit_ok; do i=$((i+1)); [ $i -gt 1200 ] && return 1; sleep 60; done; return 0; }
free_mb() { "$PY_T" -c "import shutil,sys;print(int(shutil.disk_usage(sys.argv[1]).free//(1<<20)))" "$1" | tr -d '\r\n '; }
waitfor() {   # <marker file> <max polls of 300 s> ; gives up early if P's seam chain FAILED
  local f="$1" n="$2" i=0
  while [ ! -f "$f" ]; do
    [ -f "$MP/chain.FAILED" ] && fail "p_seam_chain_failed" 3
    i=$((i+1)); [ $i -gt "$n" ] && fail "timeout_$(basename "$f")" 4
    sleep 300
  done
}
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$TIP/stack;$TIP/taniteval;$EV;$EV/p5c;$EV/field;$EV/scorerft"
export TANITAD_REPO="$TIP" REFE_EVAL_DIR="$EV" P5C_REFE_DIR="$PK/refe" TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim REFE_NAVTEST_DIR="$DATA"
export TMP="$W/tmp_readp" TEMP="$W/tmp_readp" PYTHONIOENCODING=utf-8 PYTHONUTF8=1
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
runl() {   # like run, but holding the pytest lock dir (mkdir lock) around the command; "no tests ran" / any failure = FAIL
  local st="$1"; shift
  [ -f "$M/$st.DONE" ] && { say "skip $st (DONE)"; return 0; }
  wait_commit || fail "$st" 90
  local i=0; until mkdir "$LOCKD" 2>/dev/null; do i=$((i+1)); [ $i -gt 360 ] && fail "${st}_lock_timeout" 91; sleep 60; done; HAVE_LOCK=1
  say "RUN $st (pytest lock held)"; local t0; t0=$(date +%s)
  "$@" > "$L/$st.log" 2>&1 < /dev/null; local rc=$?
  release_lock
  say "  $st rc $rc after $(( $(date +%s) - t0 )) s: $(tail -n 1 "$L/$st.log" | cut -c1-160)"
  [ $rc -eq 0 ] || fail "$st" $rc
  grep -qE "${PASSRE:-[0-9]+ passed}" "$L/$st.log" || fail "${st}_no_tests_ran" 92
  grep -qE "[0-9]+ (failed|error)|Traceback" "$L/$st.log" && fail "${st}_failures" 93
  printf '%s rc=0\n' "$(ts)" > "$M/$st.DONE"
}
say "start rehearse=$REH tipsha=$(echo "$TIPSHA" | cut -c1-8) pid $$"
# ---------------------------------------------------------------- waits (skipped in rehearsal)
if [ "$REH" != 1 ]; then
  [ -f "$MC/chain.DONE" ] || fail c_seam_not_done 2
  waitfor "$MP/chain.DONE" 2016           # 7 days of 5-min polls
  waitfor "$POD/PULLED.txt" 2016          # the MM writes PULLED.txt LAST, after every pod file is in $POD
fi
for f in config_p.json summary.json metrics_p.jsonl weight_rule_rank_p.json w0_p.json argv_p.json clip_probe.json; do
  [ -s "$POD/$f" ] || { say "missing pod artifact $POD/$f"; fail pod_artifacts 2; }
done
# ---------------------------------------------------------------- tip export (idempotent; resolved to the full sha; E: never)
if [ ! -f "$M/tip_export.DONE" ] || [ "$(tr -d '\r\n ' < "$TIP/EXPORT_SHA.txt" 2>/dev/null)" != "$(git -C "$MIRROR" rev-parse --verify "${TIPSHA}^{commit}" 2>/dev/null)" ]; then
  wait_commit || fail tip_export 90
  FULL=$(git -C "$MIRROR" rev-parse --verify "${TIPSHA}^{commit}" 2>/dev/null | tr -d '\r\n ') || true
  [ -n "${FULL:-}" ] || { say "commit $TIPSHA is not in the mirror $MIRROR"; fail tip_export 2; }
  if [ -d "$TIP" ]; then say "removing own stale export $TIP"; rm -rf "C:/Users/Admin/swp/sftreadp_tip"; fi
  mkdir -p "$TIP"
  fm=$(free_mb C:/); [ "${fm:-0}" -ge 700 ] || { say "REFUSING export: C: ${fm} MB free (< 700; the export is ~190 MB raw)"; fail tip_export 3; }
  ( set -o pipefail; cd "$TIP" && git -C "$MIRROR" -c core.autocrlf=false archive "$FULL" stack taniteval "$PKR/eval" "$PKR/refe" \
      "$PKR/raw/2026-11-24-scorerft-pick/PREREG_SCORERFT_PICK.md" "$PKR/raw/2026-11-24-scorerft-read/pod_config_c.json" \
      "$PKR/raw/2026-11-20-p5c-next/head_auc.py" "$PKR/raw/2026-11-20-p5c-next/s_rerank.py" "$PKR/raw/2026-11-12-instruments/fam_ci" | tar -x --exclude=taniteval/results ) > "$L/tip_export.log" 2>&1
  rcs=$?; [ $rcs -eq 0 ] || fail tip_export $rcs
  [ -s "$EV/scorerft/scorerft_read.py" ] && [ -s "$CARD" ] && [ -s "$POD_CFG_C" ] && [ -s "$PK/refe/train.py" ] || fail tip_export_incomplete 4
  printf '%s\n' "$FULL" > "$TIP/EXPORT_SHA.txt"; printf '%s\n' "$(ts)" > "$TIP/EXPORT_DONE.txt"
  say "tip exported: $FULL ($(du -sm "$TIP" | cut -f1) MB on C:, free now $(free_mb C:/) MB)"
  printf '%s rc=0\n' "$(ts)" > "$M/tip_export.DONE"
fi
# ---------------------------------------------------------------- B1 items 1, 2, 5, 6: pins (read from the tip's tool), shas, modules under the tip, seam reports, chain-start tips
if [ ! -f "$M/preflight.DONE" ]; then
  wait_commit || fail preflight 90
  say "RUN preflight"
  export READP_REH="$REH" READP_CARD="$CARD" READP_TRAINPY="$PK/refe/train.py" READP_POD_CFG_C="$POD_CFG_C" READP_POD="$POD" READP_O1="$O1" READP_MET_C="$MET_C" READP_MET_T="$MET_T"
  export READP_SCC="$SCC" READP_DCC="$DCC" READP_CCC="$CCC" READP_RPC="$RPC" READP_RPP="$RPP" READP_CHAINLOG="$CHAINLOG" READP_FTIP=F:/TanitAD_p5cread_tip READP_STANDIN_CHAIN="$W/CHAIN.log"
  "$PY_T" - "$TIP" > "$L/preflight.log" 2>&1 <<'PYEOF'
import hashlib, json, os, re, sys
tip = os.path.normcase(os.path.abspath(sys.argv[1]))
import tanitad, taniteval, p5c_read, p5c_read_prep, p5c_families, p5c_tactical, field_read, scorerft_read as S
for m in (tanitad, taniteval, p5c_read, p5c_read_prep, p5c_families, p5c_tactical, field_read, S):
    f = os.path.normcase(os.path.abspath(m.__file__))
    assert f.startswith(tip), f"module {m.__name__} imported from {f}, not from the tip export {tip}"
print("module paths ok (all under the tip export)")
E = os.environ
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
real = E["READP_REH"] != "1"
fails = []
def need(name, pin, path):
    if pin is None:
        fails.append(f"PIN UNFILLED: {name} is None in scorerft_read.py (a real read refuses)"); print("UNFILLED", name); return
    got = sha(path)
    print(("ok  " if got == pin else "FAIL"), name, got[:12], "pin", pin[:12], path)
    if got != pin:
        fails.append(f"{name}: sha256 {got[:12]} != pin {pin[:12]} ({path})")
# B1 item 1: the card; item 2: every C / Stage 1 input
need("PICK_PREREG_SHA256", S.PICK_PREREG_SHA256, E["READP_CARD"])
need("PICK_TRAIN_SHA256 (the tip's refe/train.py)", S.PICK_TRAIN_SHA256, E["READP_TRAINPY"])
need("C_SEAM_SHA256", S.C_SEAM_SHA256, E["READP_SCC"]); need("C_DUMP_SHA256", S.C_DUMP_SHA256, E["READP_DCC"]); need("C_CSV_SHA256", S.C_CSV_SHA256, E["READP_CCC"])
O1 = E["READP_O1"]
need("C_ARGV_SHA256", S.C_ARGV_SHA256, O1 + "/argv_c.json"); need("C_W0_SHA256", S.C_W0_SHA256, O1 + "/w0_c.json")
need("C_POD_CONFIG_SHA256", S.C_POD_CONFIG_SHA256, E["READP_POD_CFG_C"])
need("C_METRICS_SHA256", S.C_METRICS_SHA256, E["READP_MET_C"]); need("T_METRICS_SHA256", S.T_METRICS_SHA256, E["READP_MET_T"])
need("STAGE1_READ_SHA256", S.STAGE1_READ_SHA256, O1 + "/read_stage1.json"); need("STAGE1_LEDGER_SHA256", S.STAGE1_LEDGER_SHA256, O1 + "/seam_ledger.json")
if real:
    need("CLIP_PROBE_SHA256", S.CLIP_PROBE_SHA256, E["READP_POD"] + "/clip_probe.json")
else:
    print("rehearsal: CLIP_PROBE_SHA256 =", S.CLIP_PROBE_SHA256, "(not required in a dry run)")
print("PICK_CODE_TAG =", S.PICK_CODE_TAG, "(optional pin; the read's CTRL-CODE checks it when set)")
# B1 item 6: both seam reports 12,146 rows and 0 misses
for p in (E["READP_RPC"], E["READP_RPP"]):
    r = json.load(open(p))
    ok = r["rows"] == 12146 and r["tokens_asked"] == 12146 and r["misses"] == 0
    print("ok  " if ok else "FAIL", "seam report", p, r["rows"], r["tokens_asked"], r["misses"])
    if not ok:
        fails.append(f"seam report {p}: rows {r['rows']} tokens_asked {r['tokens_asked']} misses {r['misses']}")
# B1 item 5: the LAST `SFTP chain start tip=` (and SFTC) in the chain log read the harness tip, and the harness export on F: still IS that tip
txt = open(E["READP_CHAINLOG"], encoding="utf-8", errors="replace").read()
for arm in ("SFTC", "SFTP"):
    m = re.findall(rf"{arm} chain start tip=([0-9a-f]{{8}})", txt)
    t = m[-1] if m else None
    print("ok  " if t == S.HARNESS_TIP else "FAIL", f"last {arm} chain start tip =", t, "expected", S.HARNESS_TIP)
    if t != S.HARNESS_TIP:
        fails.append(f"last {arm} chain start tip {t!r} != {S.HARNESS_TIP}")
ft = open(E["READP_FTIP"] + "/EXPORT_SHA.txt").read().strip()
print("ok  " if ft.startswith(S.HARNESS_TIP) else "FAIL", "F: harness export sha", ft[:8])
if not ft.startswith(S.HARNESS_TIP):
    fails.append(f"F:/TanitAD_p5cread_tip is {ft[:8]}, not {S.HARNESS_TIP}")
if fails:
    print("PREFLIGHT FAILS:"); [print(" -", f) for f in fails]; sys.exit(2)
print("PREFLIGHT OK")
PYEOF
  rcp=$?; say "  preflight rc $rcp: $(tail -n 1 "$L/preflight.log" | cut -c1-200)"
  [ $rcp -eq 0 ] || fail preflight $rcp
  printf '%s rc=0\n' "$(ts)" > "$M/preflight.DONE"
fi
# ---------------------------------------------------------------- B1 item 3: WHOLE test files under the pytest lock dir (a "no tests ran" is a FAIL)
runl pytest_read "$PY_T" -m pytest -q -p no:cacheprovider --basetemp "$W/tmp_readp/pt_read" "$EV/scorerft/test_scorerft_read.py"
# test_clip_probe.py is a SCRIPT, not a pytest file (pytest reports "no tests ran", rc 5: MEASURED 2026-11-24): its whole-file form is `python test_clip_probe.py` -> ZZCLIPPROBE_PASS <n>, rc 0
PASSRE="ZZCLIPPROBE_PASS [0-9]+" runl pytest_clip "$PY_T" "$PK/refe/test_clip_probe.py"
# ---------------------------------------------------------------- B1 item 3b: positive control on the real logs (ctrl_s0 at rank_w 1/32 and ctrl_order must both PASS)
cat > "$OP/ctrl3b.py" <<'PYEOF'
import sys
import scorerft_read as S
r1 = S.ctrl_s0(sys.argv[1], sys.argv[2], rank_w=0.03125)
r2 = S.ctrl_order(sys.argv[1], sys.argv[2])
print("CTRL-S0   pass", r1.get("pass"), {k: v for k, v in r1.items() if k not in ("control", "pass")})
print("CTRL-ORDER pass", r2.get("pass"), {k: v for k, v in r2.items() if k not in ("control", "pass")})
sys.exit(0 if (r1.get("pass") is True and r2.get("pass") is True) else 2)
PYEOF
run ctrl3b "$PY_T" "$OP/ctrl3b.py" "$MET_C" "$MET_T"
# ---------------------------------------------------------------- B1 item 4: Stage 1 reproduction with the (modified) tool, into $OP/repro, against a COPY of Stage 1's ledger
mkdir -p "$OP/repro"
[ -f "$OP/repro/seam_ledger.json" ] || cp "$O1/seam_ledger.json" "$OP/repro/seam_ledger.json"
run repro "$PY_T" "$EV/scorerft/scorerft_read.py" --rows "$O1/pass1/rows_s*.jsonl" --t T --c C \
  --seam-t "$SE/stage2_scorerft_t.npz" --dump-t "$SE/stage2_scorerft_t.proposals.npz" --csv-t "$SC/refe_stage2_scorerft_t/refe_stage2_scorerft_t.csv" \
  --proptable-t "$PS/full_SFTT_s*of2.jsonl" --argv-t "$O1/argv_t.json" \
  --seam-c "$SCC" --dump-c "$DCC" --csv-c "$CCC" --proptable-c "$TABC" --argv-c "$O1/argv_c.json" \
  --w0-digest-c "$O1/w0_c.json" --w0-ref "$W0REF" --det-seam-t "$SE/stage2_scorerft_t_rep.npz" \
  --seam-ledger "$OP/repro/seam_ledger.json" --convert-report "$O1/plans/convert_report.json" \
  --fam-ci "$O1/families/famci_T_vs_C.json" --tactical "$O1/tactical.json" --strategic "$O1/strategic_family_T_C.json" \
  --out "$OP/repro/read_stage1.json" --text "$OP/repro/report_stage1.txt" --verdict-line "$OP/repro/VERDICT_stage1.txt"
cat > "$OP/repro_cmp.py" <<'PYEOF'
import json, sys
a, b = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
bad = []
for k in ("verdict", "T_minus_C", "levels", "goal", "mechanism", "families"):
    if a.get(k) != b.get(k):
        bad.append(k)
ca = [(c.get("control"), c.get("pass")) for c in a["controls"]]; cb = [(c.get("control"), c.get("pass")) for c in b["controls"]]
if ca != cb:
    bad.append("controls[*].pass")
print("Stage 1 reproduction vs Stage 1: keys differing =", bad or "none")
print("controls:", cb)
sys.exit(0 if not bad else 2)
PYEOF
run repro_cmp "$PY_T" "$OP/repro_cmp.py" "$O1/read_stage1.json" "$OP/repro/read_stage1.json"
# ---------------------------------------------------------------- B2-B3: plans, ONE transp_score pass (P, C, base), families P:C, tactical, strategic
if [ "$REH" = 1 ]; then say "rehearsal: steps convert / score / families / tactical / strategic replaced by Stage 1's own outputs ($ROWS, $FAMCI, $TACT, $STRAT)"
else
  run convert "$PY_T" "$EV/p5c/p5c_read_prep.py" convert --arm "P=$SP,$DP" "C=$SCC,$DCC" "base=$SB,$DB" --outdir "$OP/plans"
  run score "$PY_T" "$EV/p5c/p5c_read_prep.py" score --plans-dir "$OP/plans" --out-dir "$OP/pass1" --procs 2 --python "$PY_N"
  run families "$PY_T" "$EV/p5c/p5c_families.py" --rows "$OP/pass1/rows_s*.jsonl" --arms P C base --pairs P:C --out-dir "$OP/families" --python "$PY_N" --repo "$TIP"
  cat > "$OP/famci_labels.py" <<'PYEOF'
import json, sys
j = json.load(open(sys.argv[1], encoding="utf-8"))
print("label_a", j.get("label_a"), "label_b", j.get("label_b"))
sys.exit(0 if (j.get("label_a"), j.get("label_b")) == ("C", "P") else 2)   # else the read prints NOT-EVALUATED
PYEOF
  run famci_labels "$PY_T" "$OP/famci_labels.py" "$FAMCI"
  run tactical "$PY_T" "$EV/p5c/p5c_tactical.py" --seam "C1=$SP" "C0=$SCC" "base=$SB" --dump "C1=$DP" "C0=$DCC" "base=$DB" --out "$TACT"
  run strategic "$PY_T" "$EV/instruments/strategic_family.py" read --arm "P=$SP" --base "C=$SCC" --out "$STRAT"
fi
# ---------------------------------------------------------------- B4: the read (every flag)
[ -s "$DETP" ] || fail det_seam_missing 5
DRYARG=()
[ "$REH" = 1 ] && DRYARG=(--dry-run "REHEARSAL on Stage 1 stand-ins: P := T; not a SCORER-FT pick verdict")
run read "$PY_T" "$EV/scorerft/scorerft_read.py" --card pick --rows "$ROWS" --p "$PARM" --c C \
  --seam-p "$SP" --dump-p "$DP" --csv-p "$CP" --proptable-p "$TABP" --argv-p "$POD/argv_p.json" \
  --seam-c "$SCC" --dump-c "$DCC" --csv-c "$CCC" --proptable-c "$TABC" --argv-c "$O1/argv_c.json" \
  --w0-digest-c "$O1/w0_c.json" --w0-digest-p "$POD/w0_p.json" --w0-ref "$W0REF" \
  --det-seam-p "$DETP" --seam-ledger "$OP/seam_ledger.json" --convert-report "$CONVREP" \
  --metrics-c "$MET_C" --metrics-p "$POD/metrics_p.jsonl" --pod-config-c "$POD_CFG_C" --pod-config-p "$POD/config_p.json" \
  --seam-report-c "$RPC" --seam-report-p "$RPP" --seam-chain-log "$CHAINLOG" \
  --weight-rule-p "$POD/weight_rule_rank_p.json" --clip-probe "$POD/clip_probe.json" --stage1-read "$O1/read_stage1.json" --stage1-ledger "$O1/seam_ledger.json" \
  --fam-ci "$FAMCI" --tactical "$TACT" --strategic "$STRAT" \
  --out "$OP/read_p_stage1.json" --text "$OP/report_p_stage1.txt" --verdict-line "$OP/VERDICT_p_stage1.txt" "${DRYARG[@]}"
[ -s "$OP/VERDICT_p_stage1.txt" ] || fail verdict_line_missing 6
# MM 2026-11-25: the C1r card (PREREG_C1R_READ.md 4(b)) commits F_REFe printed beside P's result, DESCRIPTIVE, bars unchanged. Non-blocking: the verdict above is the read.
if "$PY_T" "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorerft_read/fref_qualifier.py" "$MIRROR" "$TIPSHA" "$OP/read_p_stage1.json" "$OP/F_REFE_QUALIFIER.txt" > "$L/fref_qual.log" 2>&1 < /dev/null; then
  say "ZZSFTREADP_FREF $(head -n 1 "$OP/F_REFE_QUALIFIER.txt" | cut -c1-400)"
else say "ZZSFTREADP_FREF_NOT_PRINTED (see $L/fref_qual.log)"; fi
say "ZZSFTREADP_DONE $(tr -d '\r\n' < "$OP/VERDICT_p_stage1.txt" | cut -c1-400)"
printf '%s\n' "$(ts)" > "$M/read.DONE"
# ---------------------------------------------------------------- cleanup of THIS script's own scratch (re-creatable: the tip from the mirror commit, tmp = pytest basetemps); logged (path, size)
if [ -d "$TIP" ]; then s=$(du -sm "$TIP" | cut -f1); rm -rf "C:/Users/Admin/swp/sftreadp_tip"; say "deleted C:/Users/Admin/swp/sftreadp_tip (${s} MB; re-creatable by git archive of $(echo "$TIPSHA" | cut -c1-8))"; rm -f "$M/tip_export.DONE" "$M/preflight.DONE"; fi
if [ -d "$W/tmp_readp" ]; then s=$(du -sm "$W/tmp_readp" | cut -f1); rm -rf "E:/Projects/TanitAD-artifacts/scorerft_read/tmp_readp"; say "deleted E:/Projects/TanitAD-artifacts/scorerft_read/tmp_readp (${s} MB)"; fi
