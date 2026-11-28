#!/usr/bin/env bash
# TRK-SCORE full navtest run (PREREG_TRKSCORE.md section 8.2, card sha256 5cf2c323...). Arms C1-TRK then C0-TRK, sequential.
# Per arm: hf_pull (sha-verified, re-creatable copy) -> seam (trk_run_seam.py: GPU lock inside, rc 0 or the known rc 1) -> ctrl (trk_ctrl.py,
# no PDMS) -> score (eval/stage2_sft.py: the seam exists, so it ONLY scores with the pinned harness) -> delete the pulled model (logged).
# Gates: START only after the SCORER-FT P read's read.DONE; before the seam and before every score attempt: available RAM >= 4,500 MB on
# 5 consecutive 1-min samples AND commit free >= 6 GB. Resumable: markers in $W/markers/. Log: $W/CHAIN.log (plain >> appends, no tee).
# Launch detached: run_trk_chain.cmd via WMI (CreateFlags 0x08000200, stdin NUL).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
TIP=F:/TanitAD_p5cread_tip
PK="$TIP/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"; EV="$PK/eval"
PY_T=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PY_N=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
NAV=F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896
RC=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain
DATA=E:/Projects/TanitAD/data/refe_navtest; YAW=E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtest.json.gz
TOK=$DATA/navtest_full_tokens.json
GATE=E:/Projects/TanitAD-artifacts/scorerft_read/markers_readp/read.DONE
W=E:/Projects/TanitAD-artifacts/trkscore/work; M=$W/markers; L=$W/logs; ARMS=E:/Projects/TanitAD-artifacts/trkscore/arms
mkdir -p "$M" "$L" "$W/tmp" "$ARMS"
export TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim REFE_NAVTEST_DIR="$DATA" TANITAD_REPO="$TIP" REFE_EVAL_DIR="$EV" P5C_REFE_DIR="$PK/refe"
export TMP="$W/tmp" TEMP="$W/tmp" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$TIP/stack;$TIP/taniteval;$EV;$EV/p5c"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] TRK $*" >> "$W/CHAIN.log"; }
fail() { printf 'stage=%s rc=%s time=%s\n' "$1" "$2" "$(ts)" > "$M/chain.FAILED"; say "FAILED $1 (rc $2)"; exit 1; }
[ -f "$M/chain.FAILED" ] && { n=1; while [ -f "$M/chain.FAILED.attempt_$n" ]; do n=$((n+1)); done; mv "$M/chain.FAILED" "$M/chain.FAILED.attempt_$n"; say "resuming (attempt $n kept)"; }
avail() { powershell -NoProfile -Command "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes" 2>/dev/null | tr -d '\r\n '; }
commit_gb() { "$PY_T" "$RC/tools/commit_guard.py" 2>/dev/null | tr -d '\r\n '; }
memgate() {   # $1 tag; give-up 24 h
  local ok=0 i a g
  for i in $(seq 1 1440); do
    a=$(avail); case "$a" in ''|*[!0-9]*) a=0;; esac
    g=$(commit_gb); case "$g" in ''|*[!0-9]*) g=0;; esac
    if [ "$a" -ge 4500 ] && [ "$g" -ge 6 ]; then ok=$((ok+1)); else ok=0; fi
    [ $((i % 30)) = 1 ] && say "memgate $1: available ${a} MB, commit free ${g} GB (ok ${ok}/5)"
    [ $ok -ge 5 ] && { say "memgate $1 open (${a} MB, ${g} GB)"; return 0; }
    sleep 60
  done
  return 1
}
say "chain start pid $$; card $(sha256sum "$HERE/PREREG_TRKSCORE.md" | cut -c1-16); tip $(cut -c1-8 "$TIP/EXPORT_SHA.txt" 2>/dev/null)"
# ---- the start gate: the other read must be finished (it holds the CPU)
if [ ! -f "$M/gate.DONE" ]; then
  i=0; while [ ! -f "$GATE" ]; do i=$((i+1)); [ $((i % 60)) = 1 ] && say "waiting for $GATE"; [ $i -gt 2880 ] && fail gate 9; sleep 60; done
  say "gate open: $GATE exists"; printf '%s\n' "$(ts)" > "$M/gate.DONE"
fi
run_arm() {   # $1 ARM (C1|C0)  $2 hf prefix  $3 sha256  $4 banked seam name
  local A=$1 PFX=$2 SHA=$3 BANK=$4 a; a=$(echo "$A" | tr 'A-Z' 'a-z')
  local NAME=trk_$a AW=$W/trk_$a CK=$ARMS/$A/model_final.pt
  mkdir -p "$AW"
  [ -f "$M/$A.score.DONE" ] && { say "skip $A (scored)"; return 0; }
  if [ ! -f "$M/$A.seam.DONE" ]; then
    if [ ! -f "$CK" ]; then
      "$PY_T" "$RC/tools/hf_pull.py" --arm "$A" --prefix "$PFX" --out "$ARMS/$A" --repo Sayood/tanitad-refe-p5c > "$L/hf_pull_$A.log" 2>&1 < /dev/null || fail "hf_pull_$A" $?
      grep -q "ZZP5C_HF_PULL $A OK" "$L/hf_pull_$A.log" || fail "hf_pull_verify_$A" 4
      say "pulled $A: $(grep -o 'ZZP5C_HF_PULL.*' "$L/hf_pull_$A.log" | cut -c1-120)"
    fi
    # a half-written seam from a killed attempt is moved aside (never deleted: the PI deletes)
    for f in "$DATA/seams/stage2_$NAME.npz" "$DATA/seams/stage2_$NAME.proposals.npz" "$DATA/seams/stage2_$NAME.report.json" "$AW/$NAME.npz"; do
      [ -f "$f" ] && { mv "$f" "$f.aside_$(date -u +%Y%m%dT%H%M%S)"; say "moved aside $f"; }
    done
    memgate "seam_$A" || fail "memgate_seam_$A" 8
    say "RUN seam $A"; t0=$(date +%s)
    env -u CUDA_VISIBLE_DEVICES "$PY_T" "$HERE/trk_run_seam.py" --ckpt "$CK" --expect-sha256 "$SHA" --name "$NAME" --tokens "$TOK" --yaw-table "$YAW" \
      --seam-out "$DATA/seams/stage2_$NAME.npz" --props-out "$DATA/seams/stage2_$NAME.proposals.npz" --dump "$AW/$NAME.npz" > "$L/seam_$A.log" 2>&1 < /dev/null
    rc=$?; say "  seam $A rc $rc after $(( $(date +%s) - t0 )) s; $(grep -o 'ZZTRK .*' "$AW/$NAME.seam.log" | tail -1 | cut -c1-200)"
    [ "$rc" = 0 ] || [ "$rc" = 1 ] || fail "seam_$A" "$rc"
    [ -f "$AW/$NAME.npz" ] || fail "trkdump_$A" 5
    printf '%s rc=%s\n' "$(ts)" "$rc" > "$M/$A.seam.DONE"
  fi
  if [ ! -f "$M/$A.ctrl.DONE" ]; then
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="$NAV;$PK/refe;$EV/instruments" "$PY_N" "$HERE/trk_ctrl.py" --trk-dump "$AW/$NAME.npz" --trk-seam "$DATA/seams/stage2_$NAME.npz" \
      --trk-props "$DATA/seams/stage2_$NAME.proposals.npz" --banked-seam "$DATA/seams/stage2_$BANK.npz" --banked-props "$DATA/seams/stage2_$BANK.proposals.npz" \
      --expect-n 12146 --out "$AW/ctrl_$A.json" > "$L/ctrl_$A.log" 2>&1 < /dev/null
    rc=$?; say "  ctrl $A rc $rc: $(grep -o 'ZZTRK_CTRL.*' "$L/ctrl_$A.log" | cut -c1-200)"
    # a FAILED control makes the read INVALID (card section 6) but does not stop the scoring: the numbers are then descriptive
    printf '%s rc=%s\n' "$(ts)" "$rc" > "$M/$A.ctrl.DONE"
  fi
  local att
  for att in 1 2 3; do
    memgate "score_${A}_$att" || fail "memgate_score_$A" 8
    say "RUN score $A attempt $att"; t0=$(date +%s)
    "$PY_T" "$EV/stage2_sft.py" --ckpt "$CK" --name "$NAME" --rule navsim_v1 --tokens "$TOK" --dump-proposals --ego-yaw-table "$YAW" > "$L/score_${A}_$att.log" 2>&1 < /dev/null
    rc=$?; say "  score $A rc $rc after $(( $(date +%s) - t0 )) s"
    CSV="$DATA/score/refe_stage2_$NAME/refe_stage2_$NAME.csv"
    if [ "$rc" = 0 ] && [ -f "$CSV" ]; then
      printf '%s rc=0 csv_lines=%s\n' "$(ts)" "$(wc -l < "$CSV")" > "$M/$A.score.DONE"
      say "  scored $A: $(wc -l < "$CSV") CSV lines"
      # the pulled model is a re-creatable copy of the HF file (sha-verified at pull and by CTRL-SHA): delete it, logged
      sz=$(stat -c %s "$CK" 2>/dev/null); rm -f "${ARMS:?}/${A:?}/model_final.pt" && say "deleted ${ARMS}/${A}/model_final.pt ($sz B; HF Sayood/tanitad-refe-p5c $PFX/model_final.pt, sha256 $SHA)"
      return 0
    fi
    sleep 300
  done
  fail "score_$A" 7
}
run_arm C1 p5c/c1 f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4 p5c_c1
run_arm C0 p5c/c0 58fe4aa309ceeb415de9716416a375cdd218eb452f51a5854703bb309ac88ea2 p5c_c0
printf '%s\n' "$(ts)" > "$M/chain.DONE"
say "chain DONE"
