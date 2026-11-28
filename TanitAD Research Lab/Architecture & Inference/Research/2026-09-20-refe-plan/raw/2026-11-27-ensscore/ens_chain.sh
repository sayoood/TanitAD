#!/usr/bin/env bash
# ENS-SCORE chain (PREREG_ENSSCORE.md section 9, card sha256 7423cf9d...). Smoke (controls only, twice) -> ENS1 (primary C1, member C1r) -> ENS1r
# (primary C1r, member C1), sequential. Per arm: seam (ens_run_seam.py: CTRL-SHA, GPU lock inside, rc 0 or the known rc 1) -> ctrl (ens_ctrl.py,
# no PDMS) -> score (eval/stage2_sft.py: the seam exists, so it ONLY scores with the pinned harness). A failed SMOKE control stops the chain
# before the full run; a failed full-run control does not stop scoring (the read is then INVALID and its numbers descriptive).
# Gates: START only after the TRK chain's chain.DONE (TRK holds the GPU and RAM first; nothing of TRK is read or touched); before every seam
# and every score attempt: available RAM >= 4,500 MB on 5 consecutive 1-min samples AND commit free >= 6 GB. Resumable: markers in $M.
# Log: $W/CHAIN.log (plain >> appends). Launch detached: run_ens_chain.cmd via WMI (CreateFlags 0x08000200, stdin NUL).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
TIP=F:/TanitAD_p5cread_tip
PK="$TIP/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"; EV="$PK/eval"
PY_T=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PY_N=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
NAV=F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896
RC=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain
DATA=E:/Projects/TanitAD/data/refe_navtest; YAW=E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtest.json.gz
TOK=$DATA/navtest_full_tokens.json
SMOKE_TOK=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore/smoke_tokens50.json
GATE=E:/Projects/TanitAD-artifacts/trkscore/work/markers/chain.DONE
W=E:/Projects/TanitAD-artifacts/ensscore/work; M=$W/markers; L=$W/logs; ARMS=E:/Projects/TanitAD-artifacts/ensscore/arms
SHA_C1=f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4; SHA_C1R=6e39a57c9a2e3ead0fef2b0d00d1684a708c901b6602ffe0717444f6180654a8
CK_C1=$ARMS/C1/model_final.pt; CK_C1R=$ARMS/C1r/model_final.pt
mkdir -p "$M" "$L" "$W/tmp" "$W/smoke" "$ARMS"
export TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim REFE_NAVTEST_DIR="$DATA" TANITAD_REPO="$TIP" REFE_EVAL_DIR="$EV" P5C_REFE_DIR="$PK/refe"
export TMP="$W/tmp" TEMP="$W/tmp" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$TIP/stack;$TIP/taniteval;$EV;$EV/p5c"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] ENS $*" >> "$W/CHAIN.log"; }
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
say "chain start pid $$; card $(sha256sum "$HERE/PREREG_ENSSCORE.md" | cut -c1-16); tip $(cut -c1-8 "$TIP/EXPORT_SHA.txt" 2>/dev/null)"
# ---- CTRL-CODE at start: the staged files are the frozen ones
( cd "$HERE" && sha256sum -c --quiet CODE.sha256 ) >> "$W/CHAIN.log" 2>&1 || fail code_sha 6
# ---- the start gate: the TRK chain must be finished (it holds the GPU and RAM); TRK files are only tested for existence
if [ ! -f "$M/gate.DONE" ]; then
  i=0; while [ ! -f "$GATE" ]; do i=$((i+1)); [ $((i % 60)) = 1 ] && say "waiting for $GATE$( [ -f "${GATE%/*}/chain.FAILED" ] && echo ' (TRK chain.FAILED present: waiting for its resume)')"
    [ $i -gt 5760 ] && fail gate 9; sleep 60; done
  say "gate open: $GATE exists"; printf '%s\n' "$(ts)" > "$M/gate.DONE"
fi
# ---- the unit tests (CPU) must pass before the smoke (card section 6)
if [ ! -f "$M/tests.DONE" ]; then
  bash "$HERE/run_tests.sh" > "$L/tests.log" 2>&1 < /dev/null; rc=$?
  say "unit tests rc $rc: $(grep -o 'ZZENS_TESTS.*' "$L/tests.log" | tail -1)"
  [ "$rc" = 0 ] || fail tests "$rc"
  printf '%s\n' "$(ts)" > "$M/tests.DONE"
fi
# ---- the checkpoints: re-creatable, sha-verified copies of the HF files
pull() {   # $1 ARM  $2 hf path  $3 sha  $4 out dir
  "$PY_T" "$HERE/ens_pull.py" --hf-path "$2" --expect-sha256 "$3" --out "$4" > "$L/hf_pull_$1.log" 2>&1 < /dev/null || fail "hf_pull_$1" $?
  grep -q "ZZENS_HF_PULL $2 OK" "$L/hf_pull_$1.log" || fail "hf_pull_verify_$1" 4
  say "checkpoint $1: $(grep -o 'ZZENS_HF_PULL.*' "$L/hf_pull_$1.log" | cut -c1-160)"
}
if [ ! -f "$M/ckpt_deleted.DONE" ]; then
  pull C1 p5c/c1/model_final.pt "$SHA_C1" "$ARMS/C1"
  pull C1r p5c/c1r/model_final.pt "$SHA_C1R" "$ARMS/C1r"
fi
aside() { for f in "$@"; do [ -f "$f" ] && { mv "$f" "$f.aside_$(date -u +%Y%m%dT%H%M%S)"; say "moved aside $f"; }; done; }
seam() {   # $1 name  $2 primary ckpt  $3 primary sha  $4 member ckpt  $5 member sha  $6 tokens  $7 seam  $8 props  $9 dump
  env -u CUDA_VISIBLE_DEVICES "$PY_T" "$HERE/ens_run_seam.py" --ckpt "$2" --expect-sha256 "$3" --member-ckpt "$4" --member-sha256 "$5" --name "$1" \
    --tokens "$6" --yaw-table "$YAW" --seam-out "$7" --props-out "$8" --dump "$9" > "$L/seam_$1.log" 2>&1 < /dev/null
}
ctrl() {   # $1 tag  $2 dump  $3 seam  $4 props  $5 banked A  $6 banked B  $7 out  [$8 expect-n]  [$9 det dump]
  local extra=()
  [ -n "${8:-}" ] && extra+=(--expect-n "$8")
  [ -n "${9:-}" ] && extra+=(--det-dump "$9")
  CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="$NAV;$PK/refe;$EV/instruments" "$PY_N" "$HERE/ens_ctrl.py" --ens-dump "$2" --ens-seam "$3" --ens-props "$4" \
    --banked-seam "$DATA/seams/stage2_$5.npz" --banked-props "$DATA/seams/stage2_$5.proposals.npz" \
    --banked-seam-b "$DATA/seams/stage2_$6.npz" --banked-props-b "$DATA/seams/stage2_$6.proposals.npz" --out "$7" "${extra[@]}" > "$L/ctrl_$1.log" 2>&1 < /dev/null
}
# ---- the smoke: 50 tokens, ENS1, twice; controls only (NO PDMS); any FAIL stops the chain here
S=$W/smoke
if [ ! -f "$M/smoke.DONE" ]; then
  for r in 1 2; do
    if [ ! -f "$M/smoke_run$r.DONE" ]; then
      mkdir -p "$S/run$r"
      aside "$S/run$r/stage2_ens_smoke.npz" "$S/run$r/stage2_ens_smoke.proposals.npz" "$S/run$r/stage2_ens_smoke.report.json" "$S/run$r/ens_smoke.npz"
      memgate "smoke_run$r" || fail "memgate_smoke_$r" 8
      say "RUN smoke run $r"; t0=$(date +%s)
      seam ens_smoke_r$r "$CK_C1" "$SHA_C1" "$CK_C1R" "$SHA_C1R" "$SMOKE_TOK" "$S/run$r/stage2_ens_smoke.npz" "$S/run$r/stage2_ens_smoke.proposals.npz" "$S/run$r/ens_smoke.npz"
      rc=$?; say "  smoke run $r rc $rc after $(( $(date +%s) - t0 )) s; $(grep -o 'ZZENS .*' "$S/run$r/ens_smoke.seam.log" 2>/dev/null | tail -1 | cut -c1-220)"
      [ "$rc" = 0 ] || [ "$rc" = 1 ] || fail "smoke_seam_$r" "$rc"
      [ -f "$S/run$r/ens_smoke.npz" ] || fail "smoke_dump_$r" 5
      printf '%s rc=%s\n' "$(ts)" "$rc" > "$M/smoke_run$r.DONE"
    fi
  done
  ctrl smoke "$S/run1/ens_smoke.npz" "$S/run1/stage2_ens_smoke.npz" "$S/run1/stage2_ens_smoke.proposals.npz" p5c_c1 p5c_c1r "$S/ctrl_smoke.json" "" "$S/run2/ens_smoke.npz"
  rc=$?; say "  smoke ctrl rc $rc: $(grep -o 'ZZENS_CTRL.*' "$L/ctrl_smoke.log" | cut -c1-260)"
  [ "$rc" = 0 ] || fail smoke_ctrl "$rc"
  printf '%s\n' "$(ts)" > "$M/smoke.DONE"
fi
run_arm() {   # $1 ARM  $2 primary ckpt  $3 primary sha  $4 member ckpt  $5 member sha  $6 banked A  $7 banked B  [$8 det dump]
  local A=$1 a; a=$(echo "$A" | tr 'A-Z' 'a-z')
  local NAME=ens_$a AW=$W/ens_$a
  mkdir -p "$AW"
  [ -f "$M/$A.score.DONE" ] && { say "skip $A (scored)"; return 0; }
  if [ ! -f "$M/$A.seam.DONE" ]; then
    aside "$DATA/seams/stage2_$NAME.npz" "$DATA/seams/stage2_$NAME.proposals.npz" "$DATA/seams/stage2_$NAME.report.json" "$AW/$NAME.npz"
    memgate "seam_$A" || fail "memgate_seam_$A" 8
    say "RUN seam $A"; t0=$(date +%s)
    seam "$NAME" "$2" "$3" "$4" "$5" "$TOK" "$DATA/seams/stage2_$NAME.npz" "$DATA/seams/stage2_$NAME.proposals.npz" "$AW/$NAME.npz"
    rc=$?; say "  seam $A rc $rc after $(( $(date +%s) - t0 )) s; $(grep -o 'ZZENS .*' "$AW/$NAME.seam.log" 2>/dev/null | tail -1 | cut -c1-220)"
    [ "$rc" = 0 ] || [ "$rc" = 1 ] || fail "seam_$A" "$rc"
    [ -f "$AW/$NAME.npz" ] || fail "ensdump_$A" 5
    printf '%s rc=%s\n' "$(ts)" "$rc" > "$M/$A.seam.DONE"
  fi
  if [ ! -f "$M/$A.ctrl.DONE" ]; then
    ctrl "$A" "$AW/$NAME.npz" "$DATA/seams/stage2_$NAME.npz" "$DATA/seams/stage2_$NAME.proposals.npz" "$6" "$7" "$AW/ctrl_$A.json" 12146 "${8:-}"
    rc=$?; say "  ctrl $A rc $rc: $(grep -o 'ZZENS_CTRL.*' "$L/ctrl_$A.log" | cut -c1-260)"
    printf '%s rc=%s\n' "$(ts)" "$rc" > "$M/$A.ctrl.DONE"
  fi
  local att
  for att in 1 2 3; do
    memgate "score_${A}_$att" || fail "memgate_score_$A" 8
    say "RUN score $A attempt $att"; t0=$(date +%s)
    "$PY_T" "$EV/stage2_sft.py" --ckpt "$2" --name "$NAME" --rule navsim_v1 --tokens "$TOK" --dump-proposals --ego-yaw-table "$YAW" > "$L/score_${A}_$att.log" 2>&1 < /dev/null
    rc=$?; say "  score $A rc $rc after $(( $(date +%s) - t0 )) s"
    CSV="$DATA/score/refe_stage2_$NAME/refe_stage2_$NAME.csv"
    if [ "$rc" = 0 ] && [ -f "$CSV" ]; then
      printf '%s rc=0 csv_lines=%s\n' "$(ts)" "$(wc -l < "$CSV")" > "$M/$A.score.DONE"
      say "  scored $A: $(wc -l < "$CSV") CSV lines"
      return 0
    fi
    sleep 300
  done
  fail "score_$A" 7
}
run_arm ENS1 "$CK_C1" "$SHA_C1" "$CK_C1R" "$SHA_C1R" p5c_c1 p5c_c1r "$S/run1/ens_smoke.npz"
run_arm ENS1r "$CK_C1R" "$SHA_C1R" "$CK_C1" "$SHA_C1" p5c_c1r p5c_c1
# ---- both pulled checkpoints are re-creatable copies of the HF files (sha-verified at pull and by CTRL-SHA): delete them, logged
if [ ! -f "$M/ckpt_deleted.DONE" ]; then
  s1=$(stat -c %s "$CK_C1" 2>/dev/null); s2=$(stat -c %s "$CK_C1R" 2>/dev/null)
  rm -f "E:/Projects/TanitAD-artifacts/ensscore/arms/C1/model_final.pt" && say "deleted E:/Projects/TanitAD-artifacts/ensscore/arms/C1/model_final.pt ($s1 B; HF Sayood/tanitad-refe-p5c p5c/c1/model_final.pt, sha256 $SHA_C1)"
  rm -f "E:/Projects/TanitAD-artifacts/ensscore/arms/C1r/model_final.pt" && say "deleted E:/Projects/TanitAD-artifacts/ensscore/arms/C1r/model_final.pt ($s2 B; HF Sayood/tanitad-refe-p5c p5c/c1r/model_final.pt, sha256 $SHA_C1R)"
  printf '%s\n' "$(ts)" > "$M/ckpt_deleted.DONE"
fi
printf '%s\n' "$(ts)" > "$M/chain.DONE"
say "chain DONE"
