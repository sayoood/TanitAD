#!/usr/bin/env bash
# MM 2026-11-20: navtest pre-work for ONE field arm (PREREG_AUX_FIELD_LOSS_P7.md: "Eval is on the dev box"; harness and seam as P7's read).
# HF pull (Sayood/tanitad-refe-p5c p7field/<arm>) -> sha256 == the pod's sha file -> p7_read_chain.py seam1,score (GPU lock taken INSIDE
# stage2_sft.py) -> the 64-proposal tracked table (p5c_prop_score.py, CPU, 2 shards; the TACTICAL selection-quality instrument).
# Pre-work only: no F-vs-C verdict here. F's argv is fixed and gate-pinned, so reading C first cannot influence F.
# Resumable: marker per stage in $W/markers_<arm>/. Harness = the P5-C read's export F:/TanitAD_p5cread_tip (751f3ae7), the same harness
# as the P7 base and P5-C reads. Usage: field_seam_chain.sh <ARM: C|F|Cs|Fs>.  Launch detached: launch_field_seam.ps1.
set -u
A="${1:?arm}"; a=$(echo "$A" | tr 'A-Z' 'a-z')
HOLD=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_read
RC=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain
NX=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next
TIP=F:/TanitAD_p5cread_tip
PK="$TIP/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"; EV="$PK/eval"
PY_T=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PY_N=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
DATA=E:/Projects/TanitAD/data/refe_navtest
YAW=E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtest.json.gz
LOCK=C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock
W=E:/Projects/TanitAD-artifacts/field_read; M=$W/markers_$a; L=$W/logs_$a; ARMD=$W/arms/$A
NAME=p7field_$a
mkdir -p "$M" "$L" "$ARMD" "$W/tmp"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $A $*" >> "$W/CHAIN.log"; }
release_lock() { if [ -f "$LOCK" ] && grep -q "refe-stage2-$NAME" "$LOCK" 2>/dev/null; then rm -f "$LOCK" && say "released the dev-box GPU lock left by $NAME"; fi; }
trap 'release_lock' EXIT
trap 'say "signal received"; release_lock; exit 143' INT TERM
fail() { printf 'stage=%s rc=%s time=%s\n' "$1" "$2" "$(ts)" > "$M/chain.FAILED"; say "FAILED $1 (rc $2) log $L/$1.log"; release_lock; exit 1; }
[ -f "$M/chain.FAILED" ] && { n=1; while [ -f "$M/chain.FAILED.attempt_$n" ]; do n=$((n+1)); done; mv "$M/chain.FAILED" "$M/chain.FAILED.attempt_$n"; say "resuming (attempt $n kept)"; }
commit_ok() { local g; g=$("$PY_T" "$RC/tools/commit_guard.py" | tr -d '\r\n '); case "$g" in ''|*[!0-9]*) return 1;; esac; [ "$g" -ge 6 ]; }
wait_commit() { local i=0; while ! commit_ok; do i=$((i+1)); [ $i -gt 60 ] && return 1; sleep 60; done; return 0; }
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$TIP/stack;$TIP/taniteval;$EV;$EV/p5c"
export TANITAD_REPO="$TIP" REFE_EVAL_DIR="$EV" P5C_REFE_DIR="$PK/refe" TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim REFE_NAVTEST_DIR="$DATA"
export TMP="$W/tmp" TEMP="$W/tmp" PYTHONIOENCODING=utf-8 PYTHONUTF8=1
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
say "chain start tip=$(cat $TIP/EXPORT_SHA.txt 2>/dev/null | cut -c1-8)"
[ -f "$TIP/EXPORT_DONE.txt" ] || fail preflight 2
g=$("$PY_T" -c "import shutil;print(int(shutil.disk_usage('E:/').free//(1<<30)))" | tr -d '\r\n '); [ "${g:-0}" -ge 4 ] || { say "REFUSING: E: ${g} GB free (< 4)"; fail preflight 2; }
run hf_pull "$PY_T" "$RC/tools/hf_pull.py" --arm "$A" --prefix "p7field/$a" --out "$ARMD" --repo Sayood/tanitad-refe-p5c
grep -q "ZZP5C_HF_PULL $A OK" "$L/hf_pull.log" || fail hf_pull_verify 4
SHA=$(grep -oE '[0-9a-f]{64}' "$ARMD/model_final_sha256_$A.txt" | head -1); [ -n "$SHA" ] || fail sha 4
run s2 env -u CUDA_VISIBLE_DEVICES "$PY_T" "$EV/p7_read_chain.py" --ckpt "$ARMD/model_final.pt" --name "$NAME" --expect-ckpt-sha256 "$SHA" \
    --base p6 --also p2 --ego-yaw-table "$YAW" --work "$W/s2_$a" --only seam1,score
release_lock
NPZ="$DATA/seams/stage2_$NAME.proposals.npz"; [ -s "$NPZ" ] || fail dump_missing 5
if [ ! -f "$M/proptable.DONE" ]; then
  wait_commit || fail proptable 90
  say "RUN proptable (2 shards)"; t0=$(date +%s); pids=()
  NAV="F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
  mkdir -p "$W/propscore"
  for s in 0 1; do
    PYTHONPATH="$NAV;$PK/refe" "$PY_N" "$NX/p5c_prop_score.py" --arm "$A" --proposals "$NPZ" --out "$W/propscore/full_${A}_s${s}of2.jsonl" \
      --shard $s/2 > "$L/proptable_s$s.log" 2>&1 < /dev/null &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=$?; done
  n=$(cat "$W/propscore/full_${A}_s0of2.jsonl" "$W/propscore/full_${A}_s1of2.jsonl" 2>/dev/null | wc -l)
  say "  proptable rc $rc rows $n after $(( $(date +%s) - t0 )) s"
  [ "$rc" = 0 ] && [ "$n" -ge 12146 ] || fail proptable "$rc"
  printf '%s rc=0 rows=%s\n' "$(ts)" "$n" > "$M/proptable.DONE"
fi
say "ZZFIELDSEAM_DONE $A seam $DATA/seams/stage2_$NAME.npz table $W/propscore/full_${A}_s*of2.jsonl"
printf '%s\n' "$(ts)" > "$M/chain.DONE"
