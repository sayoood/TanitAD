#!/usr/bin/env bash
# Driver for the resnet101-on-8GB fit sweep. ONE JOB AT A TIME, re-probed
# between arms -- this box has other tenants and the probe is the gate, not a
# formality: a probe that cannot read its quantity is INCONCLUSIVE, never clear.
set -u
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
D="/c/Users/Admin/tanitad-wt-bevtac/TanitAD Research Lab/Architecture & Inference/Research/2026-09-18-resnet101-8gb-fit"
OUT=/c/Users/Admin/tanitad-caches/r101fit-20260918
mkdir -p "$OUT"
WTW='C:\Users\Admin\tanitad-wt-bevtac'
export PYTHONPATH="$WTW\\stack;$WTW;$WTW\\taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6

probe () {
  read -r GU HF < <("$PY" /c/Users/Admin/qland/boxstat.py) || {
    echo "ZZABORT-PROBE-FAILED box unreadable -- INCONCLUSIVE, never clearZZ"; return 3; }
  case "$GU$HF" in *[!0-9]*) echo "ZZABORT-PROBE-SHAPE '$GU $HF'ZZ"; return 3 ;; esac
  [ "$GU" -le 2500 ] || { echo "ZZABORT-GPU-BUSY used=${GU}MiBZZ"; return 3; }
  [ "$HF" -ge 8 ]    || { echo "ZZABORT-HOST-TIGHT free=${HF}GBZZ"; return 3; }
  echo "box gpu=${GU}MiB host=${HF}GB $(date -u +%H:%M:%SZ)"
}

arm () {   # $1 tag   $2... extra rig_arm.py flags
  local tag="$1"; shift
  probe || return 3
  echo "== RIG $tag  $*"
  ( cd "$WTW"/stack 2>/dev/null || cd /c/Users/Admin/tanitad-wt-bevtac/stack
    "$PY" -u "$D/raw/rig_arm.py" --tag "$tag" --outroot "$D/raw" "$@" \
    ) > "$OUT/$tag.log" 2>&1
  local rc=$?
  grep -h "ZZARM" "$OUT/$tag.log" || echo "ZZARM-NO-LINE $tag rc=$rcZZ"
}

case "${1:-all}" in
  micro)
    probe || exit 3
    cd /c/Users/Admin/tanitad-wt-bevtac/stack && \
      "$PY" -u "$D/raw/trunk_microbench.py" --wb 8 \
            --out "$D/raw/micro_wb8.json"
    ;;
  ref)
    arm r34_b1_ref  --trunk-name resnet34.a1_in1k
    ;;
  base)
    arm r101_b1_base --trunk-name resnet101.a1_in1k
    ;;
  *)
    echo "usage: run_arms.sh micro|ref|base"; exit 2 ;;
esac
