#!/usr/bin/env bash
# The A8 -> A7 GPU gap (Master Mind's sequence, 2026-09-19): ONE job at a time, nothing training.
#   step smoke : S1 pass on A3's FINISHED checkpoint, 2 windows, GPU -> must not crash; analyze runs
#   step pass  : S1 pass + S1A.4 box read on A8's 5k checkpoint, all eligible windows, GPU,
#                with the S1A.3 replicate on 50 windows -> rows + meta; then analyze -> verdict
#   step e9    : refcv3_arm on A8 5k, CPU, small n -> wall-clock s/window (E9)
#   step a7    : relaunch the A7 panel (resumable; arms start at A7-IN-s0)
# Usage: bash gap_runbook.sh <step>. Each step checks its ARTIFACT, never only an exit code.
set -u
RUN=/c/Users/Admin/tanitad-a7-run            # the landed tip's stack/ + taniteval/ (0-diff)
RUNW='C:\Users\Admin\tanitad-a7-run'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
BOX=/c/Users/Admin/qland/boxstat.py
ART=/d/Projects/TanitAD-artifacts
B=$ART/v2ep-eval124clean-416x1024cyl-halfB
LAB=/c/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/s2_labels_v8_eval.jsonl.gz
JOIN=$ART/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz
MAPS=$ART/sam3-maps-eval
A3=/c/Users/Admin/tanitad-caches/a3-heldout-20260919/run
A8=/c/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run
OUT=/c/Users/Admin/tanitad-caches/s1-gate-20260919
S1=/d/Projects/TanitAD/taniteval/tools/s1_pass.py   # the STAGED harness (runs from D:)
mkdir -p "$OUT"
export PYTHONPATH='D:\Projects\TanitAD\stack;D:\Projects\TanitAD;D:\Projects\TanitAD\taniteval'
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6

gate() {
  read -r GU HF < <("$PY" "$BOX")
  case "$GU$HF" in *[!0-9]*) echo "ZZGAP-PROBE-INCONCLUSIVE $GU $HF ZZ"; exit 3 ;; esac
  [ "$GU" -le 2500 ] && [ "$HF" -ge 8 ] || { echo "ZZGAP-BUSY gpu=${GU} host=${HF}ZZ"; exit 3; }
  echo "ZZGAP-CLEAR gpu=${GU}MiB host=${HF}GBZZ"
}

common=(--config "" --cache "$B" --labels "$LAB" --agents "$JOIN" --maps "$MAPS"
        --expect-windows 10600 --n 1000)

case "${1:-}" in
  smoke)
    gate
    rm -f "$OUT/smoke_A3.jsonl" "$OUT/smoke_A3.jsonl.meta.json"
    args=("${common[@]}"); args[1]="$A3/config.json"
    "$PY" "$S1" --mode pass "${args[@]}" --ckpt "$A3/ckpt.pt" --device cuda --limit 2 \
        --replicate 2 --rows "$OUT/smoke_A3.jsonl" --out "$OUT/unused.json" > "$OUT/smoke.log" 2>&1
    [ -f "$OUT/smoke_A3.jsonl.meta.json" ] || { echo "ZZGAP-SMOKE-NO-METAZZ"; tail -20 "$OUT/smoke.log"; exit 4; }
    "$PY" "$S1" --mode analyze --rows "$OUT/smoke_A3.jsonl" --out "$OUT/smoke_verdict.json" \
        --n-boot 50 >> "$OUT/smoke.log" 2>&1
    [ -f "$OUT/smoke_verdict.json" ] && echo "ZZGAP-SMOKE-OKZZ" || { echo "ZZGAP-SMOKE-ANALYZE-FAILEDZZ"; exit 5; }
    tail -4 "$OUT/smoke.log" ;;
  pass)
    gate
    # ⛔ The harness runs from the SHARED D: worktree, not a pinned tree: bracket every module
    # it imports with md5 so a file changed mid-run is DETECTED, never silently mixed.
    D=/d/Projects/TanitAD
    MODS=("$D/taniteval/tools/s1_pass.py" "$D/taniteval/tools/s1_gate.py"
          "$D/taniteval/tools/refcv3_arm.py" "$D/stack/tanitad/rl/pdm_proxy.py"
          "$D/stack/tanitad/models/box3d_head.py" "$D/stack/tanitad/models/agent_slots.py"
          "$D/stack/scripts/refc_v3_train.py" "$D/stack/scripts/ddv2_rl_refcv5.py"
          "$A8/ckpt.pt")
    md5sum "${MODS[@]}" > "$OUT/pass_md5_before.txt"
    args=("${common[@]}"); args[1]="$A8/config.json"
    "$PY" "$S1" --mode pass "${args[@]}" --ckpt "$A8/ckpt.pt" --device cuda --replicate 50 \
        --rows "$OUT/rows_A8.jsonl" --out "$OUT/unused.json" > "$OUT/pass.log" 2>&1
    md5sum -c "$OUT/pass_md5_before.txt" > "$OUT/pass_md5_check.txt" 2>&1 \
        && echo "ZZGAP-MD5-BRACKET-OKZZ" || { echo "ZZGAP-MD5-CHANGED-MID-RUNZZ"; cat "$OUT/pass_md5_check.txt"; }
    [ -f "$OUT/rows_A8.jsonl.meta.json" ] || { echo "ZZGAP-PASS-NO-METAZZ"; tail -20 "$OUT/pass.log"; exit 4; }
    "$PY" "$S1" --mode analyze --rows "$OUT/rows_A8.jsonl" --out "$OUT/verdict_A8.json" \
        >> "$OUT/pass.log" 2>&1
    [ -f "$OUT/verdict_A8.json" ] && echo "ZZGAP-PASS-OKZZ" || { echo "ZZGAP-PASS-ANALYZE-FAILEDZZ"; exit 5; }
    tail -4 "$OUT/pass.log" ;;
  e9)
    T0=$(date +%s)
    CUDA_VISIBLE_DEVICES="" "$PY" /d/Projects/TanitAD/taniteval/tools/refcv3_arm.py \
        --ckpt "$A8/ckpt.pt" --config "$A8/config.json" --episodes "$B" --labels "$LAB" \
        --device cpu --episodes-n 1 --window-stride 25 --out "$OUT/e9_refcv3arm_cpu.json" \
        --dump-dir "$OUT/e9_dump" > "$OUT/e9.log" 2>&1
    RC=$?; T1=$(date +%s)
    echo "ZZGAP-E9-RC $RC wall=$((T1 - T0))sZZ"
    grep -E "\[cost\]|windows|n_windows" "$OUT/e9.log" | tail -5 ;;
  a7)
    gate
    OUTA7=/c/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919
    echo "relaunch: run in the background with its own panel log" ;;
  *) echo "usage: $0 smoke|pass|e9|a7"; exit 1 ;;
esac
