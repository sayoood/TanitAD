#!/bin/bash
# Harness identity control for the P6 layout fix: re-score P2's BANKED seam (stage2_p2_full.npz) on 24 navtest tokens through the REAL read-path scoring call
# (stage2_sft.py's own command: PY_D score_e.py --label --seam --tokens --out, cwd = banked harness, env RUN_V1_FIX1_ALLOW_DRIFT=1) on today's layout
# (archive on F:, re-rooted metadata CSV), then compare per token with P2's banked CSV (compare.py). Never a P6 checkpoint. CPU only.
ART=E:/Projects/TanitAD-artifacts/p6read_layout_fix
PKG="E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PYD=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
C=$ART/harness_control
unset TANITAD_NAVSIM_ROOT MSYS_NO_PATHCONV
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 RUN_V1_FIX1_ALLOW_DRIFT=1
mkdir -p "$C/score"
cd "$PKG/eval/harness"
echo "$(date -u +%T) start" > "$C/status.txt"
$PYD score_e.py --label refe_layoutfix_ctl --seam E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p2_full.npz --tokens "$C/tokens.json" --out "$C/score" > "$C/score_e.log" 2>&1
echo "$(date -u +%T) score_e rc=$?" >> "$C/status.txt"
$PY -X utf8 -I "$C/compare.py" > "$C/compare.log" 2>&1
echo "$(date -u +%T) compare rc=$? $(tail -n 1 "$C/compare.log")" >> "$C/status.txt"
