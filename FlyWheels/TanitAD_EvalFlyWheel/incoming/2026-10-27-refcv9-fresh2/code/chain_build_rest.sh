#!/bin/bash
# chain_build_rest.sh -- FRESH2 steps 1-2 after the v2 episode build: labels/speed/clock on the pool -> refusals logged + EXCLUDED (no replacement) ->
# labels/speed/clock re-run on the usable set + check -> split -> v9/extr/kit/windows -> the lead-pass label pipeline (join, lead, v9 WITH lead).
# CPU only.  Every step judged on its ARTIFACT (markers ZZF2<NAME>ZZ in logs/chain_build_rest.log), never on an exit code through a pipe.  No git.  No G: writes.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_fresh2; CODE=$W/code; L=$W/logs/chain_build_rest.log
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
K=F:/l1c/kit_fresh2
mk() { echo "ZZF2$1ZZ $(date +%FT%T%z) $2" >> "$L"; }
mk START "pid $$"
n_eps() { ls "$K/data/refcv6-b1-416x1024-eval139/" 2>/dev/null | grep -c '\.v2ep\.pt$'; }
mk EPS "episode files $(n_eps)"
if [ ! -f F:/l1c/fresh2/fresh2_ids.pool.json ]; then
  "$PY" -X utf8 "$CODE/fresh2_build_inputs.py" --set fresh2 --steps labels,speed,clock > "$W/logs/build_lsc_pool.log" 2>&1
  [ -s "$K/data/clock_sidecar.jsonl.meta.json" ] && [ -s "$K/labels_report.json" ] || { mk ABORT "labels/speed/clock (pool) artifacts missing: logs/build_lsc_pool.log"; exit 3; }
  mk LSCPOOL "$(tail -4 "$W/logs/build_lsc_pool.log" | tr '\n' ' ' | cut -c1-300)"
  "$PY" -X utf8 "$CODE/fresh2_refusals.py" > "$W/logs/refusals.log" 2>&1
  if [ -s "$W/result/REFUSAL_STOP.json" ]; then mk STOP "refusals above 15 % -- see result/REFUSAL_STOP.json"; exit 5; fi
  [ -s F:/l1c/fresh2/fresh2_ids.pool.json ] || { mk ABORT "refusal round did not run: logs/refusals.log"; exit 3; }
  mk REFUSALS "$(tail -1 "$W/logs/refusals.log")"
fi
n_use=$("$PY" -I -c "import json; print(len(json.load(open('F:/l1c/fresh2/fresh2_ids.json'))))")
if [ "$(n_eps)" != "$n_use" ] || [ ! -s "$K/data/refcv6-b1-416x1024-eval139/_v2manifest.pt" ]; then
  "$PY" -X utf8 "$CODE/fresh2_build_inputs.py" --set fresh2 --steps labels,speed,clock > "$W/logs/build_lsc_usable.log" 2>&1
  mk LSCUSABLE "$(tail -4 "$W/logs/build_lsc_usable.log" | tr '\n' ' ' | cut -c1-300)"
fi
"$PY" -X utf8 "$CODE/fresh2_refusals.py" --check > "$W/logs/refusals_check.log" 2>&1
grep -q ZZF2REFCHECKPASSZZ "$W/logs/refusals_check.log" || { mk ABORT "post-exclusion check failed: logs/refusals_check.log"; exit 4; }
mk REFCHECK "$(tail -1 "$W/logs/refusals_check.log" | cut -c1-200)"
"$PY" -X utf8 "$CODE/fresh2_split.py" > "$W/logs/split.log" 2>&1
[ -s F:/l1c/fresh2/split.json ] || { mk ABORT "split: logs/split.log"; exit 4; }
mk SPLIT "$(tail -1 "$W/logs/split.log")"
"$PY" -X utf8 "$CODE/fresh2_build_inputs.py" --set fresh2 --steps v9,extr,kit,windows > "$W/logs/build_v9_windows.log" 2>&1
[ -s "$K/data/v9/v9_labels_eval.npz" ] && [ -s "$K/data/v9/v9_labels_eval.md5" ] && [ -s "$K/windows_s0.json" ] && [ -s "$K/data/extrinsics_union.json" ] || { mk ABORT "v9/extr/kit/windows artifacts: logs/build_v9_windows.log"; exit 4; }
mk KIT "$(grep -E '^\[(v9|extr|windows)\]' "$W/logs/build_v9_windows.log" | tr '\n' ' ' | cut -c1-300)"
bash "$CODE/chain_label_pipeline_fresh2.sh" fresh2
[ -s F:/l1c/fresh2/kit_fresh2_lead/v9/v9_labels_eval.npz ] || { mk ABORT "lead-pass label pipeline: logs/label_pipeline_fresh2.log"; exit 4; }
mk LEADREL "$(grep ZZLPBUILDZZ "$W/logs/label_pipeline_fresh2.log" | tail -1 | cut -c1-200)"
mk END "ok"
