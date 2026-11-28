#!/usr/bin/env bash
# refav1 full-grid confirmation — the two PRE-REGISTERED arms of
# TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refav1-fullgrid-loncomb3/SPEC.md (A1, A2).
# Dev box RTX 4060, inference only, sequential. Stages verified by ARTIFACTS; opaque ZZ tokens.
set -u
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
T=C:/Users/Admin/tipsnap/b3f7ea6f
OUT=C:/Users/Admin/refav1_fullgrid
LBL=C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz
CK=C:/Users/Admin/refav1_eval_slice/ckpt_ep3
CACHE=C:/Users/Admin/tanitad-data/refav1-eval141/refav1-fp8-eval
EPS=C:/Users/Admin/tanitad-data/refav1-eval141/eps
JERK=0.02,15.11245,64.29715042415070
export PYTHONPATH="$T/stack;$T/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
cd "$OUT" || exit 2
echo "ZZFG-START $(date '+%F %T')ZZ"

# preflight: the tip snapshot's tanitad, never the G: editable install
tf=$("$PY" -c "import tanitad,pathlib;print(pathlib.Path(tanitad.__file__).as_posix())" 2>&1)
case "$tf" in *tipsnap/b3f7ea6f/stack/tanitad/*) echo "ZZFG-IMPORT-OK $tf ZZ" ;;
  *) echo "ZZFG-IMPORT-WRONG $tf ZZ"; exit 3 ;; esac

# provenance: md5 of the tool, the model code, the checkpoint and every input, BEFORE the first window
{
  echo "# refav1 full-grid provenance, $(date '+%F %T')"
  for f in "$T/taniteval/tools/refav1_arm.py" "$T/taniteval/tools/refav1_paired_delta.py" \
           "$T/stack/tanitad/refs/refa_v1.py" "$T/stack/tanitad/refs/refa_v1_plan.py" \
           "$CK/ckpt.pt" "$CK/config.json" "$LBL" "$CACHE/index.json" "$EPS/_v2manifest.pt"; do
    echo "$(md5sum "$f" | cut -c1-32)  $f"
  done
  echo "fp8 files: $(ls "$CACHE"/*.pt | wc -l)   episodes: $(ls "$EPS"/*.v2ep.pt | wc -l)"
} > provenance.txt
grep -q "1189bc020018c2c67ce03d566c390285" provenance.txt || { echo "ZZFG-CKPT-MD5-WRONGZZ"; exit 4; }
grep -q "aa12c948f062181c3297265b51526ec5" provenance.txt || { echo "ZZFG-LABELS-MD5-WRONGZZ"; exit 4; }
echo "ZZFG-PROVENANCE-OK ZZ"

common=(--ckpt "$CK/ckpt.pt" --config "$CK/config.json" --cache "$CACHE" --episodes "$EPS"
        --labels "$LBL" --nav "$LBL" --device cuda --episodes-n 0 --window-stride 40
        --no-navshuf --no-lead-block --cost-metric ccos)

for seed in 0 1; do
  tag=loncomb3_s$seed
  if [ -s "rec_$tag.json" ] && [ "$(ls dump_$tag/ep*.npz 2>/dev/null | wc -l)" -eq 141 ]; then
    echo "ZZFG-SKIP $tag (already complete)ZZ"; continue
  fi
  waited=0
  while :; do   # GPU gate: shared card; this arm runs alone
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')
    case "$g" in ''|*[!0-9]*) g=99999 ;; esac
    [ "$g" -le 6000 ] && break
    [ "$waited" -ge 21600 ] && { echo "ZZFG-GATE-TIMEOUT $tag gpu=${g}MiBZZ"; exit 5; }
    sleep 60; waited=$((waited + 60))
  done
  echo "ZZFG-RUN $tag gpu=${g}MiB waited=${waited}s $(date '+%T')ZZ"
  "$PY" "$T/taniteval/tools/refav1_arm.py" "${common[@]}" --cost-weights "$JERK" --plan-seed "$seed" \
    --a-sustain-mode a0_shift --jerk-seam a0 \
    --dump-dir "$OUT/dump_$tag" --out "$OUT/rec_$tag.json" \
    --arm "refav1-21109-full141-$tag" >> "$OUT/$tag.log" 2>&1
  n=$(ls "dump_$tag"/ep*.npz 2>/dev/null | wc -l)
  if [ -s "rec_$tag.json" ] && [ "$n" -eq 141 ]; then echo "ZZFG-DONE $tag episodes=$n $(date '+%T')ZZ"
  else echo "ZZFG-FAIL $tag rec=$([ -s rec_$tag.json ] && echo yes || echo no) episodes=$n (see $tag.log)ZZ"; exit 6; fi
done
echo "ZZFG-ALL-DONE $(date '+%F %T')ZZ"
