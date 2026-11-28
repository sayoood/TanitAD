#!/usr/bin/env bash
# chain_label_pipeline.sh -- the per-window GT lon label pipeline from RAW inputs, for one clip set:
#   obstacle.offline parquets -> agent join (build_b1_agent_join.py) -> lead pass -> v9 build (build_v9_labels.py, this tree) -> K-LABEL read.
# CPU only, no network, no GPU.  Usage:
#   bash chain_label_pipeline.sh eval139   # control: rebuild eval139's labels from its raw parquets; K-LABEL-D must reproduce the battery
#   bash chain_label_pipeline.sh fresh     # the fresh split (needs F:/refcv9_lonhead/obstacle_fresh/*.parquet for all 300 clips -- NOT fetched here)
# Markers ZZLP<TAG>ZZ in logs/label_pipeline_<set>.log; every step judged on its ARTIFACT, never on an exit code through a pipe.
set -u
SET=${1:?eval139|fresh}
W=C:/Users/Admin/refcv8_pf/refcv9_lonhead
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp \
       PYTHONPATH="$T/stack;$T/stack/scripts;$T/taniteval" PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
case "$SET" in
  eval139) K=E:/refcv6_eval_kit/data; EPS=$K/refcv6-b1-416x1024-eval139; MAN=$EPS/_v2manifest.pt; SIDE=$K/refcv6_clip_clock_sidecar.jsonl
           LAB=$K/v8labels/labels/s2_labels_v8_eval.jsonl.gz; EGO=C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo
           OBS=C:/Users/Admin/tanitad-data/physicalai/labels/obstacle_offline_b1eval; CL=F:/refcv9_lonhead/kit_eval139/eval139_clips.json
           OUT=F:/refcv9_lonhead/kit_eval139/from_parquets; ROLE=audit; SPLIT=eval139
           PAR=(--parity-role audit --parity-audit-reason "LONHEAD K-LABEL-D control: re-derive the eval139 battery lon labels from raw parquets; not supervision") ;;
  fresh)   K=F:/l1c/kit_fresh/data; EPS=$K/refcv6-b1-416x1024-eval139; MAN=$EPS/_v2manifest.pt; SIDE=$K/clock_sidecar.jsonl
           LAB=$K/v8labels/labels/s2_labels_v8_eval.jsonl.gz; EGO=F:/l1c/pai_root/labels/egomotion_flat
           OBS=F:/refcv9_lonhead/obstacle_fresh; CL=F:/refcv9_lonhead/fresh_clips.json
           OUT=F:/refcv9_lonhead/kit_fresh_lead; ROLE=""; SPLIT=eval
           PAR=(--parity-role "" --parity-mode refuse) ;;   # the fresh labels ARE supervision for LH: default role (checked against the deployed val)
  *) echo "unknown set $SET"; exit 2 ;;
esac
# DRY (mechanics) override: LP_CLIPS=<json list> LP_OUT=<dir> runs the same steps on a subset into a scratch dir
CL=${LP_CLIPS:-$CL}; OUT=${LP_OUT:-$OUT}
mkdir -p "$OUT" "$W/logs"
LOG=$W/logs/label_pipeline_$SET${LP_OUT:+_dry}.log
mk() { echo "ZZLP$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mk START "set $SET"
n_obs=$(ls "$OBS" 2>/dev/null | grep -c '\.parquet$')
mk INPUTS "obstacle parquets present: $n_obs"
# 1. join
"$PY" -X utf8 "$T/stack/scripts/build_b1_agent_join.py" --pose-source v2ep --eps-dir "$EPS" --ego-dir "$EGO" --obstacle-dir "$OBS" \
      --clips "$CL" "${PAR[@]}" --out "$OUT/agents_join.jsonl.xz" > "$W/logs/lp_${SET}_join.log" 2>&1
[ -s "$OUT/agents_join.jsonl.xz" ] || { mk JOINFAIL "see logs/lp_${SET}_join.log"; exit 4; }
mk JOIN "$(md5sum "$OUT/agents_join.jsonl.xz" | cut -c1-32)"
# 2. lead pass
"$PY" -X utf8 "$T/stack/scripts/build_v9_labels.py" lead-pass --manifest "$MAN" --sidecar "$SIDE" --join "$OUT/agents_join.jsonl.xz" \
      --ego "$EGO/{}.parquet" --out "$OUT/lead.npz" > "$W/logs/lp_${SET}_lead.log" 2>&1
[ -s "$OUT/lead.npz" ] || { mk LEADFAIL "see logs/lp_${SET}_lead.log"; exit 4; }
mk LEAD "$(tail -1 "$W/logs/lp_${SET}_lead.log" | cut -c1-80)"
# 3. v9 build WITH the lead pass (lc off: it feeds lateral targets only; K-LABEL-C reproduced the battery without it)
"$PY" -X utf8 "$T/stack/scripts/build_v9_labels.py" build --split "$SPLIT" --manifest "$MAN" --sidecar "$SIDE" --labels "$LAB" \
      --lead "$OUT/lead.npz" --ego "$EGO/{}.parquet" --out-dir "$OUT/v9" > "$W/logs/lp_${SET}_build.log" 2>&1
REL=$OUT/v9/v9_labels_$SPLIT.npz
[ -s "$REL" ] || { mk BUILDFAIL "see logs/lp_${SET}_build.log"; exit 4; }
mk BUILD "$REL $(md5sum "$REL" | cut -c1-32)"
mk END "ok"
exit 0
