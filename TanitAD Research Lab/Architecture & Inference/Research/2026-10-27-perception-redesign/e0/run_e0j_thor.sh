#!/usr/bin/env bash
# run_e0j_thor.sh <smoke|chain>   -- ON THOR, launched DETACHED (setsid nohup ... < /dev/null); costs ZERO tokens while it waits.
#   smoke : CPU only, niced, NO lock (cuda hidden): 1 batch x 1 window, every control on -- proves the plumbing before a multi-hour queue.
#   chain : queue behind agent OVF's overfit records on the Thor GPU lock (flock, <= 12 h; the same lock file r9e1_run.sh uses), then
#             stage banded    (primary: C0 argv + --r9-map-banded + near-lift 10; Gram + L/Z/T controls) -> CPU read -> ZZE0J-BANDED-DONEZZ
#             stage baseline8 (matched-instrument refcv8 reading at w = 1: the 8-way CE, near-lift 20; 3 backwards per batch) -> CPU read
#                                                                                                       -> ZZE0J-ALL-DONEZZ
#           The lock is HELD for the whole chain and released at the end; kill the chain after ZZE0J-BANDED-DONEZZ (the line is ZZE0J-BANDED-DONEZZ in chain.log) if the GPU is wanted
#           (the primary reading is then complete; baseline8 is a comparison).
# Markers (one line each) in $LOG/chain.log: ZZE0J-{WAITLOCK,LOCKED,STAGE-START,STAGE-DONE,STAGE-FAIL,BANDED-READ,BANDED-DONE,READ,ALL-DONE,SMOKE-DONE,SMOKE-FAIL}ZZ. The ARTIFACT ($OUT/<stage>/pgrad_gram.json, $OUT/E0J_RESULT.json) is the evidence.
set -u
MODE="${1:-}"
R=/home/nvidia/r8ns/r9gram; TREE=$R/tree; CODE=$R/code; OUT=$R/out; LOG=$R/logs
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock
EMA=/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt
CFG_BANDED=/home/nvidia/r8ns/r9c1/runs/r9c1-M1-smoke/config.json          # a banded rig run's config.json (the stamps the loader checks)
CFG_BASE8=/home/nvidia/r8ns/r9dump/rig_config.json                          # an un-banded rig config.json (KA-STEP0's)
ARGVJ=$R/r9e1_C0.argv.json
REMAP=/home/nvidia/r8ns/perc8/code/refcv8_trdiag_remap_thor.json             # --eval-cache -> the 139-clip TRAIN-DIAG cache (+ train labels/sidecars)
mkdir -p "$OUT" "$LOG" "$R/scratch"
export PYTHONPATH="$TREE/stack:$TREE/taniteval" PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token REFCV6_KIT=/home/nvidia REFCV6_REMAP_OVERRIDES="$REMAP"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO
mk() { echo "ZZE0J-$1ZZ $(date -u +%FT%T) $2" >> "$LOG/chain.log"; }
cd "$TREE/stack" || { mk NOTREE "$TREE"; exit 2; }

if [ "$MODE" = "smoke" ]; then
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  mk SMOKE-START "cpu, niced, no lock"
  nice -n 19 "$PY" -X utf8 "$CODE/pgrad_gram.py" --stage banded --ckpt "$EMA" --base-config "$CFG_BANDED" --argv-json "$ARGVJ" \
        --out "$OUT/smoke" --n-batches 1 --batch 1 --workers 0 --device cpu --scratch "$R/scratch" > "$LOG/smoke.out" 2> "$LOG/smoke.err"
  RC=$?
  if [ -s "$OUT/smoke/pgrad_gram.json" ]; then
    "$PY" -X utf8 "$CODE/e0j_read.py" --gram "$OUT/smoke/pgrad_gram.json" --out "$OUT/smoke/E0J_RESULT_SMOKE.json" > "$LOG/smoke_read.out" 2>&1
    mk SMOKE-DONE "rc=$RC read_rc=$?"
  else
    mk SMOKE-FAIL "rc=$RC (see $LOG/smoke.err)"
  fi
  exit 0
fi

[ "$MODE" = "chain" ] || { echo "usage: $0 smoke|chain"; exit 64; }
export OMP_NUM_THREADS=8
exec 9>"$LOCKF"; mk WAITLOCK "pid $$ (queued behind the lock holder; <= 12 h)"
flock -w 43200 9 || { mk LOCKBUSY ""; exit 3; }
for i in $(seq 1 60); do
  NAPPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"; [ "$NAPPS" = 0 ] && break; sleep 30
done
[ "$NAPPS" = 0 ] || { mk GPUNOTEMPTY "$NAPPS"; exit 4; }
mk LOCKED "GPU empty"
stage() {  # stage <name> <base-config> <extra args...>
  local name="$1" cfg="$2"; shift 2
  if [ -s "$OUT/$name/pgrad_gram.json" ]; then mk STAGE-SKIP "$name"; return 0; fi
  mk STAGE-START "$name"
  ( "$PY" -X utf8 "$CODE/pgrad_gram.py" --stage "$name" --ckpt "$EMA" --base-config "$cfg" --argv-json "$ARGVJ" --out "$OUT/$name" \
        --scratch "$R/scratch" "$@" 9>&- > "$LOG/$name.out" 2> "$LOG/$name.err" ); local rc=$?
  if [ -s "$OUT/$name/pgrad_gram.json" ]; then mk STAGE-DONE "$name rc=$rc"; return 0; fi
  mk STAGE-FAIL "$name rc=$rc (see $LOG/$name.err)"; return 1
}
stage banded "$CFG_BANDED" || { exec 9>&-; exit 5; }
"$PY" -X utf8 "$CODE/e0j_read.py" --gram "$OUT/banded/pgrad_gram.json" --out "$OUT/E0J_RESULT.json" > "$LOG/read_banded.out" 2>&1; mk BANDED-READ "rc=$? $(tail -n 1 "$LOG/read_banded.out")"
mk BANDED-DONE ""
stage baseline8 "$CFG_BASE8" --l-control 0 --z-batches 0 --t-batches 0 || { exec 9>&-; exit 6; }
"$PY" -X utf8 "$CODE/e0j_read.py" --gram "$OUT/banded/pgrad_gram.json" --baseline8 "$OUT/baseline8/pgrad_gram.json" --out "$OUT/E0J_RESULT.json" > "$LOG/read_all.out" 2>&1
mk READ "rc=$? $(tail -n 1 "$LOG/read_all.out")"
exec 9>&-
mk ALL-DONE "GPU lock released"
