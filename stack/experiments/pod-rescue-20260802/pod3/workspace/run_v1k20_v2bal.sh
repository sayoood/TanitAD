#!/bin/bash
# ⭐ PI DIRECTIVE 2026-08-02: "run v1 + rollout-k 20 from scratch with the extended data corpus".
#
# THE PAIR THIS COMPLETES — one flag apart, both FROM SCRATCH, same corpus, same architecture:
#   newpod  flagship-v1arch-v2bal-30k   v1 arch + v2bal + --rollout-k 4     (control, running)
#   pod3    THIS                        v1 arch + v2bal + --rollout-k 20    (the PI's arm)
#
# WHY IT MATTERS: RR-20 showed a 2000-step rollout-k 20 FINE-TUNE erases the longitudinal speed
# bias (+0.9397 -> -0.0092 m/s) but costs 2.2x in curvature. Whether that trade also holds when
# k=20 is trained FROM SCRATCH is UNMEASURED — v2corpus is the only from-scratch high-k arm we
# have and it staged to k=12 while ALSO moving ten other levers, so it cannot answer this.
#
# ⛔ NO --v2. --v2 is a TEN-LEVER ARCHITECTURE PACK (and would force rollout_k=12, overriding the
# flag this experiment is about). --v2-cache is the DATA loader and is the only v2 thing here.
# That distinction is the exact mistake that produced flagship-v2corpus-30k.
set -u
OUT=/workspace/experiments/flagship-v1arch-v2bal-k20-30k
LOG=/workspace/v1k20_run.log
ERR=/workspace/v1k20_run.err
ST=/workspace/v1k20_run.status
CORPUS=/workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d

# ⛔ PREFLIGHT: refuse on a truncated corpus. pod3's copy was 4047/9000 and the parity guard would
# NOT have caught it (v2bal has no registered parity key -- it only WARNS). A silently smaller
# corpus destroys comparability with the k=4 arm, which is the whole point of this run.
N=$(ls "$CORPUS" 2>/dev/null | grep -c 'v2ep.pt$')
if [ "$N" -ne 9000 ]; then
  echo "=== v1k20 REFUSED $(date -u +%FT%TZ): corpus has $N/9000 clips ===" >> "$ST"
  exit 91
fi
mkdir -p "$OUT"
echo "=== v1k20 START $(date -u +%FT%TZ) corpus=$N clips ===" >> "$ST"
cd /workspace/TanitAD/stack || exit 90
export PYTHONPATH=/workspace/TanitAD/stack
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=6
python3 -u scripts/train_flagship4b.py   --v2-cache "$CORPUS"   --config flagship4b --sigreg-free-dims 64   --rollout-k 20 --speed-input   --steps 30000 --batch-size 16 --accum 4 --grad-checkpoint   --lr 3e-4 --warmup 2000 --workers 8 --v2-lru 64 --guard-limit-gb 45   --ckpt-every 1000 --log-every 50   --out "$OUT" >> "$LOG" 2>> "$ERR"
rc=$?
{ echo "=== v1k20 EXIT $(date -u +%FT%TZ) rc=$rc ==="
  [ $rc -gt 128 ] && echo "   killed by signal $((rc-128)) (137=SIGKILL/OOM)"
  echo "   last stderr:"; tail -20 "$ERR" | sed 's/^/     /'; } >> "$ST"
