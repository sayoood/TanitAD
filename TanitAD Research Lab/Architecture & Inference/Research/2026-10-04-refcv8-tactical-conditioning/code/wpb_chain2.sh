#!/bin/bash
# wpb_chain2: the chain with TREE parameterised (2026-10-04: re-run after the I-W split-decode fix; the failed
# unsplit I-W artifact is kept as arms/I_W.FAILED_unsplit.json, never overwritten).
# SPEC_WPB (REGISTERED c952d4b4..., 2026-10-04T12:33:09Z) -- the WP-B chain on Thor.
# R1 is the substrate and runs FIRST (registration note 1): this chain waits for R1's arms to finish (arms/score.json), or
# for R1 to be gone with its caches present. Then: I-W -> timing -> arms in SPEC sec. 3 order, every GPU pass its own
# flock acquisition (<= ~40 min, interleaving with WP-D / WP-RL). The ARTIFACT, never the exit code, says a pass ran.
set -u
W=/home/nvidia/refcv8_wpb
R1=/home/nvidia/refcv8_r1
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
export REFCV6_REPO=$W/${TREE:-tree} REFCV6_KIT=/home/nvidia
export PYTHONPATH=$W/${TREE:-tree}/stack:$W/${TREE:-tree}/taniteval
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
A=$W/arms
mkdir -p $A $W/logs $W/status
cd $W/code
free_gb () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
# ---- wait for R1 ------------------------------------------------------------------------------------------------ #
for i in $(seq 1 1440); do
  if [ -s $R1/arms/score.json ]; then echo "[wpb] $(date -u +%H:%M:%S) R1 done (score.json)"; break; fi
  # the bracket keeps grep from matching its OWN command line (the pgrep self-match trap, CLAUDE.md)
  if ! ps -eo args | grep -q "r[1]_\(capture\|arms\|chain\)" ; then
    if [ -s $R1/cache/eval/record.json ] && [ -s $R1/cache/train/record.json ] && [ -s $R1/arms/timing_smoke.json ]; then
      echo "[wpb] $(date -u +%H:%M:%S) R1 is not running, its caches + timing exist -> proceeding"; break
    fi
    echo "[wpb] $(date -u +%H:%M:%S) R1 is NOT running and its artifacts are incomplete -- stopping"; exit 1
  fi
  sleep 60
done
[ -s $R1/cache/eval/record.json ] || { echo "[wpb] no R1 eval cache -- stopping"; exit 1; }
gpu () {   # tag artifact attempts args...
  local tag=$1 art=$2 n=$3; shift 3
  local i
  for i in $(seq 1 $n); do
    if [ -s "$art" ]; then echo "[wpb] $tag banked"; return 0; fi
    if [ "$(free_gb)" -lt 20 ]; then echo "[wpb] $tag REFUSED: $(free_gb) GB free"; return 1; fi
    echo "[wpb] $(date -u +%H:%M:%S) start $tag attempt $i"
    flock $LOCK $PY wpb_arms.py "$@" >> $W/logs/arm_$tag.log 2>&1 < /dev/null
    echo $? > $W/status/arm_$tag.exit
    echo "[wpb] $(date -u +%H:%M:%S) end $tag rc=$(cat $W/status/arm_$tag.exit) artifact=$( [ -s "$art" ] && echo yes || echo NO )"
    sleep 20
  done
  [ -s "$art" ]
}
gpu IW $A/I_W.json 2 --phase iw || exit 1
$PY -c "import json,sys; sys.exit(0 if json.load(open('$A/I_W.json'))['PASS'] else 1)" \
  || { echo "[wpb] I-W FAILED -- no arm runs (SPEC sec. 2)"; exit 1; }
gpu timing $A/timing_wpb.json 2 --phase timing --timing-steps 30 || exit 1
for arm in T0 T1 T1d T2 T2d T0r X1h T2s X2a X2b; do
  gpu ${arm}_train $A/$arm/train_record.json 2 --phase train --arm $arm || { echo "[wpb] $arm train failed -- next arm"; continue; }
  gpu ${arm}_eval0 $A/$arm/eval_s0.pt 2 --phase eval --arm $arm --eval-seeds 0 --ctrl || continue
  gpu ${arm}_eval1 $A/$arm/eval_s1.pt 2 --phase eval --arm $arm --eval-seeds 1 --ctrl || continue
done
echo "[wpb] ZZALLDONEZZ $(date -u +%H:%M:%S)"
