#!/usr/bin/env bash
# Wait for a refcv6 checkpoint on Thor, then run the whole NavSim suite on it, UNATTENDED.
#   bash code/milestone_waiter.sh 30000      # a milestone (model-only ckpt_<N>.pt or snapshot ckpt_step<N>.pt)
#   bash code/milestone_waiter.sh final      # the FINAL ckpt.pt, once the run's summary.json reads done: true
# ⛔ Thor access is READ-ONLY: `ssh -n … stat` / `cat summary.json` / `md5sum`, then scp (the runner's
# --fetch / --fetch-final: md5 on Thor, and for the final before AND after the copy, against the
# dev box). Nothing is started on Thor.
# ORDER: warmup → navtest (+ SPEC §12 diagnostic arms on W3's 200 tokens) → navhard; navtest waits
# for the 416 bank only if it is incomplete (it has been complete since 2026-09-24).
# Each phase is its own runner invocation into the SAME milestone dir; the runner MERGES
# MILESTONE_SUMMARY.json and bars6 re-evaluates every split present. After every phase the outputs
# are COPIED to the D: package (code/stage_to_repo.py: copy-only, content-verified, NO git — the
# Master Mind is the single committer). Unattended outputs are NOT declared landing-ready: an agent
# verifies them first (count guards, stand-ins, device stamps, seed replicate, harness controls).
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
STEP="$1"
RUN=/home/nvidia/refcv6_run/runs/refcv6-r101-s0
KIT=D:/refcv6_eval_kit/ckpt
L="$P/raw/milestones/waiter_${STEP}.log"
BANK=D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank
mkdir -p "$P/raw/milestones"
if [ "$STEP" = "final" ]; then
  echo "WAIT $RUN/summary.json done:true $(date -u +%FT%TZ)" >> "$L"
  until ssh -n -o BatchMode=yes -o ConnectTimeout=15 tanitad-thor-wifi "cat $RUN/summary.json" 2>/dev/null \
        | "$PY" -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('done') is True else 1)" 2>/dev/null; do
    sleep 600
  done
  echo "RUN DONE (summary.json done:true) $(date -u +%FT%TZ)" >> "$L"
  FETCH="--fetch-final"
else
  # the trainer's own model-only MILESTONES (5000/15000/20000/30000) land in the run dir as
  # ckpt_<N>.pt; the snapshot watcher's full copies (10k/25k/35k/40k/45k) as ckpt_step<N>.pt
  F="$RUN/ckpt_${STEP}.pt"
  F2="/home/nvidia/refcv6_run/snapshots/refcv6-r101-s0/ckpt_step${STEP}.pt"
  echo "WAIT $F $(date -u +%FT%TZ)" >> "$L"
  prev=""
  while :; do
    s=$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 tanitad-thor-wifi "stat -c %s $F 2>/dev/null || stat -c %s $F2 2>/dev/null" 2>/dev/null)
    if [ -n "$s" ] && [ "$s" = "$prev" ]; then break; fi
    prev="$s"; sleep 120
  done
  echo "PRESENT size=$s $(date -u +%FT%TZ)" >> "$L"
  FETCH="--fetch $STEP"
fi
phase() {   # $1 = split
  "$PY" "$P/code/run_navsim_refcv6.py" $FETCH --splits "$1" --device auto --gpu-wait-s 900 >> "$L" 2>&1
  echo "RUNNER_$1_EXIT $? $(date -u +%FT%TZ)" >> "$L"
  "$PY" "$P/code/stage_to_repo.py" >> "$L" 2>&1
}
local_ckpt() {
  if [ "$STEP" = "final" ]; then ls -t "$KIT"/ckpt_final_step*.pt 2>/dev/null | head -1; else echo "$KIT/ckpt_${STEP}.pt"; fi
}
bank_done() { [ "$(ls "$BANK"/shard_*.DONE.json 2>/dev/null | wc -l)" -ge 32 ]; }
diag() {
  CK=$(local_ckpt)
  if [ -n "$CK" ] && [ -s "$CK" ]; then
    bash "$P/code/navtest_diag.sh" "$CK" "$P/raw/milestones/step${STEP}_navtest_diag"
    echo "NAVTEST_DIAG_DONE $(date -u +%FT%TZ)" >> "$L"
    "$PY" "$P/code/stage_to_repo.py" >> "$L" 2>&1
  else
    echo "NAVTEST_DIAG_SKIPPED no local checkpoint $(date -u +%FT%TZ)" >> "$L"
  fi
}
phase warmup
deferred=1
if bank_done; then phase navtest; diag; deferred=0; else echo "NAVTEST_DEFERRED bank not complete $(date -u +%FT%TZ)" >> "$L"; fi
phase navhard
if [ $deferred -eq 1 ]; then
  t=0
  until bank_done || [ $t -ge 43200 ]; do sleep 300; t=$((t+300)); done
  if bank_done; then phase navtest; diag; else
    echo "NAVTEST_SKIPPED bank shards DONE=$(ls "$BANK"/shard_*.DONE.json 2>/dev/null | wc -l)/32 after 12 h $(date -u +%FT%TZ)" >> "$L"
  fi
fi
echo "ZZMILESTONE${STEP}DONEZZ $(date -u +%FT%TZ)" >> "$L"
