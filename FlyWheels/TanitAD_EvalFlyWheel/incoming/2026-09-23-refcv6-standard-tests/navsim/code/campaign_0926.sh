#!/usr/bin/env bash
# 2026-09-26 resume campaign — the RESULTS lane (Master Mind's priority order), sequential,
# UNATTENDED-SAFE:
#   1. warmup @ step 5,000 — the split the unattended run left EMPTY (its CUDA bridge was refused by
#      the gate after 900 s and nothing fell back; fixed in the runner the same day);
#   2. the step-30,000 milestone (warmup → navtest + diagnostics → navhard) via milestone_waiter.sh.
# The precision controls run in their OWN lane (code/kp_lane.sh) so a closed GPU gate never delays
# a result. Every step is gated on its own ARTIFACT; nothing reads an exit code as evidence.
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
L="$P/raw/campaign_0926.log"
KIT=D:/refcv6_eval_kit/ckpt
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval"
echo "CAMPAIGN_START (results lane) $(date -u +%FT%TZ)" >> "$L"
B5="$P/raw/milestones/step5000/bridge_warmup"
[ -f "$B5/GPU_GATE_REFUSED.json" ] && mv "$B5/GPU_GATE_REFUSED.json" "$B5/GPU_GATE_REFUSED_20260924T0426Z.json"
"$PY" "$P/code/run_navsim_refcv6.py" --ckpt "$KIT/ckpt_5000.pt" --md5 8a1e4da0dc6561353b28e3947146751e \
   --splits warmup --device auto --gpu-wait-s 900 --out "$P/raw/milestones/step5000" >> "$L" 2>&1
[ -s "$P/raw/milestones/step5000/scores_warmup/score_R6_A1.counts.json" ] && echo "STEP1_OK $(date -u +%FT%TZ)" >> "$L" || echo "STEP1_NO_ARTIFACT $(date -u +%FT%TZ)" >> "$L"
"$PY" "$P/code/stage_to_repo.py" >> "$L" 2>&1
bash "$P/code/milestone_waiter.sh" 30000
grep -q "ZZMILESTONE30000DONEZZ" "$P/raw/milestones/waiter_30000.log" && echo "STEP2_OK $(date -u +%FT%TZ)" >> "$L" || echo "STEP2_NO_MARKER $(date -u +%FT%TZ)" >> "$L"
echo "ZZCAMPAIGN0926DONEZZ $(date -u +%FT%TZ)" >> "$L"
