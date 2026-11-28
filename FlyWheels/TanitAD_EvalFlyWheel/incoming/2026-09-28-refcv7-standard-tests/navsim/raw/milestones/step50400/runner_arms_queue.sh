#!/bin/bash
# the runner's still-unscored arms, through the governed sequential queue (one at a time). Idempotent:
# PASS arms are skipped; a duplicate runner launch waits on the per-job lock and then skips.
export PYTHONPATH="C:/Users/Admin/ev7nav/stack;C:/Users/Admin/ev7nav/taniteval" TANITAD_REPO="C:/Users/Admin/ev7nav" PYTHONIOENCODING=utf-8
PYV="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"; PKG="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim"; M="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400"
$PYV $PKG/code/score_queue_gov7.py --split navtest --bridge $M/bridge_navtest --scores $M/scores_navtest --step 50400 --arms R7_A1,R7_A1_s1
$PYV $PKG/code/score_queue_gov7.py --split navhard --bridge $M/bridge_navhard --scores $M/scores_navhard --step 50400 --arms PRIOR_ha0p,R7_CEILDECL_d
$PYV $PKG/code/score_queue_gov7.py --split navtest --bridge $M/bridge_navtest --scores $M/scores_navtest --step 50400 --arms PRIOR_ha0p,R7_CEILDECL_d
