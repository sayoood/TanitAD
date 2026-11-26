#!/usr/bin/env bash
# TRK-SCORE unit tests in the SEAM's venv (driverl-eval + the DriveRL nuplan fork, as stage2_sft.py runs the seam), CPU only.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
DZ=C:/Users/Admin/dz/DriveZero/DriveRL
export TMP=E:/Projects/TanitAD-artifacts/trkscore/work/tmp TEMP=E:/Projects/TanitAD-artifacts/trkscore/work/tmp
mkdir -p "$TMP"
export CUDA_VISIBLE_DEVICES=-1 NUPLAN_MAPS_ROOT=E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0 REFE_BACKBONE_ROOT=E:/Projects/TanitAD/data/backbones
export PYTHONPATH="$DZ/nuplan-devkit;$DZ/src" PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export TRK_PKG="F:/TanitAD_p5cread_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
export TRK_NAV="F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
cd "$HERE" && C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe test_trk_score.py < /dev/null
