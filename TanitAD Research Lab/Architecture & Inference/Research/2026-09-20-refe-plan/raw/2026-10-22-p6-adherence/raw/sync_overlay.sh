#!/usr/bin/env bash
# Copies the P6-ADHERENCE files over the mirror-tip overlay (a9bc000: taniteval/, stack/tanitad/, PK/eval/, PK/refe/) so the instrument runs exactly as the MM
# re-runs package tests ("an overlay of the mirror tip, with the package's files on top"). Never touches a git tree.
set -e
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
P=C:/Users/Admin/par18/p6adh
for f in eval/command_adherence.py eval/test_command_adherence.py; do
  [ -f "$P/code/$PK/$f" ] && cp "$P/code/$PK/$f" "$P/work/overlay/$PK/$f"
done
echo overlay synced
