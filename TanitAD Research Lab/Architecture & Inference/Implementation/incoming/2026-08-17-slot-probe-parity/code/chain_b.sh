#!/usr/bin/env bash
set -u
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/sp2"
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/slotprobe"
export PYTHONUTF8=1 OMP_NUM_THREADS=6
until [ "$(ls "$S/cache/slotprobe-lead130-w120-256x640cyl/"*.v2ep.pt 2>/dev/null | wc -l)" -ge 130 ]; do sleep 30; done
echo "=== [chainB] trajectory @9000 $(date +%H:%M:%S)"
bash "$S/run_point.sh" s09000 "$SP/w9000/weights_fp16_s9000.pt" 4 0 74
echo "=== [chainB] trajectory @2000 $(date +%H:%M:%S)"
bash "$S/run_point.sh" s02000 "$SP/w/weights_fp16.pt" 4 0 74
echo "=== [chainB] CHAINBDONE"
