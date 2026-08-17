#!/usr/bin/env bash
set -u
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/sp2"
export PYTHONUTF8=1 OMP_NUM_THREADS=6
until [ "$(ls "$S/cache/slotprobe-lead130-w120-256x640cyl/"*.v2ep.pt 2>/dev/null | wc -l)" -ge 130 ]; do sleep 30; done
until [ -f "$S/ck/v6F_sw_step010000.fp16.pt.local.md5" ]; do sleep 30; done
echo "=== [chainC] trajectory @10000 $(date +%H:%M:%S)"
bash "$S/run_point.sh" s10000 "$S/ck/v6F_sw_step010000.fp16.pt" 4 0 74
until [ -f "$S/ck/v6F_sw_step009250.fp16.pt.local.md5" ]; do sleep 30; done
echo "=== [chainC] trajectory @9250 $(date +%H:%M:%S)"
bash "$S/run_point.sh" s09250 "$S/ck/v6F_sw_step009250.fp16.pt" 4 0 74
echo "=== [chainC] CHAINCDONE"
