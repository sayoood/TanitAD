#!/bin/bash
# SMOKE (not a result): does the REAL chain refuse to start on a half-pulled run, wait, and start only when the pull is complete?
#   1. stage the stand-in run, then put it in a half-pulled state: SHA256SUMS.txt withheld, ckpt_last.pt cut to 40 MB
#   2. start the chain (SMOKE, 24 windows, own OUT / LOG): it must log 'waiting for the pull' and start NO rollout
#   3. put SHA256SUMS.txt back while the ckpt is STILL half (a stale/early sums file): still no rollout
#   4. complete the ckpt: the chain must see COMPLETE and start both rollouts
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
H=/c/Users/Admin/orct1; W=$H/smoke/halfpull; HP=/e/Projects/TanitAD-artifacts/orct1/smoke_art_half
PYX=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
mkdir -p "$W"
"$PYX" -B "$H/smoke/make_smoke_standin.py" "E:/Projects/TanitAD-artifacts/orct1/smoke_art_half" > "$W/stage.out" 2>&1 || { echo "stage failed"; exit 1; }
R=$HP/M2b_s0_orc
mv "$R/SHA256SUMS.txt" "$W/SHA256SUMS.txt.held"
mv "$R/ckpt_last.pt" "$W/ckpt_last.pt.full"
head -c 40000000 "$W/ckpt_last.pt.full" > "$R/ckpt_last.pt"
echo "$(date +%T) staged half-pulled state: no SHA256SUMS.txt, ckpt $(stat -c %s "$R/ckpt_last.pt") of $(stat -c %s "$W/ckpt_last.pt.full") bytes" | tee "$W/test.out"
ORCT_SMOKE=1 ORC_ART=E:/Projects/TanitAD-artifacts/orct1/smoke_art_half ORCT_OUT=$W/out ORCT_LOG=$W/logs bash "$H/orc_t1_chain.sh" > "$W/chain.stdout" 2>&1 &
CP=$!
sleep 25
echo "$(date +%T) [1] chain waiting? rollouts started: $(ls $W/out 2>/dev/null | grep -c '^t1v8f_\|^done_') ; log: $(grep -c 'start SMOKE_' $W/logs/chain.out) start lines; last: $(tail -1 $W/logs/chain.out | cut -c1-200)" | tee -a "$W/test.out"
cp "$W/SHA256SUMS.txt.held" "$R/SHA256SUMS.txt"
sleep 12
echo "$(date +%T) [2] SHA256SUMS.txt back, ckpt still half. start lines: $(grep -c 'start SMOKE_' $W/logs/chain.out); pullcheck says: $("$PYX" -B "$H/orct1_pullcheck.py" E:/Projects/TanitAD-artifacts/orct1/smoke_art_half orc 0 | cut -c1-200)" | tee -a "$W/test.out"
cp "$W/ckpt_last.pt.full" "$R/ckpt_last.pt"
echo "$(date +%T) [3] ckpt completed; waiting for the chain" | tee -a "$W/test.out"
wait $CP
echo "$(date +%T) [4] chain rc=$?" | tee -a "$W/test.out"
grep -E "waiting for the pull|is COMPLETE|start SMOKE_|end SMOKE_|ZZORCT1" "$W/logs/chain.out" | cut -c1-210 | tee -a "$W/test.out"
