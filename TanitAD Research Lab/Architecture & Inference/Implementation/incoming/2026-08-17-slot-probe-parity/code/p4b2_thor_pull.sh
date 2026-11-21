#!/usr/bin/env bash
# Thor-sourced pull, v2. READ ONLY on Thor: `ssh -n cat` — no compute, no python.
# ⚠️ `-n` because a nested ssh inside a loop EATS THE LOOP'S STDIN (CLAUDE.md).
# Errors are LOGGED, never swallowed: v1 hid them behind `-q 2>/dev/null` and
# reported 47 identical MISSes with no cause.
set -u
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/sp2"
D="$S/cache/slotprobe-lead130-w120-256x640cyl"
R="/home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl"
until [ -f "$S/ck/v6F_sw_step009250.fp16.pt.local.md5" ] && [ -f "$S/ck/v6F_sw_step010000.fp16.pt.local.md5" ]; do sleep 15; done
mapfile -t CLIPS < "$S/thor_order.txt"
n=0
for cid in "${CLIPS[@]}"; do
  [ -z "$cid" ] && continue
  [ -s "$D/$cid.v2ep.pt" ] && continue
  [ "$(ls "$D"/*.v2ep.pt 2>/dev/null | wc -l)" -ge 130 ] && break
  if ssh -n -o BatchMode=yes -o ConnectTimeout=30 tanitad-thor-wifi "cat $R/$cid.v2ep.pt" > "$D/.t.$cid.tmp"; then
    sz=$(stat -c %s "$D/.t.$cid.tmp")
    if [ "$sz" -gt 1000000 ]; then
      mv -f "$D/.t.$cid.tmp" "$D/$cid.v2ep.pt"; n=$((n+1))
      echo "[p4b2] $n  $cid  ${sz}B  total=$(ls "$D"/*.v2ep.pt | wc -l)/130"
    else
      rm -f "$D/.t.$cid.tmp"; echo "[p4b2] SHORT $cid $sz"
    fi
  else
    rm -f "$D/.t.$cid.tmp"; echo "[p4b2] FAIL $cid exit=$?"
  fi
done
echo "[p4b2] DONE pulled=$n total=$(ls "$D"/*.v2ep.pt | wc -l)"
