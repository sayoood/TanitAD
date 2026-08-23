#!/usr/bin/env bash
set -u
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/sp2"
D="$S/cache/slotprobe-lead130-w120-256x640cyl"
R="/home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl"
for cid in edd51867-8095-4b78-9033-3cda04d1df0a 3da3657f-e7b1-49cd-b876-9b05eff72937 \
           5c96137b-1aa2-42c8-ba6e-7ad0d6c351d5 681505d8-4a59-4fc3-be0c-a8610181e55b \
           695697d2-7cce-4977-becb-0a8642522cf2 6cdcb212-9586-4588-89a3-6aa40cfb2a0d; do
  [ -s "$D/$cid.v2ep.pt" ] && { echo "[p4c] have $cid"; continue; }
  rm=$(ssh -n -o BatchMode=yes tanitad-thor-wifi "md5sum $R/$cid.v2ep.pt" 2>&1 | awk '{print $1}')
  scp -o BatchMode=yes "tanitad-thor-wifi:$R/$cid.v2ep.pt" "$D/.c.$cid.tmp"
  rc=$?
  if [ $rc -ne 0 ]; then echo "[p4c] SCPFAIL $cid rc=$rc"; rm -f "$D/.c.$cid.tmp"; continue; fi
  lm=$(md5sum "$D/.c.$cid.tmp" | awk '{print $1}')
  if [ "$lm" = "$rm" ]; then mv -f "$D/.c.$cid.tmp" "$D/$cid.v2ep.pt"; echo "[p4c] OK $cid md5=$lm  total=$(ls "$D"/*.v2ep.pt | wc -l)/130";
  else echo "[p4c] MD5BAD $cid local=$lm remote=$rm"; rm -f "$D/.c.$cid.tmp"; fi
done
echo "[p4c] DONE total=$(ls "$D"/*.v2ep.pt | wc -l)/130"
