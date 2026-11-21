#!/bin/bash
cd /home/nvidia/tanit-astra-research/data/navsim_pilot_20261003
echo "ZZ-META"; ls -la metadata metadata_probe | head -40
echo "ZZ-CUR"; ls -la current_probe history_probe | head -30
echo "ZZ-ACQ"; python3 - <<'PY'
import json
a=json.load(open("acquisition.json"))
def walk(o,d=0,k=""):
    if d>2: return
    if isinstance(o,dict):
        for kk,v in list(o.items())[:25]:
            s=json.dumps(v)[:200]
            print("  "*d+f"{kk}: {s}")
            if isinstance(v,(dict,list)) and d<1: walk(v,d+1,kk)
    elif isinstance(o,list):
        print("  "*d+f"[list len {len(o)}] first: {json.dumps(o[0])[:300] if o else None}")
walk(a)
PY
cd ../navsim_mini_front_20261003; echo "ZZ-MINI"; ls sensors | head; ls metadata | head -5; ls metadata | wc -l
echo "ZZ-END"
