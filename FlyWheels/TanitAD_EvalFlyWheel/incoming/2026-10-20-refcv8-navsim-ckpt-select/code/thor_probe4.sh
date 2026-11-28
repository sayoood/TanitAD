#!/bin/bash
cd /home/nvidia/tanit-astra-research/data/navsim_pilot_20261003
python3 - <<'PY'
import json
a=json.load(open("acquisition.json"))
print("KEYS", list(a.keys()))
for k,v in a["stages"].items():
    print("STAGE", k, json.dumps(v)[:1500])
for k in a:
    if k not in ("stages","segments"):
        print("TOP", k, json.dumps(a[k])[:600])
PY
echo "ZZ-PROBES"; cat current_probe/probe.json; echo; cat history_probe/probe.json; echo; cat metadata_probe/probe.json
