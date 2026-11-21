import json, sys, hashlib
import numpy as np, torch
p = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt"
o = torch.load(p, map_location="cpu", weights_only=True)
print("keys", list(o.keys()))
tr = o["tracks"]
print("n tracks", len(tr))
names = list(tr.keys())[:3]
for n in names: print(n, tr[n].shape, tr[n].dtype)
m = json.load(open("E:/Projects/TanitAD-artifacts/v8f-dagger/bank_s0/bank.json"))
tclips = m["common"]["train"]
print("train key type", type(tclips), (list(tclips.keys())[:5] if isinstance(tclips, dict) else len(tclips)))
if isinstance(tclips, dict):
    for k, v in tclips.items():
        print(k, (len(v) if hasattr(v, '__len__') else v), str(v)[:100])
print("sha256 tracks", hashlib.sha256(open(p,'rb').read()).hexdigest())
print("bank inputs_sha256", m["common"]["inputs_sha256"])
