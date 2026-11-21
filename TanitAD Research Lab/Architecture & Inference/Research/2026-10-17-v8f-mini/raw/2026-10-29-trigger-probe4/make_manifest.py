#!/usr/bin/env python3
"""make_manifest.py -- MANIFEST.txt (sha256 + size of every file of the package) and raw/INPUTS_SHA256.json (the inputs this probe reads, hashed).  python -I make_manifest.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import p4_lib as P

H = P.HERE
rows = []
for root, dirs, files in os.walk(H):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache")]
    for f in sorted(files):
        p = os.path.join(root, f).replace("\\", "/")
        if f == "MANIFEST.txt" or f.endswith(".tmp") or f.endswith(".tmp.npz"):
            continue
        rows.append((os.path.relpath(p, H).replace("\\", "/"), os.path.getsize(p), P.sha256_file(p)))
open(f"{H}/MANIFEST.txt", "w", newline="\n").write("# sha256  bytes  path (relative to trigger_probe4/)\n" + "\n".join(f"{h}  {n}  {r}" for r, n, h in rows) + "\n")
inp = {}
for k, p in {"windows_train.npz": f"{P.OLDRAW}/windows_train.npz", "sample_pstop.npz": f"{P.OLDRAW}/sample_pstop.npz", "sample_pmove.npz": f"{P.OLDRAW}/sample_pmove.npz",
             "eval141_features.npz": f"{P.OLDRAW}/eval141_features.npz", "probe1 oof_pstop_V_linear": f"{P.OLDRAW}/oof_pstop_V_linear.npz", "probe1 oof_pmove_V_linear": f"{P.OLDRAW}/oof_pmove_V_linear.npz",
             "probe1 oof_pstop_V_mlp": f"{P.OLDRAW}/oof_pstop_V_mlp.npz", "probe1 oof_pmove_V_mlp": f"{P.OLDRAW}/oof_pmove_V_mlp.npz", "probe3 oofg_pstop_linear": f"{P.P3RAW}/oofg_pstop_linear.npz",
             "probe3 oofg_pmove_linear": f"{P.P3RAW}/oofg_pmove_linear.npz", "probe3 pcaG_pstop.json": f"{P.P3RAW}/pcaG_pstop.json", "probe3 pcaG_pmove.json": f"{P.P3RAW}/pcaG_pmove.json",
             "probe3 S8_pstop": f"{P.P3SCR}/S8_pstop.npy", "probe3 S8_pmove": f"{P.P3SCR}/S8_pmove.npy", "probe3 S4_pstop": f"{P.P3SCR}/S4_pstop.npy", "probe3 S4_pmove": f"{P.P3SCR}/S4_pmove.npy",
             "probe3 tp_lib.py": f"{P.P3}/tp_lib.py"}.items():
    inp[k] = {"path": p, "sha256": P.sha256_file(p) if os.path.exists(p) else None}
json.dump(inp, open(f"{P.RAW}/INPUTS_SHA256.json", "w"), indent=1)
print("manifest rows", len(rows))
