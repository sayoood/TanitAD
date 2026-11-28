#!/usr/bin/env python3
"""make_manifest.py -- MANIFEST.txt (path, bytes, sha256) of every deliverable: the package under E: (skipping aside/ and __pycache__) and the big arrays under F:/TanitAD-scratch/trigger_probe3_scratch/.
Also writes raw/INPUTS_SHA256.json (the read-only inputs).   python -I make_manifest.py"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import g_lib as G  # noqa: E402


def main():
    inputs = {}
    for p in (f"{G.OLDRAW}/windows_train.npz", f"{G.OLDRAW}/sample_pstop.npz", f"{G.OLDRAW}/sample_pmove.npz", f"{G.OLDRAW}/eval141_features.npz", f"{G.OLDRAW}/oof_pstop_V_linear.npz", f"{G.OLDRAW}/oof_pmove_V_linear.npz",
              f"{G.OLDRAW}/oof_pstop_V_mlp.npz", f"{G.OLDRAW}/oof_pmove_V_mlp.npz", f"{G.OLD}/tp_lib.py", f"{G.OLDSCR}/V_eval141.npy"):
        inputs[p] = G.sha256_file(p)
    json.dump(inputs, open(f"{G.RAW}/INPUTS_SHA256.json", "w"), indent=1)
    lines = []
    for root in (G.HERE, G.SCR):
        for dp, dn, fn in os.walk(root):
            dn[:] = [x for x in dn if x not in ("aside", "__pycache__", ".pytest_cache")]
            for f in sorted(fn):
                p = os.path.join(dp, f).replace("\\", "/")
                if f in ("MANIFEST.txt",):
                    continue
                lines.append(f"{G.sha256_file(p)}  {os.path.getsize(p):>12d}  {p}")
    open(f"{G.HERE}/MANIFEST.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(len(lines), "files")


if __name__ == "__main__":
    main()
