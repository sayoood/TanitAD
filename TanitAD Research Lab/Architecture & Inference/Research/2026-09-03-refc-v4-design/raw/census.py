"""E-REFC-V4 corpus census — the design inputs, MEASURED and banked.

Reproduces the numbers PREREG_REFC_V4.md cites, from the val epcache, through
``tanitad.eval.echo_gate.corpus_echo_report`` (ONE implementation, shared with
the gate and with the model's own E14 base).

⚠️ Streams one episode at a time and keeps ONLY poses/actions: the frame buffer
is 117 MB/ep and the tiny-rig panel is holding ~6 GB of host RAM beside this.
"""
import glob
import json
import os
import sys
import types

import torch

from tanitad.eval.echo_gate import corpus_echo_report

ROOT = sys.argv[1]
OUT = sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0

files = sorted(glob.glob(os.path.join(ROOT, "ep_*.pt")))
if LIMIT:
    files = files[:LIMIT]
eps = []
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    eps.append(types.SimpleNamespace(poses=d["poses"].numpy(),
                                     actions=d["actions"].numpy()))
    del d
rep = corpus_echo_report(eps, horizons_s=(2.0, 6.0), stride=7)
rep["_provenance"] = {"root": ROOT, "n_files": len(files),
                      "instrument": "tanitad.eval.echo_gate.corpus_echo_report",
                      "evidence_class": "MEASURED"}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(rep, fh, indent=1)
print(json.dumps(rep, indent=1)[:4000])
