#!/usr/bin/env python3
"""Child for the process-level OFF-identity proof through the REAL worker path  online_labeller._job("pdm", ...).
  python job_identity_child.py <refe_dir: tip or overlay> <scenes_dir> <out.npz> <behaviour 0|1> [--mutate-navtest]
Runs _pdm_init(parts_root, behaviour=...) and _job over every scene's seeded fan; saves the label arrays (and the behaviour arrays when ON)."""
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
NAVSIM = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
refe, scenes, out, beh = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
sys.path[:0] = [NAVSIM, refe]
import numpy as np  # noqa: E402
import online_labeller as OL  # noqa: E402

assert os.path.abspath(OL.__file__).startswith(os.path.abspath(refe)), (OL.__file__, refe)
if beh:
    OL._pdm_init(os.path.join(scenes, "parts"), behaviour=True)
else:
    OL._pdm_init(os.path.join(scenes, "parts"))                      # the PRE-A2 call signature: works on the tip AND the overlay
states = json.load(open(os.path.join(scenes, "states.json")))
res = {}
for k, st in enumerate(states):
    fp = os.path.join(scenes, f"fan_{k:02d}.npy")
    if not os.path.isfile(fp):
        continue
    r = OL._job("pdm", {"log_name": st["log"], "token": st["token"]}, np.load(fp))
    if r["labels"] is None:
        continue
    res[f"labels_{k:02d}"] = r["labels"]
    if beh:
        res[f"beh_{k:02d}"], res[f"act_{k:02d}"] = r["behaviour"], r["behaviour_act"]
np.savez(out, **res)
print("child done", len(res), "arrays", "tip-or-overlay:", os.path.dirname(OL.__file__), flush=True)
