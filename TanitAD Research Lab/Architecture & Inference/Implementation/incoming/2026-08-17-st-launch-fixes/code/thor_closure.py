"""FULL import closure of the S-T launch = static + the trainer's LAZY imports.

The lazy ones were read out of train_v6_staged.py itself (grep for indented
`from`/`import`), so the list is derived from source, not remembered.
"""
import pyarrow  # noqa: F401  # MUST precede torch on the dev box
import importlib
import json
import os
import sys

REPO = os.path.abspath(sys.argv[1])
STACK = os.path.join(REPO, "stack")
TE = os.path.join(REPO, "taniteval")
sys.path.insert(0, os.path.join(STACK, "scripts"))
sys.path.insert(0, STACK)
sys.path.insert(0, TE)

MODULES = [
    # entry points
    "train_v6_staged", "v6_chain",
    # lazy, from train_v6_staged source
    "tanitad.models.metric_dynamics", "tanitad.models.flagship_v15",
    "s2_labels", "eval_flagship_v4", "train_flagship4b", "train_flagship_v4",
    "train_v58f_unicycle_head",
    # the seam instrument
    "taniteval.seam", "taniteval.seam_dump", "taniteval.ci",
    # the gate battery the S-W gate needs
    "probe_latent_state", "stage_a_probes",
]

errs = {}
for m in MODULES:
    try:
        importlib.import_module(m)
    except BaseException as e:  # noqa: BLE001
        errs[m] = f"{type(e).__name__}: {str(e)[:200]}"

files = set()
for mod in list(sys.modules.values()):
    f = getattr(mod, "__file__", None)
    if not f:
        continue
    f = os.path.abspath(f)
    if f.lower().startswith(REPO.lower()) and f.endswith(".py"):
        files.add(os.path.relpath(f, REPO).replace("\\", "/"))

print("ZZJSONZZ" + json.dumps({"errors": errs, "n": len(files),
                               "files": sorted(files)}) + "ZZENDZZ")
