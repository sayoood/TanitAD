#!/usr/bin/env python3
"""Launch the v6F S-W resume on Thor by REPLAYING the banked config's own args.

The strict resume contract (train_v6_staged.load_resume -> load_state_dict(strict=True))
is tensor-level: ARCHITECTURE flags must match the checkpoint exactly; path flags are
free. So instead of retyping ~90 flags (the exact failure mode --heldout-off-reason's
history warns about: retyping loses arguments), this reads
~/experiments/v6F-SW-30k/config.json["args"] and rebuilds argv verbatim, swapping ONLY:

    out          -> ~/experiments/v6F-SW-30k      (contains the pulled ckpt.pt)
    v2_cache     -> [~/data/physicalai-train-e438721ae894-w120-256x640cyl]
    v2_val_cache -> [~/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl]
    resume       -> auto   (already the banked value; forced for clarity)

Reconstruction rules (checked against the trainer's argparse):
  bool True  -> bare flag        bool False -> omitted (all False-valued bools here
                                               are store_true defaults)
  None       -> omitted          '' (empty string) -> omitted
  list       -> flag + one token per element (--horizons/--v2-cache are nargs='+')
  else       -> flag + str(value)
"""
import json
import os
import sys

HOME = os.path.expanduser("~")
OUT = os.path.join(HOME, "experiments", "v6F-SW-30k")
CFG = os.path.join(OUT, "config.json")
TRAIN = os.path.join(HOME, "TanitAD", "stack")
PY = os.path.join(HOME, "venvs", "tanitad-train", "bin", "python")

args = json.load(open(CFG))["args"]

args["out"] = OUT
args["v2_cache"] = [os.path.join(
    HOME, "data", "physicalai-train-e438721ae894-w120-256x640cyl")]
args["v2_val_cache"] = [os.path.join(
    HOME, "data", "physicalai-val-0c5f7dac3b11-w120-256x640cyl")]
args["resume"] = "auto"

# The checkpoint must already sit in --out, or the resume_guard falls through to
# a fresh start ON TOP of nothing — refuse loudly instead.
ck = os.path.join(OUT, "ckpt.pt")
assert os.path.exists(ck), f"REFUSING TO LAUNCH: no checkpoint at {ck}"

argv = [PY, "-u", os.path.join(TRAIN, "scripts", "train_v6_staged.py")]
for k, v in args.items():
    flag = "--" + k.replace("_", "-")
    if v is None or v == "" or v is False:
        continue
    if v is True:
        argv.append(flag)
    elif isinstance(v, list):
        argv.append(flag)
        argv.extend(str(x) for x in v)
    else:
        argv.extend([flag, str(v)])

print("[launch] argv has", len(argv), "tokens; stage",
      args.get("stage"), "resume", args.get("resume"), flush=True)
os.environ["PYTHONPATH"] = TRAIN + ":" + os.path.join(TRAIN, "scripts")
os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.stdout.flush()
os.execve(PY, argv, os.environ)
