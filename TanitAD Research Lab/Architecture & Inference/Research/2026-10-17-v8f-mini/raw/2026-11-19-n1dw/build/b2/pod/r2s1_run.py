"""r2s1_run.py A|B -- the seed-matched R2 repeat for the s1 arms (coordinator review R-N1DW-MINT item 1): the banked N1D_s1 argv (w = 1) run with --selfcheck-only on the n1dw tree,
GPU 0, the same call mint makes (strat1_check._run). A copy of b2/r2_run.py with N1D_s0 -> N1D_s1 and its own output folder. Output /root/strat1/n1dw/b2/r2s1/N1D_<A|B>/selfcheck.json."""
import json
import os
import sys

TREE = "/root/strat1/n1dw/tree"
sys.path.insert(0, os.path.join(TREE, "stack", "scripts"))
sys.path.insert(0, os.path.join(TREE, "stack"))
import strat1_check as S  # noqa: E402

assert os.path.realpath(S.__file__).startswith(TREE + "/stack/"), S.__file__
r = sys.argv[1]
toks = json.load(open("/root/strat1/navrbd/runs/N1D_s1/argv.json", encoding="utf-8"))
out = f"/root/strat1/n1dw/b2/r2s1/N1D_{r}"
toks = S._with(toks, out=out)
assert "--rb-window-w" not in toks
env = {"PYTHONPATH": os.path.join(TREE, "stack") + os.pathsep + TREE, "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_DISABLE_XET": "1"}
rc = S._run("/root/venv/bin/python", os.path.join(TREE, S.TRAINER_REL), toks + ["--selfcheck-only"], TREE, env, out + ".log")
print("rc", rc, os.path.isfile(os.path.join(out, "selfcheck.json")))
sys.exit(rc)
