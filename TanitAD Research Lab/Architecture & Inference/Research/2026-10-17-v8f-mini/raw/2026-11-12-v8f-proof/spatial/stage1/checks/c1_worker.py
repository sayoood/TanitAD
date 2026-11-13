"""c1 worker: runs inside ONE tree (PYTHONPATH set by the caller). Builds the model from the argv at init (seed 0) and dumps its state dict,
then runs the synthetic trainer for N steps. Usage: c1_worker.py <tree_root> <out_dir> <arm> <steps>"""
import json
import os
import sys

tree, out, arm, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
import numpy as np
import torch
import tanitad

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(tree).lower()), (tanitad.__file__, tree)
sys.path.insert(0, os.path.join(tree, "stack", "scripts"))
import v8f_mini_train as TR
from tanitad.models import v8f_mini as M

assert os.path.abspath(TR.__file__).lower().startswith(os.path.abspath(tree).lower()), TR.__file__
os.makedirs(out, exist_ok=True)
base = ["--synthetic", "--arm", arm, "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1", "--seed", "0"]
a = TR.build_parser().parse_args(["--out", out, *base])
cfg = TR.make_config(a)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
model = M.V8FMini(cfg, M.P10Prior.synthetic(a.seed), torch.zeros(M.N_CONS), torch.ones(M.N_CONS))
sd = model.state_dict()
torch.save({k: v.clone() for k, v in sd.items()}, os.path.join(out, "init_state.pt"))
json.dump({"keys": list(sd.keys()), "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
           "tanitad_file": tanitad.__file__}, open(os.path.join(out, "init_keys.json"), "w", encoding="utf-8"))
rc = TR.main([*base, "--out", os.path.join(out, "run"), "--steps", str(steps), "--log-every", "1"])
print("RC", rc)
