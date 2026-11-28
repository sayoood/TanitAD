import sys, tempfile
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, "tests")
import torch
torch.set_num_threads(4)
import test_map_only_ft as R
from tanitad.train import map_only_ft as M
from tanitad.train import declared_vs_built as D
model, cfg = R._build()
print("declared", sorted(M.declared_prefixes(model)))
print("total", sum(1 for _ in model.parameters()), sum(int(p.numel()) for p in model.parameters()))
for sets in (("map",), ("planner",), ("planner", "map")):
    model, cfg = R._build()
    rec = M.apply_train_only(model, sets)
    print(sets, rec["n_trainable_tensors"], rec["n_trainable_params"], rec["n_frozen_tensors"], rec["n_frozen_params"],
          "declfrozen", rec.get("declared_frozen"), rec.get("n_declared_frozen_tensors"), rec.get("n_declared_frozen_params"),
          "norms", rec["frozen_norms_pinned"], "groups", len(rec["trainable_leaf_groups"]), sum(rec["trainable_leaf_groups"].values()))
    try:
        M.apply_train_only(R._build()[0], ("tactical",))
    except SystemExit as e:
        print("tactical refusal:", str(e)[:400])
