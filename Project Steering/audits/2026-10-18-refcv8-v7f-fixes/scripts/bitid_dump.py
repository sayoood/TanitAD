"""Bit-identity of the patch with every new flag OFF: dump the refcv8 rig's forward (eval + train, speed channel fed,
ceiling filter ON) from a tree root; run once on the pristine tip tree and once on the patched tree, then compare.
usage: python bitid_dump.py <tree root> <out .pt>   |   python bitid_dump.py --compare a.pt b.pt"""
import sys
from pathlib import Path
import torch
if sys.argv[1] == "--compare":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    keys = sorted(set(a) | set(b))
    only = [k for k in keys if (k in a) != (k in b)]
    diff = [k for k in keys if k in a and k in b and not torch.equal(a[k], b[k])]
    print("keys:", len(keys), "| only-in-one:", only, "| differing tensors:", diff)
    sys.exit(1 if (only or diff) else 0)
root = Path(sys.argv[1])
sys.path[:0] = [str(root / "stack"), str(root / "stack/scripts"), str(root / "stack/tests")]
import _refcv8_rig as R
T = R.trainer()
dump = {}
for mode in ("eval", "train"):
    cfg, m = R.build(T, True, vmax=True, speed_input="n2", n_alloc=8, alloc_emit=True)
    bt = R.batch(cfg, b=5)
    b = bt["frames"].shape[0]
    out = R.forward(cfg, m, bt, train=(mode == "train"), v_max_ms=torch.tensor([30., 50., 70., 100., 120.]) / 3.6,
                    v_max_valid=torch.ones(b))
    for k, v in out.items():
        if torch.is_tensor(v):
            dump[f"{mode}.{k}"] = v.detach().float().cpu()
torch.save(dump, sys.argv[2])
print(len(dump), "tensors dumped from", root)
