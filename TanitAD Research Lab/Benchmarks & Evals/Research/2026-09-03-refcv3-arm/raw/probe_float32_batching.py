"""Quantify the float32 batching floor: same math, different batch size."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

MIRROR = Path(r"C:\Users\Admin\tanitad-wt")
sys.path.insert(0, str(MIRROR / "stack" / "scripts"))
sys.path.insert(0, str(MIRROR / "stack" / "tests"))
spec = importlib.util.spec_from_file_location(
    "tst", str(MIRROR / "stack" / "tests" / "test_refcv3_arm.py"))
tst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tst)
rc = tst.rc
v3 = tst.v3

root = Path(sys.argv[1]) / "probe"
root.mkdir(parents=True, exist_ok=True)
eps, lp, ck = tst._fixture(root, hier=False)

model, cfg, targs, prov = rc.load_model(str(ck), None, "cpu", False)
_eps, files, clip_ids, ds, lman, join, nav_src, raw_off = rc.build_corpus(
    tst._args(root, ck, eps, lp), cfg, prov)
tr = rc.trainer()
steps = int(prov["decoder_steps"])
grid = rc.grid_slots(cfg.core.trajectory.horizons, "2s")

d_b1_vs_b2, d_b1_vs_b1, d_turn = [], [], []
for wi in range(0, len(ds.index), 7):
    item = ds[wi]
    v0 = float(item["pose_last"][3])
    fr = tr.frames_to_device(item["frames"][None], "cpu")
    v1 = torch.tensor([v0], dtype=torch.float32)
    z = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        # batch 2, row 0, fed `follow`   (how `os` is produced when nav==follow)
        o2 = model(fr.expand(2, *fr.shape[1:]).contiguous(),
                   nav_cmd=torch.tensor([0, 0]), v0=v1.expand(2), steps=steps)
        # batch 1, nav_cmd=None          (how `os_navzero` is produced)
        o1n = model(fr, nav_cmd=None, v0=v1, steps=steps)
        # batch 1, fed `follow`          (same batch size, same token)
        o1f = model(fr, nav_cmd=z, v0=v1, steps=steps)
        # batch 1, fed `left`            (a real nav difference)
        o1l = model(fr, nav_cmd=torch.tensor([1]), v0=v1, steps=steps)
    a = o2["traj"][0].numpy()
    b = o1n["traj"][0].numpy()
    c = o1f["traj"][0].numpy()
    d = o1l["traj"][0].numpy()
    d_b1_vs_b2.append(float(np.abs(a - b).max()))
    d_b1_vs_b1.append(float(np.abs(c - b).max()))
    d_turn.append(float(np.abs(c - d).max()))

print(f"n windows probed: {len(d_b1_vs_b2)}")
print(f"batch2-row0 vs batch1, SAME nav  (pure batching): "
      f"max {max(d_b1_vs_b2):.3e}  mean {np.mean(d_b1_vs_b2):.3e}")
print(f"batch1 vs batch1, nav=None vs nav=0 (identical math): "
      f"max {max(d_b1_vs_b1):.3e}")
print(f"batch1 fed `follow` vs fed `left` (a REAL nav difference): "
      f"max {max(d_turn):.3e}  min {min(d_turn):.3e}")
print(f"separation ratio (real nav diff / batching floor): "
      f"{min(d_turn) / max(max(d_b1_vs_b2), 1e-12):.1f}x")
