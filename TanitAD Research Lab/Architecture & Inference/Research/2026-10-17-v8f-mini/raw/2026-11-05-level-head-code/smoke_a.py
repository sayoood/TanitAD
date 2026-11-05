import sys, tempfile
from pathlib import Path
root = Path(sys.argv[1])
for p in (root / "stack", root / "taniteval", root / "stack" / "scripts"):
    sys.path.insert(0, str(p))
import torch
from tanitad.data import v8f_mini_data as D
from tanitad.models import v8f_mini as M
import v8f_mini_train as TR
torch.set_num_threads(2)
print("tanitad file:", M.__file__)
tmp = Path(tempfile.mkdtemp(prefix="smk_", dir="F:/TanitAD-scratch/motion_c/build"))
w = D.make_synthetic_world(tmp, n_train=6, n_eval=2, T=60, seed=0)
J = D.load_join(tmp / "join_train.npz")
store = D.ClipStore(tmp, "train", lru=8)
wins = D.train_windows(store, J, w["names"]["train"])
ds = D.V8FMiniWindows(store, J, wins, hmax=10)
import numpy as np
idx = np.random.default_rng(0).choice(len(wins), 8, replace=False)
b = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
mu, sd = M.cons_stats(J)
torch.manual_seed(0)
off = M.V8FMini(M.V8FMiniConfig(**TINY), M.P10Prior.synthetic(0), mu, sd)
torch.manual_seed(0)
on = M.V8FMini(M.V8FMiniConfig(**TINY, level_head="on", w_level=0.5), M.P10Prior.synthetic(0), mu, sd)
ko, kn = list(off.state_dict()), list(on.state_dict())
print("new keys:", [k for k in kn if k not in ko], "old kept:", all(k in kn for k in ko))
print("old tensors identical:", all(torch.equal(off.state_dict()[k], on.state_dict()[k]) for k in ko))
with torch.no_grad():
    off.head_o[2].weight.copy_(torch.randn(off.head_o[2].weight.shape, generator=torch.Generator().manual_seed(1)) * 0.05)
r = on.load_state_dict(off.state_dict(), strict=False)
print("missing:", r.missing_keys, "unexpected:", r.unexpected_keys)
off.eval(); on.eval()
with torch.no_grad():
    a, c = off.forward_infer(b), on.forward_infer(b)
print("traj equal:", torch.equal(a["traj60"], c["traj60"]), "ckpt equal:", torch.equal(a["ckpt"], c["ckpt"]), "extra out:", set(c) - set(a))
lv, lo = TR.level_stage(b)
b2 = {**b, "level_y": lv, "level_ok": lo}
on.train()
o = on.loss(b2, p_gt=0.5, gen=torch.Generator().manual_seed(1))
print("terms level:", float(o["terms"]["level"]), "total:", float(o["terms"]["total"]))
print("live:", TR.live_check(on, b2, on.cfg)["pass"])
print("probe:", {k: v for k, v in TR.level_head_probe(on, b2, on.cfg).items() if k != "vision_only"})
print("vision only:", M.assert_level_head_vision_only(on, b2))
