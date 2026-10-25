import sys, os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
cal = "C:/Users/Admin/refcv8_pf/refcv9_l2/dev30k/calibrator_30k_DEV.json"
sys.argv = ["l2_roll.py", "--arm", "cal", "--calibrator", cal, "--trace-out", "C:/Users/Admin/refcv8_pf/refcv9_l2/smoke/trace_test.npz", "--require-seal", "--",
            "--ckpt", "x", "--config", "y", "--seed", "0", "--dump-dir", "z", "--windows-json", "w", "--device", "cpu", "--out-json", "o", "--os-only"]
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import l2_roll as R
import torch, torch.nn as nn
class Core: window = 8
class Cfg: core = Core()
class Fake(nn.Module):
    def __init__(self):
        super().__init__(); self.cfg = Cfg(); self.lin = nn.Linear(2, 2); self.seen = []
    def _r8_apply_force(self, out, b, dev):
        self.seen.append(out["lon_logits"].clone()); return out, None
m = Fake()
R._install(m)
assert R.ST["W"] == 8 and R.ST["hook_module"] == ""
lg = torch.randn(1, 8)
# unarmed: untouched
o, f = m._r8_apply_force({"lon_logits": lg.clone()}, 1, "cpu"); assert torch.equal(m.seen[-1], lg) and R.ST["c"]["hook_calls_unarmed"] == 1
# armed: transformed, traced
R.ST["armed"], R.ST["cur"], R.ST["tmp"] = True, (3, 10, (3, 17)), None
o, f = m._r8_apply_force({"lon_logits": lg.clone()}, 1, "cpu")
want = (lg + torch.tensor(R.CAL["b8"])) / R.CAL["T"]
assert torch.allclose(m.seen[-1], want, atol=1e-6), "seam did not receive the calibrated logits"
assert torch.equal(torch.as_tensor(R.ST["tmp"]["raw"]), lg[0]) and torch.allclose(torch.as_tensor(R.ST["tmp"]["app"]), want[0], atol=1e-6)
print("hook plumbing OK; counters", R.ST["c"])
# MUTATION: wrong calibrator sign must be detected by the K4-style comparison
bad = (lg - torch.tensor(R.CAL["b8"])) / R.CAL["T"]
print("mutation (sign flip) differs from the true apply:", not torch.allclose(bad, want, atol=1e-4))
