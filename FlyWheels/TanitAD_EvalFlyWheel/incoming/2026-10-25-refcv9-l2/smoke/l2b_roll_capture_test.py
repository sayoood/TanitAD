"""Unit test of l2b_locus_roll's capture + save with a FAKE base proxy (no model, no GPU): shapes, dtypes, batch-less tensors, absent keys, bit-exact round trip."""
import sys, os, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.argv = ["l2b_locus_roll.py", "--trace-out", "C:/Users/Admin/refcv8_pf/refcv9_l2/smoke/l2b_trace_test.npz", "--",
            "--ckpt", "x", "--config", "y", "--seed", "0", "--dump-dir", "z", "--windows-json", "w", "--device", "cpu", "--out-json", "o", "--os-only"]
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import numpy as np, torch
import l2b_locus_roll as R

class Core: window = 8
class Cfg: core = Core()
class FakeModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.cfg = Cfg()
N = 149
g = torch.Generator().manual_seed(0)
def fake_out(drop=()):
    o = {"traj": torch.randn(1, 8, 2, generator=g), "sel_idx": torch.tensor([5]), "sel_idx_base": torch.tensor([7]),
         "anchor_traj": torch.randn(1, N, 8, 2, generator=g), "sel_score": torch.randn(1, N, generator=g), "sel_score_v3": torch.randn(1, N, generator=g),
         "reach_keep": torch.rand(1, N, generator=g) > 0.2, "ceil_keep": torch.ones(1, N, dtype=torch.bool), "r8_emit_keep": torch.rand(1, N, generator=g) > 0.1,
         "r8_kind": torch.randint(0, 3, (N,), generator=g).to(torch.int8), "r8_lon6": torch.randint(0, 6, (1, N), generator=g).to(torch.int8),
         "r8_lat3": torch.randint(0, 3, (1, N), generator=g).to(torch.int8), "r8_cons_raw": torch.randn(1, N, 4, generator=g), "r8_hyp_alloc": torch.randint(0, 18, (32,), generator=g).to(torch.int8),
         "residual_prior_path": torch.randn(1, 8, 2, generator=g), "anchor_bank": torch.randn(1, 117, 8, 2, generator=g)}
    for k in drop: o.pop(k, None)
    return o
OUTS = []
def fake_call(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
    o = fake_out(drop=(("ceil_keep",) if len(OUTS) == 1 else ())); OUTS.append(o); return o
R._Base.__call__ = fake_call
p = object.__new__(R.L2BProxy)
class Ctx: index = (3, 10)
p.__dict__.update({"model": FakeModel(), "ctx": Ctx(), "conds_fed": ["nav_true"]})
for i in range(3):
    p.__call__(torch.zeros(1, 3, 4, 4), nav_cmd=torch.zeros(1, dtype=torch.long), v0=torch.tensor([7.5]))
R.save(True)
z = np.load(R.A.trace_out)
m = json.loads(str(z["meta_json"]))
assert m["final"] and m["n_windows"] == 3 and m["counters"]["fan_n_seen"] == [N] and m["counters"]["batch_gt1"] == 0, m
assert z["ws"].tolist() == [17, 17, 17] and z["e_i"].tolist() == [3, 3, 3]
assert z["fan"].shape == (3, N, 8, 2) and z["fan"].dtype == np.float32 and z["sel_score_v3"].shape == (3, N)
assert z["kind"].shape == (3, N) and z["hyp_alloc"].shape == (3, 32) and z["bank"].dtype == np.float16 and z["cons_raw"].shape == (3, N, 4)
assert np.array_equal(z["fan"][0], OUTS[0]["anchor_traj"][0].numpy()), "fan not bit-exact"
assert np.array_equal(z["sel_score_v3"][2], OUTS[2]["sel_score_v3"][0].numpy())
assert np.array_equal(z["reach_keep"][0], OUTS[0]["reach_keep"][0].numpy().astype(np.int8))
assert (z["ceil_keep"][1] == -1).all() and (z["ceil_keep"][0] == 1).all(), "absent key must be stored as -1"
assert np.array_equal(z["kind"][1], OUTS[1]["r8_kind"].numpy()), "batch-less tensor must be taken whole"
assert np.array_equal(z["plan8"][0], OUTS[0]["traj"][0].numpy()) and z["sel"].tolist() == [5, 5, 5] and z["sel_base"].tolist() == [7, 7, 7]
assert abs(z["v0_in"][0] - 7.5) < 1e-6
# MUTATION: indexing the wrong row of a batched tensor must be caught by the round-trip check
bad = OUTS[0]["sel_score"][0].numpy()[::-1]
assert not np.array_equal(z["sel_score"][0], bad)
print("L2B CAPTURE TEST PASS", m["present_counts"])
