"""Unit test of mapsel_roll's capture + save with a FAKE base proxy (no model, no GPU): shapes/dtypes, raster planes against PLANTED analytic logits, the prior-correction (not raw argmax), GT plane + liveness."""
import sys, os, json, shutil
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
S = "C:/Users/Admin/refcv8_pf/refcv9_mapsel/smoke"
shutil.rmtree(S + "/rast_test", ignore_errors=True)
sys.argv = ["mapsel_roll.py", "--trace-out", S + "/roll_test_trace.npz", "--rast-dir", S + "/rast_test", "--map-gt-root", "x", "--",
            "--ckpt", "x", "--config", "y", "--seed", "0", "--dump-dir", "z", "--windows-json", "w", "--device", "cpu", "--out-json", "o", "--os-only"]
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_mapsel/code")
import numpy as np, torch
import mapsel_roll as R

class Core: window = 8
class Cfg: core = Core()
class BrCfg: decision_rule = "prior_corrected"
class Br: cfg = BrCfg()
class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.cfg = Cfg(); self._map_hires = Br()
        self._map_hires_class_weight = torch.tensor([1., 1, 1, 1, 1, 4, 1, 1])
N, H, W = 149, 1000, 600
g = torch.Generator().manual_seed(0)
def logits():
    z = torch.zeros(1, 8, H, W)
    z[0, 1, :, :300] = 10.0          # left half: drivable, strongly
    z[0, 7, :, 300:] = 10.0          # right half: sidewalk
    z[0, 5, 100:110, :] = 1.0        # a strip where RAW argmax = edge (z5 = 1 > z1 = 0 only where z1 = 0, i.e. the right half has z7 = 10: use the left half)
    z[0, 1, 100:110, :300] = 0.0     # left-half strip rows 100..109: z1 = 0, z5 = 1  -> raw argmax 5, prior-corrected (w5 = 4 -> -1.386) -> 5: 1-1.386 = -0.386 < 0 -> class 0/1/2/.. tie at 0 -> argmax picks the first max = code 0
    return z
def fake_out():
    return {"traj": torch.randn(1, 8, 2, generator=g), "sel_idx": torch.tensor([5]), "sel_idx_base": torch.tensor([7]),
            "anchor_traj": torch.randn(1, N, 8, 2, generator=g), "sel_score": torch.randn(1, N, generator=g), "sel_score_v3": torch.randn(1, N, generator=g),
            "reach_keep": torch.rand(1, N, generator=g) > 0.2, "r8_emit_keep": torch.ones(1, N, dtype=torch.bool),
            "residual_prior_path": torch.randn(1, 8, 2, generator=g), "perception": {"map_hires_logits": logits()}}
R._Base.__call__ = lambda self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw: fake_out()
gt = np.full((H, W), 1, np.uint8); gt[:, 300:] = 7; gt[:50, :] = 255
class Ctx:
    index = (3, 10)
    item = {"map_fine": torch.from_numpy(gt), "map_fine_label": torch.tensor(True)}
class Ctx2:
    index = (4, 10)
    item = {}
p = object.__new__(R.MapSelProxy)
p.__dict__.update({"model": FakeModel(), "ctx": Ctx(), "conds_fed": ["nav_true"]})
p.__call__(torch.zeros(1, 3, 4, 4), nav_cmd=torch.zeros(1, dtype=torch.long), v0=torch.tensor([7.5]))
p.__dict__["ctx"] = Ctx2()
p.__call__(torch.zeros(1, 3, 4, 4), nav_cmd=torch.zeros(1, dtype=torch.long), v0=torch.tensor([7.5]))
R.save(True)
z = np.load(R.A.trace_out); m = json.loads(str(z["meta_json"]))
assert m["final"] and m["n_windows"] == 2 and m["counters"]["fan_n_seen"] == [N] and m["counters"]["decide_mismatch_cells"] == 0, m
assert z["e_i"].tolist() == [3, 4] and z["ws"].tolist() == [17, 17] and z["gt_label"].tolist() == [True, False]
assert z["fan"].shape == (2, N, 8, 2) and z["sel_score_v3"].shape == (2, N) and z["prior_path"].shape == (2, 8, 2)
r0 = np.load(R.RAST / "e003_w0017.npz"); r1 = np.load(R.RAST / "e004_w0017.npz")
assert r0["pdrv"].shape == (H, W) and r0["pdrv"].dtype == np.uint8 and r0["gt"].shape == (H, W)
assert "gt" not in r1.files
# planted analytic expectations (hand-derived): left half rows != strip -> p_drv = e^10/(e^10 + 7) = 0.99968 -> 255 ; right half -> 5 / (e^10 + 7) = 0.000227 -> 0 ; codes 1 and 7
assert (r0["pdrv"][200:300, :300] == 255).all() and (r0["pdrv"][200:300, 300:] == 0).all()
assert (r0["code"][200:300, :300] == 1).all() and (r0["code"][200:300, 300:] == 7).all()
# the strip rows 100..109 of the left half: z = [0,0,0,0,0,1,0,0], w5 = 4 -> s5 = 1 - ln4 = -0.386 ; s0 = 0 (first of the tied zeros) -> code 0 (NOT 5: the raw argmax would say 5)
assert (r0["code"][100:110, :300] == 0).all(), np.unique(r0["code"][100:110, :300])
# p_edge there = e^-0.386 / (7 + e^-0.386) = 0.6797 / 7.6797 = 0.0885 -> 22.6 -> 23
assert (r0["pedge"][100:110, :300] == 23).all(), np.unique(r0["pedge"][100:110, :300])
# liveness: the own drivable decision vs the GT drivable on SEEN cells (rows >= 50): the strip rows 100..109 left have p_drv = 5/(5+e^-.386+...) -> check the analytic agreement
exp_acc = 1.0 - (10 * 300) / ((H - 50) * W)           # only the strip's left cells disagree (their P(drv) = 5/7.68 = 0.65 -> 166 >= 128 -> they AGREE with GT drivable) -> acc 1.0 exactly
assert abs(z["acc_own"][0] - 1.0) < 1e-12 and np.isnan(z["acc_own"][1]), z["acc_own"]
# AMENDMENT 1: the P^(sidewalk) plane.  right half z7 = 10 -> e^10 / (e^10 + 7) = 0.99968 -> 255 ; left half -> 1 / (e^10 + 7) -> 0 ; the strip (z = [0,0,0,0,0,1,0,0], w5 = 4): p7 = 1 / (7 + e^-0.386) = 0.1302 -> 33
assert r0["psw"].shape == (H, W) and r0["psw"].dtype == np.uint8
assert (r0["psw"][200:300, 300:] == 255).all() and (r0["psw"][200:300, :300] == 0).all()
assert (r0["psw"][100:110, :300] == 33).all(), np.unique(r0["psw"][100:110, :300])
print("ROLLCAPTUREPASS")
