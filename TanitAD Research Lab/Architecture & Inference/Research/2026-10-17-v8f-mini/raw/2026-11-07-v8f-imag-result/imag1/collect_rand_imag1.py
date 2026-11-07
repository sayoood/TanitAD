"""IMAG1 IA-1 random-init rows: level-1 latents of two untrained models (same architecture as M2b) on the eval-141 windows. Writes latents_rand.pt (F: scratch is not needed: 20 MB)."""
import sys, time, numpy as np, torch
sys.dont_write_bytecode = True
import imag1_model as IM
W = IM.W
t0 = time.time()
_, cfg, _ = W.load_model(W.CKPT["M2b_s0"])
models = {f"rand{r}": IM.make_random_model(r, cfg) for r in (0, 1)}
ds, wins, eid = W.open_windows(16)
Z, A, OK, T = W.collect_latents(models, ds, len(wins))
L = torch.load("../wmact/latents.pt", weights_only=False)
assert (A == L["A_raw"]).all() and (OK == L["ok"]).all() and (T == np.asarray(L["t"])).all() and eid.tolist() == L["eid"]
torch.save({"Z1": Z, "N": len(wins)}, "latents_rand.pt")
print("saved latents_rand.pt", {k: tuple(v.shape) for k, v in Z.items()}, round(time.time() - t0, 1), "s")
