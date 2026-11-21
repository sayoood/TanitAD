"""Step 1 (CPU, ~minutes): one pass over the 2,399 eval-141 windows -> level-1 latents of both M2b seeds (and C0 s0 / s1 if their WM differs).
Writes latents.pt. Command: see HANDOFF_WMACT.md."""
import sys, time, numpy as np, torch
import wmact_lib as W

t0 = time.time()
models, cfgs, cks = {}, {}, {}
for tag, path in W.CKPT_C0.items():
    models[tag], cfgs[tag], cks[tag] = W.load_model(path)
assert all(c.hmax == 16 and c.wm == "b" and c.kin_mode == "yaw" for c in cfgs.values())
ds, wins, eid = W.open_windows(16)
N = len(wins)
Z, A, OK, T = W.collect_latents(models, ds, N)
torch.save({"Z1": Z, "A_raw": A, "ok": OK, "t": T, "eid": eid.tolist(), "N": N,
            "tanitad_file": W.tanitad.__file__}, "latents_C0.pt")
print("saved latents_C0.pt", {k: tuple(v.shape) for k, v in Z.items()}, tuple(A.shape), round(time.time() - t0, 1), "s")
