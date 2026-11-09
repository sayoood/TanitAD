"""Beside diagnostic (TUNE only, inner split of the gate): held-out R2 of D_1 on encoded latents versus the ridge multiplier. Not part of the card's read."""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import plantest3_lib as L3, plantest2_lib as Q
L = torch.load("E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wmact/latents.pt", weights_only=False)
ok = L["ok"].numpy().all(1); sel = np.nonzero(ok)[0]
eid = np.array(L["eid"])[sel].astype("U36"); t = np.asarray(L["t"])[sel]
TRK = torch.load("E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt", weights_only=False)["tracks"]
tune, score = Q.split_indices(eid)
fit_ids, held_ids = L3.inner_split(eid[tune])
vk = np.full((len(sel), 16), np.nan)
for i in tune:
    r = np.asarray(TRK[eid[i]]); g = int(t[i]) - 15
    for k in range(16):
        if 2 * (g + k) < len(r): vk[i, k] = r[2 * (g + k), 3]
yk = L["A_raw"][sel][:, :, 1].numpy().astype(np.float64)
for tag in ("M2b_s0", "M2b_s1"):
    Z = L["Z1"][tag][sel].float()
    fi = np.array([i for i in tune if eid[i] in fit_ids]); hi = np.array([i for i in tune if eid[i] in held_ids])
    X = Z[fi].reshape(-1, Z.shape[-1]); print(tag, "mean per-dim feature variance", float(X.var(0).mean()), "ridge/n 1e-3 relative to it:", 1e-3 / float(X.var(0).mean()))
    for mult in (1.0, 0.1, 0.01, 0.0):
        dec = L3.fit_decoder(None, Z[fi], vk[fi], yk[fi], ridge=1e-3 * mult, with_l2=False)
        r = L3.heldout_r2(None, dec, 1, Z[hi], vk[hi], yk[hi])
        print(tag, "ridge x", mult, "held-out R2 speed %.3f yaw %.3f" % (r[0], r[1]), flush=True)
