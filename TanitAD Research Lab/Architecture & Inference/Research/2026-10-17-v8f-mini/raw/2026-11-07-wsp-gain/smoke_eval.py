import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np, torch
import wsp_common as W, wsp_measure as WM
T = W.load_tool(); M = T.M
model, cfg, ck = W.load_model(M, W.CKPT_C0[0])
rig = WM.Rig("eval", cfg, nt=6)
wins = rig.eval_windows()
z = np.load(W.THEORY + "/wsp_read/P2B_PERWINDOW.npz", allow_pickle=True)
idx = np.flatnonzero(z["band"] == 1)[:64]
sub = [wins[i] for i in idx]
t0 = time.time()
r = WM.run_measure(rig, {"C0": (model, cfg)}, sub, bs=32, log=print)
print("wall", time.time() - t0)
print("v0 match", np.abs(r["kin3"][:, 0] - z["v0"][idx]).max(), "clip match", (r["clip"] == z["clip"][idx]).all())
print("s_tot max abs diff vs P2B", np.abs(r["C0.s_tot"] - z["val.s_tot"][idx]).max())
gk = z["gk.s_tot"][idx]
print("gv FD - autograd median abs", np.median(np.abs(r["C0.gv"] - gk[:, 0])), "ga", np.median(np.abs(r["C0.ga"] - gk[:, 1])))
print("median gains FD", np.median(r["C0.gv"]), np.median(r["C0.ga"]), " autograd", np.median(gk[:, 0]), np.median(gk[:, 1]), " base", np.median(r["gb"], 0))
print("h2", r["C0.h2"].shape)
