"""train_measure: the E-GAIN fit inputs. For the 20,000 TRAIN windows (raw/train_window_ids.txt) and BOTH C0 checkpoints: nominal plan-speed components, h2 (head_o[2]'s input), and the FD gains.
usage: train_measure.py --shard I --nshards N [--limit K] [--nt 5]   -> raw/TRAIN_MEASURE_shardI.npz   (dev-box CPU, zero training)"""
import os, sys, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import wsp_common as W, wsp_measure as WM

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--nt", type=int, default=5); ap.add_argument("--bs", type=int, default=64)
a = ap.parse_args()
t0 = time.time()
log = lambda *x: print(f"[{time.time()-t0:7.0f}s]", *x, flush=True)
T = W.load_tool(); M = T.M
models = {}
for s in (0, 1):
    m, cfg, ck = W.load_model(M, W.CKPT_C0[s])
    models[f"s{s}"] = (m, cfg)
rig = WM.Rig("train", cfg, nt=a.nt)
wins = rig.train_windows(W.HERE + "/raw/train_window_ids.txt")
assert len(wins) == 20000
sh = np.array_split(np.arange(len(wins)), a.nshards)[a.shard]
sub = [wins[i] for i in sh]
if a.limit:
    sub = sub[:a.limit]
log("shard", a.shard, "of", a.nshards, "windows", len(sub))
r = WM.run_measure(rig, models, sub, bs=a.bs, log=log)
r["flat_idx"] = sh[:len(sub)]
out = W.HERE + f"/raw/TRAIN_MEASURE_shard{a.shard}{'_smoke' if a.limit else ''}.npz"
np.savez(out, **r)
log("saved", out, {k: v.shape for k, v in list(r.items())[:6]})
