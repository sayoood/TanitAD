"""v0 distribution over the SAME 6400 sampled train windows (CPU, poses only —
no frames are read). Needed to compare the magnitude of the `zero` vs `perm`
input perturbations: RMS(v-0) vs RMS(v_i - v_j)."""
import sys, json
from pathlib import Path
import numpy as np, torch
STACK = Path("/workspace/TanitAD/stack")
sys.path.insert(0, str(STACK)); sys.path.insert(0, str(STACK/"scripts"))
from tanitad.data.mixing import load_episode
from refb_train import build_window_index
root = sorted(Path("/workspace/data/physicalai_phase0/_epcache").glob("*train*"))[-1]
files = sorted(root.glob("ep_*.pt"))
eps = [load_episode(str(p), mmap=True) for p in files]
WINDOW, MAXH = 8, 20
index = build_window_index([e.frames.shape[0] for e in eps], WINDOW, MAXH)
g = torch.Generator().manual_seed(1234)
perm = torch.randperm(len(index), generator=g)[:6400].tolist()
v = np.array([float(eps[index[i][0]].poses[index[i][1]+WINDOW-1][3]) for i in perm])
vs = v/10.0
pairs = []
rng = np.random.default_rng(0)
for b in range(400):
    blk = vs[b*16:(b+1)*16]
    pairs.append(blk - blk[rng.permutation(16)])
d = np.concatenate(pairs)
out = {"n": int(v.size), "v0_mps": {"mean": round(float(v.mean()),4),
       "std": round(float(v.std()),4), "min": round(float(v.min()),4),
       "max": round(float(v.max()),4), "frac_below_1mps": round(float((v<1).mean()),4),
       "frac_below_0.5mps": round(float((v<0.5).mean()),4)},
  "scaled_v0_over_10": {"mean": round(float(vs.mean()),4), "rms": round(float(np.sqrt((vs**2).mean())),4)},
  "perturbation_rms_scaled": {
      "zero_fill  RMS(v-0)": round(float(np.sqrt((vs**2).mean())),4),
      "within_batch_perm RMS(v_i-v_j)": round(float(np.sqrt((d**2).mean())),4)},
  "perturbation_meanabs_scaled": {
      "zero_fill  E|v-0|": round(float(np.abs(vs).mean()),4),
      "within_batch_perm E|v_i-v_j|": round(float(np.abs(d).mean()),4),
      "x2  E|2v-v|": round(float(np.abs(vs).mean()),4)},
  "literal_following_ADE_m_over_op_fwd_k4": {
      "note": "if the rollout followed the FED speed literally, ADE over steps "
              "1..4 at 0.1 s = 0.25 * E|v_fed - v_true| (metres)",
      "zero": round(float(0.25*np.abs(v).mean()),4),
      "perm": round(float(0.25*np.abs(d*10.0).mean()),4),
      "x2":   round(float(0.25*np.abs(v).mean()),4)},
  "note": "the SAME 6400 windows, same seed 1234, same batch blocking as exp_b"}
print(json.dumps(out, indent=2))
