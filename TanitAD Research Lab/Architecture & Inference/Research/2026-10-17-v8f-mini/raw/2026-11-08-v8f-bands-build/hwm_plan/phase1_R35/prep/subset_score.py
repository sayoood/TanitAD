"""Phase 1 data step (no code of record changed): restrict windows.npz to the SCORE half of PLANTEST2's split (salt plantest2-v1).
The split is computed on the distinct episode ids of the 1,976 PLANTEST pairs (as PLANTEST2 does), then windows of SCORE episodes are kept
(the card's frame-count drop was already applied by `run_hwm.py windows`). Donors are rebuilt on the subset exactly as run_hwm.py does for --n.
usage: subset_score.py <out_dir>"""
import json, sys, os
import numpy as np, torch
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/plantest2")
import plantest2_lib as P2
from tanitad.models.hjepa import donor_indices
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/hwm_plan")
import hwm_data as HD

out = sys.argv[1]
src = f"{out}/windows_all.npz"
if not os.path.exists(src):
    os.replace(f"{out}/windows.npz", src)
d = np.load(src, allow_pickle=False)
eid = np.array([str(x) for x in d["eid"]])
all_eids, _ = HD.plantest_pairs()
tune, score = P2.split_episodes(all_eids)  # on the ids of all 1,976 pairs, as PLANTEST2 does
keep = np.isin(eid, list(score))
meta = json.loads(str(d["meta"]))
meta.update(split="SCORE half of PLANTEST2 (salt plantest2-v1)", n_before_split=int(len(eid)), n_score_episodes=int(len(set(eid[keep]))), n_tune_episodes_excluded=int(len(set(eid[~keep]))))
e = eid[keep]
codes = {x: i for i, x in enumerate(sorted(set(e)))}
donor = donor_indices(len(e), episode_ids=torch.tensor([codes[x] for x in e]), seed=0).numpy()
meta["n"] = int(keep.sum())
np.savez(f"{out}/windows.npz", z=d["z"][keep], A=d["A"][keep], eid=d["eid"][keep], now=d["now"][keep], v0=d["v0"][keep], donor=donor, meta=json.dumps(meta))
print("SCORE windows", int(keep.sum()), "of", len(eid), "episodes", len(set(e)), "excluded TUNE episodes", len(set(eid[~keep])))
