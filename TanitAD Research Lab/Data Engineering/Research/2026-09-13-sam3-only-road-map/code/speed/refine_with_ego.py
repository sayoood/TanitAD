"""sam3map_refine_v6.main() UNMODIFIED, with its ego-mask step replaced by masks that were computed by the very same
`ego_masks` function in the streaming driver's process (shared, already-loaded SAM3 model, fp32). Everything after the ego masks
-- R2 ego patch, R3 thin edges, the re-lift control, the output files -- is the refine script's own code.
Usage: refine_with_ego.py <c8> <ego npz>   (env SAM3MAP_ROOT, SAM3MAP_VIEWS, SAM3MAP_TAG as for the refine script)"""
import json, sys
sys.path.insert(0, "/home/nvidia/sam3vendor"); sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, "/home/nvidia/sam3map")
import numpy as np
import sam3map_refine_v6 as R

c8, ego_path = sys.argv[1], sys.argv[2]
z = np.load(ego_path, allow_pickle=True)
out = {k: z[k] for k in z.files if k != "_info"}
info = json.loads(str(z["_info"]))
out["_spread_m"] = info["_spread_m"]


def loaded_ego_masks(sd, toks):
    assert len(toks) == int(info["_n_toks"]), "ego masks were computed for a different frame list"
    return dict(out), {k: v for k, v in info.items() if k != "_n_toks"}      # exactly what ego_masks() returned


R.ego_masks = loaded_ego_masks
sys.argv = ["sam3map_refine_v6.py", c8]
R.main()
