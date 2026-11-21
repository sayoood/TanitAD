"""t4 note: where the 4e-7 to 5e-7 of t4 comes from. Same rollout, once with the embedded actions of the full 15-step sequence sliced (what the loss does) and once with the K-step slice
embedded alone (what rollout() does at read time): the difference is the action embedding over two matmul shapes, not the rollout core."""
import json, os, sys
import torch
tree = sys.argv[1]
import tanitad
assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(tree).lower())
from tanitad.models import v8f_mini as M
cfg = M.V8FMiniConfig(wm="b", hmax=16, wm_state="grid", wm_l1_pred="cell", d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
torch.manual_seed(0)
w = M.WMAux(cfg).eval()
g = torch.Generator().manual_seed(2)
a = torch.randn(2, 15, 2, generator=g)
z = torch.randn(2, 640, generator=g)
with torch.no_grad():
    full = w.stack.act_embed(a)[:, 0:10]
    part = w.stack.act_embed(a[:, 0:10])
    emb_diff = float((full - part).abs().max())
    r_full = w.stack._roll_embedded(w.stack.predictors[0], z, full, 3)
    r_part = w.stack._roll_embedded(w.stack.predictors[0], z, part, 3)
    r_same = w.stack._roll_embedded(w.stack.predictors[0], z, full, 3)
res = {"embedding_full_vs_slice_max_abs_diff": emb_diff, "rollout_same_embedded_inputs_max_abs_diff": float((r_full - r_same).abs().max()), "rollout_full_vs_slice_embedding_max_abs_diff": float((r_full - r_part).abs().max())}
print(json.dumps(res)); json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t4_note.json"), "w"), indent=1)
