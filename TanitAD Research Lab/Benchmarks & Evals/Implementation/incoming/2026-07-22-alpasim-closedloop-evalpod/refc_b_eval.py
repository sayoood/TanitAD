"""B — REF-C base canonical eval on the val cache. Confirms ckpt+env reproduce registry ADE 0.4728."""
import json, sys, time
for p in ("/root/taniteval", "/root/TanitAD/stack", "/root/TanitAD/stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
from refc_v12_cache import load_frozen
from taniteval import bench, data, refc_eval

t0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device, flush=True)
model, cfg, step = load_frozen("/root/models/refc-base-30k/ckpt.pt", "base", None, device)
print("loaded base ckpt step", step, "anchors", cfg.anchors.n_anchors, flush=True)
files = data.list_val_episodes("/root/valdata/physicalai-val-0c5f7dac3b11", 40)
print("val episodes", len(files), flush=True)
eps = data.load_frames(files)
win = refc_eval.collect(model, eps, device)
print("collected windows", win["pred"].shape, flush=True)
res = bench.run(win)
print("=== bench.run top-level keys ===", list(res.keys()), flush=True)
# dump the model/full_set/heldout blocks so we can read ADE@2s
for key in ("model", "full_set", "heldout"):
    if key in res:
        print(f"--- {key} ---", json.dumps(res[key], indent=2, default=str)[:1500], flush=True)
print("wall_s", round(time.time() - t0, 1), flush=True)
print("B_EVAL_DONE", flush=True)
