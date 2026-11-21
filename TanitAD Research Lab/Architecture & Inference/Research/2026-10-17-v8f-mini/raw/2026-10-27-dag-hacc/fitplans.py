"""fitplans.py <name> <ckpt> <out.npz>: HBLEND's cmd_fit plan-production route (v8f_hblend.cmd_fit), checkpoint-parametrised; NOTHING else changes (DECLARATIONS D2).
Windows = the trainer's own train_windows over the 141 fit clips (fit_clips.json, k = 32); ages = cycle (a_t = i%3, a_s = a_t + (i//3)%5); model.forward_infer; tau = traj30, P = p10, y = gt.
No fit happens here; the arrays are written and the sha256 of the checkpoint and of every input is recorded."""
import hashlib, json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/stack")
sys.dont_write_bytecode = True
TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
SP = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/hblend"      # HBLEND's finished fit-set package (READ ONLY)
sys.path.insert(0, f"{TREE}/stack"); sys.path.insert(0, f"{TREE}/taniteval")
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get("FP_THREADS", "4")))
from tanitad.data import v8f_mini_data as D
from tanitad.models import v8f_mini as M


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


name, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
BATCH, LRU = 64, 8
t0 = time.time()
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
cfg = M.V8FMiniConfig(**ck["cfg"])
model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
model.load_state_dict(ck["model"])
model.eval()
assert not model.hacc_active, "the fit plans must come from the PLAIN checkpoint"
fc = json.load(open(f"{SP}/fit_clips.json"))
clips = fc["clips"]
J = D.load_join(f"{SP}/box2/join_train.npz")
tracks = D.load_native_tracks("E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt")
store = D.ClipStore(f"{SP}/cache", "train", lru=LRU, grid=ck.get("grid", "pool4x10"), kin_mode=cfg.kin_mode, tracks=tracks)
wins = D.train_windows(store, J, clips, hist=cfg.hist, start=cfg.window_start)
ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
N = len(wins)
print(f"[{name}] clips {len(clips)} windows {N} hist {cfg.hist} start {ds.window_start} grid {ck.get('grid')}", flush=True)
outs = {k: [] for k in ("traj30", "p30", "gt", "gt_ok", "ci", "t", "a_t", "a_s")}
with torch.no_grad():
    for b0 in range(0, N, BATCH):
        items = []
        for i in range(b0, min(b0 + BATCH, N)):
            ci, t, _, _ = wins[i]
            a_t = i % 3
            a_s = a_t + (i // 3) % 5
            items.append(ds.item(ci, t, a_t, a_s))
        b = D.collate(items)
        o = model.forward_infer(b)
        outs["traj30"].append(o["traj30"].float().numpy()); outs["p30"].append(o["p10"].float().numpy())
        for k in ("gt", "gt_ok", "ci", "t", "a_t", "a_s"):
            outs[k].append(b[k].numpy())
        if (b0 // BATCH) % 20 == 0:
            print(f"[{name}] {b0 + len(items)}/{N} {time.time() - t0:.0f}s", flush=True)
O = {k: np.concatenate(v) for k, v in outs.items()}
assert O["gt_ok"].all() and N == 9582, (N, O["gt_ok"].all())
np.savez_compressed(out, **O)
meta = {"name": name, "ckpt": ckpt, "ckpt_sha256": sha(ckpt), "n_windows": int(N), "n_clips": len(clips), "fit_clips_sha256_newline_joined": fc.get("sha256_newline_joined"), "k": fc.get("k"),
        "join_train_sha256": sha(f"{SP}/box2/join_train.npz"), "native_tracks_sha256": sha("E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt"), "tool_tree": TREE,
        "out_sha256": sha(out), "seconds": time.time() - t0, "cfg_window_start": cfg.window_start, "hist": cfg.hist, "ages": "cycle (a_t = i%3, a_s = a_t + (i//3)%5)"}
json.dump(meta, open(out + ".meta.json", "w"), indent=1)
print("ZZPLANS", json.dumps({k: meta[k] for k in ("name", "ckpt_sha256", "n_windows", "out_sha256", "seconds")}), flush=True)
