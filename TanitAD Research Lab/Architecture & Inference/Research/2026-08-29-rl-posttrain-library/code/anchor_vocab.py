"""EXIT C item 2 — is the ANCHOR VOCABULARY the ceiling? And is the target real?

Addenda 2+3 leave the vocabulary as the only surviving lever: reach at the usable
budget is 57% of the gap BEFORE the policy does anything. This asks the ceiling
question directly and with ZERO training: on near-miss windows, does ANY anchor in
the vocabulary clear d_safe -- and would a DATA-CLUSTERED vocabulary do better?

⭐ FOUR ARMS, THREE OF WHICH ARE CONTROLS (CLAUDE.md probe-panel rule):

  ORACLE (gt)   the HUMAN DRIVER'S OWN future. ⛔ THIS IS THE CONSTANT-CONTROL AND
                IT MUST READ A KNOWN VALUE: if the real driver does not clear
                d_safe either, then d_safe is NOT A CLEARABLE TARGET in this
                corpus, the "3.67 m gap" I published is a gap to an impossible
                place, and every conclusion resting on it -- including "the
                vocabulary is too small" -- is measuring the wrong thing.
  CURRENT       the model's own anchor bank (anchor_traj at steps=0, minus the
                classifier offset -- see TRAIN-C7 for why the tensor matters).
  CLUSTERED     k-means over REAL ego futures from this corpus, same N. The DDv2
                recipe. If the lever is real, this is where it shows.
  RANDOM        N draws from the same real-future pool WITHOUT clustering -- the
                floor. If clustering does not beat it, clustering adds nothing and
                the "re-cluster the anchors" plan is unsupported.

n and d are printed. Tier: T0. NON-PARITY corpus. Evidence class: MEASURED.
"""
import importlib.util, sys, json
import numpy as np, torch

sys.path.insert(0, "stack")
_s = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(_s); _s.loader.exec_module(P)

O, dev = r"C:/Users/Admin/tanitad-data/rl-pilot", "cuda"
D_SAFE, EGO_R, OBS_R = 5.0, 1.0, 1.0

m = P.load_model(r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt", dev); m.eval()
src = P.WindowSource(r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
                     f"{O}/pilot_val_agents.jsonl", seed=0, lru=60)
rng = np.random.default_rng(1234)
wins = []
for stem in src.stems:
    T = int(src._ep(stem)["poses"].shape[0])
    lo, hi = P.WINDOW - 1, T - P.HORIZONS[-1] - 1
    for t0 in sorted(rng.choice(np.arange(lo, hi), size=min(6, hi - lo), replace=False)):
        wins.append((stem, int(t0)))

# ---- bank the real ego futures (the clustering pool AND the oracle arm) --------
pool, keep = [], []
for stem, t0 in wins:
    w = src.window(stem, t0)
    if w is None:
        continue
    pool.append(w["gt"].numpy())
    keep.append((w, w["gt"].numpy()))
pool = np.stack(pool)                                   # [P, S, 2]
print(f"real-future pool: n={pool.shape[0]}  S={pool.shape[1]}  d={pool.shape[1]*2}")

# ---- the model's own anchor bank ----------------------------------------------
with torch.no_grad():
    b = P.collate([keep[0][0]], dev)
    o = m(b["frames"], None, b["v0"], steps=0)
    CUR = (o["anchor_traj"] - o["offset"])[0].cpu().numpy()   # [N, S, 2]
N = CUR.shape[0]
print(f"anchor vocabulary: N={N} anchors x S={CUR.shape[1]} steps")

# is the bank window-INVARIANT? (a fixed bank vs a v0-conditioned one is a
# different lever, so measure rather than assume)
with torch.no_grad():
    b2 = P.collate([keep[len(keep)//2][0]], dev)
    o2 = m(b2["frames"], None, b2["v0"], steps=0)
    CUR2 = (o2["anchor_traj"] - o2["offset"])[0].cpu().numpy()
dv = float(np.abs(CUR - CUR2).max())
print(f"bank window-invariance: max |A(w1) - A(w2)| = {dv:.6f} m"
      f"  -> {'FIXED bank' if dv < 1e-4 else 'CONDITIONED on the window'}")

# ---- clustered + random vocabularies ------------------------------------------
flat = pool.reshape(len(pool), -1)
def kmeans(X, k, seed=0, iters=60):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (a == j).any():
                C[j] = X[a == j].mean(0)
    return C
CLU = kmeans(flat, min(N, len(flat) - 1)).reshape(-1, pool.shape[1], 2)
RND = pool[np.random.default_rng(7).choice(len(pool), min(N, len(pool)), replace=False)]

def clearance(traj, obs):
    """min over (steps, obstacles) of surface-to-surface distance. traj [K,S,2]."""
    d = np.linalg.norm(traj[:, :, None, :] - obs[None, None, :, :], axis=-1)
    return np.clip(d - (EGO_R + OBS_R), 0, None).min(-1).min(-1)      # [K]

arms = {"ORACLE (human)": None, "CURRENT (model)": CUR,
        "CLUSTERED (kmeans)": CLU, "RANDOM (floor)": RND}
hit = {k: [] for k in arms}
best = {k: [] for k in arms}
n_nm = 0
for w, gt in keep:
    if not w["obs"]:
        continue
    obs = np.asarray(w["obs"], dtype=np.float32)
    if clearance(gt[None], obs)[0] >= D_SAFE:      # not a near-miss situation
        continue
    n_nm += 1
    for k, V in arms.items():
        c = clearance(gt[None] if V is None else V, obs)
        hit[k].append(float(c.max() >= D_SAFE))
        best[k].append(float(c.max()))

print(f"\n=== near-miss WINDOWS (the human's own future is inside d_safe={D_SAFE} m):"
      f" n={n_nm} of {len(keep)} ===")
print(f"{'arm':<20} {'clears d_safe':>14} {'best clearance (m)':>20}")
res = {}
for k in arms:
    h, bst = np.mean(hit[k]) if hit[k] else float('nan'), np.mean(best[k]) if best[k] else float('nan')
    res[k] = {"clear_frac": float(h), "best_clearance_m": float(bst), "n": n_nm}
    print(f"{k:<20} {100*h:13.1f}% {bst:20.3f}")
json.dump(res | {"N_anchors": int(N), "bank_max_dev_m": dv, "pool_n": int(len(pool))},
          open(f"{O}/anchor_vocab.json", "w"), indent=1)
print(f"\n-> {O}/anchor_vocab.json")
