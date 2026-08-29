"""EXIT C item 1 — is the fan's reachable set smaller than the gap it must close?

The re-run established that the policy cannot reach "avoid obstacles". This asks
whether that is STRUCTURAL. Three quantities, reported against each other rather
than in bare metres (Master Mind design note (a)):

  REACH-CEILING   |offset| the decoder emits at all — the displacement the fan
                  has from the raw anchor vocabulary.
  REACH-TRAINED   |offset_trained - offset_cold| after 2,000 RL steps — what the
                  policy ACTUALLY moved. Measured from the saved ckpt_after.pt.
  THE GAP         for candidates inside the near-miss band, the extra clearance
                  needed to reach d_safe = 5 m — i.e. what a candidate would have
                  to move to stop being penalised.

Measured at decoder_steps = 2 (deployed) AND 4/8 (design note (b)): if reach
scales with steps, "structurally dead" becomes "dead at this inference budget",
which is a different and more fixable claim.

Tier: T0 instrument measurement. NON-PARITY corpus.
"""
import importlib.util, sys, json
import numpy as np, torch

sys.path.insert(0, "stack")
spec = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
from tanitad.rl import PostTrainConfig

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
dev = "cuda"
D_SAFE = 5.0
EGO_R, OBS_R = 1.0, 1.0

cold = P.load_model(r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt", dev)
cold.eval()
src = P.WindowSource(r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
                     f"{O}/pilot_val_agents.jsonl", seed=0, lru=60)
rng = np.random.default_rng(1234)
wins = []
for stem in src.stems:
    T = int(src._ep(stem)["poses"].shape[0])
    lo, hi = P.WINDOW - 1, T - P.HORIZONS[-1] - 1
    for t0 in sorted(rng.choice(np.arange(lo, hi), size=min(4, hi - lo), replace=False)):
        wins.append((stem, int(t0)))
wins = wins[:60]


@torch.no_grad()
def fans_at(model, steps):
    """The EMITTED FAN, not `out["offset"]`.

    ⚠️ CORRECTED mid-measurement: `out["offset"]` is the CLASSIFIER-PASS offset
    (`refc.py:1364-1365`); the truncated-diffusion loop accumulates into `x` and
    returns it as `anchor_traj` (`:1394-1399`, `:1515`). Measuring `offset`
    therefore reads a quantity that CANNOT depend on `steps` — and it duly read
    identical at steps 0/2/4/8, which is what exposed the error. The policy's
    actual output is the fan.
    """
    out = []
    for stem, t0 in wins:
        b = P.collate([src.window(stem, t0)], dev)
        o = model(b["frames"], None, b["v0"], steps=steps)
        out.append(o["anchor_traj"].detach().cpu())
    return torch.cat(out)                                  # [W, N, S, 2]


@torch.no_grad()
def anchors_of(model):
    """The RAW anchor vocabulary = anchor_traj at steps=0 minus the base offset."""
    out = []
    for stem, t0 in wins:
        b = P.collate([src.window(stem, t0)], dev)
        o = model(b["frames"], None, b["v0"], steps=0)
        out.append((o["anchor_traj"] - o["offset"]).detach().cpu())
    return torch.cat(out)


print("=== REACH-CEILING: |fan - raw anchors|, by denoise steps ===")
anch = anchors_of(cold)
ceil = {}
for st in (0, 2, 4, 8):
    mag = (fans_at(cold, st) - anch).norm(dim=-1)
    ceil[st] = (float(mag.mean()), float(mag.max()), float(mag[..., -1].mean()))
    print(f"  steps={st}: mean displacement {ceil[st][0]:.4f} m · max {ceil[st][1]:.3f} m"
          f" · at the 2 s endpoint {ceil[st][2]:.4f} m")

print("\n=== REACH-TRAINED: how far 2,000 RL steps actually moved the offsets ===")
base2 = fans_at(cold, 2)
trained = {}
for arm in ("rerun-t1-w1-prox", "rerun-t2-w10-prox", "sweep-s0-w0"):
    try:
        ck = torch.load(f"{O}/{arm}/ckpt_after.pt", map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(f"  {arm:<20} no ckpt_after.pt (pre-TRAIN-C5 arm) — skipped"); continue
    cold.load_state_dict(ck["model"]); cold.to(dev).eval()
    d = (fans_at(cold, 2) - base2).norm(dim=-1)
    trained[arm] = float(d.mean())
    print(f"  {arm:<20} mean |delta FAN| {d.mean():.4f} m · max {d.max():.3f} m")
# restore
cold.load_state_dict(torch.load(r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt",
                                map_location="cpu", weights_only=False)["model"])
cold.to(dev).eval()

print("\n=== THE GAP: extra clearance a near-miss candidate must gain ===")
fan_gaps = []
for stem, t0 in wins:
    w = src.window(stem, t0)
    if not w["obs"]:
        continue
    b = P.collate([w], dev)
    with torch.no_grad():
        fan = cold(b["frames"], None, b["v0"], steps=2)["anchor_traj"][0].cpu()
    obs = torch.tensor(w["obs"], dtype=torch.float32)
    d = (fan.unsqueeze(-2) - obs.view(1, 1, -1, 2)).norm(dim=-1)
    clear = (d - (EGO_R + OBS_R)).clamp_min(0.0).amin(dim=-1).amin(dim=-1)   # [N]
    near = clear[clear < D_SAFE]
    if near.numel():
        fan_gaps.append((D_SAFE - near).numpy())
gaps = np.concatenate(fan_gaps) if fan_gaps else np.array([0.0])
print(f"  near-miss candidates: {len(gaps):,} · mean gap to clear d_safe="
      f"{D_SAFE} m: {gaps.mean():.3f} m · median {np.median(gaps):.3f} m"
      f" · p25 {np.percentile(gaps,25):.3f} m")

print("\n=== THE RATIO THAT DECIDES IT ===")
reach = trained.get("rerun-t1-w1-prox", float("nan"))
print(f"  REACH-TRAINED (2,000 steps, w=1)   {reach:.4f} m")
print(f"  REACH-CEILING (fan displacement from raw anchors, steps=2)  {ceil[2][0]:.4f} m")
print(f"  GAP (median, to leave the near-miss band)         {np.median(gaps):.4f} m")
print(f"  -> trained reach is {100*reach/max(np.median(gaps),1e-9):.2f}% of the median gap")
print(f"  -> even the FULL offset is {100*ceil[2][0]/max(np.median(gaps),1e-9):.2f}% of it")
print(f"\n  reach vs denoise steps: " +
      " ".join(f"s{k}={v[0]:.4f}m" for k, v in ceil.items()))
json.dump({"ceiling": {str(k): v for k, v in ceil.items()}, "trained": trained,
           "gap_mean": float(gaps.mean()), "gap_median": float(np.median(gaps)),
           "gap_p25": float(np.percentile(gaps, 25)), "n_near_miss": int(len(gaps)),
           "d_safe_m": D_SAFE, "n_windows": len(wins)},
          open(f"{O}/reachable_set.json", "w"), indent=1)
print(f"-> {O}/reachable_set.json")
