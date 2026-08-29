"""Did `proximity` itself improve? R1 cannot answer — it excludes proximity.

The readout measures R1 with the DEFAULT spec for cross-arm comparability, so
the new term's own before/after is invisible in it. That is a gap in AMENDMENT
2's exit B as written ("proximity contributes a non-trivial share of dR1"):
unmeasurable by construction. Closed here using the saved ckpt_after.pt — which
exists only because TRAIN-C5 forced checkpoint saving.
"""
import importlib.util, sys, json
import numpy as np, torch
sys.path.insert(0, "stack")
spec = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
from tanitad.rl import rewards as R, PostTrainConfig

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
dev = "cuda"
cfg = PostTrainConfig(decoder_steps=2, dt=P.DT_TRAJ)
src = P.WindowSource(r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
                     f"{O}/pilot_val_agents.jsonl", seed=0, lru=60)
W = dict(R.DEFAULT_WEIGHTS); W["proximity"] = 0.5
spec_full = R.RewardSpec(weights=W, dt=P.DT_TRAJ)

rng = np.random.default_rng(1234)
picks = []
for stem in src.stems:
    T = int(src._ep(stem)["poses"].shape[0])
    lo, hi = P.WINDOW - 1, T - P.HORIZONS[-1] - 1
    for t0 in sorted(rng.choice(np.arange(lo, hi), size=min(8, hi - lo), replace=False)):
        picks.append((stem, int(t0)))
picks = picks[:120]


@torch.no_grad()
def measure(model):
    model.eval()
    per = {}
    for stem, t0 in picks:
        w = src.window(stem, t0)
        b = P.collate([w], dev)
        out = model(b["frames"], None, b["v0"], steps=2)
        fan = out["anchor_traj"]
        ctx = P.build_ctx(b)
        parts = spec_full.per_component(fan, ctx)
        e = per.setdefault(stem, {k: [] for k in parts})
        for k, v in parts.items():
            e[k].append(float(v.mean()))
    return {k: float(np.mean([np.mean(v[k]) for v in per.values()])) for k in
            next(iter(per.values()))}


cold = P.load_model(r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt", dev)
base = measure(cold)
print("component means on the val fan (UNWEIGHTED):")
print(f"{'arm':<16}" + "".join(f"{k[:9]:>12}" for k in sorted(base)))
print(f"{'cold start':<16}" + "".join(f"{base[k]:>+12.4f}" for k in sorted(base)))
rows = {}
for arm in ("rerun-t1-w1-prox", "rerun-t2-w10-prox"):
    ck = torch.load(f"{O}/{arm}/ckpt_after.pt", map_location="cpu", weights_only=False)
    cold.load_state_dict(ck["model"]); cold.to(dev)
    m = measure(cold)
    rows[arm] = m
    print(f"{arm.replace('rerun-',''):<16}" + "".join(f"{m[k]:>+12.4f}" for k in sorted(base)))
print("\nWEIGHTED DELTA vs cold start (what the training actually bought):")
print(f"{'arm':<16}" + "".join(f"{k[:9]:>12}" for k in sorted(base)))
for arm, m in rows.items():
    print(f"{arm.replace('rerun-',''):<16}"
          + "".join(f"{W[k]*(m[k]-base[k]):>+12.4f}" for k in sorted(base)))
json.dump({"cold": base, **rows}, open(f"{O}/prox_delta.json", "w"), indent=1)
