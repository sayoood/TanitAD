"""H-TL-1 PRECONDITION — is there headroom in anchor SELECTION at all?

⛔ WHY THIS RUNS BEFORE ANY REASONER IS TRAINED. H-TL-1 says "reasoned priors beat
the current heads on selection quality". That experiment can only show something if
CHOOSING A DIFFERENT ANCHOR would help. If the scorer is already near the best
anchor the fan contains, no prior — reasoned or otherwise — can move the metric, and
training a reasoner would produce a null that says nothing about reasoning.

⭐ THE THREE QUANTITIES, and the gap between them is the whole answer:

  SELECTED    ADE of the anchor the scorer picks           (what we ship)
  ORACLE      ADE of the BEST anchor in the fan            (the ceiling a prior can reach)
  MEAN        ADE averaged over the fan                    (the floor: choosing at random)

  headroom  = SELECTED - ORACLE      how much better selection could get
  skill     = MEAN - SELECTED        how much the scorer already earns over chance

⚠️ CONTROLS, because a headroom number is meaningless without them:
  * MEAN is the chance floor — if SELECTED ~= MEAN the scorer is not selecting at all.
  * ORACLE is computed on the SAME fan the scorer saw, so the comparison is paired.
  * WORST is reported too: if ORACLE ~= WORST the fan is degenerate and the whole
    anchor mechanism is inert on this corpus, which would be a different finding.

Tier: T0. NON-PARITY pilot corpus. Evidence class: MEASURED (ours).
"""
import importlib.util
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "stack")
_s = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(_s)
_s.loader.exec_module(P)

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ⚠️ CONTENT-VERIFIED LOAD, not a loosened one. The current config registers a
# BUFFER `decoder.anchor_controls` [128,2] that this checkpoint predates
# (v0_conditioned=False, so it is unused). Diagnosed exactly:
#   missing keys      = {decoder.anchor_controls}  — a buffer, not a parameter
#   unexpected keys   = {} (none)
#   state_dict delta  = 104,240,267 - 104,240,011 = 256 = 128 x 2  <- exactly the buffer
# strict=False is therefore safe HERE and the assertions below prove it rather
# than assume it. A bare strict=False would hide a real architecture mismatch.
from tanitad.refs import refc as _refc
model = _refc.RefCModel(_refc.refc_config())
_ck = torch.load(r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt",
                 map_location="cpu", weights_only=False)["model"]
_r = model.load_state_dict(_ck, strict=False)
assert set(_r.missing_keys) == {"decoder.anchor_controls"}, _r.missing_keys
assert not _r.unexpected_keys, _r.unexpected_keys
assert sum(p.numel() for p in model.parameters()) == 104_191_577
model = model.to(dev)
model.eval()
src = P.WindowSource(
    r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
    f"{O}/pilot_val_agents.jsonl", seed=0, lru=60)

rng = np.random.default_rng(1234)
wins = []
for stem in src.stems:
    T = int(src._ep(stem)["poses"].shape[0])
    lo, hi = P.WINDOW - 1, T - P.HORIZONS[-1] - 1
    for t0 in sorted(rng.choice(np.arange(lo, hi), size=min(8, hi - lo), replace=False)):
        wins.append((stem, int(t0)))

sel_ade, orc_ade, mean_ade, worst_ade, rank_of_best = [], [], [], [], []
per_ep = {}

with torch.no_grad():
    for stem, t0 in wins:
        w = src.window(stem, t0)
        if w is None:
            continue
        b = P.collate([w], dev)
        out = model(b["frames"], None, b["v0"], steps=2)
        fan = out["anchor_traj"][0]                      # [N, S, 2]
        gt = b["gt_traj"][0]                             # [S, 2]
        ade = (fan - gt.unsqueeze(0)).norm(dim=-1).mean(dim=-1)   # [N]
        scores = out["sel_score"][0]                     # [N]
        sel = int(scores.argmax())
        best = int(ade.argmin())
        # where does the BEST anchor sit in the scorer's ranking? (1 = perfect)
        order = torch.argsort(scores, descending=True)
        rank = int((order == best).nonzero()[0, 0]) + 1

        sel_ade.append(float(ade[sel])); orc_ade.append(float(ade[best]))
        mean_ade.append(float(ade.mean())); worst_ade.append(float(ade.max()))
        rank_of_best.append(rank)
        e = per_ep.setdefault(stem, {"sel": [], "orc": [], "mean": []})
        e["sel"].append(float(ade[sel])); e["orc"].append(float(ade[best]))
        e["mean"].append(float(ade.mean()))

S, Or, M, W = map(np.array, (sel_ade, orc_ade, mean_ade, worst_ade))
R = np.array(rank_of_best)
n_anchors = int(out["anchor_traj"].shape[1])

print(f"windows {len(S)} · anchors/fan {n_anchors} · steps=2 · eval mode\n")
print(f"  WORST anchor      {W.mean():7.4f} m")
print(f"  MEAN over fan     {M.mean():7.4f} m   <- chance floor")
print(f"  SELECTED (scorer) {S.mean():7.4f} m")
print(f"  ORACLE (best)     {Or.mean():7.4f} m   <- ceiling any prior can reach")
print()
print(f"  skill    = MEAN - SELECTED = {M.mean()-S.mean():+7.4f} m "
      f"({100*(M.mean()-S.mean())/M.mean():.1f}% of the chance floor)")
print(f"  HEADROOM = SELECTED - ORACLE = {S.mean()-Or.mean():+7.4f} m "
      f"({100*(S.mean()-Or.mean())/S.mean():.1f}% of what we ship)")
print()
print(f"  rank of the best anchor in the scorer's ranking: "
      f"median {int(np.median(R))} / {n_anchors}  ·  top-1 {100*(R==1).mean():.1f}%"
      f"  ·  top-5 {100*(R<=5).mean():.1f}%  ·  top-10 {100*(R<=10).mean():.1f}%")

# paired episode-cluster bootstrap on the HEADROOM (the decision-grade interval)
eps = sorted(per_ep)
d = np.array([np.mean(per_ep[e]["sel"]) - np.mean(per_ep[e]["orc"]) for e in eps])
br = np.random.default_rng(7)
bs = np.array([d[br.integers(0, d.size, d.size)].mean() for _ in range(4000)])
lo_, hi_ = np.percentile(bs, [2.5, 97.5])
print(f"\n  headroom paired 95% CI over {len(eps)} episodes: "
      f"[{lo_:+.4f}, {hi_:+.4f}] "
      f"{'SEPARATED' if (lo_>0)==(hi_>0) else 'not separated'}")

verdict = ("HEADROOM EXISTS — H-TL-1 is worth running" if S.mean()-Or.mean() > 0.05
           else "NO HEADROOM — a better prior cannot move selection; H-TL-1 would return a null")
print(f"\n  => {verdict}")
json.dump({"n_windows": int(len(S)), "n_anchors": n_anchors,
           "worst": float(W.mean()), "mean_fan": float(M.mean()),
           "selected": float(S.mean()), "oracle": float(Or.mean()),
           "headroom": float(S.mean()-Or.mean()),
           "skill_over_chance": float(M.mean()-S.mean()),
           "headroom_ci95": [float(lo_), float(hi_)],
           "rank_top1": float((R == 1).mean()), "rank_top5": float((R <= 5).mean()),
           "rank_median": int(np.median(R)), "verdict": verdict,
           "_tier": "T0; NON-PARITY pilot corpus; eval-mode deterministic"},
          open(f"{O}/htl1_headroom.json", "w"), indent=1)
print(f"-> {O}/htl1_headroom.json")
