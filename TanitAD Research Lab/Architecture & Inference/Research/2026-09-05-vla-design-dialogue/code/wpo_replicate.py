"""WP-O — replicate WP-N's separated contrast, and check it is not a few episodes.

⛔ WHICH REPLICATE IS EVEN AVAILABLE HERE, ESTABLISHED BEFORE ONE IS RUN.
`CLAUDE.md` names three variances an episode-cluster bootstrap is blind to:
another draw of EPISODES (what the bootstrap answers), another TRAINING run
(`H-ESTIM-SEED-1`), and another INFERENCE run (the stochastic-planner floor).

  * INFERENCE seed: **NOT APPLICABLE**. `refc.py:2091` -- "noise only in training
    (deterministic at eval so decoding is reproducible)". REF-C's forward pass at
    eval is deterministic, so re-running the same windows reproduces S0 and S1
    bit-for-bit. Claiming an inference-seed replicate here would be theatre.
  * TRAINING seed: out of scope -- one checkpoint exists.
  * WINDOW SAMPLE: **this is the live one.** WP-N drew 40 of ~180 available
    windows per episode with `default_rng(1234)`. The episode bootstrap resamples
    EPISODES and never asks whether a different draw of WINDOWS WITHIN them would
    say the same thing.

⇒ this run re-extracts with a different window seed and asks whether
`real - cross-episode` survives.

⭐ AND A SECOND CHECK THAT COSTS NOTHING: a per-episode breakdown. A pooled
contrast of -0.0254 m could be broad-based, or it could be three episodes with
large effects and 31 with none. The bootstrap's interval does not distinguish
those, and they mean different things for a design decision.

Tier T0. NON-PARITY pilot. Evidence class MEASURED (ours).
"""
import collections
import importlib.util
import io as _io
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "stack")
_s = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(_s)
_s.loader.exec_module(P)
from tanitad.instruments.cot_influence import rerank_logits              # noqa: E402

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
JOIN = O + "/pilot_val_agents_ext.jsonl"
CKPT = r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt"
MAX_PER_EP = 40
N_DRAW = 20
SEEDS = [1234, 4321]

from tanitad.refs import refc as _refc                                   # noqa: E402


def build(zero_port):
    m = _refc.RefCModel(_refc.refc_config())
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    r = m.load_state_dict(ck, strict=False)
    assert set(r.missing_keys) == {"decoder.anchor_controls"}
    port = m.decoder.maneuver_to_anchor
    if zero_port:
        with torch.no_grad():
            port.weight.zero_()
            if port.bias is not None:
                port.bias.zero_()
    return m.to(DEV).eval()


def extract(seed):
    bank = O + "/wpn_port_tensors.pt" if seed == 1234 else \
        O + "/wpo_port_tensors_s%d.pt" % seed
    if os.path.exists(bank):
        d = torch.load(bank, map_location="cpu", weights_only=False)
        print("[wpo] seed %d: loaded %d banked windows" % (seed, len(d["EPS"])),
              flush=True)
        return d["S0"], d["S1"], d["ADE"], d["EPS"]
    m1, m0 = build(False), build(True)
    raw = collections.defaultdict(dict)
    for line in _io.open(JOIN, encoding="utf-8"):
        j = json.loads(line)
        raw[j["clip_id"]][j["frame_idx"]] = j["agents"]
    src = P.WindowSource(
        r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
        JOIN, seed=0, lru=6)
    S0, S1, ADE, EPS = [], [], [], []
    rng = np.random.default_rng(seed)
    t0 = time.time()
    with torch.no_grad():
        for stem in src.stems:
            key = next((k for k in raw if k in stem or stem in k), None)
            if key is None:
                continue
            T = int(src._ep(stem)["poses"].shape[0])
            lo, hi = P.WINDOW - 1, T - P.HORIZONS[-1] - 1
            for t in sorted(rng.choice(np.arange(lo, hi),
                                       size=min(MAX_PER_EP, hi - lo), replace=False)):
                w = src.window(stem, int(t))
                if w is None:
                    continue
                b = P.collate([w], DEV)
                o1, o0 = m1(b["frames"], None, b["v0"], steps=2), \
                    m0(b["frames"], None, b["v0"], steps=2)
                fan, gt = o1["anchor_traj"][0], b["gt_traj"][0]
                S1.append(o1["sel_score"][0].float().cpu())
                S0.append(o0["sel_score"][0].float().cpu())
                ADE.append((fan - gt.unsqueeze(0)).norm(dim=-1).mean(dim=-1)
                           .float().cpu())
                EPS.append(stem)
    S0, S1, ADE = torch.stack(S0), torch.stack(S1), torch.stack(ADE)
    torch.save({"S0": S0, "S1": S1, "ADE": ADE, "EPS": EPS}, bank)
    print("[wpo] seed %d: extracted %d windows (%.0f s) -> banked"
          % (seed, len(EPS), time.time() - t0), flush=True)
    return S0, S1, ADE, EPS


def analyse(S0, S1, ADE, EPS, seed):
    E = np.array(EPS)
    UE = sorted(set(EPS))
    N = len(EPS)
    epn = np.array([UE.index(e) for e in EPS])
    energy = -(S1 - S0)
    b = torch.arange(N)

    def ade_of(en):
        return ADE[b, rerank_logits(S0, en, 1.0).argmax(-1)]

    def cross(r):
        idx = np.empty(N, dtype=int)
        for i in range(N):
            j = int(r.integers(0, N))
            k = 0
            while epn[j] == epn[i] and k < 200:
                j = int(r.integers(0, N))
                k += 1
            idx[i] = j
        assert (epn[idx] != epn).all()
        return idx

    r = np.random.default_rng(0)
    a_real = ade_of(energy)
    a_cross = torch.stack([ade_of(energy[torch.from_numpy(cross(r))])
                           for _ in range(N_DRAW)]).mean(0)
    a_base = ADE[b, S0.argmax(-1)]
    diff = (a_real - a_cross).numpy()

    rr = np.random.default_rng(0)
    ix = {e: np.where(E == e)[0] for e in UE}
    boots = []
    for _ in range(4000):
        pick = np.concatenate([ix[UE[i]] for i in rr.integers(0, len(UE), len(UE))])
        boots.append(float(diff[pick].mean()))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    per_ep = {e: float(diff[ix[e]].mean()) for e in UE}
    neg = sum(1 for v in per_ep.values() if v < 0)
    return {"seed": seed, "n": N, "episodes": len(UE),
            "ade_base": float(a_base.mean()), "ade_real": float(a_real.mean()),
            "ade_cross": float(a_cross.mean()),
            "delta": float(diff.mean()), "ci": [lo, hi],
            "separated": bool(lo > 0 or hi < 0),
            "episodes_favouring_real": neg, "per_episode": per_ep}


res = {}
for sd in SEEDS:
    res[sd] = analyse(*extract(sd), sd)

print("\n%-8s%8s%10s%10s%10s%11s%24s" % ("seed", "n", "base", "real", "cross",
                                         "delta", "CI95"))
for sd in SEEDS:
    r = res[sd]
    print("%-8d%8d%10.4f%10.4f%10.4f%+11.4f   [%+.4f, %+.4f] %s"
          % (sd, r["n"], r["ade_base"], r["ade_real"], r["ade_cross"], r["delta"],
             r["ci"][0], r["ci"][1], "SEPARATED" if r["separated"] else "overlaps 0"))

print("\nPER-EPISODE BREADTH (is the pooled contrast broad-based?)")
for sd in SEEDS:
    r = res[sd]
    v = np.array(sorted(r["per_episode"].values()))
    print("  seed %d: %d/%d episodes favour the REAL prior | "
          "median %+.4f | worst %+.4f | best %+.4f"
          % (sd, r["episodes_favouring_real"], r["episodes"],
             float(np.median(v)), float(v[-1]), float(v[0])))

both = all(res[s]["separated"] for s in SEEDS)
same_sign = len({np.sign(res[s]["delta"]) for s in SEEDS}) == 1
verdict = ("REPLICATES - separated with the same sign under both window draws"
           if both and same_sign else
           "DOES NOT REPLICATE - the contrast is sensitive to which windows were "
           "drawn, and WP-N's separated interval should not be quoted alone")
print("\n  => %s" % verdict)
json.dump({"_what": "replicate of WP-N's real-vs-cross-episode contrast under a "
                    "second WINDOW-SAMPLING seed; the inference seed is NOT a "
                    "variance source here (refc.py:2091, eval is deterministic)",
           "seeds": SEEDS, "results": {str(k): {kk: vv for kk, vv in v.items()
                                                if kk != "per_episode"}
                                       for k, v in res.items()},
           "per_episode": {str(k): v["per_episode"] for k, v in res.items()},
           "replicates": bool(both and same_sign), "verdict": verdict,
           "_tier": "T0; NON-PARITY pilot; paired episode-cluster bootstrap 4000 it"},
          _io.open(O + "/wpo_replicate.json", "w", encoding="utf-8"), indent=1)
print("\n-> wpo_replicate.json")
