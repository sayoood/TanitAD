"""WP-N — the permuted-prior baseline, done properly, with intervals on every contrast.

⛔⛔ THE DEFECT IN WP-M's PERMUTED CONTROL, AND WHY THE 89 % FIGURE IS SUSPECT.
WP-M reported that a prior "from a different window" captures ~89 % of REF-C's
port benefit, and drew the conclusion that the port carries almost no
scene-specific information. That control was `roll_control(energy, shift=1)` --
a roll by ONE ROW. But the rows are accumulated EPISODE BY EPISODE, so row i-1 is
almost always the SAME EPISODE and usually the ADJACENT TIMESTEP. A prior from
0.1 s earlier in the same clip is not a wrong-scene prior; it is very nearly the
RIGHT one.

⇒ the control was close to a no-op, which would make the real prior's margin over
it look small for a reason that has nothing to do with scene-specificity. Same
family as every other defect this campaign: two things compared without the
affordance that was supposed to differ actually differing.

⭐ THIS RUN MEASURES BOTH, SO THE SIZE OF THAT ARTIFACT IS A NUMBER RATHER THAN A
WORRY:

    roll-1        the WP-M control, retained purely as the diagnostic
    cross-episode a prior drawn from a DIFFERENT EPISODE (the honest control)
    norm-matched  random at the same per-window spread (the magnitude floor)

⛔ AND EVERY CONTRAST GETS A PAIRED EPISODE-CLUSTER INTERVAL. WP-M's key number
(+0.0012 m) had none; a point estimate an order of magnitude inside the total
effect's own CI is not evidence of anything. The contrasts:

    real - base            does the port help at all?
    cross-episode - base   how much of that is GENERIC (any real prior)?
    real - cross-episode   how much is SCENE-SPECIFIC?          <- the bar
    norm-matched - base    does a random prior of equal size help?

⭐ The tensors are BANKED to disk, so any further contrast costs no GPU.

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

from tanitad.instruments.cot_influence import (                          # noqa: E402
    influence, norm_matched_null, rerank_logits, roll_control,
)

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PER_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_DRAW = int(sys.argv[2]) if len(sys.argv) > 2 else 20
JOIN = O + "/pilot_val_agents_ext.jsonl"
BANK = O + "/wpn_port_tensors.pt"
CKPT = r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt"

from tanitad.refs import refc as _refc                                   # noqa: E402


def build(zero_port):
    m = _refc.RefCModel(_refc.refc_config())
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    r = m.load_state_dict(ck, strict=False)
    assert set(r.missing_keys) == {"decoder.anchor_controls"}, r.missing_keys
    port = m.decoder.maneuver_to_anchor
    if port is None:
        raise RuntimeError("no maneuver_to_anchor on this checkpoint")
    n0 = float(port.weight.detach().norm())
    if zero_port:
        with torch.no_grad():
            port.weight.zero_()
            if port.bias is not None:
                port.bias.zero_()
    return m.to(DEV).eval(), n0


if os.path.exists(BANK):
    d = torch.load(BANK, map_location="cpu", weights_only=False)
    S0, S1, ADE, EPS, port_norm = d["S0"], d["S1"], d["ADE"], d["EPS"], d["port_l2"]
    print("[wpn] loaded banked tensors: %s windows" % len(EPS), flush=True)
else:
    m1, port_norm = build(False)
    m0, _ = build(True)
    print("[wpn] maneuver_to_anchor L2 = %.6f" % port_norm, flush=True)
    raw = collections.defaultdict(dict)
    for line in _io.open(JOIN, encoding="utf-8"):
        j = json.loads(line)
        raw[j["clip_id"]][j["frame_idx"]] = j["agents"]
    src = P.WindowSource(
        r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
        JOIN, seed=0, lru=6)
    S0, S1, ADE, EPS = [], [], [], []
    rng = np.random.default_rng(1234)
    t0 = time.time()
    with torch.no_grad():
        for si, stem in enumerate(src.stems):
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
                o1 = m1(b["frames"], None, b["v0"], steps=2)
                o0 = m0(b["frames"], None, b["v0"], steps=2)
                fan, gt = o1["anchor_traj"][0], b["gt_traj"][0]
                S1.append(o1["sel_score"][0].float().cpu())
                S0.append(o0["sel_score"][0].float().cpu())
                ADE.append((fan - gt.unsqueeze(0)).norm(dim=-1).mean(dim=-1)
                           .float().cpu())
                EPS.append(stem)
            print("[wpn] ep %d/%d %s: %d windows (%.0f s)"
                  % (si + 1, len(src.stems), stem, len(EPS), time.time() - t0),
                  flush=True)
    S0, S1, ADE = torch.stack(S0), torch.stack(S1), torch.stack(ADE)
    torch.save({"S0": S0, "S1": S1, "ADE": ADE, "EPS": EPS,
                "port_l2": port_norm, "_ckpt": CKPT, "_join": JOIN}, BANK)
    print("[wpn] banked tensors -> %s" % BANK, flush=True)

E = np.array(EPS)
UE = sorted(set(EPS))
N, K = S0.shape
energy = -(S1 - S0)
assert torch.allclose(rerank_logits(S0, energy, 1.0), S1, atol=1e-5)
print("\n[wpn] n=%d windows - %d episodes - K=%d" % (N, len(UE), K), flush=True)

# ---- how BAD was the roll-1 control? Measure it, do not assume ---------------
ep_idx = {e: i for i, e in enumerate(UE)}
epn = np.array([ep_idx[e] for e in EPS])
same_ep = float((epn == np.roll(epn, 1)).mean())
print("[wpn] roll-1 pairs a window with the SAME EPISODE in %.1f %% of rows"
      % (100 * same_ep))
print("      => WP-M's 'different window' control was mostly the SAME CLIP, "
      "usually the adjacent timestep", flush=True)


def cross_episode_perm(r):
    """Pair every row with a row from a DIFFERENT EPISODE.

    ⛔ This is what "a prior from another scene" has to mean. Rejection-samples
    per row and asserts the result, so a silent failure to mispair cannot pass
    as a control -- the failure mode `roll_control` hit here.
    """
    idx = np.empty(N, dtype=int)
    for i in range(N):
        j = int(r.integers(0, N))
        tries = 0
        while epn[j] == epn[i] and tries < 200:
            j = int(r.integers(0, N))
            tries += 1
        idx[i] = j
    assert (epn[idx] != epn).all(), "cross-episode permutation left same-episode rows"
    return idx


b = torch.arange(N)


def ade_of(en):
    return ADE[b, rerank_logits(S0, en, 1.0).argmax(-1)]


ade_base = ADE[b, S0.argmax(-1)]
ade_real = ade_of(energy)
r = np.random.default_rng(0)
ade_roll = ade_of(roll_control(energy, 1))
cross = torch.stack([ade_of(energy[torch.from_numpy(cross_episode_perm(r))])
                     for _ in range(N_DRAW)]).mean(0)
nm = torch.stack([ade_of(norm_matched_null(
    energy, torch.Generator().manual_seed(s))) for s in range(N_DRAW)]).mean(0)

print("\n%-34s%10s" % ("arm (ADE, lower is better)", "metres"))
for tag, v in (("base (port zeroed)", ade_base), ("REAL prior", ade_real),
               ("roll-1 prior  [DIAGNOSTIC ONLY]", ade_roll),
               ("cross-episode prior  [the control]", cross),
               ("norm-matched random prior", nm)):
    print("%-34s%10.4f" % (tag, float(v.mean())))


def boot(x, y, iters=4000, seed=0):
    """PAIRED episode-cluster bootstrap on mean(x) - mean(y)."""
    rr = np.random.default_rng(seed)
    ix = {e: np.where(E == e)[0] for e in UE}
    d = []
    for _ in range(iters):
        pick = np.concatenate([ix[UE[i]] for i in rr.integers(0, len(UE), len(UE))])
        d.append(float((x[pick] - y[pick]).mean()))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


print("\n%-40s%10s%26s" % ("paired contrast (ADE, m)", "delta", "CI95"))
res = {}
for tag, x, y in (("real - base", ade_real, ade_base),
                  ("cross-episode - base", cross, ade_base),
                  ("real - cross-episode  [SCENE-SPECIFIC]", ade_real, cross),
                  ("real - roll-1  [WP-M's number]", ade_real, ade_roll),
                  ("norm-matched - base", nm, ade_base)):
    m, lo, hi = boot(x, y)
    sep = "SEPARATED" if lo > 0 or hi < 0 else "overlaps 0"
    res[tag] = {"delta": m, "ci": [lo, hi], "separated": sep == "SEPARATED"}
    print("%-40s%+10.4f   [%+.4f, %+.4f] %s" % (tag, m, lo, hi, sep))

gen = float((cross.mean() - ade_base.mean()))
tot = float((ade_real.mean() - ade_base.mean()))
frac = gen / tot if abs(tot) > 1e-12 else float("nan")
print("\n  generic share  = (cross-episode - base) / (real - base) = %.1f %%"
      % (100 * frac))
print("  ! a share is only meaningful if the DENOMINATOR is separated from 0; "
      "'real - base' above says whether it is.")

inf_real = influence(S0, S1)
print("\n  INF (real port): flip %.4f  kl %.6f" % (inf_real.flip_rate, inf_real.kl))
json.dump({"n": N, "K": K, "episodes": len(UE), "n_draw": N_DRAW,
           "port_l2": port_norm, "roll1_same_episode_frac": same_ep,
           "ade": {"base": float(ade_base.mean()), "real": float(ade_real.mean()),
                   "roll1": float(ade_roll.mean()), "cross_episode": float(cross.mean()),
                   "norm_matched": float(nm.mean())},
           "contrasts": res, "generic_share": frac,
           "influence": {"flip": inf_real.flip_rate, "kl": inf_real.kl},
           "_tier": "T0; NON-PARITY pilot; paired episode-cluster bootstrap, "
                    "4000 iters; cross-episode control averaged over %d draws" % N_DRAW,
           "_supersedes": "wpm permuted-prior control (roll-1, mostly same episode)"},
          _io.open(O + "/wpn_port_contrasts.json", "w", encoding="utf-8"), indent=1)
print("\n-> wpn_port_contrasts.json")
