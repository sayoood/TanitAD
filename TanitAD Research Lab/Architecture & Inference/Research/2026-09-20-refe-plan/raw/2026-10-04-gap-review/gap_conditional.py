#!/usr/bin/env python3
"""REVIEW 7 -- EP lever conditional on fan cleanliness (CPU, navsim venv python). EXPLORATORY.
(a) oracle bound: leak-free GT EP used ONLY on tokens whose fan is >= 95 % NAVSIM-safe (GT-defined, not deployable);
(b) deployable: the scorer's OWN belief of fan cleanliness, b = share of proposals with p_nc*p_dac >= 0.9; when b >= f,
    select with EP := path length / longest believed-safe proposal (D5-style) and TTC tempered (p_ttc ** gamma);
    else the shipped rule. (f, gamma) chosen on one log half, scored on the other (cross-fit)."""
import hashlib, json
import numpy as np
import gap_selection as GS

OUT = GS.PKG + "/raw/2026-10-04-gap-review/gap_conditional.json"
z = np.load(GS.PROPS, allow_pickle=True)
tok = z["token"]; props = z["proposals"].astype(np.float64); logit = z["logits"]; pick = z["pick"]
N = len(tok); pos = {t: i for i, t in enumerate(tok)}
G = {k: np.full((N, 64), np.nan) for k in ("nc", "dac", "ttc", "c", "pdms", "ep")}
logs = np.empty(N, dtype=object)
for line in open(GS.CENSUS):
    r = json.loads(line); i = pos[r["token"]]; logs[i] = r["log"]
    for k in ("nc", "dac", "ttc", "c", "pdms"):
        G[k][i] = np.asarray(r[k][1:65], float)
    m = G["nc"][i] * G["dac"][i]
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(m > 0, (12 * G["pdms"][i] / np.where(m > 0, m, 1) - 5 * G["ttc"][i] - 2 * G["c"][i]) / 5, 0.0)
    G["ep"][i] = np.clip(e, 0, 1)
logs = logs.astype(str)
P = GS.sig(logit); pnc, pdac, pep, pttc, pc = (P[..., k] for k in range(5))
rows = np.arange(N); base = G["pdms"][rows, pick]
safe = (G["nc"] * G["dac"]) == 1.0
clean_gt = safe.mean(1) >= 0.95
a0 = GS.agg(pnc, pdac, pep, pttc, pc)
ep_lf = np.where(safe, G["ep"], pep)
a_or = GS.agg(pnc, pdac, ep_lf, pttc, pc)
sel_or = np.where(clean_gt, a_or.argmax(1), pick)
out = {"oracle_leakfree_EP_on_GT_clean_fans_only": GS.boot_paired(G["pdms"][rows, sel_or], base, logs),
       "n_GT_clean_tokens": int(clean_gt.sum())}
xy = props[..., :2]
L = np.linalg.norm(np.diff(np.concatenate([np.zeros((N, 64, 1, 2)), xy], 2), axis=2), axis=-1).sum(-1)
bel = (pnc * pdac) >= 0.9
b = bel.mean(1)
Ls = np.where(bel, L, 0.0).max(1, keepdims=True)
ep5 = np.where(Ls > 5.0, np.minimum(L / np.maximum(Ls, 1e-6), 1.0), 1.0)
half = np.array([int(hashlib.sha256(l.encode()).hexdigest(), 16) % 2 for l in logs])
knobs = [(f, g) for f in (0.6, 0.7, 0.8, 0.9, 0.95) for g in (1.0, 0.5, 0.25)]
sc = {}
for f, g in knobs:
    a = GS.agg(pnc, pdac, ep5, np.power(pttc, g), pc)
    sel = np.where(b >= f, a.argmax(1), pick)
    sc[(f, g)] = G["pdms"][rows, sel]
final = np.empty(N); chosen = {}
for h in (0, 1):
    fit = half == (1 - h)
    k = max(knobs, key=lambda kk: sc[kk][fit].mean())
    chosen[str(h)] = list(k); final[half == h] = sc[k][half == h]
out["deployable_conditional_crossfit"] = GS.boot_paired(final, base, logs)
out["deployable_knob_chosen_per_scored_half"] = chosen
out["deployable_per_knob_full_set_x100_NOT_crossfit"] = {f"f={f},gamma={g}": float(v.mean() * 100) for (f, g), v in sc.items()}
out["belief_clean_rate_at_f0.9"] = float((b >= 0.9).mean())
out["agreement_belief_clean_f0.9_vs_GT_clean"] = float(((b >= 0.9) == clean_gt).mean())
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
