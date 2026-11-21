"""WP-B — is there a driving analogue of TRM's `puzzle_id`?

⛔ THE TRANSFER RISK THIS TESTS. TRM's ablation: replacing the puzzle ID with a
blank or random token yields ZERO accuracy — ARC Prize calls it "a strong, limiting
dependency" and recommends "context-derived task conditioning that generalises to
unseen tasks". In ARC a puzzle's identity appears in train AND test. In driving every
window is a new puzzle with no identity. If the recursion's power needs to know WHICH
problem it is, and driving cannot say, the mechanism does not transfer.

⭐ THE QUESTION, made concrete and cheap: does a CONTEXT-DERIVED situation code carry
information about WHICH of the shortlisted candidates is the right one? If yes, the
analogue exists and the recursion has something to condition on. If it reads at
chance, the puzzle-ID role has no driving substitute and the recursive-reasoner
direction is weaker than it looks.

⛔ THE CODE IS DERIVED FROM THE SCENE, NEVER FROM THE SITUATION CLASSIFIER. Binding
PI ruling 2026-08-03: the goal input may not carry the classifier's output in any
form. These features are raw geometry and counts read off the window itself.

⚠️ CONTROLS — the whole result is unreadable without them:
  * CHANCE      1/K, the floor a code must beat
  * SHUFFLED    the same codes, permuted across windows: MUST collapse to chance,
                or the "signal" is an artifact of the fit, not of the code
  * CONSTANT    a single constant code: must read exactly chance
  * n and d printed; fit and score splits are DISJOINT by episode

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
K = 5                                    # the shortlist WP-C re-ranks

from tanitad.refs import refc as _refc                                    # noqa: E402
model = _refc.RefCModel(_refc.refc_config())
_ck = torch.load(r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt",
                 map_location="cpu", weights_only=False)["model"]
_r = model.load_state_dict(_ck, strict=False)
assert set(_r.missing_keys) == {"decoder.anchor_controls"} and not _r.unexpected_keys
model = model.to(dev).eval()

src = P.WindowSource(
    r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
    f"{O}/pilot_val_agents.jsonl", seed=0, lru=60)

rng = np.random.default_rng(1234)
rows, labels, eps = [], [], []
with torch.no_grad():
    for stem in src.stems:
        T = int(src._ep(stem)["poses"].shape[0])
        lo, hi = P.WINDOW - 1, T - P.HORIZONS[-1] - 1
        for t0 in sorted(rng.choice(np.arange(lo, hi), size=min(45, hi - lo),
                                    replace=False)):
            w = src.window(stem, int(t0))
            if w is None:
                continue
            b = P.collate([w], dev)
            out = model(b["frames"], None, b["v0"], steps=2)
            fan, gt = out["anchor_traj"][0], b["gt_traj"][0]
            ade = (fan - gt.unsqueeze(0)).norm(dim=-1).mean(dim=-1)
            top = torch.topk(out["sel_score"][0], K).indices
            best_in_top = int(ade[top].argmin())          # 0..K-1 — the label

            # ---- the CONTEXT-DERIVED situation code (scene geometry only) ----
            obs = np.asarray(w["obs"], dtype=np.float32) if w["obs"] else np.zeros((0, 2), np.float32)
            v0 = float(w["v0"])
            n_ag = len(obs)
            if n_ag:
                d = np.linalg.norm(obs, axis=-1)
                near, lat_spread, ahead = float(d.min()), float(obs[:, 1].std()), float((obs[:, 0] > 0).mean())
            else:
                near, lat_spread, ahead = 60.0, 0.0, 0.0
            cand = fan[top]                                # [K,S,2] the shortlist itself
            spread = float((cand - cand.mean(0, keepdim=True)).norm(dim=-1).mean())
            margin = float(out["sel_score"][0][top[0]] - out["sel_score"][0][top[1]])
            rows.append([v0, n_ag, near, lat_spread, ahead, spread, margin,
                         float(w["lead_xy"][0]), float(w["has_lead"])])
            labels.append(best_in_top)
            eps.append(stem)

X = np.array(rows, dtype=np.float64)
y = np.array(labels)
E = np.array(eps)
print(f"windows n={len(y)} · features d={X.shape[1]} · shortlist K={K} · "
      f"episodes {len(set(eps))}")
print(f"label distribution over top-{K}: "
      f"{np.bincount(y, minlength=K) / len(y)}")

# episode-disjoint split — a window-level split would leak the episode
ue = sorted(set(eps))
cut = set(ue[: int(0.6 * len(ue))])
tr = np.array([e in cut for e in E])
Xm, Xs = X[tr].mean(0), X[tr].std(0) + 1e-6
Z = (X - Xm) / Xs


def fit_score(Xtr, ytr, Xte, yte, seed=0):
    """multinomial logistic, plain GD — no sklearn dependency."""
    r = np.random.default_rng(seed)
    W = r.normal(0, 0.01, (Xtr.shape[1] + 1, K))
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Y = np.eye(K)[ytr]
    for _ in range(3000):
        p = np.exp(A @ W - (A @ W).max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        W -= 0.05 * A.T @ (p - Y) / len(A) + 1e-4 * W
    B = np.hstack([Xte, np.ones((len(Xte), 1))])
    return float((np.argmax(B @ W, 1) == yte).mean())


acc = fit_score(Z[tr], y[tr], Z[~tr], y[~tr])
# THE CONTROL SPEC I GOT WRONG THE FIRST TIME. There are TWO chance levels and I
# conflated them, which voided a sound experiment:
#   MAJORITY = always predict the most common rank -- the baseline to BEAT
#   UNIFORM  = 1/K                                 -- where a NOISE fit lands
# A shuffled-label model has no transferable majority signal, so it collapses
# toward UNIFORM, not MAJORITY. Judging it against MAJORITY made a correctly
# behaving control look broken.
majority = float(np.bincount(y[~tr], minlength=K).max() / (~tr).sum())
uniform = 1.0 / K
chance = majority

ysh = y.copy()
np.random.default_rng(7).shuffle(ysh)
acc_sh = fit_score(Z[tr], ysh[tr], Z[~tr], ysh[~tr])
acc_const = fit_score(np.zeros_like(Z[tr]), y[tr], np.zeros_like(Z[~tr]), y[~tr])

print(f"\n  UNIFORM (1/K)                {uniform:.4f}" f"   <- where a noise fit lands")
print(f"  MAJORITY (always rank-1)     {majority:.4f}" f"   <- the baseline to BEAT")
print(f"  CONSTANT code control        {acc_const:.4f}   <- must equal chance")
print(f"  SHUFFLED label control       {acc_sh:.4f}   <- must collapse to chance")
print(f"  CONTEXT-DERIVED situation    {acc:.4f}")
print(f"\n  lift over chance             {acc - chance:+.4f}")

# CONSTANT can only predict one class, so it must reach MAJORITY.
# SHUFFLED must not BEAT majority; landing near uniform is correct.
ok = abs(acc_const - majority) < 0.08 and acc_sh <= majority + 0.08
verdict = ("CONTROLS FAILED — result void" if not ok else
           "SIGNAL — a context-derived code carries information about which "
           "candidate is right; the puzzle-id role HAS a driving analogue"
           if acc - chance > 0.05 else
           "NO SIGNAL — these scene features do not say which candidate is right; "
           "the puzzle-id role has no cheap driving substitute")
print(f"\n  => {verdict}")
json.dump({"n": int(len(y)), "d": int(X.shape[1]), "K": K,
           "majority": majority, "uniform": uniform, "constant_control": acc_const,
           "shuffled_control": acc_sh, "context_code": acc,
           "lift": acc - chance, "controls_ok": ok, "verdict": verdict,
           "_tier": "T0; NON-PARITY pilot; episode-disjoint split",
           "_note": "features are RAW SCENE GEOMETRY, never the situation "
                    "classifier's output (binding PI ruling 2026-08-03)"},
          open(f"{O}/wpb_situation_code.json", "w"), indent=1)
print("-> wpb_situation_code.json")
