"""WP-F — do RELATIONAL edges carry anything the candidate's own geometry does not?

⛔ THIS ALSO REPAIRS WP-B. WP-B fed the classifier NINE WINDOW-LEVEL SCALARS and
asked which of K candidates is best. The feature vector was IDENTICAL for every
candidate, so the model could only learn the marginal P(best=k) — i.e. "always
predict rank-1". Its "no signal" was a TAUTOLOGY, not a measurement: the probe was
structurally incapable of discriminating. Here every feature is PER CANDIDATE.

⭐ THE QUESTION: the choice among the top-5 — is it visible in each candidate's OWN
geometry, or does it need the RELATION between a candidate and the agents?

  NODE      per-candidate geometry only     (endpoint, length, curvature, lateral)
  NODE+EDGE + candidate x agent relations   (clearance, TTC, closing, side, count)

  If NODE+EDGE > NODE, relations carry decision-relevant information and the
  graph-structured latent (v6) earns its place. If not, the graph is decoration
  and v6 should revert to a flat latent.

⚠️ CONTROLS, each of which must read a known value or the result is void:
  * MAJORITY      always predict rank-1        — the baseline both arms must beat
  * SHUFFLED-EDGE edges permuted across windows — must collapse to ~NODE, or the
                  "edge gain" is fit noise rather than relational information
  * CONSTANT-EDGE edges replaced by a constant  — must equal NODE exactly
  * episode-disjoint split; n and d printed

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
K, R_EGO = 5, 2.0

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
NODE, EDGE, LAB, EPS = [], [], [], []

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
            cand = fan[top].cpu().numpy()                      # [K, S, 2]
            LAB.append(int(ade[top].argmin()))
            sc = out["sel_score"][0][top].cpu().numpy()          # the scorer's signal
            EPS.append(stem)

            obs = (np.asarray(w["obs"], dtype=np.float32) if w["obs"]
                   else np.zeros((0, 2), np.float32))
            v0 = float(w["v0"])
            nrows, erows = [], []
            for k in range(K):
                c = cand[k]                                     # [S,2]
                step = np.diff(c, axis=0)
                length = float(np.linalg.norm(step, axis=-1).sum())
                head = np.arctan2(step[:, 1], step[:, 0])
                curv = float(np.abs(np.diff(head)).sum()) if len(head) > 1 else 0.0
                # ---- NODE: this candidate's OWN geometry, no agents involved ----
                # RANK/SCORE FIRST: without it a per-candidate arm cannot even
                # match "always pick rank-1", so it would lose to the baseline on
                # information it was never given. That is what broke run 1.
                nrows.append([float(sc[k]), float(sc[k] - sc[0]), float(k),
                              c[-1, 0], c[-1, 1], length, curv,
                              float(np.abs(c[:, 1]).max()), v0])
                # ---- EDGE: candidate x agent RELATIONS ----
                if len(obs):
                    d = np.linalg.norm(c[:, None, :] - obs[None, :, :], axis=-1)
                    clr = np.clip(d - R_EGO, 0, None)
                    per_ag = clr.min(0)                          # [A]
                    j = int(per_ag.argmin())
                    ttc = float(per_ag[j] / max(v0, 0.5))
                    side = float(np.sign(c[-1, 1] - obs[j, 1]))
                    approach = float(clr[0, j] - clr[-1, j])     # closing over the path
                    erows.append([float(per_ag.min()), ttc, side, approach,
                                  float((per_ag < 3.0).sum()),
                                  float(np.median(per_ag))])
                else:
                    erows.append([60.0, 60.0, 0.0, 0.0, 0.0, 60.0])
            NODE.append(nrows)
            EDGE.append(erows)

NODE = np.array(NODE, dtype=np.float64)          # [N, K, dn]
EDGE = np.array(EDGE, dtype=np.float64)          # [N, K, de]
y = np.array(LAB)
E = np.array(EPS)
N, _, dn = NODE.shape
de = EDGE.shape[2]
print(f"windows N={N} · K={K} · node d={dn} · edge d={de} · episodes {len(set(EPS))}")
print(f"label distribution: {np.bincount(y, minlength=K) / N}")

ue = sorted(set(EPS))
cut = set(ue[: int(0.6 * len(ue))])
tr = np.array([e in cut for e in E])


def rank_acc(F, seed=0, iters=4000):
    """Per-candidate linear scorer -> argmax. F: [N, K, d]."""
    mu = F[tr].reshape(-1, F.shape[2]).mean(0)
    sd = F[tr].reshape(-1, F.shape[2]).std(0) + 1e-6
    Z = (F - mu) / sd
    r = np.random.default_rng(seed)
    w = r.normal(0, 0.01, Z.shape[2] + 1)
    A = np.concatenate([Z, np.ones((*Z.shape[:2], 1))], -1)      # [N,K,d+1]
    Ytr, Atr = y[tr], A[tr]
    for _ in range(iters):
        s = Atr @ w
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        g = np.einsum("nk,nkd->d", p - np.eye(K)[Ytr], Atr) / len(Atr)
        w -= 0.1 * g + 1e-4 * w
    return float((np.argmax(A[~tr] @ w, 1) == y[~tr]).mean())


majority = float(np.bincount(y[~tr], minlength=K).max() / (~tr).sum())
RANK = NODE[..., :3]                       # the scorer's signal alone
acc_rank = rank_acc(RANK)                  # must be >= majority, else the fit is broken
acc_node = rank_acc(NODE)                  # + the candidate's own geometry
acc_both = rank_acc(np.concatenate([NODE, EDGE], -1))   # + relations

sh = EDGE.copy()
np.random.default_rng(7).shuffle(sh)                              # permute across windows
acc_sh = rank_acc(np.concatenate([NODE, sh], -1))
acc_const = rank_acc(np.concatenate([NODE, np.zeros_like(EDGE)], -1))

print(f"\n  MAJORITY (always rank-1)     {majority:.4f}   <- both arms must beat this")
print(f"  NODE only  (own geometry)    {acc_node:.4f}")
print(f"  + EDGE (relations)           {acc_both:.4f}")
print(f"  CONSTANT-EDGE control        {acc_const:.4f}   <- must equal NODE")
print(f"  SHUFFLED-EDGE control        {acc_sh:.4f}   <- must collapse to NODE")
print(f"\n  edge gain  = NODE+EDGE - NODE     {acc_both - acc_node:+.4f}")
print(f"  node gain  = NODE - MAJORITY      {acc_node - majority:+.4f}")

# ⛔ the sanity gate that run 1 lacked: the rank-only arm MUST reach majority,
# or the arms are being asked a harder question than the baseline.
ok = (abs(acc_const - acc_node) < 0.05 and acc_rank >= majority - 0.05
      and acc_both - acc_sh > 0.01)
verdict = ("CONTROLS FAILED — void" if not ok else
           "EDGES CARRY DECISION-RELEVANT INFORMATION — the relational latent earns its place"
           if acc_both - acc_node > 0.03 else
           "EDGES ADD NOTHING here — the graph is decoration on these features; "
           "v6 should revert to a flat latent unless a learned edge does better")
print(f"\n  => {verdict}")
json.dump({"N": int(N), "K": K, "d_node": int(dn), "d_edge": int(de),
           "episodes": len(set(EPS)), "majority": majority,
           "rank_only": acc_rank, "node": acc_node, "node_edge": acc_both,
           "constant_edge_control": acc_const, "shuffled_edge_control": acc_sh,
           "edge_gain": acc_both - acc_node, "node_gain": acc_node - majority,
           "controls_ok": ok, "verdict": verdict,
           "_tier": "T0; NON-PARITY pilot; episode-disjoint",
           "_repairs": "WP-B used WINDOW-LEVEL features, identical across "
                       "candidates, so it could only learn the marginal. Every "
                       "feature here is PER-CANDIDATE."},
          open(f"{O}/wpf_edges.json", "w"), indent=1)
print("-> wpf_edges.json")
