"""Can the features tell a pedestrian from a car IN THE SAME SCENE? The confound-free form.

⛔ THE THREE FAILED FORMS BEFORE THIS ONE, EACH KILLED BY ITS OWN CONTROL. None of them is a
retry of the last; each changed because a control refused the result, and the chain is recorded
because the traps are reusable.

  1. `cls_nonlinear.py` — macro-recall over 9 classes. LABEL-SHUFFLED control read **0.15056**
     against a 1/K of 0.11111, i.e. ABOVE the no-information value. Cause: classes of n=2,3,4
     (`stroller`, `bus`, `animal`) carry the same weight as one of n=1,141, so a single lucky hit
     contributes 0.5 to the mean. The STATISTIC was broken, not the features.
  2. `vru_probe.py` v1 — binary VRU-vs-vehicle, row-level AUC. Shuffled control **0.654**, tied
     with the real probe. First suspected cause (tie-ranking in my own `auc`) was FIXED and moved
     it only to 0.647, so that diagnosis was WRONG and is recorded as wrong.
  3. `vru_probe.py` v2 — per-arm seeds, 3-seed averaging, and a TRAIN-AUC gate. The gate passes
     (nonlinear 1.000, linear 0.784, shuffled 1.000 = memorisation), so the models really do fit.
     The shuffled control STILL generalises at **0.699 AUC on DISJOINT EPISODES — beating both
     real arms**.

⭐ A PERMUTED-LABEL MODEL CANNOT CARRY LABEL INFORMATION, SO 0.699 LOCATES THE CONFOUND EXACTLY:
it is not in the labels, it is in the EPISODES. VRU density is clustered by scene — an urban clip
is full of pedestrians, a highway clip has none — so ROW-LEVEL AUC pools across episodes and
rewards any model whose output tracks scene appearance, whatever it was trained on. Memorising the
fit set produces exactly such an output. ⇒ row-level AUC cannot answer this question at all.

⭐ THE CONFOUND IS REMOVED BY CONDITIONING ON IT, NOT BY CORRECTING FOR IT. Compute AUC **WITHIN
EACH SCORED EPISODE** (only those containing both a VRU and a vehicle), then average across
episodes. A model that merely ranks scenes has NO within-scene ordering and scores 0.5 by
construction; only genuine per-object discrimination survives. This is the same move as scoring
paired arms on identical windows rather than comparing pooled means.

⛔ The LABEL-SHUFFLED control must now return to 0.5. If it does not, this statistic is broken too
and the run is reported INCONCLUSIVE — it is not permitted to be read either way.
⛔ CPU only, no forward pass — reads `cls_rows.npz`.
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
from tanitad.models.agent_slots import AGENT_CLASSES     # noqa: E402

CACHE = pathlib.Path("cls_rows.npz")
SEED = 20260921
B = 2000
VRU = {AGENT_CLASSES.index("person"), AGENT_CLASSES.index("rider")}
VEH = {AGENT_CLASSES.index(c) for c in ("automobile", "heavy_truck", "bus", "trailer")}


def auc(yt, p):
    """Mann-Whitney AUC with TIE-AVERAGED ranks (a plain argsort ranks ties by array order)."""
    p = np.asarray(p, float)
    o = np.argsort(p, kind="mergesort")
    sp, r, i = p[o], np.empty(len(p), float), 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1, n0 = float((yt == 1).sum()), float((yt == 0).sum())
    return float((r[yt == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def _train1(Xf, yf, Xs, hidden, wd, linear, epochs, seed):
    torch.manual_seed(seed)
    cnt = collections.Counter(yf.tolist())
    w = torch.tensor([1.0 / max(cnt.get(int(y), 1), 1) for y in yf], dtype=torch.float32)
    w = w / w.mean()
    net = (torch.nn.Linear(Xf.shape[1], 1) if linear else
           torch.nn.Sequential(torch.nn.Linear(Xf.shape[1], hidden), torch.nn.GELU(),
                               torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                               torch.nn.Linear(hidden, 1)))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=wd)
    X = torch.tensor(Xf, dtype=torch.float32)
    Y = torch.tensor(yf, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        (torch.nn.functional.binary_cross_entropy_with_logits(
            net(X).squeeze(-1), Y, reduction="none") * w).mean().backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xs, dtype=torch.float32)).squeeze(-1)).numpy()


def train(Xf, yf, Xs, hidden, wd, linear=False, epochs=500, seed=0, n_seed=3):
    return np.mean([_train1(Xf, yf, Xs, hidden, wd, linear, epochs, seed * 100 + s)
                    for s in range(n_seed)], axis=0)


def within(ys, es, p, keys=None):
    """Mean AUC computed INSIDE each episode that carries both classes."""
    out = {}
    for k in (keys if keys is not None else sorted(set(es.tolist()))):
        m = es == k
        if m.sum() < 2 or len(set(ys[m].tolist())) < 2:
            continue
        out[k] = auc(ys[m], p[m])
    return out


def main() -> int:
    assert CACHE.exists(), "ZZABORT run cls_probe.py first to build cls_rows.npz"
    z = np.load(CACHE)
    F, Yc, E = z["F"].astype(np.float64), z["Y"], z["E"]
    keep = np.array([int(y) in VRU or int(y) in VEH for y in Yc])
    F, E = F[keep], E[keep]
    Y = np.array([1 if int(y) in VRU else 0 for y in Yc[keep]])

    eps_ = sorted(set(E.tolist()))
    half = set(eps_[: len(eps_) // 2])
    fit = np.array([e in half for e in E])
    sc = ~fit
    fe = sorted(set(E[fit].tolist()))
    inner = set(fe[: len(fe) // 2])
    f2 = np.array([e in inner for e in E]) & fit
    v2 = fit & ~f2

    mu, sd = F[fit].mean(0), F[fit].std(0) + 1e-6
    _, _, Vt = np.linalg.svd((F[fit] - mu) / sd, full_matrices=False)
    n_pc = 40
    P = lambda A: ((A - mu) / sd) @ Vt[:n_pc].T                      # noqa: E731

    # capacity chosen on the inner split, USING THE WITHIN-EPISODE statistic
    best, bb = None, -1.0
    for hidden in (32, 64):
        for wd in (1e-4, 1e-2, 1e-1):
            d = within(Y[v2], E[v2], train(P(F[f2]), Y[f2], P(F[v2]), hidden, wd, seed=0))
            v = float(np.mean(list(d.values()))) if d else -1.0
            if v > bb:
                best, bb = (hidden, wd), v
    hidden, wd = best

    p_nl = train(P(F[fit]), Y[fit], P(F[sc]), hidden, wd, seed=1)
    p_li = train(P(F[fit]), Y[fit], P(F[sc]), hidden, wd, linear=True, seed=2)
    rng = np.random.default_rng(SEED)
    ysh = Y[fit].copy()
    rng.shuffle(ysh)
    p_sh = train(P(F[fit]), ysh, P(F[sc]), hidden, wd, seed=3)

    ys, es = Y[sc], E[sc]
    d_nl, d_li, d_sh = (within(ys, es, p) for p in (p_nl, p_li, p_sh))
    keys = sorted(set(d_nl) & set(d_li) & set(d_sh))
    assert len(keys) >= 4, f"ZZABORT only {len(keys)} episodes carry both classes"

    rg = random.Random(SEED)
    def ci(d):
        ms = sorted(float(np.mean([d[k] for k in rg.choices(keys, k=len(keys))]))
                    for _ in range(B))
        return [round(ms[int(0.025 * B)], 5), round(ms[int(0.975 * B)], 5)]

    m = lambda d: round(float(np.mean([d[k] for k in keys])), 5)            # noqa: E731
    res = {"_what": "can box_dec's per-slot features separate a VRU from a vehicle IN THE SAME SCENE?",
           "_evidence_class": "MEASURED (ours), CPU, cached features, A8 ckpt_5000",
           "_why_within_episode": ("row-level AUC is confounded by episode composition: a "
                                   "LABEL-SHUFFLED control scored 0.699 pooled, beating both real "
                                   "arms, because VRU density is clustered by scene"),
           "n_scored_rows": int(sc.sum()), "VRU_scored": int(ys.sum()),
           "episodes_with_both_classes": len(keys),
           "d_pca": n_pc, "hidden": hidden, "weight_decay": wd, "seeds_averaged": 3,
           "within_episode_AUC": {
               "NONLINEAR": m(d_nl), "LINEAR": m(d_li),
               "LABEL_SHUFFLED_control": m(d_sh), "CONSTANT_control": 0.5,
               "THE_HEAD_ITSELF": 0.5},
           "within_episode_AUC_CI95": {
               "NONLINEAR": ci(d_nl), "LINEAR": ci(d_li),
               "LABEL_SHUFFLED_control": ci(d_sh)},
           "_head_note": ("the head emits `automobile` for EVERY slot, so within any scene it has "
                          "no ordering at all: AUC exactly 0.5, the constant value")}

    sh = res["within_episode_AUC_CI95"]["LABEL_SHUFFLED_control"]
    ok = bool(sh[0] <= 0.5 <= sh[1])
    nl = res["within_episode_AUC_CI95"]["NONLINEAR"]
    li = res["within_episode_AUC_CI95"]["LINEAR"]
    res["shuffle_control_returns_to_chance"] = ok
    if not ok:
        res["_VERDICT"] = (
            "⚠️ INCONCLUSIVE — even within episodes the LABEL-SHUFFLED control does not return to "
            "0.5. The question is left OPEN with the instrument defect named; it is NOT read as "
            "evidence in either direction.")
    elif li[0] > 0.5:
        res["_VERDICT"] = (
            "⭐⭐ THE SIGNAL IS THERE AND LINEARLY READABLE — within a single scene the features "
            "separate a VRU from a vehicle above chance with a LINEAR readout, which is exactly "
            "the function class the head's `cls` already is (agent_slots.py:368). ⇒ the collapse "
            "is an OBJECTIVE failure and re-weighting IS the lever.")
    elif nl[0] > 0.5:
        res["_VERDICT"] = (
            "⭐⭐ THE SIGNAL IS THERE BUT NOT LINEARLY — a nonlinear readout separates VRU from "
            "vehicle within a scene while a linear one does not. ⇒ the lever is the HEAD'S "
            "FUNCTION CLASS: `cls` is ONE Linear and cannot express the boundary, so loss "
            "re-weighting alone will not recover it.")
    else:
        res["_VERDICT"] = (
            "⛔ NOT EVEN VRU vs VEHICLE, WITHIN A SCENE — with a shuffle control that behaves and "
            "both readouts at chance, the features the head reads do not distinguish a pedestrian "
            "from a car. ⇒ the lever is UPSTREAM (slot queries, slot memory, trunk); NO head "
            "change of any shape recovers class. ⚠️ Bounded by this probe's capacity and n, printed.")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(res["_VERDICT"])
    pathlib.Path("vru_within.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
