"""Is the class signal there NONLINEARLY? The lever after re-weighting was ruled out.

THE CHAIN. `cls_collapse.py`: the head emits ONE class of ten on 2,000/2,000 slots.
`cls_probe.py`, properly powered on the CORRECT head (n_fit 1,240, d 41 after a FIT-only PCA):

    constant control (1/K)                     balanced accuracy  0.11111
    UNBALANCED linear probe (1 class emitted)                     0.11111
    BALANCED  linear probe (9 classes emitted)                    0.07087   <- BELOW the control

⛔ AND THE STRUCTURAL POINT THAT MAKES THAT DECISIVE RATHER THAN MERELY DISAPPOINTING: the head's
`cls` output IS a linear map of exactly these features — one `nn.Linear` sliced by `SLOT_SLICES`
(`agent_slots.py:368`). So the head's class head *is* a linear probe, and it **cannot beat the best
linear probe by construction**. The best linear solution available on these features is the
CONSTANT, and the head has already found it. ⇒ **re-weighting the `cls` loss cannot fix this** — it
would push the head off the constant toward the balanced probe's behaviour, which scores WORSE on
balanced accuracy (0.071) and catastrophically worse raw (0.054). `PREREG_BOXCLS.md`'s launch gate
therefore reads DO NOT LAUNCH, and it was written before this number existed.

⭐ THE NEXT LEVER IS THE FUNCTION CLASS, NOT THE OBJECTIVE — and it is free, because the features
are cached. CLAUDE.md is explicit that a linear negative is NOT a negative about learnability:
the implication runs one way only. So the question splits cleanly, and the two answers point at
different parts of the model:

  nonlinear probe CLEARS the control -> the signal IS in the per-slot features and a LINEAR head
        cannot read it ⇒ the lever is the HEAD'S FUNCTION CLASS (give `cls` a nonlinearity). Cheap,
        local, and testable at tiny scale.
  nonlinear probe FAILS too          -> the features do not carry class at all ⇒ the lever is
        UPSTREAM (queries, slot memory, trunk), and no head change of any shape will help.

⛔ CONTROLS, and the shuffle one is load-bearing here. An MLP with class-balanced weights on 9
classes can manufacture structure; the discriminator is a LABEL-SHUFFLED arm, trained identically
on permuted fit labels, which MUST read ~1/K. Structure surviving a shuffle is leakage, not signal
— the rule that caught four estimator bugs on 2026-08-22.
⛔ Balanced accuracy is the statistic, never raw: a collapsed predictor already scores 0.765 raw.
The constant control's balanced accuracy is EXACTLY 1/K, a KNOWN value.
⛔ Episode-disjoint fit/score; hidden width and weight decay chosen on an inner episode-disjoint
split of the FIT half only; n, d and per-class support printed.
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


def balanced_acc(yt, yp):
    ks = sorted(set(yt.tolist()))
    return float(np.mean([float((yp[yt == k] == k).mean()) for k in ks])), len(ks)


def train_mlp(Xf, yf, Xs, K, hidden, wd, epochs=400, seed=0):
    torch.manual_seed(seed)
    cnt = collections.Counter(yf.tolist())
    w = torch.tensor([1.0 / max(cnt.get(k, 0), 1) for k in range(K)], dtype=torch.float32)
    w = w / w.sum() * K
    net = torch.nn.Sequential(torch.nn.Linear(Xf.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, K))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=wd)
    X = torch.tensor(Xf, dtype=torch.float32)
    Y = torch.tensor(yf, dtype=torch.long)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    for _ in range(epochs):
        opt.zero_grad()
        lossf(net(X), Y).backward()
        opt.step()
    with torch.no_grad():
        return net(torch.tensor(Xs, dtype=torch.float32)).argmax(-1).numpy()


def main() -> int:
    assert CACHE.exists(), "ZZABORT run cls_probe.py first to build cls_rows.npz"
    z = np.load(CACHE)
    F, Y, E = z["F"].astype(np.float64), z["Y"], z["E"]
    K = len(AGENT_CLASSES)

    eps_ = sorted(set(E.tolist()))
    half = set(eps_[: len(eps_) // 2])
    fit = np.array([e in half for e in E])
    sc = ~fit
    fe = sorted(set(E[fit].tolist()))
    inner = set(fe[: len(fe) // 2])
    f2 = np.array([e in inner for e in E]) & fit
    v2 = fit & ~f2

    mu = F[fit].mean(0)
    sd = F[fit].std(0) + 1e-6
    _, _, Vt = np.linalg.svd((F[fit] - mu) / sd, full_matrices=False)
    n_pc = 40
    P = lambda A: ((A - mu) / sd) @ Vt[:n_pc].T                      # noqa: E731

    best, bb = None, -1.0
    for hidden in (32, 64, 128):
        for wd in (1e-4, 1e-2, 1e-1):
            yp = train_mlp(P(F[f2]), Y[f2], P(F[v2]), K, hidden, wd, seed=0)
            ba, _ = balanced_acc(Y[v2], yp)
            if ba > bb:
                best, bb = (hidden, wd), ba
    hidden, wd = best

    yp = train_mlp(P(F[fit]), Y[fit], P(F[sc]), K, hidden, wd, seed=1)
    ba, n_cls = balanced_acc(Y[sc], yp)
    ctrl = 1.0 / n_cls

    rng = np.random.default_rng(SEED)
    ysh = Y[fit].copy()
    rng.shuffle(ysh)
    yp_sh = train_mlp(P(F[fit]), ysh, P(F[sc]), K, hidden, wd, seed=1)
    ba_sh, _ = balanced_acc(Y[sc], yp_sh)

    # episode-clustered interval on (probe - shuffled), the honest comparison
    es = E[sc]
    keys = sorted(set(es.tolist()))
    idx = {k: np.where(es == k)[0] for k in keys}
    rg = random.Random(SEED)
    ms = sorted(
        (lambda ii: balanced_acc(Y[sc][ii], yp[ii])[0] - balanced_acc(Y[sc][ii], yp_sh[ii])[0])(
            np.concatenate([idx[k] for k in rg.choices(keys, k=len(keys))]))
        for _ in range(B))
    ci = [round(ms[int(0.025 * B)], 5), round(ms[int(0.975 * B)], 5)]

    res = {"_what": "is the class signal present NONLINEARLY in box_dec's per-slot features?",
           "_evidence_class": "MEASURED (ours), CPU, cached features, A8 ckpt_5000",
           "_why_it_matters": ("the head's `cls` is a LINEAR map of these features "
                               "(agent_slots.py:368), so a linear negative cannot be fixed by "
                               "re-weighting; only a wider function class or better features can"),
           "n_pairs": int(F.shape[0]), "d_pca": n_pc,
           "n_fit": int(fit.sum()), "n_scored": int(sc.sum()),
           "n_classes_scored": n_cls, "hidden": hidden, "weight_decay": wd,
           "balanced_accuracy": {
               "NONLINEAR_probe": round(ba, 5),
               "LABEL_SHUFFLED_control": round(ba_sh, 5),
               "CONSTANT_control_1_over_K": round(ctrl, 5),
               "LINEAR_balanced_probe_from_cls_probe": 0.07087,
               "THE_HEAD_ITSELF": round(ctrl, 5)},
           "probe_minus_shuffled_CI95": ci,
           "per_class_recall": {AGENT_CLASSES[k]: round(float((yp[Y[sc] == k] == k).mean()), 4)
                                for k in sorted(set(Y[sc].tolist()))},
           "support_scored": {AGENT_CLASSES[k]: int((Y[sc] == k).sum())
                              for k in sorted(set(Y[sc].tolist()))}}
    beats = bool(ci[0] > 0 and ba > ctrl)
    res["_VERDICT"] = (
        "⭐⭐ THE SIGNAL IS THERE AND A LINEAR HEAD CANNOT READ IT — a nonlinear probe on the SAME "
        "features clears both the constant control and a label-shuffled control. ⇒ the lever is the "
        "HEAD'S FUNCTION CLASS (`cls` is one Linear, agent_slots.py:368), NOT the loss weighting."
        if beats else
        "⛔ THE FEATURES DO NOT CARRY CLASS — a nonlinear probe with class-balanced weights fails "
        "the constant control too, on features the head reads directly. ⇒ NO head change of any "
        "shape recovers class; the lever is UPSTREAM (slot queries, slot memory, or the trunk). "
        "⚠️ Bounded by this probe's own capacity and n, both printed.")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n" + res["_VERDICT"])
    pathlib.Path("cls_nonlinear.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
