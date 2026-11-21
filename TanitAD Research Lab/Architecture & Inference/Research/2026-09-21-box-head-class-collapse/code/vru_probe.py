"""Can the slot features tell a PEDESTRIAN from a CAR? The well-powered form of the class question.

⛔ WHY THIS FILE EXISTS: `cls_nonlinear.py`'s CONTROL REFUSED ITS OWN RESULT, and that refusal is
the reason this one is designed differently rather than being a retry.

    NONLINEAR probe           balanced accuracy  0.09758
    LABEL-SHUFFLED control                       0.15056   <- ABOVE the probe AND above 1/K
    CONSTANT control (1/K)                       0.11111

A label-shuffled arm carries no information by construction, so it MUST read the no-information
value. Reading **above** it proves the STATISTIC is broken, not the representation. The cause is in
the supports: macro-recall weights `stroller` (n=2), `bus` (n=3) and `animal` (n=4) exactly as
heavily as `automobile` (n=1,141), so a single lucky hit on a class of two contributes 0.5 to the
mean. ⇒ `cls_nonlinear.py`'s negative is **INADMISSIBLE** and is reported as INCONCLUSIVE, not as
evidence that the features lack class. ⭐ This is the 2026-08-22 discipline doing its job: the
control that must read a KNOWN value is what stands between a noisy statistic and a published
conclusion about the trunk.

⭐ THE WELL-POWERED QUESTION IS BINARY, AND IT IS THE ONE THAT MATTERS. `cls_collapse.py` measured
188 `person` and 50 `rider` emitted as `automobile` — 238 vulnerable road users called cars. So ask
exactly that:

    VRU   = person + rider                                  (n = 301 scored)
    VEH   = automobile + heavy_truck + bus + trailer        (n = 1,175 scored)

Both sides are amply supported, the constant control's balanced accuracy is **exactly 0.5** — a
KNOWN value, not an estimate — and AUC is reported beside it because AUC is threshold-free and does
not depend on where a collapsed predictor happens to sit.

  probe clears 0.5 -> the VRU/vehicle distinction IS in `box_dec`'s per-slot features and the head
        is not reading it. With `cls` being ONE Linear (`agent_slots.py:368`), the lever is then the
        head's FUNCTION CLASS if only the nonlinear arm clears, or the LOSS if the linear arm does.
  probe fails 0.5  -> the distinction is not in the features the head reads ⇒ the lever is UPSTREAM
        (slot queries, slot memory, trunk), and no head change recovers it.

⛔ CONTROLS: a constant predictor (exactly 0.5 balanced accuracy, 0.5 AUC); a LABEL-SHUFFLED arm
trained identically, which must return to ~0.5 — if it does not, THIS statistic is broken too and
the run is reported INCONCLUSIVE rather than read. Episode-disjoint fit/score; capacity and weight
decay chosen on an inner episode-disjoint split of the FIT half only; n per side printed.
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


def bacc(yt, p, thr=0.5):
    yp = (p > thr).astype(int)
    return float(np.mean([float((yp[yt == k] == k).mean()) for k in (0, 1)]))


def auc(yt, p):
    """Mann-Whitney AUC with TIE-AVERAGED ranks.

    ⛔ THE PLAIN `argsort` FORM IS WRONG AND IT MANUFACTURED A RESULT. Ranking tied scores by
    argsort order assigns them DISTINCT ranks in ARRAY ORDER -- and this array is grouped by
    episode, window and slot index, so the tie-break correlates with the label. MEASURED: a
    LABEL-SHUFFLED control, which can carry no information at all, read **0.654** under that form,
    statistically indistinguishable from the real probe's 0.636. A saturated network (weight_decay
    0.1, 500 epochs) produces exactly the heavy ties that trigger it.
    ⭐ Tie-averaged ranks are the standard fix and make the shuffle control interpretable again.
    """
    p = np.asarray(p, dtype=np.float64)
    o = np.argsort(p, kind="mergesort")
    sp = p[o]
    r = np.empty(len(p), float)
    i = 0
    while i < len(sp):                      # average the ranks inside each tie block
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1, n0 = float((yt == 1).sum()), float((yt == 0).sum())
    return float((r[yt == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def train(Xf, yf, Xs, hidden, wd, linear=False, epochs=500, seed=0, n_seed=3):
    """Returns (scored_probs, TRAIN_AUC). ⛔ THE TRAIN AUC IS A GATE, NOT A DIAGNOSTIC.

    MEASURED here: with a shared seed and strong weight decay the arms did not LEARN at all --
    the label-SHUFFLED arm read AUC 0.647 while the real nonlinear arm read 0.491, same
    architecture, same seed. Two arms differing only in their labels cannot separate that way
    unless both outputs are dominated by the SHARED RANDOM INITIALISATION, i.e. the probe was
    measuring a fixed random projection of the features rather than anything it had fitted.
    ⭐ A probe that cannot fit its OWN TRAINING DATA cannot license any conclusion about the
    features, so train AUC is reported for every arm and a run whose real arm fails it is VOID.
    Seeds are averaged so no single initialisation decides the answer.
    """
    ps, tr = [], []
    for sd_ in range(n_seed):
        p_, t_ = _train1(Xf, yf, Xs, hidden, wd, linear, epochs, seed * 100 + sd_)
        ps.append(p_); tr.append(t_)
    return np.mean(ps, axis=0), float(np.mean(tr))


def _train1(Xf, yf, Xs, hidden, wd, linear=False, epochs=500, seed=0):
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
        z = net(X).squeeze(-1)
        (torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="none")
         * w).mean().backward()
        opt.step()
    with torch.no_grad():
        ps = torch.sigmoid(net(torch.tensor(Xs, dtype=torch.float32)).squeeze(-1)).numpy()
        pf = torch.sigmoid(net(X).squeeze(-1)).numpy()
    return ps, auc(yf, pf)


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
    assert Y[fit].sum() > 20 and Y[sc].sum() > 20, \
        f"ZZABORT too few VRU: fit {Y[fit].sum()} scored {Y[sc].sum()}"

    mu, sd = F[fit].mean(0), F[fit].std(0) + 1e-6
    _, _, Vt = np.linalg.svd((F[fit] - mu) / sd, full_matrices=False)
    n_pc = 40
    P = lambda A: ((A - mu) / sd) @ Vt[:n_pc].T                      # noqa: E731

    best, bb = None, -1.0
    for hidden in (32, 64):
        for wd in (1e-4, 1e-2, 1e-1):
            bb2 = auc(Y[v2], train(P(F[f2]), Y[f2], P(F[v2]), hidden, wd, seed=0)[0])
            if bb2 > bb:
                best, bb = (hidden, wd), bb2
    hidden, wd = best

    p_nl, tr_nl = train(P(F[fit]), Y[fit], P(F[sc]), hidden, wd, seed=1)
    p_li, tr_li = train(P(F[fit]), Y[fit], P(F[sc]), hidden, wd, linear=True, seed=2)
    rng = np.random.default_rng(SEED)
    ysh = Y[fit].copy()
    rng.shuffle(ysh)
    p_sh, tr_sh = train(P(F[fit]), ysh, P(F[sc]), hidden, wd, seed=3)

    es, ys = E[sc], Y[sc]
    keys = sorted(set(es.tolist()))
    idx = {k: np.where(es == k)[0] for k in keys}
    rg = random.Random(SEED)
    draws = [np.concatenate([idx[k] for k in rg.choices(keys, k=len(keys))]) for _ in range(B)]

    def ci_auc(p):
        ms = sorted(auc(ys[i], p[i]) for i in draws if len(set(ys[i].tolist())) > 1)
        return [round(ms[int(0.025 * len(ms))], 5), round(ms[int(0.975 * len(ms))], 5)] if ms else None

    res = {"_what": "can box_dec's per-slot features separate VULNERABLE ROAD USERS from vehicles?",
           "_evidence_class": "MEASURED (ours), CPU, cached features, A8 ckpt_5000",
           "_supersedes": ("cls_nonlinear.py, whose LABEL-SHUFFLED control read 0.15056 against a "
                           "1/K of 0.11111 — above the no-information value, which refutes the "
                           "STATISTIC (macro-recall over classes of n=2,3,4), not the features"),
           "n_fit": int(fit.sum()), "n_scored": int(sc.sum()),
           "VRU_fit": int(Y[fit].sum()), "VRU_scored": int(ys.sum()),
           "VEH_fit": int((Y[fit] == 0).sum()), "VEH_scored": int((ys == 0).sum()),
           "d_pca": n_pc, "hidden": hidden, "weight_decay": wd, "seeds_averaged": 3,
           "TRAIN_AUC_gate": {"NONLINEAR": round(tr_nl, 5), "LINEAR": round(tr_li, 5),
                              "LABEL_SHUFFLED_control": round(tr_sh, 5),
                              "rule": "the real arms must FIT their own training data (>0.6); "
                                      "the shuffled arm SHOULD fit it (memorisation) yet must NOT "
                                      "generalise"},
           "AUC": {"NONLINEAR": round(auc(ys, p_nl), 5),
                   "LINEAR": round(auc(ys, p_li), 5),
                   "LABEL_SHUFFLED_control": round(auc(ys, p_sh), 5),
                   "CONSTANT_control": 0.5},
           "AUC_CI95": {"NONLINEAR": ci_auc(p_nl), "LINEAR": ci_auc(p_li),
                        "LABEL_SHUFFLED_control": ci_auc(p_sh)},
           "balanced_accuracy": {"NONLINEAR": round(bacc(ys, p_nl), 5),
                                 "LINEAR": round(bacc(ys, p_li), 5),
                                 "LABEL_SHUFFLED_control": round(bacc(ys, p_sh), 5),
                                 "CONSTANT_control": 0.5,
                                 "THE_HEAD_ITSELF": 0.5},
           "_head_note": ("the head emits `automobile` for every slot, so its VRU recall is 0.0 "
                          "and its balanced accuracy is exactly 0.5 — the constant value")}

    sh = res["AUC_CI95"]["LABEL_SHUFFLED_control"]
    shuffle_ok = bool(sh and sh[0] <= 0.5 <= sh[1])
    if max(tr_nl, tr_li) < 0.6:
        res["_VERDICT"] = (
            "⛔ VOID — NEITHER REAL ARM CAN FIT ITS OWN TRAINING DATA "
            f"(train AUC nonlinear {tr_nl:.4f}, linear {tr_li:.4f}). A probe that has not fitted "
            "anything licenses no conclusion about the features, in either direction. The "
            "instrument, not the representation, is what this run measured.")
        print(json.dumps(res, indent=1, ensure_ascii=False))
        print(res["_VERDICT"])
        pathlib.Path("vru_probe.json").write_text(
            json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        return 0
    nl, li = res["AUC_CI95"]["NONLINEAR"], res["AUC_CI95"]["LINEAR"]
    res["shuffle_control_returns_to_chance"] = shuffle_ok
    if not shuffle_ok:
        res["_VERDICT"] = ("⚠️ INCONCLUSIVE — the LABEL-SHUFFLED control does not return to 0.5, so "
                           "THIS statistic is broken too and no reading is admissible from it.")
    elif li and li[0] > 0.5:
        res["_VERDICT"] = (
            "⭐⭐ THE SIGNAL IS THERE AND EVEN LINEARLY READABLE — the features separate VRUs from "
            "vehicles above chance with a LINEAR readout, which is exactly the function class the "
            "head's `cls` already is (agent_slots.py:368). ⇒ the collapse is an OBJECTIVE failure "
            "after all, and re-weighting IS the lever.")
    elif nl and nl[0] > 0.5:
        res["_VERDICT"] = (
            "⭐⭐ THE SIGNAL IS THERE BUT NOT LINEARLY — a nonlinear readout separates VRUs from "
            "vehicles above chance while a linear one does not. ⇒ the lever is the HEAD'S FUNCTION "
            "CLASS: `cls` is ONE Linear (agent_slots.py:368) and cannot express the boundary, so "
            "no amount of loss re-weighting will recover it.")
    else:
        res["_VERDICT"] = (
            "⛔ NOT EVEN VRU vs VEHICLE — with ample support on both sides and a shuffle control "
            "that behaves, neither readout separates a pedestrian from a car in the features the "
            "head reads. ⇒ the lever is UPSTREAM (slot queries, slot memory, trunk); no head change "
            "of any shape recovers class.")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n" + res["_VERDICT"])
    pathlib.Path("vru_probe.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
