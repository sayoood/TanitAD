"""Is the CLASS signal in the features and unused, or absent? LOSS DEFECT vs REPRESENTATION DEFECT.

`cls_collapse.py` measured a TOTAL collapse: 1,482/1,482 matched pairs and 2,000/2,000 slots emitted
`automobile`, top-1 accuracy exactly the majority baseline, 188 GT `person` and 50 `rider` predicted
as cars, on an UNWEIGHTED cross-entropy (`agent_slots.py:591`) over a 577.5:1 imbalance.

⛔ THAT NAMES A SUSPECT, IT DOES NOT CONVICT ONE. Two very different defects produce the identical
output, and they demand opposite fixes:

  LOSS / OPTIMISATION defect  -> the per-slot features DO separate the classes and the output layer
                                 has collapsed onto the majority anyway. Fix: a class-weighted or
                                 focal `cls` term — exactly what `NO_OBJECT_W = 0.1` already does
                                 for presence, for the reason stated at `agent_slots.py:230-231`.
  REPRESENTATION defect       -> the features do not separate the classes at all, the output layer
                                 is doing the best available thing, and a loss re-weighting would
                                 only trade majority accuracy for noise. Fix is upstream: queries,
                                 capacity, or the trunk.

⭐ ONE LINEAR PROBE SEPARATES THEM, AND IT COSTS NO TRAINING. Hook the per-slot feature vector that
feeds the slot head (the Linear whose `out_features` is the total slot width — `agent_slots.py:368`,
the same hook `presence_alignment.py` used), and fit a LINEAR classifier from it to the GT class on
episode-disjoint splits. The probe sees exactly what the head's own output layer sees.

  probe >> majority -> the signal IS there and the head is not using it ⇒ LOSS DEFECT
  probe ~= majority -> the features do not carry class ⇒ REPRESENTATION DEFECT

⛔ CONTROLS, and accuracy alone is NOT admissible here. A constant predictor already scores 0.779
because the target is 78 % `automobile`, so raw accuracy hides the entire question. The reported
statistic is BALANCED ACCURACY (mean per-class recall), whose no-information value for a constant
predictor is exactly `1/K = 0.1000` — a control that must read a KNOWN value, per the 2026-08-22
discipline. Raw accuracy is reported beside it, never instead of it.
⛔ Ridge penalty chosen on an inner episode-disjoint split of the FIT half only; n, d and the
per-class support printed; classes with no fit-side support are named rather than silently dropped.
⚠️ Function class: LINEAR. A probe that FAILS bounds linear separability only — but a probe that
SUCCEEDS is conclusive, and success is the outcome that indicts the loss.
⛔ CPU only, forward hook, read-only, no GPU.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))
import s1_pass as SP                                      # noqa: E402
from tanitad.models.agent_slots import AGENT_CLASSES, match_slots     # noqa: E402

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
N_WIN = 120
K = len(AGENT_CLASSES)


def fit_ridge(X, Y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def balanced_acc(yt, yp):
    rec = []
    for k in sorted(set(yt.tolist())):
        m = yt == k
        rec.append(float((yp[m] == k).mean()))
    return float(np.mean(rec)), len(rec)


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")

    root = getattr(mp, "model", None) or getattr(mp, "net", None)
    if root is None:
        for a in dir(mp):
            o = getattr(mp, a, None)
            if isinstance(o, torch.nn.Module):
                root = o
                break
    assert root is not None, "ZZABORT could not find the model on ModelPass"
    # ⛔ SELECT BY THE OUTPUT BEING SCORED, NOT BY A WIDTH TWO MODULES SHARE A PREFIX OF.
    # `align_fix.py` measured what the width selector costs: it binds to `core.agent_head.head`
    # (the 2-D head, 16 queries) while `s1_pass` scores `perception.box_dec` (100 slots).
    cand = [(n, m) for n, m in root.named_modules()
            if isinstance(m, torch.nn.Linear) and "box_dec" in n and n.endswith("head")]
    assert cand, "ZZABORT no box_dec head found"
    target = cand[0]
    print(f"hooking {target[0]}  in={target[1].in_features} out={target[1].out_features}")

    if pathlib.Path("cls_rows.npz").exists():
        z = np.load("cls_rows.npz")
        return probe(z["F"], z["Y"], z["E"], target[0])

    feats = []
    h = target[1].register_forward_hook(lambda mod, inp, out: feats.append(inp[0].detach()))

    wis = [w for w in SP.trainer_windows(corp.ds, 1500)
           if corp.eligibility(w) is None][:N_WIN]
    F, Y, E = [], [], []
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        feats.clear()
        out = mp.forward(item, str(corp.clip_ids[e_i]))
        if not feats:
            continue
        f = feats[-1].float().reshape(-1, feats[-1].shape[-1])
        sl = {k: v[0].float().cpu() for k, v in out["perception"]["box_slots"].items()
              if torch.is_tensor(v) and v.dim() >= 1}
        # the guard a width match could not give: hooked slots == scored slots
        assert f.shape[0] == int(sl["box"].shape[-2]),             f"ZZABORT hooked {f.shape[0]} slots but box_slots carries {sl['box'].shape[-2]}"
        m = match_slots({k: v[None] for k, v in sl.items()},
                        {"box": item["agent_box"][None].float(),
                         "valid": item["agent_valid"][None], "cls": item["agent_cls"][None]})
        r, c = m["rows"][0].tolist(), m["cols"][0].tolist()
        tc = item["agent_cls"].numpy()
        for i, j in zip(r, c):
            if i < f.shape[0] and tc[j] >= 0:
                F.append(f[i].numpy())
                Y.append(int(tc[j]))
                E.append(int(e_i))
    h.remove()
    F, Y, E = np.asarray(F, dtype=np.float64), np.asarray(Y), np.asarray(E)
    np.savez_compressed("cls_rows.npz", F=F, Y=Y, E=E)
    assert F.shape[0] > 100, f"ZZABORT too few rows: {F.shape[0]}"
    return probe(F, Y, E, target[0])


def probe(F, Y, E, hooked):
    F = np.hstack([F, np.ones((F.shape[0], 1))])

    eps_ = sorted(set(E.tolist()))
    half = set(eps_[: len(eps_) // 2])
    fit = np.array([e in half for e in E])
    sc = ~fit
    fe = sorted(set(E[fit].tolist()))
    inner = set(fe[: len(fe) // 2])
    f2 = np.array([e in inner for e in E]) & fit
    v2 = fit & ~f2

    # ⛔ d MUST NOT OUTRUN n. PCA basis fit on the FIT EPISODES ONLY, never on the scored half.
    mu = F[fit].mean(0)
    U, S, Vt = np.linalg.svd(F[fit] - mu, full_matrices=False)
    n_pc = int(min(40, max(2, int(fit.sum()) // 8, 2)))
    F = np.hstack([(F - mu) @ Vt[:n_pc].T, np.ones((F.shape[0], 1))])
    print(f"PCA on FIT only: d {int(Vt.shape[1])} -> {n_pc} (+bias); "
          f"n_fit={int(fit.sum())} n_scored={int(sc.sum())}")

    oh = np.eye(K)[Y]

    # ⭐ CLASS-BALANCED SAMPLE WEIGHTS. Without them a one-vs-rest ridge on a 78 %-majority
    # target COLLAPSES TO THE MAJORITY — i.e. the probe fails in exactly the way the head under
    # test fails, and "the features carry no class" becomes indistinguishable from "my estimator
    # collapsed". ⛔ A CHECK THAT SHARES THE DEFECT IT CHECKS FOR IS GREEN FOREVER (CLAUDE.md,
    # 2026-09-07). Weights are counted on the FIT split only.
    def wts(mask):
        cnt = collections.Counter(Y[mask].tolist())
        return np.array([1.0 / cnt.get(int(y), 1) for y in Y[mask]])

    def fit_w(mask, lam):
        X, T, sw = F[mask], oh[mask], wts(mask)
        return np.linalg.solve(X.T @ (X * sw[:, None]) + lam * np.eye(X.shape[1]),
                               X.T @ (T * sw[:, None]))

    arms = {}
    for tag, fitter in (("UNBALANCED_shares_the_head_defect",
                         lambda m, l: fit_ridge(F[m], oh[m], l)),
                        ("BALANCED_inverse_frequency", fit_w)):
        b_, bb_ = None, -1.0
        for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5]:
            bb2 = fitter(f2, lam)
            ba2, _ = balanced_acc(Y[v2], (F[v2] @ bb2).argmax(1))
            if ba2 > bb_:
                b_, bb_ = lam, ba2
        be = fitter(fit, b_)
        yp_ = (F[sc] @ be).argmax(1)
        ba_, nc_ = balanced_acc(Y[sc], yp_)
        arms[tag] = {"lambda": b_, "balanced_accuracy": round(ba_, 5),
                     "raw_accuracy": round(float((yp_ == Y[sc]).mean()), 5),
                     "distinct_classes_predicted": len(set(yp_.tolist())),
                     "per_class_recall": {AGENT_CLASSES[k]: round(
                         float((yp_[Y[sc] == k] == k).mean()), 4)
                         for k in sorted(set(Y[sc].tolist()))}}
    best = arms["BALANCED_inverse_frequency"]["lambda"]
    beta = fit_w(fit, best)
    yp = (F[sc] @ beta).argmax(1)

    ba_probe, n_cls = balanced_acc(Y[sc], yp)
    acc_probe = float((yp == Y[sc]).mean())
    maj = collections.Counter(Y[fit].tolist()).most_common(1)[0][0]
    acc_maj = float((Y[sc] == maj).mean())
    ba_maj = 1.0 / n_cls

    sup_fit = collections.Counter(Y[fit].tolist())
    sup_sc = collections.Counter(Y[sc].tolist())
    per_cls = {}
    for k in sorted(set(Y[sc].tolist())):
        m = Y[sc] == k
        per_cls[AGENT_CLASSES[k]] = {
            "support_scored": int(m.sum()), "support_fit": int(sup_fit.get(k, 0)),
            "probe_recall": round(float((yp[m] == k).mean()), 4)}

    res = {"_what": "is the class signal IN the per-slot features and unused, or absent?",
           "_evidence_class": "MEASURED (ours), CPU, forward hook, A8 ckpt_5000",
           "_function_class": "LINEAR (ridge, one-vs-rest argmax). A failure bounds LINEAR "
                              "separability only; a success is conclusive.",
           "_hooked": hooked, "arms": arms, "n_pairs": int(F.shape[0]), "d": int(F.shape[1]),
           "n_episodes": len(eps_), "n_scored": int(sc.sum()),
           "n_classes_present_in_scored": n_cls,
           "lambda_chosen_on_FIT_only": best,
           "balanced_accuracy": {
               "PROBE_on_head_features": round(ba_probe, 5),
               "MAJORITY_constant_control": round(ba_maj, 5),
               "_control_is_a_known_value": f"1/{n_cls} exactly, for any constant predictor",
               "gain": round(ba_probe - ba_maj, 5)},
           "raw_accuracy_reported_but_not_the_statistic": {
               "PROBE": round(acc_probe, 5), "MAJORITY_constant": round(acc_maj, 5),
               "THE_HEAD_ITSELF": 0.77935},
           "per_class": per_cls,
           "classes_with_no_fit_support": [AGENT_CLASSES[k] for k in range(K)
                                           if sup_fit.get(k, 0) == 0 and sup_sc.get(k, 0) > 0]}
    n_fit = int(fit.sum())
    underpowered = n_fit < 5 * F.shape[1]
    wins = arms["BALANCED_inverse_frequency"]["balanced_accuracy"] > 2.0 * ba_maj
    res["power"] = {"n_fit": n_fit, "d_after_pca": int(F.shape[1]),
                    "rule": "n_fit >= 5*d required to read a NEGATIVE as evidence",
                    "underpowered": bool(underpowered)}
    res["_VERDICT"] = (
        "⚠️ INCONCLUSIVE — UNDERPOWERED BY CONSTRUCTION (n_fit < 5d). A probe in this "
        "regime cannot produce an admissible negative, and this one is NOT reported as evidence "
        "either way." if underpowered and not wins else
        "⛔⛔ LOSS DEFECT — a LINEAR probe on the very features the head's output layer reads "
        f"recovers balanced accuracy {ba_probe:.4f} against a constant predictor's {ba_maj:.4f}. "
        "The class signal IS present and the head is not using it ⇒ the collapse is in the "
        "objective, not the representation, and the fix is the one `NO_OBJECT_W` already applies "
        "to presence: weight the cls term."
        if wins else
        "⚠️ NOT A LOSS DEFECT BY THIS TEST — a linear probe on the head's own features barely "
        "beats a constant predictor, so the features may not linearly separate the classes and a "
        "re-weighting alone would not fix it. ⛔ LINEAR only: this does not prove the signal is "
        "absent, and a nonlinear probe is the next arm.")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n" + res["_VERDICT"])
    pathlib.Path("cls_probe.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
