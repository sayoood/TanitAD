"""``blind_conditioning_baseline`` -- the circularity firewall (spec §0.1).

THE RULE. For every (input set ``X``, target ``Y``) pair, train a predictor
``Y <- X_cond`` on the model's INFERENCE-TIME CONDITIONING CHANNELS ONLY -- no
pixels, no history. If that scene-blind predictor reaches ceiling, ``Y`` is a
copy of an input and is INADMISSIBLE.

THE PRECEDENT IT EXISTS FOR (MEASURED, ``HPP0_CONFOUND_AUDIT.md`` §1.1,
``stack/scripts/refb_labels.py:172-175``)::

    route_target(nav_cmd) -> return _NAV_TO_ROUTE[nav_cmd]   # target IS input

=> route CE hits exactly 0.0 by step ~14.5k, ``route_acc_nav = 1.0000``, and
``route_skill = 0.0000`` BY CONSTRUCTION. Months of "the strategic seam is
load-bearing" rested on a lookup table. This check costs CPU-minutes.

THE PRECEDENT THAT IT IS NOT PARANOIA (PUBLISHED): DriveBench's text-only
ablation put GPT-4o at 35.37 % clean vs 36.48 % text-only -- removing the image
HELPED.

DELIBERATELY CORPUS-AGNOSTIC. ``blind_conditioning_baseline`` knows nothing
about S3: it takes a feature matrix, a target, and episode ids. The sibling
stream's reusable version (``.../2026-07-26-4brain-preconditions/``) did not
exist when this ran; whichever lands second must DELETE its copy.

CPU only, torch, seconds-to-minutes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
def _fit_mlp(Xtr, ytr, Xte, n_classes, hidden=64, epochs=400, lr=3e-3,
             weight_decay=1e-4, seed=0, class_weighted=True):
    """2-layer MLP, CPU, minutes. Returns test-set argmax predictions.

    ``class_weighted`` makes the blind baseline as STRONG as it can be on rare
    bands -- a firewall must not be able to pass just because it collapsed to
    the majority class. That would be a false ADMISSION, the expensive direction.
    """
    torch.manual_seed(seed)
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32)
    Xte = torch.as_tensor(Xte, dtype=torch.float32)
    ytr = torch.as_tensor(ytr, dtype=torch.long)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, n_classes))
    w = None
    if class_weighted:
        cnt = torch.bincount(ytr, minlength=n_classes).float().clamp_min(1.0)
        w = (cnt.sum() / (n_classes * cnt))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss(weight=w)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xtr), ytr)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(Xte).argmax(1).numpy(), float(loss.detach())


def blind_conditioning_baseline(X_train, y_train, X_test, y_test,
                                eid_test, n_classes, score_fn,
                                feature_names=None, label="blind",
                                seed=0, **mlp_kw) -> dict:
    """Train ``y <- X_cond`` scene-blind and score it on a held-out,
    EPISODE-DISJOINT test set.

    ``score_fn(y_true, y_pred) -> dict`` supplies the problem's own metric, so
    this function never decides what "good" means. Returns the blind score, the
    majority-class score, and the REFUSAL verdict.

    ``eid_test`` is carried through so the caller can bootstrap the blind score
    with the same episode-cluster estimator as the model's.
    """
    yhat, final_loss = _fit_mlp(X_train, y_train, X_test, n_classes,
                                seed=seed, **mlp_kw)
    yt = np.asarray(y_test, dtype=np.int64)
    counts = np.bincount(np.asarray(y_train, dtype=np.int64),
                         minlength=n_classes)
    maj = int(counts.argmax())
    blind = score_fn(yt, yhat)
    major = score_fn(yt, np.full_like(yt, maj))
    return {
        "label": label,
        "n_features": int(np.asarray(X_train).shape[1]),
        "feature_names": list(feature_names) if feature_names else None,
        "n_train": int(np.asarray(X_train).shape[0]),
        "n_test": int(yt.size),
        "n_test_episodes": int(len(set(map(str, eid_test)))),
        "train_final_ce": round(final_loss, 4),
        "blind": blind,
        "majority": major,
        "pred": yhat,
    }


def refusal_verdict(blind_primary: float, ceiling: float = 1.0,
                    refuse_at_frac: float = 0.98) -> dict:
    """Spec §0.1 rule 5: ``acc_blind >= 0.98 * acc_ceiling`` => the label is
    REFUSED and does not enter the program."""
    thr = refuse_at_frac * ceiling
    return {"rule": "spec 0.1 step 5: blind >= 0.98 * ceiling => REFUSE",
            "ceiling": ceiling, "threshold": round(thr, 4),
            "blind_primary": round(float(blind_primary), 4),
            "REFUSED": bool(blind_primary >= thr)}


__all__ = ["blind_conditioning_baseline", "refusal_verdict", "_fit_mlp"]
