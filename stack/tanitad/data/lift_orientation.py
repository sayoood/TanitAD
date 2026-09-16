"""refcv6: the LIFT ORIENTATION GUARD -- does the lift put SAM3 drivable cells
under the image's road pixels, and does a MIRROR pay for it?

⛔ **Why this exists.** ``R-2026-09-08-wpa-mirror`` killed a previous result: a
left-right mirrored index is shape-preserving, loss-decreasing and completely
invisible in every aggregate. A BEV lift is exactly the kind of code that can be
mirrored and still look fine, so refcv6's brief makes the guard a deliverable:

    the lift must put SAM3 drivable cells under the image's road pixels on real
    frames, and a left-right MIRRORED index must go RED

## The instrument, and why it is not a hand-designed road cue

A first attempt scored image INTENSITY at the lifted positions against a
hand-picked "road-like" cue. MEASURED 2026-09-16 on 36 frames of 12 clips it
separated drivable from non-drivable at AUC 0.707 -- but the MIRROR still scored
0.606 and won on 39 % of frames, because the ROW of a lifted cell is untouched
by a left-right mirror and carries most of that separation. An instrument that a
mirror passes 39 % of the time is a C9/C14 instrument: structurally unable to
report the answer it is cited for.

:func:`orientation_probe` instead runs **the MAP head's own operation in
miniature**: a logistic probe on PURE-IMAGE features sampled through the lift's
own ``grid``, fitted on TRAIN clips and scored on HELD-OUT clips. The BEV
position enters only through WHERE the image is sampled -- which is precisely
what a mirror corrupts -- and the probe extracts whatever image evidence exists
rather than whatever evidence the author thought of.

⚠️ The mirror does NOT fall to 0.5, and must not be expected to: a left-right
mirror preserves the row, so the vertical structure of a road scene survives it.
The guard is the **paired comparison** (true > mirror, per held-out clip), not
an absolute threshold on either.

## The features

``[Z * C]`` per BEV cell: the image average-pooled to the trunk's stride-16 grid
(16 x 40 -- ``SPEC_REFCV6_V2.md`` §2's perception stride) sampled bilinearly at
each of the lift's ``Z`` height planes. ⛔ The cell's own ``(x, y)`` is
deliberately NOT a feature: with it the probe could learn the BEV prior ("cells
straight ahead are drivable") and score well through a mirror, measuring the
prior instead of the projection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from tanitad.data.semantic_map_gt import CHANNELS

__all__ = [
    "DRIVABLE_CHANNEL", "POS_FRAC", "NEG_FRAC", "OrientationResult",
    "MIN_CLASS_N", "mirror_grid", "lift_image_features", "rank_auc",
    "fit_logistic", "sign_test_p", "sign_test_power", "min_n_for_power",
    "sign_test_critical", "MIN_CLIPS_FOR_POWER", "MEASURED_WIN_RATE",
    "apply_logistic", "orientation_probe",
]

#: index of ``"drivable"`` in :data:`semantic_map_gt.CHANNELS` -- looked up, not
#: written, so a channel-list change cannot silently retarget the guard.
DRIVABLE_CHANNEL: int = CHANNELS.index("drivable")
#: a cell counts as drivable at >= this fraction and as non-drivable at <= the
#: other; cells in between are DROPPED rather than forced to a side.
POS_FRAC: float = 0.6
NEG_FRAC: float = 0.05
#: fewest items a class may have before :func:`rank_auc` refuses to report a
#: number. An AUC on 3 positives is noise, and printing it is how a guard
#: becomes decoration.
MIN_CLASS_N: int = 20  # geometry-exempt: a sample-size floor, not an image width


def mirror_grid(grid: Tensor) -> Tensor:
    """The MUTATION: a left-right mirrored feature index.

    ``grid`` is ``grid_sample`` coordinates ``[..., 2] = (gx, gy)`` with ``gx``
    indexing the image COLUMN; negating ``gx`` maps column ``u`` to ``W-1-u``
    under ``align_corners=False``. Returns a copy; the input is untouched.
    """
    if grid.shape[-1] != 2:
        raise ValueError(f"grid must end in 2, got {tuple(grid.shape)}")
    out = grid.clone()
    out[..., 0] = -out[..., 0]
    return out


def lift_image_features(image: Tensor, grid: Tensor, *, stride: int = 16
                        ) -> Tensor:
    """``image [C,H,W]`` (or ``[1,C,H,W]``) + ``grid [Z,X,Y,2]`` ->
    ``[X, Y, Z*C]`` pure-image features.

    The image is average-pooled by ``stride`` FIRST, so the features are on the
    same grid the trunk's stride-16 map lives on and the guard tests the
    projection the MAP head will actually use.
    """
    img = image if image.dim() == 4 else image.unsqueeze(0)
    if img.dim() != 4:
        raise ValueError(f"image must be [C,H,W] or [1,C,H,W], got "
                         f"{tuple(image.shape)}")
    if grid.dim() != 4 or grid.shape[-1] != 2:
        raise ValueError(f"grid must be [Z,X,Y,2], got {tuple(grid.shape)}")
    fmap = F.avg_pool2d(img.to(torch.float32), int(stride))
    Z, X, Y, _ = grid.shape
    s = F.grid_sample(fmap.expand(Z, -1, -1, -1), grid.to(torch.float32),
                      mode="bilinear", padding_mode="border", align_corners=False)
    return s.permute(2, 3, 0, 1).reshape(X, Y, Z * fmap.shape[1])


def rank_auc(score, label) -> float:
    """Tie-corrected Mann-Whitney AUC. NaN when either class has fewer than
    :data:`MIN_CLASS_N` items (see that constant for why)."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=bool).reshape(-1)
    if score.shape != label.shape:
        raise ValueError(f"score {score.shape} and label {label.shape} differ")
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 < MIN_CLASS_N or n0 < MIN_CLASS_N:
        return float("nan")
    order = np.argsort(score, kind="stable")
    r = np.empty(score.size, dtype=np.float64)
    s = score[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fit_logistic(X, y, *, l2: float = 1e-3, max_iter: int = 300) -> dict:
    """L-BFGS logistic regression on standardised features. Returns the fit as
    a dict so :func:`apply_logistic` cannot be handed a bare tensor by mistake."""
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1)
    if Xt.dim() != 2 or Xt.shape[0] != yt.shape[0]:
        raise ValueError(f"X {tuple(Xt.shape)} and y {tuple(yt.shape)} disagree")
    mu = Xt.mean(0, keepdim=True)
    sd = Xt.std(0, keepdim=True).clamp_min(1e-6)
    Xn = (Xt - mu) / sd
    w = torch.zeros(Xn.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=int(max_iter),
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xn @ w + b, yt) \
            + float(l2) * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"w": w.detach(), "b": b.detach(), "mu": mu, "sd": sd,
            "n_fit": int(yt.numel()), "pos_rate": float(yt.mean())}


def apply_logistic(fit: dict, X) -> np.ndarray:
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    return ((Xt - fit["mu"]) / fit["sd"] @ fit["w"] + fit["b"]).numpy()


#: ⭐ THE WIN RATE THE INSTRUMENT ACTUALLY HAS, measured three times at three
#: fit sizes on 2026-09-16: 0.7304 (115 held out, 20 fit), 0.7200 (75 held out,
#: 60 fit), 1.0000 (10 held out, 20 fit). :data:`MIN_CLIPS_FOR_POWER` is derived
#: from the LOWEST of these, rounded down -- a power calculation done at the
#: rate you hope for is not a power calculation.
MEASURED_WIN_RATE: float = 0.70

#: ⛔ **The n at which :func:`orientation_probe` is POWERED**, and it is computed
#: (:func:`min_n_for_power`), not chosen: the smallest ``n`` at which the exact
#: one-sided sign test at ``alpha = 0.01`` rejects with probability >= 0.95 when
#: the true per-clip win rate is :data:`MEASURED_WIN_RATE`.
#:
#: ⚠️ **This constant exists because the guard failed on its own default run.**
#: The first version scored ~20 held-out clips; at n = 20 the test's power is
#: **0.24** at p = 0.70 (0.34 even at p = 0.73), so a CORRECT lift was expected
#: to be reported RED most of the time. A guard that goes red by default is
#: indistinguishable from the defect it exists to catch -- the
#: ``R-2026-09-08-wpa-mirror`` failure mode wearing the opposite mask. Below this
#: n the caller must SKIP and say so, never assert.
MIN_CLIPS_FOR_POWER: int = 96


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    from math import lgamma, log
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
            + k * log(p) + (n - k) * log(1.0 - p))


def sign_test_critical(n: int, alpha: float = 0.01) -> int:
    """Smallest ``k`` whose one-sided p under ``p = 0.5`` is at most ``alpha``."""
    for k in range(int(n), -1, -1):
        if sign_test_p(k, n) > float(alpha):
            return k + 1
    return 0


def sign_test_power(n: int, win_rate: float = MEASURED_WIN_RATE,
                    alpha: float = 0.01) -> float:
    """Probability the sign test rejects at ``alpha`` when the true per-clip win
    rate is ``win_rate`` -- i.e. the chance the guard REPORTS a correct lift as
    correct. MEASURED power at the sizes that matter (``win_rate = 0.70``):
    n = 20 -> 0.24, n = 50 -> 0.68, n = 96 -> 0.95, n = 115 -> 0.98."""
    from math import exp
    k = sign_test_critical(n, alpha)
    if k > int(n):
        return 0.0
    return float(sum(exp(_log_binom_pmf(i, int(n), float(win_rate)))
                     for i in range(k, int(n) + 1)))


def min_n_for_power(target: float = 0.95, win_rate: float = MEASURED_WIN_RATE,
                    alpha: float = 0.01, n_max: int = 500) -> int:
    """Smallest ``n`` whose :func:`sign_test_power` reaches ``target``.
    :data:`MIN_CLIPS_FOR_POWER` is this, and the constant is pinned against it
    by a test so the two can never drift."""
    for n in range(2, int(n_max)):
        if sign_test_power(n, win_rate, alpha) >= float(target):
            return n
    raise ValueError(f"no n <= {n_max} reaches power {target} at win rate "
                     f"{win_rate}")


def sign_test_p(n_wins: int, n: int) -> float:
    """Exact one-sided binomial p of ``n_wins`` or more under ``p = 0.5``.

    ⭐ The verdict is a SIGN TEST, not a win-rate threshold. MEASURED
    2026-09-16: over 115 held-out clips the true orientation wins 84 -- a win
    FRACTION of 0.73, which a "must win 90 %" rule would call a FAIL even though
    its p is 4e-7. Per-clip AUC is noisy (a clip is 3-5 frames); the population
    of clips is not. A threshold on the noisy quantity would have been an
    instrument that fails on the truth, which is the same defect in the other
    direction as one that passes on a mirror.
    """
    n, k = int(n), int(n_wins)
    if n <= 0:
        return float("nan")
    k = max(0, min(k, n))
    # sum_{i=k}^{n} C(n,i) / 2^n, computed in log space
    from math import lgamma, exp, log
    logc = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) for i in range(k, n + 1)]
    mx = max(logc)
    return float(exp(mx + log(sum(exp(c - mx) for c in logc)) - n * log(2.0)))


@dataclass(frozen=True)
class OrientationResult:
    """The paired comparison.

    ``verdict`` is ``"PASS"`` only when the TRUE orientation beats the MIRROR on
    significantly more than half the held-out clips (:func:`sign_test_p` below
    ``alpha``) AND the mean AUC margin is positive; anything else is ``"FAIL"``
    (or ``"INCONCLUSIVE"`` with too few scorable clips).
    """

    mean_auc_true: float
    mean_auc_mirror: float
    median_auc_true: float
    median_auc_mirror: float
    n_clips_scored: int
    n_wins: int
    mean_margin: float
    alpha: float
    p_sign: float
    verdict: str
    per_clip: tuple

    @property
    def win_frac(self) -> float:
        return self.n_wins / self.n_clips_scored if self.n_clips_scored else float("nan")


def orientation_probe(train_samples, eval_samples, *, alpha: float = 0.01,
                      min_clips: int = 5, l2: float = 1e-3) -> OrientationResult:
    """Fit on ``train_samples``, score ``eval_samples``, TRUE versus MIRROR.

    Both arguments are iterables of ``(clip_sha12, X_true, X_mirror, y)`` --
    already-extracted features, so this function opens no file and the caller
    owns the data paths. ``X_true`` / ``X_mirror`` are ``[n, F]``, ``y`` is
    ``[n]`` bool (drivable).

    ⛔ Train and eval clips must be DISJOINT; the probe refuses otherwise. A
    probe scored on its own fit clips can memorise per-clip colour and then a
    mirror costs it nothing -- the guard would pass a mirrored build.
    """
    tr = list(train_samples)
    te = list(eval_samples)
    tr_ids = {s[0] for s in tr}
    te_ids = {s[0] for s in te}
    overlap = tr_ids & te_ids
    if overlap:
        raise ValueError(
            f"{len(overlap)} clip(s) appear in BOTH the fit and the eval set "
            f"(e.g. {sorted(overlap)[0]}). ⛔ A probe scored on its own fit "
            f"clips can memorise per-clip colour, and a mirror then costs it "
            f"nothing -- the guard would pass a mirrored build.")
    if not tr or not te:
        raise ValueError("orientation_probe needs both fit and eval samples")
    per = []
    fits = {}
    for key, col in (("true", 1), ("mirror", 2)):
        X = np.concatenate([np.asarray(s[col]) for s in tr])
        y = np.concatenate([np.asarray(s[3]) for s in tr])
        fits[key] = fit_logistic(X, y, l2=l2)
    for s12, Xt, Xm, y in te:
        y = np.asarray(y, dtype=bool)
        per.append({
            "clip_sha12": str(s12), "n": int(y.size), "pos_rate": float(y.mean()),
            "auc_true": rank_auc(apply_logistic(fits["true"], Xt), y),
            "auc_mirror": rank_auc(apply_logistic(fits["mirror"], Xm), y)})
    pairs = [(p["auc_true"], p["auc_mirror"]) for p in per
             if np.isfinite(p["auc_true"]) and np.isfinite(p["auc_mirror"])]
    n = len(pairs)
    if n < int(min_clips):
        return OrientationResult(
            float("nan"), float("nan"), float("nan"), float("nan"), n, 0,
            float("nan"), float(alpha), float("nan"), "INCONCLUSIVE", tuple(per))
    t = np.array([a for a, _ in pairs])
    m = np.array([b for _, b in pairs])
    wins = int((t > m).sum())
    margin = float((t - m).mean())
    p = sign_test_p(wins, n)
    ok = p < float(alpha) and margin > 0.0
    return OrientationResult(
        float(t.mean()), float(m.mean()), float(np.median(t)), float(np.median(m)),
        n, wins, margin, float(alpha), p, "PASS" if ok else "FAIL", tuple(per))


def cells_for_probe(seen, drivable_frac, extra_mask=None) -> tuple:
    """``(mask [X,Y] bool, label [n] bool)``: the cells the probe scores.

    Only cells that are SEEN and unambiguous (``>= POS_FRAC`` or
    ``<= NEG_FRAC``) take part; the band in between is dropped, not rounded.
    """
    m = torch.as_tensor(np.asarray(seen), dtype=torch.bool)
    d = torch.as_tensor(np.asarray(drivable_frac), dtype=torch.float32)
    if extra_mask is not None:
        m = m & torch.as_tensor(np.asarray(extra_mask), dtype=torch.bool)
    m = m & ((d >= POS_FRAC) | (d <= NEG_FRAC))
    return m, (d[m] >= POS_FRAC).numpy()
