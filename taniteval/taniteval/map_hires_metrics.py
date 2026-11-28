"""refcv7 NEW-2 -- map quality at **10 cm**, banded by range, with a paired clip bootstrap.

``Project Steering/SPEC_REFCV7.md`` §6.2 pre-registers four bars (BAR-M7-1..4) on
the eval kit's 137 SAM3 GT clips at 10 cm. This module is the scorer they read:

* **per-class IoU on SEEN cells** (``code != 255``), for all 8 classes, in EVERY
  20 m range band of the grid (row 0 = nearest; 200 fine rows per band): 0-20 /
  20-40 / 40-60 m on the ``/2`` grid, plus 60-80 / 80-100 m at the refcv7 extent
  (SPEC_REFCV7 §11.2 / §12, A6/A7: 100 m x +-30 m = 1000 x 600). Nothing here types a
  grid size: the bands follow the arrays' own row count;
* **precision / recall / F1 at a 0.2 m tolerance** for the thin classes (lane /
  road line, non-drivable edge, crosswalk) -- a predicted cell is a hit when a GT
  cell of its class lies STRICTLY closer than 0.2 m (centre to centre, Euclidean
  distance transform), and a GT cell is recovered when a predicted cell of its class
  does. ⚠️ STRICT: at 0.1 m cells a 1-cell (0.1 m) shift is inside, a 2-cell
  (0.2 m) shift is ON the boundary and therefore OUTSIDE, a 3-cell shift is far
  outside. The literal tests pin exactly this;
* **the refcv6 baseline hook** -- a 0.5 m ``[9 | 8, 120, 64]`` prediction, argmax
  over the 8 CLASS channels (the ``not seen`` logit is dropped BEFORE the argmax, so
  the baseline is never penalised for calling a seen cell unseen -- the favourable
  reading, which makes a refcv7 win conservative), nearest-upsampled 5 x 5 onto the
  10 cm grid, exactly the cell correspondence ``cart_frac`` has with ``fine_codes``.
  ⚠️ refcv6 predicts ONLY its 60 m x +-16 m window: on a larger extent every other
  cell is :data:`NO_PREDICTION` (the anchored window of ``MapExtent.window_in``), so
  its seen GT there counts as missed. That is the honest reading of "refcv6 has no
  map there", and it is why far bands are reported with their n (§11.2 item 1);
* **the positional-prior control** -- each cell's majority class over a FIT set of
  clips, which :meth:`PositionalPrior.predict_for` REFUSES to score on any clip it
  was fitted on;
* **the paired episode-cluster bootstrap over clips** -- the programme's own
  :func:`taniteval.ci.paired_episode_cluster_bootstrap`, not a second estimator.

⭐ POOLED, NOT MEAN-OF-WINDOWS. Every statistic is a ratio of sums over the scored
windows (sum of intersections / sum of unions; sum of hits / sum of cells). A
per-window IoU is undefined when neither the GT nor the prediction has the class,
and averaging only the defined windows makes the WINDOW SET depend on the arm's
predictions -- two arms would be scored on different windows. The per-window
SUFFICIENT STATISTICS are banked instead, and each bootstrap draw recomputes the
pooled ratio on the resampled clips (the ``ci`` module's documented callable-reducer
path). ⚠️ This choice is the implementer's, made before any refcv7 number exists;
SPEC_REFCV7 §6.2 does not name it. Flagged for the Master Mind.

Pure numpy: NO scipy (the Thor training venv has none, MEASURED 2026-09-27 -- a
module-level scipy import made this module's tests an import error on the Thor gate)
and no torch. The 0.2 m tolerance is an exact offset footprint (:func:`_tol_offsets`),
equal to a Euclidean distance transform's ``< tol`` (tested against scipy's where scipy
exists).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from taniteval import ci
from tanitad.data.semantic_map_gt_fine import (
    BAND_M, BAND_ROWS_FINE, BLOCK, EXTENT_V2, FINE_CELL_M, FINE_CLASSES, FINE_SHAPE,
    N_FINE_CLASSES, NOT_SEEN_CODE, MapExtent,
)

__all__ = [
    "TOL_M", "THIN_CLASSES", "CLASS_NAMES", "BAND_NAMES", "STATS", "NO_PREDICTION",
    "band_rows", "band_names",
    "coarse_to_fine_codes", "logits_to_codes", "window_stats", "pooled",
    "PositionalPrior", "WindowTable", "summarize", "paired_delta",
    "evaluate_bars_m7",
]

#: SPEC_REFCV7 §6.2: the tolerance of the thin-class F1, metres (STRICT <).
TOL_M: float = 0.2
CLASS_NAMES: tuple[str, ...] = FINE_CLASSES
#: lane / road line (2), non-drivable edge (5), crosswalk (3) -- the spec's order.
THIN_CLASSES: tuple[int, ...] = (2, 5, 3)
#: per-window sufficient statistics, each [8 classes, n_bands] int64
STATS: tuple[str, ...] = ("inter", "union", "n_gt", "n_pred", "tp_pred", "tp_gt")
#: a cell an arm makes NO prediction for (the refcv6 0.5 m hook outside its own grid).
#: It is no class: it adds to no class's prediction, so seen GT there is missed.
NO_PREDICTION: int = 254


def band_rows(n_rows: int) -> tuple[tuple[int, int], ...]:
    """Every 20 m band of a 10 cm grid with ``n_rows`` rows: 200 rows each from the
    nearest row, a partial last band keeping its true edge."""
    n = int(n_rows)
    return tuple((a, min(a + BAND_ROWS_FINE, n)) for a in range(0, n, BAND_ROWS_FINE))


def _fmt(m: float) -> str:
    return str(int(round(m))) if abs(m - round(m)) < 1e-9 else f"{m:g}"


def band_names(n_rows: int) -> tuple[str, ...]:
    """``"0-20m"``, ``"20-40m"``, ... for a grid of ``n_rows`` fine rows."""
    return tuple(f"{_fmt(a * FINE_CELL_M)}-{_fmt(b * FINE_CELL_M)}m"
                 for a, b in band_rows(n_rows))


#: the ``/2`` grid's band names (``0-20m``, ``20-40m``, ``40-60m``)
BAND_NAMES: tuple[str, ...] = band_names(FINE_SHAPE[0])
if abs(BAND_ROWS_FINE * FINE_CELL_M - BAND_M) > 1e-9:
    raise RuntimeError("map_hires_metrics: 200 fine rows are not the 20 m band")


# --------------------------------------------------------------------------- #
# predictions -> 10 cm codes                                                   #
# --------------------------------------------------------------------------- #
def coarse_to_fine_codes(pred_05, extent: MapExtent = EXTENT_V2) -> np.ndarray:
    """refcv6 hook: ``[9|8, 120, 64]`` logits/probabilities -> ``extent.fine_shape``
    uint8 (``[600, 320]`` on the ``/2`` grid).

    Argmax over channels 0..7 ONLY (the ``not seen`` channel 8, when present, is
    dropped first), then each 0.5 m cell is repeated onto its 5 x 5 fine block --
    row ``i`` -> fine rows ``5i..5i+4``, col ``j`` -> ``5j..5j+4``, the correspondence
    ``semantic_map_gt_fine.block_fraction_u8`` reproduces ``cart_frac`` with. On a
    larger extent the result sits in its ANCHORED 60 m x +-16 m window and every other
    cell is :data:`NO_PREDICTION` (refcv6 has no map there)."""
    p = np.asarray(pred_05)
    if p.ndim != 3 or p.shape[0] not in (N_FINE_CLASSES, N_FINE_CLASSES + 1) \
            or tuple(p.shape[1:]) != (FINE_SHAPE[0] // BLOCK, FINE_SHAPE[1] // BLOCK):
        raise ValueError(f"a 0.5 m prediction is [9|8, 120, 64], got {p.shape}")
    am = np.argmax(p[:N_FINE_CLASSES], axis=0).astype(np.uint8)
    fine = np.repeat(np.repeat(am, BLOCK, axis=0), BLOCK, axis=1)
    if extent == EXTENT_V2:
        return fine
    r0, r1, c0, c1 = EXTENT_V2.window_in(extent)
    out = np.full(tuple(extent.fine_shape), NO_PREDICTION, np.uint8)
    out[r0:r1, c0:c1] = fine
    return out


#: the decision rules of ``map_head_hires.DECISION_RULES`` (restated: no torch here)
DECISION_RULES: tuple[str, ...] = ("prior_corrected", "raw")


def logits_to_codes(logits_10, *, rule: str = "prior_corrected",
                    class_weight=None) -> np.ndarray:
    """refcv7 hook: ``[8, H, W]`` logits -> ``[H, W]`` uint8 codes (any extent).

    ⭐ DEFAULT ``"prior_corrected"`` = ``argmax_c (z_c - log w_c)`` with the loss's
    FROZEN class weights (map-signal audit D1, accepted 2026-09-26): under a weighted
    CE the softmax learns ``q_c ∝ w_c P(c|x)``, so the raw argmax calls a rare class at
    a few percent posterior. ``"raw"`` = ``argmax_c z_c`` (the diagnostic).
    ⛔ ``class_weight=None`` with the prior-corrected rule is REFUSED -- it would
    silently be the raw rule on a weighted model; pass ones for uniform weights.
    Same formula as ``map_head_hires.decide`` (pinned by a test), including its
    rule for a weight of 0: that class is NEVER decided (``z - log 0`` would be
    ``+inf`` and decide it everywhere)."""
    p = np.asarray(logits_10, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] != N_FINE_CLASSES:
        raise ValueError(f"a 10 cm prediction is [8, H, W], got {p.shape}")
    if rule not in DECISION_RULES:
        raise ValueError(f"decision rule {rule!r} not in {DECISION_RULES}")
    if rule == "raw":
        return np.argmax(p, axis=0).astype(np.uint8)
    if class_weight is None:
        raise ValueError("the prior-corrected rule needs the loss's class weights "
                         "(ones for uniform); None would silently be `raw`")
    w = np.asarray(class_weight, dtype=np.float64).reshape(-1)
    if w.shape != (N_FINE_CLASSES,):
        raise ValueError(f"class_weight must have {N_FINE_CLASSES} entries")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError(f"class weights must be finite and >= 0, got {w.tolist()}")
    lw = np.where(w > 0, np.log(np.where(w > 0, w, 1.0)), np.inf)
    return np.argmax(p - lw[:, None, None], axis=0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# one window                                                                   #
# --------------------------------------------------------------------------- #
def _scored_mask(gt: np.ndarray, valid) -> np.ndarray:
    """The cells a window is scored on: SEEN (``gt != 255``), narrowed by an
    optional ``valid`` mask. ⛔ One definition, used by every statistic."""
    m = gt != NOT_SEEN_CODE
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        if v.shape != gt.shape:
            raise ValueError(f"valid must be {gt.shape}, got {v.shape}")
        m = m & v
    return m


def _tol_offsets(tol_m: float, cell_m: float = FINE_CELL_M) -> tuple:
    """The ``(di, dj)`` cell offsets whose centre-to-centre Euclidean distance is STRICTLY
    below ``tol_m`` -- the footprint of ``distance < tol``. At 0.1 m cells and 0.2 m: the 9
    cells of the 3 x 3 square; ``(+-2, 0)`` / ``(0, +-2)`` sit at exactly 0.2 m and are OUT.
    The distance is computed as a distance transform with ``sampling`` computes it:
    ``sqrt((di * cell)^2 + (dj * cell)^2)``. ⛔ The ONE place the strictness lives."""
    r = int(math.ceil(float(tol_m) / float(cell_m)))
    return tuple((di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
                 if math.sqrt((di * cell_m) ** 2 + (dj * cell_m) ** 2) < tol_m)


def _near(mask: np.ndarray, tol_m: float) -> np.ndarray:
    """``True`` where the nearest ``True`` cell of ``mask`` is STRICTLY closer than ``tol_m``
    (cell centre to cell centre, Euclidean): exactly ``distance_transform_edt(~mask,
    sampling=FINE_CELL_M) < tol_m`` -- the OR of ``mask`` shifted by every offset of
    :func:`_tol_offsets`, cells off the grid contributing nothing. All False when ``mask``
    is empty (no cell is near nothing)."""
    m = np.asarray(mask, dtype=bool)
    out = np.zeros(m.shape, dtype=bool)
    if not m.any():
        return out
    h, w = m.shape
    for di, dj in _tol_offsets(tol_m):
        si0, si1, di0, di1 = max(0, di), min(h, h + di), max(0, -di), min(h, h - di)
        sj0, sj1, dj0, dj1 = max(0, dj), min(w, w + dj), max(0, -dj), min(w, w - dj)
        if si1 > si0 and sj1 > sj0:
            out[di0:di1, dj0:dj1] |= m[si0:si1, sj0:sj1]
    return out


def _band_sum(m: np.ndarray) -> np.ndarray:
    return np.array([int(m[a:b].sum()) for a, b in band_rows(m.shape[0])],
                    dtype=np.int64)


def window_stats(pred, gt, *, valid=None, tol_m: float = TOL_M,
                 tol_classes=THIN_CLASSES) -> dict:
    """Sufficient statistics of ONE window: ``{stat: [8, n_bands] int64}``.

    ``pred`` ``[H, W]`` codes 0..7, or :data:`NO_PREDICTION` for a cell the arm does
    not cover (any other value is refused -- an un-decided cell has no class to be
    scored as); ``gt`` ``[H, W]`` codes with 255 = not seen, on the SAME grid (any
    extent; the bands follow ``H``). ``tp_pred`` / ``tp_gt`` are filled for
    ``tol_classes`` only (zero elsewhere)."""
    p = np.asarray(pred)
    g = np.asarray(gt)
    if p.ndim != 2 or p.shape != g.shape:
        raise ValueError(f"pred and gt must be [H, W] on the same grid; got {p.shape}, "
                         f"{g.shape}")
    if p.size:
        bad = (p > N_FINE_CLASSES - 1) & (p != NO_PREDICTION)
        if bad.any():
            raise ValueError(f"pred carries code {int(p[bad].max())}: neither a class "
                             f"0..{N_FINE_CLASSES - 1} nor NO_PREDICTION")
    s = _scored_mask(g, valid)
    out = {k: np.zeros((N_FINE_CLASSES, len(band_rows(g.shape[0]))), np.int64)
           for k in STATS}
    tol = float(tol_m)
    for k in range(N_FINE_CLASSES):
        gk = (g == k) & s
        pk = (p == k) & s
        out["inter"][k] = _band_sum(gk & pk)
        out["union"][k] = _band_sum(gk | pk)
        out["n_gt"][k] = _band_sum(gk)
        out["n_pred"][k] = _band_sum(pk)
        if k in tol_classes:
            out["tp_pred"][k] = _band_sum(pk & _near(gk, tol))
            out["tp_gt"][k] = _band_sum(gk & _near(pk, tol))
    return out


# --------------------------------------------------------------------------- #
# pooled statistics                                                            #
# --------------------------------------------------------------------------- #
def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else math.nan


def _f1(tp_pred, n_pred, tp_gt, n_gt) -> tuple[float, float, float]:
    """``(P, R, F1)`` from pooled counts. F1 = 0 when exactly one side is empty
    (a class predicted where there is none, or present and never predicted);
    NaN when neither side has the class."""
    P = _ratio(tp_pred, n_pred)
    R = _ratio(tp_gt, n_gt)
    if not n_pred and not n_gt:
        return math.nan, math.nan, math.nan
    if not n_pred or not n_gt:
        return P, R, 0.0
    return P, R, (0.0 if P + R == 0 else 2.0 * P * R / (P + R))


def pooled(stats: dict, rows=None) -> dict:
    """Pooled metrics over windows ``rows`` (all when None) of a stats dict whose
    arrays are ``[W, 8, n_bands]``: ``iou`` / ``P`` / ``R`` / ``F1`` each
    ``[8, n_bands]``."""
    sel = slice(None) if rows is None else np.asarray(rows, dtype=np.int64)
    tot = {k: np.asarray(v)[sel].sum(axis=0) for k, v in stats.items()}
    nb = int(tot["inter"].shape[1])
    iou = np.full((N_FINE_CLASSES, nb), math.nan)
    P = np.full_like(iou, math.nan)
    R = np.full_like(iou, math.nan)
    F = np.full_like(iou, math.nan)
    for k in range(N_FINE_CLASSES):
        for b in range(nb):
            iou[k, b] = _ratio(tot["inter"][k, b], tot["union"][k, b])
            P[k, b], R[k, b], F[k, b] = _f1(tot["tp_pred"][k, b], tot["n_pred"][k, b],
                                           tot["tp_gt"][k, b], tot["n_gt"][k, b])
    return {"iou": iou, "P": P, "R": R, "F1": F}


# --------------------------------------------------------------------------- #
# the positional prior                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class PositionalPrior:
    """Each cell's majority class over a FIT set (seen observations only).

    ⛔ ``predict_for(clip_sha12)`` REFUSES a clip the prior was fitted on: the
    control must never be scored on its own fit set (SPEC_REFCV7 §6.2: "each cell's
    train-set majority class"). Cells never seen in the fit set take the fit set's
    global majority class, which is recorded."""

    counts: np.ndarray | None = None
    fit_clips: set = field(default_factory=set)
    n_frames: int = 0
    #: the map extent the prior is fitted and scored on (SPEC_REFCV7 §11.2)
    extent: MapExtent = EXTENT_V2

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = np.zeros((N_FINE_CLASSES,) + tuple(self.extent.fine_shape),
                                   np.int64)
        if tuple(self.counts.shape[1:]) != tuple(self.extent.fine_shape):
            raise ValueError(f"counts {self.counts.shape} are not on the extent's grid "
                             f"{self.extent.fine_shape}")

    def add(self, clip_sha12: str, codes) -> None:
        c = np.asarray(codes)
        if c.ndim == 2:
            c = c[None]
        if c.ndim != 3 or tuple(c.shape[1:]) != tuple(self.extent.fine_shape):
            raise ValueError(f"codes must be [N, {self.extent.fine_shape[0]}, "
                             f"{self.extent.fine_shape[1]}], got {c.shape}")
        for k in range(N_FINE_CLASSES):
            self.counts[k] += (c == k).sum(axis=0)
        self.fit_clips.add(str(clip_sha12))
        self.n_frames += int(c.shape[0])

    @property
    def global_majority(self) -> int:
        return int(np.argmax(self.counts.sum(axis=(1, 2))))

    def majority_map(self) -> np.ndarray:
        if not self.fit_clips:
            raise ValueError("PositionalPrior has no fit data")
        seen_any = self.counts.sum(axis=0) > 0
        m = np.argmax(self.counts, axis=0).astype(np.uint8)
        m[~seen_any] = np.uint8(self.global_majority)
        return m

    def predict_for(self, clip_sha12: str) -> np.ndarray:
        if str(clip_sha12) in self.fit_clips:
            raise ValueError(
                f"positional prior asked to predict clip sha12 {clip_sha12}, which "
                f"is in its FIT set: the control would be scored on the data it "
                f"was fitted to")
        return self.majority_map()

    def fingerprint(self) -> dict:
        h = hashlib.sha256("\n".join(sorted(self.fit_clips)).encode()).hexdigest()
        return {"n_fit_clips": len(self.fit_clips), "n_fit_frames": self.n_frames,
                "fit_clips_sha256": h, "global_majority": self.global_majority,
                "counts_sha256": hashlib.sha256(self.counts.tobytes()).hexdigest()}

    def save(self, path) -> None:
        np.savez_compressed(path, counts=self.counts,
                            fit_clips=np.array(sorted(self.fit_clips)),
                            n_frames=np.int64(self.n_frames),
                            extent=np.array([float(self.extent.x_max_m),
                                             float(self.extent.y_half_m)]))

    @classmethod
    def load(cls, path) -> "PositionalPrior":
        with np.load(path, allow_pickle=False) as z:
            ext = (MapExtent(float(z["extent"][0]), float(z["extent"][1]))
                   if "extent" in z.files else EXTENT_V2)
            return cls(counts=z["counts"].astype(np.int64),
                       fit_clips={str(x) for x in z["fit_clips"]},
                       n_frames=int(z["n_frames"]), extent=ext)


# --------------------------------------------------------------------------- #
# many windows, several arms                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class WindowTable:
    """Per-window sufficient statistics of several arms on the SAME windows."""

    arms: tuple
    eid: list = field(default_factory=list)                 # clip sha12 per window
    stats: dict = field(default_factory=dict)               # arm -> stat -> list
    #: fine rows of the scored grid (sets the bands); fixed by the first window
    n_rows: int = 0

    @property
    def band_names(self) -> tuple[str, ...]:
        return band_names(self.n_rows) if self.n_rows else BAND_NAMES

    def add(self, clip_sha12: str, gt, preds: dict, *, valid=None,
            tol_m: float = TOL_M) -> None:
        rows = int(np.asarray(gt).shape[0])
        if self.n_rows and rows != self.n_rows:
            raise ValueError(f"window of {rows} fine rows in a table of {self.n_rows}: "
                             f"one table, one extent")
        self.n_rows = rows
        if set(preds) != set(self.arms):
            raise ValueError(f"window scored for arms {sorted(preds)}, the table "
                             f"holds {sorted(self.arms)}: every arm must be scored "
                             f"on every window (a paired test needs aligned arms)")
        for a in self.arms:
            st = window_stats(preds[a], gt, valid=valid, tol_m=tol_m)
            dst = self.stats.setdefault(a, {k: [] for k in STATS})
            for k in STATS:
                dst[k].append(st[k])
        self.eid.append(str(clip_sha12))

    def arrays(self, arm: str) -> dict:
        return {k: np.stack(v) for k, v in self.stats[arm].items()}

    @property
    def n_windows(self) -> int:
        return len(self.eid)

    @property
    def n_clips(self) -> int:
        return len(set(self.eid))

    def save(self, path) -> None:
        z = {"eid": np.array(self.eid), "arms": np.array(self.arms),
             "n_rows": np.int64(self.n_rows)}
        for a in self.arms:
            for k, v in self.arrays(a).items():
                z[f"{a}__{k}"] = v
        np.savez_compressed(path, **z)

    @classmethod
    def load(cls, path) -> "WindowTable":
        with np.load(path, allow_pickle=False) as z:
            arms = tuple(str(a) for a in z["arms"])
            t = cls(arms=arms, eid=[str(e) for e in z["eid"]],
                    n_rows=int(z["n_rows"]) if "n_rows" in z.files else 0)
            for a in arms:
                t.stats[a] = {k: list(z[f"{a}__{k}"]) for k in STATS}
        return t


class _ClipPooler:
    """Pooled metrics of a window subset, computed from PER-CLIP sums.

    A cluster-bootstrap draw holds every window of a drawn clip, once per time the
    clip is drawn, so its pooled sums are ``sum_c mult_c * clip_sum_c`` -- exactly
    what summing the resampled windows gives (pinned by a test against
    :func:`pooled`), at ``n_clips`` instead of ``n_windows`` rows per draw."""

    def __init__(self, stats: dict, eid):
        e = np.asarray([str(x) for x in eid])
        self.uniq, inv = np.unique(e, return_inverse=True)
        self.clip_of = inv.astype(np.int64)
        self.n_in_clip = np.bincount(self.clip_of, minlength=len(self.uniq))
        self.clip_sum = {k: np.zeros((len(self.uniq),) + np.asarray(v).shape[1:],
                                     np.int64) for k, v in stats.items()}
        for k, v in stats.items():
            np.add.at(self.clip_sum[k], self.clip_of, np.asarray(v, np.int64))

    def __call__(self, rows) -> dict:
        r = np.asarray(rows, dtype=np.int64)
        cnt = np.bincount(self.clip_of[r], minlength=len(self.uniq))
        mult, rem = np.divmod(cnt, self.n_in_clip)
        if rem.any():
            raise ValueError("a row set that is not whole clips: the per-clip "
                             "shortcut would be wrong, so it refuses")
        tot = {k: np.tensordot(mult, v, axes=(0, 0)) for k, v in self.clip_sum.items()}
        return pooled({k: v[None] for k, v in tot.items()})


def _metric_fn(pooler, metric: str, k: int, b: int):
    """A reducer over WINDOW INDICES computing one pooled metric (the ``ci``
    callable-reducer path)."""
    def red(rows):
        return float(pooler(rows)[metric][k, b])
    red.__name__ = f"pooled_{metric}"
    return red


def summarize(table: WindowTable, *, n_boot: int = ci.DEFAULT_N_BOOT,
              seed: int = 0) -> dict:
    """Every arm x 8 classes x EVERY band: pooled IoU (and P/R/F1 for the thin
    classes) with a clip-cluster bootstrap interval, plus n (windows, clips) and the
    per-band GT cell count (far bands are reported with their n, never dropped)."""
    idx = np.arange(table.n_windows, dtype=np.float64)
    names = table.band_names
    out = {"n_windows": table.n_windows, "n_clips": table.n_clips,
           "tol_m": TOL_M, "bands": list(names), "arms": {}}
    for a in table.arms:
        pooler = _ClipPooler(table.arrays(a), table.eid)
        rec = {}
        n_gt = table.arrays(a)["n_gt"].sum(axis=0)
        for k in range(N_FINE_CLASSES):
            for b in range(len(names)):
                rec[f"{CLASS_NAMES[k]}|{names[b]}|n_gt"] = int(n_gt[k, b])
                for metric in (("iou", "P", "R", "F1") if k in THIN_CLASSES
                               else ("iou",)):
                    key = f"{CLASS_NAMES[k]}|{names[b]}|{metric}"
                    fn = _metric_fn(pooler, metric, k, b)
                    if math.isnan(fn(idx)):
                        rec[key] = {"mean": None, "note": "undefined: the class "
                                    "is absent from GT and prediction in this band"}
                        continue
                    rec[key] = ci.episode_cluster_bootstrap(
                        idx, table.eid, reduce=fn, n_boot=n_boot, seed=seed)
        out["arms"][a] = rec
    return out


def paired_delta(table: WindowTable, arm: str, base: str, metric: str, k: int,
                 b: int, *, n_boot: int = ci.DEFAULT_N_BOOT, seed: int = 0) -> dict:
    """``pooled(arm) - pooled(base)`` for one class/band/metric, paired over clips.

    ⭐ The programme's :func:`ci.paired_episode_cluster_bootstrap`, unchanged. The
    two arms are passed as DISJOINT index ranges (``0..W-1`` and ``W..2W-1``) of one
    concatenated table, so each draw resamples the SAME clips for both and the
    reducer recomputes each arm's pooled ratio from its own rows."""
    A, B = table.arrays(arm), table.arrays(base)
    W = table.n_windows
    both = {k2: np.concatenate([A[k2], B[k2]]) for k2 in STATS}
    # the B half gets its own clip labels, so a draw's two halves stay whole clips
    eid2 = list(table.eid) + [f"{e}|base" for e in table.eid]
    fn = _metric_fn(_ClipPooler(both, eid2), metric, k, b)
    a_idx = np.arange(W, dtype=np.float64)
    r = ci.paired_episode_cluster_bootstrap(a_idx, a_idx + W, table.eid,
                                            n_boot=n_boot, seed=seed, reduce=fn)
    r.update(arm=arm, base=base, metric=metric, cls=CLASS_NAMES[k],
             band=table.band_names[b])
    return r


def evaluate_bars_m7(table: WindowTable, *, arm: str = "refcv7",
                     base: str = "refcv6_38k", prior: str = "prior",
                     n_boot: int = ci.DEFAULT_N_BOOT, seed: int = 0) -> dict:
    """SPEC_REFCV7 §6.2's BAR-M7-1..4 with §11.2's "in EVERY band", plus the
    prior-control clause.

    * M7-1 lane / road line IoU: ``arm - base > 0``, separated, in every band;
    * M7-2 crosswalk IoU: ``arm - base > 0``, separated, in every band;
    * M7-3 non-drivable edge tolerance-F1: ``arm - base > 0``, separated, in every
      band;
    * M7-4 drivable IoU not separated WORSE than ``base`` in any band;
    * every class bar must ALSO beat the positional prior (``arm - prior > 0``,
      separated), in every band. ⛔ A missed bar is FAILED; nothing here moves a
      goalpost. Each band carries its GT n; a band where the class is absent from GT
      AND from every prediction is UNDEFINED -- reported as such with n 0, never
      dropped, and it cannot make a bar PASS (the bar PASSES iff every band with GT
      cells passes and at least one band has them). ⚠️ That undefined-band rule is
      the implementer's, made before any refcv7 number; flagged for the Master Mind.
    ⚠️ The single-seed 2x-replicate-floor rule (§3) needs a replicate floor this
    function does not have; the caller applies it and says so."""
    L, X, E, D = 2, 3, 5, 1
    rows = {}
    names = table.band_names
    for bar, (metric, k) in {"BAR-M7-1": ("iou", L), "BAR-M7-2": ("iou", X),
                             "BAR-M7-3": ("F1", E)}.items():
        n_gt = table.arrays(arm)["n_gt"].sum(axis=0)[k]
        per, verdicts = {}, []
        for b, bn in enumerate(names):
            vb = paired_delta(table, arm, base, metric, k, b, n_boot=n_boot, seed=seed)
            vp = paired_delta(table, arm, prior, metric, k, b, n_boot=n_boot, seed=seed)
            undefined = int(n_gt[b]) == 0
            ok = (vb["separated"] and vb["delta"] > 0 and vp["separated"]
                  and vp["delta"] > 0)
            v = "UNDEFINED" if undefined else ("PASS" if ok else "FAILED")
            per[bn] = {"n_gt_cells": int(n_gt[b]), "vs_base": vb, "vs_prior": vp,
                       "verdict": v}
            verdicts.append(v)
        scored = [v for v in verdicts if v != "UNDEFINED"]
        rows[bar] = {"per_band": per,
                     "verdict": ("PASS" if scored and all(v == "PASS" for v in scored)
                                 else "FAILED")}
    m4 = [paired_delta(table, arm, base, "iou", D, b, n_boot=n_boot, seed=seed)
          for b in range(len(names))]
    worse = [r for r in m4 if r["separated"] and r["delta"] < 0]
    rows["BAR-M7-4"] = {"per_band": m4,
                        "verdict": "FAILED" if worse else "PASS"}
    rows["n_windows"], rows["n_clips"] = table.n_windows, table.n_clips
    return rows


def to_json(obj) -> str:
    """JSON with NaN -> null (the programme's records never carry bare NaN)."""
    def fix(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: fix(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [fix(v) for v in o]
        if isinstance(o, np.generic):
            return fix(o.item())
        if isinstance(o, np.ndarray):
            return fix(o.tolist())
        return o
    return json.dumps(fix(obj), indent=1)
