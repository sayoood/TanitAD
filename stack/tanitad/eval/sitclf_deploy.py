"""Situation classifier — THE DEPLOYED SCORING PATH, with late fusion wired in.

WHY THIS MODULE EXISTS
----------------------
:func:`tanitad.eval.sitclf.late_fuse_scores` was written to fix a MEASURED defect
and then **had no caller**. The reason turned out not to be an oversight about
fusion: *the situation classifier had no deployed scoring path in* ``stack/`` *at
all.* Probed three ways at HEAD —

  1. ``grep -rn 'head_img_ego|head_*.pt|sc_meta|scores.npz'`` over ``stack/`` and
     ``taniteval/`` matches only a docstring in ``sitclf.py``;
  2. the only situation modules under ``stack/`` are the *label* detectors
     (``tanitad/data/situations.py``), the *label* emitter
     (``scripts/emit_situation_labels.py``), ``sitclf.py`` and ``ap_ci.py`` —
     none of which score a frame with a trained head;
  3. the trainers that actually produce the deployed arm live only in the
     research hub (``…/2026-07-26-situation-classifier/scripts/sc_train.py`` and
     ``…/2026-07-29-situation-classifier-v2/sc_train_v2.py``), never promoted.

So ``late_fuse_scores`` had nothing to plug into. This module is the missing
consumer: it takes the score bundle the trainer already emits and produces the
**deployed situation score** from it.

WHAT IT FIXES
-------------
The trainer fuses the modalities at FEATURE level —
``sc_train.py:143`` / ``sc_train_v2.py:143`` both read
``np.concatenate([img, S["E"]], 1)``: a 16-dim PCA image block normalised by its
own global mean-abs (``:130-131``) against a 3-dim ego block divided by a
hand-set ``EGO_SCALE = [10, 2, 0.5]`` (``:38``, applied at ``:93``). The two
normalisations are unrelated, so the modalities enter the shared ``nn.Linear``
at an arbitrary relative scale and the wider block wins. The measured cost is
that the multimodal arm is beaten by its own ego-only ablation.

⛔ The response is **not** to drop the camera (PI ruling 2026-08-03: *"no ego
heads"*). It is to fuse where a scale mismatch cannot exist: each modality is
first reduced to ONE calibrated number, and a 2-parameter logistic combiner —
fitted out-of-fold on whole clip clusters — decides their relative weight from
data instead of from an unrelated pair of normalisation constants.
:func:`fuse_modalities` is that step, and it KEEPS BOTH MODALITIES.

WHAT IT MEASURES
----------------
:func:`four_family_report` reports the binding four families for a
*classification* target, each with the paired episode-cluster bootstrap, and
says per family where a family is genuinely not computable and why.
"""

from __future__ import annotations

import numpy as np

from tanitad.eval.ap_ci import (DEFAULT_N_BOOT, ap_episode_cluster_bootstrap,
                                ap_lift, average_precision,
                                paired_ap_episode_cluster_bootstrap)
from tanitad.eval.sitclf import EGO_SCALE, cluster_folds, late_fuse_scores

#: The arm ``sc_train`` ships as PRIMARY, i.e. what is deployed today.
DEPLOYED_ARM = "head_img_ego"
#: The two unimodal arms the deployed arm is supposed to combine. Fusing THESE
#: rather than swapping to one of them is what keeps the camera in the system.
MODALITY_ARMS: tuple[str, ...] = ("head_img", "head_ego")
#: ``head_img`` with the image features permuted ACROSS clips — the camera's own
#: null. Fusing it in place of ``head_img`` gives a combiner with the *identical*
#: parameter count and fitting protocol, so any gap between the two is the
#: camera's marginal value and nothing else.
NULL_IMAGE_ARM = "head_img_shuf"

# --------------------------------------------------------------------------- #
# ⛔ THE DEPLOYMENT CONTRACT — PI ruling 2026-08-03                            #
#                                                                             #
#   "for ground truth data of scenario classification you can use both ego and #
#    other label, for inference only vision."                                  #
#                                                                             #
# Ego (and anything else) may DERIVE the labels. At INFERENCE the classifier   #
# sees VISION ONLY. That closes `head_ego`, `head_img_ego`, AND image+ego late #
# fusion alike — a calibrated ego SCORE at inference is still an ego input.    #
# `late_fuse_scores` keeps a role only BETWEEN VISION ARMS.                    #
# --------------------------------------------------------------------------- #
#: arms whose only inference input is the camera
VISION_ARMS: tuple[str, ...] = ("head_img", "ridge_img")
#: the same arms with the image features permuted ACROSS clips — the camera's
#: own null, and the ONLY admissible baseline for "does vision work?" now that
#: the ego arms are not legal inputs (see RETRACTION_LOG R-2026-08-03-f).
VISION_NULL_ARMS: tuple[str, ...] = ("head_img_shuf", "ridge_img_shuf")


def is_vision_only(arm: str) -> bool:
    """Would this arm read an ego channel at inference time?

    A guard rather than a comment: the ruling is easy to violate by adding an
    arm whose name merely *looks* visual, and the whole point of the vision-only
    panel is that nothing in it touches ego.
    """
    a = str(arm)
    return "ego" not in a and "priv" not in a


def permute_labels_by_cluster(y, clip_cluster, seed: int = 0) -> np.ndarray:
    """Each cluster's labels replaced by another cluster's — the label null.

    Permuting whole CLUSTERS rather than rows preserves the within-clip
    correlation the cluster estimator assumes; a row-wise shuffle would destroy
    it and make the control easier than the real task.
    """
    y = np.asarray(y).ravel()
    cc = np.asarray(clip_cluster).ravel()
    rng = np.random.default_rng(seed)
    uniq = np.unique(cc)
    src = {int(u): int(v) for u, v in zip(uniq, rng.permutation(uniq))}
    out = np.zeros(y.size, np.int64)
    for u in uniq:
        dst = np.flatnonzero(cc == u)
        s = np.flatnonzero(cc == src[int(u)])
        out[dst] = y[s[np.arange(dst.size) % s.size]]
    return out


def vision_only_arms(bundle: "ScoreBundle", situation: str, *,
                     n_folds: int = 2, seed: int = 0) -> dict:
    """The VISION-ONLY panel for one situation. Nothing here reads ego.

    ``PRIMARY`` is the deployable arm; ``FUSED`` is ``late_fuse_scores`` applied
    between two VISION arms; the three ``NEG_*`` arms are the controls that make
    the panel readable — the camera's own null, the same combiner on that null,
    and the combiner fitted on permuted labels.
    """
    i = bundle.col(situation)
    have = [a for a in VISION_ARMS if a in bundle.scores]
    if not have:
        raise KeyError(f"bundle has no vision arm from {VISION_ARMS}: "
                       f"{sorted(bundle.scores)}")
    bad = [a for a in have if not is_vision_only(a)]
    if bad:
        raise ValueError(f"arms {bad} read ego at inference — forbidden by the "
                         f"2026-08-03 ruling")
    nulls = [a for a in VISION_NULL_ARMS if a in bundle.scores]
    arms = {
        "PRIMARY": bundle.arm(have[0], situation).astype(np.float64),
        "NEG_MACHINERY": fuse_modalities(bundle, situation, arms=(have[0],),
                                         n_folds=n_folds, seed=seed),
    }
    if len(have) > 1:
        arms["FUSED"] = fuse_modalities(bundle, situation, arms=tuple(have),
                                        n_folds=n_folds, seed=seed)
    if nulls:
        arms["NEG_VISION"] = bundle.arm(nulls[0], situation).astype(np.float64)
        if len(nulls) > 1:
            arms["NEG_FUSED"] = fuse_modalities(bundle, situation,
                                                arms=tuple(nulls),
                                                n_folds=n_folds, seed=seed)
    y_perm = permute_labels_by_cluster(bundle.y[:, i], bundle.clip_cluster,
                                       seed=seed + 991)
    cols = np.stack([bundle.arm(a, situation) for a in have], 1)
    folds = cluster_folds(bundle.clip_cluster, n_folds=n_folds, seed=seed)
    arms["NEG_LABEL"] = late_fuse_scores(cols, y_perm, bundle.valid[:, i], folds)
    return arms


# --------------------------------------------------------------------------- #
# the bundle                                                                  #
# --------------------------------------------------------------------------- #
class ScoreBundle:
    """The per-frame held-out scores ``sc_train`` emits, loaded and checked.

    Deliberately a thin object over the ``.npz``: the bundle is the trainer's
    output contract, and re-deriving anything here would let this module drift
    away from what was actually scored.
    """

    def __init__(self, situations, arms, y, valid, clip_cluster, scores, ego=None,
                 source: str = "<memory>"):
        self.situations = tuple(str(s) for s in situations)
        self.arms = tuple(str(a) for a in arms)
        self.y = np.asarray(y).astype(np.int64)
        self.valid = np.asarray(valid).astype(bool)
        self.clip_cluster = np.asarray(clip_cluster).ravel()
        self.scores = {str(k): np.asarray(v, dtype=np.float64) for k, v in scores.items()}
        self.ego = None if ego is None else np.asarray(ego, dtype=np.float64)
        self.source = str(source)
        n, k = self.y.shape
        if self.valid.shape != (n, k):
            raise ValueError(f"valid {self.valid.shape} != y {self.y.shape}")
        if k != len(self.situations):
            raise ValueError(f"{k} label columns vs {len(self.situations)} situations")
        if self.clip_cluster.size != n:
            raise ValueError(f"clip_cluster {self.clip_cluster.size} != {n} rows")
        for a, s in self.scores.items():
            if s.shape != (n, k):
                raise ValueError(f"arm {a!r} has shape {s.shape}, expected {(n, k)}")
        if self.ego is not None and self.ego.shape[0] != n:
            raise ValueError(f"ego {self.ego.shape} does not have {n} rows")

    @property
    def n_rows(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.clip_cluster).size)

    def col(self, situation: str) -> int:
        if situation not in self.situations:
            raise KeyError(f"{situation!r} not in {self.situations}")
        return self.situations.index(situation)

    def arm(self, name: str, situation: str) -> np.ndarray:
        """One arm's score column for one situation."""
        if name not in self.scores:
            raise KeyError(f"arm {name!r} not in bundle {sorted(self.scores)}")
        return self.scores[name][:, self.col(situation)]


def load_score_bundle(path) -> ScoreBundle:
    """Load the trainer's ``.npz`` of per-frame held-out scores.

    Accepts both the gen-1 held-out dump (``situations``/``arms``/``y``/
    ``valid``/``clip_cluster``/``ego`` + one array per arm) and any dump with
    those keys, so the same consumer serves gen-1 and v2 without a fork.
    """
    z = np.load(path, allow_pickle=False)
    need = ("situations", "arms", "y", "valid", "clip_cluster")
    missing = [k for k in need if k not in z.files]
    if missing:
        raise KeyError(f"{path}: score bundle is missing {missing}; has {sorted(z.files)}")
    arms = [str(a) for a in z["arms"]]
    scores = {a: z[a] for a in arms if a in z.files}
    return ScoreBundle(situations=z["situations"], arms=arms, y=z["y"], valid=z["valid"],
                       clip_cluster=z["clip_cluster"], scores=scores,
                       ego=z["ego"] if "ego" in z.files else None, source=str(path))


# --------------------------------------------------------------------------- #
# THE FIX — score-level fusion, both modalities kept                          #
# --------------------------------------------------------------------------- #
def fuse_modalities(bundle: ScoreBundle, situation: str, *,
                    arms: tuple[str, ...] = MODALITY_ARMS,
                    n_folds: int = 2, seed: int = 0, l2: float = 1.0) -> np.ndarray:
    """The DEPLOYED score for ``situation``: late fusion of ``arms``.

    This is the call ``late_fuse_scores`` was written for and never received.
    Each modality contributes exactly one column, so the 5.3 : 1 dimensional
    imbalance that lets the image block swamp the ego block at
    ``sc_train.py:143`` cannot arise — the combiner has ``len(arms)`` weights and
    learns the relative scale rather than inheriting it from two unrelated
    normalisation constants.

    Folds are whole clip CLUSTERS, so the combiner that scores a row never saw
    any frame of that row's clip. Rows outside the situation's validity mask come
    back ``-inf`` (never a confident negative).
    """
    i = bundle.col(situation)
    cols = np.stack([bundle.arm(a, situation) for a in arms], 1)
    folds = cluster_folds(bundle.clip_cluster, n_folds=n_folds, seed=seed)
    return late_fuse_scores(cols, bundle.y[:, i], bundle.valid[:, i], folds, l2=l2)


# --------------------------------------------------------------------------- #
# regime strata — how a classification target carries a lat/lon family        #
# --------------------------------------------------------------------------- #
def regime_strata(ego, *, ego_scale=EGO_SCALE) -> dict[str, dict[str, np.ndarray]]:
    """Longitudinal and lateral REGIME masks from the ego channels.

    ``ego`` is the trainer's ``[v, a_lon_pre, omega_pre] / EGO_SCALE`` block
    (``sc_train.py:93``), so it is undone here before thresholding in physical
    units. MEASURED on the gen-1 bundle: v median 10.6 m/s, a_lon p05/p95
    -1.32/+1.27 m/s², omega p05/p95 -0.122/+0.133 rad/s.

    ⚠️ A classification target has no predicted trajectory, so the binding
    LONGITUDINAL family ("target-speed accuracy, distance-keeping") and LATERAL
    family ("heading, curvature, yaw-rate, cross-track") have no direct form
    here. What IS answerable — and is the question those families exist to
    force — is whether decision quality **holds across the regimes**, or whether
    an arm wins overall while failing exactly where the ego is decelerating or
    turning. These strata are that test, and the mapping is stated rather than
    implied.
    """
    e = np.asarray(ego, dtype=np.float64)
    if e.ndim != 2 or e.shape[1] < 3:
        raise ValueError(f"ego must be [N, >=3], got {e.shape}")
    raw = e[:, :3] * np.asarray(ego_scale, dtype=np.float64)
    v, alon, omega = raw[:, 0], raw[:, 1], raw[:, 2]
    return {
        "longitudinal": {
            "decelerating": alon <= -0.5,
            "steady": np.abs(alon) < 0.5,
            "accelerating": alon >= 0.5,
            "low_speed_lt8": v < 8.0,
            "cruise_ge8": v >= 8.0,
        },
        "lateral": {
            "straight": np.abs(omega) < 0.05,
            "turning": np.abs(omega) >= 0.05,
        },
    }


# --------------------------------------------------------------------------- #
# operating-point statistics (the TACTICAL family beyond a ranking metric)     #
# --------------------------------------------------------------------------- #
def _top_frac_alarm(s, fin, top_frac: float) -> np.ndarray:
    """Boolean alarm mask holding the top ``top_frac`` of finite rows BY RANK.

    Rank, not a value threshold: the scores are heavily tied (a sigmoid output
    saturates, and a fused score is constant on rows a fold could not fit), and
    ``s >= quantile`` then admits every tied row at once — which silently turns a
    5 % operating point into a 100 % one and makes every arm look maximally
    early. Ranking fixes the alarm COUNT, so two arms are compared at the same
    alarm budget whatever their score distribution looks like.
    """
    idx = np.flatnonzero(fin)
    alarm = np.zeros(s.size, bool)
    if idx.size == 0:
        return alarm
    k = max(1, int(round(top_frac * idx.size)))
    order = np.argsort(-s[idx], kind="mergesort")[:k]
    alarm[idx[order]] = True
    return alarm


def precision_recall_at_budget(y, s, valid, *, top_frac: float = 0.05) -> dict:
    """PRECISION and recall at a fixed alarm budget, with both denominators.

    ⚠️ Binding rule (2026-08-03): *"report precision alongside recall — a
    recall-only frontier cannot see what it is paying."* That is not abstract
    here: the same programme published a "brake_stop 0.026 -> 0.503, free win"
    claim that was **retracted the same week** once precision was attached
    (0.2340 -> 0.1711, 380 fires for 153 true positives). AP alone cannot expose
    it either, because AP integrates over every operating point and the deployed
    system runs at ONE.

    Both denominators are returned explicitly — ``n_alarm`` (what precision
    divides by) and ``n_pos`` (what recall divides by) — so a gain that lives on
    a shrinking denominator is visible in the row rather than inferred.
    """
    y = np.asarray(y).astype(bool).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    valid = np.asarray(valid).astype(bool).ravel()
    if y.shape != s.shape or y.shape != valid.shape:
        raise ValueError(f"aligned [N] inputs required: {y.shape} {s.shape} {valid.shape}")
    fin = valid & np.isfinite(s)
    alarm = _top_frac_alarm(s, fin, top_frac)
    n_alarm = int(alarm.sum())
    n_pos = int((y & fin).sum())
    tp = int((alarm & y).sum())
    base = float(y[fin].mean()) if fin.any() else float("nan")
    prec = (tp / n_alarm) if n_alarm else float("nan")
    return {"top_frac": top_frac, "n_scorable": int(fin.sum()),
            "n_alarm": n_alarm, "n_pos": n_pos, "tp": tp,
            "precision": round(prec, 5),
            "recall": round(tp / n_pos, 5) if n_pos else float("nan"),
            "base_rate": round(base, 6),
            "precision_lift": (round(prec / base, 5)
                               if n_alarm and base > 0 else float("nan"))}


def anticipation_lead_s(y, s, clip_cluster, valid, *, hz: float = 10.0,
                        top_frac: float = 0.05) -> dict:
    """Median seconds by which the first alarm precedes the onset it anticipates.

    ``y`` is the ANTICIPATION target, so a contiguous positive run is exactly the
    ``lead_s`` frames before one onset and the onset is the frame after the run
    ends. The lead is therefore measured INSIDE the run — from the run's first
    alarm to that onset — which bounds it by the label's own horizon and makes it
    mean "how early within the window it was willing to commit".

    A run with no alarm at all contributes no lead and is counted in
    ``n_runs_no_alarm`` rather than scored as 0 s, which would reward an arm that
    never fires with a perfect-looking punctuality.
    """
    y = np.asarray(y).astype(bool).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    valid = np.asarray(valid).astype(bool).ravel()
    cc = np.asarray(clip_cluster).ravel()
    fin = valid & np.isfinite(s)
    if fin.sum() == 0 or not y[fin].any():
        return {"median_lead_s": None, "n_runs": 0, "n_runs_no_alarm": 0,
                "top_frac": top_frac}
    alarm = _top_frac_alarm(s, fin, top_frac)
    leads, no_alarm = [], 0
    for c in np.unique(cc):
        rows = np.flatnonzero(cc == c)
        yy = y[rows]
        if not yy.any():
            continue
        d = np.diff(np.concatenate([[0], yy.astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
            hit = np.flatnonzero(alarm[rows[a:b + 1]])
            if hit.size == 0:
                no_alarm += 1
            else:                       # onset is the frame AFTER the run
                leads.append((b + 1 - (a + int(hit[0]))) / hz)
    return {"median_lead_s": (round(float(np.median(leads)), 3) if leads else None),
            "n_runs": len(leads) + no_alarm, "n_runs_no_alarm": no_alarm,
            "n_alarm_rows": int(alarm.sum()), "top_frac": top_frac}


# --------------------------------------------------------------------------- #
# the report                                                                  #
# --------------------------------------------------------------------------- #
def _paired(y, a, b, eid, *, n_boot, seed):
    return paired_ap_episode_cluster_bootstrap(y, a, b, eid, n_boot=n_boot,
                                               seed=seed, lift=True)


def _finite_pair(a, b, mask):
    """Rows where BOTH arms are finite — the paired estimator's precondition."""
    return mask & np.isfinite(a) & np.isfinite(b)


def four_family_report(bundle: ScoreBundle, situation: str, *,
                       fused: np.ndarray, baseline: np.ndarray,
                       baseline_name: str = DEPLOYED_ARM,
                       fused_name: str = "late_fuse(head_img, head_ego)",
                       n_boot: int = DEFAULT_N_BOOT, seed: int = 0,
                       strata_n_boot: int | None = None) -> dict:
    """The binding four families for ONE situation, fused vs baseline.

    Per CLAUDE.md the families are reported SEPARATELY and never pooled, each
    with the paired episode-cluster bootstrap, and a family that cannot be
    computed is reported with its reason and its ``n`` rather than dropped.
    """
    i = bundle.col(situation)
    y = bundle.y[:, i]
    m0 = bundle.valid[:, i]
    m = _finite_pair(fused, baseline, m0)
    eid = bundle.clip_cluster
    sb = strata_n_boot if strata_n_boot is not None else n_boot

    tactical = {
        "_what": ("situation anticipation = the tactical decision. AP and AP-lift "
                  "per situation; the paired delta is the decision-quality change."),
        "n_rows_scored": int(m.sum()),
        "n_rows_dropped_nonfinite": int(m0.sum() - m.sum()),
        "base_rate": round(float(y[m].mean()), 6),
        "n_pos": int(y[m].sum()),
        "ap": {fused_name: round(average_precision(y[m], fused[m]), 5),
               baseline_name: round(average_precision(y[m], baseline[m]), 5)},
        "ap_lift": {fused_name: round(ap_lift(y[m], fused[m]), 5),
                    baseline_name: round(ap_lift(y[m], baseline[m]), 5)},
        "paired_delta_ap_lift": _paired(y[m], fused[m], baseline[m], eid[m],
                                        n_boot=n_boot, seed=seed),
        "single_arm_ci": {
            fused_name: ap_episode_cluster_bootstrap(y[m], fused[m], eid[m],
                                                     n_boot=n_boot, seed=seed, lift=True),
            baseline_name: ap_episode_cluster_bootstrap(y[m], baseline[m], eid[m],
                                                        n_boot=n_boot, seed=seed, lift=True),
        },
        "anticipation_lead": {
            fused_name: anticipation_lead_s(y, fused, eid, m),
            baseline_name: anticipation_lead_s(y, baseline, eid, m),
        },
        # ⚠️ the operating point, WITH precision. AP is a ranking average over
        # every threshold; the deployed system runs at one, and a recall-only
        # read of that one point is how the retracted "free win" was published.
        "operating_point_5pct": {
            fused_name: precision_recall_at_budget(y, fused, m),
            baseline_name: precision_recall_at_budget(y, baseline, m),
        },
    }

    fams: dict = {"TACTICAL": tactical}
    if bundle.ego is None:
        for fam in ("LONGITUDINAL", "LATERAL"):
            fams[fam] = {"_status": "UNAVAILABLE",
                         "_reason": "bundle carries no ego channels; no regime can be defined",
                         "n_rows": int(m.sum())}
    else:
        strata = regime_strata(bundle.ego)
        for fam, key in (("LONGITUDINAL", "longitudinal"), ("LATERAL", "lateral")):
            out = {"_what": ("a classification target has no predicted speed/headway "
                             "(LONGITUDINAL) or heading/curvature/cross-track (LATERAL) "
                             "to score, so the family is reported as decision quality "
                             "STRATIFIED BY REGIME: does the change hold where the ego "
                             "is actually decelerating / turning?"),
                   "_not_computable": ("target-speed accuracy, headway/time-gap/TTC"
                                       if fam == "LONGITUDINAL" else
                                       "heading, curvature, yaw-rate, cross-track error"),
                   "_not_computable_reason": ("the arm emits a per-frame situation "
                                              "probability, not a trajectory"),
                   "strata": {}}
            for nm, sel in strata[key].items():
                ms = m & sel
                npos = int(y[ms].sum())
                if npos < 20 or ms.sum() < 200:
                    out["strata"][nm] = {"_status": "UNPOWERED", "n_rows": int(ms.sum()),
                                         "n_pos": npos}
                    continue
                out["strata"][nm] = {
                    "n_rows": int(ms.sum()), "n_pos": npos,
                    "base_rate": round(float(y[ms].mean()), 6),
                    "ap_lift": {fused_name: round(ap_lift(y[ms], fused[ms]), 5),
                                baseline_name: round(ap_lift(y[ms], baseline[ms]), 5)},
                    "paired_delta_ap_lift": _paired(y[ms], fused[ms], baseline[ms],
                                                    eid[ms], n_boot=sb, seed=seed),
                }
            fams[fam] = out

    fams["STRATEGIC"] = {
        "_status": "UNAVAILABLE",
        "_reason": ("no route/goal/map label exists on PhysicalAI-AV — settled at five "
                    "probes (no map, lane graph, junction annotation or route signal; "
                    "egomotion is clip-local metres with no GNSS), so there is no "
                    "strategic target to score this arm against."),
        "n_rows": int(m.sum()),
    }
    return {"situation": situation, "families": fams,
            "estimator": "paired_ap_episode_cluster_bootstrap (episode-cluster, paired)",
            "n_clusters": int(np.unique(eid[m]).size), "n_boot": int(n_boot)}
