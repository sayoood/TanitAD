"""P6 - the QUERY-BUDGET drop is "keep the NEAREST", and on the refcv6 SS6 box
path the nearest set is half BEHIND THE EGO.

THE MECHANISM (class G: a filter whose semantics encode its OWN author's
precondition). ``agent_slots.match_slots:181-186`` drops the FARTHEST by
``sqrt(cx^2 + cy^2)`` when a frame has more valid targets than queries, and its
docstring declares the policy: *"the near field is the one the plan acts on"*.
That is correct **after** ``refc_agents.visible_target_filter`` has removed what
the camera cannot see -- which is what ``agent_losses:833`` does for the v6
seam. MEASURED in ``p1_box_path_filter.json``: the refcv6 box path
(``refc_v3_train.py:4372``) never calls it, so the ordering key runs over the
RAW join, where (``p2_b1_join_filter_census.json``) **50.04 % of B1 boxes have
cx < 0**.

WHAT THIS PROBE MEASURES, per join line with more boxes than queries:

  n_if_total   in-field boxes on the line
  n_if_kept    in-field boxes among the nearest N by range (what refcv6 gets)
  n_if_avail   in-field boxes among the nearest N of the IN-FIELD set
               (what the v6 path gets: filter first, then budget)

``n_if_avail - n_if_kept`` is in-field supervision destroyed purely by ordering
an unfiltered set. It is a counterfactual on the SAME rows, so it is not an
estimate.

DISCRIMINATING CONTROLS, both required:
  * an ORDER-ONLY arm -- the same N applied to the in-field set -- so a drop
    that is merely "too many boxes" is separated from one that is "the wrong
    boxes";
  * an ANALYTIC arm: lines whose in-field count already exceeds N must show
    ``n_if_avail == N`` exactly, by construction.

Clip ids appear only as sha256(clip_id)[:12].

Run (CPU, read-only):
  PYTHONPATH=D:/Projects/TanitAD/stack python p6_query_budget_bias.py \
      --join D:/Projects/TanitAD-artifacts/a40-rescue/b1_train_plus_eval_agents.jsonl.xz
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import lzma
import math
import pathlib
import time

import numpy as np

from tanitad.data.bev_raster import ALL_CLASSES, GRID_DEFAULT
from tanitad.models.agent_slots import N_QUERIES_DEFAULT, load_cls_class_weight
from tanitad.models import agent_slots as A
from tanitad.refs.refc_agents import FOV_HALF_ANGLE_RAD

HERE = pathlib.Path(__file__).resolve().parents[1]
OUT = HERE / "raw" / "p6_query_budget_bias.json"
HALF = float(FOV_HALF_ANGLE_RAD)
X_MAX, Y_HALF = float(GRID_DEFAULT.x_fwd_m), float(GRID_DEFAULT.y_half_m)


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", required=True)
    ap.add_argument("--queries", type=int, default=N_QUERIES_DEFAULT)
    ap.add_argument("--limit-lines", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    N = int(a.queries)

    t0 = time.time()
    n_lines = n_over = 0
    if_total = if_kept = if_avail = 0
    kept_behind = kept_total = 0
    analytic_ok = analytic_n = 0
    lost_by_cls = collections.Counter()
    kept_by_cls = collections.Counter()
    worst = []            # (n_if_lost, sha12, frame)
    # post-filter class counts, for the class-weight SCOPE check (whole file)
    post_cls = collections.Counter()
    raw_cls = collections.Counter()

    opener = lzma.open if str(a.join).endswith(".xz") else open
    with opener(a.join, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ags = d.get("agents")
            if ags is None:
                continue
            n_lines += 1
            cx = np.fromiter((g["cx"] for g in ags), dtype=np.float64, count=len(ags))
            cy = np.fromiter((g["cy"] for g in ags), dtype=np.float64, count=len(ags))
            cls = [str(g.get("cls")) for g in ags]
            inf = np.arctan2(np.abs(cy), cx) <= HALF
            inbx = (cx >= 0.0) & (cx <= X_MAX) & (np.abs(cy) <= Y_HALF)
            keepmask = inf & inbx
            for i, c in enumerate(cls):
                raw_cls[c] += 1
                if keepmask[i]:
                    post_cls[c] += 1
            n = len(ags)
            if n <= N:
                continue
            n_over += 1
            rng = np.hypot(cx, cy)
            order = np.argsort(rng, kind="stable")[:N]      # match_slots' policy
            k_if = int(inf[order].sum())
            t_if = int(inf.sum())
            # the counterfactual: filter to in-field FIRST, then take N nearest
            av_if = min(t_if, N)
            if_total += t_if
            if_kept += k_if
            if_avail += av_if
            kept_total += len(order)
            kept_behind += int((cx[order] < 0.0).sum())
            if t_if >= N:
                analytic_n += 1
                analytic_ok += int(av_if == N)
            lost = av_if - k_if
            if lost > 0:
                for i in np.nonzero(inf)[0]:
                    if i not in set(order.tolist()):
                        lost_by_cls[cls[i]] += 1
                for i in order:
                    if inf[i]:
                        kept_by_cls[cls[i]] += 1
                worst.append((int(lost), sha12(str(d["clip_id"])), int(d["frame"])))
            if a.limit_lines and n_lines >= a.limit_lines:
                break

    worst.sort(reverse=True)
    # ---- the class-weight SCOPE check (free arithmetic, whole file) -------- #
    try:
        _, st = load_cls_class_weight(A.CLS_WEIGHTS_B1,
                                      expect_corpus_line=A.CORPUS_LINE_B1)
        banked = st["weights"]
    except SystemExit:
        banked = None
    scope = None
    if banked:
        def inv_mean1(counts):
            v = {c: 1.0 / max(counts.get(c, 0), 1) for c in ALL_CLASSES}
            m = sum(v.values()) / len(v)
            return {c: v[c] / m for c in ALL_CLASSES}
        post = inv_mean1(post_cls)
        raw = inv_mean1(raw_cls)
        scope = {
            "banked_vector_counted_on": "the RAW join (every box)",
            "raw_counts": {c: raw_cls.get(c, 0) for c in ALL_CLASSES},
            "post_filter_counts": {c: post_cls.get(c, 0) for c in ALL_CLASSES},
            "banked_weights": banked,
            "reconstructed_from_raw": {c: round(raw[c], 6) for c in ALL_CLASSES},
            "if_counted_post_filter": {c: round(post[c], 6) for c in ALL_CLASSES},
            "ratio_post_over_banked": {
                c: round(post[c] / banked[c], 4) for c in ALL_CLASSES},
            "max_ratio_class": max(
                ((c, round(post[c] / banked[c], 4)) for c in ALL_CLASSES),
                key=lambda kv: abs(math.log(max(kv[1], 1e-9)))),
            "reconstruction_control_max_abs_dev_from_banked": max(
                abs(raw[c] - banked[c]) for c in ALL_CLASSES),
            "control_reading": "reconstruction_control ~0 proves this probe "
                               "reproduces the banked vector's own recipe from "
                               "the raw counts, so the post-filter column is a "
                               "like-for-like comparison and not a different "
                               "normalisation",
        }

    res = {
        "_evidence_class": "MEASURED (ours; artifact = this JSON + the code beside it)",
        "_line": "v7-B1 (b1_train_plus_eval_agents.jsonl.xz) -- NOT the parity line",
        "join": str(a.join), "queries": N,
        "policy": "agent_slots.match_slots:181-186 drops the FARTHEST by "
                  "sqrt(cx^2+cy^2); refcv6's box path applies no field filter "
                  "first (refc_v3_train.py:4372, p1_box_path_filter.json)",
        "lines": {"n_lines": n_lines, "n_lines_over_budget": n_over,
                  "frac_over_budget": n_over / max(n_lines, 1)},
        "on_over_budget_lines": {
            "n_infield_boxes_present": if_total,
            "n_infield_boxes_KEPT_by_nearest_N_of_the_raw_set": if_kept,
            "n_infield_boxes_AVAILABLE_if_filtered_first": if_avail,
            "n_infield_supervision_lost_to_ordering": if_avail - if_kept,
            "frac_of_available_infield_lost":
                (if_avail - if_kept) / max(if_avail, 1),
            "n_query_slots_spent": kept_total,
            "n_query_slots_spent_on_BEHIND_EGO_boxes": kept_behind,
            "frac_query_slots_behind_ego": kept_behind / max(kept_total, 1),
        },
        "controls": {
            "analytic_lines_with_at_least_N_infield": analytic_n,
            # ⛔ N/A, never True, on zero items: the advisory's class-F item 3
            # ("a gate printed [PASS] on a comparison of ZERO items"). The smoke
            # run of this probe hit exactly that and reported a vacuous pass.
            "analytic_avail_equals_N_on_all_of_them": (
                None if analytic_n == 0 else analytic_ok == analytic_n),
            "note": "by construction av_if == N whenever the line already "
                    "carries N in-field boxes; a miss here would be a bug in "
                    "this probe, not a finding",
        },
        "infield_loss_by_class": dict(lost_by_cls.most_common()),
        "infield_kept_by_class": dict(kept_by_cls.most_common()),
        "worst_lines_sha12": [{"n_infield_lost": w[0], "clip_sha12": w[1],
                               "frame": w[2]} for w in worst[:15]],
        "class_weight_scope_check": scope,
        "wall_s": round(time.time() - t0, 1),
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("class_weight_scope_check",
                                   "worst_lines_sha12")}, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
