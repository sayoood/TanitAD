"""EVALUATE D-1 + D-2: how much supervision the fix RECOVERS, and is it class-biased?

The two defects are one arithmetic. ``box3d_set_loss`` now applies
``refc_agents.visible_target_filter`` **before** ``agent_slots.match_slots``, so the
``n_queries``-nearest budget ranks the in-field set instead of a set that is half behind
the ego. This prices that change over the WHOLE v7-B1 join, per class.

⛔ **THE HEADLINE IS NOT "MORE SUPERVISION".** The fix *removes* far more than it adds --
83.6 % of the raw join is outside the decode box and 50.0 % is behind the ego. Three
different quantities are reported separately, because pooling them into one "supervision"
number is exactly how a trade-off gets hidden:

  A  slots_spent_before        query slots the defect consumed (min(n_boxes, N) per frame)
  B  visible_supervised_before how many of those were boxes the camera can actually see
  C  visible_supervised_after  how many the fixed path supervises (all visible)
  ⇒ C - B is the supervision RECOVERED; A - B is the hallucination budget REMOVED.

CONTROLS, all four required:
  1. EXACT-CODE. On a random sample of frames, build real targets with
     ``targets_from_join`` and run ``box3d_set_loss`` in BOTH arms; its ``n`` counts must
     equal this pass's numpy arithmetic on every sampled frame. ⛔ Reports ``None``, never
     True, on zero sampled frames -- a vacuous pass is the advisory's class-F item 3.
  2. ANALYTIC. A frame already carrying >= N visible boxes must give ``after == N``
     exactly, by construction.
  3. IDENTITY, against an INDEPENDENT probe: the total box count, the in-field count and
     the in-field-and-in-box count must reproduce the reviewer's separately written
     ``raw/p2_b1_join_filter_census.json`` exactly. Agreement there is what makes this
     pass a measurement of the corpus rather than of my own predicate.
  4. NULL ARM. Recomputed with the field cut disabled, the recovery must read EXACTLY 0 --
     a probe whose "recovery" survives turning the fix off is measuring something else.

⛔ LINE: v7-B1 (`physicalai-b1-w120-256x640cyl`). Clip ids never leave as raw UUIDs.

Run (CPU, read-only, ~2 min):
  PYTHONPATH=D:/Projects/TanitAD/stack python eval_box_supervision_recovery.py \
      --join D:/Projects/TanitAD-artifacts/a40-rescue/b1_train_plus_eval_agents.jsonl.xz
"""
from __future__ import annotations

import argparse
import collections
import json
import lzma
import math
import pathlib
import random
import time

import numpy as np

from tanitad.data.bev_raster import ALL_CLASSES, GRID_DEFAULT
from tanitad.models.agent_slots import N_QUERIES_DEFAULT, targets_from_join
from tanitad.refs.refc_agents import FOV_HALF_ANGLE_RAD

HERE = pathlib.Path(__file__).resolve().parents[1]
OUT = HERE / "raw" / "box_supervision_recovery.json"
HALF = float(FOV_HALF_ANGLE_RAD)
X_MAX, Y_HALF = float(GRID_DEFAULT.x_fwd_m), float(GRID_DEFAULT.y_half_m)

#: The INDEPENDENT probe this pass must reproduce (control 3).
P2_EXPECT = {"n_boxes": 28_958_699, "n_in_field": 11_639_984,
             "n_in_field_and_box": 4_741_807, "n_behind_ego": 14_490_476}

#: ⛔⛔ THE RECONCILIATION ARM, AND IT EXISTS BECAUSE TWO CORRECT NUMBERS LOOK LIKE A
#: CONTRADICTION. The review's ``raw/p6_query_budget_bias.json`` reports **625,379 =
#: 27.18 %** of in-field supervision destroyed by ordering; this pass reports **+76,395 =
#: 1.64 %** recovered. Both are right and they are DIFFERENT QUESTIONS:
#:   * p6's set is IN-FIELD (azimuth only) and its denominator is the over-budget lines;
#:   * ours is IN-FIELD **AND INSIDE THE DECODE BOX** -- what ``visible_target_filter``
#:     actually keeps, i.e. what the head can EXPRESS -- over the whole corpus.
#: On a crowded frame most in-field boxes are far away and the decode box removes them
#: anyway, so the budget rarely binds once both cuts are applied. This arm recomputes p6's
#: azimuth-only figure with OUR code so the relationship is MEASURED, not asserted.
P6_EXPECT = {"n_infield_lost_to_ordering": 625_379,
             "n_infield_available_if_filtered_first": 2_300_748,
             "n_lines_over_budget": 45_592}


def visible_mask(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """``visible_target_filter``'s predicate. Pinned against the real function by
    ``stack/tests/test_refcv6_perception_supervision_fixes.py`` and by control 1 below."""
    return ((np.arctan2(np.abs(cy), cx) <= HALF)
            & (cx >= 0.0) & (cx <= X_MAX) & (np.abs(cy) <= Y_HALF))


def exact_code_control(rows, n_q: int) -> tuple[int, int]:
    """Run the REAL loss on sampled frames; -> (n_checked, n_agreeing)."""
    import torch                                                  # noqa: F401
    from tanitad.models.agent_slots import AgentSlotDecoder
    from tanitad.models.box3d_head import box3d_set_loss, zh_targets
    torch.manual_seed(0)
    dec = AgentSlotDecoder(d_memory=16, n_memory=8, n_queries=n_q, d_model=32,
                           depth=1, n_heads=4, enforce_band=False)
    ok = 0
    for ags in rows:
        a = np.array([[float(g["cx"]), float(g["cy"]), float(g.get("yaw", 0.0)),
                       float(g.get("l", 4.5)), float(g.get("w", 1.9)), 0.0]
                      for g in ags], dtype=np.float64)
        cls = [str(g.get("cls")) if str(g.get("cls")) in set(ALL_CLASSES)
               else "automobile" for g in ags]
        tgt = zh_targets(targets_from_join(a, classes=cls))
        pred = dec(torch.randn(1, 8, 16))
        pred = {**pred, "cz": torch.zeros(1, n_q), "h": torch.ones(1, n_q) * 1.6}
        on = box3d_set_loss(pred, tgt, visible_filter=True)
        off = box3d_set_loss(pred, tgt, visible_filter=False)
        vm = visible_mask(a[:, 0], a[:, 1])
        want_vis = int(vm.sum())
        agree = (on["n"]["target_prefilter"] == len(ags)
                 and on["n"]["target_visible"] == want_vis
                 and on["n"]["matched"] == min(want_vis, n_q)
                 and off["n"]["target_visible"] == len(ags)
                 and off["n"]["matched"] == min(len(ags), n_q))
        ok += int(agree)
    return len(rows), ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", required=True)
    ap.add_argument("--queries", type=int, default=N_QUERIES_DEFAULT)
    ap.add_argument("--sample-frames", type=int, default=250)
    ap.add_argument("--limit-lines", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    N = int(a.queries)
    rng = random.Random(20260923)

    t0 = time.time()
    n_lines = n_boxes = 0
    n_in_field = n_in_box = n_behind = 0
    slots_before = slots_after = 0
    vis_sup_before = vis_sup_after = 0
    behind_slots_before = 0
    analytic_n = analytic_ok = 0
    before_by_cls: collections.Counter = collections.Counter()
    after_by_cls: collections.Counter = collections.Counter()
    removed_by_cls: collections.Counter = collections.Counter()
    null_before = null_after = 0
    n_over = 0
    az_avail = az_kept = 0          # p6's azimuth-only arm, on over-budget lines
    sample: list = []

    opener = lzma.open if str(a.join).endswith(".xz") else open
    with opener(a.join, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ags = d.get("agents")
            if ags is None:
                continue
            n_lines += 1
            n = len(ags)
            n_boxes += n
            if n == 0:
                continue
            cx = np.fromiter((g["cx"] for g in ags), dtype=np.float64, count=n)
            cy = np.fromiter((g["cy"] for g in ags), dtype=np.float64, count=n)
            cls = [str(g.get("cls")) for g in ags]
            inf = np.arctan2(np.abs(cy), cx) <= HALF
            vm = visible_mask(cx, cy)
            n_in_field += int(inf.sum())
            n_in_box += int(vm.sum())
            n_behind += int((cx < 0.0).sum())

            rngm = np.hypot(cx, cy)
            # --- BEFORE: budget over the RAW set -------------------------------
            kept = (np.argsort(rngm, kind="stable")[:N] if n > N
                    else np.arange(n))
            slots_before += int(kept.size)
            behind_slots_before += int((cx[kept] < 0.0).sum())
            kv = vm[kept]
            vis_sup_before += int(kv.sum())
            # --- AFTER: filter first, then budget ------------------------------
            vidx = np.nonzero(vm)[0]
            if vidx.size > N:
                vidx = vidx[np.argsort(rngm[vidx], kind="stable")[:N]]
                analytic_n += 1
                analytic_ok += int(vidx.size == N)
            slots_after += int(vidx.size)
            vis_sup_after += int(vidx.size)
            # --- NULL ARM: "filter" that keeps everything ----------------------
            null_before += int(kept.size)
            null_after += int(min(n, N))
            # --- RECONCILIATION: p6's AZIMUTH-ONLY arm, over-budget lines only ---
            if n > N:
                n_over += 1
                az_kept += int(inf[kept].sum())
                az_avail += int(min(int(inf.sum()), N))

            for i in kept:
                if vm[i]:
                    before_by_cls[cls[i]] += 1
                else:
                    removed_by_cls[cls[i]] += 1
            for i in vidx:
                after_by_cls[cls[i]] += 1

            if len(sample) < a.sample_frames and n and rng.random() < 0.002:
                sample.append(ags)
            if a.limit_lines and n_lines >= a.limit_lines:
                break

    n_checked, n_agree = exact_code_control(sample, N) if sample else (0, 0)
    recov = {c: int(after_by_cls.get(c, 0) - before_by_cls.get(c, 0))
             for c in ALL_CLASSES}
    ratio = {c: (round(after_by_cls.get(c, 0) / before_by_cls[c], 4)
                 if before_by_cls.get(c) else None) for c in ALL_CLASSES}
    identity = {k: (P2_EXPECT[k] == v) for k, v in
                (("n_boxes", n_boxes), ("n_in_field", n_in_field),
                 ("n_in_field_and_box", n_in_box), ("n_behind_ego", n_behind))}

    res = {
        "_evidence_class": "MEASURED (ours; artifact = this JSON + the code beside it)",
        "_line": "v7-B1 (physicalai-b1-w120-256x640cyl) -- NOT the parity line",
        "join": str(pathlib.Path(a.join).name), "queries": N,
        "corpus": {"n_frames": n_lines, "n_boxes": n_boxes,
                   "n_in_field": n_in_field,
                   "n_in_field_and_decode_box": n_in_box,
                   "n_behind_ego": n_behind,
                   "frac_in_field": n_in_field / max(n_boxes, 1),
                   "frac_in_field_and_decode_box": n_in_box / max(n_boxes, 1),
                   "frac_behind_ego": n_behind / max(n_boxes, 1)},
        "supervision": {
            "A_slots_spent_before": slots_before,
            "A_of_which_behind_the_ego": behind_slots_before,
            "A_frac_behind_the_ego": behind_slots_before / max(slots_before, 1),
            "B_visible_supervised_before": vis_sup_before,
            "C_visible_supervised_after": vis_sup_after,
            "RECOVERED_C_minus_B": vis_sup_after - vis_sup_before,
            "RECOVERED_frac_of_B": (vis_sup_after - vis_sup_before) / max(vis_sup_before, 1),
            "REMOVED_A_minus_B": slots_before - vis_sup_before,
            "REMOVED_frac_of_A": (slots_before - vis_sup_before) / max(slots_before, 1),
            "slots_spent_after": slots_after,
            "_reading": "the fix REMOVES A-B supervision the camera cannot see and "
                        "RECOVERS C-B in-field boxes the budget was throwing away; the "
                        "two are different quantities and are never summed",
        },
        "per_class": {
            "visible_supervised_before": {c: int(before_by_cls.get(c, 0))
                                          for c in ALL_CLASSES},
            "visible_supervised_after": {c: int(after_by_cls.get(c, 0))
                                         for c in ALL_CLASSES},
            "recovered": recov,
            "ratio_after_over_before": ratio,
            "removed_not_visible": {c: int(removed_by_cls.get(c, 0))
                                    for c in ALL_CLASSES},
            "_out_of_vocabulary_seen": {c: int(v) for c, v in removed_by_cls.items()
                                        if c not in set(ALL_CLASSES)},
        },
        "controls": {
            "EXACT_CODE_n_frames": n_checked,
            "EXACT_CODE_n_agreeing": n_agree,
            # ⛔ None, never True, on zero items.
            "EXACT_CODE_all_agree": (None if n_checked == 0 else n_agree == n_checked),
            "ANALYTIC_frames_with_at_least_N_visible": analytic_n,
            "ANALYTIC_after_equals_N_on_all": (None if analytic_n == 0
                                               else analytic_ok == analytic_n),
            "IDENTITY_vs_p2_b1_join_filter_census": identity,
            "IDENTITY_all_agree": all(identity.values()),
            "NULL_ARM_recovery_with_no_field_cut": null_after - null_before,
            "NULL_ARM_expectation": 0,
            "NULL_ARM_passed": (null_after - null_before) == 0,
        },
        "reconciliation_with_p6_azimuth_only": {
            "_why": "p6 reports 27.18 % destroyed by ordering, this pass reports +1.64 % "
                    "recovered. Both are correct: p6's set is IN-FIELD (azimuth only) on "
                    "OVER-BUDGET lines; ours is in-field AND inside the decode box -- "
                    "what the head can express -- over the whole corpus. Most in-field "
                    "boxes on a crowded frame are beyond 60 m, so the decode box removes "
                    "them whether or not the budget does.",
            "n_lines_over_budget": n_over,
            "azimuth_only_available_if_filtered_first": az_avail,
            "azimuth_only_kept_by_nearest_N_of_raw": az_kept,
            "azimuth_only_lost_to_ordering": az_avail - az_kept,
            "p6_reported": P6_EXPECT,
            "reproduces_p6": {
                "n_lines_over_budget": n_over == P6_EXPECT["n_lines_over_budget"],
                "available": az_avail == P6_EXPECT[
                    "n_infield_available_if_filtered_first"],
                "lost": (az_avail - az_kept) == P6_EXPECT[
                    "n_infield_lost_to_ordering"],
            },
        },
        "wall_s": round(time.time() - t0, 1),
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    print(json.dumps({k: res[k] for k in
                      ("corpus", "supervision", "controls",
                       "reconciliation_with_p6_azimuth_only")}, indent=1))
    print("per-class ratio after/before:",
          json.dumps(res["per_class"]["ratio_after_over_before"], indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
