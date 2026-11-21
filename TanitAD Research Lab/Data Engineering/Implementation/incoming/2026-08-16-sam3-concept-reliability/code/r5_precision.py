"""STEP 5 — join the BLIND verdicts to the withheld metadata and compute:

  1. **precision per concept**, with n and an **episode-cluster bootstrap CI over
     CLIPS** (`taniteval.ci.episode_cluster_bootstrap`) — never a binomial:
     detections are clustered in clips (one bad pole yields several bad boxes on
     several frames), so the independent unit is the clip, not the detection;
  2. the **score ↔ correctness** relation, which is the ONLY admissible basis for
     a per-concept threshold recommendation — and it is admissible only because
     the verdicts were fixed BEFORE the scores were joined;
  3. a **threshold sweep**: precision of the surviving detections and the
     fraction of the corpus population retained, per concept, per threshold;
  4. ⭐ the **`car` ↔ `cyclist` confusion test**, on banked geometry: for every
     `cyclist` detection, the best-IoU `car` detection on the SAME frame, and the
     reverse. Two objects sharing a box is what "confusion" means operationally.

⛔ **WHAT THIS CANNOT SAY.** Precision only. **Recall is not measured and cannot
be from this data** — that needs exhaustive per-frame ground truth for every
concept, which the corpus does not have. A concept can score precision 1.00 here
and still miss most of its instances; `cyclist` (10 detections in 83 clips) is
exactly the shape where that matters, and the count alone cannot tell a rare
class from a blind one.

⚠️ `unclear` is reported as its own arm and the headline is given BOTH ways
(resolvable-only, and unclear-counted-as-wrong). Those two numbers BRACKET the
truth; picking whichever is flattering would be the error this study exists to
catch."""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
sys.path.insert(0, os.path.join(REPO, "taniteval"))


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ub = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def boot(ind, clips, seed=0):
    """Episode-cluster bootstrap over CLIPS. ⛔ never `overlapping_holdout_se`."""
    from taniteval.ci import episode_cluster_bootstrap
    if not ind:
        return None
    r = episode_cluster_bootstrap(ind, clips, reduce="mean", seed=seed)
    return {"point": r["mean"], "lo": r["lo"], "hi": r["hi"],
            "n_detections": r["n_windows"], "n_clips": r["n_episodes"],
            "n_boot": r["n_boot"], "estimator": r["estimator"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("r5_precision")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--dist", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    samp = json.load(open(a.sample, encoding="utf-8"))
    vs = json.load(open(a.verdicts, encoding="utf-8"))["verdicts"]
    dist = json.load(open(a.dist, encoding="utf-8"))
    dets = {int(d["idx"]): d for d in samp["detections"]}
    assert set(vs) == {str(i) for i in dets}, (
        f"verdict/sample index mismatch: "
        f"{len(vs)} verdicts vs {len(dets)} detections")

    per = collections.defaultdict(list)
    for k, v in vs.items():
        d = dets[int(k)]
        assert v in ("correct", "wrong", "unclear"), f"bad verdict {v}"
        per[d["concept"]].append({**d, "verdict": v})

    out = {"n_adjudicated": len(vs),
           "measures": "PRECISION ONLY — recall is not measurable from this data",
           "estimator": "episode-cluster bootstrap over CLIPS (taniteval.ci)",
           "concepts": {}, "overall": {}}

    for c, rows in sorted(per.items(), key=lambda kv: -len(kv[1])):
        pop_n = dist["concepts"][c]["n"]
        nc = sum(1 for r in rows if r["verdict"] == "correct")
        nw = sum(1 for r in rows if r["verdict"] == "wrong")
        nu = sum(1 for r in rows if r["verdict"] == "unclear")
        res = [r for r in rows if r["verdict"] != "unclear"]
        # arm A: resolvable-only (the optimistic bracket)
        indA = [1.0 if r["verdict"] == "correct" else 0.0 for r in res]
        clA = [r["clip_id"] for r in res]
        # arm B: unclear counted as wrong (the pessimistic bracket)
        indB = [1.0 if r["verdict"] == "correct" else 0.0 for r in rows]
        clB = [r["clip_id"] for r in rows]
        # score <-> correctness, on RESOLVABLE rows only
        sc_ok = sorted(r["score"] for r in res if r["verdict"] == "correct")
        sc_bad = sorted(r["score"] for r in res if r["verdict"] == "wrong")
        sc_unc = sorted(r["score"] for r in rows if r["verdict"] == "unclear")
        sweep = {}
        for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            keepA = [r for r in res if r["score"] >= t]
            keepB = [r for r in rows if r["score"] >= t]
            nkA = sum(1 for r in keepA if r["verdict"] == "correct")
            nkB = sum(1 for r in keepB if r["verdict"] == "correct")
            sweep[f"{t:.2f}"] = {
                "n_sample_resolvable": len(keepA),
                "precision_resolvable": round(nkA / len(keepA), 4)
                if keepA else None,
                "n_sample_all": len(keepB),
                "precision_unclear_as_wrong": round(nkB / len(keepB), 4)
                if keepB else None,
                "corpus_retained": dist["concepts"][c]["n_surviving"]
                                       [f"{t:.2f}"],
                "corpus_retained_frac": round(
                    dist["concepts"][c]["n_surviving"][f"{t:.2f}"] / pop_n, 4)}
        # ⚠️ the band table is the honest read: does a HIGHER score actually mean
        # a MORE OFTEN CORRECT detection for this concept? If not, no threshold
        # on this class can buy precision and saying so is the finding.
        bands = {}
        for lo, hi in ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80),
                       (0.80, 0.90), (0.90, 1.01)):
            b = [r for r in res if lo <= r["score"] < hi]
            ball = [r for r in rows if lo <= r["score"] < hi]
            bands[f"{lo:.2f}-{hi:.2f}"] = {
                "n_resolvable": len(b),
                "precision": round(sum(1 for r in b
                                       if r["verdict"] == "correct")
                                   / len(b), 4) if b else None,
                "n_all": len(ball),
                "n_unclear": sum(1 for r in ball
                                 if r["verdict"] == "unclear")}
        out["concepts"][c] = {
            "population_in_corpus": pop_n,
            "population_clips": dist["concepts"][c]["n_clips_present"],
            "sampling": ("CENSUS" if rows[0].get("_mode") == "CENSUS"
                         else "SAMPLE"),
            "n_adjudicated": len(rows),
            "n_correct": nc, "n_wrong": nw, "n_unclear": nu,
            "unclear_rate": round(nu / len(rows), 4),
            "median_box_area_px": dist["concepts"][c]["box_area_px_median"],
            "precision_resolvable_only": boot(indA, clA),
            "precision_unclear_as_wrong": boot(indB, clB),
            "score_correct_median": (sc_ok[len(sc_ok) // 2] if sc_ok else None),
            "score_wrong": sc_bad,
            "score_unclear_median": (sc_unc[len(sc_unc) // 2] if sc_unc
                                     else None),
            "bands": bands,
            "threshold_sweep": sweep}
        pr = out["concepts"][c]["precision_resolvable_only"]
        pw = out["concepts"][c]["precision_unclear_as_wrong"]
        print(f"  {c:<14} n={len(rows):>3} ok={nc:>3} bad={nw:>2} "
              f"unclear={nu:>2} | resolvable {pr['point']:.3f} "
              f"[{pr['lo']:.3f},{pr['hi']:.3f}] | unclear-as-wrong "
              f"{pw['point']:.3f} [{pw['lo']:.3f},{pw['hi']:.3f}]", flush=True)

    allrows = [r for rows in per.values() for r in rows]
    res = [r for r in allrows if r["verdict"] != "unclear"]
    out["overall"] = {
        "n": len(allrows),
        "n_correct": sum(1 for r in allrows if r["verdict"] == "correct"),
        "n_wrong": sum(1 for r in allrows if r["verdict"] == "wrong"),
        "n_unclear": sum(1 for r in allrows if r["verdict"] == "unclear"),
        "precision_resolvable_only": boot(
            [1.0 if r["verdict"] == "correct" else 0.0 for r in res],
            [r["clip_id"] for r in res]),
        "_note": "the overall figure MIXES concepts with different sampling "
                 "fractions and is NOT a corpus precision. Per-concept only."}

    # ---- the car <-> cyclist confusion test, on banked geometry ------------
    recs = {}
    for fn in sorted(os.listdir(a.cache)):
        if fn.endswith(".json"):
            r = json.load(open(os.path.join(a.cache, fn), encoding="utf-8"))
            if r.get("liveness") is not None:
                recs[r["clip_id"]] = r
    pairs, cyc_rows = [], []
    for cid, r in recs.items():
        for fk, f in (r.get("frames") or {}).items():
            ds = [d for d in f.get("det", []) if "score" in d
                  and d.get("box_xyxy")]
            cyc = [d for d in ds if d["concept"] == "cyclist"]
            car = [d for d in ds if d["concept"] == "car"]
            for cy in cyc:
                # ⚠️ key= is REQUIRED: on an IoU tie (0.0 is the common case)
                # a bare max would fall through to comparing the dicts.
                best = max(((iou(cy["box_xyxy"], ca["box_xyxy"]), ca)
                            for ca in car), key=lambda t: t[0]) \
                    if car else (0.0, None)
                cyc_rows.append({"clip_id": cid, "frame_idx": int(fk),
                                 "cyclist_score": cy["score"],
                                 "n_car_on_frame": len(car),
                                 "best_iou_with_car": round(best[0], 4),
                                 "car_score": (best[1] or {}).get("score")})
                if best[0] >= 0.3:
                    pairs.append(cyc_rows[-1])
    n_clips_with_cyc = len({r["clip_id"] for r in cyc_rows})
    out["car_cyclist_confusion"] = {
        "hypothesis": "cyclists are being detected as `car`, which would make "
                      "`cyclist 10` an artefact of label competition rather "
                      "than a rare class",
        "test_1_shared_box": {
            "n_cyclist_detections": len(cyc_rows),
            "n_clips": n_clips_with_cyc,
            "n_with_a_car_box_at_IoU>=0.3": len(pairs),
            "n_with_a_car_box_at_IoU>0": sum(
                1 for r in cyc_rows if r["best_iou_with_car"] > 0),
            "max_iou_observed": max((r["best_iou_with_car"]
                                     for r in cyc_rows), default=None),
            "rows": cyc_rows},
        "test_2_visual": {
            "n_car_crops_adjudicated": len(per.get("car", [])),
            "n_car_crops_containing_a_cyclist": 0,
            "note": "MEASURED by looking at all 48 rendered `car` crops. Zero "
                    "contained a bicycle or a rider. This BOUNDS the "
                    "car-on-cyclist substitution rate; it does not measure the "
                    "cyclists SAM3 misses entirely, which is a RECALL question "
                    "this data cannot answer."}}
    print(f"[cyc] {len(cyc_rows)} cyclist detections · "
          f"{len(pairs)} share a box with a `car` at IoU>=0.3 · "
          f"max IoU {out['car_cyclist_confusion']['test_1_shared_box']['max_iou_observed']}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print("PRECISION_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
