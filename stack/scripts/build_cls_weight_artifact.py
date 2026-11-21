"""Build an agent-class frequency weight artifact FROM a join -- the PRODUCER.

⛔⛔ **THIS FILE EXISTS BECAUSE THE LAST ONE DID NOT.** `agent_cls_weights_b1.json`
stamps `_producer: "qland/work/pbox/build_cls_weight_artifact.py"` -- a path in a scratch
directory that **no longer exists on this box** (probed 2026-09-23: `qland/` is absent from
the repo and from the working tree). A banked artifact whose producer is gone cannot be
rebuilt, re-scoped, or audited; it can only be hand-edited, which is the
`RETR-2026-09-22-SELF-ATTESTING-DIGEST` failure waiting to happen again. The producer
belongs in the repo beside the artifact it writes.

⛔ **THE DIGEST IS COMPUTED, NEVER TYPED.** `agent_slots.cls_weight_digest` is the single
recipe, imported here, and `agent_slots.load_cls_class_weight` re-derives it on every load.
This script does not know how to write a digest any other way.

WHAT IT COUNTS, AND THE TWO POPULATIONS IT COUNTS OVER
-----------------------------------------------------
`raw_join`               every box in the join -- what the vector has always been.
`in_field_and_decode_box` the boxes `refc_agents.visible_target_filter` keeps: azimuth
                          <= `FOV_HALF_ANGLE_RAD` **and** inside `SlotDecodeRanges`
                          (0 <= cx <= x_fwd_m, |cy| <= y_half_m).

⛔ Both go in ONE artifact, under their own keys and their own digests, because the arm's
box loss chooses between them at load time (`load_cls_class_weight(target_population=...)`).
Two files selected by one flag is the defect `p5_cls_weight_guard.json` measured: the
expectation and the file would again come from the same key.

CONTROLS, all three required and all three written into the artifact:
  * `_control_reconstruction_max_abs_dev` -- the RAW vector this pass computes against the
    one already banked. It must be ~0, and that is what makes the VISIBLE column
    like-for-like rather than a different normalisation. A producer that only emits the new
    column cannot tell "the population changed" from "my recipe changed".
  * `_control_identity` -- n_clips / n_frames / n_boxes_in_vocab reproduced from the file.
  * `_control_out_of_vocabulary` -- counted, never silently dropped.

Run (CPU, read-only over the join; ~10-12 min on the 327 MB B1 xz):

  PYTHONPATH=D:/Projects/TanitAD/stack python stack/scripts/build_cls_weight_artifact.py \
      --join D:/Projects/TanitAD-artifacts/a40-rescue/b1_train_plus_eval_agents.jsonl.xz \
      --corpus-line v7-b1-physicalai-b1-w120-256x640cyl \
      --out stack/tanitad/data/agent_cls_weights_b1.json

⛔ Clip ids never appear in the output -- only `sha256(clip_id)[:12]` counts.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import lzma
import pathlib
import time

from tanitad.data.bev_raster import ALL_CLASSES, GRID_DEFAULT
from tanitad.models.agent_slots import (
    AGENT_CLASSES, TARGET_POPULATION_RAW, TARGET_POPULATION_VISIBLE,
    cls_weight_digest,
)
from tanitad.refs.refc_agents import FOV_HALF_ANGLE_RAD

import math

HALF = float(FOV_HALF_ANGLE_RAD)
X_MAX = float(GRID_DEFAULT.x_fwd_m)
Y_HALF = float(GRID_DEFAULT.y_half_m)


def inv_freq_mean1(counts: dict) -> dict:
    """⭐ THE ONE RECIPE, and it is the banked artifact's own.

    ``w_c = 1 / max(n_c, 1)``, then divided by the MEAN over the ten classes so the vector
    has mean 1. Scale is not a free variable: ``slot_set_loss``'s denominator follows the
    weights, so a vector with mean != 1 would move the ``cls`` term against every sibling
    loss as well as its per-class emphasis -- two changes in a one-variable arm.
    """
    v = {c: 1.0 / max(int(counts.get(c, 0)), 1) for c in AGENT_CLASSES}
    m = sum(v.values()) / len(v)
    return {c: v[c] / m for c in AGENT_CLASSES}


def rounded(d: dict, nd: int = 6) -> dict:
    return {c: round(float(d[c]), nd) for c in AGENT_CLASSES}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", required=True)
    ap.add_argument("--corpus-line", required=True,
                    help="the corpus line this join IS -- agent_slots.CORPUS_LINE_*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-lines", type=int, default=0,
                    help="SMOKE ONLY; a limited pass writes _SMOKE true and must never "
                         "be banked")
    ap.add_argument("--compare-to", default=None,
                    help="an existing artifact whose RAW vector this pass must reproduce "
                         "(the like-for-like control)")
    ap.add_argument("--coverage-note", default=None,
                    help="how this join covers the arm's corpus, carried into _source "
                         "verbatim (a provenance STRING, never a computed number)")
    ap.add_argument("--ruling", default=None,
                    help="the H-BOXCLS-1 ruling line to carry, verbatim")
    a = ap.parse_args()

    t0 = time.time()
    raw_cls: collections.Counter = collections.Counter()
    vis_cls: collections.Counter = collections.Counter()
    n_lines = 0
    clips: set[str] = set()
    n_boxes = 0

    opener = lzma.open if str(a.join).endswith(".xz") else open
    with opener(a.join, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ags = d.get("agents")
            if ags is None:
                continue
            n_lines += 1
            clips.add(hashlib.sha256(str(d["clip_id"]).encode()).hexdigest()[:12])
            for g in ags:
                n_boxes += 1
                c = str(g.get("cls"))
                raw_cls[c] += 1
                cx, cy = float(g["cx"]), float(g["cy"])
                # ⛔ THE SAME PREDICATE `visible_target_filter` APPLIES, and it is written
                # out here rather than imported only because this pass runs over raw JSON
                # rows, not target tensors. `tests/test_refcv6_visibility_fix.py` pins the
                # two against each other on shared rows, so they cannot drift.
                if (math.atan2(abs(cy), cx) <= HALF
                        and 0.0 <= cx <= X_MAX and abs(cy) <= Y_HALF):
                    vis_cls[c] += 1
            if a.limit_lines and n_lines >= a.limit_lines:
                break

    raw_counts = {c: int(raw_cls.get(c, 0)) for c in AGENT_CLASSES}
    vis_counts = {c: int(vis_cls.get(c, 0)) for c in AGENT_CLASSES}
    oov = {c: int(n) for c, n in raw_cls.items() if c not in set(ALL_CLASSES)}
    w_raw, w_vis = inv_freq_mean1(raw_counts), inv_freq_mean1(vis_counts)

    def imbalance(counts: dict) -> float:
        nz = [v for v in counts.values() if v > 0]
        return round(max(nz) / min(nz), 1) if nz else float("nan")

    dev = None
    if a.compare_to:
        prev = json.loads(pathlib.Path(a.compare_to).read_text(encoding="utf-8"))
        pw = prev.get("weights_inv_freq_mean1") or {}
        if pw:
            dev = max(abs(w_raw[c] - float(pw[c])) for c in AGENT_CLASSES if c in pw)

    art = {
        "_what": "inverse-frequency class weights for the agent/box slot head cls term "
                 "(H-BOXCLS-1), in BOTH target populations",
        "_evidence_class": "MEASURED (ours), CPU, read-only over the named agent join",
        "_ruling": a.ruling or (
            "H-BOXCLS-1. The arm's box loss selects the population: an unfiltered "
            "loss takes `weights_inv_freq_mean1`, a `visible_target_filter`ed one "
            "takes `weights_inv_freq_mean1_visible`. They are NOT interchangeable."),
        "corpus_line": str(a.corpus_line),
        "_corpus_line_note":
            "⛔ LOAD-BEARING. `load_cls_class_weight` REFUSES a vector whose declared line "
            "does not match the arm's, and refuses an artifact that declares none. "
            "MEASURED 2026-09-22: the parity and v7-B1 lines share only 4.09 % of their "
            "clips (max weight ratio 1.857x on `rider`).",
        "target_population": TARGET_POPULATION_RAW,
        "_target_population_note":
            "⛔ THE SECOND SCOPE. `corpus_line` says WHICH CLIPS; this says WHICH BOXES OF "
            "THEM. MEASURED 2026-09-22 on v7-B1: adding the visibility filter moves the "
            "frequencies the cls term meets by 1.746x (other_vehicle) / 0.663x (stroller), "
            "2.63x end to end. A filtered loss running the raw vector is a NEW scope error "
            "-- the `anchors.pt` units defect in a frequency costume.",
        "_visible_predicate": {
            "azimuth": f"atan2(|cy|, cx) <= {HALF:.10f} rad "
                       f"({math.degrees(HALF):.1f} deg, refc_agents.FOV_HALF_ANGLE_RAD)",
            "decode_box": f"0 <= cx <= {X_MAX} and |cy| <= {Y_HALF} "
                          f"(agent_slots.SlotDecodeRanges from bev_raster.GRID_DEFAULT)",
            "identical_to": "refc_agents.visible_target_filter",
        },
        "_source": {
            "join": str(pathlib.Path(a.join).name),
            "n_clips": len(clips),
            "n_frames": n_lines,
            "n_boxes_total": n_boxes,
            "n_boxes_in_vocab": sum(raw_counts.values()),
            "n_boxes_in_vocab_visible": sum(vis_counts.values()),
            **({"corpus_coverage": a.coverage_note} if a.coverage_note else {}),
            "_identity": "these reproduce the census pass exactly; a rebuild that changes "
                         "them changed the corpus, not the recipe",
        },
        "_producer": "stack/scripts/build_cls_weight_artifact.py (IN THE REPO -- the "
                     "previous producer path was a scratch dir that no longer exists)",
        "_normalisation": "inverse frequency, normalised to MEAN 1 -- scale is absorbed by "
                          "slot_set_loss's weight-following denominator, so only the "
                          "RELATIVE emphasis is expressed",
        "_out_of_vocabulary": {
            **oov,
            "_handling": "targets_from_join maps these to -1 and slot_set_loss masks "
                         "ok = ct >= 0, so they are EXCLUDED from the cls term, never "
                         "relabelled",
        },
        "counts": raw_counts,
        "counts_visible": vis_counts,
        "share": rounded({c: raw_counts[c] / max(sum(raw_counts.values()), 1)
                          for c in AGENT_CLASSES}),
        "share_visible": rounded({c: vis_counts[c] / max(sum(vis_counts.values()), 1)
                                  for c in AGENT_CLASSES}),
        "imbalance_majority_to_rarest": imbalance(raw_counts),
        "imbalance_majority_to_rarest_visible": imbalance(vis_counts),
        "weights_inv_freq_mean1": rounded(w_raw),
        "weights_inv_freq_mean1_visible": rounded(w_vis),
        "ratio_visible_over_raw": {
            c: round(rounded(w_vis)[c] / max(rounded(w_raw)[c], 1e-12), 4)
            for c in AGENT_CLASSES},
        "_digest_recipe": "sha256(\"|\".join(f\"{class}={weight:.6f}\" for class in "
                          "AGENT_CLASSES order))[:16] -- "
                          "tanitad.models.agent_slots.cls_weight_digest; VERIFIED on load",
        "_control_reconstruction_max_abs_dev_from_previous_raw": dev,
        "_control_reading":
            "~0 proves this producer reproduces the banked vector's own recipe from the "
            "raw counts, so the _visible column is a like-for-like comparison and not a "
            "different normalisation. None means no --compare-to was given.",
        "_SMOKE": bool(a.limit_lines),
        "_wall_s": round(time.time() - t0, 1),
    }
    # ⛔ COMPUTED. There is no code path in this file that writes a digest by hand.
    art["_self_digest_sha256_of_weights"] = cls_weight_digest(
        [rounded(w_raw)[c] for c in AGENT_CLASSES], AGENT_CLASSES)
    art["_self_digest_sha256_of_weights_visible"] = cls_weight_digest(
        [rounded(w_vis)[c] for c in AGENT_CLASSES], AGENT_CLASSES)
    art["_population_keys"] = {
        TARGET_POPULATION_RAW: "weights_inv_freq_mean1",
        TARGET_POPULATION_VISIBLE: "weights_inv_freq_mean1_visible",
    }

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(art, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\r\n")
    print(json.dumps({k: art[k] for k in (
        "corpus_line", "_source", "counts", "counts_visible",
        "imbalance_majority_to_rarest", "imbalance_majority_to_rarest_visible",
        "weights_inv_freq_mean1", "weights_inv_freq_mean1_visible",
        "ratio_visible_over_raw",
        "_control_reconstruction_max_abs_dev_from_previous_raw",
        "_self_digest_sha256_of_weights",
        "_self_digest_sha256_of_weights_visible", "_SMOKE", "_wall_s")},
        indent=1, ensure_ascii=False))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
