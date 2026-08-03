"""Produce the DEPLOYED situation score by late fusion, and measure it.

This is the consumer ``tanitad.eval.sitclf.late_fuse_scores`` never had. It reads
the per-frame score bundle the situation-classifier trainer emits, replaces the
trainer's FEATURE-level concat (``sc_train.py:143`` / ``sc_train_v2.py:143``)
with SCORE-level fusion of the same two modalities, and reports the change with
the program's binding estimator plus the controls that make a before/after
admissible.

⛔ Both modalities are kept. The ego-only swap is closed (PI, 2026-08-03).

Arms produced
-------------
``FUSED``          late_fuse(head_img, head_ego)    <- the fix
``DEPLOYED``       head_img_ego                     <- the early-concat baseline
``NEG_CAMERA``     late_fuse(head_img_shuf, head_ego)
                   identical parameter count and fitting protocol with the
                   camera destroyed -> isolates the camera's MARGINAL value
``NEG_MACHINERY``  late_fuse(head_img_ego)
                   the combiner run on ONE column: if this improves the column,
                   the fusion's free parameters are the gain and not the fusion
``NEG_LABEL``      late_fuse(head_img, head_ego) fitted on labels permuted
                   across clusters -> the protocol must land at chance

usage:
  python scripts/fuse_situation_scores.py --bundle <heldout_frames.npz> \
      --out results_sitclf_fusion.json [--n-boot 2000] [--strata-n-boot 400]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from tanitad.eval.ap_ci import (ap_lift, average_precision,
                                paired_ap_episode_cluster_bootstrap)
from tanitad.eval.sitclf import cluster_folds, late_fuse_scores
from tanitad.eval.sitclf_deploy import (DEPLOYED_ARM, MODALITY_ARMS,
                                        NULL_IMAGE_ARM, VISION_ARMS,
                                        VISION_NULL_ARMS, four_family_report,
                                        fuse_modalities, load_score_bundle,
                                        permute_labels_by_cluster,
                                        vision_only_arms)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def build_arms(b, situation: str, *, n_folds: int, seed: int) -> dict:
    """Every arm this run compares, on identical rows."""
    i = b.col(situation)
    arms = {
        "FUSED": fuse_modalities(b, situation, arms=MODALITY_ARMS,
                                 n_folds=n_folds, seed=seed),
        "DEPLOYED": b.arm(DEPLOYED_ARM, situation).astype(np.float64),
        "NEG_MACHINERY": fuse_modalities(b, situation, arms=(DEPLOYED_ARM,),
                                         n_folds=n_folds, seed=seed),
    }
    if NULL_IMAGE_ARM in b.scores:
        arms["NEG_CAMERA"] = fuse_modalities(
            b, situation, arms=(NULL_IMAGE_ARM, MODALITY_ARMS[1]),
            n_folds=n_folds, seed=seed)
    # NEG_LABEL: the combiner is FITTED on labels permuted across clusters, then
    # scored against the TRUE labels. Permuting whole clusters (never rows) keeps
    # the within-clip correlation the cluster estimator assumes.
    cc = b.clip_cluster
    y_perm = permute_labels_by_cluster(b.y[:, i], cc, seed=seed + 991)
    cols = np.stack([b.arm(a, situation) for a in MODALITY_ARMS], 1)
    folds = cluster_folds(cc, n_folds=n_folds, seed=seed)
    arms["NEG_LABEL"] = late_fuse_scores(cols, y_perm, b.valid[:, i], folds)
    return arms


CONTRASTS = [
    ("FUSED_vs_DEPLOYED", "FUSED", "DEPLOYED",
     "THE FIX: score-level fusion of both modalities vs the early-concat arm"),
    ("FUSED_vs_NEG_CAMERA", "FUSED", "NEG_CAMERA",
     "the CAMERA's marginal value: same combiner, image features destroyed"),
    ("NEG_MACHINERY_vs_DEPLOYED", "NEG_MACHINERY", "DEPLOYED",
     "the combiner's free parameters ALONE: must not be a gain"),
    ("FUSED_vs_NEG_LABEL", "FUSED", "NEG_LABEL",
     "the protocol cannot manufacture signal: fitted on permuted labels"),
]

# --------------------------------------------------------------------------- #
# VISION-ONLY preset — the PI ruling of 2026-08-03                            #
#                                                                             #
#   "for ground truth data of scenario classification you can use both ego and #
#    other label, for inference only vision."                                  #
#                                                                             #
# ⇒ ego may DERIVE the labels and may NOT be an INFERENCE input. That closes   #
#   `head_ego` AND `head_img_ego` AND any image+ego late fusion — score-level  #
#   or not, an ego score at inference is an ego input. The deployable arm is   #
#   `head_img`. `late_fuse_scores` keeps a role, but between VISION arms only. #
#                                                                             #
# The panel itself lives in `tanitad.eval.sitclf_deploy.vision_only_arms` so   #
# it is under test; this script only sequences and reports it.                 #
# --------------------------------------------------------------------------- #
VISION_CONTRASTS = [
    ("PRIMARY_vs_NEG_VISION", "PRIMARY", "NEG_VISION",
     "DISCRIMINATION CONTROL, run first: does the camera carry ANY signal over "
     "its own permuted-feature null?"),
    ("FUSED_vs_PRIMARY", "FUSED", "PRIMARY",
     "THE FIX under the ruling: score-level fusion of TWO VISION arms vs the "
     "single deployable vision arm"),
    ("FUSED_vs_NEG_FUSED", "FUSED", "NEG_FUSED",
     "the fused vision score against the identical combiner on permuted vision "
     "features — the camera's value with parameter count held fixed"),
    ("NEG_MACHINERY_vs_PRIMARY", "NEG_MACHINERY", "PRIMARY",
     "the combiner's free parameters ALONE on one vision column: must not be a gain"),
    ("PRIMARY_vs_NEG_LABEL", "PRIMARY", "NEG_LABEL",
     "the protocol cannot manufacture signal: combiner fitted on permuted labels"),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--strata-n-boot", type=int, default=400)
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--situations", default="", help="comma list; default = all")
    p.add_argument("--save-scores", default="", help="optional .npz of fused scores")
    p.add_argument("--preset", choices=("multimodal", "vision_only"),
                   default="vision_only",
                   help="vision_only is the PI ruling of 2026-08-03: ego derives "
                        "the LABELS, vision alone is the INFERENCE input")
    a = p.parse_args()
    vision = a.preset == "vision_only"
    make_arms = vision_only_arms if vision else build_arms
    contrasts = VISION_CONTRASTS if vision else CONTRASTS
    primary, fixed = ("PRIMARY", "FUSED") if vision else ("DEPLOYED", "FUSED")

    b = load_score_bundle(a.bundle)
    log(f"bundle {a.bundle}")
    log(f"  {b.n_rows:,} rows · {b.n_clusters:,} clip clusters · "
        f"situations {b.situations} · arms {b.arms}")
    sits = [s.strip() for s in a.situations.split(",") if s.strip()] or list(b.situations)

    out = {
        "_what": ("VISION-ONLY situation classification (PI ruling 2026-08-03: ego "
                  "may derive the LABELS, only vision may be an INFERENCE input)"
                  if vision else
                  "late-fusion repair of the deployed situation classifier"),
        "preset": a.preset,
        "_inference_inputs": (list(VISION_ARMS) if vision
                              else [DEPLOYED_ARM, *MODALITY_ARMS]),
        "bundle": os.path.abspath(a.bundle),
        "n_rows": b.n_rows, "n_clusters": b.n_clusters,
        "situations": sits, "bundle_arms": list(b.arms),
        "deployed_arm": (VISION_ARMS[0] if vision else DEPLOYED_ARM),
        "modality_arms": list(VISION_ARMS if vision else MODALITY_ARMS),
        "null_image_arm": (VISION_NULL_ARMS[0] if vision else NULL_IMAGE_ARM),
        "fusion": {"fn": "tanitad.eval.sitclf.late_fuse_scores",
                   "consumer": "tanitad.eval.sitclf_deploy.fuse_modalities",
                   "n_folds": a.folds, "seed": a.seed, "l2": 1.0,
                   "fold_unit": "whole clip cluster"},
        "estimator": "paired_ap_episode_cluster_bootstrap on AP-LIFT",
        "n_boot": a.n_boot, "strata_n_boot": a.strata_n_boot,
        "python": sys.version.split()[0],
        "per_situation": {},
    }

    saved = {}
    for sit in sits:
        log(f"=== {sit} ===")
        i = b.col(sit)
        arms = make_arms(b, sit, n_folds=a.folds, seed=a.seed)
        for k in (primary, fixed):
            if k in arms:
                saved[f"{sit}__{k}"] = arms[k].astype(np.float32)
        m0 = b.valid[:, i]
        m = m0.copy()
        for v in arms.values():
            m &= np.isfinite(v)
        y = b.y[:, i]
        eid = b.clip_cluster
        rec = {"n_valid_rows": int(m0.sum()), "n_rows_scored": int(m.sum()),
               "n_pos": int(y[m].sum()), "base_rate": round(float(y[m].mean()), 6),
               "ap": {}, "ap_lift": {}, "contrasts": {}}
        for k, v in arms.items():
            rec["ap"][k] = round(average_precision(y[m], v[m]), 5)
            rec["ap_lift"][k] = round(ap_lift(y[m], v[m]), 5)
        # the input arms, for context (not contrasted — they ARE the inputs)
        for k in (VISION_ARMS if vision else MODALITY_ARMS):
            if k in b.scores:
                rec["ap"][k] = round(average_precision(y[m], b.arm(k, sit)[m]), 5)
                rec["ap_lift"][k] = round(ap_lift(y[m], b.arm(k, sit)[m]), 5)
        log(f"  AP: " + "  ".join(f"{k}={v:.5f}" for k, v in rec["ap"].items()))

        for name, ka, kb, what in contrasts:
            if ka not in arms or kb not in arms:
                rec["contrasts"][name] = {"_status": "UNAVAILABLE",
                                          "_reason": f"missing arm {ka if ka not in arms else kb}"}
                continue
            d = paired_ap_episode_cluster_bootstrap(
                y[m], arms[ka][m], arms[kb][m], eid[m],
                n_boot=a.n_boot, seed=a.seed, lift=True)
            d["_what"] = what
            rec["contrasts"][name] = d
            log(f"  {name:28s} d={d['delta']:+8.4f} "
                f"[{d['lo']:+8.4f}, {d['hi']:+8.4f}] "
                f"{'SEPARATED' if d['separated'] else 'not separated'}")

        log(f"  four families ...")
        base_k = primary if fixed not in arms else primary
        fix_k = fixed if fixed in arms else primary
        rec["four_families"] = four_family_report(
            b, sit, fused=arms[fix_k], baseline=arms[base_k],
            fused_name=fix_k, baseline_name=base_k,
            n_boot=a.n_boot, seed=a.seed, strata_n_boot=a.strata_n_boot)
        out["per_situation"][sit] = rec

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    log(f"wrote {a.out}")
    if a.save_scores:
        np.savez_compressed(a.save_scores, clip_cluster=b.clip_cluster,
                            y=b.y, valid=b.valid, situations=np.array(b.situations),
                            **saved)
        log(f"wrote {a.save_scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
