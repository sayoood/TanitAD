#!/usr/bin/env python3
"""Re-derive the canonical ladder re-quote table by OPENING the banked 3-seed JSON.

This computes nothing about the model. It opens
`.../incoming/2026-08-18-ladder-3seed/raw/reread_{unpen,centred}/ll3_*.json`
and arranges the per-seed fields so a citation can be written with its provenance.

WHY IT EXISTS (C103 -> C107 -> this sweep): a citation sweep that copies the
RETRACTION instead of the ARTIFACT inherits the retraction's own provenance
errors. Two "replacement" numbers in circulation (`lead_gap` r_pv0 -0.107 and
"0.694 m worse than the null") are SEED-0 values, not 3-seed means.

RULES ENCODED HERE:
  * route A (`unpen`) and route B (`centred`) are emitted SIDE BY SIDE and
    NEVER pooled (C100/C103).
  * a seed range is emitted as `*_seed_SPREAD_not_a_CI` so it can never be
    mistaken for an estimator (C106 -> C109).
  * every K1 carries its own paired-episode-cluster-bootstrap CI and its
    C97 guard verdict; `overlapping_holdout_se` is never imported.

Tier: T0-DIAGNOSTIC. Zero GPU. Run from the repo root:
    PYTHONIOENCODING=utf-8 python "<this file>"
"""
from __future__ import annotations

import json
import os

LADDER = (
    "TanitAD Research Hub/Architecture & Inference/Implementation/"
    "incoming/2026-08-18-ladder-3seed/raw"
)
OUT = (
    "TanitAD Research Hub/Benchmarks & Eval/Implementation/"
    "incoming/2026-08-18-citation-sweep/raw/canonical_requote_table.json"
)

TARGETS = [
    "ego_v0", "ego_accel", "ego_yawrate", "ego_curv",
    "n_agents_grid", "n_agents_all", "nearest_any",
    "lead_present", "lead_gap", "lead_closing", "lead_inv_ttc",
]
SEEDS = ("0", "1", "2")

# label, route dir, filename
ARMS = [
    ("v6F@11250", "reread_unpen", "ll3_s11250.json"),
    ("v6F@11250", "reread_centred", "ll3_s11250.json"),
    ("MATCHED-RANDOM-NULL", "reread_unpen", "ll3_nullmatched.json"),
    ("C-V0 (ego speed alone)", "reread_unpen", "ll3_proxyv0.json"),
    ("EGO-ORACLE n10", "reread_unpen", "ll3_egoorc_n10.json"),
]


def load(route: str, fn: str) -> dict:
    with open(os.path.join(LADDER, route, fn), encoding="utf-8") as fh:
        return json.load(fh)


def route_name(route_dir: str) -> str:
    return "A_unpen" if route_dir == "reread_unpen" else "B_centred"


def main() -> None:
    out = {
        "_evidence_class": (
            "MEASURED (ours; re-derived by the citation-sweep agent by opening "
            "the banked 3-seed JSON, C91)"
        ),
        "eval_tier": (
            "T0-DIAGNOSTIC (frozen-latent linear readout; never driving performance)"
        ),
        "estimator": (
            "taniteval.ci.paired_episode_cluster_bootstrap, n_boot 2000, "
            "70 episode clusters"
        ),
        "forbidden": "overlapping_holdout_se",
        "arm": "v6F-SW-30k@11250 (EARLY READ, 11250/30000 = 37.5%)",
        "corpus": (
            "130-clip lead-enriched probe pool, 60 probe-train / 70 eval, "
            "clip-disjoint. NOT the 40-episode val set. No episode selected, "
            "added, removed, reordered or re-hashed."
        ),
        "seeds": [0, 1, 2],
        "routes_never_pooled": True,
        "sources": {
            "route_A_unpen": LADDER + "/reread_unpen/ll3_*.json",
            "route_B_centred": LADDER + "/reread_centred/ll3_*.json",
            "aggregate": LADDER + "/reread3_table.json",
        },
        "rungs": {},
        "arms": {},
    }

    a_doc = load("reread_unpen", "ll3_s11250.json")
    b_doc = load("reread_centred", "ll3_s11250.json")
    null_doc = load("reread_unpen", "ll3_nullmatched.json")

    for tgt in TARGETS:
        a = a_doc["targets"][tgt]["per_seed"]
        b = b_doc["targets"][tgt]["per_seed"]
        n = null_doc["targets"][tgt]["per_seed"]
        r2a = [a[s]["r2_ceiling"] for s in SEEDS]
        r2b = [b[s]["r2_ceiling"] for s in SEEDS]
        r2n = [n[s]["r2_ceiling"] for s in SEEDS]
        pv = [a[s].get("corr_partial_v0") for s in SEEDS]
        has_pv = all(v is not None for v in pv)
        out["rungs"][tgt] = {
            "gt_sd": a_doc["targets"][tgt]["gt_sd"],
            "n_eval": a_doc["targets"][tgt]["n_eval"],
            "n_eval_clusters": a_doc["targets"][tgt]["n_eval_clusters"],
            "routeA_r2_per_seed": r2a,
            "routeA_r2_3seed_mean": round(sum(r2a) / 3, 6),
            "routeB_r2_per_seed": r2b,
            "routeB_r2_3seed_mean": round(sum(r2b) / 3, 6),
            "routeA_minus_routeB_max_abs": round(
                max(abs(x - y) for x, y in zip(r2a, r2b)), 6
            ),
            "matched_random_null_r2_3seed_mean": round(sum(r2n) / 3, 6),
            "r_partial_v0_per_seed": pv,
            "r_partial_v0_3seed_mean": round(sum(pv) / 3, 6) if has_pv else None,
            # NOT an interval. C106 quoted a spread where an estimator was implied.
            "r_partial_v0_seed_SPREAD_not_a_CI": [min(pv), max(pv)] if has_pv else None,
        }

    for label, route_dir, fn in ARMS:
        doc = load(route_dir, fn)
        key = f"{label} | route {route_name(route_dir)}"
        out["arms"][key] = {}
        for tgt in ("ego_v0", "lead_gap"):
            ps = doc["targets"][tgt]["per_seed"]
            out["arms"][key][tgt] = {
                "K1_per_seed": [ps[s]["K1_delta"] for s in SEEDS],
                "K1_CI_per_seed": [[ps[s]["K1_lo"], ps[s]["K1_hi"]] for s in SEEDS],
                "K1_separated_per_seed": [ps[s]["K1_separated"] for s in SEEDS],
                "MAE_per_seed": [ps[s]["err"] for s in SEEDS],
                "MAE_3seed_mean": round(sum(ps[s]["err"] for s in SEEDS) / 3, 4),
                "alpha_per_seed": [ps[s]["alpha_chosen"] for s in SEEDS],
                "guard_per_seed": [
                    ps[s]["k1_guard"]["guard_verdict"] for s in SEEDS
                ],
            }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
