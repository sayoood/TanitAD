"""REF-C v3 — the SCALE x HIERARCHY matrix, from BUILT models.

⛔ WHY THIS EXISTS. The choice in front of the PI ("package A" vs "package B")
was carried as ARITHMETIC in a review document: a core count from one build plus
a rung count from another, added by hand. Arithmetic is not a build. This script
constructs every cell and reports what the optimiser would actually see.

Two independent axes, and the point of the matrix is that they are NOT
independent for the one decision that binds:

  SCALE       `V3_SIZES` — small / base / xl. Moves the ENCODER ONLY (C122).
  HIERARCHY   `thin`     — v3's incumbent cascade: PhiTac + linear heads.
              `v6`       — `HierarchyRung` at v6's OWN geometry (body_layers=0).
              `aligned`  — ⭐ the PI's 2026-08-21 ruling: REF-A v1 / REF-D
                           tactical CAPACITY carrying v6's OWN FTac predictor,
                           goal head and vocabulary tables.

⭐ THE FINDING THE MATRIX EXISTS TO SURFACE: `D-008` requires **>= 250 M**, and
only ONE cell clears it. So "which size" and "which hierarchy" cannot be decided
separately, which is exactly what the incumbent docstring asserts they can be
(*"the hierarchy cost is essentially CONSTANT across the ladder, so SCALE and
HIERARCHY are independent decisions"*). That claim is TRUE of the `thin`
hierarchy and FALSE of the decision, because `thin` is 4x under D-008 at every
size — the independence holds for the cost and not for the constraint.

⚠️ The rung's `d_in` is the encoder's ACTUAL `feat_dim` at that size, not a
constant. It is read off the built core rather than assumed.

Evidence class: MEASURED (ours; this script). Nothing here is a training result.
"""
from __future__ import annotations

import argparse
import json
import sys

from tanitad.models import v6 as V
from tanitad.models.hierarchy import (ALIGNED_STRATEGIC, ALIGNED_TACTICAL,
                                      HierarchyRung, HierarchyRungConfig,
                                      V6_STRATEGIC, V6_TACTICAL,
                                      aligned_body_matches_refa,
                                      rung_param_count)
from tanitad.refs import refc_v3 as R

#: `D-008` (2026-07-05, accepted by Sayed): model scale >= 250 M, tied to
#: "a scale where hierarchy is expressible".
D008_MIN_PARAMS = 250_000_000
#: REF-A v1's tactical `TokenFieldPredictor` blocks, MEASURED. The `aligned`
#: tier is defined to reproduce this, and the matrix PROVES it per cell.
REFA_V1_TACTICAL_BLOCKS = 50_384_896

TIERS = ("thin", "v6", "aligned")


def _vocabs(d_embed: int = 128) -> dict:
    """v6's OWN vocabulary tables, built from v6's canonical token tuples.

    ⚠️ v6 shares its tables BY IDENTITY between the emitting head above and the
    consuming conditioner below. REF-C v3 has no `V6Stack` to share from, so it
    constructs the same tables from the same tuples — same vocabulary by
    DEFINITION. That is the honest reading of the PI's "use the same vocab
    tables like v6"; it is not, and must not be reported as, table sharing.
    """
    return {
        "tac": V.GoalVocabulary(V.TACTICAL_GOAL_TOKENS, d_embed),
        "str": V.GoalVocabulary(V.STRATEGIC_GOAL_TOKENS, d_embed),
        "a_str": V.GoalVocabulary(V.STRATEGIC_ACTION_TOKENS, d_embed),
        "a_lat": V.GoalVocabulary(V.TACTICAL_LAT_ACTIONS, d_embed),
        "a_lon": V.GoalVocabulary(V.TACTICAL_LON_ACTIONS, d_embed),
    }


def build_rungs(tier: str, d_in: int) -> tuple[HierarchyRung, HierarchyRung]:
    """The (tactical, strategic) rung pair for a tier, at this encoder's width."""
    if tier == "v6":
        ct, cs = V6_TACTICAL, V6_STRATEGIC
    elif tier == "aligned":
        ct, cs = ALIGNED_TACTICAL, ALIGNED_STRATEGIC
    else:
        raise ValueError(f"tier {tier!r} has no rungs")
    v = _vocabs(ct.d_goal_embed)
    # d_in comes from the BUILT encoder; the strategic rung's input is the
    # tactical rung's own width, which is the tier's geometry, not the encoder's.
    ct = HierarchyRungConfig(**{**ct.__dict__, "d_in": d_in})
    cs = HierarchyRungConfig(**{**cs.__dict__, "d_in": ct.d_layer})
    tac = HierarchyRung(ct, vocab_goal=v["tac"],
                        vocab_actions=(v["a_lat"], v["a_lon"]),
                        vocab_above=v["str"])
    stra = HierarchyRung(cs, vocab_goal=v["str"], vocab_actions=(v["a_str"],),
                         vocab_above=None)
    return tac, stra


def cell(size: str, tier: str) -> dict:
    """One matrix cell, from a real build."""
    model = R.RefCV3Model(R.refc_v3_sized_config(size, hier=True))
    br = R.param_breakdown_v3(model)
    d_in = int(model.core.encoder.feat_dim)
    thin = br["total"] - br["core"]
    out = {"size": size, "tier": tier, "d_in": d_in, "core": br["core"]}
    if tier == "thin":
        out["hierarchy"] = thin
        out["total"] = br["total"]
        out["body"] = 0
    else:
        tac, stra = build_rungs(tier, d_in)
        pt, ps = rung_param_count(tac), rung_param_count(stra)
        out["hierarchy"] = pt["total"] + ps["total"]
        out["hierarchy_tac"] = pt["total"]
        out["hierarchy_str"] = ps["total"]
        out["body"] = aligned_body_matches_refa(tac)
        # the thin cascade is REPLACED, not stacked on top of
        out["total"] = br["core"] + out["hierarchy"]
    out["d008_ok"] = out["total"] >= D008_MIN_PARAMS
    out["body_matches_refa_v1"] = (out["body"] == REFA_V1_TACTICAL_BLOCKS
                                   if tier == "aligned" else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", nargs="*", default=list(R.V3_SIZES))
    ap.add_argument("--tiers", nargs="*", default=list(TIERS))
    ap.add_argument("--json", help="write the matrix here")
    a = ap.parse_args()

    rows = [cell(s, t) for s in a.sizes for t in a.tiers]

    w = "{:<7}{:<9}{:>7}{:>14}{:>14}{:>15}{:>9}"
    print(w.format("size", "tier", "d_in", "core", "hierarchy", "TOTAL",
                   "D-008"))
    print("-" * 75)
    for r in rows:
        print(w.format(r["size"], r["tier"], r["d_in"], f"{r['core']:,}",
                       f"{r['hierarchy']:,}", f"{r['total']:,}",
                       "OK" if r["d008_ok"] else "under"))
    ok = [r for r in rows if r["d008_ok"]]
    print(f"\n  D-008 (>= {D008_MIN_PARAMS:,}) cleared by {len(ok)} of "
          f"{len(rows)} cells: "
          f"{', '.join(f'{r[chr(115)+chr(105)+chr(122)+chr(101)]}/{r[chr(116)+chr(105)+chr(101)+chr(114)]}' for r in ok) or 'NONE'}")

    bad = [r for r in rows
           if r["tier"] == "aligned" and not r["body_matches_refa_v1"]]
    if bad:
        print(f"\n⛔ {len(bad)} aligned cells do NOT reproduce REF-A v1's "
              f"MEASURED tactical blocks ({REFA_V1_TACTICAL_BLOCKS:,}) — the "
              f"tier is misnamed, not merely mis-sized.")
        return 1
    if "aligned" in a.tiers:
        print(f"  aligned body == REF-A v1 tactical blocks "
              f"({REFA_V1_TACTICAL_BLOCKS:,}) in every aligned cell: PROVEN")

    if a.json:
        payload = {
            "_evidence_class": "MEASURED (ours; refc_v3_scale_matrix.py)",
            "d008_min_params": D008_MIN_PARAMS,
            "refa_v1_tactical_blocks": REFA_V1_TACTICAL_BLOCKS,
            "note": "parameter counts only; NO training result is implied",
            "cells": rows,
        }
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
