#!/usr/bin/env python3
"""Re-derive, IN THIS REPO WITH NO POD AND NO GPU, every load-bearing number the
refreshed ``DataEng/DATA_STRATEGY.md`` v4.0 quotes for the parity-contamination
finding and the feature read-set.

WHY THIS EXISTS
===============
``DATA_STRATEGY.md`` is an INDEX: it points at owners and must not become a second
source of truth. But three of its v4.0 claims decide whether the 4,472-clip build may
run at all, and the operating standard forbids a decision-grade number that is
INHERITED. So each of them is re-derived here from primary artifacts rather than
copied from ``contamination.json`` or from any prose:

1. the aug120 perception cohort and the parity-train exclusion list are the SAME SET
   (not merely the same size) -- the C113 correction to C112;
2. the catalogue rate (4.25 %), the buildable rate (78.21 %) and the 6-of-40
   deployed-val figure, computed through the committed membership oracle;
3. the PhysicalAI feature read-set counts, which are pinned in
   ``stack/tests/test_physicalai_feature_readset.py`` and must be quoted BY LAYER.

Run:  PYTHONUTF8=1 python verify_strategy_claims.py [--json OUT]

``PYTHONUTF8=1`` is required on the dev box for the same reason
``build_obstacle_join.py`` needs it: the default cp1252 stdout encoder dies on the
non-ASCII characters these artifacts carry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Repo layout. This file lives at
#   TanitAD Research Hub/Data Engineering/Implementation/incoming/
#       2026-08-18-data-strategy-refresh/code/verify_strategy_claims.py
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "stack"))

HUB = REPO / "TanitAD Research Hub"
PILOT = HUB / "Architecture & Inference/Implementation/incoming/2026-08-17-thor-concurrency-pilot"
DE_INC = HUB / "Data Engineering/Implementation/incoming"

#: The aug120 fused-record index, at the two schema revisions that exist.
V2_INDEX = DE_INC / "2026-08-17-aug120-refuse/raw/fused_aug120_v2_index.jsonl"
V3_INDEX = DE_INC / "2026-08-17-perception-floor-unify/raw/fused_aug120_v3_index.jsonl"

#: The banked exclusion list and the full Alpamayo record-id list.
EXCLUDE = PILOT / "alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL.txt"
ALPAMAYO = PILOT / "alpamayo_clip_ids.txt"

#: MEASURED by the 4,472-build inputs doc: of 4,729 augmented clips only this many
#: have w120 video built, which is the denominator that matters for a split someone
#: could build TODAY (C113).
N_ALPAMAYO_WITH_W120 = 257

#: The claim under test, from C113 / MODEL_REGISTRY.md 12.4.
CLAIMED_SET_DIGEST = "80632f17"


def _lines(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _clip_ids_from_jsonl(p: Path) -> list[str]:
    out: list[str] = []
    for ln in _lines(p):
        rec = json.loads(ln)
        cid = rec.get("clip_id") or rec.get("clip") or rec.get("id")
        if cid is None:
            raise KeyError(f"no clip id key in a record of {p.name}: {sorted(rec)[:8]}")
        out.append(str(cid))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None, help="write the result record here")
    args = ap.parse_args()

    from tanitad.data.parity import (  # noqa: E402  (path set above)
        clip_digest,
        clips_in_deployed_val,
        clips_in_parity_train,
        deployed_val_clip_digests,
        parity_train_clip_digests,
        uid_digest,
    )

    rec: dict = {
        "_evidence_class": "MEASURED (ours; this script, re-derived from primary artifacts)",
        "_what": "independent re-derivation of the DATA_STRATEGY.md v4.0 parity numbers",
        "_no_pod": True,
        "_no_gpu": True,
    }

    # --- 1. set IDENTITY, not a matching count ------------------------------- #
    v2 = _clip_ids_from_jsonl(V2_INDEX)
    v3 = _clip_ids_from_jsonl(V3_INDEX)
    ex = _lines(EXCLUDE)
    digests = {name: uid_digest(ids) for name, ids in
               (("fused_aug120_v2_index", v2), ("fused_aug120_v3_index", v3),
                ("alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL", ex))}
    all_equal = len({*digests.values()}) == 1
    rec["set_identity"] = {
        "_claim": "the aug120 perception cohort IS the parity-train exclusion list, "
                  "byte-identical as a SET -- not a coincidence of count (C113 vs C112)",
        "n": {"v2_index": len(v2), "v3_index": len(v3), "exclusion_list": len(ex)},
        "sorted_id_sha256": digests,
        "all_three_identical": all_equal,
        "set_equality": set(v2) == set(v3) == set(ex),
        "matches_claimed_prefix": all(d.startswith(CLAIMED_SET_DIGEST) for d in digests.values()),
        "serialization": "parity.uid_digest -- sha256 over the newline-joined SORTED ids",
    }

    # --- 2. the three contamination rates, through the oracle ---------------- #
    alp = _lines(ALPAMAYO)
    in_train = clips_in_parity_train(alp)
    in_val = clips_in_deployed_val(alp)
    n_val = len(deployed_val_clip_digests())
    rec["contamination"] = {
        "n_alpamayo_records": len(alp),
        "parity_train_digest_set": len(parity_train_clip_digests()),
        "deployed_val_digest_set": n_val,
        "catalogue_rate": {
            "n_in_parity_train": len(in_train),
            "denominator": len(alp),
            "frac": round(len(in_train) / len(alp), 6),
            "_why_it_flatters": "counts clips that do not exist as video",
        },
        "buildable_rate_today": {
            "n_in_parity_train": len(in_train),
            "denominator": N_ALPAMAYO_WITH_W120,
            "frac": round(len(in_train) / N_ALPAMAYO_WITH_W120, 6),
            "_why_it_is_the_one_to_quote": "a split can only contain clips that EXIST",
        },
        "deployed_val_swallowed": {
            "n_in_alpamayo_records": len(in_val),
            "denominator": n_val,
            "frac": round(len(in_val) / n_val, 4),
            "_direction": "a TRAIN corpus about to swallow the deployed val -- "
                          "the worse direction, blast radius zero TODAY",
        },
        "aug120_cohort_in_parity_train": f"{len(clips_in_parity_train(v3))}/{len(v3)}",
        "aug120_cohort_in_deployed_val": f"{len(clips_in_deployed_val(v3))}/{len(v3)}",
    }

    # --- 3. the feature read-set, BY LAYER ----------------------------------- #
    import tests.test_physicalai_feature_readset as fr  # type: ignore  # noqa: E402

    rec["feature_readset"] = {
        "_rule": "state the LAYER; the bare phrase 'our ingest' is what let this rot four times",
        "physicalai_r0.py (r0 clip selection)": len(fr.R0_SELECTION_FEATURES),
        "physicalai.py (the episode build)": len(fr.EPISODE_BUILD_FEATURES),
        "program-wide (incl. the side-car join)": len(fr.PROGRAM_WIDE_FEATURES),
        "denominator": fr.PHYSICALAI_TOTAL_FEATURES,
        "program_wide_names": sorted(fr.PROGRAM_WIDE_FEATURES),
        "pinned_by": "stack/tests/test_physicalai_feature_readset.py",
    }

    # --- 4. the oracle is one-way -------------------------------------------- #
    rec["oracle_shape"] = {
        "_claim": "membership exact, enumeration impossible",
        "digest_of_one_id_is_deterministic": clip_digest("x") == clip_digest("x"),
        "digest_len_hex": len(clip_digest("x")),
        "committed_files": [
            "stack/tanitad/data/parity_train_clip_digests.json",
            "stack/tanitad/data/deployed_val40_clip_digests.json",
        ],
        "_note": "the sets hold sha256(clip_id) only; an id can be TESTED but the set "
                 "cannot be run backwards into ids",
    }

    ok = (rec["set_identity"]["all_three_identical"]
          and rec["set_identity"]["set_equality"]
          and rec["set_identity"]["matches_claimed_prefix"]
          and rec["contamination"]["catalogue_rate"]["n_in_parity_train"] == 201
          and rec["contamination"]["deployed_val_swallowed"]["n_in_alpamayo_records"] == 6
          and rec["feature_readset"]["program-wide (incl. the side-car join)"] == 6)
    rec["ALL_CLAIMS_REPRODUCED"] = bool(ok)

    print(json.dumps(rec, indent=1, ensure_ascii=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rec, indent=1, ensure_ascii=True) + "\n",
                             encoding="utf-8")
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
