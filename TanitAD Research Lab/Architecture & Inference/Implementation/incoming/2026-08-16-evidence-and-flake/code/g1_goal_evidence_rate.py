#!/usr/bin/env python3
"""Re-run the NEW `goal_evidence` predicate over REAL fused records and report
the emission-rate change against the verdict each record already carries.

⛔ Why this shape rather than "re-fuse the corpus": the fused records are the
primary artifact and they carry every input the check reads — the VLM symbols,
the VLM `signs` block, the SAM3 tracks, and the SAM3-absence marker. So the
OLD verdict (in the record) and the NEW verdict (recomputed here from the same
inputs) are an exact A/B on the same clip, with no re-fuse and no GPU.

Scope, stated honestly: only 30 of the 201 aug120 fused records are in the
repo (`…/2026-08-16-s2-strategic-gap/raw/sample_fused_aug120/`). The
corpus-wide before/after is taken from that package's raw
`aug120_analysis.json::route_to_verdicts`, which is a primary JSON, not prose.

Usage:  python3 g1_goal_evidence_rate.py [--out raw/goal_evidence_rate.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "stack" / "scripts"))

SAMPLES = (REPO / "TanitAD Research Hub" / "Data Engineering" / "Implementation"
           / "incoming" / "2026-08-16-s2-strategic-gap" / "raw"
           / "sample_fused_aug120")
CORPUS = (REPO / "TanitAD Research Hub" / "Data Engineering" / "Implementation"
          / "incoming" / "2026-08-16-s2-strategic-gap" / "raw"
          / "aug120_analysis.json")


def v2_from_record(rec: dict) -> dict:
    """Exactly the fields `corroborate()` reads for the goal_evidence path."""
    sem = rec.get("semantics") or {}
    ego = rec.get("ego") or {}
    return {"symbols": sem.get("symbols"), "signs": sem.get("signs"),
            "scene": sem.get("scene"), "ego_state": ego.get("ego_state"),
            "speed_profile": ego.get("speed_profile"),
            "situations": ego.get("situations")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "raw" / "goal_evidence_rate.json"))
    a = ap.parse_args(argv)

    from ph1_fuse import GOAL_EVIDENCE_RETIRED, corroborate

    files = sorted(glob.glob(os.path.join(str(SAMPLES), "*.json")))
    if not files:
        print(f"NO SAMPLE RECORDS at {SAMPLES}", file=sys.stderr)
        return 2

    before = collections.Counter()
    after = collections.Counter()
    kinds = collections.Counter()
    presence = collections.Counter()
    n_route_to = 0
    changed: list[dict] = []

    for f in files:
        rec = json.loads(Path(f).read_text(encoding="utf-8"))
        old = (rec.get("corroboration") or {}).get("goal_evidence")
        if old is None:
            continue                       # not a route_to clip
        n_route_to += 1
        v2 = v2_from_record(rec)
        perc = rec.get("perception") or {}
        absent = "absent" in perc
        cor, conflicts = corroborate(v2, {}, perc.get("tracks") or [],
                                     sam3_absent=absent)
        new = cor["goal_evidence"]
        before[old["verdict"]] += 1
        after[new["verdict"]] += 1
        kinds[new.get("evidence_sign_kind")] += 1
        if not absent:
            presence[bool(new["sign_like_object_present"])] += 1
        assert new["verdict"] not in GOAL_EVIDENCE_RETIRED
        assert not any(c.get("check") == "goal_evidence" for c in conflicts)
        if old["verdict"] != new["verdict"]:
            changed.append({"clip_id": rec["clip_id"],
                            "old": old["verdict"], "new": new["verdict"],
                            "sam3_sign_tracks": new.get("sam3_sign_tracks"),
                            "sign_like_object_present":
                                new.get("sign_like_object_present"),
                            "evidence_sign_kind":
                                new.get("evidence_sign_kind")})

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    rt = corpus["route_to_verdicts"]
    n_clips = corpus["n_records"]
    out = {
        "evidence_class": "MEASURED (recomputation on banked fused records)",
        "sample": {
            "source": str(SAMPLES.relative_to(REPO)).replace("\\", "/"),
            "n_records": len(files), "n_route_to": n_route_to,
            "verdicts_before": dict(before), "verdicts_after": dict(after),
            "n_changed": len(changed), "changed": changed,
            "evidence_sign_kind_after": {str(k): v for k, v in kinds.items()},
            "sign_like_object_present_on_sam3_covered":
                {str(k): v for k, v in presence.items()},
        },
        "corpus_aug120": {
            "source": str(CORPUS.relative_to(REPO)).replace("\\", "/"),
            "n_clips": n_clips,
            "verdicts_before": rt,
            # the new predicate is UNCONDITIONAL not_computable on every
            # route_to clip, so the after-distribution is arithmetic on the
            # same denominators, not an inference
            "verdicts_after": {"not_computable": sum(rt.values())},
            "grounded_rate_before": f"{rt.get('grounded', 0)}/{n_clips}",
            "grounded_rate_after": f"0/{n_clips}",
            "sign_like_object_present_true":
                f"{rt.get('grounded', 0)}/{n_clips}",
            "evidence_sign_kinds": corpus["route_to_sign_kinds"],
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["sample"], indent=1))
    print(json.dumps(out["corpus_aug120"], indent=1))
    print(f"WROTE {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
