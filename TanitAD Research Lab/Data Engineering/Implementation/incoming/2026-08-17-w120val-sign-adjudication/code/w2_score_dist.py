"""STEP 2 — per-concept score distributions for each leg, using the STUDY'S OWN
statistics code.

⛔ **NOT A SECOND IMPLEMENTATION.** This explodes each single-file leg into the
per-clip cache layout `r2_score_dist.py` already expects and then calls
`r2_score_dist.main()`. Exactly one thing is substituted, and it is declared in
`PREREG.md` §2 as protocol delta 2:

  `r2.load()` drops any record with no `liveness` block — the aug120 C77
  staleness filter. The production engine (`ph0_prod4`, `ph0_pilot50`) PREDATES
  the liveness probe, so that gate would drop **100 %** of these records and
  produce an empty distribution that exits 0. It is replaced by "the record has
  frames". ⚠️ The justification is MEASURED, not assumed: `w1_pull_records.py`
  reports `n_errors = 0` on both legs, so there is no stale subset to exclude.

⚠️ A distribution is DESCRIPTIVE. It cannot say whether a box contains a sign;
asking the detector's own score whether the detector was right is the circularity
this whole line of work exists to catch. Precision comes from step 4 only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
STUDY = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                     "Implementation", "incoming",
                     "2026-08-16-sam3-concept-reliability", "code")
CACHE = (r"C:\Users\Admin\AppData\Local\Temp\claude"
         r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
         r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\w120sign\records")


def explode(leg: str, cache: str) -> str:
    """Single-file leg -> the per-clip layout r2 expects. Idempotent."""
    out = os.path.join(cache, leg + "_clips")
    blob = json.load(open(os.path.join(cache, f"{leg}.json"), encoding="utf-8"))
    recs = blob["clips"]
    os.makedirs(out, exist_ok=True)
    if len(os.listdir(out)) != len(recs):
        for r in recs:
            json.dump(r, open(os.path.join(out, f"{r['clip_id']}.json"), "w",
                              encoding="utf-8"))
    return out


def load_no_liveness_gate(cache: str) -> dict:
    """r2.load with the C77 staleness gate replaced — PREREG.md §2 delta 2."""
    recs = {}
    for fn in sorted(os.listdir(cache)):
        if not fn.endswith(".json"):
            continue
        r = json.load(open(os.path.join(cache, fn), encoding="utf-8"))
        if not (r.get("frames")):
            continue
        recs[r["clip_id"]] = r
    assert recs, f"no records under {cache} — refusing to write an empty dist"
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("w2_score_dist")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--legs", default="w120val600,pilot50")
    a = ap.parse_args(argv)

    sys.path.insert(0, STUDY)
    import r2_score_dist as r2                       # the study's own code
    r2.load = load_no_liveness_gate                  # the ONE substitution
    for leg in a.legs.split(","):
        cdir = explode(leg, a.cache)
        print(f"== {leg}")
        rc = r2.main(["--out", os.path.join(a.out_dir,
                                            f"score_distribution_{leg}.json"),
                      "--cache", cdir])
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
