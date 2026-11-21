#!/usr/bin/env python3
"""Are navhard's STAGE-1 camera frames inside the verified OpenScene *test* camera tarballs?

⛔ WHY (W7, 2026-09-20, after an orchestrator correction). My first census measured the UNPACKED
disk and concluded 0/5,400 — true, and its scope was too wide: I wrote "not on this box" when what I
had measured was "not UNPACKED on this box". The orchestrator measured the archives and found the
files there. This probe re-establishes that on the ARCHIVE side, from my own export, so the
correction rests on a measurement of mine rather than on a relay.

⭐ IT TESTS A WHOLE LOG, NOT A SAMPLE OF FILES. The orchestrator's positive test was 3 files of one
log (a mechanism plus one instance). The mechanism being claimed is *"the archives store WHOLE
logs"*, and the discriminating test of that is: for every navhard log present in this shard, are
**ALL** of its stage-1 files present? A partial log would falsify it, and a partial extraction is
precisely the poisoned-bank failure (an arm scored on fewer frames, silently).

⚠️ It reads a tar LISTING, never an extraction, and touches no image bytes.

    tar --force-local -tzf <shard>.tgz > listing.txt      # rc read from a FILE, never through a pipe
    python probe_stage1_in_archives.py --listing listing.txt --inputs <export json> --out raw/…json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

CAMS = ("cam_l0", "cam_f0", "cam_r0")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", required=True, help="output of `tar -tzf <shard>`")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    toks = json.load(open(a.inputs, encoding="utf-8"))["tokens"]

    # what navhard stage 1 (and stage 2, the control) actually wants, as (log, CAM, file)
    want = {1: {}, 2: {}}
    for t, r in toks.items():
        st = r["stage"]
        for c in CAMS:
            for rel in r["cams"][c]:
                parts = rel.replace("\\", "/").split("/")
                want[st].setdefault(parts[0], set()).add((parts[1].upper(), parts[2]))

    have: dict = {}
    n_entries = 0
    for line in open(a.listing, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line.endswith(".jpg"):
            continue
        n_entries += 1
        p = line.split("/")
        if len(p) < 4:
            continue
        log, cam, fn = p[-3], p[-2].upper(), p[-1]
        have.setdefault(log, set()).add((cam, fn))

    def cover(stage):
        logs_here = sorted(set(want[stage]) & set(have))
        rows, full, partial = [], 0, 0
        n_w = n_f = 0
        for lg in logs_here:
            w, h = want[stage][lg], have[lg]
            hit = len(w & h)
            n_w += len(w)
            n_f += hit
            complete = hit == len(w)
            full += complete
            partial += (not complete)
            rows.append({"log": lg, "wanted": len(w), "found": hit, "complete": complete,
                         "missing_examples": sorted(x[0] + "/" + x[1] for x in (w - h))[:3]})
        return {"n_logs_of_this_split_in_this_shard": len(logs_here),
                "n_logs_COMPLETE": full, "n_logs_PARTIAL": partial,
                "files_wanted": n_w, "files_found": n_f,
                "fraction": round(n_f / n_w, 6) if n_w else None,
                "per_log": rows[:40]}

    s1, s2 = cover(1), cover(2)
    rep = {"_what": ("does the OpenScene test camera archive contain navhard's STAGE-1 frames? "
                     "Tested per LOG (all-or-nothing), from a tar LISTING — no extraction."),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "shard": a.shard or a.listing, "listing_jpg_entries": n_entries,
           "archive_layout_example": "openscene-v1.1/sensor_blobs/test/<log>/CAM_F0/<file>.jpg",
           "stage_1": s1,
           "stage_2_control": {**s2, "role": ("navhard stage 2 uses the SYNTHETIC frames, which are "
                                              "NOT in this archive — this control is EXPECTED to be "
                                              "empty/low, and it is what shows the two stages index "
                                              "genuinely different corpora")}}
    if s1["n_logs_of_this_split_in_this_shard"] == 0:
        rep["verdict"] = ("INCONCLUSIVE for this shard: none of navhard's 76 logs appears in it, so "
                          "it says nothing either way")
    elif s1["n_logs_PARTIAL"] == 0:
        rep["verdict"] = (f"SUPPORTED: all {s1['n_logs_COMPLETE']} navhard log(s) present in this "
                          f"shard are COMPLETE for stage 1 ({s1['files_found']}/{s1['files_wanted']} "
                          "files) — consistent with 'the archives store whole logs'")
    else:
        rep["verdict"] = (f"⛔ REFUTED for this shard: {s1['n_logs_PARTIAL']} navhard log(s) are "
                          "PARTIAL — the archives do NOT simply store whole logs, and an extraction "
                          "must assert per-file presence before any stage-1 arm is scored")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in ("verdict", "listing_jpg_entries")}, indent=1))
    print(json.dumps({k: v for k, v in s1.items() if k != "per_log"}, indent=1))
    print("stage-2 control: " + json.dumps({k: v for k, v in s2.items() if k != "per_log"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
