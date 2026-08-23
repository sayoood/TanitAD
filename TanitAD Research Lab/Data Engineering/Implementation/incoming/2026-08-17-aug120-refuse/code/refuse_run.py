"""The aug120 RE-FUSE — four arms, so the delta is ATTRIBUTABLE.

⛔ WHY FOUR ARMS AND NOT ONE. Three things change between the banked v1 fused
records and the deliverable: the FUSER CODE (five defect fixes), the SAM3
CORPUS (115 named-partials become real v2 perception), and the STRATEGIC LEG
(an Engine A sidecar that the v1 run did not have, which flips g_str/a_str from
the VLM-primary fallback to geometry-primary). A single before/after would
report their SUM and attribute it to whichever one the reader had in mind —
that is the `--v2` conflation defect this programme has already paid for once.

  A0  banked v1 records          old code · v1 SAM3 (86) · no engine_a
  A1  code only                  HEAD     · v1 SAM3 (86) · no engine_a
  A2  + the v2 corpus            HEAD     · v2+v1 (201)  · no engine_a
  A3  + the strategic leg        HEAD     · v2+v1 (201)  · engine_a   <- SHIP

A0 is read, never recomputed: it is the primary source for the PI's verdicts
and for SAM3_CONCEPT_RELIABILITY.md, and it is not overwritten anywhere here.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

MISSING_REASON = "AUG120_SAM3_STAGE_GAP"


def run(stack, out, v2_json, sam3, ego, records, engine_a=None,
        missing_ok=None):
    sys.path.insert(0, stack)
    sys.path.insert(0, os.path.join(stack, "scripts"))
    from ph1_fuse import main as fuse_main
    if os.path.isdir(out):
        shutil.rmtree(out)                     # arms never resume onto stale
    argv = ["--v2-json", v2_json, "--sam3", sam3, "--ego-root", ego,
            "--records", records, "--out", out]
    if engine_a:
        argv += ["--engine-a", engine_a]
    if missing_ok:
        argv += ["--missing-sam3-ok", missing_ok]
    rc = fuse_main(argv)
    assert rc == 0, f"fuse rc={rc} for {out}"
    return json.load(open(os.path.join(out, "_summary.json"),
                          encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--aug120", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--arms", default="A1,A2,A3")
    a = ap.parse_args(argv)

    merged = os.path.join(a.aug120, "merged")
    v2_json = os.path.join(merged, "ph0_v2.json")
    sam3_v1 = os.path.join(merged, "sam3.json")
    sam3_all = os.path.join(a.work, "sam3_refuse")
    ego = os.path.join(a.aug120, "ego")
    records = os.path.join(a.aug120, "aux", "records.parquet")
    want = set(a.arms.split(","))

    summaries = {}
    if "A1" in want:
        summaries["A1"] = run(a.stack, os.path.join(a.work, "fused_A1"),
                              v2_json, sam3_v1, ego, records,
                              missing_ok=MISSING_REASON)
        print("[A1] " + json.dumps(summaries["A1"]), flush=True)
    if "A2" in want:
        summaries["A2"] = run(a.stack, os.path.join(a.work, "fused_A2"),
                              v2_json, sam3_all, ego, records)
        print("[A2] " + json.dumps(summaries["A2"]), flush=True)
    if "A3" in want:
        summaries["A3"] = run(a.stack, os.path.join(a.work, "fused_aug120_v2"),
                              v2_json, sam3_all, ego, records,
                              engine_a=a.engine_a)
        print("[A3] " + json.dumps(summaries["A3"]), flush=True)
    json.dump(summaries, open(os.path.join(a.work, "arm_summaries.json"), "w"),
              indent=1)
    print("REFUSE_RUN_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
