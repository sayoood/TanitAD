"""Run ``parity.assert_v2_splits_disjoint`` on the two wide v2 caches — and
record the result whichever way it goes.

⚠️ WHY THIS IS ITS OWN STEP. ``assert_v2_parity_cache`` checks each directory
against the manifest and NEVER compares two of them; on the v2 path the train
and val caches are just two paths a launch command supplies. A leaked val clip
does not crash anything — it makes the held-out gate probe a TRAINING episode,
so the gate reports health while the deployable surface decays. An early-stop
that cannot fire is worse than none, because it is believed.

Runs the guard twice on purpose:

  1. **the real pair** — must be disjoint;
  2. ⭐ **a RED control**: the train dir against ITSELF. If the guard is working
     it must refuse that with an overlap equal to the train clip count. A guard
     that never fires is indistinguishable from a guard that cannot fire — the
     defect this whole runbook exists to prevent — so the failing direction is
     demonstrated, not assumed.

🔒 Counts only; clip ids never leave the pod.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tanitad.data import parity

    rec: dict = {"train_dirs": a.train, "val_dirs": a.val}

    # 1. the real pair -------------------------------------------------------
    try:
        rec["disjoint"] = parity.assert_v2_splits_disjoint(
            a.train, a.val, label="v5 wide train/val")
        rec["verdict"] = "DISJOINT"
    except parity.ParityViolation as e:
        rec["verdict"] = "LEAK"
        rec["refusal"] = str(e)

    # 2. the RED control — the guard must refuse a self-pair ------------------
    try:
        parity.assert_v2_splits_disjoint(a.train, a.train,
                                         label="RED-control train-vs-train")
        rec["red_control"] = {
            "fired": False,
            "meaning": "⛔ THE GUARD DID NOT FIRE ON A TOTAL OVERLAP — it "
                       "cannot be trusted on the real pair either."}
    except parity.ParityViolation as e:
        txt = str(e)
        overlap = next((ln.strip() for ln in txt.splitlines()
                        if "overlap" in ln), "")
        rec["red_control"] = {"fired": True, "overlap_line": overlap,
                              "meaning": "the guard refuses a total overlap, so "
                                         "a DISJOINT verdict on the real pair "
                                         "is informative"}

    Path(a.out).write_text(json.dumps(rec, indent=1))
    print("V2_DISJOINT " + json.dumps(rec, indent=1))
    if rec["verdict"] != "DISJOINT" or not rec["red_control"]["fired"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
