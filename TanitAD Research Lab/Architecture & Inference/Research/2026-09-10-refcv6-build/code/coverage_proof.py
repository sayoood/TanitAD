#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv6 -- ⛔ PROVE THE COVERAGE GATE REFUSES THE 182-CLIP DISASTER.

    python coverage_proof.py --json raw/coverage_proof.json

`prelaunch_gate.py`'s C4 CLAIMS it refuses a join that shares too few clips with
the v7.2 label release. That claim is worth nothing until a join at the disaster's
exact shape has been fed to it and observed to FAIL -- and until a join at the real
shape has been observed to PASS, so the gate is not simply refusing everything.

# The failure this reproduces

⛔ A join sharing only **182** clips with v7.2 is **182 / 4,572 = 3.98 %** coverage.
The arm would have trained on ~4 % of its intended supervision and read as *"the
lever does not help"* -- **a MANUFACTURED NEGATIVE**, which is worse than a crash
because it looks like a result rather than a fault.

# The reference the PASS arm must reproduce

The B1 TRAIN agent join covers **4,427 / 4,572 clips = 0.9683** (`D-B1TRAIN-JOIN-1`).
⭐ The PASS fixture is built to that exact shape, so its computed coverage is an
INDEPENDENT reproduction of a published figure rather than a number this tool
invented -- if the arithmetic here were wrong, it would not land on 0.9683.

⚠️ These are SYNTHETIC fixtures at the real SHAPES. They exercise the gate's
arithmetic and its refusal, not the real corpus's contents.
"""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
import os
import shutil
import tempfile

from prelaunch_gate import check_label_join_coverage, DEFAULT_MIN_COVERAGE

N_CORPUS = 4572      # v7.2 TRAIN label release: one record per clip
N_B1_JOIN = 4427     # the real B1 TRAIN join
N_DISASTER = 182     # the incident


def build(tmp: str):
    clips = [f"clip_{i:06d}" for i in range(N_CORPUS)]
    labels = os.path.join(tmp, "labels.jsonl.gz")
    with gzip.open(labels, "wt", encoding="utf-8") as fh:
        for c in clips:
            fh.write(json.dumps({"clip_id": c, "g_tac": {}}) + "\n")

    def join(name, n):
        p = os.path.join(tmp, name)
        with lzma.open(p, "wt", encoding="utf-8") as fh:
            for c in clips[:n]:
                for f in range(3):          # 3 labelled frames per clip
                    fh.write(json.dumps(
                        {"clip_id": c, "frame_idx": f, "agents": []}) + "\n")
        return p

    return labels, join("join_b1.jsonl.xz", N_B1_JOIN), \
        join("join_182.jsonl.xz", N_DISASTER)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="refcv6_cov_")
    try:
        labels, good, bad = build(tmp)
        r_good = check_label_join_coverage(labels, good, a.min_coverage)
        r_bad = check_label_join_coverage(labels, bad, a.min_coverage)

        # ⛔ A gate that refuses everything is not a gate. BOTH directions must
        #    read as designed, or this proof has established nothing.
        ok = (r_good["verdict"] == "PASS" and r_bad["verdict"] == "FAIL")
        # ⭐ and the PASS arm must land on the PUBLISHED figure, not merely pass.
        reproduces = abs(r_good.get("coverage", 0) - 0.9683) < 5e-4

        out = {
            "tool": "coverage_proof.py",
            "min_coverage": a.min_coverage,
            "fixtures": {"n_corpus_clips": N_CORPUS,
                         "n_b1_join_clips": N_B1_JOIN,
                         "n_disaster_clips": N_DISASTER,
                         "note": "SYNTHETIC fixtures at the REAL shapes"},
            "pass_arm": {k: r_good.get(k) for k in
                         ("verdict", "coverage", "n_intersection",
                          "n_label_clips", "n_join_clips", "why")},
            "fail_arm": {k: r_bad.get(k) for k in
                         ("verdict", "coverage", "n_intersection",
                          "n_label_clips", "n_join_clips", "why")},
            "reproduces_published_0_9683": reproduces,
            "verdict": "PASS" if (ok and reproduces) else "REFUSE",
            "why": ("the gate PASSES the real B1 shape at the published 0.9683 and "
                    "REFUSES the 182-clip disaster at 0.0398"
                    if (ok and reproduces) else
                    "⛔ the gate did not read as designed in both directions"),
        }
        if a.json:
            os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".",
                        exist_ok=True)
            with open(a.json, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
            print(f"[coverage] wrote {a.json}")
        for name, r in (("B1 shape (4,427)", r_good),
                        ("disaster  (182)", r_bad)):
            print(f"  {name:<20} {r['verdict']:<6} "
                  f"{r['n_intersection']}/{r['n_label_clips']} = {r['coverage']}")
        print(f"[coverage] reproduces published 0.9683 = {reproduces}")
        print(f"[coverage] verdict = {out['verdict']}")
        return 0 if out["verdict"] == "PASS" else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
