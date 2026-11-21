#!/usr/bin/env python3
"""W2 (E3): CRITERIA_REGISTRY.json 2.10.1 -> 2.10.2 — integrate W6's nuScenes
`GATE_not_a_criterion` (its `code/criteria_guard.patch`).

W6's patch carried the registry hunk too, but the registry had moved (2.10.0 -> 2.10.1 while
W6 was writing), so `git apply` refused that file — correctly. The code and test hunks applied
clean and verbatim; this script lands the SAME gate object W6 wrote, on the current registry,
as a proper version bump rather than W6's placeholder "2.10.1-W6-proposed" changelog entry.

⛔ The gate is W6's text, unchanged. W2 added ONE thing to the checker: a SECOND detection
mechanism (the protocol TAG `nuScenes_OL_*`), because a suite record carries no planning block
and one detector is one detector — with its own RED mutation arms in
`tools/tests/test_criteria_check.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
REG = REPO / "products" / "P7-TanitEval" / "CRITERIA_REGISTRY.json"
W6 = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness"


def dump(d) -> bytes:
    return json.dumps(d, indent=1, ensure_ascii=False).replace("\n", "\r\n").encode("utf-8")


def main() -> int:
    raw = REG.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    if dump(d) != raw:
        print("REFUSED: the registry does not round-trip byte-exactly")
        return 2
    if d.get("version") != "2.10.1":
        print(f"REFUSED: expected 2.10.1, found {d.get('version')!r}")
        return 2
    plan = d["benchmarks"]["nuscenes"]["tasks"]["planning_openloop"]
    if "GATE_not_a_criterion" in plan:
        print("REFUSED: the gate is already present")
        return 2
    plan["GATE_not_a_criterion"] = {
        "id": "nusc.plan.not_a_criterion",
        "blocking": True,
        "rule": ("An artifact carrying a nuScenes open-loop PLANNING block must declare "
                 "`claim_bearing: false` and must NOT be an in-scope TanitAD driving artifact. "
                 "claim_bearing true or absent FAILS; nuScenes planning numbers inside a driving "
                 "artifact FAIL even when the flag says false. An honest external-comparability "
                 "record is reported as EXTERNAL_ONLY: never scored, never counted as compliant."),
        "why": ("H-EVAL-6 (SUPPORTED) + D-BENCH-PORT (PI 2026-08-29: SKIP claim-bearing). There is "
                "no official nuScenes planning protocol; the averaging convention alone moves one "
                "checkpoint 0.72 -> 1.22 m and flips the UniAD/VAD ranking; the GT human trajectory "
                "scores 0.36-0.96 % collision; and the field's high-level command is the GT future "
                "thresholded at +/-2 m - our route-echo defect, published as SOTA."),
        "detect_keys": ["benchmark.nuscenes.planning",
                        "benchmark.nuscenes.planning.protocol_tag",
                        "schema == taniteval.nuscenes_planning/1",
                        "a protocol TAG matching nuScenes_OL_* (W2: the second mechanism — a suite "
                        "record carries no planning block)"],
        "keys": ["claim_bearing", "benchmark.nuscenes.planning.protocol_tag",
                 "benchmark.nuscenes.planning.gt_control",
                 "benchmark.nuscenes.planning.command_source",
                 "benchmark.nuscenes.planning.nonstraight"],
        "harness": ("taniteval/adapters/nuscenes_planning.py (W6): both conventions verbatim to the "
                    "pinned reference code, claim_bearing:false enforced by API; "
                    "taniteval/taniteval/bench/plugins/nuscenes_ol.py runs ONE convention per run."),
        "ships_with": ("tools/tests/test_criteria_check.py: W6's EXTERNAL_ONLY arm + two deliberate "
                       "misuse arms + a control, and W2's two TAG-detected misuse arms with their "
                       "own NavSim control"),
        "authored_by": f"W6 (E5), {W6}/code/criteria_guard.patch — integrated verbatim by W2",
    }
    d["version"] = "2.10.2"
    d["changelog"].append({
        "version": "2.10.2", "date": "2026-09-20",
        "by": "W6 (E5), integrated by EvalFlyWheel W2 (E3)",
        "change": ("ADD benchmarks.nuscenes.tasks.planning_openloop.GATE_not_a_criterion and the "
                   "checker's EXTERNAL_ONLY scope: a nuScenes open-loop planning record is reported "
                   "and never scored; claiming one (claim_bearing true or absent), or embedding its "
                   "numbers in a TanitAD driving artifact, is a VIOLATION. W2 added a second "
                   "detection mechanism — a protocol tag matching nuScenes_OL_* — so a suite record "
                   "with no planning block cannot slip through."),
        "why": ("H-EVAL-6 is SUPPORTED and D-BENCH-PORT marks nuScenes-OL SKIP claim-bearing, but "
                "nothing in the machinery could SEE such a record: it read UNKNOWN_SCOPE (surfaced "
                "for a human, uncounted) or — if it carried a four_families / headline.ade_* key — "
                "would be scored as a TanitAD driving eval. A rule enforced only in prose is the "
                "failure class this instrument exists for."),
        "ships_with": ("tools/tests/test_criteria_check.py — W6's EXTERNAL_ONLY arm, two misuse arms "
                       "and a control, plus W2's two tag-detected misuse arms; the code and test "
                       "hunks of W6's patch applied verbatim (`git apply`), only its registry hunk "
                       "was re-landed here because the registry had moved to 2.10.1.")})
    out = dump(d)
    REG.write_bytes(out)
    print(f"wrote {REG} version {d['version']} ({len(raw)} -> {len(out)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
