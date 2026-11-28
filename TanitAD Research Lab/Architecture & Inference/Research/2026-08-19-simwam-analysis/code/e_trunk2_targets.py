"""E-TRUNK-2 targets — ENVIRONMENT properties, from obstacle.offline cuboids.

⭐ THE QUESTION THIS SERVES, and it is NOT the one E-TRUNK-1 answers.
E-TRUNK-1 asks whether a predictor beats persistence at predicting the FUTURE
FIELD — that is DYNAMICS PREDICTABILITY, and it is dominated by the scale and
variance structure of the representation. The PI's question is different and
prior: **does the representation CONTAIN decodable environment information?**
A constant representation is perfectly predictable and carries nothing; a rich
one can be hard to predict precisely BECAUSE it carries a lot.

⇒ The instrument for the PI's question is a DECODABILITY probe (P1's family):
freeze the representation, fit a LINEAR readout to a known environment property,
and compare against a floor, a leak control and a reference.

TARGETS — all derived from `lead130_agents.jsonl`, the pod-side join of
`obstacle.offline` (3D agent tracks, 10 classes, all dynamic agents). Cuboids
are in the EGO FRAME in metres: +cx forward, +cy left.

⛔ THESE ARE ENVIRONMENT LABELS, NOT EGO. That is the whole point — an ego-only
control (C-EGO) is scored alongside precisely because car-following statistics
let ego speed predict lead gap without any perception at all. A trunk that only
matches C-EGO has demonstrated nothing about the environment.

⚠️ LABELS MAY USE EGO / PRIVILEGED CHANNELS (PI 2026-08-03) — this is offline
label derivation, never an inference path.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
JOIN = SP / "sp2/lead130_agents.jsonl"
OUT = SP / "sp2/e_trunk2_targets.jsonl"

#: an in-lane lead: ahead of the ego, within half a lane either side.
LANE_HALF_M = 2.0
#: adjacent-lane band, used for the lateral-occupancy targets.
ADJ_NEAR_M, ADJ_FAR_M = 2.0, 6.0
#: how far ahead/behind an adjacent-lane agent still counts as "beside us".
ADJ_LONG_M = 20.0
#: vulnerable road users — the classes a planner must treat differently.
VRU = frozenset({"person", "rider", "stroller"})


def targets_for(agents: list[dict]) -> dict:
    """-> the environment target row for one frame. Absent targets are None,
    never 0 — a missing lead is NOT a lead at zero metres."""
    ahead = [a for a in agents if a["cx"] > 0.0]
    in_lane = [a for a in ahead if abs(a["cy"]) <= LANE_HALF_M]
    lead = min(in_lane, key=lambda a: a["cx"]) if in_lane else None

    def band(sign: int) -> bool:
        return any(ADJ_NEAR_M < sign * a["cy"] <= ADJ_FAR_M
                   and abs(a["cx"]) <= ADJ_LONG_M for a in agents)

    near = min((math.hypot(a["cx"], a["cy"]) for a in agents), default=None)
    return {
        # ⭐ the headline environment target: metric distance to the in-lane lead.
        # This is the LONGITUDINAL family's headway, which is where 88.7 % of the
        # oracle gap lives — so it is the target that matters, not a proxy.
        "lead_gap_m": None if lead is None else round(float(lead["cx"]), 4),
        "lead_present": int(lead is not None),
        "lead_is_vru": None if lead is None else int(lead["cls"] in VRU),
        "n_agents": len(agents),
        # log1p: counts are heavy-tailed (mean 53, max 256) and a linear probe on
        # the raw count would be fit almost entirely by the tail.
        "n_agents_log": round(math.log1p(len(agents)), 4),
        "nearest_any_m": None if near is None else round(float(near), 4),
        "left_occupied": int(band(+1)),
        "right_occupied": int(band(-1)),
        "vru_ahead": int(any(a["cls"] in VRU and abs(a["cy"]) <= 4.0
                             for a in ahead)),
        "occluded_frac": (round(sum(int(a.get("occ", 0) or 0) > 0
                                    for a in agents) / len(agents), 4)
                          if agents else None),
    }


def main() -> None:
    n, kept = 0, 0
    counts: dict[str, int] = {}
    with OUT.open("w", encoding="utf-8") as fh:
        for line in JOIN.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            t = targets_for(r.get("agents") or [])
            for k, v in t.items():
                if v is not None:
                    counts[k] = counts.get(k, 0) + 1
            fh.write(json.dumps({"clip_id": r["clip_id"],
                                 "frame_idx": int(r["frame_idx"]),
                                 **t}) + "\n")
            kept += 1
    print(f"rows in {n}  rows out {kept}  -> {OUT}")
    print("per-target coverage (non-null):")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>6} / {kept}  ({100 * v / kept:5.1f}%)  {k}")


if __name__ == "__main__":
    main()
