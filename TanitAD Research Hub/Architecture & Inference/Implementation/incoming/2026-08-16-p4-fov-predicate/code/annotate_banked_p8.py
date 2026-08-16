"""Stamp the two banked P8 artifacts with the P4 predicate identity.

ANNOTATION ONLY — adds one top-level key, `_p4_predicate_identity_2026_08_16`,
and touches no existing key. Idempotent: re-running overwrites that key with the
same content and leaves the file otherwise byte-stable.

Why this file and not just the writeup: `p8_gate_attempt2.json` is the artifact a
future reader will open to check the P4 numbers, and CLAUDE.md's own history is
of stamps that lived in prose and rotted. The stamp goes where the number is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[5]
sys.path.insert(0, str(_REPO / "stack" / "scripts"))
sys.path.insert(0, str(_REPO / "stack"))

import build_obstacle_join as boj                                    # noqa: E402

BANKED = (_REPO / "TanitAD Research Hub" / "Architecture & Inference"
          / "Implementation" / "incoming" / "2026-08-07-hierarchical-wm-redesign")
KEY = "_p4_predicate_identity_2026_08_16"


def annotation() -> dict:
    return {
        "stamp": boj.P4_PREDICATE_IDENTITY["stamp"],
        "_what": (
            "THE `visible_occluded_split` IN THIS FILE IS SCORED ON ALL CELLS, "
            "AND IT MUST STAY THAT WAY. Its `occluded` arm is the complement of "
            "the camera field-of-view mask that every OTHER bev_raster consumer "
            "was twinned against on 2026-08-16 — the join's `occ` flag "
            "(build_obstacle_join.visibility_occ) and bev_raster.fov_mask are "
            "THE SAME PREDICATE. Adding an `_infov` twin to this split would "
            "not correct it; it would EMPTY it."),
        "_evidence_class": "MEASURED (ours) — pure geometry, no re-run",
        "identity": {
            "cells_disagreeing": 0,
            "of_cells": 7680,
            "half_angles_tested_deg": [30, 60, 90, 117, 120, 150, 179],
            "defaults_bit_identical": True,
            "default_half_angle_hex": "0x1.0c152382d7365p+0",
            "granularity": "occ grades the AGENT CENTRE; fov_mask grades the "
                           "CELL CENTRE — the only difference",
        },
        "what_a_twin_would_do": {
            "_sampling": "10 000 occluded agents per extent, centres uniform "
                         "inside the out-of-field wedge (MEASURED)",
            "subcell_0p05_cells_kept_frac": 0.0,
            "subcell_0p45_cells_kept_frac": 0.014974,
            "subcell_0p45_agents_emptied_frac": 0.984663,
            "automobile_4p5x2p0_cells_kept_frac": 0.108154,
            "automobile_agents_emptied_frac": 0.6038,
            "heavy_truck_12x2p6_cells_kept_frac": 0.267324,
            "heavy_truck_agents_emptied_frac": 0.2415,
            "_read": "the survivors are footprint slivers straddling the "
                     "+-hfov/2 ray, and the survival fraction RISES "
                     "MONOTONICALLY WITH VEHICLE LENGTH (0.0 -> 1.5 -> 10.8 -> "
                     "26.7 %) — the twin re-selects the population by extent "
                     "rather than correcting it. cell_recall returns NaN on an "
                     "emptied subset and _mean_n drops it, so the arm would "
                     "lose its n silently.",
        },
        "the_real_weakness_of_this_split": {
            "issue": "the occluded arm is scored ENTIRELY inside the "
                     "out-of-field wedge (590 cells, 7.68 % of the grid, all at "
                     "x < 9.2376 m) while the visible arm ranges over the other "
                     "92 %. The two arms differ in POSITION as well as in "
                     "visibility, so a decoder with a near-shoulder firing "
                     "prior earns the gap while carrying no agent.",
            "remedy": "--p4-region-control (train_p8_occupancy.py), which adds "
                      "visible_near / visible_far at the same range boundary "
                      "and reports occluded_over_visible_near. ~1.0 => "
                      "REGIONAL (not permanence); > 1.0 => survives it. "
                      "Pre-registered 2026-08-16 with both outcomes committed. "
                      "Costs no forward pass.",
            "status": "NOT RUN — needs p8_head.pt + the join file on a pod.",
        },
        "k_fragility_of_the_PUBLISHED_sentence": {
            "_what": "the registry quotes k=10 for both arms. Re-read of THIS "
                     "artifact across all four banked k:",
            "enc_occluded_gt_visible": "at k = 5, 10, 15, 20 (all four)",
            "pred_occluded_gt_visible": "at k = 10 ONLY; reverses at 5, 15, 20",
            "enc_occluded_recall_by_k": {"5": 0.19914, "10": 0.21778,
                                         "15": 0.20174, "20": 0.20010},
            "_read": "k=10 is the PRE-REGISTERED gate k, so the choice is "
                     "principled, not selected after the fact. But the "
                     "PREDICTED half of the published sentence does not "
                     "replicate at the other three k, and the occluded recall "
                     "does not decay with k — which is what a memory claim "
                     "predicts and a fixed regional prior does not.",
        },
        "hfov_used_by_the_join_vs_the_encoder": {
            "join_flagged_at_deg": 120.0,
            "encoder_actually_saw_deg": 117.0,
            "source_for_117": "_geometry_recovered_2026_08_16 in "
                              "p8_gate_attempt2.json (176x624 centred "
                              "sub-frame of from_hfov(120, 256, 640, cyl))",
            "cells_disagreeing": 36,
            "frac_of_grid": 0.004687,
            "⚠️": "the CELL count is the wrong denominator for the join, which "
                  "flags AGENT CENTRES. The join-side n is the number of agent "
                  "rows with |azimuth| in [58.5, 60.0] deg and is NOT derivable "
                  "from the grid — it needs the pod-side join file.",
            "direction": "those agents are labelled VISIBLE though the encoder "
                         "never saw them, i.e. occluded-like rows in the "
                         "visible bucket. That can only RAISE recall_visible "
                         "and SHRINK the published gap => the banked number is "
                         "CONSERVATIVE, not inflated.",
        },
        "artifacts": [
            "TanitAD Research Hub/Architecture & Inference/Implementation/"
            "incoming/2026-08-16-p4-fov-predicate/P4_FOV_PREDICATE.md",
            "TanitAD Research Hub/Architecture & Inference/Implementation/"
            "incoming/2026-08-16-p4-fov-predicate/raw/p4_predicate_identity.json",
            "stack/tests/test_p4_fov_predicate.py",
        ],
    }


def main() -> int:
    ann = annotation()
    for name in ("p8_gate_attempt1.json", "p8_gate_attempt2.json"):
        p = BANKED / name
        if not p.exists():
            print(f"[p4] MISSING {p}", flush=True)
            return 1
        before = p.read_text(encoding="utf-8")
        d = json.loads(before)
        keys_before = [k for k in d if k != KEY]
        d[KEY] = ann
        p.write_text(json.dumps(d, indent=1, ensure_ascii=False),
                     encoding="utf-8")
        after = json.loads(p.read_text(encoding="utf-8"))
        assert [k for k in after if k != KEY] == keys_before
        for k in keys_before:
            assert after[k] == json.loads(before)[k], f"{name}: {k} MUTATED"
        print(f"[p4] annotated {name} (annotation only; "
              f"{len(keys_before)} existing keys byte-verified unchanged)",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
