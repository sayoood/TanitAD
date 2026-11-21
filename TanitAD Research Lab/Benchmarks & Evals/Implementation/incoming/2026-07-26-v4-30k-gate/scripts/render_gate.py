"""Render the flagship-v4 30k gate through run_gate.py check.

WHY A PROJECTION IS NEEDED (a real defect, not a workaround of convenience).
`GateCard` is a dataclass and `cmd_check` does `GateCard(**json.loads(card))`, so
the REGISTERED card does not load: 11 of its keys are unknown to the dataclass
  co_primary, goal_provenance, goal_provenance_note, preflight_checks,
  primary_role_note, reference_ade_0_2s, reference_note,
  registered_before_checkpoint_exists, registration_note, required_reporting,
  secondary_void
and it supplies `co_primary` as a NESTED dict where the tool expects FLAT
`co_primary_*` fields. MEASURED: `run_gate.py check --card <the registered card>`
dies with
  TypeError: GateCard.__init__() got an unexpected keyword argument
             'registered_before_checkpoint_exists'

NOTHING IS RE-TUNED HERE. Both projections below copy the card's own values
verbatim; no threshold, direction, or role is invented or altered. Two are
rendered because the tool cannot express the card's actual configuration
(`co_primary.role = REPORT_ONLY_THIS_GATE`, i.e. registered but NOT in the kill
conjunction), and the difference between them is itself the finding.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

sys.path.insert(0, "/root/TanitAD/stack/scripts")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/taniteval")
import run_gate  # noqa: E402

CARD = "/workspace/_v4gate/flagship-v4-30k.card.json"
card = json.load(open(CARD))
accepted = {f.name for f in dataclasses.fields(run_gate.GateCard)}

# ---- projection A: card fields only; co_primary NOT mapped ---------------- #
# The tool then sees has_co_primary=False and, per cmd_check, lets the DEMOTED
# ade_0_2s back into the kill conjunction -- which the card forbids. Rendered to
# show what the tool does with the card as literally projected.
projA = {k: v for k, v in card.items() if k in accepted}

# ---- projection B: co_primary mapped onto the flat fields ---------------- #
# Structurally faithful to the card's co-primary registration. NO threshold is
# invented: the card registers none (`becomes_kill_criterion_at: the next
# v4-line gate`), so co_primary_threshold stays None.
cp = card["co_primary"]
projB = dict(projA)
projB.update(co_primary_metric=cp["metric"],
             co_primary_horizon_K=cp["horizon_K"],
             co_primary_corridor_m=cp["corridor_half_width_m"],
             co_primary_threshold=None,
             co_primary_source=cp.get("surface", ""))

SEC = ["wm_canary_ade_2s=1.1409059762954712",
       "oracle_in_fan=0.23301841780357274",
       "miss_at_2m=0.2123",
       "seam_norm_ratio_max=0.1208",
       "encoder_touching_levers=2"]
# speed_benefit_recovered_frac and deploy_tick_p99_ms are DELIBERATELY not
# supplied: no emitter exists for either (v4_diagnostics records both as null
# with the reason). Supplying a guess would be the forking-paths failure.

for tag, proj in (("A_no_coprimary", projA), ("B_coprimary_registered", projB)):
    p = f"/workspace/_v4gate/_projected_card_{tag}.json"
    json.dump(proj, open(p, "w"), indent=2)
    cmd = [sys.executable, "-u", "/root/TanitAD/stack/scripts/run_gate.py", "check",
           "--card", p,
           "--log", "/workspace/_v4gate/v4fs_train_log.jsonl",
           "--reference-log", "/workspace/_v4gate/v1-speedjerk_train_log.jsonl",
           "--eval-json", "/workspace/_v4gate/results/"
                          "flagship-v4-fromscratch-30k-oracle.json",
           "--json", f"/workspace/_v4gate/GATE_30K_verdict_{tag}.json",
           "--secondary-value", *SEC]
    print("\n" + "#" * 78)
    print(f"# PROJECTION {tag}")
    print("#" * 78)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-6000:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2500:])
    print(f"[exit {r.returncode}]")
