#!/usr/bin/env python3
"""RUNG 1 — the R1 PLANNER row. ~12 GPU-min on an idle A40 (pod2 only).

WHY THIS EXISTS. Rung 1 swept the ACTION-FILTER axis and confirmed
spectacularly, but it explicitly did NOT run the ladder's R1 PLANNER row and
recorded a prediction for it instead (`TBLIND_RUNG1.md` §7.2, §9 limitation 2):

    "v1's tactical planner will land between 6 and 12 s of deployable T_blind ...
     If it lands BELOW 6.4 s, the planner is adding a failure mode the gain
     argument does not explain and that is the finding."

This job measures it. `closedloop.wp_to_control` is IMPORTED by
`blindimag.blind_rollout` — never copied — so the arm is v1's deployed
controller, and `bi_run.stage_sweep` is reused VERBATIM: the arm list is
replaced, the machinery (encode / window build / batching / rollout / dump) is
not, and `bi_run._run_arms` gained ONE branch that is inert for every non-planner
arm.

⛔ WINDOW-SET IDENTITY GATE. The four arms that already exist in the committed
dumps are re-rolled as anchors; the analysis blocks unless their dense `de`
reproduces within 1e-4 m and the `eid`/`t0` ordering is identical.

⛔ PLUMBING SELF-TEST, BOTH DIRECTIONS. (a) the planner arm must differ from BOTH
`own_kinematic` and `hold_last` by a non-trivial margin — an arm silently equal
to either would produce a flat, confident, wrong curve; (b) `wp_to_control` is
the closedloop function itself, pinned on a CPU fixture in
`taniteval/tests/test_blindimag.py` against the deployed controller's own output.

⛔ NO CADENCE KNOB IS PLACED ON THE PLANNER. Rung 1 MEASURED that reducing the
action-update frequency is catastrophic (9 steps vs a 25-step baseline) because
zero-order-holding a SAMPLE of a zero-mean oscillation removes its cancellation.
Running `wp_to_control` at a reduced re-plan cadence is exactly that refuted
intervention and is deliberately absent.

Host: **pod2 only** (A40, verified idle 0 MiB / 0 %). pod1 (v2corpus training),
pod3 (situation-classifier build) and the eval pod are never touched; the val
cache is read only.

Usage (pod2):
    PYTHONPATH=/root/bi:/root/taniteval:/root/TanitAD/stack:/root/TanitAD/stack/scripts \
    OMP_NUM_THREADS=8 python3 tb_r1_planner_sweep.py \
        --out /root/tbr1p/perwindow --episodes 600 --kmax 185
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _p in ("/root/bi", "/root/taniteval", "/root/TanitAD/stack",
           "/root/TanitAD/stack/scripts"):
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)

import bi_run as R                                    # noqa: E402

STR = "str"                                           # the calibrated readout

#: (name, state_source, action_source, update_speed_channel, readout_level)
#: Fixed by PRE_REGISTRATION.md §B.2 BEFORE any number existed. No arm is added
#: after a result is seen.
ARMS: list[tuple] = []


def _pair(tag: str, spec: str, *, upd: bool = False, a_only: bool = False):
    """One arm -> its (a) imagination arm and its (b) MATCHED comparator.

    The comparator is matched in the readout AND in the action source: the same
    planner drives the frozen-percept arm. Without that the number is not a
    `T_blind`, it is two marginal curves put side by side.
    """
    ARMS.append((f"a_{tag}", "imagination", spec, upd, STR))
    if not a_only:
        ARMS.append((f"b_{tag}", "frozen_last", spec, upd, STR))


# --- ANCHORS: already committed; they gate window-set identity ---------------- #
ARMS += [
    ("a_imagination__own__roSTR",  "imagination", "own_kinematic", False, STR),
    ("b_frozenlast__own__roSTR",   "frozen_last", "own_kinematic", False, STR),
    ("a_imagination__hold__roSTR", "imagination", "hold_last",     False, STR),
    ("b_frozenlast__hold__roSTR",  "frozen_last", "hold_last",     False, STR),
]

# --- THE PRIMARY: v1's deployed tactical planner + pure-pursuit controller ---- #
_pair("planner", "planner")

# --- reported, never mixed into the primary ---------------------------------- #
_pair("planner_vupd", "planner", upd=True)        # closedloop.build_action's v0
_pair("planner_vdec", "planner|vsrc=decoded")     # which speed the ctrl stands on

# --- DIAGNOSTIC (privileged): the controller fed a PERFECT target ------------- #
_pair("planner_gtlook", "planner|look=gt")

#: dense fed actions for the amplitude statistics (§B.1). The planner arms plus
#: the two kinematic references, so the 2.058 m/s² / 46.4 % saturation signature
#: is re-measured in the SAME run rather than inherited.
R.KEEP_FED = ("a_planner", "a_planner_vdec", "a_planner_gtlook",
              "a_imagination__own__roSTR", "a_imagination__hold__roSTR")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="flagship-30k")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--kmax", type=int, default=185)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--enc-batch", type=int, default=32)
    a = ap.parse_args()
    # peek is switched OFF entirely: empty `peek_bases` => no peek arms. The two
    # parsed-but-unused strings must still be parseable.
    a.peek_bases, a.periods, a.oracle_bars = "", "2", "0.5"
    R.SWEEP_ARMS = []
    R.EXTRA_ARMS = ARMS
    print(f"[r1planner] {len(ARMS)} arms at K={a.kmax}; "
          f"fed kept for {len(R.KEEP_FED)}", flush=True)
    import torch
    torch.manual_seed(0)
    return R.stage_sweep(a)


if __name__ == "__main__":
    raise SystemExit(main())
