#!/usr/bin/env python3
"""RUNG 1 — the PLANNER-ACTION sweep. ~35 GPU-min on an idle A40 (pod2 only).

WHY THIS EXISTS. With the calibrated (``str``, k=20) readout adopted by Rung 0,
deployable ``T_blind`` under the model's own actions is **25 steps**; under a
zero-order hold of the last observed action it is **115** — matched comparators,
identical windows (``…/2026-07-26-tblind-ladder/artifacts/rung0c_matched_tblind.json``).
**Merely removing the model from the action loop is worth 4.6x more blind horizon
than letting it choose.** The 90-step gap is a property of the ACTION TENSOR fed
back, not of the weights, so it is attackable with NO RETRAINING.

This job sweeps the filter axis between those two MEASURED endpoints, plus the
mechanism arms that separate compounding drift / clamp saturation / feedback
instability / horizon onset. Every arm is rolled in BOTH ``imagination`` (a) and
``frozen_last`` (b) unless marked a-only, so every ``T_blind`` has a comparator
matched in the readout AND in the action filter.

⛔ WINDOW-SET IDENTITY GATE. Four arms that already exist in the committed dumps
are re-rolled as anchors; ``tb_rung1_analyze.py`` blocks unless their dense ``de``
reproduces within 1e-4 m and the ``eid``/``t0`` ordering is identical.

⛔ PLUMBING SELF-TEST. ``blend=0.0`` must be BIT-IDENTICAL to the unfiltered own
arm and ``blend=1.0`` BIT-IDENTICAL to the hold arm (the blend at alpha=1 reduces
algebraically to the zero-order hold). Both endpoints are rolled as explicit
arms, so a silently-no-op filter cannot survive the analysis.

``bi_run.stage_sweep`` is reused VERBATIM — the arm list is replaced, the
machinery (encode / window build / batching / rollout / dump) is not. The action
filters ride on the ``action_source`` string, which is exactly why no change to
the sweep driver was needed.

Host: **pod2 only** (A40, verified idle: 0 MiB / 0 %). pod1 (training), pod3
(situation-classifier build) and the eval pod are never touched; the val cache is
read only.

Usage (pod2):
    PYTHONPATH=/root/bi:/root/taniteval:/root/TanitAD/stack:/root/TanitAD/stack/scripts \
    OMP_NUM_THREADS=8 python3 tb_rung1_sweep.py \
        --out /root/tbr1/perwindow --episodes 600 --kmax 185
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
#: Fixed by PRE_REGISTRATION.md §4 BEFORE any number existed. No arm is added
#: after a result is seen.
ARMS: list[tuple] = []


def _pair(tag: str, spec: str, *, upd: bool = False, a_only: bool = False):
    """One intervention -> its (a) imagination arm and its (b) matched control."""
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
# --- PLUMBING SELF-TEST: must be BIT-IDENTICAL to the two anchors above ------- #
_pair("selftest__blend0",   "own_kinematic|blend=0.0", a_only=True)
_pair("selftest__blend1",   "own_kinematic|blend=1.0", a_only=True)
_pair("selftest__every1",   "own_kinematic|every=1",   a_only=True)

# --- C  the BLEND CURVE — the priority deliverable (interpolates the endpoints) #
for _a in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875):
    _pair(f"blend{_a:g}", f"own_kinematic|blend={_a:g}")

# --- D  channel decomposition (DIAGNOSTIC — one channel amputated) ------------ #
for _c in ("steer", "accel"):
    _pair(f"chan{_c}", f"own_kinematic|chan={_c}")

# --- E  clipping to a tighter physically-plausible band ---------------------- #
for _s in (0.02, 0.005, 0.0):
    _pair(f"steerclip{_s:g}", f"own_kinematic|steer_clip={_s:g}")
for _c in (1.0, 0.3, 0.0):
    _pair(f"accelclip{_c:g}", f"own_kinematic|accel_clip={_c:g}")

# --- F  temporal smoothing (first-order low-pass on the fed action) ---------- #
for _b in (0.5, 0.8, 0.95):
    _pair(f"ema{_b:g}", f"own_kinematic|ema={_b:g}")

# --- G  reduced action-update frequency -------------------------------------- #
for _m in (2, 5, 20):
    _pair(f"every{_m}", f"own_kinematic|every={_m}")

# --- H  onset / switch time (DIAGNOSTIC, a-only) ----------------------------- #
for _m in (5, 10, 20, 40):
    _pair(f"ownbefore{_m}", f"own_kinematic|own_before={_m}", a_only=True)
    _pair(f"ownafter{_m}", f"own_kinematic|own_after={_m}", a_only=True)

# --- I  convention control + the speed channel ------------------------------- #
_pair("gtkin", "gt_kinematic")                       # DIAGNOSTIC — privileged
_pair("own_vupd", "own_kinematic", upd=True)


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
    print(f"[rung1] {len(ARMS)} arms at K={a.kmax}", flush=True)
    import torch
    torch.manual_seed(0)
    return R.stage_sweep(a)


if __name__ == "__main__":
    raise SystemExit(main())
