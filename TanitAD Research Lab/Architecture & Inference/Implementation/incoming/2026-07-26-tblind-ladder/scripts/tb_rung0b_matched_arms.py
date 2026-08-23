#!/usr/bin/env python3
"""RUNG 0b — build the MISSING readout-matched frozen-last controls.  ~17 GPU-min.

WHY THIS EXISTS. ``T_blind`` compares imagination (a) against a frozen-last-frame
control (b) that must differ in **exactly one tensor**. The blind-imagination
sweep rolled out ``a_imagination__own__roSTR`` but **no** ``b_frozenlast__own__roSTR``
— so the deployable ``T_blind`` under the 20-step readout has no admissible
comparator, and ``bi_analyze.REGIMES["A2_readout_str_own_actions"]`` was silently
skipped (its ``"b"`` is ``None``).

Rung 0 measured what substituting the ``op``-decoded control does: in the
PRIVILEGED regime, where both controls exist, the mismatch drives ``T_blind``
from **185 steps to 1** — i.e. the substitution is not conservative, it is
**destructive**, and the unmatched contrast cannot answer the question either
way. This job builds the four missing arms so it can be answered.

⛔ WINDOW-SET IDENTITY GATE. The new arms are only poolable with the committed
dense dump if the window set is bit-identical. Two arms that ALREADY exist in
the dump are re-rolled as anchors and their dense ``de`` must reproduce; the
``eid``/``t0`` ordering must match exactly. If it does not, the run reports
BLOCKED rather than pooling incomparable arms.

Host: **pod2 only** (A40, verified idle). Reuses ``/root/bi/bi_run.py`` verbatim
— the arm list is replaced, the machinery is not.

Usage (pod2):
    PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts:/root/taniteval \
    OMP_NUM_THREADS=8 python3 tb_rung0b_matched_arms.py \
        --out /root/tbl/perwindow --episodes 600 --kmax 185
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

#: (name, state_source, action_source, update_speed_channel, readout_level)
#: The four MISSING matched controls + two ANCHORS that already exist in the
#: committed dump and therefore gate window-set identity.
MISSING_AND_ANCHORS = [
    # --- the four that make the deployable T_blind decidable --------------- #
    ("b_frozenlast__own__roSTR",  "frozen_last", "own_kinematic", False, "str"),
    ("b_frozenlast__hold__roSTR", "frozen_last", "hold_last",     False, "str"),
    ("a_imagination__own__roTAC", "imagination", "own_kinematic", False, "tac"),
    ("b_frozenlast__own__roTAC",  "frozen_last", "own_kinematic", False, "tac"),
    # --- ANCHORS: already in the committed dump; must reproduce ------------- #
    ("a_imagination__own",        "imagination", "own_kinematic", False, "op"),
    ("b_frozenlast__own",         "frozen_last", "own_kinematic", False, "op"),
]


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
    # bi_run.stage_sweep reads these off args; peek is switched OFF entirely.
    a.periods, a.oracle_bars, a.peek_bases = "", "", ""
    R.SWEEP_ARMS = []
    R.EXTRA_ARMS = MISSING_AND_ANCHORS
    # peek period/bar parsing must survive empty strings
    a.periods = "2"          # parsed but unused: peek_bases is empty
    a.oracle_bars = "0.5"
    return R.stage_sweep(a)


if __name__ == "__main__":
    raise SystemExit(main())
