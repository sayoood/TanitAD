# Controls for the PhysicalAI read of a WiSE blend (R8-WISEFT-PAI, 44eec8f) -- run BEFORE any blend battery output exists

1. **SHIP lane (`../code/wise_ship_lane.sh`) on the banked 48,852 tree** (`ship_lane_control_48852/`, 2026-10-21 14:01-14:16): a fresh
   private pf package (`pf_lane.py install` with the REGISTERED table 2aacc9ff...) re-derives SHIP-R8 = **PASS** with every clause
   status AND evidence identical to the banked `SHIP_R8_step48852ema_pf.json`; FORWARD_DEPLOY8 / FORWARD_TRAINFIT / TAC / PRIOR_PARITY:
   6,986 / 13,290 / 1,689 / 239 numeric leaves, all equal except one timestamp per file.
2. **Pair tool relabel (`../code/paired_48852_wise050.py`, a mechanical relabel of 5ee51c5's `paired_40k_48852.py`)**, run with the
   blend slot pointed at the 40k tree (`relabel_control_40k.py`, `pairtool_relabel_control_40k/`): of the banked 48,852 - 40k result's
   362 `delta` leaves, **290 reproduce exactly with the sign flipped** and the other **72 are per-tree quantities equal under the slot
   swap** (0 unexplained); controls ALL_PASS. The first run read C1 FAILED only because the brief's named literals were keyed by SLOT;
   they are now keyed by TREE NAME (a tree without a published literal -- the blend -- is stated, never counted as a pass).
