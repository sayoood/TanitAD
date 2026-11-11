# PREREG_VDAC_P6_A2: the tracked footprint is admitted and becomes primary; S1 runs in two stages (PI rulings, 2026-11-11)

**Status.** Frozen by the Master Mind, by sha256, on 2026-11-11. That is after A1, and before:
- any S1 training;
- any H48 read;
- any tracked footprint computed on a P6 proposal;
- any navtest read of a VDAC arm.

It amends `PREREG_VDAC_P6.md` (sha256 `5594c2ae...`) and `PREREG_VDAC_P6_A1.md` (sha256 `8f121bdf...`). Nothing else changes.

**PI rulings (2026-11-11).** In reply to the Master Mind's three recommendations, the PI said, verbatim: *"follow your recommendation"*. For this card that means:
- **D-VDAC-3, option A.** The tracked footprint is admissible at inference. The tracked veto becomes S2's primary arm, and the pose-only veto is reported.
- **D-VDAC-4, option A.** S1 runs in stages. The first P7 pause trains HA-s0 and HA-SHUF. The second pause, for HA-s1, runs only if HA-s0 reaches S1-GO.

**Why tracked is admissible (the ruling's grounds; no P6 data).**
- The tracker starts from the token's t0 ego state, which is the 7-D ego status P6 already reads (`refe/model.py:185`). That includes the tire steering angle, covered by the REFe exemption `D-PI-YAWRATE-3` (2026-10-18).
- It reads no future state, no map and no agent. Its inputs are that state and proposal k.
- It runs NAVSIM's own controller model, so it is benchmark-specific. A deployed car would use its own controller instead.
- The October read on u400 (not P6; banked `raw/2026-10-16-auxveto-read/RESULT.md`) is the prior: with the true raster at margin 0.25 m, the tracked veto gained +0.794 [+0.628, +0.970] and the pose-only veto +0.367 [+0.015, +0.638]. No tracked number exists on P6.

## 1. The tracked footprint (definition, fixed now)

For token t and proposal k (P6's 8 poses at 0.5 s, from `proposals_p6final.npz`):

1. **Initial state:** the t0 `ego_state` that NAVSIM's scorer uses for token t, from the navtest metric cache. It holds the rear-axle pose (origin), velocity, acceleration and tire steering angle.
2. **Raw states:** NAVSIM v1.1's `transform_trajectory` + `get_trajectory_as_array` on k's poses: 41 states at 0.1 s.
3. **Tracked states:** `PDMSimulator(TrajectorySampling(num_poses=40, interval_length=0.1)).simulate_proposals(raw, ego_state)`, the LQR tracker plus kinematic bicycle.
   - The code is NAVSIM v1.1's source at the commit the scoring table used (`navsim-3e8291bf...`).
   - It follows the path `navsim/evaluate/pdm_score.py:83-140`, as the banked `raw/2026-10-16-auxveto-feasibility/tracked_dump.py` does.
4. **Points:** `state_array_to_coords_array` corners (4 per state, centre excluded), in the t0 rear-axle frame: 41 x 4 = 164 points. NAVSIM's own vehicle polygon is used, not the AUXVETO constants of the pose-only footprint.
5. **Field and d_k:** exactly the card's section 5.1 steps 3-4.
   - The field is the head's mask (`logit > 0`), turned into a signed distance by a Euclidean distance transform, positive inside.
   - It is sampled bilinearly, and d_k is the minimum over the points inside the raster extent.
   - The share of points outside the extent is reported.
   - A proposal with no point inside is not vetoed.

## 2. Arms after A2

| arm | rule | role |
|---|---|---|
| **VDAC-V0T (HA-s0)** | proposal k is dropped if its TRACKED d_k < 0; all dropped => P6's pick; else the navsim_v1 rule on P6's own logits | **primary** |
| VDAC-V0T (HA-s1) | the same, with the seed-1 head | replicate (P1r) |
| VDAC-V0 (HA-s0) | the same veto on the pose-only footprint (A1's primary) | reported comparison; cannot be adopted |
| VDAC-Phi (HA-s0), VDAC-PhiL, VDAC-V (0.25 m) | as the card | reported; cannot be adopted |
| VDAC-SHUF | **VDAC-V0T with HA-SHUF** | deliberate regression (C-SHUF) |
| O-DAC | as the card | reported ceiling |
| VDAC-PhiT | not read | S0 measured the Phi rule harmful at every width; no navtest key is spent on it |
| VDAC-Phi (HA-s1) | not read | its replicate role moved to VDAC-V0T (HA-s1) |

**Bars.** Every S2 bar, outcome and guard that the card (section 2-3) and A1 state for the primary now applies to VDAC-V0T (HA-s0), with the same values:
- P1: CI-lo > 0 and d > 0.4971;
- P1r: CI-lo > 0;
- M: CI-hi < 0 on the DAC-zero pick share;
- G1-G4;
- outcomes VD1-VD5.

In VD3, "VDAC-PhiT if admissible" is replaced by "the agent check (D-VDAC-6)".

**Inputs per arm (information disjointness, card section 6).**
- VDAC-V0T reads the head's field, P6's proposals, P6's logits and the t0 ego state.
- The HEAD still reads no ego and no command. The card's section 7 disjointness gate on the head is unchanged.
- No arm reads the situation classifier, the map at inference, or any future state.

## 3. New reader gates (fail closed)

- **G0-TRK (tracker identity).** NAVSIM's own drivable-area rule (`pdm_scorer.py:284-286`), applied with NAVSIM's own polygons to the reader's tracked states, must reproduce the scoring table's DAC sub-score on every one of the 12,146 x 64 = 777,344 proposals. Any mismatch => **INVALID** until explained.
  - This reads navtest map polygons only to validate the instrument. It selects nothing and is not keyed.
  - October precedent: `d0==table 64/64` on every token, `logs/tracked_*.log`.
- **G0-TRK-DET.** A second, independent dump of 100 fixed tokens (the first 100 by sorted token) reproduces the tracked corners bit-identically. The read records the sha256 of the full tracked-corner array.
- **C-TRK-LIVE.** The tracked points differ from the pose-only points on more than 0 proposals; otherwise the tracker is not reached and the read is INVALID. The share of proposals whose veto verdict differs between tracked and pose-only (true field, from G0-TRK's run) is reported.
- C-ZERO, C-LIVE, C-FLIP, C-SHUF and C-ODAC apply to VDAC-V0T as the card writes them for the primary.

## 4. S1 in two stages

| stage | trains | pod time (ESTIMATED, `DESIGN_VDAC_P6.md` section 7) | condition |
|---|---|---|---|
| **S1a** | HA-s0 (10,000 updates) + HA-SHUF (2,000 updates) + held-out forwards | ~3.9 L40-h, one P7 pause | PI go 2026-11-11; launch-gate PASS per arm; A3 (the H48 list) frozen first |
| read S1a | S1-P1, S1-Q, S1-F for HA-s0; C-SHUF, C-FLIP | CPU | bars unchanged |
| **S1b** | HA-s1 (10,000 updates) | ~3.2 L40-h, a second P7 pause | **only if HA-s0 reaches S1-GO with valid controls** (PI go 2026-11-11 is conditional on this) |
| read S1b | S1-R | CPU | then S2 |

- If HA-s0 does not reach S1-GO, HA-s1 is not trained, and the card's S1 branch is followed as written (S1-COMP / S1-BOUND / S1-FIT / S1-WEAK / INVALID). Each branch's own seeds and pod time go back to the PI.
- **S1-QT (reported, not deciding):** S1-Q's AUC with the tracked d_k on H48. It is read only if a navtrain metric cache with t0 ego states covers H48's sets; otherwise it is reported as unavailable, with the reason and n.

## 5. K ledger (replaces the card's section 6 count)

There is one key per arm read on E3: O-DAC, VDAC-V0T (s0), VDAC-V0T (s1), VDAC-V0 (s0), VDAC-Phi (s0), VDAC-PhiL, VDAC-V (s0) and VDAC-SHUF.
- That is **K + 8** for the card, plus 3 if the S0-NEAR oracle rows are ever read (they are not: S0 read GO).
- G0-TRK is an instrument check and is not keyed.
- P1, P1r and M are VDAC-V0T's own tests; best-of-K statements use Bonferroni intervals.

## 6. Unchanged

- H48: its 48-log list and its sha256 go into **A3**, frozen before S1a trains.
- S1's perception bars and the head recipe are unchanged.
- The section 7 launch-gate checks apply to each stage.
- The estimator (paired log-cluster bootstrap, 10,000 draws, seed 20260927) is unchanged, as are the tier (T1-class open loop, NAVSIM v1 PDMS by table lookup) and the four families (STRATEGIC PAUSED, PI 2026-09-19).
- No bar moves after a number is seen.
- **Expectation (ESTIMATED, carried from the PI brief, not a bar).** A head of S1c's October quality projects to about 0.16 points pose-only. With the tracked footprint, the head must keep about 43 % of the true-map gain to clear 0.4971, against 31 % measured in October. S1a tests that cheaply before S1b is spent.
