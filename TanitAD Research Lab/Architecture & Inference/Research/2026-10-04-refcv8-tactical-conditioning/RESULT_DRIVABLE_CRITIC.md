# RESULT — the drivable-area critic `--r8-critic-drivable` (built; landing L2)

*WP-B, 2026-10-04. Asked by the Master Mind after D6 P1'. On 1,140 navhard DAC-zero scenes the 117-candidate fan holds a
fully clean plan 89.0 % of the time, but refcv7's imitation-only selector picks one 6.5 % of the time (random: 32 %).
This note records a BUILD, not a result.*

**Entry into the launch argv is decided by two things, not by this note:**
- D6 P4, the registered zero-training gate on refcv7-50,400;
- the ladder arm V-R8-DRV (amendment A1, R ≥ 2).

## What it is

**Footprint.** The NavSim-DAC ego box at every one of the plan's 8 poses.
- The plan point is the REAR AXLE.
- The box is nuPlan's 5.176 × 2.297 m, its centre 1.461 m ahead of the axle.
- Heading comes from the plan's own segments.
- The geometry is `tanitad.rl.pdm_proxy`'s (`box_corners`, PROXY), so the repo has one footprint definition.

**LABEL (training only).** Built from the batch's SAM3 10 cm target `map_fine`.
- A candidate is 1 iff no in-range, SEEN footprint corner lies on a non-drivable code.
- Drivable codes: drivable, lane, crosswalk, arrow, hatched. Not drivable: no-class, edge, sidewalk.
- Unseen (255) and out-of-range corners carry no evidence. A candidate with none seen has no label.

**FEATURE (training and inference).** Read from the model's OWN 10 cm map head (`out["perception"]["map_hires_logits"]`):
- the minimum and the mean P(drivable) over the in-range corners, and the in-range share;
- read DETACHED, so no gradient reaches the map head from here;
- vision only, so it is NavSim-legal.

**Critic.** An MLP on [feature (3) | path geometry (8)] gives one logit.
- The selection term is `w · logsigmoid(logit)`, added to E9's blended score, the deployed pick.
- `w` is ZERO-INIT.
- The logit trains by BCE (`--w-r8-drivable`). `w` trains through the selection losses.

**Flags.**
- `--r8-critic-drivable` (needs `--refcv8`, `--map-hires on`) and `--w-r8-drivable` (> 0 with the critic, refused without it).
- G-DVB rows: `r8_critic_drivable` (built, checked against the module), `w_r8_drivable` (loss).
- Plus a REFC_WEIGHT_GATES row and a G-LIVE rule (`r8_drv`).
- The registry pin is 270.

**Regression arm.** `--r8-roll-targets map`, the existing L3 information control: the critic's label then comes from another
window's map.

## Tests (`stack/tests/test_refcv8_drivable_critic.py`, CPU)

- **Known values:**
  - a straight path in a 6 m corridor = **1**;
  - the same path shifted 3 m left = **0** (its left corners reach y = 4.15 m);
  - each code's drivability, read one at a time;
  - unseen, out-of-range and map-less windows carry no label.
- **Footprint:** the exact pdm_proxy box at a known pose.
- **Feature:** about 1 in the corridor and about 0 shifted; it never requires grad.
- **Step-0 identity on the rig:** traj, sel_idx, anchor_traj and sel_score_v3 are bit-identical with the zero-init critic attached.
- **Deliberate regression:** w = 0.5 moves the scores.
- **BCE through `r8_losses`:** gradient reaches the critic. A built critic in a training batch with no map target is refused.
- **Map roll:** the label becomes the other window's (1, 0 → 0, 1).
- **Argv rules, the weight gate, G-LIVE and G-DVB.**

## Cost and limits

**Compute.** A gather of N × 32 corners per window on the existing logits, plus a softmax over 8 classes at those points. It adds no forward pass.

**Limits.**
- The inference feature is only as good as the map head's drivable IoU. That is why D6 P4, the zero-training gate, prices it first.
- Corners beyond the map's range (100 m × ±30 m at A7) carry no evidence. The critic is therefore an UPPER bound on compliance, as `pdm_proxy.dac_from_drivable` is.
