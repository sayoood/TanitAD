# S1 read: the aux drivable-area head (nine arms). H2 NOT PROMOTED; H1 not promotable (K2) (Master Mind, 2026-10-15)

**Provenance.**
- **Read:** `eval/s1_read.py read`, from a fresh archive of `421ffc3` (A7 + A8 in force).
  - Status `ZZS1READ_READ`.
  - All nine registered arms present and clean; 0 refused; 0 instrument failures; 634 s.
- **Tier:** T0-like, i.e. held-out selection and perception on teacher-forced banked sets. **Never driving performance.**
- **Estimator:** paired log-cluster bootstrap, 10,000 resamples, seed 20260927. It answers *"would another draw of LOGS say this?"* only. Training-run
  variance is the N-vs-N seed floor (`H-ESTIM-SEED-1`).
- **Evidence class:** every number below is MEASURED, from `S1_READ.json` in this folder.
- **Excluded from the repo:** the six `instrument/heads/head_pred_*.npz` (11 MB each). Their sha256s are in `heads_sha256.txt`, and they regenerate from
  each arm's `aux_drv.pt` with the pinned forward.

## Outcome, as pre-registered (S1 prereg §4.3; A7, A8)

**H2 (feed on): `H2_NOT_PROMOTED`.** Every one of the four conditions fails:

| condition | result |
|---|---|
| 1. P3 on C3 is EVALUATED / PROVISIONAL on all three seeds (A7.1) | **`H2_KILLED` on all three**: head DAC AUC 0.772 / 0.765 / 0.781 < scorer 0.787 ("head AUC < scorer AUC ⇒ H2 killed") |
| 2. PDMS `separated and above the seed floor` on ≥ 2 of 3 pairs | **0 of 3**: seed 0 `adverse`, seeds 1–2 `not_separated` (floor PDMS 0.352, measured on three N) |
| 3. no pair `adverse` on PDMS or DAC | **fails**: H2-s0 PDMS adverse |
| 4. no K2 / K3 / K6 on any H2 run | **fails**: K2 on all three seeds, K6 on seed 0. K3 cleared (step time +2.1–2.4 % across both limbs, memory +0.51 GB) |

The outcome is the same under the supplementary readings: AMB-1's 2-of-3 reading gives `H2_NOT_PROMOTED` too.
Stamps: *capacity not excluded* (no C-shuffle arm at S1); a held-out selection verdict, never an ADOPT.

**H1 (feed off).**
- **The read's label is `H1_IDENTITY_DEFECT`** (no promotion label is emitted).
- **H1 is not promotable regardless.** Its rule needs K2 not to fire, and K2 fires on all three H1 seeds: navtest P1 IoU [0, 40) is 0.496 / 0.521 / 0.486,
  against a bar of 0.80. P1-HO (0.631 / 0.643 / 0.621) agrees.

## What the numbers say

1. **The head learns, but the frozen visual features do not carry the road at the precision K2 asks for.**
   - Against the A2.4 initial-head floor, measured before any arm trained, P3 AUC rose 0.659 / 0.563 / 0.698 → 0.772 / 0.765 / 0.781.
   - P1 IoU rose from 0.13–0.27 (A2.2) to 0.49–0.52.
   - Both still sit below the scorer's own DAC ranking (0.787) and far below 0.80. K2 fails by about 0.3 IoU, not by a margin a 1 m grid could
     close (A2: that lever "cannot rescue a head that misses 0.90 by more than ~0.012").
2. **The feed (H2) moves selection by no more than training noise.**
   - H1 vs same-seed N, which is identical by construction, differs on about 160 of 3,137 EP picks per seed (b/w 81/82, 75/94, 80/80).
   - H2 vs N differs on 179–208 (96/109, 114/94, 87/92). Every DAC / TTC / EP paired delta has a CI covering 0, except the seed-0 PDMS adverse.
   - At this head quality, the scorer gains nothing from reading it.
3. **H1 and H2 heads are the same head.**
   - IoU agrees to the 4th decimal at each seed (0.49570 vs 0.49579, 0.52110 vs 0.52112, 0.48640 vs 0.48626), and so does AUC.
   - This is expected: the feed is detached (`score_to_head_grads_all_none`), so the head trains identically. The only H1/H2 difference is the
     scorer reading it, and point 2 shows that reading is worth nothing yet.

## The H1 identity "defect": HYPOTHESIS, training non-determinism (not a wiring path)

**Why it is probably not a path.**
- On the A40, H1-s0's preflight ran in strict deterministic mode with the cuBLAS env (A6). It proved `scorer_params_bit_identical_2steps: true`,
  `reference_replay_identical: true` and `score_to_head_grads_all_none: true` (`raw/2026-10-14-s1-g12-bf16/a40_H1-s0_PREFLIGHT_PASS.json`).
- Only the **preflight** runs strict. The **training** runs do not (A6 put the env "on the preflight line only; the trainers are unchanged"). So two
  same-seed runs need not be bit-identical, and the pick differences (about 5 % of EP picks; 0–2 of 3,137 on DAC / TTC) are the size that GPU
  atomics compounding over 400 updates would give.

**What would CONFIRM it (not run).** A same-flags, same-seed replicate of N-s0, compared with N-s0: the same pick-difference rate ⇒ non-determinism.
That costs about 1.5 A40-h, after P2-s1 and P6.

**Why it is not urgent.** It changes no verdict here. H1 fails on K2 independently, and condition 2 already reads H2's selection against a
seed floor (0.352 PDMS) above the H1-vs-N differences.

**What it exposes.** The prereg's *"bit-identical by construction"* holds only under deterministic training. That is the same lesson as
`H-ESTIM-SEED-1` and `D-REFAV1-SEED-GOAL-MISMATCH`: on this rig the training-run floor is non-zero even at a fixed seed.

## Next lever: Rule Zero, from the pre-registered next-arm table

The draft's row **"P1/P3 fail (K2)"** reads: *the frozen features do not carry the road at this precision: the `scene` tap, a graft with an unfrozen trunk, or
a 1 m grid.*
1. **The `scene` tap.** The head already supports `tap="scene"` (G12 records `tap_is_scene_ctx`). It is the cheapest discriminating lever.
   - **The question:** does the scene context carry the road at P1 ≥ 0.80, when the visual context reaches about 0.50?
   - **The cheapest form** needs no A40: a head-only probe on cached frozen features, visual vs scene, log-disjoint folds, with both outcomes committed.
   - **Status:** pre-registration and build delegated, 2026-10-15.
2. **An unfrozen-trunk graft.** An architecture change: **a PI decision**.
3. **A 1 m grid.** Ranked last by A2.
