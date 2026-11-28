# PREREG_S1ZSM_P6: S1Z's twin loss with speed-neutral pairs (FROZEN 2026-11-10)

**Status.** Frozen by the Master Mind, by sha256, after S1Z's navtest read (`PREREG_S1Z_P6`, outcome Z5 SWAMPED) and before any S1ZSM code, training or number exists. It does not change `PREREG_S1Z_P6.md`, `PREREG_REP_P6.md` or `PREREG_SCORER1P6.md`.

**Authorisation.** PI, 2026-11-10, after the S1Z read: *"pause P7 and do the 1.3 h training"*. A launch-gate PASS token is still required; tokens are never hand-written. P7 is paused once for this arm (section 7).

⛔ navtest is never used for training or for choosing anything in this card.

## 0 · Why this arm (read before the freeze; no bar below is derived from it)

S1Z (MEASURED, E3, `raw/2026-11-09-scorer1p6-prereg/s1z/read/real/s1z_read.md`): the failure share fell, 0.0617 against P6's 0.0686 (−0.0069 [−0.0102, −0.0037]). EP fell too, −0.4488 [−0.6424, −0.2488]. The slow-pick share on E2 rose, +0.0426 [+0.0110, +0.0805]. PDMS gained +0.1571 against P6, inside 2F.

Design input, descriptive, no intervals (`s1zsm/why/twin_speed.json` sha256 `182a4279…`, script `twin_speed.py`):
- Data: navtrain FIT (8,489 sets, P6 in-sample; `fit_pack8.npz` sha256 `9a6f7f1b…`).
- S1Z's pair rule applied to P6's own logits, on 8-pose geometry. Approximations: v4 is the 3.5–4.0 s chord; v0 is the median first-chord speed.
- Reading: the clean twin is the SLOWER proposal in 53 % of NC pairs, 59 % of DAC pairs and **71 % of TTC pairs**. TTC carries 17,624 of the 25,092 pairs.
- A speed band alone (|Δv4| / max(v0, 1) < 0.15) brings NC and DAC to 51 %, but leaves **TTC at 68 %**.

⇒ S1Z's loss rewards "slower" on its largest head. F/S balancing inside the band removes that direction by construction. With balancing, TTC still has pairs in 419 of the 591 FIT sets where the band keeps any.

## 1 · Arm (starts from P6's `model_final.pt`, sha256 `968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1`)

S1Z's recipe exactly (`PREREG_S1Z_P6` section 1):
- C0's flags, 400 updates, seed 0, the same 60,000 train sets, model B;
- L_twin on heads NC, DAC, TTC; the 1.0 m radius; the top-8 mining.

**One change: the pairs are speed-neutral.** After S1Z's steps 1–3 select the pairs (i, j) of head h in a set:

- (a) **Speed-matched.** Keep (i, j) only if |v4_i − v4_j| / max(v0, 1) **< 0.15**. This is the complement of SCORER1's registered speed contrast 0.15.
  - v4: the 0.2 s chord at step 19 of the raw proposal, as `scorer1_layer.chord_speed`.
  - v0: ego slot 6.
- (b) **F/S balanced.**
  - F = the clean proposal is the faster (v4_i > v4_j); S = otherwise.
  - Keep min(|F|, |S|) pairs from each side, drawn without replacement with an own `torch.Generator` seeded 20261107. This is SCORER1's balancing rule and its seed.
  - If either side is empty, the set contributes no pair for that head.

λ_z is **re-measured** by the same gradient-norm rule (‖∇L_twin‖ = 0.5·‖∇L_BCE‖ over the first 50 micro-batches, `--pair-calibrate`, never in a launch argv), because the loss has changed. EP stays out of L_twin. The 0.15, the 8, the 1.0 m and the seed are fixed here and are not searched.

## 2 · Launch gate (profile `refe_s1zsm_p6`, a NEW dict; existing profiles and shared constants are not edited; all must PASS)

Everything `PREREG_S1Z_P6` section 2 requires, plus:
- G-DVB on the new lever, from the BUILT loss object: speed matching on, the 0.15, balancing on, the seed.
- G18 extended. On the probe batch, every kept pair is inside the band and F = S per set and head. The share of sets with a kept pair is reported per head, and 0 on any head is a blocker.
- Deliberate-regression arms that must turn RED:
  - the speed band removed;
  - the balancing removed;
  - L_twin dropped from `loss`;
  - the default-off speed-neutral flag left out of the argv;
  - S1Z's radius-removed and top-K-removed arms.
- Reachability traced flag → namespace → loss → `backward()` at `file:line`.
- Non-scorer weights bit-equal to P6's. A navtest log or token in the training data is refused, RED-tested.
- A CPU preflight on a tiny model first, then the real preflight on the L40.

## 3 · Evaluation sets (as `PREREG_S1Z_P6` section 3)

E2 (2,000 tokens) and E3 (12,146 tokens), navtest, eval only. G0-E2 and G0-E3 apply. Logits are computed on all E3 tokens. Changed picks are scored by the E3 truth-table lookup; unchanged tokens carry P6's banked value.

## 4 · Endpoints and bars (committed)

**Estimator.** Paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, over the set's own logs. It answers "another draw of logs" only.
- Training variance: F = |C0r − C0| = 0.3796, PDMS × 100 on E3, as in `PREREG_S1Z_P6`, so that S1Z and S1ZSM are judged by one rule.
- The planner is deterministic.
- **Tier:** T1-class open loop on E3; selection tier on E2.

| id | endpoint | bar |
|---|---|---|
| **P1, primary** | PDMS × 100 on E3 | **S1ZSM vs C0:** CI-lo > 0 and \|S1ZSM − C0\| > 2·F. **S1ZSM vs P6:** CI-lo > 0. |
| **M2, mechanism kept** | failure share on E3 (pick PDMS 0 or TTC 0), as `PREREG_S1Z_P6` | S1ZSM − P6: CI-hi < 0 |
| **M3, mechanism of this lever** | EP sub-score × 100 on E3 | S1ZSM − S1Z: CI-lo > 0 |
| **G1, guard** | TTC, NC, DAC sub-score × 100 on E3 | each point ≥ −0.5 against P6 |
| **G2, guard** | ranking regret on E2 | CI-hi of Δregret against P6 < +0.5 points |
| **G3, guard** | CRUISE overspeed share (plan v4/v0 > 1.15) on E2 | CI-hi of the difference against P6 < +0.10 |
| **G4, guard** | CRUISE slow-pick share on E2, the definition of `PREREG_S1Z_P6` | CI-hi of the difference against P6 < +0.10 |
| **Goal status** | PDMS on E3 | GOAL MET if the point ≥ 94.55; GOAL NOT EXCLUDED if CI-hi ≥ 94.55; else BELOW GOAL |
| reported, not deciding | S1ZSM − S1Z on PDMS, failure share and slow-pick share; EP against P6; the Shapley split and the top-K curve on E2; the per-head pair shares of the run; once `PREREG_REP_P6`'s read exists, its pooled seed SD s_p and t·s_p next to 2·F | |

**Four families, each with its CI on the same tokens (binding):** as `PREREG_SCORER1P6` section 4. STRATEGIC is **PAUSED** (PI 2026-09-19), not a pass.

**Adopted.** S1ZSM is adopted if it clears P1 and G1 to G4. **Family clause** as `PREREG_SCORER1P6`.

**Read count.** The E3 read counts in the K ledger (`selectors/ledger_K.json`). A best-of-K statement uses a Bonferroni-adjusted interval and prints K. P1 is the arm's own test.

⚠️ **One seed.** By `H-ESTIM-SEED-1`, a separated interval from one run is necessary, not sufficient. A lever effect is PROVEN only by the replicate rule of `PREREG_REP_P6` (R1/R2), with S1ZSM seed 1 as its second run.

## 5 · Committed outcomes (each names its next lever)

Every match is listed. The primary label follows SM1, SM2, SM3, SM4, SM5, SM6, then Unnamed.

- **SM1 GOAL MET** (adopted, point ≥ 94.55): S1ZSM seed 1 as the replicate, then the closed-loop read of `PREREG_SCORER1P6` section 6.
- **SM2 ADOPTED, BELOW GOAL** (P1 and G1–G4): S1ZSM seed 1 under the R1/R2 rule. Then stack it with S1E's loss (REP-C), or the 1,600-update dose (PI check).
- **SM3 SWAMPED** (vs C0, CI-lo > 0 inside 2·F): S1ZSM seed 1, read under the R1/R2 rule.
- **SM4 MECHANISM, NO PDMS** (M2 and M3 hold, P1 fails, not SM3): the speed direction was S1Z's EP leak and it is closed, but the gain stays inside noise at this dose. Next: the 1,600-update dose (PI check).
- **SM5 EP NOT RECOVERED** (M2 holds, M3 fails): speed direction is not the EP leak. Next, zero GPU: the Shapley split on E2, and the per-head slow-pick share. Then TTC removed from L_twin under its own card.
- **SM6 FAILURES BACK** (M2 fails): speed-neutral pairs cannot teach the heads the twin distinction, so S1Z's failure gain was carried by speed. Next: explicit relative features between proposals (lead gap, relative speed), under their own card (`PREREG_S1Z_P6` Z4).
- **Unnamed:** labelled as such; the Shapley split on E2 names the next lever.
- ⛔ A miss is reported as FAILED or NOT PROVEN, as written. The bars do not move after a read.

## 6 · What is not claimed

No closed-loop claim. Section 0's numbers are approximations on navtrain and decide nothing. The tracker-made share of failures (`D-REFE-ZEROTAIL-3`) limits this arm's reach exactly as it limits S1Z's.

## 7 · Cost and order (ESTIMATED; the run's own measurement replaces it)

P7 is paused once, at a checkpoint:

| step | estimate |
|---|---|
| calibration | about 5 min |
| preflight and five mutants | about 20 min |
| gate | about 3 min |
| training (S1Z MEASURED about 1.1 h) | about 1.15 h |
| **total** | **about 1.6 L40-h** |

Then P7 resumes. Navtest logits run on the dev box, about 4.3 h of GPU unless shared with another pass.
