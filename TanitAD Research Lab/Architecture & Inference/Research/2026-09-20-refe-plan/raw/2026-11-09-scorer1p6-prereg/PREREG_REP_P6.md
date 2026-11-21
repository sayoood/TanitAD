# PREREG_REP_P6: seed replicates of the P6 scorer arms, and how they are read (FROZEN 2026-11-10)

**Status.** Frozen by the Master Mind, by sha256, on 2026-11-10 at about 04:40Z. It is written AFTER Batch 1's E3 read (C0, C0r, S1E and ENS3 are known: section 0) and BEFORE any number of C0r2, S1E1, S1Z or S1Z1 exists. S1Z's E3 logit pass started 04:31Z on the dev box and has not finished; no S1Z logit, pick or score has been read. It does not change `PREREG_SCORER1P6.md`, `PREREG_S1Z_P6.md` or the selector cards, and it re-decides none of their verdicts.

**Authorisation.** These are the committed next levers of two frozen cards. `PREREG_SCORER1P6` O5 SWAMPED (its read, 2026-11-09): *"NOT PROVEN. Next: S1E seed 1 and a second C0 replicate"*. `PREREG_S1Z_P6` Z1 and Z5: *"S1Z seed 1 (and a second C0 replicate)"*. C0r2 and S1E1 run inside Batch 1's authorised pod time with a P7 pause (PI, 2026-11-09). S1Z1 is trained ONLY if S1Z's own read is Z1 or Z5. A launch-gate PASS token is required for every arm; tokens are never hand-written.

⛔ navtest is never used for training or for choosing anything in this card.

## 0 · What is already known (inputs, not bars)

MEASURED, `raw/2026-11-09-scorer1p6-prereg/result_e3/verdict/scorer1p6_read.md`, PDMS × 100 on E3 (12,146 tokens), paired log-cluster bootstrap vs P6:
P6 91.196; C0 +0.0325 [−0.158, +0.231]; C0r +0.4120 [+0.197, +0.636]; S1E +0.0744 [−0.127, +0.272]; F = |C0r − C0| = 0.3796.
ENS3 (logit mean of C0, C0r, S1E; `verdict_ens3/`): +0.1650 [−0.011, +0.346], not cleared.
Two runs of one recipe differing only in the data-order seed differ by 0.38 points. That single pair is the programme's whole training-noise estimate, and every 2·F bar in the scorer cards rests on it. This card exists to measure that noise better and to settle whether any recipe's effect is real.

## 1 · Arms (each starts from P6's `model_final.pt`, sha256 `968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1`)

| arm | recipe | seed | trained |
|---|---|---|---|
| **C0r2** | C0 (`PREREG_SCORER1P6` section 2) | 2 (C0 0, C0r 1) | now |
| **S1E1** | S1E | 1 (S1E 0) | now |
| **S1Z1** | S1Z (`PREREG_S1Z_P6` section 1) | 1 (S1Z 0) | only if S1Z reads Z1 or Z5 |

Argv = the parent's with exactly two values changed, `--seed` and `--out` (`test_rep_gate` R1). λ_p and λ_z are the parent's measured values, read and not re-measured. Profiles `refe_rep_p6` (C0r2, S1E1) and `refe_rep_s1z_p6` (S1Z1). Code: `refe/rep_chain.sh`, `refe/rep_pod.sh`, `refe/rep_pause_p7.sh`, `refe/launch_gate.py` (landed `75ed2b20`); record `raw/2026-11-09-scorer1p6-prereg/rep/`; package builder `rep/ship_rep.py` (landed `c1ea35a6`).

## 2 · Evaluation

E3 = full navtest, 12,146 tokens, PDMS × 100. The arm's pick follows the navsim_v1 rule on its logits (torch float32, first index on ties). Changed picks are scored by lookup in the full truth table `table_p6final_c.npz` (sha256 `3bda2080...`; control 0 differences, `CHANGED_FROM_TABLE_CONTROL.md`). Gate G0-E3 (proposals bit-identical to P6) must pass for every arm, or that arm is INCOMPLETE. E2 (2,000 tokens) carries the guards. **Tier:** T1-class open loop on E3; selection tier on E2.

**Estimator.** Paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, with **every arm on the same draws**. A recipe mean is computed inside each draw. It answers "another draw of logs" only. Training variance enters only through s_p below. Inference is deterministic.

## 3 · Quantities

- Recipe sets: **C0** = {C0, C0r, C0r2}; **S1E** = {S1E, S1E1}; **S1Z** = {S1Z, S1Z1} when S1Z1 exists. For each recipe R, x̄_R is the mean of its runs' E3 PDMS points, with n_R runs.
- Per recipe: the sample SD of its runs' points, s_R (n − 1 denominator).
- **Pooled seed SD:** s_p = sqrt( Σ (n_R − 1)·s_R² / Σ (n_R − 1) ), over every recipe with n_R ≥ 2. df = Σ (n_R − 1), which is 3 for C0 and S1E and 4 with S1Z.
- **t = t(0.975, df):** df 2 → 4.303; df 3 → 3.182; df 4 → 2.776.
- **F3** = the largest pairwise |difference| among C0, C0r and C0r2 (reported; it generalises F).

## 4 · Bars (committed)

| id | question | bar |
|---|---|---|
| **R1, primary** | is recipe R better than P6? | d = x̄_R − P6. **PROVEN** iff the paired CI-lo of d > 0 **and** d > t · s_p / √n_R |
| **R2** | is S1E (or S1Z) better than plain fine-tuning? | d = x̄_R − x̄_C0. **PROVEN** iff CI-lo > 0 **and** d > t · s_p · √(1/n_R + 1/3) |
| **guards** | per run | G1 to G3 of `PREREG_SCORER1P6` on every run of a recipe; for S1Z runs also M2 and G4 of `PREREG_S1Z_P6` (M2 as reported, not as a bar here) |
| **goal** | E3 | GOAL MET if x̄_R ≥ 94.55; else BELOW GOAL |
| reported, not deciding | | F3, s_p, every run's point and CI vs P6, the four families per run (STRATEGIC PAUSED, PI 2026-09-19), the seed ensemble ENS_R (logit mean over R's runs, float32) vs P6 with its CI |

A recipe is **adoptable** when it is PROVEN on R1 and every one of its runs passes its guards. The deployable form of an adoptable recipe is decided under its own card, because a seed ensemble and a single seed are different scorers.

## 5 · Committed outcomes (each names its next lever)

- **REP-A REAL GAIN** (some recipe adoptable): the recipe's effect is real and the 2·F bars were not hiding it. Next: (i) its seed ensemble ENS_R, read under its own card with ENS_R − best single run CI-lo > 0 (0 GPU); (ii) the 1,600-update dose of that recipe, Batch 2, with a PI check before pod time.
- **REP-B NOISE** (no recipe PROVEN on R1): a 400-update scorer fine-tune is not distinguishable from training noise on E3 for the recipes read. Next: the dose arm (1,600 updates) of the recipe with the largest x̄_R, with a PI check before pod time. The cards' 2·F bars are re-stated with F3 next to F (reported; no verdict is re-decided).
- **REP-C LOSS BEATS PLAIN** (R2 PROVEN for S1E or S1Z): that auxiliary loss is a lever beyond plain fine-tuning. Next: S1E and S1Z losses stacked in one arm, under its own card.
- **Floor clause** (stated with any outcome): if F3 > 2 · 0.3796 = 0.759, the single-pair F understated training noise, and that is written next to every result that used 2·F.
- **Unnamed:** labelled as such.
- ⛔ A miss is reported as FAILED or NOT PROVEN as written. The bars do not move after a read.

## 6 · Read count

Each run's E3 read and each recipe-mean read counts in the K ledger (`p6base/selectors/ledger_K.json`). A best-of-K statement uses a Bonferroni-adjusted interval and prints K. R1 and R2 are this card's own pre-registered tests.

## 7 · Cost and order (ESTIMATED; the runs' own measurements replace it)

C0r2 about 1.3 L40-h and S1E1 about 1.5 L40-h, in one P7 pause at a checkpoint (`rep_pause_p7.sh`, arms in sequence, P7 resumed on exit). S1Z1 about 1.5 L40-h, in a second pause, only under Z1 or Z5. The dev-box E3 logit pass is about 4 h of GPU for all arms of one pause together (MEASURED 14,303 s for three arms, `pass/arm_pass_report.json`).
