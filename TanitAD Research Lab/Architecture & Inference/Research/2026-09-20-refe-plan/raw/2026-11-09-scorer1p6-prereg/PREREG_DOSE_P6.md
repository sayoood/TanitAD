# PREREG_DOSE_P6: the 1,600-update dose of S1Z (FROZEN 2026-11-10)

Author: Master Mind, 2026-11-10, after the PREREG_REP_P6 read (REP-B) and before any dose arm exists. This card is the named next lever of REP-B: "the dose arm (1,600 updates) of the recipe with the largest mean, with a PI check before pod time".

## 0 · What is already known (inputs, not bars)

MEASURED, `raw/2026-11-09-scorer1p6-prereg/rep/read/real/rep_read.md` (landed 31ef717c), PDMS x100 on E3 (full navtest, 12,146 tokens), paired log-cluster bootstrap vs P6 91.196:
- Recipe means: S1Z {S1Z 91.354, S1Z1 91.583} = 91.468, d +0.2716 [+0.0944, +0.4617]; C0 {C0, C0r, C0r2} = 91.440; S1E {S1E, S1E1} = 91.388. None PROVEN on R1.
- Pooled seed SD s_p = 0.1791 (df 4, t(0.975, 4) = 2.776). F3 = 0.3796.
- Regret on full navtest (`raw/2026-11-09-p6-navtest-read/selection/`): P6 pick 91.196, best of 64 proposals 97.90; regret 6.70 = zero-multiplier picks (NC or DAC 0 when a non-zero proposal existed) 2.81 + TTC-0 picks 1.58 + rest 2.32.
S1Z is the recipe with the largest mean, so it is the dose recipe. Every 400-update fine-tune moved P6 by about +0.2 to +0.3; this card asks whether four times the updates moves it further, and stops the fine-tune line if it does not.

## 1 · Arm

| arm | recipe | change from the parent's argv | seed |
|---|---|---|---|
| **S1ZD** | S1Z (`PREREG_S1Z_P6` section 1) | exactly two values: `--max-updates 400` becomes `1600`, and `--out` | 0 (S1Z's) |

- Starts from P6's `model_final.pt`, sha256 `968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1`.
- λ_z is S1Z's measured value, read from S1Z's calibrate dir, never re-measured (a re-measured weight is a second moved lever).
- Everything the trainer derives from `--max-updates` (the LR schedule's length) moves with it: that IS the dose. Every other argv token is the parent's, proven by an argv diff test (two tokens differ, `--max-updates` and `--out`).
- Mutants: S1Z's own G18 regressions (radius filter removed, top-K mining removed, L_twin dropped), each must read RED.
- Pod time needs the PI's check (REP-B). Nothing in this card authorises it.

## 2 · Evaluation

As `PREREG_REP_P6` section 2: E3 = full navtest, 12,146 tokens, PDMS x100; the arm's pick by the navsim_v1 rule on its logits (torch float32, first index on ties); changed picks scored by lookup in `table_p6final_c.npz` (sha256 `3bda2080...`); gate G0-E3 (proposals bit-identical to P6) or INCOMPLETE. **Tier:** T1-class open loop on E3.
**Estimator:** paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, every arm on the same draws; the S1Z recipe mean is the mean of the S1Z and S1Z1 per-token vectors inside each draw. It answers "another draw of logs" only; training variance enters only through s_p, fixed at 0.1791 from the REP read (an input, not re-estimated with S1ZD, which is a different recipe).

## 3 · Bars (committed)

| id | question | bar |
|---|---|---|
| **D1, primary** | is S1ZD better than P6? | d = S1ZD − P6. **PROVEN** iff paired CI-lo > 0 **and** d > t · s_p = 2.776 · 0.1791 = **0.4972** |
| **D2** | is the dose better than the 400-update recipe? | d = S1ZD − x̄_S1Z. **PROVEN** iff paired CI-lo > 0 **and** d > t · s_p · √(1 + 1/2) = **0.6089** |
| **guards** | | G1 to G3 of `PREREG_SCORER1P6`; M2 and G4 of `PREREG_S1Z_P6`, as the REP read applies them to S1Z runs |
| **goal** | | GOAL MET iff S1ZD ≥ 94.55; else BELOW GOAL |
| reported, not deciding | | the four families vs P6 (STRATEGIC PAUSED, PI 2026-09-19); S1ZD's regret split (zero-multiplier, TTC-0, rest) next to P6's; the trainer's held-out pick at every 200 updates; K |

## 4 · Committed outcomes (each names its next lever)

- **DOSE-A WORKS** (D1 and D2 PROVEN, guards pass): more updates is a real lever. Next: S1ZD's seed-1 replicate (adoptability, as the REP card), then a 6,400-update dose, each with a PI check before pod time.
- **DOSE-B BEATS P6 ONLY** (D1 PROVEN, D2 not): the dose clears noise against P6 but not against the 400-update recipe. Next: S1ZD's seed-1 replicate, so D2 is read on two runs.
- **DOSE-D GUARD FAIL** (D1 PROVEN, a guard fails): not adoptable. Next: the failing guard's family split on E2 (Shapley over heads), under its own card.
- **DOSE-C SATURATED** (D1 NOT PROVEN): a longer fine-tune of the scorer's loss does not buy PDMS. Next: leave the fine-tune line and attack the largest regret term, zero-multiplier picks (2.81 of 6.70 points), with a lever outside the scorer-loss family, under its own card.
- Ordering of labels when several match: A, B, D, C. **Unnamed:** labelled as such.
- ⛔ A miss is reported as FAILED or NOT PROVEN as written. The bars do not move after a read.

## 5 · Read count

S1ZD's E3 read adds one key to the K ledger (`p6base/selectors/ledger_K.json`). D1 and D2 are this card's own pre-registered tests.

## 6 · Cost (ESTIMATED; the run's own measurements replace it)

S1ZSM, on the same trainer and L40, MEASURED 8.1 s per update and 54 min of preflight, five mutants and gate; S1Z run 2's whole pause (three mutants) was 1 h 35 min for 400 updates. S1ZD: about 3.6 h of training plus about 0.5 h of preflight, three mutants and gate, so **about 4.1 L40-h in one P7 pause**. Its E3 logit pass on the dev box: about 4.3 h, queued on the shared GPU lock.

## 7 · Relation to S1ZSM

S1ZSM's own card names a 1,600-update dose under its outcome SM4. That would be a separate arm under its own card. This card's verdict is not changed by S1ZSM's read, and S1ZSM's is not changed by this one.
