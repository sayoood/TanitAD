# PREREG_DIV_P6 — data-diverse ensemble members for P6's scorer (Decision B, the comparison arm)

**Status.** Frozen by the Master Mind, by sha256, on 2026-11-10, before any DV0 or DV1 code, run or number exists.
PI 2026-11-10, verbatim: *"Go ahead with Decision A, write the design card, do als B as co,parison"* — the go for B's pod
time (about 3 L40-h in one P7 pause). Design input: `DECISION_P6_NEXT_LEVERS.md` section 4 and `PREREG_PESS_P6.md`'s
PESS read (PE3: the eight fine-tunes share P6's blind spot; pessimism moved PDMS-0 picks only 2.96 % → 2.74 %).

## 1. Question

Do two scorer fine-tunes trained on DISJOINT halves of the training logs make a better pessimistic ensemble than two
fine-tunes trained on the same data with different seeds? If the members' shared error comes from the data they saw, data
diversity should help; if it comes from P6's frozen trunk, it should not.

## 2. Arms (two trainings; everything else is a combination of banked logits)

- **Split.** The training logs of the 60,000-set bank used by C0, sorted by name, go to half `h = int(sha256(log name), 16) % 2`.
  Every training set of a log follows its log. The held-out monitor (1,705 sets, 24 logs) is unchanged. The two halves'
  set counts and log counts are reported before training; a half below 25,000 sets is a launch blocker.
- **DV0** = C0's argv exactly, plus the split flag restricted to half 0, seed as C0. **DV1** = the same on half 1, seed as
  C0r. 400 updates each (if one epoch of a half is shorter, the run stops at the epoch and that is stated).
- Combinations, all with `read_pess.py`'s rules and the full truth table: **PESS-DV** = PESS over {DV0, DV1};
  **PESS-SEED** = PESS over {C0, C0r} (same recipe, full data, two seeds: the matched control);
  **PESS-10** = PESS over the eight PESS-read members plus DV0 and DV1.
- Launch: the launch gate with a G-DVB entry for the split flag; preflight must show every training row's log in its half,
  and a deliberate-regression mutant with the split filter removed must FAIL preflight.

## 3. Bars (E3 = full navtest, PDMS x100, paired log-cluster bootstrap 10,000 draws, seed 20260927)

| id | metric | bar |
|---|---|---|
| **Q1** | PESS-DV − PESS-SEED | CI-lo > 0 |
| **Q2** | PESS-10 − P6 | CI-lo > 0 and d > 0.4971; and PESS-10 − PESS-8 (the banked PESS arm, sha `8912f1d0...`) CI-lo > 0 |
| M | zero-pick share, PESS-DV − PESS-SEED | CI-hi < 0 |
| G1-G4 | as `PREREG_PESS_P6.md`, for PESS-10 vs P6 | as written there |

Reported, not deciding: DV0 and DV1 each vs P6 and vs C0 / C0r (half the data, so they may be weaker alone); the trainer
held-out monitor; the four metric families per arm (STRATEGIC PAUSED, PI 2026-09-19). Training variance is not resampled;
the read answers "another draw of logs" for fixed members.

## 4. Outcomes (label order DV1, DV2, DV3)

- **DV1 ADOPT** (Q2 and G1-G4): PESS-10 becomes P6's selector. Next: four quarter-split members under their own card.
- **DV2 DIVERSITY HELPS, TOO SMALL** (Q1 holds, Q2 fails): data diversity is real but the ensemble stays below the bar.
  Next: Decision A (the vision drivable-area check) carries the line; more members only if A stalls.
- **DV3 NO DIVERSITY GAIN** (Q1 fails): the shared error sits in P6's frozen trunk, not in the fine-tune data. The ensemble
  line is closed; Decision A.
- **Unnamed** if none matches.

K: + 5 (DV0, DV1, PESS-DV, PESS-SEED, PESS-10; E3 keys by combined-logit sha256).
