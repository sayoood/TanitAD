# EP-R1: Master Mind decisions (2026-10-18, before any EP-R1 outcome exists)

These are the two cross-arm decisions `PREREG_EPR1.md` (sha256 `29e79bd61cbbb209bc08aa8b42315736ee47fdaf89abd8ad0731a4d4d4f8f80a`) left to the Master Mind (its RESULT, "Integration asks"), plus one disclosure. They are written before S0 has finished, so before any fit, dev or navtest number exists for EP-R1. The PREREG itself is not edited.

## D1. The navtest look: ONE open for the REFE head programme, shared — ACCEPTED

- **One navtest open** reads up to three pre-registered rows: SR-TTC alone (`PREREG_SELRANK_ARM.md` and its amendments), EP-R1 alone (`PREREG_EPR1.md` section 7), and the joint (EP-R1 section 8, row J1/J2, SECONDARY). Each is read against **its own** bar, registered before the open.
- **No selection among them.** No reading may be promoted because another one missed, and the joint never decides the EP-R1 primary. That is PREREG_EPR1's own rule, restated here as binding on the Master Mind as well.
- **Multiplicity is disclosed, not corrected.** Two primaries from two separate registrations ride on one look, with no family-wise correction. Any report quoting either primary states the other reading and its outcome beside it.
- **Timing (new, binding).** The look opens only after **both** dev gates have been read: SR-TTC R1's and EP-R1's. The other way out is that one of the two arms is formally abandoned, with a register row saying why. Otherwise an SR-TTC confirm could spend the look before EP-R1 has a dev reading, and EP-R1 would lose the open that `D-REFE-EP-2` held for it.
- **Membership.** Per PREREG_EPR1 section 8, EP-R1 is read on navtest **iff** EP-R1 alone passes its dev gate. SR-TTC follows its own registration. If neither passes, the look stays unspent.
- **Why one open, not two.** `D-REFE-EP-2` left ONE look unspent and held it for the SR-TTC-retrained scorer. Both arms fit head-only rows of the same P2 `score_head`, on the same S0 bank, with the same folds. Two opens would spend a second look on a near-identical rig for no gain in what can be claimed.

## D2. The fit fraction: EP-R1 runs at SR-TTC R1's c\* fraction, f = 10 % — ACCEPTED

- Both rows must be fitted on **identical fit logs**. Otherwise the joint secondary combines two rows fitted on different data, and its reading cannot be attributed.
- f = 10 % is R1's fraction as set in `MM_DECISIONS_R1.md`. If R1's fraction changes by a registered amendment before EP-R1 is fitted, EP-R1 follows it. The change is recorded here, before the fit.

## D3. Disclosure: the dev set is not virgin for EP-R1

- The dev gate reads SR-TTC's held-out table. EP-1's `DIAGNOSIS.md` (c) already read the rule's margin terms on that table, and EP-1's four zero-training rules were fitted and read on it.
- So the dev gate is a **gate, not a claim**. No EP-R1 number from it is quotable as a capability result.
- Only the navtest primary (T1, NAVSIM v1.1 navtest, paired log-cluster bootstrap, 10,000 resamples, seed 20260927) is admissible for the register. It is quoted with its estimator and its one-training-seed-per-arm limit (`H-ESTIM-SEED-1`; the fit uses 2 seeds).

## Not decided here (named so they are not forgotten)

- `epr1_dev.py` and `epr1_confirm.py`, the real-bank C2/C4 reads and `families6` on the new picks are **not built** (PREREG section 11). They are BACKLOG items, gated on `S0_DONE`.
- G-LIVE and G-BANK have run on synthetic banks only. The first `mint` after `S0_DONE` is the real-manifest check, and the gate must PASS before any fit.
