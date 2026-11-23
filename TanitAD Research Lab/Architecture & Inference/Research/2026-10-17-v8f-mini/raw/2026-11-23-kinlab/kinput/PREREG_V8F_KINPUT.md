# PREREG_V8F_KINPUT (FROZEN 2026-11-23, before any KI step, census or number): the input side of rolled history: a swapped own sample keeps the LOGGED prior path `base`

**Parent cards:**
* `PREREG_V8F_ROLLHIST.md` (781eea87, with Amendments 1 and 2) binds unless changed here.
* `PREREG_V8F_KINLAB.md` (7db5d7aa), whose PARTIAL row, MECH3-NO branch, names this lever: K-INPUT.

**PI go:** none yet. The run waits on a go with its cost (section 6).

## 1. Why
* Three label-or-mode levers have been read on the same RH banks, each against its parent at T0:
  * the speed label (RL);
  * add mode instead of replacement (RA, MECH2-NO);
  * the kinematic label (RK, MECH3-NO).
* None of them moves the T0 cost of rolled history. All three keep the T1 gains against B0 on LATERAL, TACTICAL and STRATEGIC.
* G-T0 against B0 fails at s0 and holds at s1 for RH, RL, RA and RK alike. The largest s0 point is RH's +0.0503, the smallest RK's +0.0364.
* The remaining named candidate on the input side is O's prior path `base`. In a swapped sample it is `kdx_base` of the OWN kinematics. The only model reader of `base` is the prior, `p30 = base + MLP(p10f)`, to which O's residual is added (`stack/tanitad/models/v8f_mini.py:994`, `P10Prior` 327-360, at fde8b716; FACTS_KINPUT section 2).

## 2. Hypothesis and arms
**H-V8F-KINPUT-1.** Part of the T0 cost of rolled history comes from training O on prior paths built from off-distribution own kinematics. Keeping the logged row's `base` in the swapped samples lowers the T0 cost against RH and keeps G-T0, while the T1 gains against B0 stay.

**Arms KI_s0 / KI_s1** = RH_sK's argv (62 tokens) with exactly two changes, and nothing else:
* `--out /root/rollhist/runs/RH_sK/train` → `/root/rollhist/runs/KI_sK/train`;
* **added:** `--dagger-base logged`.

The result is 64 tokens. The argv check admits these two changes and refuses any third.

Everything else is RH's: code **4097a89e** on `agent/v8f-rollhist-cdb-20261120`, RH banks (s0 `e5e44c01…`, s1 `bde6566e…`), readout causal_g1x, `--dagger-p 0.1`, `--dagger-label` human (RH's default), init UE_sK.A, 3,000 steps of stage B.

**What `logged` does** (FACTS_KINPUT, MEASURED from source at fde8b716):
* The batch row of a swapped sample is the logged window at r'. Under `logged` that row keeps its own `base`, bit-identical to `kdx_base` of the store's kin3 at r'.
* `p10f`, `kin3`, `p10_t`, the poses, `nav` and every label stay as in RH.
* **The pair is deliberately inconsistent.** Under `logged` the prior reads (own `p10f`, logged `base`), a pair that never occurs at inference: the T1 executor builds `base` from the own rolled kinematics, and T0 reads the logged pair.
* So this arm is a **diagnostic of where the T0 cost sits, not a deployable recipe**. A REPAIRS name says that the input `base` carries the cost. It does not say that KI should be deployed.

## 3. Reads and bars
T1-kin/hb0.2 (executor of record), with G-T0 and MECH4 at T0. The ROLLHIST card's section 3 binds, with the same executor, windows, cells, verdict rules, two-seed F_c and readers, frozen as in the KINLAB read with RH_AKIND=KI.

* **P1 (lever, decides):** KI − B0 under readout C\* (`R0C10XHb20`), all four families, STRATEGIC in full.
* **G-T0 (guard, decides):** KI − B0 at T0 ADE6. The upper interval bound must be at most **+0.0705 m** per seed. This is the frozen F_T0, INHERITED from one seed pair.
* **MECH4 (mechanism, decides the next lever only):** KI − RH at T0 ADE6, paired, per seed. MECH4-YES needs the interval below 0 at both seeds.
* **ATTR (beside, no bar):**
  * KI − RH under `R0C10XHb20` and under `R0Hb20`, four families: what keeping the logged `base` costs at T1;
  * KI − RK at T0;
  * P2, P3 and P4 as the ROLLHIST card defines them, on KI.
* **F_T0 at four seeds (beside, printed, never used here):** MEASURED 2026-11-23 in the KINLAB read.
  * Stage-A means 1.9671 / 2.0376 / 1.9202 / 1.9806 m; pairwise |differences| 0.0135 to 0.1174 m.
  * This card does NOT adopt it. The bar stays 0.0705. Adopting a four-seed floor is a PI decision for a later card, before that card's data.
* **Lever-acts check (known values, BLOCKED if any fails):**
  * The selfcheck's `dagger_probe` passes, with `base` unchanged on every selected row under `logged`, and the selected rows and the label behaving as in RH.
  * The built levers read `dagger.dagger_base = logged`.
  * The selected rows are greater than 0 at both seeds.
  * The trainer refuses `logged` at `--dagger-p 0` (code test, MEASURED at 4097a89e).

## 4. Pre-launch records (the trainer's selfcheck on the pod, before the lock)
* Per seed: the selected rows of the probe batch, and the share of selected rows whose own `base` differs from the logged `base` (computed from the same probe batch under `own`).
* This replaces the census of the build brief, which did not run on the dev box. It is recorded, and no bar is attached.

## 5. Outcomes, committed in advance
Two seeds; the outcomes are mutually exclusive. A one-seed read prints the same name prefixed PROVISIONAL.

| name | condition | reading | next lever |
|---|---|---|---|
| **KINPUT-REPAIRS** | G-T0 holds at both seeds; in P1, at least one family BETTER at both seeds and none WORSE at either | the own `base` input carried the T0 cost | a deployable form under a new card: the prior fed a `base` that is consistent at train and test, not this inconsistent pair |
| **KINPUT-HARMS** | G-T0 fails at both seeds, or a family WORSE at both | the own `base` is load-bearing for the own-action gains | MECH4-YES: the dose of logged-base samples, under a new card. MECH4-NO: the third seed pair (below) |
| **KINPUT-PARTIAL** | G-T0 holds at one seed only, or holds at both while P1 shows no family BETTER at both seeds | part of the cost may be the base input | MECH4-YES: as HARMS with MECH4-YES. MECH4-NO: the third seed pair |
| **KINPUT-SEED-DISAGREE** | a family BETTER at one seed and WORSE at the other, or the decisive cells straddle the floor | not decidable on two seeds | the third seed pair |
| **KINPUT-BLOCKED** | a gate check or the lever-acts check fails | the arm does not launch, or its number is not read | fix and re-gate |

**Precedence:** BLOCKED, then HARMS, then SEED-DISAGREE, then the rest in table order.

**The third seed pair** (named here so that it is not chosen after the data): B0_s2, RH_s2 and KI_s2, stage B on UE_s2.A (sha256 `6d434525…`), with banks generated on UE_s2.A as RH's were.
* It asks whether the s0 failure of G-T0, shared by four arms on B0_s0, is a property of seed 0.
* It goes back to the PI with its cost; it is not run on this card's go.

## 6. Cost and the return line
* **ESTIMATED:** about 0.5-0.6 pod-h of GPU lock for the two arms. MEASURED on the same banks and steps: RA 0.52 pod-h, RK 0.55 pod-h. The human label needs no label cache.
* **Return line to the PI:** 0.9 pod-h, projected from the RK chain's own timing.
* Before the lock, on the pod CPU at nice 19:
  * the tree at 4097a89e with the full blob check;
  * G-SUITE at 4097a89e (about 37 min);
  * a dry mint with the real argv.

No KI number exists.
