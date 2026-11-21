# E1e-A RESULT — BOUND, but the frontier MOVED into territory E1c never reached

**MEASURED 2026-07-28**, pod3. Train `rc=0` 06:34:12Z, eval `rc=0` 07:14:30Z.
Artifact: `pod3:/workspace/e1e/e1e_A_frontier.json`. Pre-registration: `PRE_REGISTRATION_E1E.md`,
committed **before** the run existed (`a7a2781`).
**Estimator: paired episode-cluster bootstrap** (`taniteval/ci.py`, B=2000) over the **44 held-out
episodes** (43 clusters at K=185). `overlapping_holdout_se` used nowhere.

## 0. Controls first

⭐ **Cross-run base reproduction PASSED EXACTLY.** The evaluator re-rolled REF-C base and returned
`ADE@2s 0.4747 / acc 0.6815 / l1 0.1775 / dep 0.5877 / junc_dep 0.8414 / peakXTE 38.944 /
OODpeak 1.266` — **identical on every field to E1c's base**. That is what makes E1e's numbers
*comparable* to E1c's rather than merely similar-looking.
**Leak-guard at train time:** buffer ids 362 × held-out ids 44 → **intersection 0**, byte level.
**Split provenance:** the held-out 44 is content-verified clean at the **sensor** level (0 shared
frames) against **this base arm's own training corpus** — `…/2026-07-28-e1-heldout-frames-leakcheck/`.

## 1. Verdict: BOUND (pre-registered OUTCOME B), 0 of 4 success points

| step | `dep_overall` Δ | `dep_junction` Δ | open-loop ADE@2s Δ [lo, hi] | P1 | P2 | Ga |
|---|---|---|---|---|---|---|
| 1000 | −0.2735 | −0.3694 | +0.1631 [+0.110, +0.222] | ✅ | ✅ | ❌ |
| 2000 | −0.3603 | −0.3315 | +0.1149 [+0.072, +0.163] | ✅ | ✅ | ❌ |
| **3000** | **−0.3911** | **−0.4369** | **+0.0958 [+0.053, +0.143]** | ✅ | ✅ | ❌ |
| 4000 | −0.3883 | −0.4270 | +0.0990 [+0.055, +0.149] | ✅ | ✅ | ❌ |

**P1 ∧ P2 fired at 4/4 (100 %)** — E1c managed 15/17. **Guardrails held at 0/4**; Ga, Gb1 and Gb2 are
the only failures at every step. Every open-loop CI still excludes zero, so **Ga is genuinely not
met** and this is **not** a success point.

## 2. ⭐ What actually changed: the frontier EXTENDED, it did not merely shrink

Matched-step comparison against E1c (same estimator, same 44 episodes, same evaluator):

| step | E1c `dep_ov` / ade | E1e-A `dep_ov` / ade |
|---|---|---|
| 1000 | −0.4180 / **+0.3371** | −0.2735 / **+0.1631** |
| 2000 | −0.3832 / **+0.2822** | −0.3603 / **+0.1149** |
| 3000 | −0.4273 / **+0.2133** | −0.3911 / **+0.0958** |
| 4000 | −0.4274 / **+0.1947** | −0.3883 / **+0.0990** |

⭐ **NO E1c CHECKPOINT — at any closed-loop level, anywhere on its 17-point frontier — REACHES AN
OPEN-LOOP COST BELOW +0.1893.** E1e-A reaches **+0.0958 while holding −0.3911**. At step 3000 it
retains **88.7 %** of E1c's best closed-loop gain (−0.3911 vs −0.4407) for **44.4 %** of the
open-loop cost (+0.0958 vs +0.2158).
⚠️ **Stated precisely, not overstated:** E1c's best point (−0.4407) still has a *larger* closed-loop
gain, so neither point dominates the other outright — both lie on the achievable frontier. The claim
is the weaker and true one: **E1e-A occupies a region of the trade-off that E1c never attained.**

## 3. The structural finding

E1e-A plateaus at **~+0.10 open-loop cost from step 2000**; E1c plateaued at **~+0.20 from step
2250** — same curve shape, half the height.
⇒ **`lam_replay` sets the ASYMPTOTIC open-loop cost; it does not remove the plateau.** Longer
training cannot close Ga for E1e-A either, for exactly the reason it could not for E1c. The weight,
not the schedule, is what moves this axis.

## 4. Consequence for the pre-registered ladder

The pre-registration fixed the rule in advance: **B is skipped iff A's closed-loop gain is already
destroyed (P1 false at every step) AND Ga still fails.** **P1 is TRUE at all four steps**, so **B is
NOT skipped.** `E1e-B (lam_replay = 8.0)` **LAUNCHED** 07:15:20Z on the freed pod3 — sequentially,
never alongside another job (C53), with a mechanical concurrency guard that was demonstrated
refusing.

⚠️ **What B can and cannot settle.** Extrapolating 1→3→8 is a ~2.7× reach off two points and is
**not** a prediction this document makes. B is a *measurement*. If B also comes back BOUND with P1
retained, the honest reading is that **`lam_replay` moves the asymptote but cannot reach Ga**, and the
remaining hypothesis is E1d's: the **target** is wrong, not its weight — the buffer supervises *all*
recoverable pre-failure states, while E1d measured junction recovery as cheap-and-monotone versus
overall-corridor recovery as expensive-and-barrier-crossing. The junction-restricted buffer is then
the next experiment, and it is a change to **what** is supervised.

## 5. Honest bounds

- n = 43 episode clusters at K=185, **6 junction windows** — junction CIs are wide; do not quote
  `dep_junction` as a precise effect size.
- `lam_replay` and `lam_cl` are **not independent** — only their ratio matters up to the LR schedule.
  A and B are two points on **one axis**, not two levers.
- This is a **fine-tune of one base checkpoint**, not a from-scratch result. Any eventual winner owes
  a from-scratch confirmation before entering a headline.
- The K=20 block is reported and **NON-DECIDING** by design.
