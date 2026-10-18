# PREREG R0e-D4: does the constraint temperature that tracks the pick for TTC also track it for NC? (T = 0.01, 0.02, 0.1 on the banked R0 states)

Written 2026-10-18 by an R0e-blockers eval agent, BEFORE any NC number of this addendum was computed. sha256 in `PREREG_R0E_D4_NC.sha256`. CPU only, no pod. Parent: `PK/raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.draft.md` §8 D4 (a draft, not registered; D4 is "descriptive", decides no row). Method parent: `PK/raw/2026-10-18-so-pickcon/PREREG_SOPICKCON_A2R2_TIEAWARE.md` (sha256 `81a8f3c9…`) and its harness `so_a2r2_tieaware.py`, reused unchanged. PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## 0. What I had seen (disclosure)
* A2R2 as banked (`a2r2_tieaware.json`, TTC only): the TTC diffs at T = 0.1 / 0.05 / 0.02 / 0.01 and its outcome `A2R2-ALIGNED(T*=0.02)`.
* The draft's D4 text, and its one NC number: R0d's u1000 B − own-A NC = −0.233 [−0.532, +0.033] (RL-val, from the draft; not re-read).
* **Keys only, never values:** `fullval_a1.json` and `socarve.json` (SO-CARVE) carry `P1_NC` / `P2_NC` entries at T = 0.1 for the R0 states. I have **not** opened their NC values. They are the T = 0.1 point of this question; this addendum re-derives it, and does not use it as a control (I have not verified that the two define the same quantity).
* I have NOT computed E_π[NC] at any T, nor the pick's NC, nor any NC difference between B and A on the banked states.

## 1. The question and why it is open
The constraint of R0e-pick holds **two** terms (`--epo-con g5`, comps TTC and NC) at **one** temperature (`--epo-con-temp`). A2R2 aligned the softmax with the pick for the TTC label only. The same T moves the NC term. If the policy at T = 0.01 still understates the pick's NC loss, the dual holds a quantity that is not the pick for NC, and an NC breach on RL-val would read as "held in-sample, breached on the pick" with no way to tell it from a generalisation gap.

## 2. Quantities (unchanged from A2R2 except the label)
Banked R0 states `scores_update{250,500,1000}_{A,B}.npz` (the arm's own B and its same-pass A), RL-val labels `pdm_val` / `terms_val`, 3,000 sets, 75 logs, 64 proposals per set. `l = log(sel_v1_lane(sigmoid(logits)) + AT.EPS)`, the selection score of the constraint's own policy `softmax(l / T)`. Label `y = AT.comps(tg)["nc"]` (head column 0). Per set and state, in points (×100), loss positive = arm B worse than arm A:
* `d_T = 100 (E_πB[y] − E_πA[y])`, `L_T = −mean_sets d_T` (over ALL 3,000 sets);
* `d_P = 100 (mean_{Q^B(τ)} y − mean_{Q^A(τ)} y)`, `Q(τ) = {j : l_j ≥ l_max − τ}`, `L_P1^τ = −mean_sets d_P`; the pick is the tie-aware pick of A2R2 with `τ0 = ε_eq = T_c ln(63 / ρ) = 2.3257e-3` (fixed by A2R, not chosen here);
* `diff_NC(T, u) = L_T(all) − L_P1^{τ0}(all)`. A negative diff means the tempered expectation understates the pick's NC loss.
Grid T ∈ {0.1, 0.05, 0.02, 0.01} (A2's). Headline statistic `M(T) = max_{u ∈ {250, 500, 1000}} |diff_NC(T, u)|`, reported for T ∈ {0.1, 0.02, 0.01}.
Beside it, descriptive only: the plain argmax pick (τ = 0), the level `E_π[NC]` and the pick's NC for each arm, the paired log-cluster bootstrap CI of each diff (10,000 resamples, seed 20260927, 75 logs: the A2R2 estimator, which answers *another draw of LOGS* and nothing about training or inference variance), and the NC label's support and the share of sets on which it varies across the 64 proposals.

## 3. Controls (the read is INVALID if any fails; INVALID is not an outcome about NC)
1. **Machinery reuse.** With `y` = the TTC label, this script's own code path reproduces the banked `a2r2_tieaware.json` `L_T_all` for all four T and the τ0 diff for all four T, at all three states, to |Δ| ≤ 1e-9. (`tempsweep_a2.json`'s T = 0.1 `L_T` is the second anchor, ≤ 1e-9.)
2. **Zero control.** B := A gives exactly 0.0 for `L_T` at every grid T and for `L_P1^τ` at every τ of {0, 1e-4, 1e-3, τ0, 1e-2}, for the NC label.
3. **Pick route.** `lB.argmax` and `lA.argmax` equal `AT.picks(logits)` (the RL-val selection rule, ties to the lowest index) on every set, and the number of exact top ties is 0. So the τ = 0 NC pick loss is the one the bar S reads.
4. **Planted arrays.** `so_a2r2_tieaware.selftest()` PASS (a 2-candidate near-tie with labels 0 / 1 reads 50 points at τ ≥ the gap and the argmax label at τ = 0; a clear set reads the same at every τ below its margin).

## 4. Committed outcomes (both are stated before the data; a determinate outcome needs controls 1-4 to pass)
**VACUITY GUARD (read first).** `V = max_u |L_P1^{τ0}(u)|`, the pick's own NC loss. If `V ≤ 0.25` the bar is as large as the quantity: the **null predictor** `L_T ≡ 0` is then already within 0.25 of the pick at every state, and "aligned" says nothing. The outcome is then **D4-NC-VACUOUS**, reported with `V` and with the null predictor's own `|0 − L_P1^{τ0}|` per state, and **no alignment claim is made at any T**.
Otherwise (`V > 0.25`), `ALIGNED_NC(T) ⇔ M(T) ≤ 0.25` points (A2R2's bar, unchanged, so the TTC and NC reads are comparable):
| outcome | condition | meaning for R0e-pick | next |
|---|---|---|---|
| **D4-NC-ALIGNED(0.01)** | `ALIGNED_NC(0.01)` and V > 0.25 | one T = 0.01 tracks the pick for both constrained terms on unconstrained R0 states. D4 stands as a descriptive companion to §1.2; nothing in the draft changes | none; the margin `0.25 − M(0.01)` goes beside §1.3 |
| **D4-NC-NOT-ALIGNED(0.01)** | `M(0.01) > 0.25` and V > 0.25 | the NC half of the constraint is a wedge of size `M(0.01)` at T = 0.01. It does **not** block R0e-pick (no bar changes; NC is watched at every check and never breached in R0d). It changes how an NC breach would be read: "held in-sample, breached on the pick" is then expected for NC, and is not evidence of a generalisation gap | a **per-component constraint temperature** (a second lever; PI): a `--epo-con-temp-nc` flag would need its own G-DVB entry and reachability test, 0 GPU first. The draft's §11 gains the sentence "NC is constrained at a temperature measured not to track the pick (M = …)" |
| **D4-NC-VACUOUS** | `V ≤ 0.25` | NC alignment is unmeasurable at the bar's resolution on these states | none; state in §11 that NC alignment at 0.01 is untested, not passed |
| **INVALID** | any control fails | the machinery, not NC, is in question | fix the failing control and re-run before any number is quoted |
Committed secondary classification, only when V > 0.25, because it feeds PI decision 2 of the draft's NOTES (T = 0.01 versus A2R2's T\* = 0.02):
* **NC-A**: `ALIGNED_NC(0.01)` and `ALIGNED_NC(0.02)`: NC does not separate the two; the T choice rests on TTC.
* **NC-B**: `ALIGNED_NC(0.01)` and not `ALIGNED_NC(0.02)`: NC favours 0.01, supporting the draft's recommendation.
* **NC-C**: `ALIGNED_NC(0.02)` and not `ALIGNED_NC(0.01)`: NC favours 0.02, against the draft's recommendation on NC; the PI's choice then trades TTC margin (0.01 better) against NC margin (0.02 better).
* **NC-D**: neither aligned: both colder temperatures leave an NC wedge; the T choice rests on TTC.

## 5. What this does not claim
* R0 is **unconstrained**: whether 0.01 stays aligned along a constrained trajectory is the in-run wedge D1, not this read. The R0d precedent (TTC): −0.317 in-run at T = 0.1.
* One R0 run per state: no training-seed or inference-seed variance. The CIs answer "another draw of logs".
* Offline selection tier, not T1. NC is a PDM term (collision avoidance); LATERAL and TACTICAL instruments do not exist at this tier; STRATEGIC is PAUSED (PI ruling 2026-09-19).
* No lever effect. The 0.25 bar is A2R2's; it was not tuned to this label.
