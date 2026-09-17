# PRE-REGISTRATION — D9 reward repair (`H-DDV2RL-3`)

**Written 2026-09-17, BEFORE any repaired arm has been run.** Both outcomes are committed
below. PI directive the same day: *"dont park D9, solve it and prove it."*

**Status: PRE-REGISTERED, NOT RUN.** Nothing in this file is a result.

---

## 1. What the diagnosis established, and what it did not

Landed `b17c709`
(`…/Research/2026-09-17-refcv6-d9-reward-attribution/`), MEASURED, T1, zero GPU:

| seed | measured failure | the blind reward term |
|---|---|---|
| **rl-s0** | off-road waypoint rate **+0.0151** [+0.0057, +0.0261] ⭐ SEPARATED (0.0165 → 0.0312, ~2×) | **DAC ≡ 1** — no maps on the RL-train split, so a binary MULTIPLIER with zero within-group variance contributed **exactly zero** to GRPO's advantage |
| **rl-s1** | signed speed bias **+0.3718 m/s** [+0.2708, +0.4777] ⭐ SEPARATED; **not** more off-road | **EP** rewards progress along the human's path with no speed-appropriateness term |

⭐ Each seed is separated on **its own** failure and **not** on the other's — a single
confound acting on both arms could not produce that.

⛔ **What it did NOT establish:** that either fix works; `n = 2` seeds, so each mechanism
rests on one seed showing it; and the EP → over-speeding link is the mechanism most
consistent with the measurement, not a measured causal chain.

---

## 2. Hypothesis

**`H-DDV2RL-3`** — *A PDMS-proxy reward whose DAC term carries real map information, and
whose EP term does not reward speed beyond what the situation warrants, removes the T1
harm that `H-DDV2RL-2` measured, without introducing a new one.*

⛔ **This is not a re-run of `H-DDV2RL-2`.** The reward changes, so it is a different arm
and gets its own pre-registration, its own bars, and its own controls.

---

## 3. Arms — ⛔ ONE VARIABLE EACH

The two defects must **not** be repaired in a single arm, or a result is
non-attributable — the `--v2` conflation failure (ten levers on two axes) and the C6
confound are the programme's standing precedents.

| arm | the ONE change vs `L1-RL` | why it exists |
|---|---|---|
| **`L2-DAC`** | DAC reads the SAM3 map (live, not ≡ 1) | tests seed 0's mechanism alone |
| **`L2-SPD`** | EP gains a speed-appropriateness term | tests seed 1's mechanism alone |
| **`L2-BOTH`** | both | the deliverable, IF and ONLY IF at least one single-lever arm clears its bar |
| **`L2-NORL`** | no RL (cold start, untouched) | the base every delta is read against |
| **`L2-REPL`** | `L2-NORL`'s flags, **different seed, zero levers moved** | ⛔ the RUN-TO-RUN floor. A separated CI answers *"would another draw of EPISODES say this?"*, never *"would another TRAINING RUN say this?"* |
| **`L2-REGRESS`** | DAC forced back to ≡ 1 **on a split that HAS maps** | ⛔ the deliberate-regression arm — see §6 |

**Held constant across every arm:** corpus, split, steps (600), batch, `--il-form matched`,
`--grad-clip 100`, the 9.47 M-parameter decoder subset, anchors, `n_anchors` 117, the T1
harness, and the 41 held-out episodes. Every arm runs **two seeds**.

⛔ **Preflight refuses the launch** if a diff of the actual launch commands shows any arm
differing from `L1-RL` in more than its named variable. Intent is not the check; the argv
audit is. *(MEASURED 2026-08-22: a row-bank arm changed `n` AND silently multiplied the
effective λ by ~n/24; two sweeps were invalidated.)*

---

## 4. Committed criteria — written before the data

### 4.1 PRIMARY (the harm guard, `H-DDV2RL-3a`)

* **PASS** — T1 `ade_m` of the arm is **not worse** than `L2-NORL` by more than the
  two-seed floor measured on `L2-REPL`, **on both seeds**, with the paired
  episode-cluster bootstrap.
* ⛔ **FAIL-HARM** — `ade_m` is worse than `L2-NORL` by **more than** that floor on
  **either** seed. *(`H-DDV2RL-2` failed this at **43×** its floor.)*
* **INCONCLUSIVE** — the arms separate by less than the floor, or `L2-REPL` shows the rig
  is noisier than the effect.

### 4.2 SECONDARY, and pre-registered as mechanism checks, not as capability

These are the two instruments from the diagnosis, committed **in advance** so they cannot
be chosen after seeing the data:

* **`L2-DAC` must reduce the off-road waypoint rate** toward the cold start's 0.0165, and
  must not exceed it. Bar: `L2-DAC − L2-NORL` off-road rate CI **includes or lies below**
  zero.
* **`L2-SPD` must reduce the signed speed bias** toward the cold start's +0.045 m/s.
  Bar: `L2-SPD − L2-NORL` signed-bias CI **includes or lies below** zero.
* ⛔ A mechanism check passing while §4.1 FAILS does **not** rescue the arm. The harm guard
  is primary.

### 4.3 SUCCESS for the programme

`H-DDV2RL-3` is **SUPPORTED** only if an arm (a) passes §4.1 on both seeds, **and**
(b) improves T1 `ade_m` over `L2-NORL` with a CI that excludes zero **and** exceeds the
`L2-REPL` floor, **and** (c) reports all four metric families, per family, never pooled.

⚠️ **One-seed separation is necessary, not sufficient.** Both seeds, against the replicate
floor, or the claim is not made.

---

## 5. Controls that must read known values

| control | what it must read |
|---|---|
| **the human's own driven path** (off-road instrument) | **≤ 0.05**. MEASURED tonight: 0.0171. ⛔ If it does not, the frame or the map is wrong and **no arm comparison is readable** — the verdict is gated on this in code (`CONTROL_BAR`), because on the first run it read 0.1741 through a frame error and would otherwise have printed a confident wrong answer |
| **`L2-NORL` − human**, off-road | not separated from zero (MEASURED tonight: −0.0009 [−0.0040, +0.0018]) |
| **`ha0`** (hold action, straight ahead at `v0`) | a second, dumber reference on both instruments |
| **`L2-REPL`** | the run-to-run floor; any effect smaller than it is not an effect |

---

## 6. ⛔ The deliberate-regression arm

`L2-REGRESS` runs the repaired pipeline **with DAC forced back to ≡ 1**, on a split that
**does** have maps — so the only difference from `L2-DAC` is that the term carries no
information.

**It must reproduce the harm.** If `L2-REGRESS` does *not* come out worse than `L2-DAC`,
then the DAC repair is not what changed anything, and a PASS on `L2-DAC` means nothing.
*(Standing rule: a gate that does not FAIL the reintroduced defect cannot certify the
fix.)*

---

## 7. Splits, and what is not tuned on what

* **Fit / train:** the RL-train split — ⛔ **and it must now carry SAM3 maps**, which is
  the dependency that made DAC dead in the first place. SAM3 corpus production finishes
  ≈2026-09-22; **this pre-registration does not authorise starting before its maps cover
  the RL-train split**, because a partially-covered split reintroduces exactly the
  constant-DAC defect on the uncovered part.
* **Held out:** the 41 episodes, untouched, scored once.
* ⛔ No hyper-parameter — reward weights included — is selected on the scored split.

---

## 8. Tier and reporting

**T1 is primary** (self-action open loop). T0 may be reported as a diagnostic and ⛔ may
never be quoted as driving performance — `H-DDV2RL-2` is the programme's clearest case of
a T0 endpoint PASSING (§13.2 SUCCESS, Δfan +0.0363 / +0.0616) while T1 read FAIL-HARM.

Every number carries its evidence class, its tier, its `n`, and its estimator
(`paired_episode_cluster_bootstrap`). All four metric families — longitudinal, lateral,
tactical, strategic — **in addition** to ADE, per family, never pooled.

---

## 9. What would make me abandon this direction

Committed in advance: if **both** single-lever arms pass their mechanism checks (§4.2) —
the off-road rate and the speed bias both return to the cold start's level — and T1
`ade_m` **still** fails §4.1, then the harm is **not** caused by either missing reward
term, both mechanisms measured tonight are insufficient, and D9's problem is not the
reward. That outcome is reported as a refutation of `H-DDV2RL-3`, and the next lever is
chosen from the selection evidence (item 19: a collision gate is worth **62.4 %** of the
oracle gap from **5.1 %** of windows), not from a third reward patch.
