# E-LEWM-1 — the world-model objective adds NOTHING over a random encoder, and five confounds are excluded

`MEASURED (ours; dev-box RTX 4060)` · **T0-DIAGNOSTIC** · pre-registered in
`PREREG_E_LEWM_1.md` (**e4d58be, committed before any number**) · scored on the
**same 5,617 keys, same order, same episode-disjoint folds** as E-TRUNK-2, so
every row sits in the same table as `v6_cells` and `dino_pooled` ·
**Thor untouched throughout.**

⛔ **THE PRE-REGISTERED GATE DID NOT PASS**, so the four-way ablation
(`d2048`/`sigop`/`detach`/`terms7`) was **never run** and nothing about those
axes is claimed. What the controls establish is much sharper than the ablation
would have been.

---

## 1. Result

| arm | steps | between/within | SIGReg | `lead_gap_m` R² | `left_occ` | `right_occ` |
|---|---|---|---|---|---|---|
| `lewm` s0 | 5 k | 16.25 | 30.51 → 1.636 | −0.0257 | .4803 | .5385 |
| `lewm` s1 | **20 k** | 13.74 | 27.32 → 1.562 | −0.0219 | .5707 | .4011 |
| `wsig` s0 | 5 k | **2.99** | 18.20 → 2.041 | −0.0068 | .4858 | .5118 |
| `wsig` s1 | **20 k** | **2.64** | 18.21 → 1.879 | −0.0360 | .5043 | .4908 |
| **`aux`** s0 | 8 k | **2.80** | 18.20 → 1.387 | −0.0040 | .4822 | .5158 |
| **`aux`** s1 | 8 k | **2.71** | 18.21 → 1.913 | −0.0094 | .4744 | .5016 |
| ⭐ **`random` — UNTRAINED** | **0** | 6.89 | — | **−0.0156** | .5082 | **.5659** |
| ⭐ **supervised CONTROL** | 5 k | 1.58 | — | **+0.9934** | **.9941** | **.9958** |
| *ref `dino_pooled`* | — | *2.47* | — | *+0.3792* | *.8312* | *.8236* |
| *ref `v6_cells`* | — | *4.56* | *7.83 stalled* | *−0.0176* | *.5321* | *.5890* |

## 2. ⭐⭐ The two controls bracket everything, and the bracket is empty

**`supervised` — the ceiling.** Same encoder, frames, steps, folds; trained on
the probe targets: `lead_gap_m` **R² 0.9934**, `ego_speed` 0.9758, occupancy
**AUC 0.994/0.996**. ⇒ **The architecture and data CAN hold this content.**

⚠️ It is **representability, not generalisation** — it saw the held-out episodes'
labels. Still the right control, since every WM arm also trains on all 130 clips
and is probed identically; the arms differ only in whether the signal *asks*.
Never quote it as "0.99 on held-out episodes".

**`random` — the floor, and the finding.** An **untrained** encoder scores
`lead_gap_m` **−0.0156**, occupancy **.5082 / .5659**.

⇒ ⛔ **EVERY TRAINED ARM — `lewm`, `wsig`, `aux`, at 5 k, 8 k and 20 k steps, on
two seeds — LANDS AT THE UNTRAINED FLOOR.** Random's **.5659** is the best
non-supervised occupancy number in the table. **Training contributes nothing on
this axis. Not a little — nothing.**

### 2.1 ⛔ And the aux perception head does NOT transfer

`aux` = the LeWM objective plus a head supervised on **`n_agents_log` only**,
probed for **`lead_gap_m` and occupancy** — deliberately different quantities, so
the test is not circular.

**It fixes the mechanism** (b/w 2.80 / 2.71, matching `wsig`) **and transfers
nothing**: −0.0040 / −0.0094, occupancy at chance, both seeds.

⭐⭐ **THIS IS THE RESULT THAT CHANGES THE v6.5f RECOMMENDATION.** Grounding on one
perception target does **not** induce a generally environment-informative
representation — it encodes what it is supervised on and no more. The supervised
control reaches 0.99 **because it was trained on lead-gap directly**. ⇒ *"Add an
auxiliary perception head to ground the trunk"* — the candidate I proposed hours
ago, and the mechanism REF-A uses — **does not do what its name suggests.**

⚠️ **Bounded claim:** this refutes **this** aux design — a *linear* head on **one
scalar**. A richer auxiliary (dense occupancy grid, per-agent positions) is
untested and may behave differently. What is refuted is the general hope that
*any* perception grounding pulls the whole representation along with it.

## 3. Five confounds excluded, each by measurement

| confound | how excluded |
|---|---|
| **capacity / data / encoder / probe** | supervised control reaches R² 0.99 on the same everything |
| **undertraining** | 4× steps (5 k → 20 k) changes nothing: `lewm` −0.0257 → −0.0219, `wsig` −0.0068 → −0.0360 |
| **missing perception grounding** | `aux` supervises `n_agents_log` and transfers **nothing** to lead-gap or occupancy, on both seeds |
| **the tick was trivial** | MEASURED: at k=1 the latent moves 1.12 % of its magnitude (identity explains 98.9 %); fixed to a 1.0 s tick with a 3-step autoregressive roll, raising `pred` loss 6× |
| **between/within dominance** | `wsig` cuts it **16.25 → 2.64**, reproducibly across seeds, landing beside `dino_pooled`'s 2.47 — **and decodability still does not follow** |
| **SIGReg failing** | SIGReg **converges** in every arm (30.5 → 1.56), unlike v6's stall at 7.83 |

## 4. ⚠️ What is NOT excluded — the honest boundary

⛔ **That my implementation is an unfaithful LeWM replication cannot be ruled
out.** The pre-registered gate exists precisely to detect that, and **it failed**.
I did **not** attempt LeWM on its own benchmark (Push-T), which is the only way to
separate *"the objective does not transfer to driving"* from *"my code is wrong"*.
**This is the single largest limitation of E-LEWM-1 and it is not a small one.**

⭐ **What raises the result above "probably a bug", though:** the harness
**reproduces v6's failure signature** — v6_cells decodes −0.0176 with b/w 4.56;
these arms decode ≈0 with b/w 2.6–16. A 5.4 M model on 130 clips lands where a
336 M model on 2,376 episodes lands. That is consistent with a real property of
the objective-on-driving, not with an isolated coding error.

## 5. ⚠️ The hypothesis this leaves — still untested

In a **forward driving camera** the next-latent target is dominated by
**ego-motion-induced optical flow**: to predict the next latent given (steering,
accel) you must model how the scene sweeps past, and other agents are a small,
partly stochastic residual. In Push-T the action moves **the agent**, and the
agent and block **are** what changes — the objective cannot succeed without
encoding them.

⇒ **The same objective may reward different content in the two domains.**
Recorded as a hypothesis with a mechanism. *(Refuted tonight, in order:
collapse-onto-ego · "2.3 of 2048 dims" (C128) · "collapse" as framing · "freeze
the encoder" (C129) · "the self-target is the problem" · Diaconis–Freedman ·
"undertraining". Seven. This one waits for a measurement too.)*

## 6. ⛔ v6.5f is NOT built, and the reason changed during the night

**Nothing tested tonight produces environment decodability.** Not the LeWM
objective, not at 4× training, not with the between/within repair, and **not with
perception grounding**. Every arm sits at the untrained floor. A v6.5f assembled
from these candidates would ship changes with **zero measured benefit**.

⚠️ **And the recommendation I gave the PI hours ago is now refuted by my own
experiment.** I proposed an auxiliary perception head as *the* candidate, citing
REF-A's `--aux-egomotion` (`aux_speed_r2` 0.9825). The `aux` arm tests exactly
that idea and it **transfers nothing** (§2.1). Grounding on one perception target
encodes that target and nothing else. **The name "grounding" was doing work the
mechanism does not do.**

**What the whole table says, plainly:** the only representations in this
programme that carry environment content are **(a) directly supervised on the
quantity being read** (R² 0.99) and **(b) a foundation encoder** (`dino_pooled`
R² 0.38 / AUC .83). There is no third case. Every purely predictive
self-supervised arm — ours at 336 M and this replication at 5.4 M — is at the
random floor.

⇒ ⛔ **The programme's central bet — that an unsupervised predictive objective on
2,376 driving episodes yields a trunk carrying environment structure — has no
supporting measurement anywhere in this repository, and now has direct evidence
against it at small scale.**

**That is a finding about the THESIS, not an engineering defect, so the next move
is the PI's.** The three live options, with what each costs:

| option | what it means | evidence |
|---|---|---|
| **A — keep the bet, fix the replication** | run LeWM on **Push-T** first to prove the harness, then re-test on driving | ⚠️ §4 — the unfaithful-replication risk is real and unexcluded |
| **B — supervise the trunk** on the quantities the hierarchy needs (lead-gap, occupancy) from `obstacle.offline` | ✅ the only thing measured to work (0.99), ⛔ **but it changes what `D-003` claims** and §2.1 shows it will only give you what you supervise | measured |
| **C — foundation encoder** (REF-A v1's route) | ✅ measured to carry the content (.83 AUC), ⛔ `D-003` calls it a comparison arm, and REF-A v1's planner blocker is open | measured |

**No option is free, and I am not choosing between them** — that is exactly the
tier/decision boundary C129 was logged for.

⚠️ **What I would NOT do:** build a v6.5f now. Every candidate available tonight
is either refuted (`aux`), null (`wsig`, `lewm`), or a thesis change (B, C).

## 7. Recorded defects in this harness

* ⛔ **`train` overwrote `e_lewm_train.json`** instead of appending, twice
  dropping arms from the scoring list. Fixed; records rebuilt from logs + disk.
* ⚠️ **`d2048` would carry 7.37 M params vs 5.44 M** — declared, never run.
* ⚠️ **Two seeds, not the pre-registered three**, and the four ablation axes were
  never reached because the gate never opened.
