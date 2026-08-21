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
| ⭐ **`random` — UNTRAINED** | **0** | 6.89 | — | **−0.0156** | .5082 | **.5659** |
| ⭐ **supervised CONTROL** | 5 k | 1.58 | — | **+0.9934** | **.9941** | **.9958** |
| *ref `dino_pooled`* | — | *2.47* | — | *+0.3792* | *.8312* | *.8236* |
| *ref `v6_cells`* | — | *4.56* | *7.83 stalled* | *−0.0176* | *.5321* | *.5890* |

## 2. ⭐⭐ The two controls, and what they bracket

**`supervised` — the ceiling.** Same encoder, same frames, same steps, same
folds, trained on the probe targets: `lead_gap_m` **R² 0.9934**, `ego_speed`
0.9758, occupancy **AUC 0.994/0.996**. ⇒ **The architecture and the data CAN hold
this content.** Not scale, not data, not the encoder, not the probe.

⚠️ It is **representability, not generalisation** — it saw the held-out episodes'
labels. That is still the right control here, because every WM arm also trains on
all 130 clips and is probed identically; the arms differ only in whether the
signal *asks* for the content. Never quote it as "0.99 on held-out episodes".

**`random` — the floor, and the finding.** An **untrained** encoder scores
`lead_gap_m` **−0.0156**, occupancy **.5082 / .5659**.

⇒ ⛔ **EVERY TRAINED WORLD-MODEL ARM LANDS AT OR BELOW THE UNTRAINED FLOOR.**
`lewm` at 20 k is **worse** than random. `wsig` at 20 k is **worse** than random.
Random's **.5659** is the **best non-supervised occupancy number in the table**.

**The objective contributes nothing on this axis. Not a little — nothing.**

## 3. Five confounds excluded, each by measurement

| confound | how excluded |
|---|---|
| **capacity / data / encoder / probe** | supervised control reaches R² 0.99 on the same everything |
| **undertraining** | 4× steps (5 k → 20 k) changes nothing: `lewm` −0.0257 → −0.0219, `wsig` −0.0068 → −0.0360 |
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

## 6. ⛔ v6.5f — what the evidence supports, and the decision that is NOT mine

**No unsupervised fix was validated.** `wsig` fixes a real mechanism
(between/within, 5.4×, reproducibly) and buys **zero** decodability. So a v6.5f
built only from tonight's unsupervised candidates would ship a change with no
measured benefit.

**What the table actually says:** the only two representations in this entire
programme that carry environment content are **supervised** (R² 0.99) and a
**foundation encoder** (`dino_pooled`, R² 0.38 / AUC .83). Both had something
that *asked* for the content. Every purely self-supervised predictive arm — ours
and this replication — sits at the random floor.

⇒ **The candidate that follows from the evidence is an auxiliary perception head
on `obstacle.offline` cuboids (97.44 % corpus coverage), grounding the trunk the
way REF-A's `--aux-egomotion` reached `aux_speed_r2` 0.9825 while v6 reads
−0.005.**

⛔ **I am not building that unilaterally, because it is a THESIS decision, not an
engineering one.** `D-003` makes the from-scratch **unsupervised** 4-brain latent
world model the main track and calls that *"what makes the data-efficiency claim
disruptive"*. Adding perception supervision to the trunk changes what the
programme is claiming. **That is the PI's call**, and tonight's job was to make
it an informed one.

**If the PI says yes**, v6.5f is: aux perception head (lead-gap / occupancy /
agent-count) on the trunk + `wsig`-style within-episode SIGReg + SIGReg moved to
the encoder embeddings per LeWM Fig. 1 + the detach reconsidered. The first is
the only one with a measured effect on decodability; the rest are cheap and
principled. All are **loss-side** except the head's new keys, which
`STAGE_MAY_INTRODUCE` admits at a stage boundary — so **no `d_op` change, no
70-tensor reshape, no full retrain.**

## 7. Recorded defects in this harness

* ⛔ **`train` overwrote `e_lewm_train.json`** instead of appending, twice
  dropping arms from the scoring list. Fixed; records rebuilt from logs + disk.
* ⚠️ **`d2048` would carry 7.37 M params vs 5.44 M** — declared, never run.
* ⚠️ **Two seeds, not the pre-registered three**, and the four ablation axes were
  never reached because the gate never opened.
