# E1c — RESULTS: held-out-gated closed-loop SFT, the FULL FRONTIER

**VERDICT: `BOUND`** — and it is the informative kind. Across the **full 17-point
frontier** the pre-registered primary **fired at 15 of 17 checkpoints**
(corridor-departure@K185 CI-separated LOWER on **both** overall and junction), while
the **held-out guardrails held at 0 of 17**. The intersection is **empty**. Per
`PRE_REGISTRATION_E1C.md §4.2` that means **the closed-loop / open-loop trade is REAL
and not an artifact of the guard.** E1b's BOUND was not caused by the broken guard;
fixing the guard did not reveal a hidden good checkpoint. Nothing was re-tuned,
re-thresholded, or re-selected.

`2026-07-26 (Europe/Berlin; pod logs are UTC)` · `tanitad-pod3` (A40) ·
renderer-free imagination / kinematic closed loop (**NOT** AlpaSim). PI: Sayed.
Train `00:38:49Z → 02:17:09Z` (5,694.7 s). Frontier eval `02:17:17Z → 06:18:02Z`
(14,436.1 s, rc=0).

Evidence-class legend: `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) ·
`INHERITED` · `ESTIMATED` · `HYPOTHESIS`. Every number in §5–§9 is `MEASURED` from
`e1c_frontier_result.json` / `ft_run/heldout_gate.jsonl` in this directory and was
**generated** into these tables by `scripts/summarize_e1c.py` — never re-typed.

---

## 1. The pre-registration, restated BEFORE any number

`PRE_REGISTRATION_E1C.md` was written and staged **before** the fine-tune
launched. Its rules, verbatim in substance:

A checkpoint `c` on the frontier is a **SUCCESS POINT** iff all six hold:

| id | condition |
|---|---|
| **P1** | corridor-departure@K185 **overall**, paired Δ(c−base) **CI-separated LOWER** |
| **P2** | corridor-departure@K185 **junction**, paired Δ(c−base) **CI-separated LOWER** |
| **Ga** | held-out open-loop **ADE@2s** paired Δ **includes 0 or is lower** (not separated-worse) |
| **Gb1** | held-out open-loop **anchor_acc** not separated-worse |
| **Gb2** | held-out open-loop **anchor_traj_l1** not separated-worse |
| **Gc** | closed-loop **OOD peak ratio** ≤ 1.30 (the measured band) |

- **SUCCESS** = at least one such checkpoint exists; the frozen selection rule
  (most negative overall Δ, ties by smaller open-loop ADE Δ, then earlier step)
  names the deliverable.
- **BOUND** = no such checkpoint exists anywhere on the frontier ⇒ **the
  closed-loop / open-loop trade is real and not an artifact of the guard.**
  Equally publishable, reported as such.

**Estimator.** Every interval is the **paired episode-cluster bootstrap**
(`taniteval/ci.py`, **B = 2000**, resampling the 44 held-out **episodes**).
`overlapping_holdout_se` appears **nowhere** in the E1c pipeline.

---

## 2. What changed vs E1b — exactly two things

E1b's diagnosed defect, in one line: **the forgetting guard was monitored on the
corpus it replays.** Its replay loss *fell* (1.826 → 1.613) while held-out
open-loop ADE *rose* 41 %. A training-set loss cannot be a generalization guard.
*(Root-cause class for `RETRACTION_LOG.md`: **training-set instrument used as a
generalisation guard** — sibling of "trainer val is not eval output".)*

| | E1b | E1c |
|---|---|---|
| base ckpt | `refc-diffusion-base-v21-30k/ckpt.pt` step 29999 | **identical** |
| mined buffer | 3,537 states / 362 parity-train eps | **identical, REUSED — md5 `a32cfe9bfea4b1b5c196d3bb7f71fa5f` verified at launch, not regenerated (saves 2.3 h)** |
| seed / lr / warmup / schedule | 0 / 2e-5 / 100 / cosine 4000 | **identical** |
| cl-batch / replay-batch | 16 / 16 | **identical** |
| `lam_cl` / `lam_replay` | 1.0 / 1.0 | **1.0 / 1.0 — held fixed on purpose** |
| encoder | frozen, 13,732,945 trainable / 90,458,632 frozen | **identical (re-asserted at startup)** |
| **guard** | replay loss on the replayed corpus | **held-out open loop, paired vs base, at every checkpoint** |
| **checkpointing** | one rolling ckpt (endpoint only) | **17 step-tagged checkpoints → 18 frontier points** |

`lam_replay` is **not** a lever in E1c. Attributability beats squeezing the best
number out of this run; λ is the next experiment if the frontier says so.

## 3. THE EVALUATOR SELF-TEST — run BEFORE the deciding run

**Why this section exists.** E1b's shipped `e1b_eval.py` would have reported
**SUCCESS on a BOUND run**: its verdict string was derived from the primary
alone and never consulted the guardrails, several guardrails were unimplemented,
and the open-loop guardrail was unpaired. *A guardrail absent from the code is a
comment, not a guardrail.*

So before E1c's fine-tune produced a single number, `scripts/e1c_selftest.py`
drove the **shipped verdict code path** — `e1c_common.frontier_point_stats` →
`evaluate_point` → `select_winner` → `render_verdict`, the exact functions
`e1c_eval.py` calls — with **synthetic per-window arrays built to fail each
registered guardrail one at a time**.

**Result: `43 / 43` checks passed, `ALL_PASS = true`
(`e1c_selftest_result.json`, re-run on the final shipped code).**

| case | construction | required verdict | got |
|---|---|---|---|
| C0 | everything clean | SUCCESS | ✅ SUCCESS |
| C1 | open-loop ADE@2s separated **worse** (+0.1947) | BOUND (Ga) | ✅ BOUND, `failed=['Ga_openloop_ade2s_ok']` |
| C2 | anchor_acc separated **worse** (−0.0651) | BOUND (Gb1) | ✅ BOUND, `failed=['Gb1_anchor_acc_ok']` |
| C3 | anchor_traj_l1 separated **worse** (+0.0624) | BOUND (Gb2) | ✅ BOUND, `failed=['Gb2_anchor_traj_l1_ok']` |
| C4 | OOD peak 1.42 (out of the measured band) | BOUND (Gc) | ✅ BOUND, `failed=['Gc_ood_in_band']` |
| C5 | junction stratum an **exact null** | BOUND (P2) | ✅ BOUND, `failed=['P2_…']` |
| C6 | overall stratum not separated | BOUND (P1) | ✅ BOUND, `failed=['P1_…']` |
| C7 | closed loop separated **worse** | BOUND (P1+P2) | ✅ BOUND |
| **C8** | **E1b's REAL measured numbers replayed** | **BOUND** | ✅ **BOUND** — primary fires, all three held-out guardrails fail |
| C9 | 5-point frontier, only the middle qualifies | SUCCESS @ that step | ✅ winner 1000; the *larger* closed-loop gains at 2000/3000/4000 correctly REJECTED |
| C10 | 3 qualifying points | frozen selection rule | ✅ picks the most negative overall Δ |
| C11 | frontier with no qualifying point | BOUND | ✅ BOUND, `winner=None` |
| C12 | descriptors emitted and non-deciding | — | ✅ |

**C8 is the load-bearing one:** fed E1b's actual measured deltas, E1c's
evaluator renders **BOUND**, where E1b's shipped evaluator emitted
`"SUCCESS … Check guardrails."` Every case also asserts that the synthetic
construction *really* tripped the intended condition and **no other**, so a case
that silently stopped testing what it claims fails loudly instead of passing
vacuously.

**Scope, stated honestly.** The self-test exercises **100 % of the deciding path
from per-window arrays onward** — estimator, the six predicates, the frozen
selection rule, the verdict string. It does **not** re-verify the rollout
itself; that is E1a's `e1a_horizon.rollout` reused verbatim, whose base arm
reproduced E1a bit-identically in E1b and is re-rolled again here as a control
(§4). The evaluator was additionally executed end-to-end on real GPU rollouts
before the deciding run (`e1c_evaluator_smoke_result.json`, 4 episodes, 2
checkpoints, verdict rendered, incremental banking exercised).

---

## 4. REPRODUCTION CONTROLS — E1c is a faithful re-run, so the guard really is the only changed variable

Three independent controls, all `MEASURED` this session.

**(1) The training trajectory is BIT-IDENTICAL to E1b.** All **161 / 161** logged
rows of `ft_run/train_log.jsonl` match E1b's `clsft_train_log.jsonl` on every field
(loss, cl_loss, cl_traj, cl_cls, cl_anchor_acc, rp_loss, rp_traj, rp_cls, rp_law,
rp_anchor_acc, gnorm, lr) — `step_s` excluded as wall-time. **0 differing rows.**
The 18 in-training held-out probes therefore provably did **not** perturb the run:
the RNG save/restore around each probe worked. E1c is E1b plus instrumentation, and
nothing else.

**(2) The base arm reproduces E1a and E1b a third time**, re-rolled here on the same
44 held-out episodes:

| metric | E1c re-roll | E1a/E1b reference |
|---|---|---|
| K185 corridor-departure, overall | **0.587681** | 0.5877 |
| K185 corridor-departure, junction | **0.841441** (6 w / 6 ep) | 0.8414 |
| K185 peak \|XTE\| (m) | **38.944473** | 38.9445 |
| K185 OOD peak ratio | **1.266394** | 1.2664 |
| K20 corridor-departure, overall | **0.005326** | 0.0053 |
| open-loop ADE@2s (m) | **0.474666** | 0.4747 |
| open-loop anchor_acc | **0.681489** | 0.6815 |
| open-loop anchor_traj_l1 | **0.177492** | 0.1775 |

**(3) The step-4000 endpoint reproduces E1b's fine-tuned arm exactly.**
K185 dep **0.1603**, junction **0.4144**, peak \|XTE\| **3.0415**, OOD **1.1339**;
open loop ADE@2s **0.6693**, anchor_acc **0.6163**, anchor_traj_l1 **0.2399** —
every figure identical to `e1b_eval_result.json` at the reported precision, and the
paired deltas (−0.4274 / −0.4270 / +0.1947 / −0.0651 / +0.0624) are identical too.

*One honest wrinkle.* The **in-training** probe reports base ADE@2s **0.4738** where
the **evaluator** reports **0.4747** (0.19 % apart), because REF-C's 2-step diffusion
decode consumes RNG and the probe runs from a different RNG state. The gate is
internally consistent (both arms of every gate delta come from the same probe on
identical windows), and **every deciding number in this report comes from the
evaluator**, not the probe. Stated rather than smoothed.

### 4.1 Frontier checkpointing is lossless — MEASURED, not assumed

Each of the 17 checkpoints stores only the **non-encoder** parameters
(13,732,945 ≈ 55 MB) rather than the 417 MB full state dict. Asserted at startup:
trainable params **==** non-encoder params (13,732,945 both ways), and the **only**
non-encoder buffer is the constant `decoder.anchors`, and the encoder is frozen
*including* its BatchNorm running stats. At the end of the run a full state dict was
written and reconstructed from `base ⊕ delta`:
**`OVERLAY CHECK base(+)delta == full : True (nonzero-diff keys 0, key-set diff 0)`.**
The evaluator additionally re-asserted, per checkpoint, that the delta carries no
missing non-encoder key and that `decoder.anchors` never drifted.

---

## 5. THE FRONTIER (the deliverable)

### FRONTIER — closed-loop gain vs open-loop cost, per checkpoint

| step | CDR@K185 overall Δ | CDR@K185 junction Δ | open-loop ADE@2s Δ | anchor_acc Δ | anchor_traj_l1 Δ | OOD peak | P1 | P2 | Ga | Gb1 | Gb2 | Gc | **SUCCESS** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0 (base)** | — (ref) | — (ref) | — (ref) | — (ref) | — (ref) | 1.2664 | — | — | — | — | — | — | — |
| 100 | +0.2102 [+0.1432, +0.2827] **SEP** | -0.0243 [-0.0793, +0.0270] | +0.4542 [+0.3793, +0.5285] **SEP** | -0.1830 [-0.2347, -0.1322] **SEP** | +0.1968 [+0.1689, +0.2252] **SEP** | 1.2919 | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 250 | +0.0415 [-0.0439, +0.1309] | -0.0946 [-0.1766, -0.0162] **SEP** | +0.5744 [+0.4611, +0.6836] **SEP** | -0.2089 [-0.2754, -0.1490] **SEP** | +0.1653 [+0.1441, +0.1862] **SEP** | 1.2634 | **FAIL** | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 500 | -0.2417 [-0.3384, -0.1458] **SEP** | -0.2180 [-0.4757, -0.0441] **SEP** | +0.5048 [+0.3992, +0.6101] **SEP** | -0.1706 [-0.2293, -0.1169] **SEP** | +0.1427 [+0.1209, +0.1660] **SEP** | 1.1962 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 750 | -0.2933 [-0.3764, -0.2161] **SEP** | -0.2180 [-0.4604, -0.0505] **SEP** | +0.3255 [+0.2554, +0.3950] **SEP** | -0.1303 [-0.1756, -0.0909] **SEP** | +0.0937 [+0.0770, +0.1118] **SEP** | 1.1826 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 1000 | -0.4180 [-0.5022, -0.3330] **SEP** | -0.3910 [-0.6279, -0.1613] **SEP** | +0.3371 [+0.2477, +0.4388] **SEP** | -0.1220 [-0.1644, -0.0837] **SEP** | +0.0871 [+0.0721, +0.1036] **SEP** | 1.1483 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 1250 | -0.3821 [-0.4771, -0.2856] **SEP** | -0.3631 [-0.6171, -0.1090] **SEP** | +0.3174 [+0.2541, +0.3794] **SEP** | -0.1003 [-0.1344, -0.0653] **SEP** | +0.0899 [+0.0739, +0.1071] **SEP** | 1.1722 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 1500 | -0.3691 [-0.4620, -0.2734] **SEP** | -0.4261 [-0.6793, -0.1630] **SEP** | +0.2988 [+0.2431, +0.3559] **SEP** | -0.1138 [-0.1612, -0.0683] **SEP** | +0.0939 [+0.0778, +0.1104] **SEP** | 1.1801 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 1750 | -0.3902 [-0.4844, -0.2964] **SEP** | -0.4261 [-0.6793, -0.1684] **SEP** | +0.2714 [+0.2076, +0.3386] **SEP** | -0.0910 [-0.1303, -0.0558] **SEP** | +0.0885 [+0.0724, +0.1056] **SEP** | 1.1593 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 2000 | -0.3832 [-0.4768, -0.2929] **SEP** | -0.4135 [-0.6559, -0.1559] **SEP** | +0.2822 [+0.2140, +0.3573] **SEP** | -0.0931 [-0.1343, -0.0549] **SEP** | +0.0934 [+0.0768, +0.1128] **SEP** | 1.1694 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 2250 | -0.4401 [-0.5310, -0.3471] **SEP** | -0.4396 [-0.7072, -0.1604] **SEP** | +0.2197 [+0.1677, +0.2715] **SEP** | -0.0703 [-0.1012, -0.0383] **SEP** | +0.0814 [+0.0662, +0.0967] **SEP** | 1.1455 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 2500 | -0.4300 [-0.5208, -0.3380] **SEP** | -0.4441 [-0.7117, -0.1595] **SEP** | +0.2083 [+0.1554, +0.2659] **SEP** | -0.0672 [-0.0981, -0.0342] **SEP** | +0.0643 [+0.0525, +0.0769] **SEP** | 1.1436 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 2750 | -0.4407 [-0.5319, -0.3482] **SEP** | -0.4414 [-0.7063, -0.1631] **SEP** | +0.2158 [+0.1509, +0.2826] **SEP** | -0.0703 [-0.1026, -0.0373] **SEP** | +0.0657 [+0.0526, +0.0795] **SEP** | 1.1340 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 3000 | -0.4273 [-0.5147, -0.3370] **SEP** | -0.4450 [-0.7072, -0.1667] **SEP** | +0.2133 [+0.1517, +0.2791] **SEP** | -0.0631 [-0.0950, -0.0321] **SEP** | +0.0624 [+0.0499, +0.0756] **SEP** | 1.1380 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 3250 | -0.4205 [-0.5086, -0.3315] **SEP** | -0.4396 [-0.7054, -0.1595] **SEP** | +0.1893 [+0.1383, +0.2413] **SEP** | -0.0620 [-0.0933, -0.0310] **SEP** | +0.0623 [+0.0501, +0.0747] **SEP** | 1.1482 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 3500 | -0.4269 [-0.5159, -0.3388] **SEP** | -0.4982 [-0.7495, -0.2216] **SEP** | +0.2026 [+0.1452, +0.2641] **SEP** | -0.0683 [-0.1014, -0.0372] **SEP** | +0.0632 [+0.0515, +0.0749] **SEP** | 1.1480 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 3750 | -0.4303 [-0.5212, -0.3403] **SEP** | -0.4928 [-0.7334, -0.2333] **SEP** | +0.1969 [+0.1414, +0.2578] **SEP** | -0.0610 [-0.0919, -0.0310] **SEP** | +0.0625 [+0.0503, +0.0747] **SEP** | 1.1374 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |
| 4000 | -0.4274 [-0.5161, -0.3378] **SEP** | -0.4270 [-0.6838, -0.1648] **SEP** | +0.1947 [+0.1415, +0.2522] **SEP** | -0.0651 [-0.0961, -0.0352] **SEP** | +0.0624 [+0.0500, +0.0747] **SEP** | 1.1339 | OK | OK | **FAIL** | **FAIL** | **FAIL** | OK | no |

### FRONTIER — absolute levels (episode-cluster bootstrap per arm)

| step | CDR@K185 overall | CDR@K185 junction | peak abs XTE (m) | open-loop ADE@2s (m) | anchor_acc | anchor_traj_l1 |
|---|---|---|---|---|---|---|
| **0 (base)** | 0.5877 | 0.8414 | 38.9445 | 0.4747 | 0.6815 | 0.1775 |
| 100 | 0.7979 [0.7405, 0.8485] | 0.8171 [0.7369, 0.8892] | 47.9431 [33.7935, 63.0765] | 0.9289 [0.8588, 0.9972] | 0.4984 [0.4416, 0.5586] | 0.3743 [0.3457, 0.4014] |
| 250 | 0.6292 [0.5391, 0.7164] | 0.7468 [0.6514, 0.8396] | 13.5956 [9.8466, 17.9623] | 1.0491 [0.9232, 1.1661] | 0.4726 [0.4085, 0.5413] | 0.3428 [0.3149, 0.3689] |
| 500 | 0.3459 [0.2459, 0.4461] | 0.6234 [0.3575, 0.8207] | 7.1836 [4.3045, 10.9533] | 0.9794 [0.8574, 1.0941] | 0.5109 [0.4457, 0.5746] | 0.3202 [0.2927, 0.3471] |
| 750 | 0.2944 [0.2104, 0.3791] | 0.6234 [0.3820, 0.7955] | 5.8798 [3.4895, 9.3077] | 0.8002 [0.7015, 0.9002] | 0.5512 [0.4891, 0.6153] | 0.2712 [0.2484, 0.2953] |
| 1000 | 0.1697 [0.1040, 0.2421] | 0.4505 [0.2117, 0.6811] | 4.0464 [2.3902, 6.3932] | 0.8117 [0.6923, 0.9404] | 0.5595 [0.4990, 0.6232] | 0.2646 [0.2431, 0.2871] |
| 1250 | 0.2055 [0.1291, 0.2886] | 0.4784 [0.2179, 0.7369] | 4.5499 [2.9899, 6.4798] | 0.7921 [0.6966, 0.8920] | 0.5812 [0.5222, 0.6412] | 0.2674 [0.2452, 0.2903] |
| 1500 | 0.2186 [0.1478, 0.3003] | 0.4153 [0.1577, 0.6919] | 3.9976 [2.9822, 5.1193] | 0.7735 [0.6832, 0.8680] | 0.5677 [0.5067, 0.6298] | 0.2714 [0.2508, 0.2916] |
| 1750 | 0.1975 [0.1238, 0.2786] | 0.4153 [0.1585, 0.6847] | 3.6209 [2.5159, 4.9699] | 0.7460 [0.6473, 0.8479] | 0.5905 [0.5268, 0.6536] | 0.2660 [0.2420, 0.2896] |
| 2000 | 0.2045 [0.1304, 0.2860] | 0.4279 [0.1766, 0.6937] | 4.0753 [2.7460, 5.8688] | 0.7569 [0.6631, 0.8549] | 0.5884 [0.5274, 0.6494] | 0.2709 [0.2487, 0.2947] |
| 2250 | 0.1476 [0.0797, 0.2272] | 0.4018 [0.1189, 0.6991] | 3.1927 [2.2080, 4.4196] | 0.6944 [0.6042, 0.7893] | 0.6112 [0.5501, 0.6719] | 0.2589 [0.2342, 0.2845] |
| 2500 | 0.1576 [0.0835, 0.2443] | 0.3973 [0.1144, 0.6937] | 3.5738 [2.1283, 5.4905] | 0.6830 [0.5942, 0.7806] | 0.6143 [0.5553, 0.6729] | 0.2418 [0.2198, 0.2642] |
| 2750 | 0.1470 [0.0758, 0.2311] | 0.4000 [0.1198, 0.6955] | 2.9557 [1.9958, 4.1407] | 0.6905 [0.5916, 0.7919] | 0.6112 [0.5522, 0.6701] | 0.2432 [0.2202, 0.2670] |
| 3000 | 0.1604 [0.0886, 0.2456] | 0.3964 [0.1189, 0.6892] | 2.9158 [1.9812, 4.0963] | 0.6880 [0.5890, 0.7952] | 0.6184 [0.5564, 0.6788] | 0.2399 [0.2182, 0.2630] |
| 3250 | 0.1672 [0.0949, 0.2536] | 0.4018 [0.1207, 0.6973] | 3.1600 [2.2448, 4.3188] | 0.6639 [0.5740, 0.7603] | 0.6194 [0.5578, 0.6798] | 0.2398 [0.2176, 0.2633] |
| 3500 | 0.1608 [0.0913, 0.2434] | 0.3432 [0.0937, 0.6225] | 3.1098 [2.1664, 4.2498] | 0.6773 [0.5815, 0.7797] | 0.6132 [0.5512, 0.6739] | 0.2407 [0.2183, 0.2641] |
| 3750 | 0.1574 [0.0875, 0.2386] | 0.3486 [0.1144, 0.6108] | 3.0592 [2.1085, 4.2054] | 0.6715 [0.5779, 0.7721] | 0.6205 [0.5596, 0.6794] | 0.2400 [0.2174, 0.2641] |
| 4000 | 0.1603 [0.0855, 0.2493] | 0.4144 [0.1486, 0.6919] | 3.0415 [2.0496, 4.2845] | 0.6693 [0.5773, 0.7687] | 0.6163 [0.5553, 0.6763] | 0.2399 [0.2174, 0.2640] |

**Machine-emitted verdict object** (`e1c_frontier_result.json → VERDICT`):

```json
{
  "verdict": "BOUND",
  "winner_step": null,
  "n_points": 17,
  "n_primary_ok": 15,
  "n_guardrails_ok": 0,
  "n_success_points": 0,
  "primary_ok_steps": [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500,
                       2750, 3000, 3250, 3500, 3750, 4000],
  "guardrails_ok_steps": []
}
```

---

## 6. READING THE FRONTIER — the finding E1b's two endpoints could not show

E1b measured step 0 and step 4000 and concluded "big closed-loop win, big open-loop
cost". The trajectory says something sharper and less comfortable:

### 6.1 The cost is paid BEFORE the gain arrives

| | step 100 | step 250 | step 500 |
|---|---|---|---|
| closed-loop @K185 overall Δ | **+0.2102 SEP (WORSE)** | +0.0415 (not sep) | **−0.2417 SEP (better)** |
| held-out open-loop ADE@2s Δ | **+0.4542 SEP worse** | **+0.5744 SEP worse** (the maximum) | +0.5048 SEP worse |

At step 100 — **still inside the 100-step lr warm-up** — the model has already
sacrificed **96 %** of the total open-loop ADE it will ever lose (+0.4542 of the
+0.4747 base level), *and its closed-loop corridor-departure is CI-separated
**WORSE** than base* (0.5877 → 0.7979, peak |XTE| 38.94 m → **47.94 m**). The
intervention first **damages** the planner in both regimes; only from step 500 does
the closed-loop objective start paying back.

**This is why early stopping cannot rescue E1b.** The pre-registered stopping
point — the first checkpoint where the corrected held-out guard fires — is
**step 100**, i.e. *the first checkpoint that exists*. A correctly-guarded
early-stopping run would have halted **before any closed-loop benefit had
appeared at all.** There is no early region where the win is free.

### 6.2 The closed-loop gain saturates early; the rest of training only buys back open loop

| step | closed-loop gain (overall Δ) | open-loop cost (ADE@2s Δ) | gain per unit cost |
|---|---|---|---|
| 500 | −0.2417 | +0.5048 | 0.48 |
| 1000 | −0.4180 | +0.3371 | 1.24 |
| 2250 | **−0.4401** | +0.2197 | 2.00 |
| 2500 | −0.4300 | +0.2083 | 2.06 |
| **3250** | −0.4205 | **+0.1893** (minimum) | **2.22** (best) |
| 4000 | −0.4274 | +0.1947 | 2.20 |

By **step 1000** the closed loop already has **98 %** of the gain the endpoint ever
achieves (−0.4180 vs −0.4274). Steps 1000 → 4000 add essentially **no** closed-loop
benefit and instead let the replay branch claw back **0.142 m** of open-loop ADE
(+0.3371 → +0.1947). *The last 3,000 steps of this recipe are an open-loop repair
job, not a closed-loop improvement.* And the repair **asymptotes well short of
parity** — the cost curve flattens at ≈ +0.19 m from step 3250 on, still
CI-separated worse by a wide margin (lo = +0.1383).

### 6.3 The guardrails never come close

Ga fails at **17/17** points; so do Gb1 (anchor_acc) and Gb2 (anchor_traj_l1). The
*narrowest* open-loop violation on the entire frontier is step 3250, whose CI is
`[+0.1383, +0.2413]` — the lower bound alone is **29 % of base ADE**. This is not a
marginal miss that a slightly different threshold would flip.

**Gc (OOD envelope) passes at 17/17**, and it moves the *right* way: OOD peak ratio
falls monotonically from 1.2919 (step 100) to **1.1339** (step 4000) against base
1.2664. So at **no point** on the frontier is the closed-loop movement explained by
the loop leaving the measured perturbation envelope. The win is real; the cost is
also real.

### 6.4 What "BOUND" licenses, precisely

The pre-registration committed this outcome in advance as *equally publishable*:
**there is no point on this trajectory where failure-gated CL-SFT with
`lam_replay = 1.0` buys corridor-keeping for free.** The trade is a property of the
*objective*, not of the *stopping rule*. E1b's broken guard was a real methodological
defect — it just was not the reason E1b was BOUND.

---

## 7. THE IN-TRAINING HELD-OUT GATE — the corrected guard itself

### IN-TRAINING HELD-OUT GATE (the corrected guard, computed during the run)

| step | ADE@2s base→ft | paired Δ | sep? | anchor_acc | anchor_traj_l1 | gate_ok | probe s |
|---|---|---|---|---|---|---|---|
| 0 | 0.4738 → 0.4738 | +0.0000 [+0.0000, +0.0000] | no | 0.6825 → 0.6825 | 0.1775 → 0.1775 | OK | 39.4 |
| 100 | 0.4738 → 0.9289 | +0.4551 [+0.3810, +0.5285] | **SEP** | 0.6825 → 0.4985 | 0.1775 → 0.3743 | **STOP** | 72.7 |
| 250 | 0.4738 → 1.0491 | +0.5754 [+0.4621, +0.6838] | **SEP** | 0.6825 → 0.4726 | 0.1775 → 0.3428 | **STOP** | 70.2 |
| 500 | 0.4738 → 0.9783 | +0.5045 [+0.3987, +0.6101] | **SEP** | 0.6825 → 0.5119 | 0.1775 → 0.3202 | **STOP** | 69.8 |
| 750 | 0.4738 → 0.8003 | +0.3266 [+0.2561, +0.3967] | **SEP** | 0.6825 → 0.5502 | 0.1775 → 0.2712 | **STOP** | 69.8 |
| 1000 | 0.4738 → 0.8117 | +0.3380 [+0.2481, +0.4401] | **SEP** | 0.6825 → 0.5595 | 0.1775 → 0.2646 | **STOP** | 71.2 |
| 1250 | 0.4738 → 0.7906 | +0.3169 [+0.2525, +0.3783] | **SEP** | 0.6825 → 0.5822 | 0.1775 → 0.2674 | **STOP** | 66.9 |
| 1500 | 0.4738 → 0.7735 | +0.2997 [+0.2451, +0.3560] | **SEP** | 0.6825 → 0.5677 | 0.1775 → 0.2714 | **STOP** | 67.2 |
| 1750 | 0.4738 → 0.7480 | +0.2742 [+0.2104, +0.3408] | **SEP** | 0.6825 → 0.5905 | 0.1775 → 0.2660 | **STOP** | 66.4 |
| 2000 | 0.4738 → 0.7564 | +0.2827 [+0.2146, +0.3572] | **SEP** | 0.6825 → 0.5895 | 0.1775 → 0.2709 | **STOP** | 71.2 |
| 2250 | 0.4738 → 0.6941 | +0.2203 [+0.1690, +0.2720] | **SEP** | 0.6825 → 0.6122 | 0.1775 → 0.2589 | **STOP** | 63.8 |
| 2500 | 0.4738 → 0.6832 | +0.2094 [+0.1569, +0.2668] | **SEP** | 0.6825 → 0.6132 | 0.1775 → 0.2418 | **STOP** | 64.4 |
| 2750 | 0.4738 → 0.6897 | +0.2159 [+0.1523, +0.2831] | **SEP** | 0.6825 → 0.6122 | 0.1775 → 0.2432 | **STOP** | 67.8 |
| 3000 | 0.4738 → 0.6880 | +0.2142 [+0.1526, +0.2794] | **SEP** | 0.6825 → 0.6184 | 0.1775 → 0.2399 | **STOP** | 66.0 |
| 3250 | 0.4738 → 0.6639 | +0.1902 [+0.1392, +0.2426] | **SEP** | 0.6825 → 0.6194 | 0.1775 → 0.2398 | **STOP** | 65.0 |
| 3500 | 0.4738 → 0.6772 | +0.2035 [+0.1456, +0.2650] | **SEP** | 0.6825 → 0.6132 | 0.1775 → 0.2407 | **STOP** | 69.2 |
| 3750 | 0.4738 → 0.6716 | +0.1978 [+0.1428, +0.2594] | **SEP** | 0.6825 → 0.6205 | 0.1775 → 0.2400 | **STOP** | 68.6 |
| 4000 | 0.4738 → 0.6694 | +0.1956 [+0.1427, +0.2536] | **SEP** | 0.6825 → 0.6163 | 0.1775 → 0.2399 | **STOP** | 68.9 |

**Read.** The step-0 row is exactly `+0.0000 [+0.0000, +0.0000]` — a built-in
identity check that the probe pairs base against itself on the same windows. From
**step 100 onward the guard fails at every single checkpoint**, so
`heldout_gate_stopping_point = 100`.

**Contrast this with E1b's instrument.** E1b's replay loss on parity-train **fell**
1.826 → 1.613 over the same 4,000 steps, and its `rp_anchor_acc` **rose** 0.550 →
0.584 — the training-set instrument reported the forgetting guard *working*, while
this held-out instrument, on the identical run, reports the model **2.2× worse** on
held-out open-loop ADE at step 250 and still **1.41× worse** at step 4000. Same
weights, same steps, opposite conclusion. *(Root-cause class for
`RETRACTION_LOG.md`: **training-set instrument used as a generalisation guard** —
sibling of "trainer val is not eval output".)*

---

## 8. ENDPOINT DETAIL (step 4000) — strata, guardrails, and the 2 s instrument

### ENDPOINT (step 4000) — closed loop @ K=185, all strata

| stratum | metric | base | step 4000 | paired Δ |
|---|---|---|---|---|
| overall | dep | 0.5877 [0.5107, 0.6622] | 0.1603 [0.0855, 0.2493] | -0.4274 [-0.5161, -0.3378] **SEP** |
| overall | win_dep | 0.9302 [0.8605, 1.0000] | 0.4419 [0.3023, 0.5814] | -0.4884 [-0.6512, -0.3023] **SEP** |
| overall | peak_xte | 38.9445 [27.0163, 52.6962] | 3.0415 [2.0496, 4.2845] | -35.9030 [-49.3302, -24.1241] **SEP** |
| overall | mean_xte | 14.3062 [9.8386, 19.2428] | 1.3908 [0.8508, 2.0630] | -12.9153 [-17.6887, -8.5585] **SEP** |
| overall | peak_dpsi | 25.0635 [20.4211, 29.9424] | 13.4647 [9.5194, 17.9761] | -11.5988 [-15.1709, -8.2172] **SEP** |
| overall | ood_peak | 1.2664 [1.2422, 1.2880] | 1.1339 [1.1060, 1.1625] | -0.1325 [-0.1640, -0.0981] **SEP** |
| overall | ood_mean | 1.1583 [1.1365, 1.1796] | 1.0559 [1.0356, 1.0803] | -0.1024 [-0.1261, -0.0793] **SEP** |
| overall | out_env | 0.9070 [0.8140, 0.9767] | 0.2558 [0.1395, 0.3953] | -0.6512 [-0.7907, -0.4884] **SEP** |
| overall | ade2s | 0.4957 [0.3749, 0.6434] | 0.6363 [0.4988, 0.8020] | +0.1406 [+0.0423, +0.2738] **SEP** |
| junction | dep | 0.8414 [0.8144, 0.8667] | 0.4144 [0.1486, 0.6919] | -0.4270 [-0.6838, -0.1648] **SEP** |
| junction | win_dep | 1.0000 [1.0000, 1.0000] | 0.8333 [0.5000, 1.0000] | -0.1667 [-0.5000, +0.0000] |
| junction | peak_xte | 46.2475 [24.4878, 68.7290] | 7.0027 [2.6065, 12.0766] | -39.2447 [-63.6992, -18.5090] **SEP** |
| junction | mean_xte | 21.4716 [12.1178, 31.8233] | 3.4698 [1.0387, 6.3100] | -18.0018 [-29.2782, -9.0686] **SEP** |
| junction | peak_dpsi | 44.7871 [37.9884, 51.7735] | 27.1022 [11.6881, 43.7914] | -17.6848 [-28.3425, -7.1961] **SEP** |
| junction | ood_peak | 1.2989 [1.2989, 1.2989] | 1.2018 [1.1198, 1.2768] | -0.0971 [-0.1791, -0.0221] **SEP** |
| junction | ood_mean | 1.2442 [1.2282, 1.2572] | 1.1305 [1.0529, 1.2135] | -0.1137 [-0.1854, -0.0403] **SEP** |
| junction | out_env | 1.0000 [1.0000, 1.0000] | 0.5000 [0.1667, 0.8333] | -0.5000 [-0.8333, -0.1667] **SEP** |
| junction | ade2s | 0.7318 [0.5102, 1.0078] | 0.5961 [0.4789, 0.7149] | -0.1357 [-0.4385, +0.0620] |
| longitudinal | dep | 0.6654 [0.5613, 0.7491] | 0.0990 [0.0137, 0.2174] | -0.5664 [-0.6845, -0.4267] **SEP** |
| longitudinal | win_dep | 1.0000 [1.0000, 1.0000] | 0.3684 [0.1579, 0.5789] | -0.6316 [-0.8421, -0.4211] **SEP** |
| longitudinal | peak_xte | 56.3890 [34.8915, 79.2986] | 2.6024 [1.3844, 4.3745] | -53.7866 [-76.5947, -32.8202] **SEP** |
| longitudinal | mean_xte | 19.6665 [12.0722, 28.1526] | 1.1765 [0.5117, 2.1538] | -18.4900 [-26.7812, -11.1481] **SEP** |
| longitudinal | peak_dpsi | 21.0409 [14.9175, 27.1862] | 7.9166 [4.7181, 12.2345] | -13.1242 [-18.6587, -7.4340] **SEP** |
| longitudinal | ood_peak | 1.2712 [1.2402, 1.2927] | 1.1065 [1.0721, 1.1449] | -0.1647 [-0.2026, -0.1237] **SEP** |
| longitudinal | ood_mean | 1.1673 [1.1359, 1.1948] | 1.0310 [1.0111, 1.0597] | -0.1363 [-0.1671, -0.1028] **SEP** |
| longitudinal | out_env | 0.9474 [0.8421, 1.0000] | 0.1579 [0.0000, 0.3158] | -0.7895 [-0.9474, -0.5789] **SEP** |
| longitudinal | ade2s | 0.5165 [0.2917, 0.8080] | 0.7507 [0.4771, 1.0836] | +0.2343 [+0.0673, +0.4949] **SEP** |

**Open-loop guardrails at step 4000**

| metric | base | step 4000 | paired Δ |
|---|---|---|---|
| ade2s | 0.4747 [0.4029, 0.5528] | 0.6693 [0.5773, 0.7687] | +0.1947 [+0.1415, +0.2522] **SEP** |
| anchor_acc | 0.6815 [0.6267, 0.7381] | 0.6163 [0.5553, 0.6763] | -0.0651 [-0.0961, -0.0352] **SEP** |
| anchor_ce | 0.8757 [0.7367, 1.0261] | 1.1637 [0.9858, 1.3563] | +0.2880 [+0.2011, +0.3789] **SEP** |
| anchor_traj_l1 | 0.1775 [0.1594, 0.1975] | 0.2399 [0.2174, 0.2640] | +0.0624 [+0.0500, +0.0747] **SEP** |

**K=20 (2 s) — reported, NON-DECIDING, at step 4000**

| stratum | metric | base | step 4000 | paired Δ |
|---|---|---|---|---|
| overall | dep | 0.0053 [0.0018, 0.0096] | 0.0068 [0.0023, 0.0127] | +0.0015 [-0.0019, +0.0066] |
| overall | win_dep | 0.0352 [0.0124, 0.0663] | 0.0372 [0.0166, 0.0620] | +0.0021 [-0.0165, +0.0238] |
| overall | peak_xte | 0.3683 [0.2867, 0.4591] | 0.5176 [0.4481, 0.5965] | +0.1493 [+0.0885, +0.2112] **SEP** |
| overall | ade2s | 0.5227 [0.4456, 0.6076] | 0.6238 [0.5505, 0.6991] | +0.1012 [+0.0619, +0.1451] **SEP** |
| junction | dep | 0.0395 [0.0178, 0.0676] | 0.0359 [0.0152, 0.0609] | -0.0036 [-0.0194, +0.0083] |
| junction | win_dep | 0.2661 [0.1182, 0.4386] | 0.2016 [0.1065, 0.3077] | -0.0645 [-0.1532, +0.0081] |
| junction | peak_xte | 1.2287 [1.0061, 1.4792] | 1.0738 [0.8478, 1.3466] | -0.1549 [-0.3006, -0.0302] **SEP** |
| junction | ade2s | 0.9454 [0.7499, 1.2417] | 1.0093 [0.7709, 1.3891] | +0.0639 [+0.0052, +0.1504] **SEP** |
| longitudinal | dep | 0.0000 [0.0000, 0.0000] | 0.0047 [0.0000, 0.0129] | +0.0047 [+0.0000, +0.0129] |
| longitudinal | win_dep | 0.0000 [0.0000, 0.0000] | 0.0241 [0.0000, 0.0581] | +0.0241 [+0.0000, +0.0581] |
| longitudinal | peak_xte | 0.3471 [0.2640, 0.4382] | 0.5564 [0.4653, 0.6540] | +0.2093 [+0.1118, +0.3092] **SEP** |
| longitudinal | ade2s | 0.4330 [0.3418, 0.5440] | 0.6056 [0.5139, 0.7008] | +0.1726 [+0.1171, +0.2312] **SEP** |

**Read.** Identical to E1b's tables at every figure. The long-horizon
corridor effect is separated in **every** stratum and is *not* junction-only —
the longitudinal stratum improves most (0.6654 -> 0.0990). The honest nuance is
unchanged: at junctions the *window* departure rate is 1.0000 -> 0.8333, **not**
separated, so the FT still leaves the corridor at some point in 5 of 6 junction
windows; what collapsed is **how long** and **how far**, not **whether**.

**The 2 s instrument, again, would have rejected this fine-tune** (departure
unchanged, peak |XTE| and closed ADE@2s both separated worse) while K=185 shows a
3.7x corridor-departure improvement. E1a's lesson holds in both directions.

---

## 9. M1 lateral / longitudinal decomposition — across the whole frontier

No ADE is reported without its (lat, lon) split (`taniteval/lateral.py`). GT identity
check `max|Δ| = 0.0` at every point (both arms really are on the same windows and the
same ground truth); axis convention **verified** on every block.

### 9.1 The cross-track axis, per checkpoint (ego frame; cross-track is the safety axis)

| step | OL cross_abs@2s (m) | paired d | OL along_abs@2s (m) | paired d | lat share of sq-err | CL cross_abs@2s (m) | CL lat share |
|---|---|---|---|---|---|---|---|
| **0 (base)** | 0.2494 | - | 0.8974 | - | 0.0888 | 0.3763 | 0.1659 |
| 100 | 1.0012 | +0.7518 [+0.6487, +0.8608] SEP | 1.2859 | +0.3885 [+0.2802, +0.5023] SEP | 0.3419 | 2.4691 | 0.7866 |
| 250 | 0.7531 | +0.5037 [+0.3967, +0.6147] SEP | 1.2673 | +0.3699 [+0.2335, +0.5180] SEP | 0.3757 | 1.7466 | 0.7511 |
| 500 | 0.6813 | +0.4319 [+0.3148, +0.5532] SEP | 1.2582 | +0.3608 [+0.2492, +0.4922] SEP | 0.3431 | 0.9559 | 0.6419 |
| 750 | 0.4969 | +0.2475 [+0.1737, +0.3227] SEP | 1.1478 | +0.2504 [+0.1709, +0.3413] SEP | 0.2615 | 0.6157 | 0.4146 |
| 1000 | 0.4970 | +0.2476 [+0.1720, +0.3249] SEP | 1.1953 | +0.2979 [+0.1773, +0.4559] SEP | 0.2254 | 0.5722 | 0.2989 |
| 1250 | 0.5047 | +0.2552 [+0.1902, +0.3192] SEP | 1.1516 | +0.2542 [+0.1725, +0.3558] SEP | 0.2611 | 0.7472 | 0.4288 |
| 1500 | 0.4314 | +0.1819 [+0.1360, +0.2304] SEP | 1.2357 | +0.3383 [+0.2434, +0.4380] SEP | 0.1943 | 0.6007 | 0.3750 |
| 1750 | 0.4462 | +0.1968 [+0.1370, +0.2606] SEP | 1.1681 | +0.2707 [+0.1843, +0.3669] SEP | 0.2200 | 0.6198 | 0.3440 |
| 2000 | 0.4601 | +0.2107 [+0.1422, +0.2815] SEP | 1.1697 | +0.2723 [+0.1856, +0.3646] SEP | 0.2217 | 0.6783 | 0.3450 |
| 2250 | 0.4444 | +0.1950 [+0.1452, +0.2482] SEP | 1.0718 | +0.1744 [+0.1074, +0.2533] SEP | 0.2404 | 0.5760 | 0.4060 |
| 2500 | 0.4284 | +0.1790 [+0.1216, +0.2368] SEP | 1.0832 | +0.1858 [+0.1109, +0.2746] SEP | 0.2147 | 0.5579 | 0.3443 |
| 2750 | 0.4232 | +0.1738 [+0.1213, +0.2269] SEP | 1.1029 | +0.2055 [+0.1176, +0.3054] SEP | 0.1988 | 0.5462 | 0.3791 |
| 3000 | 0.4278 | +0.1784 [+0.1177, +0.2399] SEP | 1.0978 | +0.2005 [+0.1173, +0.2970] SEP | 0.2141 | 0.6025 | 0.3473 |
| 3250 | 0.4071 | +0.1577 [+0.1081, +0.2069] SEP | 1.0727 | +0.1753 [+0.1049, +0.2563] SEP | 0.2078 | 0.6237 | 0.3621 |
| 3500 | 0.4150 | +0.1656 [+0.1114, +0.2199] SEP | 1.0930 | +0.1956 [+0.1152, +0.2870] SEP | 0.2101 | 0.5943 | 0.3621 |
| 3750 | 0.4156 | +0.1662 [+0.1146, +0.2179] SEP | 1.0809 | +0.1835 [+0.1071, +0.2737] SEP | 0.2110 | 0.5908 | 0.3606 |
| 4000 | 0.4105 | +0.1611 [+0.1126, +0.2097] SEP | 1.0812 | +0.1838 [+0.1090, +0.2698] SEP | 0.2053 | 0.5917 | 0.3642 |

**Read — the mechanism, now visible over time.** Base open-loop error is
overwhelmingly longitudinal (lateral share of squared error **0.0888**). At
**step 100** the CL-SFT has driven the lateral share to **0.3419** and cross-track
ADE@2s from 0.2494 m to **1.0012 m** — a **4.0×** blow-up of exactly the axis the
intervention was meant to *fix*, and simultaneously the closed-loop cross-track
error explodes to **2.4691 m** (lateral share 0.7866). The score head has been pushed
toward "return to corridor" long before it has learned *where* the corridor is, and
the immediate effect is lateral chaos in both regimes.

Training then repairs this monotonically: open-loop cross-track ADE falls
1.0012 → 0.4105 m and the closed-loop cross-track falls 2.4691 → 0.5917 m, while the
long-horizon corridor-departure improves. But open-loop cross-track **never returns
to base** — it asymptotes at ≈ 0.41 m, **+65 %** over base, with the lateral share of
squared error stuck at **0.205** (2.3× base).

**The trade is lateral-for-lateral at two horizons.** The FT buys long-horizon
lateral *containment* (peak |XTE| 38.94 m → 3.04 m) by paying in short-horizon
lateral *precision* (cross-track ADE@2s +0.16 m, p90 +0.39 m). An undecomposed L2
would have reported this as a generic "+0.19 m ADE regression" and hidden which axis
moved.

### 9.2 Full decomposition at the endpoint (both frames)

### M1 lateral/longitudinal — ENDPOINT step 4000, open loop, held-out 44

GT identity check max|Δ| = 0.0 · axis convention {'mean_abs_along_final_m': 30.1962, 'mean_abs_cross_final_m': 0.9344, 'K': 4, 'horizon_s': 2.0, 'verified': True} · n_windows 967

| frame | metric | base | step 4000 | paired Δ |
|---|---|---|---|---|
| ego | ade_over_knots | 0.4747 [0.4029, 0.5528] | 0.6693 [0.5773, 0.7687] | +0.1947 [+0.1415, +0.2522] **SEP** |
| ego | cross_abs@2s | 0.2494 [0.2093, 0.2899] | 0.4105 [0.3349, 0.4840] | +0.1611 [+0.1126, +0.2097] **SEP** |
| ego | cross_p90@2s | 0.6025 [0.4899, 0.6818] | 0.9970 [0.7423, 1.2033] | +0.3945 [+0.1876, +0.5577] **SEP** |
| ego | along_abs@2s | 0.8974 [0.7476, 1.0628] | 1.0812 [0.8978, 1.2895] | +0.1838 [+0.1090, +0.2698] **SEP** |
| ego | energy share (lon/lat) | {'longitudinal_share_of_squared_error': 0.9112, 'lateral_share_of_squared_error': 0.0888, 'longitudinal_share_by_step': [0.9584, 0.9446, 0.9258, 0.9038], '_read': 'MEASURED reference 0.986 longitudinal on the IDM reconstruction: the lateral axis receives ~1.4 % of the squared-error signal, so an undecomposed L2 objective is numerically a longitudinal objective'} | {'longitudinal_share_of_squared_error': 0.7947, 'lateral_share_of_squared_error': 0.2053, 'longitudinal_share_by_step': [0.4336, 0.7123, 0.7954, 0.8197], '_read': 'MEASURED reference 0.986 longitudinal on the IDM reconstruction: the lateral axis receives ~1.4 % of the squared-error signal, so an undecomposed L2 objective is numerically a longitudinal objective'} | — |
| frenet | ade_over_knots | 0.4747 [0.4029, 0.5528] | 0.6693 [0.5773, 0.7687] | +0.1947 [+0.1415, +0.2522] **SEP** |
| frenet | cross_abs@2s | 0.2343 [0.1981, 0.2690] | 0.3900 [0.3210, 0.4591] | +0.1557 [+0.1089, +0.2051] **SEP** |
| frenet | cross_p90@2s | 0.5777 [0.4714, 0.6665] | 0.9248 [0.7346, 1.0727] | +0.3471 [+0.2147, +0.4742] **SEP** |
| frenet | along_abs@2s | 0.9039 [0.7501, 1.0736] | 1.0900 [0.9021, 1.3009] | +0.1861 [+0.1110, +0.2718] **SEP** |
| frenet | energy share (lon/lat) | {'longitudinal_share_of_squared_error': 0.9353, 'lateral_share_of_squared_error': 0.0647, 'longitudinal_share_by_step': [0.9607, 0.9544, 0.9432, 0.9312], '_read': 'MEASURED reference 0.986 longitudinal on the IDM reconstruction: the lateral axis receives ~1.4 % of the squared-error signal, so an undecomposed L2 objective is numerically a longitudinal objective'} | {'longitudinal_share_of_squared_error': 0.812, 'lateral_share_of_squared_error': 0.188, 'longitudinal_share_by_step': [0.4304, 0.714, 0.8066, 0.8424], '_read': 'MEASURED reference 0.986 longitudinal on the IDM reconstruction: the lateral axis receives ~1.4 % of the squared-error signal, so an undecomposed L2 objective is numerically a longitudinal objective'} | — |

### M1 lateral/longitudinal — ENDPOINT step 4000, closed loop, 2 s knots inside K=185

GT identity check max|Δ| = 0.0 · axis convention {'mean_abs_along_final_m': 30.1829, 'mean_abs_cross_final_m': 0.9495, 'K': 4, 'horizon_s': 2.0, 'verified': True} · n_windows 43

| frame | metric | base | step 4000 | paired Δ |
|---|---|---|---|---|
| ego | ade_over_knots | 0.4957 [0.3749, 0.6434] | 0.6363 [0.4988, 0.8020] | +0.1406 [+0.0423, +0.2738] **SEP** |
| ego | cross_abs@2s | 0.3763 [0.2482, 0.5316] | 0.5917 [0.4083, 0.8485] | +0.2154 [+0.0038, +0.4815] **SEP** |
| ego | cross_p90@2s | 0.8994 [0.6201, 1.4452] | 1.0955 [0.7189, 1.8257] | +0.1961 [-0.3202, +0.9815] |
| ego | along_abs@2s | 0.8985 [0.6382, 1.2048] | 0.9628 [0.7108, 1.2620] | +0.0643 [-0.0881, +0.1992] |
| ego | energy share (lon/lat) | {'longitudinal_share_of_squared_error': 0.8341, 'lateral_share_of_squared_error': 0.1659, 'longitudinal_share_by_step': [0.8944, 0.8608, 0.8448, 0.829], '_read': 'MEASURED reference 0.986 longitudinal on the IDM reconstruction: the lateral axis receives ~1.4 % of the squared-error signal, so an undecomposed L2 objective is numerically a longitudinal objective'} | {'longitudinal_share_of_squared_error': 0.6358, 'lateral_share_of_squared_error': 0.3642, 'longitudinal_share_by_step': [0.4933, 0.514, 0.5913, 0.6637], '_read': 'MEASURED reference 0.986 longitudinal on the IDM reconstruction: the lateral axis receives ~1.4 % of the squared-error signal, so an undecomposed L2 objective is numerically a longitudinal objective'} | — |
| frenet | ade_over_knots | 0.4957 [0.3749, 0.6434] | 0.6363 [0.4988, 0.8020] | +0.1406 [+0.0423, +0.2738] **SEP** |
| frenet | cross_abs@2s | 0.3889 [0.2617, 0.5461] | 0.5816 [0.4024, 0.8323] | +0.1927 [-0.0240, +0.4553] |
| frenet | cross_p90@2s | 0.8848 [0.6167, 1.1490] | 1.0541 [0.7360, 1.7279] | +0.1693 [-0.2136, +0.9267] |
| frenet | along_abs@2s | 0.8962 [0.6408, 1.2076] | 0.9816 [0.7313, 1.2759] | +0.0853 [-0.0386, +0.2070] |
| frenet | energy share (lon/lat) | {'longitudinal_share_of_squared_error': 0.8238, 'lateral_share_of_squared_error': 0.1762, 'longitudinal_share_by_step': [0.8959, 0.8599, 0.8384, 0.8169], '_read': 'MEASURED reference 0.986 longitudinal on the IDM reconstruction: the lateral axis receives ~1.4 % of the squared-error signal, so an undecomposed L2 objective is numerically a longitudinal objective'} | {'longitudinal_share_of_squared_error': 0.6424, 'lateral_share_of_squared_error': 0.3576, 'longitudinal_share_by_step': [0.4949, 0.5172, 0.5956, 0.6715], '_read': 'MEASURED reference 0.986 longitudinal on the IDM reconstruction: the lateral axis receives ~1.4 % of the squared-error signal, so an undecomposed L2 objective is numerically a longitudinal objective'} | — |

---

## 10. Housekeeping — the E1b checkpoint is no longer single-disk

E1b's report escalated this as an integration risk: the **527 MB FT checkpoint
existed on pod3 only**, and a volume event would have cost 3.5 h of A40 time to
rebuild. Done, `MEASURED`:

| item | value |
|---|---|
| repo | **https://huggingface.co/Sayood/tanitad-refc-base-e1b-clsft** |
| visibility | **public + `gated="manual"`** — every download is manually approved |
| `ckpt.pt` | **527,027,443 B**, md5 **`6e25dd670715cb84dc68a6f080708d07`** (computed on pod3 at upload time) |
| also uploaded | `README.md` (model card, states the BOUND verdict), `config.json`, `metrics.json`, `train_log.jsonl` |
| throughput | **103.4 MB/s** from pod3 (the dev-box relay is ~1 MB/s) |

**The gate was set before publication, not after.** The script
(`scripts/hf_push_e1b_ckpt.py`) reproduces the program's mandated order —
create **private** → set `gated="manual"` **while still private** → flip
`private=False` → **verify via the API that the repo is public AND gated**
(hard abort otherwise) → only then upload weights. Readback confirmed
`private=False gated=manual` before a single byte of weights was sent, so the
weights were **never world-downloadable**. This is the same sequencing the PI
approved as "Option A" on 2026-07-25 for `tanitad-refc-xl` / `tanitad-refc-base`
after the free-tier private-storage 403; private weight storage on this account
is exhausted, so a private repo was not an option.

The token was read from `Keys.txt` **in place**, piped over stdin, and is
**never printed, never written to disk, never in argv**. (A token file briefly
written to `~/.cache/huggingface/token` during setup was removed before the push
and the stdin path used instead.)

---

## 11. Honest bounds

- The closed loop is **map/agent-free**. This measures drift / corridor-keeping,
  **not** collision or off-road safety. Nothing here is a certified-safety claim.
- The **junction stratum is 6 windows in 6 episodes** at K=185. An
  episode-cluster bootstrap over 6 clusters is low-powered by construction —
  which is exactly why the pre-registration required P1 (overall, 43 clusters)
  **alongside** P2, so no verdict rests on 6 clusters alone.
- **Multiplicity — declared in advance and, as it turns out, moot.** 18 frontier
  points were scanned; the pre-registration (§4.4) committed a Bonferroni-adjusted
  flag (`p_delta_gt0 < 0.05/18 = 0.002778` on both P1 and P2), reported per
  checkpoint in the raw JSON. **Multiplicity inflates the false-SUCCESS rate, and
  the verdict is BOUND — so the correction can only make the conclusion stronger,
  never weaker.** The symmetric worry does apply and is the one to keep: a
  guardrail that "includes 0" could be a **power** artifact. It never arose here —
  the guardrails did not merely fail to pass, they were CI-separated *worse* at
  **17/17** points, the narrowest violation having lower bound **+0.1383 m**
  (29 % of base ADE). The ADE Δ point estimate, its CI and a `noninferior_0p05`
  descriptor are reported per checkpoint; **descriptors do not move the verdict.**
- **K=185 is the structural ceiling** on this 190–199-frame corpus (E1a §1.4);
  20 s is not reachable here.
- The P1 OOD envelope was MEASURED only to |dlat| ≤ 3.0 m / |dyaw| ≤ 12°;
  `np.interp` clamps beyond it, so a reported OOD ratio is a **lower bound** at
  long horizons.
- The recovery target is a **kinematic demonstration** (logged corridor in the
  offset ego frame), not a renderer rollout.
- The run is **seeded but not bitwise-deterministic** (cuDNN, 4 dataloader
  workers, and the diffusion decode consumes RNG). The reproduction check in §4
  is reported as measured, without smoothing.
- **Not answered here, by construction:** a different `lam_replay`, replay drawn
  from a held-out-like distribution, LoRA-style constraint, or an AlpaSim/CARLA
  loop with agents and a map. λ is the next lever, not this one.
- **No parity change.** Mining and replay stayed on `physicalai-train-e438721ae894`;
  evaluation stayed on the byte-disjoint held-out 44. Nothing re-selected episodes.

---

## 12. Artifact manifest

| artifact | where |
|---|---|
| **pre-registration (staged before the run)** | `…/2026-07-26-e1c-heldout-gated-clsft/PRE_REGISTRATION_E1C.md` |
| **raw frontier eval JSON (the quotable source)** | `…/e1c_frontier_result.json` · pod `/workspace/e1c/e1c_frontier_result.json` |
| **in-training held-out gate (the corrected guard)** | `…/ft_run/heldout_gate.jsonl` · pod `/workspace/e1c/refc-base-e1c-clsft/heldout_gate.jsonl` |
| **evaluator self-test result (43/43)** | `…/e1c_selftest_result.json` |
| evaluator end-to-end smoke (real GPU rollouts, pre-run) | `…/e1c_evaluator_smoke_result.json` |
| training log / metrics / config | `…/ft_run/train_log.jsonl`, `…/ft_run/metrics.json`, `…/ft_run/config.json` |
| run log (mine skipped, CL-SFT + probes) | `…/ft_run/e1c_run.log` |
| frontier eval log | `…/ft_run/e1c_eval.log` |
| base held-out open-loop arrays (cached at step 0) | pod `/workspace/e1c/refc-base-e1c-clsft/base_heldout_openloop.npz` |
| shared verdict/statistics module | `…/scripts/e1c_common.py` · pod `/workspace/e1c/e1c_common.py` |
| instrumented trainer | `…/scripts/e1c_clsft.py` · pod `/workspace/e1c/e1c_clsft.py` |
| frontier evaluator | `…/scripts/e1c_eval.py` · pod `/workspace/e1c/e1c_eval.py` |
| evaluator self-test | `…/scripts/e1c_selftest.py` · pod `/workspace/e1c/e1c_selftest.py` |
| table generator (every number in §5–§9) | `…/scripts/summarize_e1c.py` |
| launchers | `…/scripts/run_e1c.sh`, `…/scripts/run_e1c_eval.sh` |
| HF backup script | `…/scripts/hf_push_e1b_ckpt.py` |
| **E1b FT ckpt (was single-disk)** | **HF `Sayood/tanitad-refc-base-e1b-clsft`** (public+gated-manual) · pod `/workspace/e1b/refc-base-e1b-clsft/ckpt.pt` |
| **E1c frontier checkpoints (17 × 55 MB, POD ONLY)** | `tanitad-pod3:/workspace/e1c/refc-base-e1c-clsft/delta_step*.pt` |
| E1c full final state dict (pod only) | `tanitad-pod3:/workspace/e1c/refc-base-e1c-clsft/full_state_final.pt` |
| mined buffer (reused, already banked by E1b) | `…/2026-07-25-e1b-failure-gated-clsft/ft_run/mined_buffer.pt` |

**Estimator provenance (`MEASURED` this session).**
`/workspace/TanitAD/taniteval/taniteval/ci.py` md5 **`ef925f06febd20a99f5901491fcf75cb`**
— byte-identical to the repo's `taniteval/taniteval/ci.py` — is what the trainer
binds; `/workspace/e1a_e2a/taniteval_ci.py` carries the **same md5** and is what
the evaluator binds. `taniteval_lateral.py` md5 `897938ae40b6cb2dfa51802f0ec260b9`.
`overlapping_holdout_se` appears nowhere in the E1c pipeline.

⚠️ **A trap re-hit and worth recording.** `e1a_horizon` **prepends
`/root/taniteval` — a stale checkout whose package has no `ci` module — to
`sys.path` at import time.** Importing it before `taniteval` makes the estimator
import fail (`ImportError: cannot import name 'ci'`). E1b documented this for the
evaluator; it bit the *trainer* too. Both now bind `taniteval` **first** and
**assert the resolved path**.

**Escalation for the orchestrator (integration, not a README request).** The
**17 frontier deltas (935 MB total) live on pod3 only.** They are the frontier —
the scientific object of this experiment — and re-deriving them costs the full
CL-SFT run. If any follow-up (λ sweep, LoRA constraint, AlpaSim replay) is
authorised, they should be pushed to HF from pod3 (~103 MB/s measured) before a
pod3 volume event, exactly as the E1b checkpoint just was. They were **not**
pushed automatically because that is 935 MB of derived weights on a free-tier
account whose private storage is already exhausted, and the decision on public
gated storage for 17 intermediate checkpoints belongs to the PI, not to me.

Repo files are written into the working tree and **not** `git add`ed, committed
or pushed (Agent Operating Standard: stage, never push — the orchestrator
commits).

---

## 13. What this licenses next (proposals, not conclusions)

`HYPOTHESIS` in every line below. **None of these is a re-run of E1c with a moved
goalpost** — E1c is BOUND and stays BOUND.

1. **λ_replay is now the licensed next lever, and E1c says what to expect of it.**
   The pre-registration deliberately froze `lam_replay = 1.0` so the guard would be
   the only changed variable. The frontier now shows *why* λ is the right next knob
   and not early-stopping: the cost is incurred in the first 100 steps, so a
   *scheduler* over the trade (λ, or lr on the CL branch) is the only thing that can
   act in that window. **Both outcomes must be pre-registered again**, including the
   possibility that the whole trade-off curve is dominated.
2. **The failure is localised to one head, and E1c localises it in TIME as well.**
   Everything breaks inside the warm-up. A constraint that acts *there* — LoRA-style
   rank limit on the score head, KL/trust-region to the base policy, or freezing
   `anchor_traj` and moving only `anchor_logits` — is a cheaper discriminating
   experiment than another 4,000-step SFT.
3. **The replay branch is the wrong distribution, not just the wrong instrument.**
   Replay is drawn from parity-train while the guard is held-out; the replay loss
   fell while held-out rose. A replay branch sampled to match the *held-out* window
   distribution (or simply a held-out-gated replay weight) is testable in one run.
4. **A closed-loop objective that does not re-purpose the open-loop head.** Every
   guardrail that failed is a component of the anchor block the CL-SFT overwrote. An
   auxiliary head (separate score head for recovery, mixed at inference) would make
   the two objectives structurally non-competing rather than trading in one set of
   weights.
5. **The standing protocol should absorb two rules.** (a) *No forgetting guard may be
   computed on the corpus it replays* — E1c is the counter-example with the
   weights held fixed. (b) *No closed-loop verdict at K=20 alone* (E1a's lesson,
   re-confirmed: the 2 s instrument again calls this fine-tune neutral-to-mildly-worse
   while K=185 shows a 3.7× corridor-departure improvement).

---

## 14. Retraction-log entry proposed (root-cause class, per CLAUDE.md rule 4)

**Class: `training-set instrument used as a generalisation guard`.**
E1b reported its forgetting guard as healthy (replay loss 1.826 → 1.613 on
parity-train). E1c re-ran the **identical weights and steps** with the guard on
held-out data and measured the opposite: ADE@2s +0.4551 at step 100, still +0.1956 at
step 4000, CI-separated worse at **every** checkpoint. **The correction is not that
E1b's verdict was wrong — it was BOUND and correctly labelled — but that its
*mechanism claim* ("by its own instrument the guard was working") was unfalsifiable
by construction.** Sibling class: "trainer val is not eval output" (2026-07-21).
Generalised rule: *an instrument evaluated on data the optimiser is fitting cannot
bound generalisation, no matter how the loss moves.*

