# PREREG_V8F_TAC — does v8F-mini's operative layer use its tactical layer? (2026-10-20, before any run)

**Status: PRE-REGISTERED, NOT RUN.**
* Eval only, zero training, about 40 min on box 2 under SAFE MODE and the temperature guard.
* Box 2 runs it only on the PI's go.
* Register row: `H-V8F-TAC-1`.

## Why it exists

The PI asked, verbatim (2026-10-20):

> *"Did you implement the tactical and operative hierarchacy as included in h-jepan end2end planner and in alignment to my 4b conceot using the tactical goals and constraints in our training corpus"*

The audit (`…/2026-10-17-v8f-mini/raw/2026-10-20-tac/README.md`) was MEASURED from source and banked logs at tip `142e169d`.

**What IS built: v8F-mini, S → T → O.** Each layer runs at its own tick and is read through sample-and-hold (`stack/tanitad/models/v8f_mini.py`).

* **T** emits four heads:
  * the v9 tactical **goals**: 22 classes, including TRAFFIC_LIGHT_REACT_RED / _YELLOW / _GREEN;
  * **lateral** decisions (5) and **longitudinal** decisions (7);
  * **26 constraints**: turn geometry, lane-change crossing, target speed, stop point, lead gap and time gap, speed band.
* **The labels come from the training corpus.** The v9 release covers 4,572 / 4,572 train clips (100 %) and 139 / 141 eval clips (`LABEL_JOIN_REPORT.md`).
* **The labels reach training.** In M2b_s0's last log record (step 19,164) each T loss is non-zero: goal 0.120, lat 0.153, lon 0.771, constraints 0.140.
* **Consistency is enforced twice.**
  * Hinge penalties for SPEED / STOP / TURN / LEAD agree with the frozen numpy rules.
  * At inference, a hard projection holds the plan to the R1 speed cap and the Kamm circle.
* **O** reads T's held code through `e_code` (`run_o`).

**What is NOT built:**

1. **The H-JEPA is not the hierarchy.**
   * In v8F-mini the H-JEPA is an auxiliary (`WMAux`) on the shared adapter. The v9 tactical labels supervise T's own heads, never the H-JEPA level-2 latent.
   * `PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` §4 specifies the brains AS H-JEPA levels, with *"all v9 tactical labels supervise the z² heads"* (R3). That is not what was built.
   * β then planned in that unsupervised z² and FAILED (`PREREG_V8F_BETA` Amendment 1).
2. **RA-1, the seed-stable H-JEPA line, has no tactical layer at all.**
   * `ra1_hjepa.py`, `ra1_hjepa_train.py` and `ra1_pooled.py` contain **0** references to tactical goals, constraints or nav.
   * Its H-JEPA levels are time scales (0.2 / 0.6 / 1.8 s) feeding one trajectory readout.
3. **Whether O USES T has never been measured.**
   * S's use of nav was measured: M1-navshuf − M1 = +4.25 m checkpoint error at 50 m, separated.
   * T → O has no intervention. A wired code that O ignores would be the *"built, tested and unreachable from its caller"* class (CLAUDE.md).

So the next lever cannot be chosen until two questions are answered: does O use T, and what would a better T buy?

## The arms (one session, one code version; `v8f_mini_eval.py run --tac …`)

Each arm runs on every checkpoint `c` ∈ {M1_s0, M1_s1, M2b_s0, M2b_s1}: the banked full-corpus runs, pool4x10, yaw rate.

| tag | what O reads | tier |
|---|---|---|
| `c-tacpred` | T's own held code: a plain run, re-run so all three arms share one code version | T0/T1 open loop |
| `c-tacshuf` | a **donor** window's predicted code: another **episode**, the same within-episode rank, donor seed 1 (`hjepa.donor_indices`); S, T and the scene are untouched | T0/T1 open loop |
| `c-tacorc` | the **GT v9 code** where the window has a v9 row; T's own code elsewhere | **T0 oracle**: reads GT; ranks levers, never a result |

## The readings (`…/2026-10-17-v8f-mini/code/v8f_tac_read.py`)

**Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000.
* It answers another draw of EPISODES, with the models fixed.
* Training-seed variance enters only through F = |mean_s0 − mean_s1| per arm, computed on the tacpred runs.
* The model is deterministic, so there is no inference variance.

**TAC-USED (decides):** shuf − pred, eval-141 ADE@6s, per checkpoint.

| verdict | condition |
|---|---|
| **USED** | > 0 and separated at **all four** checkpoints |
| **DECORATIVE** | separated at **none** |
| **PARTIAL** | otherwise, including a separated *negative* delta. Reported per checkpoint; it chooses no lever. |

**TAC-HEADROOM (T0 diagnostic):** orc − pred, per checkpoint.
* An arm is **LARGE** iff the delta is < 0, separated, and |Δ| > **2·F_arm** at **both** seeds. Otherwise it is **SMALL**.
* The lever is chosen on **M2b**, the K-PASS arm. M1 is reported beside it.

**Reported beside them, deciding nothing:**
* the paired LONGITUDINAL / LATERAL component deltas of both interventions (`ra1_hjepa_eval.paired_family_block`);
* the manipulation statistics.

**Refusals:** the reader refuses (rc 2) when:
* a file is missing;
* the grids differ;
* the shuffle changed T's code on < 95 % of windows (the intervention did not happen);
* a tacpred run differs from the banked eval by more than 1e-3 m on any window.

## Outcomes, committed now

| TAC-USED | TAC-HEADROOM (M2b) | what it means | the next lever |
|---|---|---|---|
| USED | LARGE | O uses T, and a perfect T would lower 6 s error beyond the training-seed floor | **The tactical layer's QUALITY.** Build **v8F-HJ**: the v9 tactical labels supervise the H-JEPA level-2 latent, and T reads it (PLAN §4.4 R3). Weakest head first: longitudinal decisions, macro-F1 0.47–0.53. Under its own PREREG and launch gate; an architecture change ⇒ PI go. |
| USED | SMALL | T is consumed, and its errors cost less than the seed floor | **The OPERATIVE layer:** the longitudinal speed decision (oracle per-window speed lever 0.20–0.39 m, `H-V8F-LON-1`), not the tactical one. |
| DECORATIVE | — | O ignores T: the hierarchy does not function end to end | **Fix T → O first.** O must be trained to depend on the code, by code dropout with GT-code teacher forcing or by the code as FiLM on O's residual head. Re-read TAC-USED before any tactical-quality arm. |
| PARTIAL | — | live at some checkpoints only | No lever is chosen. Read the live arm and name the difference. |

⛔ **The oracle is T0: it reads GT.** Its delta is never quoted as a driving result, and never as *"the tactical layer is worth X m"*. It bounds what a perfect T could buy *through this O*.

## Instruments and their proofs (all before any run)

* `stack/tanitad/models/v8f_mini.py`: `forward_diag(batch, code_override)`.
  * It runs through the same key guard as `forward_infer`, and O reads the override.
  * `out["code"]` records the code O read.
  * Neither training nor `forward_infer` uses it.
* `stack/scripts/v8f_mini_eval.py run --tac {pred,shuffle,oracle}`. Combining it with `--plan beta` or `--nav-shuffle` is refused.
* `…/v8f_mini_box2_queue.cmd tac` and `…/code/box2_ops/chain_v8f_tac.sh` (markers ZZV8FTAC-DONE / -STOP). `v8f_guard.sh`'s end test is now **anchored**.
* `stack/tests/test_v8f_tac.py`, 6 tests:
  * `forward_diag` with T's own code is **bit-identical** to `forward_infer`, and another code moves O;
  * the three eval modes;
  * the queue dry run;
  * the reader's decision table and refusals;
  * the guard's anchoring.
  * The rest of the v8F suite stays green: 141 passed.
* Mutation proof: **15 / 15 RED** (`…/raw/2026-10-20-tac/mutate_tac.out`).

## Amendment 1 (2026-10-21, after the result and BEFORE any per-group run): WHICH part of the code carries the headroom?

**Why it exists:** the result (`…/2026-10-17-v8f-mini/RESULT_V8F_TAC.md`) is **USED + LARGE**.
* The oracle lowers 6 s error by −1.05 to −1.30 m (T0), mostly longitudinally.
* But T's GT code holds categorical **decisions** (goals, lateral, longitudinal) AND continuous **constraints**, some computed from the FUTURE (`lon_v_target_ms`, `v_end_ms`, `stop_x_m`, the lead gap).
* "A perfect tactical layer" therefore includes perfect speed prediction. Two levers are open, and they are built differently:
  * a better tactical **classifier**: v8F-HJ, with the tactical labels supervising the H-JEPA level-2 latent;
  * a **speed-target** head inside T.
* The per-group oracle separates them at zero training cost.

**The arms:** each on the same four checkpoints, with the same eval and windows.
* `v8f_mini_eval.py run --tac oracle --tac-groups <g>`: O reads the GT code in group `g` only, and T's own code everywhere else.
* The six groups PARTITION the 60-dimensional code (`v8f_mini.CODE_GROUPS`, pinned):

| group | dims | holds |
|---|---|---|
| `goal` | 22 | the v9 tactical goals (incl. traffic-light colours) |
| `latdec` | 5 | the lateral decision |
| `londec` | 7 | the longitudinal decision |
| `latcons` | 12 | turn geometry, lane-change crossing |
| `lonspeed` | 8 | target speed, reach time and distance, start time, speed band, end speed |
| `stoplead` | 6 | stop point, lead gap and time gap |

* Plus two composites: `decisions` = goal + latdec + londec, and `constraints` = latcons + lonspeed + stoplead.
* 4 × 8 = 32 evals of about 25 s each. Eval only, box 2, SAFE MODE, the temperature guard.

**The reading** (`…/2026-10-17-v8f-mini/code/v8f_tac_groups_read.py`), all T0:
* Per checkpoint and group: d_g = orc_g − pred, paired over episodes, and **share_g = d_g / d_all**, where d_all is the full oracle PREREG_V8F_TAC read.
* A group **CARRIES** for an arm iff d_g < 0, separated, and |d_g| > 2·F_arm at **both** seeds.
* **Decided on M2b:**

| verdict | condition | the next arm (own PREREG and launch gate; PI go) |
|---|---|---|
| **CLASSIFIER** | `decisions` CARRIES and share ≥ 0.5 at both seeds | **v8F-HJ**: the v9 tactical labels supervise the H-JEPA level-2 latent and T reads it (PLAN_V8F §4.4 R3). The decision group with the largest share comes first. |
| **SPEED** | not CLASSIFIER, and `lonspeed` CARRIES and share ≥ 0.5 at both seeds | **A speed-target head inside T** (vision + nav + ego dynamics only), trained on the v9 lon speed constraints with its own weight, with O conditioned on it. |
| **MIXED** | otherwise | No arm. The two largest single groups are named, and their joint oracle is the next read. |

* Reported beside it, deciding nothing: every group at every checkpoint, M1's decision, and each group's LON / LAT family deltas.
* **Refusals:**
  * a missing file;
  * files on different grids;
  * a group run whose manipulation record is not `oracle / <g> / |g| dims`;
  * a tacpred run that differs from the one PREREG_V8F_TAC read by more than 1e-3 m.
* **Interactions are not assumed away.** Shares need not sum to 1, which is why the composites are read directly and not as sums.

**Proofs, before any run:**
* `stack/tests/test_v8f_tac.py`, now 10 tests, adds:
  * the partition;
  * a group oracle that writes GT only in its group;
  * **the six groups together rebuild the full oracle bit-for-bit**;
  * the refusals;
  * the reader's decision table;
  * the queue's `tacgrp` dry run.
* Mutation proof: **24 / 24 RED** across six files (`raw/2026-10-20-tac/mutate_tac_a1.out`).
* v8F suites: 147 passed, 2 skipped.
* Chain: `code/box2_ops/chain_v8f_tacgrp.sh`, with ZZV8FTAC markers in its own log, after PREREG_V8F_TAC's chain.

## Amendment 2 (2026-10-21, after Amendment 1's result and BEFORE any of these runs): what must T predict better, and can partial accuracy pay?

**Why it exists:** Amendment 1 read **MIXED** (`…/2026-10-17-v8f-mini/RESULT_V8F_TAC_A1.md`).
* The constraints **together** carry 0.88–0.90 of the oracle's headroom; the decisions together carry 0.07–0.12.
* Single constraint groups **hurt** alone (+0.18 to +0.49 m). O was trained on whole codes only (`v8f_mini.py:601-602`), so one GT group beside T's own code is a mix it never saw.
* Its committed next read is the joint oracle of the two largest groups. Two more reads decide how the training arm is built, and both cost zero training:
  * which constraint group is **necessary** inside the consistent set;
  * whether **partial** accuracy pays at all.

**The arms**, each on the same four checkpoints with the same eval and windows. All are `v8f_mini_eval.py run --tac oracle`; queue stage `tacgrp2`, 4 × 10 = 40 evals.
* **(A)** `--tac-groups lonspeed+londec`: the read Amendment 1 committed to.
* **(B)** `--tac-groups all-<g>` for each of the six groups: GT **everywhere except** group g, where O reads T's own code.
  * These are the mixes closest to the whole code O was trained on.
* **(C)** `--tac-groups constraints --tac-alpha {0.25, 0.5, 0.75}`.
  * O reads T's constraints moved a fraction α of the way to the GT: `code = code0 + α (code_GT − code0)` in the constraint dimensions only.
  * α = 1 is Amendment 1's `constraints` arm, bit-for-bit (pinned).
  * ⚠️ This is a **proxy** for "T's constraint error shrunk by (1 − α)". It moves every constraint by the same fraction, which a trained T would not do.

**The reading** (`code/v8f_tac_groups2_read.py`), all T0, paired over episodes:
* **A CARRIES** iff d_A < 0, separated, |d_A| > 2F, and share_A = d_A / d_all ≥ 0.5, at both seeds.
* **B:** loss_g = (all−g) − all, and necessity_g = loss_g / |d_all|.
  * g is **NECESSARY** iff loss_g > 0, separated, loss_g > 2F, and necessity_g ≥ 0.25, at both seeds.
* **C:** r(α) = d_cons(α) / d_cons(1), where d_cons(1) is Amendment 1's `constraints` arm. The class is read at α = 0.5, at both seeds:
  * **PAYS** iff r ≥ 0.4 and d_cons(0.5) < 0, separated;
  * **THRESHOLD** iff r < 0.2;
  * **INTERMEDIATE** otherwise.
* **Decided on M2b:**

| verdict | condition | the next arm (own PREREG and launch gate; PI go) |
|---|---|---|
| **STRUCTURAL** | C is THRESHOLD | No loss-level arm on T: halving T's constraint error would buy under a fifth of the gain. The next arm changes what T sees, or how it predicts the constraint set jointly. That is an architecture change: named, built only on the PI's go. |
| **LON-TACTIC** | not STRUCTURAL, and A CARRIES | A training arm on T's longitudinal tactic: the lon decision + lonspeed loss terms. |
| **CONSTRAINT-ACCURACY** | otherwise | A training arm on T's constraint prediction, targeting the NECESSARY constraint groups (all three if none is necessary alone). |

* Reported beside it, deciding nothing:
  * every row at every checkpoint;
  * M1's decision;
  * the LON / LAT family deltas, for A and C vs T's own code, and for B vs the full oracle;
  * **T's own code error per group**, from the `tac_code` the eval now saves. T's values in g come from `all-g`, and the GT from another `all-h`.
* **Refusals:**
  * a missing file;
  * files on different grids;
  * a manipulation record that is not `oracle / <spec> / <n dims> / <alpha>`;
  * a `tacpred` or `constraints` run that is not the one Amendment 1 read (ADE differs by more than 1e-3 m).

**Proofs, before any run:**
* `stack/tests/test_v8f_tac.py`, now 14 tests, adds:
  * the complement spec;
  * α = 1 is the oracle bit-for-bit, and α = 0.5 is the exact midpoint, in the selected groups only;
  * leave-one-out holds T's code exactly in the left-out group and the full oracle elsewhere;
  * the α refusals;
  * the reader's decision table and refusals;
  * the queue's 40-call dry run.
* `stack/tests/test_bash_specials.py`, the `GROUPS` trap that stopped Amendment 1's pull.
* Mutation proof: `raw/2026-10-21-tacgrp/mutate_tac_a2.out`.
* Chain: `code/box2_ops/chain_v8f_tacgrp2.sh`, derived by `mk_chain_grp2.py` from the fixed Amendment 1 chain, with ZZV8FTAC markers in its own log.
