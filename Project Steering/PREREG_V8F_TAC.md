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
