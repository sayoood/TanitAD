# PREREG_V8F_STOPPERC (DRAFT): can a perception lever raise T's discrimination of stop points, and does the plan follow at T1-kin?

Status: **DRAFT for the PI's review** (drafting agent 2026-11-11; Master Mind review notes in §7, 21:25). Not frozen; no code written, nothing trained, no box-2 or pod contact. Root: `E:/Projects/TanitAD-artifacts/dagland/motion_c/`. Read tree: `F:/TanitAD-scratch/stopbal_read/tree` (STOPBAL's read export). Tier of every driving number: **T1-kin** (self-action open loop, m = 10, eval-141, first tick). STRATEGIC: **PAUSED** (PI item 25). Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0 ("another draw of episodes"); "another training run" is the replicate floor; the planner is deterministic, so there is no inference-seed floor.

## 1. Why (all MEASURED)
* T misses the stop on ~9 of 10 stop-point windows: hit 0.103 / 0.128, 0.94 of the speed-error mass in the misses (`theory/stopdec/VERDICT_STOPDEC.md`).
* The cue is present but weak: AUROC of p(STOP_POINT) 0.836 / 0.857, recall 0.321 at an eval-chosen 5 % false-positive rate, AP 0.30 (`theory/stopcue/VERDICT_STOPCUE.md`).
* Weighting moves the operating point along an unchanged ROC: AUROC NULL 0.816-0.851 against lever arms 0.816-0.857, with all four lever-seed reads inside the AUROC floor of 0.038 / 0.036 (`theory/stopbal/read/VERDICT_STOPBAL.md:10-14`).
* A correct stop decision is worth −1.539 / −1.477 m tactical ADE on the 117 stop-point windows, floor 0.405 m, and O follows it (`theory/stopbal/oracle/VERDICT_STOPORC.md:10-11`). Hardening recovers 3-6 % of that, and nothing once false stops are held at the NULL rate (`VERDICT_STOPHARD.md`, `VERDICT_STOPTAU.md`).
* ⇒ The gap is **coverage**: T's discrimination of stop points. This card targets AUROC, with false positives held, and asks whether the stop-point ADE follows.

## 2. What T sees now (file:line on the read tree)
| fact | where |
|---|---|
| Frozen **DINOv3 ViT-L/16**, bf16, CLS + 4 registers dropped, every 2nd frame (0.2 s grid) | builder `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/code/pull_encode_pooled.py:5-8` |
| **One camera**: a 256x640 cylindrical frame, hfov 120°, f_ref 305.58 px, a 16x40 grid of 16-px patches. The camera is the front-wide 120° one (INHERITED: the r0/episode read set in CLAUDE.md; hfov matches) | builder `:7`, `:395`, `:418`; `stack/tanitad/data/v8f_mini_data.py:873` |
| T reads **`pool4x10`**: the 640 patch tokens average-pooled into 4x4-token cells, fp8, `[T, 40, 1024]`. Also in the file format: `pool8x20` (2x2-token cells, `[T, 160, 1024]`, "only if requested", builder default requests it) and `mean` (fp16 patch-mean `[T, 1024]`) | builder `:11-12`, `:23-25`, `:488`; `v8f_mini_data.py:3-5` |
| Trainer can select the grid: `--grid pool4x10 / pool8x20`, `n_tok` 40 / 160, with a G-DVB check | `stack/scripts/v8f_mini_train.py:247`, `:323`, `:995-996` |
| The only trainable layer that meets the features: `adapter` = LayerNorm + Linear(1024 → 256), plus `pos[n_tok]` and `time[hist]` | `stack/tanitad/models/v8f_mini.py:700-702`, `:857-859` |
| T's scene memory: hist 4 frames x 40 tokens = 160 tokens | `v8f_mini.py:183-184`, `:750-757` |
| T: `read_t` (4 learned queries, depth 1) cross-attends [ego (P10 features), nav, limit, S's checkpoints ; scene] → `trunk_t` → `h_goal` (22, STOP_POINT = index 6) and `h_lon` (7, STOP = index 2) | `v8f_mini.py:816-829`, `:708-715`, `:46`, `:48` |
| Existing vision-only auxiliary heads (level, lead) on the current frame's adapted tokens, with a runtime guard that they read vision only | `v8f_mini.py:726-737`, `:1307`, `:1345` |

⚠️ T also reads ego dynamics, nav and `limit` (`v8f_mini.py:170-171`). `limit` is a per-clip sidecar (`v8f_mini_data.py:93`, `:185`) and is read at inference: `limit` and `limit_ok` are in `INFERENCE_INPUT_KEYS` (`v8f_mini.py:170-171`) and reach T through `e_lim` (`:809`, `:820`, `:843`). MEASURED from source at the code of record `715ef8fc`: its builder `stack/scripts/build_v8_speed_max_labels.py` declares it `g_tac.goals.SPEED_BAND.v_hi_ms`, the ego's own realised maximum speed over [t0+2 s, +6 s] (`:16-19`, `:141`), an oracle input that the PI chose to use on 2026-09-10 in place of a speed-limit service (`:3`, `:25`). trigger-probe2 measured that it carries about twice the go/stay information of the nav token (`…/2026-10-17-v8f-mini/raw/2026-10-29-trigger-probe2/RESULTS.md` §2). So T's AUROC is not a vision-only number, and stage 0 measures what the non-vision inputs carry. **Every new path in this card reads vision tokens only.**

## 3. What exists for the three levers
* **Stop-region resolution.** `pool8x20` exists as a format and a trainer flag. ⚠️ Not verified: whether box 2's production train cache (4,572 clips, `local256`) and eval-141 cache hold `pool8x20` (the v8F-mini RESULT.md lists 8x20 as UNVERIFIED). ⚠️ `--init-ckpt` refuses it: `n_tok` is in `INIT_CKPT_MATCH` (`v8f_mini_train.py:662`, refusal `:681-683`), so an 8x20 arm cannot start from M2b without a new init rule. Higher still: Thor's 416x1024 cylindrical B1 cache (builder `:356`), which would need a re-encode.
* **LoRA arm L** (`H-V8F-LORA-1`, Amendment 4 `626aa1e5`): q, v LoRA r 32 / α 32 on all 24 blocks; init M2b, M2b's recipe, not DAG-OACC; trains on the pod at 45.27 s/step (MEASURED, register `D-V8F-LORA-FPRIME-1`). Its bar is against B-prime (F-prime 0.12701 m). Neither eval script has a LoRA path (`v8f_t1_eval.py`, `v8f_mini_eval.py`: 0 hits for `lora` / `encoder`), so L is read through a merged-encoder eval cache (`Project Steering/PREREG_V8F_LORA.md:115`). L's read can add the STOPCUE instrument at zero extra training, compared with **B-prime only**, never with DAG-OACC's 0.836 / 0.857.
* **A stop-line label does not exist for PhysicalAI-AV at any of three layers**, checked at each:
  * published corpus: no map (CLAUDE.md, card quote);
  * `obstacle.offline`: 10 classes, all dynamic agents (INHERITED, CLAUDE.md). It gives the lead (queue-stop) labels, which the existing lead head already uses;
  * augmented v7.2 release (`TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/s2_labels_v7.2_{train,eval}.jsonl.gz`, tip `5b1e83ab`; MEASURED by this draft's count over the release file): `STOP_SIGN` sits in `.semantics.referents[]` in **53 / 4,572** train and **1 / 147** eval records. It is one record per clip, provenance VLM-CoT, with no frame time and no image position. Traffic-light goals: 779 / 4,572 (INHERITED, CLAUDE.md).
  * A stop line does exist on nuPlan / NAVSIM maps (REFe `AMENDMENT_6_STOP_LINE.md`), but that is another corpus and another rig.
* **What does exist: the stop LOCATION.** The v9 STOP class and `stop_x_m` / `stop_y_m` come from the ego's future: where it comes to rest, in the NOW frame (`stack/scripts/build_v9_labels.py:937-942`). The STOP_POINT goal bit is set at `:1030-1037`. Geometry does not derive the cause (`stack/tanitad/data/g_tac_geom.py:445`). Labels may use ego, so this is admissible as a **stop-location auxiliary**. It is not a stop-line one: for queue stops it marks the point behind the lead.

## 4. Stage 0: zero training, decides whether a training arm is worth it
**S0-PRE (read-only):** read `meta.grids` of 3 train and 3 eval cache files on box 2. If `pool8x20` is absent, R needs a re-encode (PI compute question, Q2).

**S0-P, probes for STOP_POINT vs not:**
* **Fit (training clips only).** Sample up to 4,000 of the 13,116 training STOP_POINT-positive windows (MEASURED, `theory/stopbal/prevalence_stopbal.json`), plus 3x as many negatives, from the 4,512 training clips. The sample is drawn by a seeded rule fixed at freeze. Features: NOW frame only.
* **Model.** Logistic regression on per-token PCA-64 features. PCA, the L2 strength and the MLP's size are chosen by 5-fold clip-grouped CV **inside the training sample**.
* **Score once on eval-141.** The windows and positives are STOPCUE's: 2,365 windows, 131 positives, 139 episodes. No eval window touches a fit, a PCA or a threshold.
* **Arms.** E (T's non-vision NOW inputs); E−limit (beside); M+E (`mean`); **V40+E** (`pool4x10`, what T reads); **V160+E** (`pool8x20`). Beside: a 1-hidden-layer MLP per arm.
* **Controls.** Label shuffle (CI covers 0.5); leak (GT appended → 1.0); episode-shuffled labels; 0 clip ids shared between train and eval; the E inputs equal the dumps' first-tick inputs (max |diff| 0).

**S0-C, coverage ceiling:** the frozen STOPORC machinery (`stoporc_read.run_arm`) with F-GT applied to the top-c fraction of the 117 stop-point windows, ranked by T's own p, for c in {0.3, 0.5, 0.7, 1.0}, on BAL-NULL s0 / s1. It reads the hit rate a lever needs before P can clear its floor (STOPBAL's P floor is **0.703 m**, MEASURED, `VERDICT_STOPBAL.md:19`). This is an approximation: a better T may flag different windows.

**S0-L, deferred until L_s0 is banked:** STOPCUE on L vs B-prime s0, both dumped T1-kin through their own caches. A gain counts only if it exceeds the AUROC floor of 0.037.

| outcome (committed) | rule | licenses | does NOT license |
|---|---|---|---|
| **RES-ADDS** | paired dAUC(V160+E − V40+E) ≥ +0.02, CI > 0 (linear or MLP) | arm R | "resolution will move T", which only stage 1 decides |
| **HEAD-UNDERUSES** | V40+E probe AUROC exceeds T's own (same windows) by ≥ 0.037, CI lower bound above T | arm A first: the cue is in T's inputs and is not extracted | "perception is fine" |
| **FLAT** | neither | R deferred; A and S0-L stay | "the cue is not learnable". A linear negative says nothing about a trained adapter. |
| **CEILING-FAR** | S0-C: clearing 0.703 m needs c > 0.7 | goes to the PI before any box-2 time (Q3) | |

## 5. Stage 1: training arms (box 2, init M2b s{k}, DAG-OACC s{k}'s argv verbatim, only the listed flags added, 3,000 steps)
| arm | flags added | seeds | architecture change (PI approval) | cost, ESTIMATED from MEASURED 2,276 s / ~40 min per run (`PREREG_V8F_STOPBAL.md:18`, `AMENDMENT_3_DOSE2.md:12`) |
|---|---|---|---|---|
| **NULL** | none | 0 (then 1) | none | 40 min each |
| **NULL_rep** | `--seed 2` | 2 | none | 40 min |
| **R** (resolution) | `--grid pool8x20 --init-pos-upsample nearest` (NEW) | 0, 1 | **yes**: `n_tok` 40 → 160 from an M2b init. `pos` is upsampled (each 4x10 cell's row copied to its 2x2 children), and `n_tok` is exempt from `INIT_CKPT_MATCH` only under that flag. Scene memory 160 → 640 tokens in the S, T and O readers. | 60-120 min each: 4x the tokens and 4x the grid bytes read from the USB HDD; the G-LIVE smoke measures it |
| **A** (stop-location auxiliary) | `--stop-aux on --w-stop-aux W` (NEW) | 0, 1 | **yes**: a new vision-only head (MLP on the NOW frame's adapted tokens) and a new loss term. The loss is per-token BCE on the 4x10 cell into which the ego's stop location projects: the column from the bearing under the **cylindrical** model (⛔ not the pinhole FOV formula, TRAPS), the row from `sensor_extrinsics`. It also carries a presence BCE, and negatives get presence 0. The head's output is read by nothing at inference: its gradient shapes the shared adapter only. W is fixed at the gate smoke so that the term equals T's goal-BCE term at step 0, never tuned. | ~44 min each |
| **L-DAG** (conditional on S0-L) | DAG-OACC from L, on an L-merged train cache | 0, 1 | encoder weights only | the re-encode, ~15.3 box-2 h ESTIMATED (`PREREG_V8F_LORA.md:155`), plus 40 min per run |

* At most three lever arms. R and A run by the stage-0 outcome: both if RES-ADDS and HEAD-UNDERUSES hold, A alone if FLAT. L-DAG follows its own PI go.
* **NULL reuse rule (cost):** the new flags default to the tip, bit for bit. If NULL_s0's checkpoint sha256 equals BAL-NULL_s0's, then BAL-NULL_s1 and BAL-NULL_rep are reused. A same-seed run on this rig is bit-identical (`PREREG_V8F_STOPBAL.md:15`), so the sha match is the test. The new code must not move code that NULL executes.
* **Total, R + A:** ~5.5-7.5 h ESTIMATED, plus ship, gates and pulls. The chain refuses a projection above 12 h, and that case goes to the PI.
* **Launch:** a launch-gate PASS per run. G-DVB carries each new flag, read back from the built model. G-LIVE needs a finite stop-aux term, a non-zero gradient on the head and the adapter, and exactly 0.0 on the head with `--stop-aux off`. A mutation that deletes the term from the total must turn the test red. A vision-only guard on the head in the form `assert_lead_head_vision_only` uses (`v8f_mini.py:1345`). G-EVAL builds 8x20.

**Rules (committed before any run).** A delta is the arm minus the same-seed NULL. A cell's floor is |NULL_rep − NULL_s0|. A **gain** is separated on the better side AND above the floor; a **cost** is separated on the worse side AND above the floor. Both seeds must agree, else SEEDS-DISAGREE (a third seed, `--seed 10/11`).
* **P (PRIMARY):** tactical ADE (knots 11-30) on the 117 stop-point windows is a gain at both seeds. Reported beside: the fraction of the 1.5 m ceiling recovered.
* **G-AUROC:** STOPCUE AUROC of p(STOP_POINT) is a gain at both seeds. The lon-STOP AUROC is not a cost.
* **G-FP:** the false-positive rate at 0.5 on non-stop windows is not a cost.
* **NR:** all-window tactical ADE is not a cost at either seed. Four families per band, every cost named; TACTICAL includes STOPDEC's hit and false-positive rates; STRATEGIC PAUSED.

| outcome | predicate | next step (committed now) |
|---|---|---|
| **PERC-ADOPT** | P, G-AUROC, G-FP, NR at both seeds | the flag joins the recipe; then the stacked arm, and BAL-L at W 4.6991 re-read on the new base, because a better ROC makes the weighting lever's operating point worth moving |
| **PERC-AUROC-ONLY** | G-AUROC + NR, P fails | STOPORC and STOPDEC on the arm, zero training; then BAL-L stacked on it |
| **PERC-P-OTHER-PATH** | P + NR, G-AUROC fails | third seed, plus the CONSORC decomposition |
| **PERC-FP-COSTS** | G-AUROC, but G-FP fails | report; that arm plus the training-window threshold rule of STOPTAU, under its own card |
| **PERC-NULL** | neither P nor G-AUROC | the lever is eliminated with its row. Next: L-DAG if S0-L passed, otherwise the 416x1024 source (Q5) |
| **PERC-REGRESSES** | NR fails at either seed | not adopted. For A, W is halved and the arm re-read; for R, the row is reported |

**Prior (HYPOTHESIS):** LOW-MEDIUM for P. STOPBAL tripled the hit rate with no P gain. P's floor (0.703 m) is 47 % of the ceiling.

## 6. Open questions for the PI
* **Q1.** Approve R's and A's architecture changes, or one of them.
* **Q2.** Box-2 slot after the second dose. Can S0-P's extraction run on box 2's CPU during a GPU chain? If `pool8x20` is absent, approve a re-encode.
* **Q3.** P's floor comes from one replicate (0.703 m). Accept it, or add a second replicate seed (+40 min)?
* **Q4.** A as a pure auxiliary (this draft), or its output fed to T as a conditioning token (more power, but a new input to T's decision)?
* **Q5.** Approve compute for a 416x1024 re-encode if R is partial.
* **Q6.** If the second dose reads BAL-FP-COSTS, do the stage-1 arms carry no weight (this draft: one lever per arm)?
* **Q7 (Master Mind).** T's stop cue partly reads `limit`, the ego's own realised speed ceiling over [t0+2 s, +6 s]. A ceiling near 0 says "the ego stops within 6 s", which is close to the STOP_POINT label's own source. Should the card also report T with `limit` masked (`limit_ok` = 0 in the T1-kin dump, zero training), so that a perception gain is read on the part of the cue that vision has to carry?

## 7. Master Mind review notes (2026-11-11 21:25; these do not change §4-§5)
* **Stage 0 first is right.** It costs no training and decides whether R earns box-2 time. Recommended order: S0-PRE, S0-P, S0-C, then the PI's go on stage 1.
* **`limit` (now MEASURED from source, §2).** It is a PI-approved input (2026-09-10), so it is admissible. But it carries ego-future speed, and the E−limit arm sits "beside" in S0-P. Recommendation: make E−limit a committed read and add the masked-`limit` T1-kin dump of Q7. Both are zero training. Without them, a probe that beats T could be beating a channel vision cannot see.
* **The bar is hard by design.** P's floor 0.703 m is 47 % of the 1.5 m oracle ceiling, from one replicate. STOPBAL tripled the hit rate with no P gain. The prior stays LOW-MEDIUM. Recommendation on Q3: add the second replicate (+40 min), because the whole verdict rests on that floor.
* **Recommendations on the PI's questions (the decisions are the PI's):** Q1, approve A first (no init-rule change, ~44 min per run) and R only on RES-ADDS. Q2, run S0-P's extraction after the dose-2 and SLOWWM chains, because both read the same USB HDD. Q4, keep A as a pure auxiliary: a conditioning token is a new input to T's decision, and attribution then needs its own arm. Q5, defer until R is read. Q6, yes, one lever per arm.
