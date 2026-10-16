# PREREG_SAFEHEAD — a scorer whose safety outputs EPO cannot move (CPU simulation, design + verdict)

Research Lab agent, 2026-10-16. PI decision executed: "experiment with the separate safety head" (2026-10-16).
Written BEFORE any safety-head number exists (no SH arm has been run, no SH gradient probe has been read). sha256 in `PREREG_SAFEHEAD.sha256`.
Tree PK = archive of d06546a. Package `PK/raw/2026-10-16-safety-head/`. CPU only; the dev-box GPU lock is never opened.
Source question: `raw/2026-10-16-phead-epo-feasibility/RESULT.md` s.7 item 2 (prereg R3 (ii)): "a scorer whose safety outputs EPO cannot move".
Evidence class of every number this package will produce: **ESTIMATED** (real scorer weights, labels, trajectories and the trainer's own loss closures; substituted visual-context cache; CPU fp32; offline selection read; not T1; not a driving claim).

## 1. What the source says (ground truth `PK/refe/scorer_finetune.py`, d06546a)

- **Scorer** = `score_q_mlp` -> `score_dec` (4 `CrossBlock`s: self-attention over the 64-proposal set, then cross-attention to 7,680 visual tokens, `model.py:426-439`) -> `score_head = Linear(256, 7)` (`model.py:551-563`, forward `REFe.score_trajectories` `model.py:699-709`; `SCORER` `scorer_finetune.py:75`; copies `ScorerCopy` `:85-94`). Rows: 0 NC, 1 DAC, 2 EP, 3 TTC, 4 C, 5 DDC, 6 lane (`train.py:502` PDM_ORDER + `--lane-head`). Parameters 4,296,967 = q_mlp 81,408 + score_dec 4,213,760 + head 1,799 (`phead groups.json`).
- **Three scorers live in every run**: `base` = frozen copy of u400 (`:629-631`, `sx0` under no_grad `:738-739`); arm A = the model's own scorer, BCE only (`sxA` `:728`; `paramsA` `:1359`, `optA` `:1361`); arm B = `armB` (`:634-636`, `sxB` `:737`; `paramsB` `:1360`, `optB` `:1362`). One backward of `(lA + lB)/accum` (`:1381`); per-arm clip 1.0 + AdamW (`:1390-1395`).
- **Selection score** `sel_score(p) = agg_v1(p[..., :6]) * p[..., 6]` (`:750-752`), `agg_v1 = NC * DAC * (5 EP + 5 TTC + 2 C) / 12` (`:80-82`). DDC (row 5) is NOT in the selection.
- **EPO** (`epo_term` `:780-790`, `epo_loss` `:218-231`, `epo_policy` `:188-191`): pi = log_softmax(log(sel_score(sigmoid(sxB)) + 1e-6) / T), T = 0.1; J = sum_i pi_i R_i exact over the 64 proposals; loss = lam_pg * mean(-J + beta * KL(pi_base || pi)), beta 0.1, KL anchor = base logits (`--epo-ref base`, the R0 default, `:194-203`). R = `rl_reward.reward` (`rl_reward.py:152-189`): pdms_v1 - lam_d * D + guard (progress shortfall on SAFE proposals, `:92-114`).
- **Arm B's loss** = BCE(7 rows incl. lane) + lam * ListNet (lam 0 in R0) + EPO (`train_losses` `:792-800`). R0 argv: `--b-mode listnet --lam 0 --lr 1e-4 --lane-head --epo` (defaults epo_temp 0.1, beta 0.1, ramp 200, epo_lam 1.0, lam_d 0.15, w_prog 0.5).
- **Why FULL can trade safety for J**: the EPO gradient reaches every selection logit (NC, DAC, EP, TTC, C, lane) through `sel_score`. EPO may therefore raise the PREDICTED TTC / NC / DAC of a proposal it wants picked; the deployed pick then reads safety predictions that have been optimised for J rather than for calibration.

## 2. Which outputs are "safety", and why

**SAFE = {NC (row 0), DAC (row 1), TTC (row 3)}.** PROGRESS = {EP (2), C (4), DDC (5, no EPO gradient: not in `sel_score`), lane (6)}.
- `rl_reward.safe_mask` (`rl_reward.py:87-89`) defines a SAFE proposal as NC = DAC = TTC = 1: the reward's own safety gate, used by the progress guard (`:92-114`). These are exactly the three rows.
- In the selection, NC and DAC are multiplicative gates (`agg_v1`), TTC is the 5/12 term that every R0-family arm loses past g5 (R0 -1.00, R0b -0.57/-0.70, R0c -0.53/-1.10, `raw/2026-10-16-r0c-read/RESULT.md`).
- Lane (row 6) is a teacher-lane agreement multiplier, not a safety term (not in `safe_mask`); C is comfort; both stay on the progress side.

## 3. The designs (each is an arm; none is chosen silently)

All three keep arm B's whole scorer (q_mlp + 4 blocks + head) trainable by BCE + EPO, so the progress outputs keep `score_dec`'s capacity (the phead lesson: the gain lives in `score_dec`). They differ only in WHERE the three safety logits of the selection come from. Combined logits: `comb[..., c] = S[..., c]` for c in SAFE, `= sxB[..., c]` otherwise (`torch.where`, so the gradient into sxB's SAFE columns is exactly 0). The policy pi, J, the KL anchor and the deployed pick all read `comb` (pick = argmax `sel_score(sigmoid(comb))`). The KL anchor keeps the base logits (unchanged). At theta_0, S = base and sxB = base, so comb = base: every design starts from FULL's exact policy.

| arm | S (the safety rows' source) | what trains S | EPO gradient reaches | extra params at inference | extra trainable |
|---|---|---|---|---|---|
| **SH_F** frozen copy | the frozen base scorer (`sx0`, already computed every micro-batch, `:738-739`) | nothing (frozen at u400) | B only, via EP / C / lane columns | 4,295,939 (a second scorer forward; head rows 0,1,3 of 7 used) | 0 |
| **SH_A** BCE-trained copy | arm A (`sxA`, the BCE-only control the trainer already trains, `:728`, `:1359-1361`) | BCE only (arm A's own loss, unchanged) | B only, via EP / C / lane columns | 4,295,939 (a second scorer forward) | 0 (arm A already exists in every run) |
| **SH_SG** stop-grad, shared | arm B itself, its SAFE logits DETACHED in the EPO term only | BCE (B's own loss) | B's trunk (q_mlp + score_dec, 4,295,168) via the progress columns; B's SAFE head rows (771 params) get EXACTLY 0 | 0 | 0 |

- **SH_SG's honest limit, stated in advance**: its safety head rows get no EPO gradient, but its safety OUTPUTS sit on a trunk that EPO moves, so they are not EPO-immune. It is the zero-cost variant and the test of whether head-level separation is enough.
- **The separation line** (all three): `S_epo = S.detach()` in the EPO input. For SH_F the base logits carry no graph in training anyway (second guard).
- **Variant named, not run: SH_A3** (a separate copy trained by BCE on the three safety rows only). Its EPO-immunity is identical to SH_A's; it costs a third scorer's training where SH_A reuses arm A. Run only if SH_A reads GO/PARTIAL and the MM wants the narrower loss.
- **Gradient diagram (SH_A; SH_F identical with base in place of A, no BCE on base):**
  `cxy, sctx -> armB -> sxB --(BCE 7 rows)--> lB` ; `cxy, sctx -> armA -> sxA --(BCE 7 rows)--> lA` ; `comb = where(SAFE, sxA.detach(), sxB) --(epo_policy, J, KL vs sx0)--> ep` ; `backward((lA + lB + ep)/accum)`: armA <- dlA only (EXACTLY 0 from ep); armB <- dlB + dep (dep enters through EP / C / lane columns only).

## 4. Arms and controls (closed loop, the phead harness's real trainer step, bank stream 0)

| arm | lam 1 | its matched lam-0 control (the A of g5 for that design) |
|---|---|---|
| FULL | banked: P `closed_P_1.0.pt` (= `full_ref.pkl`), T the MM's `closed_s0_T_FULL_R_lr1_lam1k250T.pkl` | FULL lam 0 (= arm A) |
| SH_A | run (A and B co-trained) | lam 0 makes B == A bit for bit, so comb == A: **FULL lam 0** |
| SH_SG | run | lam 0: B == A, stop-grad is inert at eval: **FULL lam 0** |
| SH_F | run | **A's progress rows + base safety rows** (evaluated on the FULL lam-0 / arm-A states) |

Also reported, not deciding: every arm against FULL lam 0 (g5 as registered: deployed B vs the run's arm A).

## 5. Metrics and estimator

- **f = Delta J_design / Delta J_FULL** per pool (train 302 sets / 30 logs, in-sample; val 62 sets / 13 logs, out-of-sample), Delta J = J(lam 1) - J(matched control), J under each arm's OWN deployable policy (comb for SH arms). **f_min = min(f_train, f_val).**
- **Safety proxies** TTC, NC, DAC: P1 = the deployed pick's true sub-score, P2 = the policy expectation sum_i pi_i y_i; arm - matched control, x100, **pooled over the 364 sets** with the paired log-cluster bootstrap (10,000, seed 20260927; `phead_common.boot_ratio`). Per-pool values reported.
  - Pooled, not per-pool as in phead's row logic: P1 on 62 val sets moves in steps of 1.6 points, so one flipped pick fails a -0.5 bar. Declared here, before data.
- **Safety condition (per proxy)**: pooled P2 >= -0.5 AND pooled P1 >= -0.5 (x100; -0.5 = the g5 bound as phead used it).
- Also reported: EP and PDMS of the pick, KL(base || pi), displacement norms.
- Estimator, named: episode(log)-cluster bootstrap over the pool's logs, evaluation sets resampled, training state FIXED: it answers "another draw of evaluation sets", never "another training run" or "another data order" (`H-ESTIM-SEED-1`).

## 6. Proxy validity (phead control (d), extended to NC and DAC)

The TTC proxy is validated (`phead proxy_validity.json`: P2 B - A negative at 3/3 R0 states, C0 floor < 0.5). The same rule is applied, with no new compute, to NC and DAC from the same banked file. The rule needs a real RL-val B - A per term: NC has it (`ab_vs_base.json`); DAC has none for R0. A proxy that fails the rule or lacks the real reference is reported **UNVALIDATED** and still enters the safety condition (conservative: an unvalidated proxy can block GO, never grant it).
Disclosure: the banked NC / DAC proxy values in `proxy_validity.json` and R0c's real RL-val sub-scores were seen before this prereg was written. Neither is a safety-head number.

## 7. Horizons

- **P, K = 50** (constant lr 1e-4, constant lam_pg = lam, phead's regime), snapshots 25, 50. Decision at 50.
- **T, K = 250** (the trainer's schedule: warm-up 50, cosine over 1,000, lam ramp over 200), snapshots 50, 100, 250. Decision at 250.
- T order (by information per CPU hour): SH_SG lam 1 (1x), SH_A lam 1 (2x, co-trains A, saves A's snapshot states, which also give SH_F's T control), SH_F lam 1 (1x). One run at a time, 6 threads. Any T arm not finished at hand-back is left running in a detached queue, and a HANDOFF.md names the reader command.

## 8. Committed outcome rows

**V (validity, decided first, from FULL lam 1 vs FULL lam 0 at T K = 250, the MM's runs, read with phead_summarize logic, P2 TTC pooled x100):**
- **V-OK**: FULL's pooled P2 TTC < -0.5 at T K = 250. The sim reproduces the R0-family TTC loss; designs are judged at T on TTC.
- **V-NO**: FULL's pooled P2 TTC >= -0.5 at T K = 250. **The sim cannot falsify a safety design on TTC** (at P K = 50 FULL already reads +0.16, phead). Then no design can earn GO on TTC from the sim: the outcome is **SH-UNTESTABLE-TTC**, f is still reported, and the only TTC test is the A40 arm of the best-f design with its matched control (PI decision). NC / DAC rows are still read.

**Per design (applied at both horizons; a missing T reading makes the row PROVISIONAL, never GO):**
- **SH-GO**: f_min >= 0.5 at both horizons AND every safety condition (TTC, NC, DAC; P1 and P2) holds at both horizons, under V-OK. Next: implement in `scorer_finetune.py` (next agent; flag default OFF, section 9) and queue ONE A40 arm with its matched control. If several designs read GO: the cheapest at inference wins (SH_SG, 0 params), else the higher f_min.
- **SH-PARTIAL**: every safety condition holds at both horizons AND 0.2 <= f_min < 0.5 at the worse horizon; or f_min >= 0.5 at both but a safety condition fails at exactly one horizon. Next: (i) report to the PI that the safe arm keeps only part of FULL's gain (the rest came from moving safety predictions); (ii) CPU lever: the same design with B's lr x4 at P K = 50 (does step size recover f with safety intact?).
- **SH-NO**: any safety condition fails at T K = 250 (or at P K = 50 when T is missing), or f_min < 0.2. Next lever = **the objective, not the architecture**: (a) a TTC-penalised reward `R' = R - lam_ttc * (1 - ttc)` in `epo_reward` (one line in `rl_reward.reward`, CPU-testable on this harness with no other change), (b) Lagrangian / constrained EPO: maximise J s.t. E_pi[TTC] >= E_piA[TTC] - 0.005, with a dual weight mu updated per update. (a) is the cheapest and is run in this package if time allows.
- **SH_SG reading SH-NO while SH_A / SH_F read GO** is the expected signature of "head-level separation is not enough; the trunk carries the safety drift" and is reported as such.

## 9. Identity controls (all must pass before a safety-head J number is reported)

- **(a) separation OFF reproduces FULL exactly**: each SH arm with `--sep off` (comb = sxB) for K = 3, P, lam 1: B's parameters equal the FULL arm's (same harness, same threads) with max |diff| == 0.0. (a') the harness's own loss assembly vs the trainer's `train_losses` closure on one micro-batch: lB + ep equal (==). (a'') the co-trained arm A at P K = 25 / 50 vs the banked FULL lam 0 states (`closed_P_0.0.pt` D[25], D[50]): rel L2 <= 1e-3 (phead's (a3) tolerance; thread count differs).
- **(b) EPO's gradient on the safety source is EXACTLY 0.0, progress side non-zero, same breath** (one micro-batch, lam_pg 1, `autograd.grad(ep, ...)`): SH_F: the base copy's parameters (made grad-enabled for the probe, live forward); SH_A: arm A's parameters; SH_SG: rows 0, 1, 3 of B's `score_head.weight` / `.bias`. Control that must read non-zero: B's head row 2 (EP), and B's `score_dec` for every design.
- **(c) mutation** `SAFEHEAD_MUTATE_NO_SEP=1` deletes the separation line (`S_epo = S`): (b) must turn RED (non-zero) for all three designs.
- **(d) proxy validity**: section 6.

## 10. What this package does NOT do

No trainer change, no A40 run, no GPU. The flag spec for the implementing agent goes in RESULT.md: `--safety-head {off, frozen, armA, stopgrad}` (default off = R0 bit for bit), touching `train_losses` (EPO input), `evaluate()` / the score dump (the deployed pick must read comb), the checkpoint meta (which scorer provides the safety rows), and a G-DVB entry + a G11-style zero-gradient probe with its mutation.
