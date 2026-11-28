# PREREG R0e-pick (FROZEN): constrained EPO with a pick-aligned constraint temperature, `--epo-con-temp 0.01`

**FROZEN, registered 2026-10-18. Drafted 2026-10-18 by a Research Lab design agent; landed by the Master Mind.**
- **Registered with the PI decisions of 2026-10-18 ~17:30 Berlin** (verbatim in `Project Steering/PI_DECISION_QUEUE.md`, section "DECIDED 2026-10-18 ~17:30 Berlin"). The PI's questions 1a-1f are decisions 1-6 of the draft's `NOTES.md`:
  - **1a** Stage 1 A40 spend AUTHORISED, queued after P6. Stage 2 runs only on outcome O1, as drafted.
  - **1b** T = 0.01, as drafted. The disclosed departure from A2R2's T\* = 0.02 is kept (§1.3).
  - **1c** R0e-id runs to **u1000**, not u250. The cost table is recomputed (§7). Build item B4 is dropped.
  - **1d** An O2 outcome does NOT earn Stage 2, as drafted.
  - **1e** The DEPLOYABLE-SAFE vs-BASE line stays report-only, as drafted. No E clause.
  - **1f** PRE-AUTHORISED: if DDC separates adverse again, g5 + DDC as a third constrained term is the committed next lever. It is added to the "next arm" text of the §6 rows only.
- **Draft sha256** (`PREREG_R0E_PICK.draft.md`, git blob `304b49d2…`): `d481ffc7078553311043485ec4803e2bb50ef664280e29006f3171e8bbe27a25`.
- **This file** is hashed in `PREREG_R0E_PICK.sha256`. The unified diff draft → this file, with one justification per hunk, is `REGISTRATION_DIFF.md` in this directory.
- **Everything not named in that diff is the draft, unchanged.** In particular the build-item status lines in §2.3 and §10 are as drafted (mirror tip `8bb47ad`). Their current status lives in BACKLOG row `R0E-PICK`, not here.
- No R0e number exists. Nothing has run: no R0e training step, no trainer edit, no pod or GPU access.
- Sources: mirror tip `8bb47ad`, plus the SO-PICKCON package being landed (`C:/Users/Admin/pickcon/out/`, trainer blob `869b1862…`).
- PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
- Parent pre-registration: `PK/raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.md` (sha256 `b404c2fb…`), with addendum 1 (sha256 `3c6a25c1…`), and `SPEC_SAFEOBJ.md`.
  - **Everything not named below is inherited from those files unchanged**: tiers, estimator, kills, the outcome precedence, and the four families.
- Evidence class on every number:
  - **MEASURED**: ours, with an artifact path.
  - **ESTIMATED**: an offline read, or arithmetic.
  - **INHERITED**: from another package or a register row, not re-verified here.
  - **HYPOTHESIS**.

## 0. Headline

- **Lever (one): the constraint's softmax temperature, 0.1 → 0.01.**
  - R0e-pick is R0d's argv token for token, plus **`--epo-con-temp 0.01`** and its own `--out` / `--dump-scores`.
  - The EPO reward term keeps `--epo-temp` 0.1.
  - The dual (η, λ_max, λ0, eps), the feed (the training micro-batches), the seed (0) and the data are all R0d's.
- **Why.** R0d's dual held its own quantity in-sample, yet breached on the pick (O4). That quantity was E_π[TTC] at T = 0.1, and on full RL-val it is not the pick:
  - for unconstrained R0 at u1000, the dual's quantity lost +0.22 [−0.15, +0.62];
  - the pick lost +1.00 [+0.30, +1.73];
  - source: MEASURED, `PK/raw/2026-10-18-so-carve/fullval_a1.json`, R49 erratum applied.
  - A colder constraint tracks the pick (§1).
- **Dual feed: training batches, exactly as R0d.** The log-disjoint carve is the committed successor lever for outcome P-O4-PICK (§6). It is not part of this arm, because:
  - it is a second lever;
  - its feed half does not exist in the trainer (§2.3).
- **Arms, in order:**
  1. **R0e-id**: the identity control, T = 0.1 named, run to **u1000** (PI 1c): a full same-seed replicate of R0d at all four checks (u250 / u500 / u750 / u1000).
  2. **R0e-pick**: seed 0.
  3. Stage 2, only if R0e-pick reads O1: **R1e-pick** (seed 1, same flags), then **R1o** (seed 1, constraint off, as registered in SAFEOBJ).
- **Bars:** S, G, L, E and K0–K6 are **unchanged** (§4). FREEPOINT read FP-NONE, so the navtest bar adds nothing.
- **Cost (ESTIMATED, §7; recomputed for PI 1c):**
  - Stage 1 is ≤ **7.36 A40-h**: 3.68 for R0e-id (to u1000) and 3.68 for R0e-pick.
  - Stage 2 adds ≤ **7.36 A40-h**.
  - The maximum is **14.72 A40-h**.
  - If R0e-pick is K3-killed at u500, Stage 1 costs 5.75 (3.68 + 2.07).
  - The draft's 4.95 / 12.31 / 3.34 assumed R0e-id stopped at u250 (1.27). The difference is the +2.41 A40-h of PI 1c.
- **PI decisions, taken 2026-10-18 ~17:30 Berlin (header block):**
  1. A40 spend: AUTHORISED for Stage 1, queued after P6 (1a).
  2. T = 0.01, the disclosed departure from SO-PICKCON's T* = 0.02 (§1.3) (1b).
  3. Stage 2 under the same rule as SAFEOBJ ("conditional": only on O1). An O2 outcome does not earn it (1a, 1d).
  4. R0e-id runs to u1000; build item B4 is dropped (1c).
  5. The DEPLOYABLE-SAFE vs-BASE line beside E is report-only (1e).
  6. If DDC separates adverse again, g5 + DDC as a third constrained term is the pre-authorised next lever (1f; §6 rows).

## 1. Why this lever, and why T = 0.01

### 1.1 The defect it targets (Rule Zero point 5: the largest measured term)

| quantity, R0 u1000, B − A on full RL-val (3,000 sets / 75 logs) | value (points, x100) | source |
|---|---|---|
| the dual's quantity, E_π[TTC] at T = 0.1 (P2) | +0.223 [−0.15, +0.62] | MEASURED, `so-carve/fullval_a1.json` |
| the pick's TTC loss (P1), which bar S reads | +1.000 [+0.30, +1.73] | same |
| the train-to-val gap of the dual's quantity, G_tv_full | −0.789 [−2.17, +0.34] (unresolved; the wrong sign for a train-to-val story) | same |

R0d's own in-sample log tells the same story. The dual drove its quantity to about zero, and the pick stayed below arm A. MEASURED, `PK/raw/2026-10-18-r0d-outcome/raw/sft.jsonl`, 31 post-ramp rows u226–u1000 (mean ± se of the 25-update rows, x100):

| quantity | value |
|---|---|
| `con_g_ttc` (the dual's quantity) | −0.085 ± 0.026 |
| `con_p1_gap_ttc` (the pick) | −0.232 ± 0.069 |
| in-run wedge (g + p1gap) | −0.317 |

⇒ The first-order defect is the **quantity**, not the data split. The split has no demonstrated gap at R0's state. This is a HYPOTHESIS for a constrained policy: SO-CARVE §9 says only an R0e run can test it.

### 1.2 What A2R / A2R2 say about T

Source: SO-PICKCON, `raw/2026-10-18-so-pickcon/a2r_neartie.json` and `a2r2_tieaware.json`. Both are MEASURED on R0's banked states. Tier: offline selection, not T1.

* **A2R, pre-registered before its numbers (sha256 `55fb333c…`).**
  * Outcome: **A2R-UNDERPOWERED**.
  * Coverage of the near-tie-free subset was 0.458 / 0.449 / 0.439, below the 0.50 floor.
  * Control 3' passed (≤ 2.2e-10).
  * **No tracking verdict comes from A2R.**
  * The headline is a finding in its own right: near-ties are the norm. 1,426–1,483 of 3,000 sets have a top-2 margin below 2.3e-3 in arm A.
* **A2R2, tie-aware pick (sha256 `81a8f3c9…`).**
  * ⚠️ **Post-hoc-motivated.** It was written after A2R's UNDERPOWERED outcome (before its own numbers), and its own §0 says so.
  * Outcome: **A2R2-ALIGNED(T\* = 0.02)**. Its committed rule names T\* = **the largest aligned grid T**, and that is 0.02.

  | T | u250 diff | u500 diff | u1000 diff | max \|diff\| | aligned (≤ 0.25) |
  |---|---|---|---|---|---|
  | 0.1 | −0.452 | −0.567 | −0.760 [−1.258, −0.272] | 0.760 | no |
  | 0.05 | −0.258 | −0.360 | −0.471 | 0.471 | no |
  | 0.02 | −0.121 | −0.141 | −0.208 [−0.521, +0.118] | **0.208** (margin 0.042) | yes, **thin** |
  | 0.01 | −0.103 | −0.032 | −0.103 [−0.334, +0.134] | **0.103** (margin 0.147) | yes |

  Each diff is L_T − L_P1^{τ0}. A negative diff means the tempered expectation **understates** the pick's loss. It does so at every T and every state.

### 1.3 The choice: T = 0.01, departing from A2R2's committed T\* = 0.02 (disclosed)

* **Why not 0.02, A2R2's committed T\*:**
  * Its margin is 0.042 at u1000.
  * It is **not** aligned at τ = 1e-2 in A2R2's descriptive sensitivity.
  * On A2R's under-covered near-tie-free subset (descriptive) it reads −0.367 at u1000, which is not aligned.
  * The three reads that favour 0.01 are all descriptive and not part of any verdict:
    * A2R2's τ-sensitivity: 0.01 is aligned at all five τ.
    * A2R's S\* subset: 0.01 reads −0.193 at u1000.
    * SO-CARVE A2, which is INVALID as registered (R49): 0.01 max 0.131 vs 0.02 max 0.225.
  * Every read, valid or not, puts 0.01 closer to the pick than 0.02. **That is a design choice informed by descriptive reads, not a claim.**
* **Why the alignment margin matters here.**
  * The dual holds the tempered quantity at eps = 0, so whatever it understates passes through to the pick.
  * R0d's in-sample pick gap (−0.23) and its RL-val pick (−0.77) differ by about 0.5. ESTIMATED: two tiers, the val CI is ±0.71, unresolved.
  * With S's tolerance at 0.5, each 0.1 of understatement is spent margin.
* **Why not a committed pair (0.02 and 0.01, both seed 0).**
  * The pair's alignment difference at u1000 is 0.105 points (0.208 vs 0.103).
  * That is below the "same run again" CI half-width of 0.13-0.14 (R0-A vs C0-A, PREREG_SAFEOBJ §7, INHERITED).
  * So two one-seed arms could not tell 0.02 from 0.01. The +3.68 A40-h buys more as a seed replicate (§5).
* **The cost of going colder** (HYPOTHESIS, untested on the real rig):
  * The softmax gradient scales about as 1/T and concentrates on near-tie sets.
  * On the tiny CPU rig, MEASURED: 0.327 at T 0.1 vs 0.882 at T 0.02 (`so-pickcon/pickcon_measure.json`).
  * It is bounded by `clip_grad_norm_(ps, 1.0)` on the joint gradient (patched `scorer_finetune.py:1693`). So the risk is not an explosion. The risk is that **the constraint crowds the EPO step out of the clipped budget**.
  * That is committed outcome O3-a, whose next lever is T = 0.02 (§6).
* ⚠️ **A2R2 was measured on UNCONSTRAINED R0 states.** Whether 0.01 stays aligned along a constrained trajectory is read in-run, as the wedge (§8, D1). It decides a row.

## 2. The lever, its argv, and the data plumbing

### 2.1 Argv (profile `refe_rls1d` extended, or a new `refe_rls1e`; arms file `eval/rls1e_arms.json`)

`BASE_R0d` is R0d's registered argv, verbatim from `PK/eval/rls1d_arms.json` → `arms.R0d.argv`. It derives from R0's token argv (sha256 `24451b62…`) plus the five `--epo-con*` flags.

| arm | argv | seed | registered `epo_con_temp` (built) |
|---|---|---|---|
| **R0e-id** | `BASE_R0d` with `--out …/arms/R0e-id --dump-scores …/arms/R0e-id/scores` **+ `--epo-con-temp 0.1`** | 0 | 0.1, named |
| **R0e-pick** | `BASE_R0d` with `--out …/arms/R0e-pick --dump-scores …/arms/R0e-pick/scores` **+ `--epo-con-temp 0.01`** | 0 | 0.01, named |
| **R1e-pick** (Stage 2) | R0e-pick with `--seed 1` and its own paths | 1 | 0.01, named |
| **R1o** (Stage 2) | SAFEOBJ's registered R1o: BASE + `--epo-con off`. `--epo-con-temp` must **not** appear (the trainer refuses it with `off`) | 1 | — |

* ⛔ **`--epo-con-temp` must be NAMED.** Its parser default `None` inherits `--epo-temp` 0.1 (`EPO_CON_TEMP_DEFAULT`, patched `:234`; resolution `:954`). That default silently **reproduces R0d**.
  * The profile's `arm_must_name` for R0e-id, R0e-pick and R1e-pick lists `--epo-con-temp`, and `arm_epo_registered` holds the value.
  * A token for an argv that omits it is refused: SO-PICKCON gate test G10.
* **Self-check in the arms file:**
  * R0e-pick's argv, minus `--epo-con-temp 0.01` and with its paths swapped to `/arms/R0d`, must equal `arms.R0d.argv` token for token.
  * The same holds for R0e-id with `--epo-con-temp 0.1`.
  * `lever_tokens_not_in_R0d` must be exactly `{--epo-con-temp, <T>, own paths}`.
* ⛔ **`--max-updates 250` is NOT an identity argv.** The cosine schedule runs over `total = min(total, a.max_updates)` (patched `:1661-1662`), so the learning rate at u ≤ 250 would differ from R0d's.
  * R0e-id therefore runs R0d's `--max-updates 1000`, **to completion (PI 1c)**.
  * It is **not stopped at u250**. Build item B4 (supervisor stop-after-dump) is dropped; no stop-after-dump feature is built.
  * R0e-id is thus a full same-seed replicate of R0d at all four checks.

### 2.2 The hop trace (SO-PICKCON, MEASURED on the CPU rig; patched trainer `869b1862…`)

1. `--epo-con-temp` (`:837`).
2. `EpoDual(..., temp=…)` (`:954`).
3. `dual.temp` → `built_facts()["temp"]`, which feeds the data event and G-DVB (`:303`, `gate_checks.py:649/662/667`).
4. `epo_con_term(..., dual.temp, ...)` (`:1068`).
5. `epo_policy(…, T)` for both arms (`:274-275`, `:201`).
6. `lB = lB + ct` (`:1069`).
7. `((lA + lB)/a.accum).backward()` (`:1678`).
8. G13 `temp_reach` doubles T in the preflight (0.01 → 0.02) and requires the L_con-only gradient to move, with arm A at exactly 0.0 (`:1593`).

Tests: `test_rl_pickcon.py` ZZTEST_PASS 20, i.e. 19 RED mutations. Default vs explicit 0.1 is bit-identical (max |diff| exactly 0.0).

### 2.3 Dual feed: training batches, not a carve, and what a carve would need

* **What exists** (tip `8bb47ad`, blob `e0639288` = R0d's trainer, INHERITED from SO-CARVE that R0d ran commit `ac07b1a`, whose blob is the same; patched line numbers in brackets):
  * The **hold-out half** exists as a data filter.
    * `--heldout-logs` `:768` [`:783`].
    * `tr_idx` drops every row whose `log_name` is listed (`:907-908` [`:925-926`]).
    * G7 asserts `train_rows_in_heldout_logs == 0` (`:1450` [`:1471`]).
  * But that file **is** the RL-val log list in the registered argv (`/workspace/data/rls1/rl_val_logs.txt`). Using it for a carve would rewrite a registered input.
* **What does not exist: the feed half.**
  * The dual reads only `con_acc` (`:1648` [`:1669`]).
  * `con_acc` is summed from the **training** micro-batches (`:1664` [`:1685-1688`]).
  * `dual.step(con_acc)` runs once per update (`:1683` [`:1704-1705`]; `EpoDual.step` `:308` [`:315`]).
  * There is one `DataLoader` over `tr_idx` (`:966` [`:985`]).
  * There is no carve flag, no second loader, and no path by which a non-training set reaches the dual.
* **To build it** (0 GPU, ESTIMATED ~1 day, the scale SO-CARVE quotes for one flag + gate entry):
  * a registered carve log list, disjoint from RL-val and frozen by sha256 before any data;
  * its exclusion from `tr_idx`;
  * a carve loader, with a no-grad forward of arms A and B on one carve micro-batch per update;
  * `EpoDual.step` fed from the carve accumulator;
  * logging of both the in-batch g and the carve g;
  * a G-DVB entry;
  * a G7-style disjointness gate (carve ∩ train = ∅, carve ∩ RL-val = ∅);
  * a reachability test with a mutation that must go RED.
* **Why it is not in this arm:**
  1. It is a **second lever**. It removes the carve's logs from training, so the 16,000-set stream and its order change. That breaks the R0d contrast, and R-family data parity with it.
  2. A1 shows no train-to-val excess of the dual's quantity at R0's state (§1.1).
  3. It cannot launch today.
* **Ordering.** The carve is the committed next lever of row P-O4-PICK (§6), and **its build should start now as a parallel 0-GPU stream**, so that row is not blocked.

## 3. Reads, tier, estimator (inherited unchanged)

* **RL-val selection tier.**
  * 3,000 sets / 75 logs, at u250 / u500 / u750 / u1000, B vs its **own** A.
  * Instruments: `rl_val_eval.py`, plus the K3 monitor `rls1_checks.py`.
  * This tier is **never an endpoint**.
* **Endpoint, T1 (NAVSIM v1.1 PDMS, full navtest, 12,146 tokens / 136 logs).**
  * It is a pseudo-simulation of a 4 s open-loop plan, not closed loop.
  * It reads the final u1000 B and A of each passing arm. There is no navtest early stopping.
* **Estimator:** paired log-cluster bootstrap, 10,000 resamples, seed 20260927, all values x100.
  * It answers *another draw of LOGS* only.
  * Training variance is §5's job. Inference variance is the I-rep's job (PLAN_IREP: A40, R0e-pick-B u1000 in place of R0d-B).
  * **Each separated CI names which of the three questions it answered.**

## 4. Bars: none changes

| bar | text (PREREG_SAFEOBJ §5 + addendum R1/R4) | change for R0e-pick |
|---|---|---|
| **S** | TTC and NC point deltas (B − own A, RL-val) ≥ −0.5 at **every** check u250–u1000; g3 at every check (pick / fan-mean path ≥ 0.894, shortest-bin share ≤ 0.55) | **none** |
| **G** | at u1000: the PDMS lower 95 % bound > 0 **and** the point ≥ G(t) = max(+0.60, F(t)), where F interpolates dose.json's vertices at full precision (addendum R1) | **none.** ⚠️ FREEPOINT measured the frontier's α-0.25 point M25 on navtest at PDMS −0.47 [−1.03, +0.24] vs BASE (D-REFE-FREEPOINT-1, INHERITED), so G's premise ("a free deployable point exists") is weaker on the endpoint than on RL-val. G is kept as written (PI 1d: an O2 outcome, safe but below G, does not earn Stage 2) |
| **L** | TTC only (A-L1): the constraint's effect (R·e-pick B−A) − (unconstrained B−A) > 0 at u1000 on both seeds, and > 2 × F_seed | **none.** The unconstrained partners are **banked R0** (seed 0) and **R1o** (seed 1, Stage 2). F_seed = max(\|ΔTTC(R0 vs R1o)\|, \|ΔTTC(R0e-pick vs R1e-pick)\|) |
| **E** | both pairs (R0e-pick, R1e-pick) on navtest: each ≥ +0.25 and mean ≥ +0.5; pooled lower bound > 0; mean > 2 × max(seed floor, I-rep \|Δ\|); per pair, the TTC and NC upper bounds ≥ 0 (A-E2); every sub-score reported | **none.** `PREREG_FREEPOINT.md`: *"FP-NONE: neither is DEPLOYABLE-SAFE ⇒ the safety-objective arm proceeds exactly as pre-registered"*, and the read was FP-NONE. So "beats R0B against BASE" is **not** added |
| **K0–K6** | `PREREG_RL_S1.md` §5. K3/K4 vs own A. K3 = any g3/g5 breach at two consecutive checks | **none.** R0e-id and R1o run K3 **observe-only**; K2 and K6 stay live |

* **Reported beside E, decides nothing.** Each passing arm's B **vs BASE** on navtest, under FREEPOINT's DEPLOYABLE-SAFE definition: the PDMS lower bound > 0, and none of NC, DAC, TTC separated below 0. This is the trade D-REFE-FREEPOINT-1 measured on R0B: PDMS +0.55 [+0.24, +0.84], NC −0.25 [−0.45, −0.06], TTC −0.40 [−0.71, −0.09] (INHERITED). It is the deployment question; E remains the registered endpoint.
* ⚠️ **One instrument must change; no bar does** (build item B3).
  * `eval/safeobj_outcome.py` accepts a nine-key `epo_con` block only when `temp` is exactly `REGISTERED_CON_TEMP = 0.1` (SO-PICKCON fix R2).
  * As landed, it would read R0e-pick as **NOT_READ**: the "built, tested, unreachable" class.
  * The registered T must become **per arm** (R0e-id 0.1; R0e-pick and R1e-pick 0.01).
  * Mutation tests: R0e-pick's log under a registered 0.1 must NOT_READ, and under 0.01 must read.

## 5. Controls

| control | what it answers | cost |
|---|---|---|
| **Launch-gate PASS token (K0)** | each of R0e-id, R0e-pick, R1e-pick and R1o starts only on a token from `refe/launch_gate.py` bound to commit, argv sha256 and data-manifest sha256s; `RLS1D_SUITE` + `test_rl_pickcon.py ZZTEST_PASS 20`; G13 incl. `temp_reach`; G-DVB `epo_con_temp` built == registered | in §7's preflight line |
| **R0e-id, the T = 0.1 identity, run to u1000 (PI 1c), vs banked R0d. The gate reading is at the first check (u250); u500 / u750 / u1000 are banked as further floors** | (i) the PICKCON bytes, as shipped, run R0d's configuration on the A40; (ii) **new information**: the "same run again" floor for **arm B** of a constrained EPO arm (training non-determinism at the same seed: bf16 autocast and the atomics-based memory-efficient SDPA backward. Patched `:1369-1373` records two head-less A40 runs differing in all 78 scorer tensors, max \|d\| 1.2e-4; the main loop runs without strict determinism). It has never been measured | 3.68 A40-h |
| **Free "same run again" floor for arm A** | R0e-pick-A vs R0d-A, same seed and flags; A never sees the lever. In SO-PICKCON's CPU test, a change of T leaves arm A bit-identical. Precedent R0-A vs C0-A: PDMS +0.01 [−0.13, +0.14], TTC 0.00 [−0.13, +0.13] (INHERITED, PREREG_SAFEOBJ §7) | 0 |
| **Second seed R1e-pick** (Stage 2, iff R0e-pick reads O1) | the training-seed floor of the constrained arm at T = 0.01; replication of S and G | 3.68 A40-h |
| **R1o** (Stage 2, after R1e-pick passes S) | bar L's seed-1 unconstrained partner, and the programme's first seed replicate of the unconstrained R trade | 3.68 A40-h |

### 5.1 R0e-id: registered readings, run in this order; the first that matches is the outcome

1. **ID-FAIL (blocks R0e-pick).** Any of:
   * the data event's `epo_con` block ≠ R0d's eight keys + `temp: 0.1`;
   * its `epo` block ≠ R0d's;
   * `train_sets` ≠ 16,000;
   * the arms-file self-check fails;
   * G13 not PASS;
   * the **step0** eval's `base.pick`, `A.pick` or `B.pick` differs from R0d's step0 by > 0.10 points. That is three sets net of 3,000: pre-training forward only, with near-ties allowed for. R0d's step0 `base.pick` was 90.8428 (MEASURED, `r0d-outcome/raw/sft.jsonl`).

   Meaning: the shipped trainer is not R0d's in this configuration. Next, 0 GPU: `launch_closure_audit.py` on the shipped bytes, a diff of the data events, and the CPU identity test on the pod; fix, then re-run R0e-id.
2. **ID-DIVERGED (blocks R0e-pick; escalate).** At u250, the same-run |Δ| of the **B − own A** contrast between R0e-id and R0d is > 0.50 for TTC or for PDMS. R0d's u250 values are TTC −1.167, PDMS +0.683 (MEASURED, `r0d-outcome/raw/rlval_B_vs_ownA_u250.json`).

   Meaning: the rig's same-run floor is as large as S's tolerance, so a one-seed S read is uninterpretable. Next: a PI decision between two seeds as the unit of S (both must pass) and stopping.
3. **ID-PASS.** Otherwise. R0e-pick launches.
   * Banked as floors, quoted beside every R0e-pick vs R0d contrast: the paired B_id − B_R0d and A_id − A_R0d on RL-val at u250 (PDMS, TTC, NC with CIs), and the contrast-level |Δ|.
   * Because R0e-id runs to u1000 (PI 1c), the same floors are banked at u500, u750 and u1000 and quoted beside the R0e-pick vs R0d contrast at the matching check. The ID-FAIL and ID-DIVERGED readings above stay at step0 and u250.
   * A **separated** paired difference here is reported as a finding (the H-ESTIM-SEED-1 family) and **does not block**. With 6 cells at ≥ 5 % per cell, a chance separation is expected about 26 % of the time, and the tiny rig measured 9.5 % per cell.

## 6. Committed outcomes, each with its next arm (Rule Zero)

Precedence is addendum R2's: identity first, then O6 > O7 > breached (O5 / O4 rows) > not breached (O1 / O2 / O3). The quantities are addendum R3's (signed window means of the logged `con_g_*`; the window is the 100 updates before the first breached check). **New in R0e-pick:** the breached rows split on the in-run wedge D1 (§8). It is read over post-ramp updates through the first breached check, and for a non-breached arm through u1000.

| outcome | condition | meaning | next arm (and what blocks it) |
|---|---|---|---|
| **ID-FAIL / ID-DIVERGED** | §5.1 | R0e-pick never launched | §5.1 |
| **O6 INERT** | λ_TTC = λ_NC = 0 at every logged update | the constraint never acted, which contradicts R0d (λ > 0 on 941 / 1,000 updates); a mechanism failure | audit `con_*` logging against the A2 harness on the run's own dumps; no new arm before that |
| **O7 OTHER KILL** | K2 (non-finite loss), K4 or K6; or g3 alone (addendum R2) | K2 at T = 0.01 implicates the 1/T gradient; g3 alone means TTC was bought with short plans | K2 → **R0e-pick at T = 0.02** (A2R2's T\*, half the 1/T scale). g3 → g3's quantity as a third constrained term. Both PI |
| **P-O5 DUAL SATURATED / LAGGING** | breached AND (λ reached λ_max, OR the in-sample window mean g > 0.001) | the dual could not hold even its own quantity | the PID dual or projected-gradient constraint (SAFEOBJ O5). CPU unit proof first; arm PI |
| **P-O4-PICK HELD IN-SAMPLE, VAL BREACHED** | breached AND g held (≤ 0.001 per term) AND λ > 0 on ≥ 50 % of post-ramp updates AND the **in-run wedge \|D1\| ≤ 0.10 points** | the pick held in-sample and did not transfer: a genuine generalisation gap for a **constrained** policy, which SO-CARVE could not measure on R0 states | **R0e-carve**: the dual fed from a log-disjoint carve of RL-train, never RL-val, at T = 0.01. Blocked on the carve build (§2.3; 0 GPU, ~1 day ESTIMATED; started in parallel) and the PI. **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |
| **P-O4-WEDGE HELD, PICK NOT ALIGNED IN-RUN** | breached AND g held AND **\|D1\| > 0.10** | T = 0.01 aligned on R0 states but not along the constrained trajectory: the softmax still is not the pick | a **hard-pick constraint**: a hinge on the pick's TTC through a straight-through or sampled estimator (SO-CARVE's named fallback). CPU unit proof with an identity at T → 0 first; arm PI. **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |
| O4+O5 / SLACK | as addendum R2 | as addendum R2 | both first steps (0 GPU), then the PI |
| **O1 SAFE + ABOVE FRONTIER** | S and G | the pick-aligned dual repairs the trade at a gain above the free frontier | **Stage 2**: R1e-pick, then R1o, then bar L, then navtest E plus the I-rep. ADOPT puts the constrained EPO at T = 0.01 into P6's scorer loss. **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |
| **O2 SAFE, BELOW FRONTIER** | S holds; PDMS lower bound > 0; point < G(t) | the pick-aligned dual over-pays | if the S margin (min over checks of TTC point + 0.5) ≥ 0.15: **R0e-pick at T = 0.02**. That is the lighter dual, which understates the pick by about 0.1 more at u1000 (A2R2): it spends measured margin on gain. Otherwise there is no margin to spend: SAFEOBJ O2(ii), the fixed λ at this run's measured mean λ over u250–u1000. Both PI. **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |
| **O3-a SAFE, FLAT: GAIN CROWDED OUT** | S holds; PDMS lower bound ≤ 0; λ > 0 on ≥ 50 % post-ramp (**the dual binds, TTC holds on RL-val, the PDMS gain vanishes**); AND **D2 J-crowding**: R0e-pick's in-sample `epo_J` mean over u751–1000 has risen from its u25 row by less than half of R0d's rise (R0d: 0.8171 → 0.8766, rise +0.0595, MEASURED, `sft.jsonl`) | the constraint's ~1/T gradient took the clipped update budget (`clip_grad_norm_ 1.0`, `:1693`); EPO never built its gain | **R0e-pick at T = 0.02**: half the gradient scale, still aligned on A2R2's committed bar. 0 GPU first: the D2 numbers. Arm PI. **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |
| **O3-b SAFE, FLAT: GAIN BUILT, NOT TRANSFERRED** | as O3-a, but D2 is not crowded | EPO built its in-sample gain and the pick-safe part of it did not reach RL-val. SO-CARVE already MEASURED EP's in-sample excess for R0 (+1.665 [+0.17, +3.32] at u1000). Re-ranking a fixed fan has no TTC-neutral gain at this representation | the **representational / proposal-side lever**: P6 trunk with hard negatives (P6-HN step 0, `8bb47ad`: expert-proximate hard negatives in 33.7 % of held-out sets; reach ESTIMATED 0.29-0.32 PDMS). The logit mixture is **not** a fallback: it was read FP-NONE on navtest. PI. **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |
| **O3 SLACK** | S holds; PDMS lower bound ≤ 0; λ > 0 on < 50 % post-ramp | S held while the dual was mostly idle: the arm is an R0-like draw, not a lever reading | report, with the seed floor beside it; no lever claim. Next: Stage 2's R1e-pick, for replication only (PI). **DDC rider (PI 1f, pre-authorised):** if D5 reads DDC separated adverse again (B − own A, 95 % CI entirely below 0, as R0d's −0.92 [−1.45, −0.48]), **g5 + DDC as a third constrained term is the committed next lever**, in addition to the lever above. |

* **R1e-pick** uses the same table.
  * R0e-pick O1 with R1e-pick not O1 reads **NOT REPLICATED**: no lever claim, R1o does not run, and the next step is O2's.
  * R1o runs only if both e-pick arms read S PASS (`rls1_checks.py safeobj-gate --dirs …/R0e-pick …/R1e-pick`; re-point, B5).

## 7. Cost, ESTIMATED from R0d's banked wall clock

**Basis, MEASURED in `PK/raw/2026-10-18-r0d-outcome/raw/sft.jsonl`:**

| event | time |
|---|---|
| data event | 22:38:46 |
| start | 22:38:53 |
| step0 eval done | 22:46:33 (460 s) |
| each 25-update row | 242-243 s (e.g. 22:50:38 → 22:54:41), i.e. **9.70 s / update** |
| each of the 5 evals | 460-461 s (step0, u250, u500, u750, u1000) |
| done | 01:59:01 |

* **Wall clock:** 12,015 s = **3.34 h**.
* The trainer's own `hours` field reads 3.208, i.e. 11,548 s. It counts from after the step0 eval (22:46:33 → 01:59:01).
* **Model:** wall(u) = 467 s + u × 9.70 s + (number of checks ≤ u) × 460 s. At u = 1000: 467 + 9,700 + 1,840 = 12,007 s ✓.
* **Gate + preflight per launch: 0.34 h.**
  * MEASURED for C0: `PK/raw/2026-10-14-rls1-r0-kill/rls1_chain.log`, labels banked 02:13:22 → preflight + gate PASS 02:30:11 → trainer pid 02:33:31.
  * R0d's own is not banked, so this is ESTIMATED for R0e and UNVERIFIED. The suite now adds `test_rl_safeobj.py` (147-310 s on the dev box) and `test_rl_pickcon.py` (39 s).
* **The constraint at T = 0.01** runs the same two log-softmaxes as at 0.1. ESTIMATED overhead change 0; the preflight speed probe measures it.

| item | arithmetic | A40-h |
|---|---|---|
| R0e-id to u1000 (PI 1c) | 0.34 + 12,007 / 3600 | **3.68** |
| R0e-pick to u1000 | 0.34 + 12,007 / 3600 | **3.68** |
| R0e-pick K3-killed at u500 | 0.34 + (467 + 4,850 + 920) / 3600 | 2.07 |
| R0e-pick K3-killed at u750 | 0.34 + (467 + 7,275 + 1,380) / 3600 | 2.87 |
| **Stage 1** (R0e-id + R0e-pick) | 3.68 + 3.68 | **≤ 7.36** (5.75 if R0e-pick is killed at u500: 3.68 + 2.07) |
| Stage 2 (R1e-pick + R1o) | 2 × 3.68 | ≤ 7.36 |
| **maximum** | 7.36 + 7.36 | **14.72** |
| adopted (PI 1c), included above: R0e-id run to u1000 instead of the draft's u250 stop, a full same-seed replicate of R0d at all four checks | +3.68 − 1.27 | +2.41 |

* The totals are sums of the rounded rows, as in the draft. Unrounded: Stage 1 7.35, maximum 14.70.
* Not in these totals:
  * The I-rep. INHERITED: PLAN_IREP proposes the A40, about 5 h of mostly idle GPU.
  * The navtest E reads (PLAN_E). Both run only after Stage 2.
* K6 stays at 6 h per arm.

## 8. Diagnostics (pre-registered; D1 and D2 decide rows in §6, the rest decide nothing)

Logged every 25 updates by the trainer already (patched `:1719-1727`): λ_TTC / λ_NC, `con_g_*`, `con_cA_*` / `con_cB_*`, `con_p1_gap_*`, `con_hinge_*`, cap hits, `epo_J`.

* **D1, in-run wedge** = the mean over rows of 100 × (`con_g_ttc` + `con_p1_gap_ttc`).
  * Window: post-ramp, through the first breached check, or through u1000.
  * Aligned ⇔ |D1| ≤ 0.10 points.
  * R0d read −0.317 at T = 0.1 (MEASURED, §1.1).
  * The mean ± se of the rows is reported. ⚠️ The rows are noisy (R0d's p1gap se is 0.069 over 31 rows), so a D1 within 1 se of 0.10 is flagged **BORDERLINE** beside its row.
* **D2, J-crowding**: defined in §6 row O3-a.
* **D3, the constrained-policy transfer gap** (0 GPU, after the run). At each check:
  * val L_T at T = 0.01, from the arm's own `scores_{tag}_{A,B}.npz` with the A2 harness;
  * against the in-sample `con_g_ttc` window mean;
  * and the same for the pick (val P1 against in-sample p1gap).
  * This is the measurement SO-CARVE §9 says only an R0e run can provide. It sizes the carve's job for row P-O4-PICK.
* **D4, NC alignment at T = 0.01.** A pre-launch, 0-GPU read on banked R0 states with `so_a2r2_tieaware.py`, column 0.
  * A2R2 was TTC-only. The single T also moves the NC term.
  * Descriptive. NC never breached in R0d (u1000 −0.233 [−0.532, +0.033]).
* **D5, the DDC watch.** R0d's DDC was −0.92 [−1.45, −0.48], separated adverse (MEASURED, `rlval_B_vs_ownA.json`). It is not a g5 term, so the constraint does not touch it. Reported at every check.

## 9. The four metric families (per family, never pooled; inherited from PREREG_SAFEOBJ §8)

* **LONGITUDINAL.**
  * RL-val: TTC and EP; the g3 path ratio and shortest-bin share; D1-D3.
  * Navtest: the RR3 component table.
  * Target-speed and headway / time-gap / TTC-to-lead: **NOT COMPUTED** on RL-val (`rl_val_eval` has no plan-vs-reference trace, n = 0).
* **LATERAL.**
  * RL-val: **NOT COMPUTED** (no lateral instrument, n = 0). DAC and DDC are reported as NAVSIM sub-scores, not lateral metrics.
  * Navtest: heading, curvature, yaw-rate and cross-track through the RR3 reader. ⚠️ It is UNVERIFIED that this reader accepts RL-S1 scorer checkpoints (SAFEOBJ IMPL step 9): a work item before E.
* **TACTICAL.**
  * RL-val: lane-change rate, precision and recall vs the label base rate (`rls1_checks.lane_change_stats`). K5 is watched.
  * Navtest: lateral-decision kappa, which may not fall by > 0.03 vs the control.
* **STRATEGIC: PAUSED** under the PI ruling of 2026-09-19 (PI_DECISION_QUEUE item 25). Reported as paused: not a pass, and not omitted.

## 10. Launch prerequisites (0 GPU; each is a launch blocker, not a footnote)

| # | item | owner |
|---|---|---|
| B1 | SO-PICKCON landed (trainer `869b1862…`, gate, `test_rl_pickcon.py`); `PK/SHIP_MANIFEST.json` regenerated | Master Mind (in progress) |
| B2 | the profile registers R0e-id (0.1), R0e-pick (0.01), R1e-pick (0.01, seed 1) and R1o, with must-name lists including `--epo-con-temp`; `eval/rls1e_arms.json` with the §2.1 self-check | build agent |
| B3 | `safeobj_outcome.py` per-arm registered constraint T, plus its mutation tests (§4) | build agent |
| B4 | **DROPPED (PI 1c, 2026-10-18):** R0e-id runs to u1000, so no supervisor stop-after-u250-dump is built | none |
| B5 | `safeobj-gate --dirs` re-pointed to R0e-pick / R1e-pick for R1o | build agent |
| B6 | the D1-D3 reader, in the outcome package | build agent |
| B7 | the D4 read (NC alignment at 0.01) | Research Lab, CPU |
| B8 | parallel, not blocking R0e-pick: the carve-feed build (§2.3) for row P-O4-PICK | build agent |

## 11. What this does NOT claim

* **No lever effect from one seed.** R0e-pick vs R0d is a one-pair, single-seed contrast. It is quoted only beside R0e-id's same-run floor and, after Stage 2, the seed floor. The paired bootstrap answers *another draw of logs* and nothing about training or inference variance.
* **No claim that 0.01 is the best T.** There is no sweep. A2R2 is post-hoc-motivated, measured on **unconstrained** R0 states, with all four aligned CIs spanning 0. A2R, the pre-registered read, is UNDERPOWERED. SO-CARVE A2 is INVALID.
* **No claim that the carve is unnecessary.** A1's "no train-to-val gap" is about R0's quantity at T = 0.1, unconstrained. A constrained policy's gap is measured here for the first time (D3).
* **No driving claim.** RL-val is a selection tier. Navtest T1 is an open-loop pseudo-simulation. No T2 / closed-loop read exists.
* **Nothing about DDC, NC alignment at 0.01, or lateral kinematics on RL-val.** These are flagged (D4, D5, §9), not covered by any bar.
* **Not a deployment decision.** The vs-BASE DEPLOYABLE-SAFE line beside E is reported; deploying is the PI's call.

## 12. Disclosures (numbers seen before writing)

The following were all read while choosing T, the feed and the diagnostics' thresholds:
* everything quoted in §1 (SO-CARVE A1/A2, A2R, A2R2 and its τ-sensitivity);
* R0d's outcome and its `sft.jsonl` (in-sample g, p1gap, epo_J, timings);
* FREEPOINT.

**None is an R0e number.**
* D1's 0.10 bar was chosen with R0d's −0.317 in view. D2's "half of R0d's rise" was chosen with R0d's J trajectory in view.
* RL-val is the hyper-parameter tier. T was chosen on RL-val-adjacent reads, which is why ADOPT is decided on navtest (E).
