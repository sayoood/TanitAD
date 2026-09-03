# RESULT — refcv3 epoch conclusions from its own metrics, and what a fair H-vs-F comparison requires

**Owner:** Benchmarks & Evals FlyWheel · **date:** 2026-09-03 · **compute:** 0 GPU, dev box, stdlib
only · **pre-registration:** `SPEC.md` (written before any number below) ·
**tier of every refcv3 number in this file: T0.**

> ⛔ **ESCALATE INTEGRATION (three items, none of which can be fixed by a note in a doc):**
> 1. **`taniteval/tools/refcv3_metrics_read.py` is new and needs to be picked up by whoever reads
>    this run again** — the file it reads is not a time series and a naive read of it is wrong in
>    three separate ways (§2).
> 2. **`t1_eval.DEFAULT_TIERS` still lacks `"ha0": "T1"`** (`taniteval/tools/t1_eval.py:145`).
>    Until that one line lands, every `ha0` run needs `--tiers ha0=T1` or it aborts with
>    *"arms ['ha0'] carry no T0/T1 tier stamp"*. BACKLOG R13, one line, outside this package's
>    ownership.
> 3. **refcv3's T1 instrument does not exist**, and the rescued draft carries the
>    steer-as-curvature defect **in both copies of the file** (`C-REFCV3-ARM-SAME-DEFECT`,
>    BACKLOG R20). `T1_CHECKLIST.md` in this package is the runbook; it is not a substitute for the
>    adapter, and the checklist says so at its first gate.

---

## 1. Headline

**Task 1 — the honest read.** refcv3's in-training eval is a **T0 loss monitor on 160 FIXED
held-out windows**, and it holds up as one: the window set is provably constant across all 55 evals
and all 10 launches. Segmented at 16,500 (nav switch) and 17,500 (eval-prior-leak gate), the run
reads:

* **Era A (≤ 16,500, contaminated twice over)** — every column falls hard, but **nothing in it is
  quotable as a rate**: only 3 of 16 columns clear R² 0.80, and the era spans a nav source the model
  no longer uses plus an eval that was feeding the held-out label marginals back into the model.
* **Era C (≥ 18,000, the only clean era, n = 15 evals over 7,500 steps)** — **the headline columns
  are FLAT at this instrument's resolution.** `eval_lat_tac` (0.36×), `eval_lon_tac` (0.41×),
  `eval_goal2s_err_m` (0.16×), `eval_anchor_acc` (0.22×) and `eval_loss` (0.87×) all move between
  the era's halves by **less than 1× the series' own step-to-step scatter**. Only three columns move
  clearly: `eval_goal_gate` **+14.5×** scatter (the strategic gate is still opening, R² 0.987),
  `eval_law` −3.7× (R² 0.829) and `eval_goal_score_absmean` −3.7× (R² 0.876).
* **The train–eval gap flips sign** across the boundary — `loss` −0.886 in era A to **+0.238** in
  era C, `goal2s_err_m` −0.020 to **+0.311**, `lat_tac` +0.113 to **+0.155** — and the flip
  **survives** the pre-registered robustness pass that drops the post-resume rows.
* **The run is healthy and the fixes worked.** 0 `eval_error` rows in 614; the last death was at
  17,000; the six deaths + three switches cost **2,800 recomputed steps (≈ 3.7 h)** plus ~0.6 h of
  relaunch startup; uint8 moved the pace **4.698 → 4.069 s/step (−13.4 %)**, measured over 143 rows.
* **Projected epoch end: 2026-09-03 22:43Z** (ESTIMATED, from the current launch's own rate; the
  last row in the pull is step 25,650 at 06:11:15Z).

**⭐ The two findings that matter more than the trends** (both MEASURED from
`refc_v3_train.py`, and both mean a number already in the register is being read as something it is
not):

1. **`eval_traj` is an ORACLE-ANCHOR-SELECTED error.** `a_star = dist.argmin(dim=1)` is the anchor
   **nearest the ground-truth trajectory** (`refc_v3_train.py:457–459`), and
   `recon = out["anchor_traj"][ar, a_star]` (`:463`) is what `loss_traj` scores. The model's *own*
   selection is scored separately (`loss_cls`, `anchor_acc` — it agrees with the oracle on **57 %**
   of windows in era C). ⇒ `eval_traj` is a **lower bound** on the trajectory error refcv3 would
   actually drive, and the bound is loose. `D-REFCV3-19K` / `D-REFCV3-21K` call 0.999 / 1.062 *"the
   run's lowest traj"*; that is true and correct, and it is the lowest **oracle-selected** traj.
2. **`eval_traj` is not an ADE.** `loss_traj` is `|Δ|.sum(-1)` divided by `sv.sum()*2` (`:462–465`)
   — a **mean absolute error per COORDINATE**, in metres, over the 8 horizon slots 0.5–6.0 s. An ADE
   is an L2 norm. ⛔ The two are different statistics and must never be put in the same column as
   refav1's or the flagship's ADE.

**Task 3 — comparability (§5).** refav1 and refcv3 share the 4,572-clip train split and the
141-clip eval split, and **that is where the shared ground ends.** refcv3 has **no
action-conditioned rollout at all** — it emits the whole 6 s path in one forward pass — so
`t1_eval.roll_closed`'s definition of "closed loop" (feed the head's own action back into the
predictor) **cannot be ported to it**. A fair H-vs-F comparison exists, but it is narrower than
anyone has assumed: **`ha0` and `ol`-as-kinematic-contract are common; `cl` is not the same object
on both sides**; and four named conditions would make it **INADMISSIBLE**.

---

## 2. How the file was read — and the three ways a naive read of it is wrong

`metrics.jsonl` is one append-only file across ten launches. Read in row order it is **not a time
series**:

| trap | what it does to a naive read | the fix used here |
|---|---|---|
| **Replayed steps.** A resume replays the 500 steps since the last checkpoint. | Step order runs **backwards** at every boundary; **56 rows are duplicates**. | canonical = **last-writer-wins per (step, kind)**; the earlier copies are counted as the recomputation cost. |
| **`elapsed_s` is cumulative SINCE THE CURRENT LAUNCH.** | Diffing across a boundary gives a negative time; summing gives a wrong total. | launch boundary := an `elapsed_s` **decrease** on a train row; pace computed **within** a launch. |
| **Three different measurements in one column.** | A single "eval_loss trend" pools a run whose nav source changed at 16,500 and whose eval was leaking the held-out marginals until 17,500. | segmentation at 16,500 / 17,500; **no fit crosses a boundary**. |

⚠️ **`elapsed_s` is NOT the `step_s` field the CLAUDE.md divisor rule is about.** That rule (÷
`--log-every`, and its inversion for `train_v6_staged.py`) does not apply to this trainer, and
applying either version here is wrong. `elapsed_s` is `time.time() - t0`, so the per-step rate is
`Δelapsed_s / Δstep`, full stop.

### 2.1 Reconstruction (R1) — and the one place it exceeds `supervisor.log`

Ten launches recovered from `elapsed_s`; **every one is corroborated** by `supervisor.log` where the
two overlap, and one is not in the log at all:

| launch | first row | resumed from ckpt | cause | `supervisor.log` line | replayed steps |
|---|---|---|---|---|---|
| L0 | 550 | 500 | **not in this file** (see below) | — | 0 |
| L1 | 1,050 | 1,000 | death @ 1,250 (CUDA OOM w/ traceback — different class) | **absent** — the log starts 20:53:01Z | 250 |
| L2 | 1,550 | 1,500 | death @ 2,000 | `relaunch #1 from step 2000` | 500 |
| L3 | 4,050 | 4,000 | death @ 4,500 | `relaunch #2 from step 4500` | 500 |
| L4 | 5,550 | 5,500 | death @ 6,000 | `relaunch #3 from step 6000` | 500 |
| L5 | 10,050 | 10,000 | death @ 10,500 | `relaunch #1 from step 10500` | 500 |
| L6 | 16,550 | 16,500 | death @ 17,000 | `relaunch #2 from step 17000` (19:19:48Z) | 50 |
| L7 | 16,550 | 16,500 | **SWITCH** — v7.2 nav (`D-REFCV3-NAV-SWITCHED`) | supervisor v2, 19:25:36Z | 500 |
| L8 | 17,550 | 17,500 | **SWITCH** — save-before-eval + leak gate (`D-REFCV3-SAVE-BEFORE-EVAL`) | `relaunch #2 from step 17500` | 0 |
| L9 | 18,550 | 18,500 | **SWITCH** — uint8 in-flight batches (`D-REFCV3-U8-SHIPPED`) | supervisor v3, 22:06:29Z | 0 |

⚠️ **Scope limit, stated rather than papered over:** this `metrics.jsonl` **begins at step 550**,
with the same ~243 s startup signature every other resume shows — i.e. **the file begins AT a
launch**, and steps 0–500 are not in it. Everything in this package covers **550 – 25,650**. The
"loss 121.7 → 16.6" figures in `D-REFCV3-THROUGHPUT` come from an earlier file that was not pulled;
they are not re-verified here.

⚠️ **A limitation of my own second probe (§4.6):** `supervisor.log`'s three *switch* lines record the
**resume** step, while its *death* lines record the **death** step. The naive rule
"resume = from_step − 500" is therefore correct for the four death lines and wrong for the switch
lines; the two rows of `pace.probe_b` that read 0.696 and 3.173 s/step are that artifact, not a
measurement, and are excluded from the cross-check.

### 2.2 The enabling check (R2) — the eval window set really is FIXED

`refc_v3_train.py:1037–1041` draws the eval subset **once**, with
`torch.randperm(..., Generator().manual_seed(12345))[:160]`, and iterates it in a fixed order. That
is a claim about code; here is the claim about the data — **all 55 eval rows, all 10 launches**:

| invariant | distinct values | value |
|---|---|---|
| `eval_windows` | **1** | 160 |
| `eval_batches` | **1** | 8 |
| `eval_slot_valid_frac` | **1** | 0.91953 |
| `eval_tac_label_rows` | **1** | 4.875 |
| `eval_tac_label_v7` | **1** | 1.0 |
| `eval_nav_injected` | **1** | 1.0 |

⇒ **R2 holds.** The same 160 windows at every step, across every relaunch and every file swap, so
the series is step-to-step comparable and the only confounds are the two in §3. (Had any of these
varied, every cross-step comparison in this document would be withdrawn — this is the load-bearing
check, and the last two rows are also the controls that must read a known value: the v7.2 tactical
labels and the nav injection reached the eval on **every** step in **every** era.)

### 2.3 What each column actually is (read from source before any value)

| column | definition (`refc_v3_train.py`) | no-information value | landmine |
|---|---|---|---|
| `traj` | mean **L1 per coordinate**, metres, over 8 slots 0.5–6.0 s, of the **GT-nearest anchor's** refinement (`:457–465`) | — | ⛔ **not an ADE**; ⛔ **oracle-selected** |
| `cls` | CE over the **128-anchor** vocabulary vs `a_star` (`refc_v3.py:_v3_core_base`, `n_anchors=128`) | **ln 128 = 4.852** | this is the model's own selection skill |
| `anchor_acc` | fraction where `anchor_logits.argmax == a_star` | **1/128 = 0.0078** | era C ≈ 0.57 = **73× chance** |
| `lat_tac` / `lon_tac` | CE of the factored **8-class** tactical heads (`tac_vocab_version v7.0`) | **ln 8 = 2.0794** | the columns the prior leak touched directly |
| `lat` / `lon` | CE of the 3-class kinematic heads | ln 3 = 1.0986 | `eval_lon` sits at 0.73, i.e. **only 0.37 nats below chance** |
| `route` | CE of the route head vs the nav target | ln 3 = 1.0986 | ⚠️ nav is an **input**; a route head can echo it (`CLAUDE.md` nav-echo, flagship v1 scored 1.0000) |
| `law` | MSE of a **0.5 s-ahead pooled-latent** prediction vs a `no_grad` encode of the future frame (`:578–582`) | — | the one column that consumes future **pixels** — a WM-fidelity aux |
| `goal_tac` / `goal2s_err_m` | tactical goal loss / **L2** metres at the 2 s slot (`:600–606`) | — | `goal2s_err_m` *is* a metric error; `traj` is not |
| `sel_v3` | survivor-set CE on the blended score (`:617–623`) | — | the gates' only gradient |
| `goal_gate` / `goal_score_absmean` / `goal_gate_grad` | the zero-init strategic gate, the score scale it multiplies, its gradient | — | the PI's CAVEAT-B telemetry |

---

## 3. Task 1 — the segmented read

Segmentation, and why it is not negotiable:

| era | eval steps | n | nav source | eval prior leak | status |
|---|---|---|---|---|---|
| **A** | 600 – 16,500 | 38 | refb-derived (`follow` on 94.6 % of windows) | **PRESENT** — partly self-referential | ⛔ contaminated |
| **B** | 17,000 – 17,500 | 2 | v7.2 token | **PRESENT** | transition only, never a trend |
| **C** | 18,000 – 25,500 | 15 | v7.2 token | **GATED** | the only clean era |

The gate shipped **at** 17,500 (20:46:55Z) and L8 resumed from the 17,500 checkpoint, so the 17,500
eval was produced by the **old, leaking** trainer. **The first clean eval is 18,000.** There is no
eval at 18,500 — by design, the uint8 switcher stopped the run right after that checkpoint.

⛔ **No fit in this document crosses 16,500 or 17,500.**

### 3.1 Era A (≤ 16,500) — large falls, almost nothing quotable as a rate

n = 38, window 600–16,500. Slope per 1,000 steps with R²; **CLAUDE.md bar: below R² 0.80 there is no
quotable rate**, so the `quotable` column is the verdict, not decoration.

| column | first | last | slope/1k | R² | quotable rate? |
|---|---|---|---|---|---|
| `eval_loss` | 19.089 | 9.304 | −0.610 | 0.561 | **no** |
| `eval_traj` | 2.432 | 1.169 | −0.0661 | 0.658 | **no** |
| `eval_lat_tac` | 1.443 | 1.070 | −0.0233 | **0.828** | yes |
| `eval_lon_tac` | 1.671 | 1.215 | −0.0245 | 0.611 | **no** |
| `eval_lat` | 0.442 | 0.127 | −0.0241 | **0.822** | yes |
| `eval_lon` | 0.879 | 0.752 | −0.0067 | 0.588 | **no** |
| `eval_route` | 0.908 | 0.704 | −0.0104 | 0.591 | **no** |
| `eval_law` | 0.210 | 0.064 | −0.0161 | 0.771 | **no** |
| `eval_goal_tac` | 21.838 | 8.279 | −0.927 | 0.598 | **no** |
| `eval_goal2s_err_m` | 8.441 | 2.737 | −0.389 | 0.582 | **no** |
| `eval_anchor_acc` | 0.175 | 0.538 | +0.0117 | 0.384 | **no** |
| `eval_goal_gate` | 0.0004 | 0.0660 | +0.00431 | **0.989** | yes |
| `eval_goal_score_absmean` | 25.879 | 9.994 | −0.868 | 0.754 | **no** |

**The reading.** The direction is unambiguous and large — everything improves, `eval_goal2s_err_m`
by 3.1× and `eval_anchor_acc` from 0.175 to 0.538 — but **3 of 16 columns clear the R² bar**, so era
A supports *"it learned a lot"* and supports **no rate at all**. And the two contaminations mean even
the direction is not attributable: `eval_lat_tac` and `eval_lon_tac` were decoded through a prior
that had been EMA'd toward **the held-out split's own class marginals** (`C-REFCV3-EVAL-PRIOR-LEAK`
puts each eval at ~7.7 % of the way toward the held-out distribution — ESTIMATED, INHERITED, not
re-verified here), which biases them **optimistically**; and the nav the model was conditioned on
for all 16,500 steps is not the nav it uses now.

⚠️ **Conflict reported, per the primary-source rule.** `C-REFCV3-EVAL-PRIOR-LEAK` says *"34 evals so
far"*. **MEASURED here: 40 leaked evals** — 38 in era A plus the two era-B points, all at steps
≤ 17,500, in `metrics.jsonl` sha256 `8207d81d…`. The register's figure was written at the time of
diagnosis (~step 17,000) and this file also **begins at step 550**, so neither count is the whole
run. The count that matters for the read is the boundary, and the boundary is unambiguous: **every
eval at step ≤ 17,500 is contaminated; 18,000 is the first clean one.**

⇒ ⛔ **Era A numbers are not a baseline for anything.** They may be quoted as *"the run was
learning"* and nothing else.

### 3.2 Era B — two points, reported as a transition, not a trend

17,000: loss 8.608 / traj 1.089 / lat_tac 1.027 · 17,500: loss 9.362 / traj 1.120 / lat_tac 1.041.
Both are **post-relaunch** evals under a nav source the model had seen for ≤ 1,000 steps. Two points
carry no slope and no interval. They are here so the boundary is visible, not to be read.

### 3.3 Era C (≥ 18,000) — ⭐ the clean era is FLAT where it matters

n = 15, window 18,000–25,500 (7,500 steps ≈ 8.5 h of A40 time). `sd_step` is the scatter implied by
**consecutive** evals; `|Δ|/sd_step` compares the half-to-half move against that scatter.

⚠️ **`|Δ|/sd_step` is a descriptive signal-to-scatter ratio. It is NOT a t-statistic and NOT a
CI** — the 15 points are autocorrelated and share one 160-window sample. It answers only *"is this
move bigger than the series' own jitter?"*

| column | H1 mean (18.0–21.5k) | H2 mean (22.0–25.5k) | Δ | sd_step | \|Δ\|/sd_step | R² of the era fit |
|---|---|---|---|---|---|---|
| `eval_goal_gate` | 0.0903 | 0.1177 | **+0.0274** | 0.0019 | **14.45** | **0.987** |
| `eval_law` | 0.0528 | 0.0438 | −0.0090 | 0.0024 | **3.73** | **0.829** |
| `eval_goal_score_absmean` | 7.721 | 5.948 | −1.773 | 0.484 | **3.66** | **0.876** |
| `eval_traj` | 1.0942 | 1.0012 | −0.0929 | 0.0547 | 1.70 | 0.514 |
| `eval_route` | 0.6811 | 0.6617 | −0.0194 | 0.0134 | 1.46 | 0.478 |
| `eval_cls` | 1.5201 | 1.4692 | −0.0509 | 0.0418 | 1.22 | 0.340 |
| `eval_loss` | 8.6276 | 8.3654 | −0.2622 | 0.3010 | 0.87 | 0.247 |
| `eval_lat` | 0.1322 | 0.1242 | −0.0080 | 0.0099 | 0.81 | 0.105 |
| `eval_lon` | 0.7418 | 0.7342 | −0.0076 | 0.0130 | 0.58 | 0.082 |
| `eval_sel_v3` | 2.1318 | 2.0802 | −0.0516 | 0.0952 | 0.54 | 0.251 |
| `eval_lon_tac` | 1.1946 | 1.1835 | −0.0111 | 0.0268 | 0.41 | 0.281 |
| `eval_lat_tac` | 0.9752 | 0.9606 | −0.0147 | 0.0408 | 0.36 | 0.043 |
| `eval_goal_tac` | 7.2696 | 7.1533 | −0.1164 | 0.4074 | 0.29 | 0.041 |
| `eval_anchor_acc` | 0.5652 | 0.5727 | +0.0075 | 0.0338 | 0.22 | 0.039 |
| `eval_goal2s_err_m` | 2.2882 | 2.2618 | −0.0263 | 0.1689 | 0.16 | 0.029 |

**R5 held** (pre-registered: the headline columns move by < 2× scatter). **Not one** of `eval_loss`,
`eval_lat_tac`, `eval_lon_tac`, `eval_goal2s_err_m`, `eval_anchor_acc` moves as much as **one**
scatter unit over 7,500 clean steps.

**The reading, stated at the strength the data supports:**

* **On held-out windows, the tactical decision surface has stopped moving.** `eval_lat_tac` 0.967,
  `eval_lon_tac` 1.189 (chance ln 8 = 2.079); `eval_anchor_acc` 0.569 (chance 0.0078). These are
  genuine, large distances from chance **and** they are flat from 18,000 to 25,500.
* **The strategic goal path is the one thing still visibly changing.** `eval_goal_gate` rises
  0.0733 → 0.1307 with R² 0.987, while the score scale it multiplies falls 8.73 → 4.75 (R² 0.876).
  This is the PI's CAVEAT-B question answered with data: the gate is **opening, not stuck** — but
  the gate opening is a statement about the *architecture's* use of the path, **not** evidence that
  the strategic signal helps: `eval_goal2s_err_m` and `eval_goal_tac` are the flattest columns in
  the table.
* **⚠️ The direction of `eval_traj` (−0.093, 1.70× scatter, R² 0.514) is suggestive and not
  established.** It is also the oracle-selected number (§2.3), so even if it is real it is not the
  trajectory refcv3 would drive.
* ⛔ **What this does NOT support:** *"refcv3 is still improving"* as a general claim. On held-out
  data, over the clean era, four of the five headline columns are flat, and the fifth
  (`eval_traj`) does not clear the bar.

### 3.4 The train–eval gap flips sign, and the flip survives its robustness pass

`gap = eval_X − mean(train X over (N−500, N])`. **The level is not interpretable** — train rows are
20-window samples of the *current* batches, the eval is a fixed 160-window set, so they are
different windows. **The trend is.**

| column | era A gap | era C gap | era A (drop 2 rows after each resume) | era C (same) |
|---|---|---|---|---|
| `loss` | **−0.886** | **+0.238** | −0.840 | **+0.282** |
| `lat_tac` | +0.113 | **+0.155** | +0.088 | **+0.141** |
| `goal2s_err_m` | −0.020 | **+0.311** | +0.003 | **+0.334** |
| `route` | +0.049 | +0.088 | +0.045 | +0.082 |
| `traj` | −0.160 | −0.101 | −0.157 | −0.096 |
| `lon_tac` | −0.060 | +0.004 | −0.060 | +0.010 |

**R7 held.** The sign change on `loss` and `goal2s_err_m`, and the widening on `lat_tac`, all survive
dropping the first two rows of every launch — so they are **not** an artifact of the post-resume
spikes inflating the train side.

⇒ **The generalisation gap is opening exactly where the eval has gone flat.** Train `loss` keeps
falling (9.49 at 16,500 → 7.72 at 25,500 on the canonical rows) while `eval_loss` does not. This is
the ordinary shape of a model that has extracted what this data + this objective will give at this
capacity — and it is **the** argument for reading the epoch-end checkpoint at T1 rather than
extending the run.

⚠️ `traj` is the one gap that is still **negative** (eval better than train). That is expected and
uninformative here: the train side is a 20-window sample with far more variance, and both sides are
oracle-selected.

### 3.5 Post-resume spikes decay inside one log interval (R8 held)

| launch | first loss | pre-boundary median | ratio | recovered |
|---|---|---|---|---|
| L1 | 24.686 | 19.522 | 1.264 | **1 row (50 steps)** |
| L2 | 19.773 | 16.953 | 1.166 | 1 row |
| L3 | 14.734 | 14.481 | 1.017 | 1 row |
| L4 | 14.289 | 14.153 | 1.010 | 1 row |
| L5 | 14.262 | 13.572 | 1.051 | 1 row |
| L6 | 12.629 | 9.159 | 1.379 | **not measurable** — L6 lived 50 steps before the nav switch superseded it |
| L8 | 11.137 | 8.959 | 1.243 | 1 row |
| L9 | 10.859 | 9.036 | 1.202 | 1 row |

⇒ A resume **re-warms**, it does not perturb. Every measurable case is back at or below its
pre-boundary level by the next logged row. ⚠️ The `eval` at a step immediately following a resume
(17,500 is the case in this file) is a **re-warm point, not a trend point** — `D-REFCV3-19K` says the
same thing and is corroborated here.

### 3.6 Pace, measured through two different clocks (R9 held), and the epoch end

| launch | steps | s/step (probe A: in-process `elapsed_s`) | s/step (probe B: `supervisor.log` wall clock, incl. startup) | agreement |
|---|---|---|---|---|
| L2 | 1,550–4,500 | 4.762 | 4.792 | 0.6 % |
| L3 | 4,050–6,000 | 4.776 | 4.787 | 0.2 % |
| L4 | 5,550–10,500 | 4.727 | 4.751 | 0.5 % |
| L5 | 10,050–17,000 | 4.690 | 4.729 | 0.8 % |
| L7 | 16,550–17,500 | 4.727 | 4.894 | 3.4 % |
| **L9 (uint8)** | 18,550–25,650 | **4.069** (n = 143 rows, 7.97 h) | — | — |

Two **different mechanisms** (an in-process timer vs. the supervisor's wall clock), agreeing to
< 1 % on four independent spans. Relaunch startup is **235.6–243.6 s** pre-uint8 and **203.7 s** at
L9.

* **uint8 speedup: 4.698 → 4.069 s/step = −13.4 %**, measured over 143 rows / 7,100 steps. (The
  register's 3.96–4.06 from the first 100 rows after the switch is corroborated and slightly
  optimistic at the full span.)
* **Cost of the interruptions: 2,800 recomputed steps** — 2,300 from the six deaths (≈ 3.02 h at
  4.73 s/step) and 500 from the switches (≈ 0.66 h) — **plus 9 × ~237 s of startup (0.59 h)**.
  Total ≈ **4.3 h** of the run's wall clock.
* **Projected end (ESTIMATED): 2026-09-03 22:43:36Z**, from the current launch's own 4.069 s/step
  over the remaining 14,634 steps (16.54 h from the last row, 25,650 @ 06:11:15Z). Assumes no
  further death; with save-before-eval in force a death now costs ~0 steps + ~204 s.

### 3.7 ⛔ What these numbers CAN and CANNOT support

**CAN.**
* A **direction** per era, per column, with its n and fit window.
* The statement that the clean era is **flat** in the headline columns *at this instrument's
  resolution* — i.e. any real improvement over 18,000–25,500 is smaller than the eval's own
  step-to-step scatter.
* Absolute distances from **chance** (ln 8, ln 128, 1/128), which need no interval to be meaningful.
* Engineering facts: pace, startup, recomputation cost, projected end, 0 eval errors.

**CANNOT — and no estimator changes this.**
1. ⛔ **No confidence interval on any `eval_*` column, by any method.** Each value is *already* the
   pooled mean over the 160 windows. The file carries **no per-window values and no episode index**;
   `taniteval.ci.episode_cluster_bootstrap` (`ci.py:225`) and its paired form (`:275`) require both.
   The interval cannot be recovered later either — the per-window values were never written.
2. ⛔ **No comparison to any other arm.** Different objective, different heads, different statistic
   (§5).
3. ⛔ **No driving claim, at any strength.** T0, and worse than ordinary T0: `traj` is
   oracle-anchor-selected and `law` consumes future pixels. EVAL_DOCTRINE rule 2.
4. ⛔ **No attribution of the era-A→C change to the leak gate or to the nav switch.** Three things
   changed within 1,500 steps of each other (nav 16,500, gate + save-order 17,500, uint8 18,500) and
   training continued throughout. The confounds are not separable from this file, and separating
   them would need an A/B that nobody should spend GPU on.
5. ⛔ **No episode-level statement.** How many of the 141 eval clips the 160 windows touch is **not
   recoverable** from this file. It is a work item (§6, gate 3), not an estimate.

**If an interval is wanted, the job that produces one is the T1 read in `T1_CHECKLIST.md`** — a dump
with per-window values and `eid`, then `paired_episode_cluster_bootstrap`. Nothing short of that
yields a defensible interval for refcv3, and no re-reading of `metrics.jsonl` will.

---

## 4. Task 2 — the end-of-epoch T1 checklist

**→ `T1_CHECKLIST.md` in this package.** It is the runbook the Master Mind executes when the epoch
ends (≈ 22:43Z), covering: which checkpoint, which split, the five arms (`cl` T1 · `ha` T1 ·
**`ha0` T1** · `cl_navshuf` T1 · `ol` T0), the four families, the paired episode-cluster bootstrap,
and the **trivial-profile instrument that must print BEFORE any family row**. It opens with the two
defects it must carry (the steer-as-curvature replay in both copies of the rescued adapter; the
missing `"ha0": "T1"` in `DEFAULT_TIERS`) and with the gate that says **the adapter does not exist
yet** — the checklist is a runbook, not an instrument.

---

## 5. ⭐ Task 3 — what a fair H-vs-F comparison actually requires

This is the section the dominance claim (`D-V7-READINESS-2026-09-02`, register decision 9, H vs F)
depends on, and the question has not been asked before. **Both arms train on the same 4,572 clips
and hold out the same 141. That similarity is doing far more rhetorical work than it can carry.**

### 5.1 The two models are not the same kind of object

| | **refav1** (H) | **refcv3** (F) |
|---|---|---|
| what it is | latent world model + **iCEM planner** (`refa_v1_plan.py`) | **supervised one-shot anchor trajectory model** (`refc_v3.py`, 128 FPS anchors × 8 slots) |
| action-conditioned? | **yes** — the predictor takes `(a, steer)` per step | **no** — no action input, no rollout, no per-step decode |
| output | a control sequence, integrated to a path | a full 6 s path in **one forward pass** |
| selection | the search's own argmin over candidates | `anchor_logits` / `sel_score_v3` over 128 anchors |
| what "closed loop" can mean | feed the **planner's own** actions back into the predictor (`t1_eval.roll_closed`) | **nothing to close** — there is no action to feed back |

**How `t1_eval.roll_closed` handles the flagship — which is also supervised — and why that does not
rescue refcv3.** For the flagship, `roll_closed` (`t1_eval.py:753–786`) is: at each of K steps the
predictor emits `z_hat`; a `UnicycleStepReadout` **head** decodes `(a_j, yaw_j)` from
`(z_prev, z_hat, v, a_prev, yaw_prev)`; the path accumulates from those; and the action fed to the
**next** predictor step is `(steer = atan(L·κ), a_j)` with the ego channel holding `v0`. The
perception context is fixed at t0 — the state window slides by appending `z_hat`, never a re-encoded
frame. Its T0 twin `decode_open` (`:731`) runs the **same head** over transitions rolled with the
**true future actions**; `ha` runs it over transitions rolled with the **t0 action held**.

⇒ **The flagship is "supervised" but it is still an autoregressive, action-conditioned predictor
with a per-step readout.** That is what makes `cl` / `ol` / `ha` three genuinely different arms for
it. **refcv3 has none of that machinery.** Porting `roll_closed` to refcv3 is not a small engineering
job — there is no object to port it to. This is precisely the "T1 definition for a supervised
trajectory model, derived with file:line" that the killed agent never wrote
(`C-REFCV3-ARM-SAME-DEFECT`), and it must be written **before** any refcv3 arm is named `cl`.

### 5.2 Which arms are common, which are not

| arm | refav1 | refcv3 | comparable? |
|---|---|---|---|
| **`ha0`** constant velocity (a = 0, κ = 0 at measured `v0`) | model-free | model-free | ⭐ **YES — identically defined, identical windows.** The only *bit-for-bit* shared arm, and the one both must be beaten against. |
| **`ha`** hold the last observed (a, κ) | model-free | model-free | **YES**, same caveat: `ha` is a control, not a floor — it can be *worse* than trivial (`refav1_arm.py:404–409`). |
| **`cl_navshuf`** nav permuted across windows | `cl` with nav permuted | whatever refcv3's deployed arm is, with nav permuted | **YES as a Δ-within-arm**; ⛔ **NO as a cross-arm level** — it inherits its own arm's definition. |
| **`ol`** | the recorded (a, κ) integrated from `v0` — for refav1 this is the **kinematic-contract control**, not a WM diagnostic | refcv3 consumes no actions, so this arm **does not exist** for it | ⛔ **NO.** Same name, different object. |
| **`cl`** | the planner's own actions closed back through the predictor | a single forward pass with the model's **own** anchor selection (must be **defined**, then built) | ⛔ **NOT YET** — and never as the *same* arm; see 5.3. |

### 5.3 Which metrics are defined for both — and the two that are not

**Defined for both, on the same windows, once refcv3 emits a per-window predicted path:** all four
binding families. They are computed from a **path**, not from a model internals — `four_families`
consumes `win["lead"]`, `speeds`, `eid` and the predicted trajectory
(`2026-09-02-b1-eval-lead-block/RESULT.md` §1a). LONGITUDINAL (target speed + headway/time-gap/TTC
against the banked 147-clip lead block), LATERAL (heading, curvature, yaw-rate, cross-track),
TACTICAL (selected vs executed manoeuvre + goal/anchor selection), STRATEGIC (route/goal setting)
are all path-derived or head-derived and portable.

⛔ **NOT defined for both, and the reason each fails:**

1. **`eval_traj` vs ADE.** refcv3's `traj` is a **mean L1 per coordinate**; ADE is an L2 norm over
   the same slots. Putting them in one column is a category error even before the oracle problem.
   *Fix:* recompute an L2 ADE from refcv3's dumped path — trivial, but it must be **recomputed**,
   never converted.
2. **Anything through refcv3's `a_star`.** `traj`, and `cls`/`anchor_acc` as *quality* numbers,
   are conditioned on the **ground-truth-nearest anchor**. refav1's `cl` has no oracle in it at
   all. *Fix:* refcv3's comparable arm must select with `sel_score_v3` / `anchor_logits`, and the
   oracle-selected number must be reported **beside** it as the ceiling it is — an `oracle_sel`
   arm, tiered **T0**, exactly as `cl_oraclegoal` is T0 for refav1.
3. **`law`.** It consumes a **future frame's** pooled latent. There is no refav1 analogue and it can
   never enter a T1 row. It stays a T0 WM-fidelity diagnostic.
4. **`route`.** ⚠️ Nav is an **input** to both arms. A route head that reproduces its own input
   scores well and has learned nothing — the flagship v1 route head was an exact bijection of the
   fed nav and scored **1.0000** (`CLAUDE.md`, nav-echo). ⇒ the STRATEGIC family for **either** arm
   needs the `cl_navshuf` control beside it, and a route number without it is inadmissible.

### 5.4 What "closed loop" means for each — the sentence that has to be in the report

* **refav1:** the *planner* proposes actions, the *predictor* imagines their consequence, the
  chosen actions are fed back, perception frozen at t0. "Closed" = the **action loop is closed
  through the model's own decisions**.
* **refcv3:** there is **no loop**. Its single forward pass is already free of future information
  **provided** its inputs are frames ≤ t0, `nav_cmd` and the measured `v0` — and **provided** the
  anchor is chosen by the model, not by `a_star`. The honest name for that arm is **not `cl`**; it
  is a **one-shot planning-free trajectory prediction at t0**, and whether the doctrine admits it as
  T1 is a **PI/Master-Mind ruling**, not something a FlyWheel may assume. The doctrine's T1 text
  ("the predictor consumes the decoder/planner's own actions") does not literally cover a model that
  consumes no actions at all.

  ⚠️ **My reading, offered as a recommendation and flagged as such:** it should be admitted as T1,
  because the doctrine's *purpose* is to forbid future information from reaching inference, and a
  correctly-gated refcv3 forward pass admits none. But it must be stamped with a **distinct arm
  name** (`os` / `oneshot`, not `cl`) so no reader ever believes the two arms' `cl` columns describe
  the same procedure. **This is decision 9's real blocker, and it is a definition, not compute.**

### 5.5 What would make an H-vs-F comparison INADMISSIBLE

Any one of these voids it:

1. ⛔ **Different window grids.** Both arms must be scored on the **same** window list of the same
   141 clips, with the grid asserted equal (as `D-REFAV1-PAIRED-READ-VOID`'s second read did) —
   otherwise the paired bootstrap has nothing to pair and the unpaired one is far weaker.
2. ⛔ **Quoting refcv3's oracle-selected `traj` against refav1's `cl`.** A model given the right
   mode and asked only to refine it is being scored on a different task. *(This is the C6-confound
   family: something the eval measures also feeds the thing being measured.)*
3. ⛔ **Comparing across tiers, or comparing `cl`-to-`cl` when the two `cl`s are different
   procedures** (5.4). EVAL_DOCTRINE: "comparisons across tiers are invalid" — and two arms sharing
   a *name* is not the same as sharing a tier.
4. ⛔ **Any arm whose trivial profile is degenerate.** If either arm's plan is the constant-velocity
   line on ~all windows, the comparison is **VOID, not negative** — `D-REFAV1-PAIRED-READ-VOID` cost
   a 2.5 h rollout learning this once. The trivial-profile instrument prints **before** any family
   row, and a read whose arms are bit-identical to another arm is stamped VOID.
5. ⛔ **A refcv3 arm evaluated through the rescued adapter's `recorded_controls` without the steer
   fix.** It replays a **steer angle** as a curvature (`C-REFCV3-ARM-SAME-DEFECT`); on refav1's eval
   slice the identical defect cost **0.716 m** of lateral error against a **0.053 m** floor. A
   refcv3 number carrying that defect would not be merely noisy, it would be **wrong in a direction
   that flatters or damns an arm depending on how much it turns**.
6. ⚠️ **A comparison with no shared floor.** Both arms must be reported against **`ha0` on the same
   windows**, with `cl − ha0` (or `os − ha0`) as the headline paired delta. Without it, a straight
   line can read as skill — measured, 2026-09-03.

### 5.6 The minimum admissible H-vs-F table

| | refav1 (H) | refcv3 (F) | paired stat |
|---|---|---|---|
| floor | `ha0` | `ha0` (**identical windows, identical values**) | — |
| control | `ha`, `cl_navshuf` | `ha`, `<arm>_navshuf` | Δ within arm |
| deployed | `cl` (T1) | `os` (T1, name pending the ruling) | ⛔ never `cl` vs `os` head-to-head as a level |
| **the admissible claim** | `cl − ha0` | `os − ha0` | **difference of the two deltas**, paired episode-cluster bootstrap over the shared eid set |
| per family | 4 families | 4 families | per family, never pooled |
| ceiling (T0, beside) | `cl_oraclegoal` | `oracle_sel` | reported, never compared to a T1 number |

**⇒ The dominance claim is a comparison of each arm's margin over the same trivial floor, per
family, with a paired interval — not a comparison of their raw numbers.** That is a stronger claim
than "F beats H" and it is the only one this pair of models can support.

---

## 6. Work items this read created (each a WORK ITEM, not an excuse)

| # | item | why |
|---|---|---|
| W1 | **`config.json` does not stamp the anchor-vocabulary size or provenance.** | The chance lines for `cls` / `anchor_acc` had to be read from `refc_v3.py` source (128), not from the run's own primaries. A run should be self-describing. |
| W2 | **The in-training eval writes no per-window values and no `eid`.** | It permanently forecloses any interval on 55 evals' worth of held-out measurement, for the cost of one extra array. |
| W3 | **The eval's episode composition is unknown.** | 160 windows over ≤ 141 clips; the clip count is not logged, so even the effective n is unknown. |
| W4 | **`t1_eval.DEFAULT_TIERS` lacks `"ha0": "T1"`.** | BACKLOG R13, one line. Blocks every `ha0` run that does not remember `--tiers`. |
| W5 | **refcv3 has no T1 instrument, and the rescued draft's steer defect exists in BOTH copies.** | BACKLOG R20. The blocker for decision 9. |
| W6 | **The T1-vs-one-shot tier ruling for a model that consumes no actions (§5.4).** | A definition the doctrine does not cover. PI / Master Mind, not a FlyWheel. |
| W7 | ⚠️ **MEASURED here, worth banking: `Select-String -Path <abs path on G:>` SILENTLY returned 0 hits for markers that were present**, on all four files, with no error and exit 0; `-LiteralPath` found every one. `-Path` is a *wildcard* parameter, so a path it fails to expand yields an empty result indistinguishable from "the marker is missing". ⇒ **use `-LiteralPath` for every verification probe on this mount.** Same family as the `git ls-tree` truncation and the `grep`-under-reports traps: a probe returning empty is not evidence of absence. *(Caught because two probes disagreed — which is the whole point of running a second one through a different mechanism.)* |

---

## 7. PROPOSED REGISTER ROWS

Status stays **PROPOSED** until the Master Mind/PI reads them. Apply verbatim or amend.

### `D-REFCV3-EPOCH-READ`

> ✅ **D-REFCV3-EPOCH-READ — refcv3's OWN in-training eval, segmented at the two boundaries that
> changed the measurement: the CLEAN ERA IS FLAT IN EVERY HEADLINE COLUMN WHILE THE TRAIN–EVAL GAP
> OPENS, AND `eval_traj` IS AN ORACLE-ANCHOR-SELECTED L1, NOT AN ADE (MEASURED 2026-09-03,
> Benchmarks & Evals FlyWheel, 0 GPU;
> `Benchmarks & Evals/Research/2026-09-03-refcv3-epoch-conclusions/`, metrics.jsonl sha256
> `8207d81d…`, 614 rows, steps 550–25,650).** **Tier T0 throughout** — an in-training loss monitor
> on 160 FIXED held-out windows, ⛔ never driving performance. **The window set is provably fixed:**
> `eval_windows` 160, `eval_slot_valid_frac` 0.91953, `eval_tac_label_rows` 4.875,
> `eval_tac_label_v7` 1.0, `eval_nav_injected` 1.0 — each **one distinct value across all 55 evals
> and all 10 launches**, so the series is step-to-step comparable and the two controls (v7.2 labels,
> nav injection) read their known value in every era. **Segmentation (binding):** era A ≤ 16,500
> (refb nav + the prior leak — ⛔ not a baseline for anything, and only 3 of 16 columns clear
> R² 0.80, so it supports *no rate*); era B 17,000–17,500 (2 points, transition only — the 17,500
> eval was produced by the OLD leaking trainer, so **the first clean eval is 18,000**); era C
> ≥ 18,000, n = 15, the only clean era. **ERA C READS FLAT WHERE IT MATTERS:** against the series'
> own step-to-step scatter, `eval_lat_tac` moves 0.36×, `eval_lon_tac` 0.41×, `eval_goal2s_err_m`
> 0.16×, `eval_anchor_acc` 0.22×, `eval_loss` 0.87× over 7,500 steps — **not one headline column
> moves a single scatter unit.** Only three move clearly, and they are the strategic-plumbing
> columns: `eval_goal_gate` **0.0733 → 0.1307, 14.5× scatter, R² 0.987** (the PI's CAVEAT-B answered
> with data: **the gate is opening, not stuck** — but `eval_goal2s_err_m` and `eval_goal_tac` are the
> two FLATTEST columns, so the path is being used, not yet shown to help), `eval_law` −3.7×
> (R² 0.829) and `eval_goal_score_absmean` 8.73 → 4.75 (−3.7×, R² 0.876). **AND THE TRAIN–EVAL GAP
> FLIPS SIGN ACROSS THE SAME BOUNDARY:** `loss` −0.886 (A) → **+0.238** (C), `goal2s_err_m` −0.020 →
> **+0.311**, `lat_tac` +0.113 → **+0.155**; the flip **survives** the pre-registered robustness pass
> that drops the first two rows after every relaunch (−0.840 → +0.282 / +0.003 → +0.334 / +0.088 →
> +0.141). ⇒ train keeps improving, held-out does not — the argument for **reading the epoch-end
> checkpoint at T1 rather than extending the run**. ⭐⭐ **TWO SEMANTIC CORRECTIONS THAT CHANGE HOW
> EXISTING ROWS READ:** (1) **`eval_traj` is ORACLE-SELECTED** — `a_star = dist.argmin(dim=1)` is the
> anchor nearest the GROUND TRUTH (`refc_v3_train.py:457–459`) and `recon = anchor_traj[ar, a_star]`
> (`:463`) is what `loss_traj` scores, while the model's own selection agrees with it on only **57 %**
> of windows (`eval_anchor_acc` 0.569, chance 1/128); so `eval_traj` is a **loose lower bound** on
> the trajectory refcv3 would drive, and `D-REFCV3-19K`/`D-REFCV3-21K`'s "lowest traj" are lowest
> **oracle-selected** traj. (2) **`eval_traj` is NOT an ADE** — it is `|Δ|.sum(-1) / (sv.sum()·2)`,
> a **mean L1 PER COORDINATE** in metres over the 8 slots 0.5–6.0 s (`:462–465`); ADE is an L2 norm.
> ⛔ The two may never share a column. **ENGINEERING (MEASURED):** 10 launches recovered from
> `elapsed_s` resets and corroborated by `supervisor.log` wherever they overlap (the 1,250 death
> predates that log); **2,800 steps recomputed** (2,300 from six deaths ≈ 3.02 h, 500 from three
> switches ≈ 0.66 h) plus 9 × ~237 s startup ≈ 0.59 h ⇒ ≈ **4.3 h** lost; **uint8: 4.698 → 4.069
> s/step, −13.4 %** over 143 rows (two clocks — in-process `elapsed_s` and `supervisor.log` wall time
> — agree to < 1 % on four independent spans); 0 `eval_error` rows in 614; **projected epoch end
> 2026-09-03 22:43Z** (ESTIMATED). ⛔⛔ **AND THE REFUSAL THAT TRAVELS WITH ALL OF IT: NO CONFIDENCE
> INTERVAL CAN BE COMPUTED FROM THIS FILE BY ANY ESTIMATOR** — each `eval_*` value is *already* the
> pooled mean over the 160 windows, and the file carries **no per-window values and no `eid`**, both
> of which `taniteval.ci.episode_cluster_bootstrap` requires (`ci.py:225`, paired `:275`). The
> per-window values were never written, so the interval is not recoverable later either. Every
> statement above is a **DIRECTION, not a verdict**; the only job that yields a defensible refcv3
> interval is the T1 read in the same package's `T1_CHECKLIST.md`. Instrument:
> `taniteval/tools/refcv3_metrics_read.py` (new, stdlib-only) — it prints the reconstruction and the
> refusal before any metric, because a naive row-order read of this file is wrong in three separate
> ways (replayed steps, per-launch `elapsed_s`, three measurements in one column). Work items opened:
> the run stamps neither the anchor-vocabulary size nor the eval's episode composition, and the eval
> writes no per-window array — the last one permanently forecloses intervals on 55 evals for the cost
> of one extra field.

### `D-HF-COMPARABILITY`

> ⛔⛔ **D-HF-COMPARABILITY — refav1 AND refcv3 SHARE A CORPUS AND SHARE ALMOST NOTHING ELSE: THE
> DOMINANCE CLAIM'S BLOCKER IS A DEFINITION, NOT COMPUTE (2026-09-03, Benchmarks & Evals FlyWheel;
> `Benchmarks & Evals/Research/2026-09-03-refcv3-epoch-conclusions/RESULT.md` §5).** Same 4,572-clip
> train split and same 141-clip eval split — and that similarity has been carrying far more
> rhetorical weight than it can. **refav1 is a latent WM + iCEM planner whose predictor is
> action-conditioned; refcv3 is a SUPERVISED ONE-SHOT ANCHOR TRAJECTORY MODEL (128 FPS anchors × 8
> slots) with NO action input, NO rollout and NO per-step decode.** ⇒ `t1_eval.roll_closed`
> (`t1_eval.py:753–786`) — which DOES handle a supervised model, the flagship, by decoding
> `(a_j, yaw_j)` from a `UnicycleStepReadout` head at each step and feeding
> `(steer = atan(L·κ), a_j)` back to the predictor with perception frozen at t0 — **cannot be ported
> to refcv3, because there is no action to feed back.** The flagship is supervised *and*
> autoregressive; refcv3 is supervised and one-shot, and that is the difference that matters.
> **WHAT IS COMMON:** `ha0` (constant velocity at the measured `v0`) is **identically defined and
> bit-comparable** on the shared windows — the only such arm — and `ha` is common as a control that
> can be worse than trivial. **WHAT IS NOT:** `ol` does not exist for refcv3 (it consumes no recorded
> actions); `cl` is not the same procedure on both sides and must not share a name.
> **INADMISSIBLE if any of these holds:** (1) different window grids, unasserted; (2) refcv3's
> **oracle-anchor-selected** `traj` quoted against refav1's `cl` (the C6-confound family: the eval's
> own target selects the thing being scored); (3) a cross-tier comparison, or a `cl`-to-`cl`
> comparison of two different procedures; (4) either arm degenerate on the trivial profile — then the
> read is **VOID, not negative** (`D-REFAV1-PAIRED-READ-VOID`); (5) any refcv3 arm run through the
> rescued adapter's `recorded_controls` without the steer→curvature fix (`C-REFCV3-ARM-SAME-DEFECT`;
> the identical defect cost refav1 **0.716 m** lateral against a **0.053 m** floor); (6) no shared
> floor — both arms must report against `ha0` **on the same windows**. **THE ADMISSIBLE CLAIM is
> therefore not "F beats H" but the DIFFERENCE OF EACH ARM'S MARGIN OVER THE SAME TRIVIAL FLOOR,
> PER FAMILY, with a paired episode-cluster bootstrap over the shared `eid` set** — `cl − ha0` for
> refav1 against `os − ha0` for refcv3, never `cl` against `os` as levels. Two metrics are **not
> defined for both** and must be recomputed or quarantined: refcv3's `traj` is a mean **L1 per
> coordinate**, not an L2 ADE (recompute from the dumped path; never convert), and `law` consumes a
> **future frame's** latent (T0 WM diagnostic only, no refav1 analogue). `route` is admissible for
> neither arm without the `cl_navshuf` control beside it — nav is an INPUT, and the flagship v1 route
> head scored **1.0000** by echoing it. ⚠️ **THE OPEN DECISION (PI / Master Mind, not a FlyWheel):
> does the doctrine admit as T1 a model that consumes NO actions at all?** Its T1 text says *"the
> predictor consumes the decoder/planner's own actions"*, which does not literally cover refcv3.
> Benchmarks' RECOMMENDATION, flagged as such: **admit it** — the doctrine's purpose is to keep
> future information out of inference, and a correctly-gated refcv3 forward pass (frames ≤ t0, nav,
> measured `v0`, anchor chosen by `sel_score_v3` and NOT by `a_star`) admits none — but give it a
> **distinct arm name (`os`, never `cl`)** so no reader believes the two `cl` columns describe the
> same procedure. **This ruling, not GPU, is what register decision 9 is waiting on.**

---

## 8. DELIVERABLE MANIFEST

| # | artifact | where it lives | only one place? |
|---|---|---|---|
| 1 | `SPEC.md` (pre-registration) | `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-refcv3-epoch-conclusions/SPEC.md` | **No** — repo, staged |
| 2 | `RESULT.md` (this file) | same dir | **No** — repo, staged |
| 3 | `T1_CHECKLIST.md` (Task 2 runbook) | same dir | **No** — repo, staged |
| 4 | `raw/refcv3_epoch_read.json` (every derived number) | same dir `/raw/` | **No** — repo, staged |
| 5 | `raw/refcv3_eval_series.csv` (55 canonical eval rows + launch id) | same dir `/raw/` | **No** — repo, staged |
| 6 | `raw/refcv3_train_series.csv` (503 canonical train rows + launch id) | same dir `/raw/` | **No** — repo, staged |
| 7 | `raw/metrics.jsonl` — **the run's primary**, sha256 `8207d81d…` | same dir `/raw/` | **No** — was ONLY at `C:\Users\Admin\refcv3_diag\` (one disk) + the training pod; now in the repo |
| 8 | `raw/supervisor.log` — sha256 `71ad68b0…` | same dir `/raw/` | **No** — same rescue |
| 9 | `raw/config.json` — sha256 `668e4520…` | same dir `/raw/` | **No** — same rescue |
| 10 | `taniteval/tools/refcv3_metrics_read.py` (the instrument) | `taniteval/tools/refcv3_metrics_read.py` | **No** — repo, staged |
| 11 | proposed register rows `D-REFCV3-EPOCH-READ`, `D-HF-COMPARABILITY` | §7 of this file **only** | **YES — §7 is the single copy.** They are not yet in `GOALS_AND_CLAIMS.md`; the Master Mind applies them. |

⛔ **Nothing in this package lives only on a pod, only in a worktree, or only in my context.** The
three `raw/` primaries were previously on **one disk** (`C:\Users\Admin\refcv3_diag\`) plus the
training pod; they are now in the repo, sha256-verified identical to the pull.

**Not done, deliberately:** no commit, no push, no branch switch; no edit to
`taniteval/tools/refcv3_arm.py` or to the rescued draft under
`Implementation/incoming/2026-09-03-refcv3-arm-UNVERIFIED/` (BACKLOG R20 — read only); the pod
`tanitad-refcv3` and Thor were not contacted.
