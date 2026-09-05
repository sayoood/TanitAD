# refav1 LON at T1 on Thor — `a_shift` / `a_sustain` measured as PLANNER arms

*ArchInf FlyWheel, 2026-09-05. Package:
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-lonshift-t1/`.
Predecessor: `…/2026-09-05-refav1-longitudinal/` (`RESULT.md`, decision `M36`), which
established the defect and the designs at the **expressivity** level. This package
measures whether the SEARCH emits them, at **T1**, four families.*

⛔ **Tier: T1** (the planner conditioned on its OWN actions). Per the binding
open/closed ruling, T1 is **self-action OPEN loop** — it is not "driving". `ol` is
**T0**, a world-model diagnostic only.

---

## 0. The one-line answer

⭐⭐⭐ **`a_shift` (D2) is a REAL longitudinal win at T1 — it survives its own
inference-seed floor by 6.6-12x, it REPLICATES on two independent GPUs to 0.002 m/s,
and it wins by making the planner DRIVE MORE. But refav1 still does NOT beat `ha0_ext`
outright: the answer is per family.**

**MEASURED on TWO rigs** (Thor `T_*`, dev box; paired episode-cluster bootstrap,
n = 40 windows / 8 clusters, n_boot 2000, known-value control `X - X` = `+0.0000
[+0.0000, +0.0000]` on every metric on both):

| | Thor | dev box |
|---|---|---|
| `LON_speed_mae_mps`, lever - baseline | **-0.2283 [-0.3227, -0.1420]** | **-0.2263 [-0.3173, -0.1411]** |
| its own INFERENCE-seed floor | -0.0190 | (baseline pair +0.0076) |
| **ratio — the admissible form** | ⭐ **12.0x** | 29.8x |
| gap to `ha0_ext` closed | **47.4 %** | 46.5 % |

⛔⛔ **AND THE SEED REPLICATE EARNED ITS PLACE.** `T_lonshift_s1` differs from
`T_lonshift` in **nothing but `--plan-seed`**, and it reads `separated` on **all three
longitudinal metrics** (-0.0190 / -0.0133 / -0.0329). A separated CI was necessary and
NOT sufficient in the most literal way available. The lever clears its own arm's noise
floor by **12.0x / 10.6x / 6.6x**, and is reported that way — never as "separated".

⭐ **It ACTS, it does not stop.** mean |a| **0.3140 -> 0.4691** (Thor; dev box 0.3131 ->
0.4655), zero-acceleration plans **19/40 -> 0/40**, emitted control continuous
(**17 -> 40** distinct values), CEM wins **every** window (`baseline_won_frac` 0.25 ->
0.00), CONSTANT-VELOCITY plans 27.5 % -> 2.5 %. The exact inverse of `kamm07` /
`l3ladder`, which bought their longitudinal gains by collapsing mean |a| to ~0.05 and
going 77.5 % zero — the bar was fixed in §2c BEFORE the numbers existed.

⛔ **Does refav1 beat `ha0_ext` at T1? NO — not outright.** Per family, on Thor:

* ⭐ **BEATS** it on **FDE** (-0.3706) and on **3 of 4 LATERAL metrics** — heading
  -3.5545, yaw-rate -0.0998, curvature -0.0401.
* **PARITY** on **ADE** (+0.0138 -> **-0.0889**, straddling) and on cross-track and
  LON along. Absolute ADE **0.7868** vs the floor's **0.8772** — below it for the first
  time.
* ⛔ **LOSES** the **LONGITUDINAL** family (+0.2535 speed, +0.1922 accel) and **both
  TACTICAL** metrics (-0.2500, -0.2750).
* **STRATEGIC UNAVAILABLE (n = 0)** with its reason recorded — a WORK ITEM, not a pass.

⭐ **What moved:** the programme's stated longitudinal target of **+0.4862 m/s** is
**roughly halved by a single one-variable lever**, and on the **31 maintain windows**
refav1 is now **0.158 m BETTER than the floor on ADE**. The residual sits in the **9
non-maintain windows**, whose ADE deficit `a_shift` still more than halves.

⚠️ **Two rows are NOT quoted as lever effects** (see §3b.5): `curvature` (+0.0115, only
1.2x the baseline's own seed sensitivity) and `TAC_traj_lat_correct` (-0.1000 = exactly
-4/40 windows). Both are **pending the in-rig baseline seed replicate `T_wk15_s1`**,
which is running.

---

## 1. P1 — ship-and-verify: the rig, reproduced on Thor

Full evidence, every check a positive content assertion: **`raw/SHIP_VERIFY.md`**.
Headlines:

* ⛔⛔ **The brief's checkpoint was WRONG and the error was caught before launch.**
  Thor's `experiments/refav1-b1-v72-1ep-21109/ckpt.pt` is **step 1000**, 387 param
  keys / 175,166,517 numel — a halted run of a **different architecture** whose
  `train_log.jsonl` stops at step 1000 and whose `DRIFT_ALARM` fired at step 950.
  The real ckpt is **step 21109**, 407 keys / 182,459,701 numel. Launching from the
  named directory would have produced a complete, plausible-looking four-family
  panel **of the wrong model**. The only tell was a 29 MB size gap.
  ⭐ Independently corroborated twice: `GOALS_AND_CLAIMS.md`'s `D-REFAV1-CCOS-EVAL`
  row already carries the same warning, and the correct run's own `summary.json`
  reads `final_step 21109, tensors 407, params 182459701` — matching my probe of the
  dev-box file exactly.
  ⚠️ **And Thor already HELD the correct ckpt** at `experiments/refav1-b1-v72-ep3-speed/`
  (md5 `1189bc02…`, identical to what I shipped). I probed only the directory the
  brief named — *absence found at ONE location is not absence*, and the same rule
  applies to identity.
* **md5-identical on both hosts:** ckpt `1189bc020018c2c67ce03d566c390285`, config
  `bd3ebed78eec2236156b176017745e2b`, labels `aa12c948f062181c3297265b51526ec5`,
  code bundle `16cdf3cd8e578a12d7cca95dac213225`.
* **16/16 episode artifacts byte-identical** to the dev box (8 fp8 + 8 v2ep).
* **Grep-verified on Thor before launch**: `refa_v1.py` 173,247 chars / 39 lever
  hits; `refav1_arm.py` 162,595 / 16; same-breath control `grep -c 'def '` = 58
  (non-zero, so a 0 would have meant absence and not an unreadable file).
* **Real import + CUDA conv**, not `git log`: every import resolves into the shipped
  tree, `RefAV1.plan` carries `a_sustain` / `a_shift` / `jerk_seam_a0`, and a real
  `conv2d` runs on device `NVIDIA Thor`. ⛔ No git was used to move anything —
  Thor's checkout is `30d6d601` and a `fetch` there hangs.
* ⭐ **Rig control:** Thor's arm header is line-for-line identical to the banked
  dev-box `wk15` — same label-join md5, `step=21109`, `535 windows over 8 episodes`,
  `windows=40`, `nav_shuffle=23/40`. Only paths differ.

⛔ **Parity:** the 8-episode eval slice is unchanged; `--window-stride 16` → **40
windows**, identical to the baseline. Nothing re-selected episodes.

---

## 2. Why a Thor-local baseline, and what is paired against what

The ckpt/labels/episodes/code are md5- or byte-identical, but **CEM on a different
GPU is not bit-reproducible**, so pairing a Thor arm against the dev box's `wk15`
would be a CROSS-RIG comparison. `T_wk15` is therefore arm 1; every lever is paired
against **it**, window-for-window, in one rig. `T_wk15` vs the dev box's `wk15` is
read as a **RIG** control, never as a lever result.

| # | arm | the ONE variable against `T_wk15` |
|---|---|---|
| 1 | `T_wk15` | — (in-rig baseline; shipped vocabulary, `--plan-seed 0`) |
| 2 | `T_lonshift` | `--a-sustain-mode a0_shift` — **D2, the headline** |
| 3 | `T_lonshift_s1` | `T_lonshift` + `--plan-seed 1` — **the seed replicate, MANDATORY** |
| 4 | `T_lonvocab` | `--a-sustain-mode a0` — D1, for ATTRIBUTION |
| 5 | `T_wk15_s1` | `T_wk15` + `--plan-seed 1` — the seed floor on the BASELINE |
| 6 | `T_lonseam` | `--jerk-seam a0` — the COST lever alone (P4 successor 1) |
| 7 | `T_loncomb` | D2 + the jerk seam |

All arms carry `ccos`, `(W_JERK, W_KAPPA, W_VEND) = (0.0, 15.11245, 64.29715042415070)`.
⚠️ The third weight is a **DEAD TERM** (`target_speed` is never passed; `test_C1`
pins the call site) — quoted for completeness, never attributed to.
⭐ P4's second named successor, *"`a_shift` composed with `W_KAPPA = 15.11245`"*, is
therefore **already the composition every arm runs**, not a further experiment.

⛔ **The bar, restated before the numbers.** A separated CI is NECESSARY AND NOT
SUFFICIENT: `D-REFAV1-CG-SEEDFLOOR` measured `separated` on **4 of 10** paired family
metrics between two arms differing only in `--plan-seed`. The admissible form is
*"the lever's paired delta exceeds the seed pair's delta on the same metric"*.
Arms 3 and 5 exist so that floor is read on **this** rig.

**Concurrency gate.** The queue gates on its **own children** (`jobs -p`), never on a
process-table grep — so it cannot self-match the way `pgrep -f` does — and the
artifact counted is the ARM (`--out` target), not a python process. Verified live:
3 STARTs, 3 distinct `--out` targets.

---

## 2b. ⭐⭐ AN UNPLANNED CROSS-RIG REPLICATION OF THE HEADLINE LEVER

While the Thor panel was running, the dev box independently began the sibling
`queueLON2.sh`, whose arm 1 is `lonshift`. Its command line was read from the live
process and is the **same lever on the same inputs**:

```
--ckpt …/ckpt_ep3/ckpt.pt --cache …/p4/fp8 --episodes …/p4/eps
--labels …/s2_labels_v7.2_eval.jsonl.gz --window-stride 16 --no-navshuf
--cost-metric ccos --cost-weights 0.0,15.11245,64.29715042415070
--plan-seed 0 --a-sustain-mode a0_shift
```

⇒ **`a0_shift` is being measured on TWO independent GPUs** (Thor, and the dev box's
4060), each paired against its **own** in-rig baseline. That is a replication of the
headline lever across rigs, which is strictly stronger than either panel alone, so
the Thor arms were NOT cancelled. Agreement makes the lever's effect rig-independent;
disagreement would itself be the finding.

⚠️ **A live confirmation of `M28` (3) in the same breath:** the process table showed
**two identical command lines** for that one arm — the parent and its child. A gate on
"processes ≤ 1" can never open; the artifact is the **arm** (a distinct `--out`
target), never a python process. Thor's queue gates on its own `jobs -p` for the same
reason.

---

## 2c. ⛔⛔ THE READING RULE, QUANTIFIED BEFORE THE ARMS LAND

*"Beats the floor" must be conjoined with "ACTS".* `raw/lon_emitted_devbox.txt`,
MEASURED on the banked dumps:

| arm | mean \|a\| | frac a==0 | ==a_goal | LON speed vs `wk15` | ADE vs `wk15` |
|---|---|---|---|---|---|
| `wk15` (baseline) | 0.3131 | 0.475 | 0.675 | — | — |
| `wk151` (**seed floor**) | 0.3568 | 0.425 | 0.625 | +0.0076 | +0.0150 |
| `kamm07` | **0.0532** | **0.775** | 0.775 | −0.0781 *better* | **+0.0993 WORSE** |
| `l3ladder` | **0.0747** | **0.775** | 0.775 | −0.0617 *better* | **+0.4301 WORSE** |

⇒ **`kamm07` and `l3ladder` improve the LONGITUDINAL family by making the planner
STOP.** mean |a| collapses 0.313 → ~0.05–0.07 and the zero-acceleration fraction
jumps 47.5 % → 77.5 %. That is `M27`'s paradox made mechanical, and a four-family
table alone would have scored both as longitudinal progress.

⭐ **THE BAR FOR `a_shift`, fixed in advance:** it must improve LON **while holding or
RAISING mean |a|** — i.e. by driving BETTER, not by driving LESS. An arm that ties
`ha0_ext` by emitting nothing has not driven. ⚠️ The seed floor on this diagnostic is
**+0.044 on mean |a|** (0.3131 → 0.3568) and **0.05 on frac a==0**, so movement below
that is noise.

⚠️ The baseline already emits **exactly +0.000000 on 19 of 40 windows** (47.5 %) — the
emitted-plan counterpart of M36's *goal* commanding `a ≡ 0` on 77.5 %.

---

## 3. P2/P3 — the four-family T1 panel, RIG 1 (dev box), `a_shift` vs baseline

`raw/devbox/pd_devbox_lonshift.md`, paired episode-cluster bootstrap, n = 40 windows /
8 clusters, n_boot 2000. ⛔ Not `overlapping_holdout_se`. Known-value control
`wk15.cl − wk15.cl` = `+0.0000 [+0.0000, +0.0000]` on every metric. ✅

### 3.1 The lever's effect — `lonshift.cl − wk15.cl`, against the SAME-RIG seed floor

| family | metric | `a_shift` − baseline | SEED FLOOR `wk151 − wk15` | × floor | verdict |
|---|---|---|---|---|---|
| ADE | `ade_m` | −0.1066 [−0.2114, +0.0193] | +0.0150 | 7.1× | improves, straddles |
| ADE | `fde_m` | −0.2658 [−0.5148, +0.0442] | +0.0567 | 4.7× | improves, straddles |
| **LON** | `speed_mae_mps` | ⭐ **−0.2263 [−0.3173, −0.1411]** | +0.0076 | **29.8×** | **SEPARATED** |
| **LON** | `along_mae_m` | ⭐ **−0.1398 [−0.2291, −0.0578]** | +0.0170 | 8.2× | **SEPARATED** |
| **LON** | `accel_mae_mps2` | ⭐ **−0.2078 [−0.3055, −0.1107]** | −0.0040 | 52× | **SEPARATED** |
| LON | distance-keeping | n = 4 / 1 / 4 | — | — | ⛔ UNDERPOWERED, not quotable |
| LAT | `cross_mae_m` | +0.0202 [−0.0394, +0.1071] | +0.0073 | — | straddles — **no lateral cost** |
| LAT | `heading_mae_deg` | +0.8225 [−0.2238, +2.2962] | +0.4148 | — | straddles |
| LAT | `yaw_rate_mae_radps` | +0.0146 [−0.0040, +0.0379] | +0.0096 | — | straddles |
| LAT | `curvature_mae_1pm` | +0.0112 [+0.0034, +0.0231] | **+0.0093 [+0.0015, +0.0221]** | **1.2×** | ⚠️ separated but **INSIDE the floor** ⇒ NOT attributable |
| TAC | `lat_correct` | −0.1000 [−0.1750, −0.0250] | **−0.1000 [−0.1750, −0.0250]** | **1.0×** | ⚠️ **EQUALS the floor exactly** ⇒ NOT attributable |
| TAC | `lon_correct` | +0.0250 [−0.1250, +0.2000] | +0.0000 [0, 0] | — | straddles |
| STRAT | — | UNAVAILABLE (n = 0), reason recorded | — | — | WORK ITEM |

⚠️ **Two "separated" rows are correctly REFUSED as lever effects.** The tactical-lat
delta is **identical to the seed floor in point estimate AND interval to 4 dp**; the
metric is a fraction over 40 windows, so −0.1000 is exactly **−4/40** — a
small-integer coincidence, read beside the raw count rather than as a shared effect.
Curvature clears `separated` at only **1.2×** its floor with heavily overlapping
intervals. Under the binding rule (*the lever's delta must EXCEED the seed pair's on
the same metric*) neither is attributable. **This is the H-ESTIM-SEED-1 discipline
doing its job on live data, not a hypothetical.**

### 3.2 ⭐⭐⭐ IT ACTS — the lever improves LON by DRIVING MORE, not by stopping

`raw/devbox/lon_emitted_lonshift.txt`:

| arm | mean \|a\| | frac a==0 | distinct a[0] | ==a_goal | cem frac |
|---|---|---|---|---|---|
| `wk15` baseline | 0.3131 | 0.475 | 17 | 0.675 | 0.750 |
| `wk151` seed floor | 0.3568 | 0.425 | 18 | 0.625 | 0.725 |
| ⭐ **`lonshift` (`a_shift`)** | **0.4655** | **0.000** | **40** | 0.050 | **1.000** |
| `kamm07` | 0.0532 | 0.775 | 10 | 0.775 | 0.750 |
| `l3ladder` | 0.0747 | 0.775 | 10 | 0.775 | 0.750 |

* mean |a| **+0.152 = 3.5× the +0.044 seed floor** — the planner commands MORE acceleration.
* the zero-acceleration plan is **eliminated: 0 of 40** windows (baseline 19/40).
* the emitted control becomes **continuous — 40 distinct values**, i.e. the
  longitudinal quantisation is gone (baseline 17, `kamm07`/`l3ladder` 10).
* **CEM wins every window** (`baseline_won_frac` 0.25 → **0.00**); `hold_v0` never wins.
* trivial-profile: CONSTANT-VELOCITY plans **25 % → 2.5 %**.

⇒ this is the **exact inverse** of `kamm07`/`l3ladder`, which bought their longitudinal
gains by collapsing mean |a| to ~0.05 and going 77.5 % zero. The bar was fixed before
the numbers existed (§2c) and `a_shift` clears it.

⚠️ `==a_goal` falls 0.675 → 0.050: the plan is no longer a copy of the goal profile.
With a REACHABLE target the search stops merely reproducing the canonical controls and
optimises around them — consistent with `cem frac` going to 1.0.

### 3.3 Attribution — both branches improved, the 9 non-maintain windows most

`raw/devbox/lon_attr_lonshift.txt`, per-window paired `cl − ha0_ext` by branch:

| stratum | n | LON speed, `wk15` | LON speed, `lonshift` | ADE, `wk15` | ADE, `lonshift` |
|---|---|---|---|---|---|
| ALL | 40 | 0.4862 | **0.2599** | +0.0162 | **−0.0904** |
| MAINTAIN | 31 | 0.3879 | — (`cl` 0.6886 → **0.4713**) | −0.0728 | — |
| **NON-maintain** | 9 | 0.8247 | **0.5676** | **+0.3228** | **+0.1486** |

The 9 non-maintain windows carried **2.13×** the longitudinal error and **all** of the
ADE deficit. `a_shift` — designed for exactly those, since it may only move a token
naming a RELATIVE target — cuts their LON error by 31 % and **more than halves** their
ADE deficit, while also improving the maintain branch. The addressable fraction falls
61.8 % → 50.9 %, i.e. the residual is now more evenly spread.
⭐ The SANITY control holds: the floor's own error barely differs by branch
(0.3007 vs 0.3231), as it must — `ha0_ext` cannot see the decoded token.

### 3.4 ⛔ THE ANSWER ON RIG 1 — does refav1 beat `ha0_ext` at T1?

`lonshift.cl − ha0_ext` (negative = refav1 better, except TAC where higher-is-better):

| family | metric | vs floor | separated | verdict |
|---|---|---|---|---|
| ADE | `ade_m` | −0.0904 [−0.2078, +0.0177] | no | **parity, improved** (baseline was +0.0162) |
| ADE | `fde_m` | **−0.3723 [−0.7473, −0.0180]** | yes | ⭐ **BEATS** |
| LON | `speed_mae_mps` | **+0.2599 [+0.1290, +0.4265]** | yes | ⛔ **LOSES** — but **46.5 % of the gap closed** |
| LON | `along_mae_m` | −0.1588 [−0.3510, +0.0121] | no | parity |
| LON | `accel_mae_mps2` | **+0.2039 [+0.0878, +0.3654]** | yes | ⛔ LOSES — **50.5 % closed** |
| LAT | `cross_mae_m` | −0.0145 [−0.0651, +0.0244] | no | parity |
| LAT | `heading_mae_deg` | **−3.6436 [−5.7301, −1.2515]** | yes | ⭐ **BEATS** |
| LAT | `yaw_rate_mae_radps` | **−0.1009 [−0.1461, −0.0525]** | yes | ⭐ **BEATS** |
| LAT | `curvature_mae_1pm` | **−0.0405 [−0.0870, −0.0106]** | yes | ⭐ **BEATS** |
| TAC | `lat_correct` | −0.2500 [−0.4000, −0.0750] | yes | ⛔ LOSES |
| TAC | `lon_correct` | −0.2750 [−0.4000, −0.1250] | yes | ⛔ LOSES |

⇒ **NO — not outright, and the honest answer is per family.** refav1 with `a_shift`
**BEATS** `ha0_ext` on FDE and on three of four lateral metrics, reaches **parity** on
ADE (moving from +0.0162 to −0.0904), and still **LOSES** the longitudinal family
(+0.2599) and both tactical metrics. ⭐ But the longitudinal gap — the programme's
stated target of **+0.4862** — is now **halved by a single one-variable lever**, and
the arm's ABSOLUTE ADE (**0.7868**) is below the floor's (**0.8772**) for the first time.

---

## 3b. ⭐⭐⭐ RIG 2 (Thor) — the in-rig panel, and the CROSS-RIG REPLICATION

`raw/thor/pd_thor.md`, paired episode-cluster bootstrap, n = 40 windows / 8 clusters,
n_boot 2000. Known-value control `T_wk15.cl − T_wk15.cl` = `+0.0000 [+0.0000, +0.0000]`
on every metric. ✅ The tool also asserts the shared floors are bit-identical across
dumps, and REFUSES outright if two arms are not on the same windows.

### 3b.1 Two independent GPUs, one answer

| metric | Thor `T_lonshift − T_wk15` | dev box `lonshift − wk15` | Δ between rigs |
|---|---|---|---|
| **`LON_speed_mae_mps`** | **−0.2283 [−0.3227, −0.1420]** | **−0.2263 [−0.3173, −0.1411]** | **0.0020** |
| `LON_along_mae_m` | −0.1405 [−0.2372, −0.0576] | −0.1398 [−0.2291, −0.0578] | 0.0007 |
| `LON_accel_mae_mps2` | −0.2157 [−0.3156, −0.1200] | −0.2078 [−0.3055, −0.1107] | 0.0079 |
| `ade_m` | −0.1027 [−0.2084, +0.0253] | −0.1066 [−0.2114, +0.0193] | 0.0039 |
| `fde_m` | −0.2636 | −0.2658 | 0.0022 |
| mean \|a\| (baseline → lever) | 0.3140 → 0.4691 | 0.3131 → 0.4655 | ≤0.004 |
| `baseline_won_frac` | 0.25 → **0.00** | 0.25 → **0.00** | — |

⭐ **The headline longitudinal effect agrees to 0.002 m/s across two different GPUs.**
This replication was unplanned — the dev box independently began the sibling
`queueLON2.sh` while Thor's panel ran — and it removes "the rig" as an explanation.

### 3b.2 ⛔⛔ THE MANDATORY INFERENCE-SEED REPLICATE FIRES EXACTLY THE WARNING IT EXISTS FOR

`T_lonshift_s1` is `T_lonshift` with **one variable changed — `--plan-seed 0 → 1`.**
Nothing else. iCEM samples, so this is the INFERENCE variance
(`D-REFAV1-SEED-GOAL-MISMATCH`), the third of the three variances a single interval is
blind to.

| metric | `T_lonshift_s1 − T_lonshift` (SEED ONLY) | separated? |
|---|---|---|
| `LON_speed_mae_mps` | **−0.0190 [−0.0392, −0.0030]** | ⛔ **YES** |
| `LON_along_mae_m` | **−0.0133 [−0.0267, −0.0011]** | ⛔ **YES** |
| `LON_accel_mae_mps2` | **−0.0329 [−0.0566, −0.0112]** | ⛔ **YES** |
| `ade_m` / `fde_m` | −0.0089 / −0.0189 | no |
| `LAT_*` (cross/heading/yaw) | −0.0035 / +0.0809 / +0.0010 | no |
| `TAC_traj_lat_correct` | **+0.0000 [+0.0000, +0.0000]** | no |
| `curvature_mae_1pm` | +0.0013 [−0.0008, +0.0044] | no |

⇒ **Changing NOTHING but the inference seed produces `separated` on all three
longitudinal metrics — the very family this package claims.** A separated CI is
therefore necessary and NOT sufficient here, in the most literal way possible.

⭐ **The sufficient form — the lever's delta against its OWN arm's seed floor:**

| metric | lever | in-rig inference-seed floor | ratio | verdict |
|---|---|---|---|---|
| `LON_speed_mae_mps` | −0.2283 | −0.0190 | **12.0×** | ✅ **CLEARS** |
| `LON_along_mae_m` | −0.1405 | −0.0133 | **10.6×** | ✅ CLEARS |
| `LON_accel_mae_mps2` | −0.2157 | −0.0329 | **6.6×** | ✅ CLEARS |

**The longitudinal claim survives its own noise floor by 6.6–12×.** It is reported this
way, not as "separated".

### 3b.3 ⛔ THE ANSWER ON RIG 2 — `T_lonshift.cl − ha0_ext`, per family

(negative = refav1 better, except `TAC_*_correct` where higher is better)

| family | metric | `a_shift` vs floor | baseline vs floor | separated | verdict |
|---|---|---|---|---|---|
| ADE | `ade_m` | **−0.0889 [−0.2049, +0.0137]** | +0.0138 | no | **PARITY, improved** |
| ADE | `fde_m` | **−0.3706 [−0.7361, −0.0293]** | −0.1070 | yes | ⭐ **BEATS** |
| LON | `speed_mae_mps` | **+0.2535 [+0.1162, +0.4227]** | +0.4818 | yes | ⛔ **LOSES — 47.4 % closed** |
| LON | `along_mae_m` | −0.1632 [−0.3535, +0.0044] | −0.0226 | no | parity |
| LON | `accel_mae_mps2` | **+0.1922 [+0.0663, +0.3562]** | +0.4079 | yes | ⛔ LOSES — **52.9 % closed** |
| LON | distance-keeping | n = 4 / 1 / 4 | — | — | ⛔ UNDERPOWERED, not quotable |
| LAT | `cross_mae_m` | −0.0100 [−0.0633, +0.0299] | −0.0347 | no | parity |
| LAT | `heading_mae_deg` | **−3.5545 [−5.6085, −1.1910]** | −4.5659 | yes | ⭐ **BEATS** |
| LAT | `yaw_rate_mae_radps` | **−0.0998 [−0.1447, −0.0524]** | −0.1155 | yes | ⭐ **BEATS** |
| LAT | `curvature_mae_1pm` | **−0.0401 [−0.0862, −0.0105]** | −0.0503 | yes | ⭐ **BEATS** |
| TAC | `lat_correct` | **−0.2500 [−0.4000, −0.0750]** | −0.1500 | yes | ⛔ LOSES |
| TAC | `lon_correct` | **−0.2750 [−0.4000, −0.1250]** | −0.2750 | yes | ⛔ LOSES |
| STRAT | — | UNAVAILABLE (n = 0), reason recorded | — | — | WORK ITEM |

### 3b.4 Attribution on Thor — the maintain branch now BEATS the floor outright

`raw/thor/lon_attribution_all.txt`, per-window paired `cl − ha0_ext` by branch:

| stratum | n | LON speed, `T_wk15` | LON speed, `T_lonshift` | ADE, `T_wk15` | ADE, `T_lonshift` |
|---|---|---|---|---|---|
| ALL | 40 | +0.4818 | **+0.2535** | +0.0138 | **−0.0889** |
| **MAINTAIN** (`a_shift` acts) | 31 | +0.3879 | **+0.1623** | −0.0728 | ⭐ **−0.1579** |
| NON-maintain | 9 | +0.8247 | **+0.5676** | +0.3228 | **+0.1486** |

⭐ On the **31 maintain windows** refav1 with `a_shift` is now **0.158 m better than
`ha0_ext` on ADE** and within **0.16 m/s** on speed. The residual deficit is
concentrated in the **9 non-maintain windows**, whose ADE deficit `a_shift` still more
than halves (+0.3228 → +0.1486).

⚠️ SANITY control holds: the floor's own LON error barely differs by branch (0.3007 vs
0.3231) — `ha0_ext` cannot see the decoded token, so it must not.

### 3b.5 ⚠️ Two rows that must NOT be quoted as lever effects yet

* **`curvature`.** `T_lonshift − T_wk15` = +0.0115 [+0.0036, +0.0238], separated. Its
  own arm's inference-seed floor is +0.0013 (8.8×), but the BASELINE arm's seed
  sensitivity on the same metric measured **+0.0093** on the dev box — making it only
  **1.2×**. ⇒ borderline; **not established**. The in-rig baseline replicate
  (`T_wk15_s1`) is running and settles it.
* **`TAC_traj_lat_correct` = −0.1000** (exactly **−4/40** windows). The lever arm is
  seed-STABLE here (`T_lonshift_s1 − T_lonshift` = +0.0000 [0, 0]), but the dev box's
  BASELINE seed pair moved this metric by exactly −0.1000 too. Two different arms'
  seed sensitivities differ, so the attribution is **pending `T_wk15_s1`**.
⭐ Both are named as pending rather than reported either way — that is what the
seed-floor rule is for, and it is doing real work here rather than being recited.

---

## 4. ⭐ An instrument gap closed in passing: curvature had no PAIRED interval

⚠️ **Scoped precisely, because the loose claim is false.** `four_families` already
reports `curvature_mae_1pm` in each arm's **absolute** panel (banked `wk15.cl`:
0.0310, bias −0.0071, `n_steps_curvature` 271, `excluded_below_min_ds` 96). What did
not exist is curvature in the **paired** panel: `refav1_arm._FAMILY_OF` lists only
`LAT_cross_mae_m`, `LAT_heading_mae_deg`, `LAT_yaw_rate_mae_radps`, so curvature has
carried **no paired delta and no interval**, and every cross-arm LATERAL comparison
in the programme has been made without it. ⇒ **the gap was the ESTIMATOR, not the
metric.**

`raw/lat_curvature.py` closes it POST HOC from the banked dumps — zero GPU, and
without touching the instrument that produced the arms, so every arm stays
bit-comparable to the banked panel. It masks with **`pair_valid`**, not `valid`:
⛔ curvature is a PAIR quantity of shape `[n, H-1]` while `valid` is `[n, H]`
(MEASURED: (3,10) vs (3,11)) — masking with `valid` mis-aligns rather than being
merely imprecise. Its known-value control `X − X` reads `+0.000000 [0, 0]`. ✅

### 4.1 MEASURED on the banked dev-box dumps (n = 40 windows, 8 episodes)

| pair | curvature MAE delta (1/m) | 95 % CI | n | separated |
|---|---|---|---|---|
| ⭐ **`wk151.cl − wk15.cl` (SEED FLOOR)** | **+0.0093** | [+0.0015, +0.0221] | 32 | **yes** |
| `wk15.cl − ha0_ext` | **−0.0504** | [−0.1097, −0.0153] | 33 | **yes** |
| `wk15.cl − ha0` | −0.0091 | [−0.0211, −0.0017] | 33 | yes |
| `wk15.cl − ha` | −0.0487 | [−0.1066, −0.0126] | 33 | yes |
| known-value control `wk15.cl − wk15.cl` | +0.0000 | [0, 0] | 33 | — ✅ |

Arm means (1/m): `wk15.cl` 0.0327 · `wk151.cl` 0.0408 · `ha0` 0.0419 · `ha` 0.0823 ·
`ha0_ext` 0.0835 · `ol` 0.0891.

⭐⭐ **Two things follow, and the first is a rule, not a result.**

1. **The inference-seed floor is `separated` on curvature too.** Two arms differing
   in **nothing but `--plan-seed`** read **+0.0093 [+0.0015, +0.0221], separated**.
   That extends `H-ESTIM-SEED-1` / `D-REFAV1-CG-SEEDFLOOR` from **4 of 10** to
   **5 of 11** paired family metrics on which a pure seed change clears
   `separated`. ⇒ **the curvature seed floor is +0.0093**, and no curvature claim
   below it is admissible.
2. Against that floor, **refav1 BEATS `ha0_ext` on curvature by −0.0504** — **5.4×
   the seed floor** — a genuine lateral win on a metric that had never carried a
   paired interval.

⚠️ `wk15.cl` reads 0.0327 here against 0.0310 in the absolute panel because the
reducers differ: this is a **per-window masked mean, then averaged over windows**
(what a paired window-level bootstrap requires), while the absolute panel pools over
the 271 valid STEPS. Same quantity, different reducer — stated so the two are never
quoted as a discrepancy.

---

## 4b. ⭐ LONGITUDINAL distance-keeping RECOVERED — and reported with its n

The binding rule names **distance-keeping (headway / time-gap / TTC)** as part of the
LONGITUDINAL family; every banked refav1 panel reported it `REFUSED — no lead block
passed`. It is attachable at **ANALYSIS** time from the dumps (`--lead-block`), so it
costs **zero GPU** and needs no re-run. The banked B1 EVAL block
(`b1_eval_lead_block.npz`, md5 `33a48e15a52eb9dbd69ecd6026fd5023`) was shipped to Thor
and the panel is emitted twice — without the block (comparable to `pd_lonbase`) and
with it (`pd_thor_lead.md`).

⛔⛔ **AND IT IS UNDERPOWERED, WHICH IS THE POINT OF SAYING SO.** MEASURED on the
banked dev-box dumps (`raw/pd_devbox_lead.md`): of 40 windows the block covers
**n = 4** for `LON_dk_headway_min_m` and `LON_dk_min_ttc_s`, and **n = 1** for
`LON_dk_time_gap_min_s`. At those n the intervals are degenerate — several read
exactly `[+0.0000, +0.0000]`, and `wk15.cl − ha0_ext` shows `min_ttc` `+1.0770
[+0.0000, +2.1539]`. ⇒ **these three metrics are reported WITH THEIR n and are NOT
quotable as results**; a `+0.0000 [0,0]` from **n = 1** is a degenerate estimate, not
a known-value control passing. Recovering the family was right; quoting it at n = 1
would have been the `df`-trap in a new costume.

---

## 4c. The reference the Thor arms are read against (dev box, banked)

`raw/pd_lonbase.md` (sibling package), n = 40 windows / 8 clusters, n_boot 2000:

| pair | ade_m | LON_speed_mae_mps | LAT_heading_mae_deg | TAC_traj_lat_correct |
|---|---|---|---|---|
| **`wk15.cl − ha0_ext`** | +0.0162 [−0.1648, +0.1980] | ⛔ **+0.4862 [+0.2825, +0.7201]** | ⭐ **−4.5589 [−6.5917, −2.0464]** | −0.1500 [−0.3000, +0.0250] |
| **`wk151.cl − wk15.cl` (SEED FLOOR)** | +0.0150 [−0.0288, +0.0817] | +0.0076 [−0.0134, +0.0282] | +0.4148 [−0.5442, +1.8460] | ⛔ **−0.1000 [−0.1750, −0.0250]** |

⇒ the standing verdict in one line: refav1 **ties** ADE, **wins** the lateral family
(heading −4.56, yaw-rate −0.115, and now curvature −0.050), and **loses** the
longitudinal family by **+0.4862 m/s**. That single number is what these arms attack.

⚠️ **And the seed floor is itself `separated` on `TAC_traj_lat_correct`
(−0.1000 [−0.1750, −0.0250])** — a **10-percentage-point** tactical floor produced by
changing nothing but `--plan-seed`. Together with curvature (§4.1) this is now
**two more** metrics on which a pure seed change clears `separated`, so no tactical-lat
or curvature claim below those floors is admissible.

---

## 5. What is NOT claimed

* **`strategic` is UNAVAILABLE on this rig, with its reason and n**, not silently
  dropped: the record states `missing ['route_pred','route_gt']` — a world-model
  fidelity pass does not traverse the hierarchy. Producing it needs a
  hierarchy-traversing eval, which remains a WORK ITEM.
* **The TACTICAL family here is trajectory-derived and is NOT "selected vs
  executed".** The record says so verbatim: both label streams are EXECUTED
  manoeuvres (the arm's and the human's). Scoring a DECLARED decision needs a
  tactical head and stays unavailable on a trajectory dump.
* `a_shift` / `a_sustain` / `jerk_seam_a0` becoming refav1's **defaults** is not this
  agent's call; all three remain OFF by default and every default stays
  bit-identical to the pre-2026-09-05 planner. These arms MEASURE them.
* Arming `W_VEND` remains a **PI decision** (`test_C1` / `test_C2` pin it).

---

## 6. Deliverable manifest

⛔ Everything below is IN THE REPO, staged, with provenance. Nothing is single-copy
off-repo. Package root:
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-lonshift-t1/`

| artifact | where |
|---|---|
| `RESULT.md` (this file) | package root, repo |
| `raw/SHIP_VERIFY.md` — P1 evidence, every check a positive content assertion | repo |
| `raw/ckpt_probe.py` — the probe that caught the wrong checkpoint | repo |
| `raw/queueTHOR.sh` · `raw/finalize_thor.sh` — the Thor rig, `jobs -p` gated | repo |
| ⭐ `raw/lat_curvature.py` — NEW instrument: the paired, CI'd LATERAL curvature delta | repo |
| `raw/arm_summary.py` — per-arm four-family reader | repo |
| `raw/lon_{emitted,oracle,attribution}.PORTABLE.py` — the sibling scripts, de-hardcoded | repo |
| `raw/instrument_validation.txt` — all three reproduce PUBLISHED values, controls pass | repo |
| `raw/thor/pd_thor.{md,json}` · `pd_thor_lead.{md,json}` — the in-rig four-family panels | repo |
| `raw/thor/lat_curv_thor.json` · `lon_{emitted,oracle,attribution}_all.txt` | repo |
| `raw/thor/rec_T_{wk15,lonshift,lonshift_s1}.json` — the arm records | repo |
| `raw/thor/lon_arms_present.txt` — NAMES the arms not yet landed | repo |
| `raw/devbox/pd_devbox_lonshift.{md,json}` · `lat_curv_lonshift.txt` · `lon_{emitted,attr}_lonshift.txt` | repo |
| `raw/pd_devbox_lead.{md,json}` — the lead-block preflight (n = 4 / 1 / 4) | repo |
| `raw/lat_curv_devbox.json` · `raw/lon_emitted_devbox.txt` — the reference floors | repo |
| ⚠️ `raw/patch_goal_reach.NOT_APPLIED.py` — P4's next lever, prepared, NOT applied | repo |
| the Thor rig itself (code, ckpt, dumps) | `thor6:/home/nvidia/refav1_lon/` |
| the dev-box arms | `C:/Users/Admin/refav1_margin/p4out/` (sibling stream's) |

### 6.1 Still running when this was written — and they are the CONTINUATION, not a gap

⛔ Per RULE ZERO this turn does not stop at the verdict. Executing on Thor:

| arm | what it settles | ETA (UTC) |
|---|---|---|
| `T_lonvocab` (`--a-sustain-mode a0`) | **D1 vs D2 attribution** — is `a_shift`'s edge real at arm level? | ~23:15 |
| `T_wk15_s1` (`--plan-seed 1`) | the **in-rig BASELINE seed floor** — settles curvature and TAC-lat (§3b.5) | ~23:15 |
| `T_lonseam` (`--jerk-seam a0`) | **P4 successor 1** — the cost lever alone | ~23:15 |
| `T_loncomb` (D2 + seam) | whether the two compose | ~00:20 |

and on the dev box, the sibling `queueLON2.sh` is running `lonseam`, then `lonshift_s1`,
`loncomb2`, `lonvocab` — a second rig for every one of them.

⭐ P4's further named lever, **`goal_reach_s`** (HANDOFF §4.2 — every token realises only
**0.6513x** its named `dv` inside the 2 s window), is **implemented and banked as a
NOT-APPLIED patch** with its own pre-registered controls. ⛔ It is deliberately not
applied: both code trees are feeding running panels, and editing either mid-flight is
the `supervise_run.sh` trap in source-code form — running processes keep the old bytes
while the next launch silently picks up new ones, putting two experiments under one name.
Apply once both queues drain, and require the default path (`--goal-reach-s 2.0`) to
reproduce `T_wk15` bit-for-bit before quoting any arm.
