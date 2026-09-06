# PRE-REGISTRATION — the refcv5 landing read

**Written 2026-09-06, BEFORE `ckpt_40284_FINAL.pt` exists.** `refcv5-ddim-b1-v72-40k` launched
2026-09-06 10:13:52 UTC on `tanitad-a40` (= `tanitad-refcv3`, one host, two aliases) at
**3.938 s/step**, ETA **≈ 2026-09-08 06:20 UTC**. At the time of writing the run is ~15 % through
its 40,284 steps and **no eval number exists for it in any form**.

⛔ **Nothing below may be edited after the numbers land.** A changed criterion is a moved goalpost
and the correction goes to `Project Steering/RETRACTION_LOG.md`, not into this file.

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-refcv5-landing-prereg/`
**Analysis script (written now, so the landing is a RE-RUN and not a design exercise):**
`raw/refcv5_landing_analysis.py`
**Evidence class of every number quoted below as a bar:** `MEASURED (ours)` — refcv4b @ 40,284,
`…/2026-09-06-refcv4b-landing/LANDING_RESULT.md` §§1–5 and its `raw/*.json`.

---

## 0. THE SCHEMA BLOCK

```yaml
hypothesis: H-REFCV5-DDIM-1        # registered in GOALS_AND_CLAIMS.md this turn
one_variable: sampler_ddim_control_space   # --sampler ddim + its mandatory --w-u0 0.5 (see §1)
held_constant: [corpus, labels, anchors_file, n_anchors, anchor_v0_conditioned,
                anchor_control_units, seed, steps, batch, workers, prefetch_factor,
                v2_lru, lr, warmup, image_hw, sel_accel_max, goal_str,
                ego_state_inject, ego_dropout, eval_cache, eval_labels,
                eval_every, eval_batches, agents, arm, size]
success: >
  PRIMARY, on ade_m, paired episode-cluster bootstrap over the SAME 4,823 windows /
  141 episodes, n_boot 2000, seed 0:
    (P1) refcv5 os - ha0_ext is NEGATIVE and separated, AND |delta| > F(ade_m);
  where F(ade_m) is refcv5's OWN measured inference-seed floor (SS2), not an
  imported one. P1 is the bar refcv4b FAILED (+0.0101, not separated).
failure: >
  (F1) refcv5 os - refcv4b os is NOT separated, or |delta| <= F(ade_m)  => NULL:
       the control-space DDIM sampler does not move driving on this surface at
       this scale; OR
  (F2) refcv5 os - refcv4b os is separated POSITIVE  => REGRESSION.
  Either way the WP-4 rung is reported as not having earned its 44 GPU-hours, and
  the next lever is named in the same document.
controls: [constant_only, raw_input_floor, deliberate_regression, inference_replicate]
splits:
  fit:  B1 train cache /root/data/train, 4,572 clips (the trainer refuses any train-eval overlap)
  val:  the SAME 141 held-out clips, read at --eval-every 500 for MONITORING ONLY;
        no hyper-parameter was selected on it because NO SWEEP WAS RUN and the
        scored checkpoint is chosen by STEP COUNT (ckpt_40284_FINAL.pt), never by
        a val score. This is disclosed, not claimed clean.
  test: the same 141 clips at --window-stride 5 => 4,823 windows, scored ONCE.
```

---

## 1. THE ONE VARIABLE — and why two flags are ONE mechanism

refcv5's argv is refcv4b's **byte for byte**, plus `--sampler ddim --w-u0 0.5`, plus an explicit
`--agents off`.

| flag | status | source |
|---|---|---|
| `--sampler ddim` | ⭐ **THE LEVER** | `refc_v3_train.py:3319`, default `none` |
| `--w-u0 0.5` | ⭐ **NOT a second lever — mandatory** | `U0_WEIGHT_DEFAULT = 0.0` (`:114`); `--sampler ddim` with `--w-u0 0` is **REFUSED** at `:302-309`; and a stamp with `w_u0 > 0` and no `control_head` is **REFUSED** at `:2113-2115`. ⇒ the sampler and its x0 loss cannot exist apart. |
| `--agents off` | ⭐ **a stamp, not a change** | `:3400` — `off` **is the default**, so passing it explicitly alters nothing |
| `--sampler-space control` · `--sampler-infer-t 8` · `--sampler-steps 2` · `--sampler-groups 1` | defaults, unpassed | `:3325`, `:3333`, `:3335`, `:3340` |

⇒ **Attribution is CLEAN, and it is clean for a structural reason rather than a bookkeeping one.**
`refc.py:2107` sets `_loop_steps = 0 if self.control_head is not None else steps`: when the sampler
is built, the pre-v5 truncated-diffusion refinement loop is **disabled**, so exactly one mechanism
refines the fan. Running both would put two perturbations on one fan and make the arm
non-attributable — the `--v2` conflation failure — and the code refuses to.

### ⛔ G-ARGV — the one-variable check is a GATE, run on the artifacts, not on this prose
The landing **asserts** the diff rather than trusting it: `config.json['argv']` of
`refcv5-ddim-b1-v72-40k` against `refcv4b-b1-v72-40k`, and the symmetric difference **must be
exactly** `{--sampler ddim, --w-u0 0.5, --agents off}`. Anything else and the comparison is
**VOID**, not weakened. *(MEASURED 2026-08-22: a row-bank arm changed `n` and silently multiplied
λ by ~n/24; two sweeps were invalidated. Diff the launch commands, not the intent.)*

---

## 2. ⛔⛔ THE THREE VARIANCES — name which one each interval answers

A separated CI is a claim about **one** of three questions. Every margin in the landing read must
say which.

| | question | what prices it here | status |
|---|---|---|---|
| **V1 episode draw** | *would another draw of EPISODES say this?* | paired episode-cluster bootstrap over the 141 clips (`taniteval/ci.py::paired_episode_cluster_bootstrap`), n_boot 2000, seed 0 | ✅ **PRICED** |
| **V2 training run** | *would another TRAINING RUN say this?* | a second 40,284-step run at a different `--seed` | ⛔ **NOT PRICED — a named blocker (§9 B2).** `H-ESTIM-SEED-1`: on the v7-tiny rig a **zero-lever replicate** (same flags, same seed, argv-audited) produced *"separated"* differences on **3 of 18** family metrics — a **~17 % false-positive rate for `separated`**. Every refcv5-vs-refcv4b margin below is therefore **necessary, not sufficient**, and must be written as *"this checkpoint differs from that checkpoint"*, never *"the sampler is worth X"* |
| **V3 inference run** | *would another INFERENCE RUN say this?* | ⭐ **the replicate arm this SPEC adds — §2.2** | ✅ **PRICED, and it MUST be: refcv5 is stochastic at eval** |

### 2.1 ⛔ WHY refcv4b's CIs CLOSED V3 BY CONSTRUCTION AND refcv5's CANNOT

* **refcv4b:** `refc.py:2110-2111` — `noise = torch.randn_like(x) * noise_std **if self.training**
  else torch.zeros_like(x)`. Deterministic outside training. V3 is a structural identity.
* **refcv5:** `refc.py:1815` — `eps = torch.randn_like(x0_n)`, **with no `self.training` guard**,
  under a comment at `:1801-1813` that says so *by design*: *"a SAMPLER cannot [zero its noise],
  because sampling is the mechanism."*

⇒ ⛔ **A separated CI from ONE refcv5 roll answers a question nobody asked.**

### 2.2 THE REPLICATE ARM — `os_R1`, `os_R2`

| | |
|---|---|
| **what it is** | the **same checkpoint**, the **same windows**, the **same argv byte for byte except `--dump-dir` / `--out`**, rolled again in a **fresh process** |
| **which variance it prices** | ⭐ **V3, the INFERENCE draw — and only that.** It moves no lever, retrains nothing and re-selects no episode |
| **why a plain re-run IS an independent draw (MEASURED, ours, 2026-09-06)** | `refcv3_arm.py` **never calls `torch.manual_seed`** — count **0** against a same-breath control of **33** `add_argument` — and its `--seed` reaches only `analyze_refcv3` / `_boot` / the nav+gstr shuffles. Torch's default generators are seeded **nondeterministically per process**: three processes on the dev-box venv gave CPU `initial_seed` **198353026560400 / 198354645806200 / 198356256509000** and CUDA `initial_seed` **6877072347531542 / 1187870223787850 / 2024662901884684**, with three distinct draws on each device. ⇒ a second invocation re-rolls `eps` at `refc.py:1815` |
| **how many** | **three rolls total** — `os_R0` (primary) + `os_R1` + `os_R2`. ⚠️ If GPU is short, two are acceptable and the floor is then a **single draw** and must be labelled as such in the record |
| **the floor** | **`F(m)` = max over the three pairwise `|delta|` on metric `m`**, computed by the same paired estimator on the same windows |
| **the quotability rule** | a margin `D(m)` between two different arms is **QUOTABLE** iff its paired CI excludes zero **AND** `|D(m)| > F(m)`. Otherwise it is **WITHIN-NOISE** and is reported as such — never as a null and never as a win |

⛔⛔ **THE FLOOR IS refcv5's OWN, MEASURED HERE. refav1's ≈0.30 m ADE floor is NOT imported.**
That number was measured on a **different rig** — an iCEM planner sampling 128 rollouts — while
refcv5 draws **one** `eps` at truncation `t = 8` and takes **2** DDIM steps. Quoting refav1's
floor as refcv5's bar would be a true measurement used outside its scope, which is the
`df` / Thor-`free` / `step_s` / cylindrical-FOV family. **refav1's 0.30 m is the REASON the
replicate is mandatory; it is never the BAR.**

### 2.3 ⛔ G-STOCH — the replicate has its own validity gate, in BOTH directions
* **`os_R0` vs `os_R1` must NOT be bit-identical.** If they are, either the sampler is not live at
  eval or the process did not re-seed, and the arm is **not the arm this SPEC describes** ⇒
  **VOID**, and the fix is an explicit per-roll `torch.manual_seed` before anything is quoted.
* **`ha`, `ha0`, `ha0_ext` MUST come back bit-identical across all three rolls.** They consume no
  model output; anything else means the harness moved something it should not have ⇒ **VOID**.
* **the estimator's self-control:** `paired(a, a)` must read `delta` exactly `0.0`, interval
  `[0, 0]`, `separated False`. If it does not, no number in the panel is admissible.

---

## 3. ⛔ THE BARS — committed in advance, verbatim

Every bar is on **`ade_m`**, paired episode-cluster bootstrap, **the same 4,823 windows / 141
episodes** refcv4b and refcv3 were scored on. The refcv4b column is `MEASURED (ours)`.

| the arm refcv5 is read beside | ADE (m) | refcv4b's own margin against it |
|---|---|---|
| `ha0_ext` — constant `a0` **and** `κ0` at t0, the echo control | **0.2874** | `os − ha0_ext` **+0.0101** [−0.0050, +0.0273] **not separated (a TIE, on the wrong side of zero)** |
| `os` (refcv4b) — the incumbent | **0.2975** | — |
| `ha` — hold-action | **0.2996** | `os − ha` **−0.0021** [−0.0178, +0.0154] **not separated** |
| `os_navshuf` — pairing broken, nav marginal preserved | **0.3013** | — |
| `os_navzero` — nav stripped, the DEPLOYMENT arm | **0.3928** | `os_navzero − ha0_ext` **+0.1054** [+0.0874, +0.1241] **separated WORSE** |
| `ha0` — constant velocity, the straight-line floor | **0.6723** | `os − ha0` **−0.3748** [−0.4319, −0.3203] separated |
| *refcv3 `os` @40,284* | *0.4419* | *refcv4b − refcv3 **−0.1444** [−0.1647, −0.1227] separated* |

### 3.1 ⭐ PASS — **P1, the only criterion that decides "progress"**

> **`refcv5 os − ha0_ext` on `ade_m` is NEGATIVE, its paired CI excludes zero, and
> `|delta| > F(ade_m)`.**

This is the bar refcv4b **failed**. Clearing it means refcv5 is the first REF-C arm that beats the
strongest model-free control on the surface it was trained for.

### 3.2 ⭐ PARTIAL — **P2, the lever works but the arm does not drive**

> **`refcv5 os − refcv4b os` is NEGATIVE, separated, and `|delta| > F(ade_m)` — while P1 does
> NOT hold.**

⛔ **This is reported as `LEVER-SUPPORTED / NOT-DRIVING`, and the headline sentence says so.**
Beating refcv4b while still only tying `ha0_ext` **is not driving**: two arms that both tie a
constant-control are two arms that have not yet shown skill, and a delta between them is a delta
between two ties. *(refcv4b already cut refcv3's deficit 7× and still did not cross this bar; a
second 7× on the wrong side of it would be the same non-result at higher precision.)*

### 3.3 ⛔ FAIL — F1 (NULL) and F2 (REGRESSION), as written in §0

* **F1 NULL** — `refcv5 os − refcv4b os` not separated, **or** `|delta| ≤ F(ade_m)`. ⇒ the
  control-space DDIM sampler does not move driving at this scale on this surface. ⚠️ Under
  `H-ESTIM-SEED-1` a null on one seed is **not** a refutation of the mechanism; it is a
  refutation of *this arm at this budget*, and the SPEC says so before the number exists.
* **F2 REGRESSION** — separated POSITIVE. ⇒ the sampler costs driving quality; the next lever is
  the ladder (`--sampler-infer-t` / `--sampler-steps`), not more training.

### 3.4 ⛔ VOID — not "negative", VOID
Any of: **G-ARGV** fails · **G-STOCH** fails in either direction · the deliberate-regression arm
`frames_blind` does **not** regress · the model-free arms are not bit-identical across arms ·
`paired(a, a)` is not exactly zero · the window grids do not match (`ws`/`eid` unequal). A panel
whose instruments have not been shown able to FAIL certifies nothing.

### 3.5 THE DEPLOYMENT BAR — reported always, gating never
> **`refcv5 os_navzero − ha0_ext`.** refcv4b read **+0.1054 separated WORSE** — i.e. **the entire
> refcv4b margin over the echo control was supplied by an ORACLE nav input.**

refcv5 moves no nav mechanism, so this is **not expected to close** and it is **not a bar**. It is
reported because a P1 pass that evaporates when the oracle is withdrawn is a P1 pass about an
oracle. ⭐ **`os_navpred` is rolled in the same pass** (`--with-navpred`, free — it rides the
nav-zero forward and maps the model's own route argmax through `refb_labels._ROUTE_TO_NAV`), which
brackets R2's gain between `os_navzero` and `os` at **zero extra GPU**.

---

## 4. THE FOUR FAMILIES — never ADE alone, never pooled

Each family carries its **estimator** (paired episode-cluster bootstrap) and its **CI** on the
**same** windows as the ADE beside it. A family that genuinely cannot be computed is refused
**per family, with the reason and the n** — a missing family is a WORK ITEM, not an excuse.

### 4.1 LATERAL — ⭐ **THE MECHANISM'S OWN SHARPEST PREDICTION, AND IT IS COMMITTED AS A BAR**

Read on **CURVATURE MAE with the straight-line floor beside it**. refcv4b, MEASURED:

| arm | curvature MAE (1/m) | cross-track MAE (m) |
|---|---|---|
| `os` | **0.008097** | **0.0979** (best of any arm) |
| `ha` | 0.004030 | 0.1226 |
| `ha0_ext` | **0.003712** | 0.1070 |
| **`ha0` — the straight-line floor** | **0.006802** | 0.3132 |
| *refcv3 `os`* | *0.008815* | *0.1084* |

⛔ refcv4b tracks curvature **WORSE THAN A PLAN THAT NEVER STEERS** while having the best
cross-track of any arm — a **SHAPE** defect, not a positional one. A cross-track-only read misses
it entirely, which is why cross-track may never be quoted alone.

> ### ⭐ COMMITTED, BEFORE THE DATA: **L1 — `refcv5 os` curvature MAE falls BELOW `ha0`'s 0.006802.**
> The sampler's stated purpose is *"the anchored Gaussian in CONTROL space, so every sample
> re-rolls through the kinematic model and is flyable by construction rather than by penalty."*
> **Curvature MAE is the metric that measures flyability.** If a mechanism that samples in control
> space leaves curvature at or above the straight-line floor, its headline claim is **REFUTED at
> the metric it names**, whatever ADE does.
> * **L1 SUPPORTED** — curvature MAE `< 0.006802`, paired margin vs refcv4b separated NEGATIVE and
>   `|delta| > F(curvature)`, **and cross-track has not degraded past `ha0_ext`'s 0.1070.**
> * **L1 REFUTED** — curvature MAE `≥ 0.006802`, or the paired margin is not separated, or it
>   improves only by trading away cross-track.
> ⚠️ Buying curvature by flattening the path is **not** a pass: L1 requires cross-track to hold.

Also reported: heading MAE (refcv4b `os` **1.2950°**), yaw-rate MAE (**1.7309 °/s**).

### 4.2 LONGITUDINAL — reported in full, **committed as an EXPECTATION, not a bar**

refcv4b, MEASURED: `os` speed MAE **0.2909 m/s** (bias +0.0327), target-speed accuracy @0.5 m/s
**0.8317**, along-track MAE **0.2555 m**; `ha`/`ha0_ext` **0.2540 / 0.8662**; `ha0` **0.4880 /
0.7047**; refcv3 **0.4516 / 0.7135**. ⛔ **`os − ha` speed MAE = +0.0368 [+0.0197, +0.0557],
separated WORSE** — ADE ties while the longitudinal family says the model is behind the control.

> **COMMITTED EXPECTATION:** WP-4 is a **decoder-side** mechanism and does **not** address the
> measured longitudinal defect, which `M84` localised to the **representation** (the lead's
> position is decodable, its **closing rate** is a clean null on every arm, +0.0061 against
> position's +0.4145 [+0.2018, +0.6120]). ⇒ **A large longitudinal movement in EITHER direction
> is a surprise that must be explained before it is quoted, not celebrated.** Specifically: an
> `os − ha` speed-MAE that flips to separated-BETTER without any perception change is more likely
> a shift in the speed *distribution* of the emitted fan than new longitudinal skill, and the
> landing must check the speed bias and the target-speed accuracy before making any such claim.

Distance-keeping is reported with its censoring: refcv4b `status OK`, n = 1,225 of 4,823 over 67
episodes, mean min headway 28.47 m, min time-gap 4.12 s (n = 1,156), min TTC 24.83 s — ⚠️ **753
of 1,225 windows never close on the lead and are censored at `TTC_CAP` 30 s; `n_closing = 472`.
The TTC mean is never quoted without it.**

### 4.3 TACTICAL — turn recall **beside** ADE, and a committed FAIL-SHAPE

refcv4b, MEASURED (v7.2 **factored kin3** labels from the trainer's own labeller): `os` LAT acc
0.9583 / **κ 0.8289**, `turn_left` **0.813** (n 251), `turn_right` **0.886** (n 396); LON acc
0.8258 / **κ 0.5178**, `brake_stop` **0.539** (n 631), `accelerate` **0.504** (n 538). `ha` LAT κ
0.7374 / LON κ **0.6071** / brake **0.697** / accel **0.649**. `ha0_ext` LAT κ 0.7548 / LON κ
0.6071. `ha0` **every κ 0.0000, both turn recalls 0.000**.

> ### ⭐ COMMITTED, BEFORE THE DATA: **T1-SHAPE — an ADE win with turns falling is a REGRESSION WEARING A WIN.**
> If `ade_m` improves (P1 or P2) **while** `turn_left` **or** `turn_right` recall falls by more
> than `F(recall)`, the result is reported as **`FAIL-SHAPE`**, and the headline says the arm
> bought ADE by driving straighter. *(The precedent is not hypothetical: refav1's best-ADE arm
> executes **zero** turns.)*

⛔ **The `|dyaw| > 0.15` turn gate is NOT used anywhere in this panel** — it demands R 19 m at
v0 1.40 m/s and **the human fails it 3 of 9**. A threshold carries its regime.
Also reported: `goal_point_error` (refcv4b `os` **0.6365 m**, `ha0_ext` 0.6323, `ha0` 1.4029),
goal-bearing MAE (1.5475°), and the **selection profile** — refcv4b uses **50 of 117** anchors,
modal anchor **48.79 %**, entropy ratio **0.4524**. ⚠️ A sampler that *collapses* the fan can
improve ADE while destroying diversity; the selection profile is read **before** the verdict.

### 4.4 STRATEGIC — present, with its anti-echo control, or the block is VOID

refcv4b, MEASURED: route accuracy **0.7786** [0.7205, 0.8324], **κ 0.4852** [0.4057, 0.5671],
n = **3,622** route-labelled windows / 128 episodes (1,201 excluded, no route label), majority-class
rate 0.6742, per-class recall `route_left` 0.357 (n 470) / `route_straight` 0.966 (n 2,442) /
`route_right` 0.411 (n 710), **nav echo index 0.6405** *(the CORRECTED value; the record originally
published 0.1621 under a route/nav index type error — `RETRACTION_LOG.md` 2026-09-06)*.

> **COMMITTED:** the route head must be re-shown **nav-INDEPENDENT** on refcv5 —
> `paired_true_minus_shuffled_accuracy` exactly **0.0000, CI [0, 0]** — **and the anti-echo
> control must have POWER**, i.e. the shuffle must actually change a material fraction of tokens
> (refcv4b: **2,406 / 4,823 = 49.89 %**). ⛔ **A zero from a shuffle that changed nothing is a
> degenerate instrument, not an identity, and the STRATEGIC block is then VOID.** On the changed
> subset the head must follow the TRUE label over the shuffled token (refcv4b: **0.7437 vs
> 0.3370** on 1,736 changed; **0.7124 vs 0.1739** on the 1,311 mutually-exclusive ones).

⚠️ **KNOWN GAP, DECLARED IN ADVANCE (W-1):** on refcv4b the LATERAL and TACTICAL families carried
per-arm CIs but **no paired margins** — `_intervals_complete: false`. **This SPEC closes it
offline:** `raw/refcv5_landing_analysis.py` computes paired margins for LATERAL directly from the
banked `ep*.npz` paths and for TACTICAL/STRATEGIC from `decisions/ep*.npz`
(`lat_label` · `lon_label` · `route_label` · `lat_pred_nav_zero` · `lon_pred_nav_zero` ·
`route_pred_nav_zero` · `sel_idx` · `nav_cmd*`), at **zero GPU**. If a family still cannot be
paired, it is refused **with its reason and its n**.

---

## 5. THE CONTROLS — each must read a KNOWN value

| skill slot | this panel's arm | the value it must read |
|---|---|---|
| **constant_only** | `ha0` (constant velocity at the measured v0) **and** the estimator self-test `paired(a, a)` | `ha0` is the straight-line floor and reads **every κ = 0.0000 and both turn recalls 0.000 by construction**; `paired(a,a)` must read **delta 0.0, CI [0,0], separated False** |
| **raw_input_floor** | ⭐ **`ha0_ext`** — constant `a0` **and** `κ0` at t0 | the "raw input" of this problem is the **measured ego state**; `ha0_ext` is what it yields with **no learning at all** (refcv4b **0.2874 m**, better than every learned arm). **A planner that does not beat it has added nothing.** `ha` (hold-action) is reported beside it |
| **deliberate_regression** | ⛔ **`--ablate frames_blind`** — every observed frame replaced by its own scalar mean ⇒ an echo **by construction** | must be **separated WORSE** than `os` (refcv4b: **1.0491 m**, paired **−0.7515** [−0.8166, −0.6899]), **and** `ha`/`ha0`/`ha0_ext` must return **bit-identical**. If the regression does not regress, the panel is **VOID** |
| **inference_replicate** ⭐ *(added by this SPEC; the rig demands it)* | `os_R1`, `os_R2` — §2.2 | must **differ** from `os_R0` on a material fraction of windows (else G-STOCH VOID) and supplies `F(m)` for every metric |

⚠️ The skill's tiny-rig ladder (**G-RANK / G-DECODE**) is **not applicable here and is not
claimed**: this is a full-scale 108 M-parameter driving arm on the B1 corpus, evaluated at
**G-DRIVE** only. Saying so is the honest form; silently omitting the first two rungs is not.
⛔ And the SPEC does not pretend refcv5 was tiny-rig-validated first — **it was not**, which is
recorded as a deviation in §9 B4 rather than glossed.

---

## 6. ⛔ METRICS EXCLUDED, AND WHY — so nobody quotes them off the landing dump

| excluded | reason |
|---|---|
| **`oracle_sel` (T0)** | ⛔ **INVALID on a v0-conditioned vocabulary.** `refcv3_arm.py:1492` binds `anchors_bank = model.core.decoder.anchors` for the `a_star` argmin, which the trainer's own comment (`refc_v3_train.py:538-545`) forbids: with a v0-conditioned vocabulary `decoder.anchors` is the family rolled at the **reference speed**, not this window's fan. refcv5 runs `--anchor-v0-conditioned`, exactly as refcv4b did, so the defect applies unchanged. On refcv4b it made the "ceiling" read **1.2154 m**, i.e. **+0.9179 separated WORSE than the arm it bounds**. ⇒ **`--with-oracle-sel` is NOT passed.** |
| **`anchor_acc`** | same mechanism (refcv4b read 0.0993 against a quoted chance of 0.008547) |
| **`sel_agrees_oracle`** | same mechanism (refcv4b read 0.0993) |
| **`--ablate sel_refined`** | not an arm here. MEASURED on refcv4b: **0.0259 m separated WORSE**, flips 30 % of picks, collapses the fan 18 → 16 anchors, pushes the longitudinal decision to the majority class. ⭐ And it is **structurally out of scope for this lever**: `refc.py:1837-1840` records that `SelectionConfig.refined` defaults **False**, so *the sampler deliberately does not touch the ranked score*. Enabling it would add a second mechanism to a one-variable arm |
| **the `\|dyaw\| > 0.15` turn gate** | a threshold quoted outside its regime — it demands R 19 m at v0 1.40 m/s and the human fails it **3 of 9** |
| **any PARITY-corpus LEVEL comparison** | ⛔ **this arm is B1 (4,572 clips), NOT the parity corpus** `physicalai-train-e438721ae894` (2,376 eps, skip-hash `f09e44db`). The v2 cache does not name a parity key and the trainer flags it itself. ⇒ refcv5-vs-refcv4b is a valid **ARM** delta; a **LEVEL** against any parity arm is **inadmissible** |
| **`nav_compliance` vs refcv3** | refcv3's banked dumps cannot produce it (six sidecar keys absent from all 141 files in both dumps) — a **W-3** re-roll, not a refcv5 number |

⚠️ **The anchor units are OPERATOR-ASSERTED, not artifact-declared.** The live 117-anchor bank
declares **no** `control_units`; the run carries `--anchor-control-units alat` and records
`control_units_source: cli-override-legacy-file`. **Column 1 of `controls` is LATERAL
ACCELERATION (m/s²), on the operator's word, not the file's.** Read as curvature the same tensor
gives `a_lat = v²κ = 36² × 3.0 = 3,888 m/s² = 396 g` with 104/117 anchors outside a μ = 0.7
friction circle; read correctly it is **0.31 g, 0/117**. A declared bank exists for the *next*
rung (`pod:/workspace/anchors_117_alat_declared.pt`) but **not** for this arm, whose bank cannot
change mid-run. **Any anchor-geometry number in the landing carries this sentence.**

---

## 7. THE ARMS AND THE EXACT COMMANDS

### 7.0 ⛔ GATE 0 — before a single GPU second

1. **The run is finished and the pod is free.** `summary.json` with `"done": true`; `metrics.jsonl`
   max step **40284** read **independently of the marker**; supervisor **and** trainer processes
   gone (checked by explicit PID, never `pgrep -f`, which self-matches the ssh command).
   ⛔ **`tanitad-a40` IS the training pod. No eval starts while it trains** — "never add GPU/RAM
   load to a pod that is training, and never eval on a training pod."
2. **Checkpoint identity:** `ckpt_40284_FINAL.pt`, its md5 recorded, its own `step` field = 40284.
3. **G-ARGV** (§1) — the config diff, asserted.
4. **Preflight import probe** — `python3 raw/refcv5_landing_analysis.py --preflight` must print
   `PREFLIGHT OK`. ⛔ *MEASURED: `taniteval` was absent from this pod entirely and
   `import taniteval.ci` would have killed the landing eval AFTER ~45 h of paid compute. A
   2-second probe at startup is the durable fix.*
5. ⛔ **`--analyze-only <dump_dir>` before re-running ANYTHING.** The rollout is the only expensive
   part; an analysis-time failure is recoverable at zero GPU. *(That trap has already destroyed a
   paid-for 2-arm / 40-episode rollout in this programme.)*
6. **refcv4b's banked dump** `pod:/workspace/eval/refcv4b_t1_dump` still present and readable. If
   it is gone, refcv4b is re-rolled — legitimate, because refcv4b **is** deterministic at
   inference (`refc.py:2110-2111`) — and the re-roll's `ha`/`ha0`/`ha0_ext` must come back
   bit-identical to the banked ones.

### 7.1 The arms

| arm | tier | why it is here |
|---|---|---|
| `os` | T1 (`tier_ruling: UNRULED`) | the deployed one-shot planner: `out["traj"]`, the model's own `sel_score_v3` pick — ⛔ never `a_star` |
| `ha` | T1 | hold-action control |
| `ha0` | T1 | constant velocity at the measured v0 — the straight-line floor, `constant_only` |
| **`ha0_ext`** | T1 | ⭐ the echo control / `raw_input_floor` — constant `a0` **and** `κ0` at t0 |
| `os_navshuf` | T1 | nav PAIRING broken, marginal preserved |
| `os_navzero` | T1 | nav SIGNAL removed — the deployment arm |
| ⭐ `os_navpred` | T1 | the model's OWN route argmax fed as nav (R2's discriminating experiment, free) |
| ⛔ `oracle_sel` | — | **ABSENT BY DESIGN** — §6 |
| `ol` | — | **ABSENT with its reason**: refcv3/v4/v5 consume no recorded actions (`refcv3_arm.py:26`) |

### 7.2 The primary roll — `os_R0`

```bash
ssh -n tanitad-a40 'cd /workspace/TanitAD && OMP_NUM_THREADS=6 \
 PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/taniteval PYTHONIOENCODING=utf-8 \
 python3 taniteval/tools/refcv3_arm.py \
  --ckpt /workspace/experiments/refcv5-ddim-b1-v72-40k/ckpt_40284_FINAL.pt \
  --config /workspace/experiments/refcv5-ddim-b1-v72-40k/config.json \
  --episodes /root/data/eval \
  --labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
  --nav-source v72 --grid 2s --action-units steer --device cuda \
  --window-stride 5 --with-navpred \
  --lead-block /workspace/eval/b1_eval_lead_block.npz \
  --n-boot 2000 --seed 0 \
  --dump-dir /workspace/eval/refcv5_t1_dump_R0 \
  --out /workspace/eval/refcv5_t1_R0.json \
  --tiers os=T1,os_navshuf=T1,os_navzero=T1,os_navpred=T1'
```

⚠️ `OMP_NUM_THREADS=6` is not optional: torch spawns ~113 threads per process and concurrent arms
then make **no progress at all** (MEASURED: 7 arms at GPU sm 0–6 % for **50 minutes**; the same
arm finished in 232 s with the variable set).
⚠️ `--window-stride 5` is chosen for **COMPARABILITY, not cost** — it reproduces the 4,823-window
grid refcv3 and refcv4b were scored on. A different stride makes this a different experiment.

### 7.3 The replicate rolls — `os_R1`, `os_R2`

**Byte-identical to §7.2 except `--dump-dir` / `--out`** (`…_R1`, `…_R2`). The landing **audits**
this by diffing the three `manifest.json` argv records; an argv difference other than those two
paths **VOIDS** the floor. *(The `A0b_replicate` precedent: same flags, same seed, zero levers
moved, verified by argv audit — and it still produced "separated" differences on 3 of 18 metrics.)*

### 7.4 The deliberate regression

```bash
  ... --ablate frames_blind \
  --dump-dir /workspace/eval/refcv5_frames_blind_dump \
  --out /workspace/eval/refcv5_frames_blind.json
```
⛔ One regime per `--dump-dir` — the tool refuses a mixed directory (`refcv3_arm.py:1532-1545`).

### 7.5 The PI-binding vision-only arm
```bash
  ... --ablate ego_zero --dump-dir /workspace/eval/refcv5_ego_zero_dump \
  --out /workspace/eval/refcv5_ego_zero.json
```
refcv4b read **1.1310 m** [1.0294, 1.2451] here, paired **−0.8335 separated** — ⛔ **the win is
NOT vision-only.** The number to report is the **step-40,284 withheld-vs-kept gap** and whether it
closed relative to refcv4b's. `--ego-dropout 0.5` is unchanged between the two arms, so a change
in this gap is attributable to the sampler.

---

## 8. THE ANALYSIS SCRIPT — written NOW

`raw/refcv5_landing_analysis.py`. It is the landing's only computation and it runs at **zero GPU**
against banked dumps.

* **`--preflight`** — a 2-second import probe (`numpy`, `taniteval.ci`) that exits non-zero with a
  named module before any rollout is started. **GATE 0 step 4.**
* **`--floor R0 R1 [R2 …]`** — the inference-seed floor `F(m)` per metric, plus **G-STOCH** in both
  directions.
* **`--pair A B`** — a paired episode-cluster bootstrap between two dumps, with the grid assertions
  (`ws`, `eid`) **before any number**, refusing rather than aligning.
* **`--families A [--vs B]`** — the four families per arm and, when `--vs` is given, the **paired
  margins** that closed W-1, including LATERAL curvature with `ha0` beside it and TACTICAL turn
  recall.
* It prints **`QUOTABLE` / `WITHIN-NOISE` / `ns`** per row against the floor, so a margin can never
  be quoted without the construction that produced it.
* ⛔ It **refuses** `overlapping_holdout_se` — it imports only `paired_episode_cluster_bootstrap`
  and `episode_cluster_bootstrap`. The preflight asserts the forbidden symbol is *present and
  never called*, so a module swap is visible rather than silent.
* ⚠️ **ASCII-only output.** Non-ASCII in `print()` is fatal under cp1252.

### 8.1 ⭐ IT HAS ALREADY BEEN RUN, AND EVERY GATE HAS BEEN PROVEN ABLE TO FAIL

`raw/selftest.log`, MEASURED 2026-09-06 this turn. **PART A** runs against synthetic
`refcv3_arm`-shaped dumps (`raw/mkfixture.py`); **PART B** runs against the **REAL banked dump**
at `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-refcv3-arm/raw/fixture_dump`.
**6 negative controls, all firing:**

| # | the control | what it read |
|---|---|---|
| A3 | two **bit-identical** rolls fed to `--floor` | `G-STOCH … 0 of 77 windows -> VOID`, **exit 1** |
| A4 | a replicate with a **lever moved** | `REPLICATE AUDIT: FAILED -> THE FLOOR IS VOID`, exit 1 |
| A5 | two dumps on **different window grids** | `REFUSED: R0 has 77 windows, SHORT has 22 — … aligning them would be a fiction` |
| A6 | a harness JSON carrying a **wrong** ADE | `MISMATCH <- the DERIVED family rows below are NOT trustworthy` |
| A7 | a dump with **no `decisions/`** | `REFUSED TACTICAL/STRATEGIC  no decisions/ep*.npz under …` |
| **B3** | ⭐ **on a REAL manifest**: `model.decoder_steps` moved, **plus** `wallclock_s` and `first_forward_s` moved | `REPLICATE AUDIT FAILED: … at 1 key(s): /model/decoder_steps` — it names **only the lever** and correctly ignores the two fields that must differ between rolls |

Positive side: `PREFLIGHT OK`; `paired(A, A)` = **delta 0.0000000000, [0, 0], separated False**;
ADE reconciliation **RECONCILED**; the floor turns bare `separated` into `QUOTABLE` /
`WITHIN-NOISE` / `ns` per row.

### ⛔ PART B EXISTS BECAUSE PART A COULD NOT HAVE CAUGHT THESE TWO — and it found both
Running the instrument against the **schema the tool actually writes** (rather than the one
inferred from source) refuted two of my own assumptions in one pass:

1. ⛔ **A real `refcv3_arm` manifest carries NO `argv`.** MEASURED: its top-level keys are
   `_unverified · absent_arms · action_units · arm_meaning · arms · corpus · doc · episodes ·
   fed_conditionings · first_forward_s · grid · head_conditionings · hold_action_rule ·
   hold_v0_rule · model · nav_null · nav_shuffle · sidecar_schema · t1_definition · tier_ruling ·
   tiers · tool · wallclock_s`, and `model` holds `ckpt`/`cfg`/`decoder_mode`/`decoder_steps`/
   `n_anchors`/`config_json`/… — **no `argv` anywhere.** The first version of the replicate audit
   read `manifest["model"]["argv"]` and would therefore have printed **INCONCLUSIVE forever on
   every real dump**. ⭐ **A gate that cannot fire is not a gate.** It now compares a **resolved
   provenance fingerprint** — strictly stronger than what an operator typed — excluding
   `wallclock_s` and `first_forward_s`, which must legitimately differ. *(The SPEC's §7.3
   "diff the three manifest argv records" is superseded by this; the audit is the same
   obligation, executed against a field that exists.)*
2. ⭐ **The decisions sidecar carries `*_pred_nav_TRUE` as well as `*_pred_nav_zero`** — all of
   `lat/lon/route_pred_nav_{true,shuffled,zero}` are present. The nav-**true** reading is the
   **deployed** one; the script now prefers it, falls back to `nav_zero`, and **names which key it
   used in every row**, so a reader never has to guess the conditioning behind a number.

⚠️ The self-test proves the **instrument**. The real dump is an **untrained test fixture** (its
~9 m ADE is the fixture's, not a model's). Nothing in the log is a refcv5 number.

---

## 9. BLOCKERS — named, with what would unblock each

| id | blocker | what unblocks it |
|---|---|---|
| **B1** | ⛔ **WP-6 (`--agents oracle`) cannot run.** The precondition *"`--agents oracle` is the first rung and needs no detector"* was **REFUTED at preflight**: true about the DETECTOR, false about the JOIN — the oracle's tokens **ARE** the ground-truth boxes, so the forward raises without `--agent-join`. No agent join exists on this pod, and the only one the programme holds is **parity**-scoped (2,308 clips ≈ 4 % of B1) | a **B1-scoped `obstacle.offline` join** (DataFlyWheel). ⛔ Not planned around |
| **B2** | ⛔ **V2, the training-run variance, is unpriced.** refcv5 is a single-seed arm; `H-ESTIM-SEED-1` gives a ~17 % false-positive rate for `separated` on a zero-lever replicate | a second 40,284-step run at a different `--seed` — **~44 h of A40**. **PI decision on compute** |
| **B3** | ⛔ **`refcv3_arm.py:1492` computes `a_star` from `decoder.anchors`** instead of `out["anchor_bank"]` (W-4). Until fixed, **no selector metric is scoreable** on any v0-conditioned arm | ~2 lines + a T0 re-roll. ⛔ **This stream does not own that file — ESCALATED to the Master Mind / Benchmarks, not edited here** |
| **B4** | ⚠️ **DEVIATION, DECLARED:** the skill's rung-1 rule is *"never validate a design on a full-scale run"*, and refcv5 **is** a full-scale run. It was launched before this SPEC existed | nothing — it is recorded, not repaired. Its consequence is that a NULL (F1) is a statement about **this arm at this budget**, not about the mechanism |
| **B5** | ⚠️ `--sel-refined` **does not exist in this trainer** (`grep -c sel_refined` = **0** against a same-breath control `sel_accel_max` = **5**), so the brief's prohibition is satisfied structurally rather than by choice | n/a — recorded so a later reader does not go looking for the flag |

---

## 10. RANKED EXECUTION ORDER — a kill at any point still leaves value

1. **GATE 0** (§7.0) — including the preflight import probe and `--analyze-only`;
2. **`os_R0`** — the headline four-family read + every model-free control;
3. ⛔ **`frames_blind`** — without it nothing above is admissible;
4. ⭐ **`os_R1`** — the inference-seed floor exists after this and **not before**; ⛔ **no margin is
   quoted until it does**;
5. **the paired reads** vs refcv4b's banked dump: P1 (`os − ha0_ext`), P2 (`os − refcv4b os`), L1
   (curvature), T1-SHAPE (turn recall);
6. **`os_R2`** — turns the floor from a single draw into a max-of-three;
7. `ego_zero` — the PI-binding vision-only number;
8. if GPU remains: `e9_off`, `h19_off`, `e7_off`, `gstr_zero` — ⚠️ each is an **eval-time**
   intervention on ONE checkpoint, so V2 does not enter, and the correct claim form is *"this
   switch changes this metric on this checkpoint"*, never *"this lever is worth X in refcv6"*.

---

## 11. ⭐⭐ THE ONE LINE, COMMITTED BEFORE THE DATA EXISTS

> **refcv5 counts as PROGRESS only if `os − ha0_ext` on `ade_m` is separated NEGATIVE by more than
> refcv5's own measured inference-seed floor — the bar refcv4b failed at +0.0101 — with curvature
> MAE below `ha0`'s 0.006802 and turn recall not falling. It is a NULL if `os − refcv4b os` is
> unseparated or smaller than that floor: a 44-hour arm whose one mechanism moved nothing a second
> roll of the same checkpoint would not have moved by itself.**
