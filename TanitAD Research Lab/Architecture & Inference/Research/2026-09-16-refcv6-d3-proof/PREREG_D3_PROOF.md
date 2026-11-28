# PREREG — the refcv6 D3 proof package

**Written 2026-09-16 (Europe/Berlin), BEFORE any number of this package was read.**
Registry id `E-D3-PROOF-1`. Authority: `Project Steering/REFCV6_CLARIFICATION.md` §2.4 (the
package the PI approved today with *"run the d3 proof package"*), and
`SPEC_REFCV6_V2.md` §8 (*"Before any pod hour: the D3 proof package … on the dev box"*).

⛔ This file is frozen at the moment of its first `git add`. Anything learned after that is
written in `RESULT.md`, never back into this file.

---

## 0. The one question

**Does refcv5-v2's planner demonstrably USE the scene, and can a frozen probe read boxes out of
its trunk?** §2.2 of the clarification splits it in two and says both halves are needed:

1. **use** — a causal intervention on the image moves the plan in the right direction
   (T-A, T-B, T-G, T-H). A probe that decodes the lead proves nothing about driving if
   masking that lead leaves the plan unchanged.
2. **extraction** — a probe on the frozen trunk beats every zero-information control
   (the box panel, on s32 AND s16).

---

## 1. Scope and budget

| | |
|---|---|
| model | refcv5-v2 `ckpt.pt` step 40,284, `C:/Users/Admin/refcv5v2_final/` (+ `config.json`) |
| harness (rung 1) | `taniteval/tools/refcv3_arm.py` — the REF-C T1 adapter, four families |
| ⛔ code tree for every roll | **`C:/Users/Admin/refcv5cmp/repo`** — the tree that produced the banked `refcv5-v2` dump. MEASURED today: `stack/tanitad/refs/refc.py` has **321** content lines of difference against the branch tip, so rolling on the tip would silently break the pairing with the banked baseline. `refcv3_arm.py` and `v2_dataset.py` are content-identical across all three trees (0 lines, line endings only) |
| grid | 2 s, K = 4, `--window-stride 5`, `--action-units steer`, T1 |
| n | **4,823 windows / 141 episodes** — the same grid as every banked refcv5-v2 number |
| tier | **T1** (self-action OPEN loop, per `open-vs-closed-loop ruling`). No arm here is closed loop |
| budget | ≤ 12 GPU-hours on the dev-box RTX 4060, ONE GPU job at a time |
| clip ids | ⛔ **sha12 = `sha256(clip_id).hexdigest()[:12]` only**, everywhere in the repo |

Already banked, so they cost 0 GPU-h and are the baseline of every paired read:

| artifact | what it is |
|---|---|
| `C:/Users/Admin/refcv5cmp/out/refcv5-v2_dump/` + `.json` | the **FULL `os`** arm, `--infer-seed 0` |
| `C:/Users/Admin/refcv5cmp/out/refcv5-v2-seed1_dump/` + `.json` | the **inference-seed replicate**, `--infer-seed 1`, everything else identical |

⚠️ **refcv5-v2 runs `--sampler ddim`, which DRAWS NOISE AT EVAL** (`refc.py:1926`). The
inference-seed floor is therefore REAL and is reported beside every effect in this package.
**The seed floor for a metric `m` is defined here as `|m(seed0) − m(seed1)|` on the FULL arm.**

---

## 2. Rung 1 — does the planner USE the scene?

### 2.1 T-A — the VOID gate (`frames_blind`, `ego_zero`)

Two separate rolls, each one registered ablation, each against the banked `os`:

| arm | mechanism (the harness's own registry, `refcv3_arm.ABLATIONS`) |
|---|---|
| `frames_blind` | every observed frame replaced by the window's own scalar mean |
| `ego_zero` | `ego_state[:,4] = 0` **and** `v0 = None` at the core |

**Registered controls that must read KNOWN values — any failure VOIDS the panel:**

| control | known value | why |
|---|---|---|
| `ha`, `ha0`, `ha0_ext` under `frames_blind` | **BIT-IDENTICAL** to the FULL run | the model-free arms read no frames |
| `ha0` under `ego_zero` | BIT-IDENTICAL | `ego_zero` is an arm intervention, not a control intervention |
| `paired(os, os)` — the banked dump against itself | **EXACTLY 0.000000** | a paired estimator that is not 0 against itself measures nothing |

**Registered reads:** ADE and the four families, `os_blind − os` and `os_egozero − os`, paired
episode-cluster bootstrap, 2,000 resamples, seed 0.

**Reference (PUBLISHED-PRIMARY, this programme):** on **refcv4b** blanking the frames moved ADE
**0.2975 → 1.0491**. It has NEVER been run on refcv5-v2.

**Bar.** T-A is a *gate*, not a hypothesis: it has no pass/fail threshold of its own. It is
registered as **reported, with the controls binding**. The number it produces is the ceiling on
every later "the planner uses the image" claim — an `os_blind − os` that is not separated would
mean the planner's image path is not load-bearing at all, and would make T-B's expected effect
zero by construction.

### 2.2 T-G — attention attribution

Hook the operative decoder's cross-attention over the 160 image tokens with `need_weights=True`,
take the **selected anchor's** row, and compute

```
concentration = (attention mass on the lead vehicle's token columns) / (those columns' area share)
```

on windows with a lead at ≤ 30 m.

| control | known value |
|---|---|
| a **uniform** attention row, scored by the same function | **EXACTLY 1.0** |
| a random equal-width column set, same windows | reported beside it |

**Bar:** registered as **reported with its control**. A concentration ≫ 1 is evidence of
addressing; ≈ 1 is evidence of none. No threshold is registered because no prior measurement of
this quantity exists on this model to calibrate one, and inventing a bar after seeing the scale
is the failure this file exists to prevent.

### 2.3 T-H — the decision probe

Regress the **planned 0–2 s longitudinal acceleration** on `(gap, closing rate, v0, a0)` over the
lead windows, with an **episode-cluster bootstrap** (2,000 resamples, clusters = clips).

| control | known value |
|---|---|
| `ha0_ext` plans `accel ≡ a0` by construction | its **gap coefficient is EXACTLY 0** |
| `ha0` plans `accel ≡ 0` | its gap coefficient is EXACTLY 0 |
| the **human** (GT future) on the same windows | reported as the reference coefficient |

**Bar:** the model's gap coefficient is **separated from 0** (95 % CI excludes 0) and has the
**same sign as the human's**. Registered as PASS/FAIL.

⛔ This test costs **0 GPU** — it reads the banked dumps' `plan_full_nav_true` and `gt_future_ext`
and the lead block. It is run on BOTH banked seeds so its own seed floor is visible.

### 2.4 T-B — the LEAD MASK ⭐ the deciding test

**Intervention.** For every window, project each **lead** cuboid at ≤ 30 m through that clip's own
front-wide extrinsics into **all** frames of the stack (the 8 provider rows × 3 channel groups of
the D-015 stack, each group at its own raw frame) and fill the covered pixels with the **frame
mean** (the same fill `frames_blind` uses, so the two arms differ only in support).

⚠️ **The ~1 m cuboid z offset is corrected** (`Project Steering/RETRACTION_LOG.md`,
`RETR-2026-09-13-SAM3MAP-A2-GHOST-GROUND`: *"PhysicalAI's tracked boxes carry z about 1 m under
the LiDAR ground; a display mask built from z − h/2 blanked the road in front of every vehicle
and the PI asked why"*). The correction and the height assumption are stated in `RESULT.md` with
the measured mask area, and a rendered still is banked.

**The comparison arm is an AREA-MATCHED RANDOM OFF-AGENT MASK**, drawn per window at the same
pixel area, at a column drawn away from every labelled agent. The effect of interest is
**lead − random**, never lead − unmasked.

| control | known value |
|---|---|
| a **zero-area** mask | the plan is **BIT-IDENTICAL** to the FULL run |
| model-free arms `ha`/`ha0`/`ha0_ext` under any mask | **BIT-IDENTICAL** |
| inference-seed replicate | the lead-mask arm is re-rolled at `--infer-seed 1`; its floor is reported beside every effect |

**Reads:** minimum time gap, headway, min TTC, planned deceleration, speed MAE in closing windows
(`taniteval.lead_metrics` / `four_families._distance_keeping`, through the harness — one
implementation, not a second).

**⭐ REGISTERED PASS (all three, on minimum time gap):**

1. the **lead − random** effect is **separated** (paired episode-cluster bootstrap 95 % CI
   excludes 0),
2. it is **≥ 3 ×** the inference-seed floor of the same statistic,
3. it is in the **unsafe direction** — masking the lead makes the plan approach it *more*
   (minimum time gap **falls**, planned deceleration **falls**, min TTC **falls**).

**REGISTERED FAIL:** a lead mask ≈ a random mask. The planner is then not reading the lead,
whatever any probe decodes.

---

## 3. Rung 2 — the BOX probe on the cached frozen tokens

**Input.** The cached frozen refcv5-v2 tokens of the **139** eval clips,
`C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens/` — `tokens_s32_fp16.npy` `[N,704,8,20]`,
`tokens_s16_fp16.npy` `[N,352,16,40]`, `pix64_u8.npy`, `index.npz` (N = 27,664 stacked rows,
keyed `clip_ordinal / stacked_row / raw_frame / clip_sha12`). **No new extraction.**
`BEVHEAD_TOK_DIR` selects the token dir, exactly as `p4_bev_head.py` does.

**Target.** The **agent join**, `C:/Users/Admin/tanitad-caches/b1-agent-join-20260906/b1eval_agents.jsonl.xz`,
md5(compressed) `3ddb42ecbd3926066795a94587af2aed` — per-frame EGO frame, +x fwd / +y left,
`occ = 0` meaning the centre is inside the 120° front field. Scored at **0–30 m**, front field only.

**Head.** A **centre heatmap + DETR slots** box head on the frozen tokens (no gradient into the
trunk). The heatmap and the slots are trained together; the slots emit `(x, y, l, w, yaw, logit)`.

### 3.1 The arms

| arm | tokens | what it is |
|---|---|---|
| `main` | s32 `[704,8,20]` | treatment, seed 0 |
| `main_s1` | s32 | seed replicate, seed 1 → the **training-seed floor** |
| `shuffled` | s32 | targets permuted across rows — **zero image↔target information** |
| `pixel` | `pix64_u8` | the **raw-pixel floor** |
| `s16` | s16 `[352,16,40]` | ⭐ the stride the spec bets perception on, seed 0 |
| `s16_s1` | s16 | seed replicate |
| `s16_shuffled` | s16 | zero-information twin at the winning stride |
| `s16_mirror` | s16 | ⛔ **MIRRORED ADDRESS** — the `R-2026-09-08-wpa-mirror` class. The token grid is flipped left–right relative to the target frame. **It must LOSE** |

Analytic controls, computed not trained: **`const`** (predict the prevalence everywhere) and
**`prior`** (the per-cell empirical rate over the train split).

### 3.2 Controls that must read KNOWN values

| control | known value |
|---|---|
| `const` | **EXACTLY the prevalence** |
| `shuffled` | **≤ prior + 0.02** |
| `pixel` | the raw-pixel floor, reported |
| `s16_mirror` | **must LOSE to `s16`** |
| seed replicate | both strides |

### 3.3 ⭐ REGISTERED BARS (frozen before any number is read)

| # | bar |
|---|---|
| **B1** | **AP@2 m − max(`shuffled`, `prior`, `pixel`) ≥ +0.05**, **separated** (clip-cluster bootstrap 95 % CI excludes 0), **≥ 3 × the seed floor**, on **2 seeds** |
| **B2** | **lead-presence AUC − within-clip shuffle ≥ 0.10.** ⛔ The shuffle is **WITHIN CLIP**: clip identity explained most of refav1's `n_agents` decode, so a global shuffle would pass on clip identity alone |
| **B3** | **s16 beats s32** on AP@2 m, separated. This is the evidence for the spec's choice to hang perception on stride-16. MEASURED reference (PUBLISHED-PRIMARY, `…/2026-09-07-wpa-readout-localisation/ORACLE_RECHECK.md`): an **ORACLE** reading 8×20 tops out at AP **0.3341** while 16×40 reaches **0.4762** (the brief quotes 0.4713; the corrected banked ladder reads 0.4762 at seed 0 and 0.4651 at seed 1 — the primary source governs and the discrepancy is reported) |
| **B4** | reported, not gating: AP@1 m and AP@4 m, **recall at fixed precision 0.50**, **closest-in-path range MAE** |

**Registered usefulness bars** (from §2.2 of the clarification, quoted): recall@2 m ≥ 0.50 at
precision ≥ 0.50, lead range MAE ≤ 2 m. Reported; they do not gate B1–B3.

### 3.4 Split

Clips are split by **clip**, never by frame — a frame-level split leaks a clip's own scene across
the split. The split is seeded and the seed is recorded; it is the SAME split for every arm.

---

## 4. What this package cannot decide

⛔ Quoting §2.4 verbatim: *"This package cannot prove refcv6 works. It decides whether refcv6
should bet on **reading** the trunk (arms c/e) or on **supervising** it (arm d), before ≥ 65 pod
GPU-hours are spent."*

Not in this package, and not claimed: the **map** panel (SAM3 maps of the eval clips are not on
this box), **occupancy on decoder outputs**, the **gradient-conflict detector**, the **tiny joint
rig**, T-C/T-D/T-E/T-F. Anything not run is reported as **NOT RUN**, never as absent evidence.

---

## 5. Evidence classes used in `RESULT.md`

MEASURED · PUBLISHED-PRIMARY · PUBLISHED-CODE · INHERITED · ESTIMATED · HYPOTHESIS.
Every eval number carries its **tier (T1 = self-action OPEN loop)** with **n windows AND n
episodes**.
