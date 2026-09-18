# The refcv6 dev-box preparation plan — and the two arithmetic facts that decide it

**Date:** 2026-09-18 · **Evidence class: MEASURED (ours) + ESTIMATED, labelled per row** ·
⛔ **ZERO GPU.** No model ran, no checkpoint was loaded, no arm was launched. This package is
file reads, a parity-manifest lookup and arithmetic.
⛔ **No tier stamp and no metric family:** nothing here is a capability claim.

The deliverable is `Project Steering/PREREG_REFCV6_DEVBOX_PREPARATION.md`. This note records
what was **measured** to build it, and the two findings that were not in the record before.

## The question

**PI, 2026-09-18:** *"Regarding the pod, we will use it only after we are finished with proven
prperation using my computer and its GPU."*

⇒ not *"what does a pod cost"* but **"what can an 8 GiB RTX 4060 actually prove about refcv6, in
what order, and what can it provably not"**.

---

## ⭐⭐ FINDING 1 — the usable held-out set is **124** clips, and the two exclusions are DISJOINT

Two exclusions were on the record and **nobody had intersected them**:

* `D-REFCV6-EVAL139-PARITY` (2026-09-18): **11 of 139** eval clips are inside
  `physicalai-train-e438721ae894` ⇒ *"quote 128, never 139, for a held-out read"*.
* the §10.6 pipeline validation: **4 of 139** clips carry **no SAM3 map** (`no_file` 684 windows
  = 4 × 171).

⛔ If the two sets overlapped, the answer would be anywhere from 124 to 128 — and **128 is what
the record currently tells a reader to quote**.

`code/clean_split.py` → `raw/clean_split.json`, offline, no GPU, no network:

| | |
|---|---|
| clips in the 416 × 1024 cache | **139** |
| SAM3 map files | **135** |
| ⭐ **`sha12` join hits** | **135 of 135** — *instrument check: the join is total, 0 orphans* |
| inside the parity TRAIN corpus | **11** ⭐ *reproduces `D-REFCV6-EVAL139-PARITY` exactly — the control* |
| without a SAM3 map | **4** |
| ⛔ **map-less clips that are ALSO parity clips** | **0** |
| ⇒ **held-out ∧ map-supervised ∧ parity-clean** | ⛔⛔ **124** |

⭐ **Disjoint, so the losses ADD:** 139 − 11 − 4 = **124**.

⇒ **Consequences, and they are not cosmetic:**

1. ⛔ The existing `splitA` (70 clips) / `splitB` (69) caches were cut from **139** and are
   **not usable for a held-out read**. Rebuilding them from the 124 is item **A0** of the plan.
2. ⛔ `SPEC_REFCV6_V2.md` §12.5 and `PREREG_REFCV6_V2.md` §2/§3 say **139**; the parity decision
   says **128**; the measured answer is **124**. ⚠️ **Raised for the owning agent — this package
   does not edit those authorities.**
3. ⭐ Same lesson, third corpus running: *the set you may quote is smaller than the set you built*
   (`4713 vs 4719`; `139 vs 128`; now **128 vs 124**).

⚠️ Clip identifiers appear only as **`sha12`**; the scan for full UUIDs in the artifact reads **0**.

---

## ⭐⭐ FINDING 2 — "a full arm is 13 days on the dev box" is true and MISLEADING

The step rate is MEASURED: **29.1967 s/step** (`resnet34.a1_in1k`, 416 × 1024, **batch 2**, every
head live) — 1,000 steps in 29,196.7 s, `…/2026-09-18-occupancy-floor/raw/maphead_1k.json`.
40,284 steps is therefore **13.19 – 13.61 days**.

⛔ **But `refcv5-v2`'s 40,284 steps were at `--batch 20`** (`PREREG_REFCV6` §4 BASE argv). A
dev-box step is **one tenth of a pod step's data**:

| | steps | wall-clock, measured band |
|---|---|---|
| **step-matched** | 40,284 | 13.19 – 13.61 d — ⚠️ *one tenth the samples: a different experiment* |
| ⛔ **sample-matched** | **402,840** | ⛔ **131.9 – 136.1 d** |

⛔ **And the gap cannot be closed by configuration.** MEASURED by `git grep -c` on the branch tip's
`stack/scripts/refc_v3_train.py`: **`accum` 0 · `autocast` 0 · `GradScaler` 0 ·
`utils.checkpoint` 0 · `--resume` 0.** There is no gradient accumulation, no AMP, no gradient
checkpointing and no resume.

⇒ **"13 days for a full arm" invites the reader to conclude the dev box could do a real arm in a
fortnight. It cannot — it could do one tenth of one arm.** The conversion must travel with every
dev-box number.

### ⚠️ And the same grep reopens a closed-looking question

⛔ *"`resnet101` OOMs on the 8 GiB card"* is MEASURED and correct (256 × 1024 batch 2 **and**
416 × 1024 batch 1). But it was measured **in fp32, with full activations, no checkpointing**.
⇒ the defensible statement is **"`resnet101` does not fit as the trainer is currently written"**,
not *"needs a bigger card"* — and the two imply different next actions.

**HYPOTHESIS, with its arithmetic:** the record's *"22.34 GiB allocated"* against 8 GiB is a
**2.8×** gap. bf16 autocast alone (≈2× on activations, 0× on params/optimiser) is **probably not
enough**; **+ gradient checkpointing** plausibly is. ⛔ **Nobody has tried it.** It is a code
change, so it needs its own pre-registration and a bit-identity proof — bucket **(C)**, not a GPU
arm. It is the cheapest thing that could change the pod request's shape.

---

## The plan in one table

| bucket | what | headline arithmetic |
|---|---|---|
| **(A) dev-box provable** | 10 items — split rebuild, per-half floors, harness probe, the **held-out** occupancy read, the S1 bound triple + zero-control + predicted gate, the conflict-detector cost, and **one** training lever (ImageNet vs random init, 2 seeds each) | **132.1 h = 5.50 d** (5.71 d with the `resnet101` branch) against a ~7-day ceiling |
| **(B) pod-scale only** | `E-REFCV6V2-DRIVE`; `resnet101` as the primary trunk; the 10-arm `PREREG_REFCV6` panel; 7 of 8 knockouts; the S1 training arms; T-CLASS/T-FLOOR; T-FLIP/T-COMPLY/T-MAXSPEED; `H-DDV2RL-3` | the 10-arm panel: **136.1 d step-matched · 1,361.3 d sample-matched · 40.6 d even at the 12,000-step `cut`**. `resnet101`: **0 steps are possible** |
| **(C) not compute** | the `resnet101`/AMP hypothesis · SAM3 (≈22 Sep) · **the corpus source frames are not on this box** · HF quota +112.9 GB · the missing trainer features · the two PI-owned weights · `t1_eval.py` at 416 × 1024 · the 124-clip rule | — |

### ⛔ The number the PI should see

**72 of the plan's 132 hours — 55 % — buy exactly ONE admissible lever claim.**
`H-ESTIM-SEED-1` measured a **14.3 %** false-positive rate for `separated` between two runs
differing in nothing, so a lever claim without a training replicate is inadmissible, and a
replicate doubles the arm. ⇒ On this box, one lever costs most of a week. **That ratio is the
evidence-backed core of the pod request.**

### ⭐ A third finding, smaller but load-bearing for the schedule

The **10.6 h / 386.5 GB** corpus-rebuild figure is a dev-box **rate** extrapolated from the
MEASURED 18.8 min / 139 clips — but the **4,713 train clips' source frames are not on this box**
(MEASURED: the whole `D:\Projects\TanitAD-artifacts` tree is **66 GB**; D: has 1.3 TB free, so
**disk is not the blocker**). ⇒ the corpus rebuild is gated on an **unpriced data transfer**, not
only on SAM3 — and it should not appear on a dev-box schedule until that is priced.

---

## What is NOT established

* ⛔ **Nothing about any model.** No checkpoint was loaded and no arm ran.
* ⚠️ The **21,207-window** figure is **ESTIMATED**: 124 **measured** clips × the **measured** 171.0
  windows/clip mean. Item A0 replaces it with a count.
* ⚠️ The **6.39 s/window** eval cost is **ESTIMATED** from two passes that also ran a train step
  ⇒ an **upper** bound. Item A2 re-measures it.
* ⚠️ The **≈65 s/step** `resnet101` figure is **ESTIMATED** by trunk-MAC scaling (**2.2257×**)
  alone; the heads do not scale with the trunk. ⛔ An order, not a rate.
* ⚠️ The AMP/checkpointing claim is a **HYPOTHESIS** and is labelled as one everywhere it appears.
* ⚠️ **`t1_eval.py` at 416 × 1024 is UNVERIFIED** — the plan's own item A2 answers it, and until it
  does, every eval hour in the schedule is costed against the **trainer's** eval path.

## Artifacts

| path | what |
|---|---|
| `raw/clean_split.json` | the 124-clip census — counts + `sha12` only |
| `code/clean_split.py` | how it was computed (`parity.clips_in_parity_train` + a `sha12` join to the map files) |
| `raw/devbox_cost_table.json` | every wall-clock figure in the pre-registration |
| `code/devbox_cost.py` | the arithmetic, with each input's evidence class inline |
| the plan | `Project Steering/PREREG_REFCV6_DEVBOX_PREPARATION.md` |

<!-- REFCV6-DEVBOX-PLAN-2026-09-18 -->
