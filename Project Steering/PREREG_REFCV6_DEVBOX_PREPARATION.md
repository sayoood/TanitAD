# PRE-REGISTRATION — the refcv6 DEV-BOX PREPARATION: what an 8 GiB card can actually prove, in what order, and what it provably cannot

**Date:** 2026-09-18 (Europe/Berlin) · **Author:** Research Lab (Architecture & Inference) ·
**Branch:** `agent/arch-inf-20260803`
**Status: PRE-REGISTRATION. ⛔ NOTHING LAUNCHED. 0 GPU seconds spent by this package.**
Every number below is MEASURED on this box with its artifact path, INHERITED with its source,
or explicitly labelled **ESTIMATED**.

## 0. The ruling this answers

> **PI, 2026-09-18, verbatim:** *"Regarding the pod, we will use it only after we are finished
> with proven prperation using my computer and its GPU."*

⇒ The question is no longer *"what would a pod cost"*. It is **"what is the most complete refcv6
proof this dev box can produce, and in what order?"** — and, inseparably, **what it cannot**, so
that the eventual pod request is **evidence-backed rather than asserted**.

⛔ **This document does not shrink refcv6's goal to fit the box.** Where the box cannot prove
something it goes to bucket **(B)** *with its arithmetic*. Making the pod request defensible is the
point; avoiding it is not.

**Authorities read at the branch tip:** `SPEC_REFCV6_V2.md` (incl. §10 amendments, §11 R1–R3,
§12 the 416 × 1024 ruling), `PREREG_REFCV6.md` (+ its two 2026-09-11 appendices),
`PREREG_REFCV6_V2.md` (+ `…ERRATUM-1.md`), `PREREG_S1_AGENT_SEAM_AND_COLLISION_GATE.md`.

---

## 1. The measured constants everything below is computed from

⭐ **All wall-clock arithmetic in this file is emitted by `code/devbox_cost.py`
→ `raw/devbox_cost_table.json`** in the companion package, so a figure cannot drift between the
prose and the table. Nothing is typed twice.

| # | constant | value | class · source |
|---|---|---|---|
| 1 | **dev-box step rate** | **29.1967 s/step** (`resnet34.a1_in1k`, **416 × 1024**, **batch 2**, every head live, RTX 4060 8 GiB) — 1,000 steps in **29,196.7 s**; the 30-step probe predicted **28.30** | **MEASURED (ours)** · `…/2026-09-18-occupancy-floor/raw/maphead_1k.json` |
| 2 | **eval cost** | **≈ 6.39 s/window** forward-with-every-head | **ESTIMATED (basis MEASURED)** · `cover_evalA` 3,221.9 s / 500 windows, `cover_evalB` 2,534.3 s / 400 windows — each pass also ran 1 train step, so this is an **UPPER** bound · `…/2026-09-17-refcv6-pipeline-validation/RESULT.md` |
| 3 | **windows per clip** | **171.0** (23,772 windows over 139 clips) | **MEASURED** · same package, map-gate report in `config.json` |
| 4 | ⭐ **usable clean corpus** | ⛔ **124 clips**, not 139 and not 128 — see §2 | **MEASURED (ours, this package)** · `raw/clean_split.json` |
| 5 | **`resnet101`** | **OOMs** at 256 × 1024 batch 2 **and** 416 × 1024 **batch 1** on the 8 GiB card; CPU run completes at **76.58 s/step** (382.9 s / 5 steps), every head live, rig coverage 139/139 | **MEASURED** · `GOALS_AND_CLAIMS.md` `D-REFCV6-R101`; `…/r101b1_416cpu/` |
| 6 | **trunk MAC ratio** | `resnet101` / `resnet34` = **2.2257×** per window position | **MEASURED** · `…/2026-09-17-refcv6-e2e-1024/raw/cost_model.json` |
| 7 | **occupancy floor** | no-information IoU **0.3412** over **10,068,274 seen cells** | **MEASURED** · `…/2026-09-18-occupancy-floor/raw/occ_floor.json` |
| 8 | **occupancy head @ 2,000 steps** | last-40 mean IoU **0.4865 = 1.43×** the floor; `prob_mean` **0.3460** vs GT prevalence 0.3412 | **MEASURED**, ⚠️ **TRAIN-SIDE, batches of 2 windows** · `…/raw/maphead_2k.json` |
| 9 | **pipeline coverage** | **900 windows, all 139 clips, every head live, 0 dropped boxes, 0 rows without a camera**, at 416 × 1024 | **MEASURED** · pipeline-validation §CLOSURE |
| 10 | **eval-side `box3d_z`/`_h`** | FIXED: `eval_box3d_n_z` **0.0 → 40.192** (A) / **35.955** (B), `n_z == n_matched`, `n_matched` **identical** before and after as the control | **MEASURED** · pipeline-validation §WORK ITEM CLOSED |
| 11 | ⛔ **seed false-positive rate** | two runs differing in **NOTHING** read `separated` on **6 of 42 family cells = 14.3 %** | **MEASURED** · `H-ESTIM-SEED-1`, `GOALS_AND_CLAIMS.md:169` |
| 12 | **trainer capability gaps** | `refc_v3_train.py` at the tip has **no `--resume`**, **no gradient accumulation**, **no `autocast`/`GradScaler`**, **no `torch.utils.checkpoint`** — all four greps return **0** | **MEASURED (ours)** · `git grep -c` on `refs/heads/agent/arch-inf-20260803:stack/scripts/refc_v3_train.py` |
| 13 | **local data inventory** | the whole `D:\Projects\TanitAD-artifacts` tree is **66 GB**; the only 416 × 1024 cache is **`v2ep-eval139-416x1024cyl`, 11 GB, 139 clips**. ⛔ **There is no train corpus and no source-frame corpus on this box.** D: has **1.3 TB free** | **MEASURED (ours)** · `du`/`df`, 2026-09-18 |

### 1.1 ⚠️ One correction to the figure I was briefed with

The briefing states *"a full-length arm is 40,284 steps = ~12.9 days"*. At the **measured** band
that is **13.19 d** (28.30 s/step) to **13.61 d** (29.1967 s/step). `~12.9 d` implies **27.67
s/step**, which is below anything measured on this box. ⛔ **The measured band is used throughout;
the difference is small but a plan built on an unmeasured rate is a plan that cannot be checked.**

### 1.2 ⭐ The single most important unit conversion, and it is not a rate

⛔ **`refcv5-v2`'s 40,284 steps were at `--batch 20` on an A40** (`PREREG_REFCV6` §4 BASE argv).
This box runs **batch 2**. A dev-box "step" is therefore **one tenth of a pod step's data**, and
constant 12 says **there is no `--accum` flag to close the gap**.

| what "a full arm on the dev box" means | steps | wall-clock (measured band) |
|---|---|---|
| **step-matched** (same 40,284 steps, batch 2) | 40,284 | **13.19 – 13.61 days** — ⚠️ and it is a *different experiment*: one tenth the samples |
| ⛔ **sample-matched** (the same data seen) | **402,840** | ⛔ **131.9 – 136.1 days** |

⇒ ⛔ **Quoting "13 days for a full arm" without the batch ratio is the `true but wrong for the
reader` failure**: it implies the dev box could do a real arm in a fortnight. It cannot. It could
do **one tenth of one arm** in a fortnight.

---

## 2. ⭐⭐ A NEW MEASURED FACT THAT RESIZES EVERY PANEL BELOW: the usable set is **124**, not 139

`D-REFCV6-EVAL139-PARITY` (2026-09-18) established that **11 of the 139** eval clips sit inside
`physicalai-train-e438721ae894`, so a held-out read must quote **128**. Separately, the pipeline
validation established that **4 of the 139** carry **no SAM3 map**. ⛔ **Nobody had intersected the
two sets.** Measured here, offline, zero GPU (`code/clean_split.py` → `raw/clean_split.json`):

| | |
|---|---|
| clips in the 416 × 1024 cache | **139** |
| SAM3 maps, joined by `sha12` | **135** — **135 of 135 join**, 0 orphans *(instrument check)* |
| inside the parity TRAIN corpus | **11** *(reproduces `D-REFCV6-EVAL139-PARITY` exactly — the control)* |
| without a SAM3 map | **4** |
| ⭐ **map-less clips that are ALSO parity clips** | ⛔ **0** |
| ⇒ **held-out AND map-supervised AND parity-clean** | ⛔⛔ **124** |

⭐ **The two losses are DISJOINT, so they ADD**: 139 − 11 − 4 = **124**. Had they overlapped the
answer would have been up to 128; it is not.

⇒ **Every panel in this document is stated on 124 clips ≈ 21,207 windows** (124 × the measured
171.0), split into two **disjoint 62-clip halves of ≈ 10,603 windows each**. ⛔ The existing
`splitA` (70) / `splitB` (69) caches were cut from **139** and are **not usable for a held-out
read** — rebuilding them from the 124 is item **A0** below.

⚠️ **Same lesson, third corpus:** *the set you may quote is smaller than the set you built*
(`4713 vs 4719`; `139 vs 128`; now **128 vs 124**).

---

## 3. THE PARTITION

### 3.1 Bucket **(A)** — PROVABLE ON THE DEV BOX

⛔ **The governing constraint, stated before the table so it cannot be read past:** on 124 clips
at batch 2, **one epoch over a 62-clip half is 5,302 steps = 43.0 h**. A 2,000-step arm sees
**0.38 of one epoch**. ⇒ **No arm in bucket (A) is a capability claim, and none may be quoted as
driving performance.** Bucket (A) proves **instruments, controls, costs, wiring and one
learning-curve lever** — that is what "proven preparation" can mean here, and stating it plainly is
what makes bucket (B) credible.

| # | arm | exact configuration | cost | criterion it must clear | lever or instrument? |
|---|---|---|---|---|---|
| **A0** | **clean-124 split rebuild** | file copies; cut the 124 into disjoint 62/62 halves (sorted-order alternating, as the 139 split was), stamp the parity record into each manifest | **≈ 0.5 h**, **no GPU** | both halves disjoint; `guard_corpus_build(role="eval", mode="exclude")` returns **124 kept, 11 excluded**; every kept clip has a map | **instrument** |
| **A1** | **per-half occupancy floor** | recompute `P(drivable \| seen)` on each 62-clip half from the SAM3 `.npz`, every 20th frame | **≈ 0.5 h**, **no GPU** | GT-against-itself reads **exactly 1.0000**; each half's floor within ±0.02 of the corpus **0.3412** | **instrument** |
| **A2** | **harness rate + 40-step wiring smoke** | the clean half, 40 steps, `resnet34`, 416 × 1024, batch 2, every head live; **plus** a `t1_eval.py` invocation at 416 × 1024 | **≈ 1.0 h** | `metrics.json` present, finite, **and moved** (a constant loss is a disconnected graph); `tacv6_n_scene_mean` = **496.0**; T1 harness runs and its **s/window is recorded** | **instrument** |
| **A3** | ⭐ **OCC-HELDOUT** — the read `PREREG_S1` §8 actually gates on | score the **existing** 2,000-step map-head checkpoint on **1,000 HELD-OUT windows** of the other half | **≈ 1.8 h** (eval only) | `map_iou_drivable` over 1,000 held-out windows vs the **A1 floor for that half** | **instrument reading** — one config, no comparison ⇒ **NO replicate** |
| **A4** | **S1-RANDOM · S1-BASE · S1-GATE-ORACLE**, same 1,000 windows | `PREREG_S1` §8: *"need no trained perception and can run first"* | **3 × 1.8 = 5.3 h** | `S1-RANDOM` reads the no-information value of the selection statistic; the fan's collision-free share reads **≈ 55.3 %** as MEASURED | **instrument + bounds** ⇒ **NO replicate**. ⛔ `S1-GATE-ORACLE` is **T0** and stamped so |
| **A5** | ⛔ **S1-GATE-CONST** — the deliberate-regression arm | identical gate, occupancy says everything is free | **≈ 1.8 h** | ⛔ must recover **exactly zero**. Anything else means the gain is the re-ranking machinery, not the signal | **control reading a known value** ⇒ **NO replicate** |
| **A6** | ⚠️ **S1-GATE-PRED** on the A3 occupancy, same windows | the deliverable of `H-SEL-GATE-1`, at dev-box `n` | **1.8 h + 1.8 h = 3.6 h** | collided-selection rate vs `S1-BASE`, reported as `(pred − base)/(oracle − base)` | ⚠️ **LEVER** — but a **selection-time** lever with **no retraining**, so its floor is the **inference-seed** floor: the replicate is a **second inference draw**, budgeted above. ⛔ It is **not** a training replicate and may never be called one |
| **A7** | ⭐⭐ **the ImageNet knockout** — the one training lever this box can afford | `resnet34` **ImageNet-init** vs **random-init**, *same architecture*, 2,000 steps, batch 2, **2 seeds each = 4 arms** | **4 × 16.22 = 64.9 h** + **4 × 1.77 = 7.1 h** eval = **72.0 h** | §3.3 below | ⛔ **LEVER CLAIM — replicate MANDATORY and budgeted.** `PREREG_REFCV6_V2` §4: *"the only control that separates 'pretraining helped' from 'this architecture is better'. It is not optional."* |
| **A8** | **occupancy continuation to 5,000 steps** + held-out read | continue A3's configuration on the clean half to 5,000; score 1,000 held-out windows | **40.6 h + 1.8 h = 42.4 h** | held-out IoU vs the A1 floor, as a **curve** with its step index, not a single point | **instrument reading** — one config ⇒ **NO replicate** |
| **A9** | **conflict-detector cost at 416 × 1024** | 200 steps `--conflict-detector on` vs 200 off, otherwise identical | **2 × 1.62 = 3.2 h** | the `±1` controls read **exactly** 1.0 / −1.0 and the detached case **NaN**; overhead reported as a % with its `n` | **instrument** — the banked **+71.7 … +104.2 %** was measured on the **smaller grid** and the ruling says *"re-measure rather than scale"* |

**BUCKET (A) TOTAL: 132.1 h = 5.50 days.** Fits under 7 days with **≈ 36 h of headroom**, which is
reserved for the `resnet101` branch (§4) and for exactly one re-run.

### 3.2 Bucket **(B)** — PROVABLE ONLY WITH POD-SCALE COMPUTE, with the arithmetic

⛔ Every row states dev-box days at the **MEASURED** rate. That arithmetic is what makes the row
bucket (B); pod days are a separate, explicitly-labelled sizing estimate (§3.4).

| # | claim / panel | why it cannot be done here — the arithmetic |
|---|---|---|
| **B1** | ⛔ **`E-REFCV6V2-TRUNK` in its PRIMARY form (`resnet101`)** | ⛔ **0 steps are possible.** It OOMs at 416 × 1024 **batch 1** on the 8 GiB card (constant 5). The dev-box cost is not large — it is **undefined**. If it fitted, the trunk-MAC ratio puts it at **≈ 65 s/step** (**ESTIMATED**, trunk-scaling only) ⇒ 2,000 steps = **36 h**, a full arm **30.3 d step-matched / 303 d sample-matched** |
| **B2** | ⛔ **`E-REFCV6V2-DRIVE`** — the only driving claim in the programme | needs T1 on a corpus-scale held-out set, all four families, separated against `ha0_ext`, **with a training replicate**. On this box the corpus is **124 clips**, 1/38th of the 4,713-clip corpus, and 2 arms at the **`cut`** budget cost **8.1 d step-matched / 81.1 d sample-matched**. ⛔ Both the compute **and** the corpus forbid it; either alone would |
| **B3** | the **10-arm `PREREG_REFCV6` panel** (V0/V0b/D/Db/A/Ab/B/Bb/C/Cb) | **136.1 d step-matched · 1,361.3 d (3.7 years) sample-matched · 40.6 d even at the 12,000-step `cut`** |
| **B4** | the **8 registered knockouts** of `PREREG_REFCV6_V2` §4 × 2 seeds = 16 arms | at the `cut` budget, step-matched: **16 × 4.06 = 64.9 d**. ⭐ **One of the eight — ImageNet-vs-random — is pulled forward into (A7) in a reduced, explicitly-labelled form.** The other seven are not reachable |
| **B5** | the **`PREREG_S1` training arms** (`S1-BASE`, `S1-AGENTS`, `S1-BOTH`, `S1-REPL` × 2 seeds = 8) | at the `cut` budget: **32.4 d**. ⭐ The **inference-only** S1 arms are pulled into (A4)–(A6); only the ones needing a trained model stay here |
| **B6** | **T-CLASS / T-FLOOR** — per-token tactical AP | arithmetic, not opinion: `n = 200` on a **10,603-window** half needs a prevalence of **≥ 1.886 %**. `SPEC` §4 records that **10 of 22 tokens already sit under the n = 200 floor on the FULL corpus** ⇒ they are **structurally unreachable** at dev-box `n`, at any step count |
| **B7** | **T-FLIP (≥ 0.50) · T-COMPLY (≥ 0.38) · T-MAXSPEED (≥ 99 %)** | all three are properties of a *trained planner at corpus scale*; today's T-FLIP **0.205** is a corpus-scale number and a 124-clip reading is not comparable to it |
| **B8** | **`H-DDV2RL-3`** (the repaired D9 the PI directed: *"dont park D9, solve it and prove it"*) | ⛔ explicitly gated: must not start until SAM3 covers the **RL-train** split, or `dac_from_drivable` returns its no-map value of **1** on the uncovered part — *reintroducing the exact defect being repaired* |
| **B9** | any comparison against the **banked** `refcv5-v2` / `refcv4b` panels (4,823 windows / 141 episodes) | those were scored on a different corpus at a different geometry; a 124-clip 416 × 1024 arm is not pairable with them |

### 3.3 Bucket **(C)** — NOT A COMPUTE QUESTION AT ALL

| # | gap | what it actually is | status |
|---|---|---|---|
| **C1** | ⛔⛔ **`resnet101` may be a TRAINER gap, not a card gap** | Constant 12: the trainer has **no AMP, no gradient checkpointing, no accumulation** — the OOM was measured in **fp32 with full activations**. ⛔ *"`resnet101` OOMs"* invites *"buy a bigger card"*; the honest form is **"`resnet101` does not fit as the trainer is currently written"** | **HYPOTHESIS** with arithmetic: the record's *"22.34 GiB allocated"* against 8 GiB is a **2.8×** gap. bf16 autocast alone (≈2× on activations, 0× on params/optimiser) is **probably not enough**; **+ gradient checkpointing** plausibly is. ⛔ Code change ⇒ own pre-registration + bit-identity proof. **Nobody has tried it** |
| **C2** | ⛔ **SAM3 corpus** | ≈ **2026-09-22** (**INHERITED**, `PI_DECISION_QUEUE` item 21) | blocks B8 and the corpus rebuild |
| **C3** | ⛔⛔ **the corpus rebuild is NOT dev-box work, and not only because of SAM3** | The **10.6 h / 386.5 GB** figure is a dev-box **rate** extrapolation from the MEASURED **18.8 min / 139 clips**. But constant 13 says **the 4,713 train clips' source frames are not on this box** (66 GB total artifacts). D: has 1.3 TB free, so **disk is not the blocker — an unpriced data transfer is** | ⭐ **NEW.** Price the transfer before the rebuild appears on any schedule |
| **C4** | **HF quota** | **+112.9 GB** (386.5 vs 273.6) against a hard ceiling | ⛔ check **BEFORE** the push, standing constraint |
| **C5** | ⛔ **the two values that are the PI's** | `--w-agent` (`MEASURED (provenance) / NOT PRE-REGISTERED`) and `--w-tac-goal` (queue item 10). `arms.py` **refuses to invent either** | arms **A** and **C** cannot launch **anywhere** — dev box or pod — until these exist |
| **C6** | ⛔ **no `--resume`** | a 5,000-step arm (A8) that dies **restarts**; `MAX_RELAUNCH=1` forbids the relaunch | ⇒ A8 is the **last** GPU item in the schedule, so a loss costs the least |
| **C7** | ⛔ **no gradient accumulation** | the dev box **cannot** reach the pod's effective batch 20 by configuration ⇒ **a dev-box arm and a pod arm are not the same experiment and cannot be made so without code** | this is why §1.2's conversion must travel with every dev-box number |
| **C8** | ⚠️ **`t1_eval.py` at 416 × 1024 is UNVERIFIED** | is the T1 harness wired for the refcv6 forward with four heads at the new geometry? | ⛔ **A2 must answer this before any eval in (A) is costed as a T1 read.** If it is not wired, A3–A6 are trainer-side reads only and must say so |
| **C9** | **the 124-clip rule** | §2 | ⛔ binding on every future held-out read of this set |
| **C10** | open reporting decisions | queue **22** (freeze the clean val at n = 400, default (a)) · **23** (lead with the **deployment** nav-zero margin, default (a)) · the **traffic-light carve-out** (default: no carve-out) | zero compute; each changes what the programme *appears* to claim |
| **C11** | ✅ **closed, recorded so it is not re-owed** | `PREREG_REFCV6`'s fact-10 re-read from raw eval JSON (discharged 2026-09-11); the eval-side `box3d_z`/`_h` defect (constant 10) | no action |

### 3.4 ⚠️ On pod days

⛔ **This document deliberately does not convert bucket (B) into pod days.** The A40 rates on
record (**4.0 / 4.11 / 4.21 s/step**) were all taken at **256 × 640**; `SPEC` §11 R1 states the
token count rises **1.63× at both strides** at the new geometry and instructs **"re-measure rather
than scale"**. ⇒ **The pod request should ask for a 200-step rate measurement as its FIRST job**,
and quote bucket (B) in **steps and samples** — which are geometry-independent — rather than in
days derived from a superseded geometry. Quoting a scaled pod-day figure here would be an
ESTIMATE dressed as a budget.

---

## 4. ⛔ BRANCHING ON `resnet101` — both outcomes, written before the answer

A parallel agent is measuring whether **any** `resnet101` configuration fits. The plan branches;
it does not assume.

### 4.1 If `resnet101` **FITS** at some configuration

* **Add A10** — a **200-step rate-and-shape arm** at whatever config fits: **≤ 5 h** (ESTIMATED at
  ≈65 s/step; **re-measure, do not scale**). Bucket (A) total → **≈ 5.7 days**, still inside 7.
* A10 is a **SIZING INSTRUMENT, not an arm**: it yields `s/step`, peak memory, and confirmation
  that every head stays live. ⛔ It is **not** a learning curve and **not** `E-REFCV6V2-TRUNK`.
* ⚠️ **If it fits only at a reduced configuration** (K = 1 instead of 3, or a smaller batch), then
  the configuration measured is **not the pre-registered one**, and A10's rate may **not** be used
  to size the pre-registered arm. It is then a lower bound, labelled as one.
* **B1 stays in bucket (B).** Fitting at batch 1 for 200 steps does not make a 40,284-step arm
  possible: at ≈65 s/step, step-matched is **30.3 d** and sample-matched **303 d**.

### 4.2 If `resnet101` does **NOT** fit — ⛔ what dies

| dies on the dev box | consequence |
|---|---|
| any `resnet101` **learning curve** | → bucket (B) |
| the **A-vs-B recipe comparison** (`PREREG_REFCV6_V2` §1) | → bucket (B). ⚠️ It was already declared *"a RECIPE comparison, not a controlled single-variable ablation"*; it is now also **unrehearsable** |
| a **measured** `resnet101` rate on this hardware | the pod arm's cost stays **ESTIMATED** from the CPU **76.58 s/step** and the **2.2257×** MAC ratio |
| the **memory ceiling**, beyond *"> 8 GiB"* | the pod request must ask for headroom it cannot justify precisely — ⛔ **and it must say so**, rather than quoting the record's *"22.34 GiB"* as a requirement: that figure is an allocator report from a failing run, not a measured working-set |
| ⛔ **the primary arm's first GPU run is its first run at scale** | `SPEC` §12.5's own words. **That is the strongest single line in the pod request**, and it is only true in this branch |

⭐ **In both branches, C1 remains open and is the cheapest thing that could change the answer.**
⛔ Do not close the `resnet101` question as *"needs a bigger card"* until AMP + gradient
checkpointing have been tried, because the current evidence cannot distinguish the two.

---

## 5. ⛔⛔ THE REPLICATE PROBLEM, CONFRONTED

**The rule:** `H-ESTIM-SEED-1` measured a **14.3 % false-positive rate** for `separated` between two
runs differing in **nothing** (6 of 42 family cells). The episode-cluster bootstrap answers *"would
another draw of EPISODES say this?"* and **never** *"would another TRAINING RUN say this?"*
⇒ **a lever claim without a training replicate is inadmissible.** A replicate **doubles** the arm.

⛔ **This is where a plan quietly becomes inadmissible, so it is tabulated per arm rather than
asserted once.**

| arm | lever or instrument | replicate needed? | cost with it |
|---|---|---|---|
| A0, A1 | instrument (no model) | **no** | 1.0 h |
| A2 | instrument (wiring + rate) | **no** — a rate is not a claim about a lever | 1.0 h |
| **A3** | **instrument reading** — one checkpoint, one config, scored against a **corpus-measured floor**, no A/B | ⛔ **no** | 1.8 h |
| **A4** | **bounds + a no-information control** — `RANDOM`/`ORACLE` are not levers; they are the axis a lever would later be read on | ⛔ **no**, **but** all three must be on the **SAME windows** or none is quotable | 5.3 h |
| **A5** | **control that must read a known value (exactly 0)** | ⛔ **no** — a control reading its known value is a check, not an effect | 1.8 h |
| **A6** | ⚠️ **LEVER** | ⚠️ **a training replicate is NOT applicable** — nothing is retrained; the gate reorders an existing fan. Its relevant floor is the **inference-seed** floor, so **a second inference draw is required and is budgeted**. ⛔ It must be reported as *"beyond the inference floor"*, never *"beyond the run-to-run floor"* | 3.6 h |
| **A7** | ⛔ **LEVER — the only training lever in bucket (A)** | ⛔ **YES, and it is the reason A7 costs 72 h of a 132 h plan.** 2 conditions × 2 seeds. Dropping the replicate would save 36 h and make the whole result inadmissible | 72.0 h |
| **A8** | **instrument reading** — one config continued, reported as a **curve** | ⛔ **no**, **provided** it is never turned into *"5,000 beats 2,000"*, which would be a lever claim against a different-length arm | 42.4 h |
| **A9** | **instrument (cost) + identity controls** | **no** | 3.2 h |

⭐ **The accounting that matters: 72 of 132 hours — 55 % of the entire dev-box plan — buys exactly
ONE admissible lever claim.** That ratio is the evidence-backed core of the pod request, and it is
what the PI should see.

⛔ **The temptation this table exists to refuse:** running A7 at one seed, seeing a separated
interval, and reporting it. At a **14.3 %** false-positive rate that reads as a finding roughly one
time in seven for free. ⛔ **If the schedule ever has to be cut, A7's replicate is the LAST thing
cut — cutting it does not shorten the plan, it deletes the plan's only claim.**

---

## 6. THE ORDERED DEV-BOX SCHEDULE

⛔ **Sequential. This box runs ONE GPU job at a time.** Ordering rules applied: zero-GPU work
first (it can overlap another agent's GPU job); every instrument before the thing it measures;
the **longest, unresumable** job last (C6).

| step | item | GPU | hours | **running total** |
|---|---|---|---|---|
| 1 | **A0** clean-124 split rebuild | — | 0.5 | **0.5 h** |
| 2 | **A1** per-half occupancy floors | — | 0.5 | **1.0 h** |
| 3 | **A2** 40-step smoke + T1 harness probe (**answers C8**) | ✅ | 1.0 | **2.0 h** |
| 4 | **A3** OCC-HELDOUT on 1,000 held-out windows | ✅ | 1.8 | **3.8 h** |
| 5 | **A4** `S1-RANDOM` · `S1-BASE` · `S1-GATE-ORACLE`, same windows | ✅ | 5.3 | **9.1 h** |
| 6 | **A5** `S1-GATE-CONST` (must read exactly 0) | ✅ | 1.8 | **10.9 h** |
| 7 | **A6** `S1-GATE-PRED` + its inference replicate | ✅ | 3.6 | **14.5 h** |
| 8 | **A9** conflict-detector cost at 416 × 1024 | ✅ | 3.2 | **17.7 h** |
| 9 | *(branch)* **A10** `resnet101` rate arm — **only if it fits** | ✅ | ≤ 5.0 | **≤ 22.7 h** |
| 10 | **A7** ImageNet knockout, 4 arms × 2,000 steps + 4 evals | ✅ | 72.0 | **89.7 h** |
| 11 | **A8** occupancy → 5,000 steps + held-out read | ✅ | 42.4 | ⭐ **132.1 h = 5.50 d** *(137.1 h = 5.71 d with A10)* |

### 6.1 ⭐ It fits under 7 days — and here is what to cut if it stops fitting

**5.50 days (5.71 with the `resnet101` branch) against a ~7-day ceiling**, leaving **≈ 30–36 h**.
That headroom is **already spoken for**: A2 can discover the T1 harness is not wired (C8), and the
schedule permits exactly **one** re-run. ⛔ It is not spare capacity for a new arm.

**If the plan must be cut, in this order, with the evidence cost stated:**

| cut | saves | ⛔ what it costs in evidence |
|---|---|---|
| **1st — A8** (5,000-step continuation) | **42.4 h** | the occupancy curve stops at 2,000 steps. ⚠️ The **1.43×** figure stays a *train-side* read; A3 still gives the held-out read `PREREG_S1` §8 requires, so **the gate's dependency stays discharged** — only the *trend* is lost |
| **2nd — A9** (conflict cost) | **3.2 h** | the detector's overhead at 416 × 1024 stays **EXTRAPOLATED** from a smaller grid, against an explicit instruction not to scale it. ⇒ the **pod budget carries an unpriced +72…+104 %** |
| **3rd — A6** (`S1-GATE-PRED`) | **3.6 h** | the deliverable of `H-SEL-GATE-1` is not attempted at all; A4's ceiling and A5's zero-control stand but nothing reaches toward them |
| ⛔ **NEVER — A7's replicate** | 36.0 h | ⛔ **deletes the plan's only admissible lever claim.** A one-seed A7 is not a cheaper A7; it is a **14.3 %-false-positive coin flip** wearing A7's name |
| ⛔ **NEVER — A0/A1** | 1.0 h | every held-out number in the package would be scored on a contaminated split against a floor measured elsewhere |

---

## 7. BOTH OUTCOMES, COMMITTED IN ADVANCE

⛔ Pre-registration, not a wish list. Each row states what would **support** and what would
**refute**, before the data exists.

| arm | ⭐ SUPPORTS if | ⛔ REFUTES / what I will report if not |
|---|---|---|
| **A0** | the guard returns **124 kept / 11 excluded**, halves disjoint, every kept clip mapped | any other count ⇒ ⛔ **§2's census is wrong and this whole document is re-sized before anything runs.** A count between 124 and 128 means the disjointness finding is wrong |
| **A1** | GT-vs-GT reads **exactly 1.0000**; each half's floor within ±0.02 of **0.3412** | GT-vs-GT ≠ 1.0 ⇒ ⛔ the comparison is wired wrong and **nothing downstream is readable**. A half whose floor differs by > 0.02 ⇒ the halves are not exchangeable and **each panel carries its own floor**, never the pooled one |
| **A2** | loss finite **and moving**; `tacv6_n_scene_mean` = 496.0; T1 harness runs and its rate is recorded | a **constant** loss ⇒ disconnected graph, stop. T1 harness not wired at 416 ⇒ ⛔ **C8 is live**, A3–A6 are trainer-side reads and are **relabelled**, not silently quoted as T1 |
| **A3** | held-out IoU **> the A1 floor for that half**, over 1,000 windows | **at or below the floor** ⇒ ⛔ **the 1.43× was a train-side artefact** and `S1-GATE-PRED`'s dependency is **NOT** discharged. ⇒ next work is perception quality, not a gate (`PREREG_S1` §10's own commitment) |
| **A4** | `S1-RANDOM` reads the no-information value; fan collision-free share ≈ **55.3 %**; oracle > base > random on the same windows | a wildly different fan share ⇒ ⛔ **the fan or the collision checker changed and no arm is readable**. Oracle ≯ base ⇒ the **+0.0485 / 62.4 %** headroom does not reproduce at 416 × 1024 on 124 clips, and **that is the finding** |
| **A5** | recovery **exactly 0.0** | **non-zero** ⇒ ⛔ the gain is the **re-ranking machinery**, not the occupancy ⇒ **a pass on A6 means nothing** and is not reported as one |
| **A6** | collided-selection rate falls vs `S1-BASE` beyond the **inference** floor, on **both** draws, with no family harmed | no movement ⇒ `H-SEL-GATE-1` **REFUTED as configured at this scale**. Movement on one draw only ⇒ **INCONCLUSIVE**, reported as underpowered, ⛔ never as a win. Recovery **< 10 %** of A4's ceiling ⇒ `PREREG_S1` §10's pre-committed abandonment: **the occupancy is not good enough to reach the headroom** |
| **A7** | ImageNet-init separably ahead of random-init on the planner loss **at matched step count, on BOTH seeds, by more than the seed floor measured in the SAME panel** | the two conditions **inside** the seed floor ⇒ ⛔ **the ImageNet prior is not visible at 2,000 steps on 124 clips.** ⚠️ That is **UNDERPOWERED, not REFUTED** — and the pre-committed reading is that `E-REFCV6V2-TRUNK` **needs corpus scale**, which is itself evidence for the pod. ⛔ Random-init **ahead** ⇒ report it exactly as measured; it would be the most interesting result in the package |
| **A8** | held-out IoU rises monotonically in step index across the reported checkpoints | flat or falling after 2,000 ⇒ the head has **saturated on 62 clips** ⇒ ⛔ more dev-box steps are not the lever and the schedule's **first cut becomes permanent** |
| **A9** | `±1` controls read **exactly** 1.0 / −1.0, detached reads **NaN**; overhead reported with `n` | a tolerated epsilon instead of an exact identity ⇒ ⛔ the float64 contract is broken and **every conflict number is void**. Overhead ≫ the banked +104 % ⇒ the pod budget rises and **the pod request says so** |
| **A10** *(branch)* | a rate, a peak memory and every head live | ⛔ does not fit ⇒ **§4.2**, and B1's *"first GPU run is its first run at scale"* becomes the pod request's load-bearing line |

---

## 8. ⛔ THE EXIT CONDITION — a checklist someone else can verify

**The preparation is "proven" and the pod question reaches the PI when, and only when, every line
below can be ticked against a named artifact.** ⛔ An unticked line is not a delay — it is the
reason the pod request is not yet evidence-backed.

### 8.1 Already true at the time of writing

- [x] **E1** The chain carries end to end at the ruled **416 × 1024** — 900 windows, all 139 clips, every head live, **0** dropped boxes, **0** rows without a camera · *pipeline-validation §CLOSURE*
- [x] **E2** Eval-side `box3d_z`/`_h` supervised and re-measured, `n_z == n_matched`, with `n_matched` unchanged as the control · *pipeline-validation §WORK ITEM CLOSED*
- [x] **E3** The occupancy **floor** exists over **10,068,274 seen cells** (0.3412), with GT-vs-GT reading exactly 1.0000 · `raw/occ_floor.json`
- [x] **E4** The dev-box **step rate is MEASURED** for the trunk that will actually run here (`resnet34`, 416 × 1024, batch 2): **29.1967 s/step** · `raw/maphead_1k.json`
- [x] **E5** `resnet101`'s **shapes** are proven (CPU, every head live, 139/139 rig coverage) — so *"the shapes are wrong"* is excluded and only *"the card is too small"* remains open
- [x] **E6** The **usable held-out set is measured at 124** clips, and the two exclusions are proven **disjoint** · `raw/clean_split.json` *(this package)*

### 8.2 Must become true — bucket (A)

- [ ] **E7** The 62/62 halves are rebuilt from the **124** and the parity record is stamped into each manifest *(A0)*
- [ ] **E8** Each half's occupancy floor is measured **in the same panel** the head is scored in *(A1)*
- [ ] **E9** **C8 is answered**: `t1_eval.py` either runs at 416 × 1024 — with its s/window recorded — or is declared unwired, and every downstream number is relabelled accordingly *(A2)*
- [ ] **E10** ⭐ **The occupancy head has a HELD-OUT reading against its own half's floor** *(A3)* — this is the literal dependency `PREREG_S1` §8 gates `S1-GATE-PRED` on, and today's **1.43×** does **not** satisfy it because it is train-side
- [ ] **E11** `S1-RANDOM`, `S1-BASE` and `S1-GATE-ORACLE` are reported **on the same windows**, and the fan's collision-free share reads ≈ **55.3 %** *(A4)*
- [ ] **E12** ⛔ `S1-GATE-CONST` reads **exactly zero** recovery *(A5)*
- [ ] **E13** **A7 has run all four arms** (2 conditions × 2 seeds) and its verdict is stated **as a ratio to the seed floor measured in that same panel** — `SUPPORTED`, `REFUTED` or `UNDERPOWERED` *(A7)*
- [ ] **E14** The conflict detector's overhead is **measured at 416 × 1024**, not scaled from the smaller grid *(A9)*
- [ ] **E15** *(branch)* `resnet101` is either **rate-measured on this card** *(A10)* or **recorded as unfittable with §4.2's consequences written into the pod request**

### 8.3 Must be answered before a pod is asked for — bucket (C), zero compute

- [ ] **E16** **C1 is decided**: AMP + gradient checkpointing either tried (with a bit-identity proof) or explicitly deferred **in writing**. ⛔ Without this the pod request cannot say whether it needs a bigger card or three trainer features
- [ ] **E17** **C3 is priced**: the transfer of the 4,713-clip source frames, which are **not on this box**
- [ ] **E18** **C4**: remaining HF quota checked against the **+112.9 GB**, before any push
- [ ] **E19** **C5**: the PI has supplied `--w-agent` and `--w-tac-goal`, or arms **A** and **C** are dropped from the pod request. ⛔ They cannot launch anywhere without them
- [ ] **E20** **C6/C7** are stated in the pod request itself: no `--resume`, no accumulation ⇒ a pod arm cannot be rehearsed here at matched batch, and a dead arm restarts

### 8.4 ⭐ The request the checklist earns

When **E1–E20** are ticked, the pod request is not *"we would like a pod"*. It is:

> Here is the chain, carrying end to end at your geometry, every instrument reading a known value.
> Here is the measured rate for the trunk that fits, and here is the trunk that does not fit and
> why. Here is the held-out occupancy read your collision gate was gated on. Here is the **one**
> lever claim an 8 GiB card could buy, and it cost **55 % of a week**. Here is the arithmetic
> showing the remaining panel is **136 days step-matched and 1,361 days sample-matched** on this
> box. ⇒ **The preparation is finished. The remainder is not a dev-box question.**

---

## 9. What this pre-registration does NOT claim, and what it OWES

1. ⛔ **No capability claim of any kind.** No arm has run. Every bucket-(A) arm is explicitly
   labelled instrument, control or lever, and even the lever (A7) is a **learning-curve** claim on
   **124 clips at 0.38 epochs** — it is **not** `E-REFCV6V2-TRUNK`.
2. ⛔ **Bucket (A) contains no T1 driving claim** and cannot: `E-REFCV6V2-DRIVE` is B2.
3. ⚠️ **The eval cost (6.39 s/window) is ESTIMATED** from two passes that also ran a train step.
   Every hour in §6 that derives from it is therefore an **upper-bound estimate**, and A2
   re-measures it. ⛔ It is not a measurement and is not presented as one.
4. ⚠️ **The 21,207-window figure is ESTIMATED** — 124 clips × the **measured** 171.0 mean. The
   clip count is measured; the window count is a product. A0 replaces it with a count.
5. ⚠️ **The ≈65 s/step `resnet101` figure is ESTIMATED** by trunk-MAC scaling alone. The heads do
   not scale with the trunk. ⛔ Treat as an **order**, never as a rate.
6. ⚠️ **C1 is a HYPOTHESIS**, not a finding. I have **not** measured whether AMP or gradient
   checkpointing changes the `resnet101` answer, and neither has anyone else.
7. ⚠️ **OWED — the `t1_eval.py` wiring at 416 × 1024** (C8). Until A2 answers it, §6's
   eval hours are costed against the *trainer's* eval path, which is the only one measured.
8. ⚠️ **OWED — no pod-day figure is given** (§3.4), deliberately. The pod request must carry a
   200-step rate measurement as its first job.
9. ⛔ **This package launched nothing, changed no model code, and used no GPU.**

---

## 10. Deliverables

| artifact | path |
|---|---|
| this pre-registration | `Project Steering/PREREG_REFCV6_DEVBOX_PREPARATION.md` |
| work package | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-18-refcv6-devbox-plan/RESULT.md` |
| ⭐ the 124-clip census (MEASURED) | `…/2026-09-18-refcv6-devbox-plan/raw/clean_split.json` · `code/clean_split.py` |
| the cost arithmetic (reproducible) | `…/2026-09-18-refcv6-devbox-plan/raw/devbox_cost_table.json` · `code/devbox_cost.py` |

⚠️ **Integration is ESCALATED, not written into a doc:** `SPEC_REFCV6_V2.md` §12.5 and
`PREREG_REFCV6_V2.md` §2/§3 both say **139**; `D-REFCV6-EVAL139-PARITY` says **128**; §2 here
measures **124**. ⛔ Those files are authorities and this package does **not** edit them — the
correction is raised for the owning agent to apply.

<!-- PREREG-REFCV6-DEVBOX-PREPARATION-2026-09-18 -->

---

## ⭐ W-BOOTSTRAP — make the paired episode-cluster bootstrap REACHABLE (spec, 2026-09-18)

⛔ **THE BLOCKER, MEASURED.** `paired_episode_cluster_bootstrap(a, b, eid, ...)` is the
ONLY admissible interval in this programme, and **no refcv6 artifact can feed it.**
Verified on tonight's own run: `metrics.jsonl` carries **0 non-scalar values and 0
episode/clip/window-id keys**. Read from source rather than guessed, it is TWO layers:

1. `compute_losses_v3` **already reduces over the batch**, so a per-window value is never
   formed; the eval loop sums those batch scalars and divides by `nb_e`. The artifact is a
   mean of means.
2. The eval **batch carries no identifier at all** — `frames`, `map_frac`, `agent_box`,
   `nav_cmd`… and nothing saying WHICH clip.
   ⚠️ `agent_valid` / `nav_valid` match an `id`-substring grep and are FALSE POSITIVES.

⇒ an aggregate-only dump cannot be bootstrapped **by any rescore**.

### ⭐ THE DESIGN IS SETTLED BY PRECEDENT — nothing here needs a decision

`sid = stable_episode_id(clip_id)` (`v2_dataset.py:69`): collision-free, 63-bit, from the
FULL clip_id. `build_refcv6_speed_max_window.py:167` already calls it *"the corpus's only
join key"* and **REFUSES a collision rather than resolving it** (`:108-115`).
⚠️ `sid` is an **id, not a clip id**, so it may be written into repo artifacts where a raw
clip_id may not.

### ⛔ The constraint that dictates the shape

`test_refc_v3_u8_batches.py:375` asserts `set(batch) == set(HEAD_BATCH01)` — an **EXACT**
key set. Adding `sid` to the TRAINING batch breaks that pin, and the pin is RIGHT: the
training batch contract must not drift for an eval feature.

⇒ **Gate the key on the flag, so the training path stays bit-identical:**
`--eval-window-dump PATH`, off by default. When set, the **eval** dataset emits `sid` and
the eval iterates **per window** (batch 1, so a batch mean IS the window value), writing
one JSON row per window: `{sid, t0, step, <every scalar compute_losses_v3 returns>}`. The
aggregate row keeps its shape, so every banked comparison is untouched.

### Acceptance — committed in advance

1. ⛔ **PARITY**: flag unset ⇒ `metrics.jsonl` **byte-identical** and
   `test_refc_v3_u8_batches` green. A flag that perturbs the default path has failed
   regardless of what it enables.
2. **REACHABILITY**: the rows load and the bootstrap runs with `eid = sid`.
3. ⛔ **THE DISCRIMINATING CONTROL**: bootstrap an arm **against itself** — the interval
   must contain 0. A harness that "separates" a run from itself is measuring its own bug,
   which is exactly what `H-ESTIM-SEED-1` is about.
4. **n is stated**; a read below the gate's floor is reported as underpowered, not quoted.

⚠️ **Cost**: per-window eval is batch-1 and slower per window. It is a **separate,
opt-in pass**, not a replacement for the in-training monitor.

⚠️ **Not implemented tonight, deliberately**: the GPU is held by the resnet101 fit
measurement (one job at a time), and this touches the live trainer's eval path, so it wants
its parity check RUN rather than argued.

<!-- W-BOOTSTRAP-EVAL-WINDOW-DUMP-SPEC-2026-09-18 -->

### ✅ W-BOOTSTRAP — DONE 2026-09-19, and one of its two premises was WRONG

⭐ **Premise 1 STANDS**: `compute_losses_v3` already reduces over the batch, so the eval's
rows are a mean of means and no per-window value exists. The fix is a batch-1 pass, where a
"batch mean" IS the window value.

⛔ **Premise 2 WAS WRONG, and the correction made the job much smaller.** It said *"the
eval batch carries no identifier at all"*. It carries one: `_contract.py:138` puts
`"episode_id": ep.episode_id` in EVERY item; it is `stable_episode_id(clip_id)`, 63-bit
*precisely so it survives torch's default int64 collate*; and
`test_refc_v3_u8_batches.py`'s pinned key set **contains it**.
⚠️ The bad reading came from grepping for keys the **trainer READS** rather than keys the
**dataset EMITS** — the trainer simply never read it. Same family as every other
wrong-scope probe in this programme: the grep answered a different question than the one
asked, and it answered it correctly.
⇒ **no dataset change, no batch-contract change, no parity risk.** Only the DUMP was
missing.

#### What shipped

`--eval-window-dump PATH`, OFF by default: a **separate** batch-1 pass over the SAME
deterministic window subset, writing one JSONL row per window with `episode_id` and every
scalar the loss returns. The in-training monitor's aggregate row is untouched.

#### Acceptance, all four MEASURED on a real run

| | |
|---|---|
| 1. parity, flag unset | `test_refc_v3_u8_batches` green; dump is a separate opt-in pass |
| 2. reachability | 16 rows loaded; `paired_episode_cluster_bootstrap(a, b, eid=episode_id)` runs |
| 3. ⛔ **self-control** | an arm against **ITSELF**: `lo 0.0, hi 0.0, separated False` — the interval contains 0 |
| 4. n stated | `metrics.jsonl` carries `eval_window_rows = 16`; every result carries `n_windows` / `n_episodes` |

⚠️ That smoke run drew 16 windows across 16 DISTINCT episodes — one window per cluster.
That is the draw, not a property of the mechanism; a real read needs many windows per
episode, and the counts ride along so an underpowered read cannot be quoted bare.

<!-- W-BOOTSTRAP-DONE-AND-PREMISE-CORRECTED-2026-09-19 -->

---

# ⭐⭐ RE-PRICING, 2026-09-19 — the rate this document was built on measured HOST PAGING

⛔ **What is void, and what is not.** §3's PARTITION and §5's REPLICATE CONFRONTATION
**stand unchanged**. Every **hour** in §1, §3 and §6 is **void**, because all of them derive
from **29.1967 s/step** — and that figure was never a compute cost.

## R.1 The measurement that voids it

MEASURED 2026-09-18/19 on this box, and reproduced by hand before being acted on:

| | peak allocated | s/step | fits (`peak_reserved ≤ 7.1 GB`)? |
|---|---|---|---|
| `resnet34` @ batch 2, **as this plan priced it** | **14.628 GB** | 29.1 | ⛔ **NO** — on an **8,188 MiB** card |
| `resnet34` @ batch 2, **+ frozen-BN + chunk-ckpt 1** | **2.512 GB** | **2.8** | ✅ |
| `resnet101` @ batch 1, same levers | **2.887 GB** | **4.2–4.9** | ✅ |
| `resnet101` @ batch 2, same levers | 3.890 GB | **8.55** | ✅ |

⛔ **On this Windows/WDDM box CUDA SPILLS PAST VRAM INTO HOST RAM INSTEAD OF RAISING.** The
arm this plan was priced from was **paging**, so 29.1967 s/step is a host-memory measurement
wearing a compute unit. ⇒ *"it did not raise"* is **not** a fit on this hardware, and the
fit rule used throughout the re-pricing is `peak_reserved ≤ 7.1 GB`.

⭐ **§4.1's own closing instruction is what produced this:** *"Do not close the `resnet101`
question as 'needs a bigger card' until AMP + gradient checkpointing have been tried,
because the current evidence cannot distinguish the two."* They were tried. Chunked
checkpointing fits; AMP alone does not (14.74 GB); timm's own `set_grad_checkpointing`
**raises** on this backbone and, once fixed, still does not fit (15.17 GB).

## R.2 A full 40,284-step arm

| configuration | hours | days |
|---|---|---|
| `resnet34` @ b2, levered | 31.3 | **1.31** |
| `resnet101` @ b1, levered (slow end) | 54.8 | **2.28** |
| `resnet101` @ b2, levered | 95.7 | **3.99** |
| ~~the figure this plan used~~ | ~~326.7~~ | ~~13.61~~ ⛔ **void** |

## R.3 Bucket (A), re-priced — **132.1 h → 33.6 h (5.50 d → 1.40 d)**

⚠️ **Only TRAINING scales.** A3–A6 are eval-only and are **unchanged**; scaling them would
have been the same class of error this document is correcting.

| arm | old | new |
|---|---|---|
| A0 · A1 (no GPU) | 1.0 h | 1.0 h |
| A2 harness + 40 steps | 1.0 h | 0.7 h |
| A3 · A4 · A5 · A6 (eval only) | 12.5 h | **12.5 h** — unchanged |
| **A7** ImageNet knockout (4 arms × 2,000) | 72.0 h | **13.3 h** |
| **A8** occupancy → 5,000 | 42.4 h | **5.7 h** |
| A9 conflict detector | 3.2 h | 0.3 h |
| **TOTAL** | **132.1 h = 5.50 d** | ⭐ **33.6 h = 1.40 d** |

**Headroom against the 7-day ceiling: 134 h** (was ≈ 36 h).

## R.4 The replicate accounting, re-derived

§5's rule is unchanged and still binds. The **share** changes: A7 is now **13.3 h of 33.6 h
= 39.7 %** (was 55 %).
⛔ **The conclusion does NOT change.** A7's replicate remains the **last thing cut**: cutting
it still does not shorten the plan meaningfully (it saves 6.7 h of 33.6) and still **deletes
the plan's only admissible lever claim**. The cheaper the arms get, the *weaker* the excuse
for a one-seed result — a 14.3 % false-positive rate is not made acceptable by a short run.

## R.5 What moves out of bucket (B) — and what does NOT

| item | old | re-derived | verdict |
|---|---|---|---|
| **B1** `resnet101` **step-matched** | ⛔ *"0 steps are possible"* | **2.28 d** @b1 · **3.99 d** @b2 | ⭐ **MOVES TO (A)** |
| **B1** `resnet101` **sample-matched** | — | **39.9–45.7 d** | ⛔ **STAYS IN (B)** |
| **B3** 10-arm panel, **step-matched** | 136.1 d | **13.05 d** | ⚠️ borderline — two weeks of exclusive box |
| **B3** 10-arm panel, **@ the 12,000-step cut** | 40.6 d | **3.89 d** | ⭐ **MOVES TO (A)** |
| **B3** 10-arm panel, **sample-matched** | 1,361.3 d | **130.6 d** | ⛔ **STAYS IN (B)** |

⭐ **THE SHAPE OF THE POD REQUEST CHANGES COMPLETELY.** It is no longer *"the dev box cannot
run the primary trunk"* — it can, in **2.3 days**. What the dev box still cannot do is
**SAMPLE-MATCHED** work: this box runs batch 2 where `refcv5-v2` ran batch 20, so matching
the data a banked arm saw costs **10× the steps**, and that is where the 40–130 day figures
live. ⇒ the pod case is now **specific and defensible** rather than general.

## R.6 ⚠️ Three caveats that travel with every number above

1. ⛔ **`--trunk-frozen-bn` CHANGES THE ARM.** Chunking alone shifts `ga_trunk` by **−40 %**
   on `resnet34`; BN is pinned to ImageNet statistics to make chunking exact (agreement
   **7.2e-6**). **There is NO configuration that both fits and reproduces the unpatched
   arm's BN statistics.** Any comparison against a banked unfrozen-BN arm must say so.
2. ⚠️ **§4.1's reduced-configuration clause is satisfied, and was checked:** the fit is at
   **K = 3** (`--trunk-in-channels 9`) and **batch 2 is available**, so this is *not* a
   reduced geometry and the rate *may* be used to size the pre-registered arm. The frozen-BN
   change is a **different axis** and is declared in (1) rather than hidden here.
3. ⭐ **§4.1's `A10` is DONE and is no longer an estimate.** It asked for a 200-step sizing
   arm at *"≈ 65 s/step (ESTIMATED; re-measure, do not scale)"*. Re-measured: **4.2–4.9
   s/step at batch 1** — the estimate was **13–15× pessimistic**, which is itself an argument
   for the instruction never to scale an estimate.

⚠️ **§3.1's governing constraint is UNCHANGED and still binds:** one epoch over a 62-clip
half is 5,302 steps. At 2.8 s/step that is **4.1 h**, not 43.0 h — so a 2,000-step arm still
sees **0.38 of one epoch**, and **no arm in bucket (A) is a capability claim.** Cheaper steps
buy more arms; they do not buy more data.

<!-- DEVBOX-PREP-REPRICED-2026-09-19 -->

---

## ⛔ CORRECTION, 2026-09-19 — A4/A5/A6 are BUILD-then-run, not run: the S1 harness does not exist

### The measurement

⛔ **1,435 code files under `stack/`, `taniteval/` and `tools/` were scanned. ZERO mention
any S1 arm name (`S1-BASE`, `S1-RANDOM`, `S1-GATE-*`, `S1-AGENTS`, `S1-REPL`) or the
`collided-selection` statistic they are all defined against.** A second, broader probe over
every `.py/.sh/.json/.md` in the tree found the arm names in exactly **one** file — this
document — and in **no code at all**.

⇒ A4 (5.3 h), A5 (1.8 h) and A6 (3.6 h) were priced as **eval runs of an existing
instrument**. There is no instrument. They are **build-then-run**, and their 10.7 h is an
under-estimate of unknown size until the build is scoped.

⚠️ `PREREG_S1` §8's *"`S1-GATE-ORACLE` and `S1-RANDOM` need no trained perception and can
run first"* is TRUE and is about **perception**; it does not say the selection-and-gate
harness exists. Reading it as *"these are cheap"* conflated *needs no trained model* with
*needs no code*.

⚠️ **AND THE ONLY AVAILABLE CHECKPOINT COULD NOT CARRY THEM ANYWAY.** Every S1 arm is
scored on a **fan of candidate trajectories**; the 2,000-step halfA map head has a planner
trained for **0.38 of one epoch** (its paired ADE delta reads ~77 m). A collision statistic
over that fan would measure the untrained planner, not the gate.

### ⛔ And a CONFLATION OF MINE, corrected in the open

When I paused bucket (A) I wrote that A4–A6 should wait because *"Item 25 shapes how their
verdict is read"*. **That was wrong.** There are two unrelated things called `S1`:

| | |
|---|---|
| `verdict_refcv6.py:121` | `Clause("S1", "STRATEGIC", "route_acc not separably worse than control, and n > 0")` — a **clause ID** in the refcv6 verdict panel, about the STRATEGIC family. **This is Item 25.** |
| `PREREG_S1_AGENT_SEAM_AND_COLLISION_GATE.md` | `S1` names the **pre-registration**; `S1-BASE` / `S1-RANDOM` / `S1-GATE-*` are its **arms**, about the agent seam and the collision gate. |

⇒ **Item 25 does NOT gate A4–A6.** I let a shared label imply a dependency — the same
family as every other wrong-scope read in this programme, with the object swapped for a
NAME. The real blocker is the one above: the instrument does not exist.

### What this does not change

A0 ✅ A1 ✅ A2 ✅ A3 ✅ stand, and A3 discharged `PREREG_S1` §8's blocking dependency at
**1.70x the floor, held out**. The S1 arms are what would *use* that occupancy; building
their harness is the next real work item, and it is a **PI-visible scope change** rather
than a 10.7 h slot in an existing plan.

<!-- A4-A6-INSTRUMENT-DOES-NOT-EXIST-2026-09-19 -->

<!-- A9-DONE-2026-09-19 -->
## ✅ A9 DONE, 2026-09-19 — the conflict detector's cost at 416 × 1024, re-measured rather than scaled

MEASURED (ours) on A3's exact configuration (resnet34, clean-124 halfA, `--trunk-chunk-ckpt 1
--trunk-frozen-bn`, batch 2, seed 0), 200 steps `--conflict-detector off` then 200 steps `on`,
nothing else changed. Rates are `elapsed_s` MARGINAL deltas at `--log-every 10`, steps ≤ 30
excluded as warm-up.

| arm | median s/step | mean | n (deltas) | wall |
|---|---|---|---|---|
| detector **off** | **2.82** | 2.759 | 16 | 640 s |
| detector **on** | **5.07** | 5.095 | 16 | 1,037 s |
| **overhead** | ⭐ **+79.8 %** (median ratio) | | | |

**Identity controls — all EXACT, no epsilon (the §7 criterion):** `cos(g, g)` = **1.0**,
`cos(g, −g)` = **−1.0**, detached `cos` = **NaN**, detached conflict **0.0**, detached `|g_aux|`
**0.0**. The OFF arm wrote **no** `conflict_controls` row (the detector was really off); the ON
arm wrote conflict keys on **20/20** logged rows.

⇒ **E14 is TICKED.** The +79.8 % sits **inside** the banked +71.7 … +104.2 % that was measured on
the smaller grid, so the new geometry did not push the detector's cost past its old upper end —
but the ruling was *"re-measure rather than scale"*, and now it is measured. ⚠️ It is the cost
**with the detector reading every step**; the pod budget should carry it at the cadence the pod
arm will actually use, not this worst case.
⛔ NOT a capability claim: 200 steps × batch 2 = 0.075 of one halfA epoch. n = 16 deltas per arm
is thin for a rate; it is quoted as an overhead RATIO measured back-to-back on one box, which
is what the criterion asks for. Artifacts:
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-19-a9-conflict-cost/`.

<!-- E17-E18-CLOSED-2026-09-19 -->
## ✅ E17 and E18 CLOSED, 2026-09-19 — by the DataFlyWheel; landed by the Master Mind after a content and leak scan (its numbers are the DataFlyWheel's MEASURED values, not re-measured here)

### ⛔ E17 / C3 — constant 13 was an ABSENCE FROM A SINGLE PROBE, and C3's transfer leg is VOID
Constant 13 said *"the 4,713 train clips' source frames are not on this box"*. It was measured
on `D:/Projects/TanitAD-artifacts` ONLY. They ARE on the box: `C:/Users/Admin/tanitad-data/
physicalai/camera/camera_front_wide_120fov` (NTFS) — 4,719 mp4s + 4,719 timestamp parquets,
checked against HF's own per-file size and LFS sha256 (revision 0ddee95d): **4,719/4,719 match,
0 mismatches, 61.6 GB** (MEASURED). The 4,713 train clips reproduce the B1 membership digest
`e8bfb98e` exactly (positive control). **Bytes: 61,545,041,700 B; transfer: 0 h** (local read
≥ 826.5 MB/s incl. sha256, MEASURED). ⇒ **the corpus rebuild is blocked ONLY by C2 (SAM3)**, not
by any transfer. Two small items remain: a local-mirror staging mode for `fetch_corpus_clips.py`
is not built, and the two sidecar tars (2.02 GB) live in `C:/Users/Admin/tanitad-wt/_s2build` —
the mirror that has DELETED repo-absent files on resync — and must be copied to D: before the
rebuild. Package: `TanitAD Research Lab/Data Engineering/Research/2026-09-19-c3-source-transfer-price/`.

### ✅ E18 / C4 — PASS, with three flags the PI must see
PRIVATE storage used **176.554 GB** over 20 repos (MEASURED, per-repo `usedStorage`) against the
PRO **1 TB** private allowance (PUBLISHED, HF storage-limits doc) ⇒ **823.4 GB free**; a PRIVATE
push of the 408×1024 cache (**386.5 GB**, ESTIMATED and probably high) leaves **436.9 GB**. PASS.
- ⚠️ **+112.9 GB is NOT a push size** — it is the 408-vs-256 geometry difference; no 273.6 GB
  cache exists on HF to replace, so a push adds the full 386.5 GB.
- ⛔ **THE CEILING DOES NOT ENFORCE ITSELF.** The account reads `canPay: true`, prepaid billing:
  storage above 1 TB is billed pay-as-you-go ($18/TB/mo), no 402/413 stops a push, and no
  endpoint exposes the ceiling. ⇒ **pre-push arithmetic is the ONLY guard.**
- ⛔ `Sayood/tanitad-physicalai-w120-256x640cyl` is **455.4 GB and PUBLIC**, and **3,000 of its
  6,061 file names carry raw clip ids**. Making it private AND pushing the 408 cache reaches
  **1,018.4 GB — 18.4 GB over, billed**. → `PI_DECISION_QUEUE.md` ITEM 26.
Package: `TanitAD Research Lab/Data Engineering/Research/2026-09-19-c4-hf-quota-check/`. (Also: the 18 D:-divergent Data files need NO merge — HEAD is
authoritative for all 18: 5 blob-identical, 10 redaction-only, 2 header+backup, 1 stub-filled;
mutation-tested. `TanitAD Research Lab/Data Engineering/Research/2026-09-19-d-sync-divergence-resolution/`.)


---

<!-- A7-AMENDMENT-BN-2026-09-19 -->
## ⛔ A7 AMENDMENT, 2026-09-19 — BatchNorm handling, pre-registered BEFORE ANY DATA

**Author:** TanitAD_TrainingFlyWheel (A7 owner, assigned by the Master Mind 2026-09-19).
**Status at the time of writing: 0 A7 GPU seconds spent. No A7 arm has been launched.**
Every fact below is MEASURED (ours) unless labelled otherwise.

### A7.1 ⛔ The confound — a SECOND variable the memory levers smuggle into A7

A7 must run with the memory levers (`--trunk-chunk-ckpt 1 --trunk-frozen-bn`): without them
the arm allocates **14.6 GB on an 8 GiB card and pages** (R.1). `--trunk-frozen-bn` pins every
backbone BatchNorm to **eval mode**, where it normalises with its **stored running statistics**
(`timm_trunk.py::_freeze_bn_`, which also replaces the bound `train` so the trainer's own
`model.train()` cannot undo it).

| condition | what the stored statistics ARE | what frozen BN therefore does |
|---|---|---|
| ImageNet-init | ImageNet's per-channel mean/var | a real normalisation — of **ImageNet's** input distribution |
| **random-init** | ⛔ **MEASURED: 36 BN layers, every `running_mean` exactly 0.0, every `running_var` exactly 1.0**, `bn_training_count` 0 | `(x − 0)/√(1+ε)·γ + β` with `γ = 1, β = 0` ⇒ ⛔ **THE IDENTITY. No normalisation at all.** |

*(Measured on `resnet34.a1_in1k`, `pretrained=False`, `frozen_bn=True`, K = 3, CPU, 2026-09-19.)*

⇒ **Unfixed, A7 compares "ImageNet weights + a normalisation" against "random weights + no
normalisation".** A random-init deficit would be uninterpretable — it could be the missing
prior or the missing normalisation — and §5's rule that A7 is the plan's **only admissible
lever claim** would be spent on a two-variable result.

### A7.2 ⭐ The BN handling, pre-registered: RECALIBRATE BOTH ARMS, THEN FREEZE

A new trainer flag, **`--trunk-bn-recalib N`** (default **0 = OFF**, bit-identical to today when
off), applied **identically to all four arms**:

1. **When:** once, after the model is built and moved to the device — i.e. after the seeded
   random init (`torch.manual_seed(args.seed)` precedes the build, `refc_v3_train.py`) — and
   **before step 1**.
2. **What:** the canonical `torch.optim.swa_utils.update_bn` procedure on the trunk's backbone
   BatchNorms only: `reset_running_stats()`, `momentum = None` (the exact cumulative mean over
   batches), BN in train mode, forward under `no_grad`, statistics kept; then BN is frozen
   exactly as `--trunk-frozen-bn` does today.
3. ⛔ **Chunked checkpointing is BYPASSED during recalibration.** With `--trunk-chunk-ckpt 1`
   every BN "batch" would be **one image**; averaging per-image variances omits the
   between-image variance, so `running_var` would be **systematically under-estimated**.
   Recalibration therefore runs the backbone on the **full** trunk batch (no gradients are
   kept, so the memory cost the lever exists to avoid does not arise).
4. **On what:** a FIXED set of **N = 256 halfA windows** (the TRAIN half — never halfB), drawn by
   **`--trunk-bn-recalib-seed 0`**, a generator **independent of `--seed`**, so all four arms
   recalibrate on **byte-identical windows in the same order**. The drawn index list is
   digested (`sha12`) into `config.json` so the identity is checkable after the fact.
5. **The trunk input is the forward's own:** `frames.reshape(b·w, …)` on the hierarchy path,
   `frames[:, -1]` otherwise — the same rule `refc.py`'s forward applies, pinned by a test that
   captures the encoder's real input with a forward pre-hook and requires `torch.equal`.
6. **Stamped:** `config.json` carries `trunk_bn_recalib: {n_windows, seed, windows_sha12,
   n_batches, n_images, n_bn, stats_sha12}`; with the flag off it carries
   `trunk_bn_recalib: null` (the baseline, distinguishable from absence). The recalibrated
   statistics themselves are banked as `bn_recalib_stats.pt` in the run directory.

⚠️ **Declared consequence:** the ImageNet arm's running statistics become **driving-data
statistics**, no longer ImageNet's. A7's ImageNet arm is therefore **deliberately NOT
bit-comparable with A3 or A8**, which ran with ImageNet statistics. That is the price of making
the two A7 conditions differ in the weights only, and it is paid knowingly.

### A7.3 ⭐ Two controls that must read known values

**(a) The identity control — both ImageNet arms must recalibrate IDENTICALLY.** Before step 1
their trunk weights are identical (a fixed download) and their windows are byte-identical, so
their recalibrated statistics must agree **to floating-point non-determinism** (cuDNN
convolution algorithms need not be bit-reproducible, so an exact byte identity is NOT the
criterion). Reported as the max relative difference over every BN channel, **and read as a
RATIO against the ImageNet-vs-random difference of the same statistics** — never against a
remembered epsilon. ⛔ If the same-condition difference is not orders of magnitude below the
between-condition difference, the recalibration is seed-dependent, the arms are not what they
claim, and **A7 stops**.

**(b) The mutation control — the flag must be REACHABLE, not merely correct.** Per the
2026-09-10/11 binding rule (*"built, tested, and unreachable from its caller"*): the test
suite proves (i) OFF is bit-identical, (ii) ON changes the backbone's running statistics
**from the trainer's own entry point**, (iii) the stamp is written, and (iv) ⛔ deleting the one
line that performs the recalibration turns the test **RED**.

### A7.4 ⚠️ The residual confound, measured rather than assumed away

Recalibration equalises the **start**. Frozen statistics then go **stale** as the weights train,
and a random trunk moves further in 2,000 steps than an ImageNet one — so the random arm's
normalisation may be **staler at the end**. Pre-registered diagnostic, computed on the trained
trunk **of every arm** on the same 256 windows (into a copy — the checkpoint is not altered):

* `bn_staleness_var` = median over BN channels of `|ln(var_frozen / var_true)|`
* `bn_staleness_mean` = median over BN channels of `|mean_frozen − mean_true| / √var_true`

**How it qualifies the verdict, committed now:** if the between-condition difference in
staleness **exceeds the within-condition seed spread of staleness**, a random-init deficit is
reported as **partly attributable to stale normalisation** and the verdict says so in its first
sentence. ⛔ It never changes the criterion below; it changes only how much of an effect may be
attributed to the prior.

### A7.5 The four arms — the ONLY differences from the A3 template

`C:/Users/Admin/qland/a3_heldout_read.sh`'s configuration, with exactly these changes:

| arm | `--trunk-…pretrained` | `--seed` | plus, on every arm |
|---|---|---|---|
| **A7-IN-s0** | `--trunk-pretrained` | 0 | `--trunk-bn-recalib 256 --trunk-bn-recalib-seed 0` |
| **A7-IN-s1** | `--trunk-pretrained` | 1 | `--eval-every 2000` (one read, at step 2,000) |
| **A7-RND-s0** | `--no-trunk-pretrained` | 0 | `--eval-window-dump <arm>/eval_windows.jsonl` |
| **A7-RND-s1** | `--no-trunk-pretrained` | 1 | `--out <arm-specific>` |

Everything else — `resnet34.a1_in1k`, 416 × 1024, batch 2, 2,000 steps, `--trunk-in-channels
9`, the levers, every head and weight, `--conflict-detector off`, halfA training, halfB held-out
`--eval-batches 500` — is **byte-identical** across the four. Run **one at a time**, only when
`boxstat.py` reads GPU ≤ 2,500 MiB and host free ≥ 8 GB.

### A7.6 ⭐ The verdict — §7's A7 row, made computable, committed before the data

* **Metric:** `eval_traj` — the held-out **planner** loss (§7: *"on the planner loss"*) — at
  **step 2,000**, over the **1,000 fixed halfB windows** (`torch.Generator().manual_seed(12345)`,
  `refc_v3_train.py`, independent of `--seed`, hence byte-identical windows across all four arms).
* **Effect** `E` = mean over the two seeds of the RND arm's `eval_traj` minus mean over the two
  seeds of the IN arm's. **Positive = ImageNet better** (lower loss).
* **Seed floor** `F` = measured **in this same panel**: the larger of the two within-condition
  seed differences, `max(|IN_s0 − IN_s1|, |RND_s0 − RND_s1|)` — the conservative choice.
* ⭐ **The headline is the ratio `R = E / F`** (the Master Mind's instruction), quoted with `E`,
  `F`, and all four arm values.

| outcome | condition | what I will report |
|---|---|---|
| ⭐ **SUPPORTS** | **both** IN arms below **both** RND arms (all four cross-pairs favour ImageNet), **and** `R > 1` | ImageNet-init is ahead of random-init on the held-out planner loss at matched steps, beyond the seed floor measured in the same panel |
| ⚠️ **UNDERPOWERED** — ⛔ *not* REFUTED | `R ≤ 1`, i.e. the gap is inside the seed floor | *"the ImageNet prior is not visible at 2,000 steps on 124 clips"* — and the pre-committed reading that `E-REFCV6V2-TRUNK` **needs corpus scale**, which is evidence for the pod |
| **RANDOM AHEAD** | `E < 0` beyond the floor | reported exactly as measured |
| ⛔ **VOID** | A7.3(a) fails, or any arm crashes / writes no `eval_windows.jsonl` | no verdict; the failure is the finding |

**Supporting statistics, reported alongside and never substituted for `R`:** the paired
episode-cluster bootstrap (`taniteval.ci`) on `eval_traj`, IN vs RND, per seed-matched pair —
⛔ with the explicit statement that it answers *"would another draw of episodes say this?"* and
is **blind to training variance**, which is exactly why `R` is the headline and not this.

### A7.7 ⚠️ What this panel cannot establish — stated now so it is not discovered later

1. ⛔ **`F` rests on two seeds per condition** — one difference each. A ratio near 1 is not
   decisive, and it is reported with that caveat rather than rounded into a verdict.
2. ⛔ **NOT a capability claim.** 2,000 steps × batch 2 = 0.38 of one halfA epoch (§3.1).
   Tier: **T0, held-out, open-loop, trainer-side read.**
3. ⚠️ **The four metric families.** The held-out read emits family **losses**
   (`eval_lon`/`eval_lon_tac`, `eval_lat`/`eval_lat_tac`, `eval_tac_v6`/`eval_goal_tac`/
   `eval_anchor_acc`, `eval_route`) and they are reported per arm with the same seed-floor ratio.
   It does **not** emit the doctrine's family **metrics** (headway/TTC, curvature and yaw-rate
   error, manoeuvre confusion, route accuracy). ⛔ That gap is recorded as a **work item**, not
   waved through; closing it is out of A7's scope and needs the T1 harness (C8).
4. ⛔ **No extra seeds, no metric switch, no post-hoc threshold.** If the result is ambiguous,
   it is reported as ambiguous.

### A7.8 ⛔ PRE-DATA ADDENDUM, same day — the window dump A7.6 leans on was measuring something else

**Status at the time of writing: still 0 A7 GPU seconds.** Found while wiring A7.6's supporting
statistic; fixed, tested and mutation-proven **before any A7 arm launched**.

1. ⛔ **The defect (MEASURED).** `refc_v3_train.py`'s W-BOOTSTRAP per-window pass ran *after*
   the aggregate eval had restored train mode, i.e. **in TRAIN mode**: `ego_dropout = 0.5` and
   `route_dropout = 0.5` fired on the held-out windows; `compute_losses_v3`'s
   `if model.training:` gate **re-opened C-REFCV3-EVAL-PRIOR-LEAK** (held-out labels EMA'd into
   `core.lat/lon_log_prior`); and its draws plus the loader iterator moved the training RNG.
   * the only real run that used the flag (`refcv6-windump-20260919`, step 1, 16 windows, 0 GPU
     to re-read): row-mean `traj` **14.50351** vs the aggregate `eval_traj` **14.59615** on the
     SAME windows;
   * the synthetic rig (`stack/tests/test_eval_window_dump_mode.py`, pre-fix): rows **11.1558**
     vs aggregate **10.7542**; `lat_log_prior` moved from `[-1.1087, -1.1087, -1.0788]` to
     `[-1.0593, -1.1489, -1.0897]`; **170 tensors** differed after one later training step
     between dump-ON and dump-OFF.
2. **The fix:** `model.eval()` plus `_RngIsolated(device, None)` (torch/numpy/python RNG forked,
   **not** reseeded) around the dump pass. Five tests pin it through `T.train`; mutation **M6**
   (delete the `model.eval()`) and **M7** (delete the fork) both go RED with the control GREEN
   (`raw/mutation_proof.json` in the A7 package).
3. ⚠️ **A second, independent non-equivalence — found by the same test, and it survives the fix.**
   `loss_traj = Σ|err|·sv / (2·Σsv)` is a **valid-slot-weighted** mean over the batch, and
   `eval_traj` averages **batch-2 pairs**. So even in eval mode the plain mean of the batch-1 rows
   is NOT `eval_traj` whenever windows carry unequal valid futures — which real data does (the
   acceptance run: 15/16 windows full, 1 at 0.5; of its 0.0926 gap, rebuilding its train-mode rows
   by the rule below closes 0.0611 and the remaining 0.0316 is the train mode — one ordering of two
   effects that need not add). The rows reproduce `eval_traj` **exactly** by the loss's own rule —
   `Σ traj_w·frac_w / Σ frac_w` per consecutive pair (`frac` = `slot_valid_frac`), then the mean
   over pairs; test (b) asserts it to rel 1e-5.
4. ⭐ **Committed now, before any data — the bootstrap statistic.** A7.6's supporting paired
   episode-cluster bootstrap computes, per draw and per arm, the **valid-slot-weighted ratio**
   `Σ traj_w·frac_w / Σ frac_w` over the resampled windows — the loss's own weighting, without the
   batch pairing, which has no meaning once episodes are resampled. The **plain window mean** is
   reported beside it as a sensitivity row. Both are supporting only; the headline stays
   **`R = E / F` on `eval_traj` from `metrics.jsonl`**, which this defect never touched (the
   aggregate block was already in eval mode).
5. **A per-arm dump gate (validity, not a verdict criterion):** after each arm the rows must rebuild
   that arm's `eval_traj` by rule 3 to rel 1e-4. An arm whose dump fails is VOID **for the
   bootstrap only**; `R` is unaffected.
6. **Scope outside A7:** of the dev-box runs, only the W-BOOTSTRAP acceptance run used the flag —
   A2, A3, A8 and A9 did not (read from each run's `config.json` argv). Its reachability claim
   (the bootstrap RUNS end to end) stands; its rows were train-mode rows and are not a measurement
   of the eval.
7. **Correction to A7.2 item 5, pre-data:** the trunk-input test does NOT use a forward pre-hook —
   the hierarchy path calls `encoder.forward_features(...)` directly, so a module pre-hook never
   fires there (measured while writing it). The test wraps `forward_features` itself, on both
   branches, and requires `torch.equal`.

<!-- /A7-AMENDMENT-BN-2026-09-19 -->

<!-- W-BOOTSTRAP-DUMP-DEFECT-2026-09-19 -->
## ⛔ CORRECTION TO W-BOOTSTRAP (2026-09-19) — the per-window dump ran in TRAIN mode

Found by the TrainingFlyWheel while building A7, verified and landed by the Master Mind.
`--eval-window-dump` (W-BOOTSTRAP, `1d1e148`) ran its batch-1 pass AFTER the aggregate eval had
already called `model.train()`, so on held-out windows: ego_dropout 0.5 and route_dropout 0.5
FIRED; the `if model.training:` gate EMA'd held-out labels into `core.lat/lon_log_prior` —
re-opening **C-REFCV3-EVAL-PRIOR-LEAK** — and the dropout draws moved the training RNG.
MEASURED on the only run that ever used the flag (`refcv6-windump-20260919`, same 16 windows):
dump-row mean traj **14.50351** vs aggregate `eval_traj` **14.59615**.
⇒ W-BOOTSTRAP's **reachability** claim STANDS (the bootstrap runs); its **row values are VOID** —
they were never a measurement of the eval. ⚠️ Scope, checked by argv: **A2, A3, A8 and A9 did NOT
use the flag**, so none of their numbers is touched.
**Fix:** `model.eval()` plus an RNG fork (`_RngIsolated`, forked, not reseeded) around the dump
pass; 5 new tests, all RED before the fix and GREEN after; 7/7 mutations caught. A second,
independent issue survives the fix and is pre-registered in **A7.8**: `loss_traj` is
valid-slot-weighted, so the plain mean of batch-1 rows is not `eval_traj` — rows rebuild it exactly
as `sum(traj·frac)/sum(frac)` per consecutive pair.
⛔ And a pinned test (`test_refc_v3_save_before_eval`) was already RED at HEAD from W-BOOTSTRAP's
second `model.train()` — a defect I shipped without seeing it. Retraction:
`RETR-2026-09-19-WINDUMP-TRAIN-MODE`.

<!-- E16-E19-CLOSED-2026-09-19 -->
## ✅ E16 and E19 CLOSED FROM THE RECORD, 2026-09-19 — neither needs a new PI value

### E16 / C1 — "resnet101 may be a TRAINER gap, not a card gap": DECIDED, and the answer was the trainer
C1 asked for AMP + gradient checkpointing to be *"tried (with a bit-identity proof) or explicitly
deferred in writing"*. **Gradient checkpointing was tried and is proven:** leading-batch chunk
checkpointing of the trunk (`--trunk-chunk-ckpt 1`) with BN pinned to ImageNet statistics
(`--trunk-frozen-bn`) fits resnet101 at 416×1024 in **2.887 GB** (batch 1) / **3.890 GB** (batch 2)
on the 8 GiB card (MEASURED, `8b1f1db`); chunking with frozen BN agrees with the unchunked forward
to **7.2e-6** (R.6). Both are real, stamped flags (`0f6036d`). ⇒ C1's hypothesis is CONFIRMED: the
OOM was *"not fitting as the trainer was written"*, never *"the card is too small"*.
**AMP is DEFERRED, in writing:** it is not needed to fit, it would change the arm's numerics (a
second variable against every banked fp32 arm), and it is a separate knockout if ever wanted.
⚠️ The frozen-BN caveat of R.6 (1) still travels: *there is no configuration that both fits and
reproduces the unpatched arm's BN statistics*; A7's recalibration amendment (A7.2) handles it for
the one lever claim that depends on it.

### E19 / C5 — "the two values that are the PI's": RESOLVED FROM THE RECORD for refcv6 V2
* **`--w-agent` = 1.0** is PRE-REGISTERED (`…/2026-09-07-p1-agent-gate/PREREG.md:46`, under
  `D-P1-AGENTCOND-1`, written before any outcome) and is the value every dev-box instrument
  (A2, A3, A8, A9) has run with. ⚠️ C5's *"NOT PRE-REGISTERED"* is corrected by this pointer.
* **The V2 tactical weights are FIXED BY THE SPEC, not open:** `SPEC_REFCV6_V2.md:90` —
  *"BCE 0.05 on goal tokens, CE 0.025 per action head, inside the existing MANEUVER_WEIGHT
  budget"* — and the trainer carries them INSIDE `--w-tac-v6` (`refc_v3_train.py`, the flag's
  own help: *"goal BCE 0.05 + lat CE 0.025 + lon CE 0.025, i.e. the existing MANEUVER_WEIGHT
  budget"*), whose outer multiplier every instrument runs at 1.0.
* **`--w-tac-goal` belongs to the OLDER V1 panel** (`…/2026-09-10-refcv6-build/code/arms.py`:
  lever `C  --w-tac-goal <PI>`, default 0.0), which drives refcv5-v2's 22-token head — the head
  V2's tactical decoder replaces. It gates only V1 arm C, a bucket-(B) pod arm.
⇒ **E19 needs no PI value for refcv6 V2.** If the PI later wants the V1 10-arm panel run on the
pod, `--w-tac-goal` becomes a live question then (queue item 10 holds its history).

<!-- S1-DESIGN-2026-09-19 -->
## ⚠️ A4–A6 DESIGN DECIDED + a dependency the prereg missed (2026-09-19)

Proposed by the TrainingFlyWheel (now owner of the S1 harness and E9), decided by the Master Mind;
all of it is pre-registered by the harness owner BEFORE any S1 data exists.
* **The fan:** refcv6's OWN emitted 117-anchor fan from the **A8** checkpoint (0.95 of a halfA
  epoch), deterministic, on the SAME 1,000 halfB windows A3 read; ONE forward pass per window;
  all five inference-only arms read post-hoc through the UNCHANGED `pdm_proxy.score_candidates`.
  ⇒ ~1 pass instead of the 10.7 h budgeted for A4–A6. ⚠️ A **deviation**: PREREG_S1's 28/493,
  55.3 % and +0.0485 were measured on refcv5-v2's DDv2 fan, so the collision-free share here is a
  NEW measurement, not a check. A6's "second inference draw" becomes a control that must read
  exactly 0 (deterministic fan). Internal validity holds; external validity to a trained planner
  does NOT (anchor_acc 0.092 vs chance 0.0085) and is stated in every verdict.
* **⛔ The dependency:** the collided-selection statistic is AGENT NC, so `S1-GATE-PRED` needs a
  **held-out BOX read** (A8's box head on the same windows: AP vs `random_ap_base_rate`, velocity
  MAE) in addition to A3's map read. A3 discharged §8 as written; the box half is owed
  (`D-S1-DEP-BOX`).
* **A diagnostic arm, ORACLE-CV:** recorded t0 boxes, constant-velocity extrapolated — splits
  ORACLE−PRED into a no-motion-forecast part and a detection part.
* **E9 re-scoped:** `taniteval/tools/t1_eval.py` does NOT run a refcv6 checkpoint by design (0 `refc`
  references; its `roll_closed` drives the flagship's action loop, which refc_v3 lacks). The refcv6
  route is `refcv3_arm.py` → `t1_eval.analyze`, which already ran end to end on a refcv6 checkpoint
  at 416×1024 in W1 (`4000946`). What E9 still owes is its **s/window**.

<!-- S1-FAN-128-ERRATUM-2026-09-19 -->
### ⚠️ ERRATUM to "A4–A6 DESIGN DECIDED" above: the fan is **128** candidates, not 117

Found by the TrainingFlyWheel before any S1 data. MEASURED: A8's and A3's `config.json` both record
`anchors = {shape [128, 8, 2], source "refc.default_anchors (SYNTHETIC)", path null,
v0_conditioned false}`, and neither argv passes `--n-anchors`. "117" is the **refcv4b fitted
bank** (PREREG_S1 §4's held-constant list); the Master Mind's decision text carried it without
re-reading the checkpoint record. ⇒ chance is **1/128 = 0.0078** (not 0.0085), so A3's
anchor_acc 0.092 is **11.8×** chance; N is READ from the checkpoint, never a literal; RANDOM's
expectation is over all N; and because the bank is synthetic and speed-independent, fan min-ADE is
reported per speed tercile. Pre-registered in PREREG_S1 as *S1A ERRATUM-1*. Class:
`D-EVALTOOL-ANCHOR-CHANCE` (2026-09-06) with the direction reversed — a constant carried from one
checkpoint's record into another's.

<!-- A8-DONE-2026-09-19 -->
## ⚠️ A8 DONE — the occupancy curve is NOT monotone, and after 2,000 steps it is FLAT inside the rig's own run-to-run spread

A3's configuration run to **5,000 steps** on clean-124 halfA (0.94 of ONE epoch), with a held-out
halfB read of 1,000 windows every 1,000 steps. MEASURED (ours); the checkpoint is `ckpt_5000.pt`.

| step | held-out `map_iou_drivable` | × the halfB floor 0.3388 |
|---|---|---|
| 1,000 | 0.45049 | 1.330 |
| 2,000 | **0.56906** | 1.680 |
| 3,000 | 0.54010 | 1.594 |
| 4,000 | 0.57437 | 1.695 |
| 5,000 | **0.58379** | 1.723 |

⛔ **§7's A8 row asked for a MONOTONE rise. It is not monotone** (3,000 falls 0.029 below 2,000),
and the whole 2,000 → 5,000 gain is **+0.0147**. ⚠️ Read that against the rig's own spread, which
this run measured for free: **A3 and A8 are the SAME configuration and the SAME seed**, and the
trainer is not reproducible across launches, so they are two draws — **0.00701 apart at 2,000 and
0.03967 apart at 1,000** (n = 2: an INDICATION of the floor, not a replicate panel). ⇒ the 3,000-step
gain is **inside** that spread at 1,000 and barely outside it at 2,000.
⇒ **The pre-committed reading applies:** the head has SATURATED on 62 clips; **more dev-box steps
are not the lever**, and the schedule's first cut (A8) can stay cut. What the run does buy: the
S1 pass its checkpoint (0.94 epoch, the best available), and a second held-out confirmation that
the occupancy clears its floor by **1.72×**.
⚠️ Rate, stated for honesty and NOT for planning: train-only median **4.74 s/step** (492 deltas),
wall 40,131 s including five 1,000-window reads. The box was shared with a NAVSIM caching job and
D: I/O all afternoon, so this is a CONTENDED figure, not the 2.51 s/step planning rate.
⛔ NOT a capability claim: a map head at 0.94 of one epoch over 62 clips.
Artifacts: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-19-a8-occupancy-5k/`.

<!-- CHECKLIST-INDEX-STALE-2026-09-20 -->
## ⛔ THE CHECKLIST AT §E IS A STALE INDEX OF THIS FILE'S OWN CONTENTS (2026-09-20)

**The defect:** §E's boxes read **6 ticked / 14 unticked**. Every closure since has been landed as
an **APPENDED SECTION further down this same file**, and no box was re-ticked. ⇒ a reader going
top-down gets **6/20** and a blocker list this file itself refutes below.

⭐ **The line that proves it is not a bookkeeping quibble:** E17's box still reads *"the
4,713-clip source frames, **which are not on this box**"* — while **§"E17 and E18 CLOSED"** below
records that constant as **an ABSENCE FROM A SINGLE PROBE** and C3's transfer leg as **VOID**, and
the pod request's P9 carries the positive measurement (4,719/4,719 matching HF size + LFS sha256,
61.6 GB, `2a88524`). **The same file asserts both.**

⚠️ Same class as the pinned feature-count rot in `CLAUDE.md` (*"2 of 36"* → 4 → 5 → 6, across
14 documents) and the 2026-08-16 stale-blocker sweep: **a count in prose rots in lockstep with
what it counts**. ⛔ The boxes are NOT rewritten here — appended beside, so the record shows what
was believed and when it was corrected.

### The closure index, by what can be QUOTED — not by what is remembered

| state | items | the evidence |
|---|---|---|
| **CLOSED, box already ticked** | E1 E2 E3 E4 E5 E6 | in §E |
| **CLOSED by a section BELOW §E in this file** | E14 E16 E17 E18 E19 | *"✅ A9 DONE"* (⇒ *"E14 is TICKED"*), *"✅ E16 and E19 CLOSED FROM THE RECORD"*, *"✅ E17 and E18 CLOSED"* |
| **CLOSED by an artifact outside this file** | E9 E10 E12 | E9: `t1_eval.py` declared unwired for any REF-C arm AND the rate recorded — ⚠️ **≤ 19.2 s/window on CPU, all-in**, an upper BOUND, per `…/2026-09-19-e9-t1eval-refcv6/RESULT.md:37` (see `RETR-2026-09-20-E9-SWINDOW`) · E10: A3's held-out table, IoU **0.57607** vs floor **0.3388** = **1.700×** · E12: `S1-GATE-CONST` **exactly 0.00000 [0, 0]** with the gate proven to re-order under real tracks |
| ⛔ **OPEN** | **E13** | A7's four arms. The GPU is held by the PI's own servers; the launcher is armed at an unchanged gate |
| ⚠️ **NOT RE-ADJUDICATED HERE** — believed closed, NOT quoted as closed | E7 E8 E11 E15 E20 | each needs one artifact read, named below |

⛔ **The five I refuse to tick, and exactly what each needs** — because ticking from memory is the
defect this section exists to correct:
* **E7** — A0's 62/62 rebuild **from the 124** with the parity record stamped into each manifest.
  The pod request's P4 cites `782571c` for *"two disjoint 62/62 halves, parity-guarded"*; what is
  not re-read is whether the **stamp** is in each manifest.
* **E8** — each half's occupancy floor **in the same panel the head is scored in**. ⚠️ A caveat
  in the register cuts directly at this: the banked 2,000-step map head **trained on the FULL
  139-clip cache**, so a per-half floor from that panel is not clean by construction. Read it
  before ticking.
* **E11** — `S1-RANDOM`, `S1-BASE` and `S1-GATE-ORACLE` on the **same windows** with the fan's
  collision-free share ≈ **55.3 %**. The 55.3 % is in the register for `L1-NORL-s0` (25/25); what
  is unread is the same-windows condition across the three arms.
* **E15** — the branch. `resnet101` **fits** (`8b1f1db`, real stamped flags `0f6036d`) and a
  dev-box rate of **2.51 s/step** is recorded (P3) — ⚠️ but **§E4 records 29.1967 s/step for
  `resnet34`**, and which trunk the 2.51 belongs to is **not stated beside the number**. That is
  the units/scope class: a true rate quoted without its arm. Settle it from the run's `argv`.
* **E20** — `POD_REQUEST_REFCV6.md` exists and §4 states C6/C7, so the item's text is satisfied;
  but the request is still a **DRAFT with §5 non-empty**, so it is recorded as **drafted, not
  discharged**.

⭐ **The durable fix, and it is the same one `CLAUDE.md` already uses for the feature count:**
a status list that is maintained by hand next to content that moves **will** drift. ⇒ **Do not
quote §E's boxes.** Quote this index, and when an item closes, append its closure **and** its row
here in the same commit — or pin the count with a test, which is what stopped the feature-count rot
after four repeats.
