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
