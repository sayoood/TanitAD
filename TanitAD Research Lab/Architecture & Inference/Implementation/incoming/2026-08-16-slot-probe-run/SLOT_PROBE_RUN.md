# F-18 SLOT PROBE — THE RUN · does the v6 world-model latent carry other agents?

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` · **Agent:** slot-probe-run
**Pre-registration:** `…/incoming/2026-08-16-agent-slot-decoder/AGENT_SLOT_DECODER.md` — followed, not redesigned.
**Eval tier:** ⛔ **T0-DIAGNOSTIC.** A frozen-latent readout is a world-model diagnostic and is
**never** driving performance. No number here may be quoted as a driving claim.

> **The PI's question:** *"do the extraction training based on the last checkpoint to see if there
> is a progress in the wm learning the environment which was one of the reasons why we changed
> architecture and training approach"*

---

## ⭐ THE ANSWER, IN ONE PARAGRAPH

**The read is NEGATIVE, and it is negative at the ENCODER, not at the readout.** At both measured
checkpoints the v6 latent does not support extraction of other agents' geometry: the probe is
**worse than predicting a constant** — K1 fails by **+12.42 m** @2000 and **+9.84 m** @9000
(positive = worse), every interval separated. The pre-registered `tokens` control **rules out the
readout as the cause**: handing the head the encoder's full **640 raw patch tokens × 768 dims**
(240× the `cells` surface) still loses to a constant by **+9.51 m**. That is the pre-registered
**D1** outcome — *the encoder does not carry agent geometry* — and **not** D2. On the PI's actual
question, *progress*: between 2 000 and 9 000 steps the headline does not cross into decodability,
though the head's mean predicted gap does move from 33.4 m toward the true 16.5 m, so the latent
is acquiring coarse forward structure. ⚠️ **Two things bound this hard: it is an EARLY-READ at
30 % of training — the pre-registration itself makes the 30 k checkpoint the primary and forbids
quoting an earlier one as the headline — and it ran on a 61-clip NON-PARITY subset with only
**13 lead-carrying eval episodes**. The DROP that D1 prescribes should therefore be executed only
after the 30 k confirmation, which costs ~12 minutes once that checkpoint exists.**

---

## 0. ⛔ READ THIS BEFORE ANY NUMBER — the stamps that bind every result

1. **`v6F-SW-30k@<step>` — the checkpoint is part of the arm (§1.4b).** Every number carries it.
2. **EARLY-READ.** 9 000 of 30 000 steps = a world model **30 % trained**. §1.4b makes the 30 k
   checkpoint the primary read; this is the baseline it will be compared against.
3. **fp16 trunk, BOTH points.** Both checkpoints were read through the trainer's own
   `weights_fp16.pt` snapshot layout, i.e. `fp16 -> fp32 (lossy)` — the programme's documented
   handover path (`train_v6_staged`: *"the snapshot is the artifact that makes a pod handover
   survivable"*). ⭐ Because **both** points go through the identical quantisation, the @2000 vs
   @9000 comparison is unaffected by it.
4. ⚠️ **NON-PARITY CORPUS — stated loudly, and it is the biggest caveat here.** The probe ran on a
   **61-clip subset** of the canonical 2 376-clip `physicalai-train-e438721ae894`, because the
   corpus is 80 GB at ~1.4 MB/s and the `obstacle.offline` label chunks cost a further ~72 MB per
   clip. `parity=False` is recorded in every cache meta. **This is admissible here and would NOT
   be for a trained arm:** the probe trains no world model and re-selects no training episodes —
   it reads a FROZEN trunk. The parity guard correctly refused until the directory was renamed to
   an unregistered key, which is the sanctioned path for a deliberate non-parity corpus.
5. ⚠️ **PROBE-TRAIN AND PROBE-EVAL ARE EPISODE-DISJOINT, BUT BOTH ARE WM-TRAIN CLIPS.** The world
   model saw these episodes. For a *negative* result this is **conservative** — it is the probe's
   best case — which is why the negative stands. A positive would have needed the val split.

---

## 1. ⛔ THE TRAJECTORY EXISTS — but only because the obvious checkpoint was the wrong one

The brief asked for several S-W checkpoints. Thor keeps **one rolling `ckpt.pt`** (`--save-every
250` overwrites), so there is no local history. Five locations were probed:

| source | artifact | step | usable as a second point? |
|---|---|---|---|
| `tanitad-thor-wifi:~/experiments/v6F-SW-30k/` | `ckpt.pt` 3.53 GB | **9000** | ✅ the late point |
| same dir | `ckpt_step*.pt` | — | ⛔ **none exist — one rolling file** |
| HF `Sayood/tanitad-v6` | `v6F-SW-30k/weights_fp16.pt` 0.67 GB | **2000** | ✅ **the early point** |
| HF `Sayood/tanitad-v6` | `v6E-SW-30k/ckpt.pt` | 1000 | ⛔ **NO — different architecture** |
| HF `Sayood/tanitad-v6` | `v6-SW-30k/ckpt.pt` | 6500 | ⛔ **NO — `enc_dim` 384 / `pred_dim` 768** |

⚠️ **`v6E` looks like the obvious earlier point and IS NOT ONE.** MEASURED by diffing the banked
configs: `pred_modern` `None` (E) vs `True` (F), `batch` 16 vs 8, and **`param_report.total`
336,589,641 (E) vs 336,542,025 (F)**. Two different models. Calling that pair a trajectory would
have manufactured exactly the confound §1.4b exists to prevent.

⭐ **AND THE ARTIFACT I NEARLY MIS-STAMPED.** `.last_ckpt_push` reads `8800` and the HF
`train_log.jsonl` ends at 8800, so `weights_fp16.pt` *looked* like an ~8800 checkpoint. Opening it
shows `_meta["step"] = **2000**` — and its `_meta["config"]["args"]["out"]` is
`/workspace/experiments/v6F-SW-30k`, i.e. the run's **pod phase before it migrated to Thor**. Its
`param_report.total` is **336,542,025**, identical to Thor's, and `pred_modern`/`batch`/`enc_dim`
all match: same run, same architecture, only paths differ. **Reading the step from the artifact
instead of from the filename is what turned a single point into a trajectory** — and is precisely
what §1.4b demands.

**How the second point was obtained without touching Thor's GPU:** a CPU-only script on Thor
(`mmap=True` load, per-tensor fp16 cast) wrote a 0.67 GB snapshot in the trainer's own layout;
pulled over ssh and **md5-verified** (`bd7762f768f3430eb1b3b6d37852b337`, both sides). This cost
0.67 GB instead of 3.53 GB. Thor's trainer (PID 25477) stayed healthy throughout.

---

## 2. What was run

| item | value | evidence |
|---|---|---|
| trunk | `v6F-SW-30k` @ **2000** and @ **9000**, FROZEN | `_meta["step"]`, both artifacts |
| memory surfaces | `cells` [16, 128] · `tokens` [640, 768] (the 16x40 patch grid) | `V6Stack.cells(z_op)` / `encode_window(..., return_tokens=True)` |
| head | `AgentSlotDecoder`, **3,211,541** params (cells) / **3,535,125** (tokens) | inside §6's 2–4 M band |
| corpus | 61-clip subset, **39 clips labelled**, 3 160 frames @ stride 2 | `parity=False` recorded |
| split | EVAL 20 clips / 1 669 windows · TRAIN 19 clips | episode-DISJOINT, provider order |
| labels | `build_obstacle_join.py` → `JoinFileReader` | 7 287 frames, 246 837 boxes, verified through the real reader |
| estimator | `paired_episode_cluster_bootstrap`, n_boot 2000 | ⛔ `overlapping_holdout_se` never imported |
| GPU | RTX 4060, peak **1.457 GB** (`torch.cuda.max_memory_allocated`) | — |

### 2.1 ⭐ `n_slot_queries` was a PLACEHOLDER and is now FITTED (prereg §2 required this)

§2 flags `n_slot_queries = 16` as *"A DECLARED PLACEHOLDER, NOT A FITTED VALUE"* and instructs:
measure the join's per-frame agent-count distribution and record it. **MEASURED over the 7 287
labelled frames:**

| population | mean | median | p90 | p95 | **p99** | max |
|---|---|---|---|---|---|---|
| all agents in the join | 33.87 | 26 | 87 | 106 | **130** | 161 |
| **in-grid** (0 < cx ≤ 60 m, \|cy\| ≤ 16 m) | 5.61 | 3 | 16 | 21 | **31** | 37 |
| in-grid AND visible (`occ` 0) | 5.17 | 3 | 15 | 20 | 29 | 35 |

⇒ **`n_slot_queries = 32`** (the in-grid p99, 31, rounded up). At the prereg's 16, **9.04 %** of
frames would silently over-flow and drop targets; at 32 it is **0.75 %** (`raw/agent_count_
distribution.json`). The in-grid population is the right denominator because `SlotDecodeRanges`
decodes into exactly those extents — an agent 200 m behind the ego is not representable, so
scoring it would measure the coordinate transform rather than the latent.

### 2.2 The frozen-trunk proof — non-vacuous, and it can fail

`assert_isolation` at the **production geometry**, head built, both arms
(`raw/sp0_isolation.json`):

| edge | violations | **n_probed** |
|---|---|---|
| `planner_to_encoder` | 0 | 151 |
| `tactical_to_below` | 0 | 342 |
| `strategic_to_below` | 0 | 427 |
| **`perception_to_trunk`** | **0** | **538** |

⭐ **The mis-wired control FIRES:** with `isolate_interp_from_encoder=False`,
`perception_to_trunk` reports **151 live forbidden edges** (`encoder.pos`, `encoder.patch.weight`,
…) and raises `IsolationViolation`. A guard that cannot fail is not a guard; this one can.

⚠️ **And the probe is isolated more strongly than that edge.** It trains on a **banked, detached**
latent tensor with no autograd graph, so no gradient path to the trunk exists even in principle.

---

## 3. ⛔ THE RESULT — measured against the pre-registered criteria

**Primary: `lead_gap_abs_err_m`.** Paired episode-cluster bootstrap, n_boot 2000, on the SAME
windows. **Positive Δ = the arm is WORSE than the control.**

| | **@2000** | **@9000** |
|---|---|---|
| paired windows (of 454 GT-lead) | 450 | 434 |
| eval episodes carrying a GT lead | 13 | 13 |
| arm `lead_gap_abs_err_m` (mean) | 19.70 m | 17.12 m |
| **C-CONST** (train-median 20.92 m) | **7.28 m** | **7.28 m** |
| **K1** Δ arm − C-CONST | **+12.42 [5.47, 19.30]** separated ⛔ | **+9.84 [5.69, 13.75]** separated ⛔ |
| **K2** Δ arm − C-SHUF | **−0.05 [−2.19, 1.41]** NOT separated ⛔ | **−0.06 [−2.09, 1.75]** NOT separated ⛔ |
| **K3** τ\*-gated `lead_presence_recall` | **0.289** ⛔ (< 0.50) | **0.324** ⛔ (< 0.50) |
| **K4** median abs err < 0.9769 m | 16.06 m ⛔ | 17.22 m ⛔ |
| **VERDICT** | **DROP / RE-SCOPE** | **DROP / RE-SCOPE** |

### 3.1 ⭐ The oracle diagnostic — which localises the failure

**DIAGNOSTIC ONLY; it never enters K1–K4.** The best in-corridor slot vs the GT lead:

| | @2000 | @9000 |
|---|---|---|
| oracle-slot abs err (mean / **median**) | 14.37 / **13.15** m | 11.83 / **7.62** m |
| head's mean predicted gap | 33.44 m | 24.52 m |
| true mean GT gap | 16.36 m | 16.47 m |

⇒ **The failure is NOT presence-ranking.** Even the head's *best* slot is 7.6–13.2 m from the true
lead. Had the oracle been small while the headline was large, the finding would have been "the
geometry is there and the selection cannot find it" — an entirely different work item. It is not.

### 3.2 ⭐ Where the progress actually is

Between @2000 and @9000, on identical windows: **oracle median 13.15 → 7.62 m (−42 %)** and the
mean predicted gap moves **33.44 → 24.52 m** toward the true 16.5 m. So the latent *is* acquiring
coarse forward structure. But **K1 stays failed, K2 stays unseparated, and K3 stays below 0.50** —
the quantity the PI asked about (*the WM learning the environment*) has **not** crossed into
decodability by 30 % of training.

---

## 4. ⛔ THE CONTROLS — and why the null is believable

| control | result | what it establishes |
|---|---|---|
| **C-CONST** | arm is **worse** by 9.8–12.4 m, separated | the head has measured nothing about *this* window |
| **C-SHUF** | **Δ ≈ 0, CI spans 0, at BOTH checkpoints** | ⛔ the anti-echo control: the head scores the same on a DIFFERENT window's latent — it is reading a corpus prior |
| **C-TOK** (`tokens`) | **K1 fails too: +9.51 [3.31, 16.43]** — §5 | ⇒ **D1, not D2**: 640×768 raw patch tokens still lose to a constant, so the readout grid is not the bottleneck |
| **C-V5F** | ⚠️ **NOT RUN** | needs the v5f trunk (Thor `~/models/v5f/ckpt.pt`, 3.5 GB) and a second full cache pass; the bandwidth went to the second v6F point, which answers the PI's actual question. Named, not silently dropped. |

### 4.1 ⭐ A PIPELINE NULL CONTROL WAS RUN FIRST, and it caught two defects in my own instrument

Before any real latent, the identical scoring path was run on a **random-latent cache with the
real targets** (`raw/nullcontrol_random_latents.json`). A probe on noise **must** fail K1.

* ✅ It did: Δ vs C-CONST **+20.54**, and C-SHUF Δ **0.037 [−0.19, 0.32] NOT separated**. So there
  is **no target→prediction leak** in the scoring code, and the real nulls above are not an
  artefact of my harness.
* ⚠️ **Defect 1 — K3 could not fail.** The first cut counted a window as "emitted" whenever ANY
  slot was in-corridor, which is true **99.8 %** of the time by geometry alone. K3 passed at 0.998
  while measuring nothing — the C13 family. Recall is now **τ\*-gated** (τ\* set on the ENCODED
  arm and frozen), and K3 correctly **fails at 0.289 / 0.324**.
* ⚠️ **Defect 2 — TTC was numerically meaningless.** With ε = 1e-3 a lead closing at 1 mm/s gives a
  30 000 s "TTC"; the column read **199.96 s mean error, CI [26.9, 589.4]**. A physical closing
  floor of **0.5 m/s** now gates it, with its own n.
* ⚠️ **A power warning, MEASURED:** on an early null run C-SHUF separated **spuriously** at 20 eval
  episodes. K2 must never be read alone — and the 13 lead-carrying eval episodes here are few.

---

## 5. ⛔ THE `tokens` CONTROL — **D1 FIRES**, not D2

Run at **@9000**, both arms on **identical windows** from one cache (stride 4, 835 eval windows,
225 paired) so `cells` and `tokens` are properly comparable (`raw/results_tokens_s9000.json`):

| | `cells` [16, 128] | `tokens` [640, 768] | **C-CONST** |
|---|---|---|---|
| `lead_gap_abs_err_m` | 22.09 [18.77, 24.99] | 16.39 [10.59, 23.07] | **6.88 [5.90, 7.67]** |
| median | 21.95 m | 14.99 m | 7.13 m |
| **K1** Δ vs C-CONST | **+15.21 [11.52, 18.47]** ⛔ | **+9.51 [3.31, 16.43]** ⛔ | — |
| **K2** Δ vs C-SHUF | −2.21 [−4.32, −0.43] sep. | +0.23 [−0.96, 1.57] not sep. | — |
| **K3** τ\*-gated recall (τ\* 0.628) | 0.106 ⛔ | 0.088 ⛔ | — |

⇒ **D1 — *"K1 fails on BOTH `cells` and `tokens`. The encoder does not carry agent geometry."***
This is **not** D2: the readout grid is not the bottleneck, because giving the head the encoder's
full 640 raw patch tokens at 768 dims — 240× the `cells` surface — **still loses to a constant by
9.5 m**. The pre-registered action for D1 is: the head is **DROPPED as a readout**, F-18's diagram
cell stays ⬜ **now with a MEASURED reason instead of an absence**, and the work item becomes an
**encoder-objective** question (a supervised detection branch, or an O-measure that makes agents
predictable) — which needs its own pre-registration and **must not be smuggled in as "tuning the
head"**.

⚠️ **BUT THE DROP IS PROVISIONAL, BY THE PRE-REGISTRATION'S OWN RULE.** §1.4b designates the
**30 k** checkpoint as the primary read and states an earlier one *"may not be quoted as the
headline"*. D1 fires at 2 000 and at 9 000; **it should be executed only after the 30 k
confirmation**, which costs ~12 min once that checkpoint exists.

### 5.1 ⚠️ Two honesty notes on my own instrument

* **K2 is NOT stable across configurations, and I am reporting that rather than the flattering
  half.** `cells` K2 came out **−0.05 [−2.19, 1.41] NOT separated** on the stride-2 cache (450
  paired windows) and **−2.21 [−4.32, −0.43] SEPARATED** on the stride-4 cache (225 windows, a
  different fit). With only **13 lead-carrying eval episodes** the paired bootstrap has little
  power, and my random-latent null separated spuriously once at 20 episodes. ⇒ **K2 must not be
  read alone here.** Nothing turns on it: **K1 is the gate and it fails by +9.5 to +15.2 m in
  every configuration run**, which is 1.4–2.2× the entire C-CONST error.
* **The oracle diagnostic is a LOWER BOUND, not a measure of latent content.** A head that
  scatters slots widely earns a small oracle for free — 32 slots spread over the 0–60 m corridor
  would give an expected min-gap of ~0.94 m by geometry alone. The observed oracle medians
  (`cells` 3.07 m here / 7.62 m on the stride-2 fit; `tokens` 11.24 m) are **worse than uniform
  scatter would produce**, so they do not rescue the arms — but they are not clean evidence either,
  and I am not treating them as such.

---

## 6. The FOUR METRIC FAMILIES

⛔ ADE is not reported: this is a perception readout, not a trajectory eval. Per family, with the
reason where it does not apply:

| family | served? | how / why not |
|---|---|---|
| **LONGITUDINAL** | ✅ **directly — this is the point of the head** | the lead slot's `cx` **is** the headway; reported above with its estimator and CI. `lead_ttc_abs_err_s` @9000 `cells` = **6.51 s [4.94, 8.49]**, n = 137 windows / 13 episodes, under a 0.5 m/s closing floor — i.e. the vision-only lead's time-to-collision is wrong by ~6.5 s, which is not usable. ⚠️ `taniteval.lead_metrics.distance_keeping` is **NOT** attached: it consumes a predicted ego **path** `(W,K,2)`, and this is a single-frame readout that produces no path. That is the T1 integration §4.6 defers — stated, not silently skipped. |
| **LATERAL** | ❌ not served | the head emits **agent** geometry, not ego path; heading / curvature / yaw-rate / cross-track are ego quantities this readout never produces. |
| **TACTICAL** | ⚠️ **enabling condition only — NOT COMPUTED** | slot-referent agreement needs a **trained goal head** with the categorical `agent_slot` arg on. S-T has not run, so `GAP_TARGET`/`YIELD_AT`/`WAIT_FOR_ONCOMING`/`EVADE_IN_CORRIDOR` still index an empty set. A follow-on, per prereg §3. |
| **STRATEGIC** | ❌ not served | `obstacle.offline` has **no map, lane graph, junction, traffic-light or route feature** — 10 classes, all dynamic agents. Nothing strategic is derivable from this label at all. |

---

## 7. ⛔ ESCALATIONS — decisions and integration this run needs

1. ⛔⭐ **THE TRAINER MUST BANK STEP-STAMPED CHECKPOINTS. This is the highest-value fix here.**
   `--save-every 250` overwrites ONE rolling `ckpt.pt`, so the entire training history of the
   programme's flagship run is unrecoverable at any price. Tonight's trajectory exists only
   because an old fp16 snapshot happened to survive on HF from the run's pod phase. **A periodic
   `ckpt_step{N}.pt` (or fp16 snapshot, 0.67 GB) every ~2 000 steps would have made the PI's
   "is there progress" question answerable directly, at ~7 GB for the whole run.**
2. ⛔ **A PRE-REGISTERED DROP IS PENDING AND NEEDS A DECISION — this is the integration item.**
   **D1 has fired at both measured checkpoints**, and its pre-registered action is to DROP the
   agent-slot head as a readout and convert F-18 into an **encoder-objective** work item. I have
   **not** executed that drop, because §1.4b makes the 30 k checkpoint the primary read. **PI /
   integrator decision: run the 30 k read (~12 min once the checkpoint exists) and then either
   execute D1 or record why not.** Leaving it unexecuted and unrecorded is the failure mode.
3. ⚠️ **The 30 k read is the primary and is NOT yet available.** This run is its baseline. Re-run
   `sp1`+`sp2` at 30 k with the SAME 61-clip cache — `code/thor_snap.py` makes the checkpoint a
   0.67 GB pull instead of 3.53 GB.
4. ⚠️ **C-V5F was not run** (§4). It is the control that would say whether O2/O3/O4 bought any
   agent content at all relative to v5f. It needs one 3.5 GB pull plus one cache pass.
5. ⚠️ **The result is on a 61-clip NON-PARITY subset with only 13 lead-carrying eval episodes.**
   Before this negative is treated as settled it should be re-run on the val40 split. That needs
   the val-corpus `obstacle.offline` join, which nobody has built (~3 GB of label chunks).
6. ⚠️ **`n_slot_queries` 16 → 32.** The prereg's placeholder is now measured. Anyone re-running
   F-18 must use 32 or accept an 8 % silent target-drop rate.

---

## 8. Test state — and my delta is ZERO

⛔ **This run changed NO repository code.** Every script it added lives under
`…/2026-08-16-slot-probe-run/code/`; nothing under `stack/` was edited (`git status --short stack/`
shows only another stream's `ph0_sam3.py` / `test_ph0_sam3.py`). So the admissible statement about
this stream is that its delta on the suite is **zero**, not "the suite is green".

MEASURED, `cd stack && PYTHONUTF8=1 OMP_NUM_THREADS=6 pytest -q tests/test_v6_agent_slots.py
tests/test_p8.py tests/test_v6_ladder_edges.py tests/test_v6_stage_init_introduction.py`:

> **1 failed, 107 passed** in 24.71 s

⚠️ The one failure is `test_v6_ladder_edges.py::test_after_init_from_exactly_the_intended_groups_
train[S-J]`. **It is not attributable to me** and I am naming it rather than letting it pass as
background noise: `stack/scripts/train_v6_staged.py` (mtime **21:38**) and `stack/tanitad/models/
v6.py` (**20:17**) were both modified during this session by the streams that own them, after
F-18's own baseline. The 41 `test_v6_agent_slots.py` tests — the module this probe exercises — all
pass. **Integrator: that failure belongs to whoever is holding those two files.**

⚠️ I did NOT run the full suite (`pytest -q`, ~7 min) — I ran the four files that cover every
module this probe touches. Stating the scope rather than implying full coverage.

---

## 9. Deliverable manifest

| artifact | where | only one place? |
|---|---|---|
| this report | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-slot-probe-run/SLOT_PROBE_RUN.md` | staged |
| isolation proof (4 edges + mis-wired control) | `repo:…/2026-08-16-slot-probe-run/raw/sp0_isolation.json` | staged |
| cells result @2000 | `repo:…/raw/results_cells_s2000.json` | staged |
| cells result @9000 | `repo:…/raw/results_cells_s9000.json` | staged |
| tokens+cells result @9000 (matched windows) | `repo:…/raw/results_tokens_s9000.json` | staged |
| cache meta, tokens pass | `repo:…/raw/cache_meta_tok9000.json` | staged |
| pipeline null control | `repo:…/raw/nullcontrol_random_latents.json` | staged |
| cache metas (both steps) | `repo:…/raw/cache_meta_s{2000,9000}.json` | staged |
| join meta (39 clips) | `repo:…/raw/join_train40_meta.json` | staged |
| agent-count fit | `repo:…/raw/agent_count_distribution.json` | staged |
| code (probe + pullers) | `repo:…/code/{sp0_isolation,sp1_cache_latents,sp2_probe,sp_common,spX_fake_cache,s1_pull_episodes,s2b_resume_pull}.py` | staged |
| Thor fp16 snapshot script | `repo:…/code/thor_snap.py` + `tanitad-thor-wifi:~/thor_snap.py` | staged |
| fp16 snapshot @9000 (0.67 GB) | `tanitad-thor-wifi:~/v6F_snap_fp16.pt` **and** `<scratch>/slotprobe/w9000/` | ⚠️ **NOT in the repo — too large.** Reproducible from Thor's `ckpt.pt` via `code/thor_snap.py`; md5 `bd7762f768f3430eb1b3b6d37852b337` |
| latent caches (2 × ~30 MB, 1 × ~1.6 GB) | `<scratch>/slotprobe/cache_s{2000,9000,tok9000}/` | ⚠️ **scratch only** — regenerable in ~7 min each from the snapshot + join |
| obstacle join, 39 clips | `<scratch>/slotprobe/joinout/train40_agents.jsonl` | ⚠️ **scratch only** — 3.9 MB, regenerable; meta IS staged |
