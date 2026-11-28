# refcv5 — VLA EXTENSION PLAN: a language backbone for the tactical and strategic layers

**status: v1 — the design is chosen, the interface is bound to `REFCV5_DESIGN_PLAN.md` §9, the work packages are ordered. Nothing here has been trained; every latency figure is ESTIMATED until WP-0.**
**author:** TanitAD Research Lab (Architecture & Inference) · **date:** 2026-09-05 · **branch:** `agent/arch-inf-20260803` · **GPU spent by this document: 0**
**Companion research file (the survey and the evidence):** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-vla-extension-frontier/RESULT.md` (§0 architecture · §1 survey, 58 banked primaries · §2 design options · §3 grounding/consistency · §4 `H-VLA-*` · §5 refusals · §6 manifest).
**Binding sibling:** `Project Steering/REFCV5_DESIGN_PLAN.md` **§9** is *the contract*. ⛔ Where this plan and §9 disagree about a **port**, §9 wins; where they disagree about the **module**, this plan wins (§9.0's own rule). Every port name below was read from §9 as it stands in HEAD on 2026-09-05.
**Owner for integration:** Master Mind. **Streams touched:** Architecture & Inference (module), DataFlyWheel (WP-2 label synthesis), Deployment & Optimization (WP-0/WP-9 Thor), Benchmarks & Eval (WP-6 criteria registry).

## PI's ask (verbatim, condensed)

> "extend the architecture by a language backbone in the tactical and strategic layer … a vision-language part sharing the same embedding space as refc … it can process nav commands, question queries, it gets also the images and ego data and it creates system initiated chain of thought as text as explanation of the behavior (this can be trained, distilled from the CoT we extracted from Alpamayo), the behavior must be grounded in the scene understanding and compliant and consistent to the trajectory/trajectory hypotheses (so it is not hallucinating internal thoughts) … the VLM/VLA part must be able to process and emit strategic and tactical goals, it complements our current strategic and tactical layer and should have a tact between 300 to 500 ms … take a small architecture from scratch or reuse, modify and fine-tune a pretrained small model like SmolVLM, or the 0.8 Qwen 3.6 or something similar … must not exceed 1 billion parameters, better smaller."

**Evidence classes:** `MEASURED (ours + artifact)` · `PUBLISHED-PRIMARY (banked PDF + table)` · `PUBLISHED-SECONDARY (card/vendor page — inadmissible for the registry)` · `INHERITED (another doc, not re-verified)` · `ESTIMATED (derivation shown)` · `HYPOTHESIS`. Eval numbers carry their **T-tier**.

---

## §1 The answer in one page

**Name:** **TanitLang** — a language *proposer and monitor* attached to REF-C, not a second planner.

**Chosen design (option (c) of the research file §2.c).** A **frozen small pretrained LM + LoRA**, with
**REF-C's trunk as the ONLY vision encoder**. The trunk's **160 perspective-view tokens** (`fmap [B, feat_dim, 8, 20]`,
§9.2) are projected once into the LM's embedding space; cascade state (`ctx`, `z_tac`, `g_str`, nav, `v0`) enters as a
handful of prefix tokens; the LM emits, every tact, a **2–6-token executable manoeuvre sequence plus non-decoded goal
vectors**, and, on a **slower clock**, the prose explanation.

| | SmolLM2-360M variant | Qwen3.5-0.8B variant |
|---|---|---|
| resident parameters | **≈ 368 M** | **≈ 806 M** |
| trainable | **≈ 6 M** (projector + LoRA r=16 + heads) | ≈ 8 M |
| ≤ 1 B ceiling | ✅ far below | ✅ just below |
| default | ⭐ **yes** — start here | the upgrade if `H-VLA-1`/`H-VLA-2` show 360 M is under-capacity |

**Why not a whole small VLM (the runner-up, option (a): SmolVLM-500M / InternVL3-1B + a projector).** It is a real
option and it is kept as the documented fallback if `H-VLA-1` fails. It loses on three measured grounds and one
structural one:
1. It brings a **second vision encoder**, which §9.1 makes a **contract violation**: *"The language module must consume
   REF-C's trunk tokens. It must not run a second vision encoder"* — because the encoder is **~90 % of a REF-C tick**
   (`refc.py:2238`, INHERITED) and encoding twice spends the tact budget twice.
2. `2504.05299` **Finding 1** (PUBLISHED-PRIMARY, SmolVLM §2.1): a large encoder beside a small LM is a net loss —
   428 M SigLIP + 135 M LM *degrades*; at 360 M LM it buys **+11.6 % for +66 % parameters**.
3. `2607.08029` (PUBLISHED-PRIMARY): **SigLIP encoders take a disproportionate INT8 latency hit on Jetson**.
4. ⭐ Structurally: with two encoders the explanation and the plan look at **different tensors**, so a divergence
   between them can no longer be attributed to reasoning rather than perception — **the USP's measurement dies**.

**Why not from scratch (option (b)).** Our entire CoT corpus is **4,729 one-sentence rationales ≈ 47 k words**
(MEASURED, research file §0.2). That is orders of magnitude below a 360 M LM's data requirement (`2502.02737`), and it
cannot deliver question answering or unseen-object naming. ⇒ **REFUSED as the module, KEPT as the mandatory floor arm**
(`H-VLA-2`): a pretrained LM that cannot beat a from-scratch template decoder on faithfulness has bought us nothing
but parameters.

**The two things that make this a USP rather than a caption generator**, both from the research file §3:
- **It is a PROPOSER, not a DESCRIBER** (§9.7's distinction). At explanation time the module does **not** see
  `sel_idx` or `g_tac`. Consistency with the plan is therefore a **measurement of two systems agreeing from the same
  scene**, not a description of an answer already in hand.
- **Referents POINT.** *"the lead vehicle"* is emitted as `REF(k)`, a softmax over agent tokens ∪ {NULL}. A
  hallucinated object becomes structurally impossible, not merely penalised.

---

## §2 Interfaces — bound to `REFCV5_DESIGN_PLAN.md` §9

### 2.1 Mount point

⭐ **The module mounts on the EXISTING `hierarchy_hook`** (§9.1): `hook(pooled_seq [B, W, F], ctx [B, d_ctx]) -> dict`,
whose returned entries fill REF-C's ports **only where the caller passed `None`**, so an explicitly supplied port
always wins and an experiment can never be silently overridden. TanitLang is therefore a **second supplier on an
existing, gated, experiment-safe seam** — not new plumbing.

⭐⭐ **`maneuver_logits [B, 5]` is LIVE and has never been filled.** It reweights the anchor prior (H19,
`refc.py:1645-1647`) and is *"an outside tactical brain speaking the 5-way surface"*. **It is the only write port that
changes behaviour with zero new wiring** — and therefore the first mode-I edge to open (WP-8a), long before E19.

### 2.2 READ ports (⚠️ widths are read from the built model, never hardcoded — §9.8)

| port | what the module does with it |
|---|---|
| `pooled_seq [B, 8, feat_dim]` | the hook's first argument — the temporal visual context |
| `fmap → 160 PV tokens` (`[B, feat_dim, 8, 20]`) | ⭐ **the vision input**, projected 1:1 into the LM embedding space. 256 × 640 **cylindrical**, `f_ref` 305.577, HFOV **120.0°** — ⛔ the pinhole formula gives 92.6° and is wrong here; any token↔azimuth mapping goes through `calib.CanonicalFrame(projection="cylindrical")` / `bev_raster.readout_column_index` |
| `ctx [B, d_ctx]` | strategic context — ⚠️ **already contains nav** (`ctx = ctx + nav_s`), so a module reading `ctx` may **not** claim a vision-only reading; it carries the nav controls instead |
| `z_tac [B, d_tac]` | the tactical conditioning vector |
| `g_str` (3, → 23 under E19) | ⚠️ **NAV_BLIND today** (Δ_shuffle = 0.0000) — every strategic claim is reported **conditional on `H-NAVC-2`** |
| `m` / `v0` / nav one-hot + `nav_known` | ⛔ under the **same** `ego_keep` withholding draw as REF-C (`ego_dropout 0.5` + X15 presence bit). A second, unsynchronised dropout is the defect the shared port exists to prevent |
| question / instruction tokens | admissible **as an input to the module only** (§9.4) |

### 2.3 WRITE ports and the two modes

| edge | port (§9.3) | mode M (monitor) | mode I (influence) |
|---|---|---|---|
| tactical prior | `maneuver_logits [B, 5]` | **gated to 0** | ⭐ **first edge to open** — zero new wiring |
| tactical goal | `g_tac_proposal [B, 12]` = 3 τ × `(x, y, heading, speed)`, τ = **2.0 / 4.0 / 6.0 s** | gated | new zero-init residual edge into `tac_goal_head` |
| strategic goal | `p_man [B, 4]` / `t_bin [B, 6]` | gated | needs **E19** first |
| selection | `selector_prior [B, N]` additive log-prior on the fan | gated | one edge |
| explanation | **no model node** | scored, never consumed | ⛔ **stays unconsumed in mode I too** |

⛔ **Mode M ships first, with every gate at zero.** Then `F7` (causal influence, research file §3.4) **must read
exactly 0.0000** — a control that reads a known value, because there is no path. Mode I opens **one edge at a time**,
each earning a T1 four-family delta with the echo gate. Every write port ships with its **eval-time switch**, so
`PREREG_REFCV4B_HIERARCHY_EVAL.md`'s ablation table extends by one row per VLA edge (§9.3); an edge without a switch
is inadmissible.

### 2.4 The vocabularies (pinned constants, `stack/tanitad/models/vocab_v7.py` — §9.4)

The module emits **only** these tokens; anything else is an unlabelled string that no consistency metric can score.

- **Strategic:** 8 `STRATEGIC_GOAL_TOKENS_V7` (`FOLLOW_ROUTE`, `TURN_L/R_FOLLOW_ROUTE`, `STOP_AT_FOLLOW_ROUTE`,
  `EXIT_L/R_FOLLOW_ROUTE`, `LANE_CHANGE_L/R_FOLLOW_ROUTE`) + 7 `STRATEGIC_ACTION_TOKENS_V7` + the **mandatory** arg
  slots `("within_m", "by_time_s")`. ⚠️ The arg slots are **not optional**: the PI's counter-example `01bee851`
  (+76° / −43° / +69°) is distinguishable only by distance and time. Band `STRATEGIC_S = (8, 30) s`; E19 `t_bin`
  = (0-2, 2-5, 5-9, 9-14, 14-21, 21-30).
- **Tactical:** the **8-way factored** `TACTICAL_LAT_ACTIONS_V7` × `TACTICAL_LON_ACTIONS_V7`, plus the 24
  `TACTICAL_GOAL_TOKENS_V7`, with `GOAL_ADMISSIBLE_LAT` / `GOAL_ADMISSIBLE_LON`. ⛔ **An emitted (goal, action) pair
  violating the admissibility maps is a contract violation the module must not produce** — it is `contradiction_rate`,
  and a compliant module reads **0**.
  ⚠️ `OVERTAKE_VEHICLE` vs `EVADE_IN_CORRIDOR` is decided by the **object** (moving vs static/VRU), not the verb;
  conflating them files **419 parked-car evasions with 13 real overtakes**.
- **Nav (input):** 3 tokens, provenance **`ego-future`** on PhysicalAI — an oracle and a **per-clip constant**
  (train: follow 2,897 / left 811 / right 864). ⇒ every nav-conditioned number carries the **shuffle and zero**
  controls.
- **Geometric goals:** `GOAL_DIMS = 4`, `(x, y, heading, speed)` per τ ∈ {2, 4, 6} s, ego frame **x forward, y left,
  z up**.

### 2.5 ⭐ The information-state declaration — three tiers, declared per arm

§9.7 makes this the difference between a result and an artefact. We extend its two regimes to three, because our
module reads cascade state that is *upstream of the pick*:

| tier | what the module sees at explanation time | is `tac_consistency` a real measurement? |
|---|---|---|
| **P0 — scene-only** | `fmap` / `pooled_seq` + question. No `ctx` (nav-carrying), no `z_tac`, no `g_str` | **yes**, and it is the strictest form. Used as a **control arm** |
| **P1 — scene + upstream cascade state** ⭐ | + `ctx`, `z_tac`, `g_str`, nav, `v0` (under the shared withholding draw). ⛔ **NOT** `sel_idx`, **NOT** `g_tac`, **NOT** the fan | **yes** — this is the **headline arm**; the pick is downstream of everything it reads |
| **D — describer** | + `sel_idx` / `g_tac` / the fan | ⛔ **no** — consistency ≈ 1.0 **by construction**. Admissible only for the *post-hoc narration* product, and then the headline is **scene faithfulness**, never plan agreement |

⛔ **Any table reporting `tac_consistency` states its tier.** A P1 number and a D number are not comparable, and a D
number is not a result.

---

## §3 Budgets and the measurement plan

### 3.1 Parameters (ESTIMATED, arithmetic shown; `feat_dim` = 704 at `size base`, 256 at the `tiny` rig rung)

| part | SmolLM2-360M | notes |
|---|---|---|
| frozen LM | 362 M | Apache-2.0; `2502.02737` |
| `proj_in`: `Linear(feat_dim → d_lm)` + 1 hidden | ≈ 1.4 M | 704 → 960 → 960 |
| `state_proj`: MLP(`ctx` 256 ⊕ `z_tac` 512 ⊕ `g_str` 3 ⊕ `v0` ⊕ nav 4) → 4 prefix tokens | ≈ 0.8 M | |
| LoRA r = 16 on q/k/v/o + FFN | ≈ 3.5 M | |
| write heads: `maneuver_logits` 5 · `g_tac_proposal` 12 · E19 (4+6) · XCoT vocabulary ≈ 64 · referent pointer | ≈ 0.1 M | all **zero-init** |
| **total resident / trainable** | **≈ 368 M / ≈ 6 M** | Qwen3.5-0.8B variant: **≈ 806 M / ≈ 8 M** |

### 3.2 Thor latency (ESTIMATED — ⛔ every figure is replaced by WP-0)

**The two anchors, both embedded, never a datacentre GPU:**
- **Thor 273 GB/s vs Orin 204.8 GB/s** (NVIDIA module pages, fetched 2026-09-05, PUBLISHED-SECONDARY).
  ⭐ Decode is **weight-bandwidth-bound**, so Thor's **7.5× compute** advantage does **not** reach the CoT; the
  transferable ratio is **1.33×**. Prefill and the trunk *do* get the compute ratio.
- **DriveVLM Table 6 on OrinX** (PUBLISHED-PRIMARY, `2402.12289`): Qwen1.8B **79.6 tok/s**, MobileLLaMA-1.4B
  **117.4 tok/s**. The table's own numbers imply an achieved efficiency **η ≈ 0.37** of the bandwidth roof (research
  file §2.0 shows the derivation, including why the FP16 reading is arithmetically impossible).

| tact stage | ESTIMATE | replaced by |
|---|---|---|
| REF-C trunk + cascade + decoder on Thor | ⛔ **UNMEASURED** — allowance 60 ms; the encoder is ~90 % of a REF-C tick | **WP-0 item 1** |
| projector (160 tokens) | < 1 ms | WP-0 |
| prefill ~230 tokens | ~15 ms | WP-0 |
| decode, 0.36 B FP8 (3.6 ms/tok) | 2–6 tokens = **7–22 ms** | WP-0 / `H-VLA-3` |
| **per-tact total** | **≈ 85–100 ms of a 300–500 ms tact** | — |
| prose, slow clock (40–80 tokens) | 145–290 ms, **off the control path**, ÷ 3.1–4.1× with FastDriveCoT field parallelism | `H-VLA-8` |

⛔ **Contract term (§9.6b): the write ports must be producible WITHOUT generating prose.** A `[B, 5]` log-prob vector
is one forward pass, not forty decode steps. **A design in which the planner waits on text is refused at the
interface, not at the eval.**

### 3.3 The measurement plan (WP-0) — what turns every ESTIMATE above into a number

1. REF-C `size base` **inference** latency on `tanitad-thor-wifi`, **batch 1**, warm, 200 windows: p50/p90 wall clock,
   split trunk / cascade / decoder / selector.
2. TanitLang prefill and per-token decode at bf16 / FP8 / NVFP4, batch 1, at the real prefix length.
3. Memory: ⛔ **only `torch.cuda.max_memory_allocated()`** — `mem_get_info`, `free`, `tegrastats` and `VmRSS` all lie
   on Thor, in both directions.
4. ⚠️ **Do not quote "Thor saturates at batch 8"** here: that is a **v6F training** throughput on a **different
   trunk**. Inference is batch 1 and latency-bound.
5. ⚠️ Quantisation is a **measurement, not an assumption**: `2607.08029` MEASURED that INT4 can *slow* generation on
   Jetson through dequantisation overhead, and that sensitivity is set by the structural paradigm rather than scale.

---

## §4 Training pipeline

### 4.1 Data — what exists, and what has to be built

| source | what it gives | n | state |
|---|---|---|---|
| Alpamayo release (`records.parquet`) | one-sentence `chain_of_causation` (median 57 chars / 10 words); `critical_components_analysis` (3,140, with **853 explicit `type: none`**); `ego_vehicle_motion_analysis` (2,943); `meta_action` triplet; `grounding_via_vqa` 2-D boxes (4,728) | 4,729 clips | **MEASURED**, held locally + on HF |
| v7.2 labels | `manoeuvre_sequence` → the strategic/tactical vocabularies of §2.4 | 4,572 train / 141 eval | **MEASURED** |
| `obstacle.offline` | 3-D agent tracks, 10 dynamic classes, 87,481 cuboids | **97.44 %** of clips | in the pod-side join |
| ⛔ **window-level explanation labels** | one CoT per *clip* at one instant is not a window-level label | **to build** | **WP-2** |

⚠️ **Anchor mismatch: Alpamayo's `t0_us` = 5,100,000 (5.1 s) vs our s2 anchor 8.0 s — every join is 2.9 s off unless
re-anchored.** WP-2's first step.

⛔⛔ **The teacher supplies FORM, not CONTENT.** Measured on our own corpus and our own teacher:
**42.5 %** reasoning fidelity, **48.3 %** reasoning-action consistency, **37.9 %** of stop-claimed cases continue
(`2605.17268`); **33.3 %** of CoTs unreliable (`2608.29583`); our own spot check 3 correct / 2 wrong with a
hallucinated cyclist. And `2608.01755` measures that a teacher shown the logged future **rationalises** it —
Alpamayo's CoC auto-labeller is given *"the ego vehicle's trajectory, dynamic states, and meta actions"*
(`2511.00088` §5). ⇒ **`H-VLA-6` decides whether our draw is anchored**, before any content is distilled.

### 4.2 Losses

| loss | target | clock | guard |
|---|---|---|---|
| `L_tac` | v7.2 factored (LAT, LON) **label** | every tact | never the model's own head — that is self-distillation and reads 1.0 by construction |
| `L_goal` | `g_tac` label points at τ = 2/4/6 s + E19 `p_man`/`t_bin` | every tact | the 30 s label is a **training target, never an input** |
| `L_ref` | referent pointer vs `obstacle.offline` critical agent \| **NULL** | every tact | 853 explicit `type: none` clips make NULL learnable |
| `L_admiss` | penalty on (goal, action) pairs violating `GOAL_ADMISSIBLE_*` | every tact | target `contradiction_rate` = **0** |
| `L_agree` | KL to `lat/lon_logits_tac` — **ramped from 0 and CAPPED** | every tact | ⛔ **100 % agreement is a REFUSAL** — the module has become the echo. Ships with a **deliberate-regression arm at 10× weight that must FAIL the F5 scene-sensitivity gate** |
| `L_cot` | the Alpamayo sentence (form) + our template (content) | **slow clock only** | prose never conditions the plan |

### 4.3 Stages

| stage | what trains | frozen | rig | GPU |
|---|---|---|---|---|
| **S1 projector-only** | `proj_in`, `state_proj` | LM, LoRA off, REF-C | dev box (RTX 4060) | ~4 h |
| **S2 + LoRA** | + LoRA r=16 | LM base, REF-C | dev box | ~12 h |
| **S3 + heads, full label set** | + write heads, `L_ref`, `L_admiss` | LM base, REF-C | 1 × A40 | ~24 h |
| **S4 prose head** | LM head on the slow clock | everything else | 1 × A40 | ~12 h |
| **S5 mode I, one edge** | the opened edge only, REF-C frozen | trunk + cascade | 1 × A40 | ~24 h per edge |

⛔ **REF-C is frozen throughout S1–S4.** The language module must never move the planner it is being measured against
— that is the C6 confound in language costume, and §9.7 refuses it at the interface.

---

## §5 Evaluation

⛔ **Four families first, always** (`EVAL_DOCTRINE.md`, CLAUDE.md): LONGITUDINAL (target speed, headway/time-gap/TTC) ·
LATERAL (heading, curvature, yaw-rate, cross-track) · TACTICAL (decision quality, confusion, goal setting) ·
STRATEGIC (compliance, `t_bin`). Never pooled. Every number carries its **T-tier** — **T1 is the capability tier**,
T0 is a diagnostic. Estimator: **paired episode-cluster bootstrap** over the 141 eval clips (`taniteval/ci.py`); never
`overlapping_holdout_se`.

**Then the FIFTH family — EXPLANATION FAITHFULNESS** (research file §3.4), *added*, never substituted:

| id | metric | must-read control |
|---|---|---|
| F1 | plan consistency (stated (LAT, LON) = selected anchor's cell class via `refc_tactical.factor_from_kinematics`) | ⭐ **constant-only must read the class prior ≈ 0.67** (corpus 86.59 % `lane_keep`, 75.76 % `steady`, 67.18 % both) — **any consistency below 0.67 is worse than a constant** |
| F2 | fan support@k | **k = 117 must read 1.0000 exactly** |
| F3 | referent grounding (precision / recall / NULL) | labels are **our geometry**, never the Alpamayo referents |
| F4 | contradiction rate, **per rule with its own n** | compliant module reads **0** |
| F5 | scene sensitivity (shuffle `fmap` within episode) | ⭐ **plan-echo control must read 0.0000** — this is what gives F5 its meaning |
| F6 | nav-echo quotient (accuracy **conditioned on nav**) + `Δ_nav-shuffle` | **nav-only control ≈ 0 beyond nav** — refcv3's 141/141 bijection, pre-armed |
| F7 | causal influence (force a token, measure the pick) | **exactly 0.0000 in mode M** |
| — | `goal_consistency` (distance to `g_tac` `(x,y)@τ`) | **shuffle** must read the marginal |
| — | `fan_endorsement` (does the endorsed candidate survive reach + Kamm?) | **random-candidate** control reads the fan's own survival rate |

**Nav compliance** is reported with its **shuffle and zero controls** (`taniteval/taniteval/nav_compliance.py`,
criteria v2.6.0: a rate without them is two violations). ⚠️ Strategic readouts are reported **conditional on
`H-NAVC-2`** while `g_str` remains NAV_BLIND (Δ_shuffle = 0.0000).

**Arms in every panel:** headline **P1** · **P0** scene-only · **C-A** constant · **C-B** plan-echo · **C-C**
scene-only-no-plan · **C-D** nav-only · **C-E** from-scratch floor · **deliberate-regression** (`L_agree` ×10). `n`
and `d` printed on every table; all hyper-parameters fit on the FIT split only.

---

## §6 Work packages

| id | work | owner | blocks on | GPU | why it is where it is |
|---|---|---|---|---|---|
| **WP-0** | Thor latency + memory baseline: REF-C batch-1 inference, then TanitLang prefill/decode at bf16/FP8/NVFP4 | Deployment & Opt | Thor free | **0 train**, ~2 h inference | ⛔ every latency number in this plan is an ESTIMATE until this lands. It is also the cheapest way to kill the design early if REF-C alone eats the tact |
| **WP-1** | `H-VLA-6` — is our Alpamayo draw trajectory-anchored? On the 4,416 non-null clips (41.7 % lateral disagreement) | Research Lab | none | **0** | decides whether the teacher may supply content as well as form |
| **WP-2** | window-level explanation labels: re-anchor 5.1 s → 8.0 s; synthesise per-window targets from `manoeuvre_sequence` + `obstacle.offline`; emit the XCoT token sequences with the deterministic ordering schema | DataFlyWheel | WP-1 | **0** | one CoT per clip is not a window label; without this there is nothing to train on |
| **WP-3** | `H-VLA-1` — the bridge: frozen SmolLM2-360M + `proj_in` (+LoRA), v7.2 class through the LM vs a linear probe on `pooled` | Arch & Inf | WP-2 (small slice) | dev box ~16 h | the single go/no-go for option (c) |
| **WP-4** | `H-VLA-7` — do 2–6 executable tokens carry the prose's decision information? | Research Lab | none | **0** | sizes the slow clock |
| **WP-5** | TanitLang S1–S3 training (mode M, all gates 0) | Arch & Inf | WP-3 | 1 × A40 ~40 h | — |
| **WP-6** | criteria registry: add `explanation_faithfulness` (F1–F7 + the controls) as **required** keys, bump the version, extend the checker | Benchmarks & Eval | none | **0** | ⭐ a family in prose is not enforced — *no prose correction can reach a glob* |
| **WP-7** | the eval harness: P0/P1/D tiers, the eight arms, the shuffle/zero interventions, paired bootstrap | Benchmarks & Eval | WP-6 | **0** | — |
| **WP-8a** | mode I, **edge 1**: fill `maneuver_logits [B, 5]` on the existing `hierarchy_hook` | Arch & Inf | WP-5, WP-7 | 1 × A40 ~24 h | ⭐ zero new wiring — the cheapest causal test of the whole thesis |
| **WP-8b** | mode I, edge 2: `g_tac_proposal` zero-init residual | Arch & Inf | WP-8a | ~24 h | — |
| **WP-8c** | mode I, edge 3: E19 `p_man`/`t_bin` — **needs E19 to land first** | Arch & Inf | E19 (refcv5 WP-8) | ~24 h | ⚠️ conditional on `H-NAVC-2` |
| **WP-9** | Thor deployment: TensorRT export (LM and any encoder split into two engines), FP8/NVFP4 bake-off, `H-VLA-3` and `H-VLA-8` | Deployment & Opt | WP-5 | inference only | — |
| **WP-10** | S4 prose head + FastDriveCoT-style field parallelism on the slow clock | Arch & Inf | WP-5 | ~12 h | the PI's text deliverable |

**Total pre-registered GPU:** ≈ **40 h** to a measurable mode-M module, ≈ **112 h** through all three mode-I edges.
WP-0/1/2/4/6/7 are **0 GPU and can all start now**.

---

## §7 Risks

| # | risk | severity | mitigation / early detector |
|---|---|---|---|
| R1 | ⭐ **The bridge is too small** — a frozen LM cannot read 160 REF-C tokens through ~6 M parameters | design-ending for option (c) | `H-VLA-1` (WP-3) answers it for ~16 dev-box hours. Fallback: staged unfreeze, then option (a) with the ceiling re-checked |
| R2 | ⭐ **The explanation is an echo of the plan** | the USP fails | F5 vs the plan-echo control, pre-armed; `H-VLA-4`. Published as a negative if it fails — no re-scoping |
| R3 | **REF-C alone eats the tact on Thor** | the module never fits | WP-0 measures it **first**, before any training |
| R4 | **The teacher is anchored and we distil rationalisation** | poisoned labels | `H-VLA-6` (WP-1), 0 GPU, before WP-2 |
| R5 | **The strategic seam is NAV_BLIND** (Δ_shuffle = 0.0000) so a strategic language claim is unfalsifiable | strategic claims void | every strategic readout is reported **conditional on `H-NAVC-2`**; the tactical path (WP-8a) does not depend on it |
| R6 | **The module wins the gradient competition** (368 M vs the 2.15 M cascade) | the hierarchy dissolves | REF-C frozen in S1–S4; every edge zero-init and individually gated; mode I opens one edge at a time |
| R7 | **A question becomes a control channel** | goal/situation disjointness broken | §9.4: the answer to a question may not route into a write port in the same tick |
| R8 | **Quantisation makes it slower, not faster** on Jetson | latency plan invalid | `2607.08029` MEASURED exactly this; WP-9 is a bake-off, not an assumption |
| R9 | **The contract moves under us** — E19 widening `str_goal_head` 3 → 23, the vocabulary v5 rebuild changing the anchor grid 13×9 → 13×11 (changes the selected-anchor class mapping), the §4 sampler changing the fan shape (changes `sel_idx`'s meaning), BEV tokens entering the decoder KV | silent invalidation | §9.8 names all four; each needs a note to the Master Mind rather than a local fix. **The eval harness (WP-7) pins the mapping by test, not by comment** |
| R10 | **Anchor mismatch (5.1 s vs 8.0 s) silently mis-joins the CoT** | 2.9 s of wrong labels | WP-2's first step; assert on content, print the joined n |

---

## §8 Manifest

| artifact | path | state |
|---|---|---|
| this plan | `Project Steering/REFCV5_VLA_EXTENSION_PLAN.md` | **in the repo**, `agent/arch-inf-20260803` |
| research file (survey, options, grounding, hypotheses, refusals) | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-vla-extension-frontier/RESULT.md` | **in the repo** |
| the binding contract | `Project Steering/REFCV5_DESIGN_PLAN.md` §9 | in HEAD (sibling stream) |
| `H-VLA-1 … H-VLA-8` | `Project Steering/GOALS_AND_CLAIMS.md` | **registered** |
| banked primaries (58 tagged `vla-frontier-2026-09-05`) | `TanitAD Research Lab/Library/papers/`, `library.json`, `LIBRARY.md` | **in the repo** |

**Escalations to the Master Mind (not "please merge" in a doc):**
1. **WP-6 is a criteria-registry change owned by Benchmarks & Eval** — the fifth family must land as *required* keys
   with a version bump, or it will be silently omitted from the first eval that could have used it.
2. **WP-8c depends on E19** landing in refcv5; the VLA stream cannot open the strategic edge before it.
3. **`H-NAVC-2`'s outcome gates every strategic language claim.** If `g_str` stays NAV_BLIND, this plan's strategic
   half is on hold and the tactical half (WP-8a) proceeds alone.
