# What SOTA driving planners condition their trajectory scorer on

**Written:** 2026-07-27 (Europe/Berlin; pods and logs are UTC) · **Host:** dev box, **CPU/web only —
no pod was contacted, no GPU was used.**
**Commissioned by:** the question Bar A left open
(`Benchmarks & Eval/Implementation/incoming/2026-07-26-bar-a-selector/BAR_A.md` §0.8, §6.2, §11.3):
*"the next probe is WHAT THE SCORE IS CONDITIONED ON, not how it is trained."*

Evidence stamps on every claim: class `MEASURED (ours + artifact)` · `PUBLISHED (cited)` ·
`INHERITED` · `ESTIMATED` · `HYPOTHESIS`, and where a published claim is quoted, whether the paper
**DEMONSTRATED** it (ran the ablation / measured it) or merely **ASSERTED** it (prose).

---

## 0. PRE-REGISTRATION — written and staged BEFORE the literature was read

> **Sequencing, stated exactly.** At the time §0 was written I had read only *our own* material:
> `AGENT_OPERATING_STANDARD.md`, `BAR_A.md`, `MODEL_REGISTRY.md`'s REF-C selection block, and the
> source of `flagship_v4.py` / `flagship_v15.py` / `refc.py` / `refc_rescorer.py`. **No paper had
> been read and no literature-search result had been returned.** Three literature agents had been
> dispatched but none had reported. §0 is unmodified from registration; §§1+ were written after.

### 0.1 The question, stated so it can come out either way

Bar A measured that **no re-scorer of our frozen fan reaches our deployed model, even in-sample**
(MEASURED, `raw/bar_a_produced.json`: in-sample ceiling `ade_0_2s` **0.4907** vs deployed v1
**0.4271**; out-of-fold **−4.20 %** and **−11.03 %**, both on the wrong side of zero). The
pre-committed reading was that the **information** is missing, not the objective.

This research asks the literature one question: **when a published planner's scorer works, what is
it reading that ours is not?** — and the symmetric one: **do the strong systems even learn their
scorer?**

### 0.2 ⚠️ PRE-REGISTERED FALSIFIER — what would make me recommend ABANDONING learned scoring

I commit, **before reading any paper**, to recommending **rule-based cost terms over our fan instead
of a learned scorer** if the literature shows any TWO of the following:

- **F1.** The strongest system on a *closed-loop* benchmark (nuPlan CLS / NAVSIM PDMS / CARLA) is
  rule-based or hybrid at the **scoring** stage, and its own paper attributes the margin to the
  rules rather than to the proposals. *(DEMONSTRATED, not asserted — an ablation table.)*
- **F2.** A paper reports a **learned-scorer-vs-rule-scorer head-to-head over the same proposal
  set**, and the rule scorer wins.
- **F3.** The learned scorers that do win are shown to depend on inputs we **cannot obtain**
  (HD map, lane graph, agent tracks with predicted futures) — because then "learned scoring" is not
  what is transferring, *the inputs are*, and on a corpus with no map the honest fallback is rules.
- **F4.** Distillation from rule-based teachers (Hydra-MDP) is shown to work **because the teacher
  is a rule**, i.e. the student's gain tracks the teacher's rule-metric rather than the imitation
  target.

**And the symmetric commitment.** I will recommend **keeping learned scoring** if the literature
shows learned scorers beating rule scorers over a *matched* proposal set on a closed-loop metric,
or if the winning systems' scorer inputs are ones we can actually build.

⚠️ **I must not simply re-run our own answer.** Our program has already measured *both* legs on
REF-C and the result is uncomfortable for **both** recommendations
(MEASURED, `MODEL_REGISTRY.md` REF-C selection block):

| | our own prior result | recovered |
|---|---|---|
| **REF-C v1.0** — hand-written cost re-rank, 0 new params | best blend weight is **λ = 0** (the untouched baseline); pure cost **−171 %** | **0.0 %** |
| **REF-C v1.2** — learned re-scorer, 47 trained arms, listwise attention over candidates, explicit kinematics, top-K | 0.46251 vs 0.47144, paired Δ +0.00893 [−0.0062, +0.0250] | **+2.9 %, NOT significant** |

So **on our own data the hand-written cost lost outright and the learned ranker won only
insignificantly.** Any recommendation that ignores this is a recommendation that has already been
falsified once at home. The live question is therefore not "learned vs rules" in the abstract but
**which INPUTS**, and whether any of the inputs the literature relies on are reachable for us.

### 0.3 ⚠️ PRE-REGISTERED CAUTION — the brief's own framing may inherit a retired claim

The brief states: *"our trajectory fan is good — `oracle_in_fan` 0.2505 vs deployed v1 0.4271, i.e.
the proposals are 41 % better than our best shipped model."*

I flag **before researching** that this sentence compares a **minimum over 256 candidates scored
against ONE realised future** with a **single realised prediction**. Those are not the same kind of
quantity. Our own registry has already retired the identical framing on REF-C
(MEASURED, `MODEL_REGISTRY.md`): *"the oracle gap is ~92 % irreducible … the 0.1640 oracle is a
minimum over 256 candidates scored against ONE realised future — most of the distance below the
incumbent is that minimum's statistics over aleatoric outcomes, not recoverable signal"*, and Bar A
§12 R-3 turned it into a rule: **"an oracle-vs-selected gap is a BOUND, not a budget."**

I therefore pre-register that **a large part of what I expect to find is that the literature's
`minADE_k`-vs-top-1 gap has exactly this property**, and that if it does, the correct deliverable is
partly a *deflation* of the premise rather than a scorer design. I commit to saying so if I find it.

### 0.4 What would make each recommendation tier admissible

| tier | admissible if |
|---|---|
| **RECOMMEND** | ≥ 2 independent papers DEMONSTRATE the input matters (ablation with the input removed), **and** we can construct the input from PhysicalAI-AV or a staged asset |
| **CONDITIONAL** | demonstrated in the literature, but the input needs an asset we do not yet have |
| **REJECT** | the input is demonstrated to matter only on a benchmark with the ego-status confound, or only asserted |

### 0.5 Constraints that bound every recommendation — stated up front

- **PhysicalAI-AV has NO map, lane graph, junction annotation, traffic-light feature or route/goal
  signal**, and `egomotion` carries no lat/lon/GNSS (MEASURED, CLAUDE.md, five independent probes,
  settled). **Every map-conditioned recommendation is therefore CONDITIONAL by construction.**
- `obstacle.offline` exists on **97.44 %** of the corpus and its enum over **87,481 cuboids is 10
  classes, all dynamic agents** (MEASURED, CLAUDE.md). **3D agent tracks ARE available to us.**
  Our ingest reads 4 of 36 features.
- **The fan is cached.** `/workspace/_bara/cache_{produced,oracle}_stride1.pt`, 4.07 GiB each,
  6,844 windows (MEASURED, `BAR_A.md` §9). A re-scoring experiment costs **~13 GPU-min**, not a
  GPU-week. This is what makes the cheapest-test deliverable real.

---

*(Sections 1 onward were written after the literature was read.)*

---

## 1. THE HEADLINE — three findings, and the third explains our measurement

**1. Every strong published scorer reads things ours does not — but the input that matters most is
not a sensor, it is a LABEL.** The systems that win NAVSIM/nuPlan do not score candidates by
*distance to the one realised future*. They score each candidate by a **per-candidate,
deterministic, rule-computed verdict** — collision, drivable-area, time-to-collision, progress,
comfort — obtained by simulating that candidate against the logged scene offline.
`PUBLISHED (cited — Hydra-MDP arXiv 2406.06978 §3; NAVSIM docs/metrics.md; WoTE arXiv 2504.01941 §3).`

**2. That difference is DEMONSTRATED to be worth more than any objective change, on a matched
proposal set.** Hydra-MDP++ holds vocabulary, backbone and architecture fixed and changes only the
training target: **imitation-only 85.0 PDMS → + rule-teacher distillation 86.5 (+1.5)**; on the
extended metric **76.8 → 80.6 EPDMS (+3.8)**.
`PUBLISHED (cited — Hydra-MDP++ arXiv 2503.12820, Tables 3 & 4, NAVSIM navtest) — DEMONSTRATED.`

**3. ⇒ Our imagination-scoring failure was not a world-model failure. It was a MISSING VERDICT.**
We measured that *"the world model does not veto an implausible plan — it obediently simulates
it"* (MEASURED, commit `d471fdf`). **WoTE (arXiv 2504.01941) does exactly what we tried — rolls a
BEV world model forward per candidate — and it works, because the rollout is not scored for
self-consistency: it is scored by a reward head distilled from a simulator's NC/DAC/TTC/Comf/EP
verdicts.** `PUBLISHED (cited — WoTE §3.3, Table 3) — DEMONSTRATED: no evaluator 81.0 → evaluator on
current state 83.2 → evaluator + world-model future states 85.6 PDMS, NAVSIM navtest.`
A world model has no opinion about whether a plan is *good*; it only says what would *happen*. The
missing component was never the simulator — it was **the judge**.

> ### The one-line recommendation
> **Stop asking our scorer "which candidate is closest to the realised future?" and start asking
> "what would happen if we drove this candidate, and is that outcome acceptable?"** — a
> per-candidate label that is a deterministic function of the candidate and the scene, not a draw
> from an aleatoric future. On our corpus the buildable version of that label is a **collision /
> time-to-collision / progress / comfort check against `obstacle.offline`'s 3D agent tracks**,
> which exist on 97.44 % of the corpus and which our ingest does not read.

⚠️ **And a deflation the brief asked for by implication.** §0.3's pre-registered caution was
correct, and I state it as a finding: **`oracle_in_fan` 0.2505 is not evidence that "our proposals
are 41 % better than our best shipped model."** It is a minimum over 256 candidates against one
sampled future. Our own registry already retired this framing on REF-C ("~92 % irreducible"), Bar A
§12 R-3 turned it into a rule, and the literature is consistent with it (§4). "The fan is good" and
"the fan contains a lucky member" are different claims; only the second is measured.

---

## 2. THE CRUX — the per-system scorer-INPUT table

**How to read the columns.** `cand geom` = does the score read the candidate trajectory's own
geometry · `agents` = other road users' current boxes/tracks · `agent futures` = their predicted or
log-replayed future motion · `map` = drivable area / lane geometry · `route` = route or navigation
goal · `ego` = ego kinematic state · **`rule verdict` = is the score trained against a per-candidate
rule-computed outcome** — the column that turns out to matter.

### 2.1 The NAVSIM / closed-loop-metric family

| system | cand geom | scene feats | agents | agent futures | map | route | ego | **rule verdict** | scoring is |
|---|---|---|---|---|---|---|---|---|---|
| **Hydra-MDP** (2406.06978) | vocab embedding | env tokens (img + LiDAR) | via env tokens | ⚠️ TEACHER only | ⚠️ TEACHER only | ✅ nav goal | ✅ ego status | ✅ **NC·DAC·TTC·C·EP distilled per candidate** | **hybrid** — learned per-rule heads, **grid-searched hand weights** |
| **Hydra-MDP++** (2503.12820) | as above | as above | as above | ⚠️ TEACHER only | ⚠️ TEACHER only | ✅ | ✅ | ✅ **+ TL, DDC, LK, EC** | hybrid, same shape |
| **WoTE** (2504.01941) | ✅ per-candidate action embedding | **BEV state rolled FORWARD per candidate** | ✅ via BEV | ✅ **predicted future BEV states** | ✅ via BEV semantics | ✅ | ✅ | ✅ **simulator NC/DAC/TTC/Comf/EP via BCE** | hybrid |
| **GTRS** (2506.06664) | ✅ trajectory tokenizer | image → BEV | via BEV | ⚠️ TEACHER only | ⚠️ TEACHER only | ✅ | ✅ | ✅ EPDMS sub-metrics | learned scorer over a super-dense vocabulary |
| **GoalFlow** (2503.05689) | ✅ | BEV | via BEV | — | ✅ **drivable-area polygon, at scoring time** | ✅ | ✅ | ✅ **DAC is a LIVE rule term** | **hybrid** — `w₁·log δ_dis + w₂·log δ_dac` |
| **DiffusionDrive** (2411.15139) | ✅ trajectory features | BEV, deformable spatial X-attn | ✅ **agent queries** | — | ✅ **map queries** | ✅ | (not enumerated) | ❌ **imitation only** | learned confidence head, argmax |

*(VAD / VADv2 / UniAD / PARA-Drive / GenAD / TCP / Transfuser and the nuPlan family are in §3 —
they were researched by parallel agents and are integrated there.)*

### 2.2 TanitAD flagship v4 — read from source, not from prose

`stack/tanitad/models/flagship_v4.py:198-238` · `flagship_v15.py:250-470` · `refs/refc.py:531-541`.
`MEASURED (ours — source read).`

| | ours |
|---|---|
| cand geom | ✅ **yes** — `q = traj_proj(x_est)`, so `conf_head(q)` does see the candidate; **plus** its implied terminal speed in the `sel_gate` term |
| scene feats | ✅ frozen-encoder readout cells (W × 16) via cross-attention |
| **agents** | ❌ **none** |
| **agent futures** | ❌ **none** |
| **map** | ❌ none — **and none exists in PhysicalAI-AV** (settled, five probes) |
| route | ✅ `route_emb` + `route_graded` (⚠️ minted; the oracle version is not deployable) |
| ego | ✅ `v0` through `measurement`, plus the `vt_band` target-speed token |
| imagination | ✅ **but NOT candidate-conditioned** — `imagine_probes` rolls a *shared probe action vocabulary*, so the imagined tokens are **identical for all 256 candidates** |
| **rule verdict** | ❌ **none.** `refc.py:544 _grounded_score` — a param-free progress-minus-lateral-excursion proxy — **exists and is hard-disabled** (`grounded_selector=False`, `flagship_v15.py:314`) |
| trained against | **`argmin` of ADE to the ONE realised future** (`v15_losses`, `flagship_v15.py:598-605`) |

> **Two structural gaps — and neither is the one we have been debating.**
>
> **(a) The imagination is not candidate-conditioned.** WoTE's entire mechanism is one BEV rollout
> *per candidate*; ours is one rollout per *probe action*, shared across the fan. **A feature that
> is identical for every candidate cannot rank them**, by construction.
> `MEASURED (ours, source read) + PUBLISHED (WoTE §3.3).`
>
> **(b) The target is aleatoric.** Every system in §2.1 supervises the score with a deterministic
> per-candidate verdict. We supervise it with the identity of whichever candidate happened to land
> nearest one sampled future. **This alone is sufficient to explain a scorer that cannot be fitted
> even in-sample** — in-sample fitting of an irreducible label is exactly what `0.4907` looks like.

### 2.3 What the rule teachers actually read — the transferability question

`PUBLISHED (cited — NAVSIM docs/metrics.md; Hydra-MDP++ §3).` The right-hand column is ours.

| rule | reads | **buildable on PhysicalAI-AV?** |
|---|---|---|
| **NC** — no at-fault collision | other agents' boxes + their **log-replay** futures | ✅ **YES** — `obstacle.offline`: 97.44 % of corpus, 10 dynamic-agent classes, 87,481 cuboids. Log-replay futures come free — the tracks *are* the future. |
| **TTC** — time-to-collision | same, plus ego kinematics | ✅ **YES**, same source |
| **C / HC / EC** — comfort | ego accel, jerk, yaw rate — **trajectory-only** | ✅ **YES — needs nothing but the candidate itself** |
| **EP** — ego progress | progress along a **reference route** | ⚠️ **PARTIAL** — no route in corpus; a heading-projected along-track proxy is buildable |
| **DAC** — drivable area | HD map polygon | ❌ **NO** — no map in PhysicalAI-AV |
| **DDC / LK** — direction, lane keeping | lane centerlines | ❌ **NO** |
| **TL** — traffic lights | traffic-light state + position | ❌ **NO** |

> **Three of seven rule families are buildable today from an asset we already have and do not read;
> three need a map we do not have; one is partial.** Note that the two most heavily weighted
> *multiplier* terms in PDMS are NC and DAC — we can build one and not the other.

**⚠️ The rules are TRAINING-TIME ONLY.** In Hydra-MDP / Hydra-MDP++ / GTRS the map and agent
metadata are needed only to *mint the per-candidate labels offline*; the deployed student reads raw
sensors. `PUBLISHED (cited — Hydra-MDP++ §3; GTRS abstract — scoring "given raw sensor data,
without access to ground-truth perception").` **This is decisive for us: we may use privileged
`obstacle.offline` tracks to LABEL our cached fan without needing them at inference.**

---

## 3. THE nuScenes END-TO-END FAMILY — and the finding that nearly inverts §1

### 3.1 The input table

`PUBLISHED (cited).` **Only VADv2 has a candidate scorer at all.** The rest regress a single
trajectory, so "what the scorer reads" is "what the planning head reads".

| system | candidate scorer? | what the planning cost/head reads | ego status |
|---|---|---|---|
| **UniAD** (2212.10156) | ❌ regression + **inference-only** optimizer | ego query from MotionFormer; BEV features (cross-attn); 3 learnable command embeddings. **Optimizer additionally reads predicted future occupancy Ô** | not in the head per the text; the release feeds it via BEV |
| **VAD** (2303.12077) | ❌ regression + 3 constraint **losses** | ego query after ego↔agent and ego↔map-vector cross-attn; driving command | optional (contested — see §3.4) |
| **VADv2** (2402.13243) | ✅ **4,096-entry vocabulary** | `p(a) = MLP(Transformer(E(a), E_env) + E_navi + E_state)`; `E_env` = map + agent + traffic-element + image tokens | ✅ additive `E_state`, **never ablated** |
| **PARA-Drive** (CVPR 2024) | ❌ pure MLP regression | BEV feature map **only**, plus ego info (command, CAN bus, history). **No map, motion or occupancy queries** — and it beats UniAD and VAD | ✅, honestly reported |
| **GenAD** (2402.11502) | ❌ generative sampling | map-aware instance tokens (ego + agents + map), latent `z`, GRU rollout | one token |
| **TCP** (2206.08129) | ❌ two branches fused by a **hand-written rule** | shared image features + `(command, current speed)` | ✅ speed explicit |
| **Transfuser** (2205.15997) | ❌ regresses 4 waypoints | 64-d fused image+LiDAR-BEV feature, current position, **goal point** | not by default |
| **Transfuser++** (2306.07957) | ❌ but **decouples** path from speed | path (10 points, 1 m apart, time-independent) + **target speed as a 4-way classification**, confidence-weighted | — |

**UniAD's collision cost, exactly** (inference only):
`τ* = argmin_τ [ λ_coord‖τ − τ̂‖₂ + λ_obs Σ_t D(τ_t, Ô^t) ]`, `D` a Gaussian kernel over occupied
cells; Newton's method; repo defaults `σ = 1.0`, `alpha_collision = 5.0`.
> **It reads exactly two things: distance to the regressed trajectory, and predicted future
> occupancy. No map, no route, no agent identity, no kinematics.**

**VAD's three constraints** — all **training losses only**, no inference enforcement: ego-agent
collision (reads *predicted* agent motion; δ_X = 1.5 m longitudinal, δ_Y = 3.0 m lateral),
ego-boundary overstepping (map boundary vectors, δ_bd = 1.0 m), ego-lane directional (nearest lane
divider direction). Worth **0.76 → 0.72 m L2** and **0.28 → 0.22 % collision**. ⚠️ nuScenes
open-loop — see §3.4.

### 3.2 ⚠️⚠️ THE FINDING THAT NEARLY INVERTS §1 — inputs are worth almost nothing; SUPERVISION is worth 17×

`PUBLISHED (cited — VADv2 arXiv 2402.13243v1, Table 3, read directly from extracted PDF text) —
DEMONSTRATED.` VADv2 is the only system in this family with a candidate scorer, and it ablates that
scorer's **inputs** and its **supervision** in the same table. 1 s L2, CARLA:

| ablation | 1 s L2 |
|---|---:|
| full | **0.082** |
| **− image tokens** | 0.083 |
| − map tokens | 0.086 |
| − agent tokens | 0.089 |
| **− the distribution LOSS** | **1.415** ← **17×** |

> **Conditioning inputs are worth ~0.007 m. The supervision scheme is worth ~1.33 m — 190× more.**
> Corroborated architecturally by **PARA-Drive**, whose planner reads **only** BEV + ego info — no
> map, motion or occupancy queries — and beats both UniAD and VAD.

**This does not overturn §1; it sharpens it, and it re-points the recommendation.** The brief asked
"what should our scorer READ". The strongest available evidence says: **the highest-leverage change
is not an extra input tensor, it is the per-candidate LABEL.** Note what VADv2's two losses are:

- `L_distribution` — KL against a **soft, distance-weighted** target: *"the ground truth trajectory
  is added to the planning vocabulary as the positive sample… **trajectories close to the ground
  truth trajectory are less penalized**."* This is **exactly** the anti-pattern to our
  `argmin`-hard-label CE — and note our own REF-C v1.2 independently found the same thing:
  *"hard-argmin is the worst target in all five feature configurations."* `MEASURED (ours) +
  PUBLISHED (cited) — two independent confirmations of the same mechanism.`
- `L_conflict` — *"if one action in the planning vocabulary conflicts with other agents' future
  motion or road boundary, the action is regarded as a negative sample, and we impose a significant
  loss weight."* **This is a rule-based per-candidate LABEL derived from agent futures.** It is the
  same construction as Hydra-MDP's NC/TTC teachers, arrived at independently.

> ⇒ **Agent tracks matter to every strong system, but they enter as a LABEL SOURCE, not as a scorer
> input tensor.** That resolves the apparent conflict between §1 and this section, and it is what
> the recommendation in §6 is built on.

### 3.3 ⛔ RULES AT INFERENCE ARE A MEASURED NET LOSER — my pre-registered falsifier fired BACKWARDS

`PUBLISHED (cited — PARA-Drive CVPR 2024, Table 3, read directly) — DEMONSTRATED.` UniAD's
occupancy-based collision optimiser is the one famous inference-time rule-based post-optimiser in
this literature. **Removing it improves BOTH metrics:**

| | collision ↓ | L2 (m) ↓ |
|---|---:|---:|
| UniAD | 0.40 | 0.83 |
| **UniAD − the occupancy collision optimiser** | **0.16** | **0.74** |

PARA-Drive's stated cause: the test-time optimisation *"is not in the training process, and it tends
to generate trajectories that deviate from human driving logs… TTO often results in zigzag-like
trajectories near multiple objects, thereby increasing the L2 error and cannot guarantee to avoid
collision."* ⚠️ **Both sides given:** UniAD's own Table 10 reports the optimiser *buying* collision
rate (0.30/0.51/1.39 → 0.13/0.42/1.05) at an L2 cost (0.44 → 0.54 @1 s). The two groups disagree on
the value, and **agree it costs L2**; PARA-Drive's is the re-measurement under a corrected collision
metric.

> ### ⛔ THE PRE-REGISTERED FALSIFIER (§0.2) RESOLVED — and it fired in the direction I did not expect
>
> **F1 and F2 are NOT satisfied.** The strong systems are hybrid, but the rules live in the
> **training target**, never in an inference-time cost. Where an inference-time rule cost was
> actually measured, it **lost**. **F3 is PARTIALLY satisfied** (DAC/DDC/LK/TL need a map we do not
> have) and **F4 IS satisfied** (Hydra-MDP++ 85.0 → 86.5 PDMS from rule distillation alone).
>
> **⇒ I do NOT recommend abandoning learned scoring for rule-based cost terms over our fan.**
> The recommendation is the specific hybrid the evidence supports: **keep the learned scorer,
> replace its LABEL with rule-computed per-candidate verdicts.**
>
> ⭐ **And this is triangulated at home.** Our own REF-C v1.0 measured a hand-written **inference-time**
> cost re-rank at **0.0 % recovered, pure cost −171 %** (MEASURED, `MODEL_REGISTRY.md`). That is
> the *same* result as PARA-Drive's, on a different system, on a different corpus, found
> independently. **Two independent measurements say inference-time rule re-ranking does not work;
> four papers say rule-derived training labels do.**

### 3.4 ⚠️ THE EGO-STATUS CONFOUND — mandatory qualifier on every nuScenes number above

`PUBLISHED (cited — BEV-Planner arXiv 2312.03031 Tables 1–2; AD-MLP arXiv 2305.10430 Table 1;
PARA-Drive Table 6) — DEMONSTRATED by three independent groups.`

- **Ego status roughly HALVES L2 on every architecture** (BEV-Planner T.1): UniAD 1.03 → 0.46,
  VAD-Base 1.25 → 0.37, BEV-Planner 0.55 → 0.35.
- **A perception-free MLP wins.** AD-MLP (21-dim input, zero perception) scores **L2 avg 0.29** vs
  VAD-Base 0.37 and UniAD 1.03. Its own ablation: the **high-level command alone** moved L2
  0.49 → 0.29 — the single most valuable input, and it is not a perception input.
- **Under PARA-Drive's standardised protocol** the ego-only MLP scores **L2 0.5568**, statistically
  indistinguishable from PARA-Drive's full stack (0.5574) and **better than UniAD (0.8317) and VAD
  (0.7830)** — and it stays better *even on the 686 turning/lane-change-only frames*
  (0.9360 vs 0.9935 / 1.0840). **Only collision-on-hard-frames and map compliance expose it**
  (its collision explodes 4.7×, 0.20 → 0.94, on the targeted set).
- **Perturbation test** (BEV-Planner T.2, VAD-Base, sensors held fixed): nominal L2 0.37;
  **v × 0.0 → 6.16**; v = 100 m/s → **208**. A sensor-based planner should barely move.

**Benchmark-hygiene findings that matter beyond nuScenes** `PUBLISHED (cited — PARA-Drive) —
DEMONSTRATED`: four protocol inconsistencies make published numbers non-comparable (averaging
convention, agent filtering, frame masking, first-frame noise), and re-measured on a common
protocol **the headline "VAD beats UniAD" margin is ~87 % protocol artifact** (δ 1.03−0.72 →
0.9474−0.9086). Worse, with axis-aligned ego boxes on a 200×200 grid **the ground-truth trajectories
themselves collide at 0.384 %**; oriented boxes → 0.32 %, finer discretisation → 0.00 %.

> **⚠️ Read against our own program:** our headline `ade_0_2s` is an **open-loop displacement
> metric in exactly the family this critique targets**, and our head takes `v0` and a target-speed
> band as explicit inputs. This does not invalidate Bar A — Bar A is a *paired* comparison of arms
> that all share the same ego inputs, so the shortcut cancels. But it is a strong argument that
> **`ade_0_2s` alone cannot adjudicate a scorer**, and it is a second, independent reason to move
> to rule-style per-candidate outcome metrics (§6).

### 3.5 The published fixes for exactly our failure — DriveSuprim

`PUBLISHED (cited — DriveSuprim arXiv 2506.06659v3, AAAI 2026, read directly from PDF) —
DEMONSTRATED.` DriveSuprim runs *our experiment* on NAVSIM and names three causes for why a learned
scorer cannot find the good candidate. All three are transferable to our fan:

| their diagnosis | their fix | measured |
|---|---|---|
| *"easy-to-reject options dominate the training process and gradient"*, so hard negatives get no signal | **coarse-to-fine two-stage scoring**: score all 8,192, filter to 256, **re-score survivors with a dedicated decoder supervised only on the filtered set** | Hydra-MDP → DriveSuprim, ViT-L, NAVSIM v1: **89.9 → 93.5 PDMS (+3.6)** |
| directional bias — only 18 % of NAVSIM GT involves turns > 30° | rotation-based sensor-space augmentation (Θ = π/6) | included above |
| **hard binary labels** | **self-distillation with EMA teacher soft labels** (δ_m = 0.15) | included above |

> ⭐ Their first fix is **top-K restriction**, which our own REF-C v1.2 already implements
> (`RescorerConfig.topk = 8`) and swept — *"K = 8–32 a flat plateau"* (MEASURED). **So we have
> already run the single most-cited fix and it did not rescue us** — which is itself informative,
> and which is why §6 does not recommend re-running it.
> **⚠️ Even so: 93.5 is still ~5 PDMS below their own 256-oracle of 98.7. Published SOTA NARROWS
> this gap; it does not close it.**

---

## 4. ORACLE-vs-SELECTED — the gap is the field's steady state, not our defect

### 4.1 Every published gap I could substantiate

`PUBLISHED (cited), single-hop extraction — see the caveat below.` PDMS is a *score*, so it is
converted to the failure domain (100 − PDMS) — the only honest way to compare a score with an error.

| system | benchmark + split | candidates | oracle | selected | **ratio** | what the paper says closes it |
|---|---|---:|---:|---:|---:|---|
| **LaneGCN** (2007.13732) T.1 | Argoverse 1 **test**, minADE | 6 | 0.87 m | 1.71 m | **1.97×** | not discussed |
| **LaneGCN** minFDE | same | 6 | 1.36 m | 3.78 m | **2.78×** | not discussed |
| **CoverNet** (1911.10298) T.2 | nuScenes prediction, 6 s | 5 / 10 | 1.96 / 1.48 m | 3.87 m | 1.97× / **2.61×** | **set construction** (§5) |
| **DriveSuprim** (2506.06659) T.1 | NAVSIM PDMS | 256 | 98.7 | 91.9 | **6.2×** (failure domain) | coarse-to-fine 2-stage + rot-aug + self-distillation; total buy **81.4 → 83.1 EPDMS (+1.7)** |
| **DriveSuprim** | same | **4** | **94.5** | 91.9 | — | ⚠️ **the oracle over just the top-4 (94.5) already ≈ human (94.8)** |
| **CLOVER** (2605.15120) T.5 | NAVSIM PDMS | 64 | 0.9933 | 0.9369 | **9.4×** | expanding proposals: oracle 0.9933 → 0.9976, selected 0.9369 → **0.9413**. **The gap does not move.** |
| **OURS** | TanitAD val, `ade_0_2s` | 256 | 0.2505 | 0.8563 | **3.42×** | — |

> ### FINDING — our 3.42× is **unremarkable**, and on the mild side.
> `PUBLISHED (cited, multi-source) — DEMONSTRATED.` **Nobody has closed this gap.** Treating our
> gap as an engineering surplus to be recovered was never supported by the literature, and Bar A
> measured it not to be. **This closes the question the brief asked in priority 4: yes, published
> versions of our gap exist; they are the same size or larger; and the answer to "how do they close
> it" is that they mostly do not.**

### 4.2 ⭐ What the literature does NOT have — and we do

`PUBLISHED-ABSENCE, verified by full-text term search across 7 papers.` The following papers contain
**no** occurrence of "oracle", "upper bound", "ceiling" or "vocabulary coverage":
**Hydra-MDP** (builds a 4,096/8,192 vocabulary and never measures what it could achieve),
**GTRS** (the NAVSIM v2 challenge *winner*, trains a scorer on a 16,384 vocabulary, same silence),
**VADv2** (no coverage analysis and **no vocabulary-size ablation at all**), **TNT**, **DenseTNT**,
**MTR**, and **SparseDriveV2** — the last despite being titled *"Scoring is All You Need"*.

> **⭐ ESCALATION — this is a publishable measurement and it is ours.**
> The literature reports only the **unrealisable** oracle, which says nothing about whether a
> ranker can be built. **Bar A's in-sample ceiling — a re-scorer fitted on the very windows it is
> then scored on, zero generalization gap — appears to be novel.** Our ordering
> **0.4271 (regression) < 0.4907 (best achievable ranker) < 0.8563 (actual)** is, as far as this
> search reaches, **the first published-grade demonstration that a regression head can beat the
> BEST POSSIBLE ranker of a frozen candidate set.** Bar A §12 R-3 already turned it into a rule
> ("an oracle-vs-selected gap is a BOUND, not a budget"); the literature search says nobody else
> has stated it. `MEASURED (ours, raw/bar_a_produced.json) + PUBLISHED-ABSENCE (7 papers).`

### 4.3 The one genuine disagreement — and our data resolves it for us

`PUBLISHED (cited) — three positions, all on NAVSIM:`

- **DriveSuprim** (2506.06659) — *the set is fine, selection is the gap*: selection-based methods
  "struggle to distinguish the optimal trajectory from similar but suboptimal alternatives".
- **TOAD** (2606.07170) — *the set is the bottleneck*: "a weak set of candidates limits planning
  performance regardless of the scorer's quality".
- **CLOVER** (2605.15120) — *both*: performance is "constrained by two coupled factors: whether the
  generator covers high-quality alternatives, and whether the scorer can identify them."

**CLOVER's own Table 5 is the sharpest datum and it undercuts the optimistic reading:** expanding
the proposal set moved the *oracle* (0.9933 → 0.9976) and left the *selected* score essentially
where it was (0.9369 → 0.9413). **A better fan did not produce a better pick.** That is the same
shape as our result, measured on a different system, on a closed-loop metric.

⚠️ **Extraction caveat, stated once and applying to all of §4–§5.** These numbers reach me through
**one automated extraction hop** (a summariser reading arXiv HTML), not my own eyes on the PDF.
Given this program's retraction history, **class them `PUBLISHED (cited) — SINGLE-HOP` and re-verify
against the PDF any number that would decide a GPU-day.** No recommendation in §6 rests on a single
one of them.

---

## 5. PROPOSAL PLAUSIBILITY — and a mechanism in our own code

### 5.1 What published systems constrain, and what it buys

| system | candidate set | constraint applied BEFORE scoring | measured benefit |
|---|---|---|---|
| **CoverNet** (1911.10298) | fixed / **dynamic** / hybrid | **forward-integrates a kinematic vehicle model from the CURRENT state** over diverse constant lat/lon acceleration sequences; lateral accel normalised across speeds; ε-coverage bagging | ⭐ **the ablation the brief hoped for.** minADE₅ (nuScenes, 6 s): fixed ε=2 **2.62** → dynamic ε=3 **2.02** → hybrid **1.96**; and reaches ε=2 m coverage with **~half** the trajectories |
| **PDM** (2306.07962) | **15** = 5 IDM target speeds × 3 lateral offsets | target speeds **{20,40,60,80,100} % of the posted limit**; every proposal simulated through **LQR + kinematic bicycle**, 4 s @ 10 Hz | "No lon." ablation **CLS-R 92 → 88** (−4). ⚠️ conflates removing proposals with removing speed variation |
| **PRIME** (2103.04027) | ~484 feasible trajectories per target | Frenét sampling with explicit limits **v_max 33.33 m/s, a ∈ ±8 m/s², κ ≤ 0.33** | no isolated generator ablation |
| **TOAD** (2606.07170) | on-the-fly, searched in **control space** | kinematic bicycle with bidirectional traj↔control mapping → "every sample is smooth and feasible" | 94.7 PDMS navtest v1 (−0.1 below human) |
| **VADv2** (2402.13243) | 4,096, FPS over human demo actions | **feasible by construction** — "sampled from driving demonstrations and thus naturally satisfies the kinematic constraints" | none reported |
| **Apollo EM** (1807.08048) | DP lattice → QP | QP bounds on speed, accel and **jerk**; curvature-based feasibility | 3,380 h / 68,000 km deployment; no numeric limits published |
| **Hydra-MDP / GTRS / SparseDriveV2** | k-means / super-dense vocab | **none stated** | — |

**Published answers for bounding a LONGITUDINAL band given current speed — they disagree on the anchor:**
1. **PRIME — anchored to CURRENT speed**, the only explicit formula found:
   `v ∈ [max(0, ṡ₀ − δ⁻·T), min(ṡ_max, ṡ₀ + δ⁺·T)]`. ⚠️ numeric δ⁺/δ⁻ **UNVERIFIED**.
2. **PDM — anchored to the SPEED LIMIT**, feasibility enforced downstream by IDM + LQR + bicycle.
3. **CoverNet — anchored to the current STATE** by forward integration, and **the only one with a
   published measurement of what it buys.**

### 5.2 ⚠️ CORRECTION to the brief's premise — our anchors ARE demonstration-derived

`MEASURED (ours — source + launch-record read).` The brief implies our fan is unconstrained. Read
from source, that is **half wrong**, and the half that is right localises the defect:

- **The anchor vocabulary is demonstration-derived and therefore feasible by construction**, exactly
  like VADv2's. `flagship-v4-fromscratch-30k` was launched with
  `--anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt`
  (`…/2026-07-23-v4-fromscratch-launch/LAUNCH_CONFIRMED.md:29`), built by
  `build_refc_anchors.py --data-root …` = **FPS over the real ego-frame waypoint targets of every
  window of the parity corpus**, `[256, 20, 2]`, pool 200 k.
  ⚠️ Note `build_refc_anchors.py` chooses **FPS deliberately over k-means** because the corpus is
  ~74 % straight — i.e. **it is designed to over-sample the extremes**, which is a feasibility
  argument in one direction and a risk in the other.
- **The refinement is unbounded.** The emitted fan is `x = x_in + offset` with `offset =
  offset_head(q)` — a plain `Linear(d, n_steps*2)` with **no kinematic constraint of any kind**
  (`refc.py:539-540`). Nothing in the emitted path is speed-bounded, acceleration-bounded or
  jerk-bounded.

> **⚠️ OPEN AND CHEAPLY CHECKABLE — I did not resolve it and I will not assert it.** Whether the
> 108.7 m / 181 km/h span lives in the **anchors** or is **manufactured by the offset head** is
> decidable at **ZERO GPU**: the anchor file is 42,550 bytes and the refined fan is already cached.
> `span(anchors)` vs `span(x_in + offset)` is a two-line comparison. **This is test T0 in §7 and it
> should run before anything else, because the two answers imply different fixes** — rebuild the
> vocabulary vs clamp the refinement.
> *(Prior probability note, offered as HYPOTHESIS not finding: `synth_anchor_pool` — the fallback
> vocabulary, NOT the one v4 used — samples `v0 ~ U(0, 30 m/s)` with `accel ∈ ±3`, which tops out
> near 36 m/s ≈ 130 km/h, below the observed 181 km/h. That is weak evidence for the offset head,
> and it is **not** evidence about the real anchors.)*

**The tension the literature leaves us with.** VADv2, Hydra-MDP, GTRS and SparseDriveV2 all obtain
feasibility *for free* by sampling from human demonstrations, and then apply **no filter**. PDM,
PRIME, TOAD and CoverNet apply an explicit kinematic model. **We are in a fifth position nobody
publishes: a demonstration-derived vocabulary whose feasibility is then destroyed by an unbounded
learned refinement, with no filter downstream.**

### 5.3 ⭐⭐ THE MOST IMPORTANT NUMBER IN THIS REPORT — a bigger fan makes the best rule scorer WORSE

`PUBLISHED (cited — LLM-Assist arXiv 2401.00125, Table 1, read VERBATIM from extracted PDF text) —
DEMONSTRATED.` nuPlan **Val14, closed-loop non-reactive.** The authors hand PDM-Closed — the
strongest rule scorer in the field — progressively more proposals:

| # proposals | **score** | collisions | **TTC** | drivable | **comfort** | progress |
|---:|---:|---:|---:|---:|---:|---:|
| **15** | **92.51** | 98.05 | 93.11 | 99.55 | 95.19 | 91.75 |
| **8,505** | **77.78** | 91.92 | **62.89** | 98.64 | **78.68** | **95.60** |

Their caption, verbatim: *"PDMClosed fails to select the best proposal when presented with too many
options, as it relies on a constant velocity simulator."*

> **Read the sub-metrics — this is textbook Goodharting.** Progress goes **UP** (91.75 → 95.60)
> while TTC **collapses** (93.11 → 62.89) and comfort **collapses** (95.19 → 78.68). A 567× larger
> fan let the search find candidates that **exploit the scorer's own simulation error**.
>
> **⇒ This is almost certainly the mechanism behind our imagination-scoring failure**, stated in
> published form: *"the world model does not veto an implausible plan — it obediently simulates
> it, so consistency-scoring ranks the absurd candidates first"* (MEASURED, ours) is the same
> sentence as *"fails to select the best proposal when presented with too many options, as it
> relies on a constant velocity simulator"* (PUBLISHED). **A large fan is not a free asset; it is
> an adversarial search against your scorer's approximation error.**

**Published fan sizes, for scale.** PDM **15** · PLUTO top-K **20** · DiffusionDrive **20** (its
Table 6: 10 → 84.9, 20 → 88.1, 40 → 88.2 PDMS — **saturated at 20**) · DTPP ≤30 · "Slow Brain,
Fast Planner" **plateaus at K ≈ 18–24** · DriveSuprim 8,192 → **coarse-filtered to 256, then
re-scored by a dedicated second-stage decoder** · **ours: 256, single-stage, unfiltered.**

---

## 6. THE RECOMMENDATION — ranked, with the implementation delta against our factorised head

**The diagnosis in one paragraph.** Our scorer is asked an **unanswerable question** (which of 256
candidates is nearest one sampled future), over a fan **12–17× larger** than any published system's,
whose feasibility is destroyed by an **unbounded** refinement, trained with a **hard argmin** label,
and evaluated by a metric that **cannot see** the failures that matter. Every strong published
system differs on **all five** axes. Bar A changed the objective and refuted; §3.2 says the
objective was never the biggest term.

### R1 ⭐ — give the scorer a per-candidate RULE VERDICT built from `obstacle.offline` agent tracks

**RECOMMEND** (§0.4 tier: ≥ 2 papers DEMONSTRATE it, and we can build the input).

| | |
|---|---|
| **what the score reads** | per candidate: **at-fault collision (NC)**, **time-to-collision (TTC)**, **comfort (accel/jerk/yaw-rate)**, **along-track progress proxy (EP)** — each a deterministic function of *(candidate, scene)* |
| **published evidence** | Hydra-MDP++ T.3/T.4 **85.0 → 86.5 PDMS, 76.8 → 80.6 EPDMS**, matched vocabulary, target-only change · WoTE T.3 **81.0 → 83.2 → 85.6** · VADv2 `L_conflict` (a rule label over agent futures + boundary) · GoalFlow M₁→M₂ **88.5 → 89.4** from a live DAC term · PDM/PLUTO/Diffusion Planner all score exactly these terms |
| **why it fixes OUR failure** | the label stops being aleatoric. **In-sample unfittability (0.4907) is the signature of an irreducible target**, and a rule verdict is reducible by construction |
| **implementation delta** | `conf_head: Linear(512,1)` → **`Linear(512,4)` + 4 sigmoids**, trained with **BCE** against offline-minted labels; combine at inference as Hydra-MDP does: `w₁·log S_im + w₂·log S_NC + w₃·log S_TTC + w₄·log(...)`, **weights by grid search**. Keep `sel_gate`. **~2 k new parameters.** |
| **cost** | label minting is CPU-only; the fan is already cached |

⭐ **The cheapness argument, published:** PLUTO Table VII measures a **learned** agent-future
predictor against a **constant-velocity** one *for scoring purposes*: **92.82 → 93.57, i.e. +0.75
only.** `PUBLISHED (cited) — DEMONSTRATED.` **We do not need a trajectory-prediction model.** A
constant-velocity projection of `obstacle.offline`'s existing 3D tracks is within 0.75 points of the
learned ceiling — and for *offline label minting* we do not even need that, because **the tracks
ARE the future** (log-replay, exactly as NAVSIM's NC/TTC teachers use them).

### R2 ⭐ — SHRINK the fan before scoring it

**RECOMMEND.** `PUBLISHED (cited) — DEMONSTRATED, five independent sources.` LLM-Assist
**92.51 → 77.78** going 15 → 8,505 · DiffusionDrive saturates at **20** · PDM ships **15** · PLUTO
top-K **20** · Slow Brain plateaus at **K ≈ 18–24**.
**Implementation delta:** `V4Config.n_anchors 256 → 16–32`, or keep 256 and hard-filter before
`select()`. ⚠️ **Honest caveat:** our REF-C v1.2 already swept top-K and found **K = 8–32 a flat
plateau** (MEASURED) — so this is *not* predicted to be a win on its own. It is recommended because
it is **a precondition for R1 and R4**: it removes the tail that Goodharts any approximate scorer.

### R3 ⭐ — CLIP the fan kinematically (the pre-registered zero-GPU follow-up, now with published backing)

**RECOMMEND.** `PUBLISHED (cited — CoverNet 1911.10298) — DEMONSTRATED and ABLATED`: dynamic,
current-state-conditioned set construction moves minADE₅ **2.62 → 2.02** *and* reaches the same
coverage with **~half** the trajectories. PRIME gives the explicit current-speed-anchored band
`v ∈ [max(0, ṡ₀ − δ⁻T), min(ṡ_max, ṡ₀ + δ⁺T)]`.
**Implementation delta: ~5 lines, and the logic already exists in our own head.**
`FlagshipV15Head.select` already computes a reachable-speed clamp for the *goal*
(`reach = sel_accel_max * horizons[-1] * 0.1`). **Apply the identical clamp to the CANDIDATES:**
mask any candidate whose implied terminal speed leaves `[v0 − reach, v0 + reach]`. `MEASURED (ours —
source read, flagship_v15.py:440-450).`

### R4 — score the CONTROLLER'S OUTCOME, not the intended waypoints (this is where our imagination belongs)

**CONDITIONAL — sequence it after R1 and R3.** The single sharpest sentence in the nuPlan
literature, verbatim from PDM §4:
> *"PDM-Hybrid compensates for this by evaluating proposals based on the **expected controller
> outcome**, causing it to match/outperform log replay in closed-loop evaluation."*

`PUBLISHED (cited — PDM arXiv 2306.07962) — DEMONSTRATED: Val14 CLS-R, PDM-Hybrid 92 vs Log Replay
(the actual human trajectory) 80.` PLUTO independently adopts the identical trick.
**This is exactly what our world model is for, and it is what WoTE does** — roll the frozen
predictor under **each candidate's** implied actions and score the *rolled* state.
⚠️ **But sequence matters:** we measured that the WM obediently simulates absurd plans. R3 (feasible
candidates) and R1 (a verdict, not a consistency score) are **preconditions**, not companions.
**Implementation delta:** `imagine_probes` currently rolls a *shared* probe vocabulary; it must
become **candidate-conditioned**. That is the one genuinely non-trivial change here.

### R5 — soft distance-weighted labels — **DEMOTED, because we already measured it**

`PUBLISHED (cited — VADv2 L_distribution; DriveSuprim self-distillation, δ_m = 0.15)` says hard
labels are the anti-pattern. **We agree and we have already implemented it twice, and it did not
rescue us:** REF-C v1.2's `soft` objective (+2.9 %, not significant) and Bar A's cost-sensitive
regret loss (**−4.20 %, wrong side of zero**). `MEASURED (ours).`
> **Report this as a resolved conflict, not an open recommendation.** The literature is right that
> hard argmin is bad *and* our measurement is right that fixing it is not sufficient. Both hold: the
> label's **shape** was never the binding constraint — its **semantics** were.

### R6 — map-derived terms (DAC, DDC, LK, TL) — **REJECT for now, CONDITIONAL later**

No map exists in PhysicalAI-AV (settled, five probes). These are the **two most heavily weighted
multiplier terms** in PDMS, so this is a real ceiling, not a nicety. **CONDITIONAL** on the
DLR OpenDRIVE / Overture lane-graph work already in the program (commit `12d6b8a`).

### ⛔ What NOT to do — each backed by two independent measurements

| do not | why |
|---|---|
| **add an inference-time rule cost over the fan** | PARA-Drive: removing UniAD's rule optimiser improves **both** collision (0.40 → 0.16) **and** L2 (0.83 → 0.74) · **our own REF-C v1.0**: hand-written cost re-rank recovers **0.0 %**, pure cost **−171 %** |
| **change the ranking objective again** | Bar A **−4.20 %** · REF-C v1.2 across **47 arms**, +2.9 % ns · and §3.2 says inputs/objective are the small terms |
| **add scorer input tensors and expect a win** | VADv2 T.3: removing **image tokens** costs 0.001 m; removing the **supervision** costs 1.33 m |
| **grow the fan** | LLM-Assist **92.51 → 77.78** |
| **adjudicate any of this on `ade_0_2s` alone** | PARA-Drive: L2 and collision are **blind to lane compliance**; GT trajectories themselves "collide" at 0.384 % under a sloppy metric; and §3.4's ego-status critique targets exactly this metric family |

### 6.1 The direct rule-vs-learned head-to-head — the answer is HYBRID, and they fail oppositely

`PUBLISHED (cited — PLUTO arXiv 2404.14327, Table VI) — DEMONSTRATED.` Same generator, only the
selection rule `π = π_rule + α·π₀` changes:

| selection | CLS-NR | comfort | progress |
|---|---:|---:|---:|
| **rule only** (α = 0) | 90.64 | 80.32 | **98.43** |
| **α = 0.3** (their default) | **93.57** | 93.17 | 93.32 |
| **learned only** | 91.66 | **96.39** | 91.30 |

> **Learned-only slightly beats rule-only; the MIX beats both by ~2 points, and the two make
> OPPOSITE errors** — rule-only maximises progress and destroys comfort; learned-only is
> comfortable and under-progresses. **This is the single most direct answer to the brief's
> pre-registered question, and it says: hybrid, with the rule weight around 0.3.**

**The scorer-vs-generator attribution the brief asked for**, four controlled experiments, nuPlan
Val14, same generator with and without a rule scorer:

| generator (fixed) | Δ CLS-NR | Δ CLS-R | source |
|---|---:|---:|---|
| PLUTO | +4.17 | **+12.05** | PLUTO T.II |
| **PlanTF** (third-party generator, PLUTO's scorer) | +4.66 | **+11.01** | PLUTO T.II |
| Diffusion Planner | +4.39 | **+10.10** | Diffusion Planner T.1 |
| GameFormer | +66.62 | +71.09 | Diffusion Planner T.1 ⚠️ third-party reproduction |

**⇒ A rule scorer is worth ~+4 CLS-NR / ~+10–12 CLS-R across four different generators.** And PDM's
own decomposition: IDM (1 trajectory, no scorer) **77** → PDM-Closed (15 proposals + sim + rule
score) **92** CLS-R.

**The one published counter-example, stated fairly.** DTPP holds generator *and* prediction model
fixed and swaps hand-crafted → learned cost: **CL-NR 0.7388 → 0.8964 (+0.1576)**. ⚠️ Three
load-bearing caveats: (a) **not Val14** — 200 scenarios from the nuPlan test subset; (b) **PDM still
beats DTPP on that same benchmark** (0.9061 NR); (c) DTPP's "learned" cost still contains a
**hand-crafted collision RBF** and reads **ego-conditioned agent futures** — *it reads more, it does
not merely learn more.*

**And the sobering one.** PDM Table 4a: letting the **learned** module own the first 2 s costs
**34 CLS-R points** (58 vs 92). **Our entire planner is the first 2 s.** `PUBLISHED (cited) —
DEMONSTRATED.` I report it because it is uncomfortable and load-bearing, not because it is
actionable today.

---

## 7. THE CHEAPEST EXPERIMENT ON OUR CACHED FAN — a 4-step ladder, pre-registered

The caches `/workspace/_bara/cache_{produced,oracle}_stride1.pt` (4.07 GiB each, 6,844 windows)
already exist and `bar_a_selector.py` has a **`cache_fidelity` self-test that proves any rebuild is
byte-faithful** (MEASURED, `BAR_A.md` §2). **Every step below reuses Bar A's harness, folds and
estimator, so results are directly comparable to `raw/bar_a_produced.json`.**

**Estimator, named in advance for all four:** paired episode-cluster bootstrap (`taniteval/ci.py`,
B = 2000, unit = episode), 5-fold episode-disjoint cross-fit, **never `overlapping_holdout_se`**.
**Report the lateral/longitudinal split** — Bar A §4 showed an undecomposed delta hides the
mechanism. Class every arm's `full_set` mean, not `heldout`.

| step | cost | what it decides | **pre-registered kill** |
|---|---|---|---|
| **T0 — span audit** | **ZERO GPU, minutes** | is the 181 km/h in the **anchors** or manufactured by the **offset head**? Compare `span(anchors)` (a 42,550-byte file) with `span(x_in + offset)` (cached). | — pure measurement; both answers are informative and they imply *different* fixes |
| **T1 — kinematic clip (R3)** | **ZERO GPU, ~1 h CPU** | re-run `argmax` over the cached fan with candidates outside `[v0 − reach, v0 + reach]` masked. **No training, no new labels.** | if `ade_0_2s` does not improve **and** `miss_at_2m` does not improve, the fan's tail is not what is hurting the pick — R2/R3 drop to CONDITIONAL |
| **T2 ⭐ — does the rule verdict even DISCRIMINATE?** | **ZERO GPU**, + one `obstacle.offline` ingest for the 40 val episodes | mint per-candidate NC / TTC / comfort / progress labels over the cached fan. **Then, before training anything, measure the label's variance and its rank correlation with `fan_err`.** | ⛔ **THE HONEST FALSIFIER: if the corpus is ~74 % straight cruise, the collision label may be near-constant (e.g. > 95 % of candidates collision-free), in which case rule distillation has NO SIGNAL here and R1 dies for the price of a CPU job.** Registered kill: **if the NC/TTC labels have < 5 % positive rate AND their Spearman ρ with `fan_err` is < 0.15, R1 is REFUTED on this corpus.** |
| **T3 — the 4-head BCE rescorer (R1)** | **~13 GPU-min** (capture already paid) | only if T2 survives. `conf_head → Linear(512,4)`, BCE on the T2 labels, Hydra-MDP grid-searched inference weights, Bar A's exact 5-fold protocol + CE_CONTROL arm | pre-register the same three-outcome structure Bar A used, against the **rule score**, not against `ade_0_2s` alone |

> ### ⚠️ T2 IS THE POINT OF THIS LADDER, AND IT MUST RUN BEFORE T3.
> Bar A's whole lesson is that we spent GPU-hours on a lever whose existence was assumed. **T2 tests
> whether the lever exists at all, at zero GPU, and it has a real chance of failing** — our corpus
> is not nuPlan, it has no map, and its collision rate may be too low to supervise anything. **I do
> not predict the outcome and I refuse to estimate its value** (the §6.2 discipline Bar A set).

**⚠️ A measurement prerequisite that is easy to skip and would invalidate T3.** If we adopt rule
labels we must also adopt a **rule-based evaluation metric**, or we will optimise something
`ade_0_2s` structurally cannot see — PARA-Drive **DEMONSTRATED** that L2 and collision are blind to
lane compliance, and NAVSIM exists precisely because open-loop L2 is fooled (Ego-Status MLP: L2
**0.5568**, tying a full perception stack, but **64.0 EPDMS**, losing to a 2022 model by 12.7).
**⇒ ESCALATION: a PDM-style composite score for our corpus is a prerequisite for R1, not a
follow-up.** NC/TTC/comfort/progress are buildable (§2.3); DAC/DDC/LK/TL are not.

### 7.1 One free idea worth a pre-registered line, because it is counter-intuitive

`PUBLISHED (cited — "Slow Brain, Fast Planner" arXiv 2606.20458) — DEMONSTRATED.` On 64
kinematically-feasible candidates (selected 1.64 m ADE vs oracle 0.39 m — **4.2×**, our shape
again), **hiding the planner's own confidence scores from the selector IMPROVED selection**:
*"when scores are visible, the [selector] tends to defer to the planner's ranking rather than apply
its own judgement."* ⚠️ **This directly contradicts our REF-C v1.2 design**, which feeds `base_logit`
in as a feature and residualises on it (MEASURED: v1.2 found the frozen conf head is "a strong
incumbent to be RESIDUALLY corrected"). **It is a one-flag ablation in T3
(`RescorerConfig.use_q` / `normalize_base`) and costs nothing to include.** ⚠️ Note the source is a
sidewalk-robot VLM selector, not a driving planner — **class it HYPOTHESIS, worth a flag, not a
redesign.**
