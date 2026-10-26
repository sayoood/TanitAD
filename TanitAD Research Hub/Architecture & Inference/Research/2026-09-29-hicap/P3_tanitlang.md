# P3 — TanitLang, the PI's own language-native / VLA design line, read for HiCAP

**Stream P3, 2026-09-29.** Scope: what TanitLang is, where a frozen-VLM + two-head + InfoNCE design (HiCAP; ancestor in-repo: REF-F, `Project Steering/PREREG_REFF_CONTRASTIVE_SELECTOR.md`) overlaps or collides with it, what the PI left open, and a terminology map.
**Evidence classes:** `PI-QUOTE` (verbatim PI words, as quoted inside the named file) · `DESIGN` (agent-authored design text, not a result) · `MEASURED-in-doc` (a number the doc reports from its own run; **not re-verified here**) · `PUBLISHED-in-doc` · `DERIVED` (my arithmetic/algebra, shown) · `UNVERIFIED` · `HYPOTHESIS`.
⚠️ Every "MEASURED-in-doc" number below is **INHERITED** by this stream. None is quotable as a registry fact; the registry (`Project Steering/MODEL_REGISTRY.md`) wins on any conflict.

---

## 0. Sources, coverage, provenance (read this before quoting anything)

All TanitLang documents live in the PI's **Google Drive**, none in this repository. I searched every git ref (`git log --all`, `git ls-tree origin/main`) for `TANITLANG`, `cot_influence`, `vla-design`: **zero hits**. `stack/tanitad/instruments/` here holds only `checks.py`, `numerics.py`.

| file (Drive title) | Drive id | size | modified | read |
|---|---|---:|---|---|
| `TANITLANG_DESIGN_V2.md` | 1mXURdDuurlUCLxeWaVlGPcZYRr50zgu5 | 18,184 | 2026-09-05 | full |
| `TANITLANG_DESIGN_V3.md` | 15hMAGDSEX3Zd7IHgkPc03zX0PafhHuuO | 17,301 | 2026-09-05 | full |
| `TANITLANG_DESIGN_V5.md` | 16UZ4Ac1vo__8Uj5lKaN0xuX3SymA3hgy | 12,745 | 2026-09-05 | full |
| `TANITLANG_DESIGN_V6.md` | 1---ne7-1MuNRsxEW0e1uQR-G-_YgzLS- | 7,785 | 2026-09-05 | full |
| `DIALOGUE_02 / 03 / 05 / 06` | 1Idvj… / 1r-iz… / 1nlxc… / 1Mj0I… | 17,743 / 14,114 / 12,745 / 7,785 | 2026-09-05 | full |
| `DIALOGUE_01_MISSION_PLAN_CROSSCHECK.md` | 1a8gekeI_1qESctsYhMnJVgYJd1M3n09N | 8,600 | 2026-09-05 | full |
| **`DIALOGUE_07_TANITLANG_ENERGY_BRIDGE.md`** | 1Sh01patV0MNcGRv2wl3eFcS3edE58Ep9 | 65,794 | 2026-09-05 (body dated 2026-09-06) | **full, 1,075 lines** |
| `BRIEF_TANITLANG_FOR_TRAINING_FLYWHEEL.md` | 1d-Eyk-QoFolR28FvlZhYf_RapE7f3AVe | 5,742 | 2026-09-05 | full |
| `HANDOVER_2026-09-06.md` (dialogue folder) | 1pAKg5Miv4Mca1SQNClNOJTuIDnmCv-c5 | 6,219 | 2026-09-06 | full |
| `RESULT_WPP_FRONTIER.md` (β frontier) | 1hQvyw0OH7W6fw20vgB7MFC3BGvShWxd8 | 11,178 | 2026-09-06 | full |
| `LIT_COT_INFLUENCE_CONSISTENCY.md` | 1oXg92onPpnbQK--_UuVJYVQyWj6fsoXt | 60,219 | 2026-09-05 | **partial**: lines 14-73, 357-407, 461-550 of 695 |
| `PI_DECISION_QUEUE.md`, `PI_VIDEO_REVIEW_2026-09-06.md` | 1Johkx0P… / 1TtG6bXR… | 30,649 / 15,352 | 2026-09-15 / 09-06 | full |

**Provenance facts found while reading (each changes how the corpus may be cited):**
1. **The DIALOGUE_02/03/05/06 files are the same documents as TANITLANG_DESIGN_V2/V3/V5/V6** (identical size for V5/V6; V2 = D02 + a "SUPERSEDED" banner; V3 = D03 + a 2026-09-05 correction appended). They are five "dialogue" files only in the folder sense. Cite the DESIGN_Vn name.
2. **There is no `TANITLANG_DESIGN_V4`, no `DIALOGUE_04`, no `DESIGN_V7`, and no README in the `vla-design-dialogue` folder** (Drive title searches, and a listing of the folder `1R8tGfE0…`, which holds DIALOGUE_01/02/03/05/06/07, HANDOVER, LIT_*, AUDIT_*, RESULT_WP*, DATA_AGENTJOIN_*, plus `raw/` and `code/`). The numbering skips 4 for both series. UNVERIFIED whether they exist elsewhere.
3. **Latest TanitLang text is 2026-09-06.** Nothing later found (Drive full-text search for `TanitLang`, `Energy Bridge`, `HiCAP`, `CLM` after that date returned no design doc). The programme has moved on since (REF-F prereg is 2026-09-29) and I cannot see whether TanitLang was amended on the branch `agent/arch-inf-20260803`, which is **not** a ref of this repo.
4. ⛔ **Stranding (rule 3 of the operating standard).** HANDOVER: *"67 files staged and blob-verified, nothing committed, nothing pushed, branch `agent/arch-inf-20260803`"* and *"`stack/tanitad/instruments/cot_influence.py` + 29 green tests"*. Neither the package nor the instrument is in any ref here. Drive holds copies of the prose, not of `cot_influence.py`. **Escalated in the report headline.**
5. **NOT read:** `REFCV5_VLA_EXTENSION_PLAN.md` (the superseded v1), `AUDIT_REFCV5_SURFACE.md`, the other `RESULT_WP*` files, `Mission Plan.md` (H-numbers are taken from DIALOGUE_01's verbatim quotes), `RETRACTION_LOG.md` (TRAIN-C13..C17), `GOALS_AND_CLAIMS.md` (`D-TLANG-*` rows; a search for them returned nothing), `MODEL_REGISTRY.md` for TanitLang numbers.
6. **"Training flywheel" is not a method.** `TanitAD_TrainingFlyWheel` is the *name of the agent stream/session* that wrote these documents (a sibling is "DataFlyWheel"); `BRIEF_TANITLANG_FOR_TRAINING_FLYWHEEL.md` is a briefing *to that session*. The documents contain no flywheel/self-improvement loop. What plays the role of a training plan is the staged build A→D (§1.7).

---

## 1. What TanitLang is (priority 1)

### 1.1 Definitive statement, and which version that is

* **Design of record = `DIALOGUE_07_TANITLANG_ENERGY_BRIDGE.md`** (2026-09-06). Its header: *"Supersedes: `DIALOGUE_02` (v2), `DIALOGUE_03` (v3), `DIALOGUE_05` (reasoner internals), `DIALOGUE_06` (relational semantics)… this file is the one to build from."* HANDOVER: *"Design of record: `DIALOGUE_07_TANITLANG_ENERGY_BRIDGE.md` (read its STATE OF THE DESIGN header first; body + 7 amendments are the working)."*
* **V6 is the last numbered `TANITLANG_DESIGN_V*` file, but it is NOT the definitive statement**: it fixes only the shape of the latent (relational graph) and is partly refuted by later measurement (§1.5). The body of D07 is also partly wrong (its own Amendments 1-3 and the *STATE OF THE DESIGN* header withdraw claims). **Read order: D07 header block → D07 Amendments → D07 body.**

**Thesis in one paragraph (DESIGN; D07 §0-§3, §6).** TanitLang is not a language model bolted beside REF-C; it is a **small reasoning module that writes into REF-C's existing anchor-selection mechanism**, with language as an optional, off-the-control-path renderer and regulariser. REF-C enumerates a fan of 128 anchor trajectories, refines them by 2-step truncated diffusion and scores them (`S0`, `τ0`). A recursive, weight-shared, TRM-style reasoner produces a reasoning state `z_T` (whose successive states `z1…zT` *are* the chain of thought). The **only** channel from `z` into behaviour is a re-ranking of the fan by a learned compatibility energy: `S1(k) = S0(k) − β·E_φ(z, τ_k)`. The same `E_φ` read the other way scores whether the emitted reasoning *prefers the trajectory finally selected* — so **influence and consistency are two readings of one function**, and neither failure (caption = zero influence; confabulation = high energy on the selected anchor) can hide behind the other. Free text never reaches the planner; the planner consumes only released, masked, 8-wide tactical/strategic vocabularies. The stated novelty (after Amendment 3 §A5) is **the measurement** (influence and consistency as two readings of one distribution over the same anchor fan, no judge/parser/LLM in the loop), **not the interface** (AnchorVLA `2607.03182` and LCDrive `2512.10226` already publish anchors-as-interface / CoT-in-action-vocabulary).

### 1.2 Version lineage (each version corrected the previous; the corrections are the content)

| version | date | what it said | what killed or narrowed it | source |
|---|---|---|---|---|
| v1 `REFCV5_VLA_EXTENSION_PLAN` | 09-05 | trunk → projector → frozen SmolLM2-360M+LoRA → write-back heads | PI: *"your proposal is mainly to take the languge model … im not recognizing the refc design in the plan"*; LM = 368 M vs strategic brain 67,587 params (0.0631 %) = 5,445×; violates H12 | BRIEF §1-§2, D01 §1-§2 |
| v2 | 09-05 | `FanReasoner` (~8 M, K=16 latent tokens, 2 steps) over the fan; scorer commits; LM renders ≤1 Hz | mixed REF-C with the v6/v7 world-model line; invented an interface REF-C already had | V2 banner, V3 §0 |
| v3 | 09-05 | language as **guidance**: write priors into REF-C's zero-init ports; two-pass CFG loop-breaker `S = S0 + w·(S1−S0)` | claimed a "language port" that is Lane-Anchored Navigation; asserted a refcv5 gap it had already closed | V3 §1-§3 + correction; D07 A1 C2 |
| v5 | 09-05 | TRM-style `z`: `z ← z + Block_think(x,y,z)`, `y ← y + Block_answer(y,z)`; four decodes of one `z`; 3 directions (free / LM-driven / combined) | `z` flat → cannot hold *relations* | V5 §1-§4 |
| v6 | 09-05 | `z` = per-entity nodes + typed edges; message-passing think-step; semantics distilled from a frozen strong backbone, train-time only | **WP-F: hand-crafted relational edges add nothing** (shuffled-edge 0.4815 > real-edge 0.4704) | V6 §1-§4; D07 §4, §7(b) |
| **D07 Energy Bridge** | 09-06 | the consolidated design above, + 7 amendments | PI objected to ~2 M trainable: budget now ≈25 M (stage A) / ≈115 M (stage B) | D07 §3, A1-A7 |

### 1.3 "The language": what it is, concretely (there is no grammar and no text interface on the control path)

The docs never define a formal grammar or DSL. "Language" is **four different things**, and only the first touches the planner:

| layer | what it is | reaches planner? | status |
|---|---|---|---|
| **A. Released vocab tokens** | lat `TACTICAL_LAT_ACTIONS_V7` (8) → `lat_prior`→`lat_to_anchor`; lon `TACTICAL_LON_ACTIONS_V7` (8) → `lon_to_anchor`; `STRATEGIC_ACTION_TOKENS_V7` (7); `STRATEGIC_GOAL_TOKENS_V7` (8); manoeuvre set (5) → `maneuver_to_anchor`. Each port is `Linear(vocab → anchors)`, **zero-init**, **log-space**, summed into selection logits. `NOT_YET_EXTRACTABLE` classes masked. (V3 §2.3) | **yes**, as anchor log-priors | DESIGN; ports MEASURED non-inert (D07 A5.1) |
| **B. Shared codebook** | special tokens `<TAC:follow_lead>`, `<TAC:yield>`, `<STR:left_at_junction>` whose embedding rows are *tied* to the planner's `maneuver_emb` / goal-embedding rows: *"Emitting the token *is* conditioning the planner. There is no translation layer to drift"* (D07 §5) | yes | **HYPOTHESIS**; must be built "from nothing" (D07 A1 C2: *no tokenizer, embedding table or text head exist anywhere in the model*) |
| **C. Latent reasoning trace** | `z1…zT` (TRM `z`) with heads `cot` (the 11 Alpamayo `cot_tokens` fields: `yield_, merge, overtake, gap, lane_change, evade_obj, oncoming, traffic_light, exit_side, speed_limit, evidence`), `referent` (pointer into agent tokens ∪ NULL), `score`, `goal` (V5 §2.2) | via `E_φ` only | DESIGN |
| **D. Free text** | Alpamayo-style causal sentence rendered by an LM decoder over `z`; optional text *input* port for nav command / question, zero-init | **never** — *"A hallucinated sentence is inert by construction"* (V3 §2.3) | DESIGN; text input not built (*"we have no question corpus"*, V5 §2.4) |

MEASURED-in-doc bounds on layer D: the 11 fields cover **27.2 %** of Alpamayo text variance (shuffled control 0.8 %), so a field-driven renderer *"would be reciting a template"* and `z` must feed the renderer directly (D07 §6, WP-A). The corpus has no `motorcycle`/`bicycle` class (both = `rider`), `occ` constant-zero, no illumination field (D07 §6, WP-G).

### 1.4 Role for REF-C ("REF-C-native")

PI critique fixed the role (BRIEF §1, verbatim): *"the idea was to extend the refc design (diffusion drive) with vision language and grounded COT in combination with our 4B multihierrahcacy archotecture"*. REF-C-native therefore means: **reuse the fan, the scorer and the zero-init prior ports; add nothing to the interface** (V3 §1, "Nothing new is invented at the interface"). Corrected position (V3 correction): *TanitLang is a **producer** for WP-7's selector, reading WP-3's candidate-conditional features and WP-6's agent tokens.* Constraints on what the planner keeps: the scorer still selects; the fan is still the planner; the hierarchy does not grow (V2 §3.2, V3 §2). **Influence mode (LM goals steering the planner) is explicitly out of the plan and is a separate PI decision** (V2 §7).
Corrections that bind any REF-C-native claim (D07 Amendment 1, from `AUDIT_REFCV5_SURFACE.md`; MEASURED-in-doc): refcv5 grounding = **agent tokens only** (`agent_tok [B,100,384]`, gated cross-attention, two flags both default False; **no BEV/LiDAR/map/lane token**); `d_model` = **384**; `lan_*` = **Lane-Anchored Navigation, not language**; `maneuver_to_anchor` exists on the base checkpoint but is **absent on refcv5 arms** (`factored_maneuver` builds `lat/lon_to_anchor`). refcv5's model seams were **not in `HEAD`** as of 2026-09-06 (C6; status today UNVERIFIED).

### 1.5 The Energy Bridge (D07 §1-§2, quoted formulas)

```
E_φ(z, τ) ∈ ℝ          z = reasoning state, τ = a trajectory; low = "reasoning and trajectory agree"
INFLUENCE   S₁(k) = S₀(k) − β · E_φ(z, τ_k),   k = 1…128        (only channel into behaviour)
   full:    S₁ = S₀ − β·E_φ(z_T, ·) + Wg·g + Wc·c → τ_sel         (β, Wg, Wc all zero-init)
CONSISTENCY L_cons = E_φ(z, τ_sel) − softmin_k E_φ(z, τ_k)       ("prefer it over the other 127")
INF_sel = 1[argmax S₁ ≠ argmax S₀] · INF_geo = ‖τ_sel(S₁) − τ_sel(S₀)‖ · INF_kl = KL(softmax S₁ ‖ softmax S₀)
CON_rank = rank of τ_sel under E_φ(z,·) among 128 · CON_gap = E_φ(z,τ_sel) − min_k E_φ(z,τ_k) · CON_text
loop-break: pass 0 (ports off) → F₀, S₀; reasoner reasons against F₀; pass 1 scores consistency against τ_sel  (CFG-style: S = S₀ + w·(S₁ − S₀), V3 §3)
```
*"E must be a scorer, not a generator. A generator can always satisfy a consistency loss by copying. A scorer cannot"* (D07 §1). ⇒ **the energy is a learned scorer over a finite candidate set, i.e. the same object class as a contrastive selector** (LIT §3: *"REF-C already IS an energy model over the fan … `sel_score` … is `E(x, y)` with `y` ranging over a discrete candidate set"*; EBT `2507.02092`, IBC `2109.00137` cited).
Architecture (D07 §3, ESTIMATED): grounding fusion ~6 M → recursive core (d=512, weight-shared, depth free in params) ~13 M → halting head ~0.3 M → goal/constraint heads + shared codebook ~2 M → **energy bridge = "traj encoder + bilinear" ~4 M** → language-free core **≈25 M**; optional LM 0.3-1.0 B, *"training-time + on-demand only, never in the control path"*. Encoder 90.5 M stays frozen in stage A.

**What the docs' own numbers say about the energy (MEASURED-in-doc; T0, non-parity pilot, `refc-base-30k`, n=1,360 windows / 34 episodes; the "energy" tested is REF-C's own manoeuvre prior, not a reasoner):**
* zero-port identity exact (flip 0 / KL 0 / geo 0.0000 m); trained port flips selection in 23.2 % of windows, moves the chosen trajectory 0.2022 m (D07 A5.1). Spearman(Δ, S0) = 0.096 → *new ordering, not an echo*.
* **Headroom:** selected 0.654 m vs fan-best 0.250 m (n=120 pilot, CI of gap [+0.267, +0.570]); best anchor median rank 3/128, top-5 75.0 %, top-1 32.5 % (D07 §9). Same gap re-confirmed at 0.65 vs 0.2315 m on the 34-episode join (D07 A5.4). ADE-only, one seed, T0: *"precondition, not a result."*
* **β frontier** (`RESULT_WPP_FRONTIER.md`, WP-P/Q): `sd[S0]=17.907`, `sd_a[E]=1.457`. β=1 → 8.1 % authority, CON_rank 64.40 vs chance 64.50 (unmeasurable), ADE 0.6521; **β=1.5 → 12.2 %, CON_rank 59.06 [53.79, 64.27] separated, ADE 0.6479 (best)**; β=2 → 16.3 %, 53.66, 0.6583; β=3 → 24.4 %, 38.70, ADE 0.8168 (+0.1524 [+0.0608, +0.2547] separated worse); β=5 → ADE 1.4864; β=160 → CON_rank 1.15 at ADE 7.6221 (11.5× base). Across β∈[0.5, 2.5] ADE is *not* separable from base. **Doc's conclusion: "The energy must get BETTER, not LOUDER."** Cross-episode-prior control (real − cross-episode) −0.0254 m [−0.0425, −0.0093] separated, replicated (−0.0375 [−0.0521, −0.0236]).
* **Authority** is my derived reading of those numbers: `authority = β·sd_a[E] / sd[S0]` (DERIVED: 1.457/17.907 = 0.0814 = the doc's 8.1 %).

### 1.6 The reasoner, semantics and interaction (D07 §4, §7; V5-V6)

* **Recursive core, not a stack** (TRM `2510.04871`, ARC-Prize HRM ablation; PUBLISHED-in-doc): the hierarchy of HRM contributes little, *the outer refinement loop carries the result*; `x`=grounded scene set, `y`=belief over the fan + goal tokens, `z`=CoT state; adaptive depth by a halting head ("fast when easy, deep when crowded"). ⚠️ TRM has **no** halting mechanism; adaptive R is *"my proposal, not a published result"* and needs a fixed-R control (V5 §3.2). HYPOTHESIS.
* **`puzzle_id` transfer risk** (V5 §3.3): TRM with a blank/random puzzle ID = zero accuracy; driving windows have no identity. Candidate analogue = a *learned, scene-derived* code. ⛔ It **may not** be the situation classifier's output in any form (binding, 2026-08-03). WP-B's "no signal" read is **void** (features identical across ranked candidates = tautology, D07 §4); the `puzzle_id` analogue is **open**.
* **Semantics** (V6 §2, D07 §7a): train-time-only distillation from a frozen strong backbone (DINOv3 ViT-L/16). MEASURED-in-doc (WP-I v3, WP-L): DINOv3 is the only one of four arms clearing its within-episode null on agent presence (`vru<20m` p 0.000; adj +0.1935 vs trunk +0.0792); not a pooling artefact (8×8 map weakest, p 0.495); far band reachable by a nonlinear read of the trunk (RBF p 0.040/0.020), **near band not** ⇒ distillation targets **near-agent semantics only**. "DINOv3 is the only arm tracking crowding" was **withdrawn** (D07 STATE header).
* **Interaction** (D07 §7b): hand-crafted edges fail (WP-F); untested = *learned* attention among the 100 agent tokens inside the recursive core, decided by an **intervention test** (perturb one agent token; the energy landscape must change more when the agent is on a collision course) = `echo_gate.ego_intervention_test` with the perturbed channel swapped ego → agent token (D07 C7).

### 1.7 Staged build and controls (the de-facto "training plan"; D07 §8, Amendment 3, consolidated table)

| stage | trains | frozen | admissible claim on completion |
|---|---|---|---|
| **A** bridge only (≈25 M) at `tac_goal_head`, β≈1.5 | `E_φ`, β, heads | all REF-C | "a reasoning state can re-rank the fan": `INF_*>0` **and** shuffled-z degradation |
| **B** joint (≈115 M) | + decoder/scorer | encoder | "the bridge improves selection": headroom closes, four families, replicate arm |
| **C** semantics | + distillation head (near band), encoder unfrozen | — | "injected semantics help" |
| **D** language | + LM decoder | — | "the CoT is faithful": `CON_text` **and** τ→z residual test |

Mandatory controls, each **shown to fire** on a deliberate-regression arm: β=0 identity (bit-exact); constant-z floor; **norm-matched semantic-null and permuted prior** (real must beat both; a shuffled or `"depicts"`-token CoT reproduced/doubled ORION's effect, `2606.12706`: 81.9 %/−0.437 m vs 81.4 %/−0.429 m vs 90.4 %/−0.846 m); shuffled-z (roll across *groups*, not index: the first control mispaired **97.5 %** same-episode and produced a retracted "89 % generic" claim); raw-pixel floor; **seed-replicate floor** (`2605.17268`: 88 % inconsistency from seed alone on our corpus; lateral 100 % vs longitudinal 62-69 %); `assert_not_echoing` vs `("ha","ha0_ext")`; **Var_a[prior]** non-degeneracy screen (PAV `2410.08146`). **`INF_kl` is the wrong axis** (REF-C's real prior lost the KL contest to random noise 0.1878 vs 0.4065): decide on `outcome_null_screen`.
Falsifiers committed in advance (D07 §10): INF_kl≈0 after stage A ⇒ caption generator; large INF but shuffled-z does not degrade CON_rank ⇒ scene-blind noise ("worse than refuted"); τ→z probe explains ~all of `z` ⇒ echo; WP-I says trunk already matches DINOv3 ⇒ §7a closed; replicate arm reproduces the effect with zero levers ⇒ rig noise.

### 1.8 Decisions the PI has made (with file + section; PI-QUOTE unless marked)

| # | decision / statement | file · section |
|---|---|---|
| P1 | **Mission:** *"extend the architecture with a ≤1B-parameter vision-language part sharing REF-C's embedding space, processing nav commands and question queries, creating system-initiated chain-of-thought grounded in scene understanding and consistent with trajectory hypotheses, able to process and emit strategic and tactical goals, 300–500 ms tact."* | BRIEF §1 |
| P2 | **H12:** *"our system must process language as additional part **and not as the core** … we will stick to the unsupervised world model as main direction, but we will extend it by a text processing part … The challenge is to find an efficient way to design a **common latent space which includes all**"* | D01 §2 (from `Mission Plan.md`) |
| P3 | **H11:** self-monitoring "assigned to the strategic part which should be generative and able to process text"; **H13:** behaviour extraction head "extracting the chosen behavior and the considered alternatives"; **H3:** world models "derive the consequences of their actions … explainability"; **H1:** brain 4 = fallback/MRC, multi-rate ("100 ms vs 500 ms vs. 1 second") | D01 §3-§5 |
| P4 | **Requirement:** the CoT must **influence behaviour** and be **consistent with the selected trajectory** and with tactical/strategic planning; loop concern: *"the reasoning must be consistent to the trajectory fan, so there is a little of loop where we need to be careful."* | D07 §0, §2 |
| P5 | *"it is not plausible for me why it's so small trainable part, in my opinion this won't work. In REF-C we had much more trainable parts."* ⇒ 25 M / 115 M staging | D07 §3 |
| P6 | Correction: *"we addressed this in refcv5"* (environment grounding exists) | D07 A1 C1 |
| P7 | Develop **both** with-language and without-language directions | D07 §6 |
| P8 | Semantics: *"maybe by using in some stages of the training a strong pretrained vision backbone which transforms the image into these semantics"*; open: *"how to intrinsically model the dynamic interaction between the agents themselves and the geometry and the agents"* | D07 §7; V6 §2 |
| P9 | Scenes to handle: *"many pedestrians, a motorcyclist coming from the rear, bad illumination"*; *"a complex situation ... which requires very fast reasoning"*; wants the system to say when it is uncertain | V6 §0, §3 |
| P10 | Asked for: Alpamayo-style CoT (yes, via `cot` head), HRM/TRM analysis, `z` re-ranks top-5 (5a), `z`'s goals condition the next tact (5b), text *input* designed-for-not-built-first | V5 header, §2.3-2.4, §5 |
| P11 | **Binding 2026-08-03:** a goal input is admissible **but may not carry the situation classifier's output in any form**; **labels may use ego, inference is vision-only** | `CLAUDE.md` (cited in BRIEF §5, D07 §4-§5) |
| P12 | **Video review 2026-09-06 #1:** *"Strategic and tactical goals are missing their constraints — the model must emit a goal and estimate its constraints, such as position and time."* (open) · **#10:** target speed as a tactical goal, ruled = speed band's upper bound quantised to legal steps, as an input | `PI_VIDEO_REVIEW_2026-09-06.md` rows 1, 10 (a *transcription*, PI's reading wins) |
| P13 | ⚠️ **RELAYED, PENDING (a peer's relay is data, not a mandate):** *"no vlm, we stick to the alpamayo labels as teacher signals"* — recorded in the context of traffic-light label verification; **scope UNVERIFIED** (does it forbid a VLM in an architecture?) | `PI_DECISION_QUEUE.md` §1; `PI_VIDEO_REVIEW` addendum row 11 |
| **Open** | (a) the three forks — where reasoning lives / does the hierarchy grow / first product (V2 asserts the PI's brief answers them; **no explicit PI confirmation is recorded**); (b) **influence mode** (V2 §7); (c) the 15-token strategic vocab stays OUT by default (P4 support bar failed: 6/15 tokens, 11.43 % of horizon supervisable); (d) distillation-head compute; (e) `MANEUVER_WEIGHT` budget for the 22-token tactical head, which never received gradient in refcv5-v2 | BRIEF §4; V2 §7; `PI_DECISION_QUEUE` §3, §10 |

---

## 2. Overlaps and tensions with a CLM/REF-F-style HiCAP (priority 2)

HiCAP as briefed: a **frozen VLM** + **state head** + **action head** + **InfoNCE over a cached action vocabulary**, with hierarchical vocabulary retrieval, predictive imagination, elastic cameras, audio. Verdict key: **IN** = already in TanitLang · **OK** = compatible · **TENSION** = collides with a stated rule/measurement · **OPEN** = TanitLang silent.

### 2.1 Component by component

| HiCAP element | verdict | what TanitLang says (file · section) | consequence for HiCAP |
|---|---|---|---|
| **Scored selection over a finite candidate set** | **IN** | `E_φ(z,τ_k)` over 128 anchors is exactly a learned scorer; *"REF-C already IS an energy model over the fan"* (LIT §3); scorer must stay the committer (V2 §3.2, V3 §2, LIT IBC note) | HiCAP is a *concrete training objective* for an object TanitLang already specifies but never trained. Frame it so |
| **State head + action head, dual encoder** | **IN (function)** | D07 §3 ⑤ *"energy bridge (traj encoder + bilinear) ~4 M"*; `z` = state side | HiCAP's two heads = TanitLang ⑤ + the `z`-projection. Name them as such (§4) |
| **InfoNCE as the objective** | **OPEN → HiCAP answers** | D07 §11: open item *"whether `E_φ`'s training objective should be contrastive, ranking, or a process reward"*; only `L_cons` exists (§2.2 below) | Headline contribution HiCAP can make to the PI's design |
| **Cached action vocabulary** | **OK, with a provenance difference** | TanitLang scores the **refined** fan `F₀` (128 anchors after 2-step diffusion; `anchor_controls [128,2]`); REF-F caches a farthest-point 64×64 = 4,096 residual vocabulary and does no denoising | TanitLang's per-anchor rationale wants **candidate-conditional** scene features (WP-3 waypoint-indexed attention, permutation-tested, V3 correction). A pure cosine dual encoder over cached embeddings is candidate-*independent*; REF-F's E-F4 (2-layer cross-attention scorer) is the matching test |
| **Hierarchical vocabulary retrieval** | **IN (as priors) / OK (as retrieval)** | Token vocab → anchor log-prior ports: lat 8 × lon 8, strategic 7/8, manoeuvre 5, all `Linear(vocab→anchors)` zero-init in log-space (V3 §2.2-2.3; V3 writes the width as 117, see §2.3): *"the shared space is Δ¹¹⁶ … needs no alignment loss"*; top-5 re-rank (V5 WP-C; refcv5 WP-7: best anchor top-5 in 75.0 %) | Coarse→fine retrieval **is** the port hierarchy re-expressed. Reuse the released vocabularies as the coarse levels rather than inventing new ones |
| ↳ **…upper levels are corpus-starved** | **TENSION** | strategic vocab: 6/15 tokens populated, 11.43 % of horizon supervisable, stays OUT by default (PI queue §3); lane change: 3 of 8 lateral actions unreachable by the emitter, text-side ceiling 105 clips ≈ 2.2 % (PI review #12); **no map/route/goal source in PhysicalAI-AV** (CLAUDE.md, settled at five probes); nav = `ego-future` oracle on 4,719/4,719 clips, 1.313 bits/clip (V2 §8; V3 §7); the 22-token tactical head **received no gradient** in refcv5-v2 (PI queue §10) | A hierarchical vocabulary whose strategic level cannot be supervised will train on nothing. Report STRATEGIC as UNAVAILABLE with reason and n, as REF-F does. Do not credit any result to the 22-token vocabulary |
| ↳ **…level-k selection feeding level-(k+1)** | **TENSION (binding)** | goal input may not carry the situation classifier's output *"in any form — class posterior, argmax, embedding, or any feature derived from them"*; state what a goal is computed from; if a shared trunk feeds both goal and situation paths, justify (CLAUDE.md 2026-08-03; D07 §5) | A coarse-level retrieval that is *also* a situation class, then conditions the fine level, is the forbidden loop (attribution dies; nav-echo in new costume). Run the admissibility question: *could this have been computed from the situation classifier's output?* |
| **Frozen VLM as the perceptual backbone** | **TENSION (3 sources) + 1 supporting datum** | (a) **H12** *"language … not as the core"*; DIALOGUE_01 D1 called an LM holding 98.6 % of parameters a **design error**; (b) **trunk as the ONLY vision encoder** is the design element D01 §7 says to *defend* ("a second encoder destroys the measurement"); V6 §2: strong backbone is *"teacher at train time, absent at inference … no second backbone on the tact"*; (c) PUBLISHED-in-doc `2607.10172` (banked): *freezing the VLM or LoRA-ing only the vision encoder significantly degrades performance*; recommends LoRA r=32 + full vision-encoder fine-tune — D07 §3 says the v3 configuration (frozen LM + LoRA, ~2 M trainable) *"was the one measured to fail"*; stage B unfreezes the encoder. **Supporting:** DINOv3, frozen, is the only one of four arms clearing its null on near/far agent presence (D07 §7a, WP-I) | Compatible **only if** the frozen VLM is the **single** encoder (preserves D01 §7's argument) **and** its text/LM side is off the control path; then the fight is `2607.10172` vs WP-I, and REF-F's E-F0 pre-gate + H-F7 arms are the discriminating experiments. Alpamayo-R1's 71 %-CoT-decode latency trap applies only to decoding; **no Thor latency exists for any candidate VLM tower** (REF-F prior-art headline 5) |
| **Predictive imagination** | **OK — not REF-C-native** | REF-C *is not a world model*; the world-model line is v6/v7, *"a different architecture"* (V3 §0); H3 (PI): world models *"derive the consequences of their actions"*; D01 D3 lists *"no imagined consequences"* as a design error; REF-C's fan = *"an enumerable imagination, free at eval"* (D01 §3); **tier rule** T0 = diagnostic, never "driving performance"; T1 = self-action open loop, still open loop (D07 §8); PUBLISHED-in-doc DINO-WM: prediction fidelity +4 % while control moved 11.5×; fan *diverges* past the trained step budget (sel-ADE 0.654 → 1.529 m at 8 steps, V2 §3.1) | Stamp every imagination number with its tier. Do not import the v7 action-insensitivity finding (`H-ARCH-ACTINS`, ~1 %) into a REF-C-family arm — V3 §0 withdrew exactly that mistake. If imagination iterates, its step count is a trained property, not a free knob |
| **Elastic cameras** | **OPEN** | not addressed. Adjacent measured-in-doc failures of the same class: label frame 120° vs encoder 51.4° (48.5 % of forward cars, 46.7 % of forward people outside the view; HANDOVER §4 #1); WP-A's azimuth address was **mirrored** (PI queue §9, `R-2026-09-08-wpa-mirror`); 256×640 cylindrical frames (V6 WP-G) | Any variable-camera design needs a stamped frame/azimuth convention and a mirror/derangement test before a result is read; the programme has lost time to a wrong frame and a mirrored address |
| **Audio** | **OPEN, and TENSION with the letter of a binding ruling** | *"for inference only vision"*; *"labels may use ego, inference is vision-only"* (CLAUDE.md 2026-08-03; D07 §6 *"keeps the vision-only inference ruling intact"*). Precedent for optional non-vision input: zero-init text-token slot, absent by default (V5 §2.4); nav commands are already a non-vision input in the PI's mission (P1) | The ruling's rationale is leakage from label sources, and audio is not one, but its **letter** says vision only. **Needs an explicit PI ruling before audio enters an inference path**; build it as a zero-init, gated, absent-by-default slot so `audio=0` reproduces the vision-only arm bit-for-bit |
| **Ego kinematics in the state head** (REF-F prereg §1) | **OK, with a scope statement** | the planner already conditions on `v0`; dropping `v0` asserts `keep=0`, the X15 zero-collision (PI queue §7 update). The vision-only ruling was applied to the goal/situation path (BRIEF §5: `plan()` builds the goal input with `ego=None`) | State per arm which path is ego-free. Any **goal or strategic-level** head in HiCAP must be ego-free at inference; REF-F's NC4 ego-only arm is the copycat check |
| **Authority of the contrastive score over the base planner** | **TENSION (measured) — and the experiment** | usable window β∈[1,2] = 8-16 % authority; β=5 doubles ADE; β=160 gives CON_rank 1.15 at 11.5× ADE (WPP). But that energy was REF-C's own 5-class manoeuvre prior, *"not a reasoner"* | A standalone HiCAP selector has authority 100 %; TanitLang's evidence says a **poorly ranking** energy given authority destroys driving, and *"the energy must RANK WELL, not merely be loud"*. Both H-F1 (standalone) and H-F3 (verifier/re-rank) in REF-F are therefore the right pair |

### 2.2 ⭐ Energy-bridge ↔ contrastive-score correspondence (formulas)

TanitLang (D07 §1): `S₁(k) = S₀(k) − β·E_φ(z, τ_k)` and `L_cons = E_φ(z, τ_sel) − softmin_k E_φ(z, τ_k)`.
REF-F (`PREREG_REFF_CONTRASTIVE_SELECTOR.md` §1, selection rule): `score_k = e^τ⟨s, a_k⟩ + β̂·log p̂(a_k)`, loss `L_nce` in-batch bidirectional, `τ₀ = log(1/0.07)`, `e^τ ≤ 100`.

Set `E_φ(z, τ_k) := −e^τ⟨ŝ(z), â(τ_k)⟩` (unit-norm heads). **DERIVED**:
1. **Influence form.** `S₁ = S₀ + β·e^τ⟨ŝ, â_k⟩`. TanitLang's `β` and HiCAP's `e^τ` are **one degree of freedom** (logit scale); only the product is identifiable. Authority (§1.5) becomes `β·e^τ·sd_k⟨ŝ,â_k⟩ / sd[S₀]`.
2. **Standalone form.** Drop `S₀`: `softmax_k(−E_φ)` *is* HiCAP's vocabulary softmax. TanitLang's "replace rather than nudge" branch (D07 A5.3) = HiCAP standalone.
3. **Consistency loss = InfoNCE, under one reading.** If `softmin_k E := −log Σ_k exp(−E_k)` (smooth minimum), then `L_cons = E_sel + log Σ_k exp(−E_k) = −log softmax_k(−E)_sel`, i.e. **the InfoNCE cross-entropy over the candidate set with `τ_sel` as the positive.** D07 **does not define `softmin`** (UNVERIFIED which is meant); if it is the softmax-weighted mean `Σ_k softmax(−E)_k·E_k`, `L_cons` is a margin-to-mean, not a cross-entropy. HiCAP's paper should state which.
4. **Different positives — do not merge them.** TanitLang's positive is the **selected** anchor `τ_sel` (a model output; kept non-tautological only by the two-pass loop-break of §1.5). HiCAP/REF-F's positive is the **ground-truth-nearest** vocabulary entry (mask ε = 0.25 m ADE@2s; soft targets σ = 0.5 m). Keep them as two terms: `L_nce` (external supervision) trains the score; `L_cons` (self-selection) is an audit/regulariser. A3 (D07): never supervise a *rationale* from the logged future (see §3).
5. **Where the prior lives.** REF-F's H-F4 / prior-art headline 3: in-batch InfoNCE ranks by **PMI** (evidence against the action prior), not likelihood, so a `log p̂(a)` correction may be mandatory. TanitLang's additive form already separates the two: `S₀` carries the base plan/prior, `−βE` carries the contrastive evidence — a **product-of-experts** reading of `S₀ − βE` (HYPOTHESIS; not stated in any TanitLang file). ⚠️ **Symbol collision:** TanitLang `β` = energy authority; REF-F `β̂` = log-prior weight. Do not reuse either symbol bare.
6. **"Two directions" is a naming collision.** D07's *"one function, read in two directions"* = influence vs consistency. InfoNCE's *"bidirectional"* = state→action and action→state. Both may be present in HiCAP; do not conflate.
7. **Ancestry is shared:** IBC `2109.00137` is cited by both LIT §3 (TanitLang) and the REF-F prior-art review as the loss ancestor; EBT `2507.02092`/JEM `1912.03263` are TanitLang's "cheapest route to the pair energy" (LIT §3, *"read the PDF before this is planned on"* — UNVERIFIED).

### 2.3 Things TanitLang already learned the hard way that HiCAP must not re-derive (all MEASURED-in-doc unless marked)

* **Magnitude is not meaning:** a shuffled CoT and a `"depicts"`-token CoT matched/doubled ORION's influence (PUBLISHED-in-doc `2606.12706`); REF-C's real prior lost a KL contest to norm-matched noise (0.1878 vs 0.4065). ⇒ HiCAP's "the score moves selection" needs a norm-matched null **and** a permuted null, judged on the *outcome* axis.
* **Controls that mispair:** index-roll on episode-grouped rows was 97.5 % same-episode; draw the partner across groups and assert no same-group pair survives.
* **The bar is the cross-episode control, not "no port":** a 5-class manoeuvre prior already moves selection −0.0254 m [−0.0425, −0.0093]; "improves on no prior" was never established (overlaps 0).
* **Consistency needs a seed floor and per-family reporting:** 88 % inconsistency from seed alone; lateral 100 % vs longitudinal 62-69 % (PUBLISHED-in-doc `2605.17268`).
* **Non-degeneracy before GPU:** `Var_a[prior]` over the candidates and its rank-correlation with `conf0` (PAV `2410.08146`; D07 A7). One forward pass.
* **Anchor count:** V2/V3 say **117**; V5/D07 and `MODEL_REGISTRY.md` (fan-quality table: *base (128 anchors)*, XL 256, base a bit-exact prefix of XL) say **128**. Cite the registry: 128. The 117 is unexplained stale text (UNVERIFIED which refc version).
* **Headroom numbers:** TanitLang's 0.404 m (0.654 → 0.250, n=120, non-parity, T0) is **not** the registry's; the registry's fan table (scaleab, `taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json`) gives base oracle-in-fan 0.1914 [0.1654, 0.2184], `sel_gap` 0.2813. HiCAP must quote the registry/raw JSON, not D07 §9.

---

## 3. Open questions the PI left that HiCAP could answer, and the constraints HiCAP must obey (priority 3)

### 3.1 Open questions (each with where it is left open, and the discriminator that would close it)

| # | question as left open | file · section | how HiCAP could answer it, and what would count |
|---|---|---|---|
| Q1 | *"whether `E_φ`'s training objective should be contrastive, ranking, or a process reward"* | D07 §11 | HiCAP **is** the contrastive arm. Pre-register both outcomes; compare against a listwise/soft-target arm (REF-F E-F2) and judge on the **outcome axis** (`outcome_null_screen`, cross-episode control), not KL |
| Q2 | *"The energy must get BETTER, not LOUDER"*; near-perfect consistency costs 11.5× ADE with a poor energy | WPP §4; D07 A5.3, A7 | Re-run the β/authority table with the HiCAP score in place of REF-C's 5-class manoeuvre prior. Success = CON_rank separated from chance **at ADE ≤ base** and at authority ≥ 12 %. Quote β, authority % and ADE together |
| Q3 | The **shared codebook** (one embedding table read by an LM and by the planner's prior ports) is HYPOTHESIS, unbuilt | D07 §5, STATE header | HiCAP's action head over a released vocabulary *is* such a table. Cheapest test: tie the coarse-level embedding rows to the `lat_to_anchor`/`lon_to_anchor` rows and check the `w=0`/zero-port identity still holds bit-for-bit |
| Q4 | Is there a driving analogue of TRM's `puzzle_id`? Open; must be a **learned** code; WP-B void | V5 §3.3; D07 §4 | HiCAP's state embedding is a learned, scene-derived code. Test *learned vs blank vs random*; run the admissibility question against the situation classifier first |
| Q5 | Dynamic agent↔agent / agent↔geometry interaction: hand-crafted edges add nothing (WP-F); learned interaction *untested* | D07 §7b | HiCAP's predictive imagination over agent tokens + the **agent-perturbation intervention** (energy landscape must move more for a collision-course agent than for one behind). Needs agent tokens; note the P1 agent-conditioning gate *failed* on the tiny rig (PI queue §9) |
| Q6 | Is a **frozen** strong backbone a decision substrate? `2607.10172` says frozen degrades; WP-I says DINOv3 carries near-agent semantics the trunk lacks | D07 §3 vs §7a | Add the frozen VLM tower as an arm of WP-I (near-band `vru<20m` null) and of REF-F H-F7; keep the encoder-unfreezing arm (stage B) as the control |
| Q7 | Is the influence/consistency tension a trade-off or dissolvable? *"we have a continuous dial where the field has two discrete points; that is the contribution"* | D07 STATE header, A2 | HiCAP adds a second dial (temperature/authority, shortlist size k). Report the **frontier**, not a point |
| Q8 | Language-free vs language-combined, same `z` | V5 WP-E, D07 §6 | If HiCAP's VLM text side is used only as a regulariser, run the language-free control on the same state head |
| Q9 | Goals lack **constraints** (position, time) | PI review #1 (open) | A hierarchical vocabulary could carry typed constraint slots; sibling P1 reports `within_m / by_time_s / at_arc_m / hold_for_s` exist in the *ancestor* HIERARCHY_VOCABULARY, not in v7 (UNVERIFIED by me) |
| Q10 | Three forks; **influence mode** | BRIEF §4; V2 §7 | Not HiCAP's to answer (PI). HiCAP can supply the evidence V2 §7 says must come first: a monitor-mode consistency signal shown non-inert with all controls firing |
| — | *not* answerable: free-form QA (*"no question corpus"*), route/strategic topology (*no source in this corpus or model*) | V5 §2.4; D07 A1 C1 | say so, per family, with n |

### 3.2 Binding constraints stated in these documents (HiCAP must obey, or state the PI ruling that lifts them)

1. **H12 / mass / mechanism.** Language is an additional part, never the core; *"a common latent space which includes all"*; a design where the language/VLM mass dominates the hierarchy repeats D01's D1 error, and the PI did not *"recognize the refc design"* in it (BRIEF §1). ⇒ HiCAP must show REF-C's fan/scorer as a recognisable part (cached vocabulary = anchor bank; contrastive score = `sel_score`).
2. **Vision-only inference; goal path information-disjoint from the situation classifier** (CLAUDE.md 2026-08-03; BRIEF §5; D07 §4-§5). State what every goal/level-selection signal is computed from; admissibility test *"could this have been computed from the situation classifier's output?"*. Labels may use ego; inference may not.
3. **Rationale training targets must not derive from the logged future** (D07 A3; `2608.01755` "trajectory anchoring bias"; LCDrive cold-starts this way on our corpus). Applies to any HiCAP head that is decoded into an explanation; not to imitation-style policy training.
4. **Scorer stays the committer; free text never reaches the planner; masked, released vocabularies only** (`NOT_YET_EXTRACTABLE` masked); **zero-init, gated ports; identity at init is bit-exact** (`β=0`, `Wg=Wc=0`, `audio=0`, `z=0` must reproduce the baseline) (V2 §3.2; V3 §2.3, §6; D07 §2).
5. **Every control must be SHOWN TO FIRE on a deliberate-regression arm** before any number is admissible; **replicate arm mandatory** (`A0b_replicate`: "separated" on 3 of 18 metrics with zero levers moved, ~17 % false positives); **seed floor**; **norm-matched + permuted nulls**; cross-episode partner drawn across groups (D07 §2, A1, A4, A6; §2.3 above).
6. **Reporting:** `CON_rank` only with β, authority % **and** ADE (WPP); four metric families per eval, never pooled; paired episode-cluster bootstrap with the estimator named (CLAUDE.md); consistency **per family**.
7. **Tier stamping:** T0 (world-model/diagnostic) is never "driving performance"; T1 self-action open loop is the primary tier and is still open loop; closed loop needs AlpaSim/NuRec or a vehicle (D07 §8).
8. **Latency:** tact 300-500 ms; Alpamayo-R1's 99 ms is an RTX 6000 Blackwell workstation figure with 71 % CoT decode (`2511.00088` Tbl 14) — never quote it as embedded; on Thor only `max_memory_allocated` is admissible (CLAUDE.md).
9. **Source discipline:** registry/raw JSON only (§2.3: anchors = 128; headroom = registry fan table). TanitLang's own numbers are T0 non-parity pilots, one seed, ADE-only: *"precondition, not a result"*.
10. **Vocabulary is frozen** (v7, 2026-08-23; adding a token = a `*_V8` tuple behind a version switch — sibling P1 §1.1, not verified by me) and **no result may be attributed to the 22-token tactical vocabulary** until its head receives gradient (PI queue §10).
11. **Traffic-light/signal heads:** not admissible without an ego-only comparison; label is an *"Alpamayo-derived teacher signal"*, never *"GT traffic light"* (PI review addendum, BINDING; the underlying PI ruling *"no vlm, we stick to the alpamayo labels as teacher signals"* is RELAYED/PENDING — its scope for an architecture VLM is **UNVERIFIED and should be put to the PI**).
12. **Do not import** the v7 world-model finding `H-ARCH-ACTINS` into a REF-C-family arm (V3 §0); do not call REF-C a world model.

---

## 4. Terminology map (priority 4): TanitLang term ↔ HiCAP / REF-F term

Use the programme's own vocabulary where it exists. Traps are flagged ⚠️.

| TanitLang (source) | HiCAP / REF-F term | note |
|---|---|---|
| anchor fan / anchor bank / `F₀` (128 anchors; `anchor_controls [128,2]`) | **cached action vocabulary** | REF-F's is 4,096 (64×64 farthest-point, residual over CTRA); TanitLang's are the *refined* fan members. Say which. ⚠️ V2/V3's "117" is stale |
| scorer `sel_score`, base score `S₀`, `conf0` | base/prior score; the contrastive score's fusion partner | `conf0` = classifier surface *before* priors (D07 A6); `Δ = conf − conf0` is the guidance delta at one subtraction |
| **compatibility energy** `E_φ(z, τ_k)` (low = agree) | **negative contrastive logit** `−e^τ⟨ŝ, â_k⟩` | §2.2 |
| reasoning state `z`, `z_T` (TRM latent) | **state embedding** (state-head output) | HiCAP's is not recursive; if it becomes so, keep `z` |
| trajectory encoder + bilinear ⑤ | **action head** (action encoder) | REF-F uses a *numeric* action tower (CoVer-VLA design), not a text encoder |
| `β` (energy weight) / **authority** `β·sd[E]/sd[S₀]` | logit scale `e^τ` (temperature) × fusion weight | ⚠️ REF-F's `β̂` is the **log-prior** weight — a different quantity |
| guidance delta `S₁ − S₀`; `INF_sel`, `INF_geo` | re-rank delta; selection-flip rate; geometric displacement | `INF_kl` is retired as a decision statistic (wrong axis) |
| `CON_rank`, `CON_gap`, `L_cons` | retrieval rank of the selected entry; loss (`InfoNCE` under smooth-min reading) | always with β, authority, ADE; per family |
| top-5 re-rank (WP-C / refcv5 WP-7) | shortlist → re-rank; **retrieve-then-re-rank** | REF-F falls back to it if H-F1 refuted |
| zero-init prior ports `lat_to_anchor`, `lon_to_anchor`, `maneuver_to_anchor` (base only), `lan_gate` | **per-level vocabulary → anchor log-prior** | ⚠️ **`lan_*` = Lane-Anchored Navigation, not language** — never reuse "lan" |
| v7 vocabularies: `g_str` (8, = `STRATEGIC_GOAL_TOKENS_V7`), `a_str` (7, `STRATEGIC_ACTION_TOKENS_V7`), `a_tac.lat` (8), `a_tac.lon` (8), `g_tac.goals` (22, the "22-token head"), `nav_command` (3) | **hierarchical vocabulary levels** | names per sibling P1 §1.1 (56 declared slots, frozen 2026-08-23; not re-verified by me). The "15-token strategic vocab" of PI queue §3 = 8 + 7 |
| two clocks / multi-rate, *tact* (300-500 ms), System-1/System-2 | decision cycle; fast retrieval vs slow re-rank | H1 (*"100 ms vs 500 ms vs. 1 second"*) |
| four brains: operative / tactical / strategic / fallback-MRC (H1); H11 monitoring per level | HiCAP hierarchy levels + supervisor | brain 4 has **no** TanitLang port (D01 §5); H11's flag has no trigger path |
| **trunk as the only vision encoder** (D01 §7) | the frozen VLM tower, if it is the *only* encoder | the property to preserve |
| stage A/B/C/D (bridge only → joint → semantics → language) | HiCAP staging (frozen → +heads → unfreeze → language) | REF-F's stages 1-3 are a different axis (loss curriculum) |
| headroom = selected − fan-best (`sel_gap`, `oracle-in-fan`) | **oracle-in-vocabulary** upper bound; selected/random/oracle | quote the registry (§2.3) |
| constant-`z` floor; shuffled-`z`; norm-matched + permuted prior; cross-episode control; raw-input floor | NC1 state permutation; NC2 random vocabulary; NC3 kNN; NC4 ego-only/image-only | different names, same job; keep TanitLang's *norm-matched* and *cross-group* refinements |
| `echo_gate.assert_not_echoing` vs `("ha","ha0_ext")`; deliberate-regression arm | anti-echo gate; deliberate-regression arm | reuse, do not re-implement |
| **learned situation/`puzzle_id` code** | learned state code | never the situation classifier's output |
| "training flywheel", "Training FlyWheel", "DataFlyWheel" | — | ⚠️ agent-stream names, **not** a method; do not use as a technical term |
| "energy bridge", "one function read in two directions" | "one score, two uses (select, audit)" | ⚠️ not InfoNCE's "bidirectional" (state↔action) |
| shared codebook / `<TAC:…>`, `<STR:…>` special tokens | vocabulary embedding table with two consumers | HYPOTHESIS in TanitLang |

---

## 5. Implications for HiCAP

1. **Present HiCAP as the missing training objective of TanitLang's Energy Bridge, not as a rival.** `E_φ` is already specified as a scorer over the fan; D07 §11 leaves its objective open; InfoNCE over a cached action vocabulary is a concrete, testable answer (§2.2). This is the alignment the PI asked for.
2. **Keep REF-C recognisable** (BRIEF §1): cached action vocabulary = the anchor bank; contrastive score = `sel_score`; released v7 vocabularies = the hierarchy levels feeding zero-init log-prior ports. A design where the VLM dominates by mass repeats the error DIALOGUE_01 D1 named.
3. **Frozen VLM: allow it only as the single encoder, keep its text/LM side off the control path, and treat `2607.10172` ("frozen degrades") as the live counter-evidence.** Put the frozen-tower arm beside DINOv3 in WP-I's near-band null and REF-F's H-F7, with TanitLang's stage-B unfrozen encoder as the control.
4. **Do not merge the two positives.** `L_nce` (ground-truth-nearest entry) trains the score; TanitLang's `L_cons` (selected anchor; InfoNCE only under the smooth-min reading, which D07 does not define) is an audit term. Keep any rationale/explanation head off logged-future supervision (A3).
5. **Adopt TanitLang's hard-won controls verbatim:** norm-matched + permuted nulls, cross-group partners with an assertion, cross-episode control as the bar, replicate arm, seed floor, `Var_a[prior]` screen, identity-at-init, every control shown to fire. Judge influence on the outcome axis, never on KL.
6. **Report the influence/consistency frontier, not a point** — a second dial (temperature/authority, shortlist k) on top of TanitLang's β. Always quote β (or `e^τ`), authority % and ADE together; consistency per family.
7. **Respect the goal-path admissibility rule in the hierarchy.** Level-k retrieval that carries a situation class into level-(k+1) is forbidden; the strategic level has no supervision source here (6/15 tokens, no route), so report STRATEGIC as UNAVAILABLE with reason and n, and credit nothing to the 22-token tactical head.
8. **Predictive imagination is the v6/v7 line's job, not REF-C's** — stamp T0/T1, never import `H-ARCH-ACTINS` into a REF-C-family arm, and treat step count as a trained property (fan diverges past it).
9. **Audio needs an explicit PI ruling** (the ruling's letter says *inference only vision*); build it as a zero-init, absent-by-default slot so `audio=0` is bit-identical to the vision-only arm. **Elastic cameras** need a stamped frame/azimuth convention plus a mirror/derangement test before any result is read.
10. **Ask the PI one scoping question:** does the relayed *"no vlm, we stick to the alpamayo labels as teacher signals"* cover a VLM inside the architecture, or only VLM label-grading? Pending until answered (PI queue §1).
11. **Use the programme's vocabulary and avoid its traps** (§4): "cached action vocabulary", "state/action head", "authority", "outcome-axis"; never "lan" for language, never "flywheel" as method, never bare `β`.
12. **Quote the registry, not TanitLang, for numbers** (128 anchors; base `sel_gap` 0.2813 / oracle-in-fan 0.1914). TanitLang's numbers are T0 non-parity pilots and, until the package is brought into git, unreproducible from this repository.

---

## 6. Deliverable manifest

| artifact | where it lives | status |
|---|---|---|
| this file `P3_tanitlang.md` | `repo:TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-hicap/P3_tanitlang.md` | staged (`git add`, verified with `git ls-files --stage`) |
| scratch copies of D07 and the LIT review (text extracted from Drive, for reading only) | `local:/tmp/claude-0/…/scratchpad/{d07,lit}.md` | **exists in ONE place, disposable**; the source of truth is Drive (ids in §0) |
| TanitLang design/dialogue package (67 files) and `stack/tanitad/instruments/cot_influence.py` + 29 tests | Drive copies of the prose only; **package and instrument are in no ref of this repo** | ⛔ **STRANDED** — not my deliverable, escalated below |

**Escalations (integration, not "please merge"):**
1. ⛔ The TanitLang package and the `cot_influence` instrument are unbanked in this repository (HANDOVER: *"nothing committed, nothing pushed, branch `agent/arch-inf-20260803`"*; zero hits in any ref). HiCAP's controls and the energy-bridge formulas cannot be reproduced from git until the owning stream stages them.
2. ⛔ PI scoping question on the relayed *"no vlm"* ruling (§5 #10).
3. ⚠️ PI ruling needed for audio at inference (§2.1).
