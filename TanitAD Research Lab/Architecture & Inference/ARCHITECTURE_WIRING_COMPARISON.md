# Architecture, Wiring & Hierarchical Capability — the six-arm comparison

**Created:** 2026-07-21 · **Status:** LIVE REFERENCE — refresh in place, do not fork
**Charter (Sayed, 2026-07-20):** *"a comparison between v1 flagship, v2, v3.5, REF-C, REF-B, REF-A regarding
their wiring and hierarchical capabilities and architecture."*

> **Why this document exists.** H26 — *hierarchical cross-alignment beats ungrounded selection* — is the
> program's **declared core-goal proof** (HYPOTHESIS_LEDGER 07-18). "Hierarchical capability" is therefore not
> a descriptive axis here; it is **the thesis under test**. So every arm is asked the same three questions:
> **what is wired to what, through which mechanism, and did that wiring ever measurably DO anything?**
>
> The program's hardest-won lesson is that **a seam that exists is not a seam that works**. This document
> refuses to award credit for a diagram.

**Method.** Every wiring fact was read from source in `stack/` (never from prose); every param count in §2.7
was **re-measured this session** by instantiating the config and counting (script in scratchpad, results
cross-checked against `Project Steering/MODEL_REGISTRY.md`); every ADE came from `MODEL_REGISTRY.md` or a raw
eval JSON. Seam measurements come from `TanitAD Research Hub/HYPOTHESIS_LEDGER.md` (the H-number companion
named by `CLAUDE.md`) and from commit `fc2c484`'s number table, itself re-checked here against the raw
`flagship-v15-*-ckpt.json` files vendored in-repo. **No pod was touched** (all four are training/benchmarking).

**Companions:** `Project Steering/MODEL_REGISTRY.md` (the only quotable source for model facts) ·
`V35_DESIGN.md` §2A (the v3.5 wiring spec) · `HYPOTHESIS_LEDGER.md` (H18/H19/H25/H26) ·
`V3_HIERARCHICAL_PLANNING_DESIGN.md` · `Project Steering/PROGRAM_OVERVIEW.md`.

---

## 0. How to read this document

### 0.1 The hierarchy-capability scale — defined **before** it is used

Applied first **per seam**, then rolled up **per arm**. The rungs are ordered by *how much useful hierarchical
work is proven*:

| Rung | Name | Definition |
|---|---|---|
| — | **UNMEASURED** | The seam exists in code and carried gradient, but **no seam-level measurement of it exists anywhere in the program.** Not a rung: an unknown. By house rule an unknown earns **no credit**. |
| ⓪ | **NONE** | No such level / no such seam. |
| ① | **HARMFUL** | Measured, CI-separated, in the **wrong** direction — the wiring costs accuracy, and the deployed path routes around it. |
| ② | **INERT** | Measured; the effect is null within CI, **or** the effect is a constant offset with `per_window_content_helps ≈ 0` (the seam carries no per-window decision). |
| ③ | **PARTIAL** | The arm has ≥1 seam that is load-bearing and ≥1 that is not. |
| ④ | **LOAD-BEARING** | **Every** seam of the arm passes the program's own gate. |

**Arm-level rule.** An arm's verdict is the roll-up: ④ only if *all* measured seams pass and *none* is
unmeasured; ③ if the arm mixes; ①/② if all measured seams fail that way; **UNMEASURED** if the arm has real
seams but no seam-level measurement. The per-seam verdicts are always printed next to the roll-up — the
roll-up alone hides the interesting part.

### 0.2 What counts as "measured" — the instrument, and its coverage hole

The program owns exactly one seam instrument: the **TanitEval hierarchy panel**
(`taniteval/taniteval/hierarchy.py`, in-repo since R2 closed). Per seam it runs the downstream layer with the
real upstream signal vs a **mean/constant** upstream signal (`per_window_content_helps`) and vs **no** upstream
signal (`helps_vs_none`), with an 8-split holdout interval and an effect-size floor of **≥0.02 acc / 0.05 m /
0.01 cos** (`hierarchy.py` MIN_* constants). `per_window_content_helps` is the honest read — `helps_vs_none`
was shown to flip sign by encoder (H26 refinement, 07-18).

> ⚠️ **The coverage hole, stated up front.** `runner.run_hierarchy` **refuses any arm that is not a trained
> 4-brain WorldModel/RefAModel**: `if not L["traj_capable"] or getattr(L["model"], "tactical_policy", None)
> is None: … skip`. REF-B's brain is named `tactical` (not `tactical_policy`) and REF-C loads with
> `step_readout = None`. **Therefore REF-B's and REF-C's seams have NEVER been measured by the only
> instrument that could measure them, and cannot be without new code.** Their "UNMEASURED" is structural,
> not an oversight of scheduling.
>
> 🟥 Also: no `hier_*.json` exists in this repo. The H26 raw panel outputs live only on
> `tanitad-eval:/root/taniteval/results/`. The numbers below are quoted from `HYPOTHESIS_LEDGER.md`, which is
> the H-number system of record, but the **raw** artifacts are a one-pod reconstruction risk (§8).

### 0.3 Diagram legend — one vocabulary for all six

```
  [ BOX ]        a brain / module that holds parameters and receives gradient
  [ BOX ]*       frozen (no gradient, not in the optimizer)
  ( item )       a non-parametric or out-of-gradient element

  ==>   seam MEASURED LOAD-BEARING (CI-separated, right direction, content matters)
  -->   seam exists, carried gradient, NEVER MEASURED
  ..>   seam MEASURED INERT (null within CI, or constant-offset only)
  xx>   seam MEASURED HARMFUL
  ??>   socket exists in code but NOTHING WAS EVER FED IT (dead seam)
  ~~>   DESIGN ONLY — not built, not trained, not measurable

  {mech}         the mechanism on the seam: FiLM / KV / add / gate / logit / action-ch
```

---

## 1. The verdict, on one screen

| Arm | State | Encoder | World model? | Trained levels | Seams | Per-seam verdict | ADE@2s (heldout) | **Hierarchy verdict** |
|---|---|---|---|---:|---:|---|---:|---|
| **flagship v1** `flagship4b-speedjerk-30k` | ✅ deployed | ViT d768×12, **trained** | ✅ latent, action-conditioned | 3 (+1 unused) | 3 | nav→str ② · ctx→tac ④ · int→op ① | **0.4522 ± 0.0312** | **③ PARTIAL — 1 of 3, and one HARMFUL** |
| **flagship v2** `flagship4b-v2-30k` | ❌ killed @7.8k | same ViT | ✅ same | 3 (+1 unused) | 5 | all **UNMEASURED** | 6.179 ± 1.2845 @6k | **UNMEASURED** |
| ↳ *v3enc* `flagship4b-v3enc-30k` | 🟢 running | same ViT | ✅ same | 3 | 5 | all **UNMEASURED** | 🟥 none | **UNMEASURED** |
| **flagship v3.5** | ⚠️ **DESIGN ONLY** | ViT (cap raised to 400 M) | ✅ intended | 3, jointly trained | 3 (S1/S2/S3) | ~~> not measurable | — | **⚠️ DESIGNED — NO MEASUREMENT EXISTS** |
| **REF-C-XL** `refc-diffusion-xl-30k` | ✅ complete | ResNet-L, **trained** | ⚠️ aux only (LAW) | 1 (+2 side-tokens) | 3 (1 dead) | ctx→cond — · man→anchor — · tgt-latent ?? | **0.458 ± 0.057** | **UNMEASURED — and structurally the flattest** |
| **REF-B v2** `refb-refbpatch-v2-30k` | ✅ complete | ViT d768×**25**, trained | ❌ **none by design** | 3 (+1 real fallback) | 4 | all **UNMEASURED** | **0.5921 ± 0.0685** | **UNMEASURED** |
| **REF-A** `refa-dinov2-4b` / `refa-dynin-30k` | ✅ reference (H4 closed) | **frozen DINOv2-B/14** | ✅ latent (adapter space) | 3 | 3 | int→op ② (content); other 2 unmeasured | 2.1322 ± 0.1821 / 2.9196 ± 0.3937 | **② MEASURED INERT** |
| *(bridge)* **v1.5** lineage `a`/`ab`/`abc` | diagnostic | v1 ViT **frozen** (v1.6: 4 blocks @ LR 1e-5) | ✅ frozen, used as a **probe** | 1 trainable head | 3 (across arms) | imag→dec ④ · VTARGET ② · ROUTE ② (⚠️ broken test) | `ab` **0.5437 ± 0.0653** | **③ PARTIAL — holds the only proven conditioning seam in the program** |

*ADE column: `heldout` split-mean ± `overlapping_holdout_se` (the deprecated estimator — retained only because
the published figures use it). Decision-grade intervals are the episode-cluster bootstrap in
`Project Steering/CI_RECOMPUTE_2026-07-20.json`; ranks 1–2 (flagship v1 vs REF-C-XL) are a **statistical tie**.*

---

## 2. The comparison axes, side by side

### 2.1 Encoder — type, frozen/trained, tokenisation

| Arm | Class | Input | Tokenisation | Trained? | Encoder params | **Encoder share of total** |
|---|---|---|---|---|---:|---:|
| flagship v1 | `ViTEncoder` d768 × depth 12, 12 heads | 9-ch 256 px (3 RGB frames @100 ms, D-015) | patch 16 → **16×16 = 256 tokens** → `SpatialGridReadout` grid 4, d_readout 128 → **state 2048** | ✅ from scratch | 87,121,280 | **33.1 %** |
| flagship v2 / v3enc | identical | identical | identical | ✅ | 87,121,280 | 31.9 % |
| flagship v3.5 (DESIGN) | ViT, **widened** d1024×16 or d768×24 | identical | identical | ✅ intended | ~170–190 M *(ASSUMPTION — scaling estimate, V35 §7.2)* | ~50 % (intended) |
| REF-C-XL | `ResNetEncoder` (torchvision-free, BasicBlock), base_width **124**, blocks (3,8,20,6) | 9-ch 256 px, **last frame only** for the map | stride 32 → **8×8 conv map, F = 992**; window seen only via a pooled sequence | ✅ from scratch | 199,496,532 | **79.2 %** |
| REF-C-base | same, base_width **88**, blocks (3,6,16,6) | same | 8×8, F = 704 | 🟡 training | 90,458,632 | 86.8 % |
| REF-B v1/v2 | the **same `ViTEncoder` class**, depth **25** | identical to flagship | identical (256 tokens → state 2048) | ✅ from scratch | 179,263,616 (incl. readout) | **66.0 %** |
| REF-A | **frozen DINOv2-B/14**, features precomputed to disk | 224 px, features only — no images in the loop | 16×16 = 256 tokens, dim 768 → `TemporalGridAdapter` (token + frame-Δ, 2D → readout) → state 2048 | ❌ **frozen; not even in the model** | **adapter 196,736** (0.13 % of trainable) | **~0 %** |
| v1.5 / v1.6 | v1's ViT, **frozen** (v1.6 unfreezes last 4 blocks at LR 1e-5) | identical | identical, then re-expanded to W×16 cell tokens | ❌ (v1.6: partial) | 0 trainable (v1.5) | 0 % |

**The single sharpest allocation fact:** REF-C spends **79 %** of its budget on the encoder and owns the
program's best proposal set; the flagship spends **33 %** and its (v1.5-measured) fan is 1.87× worse.
⚠️ Confounded (conv vs ViT, direct-head vs world-model, frozen vs trained trunk) — suggestive, **not proven**.

### 2.2 World model / imagination — and what "predicting" means per arm

| Arm | Has a world model? | What it predicts | Imagination (H15) |
|---|---|---|---|
| flagship v1/v2/v3enc | ✅ **yes** — `OperativePredictor`, causal transformer over a window of (state, action), **residual**, multi-horizon k∈{1,2,4}, action-FiLM | the **next latent state** given an action → decoded to metric Δpose by `StepDisplacementReadout` → SE(2) accumulate | ✅ `ImaginationField` over the encoder token grid — 22,055,683 params, advection prior + refinement + per-cell log-variance |
| flagship v3.5 (DESIGN) | ✅ intended (v1 trunk kept verbatim) | same | ✅ intended, and named as the design's dominant latency term |
| REF-C-XL | ⚠️ **no forward world model.** The **LAW** head is a *latent-world-model auxiliary*: `law_head([pooled, traj]) → pooled_next`, a one-shot regression, not a rollable dynamics | it predicts a **trajectory directly**: a fan of 256 anchor+offset trajectories with confidences. "Predicting" = choosing and refining a geometric plan | ⚠️ **`ImaginationField` over the CONV MAP** (20,986,339, XL only) — it refines the tokens the decoder attends, i.e. it is **belief-over-perception, not rollout-over-time** |
| REF-B v1/v2 | ❌ **none, by design.** This is the point of the arm | it predicts **actions** (5×[steer,accel] direct heads) and **waypoints** (direct per-horizon heads / anchored fan). No recursion anywhere | ❌ none |
| REF-A | ✅ yes — the **identical** `OperativePredictor`, but the latent space is the trainable **adapter** space over frozen features | next adapter-state given an action | ❌ **absent.** `RefAModel` never builds an `ImaginationField` (it operates on encoder tokens, which REF-A does not own) |
| v1.5 | ✅ **frozen v1 predictor, used as an instrument** — rolled under 8 probe action sequences, latents read at (5,10,15,20) | the fan is chosen by a decoder that has *seen the consequences* of candidate controls | ✅ inherited frozen |

> **The finding hiding in this table.** Two different things in this program are called "imagination":
> (a) the flagship's H15 belief field over *unobserved space*, and (b) v1.5's *consequence rollout* of the
> operative predictor. **Only (b) has ever been measured to help** (−0.1355 m, CI-separated). (a) has been
> flat at `vision_use ≈ 12 %` since 19 k. They should not be conflated in v3.5 planning.

### 2.3 Brain inventory — and how many levels are *genuinely* present

| Arm | Level 4 (fallback) | Level 3 (strategic) | Level 2 (tactical) | Level 1 (operative) | Genuinely trained levels |
|---|---|---|---|---|---:|
| flagship v1 | ⚠️ `FallbackMonitor` **exists as a class and is exported — but is instantiated by NO trainer and NO evaluator** (verified: only refs are the class def, `models/__init__`, and a docstring in `eval/spectral.py`) | `StrategicPolicy` d384×4, own nav embedding, emits `ctx` + `route_logits` | `TacticalPolicy` d512×6 → maneuver logits, wp heads, **target_latent**, **intent**; PLUS `tactical_pred` d512×6 (a second predictor at horizons 8/16) | `OperativePredictor` d768×10 + `InverseDynamicsHead` | **3** |
| flagship v2 / v3enc | same (unused) | same + ego `[v0,yr0]` into the nav embedding | same, but `wp_heads` **replaced** by `AnchoredTacticalDecoder` (128 time anchors, FPS) + `goal_traj_head` | same + ReZero-gated intent | **3** |
| flagship v3.5 (DESIGN) | not specified | strategic goal module → `G_s = ⟨ROUTE, VTARGET-band, ODD envelope⟩` | **the anchored-diffusion planner IS the tactical brain**; the selected plan **is** `G_t` | v1 predictor, now also conditioned by `G_s` and `G_t` | **3, jointly trained** (the stated delta vs every v1.5 arm) |
| REF-C-XL | ❌ none | ⚠️ `StrategicCtx` = a **GRU over the 8 pooled window features → one d_ctx=96 token**. 4,133,472 params | ⚠️ a `maneuver_head` (MLP on `pooled`) whose 5 logits reweight anchor confidences. **There is no tactical module** — the decoder is the whole planner | ❌ none — no action-level brain at all | **1 planner + 2 side-signals** |
| REF-B v1 | ✅ **real and instantiated**: `ConfidenceHead` (1,443,329, fully detached) + `FeatureOOD` (buffers, 0 params), logged every step | `StrategicHead` d384×4 (rev2, D-A1 — before rev2 it trained on a constant `follow`) | `TacticalHead` d512×6 → maneuver + wp + intent | `OperativeHead` d768×6 → 5 direct action heads, **no recursion** | **3 + a real 4th** |
| REF-B v2 | same | same + `[v0,yr0]` via `ego_emb` + **LayerNorm** | `TacticalHead` with an **anchored** decoder (128 time anchors, d384×4L) replacing `wp_heads` | same | **3 + a real 4th** |
| REF-A | ❌ none | the **identical** `StrategicPolicy` class/config | the **identical** `TacticalPolicy` + `tactical_pred` | the **identical** `OperativePredictor` | **3** |
| v1.5 | ❌ | goal **tokens** only (ROUTE), no strategic module | the trainable head is the planner | frozen v1 predictor (probe only) | **1 trainable** |

> **Correction to a repeated claim.** `MODEL_REGISTRY.md` §2 says the flagship and REF-A *"differ in exactly
> two things: (1) the encoder, (2) the SIGReg target."* Read from source, there is a **third**: the flagship
> holds the **H15 `ImaginationField` (22.06 M)** and REF-A does not build one at all. It is a *consequence*
> of (1) — H15 lives on encoder tokens — but "exactly two things" understates the architectural gap by 22 M
> params and one whole mechanism. See §7.

### 2.4 Every conditioning seam — source → target, mechanism, init discipline, gate/dropout

This is the core table. **Init discipline** matters because the program has twice been burned by a live seam
whose contribution norm swamped the base signal.

| # | Arm | Seam (source → target) | Mechanism | Init discipline | Gate / dropout | Status |
|---|---|---|---|---|---|---|
| F1 | **v1** | nav command → strategic | `nn.Embedding(4,128)` → **FiLM cond** of every `CausalBlock` | FiLM `to_scale_shift` **zero-init** (identity at step 0); embedding default | none | ② INERT — `route_skill_vs_chance` **0.0**; pure command echo |
| F2 | **v1** | strategic `ctx` (d256) → tactical | **FiLM cond** of every tactical `CausalBlock` | FiLM zero-init; `ctx_proj` default | none | ④ **LOAD-BEARING @30 k** (maneuver Δ **+0.044**, CI-sep, `content_matters=true`). Was ② at 19 k — **duration alone flipped it** |
| F3 | **v1** | tactical `intent` (d256) → operative | `intent_proj: Linear(256→768)`, **ADDED to the action embedding**, then FiLM | `intent_proj` **default init (live from step 0)**; FiLM zero-init | **none — ungated** | ① **HARMFUL** — cos vs-none **−0.238**; `intent_proj` norm **31.4** vs `act_emb` **28.3** |
| F4 | **v2/v3enc** | ego `[v0, yr0]` → strategic **and** tactical | `Linear(2, d_cmd)` / `Linear(2, d_cond)`, **added** to the nav embedding / ctx | default | **ego-dropout 0.25** (joint, also zeroes the v0 action channel) | UNMEASURED |
| F5 | **v2/v3enc** | tactical `intent` → operative | same additive path, **now ReZero-gated** | gate **init 0.1** (action-dominant start) | scalar gate | UNMEASURED |
| F6 | **v2/v3enc** | maneuver logits → anchor confidences (H19) | `Linear(5→128, bias=False)` on `log_softmax`, **added to the confidence logits** | **ZERO-INIT scalar `h19_gate`** — strict no-op at start | scalar gate | UNMEASURED |
| F7 | **v2/v3enc** | tactical `target_latent` → trajectory | `goal_traj_head([z_t, target_latent]) → wp` (2,101,768 params) | default | — | UNMEASURED |
| F8 | **v2/v3enc** | nav → strategic, with dropout | as F1 | as F1 | **nav-dropout 0.5** + an always-on **nav-zeroed route aux** (w 0.3) | UNMEASURED |
| C1 | **REF-C** | strategic `ctx` (d96) → decoder condition | `ctx_to_cond: Linear(96→512)`, **ADDED** to `cond_proj(measurement)` | **ZERO-INIT** (weight and bias) | none | **UNMEASURED** (structurally excluded from the panel) |
| C2 | **REF-C** | maneuver logits → anchor confidences (H19) | `maneuver_to_anchor: Linear(5→256, bias=False)` on `log_softmax`, added to conf | **default init — LIVE from step 0, deliberately** ("the coupling is the point") | **none** | **UNMEASURED as an ablation** — see §7 drift 3 |
| C3 | **REF-C** | tactical `target_latent` → decoder condition | `tgt_film`: FiLM on the condition | zero-init | — | ?? **DEAD SOCKET** — `graft_target_latent: false` in the XL run; the module is not even constructed |
| C4 | **REF-C** | measurement `[v0, nav one-hot]` → decoder | MLP → **LIVE FiLM** in every `CrossAttnLayer` (`zero_init=False`, on purpose) | live | **ego-dropout 0.5** on `v0` | UNMEASURED (but this is the *core* condition, not a graft) |
| B1 | **REF-B v1/v2** | nav (+ego) → strategic | `nn.Embedding(4,128)` (+ `ego_emb` **LayerNorm**'d in v2) → FiLM cond | FiLM zero-init | v2: **ego-dropout 0.5**, joint over `[v0,yr0]` | UNMEASURED |
| B2 | **REF-B v1/v2** | strategic `ctx` → tactical | FiLM cond of every tactical block | zero-init | none | UNMEASURED |
| B3 | **REF-B v1/v2** | tactical `intent` → operative | FiLM cond of every operative block (`cond_dim = d_intent`) — **NOT** additive-to-actions: REF-B's operative is deliberately **not fed past actions** (copycat guard) | zero-init | none | UNMEASURED |
| B4 | **REF-B v2** | maneuver logits → anchor confidences (H19) | `Linear(5→128)` on `log_softmax` → **`LayerNorm(128)`** → added to conf | ⚠️ **LayerNorm-pinned at ‖prior‖ ≈ √N ≈ 11.3 — live and un-gated** | none | UNMEASURED. *(The trained arm used this variant, not the zero-init gate — see §7 drift 1)* |
| A1 | **REF-A** | nav → strategic | identical to F1 | identical | none | UNMEASURED on this arm |
| A2 | **REF-A** | strategic `ctx` → tactical | identical to F2 | identical | none | UNMEASURED on this arm |
| A3 | **REF-A** | tactical `intent` → operative | identical to F3, **ungated** (`RefAModel` never passes `gated_intent`) | live | none | ② **INERT (content).** `helps_vs_none` positive (cos 0.936 vs 0.852) **but** `intent_proj` norm **1792** vs `act_emb` **14.5** and `per_window_content_helps ≈ 0` → the operative co-adapted to a huge **constant** offset |
| A4 | **REF-A** | ego `[v0, yr0]` → operative | **through the ACTION CHANNEL** (`action_dim` 3→4), not through a brain | n/a — a native input | **ego-dropout 0.25**, joint over the whole ego vector | ✅ the mechanism that moved fwd-ADE **3.73 → 0.83** (isolated) |
| V1 | **v1.5** | frozen predictor's imagined consequences → planner | **KV cross-attention tokens** (8 probes × 4 read horizons = 32 tokens, + a source-type embedding marking them CONSEQUENCE) | learned pos/source embeddings | — | ④ **LOAD-BEARING** — `a→ab` **−0.1355 m (−19.9 %), CI [0.038, 0.233]** |
| V2 | **v1.5** | VTARGET band → decoder condition | `Embedding(24, 128)` × ReZero gate, **added** to the measurement | gate init 0.1 | **goal-dropout 0.5** + explicit DROPPED row | ② INERT — `ab→abc` **+0.0106**, CI [−0.094, +0.072] ⚠️ **broken test, see §4** |
| V3 | **v1.5** | ROUTE class + graded → decoder condition | `Embedding(5,128)` + `Linear(1,128)`, × ReZero gate, added | gate init 0.1 | goal-dropout 0.5 | ② INERT — and the **H26 monitor FIRED at 2.80×** measurement norm despite `rt_gate` 0.10 |
| S1 | **v3.5** | strategic `G_s` → tactical planner | ~~> **KV tokens** (2–4 appended to the decoder's heterogeneous KV set) | ReZero per token-group, init 0.1 | goal-dropout ≥0.5; **hard cap: contribution ≤1.0× measurement norm, rescaled in-graph; >1.5× fails the run loud** | ⚠️ DESIGN |
| S2 | **v3.5** | strategic `G_s` → operative | ~~> **FiLM on the predictor's conditioning vector** | LayerNorm the cond + **scale to the action-embedding norm** (the literal H26 fix) | ReZero 0.1, cap ≤1.0× | ⚠️ DESIGN |
| S3 | **v3.5** | tactical plan `G_t` → operative | ~~> **inverse-dynamics → future action sequence, entering the EXISTING action channel** | n/a — a proven port, no new surface | **stop-gradient on `G_t`**; scheduled sampling ⅓→⅔; goal-dropout ≥0.5; canary evaluated with S1/S2/S3 **zeroed** | ⚠️ DESIGN |

### 2.5 What supervises each level

| Arm | Strategic | Tactical | Operative | Encoder / representation |
|---|---|---|---|---|
| flagship v1 | route-heading CE on `route_logits` (⚠️ target derived from the **same** future heading that produces `nav_cmd` → the echo) | maneuver CE + waypoint L2 (`/pose_scale`) + goal-latent JEPA (change-weighted) | JEPA at k∈{1,2,4} + K-step rollout + inverse-dynamics + **hierarchical metric grounding** (`op`/`tac`/`str`: metric-invdyn on real pairs + forward SE(2) consistency) | SIGReg `full_relaxed` (free_dims 64) + everything above, end-to-end |
| flagship v2/v3enc | + always-on **nav-zeroed** route aux (w 0.3) | + nearest-anchor **CE** and winner-takes-all **L1** (replacing the unimodal wp L2) + goal-decode wp | + jerk penalty, aux-accel | + **encoder–ego linear decorrelation** penalty (w 0.05; v3enc: 0 until 10 k then 0.02) |
| flagship v3.5 (DESIGN) | goal vocabulary (`V3_GOAL_VOCABULARY_V1`), VLM/kinematic labels | anchor CE + WTA L1, ranking CE on `sel_score` | v1 recipe verbatim | v1 recipe + wider encoder |
| REF-C-XL | ⚠️ **route CE (w 0.1) reads `pooled`, NOT `ctx`.** The `ctx` token that actually conditions the decoder has **no direct supervision at all** — only trajectory gradient through a zero-init seam | maneuver CE (w 0.1) on the `pooled`-fed head | — | trajectory L1 + anchor CE (w 1.0 each) + **LAW** aux (w 0.5) + speed-class (w 0.2, refc1 only) |
| REF-B v1/v2 | route-heading CE (D-A1 — this level had been training on a constant `follow` before rev2) | maneuver CE (0.5) + waypoint L1/L2 (1.0) (+ v2: anchor CE + WTA) | direct action regression (1.0) + 0.5 s action-sequence (1.0) + inverse-dynamics (0.5) | all of the above, end-to-end; + aux accel / aux yaw; **confidence head is fully DETACHED by construction** |
| REF-A | identical to flagship v1 | identical to flagship v1 | identical + aux ego probes (speed/yaw/accel R²) | ❌ **nothing** — DINOv2 is frozen; SIGReg is `pred_only` |
| v1.5 | — | anchor CE (t=0, vs nearest **original** anchor) **and** ranking CE on `refined_logits` (vs nearest **refined**) — the N2 rule | — | ❌ nothing (trunk frozen); v1.6 unfreezes 4 blocks at LR 1e-5 |

### 2.6 Decoder / action-emission path — what actually produces the scored trajectory

| Arm | The scored path | Does the hierarchy participate? |
|---|---|---|
| **flagship v1** | `encode(window)` → `rollout_decode`: 20 sequential `predictor(win_s, win_a)` calls → `step_readout(z_t, ẑ)` per step → `accumulate_se2` | 🔴 **NO. Verified in source:** `metric_dynamics.rollout_decode` calls `predictor(win_s, win_a)` with **no `intent` argument**, and TanitEval's operative step is `intent-free` by protocol. **The 0.4522 is produced by a path in which zero of the three seams participate.** |
| flagship v1 tactical head | `wp_heads` off the tactical latent | yes — and it scores **3.38 m** (3.150 in the P2 pass), i.e. **worse than CV**, against the same model's 0.452 rollout |
| flagship v2/v3enc | same rollout; tactical now emits an anchored fan | no (rollout still intent-free) |
| flagship v3.5 (DESIGN) | the planner's selected plan; the WM rolls under it | ✅ **by construction — that is the point**; but the canary must be evaluated with S1/S2/S3 zeroed, and **no leaderboard number may come from a GT-derived plan** |
| **REF-C-XL** | `decoder(fmap, m, ctx, maneuver_logits, steps=2)` → 256 anchor+offset trajectories → **argmax over the t=0 classifier score** → selected anchor+offset | partially: C1 and C2 both enter the confidence logits. 🔬 **The known flaw: all 256 anchors are denoised, but selection uses the UN-refined score** — `_, off = self._decode(...)` discards the refined confidences. Geometry is refined, ranking is not |
| **REF-B v1/v2** | the **tactical head's direct per-horizon waypoints** (`taniteval/refb_eval.py`), ego frame of the last window pose | ✅ yes — REF-B's scored surface is the tactical output, so its seams *are* in the scored path (they have simply never been ablated) |
| **REF-A** | identical to the flagship: grounded intent-free rollout | 🔴 no — same structural bypass |
| **v1.5** | anchored fan on frozen-trunk tokens → select on **`refined_logits`** (the REF-C flaw repaired) | ✅ yes — the conditioning is the only thing that trains |

> **This is the most important row in the document.** Our best arm's headline number is measured on a path
> that structurally excludes its hierarchy. REF-B v2 (0.5921) is the only trained arm whose scored surface is
> emitted by a **distinct tactical brain that a distinct strategic brain conditions** — and it is an arm whose
> seams have never been measured. (REF-C's scored surface is seam-touched too, but by a GRU token and five
> maneuver logits entering one confidence vector, not by brains.)

### 2.7 Params by module — **all re-measured this session**

| Module | flagship v1 (speed) | flagship v2 / v3enc | REF-C-XL | REF-B v2 | REF-A (dyn-in) |
|---|---:|---:|---:|---:|---:|
| encoder (+readout) | 87,121,280 | 87,121,280 | 199,496,532 | 179,263,616 | **196,736** (adapter only; DINOv2 frozen, off-model) |
| operative predictor (+inv-dyn) | 96,609,283 | 96,609,284 | — | 52,256,526 | 96,611,076 |
| tactical predictor (`tactical_pred`) | 26,535,424 | 26,535,424 | — | — | 26,535,936 |
| tactical policy / planner-decoder | 22,736,141 | **30,098,063** (anchored) | **22,702,345** | **30,270,742** (anchored) | 22,736,141 |
| strategic | 8,385,027 | 8,385,411 | 4,133,472 (GRU ctx) | 8,385,667 | 8,385,027 |
| H15 imagination | 22,055,683 | 22,055,683 | 20,986,339 (over the conv map) | 0 | **0 (absent)** |
| goal-traj head | — | 2,101,768 | — | — | — |
| LAW aux / measurement / aux heads | — | — | 4,082,656 / 17,280 / 513,960 | — | — |
| fallback | 0 (class unused) | 0 | — | **1,443,329** | — |
| **model total** | **263,442,838** | **272,906,913** | **251,932,584** | **271,619,880** | **154,464,916** |
| grounding heads (outside the model) | 13,432,338 | 13,432,338 | — | — | 13,432,338 |
| aux-accel | 528,897 | — | — | (booked in operative) | 3 ego probes (speed/yaw/accel), 525,057 each |
| **trainable total (as recorded)** | **277,404,073** | **286,339,251** | 251,932,584 | 271,619,880 | **≈169.5 M** ⚠️ |

*flagship, REF-C and REF-B totals reproduce `MODEL_REGISTRY.md` exactly. REF-A's breakdown is **not in the
registry** (the leaderboard prints "—"); the model total 154,464,916 was measured here from
`RefAModelPlus(adapter_kind="temporal")` at `refa4b_config()` and should be added to the registry (§8).
⚠️ REF-A's ≈169.5 M trainable total is **the model + grounding + 3 aux probes at their default `hidden=256`**;
the probe widths were not read from the run's `config.json` (pods are off-limits this session), so treat the
last ~1.6 M as **UNVERIFIED**. Note the shape of REF-A's budget: **~169.5 M trainable + a frozen ~86 M
DINOv2-B/14 outside the checkpoint ≈ 256 M total capacity** — so REF-A is *capacity*-matched to the fleet but
carries **~40 % less TRAINABLE capacity** than every arm it is scored against (the ~86 M is the published
DINOv2-B/14 size — 🟥 **UNVERIFIED here**, the weights never enter our checkpoints). That asymmetry is inherent to a
frozen-encoder arm (D-A4) rather than a defect, but it is nowhere stated, and it is the correct caveat to
attach to REF-A's 2.13/2.92.*

### 2.8 The measured seam evidence — or the explicit absence of it

| Arm | Seam measurements that exist | Source | What does **not** exist |
|---|---|---|---|
| flagship v1 | 3 seams × 2 checkpoints (19 k and 30 k), full panel | HYPOTHESIS_LEDGER 07-18 (×3 entries) | nothing material |
| flagship v2 | **none** | — | the arm died at 7.8 k; only an ADE and a rate-of-learning diagnostic exist |
| flagship v3enc | **none** | — | no checkpoint has been evaluated at all |
| flagship v3.5 | **none possible** | — | it is a design |
| REF-C-XL | **none** — and none *obtainable* without new code | `runner.run_hierarchy` skip condition | any ablation of C1 (ctx), C2 (H19) or the LAW aux |
| REF-B v1/v2 | **none** — same structural exclusion | same | any ablation of B1/B2/B3/B4. Only *proxy telemetry* exists: `route_acc 1.00`, `man_acc 0.75`, `anchor_acc 0.469`, `n_modes 33` at 29999 |
| REF-A | **1 of 3** (`intent→operative`), from the 07-18 panel preview | HYPOTHESIS_LEDGER 07-18 refinement | REF-A's own reads for `nav→strategic` and `ctx→tactical` are **not quotable** |
| v1.5 | 3 conditioning sources, fully decomposed (oracle / sel_gap / frac2x) across 4 arms | commit `fc2c484`; raw `flagship-v15-*-ckpt.json` in-repo ✅ | a *fair* test of goals (frozen trunk, 8 k steps, swamped seam, labels mid-repair) |

---

## 3. Wiring diagrams — one vocabulary, six arms

### 3.1 flagship v1 — `flagship4b-speedjerk-30k` (0.4522) — the most levels, the least load-bearing wiring

```
   nav_cmd(4)                9-ch 3-frame stack, 256px
       |                              |
       |                     [ ViT d768 x 12 ]  87.1M   (TRAINED)
       |                              |
       |                     ( SpatialGridReadout 4x4x128 -> state 2048 )
       |                              |
       |   F1 {FiLM, zero-init}       +----------------------+--------------------+
       +..>[ STRATEGIC d384x4 ] 8.4M  |                      |                    |
           route_logits (aux CE)      |                      |                    |
                  | ctx (d256)        |                      |                    |
                  |  F2 {FiLM, zero-init}                    |                    |
                  ==>[ TACTICAL d512x6 ] 22.7M <-------------+                    |
                     maneuver | wp_heads | target_latent | intent                  |
                                                          |                       |
                            F3 {add-to-act_emb, LIVE, UNGATED}                    |
                                                          |                       |
   actions[steer,accel,v0/10] ----------------------------xx> [ OPERATIVE d768x10 ] 96.6M
                                                                    |
                                        ( H15 ImaginationField 22.1M, on encoder tokens )
                                                                    |
   >>> THE SCORED PATH (0.4522) >>>  predictor(win_s, win_a)  [ NO intent ]
                                     -> step_readout -> accumulate_se2 -> ADE@2s
                                     ( grounding heads 13.4M: op / tac / str )

   ( FallbackMonitor — DEFINED IN CODE, INSTANTIATED BY NOTHING )
```

**Read:** three levels, three seams; one inert, one load-bearing (only after 30 k), one harmful — and the
number on the leaderboard comes from the dashed line at the bottom that touches none of them.

### 3.2 flagship v2 (killed @7.8 k) = flagship v3enc (running) — same wiring, different schedule

```
   nav_cmd --F8{FiLM + nav-dropout 0.5 + always-on nav-ZEROED route aux}--> [ STRATEGIC ]
   ego[v0,yr0] --F4{Linear(2,d_cmd), ADD, ego-dropout 0.25}--------------->      |
                                                                                | ctx
   ego[v0,yr0] --F4{Linear(2,d_cond), ADD}------------------------------->      |
                                                                          [ TACTICAL ]
                          maneuver logits --F6{Linear(5,128), ZERO-INIT GATE}-->|
                                                             [ AnchoredTacticalDecoder ]
                                                              128 FPS time anchors, +7.4M
                                                                  |            |
                                            target_latent --F7--> [ goal_traj_head 2.1M ]
                                                                  |
                              intent --F5{add-to-act_emb, ReZero gate init 0.1}-->
                                                                  |
   actions --------------------------------------------------> [ OPERATIVE ]
                                                                  |
   ( encoder<->ego linear DECORRELATION penalty, w 0.05 | v3enc: 0 until 10k then 0.02 )
   ( future-action dropout 0.30 | v3enc 0.15 ; rollout-k 12 | v3enc 4->8->12 )

   SCORED PATH: unchanged — still the intent-free rollout.
```

**Read:** v2 is the *correct* response to v1's H26 findings — F5 fixes the swamp, F8 fixes the echo, F6 is
zero-init disciplined — and **every one of those fixes is unmeasured**, because the arm was killed on its
rate of learning (exponent −0.50 vs v1's −0.84 on the 1500–7500 window, R² 0.541 for v1) after ten levers
were switched on at once. v3enc keeps every decode-side lever from step 0 and stages only the four
encoder-grounding ones.

### 3.3 flagship v3.5 — ⚠️ **DESIGN ONLY. NOT BUILT. NO MEASUREMENT EXISTS.**

```
              nav / ego / scene
                      |
              [ STRATEGIC goal module ]   G_s = <ROUTE, VTARGET-band, ODD envelope>
                      |
        S1 ~~>{KV tokens, ReZero 0.1, cap <=1.0x meas-norm, goal-dropout >=0.5}
                      |                                    S2 ~~>{FiLM, LayerNorm+
              [ TACTICAL = the ANCHORED-DIFFUSION PLANNER ]   norm-matched to act_emb,
              256 anchors -> fan -> select                    ReZero 0.1, cap <=1.0x}
              the SELECTED PLAN *is* G_t                          |
                      |                                           |
        S3 ~~>{inverse-dynamics -> future ACTION SEQUENCE,         |
               entering the EXISTING action channel;              |
               STOP-GRADIENT on G_t; scheduled sampling 1/3->2/3}  |
                      |                                           |
              [ OPERATIVE PREDICTOR ] <---------------------------+
              window states + actions + v0  +G_s +G_t
                      |
              latent rollout -> grounded metric dpose

   ALL THREE TRAINED JOINTLY  (the stated delta vs every v1.5 arm)
   Hard gate: ratio = ||contribution|| / ||measurement|| rescaled in-graph at >1.0,
              run FAILS LOUD at >1.5.  Canary evaluated with S1/S2/S3 ZEROED.
   Gate H (causality): sensitivity + correctness-of-direction + per_window_content
                       + effect floor + norm parity, at 5k and every milestone.
```

### 3.4 REF-C-XL — the flattest hierarchy, the best proposals

```
   9-ch 256px, last frame          window (8 frames, pooled)
          |                                  |
   [ ResNet-L base_width 124 ] 199.5M (79% of budget)
          |                                  |
     fmap [B,992,8,8]                  pooled_seq [B,8,992]
          |                                  |
   [ H15 ImaginationField 21.0M ]     [ StrategicCtx GRU 4.1M ] --> ctx (d96)
     (belief over PERCEPTION,                 |
      not rollout over time)          C1 ..? {Linear(96->512), ZERO-INIT, ADD}
          |                                  |
          |    [v0(dropout 0.5), nav one-hot] -> [ measurement MLP ] -> m (d128)
          |                                  |         |
          |                            C4 {LIVE FiLM in every CrossAttnLayer}
          |                                  |         |
          +----> KV ------------> [ AnchoredDiffusionDecoder  d512 x 6L ] 22.7M
                                    256 FPS anchors (real GT trajs)
                                    per-anchor conf + offset
                                          ^          |
   pooled -> [ maneuver_head ] --C2--------+          |  2 truncated denoise steps
             (LIVE Linear(5->256), NO GATE)           |  (offsets kept, CONFS DISCARDED)
   pooled -> [ route_head ] -- aux CE (w 0.1)         |
                                                      v
   ( tactical target_latent ) ??> tgt_film      argmax over the t=0 conf  -> traj
     C3: NEVER CONSTRUCTED (graft off)          <-- the selection flaw

   pooled + traj -> [ LAW head 4.1M ] -> pooled_next   (aux only; NOT a rollable WM)
```

**Read:** REF-C has **one** decision-making module. Its "hierarchy" is a GRU token added into a condition
vector through a zero-init linear, plus five maneuver logits added into 256 confidence logits. And it ties
the flagship on ADE, owns the best fan in the program (oracle-in-fan **0.1640**), and beats the flagship in
the high-speed stratum outright (0.3243 vs 0.5513).

### 3.5 REF-B v2 — the deepest *supervised* hierarchy, no world model

```
   nav_cmd(4) -> [ nav_emb 128 ]         9-ch 3-frame stack
        |                |                       |
   ego[v0,yr0] -> [ ego_emb -> LayerNorm ] --ADD-+   [ ViT d768 x 25 ] 179.3M
        (ego-dropout 0.5, joint)          |          (the ~130M freed by having
                                          |           no predictor/imagination)
                                    cmd (d128)              |
                                          |          ( readout -> state 2048 )
                          B1 --> [ STRATEGIC d384x4 ] 8.4M <-----+
                                 route_logits (aux CE)            |
                                          | ctx (d256)            |
                          B2 {FiLM, zero-init}                    |
                                 [ TACTICAL d512x6 ] <------------+
                                   maneuver_head        |
                                        |               |
                         B4 {Linear(5->128) -> LayerNorm, LIVE, pinned at ~sqrt(N)}
                                        |               |
                                 [ Anchored wp decoder d384 x 4L, 128 time anchors ]
                                        |
                    >>> THE SCORED PATH: the tactical waypoints <<<   (0.5921)
                                        | intent (d256)
                          B3 {FiLM, zero-init}
                                 [ OPERATIVE d768x6 ] -> 5 DIRECT action heads
                                   (no recursion, and NOT fed past actions)

   [ ConfidenceHead 1.4M — fully DETACHED ] + ( FeatureOOD, buffers only )   <- a REAL brain 4
```

**Read:** REF-B is the only arm whose *scored output* is a hierarchical output, the only arm with an
instantiated brain 4, and the arm with the most direct per-level supervision. Its seams have never been
ablated. (v1 is identical minus the ego widening and minus the anchored decoder — `wp_heads` in that slot.)

### 3.6 REF-A — the flagship's brains, verbatim, on a frozen encoder

```
   frozen DINOv2-B/14 features, precomputed to disk    [ ENCODER: FROZEN, OFF-MODEL ]*
                    |
        ( FeatureStandardizer — frozen buffers, one-shot fit )
                    |
        [ TemporalGridAdapter 196,736 ]   tokens + frame-to-frame delta -> state 2048
                    |
   nav --A1{FiLM}..>[ STRATEGIC d384x4 ]  (IDENTICAL class + config to the flagship)
                    | ctx
             A2{FiLM}
                    -->[ TACTICAL d512x6 ] + [ tactical_pred d512x6 ]
                          | intent
             A3{add-to-act_emb, LIVE, UNGATED}   ||intent_proj|| = 1792  vs  ||act_emb|| = 14.5
                          |
   actions[steer, accel, v0/10, yr0/YAW_SCALE] --A4--> [ OPERATIVE d768x10 ] 96.6M
        (ego-dropout 0.25, JOINT over the ego pair)
                          |
   SCORED PATH: intent-free grounded rollout (same structural bypass as the flagship)

   NO H15 imagination field is built.   SIGReg = pred_only (not full_relaxed).
```

**Read:** REF-A is the cleanest possible hierarchy ablation — the *same brains*, the *same configs*, the same
data — and its intent seam is the program's most extreme swamping case (**124× norm ratio**), with
`per_window_content_helps ≈ 0`. A4 is the interesting one: REF-A feeds ego dynamics **through the action
channel**, which is exactly the port v3.5's S3 wants to use — and it is the highest-yield conditioning
intervention in program history (3.73 → 0.83 fwd-ADE isolated; 2.918 → 0.4522 in the flagship causal pair).

### 3.7 The bridge — v1.5 / v1.6 (why it matters to the hierarchy question)

```
   [ ViT d768x12 ]*  FROZEN            [ OperativePredictor ]*  FROZEN
          |                                     |
      states [B,8,2048]              rolled under 8 PROBE action sequences
          |                                     |  latents read at (5,10,15,20)
    (a) re-expand to 8x16 = 128            (b) 32 IMAGINATION tokens
        cell tokens (d128 -> 512)               (d2048 -> 512) + src_embed[1]
          |                                     |
          +--------------- KV token set --------+
                             |
   v0 (ego-dropout 0.5) -> [ measurement MLP ] -> m (d128)
              VTARGET band --V2{Embedding x ReZero 0.1, ADD}--> m   (goal-dropout 0.5)
              ROUTE cls+grd --V3{Embedding+Linear x ReZero 0.1, ADD}-> m   <-- FIRED at 2.80x
                             |
                   [ V15Decoder d512 x 8L, 256 anchors ] ~30M   (the ONLY trainable module)
                             |
                   select on refined_logits  (the REF-C selection flaw REPAIRED)

   MEASURED (fc2c484, n=881):  a 0.6792 -> ab 0.5437 -> abc 0.5543
       imagination (a->ab):  -0.1355 m, CI [0.038, 0.233]   SIGNIFICANT   <== V1
       goals      (ab->abc): +0.0106 m, CI [-0.094, +0.072] null          <== V2/V3
       decomposition: imagination bought RANKING (sel_gap -0.1439), NOT proposals (oracle +0.0084)
```

---

## 4. Hierarchy-capability verdict, per arm

| Arm | Verdict | The one-line justification |
|---|---|---|
| **flagship v1** | **③ PARTIAL** — nav→str ②, ctx→tac ④, intent→op ① | The **only** arm in the program with a seam measured load-bearing. Also the only arm with a seam measured *harmful*. And its headline number is produced by a path that uses none of them. |
| **flagship v2** | **UNMEASURED** | Five seams, all with correct init discipline, zero seam measurements — the arm was killed at 7.8 k on its rate of learning, not on its wiring. Its per-lever telemetry was healthy; the failure was **simultaneity**. |
| ↳ **v3enc** | **UNMEASURED** | Same wiring as v2. No checkpoint evaluated. Its own exponent read is currently v2-like (−0.503, R² 0.225, n=72 on the canonical 1.5–5.05 k window) — *cannot presently discriminate*, per V35 N7. |
| **flagship v3.5** | ⚠️ **DESIGNED — NOT BUILT** | Three seams specified with mechanism, init, gate, dropout, norm caps as a hard in-graph gate, a stop-gradient, a scheduled-sampling schedule and a pre-registered causality gate (Gate H). **This is the first design in the program where the seam discipline is a gate rather than a log line.** It has produced no number and must never be tabled beside a measured one without this marker. |
| **REF-C-XL** | **UNMEASURED** — and *structurally the flattest hierarchy of any trained arm* | One planner; one GRU token in through a zero-init linear (C1); five maneuver logits in through a live un-gated linear (C2); one dead socket (C3). Its "strategic" token has **no direct supervision**. Ties the flagship. |
| **REF-B v2** | **UNMEASURED** | The deepest *supervised* hierarchy (every level has its own loss), the only arm with a real brain 4, and the only trained arm whose **scored surface is emitted by a distinct tactical brain conditioned by a distinct strategic brain**. Never ablated, and not ablatable without new code. |
| **REF-A** | **② MEASURED INERT** | Its one measured seam carries a **constant offset, not content** (`per_window_content_helps ≈ 0` at a 124× norm ratio). The hierarchy is present, gradient-carrying, and content-free. |
| *(bridge)* **v1.5 lineage** | **③ PARTIAL** (the `ab` arm alone would be ④ — its one seam passes) | The imagination seam is the **only conditioning mechanism in this program with a CI-separated positive effect**. The goal seams (`abc` only) measured null — but that test is broken (§5.4) and is formally **RE-OPENED**, not settled. |

---

## 5. The analytical payload

### 5.1 Which arm is actually the most hierarchical — by measurement, not by diagram?

**By diagram:** flagship v1 and REF-A (three brains + three seams + a fourth-brain class) > REF-B (three
brains + three seams + a *working* fourth brain) > REF-C (one planner + two side-tokens) > v1.5 (one head).

**By measurement, the ranking inverts into something uncomfortable:**

1. **flagship v1 — ③ PARTIAL (1 of 3).** The only arm with any seam proven load-bearing. That seam
   (`ctx→tactical`) affects the tactical output, which scores **3.38 m — worse than constant velocity**.
2. **v1.5 `ab` — ③ PARTIAL.** Its one proven seam is worth **−0.1355 m**, the largest CI-separated
   conditioning effect ever measured here. But it is a *consequence* seam, not a *goal* seam — it carries
   physics, not intent.
3. **REF-A — ② INERT.**
4. **REF-B v2, REF-C-XL, flagship v2/v3enc — UNMEASURED.** No credit.

**So: stated plainly, and it is not an embarrassment but a finding —**

> **The program's best arms are, by measurement, its least hierarchical.**
> flagship v1 (0.4522) scores through an **intent-free** rollout that bypasses all three of its seams.
> REF-C-XL (0.458, a statistical tie) has one decision module and two barely-wired side-tokens.
> The arm with the most complete, most supervised, most instantiated hierarchy — REF-B v2 — is **third**.
> And the two arms in which the hierarchy *is* in the scored path (REF-B v2 at 0.5921; the flagship's own
> tactical head at 3.38) are both **worse** than the flagship's hierarchy-free rollout.
>
> **H26 has not been demonstrated on any arm. It has been demonstrated on one seam of one arm, and that seam
> feeds a head we do not deploy.**

### 5.2 Is hierarchical wiring correlated with performance across the six?

**n = 6. No statistic is claimable, and none is claimed.** What the pattern looks like:

| Arm | Seams in the scored path | Levels | ADE@2s |
|---|---:|---:|---:|
| flagship v1 | **0** | 3 | 0.4522 |
| REF-C-XL | 2 (both into one logit vector) | 1 | 0.458 |
| REF-B v2 | 3 | 3 (+1) | 0.5921 |
| v1.5 `ab` | 1 (imagination) | 1 | 0.5437 |
| REF-A | 0 | 3 | 2.1322 |
| flagship v2 @6 k | 0 | 3 | 6.179 |

If anything, the visible trend is **negative** — fewer hierarchical seams in the scored path, better score.
**This must not be read as "hierarchy hurts."** Three reasons it cannot support that reading:

1. **The dominant axis is not hierarchy, it is the encoder.** REF-A (2.13/2.92) differs from the flagship
   (0.4522) almost entirely in the encoder — *identical brains, identical seams*. H4 closed negative on that
   axis alone. Encoder variance swamps seam variance by ~2.5 m; seam effects are ≤0.14 m.
2. **The seams were never fairly built.** Two of the three measured seams failed on **scale discipline**
   (31.4-vs-28.3; 1792-vs-14.5; 2.80× in v1.5), which is a mis-scaled implementation, not a refuted thesis.
3. **The one seam that got duration flipped.** `ctx→tactical` was inert at 19 k and load-bearing at 30 k.
   Every unmeasured arm's seams got ≤30 k, and v1.5's got 8 k head-only steps.

The honest summary: **across six arms there is no evidence that hierarchical wiring buys accuracy, and no
fair test of whether it can.** That is precisely what makes v3.5's Gate H the first real experiment.

### 5.3 What each arm's wiring predicts about its ceiling — and what that implies for v3.5

| Arm | Its wiring predicts | Observed |
|---|---|---|
| flagship v1 | An excellent **dynamics integrator** with a lossy readout: the encoder+predictor+grounding chain is dense and end-to-end supervised; every top-down path is thin, late, and un-normed. Ceiling = whatever the rollout can do; the heads are decoration | ✅ exactly: rollout 0.4522, tactical head 3.38, P2 CEM over the *same frozen model* 0.893 open-loop / **1.038 closed-loop vs the head's 1.685** |
| flagship v2/v3enc | Ten simultaneous changes to the encoder's grounding regime → the shortcut is removed by design before a replacement exists. Ceiling unknowable early because the *rate* is what changed | ✅ encoder speed-probe R² 0.30 (v1: 0.861); killed on rate, not level |
| **REF-C-XL** | A very large encoder trained end-to-end **with the decoder attached** → excellent proposals. But: **selection ranks with the un-refined score, and ~92 % of the oracle gap is aleatoric** → the ceiling is a *selection* ceiling that cannot be bought back post-hoc | ✅ oracle-in-fan **0.1640**, selected **0.4714**, gap 0.3075, `frac_sel_2x_worse` 0.454; learned re-scorer recovers **+2.9 %, n.s.**; hand-written cost **0.0 %** |
| **REF-B v2** | No world model → cannot imagine, cannot roll out, cannot plan; every gain must come from a better *direct* readout. Ceiling = the best supervised regression the encoder supports | ✅ 0.5921 — beats CV in every speed stratum, and stops there |
| REF-A | A frozen generic encoder cannot expose ego-rotation geometrically → the arm becomes "a dynamics integrator" earning ~96 % of its accuracy from integrating `v0` | ✅ 2.92 with a **monotonically improving** milestone curve (3.755→3.694→3.016→2.920): a **capability ceiling, not overfitting**. H4 closed |
| **flagship v3.5** | ⚠️ DESIGN. Its wiring predicts it inherits v1's rollout **and** REF-C's fan only if the encoder is what buys the fan | Untested |

**The juxtaposition Sayed asked about — REF-C's flat hierarchy owns the best fan (0.1640) while the flagship
lineage's deep hierarchy produces a 1.87×-worse fan (0.3073) — and what it implies for v3.5:**

First, a precision that changes the conclusion: **0.3073 is not v1's fan.** v1 has *no* fan — it has no
multi-mode decoder at all (P2 had to hand-build a 16-seed constant-action grid to plan over it). 0.3073 is
**v1.5's** fan: REF-C's own decoder algorithm, at ~1.3× REF-C's decoder capacity, bolted onto a **frozen**
v1 trunk for 8 k head-only steps.

So the correct reading of the juxtaposition is **not** "flat hierarchy → good fan." It is:

> **The fan is bought by the ENCODER and by end-to-end coupling — not by the decoder and not by the
> hierarchy.** Same decoder algorithm, more decoder capacity, worse fan — the only difference that survives
> is that REF-C's 199 M encoder was trained *with the decoder attached for 30 k steps* and v1.5's was frozen.

Three consequences for v3.5, in order of confidence:

1. **Spend the raised cap on the encoder, not the planner.** V35 §7.1 already concludes this; this comparison
   independently confirms the mechanism (v1.5 is the controlled experiment: decoder held ~constant,
   encoder-coupling varied, fan moved 1.87×).
2. **The hierarchy is cheap and is not the fan lever.** S1+S2+S3 cost ~1–3 M params. They cannot buy
   proposals; at best they buy *ranking* and *goal-directedness*. Judge them on Gate H and on
   `per_window_content`, **never** on whether they move ADE alone.
3. **v3.5 inherits an unsolved selection ceiling from its REF-C parent.** ~92 % of REF-C's oracle gap is
   aleatoric; v1.5 already repaired the un-refined-score flaw and still sits 0.086 m short of G1. If v3.5's
   headline hope is "REF-C fan + v1 trunk," the fan must be *generated* better, not *ranked* better.

### 5.4 Which seam mechanisms have EVER worked in this program — and which have never

**This is the single most decision-relevant output of this document.** v3.5 has to pick mechanisms; it should
pick ones with a track record.

#### ✅ WORKED — mechanisms with a CI-separated positive measurement

| Mechanism | Where | The number | Caveat |
|---|---|---|---|
| **The ACTION CHANNEL** — feed a quantity as an extra action dimension into the already-calibrated `act_emb` port | `v0` as the 3rd channel (all arms, D-A3); `[v0,yr0]` as channels 3–4 (REF-A dyn-in) | REF-A fwd-ADE **3.73 → 0.83**, speed-R² 0.61 → 0.965 (isolated); causal pair flagship **2.918 → 0.4522**, paired **+2.21 m CI [2.04, 2.39]** | The biggest single fix in program history. It is a *state* port, not a *goal* port — no goal has yet been sent through it |
| **KV cross-attention tokens carrying imagined CONSEQUENCES** | v1.5 `a→ab` | **−0.1355 m (−19.9 %), CI [0.038, 0.233]**; miss@2m 0.2359→0.1643 | Bought **ranking** (sel_gap −0.1439), not proposals (oracle +0.0084). The only proven *conditioning* mechanism |
| **FiLM from a state-window transformer, strategic ctx → tactical** | flagship v1 @30 k | maneuver Δ **+0.044**, CI-sep, `content_matters = true` | Only after **30 k**; was inert at 19 k. Zero-init FiLM, no gate. The only *goal-bearing* seam ever load-bearing |
| **Zero-init / identity-start graft discipline** | REF-C `ctx_to_cond`, flagship v2 `h19_gate`, v1.5 ReZero gates | No graft with a zero-init start has ever *damaged* an arm | Absence of harm, not proof of help |

#### ❌ NEVER WORKED — mechanisms with a measured null or a measured harm

| Mechanism | Where | The killing number | Diagnosis |
|---|---|---|---|
| **Ungated additive injection into the action-FiLM condition** | flagship v1 `intent→operative`; REF-A same seam | cos vs-none **−0.238** (‖intent_proj‖ 31.4 vs ‖act_emb‖ 28.3); REF-A **1792 vs 14.5**, `per_window_content ≈ 0` | **Scale, then content.** Fixing the scale (v2's ReZero gate) is necessary but the 07-18 refinement showed it is **not sufficient** — per-window content was inert on *both* arms |
| **nav-command embedding → FiLM strategic** | flagship v1, v2 (pre-fix), REF-B, REF-A | `route_skill_vs_chance` **0.0**; `route_acc_follow` == majority base rate | The route target is derived from the same future heading that produces `nav_cmd` → **the layer learns to echo its own input**. A *label* pathology as much as a wiring one |
| **Additive goal embeddings through a ReZero gate** | v1.5 VTARGET + ROUTE | `ab→abc` **+0.0106**, CI [−0.094, +0.072] | ⚠️ **BROKEN TEST — formally RE-OPENED (V35 §2A.1).** Frozen trunk (goals could not shape features), 8 k head-only steps, `rt_over_m` **2.80×** (the monitor fired), labels mid-repair. What stays binding: *goals bolted onto a frozen fan at a mis-scaled seam do not improve ADE* |
| **Post-hoc re-ranking of a fixed fan** | REF-C v1.0 (hand cost) / v1.2 (learned, 47 arms) | **0.0 %** and **+2.9 % n.s.** of the oracle gap; GT-perfect speed matcher is **worse** than baseline | ~92 % of the gap is the statistics of a min over 256 candidates against one realised future. **Not a lever** |
| **Ranking on unsupervised denoise-time confidence** | REF-C's discarded refined confs | **1.36593 = 2.9× worse** than baseline | The conf head is never supervised at denoise timesteps. **Rule: if you rank on it, supervise it** (v1.5 does) |
| **LayerNorm-pinned prior on a logit vector** | REF-B v2's H19 prior **as trained** (see §7) | Never ablated. In-code post-mortem: the LayerNorm pins ‖prior‖ at √N ≈ 11.3 while `conf_norm` decayed 21.9→5.7 by step 100 → mode collapse | **Do not use LayerNorm as a seam scaler on a logit vector** — it *pins* a magnitude rather than *bounding* it. Use a zero-init scalar gate (the v4 fix) or a ratio cap (v3.5) |

#### ?? NEVER TRIED — sockets that exist and were never fed

| Socket | Where | Status |
|---|---|---|
| `graft_target_latent` — a tactical goal latent FiLMing the decoder condition | `refc.py`, zero-init, gated **off** by default | **Never constructed in any run** (`graft_target_latent: false` in the XL config). Not a failed experiment: an unconnected socket. **v3.5's S1 is the closest living relative of this idea** |
| `grounded_selector` — a param-free progress/collision proxy blended into the score | `refc.py` | Off in every run |
| `FallbackMonitor` | `fourbrain.py` | Class exported, instantiated by nothing. The flagship's brain 4 does not exist at runtime |

#### The three rules this table hands to v3.5

1. **Prefer ports the model already calibrates.** The action channel is the only conditioning port with a
   multi-metre track record. **v3.5's S3 uses it — that is the best-evidenced seam in the design.**
2. **KV tokens beat additive-into-a-condition-vector.** Every additive-into-`m`/`cond` seam measured null or
   harmful; the one KV seam measured **−0.1355 m**. **v3.5's S1 is KV — correct.** v3.5's S2 is additive-FiLM,
   the family with the worst track record — it is the seam most likely to fail, and it is the one carrying
   the explicit H26 norm-matching fix. Watch S2 hardest.
3. **A norm cap must be a gate, not a log line.** Every failure above is a scale failure first. v1.5 *logged*
   `rt_over_m 2.80×` and shipped anyway; v3.5 rescales in-graph at >1.0 and fails the run loud at >1.5.
   **Keep that. It is the single most valuable change in the v3.5 spec.**

---

## 6. What this comparison says about v3.5's three seams

| Seam | Evidence backing it | Risk this comparison surfaces |
|---|---|---|
| **S1** strategic `G_s` → planner, **KV tokens** | Strong — the family that produced the only proven conditioning win (v1.5 imagination tokens, −0.1355 m). Also the closest living relative of REF-C's never-fed `graft_target_latent` | The proven KV win carried **consequences (physics)**, not **goals (intent)**. S1 carries goals. That is a genuinely untested transfer — do not import the −0.1355 m as an expectation |
| **S2** strategic `G_s` → operative, **FiLM** | Weakest of the three. Its family (additive-into-the-conditioning-vector) is 0-for-3: v1 intent ①, v1.5 VTARGET ②, v1.5 ROUTE ② | Mitigated correctly (LayerNorm + norm-match to `act_emb` + ReZero 0.1 + hard cap) — this *is* the literal H26 fix. But the 07-18 refinement warns the fix addresses **magnitude**, and per-window **content** was inert even where magnitude was fine. **S2 needs `per_window_content_helps`, not `helps_vs_none`, to claim success** |
| **S3** tactical plan `G_t` → operative, **inverse-dynamics → action channel** | Strongest. It reuses the single highest-yield port in program history and adds no new swamping surface | The circularity hazards are correctly identified and decided (stop-grad, scheduled sampling, goal-dropout ≥0.5, canary with seams zeroed). One addition this comparison suggests: **also report the plan-free ADE at every milestone**, because v1's entire leaderboard history is a plan-free path — without that column, v3.5 cannot be compared to its own parent |

**One structural recommendation this comparison generates that §2A does not currently state:** the hierarchy
panel must be extended to arms whose brain is not named `tactical_policy`. Today
`runner.run_hierarchy` silently skips REF-B and REF-C. If v3.5's tactical brain becomes an
`AnchoredDiffusionDecoder` rather than a `TacticalPolicy`, **v3.5 will be skipped by its own Gate H
instrument** unless the loader gate is generalised first. This is a one-line skip condition standing between
the program and its core-goal measurement. → **Escalated in §8.**

---

## 7. Code-vs-prose drift found while writing this

| # | Drift | Evidence | Severity |
|---|---|---|---|
| **1** | 🔴 **The trained REF-B v2 (0.5921) is reproduced by `refb_v3.py`, NOT by `refb_v4.py`** — the file `MODEL_REGISTRY.md` §7 R1 names as the rescue. Rebuilding from `refb_v4.py` gives **271,619,625** params (tactical 30,270,487); the registry's measured run figures are **271,619,880 / 30,270,742**. `refb_v3.py` reproduces both **exactly**. Δ = 255 = `LayerNorm(anchor_n=128)`'s 256 params minus the v4 gate's 1. The **only** diff between the two files is the H19 prior mechanism (LayerNorm-scaled vs zero-init scalar gate). Corroborating: the v4 trainer logs `anchor_ent`/`prior_gate`, and the 2026-07-18 daily report's health schema for this arm lists neither | measured this session; `git diff refb_v3.py refb_v4.py` is 3 hunks, all in the H19 prior | 🟠 **HIGH — R1 is not fully closed.** Our 3rd-best arm still cannot be rebuilt byte-exactly from the file the registry points at |
| **2** | `MODEL_REGISTRY.md` §2: *"the flagship and REF-A differ in exactly two things: (1) the encoder, (2) the SIGReg target."* Source says three: REF-A also **has no `ImaginationField`** (flagship: 22,055,683 params). `RefAModel.__init__` never constructs one | `stack/tanitad/refs/refa.py` vs `stack/tanitad/models/fourbrain.py:455` | 🟡 medium — the claim understates a 22 M / one-mechanism gap |
| **3** | `PROGRAM_OVERVIEW.md`: *"H19 — Maneuver → anchor prior · ✅ validated by the REF-C anchor-prior graft."* No in-program **ablation** of that graft exists. The ledger's 07-18 entry validates H19 at the level of *design alignment* ("our maneuver vocab == the anchor set") plus external literature. REF-C's seams are structurally un-measurable by the hierarchy panel | `taniteval/taniteval/runner.py:165` skip condition; grep of all `.md` for an H19 ablation returns none | 🟡 medium — a ✅ resting on design, not measurement |
| **4** | Residual `refc.py` docstring drift — the module docstring was corrected (working tree, staged) to the measured 104,191,577 / 251,932,584, but **three function docstrings were not**: `refc_config()` still says *"~110 M"*, `refc_xl_config()` still says *"~260 M"*, `param_breakdown()` still says *"~110 M … ~260 M"*. Measured today: **base 104,191,577 · XL 251,932,584 · small 54,690,001** (all three match the registry exactly) | measured this session | 🟢 low — the fix landed but is incomplete |
| **5** | `FallbackMonitor` (the flagship's "brain 4") is defined and exported but **instantiated by no trainer and no evaluator**. Every doc that counts the flagship as a *4*-brain stack is counting a class, not a running module. REF-B's brain 4 (`ConfidenceHead` + `FeatureOOD`) **is** real and trained | grep across `stack/` + `taniteval/`: 3 hits, all definitional | 🟡 medium — affects how "4-brain" should be worded |
| **6** | REF-C's `route_head` (the strategic aux CE) reads **`pooled`**, not **`ctx`**. The token that actually conditions the decoder has **no direct supervision** — only trajectory gradient through a zero-init linear | `refc.py:776` (`ctx = self.strategic(pooled_seq)`) vs `refc.py:799` (`route_logits = self.route_head(pooled)`) | 🟢 low as a doc defect, but 🟠 as a *design* observation for v3.5 |
| **7** | The framing *"v1 has a worse fan (0.3073)"* (widely used, incl. in the brief for this doc) is imprecise: **v1 has no fan.** 0.3073 is **v1.5-`ab`**'s fan — REF-C's decoder on a *frozen* v1 trunk after 8 k head-only steps. The comparison to REF-C's 0.1640 is therefore frozen-vs-trained-encoder, not shallow-vs-deep-hierarchy | `MODEL_REGISTRY.md` §5 ("v1 has no multi-mode decoder"); commit `fc2c484` | 🟠 **HIGH — it changes the conclusion**, see §5.3 |

*Nothing in `stack/` was edited by this task. Drift 1 and 4 need code owners; both are escalated in §8.*

---

## 8. Gaps, unknowns, and what would close them

| # | Gap | Why it matters | Cheapest close |
|---|---|---|---|
| G1 | **The hierarchy panel structurally excludes REF-B and REF-C** — and would exclude v3.5 if its tactical brain is not a `TacticalPolicy` | Half the fleet's seams are unmeasurable; **Gate H could silently skip v3.5** | Generalise the `runner.run_hierarchy` gate + add per-arm seam adapters. **Escalated — this is a one-line skip condition standing in front of the program's core-goal proof** |
| G2 | **`refb_v3.py` vs `refb_v4.py`** (drift 1) | Our 3rd-best arm is not byte-rebuildable from the named file | Re-point R1 at `refb_v3.py` (or record explicitly that v4 is the *improved* successor and v3 is the *as-trained* artifact) |
| G3 | **No `hier_*.json` in the repo** | Every H26 number in this document traces to `HYPOTHESIS_LEDGER.md` prose, not to a raw artifact in git | Vendor the 4 existing panel JSONs from `tanitad-eval:/root/taniteval/results/` next time the pod is free |
| G4 | **REF-A's param breakdown is not in `MODEL_REGISTRY.md`** (leaderboard prints "—") | It is the *only* budget-mismatched arm (154.5 M trainable vs 263–272 M) and that is never stated | Add §2.7's REF-A column to the registry |
| G5 | **flagship v2's five seams were never measured before the arm was killed** | v2 contains the *correct* fix for two of v1's three seam failures (gated intent, route-from-vision). We killed the arm on rate and **learned nothing about the fixes** | If any v2/v3enc checkpoint is ever evaluated, run the hierarchy panel on it — it is the only existing test of the H26 fixes |
| G6 | **REF-C's C2 (H19) has never been ablated** | It is on, live, un-gated, from step 0 in our best-fan arm — and PROGRAM_OVERVIEW calls it ✅ validated | A single `graft_maneuver=false` re-eval of the frozen `refc-xl-30k` ckpt would settle it (needs the loader gate from G1) |

---

## 9. Sources & maintenance

**Primary sources used.** Code: `stack/tanitad/models/{fourbrain,predictor,encoder,metric_dynamics,flagship_v15,imagination}.py`,
`stack/tanitad/refs/{refa,refb,refc}.py`, `stack/tanitad/config.py`, `stack/tanitad/train/flagship_losses.py`,
`stack/experiments/refb-v2/{refb_v3,refb_v4,refb_train_v3,refb_train_v4,launch_v2.sh}`,
`stack/experiments/reset-speed4b/{refa_plus,refa_train_plus}.py`,
`taniteval/taniteval/{hierarchy,runner,loaders,refb_eval,refc_eval,planning}.py`.
Facts/results: `Project Steering/MODEL_REGISTRY.md` (§0–§8), `TanitAD Research Hub/HYPOTHESIS_LEDGER.md`
(07-18 H25/H26 ×3 entries; 07-18 H19/REF-C redesign), `V35_DESIGN.md` §0/§1/§2A/§7, commit `fc2c484`,
raw `flagship-v15-{a,ab,abc,abc_legacy}-ckpt.json` (in-repo).
Param counts in §2.7 and §7 were **re-measured this session** by instantiation.

**Refresh contract.** Update **in place** (this is a live reference, do not fork) whenever:
a seam measurement lands · an arm's wiring changes · v3.5 moves from DESIGN to built ·
`MODEL_REGISTRY.md` is refreshed. The two columns that must never drift are **"Status"** (design vs trained)
and **"Per-seam verdict"** (measured vs unmeasured). An unmeasured seam earns no credit here, ever.
