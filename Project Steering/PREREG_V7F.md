# PREREG_V7F — the next full v7 training run, pre-registered

**Written** 2026-09-03, Architecture & Inference FlyWheel · **0 GPU** · no pod, no Thor contact
**Schema** `.claude/skills/TanitAD_ValidateAIDesign` (SPEC before compute; tiny rig first; gates in
order; controls that must read known values) · **Machine-checkable core**
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-v7f-design/SPEC.md`
**Design + traceability** `…/2026-09-03-v7f-design/DESIGN.md` · **Every number's source and evidence
class** `…/2026-09-03-v7f-design/raw/EVIDENCE_TABLE.md`
**Register rows to apply** `…/2026-09-03-v7f-design/PROPOSED_REGISTER_ROWS.md`

⛔ **NOTHING HERE LAUNCHES.** Thor is held by refav1's epoch; v7f is a QUEUED run whose slot is a PI
decision. This document is the pre-registration and the decision brief, not a launch request.

---

## 0. ⛔ HEADLINE — six things need the Master Mind / PI, and four of them BLOCK the run

| # | escalation | why it cannot wait |
|---|---|---|
| **E1** | ⛔ **"DINO as init" IS NOT WIRED.** Two probes (`grep` over `stack/`, `tools/`, `taniteval/` and PowerShell `Select-String` over the same trees) find **no loader that puts DINOv3 weights into `ViTEncoder` or `ViT5Encoder`**. DINOv3 appears only as (a) O7's **frozen teacher** (`train_v6_staged.py:931`, `O7_DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"`), (b) REF-A's feature bank (`stack/scripts/dino_precompute.py`), (c) the fp8 shipper (`stack/scripts/dinov3_fp8_encode_ship.py`). `--init-from` loads a **whole-stack** checkpoint and REFUSES a partial one (`train_v6_staged.py:7236-7247`). ⇒ the PI's directive needs **one new artifact** (a seed-checkpoint converter) or **one new flag**. Costed in §10 D1. | it is the run's premise |
| **E2** | ⛔ **BACKLOG R8 is a launch blocker and is still open in the trainer.** The 141 eval clips **with pixels** live inside the 4,713-clip B1 cache, and `build_train_episodes` / `build_v2_providers` carry no exclusion list (`D-V7-WIRING`, hazard found at two probes). A v7f WM objective on B1 would train on the eval split's pixels. Another agent is fixing it tonight; this prereg treats the fix as a **precondition** and states its verification: the window census of a `--require-parity` launch must show **0 eval clips**. | a leaked eval split voids G-DRIVE |
| **E3** | ⛔ **`G-RANK ≥ 8.56` AS WRITTEN IS INADMISSIBLE, AND THE REGISTRY SAYS SO IN CODE.** `MODEL_REGISTRY.md` §13.3 and `stack/tanitad/models/v6.py:1517-1552`: through `spectrum_report` itself, frozen DINOv3 ViT-L/16 reads **5.756** on the 12 physicalai-val clips the 8.56 is sourced to, **20.228 ± 0.327** on the 130-clip corpus the 40.77 is sourced to. *"Neither published number survives."* A participation value is comparable **only at matched CORPUS, matched EPISODE COUNT and matched AMBIENT DIMENSION**, and ⛔ *"DO NOT FAIL AN ARM ON THE PARTICIPATION CLAUSE"* until a matched-d reference exists. The skill's `≥ 8.56` and the registry conflict; **per CLAUDE.md the registry wins.** G-RANK is re-specified in §6.1 as a **matched-reference** gate. | the gate would fail or pass arms on a number no instrument reproduces |
| **E4** | ⚠️ **`--bptt-truncate 15` has no published support.** Every banked primary truncates to a gradient depth of **1–2** (V-JEPA 2-AC `2506.09985` §3.1 differentiates through **ONE** recurrent step, T = 2; What-Drives-Success `2512.24497` §C back-props only through the LAST prediction). 15 is the **refav1** value carried into the v7f flag suggestion (`D-V7-WIRING`) and is **7.5× the largest published depth**. Recommended default **4**, decided by rung R1, not by inheritance. | it is the k = 60 stability decision |
| **E5** | ⛔ **PARAMETER BUDGET IS A PI DECISION.** Config E is **336.5 M** (`HANDOVER_TO_LOCAL_2026-08-15.md:76-79`, INHERITED, verified against the run's own `config.json` 2026-08-14) against the north star's **sub-300 M** (`ROADMAP.md:24`). §10 D5 gives three options with their measured/derived costs and a recommended default. | the register must carry the PI's number |

| **E6** | ⛔ **THE GATE INSTRUMENT HAS NO INTERVAL, AND ITS BAR WOULD FLOAT.** Backlog **L-13**: *"the actdiv probe has no paired episode-cluster bootstrap. A decision statistic without an interval is a gap, not a virtue."* And `PREREG_MM_E19_K60_HORIZON.md:74-84` **DEFECT 1**: the ratio's denominator is the arm's **own** scene spread, so the k = 60 arm's `scene_factor` **1.7126** silently turned a written-down **10×** bar into **17.13×** — *"a criterion whose threshold is a function of the result is not a pre-registration"* — and, sign-blind the other way, an arm that genuinely **doubled** its sensitivity would report **+17 %** and be written up **INERT**. ⇒ before G-ACT can be read, `actdiv_anchored.py` needs a **clip-level bootstrap** and a **`--pinned-denominator <reference arm>`** option. Small, 0 GPU, and it is why §6.3's bar is `rel_pinned`, not `rel_to_scene` | it is the gate that decides the run |

**Two more, non-blocking:** `taniteval/tools/actdiv_anchored.py` and `transition_probe.py` — the
instruments this prereg gates on — are **NEW and wired into nothing** (`mm_e19_read.py`'s actdiv stage
still calls the banked `actdiv_local.py`); and the `m_t` reservation (`D-M_T-SLOT`) is three lines in
`predictor.py` / `metric_dynamics.py` that must land before the geometry freezes.

---

## 1. The hypothesis (register id proposed)

**`H-V7F-1`** — proposed text, for `GOALS_AND_CLAIMS.md`:

> **H-V7F-1 — A DINOv3-INITIALISED TRUNK, TRAINED UNDER A DISTILLATION-ANCHORED LOW-LR SCHEDULE, WITH
> AN ENCODER-SIDE LATENT-DISPLACEMENT ACTION DECODER (LDAD ON `Δz`, TARGETS `(a, κ)`, `v` EXCLUDED) AS
> THE SOLE NEW ANTI-COLLAPSE TERM, IS THE FIRST TanitAD RECIPE THAT MAKES THE PREDICTOR
> ACTION-SENSITIVE AT THE PROGRAMME'S COMMITTED BAR WITHOUT DESTROYING THE TRUNK'S DYNAMIC
> DECODABILITY.** Concretely: at the selected checkpoint the anchored action-divergence read returns
> verdict **SENSITIVE** with `rel_pinned(2σ) ≥ 0.0595` — the denominator **pinned to a named
> one-variable reference arm**, never the arm's own scene spread — while the Observer-Effect monitor's
> frozen-feature linear probe on the dynamic targets holds at **≥ 0.70 × its own step-0 value**; and
> the arm then beats hold-action at T1 on the four families with the paired episode-cluster bootstrap.

**Why the hypothesis is worth a run and is not a bet.** Every TanitAD intervention on P2 has been a
**prediction** lever (k, O14, EMA, τ-ramp) and the field measures that prediction levers leave the
action gap at ≈ 0 while prediction itself improves 4× (ActSWM `2607.26712` Table 5: rollout cosine
0.239 → 0.972 with H 3→32 and multi-step training, **gap stays ≤ 0.002**; the hinge moves it to
**0.760**). The published fixes **partition**: encoder-shaping (LDAD, SMWM, EB-JEPA's IDM) needs a
**trainable trunk**; ACID / AD-JEPA / ActSWM's hinge act on the predictor. **The PI's directive
unlocks exactly the half we have never been able to try.**

⚠️ **And the strongest evidence for the mechanism is ours, from the other side:** freezing the trunk
made the command channels **deader** — `postrain30k_freeze` steer/accel ratio **0.00078** vs the
trainable incumbent's **0.00561** (7.2× lower; paired command-channel factor **0.45 [0.28, 0.72]**),
with its scene spread 3.2× higher (`D-P2-LEAK-AUDIT` §0 finding 8, `H-LEAK-5`). Action sensitivity
behaves like an **encoder-shaped** property on our line, which is `GS-10`'s hypothesis read forwards.

---

## 2. The SPEC block

*(canonical copy: `…/2026-09-03-v7f-design/SPEC.md` §1 — reproduced here so the prereg is
self-contained; if the two ever differ, SPEC.md is the machine-read one and wins)*

```yaml
hypothesis: H-V7F-1
one_variable: trunk_policy        # {frozen | full | anchored | lastk}
held_constant: [corpus, eval_exclusion, labels, nav, seed, steps, batch, window,
                o5_k, bptt_truncate, w_ldad, cond_param, encoder_input]
success: "anchored rel_pinned(2σ) ≥ 0.0595 (denominator PINNED to the named R3
          scratch-trunk reference arm, never the arm's own scene spread) with a
          clip-bootstrap interval excluding 0.0595, AND verdict SENSITIVE, AND
          Observer-Effect monitor ≥ 0.70 × step-0 (CI excluding 0.70), AND
          G-DRIVE cl−ha ADE separated below 0, paired episode-cluster bootstrap,
          with trivial_profile_fraction < 1.0 and baseline_won_frac < 0.7571"
failure: "(a) rel_pinned(2σ) < 0.0595 at every step-stamped checkpoint ⇒ the
          encoder-shaping family is REFUTED for TanitAD on a realised-motion
          action channel; (a-WORSE) rel_pinned(2σ) below the incumbent's 0.0087
          with the interval excluding it ⇒ the trunk change REDUCED action
          sensitivity — reported as G-ACT-WORSE, never as 'inert'; (b) monitor
          < 0.70 × step-0 ⇒ revert to frozen DINOv3 + trained per-token adapter;
          (c) VOID read ⇒ stamped VOID, never 'no difference'"
controls: [constant_only, raw_input_floor, deliberate_regression,
           shuffled_action, endpoint_shuffled, zero_model, c0_identity]
splits: {fit: "B1 train (4,572 − eval index); every hyper-parameter fitted here",
         val: "carved from FIT by clip, episode-disjoint",
         test: "v7.2 labelled eval split (141 clips with pixels); never tuned on"}
```

⛔ **The honest note on `one_variable`.** v7f is a **composition** and a composition is not
attributable. Attribution is carried by the **ladder** (§8): each element earns its place on a
one-variable tiny arm with its own deliberate-regression arm, and only then joins v7f as a
held-constant. `trunk_policy` is the one element that is not already an adopted register decision.
**If a rung is skipped, this prereg is void for the element that skipped it.**

---

## 3. The design, in one paragraph

The operative world model keeps the v7 recipe's settled core (`--o5-form l1 --w-o5 1.0 --w-o6 0.1
--sigreg-subspaces 32 --sigreg-slices 512 --cond-param omega_accel_v --w-o14 1.0 --o14-mode fut
--o14-k 4 --o5-target ema` at τ = 0.996 fixed) and changes **three** things: (1) the **trunk** is
DINOv3 ViT-B/16, **trainable**, with a discriminative learning rate and a **distillation anchor** to
its own frozen copy; (2) a single new **encoder-side** term, **LDAD** — a decoder from
`Δz_t = z_{t+1} − z_t` (two ENCODER outputs) to `(a_long, κ)`, with **`v` excluded at every level**;
(3) the **horizon** keeps the 6 s forward contract (`--o5-k 60`) but the **gradient chain is bounded**
by `--bptt-truncate` at a depth chosen on the tiny rig, not inherited. Everything above the operative
level (tactical meta-action head, strategic subgoal, anchored-diffusion planner in
`UnicycleAccelCurvatureActionSpace`) is the v7r design unchanged. Full traceability, and the
"what we are NOT doing and why" section, is `DESIGN.md`.

---

## 4. Held constant — and the two places it is hard

### 4.1 The register decisions carried unchanged

`--cond-param omega_accel_v` (D2 / E-DEC-65, ADOPTED) · `--o5-target ema` with **fixed** τ = 0.996
(E-DEC-69 adopted; the τ-ramp gate CLOSED NEUTRAL 2026-08-30 and `--ema-decay-ramp` is DROPPED) ·
`--w-o14 1.0 --o14-mode fut --o14-k 4` (E-DEC-67, ABSORBED) · nav at all three layers (PI 2026-08-30,
MANDATORY; preflight refuses a v7 stage without it) · the **do-not-add list**: O1, O2, O3, O7, O8,
O9, O10, O11, O13, PSG all at **zero**.

### 4.2 ⭐ A consequence of the do-not-add list that removes a stated blocker

**BACKLOG R6 (`--bptt-truncate` does not reach the O1 stage-A rolls) DOES NOT BIND on v7f.**
MEASURED from source: `train_v6_staged.py:3691` guards the entire O1 block on
`if w.o1_ctrl or w.o1_fact or w.o1_scene:` — at `--w-o1-ctrl 0 --w-o1-fact 0 --w-o1-scene 0` the six
`rollout_transitions` calls inside `train_stage_a.stage_a_losses` are **never reached**. R6 becomes a
precondition **only if v7f ever turns O1 on**, and this prereg commits it to zero. *(Evidence class:
MEASURED — ours, source read, `train_v6_staged.py:3691`, `train_stage_a.py:274-320`.)*

### 4.3 ⛔ Where "held constant" is genuinely hard — the encoder input

DINOv3's patch embed is **3-channel**; the v7 trainer's default is `--in-channels 9` (a 3-frame
stack). An exact weight transfer therefore wants `--newest-frame-only --in-channels 3`. But that is
**H-RANK-8's own variable** (consecutive latents then share no input frames), so switching it inside
the trunk comparison would make the arms two-variable.

⇒ **The rule this prereg commits to: `encoder_input` is IDENTICAL across every arm of every rung,
including the `frozen` control and the scratch reference.** The tiny ladder therefore pays for **one
extra reference arm** — a scratch-trunk arm at the same 3-channel input — rather than comparing a
3-channel DINO arm against a banked 9-channel incumbent. Recommended value: `--newest-frame-only
--in-channels 3`, because it is the only setting under which "DINO as init" is *exact* and under which
the Observer-Effect step-0 control measures the real DINOv3 (§10 D2).

---

## 5. Controls — every panel, every rung

| control | must read | why it exists here |
|---|---|---|
| **constant-only** | the no-information value **EXACTLY** | four estimator failures in one afternoon, each of which produced a publishable-looking number, were caught only by a control reading the same value as the thing being measured |
| **raw-input floor** (8×8 pixel difference) | LDAD must **beat** it | a `Δz → (a, κ)` decode is a **visual-odometry** task on two consecutive frames. If raw pixel differences decode `(a, κ)` at the same level, LDAD added nothing the input did not carry |
| **shuffled-action** | the no-information value | on the anchored read the null is **MEASURED**, 200 within-window label permutations per axis; the gate is the **ratio** F_sep / p95 so a moving null cannot silently pass an arm |
| **endpoint-shuffled** `Δz′ = z_{π(t)+1} − z_t` | the no-information value | `H-LEAK-3`: this exact control reproduced **102 % / 100 %** of "drift". Any `Δz`-based claim carries it or it is arithmetic |
| **zero-model** (real predictor, zero action on every candidate) | `max|d| == 0.0` exactly | `actdiv_anchored.py` admissibility |
| **C0 identity** (same inputs twice) | `max|Δ| == 0.0` exactly | idem |
| **`n` and `d` printed** | — | `n ≪ d` is underpowered **by construction**, not a negative |
| **deliberate regression** | must **FAIL** its rung's gate | §8; if the gate does not fail it, a PASS means nothing |

⚠️ **A negative from a linear probe is not a negative about learnability.** Every probe states its
function class; where a negative decides something, a nonlinear probe with a **time-shuffled** control
is run beside it.

---

## 6. The gates, IN ORDER — an arm earns the next only by clearing the previous

⚠️ **G-ACT is inserted between G-DECODE and G-DRIVE.** It is not in the skill's table; it is added
because the PI's brief requires the action-sensitivity threshold to be committed in advance, and
because **no measured intervention has ever raised action sensitivity** — which is the single fact
that decides whether v7f is worth its epoch.

### 6.1 G-RANK — no collapse ⛔ RE-SPECIFIED (E3)

| | |
|---|---|
| **statistic** | `participation_ratio` (σ², **never** `effective_rank(σ)` — C132: they disagree up to **141×** and `effective_rank` PASSES a representation with 55 % of its energy in one direction) |
| **read** | **val-side**. The gate's pooled reading comes from the O4-weighted TRAIN stream and runs high (~5.5 vs ~3.4 on the same model, H-RANK-9) |
| **criterion** | ⛔ **NOT the bare 8.56.** The arm's val-side participation must exceed a **frozen-DINOv3 reference measured on the SAME corpus, the SAME episode count and the SAME ambient dimension `d`**, through `spectrum_report` itself, banked in the same run's `raw/`. The reference is a number this prereg *commissions*, not one it quotes |
| **what it refuses** | an arm whose representation has collapsed. It refuses **nothing else** |
| ⛔ **what it does NOT do** | **rank is NECESSARY, NOT SUFFICIENT** (C131). flagship v1-era had the highest rank ever measured in this programme and no environment interpretation — it was conditioned on future GT points and fails closed-loop. **Never pass an arm on rank alone** |
| **evidence for the re-spec** | `MODEL_REGISTRY.md` §13.3; `stack/tanitad/models/v6.py:1517-1552` (`O6_PARTICIPATION_REFERENCES` = {12-clip 5.756, 130-clip 20.228, 130-clip n5617 20.516}); pinned by `stack/tests/test_participation_floor_provenance.py` |

⚠️ **Cost of the re-spec: one 0-GPU measurement** — run `spectrum_report` on frozen DINOv3 ViT-B/16
patch tokens over the v7f arm's own val clips at the arm's own `d`. Until it exists, **G-RANK reports
and does not refuse.**

### 6.2 G-DECODE — the latent carries the environment, and the trunk was not corrupted

Two halves. Both must pass.

**(a) Content (the L2 form, unchanged).** The ego probe (speed / yaw / yaw-rate / `d_ego`) and the
scene probe (`n_agents`, `n_free_cols`, `occ_{l,c,r}`) must beat **BOTH** the raw-pixel floor **AND**
the constant control, paired; detection AP > `prior` and > `pixel`. Every hyper-parameter (ridge λ,
PCA basis) fitted on the FIT split only. `n` and `d` printed. Deliberate-regression arm: **`frzrand`**
(a frozen RANDOM ViT) must NOT clear it.

**(b) ⭐ The Observer-Effect monitor — NEW, and it is what makes a trainable trunk defensible.**

| | |
|---|---|
| **why** | `2602.12218` (banked, read in full): a linear probe on **FROZEN** features reads **ρ 0.91**; the **same trunk full-fine-tuned** reads **ρ 0.05**; last-layer-only fine-tune reads **0.65**. On a kinematic invariant ρ goes **0.94 → −0.03**. The damage is concentrated in the **deep blocks B5–B10** (δ(l) > 0.10, CKA < 0.2) and it destroys **time-varying** invariants while sparing static ones. And the paper's second finding is why the monitor must be a FROZEN-feature probe: an *invasive* probe reports MAPE **18 %** where the frozen read says **140 %** — *competence hallucinated by the probe learning the task* |
| **instrument** | at every step-stamped checkpoint, freeze the trunk **at that step**, extract patch tokens, fit a **LINEAR** ridge on the FIT split only, score on the val split, on **dynamic** targets (speed, yaw-rate, `d_ego`) and, as the contrast the paper predicts, **static** targets (`n_agents`) |
| **controls** | constant-only (exact no-information value), raw-pixel floor, `n`/`d` printed, episode-cluster bootstrap |
| **committed threshold** | `ρ_dynamic(step) / ρ_dynamic(step 0) ≥ **0.70**`, with the paired episode-cluster-bootstrap CI on the ratio **excluding 0.70**. Read at every checkpoint; **the run is stopped at the first checkpoint that fails it twice consecutively** |
| **why 0.70** | it is the midpoint, on the log scale, between the published frozen (0.91) and last-layer (0.65) readings expressed as a fraction of frozen: last-layer = 0.71 × frozen, full FT = 0.055 × frozen. **0.70 is "no worse than the published last-layer fine-tune"** — a bar the field has already shown is reachable, and 12.7× above the naive full fine-tune |
| **its own no-information control** | ⛔ **required**: the same probe on a **time-shuffled** feature stream must read the constant-only value. A monitor that reads high on shuffled features is measuring the probe, not the trunk |
| **its deliberate-regression arm** | the `full` trunk arm (R3). **If the monitor does not trip `full`, the monitor is VOID and `anchored`'s pass means nothing** |
| **what it refuses** | it refuses the RUN, not just the checkpoint: a trunk that has lost its dynamic content cannot be recovered by more steps, and the design reverts to frozen + adapter (§10 D1 option C) |

### 6.3 ⭐ G-ACT — the action-sensitivity gate (the one that decides the run)

| | |
|---|---|
| **instrument** | `taniteval/tools/actdiv_anchored.py` (NEW, 34 unit tests; validated on refav1 where it separated two checkpoints whose **deployed driving is bit-identical on 140/140 windows** by **195×** in κ-axis F_sep) |
| **metric** | the anchored displacement `d_i(a) = ẑ_{t+h}(a_t ← a) − ẑ_{t+h}(a_t ← 0)` — the ActSWM / Delta-JEPA Fig. 6 / AD-JEPA form, so the read is comparable to published effect sizes |
| **population** | the banked `actdiv` windows reproduced VERBATIM: 24 sorted clips of `physicalai-val-0c5f7dac3b11-w120-256x640cyl`, first 60 frames, W = 6 at `range(0, n, n//5)` ⇒ **144 windows**. Horizon **h = 1** (the only trained head, MM-E14) |
| ⛔ **lift** | the **TRAINER's** contract — `_lift3` with the checkpoint's `cond_param` and `v_last / flagship_v15.SPEED_SCALE (10.0)`. ⛔ **NOT** the banked scripts' `v_first / 30.0` (`actdiv_thor.py:47`, `actdiv_local.py:56`), which is `H-LEAK-1` and moves the legacy ratio **3–5×** |
| **candidates** | κ (the corpus' `atan(L·κ)` proxy) and `a_long`, each at ±{0.25, 0.5, 1, 2, 3}·σ of the 144 windows' own final actions, the other channel at 0 ⇒ 20 candidates + the zero anchor |
| **PASS (committed in advance)** | verdict **SENSITIVE**: `F_sep ≥ 5 × p95` of the arm's own 200-permutation label null on **both** axes **AND** Spearman ρ ≥ 0.8 on both axes **AND** `cos(m(+L), m(−L)) < 0` at **every** level on both axes **AND** **`rel_pinned(2σ) ≥ 0.0595`** |
| ⛔ **THE DENOMINATOR IS PINNED — the bar must NOT float with the arm's own scene spread** | `rel_pinned(2σ)` = per-dim RMS of `m(2σ)` **÷ the scene spread of a NAMED one-variable reference arm**, measured on the same 144 windows — **not** the arm's own. *(This is `PREREG_MM_E19_K60_HORIZON.md`'s **DEFECT 1**, applied: the k = 60 arm's `scene_factor` was **1.7126**, so a "10× ratio rise" silently became a **17.13× action rise** — a **71 % harder test than the one written down, knowable only after the arm ran**. ⛔ "A criterion whose threshold is a function of the result is not a pre-registration." And it is sign-blind the other way: an arm that genuinely **doubled** its action sensitivity would report **+17 %** and be written up as INERT.)* **The named reference for v7f is the scratch-trunk arm of rung R3** — the arm v7f is one variable away from |
| **the bar's provenance** | **0.0595 = 10 × the incumbent `postrain30k`'s banked 0.005947** — `PREREG_MM_E19_K60_HORIZON.md:61` **HORIZON-WORKS** (*"h1 ratio rises ≥ 10× (to ≥ 0.06)"*), read at 2σ because the banked instrument's rolled variants differ by ≈ √2 σ per channel |
| **co-primaries reported beside it, never folded away** | (1) the **action side** on its own (`‖m(2σ)‖` per-dim RMS, unnormalised); (2) the arm's **own** `scene_spread` — the scene side carried **73.7 %** of MM-E19's ratio fall, and a single ratio hides which side moved; (3) the **h ≥ 2 floor** (h2/h4 read ~**1e-05** on *every* arm measured so far — k = 8, k = 60 and the incumbent — a horizon-INDEPENDENT defect that must not be re-discovered per arm); (4) the **training-window count** per arm (⛔ `o5_k` also sets `o4_n`: **415,002 → 319,002**, so the k = 60 arm trained on **76.9 %** of the k = 8 arm's windows — *"derived from the one variable"* does **not** mean *"not a confound"*) |
| ⛔ **a WORSE branch, pre-committed** | **G-ACT-WORSE**: `rel_pinned(2σ)` **below** the incumbent's **0.0087** with its interval excluding it ⇒ the trunk change **reduced** action sensitivity — a distinct, reportable outcome, not to be force-fitted onto "INERT". *(MM-E19's `HORIZON-WORKS / PARTIAL / INERT` set admitted no worsening and the arm fell 0.48× — off the table entirely.)* |
| ⛔ **an interval is REQUIRED, and the instrument does not have one yet** | backlog **L-13**: *"the actdiv probe has no paired episode-cluster bootstrap. A decision statistic without an interval is a gap, not a virtue."* ⇒ **precondition:** `actdiv_anchored.py` must emit a **clip-level bootstrap** over its 24 clips (the leak audit's `actdiv` already reports clip-boot CIs, `H-LEAK-5`), and G-ACT is read on the interval, not the point estimate. ⚠️ *λ selected on a point estimate is one of the four 2026-08-22 failures.* |
| **where the field puts it** | ActSWM `2607.26712` Table 5: gap **0.002 → 0.592** (frozen random readout alone) → **0.760** (hinge + readout). AD-JEPA `2608.06706`: post-hoc centring **−0.002…+0.052 → 0.19–0.52**. Delta-JEPA `2606.31232` Table 5, `Δx` from `Δz`: LeWM r 0.765 → Delta-JEPA **0.992**. A 3.3× jump from our best arm is **well inside** what the published fixes produce |
| **where WE are today (MEASURED, the calibration)** | `postrain30k` **0.0087** (SEPARATED-NONMONOTONE) · `k8clip05p30k` **0.0138** (SNM) · `k60clip05p30k` **0.0032** (SNM) · `rdw8p30k` **0.0181** (STRUCTURED-WEAK). **None is material.** The κ axis is a clean linear map with a ~10⁻³ gain; the `a` axis loses sign consistency beyond ~1σ on the three postrain-recipe arms |
| **shuffled-action control — expected reading, committed** | the 200-permutation null on the v7 population reads **median 0.86–0.99, p95 1.01–1.28** in F_sep units (MEASURED on the four local arms). ⛔ If the measured p95 exceeds **2.0** on the v7f population the panel is **VOID** (the null band moved and the ratio is no longer interpretable). On the *realised* read the shuffled-action control must **collapse** the separation — the refav1 reference is 27.90 → ≤ 1.83 |
| **VOID conditions** | C0 ≠ 0, zero-model ≠ 0, `scene_spread ≤ 1e-6`, or all displacements identically 0 |
| **what a FAIL refuses** | it refuses **H-V7F-1**, not the run's other outputs. A FAIL is a **result**: the encoder-shaping family (LDAD / SMWM / EB-JEPA-IDM) is refuted for TanitAD **on a realised-motion action channel**, and the programme's next line is the predictor-side family (ACID, ActSWM's hinge, AD-JEPA's offset head) — which is **frozen-trunk compatible and does not need a trainable-trunk run at all** |
| **the 0-GPU diagnostic that routes a FAIL** | AD-JEPA **post-hoc action-mean centring** (`LAB-ACT-1`) on the same checkpoints: if the *centred* read is ≥ 3× the raw read with a CI excluding 1, the channel is **masked by a common mode**, not dead — and the next arm is the trained-in offset head, not a new trunk |

### 6.4 G-DRIVE — T1, four families, and the refav1 refusals FIRST

| | |
|---|---|
| **tier** | **T1** (the model conditioned on its OWN actions). T0 is never driving performance |
| **corpus** | the **v7.2 labelled eval split** — 141 clips with pixels. ⛔ **NOT val40** (v7 labels on 6 of 40 clips, D-VAL40-NOLABELS): val40 can serve TRAJECTORY/ADE and cannot serve the label-based families |
| **estimator** | **paired episode-cluster bootstrap** (`taniteval/ci.py`, 2,000 draws), `full_set` mean — never `overlapping_holdout_se`, whose central value is a mean-of-split-means and which biases the point estimate by −6.67 % to +11.69 % |
| ⛔ **REPORTED BEFORE ANY FAMILY ROW** | (1) **`trivial_profile_fraction`** — the fraction of windows whose deployed plan is the constant-velocity straight line (`tools/straight_line_probe.py`); (2) **`baseline_won_frac`** and the `cem` share from the planner's own provenance; (3) the **paired bit-identity check** (`tools/paired_dump_compare.py`) against every other arm in the read |
| **refusals** | ⛔ `trivial_profile_fraction == 1.0` ⇒ **the read is VOID**, family rows are not reported. ⛔ two arms bit-identical on the window grid ⇒ **VOID**, stamped as such and **never** reported as "no difference". ⛔ `baseline_won_frac` not below the refav1 reference **0.7571** ⇒ the planner has not departed from the CV baseline and no planning claim is available |
| **controls** | `hold-action` (holds the observed `(a, κ)`) **AND** ⭐ **`ha0`** — the straight line at `v0`, the STRONGEST trivial baseline. *(Root-cause class, RETRACTION_LOG 2026-09-03 #2: a control weaker than a trivial baseline makes any arm look like it plans — the hold-action control drifts 0.12 m even where the human drives straight, so refav1's "lateral gain over hold-action" was an artefact of the control.)* Plus `cl_navshuf` (nav-shuffle) and `copy_detector` |
| **the four families** | **LONGITUDINAL** target-speed accuracy + distance-keeping (headway / time-gap / TTC), **LATERAL** heading, **curvature**, yaw-rate, cross-track, **TACTICAL** manoeuvre-decision quality + goal/anchor selection, **STRATEGIC** decision + route/goal setting. Per-family, never pooled; each with its own CI on the same windows; a family that cannot be computed says so **per family with the reason and the n** |
| **criterion (L4)** | S-curve reproduction **exceeds** the hold-action control; ADE below the **CV floor 0.5352 m**; all four families reported |
| **deliberate-regression arm** | ⭐ **v1.7 itself** — banked 0.9785 open / **0.0430** closed, hold-action **0.0 %**. A harness that does not reproduce that collapse cannot be trusted to detect it in v7f |
| ⚠️ **a known instrument defect that must be repaired first** | `C-REFAV1-KIN-CONTRACT-LAT`: the open-loop replay of the RECORDED `(a, κ)` through the adapter's unicycle misses the human's lateral position by **0.716 m** on curved windows — more than the human's own excursion (0.496 m). Until the contract is repaired and re-verified, ⛔ **no LATERAL row of any T1 read on that adapter is quotable.** This is a refav1-adapter finding; whether the v7 T1 adapter shares the defect is **UNVERIFIED** and must be checked before G-DRIVE, not after |

---

## 7. Checkpoint selection — committed in advance

⛔ **Selection by validation loss picks action-collapsed checkpoints 17 / 36 times** (Dueling WM
`2608.06706`, Freeway; separation collapses **1.28 → 0.002** *while validation loss keeps improving*).
Our arms have been **nrmse-selected** and are exposed to exactly that error.

**v7f's rule — two keys, in this order:**

1. **FILTER on action sensitivity.** Only step-stamped checkpoints whose anchored read (§6.3) is
   verdict **SENSITIVE**, or **STRUCTURED-WEAK with `rel_pinned(2σ)` monotonically rising over the
   last three checkpoints**, are eligible. A checkpoint that is SEPARATED-NONMONOTONE is **never**
   eligible, whatever its loss.
2. **RANK the eligible set by the held-out ONE-STEP embedding loss.** What-Drives-Success
   `2512.24497` §G.3 / Tables 13–16: the metric most correlated with planning success is the
   **one-step** visual-embedding loss (mean −ρ: Push-T 0.86, Wall 0.81, Maze 0.57, Metaworld 0.47);
   unroll metrics at H > 1 win only on Metaworld.
3. **Split discipline.** Both keys are read on the **val** split carved from FIT. ⛔ The scored eval
   split is scored, never selected on.
4. **Step-stamped checkpoints are implemented** (`train_v6_staged.py:6354-6387`); `--no-step-ckpts`
   must NOT be passed, or selection has nothing to select from.

---

## 8. ⭐ THE TINY-RIG VALIDATION PLAN — FIRST, AND BEFORE ANY SCALED COMPUTE

> *"Never validate a design on a full-scale run."* — `TanitAD_ValidateAIDesign` §2

The rig: `v7-tiny` = the **real** trainer, parity corpus, ~17 min/arm on Thor / ~29 min on the dev box
at ~19 M params. ⚠️ **With a DINOv3 ViT-B/16 trunk the arms are no longer ~19 M** — see §10 D3; the
per-arm cost must be **re-measured** with `--dry-run --dry-steps` before the ladder is scheduled, and
quoted as MEASURED, not carried over.

| rung | question | one variable | arms | ⛔ deliberate-regression arm (must FAIL) | gate it must clear |
|---|---|---|---|---|---|
| **R0** ⭐ | Is LDAD **tautological** on our action channel? | *(a probe, not an arm)* | fit `Δz → (a, κ)` at **step 0** on the DINOv3-init trunk, banked features, no training | **constant-only** (must read the no-information value exactly) and the **raw-pixel-difference floor** (LDAD must beat it) | ⛔ **COMMITTED IN ADVANCE: if the LDAD loss is already at the floor at init, the objective is TAUTOLOGICAL on our realised-motion channel (`E-DEC-57` r 0.9664 corpus-wide) — the LDAD line is REFUTED for TanitAD and NO ARM IS SPENT** (GS-2 (2)). If it beats the floor by a non-trivial margin, R2 runs |
| **R1** | Is k = 60 trainable at a bounded gradient depth? | `bptt_truncate` ∈ {0, 2, 4, 15} at `o5_k = 60` | 4 | **`bptt_truncate = 0`** — the full chain, which **diverged at gnorm 2.1e9** and was killed at 9,000. It must diverge again, or the stability gate is void | no `grad_norm` > 1e3 on ≥ 3 of the last 20 rows; `loss_feat_op` last-10-row mean ≤ 1.30 × the best 10-row window (the live drift-check v3.1 guards) |
| **R2** | Does LDAD move action sensitivity, and at what λ? | `w_ldad` (λ) ∈ {0, 1, 10, 50} | 4 | **λ = 0** — Delta-JEPA's own near-collapse arm (`2606.31232` Fig. 3: λ = 0 nearly collapses, λ = 50 best). It must NOT clear G-ACT | **G-ACT** on at least one λ; λ selected on the **FIT-side val** split only. ⚠️ λ is **environment-specific** in the literature (SMWM `2606.20104`: Two-Room 0.1, Reacher 5, Push-T 30, Cube 1, and λ = 0.1 **collapsed** Reacher) — ours must be swept, never assumed |
| **R3** ⭐ | Does a trainable DINO trunk beat a frozen one **without** the Observer Effect? | `trunk_policy` ∈ {frozen, full, anchored, lastk} | 4 | **`full`** — the naive full fine-tune. ⛔ **The Observer-Effect monitor MUST trip it** (`2602.12218`: ρ 0.91 → 0.05). If it does not, the monitor is VOID. **Second regression arm `frozen`**, which must reproduce the dead command channel (steer/accel ratio ≈ 0.00078 against the trainable arm's ≈ 0.00561) | **G-RANK → G-DECODE(a) → G-DECODE(b) → G-ACT**, in order |
| **R4** | Does the composition drive? | *(none — the run)* | v7f | ⭐ **v1.7 itself** on the L4 harness (0.9785 open / 0.0430 closed, hold-action 0.0 %) | **G-DRIVE** |

⚠️ **A hazard that must be held constant across R1 and R3 and is easy to miss:** the window census
depends on `max_horizon`, which is derived from the stage's live loss horizon
(`train_v6_staged.py:3188-3214`, `tanitad/data/_contract.py:120`) — an episode shorter than
`window + max_horizon` contributes **ZERO** windows. Changing `o5_k` therefore changes the **number of
training windows**, which is the L-10 window-count confound. `o5_k` is fixed at 60 for every rung that
is not R1, and R1 reports its per-arm window census beside every number.

---

## 9. The launch line

⛔ **Illustrative and NOT runnable as written.** Three items are unresolved: `--enc-init-from` does
not exist (E1 / §10 D1), the LDAD flag does not exist (§10 D4), and the eval-exclusion flag is the
other agent's deliverable (E2). Paths are Thor-side and must be re-verified on the actual box.

```bash
PYTHONPATH=/workspace/TanitAD/stack python3 stack/scripts/train_v6_staged.py \
  --stage S-S --out /home/nvidia/experiments/v7f-b1-1ep \
  \
  `# ---- corpus + parity + eval exclusion (E2 is a PRECONDITION) ----` \
  --v2-cache /home/nvidia/data/physicalai-b1-w120-256x640cyl \
  --require-parity \
  --exclude-eval-clips <eval index>            `# ⛔ BACKLOG R8 — flag name TBC by its owner` \
  --v2-lru 64 \
  \
  `# ---- labels + nav (mandatory at all three layers) ----` \
  --s2-labels <v72>/labels/s2_labels_v7.2_train.jsonl.gz \
  --w-s2-goal 1.0 \
  --nav-labels <same> --nav-cond \
  \
  `# ---- the trunk: DINOv3 ViT-B/16, TRAINABLE, anchored (§10 D1/D2) ----` \
  --enc-init-from <seed_ckpt with the DINOv3-converted encoder subtree>   `# ⛔ NEW` \
  --newest-frame-only --in-channels 3 \
  --vit5-encoder --n-registers 4 --enc-dim 768 --enc-depth 12 --enc-heads 12 \
  --patch 16 --frame-h 256 --frame-w 640 --projection cylindrical --frame-hfov 120 \
  --trunk-lr-scale 0.1 --trunk-lr-warmup-steps 2000 \                     `# ⛔ NEW (§10 D1)` \
  --w-trunk-anchor 1.0 --trunk-anchor-model facebook/dinov3-vitb16-pretrain-lvd1689m \  `# ⛔ NEW` \
  \
  `# ---- the settled two-term core, carried unchanged ----` \
  --o5-form l1 --w-o5 1.0 --w-o6 0.1 \
  --sigreg-subspaces 32 --sigreg-slices 512 --spectrum-accum 4096 \
  --cond-param omega_accel_v \
  --w-o14 1.0 --o14-mode fut --o14-k 4 \
  --o5-target ema --ema-decay 0.996 \
  \
  `# ---- the ONE new loss term (§10 D4) ----` \
  --w-ldad <lambda from R2> --ldad-target a_kappa --ldad-form delta_z \    `# ⛔ NEW` \
  \
  `# ---- horizon: 6 s forward, bounded gradient (E4 / R1) ----` \
  --o5-k 60 --bptt-truncate 4 --rollout-grad-checkpoint on \
  \
  `# ---- the do-not-add list, explicit at zero ----` \
  --w-o1-ctrl 0 --w-o1-fact 0 --w-o1-scene 0 --w-o2 0 --w-o3 0 \
  --w-o7-distill 0 --w-o8-pixel 0 --w-o9-ema 0 --w-o10-psg 0 \
  --w-o11-cf 0 --w-o13-ego 0 \
  \
  `# ---- run mechanics ----` \
  --steps <one full epoch, D-ONE-EPOCH> --batch 8 --lr 1e-4 --clip 1.0 \
  --seed 0 --save-every 1000 --log-every 50 --print-launch
```

**Flags that already exist and are used verbatim above** (verified in the argparse of
`stack/scripts/train_v6_staged.py`): `--v2-cache --require-parity --v2-lru --s2-labels --w-s2-goal
--nav-labels --nav-cond --newest-frame-only --in-channels --vit5-encoder --n-registers --enc-dim
--enc-depth --enc-heads --patch --frame-h --frame-w --projection --frame-hfov --o5-form --w-o5 --w-o6
--sigreg-subspaces --sigreg-slices --spectrum-accum --cond-param --w-o14 --o14-mode --o14-k
--o5-target --ema-decay --o5-k --bptt-truncate --rollout-grad-checkpoint --w-o1-ctrl --w-o1-fact
--w-o1-scene --w-o2 --w-o3 --w-o7-distill --w-o8-pixel --w-o9-ema --w-o10-psg --w-o11-cf --w-o13-ego
--steps --batch --lr --clip --seed --save-every --log-every --print-launch --init-from --param-budget`.

**Flags that do NOT exist and are proposed here** (each is a DECISION, §10): `--exclude-eval-clips`
(R8's owner names it), `--enc-init-from`, `--trunk-lr-scale`, `--trunk-lr-warmup-steps`,
`--w-trunk-anchor`, `--trunk-anchor-model`, `--w-ldad`, `--ldad-target`, `--ldad-form`.

---

## 10. ⛔ DECISIONS REQUIRED — each with a recommended default and the cost of the alternatives

### D1 — How does DINOv3 become the init? *(BLOCKING; the run's premise)*

| option | what it is | cost | evidence |
|---|---|---|---|
| **A ⭐ RECOMMENDED — faithful trunk + seed checkpoint** | add a `DINOv3` encoder class that wraps HF `DINOv3ViTModel` (already imported by O7, `train_v6_staged.py:955`) so **the real weights load exactly**; a converter script builds a **seed checkpoint** — the full stack, encoder subtree = DINOv3, everything else fresh-init — which `--init-from` then loads with **zero missing keys** | ~1 day of implementation + tests; **no trainer change to the load path** | it is the only route under which *"DINO as init"* is literally true, and the only one where the Observer-Effect step-0 control measures **the published trunk** (ρ 0.91 is a fact about *those* weights) |
| **B — port DINOv3 weights into `ViT5Encoder`** | map DINOv3's LayerNorm(+bias) / biased qkv / its RoPE onto our RMSNorm / no-bias qkv / QK-Norm / joint-APE `Block5` | cheaper to launch, but the mapping is **lossy and unauditable**: `encoder.py:298-317` shows `Block5` is RMSNorm + QK-Norm + LayerScale + GELU with **no biases**; a "port" would silently drop or re-purpose parameters | ⛔ **the step-0 control would no longer be DINOv3**, so the whole Observer-Effect argument evaporates |
| **C — distill-init (today's meaning)** | keep our geometry, transfer via O7-style distillation from the frozen DINOv3 teacher | already implemented (`O7Distill`) | ⛔ but O7 distils into the **4×4 readout cells**, not the trunk, and O7 is on the do-not-add list. And `--init-from` is **NOT a lever (C164)** — `postrain30k` and `splitp30k` share the same `distill_init.pt` and sit 0.47 apart on drift |

**Recommended default: A.** ⚠️ **Its two sub-decisions, both defaults stated:** (i) trunk **ViT-B/16**
(768×12), because that is **the geometry config E already runs** — the swap is parameter-neutral
(≈ 85 M either way); (ii) load **only** the encoder subtree, and record `{"dinov3_model_id",
"sha256", "adapted_keys", "unadapted_keys"}` in the seed checkpoint, so a later audit can tell an
adapted tensor from a fresh one.

### D2 — Encoder input: 3-channel newest-frame, or 9-channel stack? *(BLOCKING for D1-A)*

**Recommended default: `--newest-frame-only --in-channels 3`.** It is the only setting under which the
DINOv3 patch embed transfers **exactly** (no channel tiling, no /3 rescale), it is what our own
DINOv3 pipeline already does (`dino_precompute.py`: *"Encodes the LATEST RGB frame"*), and the
temporal context is already carried by the predictor's `W = 6` window. **Cost:** it changes
`encoder_input` relative to every banked v7-tiny arm, so the ladder pays for **one extra scratch-trunk
reference arm** at the same input (§4.3). **Alternative** (9-channel, patch-embed weights tiled and
divided by 3): keeps the banked arms comparable, but the init is no longer exact and the step-0
Observer-Effect reference is a modified trunk.

### D3 — What size is the tiny rig for the trunk question? *(BLOCKING for the ladder's schedule)*

The skill's rig is ~19 M / ~17 min per arm on Thor. A DINOv3 ViT-B/16 trunk is ~86 M, so an arm that
tests **this** trunk's corruption is **not** a 19 M arm.

**Recommended default: run R3 with the SAME ViT-B/16 trunk v7f will deploy**, and shrink everything
else (predictor, heads) to the tiny geometry. **Reason:** the Observer Effect is a property of *those
weights at that depth* (the damage is concentrated in blocks B5–B10); a ladder on a different trunk
does not transfer, and a rung that does not transfer is not a rung. **Cost:** per-arm time must be
**re-measured** with `--dry-run --dry-steps` and quoted as MEASURED. **Alternative:** a DINOv3-S/16
trunk if one exists — ⚠️ **UNVERIFIED**; the only DINOv3 ids this repo names are
`facebook/dinov3-vitl16-pretrain-lvd1689m` and `facebook/dinov3-vitb16-pretrain-lvd1689m`.

### D4 — Which encoder-shaping term, and where does it attach? *(BLOCKING for R2)*

**Recommended default: LDAD (Delta-JEPA `2606.31232`), on `Δz`, decoding `(a_long, κ)`, `v` excluded.**

| candidate | form | why NOT chosen |
|---|---|---|
| **LDAD / Delta-JEPA** ⭐ | decoder on `Δz_t = z_{t+1} − z_t`, both **encoder** outputs | *chosen* — it is the **only** one of the three whose endpoint shape was **ablated**, and `Δz` beat `concat[z_t, z_{t+1}]` on all four envs (+4.07 / +1.07 / +12.60 / +0.67, Table 2); its λ has a published in-paper optimum (λ = 50, sweep {0…1000}, λ = 0 near-collapse); and its decode-target ablation (Reacher, Table 3: raw action **81.33**, Δjoint 80.47, both 76.40, **Δfinger 64.93**) independently reproduces our own `D-V-EXCLUDED` rule |
| **SMWM** `2606.20104` | inverse MLP on `[z_t; z_{t+1}]` | the **concat** shape — exactly the leak shape `GS-1` was raised about; and its λ is environment-specific over a **300× range** (0.1 → 30) with λ = 0.1 **collapsing** one task |
| **EB-JEPA IDM** `2602.03604` | `MLP(z_t, z_{t+1})` concat | same concat shape; its evidence is an ablation (ω = 0 → **1 ± 1 %**, total collapse) rather than a form comparison |

⛔ **The decode target, and the proof it is not decoding the target from itself.**
Target = `(a_long, κ)`; **`v` is EXCLUDED at every level (level and Δv)** per `D-V-EXCLUDED`.
The input is `Δz`, built from **two ENCODER outputs**, and the encoder's forward signature is
`[B, C, H, W] → [B, N, D]` — **images only**. The conditioning enters at the **predictor**
(`_lift3`, `train_v6_staged.py:3604-3646`), never the trunk. ⇒ the quantity being decoded is **not
among the decoder's inputs' upstream inputs**, which is precisely why `D-V-EXCLUDED` rules `(a, κ)`
clean *for encoder-side decoders* and dirty for predictor-output decoders (O13), whose input already
consumed the action.
⚠️ **And keeping the action out of the trunk is not merely hygiene — it is measured:**
`2606.07687` Table 6, the **action-conditioning paradox**: feeding the action INTO the trunk drops
action-relevant R² **0.26 → −0.32**. This is the published form of our MM-E17/E18 finding.
⚠️ **The residual confound, named:** `(a, κ)` is the ego's **realised motion**, so `Δz → (a, κ)` may
be **visual odometry** rather than action representation. That is not a leak (VO from vision is a
legitimate capability, inference stays vision-only) but it is a confound for the *claim*. Hence the
**three mandatory controls** on every LDAD panel: shuffled-action, **endpoint-shuffled**
`Δz′ = z_{π(t)+1} − z_t` (the `H-LEAK-3` control that reproduced 102 % of "drift"), and the
**raw-pixel-difference floor**. **R0 runs before any arm** and can refute the line at zero cost.

### D5 — ⛔ THE PARAMETER BUDGET *(PI's number; the register must carry it)*

Config E = **336.5 M** (`param_budget` already raised to 350) vs the north star's **sub-300 M**
(*"40× smaller than Alpamayo-1-class VLAs"* — the claim we make publicly).

| option | what changes | params (ESTIMATED, closed-form `12·d²` per block; the real number comes from `--print-launch` / `config.json`) | cost |
|---|---|---|---|
| **A ⭐ RECOMMENDED — predictor 1024×12 → 1024×8** | 4 fewer ModernCausalBlocks | 336.5 M − **≈ 50 M** ⇒ **≈ 286 M**, inside sub-300 M | ⚠️ **the predictor is the capacity that matters on a pretrained trunk**: DINO-world `2507.19468` Table 4 — predictor from scratch 46.9 / 87.1 / 59.4 vs pretrained-and-fine-tuned **59.4 / 93.8 / 68.7**. Cutting depth is the cheapest cut but it is **not free**, and it must be a **pre-registered one-variable arm** on the tiny rig if it is taken |
| **B — keep 336.5 M, raise the north star** | nothing technical | 336.5 M | ⛔ it retires a **public claim**; PI-only |
| **C — shrink the trunk** (ViT-S/16 if it exists) | smaller trunk | ≈ −65 M | ⛔ **the freeze-vs-fine-tune sign is decided by the trunk's strength** (FROST-Drive `2601.03460`: strong 14 B trunk frozen **8.17** beats fine-tuned 8.13; weak ImageNet ViT frozen 7.39 **loses** to fine-tuned 7.79). A weak trunk inverts the design's premise |

**Recommended default: A**, and the register carries `param_budget = 300` with the measured total from
`config.json`, not an estimate.

### D6 — Gradient depth *(recommended default, decided by R1)*

**Recommended default `--bptt-truncate 4`** — 2× the largest published gradient depth (V-JEPA 2-AC's
T = 2 / one recurrent step), against the inherited **15** which is 7.5× it. Forward horizon stays
`--o5-k 60` (the 6 s contract). **Pre-registered fallback:** if `grad_norm > 1e3` on ≥ 3 of the last
20 rows, drop to 2; if a non-finite gradient appears, restart from the last checkpoint under
`--precision fp32` (the refav1 precedent, `C-REFAV1-BF16-OVERFLOW`).

### D7 — Trunk schedule *(recommended default)*

**`--trunk-lr-scale 0.1` with a 2,000-step warmup during which the trunk LR is 0** (the head learns
against a stationary trunk first), and **`--w-trunk-anchor 1.0`** — an MSE anchor from the live trunk's
patch tokens to a **frozen copy of the same DINOv3 weights**. ⚠️ **Explicitly NOT LoRA:** Latent-WAM
`2603.24581` Table 5 makes **Base-LoRA the WORST row in the table (68.5)** — below Small-LoRA 84.7,
Small full 86.3 and Base full **89.3**; and Table 4 says shaping the trunk (distillation *into* it,
89.3) beats attaching frozen features (88.0). ⚠️ **The anchor's own caution:** JEPA-x `2608.24044`'s
`DISTILL` control arm read **0.516** on its drift metric, *worse* than its baseline 0.361 — a
distillation term is not automatically benign, which is exactly why `w_trunk_anchor` is inside the
`anchored` arm and the `full` arm (anchor = 0) is its regression control.

### D8 — Reserve the `m_t` slot *(recommended default: YES, three lines, zero cost)*

Per `D-M_T-SLOT`: keep `cfg.residual = True`; expose `delta` beside `out[k]` in
`OperativePredictor.forward` (`predictor.py:285-289`); keep the `(win_s[:, -1], z_hat)` pairs
(`metric_dynamics.py:279`). It costs nothing now and is expensive after the geometry freezes (the
FS-2 asymmetry).

### D9 — Slot and order *(PI)*

Thor is held by refav1 to ~09-08; the A40 by refcv3 to ~09-06; only the 4060 is free. **R0 is 0-GPU /
minutes and can run tonight on the 4060 without touching either training box.** R1–R3 need a GPU.
⛔ **Nothing in this document is scheduled.**

---

## 11. Preconditions — v7f may not launch until all are TRUE

| # | precondition | verification | owner |
|---|---|---|---|
| 1 | **eval-clip exclusion** in the B1 path (BACKLOG R8) | the window census of a `--require-parity` launch shows **0 eval clips** | the agent fixing it tonight (assumed to land; cite as the precondition, not as done) |
| 2 | **DINOv3 seed checkpoint** exists with recorded provenance | `--init-from` loads with `missing_keys == []`; the seed carries `dinov3_model_id` + `sha256` | D1-A implementer |
| 3 | **G-RANK matched reference** measured | a `spectrum_report` row on frozen DINOv3 at the arm's own corpus / episode count / `d`, banked in `raw/` | 0 GPU |
| 4 | **R0 cleared** (LDAD not tautological) | `raw/ldad_step0.json` with the constant control at the exact no-information value and LDAD above the pixel floor | 0 GPU |
| 5 | **R1, R2, R3 cleared**, each one-variable, each with its regression arm FAILING | the pairwise `--print-launch` diffs + each rung's gate JSON | tiny ladder |
| 6 | **`m_t` slot reserved** (D8) | `out_m[k]` present in the forward output; test pinned | 3 lines |
| 7 | **the T1 lateral kinematic contract** verified on the v7 adapter | recorded `(a, κ)` replayed through the adapter's unicycle reproduces GT laterally | Benchmarks (a refav1 probe is already running) |
| 8 | **`H-V7F-1` applied** to `GOALS_AND_CLAIMS.md` | grep | Master Mind |
| 9 | ⛔ **`actdiv_anchored.py` emits a clip-level bootstrap interval** (backlog **L-13**: *"a decision statistic without an interval is a gap, not a virtue"*) **and a `--pinned-denominator <reference arm>` option** (MM-E19 DEFECT 1) | the gate JSON carries `rel_pinned`, its CI, the arm's own `scene_spread`, the h ≥ 2 floor and the training-window count | ArchInf — small, 0 GPU |

---

## 12. What this pre-registration would look like if it FAILED — committed now, so it cannot be re-read later

- **G-ACT fails at every checkpoint** ⇒ ⭐ this is a **RESULT, not a null run**: the encoder-shaping
  family is refuted for TanitAD on a realised-motion action channel, on the one architecture that
  could host it. The programme's action line moves to the **predictor-side** family (ACID's
  planning-cost IDM, ActSWM's zero-action hinge, AD-JEPA's offset head), **all of which are
  frozen-trunk compatible** — which would also mean the trunk question and the action question are
  independent, and refav1's frozen design stops being a liability.
- **The Observer-Effect monitor trips `anchored` as well as `full`** ⇒ the corruption is not avoidable
  at this LR with this anchor. Revert to **frozen DINOv3 + a trained per-token adapter**
  (`2602.18639`: PointMaze under lighting+colour+geometry shift **0.48 → 0.78**, encoder-agnostic;
  end-to-end without a pretrained trunk 0.26 vs DINOv2 0.86) — a route that costs **0 trunk compute**.
- **G-DRIVE returns VOID** ⇒ report VOID, name the mechanism (trivial profile / bit-identity /
  `baseline_won_frac`), and fix the **planner's cost surface**, which is where refav1's own read
  landed: the flat-in-κ plan is a **cost** property, not a world-model insensitivity
  (`H-REFAV1-LAT-INSENSITIVE` **REFUTED** 2026-09-03, anchored κ response 2,298× / 11.8× its own null,
  perfectly linear and antisymmetric).

---

## 13. Provenance

Every register row, report, RESULT and code line cited here was opened in this session; the
per-number source path and evidence class are in
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-v7f-design/raw/EVIDENCE_TABLE.md`.
⛔ Model facts come from `Project Steering/MODEL_REGISTRY.md` or raw eval JSON only. One conflict was
found and is reported rather than resolved: **the skill's `G-RANK ≥ 8.56` against
`MODEL_REGISTRY.md` §13.3 / `v6.py:1517-1552`, where the registry wins** (§0 E3).
