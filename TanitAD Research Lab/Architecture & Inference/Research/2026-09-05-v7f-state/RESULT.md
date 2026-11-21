# v7f — THE FLAGSHIP: its true state, and the next arm

**ArchInf FlyWheel · 2026-09-05 · 0 GPU (CPU/source only; dev-box 4060 was 100 % util,
5,078/8,188 MiB, saturated by sibling refav1 arms — checked, not assumed).**
Package: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-v7f-state/`

⛔ **Vocabulary, enforced throughout.** **`v7f`** = the flagship, our 4-brain hierarchical
latent world model with our own trained encoder, trainer `stack/scripts/train_v6_staged.py`.
**`v7.2`** = the LABEL VOCABULARY (`s2_labels_v7.2_*.jsonl.gz`, `--nav-from-v7`). They share
three characters and nothing else. Every `v7` hit in `MODEL_REGISTRY.md` is the label
vocabulary — see §1.

---

## 0. THE HEADLINE, stated before the evidence so it cannot be missed

⛔⛔ **`v7f` HAS NEVER BEEN TRAINED. NOT ONCE. THERE IS NO v7f CHECKPOINT, NO v7f RUN
DIRECTORY, AND NO v7f REGISTRY ROW.** Everything the programme calls "the v7 result" is
**v7-tiny** — ~19 M-parameter proxies at 2 k–30 k steps, trained with **every planner
objective at zero**. The flagship is a *design* with a fully-built trainer, not a model.

⇒ The registry's flagship section having no entry newer than 2026-08-09 is **not** a banking
failure. **Nothing ran.** (§1 establishes this at three independent probes.)

⭐ **And the reason is not neglect — it is a binding PI directive**
(`Project Steering/V7_LAUNCH_GATE.md:3`, 2026-08-31, verbatim):

> *"we should not start training of v7 until the remaining problems are solved."*

⇒ **The brief's premise that the tau-ramp arm is the gating experiment is incorrect, and I
am saying so rather than executing it.** `D-EMA-ADOPT` (2026-08-29) gates **how v7f ships**.
`V7_LAUNCH_GATE` (2026-08-31, **two days later**) gates **whether v7f trains at all**. The
later directive dominates. The tau-ramp arm is a *pre-launch prerequisite*, not the gate. §3
gives the ranking by measured effect size, as `M23` requires.

⭐⭐ **AND THE PI'S OBSERVATION IS CONFIRMED FROM THE REGISTER, NOT MERELY ACCEPTED.** He said
the flagship had been *"forgotten completely"*. Three independent facts say he is right, and
that the flagship is **unattended rather than blocked**:

* **The corpus has been ready since 2026-08-30** — built twice, byte-verified, and a completed
  40,284-step refcv3 run has already trained on it.
* **The compute that "committed" Thor finished on 2026-09-04** (`MODEL_REGISTRY.md:2081`,
  `refav1-b1-v72-ep3-speed` COMPLETE at 21,109). Thor holds the 178 GB B1 epcache.
* **`Decisions/2026-09-05-pi-rulings.md` (R1–R7) does not mention v7f, B1 or the epcache at
  all** — a clean negative with a positive control. **No v7f slot has been requested.**

⇒ ⛔ **v7f is not waiting on a corpus and probably not on a GPU. It is waiting on four pieces
of code that do not exist and on someone asking for the slot.** §3.1 names all of them.

---

## 1. P1 — THE FOUR-ROW STATE TABLE

Evidence classes: `MEASURED (ours + artifact)` · `PUBLISHED` · `INHERITED` · `ESTIMATED` ·
`HYPOTHESIS`.

| # | row | what is **MEASURED** | what is **INHERITED / stale** | verdict |
|---|---|---|---|---|
| 1 | **`E-DEC-9b`** — the recipe does not close the self-generated-target problem | ⭐ **The switched-off term is `O7` frozen-teacher distillation, VERIFIED AT SOURCE**: `PREREG_V7F.md:350` sets `--w-o7-distill 0` in its explicit *"do-not-add list"*; `train_v6_staged.py:9320` `default=0.0`; `:6605` builds the head only when `> 0`. O7's pure-form effect is MEASURED: `n_agents` **−1.0407 → +0.3274** (+1.3681, t 12.63, 24/24 eps), `e_dec8a_distill_readout.json`. ⭐ **The O5 target is teacher-forced AT SOURCE** — `train_v6_staged.py:6920` encodes `b["future_frames"]`, the TRUE future, so an action-invariant solution is admissible. Independently re-verified today. | ⚠️ Nothing stale — but the row is **widely misread**. It says *"the term that closed it is off"*; it does **not** say *"turn it on"*. Its own body says the opposite: co-training O7 with O5+O6 gives **−0.2553** vs **+0.3274** pure (E-DEC-9), and `lib:2509.10156` (LayerLock) Table 3 measured a weighted sum collapsing SSv2 **50.1 → 3.7** on **both** schedules. | ⛔ **OPEN — and the remedy is STAGING, not a `w`-sweep.** v7f's launch line already carries the staged form (`--init-encoder-from` + `--w-trunk-anchor 1.0`). The untested part is the **ORDER** (`lib:2302.14138`: +8.9 right order, **15.1 BELOW joint** wrong order). |
| 2 | **`D-CORPUS-B1`** — B1 is the scaled-v7 corpus | ⭐⭐ **THE CORPUS EXISTS, IS BUILT TWICE INDEPENDENTLY, AND IS IN PRODUCTION USE.** `MODEL_REGISTRY.md:2713/:2900`: `physicalai-b1-w120-256x640cyl`, **4,572 train + 141 eval**. ⓘ **4,572 + 141 = 4,713 exactly** — the parity-excluded count (4,719 − 6 val40-overlap), so the `--exclude-parity-overlap` guard demonstrably held. Thor copy **4,713 eps / 177,998,547,213 B**; the A40-pod copy **byte-verified** against a known-good parity payload (`jpeg_buf` md5 `bef95bf18bf6e92d4226f06ddb55adc3`, both sides). MANIFEST sha256 **verified by RECOMPUTATION**, not citation: `5feda062a72a32ad…`. Leak-guarded by `refcv3_make_split.py`, which **refuses if the label sets intersect**. ⭐ **The strongest proof: a completed 40,284-step refcv3 run already trained on it end-to-end.** | ⚠️ The register's *"starts when Thor's GPU releases"* is **STALE by ~6 days**; `Project Steering/B1_TRAINING_PREP.md` (never updated since 2026-08-29) still schedules the build as future work and never mentions 4,713 — **it is the likely source of that stale premise and should be corrected.** ⚠️ **Open, and cheap to pre-check:** the B1 parity key was **derived from the manifest**, never checked against cache bytes (refcv3 ran `require_parity false / checked false / corpus_key null`). | ✅ **NOT a v7f blocker.** Ready and proven. |
| 3 | **`D-EMA-ADOPT`** — ship with EMA, conditional on one tau-ramp arm | ⛔⛔ **THE ARM HAS NEVER RUN.** ⭐ But the **instrument is fully built**: `--ema-decay-ramp {off,cosine}` at `train_v6_staged.py:9300`, tau-at-step at `:1819`, an auditable trace at `:1913`, wired at `:5362/:5715/:7299`, **pinned by 24 tests** in `stack/tests/test_ema_tau_ramp.py` (tracked in git). The baseline it must match, `emao14_30k`, is MEASURED at T1 (`D-T1-V7-READ`). | ⚠️ The brief calls this a **"λ-ramp"**; the register and the code both say **tau (τ)** — the EMA decay. Different object; λ is the LDAD weight. Naming corrected here. | ⛔ **UNMET — and it is a REAL unmet PI condition, correctly identified by the brief.** But it gates *shipping config*, not *training*. See §3 for why it is not the next arm. |
| 4 | **`D-V7-TRUNK-ANCHOR`** — anchor wired, monitored, **+4.25 %** | ⭐ **Wired, refuses without its monitor, costs +4.25 % of the encoder** — as the row states. Re-verified at source today: `build_trunk_anchor(...)` at `train_v6_staged.py:6597`, `EncoderTokenTap` armed only around `v6_loss_step`. | ⛔ **The row names two blockers and I re-probed BOTH. One is now WRONG, one is WORSE.** (i) `--exclude-eval-clips` *"belongs to another deliverable"* is **STALE — it is IMPLEMENTED** (`train_v6_staged.py:2898-3150`) and **defaults to exclusion**; R8 is closed and `BACKLOG.md:130` still lists it as open. (ii) *"DINOv3 ViT-B/16 is not on this box"* understates it: the seed that **is** on the box is **ViT-L/16**, which `build_trunk_anchor` **refuses by design** against a ViT-B/16 trunk. | ✅ **DONE as engineering · ⛔ UNEXERCISABLE as an experiment** — and the reason is a refused size pairing, not merely absent weights. |

### 1.1 How "v7f has never been trained" was established — THREE independent probes

**Absence at one location is not absence.** Three probes, different mechanisms:

1. ⭐ **The registry.** `grep -c -i "v7f" MODEL_REGISTRY.md` = **0**, with **same-breath
   controls that must read non-zero**: `flagship` = **238**, `refcv3` = **31**. All 30 `v7`
   hits are the **v7.2 label vocabulary**. ⚠️ The first run of this probe was **INCONCLUSIVE
   and I discarded it** — `grep -c` returning 0 exits 1, which aborted the `&&` chain before
   the control ran. Re-run with `;` separators.
2. ⭐ **The gate document.** `V7_LAUNCH_GATE.md` exists *because* training has not started,
   carries the binding directive, and its GPU queue lists **only v7-tiny arms**
   (`k60clip05p30k`, `o11p30k`, `postrain30k`).
3. ⭐ **The pre-registration's own launch line.** `PREREG_V7F.md` §9 still contains
   **unfilled placeholders** — `--w-ldad <lambda from R2>` and
   `--steps <one full epoch, D-ONE-EPOCH>`. **A launch command that still has placeholders
   has never been instantiated.**

⚠️ **Mount discipline.** The G: mount flapped repeatedly during this work. Every count above
is paired with a same-breath control and was retried to a clean read. One file
(`V7_RECIPE_AND_SCALEUP.md`) failed 12 consecutive reads; an **interleaved** control
(`V7_LAUNCH_GATE.md`) failed and recovered **together with it**, proving a mount outage rather
than an unreadable file. Code reads were taken from the off-Drive mirror `C:\Users\Admin\tanitad-wt`.

### 1.2 ⛔ WHAT ACTUALLY GATES v7f — the five problems, none of which is EMA or the corpus

From `V7_LAUNCH_GATE.md`, each with its state today:

| gate | state | closes when |
|---|---|---|
| **P1** — no v7 arm has ever beaten its own hold-action control | ⛔ **OPEN. This is the actual problem.** `D-T1-V7-READ`: both 30 k arms lose to hold-action in **every** distance metric; heading MAE ~95 deg = chance. ⭐ The one genuine positive: `copy_detector = CLEAN, echo_index 0.0000` — v7 traded v1.x's **fake skill by echo** for **honest absence of skill**. | a v7 arm beats hold-action at T1 across four families, paired bootstrap, echo control still clean |
| **P2** — the model does not use its actions | ⛔ **OPEN.** (a) horizon: **ANSWERED NO** — MM-E19, k=60 gave **0.50x** against a pre-registered **>= 10x** bar; two interventions have now moved the ratio the WRONG way. (b) **command channel: NEVER TESTED.** (c) untested. (d) **teacher-forced target: untested**, and the only candidate with a published mechanism, magnitude AND fix (`UWM-JEPA` arXiv 2605.25313). | — |
| **P3** — drift, a seed-stable 3.3x effect of unknown cause | ⛔ open | — |
| **P4** — horizon ladder short of the labels | ⛔ **does NOT close.** Reaching the band in *rollout length* is not reaching it in *capability*. Strategic (8–30 s) unreached. | — |
| **P5 / L3** — the predictor adds nothing over the current latent | ⛔ open — *"the dissociation gate, and the actual v7 bar"*. ⭐ L1 and L2 **PASS**: our own trained encoder **beats the frozen DINOv3 teacher we distil from** on `n_agents` (**+0.1220 vs +0.0998**). **The encoder learns the environment.** | `zhat` beats `z_t` paired at abs(t) >= 2.9 while the regression arm still FAILS |

⭐⭐ **P2, P3 and P5 are three instruments on ONE phenomenon**, in the gate's own words:
**the predictor is not transporting the scene, it is restating it.**

---

## 2. P2 — THE GATING EXPERIMENT, and why it is not the tau-ramp

### 2.1 Did the tau-ramp arm ever run? **NO.**

Established above: instrument built (24 tests), arm never launched. It needs a **30 k v7-tiny
arm matched to `emao14_30k`, one variable (the tau schedule)** — and `emao14_30k`'s own T1
raws live at `/home/nvidia/t1dumps/emao14_30k/t1.json`, i.e. **Thor-only**. **Cost ESTIMATED
~8.6 h** (basis: `V7_LAUNCH_GATE`'s queue prices its other 30 k v7-tiny arms at 8.6 h; the
k=60 arm at 22.6 h), and **×2 for the replicate arm `H-ESTIM-SEED-1` requires**.

⛔ **I cannot run it, and the blocker is a PERMISSION, not a shortage.** My brief forbids
touching Thor and `tanitad-refcv3`; the dev-box 4060 was measured at **100 % util,
5,078/8,188 MiB**. ⚠️ **But the programme-level blocker has changed and this matters for the
PI:** `V7_LAUNCH_GATE`'s *"~50 h of committed work on ONE machine"* premise is **stale** —
the job that committed Thor completed 2026-09-04. ⇒ **the tau-ramp arm, A0 and A1 are very
likely runnable now by whoever holds Thor authority.** SPEC: `SPEC_TAU_RAMP_ARM.md`.

### 2.2 `E-DEC-9b`'s switched-off term is **O7**, and turning it on is MEASURED not to be the fix

Verified at source (§1 row 1). The important correction: the row is routinely read as
*"turn O7 on"*. Its own evidence forbids that —

| form of the external target | `n_agents` | source |
|---|---|---|
| **O7 pure** (distillation alone, no O5/O6) | **+0.3274** | E-DEC-8 |
| **O7 co-trained** with O5+O6 (a weighted sum) | **−0.2553** | E-DEC-9 |
| **staged** — distil, then post-train teacher-free | **+0.1327** | E-DEC-12 |
| frozen vs trainable distilled encoder | **+0.3881** vs **−0.0180** | E-DEC-14b-R |

Plus `PUBLISHED`: LayerLock (`lib:2509.10156`) Table 3 — a weighted sum collapsed SSv2
**50.1 → 3.7**, and **both** weight schedules collapsed. ⇒ **a `w`-sweep is measured not to
be the fix.** v7f's launch line already implements the staged form via the seeded encoder +
trunk anchor. **The untested variable is the ORDER of staging**, where the field measures a
**+8.9 vs −15.1** swing.

### 2.3 ⭐⭐ THE RANKING BY MEASURED EFFECT SIZE (the `M23` discipline the brief demands)

| candidate arm | measured effect size backing it | cost | verdict |
|---|---|---|---|
| **tau-ramp** (`D-EMA-ADOPT`) | EMA *itself* is +24.5 % cos_ctr (E-DEC-69). The **ramp over fixed tau** has **no measured effect size at all** — it is a hyper-parameter refinement of an already-adopted lever. | ~8.6 h Thor | ⛔ **not the gate.** A PI-committed prerequisite, run it before shipping. |
| **turn O7 on** (`E-DEC-9b` read naively) | **−0.2553**, i.e. measured **negative** in the co-trained form v7f would use | ~8.6 h | ⛔ **refuted before launch.** Do not run. |
| **staging ORDER** arm | +8.9 vs −15.1 (`PUBLISHED`, `lib:2302.14138`) | ~8.6 h x2 (needs its reversed-order regression arm) | 🟡 real, but downstream of P1/P2 |
| ⭐⭐ **P2(b)+(d): a genuine COMMAND channel with counterfactual targets** | P2(a) is **dead** (0.50x, two interventions the wrong way). (b) and (d) are the only untested causes, and (d) has a published mechanism, magnitude **and** fix. | see §3 | ⭐ **THE GATE.** |

---

## 3. P3 — THE ARM TABLE: costs and committed criteria

⛔ Not a roadmap of intentions. Every row commits **both outcomes before running**.

| # | arm | what it changes | cost | committed criteria (BOTH outcomes) | blocker |
|---|---|---|---|---|---|
| **A0** | ⭐ **`o11` same-clip-negatives control** — the O11 breakout is `H-LEAK-6` *"a class-(i) positive by construction"*; without this control a positive O11 result cannot be distinguished from **scene-matching** | time-shifted negatives from the window's own clip; `--max-horizon = o5_k + s` | ⭐ **NEWLY PRICED TODAY (§4.1): −4.3 % of windows at k=8/s=8, −6.0 % at k=60/s=8.** Compute ~8.6 h | **survives** same-clip negatives ⇒ the discrimination is DYNAMICAL, P2 has its first real lever; **collapses to the floor** ⇒ it was scene-matching and O11 is void | GPU only |
| **A1** | ⭐⭐ **the COMMAND-CHANNEL arm (P2(b))** — replace `--cond-param omega_accel_v` (**MEASURED at source to be `[yaw_rate, a_long, v/SPEED_SCALE]`, the ego's realised motion**) with a true control-space command `(a_lon, a_lat)` — **the parameterisation refcv4b built and validated today** | one variable: the action channel's semantics | ~8.6 h (30 k v7-tiny) | **action/scene ratio rises above the incumbent band 0.00408–0.00595** ⇒ P2(b) is the cause and v7f has never had a command; **stays in band** ⇒ the channel's *semantics* are not the barrier and P2(d) becomes the sole survivor | GPU + a control-space action label per window |
| **A2** | **counterfactual-target arm (P2(d))** — train against counterfactual rather than teacher-forced targets (`UWM-JEPA` 2605.25313, whose abstract states the finding *applies beyond the unitary parameterisation*) | the O5 target construction | ~8.6 h | ratio **>= 10x** the incumbent band (the bar already pre-registered for MM-E19) ⇒ P2 closes; **< 10x** ⇒ three of four P2 causes are dead and (c) latent geometry is forced to the front | ⛔ **needs A1** — you cannot form a counterfactual over a channel that merely restates what happened |
| **A3** | **tau-ramp** (`D-EMA-ADOPT`'s unmet condition) | the tau schedule only | ~8.6 h | **ramp >= fixed on prediction with drift not worse** ⇒ ship the ramp; **else** ship fixed tau as measured. ⛔ Neither outcome reopens the adoption. | GPU (Thor holds the `emao14_30k` baseline) |
| **A4** | **staging-ORDER arm** + its **reversed-order deliberate-regression arm** | order of the staged curriculum | ~17 h (2 arms) | right order **> joint** and reversed order **< joint** ⇒ order is load-bearing; **regression arm does NOT fail** ⇒ the gate is vacuous and the result is void | after P1/P2 move |
| **A5** | **the v7f flagship run itself** | scale-up on B1 | ⛔ unpriced — `PREREG_V7F` §9 still has `<one full epoch, D-ONE-EPOCH>` unfilled | — | ⛔ **the 2026-08-31 PI directive — PLUS four pieces of code that DO NOT EXIST** (§3.1: no DINOv3-init loader, the only seed is a refused ViT-L/16, **LDAD is unimplemented — `grep -ril ldad` → 0 files**, and a wrong flag spelling in the published launch line) |
| **A6** | ⭐ **0-GPU, unblocked, and I DID THE DIAGNOSTIC HALF THIS TURN (§3.2)** — remaining: strike or implement the LDAD triple, add `--horizons 1`, price `D-ONE-EPOCH`, resolve the ViT-B/16-vs-ViT-L/16 seed pairing | none — a document, a flag, and a cost | **0 GPU**, hours | ✅ **diagnostic half DONE**: the launch line is **one loss term and one flag** from argparse-clean, every other flag accepted, and the alleged `--enc-init-from` blocker is **refuted**. Remaining criterion: the corrected line reaches the corpus check | none — the cheapest item on the board |

### 3.1 ⭐ What it would take for v7f to produce a T1 four-family number

⛔ **The answer is NOT an instrument change, and the register already says why.** Per
`D-VAL40-NOLABELS`, the `strategic` family read `UNAVAILABLE` at the first T1 read as a
**CORPUS FACT**: the deployed val40 carries v7 labels on only **6 of its 40 clips**, and those
6 are exactly the ones excluded from B1 training for leakage. *"Just use val40"* is not
available for the label-based families, and **no amount of eval-code work would have resolved it.**

⭐⭐ **But the fix already exists and is in production.** `D-V72-SPLIT` **supersedes
`eval_split_v3`**: the v7.2 split ships as the **data layout** — `s2_labels_v7.2_train.jsonl.gz`
(**4,572 clips**) and `s2_labels_v7.2_eval.jsonl.gz` (**147 clips**), *"the split is now a
property of WHICH FILE YOU OPEN"*. refcv3 and refcv4b already evaluate on it, and refcv3's
registry row is stamped **four families, T1**. ⇒ **v7f needs no new eval substrate.**

⇒ **The complete requirement list for a v7f T1 four-family number, all blockers named.**
⚠️ **This list grew during this work.** My first draft named three blockers; a second probe
found **four more, all of them code that does not exist**. That is exactly why the brief said
*absence found at one location is not absence* — and it applies to blockers too.

**✅ DONE — the data and measurement side is finished:**

| | |
|---|---|
| **Corpus** | ✅ B1, **4,713 clips**, built **twice independently** (Thor `/home/nvidia/data/physicalai-b1-w120-256x640cyl`, **177,998,547,213 B**; and the A40 pod), val40-clean by digest, **byte-verified** against a known-good parity payload (`jpeg_buf` md5 `bef95bf18bf6e92d4226f06ddb55adc3` both sides), and **already trained on end-to-end** by a completed 40,284-step refcv3 run. MANIFEST sha256 **verified by recomputation**, not citation: `5feda062a72a32ad…`. |
| **Labelled eval split** | ✅ v7.2, 147 eval clips, in production (`D-V72-SPLIT`) |
| **Four-family instruments + episode-cluster bootstrap** | ✅ `taniteval` |
| **`--exclude-eval-clips`** | ✅ **IMPLEMENTED** (`train_v6_staged.py:2898-3150`) and it **defaults to exclusion, not to a warning**. ⛔ **This CORRECTS my own §1 row 4 above, which inherited "belongs to another deliverable" from `D-V7-TRUNK-ANCHOR`. R8 is closed; `BACKLOG.md:130` still lists it as an open launch blocker and should be fixed.** |

**⛔ BLOCKING — and the binding constraint is CODE AND A PI SLOT, not compute:**

1. ⛔ **The PI lifts or scopes the 2026-08-31 training gate.** *A PI decision; blocks only this item.*
2. ⛔⛔ **THE TRUNK INIT — the run's own premise — and it is worse than "not wired".** ⭐ The
   converter **does exist**: `stack/scripts/dinov3_seed_checkpoint.py` (**verified by me**, and
   it carries the shape table `(768,12,12,16): "vitb16"` / `(1024,24,16,16): "vitl16"`). But its
   output was then **MEASURED against** — `dino_hf − seed_asis` **+0.2050 [+0.0896, +0.3233],
   SEPARATED** — i.e. the converter route loses a large, separated fraction of DINOv3's scene
   content, and the recommendation reverted to **§10 D1 option A** (wrap the real
   `DINOv3ViTModel`). ⇒ **a converter was shipped and then measured inadequate.**
3. ⛔⛔ **The seed that exists is the WRONG SIZE.** The launch line specifies a **ViT-B/16**
   trunk; the available seed is **ViT-L/16** — a pairing `build_trunk_anchor` **refuses by
   design**, and `dinov3-vitb16` is **not in the local HF cache**. *(R24, and worse than "weights
   absent": the weights that ARE present are refused.)* ⚠️ Consistent with source:
   `train_v6_staged.py:931` `O7_DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"`
   and `:8238` — *"it is `dinov3-vitl16` (1024-d, a DIFFERENT network)"*.
4. ⛔⛔ **THE LDAD LOSS DOES NOT EXIST.** ⭐ **VERIFIED BY ME, not inherited:**
   `grep -ril "ldad" stack/` → **0 files**, with a same-breath control (`trunk_anchor` → **8
   files**). `--w-ldad / --ldad-target / --ldad-form` appear in the prereg's launch line and
   nowhere in the code.
5. ⛔ **The gate instrument cannot yet render a verdict**: `actdiv_anchored.py` needs a
   clip-level bootstrap and a `--pinned-denominator` before G-ACT is readable, and the
   `G-RANK >= 8.56` bar is inadmissible against the registry's own numbers.
6. ⛔ **An open PI decision on param budget** — config E is **336.5 M** against the **sub-300 M**
   north star. Flagship run cost **UNPRICED** (`D-ONE-EPOCH` unfilled).
7. ⭐⭐ **COMPUTE IS PROBABLY *NOT* THE BINDING CONSTRAINT ANY MORE — and this is the single most
   actionable correction in this report.** `V7_LAUNCH_GATE`'s *"Thor is the only compute and it
   is committed"* dates from 2026-08-31 and is **STALE**: `MODEL_REGISTRY.md:2081` records
   **`refav1-b1-v72-ep3-speed` COMPLETE at step 21,109, eval landed 2026-09-04** — and Thor is
   the box holding the 178 GB B1 epcache. ⚠️ **Correctly scoped: I did NOT probe Thor** (out of
   bounds by brief), so its *current* occupancy is **UNVERIFIED**; what is MEASURED is that the
   job which committed it has finished. The A40 (`tanitad-refcv3`) is busy to ~2026-09-06.

### 3.2 ⭐⭐ I RAN THE LAUNCH LINE. EXECUTION FOUND A DEFECT NO BLOCKER LIST HAD.

⭐ **This is the part of the turn that is a measurement rather than a reading.** I executed
`PREREG_V7F` §9's launch line through the **real argparse** (venv `C:/Users/Admin/venvs/tanitad`,
torch 2.11.0+cu128, off-Drive from the mirror, `PYTHONPATH=<mirror>/stack`, `OMP_NUM_THREADS=6`,
**0 GPU**), verifying the imported package with `tanitad.__file__`.
Raw: `raw/v7f_launch_line_dryrun.json`.

**Exactly TWO real defects. Every other flag in the launch line is accepted.**

| # | defect | evidence |
|---|---|---|
| 1 | ⛔ **The LDAD triple does not exist** | `train_v6_staged.py: error: unrecognized arguments: --w-ldad 1.0 --ldad-target a_kappa --ldad-form delta_z` — **and these are the ONLY unrecognized flags in the whole line.** ⭐ **Two independent mechanisms agree**: the real argparse, and `grep -ril "ldad" stack/` → 0 files with a control reading 8. |
| 2 | ⛔⛔ **`--horizons` is NEVER PASSED, and the trainer REFUSES the default — this was on NO prior blocker list, mine included** | `PREREG_V7F` contains **zero** occurrences of `--horizons`, so it defaults to `(1,2,4)`; the trainer refuses: *"declares [2, 4], which NO loss consumes … would take EXACTLY zero gradient and then feed initialisation noise to every probe that reads them — **this has already produced two retracted findings**"*. **Fix: pass `--horizons 1`** and keep the horizon on `--o5-k 60` (= 6.0 s). |

⚠️ **Scope correction, stated rather than buried.** An earlier run of this probe *also* refused
on `--nav-cond` and `--v2-cache`. **Those were artifacts of my truncated reconstruction, not
defects** — §9 does contain `--v2-cache`, `--require-parity`, `--v2-lru`, `--s2-labels`,
`--w-s2-goal`, `--nav-labels` and `--nav-cond` (grep: 2 hits each). I am not counting them.

⚠️ **An argparse trap worth knowing, because it nearly produced the opposite conclusion:**
**argparse reports MISSING-REQUIRED before UNRECOGNIZED.** Omitting `--stage`/`--out` masks the
unrecognized list entirely and reads as *"every flag exists"*. Supply the required args first.

⚠️ **An incidental defect found by running it:** `train_v6_staged.py:10371` prints refusals as
`f"[v6] \u26d4 {p}"`. On a **cp1252** console that raises `UnicodeEncodeError`, so **the trainer
crashes instead of printing the refusal** — hiding exactly the diagnostic a launch needs.
Workaround `PYTHONIOENCODING=utf-8`; durable fix is ASCII refusal text or an explicit UTF-8 stream.

⇒ **A6 is now half-done and much cheaper than it looked: the launch line is ONE loss term and
ONE flag away from argparse-clean.**

⛔ **A REFUTED BLOCKER, removed rather than repeated.** A second probe reported *"a flag-name
error: `--enc-init-from` is not the implemented spelling"*. **It is wrong, and I only found
that because I re-verified a decision-grade claim instead of inheriting it.**
`train_v6_staged.py:8759` reads `ap.add_argument("--init-encoder-from", "--enc-init-from", …)`
— a **registered alias**, added deliberately *"so PREREG_V7F §9's [spelling works]"* (`:8767`).
The source of the error is visible: line **7927** is a docstring calling `--enc-init-from` *"the
pre-registration's draft spelling"*, and reading stopped there, 832 lines before the alias.
⚠️ **This is the `absence-at-one-location` trap in its most expensive form — a false blocker on
a launch line — and it is the reason rule 1 says a claim that decides a GPU-day may never be
INHERITED.**

⭐ **AND ONE FINDING THAT VINDICATES THE PI'S OBSERVATION EXACTLY.** The current
`Project Steering/Decisions/2026-09-05-pi-rulings.md` (R1–R7: route metrics, refcv4b hierarchy,
anchor-bank research, agent fan-out, strategic supervision, DiffusionDriveV2, REF-C RL) **does
not mention v7f, B1 or the epcache at all** — a clean negative with a positive control, not a
mount artifact. ⇒ **no v7f slot has been requested**, while the box holding its corpus has been
free since 2026-09-04. *That is the PI's complaint, confirmed from the register: the flagship
was not blocked, it was unattended.*

⚠️ **n = 147 limits per-competence numbers** (~13 stop-launch, ~24 highway ⇒ per-competence
inadmissible; needs a targeted eval RUN, i.e. GPU time, not more training data).

⚠️ **One residual data-adjacent risk worth pre-checking cheaply.** The B1 parity key registered
in `D-V7-WIRING` (`episode_uid_sha256 e8bfb98e06eb…`) was **derived from the release manifest**,
with verification deferred — *"verified on the cache at the first `--require-parity` launch"*.
The completed refcv3 run recorded `require_parity false`, `checked false`, `corpus_key null`.
⇒ **no launch has yet checked that key against the actual cache bytes**, and a v7f launch with
`--require-parity` would be the first. That is a plausible place for a launch-time surprise and
is cheap to pre-check on the machine that holds the cache.

---

## 4. P4 — WHAT TRANSFERS FROM TODAY'S REFERENCE-ARM RESULTS, AND WHAT DOES NOT

⛔ **Transfer is a claim to be checked, and I checked it at source. The headline is uncomfortable:
NONE of the four levers transfers to v7f as built — and they all fail for ONE shared reason.**

| lever | transfers to v7f? | reason, at source |
|---|---|---|
| **`feasible_decode`** (96.87 % of the fan-to-vocabulary friction gap, 1.2 mm ADE cost) | ⛔ **NO — nothing to project onto** | It is a `PlanConfig` flag over an **anchor vocabulary**: `refc.py:1494` requires *"a UNIFORM prefix grid; anchor_slots ..."*. v7f has **no anchor vocabulary and no control-space decoder** — `train_v6_staged.py` scores **0** for `feasible`, `kamm`, `anchor_meta`. |
| **`PlanConfig.kamm_mu`** | ⛔ **NO** | `refa_v1_plan.py:209-213` clamps a **candidate** into the actuator box using `v0`: `cap = kamm_mu * G / v^2`. v7f emits no control-space candidate to clamp. |
| **top-2 kinematic gate** | ⛔ **NO** | A selection-time rescorer **over candidates**. v7f produces none. |
| **`W_KAPPA`** (large lateral lever) | ⛔ **NO** | A quadratic curvature weight in a **cost surface over candidate paths**. Same missing object. |

⭐ **The shared reason, and it is the finding:** all four are **decoder-side** levers over a
control-space trajectory vocabulary. **v7f has no such decoder.** Its plan path is
`goal_head_tac(z_tac_p, cond=e_g_str)` — a latent goal head. ⇒ **v7f's open defect is
upstream of everything that moved today**, and *"port today's wins to the flagship"* would be
a category error. **The flagship's next arm is not a decoder arm.**

### 4.1 ⭐⭐ BUT ONE THING TRANSFERS, AND IT IS THE MOST VALUABLE THING IN THIS REPORT

**Not a lever — the PARAMETERISATION underneath them.**

`feasible_decode` and `kamm_mu` both presuppose a genuine **control space with units**:
`(a_lon, a_lat)`, a friction coefficient, a `v0`. refcv4b **built and validated exactly that
today** — 117 constant-`(a_lon, a_lat)` controls with curvature derived per window.

Now put that beside two source facts I verified today:

* `train_v6_staged.py:171` — **`omega_accel_v` is *"the MEASURED EGO STATE:
  [yaw_rate, a_long, v/SPEED_SCALE]"`***; the alternative `COND_INCUMBENT` is
  `[atan(L*kappa), a_long, v]`. **Both are restatements of realised motion** (`E-DEC-57`:
  r **0.9988**).
* `V7_LAUNCH_GATE` P2(b): *"our 'action' is realised motion, not a command ⇒ a genuine
  command channel has never been tested"* — state **⛔ NEVER TESTED**.

⇒ ⭐⭐ **refcv4b's anchor vocabulary IS a command channel with units — the exact instrument
P2(b) has been missing.** *"We have never given this model a command, only a description of
what it already did"* stops being a design complaint and becomes a **runnable arm (A1)**,
using an artifact the programme already owns and validated.

⭐⭐ **AND TWO SIBLING RESULTS THAT LANDED WHILE THIS WAS BEING WRITTEN MAKE THE POINT MORE
SHARPLY THAN I DID** (`3f4028d` L4, `d0a5c9c` M31, both 2026-09-05, `--kamm-mu 0.7`):
*"a penalty strong enough to prevent bad behaviour also prevents GOOD behaviour, while a
projection deletes the infeasible turn and leaves the feasible one untouched"*, and
**peak_g MAX 3.262 → 0.707 — `mu` itself, to numerical tolerance**, with the turn decisions
preserved bit-for-bit. Their conclusion: ***"a constraint with units binds exactly where its
physics says it should, which is the signature of a constraint rather than a tuned penalty."***

⛔ **That sentence is the indictment of v7f's action channel, stated in someone else's words.**
A constraint needs **units**; v7f's `omega_accel_v` has no commanded units at all — it is the
ego's realised `[yaw_rate, a_long, v]` played back. ⇒ the reference line has just demonstrated,
on the safety axis, that **the whole class of levers that works requires the very
parameterisation v7f does not have.** That is the strongest available argument for A1, and it
did not come from me.

### 4.2 ⭐ A second, smaller transfer — priced today, 0 GPU, and it corrects the gate doc

`V7_LAUNCH_GATE` leaves the O11 same-clip-negatives control scoped as *"neither is small"*,
and names the thing to measure first: **the dataset's available future-action horizon**.

**MEASURED FROM SOURCE TODAY** (`stack/tanitad/data/_contract.py:135`):

```
"future_actions": ep.actions[t + w : t + w + self.max_horizon]
```

⇒ the horizon is **exactly `max_horizon`** — and `max_horizon` is an **INDEPENDENT FLAG**
(`train_v6_staged.py:6243`, `--max-horizon`), *not* forced to `o5_k`. Its price is windows per
episode: `t_max = frames - window - max_horizon`, i.e. **192.92 − max_horizon** at the live
geometry (T = 198.92 MEASURED two ways, window = 6).

| `o5_k` | shift `s` | `max_horizon` | windows/ep | loss vs baseline |
|---|---|---|---|---|
| 8 | 8 | 16 | 176.92 | **−4.3 %** |
| 8 | 16 | 24 | 168.92 | −8.7 % |
| 60 | 8 | 68 | 124.92 | **−6.0 %** |
| 60 | 16 | 76 | 116.92 | −12.0 % |
| 60 | 60 | 120 | 72.92 | −45.1 % |

⭐ **The cost is LINEAR IN THE SHIFT, not in the horizon.** Only `s = k` is expensive. ⇒
**the gate document's "neither is small" is too pessimistic for the time-shifted option**: a
shift of 8 steps costs **4–6 % of windows at any horizon**, which un-blocks arm **A0** — the
control that decides whether P2's only positive lever was real.

⚠️ **Two honest caveats.** (i) These percentages are **conservative upper bounds**: the true
baseline is `max_h = max(need, plan.maneuver_h)` (`:6243`), which is `>= o5_k`, so the real
relative loss is smaller. (ii) **A small shift is a weak negative** — the action sequence is
autocorrelated, so shift trades window cost against negative strength. The right `s` is an
empirical question; the arithmetic only shows the cost is affordable in the usable range.

---

## 5. ⭐ THE SINGLE NEXT THING THAT WOULD MOVE v7f

> ⭐⭐ **Give v7f a real COMMAND — `(a_lon, a_lat)` from refcv4b's validated control-space
> vocabulary — instead of `omega_accel_v`, which is the ego's own realised motion in different
> units. That single substitution is arm A1; it is the only untested P2 cause with an
> instrument already in hand, it is the precondition for the counterfactual-target arm A2, and
> it attacks P1 — the actual gate — rather than the shipping config.**

⛔ **It is not the tau-ramp**, which has no measured effect size of its own and gates only how
v7f ships. ⛔ **It is not turning O7 on**, which is measured **negative** (−0.2553) in the
co-trained form v7f would use.

⚠️ **What A1 needs, named exactly:** (i) a control-space `(a_lon, a_lat)` label per training
window, derived the way refcv4b already derives it; (ii) **GPU — and this is the correction
that matters: Thor's committing job COMPLETED on 2026-09-04, so the "only machine, fully
committed" premise is stale** (occupancy unverified — confirm before launching); (iii) a
**replicate arm** per `H-ESTIM-SEED-1`, because a separated CI on a single-seed arm is
necessary but **not sufficient** — on this exact rig a zero-lever replicate produced
*"separated"* differences on **3 of 18** metrics.

⭐ **And the honest framing for the PI, since he asked why the flagship was forgotten:** v7f is
**not blocked by data and probably not by compute**. It is blocked by **four pieces of code
that do not exist** (§3.1) and by **nobody having asked for the slot** — `2026-09-05-pi-rulings.md`
does not mention it. The cheapest thing that changes that is **A6**: make the prereg's launch
line actually executable, which is **0 GPU** and unblocked today.
