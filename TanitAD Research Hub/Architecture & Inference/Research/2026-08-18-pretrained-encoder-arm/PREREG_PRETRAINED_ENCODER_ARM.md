# PRE-REGISTRATION — THE **TRAINABLE** PRETRAINED-ENCODER ARM (`E-XENC-1F`)

**Date** 2026-08-18 · **Branch** `agent/arch-inf-20260803` · **Box** dev-box RTX 4060 (Thor untouched)
**The PI's question:** *"Should we investigate a version of our architecture in parallel to v6 with a
pretrained encoder, so we can localize the problem better?"*

**Tier doctrine** every number carries **T0** (WM diagnostic) or **T1** (primary, action-closed loop).
**Estimator** paired episode-cluster bootstrap over the resampling unit named per test (`taniteval/ci.py`).
⛔ **NEVER `overlapping_holdout_se`.** **Ridge** `intercept_col=-1` on every `pc6` fit (C92).

---

## ⭐ THE ANSWER, IN ONE PAGE

**The question splits in two, and the two halves get OPPOSITE answers.**

| the question as asked | verdict |
|---|---|
| ⛔ **"a pretrained-encoder version of v6, run to length and scored on driving"** | **REFUSED.** It is REF-A's question. REF-A ran it (`refa-4brain-speed-30k`, ADE@2s **2.1675** full-set, does not beat CV) and the sibling `E-XENC-1` re-runs it under the v6 trunk at **zero incremental design cost**. A third pass costs **9.16–19.66 GPU-days** (MEASURED bound, §5) and cannot return a fact the first two do not already bracket. |
| ✅ **"a pretrained encoder that our objective is allowed to TRAIN, read out as a step-resolved diagnostic"** | **REGISTERED, CONDITIONALLY** — `E-XENC-1F`, **3 000 steps**, **0.92–1.97 GPU-days** (MEASURED bound). It is the **only cell of the design matrix this programme has never occupied**, and it is the only arm that can separate *"our encoder is badly initialised"* from *"our objective destroys geometry in whatever it is given."* |

**The design matrix, with the empty cell named.** ⭐ **MEASURED (ours + INHERITED from a 5-probe
sweep of `MODEL_REGISTRY.md`, `RETRACTION_LOG.md` and `stack/`): a pretrained-initialised but
TRAINABLE vision encoder has NEVER existed in this programme.** Every pretrained backbone that ever
touched it was frozen or under `no_grad` at every point of contact; every trainable vision encoder
was randomly initialised or warm-started from **TanitAD's own** checkpoint.

| | encoder **randomly** initialised | encoder **pretrained** |
|---|---|---|
| **encoder FROZEN** | *(vacuous)* | **REF-A** ✅ ran → 2.1675 m, plateau · **E-XENC-1** (sibling) ⏳ registered, not run |
| **encoder TRAINED** | **v6 / flagship** ✅ the incumbent | ⭐ **`E-XENC-1F` — EMPTY. This document.** |

⛔ **AND THE ARGUMENT THAT MAKES IT ADMISSIBLE AFTER REF-A'S NEGATIVE:** REF-A and E-XENC-1 are
**structurally unable** to answer the question this arm asks. In both of them the encoder never
trains, so neither can observe what training *does to* pretrained features. That is the C9/C13/C14
class — *an instrument cited for an answer it cannot produce* — and it is why "we already tried a
pretrained encoder" does not dispose of this. §2.

⚠️ **AND THE HONEST COUNTERWEIGHT, REGISTERED BEFORE ANY RESULT — it is heavier than the case for.**
The evidence that motivates the arm (C104's *"91×"*) is **scope-broken, inflated, and now formally
retracted as a comparison** (§1.2): the trivial proxy `v0` **beats DINOv2** on the headline rung
(0.467 vs 0.44997, collapsing DINOv2 to partial r² **0.120**); the two DINOv2 conditions differ by
**2.33× in field of view**; and ⛔ **C122, landed today, measured that REF-A's lead-readout advantage
buys NOTHING on lead-present windows** (contrast **−0.0146 [−0.5988, +0.5551], not separated**).

⇒ **The arm is therefore gated behind THREE conditions (§6), any one of which kills or withdraws it**,
and it is **not** recommended for launch now. ⭐ **The recommendation is: hold it, let the two cheaper
zero-GPU siblings report, and spend nothing until they do.** Refusing an expensive experiment on
evidence is a result.

⏳ **Not a competitor to C122's `E-RECON-2`** (freeze the *flagship's own* encoder, retrain adapter +
4-brain, 5 k first at ~1/6 cost). That arm separates **capacity from representation** inside REF-A's
lineage; this one separates **initialisation from objective** inside v6's. If only one is funded,
⭐ **`E-RECON-2` should go first** — it is cheaper, it is paired against a banked 5 k milestone, and it
retires the 116.9 M confound that currently makes REF-A's negative uninterpretable.

---

## 1. WHAT THE EVIDENCE ACTUALLY SUPPORTS — AND THE TWO WAYS IT IS WEAKER THAN BRIEFED

### 1.1 The indictment as stated

**MEASURED, INHERITED** (`Project Steering/RETRACTION_LOG.md:5487-5503`, C104), through the **same**
deployed `AvgPool2d((4,10))`, on the **same** windows:

| rung | `facebook/dinov2-base` (86.6 M) | ours, `v6F-SW-30k@11250` (87.3 M) | ratio |
|---|---:|---:|---:|
| `lead_gap` r² | **0.44997** | **0.00496** | **91×** |
| `ego_v0` r² | 0.71733 | 0.05240 | 13.7× |
| `lead_closing` r² | 0.01713 | 0.00000 | — |

### 1.2 ⛔ TWO CORRECTIONS THAT MUST TRAVEL WITH IT, OR THE ARM IS BUILT ON A NUMBER THAT MOVED

**(a) The trivial proxy beats the pretrained encoder on the headline rung.** **MEASURED, INHERITED**
(`RETRACTION_LOG.md:5479-5480`): `C-V0` — **ego speed alone, one scalar** — reads `lead_gap`
**0.467**, i.e. *above* DINOv2's 0.44997. And partialling `v0` out collapses DINOv2's 0.45 to
**partial r² 0.120** (`…/Research/2026-08-18-encoder-localisation/ENCODER_LOCALISATION.md:149-151`).
⇒ ⭐ **The honest headline is 0.120 vs our arm's partialled figure, not 0.44997 vs 0.00496.** This is
the C92 class verbatim — *a positive probe quoted without a trivial-proxy control*
(`RETRACTION_LOG.md:4841`) — and every rung in this document is reported **raw AND `v0`-partialled**,
with the partialled column carrying the claim.

**(b) C104's DINOv2 and REF-A's DINOv2 did not read the same tensor.** **MEASURED, INHERITED**
(sibling Stream A, `ENCODER_LOCALISATION.md` §0): C104's condition is **120.0° hfov, 3 sub-frames
concatenated (d 2304), 224×560, 16×40 = 640 tokens**; REF-A's cache is **51.39° hfov, latest frame
only (d 768), 224×224, 16×16 = 256 tokens** (`MODEL_REGISTRY.md:87`, `META.json` verbatim). That is
**2.33× less horizontal field** and **2.5× fewer tokens**. ⇒ *"a pretrained encoder reads 91× better
yet drives 5× worse"* compares two different feature tensors and is **inadmissible as stated** — the
`df` / Thor-`free` / cgroup-`usage_in_bytes` family, a probe reporting the wrong scope.

**(c) ⛔ C122 LANDED WHILE THIS DOCUMENT WAS BEING WRITTEN AND RETRACTS THE FRAMING OUTRIGHT**
(`Project Steering/RETRACTION_LOG.md`, C122, committed `d2ede52b`). *"A pretrained encoder that reads
the scene 91× better produced a model that drives 5.1× worse"* — **both numbers are real, the
comparison is not**, for three reasons established from primary sources:

1. ⛔ **DIFFERENT ARMS.** C104's substrate is *"frozen `v6F-SW-30k` snapshots"* (`MODEL_REGISTRY.md:3615`);
   REF-A's opponent is **`flagship4b-speedjerk-30k` (v1)**. **The 91× and the 5.1× never shared an arm.**
2. ⛔ **DIFFERENT QUANTITIES.** `ego_v0` is **supplied as an INPUT** to both driving arms (3rd action
   channel, `rollout.py:80`), and `lead_gap`/`lead_closing` sit in a family the driving eval
   **explicitly refuses** (`driving.py:608`).
3. ⛔ **DIFFERENT BUDGETS.** MEASURED: flagship trains **277 404 073** params, REF-A **160 514 460
   (57.9 %)** — a **116.9 M gap**.

⚠️ **And my own tier reasoning, drafted before C122, was wrong in the direction I stated.** Both
sides *are* T0 — `rollout.py:144-153` feeds the predictor *"the expert's true future actions"* — but
⛔ **`tier0` in the block name is the METRIC-SUITE tier, a NAME COLLISION**, and §6's rank table
carries **no tier stamp at all**, which is how "REF-A ADE is T1" propagated in the first place.
⇒ the doctrine does **not** forbid the comparison on tier grounds; it fails on (1)–(3) above.
**The tier statement in this document stands only as: both are T0, neither is driving performance.**

### 1.3 What survives, and it is enough to motivate a diagnostic

⭐ **MEASURED, INHERITED** (`RETRACTION_LOG.md:5922`, **C109** — C106 attacked five ways with
refutation as the default posture): **our own encoder's RANDOM INITIALISATION reads `ego_v0` better
than its TRAINED self** — Δr²c **+0.150 [+0.055, +0.226], p(Δ>0) = 1.000**, paired episode-cluster
bootstrap, n = 1 507/1 362 windows in **70 clusters**, positive in **27/27** cells.

⛔ **Quote only that half.** C109 **killed** the same claim on `lead_gap` (**0 of 27** CI-separated,
sign flips in 9/27) and **withdrew the "3.6×" ratio** (its denominator is not separated from noise).
⚠️ And the reframing that matters here: **our trained arm is NOT CI-separated from its own
matched-random null** (`lead_gap` 0/9, `ego_v0` 3/9), **while the random arm IS** (9/9).

⇒ **The one admissible motivating claim:** *on `ego_v0`, training our S-W objective **reduces**
linearly-readable content relative to this exact architecture at initialisation.* Whether that
generalises beyond `ego_v0`, beyond our initialisation, and beyond a near-linear random map is
**exactly what this arm measures** — and nothing currently in flight can measure it.

---

## 2. ⛔ THE MECHANISM BY WHICH THIS DIFFERS FROM REF-A — THE CORE OF THE DELIVERABLE

### 2.1 The four-way separation

| | REF-A (`refa-4brain-speed-30k`) | E-XENC-1 (sibling) | **E-XENC-1F (this doc)** |
|---|---|---|---|
| encoder | DINOv2-B/14, **precomputed cache** | DINOv2-base, **live but `no_grad`** | DINOv2-base, **live and differentiated** |
| encoder trainable params | **0** — features are data tensors on disk (`stack/tanitad/refs/refa.py:12-14`) | **0** in the backbone; 1 770 240 adapter | ⭐ **88 350 720** |
| what else trains | temporal adapter + 4 brains | v6 readout + predictor_op + aux | identical to E-XENC-1 |
| trunk / objective | pre-v6, supervised-head recipe | v6 S-W, label-free O1–O6 | identical to E-XENC-1 |
| geometry | 224×224, patch 14, **256 tokens**, 51.39° | 224×560, patch 14, **640 tokens**, 120.0° | identical to E-XENC-1 |
| **can it observe training's effect on pretrained features?** | ⛔ **NO** | ⛔ **NO** | ✅ **YES — that is the arm** |

### 2.2 ⛔ THE CAPACITY CONFOUND, AND HOW THIS ARM DISSOLVES IT

The brief's warning is correct and it applies to REF-A **and** to E-XENC-1: *a frozen arm losing to a
fully-trained one may be a CAPACITY result rather than an encoder result.*

⭐ **AND AS OF C122 (today) THE CONFOUND IS QUANTIFIED, WHICH MAKES THE ARGUMENT MUCH STRONGER.**
Until this morning REF-A's trainable count was published nowhere — the §6 leaderboard prints `—` in
the Params column for **both** REF-A rows (`:2789`, `:2791`). C122 measured it:

| arm | trainable params | vs flagship |
|---|---:|---:|
| flagship `flagship4b-speedjerk-30k` | **277 404 073** | — |
| REF-A | **160 514 460** | ⛔ **−116 889 613 (57.9 % of flagship)** |

⇒ ⛔ **REF-A trained 42.1 % fewer parameters than the arm it lost to.** Its 5.1× deficit is therefore
**not attributable to the encoder** without disposing of a 116.9 M capacity gap first — which is
precisely the brief's warning, now with a number on it.

⭐⭐ **That is the confound `E-XENC-1F` is built to remove: it is capacity-matched to +0.3821 %.**

⭐ **`E-XENC-1F` removes the confound by construction, and the number is MEASURED BY BUILDING:**

| arm (config E, from the live checkpoint's own `v6_config`) | total params | trainable under `apply_stage_freeze("S-W")` | Δ trainable vs incumbent |
|---|---:|---:|---:|
| **incumbent v6F** | 336 542 025 | **278 993 667** | — |
| E-XENC-1, backbone re-frozen | 337 608 009 | 193 479 171 | **−85 514 496 (−30.65 %)** |
| ⭐ **E-XENC-1F** | 337 608 009 | **280 059 651** | ⭐ **+1 065 984 (+0.3821 %)** |

⇒ ⭐⭐ **`E-XENC-1F` is capacity-matched to the incumbent to within 0.3821 %.** It is a **pure
initialisation swap at matched trainable capacity** — the single cleanest form the question admits,
and neither prior arm has it. **MEASURED (ours)** ·
`raw/xenc1f_param_delta.json` · `code/xenc1f_build_and_count.py`.

### 2.3 ⛔ A TRAP THIS BUILD FOUND, AND A SIBLING CLAIM IT AMENDS

`apply_stage_freeze` sets `requires_grad` from the **group map**, and an external backbone installed
under `encoder` lands in a group S-W trains (`STAGE_GROUPS["S-W"] = ("encoder", "readout",
"predictor_op", "aux")`, `v6.py:3595`). **I independently reproduced the sibling's §2.4 finding:**
after `apply_stage_freeze("S-W")`, the *frozen* arm's backbone reads `requires_grad = True`.

⭐ **BUT `requires_grad` IS A DECLARATION, NOT ARRIVAL — AND MEASURING ARRIVAL AMENDS THE
CONSEQUENCE.** A real forward+backward, counting parameters that actually receive gradient:

| | backbone tensors with **non-zero** grad | numel receiving grad | backbone grad L2 |
|---|---:|---:|---:|
| E-XENC-1 (frozen twin, `no_grad` in forward) | ⭐ **0 of 223** | **0** | **0.0** |
| E-XENC-1F | **222 of 223** | **86 579 712** | **37.0005** |

⇒ ⚠️ **The sibling's stated consequence — *"86 580 480 foreign parameters would have trained
silently"* — does NOT hold for its own build as written**, because that build wraps the backbone in
`torch.no_grad()`; the un-freeze is **inert** there and Adam would see `p.grad is None`. The trap
needs **both** `requires_grad=True` **and** the absence of `no_grad`. ⭐ The sibling's *remedy*
(re-assert the freeze) is still correct as defence-in-depth, and its `C-FREEZE-ASSERT` (assert
`n_trainable == 193 479 171`) is **independently confirmed here**. This is escalated, not filed in a
doc (§10). **MEASURED (ours)** · `raw/xenc1f_param_delta.json` → `grad_reach`.

⭐ The single exception is named rather than rounded away: **`embeddings.mask_token`** (768 elements,
unused on this path) is the one tensor of 223 that receives no gradient
(`raw/xenc1f_throughput.json`).

### 2.4 ⛔ WHY NOT THE OBVIOUS ALTERNATIVE — WARM-STARTING **OUR** ViT-5 FROM DINOv2

This would be the tidier arm: keep our architecture, change only the initial weights. **MEASURED
(ours), by loading both state-dicts and comparing** (`raw/xenc1f_param_delta.json` →
`warmstart_into_our_vit5`):

| | ours (`ViT5Encoder`) | `facebook/dinov2-base` |
|---|---|---|
| tensors | 149 | 223 |
| **key names in common** | ⭐ **0** | — |
| patch embed | `[768, 9, 16, 16]` | `[768, 3, 14, 14]` — **incompatible on two axes** |
| norm | **RMSNorm** (weight only, no re-centering) | **LayerNorm** (weight + bias) |
| qkv bias | **absent** (ViT-5) | **present** |
| QK-Norm | **present** (`q_norm`, `k_norm`) | **absent** |
| positional | joint learnable APE **+ 2D axial RoPE** | learnable APE only |

⇒ ⛔ **A direct warm-start is impossible: zero key correspondence, and the patch embed is
shape-incompatible in both `in_channels` (9 vs 3) and kernel (16 vs 14).** Even the optimistic
shape-only upper bound leaves **23 499 264 of 87 284 736 params (26.9 %)** with no compatible tensor
at all — and the 73.1 % that "match" match by *shape coincidence*, not by role. A surgical transfer
would additionally load LayerNorm-trained blocks into an RMSNorm/QK-Norm/no-bias function: **four
simultaneous changes**, the `--v2` conflation this programme refuses.

⇒ ⭐ **The external-backbone form is the ONLY single-variable way to put a pretrained encoder under
our objective.** That is a design conclusion, and it is measured rather than asserted.

### 2.5 The prior art this arm EXECUTES rather than invents

⭐ A **partially** trainable pretrained encoder was designed on 2026-07-17 and **never run**:
`…/Research/2026-07-17-refa-frozen-encoder-improvement-plan.md:38-42, :74` — *"Rank 3 — (a) Unfreeze
last-k blocks / LoRA-adapt"*, Exp 2 `refa-lora4-30k`, **LoRA r=16 on QKV of the last k blocks**,
k ∈ {2,4,8}, activation-caching at block 12−k, step cost *~1.4–1.8× the frozen arm*, 5 k-step probe
gate. `HYPOTHESIS_LEDGER.md` closed H4 negative with the matching recommendation and the finding
*"the ADAPTER is not the bottleneck, the FEATURES are."*

⚠️ **And LoRA cannot answer THIS question either** — and that is the point. A rank-16 adapter over a
**frozen** backbone adds low-rank deltas; the pretrained weights are still never differentiated, so a
LoRA arm cannot show whether our objective *degrades* them. LoRA answers *"is feature adaptation the
constraint?"*; `E-XENC-1F` answers *"does our objective destroy geometry?"* ⇒ **they are
complementary, and LoRA is the cheaper follow-on IF this arm returns `O-PRESERVE`** (§7).

---

## 3. THE ARM

```
frames [B,9,256,640]
  -> 3 x (3ch sub-frame -> resize 224x560 bilinear+antialias -> ImageNet norm)
  -> facebook/dinov2-base, weight-shared across sub-frames, TRAINABLE, train() mode
  -> drop CLS/register prefix -> [B,640,768] each
  -> concat -> [B,640,2304]
  -> TRAINABLE Linear(2304 -> 768)          768 OUT, NOT 2304
  -> THE SAME SpatialGridReadout -> z_op [B,2048] -> THE SAME predictor, losses, gates
```

⭐ **Everything except the encoder is byte-identical to E-XENC-1**, which is deliberate: the two arms
are a **one-variable pair** (`trainable_backbone` False/True, which controls `requires_grad` **and**
the `no_grad` in the forward — §2.3 shows why both must move together). Forward verified:
`[1,9,256,640] → tokens [1,640,768] → z_op [1,2048]`, `z_op_matches_d_op: true`.

⚠️ **Two declared asymmetries, inherited from E-XENC-1 and NOT introduced here**, both of which the
sibling's own falsifier battery has already bounded: the **3-sub-frame concatenation** (a one-sub-frame
DINOv2 keeps **95.0 %** of the `lead_gap` gap and **98.4 %** of `ego_v0` ⇒ the concat explains
essentially nothing) and the **/14 patch at 224×560** vs our /16 at 256×640.

⚠️ **A trainable backbone runs in `train()`, the frozen twin in `eval()`.** That is required — a
backbone optimised in one mode and evaluated in another is a different function — but it is a second
difference between the twins and is declared here rather than discovered later. DINOv2-base carries
no BatchNorm and dropout 0.0, so the modes are expected to coincide numerically; ⏳ **asserting that
(max abs token delta between modes at fixed input) is a runbook step before step 1**, not an
assumption.

### 3.1 ⛔ LEGAL INSERTION POINT, FROM SOURCE

`STAGE_MAY_INTRODUCE` (`stack/scripts/train_v6_staged.py:350`) and `RESUME_CONTRACT` (`:460`):

| path | trains `encoder`? | admits new keys? | verdict for this arm |
|---|---|---|---|
| `--resume` into the live S-W run | yes | ⛔ no — `load_resume` is hard `strict=True` | ⛔ **forbidden.** Would kill the live 30 k resume |
| S-T (`("cand_score.", "cond_tac_dyn.", "prop_diffusion.", "fallback.", "agent_slots.", "t2_head.")`) | ⛔ **no** — `STAGE_GROUPS["S-T"] = ("layer_tac", "planner")` | yes, but not `encoder.*` | ⛔ the encoder would not train |
| S-S — `STAGE_MAY_INTRODUCE["S-S"] == ()` | ⛔ no | ⛔ no | ⛔ |
| S-J — `STAGE_MAY_INTRODUCE["S-J"] == ()` | yes | ⛔ no | ⛔ swap adds `encoder.backbone.*` keys ⇒ refused |
| `--init-from <live ckpt>` into a fresh S-W | yes | ⛔ no — `load_stage_init` adjudicates against `STAGE_MAY_INTRODUCE["S-W"] == ()` | ⛔ the swap's keys are new ⇒ refused |
| ⭐ **fresh S-W from step 0** | ✅ yes | ✅ **nothing to load, so no allowance is consulted** | ✅ **THE ONLY LEGAL HOME** |

⭐ **STATED HONESTLY, AS THE BRIEF REQUIRES: the only legal insertion point is a FRESH S-W RUN FROM
STEP 0.** There is no warm-start path — the encoder's keys change, and every mechanism that could
admit new keys either excludes S-W (`== ()`) or excludes the encoder group. §5 costs it.

⛔ **AND THE DEFAULT BUILD IS PROVED UNMOVED.** `V6Stack(V6Config())` rebuilt from HEAD:
**87 893 449 params / 405 state-dict keys**, `params_ok: true`, `keys_ok: true` — **MEASURED by
building** (`raw/xenc1f_param_delta.json` → `invariant_default_build`). Nothing in this document
changes a default; the arm is a *construction-time encoder substitution* in a separate run
directory. The live v6F S-W resume (PID 25477, ~step 13 900) is untouched, and **no code in `stack/`
was modified by this stream.**

### 3.2 The frozen-external guard, and why this arm is not caught by it

`declare_frozen_external` / `assert_frozen_external` (`v6.py:3675-3806`) raise
`FrozenExternalViolation` when a **declared** subtree is trainable. The declaration is **opt-in**, so
`E-XENC-1F` simply does not declare — the guard is silent, correctly.

⚠️ **But silence is exactly how the two arms could be confused.** ⇒ **RUNBOOK, binding:** the launch
must assert its arm identity before step 1 —
`n_trainable == 280 059 651` for `E-XENC-1F`, `== 193 479 171` for `E-XENC-1` — **and** assert
`backbone_grad_l2 > 0` for the former after one backward. ⛔ **A `requires_grad` count alone is NOT
sufficient (§2.3): the two arms are indistinguishable by it, and separated by gradient arrival.**

---

## 4. WHAT IS READ OUT — AND WHY IT IS A TRAJECTORY, NOT A SCORE

⭐ **This arm's primary output is NOT driving ADE. It is the step-resolved trajectory of
linear-readout r² from step 0**, on the same rungs C104/C106/C109 used, at steps
**{0, 250, 500, 1 000, 1 500, 2 000, 2 500, 3 000}**.

**Why a short arm suffices — MEASURED, INHERITED, from the programme's own data:**
* C109 (`RETRACTION_LOG.md:5979-5984`): the `z_op` ladder reads `ego_v0` **0.1346 → 0.0801** and
  `lead_gap` **0.0123 → 0.0059** between step **2 000 and 9 000**, *then flat*.
* Stream C §7.6: the trained-vs-init deficit was **already fully established by step 9 250**, and the
  9 250 → 11 250 window moves the rungs by ~2 % of the gap ⇒ the damage is an **early** event.
* C109: **the step-0 → 9 250 sweep is UNAVAILABLE, not merely unretrieved** — a whole-filesystem
  probe of Thor found nothing before ≈ step 9 100. **It needs a new run.**

⇒ ⭐⭐ **This arm PRODUCES the missing early-trajectory measurement, and it produces it on the arm
with by far the most dynamic range.** **MEASURED (sibling's artifact, re-read by me at
`…/2026-08-18-encoder-experiments/raw/falsifier_summary.json`):**

| arm | `lead_gap` r² (mean of 3 proj seeds) | `ego_v0` r² | K1 |
|---|---:|---:|---:|
| `ours` (trained S-W @11 250) | 0.00490 | 0.05207 | **0/3** |
| `randenc` (our arch, random init, 3 seeds) | 0.01597 / 0.01667 / 0.02030 → **≈0.0176** | 0.2011 / 0.1736 / 0.19357 | **0/3** each |
| ⭐ `dino1f` (DINOv2 pretrained, 1 sub-frame) | **0.42743** [0.4111, 0.4400] | 0.70593 | ⭐ **3/3** |

⇒ our own encoder's step-0 headroom is **0.0176**, and C109 killed the `lead_gap` half of that
comparison precisely because the denominator is not separated from noise. **DINOv2's is 0.42743 —
24.3× more** — and it is the **only** arm in the battery that passes K1 at all. A decay measured here
therefore has power the same measurement on our own arm demonstrably lacks.

⛔ **BUT DO NOT ASSUME THIS ARM'S STEP 0 EQUALS 0.42743 — MEASURE IT.** `dino1f` is a **one-sub-frame,
768-wide** readout taken **directly off DINOv2**. This arm's step 0 passes DINOv2's 2 304-dim
3-sub-frame tokens through a **randomly initialised** `Linear(2304→768)` adapter first. A random
linear projection is a known-tolerable operation here (C104/C106 read through a *fixed random
projection* by design), but it is **not the identity**, and assuming the two coincide is exactly the
inherited-number class this programme keeps retracting. ⇒ **the ladder's step-0 point is a
REQUIRED MEASUREMENT, taken before step 1, and it — not `dino1f` — is the denominator of every
subsequent Δ.** If the arm's own step 0 lands below `randenc`'s 0.0176, the arm is **INCONCLUSIVE
before it starts** and must not be run.

⚠️ **The existing spectrum monitor does not cover this, and I checked rather than assumed.** The live
log's `effective_rank` (n = 48) reads **16.75 → 30.06 → 14.01 → 24.20 → … → 19.88 at step 12 600** —
non-monotone, no trajectory. The O6 gate's own criterion says why: *"ADMISSIBLE only at
`rank_ceiling >= 1024` — a single 48-row batch is INCONCLUSIVE by construction"*
(`train_v6_staged.py`, `STAGE_GATE_SPEC`). ⇒ the readout ladder is **not redundant** with it.

### 4.1 Four metric families — per family, never pooled

⛔ **Binding (Sayed 2026-08-02).** ADE stays and these are ADDED. Emitter
`taniteval/taniteval/four_families.py`; **`--lead` is MANDATORY on every arm** or the longitudinal
family's headway/time-gap/TTC return `UNAVAILABLE`.

| family | reported | availability, stated per family with the reason |
|---|---|---|
| **LONGITUDINAL** | target-speed accuracy, distance-keeping (headway / time-gap / min-TTC) | ✅ **with `--lead`**; ⛔ `UNAVAILABLE` without it. **88.7 % of the oracle gap is longitudinal**, and Stream B measures REF-A's deficit as overwhelmingly longitudinal (`speed_mae` Δ **+1.3044**) |
| **LATERAL** | heading error, curvature error, yaw-rate error, cross-track | ✅ all four axes |
| **TACTICAL** | manoeuvre-decision quality, goal/anchor selection | ⚠️ **PARTIAL** — anchor/goal selection is unavailable on a trajectory dump; report what exists with its `n` |
| **STRATEGIC** | strategic decision + route/goal setting | ⛔ **ABSENT, and settled at five independent probes** — PhysicalAI-AV carries no map, lane graph, junction, traffic-light or route signal. Report as absent **with the reason and n = 0**, never silently dropped |

⚠️ **A finding from Stream B that this arm must not overwrite — and I verified every row of it at the
artifact** (`…/Benchmarks & Eval/Implementation/incoming/2026-08-18-refa-reconciliation/raw/refa_vs_flagship_families.json`,
paired episode-cluster bootstrap, 881 windows / 40 episodes, `n_boot` 2 000, **tier T0**):

| family | metric | REF-A | flagship | paired Δ (REF-A − flagship) | separated |
|---|---|---:|---:|---|---|
| LONGITUDINAL | `speed_mae_mps` | 1.7754 | 0.4710 | **+1.3044 [1.1196, 1.4901]** | ✅ |
| LONGITUDINAL | `progress_abs_err_m` | 3.1009 | 0.8370 | +2.2638 [1.8516, 2.6775] | ✅ |
| LATERAL | `lat_abs_2s_m` | 0.5776 | 0.2369 | +0.3407 [0.2032, 0.4930] | ✅ |
| LATERAL | ⭐ `heading_mae_2s_deg` | **5.0346** | 6.6062 | ⭐ **−1.5716 [−7.1424, +2.5419]** — REF-A **nominally better** | ⛔ **NO** |
| LATERAL | `heading_med_2s_deg` | 1.2317 | 1.2742 | −0.0425 [−0.3661, +0.5455] | ⛔ **NO** |
| LATERAL | `curv_sign_agree` (higher better) | 0.8664 | 0.9535 | −0.0870 [−0.1192, −0.0572] | ✅ |

⇒ **REF-A is NOT uniformly worse.** Its deficit is **longitudinal**, and on two of six lateral rows it
is nominally *better* and not separated. ⛔ Any four-family table from this arm must be read against
that, never against "REF-A was worse at everything" — and note that the four *signed* diagnostics are
correctly **refused as paired deltas** in that artifact, because closer-to-zero is not smaller.

### 4.2 ⛔ TIER STAMPS

* the readout ladder and `g_op_fwd_ade_m` → **T0, WM diagnostic. NEVER "driving performance."**
* `ade_0_2s` and every four-family number → **T1 PRIMARY** (`taniteval/tools/t1_eval.py`).
* ⛔ Comparisons across tiers are invalid. The 2.1675-vs-0.4271 pair quoted anywhere in this document
  is **T0** (§1.2c).

---

## 5. COST, MEASURED — AND THE BOUND THAT MAKES THE CHEAP FORM AFFORDABLE

**MEASURED (ours)** · `raw/xenc1f_throughput.json` · dev-box RTX 4060, batch 2, 8 timed iterations
after 3 warm-ups, per-iteration timing (not accumulated), `torch.cuda.synchronize()` around each.

| arm | s/iter (median) | windows/s | peak GiB | ratio vs incumbent |
|---|---:|---:|---:|---:|
| incumbent `ViT5Encoder` | 0.19265 | 10.381 | 0.7709 | 1.0000 |
| E-XENC-1 (frozen) | 0.13326 | 15.008 | 0.4262 | ⭐ **0.6917 — FASTER** |
| **E-XENC-1F (trainable)** | **0.41360** | 4.836 | **2.8821** | **2.1469** |

⛔ **SCOPE, STATED BEFORE THE NUMBER IS USED: this is the ENCODER PATH ONLY** (encoder → readout). A
real S-W step also runs a **189 960 707-param** predictor and every loss. ⇒ the **whole-step** ratio
is **strictly smaller**, and the honest statement is a **bound**, not a point:

> **1.0000 ≤ whole-step ratio ≤ 2.1469** — MEASURED at both ends, not estimated.

**The incumbent's Thor rate, MEASURED BY ME from the run's own log** (`…/2026-08-18-o2-live-and-ridge-reread/raw/v6F-SW-30k_train_log.jsonl`), segmented by producer because **step numbers OVERLAP across machines (C68)** and `step_s` is a **cumulative mean over the process**, so marginals are differenced between rows:

| producer segment | steps | marginal s/step (median) | last-10 median |
|---|---|---:|---:|
| seg 0 (A40) | 50 → 6 300 | 17.2749 | 18.0815 |
| **seg 1 (Thor, live)** | 6 300 → 12 650 | ⭐ **26.3705** | 26.3606 |

⇒ **the cost, as a MEASURED interval:**

| arm length | at ratio 1.0 | at ratio 2.1469 |
|---|---:|---:|
| ⭐ **3 000 steps (the registered diagnostic)** | **21.98 h = 0.92 d** | **47.19 h = 1.97 d** |
| 30 000 steps (the REFUSED performance arm) | 219.75 h = **9.16 d** | 471.79 h = **19.66 d** |

⚠️ **Two limits on the ratio, declared:** it was measured at **batch 2 on an RTX 4060**, and
`CLAUDE.md` records that **Thor's 20 SMs saturate at batch 8** — where the live run sits. A ratio is
the transferable quantity but it is not guaranteed invariant. ⇒ **RUNBOOK, binding: measure the
arm's own marginal s/step between logged rows over steps 100–200 and report it BEFORE committing past
step 500.** ⛔ Do **not** quote `step_s` directly — in this trainer it is already divided by the steps
*this process* ran (`train_v6_staged.py:3034-3039`), i.e. a cumulative mean, inflated by startup and
blind to drift.

⚠️ **`--v2-lru`, not dataloader workers, is the host-RAM knob on this trainer** (`CLAUDE.md`,
corrected 2026-08-16): `train_v6_staged.py` spawns **zero** DataLoader workers. Memory headroom for
the extra 2.5 GiB of device activations is a **device** question and only
`torch.cuda.max_memory_allocated()` is admissible for it on Thor.

⛔ **Provisioning is a PI item and is not assumed.** Thor is running a 30 k S-W that resumes
tensor-strict (~4.7 days remaining) and **nothing in this document may be launched on it**, nor may
any GPU/RAM load be added to it. The arm is queued **behind** that run, or onto a pod the PI
provisions.

---

## 6. ⛔ THE TWO GATES — WHICH SIBLING OUTCOMES MAKE THIS ARM WORTH RUNNING, AND WHICH KILL IT

**Committed in advance. All three must pass. None is this stream's to decide.**

### ⛔ GATE-0 — "READABLE ≠ USABLE" HAS ALREADY BEEN TESTED ONCE, AND IT CAME BACK NEGATIVE

⭐ **C122 ran the substantive test this arm's whole premise depends on, and I must register it as a
prior against my own arm rather than discover it afterwards.** REF-A — the arm whose frozen DINOv2
*reads* lead geometry far better than our encoder — was scored on lead-present vs lead-absent windows
(the `driving.py:608` refusal turned out to be a **stale blocker**; `lead_source.py` and a val40 lead
block attach row-for-row, **881 = 881**, speed corr 1.0):

| | deficit vs flagship |
|---|---:|
| **LEAD** windows | +1.7150 m |
| **NO_LEAD** windows | +1.7295 m |
| ⛔ **contrast** | **−0.0146 [−0.5988, +0.5551] — NOT SEPARATED** |

⇒ ⛔ **The lead-readout advantage buys NOTHING where a lead vehicle is actually present**, and it is
not a speed confound (not separated in any of three speed bands).

⚠️ **What this does and does not license.** C122 states its power as a bound: **a lead-presence
benefit larger than 23–39 % of the deficit is excluded; smaller ones are not.** And REF-A carries the
116.9 M capacity gap and the 51.39° field. So GATE-0 does **not** kill this arm outright — but it
**shifts the prior hard toward `O-IRRELEVANT`** (§7), and it means:

⛔ **BINDING ON THIS DOCUMENT: `E-XENC-1F`'s readout ladder measures a quantity that has ALREADY been
shown, once, not to transfer to driving.** Any launch request must carry that sentence. If a second
independent lead-window test also returns null, **the arm is withdrawn regardless of GATE-1 and
GATE-2** — because then the trajectory it measures is a trajectory of something that does not drive.

### GATE-1 — Stream A (`E-GEOM`, zero-GPU, `…/2026-08-18-encoder-localisation/`)

Its 2×2 FOV × temporal ablation reads `lead_gap` partial-`v0` on the paired `refa1f − wide3f` delta.

| Stream A returns | consequence for `E-XENC-1F` |
|---|---|
| **G-INPUT** — `refa1f` loses ≥ 50 % of `wide3f`'s partial r², CI excluding 0 | ⛔ **KILLED.** The **input pipeline**, not the encoder, is the dominant term; C104's ratio does not describe REF-A at all. The next experiment is the **field of view**, which is far cheaper than any encoder arm. |
| ✅ **G-ENCODER** — `refa1f` retains ≥ 80 %, CI containing 0 | ✅ **PROCEEDS.** The input is exonerated and the loss really is at or below the encoder. |
| **G-MIXED** | ⏸ **HELD.** Decompose per factor first; no launch. |

### GATE-2 — Stream C (`E-XENC-1`, `…/2026-08-18-encoder-experiments/`)

| Stream C's E-XENC-1 returns | consequence for `E-XENC-1F` |
|---|---|
| **DROPPED** on its readout gates — `lead_gap` r² < 0.10 **or** partial-`v0` < 0.03 | ⛔ **KILLED.** If DINOv2's geometry does not survive our pool even when frozen, there is nothing for training to preserve or destroy. |
| **DROPPED** on **no-harm** (WM objective degraded) | ⛔ **KILLED.** The swap harms the objective before training the backbone is even in question. |
| **INCONCLUSIVE** — positive control did not fire | ⏸ **HELD.** No reading is licensed from an instrument that did not demonstrate it can detect. |
| ✅ **PROCEEDS** | ✅ **PROCEEDS** — including, importantly, the case where E-XENC-1 passes the readout gates **and lands at REF-A's ~2.1 m plateau**. That combination is the *most* informative state for this arm, because it means the tokens carry geometry the trunk cannot use, and the open question becomes what training does to it. |

⚠️ **A gate this stream imposes on itself:** if, when both gates pass, `E-XENC-1`'s own `lead_gap`
partial-`v0` figure has **not** cleared its 0.03 threshold by a margin exceeding its CI width, this
arm is **HELD** regardless. A trajectory measured on a rung that is itself at the noise floor is the
C109 error — *a ratio whose denominator is not separated from noise* — repeated one level up.

---

## 7. OUTCOMES, COMMITTED IN ADVANCE — ALL THREE REDIRECT THE PROGRAMME DIFFERENTLY

Read on **`lead_gap` and `ego_v0`, partial-`v0`**, paired episode-cluster bootstrap over the probe
corpus's **70 eval clusters**, at each ladder step against the arm's **own step 0**.

| # | outcome | criterion | what the programme does |
|---|---|---|---|
| **O-DESTROY** | our objective actively removes pretrained geometry | partial-`v0` r² falls by **≥ 50 %** from step 0 by step 3 000, paired CI excluding 0, on **≥ 1 of the two rungs**, with the positive control firing | ⭐⭐ **The OBJECTIVE is indicted, not the encoder. Kills the encoder-swap line as a FIX** (E-XENC-1 included) and redirects to objective repair: O3 mask weighting, SIGReg `free_dims`, the LayerScale schedule, the rank-collapse co-symptom. **This is the highest-value outcome and it is currently unreachable by any live stream.** |
| **O-PRESERVE** | the objective is compatible with pretrained geometry | partial-`v0` r² stays within **±25 %** of step 0 at step 3 000, CI containing 0, **AND** the WM objective is no worse than §7.1's band | ⇒ our random initialisation simply never reaches that basin. **Warm-starting becomes the fix**, and the cheap `refa-lora4`-style last-k arm (§2.5) becomes the right product path rather than a full fine-tune. |
| **O-IRRELEVANT** ⭐ **now the PRIOR-FAVOURED outcome, per GATE-0** | geometry is not the binding constraint | r² preserved (as O-PRESERVE) **AND** T1 `ade_0_2s` not better than the incumbent at matched step, paired CI containing 0 | ⭐ **Stops the encoder line entirely.** Token geometry is present, preserved, and does not move driving ⇒ attention moves to the predictor and the planner. ⚠️ **C122 already produced one measurement of exactly this shape** (lead-window contrast −0.0146, not separated). ⇒ **if this is the outcome, the arm will have CONFIRMED a null that was already half-established — which is a real but low-value result, and it is the honest reason the arm sits behind three gates rather than being launched now.** |
| **INCONCLUSIVE** | — | the positive control does not fire, **or** the trivial-proxy column is missing, **or** `backbone_grad_l2 == 0` at any ladder step | **No reading.** Report the failure and the cause; do not convert it into a weak conclusion. |

⛔ **The kill criterion for the arm itself:** if by step **1 000** the arm has neither moved the rungs
by more than the step-0 seed spread **nor** stayed inside §7.1's no-harm band, **stop at 1 000 and
report** — ~0.3–0.7 GPU-days spent instead of 2. Registered so that stopping early is the pre-agreed
action rather than a judgement call made under sunk cost.

### 7.1 ⛔ NO-HARM ON THE WM OBJECTIVE — NOT A TRADE-OFF CLAUSE

⚠️ The brief's warning is the operative one: R1's flagship citation showed an isolated intervention
costing **~10 points on IN1K and SSv2** while gaining on dense tasks. ⇒ **this arm is not permitted a
"we lost WM loss but gained geometry" defence.** A geometry gain bought with WM-objective harm is
**DROPPED**, exactly as Stream C drops E-XENC-2 on its `w_distill = 0` twin regardless of effect size.

⭐ **And the comparator costs ZERO GPU, because the incumbent already ran these steps and logged
them.** **MEASURED (ours)** · `raw/incumbent_noharm_band.json`, computed from the live run's own
train log, producer segment 0 (steps 50–6 300):

| window | `o1_factual_ade` median | min | max | p90 | n rows |
|---|---:|---:|---:|---:|---:|
| 0–1 000 | 1.80233 | 0.43794 | 6.14540 | 5.37703 | 20 |
| 1 000–2 000 | 0.67239 | 0.25215 | 2.82245 | 2.30330 | 20 |
| **2 000–3 000** | **0.60899** | 0.21544 | **11.00753** | 0.92002 | 20 |

⛔ **A TRAP THE BAND ITSELF EXPOSES, and it would have produced a false verdict:** the incumbent's own
trajectory is **spiky** — over steps 2 550–3 000, `o1_factual_ade` spans **0.31392 to 11.00753, a 35×
range across 10 logged rows** (step 3 000 alone logs `loss` 9.0975 against ~2.03 at step 2 000).
⇒ **the no-harm test MUST be a WINDOW statistic with a ROBUST reducer (median over ≥ 20 logged rows),
never a single-step comparison.** A point comparison against this reference manufactures a verdict in
either direction depending only on which row it lands on. Same family as the `df` / cgroup traps: a
statistic read at the wrong scope.

**The registered no-harm criterion:** the arm's median `o1_factual_ade`, `o3_loss`, `o5_loss` and
`o6_sigreg` over each 1 000-step window must lie **within the incumbent's [min, p90] band** for the
matched window. Outside it on any of the four ⇒ **DROPPED**, whatever the readout ladder shows.

---

## 8. CONTROLS — POSITIVE **AND** TRIVIAL-PROXY, PER ARM

⛔ **C92:** a headline died because a readout echoed **ego speed**. ⛔ **C109:** a cited positive
control (`PC-2OBJ`) was **inert by construction** at the deployed pooling ratio — opposing plants in
one cell cancel, and at p40 it reproduced the un-planted arm to **5e-05**.

| control | per arm, at every ladder step | fires when |
|---|---|---|
| ⭐ **`PC-LOCAL` / `PC-DIST`** (positive) | required | banked **0.0596 → 1.0000, K1 9/9**. ⛔ **`PC-2OBJ` is INADMISSIBLE here** — inert at the deployed pooling ratio |
| ⭐ **`C-V0`** (trivial proxy) | required | **every rung reported raw AND `v0`-partialled; the partialled column carries the claim.** Bar to clear: ego speed alone reads `lead_gap` **0.467** |
| **`C-GRADREACH`** (new, required) | required | after one backward, `backbone_grad_l2 > 0` and the set of gradient-receiving params **== `encoder.*`**. **MEASURED at build: 222/223 tensors, 86 579 712 params, L2 37.0005** — and **0 / 0 / 0.0** on the frozen twin, which is what proves the pair is a real one-variable ablation |
| **`C-ARM-IDENTITY`** (new, required) | before step 1 | `n_trainable == 280 059 651`. ⛔ Not sufficient alone — §2.3 |
| **`C-INIT`** (the BASELINE-OF-SELF rule) | required | the ladder's step 0 **is** the control: the arm is compared to **itself at initialisation**, which is precisely the control whose absence produced C106 |
| **`C-RANDBACKBONE`** | required | the same arm over a **randomly-initialised** DINOv2, trained identically. If it tracks the pretrained arm's trajectory, the result is about **training dynamics**, not about pretrained content |
| **`K1` degeneracy + `pred_sd/gt_sd`** | reported every step | ⚠️ Stream C measured **K1 0/3 for our arm** and `pred_sd/gt_sd` **0.0147** against a `SD_RATIO_FLAT_FLOOR` of 0.05 ⇒ **our incumbent is itself a flat line on these rungs.** Any r² comparison against it must carry that, and a K1 FAIL means the row is not quotable as latent-attributable |

---

## 9. PARITY, AND WHAT THE GEOMETRY MISMATCH COSTS

⛔ **Parity is sacred.** Canonical train corpus `physicalai-train-e438721ae894`, **2 376 episodes**,
skip-hash **`f09e44db`**. **How this arm preserves it:** the substitution is *inside the model*, at
`V6Stack.encoder`. Episode selection, the window builder, the loader and the skip list are all
**upstream and untouched** — the arm consumes the identical corpus through the identical path, and
the launch asserts the parity key before step 1 exactly as every other arm does. Nothing here
re-selects episodes; anything that did would be refused.

⛔ **REF-A's frozen-DINO feature cache is UNUSABLE for this arm, and the reason is geometric.**
`META.json` verbatim (`MODEL_REGISTRY.md:87`): `{"encoder":"dinov2-b14","size":224,"grid":"16x16",
"dim":768}` ⇒ **256 tokens at 224×224, 51.39° hfov**. This arm needs **640 tokens at 224×560,
120.0° hfov**. ⇒ **a 2.5× token-count and 2.33× field mismatch: the cache cannot be reused, and no
resampling of it recovers the missing field.** What that costs:

1. ⛔ **No feature cache is possible at all** — a fine-tuned backbone changes every step, so its
   features cannot be precomputed. That is the arm's real cost driver and it is why the encoder-path
   ratio is **2.1469×** while E-XENC-1's is **0.6917×**.
2. ⛔ **Do NOT pre-cache the teacher-side tokens either**: a 640×2304 fp16 token map is **2.95 MB per
   frame** ⇒ hundreds of GB against a `/workspace` MooseFS quota that `df` does not show.
3. ⚠️ ⇒ **This arm's numbers are NOT comparable to REF-A's**, and the document must never place them
   in one column. The comparison that is valid is **`E-XENC-1F` vs `E-XENC-1` vs the incumbent**, all
   three at the identical 640-token/120° geometry.

**DINOv3.** `facebook/dinov3-vitb16-pretrain-lvd1689m` is **gated `manual`**; the token gets **403**
on weights. ⛔ **No mirror, no bypass** — it is a PI licence item. The arm is designed against
**DINOv2-B/14** and DINOv3 is named as the **upgrade path**: it is **/16**, so it reproduces our
16×40 grid **with no resize at all**, removing the last declared geometry asymmetry.
⚠️ **C117 applies to the absence claim, not to the design:** the 403 was observed at *three probes*,
and *three probes of the same shape are one probe*. ⇒ the licensed statement is **"the weights
endpoint returns 403 to this token"**, not "DINOv3 is unobtainable." Re-probing with a different
shape (a differently-scoped token, or the PI's own account) is a **PI item**, not evidence this
stream may manufacture.

---

## 10. ESCALATIONS — RAISED TO THE ORCHESTRATOR, NOT LEFT IN THIS DOC

⛔ *"An orthogonality instrument sat unmerged for 10 days because the request lived in a README nobody
re-read."*

1. ⭐ **The sibling's §2.4 consequence needs amending.** *"86 580 480 foreign params would have
   trained silently"* does **not** hold for its own build, which wraps the backbone in
   `torch.no_grad()` — **MEASURED: 0 of 223 tensors receive gradient, backbone grad L2 = 0.0.** The
   *remedy* stands; the *stated failure mode* needs both conditions. `raw/xenc1f_param_delta.json`.
2. ✅ **RETIRED BY C122, SAME DAY — recorded because it shows the escalation worked, not to claim
   credit.** REF-A's trainable count was published nowhere when this stream started; C122 measured it
   (**160 514 460 vs flagship's 277 404 073**). ⇒ **the item is closed, and §2.2 now uses the number.**
   ⚠️ It is left visible rather than deleted: a stream that had merely *inherited* "REF-A's count is
   unpublished" would still be asserting it.
3. ⚠️ **`MODEL_REGISTRY.md:1899`'s *"differ in exactly two things"* is FALSE — C122 makes it at least
   four**, and independently of Stream A: `full_relaxed` SIGReg is **two** changes (it also enables
   `free_dims`, which REF-A never receives); **h15 imagination, 22.06 M params REF-A does not have**;
   **grounding depth 13.43 M vs 4.48 M**. ⛔ And *"the parity test pins the WRONG TRAINER and omits
   `imagination` entirely."* Stream A adds a fifth: the deployed arm ran `--adapter temporal`
   (`stack/experiments/reset-speed4b/refa_plus.py`), a class `stack/tanitad/refs/refa.py:190` cannot
   build (it asserts `adapter_kind in ("pool","grid")`) ⇒ a reconstruction from `tanitad.refs` alone
   silently yields the **wrong adapter**. ⇒ **REF-A is not a clean encoder-axis ablation and must
   stop being cited as one** — which is a further reason this document does not rest its case on it.
4. ⛔ **Registry internal disagreement:** §2.1 prints REF-A FDE **3.2619** / miss **0.6245**; §6 and
   the raw JSON give **3.2803** / **0.6129** (§2.3 likewise 4.5832/0.7246 vs 4.7642/0.7412). ⚠️ C122
   sharpens this: `:1972-1973`'s own estimator warning is **partly false** — FDE@2s and miss@2m *do*
   have decision-grade forms, **817 lines below in the same document**, and it is `:1927`'s pair that
   are unlabelled banned split-means.
5. ⏳ **The frozen-external guard is built and NOT YET CALLED** by `train_v6_staged.py` (C109). ⛔ It
   must be wired in the same change that introduces **either** external-encoder arm, or it is a guard
   that never runs — the C108 failure mode in advance.
6. ⏳ **Naming:** this document registers **`E-XENC-1F`** (F = fine-tuned), the trainable twin of the
   sibling's `E-XENC-1`. ⛔ `E-ENC` is already taken in code for *shared-encoder vs per-layer-encoders*
   (`v6.py:2501`, `:4468`, `:4482`; `train_v6_staged.py:3761`) and must not be reused.

---

## 11. ⛔ WHAT THIS DOCUMENT DOES **NOT** CLAIM

* **No T1 number, and no T0 number, for either encoder arm — neither has been run.** Every measurement
  here is a **build, a gradient, a wall-clock or a log re-read**.
* **No claim that DINOv2 would drive better.** REF-A is evidence it would not, at its own geometry;
  E-XENC-1's own pre-registration predicts the plateau as its DROPPED outcome.
* **No wall-clock claim for the arm on Thor.** §5 gives a **bound** from a 4060 ratio and the
  incumbent's measured Thor marginal, and requires the arm's own marginal to be measured by step 200.
* **No exponent, anywhere.** No fit, no window, no extrapolation.
* **No claim about DINOv3**, whose weights this programme cannot legally obtain today.
* **No change to any default, any config, or any file under `stack/`.** The invariant is re-verified
  at 87 893 449 / 405 **by building**.

---

## 12. DELIVERABLE MANIFEST

**Repo, staged, on branch `agent/arch-inf-20260803` — `TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-pretrained-encoder-arm/`:**

| artifact | what it is |
|---|---|
| `PREREG_PRETRAINED_ENCODER_ARM.md` | this document |
| `code/xenc1f_build_and_count.py` | builds default / incumbent / both swaps; freeze audit; **gradient-reach**; warm-start feasibility. Banks incrementally |
| `code/xenc1f_throughput.py` | per-iteration fwd+bwd timing of the three encoder paths; names the no-gradient tensors |
| `raw/xenc1f_param_delta.json` | **the invariant 87 893 449/405**, incumbent 336 542 025, both variants, freeze audits, grad-reach, warm-start key comparison |
| `raw/xenc1f_throughput.json` | s/iter, windows/s, peak GiB, ratios; `embeddings.mask_token` named |
| `raw/incumbent_noharm_band.json` | the incumbent's own step-0→3 000 WM-objective band + the spikiness trap |

**Compute:** dev-box RTX 4060 only, peak **2.8821 GiB**. ⛔ **Thor untouched** — no training launched,
no load added, no default changed, no file under `stack/` modified. **STAGED, NEVER COMMITTED, NEVER
PUSHED.**

**Not banked, deliberately:** DINOv2 weights (cached by `transformers`, regenerable) and any token
cache (§9.2 — hundreds of GB).
