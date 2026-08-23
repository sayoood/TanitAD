# R1 / R2 — pre-registered design spec against the 40:1 pooling bottleneck

> # ⛔⛔ CORRECTION 2026-08-18 (citation sweep) — **THIS DOCUMENT'S PREMISE IS REFUTED. READ THIS BEFORE ANY SECTION BELOW.**
>
> **Cite this block by its heading, never by line number** — this document is corrected in place and
> line-number citations into it do not survive (C90/C103).
>
> ## ⛔ 1. THE 40:1 POOLING BOTTLENECK IS **REFUTED** (C104) — `R1 IS DROPPED` by its own criterion
>
> **E-R1-0 tested the hypothesis by REMOVING the pool.** `MEASURED`, pre-registered, on frozen
> `v6F-SW-30k@11250`, 1 302 train / **1 507 eval windows in 70 episode clusters**, four pooling ratios
> (40:1 deployed / 10:1 / 4:1 / 1:1) differing **only in the kernel**, each forced to 2 048 features by
> a fixed random projection, **5 seeds**, `intercept_col=-1`:
> **on the four rungs the hypothesis was built to explain, removing the pool entirely moves r² by
> `|Δ| ≤ 0.0002`, with the CI containing zero on all five seeds** (`lead_closing`
> Δ **+0.00001 [−0.00597, +0.00504]**). ⇒ **No rung meets `R1 PROCEEDS`.**
> Artifact: `…/incoming/2026-08-18-pooling-ladder-ER10/raw/er10_main.json`.
>
> ⭐ **The constraint is the ENCODER/OBJECTIVE, not the pool.** Through the **same** deployed
> `AvgPool2d((4,10))` on the **same** windows, `facebook/dinov2-base` — **86 M params against our
> encoder's 87.3 M, so NOT a capacity gap** — reads `lead_gap` **0.44997 vs our 0.00496**, `ego_v0`
> **0.71733 vs 0.05240**, `lead_closing` **0.01713 vs 0.00000**. **The encoder gap is 91×; removing the
> pool on an encoder that *does* carry the signal is worth only +14 % / +30 %.**
>
> ⚠️ **C109 CORRECTS C104's CHOICE OF POSITIVE CONTROL — the conclusion stands, the citation must
> change.** ⛔ **`PC-2OBJ` is INERT AT THE DEPLOYED POOLING RATIO BY CONSTRUCTION** (two *opposing*
> plants inside one cell **cancel**; run at p40 it reproduced the un-planted arm to **5e-05**). The
> controls that actually fire are **`PC-LOCAL` / `PC-DIST`** — our own trained tokens through the
> deployed pool, **0.0596 → 1.0000, K1 9/9.** ⇒ quote those, **never `PC-2OBJ`**, when asserting the
> ladder had power to see a pooling-destroyed signal.
>
> ⭐ **WHAT IS *NOT* DROPPED:** **E-R1-1 is dropped and E-R2-0 is PROMOTED** — the 2×2's *placement*
> axis is measured inert, leaving the *target* axis (R2-cells, 16 899 params). ⛔ **ENCODER experiments
> now outrank both.** ⏳ **The document's STATUS line still reads `DESIGN + PRE-REGISTRATION`. Retiring
> or rescoping another stream's pre-registration is the PI's call, not this sweep's** — so the status
> is left as written and flagged here instead.
>
> ## ⛔ 2. §1.5's LADDER NUMBERS ARE STALE — see the correction block inside §1.5
>
> All four values quoted there moved, and the replacements have **moved a second time**. §1.5 also
> carries **five line-number citations** into `LATENT_LINEAR_LADDER.md` (`:178-194`, `:158-164`, `:264`,
> `:299-316`) which were **invalidated by two in-place rewrites of that file** — they are replaced with
> section-heading citations in the correction block.
>
> ## ⚠️ 3. §1.7's PARAMETER SHARES ARE A SCOPE ERROR — see the note inside §1.7
>
> `predictor_op` **68.5 %** / `encoder` **17.4 %** were computed from `V6Config()` **defaults**, not the
> live checkpoint. Corrected values in §1.7's note.
>
> ## ✅ 4. WHAT SURVIVES IN THIS DOCUMENT, UNTOUCHED
>
> **§1.1–§1.3's source-level derivation is CORRECT and was independently re-verified**: the operator
> really is `AvgPool2d((4,10))`, 40 tokens per cell, 64×160 px per cell, pool-before-project, and
> `encode_window(..., return_tokens=True)` really does materialise and discard the tokens R1 wanted.
> ⛔ **What was wrong was never the geometry — it was the inference that the geometry was the
> constraint.** C104's own root-cause class: *"an architectural bottleneck claim is not established
> until the bottleneck is REMOVED and the metric moves. Ablate the mechanism, don't narrate it."*
>
> **Sources:** `Project Steering/RETRACTION_LOG.md` **C104 · C106 · C107 · C109** ·
> `…/incoming/2026-08-18-pooling-ladder-ER10/POOLING_LADDER_ER10.md` ·
> `…/incoming/2026-08-18-c106-adversarial/C106_ADVERSARIAL.md` ·
> `…/Benchmarks & Eval/Implementation/incoming/2026-08-18-citation-sweep/CITATION_SWEEP.md`

**Date:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803` · **Tier of every claim below is stamped.**
**Status:** DESIGN + PRE-REGISTRATION. ⛔ **Zero GPU spent. No training-path change made.**
⛔ **SUPERSEDED IN ITS CENTRAL PREMISE — see the correction banner above.**

> **Evidence classes used throughout:** `MEASURED` (ours, with artifact path) · `PUBLISHED` (cited) ·
> `INHERITED` (another agent/doc, NOT re-verified by me) · `ESTIMATED` · `HYPOTHESIS`.
> Anything that decides a GPU-day is MEASURED or PUBLISHED.

---

## 0. What this document commits to, in one screen

| | R1 — pre-pool token-level loss | R2 — ego-compensated residual-flow target |
|---|---|---|
| **what it changes** | adds a loss that reads the encoder's **640 patch tokens** instead of the 16 pooled cells | adds a **label-free motion-residual** target the trunk must predict, computed from frames + ego poses we already have |
| **parameter cost** | see §4 — MEASURED by building the module | see §4 — MEASURED by building the module |
| **earliest legal insertion** | **NOT the live 30k S-W.** See §5 — it is an `encoder`/`aux`-reaching loss, so **only a NEW S-W run may introduce it**; `STAGE_MAY_INTRODUCE["S-T"]` cannot help, because S-T freezes `encoder`+`readout` | same |
| **kill condition** | §6.1 — pre-registered in both directions | §6.2 — pre-registered in both directions |
| **the expensive failure mode** | **R1 succeeding for the wrong reason** (§7) | R2 measuring ego residual error, not agents (§6.2 control) |
| **parity** | preserved — no episode re-selection (§8) | preserved — no episode re-selection (§8) |

---

## 1. The bottleneck, re-verified from source by me (not inherited)

The brief instructed me to verify the pooling claim myself. I did. **Every number in this section is
`MEASURED` by reading our own source; the `file:line` is quoted so it can be re-checked without me.**

### 1.1 The geometry

| fact | value | source (`file:line`) |
|---|---|---|
| encoder input field | **256 × 640 px**, `patch_size=16` | `stack/tanitad/models/v6.py:2778-2780` — `EncoderConfig(in_channels=9, image_size=256, image_width=640, patch_size=16, d_model=384, depth=8, n_heads=6)` |
| ⇒ token grid | **16 × 40 = 640 tokens** | DERIVED from the line above (256/16, 640/16) |
| readout grid | `grid=4`, `d_readout=128`, `grid_w` **unset** ⇒ `gw = grid = 4` | `stack/tanitad/models/v6.py:2781-2782` (`ReadoutConfig(grid=4, d_readout=128)`) and `stack/tanitad/models/readout.py:76` (`gw = grid if grid_w is None else int(grid_w)`) |
| pooling route chosen | `exact_pool = (16 % 4 == 0 and 40 % 4 == 0)` ⇒ **True** | `stack/tanitad/models/readout.py:87` |
| ⇒ **the operator** | **`nn.AvgPool2d((16//4, 40//4))` = `nn.AvgPool2d((4, 10))`** | `stack/tanitad/models/readout.py:88` |
| ⇒ tokens averaged per cell | **4 × 10 = 40** | DERIVED |
| ⇒ image area per cell | **64 × 160 px** (4 patches × 16 px tall, 10 patches × 16 px wide) | DERIVED |
| cells | **16** (4 × 4), `d_op = 16 × 128 = 2048` | `stack/tanitad/models/v6.py:2989-2997` (`n_cells`, `d_op = n_cells * d_readout`) |

⭐ **CONFIRMED, INDEPENDENTLY OF THE PRIOR AGENT: `AvgPool2d((4,10))`, 40 tokens per cell, 64×160 px
per cell.** The brief's headline claim survives my own re-derivation from source.

### 1.2 The one detail that matters for R1's design, and that I checked rather than assumed

`SpatialGridReadout.forward` (`stack/tanitad/models/readout.py:114-125`) **pools FIRST and projects
SECOND**:

```
x = tokens.transpose(1,2).reshape(b, d, token_h, token_w)   # readout.py:116
x = self.pool(x)                                            # readout.py:118  <- AvgPool2d((4,10))
x = x.flatten(2).transpose(1,2)                             # readout.py:124
return self.proj(x).flatten(1)                              # readout.py:125  <- Linear(384 -> 128)
```

Two consequences, both load-bearing:

1. **The mean is taken in the full `d_model = 384` space, before any learned compression.** The pool
   is therefore not a "learned summary that happened to lose things" — it is an **unweighted
   arithmetic mean applied to raw encoder features**. Nothing in the module can learn to *not*
   average. (`AvgPool2d` has no parameters — MEASURED: the module's only parameter is `self.proj`,
   `readout.py:111`.)
2. **Pool and projection are both linear, so they commute** — `proj(mean(tokens)) = mean(proj(tokens))`.
   This is why "just re-weight the objectives" cannot work, and it is also **why R1 is cheap**: the
   tokens R1 needs are the *same tensor* the readout already consumes, one line earlier.

### 1.3 `encode_window(..., return_tokens=True)` — verified, it exists

`stack/tanitad/models/v6.py:3691-3708`:

```
def encode_window(self, frames: Tensor, *, return_tokens: bool = False):
    ...
    tok = self.encoder(flat)                       # v6.py:3704
    z = self.readout(tok).reshape(b, w, -1)        # v6.py:3705
    if return_tokens:
        return z, tok.reshape(b, w, *tok.shape[1:])   # v6.py:3706-3707
    return z
```

Its own docstring states the key fact R1 depends on (`v6.py:3696-3698`): *"the tokens were always
computed and simply discarded by the readout."* **MEASURED: R1 needs no new forward pass and no
encoder change — the tensor it wants is already materialised and thrown away.**

⚠️ **Second probe (the absence rule).** I checked whether anything *already* consumes those tokens:
the only live call site is `v6.py:4049` (`z_op_win, tok_win = self.encode_window(frames, return_tokens=True)`),
which is the **F-18 agent-slot decoder's `slot_src="tokens"` arm** — an *interpretation head on a
frozen trunk*, in the `interp` group, which **`LADDER_UNTRAINED_GROUPS` forbids any ladder stage from
training** (`v6.py:3105`). ⇒ **No TRAINING objective reads the pre-pool tokens today.** That is the
gap R1 fills, and the F-18 path is proof the plumbing works, not proof the gap is closed.

### 1.4 The object-scale side of the claim (INHERITED, but re-located at its artifact)

| fact | value | class |
|---|---|---|
| median GT lead **apparent width** | **37.77 px ≈ 2.4 ViT patches** | `INHERITED` — `…/incoming/2026-08-17-probe-positive-control/PROBE_POSITIVE_CONTROL.md:585` (n = 2 721, in-corridor lead within 30 m; `width ≈ f_ref·w_obj/cx`, `f_ref = 305.577` px from cache meta, `:574`) |
| share of windows with lead **< 1 patch** | 4.34 % | `INHERITED`, same table |
| lead **height / pixel area** | ⛔ **NOT DERIVABLE** — *"HEIGHT IS ABSENT FROM THE JOIN"* (`PROBE_POSITIVE_CONTROL.md:577`) | `INHERITED` |
| ⇒ lead share of a cell's 40 tokens | **≈ 5–12 %** | ⚠️ `ESTIMATED` — depends on the missing height |

⚠️ **I did not re-measure 37.77 px** (it needs the obstacle join, which is a pod-side artifact and
would cost GPU/IO I am not authorised to spend). **It is INHERITED and I mark it so.** It is *not*
load-bearing for the direction of the argument: the pooling ratio is 40:1 from source alone, and the
prior agent's own note is the honest framing — **the lead is LARGE in patches (2.4) and small only
relative to the POOL.**

### 1.5 The independent confirmation, located at its artifact

> ### ⛔ CORRECTION 2026-08-18 (citation sweep) — EVERY NUMBER IN §1.5 IS STALE, AND §1.5's INFERENCE IS REFUTED
>
> **Cite this block as "§1.5 correction block", not by line number.**
>
> ⛔ **FIRST, THE INFERENCE.** §1.5 reads the rung profile as *"the signature an unweighted 40:1 mean
> predicts"* and calls it an **independent confirmation**. **C104 refuted the mechanism by removing the
> pool** (banner, §1 of this document). ⚠️ **Two independent methods appearing to agree did not save
> it** — a profile *consistent with* a mechanism is not evidence *for* it when it is equally consistent
> with a weak encoder, which is what the DINOv2 discriminator then showed. ⇒ **§1.5 is not a
> confirmation of anything; it is a rung profile.**
>
> ⛔ **SECOND, THE NUMBERS.** All four moved, and the replacements **moved a second time** at three
> seeds. `MEASURED`, re-derived by opening the banked per-seed JSON (not copied from a summary):
>
> | quantity | §1.5 quotes | ⭐ **current — 3-seed mean** | per seed (0 / 1 / 2) |
> |---|---|---|---|
> | `n_agents_all` r² | 0.076 | **0.1613** | 0.1519 / 0.1573 / 0.1745 |
> | `ego_curv` r² | 0.0001 | **0.0000** (5e-06) ⚠️ **BELOW its own null, 0.0005** | 0.0000 ×3 |
> | `lead_closing` r² | 0.0000 | **0.0009** | 0.0013 / 0.0000 / 0.0013 |
> | `lead_gap` `r_pv0` (partial-`v0`) | **+0.052** | ⛔ **−0.0884 — the SIGN FLIPS** | −0.1065 / −0.0665 / −0.0922 |
>
> ⚠️ **The `−0.107` that circulates as the replacement for `+0.052` is the SEED-0 value.** The 3-seed
> mean is **−0.0884**; `[−0.1065, −0.0665]` is a **SEED SPREAD, NOT a confidence interval.**
>
> ⚠️ **HALF OF ONE STALE PAIR IS NOT STALE — do not over-correct.** `lead_gap` **K1 +1.580** is a
> pre-C92 value (repaired: **+0.736 / +0.025 / +0.216**, separated on one seed of three, never PASS).
> But the ego-speed scalar's **K1 −1.562** is **CONFIRMED and seed-stable** — **−1.5618 [−2.0229,
> −1.1363], separated PASS, guard OK, identical on all three seeds and both repair routes.**
>
> ⭐ **THE PART OF §1.5 THAT SURVIVES AND STRENGTHENS.** *"The `lead_gap` signal is an ego-speed
> proxy"* is **more true, not less**: ego speed alone reads lead gap at **r² 0.4672 / MAE 3.5712 m**
> against the 2 048-dim latent's **r² 0.0069**, and partialling `v0` out drives the latent's
> correlation **negative**. ⇒ §6's *"every experiment carries a `v0`-only trivial-proxy arm by
> construction"* is **RIGHT and should be kept.**
>
> **PROVENANCE for every number in this block:** `MEASURED` · arm **`v6F-SW-30k@11250`** ⚠️ **EARLY
> READ (37.5 %)** · **T0-DIAGNOSTIC** · 130-clip lead-enriched probe pool, **70 eval clips** ⚠️ NOT the
> 40-episode val set · `intercept_col=-1` + C97 guard · **3 inner-split seeds** · estimator **paired
> episode-cluster bootstrap**, `n_boot 2000`, 70 clusters · **route A (`unpen`)** — ⚠️ **on these `r²`
> rungs route B is bit-identical except `ego_v0` seed 0 (A 0.1031 / B 0.1034), but the K1 numbers are
> NOT (`ego_v0` K1 differs by 0.3957 between routes); ⛔ the routes are never pooled.**
> Artifacts: `…/incoming/2026-08-18-ladder-3seed/raw/reread_unpen/ll3_s11250.json`,
> `…/raw/reread3_table.json` → `R6_rung_profile_r2_3seed`; re-derivation
> `…/incoming/2026-08-18-citation-sweep/raw/canonical_requote_table.json`.
>
> ⛔ **AND THE CITATION FORM ITSELF WAS BROKEN.** The five line-number citations below
> (`:178-194`, `:158-164`, `:264`, `:299-316`) point into `LATENT_LINEAR_LADDER.md`, which has been
> **rewritten in place twice**. They resolve to unrelated text. ⇒ **cite it by SECTION HEADING** —
> the rung profile now lives at its **§8.1 3-seed column**, and the 3-seed re-run at
> `…/incoming/2026-08-18-ladder-3seed/LADDER_3SEED.md` **§6a**.

*(Original §1.5 text, kept visible because the superseded claim must stay readable beside its
replacement:)*

`…/incoming/2026-08-17-latent-linear-ladder/LATENT_LINEAR_LADDER.md` §2.2 (`INHERITED`, quoted from
its own table at `:178-194`): `n_agents_all` r² **0.076** (highest rung) · `ego_curv` r² **0.0001** ·
`lead_closing` r² **0.0000**. §2.1 (`:158-164`) gives the paired nulls. ⇒ aggregate scene density
survives; individuated and relative-motion quantities are at the null. **That is the signature an
unweighted 40:1 mean predicts.**

⚠️ **And it must be read with C92 attached:** the one rung that looked positive (`lead_gap`
r +0.159) is an **ego-speed proxy** — partialling `v0` out leaves **r +0.052, r² 0.0027**
(`LATENT_LINEAR_LADDER.md:264`). Every experiment in §6 below therefore carries a **`v0`-only
trivial-proxy arm by construction**, not as an optional extra.

### 1.6 ⛔ What this does NOT license

Per C93 and the brief: this explains **D1** (the WM latent not supporting an agent readout)
**without any appeal to the world model's competence**. It does **not** show the world model is fine.
A pool that destroys individuation is *sufficient* to produce the observed profile; it is not
*evidence* that nothing else is also wrong. Both R1 and R2 are designed to be informative about
which — see the positive controls in §6.

### 1.7 ⭐ The framing number that fell out of the parameter measurement

> ### ⚠️ CORRECTION 2026-08-18 (citation sweep) — §1.7's SHARES ARE A SCOPE ERROR (C104), AND THE TOTAL IS 3.8× UNDERSTATED
>
> **Cite this block as "§1.7 correction block", not by line number.**
>
> The table below was computed from **`V6Config()` DEFAULTS**, not from the live checkpoint. `MEASURED`
> on the live S-W run (`d_model 768`):
>
> | | §1.7 as written (defaults) | ⭐ **live checkpoint** |
> |---|---|---|
> | total params | 87,893,449 | ⛔ **336,542,025** — **3.8× understated** |
> | `predictor_op` share | **68.5 %** | **55.9 %** |
> | `encoder` share | **17.4 %** | **25.9 %** |
>
> ⚠️ **C104 published the total as `336,559,305`; C106 corrected it to `336,542,025`** from two
> independent sources — the checkpoint's own `_meta.config.param_report.total`, and a fresh `V6Stack`
> instantiated from that same checkpoint's `v6_config`. **Δ = 17,280.** Use **336,542,025**.
>
> ⚠️ **A consequence that travels with it, and it is a PI item, not an editorial one:** the live run is
> **12.2 % over the programme's "Sub-300M" headline** in `CLAUDE.md`. **Not a silent breach** — it
> launched with `param_budget: 350000000`, so the assert passed by design — but the headline and the
> model now disagree. ⏳ **PI decision: restate the claim or rescope the model.**
>
> ⭐ **WHAT STILL HOLDS FROM §1.7:** the pooling operator really does hold **0 parameters** and the
> readout really is one `Linear(384→128)`. ⛔ **But the rhetorical use of the share — "68.5 % of the
> stack sits downstream of a 40:1 mean, therefore the mean is the constraint" — is exactly the
> inference C104 refuted by removing the mean.** *(Same family as `df` on a pod: a real number read at
> the wrong scope, then used as an answer.)*

`MEASURED` (built `V6Stack(V6Config())` on CPU; raw in `raw/r1r2_params.json`) ⚠️ **— on DEFAULTS, not
the live checkpoint; see the correction block above**:

| group | parameters | share |
|---|---|---|
| `encoder` | **15,327,360** | 17.4 % |
| `readout` | **49,280** | 0.06 % |
| `predictor_op` | **60,193,539** | 68.5 % |
| `layer_tac` | 5,765,165 | 6.6 % |
| `layer_str` | 4,152,993 | 4.7 % |
| `planner` | 755,320 | 0.9 % |
| `aux` (= the O3 head) | 1,649,792 | 1.9 % |
| **total** | **87,893,449** | — (budget `PARAM_BUDGET` **300,000,000**, headroom **212,106,551**) |

⭐ **68.5 % of the stack sits DOWNSTREAM of a 40:1 unweighted mean, and the perception path that
feeds it holds 17.4 %.** The pooling operator itself holds **0 parameters** (`MEASURED`:
`sum(p.numel() for p in readout.pool.parameters()) == 0`), so **nothing in the model can learn not to
average.** The entire readout — the "geometry firewall" — is one `Linear(384→128)`, 49,280 weights.

This is the shape of the problem in one table: we spend 60 M parameters modelling the dynamics of a
2048-number summary produced by a parameterless mean.

---

## 2. ⛔ FIRST: R1 and R2 are NOT two alternatives. They vary DIFFERENT axes, and conflating them is the `--v2` failure

The brief presents R1 and R2 as "two candidate repairs". Read literally they would be run as two
arms and compared — and that is **the ten-levers-on-two-axes conflation `CLAUDE.md` names as the
`--v2` failure**, because they do not vary the same thing:

* **R1 varies WHERE the loss reads** — pre-pool (640 tokens) instead of post-pool (16 cells).
* **R2 varies WHAT the loss is asked to predict** — an external, ego-compensated motion residual
  instead of the trunk's own latent.

They are **orthogonal**, so the design is a **2 × 2**, and one of its four cells is **already
trained**:

| | target = **self-predictive latent** (masked-unit) | target = **ego-compensated residual flow** (external, label-free) |
|---|---|---|
| **post-pool** — 16 cells, `d_readout` 128 | **O3 — ALREADY EXISTS AND IS ALREADY TRAINED** (`MaskedCellPredictor`, `v6.py:3245`; 1,649,792 params `MEASURED`). This is the **baseline cell, free.** | **R2-cells** |
| **pre-pool** — 640 tokens, `d_model` 384 | **R1** | **R1 ⊗ R2** (the joint cell) |

⭐ **Consequence for cost:** because the post-pool/self-predictive cell is the live run, the 2×2
needs **three** new arms, not four, and the **placement** main effect (O3 → R1) and the **target**
main effect (O3 → R2-cells) are each a single-lever contrast against an arm we already have.
**Attribution is preserved by construction.** Running R1 and R2 as an undifferentiated "repair
bundle" would forfeit exactly that and is refused here.

⚠️ **And note what the 2×2 makes visible that a two-arm race hides:** if **R2-cells** works,
the objective was wrong and the pool was not binding. If **R1** works and **R2-cells** does not,
the *placement* was binding. If only the **joint** cell works, they interact and neither alone is a
repair. A two-arm comparison cannot produce any of those three statements.

---

## 3. ⚠️ The published precedent for R1 — what I VERIFIED myself, and what did NOT survive

The brief cites **V-JEPA 2.1** and warns the prior agent re-fetched only part of it. I re-fetched
**arXiv:2603.14482v2** myself (two independent fetches of the paper plus one third-party summary).
**Here is exactly what I verified and what I could not.**

### 3.1 ✅ VERIFIED — the mechanism, and it is the same mechanism as ours

`PUBLISHED`, quoted from the paper (arXiv:2603.14482v2), on why masked-only supervision degrades
dense features:

> *"the model has no incentive to encode local information within the context tokens and can instead
> devote this computation to aggregating global information to minimize ℒ_prediction, similarly to
> register tokens."*

**This is our failure in a different costume, and the parallel is exact:** their context tokens
*choose* to become global aggregators because the loss never scores them locally; our 40 tokens per
cell are *forced* to be a global aggregator by an unweighted mean, because no loss ever scores them
at all (§1.3, second probe). ⇒ **the diagnosis transfers; the remedy — score the tokens where they
live — is the same remedy.**

Their loss (`PUBLISHED`, verbatim from the paper):

```
ℒ_ctx = (1/|C|) Σ_{i∈C} λ_i ‖ P_φ(E_θ(x), Δy)_i − sg(E_θ̄(y)_i) ‖₁ ,   λ_i = λ / √(d_min(i, M))
```

with `d_min` the distance in blocks from a context token to its nearest mask token, `sg` a
stop-gradient, and `E_θ̄` an **EMA teacher**. ⇒ dense supervision over **all** tokens, an **L1**
residual, and a **distance-weighted** coefficient. `λ` itself and the ablation's exact epoch budget
are **not stated in the paper** — I looked and report the absence.

### 3.2 ⛔ NOT VERIFIED — the brief's headline number is wrong as stated

| the brief says | what the paper's tables say (my own fetch) | verdict |
|---|---|---|
| *"DAVIS J&F 52.5 → 69.0"* | **DAVIS-S: V-JEPA 2.1 ViT-g 68.1, ViT-G 69.0** (Table 8). **No V-JEPA 2 DAVIS baseline appears in the document.** The value **52.5 does not appear.** | ⛔ **The `69.0` is real but it is V-JEPA 2.1 ViT-G's ABSOLUTE DAVIS-S score, not the top of a 52.5→69.0 delta. The delta as phrased is UNSUPPORTED by the paper's own tables and must not be quoted.** (YouTube-VOS, for completeness: 72.3 / 72.7.) |
| ADE20K pair *"internally inconsistent (22.2 vs 24.4)"* | **CONFIRMED** — V-JEPA 2 reads **22.2** in the Table 1 ablation and **24.4** in the Table 8 dense-task table. | ✅ the brief's warning is correct |

⚠️ **And I found a SECOND internal inconsistency the brief did not have:** the context-loss ablation
row reads **33.8** in Table 1 while the surrounding **text says 33.9**. Two self-inconsistent pairs
in one preprint is a property of the source, and it is why nothing here is allowed to decide a
GPU-day on this paper alone.

### 3.3 ⭐⭐ THE FINDING THAT ACTUALLY MATTERS FOR R1, AND NEITHER THE BRIEF NOR THE PRIOR AGENT SURFACED IT

The paper's **Table 1 ablation of the context loss in isolation** — i.e. exactly the intervention R1
copies, with nothing else from the recipe:

| metric | V-JEPA 2 baseline | **+ context loss ALONE** | Δ |
|---|---|---|---|
| ADE20K mIoU (dense) | 22.2 | **33.8** | **+11.6** ⬆ |
| NYUv2 RMSE (dense, lower better) | 0.682 | **0.474** | **−0.208** ⬆ |
| **IN1K top-1 (global)** | 82.2 | **72.6** | ⛔ **−9.6** ⬇ |
| **SSv2 top-1 (global/motion)** | 72.8 | **62.5** | ⛔ **−10.3** ⬇ |

`PUBLISHED` — arXiv:2603.14482v2 Table 1, read by me. **Direction independently corroborated** at a
second source, which states the same thing in words: *"Context supervision initially harms global
tasks and is mitigated by deep supervision"* (emergentmind.com/papers/2603.14482). **The magnitudes
come from one read of the paper's Table 1** and are marked accordingly.

⛔ **This is the strongest quantitative caution against R1 that exists, and it is in R1's own
flagship citation.** The isolated intervention bought dense structure and **cost ~10 points on the
global and motion tasks.** Our S-W trunk's entire job is a global dynamics task (predict `z_op`
forward). The paper *recovers* the global metrics (IN1K 82.2 → 85.5, SSv2 72.8 → 77.7) **only with
the full recipe** — deep multi-level self-supervision, hand-tuned λ with a warm-up schedule
(epochs 50–100), multi-modal tokenization, **and scaling to VisionMix-163M: 142 M images + 19 M
video samples ≈ 1.6 M hours** (`PUBLISHED`, same paper).

⇒ **Two consequences, both binding on the design below:**
1. **R1 must be pre-registered with a NO-HARM criterion on the WM's own objective, not only a
   gain criterion on structure.** An R1 arm that improves token structure and degrades `g_op_fwd_ade_m`
   is the paper's Table 1 row happening to us, and it must be a **DROP**, not a "trade-off".
   §6.1's kill condition encodes this.
2. **The published recovery path is 163 M samples we do not have.** That is not a footnote — it is
   §7's argument, and it lands before we spend anything.

---

## 4. R1 — pre-pool token-level loss: the spec

### 4.1 What it is

**The existing O3 head, moved from the 16 pooled cells to the 640 patch tokens, and nothing else
changed.** Same masking idiom, same stop-grad target, same transformer shape, same `aux` group.
Only `n_units: 16 → 640` and `d_unit: 128 → 384`.

This deliberate minimality is the design: **the arm must vary placement and only placement**, or the
2×2 in §2 stops being attributable. (The V-JEPA-2.1-faithful variant — EMA teacher, L1 residual,
`λ/√d_min` distance weighting — is named in §4.4 as the **escalation**, not the first arm, because it
varies four things at once.)

Module built and measured (`stack/` import path, CPU, zero GPU):

```
class TokenMaskedPredictor(nn.Module):          # candidate, NOT yet in the tree
    inp        = Linear(384 -> hidden)
    pos        = Parameter(1, 640, hidden)
    mask_token = Parameter(1, 1, hidden)
    blocks     = TransformerEncoder(TransformerEncoderLayer(hidden, 4,
                     dim_feedforward=4*hidden, norm_first=True, gelu), depth)
    out        = Linear(hidden -> 384)
```

### 4.2 Parameter cost — MEASURED by building it, not estimated

`MEASURED` on CPU; script `code/measure_r1r2.py`, raw `raw/r1r2_params.json`.

| variant | **total params** | `inp` | `pos` | `mask_token` | `blocks` | `out` | vs the O3 head it sits beside |
|---|---|---|---|---|---|---|---|
| `hidden=192, depth=2` | **1,160,832** | 73,920 | 122,880 | 192 | 889,728 | 74,112 | ⭐ **−488,960 — CHEAPER than O3** |
| **`hidden=256, depth=2` (RECOMMENDED)** | **1,940,864** | 98,560 | 163,840 | 256 | 1,579,520 | 98,688 | **+291,072 (+17.6 %)** |
| `hidden=256, depth=3` | 2,730,624 | 98,560 | 163,840 | 256 | 2,369,280 | 98,688 | +1,080,832 |
| `hidden=384, depth=2` | 4,090,752 | 147,840 | 245,760 | 384 | 3,548,928 | 147,840 | +2,440,960 |
| — *reference:* existing O3 `MaskedCellPredictor(16, 128, hidden=256)` | **1,649,792** | | | | | | — |

⭐ **The headline: reading 40× more units costs 17.6 % more parameters, and at `hidden=192` it costs
LESS than the head we already run.** The reason is structural and worth stating because it is
counter-intuitive: a transformer's parameters live in `d²`, not in sequence length. **Only `pos`
scales with the unit count** (163,840 vs O3's 4,096), and that is 8.4 % of the module.

**Where the parameters go:** entirely into a **new `aux`-group module**, `masked_tokens.*`
(§5.2). Zero parameters are added to `encoder`, `readout`, `predictor_op`, or any planner module.
Budget impact: **1,940,864 of 212,106,551 headroom = 0.92 %** (`MEASURED`, §1.7).

**Smoke-tested at production geometry** (`MEASURED`): input `[2, 640, 384]` → output `[2, 640, 384]`,
shape-exact, mask `[2, 640]` bool.

⚠️ **The real cost is not parameters, it is ACTIVATION and TIME.** `ESTIMATED`, and flagged as
un-measured because measuring it needs a GPU I am not authorised to use: self-attention over 640
tokens × `batch × window` sequences per step. At Thor's live `--batch 8` with `window=6` that is
**48 sequences of length 640** through `depth` layers per step, against O3's 48 sequences of length
**16**. Attention is O(n²) ⇒ **~1,600× the attention FLOPs of O3**, off a base that is small.
⛔ **This must be MEASURED (a step-time A/B on the dev-box GPU, ~10 minutes, no training) before the
arm is scheduled** — it is the single number most likely to make R1 unaffordable on Thor, and it is
not in this document. Named as an open item in §11.

### 4.3 Where the tokens come from — no encoder change, no extra forward

`encode_window(frames, return_tokens=True)` (`v6.py:3691-3708`) already returns them; the docstring
records that they *"were always computed and simply discarded by the readout"* (`v6.py:3697-3698`).
⇒ **R1's data path is one already-existing keyword argument.** `MEASURED` (§1.3).

### 4.4 The escalation variant, named but NOT the first arm

If R1 (minimal) passes §6.1 but is weak, the pre-registered escalation is the **V-JEPA-2.1-faithful**
form: dense loss over **all** tokens (not only masked), **L1** residual, `λ_i = λ/√(d_min(i,M))`
distance weighting, and an **EMA teacher** target instead of stop-grad. ⚠️ Costs a second encoder
copy (+15,327,360 params, `MEASURED` §1.7) for the EMA teacher — still 7 % of headroom, but it is a
real memory cost on Thor. **It varies four levers at once and is therefore only admissible after the
one-lever arm has a verdict.**

---

## 5. Ladder legality — where each group may LEGALLY be introduced

### 5.1 ⛔ The decisive constraint: R1 and R2 are S-W-ONLY, and cannot be bolted on later

`MEASURED` from source, and this is the fact that sets the whole schedule:

| step of the argument | source |
|---|---|
| Both R1 and R2 push gradient into `encoder` + `readout` — that is their entire point | by construction |
| `ISOLATION_MATRIX["aux"] == ("encoder", "readout", "aux")` — `aux` is the ONE non-WM group permitted to reach the trunk | `v6.py:2455ff` |
| Only **S-W** trains `aux`, `encoder` and `readout`: `STAGE_GROUPS["S-W"] == ("encoder","readout","predictor_op","aux")` | `v6.py:3135` |
| S-T trains `("layer_tac","planner")`; S-S trains `("layer_str",)` — **`aux`/`encoder`/`readout` are FROZEN in both** | `v6.py:3136-3137` |
| ⇒ **an R1/R2 loss added at S-T or S-S would flow into frozen parameters and change nothing** | DERIVED |
| `STAGE_MAY_INTRODUCE["S-W"] == ()` — S-W may introduce **no** new key over an inherited checkpoint | `stack/scripts/train_v6_staged.py:300` |
| `STAGE_MAY_INTRODUCE["S-T"]` = `("cand_score.", "cond_tac_dyn.", "prop_diffusion.", "fallback.", "agent_slots.")` — **`masked_tokens.` is not there, and adding it would be pointless** since S-T freezes `aux` | `train_v6_staged.py:339-340` |

⇒ ⛔ **THE EARLIEST ADMISSIBLE INSERTION POINT FOR BOTH R1 AND R2 IS A FRESH S-W RUN FROM STEP 0.**
Not the live 30k. Not an `--init-from` of it (blocked by `STAGE_MAY_INTRODUCE["S-W"] == ()`).
Not S-T, S-S or S-J. **There is no cheap late entry, and that is why §7's zero-training pre-gates
are the load-bearing part of this spec.**

⚠️ **One precision, because it will otherwise be mis-read as a contradiction:** S-J *does* train
`encoder`, `readout` and `aux` (`v6.py:3138-3139`), so once the module **exists** S-J would optimise
it — but `STAGE_MAY_INTRODUCE["S-J"] == ()` (`train_v6_staged.py:342`), so S-J can never be where it
**arrives**. ⇒ **introduction is S-W-only; joint polish inherits it for free.** The distinction is
exactly the one `STAGE_MAY_INTRODUCE`'s own note draws — *"an entry here has never MEANT 'this stage
optimises the module'"* (`train_v6_staged.py:325-327`) — read in the other direction.

### 5.2 The live 30k S-W run is untouched, and here is the mechanism that guarantees it

`MEASURED`: the live run resumes tensor-strict and `RESUME_CONTRACT` (`train_v6_staged.py:396-413`)
requires `same_stage`, `labelled`, and `has_optimiser`. A checkpoint carrying `masked_tokens.*` has a
different trainable-tensor count and a different optimiser param-group layout, so a resume would be
refused — **correctly**. ⇒ **Nothing in this document can be applied to the live run even by
accident.** No file in the training path is modified by this deliverable (§11 manifest).

### 5.3 ⚠️ The `06b8782` defect class — I checked for it, here is the result

The brief flags: commit `06b8782` appended `interp` to `MODULE_GROUPS` and thereby changed what S-J
trains **without touching S-J's declaration**; tuple immutability protects against mutation, not
meaning.

`MEASURED` today by building the module and comparing identities:

```
STAGE_GROUPS["S-J"] is MODULE_GROUPS               -> False
tuple(STAGE_GROUPS["S-J"]) == tuple(MODULE_GROUPS) -> False
STAGE_GROUPS["S-J"] == ('encoder','readout','predictor_op','layer_tac',
                        'layer_str','planner','aux')     # interp excluded
```

✅ **The identity alias is GONE** — S-J is now `tuple(g for g in MODULE_GROUPS if g not in
LADDER_UNTRAINED_GROUPS)` (`v6.py:3138-3139`). The defect class is closed for the case that produced it.

⛔ **But the class is NOT closed for R1/R2, and the residue is a live trap I am pre-registering
against.** R1/R2 do **not** need a new `MODULE_GROUPS` entry — they go in the existing `aux`, beside
`masked_cells.` and `sigreg.` (`v6.py:3648`). **That is the safe choice and it is the recommendation.**
If a future author instead adds a new group (say `"dense"`), then **by the documented `derived`
property** (`v6.py:3118-3121`) it is **automatically joint-polished in S-J** — the same silent
meaning-change in the *permitted* direction. The three-part edit that must travel together:

1. add `("masked_tokens.", "aux")` to `V6Stack._GROUP_PREFIXES` (`v6.py:3648`) — **without it,
   `group_of` RAISES** (`v6.py:3667-3671`: *"an ungrouped parameter is one no stage freezes and no
   stage trains"*). This is the mechanical guard that makes the omission loud instead of silent;
2. **do NOT** add a new `MODULE_GROUPS` entry unless the module genuinely must be frozen separately
   from `aux` — and if one is added, decide `LADDER_UNTRAINED_GROUPS` membership **in the same edit**;
3. state the new loss term in `V6LossWeights.for_stage` so a weight advertised in the launch line is
   one that actually trains something (the defect `v6.py:3084-3087` describes).

⚠️ **R2 has an additional adjudication that R1 does not, and I am escalating it rather than
deciding it alone** — see §5.4.

### 5.4 ⚠️ ESCALATION — is R2's target `aux` or `interp`? The isolation matrix must rule

`ISOLATION_MATRIX` (`v6.py:2455ff`) draws the line at *"no perception label, map or reward in any
trunk loss"*: `aux` may reach the encoder (O3/O6 are label-free trunk losses); `interp` may reach
**nothing but itself**, because its supervision is a **perception label**.

R2's target is built from **raw video + ego poses**, with **no annotation, no cuboid, no map, no
reward** — so by the letter of the rule it is `aux`, in the same family as O3/O6. **That is my
reading and it is what §6.2 assumes.** But it is one step further from "self-predictive" than
anything currently in `aux`, and I do not consider it mine to settle alone.

⛔ **ESCALATION (not a "please merge" in a README):** the owner of `ISOLATION_MATRIX` must rule on
whether an **ego-pose-derived geometric target** is `aux` or `interp`. **The ruling changes what R2
can prove:** if `interp`, R2 cannot shape the trunk at all and collapses to a frozen-trunk P8 probe —
which answers "does the latent already carry motion residual?" but **not** "can a motion-residual
objective put it there". The former is E-R2-0 (§6.2) and is worth running either way; the latter
would be dead. **This ruling is a prerequisite for scheduling E-R2-1 and should be obtained before
any GPU is committed.**

⚠️ Note the ruling does **not** touch the vision-only rule: R2's head is a **training-time auxiliary**
that is never evaluated at inference, so the inference path stays vision-only trivially. Ego is used
**only in label derivation**, which the 2026-08-03 binding rule explicitly permits.

---

## 6. R2 — ego-motion-compensated residual-flow target: the spec

### 6.1 The two facts from source that make R2 buildable at all

| fact | value | source |
|---|---|---|
| the encoder input is **already 3 frames** | *"D-015 stacking (3 frames @100 ms → 9 channels)"* | `stack/tanitad/data/physicalai.py:19` (and `EncoderConfig(in_channels=9)`, `v6.py:2779`) |
| the batch carries ego pose | `poses [T, 4] = (x, y, yaw, v)` | `stack/tanitad/data/_contract.py` (contract docstring + `assert_contract`) |

⭐ **The first is the design's foundation:** motion is **already inside a single encoder input**, co-located
at every patch across the channel axis. The encoder is *architecturally able* to compute a per-token
motion residual today — it is simply never asked to. R2 asks.

⚠️ **The second is a limit I am declaring, not hiding:** the pose is **planar (x, y, yaw)** — there is
**no pitch or roll**. Suspension and road-camber camera motion are therefore **not** compensated and
enter the residual as noise. This is the dominant noise term and it is why §7.3's negative control
(wrong-ΔT) is mandatory.

⚠️ **And the batch does NOT carry intrinsics/extrinsics** (`_contract.py` lists frames/actions/poses
only), while the **episode build does read `camera_intrinsics` + `sensor_extrinsics`** (`CLAUDE.md`
read-set table, `physicalai.py` layer = 5 features). ⇒ **R2's target MUST be a precomputed sidecar,
built where the calibration lives.** That is not a workaround — it is also what makes R2 free at
train time and parity-safe (§9).

### 6.2 Target construction, precisely

For window step `t`, over the sub-frame pair `(I_{t−Δ}, I_t)`, Δ = 100 ms:

1. **Ego relative pose.** From `poses`, the planar rigid motion `ΔT = (Δx, Δy, Δψ)` between the two
   sub-frames, in the ego frame.
2. **Lift to camera.** With intrinsics `K` and the camera↔ego extrinsic from the episode build,
   `ΔT` becomes a camera-frame `(R, t)`: `R` a rotation about the vertical, `t` the planar translation.
3. **Ego-induced flow, two components.**
   * **Rotational — exact and DEPTH-FREE:** `H_rot = K R K⁻¹` maps every pixel exactly at any depth.
   * **Translational — depth-dependent:** the **ground-plane homography**
     `H_pl = K (R − t nᵀ / d) K⁻¹`, with plane normal `n` and camera height `d` from the extrinsics.
     Exact for points **on the road surface**.
   ⇒ `f_ego(p) = warp(p; H_pl) − p`.
4. **Observed flow** `f_obs`: computed **once, offline, per episode** with an off-the-shelf estimator.
   ⚠️ The estimator choice (RAFT-small vs a classical DIS/Farnebäck) is an **ops decision I did not
   make** and is a declared open item (§11) — E-R2-0 must be run with the estimator that will
   actually be used, because the target's quality *is* the estimator's quality.
5. **Residual** `r(p) = f_obs(p) − f_ego(p)`.
6. **Per-token reduction** (R2-tokens): mean over each 16×16 patch ⇒ `r̄ ∈ ℝ^{640×2}`, magnitude
   `m ∈ ℝ^{640}`. fp16 sidecar ⇒ **3,840 B per window step** (`640 × 3 × 2`).
7. **Per-cell reduction** (R2-cells): ⭐ the **identical `AvgPool2d((4,10))`** applied to `r̄` ⇒ `16×2`.
   Using the same operator is what makes R2-cells vs R2-tokens a *clean placement contrast* rather
   than two different targets.

### 6.3 ⛔ The confound, stated before any result — and the tempting fix that DOES NOT WORK

After ground-plane compensation the residual is non-zero for **(a) independently moving objects** —
what we want — **and (b) static structure OFF the ground plane** (buildings, poles, signs, parked
cars), whose parallax scales with height-off-plane and `1/depth`. **The target is a
"moving-OR-off-plane" detector, not a motion detector.** That is why §7.3 validates the target
*before* any training is scheduled.

⚠️ **And the depth-free fix that everyone reaches for is REFUSED here, with its reason, so nobody
re-derives it:** the **epipolar-perpendicular residual** (distance of observed flow from the epipolar
line) is exactly zero for static points at **any** depth, which would eliminate (b) completely. **But
for a forward-facing camera under forward ego motion the epipole sits near the image centre, so a
lead vehicle directly ahead moves RADIALLY — along its own epipolar line.** Its perpendicular
residual is ≈ 0. ⇒ **the epipolar formulation is blind to precisely the object R2 exists to serve.**
`DERIVED`, and it disqualifies the elegant option.

### 6.4 Parameter cost — MEASURED

`MEASURED` (same script/JSON as §4.2). The head is a per-unit regressor onto `(u, v, |r|)`:

| placement | `hidden=0` (linear) | **`hidden=128` (recommended)** | `hidden=256` |
|---|---|---|---|
| **R2-cells** (reads `z_op` cells, `d=128`) | 387 | **16,899** | 33,795 |
| **R2-tokens** (reads patch tokens, `d=384`) | 1,155 | **49,667** | 99,331 |

⭐ **R2-cells is the cheapest arm in the entire design: 16,899 parameters, 0.008 % of headroom, and
NO attention — so it carries none of R1's O(n²) time risk (§4.2).** Smoke-tested at production
geometry (`MEASURED`): `[2,640,384] → [2,640,3]`.

Group: `aux`, prefix `("resflow.", "aux")` — subject to the §5.4 escalation.

---

## 7. The pre-registered experiments — both outcomes committed in advance

⛔ **Every experiment below carries BOTH controls, per C92 and C79: a TRIVIAL-PROXY control (does a
scalar already in the input do as well?) and a POSITIVE control (can the instrument read the answer
when handed the answer?). A margin over a random null is not evidence until both are reported.**

### 7.1 ⭐⭐ E-R1-0 — the pooling-ratio ladder on a frozen banked checkpoint. **ZERO TRAINING.**

**This is the cheapest discriminating experiment in the document, and it decides whether an S-W run
is worth spending at all** (§5.1: there is no cheap late entry, so the pre-gate carries the weight).

**Question.** Do the **pre-pool tokens** linearly carry what the **post-pool cells** do not — i.e. is
the pool the binding constraint, or did the encoder never encode it?

**Method.** One frozen banked S-W checkpoint; the same val-40 windows, rungs, split and estimator as
`LATENT_LINEAR_LADDER.md`. Four feature arms differing **only** in how much spatial averaging happens,
all produced by the same `AvgPool2d` family over the same `16×40` token grid:

| arm | kernel | units | tokens averaged |
|---|---|---|---|
| **40:1 — the DEPLOYED readout** | `(4,10)` | 16 | 40 |
| 10:1 | `(2,5)` | 64 | 10 |
| 4:1 | `(2,2)` | 160 | 4 |
| **1:1 — no pooling** | `(1,1)` | 640 | 1 |

⛔ **THE DIMENSION CONFOUND, AND THE EXPERIMENT IS INVALID WITHOUT THIS FIX.** Raw feature counts are
6,144 / 24,576 / 61,440 / **245,760** against `n ≈ 2,721` windows — a ridge on 245 k features fits
anything, and the 1:1 arm would "win" for reasons that have nothing to do with pooling. ⇒ **Every arm
is projected to EXACTLY 2048 features by a FIXED random Gaussian projection sharing one seed, so the
only thing that varies is the averaging.** Repeat over **≥ 5 projection seeds** and report the spread
as the instrument's own noise floor (the discipline `LATENT_LINEAR_LADDER.md:299-316` already applies).

⛔ **And it MUST pass `intercept_col=-1`, because the C92 repair is landed but is NOT the default.**
`MEASURED` at source today: `ridge_fit(X, y, alpha, intercept_col=None)`
(`…/incoming/2026-08-17-probe-positive-control/code/pc6_linear_readout.py:50`). C92 records that the
unrepaired path **penalises its own intercept**, so predictions collapse toward **zero** rather than
the **mean** and a no-signal arm scores worse than a constant *by construction* — *"a FAIL from a
biased floor is not a finding"*. The default was deliberately left bit-exact so banked
`pc6_ridge_*.json` artifacts keep reproducing (`taniteval/tests/test_ridge_intercept_penalty.py`
pins **both** directions). ⇒ ⚠️ **an E-R1-0 run that forgets the keyword silently inherits the biased
floor and its "no signal at 1:1" would be an artefact of the instrument, not a finding.** The
argument must appear in the run's `config.json` and be asserted, not remembered.

**Controls.**

| control | what it is | why it is mandatory |
|---|---|---|
| **TRIVIAL-PROXY** | a `v0`-only ridge, and the **partial-r after removing `v0`**, on **every arm × every rung** | **C92**: ego speed alone beat the whole 2048-d latent. ⛔ **NUMBERS CORRECTED 2026-08-18 (citation sweep, §1.5 correction block):** the scalar's **K1 −1.562** is ✅ confirmed and seed-stable (**−1.5618 [−2.0229, −1.1363] PASS, guard OK, 3/3 seeds, both routes**); the latent's **`+1.580` is a PRE-C92 value** — repaired it is **+0.736 / +0.025 / +0.216, never PASS**; and the `lead_gap` partial-`v0` correlation is **−0.0884 (3-seed mean; seed 0 −0.107), NOT +0.052 — the sign flips.** A pooling effect that vanishes under this control is an ego-speed effect. ⚠️ **And per C107 the control must be enumerated PER ARM on the arm's OWN window family, not once per study** — a `C-V0` fitted on different windows is not a control. |
| **POSITIVE** | the **geometric ORACLE arm** (`PROBE_POSITIVE_CONTROL.md`: median 0.816 m on `lead_gap`, K1 PASS) must PASS under this harness | **C79**: D1 was withdrawn because a probe failed its positive control. If the oracle does not pass here, no arm's number is readable. |
| **NEGATIVE** | random-latent null; within-episode label shuffle; constant predictor | the existing C-CONST / C-SHUF-XEP discipline |

**PRE-REGISTERED OUTCOMES — both directions, committed now:**

* ✅ **R1 PROCEEDS** iff, on the relative-motion / individuation rungs (`lead_closing`,
  `lead_inv_ttc`, and `n_agents_grid` restricted to windows with **≥ 2** agents): r² rises
  **monotonically as the pooling ratio falls**, **AND** the paired episode-cluster-bootstrap CI on
  **Δr²(1:1 − 40:1) excludes 0**, **AND** that Δ survives partialling out `v0`, **AND** the oracle
  positive control PASSES.
* ⛔ **R1 IS DROPPED** iff the CI on Δr²(1:1 − 40:1) **contains 0** on those rungs.
  **Interpretation committed in advance: the information is not in the tokens either. The pool is
  NOT the binding constraint, and no loss that merely reads the tokens can add what the encoder never
  encoded.** The follow-on becomes the encoder/corpus question (§8) — **not R1, and not a
  re-weighting.**
* 🔸 **DOWNGRADE** if only the **aggregate** rungs (`n_agents_all`, scene density) improve while the
  individuation and relative-motion rungs do not: R1's mechanism claim is unsupported, and the arm
  survives only as the §4.4 escalation.
* ⚠️ **VOID** if the oracle positive control fails — repair the instrument and read nothing (C79).

**Cost.** One frozen-encoder forward over banked val windows + 4 arms × ≥5 seeds × 11 rungs of
**closed-form** ridge. `ESTIMATED` **< 2 GPU-hours on the dev-box RTX 4060; ZERO on Thor, ZERO on any
pod.** No episode is selected (§9).

### 7.2 E-R1-1 — the training arm. **Runs only if E-R1-0 PROCEEDS.**

Two fresh S-W runs from step 0 (§5.1: there is no other legal entry), matched step budget, identical
seed, identical parity corpus, identical flags **except** `--masked-tokens` on/off.

⚠️ **On the step budget:** the "just train longer" refutation is at **~8,192 SAMPLES** in Didolkar et
al. ([arXiv:2408.09162](https://arxiv.org/abs/2408.09162), *"we do not find evidence of favorable
data scaling laws"*) — **not ~8 k steps, which is how the brief phrased it.** A step budget is chosen
for **cost**, not derived from that number, and no exponent is quoted for it (`CLAUDE.md`'s
learning-curve rule).

**Three reads:**

| read | tier | what it answers |
|---|---|---|
| **(a) mechanism** — the E-R1-0 ladder at **1:1** on both trunks | **T0** | did R1 put more into the tokens? |
| ⭐ **(b) deployment** — the same ladder at the **DEPLOYED 40:1** | **T0** | **is R1 a repair, or only a diagnosis?** |
| ⛔ **(c) NO-HARM** — `g_op_fwd_ade_m` and the WM's own loss, matched steps, paired episode-cluster bootstrap | **T0** | §3.3's kill criterion |

⭐ **Why (b) CAN succeed even though the pool is a mean — the precise mechanism, and a sharp
prediction.** An unweighted mean destroys **which token**, but it **preserves the sum**. If R1 pushes
the encoder to write a localized quantity **linearly** into token features, the pooled cell carries
`Σ` over its 40 tokens — for a **single** dominant object in a cell, that is the object's quantity
attenuated ~40×, and a linear ridge reads an attenuated linear code fine. **What a mean cannot do is
separate two objects inside one cell: their contributions add and are unrecoverable.**
⇒ **PRE-REGISTERED PREDICTION: R1's deployed-side gain appears on SINGLE-object-per-cell windows and
NOT on MULTI-object-per-cell windows.** If a gain appears uniformly across both, the mechanism claim
is wrong even if the number is good.

**PRE-REGISTERED OUTCOMES:**

* ✅ **R1 is a REPAIR** iff (b) improves with CI excluding 0 **and** (c) shows no harm.
* ⚠️ **R1 is a DIAGNOSIS ONLY** iff (a) improves and (b) does not. **Committed in advance, this
  outcome selects R3 — replace the unweighted mean with ATTENTIVE POOLING at the same 16 × 128
  output** (state_dim 2048 unchanged, geometry firewall intact, `readout.py:47-52`). Published
  support already in our own review: *V-JEPA, same frozen encoder, pooling choice only* — average
  pooling 56.7 (K400) / 50.1 (SSv2) → attentive **73.7 / 66.2**, **+17.3 / +16.1**
  ([arXiv:2404.08471](https://arxiv.org/abs/2404.08471), cited at
  `2026-08-17-O234-DESIGN-RESEARCH.md:727`). ⛔ **R3 is NOT proposed now** — naming it as the
  consequence of a specific outcome is what stops it becoming scope creep argued after the fact.
* ⛔ **R1 IS DROPPED** iff (c) shows harm on the WM objective. **That is the V-JEPA 2.1 Table 1 row
  happening to us (IN1K −9.6, SSv2 −10.3, §3.3), and the published recovery path is 163 M samples we
  do not have (§8). No "trade-off" reading is admissible.**

**Extra control, and it is not optional:** ⭐ a **matched-parameter placebo** — the same 1,940,864
extra `aux` parameters trained on a **position-shuffled** token target (token positions permuted
before scoring), which keeps the capacity and destroys the locality. **Without it, "R1 helped" cannot
be distinguished from "1.9 M more aux parameters helped."**

### 7.3 ⭐ E-R2-0 — validate the TARGET before believing anything. **ZERO TRAINING, ZERO MODEL.**

**Question.** Does the ego-compensated residual mark **agents**, or does it mark **buildings** (§6.3)?

**Method.** Compute the R2 target on a sample of val windows. Score per-token residual magnitude as a
detector of *"this token overlaps a GT dynamic-agent box"*, using `obstacle.offline` (**97.44 %** of
the corpus; **10 classes, all dynamic agents** — `CLAUDE.md`). Metric: AUC with the episode-cluster
bootstrap. ⚠️ This is a **LABEL-side** use of a privileged signal, which the 2026-08-03 binding rule
explicitly permits; **inference never sees it** and R2's head is never evaluated at inference at all.

**Controls.**

| control | what it is |
|---|---|
| ⭐ **TRIVIAL-PROXY #1** | **uncompensated flow magnitude** `\|f_obs\|`. **If raw flow does as well, the ego compensation adds nothing and R2 collapses to "predict optical flow"** — a weaker, already-published claim. |
| **TRIVIAL-PROXY #2** | **image row index / distance from the horizon alone.** A target that is really "how far below the horizon" is a road-geometry proxy we already have from the ego state. |
| **POSITIVE** | an **oracle target** rasterised from the GT boxes, pushed through the identical scoring pipeline — must score near 1.0, or the harness is broken (C79). |
| **NEGATIVE** | residual computed against a **RANDOMISED ΔT** (wrong ego transform) — must collapse the AUC. If a wrong transform scores as well, the compensation is doing nothing and §6.1's missing pitch/roll has swamped it. |

**PRE-REGISTERED OUTCOMES:**

* ✅ **R2 PROCEEDS** iff compensated-residual AUC beats **both** trivial proxies with paired
  episode-cluster-bootstrap CIs excluding 0, **and** the oracle positive control passes, **and** the
  wrong-ΔT negative control collapses.
* ⛔ **R2 IS DROPPED** iff it does not beat **uncompensated flow**. **Committed in advance: the target
  is then measuring parallax and texture rather than agents, and training a trunk to predict it would
  teach it the road-plane geometry the ego state already supplies.**

**Cost.** Flow over a val sample + a box join + AUC. `ESTIMATED` **a few dev-box GPU-hours; ZERO Thor,
ZERO pod.** No episode is selected.

### 7.4 E-R2-1 — the training arm. **Runs only if E-R2-0 PROCEEDS.**

**R2-cells FIRST** (16,899 params, no attention, no O(n²) risk — the cheapest training arm in the
design), on the same matched-step S-W protocol, with the **same no-harm kill criterion** as §7.2(c)
and the same matched-parameter placebo.

⭐ **E-R2-1 carries an interpretation the others do not, and it is why it must run before the
expensive pre-pool arms: if R2-cells improves the DEPLOYED 40:1 latent, then the OBJECTIVE was the
binding constraint and the pooling thesis is weakened.** Running the expensive pre-pool arm first
would leave that alternative untested and un-attributable — the §2 conflation, again.

R2-tokens and the joint **R1 ⊗ R2** cell run only after both single-lever arms have verdicts.

---

## 8. ⛔ The strongest argument against R1 — and the control that settles it

**The argument, stated at full strength (it belongs here, not in a footnote).** The one published
case of object structure **emerging** from our objective class **failed on a narrow corpus**: trained
on SSv2 alone — narrow, motion-centric — object-centric performance is *"almost chance-level"*; it
needs **HowTo100M-scale diversity, not volume** (Didolkar et al.,
[arXiv:2408.09162](https://arxiv.org/abs/2408.09162), cited at
`2026-08-17-O234-DESIGN-RESEARCH.md:583-586`). **Our 2,376 driving episodes are narrow in exactly
that way: one domain, one sensor rig family, one task.** And R1's own flagship precedent recovered
its global metrics only at **VisionMix-163M** (§3.3). ⇒ **If corpus narrowness is the binding
constraint, R1 will fail, and no loss placement, weight or head fixes it.**

⛔ **"The objectives just need more training" is REFUTED at three independent sources and is not
resurrected anywhere in this document:** the ~8,192-**sample** plateau with *"no evidence of favorable
data scaling laws"* (Didolkar); DINOv3's dense **locality DEGRADING** with longer training, which is
why Gram anchoring exists ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104),
`O234-DESIGN-RESEARCH.md:449`); and *"capacity scaling cannot recover the missing structure when the
objective fails to encode it"*.

### 8.1 What evidence distinguishes "our objective is wrong" from "our corpus is too narrow"

**First, the asymmetry that most people get backwards.** *Success* is not the ambiguous case —
**failure is.** If R1 succeeds, corpus narrowness was demonstrably not binding **for this capability**.
If R1 fails, the two hypotheses are indistinguishable from R1 alone, and we will have spent an S-W run
to learn nothing. ⇒ **the discriminator must run ALONGSIDE R1, not after it.**

⭐⭐ **THE CONTROL: run E-R1-0's pooling ladder on a FROZEN EXTERNAL HIGH-DIVERSITY ENCODER over the
SAME val windows.** Concretely DINOv3 (or DINOv2) ViT-B, frozen, applied to each of the three
sub-frames of our 9-channel stack and concatenated per token (so relative motion is linearly
available), then the identical 40:1 / 10:1 / 4:1 / 1:1 ladder with the identical random-projection
budget and the identical controls.

**The 2 × 2 of outcomes, committed in advance:**

| | DINOv3 tokens carry relative motion / individuation | DINOv3 tokens do NOT |
|---|---|---|
| **DINOv3 40:1 cells lose it** | ⭐ **THE POOL IS THE DESTROYER, AND THE INFORMATION IS PRESENT IN OUR IMAGES.** Our corpus is not too impoverished to contain it ⇒ **our objective/architecture is at fault, not the corpus. R1 (and R3) are warranted.** | — |
| **DINOv3 40:1 cells keep it** | the pool is survivable with a good enough encoder ⇒ the encoder is the lever, not the loss placement | — |
| **(right column)** | — | ⛔ **THE INFORMATION IS NOT LINEARLY PRESENT IN OUR IMAGES AT ALL, even to a 163 M-image encoder. That is the corpus/sensor verdict, and it kills R1, R2 and R3 together.** The follow-on is corpus composition or a sensor/geometry change — exactly the class `CLAUDE.md` records as needing AlpaSim or an external corpus. |

⭐ **Why this is the right control and a corpus-swap experiment is not:** it needs **no training at
all** — one frozen forward pass over the banked val windows — and it separates *"the information is
absent from our data"* from *"the information is present and our pipeline destroys it"*, which is
precisely the question. A corpus swap would cost an S-W run **and** break parity.

⚠️ **Its limits, declared:** DINOv3 is image-trained, so per-token motion is available to a *linear*
readout only through the 3-frame concatenation; a negative result on the relative-motion rungs is
therefore weaker evidence than a positive one. And DINOv3 is a different architecture, not just a
different corpus — so the right column is a joint statement about corpus **and** encoder class.
**Ops item:** confirm the weights are obtainable on the dev box (the `truststore.inject_into_ssl()`
route in MEMORY) before scheduling.

### 8.2 ⛔ "R1 succeeding for the wrong reason" — the expensive failure, and its test

R1 could improve the token-level probe simply because a dense reconstruction loss makes token
features **more linearly decodable in general** — a smoothing/linearisation effect with no object
structure whatsoever. Three guards, all already in §7:

1. **The differential prediction** (§7.2): a structure gain lands on **individuation-sensitive** rungs
   (≥2-agent windows, `lead_closing`) **more** than on **aggregate** rungs (`n_agents_all`, density).
   ⛔ **A uniform lift across both IS the wrong-reason signature and must be reported as such**, not
   as a win.
2. **The matched-parameter placebo** (§7.2): position-shuffled target, same capacity. Separates
   "locality helped" from "1.9 M parameters helped".
3. **The trivial-proxy control** (§7.1): `v0`-partialled everywhere, because C92 is the programme's
   most recent example of exactly this failure.

### 8.3 ⛔ Not proposed, per the brief, and I agree with the measurement

**PCGrad / CAGrad / Nash-MTL are NOT proposed.** They fire only at `cos < 0`; our most negative
measured pair is **−0.019** (`O234-DESIGN-RESEARCH.md:660`, `:917-919`), so they are **no-ops on our
geometry**; four papers find no gain over plain scalarisation (Kurin et al., NeurIPS 2022,
[arXiv:2201.04122](https://arxiv.org/abs/2201.04122)); and PCGrad measured **35× slower** on CelebA.
I re-located each of these at its citation and have no counter-measurement to offer. **I do not
disagree.**

---

## 9. Parity — how each proposal preserves it, and what is REFUSED

**The canonical train corpus is `physicalai-train-e438721ae894` (2,376 episodes, skip-hash
`f09e44db`). Anything that re-selects episodes breaks cross-arm comparability and is refused.**

| experiment | what it touches | parity status |
|---|---|---|
| **E-R1-0** (§7.1) | reads **banked val windows** and a **frozen** checkpoint; selects nothing, trains nothing | ✅ **untouched by construction** |
| **E-R2-0** (§7.3) | reads banked val windows + the `obstacle.offline` join (label-side only) | ✅ **untouched by construction** |
| **E-R1-1 / E-R2-1** (§7.2, §7.4) | fresh S-W runs on the canonical corpus; both arms identical except the one lever | ✅ preserved — **no episode is added, removed, reordered or re-hashed**; the skip-hash is asserted at launch as today |
| **R2's sidecar** (§6.2) | a per-`(episode_id, t)` **target** file over the **same** episodes | ✅ preserved — **it adds a target, not a selection.** ⚠️ A window with no sidecar entry must be **masked out of R2's loss term only**, never dropped from the batch (dropping would re-select through the back door) |
| **the DINOv3 diversity control** (§8.1) | a foreign frozen encoder over the **same** banked val windows | ✅ selects no episodes. It is compared **only to itself across pooling ratios**, never to a canonical arm — declared, per the "diagnostic arms are not canonical arms" rule |

⛔ **EXPLICITLY REFUSED, because each is re-selection wearing a different hat:**

1. filtering training windows by residual magnitude ("train on the interesting ones");
2. dropping low-optical-flow or stationary windows because the R2 target is degenerate there;
3. up-sampling multi-agent windows to make the individuation rungs easier to read;
4. any "low-ego-motion gating" of the R2 loss (the §6.3 confound mitigation) applied to **one** arm.
   ⚠️ If such gating is ever wanted it must be applied **identically to both arms**, declared in the
   launch line, and it still **may not drop an episode**.

---

## 10. The four metric families and the tier stamps — per quantity, with its EMITTER

⛔ **C95's rule applied directly: every quantity below names its TIER *and* its EMITTER, because two
same-named quantities live at two different eval tiers and the wrong-scope read is the durable error.**
`MEASURED` that the emitters exist and are distinct: `taniteval/tools/eval_four_families.py`
(the binding four-family path — *"drives `taniteval.rollout.collect` (LONGITUDINAL + LATERAL) and
`taniteval.hierarchy.run` (TACTICAL + STRATEGIC)"*, `:9-13`) and `taniteval/tools/t1_eval.py`
(*"T1 is the PRIMARY offline tier — a capability claim … requires T1 or better"*, `:5-8`; every T0
number it emits *"is stamped `\"tier\": \"T0\"`"*, `:10-11`).

### 10.1 ⛔ The blanket statement, so no reader has to infer it

**NO EXPERIMENT IN THIS SPEC PRODUCES A T1 NUMBER.** E-R1-0, E-R2-0, and every ladder read inside
E-R1-1 / E-R2-1 are **T0 — WM diagnostics, never driving performance.** A T1 capability claim
requires the full S-W → S-T ladder and `taniteval/tools/t1_eval.py`; it is gated behind a completed
ladder run and is **not** part of any go/no-go here. ⇒ **No result from this spec may be reported as
"the model drives better."**

### 10.2 The families, per experiment, never pooled

| family | quantity | **tier** | **emitter** | applicability at the T0 pre-gates |
|---|---|---|---|---|
| **LONGITUDINAL** — target speed | `ego_v0`, `ego_accel` rungs | **T0** | the E-R1-0 ladder (`pc6_linear_readout`, `intercept_col=-1`) | ✅ computable |
| **LONGITUDINAL** — distance keeping | `lead_gap`, **`lead_closing`**, **`lead_inv_ttc`** (time-gap / TTC) | **T0** | same | ✅ computable — ⭐ **and `lead_closing` / `lead_inv_ttc` ARE the primary read of E-R1-0**, since they are the rungs sitting at exactly 0.0000 today |
| **LATERAL** — heading / curvature / yaw-rate | `ego_yawrate`, `ego_curv` | **T0** | same | ✅ computable. ⚠️ **Both currently lack a positive control** (`LATENT_LINEAR_LADDER.md:234`) — building one is a named work item (§11), not a reason to omit the family |
| **LATERAL** — cross-track | — | — | — | ⛔ **NOT APPLICABLE at T0 with the reason and the n: a frozen-latent linear readout produces no predicted trajectory, so there is no track to be cross of.** Available only at T1 via `t1_eval.py`. Stated per rule 5, not silently dropped |
| **TACTICAL** — manoeuvre decision + goal setting | `sel_gap`, selected-vs-executed confusion | **T0 (log) ≠ the S-T gate's scope** | `taniteval.hierarchy.run` via `eval_four_families.py`; the **S-T gate** consumes `--gate-probes`/`X3_isolation`/`spectrum`/`x4_spectra` | ⛔ **NOT COMPUTABLE from a frozen S-W trunk — it needs `layer_tac` + `planner`, which S-W does not train.** ⚠️ **C95 applies verbatim here: `sel_gap` exists at two scopes and the trainer's T0 log key is NOT what the gate reads.** Any tactical claim must name which one |
| **TACTICAL** — a rung that *is* T0-computable | **decodability of the EXECUTED manoeuvre class from the frozen latent** | **T0** | ⛔ **DOES NOT EXIST — the ladder is regression-only today** | ⭐ **WORK ITEM, not an excuse** (binding rule 3): adding a classification rung to the ladder makes the TACTICAL family reportable at T0 for the first time, and it is cheap (a multinomial ridge on the same features). Named in §11 |
| **STRATEGIC** — route / goal quality | strategic goal + route setting | **T0/T1** | `taniteval.hierarchy.run`; S-S gate | ⛔ **NOT COMPUTABLE before S-S exists.** Declared with the reason. ⚠️ And when it is computed, the 2026-08-03 goal-input rule binds: **a supplied route on PhysicalAI is optimistic by construction** (our only route supplier is the ego's own future path) |

**Every family carries the paired episode-cluster bootstrap over the 40 val episodes
(`taniteval/ci.py`), on the same windows as the ADE it accompanies — never `overlapping_holdout_se`,
which biases the point estimate as well as the interval.** ⛔ ADE alone is never "the result": it is
one row of four.

---

## 11. ⛔ C94 — every producer→consumer join in this spec, and how it is exercised END-TO-END

C94's root-cause class is **a fixture that models the CONSUMER'S EXPECTATION instead of the
PRODUCER'S OUTPUT** — `read_sw_admission` looked for a top-level `sigma_2s_m` while the producer
wrote `references_and_ratios.sigma_perax_2s_m`; **name and nesting both differed, a hand-written
fixture certified the join, and the pre-registered reopening path could not return `FUNDED` for any
measurement.** ⇒ **RULE: a fixture standing in for another component's output must be GENERATED BY
THAT COMPONENT.**

**This spec's single highest-risk join is R2's sidecar, and it has exactly C94's shape.** Below is
every join, with the end-to-end exercise that must pin it. **None of these may be pinned by a
hand-written dict.**

| # | producer | consumer | ⚠️ the C94 risk | **how it is exercised end-to-end** |
|---|---|---|---|---|
| **J1** ⛔ **highest risk** | the offline **R2 sidecar builder** (§6.2) | the trainer's dataset, reading per-`(episode_id, t)` residual targets | **exactly C94**: a key name and a nesting level agreed by two authors, never executed together. A wrong nesting yields *all-zero targets*, which trains fine and looks like "R2 did nothing" | the pin **runs the real builder** on ≥2 real episodes, writes a real sidecar, and has the real dataset read it back; it asserts a **non-degenerate** target (variance > 0 and non-zero on ≥1 token) — because an all-zero read is the failure mode a shape-only assert passes |
| **J2** | `encode_window(..., return_tokens=True)` (`v6.py:3691`) | the R1 `TokenMaskedPredictor` and the E-R1-0 ladder | shape/order of `[B, W, 640, 384]`; a silent transpose gives a scrambled token grid that still trains | the pin calls the **real** `encode_window` on a real `V6Stack` (already the idiom in `tests/test_v6_agent_slots.py:285-291`, `MEASURED` green today) and asserts the **spatial** grid by reconstructing `16×40` and checking a planted spatial pattern survives |
| **J3** | `V6Stack.named_parameters()` | `group_of` → `_GROUP_PREFIXES` → `apply_stage_freeze` | a missing `("masked_tokens.", "aux")` entry | ✅ **already loud, not silent** — `group_of` **RAISES** on an unmapped parameter (`v6.py:3667-3671`). This is the join that is already built the right way and is the model for the others |
| **J4** | `obstacle.offline` join (pod-side, `stack/scripts/build_obstacle_join.py`) | E-R2-0's AUC scorer | box coordinate frame / units; a frame mismatch produces a plausible-looking AUC near 0.5 | the **oracle positive control** (§7.3) is the end-to-end exercise: rasterise GT boxes through the *same* join and the *same* scorer; **if it does not score near 1.0 the join is wrong**, and that is the whole point of requiring it |
| **J5** | `pc6_linear_readout.ridge_fit` | E-R1-0's per-rung table | ⚠️ **`intercept_col` defaulting to `None`** (§7.1) — the biased floor is the *default*, silently | assert the argument **out of the emitted `config.json`**, not out of the launch script; and re-run one **banked** rung and reproduce its committed number to 1e-4 (the reproduction `ll1_ladder.py` already asserts) |

⭐ **The generalisation I am pre-registering with the design: for every one of J1–J5, the
acceptance test must FAIL if the producer is changed and the consumer is not.** A test that passes
against a literal written by the consumer's author proves only that the consumer agrees with itself.

---

## 12. Open items, escalations, and what I did NOT do

### 12.1 ⛔ ESCALATIONS — these need a decision or an owner, and they are NOT parked in a README

1. **⛔ `ISOLATION_MATRIX` ruling on R2 (§5.4).** Is an **ego-pose-derived geometric target** `aux`
   or `interp`? **The ruling decides whether R2 can shape the trunk at all.** Prerequisite for
   scheduling E-R2-1. My reading is `aux`; it is not mine to settle.
2. **⚠️ `pc6_linear_readout.py` — the instrument E-R1-0 depends on — lives ONLY in
   `…/incoming/2026-08-17-probe-positive-control/code/`.** It is not in `taniteval/` or `stack/`, and
   `taniteval/tests/test_ridge_intercept_penalty.py:29-44` reaches across the repo **by path** and
   **text-extracts the function source** to test it. `MEASURED` (two probes): a repo-wide grep finds
   **10 separate `ridge_fit*` forks** across `incoming/` directories. **This is operating-standard
   rule 3 territory** — a load-bearing instrument that is not packaged. Not mine to move mid-flight;
   raising it because E-R1-0 is about to depend on it.
3. **⚠️ R1's step-time cost is NOT MEASURED and could make it unaffordable on Thor** (§4.2):
   attention over 640 tokens vs O3's 16 is `ESTIMATED` at ~1,600× O3's attention FLOPs. **A ~10-minute
   step-time A/B on the dev-box GPU settles it and must run before R1 is scheduled.** It is the
   number most likely to change the plan and it is not in this document.
4. **Optical-flow estimator choice for R2** (§6.2 step 4) — an ops decision I did not make. E-R2-0
   must run with the estimator that will actually be used.
5. **DINOv3/DINOv2 weight availability on the dev box** (§8.1) — confirm before scheduling the
   diversity control.

### 12.2 Work items named rather than excused (binding rule 3)

* a **T0-computable TACTICAL rung** — manoeuvre-class decodability from the frozen latent; the ladder
  is regression-only today (§10.2);
* **positive controls for the two LATERAL rungs** (`ego_yawrate`, `ego_curv`), which have none
  (`LATENT_LINEAR_LADDER.md:234`).

### 12.3 What I did NOT do — stated plainly rather than scoped away

* ⛔ **I changed nothing in the training path, and added no module to `stack/`.** R1's
  `TokenMaskedPredictor` and R2's `ResidualFlowHead` exist **only** in
  `code/measure_r1r2.py`, as candidate modules built to be *measured*. This is a spec, not an
  implementation.
* **I spent ZERO GPU.** All parameter measurements are CPU (`CUDA_VISIBLE_DEVICES=""`).
* **I did not re-measure the 37.77 px lead width** (§1.4) — it needs the obstacle join. `INHERITED`,
  marked as such, and not load-bearing for the pooling ratio, which comes from source alone.
* **I did not re-run the full suite.** I changed no code, so I make no claim about it beyond what I
  measured: `MEASURED` by me today, `stack/tests/test_readout_onnx_pool.py` +
  `test_geometry_configurable.py` + `test_v6_agent_slots.py` = **107 passed** — the three files that
  pin the claims in §1. The coordinator's suite-wide 3816/0 is `INHERITED`.
* **The V-JEPA 2.1 magnitudes in §3.3 come from one read of the paper's Table 1**; the *direction*
  is corroborated at a second source. The brief's `52.5 → 69.0` **did not survive** and is not used.

---

## 13. Deliverable manifest

| artifact | where it lives | in only ONE place? |
|---|---|---|
| **This spec** — `POOLING_BOTTLENECK_R1R2.md` | `repo:TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-pooling-bottleneck-R1R2/POOLING_BOTTLENECK_R1R2.md` | no — **staged in git** |
| **`code/measure_r1r2.py`** — builds `TokenMaskedPredictor` + `ResidualFlowHead`, re-derives the pooling geometry from the live `V6Config`, dumps the group/ladder audit. Re-runnable, CPU-only | `repo:…/2026-08-18-pooling-bottleneck-R1R2/code/measure_r1r2.py` | no — **staged in git** |
| **`raw/r1r2_params.json`** — the MEASURED parameter table, geometry block, and ladder audit behind §1.1, §1.7, §4.2, §5.3, §6.4 | `repo:…/2026-08-18-pooling-bottleneck-R1R2/raw/r1r2_params.json` | no — **staged in git** |

⛔ **Nothing produced here lives on a pod, in a worktree, or only in my context.** No file in
`stack/`, `taniteval/`, or any training path was created or modified.
