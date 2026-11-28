# ⛔ THE C106 ADVERSARIAL RE-READ — HALF OF IT SURVIVES, HALF OF IT DOES NOT

**Date** 2026-08-18 · **Branch** `agent/arch-inf-20260803` · **Posture** REFUTATION
**Tier** **T0-DIAGNOSTIC** throughout — a world-model diagnostic on frozen latents, **never driving
performance**. **Estimator** paired episode-cluster bootstrap (`taniteval/ci.py` `_draws` /
`episode_index`, via `er10_pool_ladder.paired_delta_r2c`). ⛔ `overlapping_holdout_se` is never
imported. **Ridge** `intercept_col=-1` on every fit (C92). **Parity** SELECTS NOTHING — banked
window caches read verbatim.

> **C106 claims** (`Project Steering/RETRACTION_LOG.md:5634`) that our encoder's random
> initialisation beats its trained self **3.6× on BOTH rungs, on all three seeds**
> (`ego_v0` 0.1894 vs 0.05207; `lead_gap` 0.0176 vs 0.00490) ⇒ *"the objective is not failing to
> add geometry; it is subtracting it."*

---

## ⭐ THE VERDICT IN ONE TABLE

| C106's component | verdict | the measurement that decides it |
|---|---|---|
| **`ego_v0`: random init reads better than trained** | ✅ **SURVIVES** — and is now *stronger* than C106 stated, because it has an interval | Δr²c **+0.150 [+0.055, +0.226]**, `p(Δ>0)=1.000`, in C106's own cell; **positive in 27/27 cells** across 3 init seeds × 3 projection seeds × 3 ridge seeds |
| **`lead_gap`: same claim, same 3.6×** | ⛔ **NOT SUPPORTED** | **0 of 27 cells CI-separated**; `p(Δ>0)` only **0.71–0.76**; **the sign flips** (9/27 cells negative). The mean ratio falls from 3.6× to **2.0×** once the ridge inner split is re-drawn |
| **"3.6×", the number itself** | ⛔ **WITHDRAW THE RATIO** | It is a ratio of two `corr²` values measured at **incomparable operating points** (ours ≈ a constant predictor at `pred_sd/gt_sd` 0.014, randenc a live readout at 0.89) and it moves to **2.8× / 2.0×** under a re-drawn inner split |
| **"our arm's readout is itself a flat line"** (C106 limit 2) | ⛔ **AN ARTIFACT OF ONE INNER SPLIT** | At `ridge_seed=2` the SAME arm chooses α=10 and reads `pred_sd/gt_sd` **0.93–0.95** — not flat, and its r²c rises to **0.069–0.077** |
| **The LayerScale mechanism** (*"random init ≈ a random linear map of raw pixels"*) | ✅ **VERIFIED FROM WEIGHTS, NOT ONLY FROM SOURCE** | random-init residual fraction **0.0002**, cos(full, linear) **1.0000**; the trained arm HAS moved (`ls` **70× init**, residual **0.38**) — so the comparison IS "random linear map vs trained deep net", as C106 assumed |
| **"the deficit was already established by step 9 250"** | ✅ **SUPPORTED, from a source C106 did not use** | the programme's own `cells`/z_op ladder at **step 2 000** reads `ego_v0` **0.1346** and `lead_gap` **0.0123**, falling to **0.0801 / 0.0059** by step 9 000 and flat after |
| **"the missing measurement is an artifact-RETRIEVAL item"** | ⛔ **CORRECTION — IT IS UNAVAILABLE, NOT UNRETRIEVED** | Thor holds **only** 9 250 / 10 000 / 11 250 / 12 000 (+ one ~step-9 100 snapshot). A whole-filesystem search found **nothing earlier anywhere.** The token-level step-0→9 250 sweep needs a **new run**, not a download |

### ⛔⛔ AND THE FINDING THAT REFRAMES BOTH C104 AND C106

> **OUR TRAINED ARM IS NOT CI-SEPARATED FROM ITS OWN MATCHED-RANDOM NULL.**
> `lead_gap`: **0/9 cells separated** (Δ vs null +0.0038, `p=0.89`). `ego_v0`: **3/9**, and in
> C106's own cell **Δ = +0.052 [−0.0007, +0.206], NOT separated.**
> The random-init arm **is** separated — `ego_v0` **9/9 cells, Δ +0.164 [+0.074, +0.367]**.

⇒ The admissible statement is **not** *"random init is 3.6× better"*. It is:
**on `ego_v0`, the random initialisation reads ego speed above the matched-random floor and the
trained encoder does not; on `lead_gap`, neither does.** That is a **signal / no-signal** statement
on ONE rung, and it replaces a quantitative ratio whose denominator is indistinguishable from noise.
**A ratio is not interpretable when its denominator is not separated from the null** — which is
exactly the objection the brief raised, now measured rather than argued.

---

## 0. THE REPRODUCTION GATE — this harness agrees with the PRODUCER first (C94)

⛔ Nothing below is readable unless this passes. `build_features`, `fit_one`, `paired_delta_r2c`
and the target loader are **IMPORTED from `er10_pool_ladder`**, the producer of the numbers under
attack — an adversarial re-read that swaps in its own solver cannot separate *"the claim is wrong"*
from *"my code is different"*.

**MEASURED (ours)** · `raw/gate_ours_base.json` · banked cache, banked α grid, `--ridge-seeds 0`:

| rung | banked `fals_ours.json` | this harness | Δ |
|---|---:|---:|---:|
| `ego_v0` | 0.05207 | **0.05207** | **0.00000** |
| `lead_gap` | 0.00490 | **0.00490** | **0.00000** |
| `lead_closing` | 0.00000 | **0.00000** | **0.00000** |

---

## 1. ⭐ ATTACK 1 — THE 3.6× WAS QUOTED WITHOUT ITS ESTIMATOR. HERE IT IS.

⛔ **C106's bracket `0.1894 [0.1736, 0.2011]` is the spread over three PROJECTION SEEDS** — the
instrument's own noise — **not an episode-cluster bootstrap over the 70 eval clusters.** CLAUDE.md:
*"Never quote an interval without its estimator."* The two quantities answer different questions and
the decision-grade one was never computed.

It is computable, because `randenc` is built from the **same row index** as the banked cache: the
identity of the eval windows is **asserted, not assumed** (`stage delta` refuses to pair on a
mismatched key). **MEASURED (ours)** · `raw/delta_randenc_s*_{base,wide}_vs_ours_*.json` ·
`n = 1507 / 1362` windows in **70 episode clusters**, 2 000 draws.

### `ego_v0` — C106 SURVIVES

| init seed | Δr²c, C106's own cell (`p0|r0`) | separated | `p(Δ>0)` | cells positive | cells sep & pos |
|---|---:|---|---:|---:|---:|
| `randenc_s0` | **+0.15029 [+0.05405, +0.22551]** | ✅ | **1.000** | 9/9 | 6/9 |
| `randenc_s1` | **+0.11829 [+0.03592, +0.18474]** | ✅ | **1.000** | 9/9 | 6/9 |
| `randenc_s2` | **+0.14590 [+0.04610, +0.22541]** | ✅ | **0.998** | 9/9 | 6/9 |

**27/27 cells positive, sign never flips, the majority separate.** The 3 non-separating cells per
seed are the ones where **both** arms collapsed to the α grid edge (§2). ⇒ **The DIRECTION of
C106's `ego_v0` claim is now supported at the programme's decision-grade estimator, which it was
not before.**

### `lead_gap` — C106 IS NOT SUPPORTED

| init seed | Δr²c (`p0|r0`) | separated | `p(Δ>0)` | cells positive | cells sep & pos |
|---|---:|---|---:|---:|---:|
| `randenc_s0` | +0.01125 [−0.03244, +0.04868] | ❌ | 0.714 | 6/9 | **0/9** |
| `randenc_s1` | +0.01391 [−0.03208, +0.05281] | ❌ | 0.758 | 6/9 | **0/9** |
| `randenc_s2` | +0.01455 [−0.03554, +0.04749] | ❌ | 0.751 | 6/9 | **0/9** |

**0 of 27 cells separated. The sign flips in 9 of 27.** Partialling `v0` out does not rescue it
(0/27 separated). ⇒ ⛔ **"3.6× on BOTH rungs" must become "on `ego_v0`".** `lead_gap` is a
coin-flip at this n.

⚠️ `lead_closing` is Δ ≈ −0.00001, 0/27 separated — both arms read nothing, as C106 says.

---

## 2. ⭐ ATTACK 2 — THE α WAS PINNED. WIDENING IT DOES NOT RESCUE OURS, BUT IT DOES KILL C106's "FLAT LINE" LIMIT

C106 reports `alpha_chosen = 1e7` **at the grid edge** on every seed and both rungs. A sweep that
never bracketed its optimum has not selected a readout. Widened to **1e13** (15-point grid) and
re-drawn the inner split at **3 ridge seeds** — the selector uses ONE 25 %-of-clips draw and our
arm's inner-MAE curve is **non-monotone across four decades**, so the draw is a live variable C106
held fixed.

**MEASURED (ours)** · `raw/adv_ours_{base,wide}.json`, `raw/adv_randenc_s*_{base,wide}.json`:

| arm | rung | C106 (1 ridge seed, α≤1e7) | **3 ridge seeds, α≤1e7** | **3 ridge seeds, α≤1e13** |
|---|---|---:|---:|---:|
| ours | `ego_v0` | 0.05207 | **0.05916 ± 0.01020** | **0.05956 ± 0.00992** |
| ours | `lead_gap` | 0.00490 | **0.00402 ± 0.00124** | **0.00408 ± 0.00128** |
| `randenc_s0` | `ego_v0` | 0.20110 | **0.16519 ± 0.05092** | **0.16441 ± 0.05201** |
| `randenc_s0` | `lead_gap` | 0.01597 | **0.00800 ± 0.00570** | **0.00800 ± 0.00570** |

⇒ **Widening α changes essentially nothing** (Δ ≤ 0.0008): the α-edge attack **FAILS to overturn
the ordering**, and that is a point in C106's favour. **But re-drawing the inner split moves the
ratio from 3.6× to 2.8× (`ego_v0`) and 3.6× to 2.0× (`lead_gap`)**, and on `lead_gap` the spread
(±0.0057) is **larger than the gap** (0.0039).

### ⛔ AND THE OPERATING POINTS ARE NOT COMPARABLE — which is why the ratio has to go

Per-cell, `ego_v0` (`raw/adv_ours_wide.json`, `raw/adv_randenc_s0_wide.json`):

| cell | ours α | ours `pred_sd/gt_sd` | ours r²c | randenc α | randenc `pred_sd/gt_sd` | randenc r²c |
|---|---:|---:|---:|---:|---:|---:|
| `p*|r0` | 1e13 | **0.0000** | 0.0522–0.0533 | 1e3 | 0.89–0.90 | 0.1953–0.2052 |
| `p*|r1` | 1e13 | **0.0000** | 0.0522–0.0533 | **1e13** | **0.0000** | 0.0893–0.0937 |
| `p*|r2` | **10** | **0.93–0.95** | **0.0692–0.0769** | 1e3 | 0.89–0.90 | 0.1953–0.2052 |

Three things follow, and each is a correction to C106:

1. ⛔ **C106's limit 2 — *"our arm's readout is itself a flat line"* — is SEED-SPECIFIC.** At
   `ridge_seed=2` the identical arm produces a well-scaled readout (`pred_sd/gt_sd` 0.95) and its
   r²c **rises to 0.077**. The flatness is a property of one inner-split draw, not of the arm.
2. ⛔ **At α = 1e13 BOTH arms produce numerically the constant predictor** — and the guard proves
   it: `K1_delta` is **identical to 5 decimal places (+0.1853 / +0.0416)** for the two arms, because
   both errors ARE the constant's error. Yet `r²_ceiling` still reads 0.052 vs 0.094. ⇒ **the
   headline metric can be non-zero for a readout with literally zero predictive power** (`R2`
   −0.005). Any future quote of `r2_ceiling` must carry `pred_sd/gt_sd` beside it.
3. ⭐ **At matched, well-scaled operating points the ordering still holds on `ego_v0`** (0.077 vs
   0.205) — which is the honest version of the surviving claim.

⚠️ **A fourth fact that belongs with any K1 statement about these arms.** When either arm is made
non-flat, its K1 delta becomes **CI-separated in the FAILING direction** (ours α=10:
K1 **+1.19, separated**; randenc α=1e3 on `lead_gap`: **+0.91, separated**). So it is not merely
that *"neither arm passes K1"* — **both are significantly WORSE than predicting the median.**

---

## 3. ⭐ ATTACK 3 — C106's LayerScale MECHANISM, VERIFIED FROM SOURCE **AND** FROM THE WEIGHTS

C106's interpretive force comes from *"at initialisation all 12 blocks contribute ~1e-5, so the
encoder is approximately `RMSNorm(patch_conv(x) + pos)` — a fixed random LINEAR map of raw patch
pixels"*. **That interpretation only holds if the TRAINED arm's LayerScale has MOVED**, and C106
never checked. It has.

**Source** (verified, not retyped): `stack/tanitad/models/encoder.py` — `Block5.__init__(…,
ls_init: float = 1e-5)` at `:302`, `self.ls1 = nn.Parameter(ls_init * torch.ones(d))` `:305`,
`self.ls2` `:312`, residual `x + self.ls1 * self.attn(...)` / `x + self.ls2 * self.mlp(...)`
`:315-316`, and `ViT5Encoder.__init__(…, ls_init: float = 1e-5)` at `:331`.

**MEASURED (ours)** · `raw/layerscale.json` · 64 REAL banked frames, fp32, every `ls1`/`ls2` forced
to zero to obtain the exact linear path:

| arm | mean \|ls\| | × init | ‖full − linear‖ / ‖linear‖ | cos(full, linear) |
|---|---:|---:|---:|---:|
| random init (3 seeds) | 1.000e-05 | **1.0×** | **0.0002** | **1.0000** |
| trained @ 9 250 | 7.271e-04 | 72.7× | 0.3801 | 0.9276 |
| trained @ 10 000 | 6.930e-04 | 69.3× | 0.3863 | 0.9253 |
| trained @ 11 250 | 6.957e-04 | 69.6× | **0.3978** | 0.9207 |

⇒ ✅ **Both halves of C106's mechanism hold.** The random arm is the linear map **to 0.02 %** —
the strongest possible form of "raw pixels through a random linear map". And the trained arm is a
genuinely deep net (70× LayerScale, 38 % of its output produced by the blocks), so the two arms are
the comparison C106 assumed. ⚠️ One precision point: `RMSNorm` applies a per-token gain, so
*"linear"* is exact only up to a per-token rescaling — worth stating, not worth retracting.

---

## 4. ⛔ ATTACK 4 — WHAT THE PROBE CORPUS LIMITS

Every number here is **T0-DIAGNOSTIC on the 130-episode probe corpus** (2 809 windows, 1 302 train
/ 1 507 eval, **70 episode clusters**) — **not** the 40-episode parity val set, and never driving
performance. Three consequences, stated rather than left implicit:

1. **The clusters, not the windows, set the resolution.** 70 clusters is what the CI is over; the
   `lead_gap` non-separation in §1 is a statement about *this* n, not a proof of no effect.
2. **No T1 number is derivable from any of this.** A registry row quoting §1 as evidence about
   driving would be malformed (EVAL_DOCTRINE §1.12).
3. **The four metric families do not apply here and saying so is not a dodge.** These are latent
   readouts, not trajectory evaluations; `four_families.py` consumes a trajectory dump. The
   longitudinal *shadows* (`lead_gap`, `lead_closing`) are what a T0 probe can carry, and they are
   reported per rung, never pooled.

---

## 5. ⭐⭐ ATTACK 5 — THE STEP SWEEP. IT CANNOT GO BACKWARD, AND THE ANSWER CAME FROM A DIFFERENT LADDER.

### 5.1 ⛔ A CORRECTION: THE MISSING CHECKPOINTS DO NOT EXIST

C106 calls the step-0→9 250 sweep *"blocked on checkpoints that are not local — an
artifact-retrieval item, not a compute one."* **MEASURED (ours, read-only `ssh -n` probes, zero
GPU):**

| location | v6F S-W checkpoints present |
|---|---|
| `thor:/home/nvidia/ckpt_snaps/` | **9 250, 10 000, 11 250, 12 000** only |
| `thor:/home/nvidia/experiments/v6F-SW-30k/` | `ckpt.pt` (live, ~13 k) + `ckpt_step10000.pt` |
| `thor:/` whole-filesystem, `*v6F*` > 100 MB | the four above + `/home/nvidia/v6F_snap_fp16.pt`, mtime **2026-08-16 22:29**, i.e. ~1 h before the 9 250 snap ⇒ **≈ step 9 100** at 26.5 s/step |
| repo / dev-box scratchpad | 9 250, 10 000, 11 250 |

⇒ **Nothing before ≈ step 9 100 exists anywhere.** ⏳ **ESCALATION: the token-level early-window
measurement is a NEW RUN, not a download.** The item should be re-classified in the
pre-registration, and a **dense early snapshot cadence** should be a launch condition of any future
S-W arm — otherwise this exact question will be unanswerable again.
*(Step 12 000 IS new relative to C106 and is pullable, but it only extends the window FORWARD, which
is the direction C106 already measured as flat. It is not worth 673 MB over Thor's WiFi.)*

### 5.2 ⭐ BUT THE PROGRAMME ALREADY HAS A STEP-2 000 POINT — ON THE DOWNSTREAM LATENT

**MEASURED**, re-read by me from raw JSON (not from prose):
`…/incoming/2026-08-18-ladder-3seed/raw/reread_centred/ll3_s0{2000,9000,9250}.json`,
`ll3_s1{0000,1250}.json`, `ll3_nullmatched.json` — the `cells` feature set (the **z_op** state,
2 049 features), `fit_mode: centred`, 3 seeds, same probe corpus, 70 clusters:

| step | `ego_v0` r²c | `lead_gap` r²c | `nearest_any` r²c | `pred_sd/gt_sd` (`ego_v0`) |
|---:|---:|---:|---:|---:|
| **2 000** | **0.1346 ± 0.0228** | **0.0123 ± 0.0041** | 0.1034 ± 0.0066 | 0.498 |
| 9 000 | 0.0801 ± 0.0133 | 0.0059 ± 0.0013 | 0.0959 ± 0.0032 | 0.564 |
| 9 250 | 0.0885 ± 0.0116 | 0.0056 ± 0.0014 | 0.0949 ± 0.0031 | 0.535 |
| 10 000 | 0.0913 ± 0.0120 | 0.0050 ± 0.0013 | 0.0965 ± 0.0031 | 0.552 |
| 11 250 | 0.0901 ± 0.0114 | 0.0069 ± 0.0020 | 0.0977 ± 0.0019 | 0.528 |
| **matched-random null** | 0.0007 | 0.0001 | 0.0007 | 0.009 |

⇒ ⭐ **On the two rungs C106 quotes, readability at the downstream latent FALLS 1.5×–2.1× between
step 2 000 and step 9 000, then is FLAT.** That is the missing direction: it is **consistent with
C106's "already established by 9 250"** and it says the loss happens **early**.
⚠️ **Three limits, and they matter.** (a) This is **z_op, not encoder tokens** — a different scope,
and the readout trains too, so it conflates encoder and readout. Quoting it as a token-level result
would be the `df`-trap family. (b) **One point before 9 000** ⇒ a trajectory cannot be fit and **no
exponent is admissible** (CLAUDE.md). (c) `nearest_any` is **flat across the same window**
(0.1034 → 0.0977), so the decline is **rung-specific, not a global collapse** — which is itself
evidence against the broadest reading of "the objective subtracts geometry".

---

## 6. ⛔ THE MANDATORY CONTROLS

### 6.1 POSITIVE CONTROL — and the one C106/C104 used **cannot fire at the deployed pool**

⚠️ **PC-2OBJ IS THE WRONG CONTROL FOR THIS BATTERY, AND I RAN IT FIRST AND PROVED IT.** Its two
planted tokens carry **opposite signs inside ONE deployed 4×10 cell**, so the deployed average
**cancels them exactly**. It is a *pooling-ratio* contrast (E-R1-0: p40 0.0000 → p1 0.9998) and at
p40 — the only arm this battery reads — it is **inert by construction**. **MEASURED:**
`adv_ours_pc2obj` reproduces `adv_ours_wide` to **5e-05**. A control that cannot fire is not a
control, and the banked `0.0000 → 0.9998` is **not** evidence that the ladder can see a signal
*inside* the deployed cell.

⇒ The controls that CAN fire at p40 were run on **both** arms — **PC-LOCAL** (a 2×2 token block
wholly inside one deployed cell, same sign, surviving the 40:1 average at 4/40 amplitude) and
**PC-DIST**. **MEASURED (ours)** · `raw/adv_{ours,randenc_s0}_pc{local,dist}.json`, planted at
1.0 × the arm's own token sd, 9 cells each:

| arm | control | `ego_v0` r²c | `lead_gap` r²c | `lead_closing` r²c | K1 PASS |
|---|---|---:|---:|---:|---:|
| **ours (trained)** | PC-LOCAL | **1.00000** | **0.99990** | **1.00000** | **9/9** |
| **ours (trained)** | PC-DIST | **1.00000** | **1.00000** | **1.00000** | **9/9** |
| `randenc_s0` | PC-LOCAL | 0.98607 | 0.96660 | 0.99350 | **9/9** |
| `randenc_s0` | PC-DIST | 1.00000 | 1.00000 | 1.00000 | **9/9** |

⇒ ⭐ **The instrument has full power on OUR OWN TRAINED TOKENS through the DEPLOYED pool:
0.0596 → 1.0000.** *"Ours reads nothing"* is therefore a statement about the tokens, not about the
ladder — which is the thing E-R1-0's control was cited for and could not actually establish at p40.

### 6.2 MATCHED-RANDOM NULL — the floor, and where each arm sits on it

**MEASURED (ours)** · `raw/adv_{ours,randenc_s0}_null.json` (`--randomise-features 4242`, per-feature
mean/sd taken from the arm's own train rows) and the paired deltas in
`raw/delta_*_vs_*_null.json`:

| rung | null floor r²c | ours | ours vs null | randenc_s0 | randenc vs null |
|---|---:|---:|---|---:|---|
| `ego_v0` | **0.00027** | 0.0596 | **3/9 cells separated**; C106's cell **NOT separated** (Δ +0.052 [−0.0007, +0.206]) | 0.1644 | ✅ **9/9 separated** (Δ +0.164 [+0.074, +0.367]) |
| `lead_gap` | **0.00029** | 0.0041 | ⛔ **0/9 separated** (Δ +0.0038, p 0.89) | 0.0080 | ⛔ 0/9 separated (Δ +0.0077, p 0.945) |
| `lead_closing` | 0.00042 | 0.0000 | 0/9 | 0.0000 | 0/9 |

⚠️ The two arms' nulls are numerically identical — by construction: after the ladder's per-column
z-scoring the matched Gaussian draw is the same matrix regardless of source arm. That is a property
of the control, not a bug, and it makes it a single shared floor.

### 6.3 TRIVIAL-PROXY CONTROL (`v0` partialled out)

**MEASURED (ours)** · same files, `r2_ceiling_partial_v0`, mean ± sd over 9 cells:

| rung | ours raw | ours partial-`v0` | randenc raw | randenc partial-`v0` |
|---|---:|---:|---:|---:|
| `lead_gap` | 0.00408 ± 0.00128 | **0.00660 ± 0.00063** | 0.00800 ± 0.00570 | **0.01960 ± 0.00736** |

⇒ Both arms **rise** when `v0` is removed, and the ordering is unchanged — so the `lead_gap` gap is
not a speed proxy. ⛔ **But the paired Δ remains 0/27 separated after partialling**, so this does
not restore the `lead_gap` claim.

---

## 7. ⭐⭐ WHAT I FOUND THAT C106 DID NOT LOOK FOR — THE TRAINED TOKEN FIELD IS RANK-COLLAPSED

C106 diagnoses *"the objective is subtracting geometry"*. There is a second reading of the same
numbers with a **different remedy**, and it is measurable.

**MEASURED (ours)** · `raw/rank.json` (384 real banked frames through er10's OWN `pool_tokens` and
`make_projection`) and confirmed directly on the **banked cache the ladder actually reads**:

| arm | token-channel eff. rank (of 768) | top direction's variance share | pooled p40 eff. rank | **z-scored DESIGN-matrix eff. rank** | final RMSNorm gain max/mean |
|---|---:|---:|---:|---:|---:|
| random init (3 seeds) | **67.1 – 68.2** | 0.29 | 9.9 – 10.2 | **16.0 – 16.9** | 1.00 |
| **trained @ 11 250** | **1.22** | **0.976** | 4.63 | **6.73** | 1.02 |

*(Independently on the banked `cache_tok11250` tokens: token-channel eff. rank **1.217**, top
direction **97.6 %**, pooled p40 eff. rank 4.76 — so this is not a probe artifact.)*

⇒ **After S-W training, 97.6 % of ALL token-channel variance sits in ONE direction, and the matrix
the ridge sees has ~6.7 usable dimensions against ~16 for the random init.** The final RMSNorm gain
is **not** the cause (max/mean 1.02) — it is the block outputs themselves.

### The discriminator: SUBTRACTED or merely SWAMPED?

If the geometry were still in the span but pushed into low-variance directions, **PCA-whitening**
would recover it — and the remedy would be conditioning at the readout, not the objective.
**MEASURED (ours)** · `raw/whiten_{ours,randenc_s0}.json`, whitening fitted on probe-train rows only:

| rung | arm | K = none | K = 16 | K = 64 | K = 256 |
|---|---|---:|---:|---:|---:|
| `ego_v0` | ours | 0.0596 | 0.0524 | **0.0684** | 0.0614 |
| `ego_v0` | randenc | 0.1644 | 0.0791 | **0.1944** | 0.1596 |
| `lead_gap` | ours | 0.0041 | 0.0000 | 0.0044 | **0.0128** |
| `lead_gap` | randenc | 0.0080 | 0.0007 | 0.0121 | **0.0283** |

⇒ **Whitening lifts BOTH arms and closes NOTHING.** `lead_gap` triples for ours (0.0041 → 0.0128)
and for randenc (0.0080 → 0.0283); the ratio stays ~2.2×. **K1 remains 0/9 at every K for both.**
⇒ **The rank collapse is real and large, but it is NOT the explanation of the ours-vs-randenc gap.**
C106's "the representation got worse for a linear readout" is not an artifact of conditioning.

⭐ **Why this still matters more than the ratio does:** a rank-1 token field is a **specific,
falsifiable, instrumentable** defect with a cheap monitor (it can be logged per step from step 0 at
no GPU cost), whereas *"the objective subtracts geometry"* is a description. ⏳ **RECOMMENDED (PI /
orchestrator): log token-channel effective rank in `train_v6_staged.py` beside the existing z_op
`spectrum` block.** The trainer already logs a z_op spectrum every 200 steps — and re-read from
`raw/…/v6F-SW-30k_train_log.jsonl`, **z_op effective rank has been ~8–30 of 2 048 since step 200**,
with no clean trend, so the z_op monitor does **not** cover the token-level effect and a new one is
needed.

---

## 8. SECOND JOB — THE `apply_stage_freeze` GUARD, PINNED IN BOTH DIRECTIONS

`apply_stage_freeze` (`stack/tanitad/models/v6.py`) sets `requires_grad` **from the group map**, and
an external backbone installed under `encoder` lands in a group S-W trains ⇒ **86,580,480 foreign
parameters would train while the run calls itself "frozen"** (E-XENC-1's measured trap).

**Shipped** in `stack/tanitad/models/v6.py`, exported from the module surface:

| name | what it does |
|---|---|
| `declare_frozen_external(module, why)` | marks a subtree (attribute `_tanitad_frozen_external`) and freezes it now — greppable, self-explaining |
| `frozen_external_prefixes(stack)` | `{prefix: reason}` for every declaration |
| `reassert_frozen_external(stack)` | the runbook step: re-freeze **AFTER** `apply_stage_freeze` |
| `assert_frozen_external(stack, stage, expect_n_trainable=None)` | ⛔ the guard |
| `FrozenExternalViolation` | its exception |

⛔ **BOTH DIRECTIONS ARE PINNED, because C95/C97 shipped a rejects-everything guard and a
passes-everything guard within one day:**

* **A — it CATCHES the un-freeze.** Any trainable parameter inside a declared subtree raises.
* **B — it CANNOT be satisfied by freezing everything.** Every group in
  `stage_trainable_groups(stage)` must still hold a trainable **native** parameter; a declaration
  that swallows a whole group raises **with the group named**.
* `expect_n_trainable` makes the runbook's exact count a **parameter of the guard**, not a number
  retyped into a launch script.

**MEASURED** · `stack/tests/test_v6_frozen_external.py` — **9 tests, 9 passed**. The
direction-A test **asserts the premise first** (that `apply_stage_freeze` really does un-freeze the
foreign subtree) and fails loudly if that ever stops being true, so the guard cannot quietly become
vacuous. It also pins that the guard is **silent on the incumbent stack** (no declaration, all four
stages) and on **S-T**, which legitimately trains no encoder.

⏳ **ESCALATION — this needs wiring, and a doc will not do it.** The guard exists and is tested; it
is **not yet called** by `train_v6_staged.py`. An E-XENC-1 launch must call
`reassert_frozen_external` then `assert_frozen_external(stack, stage, expect_n_trainable=193_479_171)`
**before step 1**. That is an orchestrator/PI integration item, deliberately not written into a
README (an orthogonality instrument sat unmerged for 10 days that way).

---

## 9. WHAT I RECOMMEND THE RETRACTION LOG SAY

1. **C106's `ego_v0` half stands and is now stronger** — it has a paired episode-cluster interval,
   positive in 27/27 cells, separated in C106's own cell at `p(Δ>0)=1.000`.
2. **C106's `lead_gap` half is withdrawn** — 0/27 cells separated, sign flips in 9/27.
   *"3.6× on both rungs"* becomes *"on `ego_v0`"*.
3. **The ratio "3.6×" is withdrawn as a number** and replaced by the signal/no-signal statement:
   the random init separates from the matched-random null on `ego_v0`; **the trained encoder does
   not separate from it on either rung in C106's own configuration.**
4. **C106's limit 2 ("our readout is a flat line") is corrected** — it is one inner-split draw.
5. **The "artifact-retrieval item" is re-classified UNAVAILABLE** — no checkpoint before ≈9 100
   exists; the early-window measurement needs a new run with a dense early snapshot cadence.
6. **New, and the actionable part: the trained token field is rank-1** (97.6 % of variance in one
   direction, design-matrix effective rank 6.7 vs 16.4). Whitening does not close the arm gap, so
   this is a **co-symptom, not the explanation** — but it is monitorable from step 0 and *"the
   objective subtracts geometry"* is not.

⭐ **ROOT-CAUSE CLASS I would log: A HEADLINE RATIO COMPUTED BETWEEN TWO ARMS AT DIFFERENT OPERATING
POINTS, WITH THE SEED THAT MOVES IT HELD FIXED AND THE ESTIMATOR NEVER APPLIED.** C106 varied the
projection seed (which moves the number by ±0.005) and reported its range as though it were an
interval, while holding the **ridge inner-split seed** (which moves it by ±0.05 — ten times more)
at one value, and never computed the paired bootstrap the programme's own instrument already
contains. ⇒ **RULE: before quoting a ratio, (a) run the estimator on the DIFFERENCE, (b) vary every
seed the pipeline exposes and report which one dominates, and (c) print `pred_sd/gt_sd` beside any
`corr²`, because a correlation is scale-free and will happily describe a constant predictor.**

---

## 10. SUITES — actuals, not inherited

**MEASURED (ours)** · `raw/suite_stack.txt`, `raw/suite_taniteval.txt` (full output banked, read —
⛔ an exit code is not evidence):

| suite | result | brief's expectation |
|---|---|---|
| `stack` | **3 893 passed · 0 failed · 7 skipped · 2 xfailed** (464.9 s) | 3868 / 0 / 7 / 2 |
| `taniteval` | **1 136 passed · 0 failed** (112.5 s) | 1136 / 0 |

⚠️ `stack` is **+25** on the brief's count, of which **9 are mine**
(`tests/test_v6_frozen_external.py`); the remaining **16** arrived from other live streams during
this turn. **Failures 0, skips 7, xfails 2 — all unchanged.**

---

## 11. DELIVERABLE MANIFEST

⛔ **STAGED, NEVER COMMITTED, NEVER PUSHED.** Nothing here lives in only one place: every artifact
is in the repo working tree and in the index. Staging verified per CLAUDE.md — `git ls-files
--cached` for NEW files, **index-blob vs `git hash-object`** for the MODIFIED tracked file.

| artifact | where | note |
|---|---|---|
| `C106_ADVERSARIAL.md` (this file) | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-18-c106-adversarial/` | NEW |
| `code/c106_adv.py` | same, `code/` | the adversarial re-read: wide α, multi-ridge-seed, cross-cache paired Δr²c, null, oracle. Imports er10's solver |
| `code/c106_layerscale.py` | same | LayerScale + residual-fraction on real banked frames |
| `code/c106_rank.py` | same | effective rank of token / pooled / design matrices |
| `code/c106_whiten.py` | same | the SUBTRACTED-vs-SWAMPED discriminator |
| `code/chain_adv.sh`, `code/chain_adv2.sh` | same | the two chains, reproducible end to end |
| `raw/*.json` (60 files) | same, `raw/` | gate, 3 arms × 2 grids, 2 nulls, 6 positive controls, 8 paired deltas, layerscale, rank, whiten |
| `raw/log_*.txt`, `raw/suite_*.txt` | same | full stdout of every run and both suites |
| **`stack/tanitad/models/v6.py`** | `repo:stack/` | **MODIFIED** — the frozen-external guard + 6 new exports. Index blob `366bc74b…` == worktree hash ✅ |
| **`stack/tests/test_v6_frozen_external.py`** | `repo:stack/tests/` | NEW — 9 tests, both directions pinned |

**Nothing was left on Thor.** Thor was touched only by **read-only `ssh -n` probes** (directory
listings, a `find`, one 73 KB zip-header read); **no file was written, no GPU used, the training run
was not touched.** No pod was used. All compute was the dev-box RTX 4060
(`torch.cuda.max_memory_allocated()` peaks recorded in `raw/rank.json`, `raw/layerscale.json`).

### ⏳ ESCALATIONS — these need a decision or a wiring step, and are deliberately NOT buried in a README

1. **`RETRACTION_LOG.md` needs the §9 amendment to C106.** I did not edit it — the log is
   append-only and the entry is the orchestrator's/PI's to write. The six numbered points in §9 are
   the proposed text.
2. **The frozen-external guard is NOT yet called by `train_v6_staged.py`.** It ships tested and
   unused. An E-XENC-1 launch must call `reassert_frozen_external` then
   `assert_frozen_external(stack, stage, expect_n_trainable=193_479_171)` before step 1.
3. **`PREREG_ENCODER_EXPERIMENTS.md` §7.6 should be corrected**: the step-0→9 250 sweep is
   **UNAVAILABLE**, not pending retrieval. And any future S-W arm should carry a **dense early
   snapshot cadence** as a launch condition.
4. **Token-channel effective rank should be logged per step** beside the existing z_op `spectrum`
   block in `train_v6_staged.py` — the z_op monitor demonstrably does not cover it.
