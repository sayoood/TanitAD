# E-SEL-1D — THE DEPLOYABLE CONSEQUENCE SCORE

**Date:** 2026-08-03 (Europe/Berlin) · **Stream:** arch-inf · **GPU cost:** **85 s** of an idle
Jetson Thor, two arms. No training launched, no training pod touched, nothing pushed.

**Pre-registration:** `…/2026-08-03-s3-deployable/PREREG_S3_DEPLOYABLE.md`, staged **before** any
statistic here was computed. Blob **`931ee1b1b338859bf4b7804d4db92f250cbac37d`**; `git ls-files -s`
and `git hash-object` **MATCH**, so §4's thresholds are the ones that were staged, and the runner
re-verifies this on every arm (`prereg_s3.thresholds_unmoved_since_staging: true`). The parent
prereg's own check also passes.

**n = the canonical 881 windows / 40 episodes**, both arms. Estimator: **episode-cluster bootstrap**
/ **paired** form, unit = episode, `n_boot = 2000`. ⛔ `overlapping_holdout_se` never called.

---

## 0. THE HEADLINE — the deployable ρ next to the upper bound

| | **`refc-base-30k`** | **`refc-xl-30k`** |
|---|---|---|
| **ρ_oracle** — E-SEL-1's number, uses the **FUTURE FRAME** `z_{t+5}` | **0.6657** [0.6183, 0.7157] | **0.6212** [0.5650, 0.6791] |
| ⭐ **ρ_deploy** — what `consequence_scores` can produce **at inference** | **+0.2493** [+0.0988, +0.3953] | **−0.2728** [−0.4592, −0.0807] |
| paired `deploy − oracle`, same windows | **−0.4163** [−0.6009, −0.2377] ✅ separated | **−0.8941** [−1.0753, −0.7049] ✅ separated |

> **Information present ≫ information reachable.** The upper bound overstates the deployable score
> by **0.42** (base) and by **0.89** (XL) — and on XL the deployable score does not merely shrink,
> **it changes sign.** Two arms of the same family disagree on whether REF-C's consequence points
> toward good candidates or away from them.

**And the realized effect, which is the number that should size S3:**

| | base | XL |
|---|---|---|
| shipped ADE@2s | 0.4728 | 0.4714 |
| **S3 grafted, honest LOEO gate** | **0.4715** | **0.4691** |
| paired ΔADE@2s | **−0.0013** [−0.0043, +0.0013] ⛔ **not separated** | **−0.0024** [−0.0067, +0.0016] ⛔ **not separated** |
| fraction of the **selection gap** closed | **0.46 %** | **0.78 %** |
| registered bar (`free_win_m`) | **0.02 m** — missed by **15×** | missed by **8×** |

⇒ **`S3 LOWER-BOUND ONLY`, both arms.** ⛔ **ρ ≈ 0.65 must not be quoted as S3's effect size.**
The deployable ρ is 0.2493 / −0.2728, and what it buys on the banked fans is **1.3 mm / 2.4 mm of
ADE, not separated from zero**.

---

## 1. ⛔ THE REGISTERED BRANCH FIRED — AND THE SUBSTANCE IS THE OPPOSITE. Read both.

| arm | **literal registered branch** | direction of the two load-bearing controls |
|---|---|---|
| base | **D-FUND** | ⛔ **both separations are ADVERSE** — the controls BEAT the score |
| XL | **D-MARGINAL** | ⛔ same |

§4's D-FUND trigger is *"ρ_deploy separated from C-ctxswap **and** separated from C-cv **and**
|ρ_deploy| ≥ 0.10"*. On base all three hold **literally** — and both separations point the wrong
way. **`separated_..._FAVOURABLY` is `false` in all four cells, both arms.**

⛔ **I am not rewriting the trigger.** This is the same gap E-SEL-0 hit and escalated: a
pre-registration whose branch table cannot express **"separated in the ADVERSE direction"**. My own
prereg inherited that defect one day after the escalation was filed. It is reported here (§7.1) and
the honest reading is stated in plain words: **the mechanism is refuted on both arms.**

---

## 2. THE CONTROLS — the two that could fire, both fired

### 2.1 Controls that CANNOT fail, labelled as such

**C-shuffled**: ρ_shuffled = **−0.0012** (base) / **+0.0002** (XL). E-SEL found permute-then-argmax
is uniform for **any** score; for a Spearman it is vacuous a second way — **`E[ρ] = 0` analytically
for any score**. It establishes only `ρ ≠ 0`. **Reported, never load-bearing.**

### 2.2 ⭐ C-ctxswap — `pooled` DERANGED across windows. **FIRED.**

Sattolo derangement (no fixed points, verified in-artifact; **97.84 %** of swaps land in a
*different episode*), `fan` unchanged.

| | base | XL |
|---|---|---|
| ρ_ctxswap | **+0.3906** [+0.2850, +0.4930] | **−0.2437** [−0.4241, −0.0519] |
| paired `deploy − ctxswap` | **−0.1413** [−0.2512, −0.0420] ✅ **separated, ADVERSE** | −0.0292 [−0.0624, +0.0037] ⛔ **not separated** |

⭐ **On base, a RANDOM OTHER WINDOW'S SCENE ranks candidates BETTER than the true one** — separated,
`p(Δ>0) = 0.0025`. On XL the true scene makes **no difference at all**. Either way the score is
**not reading "the consequence of flying this candidate IN THIS SCENE"**; on base the true context
is actively harmful. That is the claim S3 exists to make, and it does not survive its own control.

### 2.3 ⭐ C-cv — a ZERO-parameter score. **FIRED, exactly as pre-registered.**

`s_cv = −mean‖fan_i − constant_velocity‖`. `constant_velocity` derives purely from the pose at
`last` and `last−1` (`driving_diagnostic.py:113-124`) — **no future frame**, so it is as deployable
as S3 and costs **nothing**.

| | base | XL |
|---|---|---|
| ρ_cv | **+0.9951** [+0.9930, +0.9970] | **+0.9944** [+0.9921, +0.9966] |
| paired `deploy − cv` | **−0.7458** [−0.8962, −0.6002] ✅ **separated, ADVERSE** | **−1.2673** [−1.4527, −1.0750] ✅ **separated, ADVERSE** |

My registered prediction was *"C-cv is the control that bites"*. **It bit, on both arms, by a
margin larger than the entire deployable signal.**

### 2.4 The reproduction controls — all PASS

| control | base | XL |
|---|---|---|
| **C-raster** | PASS — fed **(256, 256)**, arm declares **(8, 8) = 64 tokens** | PASS |
| **C-identity** | `argmax(logits) == sel_idx` **1.0000**; ADE **0.472772** vs published **0.4728** | **0.471439** vs **0.4714** |
| **C-oracle-floor** | **0.191421** vs published **0.1914** | **0.163949** vs **0.1640** |
| ⭐ **C-reproduce-esel** | `fan`, `logits`, `refined_logits`, `gt`, `cv` **BIT-IDENTICAL** to E-SEL's bank; `sel` agreement **1.0000** | same |
| ⭐ **C-oracle-reproduce** | ρ_oracle **0.6657** — inside E-SEL's CI | **0.6212** — inside |
| **C-degenerate** | per-window std of `s_deploy`: median **0.0292**, min **0.00846**; `assert_candidate_axis` **PASS** | median **0.0172**, min **0.00460**, PASS |

⚠️ **One deviation, stated:** `cons_score` is **not** bit-identical to E-SEL's bank —
`max_abs_diff` **1.79e-07** (base) / **1.34e-07** (XL). Cause: this pass reduces `law_head` outputs
in chunks of 32 windows where E-SEL used 4, so the float32 `.mean(-1)` accumulates in a different
order. **The `fan` — the object both ρ's are computed on — is bit-identical**, which is what the
paired contrast requires, so the contrast is **not** void. Reported because a silent 1e-7 is how a
"reproduction" becomes an assumption.

---

## 3. ⭐⭐ THE FINDING THAT OUTLIVES S3 — ρ OVER THE CANDIDATE AXIS IS NOT A PROXY FOR SELECTION

*(Post-hoc diagnostic. It moves **no** threshold and decides **no** registered branch. It exists
because P2 asks to convert ρ into an ADE effect, and the honest conversion is to **measure the
selector each score produces**, not to model it.)*

### 3.1 Use each score as the selector — including the one that sees the future

| score | ρ (full axis) | **argmax ADE@2s** | `rank_acc` | `frac_2x_worse` |
|---|---|---|---|---|
| **shipped** (t=0 classifier) | — | **0.4728** / **0.4714** | 0.3292 / 0.3110 | 0.4109 / 0.4540 |
| **`cv`** (zero parameters) | **0.995** | 0.8149 / 0.8158 | 0.2860 / 0.2633 | 0.5096 / 0.5868 |
| ⭐ **`oracle`** (**sees `z_{t+5}`**) | **0.666 / 0.621** | **6.4889 / 6.4501** | 0.0227 / 0.0159 | 0.9648 / 0.9705 |
| **`ctxswap`** | 0.391 / −0.244 | 19.03 / 35.91 | 0.0034 / 0.0000 | 0.9955 / 1.0000 |
| **`deploy`** | 0.249 / −0.273 | **20.23 / 35.86** | 0.0023 / 0.0000 | 0.9955 / 1.0000 |
| *(C-shuffled, from E-SEL)* | 0.000 | *14.54 / 13.96* | *0.0078 / 0.0039* | *0.9798 / 0.9894* |

⭐ **The score with ρ = 0.6657 — the one that is allowed to look at the future — selects at
6.49 m, 13.7× WORSE than the shipped ranker.** And the deployable score selects at **20.2 / 35.9 m,
i.e. WORSE THAN RANDOM** (14.5 / 14.0 m).

### 3.2 Why — and it is measurable, not a story

**72–74 % of REF-C's fan is outside the bounded-acceleration band around `v0`, and deleting it is
MEASURED exactly inert on ADE** (D3; re-confirmed here: `frac_candidates_removed` **0.7376** /
**0.7208**, `oracle_survives_frac` **1.0000**, `clipped_ade` == `as_trained_ade` **bit-equal**).
So a rank correlation over the **whole** candidate axis is dominated by candidates **no selector
ever picks**. Restrict ρ to the S2-reachable survivors — the part of the fan that decides
anything — and it collapses:

| score | base: full axis → **reachable only** | XL: full axis → **reachable only** |
|---|---|---|
| **deploy** | +0.2493 → **−0.0286** [−0.0863, +0.0277] ⛔ **CI CROSSES ZERO** | −0.2728 → **−0.1043** [−0.1446, −0.0638] (still negative) |
| **oracle** | +0.6657 → **+0.3008** [+0.2622, +0.3421] | +0.6212 → **+0.3079** [+0.2695, +0.3446] |
| ctxswap | +0.3906 → +0.0375 [+0.0009, +0.0770] | −0.2437 → −0.0542 |
| cv | +0.9951 → +0.8820 | +0.9944 → +0.8928 |

⇒ **On the part of the fan that can be selected, the deployable consequence score carries NO
information on base and NEGATIVE information on XL.** And **E-SEL-1's 0.6657 loses more than half
of itself** the moment the unpickable tail is removed.

### 3.3 What this licenses, and what it does not

✅ **Licensed:** *`ρ(score, −ADE)` over the full candidate axis must not be used to size a
SELECTOR.* Two scores here differ by 0.75 in ρ and are indistinguishable at the argmax; one score
with ρ = 0.995 selects worse than the shipped ranker.
✅ **Licensed:** *E-SEL-1's registered statistic was answering a different question than the one
§6.3's branch spends a GPU-day on.* Its own verdict text — *"is there ANY candidate-discriminating
signal in REF-C's world model"* — is exactly right and is **still true**; what does not follow is
"…therefore reranking with it helps".
⛔ **NOT licensed: "E-SEL-1 was wrong."** It was not. Its number reproduces here **inside its CI**,
its caveat §4.1 is what sent me, and it explicitly refused to quote 0.65 as an effect size.
⛔ **NOT licensed: "the world model is useless."** ρ_oracle stays at **+0.30** on the reachable
subset — real, separated information. It is **not reachable without the future frame** (that is
§0), and it does **not** convert into a better argmax at these weights (§3.1).

---

## 4. THE FOUR METRIC FAMILIES — per family, never pooled (binding)

Grid **derived**, not assumed: `wp_steps [5,10,15,20] × 0.1 s → dt = 0.5 s`
(`four_families.infer_dt`). A hard-coded 0.1 s inflates every speed ×5 (R-2026-08-03-c).

### 4.1 The realized S3 effect (LOEO-gated rerank **minus** shipped), paired, same windows

| family | metric | **base** Δ | sep | **XL** Δ | sep |
|---|---|---|---|---|---|
| **LONGITUDINAL** | `speed_abs_err_mps` | −0.0006 [−0.0036, +0.0020] | no | −0.0019 [−0.0068, +0.0026] | no |
| | `speed_signed_err_mps` | −0.0027 [−0.0094, +0.0038] | no | **+0.0118** [+0.0065, +0.0179] | ✅ **WORSE** |
| | `along_abs_err_m` | −0.0016 [−0.0047, +0.0010] | no | −0.0021 [−0.0064, +0.0019] | no |
| | `along_signed_err_m` | −0.0032 [−0.0099, +0.0032] | no | **+0.0112** [+0.0059, +0.0168] | ✅ **WORSE** |
| **LATERAL** | `cross_abs_err_m` | +0.0009 [−0.0004, +0.0024] | no | −0.0008 [−0.0033, +0.0011] | no |
| | `heading_abs_err_deg` | +0.0086 [−0.0050, +0.0238] | no | −0.0183 [−0.0467, +0.0005] | no |
| | `curvature_abs_err_1pm` | +0.0001 [−0.0000, +0.0004] | no | −0.0001 [−0.0002, +0.0000] | no |
| | `yaw_rate_abs_err_degps` | +0.0211 [−0.0054, +0.0603] | no | **−0.0239** [−0.0550, −0.0017] | ✅ better |

⭐ **The axis test — and why the ADE row alone would have MISLED.** On base **nothing separates in
any family**: the −0.0013 m of ADE is null everywhere it is decomposed. On **XL** the −0.0024 m of
ADE comes **with a SEPARATED LONGITUDINAL REGRESSION**: `speed_signed` **+0.0118 m/s** and
`along_signed` **+0.0112 m** both separated **worse**, on top of a shipped bias that is *already*
over-predicting (`speed_bias` **+0.0209 m/s**). The one separated improvement is
`yaw_rate` **−0.0239 °/s** — **LATERAL**.

⇒ **XL's S3 graft trades a separated longitudinal-bias regression for a lateral yaw-rate gain, and
reports it as a 2.4 mm ADE improvement.** MEASURED and load-bearing: **87.60 %** of XL's selection
gap is LONGITUDINAL, and a perfect reranker buys only **0.0425 m** of cross-track. **A lever that
moves the lateral axis and degrades the longitudinal one cannot pay for itself here.** This is the
trade-off a pooled ADE row hides — the exact reason the four-family rule is binding.

### 4.2 The CEILING for reference (oracle-in-fan **minus** shipped) — replicates E-SEL §5.1

| family | metric | base Δ | sep | XL Δ | sep |
|---|---|---|---|---|---|
| **LONGITUDINAL** | `speed_abs_err_mps` | **−0.2688** [−0.3436, −0.1987] | ✅ | **−0.2943** [−0.3657, −0.2250] | ✅ |
| | `along_abs_err_m` | **−0.2781** [−0.3515, −0.2090] | ✅ | **−0.3002** [−0.3682, −0.2351] | ✅ |
| **LATERAL** | `cross_abs_err_m` | **−0.0334** [−0.0555, −0.0151] | ✅ (small) | **−0.0425** [−0.0645, −0.0236] | ✅ (small) |
| | `heading_abs_err_deg` | −0.1334 [−0.3127, +0.0307] | no | **−0.2444** [−0.4006, −0.1087] | ✅ |
| | `curvature_abs_err_1pm` | +0.0013 | no (**wrong sign**) | +0.0004 | no (**wrong sign**) |
| | `yaw_rate_abs_err_degps` | +0.1212 | no (**wrong sign**) | −0.1695 | no |

Base reproduces E-SEL's §5.1 row **to 4 dp**, from an independently decoded bank. XL's row is
**new** (E-SEL published the base decomposition only).

### 4.3 TACTICAL and STRATEGIC — per family, with reason and n

| family | status | n |
|---|---|---|
| **TACTICAL** — goal/anchor selection | ✅ **MEASURED** — this is the half D-SEL exists to move. shipped `rank_acc` **0.3292** / **0.3110**, `sel_gap` **0.2814** / **0.3075**, `frac_sel_2x_worse` **0.4109** / **0.4540**; every reranker's values in §3.1 | **881** |
| **TACTICAL** — manoeuvre decision + confusion | ⛔ **UNAVAILABLE**: a fan bank stores no decoded manoeuvre logits. **WORK ITEM.** | **0** |
| **LONGITUDINAL** — distance-keeping (headway / time-gap / TTC) | ⛔ **UNAVAILABLE**: no lead-agent track in a fan bank. The reader exists (`…/2026-08-03-longitudinal-distance-keeping/build_lead_tracks.py` over `obstacle.offline`); joining it to the val windows is a **WORK ITEM**, not a pass. | **0** |
| **STRATEGIC** | ⛔ **UNAVAILABLE**: no route/goal label, and the decode used `nav_mode='follow_constant'` so the route input was never exercised (the C6 confound, inherited deliberately — see §5). **WORK ITEM**, the S5 arm needs it. | **0** |

---

## 5. ⚠️ THE C6 CONFOUND IS INHERITED HERE ON PURPOSE

`nav_mode = follow_constant` — one constant command for every window. It is kept because every
contrast here is **paired against the published 0.4728 / 0.4714**, which were collected that way,
and because C-reproduce-esel requires the identical decode. It is a real limitation of **these
rows**, restated rather than inherited silently. A route-exercised re-collection is a separate
question.

---

## 6. WAS I RIGHT? — the registered prediction, scored

| I predicted (staged in advance) | outcome |
|---|---|
| ρ_deploy = **0.10–0.35** | base **+0.2493** ✅ **inside**. XL **−0.2728** ❌ **outside — I did not predict a SIGN FLIP**, and the sign flip is the more important half of the result. |
| most likely **D-MARGINAL** | XL D-MARGINAL ✅; base D-FUND **literally** but adverse on both controls ⚠️ — my branch table could not express what happened, which is the same defect I was sent to fix in someone else's. |
| **C-cv is the control that bites** | ✅ **correct, both arms**, by a margin larger than the whole deployable signal. |
| mechanism = `conf_head` read **off its training distribution** | consistent with the direction and magnitude, but ⚠️ **HYPOTHESIS** — not separately tested here. |

**And the thing I did not predict at all:** that ρ over the full candidate axis would turn out to be
disconnected from selection quality **for every score including the future-seeing one** (§3). That
is the finding with the longest reach and it came out of a control, not the headline.

---

## 7. 🔴 ESCALATIONS

1. **→ PI / orchestrator — a SECOND pre-registration has now hit "separated in the ADVERSE
   direction".** E-SEL escalated it for `PREREG_D-SEL…` §6.3 yesterday; `PREREG_S3_DEPLOYABLE.md`
   §4, written **today**, reproduced the defect. **The template needs a direction predicate**, not a
   fifth row: every "separated from a control" trigger should read *"separated **and** the delta
   favours the treatment"*. I did not patch my own staged file — that would void "fixed in advance".
2. **→ arch-inf / D-SEL — S3 should NOT consume a parameter on this evidence.** §6.3's *"S3 LIVE ⇒
   include S3 in the retrain arm"* was triggered by a statistic that (a) uses the future frame and
   (b) does not convert into selection at all (§3.1). The deployable realized effect is **1.3 mm /
   2.4 mm, not separated**, and on XL it is a **separated longitudinal regression**. ⚠️ This is a
   **LOWER BOUND** — `cons_gate` is zero-init and `feat_proj`/`conf_head` do receive gradient in S3
   — so it does not falsify a trained S3. It does say S3 must not be funded as a *free* win, and
   that if it is run it needs the LONGITUDINAL family as its read, not ADE.
3. **→ eval-tools — every place the programme sizes a SELECTOR from a rank correlation is
   suspect.** §3.2 gives the mechanism (72–74 % unpickable fan) and the fix (report ρ on the
   S2-reachable subset beside the full-axis ρ; both are now in the probe). This generalises beyond
   REF-C.
4. **→ ops — the 256×256 REF-C val cache is STILL a single-disk dependency** (`thor:~/valdata/
   physicalai-val-0c5f7dac3b11`), unchanged since E-SEL's escalation 3. ⭐ **Partly retired for this
   question**: `raw/fan_deploy_refc-{base,xl}-30k.pt` now bank `pooled` **and** `law_tgt` (z_{t+5})
   per window, so every consequence statistic — including ones nobody has thought of yet — is
   recomputable **off Thor, with no checkpoint and no cache**.

---

## 8. WHAT I DID NOT DO — plainly

* ⛔ **Did not train anything, launch any arm, or touch `tanitad-new`/`tanitad-pod4`.** Cost was
  85 s on an idle Thor (GPU 0 % before and after).
* ⛔ **Did not measure S3 AFTER training.** Everything here is at frozen 30 k weights with a
  zero-init gate. Stated as a LOWER BOUND throughout and never upgraded to an effect size.
* ⛔ **Did not run distance-keeping, manoeuvre-confusion or STRATEGIC** — UNAVAILABLE with reason
  and n = 0 (§4.3). Each is a work item, not a pass.
* ⛔ **Did not re-open E-SEL-0**, and did not run `refc-small-30k` (Thor holds no small checkpoint —
  unchanged from E-SEL).
* ⛔ **Did not edit `PREREG_S3_DEPLOYABLE.md` after seeing a result**, including where it would have
  been convenient (§7.1). Verified: staged blob == worktree blob on every arm's run.
* ⚠️ **Did not test the `conf_head`-off-distribution mechanism** (§6). It is a HYPOTHESIS. The
  discriminating experiment is cheap — compare `conf_head`'s response to `layer_norm(feat_proj(
  law_head(...)))` against its response to real post-attention queries — and was not run.
* ⚠️ **Wrote to Thor**: `~/refc_sel_dump_deploy.py` and `~/fan_deploy_refc-{base,xl}-30k.pt` — all
  **new paths**. Nothing E-SEL or the concurrent stream owns was overwritten; `thor:~/_dsel_backup/`
  untouched.

---

## 9. EVIDENCE CLASS

| claim | class |
|---|---|
| ρ_deploy **+0.2493** [+0.0988, +0.3953] (base) / **−0.2728** [−0.4592, −0.0807] (XL) | **MEASURED** — `raw/s3_deploy_probe_refc-{base,xl}-30k.json` |
| realized LOEO ΔADE **−0.0013** / **−0.0024** m, **not separated**; 0.46 % / 0.78 % of the selection gap | **MEASURED** — same files |
| C-ctxswap **+0.3906** > deploy on base, paired **−0.1413** separated adverse; XL not separated | **MEASURED** — same files |
| C-cv ρ **0.9951 / 0.9944**, paired **−0.7458 / −1.2673** separated adverse | **MEASURED** — same files |
| argmax of the **oracle** score = **6.49 / 6.45 m**; of `deploy` = **20.23 / 35.86 m** (worse than random) | **MEASURED** — `diagnostic_rho_vs_selection.argmax_selector` |
| ρ on the S2-reachable subset: deploy **−0.0286** (CI crosses 0, base); oracle **0.6657 → 0.3008** | **MEASURED** — `diagnostic_rho_vs_selection.rho_on_S2_reachable_only` |
| XL's realized graft is a **separated longitudinal regression** (`speed_signed` +0.0118, `along_signed` +0.0112) | **MEASURED** — `families.paired_loeo_rerank_minus_shipped` |
| the re-decode reproduces E-SEL's fan **bit-for-bit**; ρ_oracle inside E-SEL's CI | **MEASURED** — `controls.C-reproduce-esel`, `controls.C-oracle-reproduce` |
| `consequence_scores` never sees `z_{t+5}`; `cons_ctx = pooled` | **MEASURED (source)** — `refc_select.py:308-321`, `refc.py:1246-1251` + `1835-1836` |
| `constant_velocity` uses only the pose at `last`/`last−1` (no future) | **MEASURED (source)** — `driving_diagnostic.py:113-124` |
| 87.60 % / 89.28 % of the selection gap is LONGITUDINAL | **INHERITED** from `ESEL_VERDICT.md` §5.1 — **not** re-derived here; its *component* rows ARE reproduced (§4.2) |
| *"the oracle gap is ~92 % irreducible"*, v1.2's **8.4 %** | **INHERITED** — a prose note in `MODEL_REGISTRY.md` §4.1, **not a results JSON**. ⚠️ **NOT** the other 8.4 % in §1.4b (a relative change in the flagship's fan under unfreezing) |
| *"`conf_head` is off its training distribution"* is the mechanism | **HYPOTHESIS** — not tested (§8) |

**Full suite:** `cd stack && pytest -q` → see `raw/pytest_full_suite.txt`. ⚠️ The baseline is moving
under concurrent streams (the brief quoted 1932/12/2; E-SEL saw it climb 1900 → 1913 → 1932 inside
one task), so a count is only meaningful against a pinned commit. My 18 new tests are in
`stack/tests/test_s3_deploy_probe.py`.

---

## 10. DELIVERABLE MANIFEST

| artifact | path | state |
|---|---|---|
| **pre-registration** (staged BEFORE measuring, blob `931ee1b1…`) | `…/2026-08-03-s3-deployable/PREREG_S3_DEPLOYABLE.md` | repo, **staged** |
| this verdict | `…/2026-08-03-s3-deployable/S3_DEPLOYABLE.md` | repo, **staged** |
| **GPU-side dump — the deployable score, `pooled` and `z_{t+5}`** | `stack/scripts/refc_sel_dump_deploy.py` | repo, **staged** — also `thor:~/refc_sel_dump_deploy.py` |
| **analysis probe** (ρ panel, controls, LOEO sizing, four families, ρ-vs-selection diagnostic) | `stack/scripts/refc_s3_deploy_probe.py` | repo, **staged** |
| **tests (new, 18)** — derangement + LOEO no-leak guards | `stack/tests/test_s3_deploy_probe.py` | repo, **staged** |
| **results, 881 windows, 2 arms** | `…/raw/s3_deploy_probe_refc-{base,xl}-30k.json` | repo, **staged** |
| **augmented banks — carry `cons_deploy`, `cons_oracle`, `cons_ctxswap`, `pooled`, `law_tgt`** | `…/raw/fan_deploy_refc-{base,xl}-30k.pt` | repo, **staged** — also on Thor, md5-verified both ends |
| full-suite transcript | `…/raw/pytest_full_suite.txt` | repo, **staged** |

**Nothing is committed and nothing is pushed. Nothing that took effort lives only on Thor.**
