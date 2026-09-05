# STEP 2 — the magnitude head, and why it FAILS its own pre-registration

**Rows:** `D-REFAV1-KAPPA-HEAD`, `D-REFAV1-SOFT-KAPPA`, `D-REFAV1-NAV-MAGNITUDE`
**Class:** MEASURED, **zero GPU** · **Date:** 2026-09-05
**Instruments:** `tools/kappa_head.py`, `tools/soft_kappa.py`, `tools/extract_intent.py`
**Raw:** `raw/kappa_head_fit.json`, `raw/soft_kappa_s40.json`, `raw/intent_stride2_prov.json`
**Bank:** `C:/Users/Admin/refav1_margin/intent_stride2.npz` — 4786 windows / 141 episodes,
`intent` **[4786, 256]** + `intent_navshuf`, `gt_kappa`, `lat_logits`. ⚠️ **dev box only.**

---

## Why Step 2 changed target, and what forced it

`SPEC.md` committed a **class-balanced re-fit of `lat_head`**. `ROOT_CAUSE.md` then MEASURED
that on **90.15 %** of GT-turn windows `LANE_KEEP` is the **vocabulary-optimal** token — the
lateral goal can command exactly two *sustained* curvatures, `0` and `0.08` (R 12.5 m), and
the corpus curves at R 100–1000 m. **A classifier over that vocabulary cannot express a 200 m
curve however well it is fitted.** ⇒ the objective was re-specified to a **curvature
MAGNITUDE** on the same banked `intent`.

⚠️ **This is an AMENDMENT to a pre-registration and it is recorded as one**, with its trigger
measured and named, rather than applied silently. The success criteria below were fixed
before the fit ran.

## The bank, and its controls

`tools/extract_intent.py` re-runs exactly `plan()`'s call — `model._run_brains(pooled, nav)`
at `refa_v1.py:2156`, **`ego=None`** — and banks `intent` for 4786 windows. **All four
controls passed:**

| control | result |
|---|---|
| C1 reproduces the banked `lat_logits` | max abs diff **0.0**, argmax **282/282** |
| C2 reproduces the banked `gt_kappa` | max abs diff **2.4e-7**, turn-label **282/282** |
| C3 `intent` is not a field of zeros | non-zero variance on **100 %** of columns, row norms non-constant |
| C4 nav shuffle changed rows | **2455 / 4786** rows changed, `intent` differs |

⭐ C1 is what makes the bank trustworthy: the features are from the **same forward pass** that
produced every banked refav1 number, not a re-implementation.

⚠️ **`intent` is 256-d, not 1024.** `d_int = intent_dim or cfg.d_state`, and this checkpoint
passes `intent_dim = 256` — read off `lat_head.0.weight`'s shape, not assumed from
`d_state = 1024`.

---

## Arm 1 — the free fix: expected curvature under the head's own posterior. **FAILS.**

`kappa_hat = K·(p[TURN_L] − p[TURN_R])`, `p = softmax(lat_logits/T)` — a continuous signed
curvature in `[−K, +K]`, **zero new parameters, zero training**, reducing to the shipped hard
decode as `T → 0`.

| arm | RMSE | R² | Pearson r | turn non-zero + correct sign | straight non-zero |
|---|---|---|---|---|---|
| ZERO (floor) | 0.01705 | −0.004 | 0.000 | 0.0000 | 0.0000 |
| HARD argmax (shipped) | 0.02974 | −2.057 | −0.132 | 0.2895 | 0.1025 |
| SOFT `T=1.0` | 0.02138 | −0.579 | −0.083 | 0.5789 | **1.0000** |
| SOFT `T=5.0` (best RMSE) | 0.01673 | 0.034 | −0.100 | 0.5789 | **1.0000** |

⛔ **Controls C1 and C2 passed; C3 FAILED, and C3 is the one that decides.**
The shuffled-logit control reads `r = +0.0068 ± 0.0691`; the real `r = −0.0999` — **on the
wrong side of the control**, not four SDs above it. And the RMSE "win" is pure
**shrink-to-zero**: `mean|κ|` falls 0.0108 → 0.0031 as `T` rises and the ZERO floor (0.01705)
is barely beaten (0.01673, R² 0.034). Meanwhile every `T ≥ 0.5` emits non-zero curvature on
**100 % of straight windows** — the "turns everywhere" failure `SPEC.md` rules out in advance.

⇒ **The soft posterior carries no usable magnitude.** *This is CLAUDE.md probe-trap #3 in a new
costume — maximal shrinkage beating a noisy estimator — and only the same-breath shuffled
control separated them.* ⚠️ My own verdict function returned `SUCCEEDS` here; the **controls**
returned FAILS and the controls win. Recorded because a verdict rule that can be wrong is
worth knowing about.

---

## Arm 2 — a trained ridge magnitude head on `intent`. **FAILS, and the reason matters.**

Ridge `Linear(256 → 1)` on the head's own frozen `LayerNorm(intent)`, target `gt_kappa`
clipped to `±0.2`. **Episode-disjoint** 3-way split (85 train / 28 tune / 28 test episodes);
`λ` chosen on TUNE only; TEST scored once.

| arm | RMSE | MAE | R² | r | turn non-zero + correct sign | straight non-zero |
|---|---|---|---|---|---|---|
| **RIDGE on `intent` (LEARNED)** | **0.01860** | 0.00705 | **0.1165** | **0.346** | 0.4936 | 0.6506 |
| CONSTANT-ONLY (control) | 0.01980 | 0.00772 | −0.0013 | 0.000 | **0.4744** | 1.0000 |
| ZERO (floor) | 0.01982 | 0.00671 | −0.0033 | 0.000 | 0.0000 | 0.0000 |
| SHIPPED head hard κ (floor) | 0.02336 | 0.00847 | −0.3930 | 0.325 | 0.2692 | 0.0215 |
| SHUFFLED-TARGET (control) | 0.01995 | 0.00800 | −0.0158 | −0.012 | 0.4038 | 0.7278 |
| **RIDGE on NAV-SHUFFLED `intent`** | 0.01973 | 0.00742 | **0.0064** | 0.100 | 0.4231 | 0.6177 |

**n_train 2891 windows / 85 EPISODES · d = 256 · n_test 946 windows / 28 episodes.**

| pre-registered control | verdict |
|---|---|
| constant reads no information (R² ≤ 0) | ✅ −0.0013 |
| shuffled target collapses | ✅ −0.0158 |
| learned beats the ZERO floor | ✅ 0.01860 < 0.01982 |
| learned beats the SHIPPED head | ✅ 0.01860 < 0.02336 |
| **beats CONSTANT on turn-sign by > 0.10** | ⛔ **0.4936 vs 0.4744 — +0.019** |
| **not a nav echo (> 50 % of R² survives nav shuffle)** | ⛔ **0.0064 / 0.1165 = 5.5 %** |
| λ not at a grid edge | ⛔ λ\* = 1e-4, the bottom of a 1e-4…1e8 grid |

⇒ **VERDICT: FAILS.**

### The two failures, read honestly

1. ⚠️ **It barely beats a CONSTANT on the decision that matters.** A constant non-zero
   prediction already scores **0.4744** on "non-zero and correctly signed on a real turn",
   because a fixed sign is right about half the time. The learned head reaches **0.4936**.
   The R² of 0.117 is real but it lives almost entirely in *magnitude on windows that are
   nearly straight*, which is not the quantity the planner needs.
2. ⛔⛔ **94.5 % of the learned R² disappears when nav is permuted** (0.1165 → 0.0064).
   nav on PhysicalAI is ultimately supplied from the ego's own future path, so **a magnitude
   head fitted this way is reading the ROUTE, not the ROAD.** That is precisely the class the
   PI's binding ruling and CLAUDE.md's flagship route-echo warn about, and it disqualifies
   this head from deployment regardless of its RMSE.

⚠️ **Scope the nav claim exactly.** Permuting nav *randomises* a channel rather than removing
it, so this shows the fit **does not survive nav permutation** — a necessary-condition
failure. It does **not** license "the vision carries nothing": the cleaner test holds nav
CONSTANT, and that bank does not exist yet. ⭐ And it does **not** contradict the shipped
head's *classification*: nav-only AUC is **0.5967** against the head's **0.7120** for
detection and **0.9043** for direction, so detection and direction are **not** nav echoes.
**It is specifically linear MAGNITUDE regression that is nav-driven.**

3. ⚠️ **Power.** 85 training EPISODES against d = 256 is **0.33 episodes per dimension**.
   Windows inside an episode are near-duplicates, so the effective n is episodes, and this fit
   is thin by construction. λ\* pinned at the *bottom* of the grid says the fit wanted *less*
   shrinkage, i.e. it is not the classic n≪d collapse — but a null here still may not be read
   as absence.

---

## What survives, and what to do instead

⭐ **The shipped head is better than it looked, and its two useful outputs are NOT nav echoes:**

| capability | measure | control |
|---|---|---|
| detect a real turn (R ≤ 100 m) | **AUC 0.8039** | shuffled 0.500 |
| detect a sharp turn (R ≤ 33 m) | **AUC 0.8799** | shuffled 0.498 |
| **choose L vs R** (R ≤ 33 m) | **AUC 0.9043**, direction acc **0.8962** (n = 183) | shuffled **0.4984 ± 0.0213** |
| nav-only floor | AUC 0.5967 | — |

⇒ **What refav1 lacks is not detection and not direction — it is a MAGNITUDE it is allowed to
command.** Both attempts to supply one from the existing representation failed their
pre-registered controls. ⇒ **the next lever is the VOCABULARY, not another head**: a sustained
curvature between `0` and `0.08`, which is a change to `canonical_controls` and its constants
(`refa_v1.py:118-123`), flag-gated so a zero-flag arm stays bit-identical — the pattern
`lat_logit_bias` already established. ⛔ **Escalated, not implemented here:** it changes the
goal profile for every arm and so breaks bit-parity with every banked refav1 number, which
makes it a pre-registration item rather than an edit made at the end of a session.

## Deliverable status

`kappa_head.pt` was written by the first (lenient-verdict) run and is **NOT deployable** —
the stricter re-run FAILS. It is retained only as the artefact the FAILS verdict refers to.
⛔ **Do not wire it into an arm.**
