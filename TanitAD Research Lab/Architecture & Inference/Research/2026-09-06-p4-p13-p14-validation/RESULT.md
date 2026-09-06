# RESULT — P4 / P13 / P14, the three unowned refcv5 pieces

**Author:** P4/P13/P14 validation agent · **Date:** 2026-09-06 · **Status:** COMPLETE
**Pre-registration:** `PREREG.md` in this directory, written **before** any number existed.
**Governing plan:** `Project Steering/REFCV5_MISSING_PIECES_PLAN.md` §2.
**Rig:** banked fan (P14, **zero GPU**) + live mutation on the real modules (P13, P4).
⚠️ **A banked-fan / mutation PASS is entry to the composed arm, NOT a published result.**

---

## 0. The one-line verdict

| piece | bar, as written | verdict | enters the composed arm? |
|---|---|---|---|
| **P14** — sampler ranks the fan | B1 ceiling ≥ 0.05 m separated · B2 > 25 % · B3 vacuity · R valid | ⭐ **PASS on both banked rigs** | ⭐ **YES — and it is now armable, which it was not** |
| **P13** — DD's `t ~ U[0,50)` draw | B1 consumer proven · B2 schedule is DD's · R valid | ⭐ **PASS (B1, B2, R)** ⛔ **B3 benefit: NOT ANSWERED** | ⚠️ **NOT YET** — mechanism-validated, **UNVALIDATED-FOR-BENEFIT** |
| **P4** — 15 strategic tokens | A1 ≥ 6/8 and ≥ 5/7 classes at ≥ 1 % support | ⛔ **FAIL AS WRITTEN — 3/8 and 3/7** | ⚠️ **PARTIAL — 6 of 15 tokens, on the restricted vocabulary the FAIL clause requires** |

⇒ **P14 earned entry. P13 earned its mechanism but not yet its benefit. P4 failed its corpus bar
and enters only in the restricted form its own FAIL clause pre-committed to.**

---

## 1. P14 — THE SAMPLER RANKS A FAN IT HAS NOT SAMPLED

### 1.1 What was actually wrong — and it is smaller and worse than "not implemented"

⛔ **MEASURED: `refc_v3_train.py` — the trainer that launches refcv5 — contained
`sel_refined` and `sel_score_emitted` ZERO times.** Control: `sel_` appears **8** times in the same
file, so the zero is a fact about those names and not about a broken probe. Meanwhile:

* `AnchoredDiffusionDecoder` **already implements** emitted-fan scoring (`sel.score_emitted`);
* `RefCConfig.selection()` **already reads** `self.sel_refined` / `self.sel_score_emitted` /
  `self.sel_score_emitted_t` straight into `SelectionConfig`;
* `refc.py` **already stamps** `sampler_ranks_the_fan: bool(self.sel.refined)` into telemetry;
* `refc_train.py` — the **v1.2** trainer — **already carries both flags**.

⇒ ⭐ **P14 IS PLUMBING, NOT ARCHITECTURE.** Every mechanism existed and the launch path could not
reach any of it. refcv5 shipped `sampler_ranks_the_fan: False`: the fan is **SAMPLED**, then ranked
by the **classifier surface** — a score computed **before any sample was drawn**.

### 1.2 The bar, and PASS/FAIL as written

**Rig:** the banked fans, **zero GPU**. `n = 881` windows, `d = 128` (base) / `256` (XL) candidates,
**40 episodes**, horizon 2.0 s at frames [5,10,15,20]. Estimator: **paired episode-cluster
bootstrap** (`taniteval/ci.py`).

| bar | committed threshold | base-30k | xl-30k | verdict |
|---|---|---|---|---|
| **B1** ceiling | ADE gap ≥ 0.05 m **and** CI excludes zero | **+0.2813 m**, CI **[+0.2127, +0.3543]**, separated | **+0.3075 m**, CI **[+0.2397, +0.3778]**, separated | ⭐ **PASS** |
| **B2** ranking not fan quality | > 25 % of windows where fan-best beats shipped by > 2× | **41.09 %** | **45.40 %** | ⭐ **PASS** |
| **B3** vacuity | re-ranking's manoeuvre rate ≥ 50 % of shipped | oracle **22.25 %** vs shipped **21.91 %** (GT 25.09 %) | oracle **23.50 %** vs shipped **23.04 %** | ⭐ **PASS — not bought by refusing to steer** |

⭐ **INDEPENDENT REPRODUCTION.** My harness reproduces **both** numbers the programme already
quotes, from the raw fans, without having been given either: the plan's **41.09 %** (base) and
`refc.py`'s own comment **"45.4 %-of-windows ranking failure"** (XL, measured **45.40 %**).

⇒ **ADE 0.4728 → 0.1914 m is a 59.5 % reduction available from re-ranking the SAME fan.** No new
capacity, no new fan, no extra rollout.

### 1.3 The deliberate-regression arm — the gate CAN fail

⛔ **`rank_shuffled`** (the shipped scores permuted within each window):
**+13.7598 m ADE worse than shipped**, CI **[+12.6282, +15.0243]**, separated (XL: **+13.9292**).
⇒ ⭐ **VALID.** The instrument detects a knowingly-broken ranking, so the PASS above means something.

### 1.4 Controls, at their KNOWN values

| control | required value | MEASURED | |
|---|---|---|---|
| `rank_shipped` reproduces the dump's own `sel` | `n_mismatch = 0` | **0 / 881** | ⭐ PASS |
| `rank_oracle` = per-window fan minimum | exact, by construction | max abs diff **0.000e+00** | ⭐ PASS |
| `rank_shuffled` picks the fan-best anchor | the no-information value **1/K = 0.78 %** | **0.68 %** | ⭐ PASS |
| `cv_floor` (raw-input floor) | the learned ranking must beat it | CV **0.8377** vs shipped **0.4728 m** | ⭐ PASS |
| `n`, `d` | printed | `n = 881`, `d = 128 / 256` | ⭐ |

### 1.5 Four metric families — never pooled (base-30k)

| arm | ADE m *(accompanying)* | **LONGITUDINAL** along m / speed-err | **LATERAL** cross m / **curvature MAE** | **TACTICAL** manoeuvre rate |
|---|---|---|---|---|
| `rank_oracle` | 0.1914 | 0.1385 / 0.3118 | 0.0978 / **2.32114** | 22.25 % |
| `rank_shipped` | 0.4728 | 0.4166 / 0.7549 | 0.1313 / **2.30973** | 21.91 % |
| `rank_shuffled` | 14.2326 | 14.1623 / 14.6438 | 0.5402 / 0.70839 | 30.99 % |
| `rank_constant` | 11.6663 | 11.6173 / 8.4899 | 0.5281 / 2.90288 | 24.06 % |
| **`cv_floor`** (raw input) | 0.8377 | 0.4401 / 0.7874 | 0.5259 / **0.02737** | 1.36 % |
| **`straight_floor`** | 0.4662 | 0.0000 / 0.1687 | 0.4662 / **0.02737** | 0.00 % |

**STRATEGIC:** ⛔ **N/A, with its reason and its n** — the banked fan carries no route or goal label.
`n = 0`. Not silently dropped.

### 1.6 ⛔⛔ TWO FINDINGS THAT ARE NOT ABOUT P14 AND MATTER MORE THAN IT

**(a) THE RANKING GAIN IS ~99 % LONGITUDINAL AND CURVATURE IS NOT SEPARATED.**
Of the +0.2813 m ADE ceiling, **+0.2781 m (98.9 %) is along-track**; cross-track is +0.0334 m; and
**curvature MAE is NOT separated** (base **−0.0114**, CI **[−0.0327, +0.0005]**; XL **+0.1524**, CI
**[−0.1679, +0.4739]**). ⇒ **P14 is a LONGITUDINAL lever.** Reporting it as "better trajectory
selection" would over-claim; it buys speed/along-track, and the programme already knows 88.7 % of
its oracle gap is longitudinal.

**(b) ⛔ THE MODEL IS ~84× WORSE ON CURVATURE THAN A PLAN THAT NEVER STEERS.**
`straight_floor` curvature MAE **0.02737** against the model's **2.30973**. This is precisely the
failure the plan's §2.4 demands LATERAL be read for — *"an arm can win ADE while tracking the road
worse than a plan that never steers"* — and here it is, MEASURED, at two orders of magnitude.
⚠️ **Scope it honestly:** `straight_floor` borrows the GT's own along-track profile, so it is a
*diagnostic* floor, not a deployable arm; the **honest raw-input floor is `cv_floor`**, which the
model beats on ADE (0.4728 vs 0.8377) and **also loses to on curvature by the same 84×** (0.02737).
So the finding survives the correct floor. ⇒ **Registered as a new hypothesis; it is not P14's to
fix, and no P14 number should be quoted without it.**

### 1.7 What I changed, and the invariant that makes it safe

`stack/scripts/refc_v3_train.py` — added `--sel-refined`, `--sel-score-emitted`,
`--sel-score-emitted-t`; a **paired refusal** (either flag alone exits non-zero, naming why); an
**auto-correction of `sel_score_emitted_t` from −1 to 0 on a `ddim` arm** (the emitted fan IS the
denoised state; −1 would index the refinement loop's `nn.Embedding` instead), **stamped** as
`sel_score_emitted_t_source: auto-zero-on-ddim`; and a `selection.sampler_ranks_the_fan` key in
`config.json` so a reader can tell the two arms apart **from the config**, not from per-batch
telemetry nobody re-reads.

⭐ **MUTATION-PROVEN on the live decoder** (`raw/p14_wiring.json`):
OFF → `sampler_ranks_the_fan = False`; ON → `True`; the ranked score **moves**
(max abs Δ **0.008557**, so the flag is not inert); and ⛔ **`anchor_traj` is BIT-IDENTICAL between
OFF and ON** — the extra pass keeps its confidence and discards its offset, so **every banked
oracle-in-fan contrast this lever is paired against stays comparable.**

⚠️ **The pair is enforced because `--sel-refined` ALONE is the MEASURED-HARMFUL lever** (0.0259 m
separated WORSE, 29.82 % of picks flipped, both minority recalls lowered) — it ranks by a
1-pass-stale score. `--sel-score-emitted` alone is **inert** (`base = refined if sel.refined else
conf`). ⇒ **No `refc.py` edit was required.** An optional refinement — routing the emitted-fan pass
through `_decode_ctrl` (the sampler's continuous `time_mlp`) rather than `_decode`'s 3-row
`time_embed` — is described in §4 as an escalation, not a blocker.

### 1.8 The variance the interval answers

⛔ The paired episode-cluster bootstrap answers **"would another draw of EPISODES say this?"** and
nothing else. ⭐ Here that is the **whole** question: both arms are the **same checkpoint** and the
**same banked fan**, so training variance (`H-ESTIM-SEED-1`, replicate FP rate 6/42 = 14.3 %) and
inference variance (refav1 seed floor ≈ 0.30 m) are **held at exactly zero by construction**.
⚠️ **This does NOT transfer to the composed arm.** Once P14 is trained rather than re-ranked
offline, its effect is a difference between two trained, *sampling* arms and needs **both** a
training replicate and an **inference-seed** replicate.

---

## 2. P13 — DD's `t ~ U[0,50)` TRAINING DRAW

### 2.1 The defect, proven by live mutation (not by inspection)

⛔ **`--sampler-train-t-max` had no consumer.** Two independent mechanisms:
1. **scoped source probe** — the only non-binary occurrences are the dataclass default
   (`refc.py:450`), the trainer's assignment onto `core.decoder.sampler_train_t_max`, and its
   `argparse` line. Control: `sampler_space` reads **21** occurrences, so the near-empty result is
   about this name.
2. ⭐ **a LIVE SPY on the real `AnchoredDiffusionDecoder._sample`**, in `.train()` mode:

| arm | realised timesteps consumed by `_decode_ctrl` | mean | var | n |
|---|---|---|---|---|
| SHIPPED, `t_max = 50` | **{0, 10}** | 5.000 | 25.0000 | 640 |
| SHIPPED, `t_max = 1` | **{0, 10}** — **IDENTICAL** | 5.000 | 25.0000 | 640 |
| REPAIRED, `t_max = 50` | {0,1,2,…,49} | 24.163 | 204.9799 | 320 |
| REPAIRED, `t_max = 1` | {0} | 0.000 | 0.0000 | 320 |

⇒ **Changing the flag from 50 to 1 leaves the realised timesteps bit-identical.** The flag is
**dead**, and refcv5 stamped `sampler_train_t_max: 50` into a `config.json` whose launch record
celebrated passing the dead-flag check.

⚠️ **ONE CORRECTION TO THE BRIEF, MEASURED.** The brief says refcv5 "trained at `t = 8`". It did
not: it trains at **t ∈ {10, 0}** — the *inference ladder* `infer_timesteps(8, 2) = [10, 0]`. 8 is
the truncation *point*, not a consumed timestep. Same defect, different value; stated because a
number quoted one step off its artifact is how the next reader's arithmetic goes wrong.

### 2.2 Bars, as written

| bar | committed | MEASURED | verdict |
|---|---|---|---|
| **B1** the flag acquires a consumer | shipped proven inert **and** repair proven responsive | both, above | ⭐ **PASS** |
| **B2** the noising is DD's | `sigma(8) = 0.0316`; schedule = DDIM `scaled_linear` | **0.031588**; pinned **bit-equal** against `diffusers` 0.40.0 | ⭐ **PASS** |
| **B3** BENEFIT | tiny-rig arm, u0 loss improves, curvature MAE does not degrade, **with a replicate** | ⛔ **NOT RUN** | ⛔ **UNVALIDATED-FOR-BENEFIT** |

### 2.3 Deliberate-regression arm + controls

⛔ **`t_max = 1`** (the constant `t = 0`, no-noise draw): realised variance **exactly 0.0000**,
distinguishable from `t_max = 50`. ⇒ ⭐ **VALID.**

| control | required | MEASURED |
|---|---|---|
| `sigma(0)` | **> 0** (`steps_offset = 1` ⇒ `abar_0 < 1`; t = 0 is *not* zero noise) | **1.000000e-02** ⭐ |
| `sigma(8)` | published **0.0316** | **0.031588** ⭐ |
| mean drawn `t`, `t_max = 50` | **≈ 24.5** = (50−1)/2 | **24.431**, min 0, max **49** (bound EXCLUSIVE), n = 20 000 ⭐ |
| `t_max = 1` variance | **exactly 0** | **0.000000** ⭐ |
| shipped realised-`t` variance | invariant to the flag | **25.0000 both**, identical ⭐ |

### 2.4 Source of the spec, and the banked primary

DD's training draw is **not stated in the 17-page paper** — it is read from the released code
(`hustvl/DiffusionDrive@9b52ed0`), the same precedent as DD-v2's reward. ⭐ **DD-v1 `2411.15139` was
already banked and I VERIFIED IT BY CONTENT**: recorded `sha256 6ad4f8a379494eeb…2b8600` recomputed
from the 26 370 647-byte PDF on disk — **MATCH**. (Control: `library.json` holds 487 entries, 485
carrying an arXiv id, so the single DD hit is not a broken query.)

### 2.5 What I changed

`stack/tanitad/refs/refc_sampler.py` (mine) — added `draw_train_timesteps(batch, t_max, device,
generator)` and `train_noise(sched, x0_n, t, generator)`, the consumer the flag never had. The
exclusive bound, the `t_max = 1` regression arm, and a refusal of a `[B,1,1,1]` timestep (which
would broadcast silently and noise every candidate at a different level) are all pinned.

⛔ **The training branch in `_sample` is NOT wired**, because `refc.py` is another owner's file.
The exact diff is §4.

---

## 3. P4 — THE 15-TOKEN STRATEGIC VOCABULARY

### 3.1 Plumbing or architecture? **BOTH — and the split is the useful answer**

| layer | state | verdict |
|---|---|---|
| **LABEL side** | `v7_labels.HEADS["str_goal"|"str_action"]`, `V7Label.str_goal/.str_action`, `head_mask()`, `class_weights()`, `assert_mask_matches_presence()` **all already exist** | ⭐ **PLUMBING — already done** (a sibling's) |
| **MODEL side** | ⛔ **NO strategic token head exists anywhere in REF-C** | ⛔ **ARCHITECTURE — built here** |

⚠️ **THE NEAR-MISS THAT WOULD HAVE MADE THIS LOOK DONE.** `refc_v3.py` **does** define
`self.str_goal_head` — but it is `nn.Linear(d_ctx, **3**)`, a **geometric bearing** head
(cos/sin/valid), not a token classifier. v6's `goal_head_tac` has **0 occurrences** in the REF-C
line, exactly as the brief warned. There is **no 8-way or 7-way slot** in REF-C. ⇒ The new head is
deliberately named **`str_goal_tok_head`**, mirroring the sibling's `tac_goal_tok_head` /
`tac_goal_head` split, so the name collision cannot be mistaken for the work being finished.

### 3.2 Bar A1 — per-class support: **FAIL AS WRITTEN**

**Bar:** ≥ 6 of 8 goal and ≥ 5 of 7 action classes at ≥ 1 % support.
**MEASURED (n = 801 clips, banked `s2_labels_v7.jsonl`), `d = 8` and `d = 7`:**

| `str_goal` class | n | share | dead because |
|---|---|---|---|
| `FOLLOW_ROUTE` | 549 | **68.539 %** | — |
| `TURN_RIGHT_FOLLOW_ROUTE` | 132 | 16.479 % | — |
| `TURN_LEFT_FOLLOW_ROUTE` | 120 | 14.981 % | — |
| `STOP_AT_FOLLOW_ROUTE` | **0** | 0 % | ⚠️ **ABSENT from this split** (extractable in principle) |
| `EXIT_LEFT` / `EXIT_RIGHT` / `LANE_CHANGE_L` / `LANE_CHANGE_R` `_FOLLOW_ROUTE` | **0** | 0 % | ⛔ **`NOT_YET_EXTRACTABLE`** (declared, with reasons) |

| `str_action` class | n | share | dead because |
|---|---|---|---|
| `HOLD_MAIN_ROAD` | 459 | **57.303 %** | — |
| `PREPARE_TURN_R_FOLLOW_ROUTE` | 132 | 16.479 % | — |
| `PREPARE_TURN_L_FOLLOW_ROUTE` | 120 | 14.981 % | — |
| **`REDUCE_TO_FOLLOW_ROUTE`** | **90** | **11.236 %** | ⛔⛔ **OFF-VOCABULARY — see §3.3** |
| `PREPARE_STOP` / `RESUME_CRUISE` `_FOLLOW_ROUTE` | 0 | 0 % | ABSENT from this split |
| `PREPARE_EXIT` / `PREPARE_LANE_CHANGE` `_FOLLOW_ROUTE` | 0 | 0 % | ⛔ `NOT_YET_EXTRACTABLE` |

⇒ ⛔ **POPULATED: 3 of 8 goal, 3 of 7 action. THE BAR FAILS.**

⚠️ **RECONCILING WITH THE BRIEF'S "only 4 of the 8 goal tokens are populated".** Both are right
about different things: **4** goal tokens are *declared extractable* (8 minus the 4 in
`NOT_YET_EXTRACTABLE`); only **3** carry *support* on this 801-clip sample. The fourth,
`STOP_AT_FOLLOW_ROUTE`, is extractable but absent here. ⇒ ⛔ **"unpopulated" hides two different
problems with different fixes** — a detector work item vs a possible sampling artifact — and
`support_report()` now separates them per class.

⇒ ⭐ **The FAIL clause's committed deliverable is the RESTRICTED vocabulary: 6 of 15 tokens are
trainable on this corpus** (3 goal + 3 action). The other 9 are blocked on **the corpus**, not on
any head: 6 by declared non-extractability, 2 absent from this split, 1 off-vocabulary.
**"Use the whole strategic vocab" is, on PhysicalAI today, 6 of 15.**

### 3.3 ⛔⛔ A NEW BLOCKER NOBODY HAD NAMED: an 11.24 % silent hole

**`REDUCE_TO_FOLLOW_ROUTE` appears on 90 of 801 `a_str` records (11.236 %) and is NOT in
`STRATEGIC_ACTION_TOKENS_V7`.** The absence is deliberate and **PINNED** —
`test_vocab_v7_frozen.py` asserts `"REDUCE_TO_FOLLOW_ROUTE" not in STRATEGIC_ACTION_TOKENS_V7`.
⇒ **the label writer and the vocabulary disagree.**

⛔ **Why this is worse than a crash.** A `dict.get(tok, -1)` encoder returns `-1`, `ignore_index`
swallows it, and **11.24 % of the action supervision is deleted while every log still says the head
is trained**. `encode_targets` therefore **RAISES `OffVocabularyToken`**, naming the token and the
vocabulary; the silent-delete behaviour survives only as an explicitly-named `strict=False` audit
path. ⭐ **Proven reachable by tripping it**, both branches.

⚠️ **SCOPE, STATED NOT ASSUMED.** This is MEASURED on the **v7** sample, which is the only strategic
label blob on this dev box. The live arm trains on **v7.2**
(`/workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz`), which I could not read. **The refusal is
what makes that difference visible on the first batch instead of after 40 k steps.**
⇒ **WORK ITEM for the label owner: does v7.2 still emit `REDUCE_TO_FOLLOW_ROUTE`?**

### 3.4 Bar A2 — the supervisable fraction

⛔ **11.43 % of the usable horizon can be supervised; 88.57 % carries NO strategic GT.**
MEASURED: **one record per clip** (801/801 clips have exactly 1), **all at `t0_s = 8.0`**, valid
± 2.0 s ⇒ 4.0 s of a **median 35.0 s** available horizon. ⇒ ⭐ **The strategic tokens DO share the
tactical band limit** (the plan's 88.13 %; I measure **88.57 %** on this sample — the small
difference is the horizon denominator, not a different mechanism).

### 3.5 Bar A3 — label quality: **PASS**

⭐ **Provenance is `geometry` on 801/801 strategic records.** ⇒ **The 175-grounded / 604-disputed
traffic-light problem does NOT apply to the strategic tokens** — they are not `vlm-cot` and not
`disputed`. Their quality is that of the ego reconstruction, and no quality filter is needed.
(Control: the parser read `a_tac.lat` on **801/801** of the same rows, so a zero would have been a
claim about the labels and not about my parser.)

### 3.6 ⛔ Bar A4 — THE ECHO GATE: **CONDITIONAL**

The flagship route head was an **exact bijection of the nav we feed it (369/369), scored 1.0000** —
an echo read as skill. Nav is a *legitimate* inference input, so this had to be measured, not assumed.

| MEASURED, n = 801 | `g_str` | `a_str` |
|---|---|---|
| `nav → token` determinism | **0.8302** | 0.7253 |
| mutual information | **0.6444 bits of H = 1.2125** | 0.6563 of 1.6537 |
| **share of the label already in nav** | ⛔ **53.1 %** | 39.7 % |

⇒ **Not a bijection — but a MAJORITY.** Per the pre-registered ladder this is **CONDITIONAL**:
⛔ **a nav-fed strategic goal head would report half its own input back as skill.**
⇒ **`StrategicTokenHead.forward(self, ctx)` takes ONE tensor and has no `nav` parameter and no
`**kwargs`** — there is no path through which nav can arrive. A nav-fed arm is a separate,
explicitly named experiment that **owes a nav-ablated control** before any of its numbers is
quotable. Pinned by a signature test.

### 3.7 Controls at their KNOWN values, and the regression arm

⭐ **ALL CONTROLS PASS** (`raw/p4_mechanism.json`), each **proven reachable by tripping it**:

| control | required value | MEASURED |
|---|---|---|
| majority-class, macro recall over FULL vocab | **exactly 1/8 = 0.1250** | **0.1250** |
| majority-class, macro over the 3 SUPPORTED | **exactly 1/3 = 0.3333** | **0.3333** |
| a CONSTANT head's **pooled** accuracy | the majority share | **0.6854** ⛔ *would read as a working head* |
| the same head's **macro recall** | the no-information value | **0.3333** ⭐ *this is why pooled accuracy is inadmissible* |
| a class with **no support** | recall **`None`**, never 0.0 | `None` (a 0.0 would depress the macro of a class never given a chance) |
| off-vocabulary token | **RAISES** | raises `OffVocabularyToken`; `strict=False` returns −1 |
| head-width mismatch (3-wide vs 8-token) | **RAISES** | raises `StrategicVocabMismatch` |
| gradient at weight **0.0** | `p.grad` **NOT None**, values all-zero | **6/6 tensors**, all-zero ⇒ `p.grad is None` still means *not wired* |
| all-ignored batch (the 88.57 % band) | **finite**, `n_supervised = 0` | loss **0.000000**, `n_supervised = 0` ⇒ reads as *"nothing in band"*, not *"head broken"* |

⛔ **DELIBERATE-REGRESSION ARM (shuffled labels): VALID, at the second attempt — and the first
attempt is the finding.**

| arm | macro recall (held-out) |
|---|---|
| REAL labels | **1.0000** |
| **SHUFFLED labels** | **0.3044** vs the no-information value **0.3333** ⭐ |

⚠️ **The first version fit and scored the SAME 801 rows and BOTH arms read 1.0000** — a 2-layer MLP
memorises 801 shuffled labels. **The gate correctly reported VOID.** The fix is a held-out split,
and it is the same discipline the estimator rules impose on λ: *fit everything on the fit split; the
scored split is scored, never tuned on.* Logged because a leaking probe that had reported only the
REAL arm's 1.0000 would have looked like a triumph.

### 3.8 Part B is NOT answered

⛔ **Whether the head LEARNS the strategic tokens from vision is NOT established here.** That needs a
tiny-rig arm scoring **macro per-class recall over the 3 supported classes against the
majority-class control at 1/3**, with a **replicate**. P4 is **MECHANISM-VALIDATED and
CORPUS-BOUNDED**, and the corpus bound (11.43 % of frames, 6 of 15 tokens) is the honest ceiling on
anything Part B can return.

### 3.9 Reconciling the two PI instructions

⭐ `StrategicTokenHead` reads `ctx` and emits logits; **nothing downstream consumes them.** So the
strategic **VOCABULARY** is supervised (gradient reaches the trunk) while the strategic **LAYER's
conditioning path** stays switchable via `--no-strategic`. ⛔ **I did not touch, remove, or re-enable
that flag.** A return path from this head into the decoder would silently re-enable the very
conditioning the ablation exists to measure, and the module docstring says so.

---

## 4. ESCALATIONS — exact diffs, for files I do not own

### 4.1 ⛔ REQUIRED for P13 — `stack/tanitad/refs/refc.py`, `AnchoredDiffusionDecoder._sample`

The consumer exists (`refc_sampler.draw_train_timesteps` / `train_noise`) and is tested; it needs a
training branch. Insert at the top of `_sample`'s ladder section, replacing the unconditional
inference path **for training only**:

```python
        # ---- refcv5 P13: DD's TRAINING draw. `t ~ U[0, sampler_train_t_max)`,
        # ONE draw and ONE pass -- the truncated ladder [10, 0] is an INFERENCE
        # object. Without this the flag has NO CONSUMER: MEASURED, the realised
        # training timesteps are {0, 10} and are IDENTICAL under t_max 50 and 1.
        if self.training and int(getattr(cfg, "sampler_train_t_max", 0)) > 0:
            t = rs.draw_train_timesteps(b, int(cfg.sampler_train_t_max), dev)
            x_n, _ = rs.train_noise(self.sched.to(dev), x0_n, t)
            x_path = self._state_to_path(x_n * norm, v, metre)
            conf, du = self._decode_ctrl(kv, cond, x_path,
                                         t.to(torch.float32), agents, agent_pad)
            u0_hat = (x_n + du) * norm
            return (self._state_to_path(u0_hat, v, metre), u0_hat, conf,
                    {"sampler": "ddim", "sampler_space": cfg.sampler_space,
                     "sampler_train_t_max": int(cfg.sampler_train_t_max),
                     "sampler_train_t_mean": float(t.to(torch.float64).mean()),
                     "sampler_ranks_the_fan": bool(self.sel.refined)})
```

⚠️ **Do not land this without P13-B3.** Paper-faithfulness is not the bar; the tiny-rig benefit arm
(with a replicate) is. Landing it changes the recipe every banked sampler arm was trained under.

### 4.2 ⚠️ OPTIONAL for P14 — same file, the emitted-fan scoring pass

With `sel_score_emitted_t = 0` the current `self._decode(kv, cond, x, t_e, …)` scores the emitted fan
through `time_embed` (an `nn.Embedding`, 3 rows) at index 0 — the **same row the classifier surface
uses**, on a **shared** `traj_proj` / `layers` / `conf_head`. That is coherent and is why **no
`refc.py` change is required today**. A sampler-native version would route through `_decode_ctrl`
(the continuous `time_mlp`) at `t = 0`. ⛔ **Not a blocker; do not bundle it with 4.1.**

### 4.3 ⚠️ For the label owner — `v7_labels` / the v7.2 writer

Does **v7.2** still emit `REDUCE_TO_FOLLOW_ROUTE` (11.236 % of `a_str` on v7)? If yes, either add it
to `STRATEGIC_ACTION_TOKENS_V7` (and unpin `test_vocab_v7_frozen`) or map it in the writer. Until
then `encode_targets` refuses it rather than deleting that supervision silently.

### 4.4 ⚠️ Cosmetic, unowned — `refc_v3_train.py` help strings

**18 pre-existing flags** carry non-ASCII in their `help=` (`--size`, `--arm`, `--v2-cache`,
`--goal-str`, …; chars `U+2014 U+26A0 U+26D4 U+2B50 U+FE0F`). ⚠️ **NOT a break** — `--help` exits
**0** even under `PYTHONIOENCODING=cp1252:strict`; the `UnicodeDecodeError` I first saw was my own
parent process decoding the child's UTF-8 as cp1252. Recorded so the next reader does not
re-diagnose it as a defect. My three flags are ASCII and pinned as such.

---

## 5. Deliverable manifest

| artifact | where it lives |
|---|---|
| `PREREG.md` — bars, written before any number | repo, this directory |
| `RESULT.md` — this file | repo, this directory |
| `raw/p4_label_coverage.txt` · `p4_mechanism.json` | repo, `raw/` |
| `raw/p13_mutation.json` | repo, `raw/` |
| `raw/p14_fan_base30k.{txt,json}` · `p14_fan_xl30k.{txt,json}` · `p14_wiring.json` | repo, `raw/` |
| probe sources (`p4_label_probe.py`, `p14_banked_fan.py`, `p13_validate.py`, `p4_validate.py`, `p14_wiring.py`) | repo, `raw/` |
| **`stack/tanitad/refs/refc_strategic.py`** — the P4 head + losses + controls (NEW) | repo |
| **`stack/tanitad/refs/refc_sampler.py`** — `draw_train_timesteps`, `train_noise` (P13) | repo |
| **`stack/scripts/refc_v3_train.py`** — P14 flags, paired refusal, config stamp | repo |
| **`stack/tests/test_p4_p13_p14_wiring.py`** — 17 pins, all mutation-proven (NEW) | repo |

**Suite:** `test_p4_p13_p14_wiring.py` **17 passed**. Adjacent suites re-run for regression.
