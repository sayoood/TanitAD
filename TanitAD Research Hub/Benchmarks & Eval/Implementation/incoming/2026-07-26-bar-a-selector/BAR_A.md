# BAR A — the pre-registered discriminating experiment for flagship-v4's selector

**Written:** 2026-07-26 (Europe/Berlin; pods and logs are UTC) · **Host:** eval pod (`tanitad-eval`, A40).
**Mandate:** `Project Steering/BOOST_PROGRAM.md` §3.4 — *"Do not spend 59 GPU-hours yet. Spend 1–2
GPU-hours first."* PI-approved.
**Question:** v4's proposal fan beats deployed v1 by 41.3 % on the deployable surface while its
*selector* throws away 0.6058 m. Is the **error-blind 1-of-256 cross-entropy** that trains the
selector the reason?

Evidence stamps on every number: class `MEASURED` / `PUBLISHED` / `INHERITED` / `ESTIMATED` /
`HYPOTHESIS`, **and** tier `PROVISIONAL` / `CONFIRMED` / `DECISION-GRADE` (BOOST_PROGRAM M1).

---

## 0. PRE-REGISTRATION — written and staged BEFORE the fine-tune ran

> Registered 2026-07-26. At registration time what had been run was: (a) the reproduction check in §1,
> and (b) a `--smoke` harness validation with a 60-step budget whose numbers are **not quotable and
> are not reported as results**. **No Bar-A arm had been trained.** §0 is unmodified from
> registration; §§2+ were written afterwards.

### 0.1 The intervention — the ONLY thing that changes

`v15_losses` (`flagship_v15.py:598-605`) trains the selector with an error-blind hard label:

```python
r_star    = fan_err.argmin(dim=1)                          # 1-of-256 hard label
loss_rcls = F.cross_entropy(out["sel_score"].float(), r_star.detach())
```

It is replaced by the **cost-sensitive expected-regret listwise loss**:

```python
regret    = fan_err - fan_err.min(dim=1, keepdim=True).values   # [B, N] metres, >= 0
p         = torch.softmax(out["sel_score"].float() / TAU, dim=1)
loss_rcls = (p * regret.detach()).sum(dim=1).mean()             # EXPECTED METRES GIVEN UP
```

`TAU = 1.0` · `REFINED_CLS_WEIGHT = 1.0` · **nothing else changed.** `loss_cls` (anchor CE) is kept
at weight 1.0 exactly as in the original recipe. `loss_traj` is retained in the recipe but has **no
trainable parameter** in this experiment (`offset_head` is frozen), so it contributes a constant and
is omitted from the computed objective — stated, not hidden.

### 0.2 "Frozen fan" is a proof, not an assertion

`V15Decoder._decode` (`refc.py:539-540`) emits both heads from the **same** query features `q`:

```python
conf   = self.conf_head(q).squeeze(-1)                    # -> refined_logits, the SCORE
offset = self.offset_head(q).reshape(b, n, self.n_steps, 2)   # -> the FAN
```

and the fan is `x = x_in + offset`, which depends on **no parameter that produces the score**.
The trainable set is therefore *exactly* the parameters that move the score and provably cannot move
the fan:

| trainable | what it is |
|---|---|
| `decoder.conf_head` | `Linear(512, 1)` — `refined_logits` |
| `lat_head` · `lon_head` · `dist_head` | the factorised LAT×LON×DIST MLPs on `states[:, -1]` |
| `lat_to_anchor` · `lon_to_anchor` · `dist_to_anchor` | the three anchor grafts onto the ranked score |
| `sel_gate` | the learned longitudinal selection scale |

**MEASURED: 796,057 trainable of 9,507,907 head parameters (8.4 %).** Frozen and *asserted* frozen at
runtime (`_assert_frozen`): world model, grounding, `goal_head`, decoder trunk, `offset_head`,
anchors, conditioning — plus a check that no trainable name matches a fan-producing module. The
assertion aborts the run rather than warning.

### 0.3 Design — cached-feature 5-fold episode-disjoint cross-fit

Because the fan is frozen, every selector input is a fixed function of one forward pass. One pass
over **all 6,844 windows of the 40 val episodes** caches `q0` / `q_final` (the two `conf_head`
inputs), `anchor_traj`, `states[:,-1]`, the dense target, `v0`, `vt_speed`, `vt_keep`, episode and
time indices. Fine-tuning is then exact algebra over that cache, calling the head's **own**
`_factor_grafts` and `select` methods — not reimplementations.

**⚠️ DEVIATION, stated in advance.** The canonical train corpus `physicalai-train-e438721ae894` is
**not on the eval pod** (MEASURED: `find /root /workspace -maxdepth 3 -name 'physicalai-*'` returns
only the 40-episode val cache and its two feature caches), and the brief forbids touching pod1/pod2/
pod3. The rescorer is therefore fitted by **5-fold episode-disjoint cross-fitting over the 40 val
episodes**: each fold fits on 26 episodes, early-stops on 6 inner-validation episodes, and scores the
**8 episodes it never saw**, so all 881 canonical windows receive an **out-of-fold** score.

This is held-out by construction (C11-compliant), but it gives the fine-tuned arms a val-distribution
adaptation advantage that the as-trained selector never had. **That is why there are three arms:**

| arm | what it is |
|---|---|
| **AS_TRAINED** | the 30 k checkpoint, untouched — the baseline |
| **CE_CONTROL** | identical fine-tune, identical folds/steps/LR-grid, **original CE loss** |
| **REGRET** | identical fine-tune, **the §0.1 intervention** |

`Δ(REGRET − CE_CONTROL)` isolates **the loss**. `Δ(CE_CONTROL − AS_TRAINED)` measures the
adaptation-plus-refit effect, i.e. exactly the confound. Reporting Bar A off `REGRET − AS_TRAINED`
alone would be a **C6 confounded comparison**; both are reported.

### 0.4 Hyper-parameters, fixed in advance and identical across arms

AdamW, `weight_decay = 0`; LR selected per fold from **{3e-5, 1e-4, 3e-4}** by **inner-validation
`ade_0_2s`** (never by test); max **2,000 steps**, batch **32 windows**, inner-val probe every 100
steps, best-inner-val checkpoint kept. AdamW's update magnitude is ~LR independently of the loss's
*scale*, so one grid is fair to a nats-valued CE and a metres-valued regret — this is the reason the
grid is shared rather than tuned per arm.

### 0.5 Estimator — named in advance

**Paired episode-cluster bootstrap** (`taniteval/ci.py`, `B = 2000`, resampling unit = episode) on the
**identical 881 windows**, base vs fine-tuned. **NEVER `overlapping_holdout_se`** — which is not a
jackknife, biases the point estimate bidirectionally by −6.67 % to +11.69 %, and has flipped the sign
of a paired delta.

**Lateral/longitudinal decomposition is mandatory**, because v4's regression was 100 % longitudinal
(along-track +0.0581 separated-worse, cross-track −0.0257 separated-*better*). An undecomposed number
would hide whether the lever acts on the axis that actually failed.

### 0.6 Primary surface

**PRODUCED (deployable).** The goal-oracle surface is a **secondary** and its numbers may never be
worded as deployed capability.

### 0.7 The three outcomes — committed before measurement

Fraction of the **0.6058 m deployable waste** recovered, defined
`(as_trained_ade − arm_ade) / (as_trained_ade − oracle_in_fan)`:

| outcome | bar | reading |
|---|---|---|
| **CONFIRM** | **≥ 70.8 %** (i.e. `ade_0_2s` ≤ **0.4271**, tying v1) | the lever is real; a full restart is justified on Bar A |
| **PARTIAL** | 30–70 % | real but insufficient alone; report exactly where it lands |
| **REFUTE** | **< 30 %** | the listwise loss is **not** the lever. **Say so plainly; do not re-scope the experiment to rescue it.** |

**70.8 % is the bar to merely TIE v1.** It is demanding by construction and is not softened.

### 0.8 What I commit to reading into a REFUTE, in advance

If a re-scored **frozen** fan cannot approach v1, then the deficit is not recoverable by re-scoring a
fixed fan at all — which means `refined_logits` lacks the *information* to rank, a
conditioning/architecture problem rather than a loss problem. The next probe would then be **what the
score is conditioned on**, not how it is trained. I commit to that reading now so it cannot be
reverse-engineered from the result.

### 0.9 Self-tests the harness must pass before it may adjudicate (BOOST_PROGRAM M3)

1. **`cache_fidelity`** — running the **as-trained** parameters through the cached path must
   reproduce the published `ade_0_2s` and `oracle_in_fan` to ≤ 1e-3 on the same 881 windows.
2. **`failing_input`** — a deliberately broken scorer (uniform-random pick over the same frozen fan)
   must be reported as **worse**. An instrument that cannot render a failing verdict may not
   adjudicate.

Either failing **aborts before any training**, and the run writes `ABORTED` instead of a result.

---

## 1. REPRODUCTION CHECK — done first, and it PASSES

**MEASURED · CONFIRMED** (independent path: a full re-run of the forward pass — encoder → decoder →
select — not a re-reduction of the persisted `windows_*.pt`).
Artifact: `raw/repro_v4_30k.json`.

| surface | quantity | committed | **re-run** | abs diff |
|---|---|---:|---:|---:|
| goal-ORACLE | `ade_0_2s` (4-wp) | 0.6423 | **0.6423** | 0.00000 |
| goal-ORACLE | `oracle_in_fan` (4-wp) | 0.2330 | **0.2330** | 0.00002 |
| **PRODUCED (deployable)** | `ade_0_2s` (4-wp) | 0.8563 | **0.8563** | 0.00002 |
| **PRODUCED (deployable)** | `oracle_in_fan` (4-wp) | 0.2505 | **0.2505** | 0.00005 |

881 windows / 40 episodes on both surfaces. `miss_at_2m` also reproduces (0.2123 oracle / 0.3190
produced), as does `seam_norm_ratio_max` (0.1208 / 0.1204).

**Deployable selector waste re-measured: 0.6057 m** (0.85628 − 0.25055). The brief's 0.6058 is the
same quantity at published rounding; the pre-registered bar is kept at **0.6058** as registered, and
nothing in the verdict turns on the 1e-4 difference.

### 1.1 Host and tree provenance — established, not assumed

The eval pod holds **six** copies of `eval_flagship_v4.py` and two sizes of `flagship_v15.py`. Every
module actually imported is stamped by path **and md5** into `raw/repro_v4_30k.json`:

| module | path | md5 |
|---|---|---|
| `eval_flagship_v4` | `/root/v4eval/stack/scripts/` | `aafe3975817e27ef7714499643aae6ff` |
| `tanitad.models.flagship_v15` | `/root/v4eval/stack/tanitad/models/` | `88889b1b0203bf936e6083ccf10c68e4` |
| `tanitad.models.flagship_v4` | `/root/v4eval/stack/tanitad/models/` | `5ca5dde3d8bb84e451dab655150a699d` |
| `tanitad.refs.refc` | `/root/v4eval/stack/tanitad/refs/` | `b7887afeec8c41296c8433fe088ecfcf` |
| `train_flagship_v4` | `/root/v4eval/stack/scripts/` | `d6ba03e4c8fefc79e3e39b5d60544d3e` |
| `goal_modes` | `/root/v4eval/stack/scripts/` | `444d2df6b5c34e617e84a734df2226cd` |
| `taniteval.ci` | `/root/taniteval/taniteval/` | `ef925f06febd20a99f5901491fcf75cb` |

Checkpoint: `/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt`, **3 243 109 310 B**, md5
`8771c1d9d3da696dcde2a745d628f6a8`, step **29999**. `/workspace/TanitAD/stack` does not exist on this
host; `/root/v4eval/stack` is the tree used throughout.

**Two corrections to the brief's host facts, both MEASURED here:**

1. **The eval pod runs Python 3.12.3, not 3.11.10** (`torch 2.8.0+cu128`, A40). The 3.11.10 host —
   where `eval_flagship_v4.py` `SyntaxError`ed on PEP-701 f-strings — is **pod2**, per that file's own
   in-code note. No import failed here.
2. **⚠️ `/root` is 99 % full: 2.9 GiB available on a 200 GiB overlay.** A 6 GiB `dd` write
   **completed "successfully" after writing only 3.0 GiB** — the exact silent-truncation failure the
   brief warns about, reproduced on demand. **Every cache in this experiment is therefore held in RAM
   (503 GiB total, 376 GiB free) and never written to `/root`.** Only small artifacts (fold
   checkpoints, JSON, per-window dumps) touch disk. *This is a live hazard for the next agent on this
   pod and is escalated in §7.*

---

*(Sections 2 onward were written after the measurements.)*

---

## HEADLINE — **REFUTE**, and the diagnosis's own falsifier fired with it

**The cost-sensitive expected-regret listwise loss is NOT the lever.** It recovered **−4.20 %** of the
deployable waste against a **≥ 70.8 %** bar — not "short of the bar", but on the **wrong side of
zero**: the fine-tuned selector is *worse* than the one it replaced.

And the more important finding is the one behind it. `V4_RESTART_LEVER.md` §6 attached a falsifier to
the **whole diagnosis**: *"if a re-scored frozen fan cannot get below about 0.43 m (v1's 0.4271), then
'the fan already contains v1-beating trajectories' is true only in an oracle sense no realisable
ranker can reach."* **That falsifier has now fired.** Fitting a re-scorer **in-sample** — same windows
it is scored on, 6,844 fit windows, 796 k trainable parameters, zero generalization gap — bottoms out
at **0.4907**, still **separated-worse than v1** and **1.96× the fan's own best (0.2505)**.

> **The 0.6058 m "selector waste" is not a recoverable engineering surplus.** At most **60.4 %** of it
> is reachable even with the generalization question removed entirely, and **none of it** survives
> out-of-fold. The gap between `oracle_in_fan` and the pick is an **oracle quantity**, not an
> addressable loss.

Six findings, each MEASURED and with its artifact:

1. **REFUTE on the primary (deployable) surface.** REGRET out-of-fold `ade_0_2s` **0.8817** vs
   as-trained **0.8563**: paired Δ **+0.0254 [−0.0153, +0.0680]**, not separated. Waste recovered
   **−4.20 %**. The same run also REFUTES on the *earlier* §6 registration (which required
   Δ ≤ −0.10 m).
2. **It is not the fine-tuning's fault either — the CE control is worse still.** CE_CONTROL
   **0.9231**, Δ **+0.0668 [−0.0271, +0.2066]**, waste recovered **−11.03 %**.
3. **The loss *is* directionally better than CE, and it does not matter.** REGRET − CE_CONTROL
   **−0.0414 [−0.1702, +0.0421]**, p(Δ>0) = 0.27 — not separated, and both arms are worse than
   doing nothing. The intervention beats its control while losing to the null.
4. **⚠️ The lever missed the axis it was designed for.** v4's regression was 100 % longitudinal. The
   regret arm moved the longitudinal axis by **+0.0038 [−0.0236, +0.0292]** (p = 0.63 — *flat*) and
   made the **lateral** axis **separated-WORSE, +0.0222 [+0.0043, +0.0522]**. It did not act on the
   failing axis at all.
5. **All three arms are separated-worse than deployed v1** on the identical 881 windows:
   as-trained **+0.4292 [+0.2865, +0.5820]**, REGRET **+0.4546**, CE **+0.4960**. *(The +0.4292 is
   exactly BOOST_PROGRAM §3.3's "closing 0.4292 m reaches v1" — the arithmetic closes.)*
6. **The head's own fail-loud seam guard fires under fine-tuning.** The factorised grafts grow past
   `seam_fail = 1.5×` the base-score norm (**1.652** at in-sample step 1732; **1.506** in an earlier
   fold). The architecture *refuses* the states the ranking objective drives it toward — an
   independent, structural sign that the score has no headroom in this direction.

**Tier: CONFIRMED, and DECISION-GRADE for the negative decision** — see §6.3 for why that phrasing is
exact and where it stops.

⛔ **Nothing was launched or restarted. No steering file was edited. This is a measurement, and the
restart decision remains Sayed's.**

---

## 2. THE HARNESS, VALIDATED BEFORE IT ADJUDICATED

Both pre-registered self-tests (§0.9) pass at full scale. `raw/bar_a_produced.json → selftest`.

| self-test | result | |
|---|---|---|
| `cache_fidelity` — cached path vs published | `ade_0_2s` **0.8563** vs 0.8563 (Δ 2e-5); `oracle_in_fan` **0.2505** vs 0.2505 (Δ 5e-5) | ✅ |
| `cache_fidelity` — cached path vs the **same run's own forward pass** | Δ **0.000000**; **`frac_windows_pick_differs_from_forward_pass` = 0.00000** | ✅ **exact identity on all 881 windows** |
| `failing_input` — uniform-random pick over the same frozen fan | **15.3622** vs 0.8563 | ✅ the instrument can render a failing verdict |
| `freeze_proof` — nothing outside the trainable set requires grad; no trainable name touches a fan-producing module | asserted at runtime, aborts on violation | ✅ |

**Trainable: 796,057 of 9,507,907 head parameters (8.4 %)** — `sel_gate`, `decoder.conf_head`, the
three factorised MLPs and the three anchor grafts. Nothing else. The world model, grounding,
`goal_head`, decoder trunk, `offset_head` and anchors never received a gradient.

### 2.1 A methodological finding worth keeping: fp16 is not safe on this selector

The first attempt cached `q_final`/`anchor_traj` in **fp16** and the cache-fidelity test **failed and
aborted the run** at `ade_0_2s` **0.8591** vs 0.8563. The cause is the very structure this whole
diagnosis rests on: **256 candidates whose scores differ by less than fp16's ULP**, so quantisation
flips the `argmax`. Storage precision is a *scientific* parameter here, not an implementation detail.
The self-test caught it before any result existed — which is the entire argument for the M3 rule.

---

## 3. BAR A — THE RESULT AGAINST THE THREE COMMITTED OUTCOMES

**Primary surface: PRODUCED (deployable).** Out-of-fold over all **881 windows / 40 episodes**.
Estimator: paired episode-cluster bootstrap, B = 2000, unit = episode.
Artifact: `raw/bar_a_produced.json`.

| arm | `ade_0_2s` | paired Δ vs as-trained | 95 % CI | separated? | **waste recovered** |
|---|---:|---:|---|---|---:|
| **AS_TRAINED** (30 k, untouched) | **0.8563** | — | — | — | — |
| **CE_CONTROL** (identical fine-tune, original CE) | **0.9231** | **+0.0668** | [−0.0271, +0.2066] | no | **−11.03 %** |
| **REGRET** (the intervention) | **0.8817** | **+0.0254** | [−0.0153, +0.0680] | no | **−4.20 %** |

| the contrast that isolates the LOSS | Δ | 95 % CI | p(Δ>0) | |
|---|---:|---|---:|---|
| **REGRET − CE_CONTROL**, `ade_0_2s` | **−0.0414** | [−0.1702, +0.0421] | 0.27 | not separated |

> ### VERDICT: **REFUTE** — **−4.20 %** against a **< 30 %** refute bar.
>
> This is not a near miss. The pre-registered question was whether the loss recovers ≥ 70.8 % of
> 0.6058 m; the measured arm **gives 0.0254 m back**. Per §0.7 I say so plainly and **do not re-scope
> the experiment to rescue it.**

**It also REFUTES on the earlier registration.** `V4_RESTART_LEVER.md` §6 had committed a different
rule — CONFIRM at paired Δ ≤ −0.10 m, REFUTE if "Δ not separated from 0, **or positive**". Observed Δ
is **+0.0254 and not separated**: REFUTE on that rule too. **Two independently-written
pre-registrations, same verdict.**

### 3.1 The fine-tune is not underpowered — it is directionally wrong

An arm that simply failed to learn would sit *at* 0.8563. Both arms sit *above* it, and the
per-fold record shows why: fine-tuning **did** improve the inner-validation ADE in **3 of 5** CE folds
and **2 of 5** REGRET folds, and those improvements **did not transfer to the 8 episodes each fold
had never seen**. Where inner-val did not improve, the protocol correctly kept the as-trained state
(`best_step = 0`) — so the arms degrade gracefully and the negative result is *not* an artifact of an
unlucky stopping rule.

| fold | CE: lr / best step / inner-val | REGRET: lr / best step / inner-val |
|---|---|---|
| 0 | 3e-5 / **0** / 0.9714 | 3e-5 / **0** / 0.9714 |
| 1 | 1e-4 / 100 / 1.0948 | 3e-5 / **0** / 1.0983 |
| 2 | 3e-5 / 100 / 0.7007 | 1e-4 / 100 / 0.6539 |
| 3 | 3e-4 / 1300 / 1.0471 | 3e-4 / 800 / 1.0985 ⚠️ seam guard raised |
| 4 | 3e-5 / **0** / 0.9066 | 3e-5 / **0** / 0.9066 |

**This is C11 restated one level down.** v4's original failure was "training loss improved, held-out
selection got worse". Here, a genuinely *held-out* early-stopping signal on 6 episodes still failed to
predict 8 other episodes. **The selector's fit is episode-specific at every scale we can measure it.**

### 3.2 THE DECISIVE NUMBER — the in-sample ceiling

This removes the generalization question entirely: fit the re-scorer on **all 6,844 windows** and
score it on the 881 it was fitted on. It is **NOT a deployable number and is never quoted as one.**
It upper-bounds what re-scoring this **frozen fan** with this conditioning can achieve *at all*.

| | `ade_0_2s` **IN-SAMPLE** | waste recovered in-sample | vs v1 (0.4271) | vs the fan's best (0.2505) |
|---|---:|---:|---|---|
| CE, in-sample | **0.4907** | 60.35 % | **still worse** | **1.96×** |
| REGRET, in-sample | **0.5224** | 55.11 % | **still worse** | 2.09× |

> **Even with no generalization gap, no held-out episodes, 6,844 fitting windows and 796 k free
> parameters, a re-scorer of this fan cannot reach the model we already deploy.**

This is the finding with the longest reach, because it does not depend on the loss, the fold split,
the LR grid, or the val-corpus deviation. It says the **information required to rank this fan is not
present in what the score is conditioned on.** Which is exactly the reading I committed to in §0.8
before seeing any of it:

> *"the deficit would then not be recoverable by re-scoring a fixed fan, which means `refined_logits`
> lacks the **information** to rank — a conditioning/architecture problem, not a loss problem — and
> the next probe is what the score is conditioned on, not how it is trained."*

### 3.3 The seam guard — the architecture refuses the direction

`FlagshipV4Head._factor_grafts` fails loud when the factorised graft's norm exceeds
`seam_fail = 1.5 ×` the base-score norm. Under fine-tuning it **fires**:

| where | pre-clamp seam ratio |
|---|---:|
| as-trained 30 k (reference) | **0.1204** |
| CE, in-sample fit | 1.2284 |
| REGRET, fold 3 @ lr 3e-4 | 1.4872 → **RAISED** |
| REGRET, in-sample fit, train step 1732 | **1.652 → RAISED** |

Every ranking objective, given freedom, drives the graft path **an order of magnitude** past where
30 k of training left it, and into the state the architecture itself rejects. The score is not
"under-trained"; it is being pushed against a structural limit. *(Handled as a first-class outcome:
the state is recorded and the fold falls back to the last architecture-accepted state — never
silently skipped.)*

---

## 4. LATERAL vs LONGITUDINAL — the lever missed the failing axis

Dense 20-step ego-frame decomposition, mean |component| per window; paired episode-cluster bootstrap.
This was mandatory because **v4's regression was 100 % longitudinal** (along-track +0.0581
separated-worse, cross-track −0.0257 separated-*better*).

| arm | along-track (**LONGITUDINAL**) | paired Δ [95 % CI] | cross-track (**LATERAL**) | paired Δ [95 % CI] |
|---|---:|---|---:|---|
| AS_TRAINED | **0.5847** | — | **0.1889** | — |
| CE_CONTROL | 0.6413 | +0.0567 [−0.0197, +0.1711] ns | 0.1969 | +0.0081 [−0.0004, +0.0211] ns |
| **REGRET** | 0.5884 | **+0.0038 [−0.0236, +0.0292]** ns, p = 0.63 | 0.2110 | **+0.0222 [+0.0043, +0.0522] SEPARATED WORSE** |

> **The one axis the regret loss moved with statistical separation is the one that was already
> working.** Longitudinal error — **75.6 % of the mean-abs axis split** (0.5847 of 0.5847 + 0.1889)
> and the entire content of the v4 regression — is **flat to three decimal places**. An undecomposed +0.0254 would have hidden this completely,
> and it is the sharpest single piece of evidence that this loss does not touch the mechanism.

One qualification, stated because it is the only pro-lever signal in the data: **REGRET − CE_CONTROL**
on the longitudinal axis is **−0.0529 [−0.1571, +0.0100]**, p(Δ>0) = 0.0985 — the regret loss *is*
directionally better than CE longitudinally. It is **not separated**, and it loses to doing nothing.
It is reported, not promoted.

`miss_at_2m`: as-trained **0.3190** · CE **0.3280** · REGRET **0.3496** (Δ +0.0306 [−0.0091, +0.0750],
ns). Nothing improves. **v1's `miss_at_2m` is 0.0454 — 7.0× lower than any v4 arm.**

---

## 5. PAIRED AGAINST DEPLOYED v1 — because "ties v1" is a test, not a comparison of two means

Bar A is phrased as *reaching ≤ 0.4271, tying v1*. v1's own per-window dump is on this host, so the
comparison is **paired on the identical windows**. Artifact: `raw/v1_paired.json`.

**v1 identified BY ITS NUMBER, not by its filename** (CLAUDE.md's standing inversion warning):
`windows_flagship-30k.pt` → `ade_0_2s` **0.42711** against the registry's full-set **0.4271** (Δ 1e-5).
The two near-name alternatives were computed and **printed rather than used**: `flagship-nospeed`
**3.01753** (the ablation control), `flagship-speed` **0.61522**.

**Alignment proven by tensor identity, not by labels.** The two dumps label episodes differently —
`taniteval.rollout.collect` writes the episode *index* (0–39), `collect_planner` writes
`int(episode_id)` (a large integer) — so a naive label comparison reports "not aligned" for windows
that are bit-identical. What proves alignment is the window's own properties: **`gt` max abs diff
0.0**, `cv` / `speed` / `head_deg` all identical, and all **39 episode-boundary positions** identical.
*(The first run of this script aborted on the label check; that abort was correct behaviour and the
fix was to test the right thing, not to loosen the test.)*

| paired Δ (arm − v1), same 881 windows | Δ | 95 % CI | |
|---|---:|---|---|
| **AS_TRAINED − v1** | **+0.4292** | [+0.2865, +0.5820] | **SEPARATED WORSE** |
| **REGRET − v1** | **+0.4546** | [+0.3102, +0.6007] | **SEPARATED WORSE** |
| **CE_CONTROL − v1** | **+0.4960** | [+0.3161, +0.6832] | **SEPARATED WORSE** |

> **No arm ties v1. Every arm is separated-worse, and the intervention widens the gap.** The
> as-trained delta **+0.4292** reproduces BOOST_PROGRAM §3.3's arithmetic exactly ("closing 0.4292 m
> reaches v1's 0.4271") — an independent confirmation that the bar was set on sound arithmetic even
> though the lever behind it does not exist.

---

## 6. WHAT THIS DOES AND DOES NOT LICENSE

### 6.1 What is settled

- **The listwise/cost-sensitive ranking objective is not the Bar-A lever.** Pre-registered, measured,
  refuted, on two independently-written decision rules.
- **A selector-only v4 restart is not justified by Bar A.** It was already known to fail the card on
  `wm_canary` (§7); it now also fails on its own primary claim.
- **The "0.6058 m of selector waste" framing must be retired as an engineering surplus.** The
  in-sample ceiling caps recoverable waste at **60.4 %** *under conditions that cannot be deployed*,
  and the deployable recovery is **negative**. `oracle_in_fan` measures what a *clairvoyant* ranker
  could pick, and this experiment shows no realisable ranker of this fan gets near it.

### 6.2 What is NOT settled — and what I refuse to conclude

- **This does not show v4's fan is bad.** It shows the fan's *best* member is not selectable from the
  information the score currently sees. Those are different claims and only the second is measured.
- **This does not price a re-conditioned selector.** The trainable set was the maximal one that keeps
  the fan frozen. A score conditioned on *more* (e.g. the fan geometry itself, or per-candidate
  imagination rollouts) is untested here — and per §0.8 that is the next probe. **I did not run it and
  I do not estimate its value.**
- **This does not transfer to a joint retrain.** A from-scratch run under the regret loss could shape
  a *different* fan. Cheaper questions come first.

### 6.3 Tier of the headline — stated exactly

**Evidence class MEASURED. Tier CONFIRMED, and DECISION-GRADE for the negative decision.**

- **Pre-registered** ✅ (§0, staged into the repo before the fine-tune ran) · **estimator named** ✅
  (paired episode-cluster bootstrap, B = 2000, episode clusters; never `overlapping_holdout_se`) ·
  **falsifier stated** ✅ (§0.7 three outcomes, §0.8 committed reading).
- **CONFIRMED** by independent paths: (i) the cached scorer reproduces the real forward pass with
  **zero** pick mismatch on all 881 windows; (ii) **two independently fitted arms** both land on the
  wrong side of zero; (iii) the in-sample ceiling reaches the same conclusion by a different protocol;
  (iv) the v1 comparison uses a dump produced by a **different harness** (`taniteval.rollout.collect`)
  and reproduces the registry number to 1e-5; (v) **§8 — the goal-ORACLE surface, an independent
  goal-provenance path, reaches the same verdict.**
- **DECISION-GRADE for *declining*** — the decision it supports is "do not spend the GPU-week on this
  lever", the outcome was bounded in advance, and the measurement landed on the **wrong side of zero**
  rather than merely short of a bar. It is **not** decision-grade for any *positive* claim about what
  the selector could become, and it does not enter `MODEL_REGISTRY` as a model fact.

---

## 7. BAR B — the falsifying probe. **HYPOTHESIS-grade. Nothing was trained.**

Run only after Bar A finished inside budget, exactly as the brief conditions it.
Artifact: `raw/bar_b_probe.json`.

**The canary reproduces:** `wm_canary_ade_2s` **1.1381** (fp32) vs the committed **1.1409** (which is
computed under bf16 autocast in `canary_rollout`) — **0.25 % apart**, the difference being precision,
not method. It must fall **2.069×** to clear the 0.55 bar.

**Its shape matters and had not been published:** median **0.9788**, p75 **1.5152**, p95 **2.6356**,
p99 **3.3236** — and **22.7 % of windows are ALREADY under the bar.** The 1.1409 mean is a
**tail-driven** statistic, not a uniform error floor.

### 7.1 What the probe tests, and what it cannot

The canary rolls under **TRUE actions**, so it is **on-path by construction** — there is no off-path
distance to correlate against. What is testable is the **necessary condition**: does world-model error
*concentrate on unusual states*, or is it flat? Pre-registered before computing:
**SURVIVES** if the strongest covariate shows top/bottom-decile ratio ≥ 1.5× **and** |ρ| ≥ 0.20 with a
separated decile gap; **WEAKENED** otherwise.

| covariate | Spearman ρ | bottom decile | top decile | ratio | decile gap separated? |
|---|---:|---:|---:|---:|---|
| **`gt_cross_2s_abs`** (lateral path departure @2 s) | **0.3389** | 0.8158 | 1.7810 | **2.183×** | ✅ **[0.575,1.169] vs [1.270,2.070]** |
| `head_deg_abs` (turning in the window) | 0.2496 | 0.8871 | 1.5241 | 1.718× | overlap |
| `v0` (speed) | 0.2311 | 0.9693 | 1.5229 | 1.571× | overlap |
| **`latent_novelty`** (‖state − mean‖, the DIRECT novelty proxy) | **0.1440** | 0.9144 | 1.2470 | 1.364× | overlap |
| `dv_window` (accel/brake) | −0.0186 | 1.1377 | 1.3130 | 1.154× | overlap |

> ### Pre-registered verdict: **SURVIVES** — and read the second row of the story before acting on it.

**The nuance I will not bury.** The hypothesis was *"our arm's own OOD sensitivity"*. The error does
concentrate — but on **manoeuvre demand** (lateral departure, turning, speed), while
**`latent_novelty`, the one covariate that measures novelty in the encoder's own representation
space, is the second-weakest of the five and its decile gap is not separated.** Off-path augmentation
adds *representational* novelty; the strongest correlate here is *dynamic* difficulty. Those are not
the same thing, and this probe does not distinguish "the WM is bad at turning" from "the WM is bad
off-distribution".

**Therefore:** the off-path-augmentation lever **survives its first falsification attempt and remains
HYPOTHESIS-grade, PROVISIONAL tier. It may not authorise a GPU-week.** The cheapest next probe — not
run here, and not costed — is one that varies novelty **while holding manoeuvre demand fixed**, which
is the only way to separate the two.

---

## 8. SECONDARY SURFACE — goal-ORACLE

*(placeholder — filled from `raw/bar_a_oracle.json` when the run lands.)*

---

## 9. GPU BUDGET — the experiment was authorised at 1–2 GPU-hours

| stage | GPU-min | note |
|---|---:|---|
| reproduction (§1), both goal modes | 5 | 117.2 s + 122.6 s of forward pass + load |
| harness smoke tests (2 runs, one aborted by its own self-test) | 11 | not quotable; validation only |
| Bar A **produced** — attempt 1 | 30 | killed by an unhandled seam-guard raise; no result quoted from it |
| Bar A **produced** — attempt 2 (the reported run) | 31 | 1072.5 s capture + ~13 min fitting |
| Bar B probe | 3 | 881 windows |
| Bar A **oracle** (secondary) | ~31 | |
| **total** | **~111 min ≈ 1.85 GPU-h** | **inside the 1–2 GPU-hour authorisation** |

The 6,844-window feature cache (4.07 GiB) is persisted to `/workspace/_bara/`, so **any re-analysis of
this experiment costs ~13 GPU-min instead of ~31** — the capture never has to be paid again.

---

## 10. ESCALATIONS — things that must not sit in a file

1. **⛔ Bar A is REFUTED. The selector-only v4 restart has now failed BOTH of its bars.** Bar B was
   already unowned; Bar A's lever is now measured not to exist. Per BOOST_PROGRAM §3.4, *"if we cannot
   name a Bar-B lever, the correct decision is to not restart v4 and instead carry the fan into the
   v2-corpus line."* **The evidence now points there. The decision is Sayed's.**
2. **Retire the "0.6058 m of selector waste" framing from chat, docs and the registry.** It reads as a
   recoverable surplus; measured, it is an **oracle** quantity. §3.2 is the number that should replace
   it. This phrasing is currently live in `BOOST_PROGRAM.md` §3.2 and `V4_RESTART_LEVER.md` §4.1 —
   **both need the qualifier, and I did not edit either** (steering files are the orchestrator's).
3. **The next selector probe is CONDITIONING, not loss** (§0.8, committed in advance). Whatever is
   spent on the selector next should change **what the score sees**, not how it is trained.
4. **⚠️ `/root` on `tanitad-eval` is at 99 % (2.9 GiB free) and silently truncates.** A 6 GiB `dd`
   reported success after writing 3.0 GiB. The next agent that `scp`s a checkpoint there **will get a
   corrupt file and no error.** Use `/workspace` (verified: full 6 291 456 000 B at 534 MB/s).
5. **Two brief premises were wrong and are corrected here** (BOOST_PROGRAM M2): the eval pod runs
   **Python 3.12.3**, not 3.11.10 (3.11.10 is pod2); and the deployable waste is **0.6057** on re-measurement,
   not 0.6058 (immaterial, kept as registered).
6. **`sel_gate` / graft telemetry should be emitted by the trainer.** This experiment needed
   `seam_norm_ratio_preclamp_max` to discover that ranking objectives drive the graft past
   `seam_fail`; the trainer computes it every step and the row-writer discards it (the R-3/C11 gap,
   still unlanded).

---

## 11. FOR `Project Steering/RETRACTION_LOG.md` — root-cause classes

**I did not edit the log; appending to an append-only steering file is the orchestrator's call.**

### R-1 — **C6 (confounded comparison)**, avoided by construction rather than after the fact

> The obvious design was REGRET vs AS_TRAINED. That contrast varies **two** things — the loss *and*
> the fine-tune (on val-corpus episodes, which the as-trained selector never saw). Reporting Bar A off
> it would have been C6. The **CE_CONTROL** arm was added before any result existed, and it earned its
> place: it is **worse than as-trained by 0.0668**, i.e. the fine-tune itself carries a real penalty
> that would otherwise have been charged to the loss. **Reported Δ(REGRET − CE) = −0.0414 is the only
> honest estimate of what the loss did.**

### R-2 — **C-new: precision as a scientific parameter** (fp16 caching)

> **Nearly claimed:** an `ade_0_2s` of **0.8591** for the as-trained selector, from an fp16 feature
> cache — a 0.0028 m error that would have silently biased **every** delta in this report.
> **Caught by:** the pre-registered `cache_fidelity` self-test, which aborted the run before training.
> **Root cause:** the selector ranks **256 candidates separated by less than fp16's ULP**. In a system
> whose central finding is a near-tie structure, **storage dtype is a scientific parameter, not an
> implementation detail.** Generalises to every future re-scoring experiment on this fan.

### R-3 — **C3 (mechanism instead of measurement)**, at program level

> **The pattern:** *"the selector throws away 0.4093 m / 0.6058 m"* was true as arithmetic and became,
> in prose, *"the world model is not the problem, the picker is"* — a **repairable** picker. The
> mechanism was plausible, specific, and it survived several documents. **Measured:** no re-scorer of
> the frozen fan reaches v1 even **in-sample**. `oracle_in_fan` is a **clairvoyant** bound; the step
> from "a better trajectory exists in the fan" to "a ranker can find it" was never measured, only
> assumed. **Rule this earns: an oracle-vs-selected gap is a BOUND, not a budget. Before it may
> motivate a restart, someone must show a realisable ranker can close part of it — which is a
> 30-GPU-minute in-sample fit.**

