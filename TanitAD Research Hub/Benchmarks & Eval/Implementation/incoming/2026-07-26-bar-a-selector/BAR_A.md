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
