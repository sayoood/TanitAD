# WP-L — the trunk's VRU deficit is NOT spatial, and it is only partly linear

**TanitAD_TrainingFlyWheel · 2026-09-06 · Tier T0, NON-PARITY pilot.**
Evidence class **MEASURED (ours)**.
`n = 1,954` windows · 33 episodes · verified join (`pilot_val_agents_ext.jsonl`) ·
86 NO_LABEL frames masked · LOEO · 200 permutations · γ and λ fitted inside each fold only.
Code `code/wpl_trunk_pooling_kernel.py` · raw `raw/wpl_trunk_pooling_kernel.json`,
`raw/wpl_full.log`. Follows `RESULT_WPI_SEMANTIC_FLOOR.md` §9.

---

## 0. Why this ran before any distillation head

WP-I v3 left REF-C's trunk at **p_w 0.110** on the near VRU band — close to the line, and
measured through **two lossy choices that are mine, not the model's**: a 2×4 average pool
of a [704, 8, 8] map, and a **linear** ridge. `CLAUDE.md` is explicit that *a negative from
a linear probe is not a negative about learnability*, and WP-J already proved this
programme can mistake a pooling choice for a representational fact.

**Arms** — identical windows, targets, folds and nulls; only the read changes:

| arm | d | what it tests |
|---|---:|---|
| `trunk 2×4 linear` | 5,632 | the WP-I v3 anchor, reproduced |
| `trunk 8×8 linear` | 45,056 | **the full feature map — zero spatial pooling loss** |
| `trunk 2×4 rbf` | 5,632 | nonlinearity at the anchor's resolution |
| `trunk 8×8 rbf` | 45,056 | both levers together |
| `dinov3 2×4 linear` | 24,576 | the reference that cleared in v3 |

⭐ The RBF arms cost nothing to validate: the panel already solves ridge in the **dual
(Gram) form**, and a dual solve does not care whether the Gram came from a dot product or
a kernel — same folds, same λ sweep, same permutation nulls, no second estimator.

---

## 1. Controls

| control | reads | |
|---|---|---|
| CONSTANT-only | **0.5000** exactly | ✅ |
| **PC-TRUNK, per arm** | 2×4 lin **0.9232** · 2×4 rbf **0.9585** · 8×8 lin **0.9348** · 8×8 rbf **0.8596** · dinov3 0.6223 | ✅ **every trunk arm readable** |

⭐ **The per-arm positive control earned its place.** In the smoke run at n=98 it flagged
`trunk 8×8 rbf` at **0.5166** — a single broken arm, named, without voiding the panel. At
full n the same arm reads 0.8596 and is readable. A panel-level control could only have
said "something is wrong"; this one said *which arm* and then said *when it recovered*.

---

## 2. The result

`adj` = AUC − that arm's own within-episode null mean. `p_w` = within-episode permutation p.

| arm | `vru<20m IN-FOV` adj (p_w) | `vru<60m IN-FOV` adj (p_w) |
|---|---|---|
| `trunk 2×4 linear` | +0.0792 (0.155) | +0.0327 (0.185) |
| `trunk 8×8 linear` | +0.0045 (**0.495**) | +0.0078 (0.420) |
| `trunk 2×4 rbf` | +0.0374 (0.305) | **+0.0893 (0.040)** |
| `trunk 8×8 rbf` | +0.0558 (0.235) | **+0.0585 (0.020)** |
| `dinov3 2×4 linear` | **+0.1935 (0.000)** | +0.0501 (0.025) |

### ⛔ 2a. Spatial pooling is NOT the bottleneck — and the full map is WORSE

`trunk 8×8 linear` uses the **entire** [704, 8, 8] map with no pooling loss whatsoever, and
it is the **weakest** arm in the panel on both targets: adj +0.0045 (p 0.495) near,
+0.0078 (p 0.420) far — essentially its own chance level. Against the 2×4 anchor it *loses*
0.075 near.

⇒ **The WP-J hypothesis does not transfer here.** There, unfair pooling was the defect;
here, more spatial detail is strictly unhelpful. 8× more dimensions on the same ~1,900 rows
buys variance, not signal — and the null-adjustment is what makes that visible rather than
letting the extra width flatter the arm.

### ⭐ 2b. On the FAR band a nonlinear read DOES rescue the trunk

`trunk 2×4 rbf` **p 0.040** and `trunk 8×8 rbf` **p 0.020** clear their own within-episode
nulls on `vru<60m`, where **both linear arms fail** (0.185, 0.420). Null-adjusted, the RBF
trunk (**+0.0893**) is *above* a linear read of DINOv3 (**+0.0501**) on that target.

⇒ **the trunk does carry far-VRU information that a linear probe cannot see.** This is the
one-way implication of `CLAUDE.md`'s linear-probe rule paying out in the direction that
matters: the earlier linear negative was **not** a statement about the representation.

### ⛔ 2c. On the NEAR band nothing rescues it

No trunk arm comes closer than **p 0.155**; the full map reads 0.495. DINOv3 clears at
**p 0.000** with more than double any trunk arm's null-adjusted margin (+0.1935).

⇒ **near-agent presence is genuinely absent from the trunk**, in any of the four reads
tried, and the near band is the safety-critical one.

---

## 3. ⛔ CORRECTION — the script's own printed verdict is wrong

The run printed:

> `NOT RESCUED — neither a finer grid nor a nonlinear kernel brings the trunk across`

**That sentence is a defect in my verdict logic, not a finding.** `rescued` was computed on
`vru<20m IN-FOV` **alone** and then announced globally, so it asserted a negative while the
far band's two RBF arms were clearing at p 0.040 and 0.020 in the same table.

⚠️ It is the same shape as pooling a consistency score across metric families: **a verdict
that aggregates over targets hides exactly the structure the panel was built to expose.**
Fixed in code — the verdict is now emitted per target and carries its multiplicity.

**Multiplicity, stated rather than left to the reader:** 5 arms × 2 VRU targets = **10 tests
at α = 0.05**, so ≈0.5 false positives are expected by construction, and with 200
permutations the p resolution is 0.005. **p = 0.040 is a lead, not a result**; p = 0.020 is
firmer but still inside a family of ten. The near-band negative (best p 0.155, worst 0.495)
is not multiplicity-sensitive in the same way, because it is a *failure* to reject.

---

## 4. What this decides

| question | answer |
|---|---|
| Is the deficit **spatial**? | ⛔ **No.** The full 8×8 map is the weakest arm on both targets. Pooling is not what is losing the agents. |
| Is it **linear-probe artefact**? | ⚠️ **Partly.** On the far band a nonlinear read clears where linear does not. On the near band nothing does. |
| Is **distillation** the right lever? | ✅ **For the near band, yes** — and that is the safety-critical one. It is *not* justified as a fix for far-VRU, where the trunk already holds the information nonlinearly. |
| Does the reasoner get anything free? | ⭐ **Yes, on the far band.** The recursive core's own MLP is a nonlinear read of `cond`, so far-agent presence should be reachable without a teacher. |

⇒ **`DIALOGUE_07` §7(a) is narrowed, not withdrawn:** distillation targets **near-agent
semantics**; the far band is an MLP-reachable property of the existing trunk. That is a
smaller and better-aimed head than the one the v2 result would have justified.

---

## 5. Scope and what would overturn it

* **T0, non-parity pilot, one checkpoint (`refc-base-30k`), one inference seed.** No metric
  family, no driving claim.
* **The trunk is still read as a POOLED map** — 2×4 and 8×8 are both average pools over the
  final feature map. A *different layer* of the encoder, or attention-weighted pooling, is
  untested; the honest statement remains about **this** readout of **this** layer.
* **The RBF is one nonlinear family.** Its γ was swept inside each fold over three values;
  a different kernel or a trained MLP could differ. A clean overturn would be an MLP probe
  that clears the **near** band on the trunk.
* ⚠️ **Two arms cleared at p ≤ 0.04 within a family of ten.** The far-band rescue should be
  replicated — ideally with a second inference seed and more episodes — before it is quoted
  as settled. Registered as a lead, not a result.
