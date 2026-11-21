# E-TRUNK-1 — the v6 READOUT loses the dynamics, not the v6 ENCODER

> ⛔ **SUPERSEDED IN ITS ARCHITECTURAL READING — 2026-08-20.** The title claim
> and §3/§5's parallel-token-path direction **do not survive E-TRUNK-2**
> (`E_TRUNK_2_ENV_DECODABILITY.md`). A decodability probe with a DINOv3
> reference shows the readout's 40× pool costs DINOv3 only **16 %** of its
> headway R² (+0.4549 → +0.3841) and **0.016** AUC of its lane occupancy
> (.846 → .830), while **every** v6 arm sits at chance on both sides of the
> pool. **The pool is not where the information dies; the encoder is.**
> The *measurements* below stand. The inference from them to "fix the readout"
> does not, and the 40×-sequence token path must not be funded on it.

`MEASURED (ours; dev-box RTX 4060)` · **T0-DIAGNOSTIC — a future-field prediction
error is a world-model fidelity number, never driving performance** · one
checkpoint, one encoder, two granularities · 3 seeds · paired episode-cluster
bootstrap · **no load added to Thor**.

---

## 0. The confound this exists to remove

E-ACTSTREAM-2 compared **DINOv3 patch fields (640 × 1024)** against **v6 cell
fields (16 × 128)** and found the DINOv3 arms beat C-PERSIST while the v6 arms
never did. ⛔ That comparison moves **two** variables at once — the **encoder**
and the **granularity** — so it cannot say which is responsible. Reading it as
*"the v6 trunk carries no dynamics"* would be exactly the scope error this
programme keeps retracting.

⭐ `cache_tok11250` banks **both** representations for the **same frames** from
the **same checkpoint**, so the encoder is held fixed and only granularity moves:

| arm | what it is |
|---|---|
| `cells` | **the operative latent itself** — 16 × 128 = **2048 = `d_op` exactly**, not a lossy summary of it |
| `tokens` | the **pre-readout** encoder field, 640 × 768 |

Each arm is scored against **its own** C-PERSIST, because the two have different
scales and only skill-over-persistence is comparable across them.

## 1. Result (stride-8 cache, 1,509 windows, 5.6 s horizon)

| arm | MSE (3 seeds) | C-PERSIST | paired Δ vs persistence | verdict |
|---|---|---|---|---|
| **`cells`** | 0.000037 | 0.000014 | **+0.000023 [+0.000018, +0.000029]** | ⛔ **LOSES, SEPARATED** |
| **`tokens`** | 0.002744 | 0.002806 | **−0.000073 [−0.000322, +0.000213]** | beats on **3/3 seeds**, **not separated** |

⇒ **The pooled operative latent demonstrably does not support dynamics
prediction.** The same encoder's unpooled field is **not demonstrably worse than
persistence** — a different and much better position.

⚠️ **The tokens arm is NOT claimed as a win.** Its interval spans zero. What is
separated is the `cells` failure. Recommending an architecture change on a
not-separated result would be premature, which is why the proper-scale run
below exists.

## 2. ⭐ What the readout actually does — and it is mostly parameter-free

```python
x = pool(tokens)                 # 16x40 grid -> 4x4 = 16 cells   <- NO parameters, 40x spatial loss
x = x.flatten(2).transpose(1, 2) # [B, 16, 768]
return self.proj(x).flatten(1)   # Linear(768 -> 128)             <- the ONLY trained part, 98,432 params
```

**640 × 768 = 491,520 values → 2,048. A 240× compression**, of which the **40×
spatial reduction is a plain average pool with no parameters at all.**

⇒ The information is not being *transformed* away by a learned bottleneck; it is
being *averaged* away by a fixed one. That is consistent with the operative
latent's measured **4.5× between/within-episode variance ratio** — averaging 640
tokens into 16 cells preserves *where you are* and destroys *what is moving*.

## 3. Can this be fixed on v6 WITHOUT retraining?

⛔ **Not the readout in place.** Three independent blockers:

1. `V6Stack.__init__` raises on `readout.out_dim != cfg.d_op` — *"the geometry
   firewall … the single source of the state width"*.
2. `readout.py`'s own docstring: changing the geometry *"makes every downstream
   checkpoint unloadable"*.
3. `proj.weight` is `(128, 768)`. Any change to `d_readout` makes it a
   **different tensor** and the **tensor-strict resume** fails — including a
   redistribution that keeps `out_dim = 2048` (e.g. 8 × 8 × 32 → `(32, 768)`).

✅ **But the ENCODER needs no retraining**, because it is upstream and already
produces the tokens. A **parallel token-path** — predictors reading 640 × 768
directly, bypassing the pool — is **additive**: it introduces fresh keys instead
of reshaping existing ones, which is precisely what `STAGE_MAY_INTRODUCE` admits
at the **S-T boundary** (the mechanism `tac_goal_cond` already uses: default off
so the live S-W resume stays byte-identical, on at S-T).

| | |
|---|---|
| retrain the encoder / trunk | ❌ not needed |
| retrain readout + predictors, if changed in place | ✅ required |
| add a token-path in parallel at S-T | ✅ no existing weight disturbed — **but the new path must be trained** |

⚠️ **And it is not free:** 640 tokens instead of 16 cells is **40× more sequence**
into every predictor. REF-A v1's `last_only` rule exists for exactly this
(3.9 GB for a 300-candidate rollout). This is a real architecture change with a
real memory bill, not a config flip.

## 4. What this does and does not license

✅ **Licensed:** *the pooled operative latent does not support dynamics
prediction, and the loss happens at the readout rather than the encoder.*

⛔ **NOT licensed:**
* *"the v6 encoder carries dynamics"* — the tokens arm is **not separated**.
* any comparison to **v5.8**. No v5.8 arm was run. What can be said is that this
  is the test which distinguishes *"the world model is empty"* from *"the readout
  is lossy"* — the distinction never drawn for v5.8 — and that for v6 the
  evidence leans to the second.
* a redesign recommendation, until the proper-scale run separates.

## 5. Design consequence, already true

REF-A v1, v1′ and REF-D all predict in **token space** (640 × 1024), not cell
space. So this finding **supports those designs rather than challenging them** —
it is **v6's own readout** that would need revisiting.
