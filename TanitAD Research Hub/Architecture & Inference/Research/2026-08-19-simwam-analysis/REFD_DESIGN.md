# REF-D — design, training plan, and what it is pre-registered to prove

`PARKED, TRAINING-READY.` Implemented at `stack/tanitad/refs/refd.py`, 19 tests
green. **Not a replacement for anything** — REF-A v1, v1′ and v6 are untouched.

**TIER.** This document describes MECHANISM. No number here is a capability
claim; those come from `taniteval` at **T1**.

---

## 1. The thesis in one paragraph

SimWAM (2608.07468) reaches **91.5 PDMS** on NAVSIM with **6 B parameters on
32 GPUs**. Its central trick — an **isolated attention mask** that makes
future-video prediction a *training* signal and deletes the video expert at
inference — has an economic consequence worth more to us than its score:

> ⭐ **The prior's size is a TRAINING cost, not a DEPLOYMENT cost. A sub-300 M
> rule forbids SHIPPING a large prior; it does not forbid USING one.**

REF-D spends that on our hierarchy instead of on scale.

| | |
|---|---|
| **frozen prior** | Cosmos3-Edge **4 B**, pre-extracted, training-time only, **never shipped** |
| **ships** | adapter + 3-rate action-token hierarchy + flow policy = **172,539,422 (172.54 M)** |
| **vs SimWAM** | **35× smaller** at inference |

---

## 2. Architecture

```
 frozen Cosmos3-Edge field  [B, W, N, d_enc]      (pre-extracted, on disk)
            |
        WideAdapter                                (temporal mixing over W)
            |
   +--------+--------+--------------------+
   |                 |                    |
 operative        tactical            strategic     3 rates, each -> exactly 6.0 s
 (0.2 x 30)       (0.6 x 10)          (1.5 x 4)     ACTION AS TOKENS in-stream
   |                 |                    |
   +--------+--------+--------------------+
            |
     FlowControlPolicy  ->  ONE control sequence [B, 2, 60]
                            (a, kappa), squashed, integrated
```

### 2.1 Every choice, and the measurement behind it

| choice | source | evidence |
|---|---|---|
| isolated mask | SimWAM Tab. 3 | isolated **90.3** vs bidirectional 90.2 vs action→video 90.1 |
| ⭐ **action as TOKENS** | **ours, E-ACTSTREAM-1** | token beats concat **5.9×**, add **9.9×**, at parameter parity, separated at 3 widths / 2 horizons / 2 targets / 3 seeds |
| **controls, never waypoints** | ours, H1 + v5f | per-waypoint head amplified ε **25×** in acceleration; v5f dense fan **97.6 % infeasible steps / 100 % infeasible candidates** |
| OU-correlated noise | ours, F-15 | white noise on 60 steps integrates to near-cancelling jitter |
| feasible by construction | ours, W4 | every sample squashed + integrated through `unicycle_rollout` |
| shared goal vocabulary | ours, HIERARCHY_VOCABULARY §5 | `head.vocab is cond.vocab` — one table, two views |
| multi-horizon supervision | SimWAM Tab. 8 + ours | 4 s/1 Hz 90.2 ≈ 4 s/2 Hz 90.3 ≫ 2 s/2 Hz 89.9 — **coverage beats density** |
| 10 flow steps | SimWAM Tab. 10 | 1 step collapses (68.9), 5 → 90.1, 10 → 90.3, 20 → 90.2 |

### 2.2 ⛔ The load-bearing change: the policy is a generator, not a fan

v6 diffuses a **fan** and a **selector** ranks it.
`assert_selector_admissible` **refuses every selector launch** while SEL-1
stands refused (E-WC2: σ/ADE **9.9915 [7.4492, 13.5119]** against the 3.0 line).
The fan design created a selection problem that became the programme's blocker.

**SimWAM has no selector because its flow model IS the policy.** REF-D adopts
that: sample once, integrate, done.

⇒ **This does not repair the selector. It removes the need for one** — a far
cheaper route past SEL-1 than fixing it, and it costs us nothing we measured as
valuable, because the fan's value was always conditional on a selector that does
not work.

### 2.3 ⭐ The multi-horizon extension SimWAM cannot express

SimWAM has **one** action group and **one** future horizon. Our hierarchy gives
each layer its own target at its own rate — operative 0.2 s × 30, tactical
0.6 s × 10, strategic 1.5 s × 4, all reaching exactly 6.0 s, and **all invisible
at inference behind the same mask**.

Nobody has tested whether different layers of a hierarchy want different future
horizons. Our four metric families are the instrument that could attribute a
gain to a specific layer, which is what makes the question answerable here and
not elsewhere.

### 2.4 The isolated mask as a CHECK, not a convention

v6's imagination discipline lives in **defaults** (`mpc_w_consist=0.0`,
`fallback_trigger=False`) — and a default is not an invariant.
`RefD.assert_no_future_at_inference()` walks the deployment signature and
refuses any parameter that could carry a future field or a rollout.

⚠️ A test proves the guard **fires** on a mis-wired path. A guard that cannot
fail is not a guard (the C13 class).

⚠️ **Past frames are not future frames.** `act` takes `[B, W, N, d]` — a causal
window of *observed* history, which `WideAdapter` requires. It refuses a 3-D
input rather than silently dropping the temporal mixing.

---

## 3. Two declared tensions, neither resolved by assertion

### 3.1 OU noise vs Flow-GRPO

Rectified flow assumes an **isotropic Gaussian** base. Our OU-correlated noise
is not isotropic. Plain flow matching tolerates this (it is a different
coupling), but **Flow-GRPO's tractable transition likelihoods assume isotropic
steps** — and that machinery is what SimWAM's **+1.2 PDMS** of RL depends on.

⇒ `noise` is a **declared config axis** with both arms built and tested, and the
correlation property is **measured** (lag-1 > 0.7 for OU, < 0.1 for iso), never
asserted from the formula. **E-REFD-2** below settles it.

### 3.2 Frozen vs co-trained prior

SimWAM **co-trains** its video expert — that is what shapes `z(o_t)`, and what
costs 32 GPUs. REF-D **freezes** and pre-extracts, so the shaping must happen in
**our adapter**, with Cosmos features as the prediction *target*.

⚠️ **This is a genuinely different bet, and it is stated as one.** It is DINO-WM's
recipe with a driving-pretrained encoder. Their Tab. 2 (action-only 86.6 →
+video 90.3) measures co-training, not frozen-prior transfer, so **+3.7 PDMS
does not transfer to this design by assumption**.

---

## 4. ⛔ RL: designed in, not reachable, and the blocker is not compute

SimWAM's RL is worth **+1.2 PDMS** (90.3 → 91.5): Flow-GRPO, ODE→SDE, LoRA
rank 32 on the action expert only, G=8, **hard subset only** (scenes with
imitation PDMS < 90), peaking at 15k steps and **declining after**.

**Its reward is the NAVSIM PDM score. This repo has zero `import navsim`.**

⇒ The order is forced: **benchmark → reward → RL.** `rl_ready` ships the hooks;
the loop is deliberately absent.

⚠️ Building a reward from our own four metric families instead would mean
optimising the quantity we also report — the "scoring a loop" failure the
goal/situation-disjointness rule exists to prevent.

Two of their RL findings are worth carrying whenever we get there: **train on
the hard subset only**, and **stop at the peak** — both curves declined past 15k.

---

## 5. Training plan, and whether our hardware can do it

**⭐ Freezing the prior turns the expensive part into a one-off.** MEASURED
2026-08-19 on the analogous job: DINOv3 ViT-L (303 M) over 5,617 frames at
640×1024 in **~3 min on an RTX 4060**.

| stage | estimate | where |
|---|---|---|
| Cosmos3-Edge feature extraction, full 2,376-episode corpus (~95 k frames) | **~1 h** | one A40 (4 B ≈ 13× ViT-L; A40 ≫ 4060) |
| ⚠️ feature storage at 640×1024 fp16 | ⛔ **~125 GB** | **the real constraint** |
| training the 173 M stack on cached features | comfortable | one A40; Thor slower but able |

⛔ **The binding constraint is DISK, not FLOPs.** Mitigations, in order of
preference: pool 640→160 tokens (**−4× → ~31 GB**), subsample frames, or extract
per-shard on the fly.

⚠️ **Not Thor for extraction while it trains.** Thor is 3.3 days from finishing
v6F S-W at 26.44 s/step and the no-load rule applies. Training on cached
features is light enough for Thor *after* 30k.

⇒ **One A40 suffices.** That is only true because the mask makes the prior
training-time-only; co-training a 4 B prior the way SimWAM co-trains 5 B would
put us back on a cluster.

---

## 6. Pre-registered experiments

### E-REFD-1 — does the frozen-prior bet hold?
Arms: `action-only` · `+future-supervision on frozen Cosmos features`.
Their Tab. 2 measured **+3.7 PDMS** for co-training; this measures whether a
frozen prior with the objective in the adapter recovers any of it.
* **recovers most** → the cheap design stands and REF-D is on the right bet.
* **recovers little** → the gain was co-training, and REF-D must either LoRA the
  prior or be reported as a negative result about frozen transfer.

### E-REFD-2 — OU vs isotropic noise
Arms: `noise=ou` · `noise=iso`, identical otherwise, paired episode-cluster
bootstrap, all four metric families.
* **OU ≥ iso** → keep OU and accept that Flow-GRPO needs a separate treatment.
* **iso ≥ OU** → adopt iso, and the RL path opens with no further conflict.
⚠️ Must be run **before** any RL work, or the RL result is uninterpretable.

### E-REFD-3 — the multi-horizon claim
Arms: single-horizon (6 s) · per-layer horizons (2/6/longer).
Read **per family**, never pooled — the whole point is attributing a gain to a
layer.

---

## 7. What REF-D is NOT

* ⛔ **Not a replacement.** REF-A v1, v1′ and v6 are untouched.
* ⛔ **Not evidence that action tokens beat broadcast at real geometry.**
  E-ACTSTREAM-1 ran on v6 cell fields (16 × 128). The DINOv3 transfer test at
  640 × 1024 is running; until it reports, the magnitude does not transfer.
* ⛔ **Not a claim that Cosmos beats Wan.** SimWAM Tab. 4 is **90.4 vs 90.3 with
  no CI** — a tie. The 1.7-point spread often quoted is LTX (88.7) → Cosmos
  (90.4), contrasting a weak backbone with a strong one. We choose Edge because
  it **runs on Thor**.
* ⛔ **Not comparable to SimWAM's 91.5** until NAVSIM exists here.

---

## 8. Manifest

| artifact | where |
|---|---|
| `refd.py` (RefDConfig, FlowControlPolicy, RefD) | `stack/tanitad/refs/` |
| `test_refd.py` (19 tests) | `stack/tests/` |
| this document | `…/2026-08-19-simwam-analysis/REFD_DESIGN.md` |
| paper §12 | `Paper/TANITAD_PAPER.md` |
