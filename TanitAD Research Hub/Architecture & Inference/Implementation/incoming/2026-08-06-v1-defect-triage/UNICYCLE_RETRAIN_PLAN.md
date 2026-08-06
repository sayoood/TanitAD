# Retraining the trajectory head on the unicycle — what is built, and what it costs

**Asked by Sayed, 2026-08-06:** *"Can we retrain the trajectory head based on unicycle? We need
to improve the heading regress."*

**Short answer: yes, and the loss half is already runnable today.** The head half is built and
tested but not yet wired. ⛔ **No training run has been launched** — `train_flagship_v4.train`'s
own docstring says a launch is *"Sayed's go (§17), executed by the orchestrator, NOT from an
agent"*, and that is respected here.

---

## 0. Why the heading is bad — the root cause, and it is one line

```python
loss_traj = (recon - traj_tgt).abs().mean()      # flagship_v15.py, v15_losses
```

**Pure position L1.** Heading, curvature, acceleration and jerk are **not in the loss at all**,
so none of them is being learned. Every measured defect follows directly:

| | flagship v1 | human | |
|---|---|---|---|
| accel RMS | 3.8166 | 0.9075 | 4.21× |
| jerk RMS | 52.1281 | 1.7051 | 30.6× |
| net-yaw error over 2 s | — | — | never scored until 2026-08-06 |

⭐ **PINNED BY TEST** (`test_position_l1_is_blind_to_heading`): two paths that agree in position
to **0.0100 m — one centimetre** differ in heading by **2.23°** at every step. The trained loss
sees the centimetre and is blind to the degrees. *A term that is not in the loss is not being
learned* — this is not a tuning problem, it is a missing objective.

---

## 1. What is BUILT AND RUNNABLE today — the loss terms

`tanitad.models.kinematic.kinematic_losses` → wired into `v15_losses` → exposed on the trainer:

```
--kin-heading 0.3     per-step heading error (rad)
--kin-net-yaw 0.2     net yaw over the window   <-- THE heading target
--kin-accel   0.05    barrier above the human p99 (2.689 m/s^2)
--kin-jerk    0.05    barrier above the human p99 (6.369 m/s^3)
```

⭐ **`net_yaw` is the right target for "improve the heading", not `heading`.** It is
**sampling-independent** — MEASURED 2026-08-06, it is the quantity that regressed **62 %** under
inference-time re-timing *while cross-track improved 20 %*, i.e. it is the one that caught a real
lateral degradation that every position metric missed. `heading` is the denser per-step signal;
use both, weight `net_yaw` for the outcome we actually care about.

⛔ **Accel and jerk are BARRIERS, not shrinkage** — only the excess above the human's p99 is
penalised. A plain `λ·jerk²` would also punish legitimate emergency braking, i.e. train the arm
to be smooth exactly when it should be decisive.

⛔ **All four default to 0.0, and an unflagged run is BIT-IDENTICAL to v1arch.** Verified:
`v15_losses(out, anchors, tgt)` returns the same scalar before and after this change. Parity is
sacred — a silent loss change would invalidate every cross-arm number in `MODEL_REGISTRY.md`. A
run that uses them prints a banner saying the arm is **not loss-comparable** to those that did not.

**Status:** `--smoke` green, `tests/test_flagship_v15.py` green, 35 kinematic tests green.

## 2. What is BUILT BUT NOT WIRED — the unicycle head

`kinematic.unicycle_decode(base, delta, v0)` — compose the head's correction in **control space**:

```
controls = controls_of(anchor) + delta        # delta read as (d_accel, d_curvature)
traj     = rollout_unicycle(state0(v0), controls)
```

⭐ **This is the right place to cut in, and it is a small cut.** The anchor **vocabulary is
already kinematically feasible** — `fourbrain._synth_anchor_pool` builds it from random unicycle
rollouts. It is the free-form per-waypoint **offset** that destroys feasibility: nothing stops it
moving waypoint *k* by +2 m and *k+1* by −2 m, a 400 m/s³ jerk the position loss never sees.
Composing in control space makes every output feasible **by construction** and changes the
*decode*, not the diffusion machinery, the anchors, the selection head or the other losses.

⇒ And it fixes heading **structurally**: heading is `yaw += v·κ·dt`, so the head emits the
quantity heading is *made of* instead of positions from which heading is a fragile by-product.

⛔ **A trap found and fixed while building it.** The first squash was plain softsign, which is
**not the identity inside its range**: a curvature of 0.04 against the 0.33 limit came back as
0.0357 — an 11 % shrink on a control nowhere near the bound. Through a decode that must reproduce
its own anchor that cost **0.594 m**, making the entire anchor vocabulary unreachable; the head
would have had to learn to undo the squash before it could learn anything. Replaced with an
identity-below-0.9·limit map with a C¹ rational tail. Pinned by
`test_zero_delta_reproduces_the_anchor`.

**Remaining work to wire it:** `V15Decoder.forward` must emit its offset as controls and call
`unicycle_decode` instead of adding positions, gated on a `V15Config` flag so existing arms are
untouched. ~30 lines. **Not done — do not assume it is.**

---

## 3. The two runs, and what each can and cannot answer

| | **A — cheap probe** | **B — the real experiment** |
|---|---|---|
| what | fine-tune from `flagship-v1arch-v2bal-30k` @29999 with the kinematic terms on | train from the same init as v1arch, identical in every way except the loss terms |
| steps | ~2,000–3,000 | 30,000 (matched to v1arch) |
| answers | *do the terms move accel / jerk / net-yaw in the right direction, and what does ADE pay?* | *is the resulting arm better?* |
| ⛔ cannot answer | whether the final arm is better — a fine-tuned arm is **not** step-matched to a from-scratch one and must never be entered in the registry against it | — |
| cost | ~1/10 of B | a full training run |

**Recommendation: run A first.** It is cheap, it runs on the idle A40, and it discriminates: if
2,000 steps do not move `net_yaw` and `jerk` toward the human, the terms are mis-weighted and B
would burn a full run learning that.

### Pre-registration for B — both outcomes committed in advance

Read through `taniteval.four_families` **and** `tools/temporal_stability.py`. ⛔ **Never through
ADE**, which is blind to every quantity at issue.

* **PASS** — `net_yaw` error ≤ **0.12 rad** (i.e. at or below raw v1arch, so the heading is *not*
  traded away) **AND** jerk RMS ≤ **3×** human (≤ 5.1 m/s³) **AND** accel RMS ≤ **2×** human
  (≤ 1.8 m/s²) **AND** `speed_bias` < **+0.15 m/s** **AND** cross-track does not regress.
* **PARTIAL** — kinematics hit target but ADE regresses > 5 % ⇒ the arm drives more feasibly and
  predicts the *human* less well. That is a real trade, not a failure, and it is the PI's call
  whether feasibility is worth it. Say so; do not bury it.
* **FAIL** — `net_yaw` worse than raw v1arch, or jerk still > 10× human ⇒ the loss terms are not
  the lever and the defect is in the architecture or the corpus.

⚠️ **The comparison must be step-matched and read with the episode-cluster bootstrap** over the
40 val episodes — `tools/retime_ab.py` already banks `per_episode[].delta` for exactly this.

### What the inference-time projection already achieves, as the baseline to beat

`retime_path` on the **frozen** v1arch, no retraining (MEASURED, 6,834 windows): jerk
**30.6× → 2.90×** human, speed bias **−91 %**, launch transient **−99.7 %**, frame-to-frame
control jump **−61 %**, ADE **−10.6 %**, cross-track **−20 %** — but **net-yaw error +62 %**.

⇒ **That +62 % is precisely what this retrain exists to beat.** A projection can only re-time a
curve it did not choose; a trained head chooses the curve, and with `net_yaw` in its objective it
can choose one whose heading is right. **If the retrained arm does not beat +62 %, the retrain
did not do its job** — that is the sharpest single criterion available and it is why the
inference-time result had to be measured first.

---

## 4. ⛔ Waiting on the PI

The launch is Sayed's, per `train_flagship_v4.train`'s own docstring. What is needed:

1. **Go / no-go on run A** (the cheap probe) and on which pod.
2. Confirmation that the idle A40 (`pod4`) is free for it — v5f is training on the other pod and
   ⛔ **must not** be touched.
3. Whether to wire the unicycle head (§2) before or after run A. **Recommendation: after** —
   run A tests the loss terms alone, so if it fails the head change is not confounded with them.

**Artifacts.** `stack/tanitad/models/kinematic.py` (`kinematic_losses`, `unicycle_decode`,
`retime_path`, `rollout_unicycle`, `_squash`) · `stack/tests/test_kinematic_losses.py` ·
`stack/tanitad/models/flagship_v15.py` (`v15_losses`, zero-default weights) ·
`stack/scripts/train_flagship_v4.py` (`--kin-*`).
