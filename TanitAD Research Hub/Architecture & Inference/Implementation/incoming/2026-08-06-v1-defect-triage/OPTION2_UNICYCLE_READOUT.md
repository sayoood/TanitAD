# Option 2 — the unicycle trajectory decoder on a frozen trunk

**Sayed, 2026-08-06:** *"let's start with option 2 and measure the changes hoping that they will
lead to the improvements without heading regress. What about the other confirmed improvements,
are they included in option 2 or are they already implemented in the decoder and don't require
training?"*

---

## 0. Answering the second question first: what needs training and what does not

⛔ **`retime_path` is a pure post-process on the frozen model's output. It requires NO training —
and it is NOT wired into any inference path.** It exists as a tested function; `episode_rollouts`
does not call it. Wiring it is a decision, not a build.

| improvement | **re-timing** (no training, built) | **Option 2** (trained unicycle decoder) |
|---|---|---|
| launch transient → 0 | ✅ by construction | ✅ same mechanism — v0 is the integrator's state |
| accel RMS 4.21× → 1.31× human | ✅ but **clipped** post-hoc | expected ✅ and **learned** |
| jerk 30.6× → 2.90× human | ✅ | expected ✅, and jerk becomes a **1st** difference, so trainable |
| speed bias −91 % | ✅ | ⚠️ **not automatic** — needs the loss term |
| frame-to-frame control jump −61 % | ✅ (side-effect of bounding each frame) | expected, and trainable |
| ADE −10.6 % | ✅ | unknown, could go either way |
| **net-yaw +62 %** | ❌ **this is re-timing's price** | ⭐ **exactly what Option 2 exists to avoid** |

⇒ Re-timing already buys 6 of 7 for free, but it pays with the heading. It can only *re-time a
curve it did not choose*. Option 2 chooses the curve.

## 1. What v1arch's "trajectory head" actually is

⛔ **Corrected 2026-08-06.** `flagship-v1arch-v2bal-30k` has **no anchored-diffusion head** — its
checkpoint holds `encoder, imagination, inv_dyn, predictor, readout, strategic_policy,
tactical_policy, tactical_pred`. The earlier root-cause analysis pointed at
`flagship_v15.v15_losses`, which belongs to the **v1.5/v4/v5f** lineage, not to this arm. That
analysis was wrong for v1arch and is retracted.

The 20 waypoints come from
`rollout_decode(predictor, states, actions, future_actions, step_readout, k=20)`: the operative
predictor rolled 20 steps in **latent** space, each transition decoded by
`StepDisplacementReadout` into a **free (dx, dy, dyaw)**, accumulated by `accumulate_se2`.

| | params |
|---|---|
| encoder | 87.02 M |
| predictor | 91.36 M |
| **`step_readout`** (`grounding.step`) — **the head** | **6.32 M** |
| world model total | 263.44 M |

⇒ **6.32 M trainable against a 178 M frozen trunk.** That is why Option 2 is hours, not days.

## 2. Why a free (dx, dy, dyaw) decode produces exactly the measured defects

Nothing couples the three channels or ties them to the ego's real speed:

* **`dx_j` is free** ⇒ the implied speed can jump between steps ⇒ jerk RMS **52.13** vs a human
  **1.71**.
* **`dx_1` is free of the true `v0`** ⇒ launch transient **1.98 m/s²** vs a **0.55** floor.
* **`dy_j` is free** ⇒ the decoder may translate the ego **sideways**. A road vehicle cannot.
  Every metre of that is a heading error by construction.
* **`dyaw_j` is independent of speed** ⇒ the decoder can turn while stopped.

⭐ **The unicycle removes all four as representable states**, while emitting the **same
`[B,K,3]` `step_dpose`**, so `accumulate_se2` — already unit-pinned against the GT waypoints — is
unchanged and nothing downstream needs to know which decoder ran. `dy == 0` **is** the
non-holonomic constraint.

## 3. ⭐ THE TARGET IS REACHABLE — measured before spending any GPU

Recover the controls the **human's own path** implies and re-integrate them as a unicycle
(39 clips, 2 s / 20 wp):

| | |
|---|---|
| mean position residual | **0.0477 m** (p90 0.1239, max 0.2614) |
| **net-yaw error of the reconstruction** | **0.00114 rad** |
| v1arch today | 0.1201 rad |
| v1arch re-timed | 0.1944 rad |

⇒ **The unicycle parameterisation costs essentially nothing in heading fidelity** — 0.00114 rad
is 0.065°, against v1arch's 6.9°. The ceiling is ~100× better than where we are.

⇒ **Therefore any heading error remaining after Option 2 is a LEARNING or LATENT-INFORMATION
limit, not a representation limit.** That is the single most useful thing to know before the run,
and it cost 30 seconds. Had the residual been large, Option 2 would have been dead on arrival.

## 4. What is built

`stack/tanitad/models/metric_dynamics.py`:

* `UnicycleStepReadout` — same input contract and trunk shape as `StepDisplacementReadout`,
  output layer **zero-initialised** so the decode starts at *"hold the true v0, go straight"*: a
  kinematically valid trajectory rather than noise. ⛔ A randomly-initialised control head
  integrates its own noise **twice** and starts from a physically absurd trajectory — a far worse
  basin to descend from.
* `UnicycleStepReadout.warm_start_from(sr, …)` — copies the trained trunk (the expensive half:
  it already knows how to read a latent transition); only the 2-channel head is new. ⚠️ **Raises**
  on a shape mismatch rather than returning a random module labelled "warm-started".
* `unicycle_step_dpose(controls, v0, dt)` → `[B,K,3]` in `accumulate_se2`'s exact convention.
* `rollout_decode_unicycle(...)` — drop-in for `rollout_decode`. ⚠️ **The latent roll is
  byte-identical**; only the decode differs, so an ablation between the two decoders is not also
  an ablation of the rollout.

12 tests. The load-bearing ones assert the defects are now **unrepresentable**
(`test_no_sideways_translation`, `test_cannot_turn_in_place`, `test_first_step_is_exactly_v0`) —
a loss term can be out-weighted; a representation cannot be argued with.

## 5. ⚠️ A confound that must travel with any result from this arm

`rollout_decode` is conditioned on the **TRUE future actions** (`episode_rollouts` passes
`fa = actions[s+W : s+W+K]`). The deployed 20-waypoint trajectory is therefore a **grounded,
action-conditioned world-model rollout**, not an autonomous plan. That is the existing evaluation
contract and Option 2 does not change it — the comparison stays apples-to-apples because both
decoders see the same roll — but **no number from this arm may be presented as closed-loop
planning performance**, and the same caveat already applies to every v1arch number in the
programme.

## 6. Remaining before a run

1. A trainer: freeze `encoder` + `predictor`, train `UnicycleStepReadout` only, loss =
   position L1 on the 20-wp rollout **+ `kinematic_losses` at dt = 0.1** (the dense grid — no dt
   trap here, unlike the v15 head's 0.5 s horizon grid).
2. ⛔ **Not launched.** `train()`'s own docstring reserves a launch for the PI.
