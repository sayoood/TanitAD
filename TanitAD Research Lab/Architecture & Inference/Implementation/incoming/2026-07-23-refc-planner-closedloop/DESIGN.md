# DESIGN — In-envelope geometric recovery augmentation for REF-C's anchored-diffusion planner

**Stream D2** (`a1f26c92`, pod-free). **Date:** 2026-07-23 (Berlin). **Status:** DESIGN + ready-to-run,
CPU-smoked implementation, STAGED. **Nothing launched; nothing running was touched (REF-C's deployed
weights are read-only).** The experiment runs on the eval pod when it frees from `abe82f1f`.

**Evidence discipline:** `MEASURED (artifact)` / `PUBLISHED (cite)` / `INHERITED` / `HYPOTHESIS`. Read
`RETRACTION_LOG.md` first; §6 states plainly where this fails.

---

## 0. TL;DR — one sentence, one change

Teach REF-C to **recover from an off-path pose it can only see in pixels**, by warping each real training
window with an analytic homography drawn from the **MEASURED low-OOD envelope** and supervising the
**return-to-path** trajectory — the *same* warp operator the closed-loop instrument applies on-policy, so
the planner learns exactly the distribution it is scored on. It changes **one thing** vs base REF-C
training (warped input + recovery target); the frozen 90 M encoder cannot be degraded; and it is
**data-efficient** — every window is a recovery example, so it is not the memorising Gate-1 FT, not the
ruled-out free floor, and not the self-referential DAgger.

---

## 1. The lever, precisely

Base REF-C training (`refc_train.compute_losses`, MEASURED source-read): on a real window `frames`, target
= `waypoint_targets(pose_last, future_poses, horizons)` (ego-frame waypoints), losses = traj-L1 +
anchor-CE (+LAW/route/maneuver aux). The ego is **on** its path at `t`, so the target is a
follow-the-path trajectory.

**The augmentation changes the pose the window is taken from — nothing else:**
1. **Sample** a per-window in-envelope offset `(dlat, dpsi)` (left +): `|dlat| ≤ 1.75 m`, `|dpsi| ≤ 5°`,
   with a `clean_frac` (0.30) mixed in at `(0,0)` (the BC anchor; Urban-Driver/MGAIL "mix closed-loop with
   open-loop BC" law, angle1). Magnitudes are **MEASURED-justified** (P1 §1.2/1.3: lateral flat to 2.0 m,
   yaw exact and ≤1.16× to 12°; both ~17× below NuRec's 3.2×).
2. **Warp** the window by the homography for `(dlat, dpsi)`: `warp_windows` — a **byte-copy of the
   instrument's `sampling_homography`+`warp_batch`** (`perturb._assert_warp_matches_harness` verifies the
   copy is exact at run time). The frame now looks as if the ego sits `dlat` left / `dpsi` rotated.
3. **Target = the return-to-path trajectory:** the recorded future re-expressed in the **perturbed** ego
   frame, `recovery_targets(...) = waypoint_targets(perturbed_pose_last(dlat,dpsi), future_poses, H)`.
   Because the recorded future returns to and continues along the lane, expressing it from the perturbed
   pose yields a trajectory that **first corrects the offset, then follows** — a recovery. At `(0,0)` it is
   **byte-identical** to the base target (`MEASURED perturb.validate_identity: identity_target_maxerr 0.0`).
4. **Loss = base REF-C traj-L1 + anchor-CE on the recovery target** (identical weights), **+ `λ_dev`
   trust-region on clean windows** toward the base plan (the Gate-1 fix; keeps on-path behaviour from
   drifting — prevents the "3 passing scenes go newly off-road" side-effect).
5. **Scope = the anchored-diffusion decoder only** (~Gate-1's 8.6 M); the encoder + measurement + LAW +
   aux are **frozen**. A frozen-encoder forward is cheap and, crucially, **the world model / encoder can
   never be degraded** — the exact hazard that has cost v4/v4.1/v4.2 (`INHERITED`, LOOP_STATE).

This is PilotNet's shift+rotation recovery (`PUBLISHED ✓ 1604.07316`) and ChauffeurNet's
synthesize-the-worst (`PUBLISHED ✓ 1812.03079`), realised as an image homography for a modern
anchored-diffusion planner, and — the part that is ours — **bounded to a measured OOD envelope and pinned
to the instrument's own operator**.

---

## 2. Why it is DATA-EFFICIENT (the property the mission requires)

Gate-1 MEASURED that recovery FT from **real** on-policy junction scenes memorises at n≈15 (held-out
recovery-L1 5.06→5.06, Δ≈0) because there are only 13–22 distinct real junction episodes and collecting
more is a data-eng project. `MEASURED (gate1_clean_loo.json)`.

The augmentation **manufactures the recovery-supervision set from geometry**: every one of the ~881 clean-
val windows (or every window of the 2376-ep train corpus, if used) becomes a recovery example under the
envelope of offsets, from *all* road geometry — not just the scarce junctions. So the effective recovery
set is **1–2 orders larger and far more diverse than 675 labels from 15 scenes**, at **zero** new data-eng
cost. `HYPOTHESIS` (the pre-registered claim): this is *why* it can generalize to held-out episodes where
the scene-collected FT could not — and the experiment is built to measure exactly that, both outcomes
committed.

---

## 3. Why it is genuinely NEW (not the ruled-out floor, not the held FT)

| ruled-out / held lever | how this differs |
|---|---|
| **Free inference floor** (Gate-0/0b selection + gradient nudge; rung-3 WM-MPC) — MEASURED null | this is a **training** change that reshapes the *executed* plan from off-path states; the floor only reshapes the plan at a fixed (on-path-trained) policy. Gate-0b showed the plan is already on-road; the deficit is the policy's off-path behaviour, which only training moves. |
| **Gate-1 real-junction recovery FT** — HELD (data-bound, n≈15) | **data source**: synthetic in-envelope recovery from *every* window vs real recovery from 15 scenes. Same decoder, same objective family, **opposite data-efficiency**. The envelope bound also *is* the CAT-K feasibility filter (no 49 % catastrophic labels). |
| **DAgger on the WM** — HURT (self-referential) | **state source**: a *real* frame warped by analytic geometry vs the WM's *own imagined off-manifold latent*. Not self-referential; the recovery target is bounded, not an aggressive cut-back to re-close a hallucinated drift. |
| **CAT-K / RoaD sim rollouts** — blocked (3.2× OOD) | **no renderer**: a homography of a real frame stays in the encoder's distribution (P1-measured), vs NuRec at 3.2×. |
| **Analytic-grad-thru-WM** (v4 family) | **cheap + WM-safe**: decoder-only, frozen encoder, no WM in the loss → no WM-exploitation, no v4 coupling problem; ~1–2 GPU-h vs a v4-scale training project. |

**Lever accounting.** This does **not** open a new encoder-touching structural lever (the door is CLOSED,
RETRACTION_LOG 07-21 C4): the encoder is *frozen*; only the existing decoder is fine-tuned on a re-sampled
input/target. It is a *data/objective* lever, not an architecture graft.

---

## 4. Architecture fit — why REF-C is the ideal target (MEASURED source-read)

- **Pixels-only pose sensing** (`refc.py::RefCModel.forward`): REF-C's sole pose signal is the 8-frame
  window (+scalar `v0`, 0.5 ego-dropout). ⇒ a **pixel-space** perturbation *is* the covariate shift, exactly
  and completely — there is no ego-history channel that would need a separate (and causally-confusing,
  §P0.3) perturbation. The augmentation is the natural and complete recovery signal for this architecture.
- **The decoder cross-attends the conv map** (`AnchoredDiffusionDecoder`): the warped frame's features
  carry the offset; the trainable decoder learns to read them and re-select/refine toward a recovery anchor.
  The FPS anchor vocabulary already spans curved/returning trajectories (unicycle pool, yaw-rate ±0.35), so
  the recovery targets are representable. `MEASURED (source-read)`.
- **Truncated diffusion + ego-dropout are preserved** (decoder `training=True`, `ego_dropout` kept at 0.5) so
  the FT differs from base training in exactly one dimension. `MEASURED (recovery_aug_ft.py smoke: frozen-
  forward vs model.forward parity = 0.0 → identical wiring to the eval path)`.

---

## 5. Implementation (STAGED, CPU-smoked)

| file | what | smoke |
|---|---|---|
| `perturb.py` | geometry: envelope sampler · warp (byte-copy of the instrument) · perturbed-pose recovery target · `validate_identity` / `_assert_warp_matches_harness` | ✅ `identity_target_maxerr 0.0`, `H_maxerr 0.0`, perturb moves target 1.15 m |
| `recovery_aug_ft.py` | decoder-only recovery-aug FT; frozen-encoder forward (parity 0.0 vs `model.forward`); traj-L1+anchor-CE on recovery target + `λ_dev`; saves a REF-C-shaped ckpt + config.json | ✅ parity 0.0; decoder-only grad (encoder grads None); 25 k trainable (smoke cfg) |
| `recovery_probe.py` | P2a zero-training recovery-response probe (`recovery_ratio`) | ✅ metric math (demand tracks offset) |
| `eval_corridor_split.py` | P2b PAIRED held-out corridor_departure (base vs FT), reuses `lowood_lanekeep.py` verbatim | ✅ paired bootstrap math |

The FT emits a `{model, step}` ckpt + `config.json` (base cfg) so the abe82f1f instrument loads it verbatim
via `lowood_lanekeep.py --refc-ckpt <ft>/ckpt.pt`. **No instrument code is modified.**

---

## 6. Where this fails — stated plainly (both pre-registered)

1. **If REF-C already recovers from a warped view** (the probe's `recovery_ratio ≈ 1`), the departures are an
   *execution/controller* problem the augmentation cannot touch (the plan already corrects) → **P2a kills
   the FT for ~0 GPU**, and the honest verdict is "closed-loop lane departure is downstream of the plan; the
   lever is controller/receding-horizon, not planner recovery." `HYPOTHESIS`, and the reason P2a runs first.
2. **If synthetic recovery does not transfer** (held-out corridor_departure flat, `PRE_REGISTRATION` NULL),
   then even data-rich in-envelope recovery does not generalize through a decoder-only FT — the bottleneck
   is deeper (frozen-encoder features don't separate off-path enough, or the diffusion decoder capacity),
   and the next step is a heavier arm (encoder-in-the-loop FT, or the v4 analytic-grad direction). This is a
   **measured bound**, not a failure — and it is worth as much as a win (it retires a cheap lever with a
   reason, exactly as Gate-1 retired the naive rerun).
3. **If it recovers but over-steers** (held-out `peak_xte` rises — the Gate-1 high-deviation side-effect),
   the envelope bound + `λ_dev` were insufficient; the pre-named fix is a larger `λ_dev` / smaller
   `lat_max` — one further thing, re-read.
4. **Ground-plane lateral is optimistic** (P1 §2): the lateral recovery signal is an upper bound; yaw
   (exact) carries the load. We cap lateral at the corridor half-width and lean on yaw.
5. **Lane-keeping ≠ safety.** This validates the covariate-shift *mechanism* on corridor departure; a real
   off-road/collision rate still needs AlpaSim (the 3.2×-OOD instrument this escapes). The lever, if it
   wins here, is then a candidate for an AlpaSim confirmation — never promoted on the low-OOD read alone.

**Honest bottom line.** In-envelope geometric recovery augmentation is the right lever **iff** REF-C is
covariate-shift-*blind* (P2a) **and** synthetic recovery *generalizes* through the decoder (P2b). The two
pre-registered experiments return which of those worlds we are in, cheaply, with all outcomes written down
in `PRE_REGISTRATION.md` before they run.
