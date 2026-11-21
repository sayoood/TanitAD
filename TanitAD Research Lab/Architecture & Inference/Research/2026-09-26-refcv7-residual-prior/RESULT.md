# refcv7 NEW-1: the plan is a residual on a causal kinematic prior

`Architecture & Inference · 2026-09-26, revision 2 2026-09-27 · SPEC: Project Steering/SPEC_REFCV7.md §1 NEW-1, §7 A2, §10 A5 (branch 12953d2)`

## 0. Headline

**Status (revision 2, 2026-09-27): READY FOR THE THOR RE-GATE on tip `12953d2`.**

The Master Mind's Thor full-suite gate (TIP `b3f7ea6` vs TIP + the 7 batch-1 files) passed NEW-1's own tests (29 passed + 3 honest skips). It found **21 regressions in the RL stack, in two groups. Both are fixed here, and no guard is weakened (§11):**
1. **8 × `TypeError` in `test_ddv2_refc_chain.py`.** The DDv2 chain's fixed-arity capture hook did not accept `_sample`'s new `prior=`.
   - The hook now forwards the prior verbatim and records it (`SamplerInputs.prior`), and every chain roll composes on it exactly as `_sample` does.
   - The decoder now passes `prior=` only on a residual build, so an off build calls `_sample` exactly as refcv6 did.
2. **13 × `RequirementDeclarationError` / two-way-pin failures in five RL files.** NEW-1's forward channel `ego_actions` was declared nowhere. It is now declared by the seam that owns it: `kinematic_prior.FORWARD_EXCLUSIONS`, listed in `channel_admissibility.SEAM_MODULES`. The declaration is must-not-be-plumbed, TEMPORARY, with its route back. `FORWARD_KEYS` and every guard's code are unchanged.

The full files under `code/fix/` are 8 shared files and 2 new ones, built from the `12953d2` blobs; its `stack/` code equals `ab436ee`'s for every file touched. They are listed with base and new blobs in `LANDING_READY.txt`; §11d says which blobs changed since the gate.

**Verification status: NOT RUN LOCALLY. The definitive run is the Master Mind's Thor full-suite gate** (tip `56ae4eb` vs tip + these 10 files; the overlay blobs were checked against `LANDING_READY.txt`). The local clean-tree A/B was RAM-gated all session and was stood down by the Master Mind before a single test file ran (§11c). Every new test is therefore UNVERIFIED by execution; only the static checks in §11c completed.

⛔ **ESCALATIONS (revision 2):**
1. **The `ego_actions` declaration contains a design decision (§11b).** The brief said "required only when `residual_prior == "ha0_ext"`, otherwise must-not-be-plumbed". The RL machinery cannot express a per-build exclusion: a seam exclusion removes the channel from both adapters for EVERY build.
   - I chose the exclusion. SPEC §10 excludes `ha0_ext` for refcv7, and no sanctioned build needs the channel.
   - On an `ha0_ext` build the requirement is enforced one layer down: the model refuses a missing `ego_actions`, and a supplied, unread one.
   - The alternative (plumb it, and require it only for `ha0_ext`) is written in the declaration's route back. The Master Mind picks.
2. **SPEC §10 (A5)'s G-DVB clause is NOT built** (§11e). "G-DVB refuses any other mode for a refcv7 launch" needs one clause in `check_refcv7_required`, which the fixes agent's batch 2 owns. The Master Mind names the owner.

**Revision 1 (2026-09-26), unchanged below.** The status line was "READY TO LAND, rebased on the fixes batch `ab436ee`", with 5 shared files and 2 new ones.

**What is done:**
- The prior module, `stack/tanitad/models/kinematic_prior.py`.
- Its decoder, model and trainer wiring.
- A G-DVB entry.
- 32 tests (`stack/tests/test_residual_prior.py`). Each mutation arm goes RED.

**Verified on clean `git archive` trees of `ab436ee`** (tip vs tip + this package; `code/finalize_after_fixes.py`):
- `test_residual_prior.py` on tip + NEW-1: **32 passed**, 0 skipped.
- **`--residual-prior off` is bit-identical to the tip.** The off-mode digest reads `ebc4db4599e6…` on the tip AND on the candidate, equal to the test's literal.
- The curated A/B: §7a.
- Before the landing, the patch was pre-checked on the fixes batch's declared blobs.

✅ **DECIDED 2026-09-26 by SPEC §10 (A5, `a756e81`): refcv7 sets `--residual-prior ha0_ext_pose`**, the recommendation below. Revision 1's escalation is kept as the evidence the amendment cites.

⛔ **ESCALATION (revision 1), a Master Mind / PI decision: which prior does refcv7 set?** The brief's flag value `cv_yawrate` is **not** the battery's echo.

`ha0_ext` is **constant acceleration + constant curvature**, not constant velocity + constant yaw rate (§1). Three honest modes are built. MEASURED on the battery's own surface (4,754 windows, 139 eval clips; §3):

| mode | what it reads at t0 | ADE 0–2 s − echo | ADE 0–6 s − P(ha0_ext) | runs on NavSim? |
|---|---|---|---|---|
| `ha0_ext` | pose window + **recorded steer at t0** | **0.0000 [0, 0]**; it IS the echo, bit for bit on 4,754/4,754 windows | 0 | ⛔ **no**: NavSim has no steer channel |
| `ha0_ext_pose` | pose window only | +0.0055 [+0.0036, +0.0076] | +0.0223 [+0.0088, +0.0366] | ✅ same code path |
| `cv_yawrate` (the brief's name) | pose window only | **+0.2293 [+0.1971, +0.2642]** | **+0.5300** | ✅ |

**Recommendation: `ha0_ext_pose`.**
- It reads only the pose window, which refcv6 already requires at every call site: trainer, battery and NavSim bridge.
- Its admissibility is already ruled: past ego data (SPEC_REFCV6_V2 §10.3), and "measured current acceleration, yaw rate" (PI 2026-09-03, `refc_v3.py:144-147`).
- Its zero residual sits 5.5 mm behind the echo at 2 s. That is 4× smaller than refcv6's own measured miss of the echo bar (os − echo **+0.0225**, reproduced here, §3).

**`ha0_ext` is available if the PI rules two things:**
1. κ0 read from the recorded steer channel at t0 is admissible at inference. The battery's own lever panel records it as *"unruled at inference"* (`lever_panel.py:14-15`).
2. What NavSim feeds instead. **BAR-R7-N1 is scored on NavSim**, and NavSim has no steer channel.

**Do not launch `cv_yawrate`.** It drops a0, and it starts 0.23 m behind the bar it must beat.

**Other items for the Master Mind** (§8, §9):
- The G-DVB entry, `DECODER_PASSTHROUGH` membership, and a test-literal bump (202 → 203) ride in the patch. This covers the coordinator's items 1–4 (§6).
- Four refcv6 defects were found on the way and are not fixed here. One interacts with NEW-1 and is handled inside it.
- No anchor rebuild is needed (§4).

## 1. What the battery's echo `ha0_ext` is, read at source (tip 59f0d46 ≡ 5de9363 ≡ da8400b for every file cited)

The refcv6 battery loads `taniteval/tools/refcv3_arm.py` (`battery/code/refcv6_roll.py:48`), and `refcv6_panel.py:548-558` repeats the same two calls for S6.

| step | file:line | what it does |
|---|---|---|
| episode signals | `refcv3_arm.py:2081-2082` | `v_ep = ep.poses[:, 3]`, `kap_ep = ep.actions[:, 0]` |
| the steer channel | `stack/tanitad/data/physicalai.py:621,632` | `steer = arctan(2.9 * curvature)`, column 0 of `actions` |
| the window origin | `refcv3_arm.py:2087,2101` | `t0 = t + W - 1`, `v0 = float(pose_last[3])` |
| the echo's controls | `refcv3_arm.py:2125-2128` → `refav1_arm.py:445-446` | `a0 = (v[t0] - v[t0-1]) / 0.1`; pair `[a0, steer[t0]]` |
| integration | `refcv3_arm.py:1677-1692` → `refav1_arm.py:471-493` → `refa_v1_plan.py:344-373` | `kappa = tan(steer)/2.9` (`kinematic.py:57-59`); `rollout_unicycle` from `(0,0,0,v0)` at dt 0.1 |
| the integrator | `kinematic.py:220-264` | forward Euler; `yaw += v·kappa·dt`; `v = max(v + a·dt, 0)` LAST |

⇒ `ha0_ext` holds `a0` and `kappa0` constant. Its yaw rate is `v(t)·kappa0`, which changes whenever `a0 ≠ 0`. The SPEC's "(v0, ω0) constant-velocity / constant-yaw-rate" wording coincides with it only when `a0 = 0`.

⚠️ `stack/tanitad/eval/echo_gate.py::ha0_ext` is a **different** function. It is the closed-form CA/CC with the corpus's `ax` (`refc_v3.py:284-347`). This package matches the battery's version, because the bars are read on it.

## 2. The design

**The mode flag.** `--residual-prior {off, ha0_ext, ha0_ext_pose, cv_yawrate}` sets a declared `DecoderConfig.residual_prior`, default `off`.

**The three modes.** They differ only in how `(a0, κ0)` is read:
- `ha0_ext`: `a0` from the speed difference; κ0 = `tan(steer[t0])/2.9`.
- `ha0_ext_pose`: the same `a0`; κ0 = `ω0 / max(v0, 2.0)`, clamped ±0.3.
- `cv_yawrate`: `a = 0`; κ0 as `ha0_ext_pose`.

**Pose-derived quantities.** Every one comes from `ego_history.ego_channels_from_poses`, the ego-history encoder's own builder (`ego_history.py:96,98`). That builder is causal by construction and pinned by a mutation test. Its `a0` is bit-identical to the echo's.

**The two pose constants are MEASURED** (train-a6, 139 train clips, 27,504 frames; `raw/probe_kappa_sources.txt`):
- Below 2 m/s the pose yaw rate correlates only **0.22–0.46** with `v·κ_steer`. From 2 m/s up it correlates **0.996–0.998**.
- The largest `|κ_steer|` is **0.2766**.

**The composition is in control space.** The sampler's state is a per-slot control sequence (`refc.py:2499`), and every emitted path integrates controls (`refc.py:2430-2445`). The prior is itself a constant-control integration through the same integrator, so:

    a(t) = a0 + Δa(t)
    κ(t) = κ0 + alat_to_curvature(Δa_lat(t), v0)      (the vocabulary's own conversion, refc_sampler.py:312-324)

The prior's curvature is added **after** the conversion, never pushed through it. MEASURED: `|κ_steer| > 0.12`, the vocabulary cap, on **1.77 %** of train frames. Composing in `a_lat` would clip the echo itself there.

**`Δ = 0` rolls to `P` exactly.**
- `a0 + 0.0 == a0`, and `alat_to_curvature(0) == 0`.
- `kinematic_prior.roll_plan` is **the one function** that turns a residual into a path.

**Where the prior is computed.** `RefCModel.forward` computes it from the ego window it already pops (`refc.py:4409-4412`) and hands it to the decoder. The decoder composes it into:
- the bank (`roll_bank`);
- the sampled fan (`_sample` via `_roll_state`);
- the per-stage predictions.

**The residual never leaves the decoder.** `u0_hat` and `layer_u0_hat` are exported as **absolute** controls. Three consequences:
- The trainer's existing x0 target (`refc_v3_train.py:4021-4037`) needs no change. Its L1 equals the residual-space L1 in exact arithmetic.
- The one production re-roll, the trainer's F3 cascade at `refc_v3_train.py:4074`, now passes `prior=`.
- `_state_to_path` **refuses** a residual build without it.

**Withheld rows (ego-dropout 0.5 on refcv6, `config.json` `ego.ego_dropout`).**
- Their bank is rolled at 10 m/s (`refc.py:2094-2155`).
- On such a row the prior is the no-information prior: zero controls. The fan is rolled at the **bank's** speed, computed once per forward (`_prior_speed`).
- So on a withheld row the residual vocabulary is exactly refcv6's absolute vocabulary.
- ⚠️ refcv6's own sampler rolls the fan at the **raw** v0 (`refc.py:2522-2523`) while its bank uses 10 m/s. That is finding F-1 (§8). NEW-1 cannot inherit it, because a residual composed on one speed and rolled at another is not P + Δ.

**Outputs.** A residual forward emits `residual_prior_ctrl` [B, 2], `residual_prior_v` [B] and `residual_prior_path` [B, S, 2]. They pass through `RefCModel`'s whitelist and appear in `RefCV3Model`'s output: the A16 lesson.

**The consumer check.** `kinematic_prior.plan_check(out, anchor_controls, decoder=)` reads only the output and checks four things:
1. The (0, 0) anchor of `anchor_bank` equals P.
2. The emitted fan equals `roll_plan(residual(u0_hat), P)`.
3. P is non-zero wherever v > 0. This is G-LIVE's clause.
4. Everything is finite.

**Refusals.**
- At build:
  - no DDIM sampler;
  - the metre sampler;
  - a fixed-path bank;
  - no `--ego-history`;
  - no `sel_reach_clamp`.
- At forward:
  - no prior on a residual build;
  - a prior on an off build;
  - `ha0_ext` without actions;
  - actions handed to a pose mode, which would silently drop them;
  - a pose window that does not end at the forward's v0 (tolerance 1e-3 m/s).

## 3. MEASURED: the prior against the echo

`code/measure_prior_vs_echo.py` → `raw/measure_prior_vs_echo.json`. T1, the battery's banked step-30k S2 dump (`C:/Users/Admin/ev6_battery/raw/step30000/dump_s0`, inference seed 0). Real eval-kit poses and actions (`D:/refcv6_eval_kit/…/eval139/_v2manifest.pt`); every banked window's pose track matched the cache bit for bit. Paired episode-cluster bootstrap (`taniteval.ci`), n_boot 2000, seed 0, cluster = clip.

**P against the echo, all 4,754 windows:**
- `P(ha0_ext)` vs the **banked** echo: max |diff| **0.0 m**, bit-equal on **4,754 / 4,754** windows.
- `P(ha0_ext)` vs the tip's battery **functions**, called per window: max **0.0 m**, n = 4,754.
- `ha0_ext_pose`: mean |P − echo| 0.0150 m, max 1.588 m.
- `cv_yawrate`: mean 0.2348 m, max 8.444 m.

**ADE 0–2 s** (n = 4,754 / 139):

| arm | mean ADE |
|---|---|
| echo (banked) | 0.2886 |
| `P(ha0_ext)` | 0.2886 |
| `P(ha0_ext_pose)` | 0.2941 |
| `P(cv_yawrate)` | 0.5179 |
| `ha` (banked) | 0.3010 |
| refcv6 `os` @30k | 0.3111 |

Paired differences:
- `P(ha0_ext)` − echo: 0.0000 [0, 0]. This is a **structural identity**, not an estimate.
- `P(ha0_ext_pose)` − echo: **+0.0055** [+0.0036, +0.0076].
- `P(cv_yawrate)` − echo: **+0.2293** [+0.1971, +0.2642].
- **Control:** `os` − echo = **+0.0225** [+0.0068, +0.0395]. This reproduces the SPEC's "refcv6 failed at 30k (+0.0225)", so the instrument reads the registered number.

**ADE 0–6 s**, 8 slots, windows with a valid 60-step future (n = 3,668 / 139):

| arm | mean ADE | − P(ha0_ext) |
|---|---|---|
| `P(ha0_ext)` | 2.7089 | — |
| `P(ha0_ext_pose)` | 2.7312 | +0.0223 [+0.0088, +0.0366] |
| `P(cv_yawrate)` | 3.2389 | +0.5300 [+0.3465, +0.7212] |

**TRAIN replication** (`code/measure_anchor_train.py` → `raw/measure_anchor_train.json`; train-a6, 139 TRAIN clips, 3,749 windows, stride 5). Prior differences against `P(ha0_ext)`:
- `ha0_ext_pose`: +0.0071 [+0.0053, +0.0088] at 2 s and +0.0308 [+0.0177, +0.0440] at 6 s.
- `cv_yawrate`: +0.2475 [+0.2092, +0.2856] at 2 s and +0.4602 at 6 s.

The ordering is the same as on eval.

## 4. The anchor question: NO rebuild is needed

**The artifact and its builder.** refcv6's vocabulary is `refc_anchors_6s_v0cond_alat_117.pt`, file sha256 `897099148d…`, the one in the launch `config.json`. It is 13 `a_lon` × 9 `a_lat` constant controls, with `(0, 0)` at index 67. Its builder is recorded in the artifact itself: `emit_anchors_alat.py`, provenance in the artifact dict.

**The minimal change: reinterpret the same 117 controls as RESIDUALS.** Anchor `i` rolls `(a0 + a_i, κ0 + κ(a_lat_i))`, so the `(0, 0)` anchor IS P.

**Oracle-in-vocabulary**, the min over anchors of the window ADE; the same statistic as the vocabulary's own gate:

| surface | span | absolute (refcv6) | residual on `ha0_ext` | residual − absolute |
|---|---|---|---|---|
| eval S2 (4,754 / 139) | 0–2 s | 0.1999 | 0.1767 | **−0.0232 [−0.0293, −0.0171]**, better |
| eval S2 | 0–6 s (3,668) | 1.2957 | 1.2752 | −0.0205 [−0.0670, +0.0253], tie |
| train-a6 (3,749 / 139) | 0–2 s | 0.2072 | 0.1839 | **−0.0233 [−0.0332, −0.0153]**, better |
| train-a6 | 0–6 s | 1.3918 | 1.3824 | −0.0094 [−0.0584, +0.0348], tie |

- The absolute 0.1999 reproduces the artifact's own gate figure, 0.1987 on refcv3's 4,823-window surface.
- The `ha0_ext_pose` residual behaves the same way: −0.0221 at 2 s on eval, −0.0214 on train.

**The classification target** (train-a6, 3,749 windows). The trainer's `a_star` is the GT-nearest anchor over all 8 slots, squared L2, as at `refc_v3_train.py:3668-3671`.

| vocabulary | entropy (nats; max 4.76) | anchors used | windows at (0, 0) |
|---|---|---|---|
| absolute | 2.53 | 65 | 37.4 % |
| residual on `ha0_ext` | 2.29 | 75 | 44.9 % |
| residual on `ha0_ext_pose` | 2.29 | 75 | 44.7 % |

With the residual vocabulary, more of the target mass sits on P itself, and more of the vocabulary is used at all. No anchor is starved by the reinterpretation.

**Friction census, μ = 0.7.** Composed controls on eval, window × anchor, n = 556,218:
- absolute: 0 over;
- residual: **825 (0.15 %)**, because the prior already carries part of the budget.

This is reported, not gated. `feasible_decode` is off on refcv6 either way.

**What to change.** Nothing in the artifact bytes. The run record says which space the bytes mean: `config.json` `seams.residual_prior.vocabulary_space` = `residual on P` (`kinematic_prior.prior_stamp`).
- **Optional (a Master Mind call):** a metadata-only restamp that declares `control_space` in the `.pt`, in the spirit of `anchor_meta.py`. It needs no corpus pass, so no builder change is required.

## 5. Consumer audit (tip file:line): who reads the planner's output

**Every PATH consumer reads a tensor the decoder composes on P before it leaves the decoder.** The only CONTROL exports are absolute. The only re-roll must pass `prior=`, and refuses without it.

| consumer | file:line (tip) | what it reads | gets |
|---|---|---|---|
| classifier pass on the bank | `refc.py:2987,2996` | `bank` (= `x0`) | P ⊕ anchors |
| refined fan (non-sampler builds) | `refc.py:2999` | `bank + offset` | P ⊕ anchors + offset |
| LAN / goal / GP priors | `refc.py:3041, 3214-3220, 3260` | `prior_bank` / `bank` | P ⊕ anchors |
| S2b anchor prefilter | `refc.py:2953` | `bank` | P ⊕ anchors |
| consequence scores | `refc.py:3265` | `x` (fan) | P ⊕ Δ |
| nav-compliance term | `refc.py:3292` | `x` | P ⊕ Δ |
| reachability band | `refc.py:3312` | `x` | P ⊕ Δ |
| speed-ceiling filter | `refc.py:3338` | `x` | P ⊕ Δ |
| selection argmax → `traj` | `refc.py:3343, 3376` | `x` | P ⊕ Δ |
| LAW head | `refc.py:4461` | `traj` | P ⊕ Δ |
| E9 goal re-selection → `traj` | `refc_v3.py:2114, 2161-2165` | `anchor_traj` | P ⊕ Δ |
| trainer `a_star` | `refc_v3_train.py:3668-3671` | `anchor_bank` | P ⊕ anchors |
| trainer matched-anchor L1 | `refc_v3_train.py:3684` | `anchor_traj` | P ⊕ Δ |
| trainer fan-error diagnostic | `refc_v3_train.py:3871` | `anchor_traj` | P ⊕ Δ |
| trainer x0 loss | `refc_v3_train.py:4021-4037` | `u0_hat` | **absolute** controls (target unchanged) |
| trainer F3 cascade re-roll | `refc_v3_train.py:4074` | `layer_u0_hat` → `_state_to_path` | absolute + **`prior=`** (patched; refuses without) |
| battery `os` / `plan_full_*` | `refcv3_arm.py:2277, 2284` | `traj` | P ⊕ Δ |
| battery `oracle_sel` / `a_star` | `refcv3_arm.py:1585-1623, 2307, 2401-2404` | `anchor_bank`, `anchor_traj` | P ⊕ anchors / P ⊕ Δ |
| battery refcv6 extras + obedience | `refcv6_roll.py:60, 197-203` | `traj`, `anchor_traj` | P ⊕ Δ |
| NavSim export | `navsim/code/refcv6_bridge.py:455-456, 466` | `traj` → `knots_to_navsim` | P ⊕ Δ |
| trajectory metrics | `taniteval` four-families on the banked `os` | the dumped `traj` | P ⊕ Δ |
| RL adapters (not in a refcv7 launch) | `rl/refcv3_adapter.py:956-958`, `rl/refc_adapter.py:707-709` | `anchor_traj − offset` | P ⊕ Δ; their decomposition predates the sampler and is unchanged |

**Proof that none can receive Δ** (`stack/tests/test_residual_prior.py`, §7). `plan_check` passes on the real model output, and each of these mutations goes RED:
- a roll that drops the prior;
- a fan-only roll without the prior;
- exporting Δ as `u0_hat`;
- a re-roll that forgets `prior=`.

### 5a. SPEC §7 A2 (PI, 84c983c): nav compliance and the ceiling filter rank ABSOLUTE plans

**The seam: the composition is UPSTREAM of selection by construction.** No mechanism can see Δ.

Line numbers are for the DELIVERABLE, `code/fix/stack/tanitad/refs/refc.py` (revision 2, 2026-09-27: `12953d2` + NEW-1; blob `6638d443533f…`).
- The prior is computed once per forward: `_rp = self._residual_prior(...)` (`refc.py:3159-3160`). The bank is `roll_bank(..., prior=_rp)` (`:3161-3162`).
- The sampler rolls its fan as `fan = self._roll_state(u0_hat, v, metre, prior)` (`refc.py:2836`). `_roll_state` (`:2317-2329`) is `kinematic_prior.roll_plan(Δ, P)`.
- The decoder's `x` is that fan: `x, u0_hat, s_conf, smp_tele = self._sample(..., **_smp_kw)` with `_smp_kw = {"prior": _rp}` on a residual build (`:3362-3365`, §11a), then `x = self._feasible(x, v_ms)` (`:3368`).
- Both mechanisms read that same `x`:
  - `_nc, _ = v6sel.nav_compliance_prior(x, nav_cmd_sel, …)` (`:3530-3531`);
  - `v6sel.SpeedCeilingFilter(...)(x, v_limit_ms)` (`:3577-3579`), under the landed inference-only guard.
- The 8×8 tactical graft is a learned anchor-index prior and reads no geometry. The geometric LAN / goal / GP priors read the prior-composed `bank` (`prior_bank`).

**Test:** `test_A2_nav_compliance_and_the_ceiling_rank_the_ABSOLUTE_plan`, with its RED arm `test_A2_MUTATION_Delta_fed_to_selection_goes_RED`.

**Set-up.** The residual build has both mechanisms ON (τ 0.05). The crafted window's P accelerates (v0 12 m/s, a0 +1.5 m/s²) and turns LEFT (ω0 0.3 rad/s). It runs under a RIGHT command and a 50 km/h ceiling. The sampler noise is zeroed, so the (0, 0) residual candidate IS P, bit for bit. Each mechanism's input and output are recorded where the decoder calls it.

**Normal build:**
- Both mechanisms receive the emitted fan, bit for bit.
- The ceiling **masks P**, while braking residuals survive.
- Nav compliance scores P at **0**. It also scores **0** the right-turning residual composed on P, which still turns left.

**RED arm (the fan rolled without P, so Δ reaches selection):** the verdicts flip.
- The (0, 0) candidate is a straight line at 12 m/s, and it is **kept**.
- The right residual alone **complies (1.0)**.

**Results:** 2/2 pass on the clean `ab436ee` + NEW-1 tree (within the 32).

## 6. Flag, config, G-DVB, gate hooks

**Flag.** `--residual-prior` (`refc_v3_train.py:9871`; argparse choices = `kinematic_prior.RESIDUAL_PRIOR_MODES`, default `off`) is pinned in `_pin_trainer_cfg` (`:483`) onto `cfg.core.decoder.residual_prior`.
- **G-HYG:** that is a DECLARED `DecoderConfig` field (`refc.py:612`). NEW-1 sets no other config attribute.
- It refuses without `--ego-history`.
- It is replayed by every eval loader, because they rebuild through `build_parser` + `_pin_trainer_cfg`.

**`RefCModel.DECODER_PASSTHROUGH`** (`refc.py:3750-3756`) now carries `residual_prior_ctrl`, `residual_prior_v` and `residual_prior_path`:
- The F3 cascade re-roll READS the first two (`refc_v3_train.py:4291-4301`, via `kinematic_prior.prior_from_out`).
- G-LIVE reads the third.
- **Regression arm:** `test_passthrough_carries_the_prior_and_a_dropped_key_goes_RED` drops `residual_prior_v` from the tuple. The trainer's own loss then REFUSES (a partial key set), and the G-DVB entry reports the missing key.

**Stamp.** `_seam_stamp` → `residual_prior` holds the full definition, including `equals_battery_echo` and `vocabulary_space`. `assert_seams_are_built` checks the stamp against the built decoder in both directions.

**G-DVB entry, EXACT** (`declared_vs_built.py:650`, registered at `:795`; generated by `code/apply_residual_prior_edits.py`):

```python
def _c_residual_prior(m, a):
    want = str(_a(a, "residual_prior", "off"))
    d = _dec(m)
    out = _eq("residual_prior", want, str(getattr(d, "residual_prior", "<absent>")),
              "core.decoder.residual_prior")
    out += _eq("residual_prior", want, str(getattr(d.cfg, "residual_prior", "<absent>")),
               "core.decoder.cfg.residual_prior")
    if want != "off":   # the three mechanisms the prior composes with must be BUILT
        ... core.decoder.anchor_v0_cond / core.decoder.time_mlp / core.ego_hist ...
        ... and residual_prior_ctrl / _v / _path in type(core).DECODER_PASSTHROUGH ...
    return out
_b("residual_prior", _c_residual_prior)          # after _b("ego_history", _c_ego_history)
```

- **Kind:** `built`.
- **Regression arm:** a residual build with `core.decoder.residual_prior` forced to `off` must be reported. This is pinned by `test_gdvb_entry_reads_the_BUILT_prior_and_goes_RED_when_it_is_lost`, which **fails rather than skips** when the registry lacks the entry.
- **One literal moves:** `test_declared_vs_built.py`'s two-way pinned `len(dvb.REGISTRY) == 202` becomes **203** (+ `--residual-prior`). That edit is in the patch too.

**For the gate agent (G-LIVE):**
- "The residual prior is non-zero where v0 > 0" is `rep["rows_v_gt_0_prior_nonzero"] == rep["rows_v_gt_0"]`, with `rep = kinematic_prior.plan_check(out, model.core.decoder.anchor_controls, decoder=model.core.decoder)`.
- `rep["ok"]` also asserts that the plan is P + Δ.
- `rep["rows_prior_controls_nonzero"]` counts the rows that carry a measured prior at all; withheld rows are zero by design. It is informative, not gated.
- **Suggested launch-profile rule for G-DVB:** a refcv7 argv must carry `--residual-prior` set to the PI's chosen mode. `off` or `cv_yawrate` is refused, the same way SPEC §7 requires the three selection mechanisms to be ON.

## 7. Tests, the mutation matrix, and the clean-tree A/B

**`stack/tests/test_residual_prior.py`: 32 tests, CPU, about 8 s, literal expectations.**

- **(a) Analytic cases:**
  - At 10 m/s the prior is x = [5, 10, 15, 20, 30, 40, 50, 60] and y = 0, exactly.
  - A constant yaw rate lies on the forward-Euler circumcircle. Its radius is 50.00083334305565, not the continuous 50. The centre is (0.5, 49.99833332222211), derived from chord geometry in `math`.
  - Deceleration stops the car at x = 2.4 m.
  - Each mode reads what it says.
  - The floor and the cap hold.
  - Each mode refuses inputs it does not read.
  - The prior reads nothing after t0 (a mutation of the future steps).
- **(b) Real windows:**
  - P equals the **banked** echo on 5 real clips (≥ 50 windows, max ≤ 1e-5 m; MEASURED 0.0). Its control, the pose prior, must differ.
  - P equals the battery's **functions**, called per window.
  - Both skip with a named env var when the kit is not reachable.
- **(c) Off builds:**
  - **Bit-identity to refcv6.** A sha256 over every loss scalar, every parameter gradient (171 tensors) and every planner output of the refcv6-shaped smoke arm (DDIM, alat v0-conditioned anchors including (0, 0), F1–F6, ego history, ego-dropout 0.5). It was recorded on a clean `git archive` of the TIP, whose code has no such flag, as `ebc4db4599e6…` (`raw/offmode_digest_tip_59f0d46.json`). The candidate reproduces it with the flag absent and with `off` (`raw/offmode_digest_cand_{default,off}.json`). The discriminating control, a residual build, moves it (`333737ad9abf…`) with the **same** 161,489 parameters and state_dict keys. It is single-threaded: MEASURED, the digest otherwise changes with the thread count. It is platform-guarded to torch 2.11.0+cu128 / win32.
  - An off build never reaches `kinematic_prior`: every entry is patched to raise. On a residual build the same patch goes RED.
  - A zero prior reproduces the off build's bank, fan, pick and controls **bit for bit** when no row is withheld.
- **(d) The prior at work:**
  - `Δ = 0` rolls to P **bit for bit** in both units, including κ0 = 0.25 above the vocabulary cap.
  - A zero prior equals `roll_controls`.
  - The absolute and residual control conversions round-trip.
  - The model-level output passes `plan_check` for all three modes: the zero anchor − P = **0.0**, the fan re-roll < 1e-3 m, and the prior is non-zero on 4/4 moving rows.
  - The trainer's own loss is finite with `cascade` live, and every cascade control head gets gradient.
  - The stamp and the built decoder agree, and a mutated stamp is refused.
  - On a withheld row the prior is zero and the row is rolled at 10 m/s; the straight anchor is x = 10t, as a literal.
  - Training forwards with withheld rows pass `plan_check`.
  - A residual checkpoint reloads **strictly** (0 missing / 0 unexpected) into a model rebuilt from the same argv, and replays the plan and the prior **bit for bit**. This is G-EVAL / G-CKPT's NEW-1 slice.
  - The G-DVB entry reads the built decoder, and its regression arm goes RED.
  - NEW-1's keys are in `DECODER_PASSTHROUGH`. Dropping one makes the trainer refuse and G-DVB report it.
- **(e) Mutations:** a roll that drops the prior, a fan-only roll without the prior, and Δ exported as `u0_hat`. Each is RED.
- **Refusals:** each build-time and forward-time refusal in §2.

**Mutation matrix** (`code/mutation_matrix.py`). Each run is on a scratch copy, restored byte for byte, and the baseline is re-run afterwards.

Three runs:
1. **The FINAL tree, `ab436ee` + NEW-1**, the mutants that the landing touched (`raw/mutation_matrix_final_ab436ee_subset.log`). Baseline **32/32**, restored 32/32:
   - M4, NEW-1's keys dropped from `DECODER_PASSTHROUGH`: **14 RED**;
   - M3b: **1 RED**;
   - M8: **3 RED**.
2. **The fixes batch's declared blobs + NEW-1**, the full matrix (`raw/mutation_matrix_fixes_plus_new1.log`): baseline **30/30**, restored 30/30. Here M4 was the pre-tuple form of the pass-through.
3. **Pre-fixes tip + NEW-1**, before the G-DVB and A2 tests existed (`raw/mutation_matrix_tip_plus_new1_27tests.log`): baseline 27/27.

| mutant | RED tests (fixes + NEW-1) | (tip + NEW-1) |
|---|---|---|
| M1 `ha0_ext` reads the steer at t0−1 | 2 (both real-window tests) | 2 |
| M2 the prior is squeezed through the κ cap | 3 | 3 |
| M3b the fan is rolled at the RAW v0 | 1 (the withheld-row training test) | 1 |
| M4 the pass-through drops the prior keys | 11 | 9 |
| M5 the prior's speed ignores ego-dropout | 2 | 2 |
| M6 no withholding | 2 | 2 |
| M7 the prior is read one tick late | 9 | 7 |
| M8 the cascade re-rolls without the prior | 2 | 2 |
| M9 an off build computes a prior | 1 | 1 |

M3 (only `_sample`'s `v = prior[2]` dropped) is an **equivalent mutant**, recorded: `_roll_state` reads the prior's own speed, and 0 tests fail on either tree. **Every effective mutant is RED on both trees.**

The A2 RED arm (Δ fed to selection) is inside `test_A2_MUTATION_Delta_fed_to_selection_goes_RED`. The G-DVB regression arm is inside `test_gdvb_entry_…`.

### 7a. Clean-tree A/B: tip vs tip + NEW-1, per test file, JUnit XML, RAM watchdog

**The instrument.** `code/suite_ab_perfile.py` runs one pytest process per file, per tree, and reads the JUnit XML, never a log tail. A RAM watchdog aborts a file rather than starve the PI's desktop, and an aborted file is reported, never counted as a pass.

**The final base `ab436ee`, curated** (`raw/suite_ab_curated_ab436ee_state.json`):
- 5 files completed before the box ran out of headroom: `test_declared_vs_built`, `test_config_hygiene`, `test_speed_ceiling_inference_only`, `test_label_clock_guard`, `test_guard_blind_spots_fix5`.
- **79 = 79 passed, outcome-identical per test, 0 regressions.** Here the candidate carries the 202 → 203 literal and the new registry entry.
- Separately, on the fresh `ab436ee` + NEW-1 tree:
  - `test_residual_prior.py`: **32 passed**;
  - `test_declared_vs_built.py` + `test_config_hygiene.py`: **58 passed**;
  - the off-mode digest tip == cand.

**The pre-fixes tip `59f0d46`, the blast-radius subset** (`raw/suite_ab_pre_fixes_59f0d46_partial_state.json`):
- The subset is the 143 files importing `refc` / `refc_v3` / the trainer / the sampler / kinematics / ego history / the arm tools.
- **22 files completed on both sides: 409 = 409 passed, 13 = 13 skipped, outcome-identical, 0 regressions.**
- The rest were aborted by the watchdog, or stopped when the fixes landed.

⚠️ **RAM-STARVED, stated rather than hidden.** Other workloads held the dev box at 5.4–6.1 GB available. The brief's floor is 8 GB, and the watchdog aborted or blocked most files.
- **Request, per the coordinator's offer:** run the full `stack/` suite A/B on Thor, tip `ab436ee` vs `ab436ee` + this package's `code/fix/`.
- **Expected:** 0 regressions, and +32 new passes from `test_residual_prior.py`.
- **Caveat:** the `(b)` real-window tests and the `(c)` digest test will SKIP there, with named reasons: no eval kit on Thor, and a platform other than win32.

## 8. Findings outside NEW-1: refcv6 defects met on the way, NOT fixed here

- **F-1 (interacts with NEW-1; handled inside it).** refcv6's sampler rolls its fan from the raw pre-dropout `v_ms` (`refc.py:2522-2523`), while `roll_bank` rolls withheld rows at 10 m/s (`refc.py:2131-2132`). On the ~50 % withheld training rows:
  1. the fan's geometry carries the withheld speed;
  2. `a_star` is chosen on a bank in a different geometry than the fan it indexes (`refc_v3_train.py:3668-3684`), which is the D-REFCV4B-ASTAR-GEOMETRY class.

  MEASURED (`code/probe_f1_withheld_fan_speed.py` → `raw/probe_f1_withheld_fan_speed.json`; smoke arm, the trainer's own build, 4 seeds × 4 rows): an off build and a zero-prior residual build with the same weights have a **bit-identical bank on 16/16 rows** and a **bit-identical fan on the 5 kept rows**, but the fan **differs on 11/11 withheld rows**, by up to 80.99 m at 6 s. The only difference between the two builds is the fan's roll speed, so that difference IS F-1. **An off build keeps it (bit-identity).** Fixing it in off mode changes refcv6, so it is a separate lever for the Master Mind.
- **F-2.** Under `--f3-per-layer`, `decoder.control_head` is BUILT and receives **no gradient** (`p.grad is None`). The cascade's stage heads emit the fan instead (`refc.py:2423-2425`). MEASURED on the TIP with no NEW-1 code (`control_head grad: None`; the cascade heads get 620–1,802). G-LIVE's "every declared-trainable group gets a non-zero gradient" will trip on it, or must name it.
- **F-3.** `ds.ego_history` is set only inside the `if args.v7_labels:` branch (`refc_v3_train.py:7276` at tip). So `--ego-history` without `--v7-labels` refuses at the first batch ("no `pose_hist`"). No synthetic `train()` smoke can exercise ego history. That is why the tests here build through `compute_losses_v3`, as `test_built_heads_receive_gradient.py` does.
- **F-4.** Ego-dropout withholds v0 from the measurement encoder (`refc.py:4195-4200`) but not from the ego-history channel, which encodes v at every past step, v[t0] included (`refc.py:4398-4423`). So refcv6's "speed-blind" withheld rows are not speed-blind. NEW-1 zeroes the prior on those rows, so it does not add a second leak.

## 9. Decisions needed

1. **The refcv7 prior mode** (PI/MM). ✅ DECIDED by SPEC §10 (A5): `ha0_ext_pose`, as recommended here. `cv_yawrate` is not recommended (+0.229 m).
   - **`ha0_ext` needs three things outside this package:**
     1. the κ0@t0 ruling;
     2. a NavSim answer (the bridge declares only `ego_pose` and `ego_velocity`, `refcv6_bridge.py:101-113`, so there is no steer to read);
     3. an EvalFlyWheel edit: the battery proxy must pass `ego_actions=it["actions"]` beside `ego_poses` (`refcv6_roll.py:140-151`). Otherwise the model refuses, loudly.
   - **The pose modes need no caller change anywhere.** One stated caveat: on NavSim the pose window is linearly interpolated from 2 Hz states (`refcv6_bridge.py:124-160`). There, `a0` and `ω0` are 0.5 s segment means rather than 0.1 s differences. That is the same inputs the ego-history encoder already sees there, but it is a statistics shift against training.
2. **The anchor artifact.** No rebuild is needed (§4). An optional metadata restamp is an MM call.
3. **ego-dropout 0.5 in refcv7** (§8 F-1 / F-4). It is inherited unchanged from refcv6. NEW-1 is correct under it, but its purpose is already defeated by the ego history.
4. **The landing order** is unchanged: fixes → NEW-1 (this patch, re-applied on the landed tip) → NEW-2.

## 10. Rebase status: DONE on `ab436ee`, the landed fixes batch

> **Superseded for the base by §11 (2026-09-27).** Revision 2 is regenerated on tip `12953d2`, whose `stack/` code is byte-identical to `ab436ee` for all 8 shared files (same base blobs), with 3 more shared files. The history below stands; `code/diffs_vs_tip_ab436ee/` is kept for the record and is **not** in the landing list (its `refc.py` diff predates §11a).

**How the files were made.** The shared files come from `code/apply_residual_prior_edits.py` run on the RAW `ab436ee` blobs: 28 + 3 + 8 anchored edits, plus 2 in `declared_vs_built.py` and 1 in its test. Every anchor must occur exactly once, or the script FAILS naming it. `code/fix/EDIT_MANIFEST.json` records every base and output blob (copy: `raw/EDIT_MANIFEST_ab436ee.json`).

**Re-running it.** `code/finalize_after_fixes.py <work_root>` is the one command that re-does and re-verifies the rebase:
- fresh `git archive` trees of the tip;
- the patch;
- the package overlay;
- the NEW-1 tests;
- the off-mode digest on both trees;
- the printed landing lines.

The run used here is `raw/finalize_ab436ee.log`. **If the tip moves before landing, re-run it**: a moved anchor FAILS loudly.

**Superset check** against `ab436ee` (`code/diffs_vs_tip_ab436ee/`). Every removed line is an intentional in-place change:
- the extended signatures;
- the redirected `_state_to_path` calls;
- the extended `DECODER_PASSTHROUGH` tuple;
- the `202` → `203` literal.

No tip content is dropped.

**Base blobs = the landed `ab436ee` blobs:**

| file | base blob |
|---|---|
| refc.py | `177d24e92d10…` |
| refc_v3.py | `550d96b2e822…` |
| trainer | `abf5129708b0…` |
| declared_vs_built.py | `1bb1e481d973…` |
| test_declared_vs_built.py | `061238c88970…` |

**The two changes the landing made, versus the pre-checked declared blobs**, are both absorbed:
1. `DECODER_PASSTHROUGH` is now a class tuple (NEW-1's keys go there).
2. The registry grew to 202 entries (NEW-1 → 203).

## 11. The Thor full-suite gate (2026-09-27): 21 RL-stack regressions, both groups fixed, no guard weakened

**The gate.** The Master Mind ran it on Thor: TIP `b3f7ea6` vs TIP + the 7 batch-1 files. The list is `raw/thor_gate_b3f7ea6_new1_regressions.json` (copied from `C:/Users/Admin/qland/work/thorgate_n1/new1_regressions.json`).
- NEW-1's own tests: 29 passed + 3 honest skips (no eval kit on Thor; not win32).
- **21 regressions, all in the RL stack, in two groups.** In both groups the guard that fired was RIGHT.

### 11a. Group 1 (8 tests, `test_ddv2_refc_chain.py`): the DDv2 chain's capture hook did not carry the prior

**Cause.** `tanitad/rl/ddv2_refc_chain.py` swaps `decoder._sample` for a FIXED-ARITY hook, `hook(kv, cond, bank, v_ms, steps, agents=None, agent_pad=None, agent_pos=None, bev=None)` (tip `:103-104`). Batch 1's decoder passed `prior=_rp` to `_sample` on every forward, `None` included. So every capture raised `TypeError` at the call (batch-1 `refc.py:3356`).

**Fix, in two places** (candidate line numbers):
1. **The chain carries the prior faithfully** (`ddv2_refc_chain.py`):
   - `SamplerInputs.prior` (`:85`) records `_sample`'s `prior=` verbatim.
   - The hook (`:119-155`) accepts `prior`, forwards it only when given, and records `v = prior[2]` on a residual build (`:151-152`). That is `_sample`'s own rule (`refc.py:2741-2744`), and on a withheld row it is NOT `v_ms`.
   - `state_to_path` (`:175-206`) rolls a residual build's state (Δ) with the decoder's internal roll, `decoder._roll_state(Δ, v, False, prior)` = `kinematic_prior.roll_plan(Δ, P)`. It never calls `_state_to_path`, which takes EXPORTED absolute controls. It REFUSES three things: a residual build without the prior, an off build with one, and a speed that is not the prior's.
   - `make_x0_fn` (`:209-224`) passes `inputs.prior`. `native_sample` (`:256-262`) mirrors `_sample`'s tail: the fan is Δ rolled on P, and `u0_hat` leaves ABSOLUTE (`_export_controls`).
   - On an off build `prior` is `None`, and every call is the pre-NEW-1 call.
2. **The decoder passes `prior=` only on a residual build** (`refc.py:3356-3365`): `_smp_kw = {} if _rp is None else {"prior": _rp}`. An off build therefore calls `_sample` exactly as refcv6 did. Any wrapper written against the refcv6 signature keeps working on every refcv6 build, and one that cannot carry the prior fails LOUDLY on a residual build.

**Tests** (`test_ddv2_refc_chain.py`, +6). The helpers `_dec`, `_captured` and `_group_independence` gained pass-through keyword arguments; the 12 tip tests are otherwise unchanged.
- `test_NEW1_an_OFF_build_calls_sample_exactly_as_refcv6_did_and_records_no_prior`: a spy on `_sample` sees no `prior=` keyword on an off build and exactly `{prior}` on a residual one. The capture records `None`.
- `test_NEW1_parity_RESIDUAL_the_binding_reproduces_the_deployed_sampler_bitwise`: the recorded prior equals `kinematic_prior.prior_from_out(out)`, tensor for tensor. `native_sample` reproduces `_sample`'s fan AND its exported `u0_hat` bit for bit.
- `test_NEW1_parity_RESIDUAL_holds_on_a_WITHHELD_row_where_the_prior_speed_is_not_v0`: the withheld row's prior is zero and rolled at 10 m/s, not at v0 = 3.5. Parity still holds bit for bit. The pre-NEW-1 rule, `v = v_ms`, would break exactly here.
- `test_NEW1_REGRESSION_a_chain_that_rolls_Delta_without_the_prior_breaks_parity`: the RED arm. The chain's roll gets a zero prior, and the fan differs.
- `test_NEW1_REGRESSION_a_capture_that_loses_the_prior_is_REFUSED_not_rolled`: a dropped prior, a wrong speed, and a prior handed to an off build are each refused.
- `test_NEW1_groups_are_independent_queries_on_a_RESIDUAL_build`: the G×N exactness claim of the module docstring holds on a residual build.

**Not changed, and why: `scripts/ddv2_rl_refcv5.py`.** It is bound to the refcv5-v2 build. Its `score_batch` integrates controls with its own integrator (`P.ego_states_from_controls`, `:276-279`), so it is NOT residual-aware. It cannot get that far on a residual build, for three reasons:
- its capture forward (`:268`) supplies no `ego_poses`;
- a residual build requires the ego-history encoder (`refc.py:3872-3878`);
- an ego-history build refuses a forward without `ego_poses` (`refc.py:4731-4736`).

So it fails loudly at its first forward. Making it residual-aware is a separate task, needed only if an RL arm on a refcv7 build is ever planned.

### 11b. Group 2 (13 tests, five RL files): `ego_actions` was declared nowhere

**Cause.** NEW-1 added `ego_actions` to `RefCModel.forward` and `RefCV3Model.forward`. Both RL adapters derive their channel set from the live signatures, and both refuse an undeclared channel:
- `refcv3_adapter.forward_conditioning_channels` raised `RequirementDeclarationError` (`refcv3_adapter.py:638-658`);
- `refc_adapter`'s two-way pin, `FORWARD_KEYS == signature − seam exclusions`, went RED in `test_rl_forward_keys_cover_signature.py`, `test_rl_refc_adapter_robust.py:226` and `test_rl_channel_guard.py:256-288`.

**Fix: declared by the seam that owns the channel, as must-not-be-plumbed (TEMPORARY).**
- `kinematic_prior.FORWARD_EXCLUSIONS` (`kinematic_prior.py:190-257`) is one `ChannelExclusion` with an owner, a reason, a route back (`unblock`) and evidence. Its constructor refuses any of them missing or a placeholder (`channel_admissibility.py:119`).
- `channel_admissibility.SEAM_MODULES` gains `"tanitad.models.kinematic_prior"` (`channel_admissibility.py:183-188`).
- **`FORWARD_KEYS` is unchanged, and no guard's code changed.** Both adapters now subtract the exclusion (`refcv3_adapter.py:638`; the tests' `_required_channels`). Every "undeclared channel ⇒ REFUSE" path is untouched, and a mutation test proves both still fire (below).

**⛔ Why an exclusion: the design decision inside it (ESCALATED, §0).** The brief asked for "required only when `residual_prior == "ha0_ext"`, otherwise declared must-not-be-plumbed". The machinery cannot express a per-build exclusion:
- `excluded_channels()` (`channel_admissibility.py:240`) is ONE list for every build;
- it is subtracted before either adapter consults its requirement table.

So a channel is EITHER plumbed and declared, with a truthy dotted-path predicate on the config, OR excluded for every build. I chose the exclusion, for three reasons:
1. **SPEC §10 (A5) excludes `ha0_ext` for refcv7** ("no PI ruling on the steer channel at inference"). refcv7 sets `ha0_ext_pose`, which reads only the pose track the adapter already plumbs (`ego_poses`, `ego_n_past`). No seam used by any sanctioned build requires `ego_actions`, so by the brief's own rule ("plumb it in FORWARD_KEYS only if a seam requires it") it is not plumbed.
2. **The precedent is exact:** `perception_grid` / `perception_valid` are excluded as "TEMPORARY and NOT a label", an unplumbed supplier (`refcv6_perception_branch.py:77-135`).
3. **"Required when ha0_ext" is enforced one layer down, loudly.** An `ha0_ext` forward without actions refuses (`kinematic_prior.py:317-323`: "Refusing rather than substituting the pose curvature"). A supplied `ego_actions` on any other build refuses (`refc.py:4749-4755`: "SILENTLY DROPPED"). Both refusals are pinned by `test_refusals_at_build_and_at_forward`. So an RL rollout of an `ha0_ext` build stops at its first forward; it cannot run blind.

**The alternative, if the Master Mind prefers it:** plumb `ego_actions` in `FORWARD_KEYS` and declare it in both requirement tables, with a predicate that is true only for `ha0_ext`. That needs:
- a boolean view of the mode for the predicate machinery (for example a read-only `DecoderConfig` property);
- the pinned declaration count `test_rl_refcv3_used_path_guard.py:458` moved from 18 to 19.

The route is written in the exclusion's `unblock`.

**Tests** (`test_residual_prior.py`, +2):
- `test_rl_ego_actions_is_declared_must_not_be_plumbed_by_the_seam_that_owns_it`. The seam is listed, and the declaration is the module's own. It is TEMPORARY and names `ha0_ext` in its reason and its route back. It is absent from `FORWARD_KEYS`, and `signature − exclusions − FORWARD_KEYS` is empty.
- `test_rl_MUTATION_without_the_declaration_BOTH_adapters_go_RED`. With the seam removed from `SEAM_MODULES`, `refc_adapter`'s comparison reports exactly `{ego_actions}` missing, and `refcv3_adapter` raises `RequirementDeclarationError` naming it. Those are the two failures the Thor gate reported. With the seam restored, both are quiet.

### 11c. Verification on clean trees of `12953d2`

**No local test file ran. Stated, not hidden.**
- **The instrument was armed.** Clean `git archive` trees of `12953d2` were built: `C:/Users/Admin/r7rp/rebase3/tip_12953d2d23` and `…/cand_12953d2d23` = tip + the 10 files. `tanitad.__file__` was asserted inside each tree. `code/suite_ab_perfile.py` held 19 files: the 6 RL files the gate named, `test_residual_prior.py`, `test_ddv2_refc_chain.py`, and 11 more that import the changed modules. It ran one pytest process per file per tree, with a watchdog at the brief's 8 GB floor (start only at >= 9 GB available).
- **It never started.** Available RAM stayed at 6.0-7.7 GB from 00:22 to 01:30. The main holder was another agent's battery roll (`a6_roll.py`, seed 1, 7.7-8.0 GB private); other jobs kept starting. At ~01:30 the Master Mind stood the run down. Its processes were stopped by explicit PID after checking their command lines, and `ab_rl/` is empty.
- **What DID complete, all static:**
  - the anchored patch applied to the RAW `12953d2` blobs with every anchor found exactly once (28 + 3 + 8 + 2 + 1 + 10 + 4 + 1 edits; `code/fix/EDIT_MANIFEST.json`);
  - the four shared files this round did not touch regenerate BYTE-IDENTICAL to the Thor-gated batch-1 blobs;
  - the regenerated `refc.py` differs from batch 1 only by the §11a call site (diffed);
  - every changed file parses (`ast`);
  - the candidate tree's 10 overlay files equal `code/fix/` blob for blob;
  - the new tests were desk-checked name by name against the modules they call. That is a reading, not a run.
- **The definitive run is the Master Mind's Thor full-suite gate:** tip `56ae4eb` vs tip + these 10 files, 521 + 1 stack test files and 7 taniteval files. Its expected outcome, as a prediction and not a result:
  - the 21 regressions clear;
  - +6 passes in `test_ddv2_refc_chain.py`, +2 in `test_residual_prior.py` (34 there, the real-window and digest tests skipping on Thor as before);
  - 0 new failures.
- **If any NEW test fails there, suspect the test code first:** it has never been executed.

### 11d. Blobs: what changed since the Thor-gated batch 1 (for the fixes agent's batch 2, via the Master Mind)

| file | batch 1 (Thor-gated) | revision 2 | |
|---|---|---|---|
| `stack/tanitad/refs/refc.py` | `cf80722edc13…` | `6638d443533f…` | **changed** (§11a, the call site) |
| `stack/tanitad/refs/refc_v3.py` | `774d3a3b1d5f…` | `774d3a3b1d5f…` | same |
| `stack/scripts/refc_v3_train.py` | `9cac3dabca2b…` | `9cac3dabca2b…` | same |
| `stack/tanitad/train/declared_vs_built.py` | `a50ff690a364…` | `a50ff690a364…` | same |
| `stack/tests/test_declared_vs_built.py` | `5137fde792a8…` | `5137fde792a8…` | same |
| `stack/tanitad/models/kinematic_prior.py` (NEW) | `9d4895610acc…` | `3438b8f7ca7b…` | **changed** (§11b) |
| `stack/tests/test_residual_prior.py` (NEW) | `e53b1ed3377a…` | `68970c38bcf1…` | **changed** (+2 tests) |
| `stack/tanitad/rl/ddv2_refc_chain.py` | tip `1f387a4acd6c…` | `5a37c67fae4d…` | **new to the set** (§11a) |
| `stack/tests/test_ddv2_refc_chain.py` | tip `2282f8590147…` | `e080cd6d70ff…` | **new to the set** (+6 tests) |
| `stack/tanitad/channel_admissibility.py` | tip `383ae777e391…` | `e4e7fd814258…` | **new to the set** (§11b, +1 seam) |

The full 40-character hashes are in `LANDING_READY.txt` and `code/fix/EDIT_MANIFEST.json` (copy: `raw/EDIT_MANIFEST_12953d2.json`). The four unchanged files are byte-identical to what the Thor gate ran.

### 11e. Open: SPEC §10 (A5)'s G-DVB clause is NOT built here

A5 says: *"The chosen argv: `--residual-prior ha0_ext_pose`. It requires `--ego-history`, and G-DVB refuses any other mode for a refcv7 launch."*
- `check_refcv7_required` (`declared_vs_built.py:951-974`) checks only the three selection mechanisms (`REFCV7_REQUIRED_ON`, `:947-948`).
- The missing clause: argv `--residual-prior` AND the built decoder's `residual_prior` must both equal `ha0_ext_pose`, and `--ego-history` must be set.
- Its existing tests (`test_declared_vs_built.py:425-450`) build refcv7 models without the prior. Each would need `--ego-history --residual-prior ha0_ext_pose` added.
- **Why it is not added here.** The fixes agent's batch 2 is rebased on my `declared_vs_built.py` and owns this function, and this round was scoped to the 21 regressions. Adding it here would put two agents in one function.
- **Needed from the Master Mind: an owner.** It can be me, as a follow-up after this re-gate, or batch 2.
