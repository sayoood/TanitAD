# refcv6 ADVERSARIAL REVIEW — THE DIFFUSION DECODER AND ITS CONFORMANCE TO DiffusionDrive

**Reviewer:** independent adversarial reviewer, dimension = *diffusion decoder + drive-diffusion
paper conformance*. **Date:** 2026-09-22. **Repo:** `D:/Projects/TanitAD` @ `37645fc`
(`agent/arch-inf-20260803`). **Compute:** CPU only.

**Authority read:** `Project Steering/SPEC_REFCV6_V2.md` §1 / §3 / §9 · `Project Steering/
ADVISORY_FROZEN_TRUNK_DEFECT_CLASSES.md` classes **C** and **D**.

**Primary sources used (PRIMARY, both banked):**
* **PUBLISHED (primary, banked)** — DiffusionDrive, arXiv `2411.15139`, library key `2411.15139`,
  sha256 `6ad4f8a379494eeb099cf7d489b75bec1c0ec3321428bcd0b0ca0f05662b8600`,
  `TanitAD Research Lab/Library/papers/2411.15139_DiffusionDrive-…pdf`.
* **PUBLISHED (primary, released SOURCE, banked)** — `TanitAD Research Lab/Architecture &
  Inference/Research/2026-09-05-diffusiondrive-v2-analysis/raw/ddv2_src/` —
  `v1/transfuser_model_v2.py`, `multimodal_loss.py`, `blocks.py`. **Where paper prose and released
  code disagree the code is quoted**, which is also `refcv6_diffusion.py`'s own stated rule.
  ⛔ No number below is cited to an aggregator summary.

---

## 0. VERDICT IN ONE TABLE

| # | finding | class | severity |
|---|---|---|---|
| **1** | ⛔⛔ **DiffusionDrive coupling (1) — BEV at the candidate's own waypoints — is BUILT (3,210 params/decoder at d=32) and NEVER CALLED.** Forward-hooked: **0 fires** in the sampler *and* the classifier, with `bev` handed straight to `AnchoredDiffusionDecoder.forward`. Broken at **three independent points**. | **C** | **BLOCKER for the §1 claim** |
| **2** | ⛔ **On the DEFAULT configuration the ranked score is BLIND to the emitted trajectory.** Perturbing the sampler moved `traj` by **12.20 m** and `sel_score` by **exactly 0.0**; `sel_idx` unchanged. `--f5-emitting-conf` fixes it (`sel_score` +0.610, `sel_idx` 0,0→1,4) **and was OFF in the arm that validated the pipeline.** | **C** | **HIGH** |
| **3** | ⚠️ **SPEC §3's *"F1…F9, all implemented"* is true of the CODE and false of the ARM.** The pipeline-validation run's own record stamps `f5_emitting_conf false · f7_samples_per_anchor 1 · f8_flat_waypoint_noise false`. **6 of 9 ran.** | provenance | **HIGH** |
| **4** | ⚠️ **`--f7-ack-eval-join` acknowledges something that is not true.** `sel_anchor_id` is emitted (`refc.py:3259`) and has **zero consumers** outside `stack/`; `taniteval/tools/t1_eval.py` still joins `<arm>_sel_idx`. | **F #5** | MEDIUM (blocks F7) |
| **5** | ⚠️ **Every v0-conditioned guard tests the DECLARATION, not the TENSOR.** With `anchor_v0_cond=True` and `anchor_controls` all zeros the forward does **not** raise, F9 **passes**, and the bank is **exactly degenerate** (spread **0.000000000 m**, all N candidates one straight line). Reachable from argv. | **F #1/#4** | MEDIUM |
| **6** | ⚠️ **The bit-identity guard's baseline has rotted to a tautology** — it materialises "pre-refcv6" from `git show HEAD:…/refc.py`, whose blob **equals the worktree** and carries **83** "refcv6" mentions. ⭐ **Re-run against the TRUE baseline `8c7d215^` the claim still HOLDS, bit for bit.** | **F #1** | MEDIUM (guard only) |
| **7** | ⚠️ **F8 clamps once, outside the ladder; DD clamps inside every iteration** (`transfuser_model_v2.py:519`). MEASURED `\|x_n\|` reaches **15.04** after `sched.step`, 15× outside DD's box. | conformance | LOW–MED (F8 unrun) |

**Everything that is RIGHT and was checked, not assumed:** F1, F2, F3 (incl. the **detach**, proven
with a control), F4 (incl. DD's `Mish`-first order and `zero_init=False` default), F5's focal (exact
to 8 dp against an independently written reference), F6, F9's assertion, DD's `norm_odo` box
(literal-for-literal against the released source), the BEV grid map (analytic, 8e-9), and **all six
class-D unit identities against analytic targets, each with its historical-defect mutation RED**.

---

## 1. Q1 — F1…F9: WHICH ARE ACTUALLY IMPLEMENTED, AND WHICH ARE LIVE?

Instrument: `code/diag_f1_f9_liveness.py` + `code/diag_f3_f4_zeroinit.py`.
Raw: `raw/f1_f9_liveness.json`, `raw/f3_f4_zeroinit.json`, `raw/f3_stage_slice.json`.
**Every flag is proven by EFFECT, MUTATION or REFUSAL — never by the presence of an `if`.**

| F | file:line (build / consume) | MEASURED liveness | DD primary | live? |
|---|---|---|---|---|
| **F1** | `refc.py:2536-2538` · draw `refc_sampler.draw_train_timesteps` | training `refcv6_pairs = [[null,null]]` = **ONE call**, eval keeps the ladder; **48 draws, min 0 max 48, 30 distinct**; **MUTATION** `f1_t_max=1` → support collapses to **[0]** | `transfuser_model_v2.py:465-468` `randint(0,50)`, one `diff_decoder` call `:489` | ✅ |
| **F2** | `refc.py:2551-2553` · `dd_step_pairs` | ON `[[10,9],[0,-1]]` vs OFF `[[10,0],[0,0]]`; residual retained **0.9475** vs **0.2828** (from the alpha table, not asserted) | `:507` `set_timesteps(1000)` ⇒ diffusers `prev_t = t − 1000//1000` | ✅ |
| **F3** | build `refc.py:1987-1988` · emit `refc.py:2565-2569` · loss `refc_v3_train.py:3862-3888` | 4 stages exported; **MUTATION** stage-L head moves `traj` **17.40 m**, stage-0 head moves `traj` **0.0** but moves `layer_u0_hat[0]`/`[2]` by **4.0** (correct: only the last stage emits). **DETACH PROVEN:** backward from `layer_u0_hat[-1]` → `layer1.cross.out_proj` grad **712.82**, `layer0` grad **None**. ⭐ **CONTROL:** with the cascade OFF the same backward reaches layer0 at **421.67**, so the detach is what cut it | `:345-347`, `:379` (detach), `:492-497` (sum over the list) | ✅ |
| **F4** | build `refc.py:1989-1992` · apply `refc.py:2387-2388` | **MUTATION** `adaln[0]` bias +0.5 → `traj` **+10.14 m**, `adaln[1]` → **+9.51 m**; module order `[Mish, Linear]`; `f4_zero_init` default **False** (= DD); zero-init arm is an **exact** identity (delta 0.0) | `ModulationLayer :229-268`, applied `:337`; `if_zeroinit_scale=False :233` | ✅ |
| **F5** | conf `refc.py:3040-3053`, `:3067` · focal `refc_v3_train.py:3505-3508`, `:3885-3887` | focal **0.14010201** = independently written reference **0.14010201** (Δ < 1e-9; CE on the same logits is 2.206767, i.e. a different loss). Ranking: see §3 | focal `multimodal_loss.py:146-157` γ2.0 α0.25 `reduction='mean'`; same-pass emit `:554-557` | ✅ |
| **F6** | `refc_v3_train.py:386-391` | `refcv6_flags_from_args` returns the block; the pin sets `args.w_u0 = 0.0` **and** `args.ack_ddim_no_u0 = True` | `multimodal_loss.py:161` — DD's ONLY reconstruction loss is `F.l1_loss(best_reg, target_traj)` | ✅ |
| **F7** | refusal `refc.py:2778-2793` · widen `:2828-2831`, `:3258-3261` | refuses without the ack ✅; with it the fan is **[2, 12, 4, 2] = 2×N**, `sel_anchor_id` emitted, bank **group-major verified** (`bank[:, :N] == bank[:, N:2N]`) | Tab. 6 (N 20→40 = +0.1 PDMS) | ✅ code / ⛔ **consumer missing** — §4 |
| **F8** | refusal `refc.py:1993-1999` · norm `:2469-2473` · clamp `:2548-2549` | refuses on control space ✅; on `--sampler-space metre` `traj` differs; box literals **exact** vs released source; round-trip 2.4e-6; σ **0.8987 m (x) / 0.7265 m (y)** flat per waypoint | `norm_odo :432-441`, clamp `:474` (train) and `:519` (test) | ✅ **with a clamp gap** — §7 |
| **F9** | `refcv6_diffusion.py:446-466` · call `refc.py:2796-2799` | rejects n=116 and `v0_conditioned=False`; **reached from the forward** (raised on a 6-anchor build) | — (assert-only) | ✅ **but blind to the tensor** — §5 |

⚠️ **AN INSTRUMENT ARTIFACT THAT OTHERS WILL HIT, SO IT IS PUBLISHED.** My first F3/F4 arms read
`traj` delta **exactly 0.0** and looked like dead wires. They are not. `control_head` and every
`CascadeHeads.control_head` are **zero-init** (`refc.py:1705-1706`, `refcv6_diffusion.py:304-305`),
so at construction `du = 0`, `u0_hat = x_n`, and **the emitted trajectory is the anchored Gaussian,
independent of every decoder weight.** Any liveness probe that mutates an upstream module and
watches `traj` reads 0.0 on a perfectly wired model. ⭐ This is the advisory's *"a control that
cannot come out the other way is not a control"* with the sign flipped, and it is why
`diag_f3_f4_zeroinit.py` un-zeroes the emitting head before the same mutation.

### 1.1 ⚠️ The gap between "implemented" and "ran" (finding 3)

SPEC §3's heading is *"Diffusion — F1…F9, all implemented"*. The **run record** of the arm that
carried §10.6's pre-pod pipeline validation
(`…/Research/2026-09-17-refcv6-e2e-1024/raw/assembly.json`, `argv` + the `refcv6` stamp) reads:

```
f1_random_t true · f2_dd_step true · f3_per_layer true · f4_adaln true
f5_emitting_conf FALSE · f5_focal true · f6_w_u0_zero true
f7_samples_per_anchor 1 · f8_flat_waypoint_noise FALSE · f9_assert_vocab true
```

⇒ **MEASURED: six of nine.** `--f5-emitting-conf` is absent from the launch line; only `--f5-focal`
is there. The RESULT.md's own prose says *"refcv6 {f1..f6, f9 = true}"*, which reads as "F5 on" and
is why this needed the JSON rather than the summary. **F5's emitting half is precisely the one that
closes finding 2** (§3), so the validated arm is the one that ranks the anchor bank.

### 1.2 ⚠️ F3 without F1 supervises an emitting pass as an intermediate stage

`refc_v3_train.py:3863` takes `stages = out["layer_u0_hat"][:-1]`. MEASURED (`raw/f3_stage_slice.json`):

| arm | stages exported in TRAINING | slice `[:-1]` | stages that are an **emitting** pass |
|---|---|---|---|
| F3 only (2 layers, 2-step ladder) | 4 | 3 | **[1]** — pass 1's final layer |
| **F3 + F1 (the validated arm)** | 2 | 1 | **[]** ✅ |

Under F1 the slice is DD-correct (DD trains with ONE pass, `:489`). ⇒ **not a defect in the
validated arm; a note that F3 must not be run without F1.**

---

## 2. Q2 — THE THREE COUPLINGS, HOOKED AND PRINTED

Instrument: `code/diag_decoder_couplings.py`, `code/arm_bev_controls.py`, `code/arm_c2_force.py`.
Raw: `raw/couplings_measured.json`, `raw/bev_coupling_controls.json`, `raw/bev_c2_forced.json`.
Build: 2 layers, `d=32`, `sampler=ddim`, agents ON, **WP-B attached (840 params)**, **BEV coupling
attached (3,210 params)**, `bev` passed **directly** into `AnchoredDiffusionDecoder.forward`.

| SPEC §1 coupling | module | hook fires | context shape | context `data_ptr` |
|---|---|---|---|---|
| **(3) image tokens by content** | `CrossAttnLayer.cross` (`refc.py:1554`) | **3 / layer** (1 classifier + 2 denoise passes) | `(2, 16, 32)` = the flattened conv map | `5636742995968` |
| **(2) agent slots addressed by waypoint** | `CrossAttnLayer.cross_agent` (`refc.py:1556-1557`) via `_agent_bias` | **3 / layer** | `(2, 5, 32)` = 5 agent slots | `5636744020480` |
| **(1) BEV at the candidate's own waypoints** | `CrossAttnLayer.bev_wp` (`refc.py:1563-1564`) | ⛔ **0** | — | — |

⭐ **Class-C pointer assertion (the advisory's own instruction):** the image and agent contexts have
**different storage pointers** (`ptr_disjoint_image_vs_agent = true`). ⇒ **No REFe-style
`cat([tokens, registers])` defect here**; the two modules really do read different tensors, and the
context sizes are 16 image tokens and 5 agent slots, not one concatenated 21. **Couplings (2) and
(3) match the design.**

### 2.1 ⛔⛔ Coupling (1) is built and never called — the discriminating controls

A hook that fires zero times is indistinguishable from a hook that was never installed, so:

| arm | `bev_wp` fires | what it discriminates |
|---|---|---|
| **C1** direct `dec.layers[0].bev_wp(q, wp, bev)` | **1**, ctx `[2, 12, 120, 64]` | ⭐ the instrument works |
| **C2** decoder forward, `sel.anchor_prefilter=True`, `v=0.05` and `v=1.0` | **2** (one per layer), ctx `[2, 12, 120, 64]` | ⭐ the decoder *can* call it — `refc.py:2866` is the one site that forwards `bev` |
| C2′ same inputs, `v=8.0` (prefilter branch not taken) | **0** | the branch, not the probe |
| **C3** decoder forward, sampler `steps=2`, `bev` passed | **0** | **THE DEFECT** |
| **C3′** decoder forward, classifier `steps=0`, `bev` passed | **0** | **THE DEFECT** |
| C4 same as C3 with `bev` **not** passed | **0** | identical to C3 ⇒ passing `bev` changes nothing |

### 2.2 The three breaks, by AST over `refc.py`

**Break A — the sampler never receives it.** Only **2 of 7** `_decode` / `_decode_ctrl` / `_sample`
call sites carry `bev`:

| line | callee | carries `bev` | on the default path? |
|---|---|---|---|
| 2563 | `_decode_ctrl` | ✅ (forwards what `_sample` got) | — |
| 2866 | `_decode` | ✅ | only inside `sel.anchor_prefilter` (default **False**, `refc.py:664`) |
| 2878 | `_decode` | ⛔ | prefilter, `k >= n` branch |
| 2884 | `_decode` | ⛔ | **YES — the default classifier pass** |
| 2966 | `_decode` | ⛔ | the pre-v5 refine loop |
| **2997** | **`_sample`** | ⛔ | **YES — the whole diffusion path** |
| 3057 | `_decode` | ⛔ | `--sel-score-emitted` |

`refc.py:2997-2999` passes **7 positional arguments**; `_sample`'s `bev` is the **8th parameter**
(`refc.py:2439`), so it defaults to `None` and `_decode_ctrl`'s correct-looking `bev` forward at
`:2563` always forwards `None`.

**Break B — no caller ever supplies it.** `RefCModel.forward` declares `bev` (`refc.py:3848`) and
forwards it (`refc.py:4270`, the **only** `bev=` keyword call in the whole `stack/` tree, AST-scanned).
`refc_v3.py::RefCV3Model.forward`'s two `self.core(...)` calls (`:2043`, `:2098`) pass
`scene_hook` / `bev_hook` / `bev_tokens` — the **§4 tactical** path — and **never `bev`**.

**Break C — no config route.** `DecoderConfig.bev_coupling` (`refc.py:573`) is read at
`refc.py:3640-3643` and assigned in exactly **one** file in the repository: `integration/
verify_patch.py:80`. `refc_v3_train.py` has **no `--bev-coupling` argument and no assignment**, so
no training run can build the sampler at all. *(Absence probed three ways: AST assignment scan over
every `**/*.py`; `grep bev_coupling` over `stack/scripts`, `stack/tanitad`, `taniteval`; and
`find -name "*bev_coupling*"`.)*

### 2.3 Why it survived: the module is tested, the CONSUMER is not

⚠️ **Correction to a first reading of mine, stated because the first reading was wrong.** The module
docstring (`refc_bev_coupling.py:58`) cites `tests/test_refcv6_bev_coupling.py`, **which does not
exist**; the tests are in **`stack/tests/test_refcv6_perception.py:394-518`** (9 tests, including
mutation arms for the `[..., [1, 0]]` transposition and for mirroring, the 64-window gated-off
bit-identity, and a gate-is-gated-not-dead arm). They are good tests. ⛔ **Every one of them
constructs `BEVWaypointSampler` standalone and calls `s(q, wp, bev)` directly** — none runs it
through `AnchoredDiffusionDecoder.forward`. That is the advisory's `diag_consumer_conformance`
lesson verbatim: *"an expectation is a LITERAL, or it is read from a DIFFERENT consumer"*. The
module is proven correct in isolation while its consumer never calls it.

⭐ **THE MISSING GUARD IS WRITTEN AND RUN:
`code/guard_coupling1_reaches_the_decoder.py` → `raw/guard_coupling1.json`.**

```
A_sampler_passes     expected >= 4  measured 0   RED
B_classifier_pass    expected >= 2  measured 0   RED
C_direct_call        expected    1  measured 1   GREEN  (control)
D_prefilter_branch   expected >= 2  measured 2   GREEN  (control)
VERDICT: GUARD IS RED ON HEAD (the defect is real)
```

**What to fix, smallest change first:** pass `bev` at `refc.py:2997` (into `_sample`) and at
`refc.py:2884` (the default classifier `_decode`); then add a `--bev-coupling` route in
`refc_v3_train.py` writing `cfg.core.decoder.bev_coupling`; then wire `bev=` from
`refc_v3.py`'s `self.core(...)`. Land the guard above beside them. Fix `refc_bev_coupling.py:58`'s
test filename in the same edit.

### 2.4 ⭐ THE FIX, SIMULATED AND MEASURED — so whoever lands it is not guessing

*Rule Zero: a refutation is a waypoint. `code/fix_probe_coupling1.py` →
`raw/fix_probe_coupling1.json`.* The two missing arguments were injected by wrapping the two bound
methods — **nothing in `stack/` was modified** — and the result measured:

| | before | after the 2-argument fix |
|---|---|---|
| `bev_wp` fires, sampler (`steps=2`) | **0** | **6** = 2 layers × (1 classifier + 2 denoise passes) |
| `bev_wp` fires, `steps=0` | **0** | **6** *(a `ddim` build runs the sampler either way — `_sample` falls back to `cfg.sampler_steps`)* |
| context / waypoint shapes | — | `bev [2, 12, 120, 64]` · `wp [2, 6, 4, 2]` — the candidate's own waypoints |
| **removability** (zero-init gate) | — | `traj` **BIT-IDENTICAL** to the pre-fix plan, at a pinned inference seed |
| **gated, not dead** (gate → 1.0) | — | `traj` moves **3.171486 m** |

⇒ **The fix restores coupling (1) and costs the banked baseline nothing.**

⚠️ **Two traps this probe walked into, published because the next person will too.**
**(i)** The first run read `BIT_IDENTICAL: false` — the sampler draws a fresh `eps` every forward
(`refc.py:2538`), so a bit-identity claim across two unseeded forwards measures **inference noise**.
That is `CLAUDE.md`'s *third* variance in miniature, and it made a correct fix look wrong.
**(ii)** With the seed pinned, opening the gate then moved the plan by **exactly 0.0** — the zero-init
`control_head` bottleneck of §1 again. Only after un-zeroing the emitting head does the 3.17 m
appear. **Both arms are kept in the instrument so the artifact shows the reasoning, not just the
answer.**

---

## 3. Q3 — IS THE SCORING BRANCH FED THE CANDIDATE TRAJECTORY OR THE PROPOSAL?

Instrument: `code/diag_q3_scoring.py` → `raw/q3_scoring_surface.json`. Method: un-zero the emitting
head (§1's artifact), perturb **only the sampler**, and ask whether the ranked surface moves. A
ranker that scores the emitted path must move; one that scores the anchor bank cannot.

| arm | `traj` moved | `sel_score` moved | `sel_idx` | ranked surface sees the sample |
|---|---|---|---|---|
| **DEFAULT** (`sel.refined=False`, no F5) | **12.20 m** | **0.000000** | `[0,0]` → `[0,0]` | ⛔ **NO** |
| `--f5-emitting-conf` | 12.20 m | **0.609841** | `[0,0]` → **`[1,4]`** | ✅ yes |
| `--sel-refined` | 4.66 m | 0.094031 | `[0,0]` → `[0,0]` | ✅ yes |
| F3 + F5 | 52.31 m | 1.416618 | `[5,4]` → **`[2,1]`** | ✅ yes |

`SelectionConfig.refined` and `.score_emitted` both default **False** (`refc.py:637-638`), and the
decoder's own telemetry says so: `sampler_ranks_the_fan: false`.

⇒ **ANSWER: on the default configuration the ranked score is computed from the CLASSIFIER pass over
the raw anchor bank (`refc.py:2884`, `x0 = bank`) — the PROPOSAL — and is provably blind to the
denoised candidate.** It is the same shape as REFe's *"the scoring branch was fed the proposal
queries rather than the candidate trajectory — judging intent instead of path"*, in a different
architecture. The decoder documents it honestly at `refc.py:2949-2961` (S1b) and at
`refc.py:2607-2618`, and F5 is the flag that closes it:
`base = refined if (sel.refined or self.rv6.f5_emitting_conf) else conf` (`refc.py:3067`).

⛔ **It is live in the validated arm.** §1.1: `f5_emitting_conf false`, `--sel-refined` absent.

**PUBLISHED (primary):** DD gathers the emitted trajectory by the argmax of the classification head
of the **same** pass — `mode_idx = poses_cls.argmax(dim=-1); best_reg = gather(poses_reg, mode_idx)`
(`transfuser_model_v2.py:554-557`, test) and `:497-501` (train, over `poses_*_list[-1]`). Our F5 path
does exactly this: `_decode_ctrl` returns `(conf, du)` from one pass and `u0_hat = x_n + du`
(`refc.py:2566`). **F5 ON is DD-faithful; F5 OFF is not.**

---

## 4. Q4 — UNITS, RATES AND TIME BASES (CLASS D)

Instrument: `code/diag_units_identities.py` → `raw/units_identities.json`. **Every identity uses an
ANALYTIC target** — a circle's radius is exactly `1/κ`, a straight plan's lateral offset is exactly
`0`, a constant-acceleration arc length is exactly `v₀t + ½at²`, a cell centre maps to exactly
`(2k+1)/n − 1`. None is a rearrangement of the code under test.

### 4.1 The derived-quantity ledger — rate/unit and its OWNER

| derived quantity | site | unit | rate it assumes | who OWNS that rate |
|---|---|---|---|---|
| anchor bank geometry | `refc.py:2124` `rollout_unicycle(..., dt=self.anchor_dt)` | m | **0.1 s** | `AnchoredDiffusionDecoder.anchor_dt`, a **literal** at `refc.py:1701` |
| slot time | `refc.py:1712-1715` `anchor_slots = k − 1` | ticks | 0.1 s | `TrajectoryConfig.horizons` (`refc.py:420`), **in ticks** |
| curvature from lateral accel | `refc.py:2131-2134` `κ = a_lat / max(v, floor)²`, clamped | 1/m ← m/s² | — | `anchor_controls` **units declared in the artifact** |
| selection band horizon | `refc.py:1035` `horizon_s = max(horizons) * 0.1` | s | **0.1 s** | a second **literal** |
| slot spacing | `kinematic.slot_dts(horizons, tick=0.1)` | s | **0.1 s** | a third **literal** (a default arg) |
| artifact↔config check | `refc_v3_train.py:4636-4637` `horizon_s=max(hz)*0.1, dt=0.1` | s | **0.1 s** | a fourth **literal**, compared against the FILE's declared `dt` |
| max-speed ladder | `refcv6_max_speed.py:94-95` `s / 3.6` | m/s ← km/h | — | `SPEED_MAX_STEPS_KMH_V6` (road-law integers) |
| DD waypoint box | `refcv6_diffusion.py:360-361` | m | — | DD's `norm_odo` |
| BEV sample grid | `refc_bev_coupling.py:105-109` | grid ← m | — | `bev_raster.GRID_DEFAULT` (60 m × ±16 m, 0.5 m) |

⚠️ **The 0.1 s tick is written as a LITERAL at four sites** (`raw/units_identities.json` →
`tick_ownership.tick_literal_sites`). No single object owns it. The trainer's artifact check at
`refc_v3_train.py:4636` *is* a genuine cross-check — the file's declared `dt` is independent — but it
compares against the **trainer's** literal while the decoder integrates with **`refc.py`'s** literal.
If one moved the check would not see it. **Recommend: a module constant read by all four.**

### 4.2 The identities, and their historical-defect mutations

| identity | analytic reference | MEASURED | mutation (the real historical defect) |
|---|---|---|---|
| **curvature → radius** | circumradius of 3 path points | κ 0.02/0.05/0.10 → R **50.0012 / 20.0030 / 10.0060 m** vs 50/20/10; max rel err **6e-4** | κ = 0 ⇒ lateral **exactly 0.000000000000 m** (control reads the known value) |
| **lateral accel** | `a_lat = v²/R` | declared 1/2/3 m/s² → measured **1.00000 / 1.99998 / 2.99992**; max rel err **2.6e-5**; **0.102 / 0.204 / 0.306 g** | ⛔ **the 2026-09-04 defect re-introduced**: same tensor read as **curvature** at 36 m/s ⇒ implied **556.39 m/s² = 56.72 g**, **185× the declared 3.0** → **RED**. *(The post-mortem's headline 396 g is the un-capped closed form `v²κ`; the integrator's `kappa_cap` and a 2 s circumradius give a smaller but equally disqualifying number. The identity fires either way.)* |
| **longitudinal accel** | `v₀t + ½at²` | max abs err **0.0999 m** at t = 2 s, a = ±1 (forward-Euler residual, reported not hidden); `a = 0` rows are **exact** | ⛔ **dt = 0.2 s** ⇒ last slot **40.0 m** vs **20.0 m**, ratio **exactly 2.0** — the advisory's class-D row 2 |
| **max-speed ladder** | road-law integers ÷ 3.6 | `{30,50,100,120}` → `{8.333333333333334, 13.88888888888889, 27.77777777777778, 33.333333333333336}`; max err **0.0** | ⭐ **not** rounded to 4 dp — the 57.5 %-bucket-shift defect is **absent here** |
| **DD waypoint box** | released source literals | ours `x (1.2, 56.9)`, `y (20.0, 46.0)` vs `2*(x + 1.2)/56.9 - 1`, `2*(y + 20)/46 - 1` | round-trip max err **2.4e-6** |
| **BEV grid map** | `_cell_centers` + `align_corners=False` | 4 probed cells incl. both corners, max err **8e-9** | transposition would move the grid by **0.5** |

### 4.3 ⭐ The artifact DECLARES its units — verified on the live file

`D:/Projects/TanitAD-artifacts/hf-refcv5v2/anchors.pt`, file sha256 `4e0f0233dddb0b9f…`:

```
schema  tanitad.anchor_artifact/1        control_units  'alat'
anchors (117, 8, 2)                      controls (117, 2)
controls_columns  ['a_lon_ms2', 'a_lat_ms2']
horizon_s 6.0 · dt 0.1 · horizons_steps [5,10,15,20,30,40,50,60]
ref_speed_ms 10.0 · kappa_cap 0.12 · alat_v_floor 4.0
straight_ahead_control_present True
restamped_from  /workspace/experiments/refcv4b-b1-v72-40k/anchors.pt
restamped_from_file_sha256  e86cf507d55a4585435025fe52f33817d08dab879e1f65ff6a1fc9b0eb81e8fb
```

`anchor_meta.read_anchor_artifact(path, cli_control_units=None)` resolves
`control_units='alat'`, **`source='file'`** — no CLI override needed. ⇒ **the class-D artifact fix
landed and the live vocabulary is self-describing.** The restamp provenance names the exact file
from the 396 g post-mortem (`e86cf507…`), so the lineage is legible.

---

## 5. Q5 — DD STEP SEMANTICS (F2), SIDE BY SIDE

**PUBLISHED (primary, released code)** — `v1/transfuser_model_v2.py:504-553`:

```python
step_num = 2
self.diffusion_scheduler.set_timesteps(1000, device)        # :507
step_ratio = 20 / step_num                                  # :508
roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1]   # -> [10, 0]
...
img = self.diffusion_scheduler.step(model_output=x_start, timestep=k, sample=img).prev_sample
```

`set_timesteps(1000)` makes diffusers' `num_inference_steps = 1000`, so
`prev_timestep = timestep − num_train_timesteps // num_inference_steps = t − 1`.

**OURS** — `refcv6_diffusion.py:344-350` and its call at `refc.py:2551-2553`:

```python
for i, t in enumerate(ladder):
    if dd_step:
        out.append((int(t), int(t) - 1))                                   # DD
    else:
        out.append((int(t), int(ladder[i+1]) if i+1 < len(ladder) else 0))  # ours
```

**MEASURED on the ladder `[10, 0]`:**

| | pairs | residual retained (from the alpha table, `residual_retained`) |
|---|---|---|
| `--f2-dd-step` | **`[[10, 9], [0, -1]]`** | **0.9475** |
| legacy | `[[10, 0], [0, 0]]` | **0.2828** |

⇒ **YES — the implemented sampler is the paper's**, including the terminal `(0, −1)` pair that
diffusers resolves onto `final_alpha_cumprod`. The residuals reproduce SPEC §3's "~95 % vs ~28 %"
to the digit. Both emit the last pass's **prediction** (`poses_reg` there, `u0_hat` here), so our
clamp of `t_prev` into the alpha table is an identity at the terminal step, as documented.

### 5.1 ⚠️ Where F8 departs from DD: the clamp is outside the ladder (finding 7)

DD clamps **at the top of every loop iteration** — `x_boxes = torch.clamp(img, min=-1, max=1)`
(`transfuser_model_v2.py:519`) — and again once in training (`:474`). Ours clamps **once, before
the loop**: the clamp is `refc.py:2549`, the ladder is `refc.py:2559` (MEASURED
`clamp_is_inside_loop: false`). After `sched.step` the normalised sample reaches
**|x_n| = 15.04** (`raw/f3_f4_zeroinit.json` → `F8_clamp`, 2 steps), i.e. **15× outside DD's box**,
and pass 2 then denormalises through `dd_denorm_waypoints` unclamped.

⚠️ **Scope this honestly:** that magnitude was produced with the emitting head deliberately
un-zeroed at scale 0.5, so it demonstrates the MECHANISM, not the size on a trained checkpoint.
What is not scope-dependent is the source: **DD clamps inside the loop and we do not.** F8 has never
been run (§1.1), so nothing banked is affected. **Fix before F8's first arm.**

---

## 6. Q6 — F9's 117 v0-ROLLED ANCHORS: IS THE VOCABULARY JUSTIFIED, AND WHICH ONE LOADS?

**Which vocabulary a refcv6 run actually loads — MEASURED from the run record**
(`…/2026-09-17-refcv6-e2e-1024/raw/assembly.json:argv`):
`--anchors D:/Projects/TanitAD-artifacts/hf-refcv5v2/anchors.pt --anchor-v0-conditioned
--n-anchors 117`. ⇒ **the real 117-anchor v0-conditioned bank, with units declared in the file**
(§4.3). **Not the synthetic fallback.**

**Is 117 justified by oracle-in-vocabulary?** Register row `D-REFCV4-KINVOCAB1`
(`Project Steering/GOALS_AND_CLAIMS.md:119`), MEASURED on **4,823 windows**, zero GPU, raw
min-over-vocabulary with no model in either arm:

| vocabulary | oracle-in-vocabulary ADE |
|---|---|
| **v0-conditioned 13×9 = 117** | **0.2572 m** (−0.0424 [−0.0685, −0.0146] vs `ha`, **SEPARATED**) |
| v0-conditioned 11×11 = 121 | 0.2610 m (−0.0386 [−0.0642, −0.0116]) |
| `ha` reference | 0.2996 m |
| shipped **fixed-path** k-means | 0.3773 m (+0.0777, **loses**) |
| k-means 128 (6 s gate, different surface) | 0.3796 m — `D-REFCV4-VOCAB1` |
| ⛔ **synthetic `default_anchors`** | **1.0882 m** — *the ceiling below the 0.6843 m straight-line floor* |

⇒ **F9's "keep the 117 v0-conditioned vocabulary" is justified by a separated, model-free,
paired-bootstrap measurement.** The SPEC's warning is about the **fallback**, and the validated arm
does not take it.

### 6.1 ⚠️ But the guard against the fallback tests the DECLARATION, not the TENSOR (finding 5)

Instrument: `code/diag_f9_vocabulary.py` → `raw/f9_vocabulary.json`.

Three guards exist, and each quotes the same failure in its own message — *"a fixed-path bank
carries `anchor_controls` of all zeros, so the anchored Gaussian would be centred on 'do nothing'"*:

* `refc_v3_train.py:909-914` — `--sampler ddim` requires `core.anchors.v0_conditioned`;
* `refc.py:2756-2765` — the same refusal inside the decoder forward;
* `refcv6_diffusion.py:446` — `assert_f9_vocabulary(n_anchors: int, v0_conditioned: bool, expect_n)`.

**All three read a CONFIG FLAG. None reads `anchor_controls`.** `assert_f9_vocabulary`'s parameter
list is `['n_anchors', 'v0_conditioned', 'expect_n']` — **it has no tensor to read**.

**MEASURED** with `anchor_v0_cond = True` and `anchor_controls` left at its all-zero registered
buffer (i.e. the `--anchors`-omitted state):

```
forward_raised                     : false          <- no guard fires
F9 assert                          : passes
bank_max_spread_across_anchors_m   : 0.000000000    <- every candidate IDENTICAL
bank_lateral_max_abs_m             : 0.000000000    <- one straight line, N times
CONTROL, with real controls        : 7.106836 m spread
```

Reachable from argv: `--sampler ddim --anchor-v0-conditioned` **without** `--anchors`. The
trainer prints a loud warning (`refc_v3_train.py:6447-6453`) and **does not refuse** — MEASURED,
the string is a `print`, not a `SystemExit`. The `--anchors`-present checks at
`refc_v3_train.py:6397-6410` are correct but live inside `if args.anchors:` and are skipped entirely.

**Fix (cheap, and it is the discriminator the messages already describe):** make the refusal read
the tensor — `anchor_controls.abs().sum() == 0` on a `v0_conditioned` build is the exact
degenerate state, and it is a **literal expectation**, not an expression over the code.

---

## 7. THE GUARDS I RELIED ON — CONSTRUCTED AND CONFIRMED RED

⛔ *Do not trust a green test.* Each guard load-bearing for a claim above was exercised.

| guard | what I did | result |
|---|---|---|
| coupling (1) reaches the decoder | **wrote it** (`code/guard_coupling1_reaches_the_decoder.py`) | **RED on HEAD**, both controls GREEN — §2.3 |
| the class-D unit identities | re-introduced the 2026-09-04 curvature/`a_lat` confusion and a 2× `dt` | **both RED**, controls at their known values — §4.2 |
| F3's detach | backward from the last stage with the heads un-zeroed, **plus** a cascade-OFF control that must reach layer 0 | detach **holds** (712.82 / None), control **421.67** — §1 |
| F1's draw | `f1_t_max = 1` must collapse the support | support **[0]** — §1 |
| `stack/tests/test_refcv6_diffusion.py` | ran it | **34 passed** — but see below |

### 7.1 ⚠️ The bit-identity guard has decayed to a tautology (finding 6)

`stack/tests/test_refcv6_diffusion.py:63` materialises its "pre-refcv6" baseline with
`git show HEAD:stack/tanitad/refs/refc.py`. **MEASURED, with the 40-char shape assertion:**

```
HEAD blob   a8d8f9051942f5dc102905d141dd84a8792505ed
worktree    a8d8f9051942f5dc102905d141dd84a8792505ed   -> IDENTICAL
"refcv6" mentions in HEAD:refc.py                 83
"refcv6" mentions in 8c7d215^:refc.py              1   (the TRUE pre-refcv6 file)
CONTROL, same breath: "class AnchoredDiffusionDecoder" reads 1 in BOTH
```

`8c7d215` is the commit that introduced `refcv6_diffusion` into `refc.py` (2026-09-16,
`git log --reverse -S"refcv6_diffusion"`). ⇒ **the test compares `refc.py` against itself.** Its
stated claim — *"with all nine flags off the forward is bit-identical to the pre-refcv6 file"* — is
no longer tested by it; what remains is a check for **uncommitted worktree edits**, which also makes
it go red on a comment change. Advisory class **F #1**.

⭐ **BOTH OUTCOMES WERE STATED IN ADVANCE AND THE GOOD ONE HAPPENED.** I re-ran the test's own
comparison with the baseline pinned to `8c7d215^` (`code/diag_bitidentity_baseline.py` →
`raw/bitidentity_baseline.json`): **`BIT_IDENTICAL: true`** — identical `state_dict` keys, zero
weight mismatches, zero tensor diffs over 64 windows, identical `sel_tele`. ⇒ **the CLAIM holds;
only the GUARD is inert.** Fix: pin `_REL`'s baseline revision to `8c7d215^` (one line).

### 7.2 ⚠️ `--f7-ack-eval-join` acknowledges an absent consumer (finding 4)

`refc.py:2778-2793` refuses `sampler_groups > 1` unless the operator sets `f7_ack_eval_join`,
described as *"the operator's statement that the consumer reads `sel_anchor_id`"*. **MEASURED:**

* `sel_anchor_id` is emitted at `refc.py:3258-3261`;
* repo-wide grep (`--include=*.py`, excluding `.claude`): **zero consumers outside `stack/`** — the
  only non-`stack` hits are my own instrument;
* `taniteval/tools/t1_eval.py:98, 173, 440-442, 666` still keys the fan join on `<arm>_sel_idx`;
  `taniteval/taniteval/plan_fan.py:569-573` and `refc_rerank.py:285-287` assert
  `sel_idx == argmax(anchor_logits)` and treat it as an anchor id;
* ⭐ same-breath control: `grep -c sel_idx taniteval/tools/t1_eval.py` → **5**, so the grep reads.

⇒ Setting the flag today would produce exactly the silently mis-joined eval the refusal exists to
prevent. Advisory class **F #5** (*an allow-list excuses an absence nobody re-reads*).
**Fix: teach `t1_eval.py` / `plan_fan.py` to prefer `sel_anchor_id` when present, and make the
acknowledgement check for the column rather than take the operator's word.**

---

## 8. WHAT I COULD **NOT** ANSWER, AND WHAT WOULD SETTLE IT

1. **Does coupling (1) help?** **NOT ESTABLISHED** — it has never run inside the decoder, so there
   is no measurement of its effect, only of its absence. ⭐ The wiring half is now settled (§2.4:
   the fix works and is removable); what remains is the *capability* question. *Settles it:* land
   the fix + the guard, then a paired arm — with a **replicate** and an **inference-seed** repeat,
   because the planner samples and this rig's identity check already tripped on that noise.
2. **The size of the F8 clamp gap on a trained checkpoint.** **NOT ESTABLISHED.** §5.1's 15.04 came
   from a deliberately un-zeroed head. *Settles it:* run the probe against a trained `--sampler-space
   metre` checkpoint — but F8 has never been run, so there is none.
3. **Whether the corpus's true frame rate is 0.1 s.** **NOT ESTABLISHED HERE** — no corpus on this
   box; I verified only internal agreement (four literals + the artifact's declared `dt` 0.1 +
   `TrajectoryConfig`'s "2 s @ 10 Hz" comment). *Settles it:* one identity against a **log-recorded**
   speed, per the advisory's own prescription, on any `--v2-cache` episode.
4. **Whether F5-ON changes ADE.** **NOT ESTABLISHED** — §3 measures only that the ranked surface
   becomes sensitive to the sample; a ranking that moves is not a ranking that improves. *Settles
   it:* a paired arm, and — per `CLAUDE.md` — a **replicate** and an **inference-seed** repeat,
   because the planner samples.
5. **Whether any other refcv6 arm ran with different flags.** I read **one** run record
   (`2026-09-17-refcv6-e2e-1024`). *Settles it:* an `argv` sweep over every refcv6 `config.json`.
6. **`agent_pos` reaching the sampler on a full `RefCModel`.** I measured couplings (2)/(3) on the
   decoder directly and read `refc.py:4196, 4281` for the model-level wiring; I did **not** run a
   full `RefCV3Model` forward with hooks (an 86 M-param CPU forward at 256×1024 is ~39 s/clip).
   *Settles it:* the same `Recorder` attached to a full model on one synthetic window.

---

## 9. DELIVERABLE MANIFEST

All paths are in the repo working tree at `D:/Projects/TanitAD`, **staged, not committed, not
pushed.**

| path | what |
|---|---|
| `TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-refcv6-review/DIFFUSION_PAPER_REVIEW.md` | this report |
| `…/2026-09-22-refcv6-review/code/diag_decoder_couplings.py` | forward-hooks every cross-attention + BEV sampler; prints context shapes and `data_ptr`s (Q2) |
| `…/code/arm_bev_controls.py` | C1/C2/C3 controls + AST audit of the 7 `_decode`/`_sample` sites, of `bev=` callers, and of `bev_coupling` assignments |
| `…/code/arm_c2_force.py` | forces the anchor-prefilter branch — the in-forward positive control |
| `…/code/guard_coupling1_reaches_the_decoder.py` | **the missing consumer-side guard; RED on HEAD with both controls GREEN** |
| `…/code/fix_probe_coupling1.py` | **simulates the 2-argument fix and measures it: 0 → 6 fires, plan bit-identical at the zero gate, 3.17 m when opened** |
| `…/code/diag_f1_f9_liveness.py` | F1–F9 liveness by effect / mutation / refusal (Q1, Q5) |
| `…/code/diag_f3_f4_zeroinit.py` | F3/F4 re-measured past the zero-init bottleneck; F3 detach + control; F8 clamp probe |
| `…/code/diag_q3_scoring.py` | which surface ranks the fan (Q3) |
| `…/code/diag_units_identities.py` | six class-D identities against analytic targets, each with its historical-defect mutation (Q4) |
| `…/code/diag_f9_vocabulary.py` | the declaration-vs-tensor guard hole (Q6) |
| `…/code/diag_bitidentity_baseline.py` | re-runs the bit-identity comparison against the TRUE pre-refcv6 baseline |
| `…/raw/couplings_measured.json` · `bev_coupling_controls.json` · `bev_c2_forced.json` · `guard_coupling1.json` · `fix_probe_coupling1.json` | Q2 raw |
| `…/raw/f1_f9_liveness.json` · `f3_f4_zeroinit.json` · `f3_stage_slice.json` | Q1/Q5 raw |
| `…/raw/q3_scoring_surface.json` | Q3 raw |
| `…/raw/units_identities.json` | Q4 raw |
| `…/raw/f9_vocabulary.json` | Q6 raw |
| `…/raw/bitidentity_baseline.json` | §7.1 raw |

**Nothing is stranded:** no pod, no worktree, no agent context. Every number above traces to a file
in this manifest or to a `file:line` in `D:/Projects/TanitAD`.

**Escalations (not "please merge" in a doc):**
1. ⛔ **Coupling (1) wiring** — `refc.py:2997` and `refc.py:2884`, plus a trainer route and the
   `refc_v3.py` `bev=` pass-through. **The fix is already MEASURED to work (§2.4): 0 → 6 module
   calls, plan bit-identical at the zero-init gate, 3.171486 m when the gate is opened.** Land
   `guard_coupling1_reaches_the_decoder.py` with it.
2. ⛔ **`--f5-emitting-conf` in every refcv6 arm**, or the arm's ranking is documented as
   anchor-bank-only in `MODEL_REGISTRY.md`.
3. ⚠️ **Pin the bit-identity baseline to `8c7d215^`** (`test_refcv6_diffusion.py:63`).
4. ⚠️ **Make the v0-conditioned refusals read `anchor_controls`**, not the flag.
5. ⚠️ **F7 stays refused** until `taniteval` reads `sel_anchor_id`.
6. ⚠️ **F8's clamp moves inside the ladder** before F8's first arm.
7. ⚠️ `refc_bev_coupling.py:58` cites a test file that does not exist — the tests are in
   `stack/tests/test_refcv6_perception.py:394-518`.
