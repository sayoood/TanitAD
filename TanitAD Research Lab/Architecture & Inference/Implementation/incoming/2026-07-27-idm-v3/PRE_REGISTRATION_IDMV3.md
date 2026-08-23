# PRE-REGISTRATION — IDM v3

**Written 2026-07-27, BEFORE the arm ladder was scored.** The arms were launched
at 02:46 UTC (`tanitad-eval` PID 1775480); this file freezes the bars and the
predicted outcomes. Every prediction below is also frozen in the shipped code
comments (`idm3_geom.py`, `idm3_arms.py`, `idm3_geomtest.py`), which were copied
to the pod at 02:40–02:46 UTC — before any arm ran.

Agent: `idm-v3` · Pod: `tanitad-eval` (A40, 0 MiB used). pod1 / pod2 / pod3 not
touched. ⚠️ `/root` on the eval pod is nearly full and **silently truncates** —
verified by a `dd` that wrote 2.79 GB of a requested 3.00 GB with exit 0. All v3
output goes to `/workspace/idm3`.

---

## 0. Why this pre-registration is shaped the way it is

IDM v2 registered `nMedAE < 1.0` for yaw, then had to report that **no model can
reach it** on a ~74 %-straight corpus. That is the failure this document is
written to avoid: *an unachievable bar is not a strict standard, it is a wasted
experiment.* Every bar below is therefore stated **with the measured quantity
that makes it reachable**, or it is stated as a **direction** rather than a
level.

## 1. Substrate (identical to IDM v2 — this is deliberate, so v2 and v3 are comparable)

| | |
|---|---|
| encoder | flagship-v1 `flagship4b-speedjerk-30k`, **FROZEN**, md5 `b5f07d9e3dd2ca643949bc86832e6585`, step 29999, state_dim 2048 |
| corpora | `physicalai-val-0c5f7dac3b11` (40 eps) + `comma2k19-val-76b6e94a97a1` (64 eps) = 104 episodes |
| split | `idm2_lib.split_tags()` — episode-disjoint, domain-stratified, deterministic: **68 train / 36 val** |
| windows | **15,875 train** (stride 1) / **4,195 val** (stride 2), built at k=8 once |
| estimator | `taniteval.ci.(paired_)episode_cluster_bootstrap`, unit = the **36 val episodes**, B = 2000. `overlapping_holdout_se` is not called anywhere |

**Gate passed before anything else was quoted:** A0 (`idm_head_v1.pt`) reproduced
on this substrate at **speed R² +0.8651 / MAE 2.994, yaw +0.1046 (pai +0.9035,
cm +0.0114), steer +0.7419, long_accel −0.2398** — matching the committed
`IDM_V2_RESULTS.md` §3 numbers to four decimals.

## 2. The geometry substrate — MEASURED before any arm (this is new evidence, not a bar)

Pulled from PhysicalAI-AV's **own** gated `calibration/{camera_intrinsics,
sensor_extrinsics}` and joined to our 40 val episodes through the build's ordered
clip list. **The join is verified, not assumed: 40/40 episode_ids reproduce.**

- **Camera height is per-clip: 1.2450 – 1.6066 m** (mean 1.3417, std 0.0991,
  CV 7.4 %, full range **29.0 %**, 37 distinct values in 40 clips).
- **All three circulating `cam_h` constants are wrong.** 1.22 m is below the
  observed minimum; 1.43 m sits at the 93rd percentile; 1.5 m is exceeded by 5
  of 40 clips. **The reconciled answer is that there is no constant** — the
  median is 1.306 m and the quantity varies per clip.
- **Rig identity is NOT a proxy for camera height**: rig-A median 1.3127 m vs
  rig-B median 1.2931 m (1.5 % apart) while the within-rig spread is 29 %.
- Focal `f_paraxial` 920.1–938.5 px @1920 (2.0 % spread); after the f-theta
  canonicalisation every corpus lands at f_eff ≈ 266 px.

## 3. The physics that generates every prediction below

For focal `f`, height `h`, a ground point at forward distance `X` projects
`dv = f·h/X` px below the horizon, so

> **(1) v = (f·h) · Φ(image motion)** — metric speed is **linear in f·h**
> **(2) ω = (du/dt) / f** — yaw rate depends on **f alone**, not on h

Our pipeline already canonicalises `f_eff` to ≈266 px on every corpus (rig-A
266.13 / rig-B 266.10 / comma 266.50). **Therefore the yaw channel is already
geometry-matched and the speed channel is not.** That asymmetry is the
discriminator this whole design turns on.

---

## 4. PRIMARY ENDPOINTS — bars declared now

### E1 — Label repair (Phase 2, the largest measured lever)
**PASS** iff repairing the comma2k19 heading (hold the last observable direction
through standstill) raises the **deployed head's** pooled `yaw_rate` R² above the
0.4967 that *deleting* the 9 impossible windows achieves — i.e. **repair must
beat deletion**, and it must do so **without discarding data**.
**FAIL** ⇒ deletion is the right protocol and I say so, and the loader keeps its
`arctan2` derivation with a documented admissibility mask instead.

### E2 — Geometry conditioning vs its controls (Phase 3, the PI's hypothesis)
**PASS** iff the geometry arm `G1n` beats the recipe control `R0` on **speed**
paired ΔMAE with a CI excluding 0, **AND** beats **all three** controls —
`Ccorpn` (corpus one-hot), `Crign` (rig one-hot), `Cshufn` (shuffled geometry) —
on the same statistic.
**FAIL** ⇒ the gain, if any, is *knowing which dataset*, not *using the
geometry*, and it is reported as such. 🔴 **A gain over `R0` alone does NOT
count.**

### E3 — The discriminator (this is the one that makes E2 interpretable)
Geometry conditioning **must help SPEED and must NOT help YAW**, because of (1)
and (2) above. An arm that improves both by a CI-separated amount is reading
corpus identity, whatever its input is named.
**Committed in advance:** if `G1n` improves yaw as much as speed, I report E2 as
**not demonstrated** even if E2's arithmetic passes.

### E4 — The physics arm `G2` — **PREDICTED TO FAIL**
`G2` regresses the camera-independent `Φ = v/(f·h)` and multiplies the known
geometry back in. **I predict it fails**, because the closed-form version of
exactly this correction was already measured in `idm3_geomtest.py`:

| test (n = 40 PhysicalAI clips, all held out from A0) | result |
|---|---|
| apply `v̂ · h/h̄` (what eq. (1) prescribes) | MAE **2.960 → 3.236**, ΔMAE CI [+0.051, +0.551] — **significantly WORSE** |
| apply the opposite sign `v̂ · h̄/h` | 2.960 → 2.826, **not separated** |
| **SHUFFLED heights** (negative control) | 2.960 → 2.862, **not separated** — i.e. as good as the real thing |
| oracle per-clip scale (the headroom) | 2.960 → **1.607**, ΔMAE CI [−1.869, −0.881] |
| oracle scale factor `k` vs camera height | **r = −0.466**, partial r given v_mean **−0.352** — the **OPPOSITE** of the ground-plane sign |

**If `G2` nonetheless passes, the closed-form reading is wrong and I retract it.**

### E5 — `long_accel` as classification (`Dacc`)
**PASS** iff the 21-bin softmax-expectation decode reaches **R² > 0** against the
CAN label on **both** corpora — i.e. it beats predicting the mean, which the
regressor does not (−0.240).
⚠️ **Bar deliberately set at 0, not at 0.30.** IDM v2 registered `R² ≥ 0.30` for
this channel and it was unreachable: on PhysicalAI the CAN label correlates only
r = 0.434 with the vehicle's own dv/dt, so **a perfect kinematic estimator caps
at R² 0.188 against it**. A bar above 0.188 on that corpus is not a strict
standard, it is an impossible one.
The bins are over the **longitudinal axis alone**. Our 5-way manoeuvre softmax
mixed lateral and longitudinal and produced "0 of 881 accelerate"; the failure
mode is discretisation done wrong, so the axes stay separable here.

## 5. Falsifiers — what would make me withdraw a v3 claim

| if | then |
|---|---|
| `Cshufn` (shuffled geometry) matches `G1n` | the geometry token is **capacity, not information** — E2 is withdrawn regardless of its CI |
| `Ccorpn` (2-bit corpus one-hot) matches `G1n` (6 real numbers) | the arm knows the dataset, not the camera — E2 withdrawn |
| `G1n` improves yaw as much as speed | E3 fires; the mechanism is not geometry |
| the label repair moves **PhysicalAI** yaw | the repair is touching something it must not — PhysicalAI heading comes from a quaternion and is standstill-robust, so its yaw must be **bit-identical** before and after |
| A0 fails to reproduce | the substrate is wrong and nothing here is quotable |

## 5b. ADDENDUM — arms ADDED after the ladder launched (declared, not substituted)

**Written 2026-07-27 ~03:05 UTC, while wave 1 was still training and BEFORE any
`Dacc` result had been read.** The latent-action research stream banked its
proposal mid-run and raised a specific, checkable objection to endpoint E5.

> **The objection (INHERITED from that stream, then verified here):** Farebrother
> et al. (arXiv:2403.03950) ablate exactly the head I registered and find that a
> **hard/two-hot categorical target gives no gain over MSE** — the active
> ingredient is **HL-Gauss Gaussian label smoothing (σ/ς = 0.75)**, not the
> categorical parameterisation. A 21-bin hard-target softmax is therefore in the
> family measured *not* to beat regression, so E5 as registered risks measuring
> the wrong member of the family.

🔴 **E5's registered arm `Dacc` is NOT changed and NOT withdrawn.** Substituting
an estimator after seeing a bar miss is the forking-paths failure
`GATE_PROTOCOL` §0.3 forbids. `Dacc` (21 quantile bins, hard CE) remains the
pre-registered arm and is scored as such. Three arms are **added alongside it**,
and `Dacc` is now also the **CONTROL** that tests the objection itself:

| added arm | change | what it tests |
|---|---|---|
| `DaccHL` | HL-Gauss soft targets, 101 quantile bins, σ = 0.75 bin widths | is the smoothing the active ingredient? |
| `DaccHLsx` | HL-Gauss, 101 **symexp** bins (DreamerV3 spacing) | does resolution have to follow the target density? |
| `DaccHL0` | HL-Gauss off the plain `R0` recipe | is any gain independent of the v2 recipe? |

**A discretisation-fidelity check is reported for every one of these arms**
(`acc_discretisation_ceiling_r2`): the target's own binned representation is
decoded back through the bin centres, giving the ceiling the binning itself
imposes. **Already MEASURED on synthetic heavy-tailed data before the arms ran:
quantile spacing + σ/ς = 0.75 caps at R² 0.886, symexp caps at 0.9975.** So a
poor `DaccHL` result must be read against its own ceiling — otherwise a *binning*
failure would be reported as a *classification* failure.

**Declined for this run (stated so it is not a silent omission):** A3b, distilling
a feedforward geometry teacher for ego motion. It needs a teacher model we do not
hold and cannot be run inside this iteration. Logged as a follow-up.

## 5c. SOURCING DISCIPLINE — a hazard raised mid-run

A sibling stream had **an arXiv PDF fetch return a fabricated verbatim
quotation** — a named section, a quote and a recommendation, none of which
existed at full-text depth. **PDF-summarisation output is model-generated text,
not source text, and is inadmissible as a quotation.**

`CITATIONS.md` was written with a **per-entry fetch-depth ledger** and needs no
retrofit: 39 entries are `PUBLISHED (cited)` (abstract/body page fetched and the
specific mechanism verified), 31 are explicitly `UNVERIFIED`, and the remainder
name their exact depth (dblp / CVF / repo README / project page). Where a PDF
failed — e.g. PackNet's CVF PDF returning HTTP 403 — the consequence is marked
`UNVERIFIED` rather than filled in.

**The one citation this work leans on hardest is the strongest-sourced one:**
comma.ai's `calib_challenge` ignoring frames below 4 m/s is
`PUBLISHED (verified from the official repo README)` — a GitHub README, not a
PDF. And it is *supporting*, not load-bearing: the label defect is **MEASURED on
our own data** (26.27 % impossible below 0.5 m/s, 0.000 % above).

## 6. Known confounds I am NOT resolving (stated, not hidden)

- **comma2k19's `cam_h = 1.22 m` is INHERITED and UNVERIFIED** (`rr_log.py:93`).
  comma2k19 ships no per-segment mount height. It is the one geometry number in
  this work not measured from data. `Cshufn` exists partly to show the result
  does not hinge on it.
- **The clip-context token is a 4096-number description of the clip** and
  already carries whatever the 6 geometry numbers carry. Geometry stacked on top
  of it is therefore expected to be redundant; the informative contrast is
  geometry vs context vs nothing, all off `R0`. Both layers are run.
- **68 train episodes (26 PhysicalAI / 42 comma)** vs A0's 160 clips. Every
  "v3 vs A0" statement inherits that confound; "v3 vs R0" does not.
- comma2k19's speed label is the **GNSS** speed `‖enu_v‖` while its `long_accel`
  is `d/dt` of the **CAN** speed (`comma2k19.py:169-173`) — two different
  sources. Noted, not fixed here.
