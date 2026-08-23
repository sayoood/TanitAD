# The v5 gate CANNOT adjudicate its co-primary — and the run dies at its first held-out probe

*2026-07-27 (Europe/Berlin; pods UTC). Owner: the v5-gateable stream.
Repo HEAD at start `530f199`. Hosts: dev box (code) + **pod2** (both v5 caches, A40).
⛔ pod1 never contacted. ⛔ No v5 training launched.*

---

## 0. Headline

🔴 **THE MOST URGENT THING IN THIS DOCUMENT IS NOT IN THE BRIEF — IT WAS FOUND BY RUNNING THE
STAGED COMMAND.** `train_flagship_v4 --heldout-gate` **crashes at its first probe** on the real v5
configuration with `ValueError: cond_vtarget is on but no vt_band supplied`. In the staged command
`--heldout-every 2000`, so **v5 trains for 2 000 optimizer steps — several GPU-hours — and then
dies**. The mid-run held-out gate exists to *save* ~29.5 GPU-h (`PREP.md` cause #1); as shipped it
costs the run instead. **Nothing caught it because every existing test of that path uses a STUB
head.** §3.5.

| # | job | outcome | class |
|---|---|---|---|
| 0 | 🔴 **the held-out gate kills the v5 run** | ⛔ MEASURED on pod2 on the real caches: crash at the first probe, full traceback in §3.5. **Tripwire test added (5 tests); NOT fixed — the fix has to answer a semantic question that decides what v5 stops on** | MEASURED |
| 1 | 🔴 **the corridor co-primary on v2** | ⭐ **The chain RUNS end to end on the real v2 cache** — eval → `windows_*.pt` (dense) → `gate_emitters corridor` → a real `taniteval.corridor` block → `run_gate.py check`. **But the ONLY horizon it can produce is K=20, and K≤20 is REFUSED at BOTH ends** (`register` *and* `check`). ⇒ **a v5 gate today renders `INCOMPLETE`**, demonstrated (§1) | MEASURED |
| 1b | why K>20 is blocked | ⛔ Not a plumbing gap. The closed-loop emitter (`taniteval.clhorizon`) **exists and reaches K≤190** — but its re-render is a **pinhole homography hard-coded to `f=266, c=128`**, the deployed 256×256 crop. On v5's **176×624 cylindrical** frame that misplaces source pixels by a mean of **46.3 px at ±8°** (the mid-run gate's own probe value) against a true shift of 42.7 px; **99.08 % of pixels wrong by >1 px**, max **189.2 px**. CONTROL on the frame it was built for: **max 0.118 px** (§1.4) | MEASURED |
| 1c | ⛔ **and it hits the TRAINER too** | The same warp is what `pseudosim` uses, and `pseudosim` is the surface of the **mid-run held-out gate** — the fix for the v5 run's #1 measured failure cause (~29.5 GPU-h). **v5's early-stop signal would be computed through a camera model that is not v5's camera** (§1.5) | MEASURED |
| 2 | 🔴 **`PREFLIGHT: OK` could not fail** | ⭐ **FIXED and DEMONSTRATED FAILING on the real pod2 cache, all three directions**: pre-rename dir → **`BLOCKED`, exit 2**; renamed+registered dir → **`OK`, exit 0** with the guard visibly running; cache absent on this host → **`BLOCKED`, exit 2**. 9 new tests (§2) | MEASURED |
| 3 | 🔴 **`--batch 16` dies at step 0** | ⭐ **`--batch 8 --accum 8` VERIFIED on real steps on the real cache** (peak **27 469 MiB** of 45 498). And the effective batch is **measured, not assumed**: over **262 parameters** the accumulated gradient matches the single batch-of-64 to **6.51 × 10⁻⁸** relative. **Do NOT port `--grad-checkpoint`** — reasons in §3.4 | MEASURED |
| 4 | ⚠️ **2400 vs 2376** | Laid out for the PI (§4). ⛔ **Dropping the 24 is NOT possible without re-registering**: it changes both the count and the clip-id digest, so `assert_v2_parity_cache` refuses to start. And **the mapping from the 24 raw skips to v2 clip ids does not exist on pod2** | MEASURED |
| 5 | the command set | §5 — every leg run, with three corrections to the previously staged one | MEASURED |

🔴 **THE ESCALATIONS** (§6), in priority order: the held-out-gate crash above; the warp geometry;
the staged `--anchors-dense` path **does not exist on pod2**; pod2's `/workspace` **hit its MooseFS
quota mid-session** (a 3.1 GB checkpoint write took it there, and `df` shows none of it); the
176×624 token grid **does not tile the readout**; and the 2400-vs-2376 decision.

⚠️ **One hypothesis of mine was FALSIFIED by my own measurement and is recorded as such** (§3.3):
I argued from the source that `SigReg`'s O(n²) statistic would halve the LeJEPA regularizer when the
micro-batch went 16 → 8. **Measured: `S(n)` is flat in n** (S(16)/S(8) = 0.9925, inside the
slice-draw spread). The conclusion the algebra suggested was wrong; the flag change is safe.

---

## 1. 🔴 JOB 1 — THE CORRIDOR CO-PRIMARY ON v2

### 1.0 Why this outranked everything else in the brief

`GATE_PROTOCOL` §0 makes `corridor_departure_rate` @ a pre-registered `K` the **co-primary**, and
§0.1 gives the reason as a measurement, not an argument: on E1a's own 43 windows the same
trajectories give **0.0035 at K=20** and **0.5877 at K=185** — the 2 s instrument hides the dominant
failure mode by **~168×**, while the paired ADE@2s delta over those same windows
(`0.0109 [−0.0, 0.0312]`) is **not separated**. A v5 gate that silently drops the co-primary is the
v4 gate's failure repeating; that run produced **no verdict at all** because an input was missing.

### 1.1 ⭐ What DOES work on v2 — measured, end to end, on the real pod2 cache

`code/corridor_chain_v2.sh` on pod2. The checkpoint is a **25-step from-scratch SMOKE** at
176×624 (§3.1 produced it). ⛔ **Its ADE is 14.3461 m and is quoted nowhere as a result** — what is
under test is whether the artifacts the gate consumes can be produced on a v2 cache at all.

| leg | command | result |
|---|---|---|
| 1 | `eval_flagship_v4.py` MODE B, `--v2-val-cache … --v2-subframe 176x624 --require-parity` | ✅ **ran**; wrote `windows_v5smoke-176x624.pt` **carrying `pred_dense`/`gt_dense`**, plus the result JSON and the `taniteval.driving` panel |
| 2 | `gate_emitters.py corridor --windows … --out-corridor …` | ✅ **emitted a real `taniteval.corridor` block** |
| 3 | `run_gate.py register … --co-primary-horizon-K 20` | ⛔ **REFUSED** (as designed) |
| 4 | `run_gate.py register … --co-primary-horizon-K 60` | ✅ card written |
| 5 | `check --card K60 --corridor-json <the K=20 artifact>` | ⛔ **REFUSED** on the horizon |
| 6 | `check --card K60` with **no** corridor | ⚠️ **`VERDICT: INCOMPLETE`** |
| 7 | `check --card K20 (HAND-WRITTEN)` + the K=20 artifact | ⛔ **REFUSED** — the K≤20 rule fires inside `check` too |

The leg-2 artifact, quoted from `raw/corridor_v5smoke-176x624.json`:

```
corridor_departure_rate = 0.3928  [0.3058, 0.4733]
estimator = episode_cluster_bootstrap (n_boot 2000)
n = 265 windows / 12 episodes ; junction 0.3776 [0.2881, 0.4836], n=58/7
surface = open_loop_dense ; horizon_K = 20 (2.0 s) ; corridor_primary_m = 1.75
WARNING_blind_horizon: "K=20 is at or below ade_0_2s' own horizon … run_gate.py
                        will REFUSE this as a co-primary."
```

⭐ **The emitter declares its own inadmissibility.** That is the right behaviour and it is why this
gap was findable: nothing here silently produced a number the gate would have accepted.

### 1.2 ⛔ Why K cannot exceed 20 on the open-loop path — it is structural, not a flag

`eval_flagship_v4.collect_planner` persists `pred_dense = out["traj"]` at **the head's own
horizons**, and `FlagshipV4Head`'s are `DENSE_HORIZONS = (1..20)`. `GATE_PROTOCOL` §0.6 states it
directly: *"`rollout.collect`'s dense path runs to `fwd_k = 20` … the blind horizon. A K ≥ 100
co-primary requires a closed-loop rollout."* Raising K on this path is not a parameter change; it
is a different plan horizon, i.e. a different model.

### 1.3 ⭐ The refusal is at BOTH ends — there is no back door

This is the finding that makes §1.1 conclusive rather than merely inconvenient. `register` refusing
K≤20 would leave the obvious workaround of hand-writing a card. **It does not work**: leg 7 wrote
exactly such a card and `check` refused it with the same E1a citation, *before* rendering.

⇒ **There is no path by which a K=20 corridor number adjudicates a v5 gate.** The co-primary is
either produced at an admissible horizon or the gate is `INCOMPLETE`.

⚠️ **A refusal is a `SystemExit` and writes NO verdict JSON** (legs 5 and 7 produced no file). Any
automation around this gate that keys on the output file will see *nothing at all*, not a failure.
Recorded in `raw/gate_check_matrix.json`.

### 1.4 🔴 THE BLOCKER, MEASURED — the closed-loop warp is the wrong camera

`taniteval.clhorizon` **is** the horizon-capable emitter, it **is** in the package (not stranded in
`incoming/`), and it reproduces the v4 30 k gate's K=185 numbers exactly. It also runs on a v2
cache mechanically: `LazyV2Episode` exposes `.frames` (contiguous slices) and `.poses`, which is all
`corridor_rollout(frames_of=…)` needs.

What it does **not** survive is the geometry. Every re-render goes through one function:

```python
clhorizon.sampling_homography(dlat_m, dyaw_deg, f=F_EFF=266.0, c=CXY=128.0)
```

`F_EFF`/`CXY` are the **deployed 256×256 pinhole crop's** intrinsics (`calib.py`: `F_REF = 266` at
256 px, principal point at the geometric centre). `corridor_rollout` calls it with **no `f`/`c`
override**, and `c` is a single scalar used as both cx and cy.

v5's frame is **176×624 cylindrical at f_ref 305.5775**, principal point **(311.5, 87.5)**. So:

* **wrong focal length** — 266 vs 305.5775, **−12.95 %**;
* **wrong principal point** — off by **(−183.5, +40.5) px**;
* ⛔ **wrong projection model** — a homography is a pinhole operation; the v5 raster is an
  equidistant-azimuth cylinder (`calib.cylindrical_rays`: ray `(sin φ, y_n, cos φ)`,
  `φ = (u − (W−1)/2)/f_ref`).

**MEASURED** (`code/warp_geometry_audit.py`, `raw/warp_geometry_audit_2026-07-27.json`; dev box, no
data needed — the comparison is exact-vs-exact on both projections). On a cylinder a yaw rotation is
**exactly** `u → u + f_ref·ψ`, `v → v` — a uniform horizontal shift, depth-independent, rows
unmoved. Against that:

| frame | dψ | true shift | **mean err** | median | p95 | **max** | max spurious \|Δv\| | frac >1 px | frac >8 px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **v5 176×624 cyl** | **8°** | 42.67 px | **46.30 px** | 19.94 | 159.28 | **189.20** | **47.04** | **0.9908** | 0.5952 |
| **v5 176×624 cyl** | 12° | 64.00 px | 85.29 px | 36.13 | 302.58 | 364.37 | 88.49 | 0.9978 | 0.6916 |
| v5 256×640 cyl (cache) | 8° | 42.67 px | 50.31 px | 22.40 | 172.12 | 203.53 | 49.06 | 0.9938 | 0.6104 |
| v5 128×576 cyl | 8° | 42.67 px | 36.74 px | 14.92 | 126.20 | 150.25 | 41.22 | 0.9934 | 0.5674 |
| ⭐ **CONTROL: deployed 256×256 pinhole** | 8° | n/a | **0.045 px** | 0.041 | 0.089 | **0.118** | 0.080 | **0.0000** | 0.0000 |

Three readings, and the first is the one that decides it:

1. ⭐ **At ±8° the ERROR (46.30 px mean) is LARGER THAN THE ENTIRE CORRECT DISPLACEMENT (42.67 px).**
   The perturbed observation is not a degraded version of the right one; it is a different image.
2. **The warp invents vertical motion where there is exactly none** — up to **47 px** of spurious
   `Δv` on a 176-row frame, i.e. **27 % of the frame height**, against a true `Δv` of 0.
3. ⭐ **The control is 0.118 px max and 0 % of pixels above 1 px.** The instrument is not "wrong
   everywhere" — it is correct on the frame it was built for and wrong on v5's. A guard that fires
   everywhere would prove nothing; this one discriminates.

⇒ **A closed-loop corridor number rendered on the v5 frame today would be a number, and it would
not be a measurement of the model.** That is worse than `INCOMPLETE`.

### 1.5 ⛔ AND THE SAME WARP IS THE TRAINER'S OWN EARLY-STOP

`taniteval.pseudosim` imports `sampling_homography` / `warp_batch` **from `clhorizon`** and applies
them at `pseudo_evaluate` (`Hm = sampling_homography(dlat, dyaw)`). `pseudosim` is the surface of
`tanitad.train.heldout_gate` — the **mid-run held-out gate**, which is the fix for
`flagship-v5-retrain.PREP.md` cause #1 (**~29.5 GPU-h, half the v4 30 k run, spent training past the
best checkpoint**). Its shipped probe grid is `(−8, 0, +8)°`
(`heldout_gate.probe_grid`) — **exactly the dψ measured above**.

⚠️ **It does not crash. It produces numbers.** MEASURED this session: a real
`train_flagship_v4 --v2-subframe 176x624 --heldout-gate` run starts, the gate arms
(`[heldout-gate] ON — … primary=pseudosim_composite_PSS_recovery_progress@twosided_v2`) and the run
proceeds. There is no error to notice.

⛔ **This is an escalation, not a fix I made.** `pseudosim` is decision-grade and is what a live v5
run would stop on; changing its warp is a PI/owner call, and the correct fix is *not* a constant
swap (the projection model is wrong, not just the numbers). The exact, cheap correction exists and
is stated in §6.2.

### 1.6 ⭐ What the v5 gate CAN and CANNOT adjudicate — the plain statement

**CAN, today, on the v2 corpus:**

* `ade_0_2s` (K=20) as the **demoted diagnostic**, with an `episode_cluster_bootstrap` CI, from
  `eval_flagship_v4.py` MODE B — ✅ verified end to end this session;
* the whole `taniteval.driving` tier-0 panel and the v4 kill secondaries that have emitters;
* an **open-loop corridor panel at K=20**, with its full threshold grid and junction stratum —
  informative, **and inadmissible as the co-primary**;
* the mid-run held-out early-stop, **subject to §1.5**.

**CANNOT, today:**

* ⛔ **the registered co-primary `corridor_departure_rate` at any admissible K (20 < K ≤ 190)**, on
  either candidate v5 frame, because the only surface that reaches those horizons re-renders through
  the wrong camera model;
* ⇒ ⚠️ **`run_gate.py check` will render `INCOMPLETE`, and that is the correct output.** Demonstrated:
  `raw/gate_check_K60_no_corridor.json`.

⚠️ **`GATE_PROTOCOL` §0.7 is binding and applies here:** `nonav_route_beats_majority` is **VOID BY
CONSTRUCTION** and must be adjudicated **INSTRUMENT-FAIL, never MODEL-FAIL**. The v5 smoke eval
reproduces the precondition — its `kill_secondaries` block records
`nonav_route_beats_majority: null, "NOT REACHABLE on this checkpoint"`
(`raw/v5smoke-176x624_v4_diagnostics.json`). **It must be printed in the verdict, not suppressed.**

---

## 2. 🔴 JOB 2 — `PREFLIGHT: OK` WAS A GUARD THAT COULD NOT FAIL

### 2.1 The defect

Everything in `train_flagship_v4.preflight_asserts` was **argument-level**: it checked that
`--require-parity` was *present*, never that the cache *passes*. The guard that can refuse
(`parity.assert_v2_parity_cache`) runs inside `train()` — i.e. **after the orchestrator has
launched**. MEASURED (`V5_EVALUABLE.md` §9.2): `--print-launch --require-parity` printed
`PREFLIGHT: OK` against a directory whose `corpus_key_of` resolves to `None`.

### 2.2 The fix

`stack/scripts/train_flagship_v4.py` — new `preflight_parity_problems(a, *, manifest_path=None)`,
called from `preflight_asserts` **only when `--require-parity` is set** (so `--require-parity` stays
opt-in and nothing else moves). Three outcomes, all explicit; the third is the one that matters:

| state | before | after |
|---|---|---|
| dir exists, registered | `OK` | `OK` — and the guard's own `v2 VERIFIED … clip sha256 …` line is printed |
| dir exists, unregistered | ⛔ **`OK`** | **`BLOCKED`**, carrying the guard's refusal verbatim |
| dir **not on this host** | ⛔ **`OK`** | **`BLOCKED`**, naming the host and saying to stage from the pod |

⚠️ **SCOPE, stated so it is not over-read.** This is the **membership** guard only —
`corpus_key_of` + clip-id count + clip-id digest vs the committed manifest. It reads FILE NAMES and
decodes nothing, so it is preflight-cheap; it does **not** prove the raster, the codec or the
sub-frame (`assert_v2_geometry_matches` does that and needs built providers). **Nothing here hashes
pixels** — `V5_TRAINER.md` §9.3 is still open.

### 2.3 ⛔ DEMONSTRATED FAILING — on the REAL pod2 cache, both directions and the third state

`code/preflight_redgreen_pod2.sh`, transcript `raw/preflight_redgreen_pod2.txt`:

```
corpus_key_of(/workspace/data/physicalai-train-e438721ae894-w120-256x640cyl)
    = physicalai-train-e438721ae894-w120-256x640cyl
corpus_key_of(/workspace/v5gate/prerename/pai_wide120_v2png_train)   = None

1. RED   — the pre-rename directory        -> PREFLIGHT: BLOCKED   EXIT=2
             [PARITY-PREFLIGHT] --v2-train-cache: PARITY VIOLATION … unregistered v2 cache
             …  scripts/register_v2_sibling.py --cache <dir> --new-key <key>
             …  DIRECTORY NAME contains <key> (corpus_key_of resolves by path)
2. GREEN — the renamed, registered dir     -> PREFLIGHT: OK        EXIT=0
             [parity] --v2-train-cache: … v2 VERIFIED — 2400 clips, clip sha256 e61a04553df5…
             [parity] --v2-val-cache:   … v2 VERIFIED — 600 clips,  clip sha256 0b176d2e5cb4…
3. THIRD — a cache not on this host        -> PREFLIGHT: BLOCKED   EXIT=2
             … is not a directory on THIS host (08f6ce7d8e55), so --require-parity
               COULD NOT BE CHECKED here.
```

⚠️ **The negative control uses HARDLINKS under a real directory of the old name, not a symlink.**
`corpus_key_of` calls `Path.resolve()`, so a symlinked old name reads *through* to the renamed
target and the guard reads as inert on a test that is itself invalid — the exact mistake a sibling
stream made and self-corrected (`V5_EVALUABLE.md` §5.1). Hardlinks share inodes, so nothing was
copied and only the link directory was removed afterwards (**2 400 payloads verified intact**).

### 2.4 Tests

`stack/tests/test_preflight_parity.py` — **9 tests, 0 skips**. Both directions through
`main(--print-launch)` (the surface the orchestrator actually reads, asserting **exit 2** and
`PREFLIGHT: BLOCKED`), a registered *name* with the *wrong clip set*, the val cache checked as well
as the train one, the not-on-this-host state, and two tests that nothing moves without
`--require-parity`. One of them renames the offending directory to the registered key and requires
the refusal to disappear — so the test cannot pass by refusing everything.

---

## 3. 🔴 JOB 3 — THE BATCH THAT DIED AT STEP 0

### 3.1 ⭐ `--batch 8 --accum 8` VERIFIED on real steps on the real cache

Not a dry run: `train_flagship_v4.py` launched on pod2 against **both registered v5 caches**, at
`--v2-subframe 176x624`, `--from-scratch`, `--batch 8 --accum 8`. Log: `raw/v5_smoke_train.log`.

```
[parity] --v2-train-cache: … v2 VERIFIED — 2400 clips …
[v2] … SUB-FRAME 176x624f305.5775cyl sliced from 256x640f305.5775cyl
     at rows [40, 216] cols [8, 632] (codec png, bit-exact slice True)
[data] train windows=410202 val windows=102532
{"step": 0, …, "eff_batch": 64, "elapsed_s": 26.0, "total": 55.787, "wm": 16.951,
 "planner": 17.998, "gnorm_trunk": 107.021, "gnorm_encoder": 79.631, …}
```

**Peak GPU: 27 469 MiB** of pod2's 45 498 MiB — a **40 % margin**, against `--batch 16`'s measured
OOM at both candidate frames. The run reached and archived a checkpoint at step 25, which is the
checkpoint §1.1 scored.

### 3.2 ⭐ The effective batch is MEASURED, not assumed

The trainer logs `"eff_batch": 64` — but that line is literally `batch * accum` printed back
(`train_flagship_v4.py`), the same arithmetic its preflight already checks. It is not evidence.
`stack/tests/test_accum_effective_batch.py` (**11 tests, 0 skips**) tests the two things that
actually have to hold:

| claim | how it is established | result |
|---|---|---|
| `accum` **distinct** micro-batches per step | the shipped accumulation body pulls `next(it)` **inside** the loop and restarts the iterator on `StopIteration` rather than breaking | ✅ pinned against the source |
| micro-batches are **equal size** | `DataLoader(..., drop_last=True)` — without it the last micro-batch of an epoch is short and the accumulated gradient is a weighted mean with the wrong weights | ✅ pinned |
| ⭐ **the gradient equals the batch-of-64 gradient** | the REAL `v4_loss_step` on 16 windows, run as **(8 × 2)** and as **(1 × 16)**, compared parameter by parameter | ⭐ **262 parameters, `max\|Δgrad\| / max\|grad\| = 6.51 × 10⁻⁸`** |
| the bar is not vacuous | RED twin: delete the `/ accum` and re-measure | every gradient scales by **exactly 8** (min 7.9, max 8.1); the same statistic reads **> 6**, i.e. **10⁸× the bar** |

⚠️ **Two methodological traps, both hit and both recorded** — the numbers only mean what they say
because of them:

* ⛔ **`v4_loss_step` is STOCHASTIC.** The same batch scored twice gives a different loss, from
  **four** independent sources: three per-example Bernoulli dropouts (`v2_ego_dropout` /
  `v2_nav_dropout` / `v2_fa_dropout`), the truncated-diffusion `noise_std`, and `SigReg`'s fresh
  random slice directions. The accumulation arm consumes the RNG stream differently from the single
  arm, so a naive comparison measures **the draw**. First attempt read a 1.24 relative disagreement
  and meant nothing. The knobs are switched off **by value, not by `.eval()`** — the model must stay
  in train mode, or the test would also switch off any batch-coupled normalisation and could no
  longer see it. *(There is none: `encoder.py` bans BatchNorm outright — "LayerNorm/RMSNorm only" —
  which is the structural reason accumulation CAN be equivalent here.)*
* ⛔ **A per-parameter relative difference is the wrong statistic** and cost a second false alarm.
  `head.decoder.conf_head.bias` carries a gradient of magnitude **8.2 × 10⁻⁸**, so float32
  cancellation alone reads as a **41 %** "disagreement" on it while all 261 other tensors agree to
  ~10⁻⁷. The reported statistic divides by the **largest** gradient in the step.

### 3.3 ⚠️ A hypothesis of mine, FALSIFIED by measurement — recorded, not quietly dropped

Reading `sigreg.py` I concluded that `SigReg` — an **O(n²) pairwise** Epps–Pulley statistic whose own
source says *"Do NOT normalize by n: the statistic's built-in batch-scale is part of the validated
(λ=0.1, slices=512) operating point"* — would **halve** in value when the micro-batch went 16 → 8,
silently re-weighting the LeJEPA regularizer at an unchanged `batch × accum`. That would have made
`--batch 8 --accum 8` a change to the objective rather than a re-factorisation.

**MEASURED** (`code/sigreg_batch_scaling.py`, `raw/sigreg_batch_scaling_2026-07-27.json`; 16 slice
draws per size, mean ± spread):

| ratio | value | slice-draw spread |
|---|---:|---:|
| **S(16)/S(8)** — the `16×4` vs `8×8` question | **0.9925** | ±~4 % |
| S(64)/S(8) | 1.0088 | ±~4 % |
| S(16)/S(4) | 0.9974 | ±~4 % |

⇒ **flat in n, inside the draw spread. The inference from the algebra was wrong and the flag change
is safe.** *(Class: reasoning from a source comment instead of measuring. The comment is about the
statistic's absolute scale, not about how it varies with n.)* Pinned as a test so it cannot silently
change.

### 3.4 ⛔ `--grad-checkpoint`: DO NOT PORT — and why

`train_flagship_v4.py` has no `--grad-checkpoint`; `train_flagship4b.py` does. **Do not port it**,
for three reasons in order of weight:

1. **It buys nothing that is needed.** The measured requirement at `--batch 8` on 176×624 is
   **27 469 MiB peak with a 40 % margin**. Activation checkpointing exists to trade compute for
   memory; there is no memory to buy back.
2. **It costs GPU-hours on the arm's critical path.** Re-materialising activations is a second
   forward pass through the encoder, and the encoder is **0.662 of a full `v4_loss_step`**
   (INHERITED, `V5_EVALUABLE.md` §8.2) — so the extra forward lands squarely on the dominant term.
   At the ESTIMATED ≈87 GPU-h for this arm that is not a rounding error.
3. **It is a new code path on the launch path, entering untested.** The v5 run has one restart
   budget entry (0 of 2 used). Spending part of it debugging a memory lever that is not needed is the
   wrong trade.

⚠️ **The one condition that would reverse this**: pod1's A6000 is a different card and a 256×640
(un-sub-framed) arm at batch 8 peaks at **34 470 MiB** — still fitting, but with only a 24 % margin
and **not measured on pod1**. If the PI chooses `--v2-subframe none`, re-measure before launching.

### 3.5 🔴 AND THEN THE RUN DIED ANYWAY — at the mid-run held-out gate's first probe

Verifying Job 3 meant running the real trainer with the real flags. A second run
(`--heldout-every 20`, so the probe arrives in minutes rather than at step 2 000) got there:

```
Traceback (most recent call last):
  …/scripts/train_flagship_v4.py:1068  in _training_loop
      if heldout_gate is not None and heldout_gate.due(step):
  …/tanitad/train/heldout_gate.py:426  in probe
      pw = ps.pseudo_evaluate(planner, episodes, grid, device=device, …)
  …/taniteval/pseudosim.py:501         in pseudo_evaluate
      tj = planner.traj(fw, v0.to(device), g)…
  …/tanitad/train/heldout_gate.py:223  in traj
      out = self.head(states, v0.to(self.device), **kw)
  …/tanitad/models/flagship_v15.py:456 in condition
      raise ValueError("cond_vtarget is on but no vt_band supplied")
```

**The mechanism, in two lines of shipped code.**

* `DeployableSurfacePlanner.traj` builds
  `kw = (self.goal_kwargs_fn(b, device) if self.goal_kwargs_fn is not None else {})`
  — an **empty dict**;
* the trainer calls `heldout_gate.probe(step, world, head, heldout_episodes, device=…)` with **no
  `goal_kwargs_fn`**, and the real head is `v4_config()` with `cond_vtarget=True`, which
  **requires** `vt_band`.

⛔ **Why the suite was green — and it is the same class as `--print-launch`.**
`test_heldout_gate.py` uses `_FakeHead`/`_VaryingHead`; `test_v5_trainer_v2_val.py` uses a
`_Planner` stub and its trainer-level test **replaces `probe` outright**. Stubs accept `**kw` and
never assert on it. **The one component that can refuse had never been on the path.** The single
line that hides it is `_smoke_head_cfg`'s `cfg.cond_vtarget = cfg.cond_route = False`.

⚠️ **The provenance string is not merely silent, it is WRONG.** `DeployableSurfacePlanner`
advertises `goal_conditioning: "withheld/unknown defaults (zeros) — the deployed no-route state"`
in every probe record. What it passes is `{}`. A reader of the probe output would conclude the
withheld state was measured.

⛔ **NOT FIXED HERE, on purpose.** The obvious patch — build the zeros the docstring promises — is
not obviously right:

* `train_flagship_v4._goal_inputs` falls back to `vt_band = torch.zeros(b, dtype=long)`, and
  **zero is BAND 0 — a real target-speed band**, not the withheld state;
* the genuinely withheld values are `flagship_v15.VT_DROPPED` (= `N_VTARGET_BANDS`) and
  `ROUTE_DROPPED` (= 4), which the head's own dropout path uses;
* the two are **different deployable surfaces**, and `GATE_PROTOCOL` §0.8 is explicit that what the
  model is handed changes what the measurement is *of*. The S3 firewall measured `route`/
  `route_graded` alone lifting a no-image baseline QWK **0.1128 → 0.3381, separated**.

Choosing between them decides what v5's early-stop stops on. **That is an owner decision, not a
silent default.**

⭐ **What IS delivered: a tripwire.** `stack/tests/test_heldout_gate_real_head.py` (**5 tests, 0
skips**) reproduces the crash on CPU in ~40 ms instead of 2 000 optimizer steps, pins the wrong
provenance string, pins that the trainer's call site passes no `goal_kwargs_fn`, shows the **GREEN**
half (supplying the kwargs explicitly makes the same head plan fine — so this is a statement about
the gate, not the head), and asserts that **no existing held-out test builds a real head** so that
adding one trips a re-read. Its docstring instructs the fixer to **invert** the assertions and
record which goal state was chosen.

---

## 4. ⚠️ THE DECISION TO SURFACE — 2 400 CLIPS vs 2 376 EPISODES

⛔ **This is the PI's call, not mine.** What follows is the fact base, precisely.

### 4.1 The fact, from the committed manifest (primary source)

| corpus key | uid kind | count | skips |
|---|---|---:|---:|
| `physicalai-train-e438721ae894` (raw epcache, **every prior arm**) | `epcache_basename` | **2 376** | **24** (`skip_indices` = positions **1798 … 1941**) |
| `physicalai-train-e438721ae894-w120-256x640cyl` (v5) | `v2ep_clipid` | **2 400** | 0 |
| `physicalai-val-0c5f7dac3b11` and its v5 sibling | — | **600 / 600** | 0 / 0 |

The v5 sibling was registered against `parity_train_clips.txt`, whose length is **2 400** — the full
parity selection **before** the raw build's 24 decode failures were removed. Its membership proof
records `expected_decode_failures: 24`, `missing_count: 0`, `shortfall_matches_recorded_skips: true`.

⇒ **v5 would train on 2 400 = 2 376 + 24 episodes, i.e. +1.0101 % vs every prior arm.**

⭐ **Two things sharply limit the blast radius, and they should be said first:**

1. ⭐ **VAL IS UNAFFECTED — 600 = 600, identical membership.** Every *evaluation* number, every
   held-out CI, every paired comparison is on the same episodes as before. **The difference is
   train-side only.**
2. ⚠️ **But the extra 1 % is NOT a random 1 %.** The 24 skipped positions are **1798, 1835, 1841–1843,
   1847, 1854, 1857–1858, 1860, 1862–1863, 1873, 1875–1877, 1879–1880, 1885, 1888, 1892, 1896, 1898,
   1941** — a **tightly clustered band** in the ordered source list, not a scatter. Whatever made
   them undecodable in the raw build was localized, so the extra episodes are plausibly correlated in
   scene, session or capture condition. **This is the part that cannot be waved through as "1 %".**

### 4.2 ⛔ CAN the 24 be dropped? Not without re-registering — measured, not assumed

`parity.assert_v2_parity_cache` refuses on `len(built) != n_exp or got != exp_digest`. Deleting 24
payloads gives **2 376 ≠ 2 400 AND a different clip-id digest** ⇒ **`ParityViolation`, the trainer
refuses to start.** Dropping therefore requires:

1. identifying WHICH 24 clip ids correspond to raw positions 1798–1941. ⛔ **That mapping does not
   exist on pod2.** `/workspace/wfov/paritysplit/` holds `parity_all_clips.txt` (3 000),
   `parity_train_clips.txt` (2 400) and `parity_val_clips.txt` (600) — **no skip list** — and the
   **raw epcache train split is not on this pod at all**, so the position→clip-id map cannot be
   rebuilt here. *(The register step recorded `shortfall_identity_checked: true`, so the mapping
   existed somewhere at build time; finding it is a task, not a lookup.)*
2. physically moving those 24 out of the registered directory (`build_v2_providers` reads the whole
   dir);
3. re-running `register_v2_sibling.py` under a **new key**, renaming the directory to contain it, and
   **re-staging `parity_manifest.json`** — runbook 2a→2c→3 again. No rebuild, but a new corpus key,
   which means the currently committed manifest entry becomes historical.
4. ⚠️ and accepting a residual risk: if the 24 identified are off by even one clip, the corpus
   differs in a way **no guard would catch**, because the new digest would simply certify whatever
   set was produced.

### 4.3 The options, with consequences

| | option | consequence |
|---|---|---|
| **A** | **Keep 2 400** (as built and registered) | Zero work, zero risk of a wrong drop. v5's train set is a strict **superset**; cross-arm *training-corpus* comparability carries a stated +1.01 % asymmetry that is **not a random sample**. **Val is identical, so all held-out comparisons stay clean.** Requires an explicit `episode_count: 2400` in the v5 registry row and a note that the column is not shared with pre-v5 rows. |
| **B** | **Drop to 2 376** | Restores exact train-set membership — **if** the 24 can be identified. Costs: locating the position→clip-id map (not on pod2), a new corpus key, a new manifest entry, re-staging, and a residual mis-identification risk nothing can detect afterwards. **Does not improve any evaluation number**, because val was already identical. |
| **C** | **Keep 2 400 and measure the asymmetry** | The only option that turns the question into evidence: profile the 24 (speed, heading, rig, time-of-day) against the 2 376 and report whether they are distributionally distinguishable. Cheap — metadata only, no decode. If they are indistinguishable, A is free; if they are a cluster, that is exactly what the registry row must say. **Blocked on the same mapping as B.** |

⛔ **I did not drop them and I did not silently keep them.** The caches are untouched; the manifest
entry is the one the previous stream staged.

---

## 5. ⭐ THE END-TO-END COMMAND SET — every leg RUN, with three corrections

⛔ **Nothing here launches anything.** ⛔ **The PI still owns the frame choice** (`176x624` vs
`128x576`) and the go.

### 5.0 Corrections to the previously staged set (§7 of `V5_EVALUABLE.md`)

| # | was | is | why |
|---|---|---|---|
| 1 | `--batch 16 --accum 4` | **`--batch 8 --accum 8`** | 16 OOMs; 8×8 verified on real steps, gradient-equivalent to 6.5e-8 (§3) |
| 2 | `--anchors-dense /workspace/experiments/anchors/anchors_dense_1to20.pt` | **`/workspace/experiments/flagship_v4_anchors_dense.pt`** | ⛔ **the staged path does not exist on pod2** — MEASURED. The run would have died at load. Verify on whichever pod hosts the launch |
| 3 | `run_gate.py check … --out <f>` | **`--json <f>`** | `check` has no `--out`; the previously staged line would have errored |
| 4 | *(added 2026-07-27)* no stack pin on any eval leg | **`TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack` on §5.3 and §5.4** | ⛔⛔ **the highest-blast-radius of the four.** `eval_flagship_v4` imports `taniteval.bench`/`.driving` and `gate_emitters corridor` imports `taniteval.corridor`/`.rollout`; until 2026-07-27 those modules ran `sys.path.insert(0, "/root/TanitAD/stack")`, which on pod2 is a **12 MB pre-v5 tree**. The commands as published would have **evaluated a v5 checkpoint with pre-v5 code and printed a plausible number** — not an error. `…/incoming/2026-07-27-stale-import-guard/STALE_IMPORT_GUARD.md` |
| 5 | *(added 2026-07-27)* §5.4's `PYTHONPATH=…:/root/taniteval` | **`…:/workspace/TanitAD/taniteval`** | same class, other package: `/root/taniteval` is the pod's own old checkout of the *harness*. The stack guard cannot see this one — it pins `tanitad`, not `taniteval` — so it must be right in the command |

⚠️ **The frame number, once:** `--frame-hfov 120` in §5.1/§5.3 is the **PARENT cache's** render and is
correct. The arm that trains and is scored is the `--v2-subframe 176x624` slice: **HFOV 117.000° ×
VFOV 32.131°, 429 tokens** (MEASURED via `resolve_v2_frames` on pod2). **v5 is a 117° arm.**

### 5.1 TRAIN

```bash
cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
python3 -u scripts/train_flagship_v4.py \
  --v2-train-cache /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --v2-val-cache   /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --v2-lru 64 --require-parity \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 \
  --from-scratch \
  --anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt \
  --out   /workspace/experiments/flagship-v5-w120-rigclean-30k \
  --steps 30000 --batch 8 --accum 8 --lr-head 1e-4 --lr-trunk 1e-4 \
  --warmup 2000 --workers 8 --eval-every 500 --save-every 1000 --rollout-k 4 \
  --heldout-gate --heldout-every 2000 --heldout-episodes 8 --heldout-patience 2 \
  --device cuda
```

⭐ **Run it with `--print-launch` first, ON THE POD.** Since §2 that verdict means something: `OK`
now certifies that both caches resolved their corpus key and matched the committed clip-id digest.
From any other host it prints `BLOCKED` with the reason — which is the point.

⛔ **AS WRITTEN, THIS COMMAND DIES AT STEP 2 000** on `--heldout-gate`'s first probe (§3.5). It is
printed here unchanged because that is the command the PI's decisions attach to — **but escalation
§6.0 must land first.** ⚠️ And do not "fix" it by dropping `--heldout-gate`: preflight refuses that,
correctly.

⚠️ **Also read §1.5.** Even once it runs, `--heldout-gate` is scored through `pseudosim`'s warp,
which is not v5's camera.

### 5.2 GATE — register BEFORE the launch

```bash
cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack \
python3 scripts/run_gate.py register \
  --run flagship-v5-w120-rigclean-30k --gate-step 10000 \
  --primary-metric ade_0_2s --primary-threshold 0.60 \
  --co-primary-threshold 0.35 --co-primary-horizon-K 60 \
  --co-primary-junction-threshold 0.50 \
  --secondary "wm_canary_ade_2s<=0.55" --secondary "miss_2m<=0.10" \
  --reference-run flagship-v1 \
  --reference-log /workspace/experiments/flagship4b-speedjerk-30k/train_log.jsonl \
  --compare-metric g_op_fwd_ade_m --tau 1.5 \
  --lever-family encoder-geometry --restarts-used 0 \
  --card "Project Steering/Gates/flagship-v5-w120-rigclean-30k.card.json"
```

⚠️ `--co-primary-horizon-K 60` is `flagship-v5-retrain.PREP.md` §4's registered horizon. `register`
accepts it and stamps *"K=60 (6.0 s, 32 % of the K=190 corpus ceiling)"*; `check` additionally
flags it **below the horizon-honest floor K=100** — admissible, with the qualifier carried into the
verdict. ⛔ **On current code nothing can produce a corridor block at K=60 on the v5 frame** (§1.4),
so registering it is a promise the harness cannot yet keep. **Register it anyway** — a card with no
co-primary needs `--no-co-primary "<reason>"` and puts a *blind* gate on the record, which is worse.

### 5.3 EVAL

```bash
# ⭐ STEP 0 — refuse in one second rather than publish a pre-v5 number (correction 4).
cd /workspace/TanitAD/stack && TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack \
PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts:/workspace/TanitAD/taniteval \
python3 -m taniteval.stack_check --require v5 \
  --json /workspace/taniteval/results/stack_guard_v5.json

# MODE A FIRST (GATE_PROTOCOL O-03): validate the harness against the KNOWN v1 number.
#   v1 is a 256x256 raw-epcache arm, so MODE A stays on the RAW path.
cd /workspace/TanitAD/stack && TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack \
PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts \
python3 scripts/eval_flagship_v4.py \
  --ckpt /workspace/models/flagship-30k/ckpt.pt --canary-only \
  --val-cache /workspace/data/physicalai-val-0c5f7dac3b11 \
  --key v1-validation --out /workspace/taniteval/results/v1-validation.json

# MODE B — the v5 gate primary (DEMOTED to diagnostic), on the v5 corpus, at the v5 frame.
#   --frame-hfov 120 is the PARENT cache; the scored arm is the 176x624 slice = 117.000 deg.
cd /workspace/TanitAD/stack && TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack \
PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts \
python3 scripts/eval_flagship_v4.py \
  --ckpt /workspace/experiments/flagship-v5-w120-rigclean-30k/ckpt_step10000.pt \
  --anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 \
  --require-parity --v2-lru 64 \
  --episodes 40 --stride 8 --batch 8 --device cuda \
  --results-dir /workspace/taniteval/results \
  --key flagship-v5-w120-rigclean-10k \
  --out /workspace/taniteval/results/flagship-v5-w120-rigclean-10k.json

#  … and the DEPLOYABLE twin (no goal oracle) — add:  --goal-mode produced
```

⭐ `--results-dir` matters: `windows_<key>.pt` lands there and **is the only input the corridor
emitter has**. Without it the co-primary is uncomputable after the fact — the state all 30 committed
dumps were in (`GATE_PROTOCOL` §5).

### 5.4 THE CO-PRIMARY PANEL

```bash
# ⛔ TANITEVAL_STACK_OVERRIDE is REQUIRED here: gate_emitters.py:356-357 imports
#   taniteval.corridor + taniteval.rollout, and BOTH carried the stale insert.
#   ⚠️ /root/taniteval -> /workspace/TanitAD/taniteval (correction 5).
cd /workspace/TanitAD/stack && TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack \
PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/taniteval \
python3 scripts/gate_emitters.py corridor \
  --windows   /workspace/taniteval/results/windows_flagship-v5-w120-rigclean-10k.pt \
  --out-corridor /workspace/taniteval/results/corridor_flagship-v5-w120-rigclean-10k.json
```

⚠️ **VERIFIED to run on v2 — and it will emit `horizon_K: 20`**, which `check` refuses against a
K=60 card (§1.1 leg 5). Run it anyway: the panel is a legitimate open-loop diagnostic with its
threshold grid and junction stratum, and its `WARNING_blind_horizon` line is what tells the reader
the co-primary is missing rather than passing.

### 5.5 GATE — check

```bash
# TANITEVAL_STACK_OVERRIDE is defensive here — VERIFIED IN CODE: run_gate.py imports
# no taniteval (it mirrors taniteval.ood in pure arithmetic on purpose). Kept so the
# whole leg is spelled one way and nobody has to remember which command is exempt.
cd /workspace/TanitAD/stack && TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack \
PYTHONPATH=/workspace/TanitAD/stack \
python3 scripts/run_gate.py check \
  --card "Project Steering/Gates/flagship-v5-w120-rigclean-30k.card.json" \
  --log  /workspace/experiments/flagship-v5-w120-rigclean-30k/train_log.jsonl \
  --eval-json /workspace/taniteval/results/flagship-v5-w120-rigclean-10k.json \
  --secondary-value wm_canary_ade_2s=<v> --secondary-value miss_2m=<v> \
  --json /workspace/taniteval/results/gate_flagship-v5-10k.json
```

⚠️ **Deliberately WITHOUT `--corridor-json`.** Passing the K=20 panel makes `check` exit with a
refusal and **write no verdict file at all**; omitting it makes `check` render **`INCOMPLETE`** with
the co-primary recorded as `measured: false` and its reason printed. **The second is the honest
artifact** — it is a gate output that says exactly what is missing. Attach the K=20 panel to the
report beside it, never through `--corridor-json`.

---

## 6. 🔴 ESCALATIONS — decisions, not documentation

0. 🔴🔴 **THE HELD-OUT GATE KILLS THE v5 RUN AT ITS FIRST PROBE** (§3.5) — `ValueError:
   cond_vtarget is on but no vt_band supplied`, at step `--heldout-every` (2 000 in the staged
   command, i.e. several GPU-hours in). **This is the single thing that must land before v5
   launches.** ⚠️ The fix is small but not mechanical: it must choose between `vt_band = 0` (band 0,
   a real target-speed band — what `_goal_inputs` falls back to) and `VT_DROPPED`/`ROUTE_DROPPED`
   (the genuinely withheld state). That choice defines the deployable surface v5's early-stop stops
   on, so it is an owner decision. A tripwire test is staged; **the fix is not**.
   ⚠️ **Do NOT work around it by dropping `--heldout-gate`** — preflight refuses that, correctly, and
   it reinstates the exact ~29.5 GPU-h failure the gate exists to remove.
1. 🔴 **The closed-loop / pseudo-sim warp is the wrong camera for v5** (§1.4–1.5). It blocks the
   gate co-primary at every admissible horizon **and** silently mis-conditions the mid-run
   early-stop. **Owner needed before v5 launches, not at the gate step.**
2. ⭐ **The fix is cheap and EXACT, which is why it should not be improvised.** On a cylindrical
   raster a yaw rotation is exactly `u → u + f_ref·ψ`, `v → v` — a uniform horizontal shift,
   depth-independent, no approximation, *simpler* than the pinhole `K R K^-1` it replaces. What it
   needs is (a) the frame plumbed to `clhorizon.corridor_rollout` / `pseudosim.pseudo_evaluate`,
   (b) a projection-aware branch, and (c) ⛔ **a refusal when the frame and the warp model
   disagree** — the current silent mis-render is exactly the C13 shape. ⚠️ It touches a
   decision-grade instrument (`pseudosim` is what a live v5 run would stop on), so it is escalated,
   not done here.
3. ⛔ **`--anchors-dense /workspace/experiments/anchors/anchors_dense_1to20.pt` does not exist on
   pod2.** The real file is `/workspace/experiments/flagship_v4_anchors_dense.pt`. The staged command
   would have died at load. **Verify the path on whichever pod hosts the launch.**
4. ⚠️ **pod2's `/workspace` hit its MooseFS quota mid-session** — `ln` and `tee` failed with
   *"Disk quota exceeded"* while a 3.1 GB checkpoint sat in `/workspace/v5gate/run/`. Freeing it
   restored **395 MB/s** by real `dd`. ⛔ **`df` shows none of this.** A 30 k-step v5 run writes
   `ckpt.pt` **plus** milestone archives at 5k/10k/15k/20k/30k at **~3.2 GB each** ⇒ **≈19 GB**.
   Check the quota with a real `dd` before launching, and note this is the most likely cause of the
   first smoke's **silent** death (no traceback, no OOM line — the known "quota fills and bricks the
   write" signature).
5. ⚠️ **The 176×624 token grid does not tile the readout.** The trainer prints
   *"token grid 11x39 does not tile the readout grid 4x4, so pooling falls back to ADAPTIVE (uneven
   bins)"*. `state_dim` stays 2048 and it is flagged as a quality note — but v5 would train its whole
   run under adaptive pooling, and `128x576` (8×36) does not tile either. **Not a blocker; a stated
   property of the frame the PI is choosing.**
6. ⚠️ **2 400 vs 2 376** (§4) — PI decision, and option C is blocked on a mapping that is not on
   pod2.

---

## 7. What this does NOT close

* ⛔ **No v5 number exists.** Every ADE in this document comes from a **25-step smoke** and is
  arithmetic on random weights. Nothing here is a measurement of v5's quality.
* ⛔ **Nothing hashes pixels** (inherited, `V5_TRAINER.md` §9.3). The preflight fix proves
  *membership*, not content.
* **The corridor co-primary is not fixed, only diagnosed.** §6.2 states the fix; it is not written.
* **The held-out-gate crash is not fixed either** (§3.5 / §6.0) — a tripwire is staged, the fix is
  an owner decision.
* ⛔ **Because the gate crashes, the warp's effect on a held-out probe was never observed.**
  §1.4 measures the warp's geometry error analytically and exactly; what a *probe* does with those
  frames — whether the composite moves, and by how much — is unmeasured, and stays unmeasured until
  §6.0 lands. **Both defects sit on the same line of the same call stack, and the first hides the
  second.**
* **`--goal-mode produced`** is unchanged by this work and is still the only deployable reading of a
  MODE-B number (`GATE_PROTOCOL` §0.8).
* **The K=60 vs K=100 horizon question is untouched.** `check` flags K=60 as below the
  horizon-honest floor; whether the PREP card should move to K≥100 is a PI call, and on current code
  neither is producible.

---

## 8. Suites — zero new skips

| suite | before | after | new skips |
|---|---|---|---|
| `stack/` (dev box) | 1464 passed, 12 skipped | ✅ **1489 passed, 12 skipped** | **0** |
| `taniteval/` (dev box) | 606 passed | ✅ **606 passed** | **0** |
| pod2, real torchvision — `test_preflight_parity` + `test_accum_effective_batch` + `test_v5_frame_wiring` + `test_train_flagship_v4` | — | ✅ **64 passed** | **0** |

New tests: `test_preflight_parity.py` (**9**) + `test_accum_effective_batch.py` (**11**) + `test_heldout_gate_real_head.py` (**5**) = **25**.

---

## 9. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `V5_GATEABLE.md` (this) | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-28-v5-gateable/` | no |
| ⭐ **`stack/scripts/train_flagship_v4.py`** — `preflight_parity_problems` + its call site | `repo:stack/` (staged) + `pod2:/workspace/v5gate/stack/` | no |
| ⭐ **`stack/tests/test_preflight_parity.py`** — 9 tests | `repo:stack/` (staged) + `pod2:` | no |
| ⭐ **`stack/tests/test_accum_effective_batch.py`** — 11 tests | `repo:stack/` (staged) + `pod2:` | no |
| ⭐ **`stack/tests/test_heldout_gate_real_head.py`** — 5 tests, the §3.5 tripwire | `repo:stack/` (staged) | **repo only** |
| `code/warp_geometry_audit.py` | repo (staged) + `pod2:/workspace/v5gate/` | no |
| `code/sigreg_batch_scaling.py` | repo (staged) + `pod2:` | no |
| `code/corridor_chain_v2.sh`, `code/corridor_chain_v2_gate.sh`, `code/corridor_chain_v2_gate2.sh` | repo (staged) + `pod2:/workspace/v5gate/` | no |
| `code/preflight_redgreen_pod2.sh` | repo (staged) + `pod2:` | no |
| `raw/warp_geometry_audit_2026-07-27.json` | repo (staged) | **repo only** |
| `raw/sigreg_batch_scaling_2026-07-27.json` | repo (staged) | **repo only** |
| `raw/preflight_redgreen_pod2.txt` | repo (staged) + `pod2:/tmp/` | no (pod copy is in `/tmp`) |
| `raw/corridor_v5smoke-176x624.json` — the co-primary panel emitted off v2 | repo (staged) + `pod2:/workspace/v5gate/results/` | no |
| `raw/gate_check_K60_no_corridor.json` — the **INCOMPLETE** verdict | repo (staged) + `pod2:/workspace/v5gate/raw/` | no |
| `raw/gate_check_matrix.json` — all three `check` branches | repo (staged) | **repo only** |
| `raw/corridor_chain_leg1to4.log`, `raw/v5_smoke_train.log`, `raw/v5_smoke_train_log.jsonl` | repo (staged) + `pod2:` | no |
| ⭐ `raw/v5_smoke2_heldout_crash.log` — the §3.5 traceback, from a real run on the real caches | repo (staged) + `pod2:/tmp/` | no |
| `raw/v5smoke-176x624-g0-{K60,K20-HANDWRITTEN}.card.json` | repo (staged) + `pod2:/workspace/v5gate/gates/` | no |
| `raw/eval_v5smoke_176x624_TRUNCATED.json`, `raw/v5smoke-176x624_v4_diagnostics.json` | repo (staged) + `pod2:` | no |
| `pod2:/workspace/v5gate/` — the stack + run dirs the pod used | pod2 | no (every code file came from the repo) |
| ⛔ the SMOKE checkpoint `flagship-v5-SMOKE-176x624/ckpt.pt` | **DELETED on pod2** | — (3.1 GB, deliberately removed to free the quota; reproducible from `raw/v5_smoke_train.log`'s command in ~10 min, and it is a throwaway) |

**I ran no `git commit`, no `git push`, and switched no branch.** I `git add`ed only my own paths.
⚠️ `git status` also shows `.claude/settings.local.json` modified by the harness and an untracked
`4}` in the repo root — **neither is mine and I staged neither.**

🔒 **Confidentiality swept, not assumed:** every file in `code/` and `raw/` was scanned for
clip-id-shaped tokens (UUID pattern) — **0 found**. Counts, digests and positions only.

---

## 10. Provenance of every number

| claim | class | source |
|---|---|---|
| the corridor chain runs on v2; panel `0.3928 [0.3058, 0.4733]`, n=265/12, K=20, `open_loop_dense` | MEASURED | `raw/corridor_v5smoke-176x624.json`, `raw/corridor_chain_leg1to4.log` |
| `register` refuses K=20; `check` refuses a K mismatch; `check` refuses a hand-written K=20 card | MEASURED | `raw/corridor_chain_leg1to4.log` + `raw/gate_check_matrix.json` |
| `check` renders **INCOMPLETE** with no corridor artifact | MEASURED | `raw/gate_check_K60_no_corridor.json` |
| the open-loop dense path caps at K=20 (`DENSE_HORIZONS = 1..20`) | MEASURED | `tanitad/models/flagship_v4.py` + `eval_flagship_v4.collect_planner` (read this session) |
| warp error at ±8° on 176×624: mean 46.30 px, max 189.20, 99.08 % >1 px, spurious \|Δv\| ≤ 47.04; control max 0.118 px | MEASURED | `raw/warp_geometry_audit_2026-07-27.json` |
| `sampling_homography` uses `f=266, c=128` and `corridor_rollout`/`pseudo_evaluate` pass no override | MEASURED | `taniteval/clhorizon.py`, `taniteval/pseudosim.py` (read this session) |
| the mid-run gate's probe grid is `(−8, 0, +8)°` | MEASURED | `tanitad/train/heldout_gate.probe_grid` |
| preflight RED/GREEN/third-state on the real pod2 cache; exits 2/0/2 | MEASURED | `raw/preflight_redgreen_pod2.txt` |
| `--batch 8 --accum 8` runs; peak **27 469 MiB**; `eff_batch 64`; real steps on both registered caches | MEASURED | `raw/v5_smoke_train.log` + `nvidia-smi` this session |
| accumulation equivalence **6.51 × 10⁻⁸** over 262 params; RED twin scales ×8 | MEASURED | `stack/tests/test_accum_effective_batch.py`, run this session |
| `S(16)/S(8) = 0.9925`, `S(64)/S(8) = 1.0088` — SigReg flat in n | MEASURED | `raw/sigreg_batch_scaling_2026-07-27.json` |
| 2 376 / 24 skips at positions 1798–1941; v2 sibling 2 400; val 600/600; `expected_decode_failures 24` | MEASURED | `stack/tanitad/data/parity_manifest.json` (committed) |
| dropping 24 ⇒ count **and** digest mismatch ⇒ `ParityViolation` | MEASURED | `parity.assert_v2_parity_cache` (read) + the manifest |
| no skip list and no raw train split on pod2 | MEASURED | `ls /workspace/wfov/paritysplit/` + `ls -d /workspace/data/physicalai-train-e438721ae894` this session |
| `--anchors-dense …/anchors/anchors_dense_1to20.pt` absent on pod2 | MEASURED | `ls /workspace/experiments/` this session |
| pod2 quota exhaustion; 395 MB/s by `dd` after freeing 3.1 GB | MEASURED | this session |
| suites 1489 / 606 / 64 | MEASURED | `pytest -q` this session |
| the held-out gate crashes at its first probe on the real head (`cond_vtarget is on but no vt_band supplied`) | MEASURED | `raw/v5_smoke2_heldout_crash.log` — a REAL run on the REAL caches, pod2 |
| no existing held-out test builds a real `FlagshipV4Head`; `_smoke_head_cfg` sets `cond_vtarget = cond_route = False` | MEASURED | `stack/tests/test_heldout_gate_real_head.py`, run this session |
| `--batch 16` OOMs; max micro-batch 8 / 12 / 16; encoder = 0.662 of a step; ≈87 GPU-h | INHERITED | `V5_EVALUABLE.md` §8 — **not re-derived here** |
| the 168× E1a horizon result (0.0035 → 0.5877) | INHERITED | `GATE_PROTOCOL.md` §0.1 |
| pod1 is training | **NOT PROBED** | pod1 was not contacted at all this session |

🔒 No clip UUID appears in this document, in any artifact, or in any test fixture.
