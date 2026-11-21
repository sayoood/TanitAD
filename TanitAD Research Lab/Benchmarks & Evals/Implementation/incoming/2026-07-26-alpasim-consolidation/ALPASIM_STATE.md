# AlpaSim — TRUE STATE and HONEST LIMITS

**Date:** 2026-07-26 (Europe/Berlin) · **Author:** consolidation agent · **Pods touched:** `tanitad-eval` only,
read-only (`ls`/`du`/`head`/`git status`; no GPU work, no render, no writes). pod1/pod2/pod3 untouched.

**Evidence-class legend (CLAUDE.md operating standard):** `MEASURED` (ours + artifact path) ·
`PUBLISHED` (external, cited) · `INHERITED` (another of our docs, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.

**Companion documents (same folder):** `BUILD_AND_USE.md` (how to rebuild and run it) ·
`TANITSIM_FORK_RECOMMENDATION.md` (the fork decision).

---

## 0. TL;DR — the three sentences that matter

1. **AlpaSim works and is reproducible-by-recipe, not by content.** Every script, adapter and result
   JSON is committed (181 tracked files); **zero** of the heavyweight inputs are — not the simulator
   source (98 GB), not the renderer (38 GB), not a single scene (1.5–1.7 GB each). That is correct and
   unavoidable, and the recipe covers it — **with one real hole (§2.1).**
2. **The one real stranding is the renderer acquisition.** The 14,295,757,278-byte NRE image was pulled
   without Docker by a bespoke bearer-token/layer-fetch procedure that exists **only as prose in
   `Project Steering/LOOP_STATE.md`**. No script implements it, on the repo or on the pod. If
   `tanitad-eval` is reset, the single hardest step must be re-derived from a paragraph. **Fix in
   `BUILD_AND_USE.md` §3 — but the script is transcribed, not re-run.**
3. **Every closed-loop number this asset has produced is a WITHIN-SIM RELATIVE number.** REF-C's own
   open-loop ADE on these reconstructions is **1.5157 vs 0.4728** on real footage — **3.21×** OOD
   (`REFC_openloop_diagnostic.json`, 4 scenes / 288 predictions). Orderings survive; absolute rates do not.
   §5 states this in the form the next reader must copy.

---

## 1. What AlpaSim is, in one paragraph

`NVlabs/alpasim` is NVIDIA's microservice closed-loop AV simulator from the Alpamayo family. Six gRPC
services — **renderer** (NuRec neural reconstruction, GPU), **controller** (MPC vehicle model, CPU),
**physics** (ground-mesh constraints, GPU), **trafficsim** (reactive non-ego actors, off by default),
**driver** (the AV policy under test), **runtime** (orchestrator, writes `.asl` logs and runs eval).
Upstream deploys it with Docker Compose; **we run it bare on one A40**, which is the program's own
contribution and the reason it works on a RunPod container at all.

**Provenance (MEASURED 2026-07-26, `tanitad-eval`):** upstream commit **`55814289d8047bf239206712d31a745f2ad8f5ea`**
(shallow/grafted clone, `main`). `LICENSE` = **Apache License 2.0** (read directly, first 5 lines).
Local modification: `pyproject.toml` swapped for our pared-down version; original preserved as
`pyproject.toml.orig`. `git status --short` shows exactly `M pyproject.toml` + `?? pyproject.toml.orig`
— **no other upstream drift.** This matters: we are running *stock* AlpaSim with one dependency-pruning
edit, which is the cheapest possible relationship with upstream.

---

## 2. THE STATE TABLE

Legend: 🟢 **in-repo and complete** · 🟡 **in-repo but incomplete/superseded** · 🔵 **not in repo, but
fully reobtainable from a committed recipe** · 🔴 **referenced only — lives on a pod or an external
source, and the route back is NOT fully committed**.

| # | Component | Where it actually lives | State | Evidence |
|---|---|---|---|---|
| 1 | **AlpaSim simulator source** (12 packages: controller, driver, eval, grpc, physics, plugins, runtime, tools, trafficsim, utils, utils\_rs, wizard) | Upstream `NVlabs/alpasim` @ `55814289…`, Apache-2.0, **public**. On pod: `/workspace/alpa-invest/alpasim` (**98 GB** incl. `.venv` + `data/`) | 🔵 | MEASURED pod probe 2026-07-26; `LICENSE` read on pod |
| 2 | **NRE / NuRec renderer** (`pycena_nrm_full`, Bazel-packaged, hermetic py3.11 + torch 2.7.0+cu128) | `nvcr.io/nvidia/nre/nre-ga:26.04`, **NGC-credential gated, closed binary**. On pod: `/workspace/nre/rootfs` (**38 GB**), layers at `/opt/nre/layers/` | 🔴 | MEASURED pod probe; pull recipe **prose-only**, `LOOP_STATE.md:363-399` |
| 3 | **NuRec scene reconstructions** (USDZ + reference mp4 + `calibration_estimate.parquet` + embedded HD map) | HF `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` rev `26.04`, **gated**, 2.89 TB total. On pod: `/workspace/scene_dl` (1.7 GB) + **4 built scenesets** under `data/nre-artifacts/scenesets/` | 🔵 | MEASURED pod probe; **scene IDs committed** in `refc_suite_wizard_gen.sh`; downloader committed (`scene_dl.sh`) |
| 4 | **Bare-run setup** (`alpasim_setup.sh`, `pyproject_pared.toml`, `verify_imports.py`) | repo (07-22 bundle) **and** pod `/workspace/` | 🟢 | MEASURED: `verify_imports.py` bad=0 |
| 5 | **Service launchers** (`renderer_serve.sh`, `wizard_gen.sh`, `launch_services.sh`, `run_runtime.sh`, `refc_launch_services.sh`, `refc_suite_wizard_gen.sh`, `vs_suite_master{,_1080}.sh`, `vs_suite_run.sh`) | repo **and** pod | 🟢 | MEASURED — these produced every result in §4 |
| 6 | **Policy adapters — the working ones**: `refc_driver.py` (262 L, f-theta canon + gRPC, model-agnostic), `flagship_v1_driver.py` (108 L, reuses `RefCDriver`), `simple_driver.py` (224 L, constant-forward baseline) | repo **and** pod | 🟢 | MEASURED: live `CANON f_eff=265.6–266.0 == F_REF` on rendered frames |
| 7 | **`flagship_v1_policy.py`** (07-19 bundle, 275 L) | repo only | 🟡 **NEVER RUN — 11 open TODOs, SUPERSEDED by #6.** Its own header says "It has NOT been run end to end". Keep as design history; **do not present as the worked example** | MEASURED: `grep -c TODO` = 11 |
| 8 | **`closedloop.py`** (07-19 bundle, 572 L — the renderer-free imagination-in-the-loop harness) | repo (bundle) — **but the canonical is `taniteval/taniteval/closedloop.py`, 978 L** | 🟡 **STALE FOSSIL.** The bundle copy predates the estimator migration and still carries the deprecated `overlapping_holdout_se`; the canonical explicitly notes that estimator "survived in THIS module for five days". **Run the `taniteval` one, never the bundle copy** | MEASURED: `diff` — 572 vs 978 lines, estimator block absent from the bundle copy |
| 9 | **Results / metrics JSON** (M2, REF-C base/XL/small, flagship v1, the n=12 paired suite at 854 **and** native 1080, the **n=37 balanced** `scenario_stratified_scaled_results.json`, the n=12 `scenario_stratified_results.json`, `alpasim_realtime_a40.json`, OOD diagnostic + `REFC_openloop_preds.jsonl`) | repo, 07-22 bundle | 🟢 | MEASURED — all read this session |
| 9b | **Balanced-suite builder** (`kf_download.sh`, `kf_batch.py`, `select_suite.py`, `scaled_wizard_gen.sh`, `scaled_master.sh`, `scaled_aggregate{,2}.py`, `scaled_suite_labels.json`) + **54 auditable keyframes** (`keyframes/` 12, `scaled_keyframes/` 38, `scaled_roundabout_verify/` 16) | repo | 🟢 — the scene-selection pipeline that turned "0 roundabout scenes" into 8 | MEASURED |
| 10 | **Write-ups** (`RUN_RECIPE.md` 588 L, `flagship_vs_refc_suite_NOTE.md`, `…native1080_NOTE.md`, `scenario_and_realtime_NOTE.md`, **`scenario_stratified_scaled_NOTE.md`**, `gate0_prerequisite_NOTE.md`, `GATE1_*_NOTE.md`, 07-19 `INTAKE.md` + `CLOSEDLOOP_REPORT.md`) | repo | 🟢 (one stale number — §5.4) | MEASURED |
| 11 | **Rollout artifacts** (`.asl` 7.27 MB each, `metrics.parquet`, per-scene aggregates) | **pod only** `/workspace/{m2run,vs_flag,vs_refc,vs_*_1080}/rollouts/` | 🔵 regenerable | MEASURED (manifests + pod) |
| 12 | **Closed-loop videos** `REFC_{base,xl,small}_video.mp4`, `Flagship_v1_video.mp4` | **On the local working tree but UNTRACKED** — `.gitignore:24` `*.mp4` silently excludes them. Manifests claim `repo: … (staged)`; they are **not** staged and cannot be without `git add -f` | 🔴 **manifest/reality mismatch** | MEASURED, 4 probes: `git ls-files` (absent) · `ls` (present, 644–822 KB) · `git check-ignore -v` (matches `.gitignore:24`) · 1 mp4 *is* tracked elsewhere → `-f` works |
| 13 | **Model checkpoints** (`refc-{base,small,xl}-30k`, `flagship-30k` = speedjerk v1, +26 others) | pod `/root/models/` **and** HF `Sayood/` (gated) for the pushed subset | 🔵 | MEASURED pod probe (29 dirs listed) |
| 14 | **NRE image-pull procedure** | **`Project Steering/LOOP_STATE.md` prose ONLY.** Two probes: `git grep -l nvcr.io` → 12 docs, **no script**; `git ls-files \| grep -iE "nre\|ngc\|pull_image\|oci"` → **empty**. Third probe on the pod: `grep -rl nvcr.io /workspace/*.sh /workspace/*.py /opt/nre/*.sh` → **empty**; `/opt/nre/` holds only `layers/`, `pull.log` (10 B), `extract.log` (13 B) | 🔴 **THE ONE REAL HOLE** | MEASURED, 3 probes |

### 2.1 The stranding verdict — blunt

**Is anything stranded off-repo?** Yes, two things, and only two:

- 🔴 **The NRE image-pull procedure (#14) — the material one.** Everything else on this list is either
  committed or re-fetchable by a committed script. This is the single step with no code behind it, and
  it is simultaneously the hardest step (a manually minted 600 s scoped bearer token, 42-layer parallel
  fetch, ordered extraction, driver-lib symlinks) and the one with the least margin for error. It is
  also the step most likely to be needed under pressure — i.e. after a pod reset. `BUILD_AND_USE.md` §3
  turns the prose into a script; **that script is TRANSCRIBED, NOT RE-RUN** and is labelled as such.
- 🔴 **The four closed-loop videos (#12) — minor but a live falsehood.** Three separate deliverable
  manifests state these live in the repo. They do not; `.gitignore` excludes `*.mp4`. Anyone cloning
  fresh loses the only visual evidence of the collision/offroad failure modes. One-line fix
  (`git add -f`), for the orchestrator to decide — **this agent does not stage.**

**What is NOT stranded, contrary to what a quick look suggests:** the scenes. They are ~1.5–1.7 GB
apiece and could never be committed, but **the 12-scene suite list is committed verbatim** in
`refc_suite_wizard_gen.sh` (12 full `clipgt-…` UUIDs) and the downloader is committed. Given HF gated
access, the exact suite is reconstructible. This is the correct pattern and the rest of the asset
should be judged against it.

---

## 3. Dependencies, and whether each gate is open

| Dependency | Requirement | Status | Evidence |
|---|---|---|---|
| **GPU** | ≥40 GB VRAM recommended by NVIDIA (dominated by *their* 10 B policy; ours is 1–2 GB) | ✅ A40 46 GB sufficient; peak ~6.5 GB renderer + ~2 GB policy | MEASURED, RUN_RECIPE §11 / realtime note |
| **CUDA** | 12.6+ | ✅ 12.8, driver 580.159.04, cc 8.6 | MEASURED, INTAKE §2 |
| **Vulkan / EGL** | **NOT REQUIRED** | ✅ **RESOLVED TWICE OVER** — see §3.1 | MEASURED |
| **Container runtime** | Upstream requires Docker Compose | ⚠️ **ABSENT and NOT NEEDED** — we run all six services bare | MEASURED, RUN_RECIPE §2/§4 |
| **NGC credentials** | to pull `nre-ga:26.04` | ✅ key in `Keys.txt` (dev box only, never on a pod) | MEASURED, LOOP_STATE |
| **HF gated access** | `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` | ✅ granted to Sayed's token | MEASURED, `DL_EXIT=0` |
| **uv 0.9+ (standalone)** | AlpaSim's package manager | ✅ `/workspace/uvbin` | MEASURED |
| **Rust toolchain** | `utils_rs` builds via maturin | ✅ rustup → `/workspace/.cargo` | MEASURED |
| **Disk** | ~150 GB | ⚠️ **`/` is 93 % full — write EVERYTHING to `/workspace`** | MEASURED |
| **Heavy driver deps** (`vam`, `alpamayo_r1`, `alpamayo1_5`) | upstream drivers only | ✅ **excluded** by `pyproject_pared.toml`; our own driver needs none | MEASURED |
| **`tanitad` stack** | our policies | ✅ `PYTHONPATH` must include **both** `stack` **and** `stack/scripts` (`refc_v12_cache` lives in `stack/scripts/`) | MEASURED, §6 |

### 3.1 The Vulkan ICD question — closed, on two independent grounds

The brief flags this as a known trap ("the ICD is in `/etc/vulkan/icd.d/`, not `/usr/share/` — a wrong
probe once produced a 12-day false blocker"). **Both the trap and the underlying concern are resolved,
and the two resolutions are independent:**

1. **The ICD probe was simply wrong, and was corrected.** `RETRACTION_LOG.md:34` — 07-21, class **C2**:
   *"our pods cannot render / no EGL devices"* → *"Vulkan ICD is in `/etc/vulkan/icd.d/`, not
   `/usr/share/`. **Blocked AlpaSim + CARLA for 12 days**"*. `LOOP_STATE.md:404-408` records the
   positive measurement on `tanitad-eval` and `tanitad-pod2`: `/dev/dri` **card3 + renderD130**, plus a
   `libEGL.so.1` symlink. ✅
2. **NuRec does not use Vulkan at all.** NuRec is **gsplat/OptiX — CUDA**. The renderer booted bare on
   the A40 with **no Vulkan/EGL error** and served on `:6011`; the subsequent rollout scored
   `img_is_black = 0.0`, i.e. real non-black frames were rendered (`M2_results-summary.json`).
   `RUN_RECIPE.md:85`, MEASURED. ✅

**⇒ Do not re-litigate this.** The renderer's cold boot is ~4.5 min of CUDA kernel JIT (cached
thereafter at `/workspace/nrehome/.cache`); a slow first boot is *not* a graphics-stack failure. If a
future probe suggests otherwise, check `/etc/vulkan/icd.d/` **and** remember that a NuRec failure is a
CUDA/OptiX failure, not a Vulkan one.

---

## 4. What this asset has actually produced (all MEASURED, all on NuRec reconstructions)

| Result | n | Headline | Artifact |
|---|---|---|---|
| M2 bare closed loop | 1 | **AlpaSim runs bare, no Docker.** PASS, score 0.6637, `img_is_black=0` | `M2_results-summary.json` |
| REF-C variant sweep (base/small/XL) | 1 scene | all three collide at-fault; **small ≈ base > XL** | `REFC_{base,small,xl}_results-summary.json` |
| REF-C suite base vs XL | 12 | at-fault 4/12 both; **base score 0.345 > XL 0.246**; **no XL advantage** | `REFC_suite_results.json` |
| Flagship v1 vs REF-C base, paired | 12 @854 | **REF-C wins: pass 8/12 vs 2/12**, Δscore −0.4296 boot95 [−0.6457, −0.2147], sign 8–0 p=0.0078, pass-McNemar 6–0 p=0.031, **collisions TIED 1–1 p=1.0** | `flagship_vs_refc_suite_results.json` |
| Same, native 1080×1920 | 12 | **holds** — Δscore −0.295 [−0.494, −0.117], sign 7–0 p=0.016 → **model, not resolution** | `flagship_vs_refc_native1080_results.json` |
| Scenario-stratified | 12 | flagship's deficit is **off-highway** (straight/urban pass 1/8 vs 6/8); **highway ties** | `scenario_stratified_results.json` |
| ⭐ **BALANCED scaled suite — the most powerful result this asset has** | **37** | **REF-C still wins but by FAR less: Δscore −0.1228 [−0.2079, −0.0412]**, sign 13–4 p=0.049, pass 15/37 vs 9/37, McNemar 8–2 p=0.109. **Roundabout TIES (+0.002), highway ties (CI spans 0), BOTH collapse at intersections (flag 0/7, refc 1/7)** | `scenario_stratified_scaled_results.json` |
| ⭐ **OOD control** (force-GT open loop in-sim) | 4 scenes / 288 preds | **ADE 1.5157 vs 0.4728 real = 3.21×** | `REFC_openloop_diagnostic.json` |
| Real-time factor | — | **~0.75–0.98× @480×854; 0.29× native.** Renderer-bound | `alpasim_realtime_a40.json` |
| Map/lane availability | 5 junction scenes | **130–472 lane polygons + 130–393 road edges + wait-lines per scene**, loadable at inference | `gate0_prereq_probe.json` |

**The headline reversal this asset delivered:** the n=1 result "flagship v1 beats REF-C closed-loop"
was **retracted** (`RETRACTION_LOG.md:53`, class C5) by the n=12 paired suite. Flagship's failure mode
is **offroad from a high-deviation planner** (plan_dev 1.12 vs 0.34, 3.3×), not collision. That is
exactly the kind of correction the asset exists to produce.

### 4.1 ⚠️ …and then the asset corrected *itself* again — quote the n=37, not the n=12

**The n=12 suite's Δscore of −0.43 is INFLATED by category skew.** That suite was **8/12 straight-or-urban
scenes — REF-C's single best category.** The balanced n=37 suite (`scenario_stratified_scaled_NOTE.md`,
2026-07-23; 356 candidate keyframes screened from the 1606-scene `public_2604` pool, ~8 per category)
puts the honest paired delta at **−0.1228 [−0.2079, −0.0412]** — a real REF-C win, CI excluding zero,
but **~3.5× smaller** than the skewed suite implied.

| category | n | flag pass | refc pass | ΔScore (flag−refc) | read |
|---|---|---|---|---|---|
| roundabout | 8 | 2/8 | 2/8 | **+0.002** | ⭐ **dead heat** — previously 0 scenes, entirely unmeasured |
| highway | 8 | 2/8 | 3/8 | −0.074 [−0.294, +0.144] | **tie**, CI spans 0 |
| intersection | 7 | **0/7** | 1/7 | −0.063 | **both collapse**; flagship offroad 6/7 |
| traffic_light | 6 | 2/6 | 4/6 | −0.224 | REF-C wins; previously 0 scenes |
| straight_other | 8 | 3/8 | 5/8 | −0.272 | REF-C's best category |
| **OVERALL** | **37** | **9/37** | **15/37** | **−0.1228 [−0.2079, −0.0412]** | REF-C wins, **modestly and geometry-dependently** |

**Corrected read:** *not* "REF-C beats flagship everywhere off-highway". Rather — **REF-C's advantage is
concentrated in straight and signalized driving; on roundabouts and highways the two TIE; at
uncontrolled intersections BOTH fail** (flagship 0/7). Flagship's deficit is **offroad at complex
junctions** (offroad by category: intersection 0.86, roundabout 0.62, traffic-light 0.50, straight 0.25),
driven by the same wide-swerve signature (plan_dev 0.91 vs 0.33), punished most where the drivable
corridor is narrow or branching.

⚠️ Note also that **absolute rates rise sharply on the balanced suite** — at-fault collision flag 0.432 /
refc 0.297 and offroad flag 0.514 / refc 0.487, versus 0.167 for both on the easy n=12 set. Further
confirmation that **absolute AlpaSim rates are a property of the scene mix, not of the model** (§5.1).

**Caveat carried:** 1 scene (`0580c069`) failed a runtime route sanity check and was recovered by scoring
directly from its `metrics.parquet` (`scaled_aggregate2.py`, validated by reproducing the known
single-scene flagship score 0.699 exactly) → **n=37, not 38.**

---

## 5. ⚠️ THE HONEST LIMITS — read this before quoting any number from this asset

> ### 5.0 The one-line framing every AlpaSim number must carry
>
> **"REF-C / flagship v1 *on NuRec reconstructions* — a WITHIN-SIM RELATIVE comparison, not a real-world
> rate."**
>
> If a sentence built on AlpaSim output does not carry that qualifier, it is wrong.

### 5.1 The reconstruction-OOD confound — the big one

**MEASURED (`REFC_openloop_diagnostic.json`):** with the ego forced onto the GT path (so the model sees
in-distribution *poses*), REF-C base's **open-loop** ADE on AlpaSim's rendered frames is **1.5157 m**,
against **0.4728 m** on real PhysicalAI val — a ratio of **3.21×**, consistent across all 4 scenes
(1.4019 / 1.4167 / 1.4762 / 1.7681) over **288 scored predictions**.

**What that means:** a closed-loop failure in AlpaSim **confounds model quality with reconstruction
fidelity**. The model is being fed input ~3× further off its training distribution than anything it was
trained on, before it takes a single action.

**What survives and what does not:**

| Claim shape | Admissible? | Why |
|---|---|---|
| "REF-C base **beats** flagship v1 closed-loop (paired, n=12, both resolutions)" | ✅ **YES** | Paired design; both arms see the *same* OOD input, so the reconstruction term is differenced out |
| "REF-C base ≥ REF-C XL closed-loop; scale gives no advantage" | ✅ **YES** | Same paired logic |
| "flagship's failure mode is offroad, not collision" | ✅ **YES** | Relative, within-sim, corroborated by plan_dev |
| "**REF-C collides in 33 % of scenes**" / "flagship drives off-road 8/12 of the time" | ❌ **NO** | Absolute rates. These are model **×** reconstruction-fidelity. Logged as retraction **C6** (`RETRACTION_LOG.md:52`) |
| "our model would collide X % of the time on real roads" | ❌ **NO** | Not measurable with this instrument at all |

`RETRACTION_LOG.md:52` states the binding lesson: *"run the open-loop-vs-known control BEFORE
attributing failure to the model."* That control exists now and must be re-run for any new renderer.

### 5.2 The residual resolution confound — **substantially closed, not fully**

The n=12 suite ran at **480×854**; the n=1 flagship *pass* ran at **native 1080×1920**. That confound
was directly tested (`flagship_vs_refc_native1080_NOTE.md`): re-running the identical 12 scenes at native
res keeps the direction, keeps both CIs excluding zero, and keeps the sign test significant
(Δ −0.430 → **−0.295**, p=0.008 → **0.016**). ⇒ **resolution is a second-order modifier, not the
explanation.**

**But it is not zero, and it cuts in flagship's favour:** at native res flagship's offroad drops 8→6/12,
pass 2→3/12, and its deficit shrinks **30 %**. Two things remain open:
- The **pass-McNemar loses significance at native** (p=0.031 → 0.125) — fewer discordant pairs, not a
  reversal. The continuous score signal stays significant; **do not over-read the p=0.125**, and do not
  quote it as "no longer significant" without the sign test beside it.
- The **3.21× OOD figure itself was measured at 480×854.** Native-res source detail is one of the four
  named contributors to that ratio (alongside the reconstruction gap [likely dominant], ego-mask/colour
  residuals, and a possible ≤1-step timing residual). **The OOD control has never been re-run at native
  res.** It is cheap (~0.06 pod-day per the LOOP_STATE ranking) and would tighten the headline number.

### 5.3 The rest of the caveat set — each has bitten once

- **n=12 is small.** A 4/12 rate has a 95 % binomial interval of roughly **13–61 %**. Paired deltas are
  the load-bearing statistic; marginal rates are not. `RETRACTION_LOG.md:59` records a **n=12 departure
  "win" that REVERSED at n=40** — the binding lesson is *use full-corpus cross-fit for every closed-loop
  claim*.
- **One rollout per scene.** REF-C's absolute score varies run to run (0.345 / 0.410 / 0.496 across
  three runs) from diffusion sampling. The *paired delta* is stable; the absolute is not.
- **Scenario coverage — ⚠️ two suites exist; use the right one.** The **n=12** suite is skewed
  (highway 3, intersection 1, straight/other 8, **roundabout 0, traffic-light 0**) and its −0.43 delta
  is inflated by that skew. The **n=37 balanced suite** covers all five categories (~7–8 each) and is
  the one to quote (§4.1). Per-category n is still 6–8, so **per-category reads are directional, not
  powered** — only the overall −0.1228 has a CI excluding zero.
- **Neither model was closed-loop trained.** Both are open-loop-trained planners; error accumulation is
  expected and is part of what is being measured.
- **Metric horizon.** `RETRACTION_LOG.md:65` (class C6): a closed-loop "bound/closed" verdict **inherits
  the horizon of the metric that produced it** — a 2 s ADE window cannot see an 18 s corridor drift.
  AlpaSim rollouts here are 50 steps @5 Hz ≈ 10 s; size the metric to the event.
- **Not certified for safety.** The NRE container's own licence (§4h, read on the pod) names
  *"autonomous vehicle applications"* as a **Critical Application** that it is *"not tested or certified"*
  for. AlpaSim output is research evidence; it is not a safety argument. See the fork document.

### 5.4 ⚠️ A stale number inside our own write-up — correct it when touching that file

`RUN_RECIPE.md` §13 (lines 453–472) states the OOD diagnostic as **"236 scored predictions"**, **ADE
1.466**, **3.1×**. The raw artifact `REFC_openloop_diagnostic.json` says **288 predictions**, **ADE
1.5157**, **3.21×** (and `REFC_openloop_preds.jsonl` has 300 lines). Elsewhere the same file and all the
downstream notes use **~3.2×**, which matches the JSON.

**The JSON is authoritative** (CLAUDE.md: *"Any number in any report cites the registry or the raw eval
JSON, never a summary"*). The prose is a pre-final-run snapshot that was never swept. It is a small
discrepancy in the same direction, so no conclusion changes — but it is precisely the class of drift
(`prose lied to us`) that this program has been burned by, and this document is the second place it is
now recorded. **Quote 1.5157 / 3.21× / n=288.**

---

## 6. Traps this asset has already paid for — do not re-derive them

Each of these cost real time once. They are in `RUN_RECIPE.md` and the notes; consolidated here.

1. **`PYTHONPATH` needs TWO entries.** `stack` *and* `stack/scripts` — `refc_v12_cache.load_frozen`
   lives at `stack/scripts/refc_v12_cache.py` (MEASURED). One entry gives `ModuleNotFoundError`.
2. **The wizard emits container paths.** `generated-user-config-0.yaml` contains `/mnt/nre-data`;
   `run_runtime.sh` rewrites it to the host path. Skipping this fails at scene load.
3. **The renderer runs on `:6011`, the wizard writes `:6005`.** `launch_services.sh` seds the
   network-config. Skipping this hangs the runtime on a dead endpoint.
4. **Scene release must be `26.04`** to match the NRE 26.04 image.
5. **`/` is 93 % full.** Every cache (`UV_CACHE_DIR`, `CARGO_HOME`, `RUSTUP_HOME`, `XDG_CACHE_HOME`,
   `TMPDIR`, `HOME` for the renderer) must point at `/workspace`. The first NRE extraction silently ran
   out of space on `/`.
6. **HF Xet backend errors on this dataset** → `HF_HUB_DISABLE_XET=1` (`kf_download.sh`).
7. **Windows CRLF + BOM kill token piping** → `tr -d '\r\357\273\277'`.
8. **`xargs -I@` collides with `curl`'s `@file`** → use `-I{}`.
9. **A heredoc cannot coexist with a credential on stdin** — both consume stdin.
10. **No parentheses in remote `echo` labels over ssh** (hit again this session — the shell parses them).
11. **`tar: Cannot change ownership` on MooseFS is BENIGN.**
12. **Check `pgrep -fc 'tar -xf'` before concluding a file is missing from a tree that may still be
    extracting.** (Logged: two false "missing" conclusions, class C2.)
13. **Force-GT mode bypasses the controller** → the driver receives **no dynamic state** → `speed=0` and
    the policy is mis-conditioned. `refc_driver.py` now estimates `v0` by pose finite-difference. A run
    made before this fix was correctly discarded.
14. **~475 ms/step of avoidable CPU** — the driver re-canonicalizes its whole 24-frame history every
    step at native res. Caching canon'd frames is a free ~5× driver speedup. **Known, not fixed.**

---

## 7. What this asset can and cannot do — capability boundary

| Capability | Status |
|---|---|
| Closed-loop rollout of an arbitrary policy over reconstructed real scenes | ✅ **MEASURED**, bare, no Docker |
| Collision / off-road / progress / plan-deviation / dist-to-GT metrics | ✅ **MEASURED** |
| **HD map, lanes, road edges, wait-lines** at inference | ✅ **MEASURED** — `trajdata.VectorMap` embedded in every scene USDZ, loadable via `ArtifactSceneProvider`. **Our PhysicalAI corpus has none of this; AlpaSim scenes do.** |
| Native f-theta rendering matching our training canonicalization | ✅ **MEASURED** — `f_eff = 265.6–266.0` vs `F_REF = 266` live |
| TanitEval-standard videos (camera + planned traj + BEV inset + metrics) | ✅ **MEASURED** — AlpaSim's default eval-video layout already *is* our viz standard |
| Reactive non-ego agents (SMART + CAT-K) | 🟡 **IN THE APACHE-2.0 TREE, NEVER ENABLED BY US.** `src/trafficsim/alpasim_trafficsim/catk/smart/` present (MEASURED); disabled by default in every run we have made. **Untested by us — do not claim it works until a rollout proves it.** |
| Low-OOD (≤1.5×) input | ❌ **NO** — 3.21×. This is the binding limitation |
| An absolute, real-world safety rate | ❌ **NO** — see §5 |
| Real-time operation at native res | ❌ 0.29× (renderer-bound) |
| Roundabout / traffic-light behaviour | ✅ **MEASURED at n=8/6** in the balanced suite (§4.1) — roundabout is a **tie**. Directional at that n, but no longer unmeasured. **A balanced-suite builder pipeline exists and is committed** (`kf_download.sh` → `kf_batch.py` → `select_suite.py` → `scaled_wizard_gen.sh` → `scaled_master.sh` → `scaled_aggregate2.py`), with 356 screened keyframes' worth of labelling already banked |
| Authoring novel scenarios from scratch | ❌ — scenes are *reconstructions of recorded drives*; actor editing exists (`--enable-editing-actors`), whole-cloth authoring does not. **Scene *selection* is the real lever, and it is solved** (previous row) |

---

## 8. Where this sits strategically

The independent chief-scientist review (`Project Steering/Reviews/2026-07-25-independent-chief-scientist-review/R5_strategy_research_management.md:33-37`)
names a **lower-OOD reactive-agent renderer** as the build that gates *"safety-grade closed-loop, D5/D6,
the renderer-half of the beyond-ADE suite, and any NAVSIM/Bench2Drive entry (i.e. any opponent-facing
number)"*, and records that **no pod is building it**.

AlpaSim is the program's *only* instrument that has reactive agents, a map, and collision metrics at
all. Its cost of admission is the **3.21× OOD tax**. The complementary instrument — the real-footage
low-OOD harness at **1.02–1.20×** — is map-free and agent-free by construction, so it can measure drift
but **never** off-road or collision (`LOOP_STATE.md` G1clean). The two instruments are, today, mutually
exclusive on exactly the axis that matters. Closing that gap is the subject of
`TANITSIM_FORK_RECOMMENDATION.md`.

---

## 9. Recommended next actions (ranked, for the PI — none started)

1. **Commit the NRE pull script** (`BUILD_AND_USE.md` §3) and **verify it once** on a scratch path. It
   is the only unscripted step, and it is the hardest one. *(~30 min, no GPU.)*
2. **`git add -f` the four closed-loop videos**, or amend the three manifests that claim they are in the
   repo. Currently the docs assert something false. *(~2 min. Orchestrator's call — this agent does not stage.)*
3. **Correct `RUN_RECIPE.md` §13** to 1.5157 / 3.21× / n=288 next time that file is touched. *(§5.4.)*
4. **Re-run the OOD control at native 1080×1920** — the single cheapest tightening of the headline
   confound (~0.06 pod-day, already ranked #1 in `LOOP_STATE.md:112`).
5. **Delete or clearly mark the 07-19 `closedloop.py` fossil** so nobody runs a deprecated-estimator
   copy of a harness whose canonical version is 406 lines longer.
6. **Decide the fork question** — `TANITSIM_FORK_RECOMMENDATION.md`.
