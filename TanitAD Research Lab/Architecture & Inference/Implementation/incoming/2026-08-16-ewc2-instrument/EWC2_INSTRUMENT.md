# E-WC2 — the σ\* instrument, built and testable; and the §5.2 cost premise, corrected

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` · **Staged, not committed, not pushed.**
**Pre-registration:** `Project Steering/V6F_PLANNER_DESIGN.md` §5.2 (E-WC2), §3.1 (the winner's-curse
surface it composes with), §5.3 (the refutations).
**Scope:** the instrument only. ⛔ No GPU work, no ssh, no measurement was run. There is no σ\* number
in this document and no verdict — producing one is a separate, now-cheap step (§4).

---

## 0. TL;DR — three things

1. **The estimator exists and is tested.** `stack/scripts/e_wc2_sigma_star.py` fits the P1/P2
   battery's ridge under **LOEO over the 40 val episodes**, emits σ at 2 s and 6 s, the two ratios,
   the pre-registered verdict, n, and the estimator name — and **refuses to emit a verdict** when the
   pre-registered surface is not met. 62 CPU tests, all green.
2. ⭐ **THE "0 GPU" COST IN §5.2 IS STALE, BUT NOT IN THE DIRECTION THE BRIEF ASSUMED.** The brief
   said the banked latents died with the eval pod, so E-WC2 now needs a GPU pass at a training pause.
   **That is REFUTED for the REF-C surface: the banked latents are IN THE REPO.** What is missing is
   the **6 s endpoint**, which is *ground truth from poses* — no model, no GPU. §5.2's "0 GPU" holds;
   what it silently assumed and does not have is **the val40 pose arrays on a reachable disk**.
3. **A GPU pass IS still required for the surface §5.2 actually names.** §5.2 says *"frozen **S-W**
   latents"*. S-W latents have never been dumped. That pass — and only that one — needs the GPU and
   therefore a deliberate pause. Estimate and recipe in §4.2.

---

## 1. What was built

| artifact | repo path | what it is |
|---|---|---|
| **the estimator** | `stack/scripts/e_wc2_sigma_star.py` | E-WC2 end to end: LOEO ridge → σ(2 s), σ(6 s) → ratios → verdict → §5.3 REDERIVE flag. JSON out. CPU. |
| **the tests** | `stack/tests/test_e_wc2_sigma_star.py` | 62 tests on synthetic latents with a **planted** σ. |
| **the producer** | `stack/scripts/refc_dump_latents.py` | EXTENDED (not duplicated): `--endpoint-steps` on the inference pass, plus a new **0-GPU `--backfill-endpoints`** mode. |
| this writeup | `…/incoming/2026-08-16-ewc2-instrument/EWC2_INSTRUMENT.md` | — |

### 1.1 The ridge is the P1/P2 battery's, imported — not a second one

Every estimation primitive is imported from `stack/scripts/probe_latent_state.py`:

| primitive | source | what it fixes |
|---|---|---|
| `RidgeSVD` | `probe_latent_state.py:142-187` | closed-form ridge, one economy SVD of the **centred** design shared across targets and λ |
| `RIDGE_LAMBDAS` | `probe_latent_state.py:134` | `(1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)` |
| `RidgeSVD.best_lambda` / `.gcv` | `probe_latent_state.py:172-187` | λ by Golub-Heath-Wahba GCV **on the TRAIN fold only** — model selection never touches the held-out episode |
| `_standardize` | `probe_latent_state.py:269-274` | z-score by **TRAIN** mean/sd, sd floor `1e-8 → 1.0` |
| `episode_disjoint_folds` | `probe_latent_state.py:226-246` | the fold builder LOEO is constructed from |
| `r2_score` | `probe_latent_state.py:212-220` | `None` when the target has no variance — never 0-filled |

`ridge_oof_predict` is `ridge_probe_cv` (`probe_latent_state.py:280-306`) with the out-of-fold
**predictions** returned as well as the pooled R², because residuals are what σ is computed from.
`test_ridge_oof_matches_probe_battery` pins its pooled R², per-fold R² **and** per-fold λ equal to
`ridge_probe_cv`'s on the same inputs, so the two paths cannot drift.

### 1.2 ⛔ σ IS PER-AXIS — the unit that decides the verdict

§3.1 B perturbs the goal with `g = gt_end + rng.normal(0.0, s, size=gt_end.shape)`
(`sel_winners_curse_law.py:221`), and `gt_end` is `[W, 2]`. So **`s` is the per-axis SD of an
isotropic 2-D Gaussian, in metres** — *not* the radial RMS, which is `√2 ×` larger.

The unit is pinned arithmetically, not by assertion: §3.1 publishes σ\* ≈ 0.8 m, and
`0.8 / 0.4714 = 1.70` and `0.8 / 0.1639 = 4.88` reproduce its published ratios exactly. A radial-RMS
reading of the same σ would give 2.40 and 6.90.

⚠️ **This is a live failure mode, not a pedantry.** Reading σ in the radial unit against a threshold
stated in the per-axis unit inflates it by **1.414** — enough to flip **FUNDED → INCONCLUSIVE** on
arithmetic alone. The instrument emits `sigma_perax_m` as the headline, `sigma_radial_rms_m`
alongside, `sigma_long_m` / `sigma_lat_m` for the four-families decomposition, and carries a
`_unit_note` on every row so the number cannot be re-read in the wrong unit downstream.

### 1.3 LOEO is leave-one-**EPISODE**-out, and the leak is measured, not asserted

`loeo_folds` = `episode_disjoint_folds(eid, n_folds=n_unique_episodes)`, then **asserts one episode
per fold** — a builder that silently merged two episodes would turn LOEO into 39-fold CV and nothing
in the output would say so.

⭐ **MEASURED (ours, `tests/test_e_wc2_sigma_star.py::test_window_disjoint_folds_leak_and_understate_sigma`):**
on a fixture with an episode-level nuisance a window-random split can memorise, **LOEO reports
σ = 8.195 and the window-disjoint split reports σ = 0.201** — the leaky split recovers the 0.2 noise
floor exactly and **understates σ by 40×**. That is the REF-A I-JEPA defect (~80 % of val inside
train) reproduced as a number, and it is a **downward** bias on precisely the quantity that decides
whether SEL-1 is funded. A leaked E-WC2 would fund SEL-1 unconditionally.

### 1.4 It refuses rather than emitting a weak verdict

`verdict ∈ {FUNDED, REFUSED, INCONCLUSIVE, NO_VERDICT}`. **`NO_VERDICT` is a distinct token from
`REFUSED`** — `REFUSED` means "SEL-1 is refused before launch" (a real §5.2 outcome); `NO_VERDICT`
means the instrument declined to answer. Guards that force `NO_VERDICT`:

- fewer than **40 episodes** or fewer than **881 windows**;
- **the guards being relaxed on the command line at all** — `--min-episodes 10` does not buy a weak
  verdict, it buys `NO_VERDICT`. This is the part that matters: a future operator cannot obtain a
  fundable-looking number by loosening a flag;
- **no 6 s horizon** — §5.2 requires σ at 2 s *and* 6 s, so a 2 s-only dump gets `NO_VERDICT` even
  when its 2 s ratio sits comfortably inside the FUNDED band (tested);
- an **ECHO** (ego/nav-at-inference) block in the design matrix;
- the producer's `instrument_fail` being non-empty;
- the fan's last waypoint not being the verdict horizon — ⛔ *a σ at 2 s over an ADE at 1 s is not a
  quantity*, so the ratios are withheld rather than computed.

Everything measurable is still written in every branch, so a refused run is **inspectable**, not
invisible. CLI exit codes: `0` verdict emitted, `3` dump non-conformant (`--validate-only`),
`4` `NO_VERDICT`.

### 1.5 The admissibility rule is enforced, not documented

Per the binding PI rule (*"LABELS MAY USE EGO; INFERENCE IS VISION-ONLY"*, 2026-08-03) and the goal-input
rule (*a goal input must not carry the situation classifier's output*), feature blocks are
**classified**, and an unclassified block is **refused until declared**:

| block | class | source |
|---|---|---|
| `pooled`, `pooled_seq`, `ctx` | `VISION_ONLY` | `refc_dump_latents.py:25-27` states it verbatim |
| `measurement`, `v0` | ⛔ `ECHO` | `refc_dump_latents.py:28`: the ego+nav embedding, *"the v0-echo path by construction"* |
| anything else | `UNDECLARED` → **refused** | must be declared with `--declare NAME=VISION_ONLY\|ECHO` |

`measurement` can only enter with `--allow-echo-features`, and doing so **forces `NO_VERDICT`** and
stamps the arm as a labelled-inadmissible control. This is the C6-confound / nav-echo family: a goal
head reading the ego state at inference would report a **leak magnitude**, not a capability.

### 1.6 §5.3's REDERIVE check, on **matched windows**

§5.3, verbatim: *"a σ\* re-measured at 6 s exceeds 3× the 2 s value ⇒ the ratio form does not
transfer; the threshold must be re-derived, not scaled."*

Implemented as `rederive_check_5_3`, and ⛔ **no branch ever emits a scaled 6 s threshold** — not even
the passing one. When the multiple is under 3×, the output says that this licenses **re-measuring**
the threshold on a 6 s fan, not multiplying the 2 s number.

⚠️ **The comparison is made on matched windows.** The 6 s endpoint does not exist for the last ~5
windows of every episode (§2.2), so σ(2 s) is **re-fit on exactly the 6 s-valid subset** for the 3×
comparison, while the headline σ(2 s) keeps the full 881. Comparing a full-grid σ(2 s) against a
truncated-grid σ(6 s) would be the "never compare across different windows" defect, and it would bias
the REDERIVE flag in an unknown direction.

### 1.7 Estimator and stamps

Point estimates are **full-set**. Intervals are the **episode-cluster bootstrap** (`taniteval/ci.py`),
with the reducer named in the JSON. σ's interval uses `per_window = |e|²/2` with `reduce="rms"`, so
`sqrt(mean(|e|²/2)) = sigma_perax` **exactly** — the interval is on the same statistic as the point
estimate, not on a proxy. The σ/ADE interval resamples numerator and denominator under **one shared
episode draw** (a quadrature combination would be invalid — they are not independent).
⛔ `overlapping_holdout_se` appears nowhere, and a test asserts the string is absent from the emitted
estimator name.

Every output carries: `_evidence_class` (MEASURED, ours), `_class` (EXPLORATORY on a REF-C fan — the
**ratios** transfer, the absolute metres are that fan's), `_tier` (**T0-DIAGNOSTIC** — a
representation-capacity probe; **no T1 capability claim may cite it**), `_estimator`, `_ridge`, and
the full `_prereg` block.

### 1.8 The four families

Per the binding rule, per family and never pooled:

| family | what E-WC2 reports |
|---|---|
| **LONGITUDINAL** | `sigma_long_m` per horizon (along-track endpoint 1σ). Target-speed / distance-keeping **n/a with reason**: this is an endpoint-capacity probe and rolls no trajectory. |
| **LATERAL** | `sigma_lat_m` per horizon (cross-track endpoint 1σ). Heading / curvature / yaw-rate **n/a with reason**, same reason. |
| **TACTICAL** | **the headline** — σ/ADE and σ/oracle *are* the goal/anchor-selection admissibility test, i.e. §3.1 B's requirement curve read from the capability side. |
| **STRATEGIC** | **n/a with reason** — PhysicalAI-AV ships no map, lane graph, junction annotation or route signal (§6). |

---

## 2. The dump contract

`python stack/scripts/e_wc2_sigma_star.py --print-contract` prints it;
`--validate-only` checks a dump against it and exits `3` on any violation. The contract is a
module-level `DUMP_CONTRACT` dict, so a future S-W producer has one machine-checkable target.

### 2.1 What E-WC2 needs

| key | shape | why |
|---|---|---|
| `eid` | `list[int]` len n | **the only** source of LOEO folds |
| ≥1 VISION-ONLY block (`pooled`, `ctx`, …) | `[n, F]` or `[n, W, F]` | the design matrix |
| `gt_endpoint` | `[n, He, 2]` | ego-frame endpoint, **axis 0 = longitudinal (forward), axis 1 = lateral (left)** |
| `endpoint_steps` | `list[int]` | 10 Hz steps; **must contain 20 and 60** |
| `endpoint_valid` | `[n, He]` bool | ⛔ False where the horizon runs past the episode end. **Never imputed** |
| `fan`, `gt`, `sel`, `wp_steps` | — | the ratio denominators: oracle ADE and incumbent selected ADE |
| `cv` *(optional)* | `[n, T, 2]` | adds §3.1 B's deployable 0-param reference row |

### 2.2 ⛔ The K_MAX conflict — why the 6 s endpoint is masked, not "fixed"

The producer's grid is `range(0, T − WINDOW − K_MAX, STRIDE)` with `K_MAX = max(WP_STEPS) = 20`.
The obvious "fix" — raise K_MAX to 60 so every window has a 6 s future — **re-selects windows**: the
881-window grid shrinks, the fan bit-identity gate fails, and cross-arm comparability is gone.
**Parity is sacred**, so the grid is left alone and the out-of-range endpoints are written as **NaN
with `endpoint_valid=False`**. E-WC2 excludes them **per horizon with n reported** (the P1/P2
per-target-exclusion rule) and additionally drops any non-finite target defensively, so a legacy dump
without the mask cannot silently return a NaN σ that reads like a number.

`K_MAX_GRID` is now a module-level constant asserted equal to `max(dd.WP_STEPS)` at run time, and
both the inference pass and the backfill call one `window_starts()` — the two grids cannot drift.

---

## 3. ⭐ THE STALE PREMISE, CORRECTED — with what refutes it

**The brief's premise (INHERITED):** *"§5.2 says 'Cost: 0 GPU, banked latents'. Those banked latents
lived on the eval pod, and all pods are gone. So E-WC2 now needs a latent-production pass that must
be scheduled at a deliberate training pause."*

**MEASURED (ours), and it refutes the first half:**

```
TanitAD Research Hub/Architecture & Inference/Implementation/incoming/
    2026-08-04-lambda-findability/raw/latents_refc-xl-30k.pt     39,513,973 B   md5 b88737ad…
    2026-08-04-lambda-findability/raw/latents_refc-base-30k.pt   26,658,449 B
```

Loaded on this box, CPU:

| | `refc-xl-30k` | `refc-base-30k` |
|---|---|---|
| `pooled` | `[881, 992]` | `[881, 704]` |
| `pooled_seq` | `[881, 8, 992]` | `[881, 8, 704]` |
| `ctx` | `[881, 96]` | `[881, 64]` |
| `fan` / `gt` / `sel` | `[881, 256, 4, 2]` / `[881, 4, 2]` / `[881]` | `[881, 128, 4, 2]` / … |
| episodes | **40** | **40** |
| `instrument_fail` | **`[]`** | **`[]`** |
| `controls_vs_bank` | `fan_bit_identical: True`, `gt/sel/v0` all True | idem |

⇒ **The banked surface E-WC2 needs is in the repo, at exactly the pre-registered 881×40, with its
producer's bit-identity gate green.** It was dumped on Thor (`host: thor6`) on 2026-08-04 and
committed with its package. Running the instrument against it right now reports the **single** gap:

```
$ python stack/scripts/e_wc2_sigma_star.py --dump …/latents_refc-xl-30k.pt --validate-only
{ "conformant": false,
  "problems": ["missing `gt_endpoint`/`endpoint_steps` — §5.2 requires the 6 s endpoint;
                a 2 s-only dump cannot answer E-WC2"] }
```

**So the correction is:**

| claim | status |
|---|---|
| "the banked latents are gone with the pods" | ⛔ **REFUTED** for the REF-C surface — they are in-repo, verified by loading |
| "§5.2's 0 GPU is stale" | ⚠️ **PARTLY** — the *latent* half is 0 GPU and already paid. The missing 6 s endpoint is **ground truth from poses**: no model, no GPU, no re-inference |
| "a latent pass must be scheduled at a training pause" | ⛔ **NOT for REF-C.** The backfill is CPU-only and can run beside a training job |
| "§5.2's stated surface is available" | ⛔ **NO** — §5.2 names ***S-W*** latents, which have **never been dumped**. That pass is real GPU work (§4.2) |

**What §5.2 actually assumed and does not have: the val40 pose arrays on a reachable disk.**
MEASURED, two probes (the "absence at one location is not absence" rule): no `~/valdata` on this box;
no `*physicalai-val*` anywhere under the repo; and the 27 banked `taniteval/results/windows_*.pt`
carry only `pred`/`gt`/`cv`/`eid`/`speed`/`head_deg` at `wp_steps` — **a 2 s grid, no poses, no 6 s
GT**. The poses exist on Thor at `~/valdata/physicalai-val-0c5f7dac3b11` and in HF
`Sayood/tanitad-physicalai-w120-256x640cyl` (val 603 files / 21.2 GB; the 40 val40 episodes are a
small subset of that).

**Root-cause class (for `RETRACTION_LOG.md`):** *a cost estimate that priced the expensive input and
not the cheap one.* §5.2 correctly identified the latents as the costly artifact and correctly noted
they were banked — then omitted the **target**, which is free to compute and was never dumped. The
resulting "0 GPU" is true about compute and false about readiness. Same family as the `df` /
Thor-`free` / cgroup-`usage_in_bytes` traps: **a measure that answers a narrower question than the one
being asked, read as the answer.**

---

## 4. What has to happen for a σ\* number — two routes, both now one command

### 4.1 Route A — the REF-C surface. **0 GPU. No training pause.** *(recommended first)*

The endpoints are ground truth, so `refc_dump_latents.py` grew a `--backfill-endpoints` mode that
imports **no model and no CUDA** and reads only pose arrays:

```
python stack/scripts/refc_dump_latents.py --backfill-endpoints \
    --dump-in ".../2026-08-04-lambda-findability/raw/latents_refc-xl-30k.pt" \
    --val <the val40 corpus> --endpoint-steps 20,60 \
    --out ".../raw/latents_refc-xl-30k-ep.pt"

python stack/scripts/e_wc2_sigma_star.py \
    --dump ".../raw/latents_refc-xl-30k-ep.pt" \
    --features pooled,ctx --out ".../raw/ewc2_sigma_star_refc-xl.json"
```

⛔ **The backfill REFUSES** unless (1) the rebuilt per-window `eid` equals the banked `eid`
element-for-element, and (2) the recomputed 2 s endpoint is **bit-identical** to the banked `gt`
column at the coinciding waypoint. Without that gate a one-window misalignment would regress every
latent onto a **neighbour's** endpoint and return an inflated σ that looks exactly like a measurement.
Both refusal paths are tested (`test_backfill_refuses_a_misaligned_grid`,
`test_backfill_refuses_when_the_pose_source_differs`).

**Cost, ESTIMATED:** the backfill is 40 mmap'd episode loads + `~881 × 2` pose gathers —
**< 1 minute, CPU, no GPU allocation**. E-WC2 itself on `pooled`(992) + `ctx`(96) at 40 LOEO folds
over 881 windows: **~1–3 minutes CPU** (40 economy SVDs of ≈`[860, 1088]`), plus the bootstrap.
**Total well under 10 minutes on one core.**

**The only real cost is data movement**, and it is a choice:
- run the backfill **on Thor**, CPU-only, ~1 min, no GPU touched — but it does add IO/CPU to a
  training machine, which the standing rule ("never add load to a pod that is training") makes a
  **PI call**, not mine; **or**
- pull the 40 val40 episodes from HF to this box (~1.4 GB ESTIMATED, the val40 subset of the 21.2 GB
  val split) and run everything locally, adding **zero** load to Thor. ← my recommendation.

⚠️ **Class stamp on the answer this route gives:** `EXPLORATORY`. It is REF-C-XL's fan at a 2 s
horizon — the same surface §3.1's entire requirement curve was measured on, which is exactly why it
composes: the **ratios** are the transferable claim, the absolute metres are that fan's. It answers
*"can a ridge on frozen latents reach σ\* on our corpus"* for the arm the threshold was derived on.

### 4.2 Route B — the surface §5.2 actually names: **frozen S-W latents. GPU required.**

S-W latents have never been dumped. This is the pass that needs the GPU and therefore a deliberate
pause on the Thor training run.

- **What to dump:** the v6 S-W trunk latents on the same val40 grid — `z_op` / `z_plan` (and `z_tac`
  where it exists) from `V6Model.forward`, plus the same `gt_endpoint` / `endpoint_valid` block.
  ⚠️ **`plan["feat"]` is ECHO-suspect**: `emit(z_op, g_tac_embed, v0)` takes `v0`, so anything
  downstream of it carries ego at inference. Every new block must be `--declare`d, and the instrument
  refuses undeclared blocks by design.
- **Cost, ESTIMATED:** one forward pass over 881 windows of 8 frames — the same shape as the
  2026-08-04 REF-C dump, which ran over 40 episodes on Thor. **~10–25 minutes of GPU** for a
  ≈336 M-param trunk at batch 4, plus model load. ⚠️ ESTIMATED, not measured: the REF-C dump's own
  wall-clock is in `…/2026-08-04-lambda-findability/raw/dump.log` and should be read before booking a
  pause. **Small batches, few workers on Thor** — throughput is flat 12.3–14.1 windows/s across a 6×
  batch range and each worker costs ~8.6 GB host RAM.
- **Blocked on:** a pause on the live Thor training, i.e. **a PI decision**. Nothing else.

**⭐ Do Route A first.** It is free, it uses the surface §3.1's threshold was derived on, and its
answer determines whether Route B's pause is worth booking at all: if a ridge on REF-C's frozen
latents cannot reach σ\* on the arm the threshold came from, an S-W re-measurement is unlikely to
change the sign, and `ANCHOR_GOAL` supervision is the live branch.

---

## 5. Tests

`stack/tests/test_e_wc2_sigma_star.py` — **62 tests, all on synthetic latents with a planted σ.**

| group | what is pinned |
|---|---|
| **σ recovery** | the LOEO ridge returns the planted σ within 6 % at 0.4 / 0.8 / 2.0 m; `sigma_radial = √2 × sigma_perax`; a hand case (`res = [3, 4]` → radial 5, per-axis 3.5355, long 3, lat 4); the published `0.8/0.4714 = 1.70` and `0.8/0.1639 = 4.88` reproduce §3.1's ratios (the unit pin) |
| **one ridge** | `ridge_oof_predict` ≡ `probe_latent_state.ridge_probe_cv` on R², per-fold R² and per-fold λ |
| **LOEO** | one episode per fold; **no episode straddles any fold**; refusal below 2 episodes; ⭐ the leaky window-random split understates σ **40×** (8.195 → 0.201) |
| **verdict bands** | FUNDED / INCONCLUSIVE / REFUSED at the exact boundaries 1.7 and 3.0, inclusive as written; guards beat any ratio; `NO_VERDICT ≠ REFUSED` |
| **refusal** | short n, short episodes, **relaxed guards**, missing 6 s, ECHO block, undeclared block, producer `instrument_fail`, fan-horizon mismatch |
| **§5.3** | fires above 3×, not below; **no branch emits a scaled 6 s threshold**; matched-window re-fit is used and the full-grid 2 s headline is preserved |
| **contract** | conformant dump passes; every missing piece is named; 20 **and** 60 required; `endpoint_valid` required |
| **producer** | the mask marks out-of-range horizons NaN + False; the endpoint is bit-identical to `gt` where horizons coincide; **K_MAX is not widened**; the backfill's two refusal paths |
| **CLI** | `--print-contract`; `--validate-only` exits 3; end-to-end JSON exits 0; `NO_VERDICT` exits 4 |
| **interval** | the estimator string says "episode", never "overlapping"; reducer `rms`; the ratio CI shares one episode draw |

Run: `cd stack && PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python -m pytest`.

---

## 6. Escalations — requests, not notes in a README

1. ⭐ **PI decision, and it is the only blocker on a σ\* number: how to reach the val40 poses.**
   Preferred: pull the 40 val40 episodes from HF to the dev box (≈1.4 GB, ESTIMATED) and run
   Route A entirely off Thor. Alternative: run the CPU-only backfill on Thor beside the training run.
   I did not choose, because the standing rule forbids me adding load to a training machine.
   **After that, σ\* is ~10 minutes of CPU away.**
2. **`Project Steering/V6F_PLANNER_DESIGN.md` §5.2 needs one edit:** *"Cost: 0 GPU, banked latents"*
   is true about compute and misleading about readiness. Suggested replacement — *"Cost: 0 GPU on the
   REF-C surface (latents banked in-repo at `…/2026-08-04-lambda-findability/raw/`); requires a
   pose-only endpoint backfill, and the val40 corpus on a reachable disk. The S-W surface §5.2 names
   has never been dumped and needs one GPU pass."* I did not edit the design doc — it is not mine.
3. **`RETRACTION_LOG.md`** should carry §3's root-cause class: *a cost estimate that priced the
   expensive input and not the cheap one* — the target was free to compute and therefore never
   costed, so a genuinely-0-GPU experiment was 12 days un-runnable for want of a one-minute step.
4. **Before any 6 s claim anywhere in the programme**, run the §5.3 check. If σ(6 s) > 3× σ(2 s), the
   1.7 / 3.0 thresholds are **not** transferable to 6 s and must be re-derived on a 6 s fan. The
   instrument will not scale them under any branch, so nothing downstream can quietly do it either.
5. ⚠️ **§3.1's σ unit should be stated explicitly in the design doc.** It is currently recoverable
   only by reading `sel_winners_curse_law.py:221`. A reader who assumes radial RMS is off by 1.414 on
   the number that decides SEL-1.
6. ⚠️ **Two concurrency hazards fired during this turn — both cost real time, both are cheap to
   avoid, and neither is specific to this work.**
   - **The index was reset under a running agent.** Five deliverables were `git add`ed and
     **verified present** with `git ls-files --cached`; three commits then landed
     (`dc50dbc`, `b12c190`, `8e215b3`) and the same check later reported them **untracked (`??`)**.
     Re-staging fixed it. ⇒ **Verify staging again at the END of a turn, not only at the moment you
     stage** — the existing rule ("`git add` exit codes are not evidence") is necessary but not
     sufficient; a *later* index reset defeats an earlier verification.
   - **A `file:line` citation drifted between reading a file and citing it.** Those same commits
     edited `probe_latent_state.py` and `sel_winners_curse_law.py`, and the σ-injection line moved
     `222 → 221`. Caught by re-verifying every cited line against the current files; 6 references
     across 3 files were corrected. ⇒ **Re-verify `file:line` citations against HEAD before filing**,
     and prefer citing a line *together with its verbatim text* so drift is detectable rather than
     silent. Root-cause class: **a citation is a claim, and it can go stale without anyone touching
     the document that makes it.**
   - Separately, one full-suite failure (`tests/test_p8.py::test_pos_weight_default_is_auto`) was a
     **concurrent-edit race**, not a defect: `train_p8_occupancy.py` is another agent's unstaged
     `+408/−57` diff and its mtime was *the second the suite was running*, so the module compiled
     from version A while `inspect.getsource` read version B. It passes in isolation
     (30 passed / 1 skipped) and with this work's tests alongside it (119 passed / 1 skipped).
     Evidence: `raw/PYTEST_EVIDENCE.txt`.

---

## 7. Deliverable manifest

| artifact | exact location | state |
|---|---|---|
| `e_wc2_sigma_star.py` — the estimator | `stack/scripts/e_wc2_sigma_star.py` | **staged** (repo working tree) |
| `test_e_wc2_sigma_star.py` — 62 tests | `stack/tests/test_e_wc2_sigma_star.py` | **staged** |
| `refc_dump_latents.py` — `--endpoint-steps`, `--backfill-endpoints`, `window_starts`, `K_MAX_GRID`, `gt_endpoints_masked`, `backfill_endpoints`, the endpoint self-control | `stack/scripts/refc_dump_latents.py` | **staged** (was untracked before this turn) |
| this writeup | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-ewc2-instrument/EWC2_INSTRUMENT.md` | **staged** |
| the banked surface E-WC2 will consume | `…/incoming/2026-08-04-lambda-findability/raw/latents_refc-{base,xl}-30k.pt` | already in the repo, unmodified by this turn |

**Nothing is stranded**: no pod, no worktree, no agent context. Everything above is in the repo
working tree and staged. ⛔ **Nothing was committed and nothing was pushed.**

**Not done, deliberately:** no σ\* was measured, no verdict was produced, no GPU was touched, no ssh
was opened. The instrument only, as briefed.
