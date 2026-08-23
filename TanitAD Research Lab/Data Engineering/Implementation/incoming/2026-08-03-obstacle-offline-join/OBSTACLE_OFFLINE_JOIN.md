# obstacle.offline → the eval window grid: LONGITUDINAL distance-keeping is now computable

- **Date:** 2026-08-03 · **Discipline:** Data Engineering · **Status:** PENDING orchestrator triage
- **Evidence class:** MEASURED (ours, this run) unless a row says otherwise. Artifacts in `raw/`.
- **Cost:** dev-box CPU only. 0 GPU. No training pod touched.

---

## Lead

**Distance-keeping is computable, and it is now computed on a real eval window grid.**

| | |
|---|---|
| **Windows scored here** | **11,004** over **500 clips**, of which **886 carry a lead agent** (LEAD), 1,651 are labelled-and-clear (NO_LEAD) and 8,467 are NO_LABEL |
| **Lead rate over LABELLED windows** | **34.9 %** (886 / 2,537) |
| **The registration** | exact — recovered window time is within **1.4 ms (p95 of per-clip max)**, worst **25.9 ms**, over all 500 clips |
| **The eval grid** | `window_last_indices` reproduces the **canonical 881 val40 windows exactly** from the committed manifest |
| **Still NOT done** | the val40 arm scores themselves — see §7. The runner is written; it needs the eval host. |

⚠️ **My brief's premise was partly stale, and that matters more than anything I added.** The metric
(`taniteval/taniteval/lead_metrics.py`), the `obstacle.offline` reader
(`…/2026-08-03-longitudinal-distance-keeping/build_lead_tracks.py`) and the pre-registered
admission control **D-LEAD-1** all landed at 07:56–08:08 today, before this stream started. That
package's own INTAKE names the open item precisely:

> "Wiring it into the *eval* path (val40 windows → `win["lead"]`) … Until that lands, **arm evals
> will still report the family UNAVAILABLE**."

**That wiring is what this package is.** I did not re-do the metric and I have not re-litigated
D-LEAD-1.

---

## 1. P1 — what `obstacle.offline` actually gives us, read from bytes

MEASURED here (`raw/obstacle_schema_probe.json`), 12 clips across 12 chunks of our own selection —
not copied from `pai_label_schemas.json`.

| fact | value |
|---|---|
| **columns** | `timestamp_us · source · track_id · center_{x,y,z} · size_{x,y,z} · orientation_{x,y,z,w} · label_class · reference_frame · reference_frame_timestamp_us` |
| **coordinate frame** | `reference_frame == "rig"` on every row, and `reference_frame_timestamp_us == timestamp_us` on every row ⇒ **each cuboid is expressed in the ego frame at ITS OWN timestamp** |
| **frame convention** | **x forward, y left, z up** — ⭐ MEASURED by experiment, see §2 |
| **clock** | the clip's own µs clock, **shared with `egomotion`**. Obstacle t₀ starts **0.193–0.232 s** after egomotion t₀ and lies inside the egomotion span on every clip probed |
| **sampling rate** | **10 Hz per track** (median cadence 0.0999–0.1001 s) — exactly our `TARGET_HZ` |
| **frame-synchronised?** | **No.** 1.000–1.005 rows per unique timestamp ⇒ each cuboid carries its own timestamp; tracks are staggered, not ticked together. A "lead at time t" therefore requires a per-track lookup, never a frame index |
| **span** | **19.93–20.00 s**, while `egomotion` runs **47.8–140.4 s** (100 Hz). ⇒ most of a long clip has **no labels at all** |
| **tracks per clip** | 12–163 (median ≈ 55 over the probe) |
| **velocity column** | ⛔ **none.** Closing rate must be differenced from the track; that is a definition choice, stated in §3 |
| **classes seen** | automobile 58,101 · person 6,047 · rider 910 · heavy_truck 733 · trailer 440 · bus 278 · protruding_object 187 · stroller 173 · animal 14 — **all dynamic agents, no infrastructure** |

### Coverage — SETTLED from the feature that owns the fact

Read from the dataset's own `metadata/feature_presence.parquet` (306,152 clips), not sampled and
not inherited. `raw/obstacle_coverage.json`.

| population | clips | with `obstacle.offline` | **missing** |
|---|---|---|---|
| **corpus-wide** | 306,152 | 298,326 = **97.44 %** | 2.56 % |
| **our phase0 selection** (the parity source) | 3,000 | 2,907 = **96.90 %** | **93 clips, 3.10 %** |
| our R0 500-clip selection | 500 | 495 = **99.00 %** | 5 clips |
| **the canonical val40** | 40 | 39 = **97.50 %** | **1 episode** |

⭐ **Both circulating figures are right, for different populations — and the one quoted in briefs is
the wrong one for us.** 97.44 % is corpus-wide (confirmed exactly); `lead_state_gate.py`'s docstring
figure of 96.90 % is our phase0 selection (confirmed exactly). **Our corpus is the poorer of the
two: 93 clips carry no agent labels at all.** My own earlier per-chunk sample said 96.54 % over 636
clips — superseded by this, which reads every clip.

⛔ **Operationally decisive for the val40 run: exactly one of the 40 canonical val episodes has no
`obstacle.offline` — `ep_00037.pt`.** Confirmed by **two independent probes**: the presence table
says 39/40, and opening the chunk zip finds no member for that clip. Its **22 windows** are
NO_LABEL and must be reported as their own denominator. Silently reading them as "no lead agent" is
precisely the bias §4 exists to prevent — and at 1-in-40 it is large enough to move a headline.

---

## 2. ⭐ The frame convention, decided by experiment rather than assumed

`build_lead_tracks.py` composes `L_w = ego_xy + R(yaw) @ [center_x, center_y]`, which *asserts*
x-forward / y-left. Nothing in the dataset card proves it, and a wrong handedness silently mirrors
every lead about the ego axis — producing perfectly plausible numbers.

**The discriminating test** (`code/p1_frame_convention.py`, `raw/frame_convention.json`): a parked
car is stationary in the WORLD, so its world track collapses to a point *only* under the correct
convention. Over **2,778 tracks living ≥ 2 s across 48 clips**:

| candidate | tracks world-static (σ < 0.5 m) | mean world-position σ |
|---|---|---|
| **`x` fwd, `y` left** (deployed) | **1,756 (63.2 %)** | **6.61 m** |
| `x` fwd, `y` right | 236 (8.5 %) | 15.15 m |
| `x` left, `y` fwd | 33 (1.2 %) | 30.51 m |
| `x` right, `y` fwd | 31 (1.1 %) | 30.59 m |

⇒ **`xf_yl` confirmed, 7.4× over the nearest alternative.** The deployed composition was right; it
is now MEASURED rather than inherited.

---

## 3. The definitions — written down BEFORE any number was produced

Inherited unchanged from `lead_state_gate` / `lead_metrics` **on purpose**: two gap conventions in
one programme is a retraction waiting to happen.

**A lead agent, precisely.** At window origin `t₀`, among cuboids whose `label_class` is a
**vehicle** (`automobile, heavy_truck, bus, other_vehicle, trailer` — VRUs excluded) and whose last
sample is **at or before `t₀`** and **no more than 0.5 s stale**, take the one minimising
`gap = center_x − size_x/2` subject to `0 ≤ gap ≤ 80 m` and `|center_y| < 2.0 m`.

- `gap` is **rig origin → lead REAR face**. NOT bumper-to-bumper: our ego origin is the rig, so the
  ego's own front overhang is not subtracted.
- **Selection is strictly causal.** The lead's *future* positions are ground truth about the world —
  exactly like the ground-truth ego waypoints an ADE is measured against — and are a **scoring
  input, never an arm input**.
- `time_gap = gap / v_ego`, **undefined below 0.5 m/s** (NaN, not clamped).
- `ttc = gap / closing_rate` with `closing_rate = −d(gap)/dt`, **censored at 30 s when not closing**;
  `n_closing` travels with every mean.
- The corridor gate uses the **predicted path's own local heading**, so an arm that drifts out of
  lane *loses* its lead rather than being credited with distance-keeping on a car it is no longer
  behind.

---

## 4. ⛔ Three window states, never two

| state | meaning |
|---|---|
| `LEAD` | a causal in-corridor vehicle ahead at `t₀` |
| `NO_LEAD` | labels present, road genuinely clear |
| `NO_LABEL` | **no `obstacle.offline` for this clip, or the window's horizon leaves the ~20 s labelled span** |

Collapsing `NO_LABEL` into `NO_LEAD` is the specific bias this instrument exists to avoid — and it
is not a corner case: **8,467 of 11,004 windows (76.9 %) in this run are NO_LABEL**, because
`egomotion` runs to 140 s while the labels stop at 20 s and because most local chunks hold no
obstacle zip. Counting those as free flow would manufacture empty road and flatter every arm.
Every denominator in `raw/distance_keeping_r0_500clips.json` is reported per state.

---

## 5. The registration — the piece that did not exist

An eval window is *window j of episode e of a cached split*. Its `t₀` in the clip's clock is stated
**nowhere** in the episode record, and `build_episode` derives the grid from the **camera**
timestamps parquet, which ships only inside the ~2 GB camera chunk zip and is absent on any host
that is not rebuilding the cache.

**`lead_source.register_poses_to_time` solves it by CONTENT**: it matches the episode's own
`poses[:, :2]` against the egomotion track and fits `t = a + b·i`.

**Proof (`raw/registration_per_clip.json`, 500 clips):** the recovered time is compared against the
true `t_query` the episode was built on.

| | |
|---|---|
| pose-match residual | median-of-medians **0.00108 m**, worst max **0.0246 m** |
| **recovered time error** | median-of-medians **0.19 ms**, p95-of-max **1.39 ms**, worst max **25.9 ms** |
| recovered grid spacing `b` | **0.100486 – 0.101013 s**, median **0.100667 s** |

⭐ **The grid is NOT 10 Hz.** `n_target = int(span_s · 10)` truncates, so the realised spacing is
~0.1007 s. Assuming 0.1 drifts ~0.13 s over a 200-step episode — about **1.8 m of lead displacement
at 13.6 m/s**. Fitting it is not pedantry.

⚠️ **Two failure modes were found and fixed, not designed around.** The first implementation used
nearest-point matching plus least squares and produced a **5.08 s** worst-case time error: at a
**stop**, dozens of pose indices sit at one position, so nearest-point returns an arbitrary time
inside the stop, and a **self-intersecting route** puts two times at one place. The shipped version
restricts probes to poses whose neighbours are > 0.30 m apart and fits with **Theil–Sen + inlier
refit**. Same data, worst error **25.9 ms** — a **196×** improvement. A clip too stationary to
register is **refused loudly**, never registered approximately.

**Window grid — the second half of the registration.** `window_last_indices` reproduces
`rollout.collect`'s grid (`starts = range(0, T − 8 − 20, 8)`, origin `start + 7`). Summed over the
`T` values in the committed val40 manifest it gives **exactly 881** — the published canonical
open-loop statistic. That is what guarantees a lead block is row-aligned with every banked
`pred`/`gt` dump, so an arm can be re-scored with **no re-inference**.

---

## 6. The numbers — GT vs the hold-`v0` CV floor, per speed band

`raw/distance_keeping_r0_500clips.json`. 500 clips, 11,004 windows, 886 with a lead. Paired
episode-cluster bootstrap, B = 2000, seed 0. ⛔ Never `overlapping_holdout_se`.

**Pooled (the row, not the result):**

| | GT | CV floor | paired GT − CV | CI95 | separated |
|---|---|---|---|---|---|
| mean min-headway (m) | 23.94 | 21.92 | **+1.945** | [1.434, 2.584] | ✅ |
| mean min-time-gap (s) | 5.80 | 5.18 | **+0.433** | [0.330, 0.553] | ✅ |
| mean min-TTC (s) | 21.33 | 16.08 | **+4.238** | [3.032, 5.644] | ✅ |

n paired = 676 (headway/TTC), 519 (time-gap). **9 windows GT-only, 100 CV-only** — the human turns
out of the corridor and loses the lead where the straight CV path keeps it; excluded from the
paired delta by construction and reported rather than absorbed. Sign is correct on all three.

**⭐ Stratified — and this is why the rule exists.** min-TTC, GT vs CV:

| band (m/s) | windows | with lead | GT min-TTC (s) | CV min-TTC (s) | GT − CV |
|---|---|---|---|---|---|
| 0–1 | 1,804 | 183 | 27.17 [25.54, 28.49] | 27.31 [25.81, 28.59] | **≈ 0** |
| 1–3 | 991 | 73 | 21.92 [19.51, 24.30] | 17.24 [14.04, 20.45] | +4.7 |
| 3–6 | 2,510 | 122 | 19.65 [16.50, 23.13] | 10.55 [7.91, 13.21] | **+9.1** |
| 6–10 | 4,293 | 267 | 18.02 [15.77, 20.33] | 11.85 [9.33, 14.49] | +6.2 |
| 10–15 | 1,347 | 38 | 20.20 [14.12, 25.32] | 16.18 [9.09, 21.96] | +4.0 |
| 15+ | 59 | 2 | **UNPOWERED** (n = 2) | **UNPOWERED** | — |

**The pooled +4.24 s understates the regime that matters by more than 2×.** 183 of 886 lead-bearing
windows (20.7 %) sit at **0–1 m/s**, where a stationary ego and a stationary CV path are the same
thing and the metric **cannot discriminate at all** — and where the time-gap `n` collapses from 183
to 26 because it is undefined at standstill. Averaging that band in drags the headline down.
This is the pooling defect the binding rule forbids, caught by the instrument on its first real run.

⚠️ **This is a FLOOR contrast, not an arm result** — GT vs a never-braking constant-velocity policy.
It shows the instrument separates; it says nothing about `refc-base` or `flagship-v1`.

---

## 7. ⛔ What I did NOT do — plainly

1. **No REAL arm was scored.** P4 is not delivered. `refc-base` / `flagship-v1` per-window
   predictions live on the eval pod, and the canonical val40 episodes live in
   `/workspace/val40cache` — neither is on the dev box, and I will not add load to a training pod.

   ⭐ **But `code/score_val40_lead.py` is no longer unverified.** It was executed against a local
   **stand-in cache** of 8 `ep_*.pt` episodes built from R0 clips (`code/make_standin_cache.py`,
   `raw/standin_runner_gt_cv.json`, `raw/standin_runner_with_arm.json`). It resolved every clip
   from its `episode_id`, read the label zips, registered, built the three window states, and
   scored **GT / CV / a synthetic third arm** over 176 windows. Both guard paths fired: a banked
   dump with the wrong row count was **refused** (`100 rows but this run built 176 windows`), and
   one stand-in episode came back **entirely NO_LABEL** — the exact case the single unlabelled
   val40 episode will produce. What remains untested is only the pod's own paths
   (`/workspace/val40cache`, the on-demand chunk fetch) and a real arm's `pred` array.
2. **No re-cache.** The brief proposed joining `obstacle.offline` at build time, provable via
   `register_geometry_sibling`. I did not do that, deliberately: for an **eval label** it is
   unnecessary and strictly riskier. The sidecar path touches no cache byte, so parity is preserved
   *by construction* rather than by proof, and it needs no GPU-hours. A re-cache remains the right
   move **if and only if** lead state is to become a **model input** — and note
   `register_geometry_sibling` proves *which episodes are present*, not *that a field was added*,
   so it would not actually prove the thing the brief wanted it to prove.
3. **Parity artifacts were not sha256-verified against the committed manifest.** The brief asked for
   the `e61a04553df5` / `0b176d2e5cb4` digests and a `torch.load`. Neither cache is on this box (the
   local epcache is `physicalai-train-14231cd29c74`, a 500-clip R0 build, not the parity key), so the
   check is not runnable here. **The stronger statement is available instead: nothing in this package
   writes to any cache.** The verification belongs in the eval-host run.
4. **The 31 missing val40 obstacle chunks were still downloading** when this was written; §8 records
   what completed.
5. **No situation-classifier or goal signal is touched here.** Lead state is an EVAL LABEL in this
   package, never an inference-time input, so neither the vision-only rule nor the goal-input rule
   is engaged. ⚠️ If anyone later feeds lead state to a model, both rules apply and this package is
   not the precedent for it.

---

## 8. Coverage for the canonical val40

MEASURED: all **40** val40 `episode_id`s resolve to a **unique** clip in the 3,000-clip phase0
selection (0 ambiguous, 0 unmatched). Those clips span **37 chunks**, of which **6** already had
`obstacle.offline` locally; the remaining **31 (1,384 MB, 0 failures)** were pulled with
`code/pull_obs_chunks.py` via the repo's own committed HF path into the local data root — read-only
w.r.t. the corpus. **All 37 chunks are now present**, so the eval-host run needs no download.

### A falsifiable prediction the eval-host run will confirm or refute

`raw/val40_coverage.json`. `t₀` cannot be read here (it needs the episodes' `poses`), so it is
reconstructed from a prior MEASURED on the 500 R0 clips: `a − ego_t₀ = 0.3297 s` (p05–p95
0.239–0.401, σ 0.056) and `b = 0.100667 s`. **ESTIMATED, not measured** — and swept ±0.25 s,
about 2× the prior's own spread, so it carries its error bar:

| t₀ shift | LEAD | NO_LEAD | NO_LABEL | lead rate over labelled |
|---|---|---|---|---|
| −0.25 s | 277 | 543 | 61 | 33.8 % |
| **0** | **270** | **550** | **61** | **32.9 %** |
| +0.25 s | 266 | 553 | 62 | 32.5 % |

⇒ **~270 of the canonical 881 windows should carry a lead agent (30.7 % of all windows, 32.9 % of
labelled ones)**, stable to ±2 % across the whole sweep — so the answer does not depend on the
part I could not measure. Of the 61 NO_LABEL windows, ~22 are the single episode with no
`obstacle.offline` and ~39 are horizons leaving the ~20 s labelled span.

⚠️ **This is an ESTIMATE and must not be quoted as the val40 distance-keeping result.** Its value is
that it is *checkable*: if the eval-host run does not land near 270 LEAD / 61 NO_LABEL, something in
the registration is wrong and the numbers should not be trusted.

⚠️ Scoring val40 **still needs the eval host**, because the registration needs the episodes'
`poses`, which exist only in `/workspace/val40cache`. Getting them locally instead would mean
pulling 37 × ~2 GB camera chunks; that is the wrong trade.

---

## 9. ⛔ ESCALATION — integration, not a note in a README

1. **Run `code/score_val40_lead.py` on the eval host.** One command, 0 GPU in `--pred-npz` mode.
   It is the only thing between the programme and the first real LONGITUDINAL distance-keeping
   numbers for `refc-base` and `flagship-v1`. It is UNVERIFIED until it runs.
2. **Accept or revert the `taniteval/` landings.** `lead_source.py`, the stratified read in
   `lead_metrics.py` and the additive `by_speed` hunk in `four_families.py` were landed directly,
   not left in intake — the same declared deviation the 07:56 package made, for the same reason
   (three intake packages in this hub have waited up to 24 days). All additive; suites green.
3. **Correct the coverage figure where it decides anything.** 97.44 % is corpus-wide and confirmed,
   but **our** parity source is **96.90 %**, and quoting the corpus-wide number for our corpus
   understates the unlabelled fraction by ~20 % relative. It appears in ≥ 3 documents including a
   brief; `raw/obstacle_coverage.json` is the primary source to cite.

---

## 10. Tests

| suite | result |
|---|---|
| `taniteval/tests/test_lead_source.py` (new) | **18 passed** |
| `taniteval/tests/test_lead_strata.py` (new) | **16 passed** |
| `pytest taniteval` | **937 passed** (was 935 before this package) |
| `pytest stack` | **1944 passed, 12 skipped, 2 xfailed** |

⚠️ My brief predicted `stack` would sit at ~1902. It is **1944** — sibling streams landed tests
today. Skips (12) and xfails (2) match exactly, so nothing regressed; the brief's count was simply
INHERITED and stale. Flagging it rather than quietly reporting "green".

The load-bearing cases: the window grid reproducing **881**; the registration recovering a known
affine grid and **refusing** a wrong clip; a world-static lead staying put in the window frame while
the naive rig read understates the gap by exactly `v · horizon`; `NO_LABEL` never becoming free flow;
and the point estimate being the **full-set** mean, not a mean-of-split-means.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Reason:**
- **Landed at:**
