# DiffusionDriveV2 — the "usable now" bucket, implemented and measured

**status: COMPLETE.** Of the five pieces `…/2026-09-05-diffusiondrive-v2-analysis/RESULT.md` §3.2
marks **usable now**, **four were already in the tree** when this work package opened and are
re-verified here rather than inherited; **the fifth (selector augmentation + foreign vocabulary) did
not exist and now does**; and the piece-4 selector, which existed as a library with **no caller**,
now has one. **52 new tests green, 8 of 8 deliberate regressions RED.**
⛔ **No training run was launched. No capability claim is made.**

**Stream:** Research Lab · Arch+Inference · **Date:** 2026-09-10 · **Branch:** `agent/arch-inf-20260803`
· **Zero GPU** (CPU only, dev box) · Extends `H-DDA-5..7` / `E-DDA-2b`; adds `D-DDV2-PORT-1..4`.

---

## 0. The answer in one paragraph

§3.2's four-bucket table is **stale in the part that matters most to a builder**: it says the ≥GT
positive mask and the two-scalar exploration are *"usable now, **not yet in `rl/`**"*, and both had
landed **the same day the analysis was written** (`stack/tanitad/rl/advantage.py`,
`config.py`, `posttrain.py`, `refcv3_adapter.py`, all 2026-09-05, pinned by
`stack/tests/test_rl_v2_faithful.py`, 69 green — MEASURED here, not read from a changelog). The
selector — sub-score heads, BCE, margin-rank, coarse-to-fine, candidate self-attention — had also
landed as `refs/refc_selector.py` + `refc_selector_targets.py`, but **nothing imported it except an
import-probe and its own test**, so "usable now" was true of the library and false of the
programme. The one genuinely missing mechanism was the **fifth**: V2's `add_mul_noise` fan
augmentation and its foreign-vocabulary mixing. That is now
`stack/tanitad/refs/refc_selector_aug.py`, ported from the **banked upstream source**
(sha256-verified, not from the summary), and it carries one fact the analysis does not state and
which decides the port's default: **the foreign bank is TRAIN-ONLY in V2** — every `vocab`
reference lives inside `forward_train_rl`, and `forward_test_rl` mixes none. The caller is
`stack/scripts/refc_selector_e2b.py`, which runs the whole stage-II path on CPU today with a
**no-information control that reads its known value exactly (rank AUC 0.500000)**, a **raw-input
floor** the learned selector must beat (and does, 0.9323 vs 0.8757 on this rig), a **tie-group AP**
whose constant arm lands on the base rate to the last digit, and the pre-registered
**progress-only deliberate regression**, which reads AP **below** the base rate — i.e. it selects
*for* contact, exactly as `H-DDA-7` predicted.

---

## 1. Piece by piece: what landed, and what proves it

⚠️ Evidence class on every row. `PUBLISHED-CODE` = `hustvl/DiffusionDriveV2@1cd12a1`, read from the
in-repo bank at `…/2026-09-05-diffusiondrive-v2-analysis/raw/ddv2_src/`, **content-verified**:
`diffusiondrivev2_model_sel.py` sha256
`2218e4ee79f5952f4edf474c3c2796f91f0cdaf593b3ce987cec5729e1c310b6`, re-hashed 2026-09-10 and equal
to `raw/ddv2_fetched_sha256.txt`.

| # | §3.2 piece | §3.2 said | MEASURED 2026-09-10 | proof |
|---|---|---|---|---|
| 1 | **≥ GT positive mask** | *"usable now, **not yet in `rl/`**"* | ⛔ **STALE — it is in `rl/`.** `advantage.truncated_inter_anchor_advantage(gt_bar=…)`, `PostTrainConfig.use_gt_bar` (default `False`), and `posttrain.rl_objective` which **refuses** a configured bar with no value AND a bar the config did not declare | `raw/PRELANDED_verification.log`; `test_rl_v2_faithful.py` (10 tests); hand-computed control matches |
| 2 | **Intra-anchor grouping, G ≥ 2** | *"usable now on the surrogate policy"* | ✅ **present and enforced.** `grpo_advantage` **raises** on G = 1 rather than returning a silent zero; G = 2 gives the analytic `[-0.5, +0.5]` **exactly**; `composite_advantage` reports `frac_above_bar` (0.5000 on the analytic two-window case) | `raw/PRELANDED_verification.log` |
| 3 | **Scale-only 2-scalar exploration** | *"a one-line change"* | ⛔ **STALE — landed.** `PostTrainConfig.noise_mode="two_scalar"`, default still `"multiplicative"`. The per-axis ratio's spread **along** the trajectory is `3.8e-07` (analytic 0) vs `9.9e-01` for the per-coordinate default | `raw/PRELANDED_verification.log`; `test_rl_v2_faithful.py` |
| 4 | **The selector** (sub-score heads + BCE + margin-rank, coarse-to-fine, candidate self-attention) | *"usable now on the deterministic fan"* | ⚠️ **HALF TRUE.** The library existed (`refc_selector.py` 775 lines, `refc_selector_targets.py` 598) and is correct. **Nothing called it** — the only importers were `scripts/refcv5_preflight.py` (an import probe) and its own test. ⇒ *"usable"* was a property of the code, not of the programme. **A caller now exists.** | `stack/scripts/refc_selector_e2b.py`; `raw/PANEL.log`; 27 new tests |
| 5 | **Selector augmentation + foreign vocabulary** | *"usable now — our foreign vocabulary is refcv3's synthetic bank"* | ⛔ **DID NOT EXIST — now implemented.** `stack/tanitad/refs/refc_selector_aug.py` (`add_mul_noise`, `mix_foreign_bank`, `augment_fan`, `assert_pick_emittable`, `origin_summary`), everything OFF at the defaults | 25 new tests; mutations M1/M2/M3/M6 RED |

### 1.1 What the banked source says that the analysis does not

Read from `diffusiondrivev2_model_sel.py` at the verified sha256 (`PUBLISHED-CODE`):

* **`add_mul_noise` (`:1270-1287`)** draws **ONE std scalar per augmentation round for the entire
  batch** (`torch.empty(1).uniform_(std_min, std_max).item()`), then **two Gaussian scalars per
  (row, candidate)** — a `[B,N,1,1]` "horizon" and a `[B,N,1,1]` "vert" — concatenated and repeated
  over the waypoints. The original fan is kept as copy 0. ⇒ **the selector's augmentation is the
  same two-scalar law the RL stage explores in** (`D-DDV2-CODE-2`); one mechanism, two sites. The
  shared-per-round std is ported faithfully and the per-row variant is offered as a **named
  deviation** (`AugmentConfig.std_per_row`, default `False`).
* ⭐ **`int(3277 * 0.01) = 32`** — the keep-count formula reproduces the *"≈ 32 per scene"* the
  2026-09-05 analysis published from the other direction (the `[:, ::5]` slice and
  `dropout_ratio = 0.99`). An **independently authored** cross-check, not our own arithmetic
  replayed.
* ⛔⛔ **THE FOREIGN BANK IS TRAIN-ONLY, AND THE ANALYSIS DOES NOT SAY SO.** MEASURED by enumerating
  every `vocab` reference in that file — `:930`, `:932`, `:1202-1206`, `:1352`, `:1354` — **all
  inside `forward_train_rl` (`:1288-1389`) or the helper it calls**. `forward_test_rl` (`:1390-`)
  applies `add_mul_noise` at `:1444` and mixes **no vocabulary at all**.
  ⇒ `AugmentConfig.foreign_train_only` defaults to `True`.
  ⛔ **Why that is not a detail:** a foreign candidate is by construction *not something the
  generator can emit*. If the selector's argmax may land on one at inference, the arm's "selected
  trajectory" belongs to the bank, not to REF-C, and no downstream metric can tell — the number
  simply comes out better. `assert_pick_emittable` is the door lock; `augment_fan` returns
  `emittable` so a caller cannot claim it did not know; and mutation **M3** (relabelling foreign as
  emittable) goes RED on three tests.

---

## 2. Is every default path bitwise unchanged — and can the comparison FAIL?

⛔ The brief's warning is the right one: `WP-B`'s removability proof was **green with its head
deliberately corrupted**. So each new path was corrupted and **both halves** asserted.

| new path | OFF comparison under corruption | ON comparison under corruption | verdict |
|---|---|---|---|
| `--n-aug` (augmentation) | **unchanged** (`torch.equal` True) | **changes** (True) | ✅ non-vacuous |
| `--foreign-frac` (bank) | **unchanged** | **changes** | ✅ non-vacuous |
| `gt_bar` (piece 1, pre-landed) | **unchanged** | **changes** | ✅ non-vacuous |

⭐ **And the OFF proof is made in the STRONG form: the OFF path consumes ZERO RNG draws.** Tensor
equality alone would not have caught an augmentation that returned its input while advancing the
global generator — that would silently re-roll every arm downstream of it.
`test_OFF_DRAWS_NO_RANDOM_NUMBERS_so_a_downstream_stream_is_unchanged` seeds, calls the OFF path,
and asserts the next `randn` is bit-identical to a run in which the module was never called;
mutation **M2** (an OFF path that draws) goes RED on exactly that test.

⚠️ **One honest asymmetry, stated rather than hidden.** Switching `noise_mode` to `"two_scalar"`
changes the *shape* of the RNG draw (`[B,N,G,1,2]` instead of `[B,N,G,S,2]`), so the ON arm's
downstream RNG tail differs. That is a property of the ON arm, not a change to the OFF arm, and it
is why `noise_mode` is a declared field written into `config.json` by `PostTrainConfig.to_dict()`.

### 2.1 ⛔ Two mistakes of mine that the mutations caught, kept as the record

1. **My straight-line curvature control was blind to the defect it existed for.** On a **pure +x**
   fan the y column is 0 and stays 0, so per-coordinate noise leaves curvature at **exactly 0.0** —
   the control would have been GREEN against the very defect §3.2 names. The fix is an **oblique**
   fan; the mutation arm that found it is kept
   (`test_MUTATION_per_coordinate_noise_bends_an_OBLIQUE_line_but_not_an_axial_one`) and pins both
   halves. This is the *"a check that shares the defect it checks for"* class, in my own test.
2. ⛔ **`--foreign-at-eval` shipped INERT in my own driver — the `--wp-index` failure reproduced
   inside the package that cites it.** The first version scored the **native** fan at eval
   unconditionally, so the flag was parsed, stamped into the run record, and could not change a
   single number; and `assert_pick_emittable`, written specifically to stop a foreign trajectory
   reaching an emission, **was never called**. Fixed by adding `--eval-aug` (V2 does augment at
   test, `_model_sel.py:1444`, and mixes no vocabulary there), a `needs` row that **REFUSES the
   launch** when `--foreign-at-eval` cannot reach a pick, and a live call to the guard on the eval
   path. Pinned by `test_an_INERT_foreign_at_eval_is_REFUSED_before_the_run` and
   `test_the_emittable_guard_is_LIVE_on_the_eval_path`; mutations **M7** and **M8** go RED.
   ⭐ Found by asking the question the effective-weights doctrine asks — *what does this knob
   actually reach?* — not by a test.
3. **My first M5 mutation was arithmetically identical to the correct code.** Replacing the
   tie-group credit with a `for _ in range(tp_g): ap += cum_tp/cum_n` loop gives **the same number**,
   because `cum_tp`/`cum_n` are already advanced to the group's end. It read **STILL GREEN and was
   right to.** The real defect is **tie detection**, so that is what M5 now disables — and it goes
   RED on two tests. Same class as (1), inside a deliberate regression.

---

## 3. Does every new flag appear in the effective-weights stamp?

**Yes, and the check is derived from `argparse`, in both directions.**
`term_specs()` walks `parser._action_groups["E-DDA-2b knobs"]._group_actions` — never a hand list —
and `test_EVERY_argparse_knob_appears_in_the_stamp_and_nothing_else_does` asserts
`{row.term} == set(knob_dests(parser)) == set(stamp["values"])`. **30 knobs, 30 rows, 30 values.**
Mutation **M4** (drop one knob from the stamp) goes RED.

The stamp also classifies rather than merely records, using `tanitad.effective_weights`:

* `--foreign-frac 0.0001` over a 6-trajectory bank keeps `int(6*0.0001) = 0` ⇒ `NO_GRAPH`, with the
  reason `"the knob would be INERT"`. That is the `--wp-index` failure caught **before** a launch.
* `--top-k 999` over a 36-candidate fan ⇒ `"the coarse prune never fires and the coarse/fine split
  is inert"`.
* `--steps 6 --arm const` ⇒ **the launch is REFUSED** (exit 2): the operator typed a knob the arm
  discards, and the run record would have advertised `steps: 6` for an arm that never trains.
  `--allow-discarded-weights` records the override, per the `--refuse-unreached` precedent.

⭐ **A defect found in `effective_weights.explicit_dests` while wiring this** (MEASURED, reported not
patched): it gives each **action** its own sentinel but keys the returned map by **dest**, so a dest
served by **two actions** (the `--flag` / `--no-flag` pattern) has its first sentinel compared
against the second and reads **EXPLICIT on every launch** — which then trips the `DISCARDED`
refusal on a flag nobody typed. Observed with `--self-attn` / `--no-self-attn`. Workaround used
here: **one action per dest** (`--self-attn {on,off}`), asserted by
`assert len(dests) == len(set(dests))`. ⛔ Escalated as integration request #3 rather than patched
mid-flight, because `effective_weights.py` is shared by live launchers and the mount is degraded.

---

## 4. The panel — MEASURED, with n and d on every row

⛔ **TIER: a T0 analytic rig. NOT a driving claim, and it cannot become one here.** The fan is
36 constant-curvature / constant-acceleration arcs built by the script; the scene tokens encode the
same lead track the targets read, so the task is **easy by design**. The only admissible readings
are (a) the instrument runs end to end, (b) the **controls read their known values**, and (c) the
arms are separable. `--source dump` — the real refcv4 fan plus the `obstacle.offline` replay join —
**exits 3 and says it is blocked**, rather than inventing a fan.

`n`: **24** windows (12 fit / **12 scored**, disjoint), **36** candidates/window, **432** scored
candidates, **7,351** within-window rank pairs, **432** AP samples.
Base rate of the no-contact label: **0.946759**. Artifacts: `raw/panel/<arm>/e2b_*.json`.

| arm | d | params | rank AUC | AP (tie-group) | pick regret |
|---|---|---|---|---|---|
| ⛔ `const` — no-information control | 0 | — | **0.500000** | **0.946759** | 0.117783 |
| ⛔ `raw_floor` — ridge on raw coords | 25 | — | 0.875663 | 0.981269 | 0.037359 |
| `selector` | 64 | 298,764 | **0.932254** | 0.993711 | **0.008381** |
| `selector`, self-attn OFF | 64 | 231,692 | 0.934227 | 0.995529 | 0.001707 |
| `selector` + `--n-aug 2` | 64 | 298,764 | 0.893960 | 0.995396 | 0.069303 |
| `selector` + `--n-aug 2 --foreign-frac 0.01` | 64 | 298,764 (bank 264) | 0.893008 | 0.995888 | 0.062430 |
| ⛔ `progress_only` — deliberate regression | 64 | 298,764 | 0.568086 | **0.910917** | 0.270472 |

**What is readable here, and only this:**

1. ⭐ **The controls read their known values EXACTLY.** `const` reads rank AUC **0.500000** — the
   analytic no-information value, because a constant scorer ties every pair — and tie-group AP
   **0.946759**, the base rate **to the last digit**. The naive per-sample AP on the same arm reads
   **0.940207**, i.e. **−0.006552 off the base rate**. ⚠️ **Sharpening of the brief's warning: the
   naive form's bias is DIRECTIONAL, not always upward** — it inflates when the positives sit early
   in the array and deflates when they sit late (both pinned as literals in
   `test_MUTATION_the_naive_AP_misreads_a_constant_arm_IN_BOTH_DIRECTIONS`). The programme's earlier
   +22.8 % and +0.881 % are the inflating case; this rig is the deflating one. Either way it is not
   the base rate, and mutation **M5** proves the guard can fail.
2. ⭐ **The learned selector beats the raw-input floor** — AUC 0.932254 vs 0.875663, regret
   0.008381 vs 0.037359 — so on this rig it has added something over raw coordinates. ⚠️ On **this
   rig**. Nothing about refcv4 follows.
3. ⭐ **The deliberate regression behaves as `H-DDA-7` pre-registered.** `progress_only` reads AP
   **0.910917, BELOW the 0.946759 base rate** — a progress-only head does not merely fail to help,
   it **selects for contact**. ⇒ the contact readout is **not blind**, so the panel is not void.
4. ⚠️ **THE RIG CANNOT DISCRIMINATE THE SELF-ATTENTION LEVER, AND SAYING SO IS THE RESULT.**
   Self-attn OFF reads *slightly better* (0.934227 vs 0.932254; regret 0.001707 vs 0.008381) on
   **one seed, 12 scored windows, no interval**. ⛔ That is **not** evidence against candidate
   self-attention: 67,072 fewer parameters on a task this easy is a capacity story, and
   `H-ESTIM-SEED-1` says a one-seed difference is not a lever effect at all. The honest statement is
   *the rig is underpowered for this comparison*; the discriminating experiment is a replicate arm
   on the real fan.
5. ⚠️ **THE AUGMENTED ARMS' LOWER AUC IS A BUDGET CONFOUND, NOT A RESULT.** `--n-aug 2` triples the
   training candidate set at an unchanged **200** steps, so the augmented arm is under-trained
   relative to the control. Reporting "augmentation hurts" from this table would be exactly the
   scope error `CLAUDE.md` keeps paying for. What IS readable: the augmented path **runs**, its
   `n_total` and `origin` counts are correct, its AP is not degraded (0.995396 / 0.995888 vs
   0.993711), and **no foreign candidate reaches the eval fan** (`n_scored_candidates` stays 432
   because `foreign_train_only` holds).

### 4.1 The analytic cross-check the rig runs on itself

`--self-check` compares the curvature the target code **recovers** against the curvature the rig
**built** — a reference the target code never computed:

* ⭐ **`straight_arm_kappa_max = 0.0`** — the κ = 0 arm, **exact**, no discretisation error. This is
  the statement a sign flip or an axis swap breaks first (the `WP-A` mirrored-azimuth class).
* **`sign_agreement = 1.0`** and **`rank_agreement_monotone = True`** — exact, discretisation-free.
* ⚠️ **`kappa_rel_err_max = 0.0373` at `kappa_ds_rad = 0.302`** — **not** a defect: `rewards.kinematics`
  is a chord estimator and a few percent is the geometry. Quoted **with its cause** so a reader
  cannot mistake the residual for an error.

⛔ **This check FAILED on its first run and the failure was real** — max abs error **1.278** against
a built |κ| ≤ 0.06 on **16 of 32** candidates. Root cause: at `a = −2.0 m/s²`, `v0 = 10`, `T = 6 s`
the candidate's speed crosses zero at **t = 5 s** and the path **reverses**, so `ds → 0` and the
recovered curvature is meaningless. It looked exactly like a broken curvature instrument; the
instrument was fine and the **fan was unphysical**. `RigSpec` now derives `accel_min` analytically
from `end_speed_frac` and `build_arcs` **refuses** a reversing grid
(`test_a_fan_whose_candidates_REVERSE_is_refused`).

---

## 5. What went RED

`raw/RED_mutations.log` — eight deliberate regressions, each reintroducing a defect this programme
has really paid for, applied to the working tree, run, and restored. **All eight RED; the restore
check is green (52 passed).**

| id | reintroduces | RED tests |
|---|---|---|
| **M1** | per-coordinate multiplicative noise instead of V2's two scalars — the exact gap §3.2 names | 2 |
| **M2** | an OFF path that still advances the RNG | 1 |
| **M3** | a FOREIGN candidate marked emittable | 3 |
| **M4** | a knob parsed and not stamped — the `--wp-index` failure | 2 |
| **M5** | AP with ties broken by array order | 2 |
| **M6** | a foreign bank at a different horizon silently accepted — the derived-constant trap | 1 |
| **M7** | the emittable guard EXISTS but is never CALLED | 1 |
| **M8** | the `--foreign-at-eval` inert-knob refusal removed | 1 |

⚠️ **M5's first version read STILL GREEN and that was correct** — see §2.1(2). A mutation runner
that treats "still green" as a pass would have recorded a guard that does not exist;
`run_mutations.py` prints it as `⛔ STILL GREEN` and exits non-zero, because *a mutation that does
not go RED is a finding, not a pass*.

### 5.1 The full stack suite, and the attribution done by MEASUREMENT

`raw/FULL_SUITE.log`: **7,335 passed, 20 failed, 7 errors, 117 skipped** in 855 s
(`cd stack && pytest -q`, mirror recipe).

⛔ **Those 20 + 7 are PRE-EXISTING and this package did not cause them — established by
measurement, not by inference.** Re-running **only** the 12 failing modules in a session that
contains **neither of my test files** reproduces **exactly 20 failed and 7 errors** (288 passed,
54 s). Two supporting facts, each with its control: no failing module references
`refc_selector_aug` or `refc_selector_e2b` (**0 of 12**, control: the grep finds it in my own test
file, 1); and this package **edits no existing file** — `find -newermt 2026-09-10` over
`stack/{tanitad,scripts,tests}` names exactly the **4 new files** and nothing else.
⚠️ Their causes are environment, not code: `test_eval_contamination`, `test_decision_check`,
`test_runbook_commands`, `test_v6_chain` and `test_secret_scan` read banked manifests, a git hook
and `stack/experiments/`, and the documented dev-box mirror recipe **excludes `stack/experiments/`**
precisely because *"~4 test modules fail COLLECTION — expected, not a regression"*.
⇒ **Not claimed here: that the suite is green on a full checkout.** What is claimed is that this
package's delta is 0 failures, and that is what the isolated re-run measures.

---

## 6. Evidence-class ledger for every number above

| number | class |
|---|---|
| *"the 20 full-suite failures are pre-existing"* | **MEASURED** — identical counts from an isolated re-run without this package's test files (`raw/FULL_SUITE.log` + §5.1) |
| V2's `add_mul_noise` shape, the train-only vocabulary, `dropout_ratio`, `[:, ::5]` | `PUBLISHED-CODE`, `hustvl/DiffusionDriveV2@1cd12a1`, in-repo bank, sha256 `2218e4ee…` re-verified 2026-09-10 |
| *"≈ 32 per scene"*, the ablation deltas, PDMS 88.1 → 91.2 | `PUBLISHED-PRIMARY` via `…/2026-09-05-diffusiondrive-v2-analysis/RESULT.md` (not re-derived here) |
| every panel number, every test count, the curvature errors, the RNG facts, the `explicit_dests` defect | **MEASURED** (ours, this run; artifacts named in §7) |
| `D-REFCV3-AXIS1`'s 92.2 %, `D-REFCV3-40284a`'s 0.0751 m selection gap | `INHERITED` — quoted as motivation only; **no claim here rests on them** |
| "the selector will close the selection gap on refcv4" | ⛔ **not claimed.** That is `E-DDA-2b`'s question and it needs a GPU |

---

## 7. Manifest

⚠️ Nothing lives in only one place: every file below is in the repo; the `C:/Users/Admin/tanitad-wt`
mirror is a disposable run surface and the scratchpad copies are duplicates.

| artifact | where it lives | new? |
|---|---|---|
| this report | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-10-ddv2-usable-now/RESULT.md` | new |
| **piece 5** — fan augmentation + foreign bank | `repo:stack/tanitad/refs/refc_selector_aug.py` | **new, 396 lines** |
| **piece 4's caller** — the E-DDA-2b driver | `repo:stack/scripts/refc_selector_e2b.py` | **new, 823 lines** |
| tests — augmentation | `repo:stack/tests/test_refc_selector_aug.py` (25) | new |
| tests — driver, metrics, stamp, corruption pairs | `repo:stack/tests/test_refc_selector_e2b.py` (27) | new |
| mutation runner | `repo:…/2026-09-10-ddv2-usable-now/code/run_mutations.py` | new |
| pre-landed verification script | `repo:…/code/verify_prelanded.py` | new |
| GREEN baseline log | `repo:…/raw/GREEN_baseline.log` | new |
| **RED mutation log (8/8)** | `repo:…/raw/RED_mutations.log` (8/8 RED) | new |
| pre-landed pieces 1–3 verification log | `repo:…/raw/PRELANDED_verification.log` | new |
| arm panel log + 7 arm JSONs | `repo:…/raw/PANEL.log`, `raw/panel/<arm>/e2b_*.json` | new |
| full stack suite log | `repo:…/raw/FULL_SUITE.log` | new |
| register rows | `repo:Project Steering/GOALS_AND_CLAIMS.md` — `D-DDV2-PORT-1..4` appended to the `D-DDV2` block | edited |

---

## 8. ⛔ Integration requests — escalated here, not written into a README

1. **`§3.2`'s status column is stale and should be corrected in place.** The ≥GT bar and the
   two-scalar noise are **in `rl/`**, and the DDv2 analysis's integration request #1 (which asks for
   them) is **already satisfied**. Owner: whoever next edits
   `…/2026-09-05-diffusiondrive-v2-analysis/RESULT.md`. Left unedited by me: it is another package's
   banked deliverable, and `D-DDV2-PORT-1` records the correction.
2. **`refc_selector_aug` must be wired into whatever finally trains the selector on the real fan.**
   The three libraries and the driver are all CPU-runnable today; what is missing is the **banked
   refcv4 fan dump** and the **`obstacle.offline` replay join**, both pod-side. Owner: the
   Arch+Inference FlyWheel, when a GPU exists.
3. ⚠️ **`tanitad/effective_weights.py::explicit_dests` misreports a dest served by two argparse
   actions** (§3). It reads EXPLICIT on every launch and can therefore trip a spurious `DISCARDED`
   refusal. Every launcher using a `--flag` / `--no-flag` pair is exposed. **Not patched here** —
   shared module, degraded mount, and no launcher of mine depends on the fix. Owner: Tools/DevEnv.
4. **`MODEL_REGISTRY.md`:** when a selector row is created, stamp `SelectorConfig.as_dict()` **and**
   `AugmentConfig.as_dict()`, because the composition denominator and the foreign fraction change
   what the score *means*.

---

## 9. What is runnable the moment a GPU and a PI decision exist — and what is genuinely blocked

**Runnable now, zero GPU:** the whole stage-II path on the analytic rig
(`python scripts/refc_selector_e2b.py --arm {selector,const,raw_floor,progress_only}
[--n-aug 2] [--foreign-frac 0.01]`), the eight deliberate regressions, and the pieces-1–3 verification.

**Runnable the moment a banked refcv4 fan dump and the `obstacle.offline` replay join exist:** the
real `E-DDA-2b`. `--source dump` is the entry point and it currently **refuses with exit 3** rather
than approximating; the targets, heads, losses, augmentation, foreign bank, controls, tie-group AP
and the effective-weights stamp are all in place behind it.

**Genuinely blocked, each with what would unblock it:**

| blocked | blocker | what unblocks it |
|---|---|---|
| `E-DDA-2b` on the real fan | no banked refcv4 fan dump; no `obstacle.offline` replay join on this box; **the A40 pod is stopped and gone** | a GPU box + `D-REFCV4B-FANDUMP-1` (`B6`) |
| `E-DDA-3b` (the scale policy) | the RL pilot **cannot load its own cold start** — 487 keys vs 488 built, `decoder.anchor_controls` (**queue item 8, a PI decision**). ⛔ Not worked around with `strict=False`, which `refc.py:2091` already calls *"a plausible-looking WRONG experiment"* | the PI's ruling on queue item 8 |
| the `compliance` and `tactical` heads | `compliance_tau_rad` must be **derived per corpus** by `nav_compliance.derive_tolerance`, and the v7.2 `(lat, lon)` labels must be joined. Both heads are **omitted, not zero-filled** — `provenance["omitted"] = ["compliance", "tactical"]` | the label join; ⛔ the tau may not be invented |
| per-step chain REINFORCE, map/DAC rewards, the PDM reward, the 800-candidate fan | out of scope by the brief, and each blocked for the stated structural reason (no `log π(τ_{t−1}|τ_t)`; **no map in PhysicalAI**; no simulator; a deterministic generator) | ⛔ architecture change / a corpus that does not exist / a simulator |

⛔ **Nothing here is a driving claim.** The instrument, its controls and its refusals exist; the
measurement they were built for needs a fan this box does not have.
