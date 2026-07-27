# v5's trainer — ONE trainer that reads v2 AND runs a held-out val loop

**Date:** 2026-07-27 (dev box, Europe/Berlin) · **Stream:** Architecture & Inference
**Status:** staged, never committed, never pushed. ⛔ **No pod was contacted.** pod1 is training
`flagship-v2corpus-30k` and pod2 is building the 120° cache; every fixture below is synthetic.

**Predecessors:** `Benchmarks & Eval/…/2026-07-28-v2-parity-enforcement/V2_PARITY_ENFORCEMENT.md`
(§6 + escalation 0 — the trainer question) and `…/2026-07-28-wide-fov-build/WIDE_FOV_BUILD.md` §7.

---

## 0. Headline

| # | finding | class |
|---|---|---|
| **1** | ⭐ **v5 runs on `train_flagship_v4`, which now reads the v2 compressed cache.** The doc disagreement is resolved in favour of `flagship-v5-retrain.PREP.md`; `WIDE_FOV_BUILD.md` §7's `train_flagship4b --v2-cache` command is **not** the v5 launch. | **MEASURED** |
| **2** | ⭐ **Option B is not a "port", and that is a structural fact, not a preference.** `train_flagship4b`'s selected plan is **4 points at 0.5 s spacing** (`waypoint_horizons (5,10,15,20)`); the gate's pseudo-simulation differentiates at **dt = 0.1 s** and **raises `NonDensePlanError` on it — measured by running the real constructor on the real config**. Giving 4b a dense plan means giving it `FlagshipV4Head` and the loss that trains it. | **MEASURED** (`raw/option_size_2026-07-27.json`) |
| **3** | ⭐ **Size, normalised: 0 lines of NEW MACHINERY for option A against a ≥586-code-line FLOOR for option B** — and only **59** of A's code lines are the v2 wiring itself; **143** are strengthenings option B would also have had to write. | **MEASURED** (`raw/diff_size_2026-07-27.json`) |
| **4** | ✅ **pod1 is safe by construction: `stack/scripts/train_flagship4b.py` is BYTE-IDENTICAL** (sha256 pinned in a test). Its exact argv still parses and the resume branch still fires — proven by running the real `train()` twice. | **MEASURED** (`raw/pod1_resume_safety_2026-07-27.json`, `ALL_PASS: true`) |
| **5** | ⭐ **The gate provably reads the val loop's number.** Asserted by **object identity** through the real `train()` (the probed episodes ARE the `--v2-val-cache` providers), and end-to-end through the real `_training_loop`, which stops the run, writes `ckpt_best.pt` and logs the record. | **MEASURED** (`tests/test_v5_trainer_v2_val.py`) |
| **6** | ⭐ **The failing direction is REAL and it caught a real defect in my own test.** A degraded arm is stopped at **−0.0647 [−0.0844, −0.0517]**, paired episode-cluster bootstrap. ⚠️ My *first* degradation (slowing the planner) made the composite go **UP +0.1698** — pseudosim gives a barely-moving plan `recovery = NaN` by construction. A failing test that fails for the wrong reason is the same defect one layer down. | **MEASURED** (`raw/gate_both_directions_2026-07-27.json`) |
| **7** | ⭐ **Five guards removed one at a time; each one's test went RED, and the files restored bit-exact.** Including the one that did *not* go red on the first attempt and why. | **MEASURED** (`raw/wiring_redgreen_2026-07-27.json`, `ALL_PASS: true`) |
| **8** | ⭐ **Geometry IS now bound into what the trainer verifies — in two layers, of unequal strength, and it still does not hash pixels.** §7 says exactly what each layer proves and what neither does. | **MEASURED** |
| **9** | 🔴 **v5 IS STILL NOT LAUNCHABLE, and the blocker is data, not code: the 120° VAL split does not exist.** pod2 is building the 2 400-clip TRAIN split only. The trainer now **refuses to start** without a val cache and says why. | **MEASURED** (`WIDE_FOV_BUILD.md` §8, read directly) |
| **10** | ⚠️ **A default already moved, and it is not mine.** `apply_geometry_args` silently rewrites the **smoke** config's encoder 64→256 (1024 tokens instead of 256) while printing `DEPLOYED (unchanged)`. It is a genuine no-op for `flagship4b` and `flagship4b_reduced`, so **pod1 is unaffected** — measured, not assumed. Reported, not fixed. | **MEASURED** (`raw/pod1_resume_safety_2026-07-27.json` §4) |
| **11** | ✅ **No default moved by THIS change.** Every v4 command that exists today resolves to the raw path, requires both raw caches, and emits an identical staged command. Asserted in a test, not promised. | **MEASURED** |
| **12** | ⚠️ **A green suite was green for the wrong reason, and this stream found it in its own tests.** `tanitad/lake/filtering.py:96` reads the parity manifest at MODULE SCOPE, so a test that patches `parity.MANIFEST_PATH` before that first import corrupts the real skipset process-wide. It only passed because alphabetical collection put the test that catches it **first**. Fixed and pinned — §10.0. | **MEASURED** |

**Evidence-class note.** Everything here except §6's interval is a **code-behaviour** fact: deterministic,
`n = 1` run, no estimator — a path either executes or it does not, and tier language (DIRECTIONAL /
CONFIRMED) does not apply. §6's numbers are a **paired episode-cluster bootstrap (B = 400, unit = held-out
episode)** over **4 synthetic episodes / 40 windows**; `overlapping_holdout_se` appears nowhere.
Storage figures (697 GB / ~95 GB / ~24 GB) are **INHERITED** from `WIDE_FOV_BUILD.md` §3 and §8 and were
**not** re-measured; no decision here turns on them.

---

## 1. The contradiction, re-verified before anything was changed

| trainer | v2 / wide cache | held-out val loop |
|---|---|---|
| `train_flagship4b` | ✅ `--v2-cache` | ⛔ **NONE** — `scripts/train_flagship4b.py:421` reads `ds_val = None  # this trainer runs no val loop`, and `ds_val` is never read on either branch |
| `train_flagship_v4` | ⛔ **none** | ✅ the mid-run gate (`tanitad/train/heldout_gate.py`, 16 `ds_val`/`val_loss` sites) |

⇒ v5 had to give up **parity-capable storage** or **its early-stop**, and giving up the early-stop is
cause #1 of the previous run: **~29.5 GPU-h — half the run — spent training past the best checkpoint
while every training term improved.**

---

## 2. ⭐ THE MEASURED SIZE COMPARISON

`code/measure_options.py` → `raw/option_size_2026-07-27.json` (before either option was written)
`code/measure_diff.py` → `raw/diff_size_2026-07-27.json` (after option A existed, normalised)

### 2.1 The decisive fact: option B is not a port

`HeldoutGate` probes the **deployable surface** — `DeployableSurfacePlanner(world, head)` calls
`head(states, v0, …)` and reads `out["wp_seq"]`. Asked directly, in code:

| | `train_flagship_v4` | `train_flagship4b` |
|---|---|---|
| selected-plan horizons | **1…20** (dense, 0.1 s) | **(5, 10, 15, 20)** (0.5 s) |
| decoder output keys | `wp_seq`, `traj`, … | `anchor_logits`, `anchor_traj`, `offset`, `traj`, `sel_idx`, `waypoints` — **no `wp_seq`** |
| `DeployableSurfacePlanner` accepts it | ✅ **True** | ⛔ **False** |

The refusal, verbatim from the constructor (not from a docstring):

> *pseudo-simulation scores accelerations/jerk by finite differences at dt=0.1 s, so it needs a DENSE
> consecutive plan (horizons == 1..K). This head emits (5, 10, 15, 20).*

⇒ **A "val loop in 4b" is not moving a loop.** It is giving `train_flagship4b` a dense-plan head, the
loss that trains it, its labels and its optimizer group — i.e. making it `train_flagship_v4`. An
untrained head would be worse than nothing: a gate probing a random planner reports a number and
decides on noise.

### 2.2 Size, on one metric (non-blank, non-comment, non-docstring lines)

| | option A — **CHOSEN** | option B — rejected |
|---|---:|---:|
| **NEW machinery required** | **0** | **≥586** *(floor)* |
| core "read v2" wiring | **59** (`assert_corpus_args` 37 + `_assert_parity_v2` 22) | — |
| strengthenings (geometry binding, train/val leak guard, missing-manifest hint) — **needed on either path** | 143 | 143 |
| in-place branches (data block, geometry call, CLI, preflight, staged command) | 173 | — |
| **product total** | **375** | ≥586 **plus** all of column A's val-cache wiring |
| tests added | 519 | — |

Option B's 586 counts only definitions that would have to **move** (`FlagshipV4Head` 87 ·
`FlagshipV15Head` 173 · `v4_loss_step` 52 · `FlagshipV4Dataset` 32 · `_training_loop` 177 ·
`canary_rollout` 34 · `evaluate_planner` 31). It excludes the wiring, the tests, the second optimizer
group, **and the v2 val-cache branch it also needs** — which is option A's work done twice.

⚠️ **The line counts are the weaker half of this argument and are reported as such.** 375 vs 586 is not
a landslide. The argument is §2.1: option A composes machinery that already exists and is already
tested; option B puts **new, untested planner code on the critical path of the most expensive run in
the program**.

### 2.3 What option A reuses rather than writes

| capability | where it already lives | option A's cost |
|---|---|---|
| the v2 loader (lazy, LRU-bounded, `ToyEpisode`-shaped) | `tanitad/data/v2_dataset.build_v2_providers` | two call sites |
| the clip-id **membership** proof | `tanitad/data/parity.assert_v2_parity_cache` (landed 2026-07-27) | one dispatch |
| the input frame | `tanitad.geometry.apply_geometry_args` (already used by 4b) | one call + `add_geometry_args` |
| the mid-run held-out gate | `tanitad/train/heldout_gate.HeldoutGate` | **unchanged** — it takes episode OBJECTS with `.poses`/`.frames`, and `LazyV2Episode` supplies both, so the gate never learns that the cache format changed |

---

## 3. ⛔ pod1 — the run that must not be disturbed

pod1 (`tanitad-pod`) is ~18 k/30 k steps into `flagship-v2corpus-30k` on `train_flagship4b --v2-cache`.
`raw/pod1_resume_safety_2026-07-27.json`, **`ALL_PASS: true`**:

| # | check | result |
|---|---|---|
| 1 | `stack/scripts/train_flagship4b.py` sha256 | **`53f3ab5b…1b5d8ab8`, unchanged** — the chosen option edits `train_flagship_v4` only, so no behavioural argument is needed at all. Pinned by `test_train_flagship4b_is_untouched_by_this_change`. |
| 2 | pod1's **exact argv** (lifted from `TRAIN_CMD` in the staged supervisor env, not retyped) still parses | ✅ · `require_parity False` · every geometry flag `None` |
| 3 | the **resume branch** still fires | ✅ real `train()` run twice: run 1 ends at step 3, run 2 prints `[resume] resuming at step 4` and ends at step 6 — continued, not restarted |
| 3b | its unregistered `physicalai-v2bal` cache still **warns and proceeds** | ✅ `NON-PARITY v2 corpus`, then trains |
| 4 | `apply_geometry_args` is a **no-op for `flagship4b`** | ✅ `(256, None) → (256, None)` — see finding 10 for the config where it is *not* |

⚠️ **Scope of the claim.** This proves the repo's code is safe for pod1. pod1 runs its own on-disk copy
and is not updated by staging; nothing here was copied to any pod.

---

## 4. ⭐ The val loop, and the proof the gate reads it

### 4.1 What was wired

`train()` gains one branch. The raw path is untouched:

```python
if use_v2:
    from tanitad.data.v2_dataset import build_v2_providers   # inside the branch
    train_eps = build_v2_providers(a.v2_train_cache, lru_size=a.v2_lru)
    val_eps   = build_v2_providers(a.v2_val_cache,   lru_size=a.v2_lru)
    …geometry binding…
else:
    …the existing load_episode(ep_*.pt) path, unchanged…
```

Everything downstream is the code that already existed: `ds_val = FlagshipV4Dataset(val_eps, …)` feeds
`canary_rollout` and `evaluate_planner`, and `hg_eps = val_eps[:heldout_episodes]` feeds
`HeldoutGate.probe`.

### 4.2 ⛔ "A val loop whose result nothing reads is the defect you are fixing" — the proof

| # | what is asserted | how |
|---|---|---|
| 1 | the gate's episodes **ARE** the `--v2-val-cache` providers | **object identity** (`id()`) through the real `train()`, plus the negative: none of them is a train provider |
| 2 | `ds_val` (canary + planner eval) is built from the val providers, and the training DataLoader from the train providers | identity, same test |
| 3 | both caches are loaded, in order, at the right LRU | the stubbed loader records its calls |
| 4 | the gate is **ON by default** on the v2 path, with `patience ≥ 2` | asserted on the object the loop receives |
| 5 | ⭐ **the LOOP acts on the number** | the real `_training_loop` (via `smoke_loop`) with a gate driven to stop: `early_stopped True`, `final_step 4 < 6` (the budget), ≥3 history records each carrying `primary_value`, **`ckpt_best.pt` written**, and `heldout_gate` rows in `train_log.jsonl` with the last one `"stop": true` |

Row 5 is the one that matters: it is the loop **reading** the val number and changing what the run does.

### 4.3 ⭐ Both directions on the gate — with the failing one real

`raw/gate_both_directions_2026-07-27.json` — the REAL `pseudo_evaluate` → composite → paired
episode-cluster bootstrap. **Nothing stubs `observe`.**

| arm | probe 0 | probe 1 | probe 2 | verdict |
|---|---:|---:|---:|---|
| **STABLE** (4 probes) | 0.064679 | 0.064679 | 0.064679 | ✅ **not stopped**, `separated_worse` False throughout |
| **DEGRADED** — planner drifts 2 m/s sideways after probe 0 (~4 m off the logged path at 2 s) | 0.064679 | **0.0** | **0.0** | ⛔ **STOPPED** at probe 2 |

Paired delta at both degraded probes: **−0.0647 [−0.0844, −0.0517]**, `separated: true`,
`paired_episode_cluster_bootstrap`, n_episodes 4, n_windows 40. One separated probe does **not** stop
the run (`patience = 2`); the stop reason names `pseudosim_composite_PSS_recovery_progress` and the
29.5 GPU-h it exists to stop spending. `ade_0_2s` is carried as a diagnostic and never consulted.

⚠️ **THE DEFECT THIS PROBE FOUND IN ITSELF.** The first degradation I wrote was a **slowdown** (cut
forward motion to a tenth). Measured, the composite went **UP: +0.1698 [+0.1518, +0.1878]** — because
`pseudosim.score_windows` gives a barely-moving plan `recovery = NaN` by construction (the
progress-matched denominator, added precisely because *"standing still is not recovery"*). The test
would have been red for the wrong reason had the SIGN not been checked. It is recorded in the raw JSON
as case 3 rather than deleted.

⚠️ **Limit of this fixture, stated rather than buried:** on these synthetic episodes only **`recovery`**
clears `discriminative_range` and is admitted into the composite (`admitted_components: {"recovery":
5.0}`). So this proves the gate's **decision machinery** end to end; it does not exercise
`ego_progress` or `comfort`. On the real corpus the admitted set is pinned at the first probe and
travels in every emitted node.

---

## 5. Parity — the just-landed guard is CALLED, not reimplemented

`_assert_parity_v2` is a **dispatcher**. It calls
`parity.assert_v2_parity_cache(…, require=a.require_parity)` on the train cache and
`require=False` on the val cache — mirroring `_assert_parity`'s decisions exactly, so the raw and v2
paths cannot drift. Its return shape matches `_assert_parity`'s, so `config.json`'s `"parity"` block
stays one schema.

**One fact was added, because `assert_v2_parity_cache` structurally cannot see it:**
`parity.assert_v2_splits_disjoint(train_dirs, val_dirs)` — it checks each directory against the
manifest and never compares two of them. On the raw path the splits are separate *registered* corpora
so the digests catch an overlap; on the v2 path they are two paths a launch command supplies.

> ⚠️ A leaked val clip does not crash anything. It makes the early-stop probe a **training** episode,
> so the gate reports health while the deployable surface decays. **An early-stop that cannot fire is
> worse than none, because it is believed.** Refused, with counts only — 🔒 no clip ids.

**`--require-parity` stays opt-in** (nothing existing moves), so the place a v5-class omission becomes
visible is **preflight**, which in `train_flagship_v4` is a hard block before `train()`:

```
[PARITY] --v2-train-cache without --require-parity: an unregistered or mismatched v2 cache
prints ONE NON-PARITY line and TRAINS ANYWAY. Every cross-arm number off such a run is void,
invisibly.
```

⛔ **What `--require-parity` does NOT prove here:** it proves the **TRAIN** split's membership. The val
cache is checked at `require=False`, mirroring the raw path — so an **unregistered val sibling warns and
proceeds**. Registering the val sibling is a runbook step (§8), not something the flag enforces.

### 5.1 ⭐ Runbook step 3, made legible

Step 3 — *commit the manifest* — is the one that gets forgotten. The registration runs on a pod; if the
diff is never staged, the cache reads NON-PARITY on every other host and `--require-parity` refuses to
start. The refusal was accurate and useless. It now appends:

```
  🔴 MISSING MANIFEST ENTRY: 'physicalai-train-e438721ae894-w120-256x640cyl'
     (it EXTENDS the registered key 'physicalai-train-e438721ae894', which is a RAW epcache corpus)
     is NOT in <repo>/stack/tanitad/data/parity_manifest.json
     — so this host cannot verify the cache even though the pod that
       built it may have registered it perfectly.

     This is RUNBOOK STEP 3, the one that gets forgotten:
         git add stack/tanitad/data/parity_manifest.json
```

It fires on **both** v5 failure shapes — the unregistered dir *and* the sibling dir wearing its
parent's key (which resolves to the parent and refuses on uid-kind) — and is **silent** when the entry
exists, because noise in a refusal is how the real line gets skipped. All three pinned by tests.

---

## 6. Geometry — accepted end to end

The four flags are the same ones `train_flagship4b` and the cache builder take
(`tanitad.geometry.add_geometry_args`), so a wide run is spelled identically everywhere. Applied to the
**real** `flagship4b_config`:

```
[geometry] train_flagship_v4: NON-DEFAULT - 256x640px, f_ref 305.58, cylindrical,
           HFOV 120.00deg / VFOV 45.46deg, tokens 640 (16x40 @ patch 16),
           state_dim 2048, cache fragment {'geom': '256x640f305.5775cyl'}
```

| quantity | MEASURED |
|---|---|
| tokens | **640** (16 × 40 @ patch 16) |
| `state_dim` | **2048** (unchanged) |
| encoder params | **87,022,848 → 87,317,760** (+**294,912** = exactly `(640−256) × 768`, the positional embedding) |

The run's `config.json` now carries `"geometry"` and `"corpus_format"`, so a wide arm can never be read
later as an ordinary row of the parent corpus. ⚠️ `apply_geometry_args` still warns that 120° exceeds
comma2k19's entire field (65.203°) — **a PI decision, not a default**, unchanged by this work.

---

## 7. ⭐ Geometry bound into the parity verification — and what it still cannot prove

The parity stream's standing gap: *nothing hashes PIXELS; a wrong-FOV cache with the right clips
PASSES.* `parity.assert_v2_geometry_matches(rec, frame, providers=…)` now closes part of it, in **two
layers of unequal strength**:

| layer | what it compares | strength |
|---|---|---|
| **1. SHAPE** | every provider's actual `frames.shape[-2:]` (from each payload's `image_h`/`image_w`, written at BUILD time) vs the run's declared frame | ⭐ **strong** — a property of the bytes on disk, not of a declaration. Catches the exact `WIDE_FOV_BUILD.md` §7 failure ("omit the flags and the trainer builds a 256×256 encoder and is fed 256×640 frames"), and catches **mixed geometries in one cache**. |
| **2. DECLARATION** | the registered entry's `provenance.geometry` (the builder's `_geometry.json`) vs the run's frame — `f_ref` and `projection` | weaker but not redundant: 256×640 at f_ref 305.58 (120°) and at f_ref 407 (90°) have **identical shape**. Only this layer separates them. |

⛔ **What neither proves, said plainly:** **nothing here hashes pixels.** A cache whose `_geometry.json`
records 120° but whose resampler actually produced 90° passes both layers. Only the builder's
pre-decode `_assert_geometry_deliverable` binds the *record* to the *resampler*; it runs hours earlier,
in a different stream, and if it is wrong nothing here catches it. That sentence lives in the function's
docstring and in the returned record (`pixels_are_not_hashed`), which is written into the run's
`config.json` — so the limit travels with every artifact rather than living in prose.

Both refusals are tested, and the refusal tells the operator what to pass:

```
GEOMETRY VIOLATION [--v2-train-cache] — the cache is not the frame the run declares
  run declares : 256x256 px, f_ref 266.0000, pinhole
  cache holds  : 256x640 px   <-- MISMATCH
  Pass the cache's own geometry, e.g.:
      --frame-h 256 --frame-w 640 --frame-hfov <deg> --projection <pinhole|cylindrical>
```

---

## 8. ⭐ The exact command a v5 launch runs

**Runbook: build BOTH splits → verify → register → COMMIT THE MANIFEST → train.**
Steps 2–3 are `V2_PARITY_ENFORCEMENT.md` §7 verbatim, run **twice** (train and val). Step 4 changes.

```bash
# ---- 1. THE VAL SPLIT MUST EXIST FIRST.  🔴 IT DOES NOT YET. -----------------
#      pod2 is building the 2,400-clip TRAIN split only. WIDE_FOV_BUILD.md §8:
#      "the val split, if wanted (600 clips, ~24 GB, same command, one flag changed)".
#      ⚠️ It must be a COMPLETE 600-clip build: no skip_indices are committed for
#      physicalai-val-0c5f7dac3b11, so a shortfall cannot be verified as decode
#      failures and register_v2_sibling.py will refuse it.

# ---- 2/3. VERIFY + REGISTER + STAGE, once per split (V2_PARITY_ENFORCEMENT §7)
#      train: --new-key physicalai-train-e438721ae894-w120-256x640cyl
#      val  : --new-key physicalai-val-0c5f7dac3b11-w120-256x640cyl
#      then, on the dev box:  git add stack/tanitad/data/parity_manifest.json

# ---- 4. TRAIN -----------------------------------------------------------------
cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
python3 -u scripts/train_flagship_v4.py \
  --v2-train-cache /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --v2-val-cache   /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --v2-lru 64 \
  --require-parity \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --from-scratch \
  --anchors-dense /workspace/experiments/anchors/anchors_dense_1to20.pt \
  --out   /workspace/experiments/flagship-v5-w120-30k \
  --steps 30000 --batch 16 --accum 4 --lr-head 1e-4 --lr-trunk 1e-4 \
  --warmup 2000 --workers 8 --eval-every 500 --save-every 1000 --rollout-k 4 \
  --heldout-gate --heldout-every 2000 --heldout-episodes 8 --heldout-patience 2 \
  --device cuda
```

- ⛔ **`--require-parity` is not optional for a v5-class run.** Preflight blocks the launch without it.
- ⛔ **`--v2-val-cache` is not optional.** Without it `train()` refuses and names the gate and the
  29.5 GPU-h; preflight blocks it too.
- ⭐ **`--anchors-dense` matters here.** Without it the trainer prints *"the operative fan uses the
  head's DEFAULT anchor buffer (fine for a smoke, NOT for a gate run)"* and does not stop.
- Run `--print-launch` first: it prints this exact string (reconstructed from the parsed args, with the
  v2 dirs, the geometry flags and `--require-parity` all preserved) plus the preflight verdict, and
  exits without training.
- ⛔ **A launch is the PI's go, executed by the orchestrator. Nothing here launches anything.**

⭐ **The command above was RUN through `--print-launch`, not written from memory** —
`raw/print_launch_v5_2026-07-27.txt`, exit 0, ending:

```
staged command (run on the pod, NOT here):
  PYTHONPATH=/workspace/TanitAD/stack python3 scripts/train_flagship_v4.py
    --v2-train-cache /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl
    --v2-val-cache   /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl
    --v2-lru 64 --frame-h 256 --frame-w 640 --frame-hfov 120.0 --projection cylindrical
    … --from-scratch --heldout-gate --require-parity
PREFLIGHT: OK
```

---

## 9. ⛔ What is still open

| # | item |
|---|---|
| 1 | 🔴 **The 120° VAL split does not exist.** ~600 clips, ~24 GB, ~1 h on 8 shards. **v5 cannot start without it** — by design, now. |
| 2 | ⚠️ **`--require-parity` proves the TRAIN split's membership only.** The val cache is checked at `require=False` (mirroring the raw path), so an unregistered val sibling warns and proceeds. Registering it is runbook discipline, not enforcement. |
| 3 | ⛔ **Nothing hashes pixels** (§7). |
| 4 | ⚠️ **The v2 path has no subset mode.** `--episodes N`-style truncation is not honoured on the v2 branch; a v2 sibling is all-or-nothing. Inherited from `parity.py` §9 §4. |
| 5 | ⚠️ **The gate's `heldout_episodes` are a FIXED PREFIX of the val providers** (`val_eps[:8]`), and `build_v2_providers` orders clips by filename. That is deterministic (which is what `WindowAlignmentError` needs) but it is **not a random sample of the val split**, so the gate's 8 episodes are a fixed, arbitrary corner of it. Unchanged from the raw path; worth a decision before the run. |
| 6 | ⚠️ **`apply_geometry_args` moves the SMOKE config's encoder 64 → 256** while printing `DEPLOYED (unchanged)` (finding 10). Not mine, not fixed, no effect on pod1 or on v5. |
| 7 | ⚠️ **`MODEL_REGISTRY` needs the CORPUS KEY per row**, not just the episode set: two rows can share `e438721ae894`'s episodes and differ in pixels. The run's `config.json` now carries `corpus_format` + `geometry`; **nothing reads them yet.** (Inherited escalation, still open.) |

---

## 10. Tests — before and after

| suite | before | after | delta |
|---|---|---|---|
| `stack/` | **1298 passed, 12 skipped** | ✅ **1324 passed, 12 skipped** | **+26**, all in `tests/test_v5_trainer_v2_val.py` |
| `taniteval/` | **559 passed** | ✅ **559 passed** | 0 |

*(dev box, `C:\Users\Admin\venvs\tanitad`.)*
⚠️ **Zero new skips.** `tanitad.data.v2_dataset` imports torchvision, which this host does not have;
every test **stubs it into `sys.modules`** rather than `importorskip`, because a guard test that skips
on the host where it is most likely to be run is a guard that cannot fail.

### 10.0 ⚠️ A cross-test defect this file found in itself

The full suite was green at every stage, and it was green **for the wrong reason**.
`tanitad/lake/filtering.py:96` evaluates `PARITY_SKIP_INDICES` from `parity_manifest.json` **at module
scope**. My tests monkeypatch `parity.MANIFEST_PATH` to a synthetic manifest, and `train()` imports
`flagship_v4_data` → `tanitad.lake.vtarget` → `tanitad.lake` — so the first such import inside a patched
test replaced the real 24-index skipset with the synthetic one **for the whole process**.

**MEASURED:** run `tests/test_v5_trainer_v2_val.py tests/test_parity_manifest.py` in that order and
`test_lake_filtering_skipset_is_index_reproducible_from_the_repo` **FAILS**; alone, or in the full
alphabetical run, it passes. **The suite's green was an artifact of collection order.**

Fixed by importing the manifest readers at this file's module scope, and pinned by
`test_the_real_parity_manifest_survived_this_file` — deliberately the **last** test in the file, so it
runs after every monkeypatch and the alphabetical accident stops being load-bearing.

### 10.1 RED/GREEN — each guard removed in turn

`code/wiring_redgreen.py` → `raw/wiring_redgreen_2026-07-27.json`, **`ALL_PASS: true`**,
`files_restored_bit_exact: true`.

| guard removed | test that went RED |
|---|---|
| the gate probes the TRAIN half instead of the held-out one | `test_the_gate_probes_the_VAL_providers_not_the_train_ones` |
| the v2 parity guard is skipped | `test_the_parity_guard_runs_BEFORE_the_v2_loader` |
| the geometry binding is removed (**both halves**) | `test_a_wide_cache_read_with_DEFAULT_flags_is_refused` |
| the train/val leak check is removed | `test_a_train_val_LEAK_is_refused` |
| the missing-manifest hint is removed | `test_an_unregistered_v2_cache_NAMES_the_missing_manifest_entry` |

⚠️ **The third row did not go red on the first attempt**, and the reason is worth keeping: removing only
the *train* binding left the *val* binding refusing the same run. Defence in depth is good; a probe that
proves nothing is not. The mutation now removes both, and the near-miss is recorded in the probe.

---

## 11. Deliverable manifest

**Everything `git add`ed into the working tree. Nothing committed. Nothing pushed. No pod contacted.**

| artifact | where it lives | only one copy? |
|---|---|---|
| `V5_TRAINER.md` (this file) | `repo:…/incoming/2026-07-28-v5-trainer/` | no |
| ⭐ `stack/scripts/train_flagship_v4.py` — the v2 branch, `assert_corpus_args`, `_assert_parity_v2`, geometry flags, preflight, staged command | `repo:` **staged** | no |
| ⭐ `stack/tanitad/data/parity.py` — §9 `assert_v2_splits_disjoint`, `assert_v2_geometry_matches`, `_missing_entry_lines`, `_sibling_candidate_key` | `repo:` **staged** | no |
| ⭐ `stack/tests/test_v5_trainer_v2_val.py` — **new**, 26 tests | `repo:` **staged** | no |
| `code/measure_options.py` + `raw/option_size_2026-07-27.json` | `repo:` | no (regenerable) |
| `code/measure_diff.py` + `raw/diff_size_2026-07-27.json` | `repo:` | no (regenerable) |
| `code/pod1_resume_safety.py` + `raw/pod1_resume_safety_2026-07-27.json` | `repo:` | no (regenerable) |
| `code/wiring_redgreen.py` + `raw/wiring_redgreen_2026-07-27.json` | `repo:` | no (regenerable) |
| `code/gate_both_directions.py` + `raw/gate_both_directions_2026-07-27.json` | `repo:` | no (regenerable) |
| `raw/print_launch_v5_2026-07-27.txt` — §8's command, executed through `--print-launch`, `PREFLIGHT: OK` | `repo:` | no (regenerable) |
| ⛔ **`stack/scripts/train_flagship4b.py`** | **NOT TOUCHED** — sha256 `53f3ab5b…` pinned in a test | — |

⚠️ `git status --short` may show other streams' work. Per `CLAUDE.md`: read it FIRST, prefer a
pathspec-free `git commit -F <msgfile>` after confirming every index entry is intended program work
(`git commit -- <pathspec>` segfaults on this repo).

---

## 12. Escalations

0. ⭐ **THE DOC DISAGREEMENT IS RESOLVED: v5 runs on `train_flagship_v4`.**
   `Project Steering/Gates/flagship-v5-retrain.PREP.md` §0 cause #1 and §3 are correct;
   **`WIDE_FOV_BUILD.md` §7's `train_flagship4b --v2-cache` command is NOT the v5 launch** and should be
   annotated to point here. `V2_PARITY_ENFORCEMENT.md` §6 refused to add v2 to v4 *"because v5's trainer
   is UNDECIDED"* — that refusal was right at the time and its premise has now been decided by
   measurement.
1. 🔴 **BUILD THE 120° VAL SPLIT.** It is the only thing between here and a launchable v5, it is ~1 h of
   pod time, and the trainer now refuses to start without it.
2. 🔴 **Runbook steps 2–3 are owed TWICE** (train and val), and **step 3 — `git add` the manifest — is
   the one that gets forgotten.** The refusal now names the missing entry; that is a safety net, not a
   substitute.
3. ⚠️ **`apply_geometry_args` silently rewrites the smoke config's encoder 64 → 256** while printing
   `DEPLOYED (unchanged)`. Harmless for `flagship4b`/`flagship4b_reduced` (measured), so pod1 and v5 are
   unaffected — but it is a moved default inside the guard that exists to stop moved defaults, and it
   belongs to the geometry stream.
4. ⚠️ **The gate's 8 held-out episodes are `val_eps[:8]`, a fixed filename-ordered prefix.** Deterministic
   by necessity; arbitrary by accident. A decision before the run, not after.
5. ⚠️ **`tanitad/lake/filtering.py` reads `parity_manifest.json` at MODULE SCOPE** (§10.0). Any current or
   future test that patches `parity.MANIFEST_PATH` before that first import silently replaces the real
   24-index skipset for the whole process, and the test that catches it only runs first by alphabetical
   accident. Fixed and pinned **in this file's tests**; the underlying import-time read is unchanged and
   belongs to the lake stream.
