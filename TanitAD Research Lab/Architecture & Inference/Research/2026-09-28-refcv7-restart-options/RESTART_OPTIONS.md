# refcv7-r101-s0: restart options, priced and ready (2026-09-28)

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-28-refcv7-restart-options/`
**For:** the PI (the decision) and the Master Mind (the single committer).
**Author:** Architecture & Inference FlyWheel. ⛔ **Nothing was launched, restarted, stopped, committed or staged.**
Thor was touched READ-ONLY (`ls`, `cat`, `md5sum`, `scp` pulls; every pull md5-verified against Thor).

The live run: `refcv7-r101-s0` on Thor, launched 2026-09-28 00:03 Berlin on `fec3a0d` by the launch
gate's PASS token (MODEL_REGISTRY row "refcv7-r101-s0"). This package makes the NEXT restart cheap and
fully decided. **A restart is the PI's decision.**

---

## 0. Headline

1. **Recommended default: R4, "the bundle", at the next PLANNED stop, only if the PI accepts the
   conflict-probe cadence change (10 → 50).** The bundle is: conflict cadence 50 + lever 2 (per-class
   map signal and box census on logged steps only) + the owed freeze of the 10 admitted modules +
   the `__getstate__` fix + the I3 loader fix. It ships as ONE commit and pays ONE re-binding.
   * Saving **0.76–1.40 s/step** (ESTIMATED; derivation in §3).
   * Stopped training **≈ 2.8 h** (MEASURED from the launch night's own logs, §2).
   * **Net +6.8 to +14.9 h if restarted at step 5,000; +5.8 to +12.9 h at step 10,000.** Over all
     of the remaining 48,600 steps it would be +7.5 to +16.1 h, an upper bound: no restart can
     happen at step 1,800. It breaks even if the restart happens before step ~37,300 (worst case) /
     ~43,300 (best case).
   * Training numerics: **bit-identical** for every parameter the run trains. MEASURED end to end on
     the real trainer (CPU): the bundle tree vs the launch code, 590 parameters × 12 steps, 0 bytes
     differ, and likewise for each component (§4). The only thing that changes is when two
     instruments read.
2. **If the PI declines the cadence change: NO restart** (R0). Without it the bundle saves only lever 2's
   0.2–0.7 s/step, which nets −0.3 to +6.1 h at step 5,000. At the low end it never pays. The freeze,
   the getstate fix and the I3 fix are then carried to the next restart that happens for another
   reason.
3. **Pure crash-restart is free.** The supervisor re-verifies the SAME token and resumes from
   `ckpt.pt`. It costs ~6 min plus the steps since the last save (≤ 499 steps ≈ 83 min). **Nothing
   in this package may ride on it**: any code or argv change needs a new gate run.
4. **Closure-touching (PACKAGE-ONLY, no repo target):** the freeze (`refc_v3.py`,
   `declared_vs_built.py`), lever 2 (`refc_v3_train.py`), the getstate fix (`v2_dataset.py`), and the
   I3 fix (`refcv6_loader.py`, `g_box_overfit.py`). Every one of these files is in the binding
   closure(s) of the live run's overfit records.
   * **The argv levers are not closure-touching, but they void the same records.** Both records
     bind the LAUNCH argv sha256 `6402d33d…`.
   * ⇒ **Every option except R0 costs the same ~2 h of exclusive-GPU re-binding.** That is why the
     options are bundled.

---

## 1. The decision table

Remaining work at step 1,800: 48,600 steps × 9.938 s = **134.2 h** (MEASURED pace, §2.1).
`net` = saving × remaining steps − stopped time. `S` is the step at which training restarts.

| option | what | saving s/step | numerics | closure-touching | checks that must RUN again | stopped training | net h if S = 1,800 (upper bound) | S = 5,000 | S = 10,000 | S = 20,000 | break-even S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **R0 no restart** | — | 0 | unchanged | — | none | 0 | 0 | 0 | 0 | 0 | — |
| R1 freeze (+ getstate + I3) alone | the owed items only | **0** | bit-identical (§4) | yes (both closures) | all 10; **both binding RUNS** | 2.8 h | −2.8 | −2.8 | −2.8 | −2.8 | never |
| R2 conflict 50 alone | argv `--conflict-every 50` | 0.56–0.70 | bit-identical (MEASURED, §4 K12) | no, but **argv** | all 10; **both binding RUNS** (records bind the argv sha) | 2.7 h | +4.8…+6.7 | +4.3…+6.1 | +3.5…+5.1 | +2.0…+3.2 | 32,800–36,300 |
| R2b conflict 50 **+ a gate amendment** (the overfit records bind the argv's MODEL view) | argv + `launch_gate.py` | 0.56–0.70 | bit-identical | no | all 10, records only re-judged | 0.8 h | +6.8…+8.7 | +6.3…+8.1 | +5.5…+7.1 | +4.0…+5.1 | 45,500–46,500 |
| R3 lever 2 alone | code | 0.20–0.70 | bit-identical (MEASURED, §4 L12) | yes (both) | all 10; both RUNS | 2.7 h | −0.0…+6.7 | −0.2…+6.1 | −0.5…+5.1 | −1.1…+3.2 | ≤ 36,300 (never at the low end) |
| **R4 THE BUNDLE (recommended)** | conflict 50 + lever 2 + freeze + getstate + I3 | **0.76–1.40** | **bit-identical** (MEASURED end to end, §4 B12) | yes (both) | all 10; both RUNS | **2.8 h** | **+7.5…+16.1** | **+6.8…+14.9** | **+5.8…+12.9** | **+3.6…+9.0** | **37,300–43,300** |
| R4c the bundle without the cadence change | lever 2 + freeze + getstate + I3 | 0.20–0.70 | bit-identical | yes (both) | all 10; both RUNS | 2.8 h | −0.1…+6.7 | −0.3…+6.1 | −0.5…+5.1 | −1.1…+3.1 | ≤ 36,100 |
| R5 = R4 + map grad-ckpt off | + argv `--map-hires-grad-ckpt off` | 0.76–1.70 | bit-identical on CPU (§4 O12); **not shown on CUDA** | yes | all 10; both RUNS | 2.8 h | +7.5…+20.2 | +6.8…+18.7 | +5.8…+16.3 | +3.6…+11.6 | 37,300–44,500 |
| R6 = R4 + eval every 1,000 | + argv `--eval-every 1000` | 0.98–1.63 | bit-identical (the eval is a monitor) | yes | all 10; both RUNS | 2.8 h | +10.5…+19.2 | +9.6…+17.8 | +8.2…+15.5 | +5.5…+11.0 | 40,200–44,300 |

**Why R4 and not R2b.**
* R2b needs a SPEC_REFCV7 A11 amendment first: *"the overfit records bind the launch argv"* would
  become *"…bind the launch argv minus instrument-only flags"*. That is the PI's call, and the gate
  change would need its own tests. It is **not** built here.
* R2b also leaves the owed freeze owed.
* R4 beats R2b whenever lever 2 saves ≥ 0.16 s/step: the extra 2.0 h of re-binding amortised over
  ~45,000 steps.

**Why not R5 / R6 by default.**
* **R5:**
  * grad-ckpt off costs ~+21 GB at b16, on top of the MEASURED live peak of 24.84 GB, against 128 GB
    unified. That fits.
  * Its saving was **not separated** from zero on the dev box.
  * Its bit-identity on CUDA is not shown. The recompute re-runs the same forward, but a
    non-deterministic CUDA kernel would not replay bit for bit. The CPU arm in §4 is informative
    only.
* **R6:** halves the Training Watch's in-training eval cadence. That is an instrument decision like
  the conflict one. Offer it; do not default it.

⚠️ **The PI decisions this table needs, and nothing else:**
1. **Accept the conflict-probe cadence change 10 → 50.** It gives ~900 readings over the rest of the
   run instead of ~4,500.
   * `PREREG_REFCV6_V2.md` §8: *"a sparse median is a different statistic and is reported as
     one"*. So the run's conflict record is reported as two segments.
   * PREREG_REFCV7 and SPEC_REFCV7 carry **no** criterion on the conflict readings (grep: the only
     hit is §25's cost-probe setup line).
2. **Restart at a planned stop**, or not.
3. Optionally, R5 / R6 / the R2b amendment.

---

## 2. What a restart costs (MEASURED from the launch night's own logs)

### 2.1 The live pace (MEASURED, Thor `metrics.jsonl` pulled read-only at step 1,800; `raw/live/pace_step1800.json`)

| quantity | value |
|---|---|
| steady 50-step block (no eval), median of 32 blocks | **474.65 s** ⇒ 9.493 s/step (the conflict probe included) |
| the block holding the every-500-step eval + checkpoint | 691.3–708.1 s ⇒ **+216.6…+233.5 s per 500 steps** (0.43–0.47 s/step) |
| long-run pace (9 plain blocks + 1 eval block per 500 steps) | **9.938 s/step** (steps 500→1,500 read directly: 9.966) |
| `cuda_max_mem_gb` | 24.836 |
| checkpoint cadence | every 500 steps = **82.7 min**. The save is written BEFORE that step's eval (`refc_v3_train.py`, save-before-eval) |

### 2.2 The four cost components

| component | MEASURED duration | source |
|---|---|---|
| **map binding RUN** (`map_hires_overfit.py`, A19 MAIN-only, 3,000 steps) | **34.1 min** (16:05:13Z → 16:39:21Z; the closure's `elapsed_s` 2,047.3) | `raw/thor_bind/chain_bind.log`, `gmo_closure.REDACTED.json` |
| **box binding RUN** (`g_box_overfit.py`, A19 MAIN-only, 2,000 steps) | **84.5 min** (16:39:51Z → 18:04:18Z; `elapsed_s` 5,065.7) | same, `gbo_closure.REDACTED.json` |
| **Thor gate stage** (G-LIVE 100-step smoke + G-CKPT resume ⇒ 38.7 min; G-HYG/G-DVB/G-EVAL/G-CLOCK/records ⇒ +0.9 min; G-SUITE consume + finalize ⇒ ~0.3 min) | **~40 min** | `raw/thor_gate/job_*.log`, `gate_thor.log`, `gate_finish.log` |
| **relaunch → first training step** (exec 22:03:37Z; config.json written +5.6 min; compile warm-up ~0.5 min) | **~6 min** | Thor run dir listing + `metrics.jsonl` (step 50 at `elapsed_s` 505.6 vs a 474.65 s steady block) |
| steps lost since the last save | **0** at a planned stop right after the `[v3:hier] ckpt step N` line; ≤ 499 steps (82.6 min) if unplanned | `train.log` |
| freeze checkpoint conversion (R1/R4 only) | ~1–2 min CPU (ESTIMATED; the dev box converted the live 1.17 GB checkpoint in well under a minute) | `raw/freeze/live_ckpt1500_convert_record.json` |
| dev-box side, **not stopped time**: G-SUITE-PINNED | 11.4 min (21:22:56Z → 21:34:19Z) | `raw/thor_gate/G-SUITE-PINNED.json` |
| dev-box side, not stopped time: the full-suite lander verdict (G-SUITE (a)) | UNVERIFIED here (it is the Master Mind's landing run) | — |

**Restart classes, stopped-training hours:**

| class | when | stopped |
|---|---|---|
| A. resume under the SAME token | crash, reboot, no code or argv change | ~0.1 h + lost steps |
| B. new token, the records still bind | a change OUTSIDE both closures with the argv unchanged. Example: a new test or tool under `stack/`, or `launch_gate.py` itself. Research Lab files are not in the tree at all | **~0.8 h** (Thor gate stage + relaunch) |
| C. new token + both binding RUNS | ANY closure-touching change, or ANY argv change | **~2.7 h** (+ 2 min for the freeze conversion) |

⚠️ **Risk not in the numbers.** The launch night needed **six** launch commits
(60f4c06 → 6fa5e8b → 53ecf9b → 02ffceb → 14907a0 → fec3a0d). The gate defects were only reachable
on Thor with the real model. A restart re-uses a gate that has now passed once on this model, but
each Thor-only failure costs at least one more G-LIVE smoke (~40 min of stopped time).

### 2.3 The closure map (item 1)

`code/closure_map.py` → `raw/closure_map.json`. It runs over the two BINDING closures pulled from Thor
(`/home/nvidia/refcv7_bind/run_map/gmo_closure.json` md5 `e740fc67…`,
`/home/nvidia/refcv7_bind/run_box/gbo_closure.json` md5 `be07edab…`). Clip ids are redacted in the
banked copies; the module lists are untouched.

* **Map closure:** 101 modules, 17 data files. **Box closure:** 108 modules, 46 data files.
  * Every recorded blob equals the launch commit's: 101/101 and 108/108.
  * Only in box: `g_box_overfit.py`, `train_p8_occupancy.py`, `clip_clock.py`, `refcv6_loader.py`,
    `flagship_v15.py`, `slot_query_select.py`, `v6.py`, `anchor_meta.py`.
  * Only in map: `map_hires_overfit.py`.
  * The whole `stack/tanitad` model/refs/train/data core and **`refc_v3_train.py` are in BOTH.**
* **The argv binding:**
  * Both records carry the launch argv sha256 **`6402d33de75b7f1c…`**. The map harness has it as
    `--launch-argv-sha256`; the box harness hashes the argv file it builds from.
  * The box closure's data list also hashes that argv FILE.
  * `judge_map_overfit` / `judge_box_overfit` refuse a record bound to any other argv.
* **The env binding:** torch 2.13.0+cu130, timm 1.0.29, CUDA 13.0, cuDNN 92000, Python 3.12.3. The
  bindings can only be re-run on Thor.
* **The token binding** (`verify_token`): commit, tree sha256 (stack + taniteval), argv sha256, and
  every data input's fingerprint.

| candidate | files | in map closure | in box closure | new token | map binding RUN | box binding RUN |
|---|---|---|---|---|---|---|
| freeze | `refc_v3.py`, `declared_vs_built.py` | ✔ | ✔ | ✔ | ✔ | ✔ |
| lever 2 | `refc_v3_train.py` | ✔ | ✔ | ✔ | ✔ | ✔ |
| getstate | `v2_dataset.py` | ✔ | ✔ | ✔ | ✔ | ✔ |
| I3 patch | `refcv6_loader.py`, `g_box_overfit.py` (+ its test) | — | ✔ | ✔ | — | ✔ |
| conflict 50 / ckpt off / eval 1,000 | argv only (+ the `_FINAL_ARGV_SHA256` literal in `test_launch_gate.py`) | — | — | ✔ | ✔ (argv sha) | ✔ (argv sha + argv file) |
| a new test or tool file under `stack/` | new file | — | — | ✔ | — | — |
| Research Lab files (this package) | outside stack/ taniteval/ tools/ | — | — | **no** | — | — |

SPEC_REFCV7 §25 already states this rule in one sentence: *"A speed-up found later is applied at a
checkpoint with a NEW gate run and new binding runs on the new argv."*

---

## 3. Where the savings come from

* **Conflict cadence** (ESTIMATED from Thor measurements):
  * The launch gate's own G-LIVE smoke on Thor reads a median of 9.074 and a mean of 9.952 s/step
    over 89 intervals, which include 9 conflict steps. ⇒ One probe step costs **+95.7 %** of a
    median step. The dev box measured +91–101 %.
  * Against the live 474.65 s steady block (5 probe steps per 50), the plain step is **8.66 s** and
    one probe costs **8.29 s**. That is **0.83 s/step at every 10**.
  * Every 50 saves **0.66 s/step**; every 100 saves 0.75; off saves 0.83.
  * With the probe cost at +80…+101 %, the every-50 saving is **0.56–0.70 s/step**.
  * ⚠️ One assumption: the smoke's mean−median gap is all probe steps. The smoke also carries the
    gate's per-parameter gradient capture.
* **Lever 2** (INHERITED from the step-cost package, ESTIMATED for Thor): dev-box in-process A/B
  **−0.063 s @ b2** (90 % CI [−0.069, −0.016]); per-sample 0.015–0.044 s × 16 ⇒ **0.2–0.7 s/step**.
  Not measured on Thor.
* **Grad-ckpt off:** dev box −0.019 s @ b1, CI [−0.030, +0.009], **not separated**. ESTIMATED 0–0.3 s.
* **Eval every 1,000** (MEASURED block excess): half of 216.6–233.5 s per 500 steps ⇒ **≤ 0.22–0.23
  s/step**. The checkpoint save inside that excess is not separated.
* **Freeze / getstate / I3:** 0. The frozen tensors never had a gradient, so AdamW and the clip
  already skipped them. The getstate path is never taken under fork. I3 is not on the training path.
* **Measuring the real saving costs nothing afterwards:** the restarted run's own 50-step blocks
  against the 474.65 s baseline. `…/2026-09-27-refcv7-step-cost/code/thor_confirm.sh` (≤ 20 min, GPU)
  would measure it before, but only inside the stopped window. It is not needed for R4, whose parts
  are all bit-identical.

---

## 4. Numerics (MEASURED)

**The rig.** `code/freeze/numerics_ab.py` runs the **real trainer** (`refc_v3_train.main`) in-process.
* **Argv:** the **canonical 154-token launch argv**, path-mapped exactly as the step-cost package maps
  it (the eval139 cache as the train set; `rc7_argv.py`). This is a numerics rig, **never a model
  result**.
* **Dev-box departures**, stated in every `run_spec.json`: CPU, b1, workers 0, no eval,
  `resnet34.a1_in1k` instead of r101, no bf16, `--trunk-chunk-ckpt 1`, no `--trunk-compile`.
  Everything else is the launch recipe: `--opt dd` (AdamW, wd 1e-4, 0.5× encoder), clip 10, the 10 cm
  map, the A9 box head, the three selection mechanisms, `--no-strategic`, `--graft-tac8-prior`, and
  the conflict probe every 10.
* **The instrument** wraps `build_optimizer` and the optimiser instance's `step`. It records a
  sha256 of EVERY parameter after every step and of every gradient before it, the grad state
  (None / exact-zero / non-zero), and at the end every AdamW state tensor by NAME.
  `code/freeze/compare_digests.py` compares two runs parameter by parameter, step by step, plus the
  run's own `metrics.jsonl` value by value (wall clocks excluded).
* **Batteries:** `code/freeze/run_ab_batch.sh` / `run_ab_batch2.sh`. Evidence is in
  `raw/freeze/battery1/`, `raw/freeze/battery2/` and `raw/freeze/live_ckpt1500_convert_record.json`.
  Each run's verdict is its `digests.json` `error` field, never a shell exit code.
  * ⚠️ The batch scripts first logged `rc=$?` after a `$(date)` substitution. That printed `rc=0`
    for F6N, which in fact REFUSED. The scripts are fixed; the verdicts below were read from the
    artifacts.

| comparison | what it tests | live params × steps compared | param bytes differ | grad bytes differ | AdamW state (by name) differ | logged values differ | the ten |
|---|---|---|---|---|---|---|---|
| **U12 vs U12b** (launch code twice) | **determinism control**: does this rig reproduce itself? | 610 × 12 = 7,320 | **0** | **0** | 0 / 604 | 0 / 6,936 | grad None every step; never moved from init |
| **U12 vs F12** (launch vs FROZEN) | **the freeze** | 590 × 12 = 7,080 | **0** | **0** | 0 / 584 | 0 / 6,936 | launch: IN the optimiser (20 tensors), grad None every step, no AdamW state, never moved. Frozen: `requires_grad False`, out of the optimiser (604 → 584 entries), never moved |
| **U6R vs F6R** (both RESUME U6's step-6 pre-freeze checkpoint; F6R after `refcv7_ckpt_freeze_convert.py`, `ZZCONVERT-OK-584-20ZZ`) | **resume across the freeze** | 590 × 6 = 3,540 | **0** | **0** | 0 / 584 | 0 / 3,460 | as above |
| **F6N** (frozen code, the UNconverted checkpoint) | the loud failure | — | — | — | — | — | **REFUSED**: `ValueError: loaded state dict contains a parameter group that doesn't match the size of optimizer's group` at `opt.load_state_dict`, 0 steps |
| U12 vs **U12L5** (launch code, log every 5) | **log-cadence control** for lever 2's arms | 610 × 12 = 7,320 | **0** | **0** | 0 / 604 | 6 / 1,793: all six are the interval COUNTERS `trunk_frame_slots` / `trunk_frames_computed` (5× at log-every 5, as they must be) | — |
| U12L5 vs **L12** (lever 2, log every 5) | **lever 2** on the real trainer (it is a no-op when every step is logged, hence log-every 5) | 610 × 12 = 7,320 | **0** | **0** | 0 / 604 | **0 / 1,793**; the identical **597-key** sets on every logged row (steps 5, 10, 12) | — |
| U12 vs **K12** (`--conflict-every 50`) | **the cadence lever** (the probe ran at pre-step 10 in U12 and not in K12) | 610 × 12 = 7,320 | **0** | **0** | 0 / 604 | **0 / 6,893** (the `cd_*` reading row exists on one side only) | — |
| U12L5 vs **B12** (**THE BUNDLE, R4**: the tree with all four patches, blob-equal to `apply_bundle.py --option R4`'s output on all 11 code/test paths; `--conflict-every 50` passed on the command line; log every 5) | **the recommended default, end to end** | 590 × 12 = 7,080 | **0** | **0** | 0 / 584 | **0 / 1,749** | frozen, out of the optimiser (604 → 584), never moved; in the reference: grad None every step, no state |
| **positive control** (U6R vs U12 on global steps 6–11) | the comparator DETECTS real divergence: a resume does not restore the torch RNG | 610 × 6 | **464 / 490 / 505 / 517 / 517 / 575** of 610 per step | — | — | — | (and U6 vs U12 on steps 0–5: 3,660 / 3,660 equal) |
| U12 vs **O12** (`--map-hires-grad-ckpt off`) | the grad-ckpt lever (informative: CPU) | 610 × 12 = 7,320 | **0** | **0** | 0 / 604 | 0 / 6,936 | — (the run log confirms `grad_ckpt False`) |

**Read this as.**
* **The freeze moves no parameter the run trains, on the real trainer.** Not one byte, in 12
  from-scratch steps or in 6 resumed steps. The frozen modules are identical too: they never moved in
  either arm. This is MEASURED on CPU.
* **On Thor** the same holds by the mechanism shown on the live checkpoint. The ten have no AdamW
  state after 1,500 steps, so every CUDA step also saw `.grad is None`, and `clip_grad_norm_` and
  AdamW receive the same tensor lists with or without the freeze. That mechanism is not a numerics
  measurement ON Thor, and none can be made without stopping the run.
* **The determinism control is what makes "0 differ" informative.** Two launch-code runs agree bit
  for bit on this rig. A difference in any other comparison would therefore be the lever's, not the
  rig's.
* **Lever 2 and the cadence lever are bit-identical too** (L12, K12), and so is the whole
  bundle (B12). So R4 changes nothing about what the run learns. It changes when two instruments
  read: the per-class 10 cm signal and the box census on unlogged steps (never logged anyway), and
  the conflict probe (every 50 instead of every 10).
* ⚠️ **"Bit-identical" here means "this code change alters no arithmetic".** It does not mean "the
  restarted run continues bit-for-bit". The trainer checkpoints no RNG state, so ANY resume draws
  fresh noise from the first resumed step. That includes the free crash restart. The positive
  control above shows exactly that: 464–575 of 610 parameters differ per step. It is a property of
  every restart, and the levers add nothing to it.
* ⚠️ **Which variance these answer.** These are single-rig, CPU, b1 replays. They answer "does this
  code change alter the arithmetic?" (no), not "would another training run agree?". No such question
  arises for a bit-identical change.

---

## 5. The items, one by one

### 5.1 Item 2: the model-side freeze of the 10 admitted modules (CLOSURE-TOUCHING, package-only)

**What.** The G-LIVE admission list becomes a model-side declaration, using the batch-3 mechanism
already used for `offset_head`, `control_head` and `scorer.goal_point`.

* `refc_v3.BYPASS_DECLARATIONS` + `declare_bypassed_by_flag(model)`, called LAST in
  `RefCV3Model.__init__`. Each module is frozen with `declare_grad_unreachable(module, why)`, gated
  on the core-config flag that bypasses it: `no_strategic` for 8 modules, `graft_tac8_prior` for 2.
* `declared_vs_built.GRAD_UNREACHABLE_BYPASS_RULES` is the independently written ARGV mirror
  (`--no-strategic` / `--graft-tac8-prior`), folded into `expected_grad_unreachable`. So G-DVB
  refuses in BOTH directions: an undeclared bypass, or a declaration argv does not justify.
* A bypass row whose module the build did not construct is not demanded. Example: `nav_to_str`
  under `--goal-point-inject`.
* `code/freeze/apply_freeze.py` applies it exact-once, REFUSES any base but the launch blobs, and
  asserts the result blobs: `refc_v3.py` a5e71567 → **72b62a0c**, `declared_vs_built.py`
  87cad54e → **d314cd13**.
* The gate profile's `live_dead_admitted` rows are **left as they are**. Once the groups are frozen
  they leave G-LIVE's tracked set, and a lost freeze would be refused by G-DVB anyway. Removing the
  rows is an optional gate clean-up: `test_launch_gate.py` pins them, and so does test 1 here.

**Numerics.** §4. Every other parameter's trajectory is bit-identical (MEASURED). The ten had
`.grad is None` on every step, in the launch code and on Thor, and never moved from init.

**Checkpoint compatibility** (MEASURED on the LIVE checkpoint: step 1,500, `ckpt.pt` md5
`c35966f7…`, pulled read-only and md5-verified against Thor):
* The model state dict loads **strict** into the frozen build (the canonical 154-token argv, r101,
  416×1024, built by the trainer itself on the dev box).
* The optimiser **cannot** resume unconverted: `param_groups` [316, 492] → [316, 472].
  `torch.optim.Optimizer.load_state_dict` refuses a size mismatch, loudly, as the frozen trainer
  would (the refusal is pinned by test 7a and re-run on the real trainer in §4 F6N).
* `stack/scripts/refcv7_ckpt_freeze_convert.py` rewrites `ckpt["opt"]`.
  * The mapping is proven on all **788** state entries (shape by name).
  * The **20** dropped entries carry **no** AdamW state. On the live checkpoint they are exactly the
    20 of 808 with no state: indices 323–328, 455–458, 569–570, 599–604, 608–609.
  * The result loads into the trainer's own `build_optimizer`: `ZZCONVERT-OK-788-20ZZ`
    (`raw/freeze/live_ckpt1500_convert_record.json`).
* ⚠️ **What the no-state fact proves.** AdamW creates a parameter's state on the first step it has a
  `.grad`. So "no state at step 1,500" means `.grad is None` on every one of the first 1,500 Thor
  steps. That also means the decoupled weight decay never touched them, because torch applies it
  only to parameters with a gradient. This is what makes the freeze bit-identical for everything
  else: `clip_grad_norm_` and AdamW both already skip them.

**Tests** (`code/freeze/stack/tests/test_refcv7_admitted_freeze.py`, 12 tests, every expectation a
literal):
* the three tables are the same literal ten, and batch 3 is untouched;
* the refcv7 rig (the eval-loader test's rig, the REAL `train()` captured at the model build)
  declares the literal 13 and freezes their 26 tensors, 20 of them the ten's;
* G-DVB agrees, and goes RED without either flag or on an undeclared build;
* the eight `--no-strategic` declarations are MEASURED true by `probe_grad_unreachable` on the
  trainer's own loss;
* flags OFF leaves the build untouched; flags ON gives a state dict bit-equal to an undeclared
  build, so no RNG is consumed;
* `build_optimizer` excludes exactly the 20;
* the converter loads only after conversion, and REFUSES a dropped entry that carries state;
* existence: `nav_to_str` is not demanded under `--goal-point-inject`.

**Results:**
* **GREEN 12/12** on the frozen tree.
* **RED 12/12 on the launch code (M0).**
* Each mutation arm is **RED**:
  * M1, the argv table loses a row: 6 failed;
  * M2, the freeze is undone after declaring: 8 failed;
  * M3, the existence filter is removed: 1 failed;
  * M4, the converter's state-drop refusal is removed: 1 failed;
  * M5, the model table drops the tac8 pair: 5 failed.
* Logs: `raw/freeze/`.

### 5.2 Item 3: the speed levers as ready-to-apply code

| lever | closure-touching | argv-sha-touching | numerics | ready code | saving (§3) |
|---|---|---|---|---|---|
| conflict cadence 10 → 50 (or 100) | no (but the box closure hashes the argv FILE) | **yes**: `6402d33d…` → `e46ad3eb…` (100: `98e17170…`) | **bit-identical, MEASURED** (§4 K12: 0 of 7,320 parameter-steps differ although the probe ran in one arm only) + pinned by `tests/test_refcv6_grad_conflict.py` (`autograd.grad` never writes `.grad`) | `code/argv/refcv7-r101-s0.conflict50.argv.json` (+ `conflict100`, `conflict50_ckptoff`, `conflict50_eval1000`); `code/argv/make_argv_variants.py`. Each sha is cross-checked with the gate's own `argv_sha256` | 0.56–0.70 s/step |
| lever 2: per-class 10 cm signal + box census on LOGGED steps only | **yes**: `refc_v3_train.py` (203b437f) is in BOTH closures | no | **bit-identical, MEASURED here on the real trainer** (§4 L12: 0 of 7,320 parameter-steps, identical 597-key logged rows) + INHERITED dev-box GPU in-process A/B (step-cost package) | the landed step-cost `code/apply_lever2.py` (203b437f → fd874293). **Re-verified today against the launch blob: OK** | 0.2–0.7 s/step |
| grad-ckpt off | no | yes (`62bf43b7…` combined with conflict 50) | equivalent: **bit-identical on CPU, MEASURED** (§4 O12: 0 of 7,320). **CUDA bit-identity NOT shown**, because the recompute replays the forward on possibly non-deterministic kernels | the `conflict50_ckptoff` argv | 0–0.3 s/step; +~21 GB |

⚠️ **The gate-level risk of lever 2, analysed and not rehearsed.** It gates only per-class METRIC
keys and census keys, on unlogged TRAINING steps.
* G-LIVE's declared terms are LOSS keys (`map_hires`, `box3d`, `agent_presence`, `agent_centre`,
  `tac_v6`, `cascade`, `loss`/`traj`/`cls`), and lever 2 leaves all of them in place.
* `map_train_row_reasons` reads LOGGED rows. The step-cost smoke showed the identical 598-key sets
  at logged steps.
* The eval path is untouched (`or not model.training`).
* ⇒ G-LIVE should pass. **UNVERIFIED**: the G-LIVE smoke needs Thor. The restart's own gate run
  settles it.

### 5.3 Item 4: `V2CompressedCache.__getstate__` drops `newest_frame_only` (CLOSURE-TOUCHING, package-only)

* **Defect.** `__getstate__` returns 5 of the 6 constructor attributes. `decode_stacked_range` reads
  the sixth on every decode. So a SPAWN-started DataLoader worker dies on its first decode with
  `AttributeError`, whatever the flag's value. This was reproduced exactly by the new test's spawn
  arm on the launch code.
* **The live run is unaffected** (MEASURED read-only on Thor):
  * Thor runs CPython **3.12.3** on Linux, where fork is the default start method.
  * The trainer's train DataLoader passes no `multiprocessing_context`, and nothing in the trainer
    calls `set_start_method` (grep: 0 hits).
  * The six live `pt_data_worker` processes are children of the trainer PID and carry **the
    trainer's own argv** in `/proc/<pid>/cmdline`. A forked child inherits it; a spawned one would
    read `-c "from multiprocessing.spawn import …"`.
  * ⚠️ Python ≥ 3.14 makes forkserver the Linux default. The defect would then reach Thor.
* **Fix.** `code/getstate/apply_getstate_fix.py`: ee61ea93 → **f20aca4c**. The key is carried; an
  old pickle without it reads False.
* **Test.** `test_v2_cache_pickle_state.py`: 5 tests, literals.
  * The attribute census is derived from a FRESH instance, never from `__getstate__`.
  * One arm runs a real `multiprocessing_context="spawn"` worker.
* **Results:**
  * **RED 5/5 on the launch code**;
  * **GREEN 15/15** with `test_v2_dataset.py` and `test_newest_frame_only.py`;
  * partial-fix arms **RED**: G1 (setstate forgets the key) 5 failed; G2 (an old pickle reads True)
    1 failed.

### 5.4 Item 5 (I3): old 100-query records through `refcv6_loader.build_model` (CLOSURE-TOUCHING, package-only)

* **Routing through the refcv7 loader does NOT solve it** (read from source):
  * `refcv7_loader` applies the as-trained rule to the AGENT head only (`agent_queries_as_trained`);
  * its box head is built at the `PerceptionBranchConfig` default of 300 (`build_perception_block`
    reads no `n_queries` stamp).
* `taniteval/tools/refcv3_arm.load_model` (outside both closures) remains the working route for
  pre-A9 records **today**. The I3 agent MEASURED it at 100/100, strict.
* **Patch** (`code/i3/apply_i3_patch.py`): the two lines of refcv3_arm's rule go into the vendored
  loader.
  * The agent head uses `agent_queries_as_trained` after the re-parse.
  * The box head takes `n_queries` from the `refcv6_perception` stamp.
  * The A11 pin is re-made in the same patch:
    * audit body blob 11808258 → **4e823c73**;
    * `refcv6_loader.py` 14450c78 → **e66b4bc0**;
    * `g_box_overfit.py` ad88b61e → **99108358** (`AUDIT_LOADER_BLOB`);
    * `test_g_box_overfit.py` cb6812e1 → **7cfba30d** (the literal).
  * The provenance block now says "the audit file + the I3 patch".
  * The EvalFlyWheel battery copy and the audit copy are banked evidence and are NOT edited. The
    battery runs on its pre-A9 `ev6` tree and is not exposed.
* **Test.** `test_refcv6_loader_stamped_queries.py`, 4 tests.
  * **RED 3/4 on the launch code.** The exact I3 strict-load failure on BOTH heads:
    `core.agent_head.queries` and `_perception.box_dec.queries`, [1,100,256] vs [1,300,256].
  * **GREEN 6/6** with the two vendored-loader pins of `test_g_box_overfit.py`.
  * Partial-fix arms **RED**: I1 (agent rule removed) 3 failed; I2 (box stamp ignored) 2 failed.

### 5.5 The combined tree: neighbour suites

* **The combined tree** (all four patches + the three new test files; blob-equal to
  `apply_bundle.py --option R4`'s output on these 11 paths; the canonical argv unchanged):
  * 32 neighbour test files, **688 tests**;
  * **0 new failures and 0 fixed** against the launch tree's 667 on the same 29 shared files
    (per-test JUnit diff, `raw/suites/junit_diff_combo_vs_launch.json`).
* `test_launch_gate.py` reads 150 passed, 2 skipped on both.
* **The argv change and its test pin** were tested on the full R4 tree (§6 step 2):
  `test_launch_gate.py` + the three new files → **171 passed, 2 skipped**.
* `test_tactical_goal_underpowered_matches_census.py` fails 10/11 **identically on both trees**. It
  needs a Research Lab artifact that a stack-only archive does not carry, so it is environment, not
  a regression.
* ⚠️ **This is not the full suite.** G-SUITE (a) at the restart commit is the lander's full-suite
  run.

---

## 6. Runbook for R4 (the PI decides; the Master Mind lands; the operator runs)

**Before the stop (training continues):**
1. **The PI rules:** R4 (with the cadence change), or R0.
2. **The Master Mind builds the commit.**
   * Extract the tip with `core.autocrlf=false`.
   * Run `python code/apply_bundle.py <tree> --option R4 --check`, then without `--check`. It
     REFUSES any drifted base and asserts every result blob. MEASURED on a fresh archive of the tip
     0c44408: `raw/apply_bundle_R4_on_clean_0c44408.log`. It changes **13** paths
     (`raw/apply_bundle_R4_tree_blobs.json`); the 11 code/test paths are blob-equal to the tree §4
     and §5.5 tested.
   * ⚠️ **The argv change also moves a TEST PIN.** `test_launch_gate.py::test_the_canonical_argv_is
     _the_FINAL_list…` pins `_FINAL_ARGV_SHA256 = 6402d33d…`.
     * MEASURED: it goes **RED** on the R4 tree without the update.
     * The applier updates the literal (`374dd421` → `e1ad8320`).
     * With it, on the R4 tree: `test_launch_gate.py` + the three new files → **171 passed, 2
       skipped** (`raw/pytest_R4_bundle_tree_gate_and_new_tests.log`).
   * Land the result as ONE commit. The argv file `stack/ops/runs.d/refcv7-r101-s0.argv.json`
     becomes the conflict-50 variant (gate sha256 `e46ad3eb4099…`).
   * ⚠️ `--check` refuses if a closure file has moved on the tip since `fec3a0d`. Every applier
     pins its base blob.
3. **The lander's full-suite verdict** at that commit (G-SUITE (a)), then `suite-bind`. The dev-box
   **G-SUITE-PINNED** (~11 min).
4. **Ship the new tree to Thor** (git-archive tar, md5).
   * Prepare the two binding run dirs exactly as `/home/nvidia/refcv7_bind/run_{map,box}` were
     prepared, on the NEW tree and argv.
   * Run both preflights with `PREFLIGHT_ONLY=1`. At launch they checked every input while the GPU
     was busy.

**The stop (Thor):**
5. **Stop cleanly.**
   * Wait for the `[v3:hier] ckpt step N -> ckpt.pt` line in `train.log`. The save precedes the
     eval, so no training step is lost.
   * Then wait ~4 more minutes for that step's `[v3:eval] step N` line. A resume at N does not
     re-run the step-N eval, so killing earlier leaves a hole in the Training Watch series.
   * Kill the **supervisor** by explicit PID, then the **trainer** by explicit PID. Never `pkill -f`.
   * Confirm the GPU is empty.
   * Copy `ckpt.pt` to `ckpt_prefreeze_stepN.pt` and md5 both.
6. **Map binding RUN**, then **box binding RUN** (~34 + ~85 min, sequential, exclusive GPU).
7. **Thor gate stage** (~40 min) with the new records and closures, the suite verdict and
   `pi_cost_approval.json`. Read the verdict from the TOKEN FILE, never from the exit code.
8. **Convert the checkpoint** (R1/R4 only):
   `python stack/scripts/refcv7_ckpt_freeze_convert.py --tree <new tree> --argv-file <new argv> --ckpt-in <out>/ckpt.pt --ckpt-out <out>/ckpt.pt.frozen`.
   * It prints `ZZCONVERT-OK-788-20ZZ` on the live run's shape.
   * It REFUSES if any dropped entry carries AdamW state.
   * Move `ckpt.pt` aside and put `ckpt.pt.frozen` in its place.
9. **Relaunch.**
   * Write a new manifest (CODE, COMMIT, GATE_TOKEN, GATE_ARGV_FILE). OUT_DIR is unchanged.
   * Start a FRESH supervisor. The old lock fd rule applies; wait for the old processes to be gone.
   * Assert that it is running (`ps -eo args | grep -c 'sup[_]refcv7'`).
   * The trainer RESUMES at step N.
10. **Record the switch.**
    * MODEL_REGISTRY: "steps 0–N on fec3a0d, N+ on <new commit>, training numerics bit-identical
      (this package §4); conflict readings every 10 until N, every 50 after".
    * After the freeze, `config.json` counts **96,199,317** trainable parameters, not 100,455,641.
      The 4,256,324 never-trained parameters leave the declared budget.

---

## 7. Side findings, conflicts, incidents

* **Every restart is a fresh RNG draw** (MEASURED, §4 positive control). `ckpt.pt` carries model,
  optimiser, step and data position, but **no RNG state**. So even the free crash restart does not
  continue the trajectory bit for bit. This is pre-existing, and it applies to R0 as much as to R4.
  Whether to checkpoint the RNG is a separate, small, closure-touching change; it is not proposed
  here.
* **A registry-vs-evidence conflict (minor).**
  * MODEL_REGISTRY's refcv7 row quotes the G-LIVE smoke at **9.946 s/step**.
  * The PASS token's own G-LIVE evidence (`/home/nvidia/refcv7_gate_fec3a0dccf/evidence/G-LIVE.json`,
    pulled) reads `s_per_step_mean` **9.952** and median **9.0742**.
  * Not decision-relevant. Flagged per the registry rule.
* **The binding runs were A19 MAIN-only** (SPEC_REFCV7 §24). A restart re-binding under the same
  policy costs the 1.98 h measured. Re-binding WITH the must-fail arms would cost more; it was not
  priced, because the profile's `overfit_main_only` policy still admits MAIN-only.
* **The G-SUITE (a) full-suite verdict** at the restart commit is the lander's run. Its duration is
  not measured here. It is dev-box time, not stopped time.
* **Incident: my own battery script (disclosed).**
  * It first logged `rc=$?` after a `$(date)` substitution, which printed `rc=0` for the F6N run that
    had in fact REFUSED. This is the CLAUDE.md `$?`-through-a-substitution trap.
  * Every verdict in §4 was read from the artifacts (`digests.json` `error`, the `ZZCONVERT-OK`
    token). The scripts now capture `rc` first.
* **Incident: battery 2 ran twice for ~1.5 min (disclosed).**
  * A `TaskStop` on its queued wrapper left the script's own bash alive, and it started `U12L5`
    beside the relaunch, into the same directory.
  * Both instances were killed by explicit PID (no pattern kill). The directory was deleted and
    battery 2 re-ran from scratch as ONE instance. Every battery-2 number is from that clean rerun.
  * The script now refuses a second instance (`mkdir` lock).
* **Dev-box resources.**
  * **CPU only.** The GPU lock was never taken and the RTX 4060 was never used.
  * The eval agents' processes ran alongside, and no RAM-floor event occurred (free commit ≥ 18 GB
    throughout).

---

## 8. Deliverable manifest

Every path below is under the package `TanitAD Research Lab/Architecture & Inference/Research/2026-09-28-refcv7-restart-options/`
unless it says otherwise. **Nothing is staged, committed or pushed**; the Master Mind lands per
`LANDING_READY.txt`.

| artifact | where | one place only? |
|---|---|---|
| `RESTART_OPTIONS.md` (this file), `LANDING_READY.txt` | package | yes, until landed |
| **closure map**: `code/closure_map.py` → `raw/closure_map.json` | package | yes, until landed |
| **freeze** (package-only): `code/freeze/apply_freeze.py`, `code/freeze/stack/tanitad/refs/refc_v3.py` (72b62a0c), `code/freeze/stack/tanitad/train/declared_vs_built.py` (d314cd13), `code/freeze/stack/scripts/refcv7_ckpt_freeze_convert.py`, `code/freeze/stack/tests/test_refcv7_admitted_freeze.py`, `code/freeze/mutate_freeze.py` | package | yes, until landed |
| **numerics battery**: `code/freeze/numerics_ab.py`, `compare_digests.py`, `run_ab_batch.sh`, `run_ab_batch2.sh`, `bank_battery.py`; evidence `raw/freeze/battery1/`, `raw/freeze/battery2/` (per run: `run_spec.json`, `digests.json.gz`, `metrics.jsonl.gz`, `log_tail.txt`; `cmp_*.json`) | package | yes, until landed |
| freeze test logs: `raw/freeze/pytest_FROZEN_green.log`, `pytest_M0_LAUNCH_code_RED.log`, `raw/freeze/mutations/` (M1–M5) | package | yes |
| **live checkpoint conversion record** `raw/freeze/live_ckpt1500_convert_record.json` + `live_ckpt1500_convert.log` (`ZZCONVERT-OK-788-20ZZ`) | package | yes |
| the live step-1,500 checkpoint copy `ckpt_step1500.pt` (1.17 GB, md5 `c35966f7…` = Thor's) and its converted twin `ckpt_step1500.frozen.pt` (sha256 `7bcf7d61…`) | **dev box only**: `C:/Users/Admin/rc7restart_work/liveckpt/`. NOT banked: the original lives on Thor, and the converted file is regenerated by the converter in ~1 min | dev box only (deliberately) |
| **argv levers**: `code/argv/make_argv_variants.py`, `code/argv/refcv7-r101-s0.{conflict50,conflict100,conflict50_ckptoff,conflict50_eval1000}.argv.json`, `code/argv/argv_variants.json` | package | yes |
| **getstate** (package-only): `code/getstate/apply_getstate_fix.py`, `code/getstate/stack/tanitad/data/v2_dataset.py` (f20aca4c), `code/getstate/stack/tests/test_v2_cache_pickle_state.py`; logs `raw/getstate/` | package | yes |
| **I3** (package-only): `code/i3/apply_i3_patch.py`, `code/i3/stack/tanitad/eval/refcv6_loader.py` (e66b4bc0), `code/i3/stack/scripts/g_box_overfit.py` (99108358), `code/i3/stack/tests/test_g_box_overfit.py` (7cfba30d), `code/i3/stack/tests/test_refcv6_loader_stamped_queries.py`; logs `raw/i3/` | package | yes |
| partial-fix mutation arms: `code/mutate_small_fixes.py`, `raw/mutations_small_fixes/` | package | yes |
| **bundle applier** `code/apply_bundle.py` (+ `raw/apply_bundle_R4_check.log`, `raw/apply_bundle_R4_on_clean_0c44408.log`, `raw/apply_bundle_R4_tree_blobs.json`, `raw/pytest_R4_bundle_tree_gate_and_new_tests.log`) | package | yes |
| neighbour suites: `code/run_neighbour_suites.sh`, `code/diff_junit.py`, `raw/suites/` (JUnit XML ×61, `run.log`, `junit_diff_combo_vs_launch.json`) | package | yes |
| Thor pulls (read-only, md5-verified): `raw/thor_bind/` (the closures with clip ids REDACTED, both binding records, chain logs, `thor_md5.txt`, `REDACTION.json`), `raw/thor_gate/` (the PASS token, verdicts, job logs, G-SUITE-PINNED / G-CKPT evidence) | package; the originals stay on Thor | no (Thor holds the originals) |
| live pace data (Thor `metrics.jsonl` / `train.log` / `config.json` at step 1,800) | `raw/live/` | no (Thor holds the originals) |
| `code/make_landing_ready.py` | package | yes |
| scratch trees `C:/Users/Admin/rc7restart` (launch code), `rc7frozen`, `rc7lever2`, `rc7combo`, `rc7bundle` (R4 applied), `rc7restart_work/` (battery run dirs incl. 0.6 GB timing-rig checkpoints) | dev box only | dev box only; worthless as models, reproducible from the package |
