# STEP 1 IS BUILT — the v6 S-W latent dumper, and the second dead-gate coupling it exposed

**Date:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803` · **Author:** arch-inf subagent (S-W dumper stream)
**What it unblocks, verbatim:** the PI's *"eventually we need a tactical selector"* — E4 (2026-08-17) repaired the READER half of SEL-1's reopening path and MEASURED that the PRODUCER half did not exist. **This is that producer.**
**Nothing was launched. Thor's GPU was never touched** — two read-only `ssh -n` probes. All verification ran CPU-only on the dev box.

---

> ## ⛔ ESCALATION — FOUR THINGS THE ORCHESTRATOR MUST CARRY
>
> **1. ⭐ STEP 1 EXISTS AND THE WHOLE PATH IS PROVEN END-TO-END — but the MEASUREMENT is still outstanding.** `scripts/v6_dump_sw_latents.py` is built, wired into `v6_chain.py admission` as a runnable command (step 1 no longer reads `⛔ NOT BUILT`), and the join **producer → estimator → chain reader → `assert_selector_admissible`** is MEASURED on a planted σ at **all three verdicts**: planted 0.30 → recovered **0.3046** → **FUNDED** and the selector launch is ADMITTED; 1.10 → **1.1274** → **INCONCLUSIVE**; 2.00 → **1.9856** → **REFUSED**. The script existing is not the dump existing.
>
> **2. ⛔ THE SCRIPT IS NOT ON THOR, AND `git fetch` ON THOR HANGS.** MEASURED this session: `~/TanitAD/stack/scripts/v6_dump_sw_latents.py` → **absent** (`raw/thor_readiness.json`). Every pod-side script this campaign arrived by md5-verified FILE-SHIP; a chain step that git-syncs would hang and, worse, could reset the tree to an ancient HEAD. ⇒ **Ship `v6_dump_sw_latents.py` (and the current `v6_chain.py`/`e_wc2_sigma_star.py`) to Thor and grep-verify BEFORE the S-W→S-T boundary.** This is the one remaining hand-off and it is not mine to do unilaterally.
>
> **3. ⛔ A SECOND DEAD-GATE COUPLING, MEASURED — AND IT IS THE SAME DEFECT AS C94 ONE DOOR ALONG.** `e_wc2_sigma_star.run` writes `references_and_ratios.sigma_perax_2s_m` **only inside `if refs.get("available") and vstep in sig`**, and `fan_references` returns `available: False` when the dump carries no `fan`/`gt`. That key is exactly what `v6_chain.SW_SIGMA_LOCATIONS` resolves. ⇒ **A latent-ONLY dump — the obvious reading of "dump the frozen S-W latents" — produces an admission artifact with NO σ in it at all, and the gate stays DEAD.** The contract called `fan`/`gt`/`sel` *"required_for_the_ratios"*, which is why this was easy to miss. Executed and pinned (`test_a_dump_with_no_fan_leaves_the_admission_gate_with_nothing_to_read`), the contract's own text corrected, and the fan made non-optional in this producer.
>
> **4. ⚠️ EXPECT `NO_VERDICT` FROM THE ESTIMATOR ON THE REAL RUN — IT IS NOT A FAILED DUMP.** The live S-W arm has `selector: null` (MEASURED in its own `config.json`), so `emit` produces no `sel_*` key and the dump carries **no `sel`**. σ/ADE is therefore not computable and `e_wc2_sigma_star` correctly emits `NO_VERDICT` for its OWN §5.2 ratio test — **while still writing the per-axis σ in ABSOLUTE metres, which is what `SW_LATENT_ADMISSION` (≤ 0.80 FUNDED · > 1.41 REFUSED) adjudicates.** Both halves MEASURED together in `raw/sw_dumper_roundtrip.json`. An operator who sees `NO_VERDICT` and stops has thrown away a good measurement.

---

## 1. What was blocking, and what the fix had to be

E4's finding, restated from its own source: *"Step 1 of the four-step admission recipe **does not exist**: nothing in `stack/scripts/` dumps v6 S-W latents in E-WC2's dump contract (MEASURED at three probes; `refc_dump_latents.build_model` builds a `RefCModel` and cannot load a v6 checkpoint)."* Steps 2–4 were runnable.

⇒ The deliverable is a **producer**, and the binding constraint on it was stated by the brief in the sharpest possible terms: **C94's root cause was a fixture that modelled the CONSUMER'S EXPECTATION instead of the PRODUCER'S OUTPUT.** So the contract below is derived line-by-line from the consumer's source, and **no test in this stream hand-writes a dump** — the pins run the real producer over a real `V6Stack` and hand its actual output to the real estimator and the real reader.

---

## 2. The contract, derived from the consumer — and the two couplings that were NOT in it

| key | shape | where it comes from | which consumer line requires it |
|---|---|---|---|
| `eid` | `list[int]` n | the episode index of each window | `e_wc2_sigma_star.loeo_folds` — LOEO folds are built from this and nothing else |
| `pooled` | `[n, d_op]` | `V6Stack.forward → z_op` | `FEATURE_ADMISSIBILITY["pooled"] = VISION_ONLY`; the recipe's default `--features pooled,ctx` |
| `pooled_seq` | `[n, W, d_op]` | `z_op_win` | same |
| `ctx` | `[n, d_str]` | `z_str` — the v6 analogue of REF-C's StrategicCtx summary | same |
| `z_tac` | `[n, d_tac]` | `z_tac` | no built-in class ⇒ needs `--declare z_tac=VISION_ONLY` |
| `v0` | `[n]` | `pose_last[:, 3]` | `MEASURED_PRESENT` (PI 2026-08-16), carries the anti-echo obligation |
| `gt_endpoint` / `endpoint_valid` / `endpoint_steps` | `[n, 2, 2]` / `[n, 2]` / `[20, 60]` | `refc_dump_latents.gt_endpoints_masked`, **imported** | `validate_dump` — §5.2 requires 2 s **and** 6 s; ⛔ NaN + `valid=False`, never imputed |
| `fan` | `[n, C, 4, 2]` | `plan["waypoints"][:, :, [4,9,14,19]]` | ⛔ **the σ itself** — see below |
| `gt` | `[n, 4, 2]` | `driving_diagnostic.gt_ego_waypoints(poses, last)` | ⛔ same |
| `wp_steps` | `[5,10,15,20]` | `dd.WP_STEPS` | `max(wp_steps)·DT` must be **2.0** or `run` emits `MISMATCH` and no σ |
| `cv` | `[n, 4, 2]` | `dd.baseline_waypoints(...)["constant_velocity"]` | optional — adds §3.1 B's 0-param reference row |
| `sel` | `[n]` long | `plan["sel_score"].argmax(-1)` | **only when the arm really has a scorer** — §5 |
| `controls_vs_bank` / `instrument_fail` | dict / list | the producer's self-controls | `instrument_fail` non-empty ⇒ E-WC2 refuses |

### 2.1 ⛔ COUPLING ONE — the fan is not optional, and the contract said it was

**MEASURED by execution** (`test_a_dump_with_no_fan_leaves_the_admission_gate_with_nothing_to_read`): strip `fan`/`gt` from a real dump and `references_and_ratios` comes back `{"available": false}` with **no `sigma_perax_2s_m` at any level**; `read_sw_admission` then returns `verdict: null` and *"Absence of the number is NOT an admission"*, and `assert_selector_admissible` refuses.

`DUMP_CONTRACT` filed `fan`/`gt`/`sel` under **`required_for_the_ratios`**, and that name is what makes the trap: the σ is *nested inside* the ratios block, so the ratios' inputs are the σ's inputs too. **Root-cause class: the same one E4 repaired — a key whose LOCATION in the output is load-bearing and was documented by its purpose rather than by its address.** Corrected in `DUMP_CONTRACT` itself, in the words an operator will actually read, with the pinning test named inline. Only `sel` is genuinely ratio-only.

### 2.2 ⛔ COUPLING TWO — `--max-horizon 60` must NOT be inherited

MEASURED in the live run's own `config.json`: **`max_horizon: 60`**. `EpisodeWindowDataset`'s index rule is `t_max = frames - window - max_horizon`, so inheriting the run's training horizon would produce a **different, smaller window set** — parity broken, `eid` no longer element-for-element with the banked REF-C dumps, §3.1's surface no longer the same rows. The dumper builds its grid at the canonical `max_horizon = K_MAX_GRID = 20` regardless of what the run trained at, and reads the 6 s endpoint from the episode's own pose array with the pad-and-mask rule instead.

---

## 3. The grid — canonical, and why the model's window does not move it

MEASURED in the live run's args: **`window: 6`**. The canonical val40 grid is `WINDOW=8, STRIDE=8, K_MAX=20` → **881 windows / 40 episodes**.

⇒ **The grid stays at WINDOW=8 and the model is fed the LAST `cfg.predictor.window` frames of each window.** Both windows end on the same frame, so `last`, the ego frame, `pose_last`, `gt` and `gt_endpoint` are on identical rows, and the 6-frame history the encoder sees is exactly the history it trained on. A model window **> 8** is REFUSED (`test_a_model_window_wider_than_the_grid_is_REFUSED`) rather than silently truncated — widening the grid to fit a model would re-select windows, and parity is sacred.

The grid constants and `window_starts` are **imported from `refc_dump_latents`**, and `select_grid` additionally asserts, per episode, that the dataset's own index equals `window_starts` element-for-element. Two producers, one grid definition, and a drift between them cannot be silent.

⭐ **The grid arithmetic was checked against the REAL corpus, not only against the synthetic one.** MEASURED by loading the banked REF-C dump in the repo (`…/incoming/2026-08-04-lambda-findability/raw/latents_refc-xl-30k.pt`): **881 windows / 40 episodes, 22–23 windows per episode**, `wp_steps [5,10,15,20]`, `instrument_fail []`. The test corpus here is 39×22 + 1×23 = **881 at the identical per-episode distribution**, so the grid arithmetic the producer will do on Thor is the arithmetic that was exercised. *(REF-C's fan is 256 candidates and v6's is 8 — irrelevant to σ, which is in absolute metres, and the ratios are not computable on a no-scorer arm anyway.)*

⚠️ **The E-ENC arm (b) is REFUSED up front, not in the first batch.** A `shared_encoder=False` checkpoint needs `own_frames_tac`/`own_frames_str`, and there is no per-layer frame this producer could defensibly invent. It says so before the corpus mounts rather than raising inside `forward` after it. *(The live arm is `shared_encoder=True`; this is a guard against a future arm, not a live blocker.)*

---

## 4. ⭐ VISION-ONLY IS MEASURED HERE, NOT ASSERTED FROM THE DIAGRAM

The argument is available — `z_op` comes from `encode_window(frames)` and `z_tac`/`z_str` from adapters over it, while `actions`/`v0` reach only the predictors and the emission. **The argument is not the measurement.** So the producer re-runs its first batch with `v0` **and** `actions` **permuted across the batch** and requires `pooled`/`pooled_seq`/`ctx`/`z_tac` to be **bit-identical**. Same construction as `probe_latent_state.py --speed-echo-control`: a claim about what an input cannot reach, tested by changing that input.

MEASURED on the real producer (`raw/sw_dumper_roundtrip.json`): `{"vacuous": false, "permutation_changed_inputs": true, "blocks": {"pooled": true, "pooled_seq": true, "ctx": true, "z_tac": true}, "ok": true}`.

**And the control can FAIL, which is what makes it a control.** `test_the_vision_only_control_REPORTS_A_FAILURE_when_a_block_reads_v0` runs it against a stand-in whose `z_op` *does* read `v0`: every block reports `false` and the max abs difference is recorded. Two vacuity branches are refused rather than passed: batch < 2, and a permutation that moved neither input — a degenerate batch carries no evidence, and a control that congratulates itself on one is the `endpoint_agreement` vacuity trap in a new place.

⚠️ `waypoints` is deliberately **excluded** from the invariance set: the unicycle emission reads `v0` as its integration constant by design. Requiring it to be invariant would be a wrong test, not a stricter one.

---

## 5. ⛔ `sel` IS NOT FABRICATED, AND THAT COSTS THE ESTIMATOR'S OWN VERDICT

The live arm is `--selector none`. E4 MEASURED on the production stack that such an arm emits **`sel_* keys: []`**. There is no incumbent selection to record.

The two available fabrications are both disqualifying: **candidate 0 is arbitrary**, and **argmin-over-candidates is the ORACLE**. Either would manufacture `sel_ade` — the σ/ADE denominator — and with it a §5.2 FUNDED/REFUSED verdict out of nothing. **That is precisely this session's root-cause class in a new costume.** So `sel` is absent, `sel_absent_reason` records why in the dump itself, and the consequences are stated rather than hidden:

| | on a `--selector none` S-W arm |
|---|---|
| `e_wc2_sigma_star`'s own §5.2 verdict | **`NO_VERDICT`**, refusal reason *"missing `fan`/`gt`/`sel` — the ratio denominators are not computable"* — correct |
| `references_and_ratios.sigma_perax_2s_m` | ⭐ **written** (it is computed before the guards run) |
| `v6_chain.read_sw_admission` | ⭐ **a real verdict**, on ABSOLUTE metres |

Both halves are MEASURED together (`test_the_admission_gate_reads_sigma_even_though_e_wc2_refuses_its_verdict`), because an operator who reads `NO_VERDICT` as "the dump failed" discards a good measurement — and that reading is the natural one.

---

## 6. ⭐ THE PLANTED-σ ROUND TRIP — producer → estimator → chain, all three verdicts

The corpus is built in two passes so σ is planted against **the encoder's own latents** without faking either half: run the real producer once → rewrite the episodes' POSES so the 2 s ego-frame endpoint is exactly `pooled @ W + N(0, σ)` per axis → **run the real producer again, unmodified**, and push its output through the whole chain. Only pose indices `last + 20` (≡ 3 mod 8) move; the ego-frame reference `last` (≡ 7), the CV baseline's `last-1` (≡ 6) and the other fan waypoints (≡ 4, 1) are untouched, and only x/y are written so `v0` cannot move.

⛔ **The plant must not reach the model side, and that is asserted:** the frames never change, so pass 2's `pooled` and `ctx` must come back **bit-identical** to pass 1's. That assertion is what makes this a plant rather than a circular fit.

MEASURED (`raw/sw_dumper_roundtrip.json`, 881 windows / 40 episodes, `--features pooled,ctx`, 11.8 s CPU):

| planted σ (per-axis, m) | recovered | error | R² OOF (long / lat) | estimator's §5.2 verdict | **chain admission verdict** | selector launch |
|---|---|---|---|---|---|---|
| **0.30** | **0.3046** | **+1.53 %** | 0.9950 / 0.9956 | `NO_VERDICT` (no `sel`) | ⭐ **FUNDED** | **ADMITTED** |
| **1.10** | **1.1274** | **+2.49 %** | 0.9626 / 0.9091 | `NO_VERDICT` (no `sel`) | **INCONCLUSIVE** | REFUSED |
| **2.00** | **1.9856** | **−0.72 %** | 0.7447 / 0.8532 | `NO_VERDICT` (no `sel`) | **REFUSED** | REFUSED |

Read at `references_and_ratios.sigma_perax_2s_m` in every case — the address E4 repaired — with `producer_instrument_fail: []`, 881 windows at 2 s and **721 at 6 s** (the K_MAX mask: 0.8184 valid, the last ~5 windows of every episode excluded and never imputed).

*(For comparison, E4's own repair proof planted 0.30 through a hand-written dump and recovered 0.3026. This recovers 0.3046 through the real producer — the same answer with the producer in the loop, which is the half that was missing.)*

---

## 7. The producer's self-controls — and the teeth test for each

| # | control | what a silent failure would cost | teeth test |
|---|---|---|---|
| 1 | **row alignment** — `v0` from the dataset's `pose_last` vs `v0` from the episode pose array at `last`, bit-identical | two independent index paths; if they disagree every latent sits on a different window's targets and σ is a wrong number that looks right | `test_the_row_alignment_control_CATCHES_a_one_window_shift` |
| 2 | **ego frame / fan rows** — `gt_endpoint[:, 0]` == `gt[:, 3]` bit-identically at the coinciding step 20 | the 6 s horizon has nothing to check against on its own; this pins its frame too | `test_the_endpoint_frame_control_CATCHES_a_wrong_ego_frame` |
| 3 | **vision-only** — §4 | a block that moves with `v0` is a leak magnitude, not a capability | `test_the_vision_only_control_REPORTS_A_FAILURE_when_a_block_reads_v0` |
| 4 | **pre-registered surface** — 881/40, 6 s endpoint present, fan horizon exactly 2.0 s | E-WC2 would refuse three steps later, after the GPU was spent | `test_a_short_surface_is_flagged_by_the_producer_not_three_steps_later` |
| 5 | **non-finite feature values** | a NaN propagates into the ridge as a NaN σ that reads like a number | covered by the real-dump assertion |

MEASURED on the real producer: `v0_batch_matches_poses: true`, `endpoint_20_matches_gt: true`, `reference_horizon_s: 2.0`, `fails: []`.

**Preflight, at startup, before the checkpoint and before CUDA.** Nine module probes plus `taniteval.ci` resolved through `e_wc2_sigma_star._load_ci` — i.e. **step 2's imports are probed by step 1**. This is the `t1_eval.py` lesson (both arms, 40 episodes, ~11 min/arm rolled, then `analyze()` died on `from taniteval import selgap`): an analysis-time import that fails after the expensive part destroys the run while the compute is already paid for. `--preflight-only` answers with no `--ckpt`, no `--out` and no corpus.

---

## 8. Thor readiness — read-only, one `ssh -n`, GPU untouched

`raw/thor_readiness.json`, produced by `code/thor_readiness_probe.sh` (opaque `ZZ…ZZ` marker computed **pod-side**, so no client-side filter contains the token it searches for).

| | |
|---|---|
| live run **PID 25477** | ✅ `Ssl`, elapsed **1-23:51:59**, RSS 9 434 232 kB · at hand-off ✅ `Ssl`, elapsed **2-00:01:22** |
| step | **12 750 / 30 000** · `step_s` **26.4735** · loss **2.3086** — advancing (E4 handed off at 12 650). ⚠️ Same step at both reads by construction: the trainer logs every 50 steps ≈ 22 min, and only 9.5 min separated them — that is a log cadence, not a stall |
| `<sw_dir>/config.json` beside `ckpt.pt` | ✅ **present** ⇒ the dumper rebuilds the architecture from the run's own record; **`--args-from` is not needed** on the canonical path |
| `v2_val_cache` in the run's args | ✅ `/home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl`, **exists** ⇒ no `--v2-val-cache` override needed either |
| run geometry the dump will replay | `window 6` · `max_horizon 60` · `v2_subframe null` · `256x640` · `in_channels 9` · `readout 4x128` (⇒ `d_op` 2048, matching E4's measured `z_op [1, 2048]`) |
| `~/ckpt_snaps` | 4 × fp16 weights-only, **673 312 891 B** each (`step009250/010000/011250/012000`) — readable via `--args-from <run dir>`, pinned by `test_a_weights_only_snapshot_is_readable_with_its_run_record` |
| ⛔ `~/TanitAD/stack/scripts/v6_dump_sw_latents.py` | **ABSENT** — escalation 2 |

⚠️ **No real S-W dump was taken and no real snapshot was pulled.** The val40 corpus lives on Thor, so an end-to-end run against a real checkpoint needs Thor's GPU — which is 42 % through a 30k run and must not be disturbed. The contract, the grid, the controls and the whole join are proven on a real `V6Stack` on the dev box instead; **what is UNVERIFIED is the production geometry's wall-clock and memory**, and the recipe's *"~10–25 GPU-min"* stays **INHERITED**, not re-measured.

---

## 9. Inertness — the live run cannot be affected

| | |
|---|---|
| default build | ⭐ **87 893 449 params / 405 keys** — **unchanged** (`code/default_build_invariant.py`, E4's own checker, re-run) |
| vocabulary tuples | **not touched** — `GoalVocabulary` sizes embedding tables and an edit breaks tensor-strict resume |
| trainer / model source | `train_v6_staged.py` and `tanitad/models/v6.py` **not modified** by this stream |
| what the dumper writes | one `.pt` at `--out`; it never writes into a run directory unless told to, and it loads with `strict=True` |
| Thor | two read-only `ssh -n` probes; nothing started, stopped, written or loaded |

---

## 10. Suites

Run with the interpreter named: `cd stack && PYTHONUTF8=1 OMP_NUM_THREADS=6 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q`.

| suite | result | baseline | delta |
|---|---|---|---|
| **`stack` — full** | ⭐ **3842 passed · 0 failed · 7 skipped · 2 xfailed** (438 s) | 3816 / 0 / 7 / 2 (INHERITED from E4) | ⭐ **+26 — exactly this stream's new file, fully attributable** |
| `stack` — new file `tests/test_v6_dump_sw_latents.py` | **26 passed** (22.6 s) | — | the +26 |
| `stack` — the neighbours (`test_v6_st_launch_fixes` · `test_v6_chain` · `test_e_wc2_sigma_star` · `test_v6_staged` · `test_v6_gstr_port`) | **279 passed** | — | 0 regressions |
| `taniteval` | **1132 passed · 0 failed** (181 s) | 1107 / 0 (INHERITED) | ⚠️ **+25 is NOT MINE.** No `taniteval/` file was touched by this stream; the delta comes from another stream's live untracked files (`taniteval/taniteval/degeneracy.py`, `taniteval/tests/test_k1_degeneracy_guard.py`, both `??` in `git status`). Reported as GREEN, attributed to nobody here. Claiming it would be the provenance error the operating standard exists to stop. |

⚠️ **One incumbent pin was CHANGED, deliberately, and it must be called out at commit:** `test_v6_st_launch_fixes.py::test_E4_the_reopening_recipe_is_EMITTED_not_described` asserted `steps[1]["status"].startswith("⛔ NOT BUILT")`. That assertion encoded a fact this stream changes. It now asserts the repaired state (`✅` + the command naming `v6_dump_sw_latents.py`) and **keeps** the `why_not_reusable` assertion, because *why the REF-C producer could not be reused* is the finding and must not be lost.

---

## 11. Evidence classes

| claim | class |
|---|---|
| every recovered σ, R², window/episode count, control result, param/key count, Thor step/`step_s`/file listing | **MEASURED (ours)** — producers in `code/`, outputs in `raw/` |
| "a latent-only dump leaves the admission gate with nothing to read" | **MEASURED (ours)** — executed on a real dump with `fan`/`gt` stripped |
| "the live arm has no scorer ⇒ no `sel` ⇒ estimator `NO_VERDICT`, chain verdict still real" | **MEASURED (ours)** for the mechanism · **MEASURED** that `selector: null` in the run's own `config.json` |
| "v6 emits `sel_* keys: []` on the production stack" | **INHERITED** — E4 §2, not re-measured here (re-derived independently from `V6Stack.emit`'s `if self.cand_score is not None`) |
| the S-W admission thresholds (0.80 / 1.41 m) | **PRE-REGISTERED** 2026-08-16, before any measurement |
| *"~10–25 GPU-min at the S-W→S-T boundary"* | **INHERITED** (`SW_LATENT_ADMISSION["cost"]`) — ⚠️ **not re-measured; no production-geometry pass was run** |
| σ recovery at the PRODUCTION geometry (`d_op` 2048, `in_channels` 9) | ⚠️ **UNVERIFIED** — the round trip ran at `d_op` 64. n/d changes by ~32×, so the *recovery precision* above is the tiny stack's, not the production stack's. The JOIN and the CONTRACT are geometry-independent; the recovery percentage is not |

---

## 12. Deliverable manifest

| artifact | where it lives | staged |
|---|---|---|
| `stack/scripts/v6_dump_sw_latents.py` — **the producer (step 1)** | `repo:` | yes |
| `stack/tests/test_v6_dump_sw_latents.py` — 26 pins, all on the REAL producer | `repo:` | yes |
| `stack/scripts/v6_chain.py` — step 1 emits a runnable command; recipe docstring + `assert_selector_admissible` message corrected | `repo:` | yes |
| `stack/scripts/e_wc2_sigma_star.py` — `_producer_v6`, the fan-coupling correction in `DUMP_CONTRACT`, the header's producer note | `repo:` | yes |
| `stack/tests/test_v6_st_launch_fixes.py` — the `⛔ NOT BUILT` pin updated to the repaired state | `repo:` | yes |
| `SW_LATENT_DUMPER.md` (this file) | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-18-sw-latent-dumper/` | yes |
| `code/sw_dumper_roundtrip_probe.py` — executes producer → estimator → chain on a planted σ | same, `code/` | yes |
| `code/thor_readiness_probe.sh` — the read-only Thor probe | same, `code/` | yes |
| `raw/sw_dumper_roundtrip.json` — every number in §6, §4, §7 | same, `raw/` | yes |
| `raw/thor_readiness.json` — every number in §8 | same, `raw/` | yes |

**Nothing is stranded.** No file was written on Thor; no Thor artifact was created or modified.

⛔ **Not in this stream's ownership and NOT touched:** `Project Steering/MODEL_REGISTRY.md`, `Project Steering/V6F_PLANNER_DESIGN.md`, `train_v6_staged.py`, `tanitad/models/v6.py`, any v6 vocabulary tuple.

---

## 13. What happens next, in order

1. ⛔ **Ship** `v6_dump_sw_latents.py` + current `v6_chain.py` / `e_wc2_sigma_star.py` to Thor by md5-verified file-ship (never `git fetch`), and grep-verify the file is present before the boundary.
2. Run `python3 scripts/v6_dump_sw_latents.py --preflight-only` on Thor — 2 seconds, no GPU, and it answers whether step 2 could run there at all.
3. At the S-W→S-T boundary (**~5 days**, step 30 000): `v6_chain.py admission` prints all four commands. Run step 1, then 2, then 3.
4. The verdict decides itself: **FUNDED** ⇒ `sel_gap` BINDS at the S-T gate and the selector arm is schedulable; **INCONCLUSIVE / REFUSED** ⇒ SEL-1 stands and the S-T certificate keeps `sel_gap` as `UNMEASURED_BY_CONSTRUCTION` with this same recipe attached. Both outcomes were committed 2026-08-16, before the measurement existed.
