# HANDOFF_NAVSIM_R8 — the refcv8 NavSim read, built and smoke-tested; the full read is waiting on a checkpoint, a GPU slot and four PI decisions

EvalFlyWheel · 2026-10-20 (Europe/Berlin) · package `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/` (NEW files only; the plan, the censuses and `camera_smoke_r8.py` already in the repo are untouched).
Nothing committed, nothing staged, nothing written to G:. Tier on every number: **NavSim open-loop benchmark (T1-family), zero-shot PhysicalAI-AV → nuPlan cameras, non-parity, never closed loop.**
Evidence classes: MEASURED (this package's `raw/`) · INHERITED (plan / refcv7 package, not re-verified) · ESTIMATED.

## 1. What exists

| file | what it is |
|---|---|
| `code/boot8.py` | one place that decides the tree: the refcv8 launch tree `C:/lgs/Lfb47a58376` (read-only, no `__pycache__`), `r8_repoint` (battery code dir), `load_retime_stage()` |
| `code/refcv8_bridge.py` | the port: arms, command→token, `r8_nav`, speed row, `load_refcv8` (26 premises), `forward_kwargs8`, `run_model8`, `retime_plan8`, mutations. Everything unchanged from refcv7 (frames, ego window, prior, lift, spline) is IMPORTED from the refcv7/refcv6 suites, not copied |
| `code/run_bridge8.py` | per-split bridge → `rows_<arm>.jsonl` (resumable), `seam_<arm>.npz` + manifest, `PRIOR_ha0p`, `R8_RT` |
| `code/run_navsim_refcv8.py` | one command: checkpoint → bridge → official scorers → `INDEX.json` |
| `code/score_navtest8.py` | recorded port of `score_navtest7.py` (label prefix r7→r8, nothing else) |
| `tests/test_inputs8.py` (63) · `test_mutation8.py` (8) · `test_model_seam8.py` (6) | feed-parity + known-value tests, deliberate-regression arms, real-checkpoint controls |

## 2. The feed (what refcv8 receives from a NavSim `AgentInput`)

* **frames** 416×1024 CYLINDRICAL re-stitch of CAM_L0/F0/R0, f_ref 488.924 px, column linear in azimuth, hfov 120.000° (the pinhole formula gives 92.6° — pinned by a test). Re-stitch of the warmup scene reproduces the landed smoke bit for bit (sha16 `d44e7bc225ec67a7`, max |Δ| 0, observed 0.8817 — `test_restitch_reproduces_the_landed_smoke_geometry`, MEASURED). ST: every 10 Hz slot ← t0.
* **nav_cmd** = the token of `driving_command[3]` ALONE (the only declared command field; earlier-frame commands are randomised by test K5 and the inputs stay byte-identical, so the token is the t0 / NOW-row token by construction and a clip-level token cannot enter — R8-EVAL-NAV-FEED-1):

  | NavSim one-hot | key | refcv8 `nav_cmd` | name |
  |---|---|---|---|
  | `[1,0,0,0]` | left | 1 | TURN_L |
  | `[0,1,0,0]` | forward | 0 | FOLLOW (never 3 = "straight": the trainer never feeds 3) |
  | `[0,0,1,0]` | right | 2 | TURN_R |
  | `[0,0,0,1]` | unknown | 0 | FOLLOW (LEGAL fallback) |

  Vocabulary source lines (substring-pinned by `test_vocabulary_source_lines_are_pinned`): `stack/scripts/build_v9_labels.py:260` `NAV_FOLLOW, NAV_TURN_L, NAV_TURN_R = 0, 1, 2`; `stack/scripts/refc_v3_train.py:4686` `R8_V9_NAV_TO_NAV_CMD = {0: 0, 1: 1, 2: 2}`; `stack/tanitad/train/refcv8_train.py:71-72` `NAVSIM_LEGAL_NAV_CMD = {"left": 1, "forward": 0, "right": 2, "unknown": 0}`. A non-one-hot vector is REFUSED.
* **r8_nav** = `scale_nav(zeros[1,5], known=False)` = exactly zeros [1,6] float32, passed on EVERY forward. Training's lines: `refcv8_train.py:563-567` `known &= ~(gen.rand((b,), dev) < float(model._r8_nav_args_dropout))` … `if legal: known = torch.zeros_like(known)` … `fwd["r8_nav"] = scale_nav(nav, known)`; `scale_nav` (`:367-378`) returns `cat([out * k, k])`, so a dropped row is zeros even when the args are non-zero (`test_unknown_row_equals_the_dropout_path_even_with_nonzero_args`). `r8_rc` is not passed (`--r8-no-rc`). `None` and the zero row give the same plan (`test_KNAV`, MEASURED on the 40k EMA).
* **speed** R8_LEGAL = `speed_input_treatment(..., legal=True)` → `(v_max_ms 0, v_max_valid 0)`. R8_PRIV = the map limit snapped to its containing bin, fed as `bin_kmh/3.6` computed in Python float64. ⚠️ MEASURED while testing: refcv7's `ceiling_ms` (float64 scalar rule) reads the float32 cast of 50/3.6 as ABOVE the 50 km/h step and returns the 100 km/h ceiling (27.78 m/s); the model's float32 tensor rule keeps it in the 50 bin. `refcv8_bridge.ceiling_ms` bins through the tensor path (pinned by a test). refcv7 fed raw map limits (not snapped edge values), so it is not known to be affected — not audited here.
* **ego**: v0 = |v| at t0; window 8×(x,y,yaw,v) linear between the 2 Hz states; a0 and yaw rate derive INSIDE the forward (`ha0_ext_pose`); no curvature and no acceleration field is declared. **Output**: `out["traj"]` 8 knots at 0.5…6 s → E2's `knots_to_navsim` (C2 not-a-knot cubic spline through (0,0) + knots, evaluated at 0.5…4.0 s; heading = atan2 of the tangent; GT round-trip < 1e-3 m).
* **R8_RT**: opt-in. `--retime --retime-table <json>` (+ `--retime-stage-file`; the launch tree PREDATES `retime_stage.py`). No default table; the stage refuses a table whose `.sha256` sidecar disagrees. The 8-slot call = SHIP-R8 AMENDMENT 2's function (the stage docstring's warning about the 8-slot call predates it). Derived from R8_LEGAL's rows: no extra forward.

## 3. Pre-flight (all must hold, in this order — the cost of skipping any is in CLAUDE.md)

1. `E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt` and `ckpt_48852.pt` exist; the runner reads the step INSIDE the file and labels `RESULT` only for 48,852 with `--weights` matching the file name; any other step is `VALIDATION ONLY`.
2. `python C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py show` → no holder. **At this writing the holder is `refcv8-navfeed`** (the nav-feed experiment) and `chain_final_r8.sh` / `pull_final_r8.sh` / `ship_after_r8.sh` are live. YIELD to: the per-frame train-fit (`hold_r8pf`, `chain_trainfit_pf`), the 48,852 final chains, the nav-feed experiment. The runner queues on the lock (no priority, no preemption). `--device cuda` waits `gpu_wait_s` (900 s warmup, 10,800 s navtest/navhard; `--gpu-wait-s` overrides) and then REFUSES — it never degrades to CPU; `--device auto` would fall back to a CPU split (~76 h for the full set). **Use `--device cuda --gpu-wait-s 86400` for the real read.**
3. RAM: ≥ 6 GB free (the runner waits); `export OMP_NUM_THREADS=6` (torch spawns ~113 threads per process; unset, concurrent arms stall); `R7_SCORE_RAM_GB` default 8 gates the scorers.
4. Tests on the FINAL checkpoint: `R8_TEST_CKPT=<ckpt_ema_48852.pt> pytest -q tests/test_model_seam8.py` plus `pytest -q tests/test_inputs8.py tests/test_mutation8.py` (CPU, ~7 min).

## 4. Commands (dev box, Git Bash; bridge = TanitAD venv, scorers = NavSim venvs, chosen by the runner)

```
export TANITAD_REPO=C:/lgs/Lfb47a58376 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval'
export R8_BATTERY_CODE=<dir with r8_repoint.py + refcv8_eval_remap_devbox.json>   # default: ../2026-10-15-refcv8-milestone-battery/code
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; P=<this package>; K=E:/refcv8_eval_kit/ckpt; O=C:/Users/Admin/refcv8_eval_archive/navsim
# 1. warmup first (minutes): the end-to-end gate; R8_BLIND must FAIL N3 or the panel is VOID (SPEC_REFCV8 6.2)
$PY -X utf8 $P/code/run_navsim_refcv8.py --ckpt $K/ckpt_ema_48852.pt --weights ema --splits warmup --device cuda --gpu-wait-s 86400 --out $O/step48852_ema
# 2. the BAR read: EMA weights, R8_LEGAL (seed 0) + R8_LEGAL_s1 (seed 1) + R8_PRIV (diagnostic) [+ R8_RT iff D-NS8-3 allows a table]
$PY -X utf8 $P/code/run_navsim_refcv8.py --ckpt $K/ckpt_ema_48852.pt --weights ema --splits navtest,navhard --device cuda \
    --out $O/step48852_ema [--retime-table <gains.json> --retime-stage-file <retime_stage.py>]
# 3. the RAW weights, beside (PI 2026-10-16)
$PY -X utf8 $P/code/run_navsim_refcv8.py --ckpt $K/ckpt_48852.pt --weights raw --splits warmup,navtest,navhard --device cuda --gpu-wait-s 86400 --out $O/step48852_raw
```
Resumable: rerun the same command (rows are skipped, a scorer whose counts read PASS is skipped). Arms per split (A-R8-NS1): EMA = R8_LEGAL, R8_LEGAL_s1 on every split, R8_PRIV on navtest, R8_BLIND on warmup, R8_RT derived; RAW = R8_LEGAL. `--arms` overrides (e.g. `R8_NT`, `R8_NAVOFF` are built but not default). Artifacts: `bridge_<split>/seam_<arm>.npz|manifest.json`, `scores_<split>/…counts.json` (guard) + CSV, `families_<split>_<arm>.json`, `INDEX.json`, `MILESTONE_SUMMARY.json`. **Assert on the artifacts (`counts.status == PASS`, manifest `partial == false`, `KPR.verdict == PASS`), never on an exit code.**

## 5. Compute

| quantity | value | class |
|---|---|---|
| CPU fp32, exact-dedup, 4 threads, forward per scene (5 navtest scenes × 3 arms, 40k EMA) | median **4.09 s** (3.16–6.71, n = 15); 5.11 s (4.00–7.07, n = 15) with CPU scorers running concurrently | MEASURED `raw/smoke_cpu_ema40k_VALIDATION_ONLY/` rows (`s`) |
| model load (CPU) | 10–27 s; RSS 1.4 GB after load; box free RAM 6.4–8.0 GB during the smoke | MEASURED |
| scoring wall for 5 tokens | 17–45 s (start-up dominated; the 342 s PRIV job in the first run waited on the RAM governor) | MEASURED |
| refcv7 CUDA bf16 (4060) | 0.35–0.42 s/scene | INHERITED (plan §3) |
| **refcv8 CUDA** | **UNVERIFIED** — the GPU lock was held (nav-feed), so no CUDA forward was timed. CPU ratio refcv8/refcv7 = 4.09/2.58 = 1.6 ⇒ **ESTIMATED 0.35–0.67 s/scene** | ESTIMATED |
| forwards | navtest 12,146 × 4 (EMA s0, s1, PRIV, raw) = 48,584; navhard 5,912 × 3 = 17,736; warmup 220 × 4 = 880 → **67,200** | arithmetic |
| ⇒ GPU time | **≈ 6.5–12.5 GPU-h** (plan said 7–10; the upper end is the CPU-ratio case) · CPU fallback 67,200 × 4.1 s ≈ **76 h: refused** | ESTIMATED |
| CPU scoring | navtest v1.1 15–25 min/arm × ~5 arms ≈ 1.3–2.1 h; navhard ≈ 3.2 h/arm, ≤ 4 in parallel ≈ 3.5–7 h wall | INHERITED (refcv7) |
| disk | rows ≈ 4 KB/scene/arm ⇒ ~0.3 GB for the whole read; C: has 62 GB free | MEASURED/arithmetic |

**Replace the GPU estimate with a measurement** the first time the lock is free: run `--splits warmup` (880 forwards, ~10 min) and read `s_per_scene_forward_mean` from the seam manifests.

## 6. Where it runs (PI decision D-NS8-1) — both written out

* **Dev box 4060 (default if silent).** Everything above works as written. Backend-consistent with the banked floors (STOP 61.8202, CV 20.6517, HUMAN 94.5514, PRIOR_ha0p 58.6613, refcv7-50,400 LEGAL 70.2048 — INHERITED, plan §5.2). Waits behind the nav-feed experiment and the 48,852 final chains on the lock.
* **Thor.** `RULING_BACKEND_POLICY.md` §3 says refcv8 NavSim rows go to Thor first. **Not validated here and not touched (Thor is training).** Needs: the launch tree on Thor; a Thor twin of `r8_repoint` (its `REMAP_OVERRIDES` file is hard-wired to the dev-box kit paths — use `stack/ops/refcv8_eval_remap_a2_thor.json` and `R8_EVAL_KIT`); the 22 GB navtest bank (~6 min at ~63 MB/s, MEASURED per the memory note) + the navhard/warmup banks (size not measured) over the LAN; the tanitad-edge venv for inference. The one-command runner has dev-box paths (`PY`, `BANKS`, `CONFIG`); on Thor call `run_bridge8.py` directly (every path is an argument), copy the seams back, and **score on the dev box** (the scorers need the NavSim devkit venvs, not a GPU). Floors must then be re-scored on Thor or the comparison stated as MIXED backend.
* A40 pod: not recommended (plan §3).

## 7. Blocked — each item, and what unblocks it

| blocker | owner | unblocks |
|---|---|---|
| the final checkpoint `ckpt_ema_48852.pt` / `ckpt_48852.pt` (expected 2026-10-20 evening) | refcv8 training → `chain_final_r8.sh` | the extract step lands the files |
| the dev-box GPU lock (`refcv8-navfeed`; then the final-battery chain) | whoever holds it | release; this read queues, never preempts |
| **D-NS8-1** where the read runs | PI | dev box (default) or Thor (§6) |
| **D-NS8-2** download the Ego-Status-MLP control (3 × ~6.5 MB) | PI | until then `PRIOR_ha0p` stands in; the control script is not built (it needs the weights) |
| **D-NS8-3** does R8_RT carry a bar | follows SHIP-R8 at 48,852 + D-R8-SHIP-RETIME | default: the bar stays on R8_LEGAL, R8_RT beside; the table is the SHIP-R8 deploy table (`stack/tanitad/configs/refcv8_retime_deploy_gains.json`, sidecar sha256 `f18359be…219e`, INHERITED) — the smoke used it only to exercise the path |
| **D-NS8-4** leaderboard go/no-go | PI | default no submission; ⛔ nothing here submits anything |
| **statistics / bars layer NOT BUILT** (`parse*8`, `decompose8`, `bars8` for BAR-R8-N1…N4; paired log-cluster bootstrap vs STOP and vs refcv7; stop-rate column; four-family tables beyond `families7`) | EvalFlyWheel — ~1 agent-day, needs SPEC_REFCV8 §6 re-read | the refcv7 copies hard-code `R7_` arm names and SPEC_REFCV7's bars, so they were not blindly ported; `INDEX.json` carries the scorer's per-arm rows meanwhile |
| **strata not computable yet** | MM/PI | A-R8-NS1 registers strata "t0-speed band" **without edges**; fix the edges BEFORE any refcv8 number exists (nothing else blocks city / command / EP-clause strata, which come from the inputs JSON + scorer CSV) |

Rule-Zero levers if a bar misses (all 0-training, from plan §5.3): R8_NT (static-stack sensitivity), a mirror arm on the 1,914 Singapore tokens (needs pre-registration), D6 P4's drivable gate on the refcv8 fan through `refe/navsim_dac.py`, R8_PRIV − R8_LEGAL with a paired CI to price the speed input.

## 8. Smoke — VALIDATION ONLY (never refcv8's NavSim score)

`raw/smoke_cpu_ema40k_VALIDATION_ONLY/` — `ckpt_ema_40000.pt` (md5 `57807456bd7169800be159ff95871d22`), CPU fp32, 5 navtest tokens (`raw/smoke5_tokens.json`: left/Boston, right/Singapore ×2, forward at speed, forward standing), through `run_navsim_refcv8.py` (bridge → devkit v1.1 PDM scorer → families). n = 5, so no interval and no inference. Finite sub-scores, all guards PASS (`C1_max_abs_delta 0.0`, 5/5 rows), KPR (model-emitted prior vs model-free) max 0.0 m:

| arm | NC | DAC | EP | TTC | C | DDC | PDMS (×100) |
|---|---|---|---|---|---|---|---|
| R8_LEGAL (seed 0) | 100.0 | 60.0 | 15.9292 | 100.0 | 100.0 | 100.0 | 41.6371 |
| R8_LEGAL_s1 | 100.0 | 60.0 | 14.2611 | 100.0 | 100.0 | 90.0 | 40.9421 |
| R8_PRIV | 100.0 | 60.0 | 28.7829 | 100.0 | 100.0 | 100.0 | 46.9929 |
| R8_RT (deploy table, path exercise) | 100.0 | 40.0 | 14.8842 | 100.0 | 100.0 | 100.0 | 29.5351 |
| PRIOR_ha0p (model-free) | 80.0 | 20.0 | 6.8809 | 80.0 | 100.0 | 40.0 | 14.5337 |

A same-config rerun reproduced every row exactly (determinism, MEASURED n = 5). Tokens fed: TURN_L, TURN_R, TURN_R, FOLLOW, FOLLOW for the five commands; `r8_nav` all-zero and `v_max_valid` 0 on every R8_LEGAL row (read back from the rows).

## 9. After the rolls: parse + bars (added 2026-10-20 by the Master Mind; A-R8-NS1 AMENDMENTS 2 and 3, GOALS_AND_CLAIMS)

The rolls of section 4 produce scores, not verdicts. The verdicts come from two CPU steps, in this order:

```bash
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; P=<repo>/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim; O=<…>/step48852_ema
F=D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/floors
# (a) navhard / warmup summaries with R8 arm names (parse8 = parse7's statistics, plus the seed-1 paired rows; PARSE8_README.md)
$PY -X utf8 $P/code/parse8.py --split warmup_two_stage --scores $O/scores_warmup --bridge $O/bridge_warmup --floors $F/warmup_two_stage \
  --inputs D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/raw/navsim_agent_inputs.json --label RESULT --out $O/summary_warmup.json
$PY -X utf8 $P/code/parse8.py --split navhard_two_stage --scores $O/scores_navhard --bridge $O/bridge_navhard --floors $F/navhard_two_stage \
  --inputs C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json --csv-suffix __navhard_two_stage --label RESULT --out $O/summary_navhard.json
# (b) the four bars. --city-census is EXPLICIT (Amendment 2): the landed raw/navtest_city_census.json (sha256 ccb9fc42…; dev-box copy
#     C:/Users/Admin/navsim8/navtest_city_census.json). --ship-r8-passed 1 ONLY if SHIP-R8 passed on the PER-FRAME feed (SHIP-R8 Amendment 3).
PYTHONPATH="$P/code;<repo>/taniteval" $PY -X utf8 $P/code/navsim_bars8.py --run-dir $O --tag step48852_ema --out-dir <dir> \
  --navhard-summary $O/summary_navhard.json --warmup-summary $O/summary_warmup.json \
  --city-census C:/Users/Admin/navsim8/navtest_city_census.json --ship-r8-passed <0|1>
```

**Check before reading any verdict:** `controls.city_census.status == "OK"` (1,914 Singapore / 12,146 tokens); `ep_clause_source_consistent`
and `v0_inputs_vs_scorer_rows` read OK; the label reads `RESULT (EMA 48,852, the bar read)`.

**The rules the verdicts apply (registered before any 48,852 NavSim number):**

| bar | rule |
|---|---|
| N1 | PDMS(R8_LEGAL) > STOP; paired interval excludes 0 above at log_name AND nuplan_drive; margin > 2× the full-split inference floor; both seeds |
| N2 | official two-stage EPDMS > max(STOP, CV, ECHO); paired vs STOP above 0 at log_name AND nuplan_drive (Amendment 3; the record's own `separated` is direction-agnostic and is never read as a pass); margin > 2× the seed floor; R8_LEGAL and R8_LEGAL_s1 each, combined strictly |
| N3 | warmup S2-EPDMS-u > max(CV, STOP, ECHO); margin > 2× the seed floor; both seeds (Amendment 3); R8_BLIND must FAIL or the panel is VOID |
| N4 | navtest PDMS(R8_LEGAL) − PDMS(refcv7-50,400 LEGAL) > 0, separated; margin > 2× the inference floor (Amendment 2: the SPEC's general NOT PROVEN rule; 1× printed beside); both seeds; MIXED-BACKEND stated |

A seed-1 category that differs from seed 0, or a seed-1 mirror that cannot be evaluated, reads **NOT PROVEN**, never PASS.
End-to-end control (refcv7-30k through the same code): N2 FAILED at both seeds, N3 NOT PROVEN, the SPEC 6.2 reference row
(`raw/e2e_parse8_bars8_am3/E2E_CONTROL_refcv7_step30000.json`).
