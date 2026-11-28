# RESULT -- L7 train-split fan capture (H-L7-1): what was built, what each control read, what it costs

Written 2026-10-22 00:30 +0200 (Berlin) by the Architecture & Inference agent (l7cap), dev box. Zero GPU, no compute on Thor, nothing staged or committed.
Evidence classes: **MEASURED** = produced here on 2026-10-21/22, artifact named; **INHERITED** = read from a banked artifact; **ESTIMATED** = arithmetic on a MEASURED rate;
**UNVERIFIED** = could not be run here. Eval tier: every number below is a **CPU fp32-trunk smoke** or a **machinery check**, not a model result (no tier stamp applies; nothing here is driving performance).
The sealed prereg (`scorer9/PREREG_L7_SCORER.md`, sha256 `f9c1431c...`) and everything under `scorer9/` and `stack/` were **not edited**.

## 1. What exists (one line each; paths relative to `C:/Users/Admin/refcv8_pf/l7cap/`)

| piece | file | state |
|---|---|---|
| train window rule + 4,855 windows | `RULE.md`, `TRAIN139_WINDOWS.json` (+ `.sha256`, `TRAIN139_VERIFY.json`) | MEASURED, 3 controls green |
| capture wrapper | `code/fan_roll_r8_l7cap.py` + `code/l7cap_hooks.py` | MEASURED on CPU smoke (train + eval) |
| controls as functions | `code/controls_l7cap.py`, `code/build_gt6.py` | MEASURED |
| clip-id scrub (new, 00:2x) | `code/scrub_clip_ids.py` | MEASURED, 5 tests, whole package CLEAN |
| dev-box GPU launcher | `launch/run_l7cap_devbox.sh` | WRITTEN, NOT RUN; queue/lock/marker logic tested (T1-T4) |
| Thor launcher + chain | `launch/run_l7cap_thor.sh`, `launch/chain_l7cap_thor.sh`, `launch/make_a6cache_thor.py`, `launch/refcv8_train_remap_thor_l7cap.json` | WRITTEN, NOT RUN; `bash -n` OK; md5 literals frozen |
| CPU smoke chain | `launch/smoke_l7cap_cpu.sh`, `code/run_capped.py` | RUN (this is the P4 evidence) |
| tests | `code/tests/` (hooks 6, controls 4, make_a6cache 3 arms, devbox script T1-T4, scrub 5) | all PASS, logs in `smoke/test_*.log` |

## 2. What the capture dumps per window (MEASURED, `smoke/wrapped/s0/fan_roll_s0.json`)

`fan_extras.npz`: `anchor_traj [N,149,8,2]` (the full fan), `sel_score`, `refined_logits`, `r8_sub_logits [N,149,4]`, `is_alloc` (32 True = the 32 allocated, 117 base), the
`e9_keep`/`e9_argmax` masks from the model's own e9 function, `reach_keep`/`ceil_keep`/`r8_emit_keep`, `sel_idx`, `sel_idx_base`, `v0`, `goal_dist`, `inference_seed`.
`l7cap_scene.npz`: detached `r7_agent_tokens [N,300,384]`, `r7_agent_pad [N,300]`, `r7_bev_tokens [N,480,96]` (fp16, 0 fp32-fallback rows, max |x| 0.73 / 6.3).
`r7_bev_pad` is **absent in all 3/3 train and 8/8 eval windows** (the scene hook passes none for refcv8; the reason is INFERRED, not read from the model).
`l7cap_labels.npz`: `r7_agent_fut [N,8,397,5]` + `_valid` + `_label` (agent join, train and eval); `map_drivable_u8 [N,120,64]` + `map_seen` + `map_label` (SAM3 crop, **eval only here**).
`cond` / `r7_cond` (the nav + fed ceiling) are **deliberately not stored** (nav-echo rule); `FORBIDDEN_BANK_KEYS` is asserted at write time, and `conds_seen = [nav_true]`, `n_calls_navzero = 0`.

## 3. Controls (all MEASURED on 3 train windows of 1 clip and 8 eval windows; CPU, fp32 trunk, `CUDA_VISIBLE_DEVICES=-1`)

| control | arm | expected (committed before reading) | read | verdict |
|---|---|---|---|---|
| (a) K1 identity, train | wrapped vs 2 independent unwrapped rolls (`smoke/K1_train139_s0.json`) | bit-identical to >= 1; every int array identical; float diff <= unwrapped-vs-unwrapped envelope | 72 arrays, bit-identical to **both**, envelope 0.0 | **PASS** |
| (a) K1 identity, eval os-only | wrapped vs 1 unwrapped (`smoke/K1_eval139_s1.json`) | bit-identical (envelope undefined with one reference, so 0) | 124 arrays bit-identical | **PASS** |
| (a) K1 discrimination | unit tests (`test_l7cap_controls.py`) | an empty or one-sided comparison, a differing pick, and a float diff with no envelope must each FAIL | all three fail | green |
| (b) machinery, train | `l7_targets_pai.build` own-pick ADE vs recompute from dumped `sel_idx` candidate vs recompute from the roll's emitted plan `os` | agree to 1e-6 m | n=3, mean ADE(2 s) 0.4635 m, max |diff| **8.9e-9 m** on both routes | **PASS** |
| (b) machinery, eval | same | same | n=8, mean 0.4143 m, **2.2e-8 m** | **PASS** |
| (b) literal | synthetic fan, candidate k = GT + 0.1k m, pick 7 | own-pick ADE exactly 0.7 m | 0.7 within 1e-5 | green |
| (c) deliberate regression | wrapper run with `L7CAP_MUTATION=wrong_sel_idx` (dumps `sel_idx+1`) | RED | roll-time K3a **4.17 m** and child rc=1; machinery builder-vs-emitted-plan **0.3358 m** > 1e-6 -> PASS False | **RED, as required** |
| (c) literal | synthetic world, same mutation | 0.8 vs 0.7 = 0.1 m | 0.1 within 1e-5 | green |
| (c) scene mutation | `swap_sources` in the hook (`test_l7cap_hooks.py`) | RED | RED | green |
| roll-time K3a | max |fan[sel_idx] - emitted traj| | 0.0 m | 0.0 (train and eval) | pass |
| roll-time K3b / K4 | traj == sidecar `nav_true`; `e9_argmax == sel_idx` | True; fraction 1.0 | True; 1.0 (min `e9_keep` per row 147 train, 142 eval) | pass |

Reading note (machinery): the route "recompute from the dumped fan at `sel_idx`" is **blind** to a wrong-index mutant (builder and fan agree: 1.3e-8 m on the mutant); only the
second route, against the roll's emitted plan, catches it. Both are reported so the blind spot is visible.

**Not decided on CPU: K3** (the captured fan equals the banked `fanceil` fan for the same checkpoint, windows and seed). The CPU fp32-trunk smoke is not the Thor GPU bf16 roll that produced
the banked fan: on the 8 eval windows the median |diff| against `fanceil` s1 was 7.6e-2 m and only 19 % of cells were within 1 cm. That is cross-hardware numerics, **not** a capture defect
(K1 shows the wrapper changes nothing on one box), but it also means the K3 read is **UNVERIFIED until the Thor GPU eval139 capture runs** (chain steps 3-4).

## 4. CPU smoke: memory and time (MEASURED, `smoke/smoke.log`, `run_capped.py` job-object sampling; limit 6 GB commit, kill at 3.8 GB working set)

| run | working set peak | commit peak | wall |
|---|---|---|---|
| wrapped train139 s0, 3 windows, agent join | **3.70 GB** | 4.48 GB | 121 s |
| unwrapped train139 s0 (reference U / U2) | 3.63 / 3.62 GB | 4.41 GB | 38 / 44 s |
| mutant | 3.62 GB | 4.42 GB | 44 s |
| wrapped eval139 s1 os-only, 8 windows, agents + map | **3.76 GB** (max of all) | 4.48 GB | 165 s |
| unwrapped eval139 (K1 reference) | 3.37 GB | 4.09 GB | 85 s |

* **Every process stayed under 4 GB working set** (max 3.76 GB, none killed). The **commit charge reached 4.48 GB** (the 6 GB cap never bound). The margin is ~0.25 GB: a 4 GB RAM budget is met on the
  working set only, and the dev-box commit limit is the thing that binds here (`devbox-commit-limit-binds`).
* **Seconds per window (CPU, 4 threads, fp32 trunk):** model call median **8.72 s** (train, n=3, range 7.97-9.42) and **9.18 s** (eval, n=8). Wall including dataset build: 39 s/window on the 3-window train smoke and
  20 s/window on the 8-window eval smoke (setup-dominated; ~14 s fixed). **CPU is not a route for the real job:** 4,855 windows x ~9 s = ~12 h of model calls per seed.
* **Capture overhead is NOT isolated.** The wrapped train smoke took 121 s against 38-44 s for the unwrapped references (3 windows each), and the wrapped eval smoke 165 s against 85 s for the unwrapped os-only one.
  The extra ~80 s appears in both pairs and is most plausibly the one-time load of the agent-join file (the wrapped log shows `agent join: 139/139 episodes, 22946/23751 windows labelled (96.6 %)`; the unwrapped runs skip it) -- INFERRED, because no
  join-free wrapped run was made. The wrapped model-call time (median 8.7 s) is consistent with the unwrapped totals minus ~14 s of setup over 3 windows, so there is no visible per-window overhead on CPU, but
  the per-window capture cost on GPU is **UNVERIFIED** (the 10-30 % allowance in section 6 is a guess, not a measurement). The 96.6 % join coverage means ~3.4 % of the windows carry no agent future (`_valid` False); a scorer that trains on the agent-future target must mask them.
  (Two earlier `W2` log lines, 23:57 and 00:02, are runs of the buggy smoke script that executed the UNWRAPPED command; they are not used anywhere.)
* A GPU run on the dev box would hold the scene rows in host RAM until written (~1.5 GB per seed, below); whether the GPU process stays under the 4 GB working set is **UNVERIFIED**
  (the CPU runs above carry the model activations in host RAM, so a GPU run should have a lower base, but this was not measured).

## 5. Bank size (MEASURED per window from the smoke dump `smoke/wrapped/s0/dump`; scaled)

`fan_extras.npz` 46,706 B / 3 windows = 15.6 KB/window; `l7cap_scene.npz` 904,998 B / 3 = **301.7 KB/window**; `l7cap_labels.npz` 5,795 B / 3 = 1.9 KB/window (compressed; worst case 63.5 KB/window if
the agent futures are dense float32).

| bank | windows | scene | fan + labels | total |
|---|---|---|---|---|
| train139 pilot, per seed | 4,855 | 1.46 GB | 0.08 GB (up to 0.4 GB dense labels) | **~1.6 GB** (2 seeds ~3.2 GB) |
| eval139 K3 capture, per seed | 4,754 | 1.43 GB | 0.08 GB | ~1.5 GB |
| full plan (train4572 at 1 in 4, **~39.9 k windows**, 2 seeds) | ~79.8 k | ~24 GB | ~1.3 GB | **~25 GB** (ESTIMATED) |

The full plan cannot sit in one process's RAM (scene rows ~12 GB per seed): it must be sharded (`RULE.md` proposes shards of 139 clips, shard 0 = the pilot). Thor has 103 GB free of 937 GB (89 % used, probe 00:21).

## 6. GPU-time estimate (ESTIMATED; basis = MEASURED Thor rate)

**Basis.** The banked Thor GPU fan roll of the same checkpoint on eval139: `fanceil/runs/s1/fan_roll_s1.json` -- 4,754 windows in **1,346.5 s = 0.283 s/window** (s0: 1,327.3 s = 0.279), `cuda_max_memory_allocated` 1.257 GiB.
Capture adds hook + fp16 conversion + banks + label join; on CPU that is inside the noise, so I carry **+10 to +30 %** as an allowance (UNVERIFIED on GPU).

| job | windows | model time | with allowance |
|---|---|---|---|
| train139, one seed | 4,855 | 22.9 min | **25-30 min** |
| train139, seeds 0 and 1 | 9,710 | 45.8 min | 50-60 min |
| eval139 K3 captures, seeds 1 and 0 (`--os-only`) | 9,508 | 44.8 min | 49-58 min |
| **the chain (4 steps)** | 19,218 | 90.6 min | **~1.7-1.9 h Thor GPU** |
| K3 only (eval139 seed 1) | 4,754 | 22.4 min | 25-29 min |
| full plan, 2 seeds, ~79.8 k windows | ~79.8 k | 6.3 h | 7-8 h |

The dev-box RTX 4060 rate for refcv8 is **not measured** (no banked dev-box GPU roll of this model was used here), so no dev-box GPU-time is quoted. Thor was running another GPU job at probe time;
`run_l7cap_thor.sh` waits on `thor_gpu.lock` and fails closed if `nvidia-smi --query-compute-apps` is not empty.

## 7. Window rule facts that a reader must not lose (MEASURED, `TRAIN139_VERIFY.json`)

139 clips, **4,855 windows**, episode-disjoint from eval139 (0 of 139 by sha12), the same A6/RETIME sample the registry already carries (K-CLIP equals the banked `train_clips_sha12.json` and the A6 cache names),
windows equal the banked train-fit roll's windows on all 139 clips. **Stride and horizon rule match eval139 (5 ticks, `ws = t + 7`); the phase does not** (train139: `t mod 5 = 0` on every clip;
eval139's first-window phase varies by clip). The generator was trained on these clips, so the fans are in-sample (closer to GT); the pilot must report the pick-to-oracle gap beside eval139's 0.215 m and
may never tune on it (`RULE.md`).

## 8. Clip-id rule

No file in `l7cap/` holds a full clip id or an 8+ hex prefix of one. The 12 smoke text files that did (dump `manifest.json` x6, `roll.log` x6; 16 tokens) were rewritten to `clip-sha12-<12 hex>` by
`code/scrub_clip_ids.py` (report `smoke/SCRUB_CLIP_IDS.json`); the whole tree then read CLEAN against the 4,369-id Thor pool list plus the ids learned from the manifests (79 text files, 34 npz string-checked,
0 residual). The raw-id pool list used for the check stays in the session scratchpad and is **not** in the package. Both launchers now run the scrub after the roll and
assert on the `CLEAN` flag in `SCRUB_CLIP_IDS.json` (dev box: marker `ZZL7CDSCRUBZZ clean=<0|1>`; Thor: `scrub=<0|1>` in `ZZL7COUTZZ`, and exit code 4 if the roll worked but the scrub did not read CLEAN).
Machinery re-run on the scrubbed smoke dump: PASS, identical numbers (`smoke/MACHINERY_wrapped_s0_after_scrub.json`).

## 9. Defects found and fixed on the way (the learning, not just the fix)

| defect | how it showed | fix / class |
|---|---|---|
| K-WIN used `range(T-W-20+1)` (4,861 vs 4,855 on 6 clips) | the control against the banked trainfit dump failed | bound quoted from `_contract.py:121`; the independent check is what caught it |
| K1 returned PASS on an empty comparison | I read a 0-array PASS | `n_arrays > 0` and identical array sets required; test added (empty-string-blob family) |
| the mutant and the second wrapped run wrote no banks | my branch rewrite of the smoke script silently did not apply (both runs were unwrapped) | assert the bank files exist before reading a control; reran the whole chain on final code |
| the first `run_capped.py` measured the python launcher, not the job | peak looked tiny | job process list sampled; `PeakProcessMemoryUsed` added |
| Thor sha12 probe read 0 present | the list had CRLF endings | `\r` stripped; list passed via ssh stdin |
| raw clip ids in smoke manifests/logs | the scrub scan | scrub tool + both launchers |
| a python `.replace` that silently no-ops (twice) | the file was unchanged | assert `count == 1` before replacing (used for the launcher edits) |
