# refcv6 launch readiness — three defects found and fixed before the first GPU-hour (2026-09-23, morning)

**Author:** Master Mind · **Evidence class:** MEASURED (ours) unless stated · **Tier:** none — no
training step has run; everything here is code, data plumbing and run-record correctness.

The PI's order (2026-09-23): train refcv6 once ready — newest corpus with the correct tactical
labels, nav and ego as input, 416 × 1024, SAM3 maps and dataset agents as supervision, end to end,
the DiffusionDrive planner. Last night's fixes (`e152e40`) made the arm correct; this morning's
pass read the LAUNCH itself — the command line, the run record, the resume path and the data the
3-D join needs — and found three more defects.

## 1. The launch line trained an AGENT-ONLY tactical decoder, and `config.json` would have hidden it

* ⛔ My overnight launch line carried no `--tac-decoder-d-bev`. Its default `0` builds the
  behaviour decoder over agent slots only — against PI ruling R2 (*"use also the map for tactical
  behavior decoding"*, binding since 2026-09-17). MEASURED from the smoke's own record:
  `seams.tac_decoder_v6.decoder_cfg.d_bev = 0`, `sources ["agent"]`,
  `bev_tokens_reach_decoder false`.
* ⛔ And `config.json['refcv6_tactical']` wrote `scene_sources_live ["agent"]`,
  `scene_sources_blocked ["bev"]` and an AGENT-ONLY warning as **literals** on every arm — so a
  correct BEV arm would have recorded itself as the arm R2 ruled out.
* ✅ Launch line: `--tac-decoder-d-bev 96` (= `BEVEncoderConfig.d_out`; the pin refuses any other
  width). Trainer: `_refcv6_tactical_block` reads the constructed decoder and the attached
  perception branch — the two facts `RefCV3Model._bev_hook` gates the feed on.
* 🧪 `stack/tests/test_refcv6_bev_tactical_wiring.py` §7 — 5 tests, each cross-checked against the
  forward's own `bev_tokens_fed`. **Mutation 5/5** (`raw/mutation_proof_tac_record.json`,
  harness `code/mutate_tac_record.py`): the historical literal, bev-live-from-width-alone,
  bev-forced-live, the call site dropping the builder, the blocked list back to its literal.

## 2. A relaunch RESUMED and REPLAYED the data order

* `train()` resumes by itself from `<out>/ckpt.pt` (model, optimiser, step) — the prereg's
  *"no `--resume`, a relaunch RESTARTS the arm"* was false. But the loader was
  `DataLoader(shuffle=True)`, a permutation drawn from the global RNG that every launch re-seeds:
  a run resumed at step k re-trained `perm[0 : k·B]` and never reached
  `perm[(steps − k)·B : steps·B]`. On a one-epoch budget resumed at its midpoint, half the corpus is
  trained twice and half never.
* ✅ `ResumableEpochSampler` (epoch-e permutation = pure function of `(seed, e)`),
  `next_train_batch` (the loop step, callable by tests), `resume_data_position` (refuses a changed
  `--batch`, corpus size or seed, naming the field; a legacy checkpoint resumes and says so).
  `ckpt.pt['data_pos']`, `config.json['data_order']`, and `data_epoch`/`data_batch` on every row.
* 🧪 `stack/tests/test_resume_continues_the_data_order.py` — 8 tests, including `train()` end to
  end: a 3+3-step resumed run trains on exactly the windows a 6-step uninterrupted run trains on in
  steps 4–6. **Mutation 7/7** (`raw/mutation_proof_resume_order.json`, harness
  `code/mutate_resume_order.py`); arm D3 restores `shuffle=True` — the historical defect itself.

## 3. The 3-D join build died on a clip that never moves

* The first build died at clip ~340: `RegistrationError: only 0 of 200 poses are moving`.
  Position registration cannot time a stationary clip. The 2-D TRAIN join timed exactly **25** such
  clips from the **camera timestamp grid** (its meta's own `time_source` census: 4,402 by
  registration, 25 by grid).
* ✅ `code/build_grid_block_stationary.py` rebuilds that time base with the 2-D builder's own
  functions (`read_reconstructed_poses` → `grid_reference`). Controls: **25 / 25** refuse position
  registration (so each genuinely needs the block), **25 / 25** cover every frame of their 416 × 1024
  v2ep record (so C9 cannot fail on coverage). The build was relaunched with `--lead-block`, and its
  C9 control requires the recovered time to round to the banked line's `t_s` on EVERY line.
* Per clip (sha12 only): `raw/stationary25_time_base.txt`.

## 3b. The branch could not import its own trainer — found by gating on a CLEAN tree

* The first suite on a `git archive` of the tip plus this commit's files died at collection in
  16 modules: `refc_v3_train.py` imported `refcv7_heads` / `refcv7_oracle` at module scope since
  `d014414`, which landed the trainer from a worktree holding the UNLANDED refcv7 stream. Every
  earlier suite ran on that worktree and stayed green.
* ✅ ONLY the refcv7 modules' own absence is tolerated at import; `--refcv7` refuses by name while
  they are missing. 🧪 `stack/tests/test_trainer_imports_without_refcv7.py` (3 tests; the control
  blocks a DIFFERENT module and must fail). **Mutation 2/2** on the clean tree
  (`raw/mutation_proof_r7_import.json`, `code/mutate_r7_import.py`).
* ⭐ The launch therefore ships a `git archive` of the landed commit to Thor, never a worktree.

## 4. Held-out eval, built so the run reports while it trains

eval-139 view on Thor (139 clips, **0** overlap with the train view); v8 eval labels (md5
`eefc38d1…`, three copies agree); a merged per-clip extrinsics table (4,508 clips; eval camera
heights 1.2131–1.6622 m); SAM3 GT on **137 / 139** eval clips (train control 4,369 / 4,369).
⚠️ The in-run eval scores trajectory, tactical and MAP. Its agent/box terms read **n = 0** — the train
join holds no eval clip — which is the trainer's documented control, not a result.

## 5. The launch line (Thor), with every flag the review and the PI's order require

⚠️ **The eval split needs its OWN max-speed sidecar.** The trainer falls back to the train
sidecar only when the eval labels are the same blob, and its md5 guard refuses otherwise — so the
first launch line would have refused at eval setup. Built from the v8 eval labels
(`refcv6_speed_max_v8_eval.jsonl`, md5 `bafd84e7…`, source md5 `eefc38d1…`): **147 / 147** eval
clips carry a speed band, buckets 30/50/100/120 km/h = **49 / 52 / 39 / 7**, one clamped above
120 km/h, entropy 1.7757 bits (train: 1.7555).


```
PYTHONPATH=/home/nvidia/refcv6_v2/stack:/home/nvidia/refcv6_v2/taniteval  OMP_NUM_THREADS=8
refc_v3_train.py --arm hier --size small --trunk timm --trunk-name resnet101.a1_in1k
  --trunk-mode shared --trunk-fuse concat1x1 --ego-history --no-strategic --image-hw 416 1024
  --sampler ddim --f1-random-t --f2-dd-step --f3-per-layer --f4-adaln --f5-focal
  --f5-emitting-conf --f6-w-u0-zero --f9-assert-vocab
  --anchors /home/nvidia/data/anchors/refc_anchors_6s_v0cond_alat_117.pt --anchor-v0-conditioned --n-anchors 117
  --v2-cache /home/nvidia/data/refcv6-b1-416x1024-train
  --v7-labels /home/nvidia/data/v8labels/labels/s2_labels_v8_train.jsonl.gz --nav-from-v7
  --eval-cache /home/nvidia/data/refcv6-b1-416x1024-eval139
  --eval-labels /home/nvidia/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz --eval-every 500 --eval-batches 8
  --tac-decoder-v6 --w-tac-v6 1.0 --tac-decoder-d-bev 96 --graft-behaviour-sel
  --max-speed-input-v6 --speed-max-sidecar-v6 /home/nvidia/data/refcv6_speed_max_v8_train.jsonl
  --speed-max-sidecar-v6-eval /home/nvidia/data/refcv6_speed_max_v8_eval.jsonl
  --agents head --agent-join /home/nvidia/percprobe/raw/b1train_agents.jsonl.xz
  --agent-rig-camera extrinsics --agent-rig-extrinsics /home/nvidia/data/refcv6_train_eval139_extrinsics.json
  --agent-cls-weight b1 --w-agent 1.0
  --w-map 1.0 --map-gt-root /home/nvidia/data/sam3_corpus
  --w-box3d 1.0 --join3d /home/nvidia/data/join3d/b1train_agents_3d.jsonl.xz --bev-coupling
  --equalize-bottom-rows 43 --opt dd --lr 1e-4 --warmup 2000 --seed 0 --u8-batches
  --save-every 500 --log-every 50 --batch <B> --steps <ceil(805,680 / B)> --conflict-every <N>
  --out /home/nvidia/refcv6_v2/runs/refcv6-r101-s0
```

⛔ **The in-run eval needs joins that ALSO cover the eval split — found by reading the eval
setup, not by a crash.** The trainer builds its eval join reader from the SAME `--agent-join`,
and `enable_agent_join` REFUSES a join covering zero of a split's episodes. The B1 train joins
hold no eval clip, so the line above would refuse at startup. `code/run_refcv6.sh` therefore
takes the joins and the eval block as parameters (`EVAL=1 AGENT_JOIN=… JOIN3D=…`), off by
default; a train + eval join — which needs its own registered basename in
`agent_slots.JOIN_CORPUS_LINES` and its own digest sidecar — turns it on.

✅ **Built the same morning, so the launch runs with `EVAL=1`:** the train + eval joins on Thor.
2-D `joins/b1_train_plus_eval_agents.jsonl.xz` = `cat` of the train join and the eval join
(the reconstruct-mode eval build is byte-identical to the banked v2ep one, md5 `3ddb42ec…`):
**875,657** rows = 849,263 + 26,394, md5 `0c31a3a6…` — the digest the 2026-09-07 join-repro audit
first recorded — with a `join_meta.declare` sidecar that `join_meta.verify` accepts; its first
317,028,572 bytes are md5 `1c985e6d…`, i.e. exactly the train join the smoke loaded. Its basename is
already on the B1 line in `JOIN_CORPUS_LINES`. 3-D `join3d/b1_train_plus_eval_agents_3d.jsonl.xz` =
the new train 3-D join (**849,263** lines, all controls passed, md5 `e39d5351…`) + the banked eval
3-D join: **875,657** lines, md5 `401c9c31…`.

**The train 3-D join's controls, MEASURED:** C1 / C2 / C5 / C6 **0 failures over all 849,263
lines**; C9 **849,263 / 849,263** lines on the exact time (4,402 clips by registration, 25 by the
camera grid), max |Δt| **0.0**; C3 road plane: median ground-standing base **+0.0084 m** over
21,952,289 agents (tolerance 0.20 m); content re-read: 0 mismatches; z/h on **28,053,187 /
28,053,187** agents.

`<B>`, the step count and `--conflict-every` are set by the post-reboot smoke: `cuda_max_mem_gb`
per batch (the only admissible memory probe on Thor) and s/step with the conflict probe at every
step vs amortised. The budget is fixed in **samples** — the prereg's `full` = 40,284 × 20 =
**805,680** windows ≈ 1.08 epochs of 746,946 — so the batch decision cannot move it.

## ⛔ Blocker — ✅ RESOLVED 2026-09-23: no reboot was needed (§7, `CORR-2026-09-23-THOR-REBOOT-AND-OOM`)

Thor MemAvailable **13.4 GB of 128.8 GB** with no process accounting for it (largest RSS 1.7 GB;
Slab 1.2 GB; Shmem 0.2 GB) after 38.4 days of uptime — kernel-side GPU memory that only a reboot
frees. There is no sudo on the box. The smoke OOMs on it; the PI has been asked for the reboot.

## 6. One more silent path, closed before the launch (commit after `LAUNCH_READINESS_FIXES` landed)

* ⛔ `open_join3d` returns `None` for a path that does not exist — right for its default caller —
  and `train()` handed that straight to `enable_join3d`. A mistyped `--join3d`, or a 3-D build
  that had not finished writing, therefore trained the **2-D rung** (`zh_mask` all-False,
  `box3d_n_z 0`) for the whole run while `argv` named a 3-D join; so did a join that exists but
  covers none of the corpus.
* ✅ `require_join3d` — called at BOTH join3d sites — refuses a missing path and a zero-coverage
  TRAIN join by name; on the EVAL split zero coverage warns that its z/h terms read n = 0.
* 🧪 `stack/tests/test_join3d_asked_for_is_required.py` — 5 tests against the real `AgentJoin3D`
  on a file in the builder's line format, with a GREEN control. **Mutation 3/3**
  (`raw/mutation_proof_join3d_required.json`, `code/mutate_join3d_required.py`).

## 7. The first real steps on Thor — three more defects, and the number that decides the budget

**Evidence class:** MEASURED (ours, Thor, 2026-09-23, the launch line with in-run eval and the train +
eval joins). Raw per-smoke summaries: `raw/smoke_summaries_2026-09-23.json`; decision rules fixed
before any smoke ran: `SMOKE_DECISION_RULES.md`.

| smoke | batch | trunk | conflict probe | steps | peak `cuda_max_mem_gb` | s/step (from step 5) | eval rows | outcome |
|---|---|---|---|---|---|---|---|---|
| `b4_ce1` | 4 | unchunked | every step | 0 | — | — | 0 | **OOM-killed** in the first step (`NVRM … ADDR_SYSMEM`) |
| `ck_b4_ce1` | 4 | chunk 1 + frozen BN | every step | 0 | — | — | 0 | **`NameError: args`** in `compute_losses_v3` |
| `ck_b4_ce1_fix` / `plain_b1_cd` | 4 / 1 | chunk 1 / unchunked | every step | 0 | — | — | 0 | the detector's controls **refused** (NaN) — checkpointing ruled out as the cause |
| `ck_b4_nocd` | 4 | chunk 1 + frozen BN | off | 40 | 6.68 | 8.61 | 2 | trains; loss 196.0 → 139.4 |
| `S1_b4_ce1` | 4 | chunk 1 + frozen BN | every step | 24* | 6.87 | 19.27 | 1 | trains; controls read exactly; *stopped after its step-20 eval |
| `S3_b8_ce10_ck8` | 8 | chunk 8 + frozen BN | every 10th | 30 | 18.03 | 18.16 (incl. eval) | 2 | trains; loss 196.6 → 125.7 |

**The three defects, each fixed and mutation-proven:**

1. ⛔ **Unchunked resnet101 at 416 × 1024 needs ~22 GB PER SAMPLE** — `--arm hier` sends all 8 window
   steps × 3 frames through the trunk (the trainer's own help, MEASURED 2026-09-18) — so batch 4 OOMs
   on Thor. `--trunk-chunk-ckpt` fixes memory (batch 8 at 18 GB) and REQUIRES `--trunk-frozen-bn`
   (BatchNorm pinned to ImageNet statistics). `run_refcv6.sh` takes it as `TRUNK_CHUNK`.
2. ⛔ **`compute_losses_v3` read `args`, which it does not have** (`--map-lift-valid-mask`,
   `--box3d-visible-filter`, both from the 2026-09-23 perception fix). The first map loss on real data
   died with `NameError`; no test reached either line (the fake branch has no lift). The knobs now
   travel on the model. ⭐ A sweep of every `LOAD_GLOBAL` in the 82 modules on the refcv6 import path
   found exactly these two; `stack/tests/test_no_unresolved_globals.py` keeps it at zero (positive
   control plants the defect; **mutation 2/2**).
3. ⛔ **The conflict detector refused to START on refcv6**: `cos(g,g)` read NaN. MEASURED with the real
   `train()` (`code/diag_trunk_reach.py`): on step 1 the trajectory loss reached **all 316 trunk
   tensors and every gradient was exactly 0**; after ONE optimizer update **all 316 were non-zero**.
   Cause: `control_head` is zero-initialised by design (the first refinement pass is the identity).
   A 0/0 control is UNDEFINED, not missed — the detector now defers while the plan gradient is
   exactly zero and REFUSES if that lasts past `max_deferred_steps` (200), because a real detach is
   also exactly zero, forever. `stack/tests/test_conflict_controls_defer_on_zero_init.py`;
   **mutation 4/4** (no deferral; no bound; NaN deferred; the trainer measuring on a deferred step).
   ⭐ **So refcv6 IS trained end to end**: every loss term reaches the trunk from step 2.

**What the in-run eval now carries:** 84 metrics per eval row, with the eval split's box z/h
supervised (`eval_box3d_n_h` 19–21) — the long-open *"the eval path supervises neither z nor h"*
question is answered by the train + eval 3-D join.

**Conflict probe cadence (rule fixed before the data):** every step costs **+124 %** (19.27 vs 8.61
s/step at batch 4) → above the 25 % threshold → **every 10th step**, stamped in `config.json`.

⛔ **THE BUDGET IS A PI DECISION, and this is the number that makes it one:** about **0.44–0.47
samples/s** at batch 8. The pre-registered `full` budget (805,680 windows) is **~20 days** of Thor;
the pre-registered `cut` (12,000 steps × 20 = 240,000 windows) is **~6 days**. Thor's GPU draws
24–31 W during steps while ONE CPU core sits at 100 % — the next efficiency levers (bf16 autocast,
CPU-side work in the step) are measured next, not assumed.

**Thor memory, corrected:** the "~105 GB leak that only a reboot frees" is the GPU allocations of
processes that were KILLED — released by the driver with a delay (MEASURED: 19 → 93 GB within ~40 min
after an OOM-kill; a normally-exiting smoke released at once, 102 GB free after S3). No reboot needed.

## 8. Faster first (the PI's choice, 2026-09-23): 1.56x from two backbone levers, measured

**Evidence class:** MEASURED (ours, Thor). Profile: `raw/profile_S3_fp32_by_cuda.txt` /
`_by_cpu.txt` (`code/profile_step.py`, the real `train()`, 2 recorded steps after 4 warm-up).
Smokes: `raw/speed_smokes_2026-09-23.json`.

**Where the time went (fp32, batch 8, chunk 8):** the step is **GPU-bound** — the CPU core at 100 %
is waiting on a full command buffer (46 % of CPU time), not working. About **two thirds of GPU time
is memory-bound elementwise work** in the backbone (BatchNorm fwd+bwd ~23 %, ReLU ~17 %, residual
adds ~14 %) plus **~12 % NCHW↔NHWC layout conversions** cuDNN inserts around every convolution;
convolution math itself is ~39 %.

| smoke | batch | trunk | levers | s/step | peak `cuda_max_mem_gb` | verdict |
|---|---|---|---|---|---|---|
| S3 | 8 | chunk 8 + frozen BN | fp32 | **17.99** (steps 3→30) | 18.03 | baseline |
| **S4** | 8 | chunk 8 + frozen BN | **bf16 + channels_last** | **11.56** (steps 4→30) | **13.17** | ⭐ **1.56× faster**, −27 % memory |
| S5 | 8 | chunk 24 + frozen BN | bf16 + channels_last | 11.67 (4→14) | 21.32 | bigger chunks: no gain |
| S6 | 8 | chunk 8 + frozen BN | bf16 + channels_last + cuDNN autotune | 11.24 (4→14) | 13.82 | autotune: no gain → stays opt-in |
| S7 | 4 | **no checkpointing** + frozen BN | bf16 + channels_last | 4.90 (4→16) = 1.225 s/sample | **57.24** | ~3 % per sample for 4.5× the memory: not worth it |

S3/S4 windows both contain one in-run eval and the every-10th-step conflict readings. The losses
track step for step (S3 196.61 / 232.00 / 188.51 …, S4 195.84 / 231.46 / 187.91 …): the levers
change the arithmetic precision of the BACKBONE only, whose outputs return to float32 before the
fusion, the decoders, the trajectory integration and every loss.

**The levers** (`--trunk-bf16`, `--trunk-channels-last`, opt-in `--cudnn-benchmark`; launch-script
knobs `TRUNK_BF16` / `TRUNK_CL` / `CUDNN_BENCH`): real config fields, passed by `build_encoder`,
recorded in the trunk's `memory_levers`, stamped in `config.json`, refused for the `refc` trunk.
`stack/tests/test_trunk_speed_levers.py` — 7 tests, the key one a forward hook that must SEE a
bfloat16 activation in the stem; **mutation 5/5** (autocast ignoring the flag, NHWC removed,
outputs left in bf16, the trainer not pinning the flag, `build_encoder` dropping it).

**The launch configuration now:** batch 8, chunk 8, frozen BN, bf16 + channels_last, conflict probe
every 10th step, in-run eval every 500 steps ⇒ ~0.73–0.76 samples/s ⇒ pre-registered `cut`
(240,000 windows, 30,000 steps) ≈ **3.8 days**, `full` (805,680 windows, 100,750 steps) ≈ **12.6 days**.

## 9. Faster still (the PI's second choice, 2026-09-23): the frozen BatchNorms folded into their convs — 1.27× more, 1.98× over fp32

**Evidence class:** MEASURED (ours — Thor, plus the dev box where named). Smoke S8; probes
`code/fold_probe.py`, `code/fold_precision.py`, `code/fold_bias_mechanism.py`,
`code/fold_bias_speed.py`; the identical smoke read `code/step_budget.py` →
`raw/step_budget_S3_S4_S8.json`.

**The lever (`--trunk-fold-bn`, launch knob `TRUNK_FOLD_BN=1`).** Chunked checkpointing already
REQUIRES BatchNorm frozen at its ImageNet statistics (§7), and a frozen BN is a fixed per-channel
affine. `BN(conv(x; W)) = conv(x; s·W) + (β − μ·s)` with `s = γ/√(σ²+ε)` is the same function, so
the conv computes it directly and the BN becomes the identity — no separate BN pass forward or
backward, and no activation saved for BN's own backward. γ and β still train (through `s` and the
bias). Only the two instances' `forward` change, so the module tree and every `state_dict` key are
identical: a folded run's checkpoint loads into an unfolded model and back. Refused without
`--trunk-frozen-bn` and for the `refc` trunk. resnet101: **104 conv/BN pairs folded** (MEASURED on
Thor; resnet18 in CI: 20).

**Is it the same function?** Against a strict-fp32 reference (unfolded, TF32 off), resnet101
pretrained, 416 × 1024, relative error per feature level:

| arm | level 1 | level 2 | reads |
|---|---|---|---|
| folded, strict fp32 | 1.4e-6 | 4.2e-6 | exact up to rounding |
| unfolded, TF32 (cuDNN's fp32 default) | 0.19 % | 0.61 % | — |
| folded, TF32 | 0.18 % | 0.54 % | no worse |
| unfolded, bf16 + NHWC (§8, the launch precision) | 1.85 % | 4.72 % | — |
| **folded, bf16 + NHWC** | **2.10 %** | **5.81 %** | a little worse — see below |

⚠️ **The bf16 cost is real and its mechanism is measured, not argued.** Under autocast the conv
receives its folded bias in bf16, so each channel's whole offset carries ONE rounding — an error
coherent over the feature map. The same fold with its bias added in fp32 after a bias-free conv
reads **1.76 % / 4.66 %** — better than not folding (dev box: A/B there 1.83 % / 4.80 % and
2.04 % / 5.70 %, reproducing Thor). But that variant costs an extra pass: backbone fwd+bwd
**0.318 s** against the shipped fold's **0.189 s** and the unfolded **0.283 s** — slower than not
folding. Its adoption bar, committed before running (within 5 % of the shipped fold), failed ⇒
the shipped fold stays; the precision gain is recorded as priced, not taken.

**Does training notice?** Same seed, same data order, 30 steps, relative loss difference per step:

| pair | max | mean | reads |
|---|---|---|---|
| S4 vs S3 (bf16 + NHWC alone) | 0.66 % | 0.35 % | §8's lever |
| **S8 vs S4 (the fold)** | 1.06 % | 0.36 % | — |
| **S8 vs S3 (both levers vs fp32)** | 0.70 % | **0.22 %** | no further from fp32 than S4 is |

In-run eval (2 batches at steps 20 and 30, 46 metrics): median relative deviation 0.09–0.10 % for
S8 vs S4 against 0.03–0.04 % for S4 vs S3. The largest (agent yaw 0.848 → 0.679 at step 30; box
height ~8 %) are means over **21 matched objects**, where one object moves the mean; the per-step
TRAINING agent terms track within ~1 % (agent presence 0.1861 / 0.1865 / 0.1858 for S3 / S4 / S8).
⛔ **No replicate arm was run**, so this rig's run-to-run floor is NOT measured (`H-ESTIM-SEED-1`).
The claim is "tracks within bf16's own deviation from fp32 on the training loss", not "identical".

**Speed — every smoke read the same way** (`code/step_budget.py`: plain step = median delta over
steps with no conflict reading and no preceding eval; conflict and eval costs as extras over it;
launch projection = plain + conflict/10 + eval-per-batch × 8/500):

| smoke | levers | plain step (median, n=25) | conflict reading (+s, n=1) | smoke eval (+s, 2 batches) | **projected launch s/step** | samples/s | peak `cuda_max_mem_gb` |
|---|---|---|---|---|---|---|---|
| S3 | fp32 | 15.8 | 21.3 | 15.6 | 18.06 | 0.443 | 18.03 |
| S4 | bf16 + NHWC | 10.1 | 13.9 | 11.4 | 11.58 | 0.691 | 13.17 |
| **S8** | **bf16 + NHWC + fold** | **8.1** | **9.4** | **10.3** | **9.12** | **0.877** | **11.84** |

Backbone alone (8 images, fwd+bwd, `raw/fold_probe_thor_2026-09-23.json`): fp32 0.498 → 0.453 s
(1.10×), 8.81 → 4.99 GiB; bf16 + NHWC 0.328 → 0.207 s (**1.58×**), 5.04 → 3.14 GiB.

⇒ **The launch configuration now: batch 8, chunk 8, frozen + FOLDED BN, bf16 + NHWC, conflict
probe every 10th step, eval every 500 ⇒ 0.877 samples/s ⇒ `cut` (240,000 windows) ≈ 3.2 days,
`full` (805,680) ≈ 10.6 days.**

⚠️ **CORRECTION to §8's budget line.** §8 quoted 0.73–0.76 samples/s for S4 (`cut` ≈ 3.8 d, `full`
≈ 12.6 d). The same smoke read with the identical method above gives **0.691 samples/s** (`cut`
≈ 4.0 d, `full` ≈ 13.5 d) — §8's derivation was not banked with its number, so it cannot be
re-checked; `code/step_budget.py` now banks the derivation and applies it to every smoke.

**The record now carries the fact, not only the flag.** `config.json` gains `trunk_memory_levers`,
read off the BUILT trunk (`bn_folded`, `bn_pinned`, `chunk_ckpt`, `bf16`, `channels_last`) beside
the seam stamp, which records only what was asked for.

**Tests.** `stack/tests/test_trunk_speed_levers.py` +8: the folded trunk is the same function with
the same gradients (relative-norm tolerance 1e-5 — MEASURED 2.1e-6 for the correct fold against
2.2e-4 for the subtlest wrong one, eps dropped: `raw/fold_margin_cpu_2026-09-23.json`); the BN passes its input through untouched; every
`state_dict` key survives and loads into an unfolded trunk; a training BN and the `refc` trunk are
refused, the refusal naming the flag; the flag reaches the BUILT trunk; and a real `train()` on
the synthetic rig writes `trunk_memory_levers.bn_folded == 20`. **Mutation 14/14**
(`raw/mutation_proof_trunk_fold_bn.json`) — including the fold STAMPED but not applied, eps
dropped, the shortcut BNs left unfolded, and the record reading the wrong module.

## 10. Each distinct frame through the backbone ONCE (`--trunk-dedup-frames`), and batch 16 by the pre-registered rule

**Evidence class:** MEASURED (ours, Thor). Profile of the §9 configuration
(`raw/profile_S8_fold_by_cuda.txt`); smokes S9 / S10 (`raw/step_budget_S9_S10.json`, read with
`code/step_budget.py`); same-seed tracking `raw/tracking_S3_S4_S8_S9.json` (`code/smoke_tracking.py`).

**What the §9 profile still said.** With BN folded, ~60 % of GPU time was still memory-bound
elementwise work in the backbone (`aten::add_` 31.7 %, ReLU 9.2 % + its backward 7.3 %, copies
6.0 %) against 31 % in convolutions — the backbone simply had too much to do. Reading the data
path said why: `--arm hier` sends EVERY window row's 3-frame stack through the trunk
(`RefCModel.forward` → `encoder.forward_features(frames.reshape(b * w, ...))`), and a D-015 row
stacks raw frames (j, j+1, j+2) oldest → newest (`tanitad/data/v2_dataset.py::_decode_stacked`).
Consecutive rows share two frames, so each sample made **24 backbone passes for 10 distinct
frames**. Every row matters (the hierarchy hook reads the pooled feature of all 8), so rows
cannot be skipped — but no frame needs computing twice.

**The lever.** Compute each distinct frame once and gather it into every slot that uses it. With
BN frozen a frame's features depend on that frame alone, so this is the same function; the
gather's backward SUMS each frame's gradient over its slots, which is what the separate passes'
weight gradients summed to. ⛔ **The overlap is VERIFIED per batch, never assumed**: row i+1
reuses row i's frames only where the data are exactly equal, so a window boundary, another clip or
any input without the D-015 structure computes all three frames — the lever is exact on every
input and merely does nothing where there is nothing to share. Requires `--trunk-frozen-bn`;
"shared" mode with K ≥ 2 only; refused for the `refc` trunk. Launch knob `TRUNK_DEDUP=1`.

**On the real data** (S9 = §9's S8 + dedup, batch 8), the run log proves it every step:
`trunk_frame_slots 216 / trunk_frames_computed 104` — the window's 192 slots from **80 frames
(8 samples × 10, every window's overlap verified)** plus the 24 future frames, which share
nothing — and 648 / 312 on conflict-reading steps. `config.json`'s `trunk_memory_levers`, read off
the built trunk: `chunk_ckpt 8, frozen_bn, bn_pinned 104, relu_out_of_place 100, bf16,
channels_last, bn_folded 104, dedup_frames`.

**Same function, in practice too.** Step 1's loss is identical (196.00 / 196.00). Through step 23,
S9 stays within **0.07 %** of S8 — tighter than bf16 alone (0.2–0.66 % from fp32) or the fold
(0.1–0.9 %). From step 24 the difference grows (max 2.17 % at step 29), which is how any two
trajectories drift apart through the optimizer: S8 vs S4 reaches 1.05 % at the same step. In-run
eval median deviation from S8: 0.005 % (step 20) and 0.02 % (step 30); eval loss 153.63 → 153.61
and 107.23 → 107.76. ⛔ No replicate arm → the rig's run-to-run floor is still not measured.

**Speed** (identical read, `code/step_budget.py`):

| smoke | batch | levers | plain step | conflict reading | smoke eval (2 batches) | **projected launch s/step** | **samples/s** | peak `cuda_max_mem_gb` |
|---|---|---|---|---|---|---|---|---|
| S8 (§9) | 8 | bf16 + NHWC + fold | 8.1 | +9.4 | +10.3 | 9.12 | 0.877 | 11.84 |
| **S9** | 8 | + **dedup** | **4.4** | +4.9 | +6.6 | **4.94** | **1.618** | 11.37 |
| **S10** | **16** | + dedup | 8.0 | +8.7 | +18.1 | 9.02 | **1.775** | 21.23 |

**The batch, by the rule fixed before any smoke** (`SMOKE_DECISION_RULES.md`: the largest of
{4, 8, 12, 16} whose peak fits in 60 % of MemAvailable, and among those the highest samples/s).
§7 smoked only batches 4 and 8; with the step now 3.7× cheaper the rule was run to the end. Batch
16 peaks at 21.2 GB — far inside 60 % of Thor's ~79 GB available — and is 9.7 % faster per
sample than batch 8 (the conflict reading and the per-step overheads amortise over twice the
samples). ⇒ **batch 16.** Batch 12 was not smoked; step time is close to linear in the batch, so
it would fall between. The budget stays fixed in SAMPLES, so the batch does not move it: `full` =
ceil(805,680 / 16) = 50,355 → **50,400 steps**; `cut` = 240,000 / 16 = **15,000 steps**.

**Tests.** `stack/tests/test_trunk_dedup_frames.py` — 9 tests: every slot gets the features the
plain backbone computes; `forward_features` and the parameter GRADIENTS match (fusion randomised,
so every stack position matters); a hook on the stem counts exactly B × (W + K − 1) frames; a
batch with no overlap computes every slot; one corrupted overlapping frame is not reused; the
refusals; the composition with chunking, bf16, NHWC and the fold; the flag reaches the BUILT trunk
and the stamp; and a real `train()` on D-015-structured synthetic windows logs fewer computed
frames than slots, per row, with the plain run's loss. **Mutation 13/13**
(`raw/mutation_proof_trunk_dedup_frames.json`) — including the dedup stamped but not applied, the
slot map off by one, the overlap check that always says "same", and a log counter never reset.

### 10b. The backbone through `torch.compile` (`--trunk-compile`) — +33 %, and closer to fp32

**Evidence class:** MEASURED (ours, Thor). `code/compile_probe.py` → `raw/compile_probe_thor_2026-09-23.json`;
`code/compile_precision.py` → `raw/compile_precision_thor_2026-09-23.json`; smokes S11 / S11b
(`raw/S11_compile_donated_buffer_failure.txt`, `raw/step_budget_S10_S11b.json`,
`raw/tracking_S10_S11b.json`).

**The micro-benchmark** (resnet101, 416 × 1024, bf16 + NHWC, frozen + folded BN, chunk 8, 24
images, fwd+bwd): eager 0.826 s → compiled **0.539 s (1.53×)**, compile 32 s once, memory unchanged.

**⛔ It FAILED the precision bar committed with it**: relative difference vs EAGER ≤ 1 %, read
2.05 % / 5.46 % ⇒ declined as registered. That bar compared two bf16 computations with each other,
and bf16 alone sits ~2 % / ~5 % from fp32, so it could not tell "compile is wrong" from "two bf16
roundings differ". **The cheapest discriminating test, pre-registered as a NEW arm before it ran**:
against strict fp32 (TF32 off), compile is admissible iff its error ≤ 1.10 × the eager bf16 path's
at both levels. Read: **compiled 1.75 % / 4.67 % vs eager 2.08 % / 5.74 % (ratio 0.84 / 0.81)** —
the compiled backbone is CLOSER to fp32 (Inductor keeps fused intermediates in fp32 rather than
rounding to bf16 between operations). ⇒ admissible.

**⛔ The first real step crashed, and the fix is the documented switch.** S11 (batch 16, every lever
+ compile) died at step 1: *"This backward function was compiled with non-empty donated buffers
which requires create_graph=False and retain_graph=False"*. The trainer backpropagates through one
graph more than once — the conflict detector's per-term gradients use `retain_graph=True` before
the step's own backward — and AOTAutograd's donated buffers forbid that. With compile on, the trunk
now sets `torch._functorch.config.donated_buffer = False` and records it
(`trunk_memory_levers.compile_donated_buffer: false`). The CPU backends used in CI do not donate,
so the test pins the switch and runs the trainer's retain-graph pattern; **S11b is the proof**.

| smoke | batch | levers | plain step | conflict reading | **projected launch s/step** | **samples/s** | peak `cuda_max_mem_gb` |
|---|---|---|---|---|---|---|---|
| S10 | 16 | bf16 + NHWC + fold + dedup | 8.0 | +8.7 | 9.02 | 1.775 | 21.23 |
| **S11b** | 16 | + **compile** | **6.0** | +6.7 | **6.79** | **2.358** | 21.24 |

Same seed and data, S11b vs S10, per-step training loss: **max 0.57 %, mean 0.16 %** — smaller than
bf16's own shift (0.66 % / 0.35 %) or the fold's (1.06 % / 0.36 %); in-run eval median deviation
0.07 % (eval loss 140.87 → 140.66, 99.41 → 99.34). Dedup at batch 16: 432 slots / 208 frames per
step. `config.json` of S11b records, read off the built trunk: `chunk_ckpt 8, frozen_bn,
bn_pinned 104, relu_out_of_place 100, bf16, channels_last, bn_folded 104, dedup_frames,
compile_donated_buffer false, compile inductor`.

**Tests.** `stack/tests/test_trunk_compile.py` — 7 tests (`aot_eager` on CPU: the dev box has no
Triton): OFF is the eager network itself; same outputs and gradients; the compiled callable is what
RUNS (a counting wrapper sees every chunk); state_dict keys unchanged and loadable both ways;
composes with chunking, bf16, NHWC, fold and dedup; the flag reaches the BUILT trunk and the stamp
and is refused for the `refc` trunk; donated buffers off. **Mutation 9/9**
(`raw/mutation_proof_trunk_compile.json`). The fold proof re-run on this commit's files: **14/14**
(`raw/mutation_proof_trunk_fold_bn_on_F.json`, two anchors moved with the new flags).

### 10c. The launch — the PI's decisions, 2026-09-23 evening

Asked with the numbers above, the PI chose, verbatim: **"Full, ~4.0 days (Recommended)"** and
**"Keep every 10th (Recommended)"** — the pre-registered budget and the pre-registered conflict
cadence, so the launch carries **no deviation** from `SMOKE_DECISION_RULES.md`.

| | |
|---|---|
| batch | **16** (the rule: largest that fits, highest samples/s) |
| steps | **50,400** = ceil(805,680 / 16) rounded up to 50 — the pre-registered `full` |
| trunk | resnet101.a1_in1k, 416 × 1024, chunk 8, frozen + folded BN, bf16 + NHWC, dedup, compile |
| conflict probe | every 10th step (pre-registered) |
| in-run eval | every 500 steps, 8 batches, train + eval joins |
| projected | **6.79 s/step ⇒ 2.36 samples/s ⇒ ≈ 3.96 days** (fp32 this morning: 18.06 s/step at batch 8 ⇒ ≈ 21 days) |

## 11. The launch — and the conflict readings it was throwing away

**Evidence class:** MEASURED (ours, Thor). Banked under `raw/launch_2026-09-23/`: the manifest that
was run (`refcv6-r101-s0.env`), the dry-run command line and the LIVE one read from the trainer's
own `/proc/<pid>/cmdline`, the supervisor's start output, `config.json` as the run wrote it, and
its first metrics rows.

**What was launched.** 2026-09-23 **20:42 Europe/Berlin** (18:42:47Z), commit **`284393c`** shipped
by `ship_commit_to_thor.py` (**2,200 files, 0 missing, 0 differing** by md5 on Thor; `tanitad`,
`taniteval` and the trainer import from that tree, asserted by `__file__`), under
`sup_refcv6.sh refcv6-r101-s0` (supervisor pid 3336492, trainer pid 3336502, lock
`runs.d/refcv6-r101-s0.lock`, `MAX_RELAUNCH` 4). The live command line equals the dry run
flag for flag: batch **16**, **50,400** steps, 416 × 1024, resnet101, chunk 8, frozen + folded BN,
bf16 + NHWC, dedup, compile, conflict probe every 10th step, eval every 500 × 8 batches, the v8
labels with nav from v7, ego history, the SAM3 map, the agent and 3-D box joins (train + eval),
DDIM diffusion. `config.json` records, read off the BUILT trunk, `chunk_ckpt 8, bn_pinned 104,
bn_folded 104, bf16, channels_last, dedup_frames, compile inductor, compile_donated_buffer false`,
a fresh data order (`resumed_from: null`), seed 0.

| step | loss | elapsed s | s/step (per 50) | peak `cuda_max_mem_gb` | frames computed / slots |
|---|---|---|---|---|---|
| 50 | 83.17 | 347.4 | — | 21.236 | 10,400 / 21,600 |
| 100 | 62.58 | 667.9 | 6.41 | 21.237 | 10,400 / 21,600 |
| 150 | 58.00 | 987.9 | 6.40 | 21.237 | 10,400 / 21,600 |
| 200 | 56.40 | 1,308.7 | 6.42 | 21.237 | 10,400 / 21,600 |

⇒ **6.4 s/step** on the real run (the §10 projection was 6.79 incl. eval) ⇒ ≈ **3.8 days** with
the in-run evals ⇒ finish ≈ **2026-09-27** (Berlin).

⛔ **THE DEFECT THE SMOKES COULD NOT SHOW.** The rows at steps 50, 100, 150 and 200 carried **no
`cd_*` key**. The detector measures on pre-increment steps divisible by `--conflict-every` —
logged steps **10k+1** (the smokes show readings at 11 and 21) — and its reading was merged ONLY
into a log row; `_cd_row` is cleared every step, deliberately, so a stale reading cannot be
re-logged. With `--log-every 50` a reading step is never a log step: **every reading was computed —
about 10 % of the run's time — and discarded.** The smokes logged every step, which is exactly
why they never showed it.
**Class:** a smoke that differs from the launch in an OPERATIONAL knob (the log cadence) cannot
see a defect that only the launch's value exhibits — the same family as "a check that shares the
defect it checks for". ⇒ smoke the launch line **as launched**, log cadence included.

**The fix (this commit):** a reading that falls on a step the log skips is written as a ROW OF ITS
OWN, under its own step (`{"step": s, "cd_*": ...}` — no loss keys, so every reader that selects
training rows by their `loss` is unaffected). `stack/tests/test_conflict_readings_are_logged.py`
runs the real `train()` with the detector forced on (`--conflict-every 2 --log-every 5`, six steps)
and requires readings on steps **1, 3, 5** — 1 and 3 as own rows, 5 inside its log row — with the
set written as a literal; **mutation 2/2** (`raw/mutation_proof_conflict_log.json`: the branch
removed; the own row under the wrong step).

**Why not simply `--log-every 1`, as the smokes ran?** It would record the readings with no code
change, but ESTIMATED from two runs (not a controlled measurement): the launch at `--log-every 50`
reads 6.40 s/step with the probe, while S11b's per-step-logging figures predict 6.67 — per-step
logging (a row of ~90 scalars, each a device sync) plausibly costs ~0.26 s/step, ~4 % or about four
hours of this run. The code fix records every reading at the 50-step cadence without that cost.

**Switching the live run to this commit** is a logging-only change and follows the supervisor's rule
7: ship the commit, point the manifest's `CODE` at it, and — right after a checkpoint and its eval —
kill the supervisor, then the trainer (explicit pids), then start a fresh supervisor, which resumes
from `ckpt.pt` with the data position. The switch step and the new pids are recorded in
`GOALS_AND_CLAIMS.md` when it happens.

**Monitoring:** two session crons on the dev box — a progress check at 08/12/16/20:13 Berlin and a
night check at 00/04:13 — each read both pids, the last training row, the supervisor's opaque
progress token and the latest in-run eval, and stay SILENT on a healthy run.

## 12. Two health flags at step 32,050, both false — and the two defects behind them (2026-09-26)

**What tripped.** The 08:13 Berlin watch refresh read the run at **step 32,050 / 50,400** with both
pids alive, 0 unplanned relaunches and the conflict readings current, but two of its health
criteria failed: `stderr_bytes` **112** (it had been 0) and `n_err` **50,400**. Neither was a
fault in the run. Each was a defect in how the run was being read, and one of them had been hiding
the stderr line for two days.

**The stderr line (MEASURED, read on Thor).** `train.stderr.log` is one line, 112 B, written by
the trainer's pid 3346338 at 2026-09-24 09:03:45 (Thor's clock is Berlin time: `date -u` read
06:54:53Z while `metrics.jsonl` showed an mtime of 08:54:12):

```
W0924 09:03:45.814000 3346338 torch/_inductor/utils.py:1953] [0/2] Not enough SMs to use max_autotune_gemm mode
```

It is Inductor's notice that Thor's 20 SMs are too few for max-autotune GEMM, a mode this run does
not use. `[0/2]` is frame 0's **third** compile of the backbone: the first split supervisor token
(below) sits at **step 6,571**, and the 50-step interval ending at 6,600 read **7.27 s/step**
against ~6.56 around it, so the recompile cost roughly **35 s, once**. The median plain pace (50-step
intervals with no in-run eval, from the run's own `elapsed_s`) moved from **6.552 s/step** over
3,000–6,500 (n = 63) to **6.606** over 6,600–9,000 (n = 44), a step of about +0.8 %. The slow drift
around it began before the recompile: 6.516 over 500–3,000 (n = 45), then 6.610 / 6.617 / 6.625 /
6.630 / 6.654 / 6.654 over the following windows up to step 32,100 (n = 54–72 each), +2.1 % in all.
Nothing here is worth a restart.

**The error count of 50,400.** `sup_refcv6.sh` counts tracebacks only once stderr is non-empty (a
`[ -s ]` guard), with `n_err="$(grep -Ec "$pat" "$ERRLOG" 2>/dev/null || echo 0)"`. On zero matches
`grep -c` prints `0` **and exits 1**, so `|| echo 0` prints a second `0`. From step 6,571 onward every
token was split over two lines (`…-50400-0` then `0-1ZZ`): at the time of reading the log held
**574 split tokens and 149 whole ones**. The watch parser split the log on whitespace and took the
last two fields of the first half, so it read the **step total** as the traceback count. The real
count is **0**, twice over: the supervisor's own grep (the first `0`) and, independently, the stderr
content read line by line (one line, no traceback or OOM). The same line hid a second defect: when
grep cannot read the log (exit 2) it prints nothing, so `|| echo 0` reported an unread log as
**zero errors**.

**Why it went unseen for two days.** No watch refresh ran between the step-1,150 build
(2026-09-23 evening) and this one. The session did not run in that window; the last messages
before the gap were both EvalFlyWheel agents stopping on the weekly usage limit. The first refresh
afterwards caught both flags.

**The fixes (this commit):**

1. `taniteval/tools/training_watch/build_watch_refcv6.py` matches the token **whole, across line
   breaks** (`TOKEN_RE`). It takes grep's own count, the first integer, since `|| echo 0` only
   appends one, and flags a split token on the page. It reads the stderr **content** in the same
   ssh call as its size. **Every line must be diagnosed by its exact text** in
   `FACTS["stderr_diagnosed"]`, with a dated diagnosis, or it counts as undiagnosed and fails a
   new stderr chip. A second instance of a known warning carries a new timestamp, so it is
   undiagnosed again. A pulled content shorter than the file fails as NOT READ, and tracebacks
   are also counted from the content itself (`n_err_client`). A `U` from the supervisor reads as
   NOT READ, never as zero.
2. `…/2026-09-10-refcv6-build/code/sup_refcv6.sh` is fixed for the **next** launch:
   `n_err="$(grep …)" || rc=$?`, and `U` when grep exits 2. The **live supervisor is not edited**,
   because bash reads a running script by byte offset. It keeps emitting split tokens, which the
   parser now reads correctly.
3. The two session crons now judge health on `stderr_undiagnosed == 0` and `stderr_read_ok`,
   `n_err == 0` **and** `n_err_client == 0`, and `token_ok`, **instead of** `stderr_bytes == 0`.
   That criterion would stay red for the rest of the run over one diagnosed line. A new stderr line
   still fails until it has been read and diagnosed.

**Proofs.** On a clean tree (git archive of `fe5872f`'s `taniteval/` plus the two changed files)
`test_training_watch_refcv6.py` gives **10 passed**. The **mutation proof is 9/9 RED**, restored,
with a green final run (`raw/mutation_proof_training_watch_refcv6_stderr.json`). The arms are the
three from commit I and six new ones. One of them restores the historical whitespace-split parser
**verbatim**, which reads the split token as 50,400 again. The others: the content ignored, the
read check dropped, a diagnosis matched by message instead of by exact line (a recurrence then
passes silently), the client count dropped, and an unread count read as zero. The supervisor fix
is proven on the script's **real** block, cut out by its anchors, over four stderr logs, with the
expected tokens written as literals. The old block splits the token on the live run's benign line
and reports an unreadable log as `0`; the fixed block emits one line and `U`. That is **8/8 cases**,
in `raw/sup_token_proof.json`, via `code/run_block.sh` and `code/sup_token_proof.py`.

**The in-run evals, for the record (T0, 128 fixed windows, never a driving claim):** at step 32,000
eval traj **0.586** (1.219 at step 1,000), 2 s goal error **3.59 m** (5.21), anchor accuracy
**0.5625** (0.055; chance 1/117 = 0.0085), map drivable IoU **0.677** (0.583). The all-in pace over the last
500 steps, one eval included, is **6.78 s/step**. Each in-run eval adds about 64 s: the 50-step
intervals that follow one read ~7.93–7.98 s/step against ~6.65. The finish is therefore
**≈ 2026-09-27 19:30 Berlin**, not the ~17:50 projected at step 1,150 from a window with no eval in it.

## 13. The PI's A16 switch — F3's cascade loss and the true label clock, from step 34,500 (2026-09-26)

**Why.** The A16 frozen-trunk audit (`…/2026-09-26-refcv6-frozen-trunk-audit/`, 92337fa) found two
defects in the running arm. The Master Mind confirmed the first independently.

- **F3's per-stage cascade loss had never run.** `RefCModel.forward` dropped `layer_u0_hat` and
  `layer_logits`, so the trainer's guard skipped the term silently:
  - 0 of 3,871 log rows carried `cascade`;
  - the stage-0–2 heads were bit-identical at steps 1k, 5k and 30k.
- **Every tactical label was read ~0.37 s early.** 10.6 % of the tactically supervised windows
  fell outside the true band.

**The decision.** The PI was asked with four options: finish then run a corrected arm; finish then
fine-tune; stop and resume with the fixes; restart clean. The PI chose, verbatim, **"Stop now,
resume with fixes"**.

**The fix, 82c2331.**
- The two keys pass through `RefCModel.forward`, and the F3 block now REFUSES instead of skipping.
- Labels are read at `grid_start + (t + w − 1 + n_stack − 1) · dt` on each clip's measured clock,
  via a new `--clip-clock-sidecar` and the `CLIP_CLOCK` knob in `run_refcv6.sh`.
- Tests on a clean tree:
  - `test_refcv6_f3_cascade_reaches_loss` runs the real `train()`; it FAILS on the unfixed tip and
    PASSES on the fix.
  - `test_refcv6_label_clock` has literal bands and the old formula as its regression arm.
  - The two new test files give 8 passed. The refcv6-related suites show 0 regressions against the
    tip: every other failure fails identically there.
- The sidecar, `raw/refcv6_clip_clock_sidecar.jsonl` in the audit package, was built on the dev box
  from each clip's 100 Hz egomotion log:
  - 4,483 clips clocked, 25 refused by the controls;
  - median dt 0.1006666 s, median grid start +0.1135 s;
  - keyed by sid, with no clip ids.

**The switch, MEASURED.** It was a Thor-side `setsid nohup` script, `switch_a16.sh`; its log is in
`raw/launch_2026-09-23/switch_a16/`. It followed the procedure that has worked before:
1. Shipped 82c2331 to `/home/nvidia/refcv6_run/82c2331a2f`: 2,670 files, md5 checked per file on
   Thor.
2. A preflight with Thor's own training interpreter: the new tree imports, the sidecar reads 4,483
   rows, and the pass-through is present.
3. Waited for the **step-34,500 checkpoint**: ckpt.pt written 13:28:31 Berlin, md5
   `3fbbde7470914b8bd55176e60fe399d3`.
4. Swapped the run's config file to v3 (md5 `b1b3c99165b118260b0c530f3a507191`). The diff is
   exactly `CODE` → the new tree, plus `CLIP_CLOCK`.
5. Killed the SUPERVISOR first by explicit pid (3346328). The trainer (3346338) ignored INT and was
   stopped by TERM; then its recorded children. Lock holders afterwards: none.
6. A fresh supervisor started from the new tree (3410715). It is the supervisor script fixed in
   b08f278, so its status tokens are no longer split.
7. The new trainer is pid 3410728:
   - its argv carries `--clip-clock-sidecar`;
   - it logs `resumed hier at step 34500` with its data position (`epoch 0, 34500 batches in`);
   - `config.json` `label_clock`: train 4,347 / 4,369 clips from the sidecar (13 fall back to
     pose dt, 9 to nominal), eval 136 / 139, raw offset +2 rows;
   - the **first row, step 34,550, carries `cascade` 3.097**;
   - stderr is unchanged.
Wall-clock lost: about 2 minutes of stop and restart, plus the load and compile. The steps between
34,500 and the kill are re-done on the fixed code, from the resumed data position.

**What the run is now, and how to quote it.**
- Checkpoints up to and including step 34,500 (the trainer's 5k/15k/20k/30k, the snapshots at
  10k/25k): *"F3 detach-only, F4 on the last layer only; tactical labels ~0.37 s early"*.
- Every checkpoint after it, the FINAL included: *"hybrid — F3 cascade loss and the true label clock
  from step 34,500"*.
- A difference between a pre-switch and a post-switch checkpoint mixes training time with the fix,
  and is never attributed to the fix alone.
- The cascade loss is part of the total loss, so the loss curve steps at the switch.
- The Training Watch now carries the switch as its third planned segment, and reads the first
  `cascade` row from the log itself.
