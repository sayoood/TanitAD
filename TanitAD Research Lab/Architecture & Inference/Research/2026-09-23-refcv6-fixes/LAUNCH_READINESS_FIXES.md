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

## ⛔ Blocker

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
