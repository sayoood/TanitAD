# E1.4 — running T1 on the v5f / v5.8f stack (design note)

**Date:** 2026-08-11 · **Scope:** make `taniteval/tools/t1_eval.py` (the
byte-close-validated T1 instrument) runnable on the w120 flagship stack without
touching its validated math. **Evidence class of everything below:**
`MEASURED (ours)` for the CPU checks named with their test, `HYPOTHESIS` for the
runtime estimate, and *nothing here is a T1 result* — the rows themselves come
out of the pod run this note describes.

---

## 1. The two gaps, and the approach chosen

| gap | before | after |
|---|---|---|
| **corpus** | `--corpus <dir of ep_*.pt>` (raw epcache) only. The w120 val split exists ONLY as a v2 compressed cache (`*.v2ep.pt`) — the raw epcache is ~697 GB/split and fits on no host in the fleet. | `--v2-val-cache DIR [DIR…]` + the geometry flags, read through the **same** providers `eval_flagship_v4` / `w7_roll_rerank` use. |
| **decoder** | a separate `--head` `UnicycleStepReadout` checkpoint. v5f/v5.8f carry their own metric decoder **inside the checkpoint** (`grounding.step['op']`). | `--grounding-readout`. |

**Chosen: approach (A)** — additive input paths inside `t1_eval.py` — over (B),
a shard-materialising adapter script. Why:

1. **(B) would have to write a second copy of the val split to disk.** At w120 a
   40-episode stride-1 shard set is tens of GB of decoded uint8; the whole reason
   the v2 cache exists is that this does not fit. It also creates a corpus with
   no parity record, which is exactly the class of artifact
   `parity.assert_v2_parity_cache` exists to refuse.
2. **(B) would be a fourth v2 decode path.** The repo already carries three
   (`v2_compressed.py`, `slice_v2_cache.py`, `v2_dataset.py`), and
   `build_v2_val_episodes`'s docstring names "adding a fourth" as how the
   rig-clean fix came to be applied to a function nobody's training path reached.
   (A) **imports** `build_v2_val_episodes` → `build_v2_providers` and adds no
   decode of its own.
3. **The sub-frame seam is only correct if it is the trainer's own.**
   `--v2-subframe 176x624` is resolved by `eval_flagship_v4.resolve_eval_frames`
   → `train_flagship_v4.resolve_v2_frames`, so this tool cannot spell the
   geometry a second way. A materialiser would have had to re-derive it.
4. **(A) leaves the validated path literally untouched** (§2), which was the
   binding constraint. (B) would have left it untouched too, but at the cost of
   1–3.

The adapter is `V2RawEp` — a **rename, not a decode**: `LazyV2Episode` already
exposes `.frames` / `.poses` / `.actions`, and `taniteval.data.RawEp` (what the
roll consumes) wants `.feats` / `.poses` / `.actions`. The lazy partial JPEG/PNG
decode stays inside the provider's own LRU-bounded proxy.

---

## 2. What stays byte-identical, and how that is enforced

`--corpus` + `--head` still routes to **`run_rollout`**, which is unchanged, as
are **`roll_closed`**, **`decode_open`**, the whole **`analyze`** half and
`--byte-check-dump`. The adapter lives in a separate function,
**`run_rollout_ext`**, selected only by `uses_ext_rollout(a)` (true iff
`--v2-val-cache` or `--grounding-readout` was passed).

Enforced by `stack/tests/test_t1_v2_adapter.py`:

* `test_legacy_roll_source_is_byte_identical` — md5 over the source of
  `decode_open` + `roll_closed` + `run_rollout`
  (`326b40f747c2c1c663f7e52a444b05df`). **If you edit those three, re-run the
  pod-side byte-close gate (`--analyze-only` over the original `dump_cl`
  reproducing `closed_loop_analysis.json`) and update the constant in the same
  change.**
* `test_legacy_roll_carries_no_adapter_token` — no adapter identifier appears
  inside them; their own episode source, decoder and stride-1 grid still do.
* `test_public_signatures_unchanged`, `test_dispatch_predicate_…` — the legacy
  argument combination can never reach the adapter.
* `test_byte_check_dump_mode_still_works` — the mode E1.4 validated through is
  exercised end to end via the CLI.

**The duplicated window loop is deliberate.** Parameterising the validated loop
would put the byte-close gate at risk for a cosmetic gain. The *math* is not
duplicated: in `--head` mode the adapter calls the same `roll_closed` /
`decode_open`; in `--grounding-readout` mode it calls
`metric_dynamics.decode_transitions` — the imported function, not a copy.

---

## 3. The one piece of new math (attack this first)

T1 means the predictor consumes **the decoder's own actions**. The grounding
readout emits a Δpose, not controls, so the loop needs the inverse of the corpus
action mint. `implied_controls` (t1_eval.py):

```
v        = ||(dx, dy)|| / dt          # the SAME speed definition `controls()`
                                      # (the verbatim analyze_cl port) scores with
accel    = (v - v_prev) / dt          # v_prev starts at the OBSERVED v0
yaw_rate = dyaw / dt
steer    = atan(wheelbase * yaw_rate / max(v, 0.3))    # physicalai.py:621 mint,
                                      # legacy const2p9; the clamp roll_closed uses
```

* It is an **inversion of `physicalai.signals_at` (physicalai.py:596-641)**, and
  it reuses `roll_closed`'s own `clamp_min(0.3)` — the two closed loops now
  differ only in *where the motion came from*.
* **Speed is the hypot, not `dx`.** Taking `dx` alone would silently discard the
  free decoder's lateral component — the very defect `UnicycleStepReadout` exists
  to remove — and hide it from the feedback path.
* The ego channel holds the observed `v0` for the whole roll: the trunk's own
  rollout contract (the canary and `w7_roll_rerank` hold it constant too).
  ⚠️ Stated limitation inherited from W3/W7, not introduced here.
* `ol` (T0) and `ha` (hold-action) use **none** of this — they consume recorded
  or held actions — which is why running the triplet in one pass is what makes
  the T0-vs-T1 gap attributable.

⭐ **Free cross-check:** with `--grounding-readout --with-t0-open-loop`, the `ol`
arm is `decode_transitions(rollout_transitions(…recorded actions…))`, i.e.
**exactly the WM canary quantity** (`train_flagship_v4.py:584-586`), pinned in
`test_grounding_t0_arm_reproduces_the_canary_rollout_decode` against the imported
`rollout_decode`. Its dense ADE should land near the run's banked canary; a large
divergence means the frame/sub-frame or the checkpoint is wrong, and should be
read that way before any T1 number is quoted.

---

## 4. The pod command lines (what `t1_v58f_chain.sh` runs)

```bash
export PYTHONPATH=/workspace/TanitAD/stack        # or: ModuleNotFound: tanitad
export OMP_NUM_THREADS=6                          # torch: ~113 threads/proc

python3 -u /workspace/TanitAD/taniteval/tools/t1_eval.py \
  --arm v5f-30k \
  --ckpt /workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt \
  --grounding-readout \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 \
  --episodes 40 --window-stride 1 --chunk 16 \
  --with-t0-open-loop --with-hold-action --n-boot 2000 \
  --dump-dir /workspace/experiments/t1-v58f/dump_v5f_30k \
  --out     /workspace/experiments/t1-v58f/t1_v5f_30k.json

# ... identical flags, second arm:
#   --arm stage-a-repaired
#   --ckpt /workspace/experiments/stage-a-predictor/ckpt_stage_a.pt
#   --dump-dir …/dump_stage_a_repaired  --out …/t1_stage_a_repaired.json

python3 -u /workspace/TanitAD/taniteval/tools/t1_summary.py \
  --arm v5f-30k=/workspace/experiments/t1-v58f/t1_v5f_30k.json \
  --arm stage-a-repaired=/workspace/experiments/t1-v58f/t1_stage_a_repaired.json \
  --paired stage-a-repaired,v5f-30k --n-boot 2000 \
  --out /workspace/experiments/t1-v58f/t1_summary.json
```

`t1_v58f_chain.sh` wraps exactly this: waits for `COTRAIN_EXIT` in
`/tmp/cotrain.log` (one trainer per pod), **greps the shipped `t1_eval.py` for
the adapter flags and functions and refuses with
`T1_EXIT=SYNC_FAILED_FLAG_MISSING` if any is absent** (pods have no git — a
missing flag means a stale file, and running the old code silently would produce
a legacy-path error, not a T1 row), runs the adapter's own CPU tests as a
pre-flight (**the pod has torchvision, so `test_real_v2_provider_path` — which
SKIPS on the dev box — actually drives `build_v2_providers` → `V2RawEp` there,
before any GPU time**; a failure is `T1_EXIT=ADAPTER_TESTS_FAILED`), then runs
both arms and the summary and ends with `T1_EXIT=`. Grid knobs are env-overridable (`T1_EPISODES`, `T1_STRIDE`,
`T1_CHUNK`, `T1_NBOOT`) so the run can be time-boxed **without editing the file**;
whatever is used is stamped into every JSON under `rollout_provenance.grid`.

---

## 5. What every number will carry

* **tier** — `cl`/`ha` = **T1** (primary), `ol` = **T0** (never quotable as
  driving performance). Un-stamped arms are a hard error in `analyze`.
* **estimator** — full-set pooled point estimates; episode-cluster bootstrap
  intervals; **paired** episode-cluster bootstrap for `cl − ol` (within arm) and
  for stage-a − v5f (across arms, in `t1_summary.json`). Never
  `overlapping_holdout_se`, never quadrature.
* **four families** — LONGITUDINAL / LATERAL from the waypoints;
  TACTICAL / STRATEGIC report `UNAVAILABLE` with reason + n (a T1 dump traverses
  no decision heads) — a **WORK ITEM, not a pass**. Distance-keeping (headway /
  TTC) needs a lead block: build it with `taniteval/tools/build_lead_block.py`
  **on this dump's grid** and re-run with `--lead`, or it stays UNAVAILABLE.
* **provenance** — checkpoint, corpus key, decoder, cache/model frame,
  sub-frame, grid (episodes / window / k / stride / n_windows), amp.

---

## 6. Known risks (stated, not resolved)

1. **⚠️ Nothing GPU-side is verified on the dev box** (no CUDA, no checkpoint, no
   v2 corpus, and **no torchvision** — so the one test that drives the real
   `build_v2_providers` SKIPS here and runs on the pod). First pod run must be
   read as a *smoke test as well as a result*: check `ol`'s dense ADE against the
   banked canary before quoting anything.
2. **Runtime is a HYPOTHESIS, not a measurement.** Assuming ~200-frame clips:
   40 episodes × ~(T−28) ≈ 172 stride-1 windows ≈ 6.9 k windows × (encode 8
   frames @176×624 + 3 rolls × 20 predictor steps), plus ~10 PNG decodes per
   window (the frames proxy re-decodes the 7-frame overlap between neighbouring
   stride-1 windows — the LRU caches the compressed payload, not decoded
   frames). If it over-runs the slot, set `T1_STRIDE=8` (the 881-window W4/eval
   grid) — the grid is stamped, and both arms must use the SAME value or the
   cross-arm paired join REFUSES (by design).
3. **`implied_controls` is new math** (§3). It is an inversion of a documented
   mint and it is unit-pinned, but it has never run against a real decoder. If
   the v5f readout emits large lateral `dy`, the implied `steer` will be driven
   by a component the corpus action channel cannot express — that would show up
   as a T1 collapse and must be diagnosed, not quoted.
4. **AMP.** bf16 autocast is ON by default for `--grounding-readout` on cuda,
   because that is the regime the canary/W7/stage-a numbers this arm is compared
   against were produced in. Control feedback is computed in fp32 inside the
   roll. `--no-amp` switches it off; the resolved value is stamped.
5. **Episode identity.** v2 providers carry the collision-free
   `stable_episode_id`, but the bootstrap clusters on the **dump file** (one npz
   per episode), so clustering is correct regardless. The ids are recorded in
   `rollout_provenance.corpus.episode_ids` for cross-checking against W4/W7 runs.
6. **`stage-a-repaired` shares v5f's grounding heads** (Stage-A trains the
   predictor only and re-saves the original head/grounding). So the two arms
   differ **only** in the predictor — which is exactly the attribution this pair
   is for, and worth stating wherever the delta is quoted.
