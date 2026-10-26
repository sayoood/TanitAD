# Dense Alpamayo2-Super reasoning — pod RUNBOOK

**Status (2026-10-25): code + CPU tests staged; NOTHING launched.** Spend (which GPUs, how many hours, which rate) is the PI's
decision. Code: `stack/scripts/alpamayo_dense_reasoning.py` (generator), `stack/scripts/merge_dense_reasoning.py` (merge),
`stack/tests/test_alpamayo_dense_reasoning.py` (24 CPU tests).

PI request (2026-10-25): *"I thought the Alpamayo reasoning exists for each frame — we need it for each frame."*
Today the corpus `alpamayo/records.parquet` (= HF `Sayood/tanitad-alpamayo2-augmentation`) has nvidia/Alpamayo2-Super outputs at
**one** anchor per clip, t0_us = 5,100,000, for 4,729 clips / 23,644 rows (MEASURED on the parquet; the repo README's "4,800 clips,
23,999 rows" is INHERITED and does not match the file). This pipeline generates them for every v9 row.

---

## 0. Read this first — what "every frame" can and cannot mean

| fact | value | class |
|---|---|---|
| target frames at 10 Hz (= every v9 row) | train **869,278** rows / 4,369 clips; eval139 **27,664** / 139 | MEASURED (`plan`, = v9 manifests) |
| spacing of consecutive frames | **≈ 0.1 s** (v9 dt 0.1000–0.1010 s) | MEASURED (v9 `clip__dt_s`) |
| rows with t0 ≤ 1.6 s → NVIDIA's loader asserts `t0_us > 1_600_000` (16-pose history) | **59,973 = 6.7 %** of rows, both tasks | MEASURED (v9 clock) |
| rows whose ego future < 6.4 s → `auto_labeling` (conditioned on the ground-truth future, NVIDIA default) cannot run | **≈ 284,000 = 31.7 %** of rows (the last 6.4 s of each ~20 s clip) | ESTIMATED (proxy: ego log ends 0.1 s after the last v9 row) |
| ⇒ eligible: meta_action ≈ 93.3 % of rows, auto_labeling ≈ 61.6 % | | ESTIMATED |

**Near-duplicate inputs.** The model sees 4 frames per camera at t0−0.3, −0.2, −0.1, t0 and a 16-pose history. Two consecutive
10 Hz frames therefore share **3 of 4 images per camera and 15 of 16 history poses**; only one new image per camera enters. The outputs
are strongly correlated, so information per GPU-hour falls as the rate rises: 10 Hz costs 5× the GPU-hours of 2 Hz and 10× of 1 Hz
(table §6) for mostly re-worded neighbours. **The default is nevertheless every row (`--hz 10`) because the PI asked for it.** Every
lower rate (`--hz 5|2|1`) keeps, in every clip, the row nearest t0 = 5.1 s, so it stays comparable with records.parquet.

**Escalated decisions (PI):**
1. **auto_labeling on the last 6.4 s of every clip.** Options: (a) accept ≈ 61.6 % coverage, rows written as
   `ineligible:ego_future<6.4s(...)` (default); (b) `--autolabel-future predicted` (NVIDIA-supported: sample the expert trajectory first
   and auto-label it; hf backend only, adds one trajectory sampling per frame, and the CoC then explains a *predicted* future — a different
   label); (c) meta_action only for those rows. Nothing silently drops — every row carries its reason.
2. **First 1.6 s.** `--relax-history-assert` lifts NVIDIA's assert when the ego log covers t0−1.5 s (flagged `hist=relaxed`). Outside
   NVIDIA's contract → default OFF.
3. **Rate and GPU budget** (§6).

**Leak rule (CLAUDE.md BINDING).** Both tasks read the ego history; `auto_labeling` reads the ego **future** trajectory. Every output
here is a LABEL / teacher signal. It may supervise; it must never be an inference input of a deployable arm, and it must not enter a goal
input that is supposed to be information-disjoint from the situation classifier.

---

## 1. What is reproduced, and from where (all pinned)

| item | source | pin |
|---|---|---|
| model | HF `nvidia/Alpamayo2-Super`, 35,813,934,322 BF16 params, 15 shards, **71,627,868,644 B** | revision `00554695e729a6ff0b6281fd2c81b18d06e33dbe` (HF API sha, read 2026-10-25) — MEASURED |
| licence | weights openmdw-1.1; code Apache-2.0 | model card — PUBLISHED |
| code | github.com/NVlabs/alpamayo2, package `alpamayo2_super` (`load_physical_aiavdataset.py`, `input_profiles.py`, `text_tasks.py`, `notebooks/meta_actions.ipynb`, `notebooks/autolabeling.ipynb`) | commit `59776920bbd981f2c989213bba5ee8594e71ed1d` (2026-10-06) |
| data API | `physical-ai-av` (range-reads one clip's ZIP members) | 0.2.3 |
| input profile | cameras **[0,1,2,3,5,6]** (cross-left, front-wide, cross-right, rear-left, rear-right, front-tele) × frames [0..3] at t0−0.3…t0 s, each = closest frame **at or before** the request | = records raw_json `input_profile`, `camera_indices`, `frame_indices` |
| ego | 16 history poses t0−1.5…t0, 64 future poses t0+0.1…t0+6.4 s (0.1 s), in the t0 ego frame | loader |
| prompts | meta_action: "output the chain-of-thought reasoning of the driving process, then output meta actions, then output the future trajectory."; auto_labeling: future-trajectory tokens + "output a comprehensive analysis of the driving scene in JSON format." | `text_tasks._TASK_PROMPTS`, `conversation.py` |
| sampling | top_p 0.98, temperature 0.6, top_k None, max_new_tokens 512 / 1024, trajectory tokens masked, meta_action stops after `<|traj_future_start|>` | `text_tasks.generate_text` + notebooks |
| seed | `torch.cuda.manual_seed_all(42)` before each generate | notebooks; records `seed` = 42 for every row (MEASURED) |
| original quantisation | "NF4-backbone-4bit-UNVALIDATED" / "4-bit bnb load" — exact bitsandbytes config **not published** | records column / repo README — INHERITED |
| original speed | median `wall_s` meta_action **10.0 s**, auto_labeling **18.1 s** (single stream, GPU unknown) | records.parquet — MEASURED on the file, INHERITED as a speed (hardware not recorded) |
| original code | not in our repo (searched); reconstructed from the above | — |

`build_frame_data` re-implements the loader + `select_input_profile` so a clip's videos are opened **once** (not per frame) and the 3
frames shared by neighbouring rows are decoded once. **`self-test-assembly` compares it tensor-by-tensor (and the fused prompt token ids)
with NVIDIA's own loader before any run — it must pass.**

---

## 2. Hardware

| quantity | value | class |
|---|---|---|
| full checkpoint (VLM + 2.3 B expert) bf16 | 71.6 GB | MEASURED (index `total_size`) |
| VLM only (all text tasks; the expert is used only by `trajectory` / `--autolabel-future predicted`) | ≈ 67 GB (≈ 33.5 B params) | ESTIMATED (35.8 B total − 2.3 B expert, card) |
| NVIDIA-measured peak, H100 80 GB, 7 cams × 4 frames, batch 1, bf16, SDPA | 72,115 MiB device | PUBLISHED (model card) |
| KV cache per token (64 layers × 8 KV heads × 128 × 2 (K,V) × 2 B) | 256 KiB | from config.json |
| prompt length: 24 images × 180 visual tokens (1080×1920 → 320×576 under min/max 163,840 / 196,608 px, 32-px merged patches) + ≈ 300 text/trajectory tokens | ≈ 4,600–4,700 tokens → ≈ 1.2 GB KV per request | ESTIMATED from config + Qwen smart-resize |
| NF4 (bnb) LM backbone + bf16 vision / lm_head / expert | ≈ 25 GB | ESTIMATED |

| GPU | fits | role |
|---|---|---|
| **1× H100 80 GB** | bf16 HF single stream (as NVIDIA tested); vLLM bf16 fits but leaves only ≈ 5 GB KV ≈ 4 concurrent requests | single stream / pilot; vLLM only with `--quant fp8` (outputs then differ — flagged in `quantisation`) |
| **2× H100 80 GB, vLLM `--tp 2`** | ≈ 80 GB KV ≈ 60+ concurrent requests | **recommended production unit** |
| **1× H200 141 GB / B200 192 GB** | ≈ 60 / 110 GB KV | recommended production unit (no TP) |
| A100 80 GB | as H100 at ≈ ⅓ the bf16 FLOP/s | acceptable, slower |
| **1× A40 48 GB (our fleet)** | bf16 does **not** fit; NF4 HF fits (≈ 25 GB) | reproduces the original quantised run (`--backend hf --quant nf4`), slow (§6) |
| 2× A40, vLLM `--tp 2` bf16 | ≈ 20 GB KV ≈ 15 concurrent requests | possible; ≈ 10 A40-GPU-s/frame (§6) |

Host: ≈ 10 GB RAM per 128 queued vLLM requests (24 uint8 1080p frames per frame-request, shared by its two tasks); `--vllm-image-mode
float` (NVIDIA-exact float pixels) costs ≈ 600 MB per request — small batches only. Each clip opens 6 videos (≈ 12 MB each).

---

## 3. Environment (pod)

Traps (CLAUDE.md): `PYTHONPATH=/workspace/TanitAD/stack` is REQUIRED; set `OMP_NUM_THREADS=6`; never judge disk with `df` (use `dd`);
never `pgrep -f`/`pkill -f` — kill by explicit PID; sync the pod checkout and verify with a real import before launch; never run on a
training pod.

```bash
export REPO=/workspace/TanitAD
export PYTHONPATH=$REPO/stack
export OMP_NUM_THREADS=6
export HF_HOME=/workspace/hf                       # model + physical_ai_av subset cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export V9_DIR=/workspace/data/v9                   # v9_labels_{train,eval139}.npz (check npz_md5 in the manifests)
export CI_DIR=/workspace/data/index                # clip_index_v7.2_{train,eval}.json (corpus index/)
export RECORDS=/workspace/data/records.parquet     # Sayood/tanitad-alpamayo2-augmentation/records.parquet (sha256 ecae276d…5ed2)
export MODEL=/workspace/models/Alpamayo2-Super
export VLLM_MODEL=/workspace/models/Alpamayo2-Super-vlm-vllm
export OUT=/workspace/reasoning_dense
cd $REPO && git fetch && git status && git log -1 --oneline          # must contain the staged scripts once committed
python $REPO/stack/scripts/alpamayo_dense_reasoning.py --help >/dev/null && echo SCRIPT-OK     # real import, not git log

# disk: real write test (quota is per pod, df lies)
dd if=/dev/zero of=/workspace/.ddtest bs=1M count=20000 oflag=direct && rm /workspace/.ddtest   # needs ≥ 20 GB free to pass
```

**A2S venv (hf backend, assembly self-test, export):**
```bash
git clone https://github.com/NVlabs/alpamayo2 /workspace/alpamayo2
cd /workspace/alpamayo2 && git fetch --depth 1 origin 59776920bbd981f2c989213bba5ee8594e71ed1d && git checkout 59776920bbd981f2c989213bba5ee8594e71ed1d
export UV_PROJECT_ENVIRONMENT=/workspace/venv-a2s UV_CACHE_DIR=/workspace/uv-cache
uv sync --locked --dev                      # torch 2.8.0, transformers 4.57.1, physical-ai-av, flash-attn build (needs nvcc; MAX_JOBS=8 if RAM-bound)
source /workspace/venv-a2s/bin/activate
uv pip install bitsandbytes                 # only for --quant nf4
hf auth login --token "$(grep -oE 'hf_[A-Za-z0-9]+' /workspace/Keys.txt | head -1)"   # read in place; account must have accepted the gated model + dataset
python -m alpamayo2_super.inference_smoke --help >/dev/null && echo A2S-OK
```

**vLLM venv (batched backend):** HYPOTHESIS to verify on the pod before relying on it: `vllm==0.11.0` supports
`Qwen3VLForConditionalGeneration` and pins `torch==2.8.0` (same as A2S).
```bash
uv venv /workspace/venv-vllm --python 3.12 && source /workspace/venv-vllm/bin/activate
uv pip install --dry-run "vllm==0.11.0" | grep -E "torch==|transformers=="          # check before installing
uv pip install "vllm==0.11.0" "transformers==4.57.1" physical-ai-av==0.2.3 hydra-core einops av scipy pandas pyarrow safetensors
uv pip install --no-deps -e /workspace/alpamayo2                                      # alpamayo2_super code (CPU tokenisation only)
python -c "from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration; print('vllm-qwen3vl-OK')"
```

**Model download (≈ 72 GB; gated):**
```bash
hf download nvidia/Alpamayo2-Super --revision 00554695e729a6ff0b6281fd2c81b18d06e33dbe --local-dir $MODEL
python - <<'PY'
import json,os; d=json.load(open(os.environ["MODEL"]+"/model.safetensors.index.json")); print(d["metadata"])   # total_size 71627868644
PY
# vLLM checkpoint: the VLM sub-model only (keys vlm.* -> *, expert dropped, + Alpamayo tokenizer); ≈ 67 GB more disk
python $REPO/stack/scripts/alpamayo_dense_reasoning.py export-vllm --a2s-dir $MODEL --out-dir $VLLM_MODEL
```

---

## 4. Camera + egomotion inputs (by reference, member-only)

The programme corpus ships only `camera_front_wide_120fov`. The other cameras are upstream, by reference:
`DataEng/v10/camera_ref/v10_camera_ref.parquet` (+ MANIFEST: all 7 cameras present for all 4,508 v9 clips). `prefetch` range-reads, for
each clip of a shard, only that clip's `video` + `frame_timestamps` members of the 6 driving cameras and its `egomotion` member, through
NVIDIA's own `physical_ai_av.get_clip_feature_paths(..., members=[...], maybe_stream=True)` (subset cache under `--cache-dir`). It never
downloads whole chunks.

Bytes (ESTIMATED, camera_ref MANIFEST `member_bytes_estimated`): cross-left 65.7 GB, cross-right 69.0, rear-left 71.0, rear-right 75.1,
front-tele 51.4, front-wide ≈ 55 (12 MB × 4,508; not in the MANIFEST) ⇒ **≈ 387 GB for the driving profile over all 4,508 clips**,
≈ 387/N GB per shard of N. `--front-only` needs ≈ 55 GB. At the INHERITED ≈ 118 MB/s HF rate ≈ 1 h for everything; range reads may
be slower — measure on the first shard.

```bash
python $REPO/stack/scripts/alpamayo_dense_reasoning.py prefetch --v9-dir $V9_DIR --clip-index-dir $CI_DIR \
    --split train --shard 0/8 --cache-dir $HF_HOME/paiav --workers 8 | tee $OUT/logs/prefetch_train_0of8.log
```

---

## 5. Pre-flight (in this order; each is a gate)

```bash
mkdir -p $OUT/logs
# (a) suite green on the pod checkout
PYTHONPATH=$REPO/stack:$REPO/taniteval:$REPO/stack/tests python -m pytest -q $REPO/stack/tests/test_alpamayo_dense_reasoning.py

# (b) frame plan (must print 869278 / 27664 at --hz 10)
python $REPO/stack/scripts/alpamayo_dense_reasoning.py plan --v9-dir $V9_DIR --clip-index-dir $CI_DIR

# (c) input assembly == NVIDIA's loader, tensor by tensor + fused prompt ids (CPU, A2S venv). MUST print "ok": true
python $REPO/stack/scripts/alpamayo_dense_reasoning.py self-test-assembly --v9-dir $V9_DIR --clip-index-dir $CI_DIR \
    --split eval139 --k 8 --model $MODEL --cache-dir $HF_HOME/paiav

# (d) real data, no model: one clip through the whole I/O path
python $REPO/stack/scripts/alpamayo_dense_reasoning.py run --v9-dir $V9_DIR --clip-index-dir $CI_DIR --split eval139 \
    --clips <one sha12> --backend mock --out /workspace/reasoning_dense_smoke --cache-dir $HF_HOME/paiav

# (e) input-assembly parity on the anchors, model-free: camera_tmin of every records clip (records keep it on the trajectory row)
python $REPO/stack/scripts/alpamayo_dense_reasoning.py run --v9-dir $V9_DIR --clip-index-dir $CI_DIR --check-anchors $RECORDS \
    --backend mock --out /workspace/rd_anchor_mock --cache-dir $HF_HOME/paiav --tasks meta_action
python $REPO/stack/scripts/alpamayo_dense_reasoning.py report-anchors --out /workspace/rd_anchor_mock --records $RECORDS
#     gate: input_assembly.camera_tmin_equal.rate == 1.0  (anything less = assembly bug; stop)

# (f) vLLM pilot: one eval clip, prompt parity enforced (first 8 prompts compared token-for-token with NVIDIA's processor;
#     the run ABORTS on mismatch), and a throughput measurement
python $REPO/stack/scripts/alpamayo_dense_reasoning.py run --v9-dir $V9_DIR --clip-index-dir $CI_DIR --split eval139 \
    --clips <one sha12> --backend vllm --model $MODEL --vllm-model-dir $VLLM_MODEL --tp 2 --out /workspace/rd_pilot \
    --cache-dir $HF_HOME/paiav 2>&1 | tee $OUT/logs/pilot.log
#     measure: (end - start) / n_requests from the log -> replace the ESTIMATED s/frame in §6 with MEASURED before budgeting

# (g) check-anchors (text agreement with records). Pre-registered: compare each config's agreement with records against the
#     SAMPLING FLOOR = agreement between two seeds of the same config (--seed-mode per_frame vs fixed), on the same clips.
#     hf runs in /workspace/venv-a2s, vllm in /workspace/venv-vllm (activate the matching venv per iteration).
for CFG in "hf --quant nf4" "vllm --tp 2"; do
  python $REPO/stack/scripts/alpamayo_dense_reasoning.py run --v9-dir $V9_DIR --clip-index-dir $CI_DIR --check-anchors $RECORDS \
      --clips @/workspace/data/anchor_pilot_200.txt --backend $CFG --model $MODEL --vllm-model-dir $VLLM_MODEL \
      --out /workspace/rd_anchor_$(echo $CFG | tr -d ' -') --cache-dir $HF_HOME/paiav
  python $REPO/stack/scripts/alpamayo_dense_reasoning.py report-anchors --out /workspace/rd_anchor_$(echo $CFG | tr -d ' -') --records $RECORDS
done
#     outcome A: HF-NF4 seed-42 raw_exact >= 0.9 -> the original pipeline is reproduced; vLLM-bf16 is admissible if its
#                class_all_three agreement with records lies inside HF-NF4's sampling-floor CI.
#     outcome B: HF-NF4 raw_exact < 0.9 -> the original quantisation/GPU is not reproduced (expected: config unpublished);
#                the dense set is then a NEW teacher run, reported as such (quantisation column), and class agreement vs
#                records + the sampling floor are its documented relationship to the old anchor rows.
#     Full 4,729-anchor check: drop --clips (≈ 2 vLLM H100-h or ≈ 37 h single-stream at the INHERITED speed).
```

---

## 6. Launch per shard (after the PI picks rate + GPUs)

One process per GPU group; shards are clip-partitioned (`rank % N`), outputs are idempotent across re-sharding (a key present
anywhere under `$OUT/<split>/` is skipped), progress is flushed every 256 rows into `shard_XXXX.parts/`, finalised into
`shard_XXXX.parquet` when the shard is complete.

```bash
N=8   # shards per split; e.g. 4 nodes x 2 groups of 2xH100 (TP2)
for I in 0 1; do                     # this node's shard ids; CUDA groups 0,1 and 2,3
  CUDA_VISIBLE_DEVICES=$((2*I)),$((2*I+1)) nohup python $REPO/stack/scripts/alpamayo_dense_reasoning.py run \
      --v9-dir $V9_DIR --clip-index-dir $CI_DIR --split train --hz 10 --shard $I/$N \
      --backend vllm --model $MODEL --vllm-model-dir $VLLM_MODEL --tp 2 \
      --out $OUT --cache-dir $HF_HOME/paiav > $OUT/logs/train_${I}of${N}.log 2>&1 &
  echo "shard $I pid $!" | tee -a $OUT/logs/pids.txt          # kill by THIS pid only
done
# eval139 (139 clips) as one extra shard:  --split eval139 --shard 0/1
# degraded front-only variant (separate output root!):  --front-only --out /workspace/reasoning_dense_frontonly
# original-quantisation reproduction on A40s:  --backend hf --quant nf4   (≈ 28 s/frame)
# resume after a crash: re-run the identical command (done keys are skipped);  --retry-errors re-runs non-ineligible failures
```

Monitor: `tail $OUT/logs/train_*of8.log` (one JSON stats line per finished run), and
`python -c "import alpamayo_dense_reasoning as a; print(len(a.load_done_keys('$OUT','train')))"` (from `$REPO/stack/scripts`).

### Cost (GPU-hours, both tasks)

Frames per rate (MEASURED, `plan`, train + eval139): **10 Hz 896,942 · 5 Hz 448,633 · 2 Hz 180,235 · 1 Hz 90,072.**
"eligible" applies §0 (meta_action 93.3 %, auto_labeling 61.6 %, ESTIMATED proxy, same fraction at every rate).

| s / frame (both tasks) | basis | 10 Hz all / eligible | 5 Hz | 2 Hz | 1 Hz |
|---|---|---|---|---|---|
| **28.1** single stream | INHERITED: records median wall_s 10.0 + 18.1, NF4, GPU unknown | **7,001 / 5,105** | 3,502 / 2,553 | 1,407 / 1,026 | 703 / 513 |
| 7.6 single stream, HF bf16, 1× H100 | ESTIMATED: prefill 2 × 3.1e14 FLOP at 40 % of 989 TFLOP/s + decode ≈ 45 + 165 tokens at 28.6 ms/token (67 GB / (0.7 × 3.35 TB/s)) | 1,894 / 1,333 | 947 / 667 | 380 / 268 | 190 / 134 |
| **1.5** vLLM batched, per H100-equivalent | ESTIMATED: prefill-bound, 6.2e14 FLOP/frame at 50 % MFU = 1.25 s + batched decode ≈ 0.2 s; needs ≥ 30 concurrent requests (TP2 or H200) | **374 / 290** | 187 / 145 | 75 / 58 | 38 / 29 |
| 0.85 vLLM + prefix-cache reuse of the shared image+history prefix between the two tasks of a frame | HYPOTHESIS (cache hits not guaranteed when both requests prefill in the same step) | 212 / 164 | 106 / 82 | 43 / 33 | 21 / 16 |
| 10 (A40-GPU-s) vLLM TP2 on 2× A40, bf16 | ESTIMATED: 149.7 TFLOP/s bf16 at 50 % + 696 GB/s decode | 2,492 / 1,930 | 1,246 / 966 | 501 / 388 | 250 / 194 |

Hardware peaks are PUBLISHED datasheet numbers (H100 SXM 989 TFLOP/s bf16 dense, 3.35 TB/s; A40 149.7 TFLOP/s, 696 GB/s); the
efficiency factors (40–50 % MFU, 0.7 of bandwidth), token counts (≈ 4 chars/token over the records' median 154 / 654 output chars) and
the eligibility proxy are ESTIMATED. **No A2S-on-vLLM throughput has been published or measured — step 5(f) replaces this row with a
MEASURED number before any budget is committed.** The 3.7–19× speed-up of vLLM over the INHERITED single stream is mostly bf16 vs NF4
(bitsandbytes 4-bit decode is slow) plus batching.

### Storage

| item | size | class |
|---|---|---|
| outputs, 10 Hz, both tasks (raw + cot + meta + coc + keys) | ≈ 1.3 GB raw text → ≈ 0.3–0.5 GB zstd parquet | ESTIMATED (records median lengths × rows) |
| camera + egomotion member cache, driving profile, all clips | ≈ 387 GB (÷ N per shard) | ESTIMATED (camera_ref) |
| model: A2S snapshot + vLLM VLM export | 71.6 + ≈ 67 GB | MEASURED + ESTIMATED |

---

## 7. Verification + delivery

```bash
for S in train eval139; do
  python $REPO/stack/scripts/merge_dense_reasoning.py --in $OUT --split $S --hz 10 --v9-dir $V9_DIR --clip-index-dir $CI_DIR --out $OUT/merged
done
```
The merge refuses: rows that are not target frames at that `--hz`, `t_us` ≠ round(now_s·1e6) of the row, rows of another split, mixed
model / revision / quantisation / input profile (unless `--allow-mixed`), and missing (frame, task) pairs (unless `--allow-missing`).
It writes `v10_reasoning_dense_<split>.parquet`, `….coverage.parquet` (per clip × task: target / ok / ineligible / error / missing) and
`….MANIFEST.json` (sha256 of inputs and outputs, coverage per clip, error reasons, generation params, run ids, dataset revision).

Acceptance: `n_missing == 0`; `error` reasons other than `ineligible:*` < 0.5 % (else `--retry-errors`, then report); spot-read 20
random rows per task against the decoded frames.

Delivery: the merged parquet (> 2 MB) goes to a HF dataset (push is the PI's call; HF has been 403-storage-full — fall back to pod→pod
direct SSH per CLAUDE.md, verify md5). Only the MANIFESTs + coverage summaries (< 2 MB) are `git add`ed under `DataEng/v10/reasoning_dense/`.
Register the dataset in `Project Steering/MODEL_REGISTRY.md`-adjacent data docs with this run's revision pins.

---

## 8. Output schema (`tanitad.reasoning_dense/1`)

key `(split, sha12, v9_row, task)` · `clip_id` (UUID) · `t_us` (clip-local µs = round(v9 now_s·1e6); 5,100,000 for check-anchors) ·
`model_id`, `model_revision` (`+vllm_export` for vLLM), `quantisation` (`bf16`, `NF4-backbone-4bit-RECON(...)`, `vllm-fp8`, `|img=uint8`) ·
`input_profile` (`a2s_driving_6cam_4f|cams=0,1,2,3,5,6|fut=gt`; `a2s_front_only_4f_DEGRADED|cams=1|…`; `|hist=relaxed`) · `seed` ·
`wall_s` (HF: per call; vLLM: batch wall / requests) · `raw` (decoded generation, special tokens kept) · `cot`, `meta_action` (meta_action
task), `coc` (auto_labeling `chain_of_causation`; the other three JSON fields stay in `raw`) · `error` (`ineligible:<reason>`,
`source:`, `assembly:`, `parse:`, `parser_mismatch_vs_alpamayo2_super`, backend exception) · additions: `front_only` (bool, every row),
`camera_tmin_us` (first decoded frame time; = records' `camera_tmin` at the anchor), `written_unix`, `run_id`.
Parsing (`parse_generation`) mirrors `alpamayo2_super.token_utils.extract_text_tokens` / `text_tasks.parse_auto_labeling_json` and
reproduces **all 4,729 meta_action and 4,729 auto_labeling rows of records.parquet exactly** (MEASURED, CPU test).

## 9. Known deviations / open risks

* **vLLM ≠ HF draw.** vLLM's sampler RNG is not torch's: same weights + seed ⇒ different text. Measured by step 5(g), not assumed away.
* **vLLM pixels.** Default `uint8` images resize in uint8 (rounding ≤ 0.5/255 before normalisation) where NVIDIA resizes float [0,1];
  `--vllm-image-mode float` is exact at ≈ 600 MB host RAM per request. Prompt TOKEN parity is enforced either way.
* **vLLM vocab mask** excludes ids ≥ len(tokenizer) (untrained padding rows of the 155,776-row head) in addition to the 4,000
  trajectory tokens NVIDIA masks; never sampled in practice.
* **NF4 reconstruction.** The original 4-bit config was not published; ours quantises the LM backbone and keeps vision / lm_head /
  expert bf16. Labelled `RECON` so it can never be confused with the records' run.
* **front-only** is not an NVIDIA-validated profile (the model was trained on 6–7 cameras): flagged DEGRADED in every row; keep it in a
  separate output root.
* **Egomotion coverage** is estimated; the true ineligible counts come from the run (`error_reasons` in the MANIFEST).
* Producer/consumer overlap (decode next batch while the GPU generates) is not implemented: ESTIMATED ≈ 5 % idle at batch 128.
