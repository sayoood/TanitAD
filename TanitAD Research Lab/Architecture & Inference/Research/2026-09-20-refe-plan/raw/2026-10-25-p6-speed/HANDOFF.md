# HANDOFF: P6 speed study (2026-10-25)

**State.** The GPU side is DONE (see `RESULTS.md`, ⭐ FINAL SUMMARY at the top). Two GPU queues are still running on the pod: refinements for the PI decision. The CPU side (labeller + loader profile) is **BLOCKED** until `/workspace/venv-teacher` (NAVSIM + cv2) and `/workspace/data/refe_navtrain/metric_cache` (the D9 parts) land on the pod. Both were still empty at hand-off.

**Headline.**
* A 50 % recipe-preserving cut is physically out of reach on this A40. The GEMM + FA2 floor is 0.617 s per micro-batch, against 0.895 s today.
* All 12 recipe-preserving GPU levers measured 0 % (within the 0.47 % noise).
* The only open preserving item is the labeller stall (1.77 %, argv-only).
* 50 % needs a PI decision: a recipe change (384×704 measured −47.5 %, LoRA top-6 −41.5 %) or a faster GPU. Either way, the labeller must also get ~40–50 % cheaper per set.

**P6 can relaunch now.** `guard.sh` is running on the pod. It kills every speed-study queue and bench process within ~5 s of `/root/tad/launch_b` appearing, and logs this to `logs/guard.log`.

**Pod (`<POD>`, `/root/tad/speed/`):**
* `refe/` (an f2f604f copy), the scripts, `logs/` (`runs.jsonl` = one JSON row per timing run), `eq/*.pt` (equivalence dumps, about 150 MB each), `dinov3-vitl16/` (local weights copy)
* every number so far: system python3, **torch 2.8.0+cu128**; none re-confirmed under `venv-teacher` yet

## Exact next steps (a Sonnet agent is enough; Haiku for 1–2)

1. **Bank queues 4 and 5** once the pod has `grep -c "ZZQ DONE" logs/queue5.log` = 1. If the guard fired, they stop early; say so.
   * queue 4: `logs/equiv2_table.jsonl`, one JSON row per arm vs `e2_base_a`, with `wta_flips` and `grad_rel_l2_by_group`.
     * The **key read** is `e2_perturb6` (images × (1+1e-6), an ULP-level forward change). If its `grad_rel_l2` is ≈ 4e-2 like parts/max-autotune, then WTA winner flips explain the gap and "not equivalent" means "not bit-reproducible".
     * Write it into RESULTS.md's EQUIVALENCE section, replacing the HYPOTHESIS note, and give the `--bf16-resid` row.
   * queue 5: `logs/queue5.log` has the recipe-changing **combinations** (lora12+bf16, lora12 at batch 8, res448+lora12, lora18+bf16, res448+bf16). Add them to the RECIPE-CHANGING table (s/step = s/micro × 64 + 0.033).
2. **Labeller profile.** When both `ls /workspace/venv-teacher/bin/python` and `ls /workspace/data/refe_navtrain/metric_cache | head` succeed, and only while P6 is **not** training:
   ```
   cd /root/tad/speed && /workspace/venv-teacher/bin/python prof_labeller.py \
     --cache /workspace/data/refe_navtrain/metric_cache --refe /root/tad/speed/refe --n-sets 24 --out logs/prof_labeller.json
   ```
   * Runs on one core for ~2 min and writes 24 decompressed parts to `mc_raw/`.
   * Reports CPU-s per set as read / lzma / unpickle / plans8 / score, plus a cProfile.
   * Tests three exact levers, each with a `*_bit_identical` flag: `raw` (decompressed parts), `colo` (both chunks of a set on one worker), `chunks1`.
   * The bar is ≤ ~0.5 CPU-s/set for any GPU cut above 30 % (arithmetic in RESULTS.md).
   * If P6 is training, run it on the dev box instead, after copying ~24 parts files.
3. **Loader CPU** (`venv-teacher` has cv2): time `TargetBank.__getitem__` (4 cameras, float path vs uint8 path) on real `/workspace/data/navtrain_pixels`. Assert bit-identity: float path == uint8 path → `/255` with a 0-dim CUDA divisor. If the float conversion + 4× copy costs ≥ 0.1 CPU-s per micro-batch, land `img_u8.patch`.
   * The patch applies cleanly to f2f604f (checked with `git apply --check`).
   * ⚠️ It is UNVERIFIED end-to-end: G-SUITE has not been run on it.
4. **`--label-workers 8`** (argv only, rides the resume, gate re-mint): a candidate for the 1.77 % stall. Measure the stall in the first P6 windows after a relaunch with it, and watch cgroup throttling (`cpu.stat nr_throttled`).
5. **PI decisions to escalate (Master Mind):**
   * (a) accept that the recipe-preserving cut is ~2 % at most;
   * (b) the recipe-changing table: each is a new pre-registration, because `--declare-change` is forbidden in the P6 gate profile;
   * (c) a faster GPU.

**Guard.** Before any GPU call, check that `ls /root/tad/launch_b/` fails.
