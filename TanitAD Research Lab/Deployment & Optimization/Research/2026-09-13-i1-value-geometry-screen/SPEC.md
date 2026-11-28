# SPEC / PRE-REGISTRATION — `E-DEP-VGEO-1`: is the frozen trunk's latent geometry a usable goal-distance value?

**Date** 2026-09-13 (Europe/Berlin) · **Author** Research Lab (daily pass, LAB-RUN-012) ·
**Branch** `agent/arch-inf-20260803` · **Serves** INJECTED row **I-1** (top of lane) via proposed row **D10-1**.
**Status:** written **BEFORE any statistic below was computed.** The only facts known are the bank's
shape/metadata (quoted with sources).

## 0. Why this, today

D10-1 (`LAB_BACKLOG.md`, 2026-09-10) is the cheapest discriminating step left on I-1: *"Fit
`V(s,g) = −‖z_s − z_g‖` on frozen-trunk latents and measure rank correlation with realised
time-to-goal … Committed: ρ ≥ 0.5 ⇒ the geometry is adequate and I-1 returns to being a scoring
question; at or near the shuffled control ⇒ the geometry IS the binding constraint, I-1 is blocked on
the trunk-freeze, and no value-head arm runs this cycle."* It has been open three days at 0 GPU.

## 1. Data — the bank, named with its consumer

* **Bank:** `C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens/` — built today by the A&I
  FlyWheel's `code/p4a_extract_tokens.py` (repo:
  `TanitAD Research Lab/Architecture & Inference/Research/2026-09-13-bev-lidar-corpus-and-head/code/`).
* **Trunk:** refcv5-v2 `refc.ResNetEncoder`, **90,458,632** params, `ckpt_40284.pt` strict-loaded
  (index meta `ckpt_step 40284`). Frozen; one inference pass; no training.
* **Rows:** `tokens_s32_fp16.npy` **[27,664, 704, 8, 20]**; `pix64_u8.npy` **[27,664, 9, 64, 160]**
  (4×4 area mean of the same 9-channel stack the trunk saw); `index.npz` `clip_ordinal` over
  **139 clips** (B1 EVAL join), 197–205 rows/clip, `raw_frame` consecutive.
* **Clock:** the v2ep episode grid, `int(span·10)` ⇒ **1 row step = 0.1 s** (A&I RESULT §3.1,
  INHERITED, not re-derived here).
* ⚠️ **Scope vs D10-1's wording.** D10-1 says *"banked v7 latents"*. No v7-trunk frame bank exists on
  the dev box (searched: repo `*latent*.pt`, `tanitad-caches`, `tanitad-v7f*`). The refcv5-v2 trunk is
  the programme's current frozen trunk and the bank carries exactly what the screen needs (ordered
  frames per clip + a raw-pixel floor on the same rows). Declared as a deviation, not hidden.
* ⚠️ **Load discipline.** A Master-Mind BEV-head arm is live on the 4060 (95 % util, ~13.5 GB host RAM,
  ~2 GB free). ⇒ **CPU only, below-normal priority, every 4th row subsampled** (≈50 rows/clip,
  ≈6.9 k rows), mmap row reads. No GPU.

## 2. The statistic

* Representation `z` (primary): s32 tokens **mean-pooled over the 8×20 grid ⇒ 704-d**, fp32.
  Secondary (declared now): **2×5 area pool ⇒ 7,040-d** (keeps coarse azimuth, per our measured
  readout-geometry ceiling).
* Pixel floor `x`: pix64 area-pooled to **9×8×20 = 1,440-d**, /255.
* Pairs: every ordered pair (s, g) in the **same clip** with g later than s, `Δ = raw_frame_g − raw_frame_s`
  in **[4, 160] rows = 0.4–16 s**. Bands reported separately: **short Δ ≤ 60 (≤ 6 s, tactical)** and
  **long Δ > 60**.
* `V(s,g) = −‖z_s − z_g‖₂`. Per clip: Spearman ρ between `‖z_s − z_g‖` and `Δ` (sign so that
  "further in time ⇒ further in latent" is positive). **Primary = mean over clips of per-clip ρ.**
* **Estimator:** clip-cluster bootstrap over the 139 clips, 2,000 resamples, 95 % percentile CI.
  Paired latent−pixel difference bootstrapped on the same clip draws.
* ⚠️ Which variance the CI answers: **another draw of CLIPS only.** One frozen checkpoint, one extraction ⇒
  blind to training variance (`H-ESTIM-SEED-1`). No inference sampling is involved (deterministic
  encoder), so the inference-variance question does not arise.

## 3. Controls — each must read a known value

| control | construction | must read |
|---|---|---|
| **C-const** | `V ≡ 0` for every pair | ρ **undefined ⇒ reported as exactly 0.0000** (ties) |
| **C-shuf** | for each s, replace g by a uniformly random row of the SAME clip, keep the ORIGINAL Δ label | **≈ 0** (CI must contain 0) |
| **C-pix** | same V on the 1,440-d pixel floor | a number; **the latent must beat it** |
| **C-mut** (deliberate regression) | latent rows **permuted within clip** before the statistic | **≈ 0** — proves the pipeline cannot manufacture ordering |

## 4. Committed outcomes (all branches, incl. WORSE)

| outcome | rule (short band Δ ≤ 60 is the decision band) | consequence |
|---|---|---|
| **ADEQUATE** | ρ_latent ≥ 0.50 **and** paired (latent − pixel) CI lower bound > 0 | I-1 returns to a scoring question; D10-1 ✅; value-head arm admissible |
| **ADEQUATE-BUT-NOT-LEARNED** | ρ_latent ≥ 0.50 **but** paired (latent − pixel) CI includes or is below 0 | temporal ordering is a smoothness property of video, not of the trunk ⇒ this screen cannot discriminate; **next lever = a CROSS-CLIP goal-retrieval test** (does `V` rank a same-scene future above a different-clip frame at matched pixel distance?) — run in the same pass if cheap |
| **PARTIAL** | 0 < ρ_latent < 0.50 with CI excluding the C-shuf value | geometry carries ordering but below bar ⇒ I-1 is not blocked outright; a learned metric head over frozen tokens (quasimetric) is the next lever, not an encoder unfreeze |
| **BINDING** | ρ_latent CI overlaps C-shuf | geometry IS the constraint ⇒ no value-head arm this cycle; I-1 blocked on the trunk freeze (escalate to MM) |
| **WORSE** | ρ_latent < 0 with CI excluding 0 | latent distance anti-orders time — a defect claim; re-check pooling on the 7,040-d variant before stating it |
| **INVALID** | any control misses its known value | no verdict; instrument fault reported |

⛔ Tier: **none** (no driving, no rollout). This is a representation diagnostic on frozen features.
⛔ Nothing here is a four-family driving eval and it is not presented as one.
