# PLAN: the LoRA run on the PI's pod, and how it turns into scale (Master Mind, 2026-11-06 ~21:50)

The PI's go (2026-11-06, verbatim): *"Ill give now a pod to use and fine tune with lora. Use it effectivey to scale and develop a smart pan. Give me the detailed plan"*

The arm is `PREREG_V8F_LORA` as registered (Amendments 1 to 3): rank-32 LoRA on q and v of all 24 blocks of frozen DINOv3-L/16, on the fly inside v8F-mini M2b, one lever, seeds 0 and 1, bars unchanged. This plan changes no bar. It decides the venue, the order and the throughput, and it says what the result unlocks.

## 1. What the pod is (MEASURED 2026-11-06 21:35, over ssh)

| item | value | consequence |
|---|---|---|
| GPU | **RTX 4000 Ada, 20 GB** (sm_89), driver 595.91, CUDA 13.2 | not the A40 the card's estimate assumed; same Ada generation as box 2's RTX 4060 |
| CPU | cgroup quota 15.3 CPUs (`cfs_quota_us` 1,530,000); `nproc` shows 128 | loader and frame build use quota − 2 = 13 workers, never `nproc` |
| RAM | cgroup limit 71.0 GB | fine for the loader and page cache |
| disk | 500 GB overlay at `/` and `/workspace`; **no network volume** | a pod stop loses everything not banked: frames and checkpoints go to HF private as they are made |
| python | 3.12.3, torch 2.8.0+cu128 preinstalled | a venv with box 2's torch **2.11.0+cu128** is building (identity needs the same numerics) |

The pod's address lives only in `C:/Users/Admin/lorapod/pod_direct.txt`, never in a repo file. The repo is public.

## 2. How long one seed takes here (the number that decides everything)

**MEASURED on the pod, a proxy** (`C:/Users/Admin/lorapod/bench_vitl_lora.py`, random weights, ViT-L/16 shape of DINOv3-L at 256 × 640 without its RoPE, frozen bf16, rank-32 fp32 LoRA on q and v, calls of exactly 8 frames as Amendment 3 requires):

| pass | ms per frame | peak per 8-frame call |
|---|---|---|
| forward, no grad | 9.44 | — |
| forward + backward, no checkpointing | 22.2 | 4.06 GiB |
| forward + backward, checkpointing | 29.4 | 1.09 GiB |

* Trainable parameters in the proxy: **3,145,728**, the card's exact count.
* AR-2 MEASURED 9 unique frames per window, so a batch of 64 encodes about 576 frames.
* The exact gradient-caching step (Amendment 1 item 6) encodes every frame twice, once without grad and once with. Checkpointing is not needed on 20 GB. That is 31.6 ms per frame, about **18 s per step**.
* **One seed (19,164 steps) ≈ 97 h of encoder time, ESTIMATED; call it 100–120 h with the heads, the loader and DINOv3's real attention.** About 4–5 days per seed.
* Box 2 projects 470 h per seed (MEASURED step time, `raw/2026-10-25-lora-ar2/run3/`), so this pod is about 4–5× box 2.
* The real 200-step smoke on this GPU (Phase 1) replaces this estimate before anything trains.

## 3. The plan, phase by phase

### Phase 0: make the pod ready (tonight, about 2–3 h wall, no training)
1. ✅ Address kept out of the repo. ✅ HF token placed by the PI's 2026-10-21 method (37 bytes, mode 600, never printed).
2. ⏳ venv with torch 2.11.0+cu128 (box 2's), then the stack shipped at the pinned tip as a tree export (pods have no git credentials). `tanitad.__file__` is asserted.
3. DINOv3-L weights fetched, sha256 must equal the builder's `dcb2e451…8179`.
4. **Rebuild the frames on the pod.** Box 2 deleted 3,972 of its 4,572 train frame files after encoding, so the on-the-fly arm needs them again.
   * Source: the private HF B1 bank of PhysicalAI-AV `camera_front_wide_120fov` mp4s, per clip.
   * Builder: the production route (`pod_build_b1_epcache.build_one`, native 256 × 640 cylinder), read and never edited.
   * Size: about 165 GB for 4,572 clips plus the 141 eval clips. About 8.2 s per clip per CPU (MEASURED, dev box) ⇒ about 1 h with 13 workers, ESTIMATED.
   * Byte gate: the two control clips `8f387977`, `bf5ec76e` and eval clip `dca157a0` must be byte-identical to the banked hashes (`frames_sha256.txt`). A mismatch stops the build.
   * Banked to HF private in chunks as they finish (the card's AR-4), so a pod stop costs a download, not a rebuild.

### Phase 1: prove the pod is the same instrument (about 1 h, after box 2's stage 4 frees it)
5. **Same-device identity on this GPU.** At LoRA B = 0, the on-the-fly tokens are compared with box 2's stored cache on the 6 AR-2 clips. τ_device is 1e-3, unchanged.
   * The cache rows for those 6 clips are a small pull from box 2, made after its stage-4 job ends. Box 2 runs one job at a time.
   * **If it passes:** the banked M2b (B, F = 0.1207 m) stays the base.
   * **If it misses:** this is a new device class, so a miss is likely. The card already commits the fallback (Amendment 1 item 3, kept by Amendment 2):
     * re-encode the train and eval caches on THIS GPU: about 470k frames at 9.4 ms, about 1.3 h;
     * retrain B′ = M2b at seeds 0 and 1 on them: hours, not days;
     * take F′ from B′.
     * L is then read against B′. Base and arm are on one device by construction, so the arm stays one lever.
6. **The real smoke on this GPU:** 200 real-data steps.
   * Every loss term finite.
   * A non-zero gradient on every LoRA group, and exactly 0.0 on the frozen DINOv3 weights, with a detach mutant that must go RED.
   * The training-path identity probe.
   * **The measured s/step, which becomes the projection of record.**
7. **Launch gate PASS** on the exact commit and argv, with the encoder sha bound in the token. No training without it.
8. **Amendment 4, written and frozen BEFORE any LoRA training step.** It records:
   * the venue (this GPU, torch 2.11);
   * the identity outcome, and therefore B or B′;
   * the measured step time;
   * checkpoints every 1,000 steps, each banked to HF;
   * the four families plus the speed and crawl rows (M1) reported beside, deciding nothing.
   * No bar, estimator or decision row moves.

### Phase 2: seed 0, the card's Stage 1 (about 4–5 days on this GPU)
9. One detached run under the supervisor, with the launch-gate token. A script watches it, not a live agent (token budget rule 1). Checkpoints go to HF as insurance, because there is no network volume.
10. The read: the PRIMARY (δ₀ < 0, separated, |δ₀| > 2F) and M-LAT (T's present-slot latcons −12.5 %), with all four families (STRATEGIC PAUSED, PI item 25) and the speed rows beside.

### Phase 3: what each verdict does next (all committed in the card; Rule Zero: no stop at a verdict)
| seed-0 read | next, executed in the same turn |
|---|---|
| PASS | seed 1 (Stage 2). If both pass, **SUCCESS**: L becomes the base, then the T1-kin read of L, then a T2 dominance stage on box 2 with L's composite (new card), then D3 on L |
| FAIL-MECHANISM-OK | T got better and O did not convert it: the O-side arm (O's longitudinal head read against T's own code), under its own card |
| FAIL-NO-MECHANISM | the frozen-plus-LoRA features are not T's limit at this data scale: D3 (more clips), else T's reader (the tactical heads read the WM latent z²) |

## 4. Where "scale" comes in, and what I will ask for

* **The infrastructure built here IS the scaling path.** Building frames on a pod from the HF bank, plus the on-the-fly encoder, is exactly what D3 (more PhysicalAI-AV clips than our 4,512) needs. After Phase 0, scaling data costs only disk and hours.
* **Wall-clock is the constraint, not money.** At the usual rate for this GPU class (about $0.2–0.3/h, ESTIMATED; your console shows the real one), a seed costs about $25–35 but takes 4–5 days. Two seeds in sequence take 9–10 days.
* **The decision I will bring after Phase 1's measured step time** (it is the PI's to provision; nothing is assumed):
  * **(A) stay on this pod:** about 9–10 days for both seeds, the cheapest option;
  * **(B) a second pod of the same class:** seed 1 runs beside seed 0 for both seeds in about 5 days at the same total cost. The card's stage rule is unchanged: seed 1 is read only if seed 0 passes;
  * **(C) a larger GPU (A100 / H100 class):** ESTIMATED 5–8× this one from published bf16 throughput, so about 15–25 h per seed for a similar dollar cost. It needs the B′ base on that device (Phase 1 step 5, about 3 h there).
* **The scale signal itself (data and model size) comes only with a counted DOMINANT**, as you set.

## 5. What runs elsewhere meanwhile (so the fleet is not idle while the pod trains)

* **Box 2:** the T2 dominance stage 4 (running since 20:18); then the read of record on 48 of 62 scenes.
* **Dev box, CPU:** the free speed-input check of the crawl (does the speed history alone reproduce it?). It informs the reading of L's M1 rows, and costs no GPU.
* **Pod CPU while the GPU trains:** none planned. The loader needs the quota.

## 6. Risks, named

| risk | handling |
|---|---|
| pod stops, container disk lost | frames and checkpoints banked to HF; restage from HF, never rebuild (pod checklist §10) |
| identity misses on this GPU | the card's committed B′ fallback, about 1.3 h encode plus the two retrains, all on this pod |
| step time worse than the proxy | the real smoke measures it before launch; option (B) or (C) is the remedy, the PI's choice |
| the pre-registered prior for a PRIMARY pass is LOW (the card, §Why) | the MECHANISM read and the M1 rows are informative either way, and each verdict has its next arm committed |
| the address leaking into the public repo | it lives only in `C:/Users/Admin/lorapod/`; every bank hold is grepped for the IP, port and pod id before landing |
