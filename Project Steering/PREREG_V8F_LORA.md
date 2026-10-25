# PREREG_V8F_LORA: LoRA on DINOv3-L/16 inside v8F-mini, the plan's M4 (written 2026-10-25, before any instrument or data)

**Status: PRE-REGISTERED, NOT RUN.**
* **The PI's go (2026-10-25, verbatim):** *"AR-2 go: build the LoRA-on-DINOv3 pre-registration and instruments now (no GPU)."*
  * It covers this file and the instruments.
  * Every GPU step below (the AR-2 smoke on box 2, the training run on box 2 or the A40) needs its own go. The A40 is the PI's to provision.
* **The direction is the PI's ruling of 2026-10-17** (`PI-V8F-2026-10-17` item 1: *"I approve the DINOv3 L + LoRA direction"*). The ladder is `PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` §13.5: M4, then AR-2, then the full LoRA run.

## Why this lever

* **v8F-mini's remaining error is T's code error, shared by every trained model.** MEASURED 2026-10-25 (`…/2026-10-17-v8f-mini/raw/2026-10-25-ensemble/`):
  * the average of six banked models' plans scores ADE6 2.001 m, ns against the best single model (M2b_s1, 2.055 m);
  * the GT-code oracle scores 0.971 m (T0).
* **Five levers on the frozen tokens have not moved T.**
  * WC (loss weight), PGT (O's exposure), GRID (space) and HIST (time) moved T's latcons RMSE −1 % to +1.2 %.
  * D1 (data balance) was deprioritised for WC's reason: T fits its training set and does not generalise (`RESULT_V8F_WC.md`).
* **The encoder is the input none of them changed.** v8F-mini reads frozen pooled tokens, so no gradient has ever reached the features T decides from (plan §13.5 M1: *"No LoRA (impossible on cached features)"*).
* **Published and inherited support, and its size.** REFe, frozen DINOv3 ViT-L/16 + LoRA (322 M total, 19 M trainable), went from 88.25 to 88.82 PDMS on navtest (plan §3 table, INHERITED from the registry, REG §14.2). That is a modest gain on a different rig.
* ⚠️ **Prediction, ESTIMATED, written down so it can be wrong:** a PASS needs T about 25 % of the way to the GT code (WC's dose curve: ≈ 0.19 m of ADE6 at α = 0.25). No measured lever has moved T by more than 1.2 %. The prior for a PRIMARY PASS is therefore LOW, and the MECHANISM read (does T get more accurate at all?) is the more likely informative outcome.
* **`PREREG_V8F_DSCALE` sets the run order, not this file's bars.**
  * NOT DATA-LIMITED: this arm runs first.
  * DATA-LIMITED: D3 runs first, and this arm is stacked on D3's base under an amendment that names the new base before its data.

## The arm (ONE lever: LoRA on the encoder; everything downstream is M2b's)

* **B = M2b**, the banked runs: seeds 0 and 1, F = 0.1207 m (2F 0.2415), T's latcons RMSE 0.6861 / 0.6846.
* **L = M2b with the frame encoder run on the fly: DINOv3 ViT-L/16, frozen, plus LoRA** (plan §13.2: *"LoRA r 32 on q / v"*):
  * rank 32 on the query and value projections of all 24 blocks; α = 32; dropout 0; B-matrices zero-initialised;
  * if the implementation fuses q, k and v into one linear map, LoRA adapts only the q and v slices;
  * trainable encoder parameters: 32 × (1024 + 1024) × 2 × 24 = **3,145,728**; the instruments assert the exact count.
* **The features L's heads read are produced by the cache builder's exact preprocessing and pooling** (`pool4x10`), applied on the fly to the same frames.
  * Everything else is B's: the clips (4,512, RA-1 holdout withheld), the windows (306,566), the step budget (19,164), the effective batch 64, the held-age rule, hist 4 / hmax 16, the losses and weights, the heads' initialisation seed, and the eval (eval-141, 2,399 windows, 141 episodes, `--ages cycle`).
  * Gradient accumulation is allowed to reach batch 64 (the ViT has no batch statistics, so accumulation is exact up to float order). bf16 autocast and gradient checkpointing are allowed.
* **The LoRA learning rate** is the heads' learning rate unless the instruments' smoke shows divergence. Any change is committed in an amendment BEFORE training, with the smoke's evidence.
* **At inference the LoRA is merged into the encoder's weights**, so the deployed encoder costs exactly what frozen DINOv3-L costs. This arm does not change the 100 ms Thor budget.
* **Admissibility:** the encoder sees images only. Labels (including ego-derived ones) enter only through the training losses, as for B. No ego state and no situation-classifier output reaches the encoder or the goal path.

## Proofs and gates, before any training

1. **Identity control (binding, the arm's anchor):** with the LoRA B-matrices at zero, L must equal B.
   * On real frames, the on-the-fly pooled tokens equal the cached tokens within a tolerance τ. MEASURED over at least 200 train and 200 eval frames: the max and mean abs difference.
   * A banked M2b checkpoint evaluated through the on-the-fly path reproduces its banked per-window ADE6 within ε.
   * τ and ε are committed in an amendment from the measurement, before any training.
   * A mutation that changes the preprocessing (one resize or normalisation constant) must break the identity control.
2. **Reachability (CLAUDE.md, "built, tested, and unreachable"):** the flag → config → model → encoder → the LoRA modules in the forward → loss → `.backward()`, `file:line` at each hop.
   * One real-data step: a non-zero gradient on every LoRA A- and B-matrix group and on every head, and **exactly 0.0** on the frozen DINOv3 weights.
   * A mutant that detaches the encoder output must turn this red.
3. **G-DVB:** every new flag (encoder mode, rank, α, targets, LoRA learning rate) has its declared-vs-built entry. `built_levers` records them with the exact trainable count.
4. **AR-2 (the plan's M4 gate, its own box-2 go): a 200-step real-data smoke at batch 1, with checkpointing and bf16.**
   * Every declared loss term finite; the gradients of proof 2.
   * The measured step time and peak memory.
   * The projected hours for one 19,164-step seed on box 2 and on an A40, stated as ESTIMATED from the measured step time.
5. **Venue rule, set now:**
   * if the projected box-2 time for one seed is ≤ 36 h, the run is on box 2 (one job, safe mode, the temperature guard), with the PI's go;
   * otherwise it waits for the A40 under AR-1 to AR-4 (plan §13.5).
6. **The launch gate (G-HYG … G-SUITE) passes on the exact commit and argv**, with the encoder's weights' sha256 bound in the token.

## The reading (the instrument and bars of `PREREG_V8F_GRID` / `PREREG_V8F_HIST`, with L in place of G / H)

* **Tier:** T1-class open loop (eval-141). The mechanism read uses the absmask and GT-oracle evals (T0).
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, over episodes; training-seed variance enters through F.
* **PRIMARY**, per seed s: δ_s = ADE6(L_s) − ADE6(B_s); PASS iff δ_s < 0, separated, |δ_s| > 2F.
* **MECHANISM (M-LAT):** T's slot-pooled present-slot latcons RMSE falls ≥ 12.5 % relative to B_s, separated. It decides the reading of a FAIL, never the verdict.
* **Stage 1 (seed 0):** CONTINUE iff seed 0 PASSES; otherwise STOP with the seed-0 verdict. **Stage 2:** the final reading.

| verdict | when | the committed next step |
|---|---|---|
| **SUCCESS** | both seeds pass | L becomes the base: the T1-kin read of L, then D3 on L under its own PREREG |
| **FAIL-MECHANISM-OK** | latcons fell ≥ 12.5 % at every seed read; driving did not move beyond 2F | T got more accurate but O does not convert it: an O-side arm (O's longitudinal head read against T's own code, `RESULT_V8F_PGT.md` §4), under its own PREREG |
| **FAIL-NO-MECHANISM** | latcons did not fall ≥ 12.5 % | the frozen-plus-LoRA representation is not T's limit at this data scale: D3 if not already run, else T's reader (the plan's §13.6 tactical branch: the tactical heads read the WM latent z²), under its own PREREG |

* **Reported beside, deciding nothing:**
  * the four families;
  * T's present-slot RMSE per group and per latcons slot;
  * T's training constraint loss over the last 10 % of steps, L against B (does LoRA fit the training set more and generalise no better?);
  * Λ and the GT-code oracle ADE6;
  * the LoRA update's size, ‖BA‖_F / ‖W‖_F per block.
* **Refusals (rc 2):**
  * L's argv is not B's plus the LoRA flags;
  * the built rank, α, targets or trainable count differ from the above;
  * no identity-control file, or one outside its committed τ / ε;
  * the encoder weights' sha256 is not the cache builder's;
  * runs not on one clip list or step count;
  * evals not on one grid;
  * GT codes that differ;
  * any loss weight different from B's (`w_diff`).

## Amendment 1 (2026-10-25, written with the instruments, before any LoRA training or smoke)

**MEASURED by the instruments** (`…/2026-10-17-v8f-mini/raw/2026-10-25-lora/identity_control.{py,json,out}`, CPU):

* **The cache encoding, from source.**
  * The checkpoint is `facebook/dinov3-vitl16-pretrain-lvd1689m`; its safetensors sha256 `dcb2e451…8179` equals the builder's.
  * Cache step g is raw frame 2g. Preprocessing is (x.bf16 / 255 − mean) / std in bf16, with no resize: 256 × 640 frames give 16 × 40 patches. CLS and the 4 register tokens are dropped.
  * Pooling to 4 × 10 is in fp32, then the tokens are stored as fp8 e4m3fn. "fp8" is the storage dtype only; the encoder runs in bf16.
  * A window of M2b reads up to 16 frames (hmax 16), so a batch of 64 is up to 1,024 encoder forward and backward passes.
* **Same-device identity is EXACT.** On the same device, the LoRA module at B = 0 equals the builder's `encode_episode`: max abs 0.0 on the field and on the pooled fp8 payload (`SAME_DEVICE_bf16_cpu`).
* **Against the STORED caches it is not exact, and the train cache differs most.**
  * Eval frames, fp32 CPU, pooled fp8 vs the eval cache: rel-L2 3.7e-5 (`E_POOLED_fp32_cpu`). The same in bf16 on CPU: 1.3e-2 unquantised (`E_FIELD_bf16_cpu`).
  * Eval frames rebuilt natively at 256: 3.65e-2, 60 % bitwise equal.
  * **Train frames rebuilt natively at 256 vs the train cache, which was built through the `thor416` route on GPU in bf16:** rel-L2 7.9e-2 and 9.0e-2 (two clips × 12 frames), 32 % bitwise equal (`ROUTE_rebuilt_native256`).

**What this changes (committed now):**

1. **The frames L trains on must reproduce B's TRAIN cache, or the arm is two levers** (LoRA + a pixel-route shift).
   * Before the AR-2 smoke, the route that built the train cache is reproduced: the script, the frame size, the rectification.
   * A pooled-fp8 comparison against the train cache on ≥ 2 clips is then MEASURED on CPU fp32.
2. **τ, set now from the same-device and device-noise measurements:**
   * **τ_route = rel-L2 1e-3** on pooled fp8 against the stored cache, CPU fp32. That is about 27× the eval route's measured 3.7e-5, and 79× below the native-256 train gap.
   * **τ_device:** on box 2's GPU (the device that built the train cache), the AR-2 smoke's on-the-fly B = 0 features against the train cache must reach rel-L2 ≤ 1e-3 too.
   * Same device and same route must stay EXACT (0.0), as measured.
3. **If the train route cannot be reproduced within τ_route, the base is rebuilt instead of the arm being weakened.**
   * The train cache is re-encoded from the on-the-fly route with the frozen encoder, and B' = M2b is retrained on it at seeds 0 and 1, with F' from them.
   * L is then read against B'. This needs its own box-2 go (≈ 6–16 h encode, ESTIMATED from `D3_SCOPING.md`, plus 2 × 4.4 h).
4. **ε (eval):** a banked M2b checkpoint scored through the merged-encoder eval cache at LoRA = 0 reproduces its banked per-window ADE6 with a max |Δ| ≤ 1e-3 m. The eval cache is re-encoded on box 2's GPU at the merged weights.
5. **Instrument defaults, accepted as built:**
   * gradient-norm clipping by group (the heads by B's rule; LoRA by its own norm at the same max-norm);
   * LoRA weight decay = `--wd`;
   * A-matrices Kaiming-initialised from a PRIVATE generator, so the heads' initialisation stream is B's exactly;
   * the fp8 storage quantisation kept in training as a straight-through estimator, so the heads see the fp8 grid the cache gives B;
   * G-LIVE on 16 windows.
6. **Accumulation is gradient caching, not plain accumulation.**
   * Plain micro-batch accumulation is INEXACT for M2b, because its WM loss reads batch-level statistics (the instruments prove it).
   * The arm encodes each micro-batch, computes the full-batch-64 loss on the features, and back-propagates the feature gradients through the encoder micro-batch by micro-batch. That is exact up to float order.
7. **Data and venue (a PI decision, stated now):**
   * on-the-fly training needs PNG frames for all 4,512 train clips, ≈ 160 GB ESTIMATED;
   * box 2's D: has ≈ 81 GB free (MEASURED 2026-10-25).
   * So a full run needs a disk provision on box 2, or the A40 with the frames on HF private (plan §13.5 AR-4).
   * The venue rule above (≤ 36 h per seed projected) still applies once AR-2 measures the step time.
* **Nothing else changes:** the bars, the estimator, the stage rule, the decision table, the refusals.
