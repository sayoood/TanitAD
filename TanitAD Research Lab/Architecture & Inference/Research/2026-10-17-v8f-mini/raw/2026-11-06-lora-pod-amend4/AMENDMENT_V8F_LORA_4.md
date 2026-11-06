# PREREG_V8F_LORA, Amendment 4: the venue, the base B′ and the frame route (Master Mind, 2026-11-06 22:36 dev-box clock; a dated amendment beside the card, written before any B′ or LoRA training step)

The card is `Project Steering/PREREG_V8F_LORA.md`, with Amendments 1 to 3. This amendment changes no bar, no estimator, no stage rule, no decision row and no refusal.

## 0. What has been seen before this amendment (disclosure)
* **The venue.** The PI provided a GPU pod for the run on 2026-11-06. MEASURED: an RTX 4000 Ada, 20 GB (sm_89), with a 15.3-CPU cgroup quota, 71 GB of RAM and 500 GB of container disk. torch is **2.11.0+cu128**, box 2's version. The DINOv3-L safetensors sha256 equals the builder's `dcb2e451…8179`.
* **Frames.** The production builder, run on this Linux pod, reproduces the dev-box banked frames to within ±1 level on 1.377 % of pixel values, in all 201 frames of `8f387977`.
  * The cause is that Linux and Windows CPU `grid_sample` differ: MEASURED. Raw decode and mask are identical on both hosts.
  * `lora_pod/HANDOFF_LORAPOD0.md`.
* **Token identity on this GPU** (`lora_pod/identity_pod/IDENTITY_POD.json`). Pooled fp8 tokens against box 2's stored `cache_full`, rel-L2 per clip:
  * **a vs c**, the device alone (byte-identical banked frames): 2.11e-2, 2.25e-2, 2.14e-2;
  * **b vs c**, the pod frames on this GPU: 2.42e-2, 2.63e-2, 2.25e-2;
  * **a vs b**, the route alone, same GPU: 2.42e-2, 2.62e-2, 2.27e-2;
  * the max abs difference is one fp8 step (0.25) in every case.
  * Controls: a repeated encode is bit-equal; the arm's module at B = 0 equals the builder's `encode_episode` bitwise.
* **The decision rule was fixed before the measurement** (`LOOP_STATE_2026-11-02.md`, the entry headed *pod phase 0 done; frame build RELEASED*, whose stamp 22:35 runs ahead of the dev-box clock: it was written at about 22:05, before the identity agent launched): B is KEPT iff b vs c ≤ τ_device = 1e-3 on all three clips; otherwise the card's committed fallback (Amendment 1 item 3, kept by Amendment 2) runs.
  * **⇒ The B′ route.** b vs c is about 24× the bar.
* **Not seen:** no B′ run, no LoRA step, no LoRA smoke on this GPU, and no row of any arm on the re-encoded cache.

## 1. The amendment
1. **Frames.** Both arms read the pod-built frames (`v2ep_pod.py`, sha256 `75305c08…`, the production builder read and not edited, Linux route) for all 4,713 clips: 4,572 train and 141 eval.
   * The banked-hash mismatches are recorded per clip in the ledger.
   * Because B′ and L read the same frames, the ±1 route difference is not a second lever.
2. **The base B′ = M2b retrained at seeds 0 and 1 on `cache_pod`.**
   * `cache_pod` is the train and eval caches re-encoded on THIS GPU from those frames, by the builder's own `encode_episode`: bf16 weights, batch 8, `--amp` off, `pool4x10`, fp8 e4m3fn.
   * B′'s argv is the banked M2b argv with ONLY the cache roots changed. The step budget (19,164), the clips (4,512 trained, RA-1 holdout withheld), the windows, the losses and weights, the heads' initialisation seed and the eval-141 grid are B's.
   * **F′** is computed from B′_s0 and B′_s1 by the same definition as F. The PRIMARY bar becomes |δ_s| > 2F′, with δ_s = ADE6(L_s) − ADE6(B′_s).
   * The banked B and F stay reported beside, deciding nothing.
3. **The identity control on this device** (the card's proof 1, kept):
   * **at LoRA B = 0, the training path's pooled fp8 tokens equal `cache_pod`'s, rel-L2 ≤ τ_device = 1e-3** on ≥ 2 train and ≥ 2 eval clips. The same device and the same route are expected to read exactly 0.0, as on box 2 in AR-2 run 3.
   * The arm's ε (Amendment 1 item 4) is read against B′_s0: its checkpoint scored through the merged-encoder eval cache at LoRA = 0 reproduces its per-window ADE6 within 1e-3 m.
4. **The smoke on this GPU** (AR-2's content, rerun here): 200 real-data steps.
   * Every loss term finite.
   * A non-zero gradient on every LoRA A- and B-matrix group and every head, **exactly 0.0** on the frozen DINOv3 weights, and a detach mutant that must go RED.
   * The training-path identity probe of item 3.
   * The measured s/step, which becomes the projection of record.
5. **Launch.** B′ and L each start only with a `launch_gate.py` PASS token bound to the exact commit, argv and data manifests, with the encoder sha in L's token.
   * Checkpoints every 1,000 steps, each copied to HF private.
   * The pod has no network volume.
6. **Reported beside, deciding nothing:**
   * the four families, STRATEGIC PAUSED (PI item 25), plus the M1 crawl rows;
   * L's T2 read follows `t2_prereg/DRAFT_PREREG_T2_DOM_LORA.md`, which is frozen before any L checkpoint exists.

## 2. Why this is the committed rule and not a new choice
* Amendment 1 item 3 named this fallback before any data: re-encode the train cache with the frozen encoder and retrain B′ = M2b at seeds 0 and 1. Amendment 2 kept it as the rule if the same-device control fails.
* What is new is only the device and host on which it runs. Both were measured above, before any B′ or L step.
