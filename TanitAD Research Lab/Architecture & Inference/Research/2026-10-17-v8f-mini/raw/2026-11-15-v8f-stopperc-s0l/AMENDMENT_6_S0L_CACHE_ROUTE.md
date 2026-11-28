# AMENDMENT 6 to S0-L (Master Mind, 2026-11-15, written before any L_s0 dump, STOPDEC or read): the L eval cache of record is the TRAINED encoder path

**What changes.** README_S0L.md step 3 said: fold L_s0's LoRA into DINOv3-L, then re-encode eval-141 with the B-prime cache's own pipeline at the merged weights.
The cache of record for S0-L is now the **trained encoder module, adapters unmerged** (`tanitad.models.v8f_lora_encoder`: frozen bf16 base plus the fp32 LoRA branch added in bf16), run through the same B-prime pipeline (`/root/identity_pod/cache_pod.py`, unedited: decode, normalise, batches of 8, pool 4x10 / 8x20 in fp32, fp8 e4m3fn), on the same pod GPU (RTX 4000 Ada). The folded cache is built too and read **beside**, not deciding.

**Why (MEASURED on the pod, 2026-11-15, `F:/TanitAD-scratch/s0l_read/step3/cache_lmerged.py`, checks in each cache's `lmerge_checks.json`):**

| check | result |
|---|---|
| fold route, B = 0 (`zero`) | 0 tensors changed; 3 eval clips bitwise equal to the B-prime cache (`cache_pod/eval`) |
| fold route, L_s0 (`merged`) | exactly 48 tensors changed (q and v of 24 blocks) |
| c2: folded vs trained module, 8 frames of clip 01be5919, pooled 4x10 fp32 | **rel-L2 0.0497**, max abs 0.139 |
| scale: trained module vs frozen, same frames | rel-L2 1.404 (the LoRA moved the features by more than their own norm) |
| c4: the same fold in fp32 vs the trained module in fp32 | rel-L2 4.8e-6, max abs 1.3e-5: the fold arithmetic is right, and the 0.0497 is the bf16 rounding of W + BA |
| trained route, B = 0 (`lora-zero`) | the module's frozen weights equal the builder's bit for bit; 3 eval clips bitwise equal to the B-prime cache |

The heads of L_s0 were trained on the trained module's output. A cache 5 % (rel-L2) away from it would feed L's heads inputs they never saw and bias S0-L toward NO-GAIN. The trained route keeps every pipeline step of B-prime's cache and changes only the encoder's weights, which is what S0-L asks.

**Unchanged:** the CI rule (Amendment 5: GAIN iff paired dAUC point > 0.037 and CI lower > 0), the controls, the steps 4 to 9, one seed per arm, and what a GAIN licenses (L-DAG proceeds; never a driving-quality claim). The fold's bf16 precision is a deployment question (fold in fp32 or keep the adapters), reported beside.
