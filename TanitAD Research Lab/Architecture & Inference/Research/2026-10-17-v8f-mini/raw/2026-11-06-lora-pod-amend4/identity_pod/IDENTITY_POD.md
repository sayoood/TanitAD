# IDENTITY_POD: token-level identity of the pod encoder against box 2's stored cache (2026-11-06)

Evidence class: **MEASURED** on the pod GPU (`IDENTITY_POD.json`, `run.log`, scripts beside this file). Instrument: the AR-2 identity driver's own functions (sha256 `fec46546...841d`, the banked run 3 driver), the arm's `v8f_lora_encoder` at B = 0, and the cache builder's `encode_episode`, in one process. Eval tier: feature identity only (no driving claim).

## Result (the rule fixed before the measurement)

**B' ROUTE.** B KEPT needed `b vs c` rel-L2 <= 1e-3 on all 3 clips. It reads 2.42e-02, 2.63e-02, 2.25e-02 (8f387977, bf5ec76e, dca157a0): 22x to 26x over the bar on every clip. The builder path gives the same numbers bit for bit, so the verdict does not depend on which encoder path is read.

* a = dev-box banked v2ep frames (sha256 equal to `banked_hashes.json`, 6 of 6), b = pod-built v2ep (size and sha256 equal to the pod ledger), c = box 2's stored pooled fp8 tokens (`cache_full`, md5 equal to the manifest rows, 6 of 6). Pod GPU: RTX 4000 Ada (sm_89); box 2: RTX 4060 (sm_89). torch 2.11.0+cu128 on both, transformers 5.15.0 on both, bf16 weights (sha256 `dcb2e451...8179`), batch 8, `--amp` off, pooled 4 x 10, fp8 e4m3fn, all 101 cache frames of each clip.
* The three decision clips are the first three groups. The other three clips (supplementary, not part of the rule) read the same.

## Per clip (MODULE path = the arm's encoder at B = 0; rel-L2 on float64 norms of the fp8 values)

| clip | split | comparison | rel-L2 | max abs | bitwise-equal values | bitwise-equal tokens (1,024-vectors) |
|---|---|---|---|---|---|---|
| 8f387977 | train | a vs c (device) | 2.115e-02 | 0.25 | 80.2 % | 0.0 % |
| 8f387977 | train | b vs c (device + Linux route) | 2.425e-02 | 0.25 | 75.9 % | 0.0 % |
| 8f387977 | train | a vs b (route alone) | 2.421e-02 | 0.25 | 75.9 % | 0.0 % |
| bf5ec76e | train | a vs c (device) | 2.248e-02 | 0.25 | 78.8 % | 0.0 % |
| bf5ec76e | train | b vs c (device + Linux route) | 2.635e-02 | 0.25 | 73.3 % | 0.0 % |
| bf5ec76e | train | a vs b (route alone) | 2.615e-02 | 0.25 | 73.5 % | 0.0 % |
| dca157a0 | eval | a vs c (device) | 2.145e-02 | 0.25 | 79.8 % | 0.0 % |
| dca157a0 | eval | b vs c (device + Linux route) | 2.254e-02 | 0.25 | 78.2 % | 0.0 % |
| dca157a0 | eval | a vs b (route alone) | 2.271e-02 | 0.25 | 78.0 % | 0.0 % |
| 0e1526b3 (suppl.) | train | a vs c (device) | 2.342e-02 | 0.25 | 76.7 % | 0.0 % |
| 0e1526b3 (suppl.) | train | b vs c (device + Linux route) | 2.503e-02 | 0.25 | 74.3 % | 0.0 % |
| 0e1526b3 (suppl.) | train | a vs b (route alone) | 2.522e-02 | 0.25 | 74.2 % | 0.0 % |
| 83e32d0f (suppl.) | train | a vs c (device) | 2.233e-02 | 0.25 | 78.8 % | 0.0 % |
| 83e32d0f (suppl.) | train | b vs c (device + Linux route) | 2.446e-02 | 0.25 | 75.4 % | 0.0 % |
| 83e32d0f (suppl.) | train | a vs b (route alone) | 2.465e-02 | 0.25 | 75.2 % | 0.0 % |
| e6cb7cd4 (suppl.) | eval | a vs c (device) | 2.091e-02 | 0.25 | 80.0 % | 0.0 % |
| e6cb7cd4 (suppl.) | eval | b vs c (device + Linux route) | 2.203e-02 | 0.25 | 78.3 % | 0.0 % |
| e6cb7cd4 (suppl.) | eval | a vs b (route alone) | 2.219e-02 | 0.25 | 78.1 % | 0.0 % |

`max abs 0.25` is one fp8 e4m3 step at that magnitude. 0.0 % of whole tokens are bitwise equal in every cell because a token has 1,024 values and 20 to 27 % of values differ.

## What the three comparisons say

1. **The device alone already fails.** `a vs c` (byte-identical banked frames; only the GPU differs) reads 2.1e-2 to 2.3e-2, about 21x to 23x the bar. So frames built on a Windows host and shipped to the pod (the handoff's option B) would not rescue identity on this GPU: the miss does not come from the Linux frame route.
2. **The route alone is the same size.** `a vs b` (same pod GPU, banked vs pod frames; the frames differ by +-1 level on 0.6 to 1.6 % of pixel values in all 101 frames) reads 2.2e-2 to 2.6e-2. Before the fp8 rounding the pooled fp32 field differs by only 0.9e-2 to 1.3e-2; the fp8 grid flips rounding boundaries and roughly doubles it. `b vs c` is the two effects together (2.2e-2 to 2.6e-2).
3. **The floor is the fp8 token grid, not a defect.** On the same device and the same frames, changing the encoder batch from 8 to 16 reads 2.21e-02 (79.1 % bitwise-equal values) and the AR-2 pixel-scale mutant (255 to 256) reads 2.66e-02 (`ident_extra.json`, clip 8f387977). So any last-ULP change of the bf16 forward lands near 2e-2 after the fp8 rounding. tau_device = 1e-3 can only be met bit for bit, on the same device class and kernels.

## Controls

* Re-encoding the same frames twice on the pod GPU is bit-equal on all 6 clips (rel-L2 0.0, 100 % bitwise): True. The pod encoder is deterministic run to run, so every number above is a device or input effect, not noise.
* The arm's encoder at B = 0 equals the builder's `encode_episode` path bit for bit on this GPU, on both frame sets: True. My local copy of the driver's steps equals the driver's own `encode_pooled`: True.
* The AR-2 pixel-scale mutant against box 2's cache reads 2.67e-02. That control, as written, is satisfied by the device effect alone (about 2e-2), so it does not show sensitivity on this device. The same-device reading above (2.66e-02 against the unmutated pod encode) is the sensitivity evidence.
* The pod toolchain: `/root/venv` torch 2.11.0+cu128 untouched (asserted in the script); transformers 5.15.0 and huggingface_hub 1.33.0 live in an overlay directory (`/root/identity_pod/site`) used only by these scripts, so the running frame build's environment was not changed.

## Throughput of the real encoder on the pod GPU (no grad, batch 8, bf16, 120 timed batches after 15 warm-up)

| path | ms per frame | frames per s | peak allocated | 470k frames |
|---|---|---|---|---|
| builder model forward (the B' re-encode path), GPU only | 10.56 | 94.7 | 1.28 GiB | 1.38 h |
| same, plus the fp32 copy to the CPU the builder does | 10.84 | 92.3 | 1.28 GiB | 1.42 h |
| arm's LoRA module at B = 0, forward only | 12.73 | 78.5 | 1.29 GiB | 1.66 h |

One clip through `encode_episode` (PNG decode included) took 1.88 s (18.6 ms per frame) and the CPU pooling of both grids plus the fp8 cast 0.43 s, measured while the 13-worker frame build held the CPU quota. The B' re-encode is GPU-light and CPU-bound: the 470k-frame GPU time is about 1.4 h, the wall time is set by PNG decode and pooling. The detached job measures it end to end (about 2.8 s per clip under load).

## The trailing re-encode (item 7), smoke

* Two clips (8f387977, dca157a0) were encoded into `/root/data/cache_pod_smoke` by `cache_pod.py`: 3.1 to 3.8 s per clip while the build ran. M2b's own `ClipStore` and `read_manifest` (stack tip, CPU) open the output: `feats` shape (101, 40, 1024), manifest row keys equal box 2's, file keys, dtypes, shapes and `meta` dict equal box 2's, file size equal (20,904,597 B). `v`, `a`, `steer`, `pose_grid`, `kin3`, `p10` are exactly equal to box 2's; `yaw_rate` differs by at most 4.7e-10 on one clip (Linux vs Windows float op, not read in the steer kinematics mode); `mean` differs by 0.014 (it is encoder output). See `loader_check_smoke.json`, `smoke_compare_box2_*.jsonl`.

## What this means for the plan

* The card's committed fallback (Amendment 1 item 3, kept by Amendment 2) applies as written: the train and eval caches are re-encoded on this GPU from the pod frames and M2b is retrained at seeds 0 and 1 on them; F' comes from B'. The detached job is that re-encode.
* Amendment 4 should name two facts: the venue (RTX 4000 Ada, torch 2.11.0+cu128, transformers 5.15.0) and the Linux CPU frame route of the pod, with the measured sizes above. No tau, bar or estimator moves.
* Under B' the base and the arm share the device, the frames and the encoder numerics by construction, so L stays one lever against B'.
