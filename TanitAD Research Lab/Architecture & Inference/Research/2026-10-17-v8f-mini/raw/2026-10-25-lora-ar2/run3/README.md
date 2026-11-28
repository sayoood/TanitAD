# PREREG_V8F_LORA AR-2, run 3 (box 2, 2026-10-25 07:35–07:49): PASS

`code/box2_ops/chain_v8f_lora_ar2.sh` ran at tip `c0769eaa`, with Amendment 3's builder-shaped encoder calls, on the PI's go.

* **Same-device identity control: PASS, bit for bit.** Rel L2 is 0.0 on all 6 clips (RTX 4060). The mutant `PIXEL_SCALE_255_to_256` is caught at 0.028 (`identity_read.json`).
* **The training path's identity probe: rel L2 exactly 0.0, within τ_device = 1e-3** (`smoke/run/selfcheck.json` → `identity_probe`). It read 2.41e-2 in run 2, when it encoded 9 frames in one call. With every call at the builder's 8 frames it is bit for bit the cache. The batch-shape diagnosis of Amendment 3 is therefore MEASURED.
* **Smoke:** 200 steps at batch 1, bf16, `--amp` off, 1.96 s/step, peak 1.44 GiB allocated / 1.63 GiB reserved. A full seed projects to 470.2 h on box 2 (ESTIMATED), so the venue rule names the A40, which needs the PI's go (`ar2_read.json`).
* Run 1 (generator bug) is in `../run1_genbug/` and run 2 (the identity miss) in `../run/`.
