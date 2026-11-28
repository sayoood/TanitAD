# PREREG_V8F_LORA AR-2, run 2 (box 2, 2026-10-25 06:55–07:10): identity PASS, smoke ran, AR-2 FAILED on the training path's identity probe

`code/box2_ops/chain_v8f_lora_ar2.sh` ran at tip `771a4eac`, with the generator fix, on the PI's go.

* **Same-device identity control: PASS, bit for bit.** Rel L2 is 0.0 and 100 % of tokens are bitwise equal on all 6 clips (NVIDIA GeForce RTX 4060). The mutant `PIXEL_SCALE_255_to_256` reads 0.028 and is caught (`identity_read.json`, `identity/`).
* **Smoke: 200 steps at batch 1, bf16, `--amp` off, `--enc-ckpt`.**
  * Step time 2.02 s, peak 1.65 GiB allocated / 1.93 GiB reserved (`smoke/smoke_report.json`, `smoke/smoke_steps.jsonl`).
  * Projected 494.3 h per 19,164-step seed at effective batch 64 (ESTIMATED), so the venue rule (≤ 36 h per seed on box 2) names the A40.
* **AR-2: FAILED, as committed** (`ar2_read.json`). The TRAINING path's own identity probe reads rel L2 2.41e-2 against τ_device = 1e-3 (`smoke/run/selfcheck.json` → `identity_probe`).
  * Its fields: 75.6 % of 655,360 values bitwise equal; max |Δ| 0.25 (one fp8 e4m3 step at that magnitude); mean |Δ| 0.0019; 9 unique frames.
  * That is the signature of fp8 rounding-boundary flips from small bf16 differences, not of a systematic error.
  * The probe encoded its 9 frames in one call. The builder and the identity driver encode in batches of 8.
* **The next step is `PREREG_V8F_LORA.md` Amendment 3:** builder-shaped encoder calls of exactly 8 frames on the training path, then an AR-2 rerun.
* Run 2's files keep their canonical names. Run 1's are under `../run1_genbug/`.
