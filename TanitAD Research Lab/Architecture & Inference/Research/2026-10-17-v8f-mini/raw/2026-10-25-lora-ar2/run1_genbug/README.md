# PREREG_V8F_LORA AR-2, run 1 (box 2, 2026-10-25 06:14–06:25): identity PASS, smoke crashed on a generator device bug

`code/box2_ops/chain_v8f_lora_ar2.sh` ran at tip `59c27d92` on the PI's go.

* **Frames:** 4 train and 2 eval clips were built on the dev box by the production builder and found equal to the banked production frames tensor by tensor (`frames_vs_banked.json`). They were shipped and sha256-checked on box 2 (`frames_sha256.txt`). The preflight is in `preflight_manifest_only.json`.
* **Identity (Amendment 2, same device):** PASS. Rel L2 is exactly 0.0 and the tokens are 100 % bitwise-equal on all 6 clips (NVIDIA GeForce RTX 4060). The mutant `PIXEL_SCALE_255_to_256` reads 0.028 and is caught (`identity_read.json`, `identity/`).
* **Smoke:** rc 1 about 10 s in. The traceback in `smoke/smoke.log` reads `RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'`: `live_check_lora` built `torch.Generator()` on the CPU for a model on cuda. The CPU live check (`live_check(model.cpu(), ...)`) is correct as it stands.
* **Fix:** `torch.Generator(device=dev)`, one line in `stack/scripts/v8f_mini_train.py`, pinned by `stack/tests/test_v8f_lora_device.py` (an AST check, a control, a mutant). `fixgen_suites.out` holds the v8F suites run on CPU against the fixed tree.
* Run 1's logs were renamed `*.run1_genbug*` on both machines (nothing deleted). The rerun writes the canonical names.
