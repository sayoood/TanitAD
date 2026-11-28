# Pin 5 of the T2 dominance read (2026-11-07)

* **The read:** `read_t2_dominance_pin5.py`, sha256 `56a91667df03523093ce9bb8b0e96282e9621919a8d63c97c58b1ca5574197de`. It is a separate file. Pin 4 (`t2v8f/code/read_t2_dominance.py`, `7fd65ffd...`) is unchanged.
* **What it adds:** `--arm-seeds {0,1 | 0}`. The default `0,1` is pin 4's read, field for field. `0` reads seed 0 only, and its outcome is SEED0-ONLY, never DOMINANT-T2. That is the form the LoRA arm L can be read in while one seed is trained (`PINS_PIN5.txt`, `PIN5_DIFF.txt`).
* **Identity on the real pulls:** `logs/pin5_identity.json` against the read of record, and `logs/pin5_vs_pin4.json` against pin 4's read. Both read identical: true.
* **Tests:** 12 passed (`logs/pytest_pin5_final.log`). The regression run of the frozen tests against pin 5 gives 71 passed (`logs/pytest_regression_on_pin5.log`).
* **Mutations: 23 of 23 RED.**
  * The full run (`logs/mutations_pin5.log`, 04:02) read 21 of 23. All 21 synthetic arms were RED. The two REAL-data arms printed `the mutated read failed rc=1` with an empty stderr.
  * Cause, measured: the harness called a bare `"bash"`. Windows CreateProcess searches System32 before PATH, so it ran WSL's `bash.exe`, which exits 1 with its message in UTF-16 on stdout. The REAL arms never reached the read.
  * A first rerun with the unchanged harness reproduced it (`logs/mutations_pin5_real_rerun.WSLBASH.log`).
  * `run_mutations_pin5.py` now calls `shutil.which("bash")` (Git bash), refuses a System32 path, and prints stdout on failure. Its sha256 is c6e5024fed991a2731add16a1adda00bd80cd8b0b3b8537b2592fe38f3a1f7b4. Only `real_arm` changed, so the synthetic arms are the ones already run.
  * `rerun_real_arms.py` reran the two REAL arms through the harness's own `real_arm` and `REAL_MUT`: **2 of 2 RED** (`logs/mutations_pin5_real_rerun.log`). Seeds swapped changes the read (LATERAL[0] delta 11.378 against 6.263). Seed-1 records read as seed 0 changes the seed-0 block.
* **Left out of this folder:** two zero-byte nohup logs, and the `_aside_pycache` folder.
