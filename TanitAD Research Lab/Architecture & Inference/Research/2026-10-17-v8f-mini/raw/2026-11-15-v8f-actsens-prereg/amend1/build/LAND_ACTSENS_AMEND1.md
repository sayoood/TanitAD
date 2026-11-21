# LAND_ACTSENS_AMEND1 (Master Mind, 2026-11-15): code of record for AMEND_ACTSENS_1

* **Code of record: `81cee6c1760068cf76c6f0465d1fffd9bc1d7215`** on `agent/v8f-actsens-20261115`, a fast-forward child of `6b2bca1f` (pushed, verified by `ls-remote`). It changes two files, the trainer and `test_v8f_actsens.py`, and both carry the blobs of the agent's tested throwaway commit `d0303505`.
* **MM review:**
  * Trainer diff read in full. `wm_only_live_check` keeps the step-0 terms, and `main()` writes `sc["live_first"]` only when `--wm-idm-gamma` is named, so `live` keeps its key set.
  * Checker diff read. C3 compares IDM's `live_first` with REF's `live_first`. A missing or refused REF refuses the IDM run, and REF can never train. REF's `live.terms` must equal R35's banked terms on every key.
* **Tests:**
  * MM re-run: `test_v8f_actsens.py` 20 passed.
  * Agent: actsens + divrs + ladder 67 passed; mutation MA1 (recording the last step) goes red.
  * Package: UNIT 633/0, DRYRUN 122/0.
* **Ship check against the code of record:** 42 members, tar sha256 `f2c92f68`, equal to the agent's check against its throwaway repo.
* **UNVERIFIED** until the first real gate on box 2: the real REF selfcheck carrying `live_first` under the names the checker reads; R35_s1 reproducing its banked terms (only s0 was measured, 25 of 25).
