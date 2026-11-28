# raw/2026-10-22-wc: PREREG_V8F_WC's instruments, proven before any run

* **`mutate_wc.py` / `.out`:** the mutation proof. Every mutation of the lever's path (model, trainer, reader, queue) must turn `stack/tests/test_v8f_wc.py` RED; the unmutated tree is GREEN before and after.
  * Its first run found a real gap: the REAL `trainwc` line could drop `--w-cons` while the dry-run test stayed green. The queue now builds ONE command string per stage, and a static test pins that the real line executes it.
* **`real_queue_probe.py` / `.out`:** the real (non-dry) `trainwc 0` and `gatewc` lines, run once against a fake python that records its argv. Both carry `--arm M2b --w-cons 4.0`.
* **`mk_hold11.py`:** the hold builder for this land.
