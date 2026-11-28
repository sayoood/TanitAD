# Master Mind review of the S1 read driver, and Amendment A7 (2026-10-15)

- **Reproduced.** The agent's `test_s1_read.py` on its own `s1_read.py`, on a fresh tip copy (`ed39928`), reads **`ZZTEST_PASS 94`**: 29 checks, 4 real-data checks and 60 RED mutations (`test_orig.log`).
- **The driver's four ambiguities are ruled** by **`eval/PREREG_AUX_DRIVABLE_S1.md` Amendment A7**, written before any H-arm read: H2-s0 was still training, and no H-arm number existed.
  - A7.1 to A7.3 adopt the driver's own reading. Each is the stricter one.
  - A7.2 notes that A3.3 makes the "H1-s0 only" alternative moot.
  - A7.4 adds a **median per-window step-time limb** to K3, because the literal run-mean is contaminated by Master-Mind CPU jobs on the pod during N-s0, N-s1 and H2-s0. The windows are disclosed in A7.4.
- **The landed `eval/s1_read.py` and `eval/test_s1_read.py` are the A7 versions.** `window_rates()` was added, `read_k3` fires on run-mean OR median-window, and `AMBIGUITIES` carry `ruled: A7.x`. They read **`ZZTEST_PASS 97`**, with 3 new RED mutations (`test_a7.log`):
  - `k3_median_limb_off`;
  - `k3_window_not_differenced`;
  - `a7_rulings_dropped`.
- **The originals.** The agent's files in this folder are kept unmodified as the record of what was delivered.
- **Open (backlog, not blocking):**
  - `rl_val_eval.load_scores` keeps a view of the whole score array per row: a MemoryError on real S1 dumps, about 33.5 GB per arm. The driver's shim proves itself equal to the original on every run.
  - Head-arm field names are UNVERIFIED until H2-s0's evidence is on HF. The driver refuses a wrong name, by name.
