# PREREG R8-PICKER, ADDENDUM 2 (K1 as first coded FAILED; the discriminating control; the K1 reading rule)

* **Clock read before sealing:** 2026-10-21 21:22:28 +0200 (Berlin, dev-box clock).
* **Parent:** PREREG_R8_PICKER.md, sha256 e6ec30e7…. Addendum 1 is 994bd849….
* **When sealed:** BEFORE any GPU capture roll. `runs/` is empty. The only rolls that exist are the CPU smoke's.

## What happened (MEASURED, `smoke/`)

**K1 as first coded FAILED** on the seed-0 FULL-ARM smoke: the wrapped roll against the first unwrapped roll.

* The difference is in 62 float arrays. The max |Δ| is **6.1e-5 m**, on `nav_true.traj`. The tacv6 logits differ by
  about 2e-6, which places the difference INSIDE the forward.
* All integer arrays are identical, so every pick, `sel_idx` and `sel_idx_base` is the same.
* The seed-1 `--os-only` smoke passed bit-identically: 5 files, 124 arrays.

**The discriminating control** was a second UNWRAPPED seed-0 roll, `smoke/unwrapped2/s0`, with the same argv and
device.

| comparison | float arrays differing | max \|Δ\| | integer arrays differing |
|---|---|---|---|
| unwrapped vs unwrapped2 | 62 | 6.1e-5 m | 0 |
| **wrapped vs unwrapped2** | **0** (bit-identical) | 0 | 0 |

**Reading.**

* The full-arm CPU roll is NOT bit-reproducible run to run: two unwrapped runs differ by 6.1e-5 m. This is most likely
  thread-scheduling-dependent float reduction order, because the first unwrapped run overlapped two other CPU jobs.
* The wrapper is inert. It is bit-identical to an unwrapped run, and its picks equal those of every run.
* K1 "bit-identical against ONE unwrapped run" was therefore an unsound control for the full arm. It could fail
  without any wrapper defect. This is reported as a FAILED control as first coded. It is not hidden.

## K1 reading rule (applies to the smoke as banked; nothing in the outcome bars changes)

K1 PASSES iff all three hold:

* the wrapped roll is bit-identical (equal_nan) to AT LEAST ONE unwrapped roll of the same form;
* every integer array (the picks) is identical across ALL unwrapped rolls;
* the wrapped roll's max float |Δ| against every unwrapped roll is no larger than the unwrapped-vs-unwrapped envelope.

The envelope (6.1e-5 m) is reported beside. It is far below the effect sizes this prereg reads, which are about 0.016 to
0.031 m at 2 s. It is also far below K7's cross-device drift (median 1.0 mm on the fanceil s1).

## Consequence for the GPU rolls

None. K2 (same device vs the fanceil Thor s1) and K7 (vs the bank) are agreement bars of ≥ 0.99, not bit-identity.
Every picker − SHIP cell is computed inside ONE roll, so run-to-run float noise cannot enter a paired difference.
