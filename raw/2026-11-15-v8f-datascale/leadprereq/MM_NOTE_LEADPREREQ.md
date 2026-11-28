# MM note: lead-head prerequisites on the union tree (reach proof and label clock), 2026-11-19

These are MEASURED checks (ours), selfcheck-only. No model was trained and no gate token was involved. The worker's handoff is `HANDOFF_LEADPREREQ.md`.

## (a) Reach proof: PASS for both wirings (`add` and `token`)
The proof ran the trainer `--selfcheck-only` on tree cdbff971, with UE_s1's argv plus `--lead-head on --w-lead 3.362`, on the union lead labels.

| field | add | token |
|---|---|---|
| record | `leadreach/add/selfcheck.json` (sha256 49a65c97) | `leadreach/token/selfcheck.json` (sha256 52cb4a00) |
| lead term at step 0 | 0.7074 | 0.7074 |
| head gradient | 170.3, nonzero on both the presence row and the gap row | same |
| detach control at w 0 | exactly 0.0 | exactly 0.0 |

* **Step-0 lead share** is 7.29 % at w 3.362, LLR's frozen weight (H-LEVELHEAD-2). Under the 10 % rule of `PREREG_LEVELHEAD_A2`, w would be 4.747 on this corpus. That number is printed, not adopted: a dose change goes to the PI.
* **Union labels:** 13,928 of 14,512 clips carry lead labels, with n_near 383,889.

## (b) Label clock: best shift 0 on both files
Results: `clockcheck.json` (sha256 19b87a85).

* The check compares the label speed against the cache speed on 60 clips per file.
* The median residual at shift 0 is 0.0008 m/s (10k file) and 0.0013 m/s (corpus file).
* Every shifted arm reads 30 to 80 times worse.
* The gap, time-gap and arc fields share the speed's clock by construction. This was read from source, not separately measured.

## Not built
There is no closing-speed field. The handoff's section on it says what an honest derivation would need.
