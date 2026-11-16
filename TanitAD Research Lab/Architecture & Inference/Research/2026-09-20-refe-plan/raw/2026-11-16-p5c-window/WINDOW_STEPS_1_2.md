# P5-C pod window, steps 1–2 (A3.1 §4): chain C on the pod, and the join re-run. MM, 2026-11-16

Evidence class: **MEASURED**, from the artifacts in this directory. Everything they reference is banked on HF `Sayood/tanitad-refe-scorer1p6` `evidence/2026-11-16-rj5-chainc-join/` (`ZZJOINC_BANK_VERIFIED`).

## Step 1 · Chain C on the pod (`a31/pod_rj5c/rj5c_pod.sh`, blob `b3619c0b`)
- **Ran:** 06:25Z–08:07Z after `ZZP7_DONE`, with no trainer on the pod. 192 logs on 14 procs under the 15-CPU quota, no `--fetch`.
- **Pins:** all 6 ship files were sha-verified on the pod. The dry run was clean, and the run ended with `ZZRJ5C_DONE`.
- **Rows:** `run_fetch_C/rows` (60 jsonl), `run_control36/rows` (14 jsonl). Tar `rj5c_rows.tar` sha256 `71a5a4b2…` was pulled binary-clean and its sha checked.

## Step 2 · Join re-run with C (`rj5_bank_join.py`, blob `4713739f`)
- **Command:** `--min-coverage 0.99`, over local / A / B / C. Result: `ZZJOIN_OK`.
- **Coverage:** **1.0** (92,896 / 92,896 rank-0 keys), 0 conflicts, 0 knob mismatches, 0 rows not in the bank. C won 13,949 keys.
- **Inputs:** all 771 local / A / B input files are sha-identical to the banked join (`raw/2026-11-15-rj5-label-path/rj5_bank_join_manifest.json`); the 60 new files are all C. The bank sha `13459760…` is unchanged.
- **`P5C_JOIN_SHA256` = `4670f88ac409e71a1f7e1bf02328d93229fb3e153ad63f4127f08d53bbfed739`** (`rj5_bank_join.jsonl.gz`, 67,678,735 B).

## Pod R0 control (A3 §7): OUTCOME B
- **Result:** the [H] prediction, "pod R0 == bank bit-for-bit where the dev box split by 1 ULP", is **REFUTED**.

| | logs | rows over 1e-6 m | max dxy |
|---|---|---|---|
| CPU-built bank rows, 35 bulk control logs | 30 of 32 diverge, 2 exact | 1,956 of 2,910 | 0.746 m |
| CUDA-built bank rows (split out per A3 §7) | 4 | 476 of 480 | — |
| chain C's own every-20th R0 | 10 of 10 diverge | 530 of 954 | 0.611 m |

- **Cross-platform:** the 2 pod-exact logs were dev-box-diverged, and every dev-box-exact log diverges on the pod.
- ⇒ **The bank builder's numerics differ from both platforms.** Under A3 §7's committed ruling this is a floor, not a bias: neither branch blocks C1, and the floor is disclosed per log (TB-8).
- ⇒ **Implication for C1:** the RJ5 labels of the 13,949 pod-rolled keys sit on a pod-specific floor, as the dev-box-rolled keys sit on a dev-box floor. This is a reproducibility bound on "RJ5 minus bank" per row, not a label error.
- Script: `pod_r0_control.py`. Output: `pod_r0_control.json`.
