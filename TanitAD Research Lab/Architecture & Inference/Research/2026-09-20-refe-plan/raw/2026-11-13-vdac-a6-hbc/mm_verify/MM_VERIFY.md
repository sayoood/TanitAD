# MM verification of the VDAC A6 hold (vdac_a6_hbc), 2026-11-13/14

**What was compared:** two exports of tip `df49fd59`, which equals the landing base for `refe/`: nothing under `refe/` changed between `25b07b01` and the landing tip.
* **pristine** = the tip as is.
* **patched** = the tip with the hold's 7 `refe/` files (6 edited, 1 new) overlaid. This includes the MM's own additions for PREREG_VDAC_P6_A5_A2: stage `A5HCSHUF` in `vdac_pause_p7.sh` and its check in `test_vdac_hbc.py`.

**Which tests:** every `refe/test_*.py` that reads `launch_gate`, plus every `refe/test_vdac_*.py`. They ran on the dev box CPU with `CUDA_VISIBLE_DEVICES=-1`. Evidence class: MEASURED. The files are `res_*.tsv`, `res2.tsv` and `res3.tsv` in this folder, and the logs named below.

## Runs

1. **`res_pristine.tsv` / `res_patched.tsv`: first run, SPOILED.**
   * The two trees ran in PARALLEL, and test temp dirs filled C: to 0.46 GB. That caused ENOSPC failures (in `p6_levers`, `p7_*` and `session_reserve`), and it also ENOSPC-failed another session's pinned run.
   * Kept as evidence only; no verdict is drawn from its ENOSPC rows.
   * Lesson recorded: memory `refe-test-temp-leak`, and the test_p6_levers temp-dir fix (separate land).
2. **`res2.tsv`: rerun of every spoiled test plus all remaining `test_vdac_*`.** SERIAL, pristine then patched per test, TMP on F: and deleted after each test.
3. **`res3.tsv`: rerun on the patched tree of the two tests that hit the 1,500 s timeout (rc 124) under load.** Timeout 3,600 s.

## Verdict: 0 regressions

| test | pristine | patched | reading |
|---|---|---|---|
| test_launch_gate --static | ZZLG_STATIC_PASS 1184 | 1184 | identical |
| test_vdac_gate | PASS | PASS | identical |
| test_vdac_hb | 133 | 133 | identical |
| test_vdac_hb_accum | 40 | **48** | +8 = the hold's HBC 4 x 2 checks |
| test_vdac_hb_h48x | 49 | 49 | identical |
| test_vdac_hbmem | 83 | 83 | identical |
| test_vdac_g12_bound | 15 | 15 | identical |
| **test_vdac_hbc** (new) | absent | **ZZTEST_PASS 97** (2,351 s) | the agent's 96 + the A5HCSHUF check; includes the post-gate RED at the 2 m default and PASS at 3,423,490 |
| test_p6_levers | ZZP6_PASS 199 | **ZZP6_PASS 199** (1,559 s) | identical (res2's patched rc 124 was the 1,500 s timeout, mid-mutations, every mutation RED so far) |
| test_p7_chain / head_wiring / integration | 110 / 182 / 225 | 110 / 182 / 225 | identical |
| test_session_reserve | 43 | 43 | identical |
| test_rl_safeobj | 61 | 61 | identical |
| test_rls1_queue | FAIL in res2 ("returned before S1 was over", 4 s); **PASS 254 alone** (`pristine_rerun_test_rls1_queue.log`) | 254 | A load-sensitive timing scenario, not a tree difference. It runs before any launch_gate import, so the A6 edits cannot reach it. |
| every other launch_gate reader (res_*.tsv) | PASS | PASS | identical counts |

**The same FAIL on both trees, from the export rather than the code (not regressions):**
* `p2r_chain`, `rls1_chain`, `s1aux_chain` and `sft4r_chain`: SHIP_MANIFEST.json and raw files are absent from a partial export.
* `s1z_preflight` and `s1zsm_preflight`: the same cause.
* `p7_ship_list`: `real.json` is absent.
* `vdac_stage0` and `vdac_stage0_forward`: they REFUSE on the exported `eval/s1_read.py` sha256.

**Not run:**
* `stack/tests` launch_gate readers: they read `stack/scripts/launch_gate.py`, a different file.
* The pod's G-SUITE: it is the binding suite check, and it runs in the HC-SHUF chain itself on the pod.
