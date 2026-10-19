# MANIFEST: L1 Step A run (2026-10-19)

Repo dir (from `paths.json`, which this report follows): `TanitAD Research Lab/Architecture & Inference/Research/2026-10-18-ep-rl-lit/l1_stepA/run_2026-10-19/`. The brief named `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-l1-stepA/`; the existing package already maps to the `ep-rl-lit/l1_stepA/run_2026-10-19/` dir beside the landed `l1_stepA/` (commit `d8b77c1`), so it was kept. `paths.json` format: `[[repo path, absolute source, "raw"], ...]`.

**STAGED, NOT COMMITTED, NOT PUSHED.** Staging area = `C:/Users/Admin/par18/l1run/out/` on this dev box. The Master Mind lands from `paths.json`. Until landed, every row below with source under `C:/Users/Admin/par18/l1run/out/` exists in that ONE place (the run originals of RESULT.json, scores.npz, labels.npz and the gate JSONs are also on `D:` in W; STATE.md, logs and cmd files are also under `C:/Users/Admin/par18/l1run/`).

Text copies marked LF-normalised (`raw/sample.json`, `raw/g0_rerun.json`, `raw/g0_rerun.log`) hold identical content to the CRLF originals in W; the new JSON files are byte-exact copies. Verify with `sha256sum <source>`.

## A. Staged files (35 entries in `paths.json`)

| repo path (under the dir above) | bytes | sha256 | what |
|---|---|---|---|
| `STATE.md` | 6467 | `0790c809100f13bf2da5a2e81c38fde4b270f020bbd01be87064613e99168be4` | UPDATED handoff (the Master Mind's 06:45 / 11:32 / 11:47 updates + this agent's 13:30 update). |
| `launch_stepA_ctype.py` | 11892 | `8b8c7eba0d0f36276644e8e7362aebf25157822feb9c167ac642f0a69d4a68b9` | PATCHED launcher copy (PI decision 2026-10-19 'Allow compute-type check'); the landed launch_stepA.py stays frozen. |
| `probe3.cmd` | 396 | `660e6f5d919df9ac55a085497d46a9b24233e466e6790992729f38f95d3ddeb2` | WMI detached-launch probe. |
| `raw/STEP0_SHA256.txt` | 283 | `1f29847ed5a25b41ec512ff9d37098c437f712cbf61a11fc240ed918700ef7e0` | EARLIER staged (06:20-06:45 package, unchanged). |
| `raw/TREE_HYBRID_PROVENANCE.txt` | 1808 | `1eb79112e443a7e5f0d4d3ab42c4bfa4ab6e780764c754a18e9a170c7604e25c` | EARLIER staged (06:20-06:45 package, unchanged). |
| `raw/bundle_preflight_cpu.log` | 609 | `85cc4764b49b92cc8e953276678a7d6b4ad9e5e335df72088308c442687f01d7` | EARLIER staged (06:20-06:45 package, unchanged). |
| `raw/bundle_preflight_cpu.py` | 2028 | `60c469bc9e2ba575dca4aec3acafe56b87cbfabb0c71dfec084933862057a62b` | EARLIER staged (06:20-06:45 package, unchanged). |
| `raw/g0_rerun.json` | 15724 | `6221132e87e6e874087efd27df882932d0f9c83e630dcca1357bed7afc427f23` | EARLIER. G0 re-run (M-PDM PASS, M-SCENE overlap, control FAILS as required); LF-normalised copy of W/g0_rerun.json. |
| `raw/g0_rerun.log` | 1811 | `92512f482b2ab46f8283966046bc8c524f5ba9b8259352a1a68b3f7a57745aa4` | EARLIER staged (06:20-06:45 package, unchanged). |
| `raw/launch_stepA_ctype.diff` | 4795 | `d5ca50b0ee5aa19fa7921650f5f188b50a283b6afde3187f651e4d6236994eb7` | EARLIER staged (06:20-06:45 package, unchanged). |
| `raw/launcher_ctype_probe_live.json` | 350 | `ae70a5c5e019108c901ab037b16246a78b0ad71f120d1036d698a59bebd984d5` | EARLIER staged (06:20-06:45 package, unchanged). |
| `raw/nvidia_smi_live_20261019.txt` | 4909 | `1b9e9d36081ee4f1818abfb959e63c71034a0c3b10356a0dfc6ad9e31914cbaf` | EARLIER staged (06:20-06:45 package, unchanged). |
| `raw/probe3_wmi_probe.log` | 478 | `27b2a2c0a500cf80fd0a034890ddfaf59c1e4c69d486b2212e1af4e4001d7723` | EARLIER staged (06:20-06:45 package, unchanged). |
| `raw/pytest_launch_stepA_ctype.log` | 101 | `bad063d2d24dbb5c85e96a0b5fa26db8f049a70f01ae040db4f9bf9242ded973` | EARLIER staged (06:20-06:45 package, unchanged). |
| `raw/pytest_original_suite_vs_patched_copy.log` | 99 | `bd60dc26417626b9c8f98600a4253fd9b7fb1ee67f74c33c8e967204d025e2f0` | EARLIER staged (06:20-06:45 package, unchanged). |
| `raw/sample.json` | 9019 | `520bcab5b7baeab52ff52517d097b63699b61ad1f8b670f031526a7ec1b3f48c` | EARLIER. 400-token sample, token-list sha256 211e2640...; LF-normalised copy of W/sample.json (CRLF). |
| `raw/step3_launch_snapshot_0620.log` | 188 | `dee576117e8d94087199c46ec66923c3fb210905399fe3aed8b0201f8a2644b2` | EARLIER staged (06:20-06:45 package, unchanged). |
| `raw/tree_hybrid_MANIFEST.sha256` | 64873 | `844e32cf574165c4dd8417ea68a0f3546ae51de85f0c2451e21d12cd9b5a1e4d` | EARLIER staged (06:20-06:45 package, unchanged). |
| `step3.cmd` | 873 | `ff12d11b2f38acc52e45a404da10eeef2ed8be92ec717cd3f1064ae86ee57d6b` | Step 3 command line (attempt 3 identical to attempts 1-2). |
| `steps4to7.cmd` | 2363 | `7498cc32d3b3381bf267b94c2d8cd736fba64c791b0f93d9aa60fd2a4aafe5ac` | Steps 4-7 command line; refuses itself (exit 9) without the G2b gate. |
| `test_launch_stepA_ctype.py` | 15165 | `64923aba37d9f84fbfa9dcbe2893131e4c224d397d79b11f27fdfb82407d276b` | 41 tests for the patched launcher. |
| `raw/step3_attempt1.log` | 1782 | `5c24fce9303521669a98551c6bb0d84110c924a1974113873c1f5adffa2d8a79` | EARLIER staged (06:20-06:45 package, unchanged). |
| `RESULT.md` | 23729 | `0960e7dcf7bf6e00f2662aee976cd7ed107578c84f14956fb729eb5be6073c7a` | NEW. The Step A result report (this agent). |
| `RESULT.json` | 4158 | `e56306c9bfc2318b9ebdf49f6a56bca203e31d3b429c410d47d0f2f620c49006` | NEW. Step 7 output, byte-exact from W (CRLF). Outcome O2 both rules; A1-A5b with log-cluster CIs. |
| `MANIFEST.md` | - | `-` | NEW. This file (its own sha256 cannot be listed in itself). |
| `derived_tables_stepA.py` | 12279 | `2716f0a6434eee4cef28056909d5ef8d95d88a5ba7f2a56bdaee97d8d2d2b4fe` | NEW. Reproduces RESULT.json (control, max abs diff 0.0), re-checks G2b from the saved bundle, writes the post-hoc A1 tables, pair inventory, numerics bound, tactical CI. CPU, numpy only, ~12 s. Imports `stepA.py` from the run's code dir (= the landed d8b77c1 blob). |
| `make_stepA_bundle.py` | 8036 | `8e544b8676c18c537076a52243df26a761ae299397fc25ec69653eedd40aa237` | PATCHED COPY the run used for attempt 3 (bf16 rounding of visual_ctx, amendment C9). Pre-patch copy == landed d8b77c1 blob (LF sha256 a7ae5559...). Patched LF sha256 8e544b86... |
| `raw/scores.npz.gates.json` | 5834 | `642131d85814d363812c3b53976adcfe011b1b1edb7487e1b9786789c55bdef1` | NEW. G2a / G2c / G3 / G2b_cross_device, byte-exact from W (CRLF). |
| `raw/labels.npz.g1.json` | 623 | `af5c129fe6b3942f6f9794dc7793048165d9c6f80f9be1c5349fc96208e5b2db` | NEW. G1 label identity + control, byte-exact from W (CRLF). |
| `raw/derived_tables_stepA.json` | 12268 | `4222d7417a87d1db0edaeae91bb3ee3a95fd59e21eb76e138589ed4614a4252d` | NEW. Output of derived_tables_stepA.py (C1, C2, T_A1_descriptive, T_tactical, I_pair_inventory, N_numerics_bound, G_summary). |
| `raw/scores.npz` | 3073026 | `d47b11ed729f27a1f314b91c4c21bc9887113d3972c1e8f31e0d097c7299b6a8` | NEW. Banked scorer logits: p_logit [400,96,6], a_logit [400,64,6]. 3.1 MB (<5 MB). |
| `raw/labels.npz` | 2151734 | `7c986dcc736a6dcaf49dfaccd33f3ded9e9c4f2ec49b88f28bb728a71ebf9b27` | NEW. Banked true PDM sub-scores / PDMS of the 38,400 perturbations. 2.2 MB (<5 MB). |
| `raw/step3.log` | 34527 | `a6c518e8d40fc74b8e3d7fd0e200021758f1e2b9e5728c392fd0747cf996a931` | NEW. Full step 3 log, all THREE attempts (attempt 1 rc 1, attempt 2 rc 1, attempt 3 rc 0). |
| `raw/steps4to7.log` | 1915 | `29971d5dcaf3ff52e51bca2afb5763f19aa4ade29d72964650e55bdc1d09e0d2` | NEW. Steps 4-7 log (STEP4..STEP7 markers, ZZANALYSE_OK O2). |
| `raw/make_stepA_bundle_bf16.diff` | 1133 | `7b4a618ad2567b1f202c9a01c459560241656782c2ee90f0888a927cc7453260` | NEW. diff -u of the bf16 fix (pre-patch vs patched). |

## B. On `D:` ONLY (W = `D:/Projects/TanitAD-artifacts/l1_stepA_run/`), over 5 MB so NOT staged

`D:` is an EXTERNAL exFAT drive (CLAUDE.md): a single point of failure. bundle.npz and ctx.npy cannot be rebuilt without the GPU.

| path under W | bytes | sha256 | what |
|---|---|---|---|
| `bundle/bundle.npz` | 8309154 | `77e8584d8a02ee7a93b9239e1e33a16edade75689f36e1f32959960d0e2eb5c4` | The bundle forward: raw fan [400,64,20,3], GPU logits, pick, banked labels. RESULT.json `inputs.bundle`. NOT regenerable without a GPU forward (0.2 h wall under the dev-box lock). |
| `bundle/ctx.npy` | 1572864128 | `8a77a1fcdebf926f2012ad51c52c65fe23f54d729956e27c0c67c1d45e68adf9` | visual_ctx of P2-s0, bf16 bits [400,7680,256]. Equals `ctx.json` `ctx_sha256` (re-hashed today). NOT regenerable without a GPU forward. |
| `bundle/ctx.json` | 357 | `-` | 357 B, ctx shape/hash record. |
| `perts.npz` | 18972106 | `40ecafc88d401463767d74af5d41809d73b36009083e19d62e197f4bfd6bb8d5` | Perturbed plans [400,96,20,3]. RESULT.json `inputs.perts`. Regenerable on CPU from the bundle (checked today: array-for-array equal). |

## C. Run-dir originals that are also staged (byte-exact unless noted)

`RESULT.json` e56306c9bfc2318b9ebdf49f6a56bca203e31d3b429c410d47d0f2f620c49006 ; `scores.npz` d47b11ed729f27a1f314b91c4c21bc9887113d3972c1e8f31e0d097c7299b6a8 ; `labels.npz` 7c986dcc736a6dcaf49dfaccd33f3ded9e9c4f2ec49b88f28bb728a71ebf9b27 ; pre-registration files in W: parent a76f62d8043e4cc9d6c73b35cf2884f27027a1e01bcd3deec658380cc58bd801 (LF), amendment 45f5c7917feef407b3ce18951e35f7abdcef5b7a9fb65e41e9cebd3d50fd72ff, MM adoption 1f60a939ffdb1addf752719f51b3b2fe7e93a24008b92bd23913d5e19e6c7bee (`raw/STEP0_SHA256.txt`; re-hashed today, equal). These three are already in git (`d8b77c1`) and are not re-staged.

## D. Code the run used that is NOT staged because it is byte-identical to the landed `d8b77c1` blobs (LF sha256 compared today)

`stepA.py`, `label_perturbations.py`, `real_scorer.py`, `tts_cem.py`, `test_stepA.py`, `test_launch_stepA.py`, `launch_stepA.py` (frozen; the run used the patched copy above), `g0_overlap.py`, `exposure_diagnostic.py`. Run copies live in `C:/Users/Admin/par18/l1run/code/`.

## E. Reproduce the derived tables (CPU)

```
set CUDA_VISIBLE_DEVICES=-1
python -I derived_tables_stepA.py --run-dir D:/Projects/TanitAD-artifacts/l1_stepA_run --code-dir <the landed l1_stepA dir or the run's code dir> --out derived_tables_stepA.json
```
Prints `ZZDERIVED_OK C1 True C2 True 23 rows`. `C1` = the headline reproduced from the banked arrays against `RESULT.json`; `C2` = G2b re-checked from the saved bundle against the banked P2 table.
