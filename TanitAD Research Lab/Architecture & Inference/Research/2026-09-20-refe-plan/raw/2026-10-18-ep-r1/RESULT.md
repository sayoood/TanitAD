# RESULT: EP-R1 BUILT AND PRE-REGISTERED (not run). A conditional EP row plus a safe-contender pairwise term, fitted head-only on S0's bank, with a hashed PREREG, 46 + 69 green tests, a launch gate and a refusing launcher

Agent for the Master Mind, 2026-10-18 (EP-R1 build). CPU only, nothing fitted, **no outcome statistic computed**, no pod touched, no GPU lock touched. Package target `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-ep-r1/`. **Every file exists in ONE place (this dev box, `C:/Users/Admin/par18/epr1/out/`) until the Master Mind lands it.** Not staged to git, by instruction.

## Integration asks (headline)
1. **Register `PREREG_EPR1.md`** (sha256 `29e79bd61cbbb209bc08aa8b42315736ee47fdaf89abd8ad0731a4d4d4f8f80a`, written in `PREREG_EPR1.sha256` before any outcome exists) and land the package. The two cross-arm decisions in it are yours: (a) **navtest**: it assumes ONE navtest open for the whole REFE head programme (SR-TTC alone, EP-R1 alone, the joint), each against its own pre-registered bar. `D-REFE-EP-2` held the unspent look for the SR-TTC-retrained scorer, so this needs your explicit yes (PREREG section 8). (b) EP-R1 must run at **the same nested fraction as SR-TTC R1's c\***, so both rows are fitted on identical logs.
2. **Not built, named:** `epr1_dev.py` / `epr1_confirm.py` (apply c\*'s rows to the held-out / navtest pre-head features). They need SR-TTC's gate and S3 forwards, which do not exist. If SR-TTC's dev gate fails, S3 never runs and EP-R1's confirm has no navtest features (about 4.86 GPU-h ESTIMATED to produce them: a PI/MM decision).
3. **The dev set is not virgin** for this arm: EP-1 read it descriptively and fitted four rules on it (PREREG section 7). Disclosed, not hidden.
4. Nothing runs before `S0_DONE` (S0 was at 15 of 23 batches). Launch order: land, deploy to `C:\Users\Admin\epr1\code`, `epr1_gate.py --fraction <f> --commit <landed>`, then `launch_epr1.ps1 -Fraction <f>`.

## 1. What was built
| artifact | what it is |
|---|---|
| `PREREG_EPR1.md` + `.sha256` | the design, bars, 14-row (15-outcome) table with a committed next arm per row (section 8), controls C0-C7 |
| `code/epr1.py` | the fit: EP row (column 2, 257 params) of P2's `score_head`; masked BCE anchor on safe proposals; pair term over safe frozen-top-8 pairs with differing EP; 9-config grid, 5-fold grouped-by-log CV; stages `run`, `lc`, `c2` |
| `code/epr1_gate.py` | G-COMMIT, G-DVB, G-PATH, G-BANK, G-LIVE, G-SUITE; HMAC token bound to {commit, argv sha256, S0 manifest sha256, `epr1.py` sha256} |
| `code/launch_epr1.ps1` | refuses without S0_DONE, the manifest, the deployed code, and a verified token; launches the argv the gate printed via WMI |
| `code/test_epr1.py` (46 tests), `code/test_epr1_gate.py` (69 tests) | synthetic banks only; RED mutation arms |
| `code/template_pins.json`, `template_pins_make.py` | 28 generic objects of `srttc_r1.py` pinned byte-identical (PathGuard, every `_io_*`, folds, manifest, frozen-rule helpers, bootstrap, LC call) |
| `code/zero_tail.py` | vendored byte-identical (sha256 `149c3e97...`) so the estimator-equality test is live |
| `code/reach_proof_synthetic.py`, `raw/reach_proof_synthetic.json` | the reach numbers |
| `raw/test_epr1_46.log`, `raw/test_epr1_gate.log`, `raw/calibration_synthetic.log` | the runs |

## 2. Design, in one paragraph (full text: PREREG sections 3, 5, 7, 8)
Target `EP | NC = DAC = 1` (NC = 0.5 is not safe); anchor = mean BCE over safe proposals; pairs among the frozen top-8 with both members safe and `y_i > y_j`; Adam, 10 epochs, 256 sets, grid lr {1e-4, 3e-4, 1e-3} x anchor {0.03, 0.1, 0.3}, two seeds; **c\* = argmax of the seed-mean pooled out-of-fold safe-contender concordance** (the EP analogue of `MM_DECISIONS_R1.md` row 2); a pick-level out-of-fold read (paired bootstrap on c\*) and a violator-leak read are REPORTED and never select c\*. Deployed rule unchanged. Dev bar: >= +0.30 PDMS x100 on both seeds. Navtest primary: EP-R1 alone, lower bound > 0 and point >= +0.30 and no NC/DAC/TTC/C/DDC component or lateral/longitudinal family separated adverse; secondary: EP-R1 + SR-TTC's c\* TTC row, on the same look, not deciding the primary.

**Two design choices the Master Mind may want to overrule before registering:** (i) c\* is chosen on safe-contender concordance (mirror of SR-TTC) rather than on the OOF pick-level PDMS, because the A2.4 learning-curve rule and its delta of 0.010 are defined on concordance; the pick-level read is reported precisely because concordance on the OLD top-8 is blind to the violator leak and to contender-set change. (ii) The pairwise term is restricted to the frozen top-8 (mirror), so it trains ranking only inside the old contender set; the anchor carries the level correction over all 64.

## 3. Evidence that the build is correct (MEASURED, synthetic banks, CPU, `CUDA_VISIBLE_DEVICES=-1`)
- **`test_epr1.py`: Ran 46 tests, OK** (`raw/test_epr1_46.log`). Includes:
  - **Reach (C1/C6):** EP-row gradient > 0 for full / pair-only / anchor-only; **exactly 0.0** on all six frozen rows and on the features; the zero-weight control reads exactly 0.0; a loss that reads column 3 is caught (positive control). `raw/reach_proof_synthetic.json`: full-loss EP-row |grad| 10.86 (weight), 0.0112 (bias); one optimiser step moves the row by 1.0e-3.
  - **C6 conditional-target identity:** gradient bit-identical when unsafe-proposal EP labels change; exactly 0.0 on an all-unsafe batch; changes when safe labels change (the positive control).
  - **Five RED mutation arms**, each deleting one line of `epr1.py` and requiring `ReachabilityError`: pairwise line, feature `detach`, `opt.step()`, **anchor safe mask**, **pair safe mask**. The last two are caught by C6 (i) and (ii) with a same-breath GREEN control on the unmutated module.
  - **The reason the mask exists, measured:** on the synthetic rig, mean label on safe proposals 0.501; a head trained with the mask predicts 0.501 on them, a head trained without it (violators = 0, the unconditional label) predicts 0.398. That under-prediction is the double-charge signature of `D-REFE-EP-1`, reproduced in miniature.
  - Analytic gradients of the masked pair term (8 admitted pairs, two unsafe contenders correctly excluded) and the masked anchor term; lr = 0 reproduces P2's picks bit-for-bit and the pick-level gain reads exactly 0.0; learnable problem (P2 EP row 0.50-0.52 -> fitted 0.81-0.83 over seeds 0-2) and shuffled labels sit at chance (0.50-0.52); determinism; the A2.4 call logic; `boot_draws` equals `zero_tail.Boot`; the AST scan that no file-system call lives outside an `_io_*` function; the path guard refuses every dev-gate / navtest token.
- **Template pins: 28 of 28 generic objects byte-identical** to `srttc_r1.py` at tip `8ebe82d` (template file sha256 `b841347d...`); a one-character drift in `_under` is caught.
- **`test_epr1_gate.py`: Ran 69 tests, OK** (`raw/test_epr1_gate.log`): each check has a regression arm that blocks the token, with a PASSing control; the full gate PASSes end to end on a synthetic world, and again with the REAL `test_epr1.py` as G-SUITE on a clean `git archive` export.
- **Bank header probe (MEASURED, one real unit `L1b95fda46f.sets.npz`, headers only):** `label_order` = `[nc, dac, ep, ttc, c, ddc]` so EP = column 2; `feat` float16 [479,64,256], `logits` float32 [479,64,6], `top8` int16 [479,8], `labels6` float32 [479,64,6]. No value read.

## 4. What was NOT done (plainly)
- No fit, no concordance, no pick-level number, no c\*, no dev or navtest value. S0 is unfinished. Nothing here is a result about H-EPR1.
- Nothing was run on the real bank beyond the header probe; **G-LIVE and G-BANK have only run against synthetic banks.** The real-manifest shape was handled the way the SR-TTC build did (the loader keeps only `*.sets.npz` entries), but the real post-S0 manifest does not exist yet: UNVERIFIED until the first `mint` after `S0_DONE`.
- `epr1_dev.py`, `epr1_confirm.py`, the real-bank C2/C4 reads, `families6` on the new picks: not built (PREREG section 11).
- The launcher was tested for its refusals and its held WMI command line only; it was never run to launch.
- Fresh gate KEY: the HMAC key file is created on first mint on the dev box (`~/.config/tanitad/launch_gate.key` or `TANITAD_GATE_KEY_FILE`), outside the code tree.

## 5. Proposed register rows (not edited; the Master Mind lands them)
| **H-EPR1** (2026-10-18) | **A head-only retrain of P2's EP output as a conditional expectation (EP given NC = DAC = 1) plus a pairwise ranking term among safe frozen-top-8 contenders raises navtest PDMS by >= +0.30 (lower bound > 0, both seeds) with no NC/DAC/TTC component separated adverse.** PRE-REGISTERED, NOT RUN. `PREREG_EPR1.md` sha256 in `PREREG_EPR1.sha256`; built and gated (`epr1.py`, `epr1_gate.py`, 46 + 69 green synthetic tests, five RED mutation arms); S0 unfinished; the dev gate shares SR-TTC's held-out table and is not virgin for this arm. `raw/2026-10-18-ep-r1/` | **HYPOTHESIS** (design); the build is MEASURED on synthetic banks only. |

Draft BACKLOG line: `EP-R1 | BUILT + PRE-REGISTERED 2026-10-18, 0 GPU (raw/2026-10-18-ep-r1/): fit on S0's bank after S0_DONE at SR-TTC R1's fraction; gate token then launch_epr1.ps1; dev gate and confirm appliers NOT built | decisions: shared navtest open (PREREG s8), fraction = SR-TTC's`.

## 6. How to run (after S0_DONE and landing)
```
copy the landed code dir to C:\Users\Admin\epr1\code
C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe -I C:\Users\Admin\epr1\code\epr1_gate.py --fraction <f> --commit <40-hex landed commit>
.\launch_epr1.ps1 -Fraction <f>          # then -Lc, then -C2, each with its own mint (--stage lc / --stage c2)
```
Paste the script body into a PowerShell prompt (script execution is disabled on this box; never `-ExecutionPolicy Bypass`). A fit is minutes on CPU; if it runs longer than 4 minutes it is detached by the launcher and polled by script.

## Deliverable manifest
All under `C:/Users/Admin/par18/epr1/out/` (dev box only); `paths.json` lists `[repo path, source, "raw"]` for every file. **Everything is in ONE place until the Master Mind lands it.**

| artifact | where |
|---|---|
| `PREREG_EPR1.md`, `PREREG_EPR1.sha256` | dev box only |
| `code/epr1.py`, `epr1_gate.py`, `launch_epr1.ps1`, `test_epr1.py`, `test_epr1_gate.py`, `template_pins.json`, `template_pins_make.py`, `zero_tail.py`, `reach_proof_synthetic.py`, `CODE_SHA256.json` | dev box only |
| `raw/reach_proof_synthetic.json`, `raw/test_epr1_46.log`, `raw/test_epr1_gate.log`, `raw/calibration_synthetic.log` | dev box only |
| `RESULT.md` (this file; also listed at the package path) | dev box only |
Scratch (not for landing): `C:/Users/Admin/par18/epr1/src/` (template copies read from the mirror tip) and `.../work/` (builders `mk_*.py`, which regenerate the code files from the template).
