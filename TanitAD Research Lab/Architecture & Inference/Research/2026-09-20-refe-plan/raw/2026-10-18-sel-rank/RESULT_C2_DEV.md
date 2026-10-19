# SRTTC-C2-STAGE and the SR-TTC dev read at f = 10 %: the C2 gate stage is built and was used; the C2 control read RED; the dev verdict is VOID and the one read is consumed

Date 2026-10-19 (run 05:50-06:14 Berlin). Agent: Arch+Inf. Brief: the Master Mind's task text (no brief file). CPU only. Landed tip used: `1afc54f6fa5ee865f2966e3c0a7b3141f24ef0c1` (fetched, blobs byte-exact).
Not touched: `C:/Users/Admin/epr1/`, `srttc_s0/` (read only via the gate), any lock, any navtest file. The JOINT was not computed. No git add/commit/push.

Evidence classes: **MEASURED** = our run, artifact path given. **INHERITED** = another document, not re-run. **ESTIMATED** = arithmetic. Dev numbers are **SELECTION tier on SEEN scenes (D-L1-G0-1)**: a gate, not a claim; PDMS by lookup on the frozen fan (T1-style NAVSIM self-action PDMS, not a capability claim).

## 0. Headline and escalations

| item | result |
|---|---|
| (1) C2 gate stage | **BUILT, token minted PASS 6/6, and used.** `srttc_c2.py` itself cannot check a token without being edited (it is landed and frozen), so a NEW wrapper `srttc_c2_run.py` does; the gate is `srttc_c2_gate.py`. **`--no-gate-stage-ack` is not accepted** by either. |
| (2) C2 fit at f = 10 % | done through the gated entry; `r1_c2_result.json` sha256 **`d23aa0cb7f4d0674ab5e8c4d258a1a3164a99a49c1be38a75a20aa1efde43424`** |
| (3) SR-TTC dev gate, read once | verdict **VOID** (control C2 RED). Gains x100: seed 0 **+0.100 [-0.166, +0.325]**, seed 1 **+0.078 [-0.199, +0.309]** vs the +0.30 bar. **Read consumed (R3).** Result sha256 `55fc372448ad725d9553848af16031d549582d5117c9060a7383582b2b37e66f`. |
| precondition not named in the task | the **A2.4 learning-curve call at f = 10 %** is required before the dev gate (`srttc_dev.py check_lc` refuses without `lc_result.json`). I ran it under its own gate token: **FLAT**, so the registration permits the read. |

**ESCALATE (integration, needs the Master Mind):**
1. **Land `srttc_c2_gate.py`, `srttc_c2_run.py`, `test_srttc_c2_gate.py`** (staged, LF). The C2 token marks the first two `landed: false` (disclosed, re-measured at every verify). After landing, `srttc_c2.py`'s own CLI path `--no-gate-stage-ack` should be removed in a later landing: run directly with that flag it still fits without a token (not edited here).
2. **Rule on the VOID.** R3 already says: the read is consumed, SR-TTC does not reach the look, the successor is a NEW registration. The facts you need for that registration are in section 5 (C2 as registered reads RED on a null at a high rate; the real gains are below the bar with the controls aside).
3. **R9 / the shared look:** SR-TTC is now `READ, VOID`, so by the code of `eligible_readings` (read, not run) the look cannot open through SR-TTC (MEASURED: the landed `refe_head_confirm.read_dev_result` accepts the file and returns verdict VOID with the two row shas and the r1 sha). The look can open only through EP-R1ub or a formal abandonment. Not acted on.

## 1. What was built (NEW files, LF; the landed `srttc_r1_gate.py`, `srttc_r1.py`, `srttc_c2.py`, `srttc_dev.py` are byte-identical to their blobs at 1afc54f)

| file (REFE = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`) | repo dir | sha256 | role |
|---|---|---|---|
| `srttc_c2_gate.py` | `REFE/raw/2026-10-18-sel-rank/code/` | `ec707825a9d71d823efb123154a91f1fc1fb518f3af301cc4f0aae26e81e12a6` | mint / verify / argv for stage `c2`; the enforcing `run_main`; attestation writer + verifier |
| `srttc_c2_run.py` | same | `98e6752da4e6dd836e53892d261f06539c059e3bde51fd8be6cf1eb719946986` | the only sanctioned entry; refuses (exit 3) with no matching token, exit 2 on `--no-gate-stage-ack` or `lc` |
| `test_srttc_c2_gate.py` | same | `ce5aee08b6874ab95608c5820a685ad79abf9fdbd647dcbfef86a28cffc6c3f6` | 39 tests |

**Why a wrapper.** `srttc_c2.py` main refuses without `--no-gate-stage-ack` and has no token check; changing it would edit a landed, hash-pinned file. The wrapper verifies the token, then execs the landed `srttc_r1.py` / `srttc_c2.py` from the exact bytes the token binds (`import srttc_r1` inside `srttc_c2.py` resolves to the verified module) and calls `run_c2`. It then writes the sidecar `r1_c2_attestation.json` (token sha256 + HMAC, argv sha, r1 sha, result sha, rows). `srttc_c2.py`'s result file cannot carry it.

**The token binds** {commit `1afc54f6...`, argv sha256 `2a99d565e5a8a49e214cdb90f11d58285fbb23c53465b9996f9507118a7f820c`, S0 bank manifest sha256 `6296dfc0b03905038fbf132f82009f6b6d7e8d61fa9abfb65c9164f487a5f2a1`, sha256 of `srttc_c2.py` / `srttc_r1.py` / `srttc_r1_gate.py` (each equal to its blob at the commit), sha256 of the two new files, `r1_result.json` sha256 `f447e715...` (so C2 runs at THE registered c*), shuffle seed 20261018}. Profile `srttc-c2`, stage `c2`: an R1 `run` or `lc` token can never open C2 (tested).

**Checks, mirroring `srttc_r1_gate.py` / EP-R1's `--stage c2`:**
* **G-COMMIT** blobs at the commit; **G-DVB** the frozen gate's declared-vs-built check applied to the same argv (levers, the BUILT grid of 9, Cfg sha, constants, paths, pins) **plus** the C2 object (`C2_SHUFFLE_SEED`, `RESULT_NAME`, `ROW_FMT` = what `srttc_dev.py` reads; `srttc_c2.R` is the verified module) **plus c***: `r1_result.json` equals the registered sha256, its params equal the argv, c* is in the registered 9 and its config sha256 rebuilds, it was written by this `srttc_r1.py`, no C2 output exists yet. **The G-DVB entry for the new stage is this block; `--no-gate-stage-ack` and the `lc` sub-command are FAIL reasons.**
* **G-PATH** / **G-BANK** the frozen, tested checks, plus "r1_result.json was fitted on THIS bank manifest".
* **G-LIVE** (the reachability class, "built, tested, unreachable from its caller"): R1's one-step probe on the first real unit (TTC-row gradient > 0; EXACTLY 0.0 on the other five rows and the features; zero control 0.0; a leaky loss reads non-zero) **and** C2's own: the shuffle moves labels, keeps each set's multiset, is deterministic and seed-dependent, does not modify the real labels; and **through `fit_c2_rows` -> `fit_row` a recording loss receives the SHUFFLED labels** (loss called exactly once; shuffled != real gradient, L1 = 34.79 on the real unit; same labels twice differ by exactly 0.0).
* **G-SUITE** on a clean `git archive` export at the commit: `test_srttc_r1.py` 35 tests OK + `test_srttc_dev.py C2Fit` 8 tests OK.

**Tests, MEASURED, `raw/c2_gate/test_srttc_c2_gate.log` (final bytes): 39 OK in 111.6 s.** Deliberate-regression arms that go RED (each with a same-breath control that passes): G-LIVE: C2 on the REAL labels, identity shuffle, shuffle across sets, shuffle in place, seed ignored, a fit that bypasses R1's loss, a detached R1 row; G-DVB: ack flag, lc, wrong entry/dir, epochs 11, wrong/absent registered sha, 8 edits of r1_result.json (fraction, stage, grid, config sha, code sha, pins [strict], replacements, params), 3 module constants, `R` not the verified module, outputs already present; G-BANK: a corrupted unit byte, a changed manifest; G-PATH: a navtest path; G-SUITE: mutated / emptied / skipped / second spec alone / no file at the commit; G-COMMIT: edited file, unlanded, new file that differs from its landed copy, frozen gate edited, unknown commit; token: argv X used with Y, edited token, other key, synthetic profile on a real launch, code/manifest/marker/r1_result changed after the mint, an R1-stage token, a FAIL report, a stale PASS demoted. **Enforcement mutations** (one marked line mutated, a scenario GREEN on the original must go RED): verify skipped, argv comparison, HMAC, stage, the four code-sha layers together, the inputs re-measure, the registered-sha comparison, the ack message, the shuffled-label decision. Disclosure: `--no-gate-stage-ack` is refused by three independent layers, so removing the explicit message alone still refuses (tested as defence in depth).

**Real-machine refusals before the fit** (`raw/c2_gate/refusals_real.log`): `--no-gate-stage-ack` -> REFUSED (rc 2 text); `--epochs 11` -> REFUSED by the gate (argv sha differs); the landed `srttc_c2.py` without its ack flag -> its own refusal. Nothing was written to the run dir by any of them.

## 2. LC at f = 10 % (a registered precondition, run with its own token)

Token `f10_lc` minted by the landed gate (`--stage lc`, commit 1afc54f, PASS 6/6), launched by the landed `launch_r1.ps1 -Lc` (dry run first). MEASURED, `lc_result.json` sha256 `25138a9ff1ca8f1ef7c82f527b2854c9aa483be23d509927093c4637ceaf8b06`, p100 reproduces R1 on both seeds:

| seed | pooled OOF concordance at 25 / 50 / 100 % | Delta = c(100) - c(50), paired log-cluster bootstrap B 10,000 seed 20260927 |
|---|---|---|
| 0 | 0.68857 / 0.69821 / 0.70069 | +0.00248 [-0.00359, +0.00843] |
| 1 | 0.68668 / 0.69900 / 0.69945 | +0.00046 [-0.00548, +0.00576] |

Call (A2.4 item 3, both-rule `flat`): upper bounds 0.0084 and 0.0058 are both below delta = 0.010 -> **FLAT** -> action: freeze c*, read the dev gate once. (Seed 0's upper bound is 0.0016 below the threshold.) This interval answers "another draw of logs" only; fit seeds answer head-fit noise.

## 3. C2 at f = 10 % (MEASURED)

`r1_c2_result.json` sha256 `d23aa0cb...3424`; rows `ttc_row_c2shuffle_seed0.npy` `df2830c459f047c95e7be506c1edb44931f59c21067c70eaaa6029d7d78818ed`, `..._seed1.npy` `7a19d5a1ceb7184d747c95fb4a7955a580686fcc0dda93c6ee3e163ee0dcf3e8`; 16.0 % of labels moved, 15,808 sets / 117 logs, both rows differ from the real-label rows; attestation `r1_c2_attestation.json` sha256 `15ad41ec8666ed83f80ddc713920fd2957adacf561287b2cc04afa0b16ca4572`, verified (HMAC, token sha, result sha, rows). Shuffle seed 20261018 (UNREGISTERED for SR-TTC, GAP G1, recorded). Token sha256 `6d2ae1a6c2df1b47ac0d59426b966a4dca5da395da2fa4cafdc3f3556d30d72e`.

## 4. The dev read (MEASURED, `srttc_dev_result.json`; ONCE; SELECTION tier)

Command: `srttc_dev.py run --run-dir C:/Users/Admin/srttc_r1/run/f10 --registered-result-sha256 f447e715... --p2ho-dir <the tip's P2HO attestations>`. The D: worktree does not hold `raw/2026-10-18-p2-heldout-table/`, so its three attestation files were materialised byte-exact from the tip (`git cat-file blob`; sha list in `raw/srttc_dev/f10/p2ho_attestations.sha256`). A2.10: all four conditions hold (1,705 tokens / 3,137 sets / 24 logs; every package gate PASS; 333 tokens bit-identical; table files = the landed pins). A no-outcome inputs check ran first and read nothing (`outcomes_opened: false`).

Marker `srttc_dev_READ_STARTED.json` written 06:14:06 before the outcomes opened. Estimator: paired log-cluster bootstrap, B 10,000, seed 20260927, **24 logs / 1,705 tokens**; it answers "another draw of 24 logs" only. Intervals are reported, never a bar (24 logs support no lower-bound rule).

| | seed 0 | seed 1 | bar |
|---|---|---|---|
| paired total gain, PDMS x100 (P2 base 90.637) | **+0.100 [-0.166, +0.325]** (new 90.737) | **+0.078 [-0.199, +0.309]** (new 90.715) | >= +0.30 on BOTH |
| picks changed (of 1,705) | 548 | 541 | |
| bar met | no (0.200 short) | no (0.222 short) | |
| EP component | +0.464 [+0.050, +0.801] | +0.528 [+0.146, +0.836] | |
| TTC component | -0.059 [-0.625, +0.483] | -0.176 [-0.753, +0.394] | |
| DAC / DDC | -0.117 / +0.117 | -0.117 / +0.147 | |
| unsafe-pick rate | 0.0328 -> 0.0340 | same | |
| **C2 (shuffled-label row) gain** | **+0.078 [-0.342, +0.452]**, 766 picks changed | **+0.137 [-0.277, +0.508]**, 795 picks changed | **<= 0.0, strict, no tolerance (R2)** |

Controls: **C0 GREEN** (unchanged TTC row -> P2's pick on 1,705/1,705, gain exactly 0.0); **C1 GREEN** (recorded gradients, R1 `run` token HMAC-verified and bound to this c*/bank, applier reach: only column 3 moves, other five bit-identical, negated-row positive control changes 1,319 picks); **C4 GREEN** (max abs pose diff 0.0 m on 1,705); integrity GREEN (max 1.45e-3 vs 2e-3, next-token control 15.2); **C2 RED** (both seeds > 0). C3: two seeds, rows distinct, |gain_0 - gain_1| = 0.022. c* = lr 1e-3, anchor 0.3 (lr and anchor at the grid maximum; SR-TTC registered no edge rule, R8: disclosed only).

**Verdict (registered rule, `verdict_of`): VOID** (a control is RED: "each is a gate; a miss voids the arm", PREREG:67). Rows fired: `VOID` only. Per **R3 the read is consumed**; the arm does not reach the navtest look; its successor is a NEW registration.

## 5. Reading (what the numbers say; nothing here changes the verdict)

* **The VOID is not hiding a PASS.** Independent of C2, both real-label gains are below the +0.30 bar by 0.20 and 0.22 points. Had the controls been green, the registered row would be DEV-FAIL, and with the LC call FLAT the A2.6 row is "HEAD/FEATURE-LIMITED: more fit data of this kind is NOT the lever" (not applied: the verdict is VOID, the MM rules).
* **The real head does not separate from a shuffled-label head on this set.** Point gains 0.100 / 0.078 (real) vs 0.078 / 0.137 (C2) have overlapping intervals that all contain 0. I did not compute a paired real-vs-C2 difference: that would be a new look at the spent outcomes. Also, the gain that exists is carried by the EP component (+0.46 / +0.53, intervals above 0), while the TTC component moved the wrong way (point -0.06 / -0.18, intervals spanning 0): a TTC-row retrain whose measurable effect is on EP is a mechanism question for the successor, not a result.
* **C2 as registered rejects a null most of the time (ESTIMATED, HYPOTHESIS-grade).** The shuffled head changed MORE picks (766 / 795) than the real head (548 / 541), so its gain is a draw of random sign. If that draw is symmetric about 0 and independent across the two seeds, P(both <= 0) = 0.25, i.e. a **false-RED rate of about 75 %**. The symmetry is unverified. R2 kept the rule strict and disclosed "a chance RED burns the one dev read"; this is that event, and it also happens to coincide with a failed bar. Any successor registration should fix its C2 (replicate shuffles and a one-sided rule against the bar, registered before the read), not reuse this one.
* **LC FLAT with a barely-flat seed 0** (0.0084 vs 0.010): "FLAT" licensed the read; it does not show the head is saturated, only that doubling 50 -> 100 % of the folds' training logs moved the pooled OOF concordance by less than 0.01 (c* pooled OOF concordance 0.7007; there is no like-for-like reference, because P2's own row on these fit pairs is in-sample for P2 and 0.722 is a navtest number).

## 6. Rule Zero: the next levers (executed where unblocked; named where not)

| lever | state |
|---|---|
| (a) the registered DEV-FAIL probe: a 2-layer TTC probe on the cached S_10 pre-head features, 5-fold-by-log CV, contender concordance vs 0.72 (PREREG section 7). CPU, minutes, reads no dev outcome. | **UNBLOCKED, NOT RUN.** Not run because (i) the verdict is VOID, not DEV-FAIL, so the MM must say the row applies, and (ii) the probe's width / epochs / optimiser are not registered, so running it now would invent them. It needs a 5-line pre-registration, then it is minutes. |
| (b) successor registration with a powered C2 (see section 5) | needs the MM / PI decision on the VOID (R3). |
| (c) f = 25 % increment (GPU S0, ~6 GPU-h central) | **not triggered**: LC FLAT; blocked on GPU/PI anyway. |
| (d) EP-R1ub (R7) | another stream; the look needs it or an abandonment (R9). |

## 7. Proposed register rows (ids to be assigned on the tip; not edited)

* `D-SRTTC-C2GATE`: SR-TTC C2 now has a launch-gate stage (`srttc_c2_gate.py` + enforcing entry `srttc_c2_run.py`; 39 tests, enforcement mutations; token `6d2ae1a6...` PASS 6/6; `--no-gate-stage-ack` not accepted). Closes MM R5. Open: land the files; remove the ack CLI path of the landed `srttc_c2.py`.
* `D-SRTTC-LC-F10`: LC at f = 10 %: FLAT (seed 0 +0.00248 [-0.00359, +0.00843], seed 1 +0.00046 [-0.00548, +0.00576]); `lc_result.json` `25138a9f...`.
* `D-SRTTC-DEV-F10`: SR-TTC dev gate at f = 10 %: **VOID (C2 RED), read consumed (R3)**. Gains x100 +0.100 / +0.078 (bar 0.30), C2 +0.078 / +0.137; SELECTION tier, 24 logs. `H-SELRANK-TTC`: not confirmed, not refuted; navtest unspent.
* `R-G2` (realised): strict no-tolerance dev C2 read RED on a null-like head; the successor must register C2 power.
* BACKLOG: (1) pre-register + run the feature-vs-capacity probe (CPU, minutes); (2) draft the SR-TTC successor registration (C2 with replicates and a one-sided rule); (3) land the three new files and re-mint on the landing commit; (4) R9: look gating with SR-TTC VOID.

## 8. Disclosures

* The task named the C2 stage and the dev read; the LC stage was added because the registration and `srttc_dev.py` require it (`check_lc`). It used the landed gate and launcher unchanged.
* New code was deployed and run from `C:/Users/Admin/par18/src2/code/` (a new top-level directory under `C:/Users/Admin/` was refused); the minted argv names that directory. The landed files there equal their blobs at 1afc54f (`G-COMMIT`). `DEFAULT_CODE_DIR` in the staged gate names `C:\Users\Admin\srttc_r1\code`, where the MM would deploy; pass `--code-dir` otherwise.
* The commit binding covers only landed files; the two new files are bound by sha256 and flagged `landed: false`.
* The venv `python.exe` is a shim: each logical Python process shows two OS pids. At most one logical Python process of mine ran at a time (LC mint, LC fit, tests, C2 mint, C2 fit and the dev read were all sequential).
* A2.10's attestation directory is a byte-exact copy from the tip, not the D: worktree.
* Not done: JOINT; any navtest file; the probe in section 6(a).

## 9. Deliverable manifest

All files are staged under `C:/Users/Admin/par18/src2/out/` mirroring repo paths; `out/paths.json` lists `[repo path, absolute source, "raw"]`. **Each exists on ONE disk (this dev box, C:) until the Master Mind lands it**; run-dir originals are also in `C:/Users/Admin/srttc_r1/run/f10/` and `.../gate/f10_{lc,c2}/` (same disk).

| artifact | where |
|---|---|
| `srttc_c2_gate.py`, `srttc_c2_run.py`, `test_srttc_c2_gate.py`, `CODE_SHA256_C2.json` | repo path `REFE/raw/2026-10-18-sel-rank/code/` (staged in out/); working copy `C:/Users/Admin/par18/src2/code/` |
| `lc_result.json`, `lc.log`, `lc_gate_token.json`, `lc_gate_mint.log` | `REFE/raw/2026-10-18-sel-rank/raw/r1/f10/` |
| `c2_gate_token.json`, `c2_gate_mint.log`, `c2_run.log`, `r1_c2_result.json` (+ `.sha256`), `r1_c2_attestation.json`, `ttc_row_c2shuffle_seed{0,1}.npy` | same |
| `test_srttc_c2_gate.log`, `refusals_real.log` | `REFE/raw/2026-10-18-sel-rank/raw/c2_gate/` |
| `srttc_dev_result.json` (+ `.sha256`), `srttc_dev_READ_STARTED.json`, `srttc_dev.log`, `srttc_dev_inputs_check.json`, `p2ho_attestations.sha256` | `REFE/raw/2026-10-18-sel-rank/raw/srttc_dev/f10/` |
| this file | `REFE/raw/2026-10-18-sel-rank/RESULT_C2_DEV.md` (copy `out/RESULT.md`) |
| not staged (tooling / working state) | `C:/Users/Admin/par18/src2/STATE.md`, `stage_out.py`, `p2ho/`, `epr1code_ro/` (read-only extraction used to run the landed `read_dev_result`), `src/` |
