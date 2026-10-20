# AMENDMENT C2 (DRAFT), SEARCH-P2-CONFIRM READ C2: the navtest look (T1, ONE look) for the CEM search on P2's own scorer

Status: **DRAFT** (written 2026-10-20 ~03:50 by the C2-amendment agent for the Master Mind, who seals it. NOT sealed. Nothing here is an outcome. No navtest token has been bundled, searched or scored by this work, and no navtest label, score or search output was read to write it.)

**What this is.** `H-SEARCH-P2-1`'s second read. C1 (`PREREG_SEARCH_P2_CONFIRM.md`, sha256 `658ba1576af175a9a345692e284b1515604a6065279b921b2a4ca6dead6529d1`) asks whether the search gains again on 905 token-fresh tokens at the selection tier. C2 asks the T1 question on navtest: do the same search outputs raise NAVSIM v1.1 PDMS over P2's own pick on 12,146 tokens / 136 logs that are scene-disjoint from navtrain?
**What licenses it.** Only `MM_RULING_C2_LICENCE.md` (sha256 `fa7a26046a2fb5f8d8e69b2644c97a6cfce8907201d40d5d3c5be5358560ba6a`, sealed 2026-10-20 02:56): C1 reads O1_CONFIRMED or O2_DIRECTIONAL, AND each safety point >= -0.25, AND moved share >= 5 %, checked by code (section 6). EP-ADMISSION no longer licenses this arm and is not consulted.
**What sets the bar.** `MM_RULING_SPC.md` (sha256 `e2b8b8e1437534bc615410063ba313fff3f30f1882c415982d590ca8807e76d8`) DQ3. The numbers below are copied from it and may not be loosened here.
**Evidence classes.** MEASURED (ours, artifact named) / PUBLISHED / INHERITED (another package, not re-verified) / ESTIMATED / UNMEASURED / HYPOTHESIS. STRATEGIC is PAUSED (PI ruling 2026-09-19, ITEM 25): printed as PAUSED, never as a pass, never extended to the other three families.

## 1. The object: the sha-pinned search, identical to C1's arm, on navtest's tokens
| lever | value | pin |
|---|---|---|
| scorer | P2-s0 `model_final.pt`, **no head substitution** (argv carries no head flag; the merged search meta must read `arm P2HEAD`, `head_sha256` null) | `ckpt_sha256 ea31db22...`, `head_s0_sha256 bded124e...` (the gate's probe head only; the P2HEAD stamp ignores it) |
| search | `tts_cem.cem_search`: K = 5 iterations, M = 64 per iteration scored as ONE complete set, 8 elites, warm start at P2's pick, final pick re-scored inside the original fan | L1 `tts_cem.py` `9dea9603...` |
| config | C8 `stepB_config.json` (sigma0 0.08126 / 0.15; clips u in [ln 0.85, ln 1.15], d in +-0.3 m) | `config_sha256 299c6e63...` |
| search seeds | 20260927 / 20260928 / 20260929 | `tts_cem.SEARCH_SEEDS` |
| code | Step B's `stepB.py`, `l2_scorer.py`, `g_reach_stepb.py`, `label_stepB.py` (fixture only), `analyse_stepB.py`, L1's four files: pinned in C1's sealed text (section 12) and unchanged here | C1 PINS |
| pipeline | `navpipe_core.py`, `navpipe_chain.py`, `navpipe_launch.py`, `navpipe_analyse.py`, `make_navtest_bundle.py`, `navpipe_bridge.py`, `navpipe_gpu_stage.py` | `code:` pins below, filled at the seal |
Per token the search costs 5 population set-scorings + 1 base-fan scoring + 8 elite re-scorings = **14 set-scorings** (`l1run/code/tts_cem.py` lines 52-69, the sha-pinned L1 file, read here: MEASURED by source reading), on the three scorer modules only (`score_q_mlp`, `score_dec`, `score_head`, 78 tensors; `real_scorer.py` docstring).

**Tokens.** ALL of navtest, none dropped, none re-selected: **12,146 tokens / 136 logs**, `D:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json` (`{rule, tokens, token_log}`).
- file sha256 `927bbe3838dd70af5fa27eadceb09f55acc9c1cfd04cf0a6b520f1519175a955` (md5 `3a8b3653fd70891e50a230a37eb76ae8`, 996,128 bytes), MEASURED here by hashing the file (it holds no label).
- **token-list sha256** (the 12,146 tokens sorted and joined by newlines = `all_tokens_sha256` of the plan): `8d42fef68542f095d0e03894fdaf38babfdc89f3fd7cdc26080c3a6009cb4108`.
- **chunk plan:** 14 chunks of whole logs, cap 1,000, sizes `[990, 903, 868, 591, 986, 946, 916, 963, 502, 918, 983, 993, 958, 629]`, canonical plan sha256 `b3b609192fcd97971bce6771b5322caef853ddc70257ae017c55b5da5368df36` (recomputed here from the token file with `NC.plan_doc` and equal to `navpipe/raw/navtest_chunk_plan_cap1000.json`; the chain recomputes it again and refuses on a difference). Each chunk carries its own `tokens_sha256`, the per-chunk data clause.
- **Parity:** this is the exact token set P2's 88.82 was measured on (INHERITED from `common.md` and the register); the pipeline refuses any other count.
**What C2 does NOT share with C1:** tokens (navtest, not B-REST), the label route (T1 harness, not `label_stepB`), the scene footing (navtest is scene-disjoint from navtrain by construction, so C2 is the scene-unseen read; C1 was SEEN-SCENE), the bar (DQ3, not O1).

## 2. The endpoint and tier: the T1 route (DESIGN section 3, `file:line` there)
- **Endpoint:** per token n and search seed s, d(n, s) = 100 x [T1 PDMS(search output) - T1 PDMS(P2's pick)]; the token's value is the mean over the 3 seeds; unit **PDMS points**.
- **Route:** the search `exec` (20 poses) -> the seam's own `to_navsim` (`eval/refe_navtest_seam.py:58`, vendored and sha-pinned `ceb2c0f8...`) -> a W3 seam npz -> `score_navtest_refe.py` (NT-FIX1 version, `94f7b024...`) -> W3 `run_v1.py` `cmd_score` on the prebuilt navtest metric cache -> per-token CSV -> `NC.read_harness_csv`, which refuses a missing token, an invalid row, a non-finite value or PDMS != NC x DAC x (5 EP + 5 TTC + 2 C) / 12 (<= 1e-12).
- **P2's pick baseline** is its BANKED CSV row (`refe_stage2_p2_full.csv`, sha-pinned below), not a re-score. Guard: the **T1 planted identity**, per chunk on 24 tokens, sends the pick's own plan through the SAME route and must reproduce the banked row (six sub-scores and PDMS) to <= 1e-9, and a plan shifted 0.5 m ahead must change at least one token (control). A failure STOPS the chain (exit 7) with the ctx kept. (DQ6 asks whether to re-score all picks instead.)
- **Harness caveat (INHERITED, stated):** P2's 88.82 ran the pre-NT-FIX1 script (`91ec3e65...`); the pipeline runs the tip's NT-FIX1 script (cross-drive relpath, stale-CSV quarantine, count-mismatch summary: non-scoring fixes). That the fixes are score-neutral rests on `2026-10-19-navtest-harness-fix/RESULT.md` (not re-verified here); the planted identity is the in-run check.
- **Tier stamp on every C2 line:** `T1 = NAVSIM v1.1 PDMS pseudo-simulation of an open-loop plan against logged agents (NOT closed loop); navtest, 12,146 tokens / 136 logs; ONE look; ONE training seed (H-ESTIM-SEED-1 open); 3 search seeds; the interval answers "another draw of LOGS" only; no head substitution; STRATEGIC PAUSED`. A C2 number never appears without it.

## 3. The estimator
**Paired log-cluster bootstrap**: `stepA.ratio_ci` over `stepA.boot_indices`, ratio of sums, **cluster unit = the log (136 logs, whole logs resampled)**, **B = 10,000, seed 20260927**. Paired: one set of draws is reused for the endpoint, the four safety deltas, dEP, dDDC and each seed's own point, on the same tokens. It answers *"would another draw of LOGS say this?"* and nothing else: not another training run (H-ESTIM-SEED-1), not another inference run (the 3 per-seed points and their range are the only inference variance sampled, and they gate the decision through the seed-range term). `overlapping_holdout_se` and any quadrature of unpaired intervals are not used.

## 4. The decision rule, verbatim from `MM_RULING_SPC.md` DQ3
> **Decision rule:** lower CI > 0 AND point >= **+0.10 PDMS points** (the minimum worthwhile effect) AND point > the 3-seed range AND each of dNC, dDAC, dTTC, dC >= -0.25. The four families are reported beside it, per family.
- **Numbers, sealed here and not loosenable:** lower CI > 0; point >= +0.10 (pin `bar_dpdms_points 0.10`); point > the range of the three per-seed points; dNC, dDAC, dTTC, dC POINT estimates each >= -0.25 (pin `bar_safety_points -0.25`). Implemented by `navpipe_analyse.c2_terms`; the bars come from the sealed pins, no default exists in code. A term met exactly at its bar (point = +0.10, a safety point = -0.25) is met.
- **Why +0.10 (the ruling's):** one third of L1's +0.30 materiality line; navtest can resolve it (mean CI half-width 0.065 under R to 0.178 under E, ESTIMATED, `PREREG_SEARCH_P2_CONFIRM.md` section 8). For scale: P2 88.82 -> target 94.55 is a 5.73-point gap, so +0.10 is 1.7 % of it.
- **Power (ESTIMATED, inherited from that table, not re-run):** P(lower CI > 0) at a true +0.16 is 100 % (R) / 47 % (E) and at +0.327 is 100 % / 99.7 %. The full rule is a subset of that event, so these are UPPER bounds on its power; the seed-range term's power was not simulated (UNVERIFIED).
- **A pass is a T1 result on one training seed.** It does not say the search generalises across training seeds, and it is not a closed-loop result.

## 5. The four families, reported beside the rule, per family (never pooled; ADE is not a result)
Each from `analyse_stepB.analyse` UNCHANGED on the same tokens, with the paired interval above:
- **LONGITUDINAL:** dEP and dTTC (points, CI); share of outputs that are speed-ups / slow-downs and dPDMS given each (descriptive). *Distance-keeping / headway: n = 0, the T1 harness carries no lead-agent headway row beyond TTC.*
- **LATERAL:** dDAC and dDDC (points, CI); mean |offset| of moved outputs; heading deviation at the last non-repaired pose; largest yaw-rate deviation against the pick (descriptive geometry). *Curvature error and cross-track: n = 0, a pseudo-sim open-loop tier has no closed-loop rollout.*
- **TACTICAL:** share of outputs whose command class (`stepA.command_class`, a bearing rule on the end point; HYPOTHESIS: a proxy) differs from the pick's. *Manoeuvre-decision quality and tactical goal selection: n = 0, no manoeuvre decision is scored at this tier.*
- **STRATEGIC: PAUSED** (PI ruling 2026-09-19, ITEM 25).
- *The in-fan decomposition (arm argmax vs pick) is NOT COMPUTABLE at T1: only the executed plan has a label.* A3 / A4 / A5 are not computed at T1 for the same reason.
- Inference stays vision-only: the search reads the token and P2's own scorer; no privileged label is an input.

## 6. The licence check: read by code, before any navtest token
`navpipe_chain.py --mode navtest` calls `NC.check_c2_seal(...)` FIRST (guard `C-SEAL-FIRST`), before the token file is opened. After the amendment's own checks (three-way sha, SEALED status, pins, code pins, cost status stated, licence-ruling pin) it calls `NC.check_c1_licence`, which refuses (class `seal`, exit 5) unless ALL hold:
| term | what is read | refusal reason |
|---|---|---|
| the file is the pinned one | `--c1-result` exists; its sha256 equals the typed `--c1-result-sha256` (and the amendment's `c1_result_sha256` pin if one exists) | `LICENCE_MISSING_ARGS`, `LICENCE_RESULT_MISSING`, `LICENCE_RESULT_SHA_MISMATCH`, `LICENCE_RESULT_PIN_MISMATCH`, `LICENCE_RESULT_UNREADABLE` |
| it IS the C1 read | `read == "C1"`, 905 tokens, 24 logs, arm P2HEAD; the `chain_state.json` beside it carries `prereg_sha256` = the pinned `c1_prereg_sha256` (`658ba157...`) and all six analysis stages done | `LICENCE_NOT_C1`, `LICENCE_CHAIN_STATE_MISSING`, `LICENCE_PREREG_MISMATCH`, `LICENCE_C1_INCOMPLETE` |
| outcome | `outcome` is `O1_CONFIRMED` or `O2_DIRECTIONAL` | `LICENCE_OUTCOME` (TRADE, O3_NULL, NEGATIVE, O4_INERT do not license) |
| safety | `safety_points` dNC, dDAC, dTTC, dC each a finite number >= -0.25 (O2 too) | `LICENCE_SAFETY_UNREADABLE`, `LICENCE_SAFETY_MISS` |
| moved share | `inert.frac_moved` finite and >= 0.05, and `inert.inert` is false | `LICENCE_MOVED_UNREADABLE`, `LICENCE_INERT` |
- The identity term exists because the carve-B read also reads O1_CONFIRMED (+0.3273, 400 tokens): a valid-looking result from the wrong read must not license. The `chain_state.json` binding exists because the SEARCH-DOSE arm runs on the SAME 905 tokens through the same writer (its result says `read C1` too); only the prereg sha tells them apart.
- **Recorded:** the chain prints `ZZNAVPIPE_LICENCE_OK <outcome> safety <...> moved <...> c1_result <sha12>` and banks `licence_c1.json` (the numbers as read, with the amendment sha) beside the plan.
- **Command line (all values from the sealed record, nothing defaulted):** `--mode navtest --amendment <sealed.md> --amendment-sha256 <sha> --c1-result D:/Projects/TanitAD-artifacts/spc/RESULT_SPC.json --c1-result-sha256 <sha, typed from the register> --table-ref D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz` (DQ11), first with `--dry-run` (seal + licence + data + disk, no child), whose `ZZNAVPIPE_DRYRUN_OK` log is banked.
- Reported beside C1 with no authority (the ruling): A2 and A1-r1 through the adapter, A3 / A4 / A5 of the search outputs. They are not read here.

## 7. Inference cost per token on the deployment target (Jetson Thor)
**Status: UNMEASURED** (pin `inference_cost_status UNMEASURED`; DQ3 asks whether to keep it so).
- **What exists:** no Thor latency of P2 or of the search was found at three locations (a keyword search of `GOALS_AND_CLAIMS.md`, `MODEL_REGISTRY.md` and the REFE plan documents on the mirror at FETCH_HEAD `c3a5c329`, not re-fetched; `REFE_PLAN.md` lists "REFe's measured latency on Thor" as a to-do; UNVERIFIED that none exists elsewhere). The only cost numbers are dev-box CPU: the search is 3.95 s/token for 3 seeds at 4 threads (MEASURED, `spc/raw/cost.json`), i.e. about 1.32 s per token-seed and about 94 ms per set-scoring over the 14 (ESTIMATED arithmetic). **That is not a Thor number and nothing in this draft extrapolates it to Thor** (the only inherited Thor fact, batch-1 launch-bound ticks, H-DEPLOY-3, is for a different model).
- **Why UNMEASURED now:** measuring needs a GPU session on Thor, which this work may not touch, and a script that has never run on Thor would be untested code in a pin.
- **How it would be MEASURED (the committed protocol, a separate sealed measurement, ~1 Thor-hour plus staging):**
  - *Script* `thor_search_latency.py` (to be written and CPU-tested first; it reuses `real_scorer.py` and `tts_cem.cem_search` UNCHANGED, both sha-pinned).
  - *Inputs:* P2-s0 scorer modules from `model_final.pt` (`ea31db22...`); the first 200 tokens (sorted) of **carve B** (token list `0629811f...`; already read, not navtest, not B-REST) with their banked `ctx.npy` rows (200 x 3.75 MiB = 750 MiB) and raw fans from the carve-B bundle; C8 config; ONE search seed (the deployed configuration).
  - *What is timed:* wall clock per token, `torch.cuda.synchronize()` bracketed, from "fan and ctx on device" to "final pick returned", i.e. the **added** latency of the search: 14 set-scorings on the GPU plus the ARM-CPU numpy parts (`composite`, `aggregate`) and the small H2D / D2H of the [64,20,3] population and [64,6] logits. 30 warm-up tokens discarded, 200 timed, batch 1, repeated 3 times. Reported: mean, p50, p95, p99, max in ms, a per-stage breakdown (CUDA events for the scorer, `perf_counter_ns` for CPU parts), and the ratio to P2's own pick-only forward (separate measurement, 50 tokens with frames).
  - *Recorded state:* `nvpmodel` mode, `jetson_clocks` state, GPU clock, thermal at start and end, torch build and sm_110 driver, no other GPU job (utilisation < 5 % before start).
  - *Controls that must read known values:* a no-op `score_fn` (isolates the ARM numpy cost); `n_iters = 0` returns the pick bit for bit; the set-scoring counter reads exactly 14 per token; doubling M must raise the per-scoring time (the timer sees the work); Thor logits on the 3 bridge tokens within a stated tolerance of the banked dev-box logits (cost-only: Thor results are NOT used for the endpoint).
- **What it would take:** (1) a Thor slot that does not collide with a training or eval run (current Thor load UNVERIFIED); (2) a brief and an agent to write the script; (3) an aarch64 torch build that imports `refe/model.py` (UNVERIFIED); (4) a latency budget to read the number against (UNSTATED anywhere found; DQ4); (5) about 1 GB of staging.
- **What a C2 PASS may claim while UNMEASURED:** "T1 improvement on navtest at one training seed". It may NOT claim "deployable", "drives with the search", or any latency statement (the ruling: *a gain whose latency the vehicle cannot carry is a selection-tier fact, not a deployable one*).

## 8. The fixture precondition: carve B reads IDENTICAL
No real navtest run starts until `navpipe_chain.py --mode fixture` has ended `ZZNAVPIPE_FIXTURE_IDENTICAL` **on the code whose shas are pinned below**: the chained pipeline reproduces Step B's banked P2HEAD read on carve B exactly (same floats for `endpoint_dPDMS_points`, `per_seed_points`, `sub_score_deltas_points`, `families`, `inert`, `moved_share`) and every chunk's search and label rows bit for bit.
- **State at this writing (MEASURED, `navpipe/raw/fixture_result.json`, `navpipe/RESULT.md`): NOT SATISFIED.** The fixture STOPPED at `RAM_LOW 1.6 GiB < 3.0` after chunks 0 and 1 reached DELETED and chunk 2 reached SEARCHED (3 of 5), with C: and D: nearly full; no IDENTICAL verdict exists. The 119-test suite and the unchunked-equals-chunked test on the real carve-B arrays are green, but they are not the fixture.
- The licence change edits `navpipe_core.py` and `navpipe_chain.py`, so the fixture must be (re)run on the landed code, with free RAM >= 3 GB and enough disk. Its `ZZNAVPIPE_FIXTURE_IDENTICAL` line and `RESULT_NAVPIPE.json` sha are banked with the code shas. As built, this precondition is checked by the operator, not by code (DQ7).
- The GPU stage (`make_navtest_bundle.py`, `navpipe_bridge.py`, `navpipe_gpu_stage.py`) has NEVER run on a GPU. Its first real run is chunk 0's bridge on 3 navtest tokens (identity AND a one-pixel RED arm that must fire): the first navtest-token touch, after the seal, computing no label. If the fresh forward is not bit-reproducible against P2's banked seam (thread count, cudnn, driver, device), the bridge fails cleanly before any chunk is spent: **an infrastructure stop, not an outcome**, whose committed next arm is the diagnosis of the thread count first (`--threads 4` is Step B's setting, `MM_RULING_SPC.md` DQ8: bit for bit, a loosened gate needs its own amendment).

## 9. Cost (from `navpipe/COST.md`, ESTIMATED from MEASURED rates; the GPU-lock wait is not counted, C1's chain waited 2 h 20 min on 2026-10-19)
| item | value |
|---|---|
| chunks / tokens | 14 / 12,146 |
| GPU (bundle forward 1.4736 s/token + 46.6 s per lock hold, plus a 180 s bridge once) | **5.2 GPU-h**, in 14 locked holds on the dev-box GPU |
| search, 3 seeds, `--threads 4` | 15.28 CPU-h on 1 process (3.95 s/token, MEASURED) |
| T1 scoring (3 seed calls + 2 identity calls per chunk; 15.9 s + 0.0898 s/token) | 1.24 h |
| wall, 1 search process, sequential | 22.35 h |
| wall, 2 search processes (never measured; torch thread contention is a known trap) | 14.71 h |
| wall, prefetch 1 + 2 processes | 9.97 h |
| peak disk on the work volume, sequential / prefetch | 9.64 GiB / 13.27 GiB (the guard needs free minus the chunk's ctx >= 6 GiB) |
| banked outputs kept | 0.23 GiB |
**Resource preconditions (all UNMET at this writing):** the work volume (default `D:/Projects/TanitAD-artifacts/navpipe_c2`) has >= 9.64 GiB free (D: read about 0 and C: about 0.1 GB when this was written); free RAM >= 3 GB at every stage; the dev-box GPU lock available per chunk (held by another session's battery); at most 2 concurrent Python processes. A PI/MM decision on disk and GPU time is a named blocker, not a thing the pipeline works around. (RAM: the banked-seam loader held about 14 GB for the whole run until it was fixed alongside the licence guard; see `LICENCE_GUARD.md`.)

## 10. Every gate, with its RED arm (test names in `test_navpipe*.py`; "mutation" = the `@GUARD` line deleted and the suite must go RED)
| gate | refuses / stops when | RED arm | mutation |
|---|---|---|---|
| G-SEAL | no / edited amendment or sidecar; typed sha != file != sidecar; first `Status:` not `**SEALED`; pin missing; operative file unpinned or code pin differing; cost status not MEASURED/UNMEASURED; licence-ruling pin != `fa7a2604...`; S0 amendment offered for navtest and the reverse | `test_seal_refuses` (14 cases), `test_a_c2_amendment_cannot_unlock_the_s0_bank_nor_the_reverse`, `test_the_amendment_must_pin_the_licence_ruling_this_code_enforces` | `G-SEAL-*` (10 lines), `C-SEAL-FIRST`, `C-S0-SEAL-FIRST` |
| G-LICENCE | any term of section 6 | `test_navpipe_licence.py` (80 tests): every non-licensing outcome, each safety family for O1 and O2, NaN / null / missing safety and moved share (they compare False against every bar, so they get their own refusal), boundary passes at exactly -0.25 and 0.05, the inert flag, six identity cases, chain-state / prereg / each of the six C1 stages, a missing / wrong / edited / amendment-pinned result file, and a test that feeds the REAL `analyse_c1` writer's output through the guard | `G-LICENCE-*` (12 lines) |
| chain order | the licence or seal is read AFTER data is opened | `test_chain_refuses_an_unlicensed_c1_before_opening_any_data_or_child` (4), `test_chain_refuses_before_opening_any_data_without_the_seal` | `C-SEAL-FIRST` |
| G-DISK | work-volume free minus the chunk's ctx < 6 GiB | `test_disk_guard`, `test_chain_red_disk_guard_*` | `G-DISK` |
| G-IDENT (+ bridge) | one byte of P2's pick differs from the banked seam row; `--table-ref`: any of 64 proposals, 6 logits, the pick | `test_identity_*`, `test_chain_red_identity_stops_and_keeps_the_ctx`, `test_bridge_judge_pass_and_every_red_arm` | `G-IDENT-BYTES`, `C-IDENT-STOPS`, `BR-*` |
| T1 planted identity | > 1e-9 from P2's banked row, or the shifted control changes nothing | `test_t1_planted_identity_and_its_red_arms`, `test_chain_red_t1_identity_*` | `G-T1ID-*`, `C-T1ID-STOPS` |
| data clause | plan, bundle or search-meta token list != the chunk's | `test_launcher_*`, `test_chain_red_wrong_data_clause_in_the_search_meta` | `L-PLAN-SHA`, `C-CLAUSE-IN-META` |
| G-REACH (Step B's stamp, per bundle) | no PASS stamp, or one bound to another bundle | `Defective` stamp tests in `test_navpipe_guards.py` | `C-GATE-STAMP` |
| G-NOHEAD | any `--head*` flag, any arm but P2HEAD, `head_sha256` not null | `test_launcher_refuses_a_head_and_any_arm_but_p2head` | `L-NOHEAD`, `L-ARM`, `C-NOHEAD` |
| G-BANK | ctx deleted before its outputs are banked and re-hashed from disk | `test_delete_red_arms`, `test_bank_refuses_*` | `G-DELETE-*` (see the survivor, section 13) |
| analysis | the DQ3 terms | `test_c2_terms_each_term_has_a_red_arm` | (pure function, tested) |
Mutation result of the licence change: see section 13.

## 11. Outcomes and their committed next arms (RULE ZERO: no outcome ends at a verdict). Names and precedence are DQ5
The chain reports every failed term; the headline name is the first that applies in this order. One look: **no outcome licenses a second navtest look**, and nothing is tuned on navtest.
| outcome | rule | committed next arm (named now) |
|---|---|---|
| **C2_PASS** | all four DQ3 terms met | (1) `H-SEARCH-P2-1` reads CONFIRMED AT T1, one training seed, at the pre-registered bar, with its tier stamp; the MM writes the register row from `RESULT_NAVPIPE.json` only. (2) The Thor measurement of section 7 becomes the gate on any deployability claim and runs next (0 GPU for the script, ~1 Thor-hour). (3) If the latency budget (DQ4) is exceeded, the cheapest amortisation arm: a search-distilled pick on the fit sets (RL-val), selection tier. (4) The remaining gap to 94.55 is 5.73 minus the measured point (about 5.4 at +0.327), so the next lever is ranked by measured effect from the register, not chosen because the search worked. |
| **C2_FAIL_LOWER_CI** | lower CI <= 0 (flag NEGATIVE if upper CI < 0) | The search is CLOSED for P2's unchanged scorer on T1 at one training seed; reported plainly with the per-seed points. The report NAMES the next levers and what unblocks each: the P6 trunk (PI authorisation of trunk compute); an L2-BRANCH with a TTC-protecting term (own amendment, a `launch_gate.py` PASS token, the GPU lock); the SEARCH-DOSE (M = 256) result already read on C1's bundle. NEGATIVE adds: do not deploy; the selection-to-T1 gap is the finding to localise on fit sets (C1 vs C2 per-family deltas). Not a refutation of the idea at larger dose or on a better scorer. |
| **C2_FAIL_BAR** | lower CI > 0, point < +0.10 | Directionally real, below the minimum worthwhile effect. Next: SEARCH-DOSE's committed dose contrast (M256 - M64, C1 bundle) says whether more search moves the point; if it does, a raised-M search is built and validated on fit sets (CPU) before anything else, and a second navtest look would need a NEW sealed MM ruling. |
| **C2_FAIL_SEED_RANGE** | lower CI > 0, point >= +0.10, point <= the 3-seed range | The gain sits inside the inference-seed variance: reported with the three per-seed points. Next (CPU, fit sets only): estimate the inference-seed floor on B-REST + carve B with >= 8 search seeds, and whether the dose contrast shrinks it. No second navtest look without a new ruling. |
| **C2_SAFETY_MISS** | the three dPDMS terms met, at least one of dNC, dDAC, dTTC, dC < -0.25 | The TRADE-restricted arm (`PREREG_SEARCH_P2_CONFIRM.md` section 6 / Step B section 3: elites and pick restricted to candidates whose sigma(NC) and sigma(DAC) are >= the pick's inside the same set; no new head), built ON the named family, first on C1's bundle (CPU). A safety miss is also FLAGGED under every other name above. |
| **INFRASTRUCTURE STOP** | the chain ends before `ZZNAVPIPE_ANALYSIS_DONE` | Not an outcome and the look is not consumed (no endpoint is computed before every chunk is BANKED). Resume from the state file after the named cause is fixed (bridge identity: thread count first; disk; RAM; GPU lock). **No chunk output, score or CSV is opened or summarised before `ANALYSIS_DONE`; a partial run (`--max-chunks`) computes no analysis and its banked scores stay unread.** |

## 12. What this does NOT license
No second navtest look. No pooling C2 with C1 or carve B. No change to C1's bars or to DQ3's numbers. No use of a navtest result to choose a config, M, a threshold or a head. No deployability or latency claim while section 7 is UNMEASURED. No extension to P6, R0e-pick, RL or head candidates (the EP-ADMISSION gate stays in force for them). No capability claim beyond the T1 stamp. STRATEGIC stays PAUSED.

## 13. State of the guard change that made this draft sealable (filled by the agent at hand-back)
Full suite (navpipe_licence + navpipe + navpipe_s0 + navpipe_guards): `201 passed in 15.22s` (was 119 passed before the change). Mutation check (`mutation_check.py`, every `@GUARD` line disabled once in a scratch copy, suite must go RED, a catch caused by the box (ENOSPC etc.) is reported INFRA_ERROR, never CAUGHT): **60 of 60 judged guards caught**, survivors / infra errors: none, untestable without the GPU: ['MB-IDENT-REF']; the licence and seal guards (24): 24 caught. The unmutated control read green in every one of the 1 batches. Log with the first failing test per mutant: `raw/mutation_check.log`; merged: `raw/mutations_merged.json`.

## 14. DECISIONS PUT TO THE MASTER MIND (DQ1..DQ11; each with my recommendation; I have chosen none)
- **DQ1, how C1's result sha is bound.** (a) typed `--c1-result-sha256` on the command line, plus an optional amendment pin if one exists [recommended: lets this amendment be sealed BEFORE C1 reads, so nothing C1 shows can shape it]; (b) seal after C1 reads and pin the sha inside; (c) both. Code supports all three. The typed sha is the operator's independent record of which file; write it into the register when C1 reads.
- **DQ2, what "moved share" means in the licence.** (a) `inert.frac_moved`: the share of pooled (token, seed) outputs whose ADE vs the pick exceeds 0.5 m, the quantity whose shortfall IS O4_INERT [recommended: the ruling says "(not INERT)"]; (b) `moved_share`, the mean of the search's own `moved` flag, which is true for any non-zero move however small. Different numbers on the same file.
- **DQ3, inference-cost status at seal.** (a) UNMEASURED, with the section 7 protocol committed and the claim restriction in force [recommended: it does not block the look and costs nothing now, and the PASS claim is capped]; (b) MEASURED before the look, which makes a Thor slot, an agent and a latency budget blockers of C2.
- **DQ4, the latency budget.** None is stated anywhere found. (a) Report the measured ms per token and the ratio to P2's pick-only forward with no pass/fail until a budget exists [recommended]; (b) the MM states an absolute budget now (the vehicle's planning cycle); (c) a relative budget (<= k x P2). Needed before section 7 can gate anything.
- **DQ5, outcome names and precedence.** The order in section 11 (lower CI, then bar, then seed range, then safety, with the safety miss always flagged) versus safety-first (a safety miss headlines whenever present). [recommended: section 11's order: a safety dip beside a null gain is noise, and the TRADE arm is only meaningful when there is a gain]. The names are mine and may be replaced.
- **DQ6, P2's pick baseline.** (a) the banked CSV row plus the planted identity on 336 tokens (24 per chunk) [recommended: tested and pinned]; (b) re-score all 12,146 picks through the identical route, +about 18 min of harness time (15.9 s x 14 + 0.0898 s x 12,146, ESTIMATED) but a code change, which removes the NT-FIX1 harness-version question entirely.
- **DQ7, the fixture precondition.** (a) operator-checked, as built; (b) code-checked: a pin `fixture_record_sha256` and a guard that reads a fixture record (verdict IDENTICAL, code shas equal the pinned ones), same pattern as the licence [recommended (b) as a follow-up; not built here because the fixture chain writes no such record yet].
- **DQ8, seal timing and re-pin policy.** The numbers are already fixed by two sealed rulings, so sealing after C1 cannot move the bar, but outcome names, next arms and the cost status could still be shaped. (a) seal when the landed code is final and the fixture reads IDENTICAL, before C1 reads if possible [recommended]; (b) if C1 reads first, seal anyway and record that. If any pinned code changes after the seal the chain refuses (`SEAL_PIN_MISMATCH`); the remedy is a "Revision 1" that re-pins the code with the terms text byte-identical, as C1 did.
- **DQ9, a seal helper.** The pins below need about 25 sha256 values and the sidecar. [recommended: yes, a `seal_c2_amendment.py` modelled on `seal_spc.py` (fill pins, write the sidecar and `SEALED_AT`, run the chain's own seal check). Not built here: it is outside the brief and would be one more file needing a pin.]
- **DQ10, GPU venue.** (a) the dev-box GPU, a lock hold per chunk, as built [recommended: its estimated rate (1.4736 s/token) is MEASURED there]; (b) a pod. Either way the bridge must pass bit for bit against the banked seam; on which device the banked seam was produced is UNVERIFIED, so (a) may fail the bridge for a device reason.
- **DQ11, the optional table identity.** `--table-ref` adds an identity over all 64 proposals and the six logits against `table_p2_full.npz`, so the fan the search scores against is checked, not only the pick's 8 poses. [recommended: ON]. The file physically holds navtest `pdms`/`sub`/`valid` for 64 proposals per token; the pipeline reads `token`, `proposals`, `logits` and `pick` only (a test spies on `np.load`), so enabling it is a read of the file, not of its labels. If the MM prefers no contact with that file at all, the identity is the pick's 8 poses only.

## 15. Pins (machine-readable; the chain parses this block; values marked `@@SEAL@@` are computed at the seal by the MM's tooling; this agent did not hash the banked CSV, seam or table because they hold navtest scores)
<!-- PINS-BEGIN -->
```
mode navtest
token_file_sha256 927bbe3838dd70af5fa27eadceb09f55acc9c1cfd04cf0a6b520f1519175a955
chunk_plan_sha256 b3b609192fcd97971bce6771b5322caef853ddc70257ae017c55b5da5368df36
ckpt_sha256 ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51
config_sha256 299c6e63dd3016100149f88b6ca498529d79f8a5fbcfc6845753c8121d62c441
head_s0_sha256 bded124e1322b56d9bfd749f73b22c86430ef41f6829856927e5a08831320df4
banked_seam_sha256 @@SEAL@@
banked_csv_sha256 @@SEAL@@
c1_prereg_sha256 658ba1576af175a9a345692e284b1515604a6065279b921b2a4ca6dead6529d1
licence_ruling_sha256 fa7a26046a2fb5f8d8e69b2644c97a6cfce8907201d40d5d3c5be5358560ba6a
inference_cost_status UNMEASURED
bar_dpdms_points 0.10
bar_safety_points -0.25
harness_script_sha256 94f7b024ab2a375a62cf7268e85b0d245ac81ff7b69f2f0c4a9295050ab962b7
seam_module_sha256 ceb2c0f84ea2f7dd45be0c7bb61d98e92ab4e113c53bffaf6235abdd9b8db070
code:navpipe_core.py @@SEAL@@
code:navpipe_chain.py @@SEAL@@
code:navpipe_launch.py @@SEAL@@
code:navpipe_analyse.py @@SEAL@@
code:make_navtest_bundle.py @@SEAL@@
code:navpipe_bridge.py @@SEAL@@
code:navpipe_gpu_stage.py @@SEAL@@
```
<!-- PINS-END -->
Notes on the block: `mode navtest` is read by `G-SEAL-MODE` (an S0 amendment pins `mode s0bank`). `c1_result_sha256` is deliberately absent (DQ1 (a)); add it only if the amendment is sealed after C1 reads. The block carries no `admission_record_sha256`: that pin was removed from `REQUIRED_PINS` with the guard change.
