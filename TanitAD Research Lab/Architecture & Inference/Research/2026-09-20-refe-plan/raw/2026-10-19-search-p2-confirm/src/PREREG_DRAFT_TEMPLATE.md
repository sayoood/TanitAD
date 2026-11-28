# PREREG SEARCH-P2-CONFIRM (DRAFT, NOT SEALED): does the CEM search on P2's OWN unchanged scorer gain again, on tokens it has not been read on?

Status: **DRAFT** (written 2026-10-19 ~23:30 by the SEARCH-P2-CONFIRM build agent for the Master Mind). The bar choices are put as DQ1..DQ8 in section 11 and are NOT chosen here.
**No B-REST token has been bundled, searched or labelled. No outcome number of this arm on B-REST exists. Nothing in this file is an outcome.** The one number that motivates it is carve B's DESCRIPTIVE control (section 0), which is post hoc and is confirmed only by its own sealed read.
It edits none of: `PREREG_L2_STEPB.md` (sealed, sha256 `c661588c952d1f8bf65474f9d6346cdb243eb8cf5e882101e040a9cd4f03172b`; its O1 bar and C8 config), `PREREG_L1_SEARCH_ADMISSION.md` (`a76f62d8043e4cc9d6c73b35cf2884f27027a1e01bcd3deec658380cc58bd801`), `PREREG_L2_FRESH.md` (`5006ed844979578f4c42599cd79b09c15bd07862d9c9bda94b21481d5e8d4e54`, carve B), `PREREG_EP_ADMISSION.md` (`7147e3dfc2984a504b8c42f96e0f2b4ed980bcc668cf879a781bf0f3c02f8367`).
Evidence classes: MEASURED (ours, artifact named), INHERITED (another package, not re-verified), ESTIMATED, HYPOTHESIS. STRATEGIC is PAUSED (PI ruling 2026-09-19, ITEM 25): printed as PAUSED, never as a pass, never extended to the other three families.

## 0. Why, and what is NOT being claimed
- **MEASURED** (`GOALS_AND_CLAIMS.md` `D-REFE-L2-STEPB-1`; `l2stepb/run/RESULT_STEPB.json` arms.P2HEAD; re-derived bit for bit by `code/design_calc.py` with L1's own `ratio_ci` at B = 10,000): on carve B (400 tokens, 24 logs, selection tier, SEEN-SCENE) the CEM search on P2's own scorer, warm-started at P2's pick, gained **dPDMS +0.3273 [+0.0565, +0.7216] points** (3 search seeds 0.398 / 0.184 / 0.400, range 0.216; dNC 0, dDAC +0.17, dEP +0.30, dTTC +0.25, dC 0; 16.5 % of tokens had a non-zero token delta).
- It was the DESCRIPTIVE control of a different test (never gated, 1 of 3 arms read). It is **POST HOC** and the largest measured effect in this line, so RULE ZERO makes it the lever to confirm and the evidence rules forbid promoting it without its own sealed read. A winner's-curse discount is plausible, so the power tables below run at +0.327 AND at about half of it (+0.16).
- This read answers one question: **does the search gain again on 905 token-fresh, LOG-SHARED tokens?** It does not answer "does it generalise to scenes the trunk has not seen / is it T1" (READ C2, the navtest look: its scenes are disjoint from navtrain by construction), "would another training run agree" (H-ESTIM-SEED-1: one training seed) or "would another inference run agree" (3 search seeds are the only inference variance sampled).

## 1. The arm (IDENTICAL to Step B's P2HEAD; sha-pinned; nothing is tuned)
| lever | value | source / pin |
|---|---|---|
| scorer | P2-s0, `model_final.pt` sha256 `ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51`, **no head substitution** (the argv carries no head flag; the output meta must read `head_sha256: null`) | Step B pins |
| search | `tts_cem.cem_search`, K = 5, M = 64 per iteration scored as one complete set, elites 8, warm start at P2's pick (`bundle['pick']`), final pick re-scored inside the original fan | L1 sections 4/5; Step B section 1 |
| config | C8, `stepB_config.json` sha256 `299c6e63dd3016100149f88b6ca498529d79f8a5fbcfc6845753c8121d62c441` (sigma0 0.08126 / 0.15; clips u in [ln 0.85, ln 1.15], d in +-0.3 m) | Step B section 2 |
| search seeds | 20260927 / 20260928 / 20260929 | `tts_cem.SEARCH_SEEDS` |
| code | `stepB.py`, `l2_scorer.py`, `g_reach_stepb.py`, `label_stepB.py`, `analyse_stepB.py` and L1's four files, each sha256-pinned to its Step B value (section 12) | Step B PINS block |
| new code (pinned at the seal) | `spc_launch.py` (passes `carve_sha256=` to `stepB.run_search_cli` / `g_reach_stepb.run_gate`, a parameter no Step B CLI flag reaches), `spc_chain.py`, `spc_gpu_stage.py`, `spc_cid.py`, `spc_analyse.py`, `seal_spc.py` | section 12 |
The search is an inference-time search on CPU (no optimiser, loss or gradient). `launch_gate.py` governs TRAINING launches; the Step B ruling's G-REACH-STEPB (a sha-bound stamp, minted on EACH new bundle) is the gate that applies (MM ruling decision 8).

## 2. Data: B-REST (enumerated, not assumed)
- **B-REST = P2's held-out plan tokens (1,705 tokens / 24 logs, `heldout_tokens.json`) in NEITHER Step A's 400 NOR carve B's 400.** MEASURED by `code/b_rest.py` (`raw/b_rest.json`): **905 tokens, 24 logs** (the ruling's "905" is verified, not assumed). |B-REST ∩ Step A| = 0, |B-REST ∩ carve B| = 0, |Step A ∩ carve B| = 0, and 400 + 400 + 905 = 1,705; every token has a banked row in P2's held-out table (G2b needs it). **Token-list sha256 `3a85cc8124b3dd02297fffa176787b629369a75d82ed52c4440b32aa4cc70c4b`** (sha256 of the sorted tokens joined by newlines = the bundle's `ctx.json` `tokens_sha256`). Tokens per log: min 5, median 21.5, max 131; effective cluster count (sum n)^2 / sum n^2 = 12.4.
- **Footing = carve B's for P2:** SEEN-SCENE (all 905 tokens are inside M-SCENE, the final-run target bank: declared exposure A1.3); M-PDM overlap 0 (G0 A1.2 PASS, its deliberate-overlap control reading FAIL as it must; `raw/b_rest.json` `g0`). **Token-fresh to the search (the search was read on carve B only), LOG-SHARED with Step A and carve B (the same 24 logs): NOT a fresh-log test.**
- **The confirm is READ C1 alone, never pooled with carve B** (MM ruling). A pooled number may appear only as a labelled descriptive line, never as the verdict (DQ5).
- **Data clause (refuses):** the bundle's token list must hash to the sealed B-REST sha in four places (chain, gate, launch via `carve_sha256=`, ctx.json / ctx.npy sha), exactly as Step B did for carve B. One token replaced / dropped / swapped / appended and an empty list are RED arms (section 12).
- **Disk:** the B-REST ctx.npy is 3.31 GiB (one file, uint16 bf16 bits, [905, 7680, 256]) on D: (24 GB free at measurement, exFAT); the chain refuses to start the GPU stage with less than 12 GiB free on the target volume.

## 3. Reads
| read | tokens / logs | footing | gate | status |
|---|---|---|---|---|
| **C1 = B-REST** | 905 / 24 | SEEN-SCENE, token-fresh, LOG-SHARED, SELECTION tier, label-only | this package's chain | the replication AND the source of the EP-ADMISSION bank inputs (section 7) |
| **C2 = navtest T1** | 12,146 / 136 (`navtest_full_tokens.json`) | **T1** (NAVSIM v1.1 pseudo-sim PDMS), ONE look | **only if EP-ADMISSION admits** (section 7), plus its own sealed amendment (bar: DQ3), plus a chunked pipeline that does not exist (44.5 GiB of ctx, section 10) | NOT built, NOT licensed |
| **C3 = unseen stratum U** | - | - | **not available (no public frames; `C:/Users/Admin/par18/unseen_s2s4/out/RESULT.md`: the nuPlan public val sensor archives hold frames for 225 logs, 214 navtrain + 11 U, and 0 of the 122 fetch-list logs; INHERITED from the coordinator, not re-read here)** | DROPPED; the scene-unseen read is C2 |
C1 needs the GPU lock once (the bundle). C2 needs its own sealed amendment before any token is touched; this draft fixes its gates and the shape of its bar only.

## 4. Endpoint, estimator, tiers (as Step B; nothing new)
- **Endpoint:** per token n and search seed s, d(n, s) = 100 x [true PDMS(search output) - true PDMS(P2's pick)], the pick's label the BANKED one; the token's value is the mean over the 3 seeds; **PDMS points** (Step B ruling decision 1).
- **Estimator:** the paired log-cluster bootstrap, `stepA.ratio_ci` over `stepA.boot_indices` (ratio of sums, WHOLE LOGS resampled), **B = 10,000, seed 20260927**. It answers "would another draw of LOGS say this?" and nothing else: not another training run (H-ESTIM-SEED-1), not another inference run (the 3 per-seed points and their range are reported beside it).
- **Tier and stamps on every line:** MEASURED, SEEN-SCENE, SELECTION tier, label-only (not T1, not navtest), LOG-SHARED (C1), one training seed, STRATEGIC PAUSED.
- **Labels:** `label_stepB.py` (5 plans per token: identity, 3 seeds, control) with the PLANTED IDENTITY enforced (the (0, 0) composite of the pick reproduces the pick's banked label to <= 1e-6, and the shifted control changes at least one label); exit 3 = STOP.

## 5. Gates (each has a RED arm that must make the chain REFUSE; section 12)
| gate | what it asserts | GPU |
|---|---|---|
| **G-SEAL** | this text's sha256 equals its `.sha256`, `Status: SEALED` is present, every pin equals the file / constant the chain will use; otherwise exit 5 | no |
| **G-DATA** | B-REST token list sha = the pinned value; disjoint from Step A's 400 and carve B's 400; 905 tokens / 24 logs; every token banked | no |
| **G-BRIDGE** | `bridge_g2b.py` (L2-FRESH; certified make_stepA_bundle sha `8e544b86...`) on `bridge3_sample.json`: IDENTITY on 3 tokens AND the one-pixel RED arm makes the bundle script's own G2b gate fire on all 3 (a blind gate certifies nothing) | yes, same lock hold |
| **G-BUNDLE (G2b)** | `make_stepA_bundle.py` prints `ZZBUNDLE_OK 905` only if the forward reproduced the banked fan, logits and pick bit for bit on EVERY token; the chain greps for it | yes |
| **C-ID** | the arm re-run on **16 carve-B tokens** (`raw/cid_sample.json`: 8 the banked search MOVED, 8 it never moved; fixed rule) through the SAME path (a fresh GPU bundle for those 16, then the CPU search) reproduces the banked `search_P2HEAD.npz` rows **bit for bit** (exec, theta, moved, scorer_in, pred_logit_pick, pred_logit_out, pick_exec, hist_*), AND the re-built bundle rows (traj_raw, logits, pick, sub, pdms, valid) and ctx rows equal carve B's. One differing bit = STOP | yes (the 16-token bundle) |
| **G-REACH** | `g_reach_stepb.run_gate(..., carve_sha256=<B-REST sha>)` on the B-REST bundle, with the final head s0 file as the gate's PROBE head (the gate's arm is L2 only; a P2HEAD stamp binds checkpoint, config, bundle and code and ignores the head): checks (a)-(f), all 8 mutation arms RED; the PASS stamp binds the B-REST bundle sha | no |
| **G-NOHEAD** | the merged search file's `meta.arm` = P2HEAD and `head_sha256` is null; the search argv carries no `--head` | no |
| **G-LABEL** | the planted identity at the labeller (section 4) | no |
| **G-ORDER** | the analysis refuses unless every gate above left its token in `chain_state.json` and the search / label files are for B-REST's tokens in bundle order | no |

## 6. Outcomes of READ C1, each with its committed NEXT arm (RULE ZERO)
The bar terms are the sealed Step B O1 terms (point >= +0.30; lower CI > 0; point > the 3-seed range; dNC, dDAC, dTTC, dC points each >= -0.25), plus the SIGN outcome the brief asks to be priced. **Which outcome CONFIRMS and which LICENSES is DQ1 / DQ2; the table is written with my RECOMMENDATION and names every outcome's next arm either way.**
| outcome (read once, C1 alone) | rule | committed next arm (named now) |
|---|---|---|
| **O1 CONFIRMED (verbatim)** | point >= +0.30 AND lower CI > 0 AND point > seed range AND each safety point >= -0.25 | (i) build the search-output EP-ADMISSION bank from C1's own outputs and labels (candidate block, section 7, already written), after DQ6; (ii) the navtest look C2 only after admission and its own amendment. In the SAME turn, because the search is the largest measured lever, the **SEARCH-DOSE arm**: the pinned O4 config `stepB_config_O4.json` (sha256 `1ebd719617e8978a03cea0254edd6f531504c1d070792d066736130ceb929013`, M = 256) on the SAME B-REST bundle (CPU only, no GPU), under its own amendment sealed before launch |
| **O2 DIRECTIONAL** | lower CI > 0 but O1 not met (point < +0.30, or the seed-range term); the four safety points are reported beside it and a safety miss is FLAGGED, not hidden | the SEARCH-DOSE arm on the same bundle; the admission bank is built only if DQ2 licenses it on O2, and then the navtest path. The size question then goes to navtest, whose power is far higher (section 8) |
| **TRADE** | the three dPDMS terms clear and a safety family is below -0.25 | the TRADE-restricted arm (Step B section 3: elites and pick restricted to candidates whose sigma(NC) and sigma(DAC) are >= the pick's inside the same set; no new head), built ON the TRADE |
| **O3 NULL** | the CI contains 0 | not a refutation (power at half the observed effect is 27 % / 8 %, section 8). the SEARCH-DOSE arm on the same bundle (cheap: CPU only). If it also reads NULL the search is CLOSED for P2's scorer on this footing and the report NAMES the next levers and what unblocks them (the navtest look is the only scene-unseen read, and stays gated by admission): the P6 trunk (PI authorisation of trunk compute) and an L2-BRANCH with a TTC-protecting term (its own amendment, a `launch_gate.py` PASS token, the lock) |
| **NEGATIVE** | upper CI < 0 | the search HURTS on this footing: reported plainly. Next: the levers named under O3 (the SEARCH-DOSE arm only if the moved share is < 5 %) |
| **O4 INERT** | the output differs from the pick by > 0.5 m ADE on < 5 % of the pooled (token, seed) outputs | the raised-M arm ONCE (Step B decision 5), then CLOSE with a named next lever |
**Stated expectation, before any data:** carve B moved the pick on 16.5 % of tokens, so INERT is not expected. If the effect is real at about +0.3, O1 and O2 together are likely and O1 alone is close to a coin flip at 905 tokens (section 8); if +0.327 was a post-hoc draw, O3 is likely. **A miss of the O1 point bar at a true +0.3 is the bar's own noise, not a refutation:** the point's sd at B-REST is about 0.13.

## 7. EP-ADMISSION: the candidate block, written BEFORE any data, and a geometry problem put to the MM
```
candidate:          P2 + CEM search   (the deployed system picks the SEARCH OUTPUT, so the object is a bank of SEARCH OUTPUTS)
checkpoint(s):      D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt  sha256 ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51   (P2-s0; ONE seed: the gate's "any seed" quantifier has n = 1 here)
search config:      stepB_config.json sha256 299c6e63dd3016100149f88b6ca498529d79f8a5fbcfc6845753c8121d62c441, K 5, M 64, elites 8, warm start at P2's pick, search seeds 20260927 / 20260928 / 20260929
bank:               built from C1's merged search file (pred_logit_pick, pred_logit_out) and its labels; path and sha256 are appended HERE BEFORE the bank is read: <to be appended>
appended:           2026-10-19 (this block) by the SEARCH-P2-CONFIRM build agent; the bank fields before the bank was read
```
- **What the frozen gate reads (MEASURED from `ep_admission.py`, tip 5ad79d0):** `load_bank_units(bank_dir, ranked_rows)` and `align_candidate`, whose C-KEYS check requires the candidate's keys to be a **bijection onto P2's S0 bank** (117 logs, 15,808 sets, `ranked_fit_logs.tsv` sha `a6127503...`). A B-REST search-output bank (24 logs, 905 tokens x 3 seeds) is **not that object**: C-KEYS would refuse it, and the gate's measured error rates (A1-r1 false refusal 2.7 % / 3.3 %, A2 at 1.05 false refusal 1.9 % / 1.8 %, DAC power 77-78 % at x1.25) were simulated on S0's geometry and **do not transfer to 24 logs**.
- So **admission on C1's bank needs (i) a bank adapter to `ep-admission-v1` (a set per token and search seed: the fan with the pick's slot and the search output as a candidate carrying its own label) and (ii) a re-simulation of the false-refusal and power tables at 24 logs / 905 tokens x 3 seeds** (the R1 simulator `ep_admission_sim.py` exists; CPU, about 1 h, ESTIMATED), both under an MM amendment before the bank is read. **This is DQ6.** Until then the chain banks the inputs (the merged search file with its logits, the labels) and **does not run `ep_admission.py`; it never prints an ADMIT.**
- Reported-only beside it: A3 / A4 / A5 of the search outputs (section 8b).

## 8. Power (ESTIMATED; `code/design_calc.py`, `raw/design_calc.json`, table `raw/design_calc_table.md`; reads NO B-REST label)
**Inputs:** carve B's already-read P2HEAD per-token deltas only (400 tokens, 24 logs, 3 seeds, dNC / dDAC / dTTC / dC), null-centred per series, then a true effect added to the four PDMS series (the safety series scaled with it). B-REST's, navtest's and U's STRUCTURE (tokens per log) comes from token lists only. **Estimator in the simulation:** the same paired log-cluster bootstrap with **B_SIM = 2,000 resamples per simulated read and 2,000 simulated reads per condition** (the real read uses 10,000; a 300-read check at 10,000 agrees within Monte-Carlo error: O1 0.533 vs 0.557, sign 0.933 vs 0.934); Monte-Carlo se of a probability <= 0.011. **Identity:** the real Step B P2HEAD read is reproduced exactly (+0.3273 [+0.0565, +0.7216], per-seed 0.398 / 0.184 / 0.400, range 0.216, safety points).
**Schemes (assumptions, not measurements):** R = random log effects (the question the interval itself answers, "another draw of LOGS"); S = B-REST and the carve-B replica only, log effects SHARED with carve B (empirical-Bayes shrunk); E = a conservative two-stage resampling that carries carve B's own log-mean noise (overstates variance). **MEASURED variance components of the carve-B dPDMS series:** sigma^2 within log 14.3 pts^2; the unbalanced ANOVA moment estimate of the between-log variance tau^2 = **0.0** (floored; ICC 0; with 24 logs the data cannot separate it from 0, so R is iid-like and E is the honest upper bracket). **The "about 77 % between-log" quoted for carve B belongs to the A1 concordance statistic (`design_calc_ball.json`), not to dPDMS.** Navtest and U assume carve B's heterogeneity: an ASSUMPTION about other populations (the effect on T1 navtest may differ in size and mix).

@@POWER_TABLE@@

**Reading it (all ESTIMATED):**
- **B-REST (905 tokens, 24 logs; schemes R / S / E):** at the observed +0.327 the sign bar is hit 93 % / 94 % / 49 % and the O1 bar 56 % / 57 % / 34 %: the +0.30 point bar binds (the point's sd is about 0.13), so even a truth equal to the observed estimate passes O1 about half the time. **At about half the observed effect (+0.16) the sign bar is hit only 27 % / 27 % / 8 % and O1 10 % / 11 % / 6 %.** At a true 0 the false-positive rates are sign 1.8 % / 1.9 % / 0.8 % and O1 0.4 % / 0.8 % / 0.5 %. Mean CI half-width 0.22 (R, S) to 0.38 (E).
- **Carve-B replica (the structure that produced the observed read):** sign 64 % / 63 % / 43 % at +0.327. The observed lower CI (+0.056) sitting close to 0 is typical of that structure, not a warning.
- **Navtest (12,146 tokens, 136 logs), assumption-laden:** half-width 0.065 (R) to 0.178 (E). The sign bar at +0.16 is hit 100 % (R) / 47 % (E), at +0.08 71 % / 12 %; the +0.30 point bar at +0.327 is hit 79 % / 58 %. **Navtest is where the SIZE of an effect below +0.3 can be resolved; B-REST cannot.**
- **What a NULL at B-REST means:** a true +0.16 reads NULL about 73 % of the time (R). A null C1 is weak evidence against a half-size effect, which is why O3 is not a refutation and carries a committed next arm.

**8b. H-SEARCH-ADMIT-A3, as a REPORTED, UNGATED quantity.** For every read the chain reports A3 (the scorer's aggregate concordance with the true PDMS over qualifying (pick, search-output) pairs, those whose TRUE aggregate differs by >= 0.05; L1's `stepA.admission_terms`, paired log-cluster bootstrap), A4 and A5, beside dPDMS, per search seed and pooled, from the banked `pred_logit_pick` / `pred_logit_out` and the labels. Carve B's banked values (Step B Amendment 1: 12-16 qualifying pairs; P2HEAD A3 1.000 on 14 pairs, L2 arms 0.75 / 0.625) are the reference. **The relation "a search's gain tracks the scorer's A3, not its A1" has ONE scorer in this read, so it cannot be tested here** (n = 1 scorer; carve B and C1 are two reads of the SAME scorer). It is stated, reported and ungated, and is tested only when at least three scorers have been read on the same footing (the P2 search, plus any future scorer arm). No verdict, bar or outcome depends on it.

## 9. The four families (never pooled; ADE is not a result; STRATEGIC PAUSED)
Each read per family with the paired log-cluster interval above, on the same tokens (`analyse_stepB.analyse`, unchanged):
- **LONGITUDINAL:** dEP and dTTC (points); the share of outputs that are speed-ups / slow-downs and dPDMS given each (the Step B mechanism: the L2 heads over-weighted EP and gave TTC back; P2HEAD's slow-downs gained +0.65). **Distance-keeping / headway: n = 0, the label-only tier carries no lead-agent headway row beyond TTC** (stated, not hidden).
- **LATERAL:** dDAC and dDDC (points), the mean |offset| of moved outputs, the heading deviation at the last non-repaired pose and the largest yaw-rate deviation against the pick (geometry, descriptive). **Curvature error and cross-track: n = 0, reason: a label-only selection tier with no closed-loop rollout; DAC and NC are the lateral-bearing rows.**
- **TACTICAL:** the share of outputs whose command class (`stepA.command_class`, a bearing rule on the end point; HYPOTHESIS: a proxy) differs from the pick's. **Manoeuvre-decision quality and tactical goal selection: n = 0, no manoeuvre decision is scored at this tier.**
- **STRATEGIC: PAUSED** (PI ruling 2026-09-19, ITEM 25). Not a pass.
- **Inference stays vision-only:** the search reads the same token and P2's own scorer; the privileged PDM labels are the endpoint's instrument, never an input to the search.

## 10. Cost (ESTIMATED from MEASURED rates; `raw/cost.json`)
| item | GPU | CPU wall (one process; the search at 4 threads) |
|---|---|---|
| (a) B-REST bundle forward (905 tokens, ONE locked forward; carve B took 636 s for 400 tokens, Step A 664 s) | **24.0-25.0 min** (the lock WAIT is extra: carve B's chain waited 2 h 20 min) | - |
| (b) P2HEAD search on B-REST, 3 seeds (Step B: 3.95 s per token for all 3 seeds, 1,581 s for 400) | 0 | **57 min central** (49 min at the best shard rate; the worst shard rate of any arm would give 4 h: a shared-box outlier) |
| (c) labels, 5 plans per token (Step B: 444 s for 400) | 0 | **17-27 min** |
| C-ID (16 tokens: a small GPU bundle + a CPU search), G-BRIDGE, G-REACH | about 6 min | about 3 min |
| **C1 total** | **about 30 min GPU** | **about 80 min CPU** |
| (d) the same on **navtest** (12,146 tokens): bundle forward | 5.4-5.6 GPU-h at the plan-token rate (UNVERIFIED: no banked fan exists for navtest; a new builder or bridge is needed) | search **12.8 h** (6.4 h on two processes); labels 3.7-6.1 h via `label_stepB`, or about 0.9 h via the NAVSIM harness seam route (P2's full seam scored in 1,106.6 s; UNVERIFIED, no seam builder exists) |
| navtest disk | ctx 44.5 GiB (D: has 24 GB) | **needs a chunked bundle -> search -> delete pipeline that is not built** |

## 11. DECISIONS PUT TO THE MASTER MIND (DQ1..DQ8; each with my recommendation and the numbers; I have chosen none)
@@DQS@@

## 12. The chain, the controls, the pins and the seal
**Chain** (`code/spc_chain.py`; ONE detached process; `CUDA_VISIBLE_DEVICES=-1` for every CPU child; the GPU stage only through `launch_stepA_ctype.py run`, which takes the dev-box lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` the way `l2fresh/code` does and never removes or touches a foreign holder's lock; refuses (exit 5) without the seal; markers `ZZSPC_*` at line start; resumable): START -> SEAL_OK -> DATA_OK -> GPU_STAGE (bridge, the 16-token C-ID bundle, the B-REST bundle: `ZZBUNDLE_OK 16`, `ZZBUNDLE_OK 905`) -> CID_PASS -> REACH_PASS -> SEARCH_DONE -> LABEL_DONE -> ANALYSIS_DONE -> `ZZSPC_DONE <outcome>` with per-family lines, the NAMED next arm and the stamps. Any `ZZSPC_STOP <reason>` is a stop (exit 3 gate, 4 timeout, 5 seal / data, 6 search, 7 label / RAM, 8 analysis, 9 crash).
**Controls:** every gate in section 5 has a RED test that must make the chain refuse (`code/test_spc.py`); the seal guard has one RED arm per pin class.
**Pins** (machine-readable; `seal_spc.py` writes the sha256 values; the chain parses this block; the Step B values are copied from `PREREG_L2_STEPB.md` PINS and re-checked at the seal):
<!-- PINS-BEGIN -->
```
@@PINS@@
```
<!-- PINS-END -->

## 13. Not licensed
No navtest look, no capability claim, no registry row beyond the read's own. A C1 O1 does not admit anything: EP-ADMISSION is separate and frozen, and its bank geometry is DQ6. The PI's ruling on the strategic pause is untouched: STRATEGIC stays PAUSED.
