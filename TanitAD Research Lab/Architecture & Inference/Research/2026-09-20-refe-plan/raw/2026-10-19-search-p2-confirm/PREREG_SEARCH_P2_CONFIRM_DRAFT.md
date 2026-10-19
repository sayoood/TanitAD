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

| structure (tokens / logs) | scheme | true effect (pts) | P(O1 verbatim) | P(2-term: point >= 0.30 and lo > 0) | P(sign: lo > 0) | P(NULL: CI contains 0) | P(NEGATIVE: hi < 0) | mean CI half-width |
|---|---|---|---|---|---|---|---|---|
| B_REST (905 / 24) | R | 0.327 | 0.557 | 0.557 | 0.934 | 0.066 | 0.000 | 0.228 |
| B_REST (905 / 24) | R | 0.16 | 0.102 | 0.109 | 0.270 | 0.725 | 0.005 | 0.226 |
| B_REST (905 / 24) | R | 0.0 | 0.004 | 0.009 | 0.018 | 0.897 | 0.086 | 0.224 |
| B_REST (905 / 24) | S | 0.327 | 0.574 | 0.574 | 0.937 | 0.063 | 0.000 | 0.222 |
| B_REST (905 / 24) | S | 0.16 | 0.110 | 0.121 | 0.265 | 0.728 | 0.007 | 0.224 |
| B_REST (905 / 24) | S | 0.0 | 0.008 | 0.013 | 0.019 | 0.899 | 0.082 | 0.225 |
| B_REST (905 / 24) | E | 0.327 | 0.342 | 0.358 | 0.488 | 0.511 | 0.000 | 0.391 |
| B_REST (905 / 24) | E | 0.16 | 0.064 | 0.070 | 0.080 | 0.890 | 0.030 | 0.380 |
| B_REST (905 / 24) | E | 0.0 | 0.005 | 0.008 | 0.008 | 0.793 | 0.199 | 0.375 |
| carveB_replica (400 / 24) | R | 0.327 | 0.477 | 0.484 | 0.642 | 0.357 | 0.000 | 0.326 |
| carveB_replica (400 / 24) | R | 0.16 | 0.080 | 0.091 | 0.098 | 0.867 | 0.035 | 0.326 |
| carveB_replica (400 / 24) | R | 0.0 | 0.009 | 0.018 | 0.018 | 0.854 | 0.129 | 0.328 |
| carveB_replica (400 / 24) | S | 0.327 | 0.450 | 0.455 | 0.627 | 0.372 | 0.000 | 0.324 |
| carveB_replica (400 / 24) | S | 0.16 | 0.083 | 0.096 | 0.102 | 0.864 | 0.035 | 0.329 |
| carveB_replica (400 / 24) | S | 0.0 | 0.004 | 0.009 | 0.009 | 0.852 | 0.140 | 0.323 |
| carveB_replica (400 / 24) | E | 0.327 | 0.312 | 0.333 | 0.428 | 0.570 | 0.002 | 0.416 |
| carveB_replica (400 / 24) | E | 0.16 | 0.053 | 0.061 | 0.067 | 0.876 | 0.057 | 0.398 |
| carveB_replica (400 / 24) | E | 0.0 | 0.004 | 0.006 | 0.006 | 0.818 | 0.175 | 0.404 |
| navtest (12146 / 136) | R | 0.327 | 0.786 | 0.786 | 1.000 | 0.000 | 0.000 | 0.065 |
| navtest (12146 / 136) | R | 0.16 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.064 |
| navtest (12146 / 136) | R | 0.0 | 0.000 | 0.000 | 0.021 | 0.953 | 0.026 | 0.065 |
| navtest (12146 / 136) | E | 0.327 | 0.581 | 0.581 | 0.997 | 0.003 | 0.000 | 0.176 |
| navtest (12146 / 136) | E | 0.16 | 0.079 | 0.085 | 0.474 | 0.525 | 0.000 | 0.177 |
| navtest (12146 / 136) | E | 0.0 | 0.001 | 0.002 | 0.013 | 0.916 | 0.071 | 0.178 |

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
Each DQ gives the options, the design-calc numbers (section 8; R = random log effects, E = conservative; ESTIMATED), and my recommendation. The numbers are for C1 (905 tokens / 24 logs) unless stated.

**DQ1. What does C1 count as CONFIRMED?**
- (A) **O1 verbatim only** (the ruling's wording). At a true +0.327 it passes 56 % (R) / 34 % (E); at +0.16, 10 % / 6 %; false pass at 0: 0.4 % / 0.5 %. A fail is mostly the point bar's noise (point sd about 0.13), so it cannot refute.
- (B) **the sign bar only** (lower CI > 0). True +0.327: 93 % / 49 %; +0.16: 27 % / 8 %; false pass at 0: 1.8 % / 0.8 %.
- (C) **two named outcomes from the one read**: O1 = CONFIRMED (size and direction), O2 = DIRECTIONAL (lower CI > 0, point below +0.30 or the seed-range term). No new look is spent; the O1 bar is unchanged.
- **Recommendation: (C),** written in section 6. It keeps L1's bar verbatim as the word "confirmed", and it stops a true effect near the observed one being read as "not confirmed" 44-66 % of the time (1 - P(O1) at +0.327, R / E).

**DQ2. What licenses the EP-ADMISSION bank build and, through it, the navtest look?**
- (A) O1 only (the strict reading of Step B decision 0). (B) O1 or O2.
- At the observed +0.327 an O1-only licence blocks navtest in 44 % (R) / 66 % (E) of reads, at +0.16 in 90 % / 94 %. Navtest is where the size is resolved (half-width 0.065 R / 0.178 E; sign bar at +0.16: 100 % / 47 %), and navtest is still gated by admission and by its own bar (DQ3).
- **Recommendation: (B).** Nothing is licensed by C1 alone beyond building the bank and asking admission.

**DQ3. The navtest (C2, T1) bar, to be sealed in its own amendment.**
- (A) L1's O1 verbatim on T1 PDMS: at +0.327, 79 % (R) / 58 % (E); at +0.24 only 5 % / 25 %. (B) the sign bar (lower CI > 0) with the point and the four families reported: at +0.16, 100 % / 47 %; at +0.08, 71 % / 12 %; false pass at 0, 2.1 % / 1.3 %.
- Navtest P2 is 88.82 [87.86, 89.77] (T1, MEASURED), against the paper's 94.55 (PUBLISHED). A gain smaller than +0.30 is still what the programme wants, so a +0.30 magnitude bar would call a real +0.24 a miss 95 % of the time (R).
- **Recommendation: (B)** for the decision, with a pre-stated minimum worthwhile effect to be chosen by you (I do not have a justified number), and the safety families and the per-seed range as sealed O1 terms.

**DQ4. Keep the seed-range term (point > max - min of the 3 per-seed points)?** It passes 90 % at +0.327 and 70 % at +0.16 at B-REST, so it is not what binds. **Recommendation: keep** (it is the only guard against the inference-seed variance this rig has been shown to have).

**DQ5. May a pooled carve B + B-REST line (1,305 tokens) appear?** Never as the verdict (your ruling: the confirm is C1 alone; carve B is post hoc and shares the same 24 logs). **Recommendation: allow it only as a descriptive line labelled POST-HOC-POOLED, with no outcome attached.** Its power was not simulated.

**DQ6. EP-ADMISSION's bank geometry (section 7).** `ep_admission.py` is hard-wired to S0 (117 logs, 15,808 sets, a C-KEYS bijection onto P2's S0 bank), and its error rates were simulated there.
- (a) Build a bank adapter and re-simulate false refusal and power at 24 logs / 905 tokens x 3 seeds under an amendment sealed before the bank is read (CPU, about 1 h, ESTIMATED; the R1 simulator exists).
- (b) Admit on an S0-geometry bank of the search (P2 + search on S0's 8,922 in-sample navtrain tokens: needs a new GPU forward, hours, and S0 is in-sample for P2).
- (c) Waive A1 / A2 for the search (contradicts your ruling's decision 9; I do not recommend it).
- **Recommendation: (a).** If the re-simulated power at 24 logs is too low to decide, fall back to (b) and say so.

**DQ7. The SEARCH-DOSE arm (the pinned O4 config, M = 256) after C1.** It reuses the B-REST bundle and costs CPU only (about 4 h ESTIMATED if linear in M, which is 4x). **Recommendation: not in this seal; its own amendment after the C1 read,** because it is a second lever on the same tokens. RULE ZERO is met by naming it now and by the chain leaving the bundle on disk.

**DQ8. C-ID strictness.** The brief says bit for bit; I implemented bit for bit on 16 carve-B tokens (8 moved, 8 never moved) for the bundle rows, the ctx rows and the search outputs. UNVERIFIED hazard: a CPU float reduction can differ in the last bits across thread counts; the search is run at `--threads 4`, the Step B setting. **Recommendation: keep bit for bit; on a miss, diagnose threads first; do not loosen the gate silently.**

## 12. The chain, the controls, the pins and the seal
**Chain** (`code/spc_chain.py`; ONE detached process; `CUDA_VISIBLE_DEVICES=-1` for every CPU child; the GPU stage only through `launch_stepA_ctype.py run`, which takes the dev-box lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` the way `l2fresh/code` does and never removes or touches a foreign holder's lock; refuses (exit 5) without the seal; markers `ZZSPC_*` at line start; resumable): START -> SEAL_OK -> DATA_OK -> GPU_STAGE (bridge, the 16-token C-ID bundle, the B-REST bundle: `ZZBUNDLE_OK 16`, `ZZBUNDLE_OK 905`) -> CID_PASS -> REACH_PASS -> SEARCH_DONE -> LABEL_DONE -> ANALYSIS_DONE -> `ZZSPC_DONE <outcome>` with per-family lines, the NAMED next arm and the stamps. Any `ZZSPC_STOP <reason>` is a stop (exit 3 gate, 4 timeout, 5 seal / data, 6 search, 7 label / RAM, 8 analysis, 9 crash).
**Controls:** every gate in section 5 has a RED test that must make the chain refuse (`code/test_spc.py`); the seal guard has one RED arm per pin class.
**Pins** (machine-readable; `seal_spc.py` writes the sha256 values; the chain parses this block; the Step B values are copied from `PREREG_L2_STEPB.md` PINS and re-checked at the seal):
<!-- PINS-BEGIN -->
```
# Step B pins, copied verbatim from PREREG_L2_STEPB.md (the arm is IDENTICAL; seal_spc.py re-checks each against the file / constant)
carve_sha256 0629811ff5d05879eb62abc5726005ccc4924f03539423003503a099ba29577c
head_s0_sha256 bded124e1322b56d9bfd749f73b22c86430ef41f6829856927e5a08831320df4
head_s1_sha256 86f0b20f6ab0d2c46af6b3ae9874b73cc9d94a8065cf7fc0344acbeadcfd4654
ckpt_sha256 ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51
config_sha256 299c6e63dd3016100149f88b6ca498529d79f8a5fbcfc6845753c8121d62c441
config_o4_sha256 1ebd719617e8978a03cea0254edd6f531504c1d070792d066736130ceb929013
fresh_prereg_sha256 5006ed844979578f4c42599cd79b09c15bd07862d9c9bda94b21481d5e8d4e54
code:analyse_stepB.py b829d932b42941a397890c3edc10a1ee650da1fe3211c137acb83ade8734db7d
code:cost_stepB.py 2aaea18cca37a020514b23674e723e1588e307a018cd908bf600226ac5ee11df
code:fake_scorer.py 249a603e5e4f0a9041342b8e932ae13113aad873243841df1941af92c04cb6bb
code:g_reach_stepb.py e3cf8eaf892635177d86ce2b838a715be7a3516dd73af007027e30dc3199d47b
code:l2_scorer.py 8584bb1e91a04f247e79552910651330d2e4dff1f3c7248bf8c8a0fdb97be286
code:label_stepB.py 9e377fd8d0dfdfb6a9af4733fa3546e6d7428e9fecea8bde678ef2897061fd1c
code:make_test_head.py 3031194705f35aa53d68cd0d22cb87f949ae54dc598844838b2e9e25ac2de8a4
code:stepB.py 74e60d3de6f16aa1bcb3a9503cba54d81bc6e79fa55e4fe6ecb9efc60e390ea8
code:stepB_chain.py e8d63de0234736437973b18ac54422d61d60ebf594ef325d7ba1626db625fd00
code:test_stepB.py cc1a62dd39687087dff5575c913ec4db0c64e5bd20427d0f465292c200bd9332
code:test_stepB_chain.py ba294cc0fc4440e8ab909294f188a8b2c2ce26649a2b088a1d417b74a75a8005
l1:stepA.py 05d182c7cf1b9e44665b5f9928a35dfa2dd218e89bf7846bdee8a230456313f8
l1:tts_cem.py 9dea96038d18ae0b7f7a5d522939cb50c5424b372a6de4282ab49c02977bbedb
l1:real_scorer.py 05c4b4fb771ac924cbbf291b4e019f854411c318dc57e80ae09b02f3312ffbfa
l1:label_perturbations.py 9e2caca122d1230ff920a3d4722bbb9591d39426134d03af01684fc130c471e5
ruling_sha256 d8c2b150bc2a2210158180e8b9b901d602323aee64d661cd289334f52f425167
draft_sha256 8a9a4dd021c4ae828864c8fc8e517f6f725f4dd2a90f3d629c279b143c33846f
# this package
b_rest_tokens_sha256 3a85cc8124b3dd02297fffa176787b629369a75d82ed52c4440b32aa4cc70c4b
carve_b_tokens_sha256 0629811ff5d05879eb62abc5726005ccc4924f03539423003503a099ba29577c
step_a_tokens_sha256 211e26406f5b4eb083ae38b1c73839826d560042cd75ccf9ff6656262e7c0ffd
bridge_bundle_script_sha256_lf 8e544b8676c18c537076a52243df26a761ae299397fc25ec69653eedd40aa237
raw:b_rest.json 844d781f7611b7eec414a5dea522ec128fac9780d3caa3ec89e30d434592a0eb
raw:cid_sample.json c626d137a8c89717f8ac9d40300e6a865d2ef743ce05eed2b5c4643d2efa0ab4
raw:design_calc.json 851b999f80e5b910027e49308c0f7135e0b4a24c11aaff987b7665e71ca9f5a6
raw:cost.json 78ccf5e777bde1d3ca3e69e62a79499e792d1935c2a3d3bd5f14035766794ea8
code:spc_launch.py a97bf821f2f632ed7b6f653d1744c58c9000634d96c993651459c48bf8d11f0d
code:spc_chain.py c64071190e7ab0894cd7f50aba4a93556e4db222d8bf36ffc155e6b6e9654e4c
code:spc_gpu_stage.py 159feace524ae31a6f57ab1c6ffd8e8a37f188711e33b29d567deb505ba42ba9
code:spc_cid.py e50595f922245e9382b6102fdcfe22160d5d14006f59869ea062b13a7ca0e461
code:spc_analyse.py 147f7598ad664c13d65423336fbc4160547f9e10e27a1060eef3f98e10018870
code:b_rest.py df9ba008301cb73094dbf1f31ef6f8a38a6162120b8b9cec8abaf03f66ba2b1a
code:make_cid_sample.py 4a53dfa728746eccc44f725c7bd102f80354980434d1e307b3f30f670eeff4ec
code:design_calc.py c365517f3d58dd3d3cd783d64c0e21433bd1555184b67c9c3f18c600accd9f86
code:cost_spc.py 35d43b41bc8f7086c364fb7b75302d6b854f46fa31db7da8da4f97292cfd07de
code:power_table.py f457b9d020f3a12047cf8cc2e33ef918c802e8e39af51a8d280971437529cc1f
code:test_spc.py f0d6d81601dc767188192cda5397c689401e6a52c0093cd9fab080ee9da1097f
code:mutation_check.py 995a9aaf6ad78246ade3baa8284a2d2ac2cba48ad100f065d56838d406e24bc9
code:seal_spc.py d24d0f835d21084d9c4c503ff6b8b91ea3a380bb9ffad4592aee5166bd560aa3
```
<!-- PINS-END -->

## 13. Not licensed
No navtest look, no capability claim, no registry row beyond the read's own. A C1 O1 does not admit anything: EP-ADMISSION is separate and frozen, and its bank geometry is DQ6. The PI's ruling on the strategic pause is untouched: STRATEGIC stays PAUSED.
