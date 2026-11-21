# PREREG L2-STEPB (SEALED): L1 Step B, the CEM search, on P2's scorer with the L2 EP head

Status: SEALED 2026-10-19 (the exact time is in `SEALED_AT.txt`; this text's sha256 is in `PREREG_L2_STEPB.sha256`; the detached chain `code/stepB_chain.py` refuses to run without both, section 11)

Written 2026-10-19 by the L2-STEPB-BUILD agent as a draft (`PREREG_L2_STEPB_DRAFT.md`, sha256 `8a9a4dd0…`), amended by the Master Mind's ruling (**Revision 1, below, verbatim**) and SEALED before any Step B search ran.
**No Step B search has been run, no search output has been labelled, no dPDMS has been read, and no carve-B label or score was read by this agent: nothing in this file is an outcome.**
It is an amendment to `PREREG_L1_SEARCH_ADMISSION.md` (sha256 `a76f62d8043e4cc9d6c73b35cf2884f27027a1e01bcd3deec658380cc58bd801`, section 4 O1 and section 5), L1 Amendment 1 (sha256 `45f5c7917feef407b3ce18951e35f7abdcef5b7a9fb65e41e9cebd3d50fd72ff`, C8)
and `PREREG_L2_OFFPROP.md` (sealed, sha256 `3592a1039693dcb5cac1df465fa4dd597ee190d80a934d89fc765d223607a983`, outcome L2-ADMIT); none of the three is edited. It stands beside `PREREG_L2_FRESH.md` (sealed 2026-10-19 16:25, sha256 `5006ed84…`), which defines carve B.
Evidence classes: MEASURED (ours, artifact named), INHERITED (another package, not re-verified), ESTIMATED, HYPOTHESIS.

## Revision 1 (MM ruling, pre-seal)
The Master Mind's ruling on the draft, folded in VERBATIM (file `MM_RULING_STEPB.md`, sha256 `d8c2b150bc2a2210158180e8b9b901d602323aee64d661cd289334f52f425167`). **Where the body below (the draft, edited only where the ruling says so, each edit tagged `[Rev1 #n]` or `[Rev1 D0]`) and the ruling differ, THE RULING PREVAILS.**

````markdown
# MM ruling on the L2 Step B draft (Master Mind, 2026-10-19 ~16:50; no Step B search has run on real data, no search output has been labelled, no dPDMS has been read)

## Decision 0 (new): Step B runs NOW as a pre-registered DIAGNOSTIC on carve B; it licenses nothing alone.
- The draft's precondition (Step A O1 first) assumed Step B was the costly stage. MEASURED: it costs 1.8-2.7 CPU-h for three arms, with no GPU. It is also the only read of the actual endpoint (the true PDMS of the search output against P2's pick).
- Under RULE ZERO, the cheapest discriminating experiment toward driving quality is run as soon as its inputs exist, not after a gate the geometry says will probably read O3 (`MM_RULING_L2_FRESH.md`).
- **It launches** once the L2-FRESH B chain prints `CHAIN_DONE_NO_READ` with its gates holding, and once BOTH amendments are sealed.
- **Data:** carve B's bundle. Its token list must equal the sealed carve (sha256 `0629811f…`); a mismatch REFUSES.
- **Licensing is UNCHANGED.** EP-ADMISSION and the shared navtest look require a Step A O1 (READ 1 or READ 2) AND a Step B O1 on both head seeds. Every outcome of Step B is reported whatever Step A reads, so no result is a forking path: neither read can license alone, and both rules are fixed now.
- **Stamps:** SEEN-SCENE; the head is token-fresh and LOG-SHARED on B; SELECTION tier, label-only.

## The ten decisions
1. **Units: PDMS POINTS (100 × [0, 1]). RATIFIED, as an interpretation recorded before any data.**
   - L1's Step A thresholds are on [0, 1]. Its Step B bar is not stated in units, and +0.30 on [0, 1] (30 points) is unreachable; the motivating TOAD numbers are points.
   - In points the safety tolerance −0.25 equals ONE token in 400 losing a binary sub-score. That reading is coherent, and it is the strict one.
2. **The seed-spread term: dPDMS point > (max − min) of the 3 per-search-seed points. RATIFIED.**
3. **Two head seeds: the WORST of the two. RATIFIED.**
4. **P2HEAD control: KEPT** (descriptive attribution; it costs one arm). Without it, a gain cannot be assigned to the L2 head rather than to the search.
5. **NOT CLEARED ⇒ L2-BRANCH. RATIFIED.** O4 INERT ⇒ the raised-M widened arm once. If that is also INERT, the search is CLOSED for this scorer, and the report must NAME the next lever (L2-BRANCH, or the P6 trunk) with what unblocks it. Closing is never the end of the turn.
6. **Start and baseline = P2's pick. RATIFIED.** The L2 head was validated (A1) on perturbations OF P2's pick, so searching around that pick keeps the search inside the domain the head was admitted on. The in-fan re-ranking stays descriptive.
7. **Clips stay at the hull under O4's widening. RATIFIED** (same reason as 6).
8. **E7: NARROWED, with the reason recorded.**
   - `launch_gate.py` governs TRAINING launches (the PI rule of 2026-09-26: "if we are starting a training session"). Step B is an inference-time search: no optimiser, loss or gradient.
   - The right gate is G-REACH-STEPB, with its sha-bound PASS stamp (head, checkpoint, config, bundle, code) re-asserted inside the launch.
   - A profile whose G-LIVE, G-CKPT and G-CLOCK have no object would certify less than a gate built on the real risk.
   - My E7 phrase "Step B and beyond" now reads **"any arm that TRAINS with an L2 head (L2-BRANCH and beyond)"**. BACKLOG `L2-STEPB-GATE` is resolved by G-REACH-STEPB.
   - G-REACH-STEPB is re-run on EACH real head (s0, s1) before its arm runs. A stamp minted on the test head does not carry over.
9. **EP-ADMISSION's object: BOTH banks, and ADMIT needs both to admit.** The thresholds, estimator and rules stay frozen; only the candidate block (appended per read) names the objects.
   - (i) The frozen literal object: an in-fan bank of "P2 + the L2 EP row" (head seeds 0 / 1 = the candidate's two seeds), via a head-file input to `build_candidate_bank.py` (`l2_scorer.install_ep_head`).
   - (ii) A search-output bank in the same schema, built from Step B's labelled outputs. That is what the deployed system picks.
   - Both are built ONLY after a Step B O1 (not now). The candidate block is written before either bank is read.
10. **The TRADE-restricted arm: built ON a TRADE, not now.** Its spec is committed in §3 and is unchanged.

## The chain
After the seal, ONE detached CPU chain runs:
- G-REACH-STEPB on s0 and s1;
- the arms L2-s0, L2-s1 and P2HEAD (3 search seeds each);
- the labels, then the registered analysis;
- `ZZSTEPB_DONE` and the verdict line.

A failed gate stops it before any search. `CUDA_VISIBLE_DEVICES=-1`, no lock.
````

### How the ruling is applied to the body (a map, not a second source)
| ruling | applied in |
|---|---|
| Decision 0: a diagnostic on carve B now; licensing UNCHANGED; stamps | section 0, section 1 (data), section 7 (stamps), section 10 (licensing), section 11 (chain) |
| 1 units (PDMS points) | section 3 |
| 2 seed-spread term | section 3 |
| 3 worst of the two head seeds | section 3 |
| 4 P2HEAD control kept | section 1 |
| 5 NOT CLEARED => L2-BRANCH; O4 => raised-M arm once, then CLOSED with a NAMED next lever | section 3 table; `analyse_stepB.NEXT_ARM` (printed by the chain) |
| 6 start and baseline = P2's pick | section 0 |
| 7 clips stay at the hull | section 2 |
| 8 E7 narrowed | section 5 |
| 9 EP-ADMISSION on BOTH banks, built only after an O1 | section 6 |
| 10 TRADE-restricted arm built ON a TRADE | section 3 table |
| the chain | section 11 |

## 0. When this runs, and what it is
- **[Rev1 D0] Step B runs NOW, as a pre-registered DIAGNOSTIC on carve B; it licenses nothing alone.** The draft's precondition (Step A O1 first) assumed Step B was the costly stage; it is not (MEASURED, section 8: 1.8-2.7 CPU-h for three arms, no GPU) and it is the only read of the actual endpoint (the true PDMS of the search output against P2's pick).
  It launches once the L2-FRESH B chain prints `CHAIN_DONE_NO_READ` with its gates holding and BOTH amendments are sealed (section 11). **Licensing is UNCHANGED:** EP-ADMISSION and the shared navtest look require a Step A O1 (READ 1 or READ 2) AND a Step B O1 on BOTH head seeds. Every outcome of Step B is reported whatever Step A reads, so no result is a forking path.
- **What it is:** L1's Step B (`tts_cem.cem_search`, landed, frozen, sha-pinned) run against P2's scorer with ONLY the EP row of `score_head` replaced by the L2 head file. An INFERENCE-time search on CPU: no optimiser, no loss, no gradient.
- **[Rev1 #6] Search start and baseline = P2's own pick** (`bundle['pick']`, the banked argmax of P2's aggregate), not the L2-scorer's in-fan argmax. The L2 head was validated (A1) on perturbations OF P2's pick, so searching around that pick keeps the search inside the domain the head was admitted on. The endpoint is therefore the effect of the SEARCH on top of P2's pick.
  The in-fan re-ranking by the L2 head is read separately and descriptively from banked labels (section 3, "decomposition"); a search output replaces the pick only if the L2 scorer scores it above P2's pick inside the original 64-fan (`tts_cem.cem_search`, final re-score).

## 1. Arms and data (fixed now)
| arm | scorer | role |
|---|---|---|
| **L2-s0**, **L2-s1** | P2 + `final_head_s0.npz` / `final_head_s1.npz` (the sealed recipe fitted on all 400 Step A tokens; weights + sha256 checked at load, refused on mismatch; the file's five non-EP rows must equal P2's bit for bit) | gating |
| **P2HEAD** (control; KEPT, [Rev1 #4]) | plain P2, same search, same seeds, same tokens | DESCRIPTIVE attribution only: it separates "the search found something" from "the L2 head is why" (paired difference reported). Never gating. |

- **Data (the DATA CLAUSE) [Rev1 D0]:** carve B's bundle (same schema as L1's `bundle.npz`: token, log, traj_raw, pick, banked sub / pdms), built by the L2-FRESH chain (`D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint/bundle/`).
  **Its token list must equal the sealed carve** (400 plan tokens, seed 20261019, token-list sha256 `0629811ff5d05879eb62abc5726005ccc4924f03539423003503a099ba29577c` = sha256 of the tokens joined by newlines in bundle order = the bundle's `ctx.json` `tokens_sha256`); **a mismatch REFUSES**, in four places: the chain (before any child process),
  the gate (`g_reach_stepb.run_gate`, no stamp is minted), the launch (`stepB.run_search_cli`, the line tagged `@DATACLAUSE`) and the `ctx.json` / `ctx.npy` sha256 check. No CLI flag overrides the value (only a Python caller, i.e. a test, can pass a planted carve's sha256).
  **RED arms, MEASURED before this seal on planted bundles:** one token replaced, one dropped, two swapped (the same SET in another order), one appended and an empty list are each refused (`test_data_clause_red_arms_*`, `test_refuses_a_token_mismatch_*`); Step A's REAL bundle (sha `211e2640…`, 400 tokens, a legitimate bundle that is not the carve) is refused by the gate's default;
  deleting the launch's `@DATACLAUSE` line lets the planted bundle past the data and the launch fails LATER (`test_data_clause_mutation_arm_*`), and replacing the chain's call by a no-op lets the swapped-token bundle run the whole chain (`test_the_data_clause_red_arm_*`): each refusal IS its line. GREEN control: the real carve's token list (a list of tokens, no label) hashes to the pinned value.
  The carve (L2-FRESH, sealed; MEASURED by that package, `raw/fresh_sample.json`): the literal carve (the held-out logs minus the 24) is EMPTY (0 tokens, 0 logs); B = 400 plan tokens outside Step A's 400, inside the SAME 24 logs (1,305 candidates; G0 A1.2 PASS). The L2 head was fitted on Step A's 400 tokens, so on B it is **token-fresh and LOG-SHARED**:
  it has seen neighbouring windows of every log it is searched on, and this is NOT a fresh-log test. Every Step B line is stamped SEEN-SCENE (L1 A1.3) and says so.
- **Search seeds:** 20260927 / 20260928 / 20260929 (`tts_cem.SEARCH_SEEDS`). K = 5, M = 64 per iteration scored as ONE complete set, elites M/8 = 8, warm start at the pick, final pick re-scored inside the original fan (L1 section 4 and 5, unchanged).
- **Scorer path:** `real_scorer.make_score_fn` (Step A's own step 6) with `load_scorer` wrapped to install the head. CPU fp32 (the same numerics Step A's scoring phase used).

## 2. C8: the sampling widths and clips, by a rule stated before any outcome
L1 Amendment 1 C8 left them as placeholders "to be fixed in a Step B amendment before it runs". **The rule** (code: `stepB.c8_from_grid`, which reads ONLY the perturbation grid Step A measured, never an outcome):
- **clips = the closed hull of the measured grid:** speed scale in [0.85, 1.15], lateral offset in [-0.3, +0.3] m. The search cannot leave the region the admission test covered (`tts_cem.py`'s own docstring intent). The placeholder clips (x0.70-1.30, +-0.6 m) left it by about 2x, where Step A measured nothing.
- **sigma0 = (hull half-width) / 2 per axis:** the grid's extremes sit at 2 sigma0 of the warm start (the pick, the grid's centre). half-width = max(|lo|, |hi|).

| | placeholder (`tts_cem.py`) | C8 value |
|---|---|---|
| sigma0 on u = ln(speed scale) | 0.10 | **0.08126** (= ln(1/0.85) / 2) |
| sigma0 on lateral offset d (m) | 0.20 | **0.15** |
| u clip | [ln 0.70, ln 1.30] = [-0.357, +0.262] | **[ln 0.85, ln 1.15] = [-0.16252, +0.13976]** |
| d clip (m) | +-0.60 | **+-0.30** |

Frozen as `stepB_config.json`, sha256 `299c6e63dd3016100149f88b6ca498529d79f8a5fbcfc6845753c8121d62c441` (LF). The launcher refuses any config that is not this rule's output (a hand-edited clip or width is refused, not warned).
- **O4's "raise M to 256 and widen the sampling covariance once"** is `c8_from_grid(m_pop=256, n_elite=32, widen=2.0)`: sigma0 x2, M = 256, elites 32. **The clips stay at the admitted hull**, so a widened search puts probability mass ON the clip (atoms at the grid's extremes) instead of leaving the admitted region. Its config file is written NOW, before any outcome, and pinned: `stepB_config_O4.json`, sha256 `1ebd719617e8978a03cea0254edd6f531504c1d070792d066736130ceb929013` (LF); the arm needs its own G-REACH-STEPB stamp per head.

## 3. Endpoint, bar, decision (L1 section 4, made operational)
- **Endpoint:** for each token n and search seed s, d(n, s) = true PDMS(search output) - true PDMS(P2's pick), the pick's label being the BANKED one. The token's value is the mean over the 3 seeds. Interval: **paired log-cluster bootstrap** (`tts_cem.delta_ci`: ratio of sums, WHOLE LOGS resampled, B = 10,000, seed 20260927, the same draws for every quantity).
  It answers "another draw of LOGS" and nothing else: not another head seed (`H-ESTIM-SEED-1`; the L2 sealing note E8 records the two head seeds as MEASURED near-replicates, A1 differing by 0.0006), not another search seed (the third variance; its spread is reported beside the interval).
- **Units [Rev1 #1, RATIFIED as an interpretation recorded before any data; L1 did not state them]:** PDMS **points** = 100 x the [0, 1] difference. L1's "+0.30" is on the published 0-100 scale (its own motivating numbers, TOAD's "+1.1 ... +7.4", are points), and +0.30 on [0, 1] is unreachable. In points the safety tolerance -0.25 equals ONE token in 400 losing a binary sub-score: a coherent reading, and the strict one. dNC, dDAC, dTTC, dC are in the same points.
- **Label path identity (enforced by `label_stepB.py`, exit 3 otherwise):** the composite (u, d) = (0, 0) of the pick, labelled by the same path as every search output, must reproduce the pick's banked label to <= 1e-6, and a plan shifted 0.5 m must change it for at least one token. MEASURED on 3 tokens (smoke, mechanics only): max |difference| exactly **0.0**, control changed the label on 1 of 3.
  Without it the endpoint could carry an offset between the pick's table-path label and a search output's executed-plan-path label.

| outcome | rule (fixed now), applied in this order |
|---|---|
| **O4 INERT** | the search output differs from the pick by > 0.5 m ADE on < 5 % of the pooled (token, seed) outputs (`tts_cem.inert_verdict`). Reported INERT, not refuted (precedent P4 F1, 1 of 120). Endpoint is then descriptive. **Route [Rev1 #5]:** the raised-M widened arm ONCE (M = 256, 32 elites, widen 2.0, clips at the admitted hull; `stepB_config_O4.json`, section 2). If that is ALSO INERT the search is CLOSED for this scorer, and the report must NAME the next lever with what unblocks it: L2-BRANCH (unblocked by its own sealed amendment, a `launch_gate.py` PASS token and the dev-box GPU lock) or the P6 trunk (unblocked by the PI's authorisation of the trunk compute). Closing is never the end of the turn. |
| **O1 PASS** | dPDMS point >= +0.30 AND lower CI > 0 AND dPDMS point > the search-seed range (max - min of the 3 per-seed points) AND dNC, dDAC, dTTC, dC points each >= -0.25. Only an O1 on BOTH head seeds, TOGETHER WITH a Step A O1 (READ 1 or READ 2) [Rev1 D0], licenses EP-ADMISSION (section 6) and then the shared navtest look. |
| **TRADE** | the three dPDMS terms clear and at least one of dNC / dDAC / dTTC / dC point < -0.25. Reported as a TRADE with the numbers (L1 section 4). Next arm, committed now: the same search restricted to the DAC/NC-safe subset of candidates by the scorer's own DAC and NC heads (no new head). Spec: a candidate enters the elite set, and the final pick, only if its sigma(NC) and sigma(DAC) are >= the pick's inside the same set. **Built ON a TRADE, not now [Rev1 #10]** (it needs a modified elite selection plus its own identity and reach checks). |
| **NOT CLEARED** | anything else (the dPDMS bar is not met). **L2-BRANCH [Rev1 #5, RATIFIED]:** the EP-led search lever is closed for this scorer and the next arm is L2-BRANCH (last scorer-branch block trainable), the arm the L2 prereg committed for L2-PARTIAL. L1 names no next arm for this row. |

- **Two head seeds [Rev1 #3, RATIFIED]:** the combined outcome is the WORST of the two (O1 < TRADE < NOT CLEARED < O4), so one non-pass head is visible and not averaged away (as L2-ADMIT is "on BOTH seeds").
- **Seed-spread term [Rev1 #2, RATIFIED]:** operationalises L1 section 5, "an effect below that spread is not an effect", as dPDMS point > range of the 3 per-seed points.
- **Decomposition (descriptive, from banked labels, no new labelling):** share of tokens where the L2 scorer's in-fan argmax differs from P2's pick; dPDMS(L2 argmax - P2 pick); dPDMS(search output - L2 argmax).

## 4. The four families (never pooled; ADE is not a result)
Each with the paired log-cluster interval above, read on the same tokens. Instrument: `analyse_stepB.py`.
- **LONGITUDINAL:** dEP and dTTC (true sub-scores, points); share of outputs that are speed-ups / slow-downs and dPDMS given each (L1's post-hoc finding was that EP ordering is a coin flip on speed-ups, so this is where a gain must appear).
- **LATERAL:** dDAC and dDDC (points); mean |offset| of moved outputs; heading deviation at the last non-repaired pose and the largest yaw-rate deviation against the pick (geometry, descriptive).
- **TACTICAL:** share of outputs whose command class (`stepA.command_class`, the nav_command bearing rule on the end point; HYPOTHESIS: a proxy) differs from the pick's.
- **STRATEGIC: PAUSED** (PI ruling 2026-09-19, ITEM 25). Printed as PAUSED, not as a pass.

## 5. Controls and the gate. E7: which gate applies
- **G-REACH-STEPB** (`g_reach_stepb.py`, PASS stamp bound to head sha256, checkpoint sha256, config sha256, bundle sha256, the code's sha256 and L1's four pinned files; `stepB.py search` REFUSES without a matching stamp and re-runs check (d) itself at start).
  On the actual caller chain (argv, `build_scorer`, head file, install, `make_score_fn`, `run_search`, `cem_search`, `score_fn`): (a) C-ID, a head equal to P2's row gives the search output bit for bit P2's, on a token where the search MOVES the pick; (b) a planted head with the EP row negated gives EP logits the search sees exactly negated on every set both arms were handed identically;
  (c) MUTATE: deleting the install line makes the read-back raise, deleting install AND read-back turns (b) alone RED, the caller wiring (`@WIRE`) turns (b) and (d) RED, the clip lines and the sigma kwarg turn (e) RED, the launch's own scorer build turns the in-launch preflight / (f) RED; (d) the L2 head's EP logits differ from P2's on a real set and the five other columns are EXACTLY 0.0 (and exactly 0.0 under (a));
  (e) the first population equals the one recomputed from (seed, sigma0, clips) bit for bit and, under planted scorers that reward each extreme, the observed (u, d) extremes EQUAL the configured clips and never exceed them; (f) the launch itself, run end to end on one real token, writes a shard whose recorded logits show the head.
  MEASURED on a TEST head (the sealed recipe fitted by me, not the L2 head): PASS on the real P2 scorer, all 8 mutation arms RED, about 1 min (`raw/gate_testhead/`). It must be re-run for each real head file (about 1 min): **the chain re-runs it on EACH real head (s0, s1) before that head's arm, and a stamp minted on the test head does not carry over [Rev1 #8].** Re-run after the Revision 1 edits on the TEST head and Step A's bundle (explicit planted-carve override; the default refuses that bundle): PASS, 8 of 8 mutation arms RED (`raw/gate_testhead_rev1/`).
- **Planted identity at the labeller** (section 3) and L1's pins (`verify_l1_code`) are enforced in the launch path.
- **E7, stated (NARROWED, with the reason recorded) [Rev1 #8, RATIFIED]:** Step B is an inference-time search, so **`launch_gate.py` does not apply as written and G-REACH-STEPB is the right gate**, for these reasons. (i) The PI rule (2026-09-26) and `launch_gate.py`'s own header scope it to "starting a training session": its token, supervisor refusal (`verify` / `exec`) and profiles key on a trainer's argv and `tanitad.train` objects;
  REFe's trainer has its own `refe/launch_gate.py`. (ii) Three of its seven checks have no object in Step B (G-LIVE: declared loss terms and trainable-group gradients; G-CKPT: trainer save/resume; G-CLOCK: the label clock of training caches) and a fourth has none in its trainer-vs-loader form (G-EVAL). A profile with vacuous members would certify less than a narrow gate on the real risk.
  (iii) The real risk of Step B is exactly reachability of the head and the C8 values, which no `launch_gate.py` check covers and G-REACH-STEPB does, with mutation arms. **Nothing from the PI rule is dropped, it is mapped:** G-HYG to the strict config (every field required, unknown or missing keys refused, hand edits refused) and the head loader (shape, dtype, finite); G-DVB to "declared == built, read back"
  (head sha256 to installed EP row, config to the (u, d) the search requests, clips to the observed extremes; head flags on the control arm refused); G-LIVE to "non-zero effect plus an exactly-zero control" ((d), (b), (a)); G-EVAL to one scorer builder for gate and launch plus C-ID; G-CKPT to the sha256 pins on head, checkpoint and config; G-SUITE to `pytest test_stepB.py test_stepB_chain.py` green (counts in section 11), bound by code sha.
  **Where `launch_gate.py` DOES apply:** any arm that TRAINS with an L2 head (L2-BRANCH and beyond, a REFe retrain). The MM's phrase "Step B and beyond" now reads **"any arm that TRAINS with an L2 head (L2-BRANCH and beyond)"**; BACKLOG `L2-STEPB-GATE` is resolved by G-REACH-STEPB. The reason, recorded: `launch_gate.py` governs TRAINING launches (the PI rule of 2026-09-26, "if we are starting a training session"); Step B is an inference-time search with no optimiser, loss or gradient, and a profile whose G-LIVE, G-CKPT and G-CLOCK have no object would certify less than a gate built on the real risk. The right gate is G-REACH-STEPB with its sha-bound PASS stamp (head, checkpoint, config, bundle, code) re-asserted inside the launch.

## 6. EP-ADMISSION and the navtest look (frozen, unchanged)
`PREREG_EP_ADMISSION.md` (tip `d2f4a06`, sha256 `7147e3dfc2984a504b8c42f96e0f2b4ed980bcc668cf879a781bf0f3c02f8367`), `ep_admission.py --rules r1`, runs BEFORE any navtest look: A1 (EP reliability at the candidate's own picks: FAIL iff on ANY seed the D_M lower CI > 0 or the point > +0.05), A2 (safety rows not over-confident:
FAIL iff for NC, DAC or TTC on ANY seed the lower CI > 1.05), A3 realised gain (reported, never gated). ADMIT iff A1 and A2 admit. Order: Step B O1 on both head seeds, then EP-ADMISSION, then the shared navtest look. A candidate that is not admitted gets no look.
- **EP-ADMISSION's object: BOTH banks, and ADMIT needs both to admit [Rev1 #9, RULED].** The thresholds, estimator and rules stay frozen; only the candidate block (appended per read) names the objects. (i) The frozen literal object: an in-fan bank of "P2 + the L2 EP row" (head seeds 0 / 1 = the candidate's two seeds), via a head-file input to `build_candidate_bank.py` (`l2_scorer.install_ep_head`). (ii) A search-output bank in the same schema, built from Step B's labelled outputs: that is what the deployed system picks. **Both are built ONLY after a Step B O1 (not now); the candidate block is written before either bank is read.** Order: Step A O1 (READ 1 or READ 2) AND a Step B O1 on both head seeds, then EP-ADMISSION on both banks, then the shared navtest look.

## 7. Tier and stamps (every Step B line)
SELECTION tier, label-only (not T1, not navtest); SEEN-SCENE (L1 A1.3, and section 1 above); the head is **TOKEN-FRESH and LOG-SHARED on carve B** (400 tokens disjoint from the 400 it was fitted on, inside the SAME 24 logs): NOT a fresh-log test [Rev1 D0]; ONE training seed per head arm (`H-ESTIM-SEED-1` open); the interval is the paired log-cluster bootstrap over the carve's logs (24 logs if the carve stays inside them), "another draw of LOGS" only; the search-seed spread beside it. No capability claim.

## 8. Cost (MEASURED on the first 3 tokens of Step A's bundle, CPU 4 threads on a shared box; projections ESTIMATED; `raw/cost.json`)
- Search: **1.31 s per token and seed (median 1.29, max 1.64), 4.02 s per token for 3 seeds**, so 400 tokens x 3 seeds is about **27 min per arm**.
- Labelling: Step A's labeller took 591 s for 400 tokens at 96 plans each (**1.48 s per token, MEASURED**, `l1run/steps4to7.log`), an upper bound for the 5 plans per token here (identity, 3 seeds, control), so about 10 min per arm; the cold 3-token smoke read 4.2 s per token (about 28 min). Range **10 to 28 min per arm**.
- **Per arm 37 to 55 CPU-min; three arms (L2-s0, L2-s1, P2HEAD) 1.8 to 2.7 h sequential**, half that with two processes. No GPU, no lock.

## 9. The ten decisions (Revision 1: all RULED; the ruling's text prevails)
| # | decision | ruling | applied in |
|---|---|---|---|
| 1 | units of the L1 bar | PDMS POINTS (100 x [0, 1]); RATIFIED as an interpretation recorded before any data | section 3 |
| 2 | seed-spread term | dPDMS point > (max - min) of the 3 per-search-seed points; RATIFIED | section 3 |
| 3 | two head seeds | the WORST of the two; RATIFIED | section 3 |
| 4 | P2HEAD control | KEPT (descriptive attribution; it costs one arm) | section 1 |
| 5 | NOT CLEARED / O4 | NOT CLEARED => L2-BRANCH; O4 INERT => the raised-M widened arm once; if also INERT the search is CLOSED for this scorer and the report NAMES the next lever (L2-BRANCH or the P6 trunk) with what unblocks it | section 3 table, `analyse_stepB.NEXT_ARM` |
| 6 | start and baseline | P2's pick; RATIFIED. The in-fan re-ranking stays descriptive | section 0 |
| 7 | clips under O4's widening | stay at the hull; RATIFIED | section 2 |
| 8 | E7 | NARROWED, with the reason recorded: G-REACH-STEPB, not `launch_gate.py`, governs an inference-time search | section 5 |
| 9 | EP-ADMISSION's object | BOTH banks; ADMIT needs both; built ONLY after a Step B O1 | section 6 |
| 10 | TRADE-restricted arm | built ON a TRADE, not now | section 3 table |

## 10. Not licensed
No navtest look, no capability claim, no registry row. **Step B licenses nothing alone [Rev1 D0]:** a Step B O1 on BOTH head seeds, together with a Step A O1 (READ 1 or READ 2), licenses EP-ADMISSION on both banks and then the shared navtest look, nothing more. Every Step B outcome is reported whatever Step A reads; this chain does not read Step A. Inference stays vision-only: the head reads the same token at inference; the privileged PDM labels are the endpoint's measuring instrument, never an input to the search.


## 11. The chain, the pins and the seal [Rev1: the chain]
**The chain** (`code/stepB_chain.py`, ONE detached CPU process: `CUDA_VISIBLE_DEVICES=-1`, threads capped at 4 (label step 1), no lock, no GPU, no pod; below-normal priority; markers ZZSTEPB_* at the START of their lines, children's output in `run/logs/`; resumable: valid stamps, shards, merged files and label files are skipped):
(a) WAIT until the sealed text and its `.sha256` hold (the seal guard), L2-FRESH's chain printed `CHAIN_DONE_NO_READ`, L2-FRESH's own `PREREG_L2_FRESH.sha256` exists and names the same carve and heads, and L2-FRESH's gates hold (`ZZBUNDLE_OK 400`, labels G1 identity and its control, G2a / G2c / G3, cross-device pick agreement >= 0.99, C-FEAT); a FAIL marker of that chain, a dead chain with no marker, or a timeout STOPS this chain.
(b) the DATA CLAUSE (section 1) and G-REACH-STEPB on EACH real head (s0, s1): both PASS stamps exist BEFORE any search, else `ZZSTEPB_STOP`.
(c) arms L2-s0, L2-s1, P2HEAD x 3 search seeds. (d) the labels (`label_stepB.py`: the planted identity at the labeller, exit 3 = STOP). (e) the registered analysis (`analyse_stepB.py`): the four families (STRATEGIC PAUSED), O1 / TRADE / NOT CLEARED / O4, worst of the two head seeds, the stamps, the NAMED next arm. (f) `ZZSTEPB_DONE <combined outcome>`.
**Seal guard** (`stepB_chain.Chain.check_seal`, run at the start AND again after the wait; REFUSES, exit 5): the text's sha256 differs from `PREREG_L2_STEPB.sha256`; the marker `Status: SEALED` is absent; a pin below is missing or differs from the file or constant the chain will use (code files by LF sha256, L1's four files, both configs, both head files and `final_heads.json`, the carve pin vs `stepB.SEALED_CARVE_SHA256`);
an operative code file is not pinned. Editing any pinned code file after this seal makes the chain refuse; a change goes into a sealed amendment.
**Outputs** (CPU, small, on C:): `run/` beside this text: stamps, shards, merged search files, labels, `RESULT_STEPB.json`, `chain_state.json`, logs.
**MEASURED before this seal (planted inputs only; nothing of carve B or Step B was read):** `pytest code/test_stepB.py code/test_stepB_chain.py`: **98 passed** (`98 passed in 39.98s`; `logs/pytest_test_stepB_sealed.log`); the chain on the REAL inputs without a seal read `ZZSTEPB_STOP SEAL_MISSING`, exit 5.

**Pins (machine-readable: the chain parses this block; `make_stage.py` is excluded because it only copies files for landing and never runs in the chain):**
<!-- PINS-BEGIN -->
```
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
```
<!-- PINS-END -->

*Code files pinned: analyse_stepB.py, cost_stepB.py, fake_scorer.py, g_reach_stepb.py, l2_scorer.py, label_stepB.py, make_test_head.py, stepB.py, stepB_chain.py, test_stepB.py, test_stepB_chain.py.*
