# PREREG L2-STEPB (DRAFT, NOT ADOPTED): L1 Step B, the CEM search, on P2's scorer with the L2 EP head

Written 2026-10-19 by the L2-STEPB-BUILD agent. **No Step B search has been run and no label of a search output has been read; nothing in this file is an outcome.** It is an amendment-in-waiting to
`PREREG_L1_SEARCH_ADMISSION.md` (sha256 `a76f62d8043e4cc9d6c73b35cf2884f27027a1e01bcd3deec658380cc58bd801`, section 4 O1 and section 5), L1 Amendment 1 (sha256 `45f5c7917feef407b3ce18951e35f7abdcef5b7a9fb65e41e9cebd3d50fd72ff`, C8)
and `PREREG_L2_OFFPROP.md` (sealed, sha256 `3592a1039693dcb5cac1df465fa4dd597ee190d80a934d89fc765d223607a983`, outcome L2-ADMIT). None of the three is edited. It is for the Master Mind to adopt, amend or strike, and its sha256 is to be written
beside it BEFORE the first search. Evidence classes: MEASURED (ours, artifact named), INHERITED (another package, not re-verified), ESTIMATED, HYPOTHESIS.

## 0. When this launches, and what it is
- **Precondition (not decided here):** L2-OFFPROP reads L2-ADMIT on both head seeds AND L1 Step A, re-run on the fresh carve with the L2 head, reads O1 (L2 prereg section 4, row L2-ADMIT). Otherwise this arm never starts; the committed next arm is then L2-BRANCH or the trunk (P6), per that table.
- **What it is:** L1's Step B (`tts_cem.cem_search`, landed, frozen, sha-pinned) run against P2's scorer with ONLY the EP row of `score_head` replaced by the L2 head file. An INFERENCE-time search on CPU: no optimiser, no loss, no gradient.
- **Search start and baseline = P2's own pick** (`bundle['pick']`, the banked argmax of P2's aggregate), not the L2-scorer's in-fan argmax. The endpoint is therefore the effect of the SEARCH on top of P2's pick. The in-fan re-ranking by the L2 head is read separately and descriptively
  from banked labels (section 3, "decomposition"); a search output replaces the pick only if the L2 scorer scores it above P2's pick inside the original 64-fan (`tts_cem.cem_search`, final re-score).

## 1. Arms and data (fixed now)
| arm | scorer | role |
|---|---|---|
| **L2-s0**, **L2-s1** | P2 + `final_head_s0.npz` / `final_head_s1.npz` (the sealed recipe fitted on all 400 Step A tokens; weights + sha256 checked at load, refused on mismatch; the file's five non-EP rows must equal P2's bit for bit) | gating |
| **P2HEAD** (control) | plain P2, same search, same seeds, same tokens | DESCRIPTIVE attribution only: it separates "the search found something" from "the L2 head is why" (paired difference reported). Never gating. |

- **Data:** the bundle of the fresh-carve Step A re-run (same schema as L1's `bundle.npz`: token, log, traj_raw, pick, banked sub / pdms), its sha256 pinned at adoption. Its definition is the L2-FRESH-STEPA package's, not this file's.
  INHERITED (`l2fresh/raw/fresh_universe_probe.json`, read-only, only its first fields seen, not re-verified): the literal carve (held-out logs minus the 24) is EMPTY (0 tokens, 0 logs); the available carve is token-disjoint from the 400 Step A tokens inside the SAME 24 logs
  (1,305 tokens), so the L2 head has seen neighbouring windows of every log it is searched on. Every Step B line is stamped SEEN-SCENE (L1 A1.3) and says so.
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
- **O4's "raise M to 256 and widen the sampling covariance once"** is `c8_from_grid(m_pop=256, n_elite=32, widen=2.0)`: sigma0 x2, M = 256, elites 32. **The clips stay at the admitted hull**, so a widened search puts probability mass ON the clip (atoms at the grid's extremes) instead of leaving the admitted region. It needs its own config file and sha256, written before that arm runs.

## 3. Endpoint, bar, decision (L1 section 4, made operational)
- **Endpoint:** for each token n and search seed s, d(n, s) = true PDMS(search output) - true PDMS(P2's pick), the pick's label being the BANKED one. The token's value is the mean over the 3 seeds. Interval: **paired log-cluster bootstrap** (`tts_cem.delta_ci`: ratio of sums, WHOLE LOGS resampled, B = 10,000, seed 20260927, the same draws for every quantity).
  It answers "another draw of LOGS" and nothing else: not another head seed (`H-ESTIM-SEED-1`; the L2 sealing note E8 records the two head seeds as MEASURED near-replicates, A1 differing by 0.0006), not another search seed (the third variance; its spread is reported beside the interval).
- **Units (an interpretation L1 did not state):** PDMS **points** = 100 x the [0, 1] difference. L1's "+0.30" is on the published 0-100 scale (its own motivating numbers, TOAD's "+1.1 ... +7.4", are points), and +0.30 on [0, 1] is unreachable. dNC, dDAC, dTTC, dC are in the same points.
- **Label path identity (enforced by `label_stepB.py`, exit 3 otherwise):** the composite (u, d) = (0, 0) of the pick, labelled by the same path as every search output, must reproduce the pick's banked label to <= 1e-6, and a plan shifted 0.5 m must change it for at least one token. MEASURED on 3 tokens (smoke, mechanics only): max |difference| exactly **0.0**, control changed the label on 1 of 3.
  Without it the endpoint could carry an offset between the pick's table-path label and a search output's executed-plan-path label.

| outcome | rule (fixed now), applied in this order |
|---|---|
| **O4 INERT** | the search output differs from the pick by > 0.5 m ADE on < 5 % of the pooled (token, seed) outputs (`tts_cem.inert_verdict`). Reported INERT, not refuted (precedent P4 F1, 1 of 120). Endpoint is then descriptive. |
| **O1 PASS** | dPDMS point >= +0.30 AND lower CI > 0 AND dPDMS point > the search-seed range (max - min of the 3 per-seed points) AND dNC, dDAC, dTTC, dC points each >= -0.25. Only an O1 on BOTH head seeds licenses the single navtest read (after section 6). |
| **TRADE** | the three dPDMS terms clear and at least one of dNC / dDAC / dTTC / dC point < -0.25. Reported as a TRADE with the numbers (L1 section 4). Next arm, committed now: the same search restricted to the DAC/NC-safe subset of candidates by the scorer's own DAC and NC heads (no new head). Spec: a candidate enters the elite set, and the final pick, only if its sigma(NC) and sigma(DAC) are >= the pick's inside the same set. **Not built** (it needs a modified elite selection plus its own identity and reach checks; built only if a TRADE occurs). |
| **NOT CLEARED** | anything else (the dPDMS bar is not met). L1 names no next arm for this row; proposed here: the EP-led search lever is closed for this scorer and the next arm is L2-BRANCH (last scorer-branch block trainable), already committed by the L2 prereg for L2-PARTIAL. |

- **Two head seeds:** the combined outcome is the WORST of the two (O1 < TRADE < NOT CLEARED < O4), so one non-pass head is visible and not averaged away (as L2-ADMIT is "on BOTH seeds").
- **Seed-spread term:** operationalises L1 section 5, "an effect below that spread is not an effect", as dPDMS point > range of the 3 per-seed points.
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
  MEASURED on a TEST head (the sealed recipe fitted by me, not the L2 head): PASS on the real P2 scorer, all 8 mutation arms RED, about 1 min (`raw/gate_testhead/`). It must be re-run for each real head file (about 1 min).
- **Planted identity at the labeller** (section 3) and L1's pins (`verify_l1_code`) are enforced in the launch path.
- **E7, stated:** Step B is an inference-time search, so **`launch_gate.py` does not apply as written and G-REACH-STEPB is the right gate**, for these reasons. (i) The PI rule (2026-09-26) and `launch_gate.py`'s own header scope it to "starting a training session": its token, supervisor refusal (`verify` / `exec`) and profiles key on a trainer's argv and `tanitad.train` objects;
  REFe's trainer has its own `refe/launch_gate.py`. (ii) Three of its seven checks have no object in Step B (G-LIVE: declared loss terms and trainable-group gradients; G-CKPT: trainer save/resume; G-CLOCK: the label clock of training caches) and a fourth has none in its trainer-vs-loader form (G-EVAL). A profile with vacuous members would certify less than a narrow gate on the real risk.
  (iii) The real risk of Step B is exactly reachability of the head and the C8 values, which no `launch_gate.py` check covers and G-REACH-STEPB does, with mutation arms. **Nothing from the PI rule is dropped, it is mapped:** G-HYG to the strict config (every field required, unknown or missing keys refused, hand edits refused) and the head loader (shape, dtype, finite); G-DVB to "declared == built, read back"
  (head sha256 to installed EP row, config to the (u, d) the search requests, clips to the observed extremes; head flags on the control arm refused); G-LIVE to "non-zero effect plus an exactly-zero control" ((d), (b), (a)); G-EVAL to one scorer builder for gate and launch plus C-ID; G-CKPT to the sha256 pins on head, checkpoint and config; G-SUITE to `pytest test_stepB.py` (36 tests) green, bound by code sha.
  **Where `launch_gate.py` DOES apply:** any later arm that TRAINS with the L2 head (L2-BRANCH, a REFe retrain). **This narrows the literal wording of MM ruling E7's addendum ("Step B and beyond ... a launch blocker for Step B"); the Master Mind must confirm it in one line, or ask for a thin `stepb` profile in `launch_gate.py` that hosts this stamp as its one check (I did not edit `launch_gate.py`).**

## 6. EP-ADMISSION and the navtest look (frozen, unchanged)
`PREREG_EP_ADMISSION.md` (tip `d2f4a06`, sha256 `7147e3dfc2984a504b8c42f96e0f2b4ed980bcc668cf879a781bf0f3c02f8367`), `ep_admission.py --rules r1`, runs BEFORE any navtest look: A1 (EP reliability at the candidate's own picks: FAIL iff on ANY seed the D_M lower CI > 0 or the point > +0.05), A2 (safety rows not over-confident:
FAIL iff for NC, DAC or TTC on ANY seed the lower CI > 1.05), A3 realised gain (reported, never gated). ADMIT iff A1 and A2 admit. Order: Step B O1 on both head seeds, then EP-ADMISSION, then the shared navtest look. A candidate that is not admitted gets no look.
- **Open (for the Master Mind):** the frozen text defines the candidate by a bank built from C's own forward and C's picks (`build_candidate_bank.py`). For L2 the natural candidate is "P2 with the L2 EP row as an in-fan picker" (head seeds 0 and 1 as the candidate's two seeds). Step B's search outputs are NOT rows of any bank, so the gate as frozen does not read them. Whether a bank of search outputs is also required is a ruling, not something to infer.
  The bank builder takes a checkpoint; a head-file input is not built (the piece it needs is `l2_scorer.install_ep_head`).

## 7. Tier and stamps (every Step B line)
SELECTION tier, label-only (not T1, not navtest); SEEN-SCENE (L1 A1.3, and section 1 above); ONE training seed per head arm (`H-ESTIM-SEED-1` open); the interval is the paired log-cluster bootstrap over the carve's logs (24 logs if the carve stays inside them), "another draw of LOGS" only; the search-seed spread beside it. No capability claim.

## 8. Cost (MEASURED on the first 3 tokens of Step A's bundle, CPU 4 threads on a shared box; projections ESTIMATED; `raw/cost.json`)
- Search: **1.31 s per token and seed (median 1.29, max 1.64), 4.02 s per token for 3 seeds**, so 400 tokens x 3 seeds is about **27 min per arm**.
- Labelling: Step A's labeller took 591 s for 400 tokens at 96 plans each (**1.48 s per token, MEASURED**, `l1run/steps4to7.log`), an upper bound for the 5 plans per token here (identity, 3 seeds, control), so about 10 min per arm; the cold 3-token smoke read 4.2 s per token (about 28 min). Range **10 to 28 min per arm**.
- **Per arm 37 to 55 CPU-min; three arms (L2-s0, L2-s1, P2HEAD) 1.8 to 2.7 h sequential**, half that with two processes. No GPU, no lock.

## 9. Decisions this draft needs from the Master Mind
1. Units of the L1 bar (points = 100 x [0, 1]). 2. The seed-spread operationalisation. 3. Worst-of-two for the two head seeds. 4. The P2HEAD control arm (strike it if not wanted; it costs one more arm). 5. NOT CLEARED's next arm (L2-BRANCH proposed) and O4's route (L1 says "go to L2", which is already consumed: proposed, after the raised-M arm is also INERT, close the search for this scorer).
6. Baseline and start = P2's pick (not the L2 argmax). 7. Clips stay at the hull under O4's widening. 8. E7 (section 5). 9. EP-ADMISSION's object (section 6). 10. Whether the TRADE-restricted arm should be built now (about 1 h with its identity and reach checks) or only on a TRADE.

## 10. Not licensed
No navtest look, no capability claim, no registry row. A Step B O1 licenses EP-ADMISSION and then the shared navtest look, nothing more. Inference stays vision-only: the head reads the same token at inference; the privileged PDM labels are the endpoint's measuring instrument, never an input to the search.
