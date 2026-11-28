=== V43S ===
v43 adds §7.45 and §7.46. T2 is still re-perception on NuRec scenes with the other agents replayed at log time: never a reactive-traffic result and never pooled with T1-kin (paired scene-cluster bootstrap, B = 2,000, seed 0; two training seeds and no replicate arm). §7.45 attributes the composite's crawl on the 48 read scenes: post-training is PT-ACTIVE (C0C − MC separated at both seeds on ADE6, along-track MAE, acceleration, DK-HW and the M1 crawl share) and adds 0.06 / 0.11 to a crawl that is mostly in the base (MC alone: M1 +0.3493 / +0.2919 against C0C's +0.4073 / +0.4041, NOT-DOMINANT); the pod route is RT-WITHIN-FLOOR, narrowly (at s1 all three rows are separated and below MC's seed floor by 0.0064, 0.045 and 0.038); F′ = 0.12701 m, so the PRIMARY bar for the LoRA arm L is 0.2540 m; L is training on the PI's GPU pod (44.97 s per step over its first 200 steps, about 241 h per seed ESTIMATED) and no LoRA driving result exists; the T1-kin BLIND read is VISION-USED (the BLIND contrast moves ADE6 by +0.8313 / +0.8158 m), so the T2 null for BLIND is a render effect, and the render gap binds at T2 (GAP48 cosine 0.8401 between a render and its real frame; a per-channel affine map reaches 0.8937 and misses its 0.90 bar: ALIGN-FAILS-MECH). §7.46 gives the theory after six reads: T1′ falls (P1′b FAILS), WSP-GAIN reads HARMS in 4 of 4 cells, T5 falls (P5 FAILS, λ 0.968), the data-matched edit WSP-GAIN-D reads HARMS, AUCMAP reads outcome (iii) (no gate on this lever family reaches the corner, even at AUC 1.0), T6 survives its first test (VISION-USED) while its mechanism sentence does not, and the committed next lever RENC (encoder adaptation on the render gap) is pre-registered and not run. No DOMINANT exists, there is no scale signal, and nothing in v43 is a driving result.

=== S81 ===
33. **The T2 attribution, the LoRA launch and the theory after six reads** (§7.45, §7.46). The T2 reads use the paired scene-cluster bootstrap (B = 2,000, seed 0) on the 48 read scenes; the T1-kin reads use the paired episode-cluster bootstrap on eval-141; the theory reads are zero-training diagnostics on banked artifacts and logged windows.
    * **Designed and frozen before any cell or measurement (MEASURED).** `PREREG_T2_ATTRIB` (frozen `f7d2cfa1…`; file `{{RES_AT}}…`, landed `{{LAND_AT}}`), `PREREG_T1_BLIND` (frozen `9619414a…`), `PREREG_T2_RALIGN` (frozen `a6c84433…`), `PREREG_T2_RENC` (frozen prefix of 5,544 bytes, `425d3496…`; file `{{RES_RENC}}…`, landed `{{LAND_RENC}}`), `THEORY_CRAWL_AMEND_3.md` (frozen `dcb82ae9…`; file `{{RES_A3}}…`), `THEORY_CRAWL_AMEND_4.md` (`{{RES_A4}}…`), `PREREG_P4_CUECLASS` (frozen `6717a0cf…`; file `{{RES_P4C}}…`) and the weight-edit cards `PREREG_WSP_GAIN` (`92d61dcb…`) and `PREREG_WSP_GAIN_D` (first 5,211 bytes, `d6e691a3…`).
    * **Result.** <<V43S>> Its limits are in §7.45.8, §7.46.10 and section 9 item 21.

=== R83 ===
**Update (v43).** Three items are written up here, and the adopted base is unchanged (DAG-OACC):
* **The T2 attribution (§7.45, T2, 48 scenes, agents replayed):** post-training is PT-ACTIVE (C0C − MC separated at both seeds) and adds 0.06 / 0.11 to the M1 crawl share, so most of the T2 crawl is already in the base (MC alone is NOT-DOMINANT, M1 +0.3493 / +0.2919 against C0C's +0.4073 / +0.4041); the pod route is RT-WITHIN-FLOOR, narrowly (at s1 three rows separated, below MC's seed floor by 0.0064, 0.045 and 0.038). No DOMINANT exists.
* **The LoRA arm L (§7.45.4, §7.45.5):** F′ = 0.12701 m (B′ against M2b, eval-141, T1-kin), so the PRIMARY bar is \|ADE6(L) − ADE6(B′)\| > 0.2540 m; L launched on the PI's GPU pod with a launch-gate PASS and trains at 44.97 s per step (MEASURED over its first 200 steps; about 241 h per seed, ESTIMATED). No LoRA driving result exists.
* **The render gap binds at T2, and the theory after six reads (§7.45.6, §7.45.7, §7.46):** the T1-kin BLIND read is VISION-USED (ADE6 +0.8313 / +0.8158 m), GAP48 puts renders at cosine 0.8401 from their real frames, and the per-channel map reaches 0.8937 (ALIGN-FAILS-MECH). P1′b FAILS, P5 FAILS (λ 0.968), both weight edits read HARMS (three reads), AUCMAP reads outcome (iii), and T6 survives its first test.
* **What is blocked:** no counted DOMINANT exists, so no evidence video and no scale signal follow. The next lever is RENC (`PREREG_T2_RENC.md`, pre-registered, not run). L's own T2 read (`PREREG_T2_DOM_LORA.md`, frozen `{{RES_DOM}}…`, not run) waits on L's training (about 241 h per seed, ESTIMATED) and, as a vision lever, may not be creditable at T2 until the render gap is closed (§7.46.9, **[H]**).

=== ROW ===
| 1bg | **T2 attribution: post-training term PT and route term RT** (`H-T2-ATTRIB-1`; zero training; T2, 48 scenes, agents replayed) | card `PREREG_T2_ATTRIB.md` `{{RES_AT}}…` (frozen `f7d2cfa1…` before any stage-5 cell): PT-ACTIVE if any of the 19 rows of C0C − MC is separated at both seeds; RT-WITHIN-FLOOR if, on M1 crawl share, ADE6 and along-track MAE, BPC − MC is not separated at either seed or is below that row's seed floor of MC at both seeds | **PT-ACTIVE** (§7.45.2): adds 0.06 / 0.11 to the M1 crawl share; MC alone NOT-DOMINANT. **RT-WITHIN-FLOOR, narrowly** (§7.45.3): s1 rows separated, below the floor by 0.0064, 0.045, 0.038. T2 [M] |
| 1bh | **The LoRA arm L and its T2 dominance read** (`D-V8F-LORA-FPRIME-1`, `D-T2-LC-1`, `D-T2-PIN5-1`; the PRIMARY bar and the LC read) | PRIMARY \|ADE6(L) − ADE6(B′)\| > 2F′ = 0.2540 m, separated, both seeds (`AMENDMENT_V8F_LORA_4.md`); the T2 read of LC = composite(L) by `PREREG_T2_DOM_LORA.md` `{{RES_DOM}}…` (frozen; folder `raw/2026-11-07-t2-dom-lora-freeze/`), which fixes LC as the same two post-training stages on L's cache with the same pipeline on B′ as its control, LC − C0C standing beside that control while RT-WITHIN-FLOOR holds; pin 5 of the read (seed-0-only outcome SEED0-ONLY, never DOMINANT-T2) | **L is training** (§7.45.5): 44.97 s per step over its first 200 steps [M], about 241 h per seed [E]; F′ = 0.12701 m [M, T1-kin]; **no LoRA driving result exists** and no LC cell has run |
| 1bi | **T2 render alignment RALIGN and GAP48** (`D-V8F-T2-RALIGN-1`; tier T2-RA, never pooled) | `PREREG_T2_RALIGN.md` (frozen `a6c84433…`): GAP48 pairs, a scene-fold cross-fit, a per-cell affine map, gate M-ALIGN (held-out cosine rises by 0.05 and reaches 0.90); outcomes RA-DOMINANT, RA-MOVES, RA-NULL or RA-HARMS, ALIGN-FAILS-MECH | **GAP48** (descriptive, §7.45.7): render–real cosine 0.8401. **ALIGN-FAILS-MECH** (M-ALIGN 0.8937, bar 0.90): no T2-RA stage ran |
| 1bj | **RENC, encoder adaptation on the render gap** (`H-T2-RENC-1`; the committed next lever) | `PREREG_T2_RENC.md` `{{RES_RENC}}…` (frozen prefix `425d3496…`): G-ID0, M-RENC (held-out cosine rises above 0.840 by 0.05 and reaches 0.90), G-KEEP (real-frame cosine at least 0.99); outcomes RE-DOMINANT, RE-MOVES, RE-NULL or RE-HARMS, RE-FAILS-MECH (§7.46.9) | **pre-registered, not run** [H]: no training image rendered, no adapted encoder exists |
| 1bk | **Weight edits, the corner and the cue classes** (`D-V8F-WSPGAIN-1`, `D-V8F-WSPGAIN-D-1`, `D-THEORY-AUCMAP-1`, `H-THEORY-P4-1`; zero training) | the cards `PREREG_WSP_GAIN` (`92d61dcb…`), `PREREG_WSP_GAIN_D`, `PREREG_AUCMAP` and `PREREG_P4_CUECLASS` (frozen `6717a0cf…`) | **HARMS** for both edits (§7.46.3, §7.46.6; `share_closed` 0.178 to 0.247 and 0.329 / 0.289); AUCMAP **outcome (iii)** (§7.46.4); P4 **not evaluable** until an L feature exists (§7.46.7) |
| 1bl | **Weight-space read of the LoRA deltas of RENC and L** (`H-WSP-LORA-1`; zero training; asked for by the PI on 2026-11-07) | card `PREREG_WSP_LORA.md` `{{RES_WLR}}…` (frozen; folder `raw/2026-11-07-wsp-lora-prereg/`), written before any RENC encoder exists and before any L checkpoint is read | **pre-registered, not run** [H]: no weight-space read of the RENC or L deltas has been made |

=== I21 ===
21. **v43 reads and their limits.**
    * **What the interval answers.** The T2 reads (MC, BPC, GAP48) use the paired scene-cluster bootstrap (B = 2,000, seed 0), which answers another draw of SCENES only; the T1-kin reads use the paired episode-cluster bootstrap over eval-141. The training-seed question is answered by requiring both seeds; there is no replicate arm on this rig (`H-ESTIM-SEED-1`) and no inference seed is varied. The T2 seed floor is wide: one more training run moves ADE6 by up to 0.46 m, so a T2 lever claim below it is not an effect.
    * **Population and tier.** MC and BPC read the same 48 scenes (rule R-SCN refused the same 14); the agents replay, so no number is a reactive-traffic result; P1′b reads 60 unseen holdout clips as a T0-class plan read; the BLIND read uses 141 eval-141 episodes. T2, T2-RA and T1-kin are never pooled.
    * **The attribution.** RT-WITHIN-FLOOR is the outcome of the rule as written and is narrow: at s1 all three rows are separated and sit below the floor by 0.0064, 0.045 and 0.038. PT-ACTIVE rests on two seeds and no replicate.
    * **The LoRA arm.** The step time is MEASURED over the first 200 steps (44.97 s per step); the 241 h per seed is ESTIMATED and not re-derived. The per-batch training values are not an evaluation. No L checkpoint has been scored at any tier, and the dominance read of LC (`PREREG_T2_DOM_LORA.md`, frozen) has not run.
    * **The render gap.** GAP48 is descriptive on 48 scenes. The mechanism sentence of T6 (renders sit about as far from the mean as from their real frame) does not hold; the behavioural claim rests on a T2 null whose M1 interval is wide (the stage-2 BLIND intervals are quoted from the amendment, not re-read from the stage-2 files). RALIGN's same-fold arm (0.9015) is a leak arm and is not a result.
    * **The theory reads.** P3's chain of five admissible levers is post hoc (n = 5). AUCMAP tests one lever family, not every operator. The P5 ridge row-mean grouping is UNVERIFIED in its source. The claim-row ids were not re-read on the tip's register.
    * No reactive-traffic number, no LoRA driving result and no DOMINANT exist.

=== ABS1 ===
* **The T2 attribution, the LoRA launch and the render gap (§7.45; T2 on the 48 read scenes, agents replayed; paired scene-cluster bootstrap, B = 2,000, seed 0; two training seeds, no replicate; T1-kin for the BLIND read; evidence class [M] unless marked).** Post-training is PT-ACTIVE (C0C − MC separated at both seeds) and adds 0.06 / 0.11 to the M1 crawl share; MC alone is NOT-DOMINANT with M1 +0.3493 / +0.2919 against C0C's +0.4073 / +0.4041, so most of the T2 crawl sits in the base. The pod route is RT-WITHIN-FLOOR, narrowly: at s1 the three rows are separated (BPC worse) and below MC's seed floor by 0.0064, 0.045 and 0.038. F′ = 0.12701 m, so the PRIMARY bar for the LoRA arm L is \|ADE6(L) − ADE6(B′)\| > 0.2540 m. L launched on the PI's GPU pod and trains at 44.97 s per step (about 241 h per seed, **[E]**); **no LoRA driving result exists**. On real frames the planner uses its scene tokens (T1-kin BLIND read, VISION-USED: ADE6 +0.8313 [+0.5576, +1.1530] / +0.8158 [+0.5309, +1.1454] m), so the T2 null for BLIND is a render effect, and the render gap binds at T2: renders sit at cosine 0.8401 from their real frames, and a per-channel affine map reaches 0.8937 and misses its 0.90 bar (ALIGN-FAILS-MECH). No DOMINANT exists and there is no scale signal.
* **The theory after six reads (§7.46; zero training; T1-kin unless marked).** T1′ falls (P1′b FAILS: the difference of median shrinkage ratios is +0.245 [−0.028, +0.407] at s0, not separated). Both weight edits harm, in three reads: WSP-GAIN at two doses (`share_closed` 0.178 to 0.247, 6 to 10 new separated-worse rows) and WSP-GAIN-D (0.329 / 0.289, nine new rows at each seed). AUCMAP reads outcome (iii): no gate on the floor and forced-level levers reaches the corner, even with a perfect go/stay posterior. T5 (regression dilution of the acceleration input) falls: P5 FAILS at λ 0.9680 [0.9659, 0.9698]. T6 (render-domain information loss) survives its first test, P6 (VISION-USED), and loses its mechanism sentence; the committed next lever RENC (encoder adaptation on the render gap) is **pre-registered and not run** **[H]**.

=== NXT ===
v43 reads the T2 attribution (§7.45): post-training adds 0.06 / 0.11 to a crawl that is mostly in the base, the pod route is within the seed floor (narrowly), and the planner uses its scene tokens on real frames (VISION-USED), so the render gap, which a per-channel map does not close (ALIGN-FAILS-MECH), binds at T2. The LoRA arm L is training on the PI's GPU pod and no LoRA driving result exists. The theory (§7.46) refutes T1′ and T5 and reads both weight edits as HARMS; the committed next lever is encoder adaptation on the render gap (RENC, pre-registered, not run), then L's own T2 read.

=== APP ===
**Reads added in v43 (tip `{{TP}}`).** The raw folders under `R/raw/` that §7.45 and §7.46 read, one row each (sha256 prefix of the key file; commit that added it); the claim rows are in `Project Steering/GOALS_AND_CLAIMS.md`.

| raw folder | key file (sha256) | landed | claim row |
|---|---|---|---|
| `2026-11-07-t2-attrib-prereg/` | `PREREG_T2_ATTRIB.md` `{{RES_AT}}…` | `{{LAND_AT}}` | `H-T2-ATTRIB-1` |
| `2026-11-07-t2-mc-read/` | `stage5_MC/pair_C0C_MC/PAIR_T2.md` `{{RES_MCP}}…` | `{{LAND_MCP}}` | `D-T2-ATTRIB-PT-1` |
| `2026-11-07-t2-bpc-read/` | `RT_VERDICT.json` `{{RES_BPC}}…` | `{{LAND_BPC}}` | `D-T2-ATTRIB-RT-1` |
| `2026-11-07-lora-fprime/` | `out/FPRIME.json` `{{RES_FPR}}…` | `{{LAND_FPR}}` | `D-V8F-LORA-FPRIME-1` |
| `2026-11-07-t2-pair-fprime-prep/` | `MANIFEST.txt` `{{RES_PFP}}…` | `{{LAND_PFP}}` | `D-T2-PAIR-1` |
| `2026-11-07-t2-lc-support/` | `HANDOFF_LC.md` `{{RES_LCS}}…` | `{{LAND_LCS}}` | `D-T2-LC-1` |
| `2026-11-07-t2-pin5/` | `README_BANK.md` `{{RES_PN5}}…` | `{{LAND_PN5}}` | `D-T2-PIN5-1` |
| `2026-11-07-t2-stage5-driver/` | `HANDOFF_S5.md` `{{RES_S5D}}…` | `{{LAND_S5D}}` | `D-V8F-T2-STAGE5-1` |
| `2026-11-07-t1-blind/` | `RESULT_T1BLIND.md` `{{RES_T1B}}…` | `{{LAND_T1B}}` | `D-V8F-T1-BLIND-1` |
| `2026-11-07-t2-ralign-gap48/` | `gap48/RESULT_GAP48.md` `{{RES_GAP}}…` | `{{LAND_GAP}}` | `D-V8F-T2-RALIGN-1` |
| `2026-11-07-t2-renc-prereg/` | `PREREG_T2_RENC.md` `{{RES_RENC}}…` | `{{LAND_RENC}}` | `H-T2-RENC-1` |
| `2026-11-07-theory-p1pb/` | `P1PB_RESULT.md` `{{RES_P1B}}…` | `{{LAND_P1B}}` | `D-V8F-THEORY-P1P-1` |
| `2026-11-07-wsp-gain/` | `raw/RESULTS_READ2.md` `{{RES_WG2}}…` | `{{LAND_WG2}}` | `D-V8F-WSPGAIN-1` |
| `2026-11-07-aucmap/` | `RESULTS_AUCMAP.md` `{{RES_AUC}}…` | `{{LAND_AUC}}` | `D-THEORY-AUCMAP-1` |
| `2026-11-07-theory-amend3/` | `THEORY_CRAWL_AMEND_3.md` `{{RES_A3}}…` | `{{LAND_A3}}` | `H-THEORY-T5-1` |
| `2026-11-07-p5-dilution/` | `p5/P5_RESULT.md` `{{RES_P5R}}…` | `{{LAND_P5R}}` | `D-THEORY-T5-1` |
| `2026-11-07-wsp-gain-d/` | `RESULT_WSPD.md` `{{RES_WGD}}…` | `{{LAND_WGD}}` | `D-V8F-WSPGAIN-D-1` |
| `2026-11-07-p4-cueclass/` | `PREREG_P4_CUECLASS.md` `{{RES_P4C}}…` | `{{LAND_P4C}}` | `H-THEORY-P4-1` |
| `2026-11-07-dk141/` | `README_DK141.md` `{{RES_DK}}…` | `{{LAND_DK}}` | `D-DK141-1` |
| `2026-11-07-t2-dom-lora-freeze/` | `PREREG_T2_DOM_LORA.md` `{{RES_DOM}}…` | `{{LAND_DOM}}` | `H-T2-DOM-LORA-1` |
| `2026-11-07-wsp-lora-prereg/` | `PREREG_WSP_LORA.md` `{{RES_WLR}}…` | `{{LAND_WLR}}` | `H-WSP-LORA-1` |

The v43 build is `patch_v43.py` (anchored edits, every table copied line for line from its source or hand-written and counted), checked by `checks_v43.py` and re-measured by `audit_v43.py` (`v43_NUMBERS_AUDIT.md`).

=== C3A ===
research report, v42 of 2026-11-07, results through repository tip `1887e9b8` and, within it,
-->
research report, v43 of 2026-11-07, results through repository tip `{{TP}}` and, within it,

=== C3B ===
and the theory of the crawl with the weight-space reading (§7.44) (v41: `bd98063d`;
-->
the theory of the crawl with the weight-space reading (§7.44), the T2 attribution and the LoRA launch (§7.45), and the theory after six reads (§7.46) (v42: `1887e9b8`; v41: `bd98063d`;

=== C42S ===
and the pre-registered weight edit WSP-GAIN is built and not yet read. No counted DOMINANT exists, there is no scale signal, and nothing in v42 is a driving result.
-->
and the pre-registered weight edit WSP-GAIN was built and not yet read at v42 (read in v43: HARMS, §7.46.3). No counted DOMINANT exists, there is no scale signal, and nothing in v42 is a driving result.

=== C63 ===
the pre-registered weight edit WSP-GAIN is built and not yet read. **[H]** for the interpretation
-->
the pre-registered weight edit WSP-GAIN was built and not yet read at v42 (read in v43: HARMS, §7.46.3). **[H]** for the interpretation

=== C64 ===
and leaves the weight edit WSP-GAIN unread; the lever ranking is perception first, then a zero-training frontier point.
-->
and left the weight edit WSP-GAIN unread at v42 (v43 reads it and P1′b: §7.46); the lever ranking is perception first, then a zero-training frontier point.

=== C5951 ===
the 0.2 s backward-difference acceleration, the curvature)
-->
the 0.1 s backward-difference acceleration (the v42 text said 0.2 s; §7.46.5), the curvature)

=== C6030 ===
**No WSP-GAIN result exists and none is quoted.**
-->
**No WSP-GAIN result existed at v42 and none was quoted; the read is in §7.46.3 (HARMS in 4 of 4 cells).**

=== C6036 ===
| P1′ | shrinkage ratio rises with go/stay confidence (T1′) | **NOT EVALUABLE** | CONFIDENT 24 windows (floor 30)
-->
| P1′ | shrinkage ratio rises with go/stay confidence (T1′) | **NOT EVALUABLE** at v42 (P1′b on 60 holdout clips: FAILS, §7.46.2) | CONFIDENT 24 windows (floor 30)

=== C6040 ===
| WSP-GAIN | restoring the base's gains is an admissible chain point | **PRE-REGISTERED, not yet read** | no number | [H] |
-->
| WSP-GAIN | restoring the base's gains is an admissible chain point | **HARMS in 4 of 4 cells (read in v43, §7.46.3)**; at v42 pre-registered, not yet read | `share_closed` 0.178 to 0.247 | [M], T1-kin |

=== C6118 ===
and the weight edit WSP-GAIN is pre-registered, built and not yet read.
-->
and the weight edit WSP-GAIN was pre-registered, built and not yet read at v42 (read in v43: HARMS, §7.46.3).

=== C6247 ===
**pre-registered, built and unit-tested, not yet read** (§7.44.6): no edited checkpoint was scored and no number is quoted |
-->
**read in v43: HARMS in 4 of 4 cells** (§7.46.3); at v42 pre-registered, built and unit-tested, not yet read (§7.44.6) |

=== C6300 ===
and WSP-GAIN was not read.
-->
and WSP-GAIN was not read at v42 (v43 reads it: §7.46.3).

=== NOTES ===
Carried v42 lines edited in place (blocks C3A, C3B, C42S, C63, C64, C5951, C6030, C6036, C6040, C6118, C6247, C6300; each is an old text, a line with three characters --> and the new text; C42S occurs twice in the base, in the status line and in 8.1 item 32). Every other v42 line is untouched; the v42 sections 7.42 to 7.44 and the v42 8.3 Update, 8.4 rows and section-9 item are kept as dated statements of what was known at v42 (for example the sentence that no LoRA step had run at the v42 build stays in Update (v42)).
Inserted blocks: V43S (shared sentence: status line end and 8.1 item 33), ABS1 (two abstract bullets before the next bullet), NXT (appended to the abstract's next bullet), S81 (8.1 item 33), R83 (8.3 Update v43), ROW (8.4 rows 1bg to 1bl), I21 (section 9 item 21), APP (section 10 block, one row per new raw folder: 21 rows).
Section files sec_745_t2attrib.md and sec_746_theory.md are inserted as written (directive lines replaced by their tables; no other change). The a0 difference is written 0.1 s in the v43 text; the v42 line 5951 said 0.2 s and is edited (C5951).

Edits made to the delivered inputs by the build agents (each original is kept in scratch/ with its sha256 recorded in HANDOFF_V43.md):
1. sec_745_t2attrib.md, header of the RT table: the cell `|RT| below floor at both seeds` became `\|RT\| below floor at both seeds` (the bare pipes made the header seven cells; the patch stopped on ragged columns).
2. sec_746_theory.md: `M1 of $X - $ P10 is` became `M1 of $X$ − P10 is` (a space before the closing dollar sign rendered as literal text and the math guard stopped on it).
3. Margin of the s1 ADE6 row of the RT table (build agent 4): 0.046 became 0.045 in sec_745_t2attrib.md (the table cell and the sentence after it) and in the blocks V43S, R83, ROW (row 1bg), I21 and ABS1. The margin is the seed floor of MC less |RT_s1| from RT_VERDICT.json: 0.508858507438103 − 0.463408725035914 = 0.04545, which is 0.045 at three decimals; 0.046 came from subtracting the four-decimal cells 0.5089 and 0.4634 (0.0455). The other two margins (0.0064 and 0.038) were right. The earlier audit matched 0.046 to an unrelated source line, so the audit could not see it; checks_v43.py now recomputes all three margins from the JSON.
4. Two cards were banked after the sections were written (build agent 4). Row 1bh (8.4) and the Update (v43) bullet in 8.3 named `DRAFT_PREREG_T2_DOM_LORA.md` as a draft that is not frozen: both now name the frozen card `PREREG_T2_DOM_LORA.md` (sha256 `cad25e83…`, folder `raw/2026-11-07-t2-dom-lora-freeze/`), and row 1bh does not say "log item 4" (that was the draft's log; the frozen card states the same LC and B′ control in its section 2). Section 9 item 21 says the dominance read of LC is frozen and has not run. NEW row 1bl (8.4): `PREREG_WSP_LORA.md` (frozen `0529fcfb…`, folder `raw/2026-11-07-wsp-lora-prereg/`), a weight-space read of the RENC and L LoRA deltas, pre-registered at the PI's request, not run. Both folders are added to the section 10 table (rows 20 and 21; claim rows `H-T2-DOM-LORA-1` and `H-WSP-LORA-1`, both in the register). Nothing is quoted from the two cards except their names, shas and status. Section 7.45.1 still speaks of the LoRA dominance draft as it stood when the section was written; it was not edited.
5. Patch guards changed (patch_v43.py): (a) the file name `RETRACTION_LOG` is removed before the `retract` substring guard, because it is a path and not prose; the banned past-tense word itself is still scanned with a word boundary; (b) the italic-quotation corpus is compared modulo the `**` bold markers of the sources, an escaped star `\*` counts as a literal star, and `BUILD_BRIEF_V43.md` is a source for the PI's verbatim message of 2026-11-07 (the only quotation that is in no repository file); (c) the expected row list is 1bg to 1bl and the appendix table 23 lines; (d) the register rows `H-T2-DOM-LORA-1` and `H-WSP-LORA-1` and the sha prefixes `cad25e83` and `0529fcfb` are asserted.
6. Audit change (audit_v43.py): `.jsonl` and `.out` files join the source suffix list, and the printed 1 to 4 decimal roundings of the longer floats in three named files only (`wsp-gain/raw/READ1.json`, `lora_L_launch/L_s0/train_log.jsonl`, `lora_L_launch/timing_b64.out`) are registered as "rounding of a longer float" (1.7804, 1.8993, 2257.97, 4509.05, 6744.30, 8994.28, 907.7). No number is on an allow-list without a source.
7. Check change (checks_v43.py): the removed-lines check is derived from this file (the removed lines are exactly the v42 lines that contain the old text of the twelve C blocks); the table, heading, equation, row and appendix counts are those of v43; 18 new checks tie each headline number of 7.45 and 7.46 to its raw file on the tip (VISION-USED, MC and BPC reads, RT-WITHIN-FLOOR and its margins, F′ and 2F′, M-ALIGN and GAP48, WSP-GAIN-D, WSP-GAIN READ 2, P5, the L step time, AUCMAP, P1′b, the stage-2 BLIND intervals, the two late cards, the 21-row appendix table re-derived, every 8-hex prefix, the a0 entry of the retraction log).
