# PREREG_AUCMAP: how much go/stay discriminability does the dominance corner need? (T2, quantitative form)

Agent: AUCMAP (Sonnet). Parent text: `theory/THEORY_CRAWL.md` section T2 and prediction P3 (frozen `6e83451c`), `THEORY_CRAWL_AMEND_1.md`, `THEORY_CRAWL_AMEND_2.md`. Dev box, CPU only, zero training, `CUDA_VISIBLE_DEVICES=-1`.
Freeze: the sha256 and the `date` timestamp of this file are in `PREREG_AUCMAP.sha256`, written BEFORE any gated number was computed. Evidence classes: [M] measured (artifact named), [P] published, [E] estimated, [H] hypothesis. Tier of every number below: T1-kin (logged cameras; `reindexed` = longitudinal-only approximate re-perception; NOT T2).

## 0. Standing label on every table
**CEILING-TYPE.** Synthetic posteriors are built from the window's true label (ego and future allowed for labels), and the oracle-level levers force the logged speed (a future-derived channel). Neither is admissible at inference. These arms decide NO driving claim and are never a driving result. The only admissible arm is the REAL-posterior anchor gated by the admissible (heads) level. STRATEGIC family: PAUSED (PI ITEM 25, 2026-09-19), reported beside, never a pass; the other three families keep full force.

## 1. Question and the lever
T2: no lever downstream of the features beats the Bayes ROC of the go/stay decision on those features; every such lever picks a point on that frontier. This card measures where the corner sits on the frontier by gating banked levers with a posterior of controlled AUC.
**The gated lever.** For a banked lever arm X (rollouts exist for every window, per seed) and the ungated base R = RIC0, the gated arm takes, per window w, the 19 per-window row values of X where the gate is ON and of R where it is OFF. Windows are independent rollouts, so a per-window selection is the arm "apply the lever on windows whose go-posterior is at or above tau". Gate ON for a window is decided once, from the tick-0 posterior (the lever then acts on all three replans of that window).
**Levers (all reindexed scene, bare = no curvature box; per-window arrays under `dagland/motion_c/*_read/raw/arrays`):**
* ADMISSIBLE (heads level, `RILF*`): LF050, LF075 (lf_read), LF100 (lf5_read, A5), LFB05, LFB10 (lf6_read, A6 bounded lifts).
* CEILING-LEVEL (logged level, `RILFO*`): LFO075 (lfo_read), LFO100 (lf5_read), LFOB05, LFOB10 (lf6_read). Read beside; the double ceiling (oracle level and oracle gate) bounds what ANY gate can do.
* NOT GATED, named: `lonfloor` / `kbox2lf` (decode levers on DAG-OACC / SOUP bases; text-only numbers in `feas2_t1/RESULTS_FEAS3.md`, no per-window arrays in this layout). `LFN*` noise arms: not gated.
* P10 = the `b_i` arrays of the same files. Windows: 2,399 per seed, 141 episodes, identical across arms (asserted).

## 2. Pre-freeze inspections (structure only; no gate was computed)
Keys and shapes of the arrays (19 rows, `a_i` arm, `b_i` P10, eid, v0, g, ws, clip_index); ungated M1 mass by v0 band of RIC0 / RILF100 / P10 (excess of RIC0 over P10, windows in band: v0<0.5: 3.6 of 101; 0.5-3: 24.6 of 174; 3-6: 48.3 of 333; 6-8: 25.7 of 253; 8-12: 8.0 of 583; >=12: 2.1 of 955; total about 112, so v0<8 carries about 91 %); keys of `P1P_PERWINDOW.npz` (861 windows = every window with v0<8, all in the 2,399 set by (clip_index, ws) = (ci, t0)); the 19 row labels of the T2 card. The banked ungated rows of LF050 / LF075 / LF100 / LFO* / LFB* (RESULTS.md files) are inputs, not gated numbers.

## 3. Label
GO (the registered probe label, population-free here): `v6 = |g[29] - g[28]| / 0.2` (chord speed over the last 0.2 s of the 6 s logged horizon), **GO = (v6 >= v0 + 1.0)**, v0 the window's start speed (`trigger_probe/PREREG_FROZEN.md` section 2; `gapdecomp.classify` uses the same chord). Provenance: ego and future, offline label only. Control C6: equals `P1P_PERWINDOW.npz` field `go` on all 861 windows.
Sensitivity label KEEP (beside, never decisive): `v6 >= v0 - 1.0` (hold-or-accelerate). The registered label means accelerate; the lever also acts on windows where the human holds speed, so A* is also read under KEEP.

## 4. Scopes
* **ANCHOR scope (headline):** windows with v0 < 8 (861). Outside it the gate is OFF. This is the coverage of the real posterior.
* **ALL scope (beside):** all 2,399 windows scored (synthetic only).

## 5. Posteriors
* **Synthetic:** `s = y * d' + eps`, eps ~ N(0,1), `default_rng([20261107, level_index, draw])`, y the label as 0/1, d' = sqrt(2) * Phi^-1(AUC) for AUC < 1; AUC 1.0 uses `s = 1e6 * y + eps` (perfect separation). Nominal AUC levels: **{0.50 (no skill), 0.68, 0.785, 0.85, 0.90, 0.95, 0.99, 1.0}**, K = 20 noise draws each. The realized AUC (Mann-Whitney, on the scope windows) is reported per draw.
* **Shuffled-label arm:** y permuted across the scope windows (seed 20261107), then the AUC-1.0 construction; its realized AUC against the TRUE label must be about 0.5 and its curve is the no-skill line.
* **Real anchor:** `post_primary` of `theory/wsp_read/P1P_PERWINDOW.npz` (P-MOVE V+E linear probe, five fold models averaged, never fitted on an eval window; banked, hash in `P1P_FILES.sha256`), windows v0 < 8. It is the registered probe applied to windows outside its 0.5-6 m/s population (P1P / P1PB convention); that extrapolation is stated, not hidden.
* **Gate sweep:** ON fraction q in {0, 0.05, ..., 1.0} (21 points) of the scope windows, the top-q by score (rank form of tau; q = 0 is tau = +inf, ungated base; q = 1 in ALL scope is tau = -inf, the ungated lever).

## 6. Estimator and rows
Paired episode-cluster bootstrap, B = 2,000, seed 0, the `gapdecomp.decompose` draw stream (`cached_draws`), re-implemented vectorised in `aucmap.py` and checked equal to `rr.rows_of` (control C3). It answers another draw of EPISODES only. It is blind to training variance (two seeds, one training run each: `H-ESTIM-SEED-1`) and to inference variance (the iCEM-style sampler is not involved in these rollouts). The noise draws of the synthetic posterior are a third, separate variance, reported as the reach rate.
Rows: the 19 T1-kin rows. All 19 rows of the T2 dominance card have a T1-kin counterpart (same labels). What has NO T1-kin counterpart: the T2 re-perception renderer and its gates G1-G4, the R-SCN scene refusal, the scene-cluster unit (48 scenes, 93 starts versus 141 episodes, 2,399 windows here), and the T2 tier itself (reindexed T1-kin re-perceives longitudinally only).

## 7. The corner (both seeds, X = gated arm, R = RIC0, P = P10)
* **M1-P:** M1 row of X - R separated-better, and the mean over seeds of share_closed = -delta(X - R) / delta(R - P10) (`rr.m1_cell`) >= 0.5 (the programme's P bar).
* **M1-FULL:** M1 row of X - P10 NOT separated-worse (interval lower bound <= 0) at both seeds, and M1-P.
* **GUARD:** ADE6 delta(X - R) <= +0.0604 m, and no ACTIVE-family non-crawl row separated-worse in X - R. ACTIVE = LONGITUDINAL, LATERAL, TACTICAL rows other than the crawl rows (yaw-rate, curvature, L2 curvature, M1); S-POS rows listed beside (STRATEGIC PAUSED). Direction-aware (`SR.worse`).
* **CORNER-FULL (headline) = M1-FULL and GUARD.  CORNER-P (beside) = M1-P and GUARD.**
* Beside, deciding nothing: Read A (D1, D2, D3) of X - P10 as `rr.read_a_block`.

## 8. A* and its interval
For a (lever, scope, corner): a (draw, AUC level) REACHES if some q in the grid satisfies the corner at both seeds. The reach rate of a level is the fraction of its 20 draws that reach. **A* = the smallest nominal AUC level with reach rate >= 0.5**, reported as a grid cell (previous level not reached, this level reached). Interval = [smallest level with reach rate >= 0.05, smallest level with reach rate >= 0.95]; the episode-cluster bootstrap sits inside each predicate through the percentile intervals. "Not reached" at a level means no q reaches at both seeds in at least half of the draws. Multiplicity, disclosed: 21 q points (adjacent points are correlated), 5 admissible levers for the headline "any admissible lever" A*_any = min over the 5, 2 corners, 2 scopes, 2 labels. The per-lever A* are listed; a corner reached only at a single lever and a single q is flagged FRAGILE.
**Headline:** A*_any under CORNER-FULL, ANCHOR scope, label GO, ADMISSIBLE levers.

## 9. The real anchor and what "near" means (fixed now)
For each admissible lever: the anchor curve (share_closed, ADE6 delta(X - R)) over the q grid, both seeds. It is NEAR the synthetic curve if, at the synthetic level whose nominal AUC is closest to the anchor's realized AUC (and also at 0.785), at least 75 % of the 21 q points have both coordinates inside the pointwise [2.5, 97.5] percentile band of the 20 draws, at both seeds. The anchor's corner status is read with the same predicates (one posterior, so no reach rate).

## 10. Outcomes, committed before any gated number
* **(i) T2 stands quantitatively:** the headline reach rate at the 0.785 level is < 0.5 and the real anchor does not reach CORNER-FULL, and A*_any <= 1.0. Then A*_any is the go/stay AUC target for L (the LoRA arm's P4 read compares its AUC with A*; reaching it is necessary for the corner under this lever family, not sufficient).
* **(ii) T2's bound is not binding here:** the headline reach rate at the 0.785 level is >= 0.5 (or the real anchor reaches CORNER-FULL). Then the lever ranking in THEORY_CRAWL section T2 is wrong for this family, and the next arm is the posterior-GATED lever itself: the named (lever, q) at the real posterior, built as a rollout arm (gate inside the replan loop), under its own card. Flag it if only one lever and one q reach.
* **(iii) The corner is not reachable by any gate on this lever family:** no admissible lever reaches CORNER-FULL even at nominal AUC 1.0, and (read beside) neither does the CEILING-LEVEL family at AUC 1.0. Then A* is undefined (> 1.0), the guard rows are driven by the lever operator and not by go/stay discrimination, consistent with the A5 joint reading (OPERATOR-IS-THE-PROBLEM), and the next arm changes the lever / the decode path (F), not perception. If only the CEILING-LEVEL family reaches at AUC 1.0, A* is read on that family and flagged double-ceiling.
* **(iv) Scope artefact guard:** if CORNER-FULL is reached in ALL scope but not ANCHOR scope, the verdict is that the probe's coverage, not the AUC, binds; A* is then reported for ALL scope and the anchor is flagged out-of-reach by scope.

## 11. Controls and tests (all before the headline is read)
C1 q=0 reproduces R's rows exactly. C2 ALL scope, q=1 reproduces the lever's banked rows exactly (RILF100 s0 share_closed 0.899 on the P3 record; delta equal to `rr.rows_of` to 1e-12). C3 vectorised bootstrap equals `rr.rows_of` (delta, lo, hi) on the 19 rows for 6 arms. C4 realized AUC of every synthetic draw within 0.02 of nominal (AUC<1). C5 no-skill and shuffled arms: reach rate at most the nominal false-positive allowance and share_closed vs ADE6 curve on the lever's own ungated line. C6 label equals P1P `go`. C7 AUC 1.0 gated lever: NEW arm (no banked oracle-gated lever exists); its q=1 point equals the ungated lever. C8 window alignment across arms and `P1P` v0 versus array v0. Tests with deliberate-regression mutants (selection inverted, scope mask dropped, q off by one, labels not permuted) each must turn a test RED.

## 12. Four families, per arm
Reported per arm at the reached/best q: LONGITUDINAL, LATERAL, TACTICAL rows with interval and estimator; STRATEGIC: PAUSED (PI ITEM 25), S-POS rows beside. ADE alone is never the result.
