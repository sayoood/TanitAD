# PREREG_FROZEN: H-GAPDECOMP-M-1, class M of the gap against P10 split by GT path shape and speed

Frozen before any subclass number is computed (sha256 and wall clock in `FROZEN_SHA.txt`). Zero training, CPU. Tier **T1-kin** (own executed state, logged cameras; a self-action open loop). Evidence class of every number: **MEASURED (ours; artifacts in the manifest)**, except the Beside block, **ESTIMATED**. The Master Mind registers the claim row; this card does not.

**Question.** The landed split (`dagland/gapdecomp`, H-GAPDECOMP-1) put the part of the gap against P10 that keeps Read A from DOMINANT in class M (2,032 moving, no-launch windows of the 2,399). Does that M gap sit in one identifiable part of M?

**Windows.** Class M exactly as the landed split defines it (`gapdecomp.classify`: `v0 >= 0.5` and not GT-launch). Same arrays (`dagland/gapdecomp/raw/arrays/{CTRL,SOUP}_s{0,1}.npz`), same 19-row definitions, same draws. Imported, not copy-edited.

**Subclasses of M** (from the GT track `g` and the logged `v0` of the dump only; thresholds fixed here):
* **Shape**, from the GT heading change over the 6 s horizon: `|dpsi|` = absolute value of the sum, over the pair-valid GT steps, of the wrapped step-to-step heading change of `four_families._seq_geometry(g, 0.2)` (the instrument's own heading and pair-valid mask; steps below 0.5 m/s carry no heading), in degrees. **STRAIGHT** `|dpsi| < 10`; **GENTLE** `10 <= |dpsi| <= 45`; **TURN** `|dpsi| > 45`. A window with no pair-valid GT step has `dpsi = 0` and is STRAIGHT; their count is reported.
* **Speed**, by `v0` (m/s): `[0.5, 3)`, `[3, 8)`, `[8, 15)`, `[15, inf)`.
* **Cross** 3 x 4. Support floor of a cell: **>= 60 windows and >= 20 episodes**, counted on the 2,032 M windows (so the partition is the same for every row). A cell below the floor is pooled: all below-floor cells of one shape form one cell `<SHAPE>/other`; if that pooled cell is still below the floor, it joins the single cell `ALL/other` with the other such cells; `ALL/other` is reported but cannot be named.
* Three partitions of M, each covering the 2,032 windows exactly once: shape (3), speed (4), cross (12 after pooling).

**Rows** (per-window arrays of the landed split): ADE6, yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share. Row `r` of arm X against P10 is the mean of `a - b` over the row's valid windows (NaN pairs dropped); Delta_M is that mean over the valid M windows; `sum_sub w_sub Delta_sub = Delta_M` exactly. **Arms:** DAG-OACC (manifest role CTRL, the headline) and SOUP, each against P10, at s0 and s1.

**Estimator.** Paired episode-cluster bootstrap, B = 2,000, seed 0, the landed draw stream (`taniteval.ci`: `default_rng(0)`, `choice(uniq, n_ep)` over the row's own valid-pair episodes), subclass statistics recomputed on every draw, percentile 95 % interval. It answers **another draw of episodes only**; it is blind to training variance (`H-ESTIM-SEED-1`; one training run per seed); the planner is deterministic, so no inference seed.

**Known answers, all before any subclass number is read** (a failure refuses that row or arm):
* KA1: each partition covers M once (counts sum to 2,032 over 140 episodes, equal to the landed class M); the pooling rule leaves every named cell at or above the floor.
* KA2: for each of the 5 rows, 2 arms, 2 seeds, 3 partitions: `sum_sub w_sub Delta_sub`, and the recomputed Delta_M interval, contribution and n, equal the landed `CLASS_SPLIT.json` class M entry within **1e-9** (same arrays, same draws, same code path).
* KA3: (X, X) reads exactly 0.0 in every subclass, interval [0, 0].
* KA4: the shares of a partition sum to 1 (1e-9) at the point and in every draw.
* KA5 (pytest): a planted +d in one subclass is named LOCALISED; the same +d in every subclass is DIFFUSE; a shrunken support floor refuses to name.
* KA6: the Beside zeroing with an empty subclass reproduces each row's landed Delta and the landed `D2_worse_rows`.

**Verdict per row, arm, seed** (a candidate is a subclass of any partition at or above the support floor on the row's valid windows):
* **NOT DEFINED** at a seed if the interval of Delta_M contains 0 (a share of a difference that may be zero has no meaning).
* A candidate **qualifies** at a seed iff its window share `w_sub` of the row's valid M windows is **<= 0.30**, its share of Delta_M has point **>= 0.5**, and the lower bound of its share interval is **> 0.3**.
* **LOCALISED** iff one candidate qualifies at **both** seeds (DAG-OACC for the headline; SOUP by the same rule, stated separately). The named subclass is the qualifying candidate with the highest min-over-seeds point share; every qualifying candidate is listed.
* **DIFFUSE** iff Delta_M is separated at both seeds and no candidate qualifies at both.
* For ADE6 the M gap of DAG-OACC is a gain (Delta_M < 0); the share is of that signed gain and the verdict says where the gain lives.
* Scope of the interval: it is per candidate and does not carry the choice among candidates. The number of eligible candidates per row is reported, and, descriptively, how often the named subclass is the argmax-share eligible candidate across the draws. No multiplicity adjustment is applied or implied.

**Beside (descriptive; ESTIMATED where marked; decides nothing).** B = the subclasses named LOCALISED for any row of DAG-OACC. If B is empty, B = the one candidate with the highest mean DAG-OACC point share over the four rows yaw-rate, curvature, L2, M1 and both seeds among candidates with `w_sub <= 0.30` and support on all four rows, labelled **leading, NOT localised**.
* Plan against GT in each B subclass (MEASURED, from the dumps' `cl` track of the arm and `fl` track of P10, same windows, the instrument's `_seq_geometry`): per window the mean over valid steps, then the mean over the subclass's windows. Speed: plan, GT, plan minus GT. Curvature over pair-valid steps: mean |kappa| of the plan, of the GT, their difference, and the signed mean error (plan minus GT, left positive). Sign of every error is plan minus GT.
* D2' (ESTIMATED, decomposition arithmetic, not a retrained model): for each B subclass, arm (DAG-OACC, SOUP) and seed, `Delta' = Delta` with the subclass windows' contribution set to 0 (n unchanged; trigger windows keep their gap), all 19 rows, interval from the same draws; D2' holds iff no row of the 19 is separated-worse (TAC rows higher is better, every other row lower is better).

**Not decided here.** Which lever closes the subclass; that is the next arm. A DIFFUSE verdict is a finding that no single subclass owns the M gap and the next lever is then named in RESULTS.md. No threshold, subclass, candidate or row is changed after data.
