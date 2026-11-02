# PREREG_FROZEN: H-GAPDECOMP-1, the gap against P10 split by window class

Frozen before any class-level number is computed (sha256 and wall clock in `FROZEN_SHA.txt`). Zero training, CPU. Tier **T1-kin** (own executed state, logged cameras; a self-action open loop). Evidence class of every number: **MEASURED (ours; artifacts in the manifest)**. The Master Mind registers the claim row; this card does not.

**Question.** How much of the gap that keeps Read A from DOMINANT lives in the go/stay trigger windows (GT-launch plus GT-stay)?

**Classes** (eval-141, the 2,399 causal-lane windows of the SOUP read; from the GT track `g` and the logged `v0` of the dump only; definitions of H-CRAWL-01, `crawl_diag.py:144` and `mlaunch.py:84`):
* **L** GT-launch: `v0 < 6` and `|g[29]-g[28]|/0.2 >= v0 + 1` (the 327).
* **S** GT-stay: `v0 < 0.5` and the GT chord speed never reaches `v0 + 1` over the 6 s horizon.
* **O** other-stopped: `v0 < 0.5`, neither L nor S.
* **M** moving, no launch: `v0 >= 0.5`, not L.
* Trigger set **T = L + S**. Control K1: v0 finite; L, S, O, M pairwise disjoint and summing to 2,399; L equals the banked `launch` of `crawl01_arrays.npz` (327 windows / 49 episodes) and S its banked `stay`.

**Rows.** The 19 rows exactly as the SOUP / LAUNCH-INTENT readers compute them (DAG-ORC `Rig`, Read A comps rows, t1kf TAC / DK rows, t1hd S-POS rows; `extract_arrays.py` writes their per-window arrays). Every row is a window mean of a paired per-window difference `a - b` (arm X minus P10; S-POS: S minus CTRV within X's run), over its valid windows `n` (NaN pairs dropped). So the identity is exact:
`Delta = sum_c w_c * Delta_c`, `w_c = n_c / n`, share_c `= w_c*Delta_c / Delta` (not clipped; shares sum to 1). A row that fails K2 is reported NOT DECOMPOSABLE with the reason and gets no class number.

**Arms.** DAG-OACC (manifest role CTRL), REP, SOUP, each against P10, at s0 and s1. All six dumps and tick files exist (manifest `runs_soup.json`, sha `35833493...`).

**Estimator.** Paired episode-cluster bootstrap, B = 2,000, seed 0, per seed. Each row uses the instrument's own draw stream (`taniteval.ci`: `default_rng(0)`, `choice(uniq, n_ep)` over the row's own episode pool: valid-pair episodes, S-POS all 141); the class statistics are recomputed on each draw; percentile 95 % interval. It answers **another draw of episodes only**; it is blind to training variance (`H-ESTIM-SEED-1`; each arm is one training run per seed). The planner is deterministic, so there is no inference seed.

**Known answers, all before any class number is read** (a failure refuses that row or arm):
* K2: for every row, arm, seed: `sum_c w_c Delta_c`, lo, hi equal the banked row (`SOUP_READ.json` tables `CTRL_P10`, `SOUP_P10`; the LAUNCH-INTENT read `REP_P10`) within 6e-5 (the banked values are rounded to 4 dp), and `separated` / `separated_worse` equal.
* K3: (X, X) reads exactly 0.0 in every class and every row, interval [0, 0].
* K4: my total interval equals `paired_episode_cluster_bootstrap` called directly on the same arrays, at its printed precision.
* K5: the four shares sum to 1 (1e-9) at the point and in every draw; the planted-effect test in `test_gapdecomp.py` passes.

**Verdict per row in {ADE6, yaw-rate MAE, curvature MAE, L2 curvature MAE, M1 crawl share}, per arm and seed** (T-share = share_L + share_S, with its bootstrap interval; the other 14 rows are reported, not judged):
* **NOT DEFINED** if the interval of `Delta` contains 0 (a share of a difference that may be zero has no meaning).
* **TRIGGER-BOUND** iff T-share point >= 0.7 and its interval lower bound > 0.5.
* **NOT TRIGGER-BOUND** iff T-share point <= 0.3 and its interval upper bound < 0.5.
* **MIXED** otherwise.
* Per arm: TRIGGER-BOUND (resp. NOT TRIGGER-BOUND) iff both seeds are; a NOT DEFINED seed is stated as such; any other combination is MIXED. The headline arm is DAG-OACC (both seeds). The same rule is applied to REP and SOUP, each stated separately.

**Beside (ESTIMATED; decides nothing).** For each arm, seed and row: `Delta' = sum over c in {O, M} of w_c Delta_c` (the trigger windows' Delta set to 0, `n` unchanged), its interval from the same draws, and whether the row is separated-worse (worse side by the row's direction: TAC rows higher is better, all others lower). D2' = no row of the 19 separated-worse; reported per seed and for both seeds, for DAG-OACC and SOUP (and REP). This is decomposition arithmetic, not a retrained model: no intervention gives the trigger windows P10's error.

**Not decided here.** Whether any lever closes the trigger windows; that is the next arm. A refutation of "TRIGGER-BOUND" is a finding that the gap lives in O and M, and the next lever is then named in RESULTS.md. No threshold, class or row is changed after data.
