# PREREG PRIVGEOM-SAFE: does the information the NC and DAC rows miss exist UPSTREAM, in privileged per-proposal geometry? (Research Lab agent for the Master Mind, 2026-10-19)

**Written BEFORE any label is read with the scalars.** The sha256 of this file is recorded in `PREREG_PRIVGEOM_SAFE.sha256` / `SEALED_AT.txt` before the first stage that opens a label array. A change comes only by a written amendment (`PREREG_PRIVGEOM_SAFE_AMENDMENT<k>.md`) saved and hashed before the number it governs. Amendment 1 (the fit / analysis code pins) is sealed before the gate stage, which is the first stage that reads a label.

- Tier: SELECTION, SEEN scenes (`D-L1-G0-1`), P2 IN-SAMPLE (its frozen rows are in-sample; every ARM below is out-of-fold). 0 GPU, CPU only. No navtest and no dev table is read.
- ⛔ **DIAGNOSTIC ONLY. Privileged geometry is NEVER an inference input** (inference is vision-only; `CLAUDE.md` binding rule of 2026-08-03). Nothing here is a deployable arm; the result decides which GPU arm is worth registering.
- Registers read at the tip `51099763`: `D-REFE-FASTNEG-CENSUS-1`, `D-REFE-PICKRULE-NC-1`, `H-REFE-RL1-NCRES`. Lab report `2026-10-19-safety-head-lit/REPORT.md` section 4.5 (D2); `MM_RULING_PICKRULE.md`.

## 0. Disclosures (what I had seen before sealing)

1. The three register rows above, including the census numbers they quote (DAC F minus Q1-3 within-token AUC -0.0718 / -0.0712; NC +0.0323) and the pick-rule audit outcome (ROW-MISS, Q1 0 of 82 / 0 of 83). I know the census L rows, not their values in this subset.
2. I opened NO `labels6` array and NO stored label of any kind; the census meta was read through its non-label members only (`log_idx`, `fold`, `code`, `key_*`, `log_rank`, `log_names`). The census `strata.json` (sha256 `67f43e55dd7c693e31cd991efc3d51fedc278331ba4f8f882e6adbdae8203dc6`) is label-free by its own proof; its `code` array is reused unchanged.
3. The Step 1 cost run (20 seeded sets, `run/cost/cost.json`, sha256 `92529fc585e25031e8dad31cf3793f85b0abb943f7e502bd6f9a5989178f47cd`) computed the scalars and NAVSIM's verdicts for 20 sets and **kept none of them**: only timings, RAM, the chosen keys and failure strings were saved. Nothing was compared with a stored label, and no scalar was compared with a recomputed verdict.
4. I know, from reading the PDM scorer source (`navsim @ 3e8291b`, `pdm_scorer.py`), that two of the fixed scalars contain the label-producing test (section 3.4). That is a property of the code, not of any data value, and it is why section 3.4 and the companion variants exist.

## 1. Question

`D-REFE-FASTNEG-CENSUS-1`: the DAC row has a fast-band deficit (-0.07 within-token AUC) that neither more fast labels nor a 2-layer head closes. `D-REFE-PICKRULE-NC-1`: the NC row scores an EP-led pick's NC-unsafe choices as confidently safe (median sigmoid 0.960), and no veto recovers. `H-REFE-RL1-NCRES`: the remaining levers change what the rows SEE. **Question: does the missing information exist upstream?** I.e. given a fixed list of privileged per-proposal geometry scalars beside the 256-d head token, does a linear row's within-token AUC in the fast stratum F rise (INFO-UPSTREAM), or not (NO-JUMP)?

## 2. Data and subset (every fact below was fixed before any label was read)

- S0's 15,808 fit sets at f = 10 % (117 logs, census order `meta.npz`, bank manifest sha256 `6296dfc0b03905038fbf132f82009f6b6d7e8d61fa9abfb65c9164f487a5f2a1`); per-log stores `C:/Users/Admin/srttc_s0/work/sets/<log>.npz` (READ ONLY; `traj3 [S,64,20,3]`); features `par18/fastneg/run/inputs/feat_f10.npy` (sha256 `f96ef18616ce37be5149ffc32a54046f6b6be5bf0bc3c8405f19bb20770c1af3`); census `meta.npz` sha256 `060af1a3f8f96f525056a37c7f6938050b3a0fbbb08e7b69bacbb85b43031675`.
- **Eligibility = a locally held nuPlan DB** (`navtrain_scenarios.index_dbs()`; `D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits`). **41 of the 117 f10 logs** qualify: 4,941 of 15,808 sets (31.3 %), 2,782 distinct tokens. The other 76 logs' DBs live only on the pod (`fetch_navtrain_dbs.py`); the brief forbids touching the pod. ⚠️ **This, not CPU, bounds the subset.** The eligible pool is a filter on FILE PRESENCE, chosen before any label; it is not a random sample of the 117, and what it licenses is a statement about these 41 logs.
- **Cost (MEASURED, Step 1, 20 seeded sets in 14 logs, 20 tokens, seed 20261019, `NAVSIM`'s own PDM objects on the landed labeller path):** per proposal **5.31 ms** NAVSIM simulate + score and **2.94 ms** scalars; per token 1.36 s (metric cache 1.08 s + map / observation prep 0.28 s); per log 0.21 s; per set 0.53 s. **Peak RSS 0.43 GB** per process. Amortised on the 20-set sample, 31.9 ms per proposal (about 1 set per token); on the full pool 20.3 ms per proposal.
- **Subset rule (fixed):** order = `numpy.random.default_rng(20261019).permutation(sorted(eligible logs))`; take the longest prefix, whole logs, whose projected cost (0.21 s/log + 1.36 s/token + 0.53 s/set, times a SAFETY of 1.5) fits **16 CPU-h**; STOP and report the cost if fewer than 8 logs fit. **Result: all 41 logs fit** (projected 1.78 CPU-h, 2.67 CPU-h with the safety; cap 16). `run/subset.json` sha256 `5574b57fae127e141788dcc001399e9e323348a8d4f203dc35c2ca3d0de61a25`. Folds = the census fold of each log (`rank mod 5`): logs per fold 10 / 8 / 8 / 3 / 12. Proposals per stratum S / Q1 / Q2 / Q3 / Q4 = 3,392 / 66,772 / 84,237 / 84,538 / 77,285 (strata = the census's, cuts fixed at 0.9076 / 1.0453 / 1.2228 from all 117 logs; F = Q4).
- Execution budget: at most 2 Python processes; psutil available >= 4 GB to start, the extraction pauses (scripted) below 3 GB; `CUDA_VISIBLE_DEVICES=-1`.

## 3. The scalars (the FIXED list; K = 15) and what they are

Computed per proposal on NAVSIM's SIMULATED states (the states the verdict itself is computed on; read from `PDMScorer._states` / `_ego_coords` after the landed `navsim_pdm_targets.score_plans` call), 41 steps at 0.1 s, the ego footprint = the 4 Pacifica corners. Definitions and caps live in `code/pg_geom.py` (sha256 pinned in section 12) and are exercised by 19 synthetic tests, four of them RED mutations.

**DAC row (11).** Drivable area = the union of the PDM drivable map's ROADBLOCK, INTERSECTION, DRIVABLE_AREA and CARPARK_AREA polygons (NAVSIM's `drivable_area_idcs`).
- `dac_clear_min`: min over steps and the 4 corners of the signed clearance (m; + inside, - outside) to the union's boundary, capped to [-10, +10].
- `dac_frac_out`: fraction of the 41 steps with fewer than 4 corners inside (NAVSIM's non-drivable step test).
- `dac_clear_t00 ... t40` (9 values): the per-step signed clearance (min over the 4 corners) at steps 0, 5, ..., 40 (t = 0, 0.5, ..., 4.0 s).

**NC row (4).** Agents = every tracked box of the metric cache's observation except the red-light geometry; velocities by forward finite difference of the box centroid between 0.2 s observation slots.
- `nc_min_dist`: min over steps and agents of the distance (m) from the ego footprint polygon to the box, capped at 20.
- `nc_min_tgap`: min over steps of gap / max(ego speed, 0.5 m/s), capped at 10 s; 10 s when there is no lead. LEAD = the box with the smallest gap among those whose polygon meets the STRAIGHT corridor ahead of the front bumper (0 to 50 m, half width 1.4485 m) in the ego frame of that step; gap = max(0, nearest longitudinal coordinate of the box's 4 vertices from the front bumper). ⚠️ A straight corridor APPROXIMATES "along the path"; a lead in a sharp turn is missed.
- `nc_max_close`: max over steps of max(0, ego speed - the lead's speed along the ego heading), capped at 20 m/s; 0 with no lead.
- `nc_min_ttc_margin`: min over steps of (TTC - 1.0 s); TTC = 0 if gap <= 0, gap / closing if closing > 1e-3, else 10 s.

The NC row receives its 4 columns and the DAC row its 11 (row-specific, not the pooled 15).

### 3.4 ⛔ The tautology (stated before any data): the primary G contains the label-producing test

- `min over steps of the signed clearance < 0` **is** "some step has a corner outside the drivable union", which **is** NAVSIM's DAC = 0 (up to polygon seams). So within a set every unsafe proposal has `dac_clear_min < 0 <= ` every safe proposal's: **the within-set AUC of `dac_clear_min` alone is 1 by construction**, and `dac_frac_out > 0` is the label again.
- `nc_min_dist = 0` is the box overlap that the at-fault NC test scores (NC = 0 or 0.5 additionally needs the at-fault type: a front or stopped-track contact, or a lateral contact while off the lane), and TTC margin = -1 and time gap = 0 are the same event.
- ⇒ **The primary read is a SUFFICIENCY test, not an information-content test.** INFO-UPSTREAM on DAC is expected by construction (forecast, section 10), and an INFO-UPSTREAM call on either row, taken alone, shows that the geometry that DEFINES the label closes the label. It does NOT show that a vision-reachable boundary-proximity signal exists. The MM's committed next arms (section 9) are kept as written; the **companion variants below are what separates the two readings**, and the hand-back must carry both.
- **Companion variants (registered now; DESCRIPTIVE, no authority over any branch; the same rule is applied to them and reported beside the primary):**
  - `DAC-nosign`: the DAC columns as `|clearance|` (`dac_clear_min`, the 9 key times), `dac_frac_out` DROPPED (10 columns). Inside and outside at the same distance look alike; no column is a monotone function of the label.
  - `NC-floored`: the NC columns with `nc_min_dist` and `nc_min_tgap` floored at 0.5 and `nc_min_ttc_margin` floored at 0.0: contact and a near miss within 0.5 m (or TTC < 1 s) become indistinguishable.
  - Committed reading: primary INFO-UPSTREAM **and** the companion INFO-UPSTREAM -> boundary / lead PROXIMITY information beyond the contact event exists upstream (this is the reading that supports SH5 / SH3); primary INFO-UPSTREAM and the companion NO-JUMP or INCONCLUSIVE -> only the label's own test closes it, and SH5 / SH3 are NOT supported by this arm.

## 4. Arms, recipe, folds

Per row `rho` in {NC (logit column 0), DAC (column 1)}, per training seed in {0, 1}, 5 folds = the census fold of the log (train on the other four, score the held-out one: OUT-OF-FOLD logits for every set of the subset). Within the subset only; **L, L+G and G are all refit on the same training sets.**
- **L** (the 256-d token; HEADROW-PROBE section 3.2 linear recipe, as in the census): `z = z_P2 + feat . (w - w_P2) + (b - b_P2)`, `(w, b)` initialised at P2's row, fp16 features cast to fp32 and detached, BCE-with-logits on the (soft) label `labels6[..., rho]`, Adam lr 1e-3, weight decay 1e-4, betas (0.9, 0.999), eps 1e-8, 20 epochs, batches of 256 sets, batch order = `torch.randperm` from `Generator().manual_seed(seed * 1_000_003 + n_train_sets)` reshuffled every epoch.
- **L+G** (the token plus the standardised scalars): `z = z_L + g_std . u`, `u` zero-initialised, **its own Adam group at lr 1e-2** (same weight decay, betas, eps). ⚠️ At lr 1e-3 a zero-initialised block cannot move more than about 0.32 in 20 epochs (about 320 steps on this subset), so the G block would be mute and the arm would read NO-JUMP by construction; lr 1e-2 is registered for the G block ONLY, and the planted-signal controls of section 7 are what prove it is not mute. `g_std` = each of the row's columns minus the TRAINING sets' mean over their proposals, divided by their std (a column with std < 1e-8 is set to 0); applied to the held-out fold with the training statistics.
- **G-only**: `z = g_std . u + c`, `(u, c)` zero-initialised, one Adam group lr 1e-2, no token. Reported ALONGSIDE; it takes no part in the decision.
- Stored logits are P2's frozen `z6` (census `meta.npz`); L+G and G-only differ from L only in the added block.

## 5. Reads and estimator

- **Within-token AUC** (the census definition, Amendment A1.3 of `PREREG_FASTNEG_CENSUS`): unsafe = label == 0, safe = label == 1 (NC = 0.5 excluded from pairs); pairs inside ONE set, both members in the stratum; score = the row's OOF logit; ties 0.5. Strata from the census `code`: **F = Q4**, **Q1-3 = Q1 u Q2 u Q3 (cross-quartile pairs inside the union included)**. Reported per arm, per row, per seed, in F and in Q1-3, with the pair counts.
- **Estimator:** paired LOG-cluster bootstrap over the 41 logs, B = 10,000, `numpy.random.default_rng(20260927).integers(0, 41, (10000, 41))`, the SAME draws for every quantity, percentile 2.5 / 97.5, every ratio a ratio of per-log sums, every difference computed on the same resampled logs. It answers "would another draw of LOGS say this?" and **nothing about another training run** (`H-ESTIM-SEED-1`); the two training seeds are a floor on training variance, not an estimate of it.
- **Decision quantity:** `Delta_rho,s = AUC_F(L+G) - AUC_F(L)`, one interval per row and seed.

## 6. Decision per row (fixed now)

- **INFO-UPSTREAM** iff the lower CI of `Delta` is > +0.02 on BOTH seeds.
- **NO-JUMP** iff the upper CI of `Delta` is < +0.01 on BOTH seeds.
- Otherwise **INCONCLUSIVE** (two seeds that disagree are INCONCLUSIVE).
- G-only's F-AUC is reported beside it; no branch reads it.
- **Conversions (they can only REMOVE a call, never create INFO-UPSTREAM):**
  1. **VOID** a row if the powered null fires on >= 2 of 5 replicates (section 7, C-NULL).
  2. **VOID-INPUTS** a row if C-AGREE fails (section 7).
  3. A NO-JUMP call is converted to **UNDERPOWERED** (read as INCONCLUSIVE) if the weak planted signal does NOT fire the INFO-UPSTREAM rule on both seeds (C-POWER): an instrument that cannot see a planted effect cannot call "no jump".
  4. No row is called at all if C-REACH, C-JOIN, C-ID-L or the RED mutation table fails (gate FAIL).

## 7. Controls (each is built so it can fail; every one is reported)

- **C-REACH (planted scalar equal to the label, on the CODE path):** the SAME fit-and-score functions the real arms use, with `G := the row's label` as a single column, must give F-AUC >= 0.99 for L+G AND for G-only on BOTH seeds. Run on synthetic arrays in the tests AND on the real subset arrays in the gate. Never stated as a property of the real data.
- **C-POWER (weak planted signal):** `G := label + 1.0 * N(0, 1)` (`numpy.random.default_rng(20261019 + 31 + col).standard_normal`, `col` = 0 for NC, 1 for DAC), a single column; the registered INFO-UPSTREAM rule on `AUC_F(L+G_plant) - AUC_F(L)` must FIRE on both seeds (known partial information, a scalar-alone AUC of about 0.76).
- **C-NULL (powered, `RETRACTION_LOG` R51):** the row's G columns are permuted across the 64 proposals WITHIN each set (one permutation per set moves the whole K-vector; token and label untouched); 5 replicates, permutation seed `20261019 + 7919 * rep`, L+G refit for both seeds. A replicate FIRES for a row iff the INFO-UPSTREAM rule holds on BOTH seeds on its `AUC_F(L+G_perm) - AUC_F(L)`. **A row is VOID iff >= 2 of 5 replicates fire.** False-fire rate of the interval rule under a TRUE null, ESTIMATED analytically before registering: if SE(Delta) = s, one replicate fires with probability Phi(-(0.02 / s) - 1.96), i.e. 4e-5 at s = 0.01 and 0.0026 at s = 0.02, so a fire on >= 2 of 5 is a systematic leak or bias and not noise; the empirical false-fire rate on synthetic noise blocks is measured in `test_pg_lib.py` and reported with the result.
- **C-AGREE (provenance of the geometry):** the recomputed NAVSIM verdicts (`rl6`, clipped to [0, 1] as the trainer's `cg`) equal the stored `labels6` on the NC and the DAC columns for >= 99 % of proposals EACH; otherwise the geometry was not computed on the proposals and scenes the labels belong to and the row is VOID-INPUTS. Also reported, with no gate: the label-identity rates of section 3.4 (`P(dac_frac_out > 0 | DAC = 0)`, `P(dac_frac_out = 0 | DAC = 1)`, `P(nc_min_dist = 0 | NC = 0)`).
- **C-JOIN:** the extraction's keys (log, token, step, rank, ckpt_step) match the census bank 1:1 on every retained set; the census `z6` equals the pickrule package's `z6` on those sets (the EP-R1 columns are keyed to the same rows). Sets lost in extraction are COUNTED and listed; a loss above 5 % of the subset's sets is disclosed in the headline.
- **C-ID-L:** the L arm of the new fit code equals the landed `fastneg_lib.fit_rows` + `score_rows` on identical inputs (allclose 1e-5) on a synthetic problem.
- **C-RED:** a branch table over `info_call`, `no_jump_call`, `row_call`, `null_void`, `null_fires`, `power_ok` with RED mutations (comparator flipped, bound swapped, seed `all` -> `any`, a threshold changed) that must each make the table FAIL.
- **C-CONV:** on a synthetic problem with a known logit signal the L+G arm's F-AUC is within 0.01 of the oracle's.

## 8. The pick-level read (DESCRIPTIVE, no authority over any branch)

On the subset's sets, per training seed, and in the arrays of `PREREG_PICKRULE_AUDIT_NC`: R-P2 = the stored `z6`; the EP-led composite R-EP = R-P2 with column 2 (EP) replaced by EP-R1's c* OOF EP column (`ep_columns.npz` of the pickrule package, sha256 `9c16628594fa1286c4efd029b3766fed44a9198700aa0098058ec95eea5a7d49`, seed-matched); then **R-EP[L]** and **R-EP[L+G]** = R-EP with columns 0 and 1 replaced by the arm's OOF NC / DAC logits (and the NC-only / DAC-only replacements). Picks by `pickrule_lib.rule_pick` (float64, `planner.py:558-561`); NC-unsafe, DAC-unsafe and "unsafe" (EP-R1's) pick rates and label-lookup PDMS x 100 for P2 and each composite, with paired log-cluster differences to P2 and `R-EP[L+G] - R-EP`, `R-EP[L+G] - R-EP[L]`. Tier: label-lookup, SEEN, P2 in-sample (a bias AGAINST every out-of-fold composite).

## 9. Committed next arms (by outcome; fixed now, as briefed)

| outcome | next arm |
|---|---|
| INFO-UPSTREAM on DAC | a scorer-branch geometry read from cached `visual_ctx` (lever SH5; a dev-box GPU registration) |
| INFO-UPSTREAM on NC | an agent / lead-aware scorer read (SH3) |
| NO-JUMP (either row) | a label-learnability audit of the fast band, and the trunk (P6) |
| INCONCLUSIVE / UNDERPOWERED / VOID | named by the report; the 76 logs without a local DB are the first lever (section 11) |

Reading note carried with the table (section 3.4): an INFO-UPSTREAM call is read together with its companion variant before a GPU arm is registered.

## 10. Forecast (a prediction, not a bar; HYPOTHESIS, committed before any number)

DAC primary: **INFO-UPSTREAM, about 0.95** (tautology, section 3.4); DAC-nosign: INFO-UPSTREAM 0.45, INCONCLUSIVE 0.35, NO-JUMP 0.20. NC primary: INFO-UPSTREAM 0.65 (a contact event plus approach geometry), INCONCLUSIVE 0.25, NO-JUMP 0.10; NC-floored: INFO-UPSTREAM 0.30, INCONCLUSIVE 0.40, NO-JUMP 0.30. The pick-level read: I expect R-EP[L+G] to cut the unsafe-pick rate below P2's because the contact test is inside G.

## 11. Limits stated in advance

- 41 of 117 logs (31.3 % of the sets); file-presence filter, not random; **the lever to widen is the pod's DBs** (`fetch_navtrain_dbs.py`), a decision outside this arm.
- SELECTION tier, SEEN scenes, P2 in-sample; one rig, two refit seeds; the interval answers "another draw of LOGS" only; with 41 logs and 3 logs in one fold the intervals are wider than the census's.
- The lead is a straight-corridor approximation; agent velocity is a finite difference of an interpolated track; the geometry is on NAVSIM's simulated (LQR + bicycle) states, not the raw plan.
- The primary G contains the label's own test (section 3.4). A linear row on standardised scalars cannot reproduce a sign test exactly, and the 20-epoch recipe may not converge to the AUC-optimal weighting: the planted controls bound that, nothing else does.
- Evidence class for every number this arm produces: MEASURED, SELECTION tier, T0-class label lookup; no capability claim.

## 12. Code pins (the extraction tier; the fit / analysis code is pinned by Amendment 1)

| file | sha256 |
|---|---|
| `code/pg_geom.py` | `5b78a694f40b6b2b4030fdeb3bf5be2c65b349c8887f9c11c78d0dfd8d9df7de` |
| `code/pg_extract.py` | `64a148906329f8aeba5dc3d87d30aba9f2e91d90389ea77c0169614e1bade23b` |
| `code/pg_select.py` | `51f85a63e5ffcfe4f4355004da17fb2e9fc3f808609532b2d3b01137657e03d6` |
| `code/test_pg_geom.py` | `3cf48766adc17b4c1ac07bb1b043e3798ffa2a680226196f124fc69f56406b64` |
| `ref/fastneg_lib.py` (census, landed) | `0c0b399b37a47ee3e17f9d5484bc7f9521ff7dbe6bea37e8c7a38042668667ca` |
| `ref/pickrule_lib.py` (landed) | `555c0a3468acd94adeeeb10f47527644a2b9d49722f554fed2da1fb29668eaeb` |
| `ref/epr1.py` (landed) | `a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8` |

Environment: `C:/Users/Admin/venvs/driverl-eval`, NAVSIM `3e8291bfa89ff247231e0227778840cd0a036896` (`D:/Archive/devbox-C/navsim/navsim-3e8291bf...`), landed `navsim_pdm_targets.py` / `navtrain_scenarios.py` of the p2heldout tree.
