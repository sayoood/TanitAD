# PREREG_V8F_SOUP: a same-init weight average ("soup") of O fine-tunes, read at T1-kin against DAG-OACC and against P10

**Status: DRAFT for the Master Mind, NOT RUN (2026-10-28).** Zero training. Dev-box CPU only (`CUDA_VISIBLE_DEVICES=-1`). No soup checkpoint has been rolled out or evaluated; no T0 or T1 number of any soup exists. Proposed claim row `H-V8F-SOUP-1` (to be registered at promotion).
Package: `E:/Projects/TanitAD-artifacts/dagland/soup/` (this file, `soup_build.py`, `read/`, `tools/`, `argv/`, `evidence/`).

**Disclosure of what was read before writing** (all from the git mirror at `d7dc9f11` unless a path says otherwise):
* `Project Steering/PREREG_V8F_LAUNCH_INTENT.md` (the model for this file), `Project Steering/PREREG_V8F_DAG_REREAD.md` §2 and §5 (Read A).
* The banked LAUNCH-INTENT read of record `raw/2026-10-28-launch-intent-read/read_A1/LAUNCH_INTENT_READ.json` (sha256 of the git blob in `evidence/inputs_sha256.txt`): its outcome, its `read_a` tables (A − CTRL, Arep − REP, CTRL − P10, REP − P10, REP − CTRL, A − P10, Arep − P10), its floors and its controls.
* `E:/Projects/TanitAD-artifacts/dagland/launch_intent_mlaunch/MLAUNCH.json` (sha256 `f5ebb4b2…`): M-LAUNCH and its training-seed floor.
* The reader `raw/2026-10-28-launch-intent-amend1/read/launch_intent_read.py` (sha256 `fb3d2d7d…`), the T1-kin chain `E:/Projects/TanitAD-artifacts/launch_intent_t1/` (`NOTES.md`, `LAUNCH.txt`, `argv/*.tmpl`, `logs/chain.out`) and the tool tree `F:/TanitAD-scratch/launch_intent_t1/tree_li` (only `v8f_t1_eval.py:505-516` and `v8f_mini_eval.py:236-268`, the checkpoint loaders).
* The checkpoint STRUCTURE of DAG-OACC_0, REP_0, A_0 and M2b_0 (top-level keys, state-dict keys, dtypes, cfg keys; `evidence/structure.txt`). ⚠ **No weight distance and no weight statistic was computed before §6 (K-BASIN) was written.** The pre-measurement text of this file is banked as `evidence/PREREG_pre_measure.md` with its sha256 and time in `evidence/pre_measure_stamp.txt`.
* **Not read:** any soup row (none exists).

## 0. Items the Master Mind fixes before promotion

| # | item | state |
|---|---|---|
| 1 | **The replicate pair's control is A_s, not DAG-OACC_s** (§2). The brief names DAG-OACC_s as the control; this draft reads the main soup against DAG-OACC_s and the replicate soup against its own seed-s member A_s, so that each pair differs in the soup lever alone. SOUP'_s − DAG-OACC_s is reported beside. | CONFIRMED (Master Mind, 2026-10-28): the replicate pair's control is A_s |
| 2 | **SOUP'_s carries the launch-intent lever** (`--launch-intent-w 1.58926`) on both members. Disclosed: that lever read HARMS (`H-V8F-LAUNCH-INTENT-1`) and M-LAUNCH NOT MOVED. Inside the replicate pair both arms carry it, so it is not a difference of the pair. | DISCLOSED |
| 3 | **The main soup's s0 partner is a known slow-T draw.** REP_0 − DAG-OACC_0 is separated-worse on ADE6 by +0.4025 m (MEASURED, banked read). The first-order (linear-interpolation) prediction for SOUP_0 − DAG-OACC_0 on ADE6 is therefore about +0.20 m, worse (ESTIMATED; a soup gain larger than that is what the lever would have to deliver). R-ADE at s0 is the likely failing guard. This draft keeps the member set fixed (no selection on eval-141) and routes that outcome in §10. | CONFIRMED (Master Mind, 2026-10-28): the fixed member set is ACCEPTED; no selection on eval-141 |
| 4 | **The soup's T0 evals run on the dev-box CPU**, while every member's standard T0 eval ran on box 2. R-T0 is read same-device (§5, K-T0DEV). | CONFIRMED (Master Mind, 2026-10-28): R-T0 is read from the dev-box T0 of both arms (K-T0DEV) |
| 5 | **Banking the primary** (`kb_add.py 2203.05482`) was blocked in this package: it downloads a file and writes into `TanitAD Research Lab/Library/`, outside the work dir. Command for the lander: §14. Until banked, every literature statement here is PUBLISHED-SECONDARY and decides nothing. | BLOCKED, named: the banking stays blocked here; the Master Mind banks it later, and until then literature statements decide nothing |

## 1. Why it runs

* **The largest measured effect on the crawl rows is the training draw, not any lever tried.** REP_s and DAG-OACC_s share init M2b_s, the iteration-1 bank, 3,000 steps and the argv apart from `--out` and `--seed` (MEASURED, `compare_rep_argv.out`, INHERITED from `PREREG_V8F_LAUNCH_INTENT` §4). At s0 their T1-kin difference REP − DAG-OACC is (MEASURED, banked read, paired episode-cluster bootstrap, B = 2,000, 141 episodes / 2,399 windows, T1-kin):

| row | s0 | s1 |
|---|---|---|
| ADE6 (m) | **+0.4025 [+0.2761, +0.5343]** | +0.0116 [−0.0531, +0.0765] |
| yaw-rate MAE | **+15.8553 [+11.9088, +20.2953]** | +0.8903 [−0.5695, +2.5371] |
| curvature MAE | **+0.1324 [+0.0904, +0.1795]** | −0.0008 [−0.0335, +0.0292] |
| L2 curvature MAE (GT ≥ 5.2) | **+0.0870 [+0.0503, +0.1344]** | +0.0060 [−0.0092, +0.0216] |
| M1 crawl share | **+0.0388 [+0.0270, +0.0519]** | +0.0048 [−0.0019, +0.0117] |

  M-LAUNCH's training-seed floor, REP − DAG-OACC at s0, is **−0.556 [−0.699, −0.398] m/s** (MEASURED, `MLAUNCH.json`, `beside_training_seed_floor`; 327 windows / 49 episodes).
* **The levers tried did not move the crawl.** LAUNCH-INTENT read **HARMS** (MEASURED, `LAUNCH_INTENT_READ.json`): R-MOVE false at both seeds; R-ADE failed at both seeds; its separated ADE6 cells flip sign between the two same-seed one-lever pairs (A − CTRL s0 −0.0294 / s1 +0.0483 m; A_rep − REP s0 +0.0431 / s1 −0.0341 m). M-LAUNCH: **not moved** (MEASURED, `MLAUNCH.json`).
* **The crawl rows against P10** (MEASURED, banked read, T1-kin): DAG-OACC − P10 is separated-worse at both seeds on yaw-rate MAE (+3.4726 / +8.6844), curvature MAE (+0.0854 / +0.1352), L2 (+0.0443 / +0.0529) and M1 (+0.0568 / +0.0604), while ADE6 is −1.2012 / −1.1231 m, separated-better. That is Read A's D2 failing on the crawl rows alone.
* **The lever.** Averaging the weights of fine-tunes that share one init is the textbook variance reducer for a training draw: Wortsman et al. 2022, *Model soups* (arXiv 2203.05482, PUBLISHED-SECONDARY: primary not banked, §0 item 5). The mechanism it rests on is that fine-tunes of one init stay in one linearly connected low-loss region; across different random inits the average is generally broken (Frankle et al. 2020, arXiv 1912.05671, PUBLISHED-SECONDARY). HYPOTHESIS here: part of the crawl-row error of one draw is draw-specific, and an average of two draws keeps the shared part and halves the draw-specific part.
* **Cost.** No training, no box 2. A soup is a weights-only change: same architecture, same cfg, same inputs; inference stays vision plus measured ego history (§2).

## 2. The lever, exactly (built and tested, not rolled out)

**The soup.** For members θ_1 … θ_N (N = 2 or 4) with one shared init: θ_soup[k] = (1/N) Σ_i θ_i[k] for every float tensor k of `ckpt["model"]`, computed in float64 and rounded once to the members' dtype (float32). `soup_build.py` refuses unless every member has the identical key list, shapes and dtypes, every non-float tensor is equal (there are none: MEASURED, 276 float32 tensors, 7,233,343 elements, `evidence/structure.txt`), and every declared must-equal non-weight entry is equal (§2, cfg rule). It records the members' sha256 (re-hashed before and after loading), the output sha256, the builder's sha256 and the K-BASIN numbers in `soup.json` and in `ckpt["soup"]`.

**The checkpoint the T1 tool loads.** Both tools read `ck["cfg"]`, `ck["model"]` and `ck.get("grid")` only (MEASURED: `v8f_t1_eval.py:505-516`, `v8f_mini_eval.py:236-268` in `tree_li`). The soup checkpoint is the **donor member's checkpoint with `model` replaced by the average and `opt` set to `None`** (an optimizer state belongs to one training run; a soup has none). Every other top-level entry is the donor's, verbatim, and each member's own values of those entries are recorded under `ckpt["soup"]["members"]`.
* **The donor is the member the soup is read against**: DAG-OACC_s for SOUP_s and SOUP4_s, A_s for SOUP'_s. So a soup and its control differ in `model` alone (the one-lever form of K-PAIR).
* **cfg rule.** All members' cfg dicts must be equal on every key except the declared training-only allowlist `{launch_intent_w}`, else the build REFUSES. `launch_intent_w` enters only O's training loss (`v8f_mini.py:908`, `:926`; `PREREG_V8F_LAUNCH_INTENT` §2; the chain's AST `modelproof` found no change on the inference path; INHERITED). MEASURED: DAG-OACC_0 and REP_0 cfg are identical; A_0 adds `launch_intent_w` and is otherwise identical.
* **Must-equal non-weight entries:** `cfg` (minus the allowlist), `cfg_tuple_fields`, `grid`, `arm`, `p10_path`, `cons_mu`, `cons_sd`. Member-specific, recorded per member: `step`, `epoch`, `batch_in_epoch`, `argv_sha`, `seed`, `init_ckpt`, `ego_perturb`, `ego_totals`, `dagger`, `dagger_totals`, `launch_totals`.
* **Inference is untouched.** The soup changes no input, no head and no planner setting. It is not a goal input and carries nothing from the situation classifier.

## 3. The arms (zero training; every member banked)

| arm | members (init M2b_s, iteration-1 bank, 3,000 steps) | donor | role |
|---|---|---|---|
| **SOUP_s** | DAG-OACC_s (trainer seed s), REP_s (seed 10 + s) | DAG-OACC_s | **decisive**, main pair |
| **SOUP'_s** | A_s (seed s), A_rep,s (seed 10 + s); both `--launch-intent-w 1.58926` | A_s | **decisive**, replicate pair |
| **SOUP4_s** | all four | DAG-OACC_s | beside, decides nothing (its members overlap both pairs) |
| XINIT | DAG-OACC_0 with DAG-OACC_1 (declared at s0), and DAG-OACC_1 with DAG-OACC_0 (declared at s1) | — | **deliberate regression on weights**: must FAIL K-BASIN, never rolled out |

Member checkpoints (sha256, MEASURED by re-hash; paths in `soups.json`): DAG-OACC `67265ec1…` / `90b01474…`; REP `b03fd6ef…` / `77853e56…`; A `deb8f28b…` / `abe0fd24…`; A_rep `b86f7d43…` / `2900ff73…`; inits M2b `601cb435…` / `6104fb15…`.

**The pairs, at each seed s** (the two-pair standard of `PREREG_V8F_LAUNCH_INTENT` §10):
* **main:** X = SOUP_s, Y = DAG-OACC_s. The lever is "average the seed-s draw with a second draw of the same recipe".
* **replicate:** X = SOUP'_s, Y = A_s. The same lever on disjoint members (the A recipe). Its members carry a lever with no measured effect (§0 item 2), on both sides of the pair.
* **Read A (the headline):** X − P10 for X = SOUP_s and X = SOUP'_s (P10 = the `fl` arm of every dump, the programme's DOMINANT test, `PREREG_V8F_DAG_REREAD` §2 and §5).
* **Beside:** X − (the other member) (SOUP_s − REP_s, SOUP'_s − A_rep,s); SOUP'_s − DAG-OACC_s; SOUP4_s − DAG-OACC_s and SOUP4_s − P10; the soup gain SG (§7).

## 4. Members' records (inherited, re-checked)

The four member runs are the four arms of the LAUNCH-INTENT read of record; its controls all PASS there (MEASURED, `controls_PASS`, 206 entries, all true): K-INIT, K-PAIR, K-SHARE, K-MASK, K-REP-LICENCE, K-BUILD, K-GATE, K1–K7, C3, C4. The soup reader re-runs those record controls on the same manifest with the LAUNCH-INTENT reader's own functions (imported unchanged, sha256 pinned) before any soup row is read (K-MEMBERS, §9).

## 5. Evaluation and tiers

| where | what | tier |
|---|---|---|
| dev-box CPU (decisive) | `v8f_t1_eval.py run --replan-steps 10 --limit causal --dump-ticks --dump-heads`, `--nav deadreckon --scene logged --schedule periodic`, on eval-141; the LAUNCH-INTENT chain's `argv/rollout_causal.tmpl` token for token, `--ckpt` / `--tag` / output paths the only changes; tool tree `tree_li` (digest `579183a3…`). SOUP_s and SOUP'_s at both seeds (4 rollouts); SOUP4_s beside (2) | **T1-kin**: self-action OPEN loop. ⛔ Never "closed loop" |
| dev-box CPU (K4 only) | one `--limit logged` rollout and one `--ages fresh` T0 eval per soup (`rollout_logged.tmpl`, `t0fresh.tmpl`) | control |
| dev-box CPU (R-T0) | the standard T0 eval (`argv/t0std.tmpl` = `t0fresh.tmpl` without `--ages fresh`; the default is `cycle`, as box 2's `evaldag` line runs it) of every soup **and of each pair's control** (DAG-OACC_s, A_s), so R-T0 compares two dev-box evals | **T0**: teacher-forced, own code. ⛔ Never "driving performance" |

**Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed and per pair, paired, same windows and matched cells. **Three variances, named:** another draw of EPISODES (the bootstrap answers it); another TRAINING RUN (the replicate pair at each seed answers it, and the two seeds add two init draws; a soup has two training draws inside it, and a pair is credited at a seed only when both pairs show it); another INFERENCE RUN (none arises: the planner is deterministic and a soup is a fixed set of weights).

## 6. Weight-space controls, declared before any weight distance was computed

All on `ckpt["model"]`'s float tensors, flattened in key order, in float64. θ_s is the declared init M2b_s (sha256 pinned); θ_s' is the other seed's init. ‖·‖ is the L2 norm.

* **K-BASIN (the weight-space check; a soup that fails it is REFUSED at build and never rolled out).**
  * r_i = ‖θ_i − θ_s‖ (member i's displacement from the declared init), r'_i = ‖θ_i − θ_s'‖, d_ij = ‖θ_i − θ_j‖.
  * **The same-init distance** R_ft(s) = the median of r_i over the four banked fine-tunes of M2b_s (DAG-OACC_s, REP_s, A_s, A_rep,s). A median of four is moved by one foreign member by at most one rank.
  * **B1, init attribution, per member:** r_i ≤ r'_i / 2 (each member is at least twice as close to its declared init as to the other seed's init).
  * **B2, member spread against the same-init distance:** max_{i<j} d_ij ≤ 3 · R_ft(s). Why 3: two fine-tunes of one init whose displacements are at most 1.5 R_ft are at most 3 R_ft apart (triangle inequality, d_ij ≤ r_i + r_j). A pair farther apart cannot both be fine-tunes of θ_s at the reference scale. The factor is analytic and fixed now.
  * PASS iff B1 holds for every member and B2 holds.
* **K-XINIT (the deliberate regression).** The cross-init soups of §3 must FAIL K-BASIN, and each clause must be load-bearing: XINIT fails B2 with B1 switched off, and fails B1 with B2 switched off. If XINIT passes, K-BASIN does not discriminate a broken average and the read is REFUSED.
* **K-ZERO-W.** A soup of a checkpoint with itself (N = 2, 3 and 4 copies) equals it bit for bit on every tensor (`torch.equal`, same dtype), and every non-weight entry equals the donor's. Exactness is by construction: N equal float32 values sum exactly in float64 and the correctly rounded quotient by N is the value itself.
* **K-MID (per soup).** The written tensors equal an independent numpy float64 mean rounded to float32 with max |Δ| = 0.0; every element lies between its members' minimum and maximum; for a two-member soup |d(SOUP, a) − d(a, b)/2| ≤ 1e-6 · d(a, b) and the same for b.
* **K-SOUP-REC (per soup).** `soup.json` re-verified by content: the output file hashes to its record; every member file hashes to the manifest pin above; the donor is the pair's control; `cfg` is byte-equal (JSON-canonical) to the donor's; `opt` is `None`; the builder file hashes to the recorded builder sha256.

## 7. The bar, the rows and the soup gain

**The bar: `2 × max(F_r, RF0_r)` per row, reused unchanged** from `PREREG_V8F_LAUNCH_INTENT` §6 (F = 0.1207 m on ADE6, 0 elsewhere; RF0 = the frozen `P0_RF0.json`, sha256 `1608d70b…`; the 19 values as banked in `LAUNCH_INTENT_READ.json` `floors.bar`). Why reuse it: same rig, same 19 rows, same estimator, same windows, and it is reachable on every crawl row (each crawl bar is below the CTRL − P10 gap at both seeds, `PREREG_V8F_LAUNCH_INTENT` §6). **RF** (max over seeds of |REP − DAG-OACC|) is reported beside every row and decides nothing; ⚠ here RF is also the main soup's own member spread, which is one more reason it cannot be a bar term.

**Rows, in each pair** (X the pair's soup, Y its control; `PREREG_V8F_LAUNCH_INTENT` §7 verbatim with these X and Y):

| row | contrast, in each pair | role |
|---|---|---|
| **R-CRAWL** | every crawl row (yaw-rate MAE, curvature MAE, L2, M1) is not separated-worse for X − P10 | decisive |
| **R-MOVE** | every crawl row clears for X − Y: separated-better AND improvement ≥ bar_r | decisive |
| **R-TLAT / R-TLON** | T-head lat / lon macro-F1 at ticks ≥ 1, X − Y: not separated-worse | guard |
| **R-ADE** | T1-kin ADE6, X − Y: not separated-worse | guard |
| **R-T0** | T0 own-code ADE6 (dev-box, both arms), X − Y: point ≤ +F/2 = **0.0604 m** | guard |
| **R-NEW** | no row of the pair's domain D fires (D = every row not separated-worse for Y − P10) | guard |
| **R-KEEP** | T1-kin ADE6, X − Y: point ≤ +0.0604 m | adoption |
| **R-NOBASE** | no row of D has X − P10 separated-worse AND X − Y separated-worse with worsening ≥ bar_r | adoption |
| **Read A** | X − P10, `PREREG_V8F_DAG_REREAD` §2: D1 ADE6 separated-better with point ≤ −2F = −0.2415 m; D2 no row of the 19 separated-worse; D3 each family has a separated-better row | **headline** |

**The soup gain SG (the mechanism row; it routes §11 and decides no outcome).** For a soup X with members m_1 … m_N, per window w: sg_w = ADE6_X,w − (1/N) Σ_i ADE6_{m_i},w (T1-kin, causal lane, the rig's own `comps` on each dump), paired episode-cluster bootstrap. SG < 0 separated: the weight average beats the average of its members' errors. SG > 0 separated: a **barrier** (the average is worse than its members' mean). Reported per soup, per seed; the same on the four crawl rows where a per-window array exists, and NOT BUILT with its reason where it does not.

**Beside (never decisive):** RF per row; the slow-draw flag of each soup against its pair's control (the method of `rep_s0_speedcells.py`, as `PREREG_V8F_LAUNCH_INTENT` §7, on the dev-box standard T0); the dev-box − box-2 T0 ADE6 of each control (K-T0DEV); SOUP4_s on every row; the anti-echo block (NOT BUILT in the rig, stated per family).

## 8. The four metric families (all reported, per seed, per soup, never pooled)

`PREREG_V8F_LAUNCH_INTENT` §8 applies verbatim, for each pair and for Read A. **STRATEGIC is not paused for v8F** (PI ruling 2026-10-17). T1-kin carries LONGITUDINAL (speed, along-track, accel MAE, DK-VIOL / DK-HW / DK-TG), LATERAL (heading, yaw-rate, curvature, cross-track, L1 lateral-accel, L2), TACTICAL (TAC-LAT / TAC-LON; R-TLAT / R-TLON), STRATEGIC (S-POS@30 / 50 / 80 against CTRV within a run). T0 carries the families of `--families` on the dev-box standard eval. A family that cannot be computed is listed per family with its reason and n. **The crawl rows are LATERAL plus M1, so the LATERAL family is read first.**

## 9. Controls (each must pass, or the read is REFUSED)

| control | what |
|---|---|
| K-BASIN, K-XINIT, K-ZERO-W, K-MID, K-SOUP-REC | §6, per soup; re-measured by the reader from the member files, not trusted from `soup.json` |
| K-MEMBERS | the LAUNCH-INTENT reader's record controls (K-INIT, K-PAIR, K-SHARE, K-MASK, K-REP-LICENCE, K-BUILD, K-GATE) on the four member roles, same manifest, all PASS |
| K-RC | before any soup row: the soup reader reproduces the banked `LAUNCH_INTENT_READ.json` tables A − CTRL, Arep − REP, CTRL − P10, REP − P10, REP − CTRL, A − P10, Arep − P10 at both seeds with 0 differences, and its RF equals the banked `floors.RF` |
| K-ZERO | (X, X) reads exactly +0 [0, 0] on every row for every soup and control; a reader with the bar's RF0 term deleted fails its self-test |
| C3, C4 | every soup dump has CTRL's windows (clip_index, ws, eid, g, v0 bit-equal; 2,399 / 141) and P10's tracks bit-equal |
| K4, T1-CONFIG, GRID, DUMP-WINDOWS, R2-ADE6 | the DAG-ORC rig's per-run pipeline controls, on every soup, as for a new arm |
| K-T0DEV | R-T0 reads dev-box T0 on both arms of the pair; the dev − box-2 ADE6 of each control is reported beside |
| K-TREE | every soup rollout ran in `tree_li` (digest `579183a3…`); the soup's cfg builds `V8FMiniConfig` there |
| K-INPUT | each soup's cfg equals its control's (byte-equal), so the inference inputs are the control's |
| PREREG-TEXT | the reader refuses unless this file's sha256 equals the pin it is given |

## 10. Outcomes, committed now

**Per-seed booleans.** Each holds at seed s only when it holds in **both pairs** of s (main: SOUP_s − DAG-OACC_s; replicate: SOUP'_s − A_s):
* **H_s:** R-TLAT, R-TLON, R-ADE, R-T0 and R-NEW all hold.
* **MV_s:** R-MOVE holds.
* **AD_s:** R-KEEP and R-NOBASE hold.
* **DM_s:** Read A's D1, D2 and D3 hold at s for SOUP_s − P10 **and** for SOUP'_s − P10.

| outcome | condition | committed next |
|---|---|---|
| **REFUSED** | any control fails | fix and re-read; no row is reported |
| **HARMS** | controls pass; H fails at ≥ 1 seed | the uniform two-draw soup is not adopted. **NEXT-SOUP** (§11) |
| **DOMINANT / ADOPTED** | H at both seeds; DM at both seeds; AD at both seeds | adopt SOUP_s as the base; register *"at T1-kin, the two-draw same-init soup of DAG-OACC is no worse than P10 on any row of the four families, better on each family, and ≥ 2F better on ADE6, at both seeds, replicated on disjoint members"*; re-read `PREREG_V8F_DAG_REREAD` Read B-2 on SOUP (banked FLAT dumps, zero training); signal the PI |
| **DOMINANT / NOT ADOPTED** | H at both seeds; DM at both seeds; AD fails at ≥ 1 seed | register the dominance claim on SOUP_s; the base question (SOUP_s trades the R-KEEP point against DAG-OACC) goes to the PI; meanwhile **SOUP-K** (§11) |
| **MOVED / ADOPTED** | H at both seeds; DM fails at ≥ 1 seed; MV at both seeds; AD at both seeds | adopt SOUP_s as the base; the rows that still fail D2 go to **SOUP-K** on the pure recipe |
| **MOVED / NOT ADOPTED** | H at both seeds; DM fails at ≥ 1 seed; MV at both seeds; AD fails at ≥ 1 seed | the lever is attributed but SOUP_s cannot replace DAG-OACC. **NEXT-SOUP** (§11) |
| **ONE SEED** | H at both seeds; DM fails at ≥ 1 seed; MV at exactly one seed | not adopted, never called an effect of the lever (`H-ESTIM-SEED-1`). **NEXT-SOUP** (§11) |
| **NO EFFECT** | H at both seeds; DM fails at ≥ 1 seed; MV at neither seed | averaging two draws does not move the crawl at this N. **NEXT-SOUP** (§11) |

**Precedence.** REFUSED, then HARMS. With the guards held, DM splits DOMINANT from the rest; among the rest MV partitions both seeds / one seed / neither; adoption is decided last. Completeness and exclusivity are proved by `tools/enumerate_cells_soup.py` over every assignment of the 9 per-seed composed booleans (512 cells) and over 200,000 seeded random pair-level assignments (`evidence/enumerate_cells_soup.out`). Read A is also reported per soup (SOUP_s, SOUP'_s, SOUP4_s) with its own `PREREG_V8F_DAG_REREAD` §5 label; a single soup's DOMINANT that the other decisive soup does not share is reported as **not replicated** and registers nothing.

## 11. NEXT-SOUP: what follows a non-adopting outcome, registered now (Rule Zero)

The routing variable is **SG on ADE6** (§7), read on SOUP_s and SOUP'_s at both seeds: **BARRIER** = SG separated-worse (> 0) in at least one of the four soup-seed cells; **FLAT** otherwise.

| SG | what it says | committed next |
|---|---|---|
| **BARRIER** | the members are not linearly connected at this scale: the weight average is worse than the mean of its members' errors | **ENS2**: the output ensemble. The two members' plans averaged knot by knot at every replan inside the T1-kin rollout (zero training; a `v8f_t1_eval.py` change that loads two checkpoints, under its own PREREG with its own K-ZERO: an ensemble of a model with itself reproduces the model's dump bit for bit). It asks the soup's question without the linear-connectivity assumption |
| **FLAT** | the weights average at least as well as the metrics; the harm or the shortfall is the members' draws | **SOUP-K**: K = 4 pure-recipe draws per seed (DAG-OACC_s's argv at trainer seeds 20 + s and 30 + s added to DAG-OACC_s and REP_s; 4 new box-2 runs, ≈ 2.8 h ESTIMATED from the LAUNCH-INTENT runs; its own PI go), read under this table with SOUP-K_s in place of SOUP_s |

**If both are exhausted**, the crawl rows are not a training-draw variance that averaging removes. The blockers, by name: **LOOP-STATE** (zero training, registered in `PREREG_V8F_LAUNCH_INTENT` §11: decompose the crawl by plan index in the tick dumps) names the loop term; a PI decision on an architecture change that gives O a launch-intent input. Neither is launched without that read or that decision.

## 12. What this read cannot claim

* **It is not closed loop.** T1-kin is a self-action open loop.
* **A soup effect is the effect of averaging these two draws.** N = 2 per decisive soup; nothing here says a soup of K > 2 behaves the same (SOUP4 is beside and its members overlap both pairs).
* **The control is asymmetric by design.** Each soup is read against its trainer-seed-s member (the incumbent's draw), fixed before data; against the other member it is reported beside. A soup that is better than REP_s and worse than DAG-OACC_s is not an improvement of the base.
* **The replicate soup carries the launch-intent lever.** Its result is the soup lever on the A recipe; it attributes nothing to the launch weight.
* **R-T0 is a dev-box CPU T0 reading** on both arms; it is not box 2's standard eval.
* ⛔ No bar, row, member set, donor rule, K-BASIN factor, routing variable or outcome changes after any soup row is seen.

## 13. Queue, cost and launch preconditions

| item | MEASURED basis | ESTIMATED |
|---|---|---|
| build 6 soups + 2 XINIT checks | MEASURED: 3.9–4.8 s per soup, 36.9 s for six; `basin` 14.3 s (`evidence/build.out`, `evidence/basin.out`) | done |
| T1-kin causal rollout | LAUNCH-INTENT chain log: 178–222 s per rollout (`launch_intent_t1/logs/chain.out`) | ≈ 3.7 min each |
| T1-kin logged rollout (K4) | same log: 178–218 s | ≈ 3.7 min each |
| T0 eval (fresh or standard) | same log: fresh T0 50–88 s | ≈ 1.5 min each |
| **decisive (SOUP_s, SOUP'_s; 4 soups)** | 4 × (causal + logged + fresh T0 + standard T0) | **≈ 42 min** |
| controls' dev-box standard T0 (DAG-OACC_s, A_s) | 4 × T0 | ≈ 6 min |
| SOUP4_s beside (2 soups) | 2 × (causal + logged + fresh T0 + standard T0) | ≈ 21 min |
| reader | the LAUNCH-INTENT read of record: 341.6 s on 8 runs (`provenance.wall_s`) | ≈ 8–10 min on 10 runs |
| **total dev-box CPU, one torch process at a time** | | **≈ 80 min**; dumps ≈ 4 GB on F: (LAUNCH-INTENT: ≈ 0.33 GB per decisive rollout, INHERITED); soup checkpoints ≈ 29 MB each on E: |

**Launch preconditions.**
1. This PREREG is promoted, and its sha256 is pinned in the reader call.
2. Every soup is built by `soup_build.py build` and passes K-BASIN, K-ZERO-W, K-MID and K-SOUP-REC; K-XINIT fails as required (`evidence/basin.json`).
3. The rollout chain (§13a) exists, with the LAUNCH-INTENT chain's guards (RAM ≥ 4 GB, F: ≥ 5 GB, checkpoint and tool-tree re-hash before and after every step, done records, one process at a time). It is an integration step and not part of this package.
4. The soup reader's heavy layer passes K-RC on the banked LAUNCH-INTENT dumps before it reads a soup.

**§13a. The chain adaptation, exactly** (from `E:/Projects/TanitAD-artifacts/launch_intent_t1/`):
* **Readiness** is the soup's `soup.json` with K-BASIN PASS and the output file's sha256, not a box-2 pull line; `lit1_lib.py runrecord_problems` is replaced by K-SOUP-REC (there is no `summary.json`, `selfcheck.json` or launch-gate token for a soup).
* **Templates:** `rollout_causal.tmpl`, `rollout_logged.tmpl`, `t0fresh.tmpl` unchanged; a new `t0std.tmpl` (= `t0fresh.tmpl` minus the two tokens `--ages fresh`); every template is copied into `argv/` here with its sha256 (`argv/TEMPLATES.sha256`).
* **Steps per soup:** causal, logged, fresh T0, standard T0 (4, not 3); plus the standard T0 of DAG-OACC_s and A_s once.
* **Tool tree:** `tree_li` as pinned (digest `579183a3…`); no new overlay (a soup's cfg is its donor's, which `tree_li` accepts: MEASURED for A, whose cfg carries `launch_intent_w`).
* **Manifest:** `soup_read.py`'s (`read/soup_read.py` docstring): the LAUNCH-INTENT `runs.json` entries for CTRL, REP, A, Arep, plus SOUP / SOUPR / SOUP4 entries and `t0_dev` for CTRL and A.
* **Reader step:** `soup_read.py --manifest … --prereg PREREG_V8F_SOUP.md --prereg-sha <pin> --li-reader <launch_intent_read.py, sha256 fb3d2d7d…> --orc-dir E:/Projects/TanitAD-artifacts/dagorc_rebase/read`.
* **Rendered commands** for every step are in `argv/PLAN.txt` (`tools/render_plan.py`; rendered, never executed).

## 14. Evidence register

| number | class | path |
|---|---|---|
| REP − DAG-OACC rows, CTRL − P10 rows, A − CTRL / Arep − REP ADE6, the 19 bars, RF | MEASURED (banked read of record) | `raw/2026-10-28-launch-intent-read/read_A1/LAUNCH_INTENT_READ.json` at `d7dc9f11` |
| M-LAUNCH floor −0.556 [−0.699, −0.398] m/s | MEASURED | `E:/Projects/TanitAD-artifacts/dagland/launch_intent_mlaunch/MLAUNCH.json` |
| member and init sha256; checkpoint structure (276 float32 tensors, 7,233,343 elements, cfg diffs) | MEASURED | `evidence/structure.txt`, `evidence/inputs_sha256.txt` |
| K-BASIN, K-XINIT, K-ZERO-W, K-MID numbers | MEASURED after §6 was written | `evidence/basin.json`, `evidence/build.out` |
| rollout and T0 wall times | MEASURED (LAUNCH-INTENT chain log) | `E:/Projects/TanitAD-artifacts/launch_intent_t1/logs/chain.out` |
| SOUP_0 − DAG-OACC_0 ≈ +0.20 m ADE6 at first order | ESTIMATED (linear interpolation of a banked number) | §0 item 3 |
| "part of the crawl error is draw-specific"; the soup mechanism | HYPOTHESIS; literature PUBLISHED-SECONDARY until banked | §1 |
| the outcome table's completeness | MEASURED | `evidence/enumerate_cells_soup.out` |
| builder tests, mutation pass | MEASURED | `evidence/pytest_*.txt`, `evidence/mutation_builder.json` |

**Banking the primary (blocked here, §0 item 5).** For the lander, from the repo root:
`python tools/kb_add.py 2203.05482 --tag soup --note "uniform and greedy weight averages of fine-tunes from one pretrained init; the average stays in one low-loss basin and improves accuracy without inference cost" --cited-by "Project Steering/PREREG_V8F_SOUP.md"`

## Amendment A0 (2026-10-28): non-finite buffers in K-BASIN and K-MID, before any finite weight distance was seen

**When.** After §6 was written and stamped (`evidence/pre_measure_stamp.txt`), the first `soup_build.py basin` run printed `nan` for every distance and every ratio, and reported `NOT AS REQUIRED` (`evidence/basin_run1_nan.out`, `.json`). No finite distance, no ratio and no soup row had been seen. Its XINIT line read "regression_as_required True" only because every comparison with `nan` is false: a vacuous pass, which the run's own `ALL_AS_REQUIRED` refused.
**Why (MEASURED, `evidence/nonfinite_buffers.txt`).** Six float32 buffers, `wm.stack.prior_lo_k` / `prior_hi_k` for k = 1, 2, 3 (2 + 2 + 32 + 32 + 32 + 32 = 132 elements), are ±inf in all eight members and both inits. (−inf) − (−inf) is `nan`, so the L2 norm of §6 is undefined on them.
**The amendment (definitional; no factor, member, reference or outcome changes).**
* A non-finite element must sit at the same position with the same bits in every member (and in the init a distance is taken to); otherwise the build and the distance REFUSE. Such an element is copied unchanged into the soup and contributes exactly 0 to every distance.
* K-MID checks non-finite elements explicitly (bit-equal to the members') and every finite deviation is required finite, so a `nan` cannot pass by comparison.
* Every distance must be finite, or K-BASIN REFUSES.
* Tests: `tests/test_soup_build.py` (`…infinite…`, four tests), green before the re-measurement.

## Weight-space readings (MEASURED after §6 and Amendment A0; weights only, no rollout, no eval)

`evidence/basin.json` (sha256 `4417351b…`), `evidence/build.out`, `evidence/verify.out`, `evidence/kzero.out`, `evidence/loadcheck.out`; builder `soup_build.py` sha256 `c9d7e7b9…`, spec `soups.json` sha256 `d8935b41…`.

| quantity | s0 | s1 |
|---|---|---|
| R_ft (median displacement of the four fine-tunes from M2b_s) | 10.2887 | 10.1957 |
| members' r to the declared init (min – max) | 10.1928 – 10.4187 | 10.0061 – 10.3646 |
| members' r' to the other seed's init (min – max) | 175.4631 – 175.4828 | 175.4360 – 175.4572 |
| d(DAG-OACC_s, REP_s); spread ratio of SOUP_s | 11.9909; **1.1654** | 11.8859; **1.1658** |
| d(A_s, A_rep,s); spread ratio of SOUP'_s | 12.1298; **1.1789** | 11.8326; **1.1605** |
| SOUP4_s max pairwise d; spread ratio | 12.1298; 1.1789 | 11.8859; 1.1658 |
| **XINIT** d_max; spread ratio | 175.9275; **17.0991** | 175.9275; **17.2551** |

* **K-BASIN PASS on all six soups** (B1 and B2); bar on the spread ratio 3. **K-XINIT as required:** both cross-init soups FAIL, and each fails with B1 alone and with B2 alone.
* **K-MID PASS on all six:** max |soup − numpy float64 mean| = 0.0; 0 elements outside the members' range; for the two-member soups |d(soup, a) − d(a, b)/2| ≤ 2e-9 against d(a, b) ≈ 12 (bound 1e-6 · d(a, b)).
* **K-ZERO-W PASS** on two real members (DAG-OACC_0, A_1) at N = 2, 3, 4. **K-SOUP-REC PASS** on all six (re-read from the files). **Loader:** all six load through `tree_li`'s own `V8FMiniConfig(**cfg)` and `V8FMini(...).load_state_dict` with 0 missing and 0 unexpected keys, `kin_mode` yaw.
* Soup checkpoints (`ckpt/<run_dir>/ckpt_last.pt`, 29.0 MB each; `opt` is `None`): SOUP_s0 `238aab03…`, SOUP_s1 `81924b1a…`, SOUP'_s0 `3db5166a…`, SOUP'_s1 `e0d1b6ec…`, SOUP4_s0 `006b2a0a…`, SOUP4_s1 `8b07ab1a…`.
* Reading (HYPOTHESIS, decides nothing): the two fine-tunes of one init are 1.17 R_ft apart, below the √2 R_ft of independent displacements of that size, so their fine-tune steps share direction; the two inits are 17 R_ft apart.

## Package status (2026-10-28)

* **Builder:** `soup_build.py` + `tests/test_soup_build.py`, 33 passed (`evidence/pytest_builder.txt`). Mutation pass `tools/mutation_builder.py`: run 1 had 33 mutants, 30 killed, 3 survived (`evidence/mutation_builder_run1_3survivors.*`): two were equivalent (K-MID's range clause and distance identity are implied by its exact-mean clause; listed under NOT MUTATED), one was a real gap (no test made `k_zero` fail; a test was added). Run 2: **31 mutants, 31 killed, 0 survived, 0 invalid** (`evidence/mutation_builder.json`).
* **Reader:** `read/soup_read.py` + `read/test_soup_read.py`, 25 passed on a fake heavy layer (`evidence/pytest_reader.txt`); it imports the landed LAUNCH-INTENT reader (`read/ref/launch_intent_read.py`, sha256 `fb3d2d7d…`, pinned). PREREG-TEXT reads 0 problems on this file. ⚠ **The heavy layer (`SoupHeavy`) has not run on real dumps**; launch precondition 4 (K-RC on the banked LAUNCH-INTENT dumps) is its first execution. No mutation pass was run on the reader.
* **Outcome table:** `tools/enumerate_cells_soup.py` OK: 512 composed cells each match exactly one row (REFUSED 256 · HARMS 192 · DOMINANT / ADOPTED 4 · DOMINANT / NOT ADOPTED 12 · MOVED / ADOPTED 3 · MOVED / NOT ADOPTED 9 · ONE SEED 24 · NO EFFECT 12); 200,000 random pair-level assignments, 0 unplaced, all 8 rows reached; 8 of 8 per-seed AND cases.
* **Chain:** `argv/*.tmpl` (sha256 in `argv/TEMPLATES.sha256`) and `argv/PLAN.txt` (28 steps, rendered, never executed). `tools/render_plan.py` reproduces the LAUNCH-INTENT chain's causal and fresh-T0 commands for A_0 token for token. The guarded chain itself (RAM / disk guards, done records, re-hashes) is the integration step of §13 item 3.
