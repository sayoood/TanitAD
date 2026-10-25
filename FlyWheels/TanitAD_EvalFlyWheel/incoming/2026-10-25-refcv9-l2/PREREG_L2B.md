# PREREG_L2B -- brake LOCUS at the shipped checkpoint (per-frame nav): is the cruise false-brake a SELECTION or a CANDIDATE event?

Registered 2026-10-25 (dev-box clock, before ANY fan / per-candidate number exists for this checkpoint). Author: Research Lab L2 agent, at the Master Mind's direction ("run the named next lever": L2 hand-off, `HANDOFF.md` next-lever section).
Evidence class of everything below: HYPOTHESIS under test. Tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**.
Estimator: episode-cluster bootstrap (`pair_read_r8.paired_cell` -> `taniteval.ci`, B 2000, seed 0, over EPISODES) = "another draw of episodes" only; NOT another training run (one training seed), NOT another inference run (seed 1 is the replicate and is never pooled).

## 0. Why, and what is disclosed

L2 (PREREG_L2 sha256 4267cd68..., verdict FAIL, outcome O5 CHANNEL-INERT) showed forcing CRUISE at the lon-head seam does not move false-decel (delta -0.0043, not separated) while forcing the GT label cuts missed-decel 0.470 -> 0.341. So the cruise brake is decided DOWNSTREAM of the lon class: in which candidates the fan contains, or in how the fan is scored. The 30k brake-locus work concluded "SELECTION vs CANDIDATE: NOT MEASURABLE from banked dumps (the refined fan is not banked)" (`raw/refcv9_headroom/BRAKE_LOCUS.md`). This roll banks the fan.
Pre-seal looks (disclosed): the L2 0-GPU readings (false-decel concentrated in FOLLOW-nav windows; dv of the false decels is deep, median -2.6 m/s; head calibration cannot reach 0.10) and the final L2 result. NO candidate-level quantity for this checkpoint has been seen. The FB set size at seed 0 (259) is known from the L2 rates.

## 1. Locus of the decision (source, tree fb47a58)
`refc_v3.py:2550-2640`: the fan `anchor_traj` [B, N=149, 8, 2] (117 base anchors + 32 allocated) is scored (`sel_score`, then `sel_score_v3` = blended with the goal graft and the drivable term); `e9_rank` (`refc_v3.py:1024-1054`) masks the score by `reach_keep`, `ceil_keep` and `r8_emit_keep`; `idx = rank.argmax`; `traj = fan[idx]`. The pre-blend refcv6 pick is `sel_idx_base`. `cfg.refcv7_select` plays no role in refcv8 (no refcv7 head is built). The shipped stage (DEPLOY8 per-frame table) is a post-process of `traj`.

## 2. The roll (ONE GPU chain, shared dev-box RTX 4060 through `gpu_lock`)
`code/l2b_locus_roll.py` (adapted from `l2_roll.py` / the CPU re-roll's field list): native arm (NO hook, NO force, NO calibrator), EMA-48,852 (md5 b4364029...), per-frame nav feed, os-only battery child, `windows_s0.json` (4,754 windows, md5 a3a78dba...), inference seed 0 (then seed 1, rule below). It banks per window: fan [149,8,2] fp32, `sel_score`, `sel_score_v3`, the three masks, `kind`/`lon6`/`lat3`, `cons_raw`, `hyp_alloc`, `prior_path`, `bank` [117,8,2] fp16, `plan8`, `sel_idx`, `sel_idx_base`, v0. Why the FULL window set and not a subset: the L2 native roll is bit-equal to L1's BASE-os on 308/308 windows of a PREFIX of the roll; a non-prefix subset would change the sampler stream. The full roll keeps every plan bit-reproducible.
GPU budget (ESTIMATED from the L2 rolls: 2,209 / 2,409 / 2,542 s per 4,754 windows): seed 0 ~2,500 s; a 9-clip smoke (~160 s) gates it; **seed 1 runs iff elapsed-since-lock + 1.05 x (seed-0 wall) <= 5,400 s** (the ~1.5 h cap). Anything not rolled is PENDING, never inferred.

## 3. Definitions (exact)
* Surface: S6 compliant windows (3,251 / 138 episodes), as in L2. `class(P)` = `s6_lon_decomp.plan_lon_class` (last chord 5->6 s minus v0, +-1.5 m/s, dt 1): decel / cruise / accel. **A "cruise candidate" is a fan candidate whose STAGED S6 path has class cruise**: candidate j -> the shipped DEPLOY8 stage with the window's own `prior_path` -> slots [1,3,4,5,6,7] -> `class`. (RAW-path classes are reported beside, as a sensitivity.)
* ELIGIBLE candidates = the exact finite mask of `e9_rank` (reach, ceil, emit masks with the dead-row fallbacks), computed from the banked masks; the emitted candidate must be eligible and the argmax of `sel_score_v3` over the eligible set (control K_C).
* Window sets (all from the ROLL's own emitted staged plan, seed s): **FB** = GT-cruise compliant AND emitted class decel (the false decels; expected 259 at seed 0); **CT** = a 1:1 deterministic match of FB among GT-cruise compliant windows whose emitted class is cruise ("correct cruise"), exact on the per-frame nav token, nearest |v0|, without replacement, processed in (episode, ws) order; **DR** = GT-decel compliant AND emitted class decel (correct braking; reference); **MB** = GT-decel compliant AND emitted class != decel (missed brakes; the mirror question, secondary).
* EXIST(set, target) = at least one ELIGIBLE candidate of class `target` (cruise for FB/CT/DR; decel for MB). GOOD = EXIST with a candidate whose staged ADE_6s vs GT <= tau, where **tau = the 0.90 quantile of the CT windows' emitted staged ADE_6s** (a control-derived tolerance: "as good as what the model emits when it is right"); ADE_6s = mean over the 6 instants of the L2 distance, equal to the battery's `ade_m` to float32 noise (K_H).
* Per FB window: SELECTION-window = EXIST (a cruise candidate existed and scored lower, necessarily, since the emitted one is the eligible argmax); CANDIDATE-window = not EXIST. Also read: the class of the pre-blend pick `sel_idx_base` (was the cruise candidate already lost before the goal/r8 blend?), the score gap emitted minus the best eligible cruise candidate (`sel_score_v3`), the count of cruise candidates.

## 4. Outcomes (pre-registered; seed 0 is primary, seed 1 the replicate)
R1 = label from s1 = P(EXIST | FB) with its episode-cluster CI: **SELECTION** iff CI lower > 0.67; **CANDIDATE** iff CI upper < 0.33; **MIXED** otherwise (both stated with s1).
R2 = the same rule on s2 = P(GOOD | FB).
**Informativeness guard**: if the point estimates of P(EXIST | FB) AND P(EXIST | DR) are both >= 0.95, class-level existence is saturated (a 149-candidate fan almost always holds a cruise-class path) and is non-discriminating; the final label is then R2, otherwise R1. The guard and both R1/R2 are always printed.
Replicate: seed 1 is read by the same rule; if its label differs from seed 0's the report says so and the final label is stamped "seed-unstable" (never pooled).
Meaning for the next lever: SELECTION -> a TRAIN-fitted selector / score correction (the L1-type lever; floor FD if every SELECTION window were fixed is printed); CANDIDATE -> a fan-generation / allocation / decoder-profile lever; MIXED -> both, ranked by their window shares and by the FD each could remove (printed: FD now; FD floors under "every SELECTION window fixed" for EXIST and for GOOD).
Descriptive headroom (never a bar): on FB windows with a cruise candidate, the ADE_6s of the best-ranked eligible cruise candidate minus the emitted plan's, and the GT-oracle candidate's minus the emitted plan's (paired, episode-cluster CI).

## 5. Controls that must read known values (a failed HARD control stamps the label INCONCLUSIVE)
* K_G (hard): every window carries every field (fan, scores, masks, kind, lon6, lat3, prior), fan size exactly 149, batch always 1.
* K_B (hard): the emitted plan equals `fan[sel_idx]` EXACTLY (bitwise), all 4,754 windows.
* K_C (hard): `sel_score_v3[sel_idx]` equals the max over the e9-eligible set, 100 % of windows (the replica of `e9_rank` is exact).
* K_A (reported): the re-roll's emitted 8-slot plan equals L1's un-hooked BASE-os plan bit-for-bit (expected 4,754 / 4,754); if not, the sets are those of the re-roll and the share of windows that changed class is printed.
* K_D (hard): the roll's `prior_path` equals the banked `S6_prior8` (tolerance 1e-4).
* K_F (hard): set sizes; at seed 0, |FB| = 259 (= 0.1395 x 1,856, the L2 BASE-os + stage false-decel count).
* K_H (hard): my ADE_6s of the emitted staged plan equals the battery's `ade_m` (tol 1e-4: the battery computes it in float32; measured float-noise max 1.6e-6).
* K_I (hard, known value): in CT the emitted plan is a cruise candidate, so EXIST(CT) = 1.000 EXACTLY; CT/FB balance on v0 and nav is printed.
* K_J (hard): the class of the staged `fan[sel_idx]` (per-candidate pipeline) equals the class of the staged emitted plan (window pipeline), 100 %; the score gap is >= 0 wherever defined.
* K_DRY (before launch): a synthetic trace with PLANTED shares (FB cruise-candidate 0.7, GOOD 0.35, MB decel 0.6) is read back EXACTLY by `l2b_score.py`, and a class-code mutation makes it disagree (`smoke/l2b_dryrun.py`).
* Reference shares, stated as REFERENCES not bars: EXIST(CT) = 1 by identity; EXIST(DR) is expected high (>0.9, ESTIMATED: a fan of 149 spans cruise-class paths) -- if so the guard fires and R2 decides; EXIST(MB, decel) likewise.

## 6. What is NOT computed / limits
Distance-keeping (no lead track in the dumps) and STRATEGIC (`--no-strategic`): n = 0. The stage is applied to every candidate with the WINDOW's prior, not a per-candidate prior (the prior is a window quantity: K_D). Classes are by the 6 s last-chord rule only. A locus reading says WHERE the brake is decided, not that a fix exists: the next arm is pre-registered separately.

## 7. Deviations log
Any deviation is appended to `DEVIATIONS_L2B.md` with the time; none is allowed after the first real fan number exists.
