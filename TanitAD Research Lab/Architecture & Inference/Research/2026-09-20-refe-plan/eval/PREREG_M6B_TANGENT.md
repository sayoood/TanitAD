# PRE-REGISTRATION (DRAFT): M6b -- a target-free tangent-consistency heading term on ALL 64 slots

**Status: DRAFT rev 2, 2026-09-27 ~23:35 Berlin (rev 1, blob cb5808f0, was never registered: its ADOPT branch was
unreachable on this rig's measured seed floors -- see section 5). Nothing below has run.** The coordinator records this file's git blob before
any arm starts. Any later change is a numbered amendment with its own blob. It is drafted AFTER M6's epoch-1 verdict
(REFUTED, `raw/2026-09-27-m6-proxy/RESULT_M6_YAWLOSS.md`) and the P0e2 preview. It is a NEW lever, not a re-test of
M6's arms.

## 1. Question, and why this lever
M6 measured the following (all at T0 proxy tier, snapshot 015, 1,123 navtest tokens):
- `--yaw-loss plain` removes most of the step-19 heading trap but misses both PRIMARY bars:
  - winner median 0.135 / 0.131 rad;
  - 1.81 % / 1.85 % of all raw step-19 headings beyond +-pi.
- One more epoch moves the winner median (P0: 0.135 -> 0.105) but barely moves the beyond-pi share (1.81 % -> 1.70 %).

**Mechanism:** under WTA only the winning slot receives a heading gradient. Slots that rarely win keep their trapped
branch, and a loss on the winner cannot reach them.

**Lever:** a TARGET-FREE term on EVERY slot. Each slot's heading is pulled toward the tangent of its OWN predicted
path. It needs no ground truth, so it reaches all 64 slots. It is a LOSS change, like M6: no architecture change, no new
parameters, no checkpoint migration.

**Question:** does `--yaw-loss plain_tangent` (M6's plain winner term + the tangent term) clear M6's heading bars on ALL
slots, without costing driving quality?

## 2. The term (fixed here, before any data)
For slot m, native step t = 0..19:
- positions `p_t` in the ego frame, with `p_{-1} = (0, 0)`;
- the tangent `theta_t = atan2(dy, dx)` of the displacement `d_t`, using the CENTRAL difference `(p_{t+1} - p_{t-1}) / 2`
  for t = 0..18 and the backward difference `p_19 - p_18` for t = 19;
- MASK: `|d_t| >= 0.2 m` (1 m/s at 5 Hz), and `|theta_t| <= 3.0 rad`. The second condition is the seam guard's limit,
  `measures.PLAIN_YAW_MAX_TARGET`: near +-pi the atan2 target itself flips branch;
- `L_tan = mean over (b, m, t) in the mask of |h_{b,m,t} - stopgrad(theta_{b,m,t})|`. The difference is PLAIN (not
  wrapped), so a heading sitting 2*pi away is pulled back;
- the target is stop-gradient, so positions get NO gradient from this term;
- total loss: `wta_loss_yaw(mode="plain") + tan_w * L_tan` with `tan_w = 0.1` (= yaw_w), i.e. `--yaw-loss plain_tangent
  --tan-w 0.1`.

Code (to be written after registration, before any arm):
- `measures.wta_loss_yaw(mode="plain_tangent", tan_w, tan_min_step=0.2)`;
- the flags in `refe/proxy_train.py`;
- the train.py hook and resume identity (`yaw_loss` value + `tan_w`) in `measures_hooks.patch`;
- the new all-slot metric in `eval/proxy_eval.py`;
- validity checks with mutation arms in `refe/selftest_measures.py` (T11) and `eval/selftest_proxy_eval.py`:
  - a straight path gives tangent exactly 0; a circle gives the analytic tangent;
  - a slot 2*pi off receives a gradient toward the principal branch;
  - masked steps contribute exactly 0;
  - positions receive EXACTLY zero gradient from L_tan;
  - MUTATION: removing the stop-gradient makes positions receive gradient and goes RED;
  - MUTATION: a wrapped difference gives the 2*pi slot zero pull and goes RED.

## 3. Design (M6's rig and settings, unchanged except where stated)

| arm | loss | seed (data order) |
|---|---|---|
| Wt0, Wt1, Wt2 | `--yaw-loss wrapped` (fresh, on the NEW trainer bytes) | 0, 1, 2 |
| T0, T1, T2 | `--yaw-loss plain_tangent --tan-w 0.1` | 0, 1, 2 |

THREE seeds per loss, so the training-run variance is MEASURED by replicates (H-ESTIM-SEED-1), not assumed away. See
section 5 for why two were not enough.

- Start from snapshot 015. 50 zero-LR steps, then 403 steps at 64 x 4 on the live cosine from t = 4933. Sets at
  ckpt_step <= 4933.
- Same train cache: the verified NVMe copy if the coordinator authorises it, else D:. ALL SIX arms read ONE path, so G5
  holds by its letter.
- Same fp32 eval caches of 1,123 tokens, same decodes, harness, families6 and estimator.
- The wrapped arms are re-run rather than reusing M6's W0/W1. That way every arm runs the same trainer bytes, which G4
  checks per arm.

## 4. Gates: M6 G1-G5, unchanged
G4 now also requires the T11 tangent checks. G5 requires Wt_s and T_s to differ only in `--yaw-loss` and `--tan-w`.

## 5. Metrics and bars
**PRIMARY -- EVERY T arm (T0, T1, T2) must pass (a), (b) and (c):**
- (a) winner median |wrap(h19 - human 4.0 s)| <= 0.10 rad. This is M6's (a).
- (b) exactly 0.0 % of ALL raw step-19 headings (64 slots x 1,123 tokens) beyond +-pi. This is M6's (b).
- (c) NEW, ALL slots: the median over all 64 x 1,123 step-19 slot-headings of |wrap(h19 - theta_19)| <= 0.10 rad,
  within the section-2 mask. It is each slot's own-path consistency, the quantity the lever targets. The masked count is
  reported.

**MANIPULATION CHECK:** every Wt_s's winner median >= 0.50 rad.

**PDMS non-inferiority, margin 1.0 (as M6; the margin is NOT widened):**
- Per seed s: D_s = PDMS(T_s, repair OFF) - PDMS(Wt_s, repair ON), per token.
- GATING STATISTIC: the SEED-MEAN paired D. Per token, the mean over s = 0, 1, 2 of D_s; then the paired log-cluster
  bootstrap (10,000 resamples, seed 20260927, the same log resample for every arm). Its CI lower bound must be
  >= -1.0.
- AND in at least 2 of the 3 PER-SEED contrasts D_s, the CI lower bound must be >= -1.0.
- REPORTED (never gating): the between-seed SD of D, from the three replicates, and the between-seed SD of each loss's
  own PDMS.
- WHY (MEASURED, M6, `raw/2026-09-27-m6-proxy/result_m6.json`): this rig's seed floors were F_W = 2.34 and
  F_P = 1.51 with two seeds. M6's clause "both floors <= 1.0" was therefore UNREACHABLE even for a perfect lever: a
  test whose ADOPT branch cannot be reached is not discriminating. Averaging D over three training replicates shrinks
  the training-run noise in the gating statistic by about 1/sqrt(3) relative to one seed. The 2-of-3 per-seed clause
  keeps a single lucky seed from carrying ADOPT. Training variance stays MEASURED (the SD above), never assumed away.
- For reference, M6 measured D = -2.74 [-4.38, -1.08] for plain alone. This clause is where M6 failed.

**Position guard:** T_s - Wt_s winner ADE <= +0.10 m, for every seed.

**Reported, not gating:**
- PDMS with the repair ON for every arm (T+repair vs Wt+repair);
- the four families;
- by-pose heading profiles;
- the beyond-pi share per native step;
- the M6 arms beside the new ones (different trainer bytes, so reported only, never pooled).

## 6. Decision (all outcomes committed now)
- **ADOPT:** gates, PRIMARY (a)(b)(c) for ALL THREE T arms, manipulation for ALL THREE Wt arms, PDMS (the seed-mean
  D's CI lower >= -1.0 AND at least 2 of 3 per-seed CI lowers >= -1.0) and the position guard for all three seeds. -> A
  recommendation to the PI: switch live with
  `--yaw-loss plain_tangent --tan-w 0.1 --declare-change yaw_loss --declare-change tan_w`. Conditions:
  - the seam guard on the full pod bank;
  - the launch gate for the new flag (a G-DVB entry);
  - Amendment 7's repair stays ON until a live snapshot shows step 19 fixed.
- **REFUTED:** gates pass AND (PRIMARY fails for ALL THREE T arms, OR the seed-mean D's CI upper bound < -1.0, OR the
  position guard fails for all three seeds). -> The branch-free heading head (sin/cos) is the next lever, an architecture change that needs
  the PI's go.
- **NOT PROVEN:** everything else, with the reason named.

## 7. Limits (M6's section-7 limits apply, plus)
- The tangent target lags the true heading by half a step at t = 19 (backward difference). MEASURED 2026-09-27 on the
  18,313 ground-truth rows of the train cache (rear-axle poses): GT heading vs its own path tangent at step 19 is
  median 0.0047 rad, p90 0.0377, p99 0.0862 (89.9 % of rows inside the mask); at steps 0-18 (central difference) it is
  median 0.0018, p90 0.0134. Max |GT heading| is 2.452 and max masked-in |tangent| is 2.357, both far from the seam.
  So the term does not fight the winner's ground-truth heading term, and bar (c) = 0.10 rad sits about 20x above the
  ground truth's own inconsistency.
- The term also constrains NON-winning proposals to be kinematically self-consistent. That is intended, but it changes
  what the scorer sees for every slot, and the scorer's labels still come from the trapped model (the bias is against
  adoption).

## 8. Artifacts
Everything goes to `raw/2026-09-2x-m6b-tangent/`. `RESULT_M6B_TANGENT.md` is written by the analysis script, and no
number in it is typed by hand.
