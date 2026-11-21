# PRE-REGISTRATION: M6 `--yaw-loss plain`, dev-box decoder-side proxy (REFe, snapshot 015)

**Status: REGISTERED BEFORE ANY ARM RUNS.** The coordinator records this file's git blob before the first proxy arm starts.
The result must quote that blob. Any change after registration is a numbered amendment with its own blob. A result
produced under an unregistered change is not admissible for this decision.

## 1. Question
The live model's native step-19 heading is corrupted in every evaluated snapshot, 005 to 015.
- The WTA winner's error against the human future at 4.0 s is 0.93-1.61 rad.
- 60-72 % of all slots' raw step-19 headings lie beyond +-pi.

The cause (MEASURED 2026-09-27, `raw/2026-09-27-training-measures/m6_*`) is the wrapped heading L1 in `model.wta_loss`:
- A 2*pi branch shift costs exactly 0.0 under that loss.
- A heading output that drifts onto another branch is never pulled back.

**Question:** does `--yaw-loss plain` (`measures.wta_loss_yaw`: the same winner selection and position term, heading L1
not wrapped) remove the trap without costing driving quality, starting from the live model?

## 2. Design (four arms, identical except where stated)

| arm | loss | seed |
|---|---|---|
| W0 | `--yaw-loss wrapped` (today's; `model.wta_loss` bit-for-bit) | 0 |
| W1 | wrapped | 1 |
| P0 | `--yaw-loss plain` | 0 |
| P1 | plain | 1 |

The seed sets the data order only; nothing else is random, because no layer has dropout. For each seed, W and P see the
identical sample sequence. The two arms of a seed must differ in `--yaw-loss` alone, which the argv diff checks (gate G5).

**Start.** Snapshot 015: `D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt`, md5
d7c59f4f2fbcbde3e2dec8f67d63a7e7, live step 4933.

**Frozen, and cached once per frame** (`refe/proxy_cache.py`):
- the DINOv3 ViT-L trunk and its LoRA;
- the task registers, the pos3d MLP and the register compression;
- `scene_proj`.

In total: 10,306,304 of the 18,991,426 live-trainable parameters, plus the frozen trunk.

**Trained:**
- the trajectory side: `ego_enc`, `queries`, `dec`, `traj_head` (4,388,412 parameters);
- the scorer: `score_q_mlp`, `score_dec`, `score_head` (4,296,710 parameters).

**Train data.** The 10,000-frame proxy manifest (`raw/2026-09-27-training-measures/proxy_manifest_10k.json`), after
`finalize` against the pod index.
- It is stratified by log with seed 20260927 and includes every rank-1 twin of a picked scene.
- One sample per scene per proxy epoch, using the live `SceneEpochSampler`'s member rotation.
- Contexts come from the train split of the cache, in bf16 (the live forward's numerics).

**Trajectory loss.** `measures.wta_loss_yaw(mode=wrapped|plain)`, with `yaw_w` 0.1 as live.

**Scorer loss.** The live on-policy loss:
- a per-component BCE on the labelled set of each covered sample;
- the constant `cov_norm` denominator;
- `score_w` 0.1.

The labelled sets are the pod's `onpolicy/sets` lines for the manifest's (frame, rank) keys. For each key the proxy uses
the newest (`ckpt_step`, `label_version`) with `ckpt_step` <= 4933, which is what the live run could have read at the
epoch-16 boundary. Uncovered samples contribute no scorer gradient, exactly as live.

**Optimiser and schedule.** AdamW, weight decay 0.01, one global grad-norm clip of 1.0 over the proxy's trainable
parameters. The live cosine continues from step 4933: lr = 2e-4 x (1 + cos(pi t / 10075)) / 2 for t = 4933 .. 5335,
that is 1.033e-4 falling to 0.907e-4.
- Adam moments are re-estimated first with 50 zero-LR steps, because snapshots carry no optimiser state. All four
  arms get the same 50 steps, on their own seed's data.
- Then **403 optimiser steps at effective batch 256**, which is one live epoch (103,168 sample passes).
- bf16 autocast for the forward, as live; every loss and assignment in fp32.

**Eval tokens: 1,123 navtest tokens.**
- W3's 200 (`A1_sub200_tokens.json`), where the defect was measured.
- Amendment 5's 923 (`eval/raw/a5_confirm/a5_confirm_tokens.json`, md5 3098d178..., 43 logs, none in W3's 200), where
  Amendment 7 was confirmed.

Contexts come from the eval split (`eval/proxy_eval_cache.py`): snapshot 015's encoder side through the planner's own
input path, in fp32. Each arm's decoders are run on those contexts; the pick uses the planner's shipped rule, the
navsim_v1 aggregate over the arm's own scorer logits.

## 3. Gates (all must pass before any statistic is read)
- **G1 train cache.**
  - C1 bit-exact at build (decoding the captured context equals the full forward).
  - `proxy_cache.py verify` returns ok.
  - Its frozen-in-proxy fingerprint equals snapshot 015's.
- **G2 eval cache.**
  - C1 bit-exact.
  - C2: on W3's 200, the full forward reproduces the stored `sub200_ep015` E-6 table (to_navsim, <= 1e-4 m, on the GPU).
  - On the 923, the planner's pick reproduces Amendment 7's recorded native executed plan (<= 1e-4 m).
- **G3 zero-LR identity.** A proxy run at lr 0 for 5 steps must:
  - leave every decoder and scorer tensor bit-identical to snapshot 015;
  - reproduce the eval cache's proposals (max |delta| = 0.0).
- **G4 code.** `refe/selftest_measures.py` passes on the exact code used, with the tested bytes' sha256 recorded, and
  its M6 checks (T10) pass.
- **G5 arms.**
  - Each arm's argv and config are recorded.
  - W_s and P_s differ only in `--yaw-loss`.
  - Every arm completes 50 + 403 steps.
  - No NaN in any loss.

## 4. Metrics
The winner is the proposal, among the arm's 64, with the smallest mean |dx| + |dy| to the human future over the 8 NAVSIM
poses (to_navsim of the native output; the 4.0 s pose is native step 19 verbatim).

**PRIMARY (the heading):** for each plain arm P_s, both of the following:
- (a) the median over the 1,123 tokens of |wrap(winner heading at step 19 - human heading at 4.0 s)| is <= 0.10 rad;
- (b) the share of ALL raw native step-19 headings (64 slots x 1,123 tokens) with |h| > pi is exactly 0.0 %.

**MANIPULATION CHECK:** each wrapped arm W_s must stay defective, meaning a median error (a) >= 0.50 rad. A proxy that
un-traps its own control cannot attribute the fix.

**PDMS (non-inferiority, margin 1.0).** Per seed and per token:

    D_s = PDMS(P_s pick, Amendment-7 repair OFF) - PDMS(W_s pick, repair ON)

- Scoring: `score_navtest_refe.py` unchanged, NAVSIM v1.
- The repair is applied exactly as `planner.repair_last_heading` does, to the executed plan before the NAVSIM conversion.
- Statistic: the per-token mean over the two seeds of D_s, then the mean over tokens.

**Guards:**
- **Position:** for each seed, the winner's mean position error of P_s minus that of W_s is <= +0.10 m.
- **Four families:** `families6.py` on every gating seam, reported in full. Any family whose CI separates adversely is
  named in the verdict. Strategic is unavailable in NAVSIM, and that is stated.
- **Seed floor:** F_W = |mean PDMS(W0, repair ON) - mean PDMS(W1, repair ON)| and F_P = |mean PDMS(P0, OFF) - mean
  PDMS(P1, OFF)|, reported beside every effect.

**Reported, not gating:**
- the PDMS of every arm with the repair on and off (with W_s it re-reads Amendment 7 inside the proxy);
- the best of 64;
- the NAVSIM sub-scores;
- the winner error per native step, 0 to 19;
- all metrics separately on W3's 200 and on the 923.

## 5. Estimator and seeds
Paired log-cluster bootstrap over the eval tokens' logs:
- 10,000 resamples, percentile 95 %, seed 20260927 (Amendment 7's estimator);
- the same resample of logs is applied to every arm and seed.

What it answers is "another draw of episodes". Two things are measured separately:
- the **training-run variance**, only through the seed floor;
- the **inference variance**, which is **zero by construction**: one deterministic forward per arm, no sampling.

## 6. Decision (both outcomes committed now)
**ADOPT**, if all of the following hold:
- G1-G5 pass;
- PRIMARY passes for BOTH plain arms;
- the MANIPULATION CHECK holds for BOTH wrapped arms;
- PDMS: the CI lower bound is >= -1.0, AND both F_W and F_P are <= 1.0 (the rig resolves the margin);
- the position guard holds for BOTH seeds.

**REFUTED**, if the gates pass AND at least one of these holds:
- PRIMARY fails for BOTH plain arms;
- the PDMS CI upper bound is < -1.0;
- the position guard fails for BOTH seeds.

**NOT PROVEN**, in every other case, with the reason named: split seeds; a PDMS CI straddling -1.0; a seed floor above
1.0; a control arm that un-traps; a failed gate.

**On ADOPT.** A recommendation to the PI (a training change needs the PI's decision): `--yaw-loss plain --declare-change
yaw_loss` at the next resume of the live run. The conditions that go with it:
- The trainer's own seam guard must pass on the full pod bank.
- Amendment 7's repair stays ON until a live snapshot shows the step-19 heading fixed. Switching the repair off is a
  separately registered step.
- The scorer's selection skill is re-read at the first snapshot after the switch, because its labelled sets carry the
  trapped headings until new dumps arrive.

**On REFUTED.** No live change. The next lever is a branch-free heading parametrisation: a sin/cos pair per step, or a
bounded pi*tanh output. That is a `traj_head` shape change, which needs new parameters, a checkpoint migration and its
own pre-registration.

**On NOT PROVEN.** No live change. The named discriminating step comes next: more seeds for a seed floor above margin,
or more tokens for a wide CI.

## 7. Limits, stated before the data
1. **The encoder side is frozen**, which is 54 % of the live-trainable parameters. The trap sits in `traj_head` (row 59),
   so it is inside the proxy. Any encoder-side part of the fix, or of a side effect, is invisible here.
2. **The global grad-norm clip sees decoder-side gradients only**, whereas the live clip also sees encoder-side ones.
3. **Adam moments are re-estimated**, not continued.
4. **Heavy repetition:** 10,000 frames are seen about 10 times in 403 steps, against once per live epoch.
5. **The scorer trains only on sets labelled from the live, trapped model**, because no dump or labeller runs in the proxy.
   The plain arms' fans are therefore partly out-of-distribution for their scorer. This biases the PDMS clause AGAINST
   adoption, which makes it the conservative direction.
6. **Absolute proxy PDMS is not comparable** with live snapshots. Only the paired contrasts above are read.

## 8. Artifacts
Every artifact goes to `raw/2026-09-27-m6-proxy/`:
- per-arm argv and config;
- the metrics logs;
- the final decoder and scorer weights;
- the eval dumps (native [64, 20, 3] per token);
- the seams and harness CSVs;
- the families JSONs;
- `RESULT_M6_YAWLOSS.md`, written by the analysis script from those files. No number in it is typed by hand.

The code used is `refe/proxy_train.py`, `eval/proxy_eval.py`, `refe/measures.py`, `refe/proxy_cache.py` and
`eval/proxy_eval_cache.py`. Their sha256 values are recorded in the result.


---

## AMENDMENT 1 (2026-09-27 ~21:55 Berlin): one more proxy epoch -- a NEW question, the SAME bars

**Status: DRAFTED AFTER epoch-1 data was seen.** This section is appended below the registered text (blob de651f51).
Its own blob is recorded in `eval/raw/SPEC_PREREG_HASH.txt` BEFORE any arm below runs. Sections 1-8 above are
unchanged, and the EPOCH-1 VERDICT is computed and reported exactly as section 6 defines it. This amendment neither
re-reads nor re-labels that verdict.

### A1.1 What was seen, and why a follow-up is admissible
- The seed-0 preview of the heading metrics was seen before this was written:
  - P0: 0.135 rad median winner step-19 error, and 1.81 % of raw step-19 headings beyond +-pi;
  - W0: 0.793 rad and 60.8 %.
- Seed 1 and every PDMS number were NOT seen.
- After one epoch the plain loss removes most of the trap but misses both PRIMARY bars. The live run has about nine
  epochs left. So the decision-relevant question is a NEW one: **does one more epoch of the same training clear the
  UNCHANGED bars?**
- This is not a relaxed bar:
  - every threshold below is the section-6 threshold, verbatim;
  - a pass is reported as "passes after 2 proxy epochs", NEVER as the epoch-1 test passing;
  - the epoch-1 result stands as reported.

### A1.2 Arms (each continues its epoch-1 arm for EXACTLY one more proxy epoch; the settings are the epoch-1 settings)

| arm | starts from | loss | seed (data order) |
|---|---|---|---|
| W0e2 | W0's final decoder + scorer tensors | wrapped | 0 |
| W1e2 | W1's final tensors | wrapped | 1 |
| P0e2 | P0's final tensors | plain | 0 |
| P1e2 | P1's final tensors | plain | 1 |

- **Adam:** 50 zero-LR moment re-estimation steps, because the epoch-1 checkpoints carry no optimiser state (as in
  section 2).
- **Steps:** then 403 optimiser steps at effective batch 256, as micro-batch 64 x accum 4 (the split declared for epoch 1).
- **Learning rate:** the live cosine continued: lr = 2e-4 (1 + cos(pi t / 10075)) / 2 for t = 5336 .. 5738, i.e.
  9.07e-5 falling to 7.83e-5.
- **Data order:** the same seed's SceneEpochSampler stream CONTINUED past the 115,968 samples that epoch 1 consumed. No
  order is replayed.
- **Unchanged from epoch 1:** the on-policy sets (ckpt_step <= 4933), the train cache (manifest sha256 unchanged), the
  fp32 eval caches of 1,123 tokens, bf16 autocast, clip 1.0, weight decay 0.01, score_w 0.1.
- **Code:** `refe/proxy_train.py` gains two default-off flags:
  - `--init-from` loads an epoch-1 arm's final tensors;
  - `--skip-samples` continues the stream.
  Each gets a validity check with a mutation arm in `refe/selftest_proxy_train.py`, recorded separately under
  `raw/2026-09-27-m6-proxy/epoch2/`. Epoch 1's G4 record is untouched.

### A1.3 Gates, metrics, estimator: sections 3-5, unchanged, applied to W0e2 / W1e2 / P0e2 / P1e2
- G5 reads 453 steps per arm, and W_se2 / P_se2 must differ in `--yaw-loss` and `--init-from` only.
- G1-G4 are as registered.
- PRIMARY for BOTH plain arms: median winner step-19 error <= 0.10 rad AND exactly 0.0 % of raw step-19 headings beyond
  +-pi.
- MANIPULATION CHECK: W_se2 >= 0.50 rad.
- PDMS non-inferiority (margin 1.0; seed floors <= 1.0) and the position guard (<= +0.10 m) exactly as section 4.
- The four families are reported.

### A1.4 Decision (all outcomes committed now)
- **PASS**, which is section 6's ADOPT applied to the epoch-2 arms: the recommendation to the PI is the live switch,
  `--yaw-loss plain --declare-change yaw_loss`. It is reported as "PASSES AFTER 2 PROXY EPOCHS". The section-6
  conditions travel with it: the seam guard on the full pod bank (it PASSED at 16:52), the heading repair ON until a live
  snapshot shows step 19 fixed, and the scorer's selection skill re-read at the first snapshot after the switch.
- **FAIL BUT STILL SHRINKING:** the epoch-2 arms miss ADOPT, AND for BOTH plain arms the epoch-2 value is at least 25 %
  below the epoch-1 value on BOTH heading metrics (the median winner step-19 error AND the beyond-pi share).
  - Report the per-epoch slope of each metric per plain arm.
  - Report what the slope predicts, ONLY up to epoch 4, which is 2x the fitted two-epoch range. That is the programme
    rule: never extrapolate more than 2x beyond the fitted range.
  - The live-switch decision then goes to the PI with that evidence.
- **PLATEAU:** any other epoch-2 heading outcome, i.e. either plain arm improves either heading metric by less than 25 %
  relative. The next lever is the branch-free heading head (a sin/cos pair per step, or pi*tanh). That is an
  ARCHITECTURE change, and it needs the PI's go and its own pre-registration.
- **NOT PROVEN:** the epoch-2 heading PASSES but PDMS, the position guard, a gate or the manipulation check does not.
  The reason is named, exactly as section 6.

### A1.5 Artifacts
- The arms go to `raw/2026-09-27-m6-proxy/arms/{W0e2,W1e2,P0e2,P1e2}/`; everything else to
  `raw/2026-09-27-m6-proxy/epoch2/`.
- `RESULT_M6_YAWLOSS_EPOCH2.md` is written by the analysis script, and carries the epoch-1 -> epoch-2 trend table.
