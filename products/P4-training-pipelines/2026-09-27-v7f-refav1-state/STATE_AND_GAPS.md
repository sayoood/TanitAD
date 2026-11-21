# v7F and refav1 — where they stand, what is missing, and the shortest road to a positive result

**2026-09-27 · TrainingFlyWheel (P4 training pipelines) · read against tip `b3f7ea6f` · STAGED, not committed.**
PI request (2026-09-26, verbatim): *"focus on continuing the work for both v7f and also refav1, deeply gather
the current state, where we are in the prep for training, how to prove and review the whole design, because we
need urgently positive results … benefit and transfer the findings from refe synthesized review reports."*

## ⭐⭐⭐ UPDATE 2026-09-27 ~16:50 Berlin — the refav1 trunk verdict, a learned kinematic bar, and all four build streams in

1. ⛔ **refav1's frozen trunk carries no trajectory information beyond kinematics** — pre-registered A2 + A3, fitted on
   600 TRAIN episodes, scored on the 141 EVAL episodes, validity controls OK (`…/2026-09-27-refav1-trunk-probe/RESULT.md`
   §3). Every readout of the trunk — linear or MLP, global or 4 × 10 spatial — fails to beat the SAME readout on the 8
   t0 kinematic features; adding the trunk to the MLP is separated-WORSE (+0.0054 m @ 2 s, +0.0987 m @ 6 s). ⇒ a
   heads-only retrain (H) is NOT the lever; the trunk itself (T) is (`REFAV1_RETRAIN_DECISION.md`, FINAL).
2. ⭐ **POSITIVE: a learned NONLINEAR kinematic readout is the strongest thing measured today** — an MLP on 8 t0
   kinematic features (past measurements only), fitted on TRAIN, scored on held-out EVAL: **6 s ADE 2.93 m** vs `damp50`
   3.45, `kd_x` 3.56, the linear readout 3.43 (−0.50, separated), `ha0_ext` 5.36; at 2 s it beats `ha0_ext` by −0.35 m
   ADE / −0.93 m FDE / −2.9° heading. AD-MLP's finding (`2305.10430`) on our data. ⇒ the programme's bar for "vision
   helps", and the best residual prior for every R5 head (refav1, v7F option B, refcv7 NEW-1).
3. ✅ **All four implementation streams delivered and were independently verified:** v7F R1+R4 (`v7f_r1r4/`), v7F R3
   (`v7f_r3/`: every tactical label family, incl. each traffic-light colour, now reaches a training loss behind
   `--w-tac-label-all`), the `v7f` launch-gate profile (`v7f_gate/`), refav1 R1/R3/R5/R6 (`refav1_r1r6/`, 68/68 staged).
   The v7F streams are MERGED in `v7f_merge/` (their two incompatible same-path modules unified; 3 conflicts resolved;
   full 107-file suite: 2,655 passed, failure set identical to the tip's; bit-identical to the tip with every new flag off).
4. ⛔ **CORRECTION (my 4th of the day):** the v7F gate's G-DVB found that `--nav-cond` was never mapped into the built
   model — my three earlier nav fixes had been tested on directly-built configs only, so no v7F launch had nav at all.
   Fixed in `v7f_merge/` and pinned through the REAL launch path (mutation-checked). The gate's own tests are being
   updated for the fixed state (they pinned the defect).
5. **Open for the PI:** refav1 option T's storage/compute; v7F option C + releasing the hold for S-W → S-T under R1–R6;
   D1 (the DINOv3 wrap); the R4 mode; the refav1 build's four decisions (κ0 source, cot-absence ruling for v7.2,
   SPEED_BAND target vs max-speed input, default prior); `--strategic-off` required (D3).

## ⭐⭐ UPDATE 2026-09-27 ~14:30 Berlin — the refav1 full-grid verdict, the planner eliminated, the next lever run

1. ⛔ **refav1 `loncomb3` FAILS its pre-registered full-grid test — FAIL-WORSE at both inference seeds**
   (`…/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md`; 141 clusters, T1, controls valid): `A1 − ha0_ext` ADE
   **+0.0896 [+0.0117, +0.1756]**, `A2 − ha0_ext` **+0.1515 [+0.0710, +0.2424]**. BAR-B1 (50/50 blend with `ha0_ext`)
   PASSES (−0.1072 / −0.0797) but BAR-B2 FAILS: the planner-free `damp50 = 0.5·ha0 + 0.5·ha0_ext` beats the blend,
   separated, at both seeds (+0.0505 / +0.0779). `damp50 − ha0_ext` **−0.1576**, separated ⇒ **`ha0_ext` is no longer
   refav1's strongest do-nothing floor.** The inference-seed replicate is itself separated: `A2 − A1` +0.0619.
2. ⛔ **CORRECTION to item 4 of the 01:15 update below:** "the planner's own contribution over the damped hold is
   LONGITUDINAL" came from the 8-episode p4 panel. On the full grid it holds at seed 0 only, and on the SAME damped
   path the planner's longitudinal profile LOSES to simply holding the measured acceleration (+0.078 / +0.095 m ADE,
   separated; EXPLORATORY). ⇒ **Every zero-training planner lever on checkpoint 21,109 is eliminated**: the planner
   adds nothing over t0 kinematics on either axis. Its lateral decisions are worse than planning straight (the shipped
   cost always plans κ = 0 and beats `loncomb3` laterally by 0.20 m cross-track) — p4's 8 turn-dense episodes rewarded
   a lateral behaviour the lane-keep-dominated grid (247/282 windows) punishes. **p4 is not an admissible tuning surface
   for lateral levers.** From source: the world model's lateral cost contribution is 1.63e-10 (`refa_v1.py:403`), and
   the proposal head was never trained (`w_aux_head 0.0`).
3. ⭐ **A stronger model-free floor** (EXPLORATORY, reported because it only makes tests stricter): the damped path
   re-timed to `ha0_ext`'s travelled distance (`kd_x`: half the measured curvature, the measured acceleration held)
   beats `ha0_ext` by **−0.2282** on the 282-window grid and **−0.2638 [−0.3374, −0.2005]** on the 2,399-window
   stride-4 grid of the same episodes.
4. **The next lever (Rule Zero), pre-registered and run:** does the frozen refav1 trunk carry trajectory information
   BEYOND kinematics at all? (`…/2026-09-27-refav1-trunk-probe/`, ridge readouts, 5-fold grouped-by-episode CV, all
   controls.) ⛔ **NEGATIVE with the mean-pooled state `plan()` feeds its heads:** trunk + kinematics is
   separated-WORSE than kinematics alone (`P4 − P2` **+0.0042 [+0.0016, +0.0071]**); raw DINOv3 at the same pooling
   likewise (+0.0045). ⭐ A LEARNED kinematic readout beats `kd_x` by −0.0558 (separated, held-out episodes). The
   committed next arm — the same probe on a 4 × 10 SPATIAL pooling of the token field (Amendment A1, staged before any
   spatial feature existed) — is running now.
5. **R1 full-grid verification pre-registered and queued** (`R1_FULLGRID_CHECK.md`): A1 + `--vmax-from-labels`; the
   bar is 0 plan exceedances on all 282 windows. It waits for 5 idle GPU minutes (the card is shared with other
   sessions' jobs).
6. **v7F:** the binding launch gate `stack/scripts/launch_gate.py` landed at origin `c36b6ddd` (refcv7 profile only);
   `v6.py` / `train_v6_staged.py` / every refav1 file are unchanged there, so every base blob in `LANDING_READY.txt`
   stays valid. Two option-independent R1–R6 streams run in private copies of `c36b6ddd`: **R3** (all tactical labels
   incl. the traffic-light goals reach a training loss — the `D-TLIGHT-1` fix) and **R1 + R4** (max speed as a
   tactical input + plan cap; prove or build tactical → operative conditioning).
7. ⛔ **CORRECTION to the v7F decision I put to the PI (`V7F_PLAN_R1_R6.md` §3′):** I wrote that S-T "chooses by
   imagining each candidate with the world model", so P2 (action-deafness) blocks it. The source says otherwise: S-T
   emits a goal-conditioned fan and SELECTS by goal distance / a learned scorer; world-model roll-consistency as a
   selector was already measured +5.98 m worse and is refuted. **What gates S-T is `SEL-1`** (selector launches refused
   since E-WC2, σ/ADE 9.99 vs a 3.0 line — *"these latents are the wrong surface"*), whose committed unblock is the
   E-WC2-SW latent dump after S-W training. Option B (one decoded trajectory, residual over a kinematic prior) needs no
   selector, so **C (B first) still stands — for the SEL-1 reason, not the P2 one.** Refav1's trunk probe finds the
   same "latents do not beat kinematics" pattern on a second architecture. ⭐ **And option B is ALREADY a
   configuration of the existing trainer, not a build:** `--n-candidates 1 --proposals query --selector none` emits
   ONE goal-conditioned 6 s trajectory with no selector (so SEL-1 does not apply); it still needs R1 and R3.
8. ⭐ **At R5's 6 s horizon the do-nothing floors separate by metres** (EXPLORATORY, same 2,399 windows / 141
   clusters): holding (a0, κ0) for 6 s (`ha0_ext`) ADE **5.36 m** / FDE **15.39 m**; constant velocity 4.13 / 11.21;
   the damped `damp50` **3.45 / 10.15** (`damp50 − ha0_ext` ADE −1.90 [−2.36, −1.50]). Any 6 s claim read against
   `ha0_ext` is inflated by ~2 m; the R5 head's prior is now a flag (`damp50` at 6 s, `kd_x` at 2 s). Script:
   `…/2026-09-27-refav1-trunk-probe/raw/floors_6s_exploratory.{py,json}`.

## ⭐ UPDATE 2026-09-27 ~01:15 Berlin — three things ran after the first version of this report

1. **v7F S0 ran (pre-registered, `PREREG_V7_SEED_POS.md` §6) at the launch geometry ViT-B/16** —
   `TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-v7f-s0-b16/RESULT.md`. All controls hold.
   ⛔ **The cheap repair (ImageNet fold + zero `pos`) does nothing for scene content** (−0.0053 [−0.1663, +0.1393]),
   replicating the L/16 null — the pre-registered binding bar FAILED, as its §0b predicted. ⭐ The seed as wired
   already carries scene content beyond random init (+0.1941, separated). ⭐ What the transplant loses vs the real
   DINOv3 at B/16: a **separated** share of ego-speed content (+0.2331); the scene-content loss (+0.1264) is **not
   established** on 24 clips. ⚠️ **CORRECTION to §3/§4 below:** I first framed "fold ImageNet mean/std into the
   patch conv" as the v7F trunk fix. It is free hygiene, not a lever — measured null twice. The lever is
   **`PREREG_V7F` §10 D1 option A** (wrap the real `DINOv3ViTModel` as the trainable initialisation, ~1 day +
   tests) — an architecture choice, so **a PI decision**; S0 supports it on the ego axis.
2. **refav1: the 17 stranded arms are READ** —
   `…/2026-09-05-refav1-close-the-gaps/raw/refav1_margin_banked_2026-09-27/RESULT_HARVEST.md`.
   ⛔ `kammshift` **FAILS its pre-registered bar** (ADE 1.0652 > 0.8892). ⭐ `loncomb3` (`lonshift` + jerk 0.02 +
   jerk-seam a0 — the corrected design after `D-REFAV1-LON-SEAM-INERT`) is the **best refav1 configuration
   measured**: vs the strongest do-nothing floor `ha0_ext` it is separated-better on along-track (−0.2660),
   heading (−3.36°) and yaw-rate (−0.1064), **still worse on speed (+0.1608)**, ADE −0.1033 not separated on 8
   clusters. The inference-seed replicate is itself "separated" on LON speed (−0.0279) — `loncomb3`'s LON gain over
   `lonshift` is 3.6× that floor.
3. **refav1 full-grid confirmation PRE-REGISTERED before any data** —
   `…/2026-09-27-refav1-fullgrid-loncomb3/SPEC.md` (staged 01:05:48): `loncomb3` × 2 inference seeds on the
   141-episode / 282-window grid; BAR-L1 = beats `ha0_ext` on ADE, separated at both seeds, > 2× the replicate
   floor. Needs ~2 × 3.1 h on Thor or ~2 × 1.6 h on the 4060 after a LAN copy; not started (compute below).

4. **⭐ The do-nothing floors are too weak — a model-free DAMPED hold beats all of them** (exploratory, zero GPU;
   `…/2026-09-27-refav1-fullgrid-loncomb3/raw/`). On the FULL 141-episode grid, `dampha = 0.5·ha0 + 0.5·ha` beats
   `ha` by **−0.1559 [−0.2249, −0.0948]** and `ha0` by **−0.1484 [−0.1846, −0.1160]** ADE (separated, 141 clusters,
   w=0 control exactly 0). On p4 the planner blend `0.5·loncomb3 + 0.5·ha0_ext` beats `ha0_ext` by −0.2723, but the
   planner-free `0.5·ha0 + 0.5·ha0_ext` already gets −0.2152 of it: **~80 % of the "blend win" is damping**; the
   planner's own contribution over the damped hold is LONGITUDINAL (speed −0.0918, along −0.0978, separated).
   ⇒ "beats doing nothing" must be read against a damped hold; refav1's full-grid bars now include it
   (Amendments A1–A3, all before any `loncomb3` full-grid data). Sent to the Master Mind: SPEC_REFCV7's NEW-1
   rationale ("blend with the hold beats the echo") needs the same planner-free control; a damped prior is also a
   free refcv7 lever.
5. **Still waiting:** the Master Mind's answer on ~6 h of idle Thor for the two full-grid arms (the 4060 is held
   by another stream); the PI on `W_VEND` (pinned by `test_steer_conversion_complete.py::test_C1` — preparing it
   default-off would still trip the pin, so it is NOT prepared), v7F D1, and the v7 hold.

6. **PI directive 2026-09-27 (binding) — requirements R1–R6 for refav1 AND v7F**, and **"close P3, solve the
   rest"**. Compliance audit from tip source:

   | req | refav1 | v7F |
   |---|---|---|
   | R1 max speed = input + hard cap | ✅ **cap IMPLEMENTED 2026-09-27** (inference; `PlanConfig.v_max`, `--vmax-from-labels`; 128/128 tests, 0 plan exceedances in a live smoke); as a tactical INPUT it needs the retrain | ABSENT (0 hits in the trainer) |
   | R2 nav → tactical + operative, train + inference | ✅ MET (`refa_v1.py:899,1475-1481,1664`) | ⛔ **CORRECTED 2026-09-27 afternoon:** present in the MODEL (`--nav-cond`) but a real `--nav-cond --nav-labels` launch cannot START at the tip — MEASURED by the R1/R4 stream: `NavTokenMissing` at stack build (`assert_isolation` on a `synthetic_batch` with no nav token, `v6.py:6181-6201`) and at step 1 (a literal `"nav_token": b.get("nav_token")` at `train_v6_staged.py:7336` overwrites the NavEmitter splat at `:7314`; pinned by `tests/test_nav_v6stack.py:195`). Small fix, folded into the v7F merge (`v7f_r1r4/RESULT.md` §7) |
   | R3 ALL tactical labels train the tactical layer | PARTIAL — only lat/lon decision labels (`loss_lat_label`, `loss_lon_label`) | ABSENT at S-W (goal heads exist, not trained at S-W) |
   | R4 tactical conditions the operative plan | ✅ goal token → cost; FiLM tactical → operative | not at S-W |
   | R5 combined trajectory to 6 s | ⛔ plans 2 s; the goal is already 6 s but the imitation proposal head is sized to 2 s (`refa_v1.py:1547`) → a clean 6 s needs the retrain | 6 s plan exists (60 × 0.1 s) but is not trained at S-W |
   | R6 strategic OFF | ⛔ `--no-hierarchy` drops tactical too, and `nav_to_ctx` is sized from `strategic_cfg` (`refa_v1.py:1562`) → an architecture change + retrain | the goal/selection stage also trains the strategic layer → a stage redesign |

   ⛔ **Infrastructure:** refav1's fp8 feature cache is deleted from Thor (0/4,572 train, 0/141 eval links resolve;
   Thor 89 % full); a healthy 141-episode EVAL copy survives on the dev box, so inference work continues there; any
   retrain needs the train features rebuilt (~303 GB of fp8 features — 4,572 × ~66.2 MB (`T_c × 655,360 + 1,908` B per episode, the encoder's documented size); ⛔ CORRECTED 2026-09-27: I first quoted ~155 GB, which is the v2ep SOURCE-frame cache (34 MB/ep) the encoder reads, not the fp8 cache the trainer reads (class C82: price the file the CONSUMER opens)) and a disk to hold them.

## 0. How this was built, and what I verified myself

- The local `D:` tree is **186 commits behind** the tip. Everything was read from a snapshot of `b3f7ea6f`
  (`C:/Users/Admin/tipsnap/b3f7ea6f/`, 3,795/3,795 files verified) or with `git show b3f7ea6f:<path>`.
- Three read-only surveys, banked verbatim in `raw/`: `SURVEY_V7F.md`, `SURVEY_REFAV1.md`, `SURVEY_REFE_LESSONS.md`.
- **Re-checked by me against the primary artifact (MEASURED this session):**
  - refav1 full grid, `cl − ha0` ADE **+0.0158 [+0.0007, +0.0315], separated, worse**; `cl − ha` LON speed
    **+0.2706 [+0.2140, +0.3312]**, worse; LAT cross −0.1569 (better, separated); TAC lon −0.1277 (worse, separated)
    — `taniteval/results/RESULT-refav1-21109-openloop.md:45,396-422` at `b3f7ea6f`.
  - refav1 p4 `lonshift`: vs `ha0_ext` ADE **−0.0889 [−0.2049, +0.0137]** (seed 1: −0.0978), FDE **−0.3706
    [−0.7361, −0.0293] separated**, LON speed **+0.2535 [+0.1162, +0.4227] worse** — `…/2026-09-05-refav1-lonshift-t1/raw/thor/pd_thor.md`.
  - v7F input normalisation: `_contract.py:51-56` is `x.float().div(255.0)` and nothing else;
    `dinov3_seed_checkpoint.py:344-349` copies the patch conv with transform `"identity"`; `encoder.py:161-163`
    is patch → flatten → `+ pos`; **0** ImageNet mean/std literals in the five v7F-path files; the only ones are
    the O7 teacher's (`train_v6_staged.py:1008-1009`), which v7F sets to weight 0.
  - The v7F §9 launch line through the real argparse + preflight (`raw/v7f_preflight_probe.*`): fails in argparse
    on the three LDAD flags; minus LDAD, 3 preflight refusals + the anchor refusal; with three fixes, **0 problems,
    anchor PASS**. v7F-pinning tests: 165 passed / 2 skipped (`raw/pytest_v7f.out.txt`).
  - Thor (read-only ssh, 2026-09-26 22:55): **no trainer running**; refcv6 carries `STOPPED_BY_PI.json` (20:40:45),
    last checkpoint step 38,000; **no v7 checkpoint exists on Thor**; refav1's newest checkpoints are `refav1_lon/ckpt/ckpt.pt`
    (2026-09-05) and `experiments/refav1-b1-v72-ep3-speed/ckpt.pt` (2026-09-04); fp8 cache 4,572 train + 141 eval.
  - The v7F DINOv3 B/16 seed exists only on the dev box: `D:/Projects/TanitAD-artifacts/dinov3-seeds/dinov3_vitb16_seed.pt`.
- Everything else below is **INHERITED** from the surveys, which cite the MEASURED source `path:line`.

## 1. The rules and the compute that bind BOTH lines

| constraint | what it means for v7F / refav1 | source |
|---|---|---|
| ⛔ **Launch gate, BINDING since 2026-09-26** | No training run starts without a PASS token from `stack/scripts/launch_gate.py`, bound to commit + argv sha256 + data-manifest sha256s; seven checks (G-HYG, G-DVB, G-LIVE, G-CLOCK, G-EVAL, G-CKPT, G-SUITE), each with a regression arm that must FAIL it; **a trainer flag without a G-DVB entry is refused**. **The file does not exist at the tip**, and the spec covers only `refc_v3_train.py`. ⇒ **neither v7F nor refav1 can train today**, however good the recipe. | `CLAUDE.md:1287-1311`; `SPEC_REFCV7.md:37-51` |
| ⛔ **v7 training hold, PI 2026-08-31** | *"we should not start training of v7 until the remaining problems are solved."* Only the PI closes items. Not lifted, and no queue item asks for it. | `V7_LAUNCH_GATE.md:3-15` |
| PI work order 2026-09-15 | **refcv6 → refav1 → v7**; preparation and final design on the **dev box**; heavy training on **a pod the PI provides**. | `Decisions/2026-09-15-pi-directives.md` (e), (f) |
| PI 2026-09-26 20:40 | refcv6 stopped; **refcv7 is the next Thor arm**, launched only after its gate passes on the dev box and on Thor. | `PI_DECISION_QUEUE.md:1813-1818` |
| Compute now | Thor free (refcv7 still being built); A40 on REFe until ~Oct 1 02:19; dev-box RTX 4060 shared with other streams. | Thor/dev-box probes above |

⭐ **The consequence that matters most:** refav1's next step needs **no training at all** — it is planner arms on the
existing 21,109-step checkpoint, which run on the dev-box 4060 (~1.6 h per full-grid arm, ESTIMATED from 20.96 s/window).
v7F's next steps are dev-box probes too. **Neither line needs to wait for Thor or a pod to make progress this week.**

## 2. refav1 — current state

**What it is:** frozen DINOv3 ViT-L/16 fp8 features → WideAdapter → three token-field world models (0.2 s×30,
0.6 s×10, 1.5 s×4) chained by FiLM → a test-time **iCEM planner** (300×30) that also injects constant-velocity, hold
and brake plans into its population. 182.5 M params. Trainer `stack/scripts/refa_v1_train.py`. Corpus B1/v7.2
(4,572 episodes; not the parity corpus, allowed by PI ruling).

**Training:** one complete run, `refav1-b1-v72-ep3-speed`, 21,109 steps, finished 2026-09-04 (copies on Thor and the
dev box, md5 `1189bc02…`, no HF copy). ⚠️ It was trained from Thor's older stack (`refa_v1.py` 1,885 lines vs the repo's
3,030), so a rerun from HEAD is not a replication. Parked since 2026-09-06.

**The verdict at T1 (self-action open loop), full grid 282 windows / 141 episodes:** the shipped planner returns an
injected do-nothing plan on **282/282** windows and **does not beat doing nothing** (the `cl − ha0` / `cl − ha` rows in §0).

**What training DID buy (T0 / probes):** the world model beats "persist the last feature field" from 1 s (+0.1129)
to 6 s (+0.2454); the LON tactical head 0.4326 vs a 0.2979 floor; the lead gap is decodable from the latent
(R² 0.3632 vs pixels −0.0513), its closing rate is not.

**Diagnosis (the flat plan is the COST, not the weights):** a perfect goal makes the plan 2× worse ⇒ the search is
healthy and the objective is misspecified. Goal term ~1e-7 vs penalties ~0.1 excludes 100 % of the population before
the world model is consulted; the turn token commands κ 0.08 (5–10× too sharp); the LON goal token commands a≡0;
`W_VEND` never acts because `target_speed` is never passed.

**Best measured lever:** `lonshift` (`ccos` + `W_KAPPA` 15.11 + `a0_shift`) — **ADE parity with `ha0_ext` and a
separated FDE win, but LON still worse** — measured only on the 40-window / 8-episode p4 panel; never on the full grid.

## 3. v7F — current state

**What it is:** `train_v6_staged.py` at stage **S-W**; a ViT 768×12×12 trunk (86.1 M) **initialised from DINOv3 B/16
and trainable** (LR ×0.1, MSE anchor to a frozen copy of the seed); predictor conditioned on measured ego state
(`omega_accel_v`); nav mandatory; O5 L1 rollout + O6 SIGReg + O14 future-observation; 256×640 cylindrical B1 corpus.
**245.6 M params measured** (the prereg's "336.5 M" is stale). Hypothesis `H-V7F-1`: the trunk + anchor + an LDAD
term make the predictor action-sensitive without destroying decodability.

**Training:** **never trained.** No registry row, no checkpoint anywhere. All v7 evidence is **v7-tiny, T0**: non-collapse
holds relative to a reference arm; L3 (does the predictor add over `z_t`) **fails on every arm**; the predictor ignores
its actions (h1 action/scene ratio 0.50× against a ≥10× bar; 97 % of the response comes from the speed channel); the
only T1 read ever made is a **FALSE result** (decoded through a random-init readout) that `MODEL_REGISTRY.md:4724-4742`
still prints.

**Defects found or confirmed (all before any launch, so they cost nothing yet):**

| # | defect | status |
|---|---|---|
| 1 | ⛔ the DINOv3-seeded trunk receives **[0,1] frames with no ImageNet mean/std** | CONFIRMED at source (§0). REFe measured this class on a FROZEN ViT-S as rel-L2 displacement 0.562; ⚠️ for v7F's seed, **S0 measured that adding the normalisation (with `pos` zeroed) changes scene decodability by −0.0053, i.e. nothing** — correct it as hygiene, do not expect it to move a result |
| 2 | learned `pos` table left at **random init** where DINOv3 uses RoPE | declared in the seed stamp (`left_at_init_keys ['pos']`) |
| 3 | DINOv3 CLS + register tokens dropped | declared |
| 4 | the anchor is a frozen copy of the **mis-fed seed**, not of published DINOv3 | HYPOTHESIS: it anchors to the wrong function |
| 5 | §9 launch line: LDAD flags do not exist; `--horizons` default refused; `--w-s2-goal` refused at S-W; `--obs-monitor-every` missing | MEASURED (`raw/v7f_preflight_probe.*`) |
| 6 | **S-W cannot answer its own success criterion**: it trains only WM groups and leaves the metric readout at random init, yet G-DRIVE needs a four-family T1 read | from source |
| 7 | no training-seed floor anywhere in the v7 line — the 14.3 % replicate panel belongs to the **REF-C** tiny rig (`--arm/--withheld-bank` exist only in `refc_v3_train.py`), and `CLAUDE.md:180-188` misattributes it | MEASURED (flag ownership) |
| 8 | the metric-decode refusal guard the register calls "shipped" does not exist (its test is a module-level skip) | MEASURED |
| 9 | no tactical loss and no max-speed learning, against PI directive 2026-09-15 (c) | from source |

### 3b. What closes the 2026-08-31 v7 hold (`V7_LAUNCH_GATE.md` — each item carries its own closure rule; only the PI closes)

| item | closes when (the document's own words, condensed) | state 2026-09-27 |
|---|---|---|
| **P1** no v7 arm has beaten its own hold-action control (`:19`) | a v7 arm beats hold-action at T1 across the four families | OPEN — the only T1 read is void (random-init readout), and v7F's S-W stage cannot yield a T1 read |
| **P2** the model does not use its actions (`:37`) | a cause is established; a genuine COMMAND channel (not realised motion, r 0.9988) has never been tested | OPEN by PI ruling |
| **P3** drift, a seed-stable 3.3× effect of unknown cause (`:167`) | the cause is known | likely closable — the register later measured the effect null once its control is subtracted (`GOALS_AND_CLAIMS.md:7749`); the PI closes |
| **P4** the horizon ladder is short of the labels (median 12.5 s) (`:271`) | the recipe reaches the tactical band (MM-E19) | OPEN — MM-E19 answered no (2026-09-02) |
| **P5** the predictor adds nothing over the current latent — L3 (`:228`) | `zhat` beats `z_t` on the same targets, paired | OPEN — L1/L2 pass, L3 fails on every arm |

⇒ The designed route through P2/P5 is `PREREG_V7F`'s tiny-rig ladder (R0 → R3 + the LDAD term, which does not exist
yet); P1 additionally needs a stage that trains the readout, so that the first admissible v7 T1 read exists.

### 3c. G-DVB flag inventories (raw material for the binding gate)
`code/flag_inventory.py` (static AST scan, with a deliberate-regression self-test that must catch a planted unread
flag) → `raw/flag_inventory_{refa_v1_train,train_v6_staged}.{md,json}`: **40** and **232** flags, all read somewhere
in the stack (2 first-pass "unread" v6 flags were read through `resolve_gc(a, "<name>")` — false positives, fixed in
the tool); **4** and **36** loss/model levers default OFF, each needing a declared-vs-built check whenever a launch
line switches it on; 13 and 91 flags need the gate owner's classification.

## 4. What is missing — ranked, with the owner

### refav1

| # | missing | owner | cost |
|---|---|---|---|
| 1 | ✅ **DONE tonight:** 305 stranded files banked (the 17 arm records never read back, incl. pre-registered `kammshift`, `loncomb3`, `lonshift_s1`) — see §7 | me | — |
| 2 | **Read them**: paired four-family deltas vs `ha`, `ha0`, `ha0_ext`, each against its own inference-seed floor; score `kammshift` against its pre-registered bar (ADE ≤ 0.8892 and turn_left > 0) | me | CPU, hours |
| 3 | **Pre-register and run the full-grid confirmation** of the best lever(s) × 2 inference seeds — every lever result so far rests on 8 clusters | me (+ dev-box GPU) | ~1.6 h/arm on the 4060 |
| 4 | **PI rulings:** arm `W_VEND` with target max(0, v0 + a0·T) (reserved to the PI; the only lever aimed at the remaining LON gap); make the winning levers the default; turn-vocabulary magnitude | **PI** | a decision |
| 5 | Code: distance-keeping hook missing at the tip; `plan_source` mislabels seed winners; `cost_fidelity` (design gate G1) has no caller and is a pooled estimator; the trainer writes no done-marker (80 relaunches); the feature standardiser is fit on ONE batch (`refa_v1_train.py:581-582`) | me | small |
| 6 | For any **retrain**: a launch gate for `refa_v1_train.py`; code provenance (A13c); a closing-rate representation | me + PI | days |

### v7F

| # | missing | owner | cost |
|---|---|---|---|
| 1 | ✅ **DONE 2026-09-27: S0** at B/16 — the repair is null on scene content; the transplant loses a separated share of ego-speed content vs the real DINOv3 (see the UPDATE at the top) | me | done |
| 2 | ⭐ **PI decision: `PREREG_V7F` §10 D1** — keep the transplanted seed, or option A (wrap the real `DINOv3ViTModel` as the trainable initialisation, ~1 day + tests). The ImageNet fold is free hygiene either way, NOT the fix (measured null at L/16 and B/16) | **PI**, then me | a decision, then ~1 day |
| 3 | Make §9 preflight-clean (three flag fixes, explicit readout grid) as a dated **amendment** of `PREREG_V7F.md`, and **build or strike LDAD** | ArchInf / MM | small |
| 4 | A **launch gate for `train_v6_staged.py`** (11,057 lines; every flag needs a G-DVB entry) | me + the gate agent | days |
| 5 | Make the driving gate readable (S-W cannot yield G-DRIVE); register `H-V7F-1`; repair `actdiv_anchored` (pinned denominator + clip bootstrap); R0 / G-RANK | me / ArchInf | hours–days |
| 6 | A training-seed floor for the v7 line (a replicate arm) | compute | one extra arm |
| 7 | Tactical loss + max speed in the v7 trainer (PI directive 2026-09-15 c) | ArchInf | days |
| 8 | **Lift the 2026-08-31 hold, and a slot/pod** | **PI** | a decision |

## 5. How to prove and review the design — one method for both lines

1. **The binding launch gate, per trainer.** The seven checks of `SPEC_REFCV7.md` §2, each with its regression arm,
   adapted to `refa_v1_train.py` and `train_v6_staged.py`. This is what makes "we are sure about the correctness of the
   config" a token rather than a feeling.
2. **Pre-registered bars at T1, before data.** Beat the do-nothing floors (`ha`, `ha0`, `ha0_ext`; for v7F the echo),
   all four metric families, paired episode-cluster bootstrap — and read against the **right variance**: refav1's iCEM
   samples, so every lever needs an **inference-seed replicate** (its measured floors run 0.0047–0.2956 m depending on
   the panel); neither line has a **training-seed** floor, so a separated CI from one run is necessary, never sufficient.
3. **The REFe review checklist** (`raw/SURVEY_REFE_LESSONS.md` §2), which found blocking defects in every one of six
   rounds that the previous round's tests had passed: primary tables row by row; md5-frozen code; displacement of every
   pretrained component vs the published model with SELF/JITTER controls; hooks on every cross-attention; one identity per
   derived quantity; what every label term's producer writes and whether it varies **within** the candidate set; a
   regression arm per guard; count what every filter removes; rehearse the real launch scripts; evaluate the first
   checkpoint through the real eval path with the do-nothing floor and an oracle / random / pick triple.
4. **Decompose before building.** REFe's single E-6 table (0 extra training) located ~47 PDMS of headroom in
   **selection**, not in the proposals. refav1 is the same shape — the search is healthy and the selector (the cost)
   picks the do-nothing plan — so its oracle / random / pick table over the iCEM population is the cheapest decisive read.
5. ⭐ **(added 2026-09-27 afternoon, from today's measurements) Raise the floor before claiming anything.**
   (a) Every T1 claim is read against the DAMPED floors (`damp50`, `kd_x`), not only `ha`/`ha0`/`ha0_ext` — on refav1's
   grid they beat `ha0_ext` by 0.16–0.26 m at 2 s and by **~1.9 m at 6 s**. (b) The "same inputs minus vision" floor —
   a readout fitted on t0 kinematics alone — is the bar for any learned head (it beats `kd_x` by a further 0.056 m on
   held-out episodes); a head that does not beat it has learned kinematics, not driving. (c) Read the claim inside
   GT-defined event strata (turn / stop / accelerate / hard windows) as well as pooled — the pooled 2 s mean is
   dominated by windows kinematics already solve (PUBLISHED: `2305.10430`, `2312.03031`). (d) Report 2 s AND 6 s.
6. ⭐ **Probe the trunk before training a head on it.** A ridge / MLP readout from the frozen state against the
   kinematic readout, fitted on TRAIN and scored on EVAL, with a shuffle control and n / d printed, costs ~1 h on the
   dev box and says whether a heads-only head has anything to decode (`…/2026-09-27-refav1-trunk-probe/`). v7's own
   E-WC2 is the same instrument for the selector; for option B's decoder it should be run on the S-W trunk before S-T.
7. **Never tune on a small panel.** refav1's 30 p4 arms (8 turn-dense episodes) chose a lateral behaviour the full grid
   punishes by 0.20 m cross-track; tuning happens on a split that is not the one the claim is read on.

## 6. REFe lessons that bite NOW

| REFe finding | v7F | refav1 |
|---|---|---|
| Frozen DINOv3 fed un-normalised images | ⛔ **present** (§3 #1) | clean (HF processor mean/std) |
| Positional scheme with no checkpoint counterpart | ⛔ present (`pos` at init) | clean (RoPE) |
| Declared-but-not-built levers | ⛔ LDAD | levers default 0.0 |
| Selection-bound: good candidates exist, the pick is bad | later | ⛔ **the core defect** (282/282 baseline) |
| A pooled estimator judging a within-scene choice | UNVERIFIED | ⛔ `cost_fidelity` |
| A statistic read from a label, not content | readout at random init | ⛔ `baseline_won_frac` 0.6631 vs true 1.0 |
| Argmax over a larger pool harvests the critic's bias | later | shape present |
| Replicate identity must include env vars | yes | yes |
| Operating-point statistics from a small sample | — | ⛔ standardiser from one batch |

Reusable REFe instruments: `refe/diag_rope.py` (displacement probe), `refe/model.py:214,262` (bit-identical RoPE),
`refe/load_dinov3.py`, `eval/selection_readout.py` + its mutation self-test, `eval/proposal_table.py`,
`code/switch_direction.py`. ⛔ Do not reuse `eval/write_result_e6.py:101-105` as is (hardcoded verdict prose).

## 7. Done tonight (all staged, none committed)

| artifact | where | verified |
|---|---|---|
| refav1: 305 stranded files banked from the dev box — 17 arm records, their logs and window dumps, the panel analysis; 278 others already at the tip (by md5) not duplicated; 16 corpus input slices deliberately excluded; 556 clip UUIDs rewritten as sha12 | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-close-the-gaps/raw/refav1_margin_banked_2026-09-27/` (+ `README.md`, `manifest.json`) | 307/307 index blobs == worktree; 0 UUIDs in banked text |
| This report + the three survey reports + the v7F launch-line probe | `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/` | blob-verified at staging |
| refav1 harvest: paired four-family deltas for the stranded arms, `kammshift` scored against its bar, `RESULT_HARVEST.md`; a second sha12 pass for 240 clip-id **prefixes** the first pass missed | `…/refav1_margin_banked_2026-09-27/{RESULT_HARVEST.md,harvest/}` | 316/316 blob-verified; 0 ids |
| refav1 full-grid pre-registration (before any data) | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refav1-fullgrid-loncomb3/SPEC.md` | verified 01:05:48 |
| v7F S0 at ViT-B/16 — code, raw results, RESULT | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-v7f-s0-b16/` | 24/24 blob-verified; 0 ids |

⚠️ Corrections I made along the way: a whole-tree blob probe first said **0 of 30** records were banked — wrong, because
I hashed with `--no-filters` (raw CRLF bytes) while the tree stores LF blobs; the md5 probe (13 banked) was right and the
mechanism is verified on one file. The first staging check read 0/0/0 because `git ls-files` has no
`--pathspec-from-file`; the count line exposed it and the re-check read 307/307. The refav1 survey counted **nine**
stranded records; the content check found **17**.

## 8. What I do next (mine, no decision needed) — and what needs YOU

**Mine (items 1–3 DONE 2026-09-27 ~01:10, see the UPDATE at the top):**
1. ✅ v7F S0 at B/16. 2. ✅ refav1 harvest + `kammshift` scored. 3. ✅ refav1 full-grid pre-registration.
4. **Run the refav1 full-grid confirmation** (2 arms) — on the dev box after a ~10 GB LAN copy of the 141-episode
   eval inputs to D: (C: has 30 GB free), gated on GPU headroom; or on Thor if the Master Mind confirms refcv7 does
   not need it in the next ~6 h.
5. Hand the Master Mind's gate agent the G-DVB inventories for `refa_v1_train.py` and `train_v6_staged.py`, so the
   binding gate covers both lines when it lands.
6. An exploratory p4 arm `wk7` + the `loncomb3` levers — can the best longitudinal config keep its left turns?

**Needs the PI:**
- refav1: arm `W_VEND` (target max(0, v0 + a0·T)); promote winning levers to defaults; the turn-vocabulary magnitude.
- v7F: what closes the 2026-08-31 hold, and which compute slot (Thor after refcv7, or the promised pod).

## 9. Corrections the register needs (for the Master Mind)
- `MODEL_REGISTRY.md:4724-4742` prints the void v7 T1 table as a measured floor.
- `GOALS_AND_CLAIMS.md:8946` says the metric-decode guard shipped; it does not exist.
- `CLAUDE.md:180-188` attributes the 14.3 % (now 9.5 %) replicate panel to v7-tiny; it was the REF-C tiny rig.
- `MODEL_REGISTRY.md:2095` still says `ccos` "IN PROGRESS" (delivered and refuted); registry §2.4 lacks the `a0_shift`,
  turn-asymmetry and perception rows.
- The refav1 "≈0.30 m seed floor" is cited to `D-REFAV1-SEED-GOAL-MISMATCH`; it was measured in `D-REFAV1-DRIVE-AB`.
- `PREREG_V7F.md` still quotes 336.5 M params (245.6 M measured) and says existing flags "do not exist".
- `CLAUDE.md:1297` names `launch_gate.py`, `SPEC_REFCV7.md:39` names `launch_gate_refcv7.py` and says it "landed";
  neither exists at `b3f7ea6f`.
