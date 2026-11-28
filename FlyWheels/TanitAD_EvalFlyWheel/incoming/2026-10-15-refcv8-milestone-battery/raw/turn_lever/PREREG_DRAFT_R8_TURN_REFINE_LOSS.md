# PRE-REGISTRATION (DRAFT): H-R8-TURN-REFINE-1, a near-horizon turn loss on refcv8's refined candidates

* **Status:** DRAFT, design only. Nothing in here has been trained, evaluated or launched. **Nothing launches without (1) a PI compute authorisation and (2) a launch-gate PASS on the exact argv.**
* **Written:** 2026-10-18, against the refcv8 launch tree `C:/lgs/Lfb47a58376/stack` (gate commit `fb47a58`). The running job `refcv8-r101-s0` on Thor is not touched by anything in this document.
* **Defect it targets:** `H-R8-TURN-UNDERCORRECT-1`. On sharp turns refcv8's refinement step gives back less of the bank's under-rotation than refcv7's does.
* **Tier of every result number below:** T1-style **open loop** on logged eval139 frames (own action), S2 panel, unless a line says otherwise. Not driving performance.
* **Owed by the landing session (I may not edit existing files):** register `H-R8-TURN-REFINE-1` in `Project Steering/GOALS_AND_CLAIMS.md` and the PI item in `PI_DECISION_QUEUE.md`.

---

## 0. What the PI must decide

| # | decision | why it is the PI's call |
|---|---|---|
| **PI-1** | Authorise the **eval-only Gate G0** (§5): about 3 to 5 battery runs, no training. | It needs GPU time on the dev box or Thor. Thor is busy with refcv8 until about **2026-10-20 evening (ESTIMATED, §8)**, and sharing Thor costs both jobs time (INHERITED: memory note "Thor GPU sharing is net-negative"). |
| **PI-2** | Authorise the **training arms** (§6): 3 arms minimum, 4 preferred, each 6,000 warm-start steps (about 16 h each on Thor, ESTIMATED), run **after** refcv8 finishes or on an A40/HF box. | Compute spend, and choosing the box. |
| **PI-3** | Only if G0-b (§5) reads positive: whether an **eval-time switch** (`modulate_base = False`) may change the deployed refcv8 configuration. | It changes the shipped planner without retraining. |

---

## 1. The measured defect (primary artifacts only)

All numbers are MEASURED: paired episode-cluster bootstrap, B 2000, one shared resample matrix over 139 episodes, two inference seeds. **This estimator answers only "would another draw of EPISODES say this?"** Each arm is one checkpoint from one training run, so nothing here says another training run would agree (`H-ESTIM-SEED-1`).

| quantity (sharp = GT \|yaw rate\| >= 0.15 rad/s; last segment 1.5 -> 2.0 s) | value [95 % CI] | artifact |
|---|---|---|
| Through-origin slope of turn-aligned heading error on GT heading, refcv7-50,400 | -0.0921 [-0.1765, -0.0255] | `raw/heading_drift/HEADING_DRIFT_step20000ema.md`, multiplicative-under-rotation table |
| same, refcv8 EMA-5k / 10.5k / 15k / 20k | -0.1302 / -0.1545 / -0.1409 / -0.1411 | same table |
| refcv8 minus refcv7: EMA-15k / EMA-20k | -0.0488 [-0.0867, -0.0176] SEP / -0.0490 [-0.0825, -0.0218] SEP | same table |
| Slope of the selected anchor before refinement, r8 / r7 (EMA-15k pairing, in-bank windows, n 335 windows / 41 episodes) | -0.2386 / -0.2714 | `raw/heading_drift/REFINE_SHRINK.md` §1 |
| Slope of the plan after refinement, r8 / r7 | -0.1032 / -0.0711 | same |
| Refinement effect r8 / r7 (refined minus unrefined) | +0.1354 / +0.2003 | same |
| **DIFF_EFFECT (r8 effect minus r7 effect)** | **-0.0649 [-0.0923, -0.0398] SEP** | same |
| **Recovery fraction** (1 - refined/unrefined slope), r8 / r7 / difference | **+0.567 / +0.738 / -0.171 [-0.248, -0.097] SEP** | same (DERIVED there) |
| Same-anchor subset: both models picked the SAME bank index, so the unrefined path is identical (n 127 / 37 ep) | DIFF_EFFECT -0.0286 [-0.0486, -0.0123] SEP; recovery 0.472 vs 0.577 | `REFINE_SHRINK.md` §1a |
| Before-refinement gap (r8 - r7, selected anchors) | +0.0328 [-0.0010, +0.0630], not separated | `REFINE_SHRINK.md` §1 |
| 5 to 6 s terminal chord: DIFF_EFFECT | +0.0134 [-0.0233, +0.0526], not separated | `REFINE_SHRINK.md` §5 item 7 |
| Fan coverage: a candidate within 10.3 deg of the GT terminal heading exists | r8 99.2 %, r7 100.0 % (n 262 / 32 ep) | `REFINE_SHRINK.md` §3 |
| Candidate vs selection split of the gap | CANDIDATE -0.0347 SEP (share 0.71); SELECTION -0.0141, not separated | `raw/heading_drift/FAN_VS_SELECTOR.md` §4 item 2 |
| Oracle ceiling (nearest-GT **refined** anchor; T0 diagnostic) r8 / r7 | -0.0826 / -0.0479 | `FAN_VS_SELECTOR.md` §4 item 3 |

**Two MEASURED facts that change how this has to be designed:**

1. **The raw weights swing by more than the EMA gap from one checkpoint to the next.** Raw sharp last-segment slope: RAW-5k -0.2335, RAW-10.5k -0.1365, RAW-15k -0.2174, RAW-20k **-0.1117**. **RAW-20k minus refcv7 is -0.0208 [-0.0704, +0.0183], not separated** (`raw/heading_drift/HEADING_DRIFT_step20000raw.md:498`; RAW-15k `HEADING_DRIFT_step15000raw.md:380`). So the -0.049 EMA gap is the average of a raw trajectory that oscillates by about ±0.1. A lever must therefore be read against a training replicate (§6), not against the episode CI alone.
2. **The comparison is at mismatched learning-rate phase.** The cosine LR of refcv7-50,400 had reached 5.3e-14, i.e. fully annealed. refcv8 at 15k / 20k is at 4.02e-5 / 3.30e-5 (logged LR of param group 0; peak 5.0e-5). refcv7 at its own 15k / 20k was at 4.16e-5 / 3.48e-5. Evidence class: MEASURED from trainer telemetry. Readers: `turn_lever/train_telemetry_read.py`, output in `turn_lever/train_telemetry_read.out`; inputs pinned by sha256 in `turn_lever/inputs_sha256.txt`. **refcv7 checkpoints at 15000 / 20000 / 30000 exist** (`E:/refcv7_eval_kit/ckpt/ckpt_15000.pt`, `ckpt_20000.pt`, `ckpt_30000.pt`; checked with ls). That makes a matched-phase control possible without any training (§5, G0-a).

---

## 2. Code map: where candidates, refinement and losses live

Source: launch tree `C:/lgs/Lfb47a58376/stack`; run argv from `E:/refcv8_eval_kit/ckpt/config.json` and `E:/refcv7_eval_kit/ckpt/config.json` (diff banked: `turn_lever/argv_diff_r7_r8.out`). Evidence class for this section: MEASURED-FROM-SOURCE (read, not run).

### 2.1 Candidate production (anchor bank, candidates)

* **Anchor vocabulary:** 117 anchors, control units `alat`, v0-conditioned. The bank is **bit-identical in both arms** (sha256 `51f930dc…` in both config.json files). Artifact constants: `kappa_cap 0.12`, `alat_v_floor 4.0`, `horizon_s 6.0`.
* **Residual prior** `ha0_ext_pose`. The bank is the vocabulary composed as residuals on a constant-(a0, kappa0) prior: `refc.py:2356-2373` (`_roll_residual_bank`). The prior is built in `refc.py:2306-2354` (`_residual_prior`).
* **refcv8 extension:** `refc.py:2486-2588` (`_r8_extend`). It tags the 117 base candidates (`refcv8_conditioning.py:225-264`, `tag_paths`: TURN at 30 deg or more), appends **32 allocated candidates** (`n_alloc 32`, top-k hypotheses, `refc.py:2509-2538`) and builds a per-candidate conditioning vector `phi`.
  * Base candidates DO receive `phi` (tag one-hots plus the tactical head's log p of that tag; no constraint values, because `base_constraints false`): `modulate_base: true` in config.json, `refc.py:2573-2574`.
  * In training, `phi` is zeroed per row with p = 0.15 (`refc.py:2575-2577`).
* **Selection:** argmax of the emitting pass's own confidence (`--f5-emitting-conf`). Extras can win only while `r8_emit_live` is set (`refc.py:2578-2583`; schedule `refcv8_train.py:421-434`, `emit_start 1000`).

### 2.2 Refinement: the residual applied to each candidate

* **Sampler:** `refc.py:2910-3157` (`_sample`). Control space, `sampler_infer_t 8`, 2 DDIM steps at eval. In training it makes ONE decoder call at t ~ U[0, 50) (`refc.py:3027-3028`, F1).
* **The refinement itself:** `x0_hat_n = x_n + du` (`refc.py:3090`), and the fan is rolled from `u0_hat` (`refc.py:3107`).
  * `du` is produced per decoder layer by the F3 cascade heads: `refc.py:2845` -> `models/refcv6_diffusion.py:334-339`.
  * Each head is an **unbounded `nn.Linear`**. The emitted value is the last layer's (`refc.py:2858`).
  * The query handed between layers is detached (`refc.py:2854`). So a loss on the final stage reaches only the last layer's parameters plus the shared scene paths (kv, cond, time MLP).
* **refcv8 changes on this path:**
  * `q = r8_mod(i, q, phi)` after every layer (`refc.py:2834`; `refcv8_conditioning.py:373-392`). It is zero-init, so it was a no-op at the warm start and is learned from there.
  * The last-layer query `q` is exported as `_r8_last_q` (`refc.py:2843`) and **feeds the sub-score critic without a detach** (`refc.py:3842`; `refcv8_conditioning.py:530-532`).
  * Base and extra candidates are decoded in two passes of the same weights (`refc.py:2719-2750`).
* **Lateral-prior dropout (X2a):** in training, kappa0 is set to 0 with p = 0.3 per row (`refc.py:2347-2353`; `--r8-lat-prior-dropout 0.3`). Never at eval.
* **Clamps:** F8 is off (no flag); `feasible_decode: false` (config.json); `du` has no bound. The only geometric caps are `kappa_cap` / `alat_v_floor`, which belong to the anchor artifact and are **shared bit-identically by r7 and r8**.

### 2.3 Losses on the refined candidates (which candidates, which terms)

| term | candidates | geometry | weight r7 / r8 | source |
|---|---|---|---|---|
| `loss_traj` | **only `a_star`** = the GT-nearest **unrefined base** anchor (L2 over valid slots, base 117 only in r8) | **L1 on (x, y) positions**, 8 slots 0.5 to 6 s | 1.0 / 1.0 | `refc_v3_train.py:5450-5453` (`a_star`), `:5466-5468`, weight `TRAJ_WEIGHT` `scripts/refc_train.py:76`, sum at `:5602` |
| anchor CE (focal) | target `a_star` | classification | 1.0 / 1.0 | `refc_v3_train.py:5460-5465` |
| F3 cascade | `a_star` at every **non-final** stage | position L1 + focal CE | on / on | `refc_v3_train.py:5844-5876` |
| x0 / `u0` control loss | `a_star` | control space | **0 / 0** (`--w-u0` OFF_BY_DEFAULT, `--f6-w-u0-zero`) | `refc_v3_train.py:5785-5823`; config.json `effective_weights` |
| `r8_alloc_l1` | best **allocated** candidate of the GT hypothesis (winner-take-all) | position L1 | — / **1.0** | `refcv8_train.py:672-689` |
| `r8_sat` | **allocated** candidates only | hinge: 6-s progress plus **6-s terminal heading, 15 deg tolerance** | — / 0.1 | `refcv8_train.py:690-696`; `refcv8_conditioning.py:827-845` |
| `r8_subscore` | **all** 149 candidates (BCE on 4 critics; reads the non-detached last-layer `q`) | terminal-heading class, 10 % progress, 15 deg terminal heading, lat3 tag | — / 0.5 | `refcv8_train.py:705-712`; `refcv8_conditioning.py:537-557` |
| `r8_cons`, `r8_v9_cons` | constraint heads (not the plan) | — | — / 0.05, 0.05 | `refcv8_train.py:661-665`, `:640-646` |

**What the code says:**

1. **No term anywhere supervises the 0.5 to 2.0 s heading or curvature of a refined candidate.**
   * The base refinement is trained only by a 6-s position L1 on one candidate per row.
   * The only heading terms in the codebase act on the **6-s terminal chord**, and on **allocated** candidates (`r8_sat`) or as a critic target (`r8_subscore`).
   * The terminal chord is exactly where the measured DIFF_EFFECT is **not separated** (+0.0134). The defect is at 2 s and below, where nothing is supervised.
   * Near-horizon heading recovery is therefore an **unsupervised by-product** of the position L1, in refcv7 and refcv8 alike. That is consistent with both oracle ceilings under-steering (-0.083 / -0.048).
2. **The supervised candidate is not the emitted one.** Loss goes to `a_star` (nearest unrefined base anchor). On sharp turns the selector agrees with the oracle anchor on only 0.280 (r8) / 0.283 (r7) of windows (`FAN_VS_SELECTOR.md` §4 item 5). So the refinement of the plan that is actually emitted is learned by generalisation from `a_star`.
3. **refcv8 dropped or down-weighted no refcv7 loss term.**
   * Every refcv7 term with non-zero effective weight (`--w-agent`, `--w-box3d`, `--w-map-hires`, `--w-tac-v6`, all 1.0) has the same weight in refcv8. `--w-r7-wta` / `--w-r7-scorer` are 0 in both.
   * The argv diff has **no refcv7-only flag** (`turn_lever/argv_diff_r7_r8.out`).
   * ⇒ **The lever "restore a refcv7 loss term that refcv8 dropped" is ELIMINATED by source.**
4. **What refcv8 changed on the refinement path:**
   * (a) `phi` modulation of the BASE candidates;
   * (b) three new gradient sources into the same last decoder layer: `alloc_l1` at 1.0, `subscore` at 0.5 through the non-detached `q`, and `sat` at 0.1;
   * (c) kappa0 dropout 0.3;
   * (d) a cosine warm RESTART: fresh AdamW, warmup 1000, peak LR again;
   * (e) training corpus 4,369 -> 4,572 clips (`--v2-cache` / `--vis1-sidecar` diff), plus the v9 nav/speed inputs with their dropouts.
5. **Magnitude (MEASURED, trainer telemetry):**
   * At refcv8 steps 19,000 to 20,500 the weighted `r8_alloc_l1` is 0.806 against `loss_traj` 0.794. The extras' winner-take-all position loss is as large as the base trajectory loss.
   * `r8_subscore` contributes 0.5 × 0.339 = 0.17; `r8_sat` contributes 0.1 × 0.253 = 0.025.
   * In-training EMA eval `traj` at 20.5k to 22k reads 0.5154 (n 3). refcv7's final `eval_traj` reads 0.5258 (n 6).
   * So **the objective the run optimises cannot see the defect**: matched-anchor position L1 is at parity while the sharp-turn heading slope is worse.
   * ⚠️ These are trainer-log values on 8 eval batches with no CI. Descriptive only, never a result.

---

## 3. Levers, ranked by expected effect on the recovery gap

| rank | lever | code evidence | expected effect | how it is attributed |
|---|---|---|---|---|
| **1** | **A1: a \|yaw\|-weighted 0.5 to 2.0 s chord-heading + heading-increment (curvature × arc) L1 on the refined `a_star` candidate** | §2.3 items 1 and 5: the defect sits where no term acts, and the position objective is blind to it | ESTIMATED: aimed at the whole in-bank DIFF_EFFECT (-0.065), and can go past refcv7 into the shared under-rotation. No dose-response measured yet. | Training arm vs a same-flags continuation control plus a replicate (§6) |
| **2** | **B: keep refcv8's additions off the base refinement path.** B1: `modulate_base = False` (phi only on extras). B2: detach `q` into `SubScoreHeads`. | §2.2: phi enters every base layer (`refc.py:2834`); the critic shares the last-layer `q` with the refinement head (`refc.py:2843`, `:2845`, `:3842`) | UNKNOWN. These are the only refinement-path differences between r7 and r8, but their size is unmeasured. | **B1 can be attributed with zero training** (G0-b, §5): `phi = 0` is in-distribution because the 0.15 condition-dropout trains exactly that state. B2 needs training. |
| (control, not a lever) | **C: finish the cosine.** The comparison is mid-schedule vs fully annealed; raw checkpoints swing by ±0.1 | §1 facts 1 and 2 | Possibly large; RAW-20k is already not separated from refcv7 | G0-a (matched-phase refcv7) and G0-c (refcv8 final), no training |
| eliminated | refinement capacity / clamp | `du` unbounded (`refcv6_diffusion.py:334-339`, `refc.py:3090`), F8 off, `feasible_decode false`. The kappa / alat caps are shared bit-identically by both arms, so they cannot produce a between-arm difference. | — | — |
| eliminated | restore a refcv7 term | §2.3 item 3 | — | — |
| 4 (not in this prereg) | kappa0 dropout 0.3 (`refc.py:2347-2353`) | Training-only input shift; the **sign** of its effect on turn recovery cannot be derived from source | HYPOTHESIS | Training ablation only |

**Why A1 ranks first.** The attribution says the refinement term dominates: DIFF_EFFECT is about 2× GAP_refined, the selection share is not separated, and coverage is saturated. A1 is the only lever that acts **directly on that term**, and it is the only one with a known mechanism, namely that the target quantity is unsupervised. B could be cheaper (B1 needs no training if G0-b is positive), but there is no measured size behind it yet. That is why G0-b runs first and can **re-rank** A1 and B1 (rule in §5).

---

## 4. Hypothesis

**H-R8-TURN-REFINE-1.** Adding to the training objective a near-horizon turn term on the refined `a_star` candidate, weighted by GT \|yaw rate\| and zero on straights, will raise refcv8's sharp-turn refinement recovery fraction and its emitted-plan heading slope. The rise is measured against a same-flags continuation control, beyond that control's training-replicate floor, and to at least refcv7-50,400's level, with no regression in the four metric families.

Admissibility:
* The term uses GT future yaw **only as a training label** (PI 2026-08-03: labels may use ego). It adds **no inference input**, so the vision-only rule and goal/situation disjointness are unaffected.
* Corpus is refcv8's own (`refcv8-b1-416x1024-train-4572`). That is NOT the canonical parity corpus `physicalai-train-e438721ae894`; this arm inherits the choice already made for refcv8, and every arm here shares it.

---

## 5. Gate G0: zero-training controls (read these before any training compute)

All of G0 uses the frozen instrument: `refine_shrink.py` sha256 `cce81102…`, `heading_drift.py` `27c2c7c9…`, `fan_vs_selector.py` `3376bb19…`, all in `raw/heading_drift/`. Same S2 panel (4,754 windows / 139 episodes), same two inference seeds, same bootstrap matrix (rng seed 29). The known-value controls of `REFINE_SHRINK.md` §0 must re-PASS (GT-as-plan slope = 0; mispaired control differs from paired; banked values reproduce exactly) before any G0 number is read.

| id | what | cost | committed reading |
|---|---|---|---|
| **G0-a** matched-phase control | refcv7 RAW-15000 and RAW-20000 (LR phase within 5 % of refcv8's 15k / 20k, §1 fact 2) vs refcv8 RAW-15k and RAW-20k through `refine_shrink.py` | 2 new battery runs (refcv7 15k / 20k); refcv8 raw trees exist. ⚠️ UNVERIFIED whether the raw trees carry the `sel_bank` sidecars `refine_shrink.py` needs; if they do not, 2 more runs. | **(i)** r8 - r7 recovery separated-negative at BOTH points ⇒ the r8-specific deficit is real; A1 is primary as planned. **(ii)** Not separated at either point ⇒ `H-R8-TURN-UNDERCORRECT-1` is downgraded to "not distinguishable from LR phase" in the register; A1 **still runs**, aimed at the shared under-rotation, with the absolute bar of §7 unchanged. **(iii)** Mixed ⇒ report as mixed; A1 runs. *refcv7 had no EMA (INHERITED, memory "refcv8 bars scored on EMA"), so this is raw vs raw. Raw swings by ±0.1, so a sign must agree at both points before it counts.* |
| **G0-b** base-modulation attribution (Lever B1) | refcv8 EMA-20k (and the final EMA when it exists) with `model.core.decoder.r8_cfg.modulate_base = False` set after load, recorded in `overrides`, vs as-trained, same windows / seeds | about 2 battery runs plus a 4-line override in `taniteval/tools/refcv3_arm.py`, mirroring Route A (`REFINE_SHRINK.md` §4). `modulate_base` is a declared field, so `strict_fields` allows the set (`config_hygiene.py:59-68`). | **Promote B1 above A1** iff recovery rises by **≥ +0.10 with the CI excluding 0** AND the sharp last-segment slope improves with the CI excluding 0. In that case the eval-time switch becomes a zero-training candidate fix and goes to PI-3, after a four-family read on the same run. Otherwise B1 is demoted and A1 stays primary. |
| **G0-c** in-flight anneal | refcv8 final (48,852) EMA through the same instrument | free once the run ends (battery chain) | If the final EMA reaches **both** recovery ≥ refcv7's value on the same window intersection **and** slope ≥ -0.0921, the defect closed by annealing: the A1 arms run as an **excellence** arm against the stretch bar (§7.3), and the r8-vs-r7 deficit claim is marked closed. Otherwise A1 runs as planned. |

---

## 6. Training arms (PI-2)

**Common to all arms.**
* Warm start `W` = refcv8-r101-s0 **final EMA weights** (48,852), exported as a model state dict and loaded with `--init-from W`. `refcv8_train.py:914-950` accepts a refcv8 source. ⚠️ UNVERIFIED that an EMA-as-model export exists; if not, it is a roughly 10-line exporter with an md5 record.
* The fallback, if the PI wants it before the run ends, is EMA-20k; G0-c is then not available at launch.
* argv identical to the refcv8 run (`turn_lever/argv_diff_r7_r8.out`) except:
  * `--init-from W --steps 6000 --warmup 300 --lr 2e-5`: a short cosine to 0, so **every arm ends annealed** and the LR-phase confound of §1 cannot reappear between arms. The LR is ESTIMATED, a quarter of the original peak.
  * `--ema-decay 0.999` (horizon about 1k steps). At 0.9998 about 30 % of `W` would remain in the EMA at step 6,000, diluting any lever. The bars are read on EMA, with raw reported beside (PI 2026-10-16, INHERITED).
  * `--r8-alloc-emit-start 0` and `--r8-tf-start 0.25 --r8-tf-end 0.25`. The emit and teacher-forcing schedules restart at step 0 of a warm start (`refcv8_train.py:349-355`, `:421-434`). Without this the extras would stop emitting for 1,000 steps and teacher forcing would jump back to 1.0: a different model from `W`, in every arm.

| arm | lever | seed | role |
|---|---|---|---|
| **CTL-s0** | none (`--w-r8-turn-head 0`) | 0 | the control: continuation plus anneal without the lever |
| **CTL-s1** | none | 1 | **replicate**: the training-run floor (`H-ESTIM-SEED-1`; the same flags measured "separated" on 14.3 % of cells on the tiny rig). A different seed, because a same-seed rerun on this trainer is already a fresh draw (INHERITED, memory "same-seed rerun is a fresh draw"). |
| **A1-s0** | `--w-r8-turn-head 0.2` (§6.1) | 0 | the lever |
| A1-s1 (preferred) | same | 1 | lever replicate. Without it, a PASS is "beyond the control's floor" only. |
| REG (optional, PI) | the same term with the GT turn target scaled ×0.7 | 0 | deliberate regression: must make the slope MORE negative than CTL-s0, otherwise the arm cannot move the metric in either direction and a FAIL is uninformative |

### 6.1 Exact change (pseudo-diff; reference implementation tested on CPU)

Reference implementation and analytic checks: `turn_lever/turn_head_loss_ref.py`. **ALL PASS, 10 / 10** (`turn_lever/turn_head_loss_ref.out`). Checks:
* plan == GT gives exactly 0;
* a 0.7×-curvature plan gives the literal **2.7215495**;
* the gradient raises the curvature of an under-rotated plan;
* chords shorter than 0.5 m are gated and have finite gradients;
* the sign-flip **mutation** is caught.

New module, `stack/tanitad/refs/turn_head_loss.py`, a copy of `turn_head_loss_ref.py` without `main`:
```
def turn_head_loss(plan, gt, slot_valid, w_win, *, n_slots=4, min_chord_m=0.5, w_inc=0.5, sigma_h=radians(2.0)):
    # chord k = slot k-1 -> slot k for slots 0.5/1.0/1.5/2.0 s (the instrument's segments)
    e_h = |wrap(atan2(chord_plan) - atan2(chord_gt))|                         # rad, per chord
    e_i = |wrap(dh_plan - dh_gt)|, dh[k] = h[k] - h[k-1]                     # curvature x arc, rad
    ok  = |chord_plan| >= 0.5 m  &  |chord_gt| >= 0.5 m  &  slot_valid
    return (sum w*e_h*ok + w_inc * sum w*e_i*ok_i) / max(sum w*ok, 1) / sigma_h
def window_weight(yaw_rate_gt, v0, cap=3.0):   # 0 below 0.02 rad/s, 1 at 0.15, capped at 3; 0 when v0 < 2 m/s
```
`stack/scripts/refc_v3_train.py`:
```
@@ g8 argument group, after :12247 (--r8-tf-end)
+    g8.add_argument("--w-r8-turn-head", type=float, default=0.0,
+                    help="H-R8-TURN-REFINE-1: 0.5-2.0 s chord-heading + increment L1 on the refined a_star candidate")
+    g8.add_argument("--r8-turn-head-cap", type=float, default=3.0)
@@ effective-weights registry (the table holding "w_r7_wta" near :2854): register --w-r8-turn-head so config.json
@@ stamps declared/requested/effective and builds_graph (the 42/138 lesson: a weight that never builds a graph is refused)
@@ model attribute set next to `model._w_u0 = ...` (:9156)
+    model._w_r8_turn_head = float(getattr(args, "w_r8_turn_head", 0.0) or 0.0)
@@ compute_losses_v3, immediately AFTER the loss assembly at :5602-5606
+    _w_th = float(getattr(model, "_w_r8_turn_head", 0.0))
+    if _w_th > 0.0:
+        _yaw = _wrap(fut_ext[:, 19, 2] - pose_last[:, 2]).abs() / 2.0          # |net heading change 0->2 s| / 2 s
+        _ww = turn_head.window_weight(_yaw, v0.reshape(-1), cap=float(args_cap))
+        l_th, n_th = turn_head.turn_head_loss(recon, traj_tgt, slot_valid, _ww)   # recon = anchor_traj[ar, a_star] (:5466)
+        loss = loss + _w_th * l_th
+        extra["r8_turn_head"] = l_th.detach(); extra["r8_turn_head_rows"] = float(n_th)
```
* `fut_ext[:, 19]` is t + 2.0 s: the trainer reads slot h from `fut_valid[:, h - 1]` (`refc_v3_train.py:5439`), so horizon 20 ticks = index 19. Column 2 is yaw, as `refcv8_conditioning.py:283-312` reads it.
* ⚠️ The training weight uses the **net** 0 -> 2 s heading change. The instrument bins on its own GT yaw-rate definition. The two need not coincide, so the primary metric is read **only on the instrument's bins**.
* Gradient path: `recon` -> `anchor_traj` -> roll of `u0_hat` -> last-stage `du`. Reachability of `anchor_traj` is pinned for the single `control_head` (`refc.py:3620-3627`, `tests/test_u0_control_head_reachability.py`). ⚠️ For the F3 cascade's last head (`cascade.control_heads[-1]`, the head this build emits through) it is UNVERIFIED, which is why that test is owed below.

**Dose (ESTIMATED, rule fixed now).**
* `sigma_h = 2 deg`, `w = 0.2`. Sharp-window heading MAE at 0.5 to 2 s is 0.9 to 5.4 deg (`HEADING_DRIFT_step20000ema.md` per-segment table), so the term should start at roughly 0.3, against `loss_traj` of about 0.8.
* **Rescale rule:** if the measured mean of `w × L_turn` over the first 50 steps of a CPU or launch-gate smoke falls outside **[0.10, 0.50] × loss_traj**, `w` is rescaled **once, before launch**, to hit 0.25 × loss_traj. The rescale is recorded in config.json. No rescale after launch.

**Tests owed before the launch gate** (literal expectations; mutation must go RED):
* the analytic checks of `turn_head_loss_ref.py`;
* a forward/backward test showing `--w-r8-turn-head > 0` puts non-zero gradient on `cascade.control_heads[-1]`, and exactly 0 with the weight at 0 and the term absent;
* a registry test showing config.json stamps the term.

### 6.2 Cost

* **Rate:** MEASURED **9.76 to 9.82 s/step** on Thor for this exact argv (refcv8 steps 15k to 22k, `elapsed_s`; `turn_lever/train_telemetry_read.out`).
* **Per arm:** 6,000 steps ≈ **16.3 h**. Minimum 3 arms ≈ **49 h**; 4 arms ≈ **65 h**; plus one battery per arm (2 inference seeds). Battery wall time is UNVERIFIED here.
* **From scratch:** about 48,852 × 9.8 s ≈ **133 h per arm**. Not proposed.
* **When Thor is free:** refcv8 was at step 22,250 in the latest telemetry; 26,602 steps remain ≈ 72 h, so it finishes about **2026-10-20 evening** (ESTIMATED).
* Do not co-run on Thor.

---

## 7. Metrics, estimator, and BOTH outcomes committed in advance

**Estimator:** paired episode-cluster bootstrap over the 139 eval episodes, B 2000, ONE resample matrix shared by every arm and contrast (rng seed 29, as in `refine_shrink.py`), 2-inference-seed mean.
* **What it answers:** "would another draw of EPISODES say this, models fixed?"
* **Training-run variance:** answered only by the CTL-s1 / A1-s1 replicates.
* **Inference variance:** answered only by the per-seed rows.

Every contrast prints `n` windows / episodes. The recovery fraction is computed on the intersection of windows that are in-bank in every arm of the contrast.

### 7.1 Primary (same S2 windows, same instrument as `REFINE_SHRINK.md`, EMA weights, raw beside)

* **P1:** sharp (≥ 0.15) last-segment through-origin slope of the emitted plan (`hd.shrinkage`, `heading_drift.py:352`).
* **P2:** sharp last-segment refinement recovery fraction, 1 - refined/unrefined slope (`refine_shrink.py`).
* Also reported, not bar-bearing: segments 2..K, curved (≥ 0.02), the same-anchor subset (§1a), and the 5 to 6 s chord.

### 7.2 PASS: all of the following

1. **Lever effect, P2:** recovery(A1-s0) - recovery(CTL-s0) **≥ +0.10**, paired CI lower bound > 0, **and** greater than **2 × \|recovery(CTL-s1) - recovery(CTL-s0)\|**.
2. **Lever effect, P1:** slope(A1-s0) - slope(CTL-s0) > 0, CI lower bound > 0, and greater than 2 × \|CTL-s1 - CTL-s0\| on P1.
3. **Reference bar:** recovery(A1-s0) ≥ refcv7-50,400's recovery **recomputed on the same window intersection** (0.738 on the EMA-15k pairing), **and** slope P1(A1-s0) ≥ **-0.0921**.
4. **Inference floor:** items 1 and 2 hold in sign on inference seed 0 AND seed 1 separately.
5. **The refinement function itself moved:** on the same-anchor subset (A1-s0 vs CTL-s0 picking the same index), DIFF_EFFECT ≥ 0 with the CI lower bound > -0.01.
6. **No regression in any family (§7.4).**
7. If A1-s1 ran: items 1 and 2 hold for A1-s1 vs CTL-s1 in sign.

### 7.3 Stretch bar (reported, never PASS-defining)

P1 ≥ **-0.05**, i.e. about half of refcv7's own under-rotation. The shared defect is the real target: even the refcv7 oracle ceiling sits at -0.048.

### 7.4 Secondary: four metric families, no-regression bar

Each family is A1-s0 minus CTL-s0, EMA, 2-seed mean, on S2 and S6, read from the milestone battery's `PAIRED_READ`. The tolerances below are fixed now; **the PI may tighten them before launch, never after.**

| family | metric | bar |
|---|---|---|
| ADE | ADE 0-2 s (S2) / 0-6 s (S6) | upper CI ≤ **+0.03 m** / **+0.10 m** |
| LONGITUDINAL | speed MAE; along-track error; distance-keeping (headway / TTC) where the battery computes it | speed MAE upper CI ≤ **+0.05 m/s**; along-track ≤ **+0.05 m** (S2) / **+0.15 m** (S6) |
| LATERAL | heading MAE (all; and **straight < 0.02** separately), curvature MAE, yaw-rate MAE, cross-track | heading MAE upper CI ≤ **+0.05 deg**, all and straight; curvature / yaw-rate / cross-track upper CI ≤ **+5 %** of CTL-s0's value |
| TACTICAL | tactical decision correctness (lat / lon heads vs v7.2 / v9 labels; selected vs executed manoeuvre) | lower CI ≥ **-1.0 pp** |
| STRATEGIC | — | **Not computable on this arm.** `--no-strategic`, and nav comes from the v9 labels. Reported per family as N/A with that reason. |
| perception (guard) | box3d AP@2 m, map-hires | the run's own `b1` guard must not fire |

⚠️ UNVERIFIED: the exact `PAIRED_READ` key for each row above. If a family has no instrument in the battery, it is a **work item that blocks PASS**, not a silent omission (four-families rule).

### 7.5 FAIL: what each failure means, and the next lever (committed now)

| outcome | reading | next lever (executed, not just named) |
|---|---|---|
| **F-A:** `r8_turn_head` fell by ≥ 30 % in training, but P2 / P1 did not move | the gain on `a_star` does not transfer to the **selected** candidate (§2.3 item 2) | **A2:** apply the same term also to the **emitted** candidate (`sel_idx`) and to the top-4 GT-nearest base candidates sharing GT's lat3 tag. Same arms and bars. |
| **F-B:** `r8_turn_head` did not fall | the optimisation could not use the term | **A1 ×4 dose**, with the term also on the F3 non-final stages; then check `du` magnitude on sharp windows against CTL |
| **F-C:** P1 and P2 PASS but a secondary family regresses | a trade-off | ramp start at 0.06 rad/s instead of 0.02, half dose. The trade-off goes to the PI with both readings. |
| **F-D:** CTL-s0 alone closes the gap to refcv7 (CTL vs refcv7 not separated on P1 and P2) | the deficit was LR phase | Retract the r8-specific claim. A1 is still judged on its own contrast vs CTL (items 1 and 2); the excellence target is the stretch bar. |
| **F-E:** \|CTL-s1 - CTL-s0\| ≥ the A1 effect (INCONCLUSIVE) | below the training-run floor | **More seeds, not more windows**: 2 more of CTL and A1 |
| **F-F:** the REG arm does not move the slope negative | the arm cannot move the metric at all | Stop the A family. Pivot to **bank coverage** of tight turns at 1.5 to 2.0 s (`REFINE_SHRINK.md` §5 lever 2). That changes the parity key, so it is a PI item. |

---

## 8. UNVERIFIED and limits (stated once)

* Whether `kappa_cap 0.12` binds on any sharp window. This only matters for the shared under-rotation; it cannot explain r8 vs r7. Not counted.
* Whether an EMA-as-model export exists for `--init-from` (§6). Whether the refcv8 raw battery trees carry the `sel_bank` sidecars (G0-a). The exact `PAIRED_READ` keys per family (§7.4). Battery wall time.
* The continuation step rate assumes it equals the run's measured 9.8 s/step; the extra term is a few small tensor ops. ESTIMATED.
* The training-loss values in §2.3 item 5 are trainer telemetry on 8 eval batches: descriptive, never admissible as results.
* The sharp-window share (7.7 % of S2 eval windows, 368 / 4,754, `HEADING_DRIFT_step20000ema.md` census) is an eval-set number. The train-set share is UNVERIFIED.
* Everything in §1 is one checkpoint per arm from one training run. `H-ESTIM-SEED-1` applies to the defect itself, which is why §6 carries replicates.

---

## Deliverable manifest (this package)

All files live in `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/raw/turn_lever/`. **This is the ONLY copy until the session's landing tool lands the package.**

| file | what |
|---|---|
| `PREREG_DRAFT_R8_TURN_REFINE_LOSS.md` | this pre-registration |
| `turn_head_loss_ref.py` | reference implementation of L_turn plus 10 analytic / mutation checks (CPU) |
| `turn_head_loss_ref.out` | its output: ALL PASS |
| `argv_diff_r7_r8.py` / `.out` | refcv7 vs refcv8 argv and effective-weight diff (read-only on the two config.json files) |
| `train_telemetry_read.py` / `.out` | LR phase, loss magnitudes, Thor s/step from the two runs' metrics jsonl |
| `thor_step_rate_read.py` | step-rate reader (output appended to `train_telemetry_read.out`) |
| `inputs_sha256.txt` | sha256 of the four input files read (two metrics jsonl, two config.json) |
