Verification complete — 15 refutations across 47 checked claims. Here is the adversarial finding list and the corrected report.

---

# ADVERSARIAL VERIFICATION — IDM stream. VERDICT: **ACCEPT WITH 15 CORRECTIONS; 2 PLAN ITEMS ARE NOT EXECUTABLE AS WRITTEN**

47 numeric/structural claims re-derived from raw artifacts. **32 confirmed exact.** The §2 headline table is fully sound. The failures cluster in the *plan* and in three inherited splices.

## A. REFUTATIONS THAT CHANGE A DECISION

**R1 ⛔ Plan ① and ② are NOT executable — the banked JSONs contain ZERO per-window predictions.**
MEASURED (mine, programmatic scan for any numeric array >200 elements):
`…/2026-07-27-fleet-sync-idm-steer/raw/idm5_ensemble.json` → **0 arrays**; `…/2026-07-27-idm-v3/results/compare_v3.json` → **0**; `…/results/arms_v3.json` → **0**. All three hold aggregates only (`r2/rho/mae/medae/rmse/gt_std/n/per_domain` + CI blocks). The claims *"predictions persisted in …{arms_v3,compare_v3}.json"* and *"Predictions are already in idm5_ensemble.json; only the 20-episode mask is needed"* are **FALSE**. Both need the head re-run over latents at `/root/idm2/lat` (pod-only, per `idm5_ensemble.json.meta.latent_dir_checked_for_leak`) — not in-repo. The dev box holds only `C:\Users\Admin\tanitad-data\eval\comma2k19-val-61c46fca8f7f` (comma; no PhysicalAI val cache), and the one run that used it **re-encoded** latents — its own `controls.encode_fidelity` warns that path reads comma speed R² **0.5545 vs 0.7590** on the pod substrate, *"a plausibility band, not an equality check"*. ⇒ ① is not "0 GPU-h / 3 h", ② is not "0 GPU-h / 1 h".

**R2 ⛔ "The measured channel ordering matches [the physics] without being fitted" is REFUTED on like-for-like slices.**
The ordering `yaw +0.9035 → speed +0.8651 → long_accel −0.2398` splices **PhysicalAI-only yaw** against **pooled speed** and **pooled long_accel**. MEASURED, same head, same 4,195 windows (`idm5_ensemble.json` → `a0_on_these_windows`):

| slice | speed | yaw | ordering |
|---|---:|---:|---|
| pooled | **+0.8651** | +0.8108 | speed > yaw — **inverted** |
| PhysicalAI | **+0.9070** | +0.9035 | speed > yaw — **inverted** |
| comma2k19 | **+0.7590** | +0.3308 | speed ≫ yaw — **inverted** |

On **every** matched slice A0's speed beats its yaw. Only "long_accel worst" survives. (For the *ensemble* it holds pooled and pai but inverts on comma: yaw +0.6948 < speed +0.7453.) The physics itself (Nistér 2004; Longuet-Higgins & Prazdny 1980) is untouched and stays PUBLISHED — the claim that **our measurement independently confirms it** is not supported. Splice originates at `IDM_V3.md:151-153`.

**R3 ⛔ *"no caller in the repo uses the `observable` mask"* is FALSE at HEAD** — the exact failure class the standard warns about. Second probe found the consumer: `…/2026-07-27-anchor-settlement/code/admissibility_consequence_61c.py:14-15`, whose own docstring reads *"the first consumer of `comma2k19.yaw_rate_from_heading` / `heading_admissible_centers`. The API is exercised on real corpus data here"* — calls at `:146,150,153,187,197`. Also `stack/tanitad/data/comma2k19.py:322,339` (definitions), `:370` (internal call), `stack/tests/test_yaw_admissibility.py:25-27,46,55,57,90`. The quote is from the 07-27 rescore JSON and went stale **the same day**.

**R4 ⛔ Plan ④'s premise is incomplete — the ensemble is not uniformly better than A0.** `ensemble_3seed.paired_vs_a0`: yaw + steer separated-better on both corpora ✓, but **speed pooled Δ +0.2287 [−0.377, +0.835]** and **comma speed Δ +0.4262 [−0.348, +1.218]** — nominally **worse** on the one channel the YouTube pipeline ships as `primary`. "Not separated on either" is true but omits the sign.

**R5 ⛔ "comma2k19 is flat from rung 200" is ESTIMATED, not MEASURED, and omits steer.** `idm4_steer.json` carries `paired_vs_a0` per rung but **no rung-to-rung paired comparison** — no estimator on any rung delta. Comma **steer rises monotonically +0.745 → +0.807 (+0.062)** over 200→757, with speed +0.051 and yaw +0.025. Excluding "more data" from the plan on this basis is not decision-grade.

**R6 ⛔ MODEL_REGISTRY carries NO `idm_head` row, and its IDM headline still quotes a WITHDRAWN number.** Two probes: `grep "idm_head\|idm-head"` → empty; `grep -i idm` → only §8.1 #6 (line 1851) and §10 (line 1924, Branch-B encoder). So v1/v3/v4_steer/v4_steer_ens3 have **no registry entry** — every §2 headline is admissible as raw-eval-JSON only. Worse, **§8.1 #6 line 1852 still reads *"reaches PhysicalAI held-out speed R² 0.930"*** — that is A2 (`in_corpus_heldout_paival`, 0.929725), which `LEAKY_CACHE_AUDIT.md:152` **withdrew as 77.5 % leaked**, and §8.1 #6 is **not** among the sites the audit's fix-table lists as corrected. ⇒ integration escalation.

## B. CORRECTIONS THAT CHANGE A NUMBER OR A CITATION

| # | claim | verdict |
|---|---|---|
| **R7** | *"3 of 4 pre-registered bars FAIL"* (idm-v2) | ⛔ **4 of 4 vs A0.** `IDM_V2_RESULTS.md:33-36` — yaw FAIL, speed **FAIL vs A0** (PASS only vs the v1-recipe control B0), long_accel FAIL, steer FAIL. The doc never says "3 of 4". |
| **R8** | *"≈92 % of the parity ceiling"* (YouTube pilot) | ⚠️ **Not in the artifact.** `results_youtube_pilot_downstream.json:128` → `"fraction_of_ceiling_speed_r2": null`. 92 % is derived in `NOTE.md:159-162` using the **parity** FLOOR −0.4387 / CEILING 0.6507 (`RESULTS_idm_parity_validation.md:28`) — *not* the pilot's own floor −0.520 quoted in the same sentence. Recomputed with the pilot's floor: **0.925**. Both real; juxtaposing them unflagged is misleading. |
| **R9** | *"2 domains × 2 metrics × 8 seeds"* | ⚠️ **5 + 3, not 8 per condition.** `RESULTS_idm_downstream_ablation.md:21` comma = 5 seeds, `:23` rig-B = 3; `meta.seeds = 5`. Range **0.917–0.984 is correct** (0.965/0.984/0.960/0.917). Inherited verbatim from the doc's own line 56. |
| **R10** | `stack/tanitad/data/comma2k19.py:169-173` | ⛔ **Wrong lines** (those are inside a heading-mode error string). **Substance is CORRECT.** Real: `:553` `speed = np.interp(t, *can_speed)` (CAN) · `:556` `accel = np.gradient(speed, t, …)` · `:560-561` `yaw = arctan2(enu_v…)`, `v = norm(enu_v)` (GNSS) · `:563` `poses = column_stack([…, yaw, v])`; pickup at `stack/scripts/idm_head.py:66` `speed = poses[t,3]` (GNSS) vs `:68` `accel = actions[t,1]` (CAN). |
| **R11** | *"`ci.py` md5 pinned to HEAD `c92618a0`"* | ⚠️ **Conflates md5 with a git commit.** `estimator_md5_pinned = c92618a02b36f8191a581fb74a491a8d` is the **md5 of the file**; repo HEAD is `7582253d`. Same meta block records a **stale sibling** at `ef925f06…` and warns `idm2_lib.py:19` / `idm3_arms.py` each `sys.path.insert(0,'/root/taniteval')`, defeating the pin — held only by import order + a runtime assert. |
| **R12** | A0 rescore *"genuinely held-out"*; yaw *"+0.0000 / −0.0004 [−2.2876, 0.1485]"* | ⚠️ Disjointness is **episode_id-level**, not content (`leak_probe.disjoint_here = true`) — weaker than the program's own content standard. The CI **[−2.2876, 0.1485] belongs to the ON arm only**; the OFF arm's is [−0.0042, +0.0032]. And `controls.encode_fidelity` explicitly declines the 0.5545-vs-0.7590 inference: *"a DIFFERENT substrate … a plausibility band, not an equality check"* — so the gap cannot be attributed to held-out-ness. |
| **R13** | 07-24 YouTube pilot row, class MEASURED | ⚠️ **Must carry ⛔ UNVERIFIED.** `LEAKY_CACHE_AUDIT.md:355` lists `youtube-idm-pilot` as UNVERIFIED. (Its stated reason — *"no committed result JSON"* — is itself wrong; `git ls-files --cached` shows the JSON **is** tracked.) But its downstream substrate is `physicalai-val both rigs, finetune 15 / test 65` — the same family measured 75.9–80.8 % leaked, **never cleared**. |
| **R14** | `go_no_go.PASS = false` | ⚠️ Cosmetic. Top keys are `['meta','experiments','verdicts','go_no_go']`; the quoted booleans live at `verdicts.{rigA_to_rigB,physicalai_to_comma2k19}.PASS = false`. |
| **R15** | Plan ① tactical step | ⚠️ **Under-costed.** `four_families.tactical()`/`.strategic()` (`:206-253`, `:254-312`) consume a **window dict** with `pred_key`/`gt_key` via `_decision_family` (`:166`), **not** trajectories. Where the WM's manoeuvre-derivation rule lives is **UNVERIFIED** — no such function is reachable from any IDM script (`idm3_arms.py` mentions "manoeuvre" once, in prose at `:46`). Also, beyond `DT_S`, a sparse 4-waypoint input leaves `lateral()` only ~3 pair-valid steps for curvature/yaw — far noisier than dense 20-step WM trajectories. |

## C. CONFIRMED EXACT — no change

§2's entire table (12 R² + 4 MAE + all per-domain), n = 4,195/36 ep (pai 1,203/14, cm 2,992/22); every paired ΔMAE and its separated flag; the full ladder (15,875/37,444/74,854/141,628 + all 24 per-domain R²) — and **`seed_mean` genuinely IS mean-of-predictions** (`idm4_steer.py:300` `Pm = np.mean([p["S"] for p in Ps], axis=0)`), so *"ensembled predictions per rung"* is right. Leak check (770 candidates, the 4 named collisions, 77 dupes, residual 0); ckpt md5 `ab8f0e49…` / 34,841,012 B / reload delta 0.0; **all 4 checkpoints tracked** (`git ls-files --cached`, 4/4). 07-22 gate (−2.4654/−0.1094/4.0099; +0.6572/+0.00047/2.3988; PASS false ×2) and the gate text at `IDM_VIDEO_PRETRAIN_DESIGN.md:139`. Audit A1 (2.703221→3.856014 = +42.7 %, speed MAE +43.1 %, long_accel +0.081120→−0.184694 SIGN FLIP, n 3,517 vs 3,521), A2 77.5 %, A3 75.9 %. Anchor settlement in full — including that **ρ 0.598 is correctly the R0 CLEAN20 yaw ρ** (`ANCHOR_SETTLEMENT.md:232`, 0.602→0.598), CLEAN20 = 2,720 win/20 ep, 4.29×/18.4×/9.1 %-carries-61 %, all 18 arms −0.36 to −0.58. A0 rescore (30 clips/4,140 win; speed +0.5545, steer +0.4090, long_accel −0.7375, all CIs); `cm_00045` T=300, n_observable=0, 84 survivors to **15.275** rad/s. All six IDM_V3 geometry contrasts. Derisk (0.6248/0.632, 0.6575/0.7107, 0.862). Parity validation 109/107/71 %, 4 seeds. 07-28 geocalib in full (2,240 win, 14.212, 5.815, −0.19, frac_in_band 1.0, licence {None:27, CC-BY:1}, *"no accuracy claim … none may be quoted"*, labeler on 16,063 parity windows). recon ADE@2s 2.533146. `spot_check_speed.py:6`, `VALIDATION.md:110-111`, HF rev `8efe8c3a9274…` private. **No `four_families` import in any IDM script — two probes, confirmed**; all consumers are WM. `DT_S=0.1` at `four_families.py:52`. 215 tracked `idm` files. scaleup `pod_artifacts/` genuinely empty.

**0.4271 check: the report never quotes v1's 0.4271 nor any 19-episode number → no violation.**

---

---

# ✅ CORRECTED REPORT — IDM PERFORMANCE & VALIDATION AUDIT

`REPO = G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD` · HEAD `7582253d` · 215 tracked files match `idm`.

## 1. ARTIFACT INVENTORY

Code (all MEASURED, source-verified): `stack/scripts/idm_head.py` (347 L — `IDMHead:124`, `build_windows:97`, `Standardizer:166`, `idm_loss:182`, `train_head:238`, `traj_targets_at:74`), `run_idm_proof.py` (394 L), `run_idm_{ft,pipeline_derisk,downstream_ablation,parity_validation}.py` (`PseudoLabeler:40`, `label_clip:54` in derisk), `stack/tests/test_idm_head.py` (75 L). Design + gate: `TanitAD Research Hub/Architecture & Inference/IDM_VIDEO_PRETRAIN_DESIGN.md:139`.

Results chronology as in the original **with these row changes**:
- **07-22** — add: the PASS booleans live under `verdicts.*`, not `go_no_go`.
- **07-24 downstream-ablation** — **5 seeds (comma) + 3 seeds (rig-B)**, not "8 seeds"; fractions 0.917–0.984 correct.
- **07-24 youtube-idm-pilot** — reclass **MEASURED (directional) + ⛔ leak-status UNVERIFIED**; the "≈92 %" is a `NOTE.md` derivation on the *parity* floor, and `fraction_of_ceiling_speed_r2` is `null` in the JSON.
- **07-26 idm-v2** — **4 of 4** pre-registered bars fail vs A0 (speed passes only vs control B0).

All four checkpoints **tracked in git** (verified, 4/4). `idm_head_v3.pt` also on HF `Sayood/tanitad-idm-head-v3`, rev `8efe8c3a9274e4ca193e4c8c06e6c5333bcbb78e`, private.

⚠️ **No IDM head has a `MODEL_REGISTRY.md` row** (two probes). The registry's only IDM content is §8.1 #6 and §10 — and **§8.1 #6:1852 still quotes the withdrawn 77.5 %-leaked "held-out speed R² 0.930"**. Every number in §2 is therefore admissible as **raw eval JSON**, not registry. **Escalation, not a doc note.**

## 2. CURRENT MEASURED PERFORMANCE

**Best head = `idm_head_v4_steer_ens3.pt`** (md5 `ab8f0e49364435fafa927a7986ea948d`, 34,841,012 B) at `…/Benchmarks & Eval/Implementation/incoming/2026-07-27-fleet-sync-idm-steer/idm_head_v4_steer_ens3.pt`. Recipe R0 (k=4/9 frames, d_model 256), repaired labels, 3-seed **mean of predictions**, rung 757 (cm 121 / pai 636) / 141,628 train windows. Estimator `paired_episode_cluster_bootstrap`, unit=episode, B=2000; `ci.py` **md5** `c92618a0…` (⚠️ *file* md5, not a commit; a stale sibling `ef925f06…` exists and the `sys.path` pin is defeated by `idm2_lib.py:19` — held only by import order + runtime assert). n = 36 val ep / 4,195 win (pai 14/1,203; cm 22/2,992). Raw `…/raw/idm5_ensemble.json`. **[table unchanged — verified exact]**

| channel | pooled R² | PhysicalAI | comma2k19 | pooled MAE |
|---|---:|---:|---:|---:|
| speed | +0.8650 | **+0.9312** | +0.7453 | 3.223 m/s |
| yaw_rate | +0.9188 | **+0.9624** | +0.6948 | 0.01697 rad/s |
| steer | +0.7993 | +0.7858 | +0.8071 | 0.00865 |
| long_accel | −0.0591 | −0.0369 | −0.2258 | **NOT SHIPPED** |

Paired ΔMAE vs A0, per corpus (negative = ensemble better) — verified exact: `steer` pai −0.0042 [−0.0066,−0.0013] SEP / cm −0.0030 [−0.0048,−0.0017] SEP · `yaw` pai −0.0135 [−0.0183,−0.0091] SEP / cm −0.0089 [−0.0119,−0.0061] SEP · `long_accel` not separated.
⚠️ **ADDED (R4):** `speed` is not merely "not separated" — it is **nominally worse**: pooled **+0.2287 [−0.377,+0.835]**, comma **+0.4262 [−0.348,+1.218]**, pai −0.2625 [−0.988,+0.337]. The ensemble beats A0 on rotation/steer and **does not beat it on speed**.

Ladder (verified exact; `seed_mean` = mean of per-seed predictions):

| rung | train win | speed pai/cm | yaw pai/cm | steer pai/cm |
|---:|---:|---|---|---|
| 68 | 15,875 | +0.654/+0.778 | +0.855/+0.666 | +0.371/+0.584 |
| 200 | 37,444 | +0.898/+0.694 | +0.958/+0.670 | +0.707/+0.745 |
| 400 | 74,854 | +0.925/+0.728 | +0.957/+0.707 | +0.732/+0.795 |
| **757** | **141,628** | **+0.931/+0.745** | **+0.962/+0.695** | **+0.786/+0.807** |

⇒ PhysicalAI saturating. ⚠️ **CORRECTED (R5):** comma is *not* established as flat — there is **no estimator on any rung delta** in the artifact, and comma **steer rises monotonically +0.745→+0.807** over 200→757. "More data is not the cross-domain lever" is **ESTIMATED**, not measured.

### ⛔ Is there a DECISION-GRADE number for what the IDM is FOR? **No.** (all four points verified)
1. Zero GT and zero accuracy measurement on action-free web video (`YOUTUBE_GEOCALIB_IDM_RESULT.md` §5: *"No accuracy claim is made and none may be quoted"*; n=2,240, mean 14.212, std 5.815, **min −0.19 m/s physically impossible**, 100 % in band; **3 videos**, hfov 53°/58° logged for 2 of them).
2. The pre-registered gate (`:139`) has never passed and never been re-run clean; the only run is 07-22 `PASS=false` on a **77.5 %-leaked** in-domain denominator.
3. The deployed labeler is A0, not the ensemble — ⚠️ but per R4 the ensemble's advantage is **rotation/steer only**.
4. A0 on `cm_[40:70]` (30 clips / 4,140 win, `episode_cluster_bootstrap` B=2000): speed **+0.5545 [0.3154,0.7169]**, yaw **+0.0000 (OFF) / −0.0004 (ON, CI [−2.2876,0.1485])**, steer **+0.4090 [−9.549,0.5941]**, long_accel **−0.7375 [−1.0165,−0.5286]**. Repair does not move R²; `cm_00045` (T=300, 0 observable frames, 84 impossible survivors to 15.275 rad/s) dominates; the fix is **admissibility, not repair**.
   ⚠️ **CORRECTED:** disjointness here is **episode_id-level, not content-level**; and the artifact itself refuses the comparison to 0.7590 (*"a plausibility band, not an equality check"*).
   ⚠️ **CORRECTED (R3):** the admissibility API **does** exist and **is** consumed — `comma2k19.heading_admissible_centers` / `yaw_rate_from_heading` (`:322,339`), used by `…/2026-07-27-anchor-settlement/code/admissibility_consequence_61c.py:146-197` and covered by `stack/tests/test_yaw_admissibility.py`. The remaining gap is that **no IDM scoring path calls it by default**, not that it has no caller.

## 3. VALIDATION: HAVE vs NEED

Unchanged and verified: CAN/GNSS GT on two corpora only (`STEER_RATIO = 15.3` at `comma2k19.py:45`; **26.27 %** of sub-0.5 m/s frames impossible, `:53,124,163`); no CAN validation on any web clip ever. Leak table (A1 80.0 %/40.0 %, A2 77.5 %, A3 75.9 %, A4/A5 same substrates, **A5 carried the GO**) verified exact, incl. 2.703221→3.856014 (+42.7 %) and the `long_accel` sign flip. What survives clean: `physicalai-val-0c5f7dac3b11` content-verified; the v4/v5 leak check (770 candidates, 4 named val collisions, 77 dupes, **residual 0**) is the best in the program. CLEAN20 fragility verified exact (A0 +0.3308→−0.746 [−1.574,−0.177]; R0 +0.6791→**+0.3038 [+0.054,+0.479]**, ρ 0.602→0.598; 9.1 % of windows carry 61 % of SS; all 18 arms −0.36 to −0.58). ⛔ The shipped ensemble's `cm yaw +0.6948` has never been re-scored on CLEAN20.

Scale ambiguity — ⚠️ **REWRITTEN per R2.** Physics stays **PUBLISHED** (Nistér TPAMI 2004; Longuet-Higgins & Prazdny 1980): rotation observable, translation up to a positive scalar. ⛔ **But our measured ordering does NOT independently confirm it.** On matched slices A0's speed R² **beats** its yaw R² everywhere (pooled .8651>.8108; pai .9070>.9035; cm .7590>.3308). Only "long_accel worst" survives. The geometry route is **REFUTED with three controls** (+0.4944 [+0.0001,+1.1021] worse; real-vs-shuffled −0.1479 [−0.4830,+0.1821] **indistinguishable**; physics form +0.6712 [+0.2385,+1.1531] worse; closed-form MAE 2.960→3.236) — unchanged. Oracle per-clip rescale headroom **2.960→1.607, ΔCI [−1.869,−0.881]**, mechanism **UNKNOWN** — unchanged. GeoCalib is partial mitigation; speedometer-OCR **not implemented** (`spot_check_speed.py:6`). Two-source defect confirmed, **citation corrected to `comma2k19.py:553,556,560-561,563` + `idm_head.py:66,68`**.

## 4. FOUR-FAMILY RULE — VIOLATED (upheld in full)

**No IDM script imports `four_families`** — two independent probes, both empty; every consumer is world-model side. LONGITUDINAL **PARTIAL** (distance-keeping absent; `four_families.py:97-128` marks it UNAVAILABLE — ingest does not read `obstacle.offline`). LATERAL **PARTIAL** (no curvature/heading/cross-track bias; exactly what `lateral():131-164` emits). TACTICAL ⛔ **ABSENT** — the IDM emits no manoeuvre class (`idm3_arms.py:46` is prose only); the only manoeuvre HUD comes from flagship-v1's policy brains (`VALIDATION.md:110-111`). STRATEGIC ⛔ **ABSENT**.

**The hazard stands and is now sharper.** ⚠️ Per R2 the honest framing is *not* "strong on speed because rotation is scale-free" — on A0, **speed beats yaw on every slice**, and the cross-domain rotation number on content-clean comma is **+0.3038**. A pseudo-label set whose rotation channel reads +0.30 out-of-corpus while its only YouTube-side check is a speed histogram is exactly "speed right, manoeuvre wrong", undetectable. The 07-24 parity validation already shows yaw at **71 % of ceiling vs 109 % speed**.

## 5. RANKED PLAN — corrected for executability

**① `E-IDM-FF` — four families + manoeuvre confusion on the IDM.**
⚠️ **COST CORRECTED: NOT 0-GPU.** Predictions are **not** banked (R1). Requires re-running `idm_head_v4_steer_ens3.pt` (in-repo ✓) over latents — pod path `/root/idm2/lat`, or a dev-box re-encode whose fidelity the program has already flagged. **Realistic: ~1 GPU-h + ~3 h dev.** Two code items, not one: (a) `four_families` hard-codes `DT_S = 0.1` (`:52`) vs IDM horizons 0.5–2.0 s — pass cadence or resample (flagged at `…/2026-08-02-ctrv-floor/run_four_families_vs_floors.py:9`); (b) `tactical()`/`strategic()` consume a **window dict** with `pred_key`/`gt_key` (`:166,206,254`), not trajectories — **the WM's manoeuvre-derivation rule must be located first; its home is UNVERIFIED.** Note curvature/yaw from 4 sparse waypoints yields ~3 pair-valid steps and will be noisy. STRATEGIC → `UNAVAILABLE` with reason + n. **Falsifier unchanged** (≥0.85 pai AND ≥0.70 cm, no class recall <0.50; below ⇒ YouTube pseudo-labels may train speed + longitudinal trajectory only).

**② `E-IDM-CLEAN20` — re-score the ensemble's comma channels on the 20 content-clean episodes (2,720 win).**
⚠️ **COST CORRECTED: same re-run dependency as ①** — run it in the same pass. **Falsifier unchanged** (<+0.30 ⇒ `yaw_rate` stays caveated).

**③ `E-IDM-REGATE` — re-run the gate with numerator AND denominator content-clean, on the v4 ensemble. ~1 GPU-h.** Unchanged; still the highest-leverage item, since every existing gate number rests on a 75.9–77.5 % leaked denominator.

**④ `E-IDM-SWAP` — re-label the 20 banked YouTube clips with the ensemble. ~2 GPU-h.**
⚠️ **PREMISE CORRECTED (R4):** the ensemble is better on **yaw and steer**, and **nominally worse on speed** (comma Δ +0.4262). Since the pipeline ships `speed`+`long_traj` as *primary* and `yaw_rate` as *caveated*, a swap **trades the primary channel for the caveated one**. Reframe as: does the ensemble's rotation gain justify a possible speed regression? Answer ② first.

**⑤ `E-IDM-OCR-GT` — speedometer-OCR GT on web video. ~1 GPU-day + human.** Unchanged, including both gates: exactly **one** gentle harvest run, and the PI must confirm the licensing posture (`license_distribution = {None: 27, CC-BY: 1}`, `not_cc_kept: 27`).

**⑥ NEW — `E-IDM-REGISTRY` (0 GPU, ~1 h).** Two live defects (R6): the IDM head family has **no registry row**, and **§8.1 #6:1852 still publishes the withdrawn 77.5 %-leaked "held-out speed R² 0.930"**. Under the source-of-truth rule these are blocking, and the audit's own fix-table does not list §8.1 #6 as closed.

**Ranking logic (revised):** ③ and ⑥ move to the top — ⑥ is genuinely 0-GPU and the registry is the only quotable source; ③ is the gate every downstream GO rests on. ① and ② remain the binding four-family work but are **~1 GPU-h, not free**. ④ is demoted pending ②.

**Not on the list, deliberately:** further camera-geometry conditioning (refuted with three controls incl. an indistinguishable-from-shuffled negative). ⚠️ **"More training data" is NO LONGER excluded** — the flatness claim has no estimator and comma steer is still climbing (R5).