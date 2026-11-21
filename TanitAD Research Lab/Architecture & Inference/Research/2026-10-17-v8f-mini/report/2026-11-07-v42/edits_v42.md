=== V42S ===
v42 adds §7.42 to §7.44. T2 here is re-perception on NuRec scenes with the other agents replayed at log time: never a reactive-traffic result and never pooled with T1-kin (paired scene-cluster bootstrap, B = 2,000, seed 0; two training seeds and no replicate arm). §7.42 reads 48 of the 62 scenes (rule R-SCN refused 14 before any planner ran; 93 of 121 starts): the launch crawl persists under a re-rendered camera (CRAWL-PERSISTS, branch a: X1 +0.3388 / +0.4256, separated; the camera-clock contrast X2 and the BLIND contrast are not separated at either seed), and the composite C0C is NOT-DOMINANT over P10 at both seeds (ADE6 +0.7893 / +0.6399 m; M1 crawl share +0.4073 / +0.4041; D1, D2 and D3 all fail), with the failure longitudinal and the lateral rows holding up. §7.43 moves the LoRA run to the PI's GPU pod (RTX 4000 Ada, 20 GB): the device alone misses the token-identity bar by 21 to 23 times, so the base is B′, retrained on the pod's own cache (Amendment 4); no LoRA training result exists. §7.44 gives the theory: the level-contraction form of the crawl is refuted (P1 FAILS), ego dominates vision in the local Jacobian with a scale caveat (P2 HOLDS; the weight-only claims matched 2 of 4), P3 FAILS as written (all 26 dominating pairs have a CEILING dominator), P1′ is NOT EVALUABLE as pre-registered (24 CONFIDENT windows against a floor of 30), and the pre-registered weight edit WSP-GAIN is built and not yet read. No counted DOMINANT exists, there is no scale signal, and nothing in v42 is a driving result.

=== S81 ===
32. **T2 re-perception reads the crawl and the composite on 48 NuRec scenes, the LoRA run moves to the PI's GPU pod, and the theory of the crawl is tested** (§7.42 to §7.44). The T2 reads use a paired scene-cluster bootstrap (B = 2,000, seed 0) on 93 starts; the theory reads are zero-training diagnostics on banked artifacts.
    * **Designed and frozen before any rollout or measurement (MEASURED).** `PREREG_T2_CRAWL` (`{{RES_CRAWLC}}…`, landed `{{LAND_CRAWLC}}`), `PREREG_T2_DOMINANCE` (frozen `38aa9df3…` before any stage-2, 3 or 4 row; with its log `{{RES_DOMC}}…`), the R-SCN amendment (`{{RES_RSCN}}…`), Amendment 4 of the LoRA card (`{{RES_AM4}}…`, landed `{{LAND_AM4}}`), `THEORY_CRAWL.md` (`{{RES_THC}}…`, landed `{{LAND_TH}}`) with its Amendment 1 (`{{RES_THA}}…`), `PREREG_WSP_GAIN.md` (`{{RES_WG}}…`) and the weight-only claims `CLAIMS_P2.md` (`{{RES_CP2}}…`).
    * **Result.** <<V42S>> Its limits are in §7.42.6, §7.43.7 and §7.44.8.

=== R83 ===
**Update (v42).** Three items are written up here, and the adopted base is unchanged (DAG-OACC):
* **T2 re-perception on NuRec scenes (§7.42, T2, 48 scenes, zero training):** the crawl read is CRAWL-PERSISTS (branch a) and the dominance read of record is NOT-DOMINANT at both seeds; the failing terms are D1 (ADE6 +0.7893 / +0.6399 m), D2 (separated-worse on ADE6, the M1 crawl share, TAC-LON and along-track MAE) and D3 (no separated-better TACTICAL row; STRATEGIC is PAUSED by ITEM 25). The agents replay, so this is not a reactive-traffic result.
* **The LoRA run on the PI's GPU pod (§7.43):** the venue is measured, the device alone misses the token-identity bar (2.11 to 2.25e-2 against 1e-3), and Amendment 4 commits the base B′ retrained on the pod's cache. No LoRA training step had run at the build of this report.
* **The theory of the crawl and the weight-space reading (§7.44):** P1 FAILS, P2 HOLDS with a scale caveat, P3 FAILS as written, P1′ is not evaluable as pre-registered, P4 is not evaluable, and the weight edit WSP-GAIN is pre-registered, built and not yet read.
* **What is blocked:** no counted DOMINANT exists, so no evidence video and no scale signal follow. The committed next read is the LoRA arm's dominance read at T2 on box 2 (`DRAFT_PREREG_T2_DOM_LORA.md`, a draft, not frozen), and it waits on the pod's gates (the B′ base, the batch-64 timing, the smoke and the token verification) and on the PI's venue choice. The remaining downstream levers are points on one frontier (§7.44.3), so the evidence points to a change of the features, not of the head.

=== ROW ===
| 1bd | **T2 crawl read and T2 dominance read** (`H-T2-CRAWL-1`, `H-T2-DOM-1`; zero training; T2 re-perception on 48 NuRec scenes, agents replayed) | cards `PREREG_T2_CRAWL.md` `{{RES_CRAWLC}}…` (frozen `2c4c6557…`) and `PREREG_T2_DOMINANCE` (frozen `38aa9df3…`), both before the rows they decide: CRAWL-PERSISTS iff X1's lower bound is above 0, X1 is at least h and X1 exceeds the seed gap at both seeds (branch a when BLIND − C0 is not separated); DOMINANT iff D1 (ADE6 separated-better, point at most −0.2415 m), D2 (no separated-worse row of 19) and D3 (a separated-better row in every family) hold at both seeds; D2 failing on M1 and the longitudinal rows ⇒ the perception lever at T2 | **CRAWL-PERSISTS (a)** (§7.42.4, T2): X1 +0.3388 [+0.2555, +0.4291] / +0.4256 [+0.3231, +0.5224]; X2 and BL not separated. **NOT-DOMINANT at both seeds** (§7.42.5, T2): ADE6 +0.7893 / +0.6399 m, M1 crawl share +0.4073 / +0.4041, D1, D2 and D3 false; 48 of 62 scenes after rule R-SCN; the committed next is the LoRA perception lever at T2 (`DRAFT_PREREG_T2_DOM_LORA.md`, a draft) |
| 1be | **LoRA on DINOv3 on the PI's GPU pod: Amendment 4** (`D-V8F-LORA-POD-1`; the venue, the frame route and the base B′) | `AMENDMENT_V8F_LORA_4.md` `{{RES_AM4}}…` (frozen `626aa1e5…` before any B′ or LoRA training step): B is kept iff the pooled-fp8-token relative L2 against box 2 is at most τ_device = 1e-3 on all three clips, else B′ (M2b retrained on the pod's cache at seeds 0 and 1) and F′ from B′; the PRIMARY bar becomes \|δ_s\| > 2F′ | **B′ route committed** (§7.43.4, [M]): the device alone reads 2.11 / 2.25 / 2.14e-2, the pod frames 2.42 / 2.63 / 2.25e-2, against 1e-3; the frame route differs by ±1 level on 1.377 % of values; the batch-64 step time is not measured; no B′ or LoRA step had run; **no LoRA result exists** (§7.43.6) |
| 1bf | **WSP-GAIN** (`D-V8F-THEORY-CRAWL-1`; zero training; a rank-one edit of the last linear layer of `head_o` that restores the base's gains) | `PREREG_WSP_GAIN.md` `{{RES_WG}}…` (frozen `92d61dcb…` before any edit was built or scored): MECH-FAIL unless the edited gains are within 10 % of the target in all four (seed, α) cells; then HARMS, MOVES-M1 with guards holding (the T2 stage on box 2) or NO-EFFECT; the prior for DOMINANT-T2 is LOW [E] | **pre-registered, built and unit-tested, not yet read** (§7.44.6): no edited checkpoint was scored and no number is quoted |

=== I20 ===
20. **v42 reads and their limits.**
    * **What the interval answers.** The T2 reads use the paired scene-cluster bootstrap (B = 2,000, seed 0), which answers another draw of SCENES only. The training-seed question is answered by requiring both seeds; there is no replicate arm on this rig (`H-ESTIM-SEED-1`), and no inference seed is varied (the planner is deterministic).
    * **Population.** Rule R-SCN refused 14 of the 62 scenes before any planner ran, so the reads cover 48 scenes and 93 of 121 starts; the 62 were selected on the human's logs only, launch scenes are over-represented (42 of 62), and the rollouts exclude v8F's training clips. A T2 result is on these scenes, never on PhysicalAI-AV at large.
    * **The tier.** The other agents replay at log time and do not react to the ego, so no T2 number says the car is safe or unsafe. T2 is never pooled with T1-kin, and the T1-kin and T2 levels of one contrast (+0.0583 / +0.0586 against +0.3388 / +0.4256) are on different populations. Each start carries 3 ticks, so any iterated quantity is read at depth at most 2.
    * **The render gap.** The rendered frame sits about 20 times above the encoder's own floor in relative L2 (stage 1, 2 scenes); the attribution contrast X2 is therefore measured inside one renderer and the gap is beside, never deciding.
    * **The LoRA pod.** The throughput projection (about 97 h of encoder time per seed, 100 to 120 h with the heads) is ESTIMATED from a proxy, and the batch-64 step time is not measured at the build; the gate route follows `E7_DECISION.md`, not `launch_gate.py`. The pod handoffs and the Amendment 4 log are E: files that have not landed. No LoRA result exists.
    * **The theory reads.** The weight-only claims matched 2 of 4: a weight claim is made at an assumed operating point, and the two that failed mispredicted the GELU slopes and where attention sits. The Jacobian is local (586 windows, tick 0, 3 to 8 m/s, C0 only); P2 is scale-sensitive (3.29 and 2.53 within the band). P3 failed as written, and the single chain of five admissible levers is post hoc (n = 5). The information bound is a theorem; its tightness is a hypothesis [H], because the probe AUCs bound the Bayes AUC from below. P1′ was not evaluable (24 CONFIDENT windows against a floor of 30; its read is an E: file not yet landed) and WSP-GAIN was not read.
    * No reactive-traffic number exists.

=== ABS1 ===
* **T2 re-perception on NuRec scenes: the crawl persists under a re-rendered camera and the composite is NOT-DOMINANT at both seeds (§7.42; T2 = the camera rendered at the executed pose, the other agents replayed at log time; paired scene-cluster bootstrap, B = 2,000, seed 0; 48 of 62 scenes after rule R-SCN, 93 starts; **[M]**).** The crawl read (`PREREG_T2_CRAWL`) is CRAWL-PERSISTS, branch (a): the M1 crawl-share contrast C0 − P10 at T2 is +0.3388 [+0.2555, +0.4291] and +0.4256 [+0.3231, +0.5224] (separated at both seeds), the camera-clock contrast X2 is −0.0021 [−0.0400, +0.0375] and +0.0208 [−0.0067, +0.0525], and replacing the scene tokens by their training mean (BLIND) does not change it. The dominance read of record (`PREREG_T2_DOMINANCE`, frozen before any row) is NOT-DOMINANT at both seeds: ADE6 C0C − P10 +0.7893 [+0.3313, +1.2386] and +0.6399 [+0.1070, +1.1504] m against a bar of −0.2415 m, the M1 crawl share +0.4073 and +0.4041, and TAC-LON and along-track MAE separated-worse, while yaw-rate, curvature, cross-track and lateral-acceleration MAE are separated-better. The failure is longitudinal. The agents replay, so this is not a reactive-traffic result; the 62 scenes were selected on the human's logs and a result on them is never a result on PhysicalAI-AV at large.
* **The LoRA run moves to the PI's GPU pod, and a theory of the crawl is tested with a weight-space reading (§7.43, §7.44).** On the pod (RTX 4000 Ada, 20 GB) the device alone puts the pooled fp8 tokens at a relative L2 of 2.11 to 2.25e-2 from box 2's cache, against a bar of 1e-3, so Amendment 4 commits the base B′ retrained on the pod's own frames; no LoRA training result exists. The theory predicted a contraction of the speed level; **P1 FAILS** (C0C's planned speed lies above its executed speed in every occupied 3 to 8 m/s bin), the test of the acceleration-shrinkage form, P1′, is not evaluable as pre-registered (24 CONFIDENT windows against a floor of 30), **P2 HOLDS with a scale caveat** (the ego-to-vision gain ratio is 14.54 [12.63, 16.74] and 11.22 [10.04, 12.55] in the local Jacobian and 3.29 and 2.53 under a within-band ablation; the weight-only claims matched 2 of 4), and **P3 FAILS as written** (all 26 dominating pairs have a CEILING dominator). The data-processing inequality bounds every lever downstream of the features, so only a change of the features can pass the frontier; the pre-registered weight edit WSP-GAIN is built and not yet read. **[H]** for the interpretation, **[P]** for the method (Takehi et al., a blog post, not peer-reviewed).

=== NXT ===
v42 reads T2 re-perception on 48 NuRec scenes (§7.42): the crawl persists (CRAWL-PERSISTS) and the composite is NOT-DOMINANT over P10 at both seeds, with the failure longitudinal, so the committed next is the perception lever evaluated at T2. The LoRA run is on the PI's GPU pod (§7.43); the device alone misses the token-identity bar, the base is B′, and no LoRA step had run at the build. The theory (§7.44) refutes the level-contraction form, holds the information bound as a theorem with its tightness a hypothesis, and could not evaluate P1′ (a group floor) and leaves the weight edit WSP-GAIN unread; the lever ranking is perception first, then a zero-training frontier point.

=== REL ===
**Weight-space learning and reading (v42).** Takehi, Teoh, Zhang and Khattab (2026-11-02, *Sparks of Weight-Space Programming*, a blog post, not peer-reviewed; Library key `takehi2026weightspace`, banked page sha256 `a326bc5b…`; the Library index entry is not on the tip) **[P]** report that an LLM read the circuits of a 7.16M-parameter network from its weights with no forward pass, that its claims matched ablation ground truth, and that an edit of one direction removed an obstacle, not that it added an ability. We applied the protocol to v8F's C0 in §7.44.5: claims from the tensors alone, hashed before any forward pass, then checked by Jacobian and ablation on held-out windows. Two of four weight-only claims matched. Descriptions here follow the banked page; none of it is our result, and we do not claim the method generalises.

=== APP ===
**Reads added in v42 (tip `{{TP}}`).** The raw folders under `R/raw/` that §7.42 to §7.44 read (sha256 prefix of the key file; landing commit); the claim rows are in `Project Steering/GOALS_AND_CLAIMS.md`.

| raw folder | key file (sha256) | landed | claim row |
|---|---|---|---|
| `2026-11-06-t2-harness-v8f/t2_prereg/` | `PREREG_T2_CRAWL.md` `{{RES_CRAWLC}}…` | `{{LAND_CRAWLC}}` | `H-T2-CRAWL-1` |
| `2026-11-06-t2-stage1-box2/` | `STAGE1_READ.md` `{{RES_S1R}}…` | `{{LAND_S1R}}` | `H-T2-CRAWL-1` |
| `2026-11-06-t2-crawl-read/` | `stage2/RESULT_T2_STAGE2.md` `{{RES_R2}}…`, `AMENDMENT_T2_SCENE_REFUSAL.md` `{{RES_RSCN}}…` | `{{LAND_R2}}` | `H-T2-CRAWL-1` |
| `2026-11-06-t2-stage4-c0c/` | `MANIFEST.txt` `{{RES_S4}}…` | `{{LAND_S4}}` | `H-T2-DOM-1` |
| `2026-11-06-t2-dom-repin/` | `MANIFEST.txt` `{{RES_REPIN}}…` | `{{LAND_REPIN}}` | `H-T2-DOM-1` |
| `2026-11-06-t2-dominance-read/` | `stage4/score/DOMINANCE_T2.md` `{{RES_DOM}}…`, `lora_pod/PLAN_V8F_LORA_POD.md` `{{RES_PLAN}}…` | `{{LAND_DOM}}` | `H-T2-DOM-1`, `D-V8F-LORA-POD-1` |
| `2026-11-06-lora-pod-amend4/` | `AMENDMENT_V8F_LORA_4.md` `{{RES_AM4}}…`, `identity_pod/IDENTITY_POD.md` `{{RES_IDP}}…` | `{{LAND_AM4}}` | `D-V8F-LORA-POD-1` |
| `2026-11-06-theory-crawl/` | `THEORY_CRAWL.md` `{{RES_THC}}…`, `PREREG_WSP_GAIN.md` `{{RES_WG}}…`, `wsp_read/WSP_READ.md` `{{RES_WR}}…` | `{{LAND_TH}}` | `D-V8F-THEORY-CRAWL-1` |

E: only, not yet landed (dev box, `E:/Projects/TanitAD-artifacts/dagland/motion_c/`): `lora_pod/HANDOFF_LORAPOD2.md`, `lora_pod/HANDOFF_LORAPOD3.md`, `lora_pod/AMENDMENT_V8F_LORA_4_LOG.md` and `theory/wsp_edit/`. The v42 build is `patch_v42.py` (anchored edits, every table copied line for line from its source and asserted to be a source line), checked by `checks_v42.py` and re-measured by `audit_v42.py` (`v42_NUMBERS_AUDIT.md`).

=== C3A ===
research report, v41 of 2026-11-06, results through repository tip `bd98063d` and, within it,
-->
research report, v42 of 2026-11-07, results through repository tip `{{TP}}` and, within it,

=== C3B ===
(§7.40), and the progress-floor addenda A4 to A6 (§7.41) (v40: `ffa2637a`;
-->
(§7.40), the progress-floor addenda A4 to A6 (§7.41), T2 re-perception on NuRec scenes (§7.42), the LoRA run on the PI's GPU pod (§7.43), and the theory of the crawl with the weight-space reading (§7.44) (v41: `bd98063d`; v40: `ffa2637a`;

=== C3C ===
Open-loop results only (T0, T1 and the self-action T1-kin); **no <<CLH>> (T2) number exists** and no published baseline exists on this split.
-->
Open-loop results (T0, T1 and the self-action T1-kin) and, from v42, T2 re-perception results on 48 NuRec scenes with the other agents replayed (§7.42, NOT-DOMINANT); **no reactive-traffic number exists** and no published baseline exists on this split.

=== C5 ===
T2 <<CLS>>, unprovisioned)
-->
T2 re-perception on NuRec scenes, agents replayed at log time)

=== C17 ===
We claim neither state of the art nor <<CLH>> driving. The dominance above is over the programme's own kinematic floor, in open loop. In a self-action open loop it becomes a margin, not dominance (next paragraph); T2, the <<CLH>> tier, is not provisioned.
-->
We claim neither state of the art nor reactive-traffic driving. The dominance above is over the programme's own kinematic floor, in open loop. In a self-action open loop it becomes a margin, not dominance (next paragraph); at T2 (re-perception on 48 NuRec scenes, agents replayed, §7.42) the composite is NOT-DOMINANT over that floor at both seeds, and no reactive-traffic tier exists.

=== C335 ===
**T2, the only <<CLS>>, is not provisioned, so no <<CLH>> number exists**
-->
**T2 (re-perception: the camera is rendered at the executed pose and the other agents replay their logs) was provisioned on 2026-11-06 and is read in §7.42; no reactive-traffic tier exists, so no number of this report is a reactive-traffic result**

=== C602 ===
(T2, re-perception, is not provisioned), so ⛔ this is an open loop, not <<CLH>> driving.
-->
(T2, re-perception, is read separately in §7.42), so ⛔ this is an open loop, not reactive-traffic driving.

=== C5719 ===
Until that read exists, `reindexed` stays a longitudinal-only approximation and no T2 number exists.
-->
Until that read exists, `reindexed` stays a longitudinal-only approximation and no T2 number exists. That read exists as of v42 (§7.42: T2 re-perception on 48 NuRec scenes, CRAWL-PERSISTS and NOT-DOMINANT); `reindexed` remains a longitudinal-only approximation inside T1-kin.

=== C5792 ===
all numbers are open loop (T0/T1), and no T2 <<CLH>> run exists.
-->
all numbers are open loop (T0/T1) or T2 re-perception with replayed agents (§7.42), and no reactive-traffic run exists.

=== C5861 ===
(iv) there is no <<CLH>> evidence: in the self-action open loop (T1-kin),
-->
(iv) there is no reactive-traffic evidence, and the T2 re-perception read finds the composite NOT-DOMINANT over the kinematic prior at both seeds (§7.42.5: ADE6 +0.7893 / +0.6399 m on 48 scenes, agents replayed); in the self-action open loop (T1-kin),

=== C5926A ===
read of HACC; T2 (<<CLS>>) |
-->
read of HACC; T2 (re-perception) |

=== C5926B ===
−0.728 / −0.641 m. T2 is not provisioned |
-->
−0.728 / −0.641 m. T2 re-perception of the composite C0C (§7.42.5, v42) is NOT-DOMINANT at both seeds |

=== C18 ===
    * No T2 number exists.
19. **v41 reads and their limits.**
-->
    * No T2 number exists for the reads of this item (T2 re-perception is §7.42).
19. **v41 reads and their limits.**

=== C19 ===
    * No T2 number exists.

---

## 10. Reproducibility appendix
-->
    * No T2 number exists for the reads of this item (T2 re-perception is §7.42).

---

## 10. Reproducibility appendix

=== C5861B ===
Closing the gap needs, at minimum, a <<CLH>> (T2) or published-protocol (NAVSIM-style) evaluation, and gains on all four families at both horizons.
-->
Closing the gap needs, at minimum, a reactive-traffic evaluation (T2 with agents that respond to the ego; the T2 of §7.42 replays them) or a published-protocol (NAVSIM-style) evaluation, and gains on all four families at both horizons.

=== NOTES ===
Changes made to the final section files (the only ones): sec_744_theory.md, the bullet on the fit in 7.44.6: `$\Delta g = $ base gain` became `$\Delta g$ = base gain`. The page template's inline-math pattern needs a non-space character before the closing dollar sign, so the first form would have rendered as literal text and the math guard stopped the build on it. sec_742_t2.md and sec_743_lorapod.md are inserted as written. Directive lines are replaced by their tables, and the fill-in placeholders (the tip, the landing commits and the sha256 prefixes of the sources) by their values.
Carried v41 lines edited in place (blocks C3A to C3C, C5, C17, C335, C602, C5719, C5792, C5861, C5861B, C5926A, C5926B, C18, C19; each is an old text, a line with three characters --> and the new text; the old wording of the T2 tier is written there through placeholders so that this file never holds it): the status line (version, tip, version list, the T2 sentence), the evidence-class line, the abstract contribution paragraph, the section 6 tier paragraph, the T1-kin paragraph of 7.15, the 7.41.6 (b) paragraph (the 7.42 pointer is appended, nothing is removed), the 8.2 driving-claim bullet, the 8.3 (iv) clause and the closing sentence of that paragraph, the 8.4 row 6, and the two section-9 limit bullets that read "No T2 number exists" under items 18 and 19. The abstract next bullet is the twelfth changed line (the sentence NXT is appended). The handoff listed lines 3, 5, 17, 335, 602 and 5861; lines 5792, 5926 and the two limit bullets said the same thing and are edited with them.
Inserted blocks: V42S (shared sentence: status line end and 8.1 item 32), ABS1 (two abstract bullets before the next bullet), NXT, REL (section 3), S81 (8.1 item 32), R83 (8.3 Update v42), ROW (8.4 rows 1bd to 1bf), I20 (section 9 item 20), APP (section 10 block, a table with one row per new raw folder).
State of the late results at the build (checked on the dev box, 2026-11-07 00:20): the pod's batch-64 timing is in PRELIM_SUMMARY.json on the pod only and HANDOFF_LORAPOD3.md holds no number, so 7.43.2 keeps "not yet measured"; theory/wsp_edit/raw holds no READ1.json and no edited checkpoint, so 7.44.6 and the WSP-GAIN row keep "pre-registered, not yet read" (the fit file and the edit parameters exist and are named as such); wsp_read holds P1P_RESULT.md and P1P_RESULT.json (written 00:35), which read P1′ NOT EVALUABLE (CONFIDENT 24 windows against a floor of 30), and 7.44.2, the 7.44.7 row, the intro line and 7.44.8 carry that read as written; motion_c/t2v8f holds HANDOFF_Q.md (written 00:41), the read re-pin with `--arm`: 7.42.6 item 3 and 7.42.7 state it as written (fourth pin 7fd65ffd, identical to the read of record on the real pulls, 80 of 80 mutations RED, suite in flight; E: only).
Tip: the build tip is 1887e9b8 (the tip moved from b17b5492 during the build; the new commit adds a yaw-rate fine-tune pre-registration under 2026-09-20-refe-plan and one register row; nothing under the v8F folder changed). Every landing commit named in the text is an ancestor of it.
Audit: one token (2.4e-6, the maximum total-speed difference of the edit code) is found only in an E: file (HANDOFF_WSPEDIT.md, cited in 7.44.6 as not yet landed) and the audit reports it as such; the derived numbers of 7.43 and 7.44 are in an explicit allow-list and recomputed in python. "21 to 23 times" is 2.11e-2 and 2.25e-2 over 1e-3 rounded half up (22.5 reads 23).
