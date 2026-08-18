# Evidence dossier B — Alpamayo action-space evidence & REF-C tactical-head history

**Provenance:** compiled 2026-08-18 by a read-only research subagent for the REF-C improvement
review (`REFC_IMPROVEMENT_REVIEW.md`, same directory). Worktree `/c/Users/Admin/wt-tanitad-local`,
branch `agent/arch-inf-20260803`. Evidence classes: **MEASURED** (our probe), **PUBLISHED** (NVIDIA
card/config quoted in our banked notes), **INHERITED** (quoted from an owning package, not
re-derived here). This file is the verbatim agent report, banked for provenance.

⚠️ **Line-number drift note:** several cited docs quote `stack/tanitad/models/v6.py:136-140` /
`:161` / `:217-223`. At worktree HEAD the same constants are at **`v6.py:147` (`PLAN_STEPS = 60`),
`:149` (`HORIZON_S`), `:150` (`OP_BAND_S`), `:151` (`TAC_BAND_S`), `:183` (`TACTICAL_LAT_ACTIONS`),
`:187` (`TACTICAL_LON_ACTIONS`), `:228` (`TACTICAL_GOAL_TOKENS_LAT`), `:231`
(`TACTICAL_GOAL_TOKENS_LON`)** — MEASURED. Values unchanged; only offsets moved.

---

## 1. Alpamayo tactical validation package (2026-08-16-tactical-labels)

Package: `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-16-tactical-labels/`
— `TACTICAL_LABEL_VALIDATION.md` (665 lines) + `code/` (5 scripts) + `raw/` (7 artifacts:
`a1_alpamayo_taxonomy.json` + per-clip jsonl, `a2_reasoning_legs.json`,
`a3_three_leg_agreement.json` + per-clip jsonl, `a4_horizon_sweep.json`,
`a5_gtac_reason_coverage.json`).

### 1.1 The 40.62 % dual-axis figure

**MEASURED, n = 4,729 clips, 0 unparsable rows** (`TACTICAL_LABEL_VALIDATION.md:86-89`;
`raw/a1_alpamayo_taxonomy.json:719-726`):
> `simultaneous_lon_and_lat`: **n = 1,921 / 4,729 = 40.62 %**, **98 distinct joint cells**.
> Definition (verbatim, `a1_alpamayo_taxonomy.json:721`): *longitudinal ∉ {null, 'Constant Speed'}
> AND lateral ∉ {null, 'Go Straight'}* — non-trivial on both axes at once, **which a single 5-way
> softmax over `[lane_keep, turn_left, turn_right, accelerate, brake_stop]` cannot represent**.

Registry restatement: `Project Steering/MODEL_REGISTRY.md:3467` (§11.1a) — "direct label-side
support for v6's LAT×LON factoring".

### 1.2 The meta_action taxonomy — THREE axes × 7 values, per clip

**MEASURED** from `records.parquet` (sha256 `ecae276d…`), 4,729 clips
(`TACTICAL_LABEL_VALIDATION.md:59-84`):

| Longitudinal (7) | n | % | Lateral (7) | n | % | Lane (7) | n | % |
|---|---|---|---|---|---|---|---|---|
| Gentle Deceleration | 1594 | 33.71 | Go Straight | 2504 | 52.95 | Lane Keep | 4035 | 85.33 |
| Maintain Speed | 1225 | 25.90 | Steer Right | 1020 | 21.57 | Turn Right | 101 | 2.14 |
| Gentle Acceleration | 1151 | 24.34 | Steer Left | 649 | 13.72 | Turn Left | 85 | 1.80 |
| Stop | 304 | 6.43 | Sharp Steer Right | 135 | 2.86 | Right Lane Change | 82 | 1.73 |
| Strong Deceleration | 267 | 5.65 | Sharp Steer Left | 115 | 2.43 | Slightly Shift Left | 69 | 1.46 |
| Strong Acceleration | 182 | 3.85 | Reverse Right | 1 | 0.02 | Slightly Shift Right | 31 | 0.66 |
| Reverse | 6 | 0.13 | Reverse Left | 1 | 0.02 | Left Lane Change | 22 | 0.47 |

- **Kind:** structured-enum-in-text — closed vocabulary on three labelled lines
  (`Longitudinal: … / Lateral: … / Lane: …`), parsed by regex at
  `code/tac_a1_alpamayo_taxonomy.py:44-49`.
- **The 304 lateral/lane nulls are NOT parse failures** — they are exactly the 304 `Stop` rows;
  a stopped vehicle emits ONE axis (`:81-84`). Treat as *not applicable*, never impute (C77 family).
- **Granularity: ONE declaration per clip, at t₀** — 4,729 meta_action rows / 4,729 unique clips;
  "Alpamayo's meta-action is *the decision at t0*" (`:366`). **Not per-timestep, no horizon
  attached.**
- ⚠️ `Stop` short-circuits the axes (all 5 of the n=39 pilot `Stop` rows ended generation before
  Lateral/Lane; `ALPAMAYO2_SUPER_ANALYSIS.md:741-745`) ⇒ the axes are not fully independent in
  Alpamayo's own scheme.
- ⚠️ **`Stop` is a STATE, not an action** — MEASURED on `Stop` rows: ego v(t₀) = **0.51 m/s**,
  rising to **2.95 m/s** by 2 s (Δv **+2.44**); `cot` says *"Resume speed from stop"*
  (`MODEL_REGISTRY.md:3471-3475`).

### 1.3 "VLM tactical leg information-free on the longitudinal axis"

**MEASURED**, 201-clip aug120 cohort, 270 action emissions, through the shipped `ph1_fuse.py`
`LAT_RULES`/`LON_RULES` — `TACTICAL_LABEL_VALIDATION.md:125-164`:
- Vocabulary (`stack/scripts/ph0_v2.py:39-43`): 11 `GOAL_KINDS`, **6 `ACTION_VERBS`**.
- **Defect 1:** `reduce_to` — the VLM's ONLY deceleration verb — **maps to NOTHING on either
  axis**; **49/270 = 18.1 %** of emissions silently dropped.
- **Defect 2:** `hold_corridor` (a LATERAL verb) matches `LON_RULES` substring `"hold"` → `HOLD`;
  **159/270 = 58.9 %**.
- Result: `vlm_lon3 = {decelerate: 162, None: 39}` — **exactly ONE value wherever it speaks**.
  Cohen's κ vs both other legs = **exactly 0.0000** — *a constant*, not "no skill".
- **104/270 = 38.5 %** produce no LON token; **57/270 = 21.1 %** no LAT token.
- ⇒ "The VLM's tactical longitudinal contribution to every fused record in the corpus is a
  **mapping artifact**" — a code fix, not a re-labelling job.

**And the lateral leg is an ego echo, not vision** (`:165-196`): `_ego_prompt_mode == 'past'` on
**201/201**; the ph1 ego voter reads exactly `turning`+`motion` (`ph1_fuse.py:318-325`), **both in
the VLM's prompt**. κ VLM↔ego-geom (LAT) **0.7608** vs Alpamayo↔VLM **0.1717** and Alpamayo↔ego
**0.2089** ⇒ agrees ≈4.4× better (in κ) with the leg printed in its prompt ⇒ a 2-of-3 majority
satisfied by {ego, vlm} is **ONE source counted twice**. **ESCALATION:** every fused aug120 record
stamps `_provenance.vlm = "vision"` + `semantics` in `inference_admissible` — **MEASURED 201/201
contradicted**; `ph1_fuse.py:556-561` is correct at HEAD, the **banked corpus predates the fix**.

### 1.4 Other measured numbers (selection)

**Reasoning legs (n=4,729):** `meta_action.cot` coverage 100.00 %, 1,103 distinct (23.32 %);
`auto_labeling.chain_of_causation` 100.00 %, 1,685 distinct; `answer` byte-identical to `cot` on
4,729/4,729 ⇒ duplicate field, never corroboration; `trajectory_analysis` non-null 133 (2.81 %) —
effectively absent. Two legs semi-independent: exact match 22.18 %, token-Jaccard median 0.5714.

**Pairwise agreement (n as marked):** LAT Alpamayo↔ego-geom κ 0.2089 (n=193); LAT Alpamayo↔VLM
0.1717 (185); LAT VLM↔ego-geom **0.7608** (not independent); LON Alpamayo↔ego-geom 0.1871 (201);
LON Alpamayo↔VLM **0.0000** (162); `g_tac_lat` Alpamayo lane↔ego lane-events **−0.0240** (153).
Three-way unanimity: LAT 56.22 %, LON **19.14 %**.

**Lane-change corroboration:** of 81 clips where engine-A lane-change events fire and Alpamayo
speaks, Alpamayo corroborates **2 = 2.47 %**; PI's by-eye adjudication of 19 clips gave ≈2 % — two
independent instruments, 2 % vs 2.5 %.

**Horizon sweep (201 clips, 10 Hz, t₀ = index 80; best threshold per cell):**

| horizon | 0.5 s | 1.0 s | 2.0 s | 3.0 s | 5.0 s | 8.0 s | 11.8 s |
|---|---|---|---|---|---|---|---|
| **LON κ** | 0.331 | 0.355 | **0.365** | 0.344 | 0.253 | 0.187 | 0.188 |
| **LAT κ** | 0.337 | 0.460 | **0.469** | 0.462 | 0.444 | 0.396 | 0.290 |

H-HORIZON SUPPORTED as a **ridge at 0.5–3.0 s** (16 of 72 cells within 0.05 κ of peak, all in that
range). Abstaining on the two `Gentle` classes lifts LON κ 0.3655 → 0.4311 (n 201 → 85).
⛔ **§3.1 correction (C89): every §3 κ is a SEAM value** — `tac_a4_horizon_sweep.py:140-148`
anchors every horizon at t₀; no row of the banked sweep is the tactical band. Restated at
PRODUCTION thresholds, episode-cluster bootstrap: **TAC band (2, 6] LON κ 0.1428 [0.0540, 0.2250] /
LAT 0.1777 [0.0658, 0.2953]** (n = 201/193); seam (0, 2] 0.3270/0.3132; full (0, 6] 0.2210/0.3806;
paired band − seam separated on both axes. Turn-direction asymmetry: `Steer Left` mean Δyaw grows
**6.72×** from seam into band; ordinary `Steer *` reads `straight` by `mean_band` on 77 % of clips.

**v6 vocabulary mapping:** LANE axis → `TACTICAL_LAT_ACTIONS` maps cleanly **except `Turn Left`/
`Turn Right` — NO TOKEN EXISTS** (1.80 %/2.14 %, n = 186 clips); `ABORT_LC` would receive 0 labels.
LON axis **DOES NOT MAP** — v6's LON vocabulary is **REASON-typed** (`FOLLOW, YIELD_MERGE, CREEP,
HOLD, BRAKE_TO, CRUISE`), Alpamayo's is **MAGNITUDE-typed**. Reason-reachability (A5, n=4,729):
≥1 `g_tac` token 93.80 % single leg / 97.02 % union; LON token 77.31/82.77 %; LAT token
20.00/27.13 %. Cross-leg LON-token consistency **2,473/3,168 = 78.06 %** ⇒ **≈22 % expected label
error**; top confusions `STOP_POINT`↔`YIELD_AT` 211, `GAP_TARGET`↔`TRAFFIC_LIGHT_REACT` 108.

**Third emission defect:** `ph1_fuse.py:331-347` emits `g_tac_lat`/`g_tac_lon` on every fused
record — but fills the **GOAL**-named field from the **ACTION** vocabulary, and
`stack/tests/test_ph1_fuse.py:70-71` pins the mismatch. **No tactical loss term exists**:
`V6LossWeights` (`stack/scripts/train_v6_staged.py:151-217`) ends at `w_s2_goal: float = 0.0`;
there is no `w_tac_*`.

**Scaling:** 257 clips have w120 video (201 train + 56 val); **4,472 have none (94.6 %)**. Build
MEASURED 19.4 s/clip single-shard, 660 clips/h at 8 shards, ~40 MB/clip; remaining 4,472 ESTIMATED
≈6.8 h at 8 shards / ≈179 GB. Registry adds: 1,418 of 3,146 camera chunks ≈ 1.84 TB (count
MEASURED, mean chunk size ESTIMATED), deferred by PI 2026-08-17; densest-first top-50 chunks ⇒
1,317 clips / 65 GB (`MODEL_REGISTRY.md:3504-3517`).

---

## 2. Alpamayo-2's OWN action-space design

### 2.1 PUBLISHED (NVIDIA card / `config.json` / GitHub README, quoted in our banked notes)

Source: `TanitAD Research Hub/Benchmarks & Eval/Research/2026-08-05-alpamayo2-super/ALPAMAYO2_SUPER_ANALYSIS.md`
(+ `raw/hf_card.md`, `raw/gh_readme.md`), retrieved 2026-08-05; re-quoted
`Project Steering/Reports/2026-08-15-2200-campaign-science-addendum.md:985-1010`.

- **34.3 B = 32 B Qwen3-VL backbone ("Cosmos 3 Super Reasoner") + 2.3 B action expert** (`:18-20`).
  GR00T is NOT linked to Alpamayo anywhere in the banked corpus;
  `Project Steering/REFERENCE_ARCHITECTURES.md` and `REFERENCE_SYSTEMS_RANKED.md` contain **zero**
  occurrences of "Alpamayo" (MEASURED by grep).
- **Action space (`:31-55`):** `action_space_cfg: UnicycleAccelCurvatureActionSpace`,
  **n_waypoints 64 · dt 0.1**; `accel_bounds [−9.8, +9.8]` (mean 0.0290, std 0.6810),
  `curvature_bounds [−0.33, +0.33]` (mean 0.000269, std 0.02615, ≈3 m min radius); ridge/λ on
  a, κ, θ, v (least-squares action fit). **Not free XY — every output dynamically feasible by
  construction.**
- **Decoder:** `diffusion_cfg: FlowMatching`, Euler, **10 inference steps**,
  `inference_guidance_weight 3.0`, `use_classifier_free_guidance false`; `action_in_proj:
  PerWaypointActionInProjV2` (Fourier 20, 2 enc layers, h=512); expert 64 layers, hidden 1536,
  16 heads / 8 KV (GQA); **`expert_non_causal_attention: true`** (denoises the whole 64-step plan
  jointly); **`cotrain_expert_vlm: false`** (frozen backbone).
- **Second, DISCRETE trajectory path — action tokens (`:65-70`):** `future_vocab_size 3000`,
  `history_vocab_size 1000`, `traj_vocab_size 4000`, `tokens_per_future_traj 128`,
  `tokens_per_history_traj 45` ⇒ the VLM reads/writes trajectories as text tokens while the
  diffusion expert emits the continuous plan.
- **Decoding horizon/rate:** **64 waypoints @ 0.1 s = 6.4 s (10 Hz)**, ego-frame XYZ **+ 3×3
  rotation per waypoint (full SE(3))** (`:89`; `hf_card.md:89`).
- **Chain-of-Causation (`:92-106`):** ≈3.7 M CoC traces; released inference = backbone generates
  CoC text → action expert samples the trajectory conditioned on it. R1 predecessor (arXiv
  2511.00088, NOT Super): +12 % planning accuracy on challenging cases, −35 % close-encounter rate
  closed-loop, RL post-training +45 %/+37 %, 0.5 B→7 B monotone, 99 ms on-vehicle.
- **Inputs (`:72-83`):** 6 of 7 cameras × 4 synchronised frames ending at t₀; ego history =
  translation + 9-D rotation with timestamps; **no lidar, no radar**.
- **Performance (`:128-139`):** LingoQA 79.2; AlpaSim closed-loop 910 NuRec scenarios 1.50 ± 0.13;
  open-loop 1,434 challenging PhysicalAI-AV samples **minADE₆ @ 6.4 s = 0.911 m** ⚠️ best-of-6
  oracle selection.
- **Envelope (`:141-152`):** 1× H100 80 GB; measured peak 72,115 MiB; ⛔ does not fit our A40.
- ⛔ **No paper for Super** — every architectural inference is read off `config.json` (`:175-177`).

### 2.2 MEASURED (our probes of `records.parquet`)

- **`Sayood/tanitad-alpamayo2-augmentation/records.parquet`: 23,644 rows · 4,729 clips · 5 tasks**
  — trajectory / meta_action / auto_labeling / vqa 4,729 each, grounding_via_vqa 4,728; `error`
  non-null 0; `model_id=nvidia/Alpamayo2-Super` on all rows
  (`…/2026-08-16-s2-strategic-gap/S2_STRATEGIC_GAP.md:155-161`; `MODEL_REGISTRY.md` §11.1).
- **Trajectory-task shape:** `pred_xyz_shape = [1, 1, 1, 64, 3]`, `num_trajectory_samples: 1` —
  independently confirms the 64-waypoint contract (`campaign-science-addendum.md:450-453`).
- ⛔ **Two disjoint `raw_json` schema variants (corrected 2026-08-16):** 4,474 rows carry
  `cot · pred_xyz · pred_rot · logprob · ade_vs_gt_m` (`ade_vs_gt_m` mean **2.2584 m**, median
  1.5245, p10 0.4061, p90 5.0252); 255 metric-block rows carry `min_ade_m` (mean 2.3469) but **no
  `pred_xyz`**; 4,474 + 255 = 4,729. ⛔ Not comparable to the card's minADE₆ 0.911 m; do not pool
  the two estimators.
- **Throughput MEASURED:** full 5-task battery ~59.6 s/clip (78.4 wall-hours total) — 3.4× cheaper
  than the DESIGN estimate.
- **Meta-action label structure, precisely:** three axes × 7 values, ONE emission per clip at t₀,
  no per-timestep index, no declared horizon. **NOT a lat×lon product** — three parallel enum
  lines, one of which (`Lane`) is a third axis our v6 vocabulary partly folds into LAT.
- ⛔ **Registry usage limit (`MODEL_REGISTRY.md:3476-3480`):** "The LON axis always abstains ⇒
  Alpamayo cannot corroborate a longitudinal tactical label. This is why `a_tac_lon` is
  independently corroborated on **0/201** clips of the aug120 fused corpus."
- **Alpamayo-vs-us tactical comparison (n=39, `ALPAMAYO2_SUPER_ANALYSIS.md:663-760`):** two
  instrument defects corrected (stopped-window yaw; a 0.15 rad gate ~6.5× the typical turn — human
  median |net yaw| over 2 s = 0.023 rad). Gate sweep: at 0.10 rad both arms **0.7292 vs 0.7263 —
  indistinguishable**; at 0.01 Alpamayo declared 0.4660 vs ours 0.1159. ⛔ RETRACTED: "our executed
  κ 0.4968 beats Alpamayo's 0.3333". ⭐ The arms move in **opposite directions** as the gate
  tightens — Alpamayo's declaration carries **fine** lateral information (nudges) that a
  severity-free 5-way softmax has no slot for.
- **PARITY stamp (`MODEL_REGISTRY.md:3452-3456`):** A2 is a SEPARATE labelled corpus, never an
  extension of the parity set `physicalai-train-e438721ae894` (2,376 episodes, skip-hash
  `f09e44db`).

---

## 3. C112/C113 parity exclusion — registry §12.4 + `parity.py` §10

**Registry §12.4** (`MODEL_REGISTRY.md:3673`-onward): **201 of 4,729** Alpamayo clips are inside
`physicalai-train-e438721ae894` (mtimes prove genuine selection membership). ⛔ **The 4.3 % figure
is WRONG in the flattering direction — CORRECTED 2026-08-18 (C113). QUOTE 78.21 %**: only **257**
of the 4,729 have w120 video built; **201 of those 257 are parity-train** ⇒ the buildable Alpamayo
eval split is **78.21 % contaminated — REF-A I-JEPA SCALE (~80 %)**. Root-cause class: *a
contamination rate quoted over the CATALOGUE rather than over the BUILDABLE SET.*
⭐ The 201 **ARE** the aug120 perception corpus exactly (index hashes byte-identical to the
exclusion list, 201/201). ⭐⭐ Other direction: **6 of the 40 canonical val episodes (15.0 %) are
inside the Alpamayo record set** (verified two ways); blast radius today ZERO; trigger already
scheduled — the 4,472-clip build. ✅ Blast radius on published numbers: ZERO — all 73
`taniteval/results/*.json` opened; none evaluate on Alpamayo.

**The exclusion oracle — `stack/tanitad/data/parity.py` §10** (header `:1805-1848`; file 2,446
lines): `clip_digest` `:1858-1864` (sha256 of ONE clip id); `load_clip_digests` `:1867-1943`
(self-checks + `digest_of_digests`); `parity_train_clip_digests` `:1946-1948` (frozenset
membership oracle); `clips_in_parity_train` `:1951-1959` (counts-only printing); ⭐
**`assert_eval_clips_disjoint_from_parity_train`** `:1962-2034` — raises `ParityViolation` with
*"in {corpus_key}: {nb} clip(s) … <-- LEAK"* and the text *"This is the REF-A I-JEPA class (~80 %
of val inside train…) — **Provenance is not disjointness; ids are.**"*; the only bypass is
`sanctioned_audit=<reason>` (a string, not a boolean), which stamps `decision_grade: False`
(`:1997-2005`). `filter_eval_clips` `:2037-2067` is the sanctioned removal path; §10b
`filter_train_clips` `:2071-2215` the converse; §10c the INGEST GATE `:2217-2260` — *"§10 and §10b
are available; nothing invokes them. A rule that depends on the next agent remembering it is not a
guard."* 🔒 Clip ids are never printed.

---

## 4. REF-C tactical head history — D-TAC1 / D-TAC1B

### 4.1 The factorised LAT×LON head design (`Project Steering/PREREG_D-TAC1_FACTORED_TACTICAL_HEAD.md`, 308 lines, 2026-08-03)

Defect chain (`:59-68`): 5-way softmax over 3 lat + 2 lon classes; label = PRIORITY collapse
`turn > brake > accel > lane_keep` (`stack/scripts/refb_labels.py:100-109` v1, `:339-347` v2);
trainer supervises with v1, unweighted CE, `MANEUVER_WEIGHT = 0.1`; head logits enter anchor
confidences (H19, `Linear(5, n_anchors, bias=False)`); ⭐ the manoeuvre head never sees ego speed
while its label is `dv = v(t+2s) − v(t)`. The algebra (`:74-97`): `P5(accelerate) =
P_lat(lk)·P_lon(acc)` — a PRODUCT gated by lateral comparisons irrelevant to the longitudinal
question. §2.1 refutes "the lateral classes win every argmax" — turns emit at true rate; the
longitudinal mass lands in `lane_keep` (within-lane_keep marginal steady 74.9 / brake 11.5 /
accel 13.7 %).
Design (`:142-158`): one shared trunk, two linear readouts — MEASURED 104,191,577 → **104,192,474
= +897 params** (two independent MLPs would cost +272,001; the capacity test caught it);
`tactical_speed_input` = speed channel only (nav_cmd constant at eval ⇒ C6); two summed anchor
grafts `lat_to_anchor` (default init) + `lon_to_anchor` (zero-init); `man_prior_tau` over
uniform-init prior buffers. Trainer: two masked CEs at LAT 0.05 + LON 0.05 = 0.10 exactly;
fail-loud labeler-drift guard. 24 tests.

### 4.2 Pre-registered outcomes vs measured results (D-TAC1)

Result package: `…/Architecture & Inference/Implementation/incoming/2026-08-03-dtac1-tactical-head/`
(`DTAC1_RESULTS.md`, `DTAC1B_RESULTS.md`, probe JSONs, substrate `.pt` md5 `e7793439…`,
`adversarial-verification/`). Run: refc-base ckpt 29999, canonical val, **39 episodes / 1364
windows**, stride 5, on `tanitad-thor`. Controls first: label-derivation agreement 1.0000; shuffled
`auc_lon_active` 0.4933, factored macro-recall 0.3278 (chance).
**Shipped 5-way decode:** lane_keep 818/1078/0.9743; turn_left 174/165/0.8218; turn_right
109/114/0.8349; **accelerate 146/0/0.0000**; **brake_stop 117/7/0.0256**; accuracy 0.7581. Lateral
in isolation: macro-recall 0.8290.
⭐ **The label destroys 9.68 % outright** (132/1364 live-longitudinal windows labelled turn).
**E-A1:** `auc_lon_active` **0.7294** (≥0.65 threshold) ⇒ **READOUT-limited**; ⛔ the registered
INPUT-limited prediction REFUTED (R-2026-08-03-dtac1).
**E-A2** (episode-disjoint logistic): pooled 0.3833; v0 0.3416; **pooled+v0 0.4346** (+0.051,
seed-stable).
**τ frontier** (true counts brake 153 / steady 969 / accel 242): at τ=0 brake recall 0.072, accel
0.045; τ=0.5 brake **0.503** at acc −0.109; τ=1.0 macro-R peak 0.4761; ⛔ **accelerate NOT
recoverable at ANY τ** (peaks 0.153). Revised lever ordering F3 > F2 > F1 (prereg had F1 > F2 >
F3). `pytest`: 1808 passed.

### 4.3 τ selection and the F-1 arm (D-TAC1B)

Two defects closed: τ AND the class prior were read off val (R6); F1 was coupled to F2 (no arm
isolated INPUT). Protocol: LOEO over 39 folds, two co-primary criteria fixed in advance, both
denominators (ALL 1364 / REPRESENTABLE 1232), prereg sha256 hashed before the run.
**τ selection:** PRIMARY-B modal OOF τ **0.50** (36/39 = 92.3 %) — the parent's published τ was not
badly overfit. **Paired bootstrap (B=4000) vs τ=0:** macro-F1 rule on ALL — Δmacro-R +0.0719
[+0.0200, +0.1286] separated, but Δacc **−0.1129 [−0.1861, −0.0471] separated WORSE**; on
REPRESENTABLE — **ΔF1 +0.0107 [−0.0418, +0.0665] NOT separated** (the patch is indistinguishable
from doing nothing where the 5-way label can represent the answer). Selecting on macro-recall is
self-defeating (buys +0.1069 recall for −0.2757 separated accuracy). Empirical null: the full OOF
pipeline on shuffled logits scores macro-recall 0.3678, not 1/3.
**The F-1 arm:** `refc_f1only_config()` — **+384 params = exactly one speed column** into
`maneuver_head.0`; decoder bit-identical; negative control: with the flag OFF, `maneuver_logits`
is **bit-identical across v0 = 0 → 25 m/s — that IS the defect**; ON, they move and
`anchor_logits` moves too (speed reaches SELECTION). Pre-committed thresholds
(`PREREG_D-TAC1B…:154-156`): load-bearing = LON macro-recall separated above base **and ≥ +0.03**;
inert = CI includes zero; hurts = separated below on LAT or LON ⇒ revert. ⛔ +0.051 is a
linear-readout LOWER BOUND, not a prediction.
**Why no train-selected τ:** no train epcache reachable (5 probes, 2 hosts); the banked substrate
is val-only; refc-base predates the prior buffers. A train-selected τ would be biased low.

### 4.4 Current status

| lever | status |
|---|---|
| **F3** prior-corrected decode | built, run, honestly re-selected. ⚠️ **Do NOT publish as default** — separated accuracy loss, brake precision 0.1711, null on representable windows. Optional reporting mode at τ = 0.5 (`DTAC1B_RESULTS.md:261-266`). |
| **F2** factorised head + `lon_to_anchor` graft | **code + tests + trainer wiring staged; NOT TRAINED.** Retrain justification: 9.68 % label destruction, `accelerate`, the selection graft. |
| **F1** speed input | **decoupled and staged; NOT TRAINED.** |
| arms `refc-base-30k` / `dtac1-full` / `dtac1-f2only` / `dtac1-nolon-graft` (+ `dtac1-f1only`) | **pre-registered, none launched. 0 GPU spent.** |
| R13 | stands: `maneuver_decision` still collapses the longitudinal class on turns; readers must use `lon_decision`. |

### 4.5 The 5-way longitudinal-blindness root cause — registry citation

`MODEL_REGISTRY.md:1703-1715` (T1 four-family rescore, 6,844-window grid): lateral decision acc
0.7515 κ 0.3795; **longitudinal acc 0.3327 κ 0.0405 (chance)**; collapsed 5-way κ 0.1404 —
verbatim: *"The collapsed 5-way (κ 0.1404) sits between the two axes and reports neither — the
direct measurement of the lat/lon-mixing softmax defect CLAUDE.md names as our largest known
architectural problem, and it is only visible because the family is reported factored."*
`MODEL_REGISTRY.md:3467` names the anchor `longitudinal-blindness-root-cause`; the same defect one
level up in the goal vocabulary is documented at `v6.py:215-225`.

---

## 5. `refc_tactical` probe headers

**`stack/scripts/refc_tactical_probe.py`** (docstring `:1-56`): E-A1 counterfactual factored decode
via `invert_man5` (threshold-free `auc_lon_*`; AUC ≈0.5 ⇒ INPUT-limited, well above 0.5 with 0
emissions ⇒ READOUT-limited); E-A2 episode-disjoint logistic probe; negative controls FIRST
(shuffled; label_source — a disagreement ⇒ "NOTHING below is quotable"). Estimator
`taniteval.ci.episode_cluster_bootstrap`; never on a training pod.
**`stack/scripts/refc_tactical_tau_select.py`** (docstring `:1-60`): LOEO τ+π selection, 0 GPU;
names R6 (the prior was the VAL label marginal; ±25 % brake-prior perturbation swings accelerate
recall 5.6×); reporting rules R3 (recall AND precision AND F1), R2/R10 (both denominators), R7
(paired episode-cluster bootstrap only), R8 (the lateral readout is CLASSIFICATION, not the LATERAL
kinematics family). States: "THIS IS NOT A TRAIN-SELECTED TAU".

---

## 6. Comfort / jerk in the program record (excluding C101's CEM cost)

### 6.1 Where jerk/comfort live in `taniteval` (MEASURED by grep + read)

| module | metric | line |
|---|---|---|
| `closedloop.py` | `_comfort(win)` → `mean_abs_jerk_mps3`, `frac_steps_exceed_jerk_comfort`, `mean_abs_accel_mps2`, `mean_abs_lat_accel_mps2`, …; `jerk = |Δa|/DT` | `640-668`; constants `A_LON_COMFORT 2.0`, `A_LAT_COMFORT 3.0`, **`JERK_COMFORT 2.0 m/s³`** at `164-166` |
| `generalization.py` | `jerk_max` (3rd difference of position); `smooth` deliberately separated from `feasible` | `237-249`; `JERK_MAX 60.0` at `79-82` |
| `pseudosim.py` | `comfort` within-bounds flag; `COMFORT_LIMITS {a_lon 3.0, a_lat 3.0, jerk_max 8.0, yaw_rate 0.95}` ⚠️ PROPOSED constants | `1206`, `1245-1256`, `803-805` |
| `tanitad_metrics.py` | **TMS** = `1/(1 + α·∫|jerk| + β·∫|steer_rate|)`; `JERK_BRAKE_THRESHOLD −1.5` | `18`, `63-66`, `165-202` |
| `bench.py` | `tms_openloop` on predicted waypoint paths | `75-97` |
| `planner_p2.py` | `comfort = mean(a²) + mean(jerk²)` inside the P2 cost | `34`, `187-189` |
| `refc_rerank.py` | reuses `planner_p2.cost_fn` VERBATIM on REF-C's 256 candidates | `20-35` |
| `driving.py` | jerk is tier-1 UNIMPLEMENTED (no longer BLOCKED — dense 20-step path persisted since 2026-07-25) | `217`, `573-577` |
| ⛔ `four_families.py` | **contains no `jerk` and no `comfort`** — comfort is NOT a member of the binding metric block | MEASURED by grep |

### 6.2 Measured jerk numbers of arms

**v1 flagship baseline defects (`…/2026-08-06-v1-defect-triage/`):** intra-plan jerk RMS
**52.2148 vs human floor 1.7066 = 30.6×** (6,834 windows); triage doc: 64.2966 = 35.8× — *"a third
difference of waypoints, which is exactly why 64.3 is reachable without anything noticing"*.
Retiming A/B: 52.1281 → **4.9535** (−90 %); human p99 pass values **accel 2.6890 m/s², jerk
6.3686 m/s³**.
**v1.6 `unicycle-readout-v2-latentsonly` (`MODEL_REGISTRY.md:1179-1259`):** frozen trunk +
`UnicycleStepReadout` (2.11 M trainable), loss = pos-L1 + 0.3·heading + 0.5·net-yaw + 0.05·accel-
barrier + 0.05·jerk-barrier (barriers above TRAIN-corpus human p99, dense 0.1 s grid). On 6,834
windows, paired episode-cluster bootstrap: **jerk RMS 36.1682 → 1.1334, Δ −35.0348 [−46.6907,
−24.8134] separated** (human ≈1.71); accel RMS 2.9465 → 0.7172 separated; **replan accel jump
1.1310 → 0.1016 (11×)**; gate N2 jerk ≤ 3.42 → 1.567 PASS. ⚠️ The PRIMARY gates P1/P2 FAILED
(reliance 0.1547 vs ≥0.40) — the readout fixed comfort, not reliance. Design note: *"a delta
head's natural output scale IS THE JERK"*; a plain `λ·jerk²` would punish legitimate emergency
braking — the barrier form exists for that reason.
**WM-MPC free-floor rung 3 (`…/2026-07-23-freefloor-rung3-wm-mpc/VERDICT.md:49-50`):** stochastic
action selection adds lateral jitter — arm C mean |jerk| 3.78 m/s³, separated-worse off-road proxy.
**⛔ C46 (`RETRACTION_LOG.md:680-691`) — a comfort term that rewards not driving:** the human's own
logged path fails the comfort bounds on **16.60 %** of windows while `cv_holdv0` and `stand_still`
both score a perfect 1.0000; every learned planner floors on the jerk clause ⇒ weight set to 0.0,
measurement retained as diagnostic. Standing consequence: *validate any threshold against the
ground truth before it carries weight.*

### 6.3 The speedjerk-30k story (registry)

⛔ **Jerk was never an ACTION INPUT.** `flagship4b-speedjerk-30k` (`MODEL_REGISTRY.md:166-197`)
sets three flags at once: `--speed-input --jerk-weight 0.02 --aux-accel`, `action_dim=3`. **Speed
IS the action input** (`v0 = poses[t,3] / 10.0` as the 3rd action channel,
`stack/tanitad/train/flagship_losses.py:228`; `SPEED_SCALE = 10.0` a hard contract). **Jerk is a
LOSS weight** and `aux_accel` a 528,897-param aux head. **The causal attribution belongs to speed**
(D-A3: REF-A 3.73 → 0.83 m validated in isolation; flagship no-speed 2.918 vs speed 0.452, paired
+2.21 m [2.04, 2.39], win-rate 83.8 %). The control arm differs in `speed_input`, `action_dim`,
`jerk_weight`, `aux_accel` together ⇒ **the ladder isolates SPEED and leaves the jerk loss
confounded; no arm anywhere isolates the jerk loss.** Deployed v1 numbers (881 windows): ADE@2s
0.4522 heldout / 0.4271 full-set — ⛔ this is `wm_fidelity_ade_2s` (`rollout.py:170`
`actions_source="expert_future"`): **not a planning bar.** 🟥 RECONSTRUCTION RISK
(`MODEL_REGISTRY.md:410-416`): the committed trainer has no `--jerk-weight`/`--aux-accel` args yet
the run's `config.json` records them — a clean-checkout rebuild of v1 is not byte-exact today.
Downstream: `v2_traj_jerk = 0.02` was lever #6 of the ten-lever v2 pack (killed at 6 k, attributed
to all ten levers at once, not to any one).

---

## 7. Source manifest

Principal files (under the worktree root): the 2026-08-16-tactical-labels package (read fully);
`PREREG_D-TAC1_FACTORED_TACTICAL_HEAD.md` + `PREREG_D-TAC1B_TAU_SELECTION_AND_F1_ARM.md` (full);
`…/2026-08-03-dtac1-tactical-head/DTAC1_RESULTS.md` + `DTAC1B_RESULTS.md` (full);
`stack/tanitad/data/parity.py:1805-2260`; `MODEL_REGISTRY.md` §§1.1-1.4, v1.6, T1 rescore, 11.1a,
12.2-12.4; `RETRACTION_LOG.md` (C46 + jerk/comfort hits);
`…/2026-08-05-alpamayo2-super/ALPAMAYO2_SUPER_ANALYSIS.md` + raw card/readme;
`Reports/2026-08-15-2200-campaign-science-addendum.md`; `…/2026-08-16-s2-strategic-gap/
S2_STRATEGIC_GAP.md`; `refc_tactical_probe.py` + `refc_tactical_tau_select.py` headers; targeted
taniteval reads (§6.1 table); `v6.py:147-231`; the v1-defect-triage package; `…/2026-07-23-
freefloor-rung3-wm-mpc/VERDICT.md`.
**Negative findings (MEASURED by grep):** REFERENCE_ARCHITECTURES.md and
REFERENCE_SYSTEMS_RANKED.md contain zero "alpamayo" occurrences; `four_families.py` contains no
jerk/comfort; no banked doc reports the Alpamayo trajectory task as per-timestep meta-action
supervision.
**Not verified here:** no probe re-run; `records.parquet`, the substrate `.pt`, and several raw
JSONs not opened — numbers marked MEASURED are measured by their owning package and INHERITED into
this dossier, except file listings, grep counts, and the `v6.py`/`parity.py` line offsets, which
were measured directly. No files were created, modified, staged, or committed by the compiling
agent.
