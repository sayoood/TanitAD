# RESULT — does the frozen refav1 trunk carry trajectory information beyond t0 kinematics?

**2026-09-27, TrainingFlyWheel. Staged, never committed.** SPEC: `SPEC.md` (§1–§6 staged 14:07:38; Amendment A1
14:16:23; A2 14:32:15 — each before the data it governs existed). Checkpoint `refav1-b1-v72-ep3-speed` step 21,109
(md5 `1189bc02…`). **Estimator:** paired episode-cluster bootstrap over the 141 EVAL episodes, n_boot 2,000.
**Tier:** T1-equivalent (emitted from the t0 state only; no future input, no action rollout). Function class: stated
per arm. Every number is **MEASURED** from the JSON named beside it.

## 1. §4 — mean-pooled state (exactly what `plan()`'s heads read), 5-fold CV inside the 141 eval episodes
`raw/out/probe_result.json`, `raw/out/pd_probe.{md,json}` · n = 2,399 windows / 141 episodes (~113 fitted per fold) ·
d: trunk 1,024, raw 3 × 1,024, kinematics 8 · λ and PCA m chosen by inner grouped CV on the training folds only.

| pair (ADE @ 2 s, m) | Δ [95 % CI] | reading |
|---|---|---|
| **`P4 trunk − P2 kin` (BAR-T1, primary)** | **+0.0042 [+0.0016, +0.0071]** | trunk readout separated-**WORSE** than kinematics alone ⇒ **NEGATIVE** |
| `P4 trunk − P0 kd_x` (BAR-T2) | −0.0516 [−0.0669, −0.0376] | the gain is the kinematic readout's, not the trunk's (next row) |
| `P2 kin − P0 kd_x` | **−0.0558 [−0.0702, −0.0429]** | ⭐ a LEARNED kinematic readout beats the strongest fixed floor on held-out episodes |
| `P3 raw − P2 kin` | +0.0045 [+0.0020, +0.0072] | raw DINOv3 at the same pooling: also worse |
| `P4 trunk − P3 raw` | −0.0003 [−0.0022, +0.0017] | adapter ≈ raw encoder at 2 s |
| `P5 shuf − P2 kin` (validity) | +0.0022 [+0.0009, +0.0037] | shuffled trunk NOT better ⇒ valid |
| `P1 const − P2 kin` (validity) | +0.0434 [+0.0331, +0.0553] | constant NOT better ⇒ valid |
| `P0 − P0` (known value) | +0.0000 [+0.0000, +0.0000] | exact; `P0` reproduces `kd_x` with max \|Δ\| 0.0 |

Inner CV picked the SMALLEST PCA size (m = 8) for every image arm in every fold at 2 s (λ 100–316); at 6 s several
folds hit the λ grid's ceiling (1e4) — the fit wanted MORE shrinkage than the grid allowed, i.e. the 6 s rows are
underpowered by construction and are reported, not read. Four families (`raw/out/pd_probe.md`): `P4 − P2` is
separated-worse on ADE, FDE, all three LON metrics, cross-track, yaw-rate and lateral tactical correctness.
6 s (n = 2,399): `P4 − P2` ADE +0.0516 [+0.0118, +0.0911]; `P4 − P3` −0.0501 [−0.1027, −0.0010].

## 2. A1 — the same readouts on a 4 × 10 SPATIAL pooling of the token field (5-fold CV inside eval)
`raw/out_sp/probe_sp_result.json` · n = 2,399 · d: trunk_sp 40,960, raw_sp 81,920 (frames t0, t0−1), kin 8 · PCA by the
Gram trick, basis on training rows only · known-value control: this run's `P2` reproduces §1's `P2` with max |Δ| 0.0.

| pair (ADE @ 2 s, m) | Δ [95 % CI] | reading |
|---|---|---|
| **`P6 trunk_sp − P2 kin` (BAR-S1, primary)** | **+0.0037 [+0.0011, +0.0065]** | separated-**WORSE** ⇒ **NEGATIVE** |
| `P8 shuf_sp − P2` (validity) | +0.0009 [−0.0002, +0.0019] | shuffled NOT better ⇒ valid |
| `P7 raw_sp − P2` | +0.0069 [+0.0038, +0.0102] | raw DINOv3, same pooling: worse |
| `P6 − P7` (adapter vs raw, same pooling) | **−0.0032 [−0.0057, −0.0010]** | ⭐ the adapter's field IS better than raw DINOv3 — but neither beats kinematics |
6 s (n = 2,399; λ at the grid ceiling in 3/5 folds — underpowered, reported): `P6 − P2` ADE +0.0553 [+0.0092, +0.1033];
`P6 − P7` ADE **−0.0462 [−0.0883, −0.0052]**, FDE **−0.1751 [−0.3192, −0.0329]**.
Inner CV again chose the smallest image capacity (m = 8) in every fold at 2 s. Four families
(`raw/out_sp/pd_probe_sp.md`): `P6 − P2` separated-WORSE on ADE (+0.0037), LON speed (+0.0056), LON along (+0.0037)
and cross-track (+0.0016); better on none; `P6 − P7` better on ADE and LON along only.
⇒ Keeping the spatial layout does not rescue a LINEAR readout fitted on ~113 episodes. A1's committed consequence
(a nonlinear readout) is Amendment A2's P9, run together with A2's power fix (fitting on 600 TRAIN clips).

### 2b. How to read §1–§2 (PUBLISHED anchors, banked in the Library)
The pattern — t0 ego kinematics dominate a 2 s open-loop displacement metric and learned scene features add little —
is a documented property of open-loop evaluation, not only a refav1 defect: on nuScenes an MLP fed ONLY ego-state
inputs (past trajectory, velocity) matches perception-based planners on L2 (Zhai et al. 2023, library key
`2305.10430`), and end-to-end models that include ego status come to rely predominantly on it and under-use
perception (Li et al., CVPR 2024, `2312.03031`). The programme measured the same thing on a second architecture before
today: v7's E-WC2 (2026-08-16) found a 0-parameter constant-yaw-rate goal predicting better than a ridge from REF-C
latents (`scripts/v6_chain.py:501`, *"these latents are the wrong surface"*). ⇒ Where vision must matter is where
kinematics fail: longer horizons (R5's 6 s) and the event windows (turn onsets, stops, lead-vehicle braking) — the
readouts and the retrain bar should be read there, not on the pooled 2 s mean alone.

## 3. A2 + A3 — fitted on 600 refav1-TRAIN clips, scored on the 141 EVAL episodes
`raw/out_te/probe_te_result.json` (A2 alone: `probe_te_result_A2.json`), `verdict_a2.txt`, `verdict_a3.txt`,
`pd_probe_te.md` / `pd_probe_a3.md` (four families) · fit: 5,398 TRAIN windows / 600 episodes · scored: the same 2,399
EVAL windows as §1/§2.

| arm (MEASURED, T1-equiv.) | ADE @ 2 s | ADE @ 6 s |
|---|---|---|
| P0 `kd_x` (fixed damped prior) | 0.3085 | 3.5584 |
| P2 `kin` (ridge, 8 kinematic features) | 0.2518 | 3.4281 |
| P4 `trunk` (ridge, global mean) / P6 `trunk_sp` (ridge, 4 × 10) | 0.2539 / 0.2530 | 3.4678 / 3.4526 |
| P3 `raw` / P7 `raw_sp` (ridge, raw DINOv3) | 0.2521 / 0.2523 | 3.4600 / 3.4554 |
| P9 `trunk_sp_mlp` (MLP, trunk 4 × 10 + kinematics) | 0.2282 | 3.0244 |
| **P10 `kin_mlp` (the SAME MLP, kinematics ONLY)** | **0.2229** | **2.9257** |
| P11 `shuf_sp_mlp` (P9 with shuffled trunk rows) | 0.2312 | 2.9738 |

**A2 (as committed): PRIMARY `BAR-A2` NOT MET** — `P6 − P2` +0.0012 [−0.0014, +0.0040]; validity OK. The nonlinear
secondary `P9 − P2` was separated-better (2 s −0.0236, 6 s −0.4037) — but `P2` is linear, so A3 was pre-registered
(staged 16:36:22) before either capacity control ran.
**A3 (as committed): NEGATIVE, validity OK.** `P9 − P10`: 2 s **+0.0054 [+0.0016, +0.0093]**, 6 s **+0.0987
[+0.0332, +0.1658]**, 6 s FDE +0.2349 — adding the trunk to the MLP makes it separated-WORSE; `P11 − P10` +0.0083 /
+0.0481 (the shuffled trunk does not help either). Four families (`pd_probe_a3.md`): `P9 − P10` worse on ADE, FDE, LON
speed, LON accel, cross-track and yaw-rate; better on nothing. The only stratum where the trunk helps is 2 s `stop`
(−0.0690, 14 windows / 11 clusters) — reported, not read.
⇒ **The frozen refav1 trunk (DINOv3-L fp8 → WideAdapter at step 21,109) carries NO trajectory information beyond the
t0 kinematics that any readout we tried extracts** — linear or MLP, global or 4 × 10 spatial pooling, fitted on ~113 or
600 episodes, at 2 s or 6 s. A2's apparent nonlinear gain was nonlinearity on the KINEMATICS.

⭐ **The positive result the same test produced: a learned NONLINEAR kinematic readout.** `P10` — an MLP on 8 t0
kinematic features (v0, a0, κ0 and products; past measurements only), fitted on 600 TRAIN episodes, scored on the 141
held-out EVAL episodes — beats every kinematic floor, separated: vs the linear readout 2 s −0.0289 [−0.0421, −0.0171],
**6 s −0.5024 [−0.6331, −0.3767]**; vs `kd_x` 2 s −0.0857; vs `ha0_ext` 2 s ADE **−0.3495**, FDE −0.9272, heading
−2.92°, cross-track −0.320 m (worse only on acceleration MAE +0.104). At 6 s: **2.93 m ADE vs `damp50` 3.45, `kd_x`
3.56, `ha0_ext` 5.36.** This is AD-MLP's result (`2305.10430`, banked) on our data: ego status alone, fitted, is a
strong planner at open-loop metrics. ⇒ (1) it is the bar any refav1 / v7F / refc head must beat before "vision helps"
is claimed; (2) it is the best measured residual PRIOR for an R5 trajectory head (a learned kinematic prior beats every
fixed one by 0.5 m at 6 s).

**Run provenance.** Train set: 600 clips (first by sha12 of the 4,572; `raw/train/train_pick_sha12.json`), pulled
read-only from Thor, encoded 600/600 with 0 skips (2,834 s), extracted to **5,398 windows** at stride 8 (684 s).
⚠️ **The first A2 run CRASHED** (`MemoryError`: 632 MiB for a (4,048 × 40,960) float32 copy) at the first spatial arm,
while the full test suite and the R1 GPU arm shared the box; its wrapper reported exit 0 (the `echo` after it) and
only the artifact check caught it (`ZZA2-NO-RESULTZZ`). It is kept as `code/probe_te_v1_oom.py`. The re-run uses
`code/chunked_gram.py`: the SAME Gram-trick PCA computed in column chunks from the fp16 features (no full float32
copy), **equivalence-checked against the original readout to max |Δ| 2.6e-7 on outputs of scale 0.70** before use;
arms, folds, grids and the shuffle permutation draws are unchanged (same RNG, same arm order).
⚠️ **The second run crashed too** — on a 63 MiB allocation, after completing P2–P6, with 11 GB of PHYSICAL RAM free:
the Windows COMMIT limit (86.6 GB = RAM + pagefile) was exhausted by concurrent processes, the largest a parallel
refav1 test suite committing 23 GB (one of this session's own streams). ⇒ The final run is RESUMABLE per arm
(`code/patch_resume.py`: each arm's out-of-fold predictions are saved atomically as `raw/out_te/arm_<name>.npz` the
moment it completes; the shuffle permutations are drawn in arm order even for a resumed arm, so a resumed run equals
an uninterrupted one).

## 3b. What this decides (the committed consequences, applied)
- **Retrain option H (heads-only on the frozen trunk) is NOT the lever** — A3's committed NEGATIVE branch. A head on this
  trunk would learn P10 (the kinematic MLP) and nothing more.
- **The lever is the trunk itself (option T)**: the adapter / world model retrained with the trajectory objective as a
  first-class loss (the R5 head trained jointly, not frozen) — PI decision on the ~303 GB fp8 features (or on-the-fly
  encoding on Thor from the v2ep cache already there) and a slot, behind a `refav1` launch-gate profile.
- **Immediately usable at zero cost:** the learned kinematic prior (P10) as (a) the programme's "beats doing nothing" bar
  for learned heads and (b) the residual prior of the R5 head (refav1 `--r5-prior`, v7F option B's decoder, refcv7 NEW-1).
- ⚠️ Scope: one checkpoint (21,109), frozen trunk, open-loop T1-equivalent readouts, PhysicalAI eval grid. It says
  nothing about a trunk trained WITH a trajectory objective, and nothing about closed loop.

## 4. Controls and provenance
- The extractor's GT and every floor are **bit-identical** to the banked full-grid A1 dump on every shared window
  (checked on 2 episodes before the SPEC; `extract.py` imports `refav1_arm`'s own loader, window selection, GT and
  floor builders).
- Spatial pooling checked two ways: the mean of the 40 regions equals the global `pooled` (to fp16 storage precision,
  4.8e-4), and a synthetic token-index ramp gives region (1, 7) = 249.5 by reshape and by explicit index.
- A2's train features: the encode path reproduces the SHIPPED eval fp8 cache **bit-exactly** on an eval clip (100.00 %
  identical e4m3 codes, rel-L2 0.0000); the train set equals the v7.2 TRAIN label release exactly (4,572 = 4,572,
  0 either way) and is disjoint from the 141 eval clips; Thor was read, never written.
- Code: `code/extract.py`, `code/probe.py`, `code/probe_sp.py`, `code/probe_te.py`, `code/pull_encode.py`,
  `code/run_chain.sh`. Features (0.6–4 GB) stay on the dev box (`C:/Users/Admin/refav1_probe/feat_*`,
  `C:/Users/Admin/refav1_trainprobe/`): regenerable in minutes from the eval cache / Thor's B1 sources by the banked code.

## 5. Erratum (the SPEC is left as staged; corrected here)
`SPEC.md` §5 and §8 describe the heads-only alternative as avoiding *"the ~155 GB fp8 cache"*. **Wrong artifact:**
~155 GB is the v2ep SOURCE-frame cache (4,572 × 34 MB) that the encoder reads; the fp8 feature cache the refav1
trainer reads is **~303 GB** (4,572 × ~66.2 MB, `T_c × 655,360 + 1,908` B per episode — the encoder's documented
size; the 2 train episodes encoded in A2's smoke are 66,193,289 B each). The comparison's direction is unchanged (a
heads-only head needs KB/window, a trunk retrain needs the full cache); only the size was wrong — by 2×. Class C82
(price the file the CONSUMER opens), committed by me; logged for `RETRACTION_LOG.md` via the handoff.
