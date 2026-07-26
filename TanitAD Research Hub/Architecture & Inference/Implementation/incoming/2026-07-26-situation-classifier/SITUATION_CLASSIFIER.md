# THE SITUATION CLASSIFIER — lane change · roundabout · intersection

**Date:** 2026-07-26 (local, Europe/Berlin). **Author:** research engineer (situation-classifier stream).
**Pre-registration:** `PRE_REGISTRATION.md`, this folder, **written and staged before any classifier
weight existed**. Every checkpoint and every JSON in `artifacts/` is later.
**Host:** pod3 (A40, exclusive). pod1 (training), pod2 (the `tblind-ladder` co-tenant) and the eval
pod were not used for compute.

**Evidence classes:** `MEASURED` (ours + path) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Tiers:** `PROVISIONAL` / `CONFIRMED` / `DECISION-GRADE`.

🔒 **PhysicalAI-AV is gated-confidential.** No clip UUID and no raw content appears anywhere in this
folder; clips carry an integer index only. The UUID map lives on the dev box, outside the repo.

*(Every table in §3–§6 is rendered from `artifacts/sc_results.json` by `scripts/sc_report.py`. **No
number in those sections is typed by hand.**)*

---

# 0. VERDICT IN ONE BOX

<!-- VERDICT -->
*(filled from the held-out run)*
<!-- /VERDICT -->

---

# 1. What was built

The PI asked for a model that **classifies the situation so that additional cameras can be requested
only when the situation demands them**, reading **the front camera only**. Three things had to exist
and none of them did:

1. **The three situations had no label.** The label that existed (`L2`) triggers on `a_req` — required
   ego **deceleration**, a longitudinal braking hazard. The "junction" stratum used elsewhere is
   `JUNCTION_DEG = 10.0`, a heading-change threshold `corridor.py` explicitly forbids renaming
   "intersection". ⚠️ A sibling stream MEASURED that this proxy is only **37.2 % junction-scale**
   (49.8 % ambiguous, 13.0 % plain road curve) — INHERITED, `2026-07-26-situation-semantics/`.
2. **The supervision had to come from somewhere the model cannot see.** PhysicalAI-AV has **no map,
   no lane graph and no junction annotation** (settled at five independent probes; not re-probed
   here). The labels are therefore built from **privileged geometry**: the ego's own pose track and
   `obstacle.offline` 3-D agent tracks. **Neither is a model input** — v1 receives the 256 px /
   51.4° front crop and its own speed (`action_dim = 3`).
3. **A trigger that fires *during* a situation is useless.** The frames are already gone. So the
   target is **anticipation by construction** (§2.4), and the **lead-time distribution is a
   first-class result with a minimum registered before it was measured**.

### 1.1 ⭐ The structural finding that made this study 5–9× better powered than H2

`physicalai.build_episode` stores `poses = [x, y, yaw, v]` at 10 Hz, produced by
`signals_at(ego, t_query)` and trimmed by the 3-frame stack — **index-for-index with the frames the
encoder sees** (`stack/tanitad/data/physicalai.py:470-509`). **The ego trajectory is already inside
every cached episode, on the camera clock.** Three consequences, all MEASURED:

| | H2 | this stream |
|---|---|---|
| universe | 26 `obstacle.offline` chunks ∩ the pod2 cache = **582 clips** | **2,376 cached parity episodes** (4.1×) |
| alignment | two clocks, an integer-lag speed correlation, **10.65 % of clips dropped**, the guard tripped on its own degeneracy | **no alignment step exists** for any ego-derived label — the label index *is* the feature index |
| held-out positive clusters | **35** | **158** (lane change) · **269** (intersection) · 27 (roundabout) |

### 1.2 Two coverage claims in the brief — one confirmed, one corrected

MEASURED from `metadata/feature_presence.parquet` (306,152 clips — the tool that owns the fact):

- ✅ **`obstacle.offline` coverage = 97.444 %** (298,326 / 306,152); **96.900 %** on the 3,000-clip
  parity selection. The brief's INHERITED-UNVERIFIED figure is **confirmed to three decimals**, and
  it matches the sibling stream's independent 97.4438 %.
- ⛔ **The brief's *"~500 multi-view clips (front + L + R + rear)"* is WRONG.** All **seven** cameras
  are present on **306,152 / 306,152 = 100.000 %** of clips and on **3,000 / 3,000** of the parity
  selection. **Multi-view coverage is not a limit on this study at all.** What is limited is the
  local *calibration* pull and the decoded-video cache — geometry and bytes, not availability.

---

# 2. The three labels

Defined once in `scripts/sc_situations.py`, imported everywhere, frozen in `PRE_REGISTRATION.md` §2.

- **Lane change** — over a 4 s window at ≥ 8 m/s: net heading change ≤ 8°, net lateral offset
  2.4–5.5 m (one lane), the offset is a *net* displacement not a wobble, **and** the yaw rate shows
  both lobes of an S (≥ 1.5° each) with the first lobe pointing toward the target lane.
- **Roundabout** — a same-sign curvature run at R ≤ 50 m lasting ≥ 3 s with |Δψ| ≥ 90°, curvature
  constancy (σ/μ ≤ 0.5), speed 2–14 m/s, **and** an entry/exit counter-deflection of ≥ 3° within 3 s
  on at least one side.
- **Intersection** — a **tight, quantised quarter-turn** (|Δψ| ∈ [45°, 135°], ≤ 6 s, R ≤ 25 m,
  roundabouts subtracted) **∪** a block of ≥ 1 s of **perpendicular cross traffic whose
  constant-velocity path CROSSES the ego's realised path within 40 m ahead**.

### 2.1 ⭐ The target is ANTICIPATION, enforced in the label

```
y_S(t)     = 1  iff an event ONSET falls in (t, t + 3.0 s]
valid_S(t) = 0  for every frame INSIDE an ongoing S event, and for the last 3.0 s of the episode
```

**Masking the in-progress frames is the whole point.** A head is never scored on a frame in which
the manoeuvre is already happening, so it cannot win by *recognising* what it is already doing. The
registered minimum useful lead time is **1.0 s** — a classifier below it **fails regardless of AP**.

---

# 3. The substrate and the power ceiling — measured BEFORE any classifier was read

<!-- TABLES:SUBSTRATE -->
<!-- /TABLES:SUBSTRATE -->

---

# 4. Label validation — five checks, each of which can fail

<!-- TABLES:VALIDATION -->
<!-- /TABLES:VALIDATION -->

---

# 5. Held-out results

<!-- TABLES:RESULTS -->
<!-- /TABLES:RESULTS -->

---

# 6. The multi-camera need

<!-- TABLES:CAMERA -->
<!-- /TABLES:CAMERA -->

---

# 7. Limitations, stated plainly

1. ⛔ **Every clip in this study is encoder-seen.** pod3 holds only the *train* parity cache, so the
   encoder-unseen sensitivity H2 could run cannot be run here (amendment A3). The confound is
   **balanced across TRAIN and HELD-OUT**, so it can inflate absolute AP but cannot manufacture a
   *difference between arms* — which is what the primary comparison measures. It nonetheless means
   **no absolute AP in §5 may be quoted as a generalisation number to unseen scenes.**
2. ⚠️ **`obstacle.offline` is `scene:obstacles:autolabels:v2` — machine labels** (`prov: "autolabel"`).
   Systematic misses of small or distant agents attenuate the cross-traffic half of the intersection
   label, the §4 V4 validation and the whole of §6.
3. ⚠️ **The labels are geometry, not ground truth.** There is no map, no lane graph and no junction
   annotation in this corpus, so *no* label here can be checked against a survey. What can be
   checked is internal and external consistency, which is what §4 does — and the roundabout label
   fails that test at ~18 % out-of-sample impurity.
4. ⚠️ **The cross-traffic half covers 450 of 2,376 clips** (obstacle chunk ∩ calibration ∩ the
   alignment floor). The intersection label on the other 1,926 clips is its TURN half alone. §4 V4
   is what licenses that, and it licenses it only to the strength of its own interval.
5. **The compute model is a per-camera encoder-pass model** (H2's, re-used). It excludes ISP,
   memory bandwidth for a second sensor stream, and the wake-up latency of a camera that is not
   already streaming. It is therefore **conservative** on the saving and **optimistic** on latency.
   Wall-clock, where quoted, is **A40 wall-clock and is not an Orin/Thor number.**
6. **The 3 s anticipation horizon and the 1.0 s minimum lead are registered choices, not derived
   optima.** They were fixed before measurement precisely so they could not be tuned to the result.

---

# 8. Amendments

| # | what changed | why, and what it can and cannot bias |
|---|---|---|
| **A1** | Two arms added to `PRE_REGISTRATION.md` §5.2 — a **closed-form ridge ladder** (`ridge_img_ego`, `ridge_img`, `ridge_ego`, `ridge_img_shuf`) — and **rank 16 promoted to primary** over rank 64 | Made **before any held-out score existed** (feature extraction was still running; the pre-registration had already been staged). A sibling stream MEASURED, on the same frozen v1 state: (i) a monotone swamping dose-response, ego 3.659× → +k16 3.685× → +k64 3.000× → +k256 2.116× → +k2048 1.59×, with 16 PCs carrying 97.0 % of the variance; (ii) a **2,049-parameter linear ridge probe SEPARATING** on `NOT_T_seen` where H2's 2.17 M-parameter head did **not**. Both are INHERITED. The additions are strictly **additive** — no bar, no primary arm, no operating-point rule and no outcome condition was changed. The ridge arms make a null *harder* to obtain, i.e. they bias against Outcome B. |
| **A2** | `detect_curves` (the §6.2 **control** population) uses its own lower curvature deadband (R ≤ 400 m instead of R ≤ 50 m) | With the roundabout deadband the control returned **3 events on the whole corpus** — a control that cannot be populated is the C13 failure mode (a guard that cannot fire). The lower deadband yields **180** curve events. The roundabout and turn detectors are untouched, so this cannot move any situation's label. |
| **A3** | The universe is the **train parity cache only** (2,376 of 2,976 episodes) | pod3 — the assigned host — holds `physicalai-train-e438721ae894` but **not** the val cache (`s3parity/views/…val…` is a directory of 4 KB stubs, MEASURED). ⚠️ **Consequence, declared: every clip in this study is encoder-seen, so the encoder-unseen sensitivity H2 could run cannot be run here.** The confound is *balanced* across TRAIN and HELD-OUT, so it can inflate absolute AP but cannot manufacture a difference *between arms* — which is what the primary comparison measures. |

---

# 9. Deliverable manifest

**Everything below is in the repo working tree and STAGED (`git add`). Nothing was committed or
pushed.** Path: `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-situation-classifier/`

| artifact | what it is | where it lives |
|---|---|---|
| `PRE_REGISTRATION.md` | label definitions, split, operating-point rule, minimum lead time, success bars, **both outcomes** — staged before any training | **repo** |
| `SITUATION_CLASSIFIER.md` | this document | **repo** |
| `artifacts/sc_results.json` | **the held-out result** — discrimination, above-chance test, operating points, lead-time distributions, efficiency curves, camera need, every control | **repo** |
| `artifacts/universe.json` | per-chunk / per-side counts per situation, and **C-POW measured before any classifier score** | **repo** |
| `artifacts/join_proof.json` | the 2376/2376 + 600/600 episode↔clip join proof, and the 0-left-hand-traffic-clips fact | **repo** |
| `artifacts/label_validation.json` | V1 heading-gate sweep · V2 roundabout direction purity · V3 turn balance · **V4 turn-is-a-junction** · V5 label profiles | **repo** |
| `artifacts/cross_summary.json` | C-ALIGN in metres + the **`obstacle.offline` clock-join proof** | **repo** |
| `artifacts/round_sweep.json` | the DEV roundabout constant sweep behind the pre-registered choice | **repo** |
| `artifacts/train_summary.json` | CV grid, selected config/epoch per arm, fold composition | **repo** |
| `artifacts/heldout_frames.npz` | ⭐ **every held-out frame**: labels, valid masks, ego channels and every arm's score — so any bar in this document is recomputable **with no GPU** | **repo** |
| `artifacts/sc_index.csv.gz` · `artifacts/sc_cross_index.csv.gz` | per-clip label and cross-traffic indices (`*.parquet` is git-ignored in this repo) | **repo** |
| `checkpoints/*.pt` | ⭐ **the trained heads** — `head_img_ego` (PRIMARY), `head_img`, `head_ego`, `head_img_ego_concat` (the swamping ablation), `head_priv` (C-POS), `head_img_shuf` (C-NEG), each with its config, PCA rank, ego scaling and target names | **repo** |
| `checkpoints/pca.npz` | the PCA bases (rank 16 and 64) fitted on TRAIN rows only — needed to reproduce any image arm | **repo** |
| `scripts/sc_situations.py` | ⭐ **the three detectors** — the single definition, imported everywhere | **repo** |
| `scripts/sc_dump_poses.py` | pod-side dump of every cached episode's `poses` | **repo** |
| `scripts/sc_build_labels.py` | the label bundle, the join proof, the universe, the DEV sweep | **repo** |
| `scripts/sc_cross.py` | cross traffic + per-camera need + the C-ALIGN clock map | **repo** |
| `scripts/sc_validate_labels.py` | the five label validations | **repo** |
| `scripts/sc_extract_trunk.py` | extracts the frozen v1 encoder+readout from the full v1 ckpt | **repo** |
| `scripts/sc_features.py` | frozen-encoder feature extraction on pod3 | **repo** |
| `scripts/sc_train.py` | the head, the CV, every arm, the ridge ladder | **repo** |
| `scripts/sc_eval.py` | the held-out evaluation (imports `h2c_stats` → `taniteval.ci`) | **repo** |
| `scripts/sc_report.py` | renders every table from the JSON — no number hand-typed | **repo** |
| `scripts/sc_recheck.py` | ⭐ **recomputes the headline numbers from `heldout_frames.npz` alone** — no GPU, no pod, no checkpoint — and diffs them against `sc_results.json` (non-zero exit on any disagreement) | **repo** |

**Not in the repo, and deliberately so:**

| | where | why |
|---|---|---|
| per-clip frozen features (`pod3:/workspace/sitclf/feats`, ~1.9 GB) | **pod3 only** | derived from a gated corpus; rebuilt in ~45 GPU-min by `sc_features.py` |
| `pod3:/workspace/sitclf/v1_trunk.pt` (348 MB) | pod3 + dev-box scratch | the deployed v1's encoder+readout, extractable from the v1 ckpt in one command; sha256 `10fc27f2…e329b5`, verified identical on both ends |
| the de-identified label bundle (`sc_labels.npz`, 26 MB) | pod3 + dev-box scratch | rebuilt by `sc_build_labels.py` in ~2 CPU-min |
| the clip-UUID map (`_LOCAL_ONLY_k2clip.json`) | **dev box only, outside the repo** | 🔒 gated corpus — UUIDs may never enter a derived artifact |

**Reproduction, end to end**

```
# pod (any host holding the parity episode cache)
python3 sc_dump_poses.py --out <poses>
# dev box
python  sc_extract_trunk.py <v1_ckpt.pt> <v1_trunk.pt>
python  sc_build_labels.py  <poses> <r0_selection.parquet> <bundle> train
python  sc_cross.py         <poses> <bundle> <cross>
python  sc_validate_labels.py <poses> <r0_selection.parquet> <bundle> <cross> label_validation.json
# pod3 (PYTHONPATH=/workspace/TanitAD/stack)
python3 sc_features.py --bundle <bundle> --out <feats> --trunk <v1_trunk.pt>
python3 sc_train.py    --bundle <bundle> --feats <feats> --out <run> --epochs 8 --folds 5
# dev box
python  sc_eval.py   --run <run> --bundle <bundle> --cross <cross> --out artifacts
python  sc_report.py artifacts artifacts/_tables.md SITUATION_CLASSIFIER.md
```

---

# 10. 🔴 What this unblocks, and what it escalates

**BOOST_PROGRAM §7.4 makes this a required field: a stream whose result nobody reads has the value
of a stream that produced nothing, at higher cost.**

### What this unblocks

1. **The H2 sensor-need stream, which is currently `UNDERPOWERED` at 35 positive clusters.** Its
   escalation was *"authorise a ~52 GB gated camera re-download so the label can run on 2,320 clips
   instead of 582."* ⭐ **That download is not needed for an ego-derived label.** `poses` is already
   in every cached episode, so this stream reached **2,376 clips with no download at all**, and the
   same trick applies to any label that is a function of the ego trajectory. The re-download is only
   needed for labels that require *agent* geometry.
2. **The three situations now have labels, code and a validation suite** — `sc_situations.py` is
   importable by any stream, and the anticipation-target construction (`anticipation_target`) is
   reusable for any "predict the onset" target in the program.
3. **The `corridor.JUNCTION_DEG = 10.0` stratum has a measured replacement.** A sibling stream
   MEASURED that proxy at only **37.2 % junction-scale** (INHERITED); §4 V4 here shows a
   tight-radius turn carries **2.4× [1.06, 7.93]** the perpendicular cross traffic of a
   matched-heading road curve, separated from 1.0. Any stream that strata-splits on "junction"
   should use `detect_turns` + `cross`, not a heading threshold.

### 🔴 Escalations — raised here, not buried in a README

1. **🔴 `taniteval.blind_baseline` cannot pass on a rare-positive binary target, and this is not
   specific to my study.** Its `CIRCULAR` branch fires when `blind_accuracy ≥ 1 − 0.02`. On a
   target with a positive rate below 2 %, the **majority-class predictor alone** clears that bar, so
   the verdict is forced regardless of whether the context carries anything. The firewall's own
   companion numbers prove the difference here — `blind_skill_over_majority = 0.0000` and
   `context_leaks = 0` — but a reader who quotes only the verdict would record a leak that the same
   record refutes. **Its MDE against the effect it exists to catch is stated in §5; the check needs
   a rare-positive mode (skill-over-majority or AP-based) before it is applied to any binary target
   below ~2 % positives.** This is the C13 pattern inverted: a guard that cannot *pass*.
2. **🔴 The roundabout situation is not decidable on PhysicalAI-AV, and the PI should be told
   plainly.** Two of his three situations are buildable here; the third is not. §3 and §4 give the
   numbers (27 held-out positive clusters against a 40-cluster bar; ~18 % of held-out detections
   turn the wrong way in a corpus with zero left-hand-traffic clips). A sibling stream reached the
   same conclusion from a different direction (INHERITED: *"0 of 2,482 clips reach a 270° sweep"*).
   ⚠️ **My own measurement disagrees with the sibling's stated corpus maximum** — I measure a
   **282.1°** maximum same-sign sweep on the 2,376-episode parity train cache against their 252°.
   The disagreement does not change either verdict (both are far below what a full roundabout
   traversal population would look like) but it is a live cross-stream contradiction and it is
   flagged rather than smoothed over.
3. **pod2 was double-assigned.** This stream was briefed onto pod2 while the `tblind-ladder` stream
   was already running there; the correction to pod3 arrived mid-run. **pod3 turned out not to hold
   the v1 checkpoint**, which cost a 348 MB dev-box→pod transfer at the MEASURED ~1 MB/s relay
   (0.99 MB/s over a 32 MiB probe). A one-line "which pods hold which checkpoints and caches"
   registry would have removed that entirely.
