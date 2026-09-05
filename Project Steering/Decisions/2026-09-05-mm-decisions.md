# Master Mind decisions, 2026-09-05

## M1. The pod's stale `train_v6_staged.py` ships at the NEXT v7 launch preflight — never mid-run

From the pod-currency audit (`…/2026-09-04-pod-currency-audit/POD_CURRENCY_AUDIT.md` §2 rank 2,
commit `94ceaf4`): `tanitad-refcv3:/workspace/TanitAD/stack/scripts/train_v6_staged.py` is
**113 KB (~19 %) behind HEAD** and lacks the `--bptt-truncate` wiring. A v7 launch from that box
would run the O5 rollout with the unbounded BPTT chain — the MEASURED `PREREG_MM_E19` failure
(gnorm 2.1e9, run killed at step 9,000) — and would fail *silently*, because the box is
stale-but-internally-consistent. The escalation was correct: this file must not be shipped while
`refcv4b` trains (a supervisor relaunch imports whatever is on disk), and it must not be shipped
in isolation (its compatibility with a future v7 config cannot be verified from here).

⇒ **Decision:** the v7 launch runbook gains a mandatory step: run
`stack/scripts/pod_currency_audit.py` against the target pod, ship `train_v6_staged.py` and
`tanitad/models/metric_dynamics.py` as part of that preflight, verify by md5 at both ends and by a
real `import` on the pod, and grep-verify `--bptt-truncate` is present before any launch. Owner:
whoever launches the next v7 arm. Not before refcv4b finishes.

## M2. `metric_dynamics.py` on the pod stays as it is until M1

Imported but never called by the live trainer; not a pure addition (38 pod-only lines);
default-equivalence only INHERITED from a docstring and a test not run against this pod. Zero
benefit to the live run, non-zero relaunch risk. Ships with M1.

## M3. The ego-dropout burden is NOT a live-run change

`D-REFCV4B-EGODROP2` (commit `ef0d5ec`): real, longitudinal, shrinking on its own. The fix
(`H-EGODROP-PRED`, roll the withheld bank at the model's own predicted speed) is pre-registered
for refcv5 on the v7-tiny ladder with a shuffle-the-prediction control. The live run is read
again on its final checkpoint.

## M4. Never run two `kb_add` writers concurrently

`library.json` was found at 0 bytes after the API-limit deaths — a rewrite interrupted mid-flight.
Restored from HEAD and re-verified (`7d1f48f`). Every literature brief now carries the rule, and a
0-byte `library.json` is to be read as the signature of an interrupted writer, never as an empty
library.

## M5. The three unapplied cost-repair rows are SUPERSEDED, not applied

The units stream (commits `8cb69ac`…`1291bf5`) correctly refused to apply
`…/2026-09-03-cost-repair/PROPOSED_REGISTER_ROWS.md` §2–4 (`H-COST-WEIGHTS-1`,
`D-COST-ARGMIN-MOVES`, `D-COST-SURFACE-REPRODUCED`) and its §5 prediction-retraction: the
register already frames `D-COST-CHORD` as FAILING its own pre-registered criterion (BACKLOG R41,
R55/R56), and those three rows are intermediate findings of the chord experiment. The live
hypothesis is no longer the chord (monotone-equivalent, cannot re-rank) but the centred cosine
(`ccos`, commit `cee5d99`), whose panel — exclusion fraction, L/R share, distinct plans, κ ≡ 0,
weight-neutrality factor, under `cos` / `chord` / `ccos` with constant-only and
deliberate-regression controls — is running now and measures exactly what those rows claimed.

⇒ **Decision:** none of the three is registered as written. `H-COST-WEIGHTS-1` is replaced by the
`ccos` weight-compensated arm; `D-COST-ARGMIN-MOVES` and `D-COST-SURFACE-REPRODUCED` are re-read
against the `ccos` panel when it lands and registered then, in whichever form survives it. The
§5 retraction stays in the package file as the author's own record; nothing is lost.

## M6. `H-EGO-LIT-4` gets an owner: the 5-arm withheld-bank panel runs on the v7-tiny rig now

The literature stream escalated the discriminating experiment for `H-EGODROP-PRED` with no
owner. It is the cheap pre-retrain validation the PI asked for, it needs no pod GPU, and both
outcomes are already committed. Launched 2026-09-05 as its own stream: implement the stamped
`--withheld-bank {fixed,pred,random,none}` trainer flag, run A0 fixed / A1 predicted / A2
random-marginal control / A3 dropout 0.25 / A4 speed-blind vocabulary, score on the `H-ECHO-8`
separation instrument and four families on kept AND withheld rows — never on ADE.

## M7. ⭐ PI RULING — the environment extension ships in TWO releases: v5a pure vision, then v5b LiDAR

**PI, verbatim (2026-09-05):** *"i prefer to do the environment extensions in two versions/steps, let
start by pure vision and then add lidar. So check, what we can do maximally with vision, bev, und was
else?"*

⇒ `REFCV5_DESIGN_PLAN.md` §3/§7 were written on a single-track assumption and must be restructured
into **v5a (camera-only, maximised)** and **v5b (LiDAR BEV)**. The §7 ladder its author is writing now
is still valid as a set of work packages; only the release boundary changes. A dedicated study is
running: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-vision-only-maximum/`.

**The inventory that makes v5a large, MEASURED** (`…/Data Engineering/Implementation/incoming/
2026-07-26-physicalai-feature-probe/PHYSICALAI_FEATURE_PROBE.md`, the dataset's own 36-row
`features.csv`: 7 camera + 6 calibration + 3 label + 1 lidar + 19 radar):

* **We read ONE of SEVEN cameras.** `camera_front_wide_120fov` only (`physicalai.py:232`, pinned at
  2 / 5 / 6 features by layer in `test_physicalai_feature_readset.py`). Unread: `front_tele_30fov`,
  `cross_left_120fov`, `cross_right_120fov`, `rear_left_70fov`, `rear_right_70fov`,
  `rear_tele_30fov` — each 1.6–2.5 GB/chunk, 5.1–7.9 TB total. ⭐ **This is the single biggest
  vision-only lever and it needs no new sensor modality.**
* **`camera_intrinsics` already covers all 7 cameras at 100 %** and we already read it; the rig
  transform for every sensor is in `sensor_extrinsics`, also already read. Adding a camera is a
  DECLARATION change in the episode build, not a new calibration problem.
* **`camera_intrinsics.offline` carries `ego_mask_image_png`** (97.44 %, 0.2 MB/chunk, unread) — a
  free ego-occlusion mask.
* ⚠️ **A third option the two-release framing does not cover: RADAR.** 19 radar features in three
  mutually-exclusive rig configurations on **160,761 clips (52.49 %)**; the probe's own words:
  *"the whole radar suite is cheaper than one camera."* Radar gives measured RANGE-RATE directly —
  the LONGITUDINAL family is where 88.7 % of our oracle gap lives. It is neither vision nor LiDAR,
  and it is the cheapest sensor in the dataset. **Flagged to the PI as a possible v5a-plus or v5b
  alternative; no decision taken.**
* LiDAR for comparison: **32,340 MB/chunk, 101.7 TB**, 97.44 % coverage (this census; §3 of the plan
  read 99.6 TB from the HF blob listing — same order, two probes).

**What v5a can recover of DiffusionDrive's three grounded attentions** (to be settled by the study,
stated here as the question): waypoint-indexed sampling in PERSPECTIVE view needs no BEV at all
(`H-DDA-1`, already pre-registered); agent cross-attention is reachable from cameras because
`obstacle.offline` supplies 3D cuboids as **LABELS** on 97.44 % of the corpus and the binding rule is
*labels may use privileged signals, inference is vision-only* — so a monocular 3D detector trained on
them yields agent tokens at inference without LiDAR; a metric BEV is the one that genuinely wants
either surround cameras (a lift) or v5b.

⇒ **Decision:** v5a is maximised and shipped first. v5b is judged on what gap remains AFTER v5a, not
on its own merits.

## M8. `E-DDA-1` runs the PLAN's four-arm form with its three controls — not the study's two-arm form

The reconciliation surfaced a genuine conflict of FACT (not naming) and correctly refused to
adjudicate it: `E-DDA-1` (waypoint-indexed sampling in perspective view) is **4 arms ≈ 2 h with
three named controls** in `REFCV5_DESIGN_PLAN.md` §7, and **2 arms ≈ 1.0 h with none listed** in
`…/2026-09-05-vision-only-maximum/RESULT.md`.

⇒ **Decision: the four-arm form.** The hour saved buys nothing; the control it drops is the
**projection control** — the arm that proves the instrument can SEE a wrong projection. Without
it a null result is uninterpretable (the sampler may be reading the wrong pixels and we could not
tell), and a positive result is unfalsifiable. This programme has paid for that lesson twice this
week: the anti-echo gate is only meaningful because a deliberately image-blind arm FAILS it
(`H-ECHO-4`), and the refcv3 route metric scored 1.0000 while measuring nothing because no
intervention control existed. ⛔ A control that must read a known value is not an optional cost line.

⚠️ Corollary recorded so the cheaper number does not leak into a plan: the reconciled v5a total is
**≈ 29–38 rig-GPU-h (the union of both ladders)**, NOT the study's ≈ 13.4 h. Quote the union.

## M9. Rung A1 is launched now, ahead of refcv4b's finish

`ha0_ext` is absent from the REF-C harness (0 occurrences in `refcv3_arm.py` against 12 in
`refav1_arm.py`, same-breath control 28 `add_argument` calls — the file reads fine), so half the
refcv5 acceptance bar is unreadable; and **8 of the 12** registered hierarchy ablations have no CLI
flag. The eighth missing one is the **frame-blind deliberate regression** (`--ablate-frames`), which
the prereg's own escalation omits and on which the panel's validity depends — a gate never shown to
FAIL an image-blind arm certifies nothing.

Both are zero-GPU harness work and both block the post-training hierarchy panel, which becomes
runnable when refcv4b finishes (~2026-09-06 08:00 UTC). Launched 2026-09-05 as its own stream rather
than scheduled, because the window is ~20 h and the work is free.

## M10. Open, carried, NOT decided

* `D-V5A-CAM3` is a **retraction-log candidate** — a banked DataFlyWheel camera figure is
  ×1.39–1.58 high. Nobody has written it to `RETRACTION_LOG.md`; it needs the same treatment as
  #23.
* The **B1 chunk spread is unbanked** and BOTH ladders rest on it — a one-query readout for the
  DataFlyWheel. If the build pulls whole chunks rather than per-clip ranges, the 429 GB camera
  figure silently becomes terabytes.
* The study's manifest claims **11** `D-V5A-*` rows; the register holds **10**. Left as found —
  ⛔ adding a row to make a manifest agree is how a register stops being evidence.
