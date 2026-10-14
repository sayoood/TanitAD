# Auxiliary drivable-area head, host S1: registration

> **IN FORCE on landing.** Written 2026-10-13, **before any S1 arm has started**: no `refe_aux_s1` launch-gate token exists, and no S1 output directory exists.
> After landing, nothing below changes without a dated amendment written before the read it affects.
>
> **What this file does:**
> - It is the S1 instance of `eval/PREREG_AUX_DRIVABLE_DRAFT.md` (the draft): it fills that draft's blanks for host S1 and approves the values in
>   `raw/2026-10-13-aux-head/PREREG_BLANKS_PROPOSED.md` (the proposal) §1–§4 and §6, unchanged.
> - It resolves the four ambiguities the readout harness (`eval/aux_s1_readout.py`, commit `20fa3548`) found (§3).
> - It states where each read is taken at S1, and the S1 → S2 promotion rule the draft does not have (§4).
> - It corrects one inherited count (§3, R4).
>
> Where this file and the draft disagree for S1, this file governs. For S2 and S3 the draft is unchanged.
>
> **Amendment A2 (2026-10-13, before any S1 arm)** at the end of this file adds a paired floor to P3 and fixes the ctx of record.
> **Amendment A3 (2026-10-13, before any S1 step)** places S1 on the A40 pod after RL-S1 (the PI's option A), resolves B1 and B2, and states that the train teacher-lane labels are a regenerated draw.
> **Amendment A4 (2026-10-14, before any S1 step)** runs S1's CPU-only stages beside RL-S1's training; the GPU stages still start only after RL-S1.
>
> Evidence classes: **MEASURED** (ours, artifact named) · **INHERITED** · **ANALYTIC** · **ESTIMATED** · **UNVERIFIED**.

## 1 · Approved without change (from the proposal)

| item | value | source |
|---|---|---|
| head learning rate | **2e-4** peak, follows the scorer's schedule ×2 (rule L) | proposal §1 |
| loss weight `w_aux` | **0.1** (rule W). The §2.3 balance is undefined with a frozen trunk; it is read at S2 | proposal §1 |
| head config | cell 2 m, 2 × `CrossBlock(256, 8)`, tap `visual`, feed **off** (H1) / **on** (H2) | proposal §1; `refe/aux_drivable_head.py` |
| arms and seeds | **N** (no head), **H1**, **H2** × seeds **0, 1, 2**; the priority order and the H1-s1/H1-s2 drop rule as written | proposal §2 |
| init | `model_sft4_B_u400.pt`, md5 `8e382eeb…`, 1,288,489,838 B (`eval/RESULT_SFT4.md`), then 400 updates of the SFT-4 recipe | proposal §2 |
| argv | proposal §3, character for character; the gate (`launch_gate.py` profile `refe_aux_s1`) binds it | proposal §3 |
| travel rules | no resume; no context cache; one data order per seed; exact zero-weight / detached tests in every head arm's preflight; tier **T0-like** | proposal §4 |
| shuffle control | **not an S1 arm** (Master Mind ruling, 2026-10-13); derangement seed **20261011** for any future amendment | proposal §6 |

**Deviation from the draft's S1 row, stated.** The draft listed H2, C-shuffle and C-seed. This registration runs:
- **N**, which is the draft's C-seed (the no-head replicate, and the seed floor);
- **H2**;
- **H1**, added as the CUDA identity of the scorer and a feed-free perception arm.

C-shuffle is dropped by ruling. So the draft's rule 4 does not apply at S1, and K5's shuffle limb cannot fire. ⇒ **Every S1 H2 gain is reported with "capacity not excluded"**, and C-shuffle is the first amendment if H2 is promoted (§4.3).

## 2 · The draft's blanks, filled for S1

| blank | value |
|---|---|
| label store (train) | already filled in the draft: `D:/Projects/TanitAD-artifacts/aux_drivable/label_store/`, **103,139 frames**, `store_sha256` `4a17db673b1548ba35724ab848a7dc551e4b0bc1e49c57ae93e6357454bf287a` |
| **navtest label store** | `D:/Projects/TanitAD-artifacts/aux_drivable/label_store_navtest/`, **12,146 frames**. `store_sha256` `985a3bc77f3ab7a96b0dceace48e73b80e5b69b70710958635060939bce0d7c1`; `manifest.json` sha256 `757788495c2a81bc56767deba02726643efa4c465754a5d1ca0f2da14cabac94`. MEASURED: gate **PASS** on all six criteria (AB, ORIGIN, CORNER, VERDICT, TARGET, LEAK); key, token and log overlap with the train store **0 / 0 / 0**; the y-flip mutation arm fails the gate as it must (`raw/2026-10-13-aux-navtest-store/s0_gate_navtest.json`) |
| head config, `w_aux`, balance ratio | §1. The balance ratio is **not measurable at S1**: the trunk is frozen, so no parameter receives both gradients |
| comparators and sha256 | the **comparator of record is the same-seed N run**. u400 (above) is the init and a reference row only; never an attribution |
| seeds | 0, 1, 2 for N, H1 and H2; derangement 20261011 (no S1 arm) |
| harness commit and seam flags | readout `eval/aux_s1_readout.py` @ `20fa3548` (tool `aux_s1_readout/1`); estimator: paired log-cluster bootstrap, 10,000 resamples, seed 20260927. **No navtest seam at S1** (§4.2), so no seam flags |
| gate PASS token ids | one per arm, profile `refe_aux_s1`, minted at launch. Each token id is written into `raw/<date>-aux-s1-runs/MANIFEST.json` **before that arm's first step** |

## 3 · Rulings on the four ambiguities (made before any S1 data)

**R1. P1's bands are LONGITUDINAL.**
- The bands are x of the 2 m cell centre in the rear-axle frame at t0, which is what the harness implements (`aux_s1_readout.band_cells`).
- Why: the label grid is x ∈ [−8, 72) m, y ∈ [−32, 32) m (`refe/aux_drivable_head.py:57`). The draft's last band ends at **72 m = x1**, the forward extent.
- A radial reading would mix up to 32 m of lateral extent into the 40–72 m band, and would leave radial distances up to 78.8 m in no band.
- The rear cells x ∈ [−8, 0) lie in no band. They enter only the trainer's overall IoU, which is reported, not decided on.

**R2. P3's decision read is the POOLED AUC.**
- The draft's words are "over the 64 proposals × the 1,058 C3 tokens".
- The mean within-set AUC is **reported, never decided on**.
- The two differ by a lot on this set. MEASURED, the u400 scorer on C3 (`raw/2026-10-13-aux-s1-readout/p3_reference.json`):
  - pooled **0.787 [0.731, 0.831]**;
  - within-set **0.656 [0.571, 0.739]**.
- So the pooled read credits ranking between tokens (some tokens' fans are almost all-pass or all-fail). The within-set read is the one that matches selection, which is why both are published.

**R3. The band the draft left uncommitted now has a consequence.**
- The band is: P3 pooled head AUC ≥ the scorer's AUC, but < max(0.90, scorer + 0.05). On C3 the threshold is **0.90**.
- In this band the read is **H2_PROVISIONAL**. H2 may be promoted to S2 only on stronger behavioural evidence: **3 of 3** seed pairs instead of 2 of 3 (§4.3).
- The harness's label `UNCOMMITTED_BAND` is read as `H2_PROVISIONAL` under this ruling. The decision stays on the point estimates, as the draft wrote it. The paired CI of head − scorer is published beside it.
- Reachability of the bar, MEASURED before any arm (same JSON): a head that outputs the **true** 2 m label scores **0.932 [0.918, 0.945]**, and the native 0.5 m label scores **0.944 [0.933, 0.955]**.
  - So 0.90 sits **0.032 below** what any head of this representation can reach.
  - The 2 m grid costs **0.012** of that ceiling (point estimates).
  - ⇒ Among the K2 next arms (draft, "Next arm"), the **1 m grid ranks last**: it cannot rescue a head that misses 0.90 by more than ~0.012. The other two arms' effects are unmeasured.

**R4. The u400 DAC-zero count is 416, not 415.**
- MEASURED on `raw/p2_r2_instrument/baseline/u400_navtest_full.csv` (sha256 `df005c4750a186de606d79a6f7cbda0fe725047a7b98bbf036b148fce62a1051`): 12,147 rows = 12,146 tokens + one `average` row; **416** tokens have DAC = 0, a share of **3.425 %**. 529 tokens have NC × DAC = 0, which matches E3.
- The draft's "415" was **back-derived from the rounded 3.42 %** (3.42 % × 12,146 = 415.4).
  - Root-cause class: *a count derived from a rounded percentage*, the same family as "a summary is not a path".
- Consequences:
  - E2's u400 reference row is 416 / 3.425 %.
  - The LARGE size label is **≤ 291 tokens (≤ 2.396 %)**, a ≥ 30 % cut of 416. These are labels, not bars.
- The draft and `eval/PREREG_AUX_FIELD_LOSS_DRAFT.md` carry a dated correction pointing here.

## 4 · Where each read is taken at S1, and what S1 can conclude

### 4.1 Perception (P1, P2, P3; K2)

| read | taken on | instrument | if the instrument is missing when the arms finish |
|---|---|---|---|
| **P1** (K2 on [0, 40) m, all frames, < 0.80) | **navtest**, the 1,058 C3 tokens, split u400 DAC = 0 (**416**, all of u400's DAC zeros) vs not (642), MEASURED against the u400 csv; labels from the navtest store | the head's outputs on navtest frames: `eval/aux_navtest_forward.py` (in build 2026-10-13). S1's trunk is frozen, so one u400 trunk forward serves every arm's head | K2 is evaluated on **P1-HO** with the same 0.80 bar, and the verdict says so |
| **P1-HO** | held-out navtrain frames: the 24 held-out logs, which are in the train store and log-disjoint from the head's training (harness guard `LeakRefused`) | the trainer's `head_pred_*.npz` dump (`--dump-scores`, patch in `20fa3548`) | always available; **reported** |
| **P2** | the same frames as P1 | harness | reported |
| **P3** | **C3 as written** (pooled, R2) | navtest forward + the C3 table | **P3 NOT READ** ⇒ H2 carries "perception gate unread" and **is not promoted** to S2 on S1 evidence |

If the navtest P1 and P1-HO disagree on K2, the **navtest P1 governs**, as the draft wrote it.

### 4.2 What S1 does NOT read

- **Navtest PDMS, DAC-zero share and zeroed share (E1, E2, E3) are not read for H2 at S1.** The planner seam does not apply H2's feed (proposal §5; one `model.py` + `planner.py` change, outside S1).
- So S1's H2 verdict is a **held-out selection verdict**. It is never written as the draft's ADOPT.
- The four families on navtest are not computed at S1, for the same reason. STRATEGIC is **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25).

### 4.3 The S1 → S2 promotion rule (committed now)

The harness classifies each seed pair Hx-sK − N-sK on held-out pick PDMS and pick DAC as one of: `adverse`, `not_separated`, `separated but within the seed floor`, or `separated and above the seed floor`. The floor is the largest |Δ| between two N seeds, and it is measured only with all three N runs.

**H2 is promoted to S2** iff all of:
1. the P3 decision on C3 is **H2_EVALUATED**, or **H2_PROVISIONAL** (R3);
2. PDMS reads `separated and above the seed floor` on **≥ 2 of 3** seed pairs (**3 of 3** under H2_PROVISIONAL);
3. **no** pair reads `adverse` on PDMS or on DAC;
4. none of K2, K3 or K6 fired on any H2 run.

Otherwise H2 is not promoted. The draft's next-arm table applies, and every S1 H2 read carries "capacity not excluded" (§1).

If H2 is promoted, the first amendment is **C-shuffle** (proposal §6, seed 20261011): the control that separates map information from capacity, run before any S2 spend.

**H1 is promoted to S2** (the representation lever, where the trunk trains) iff:
- K2 does not fire on H1's head (P1 [0, 40) ≥ 0.80, read per §4.1);
- K6 did not fire.

H1's scorer is bit-identical to N by construction (no path from the head to the scorer). Any H1-vs-N difference in selection is therefore a **defect**, investigated before any read.

**Fewer than three N runs** ⇒ every classification is capped at "separated, single seed, not yet a lever claim" (draft rule 5). No promotion follows from a capped read.

## 5 · Launch blockers (each named with what unblocks it)

| id | blocker | unblock |
|---|---|---|
| **B1** | the PI's decision on dev-box GPU time: 9 runs, **35–46 h ESTIMATED**, ceiling **107 h**, on the GPU shared with the PI's other session | the PI (asked 2026-10-13) |
| **B2** | S1's inputs on the dev box. SFT-4 ran on an **earlier pod** (`refe/sft4_pod.sh`: label root `refe_sft2/{train_teacher_lane, heldout_teacher_lane, heldout_lane, heldout_pdm, train_pdm}`, `refe_heldout/`, `navtrain_pixels`). The current pod holds only `refe_sft2/train_pdm` of the label root (MEASURED 2026-10-13 by `ls`) | the S1 input audit (`raw/2026-10-13-aux-s1-data-audit/`, running). An unresolved path is a launch blocker. If the inputs cannot reach the dev box, the draft's fallback applies: S1 → S2, perception from a head-only fit on cached C3 `visual_ctx` |
| **B3** | per-arm PASS tokens | minted at launch (§2) |
| — | (a read blocker, not a launch blocker) the navtest forward for P1 and P3 | in build; §4.1 says what happens without it |

## Amendment A2 (2026-10-13, before any S1 arm exists): a paired floor for P3, and the ctx of record

**What triggered it, stated plainly.**
- The navtest head-forward tool (`eval/aux_navtest_forward.py`) ran a **plumbing** read on 32 C3 tokens from one log, using a **never-trained** head (seed 0).
- That head scored a P3 pooled AUC of **0.8245**, against the u400 scorer's **0.8005** on the same 32 tokens (`raw/2026-10-13-aux-navtest-forward/RESULT.md` §5, `readout_plumbing_slice32.json`). One log means the interval is degenerate, so this is a signal, not an estimate.
- The mechanism: the head's learnable BEV queries and position features give it a **position-dependent SDF field before any training**. The minimum along a footprint then tracks how far and how wide a proposal reaches, and that already correlates with DAC = 0.
- ⇒ The bar max(0.90, scorer + 0.05) is partly a **position-prior** bar. A head could clear it without having learned the road.

**Why this is not a goalpost moved after the data.**
- No S1 arm has trained; the number seen is a property of the instrument on an untrained head.
- The amendment **adds** a condition and removes none, so every outcome it changes becomes stricter.

**A2.1 · The P3 floor (an added condition).**
- For each head arm, the floor is **that arm's own head at initialisation**: `AUX.AuxDrivable(AUX.AuxDrivableConfig(tap="visual", depth=2, feed=<the arm's>, seed=<the arm's seed>))`.
  - Its weights depend on the seed alone (`refe/aux_drivable_head.py:179-180`, `fork_rng` + `manual_seed(cfg.seed)`), so it is exactly the head the trainer starts from.
  - The feed module has its own fork (`:294-295`), so feed off and feed on share one initial head per seed.
- The floor is read on the same tokens, through the same ctx (A2.3) and the same statistic.
- **H2_EVALUATED and H2_PROVISIONAL now also require** the paired pooled-AUC difference *(trained head − its initial head)* to have a 95 % lower bound **> 0**. The estimator is the paired log-cluster bootstrap over the C3 logs, 10,000 resamples, seed 20260927.
- Otherwise the read is **P3_AT_FLOOR**: the head's DAC ranking is not shown to come from training. H2 is **not promoted**, and the draft's "P1/P3 fail (K2)" row of the next-arm table applies.
- The 0.90 and scorer + 0.05 conditions are unchanged. "Head AUC < scorer AUC ⇒ H2 killed" is unchanged.

**A2.2 · The P1 floor is reported.** The initial head's P1 IoU on the same frames is published beside every P1 read. K2's 0.80 is unchanged.

**A2.3 · The ctx of record is the planner's fp32 ctx.**
- This is the default of `eval/aux_navtest_forward.py`. Its identity is MEASURED: scorer logits bit-identical to the C3 table on 32 of 32 tokens, 14,336 values, max |d| 0.0. The camera-swap regression arm makes it fail.
- Why: it is the path a deployed feed would read, and the floor is read through the same ctx, so the comparison is paired.
- The trainer's bf16-autocast ctx differs by 1.05–1.46 % relative L2 (MEASURED, 32 tokens, `smoke.json`). It is **supplementary** if run, and never decides.

**A2.4 · The floor is measured before the first S1 step.**
- Seeds 0, 1 and 2, on all 1,058 C3 tokens, published as an instrument property.
- It changes no number in this file. Its only role is to show, before any arm trains, how much of the 0.90 bar a position prior already covers.

**A2.4 result (MEASURED 2026-10-13, still before any S1 arm; `raw/2026-10-13-aux-p3-floor/full_1058/FLOOR_1058.md`).**
- On all 1,058 C3 tokens over 123 logs (the ctx bit-identical to the C3 table on all 473,984 scorer values), the initial heads read a P3 pooled AUC of:
  - seed 0: **0.659** [0.623, 0.694];
  - seed 1: **0.563** [0.516, 0.605];
  - seed 2: **0.698** [0.654, 0.737].
  The scorer reads **0.787** [0.731, 0.831], and every initial head is separated below it.
- Within-set the initial heads read 0.49–0.57, i.e. at chance. Their P1 IoU on [0, 40) m is 0.13–0.27.
- The 32-token 0.8245 that triggered A2 was a one-log artefact. The position prior is real but seed-dependent, which is why A2 pairs each arm with its own initial head.
- **No number in this file changes.**

## Amendment A3 (2026-10-13 21:40 Berlin, before any S1 step): where S1 runs, and the labels it trains on

**The ruling.** The PI chose **option A** on 2026-10-13 (*"go with A"*), between:
- **A:** the A40 pod, after RL-S1 and before P6;
- **B:** the pod, after P6;
- **C:** the dev box.

This resolves **B1** and **B2** of §5. No S1 step, token or label existed when this was written. The pod's `/workspace/data/s1aux` did not exist, and RL-S1 was still labelling (MEASURED 2026-10-13 19:38Z).

**A3.1 · Host and order.**
- **Host:** one A40 and 11 vCPU, running the chain `refe/s1aux_pod.sh` from `/workspace/refe-plan-s1aux`. This is a directory of its own, so the RL-S1 and P6 trees are never re-measured or touched.
- **Tree:** shipped from git at the commit named in its `SHIP_MANIFEST.json`. The chain refuses a drift, and every arm's gate re-measures the tree.
- **Order:** the chain starts only when all four hold:
  - RL-S1 has written `ZZRLS1_DONE`;
  - the P2 and RL-S1 locks are free;
  - the GPU has been idle on 3 consecutive polls;
  - the GPU is re-checked before every arm.
- **If RL-S1 ends without `ZZRLS1_DONE`:** the chain refuses (S1-D4).
- **P6** does not know about S1 (S1-D5). It is started **by hand after `ZZS1AUX_DONE`**, and still needs P2's R2 GO plus the PI's confirmation.
- **The §5 B1 estimate (dev box, 35–46 h) no longer applies.** ESTIMATED A40 wall time is about **12–16 h** (`eval/S1_POD_RUNBOOK.md` §5):

| step | ESTIMATED time |
|---|---|
| `heldout_pdm` | 3–20 min |
| `train_teacher_lane` at 2,500 sets | 0.7–0.9 h |
| nine arms (1.3–1.6 h each) | 11.7–14.4 h |

  P6 starts later by the same amount.
- **HF storage:** private storage is **213.95 GB** before S1 (MEASURED 2026-10-13: Σ `usedStorage` over 56 repos). S1's pushes add about 15 GB (ESTIMATED: nine arms of ~1.3–1.6 GB, plus labels and logs), well inside the 1 TB private ceiling.

**A3.2 · The inputs, and which are regenerated.** The audit (`raw/2026-10-13-aux-s1-data-audit/`) left two inputs unresolved anywhere. On the pod they are produced by the chain, **before any arm**:

| input | source at S1 | same as SFT-4's? |
|---|---|---|
| `heldout_pdm` | recomputed: `onpolicy_relabel_pdm.py --served-only` (P2's recipe) over the 3,137 held-out sets; accepted only at `pdm_label_complete.py --expect 3137` and ≥ 3,137 rows | same recipe, same sets; recomputed |
| `train_teacher_lane` | regenerated: `select_bank_subset.py --seed 20261004 --n 2500` over the HF banks' `onpolicy_queue.tar.zst` (1,817,869,216 B, md5 from the repo's `MD5SUMS.txt`), then `onpolicy_relabel_teacher_lane.py --repair-last-heading`; refused below **2,109** rows | **NO: a different draw of the same recipe** (below) |
| held-out lane + teacher-lane labels | SFT-4's own, restored from `raw/2026-10-04-lane-discipline/*.xz`; refused unless 3,137 rows each | yes |
| `train_pdm`, `navtrain_pixels`, targets, on-policy bank | P2's, on the pod (the bank after the 24-log carve) | as P2 |
| aux label store (train) | shipped; verified by the store's own sha check, `store_sha256` `4a17db67…f287a` | — |

**Why the teacher-lane draw differs, and what that costs.**
- The queue SFT-4 drew its subset from was **never banked** (`eval/PREREG_SFT4R.md`, NAMED GAP).
- u400 trained on the **2,109** rows written by its launch.
- S1 uses the first **2,500** sets of the seed-20261004 order, i.e. a prefix of the 15,000-set subset the recipe registers.
- **2,500 rather than 15,000:**
  - it is the size nearest u400's own 2,109 that clears the floor;
  - its cost is 0.7–0.9 h, against 4.3–5.7 h for 15,000.
  - A later amendment can extend to 15,000 without changing these 2,500 (same seed, prefix order).

**Consequences, stated before the data:**
1. **Attribution is unaffected.** Every S1 arm (N, H1, H2, all seeds) trains on the **identical** regenerated label set. The comparator of record is the same-seed N (§2).
2. **The labels are banked before the first arm** (`labels/` in the private `Sayood/tanitad-refe-s1aux`, proven by the remote listing and content, stage `bank_labels`). The set every arm saw is therefore recoverable.
3. **S1's N arms are not a replication of SFT-4's 400 updates.** An N-vs-u400 difference is not attributable. u400 stays a reference row only, as §2 already says.

**A3.3 · The H1-s1 / H1-s2 drop rule is not exercised.**
- On the pod, all **nine** registered arms run unattended in the registered order.
- Exercising the drop rule would need a model-turn gate between arm 7 and arm 8, with the A40 idle while it waits. The saving would be about 2.6–3.2 A40-h.
- Nine runs are the cost the PI accepted with option A.
- Running a registered arm moves no criterion. Both H1 replicates are reported.

**A3.4 · Where the token ids are first written.**
- §2's rule is "written **before that arm's first step**". On the pod, that write is the chain's `arms_manifest.jsonl` row, appended before `trainstart`.
- The row is pushed to HF with the arm.
- `raw/<date>-aux-s1-runs/MANIFEST.json` is transcribed from it when the arms are banked into the repo. The order of events is unchanged; only the first file holding the id differs.

**No number, criterion or read in this file changes.** B3 is unchanged: tokens are minted per arm at launch.

## Amendment A4 (2026-10-14 ~08:30 Berlin, before any S1 step): the CPU stages run beside RL-S1's training

**The ruling.** PI, 2026-10-14, verbatim: *"The pod is idling, this is not acceptable"*.

**What was idle (MEASURED on the pod, RL-S1's `chain.log`):**
- the A40 had no work for **~11.5 h** (2026-10-13 14:59Z → 2026-10-14 02:33Z), while RL-S1 labelled on the CPU before its first GPU stage;
- it was idle for **~28 min between consecutive arms** (C0 trainer exit 05:55:55Z → R0 trainer start 06:24:19Z).

Under A3, S1 would add the same pattern: about 1–1.5 h of CPU-only stages (u400 fetch, held-out data, `heldout_pdm`, `train_teacher_lane`, label banking) with the GPU idle after RL-S1 ends.

**The change (mechanics only).**
- `refe/s1aux_pod.sh` no longer waits at precheck. Precheck still **refuses** a refused upstream (RL-S1's lock free with no `ZZRLS1_DONE`).
- The stages `env`, `fetch`, `data`, `relabel` and `bank_labels` then run **beside RL-S1's training**:
  - the labellers at `nice -n 19`;
  - label workers capped at the cgroup cores minus 4 (7 of 11), leaving the RL-S1 trainer and its two loaders their cores. MEASURED: they use ~2.3 cores while an arm trains. Memory has room: ~4.5 GB RSS against the 71 GB cgroup limit.
- A new stage, `gpu_wait`, sits **before the first GPU use** (the first arm's preflight). It waits until:
  - the P2 and RL-S1 locks are free;
  - RL-S1 has written `ZZRLS1_DONE`;
  - the GPU has been idle on 3 consecutive polls.

  It then runs the disk write test. The per-arm GPU re-check is unchanged. **S1 still never puts a GPU job beside a training job.**

**Why this changes no read and no criterion.**
- The labels are a function of their inputs and seeds (seed 20261004, 2,500 sets, the same queue and the same recipe), not of when they are computed.
- `bank_labels` still banks them, proven by content, before the first arm.
- The arms, argv, tokens, order and every criterion of this file are untouched.

**The guard on RL-S1.**
- RL-S1's update rate is read from its own egress ticks before and during the overlap: C0 took ~58 min per 250 updates.
- If the rate drops by more than 5 % during the overlap, the S1 labellers are stopped by explicit PID, and the drop is reported.
- Rationale: the binding rule "never add GPU/RAM load to a training pod" exists to protect the training run. This overlap adds no GPU load and ~2–3 GB of RAM. Its CPU load runs at the lowest priority, and it is measured against the run it could harm.
