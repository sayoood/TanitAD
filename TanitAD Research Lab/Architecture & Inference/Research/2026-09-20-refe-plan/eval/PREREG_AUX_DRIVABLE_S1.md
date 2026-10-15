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

## Amendment A5 (2026-10-15 ~03:40 Berlin, before any S1 read): execution order after the H-arm refusal

**What happened.**
- N-s0 trained and was banked (`ZZS1AUX_ARM_DONE N-s0`, 2026-10-14 21:01Z).
- The chain then REFUSED at the preflight of the next two arms:
  - H2-s0 at 21:07Z;
  - H1-s0 at 23:31Z after a restart.
- In both cases G12 failed under CUDA bf16 while passing on the dev box (CPU, fp32). A fix is in build; it touches only the preflight check.

**The change.**
- To keep the A40 busy, the chain was restarted on 2026-10-15 00:51Z with `S1AUX_ARMS="N-s1 N-s2"`. That is a subsequence of the registered order, and the chain
  accepts it by design (`s1aux_pod.sh:13`).
- The H arms (H2-s0, H1-s0, H2-s1, H2-s2, then H1-s1 / H1-s2 under A3.3) run afterwards, in their registered relative order, once the fix is landed,
  shipped and has passed the gate ON THE POD.

**Why this changes no read and no criterion.**
- Every arm is an independent fine-tune from the same u400 checkpoint, with its own seed and its own data order. Nothing passes between arms; there is
  no resume (section 2's travel rules).
- The order mattered only as a PRIORITY: a chain killed early should still yield the most important arms. With the H arms blocked, it cannot be honoured
  for them.
- The arms, argv, tokens, criteria and the A3.3 drop rule are untouched. The first S1 read happens after all registered arms exist.

**Disclosure.** Line 234's "all nine registered arms run unattended in the registered order" no longer holds as written: the execution order is
N-s0, N-s1, N-s2, then the H arms. Each arm's `chain.log` timestamps record it.

## Amendment A6 (2026-10-15, before any H arm trains): two instrument fixes in gate G12, found on the A40

**What this is.** Two bugs in the G12 preflight probe of `scorer_finetune.py`, both invisible to the CPU suite because it switches off the bf16 autocast and the GPU kernels the pod runs. It is an instrument fix. It changes **no arm, argv, bar, outcome, loss, weight, default, or training path** (N, H1 and H2 train exactly as before; the edited code runs only under `--preflight`). The launch gate's `consume_scorer_preflight` and `gate_checks.py` are untouched and still require every field they required before.

**The failures, quoted (MEASURED on the A40 pod).**
- **H2-s0, 2026-10-14 21:07Z**, `scorer_finetune.py --preflight`, gate `G12_aux_drv_live_and_identity`: every sub-check true except **`feed_probe_reads_difference: false`** (`feed_zero_init`, `feed_step0_equal_base`, `score_to_head_grads_all_none` held; `feed_proj_grad_abs` 1.22; Adam state 43/43). Arm N-s0 (no head) trained fine.
- **H1-s0, 2026-10-14 23:31Z**, same gate: **`reference_replay_identical: false`** and **`scorer_params_bit_identical_2steps: false`**; `identity_probe_reads_one_ulp`, `score_to_head_grads_all_none`, `bce_near_ln2` true; Adam state 43/43.

**Root causes.** Measured on this machine's GPU (RTX 4060, torch 2.11.0+cu128, CUDA bf16 autocast), with the **real trained scorer** (`D:/Projects/TanitAD-artifacts/refe-final-2026-10-01/model_final.pt`, 6 outputs) and the real head, in `PK/raw/2026-10-14-s1-g12-bf16/` (`probe_b_rootcause.json`, `probe_d_*.json`, `probe_d_summary.json`).

*H2 — cause (c), the autocast weight-cast cache; (a) and (b) are refuted.*
- Inside one `torch.autocast` region the first bf16 use of a requires-grad fp32 leaf caches its bf16 cast until the outermost region exits, whatever the grad mode. The old probe used `feed.proj.weight` (all zeros) in `feed_step0_equal_base`, then `copy_`-ed 1e-2 into it **in the same region**, then scored again. The second score re-read the cached zeros and equalled the base scorer exactly. On CPU autocast is disabled (no cast, no cache), so the dev-box test passed.
- Feed output `|max|` through the real `DrivableFeed`: **0.0** at w = 0; **0.0** after `copy_` of 1e-2 in the same region (stale); **0.0564** after `torch.clear_autocast_cache()`. The old probe code on CUDA bf16 read `feed_probe_reads_difference: false`, `max |diff|` 0.0, at context scales 0.05, 1 and 5 — the pod's signature (`feed_zero_init`, `feed_step0_equal_base` true).
- **(a) rounding is refuted.** The 1e-2 contribution has median |·| 0.042 (max 0.056) against the base first-layer pre-activation median |·| 2.45 (p95 11.6), whose bf16 half-ulp at the median is 0.0078; 87.9 % of elements exceed the half-ulp. Read in a fresh region, **691 of 768** bf16 score outputs change (max |d| 0.28). The smallest constant that still changes the output is about 1e-5 (248 of 768 elements).
- **(b) degenerate step-0 scalars is refuted.** Head at step 0, bf16: SDF range [-4.25, 4.94]. Over 128 (batch, proposal) rows: min-corner SDF in [-2.94, 1.55], centre-mean SDF in [2.24, 4.19], share below 0 in [0, 0.21]; all **128 of 128 rows distinct**.

*H1 — `warn_only=True` leaves the memory-efficient SDPA backward non-deterministic; bf16 and cuBLAS are refuted.*
- The three arms ran under `use_deterministic_algorithms(True, warn_only=True)`. torch then only warns ("Memory Efficient attention defaults to a non-deterministic algorithm. To explicitly enable determinism call `torch.use_deterministic_algorithms(True, warn_only=False)`"). The scorer's and the head's `nn.MultiheadAttention` at 7,680 visual tokens (B = 2, K = 64) take that kernel.
- Measured two identical head-less runs, X1 and X2, then the head arm Y, 2 steps, AdamW 3e-5, same process:

  | variant | X1 == X2 | X1 == Y | params differing X1 vs X2 (of 78) | max \|d\| |
  |---|---|---|---|---|
  | as the trainer ran it (`warn_only=True`, `setdefault` of the cuBLAS config after CUDA init) | no | no | 78 | 1.19e-4 |
  | `warn_only=True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` set BEFORE the CUDA context | no | no | 78 | 1.17e-4 |
  | no determinism flag at all | no | no | 78 | 1.19e-4 |
  | `warn_only=True`, fp32 (no autocast) | no | no | 65 | 1.16e-6 |
  | `warn_only=True`, SDPA forced to the math kernel | yes | yes | 0 | 0 |
  | **`warn_only=False`**, cuBLAS config unset / set late / set early | **yes** | **yes** | **0** | **0** |

- So: the op is the memory-efficient SDPA backward; **bf16 is not the cause** (fp32 differs too); the late `CUBLAS_WORKSPACE_CONFIG` is a no-op but **not the cause** (setting it early does not help under `warn_only=True`, and strict mode is bit-identical without it). `warn_only=False` makes that kernel deterministic and **does not raise** here.

**The fixes (G12 probe only; `refe/scorer_finetune.py`).**
1. *Feed branch (H2).* The zero-weight reads keep their autocast region. The weight is then set to 1e-2 **outside** any autocast region, `torch.clear_autocast_cache()` is called (the probe may itself run inside an outer region), and each read (the head's own step-0 field, and a **known** field: 3 m plus a ramp along y, clamped to the SDF bound) gets its own region. The weight is restored in a `finally` and the cache cleared again. `feed_zero_init`, `feed_step0_equal_base`, `feed_proj_grad_abs`, the gradient checks, the restore, and every other field are unchanged. New raw diagnostics (not re-judged by the gate): `feed_probe_known_field_reads_difference`, `feed_probe_max_abs_diff`, `feed_probe_elems_differ`, `feed_probe_elems`, `feed_probe_scalar_std_over_proposals` — they tell a failing run apart as rounding, a degenerate field, or a cut path.
2. *No-feed branch (H1).* `torch.use_deterministic_algorithms(True, warn_only=True)` becomes `(True, warn_only=False)`, and the previous warn-only state is restored. **The identity check is still a bit comparison; no tolerance was introduced.** A head that does reach the scorer still reads different (below).

**What this does NOT show (UNVERIFIED).**
- It has **not run on the A40**. The local GPU is an RTX 4060 (sm_89) on torch 2.11.0+cu128; the pod's torch is not the same build. Strict mode raises on an op with no deterministic implementation, and torch's cuBLAS config check can raise when `CUBLAS_WORKSPACE_CONFIG` is unset: either is a **loud** preflight failure, never a silent pass. The first `gate_H1-s0` / `gate_H2-s0` on the pod is the confirmation; if the cuBLAS check raises, export `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the launch environment (`s1aux_pod.sh` does not set it today).
- The full `scorer_finetune.py --preflight` could not run here with the real data banks; the CUDA tests run the real `main()` `--preflight` against a stub trunk (and, for H1, a **real** `CrossBlock` at 7,680 tokens).

**Proof on CUDA bf16 (`refe/test_aux_g12_bf16.py`, source unrewritten, this GPU).**
- H2: the fixed probe passes (`max |d|` and the differing-element count read, the known field reads, the weight is set to 1e-2 during the probe and restored to exactly 0.0, the probe ran under CUDA autocast and returned bf16); the old block put back reads `feed_probe_reads_difference: false` on CUDA and `true` on CPU (the blind spot, reproduced); `aux.score` ignoring the feed reads RED and the launch-gate consumer refuses it.
- H1: with a real `CrossBlock` at 7,680 tokens, X1 == X2 and X1 == Y bit for bit and the one-ulp control reads; `warn_only=True` put back breaks the identity probe on CUDA (the replay is stochastic, so the check takes the first of at most 3 runs that reads RED) and stays green on CPU; a head whose loss reaches the scorer's parameters reads `scorer_params_bit_identical_2steps: false` while the replay stays identical, and the consumer refuses it.
- Counts: `test_aux_g12_bf16.py` `ZZTEST_PASS 7` (30 checks on the unmutated source + 6 RED mutations); `test_aux_s1_wiring.py` `ZZTEST_PASS 20`; `test_rl_s1b.py` `ZZTEST_PASS 18` (the EPO / loss pins hold); `test_aux_drivable_head.py` `ZZTEST_PASS 68`; `test_s1aux_chain.py` `ZZTEST_PASS 361`; `test_launch_gate.py --static` `ZZLG_STATIC_PASS 637`; `test_rl_epo.py` `ZZTEST_PASS 42`.

**Consequences.** The tree changes (`scorer_finetune.py`), so the S1 manifest and the launch-gate tokens (bound to commit, argv and data hashes) are re-derived before the next S1 preflight; N-s0 already trained under the earlier file, and the edited code is outside every code path it ran after `--preflight`. No registered claim, bar or promotion rule moves.

*Numbering.* Drafted as "A5" (the probe artifacts under `raw/2026-10-14-s1-g12-bf16/` still say so). It became **A6** on landing, because A5 (the
execution order) had landed first, in `6c3dbe4`.

**A6 addendum (Master Mind, 2026-10-15 ~04:40 Berlin, before any H arm trains): the pod's torch build needs `CUBLAS_WORKSPACE_CONFIG` at process start.**

- **The pod runs torch 2.7.1+cu128 (MEASURED).** The local `driverl-eval` venv has the same build, so `test_aux_g12_bf16.py` was run there on the local GPU.
- **As written, it FAILS.** In strict mode, torch 2.7.1 raises its cuBLAS determinism error ("...uses CuBLAS and you have CUDA >= 10.2..."). The probe's
  in-process `setdefault` comes too late for that build's check, and every H1 identity check fails.
- **With `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment at process start, it PASSES:** `ZZTEST_PASS 7` (MEASURED).
- **So `s1aux_pod.sh` exports the variable on the preflight command line ONLY.** The trainers' environment is unchanged, and stays identical to the N
  arms that already trained.
- `test_s1aux_chain.py` pins it (`PREFLIGHT_NO_CUBLAS_ENV`), and a mutation removing it goes RED.
- The A40 run itself is still to come. It happens at the first H arm's preflight, after a pod pre-run of the suite.

## Amendment A7 (2026-10-15 ~07:00Z, before any H-arm read): rulings on four wordings the read driver found ambiguous

**What triggered it.**
- The end-to-end read driver (`eval/s1_read.py`, package `raw/2026-10-15-s1-read-driver/`) was built on 2026-10-15. It found four places where this
  file's words admit two readings: AMB-1 to AMB-4 in its `NOTES.md` §6.
  - The driver implements one reading and recomputes the outcome under the other. Without a ruling, nothing says which reading governs.
- **When.** At this writing H2-s0 is still training (`ZZS1AUX_ARM_TRAIN_START H2-s0` 05:24:53Z) and no other H arm has started.
  - The driver has been run only on the three N arms, where it reads `INCOMPLETE` by design.
  - **No H-arm number has been produced or seen by anyone.**

**Why this is not a goalpost moved after the data.** Each ruling below picks the reading that makes **promotion harder or a kill easier**, or it
names one number where the text named none. No ruling relaxes a bar.
- The rejected reading of AMB-1 to AMB-3 is still computed and published beside the governing one, as **supplementary**. It never decides
  (the A2.3 pattern).
- AMB-4's added limb only adds a way for K3 to fire.

**A7.1 · AMB-1, the H2 P3 condition with three H2 seeds (§4.3 cond 1).**
- **Governing:** cond 1 holds only if **every one of the three H2 seeds** reads `H2_EVALUATED` or `H2_PROVISIONAL` (A2.1's floor included).
- **Regime:** `H2_PROVISIONAL` governs cond 2 (3 of 3 pairs) if **any** seed is PROVISIONAL.
- **Why:** cond 4 already reads "on any H2 run", and promotion is the claim, so the condition applies to every run.
- **Supplementary:** 2 of 3 seeds.

**A7.2 · AMB-2, H1's K2 / K6 with three H1 seeds (§4.3 H1 paragraph).**
- **Governing:** K2 and K6 are read on **all three H1 seeds**. Any firing means H1 is not promoted.
- The alternative ("H1-s0 only", from the runbook's S1-D3 drop rule) does not arise: **A3.3 says the drop rule is not exercised**, and all
  nine arms run.
- **Supplementary:** H1-s0 alone.

**A7.3 · AMB-3, K6's step-0 value and its "after 10 % of the steps" row.**
- **Governing step-0 value:** G12's `bce_declared`. That is the head's BCE on a real training batch through the training loss path, at the
  initial weights, recorded in the arm's `PREFLIGHT_PASS.json`.
- **Governing row:** the first `train` event with `update ≥ ceil(0.10 × 400) = 40`. The trainer logs every 25 updates (`scorer_finetune.py:1368`),
  so this is the update-50 row. Its `aux_bce` is a mean over updates 26–50.
- K6 fires iff that row's `aux_bce ≥ 0.9 × bce_declared`.
- **Supplementary, never deciding:** the update-25 row; and ln 2 as the step-0 value.

**A7.4 · AMB-4, K3 at the S1 batch, and a disclosed contamination of its time limb.**
- **Batch.** The draft's K3 was written "at micro-batch 4"; S1's argv uses `--batch 8`.
  - **Governing:** K3 is read at the arm's own batch, 8. Head activations grow with the batch, so the 1 GB memory bar read at 8 is **stricter** than at 4.
  - The memory limbs are as the driver implements them: G12 `head_retained_gb > 1.0`, or the preflight `gpu_mem_gb` of H minus that of the same-seed N `> 1.0`.
- **The time limb: a contamination disclosed BEFORE the read.** The Master Mind ran CPU jobs on the pod (tests, gate pre-runs and diagnostics,
  all at `nice 19`) while some S1 arms trained. End times are the files' mtimes; start times are approximate.

  | arm (training window, UTC) | overlapping Master Mind CPU jobs (end time) |
  |---|---|
  | N-s0 (10-14 19:35:25 – 21:01:30) | `p6suite` (19:47:17), `rls1b_suite` (20:09:38) |
  | N-s1 (10-15 01:23:22 – 02:49:38) | `p2rep_suite_diag2` (01:36:30), `fpbench` gate-token listing (01:45:56), `p2rep_suite_prerun` (02:00:59, ~817 s), `p2rep_fix2` (02:17:23), `count_pod` (02:31:08) |
  | N-s2 (03:19:07 – 04:45:14) | none known |
  | H2-s0 (from 05:24:53) | `p2rep_gsuite_exact` (05:13:26 – 05:33:44, ~9 min of overlap) |

  The trainer's `s_per_update` is the **cumulative** run mean, `(time.time() − t0) / upd` (`scorer_finetune.py:1371`). Any contention window is
  therefore averaged into the literal number, in either arm of a pair and in either direction.
- **Governing:** K3's time limb fires if **either** of two estimates exceeds +3 %, each taken as H vs the same-seed N:
  1. the run-mean at update 400 (the literal reading);
  2. the **median per-window** s/update. With rows every 25 updates, there are 16 windows per arm, each derived exactly from consecutive
     cumulative rows: window (u1, u2] = (s2·u2 − s1·u1) / (u2 − u1).

  The median is robust to a transient contention window. Taking either estimate only adds ways to fire. Both are published.
- **Supplementary:** the mean of the three N seeds as the comparator.

**What it changes.** The read driver only:
- `AMBIGUITIES` entries carry `ruled: A7.x`;
- `read_k3` gains the median-window limb, with a check and a RED mutation in `test_s1_read.py`.

Unchanged: every bar, arm, argv, seed, estimator, outcome name and the read order (A5: the first read happens after all registered arms exist).

## Amendment A8 (2026-10-15 ~16:30Z, after all nine arms exist and BEFORE any S1 outcome is read): a duplicated held-out frame, and two read-path fixes

**What happened.** The first two attempts at the nine-arm read stopped before any outcome was computed. No S1 outcome number has been read by anyone.
- **Attempt 1.** The driver passed a relative `--out` to instrument subprocesses that run with `cwd` = `eval/`. The selection step could not
  write its output; the navtest perception wrote its JSON under `eval/` and returned 0. The read then crashed on the missing file.
  - **Fixed:** `eval/s1_read.py` `abs_paths()` makes every path argument absolute before any is used.
  - **Proof:** check `chk_abs_paths`, which is behavioural (it stops `cmd_read` at its first path use), plus RED mutations
    `abs_paths_not_called` and `abs_paths_noop`. `test_s1_read.py` 97 → 100.
- **Attempt 2.** The held-out perception step (P1-HO) refused H1-s0's `head_pred_update400.npz`: *"key (…, `b7e55cdff61159db`) appears twice with
  DIFFERENT head outputs"*.

**The duplicated frame (MEASURED on the nine arms' banked dumps; instrument properties, not outcomes).**
- **The dumps.** The trainer's held-out head dumps hold the 3,137 held-out sets, which are **1,705 unique t0 frames**: 1,432 tokens appear under
  two ranks. The dumps carry `log`, `token` and `step`, but not `rank`, and `step` is 0 for every duplicate. The harness deduplicates by
  `(log, token)` and required the two copies to be **bit-identical**.
- **The pairs.** In all six head arms and both tags (`step0`, `update400`), **1,431 of 1,432** duplicate pairs are bit-identical.
  - The single exception is the same frame in every arm and tag: token `b7e55cdff61159db`, **rows 95 and 3136**. Row 3136 is the dump's last
    row, i.e. the final partial batch.
  - Max |d| is **0.031 logit** and **0.094 m** of SDF. That is bf16 batch-composition noise, not a different input.

**The change (instrument only).** `eval/aux_s1_readout.py` `load_head_pred`:
- A duplicated `(log, token)` t0 frame keeps its **FIRST** copy, row 95, from a full batch.
- A non-identical duplicate is accepted when max |d| ≤ **1.0 logit** and ≤ **2.0 m SDF**, i.e. one logit unit and one 2 m grid cell. These bounds come from
  the instrument's own units, not from this data: they sit about 30× and 20× above the measured noise. They separate numerical noise from a
  different frame filed under the same key, which is what the check exists to catch. Beyond either bound it still refuses.
- **Reporting:** accepted duplicates are counted (`duplicate_keys_within_tol`) and their max |d| is published (`duplicate_max_abs_diff`) in every perception output.
- **The first bound I tried was rejected.** A relative tolerance of 2⁻⁵ was checked first and failed on the SDF near its zero crossing. It was not
  adjusted to fit the data. The unit-based bound above replaced it.
- **Tests:** `test_aux_s1_readout.py` 63 → 66, with RED mutations `dup_bound_dropped`, `dup_sdf_bound_dropped` and `dup_keeps_last`. The existing
  refuse-fixture moved from +1.0 to +3.0 logit, because +1.0 sits exactly on the new bound.

**What it changes.** One frame out of 1,705 per head arm, in the held-out perception read (P1-HO) only, now uses the full-batch copy instead of
refusing the arm. Unchanged: every bar, arm, argv, seed, estimator, outcome name, the navtest reads (P1 and P3, which govern K2) and the selection read.
The harness tool id stays `aux_s1_readout/1`: its output contract only gains two reported fields.

**Upstream cause, for the next trainer change (not S1).** The held-out head dump should carry `rank`, or write one row per t0 frame, so the
duplicate never reaches the reader.
