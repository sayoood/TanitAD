# PREREG_VDAC_P6_A5 (DRAFT, NOT FROZEN): the arm after a failed S1a: Stage 0, HB, HC

**Status: DRAFT for the Master Mind, 2026-11-12.** Written by the REFe scorer-stream teammate. Nothing here is frozen, trained or run.
It would amend `PREREG_VDAC_P6.md` (sha256 `5594c2ae...`) + A1 (`8f121bdf...`) + A2 (`c3d56358...`) + A3 (`a7708573...`) + A4 (`045a10c8...`)
in the S1 branch table only. It follows the card's own S1 outcome rows (`PREREG_VDAC_P6.md` section 2, S1): both bars failed, so the card
commits **HB** (S1-FIT) or **HB + HC as one registered pair** (S1-WEAK), and Stage 0 below is what splits those two rows.
Primary sources read: the five cards, `DESIGN_VDAC_P6.md`, `s1/HANDOFF.md`, `s1a_read/{S1A_READ,MM_REVIEW,READER_FREEZE}.md`,
`s1a_read/s1_read.py` (sha prefix `58aa3a77`), and at git tip `874ac6aa` the files `refe/{model,scorer_finetune,aux_drivable_head,vdac_probes}.py`.
Evidence classes: MEASURED / INHERITED / ESTIMATED / UNVERIFIED / HYPOTHESIS. Implementation audit: `READINESS.md` (same folder).

## 0. Where we stand (MEASURED, `s1a_read/s1a_read.json`, `S1A_READ.md`)

| item | value |
|---|---|
| HA-s0, frozen P6 trunk, 10,000 updates, 3,207 H48 frames, 45 logs | S1-P1 **0.7802** [0.7587, 0.7974] (bar 0.80); S1-Q **0.8332** [0.7951, 0.8691] (bar 0.90) |
| per city IoU [0,40) | LV 0.891, BOS 0.739, PIT 0.723, SG 0.696 |
| per band | 0-20 m 0.809, 20-40 m 0.750, 40-72 m 0.661 |
| controls | C-SHUF 0.4468 (< 0.50 valid), C-FLIP 0.7056 (< 0.80 valid) |
| the ceiling on the same proposals | true 2 m field AUC 0.9408 [0.9320, 0.9491] (HA-s0 is 0.108 below) |
| trainer's own held-out `aux_eval` IoU (pooled, unweighted, full grid; NOT S1-P1) | 0.212 (step 0), 0.799 (5,000), 0.823 (10,000): still rising |
| missing | in-sample IoU; S1-F. The in-sample IoU is the only missing input that moves the outcome row |

Unchanged from the cards: H48 and its exclusion (A3), the estimator (paired log-cluster bootstrap, 10,000 draws, seed 20260927),
the S1 bars, C-SHUF / C-FLIP, the four metric families (STRATEGIC PAUSED, PI 2026-09-19), the tier (T0-like at S1), the information-disjointness
rules (card section 6), and A4's G20 bar 0.99. No bar below moves after a number is seen.

## 1. Stage 0: the in-sample read (first thing in the next P7 pause; minutes)

**Purpose:** read mechanically the one input S1a could not: HA-s0's IoU on frames it trained on. Not a training run; needs no launch-gate token
(it is a measuring mode, like `--pair-calibrate`, and is forbidden in a launch argv). Tool: `refe/vdac_stage0.py` (to be built, `READINESS.md` item S0;
**it does not exist today**: no `aux_drv.pt` loader exists in the trainer).

**Population (frames the head trained on).** HA-s0 trained 10,000 updates x batch 8 = 80,000 draws from P6's 60,000-set bank (A3, INHERITED;
1.33 epochs, `drop_last`), with the 24 extra H48 logs' labels set `valid = 0`. Every labelled frame of a log outside H48 in the training index was
drawn at least once in the first epoch except at most 7 dropped by `drop_last` (ARITHMETIC; which 7 is not recorded: expected contamination
2,048 x 7 / 60,000 = 0.24 frames, disclosed). Draw: per city (LV / BOS / PIT / SG) the **512 frames with the lowest `sha256("<log>|<token>")`** among
labelled bank rows (row step 0, rank 0, as the trainer's own head dump) whose log is **not in H48** and not in the 24 SFT held-out logs
(`PK/splits/sft_heldout_logs.txt`), N = **2,048**. The list and its sha256 are written by the tool before any forward and are re-verified
by the chain. City per log from the label-store manifest (as A3).

**Read.** HA-s0's final head (`aux_drv.pt`, LFS sha256 `281320071fea...`) through P6's frozen trunk (`model_final.pt` sha256 `968a469d...`), the trainer's own
`fwd()` path, bf16 autocast. IoU = the frozen reader's `frame_stats` + `city_iou_read` (**imported from `s1_read.py`, sha256 prefix `58aa3a77` asserted**),
x in [0, 40) m, mask `logit > 0`, city-reweighted to navtest's mix: the SAME formula as S1-P1. Point decides (as the card words its bars); a
log-cluster 95 % interval (10,000 draws, seed 20260927) is reported.

| outcome (mechanical) | condition | next (committed) |
|---|---|---|
| **S0-FIT** (= the card's S1-FIT) | in-sample IoU point >= 0.85 | HB alone (section 2) |
| **S0-WEAK** (= S1-WEAK) | in-sample IoU point < 0.85 | HB **and** HC as one registered pair (sections 2, 3) |
| **S0-INVALID** | any control below fails | fix the tool, re-run; no HB / HC row starts |

**Controls (fail closed):** (C0-ID) the same tool re-forwards 256 H48 frames of the 24-log set and must reproduce the banked `head_pred_update10000.npz`
rows: mask sign agrees on >= 99.9 % of cells and IoU on those 256 frames within 0.002 of the value computed from the export (a tool that cannot
reproduce S1a's own rows has no authority over the in-sample number); (C0-FLIP) the y-flipped in-sample mask reads at least 0.10 below the in-sample
IoU (the instrument sees misalignment); (C0-FLOOR) the all-drivable constant head's IoU on the same frames is reported (the floor).
Also reported, not deciding: in-sample minus held-out gap (S1-P1 0.7802), per city, per band.

**S1-F, from the existing step-0 predictions (reported; not deciding, because both bars already fail).** The pod holds the seed-0 initial head's H48 predictions
for the 24 SFT logs, `/workspace/data/refe_vdac/ha-s0/scores/head_pred_step0.npz` (1,705 frames; MM probe, `MM_REVIEW.md`). The reader needs the initial head on
all 3,207 frames, so Stage 0 also forwards the **seed-0 initial head** (rebuilt by `build_head(seed 0)`, `aux_drivable_head.py:175-185`; reproduces the init
stream by construction) on the 1,502 extra-log H48 frames and writes it in the reader's schema (`log, token, step, logit, sdf_m`) as `head_init0_b.npz`.
Check: the rebuilt head reproduces `head_pred_step0.npz` on the 1,705 frames (sign agreement >= 99.9 %); if it does not, S1-F is reported for the 1,705-frame
subset only and says so. Then the frozen reader is re-run on CPU with `init0` present and the Stage 0 in-sample value supplied; its own `verdict()` emits S1-FIT / S1-WEAK.
If the Stage 0 tool and the reader disagree on the row, the card (this table) is read first and the discrepancy is reported.

**Stage 0 cost (ESTIMATED):** forwards = 2,048 + 1,502 + 256 = 3,806 frames x 0.066 s (the WIRING_NOTE assumption: forward = 1/3 of P6's 0.197 s/sample;
INHERITED via `DESIGN_VDAC_P6.md` section 7) = 251 s = 0.07 L40-h, + load and bank open ~0.05 = **~0.12; budget 0.15 L40-h; upper 0.25** if frame decode dominates.

## 2. HB: a LoRA-adapted COPY of P6's trunk feeding the head only

### 2.1 Architecture delta (one lever: the head's input trunk is adapted; everything else equals HA-s0)

| item | HA-s0 (S1a) | HB |
|---|---|---|
| tensor the head reads | `visual_ctx = scene_proj(backbone(img) + pos3d)` of P6 (frozen) | `visual_ctx_hb = scene_proj(backbone_copy(img) + pos3d)`, shape [B, 7,680, 256] (same shape), `backbone_copy` = a **`copy.deepcopy` of P6's backbone** held as a SIBLING (not a submodule of `model`, not in `model.state_dict()`) |
| trunk parameters that train | none | the copy's existing **rank-32 Q/V LoRA tensors `A`, `B`** (24 blocks x {q, v} x 32 x (1,024 + 1,024) = **3,145,728**, ARITHMETIC from `model.py:216-232, 295-310, 397`, `BACKBONES["vitl16"]` depth 24 width 1,024; UNVERIFIED against checkpoint tensors until the preflight prints it) |
| LoRA initialisation | - | **P6's own trained LoRA** (not zero): at step 0 `visual_ctx_hb` is bit-identical to P6's `visual_ctx` (asserted, section 2.3), so HB starts exactly where HA-s0's head read, and the only change is that the trunk copy may now move |
| frozen in the copy | - | every base weight of the ViT-L (patch embed, `cls`/register tokens, norms, MLPs, K, output projections, `LoRALinear.base`): `requires_grad False`, as `model.py:216-232` builds them |
| shared, frozen, executed on the copy's tokens | - | P6's `pos3d_mlp` and `scene_proj` (their parameters stay `requires_grad False`; gradient flows through them to the copy; they are not updated) |
| head | S1c head, 2,440,450 params, 2 m cells, `--aux-drv-tap visual`, depth 2, feed off | **identical** (same seed 0 init stream) |
| head loss | BCE + 1.0 x Huber(beta 0.1) | identical (`--w-aux-drv` named, as HA) |
| optimiser | head AdamW (`--aux-drv-lr` = S1c's registered value, UNVERIFIED here) | head AdamW unchanged **plus** a second param group (the copy's LoRA), AdamW, weight decay 0.01, **`--aux-drv-lora-lr` = 5e-5** (HYPOTHESIS: a quarter of `train.py`'s 2e-4 global default; P6's own LoRA-group lr is UNVERIFIED and must be read from P6's argv before freeze; same warm-up + cosine shape as the head), own grad-clip norm |
| gradient into the copy | - | from `L_aux` only. The scorer arms read P6's ctx, not the copy's; `autograd.grad(scorer loss, copy LoRA)` is None (asserted) |
| P6's trunk, decoders, 64 proposals, six scorer heads | frozen | frozen and **bit-identical** (proof below) |

**How P6's trunk is proved untouched (four independent facts, all asserted by the gate, not read off the code):**
1. *Structure:* the existing freeze (`scorer_finetune.py:1739-1747`) leaves every P6 parameter `requires_grad False` except the scorer copies; the copy is a different object (`deepcopy`), its parameters are the only trunk tensors in any optimiser.
2. *Gradient:* after a real head backward, every parameter of `model` outside `SCORER` has `.grad is None` (or exactly 0.0), while the copy's 3,145,728 LoRA tensors read non-zero, and with `--w-aux-drv` forced to 0.0 they read exactly 0.0 (the control).
3. *State:* the sha256 of `model.state_dict()` (every P6 tensor) before the run equals the sha256 after the last update, and the checkpoint `model_final.pt` sha256 `968a469d...` is re-hashed by the chain.
4. *Function:* the P6 forward of the same pass reproduces `proposals_p6final.npz` `logits` bit-identically on the verification frames (G0-ID of the card, now also read at training time on 64 frames).

### 2.2 Data, updates, seeds, arms

- **Head training data and the H48 exclusion: unchanged.** Same P6 60,000-set bank draw, `--epochs 2` named, `--max-updates 10000` binds, batch 8 effective
  (micro-batch set by the preflight's memory probe, `--accum` to 8; BN-free, so equivalent: ESTIMATED, the memory peak at P6's launch config was <= 44 GB on the 48 GB card, `p6_pod.sh` G-LIVE-MEM, and
  HB adds a second ViT-L forward with trunk backward plus 1.2 GB for the copy; UNVERIFIED that batch 8 fits without accumulation), `--aux-drv-exclude-logs` = H48 list sha256 `2ae13935...`, `--aux-drv-labels` = the navtrain store (103,139 frames, sha256 `4a17db67...`).
  **Same 10,000 updates as HA-s0 on purpose:** a longer HB would confound the trunk lever with training length (the held-out IoU was still rising from update 5,000 to 10,000, section 0). Training length is a named successor (section 5), not part of this arm.
- **HB-s0** (seed 0, the card's single-lever comparison against HA-s0). **HB-SHUF** (seed 0, `--aux-drv-shuffle-seed 0`, 2,000 updates, a derangement of the label keys): the deliberate regression for HB. It matters
  more here than for HA: a head plus a trainable trunk copy is a bigger memoriser, so C-SHUF < 0.50 is the proof that HB's gain is map information and not a leak through the copy. **HB-s1** (seed 1, the replicate) is trained **only if HB-s0 reaches X-GO** (section 4), as A2 section 4 does for HA.
- Banked per arm: the initial head (init floor), `aux_drv.pt` (format `aux_drv/2`: head + LoRA tensors + both optimiser states), `head_pred_update*.npz`, `head_pred_h48x.npz`, an in-sample dump of the same 2,048 Stage 0 frames (so the in-sample gap exists for every arm), the H48 export `h48_proposals_subscores.npz` (shared with S1a: P6's proposals do not change).

### 2.3 What the S1 read needs, and whether it needs a reader change

The frozen reader `s1_read.py` (sha256 `58aa3a77b976...`, `READER_FREEZE.md`) reads a head from **two npz files** of the schema `log, token, step, logit, sdf_m` ([N,40,32] float32; `step` must be 0, the bank row step), re-hashed against a manifest.
- **HB (2 m grid, same head): the metric code needs no change** (IoU, d_k, AUC, controls, bootstrap are grid-identical). **The loader does:** `load_real` reads fixed arm names (`s0 / shuf / init0`, `s1_read.py:562-565`), `run_read` reports "HA-s0" / "HA-SHUF" (`:442-445`), and the paired delta needs an HB-minus-HA-s0 block through the same log resamples (`diff_block`, `:343`). So: **yes, a reader change is needed, and it must be frozen before any HB row is read** (the card's order: instrument before data). Form: `s1_read_v2.py` = the frozen module's functions imported unchanged (sha asserted) + (i) a manifest-driven arm table, (ii) paired HB-vs-HA-s0 blocks for S1-P1 and S1-Q (reported; defines X-LIFT, section 4), (iii) the verdict function applied per arm with that arm's own SHUF as C-SHUF and its own y-flip as C-FLIP.
- **Regression identity (the freeze condition):** `s1_read_v2.py` run on the S1a inputs must reproduce `s1a_read.json` (S1-P1 0.7802, S1-Q 0.8332, C-SHUF 0.4468, C-FLIP 0.7056, all CIs) to the last digit, and its mutation arms (`no_reweight`, `frame_resample`) must still go RED. The freeze record (`READER_FREEZE`-style, sha256 of the file, its tests and the manifest) is written before any HB row.
- **HC (1 m grid) needs more:** CELL / NCX / NCY are module constants (`:56`), `block_targets` is a 4 x 4 reduction (`:111-119`), `band_cells`, `sdf_from_mask(.., CELL)` (`:289`) and the sigma_b block (`:454`) assume 40 x 32, and the load buffers are `[N, 40, 32]` (`:566-567`). Defined now so S1-P1 stays comparable to S1a: **HC's S1-P1 is computed on the 2 m reference grid**, the four 1 m logits of a 2 m cell averaged in probability space and thresholded at 0.5 (a fixed rule, not tuned), against the same 2 m label mask; HC's **S1-Q uses the 1 m mask** (EDT at cell 1.0, bilinear at 1 m) because the 1 m field is the lever; the 1 m IoU against a 1 m label mask is reported. This is a second reader change, frozen with the first.
- **Inputs the new arms' files must carry:** the same npz schema, one file per H48 group (24 SFT logs, 24 extra logs), one per control arm, the initial head of each arm's seed (S1-F), and the manifest sha256 entries. For HB the head predictions come from the copy's ctx; the file name and schema are identical to HA's.

### 2.4 Inference cost of HB (report, do not gate; PI ruling: max 100 ms end to end on Thor, 4 cameras; never gate research on latency)

HB needs the trunk to run twice per frame (P6's frozen pass for the proposals and the scorer, the copy's pass for the head), since the copy's LoRA differs from P6's after step 0.
- Compute: the head is 32.2 GFLOP = 0.53 % of the trunk's forward (`SPEC_AUX_DRIVABLE_HEAD.md` section 0, as cited in `DESIGN_VDAC_P6.md` section 8) => trunk forward = 32.2 / 0.0053 = **~6.1 TFLOP per sample** (ESTIMATED, derived). HB adds one more trunk forward: **+~6.1 TFLOP, perception compute x2** (the rest of P6 is small beside the ViT-L).
- Latency: P6's own Thor latency is **not measured** (F6 item 6 of `FEASIBILITY.md`, cited by the design card), so no absolute figure exists. Relative: sequential second pass ~ +100 % of the trunk's share; if the two passes are batched on the GPU, the saving is ESTIMATED to be small because the trunk is compute-bound at 4 x 1,920 tokens (UNVERIFIED).
- A cheaper variant exists and is **named, not registered**: adapt only the last k blocks, share the first (24 - k) blocks' activations with P6 (a second pass of k/24 of the trunk; k = 8 gives +33 %). It would change the lever (fewer adapted blocks), so it is a successor if HB passes S1 and the latency matters, never a reason to skip HB.

## 3. HC (registered only on S0-WEAK): finer cells and a boundary-weighted loss on the frozen trunk

| item | HC |
|---|---|
| trunk | P6's frozen trunk, as HA-s0 (no copy, no second pass) |
| head | the same head class at `cell = 1.0` m: 80 x 64 = 5,120 learnable queries over the same extent x [-8, 72), y [-32, 32), depth 2, ctx 256 (the class supports it: `aux_drivable_head.py:142-161`; **the trainer cannot reach it today**, `READINESS.md` item C1: `scorer_finetune.py:1767` builds `AuxDrivableConfig` without `cell`) |
| params | ~3.42 M (ESTIMATED by arithmetic from the class: only the query table grows, 5,120 x 256 = 1.31 M vs 1,280 x 256 = 0.33 M at 2 m, on top of the measured 2,440,450; the preflight prints the real count and G-DVB asserts it) |
| loss | `BCE + w_sdf x Huber` as HA, with the Huber term multiplied per cell by `1 + 3 x 1[ |block-mean SDF_gt| < 3 m ]` (**boundary band 3 m = the card's near-boundary definition; weight 3 is HYPOTHESIS, fixed now, not tuned**); BCE unweighted; both at 1 m block statistics (`block_stats` with cell 1.0, supported: b = 2) |
| updates, seeds, data, exclusion | as HB: 10,000 updates, batch 8, seed 0, same bank draw, H48 exclusion unchanged; **HC-SHUF** seed 0, 2,000 updates, derangement |
| reader | the HC reader change of section 2.3, frozen before any HC row |

HC addresses the S1-BOUND failure mode (boundary precision; near-boundary SDF MAE 1.63 m, RMS 2.36 m in S1a, MEASURED) and HB addresses the S1-FIT mode (features frozen to P6's scene objective). Under S0-WEAK the card does not say which one the deficit is, which is why it registers both.

## 4. Outcomes, committed before any data (Rule Zero: every row names its successor)

Each new arm X in {HB, HC} is read by the same S1 bars as HA-s0, with its own controls and the paired delta to HA-s0 on the same 3,207 frames and the same log resamples:
S1-P1 >= 0.80; S1-Q >= 0.90; S1-F (trained minus its own seed's initial head, both 95 % lower bounds > 0); C-SHUF (X-SHUF's S1-P1 < 0.50); C-FLIP (X's y-flipped S1-Q < 0.80).
**X-LIFT** = both paired deltas (S1-P1 and S1-Q, X minus HA-s0) have 95 % lower bound > 0. A one-seed separated delta is necessary and not sufficient for a lever claim (`H-ESTIM-SEED-1`); it triggers the replicate, not a claim.

| id | condition (arm X) | next (committed) |
|---|---|---|
| **X-GO** | S1-P1, S1-Q, S1-F hold, controls valid | **S1b for X** (seed 1, the A2 rule: ~ HB 4.4 to 7.6, HC 4.3 L40-h + chain, the PI's pod time), then **S2** under the card's table with X's head replacing HA-s0 (VDAC-V0T primary, bars P1 0.4971 / P1r / M / G1-G4 unchanged, K + 8 for the substituted arm set, HA arms are not read on E3). If both HB and HC are X-GO, **both** go to S2, the one with the higher S1-Q point is named primary before S2, the other is a reported comparison. |
| **X-COMP** | S1-Q holds, S1-P1 fails, LV >= 0.80, a non-LV city < 0.70 | HA-CB (city-balanced draws, the card's registered fallback) applied on X, its own seed pair |
| **X-BOUND** | S1-P1 holds, S1-Q fails | if X = HB: **HBC** (1 m cells + boundary weight on the LoRA copy); if X = HC: **HBC** likewise. Cost ~ HB per arm + 1.1 L40-h |
| **X-LIFT** | both bars fail, X-LIFT holds | the lever is real and short of the bar: (a) if both HB and HC lift: **HBC**; (b) if one lifts: that arm at **20,000 updates** (the held-out curve was still rising; the one registered training-length arm), one seed first; PI pod time |
| **X-NULL** | both bars fail, no separated lift on either delta | perception of the DAC from P6's single-trunk features is not moved by this lever. If every arm run is X-NULL: **the DAC-only perception line is closed on these levers**; next is the agent check (D-VDAC-6; label build from NAVSIM's agent annotations, NC 1.12 % + TTC 1.527 points, the larger term by `FEASIBILITY.md` F5), with its zero-GPU ceiling (an S0-style read) first. Rule Zero: that card's first read is named here so the line does not stop at this row |
| **INVALID** | C-SHUF or C-FLIP fails for X, a launch-gate fact contradicts the arm, or the reader's regression identity fails | fix, re-run the affected arm; no row counts |

**Branch read.** S0-FIT: HB alone is read. S0-WEAK: HB and HC are read together, each against the table above, and the branch result is the pair of rows. If only one arm is X-GO, only that arm continues; the other's row stands as written.
Reported with every number: the A3 disclosures (H48 in P6's own training set; the 60,000-set bank draw with `--epochs 2`; HA-SHUF's deranged key map), the A4 G20 history, and "single head seed" until S1-R.

## 5. Cost in L40-h (the pod GPU is an L40; A40-h x 0.863 ESTIMATED from P6's rate, `pod-gpu-is-l40` note)

Anchors: HA arm 3.20 L40-h per 10,000-update arm (3.71 A40-h x 0.863; MEASURED S1c rate x ESTIMATED factor, `DESIGN_VDAC_P6.md` section 7), i.e. 0.144 L40-s per draw; P6's own step 50.5 s per 256 samples = 0.197 s per sample (INHERITED, `PREREG_SCORER1_AMENDMENT_3.md:24`); chain overhead 0.45 L40-h per arm (INHERITED, `s1/HANDOFF.md`) + 0.1 for the extra G21 probes and the memory probe.

| stage | arithmetic | lower | upper |
|---|---|---|---|
| Stage 0 | section 1 | 0.12 | 0.25 |
| HB-s0 | lower = the card's HB figure, 80,000 x 50.5 / 256 s = 15,781 s = 4.38 h (P6's own step cost, the head replaces the decoders); upper = HA 3.20 + a full P6-rate copy fwd+bwd 80,000 x 0.197 / 3,600 = 4.38 => 7.58 | 4.4 | 7.6 |
| HB-SHUF | 0.2 x HB-s0 | 0.88 | 1.52 |
| chain overhead, 2 arms | 2 x 0.55 + one-time export 0.1 | 1.2 | 1.2 |
| **S0-FIT total (Stage 0 + HB)** | sum | **6.6** | **10.6** |
| HC-s0 | HA 3.20 + 1.1 (4x queries, the card's ESTIMATE) | 4.3 | 4.3 |
| HC-SHUF | 0.2 x 4.3 | 0.86 | 0.86 |
| chain overhead, 2 arms | 2 x 0.5 | 1.0 | 1.0 |
| **S0-WEAK total (Stage 0 + HB + HC)** | sum | **12.8** | **16.7** |
| conditional later: HB-s1 replicate | one arm + chain | 5.0 | 8.2 |
| conditional later: HC-s1 replicate | one arm + chain | 4.8 | 4.8 |
| conditional later: S2 forward of an HB head (navtest, 12,146 tokens) | P6 forward 0.22 (card) + the copy's pass ~0.22 | 0.44 | 0.44 |
| CPU | the S1 reads and the reader regression: minutes; no GPU | | |

**The bracket is wide on purpose:** the one unmeasured quantity is the step cost of a ViT-L trunk copy with LoRA backward sitting beside P6's forward and the head. The preflight's speed probe (the trainer already measures `s_per_microbatch`, `scorer_finetune.py` preflight) prints the real figure; the PI approves the **upper** bound as the cap and the pause script's timeout is set from the measured rate (the S1a values, 18,000 s per 10,000-update arm, are too short for HB's upper bound of ~27,300 s; to be set to 36,000 s, UNVERIFIED how `vdac_pause_p7.sh` takes it).
**P7 pause:** one pause for Stage 0 (minutes) then the branch's arms, sequentially on one GPU. For S0-WEAK the PI may split HB and HC into two pauses; the card does not require it.

## 6. Launch-gate obligations (no training launch without a PASS token bound to commit, argv sha256 and data-manifest sha256s)

Profile `refe_vdac_p6` gains arms HB-s0, HB-SHUF (and HC-s0, HC-SHUF), as new entries of `vdac_argv(arm)` in a deep copy; shared constants untouched; all ~69 launch_gate-reading tests run before landing (`pin-edit-moves-profile-fingerprints`).

| flag (all default OFF; each must be NAMED in its arm's argv) | G-DVB entry | deliberate regression that must turn it RED |
|---|---|---|
| `--aux-drv-trunk-lora on` (HB) | built facts: `trunk_copy_built`; `n_lora_trainable == 3,145,728`; copy is a `deepcopy` of the loaded `model.backbone` (storage differs); copy not in `model.state_dict()`; `p6_params_requires_grad == 0` | name the flag but build no copy; build the copy sharing P6's storage |
| `--aux-drv-lora-lr` (HB) | named; equals the LoRA param group's peak lr | omit it (refused like `--aux-drv-lr`) |
| `--aux-drv-cell 1.0` (HC) | head `nx, ny = 80, 64`; `n_params` equals the built count | declare 1.0, build 2.0 (the unreachable-flag class: today the cell never reaches the build) |
| `--aux-drv-bw 3.0 --aux-drv-bw-band-m 3.0` (HC) | the multiplier actually applied in the built loss, read off a probe cell inside and a cell outside the band | name it, drop it from the loss |
| `--stage0-*` (the measuring mode) | refused in any launch argv | present in a launch argv |

| gate | how it covers HB / HC |
|---|---|
| **G19** head disjoint from ego and command | Unchanged probe (`vdac_probes.disjoint_probe`, called with the real `fwd`): autograd of the head output w.r.t. the ego and command tensors None or exactly 0.0, with a non-zero control. **For HB it holds only if `actx` is assigned from the copy's ctx before the existing mutation line (`scorer_finetune.py:1982-1985`),** so the existing mutant `REFE_AUX_MUTATE_EGO=1` (adds the ego token into `actx`) still lands on the copy's output. **New copy-specific mutant `REFE_AUX_MUTATE_EGO_TRUNK=1`** adds the ego token to the copy's visual tokens before `scene_proj`; G19 must turn RED under both. The copy reads images and the camera calibration table only (the same inputs P6's `visual` reads); no ego, command, map or route. |
| **G20** label alignment | Unchanged and independent of the trunk (label vs P6's rank-0 target points; bar 0.99, A4; both mutants < bar). Re-run in each arm's own preflight; HC's 1 m grid does not touch it because the check reads the 0.5 m label store directly. |
| **G21 (new): copy isolation** | the four facts of section 2.1 as raw fields re-judged by `judge_g21` (pure function in `vdac_probes.py`, as G19/G20): P6 params outside `SCORER` have no gradient; the copy's LoRA tensors are non-zero; with `w_aux = 0` they are exactly 0.0; step-0 `ctx_hb` equals P6's `ctx` bit-for-bit; scorer-loss gradient on the copy is None; the `model.state_dict()` sha256 is unchanged across 2 updates. Mutants (each RED): the head reads P6's ctx instead of the copy's (LoRA gradient reads None); unfreeze P6's `scene_proj` (P6 gradient non-None); detach the copy's output. |
| G12 tap (existing) | its "tap is `visual_ctx`" row reads `outs[1]` of `scene_proj`'s hook (`scorer_finetune.py:2408-2414`). HB calls `scene_proj` a second time, so `outs[2]` is the copy's ctx; the row must be re-pointed and asserted (`scene_proj_calls == 3`), otherwise it goes RED or, worse, passes on the wrong tensor. |
| G-LIVE | initial `aux_bce` ~ ln 2 +/- 0.15; head and LoRA gradients non-zero; trunk of P6 exactly zero; scorer arms bit-identical to a head-less run (feed off) |
| G-CKPT | head, LoRA tensors and BOTH Adam states round-trip; 2 resumed updates bit-identical |
| post-training | every head AND LoRA parameter has Adam state (`probe_zerograd3.py` pattern) |
| G-EVAL | S2's navtest tool builds the identical head **and the copy** (strict load of `aux_drv.pt` format 2) and reproduces P6's banked logits (G0-ID); **not built for any arm today** (`HANDOFF.md`) |
| G-SUITE | clean-tree tests incl. the new ones of `READINESS.md`; the pod pre-run (`suite-prerun`) |

## 7. What must be frozen before the first number

1. `s1_read_v2.py` + tests + freeze record (regression identity on S1a, section 2.3). 2. `vdac_stage0.py` + tests + freeze record + the in-sample list sha256 (the list is data-independent, written before any forward). 3. This card, by sha256, with `--aux-drv-lora-lr` and the HC weight confirmed or changed by the MM (both are HYPOTHESIS values here). 4. The profile arms and a launch-gate PASS token per arm. 5. The PI's pod time and the P7 pause.
**Open decisions for the PI:** the pod-hour cap (6.6 / 10.6 for S0-FIT, 12.8 / 16.7 for S0-WEAK) and the P7 pause(s); whether HB and HC may share one pause.
