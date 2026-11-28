# PREREG_VDAC_P6_A6_HBC_DRAFT: HBC, the combined arm after HB-s0 read X-BOUND

**Status: DRAFT, NOT FROZEN, NOT LAUNCHED.** Written 2026-11-13 by an engineering agent for the Master Mind (MM). The PI decides the pod time. Nothing in this draft changes a bar of `PREREG_VDAC_P6_A5.md` (sha256 in `PREREG_VDAC_P6_A5.sha256`) or its amendment A1 (`2b40a672...`, landed `2efb14dd`); it adds one arm pair to the S1 branch table and nothing else.
Evidence classes: MEASURED (artifact path) / PUBLISHED / INHERITED (not re-verified) / ESTIMATED / HYPOTHESIS / UNVERIFIED. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Tip read: `25b07b01`.

## 0. Why this arm (the row it executes)
The A5 card (section 8) commits, for an arm X with S1-P1 holding and S1-Q failing, **X-BOUND -> "HBC (1 m cells + boundary weight on the LoRA copy), ~HB per arm + 1.1 L40-h"**. HB-s0 read exactly that row (MEASURED, `PK/raw/2026-11-09-scorer1p6-prereg/vdac/a5_read/hb_read/READ_HB.md`, `out/s1_read_v2.json`):

| arm | S1-P1 [0,40) (bar >= 0.80) | S1-Q (bar >= 0.90) | paired vs HA-s0 |
|---|---|---|---|
| HA-s0 | 0.7802 [0.7587, 0.7974] | 0.8332 [0.7951, 0.8691] | reference |
| **HB-s0** | **0.8113 [0.7924, 0.8260] holds** | **0.8376 [0.7963, 0.8744] fails** | P1 +0.0311 [+0.0224, +0.0359] separated; Q +0.0043 [-0.0046, +0.0119] not separated; X-LIFT False |

The LoRA trunk copy moved drivable-area IoU over its bar and did not move the ranking of DAC-0 against DAC-1 proposals. The remaining gap on S1-Q is 0.0624 (0.8376 to 0.90). HBC asks whether adding the head-side precision levers (1 m cells + a boundary-weighted Huber) on top of the adapted trunk moves S1-Q. S1-Q is the clause that gates S2 (READ_HB.md, "Next" item 3): until it holds, a better mask gives the scorer nothing.
Single seed, single arm per cell. Every number below that will be printed carries "single head seed" until a replicate exists (`H-ESTIM-SEED-1`).

## 1. The lever, stated honestly: a TWO-lever arm
HBC = **HB's lever** (the head reads a LoRA-adapted COPY of P6's trunk, `--aux-drv-trunk-lora on --aux-drv-lora-lr 0.0002`) **+ HC's levers** (`--aux-drv-cell 1.0 --aux-drv-bw 3.0 --aux-drv-bw-band-m 3.0`: 80 x 64 = 5,120 queries, Huber multiplied by `1 + bw` = 4 inside the 3 m near-boundary band). HC is itself two levers in one arm (cell and weight) by the A5 card's own construction; HBC therefore carries **three** knobs (copy, cell, weight) and the card says so.
Comparators and what each can and cannot give (all on the same 3,207 H48 frames, the same log resamples):

| contrast | reads as | cannot say |
|---|---|---|
| HBC-s0 minus **HB-s0** | the effect of (1 m cells + boundary weight) GIVEN an adapted trunk. This is the decisive contrast for the X-BOUND successor question (does head precision move S1-Q on top of the copy). | whether the cell or the weight did it (HC's own confound, inherited) |
| HBC-s0 minus **HC-s0** | the effect of the adapted trunk GIVEN the 1 m head and the weight | the same single-seed limit |
| HBC-s0 minus **HA-s0** | the joint effect of all three knobs against the S1 head | an attribution to any one knob |
| interaction (HBC - HB) - (HC - HA) | whether the two levers add, over-add or fail to add | NOT computed by the frozen reader (`eval/s1_read_v2.py` builds `pairs` of {x, ref} one contrast each: a 4-arm difference is not a pair). **UNVERIFIED** that it can be derived without a reader change; if wanted it is a derived quantity from the per-frame logits through the SAME resamples and needs its own freeze before the read. It never decides a row below. |

Not separable here, by construction: the cell versus the weight (one HC arm, one HBC arm); the copy's LoRA learning rate (fixed at P6's 2e-4, A5 D1); training length (the same 10,000 updates as every A5 arm, on purpose: a longer arm would confound the lever with length). **A separated one-seed interval is necessary, not sufficient, for a lever claim** (the paired bootstrap resamples logs with the models held fixed; it is blind to training variance; `H-ESTIM-SEED-1`). The replicate (HBC-s1, seed 1) is the sufficient form and is a named conditional successor, not part of this arm pair.

## 2. The arms and their argv (every lever NAMED; each default disables its lever)
Parent argv: `scorer1p6_pod.sh ARM=C0 MODE=argv` with `--max-updates / --eval-every / --seed / --out` replaced, then the common tail of the A5 card section 5 (`--epochs 2 --aux-drv --w-aux-drv 0.1 --aux-drv-lr 0.0002 --aux-drv-tap visual --aux-drv-depth 2 --aux-drv-labels <store> --aux-drv-exclude-logs <H48> --aux-drv-exclude-sha256 2ae13935...`).

| order | arm | seed | updates | micro-batch | extra flags, ALL NAMED | purpose |
|---|---|---|---|---|---|---|
| 1 | **HBC-s0** | 0 | 10,000 (eval every 5,000) | `--batch 4 --accum 2` (effective 8) | `--aux-drv-trunk-lora on --aux-drv-lora-lr 0.0002 --aux-drv-cell 1.0 --aux-drv-bw 3.0 --aux-drv-bw-band-m 3.0` | the combination |
| 2 | **HBC-SHUF** | 0 | 2,000 (eval every 1,000) | `--batch 4 --accum 2` | HBC-s0's flags + `--aux-drv-shuffle-seed 0` | deliberate regression: a derangement of all 103,139 label keys (C-SHUF) |

* **Code reach (MEASURED on CPU by `refe/test_vdac_hbc.py`, see the MANIFEST for the run):** `launch_gate.VDAC_ARMS` holds both arms (`extra: "hbc"`, classes `H1hbc` / `H1hbcshuf`); `vdac_argv` names HB's two flags then HC's three; `vdac_pod.sh` MODE=argv equals it token for token; `vdac_chain.sh` has both arms in its table with the 1 m head count 3,423,490 passed to the post-training gate; `vdac_pause_p7.sh` has the stage **`A6HBC`** (`HBC-s0` then `HBC-SHUF`, as `A5HB`).
* **Batch and allocator:** HBC runs HB's registered fallback from the first step (HB-s0 at `--batch 8` raised `torch.OutOfMemoryError`, L40 44.39 GiB, 37.78 GiB allocated, MEASURED 2026-11-12, `launch_gate.py` comment at the arm table) and HB's `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`vdac_pod.sh`). Effective batch, seed, update count, lr and data are the A5 card's. A5 section 9.2 allowed the fallback as an implementation detail "only before any HB step trains"; for HBC it is part of the arm from the start, so no step of HBC at batch 8 exists to be contrasted.
* **Comparability caveat (stated, not fixed):** HB-s0 trained at batch 4 x accum 2 and HC-s0 at batch 8 x accum 1. HBC (4 x 2) matches HB-s0 and differs from HC-s0 and HA-s0 in micro-batching only; the effective batch and the update count are equal. The accumulation is a numerical-order difference, not a lever (HYPOTHESIS: no measured size; same-seed replicates are bit-identical only at identical micro-batching).

## 3. Gates, controls and regressions (no launch without a PASS token; PI 2026-09-26)
Profile `refe_vdac_p6`, argv pinned token for token. The chain mints a token per arm. The G-DVB rows read all five levers against the data event of ONE build: head `nx, ny == 80, 64`, `n_params == 3,423,490`, optimiser covers it (cell); the multiplier the BUILT loss applies, 4 inside / 1 outside (weight); copy built, `n_lora_trainable == 3,145,728` (profile `hb_lora_params`), all trainable tensors LoRA, storage disjoint, `p6_params_requires_grad == 0` (copy); G21, G12 (tap = the copy's ctx, `scene_proj_calls == 3`, `tap_index == 2`; `head_retained_bound_gb` 4.0 at cell 1.0), G19 (both mutants), G20, G-LIVE, G-CKPT (`aux_drv/2`), the exclusion audit and the post-training gate are unchanged from the A5 card section 6.
**Defect found and fixed in this build (MEASURED by reading `vdac_chain.sh` at the tip):** the chain called `vdac_postgate.py --updates N` with no `--n-params`, whose default is the 2 m head's 2,440,450; a 1 m arm's Adam state covers 3,423,490, so the post-training gate would have printed `ZZVDAC_POSTGATE_FAIL ... the card's built head has 2440450` for **HC-s0, HC-SHUF and any HBC arm** after the whole training run. The chain now passes `--n-params $HEADN` (2,440,450 or 3,423,490 per arm). The tip's HC arms are exposed to it (see the hand-back).
Deliberate regressions (each in `test_vdac_hbc.py`, each must go RED): the copy's build line deleted while the 1 m head is built; the `select_actx` wiring line deleted; the `lora_step` line deleted; the cell and the weight dropped from the build while the copy is built; ONLY the weight dropped; the shuffle seed dropped from the label source; the four HB runtime switches (`P6CTX`, `DETACH`, `SHARE`, `UNFREEZE_SP`) with the 1 m head; the weight line deleted from the loss; the 1 m query tensor frozen; the head's backward never run; a dead weight-0 control; the post-training gate without `--n-params` on a 1 m arm. The control that reads EXACTLY 0.0: at `--w-aux-drv 0` the head's 5,120 query rows, every head tensor and every LoRA tensor read 0.0 (a tensor, not None).

## 4. Bars (unchanged from the A5 card section 7.3; none moves after a number is seen)
On the same 3,207 H48 frames: **S1-P1 >= 0.80** (point decides; HBC's P1 is read on the 2 m reference grid exactly as HC's, the four 1 m logits averaged in probability space, threshold 0.5; the 1 m IoU is reported, not deciding); **S1-Q >= 0.90** (the 1 m mask); **S1-F** (trained minus its OWN initial head, both 95 % lower bounds > 0); **C-SHUF** (HBC-SHUF S1-P1 < 0.50; it matters most here: a copy plus a 5,120-query head is the largest memoriser in the family, so C-SHUF < 0.50 is the proof that any gain is map information and not a leak); **C-FLIP** (y-flipped S1-Q < 0.80). **X-LIFT** = the paired deltas (S1-P1 and S1-Q) have 95 % lower bounds > 0 against the contrast named below.
* **X-LIFT's reference for HBC is HB-s0** (the committed successor asks what the head levers add to the copy). The reader's `pairs` list is ordered `HBC-s0 - HB-s0`, `HBC-s0 - HC-s0`, `HBC-s0 - HA-s0`; the reader takes the arm's `x_lift` from the FIRST pair naming it (`s1_read_v2.py`, MEASURED by reading the file), so the order is part of the freeze.
* **S1-F's initial head:** HBC's step-0 ctx is bit-identical to P6's (G21), so its step-0 head equals HC's step-0 head (same seed, same cell). The 1,705 SFT-held-out frames come from HBC's own `scores/head_pred_step0.npz`; the 1,502 extra-log frames need `vdac_stage0_forward.py --mode init` with `expected_aux_cfg` passing `cell` (A5 section 9.4, not built). **Pre-registered fallback, as HC's: S1-F read on the 1,705 frames, stated in the verdict row.**
* **Reader:** `eval/s1_read_v2.py` with a manifest adding the HBC arm (`cell` 1.0, `role` trained, its own `shuf` and `init`) and the three pairs above. The reader's regression identity and mutation arms must re-pass before the first HBC row is read (A5 section 7.4). Reader changes: none expected; UNVERIFIED that the manifest accepts a trained arm with a copy AND `cell` 1.0 together (HB and HC each exercised one).

## 5. Outcomes, committed before any data (Rule Zero: every row names its successor)
Let X = HBC-s0. Rows are mechanical against section 4.

| id | condition | next (committed) |
|---|---|---|
| **HBC-GO** | S1-P1, S1-Q, S1-F hold; C-SHUF, C-FLIP valid | **HBC-s1** (seed 1 replicate, ~ the cost of HBC-s0, the PI's pod time), then **S2** with HBC's head replacing HA-s0 under the A5 card's X-GO row (VDAC-V0T primary, bars unchanged, K + 1 for the arm read on E3 at S2 only). If HB-s0 or HC-s0 is also X-GO by then, the higher S1-Q point is named primary BEFORE S2 (A5 section 8). |
| **HBC-BOUND** | S1-P1 holds, S1-Q fails (HB's own row) AND the paired S1-Q delta vs HB-s0 has lower 95 % bound > 0 | the head levers move the ranking but not to the bar: **HBC at 20,000 updates**, seed 0 (the held-out curve was still rising, A5 section 5), then, if S1-Q still fails, the agent check below |
| **HBC-BOUND-NULL** | S1-P1 holds, S1-Q fails, and neither the HB-s0 nor the HC-s0 paired S1-Q delta is separated | boundary precision and an adapted trunk, alone or combined, do not move the DAC ranking. The DAC-only perception line is closed on these levers; **next is the agent check (D-VDAC-6)**: label build from NAVSIM's agent annotations (NC 1.12 % and TTC 1.527 points, the larger term by `FEASIBILITY.md` F5), its zero-GPU ceiling read first (A5 section 8, X-NULL) |
| **HBC-COMP** | S1-Q holds, S1-P1 fails, LV >= 0.80, a non-LV city < 0.70 | HA-CB (city-balanced draws, the card's registered fallback) on HBC, its own seed pair |
| **HBC-LIFT** | both bars fail, both paired deltas vs HB-s0 separated > 0 | the levers are real and short: HBC at 20,000 updates, one seed first; PI pod time |
| **HBC-REG** | S1-P1 falls below HB-s0's 0.8113 with a separated negative paired delta | the levers interfere (a head with 4x queries on an adapting trunk); report per city and per band; successor = HC-s0's row decides: if HC-s0 alone is X-GO, S1b on HC-s0 (A5 section 8 X-GO) and HBC is closed as a combination |
| **INVALID** | C-SHUF or C-FLIP fails for HBC, a launch-gate fact contradicts the arm, or the reader's regression identity fails | fix, re-run the affected arm; no row counts |
| **NOT RUN** | blocked at preflight, gate or timeout | stands as "NOT RUN: <stage, reason>", never read as a null; the blocker's smallest fix is the next step. A **memory** refusal names the second fallback below as the next step |

**Branch read.** Reported with every number: the A3 disclosures (H48 in P6's own training set; the 60,000-set bank draw with `--epochs 2`; the SHUF key map), "single head seed", the contrast each delta answers (logs, not training runs), and that HBC is a three-knob arm. The A5 pair row of HB-s0 (X-BOUND) is not revised by anything here.

## 6. Memory and the registered fallback
MEASURED facts: L40 44.39 GiB; HB-s0 at `--batch 8` OOM at preflight (37.78 GiB allocated, 44.36 GiB in use); HB's graph-release and per-gate memory instrument landed (`PK/raw/2026-11-13-vdac-a5-hbmem/HB_MEMORY.md`); HB-s0 then ran 10,000 updates at `--batch 4 --accum 2` with `expandable_segments:True` (A1: 3.24 s/update, INHERITED from the MM's LOOP_STATE). HC's head at 5,120 queries retained **1.231 GB** at the preflight batch against **0.463 GB** at 1,280 queries (MEASURED on the pod 2026-11-12, `refe/test_vdac_g12_bound.py:4`; those two preflights ran at `--batch 8`).
ESTIMATE (arithmetic on those figures, not a measurement): the head's retained activations scale with the micro-batch, so at 4 the 1 m head retains about 0.62 GB against 0.23 GB for the 2 m head HB trained, **+0.39 GB**; its weights, gradients and two Adam moments add 0.98 M parameters x 16 B = **+0.016 GB**; a backward transient of two to three times the retained figure puts the total at **about +0.4 to +1.2 GB over HB-s0 at batch 4**. HB-s0's own batch-4 PEAK is **UNVERIFIED here**: the `preflight_mem` rows (`aux_hb.MemGates`) sit in the pod's `preflight.jsonl` for HB-s0, and no copy of them is in the repo at the tip. **One MM action before launch: read HB-s0's batch-4 `max_allocated_bytes` per gate from the pod; HBC fits iff that peak + 1.2 GB < 44.39 GiB.** If it is within 1.2 GiB of the cap, HBC may refuse at preflight.
**Registered second fallback (NAMED, NOT BUILT, to be built only if an HBC preflight reports `preflight_mem_fail`):** `--batch 2 --accum 4` (effective 8), for both HBC arms, with the same one-line addendum rule as A5 section 9.2 (allowed only before any HBC step trains; no lever, data, seed or update count changes). Smallest fix: `batch`/`accum` of the two `VDAC_ARMS` entries, the two `vdac_pod.sh` lines, `test_vdac_hbc.py`'s `ns.batch == 4` lines, `vdac_chain.sh`'s run check (unchanged: it uses `min(upd, ...)`), then the ~69 launch_gate-reading tests and a re-land. Activation checkpointing of the copy's blocks (HB_MEMORY.md's recommendation) is NOT registered here; it would add roughly a third to the step time.

## 7. Cost, timeouts, stop rules, K ledger
| item | L40-h | basis |
|---|---|---|
| HB-s0 (the reference cost) | about 10.0 | A1 section 5(f): 3.24 s/update x 10,000 = 9.0 h of training plus preflight, mutant, gate, post-gate, export, egress; INHERITED from the MM's LOOP_STATE, not re-read from a log |
| head cost of 4x queries | +1.1 per 10,000-update arm | the A5 card's own HC estimate ("HA 3.20 + 1.1"), ESTIMATED; HC-s0's measured cost is not in yet |
| **HBC-s0** | **about 11.1** (range 10.5 to 12.5) | ESTIMATED = 10.0 + 1.1 |
| **HBC-SHUF** | **about 2.3** (range 2.0 to 2.8) | ESTIMATED = 2,000 of 10,000 updates (0.2 x 9.0 + 1.1 x 0.2 = 2.0) + about 0.3 of chain overhead; HB-SHUF's measured cost is not in the repo (UNVERIFIED) |
| one-time export | 0 | already on the pod from the A5 pause if the data root is kept (A5 section 10 charged 0.1 once) |
| **pair total** | **about 13.4** (range 12.5 to 15.3) | ESTIMATED; this is **new pod time and the PI's decision**; it is not inside the 12.8 to 16.7 L40-h approval of A5 D4, which is spent |
Pause timeouts (`vdac_pause_p7.sh`): HBC-s0 50,400 s (14 h, against ~11.1 h estimated, +26 %), HBC-SHUF 21,600 s (6 h, against ~2.3 h). HB's own were 36,000 s against ~10.0 h (+3 % over its observed cost on 7.6 estimated): these are bounds, not predictions; the first preflight's `s_per_microbatch` replaces them before freeze.
**Stop rules** (as A5 section 10): any chain refusal, a missing `ZZVDAC_CHAIN_DONE <arm>` or a timeout stops the stage's remaining arms and P7 resumes (EXIT trap); no partial read of a timed-out arm; no lever, seed, update count or bar changes after a number is seen; a killed arm restarts at update 0 under a fresh preflight and token.
**K ledger.** S0 and S1 are navtrain reads and are **not keyed**: the A5 card says so (section 10: "S0 and S1 are navtrain reads and are not keyed ... keys are written at S2 only"; `PREREG_VDAC_P6.md` section 6; A2 section 5). HBC-s0 and HBC-SHUF are S1 reads on H48 (navtrain), so **no K entry is made at freeze and none by their reads**. An S2 read of an HBC head would add one key (`VDAC-HBC-s0:<sha256 of the per-token selection scores>:E3`).

## 8. The four metric families and the tier
Tier: **T0-like perception** on banked navtrain H48 frames; never driving performance. LONGITUDINAL, LATERAL, TACTICAL: not computable at S1 (a mask head; n = 0); STRATEGIC: **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25), not a pass, not omitted. The perception line reports IoU, near-boundary SDF MAE (HB-s0 1.436 m, HA-s0 1.630 m: the quantity the boundary weight targets), per city and per band, S1-P1 / S1-Q / S1-F, the paired deltas. The four families apply at S2. Information-disjointness is the A5 card's section 12 (vision only at inference; G19 through the copy).

## 9. HC-s0 row (FILLED 2026-11-14 by the MM from the HC read and the HC-s0 chain logs)
**Filled.** HC-s0 was read under the A5 card with `manifest_a5_hc.json` (filled manifest sha256 `cb147151…`, filled before any array was opened), using the frozen reader (`s1_read_v2.py` `796aea21…`).
Sources:
- `PK/raw/2026-11-09-scorer1p6-prereg/vdac/a5_read/hc_read/READ_HC.md` and its `out/s1_read_v2.json`, `out/hc_s1f_subset.json` (landed `33900a9f`);
- for the cost and gate rows only, the chain logs banked on HF `vdac/hc-s0/` (`chain_hc-s0.log`, `sft.log`, `preflight.log`, `postgate.log`, `postgate_a2.log`).

Evidence class MEASURED; single head seed; log-cluster bootstrap (n_boot 10,000, seed 20260927), 3,207 H48 frames over 45 log groups.

| field | value |
|---|---|
| HC-s0 S1-P1 (2 m reference grid) [CI] | **0.7757** [0.7528, 0.7932]: fails 0.80 |
| HC-s0 S1-Q (1 m mask) [CI] | **0.8346** [0.7919, 0.8751]: fails 0.90 |
| HC-s0 S1-F lower bounds (on 1,705 frames? y/n) | P1 0.485, Q 0.164: both > 0, holds. **y**: the SFT-held-out subset (1,705 frames, 24 groups, the PREREG 7.5 fallback). The P1 term had 3,787 of 10,000 resamples undefined, which the wrapper reports. |
| C-SHUF (HC-SHUF) / C-FLIP | 0.4447 (< 0.50, valid) / 0.7064 (< 0.80, valid) |
| paired HC-s0 minus HA-s0: P1, Q, X-LIFT | P1 **−0.0045** [−0.0121, −0.0008] (separated, WRONG direction); Q +0.0013 [−0.0086, +0.0114] (not separated); X-LIFT **False** |
| near-boundary SDF MAE | **1.241 m** overall (HA-s0 1.630, HB-s0 1.436). IoU per band 0–20 / 20–40 / 40–72 m: 0.811 / 0.740 / 0.649. READ_HC.md does not publish a per-band SDF MAE, so it is not quoted here. |
| A5 row HC-s0 read | **X-NULL** |
| HC-s0 measured cost | 10,000 updates, `s_per_update` **1.0630** (sft.log, last train event; batch 8 × accum 1, so this is also the per-microbatch figure). Train wall 17:59:27 → 21:10:06Z = **3.18 h**. Chain arm 17:30:07Z → chain end 21:10:08Z = 3.67 h (before the A2 completion). Preflight `gpu_mem_gb` 6.29 (preflight.log). |
| post-training gate result (the `--n-params` defect of section 3) | The chain gate FAILED exactly as predicted: `ZZVDAC_POSTGATE_FAIL the Adam state covers 3423490 head parameters, the card's built head has 2440450` (21:10:08Z). It was completed WITHOUT retraining under amendment A2 (`422541f5…`, landed `45c5331f`): `postgate_a2` `ZZVDAC_POSTGATE_OK` at 3,423,490 head parameters, 10,000 steps. HC-SHUF then ran on the FIXED chain (`a469d4a5`) and its own post-gate passed at 3,423,490. |

How the filled row enters, **committed now so it cannot be chosen after the data**:
1. If HC-s0 is X-GO, the S1b of HC-s0 takes priority over HBC in the pod queue, and HBC stays a named successor (the A5 branch read: "only that arm continues").
2. If HC-s0 is X-LIFT or X-BOUND, HBC runs as drafted.
3. If HC-s0 is X-NULL (no separated lift on either delta), the head-precision lever has no measured effect alone. HBC then tests the copy plus an inert lever, and the MM decides with the PI whether the pod time is better spent on HB at 20,000 updates (A5 X-LIFT row b) or on the agent check. This draft does not pre-decide that.
4. If HC-s0 is NOT RUN or INVALID, HBC's section 4 reference to HC-s0 is dropped, and the contrast list reads HB-s0 and HA-s0 only, as stated in the verdict row.

**Branch taken: (3).** HC-s0 is X-NULL. The decision therefore goes to the PI.

**The MM's recommendation (a recommendation, not a ruling; the PI decides the pod time):** the agent check first, with the DAC pod line (HBC, or HB at 20,000 updates) held. Three MEASURED facts support it:
- **S1-Q is flat across all three levers:** HA-s0 0.8332, HB-s0 0.8376, HC-s0 0.8346. No paired S1-Q delta is separated, and the true-field ceiling of the same metric is 0.9408. HBC combines one lever that did not move S1-Q with another that did not move S1-Q.
- **On P6's navtrain held-out fit pack (1,705 sets, TRUE sub-scores, P6's own ranking), the oracle agent veto is worth +1.666 PDMS (NC+TTC) and the oracle DAC veto +0.636** (`PK/raw/2026-11-14-agent-check-feas/ceiling_fitpack.json`, re-computed by the MM with the canonical `sel_common.agg_torch` pick, `MM_VERIFY.md`). The agent term is the larger lever by 2.6×.
- The agent check's first gate (S0-B, `PREREG_AGENT_CHECK_DRAFT.md` section 3) needs **no GPU and no pod time**, and is being built now.

If the PI prefers to keep the DAC line moving in parallel, HBC as drafted (pair total about 13.4 L40-h, range 12.5 to 15.3, ESTIMATED, section 7) is the committed arm. The prior that it clears S1-Q is low, for the reason above, and that is disclosed here before any HBC data exists.

## 10. Freeze checklist (for the MM; nothing here is done)
1. ~~Fill section 9~~ DONE 2026-11-14 (branch (3), PI decision); re-read sections 5 and 7 against it. 2. Pin the code by git blob (the A5 card's section 3 form) for every file in the hold's MANIFEST: `refe/launch_gate.py`, `refe/vdac_pod.sh`, `refe/vdac_chain.sh`, `refe/vdac_pause_p7.sh`, `refe/test_vdac_gate.py`, `refe/test_vdac_hbc.py`. 3. Run the tests of the MANIFEST on a clean export of the landed tree and the ~69 launch_gate-reading tests under `stack/tests` and `refe/` (a launch_gate edit moves profile fingerprints pinned elsewhere; the build agent ran the `refe/` set and could NOT run `test_launch_gate.py` past its data dependency, see the MANIFEST). 4. Ship the tree as a FRESH `/workspace/refe-plan-vdac` tree (A5 section 9.6) and the card's bytes into the data root (A5 section 9.7: the chain verifies the P6 card, A1 and A2 only). 5. Read HB-s0's batch-4 `preflight_mem` peak (section 6). 6. The PI's pod-time decision (section 7). 7. The reader manifest `manifest_a5_hbc.json` with the three ordered pairs, filled by `fill_manifest_sha.py` before any array is opened.
