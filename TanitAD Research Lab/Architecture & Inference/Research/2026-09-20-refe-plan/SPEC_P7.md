# P7: the next trunk = P6's recipe + three changes (design spec, DRAFT)

> Written 2026-10-22, **design only**: no GPU, no pod, no network, no file of the repo edited. Plan step S3 (`PK/PLAN_BEYOND_PAPER.md`). Compute is the PI's call when P6 reads.
> Evidence classes: **MEASURED** (ours, artifact named) · **PUBLISHED** · **INHERITED** (not re-verified) · **ESTIMATED** · **HYPOTHESIS** · **UNVERIFIED**.
> `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`; code anchors are the mirror tip `f32ed6b` and move. Package: `PK/raw/2026-10-22-p7-session-reserve/` (`SESSION_RESERVE.md`, `code/`, `raw/` below resolve there). **MM rulings on §6: `PK/raw/2026-10-22-p7-session-reserve/MM_RULINGS.md` (R-A kept; R-B not used).** The reserve: `SESSION_RESERVE.md` (R-A: 16 sessions, 10,141 scenes).

## 0 · Headline

| | |
|---|---|
| **What** | P6 unchanged (`refe_p6`'s argv: `--scorer-sees-trunk --yaw-wrap off --nav-mode command --labeller online --label-chunks 2 --epochs 25 --batch 4 --accum 64 --lr 2e-4 --weight-decay 0.01 --amp bf16 --score-w 0.1`, seed 0) **plus** (1) the P7.1 aux head, (2) the session reserve out of every training input, (3) the bank on HF before step 0. |
| **Cost** | **≈ 6.2–6.3 A40-days (148–151 h)** central, **≤ 7.5 d** worst case; 0 extra GPU for changes 2 and 3 (the reserve *saves* 1,000 steps, 15.8 h). Pre-run: 0.1–0.2 d on P6's pod, 1–2 d GPU-idle on a fresh one. |
| **Effect vs P6** | HYPOTHESIS **−0.25 to +1.5 PDMS**. Only the head can add; the reserve costs a little. **P7 alone reaching 94.55 is LOW (HYPOTHESIS).** Its jobs: the DAC lever, and unseen-session data for S4. |
| **Not attributable** | P7 − P6 bundles head + 9.84 % fewer scenes. A head lever claim needs a head-off twin (§4, Q1). |
| **Wiring state** | The head module and a label store exist (the store's disk location is UNVERIFIED here); **`train.py` has no head wiring** (it has the L_field loader tail the head can reuse). No P7 gate entry, profile or exclusion code exists yet. |

## 1 · Change 1: P7.1, the training-only drivable-area head (design: `PK/SPEC_AUX_DRIVABLE_HEAD.md`, not redesigned)

**Settled by S1 and the scene-tap probe (MEASURED):** tap = **`visual_ctx`** (the scene tap was ELIMINATED, `raw/2026-10-15-scene-tap-probe/`: 0.609 vs 0.637 held-out IoU); **feed off** (H2 NOT PROMOTED, `raw/2026-10-15-s1-read/`); 2 × `CrossBlock`, 2,440,450 params; `w_aux = 0.1`, set once by the gradient-norm band 0.05–0.5 at G-LIVE, never tuned on PDMS. The frozen-trunk head reached navtest IoU 0.49–0.52 against K2's 0.80, so S1 never tested the **strong version**; P7 is that test (the head acts on the LoRA from step 0).

| code | `refe/train.py` (tip): build the head as a **sibling** with its own `torch.Generator` after `REFe(cfg)`; hook `scene_proj`'s 2nd call for `visual_ctx`; `l_aux` into **both** backward sites (online deferred `(tot_ / a.accum).backward()` ≈ l.1754, plain `loss.backward()` ≈ l.1769: editing one leaves the other arm head-less, the "V2" failure); append the dests to `IDENTITY_ARGS` **and** `IDENTITY_DEFAULTS` (≈ l.996–1006); `aux_drv` key in every checkpoint dict; labels via TargetBank's existing `(sdf, valid)` tail (≈ l.635–653, 906, built for `--field-labels`). Steps: `PK/raw/2026-10-13-aux-head/WIRING_PLAN.md` §3. Not in `model_final.pt`: the planner is unchanged. |
|---|---|
| **flags** | `--aux-drv --w-aux-drv 0.1 --aux-drv-tap visual --aux-drv-depth 2 --aux-drv-feed off --aux-drv-labels <P7 training store>`; `--w-field 0.0` named (the field loss is its own arm, D-AUX-4); `--aux-drv-shuffle-seed` forbidden (control arms only). **`--w-aux-drv` defaults to 0.0, so the argv must name it** (default-disables class). |

**G-DVB, one entry per new flag** (profile `refe_p7` cloned from `refe_p6` in `refe/launch_gate.py`; dests added to `DVB_KINDS` in `refe/gate_checks.py`, whose pin to the parser moves with them; the scorer-host table `DVB_KINDS_SCORER` already holds all seven, plus the scorer-only `aux_drv_lr`):

| flag | declared = built | mutation that must FAIL |
|---|---|---|
| `aux_drv` | head built iff flag | flag on, head not built |
| `w_aux_drv` 0.1 | the multiplier in the loss at **both** sites | delete the term at one site; `--w-aux-drv` left at its 0.0 default |
| `aux_drv_tap` | `data_ptr` equals `visual_ctx` | tap `scene_ctx` |
| `aux_drv_depth` | 2 blocks built | build 1 |
| `aux_drv_feed` off | scorer input width 60 and **no autograd path head → score** (`allow_unused` all `None`) | wire the feed with the flag off |
| `aux_drv_labels` (data) | store sha256 and key count == the P7 pins (§2); coverage ≥ 99.9 % reported | the full store, reserve keys present |
| `w_field` 0.0 | `l_field` contributes exactly 0.0 | `--w-field 0.1` |

**Other gates (listed, not implemented):** G-HYG (an `aux_drv_feeed` typo raises); G-LIVE: `aux_bce`, `aux_huber` finite, head gradient non-zero, aux-only backward **non-zero on LoRA / `scene_proj` / `pos3d_mlp` and exactly 0.0 on `traj_head` / `reg_compress`**, `w_aux = 0` ⇒ every gradient exactly 0.0, initial `aux_bce` ≈ ln 2 ± 0.15, label alignment ≥ 95 % on ≥ 64 frames (flip-y and off-by-one mutations FAIL); G-LIVE-MEM (head ≤ 1 GB; P6's L1 bound is 44 GB, so micro-batch 2 is the fallback); G-EVAL (bit-identical planner output without the head weights); G-CKPT (head + Adam state round-trip); G-CLOCK (label key = image key); detached-control bit-identity; the **post-training zero-gradient gate** (`probe_zerograd3.py` pattern: every head parameter has Adam state).

**Cost (A40):** step time **+0.8–1.6 %** (ESTIMATED, FLOP share ×2 for attention) up to **+2.1–2.4 %** (MEASURED on S1's scorer-only host, with +0.51 GB; a different host) ⇒ **+1.2–3.5 h ≈ +0.05–0.15 d**. Engineering: wiring + gate entries, ≈ 2 agent-days (ESTIMATED), 0 GPU.
**Effect:** ceiling **DAC oracle +5.23 PDMS [+4.23, +6.33]**, MEASURED on an *earlier* scorer (pick 83.24, `PK/REVIEW_7_GAP_TO_PAPER.md` §2.4): the value of *perfect* DAC knowledge, an upper bound, not the head's effect. P2's DAC deficit to Table A13 is 99.25 − 96.87 = 2.38 points, so a P2-scale DAC lever is **≤ ≈ +2.4 PDMS (ESTIMATED)**. Expected **+0.0 to +1.5 PDMS, DAC +0.5 to +1.5 points (HYPOTHESIS)**: the aux spec's 30–50 % rescue of u400's DAC zeros is +1.0 to +1.6 (ESTIMATED, a different model); precedent is split (NAVSIM Table 2 about −2.3 when removed; ReflectDrive-2 +2.4 for a *field loss*; Hydra-MDP++ −0.5; all PUBLISHED, one run each).

## 2 · Change 2: the session reserve out of every training input (`SESSION_RESERVE.md`)

| code | **new** `refe/filter_reserve.py` (CPU, ≈ 1–2 CPU-h ESTIMATED): from `reserve_sessions.json` write `train_grow_cmd_3c_p7/` (rank-0 and rank-1 rows without reserve logs, a manifest with counts, `targets_sha256` and the reserve's `logs_sha256`); **move** (not copy) the reserve's D9 metric-cache dirs, image zips and calibration rows to `*_reserve/` roots; re-finalise the label store twice (`aux_label_store.py finalize`: a **training store** without reserve logs, a **reserve store** only for evaluation); attribute the 103 resolvable logless tokens by session (5 fall in R-A; the other 148 stay UNVERIFIED). **New** trainer flag `--reserve-manifest`: `TargetBank` and the label source **refuse** a row, key, cache dir or zip in the reserve (the refusal string must exist in `train.py`, not only in its test). **New** `refe/p7_pod.sh` (from `p6_pod.sh`). |
|---|---|
| **registered before step 0** | scenes 92,896; **steps 9,075** = 25 × ⌈92,896/256⌉; `expect_schedule` {epochs 25, steps 9,075, epoch_len 92,896}; a `train_grow_cmd_3c_p7` entry in `P6_BANK_TUPLES`' successor (92,896 rank-0 + the filter's rank-1 count, ≈ 150.8k ESTIMATED; a bank with no entry FAILS G-CLOCK); the P7 store's sha256 and key count (≈ 92,99x ESTIMATED = 103,139 − 10,141 − logless). |
| **gates** | G-DVB `reserve_manifest` (data kind; file sha256 pinned): declared `logs_sha256` == the bank manifest's stamp == the digest the trainer recomputes. **G-RESERVE (new, in G-LIVE):** zero overlap of the reserve with all **five** inputs (bank rank-0, rank-1, metric cache, label store, image/calibration roots) and with navtest (sessions, logs, tokens); train ∪ reserve = navtrain's 1,192 logs; every key has a resolved log. Mutations that must FAIL: one reserve row in rank-0, one in rank-1, one label key, one cache dir, one zip, a wrong manifest sha. The draw's own suite (`ZZRESERVE_PASS 17`) joins G-SUITE. |
| **cost** | **−1,000 steps = −15.8 h ≈ −0.66 d** vs P6 (saving). CPU/disk only otherwise. |
| **effect** | −9.84 % scenes: **−0.06 to −0.25 PDMS (ESTIMATED**, by scaling SPEC_P6's −0.3 to −1.2 for 48 % fewer steps; weak basis, never measured for this recipe). Its value is S4/S5 unseen-session data, not PDMS. |

## 3 · Change 3: the bank on HF before training

| code | `refe/p7_bank_hf.sh` (pattern: `PK/raw/2026-10-22-p6-newpod-rebuild/code/bank3c_hf.sh`): pack `train_grow_cmd_3c_p7` + manifest, `reserve_sessions.json`, both label stores (≈ 0.35 GB), `filter_report.json` to the private dataset `Sayood/tanitad-refe-navtrain-banks`; token read in place, `HF_HUB_DISABLE_XET=1`. `p7_pod.sh` **refuses to start the trainer without the receipt**. |
|---|---|
| **gate** | G-EGR bank leg: verify by the **remote listing** (size and LFS sha256), never the push status; then a **restore test**: download into an empty directory, untar, recompute `targets_sha256` == registered. Receipt = a `data_flags` entry, so the token fingerprints it. Mutations: wrong expected size FAILS; truncated archive FAILS the restore. |
| **cost / effect** | 0 GPU; network only (the 3-C bank's size is UNVERIFIED; the older bank was 8.10 GB INHERITED), under 1 h each way at tens of MB/s. No PDMS effect. It closes the single-disk risk that cost P6 **≈ 23 h of CPU and an idle A40** (`p6-newpod-rebuild/RESULT.md` §1, 4). **Open: also bank the D9 metric cache?** (size UNVERIFIED; quota 199.4 of 1,000 GB used, INHERITED). |

## 4 · Cost and attribution

| item | A40-h | class |
|---|---|---|
| P6 step 57.0 s (epochs 0–11 of the final run) | | MEASURED (`SPEC_P6` §0.2); **re-derive from P6's resume rate on the new pod** |
| P7: 9,075 × 57.0 s | 143.7 | arithmetic |
| + checkpoints/egress 2 % + labeller stall 0.41 % (6 workers) | +3.5 | ESTIMATED / MEASURED (`launch_gate.py` comment) |
| + head +0.8 to +2.4 % | +1.2 to +3.5 | §1 |
| **central** | **148–151 = 6.2–6.3 d** | |
| worst case (micro-batch 2: P6's 8.2 d × 0.9007, + head) | ≈ 180 = 7.5 d | ESTIMATED |

**Attribution (Q1).** P7 − P6 = head + 9.84 % fewer scenes + noise (floor F = 0.33 PDMS, one draw, `RESULT_P2REP`). Options: (a) accept the bundle, and claim only "P7 vs P6" (the aux spec's option (a): PI chose (b) for the head in P6); (b) a head-off twin, same bank, `--w-aux-drv 0.0`, **+6.2 A40-days** (the only clean head attribution); (c) the S2 graft pair (~40 A40-h, weaker). **Recommendation: (a) now, (c) in parallel; (b) only if P7 reads ADOPT and the head is the claim.**

## 5 · Draft pre-registration outline (mirrors `PK/eval/PREREG_P6.md`; to be finalised and landed before step 0)

* **Arm.** The §0 argv + §1–§3 flags, under one launch-gate PASS token bound to commit, argv sha, `reserve_sessions.json` sha256 (**R-A `53f3bcd4…`**, or R-B once ruled), `logs_sha256`, bank and store pins.
* **Endpoint.** Full navtest PDMS, 12,146 tokens / 136 logs, NAVSIM v1.1, `model_final.pt`, T1 (open-loop plan vs logged agents), **one look**. Inference determinism asserted (a second eval bit-identical).
* **Comparators** (paired, same tokens): **C1 = P6** (level filled when it reads); **C2 = P2** 88.82 [87.86, 89.77] (seed 1: 89.15). Goal **94.55** (DriveZero A13, PUBLISHED); the harness gap ≤ ~0.25 is reported, never subtracted.
* **Estimator.** Paired log-cluster bootstrap, 10,000 resamples, seed 20260927 (answers "another draw of logs" only).
* **Rule.** ADOPT iff the lower bound of Δ vs P6 > 0 **and** no longitudinal or lateral family component is adverse; **lever claim** only if that lower bound > F = 0.33; REFUTED iff the upper bound < 0; else NOT PROVEN. GOAL MET iff P7 ≥ 94.55 **and** ADOPT; GOAL NOT EXCLUDED iff the point < 94.55 ≤ the level's upper bound; else BELOW GOAL. Δ vs P6 is a **bundle claim** (§4).
* **Mechanism endpoints (reported, not gating):** DAC sub-score and DAC-zero share vs P6; head IoU [0, 40) m on navtest and on the reserve (bar 0.80, K2); head-vs-scorer DAC AUC.
* **Reserve reads (fit/measure, never gating):** P7 on the reserve vs P7 on a seeded same-size subset of its training sessions (session-grouped bootstrap); P6 on the reserve is a **seen** reference only; all S4 fits by session-grouped 4-fold.
* **Four families** (per family, never pooled; same instrument as `sft4_B_u400_families.json`): LONGITUDINAL (speed, along-track, EP, TTC, headway), LATERAL (heading, cross-track, curvature, yaw rate, **DAC/DDC**), TACTICAL (decision κ, command adherence, random/actual/best-of-64), **STRATEGIC PAUSED** (PI 2026-09-19, item 25: not a pass, not omitted).
* **Replicate.** Floor F = 0.33. A P7 seed-1 replicate (+6.2 A40-d) only if ADOPT and Δ < 2F; else "separated, single seed, not yet a lever claim".
* **Kills.** K6: `aux_bce` ≥ 0.9 × its step-0 value after 10 % of steps ⇒ stop (≤ 0.6 d spent); K7: DAC-zero share falls but EP's upper bound < −1.0 and Δ ≤ 0 ⇒ route to selection calibration; K3: step time > +3 % or memory > +1 GB at G-LIVE-MEM ⇒ 1 block / micro-batch 2; **R1: any reserve overlap, at the gate or found later, voids the reserve reads.**
* **Next arm (Rule Zero).** GOAL MET: replicate, then P7.2. ADOPT below goal: S4 on the reserve; EP gap largest ⇒ selection/progress calibration; DAC/NC/TTC largest ⇒ P7.2 ensemble + pessimism. NOT PROVEN: read the DAC-zero share and head IoU; head IoU < 0.80 even trunk-trained ⇒ the 1 m grid or the field-loss arm (`--w-field`, D-AUX-4). REFUTED: bisect P7-head-off, `w_aux` ÷ 4, field-loss-only.

## 6 · Questions for the MM / PI, and conflicts found

1. **Reserve: R-A or R-B** (`SESSION_RESERVE.md` §5): R-A's 16 sessions all keep a same-vehicle same-day sibling in training; navtest has none.
2. **Attribution (§4).** 3. **Reserve power:** 16 sessions vs Stage 0's 21 (NULL-UNDERPOWERED); 24–32? 4. **Held-out exclusion** (my rule; the pool shrinks to 109 sessions). 5. **Metric cache on HF?** 6. **Seed replicate budget (+6.2 d).** 7. **Authorisation** of ≈ 2 agent-days of wiring now (0 GPU), so P7 is launch-ready when P6 reads.
* **Conflicts (reported, not fixed):** (i) `SPEC_P6` §0.2 / D8 give a **189,888**-tuple bank; the pod's 3-C bank is **167,300** (rank-0 103,037 + rank-1 64,263; `launch_gate.P6_BANK_TUPLES` holds both). (ii) `SPEC_AUX_DRIVABLE_HEAD` §5, §7 say `launch_gate.py` is not on the tip and `REFE_LAUNCH_GATE.md` does not exist; `refe/launch_gate.py` with a `refe_p6` profile **is** at `f32ed6b`. (iii) the aux spec's `train.py` anchors (l.1588 etc.) are stale: the tip has the online deferred backward and the L_field wiring.

## Deliverable manifest

`SPEC_P7.md`, `SESSION_RESERVE.md`, `code/`, `raw/` in `C:/Users/Admin/par18/p7spec/` only (nothing staged, committed or pushed; the MM lands with the scoped lander). Suggested: `PK/SPEC_P7.md`, `PK/raw/2026-10-22-p7-session-reserve/`.
