# NT-DECIDE deliverables 2 and 3: a minimal design that isolates DATA, and what it costs

Research Lab (Architecture & Inference), 2026-10-21, Berlin. Nothing here has run. This file reads with `FEASIBILITY_AUDIT.md` (the `file:line` citations are there) and `PREREG_NT_DECIDE.md` (the committed rules).

## 1. The question, restated so that one arm can answer it

REFe reaches 88.82 / 89.15 navtest PDMS; θ_DEC is about 72.8.
* θ_DEC's navtest score is not read. θ_0.5 reads 72.78, and θ_DEC ties θ_0.5 on navtrain (+0.03, registry §16.11).

REFe differs from refc in **five** things at once, not three:

| # | factor | REFe | refc (θ_DEC) |
|---|---|---|---|
| 1 | training data | navtrain | PhysicalAI only |
| 2 | cameras | F0 + L0 + R0 + B0, 512×960 each | one 120° virtual stitch of L0/F0/R0 at 416×1024 (0.63× REFe's angular resolution) |
| 3 | backbone | frozen DINOv3 ViT-L + LoRA | R101 |
| 4 | **selection method** | 64 proposals plus a **learned PDM scorer** (teacher-labelled, on-policy; SFT-4 / P2) | log-imitation critics |
| 5 | **navigation input** | a route-derived **goal point** | the NavSim `driving_command` only |

Factors 4 and 5 are MEASURED on REFe itself (registry §14.2):
* model_final 83.24
* plus the route fix: 84.46
* plus the SFT-4 scorer: **88.25 (+3.79 [+3.05, +4.54])**
* plus P2: 88.82 / 89.15

So, of REFe's ~89 points:
* **about 4.5 come from the learned scorer** (84.46 → 88.99, the P2 mean);
* another **1.2 come from the route fix** (83.24 → 84.46).

Neither is data, cameras or backbone. Every REFe row also carries the route-derived goal-point input that refc does not have.

The fine-tune changes factor 1 only. Its natural ceiling is therefore REFe's planner **without** the learned scorer, **84.46**, not 89. That comparator is still optimistic for refc, because it keeps REFe's goal input.

**PUBLISHED prior** (banked `2406.15349`, NAVSIM Tables 1–3):
* **LTF**, camera-only, three front cameras stitched into 1024×256, ResNet-34, trained on navtrain: **83.8 navtest PDMS**, and **83.5 ± 0.6** over 3 training seeds.
* TransFuser's camera-FOV ablation: 60° → 80.3; 160° → 82.8; 240° → 84.1.

A front-wide ResNet trained on navtrain therefore lands at about 84 (ESTIMATED transfer to refc).

⇒ **The most likely outcome of this experiment is the middle band, near its top.** That is why the design is built to *name the split*, not only to pass or fail a line.

## 2. Arms

| arm | what | trainable | role |
|---|---|---|---|
| **A0** | θ_DEC (`ckpt_wiseft_mod2_DEC.pt`, md5 `dad4cb5b…`), no training | — | baseline and control (K0). Needs its **first navtest read**. |
| **A1** ⭐ PRIMARY | θ_DEC fine-tuned on navtrain: front 120° ST stitch, **refc architecture and every refcv8 flag unchanged** except the NavSim data inputs in §3 | **everything** (trunk at 0.5× lr, as in the recipe) | the DATA arm |
| A1r | A1 with `--seed 1` | everything | **replicate** (H-ESTIM-SEED-1). Mandatory inside the ±1.0 boundary band, recommended otherwise. |
| A2 (optional) | A1 with a **~170° cylindrical stitch** of the same L0/F0/R0 into the same 416×1024 tensor (`f_ref` ≈ 345) | everything | CAMERA proxy. ⚠️ A declared geometry change, not an architecture change. It halves resolution against REFe and has no rear camera. Run only if A1 lands in the middle band. |
| E-FOV (optional, 0 training) | REFe P2-s0 at inference with B0 blanked and L0/R0 masked outside ±60° | — | how much REFe *uses* the views refc lacks. Off-distribution input makes the drop an **upper bound** on the camera share. |

**Why A1 trains everything, not only the heads.** The heads-only option (H: decoder + tactical stack trainable, trunk/BEV/agent frozen) is cheaper per step but answers only half the question:
* a **pass** under H says data is the gap;
* a **fail** under H is confounded, because the PhysicalAI-shaped trunk was never allowed to adapt to nuPlan imagery. It could not support the "sensors/backbone ⇒ adopt REFe's backbone" verdict.

Only the full arm can produce **either** verdict admissibly. REFe also adapted its backbone (LoRA), so A1 is the like-for-like data arm.

⚠️ P-GRAD reads the trajectory loss at **0.016 %** of the trunk update. With trajectory losses alone the trunk would barely move. A1 therefore keeps the **agent + box3d losses**, now on OpenScene's 3D boxes, which are the losses that actually train the trunk. The trunk's relative weight drift is logged as a check that the trunk did adapt.

## 3. A1's inputs and losses (the only departures from the refcv8 argv)

| item | A1 | why |
|---|---|---|
| frames | ST: every one of the 10 slots ← the t0 stitch, `[8, 9, 416, 1024]` u8, stored as **PNG** v2ep (lossless; round trip 40/40 bit-exact, MEASURED) | equals the R8_LEGAL inference feed bit for bit (K1) |
| nav token | `NAVSIM_LEGAL_NAV_CMD[driving_command at t0]` (`refcv8_bridge.py:163`) | the inference function; replaces the future-derived v9 token |
| nav args | v9 builder args from the GT future, recipe dropout 0.5; unknown at inference | unchanged recipe asymmetry, as on PhysicalAI |
| speed ceiling | the **UNKNOWN row always** | = the LEGAL inference feed; declared departure from p_unknown 0.45 |
| trajectory target | `future_poses_ext` `[60, 4]` at 10 Hz: a C2 spline through the 2 Hz log poses (exact at 0.5 s multiples), masked past the log end | NavSim logs are 2 Hz |
| tactical lat/lon + v9 constraints | the v7 / v9 geometric builders on the 2 Hz track (gap 0.5 s ≤ `g_max_s` 1.0) | VLM-CoT goals → IGNORE |
| agent + box3d | from OpenScene `anns`: vehicle → automobile, pedestrian → person, bicycle → rider, the rest ignored | truck/bus/trailer get no positives (declared) |
| vis1 sidecar | rebuilt for the stitch: azimuth within ±60°, depth > 0, pixel observed (`src >= 0`) | |
| map head | `w_map_hires 0`; an all-IGNORE GT store (or a w = 0 guard relaxation); head grads exactly 0 (K4) | no SAM3 GT; `map_hires_pool` stays the planner's BEV |
| rig / lift | per-token `FrontWideExtrinsics` converted from `frames416.rig_record` + road plane | K1 lift parity |
| LAW target | the t0 + 0.5 s stitch (the next token's current frame); fallback to t0, counted | |
| dropped | `--pose-sync-sidecar`, `--clip-clock-sidecar`, `--join-defect-masks` | PhysicalAI timing/defect artifacts |
| kept | anchors (117, alat), every model flag, `--init-from` θ_DEC, lr 1e-4 (trunk 0.5×), warmup 1000, `--r8-alloc-emit-start 1000`, EMA 0.9998, batch 16, `--opt dd`, PhysicalAI eval139 in-run eval as a **forgetting monitor** | architecture and recipe unchanged |
| held out | all **38** selection-scan logs (3,464 tokens) and every navtest log | training pool **99,824 tokens / 1,154 logs** (MEASURED arithmetic) |

## 4. Step budget, snapshots, selection

* **2 epochs = 12,478 steps** at batch 16 (99,824 / 16 = 6,239 steps per epoch).
* EMA and RAW snapshots at **1,560 / 3,120 / 6,239 / 9,359 / 12,478** (0.25, 0.5, 1, 1.5 and 2 epochs).
* **No reads while training.** The box never evals while it trains, so all ten snapshots are read afterwards on the registered sample (2,089 tokens / 30 logs, sha `9dd68698…`), R8_LEGAL, inference seeds 0 and 1.
* **Selection:** the EMA snapshot with the highest mean PDMS over the two seeds. RAW is reported beside (the PI's EMA rule, 2026-10-16).
* **Plateau rule:** a non-DATA verdict is admissible only if EMA(2.0 ep) − EMA(1.0 ep) ≤ max(2 × F_inf, 1.0) PDMS. Otherwise the verdict is **UNDERTRAINED**, and the next lever (extend to 4 epochs) is a PI compute item.
* **One navtest read** of the chosen snapshot and of A0, both **only after the PI's go**.

## 5. Engineering before any GPU (0 GPU; ESTIMATED 3–4 agent-days)

| # | work | ESTIMATED |
|---|---|---|
| E1 | token-episode v2ep writer: streaming shard → L0/F0/R0 extract → `frames416` stitch (imported, never copied) → PNG v2ep with interpolated poses/actions and the LAW frame; manifest + per-token sha | 0.5–1 d |
| E2 | label builders: tactical lat/lon (v7 geometry), v9 release npz (new md5), nav token from `driving_command`, unknown speed row | 0.5–1 d |
| E3 | OpenScene `anns` → `--agent-join` / `--join3d` / vis1 sidecar with the class map; per-token rig table + road plane | 1 d |
| E4 | held-out guard (selection logs + navtest logs); argv; launch-gate profile `refcv8-nt` (G-HYG … G-SUITE-PINNED); map-off store | 0.5 d |
| E5 | controls K0–K7 and the mutation tests (`PREREG_NT_DECIDE.md` §6) | 0.5–1 d |

## 6. Costing

**Bases:**
* MEASURED refcv8 on Thor at batch 16, 416×1024:
  * G-LIVE smoke **8.94 s/step**, peak 31.26 GB (registry §16);
  * realised whole-run pace **9.83 s/step**, i.e. 133.43 h for 48,852 steps (launch 2026-10-15 04:54 → finish 2026-10-20 ~18:20, registry §16 / §16.1). The realised pace includes evals, grad-share reads and saves.
* MEASURED Thor NavSim bridge: **0.193 s per forward** including I/O (806.1 s for 2 × 2,089 tokens, `RUN_LEDGER.json`).
* ⚠️ The step time is the **planning basis and probably an upper bound**. Under ST, the inference forward recorded **1 trunk frame computed for 24 slots** (`last_dedup (24, 1)` on 30/30 smoke rows, MEASURED).
  * That is not what a plain reading of `timm_trunk._backbone_dedup` predicts (at least 3 frames), so the mechanism is UNVERIFIED.
  * Whether training dedups the same way is unmeasured.
  * The launch-gate G-LIVE smoke measures the real NavSim step time before the run is committed.

### Thor (GPU exclusive; ESTIMATED from the MEASURED rates above)

| item | GPU-h | CPU / wall | basis |
|---|---|---|---|
| data: download 31 `navtrain_current` shards, 295.1 GB | — | **6.4 h** at 12.8 MB/s (MEASURED S1); 2.6 h at 31.2 MB/s (INHERITED) | `RUN_LEDGER.json` |
| data: extract 304.5 GB | — | ~0.3 h at 248 MB/s | MEASURED (`BUILD_REPORT.json` wall − stitch) |
| data: stitch about 99.8k frames | — | **3.0 h** at 9.15 frames/s (6 threads) | MEASURED (`BUILD_REPORT.json`) |
| data: PNG encode about 99.8k frames | — | ~0.5 h at 6 threads | MEASURED on the dev box: 0.115 s/frame/thread (`PNG_PROBE.json`); Thor's aarch64 rate is unmeasured |
| data: labels, joins, rig table | — | ~1–2 h | ESTIMATED |
| data prep wall, pipelined (download-bound) | — | **≈ 6.5–8 h** | ESTIMATED |
| launch-gate smokes (G-LIVE, G-CKPT, …) | ~1 | | ESTIMATED |
| **A1 training, 2 epochs, 12,478 steps** | **31.0–34.1** | | 8.94–9.83 s/step × 12,478 |
| K0 (θ_DEC re-read) + 10 snapshot reads | 2.5 | scoring ~0.7 h CPU | 13.4 min per checkpoint (MEASURED) |
| navtest: A0 + A1, each s0 + s1 + BLIND (2 × 36,438 forwards) | **3.9** | scoring ~0.9 h CPU (8.4 min/arm, INHERITED) | 0.193 s/forward |
| **A1 total** | **≈ 38–42 GPU-h ≈ 1.6–1.75 Thor-days** | | |
| A1r replicate (conditional) | +34–37 | | same |
| A2 FOV-170 (optional) | +36–39 | +3.0 h stitch in the same streaming pass, +~50 GB | same |

Disk: the shards stream (at most 2 × 9.5 GB in flight). The PNG bank is **53.8 GB** (538,884 B per frame MEASURED × 99,824; raw u8 would be 128 GB). Labels are under 5 GB (ESTIMATED). The peak is ≲ 80 GB of the 169 GB free (MEASURED).

### A40 (new pod)

* **No same-job refc step time on an A40 exists.**
* Planning number = Thor's pace × r, with r UNMEASURED and bracketed 0.6–1.6 (ESTIMATED, not derived from a measurement). For A1 that gives **20–55 A40-h; 34 h at r = 1**.
* The old pod refuses ssh and had no volume, so a new pod needs PI spend plus the same data preparation on the pod (305 GB pull ~1.2 h INHERITED pod rate; stitch ~3 h on a ~7.65-CPU quota).
* Memory fits (48 GB vs 31.26 GB measured).
* ⛔ The first step on an A40 is a **200-step timing smoke (≈ 0.5 A40-h)** that replaces the bracket with a number before the run is committed.

### Wall-clock to a verdict on Thor (ESTIMATED)

1. Engineering: 3–4 agent-days (0 GPU). Data prep overlaps it.
2. Training: 1.4 days.
3. Reads: 0.3 day.

That is **about 5–6 calendar days**, of which **about 1.7 are exclusive Thor-GPU days**.

## 7. What the result changes, per outcome

| A1 navtest PDMS | verdict | refcv9 |
|---|---|---|
| **≥ 85.0** | **DATA** | Keep the refc trunk. refcv9 trains on (or co-trains with) navtrain. Then add the DDv2-style PDM scorer (+3.8 on REFe). |
| **75.0 < A1 < 85.0** | **BOTH** | Name the split (PREREG §4):<br>• data share D = (A1 − A0)/(88.99 − A0);<br>• method-matched share D_M = (A1 − A0)/(84.46 − A0).<br>If D_M ≥ 0.8, the residual to REFe is mostly the scorer; port the scorer before switching backbone. If D_M < 0.5, adopt REFe's backbone. In between, run A2 or E-FOV to split cameras from backbone. |
| **≤ 75.0** (plateaued) | **SENSORS/BACKBONE** | refcv9 adopts DINOv3 ViT-L + multi-camera input. ⚠️ That is an architecture change and needs the PI. |
| any, not plateaued | **UNDERTRAINED** | Extend to 4 epochs (+31–34 GPU-h). Blocked on PI compute. |

## 8. Risks

1. **The 85 line is above the PUBLISHED and MEASURED comparators.** LTF reads 83.8 and REFe without its scorer reads 84.46. A "DATA" verdict needs refc to beat both. The PI may adopt the method-matched line **84.5** at authorisation (PREREG §4); the choice is recorded before the run.
2. **One city in selection.** The selection sample is all Las Vegas, so a selection overfit to LV is possible. navtest (4 cities) is the decision.
3. **Training on 2 Hz.** The 10 Hz targets are interpolated, and ST removes temporal cues. Both are identical at inference, so this does not bias the comparison, but it caps what refc can learn.
4. **Label-distribution shift** in the agent classes (truck/bus/trailer never positive) and map supervision off. The perception heads will not be comparable to PhysicalAI reads, and A1 is not a deliverable for PhysicalAI.
5. **PhysicalAI forgetting.** The eval139 in-run monitor reports it beside. The verdict concerns NavSim only.
6. **Thor contention.** The run needs the GPU exclusively for about 1.4 days. Today the GPU is at 95 % with other agents' jobs.
