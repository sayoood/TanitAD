# DESIGN_VDAC_P6: a vision-built drivable-area check in P6's proposal selection (Decision A)

**Status: DRAFT design card, 2026-11-10.** PI go for Decision A, 2026-11-10 (*"Go ahead with Decision A, write the design
card"*). Design and feasibility only: nothing trained, no GPU, no pod, no navtest truth read. Facts and their sources are
in `FEASIBILITY.md` (same folder); this card cites its sections as F1-F6. Pre-registration draft:
`PREREG_VDAC_P6_DRAFT.md`. Evidence classes: MEASURED / INHERITED / ESTIMATED / UNVERIFIED / HYPOTHESIS.

## 0. Recommendation (read this first)

1. **Head:** the S1c drivable head, unchanged (2,440,450 params, 2 m BEV SDF + mask over x in [-8, 72) m, y in [-32, 32) m),
   trained on **P6's frozen `visual_ctx`** with the banked navtrain labels, 10,000 updates, two seeds. No trunk change, so
   P6's 64 proposals and its six scorer heads stay bit-identical.
2. **Entry:** a **soft multiplier on P6's DAC probability**, not a hard veto: p_DAC' = p_DAC x Phi(d / sigma_b), d = the
   minimum predicted signed distance under the proposal's footprint over 0-4 s, sigma_b = the head's own boundary error,
   measured on held-out navtrain logs before any navtest read. Zero parameters fitted on DAC or PDMS outcomes.
3. **First, a zero-GPU ceiling (S0):** the same rule fed the TRUE raster on P6's navtrain proposals. The October veto read
   says a hard veto captures 14 % of the oracle even with a perfect raster, which projects to ~0.22 PDMS on P6 against a
   0.4971 bar (F5, ESTIMATED). If no rule clears a 0.33 capture ratio with the true raster, no GPU is spent on a DAC-only check.
4. **Compute:** S1 ~ **7.0 L40-h** (two head seeds + a shuffled-label control) + S2 ~ 0.2 L40-h navtest forward, one P7
   pause; S0 and every read on CPU (section 7).
5. **The memo's premise needs a correction before the card freezes (F0 C1, C2):** the "IoU 0.18-0.27 frozen probes"
   were untrained heads on u400's features; trained frozen-trunk heads reached 0.72-0.82, and the binding problem
   measured in October was the raster-to-verdict conversion, not IoU. This design is built on the corrected premise.

### Decisions needed

| id | who | decision | recommendation |
|---|---|---|---|
| D-VDAC-1 | MM | correct `H-VDAC-P6-1` and the memo (F0 C1/C2), log the root-cause class (a number read outside its scope: an untrained-floor instrument quoted as a trained probe) | yes, before freezing |
| D-VDAC-2 | MM | S0 on navtrain only (needs P6's label bank from the pod), or also the E3 oracle ceilings on the banked P6 table + navtest GT store (a navtest-truth read of ORACLE rows, K + 3) | navtrain first; E3 oracle rows only if S0-NT is ambiguous (section 5.4) |
| D-VDAC-3 | PI | is the TRACKED footprint admissible at inference (NAVSIM's LQR tracker from P6's own ego kinematics, which include a tire steering angle) | ask now; it is the largest measured conversion lever (+0.794 vs +0.367 on u400 GT) |
| D-VDAC-4 | PI | pod time ~7.2 L40-h in one P7 pause, after S0 passes | go conditional on S0 |
| D-VDAC-5 | MM | a city-stratified held-out set H48 (needs one new flag, section 3.2) vs the existing 24 logs (SG = 1 log) | H48 |
| D-VDAC-6 | MM/PI | the agent check (NC 1.12 % + TTC 1.527 points, a larger term than DAC by F5's estimate) has no labels; it is this card's committed next lever and needs its own card and label build | register now, build after S2 |

## 1. What the check is for, and its band

- **Target:** the DAC part of the 3.040-point zero-multiplier term (249 DAC-zero tokens, 2.050 %; F1). The oracle DAC veto on P6
  is ESTIMATED at ~1.52 points (F5); the P1 bar is +0.4971, so the check must capture about a third of the oracle.
- **Band (PI ruling 2026-11-08):** NAVSIM plans are 4 s, so the check covers the **operative (0-2 s) and tactical (2-4 s)** parts of
  the 6 s trajectory band. Every read reports d split by band (0-2 s, 2-4 s). STRATEGIC is PAUSED (PI 2026-09-19) and is not touched.
- **Tier:** NAVSIM v1 PDMS, open-loop pseudo-simulation (T1-class), table lookup. Head perception reads are T0-like.

## 2. The head

### 2.1 Architecture (built; `PK/refe/aux_drivable_head.py:7-15, :142-171`)

| item | value |
|---|---|
| input | `visual_ctx` [B, 7,680, 256] of P6 = `scene_proj(backbone tokens + 3D position)`, `PK/refe/model.py:759`; one tensor, no ego, no command, no map |
| queries | 40 x 32 learnable BEV queries + Fourier code of the cell centre |
| body | 2 x `CrossBlock(256, 8 heads, ctx 256)` (P6's own block class) |
| output | per 2 m cell: mask logit and SDF `6 m * tanh(.)`, over **x in [-8, 72) m, y in [-32, 32) m**, rear-axle frame at t0 |
| size | **2,440,450 params** (MEASURED, `run.json` n_params) = 12.8 % of P6's 18,998,850 trainable; P6 + head = 324,518,724 |
| loss | BCE(mask logit, 4 x 4 block mask fraction) + 1.0 x Huber(SDF/6, block-mean SDF/6, beta 0.1) (`aux_drivable_head.py:15`) |
| trained | the head only, its own AdamW (`--aux-drv-lr`, a registered value); **trunk and proposal decoder frozen**, feed `off` |

**Why the 2 m grid suffices:** with the TRUE field, a 2 m bilinear SDF gives NAVSIM's exact corner verdict on 99.26 % of
corners and the whole-proposal verdict on 98.08 %, the same as the native 0.5 m grid (`PK/SPEC_AUX_DRIVABLE_HEAD.md` 2.2,
MEASURED on 20 samples). The output grid is not the limit; the predicted boundary's accuracy is.

### 2.2 Why "the frozen probe failed" is the wrong premise, and why the frozen trunk is still the first rung

- MEASURED (F2): on u400's frozen trunk a TRAINED head reached IoU [0, 40) 0.723 (C3), 0.764 (natural navtest), 0.821 (held-out
  navtrain); it fits the features in-sample at 0.894 (A2). The 0.18-0.27 the memo quotes are untrained initial heads.
- K2 (IoU >= 0.80 on navtest) missed by 0.04-0.08, driven by non-LV cities (SG 0.632). S1d found the draws 66 % LV.
- **P6's trunk is a different object:** 25 epochs from DINOv3 + fresh LoRA, with the PDM loss reaching the trunk
  (`--scorer-sees-trunk`, F1). Whether that context carries the road better or worse than u400's is UNVERIFIED and is S1's
  first number. A frozen head costs 3.2 L40-h; a trunk change costs P6's identity. So the frozen trunk goes first.

### 2.3 Registered fallbacks (each its own arm, named now, run only on the S1 branch that calls for it)

| id | change | when | cost |
|---|---|---|---|
| HA-CB | city-balanced label draws (weights inverse to city frame counts) | S1 fails on composition (non-LV cities below the bar) | new sampler flag + G-DVB; 3.2 L40-h |
| HB | a LoRA-adapted COPY of P6's trunk feeding the head only (P6's own trunk untouched) | S1 fails the boundary bar with in-sample fit high | 4.4 L40-h per arm (section 7); inference needs a second trunk pass |
| HC | 1 m cells and a boundary-weighted Huber | S1 passes IoU, fails S1-Q | 1.1 L40-h extra per arm (4x queries, ESTIMATED from attention cost) |

## 3. Labels

### 3.1 Source and admissibility

- **Store:** `E:/Projects/TanitAD-artifacts/aux_drivable/label_store` (navtrain, 103,139 frames / 1,192 logs, sha256 `4a17db67...`)
  and `label_store_navtest` (12,146 / 136 logs, sha256 `985a3bc7...`); 0 shared tokens or logs (F3).
- **Built from privileged data:** nuPlan maps and the ego's map pose. **Admissible as LABELS only** (PI 2026-08-03: labels may use
  ego; inference is vision-only; PI 2026-10-11: map-derived drivable area is a training label, never an inference input).
  At inference the check reads the camera images (through P6's trunk), the head, and P6's own proposals. No map, no route, no lane graph.
- **The leak question asked generally (CLAUDE.md):** the label is derived from map + ego pose; the head's input contains neither
  (`visual_ctx` is formed before any ego path, `model.py:759` vs `:771`). The check's footprint uses the proposal, which is P6's output,
  not a label input.
- **navtest labels** are used only to evaluate the head at S2 and are never trained on or used to choose anything.

### 3.2 Held-out set (head never trains on it)

**H48** = the 24 SFT held-out logs (`PK/splits/sft_heldout_logs.txt`; LV 15, PIT 4, BOS 4, SG 1, 1,708 frames, F3) **+ 8 SG,
8 BOS and 8 PIT logs**, each the 8 lowest `sha256(log name)` among that city's navtrain logs outside the 24, listed and hashed in the
card before training. Reads on H48 are reweighted to navtest's city mix (LV 33.5 %, BOS 30.7 %, PIT 20.1 %, SG 15.8 %). The 24 SFT
logs are already outside `scorer_finetune.py`'s data (docstring l.17-18); the 24 extra logs need one new flag
`--aux-drv-exclude-logs <file>` that sets `valid = 0` on their labels (a frame without a label contributes exactly 0 to the loss).
The trunk is frozen, so the scorer arms seeing those frames does not touch the head. ⚠️ All H48 logs are in P6's own training set (P6
used all of navtrain): held-out for the HEAD, in-sample for P6's trunk and scorer, disclosed with every S1 number.

## 4. Training data and steps (S1)

- **Recipe = S1c's** (`PK/eval/PREREG_S1C.md` row "Arm": `--max-updates 10000 --batch 8 --accum 1 --eval-every 5000`, feed off,
  visual tap, depth 2), with **`--ckpt` = P6's `model_final.pt`** (sha `968a469d...`) and the H48 exclusion. 80,000 frame draws;
  S1c's equal draw reached 64,146 distinct frames from 1,161 logs (F2). `--aux-drv-lr` = S1c's registered value (to be copied from
  `S1C_DESIGN.md`, UNVERIFIED here). `--w-aux-drv` named in the argv.
- **Arms:** HA-s0 (seed 0), HA-s1 (seed 1, the training-variance replicate, H-ESTIM-SEED-1), HA-SHUF (seed 0, `--aux-drv-shuffle-seed`,
  a derangement of the label keys, 2,000 updates: a deliberate regression that must fail S1). Initial heads of each seed are banked
  for the init floor (A2 pattern).
- The scorer arms that `scorer_finetune.py` trains alongside are by-products and are not used; the check uses P6's unchanged scorer.

## 5. How the check enters selection

### 5.1 The rule (primary, VDAC-Phi)

For each navtest token, proposal k (P6's executed 8 poses at 0.5 s, `proposals_p6final.npz` [12146, 64, 8, 3]):

1. Poses: the t0 pose (0, 0, 0) plus the 8 poses, yaw unwrapped, linearly interpolated to 0.1 s: 41 poses.
2. Footprint: AUXVETO's (`PK/eval/PREREG_AUXVETO.md` section 4: half-length 2.588 m, half-width 1.1485 m, centre offset 1.461 m),
   4 corners per pose: 164 points.
3. Field: the head's MASK (`logit > 0`) on the 2 m grid, turned into a signed distance by a Euclidean distance transform (positive
   inside), exactly as AUXVETO read every predicted raster (the head's SDF channel disagreed with the mask sign on 2.0 % of cells there).
4. d_k = the minimum of the bilinearly sampled field over the points inside the raster extent. Points outside are excluded and their
   share is reported; a proposal with no point inside gets q_k = 1.
5. **q_k = Phi(d_k / sigma_b)** (standard normal CDF). sigma_b = the RMS error of the predicted field against the label field over cells
   with |SDF_gt| < 3 m, on H48, same pipeline. An instrument property measured at S1, frozen into the S2 reader before any navtest read.
6. **Pick = argmax_k p_NC x (p_DAC x q_k) x (5 p_EP + 5 p_TTC + 2 p_C) / 12** with P6's own logits (`read_scorer1p6.pick_logits` with the
   DAC probability multiplied), torch float32, first index on ties.

**Why soft.** The October veto failed on false vetoes: V-S1C recovered 20 of 529 zeroed picks but vetoed P6's predecessor's good pick on 30 of
529 (F2). Under a multiplier, a false alarm only moves the pick when another proposal is close in score, while a confident off-road
estimate (d << -sigma_b) still drives p_DAC' toward 0. Phi(d / sigma_b) is the probability that a point is inside if the boundary
error is Gaussian with scale sigma_b; the minimum over 164 points makes it conservative. **HYPOTHESIS, not a measured property.**

**What stays fixed.** NC, EP, TTC and C are P6's heads; NC x DAC stay separate factors (PDMS rule shape kept); P6's proposals are
bit-identical. The **zero control** (q = 1 everywhere) must reproduce P6's pick on 12,146 / 12,146 tokens.

### 5.2 Secondary and comparison rules (read, reported; they cannot be adopted under this card)

| arm | rule | role |
|---|---|---|
| VDAC-PhiL | q_k = sigmoid(a + b d_k + c s_k + e t_k), s_k = share of points with d < 0, t_k = first time with d < 0 (4 s if none); a, b, c, e fitted on H48's P6 proposals with NAVSIM DAC labels, head out of sample, **P6's logits not used** (they are in-sample on every navtrain log) | does a calibrated map beat the parameter-free one |
| VDAC-V | AUXVETO's hard veto, margin 0.25 m, fallback to P6's pick | the October rule on P6: soft vs hard |
| VDAC-PhiT | VDAC-Phi on NAVSIM's LQR-tracked states | only if D-VDAC-3 rules it admissible |
| VDAC-SHUF | VDAC-Phi with the shuffled-label head | deliberate regression: must not pass P1 |

### 5.3 The H2 feed is not the primary entry

Feeding raster scalars into `score_q_mlp` and fine-tuning the scorer is built (`--aux-drv-feed on`) but it was not promoted at S1 (F2), and
every scorer fine-tune of P6 so far shares P6's blind spot (`H-PESS-P6-1`, outcome PE4): the failing picks are out of sample (PDMS-0
picks 1.01 % in-sample vs 3.06 % on E1, `why_miss/why_miss.json`). A fine-tune learns to trust what is right in sample. It stays the
fallback named in the PREREG's branch list, not the first arm.

### 5.4 S0: the zero-GPU ceiling, before any training

On P6's navtrain label bank (~60,000 sets with per-proposal NAVSIM sub-scores; pod `/workspace/data/p6base_labels`, F6 item 3), with the
**TRUE** raster from the navtrain store in place of the head:

- arms: O-DAC (oracle), V0 / V25 (veto at 0 / 0.25 m), Phi-GT = the section 5.1 rule on the TRUE field at sigma in {0.5, 1.0, 2.0} m
  (it measures how much a soft rule of that width keeps of the oracle; boundary NOISE is not simulated, because a per-cell noise model
  would misstate a spatially correlated error), each pose-only, and tracked if D-VDAC-3 allows;
- quantity: capture ratio R = gain(rule) / gain(O-DAC) in PDMS over the sets, paired log-cluster bootstrap;
- decision: **R_needed = 0.4971 / 1.52 = 0.33** (ESTIMATED from F5). If V0 and every Phi-GT rule read R < 0.33, then even a perfect
  raster under this family of rules cannot clear P1, and S1 does not launch for DAC alone (the PREREG's S0 branches name what does).
  This is a go/no-go for spending, taken as the maximum over five rules, so it is lenient by design and claims nothing.
- Caveat: navtrain is in-sample for P6, so the failure set is smaller (PDMS-0 picks 1.01 %) and R can differ on navtest. If R is
  within 0.05 of 0.33, the MM may authorize the E3 oracle rows (D-VDAC-2) as the tie-break.

CPU: AUXVETO's full read took 22.0 s for 7 arms on 1,058 tokens (MEASURED); 60,000 sets scale to ~21 min (ESTIMATED), tracker excluded.

## 6. Admissibility and information disjointness (binding rules, checked one by one)

1. **Vision-only at inference** (PI 2026-08-03): the head reads images through P6's trunk; the rule reads the head and P6's proposals and
   logits. The tracked variant also reads ego kinematics, which P6 already consumes (`model.py:185`); its admissibility is D-VDAC-3.
2. **Goal input must not carry the situation classifier** (PI 2026-08-03): P6 has no situation classifier; its command is NAVSIM's
   `driving_command`, a route input that stays an input (PI 2026-09-19). The check reads neither; its output multiplies the DAC factor only.
   **Gate test (G-DVB):** autograd of q with respect to the ego and command tensors is exactly `None`/0.0; mutation: route `ego_tok` into the
   head's context, and the check must turn RED.
3. **Labels may use ego and maps** (PI 2026-08-03, 2026-10-11): yes, offline only (section 3.1).
4. **No HD-map geometry for planning** (PI 2026-09-28, INHERITED via `SPEC_AUX_DRIVABLE_HEAD.md` 6.2): the map reaches only the loss.
   Test: `grep -c "map_api\|nuplan\|origin_world"` on the reader's inference path must be 0.
5. **Built, tested and unreachable** (2026-09-10/11): `--w-aux-drv` defaults to 0.0 and `--aux-drv-feed` to off; every arm names its levers
   in the argv; the post-training gate checks that every head parameter has Adam state; the S2 reader proves the multiplier is live with a
   non-zero pick change on a planted raster and an exact zero change under q = 1.

## 7. Compute estimate (one L40; arithmetic shown)

| stage | arithmetic | estimate | class |
|---|---|---|---|
| S0 ceiling | 60,000 / 1,058 x 22.0 s | ~21 min CPU (+ tracker, unmeasured) | ESTIMATED from MEASURED |
| HA-s0 | S1c: 10,000 updates x 1.2753 s/update = 3.54 h, 3.71 h with evals, on an A40 (MEASURED, S1c read section 1); same trunk shapes; L40 = A40 x 0.863 (memory note `pod-gpu-is-l40.md`, ESTIMATED from P6's rate) | 3.71 x 0.863 = **3.20 L40-h** | ESTIMATED |
| HA-s1 | same | **3.20 L40-h** | ESTIMATED |
| HA-SHUF | 2,000 / 10,000 x 3.20 | **0.64 L40-h** | ESTIMATED |
| held-out forwards | 1,708 + ~2,000 frames x 0.066 s | ~0.07 L40-h | ESTIMATED |
| **S1 total** | | **~7.1 L40-h** | one P7 pause |
| S2 navtest forward | P6 trains 256 samples in 50.5 s (INHERITED, `PREREG_SCORER1_AMENDMENT_3.md:24` via WIRING_NOTE section 1); forward = 1/3 of that (ASSUMPTION, as WIRING_NOTE) = 0.066 s/sample; x 12,146 tokens = 800 s; two heads add ~1 % | **~0.22 L40-h** (or the dev box, not measured, ~3.5 h ESTIMATED from ~1 s/sample fp32 forward) | ESTIMATED |
| S2 reads | AUXVETO / PESS readers: seconds to minutes | CPU | MEASURED analogues |
| HB fallback (if called) | 80,000 samples x 50.5 / 256 s = 15,781 s | 4.4 L40-h per arm | ESTIMATED |

**Total to the S2 verdict: ~7.3 L40-h + ~1 CPU-h.** Memory: S1c ran at batch 8 on an A40 (48 GB); the head adds about 0.13 GB of
activations and 0.04 GB of parameters and Adam state at micro-batch 4 (`SPEC_AUX_DRIVABLE_HEAD.md` 3.4, ESTIMATED).

## 8. Latency (PI: at most 100 ms end to end on Thor with 4 cameras; never gate research on latency)

- Head forward: 32.2 GFLOP per sample = **0.53 % of the trunk's forward** (`SPEC_AUX_DRIVABLE_HEAD.md` section 0, analytic). P6's
  own Thor latency is not measured (F6 item 6), so the absolute figure is open.
- Rule: one 40 x 32 distance transform + 64 x 164 bilinear lookups + 64 normal CDFs: well under 1 ms on a CPU (ESTIMATED).
- Tracked variant: 64 LQR rollouts of 40 steps per token; NAVSIM's tracker is Python and its cost was not measured. If D-VDAC-3 admits it,
  a vectorised tracker is an optimisation item, not a research gate.
- HB would add a second trunk pass (roughly doubling perception latency); that is a reason it is a fallback, not a blocker.

## 9. Risks and what would change the plan

| risk | evidence | what the card does |
|---|---|---|
| the ceiling is below the bar | F5: a perfect-raster hard veto projects to ~0.22 against 0.4971 | S0 measures the soft rule's ceiling at zero GPU and stops a DAC-only build if R < 0.33 |
| the boundary error dominates (~2 m vs the ~0.25 m a verdict needs) | S1c near-boundary MAE 1.97 m (F2) | sigma_b enters the rule, so a worse head makes the check softer rather than wrong; S1-Q bars the proposal-level ranking |
| P6's trunk reads the road worse than u400's | UNVERIFIED (no head on P6 yet) | S1 reads it first; HB is the named fallback |
| city composition (SG, PIT, BOS weaker) | S1c natural per-city IoU (F2) | H48 city-stratified bars; HA-CB fallback |
| training variance | one-seed heads (H-ESTIM-SEED-1) | two seeds; the replicate must agree in sign before ADOPT |
| the tracker holds 27.8 % of DAC failures, invisible to a pose-only footprint | INHERITED (`AGREEMENT.json` via `PREREG_AUXVETO.md` section 1) | VDAC-PhiT, pending D-VDAC-3 |
| in-sample navtrain everywhere for P6 | P6 trained on all navtrain, incl. H48 | disclosed; navtrain is used for perception bars and sigma_b, never for P6-logit fits |
| the larger term is agents, not DAC | NC 136 tokens + TTC 1.527 points (F1, F5) | the agent check is the committed next lever (D-VDAC-6) |
| harm from false penalties (Hydra-MDP++ style, V-PRED's -0.788) | F2 | G1 guards (DAC/NC/TTC sub-scores >= -0.5); the zero control; HARM branch names the next arm |
| a lever built and never reached | the 2026-09-10/11 class | section 6 item 5 and the PREREG's gate list |

## 10. What must be built (none of it exists yet)

| item | where | note |
|---|---|---|
| `--aux-drv-exclude-logs` | `PK/refe/scorer_finetune.py` (label lookup) | G-DVB entry + mutant (section 3.2) |
| launch-gate profile `refe_vdac_p6` | `launch_gate.py` | a NEW dict built from `refe_aux_s1c`; shared constants untouched (memory `pin-edit-moves-profile-fingerprints.md`: ~69 launch_gate-reading tests run before landing) |
| `vdac_read.py` (S0 and S2) | this folder, then `PK/eval/` | built from `auxveto_read.py` (validity checks, footprint) and `read_pess.py` (E3 bars, G1-G4, families, K keys) |
| P6 navtest head forward | `PK/eval/aux_navtest_forward.py` with P6's checkpoint | identity: the same pass's scorer logits bit-identical to `proposals_p6final.npz` `logits` on 12,146 / 12,146 |
| the H48 log list + sha256 | this folder | written before any training |
