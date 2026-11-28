# PREREG_VDAC_P6_A5: the arm after a failed S1a. Stage 0 read S0-WEAK, so HB and HC run as ONE registered pair

**Status: FROZEN 2026-11-12T21:30:35Z by the Master Mind (MM).** Written by the REFe scorer-stream teammate from `PREREG_VDAC_P6_A5_DRAFT.md` (sha256 `227fa6087f6b...`), with every value the draft left to freeze resolved below.
It amends `PREREG_VDAC_P6.md` (sha256 `5594c2ae777a...`) + A1 (`8f121bdf9d6f...`) + A2 (`c3d563586be6...`) + A3 (`a7708573...`) + A4 (`045a10c8f712...`) **in the S1 branch table only**. None of them is edited.
It follows the card's own S1 outcome rows (`PREREG_VDAC_P6.md` section 2, S1): both S1 bars failed, so the card commits HB (S1-FIT) or **HB + HC as one registered pair** (S1-WEAK). Stage 0 split those two rows (section 2).
Evidence classes: MEASURED (artifact path) / PUBLISHED / INHERITED (not re-verified) / ESTIMATED / HYPOTHESIS / UNVERIFIED. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
Every code and data pin below is a MEASURED fact at git commit **`a01be4a4`** ("Build VDAC A5's HB + HC pair ... not launched"), read through the mirror with `git show a01be4a4:<path>`.

## 0. Decisions made before this card was frozen (stated, not re-opened)

| # | decision | by, date | basis |
|---|---|---|---|
| D1 | **HB's LoRA learning rate is `--aux-drv-lora-lr 0.0002` (2e-4).** It is P6's own single AdamW group: `train.py:1462` (`params = [p for p in model.parameters() if p.requires_grad]`) and `train.py:1469` (`AdamW(params, lr=a.lr)`); P6's recorded argv carries `--lr 2e-4` (`p6_pod.sh:487`). The draft's 5e-5 was a HYPOTHESIS and is replaced. | MM, 2026-11-12 | BUILD4 section 1 (MEASURED from source). Caveat, INHERITED: P6's `--lr` is read from the pod script's argv text, not from a run record of the final checkpoint. |
| D2 | **HC's boundary-band weight is `--aux-drv-bw 3.0`** with `--aux-drv-bw-band-m 3.0` (the card's near-boundary definition). The multiplier is `1 + bw`: 4 inside the band, 1 outside. **HYPOTHESIS value, fixed now, never tuned.** | MM, 2026-11-12 | draft section 3; `launch_gate.py:913` |
| D3 | **The pair runs because Stage 0 read S0-WEAK** (in-sample IoU 0.7864 < 0.85), the register row `D-VDAC-S0-1`. | MM, 2026-11-12 | section 2 |
| D4 | **Pod time approved: 12.8 to 16.7 L40-h (ESTIMATED), run as P7 pauses.** The PI approves the UPPER bound as the cap. | PI, 2026-11-12 | draft section 5 |

No bar below moves after a number is seen.

## 1. Short diff against the draft (what changed when it was frozen)

| item | draft | this card |
|---|---|---|
| `--aux-drv-lora-lr` | 5e-5 (HYPOTHESIS) | **2e-4 = P6's own lr** (D1) |
| HC weight | "3 is HYPOTHESIS, fixed" | **3.0, band 3.0 m, decided D2**; the built multiplier is `1 + bw` = 4 / 1 |
| Stage 0 | "to be built, first in the pause" | **DONE**, verdict S0-WEAK (section 2); no Stage 0 cost remains |
| `s1_read_v2.py`, HC reader | "must be frozen before any HB row" | **built and landed at `a01be4a4`** (`eval/s1_read_v2.py`, section 7); its freeze record is a separate gate before the FIRST READ, not before the launch (section 7.4) |
| HB pause timeouts | 36,000 s "UNVERIFIED how taken" | **set in `vdac_pause_p7.sh:59`**: HB-s0 36,000 s, HB-SHUF 14,400 s, HC-s0 21,600 s, HC-SHUF 7,200 s |
| "in-sample dump of the Stage 0 frames banked per arm" (draft 2.2) | a deliverable of every arm | **NOT BUILT (H10)**: dropped from the deciding set. The in-sample gap of HB / HC is reported only if a loader exists at read time, else "UNAVAILABLE, reason" (section 7.6). It never decided anything. |
| draft 2.3 "HB needs a reader change" and "HC needs a second reader change" | to build | built in `eval/s1_read_v2.py` (sha256 `796aea21506e...`, the same file that read Stage 0) |
| micro-batch | "set by the preflight's memory probe" | argv is `--batch 8 --accum 1` (`launch_gate.py:vdac_argv`, `vdac_pod.sh:40`); a **pre-registered implementation fallback** `--batch 4 --accum 2` (effective batch unchanged) is allowed for HB ONLY before any HB step trains (section 9.2) |
| K ledger | "K + 8 for the substituted arm set" | unchanged: S0 and S1 are navtrain reads and are **not keyed** (`PREREG_VDAC_P6.md` section 6); keys are written at the S2 read only (section 10) |
| HB-s1 / HC-s1 replicate arms | conditional | still conditional on X-GO and **not built** (`VDAC_ARMS` holds only s0 / SHUF for HB and HC): named blocker, section 9.3 |
| nothing else | | bars, estimator, controls, outcome table, four-family rule, gates: **unchanged from the draft** |

## 2. The Stage 0 verdict that selects the pair (MEASURED)

`PK/raw/2026-11-12-vdac-stage0-read/STAGE0_READ.md` (sha256 `58865e713b7f...`), `stage0_read.json` (sha256 `f3296ef49fc2...`), `in_sample_iou.json` (sha256 `1547bd30ea6f...`), at `a01be4a4`.

| item | value |
|---|---|
| verdict (mechanical, draft section 1 table) | **S0-WEAK**: in-sample IoU [0,40 m) **0.7864** [0.7803, 0.7924] < 0.85; `invalid_controls: []` |
| controls | C0-ID ok (256 frames, sign agreement 1.0, banked IoU 0.84770 = repro 0.84770); C0-FLIP ok (0.7864 vs flipped 0.3419); C0-FLOOR (all-drivable constant head) 0.3764 |
| in-sample minus held-out (S1-P1 0.7802) | +0.0062, reported, not deciding |
| per city LV / BOS / PIT / SG | 0.8942 / 0.7441 / 0.7366 / 0.7034 |
| per band 0-20 / 20-40 / 40-72 m | 0.8113 / 0.7611 / 0.6923 |
| S1-F (HA-s0, from the rebuilt step-0 head) | holds: P1 lower 0.5120, Q lower 0.0891; reader v2 verdict S1-WEAK = Stage 0 row (agree) |
| estimator of the 0.7864 interval | frame-level bootstrap, 10,000 draws, seed 20260927, over 2,048 frames (512 per city); NOT a training-seed floor |
| frame list sha256 | `1d353309a1d81a78cc04197ac39bd4e7ef467853ff59f5f7d3e97fde538eda45`, frozen before the forward (`FROZEN_SHAS.txt`, sha256 `5612e6024f22...`) |
| tool | `vdac_stage0.py` sha256 `04a3d5f4dde23dd669167c18a6090cb16a37c3a680ee3351729b4fd8b51dc2b7` (the Stage 0 read ran at mirror commit `28db77fc`) |

Reading (not a claim beyond it): the HA-s0 head is weak even on frames it trained on, so the S1a shortfall is under-fit or representation-limited, not a generalisation gap. Stage 0 does **not** say which of HB (features) or HC (boundary precision) addresses the deficit; that is why both are registered.

## 3. Code pinned by git blob at `a01be4a4` (every file an arm runs; blob = `git rev-parse a01be4a4:<PK>/<path>`; sha256 over the blob bytes)

| path under PK | blob | sha256 |
|---|---|---|
| `refe/scorer_finetune.py` | `4411bbdada5891fe044eafdf9ceded862dd15f73` | `8ed4168dde65997af5bd5a7d28967dd42e5e0a078b191070f36561cba7b9d87c` |
| `refe/aux_hb.py` (NEW) | `3f8fc318737c6707a70cd0cb590d72abb29dda7e` | `112e6004f430a8731a0c681e72024988d32ef7b7d7a279cc86946d1f80f16005` |
| `refe/vdac_a5_layer.py` (NEW) | `00a2fca59355d7b07a80113e8495032a07d69e2c` | `433fe03fa0b53069b305a0e1a757fe762ab6eff22c7d0114d4f5c25e23f2af7c` |
| `refe/test_vdac_hb.py` (NEW) | `9a90f38ac5111428cb4fbb55e18426c6881d2f8f` | `dcb7285545d021896e29aa0f25de2fa4708bf3cea2483a0c6e744563c5fda130` |
| `refe/aux_drivable_head.py` | `9d29e555f7d13a6d2f02825129b14bac41bd33b8` | `8a08681178b78bd85c5bbb41cb4cc6092bf923b23a6e16ffee2a53b89ee26509` |
| `refe/gate_checks.py` | `5b85d51d8982c2f4075687b96c42e83607c1bd6f` | `60781450ebd63de0c4c66f94bdbffa276cd6eb0e79a0d7ee3d8753ca8c5ce515` |
| `refe/launch_gate.py` | `abb32e5b5b193fb868e1e7acf701deba0e808928` | `73ebe594a91048bf24db6bd5a62f844ae69834e52727549d2de08510abe3a61c` |
| `refe/gate_token.py` | `e18c418a4e135b8f03c22b174ea5aa5529530b4d` | `d8184aa096996fe2711eb64fb64348916c7add36eccb0d908444f61a078f5f96` |
| `refe/vdac_probes.py` | `5a2899d406afab99395427769d3c086b90bb6911` | `9e788d0038ea003591c0c743fee6f0e68aac9f31764c1eeb85ee7f5df558f987` |
| `refe/vdac_postgate.py` | `6c5978ee0a19c52fc45fc596635b3537076655f2` | `75510c2e5284dafe36691afcadeba49b53449f0ebef51becc956ffc30bf1d66c` |
| `refe/vdac_layer.py` | `0943c8c8166d81a7aeb07131c1e35531932e699c` | `7fe864583b1ce6f63066bb890cd86d057ffaf83f805dd1d6036897a68db9bced` |
| `refe/vdac_pod.sh` | `2515c9da1474b0d0e46d4e43aa1c9603f8dfaba8` | `713f8ea408ba6ae9ea7b572a1ff8e00c2a762631feaa38f4dc96501883d0baa4` |
| `refe/vdac_chain.sh` | `02e4796256c3af6ceb8a76c8d2ed9c5d570bc945` | `c484a22fd28982aa42077d6fda90c44c97c20ef002c8da013f1d5556b88f6b9d` |
| `refe/vdac_pause_p7.sh` | `5417799cc0e048c080794bcca0085c9f39fa3b42` | `ab638780224a3ffe8f02789a3d33926b9e0ee8578894bfd38d989836db6075b1` |
| `refe/vdac_export_h48.py` | `80cecdc7ae9719ee322eb280b9203d1bd7965466` | `fa432da4576f578f68db5478af182df634f855e37f724f2f51c32c28fc4e7f5e` |
| `refe/scorer1p6_pod.sh` (the parent argv) | `8c20d3094b953a40a32e2a2f10927f70a949f7aa` | `2aef6fc1f6cae55d65896c0d0c16ff6246afe6f983bb487f18e184b0b1b1ec8a` |
| `refe/test_vdac_gate.py` (G-SUITE member) | `e4a13c19f69d05b25722a71a042ebba31aefa28b` | `334dc8de71dc64bb2c132edb7ceaf49e23bc28a3b2e29fd60c2131158d7008b1` |
| `refe/test_aux_s1_wiring.py` | `5f6258c0e89e8ae316a72854af1f5c2bd9bdb5d1` | `095c5fe2e2a44af8681c681087f75e7c720a724e97e09c08e9e593ab31a3170b` |
| `refe/test_dump_scores.py` (the dest-whitelist fix) | `5cefe88ed117f8da9ccf9050f29403c2dc4cb6d9` | `d98d16a207a5d3126c190f2d0fc631e0131f90b6f9af42760e301aadbcb8be7d` |
| `refe/model.py`, `train.py`, `ckpt_io.py`, `planner.py` (UNTOUCHED by the build) | `189e1ac0...`, `0e115c0d...`, `7e74b982...`, `6fb6598a...` | `e6011cd8...`, `eafd03e5...`, `240de348...`, `91437cd0...` |
| `eval/s1_read_v2.py` (the reader of section 7) | `fa73bd4976b30add9653efd318c9bbfa9330e1d0` | `796aea21506e27d1646f5b7aa71b08089ced8c6d92a5b02429d35431692b09f4` |
| frozen S1a reader `raw/2026-11-09-scorer1p6-prereg/vdac/s1a_read/s1_read.py` | (read by sha) | `58aa3a77b976a6fb4b08b4bc5126d572e8be8ef1fc37926d9c9ce0983eed718e` |

⚠️ `eval/s1_read.py` at the tip is a DIFFERENT file (sha256 `80a9f41007fe...`) from the frozen reader above. The frozen one is the `s1a_read/` copy; every read sets `S1_READ_V1` to it (section 7.3).
Landing note (MM): the landed commit must contain these blobs byte for byte in `refe/`; `ship-manifest` refuses a file that is not its blob (`gate_token.py:768-785`).

## 4. Data pinned by the bytes the tools assert

| input | pin |
|---|---|
| P6 checkpoint `refe_runs/p6_paper_faithful/model_final.pt` | sha256 `968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1` (`launch_gate.py:517`, chain `CKPT_SHA`) |
| P6 label dump | `SCORER1P6_DATA_PIN` = 60,000 train sets, 1,705 held-out sets (`launch_gate.py:530`) |
| aux label store | store sha256 `4a17db673b1548ba35724ab848a7dc551e4b0bc1e49c57ae93e6357454bf287a`, 103,139 stored keys (`aux_drivable_head.py:64-65`) |
| H48 list `splits/h48_logs.txt` | sha256 `2ae13935a3ead7f30b015f33b0bebc7a35e70eb2a4194161f8cbb66ac89b8da8`, 48 logs, blob `596195563b0dba1010c490d5434847799c78c2da` |
| navtest log list / token list | `79daa3c2ba3fbf1c6d45d0e3b630f43bb26a073160a34ca5f9c98553c0bf43e0` / `e61613ce129bb67dc654a5a575de2889a22fcfcbc64e4ec389a38295b8df9c7a` (`launch_gate.py:459`) |
| aux navtest list (G-SUITE E10), CR-stripped | `3ba1c96488f157fd2ca7c57c83abe439b4c9680ebdb1eadf65c34ac323d42d44` (`vdac_chain.sh:52`) |
| `splits/navtrain.yaml` | sha256 `5f3c406a06c961e75d69d33bc127acd5704a3a440a7186bfa2212493ecf9fc01` |
| `splits/sft_heldout_logs.txt` | sha256 `b814998deeb40a9d7839cce2807f8f282ede76c7f2c5ccce97adc064b27f4902` |
| H48 export `h48_proposals_subscores.npz` (P6's proposals and NAVSIM sub-scores, shared by HA, HB, HC) | sha256 `14560686490a2edbb0a9611d69f1285f84b396bad1c28138cf192964b2809345` (S1a manifest, `PK/raw/.../s1a_read/inputs_manifest.json`, sha256 `e9068856f117...`); a pod re-export that differs is read against THIS one only after its difference is explained |
| HA-s0 reference files (the paired comparator) | `head_s0_a` `6b9ac2bc1bc5989e7904e1c1b1d12eb6e4c2356fe7ad5658cc6fecc5196c03eb`, `head_s0_b` `64d2f71be45154bd36d71d3955a9151023d6f10365f9bac5376c6e3ce02aac3b`; HA-SHUF `5cb8c49255de...` / `4ee7816cc7a1...`; HA initial head `head_init0_a` `2ebe4b540b163e2638532a71d0a006a35e7ce9e44ee0c0cb1bce8fd10903f94a`, `head_init0_b` `8b556eb23c842732ec91837781f7181c1f90406f3a1aa2316e2e9493bb542ed4` (`manifest_s1f_v2.json`, sha256 `8ee3f4add96e...`) |
| banked S1a read (regression identity target) | `s1a_read.json` sha256 `23d54a9e4110b1e896ef8568fcf90402e179cb876eb143a7d527d53a034bf356`; S1-P1 0.7802 [0.7587, 0.7974], S1-Q 0.8332 [0.7951, 0.8691], C-SHUF 0.4468, C-FLIP 0.7056 |

Unchanged from the cards: H48 and its exclusion (A3), the P6 60,000-set bank draw with `--epochs 2`, the A3 disclosures, A4's G20 bar 0.99.

## 5. The arms, their order and their argv (single lever each)

Parent argv: `scorer1p6_pod.sh ARM=C0 MODE=argv` with `--max-updates / --eval-every / --seed / --out` replaced; the head recipe of HA-s0 follows. **Common tail (every arm):** `--batch 8 --accum 1 --epochs 2 --aux-drv --w-aux-drv 0.1 --aux-drv-lr 0.0002 --aux-drv-tap visual --aux-drv-depth 2 --aux-drv-labels <store> --aux-drv-exclude-logs <H48> --aux-drv-exclude-sha256 2ae13935...` (`launch_gate.py:vdac_argv`, `vdac_pod.sh:40`).

| order | arm | seed | updates | extra flags, ALL NAMED (each one's default disables its lever) | purpose |
|---|---|---|---|---|---|
| 1 | **HB-s0** | 0 | 10,000 (eval every 5,000) | `--aux-drv-trunk-lora on --aux-drv-lora-lr 0.0002` | the head reads a LoRA-adapted COPY of P6's trunk (lever: the head's input trunk is adapted) |
| 2 | **HB-SHUF** | 0 | 2,000 (eval every 1,000) | HB's flags + `--aux-drv-shuffle-seed 0` | deliberate regression of HB: a derangement of all 103,139 label keys |
| 3 | **HC-s0** | 0 | 10,000 | `--aux-drv-cell 1.0 --aux-drv-bw 3.0 --aux-drv-bw-band-m 3.0` | 1 m cells (80 x 64 queries) + boundary-weighted Huber, frozen trunk |
| 4 | **HC-SHUF** | 0 | 2,000 | HC's flags + `--aux-drv-shuffle-seed 0` | deliberate regression of HC |

* **Pause plan:** one stage per P7 pause (`vdac_pause_p7.sh:18-19`): stage `A5HB` runs `HB-s0` then `HB-SHUF`; stage `A5HC` runs `HC-s0` then `HC-SHUF`. The pair is registered as one; HB first because it is the more expensive and its memory is the one unmeasured quantity. The first arm of a stage that does not finish stops that stage (`vdac_pause_p7.sh:65`).
* **Same 10,000 updates as HA-s0 on purpose**: a longer arm would confound the lever with training length (the held-out IoU was still rising from update 5,000 to 10,000: 0.799 to 0.823, `s1a_read`). Training length is a named successor (section 8, X-LIFT row), not part of these arms.
* **Same seed 0 as HA-s0, not a replicate**: the single-lever comparison against HA-s0. A same-seed re-run is bit-identical (`same-seed-replicate-is-bit-identical`), so a replicate MUST change the seed (HB-s1 / HC-s1, section 9.3).
* **Comparator:** HA-s0 (banked, S1a). **Not retrained.**

### 5.1 HB (the only trunk change) vs HA-s0

| item | HA-s0 | HB-s0 |
|---|---|---|
| tensor the head reads | P6's frozen `visual_ctx` | `scene_proj(backbone_copy(img) + pos3d)`, a `copy.deepcopy` of P6's backbone held as a SIBLING (not in `model.state_dict()`), same shape [B, 7,680, 256] |
| trainable trunk parameters | none | the copy's existing rank-32 Q/V LoRA `A`,`B` tensors: **3,145,728** (24 blocks x {q, v} x 32 x (1,024 + 1,024); registered as `launch_gate.py:912` `VDAC_HB_LORA_PARAMS`, asserted by G-DVB and G21 against the BUILT count) |
| LoRA init | | **P6's own trained LoRA**, so step-0 `ctx_hb` is bit-identical to P6's `visual_ctx` (G21 `step0_ctx_equal`) |
| LoRA optimiser | | second AdamW group, weight decay 0.01, peak lr **2e-4 (D1)**, P6's warm-up + cosine shape (`lr_at(upd) x lora_lr / lr`, `scorer_finetune.py:3027`), own grad-clip norm 1.0 |
| gradient into the copy | | from `L_aux` only; the scorer arms read P6's `sctx`, never the copy's ctx |
| P6's trunk, decoders, 64 proposals, six scorer heads | frozen | frozen, bit-identical (four asserted facts: section 8.1) |
| head, loss, data, exclusion, updates, seed | S1c head 2,440,450 params, BCE + 1.0 x Huber(beta 0.1), `--w-aux-drv 0.1`, H48 excluded | identical |

HB inference cost (reported, never gated; PI ruling 100 ms end to end on Thor, results first): the trunk runs twice per frame, ESTIMATED +~6.1 TFLOP per sample (head 32.2 GFLOP = 0.53 % of the trunk forward, `SPEC_AUX_DRIVABLE_HEAD.md` section 0). P6's Thor latency is not measured. A last-k-blocks variant is **named, not registered**.

### 5.2 HC (head and loss only; trunk frozen as HA-s0) vs HA-s0

| item | HC-s0 |
|---|---|
| head | the same class at `cell = 1.0` m: 80 x 64 = 5,120 learnable queries over x in [-8, 72), y in [-32, 32), depth 2, ctx 256; built params **3,423,490** (`gate_checks.scorer_dvb_rows`, registered; BUILD4 section 2) vs 2,440,450 |
| loss | `BCE + w_sdf x Huber` as HA, with the Huber term multiplied per cell by `1 + bw x 1[ |block-mean SDF_gt| < 3 m ]` (**bw = 3.0 -> 4 inside, 1 outside, D2**); BCE unweighted; `bw 0.0` executes nothing extra (bit-identical to the unweighted loss, asserted) |
| reachability | the cell reaches the build at `scorer_finetune.py:1770` (`AuxDrivableConfig(..., cell=a.aux_drv_cell, bw=..., bw_band_m=...)`); at the tip before `a01be4a4` it did NOT (the class of "built, tested, unreachable"); G-DVB now binds the DECLARED cell to the BUILT `nx, ny, n_params` |

## 6. Gates, controls and deliberate regressions (no launch without a PASS token; PI 2026-09-26)

Profile **`refe_vdac_p6`**; `launch_gate.VDAC_ARMS` holds the four arms (`launch_gate.py:906-910`), the argv is pinned token for token (`vdac_arm_refusals`), the token is bound to commit, tree sha, argv sha256 and the data manifest. A token is minted by the chain per arm and never by hand. The suite `VDAC_SUITE` (`launch_gate.py:922`) does NOT include `test_vdac_hb.py` (BUILD4 section 6 item 4: not added, because it would change the HA arms' suite); that test is run as its own pre-launch step (section 6.2).

| flag / mechanism | G-DVB entry (BUILT facts vs argv) | deliberate regression that must turn it RED (MEASURED, `test_vdac_hb.py`: `ZZTEST_PASS 133`, 39 mutation arms, BUILD4 section 3) |
|---|---|---|
| `--aux-drv-trunk-lora on` | copy built; `n_lora_trainable == 3,145,728`; all trainable are LoRA; copy storage disjoint from P6's; copy not in `model.state_dict()`; `p6_params_requires_grad == 0` | `REFE_AUX_MUTATE_HB_SHARE` (no copy), `_UNFREEZE_SP`, `_DETACH`, `_P6CTX` (head reads P6's ctx), `_STALE_TAP`; the `select_actx` wiring line DELETED; the `lora_step` line DELETED |
| `--aux-drv-lora-lr 0.0002` | the LoRA group's peak lr equals it; the optimiser holds exactly the copy's tensors | omit it (refused like `--aux-drv-lr`); a LoRA group at another lr; a LoRA tensor outside the optimiser |
| `--aux-drv-cell 1.0` | head `nx, ny == 80, 64`; `n_params == 3,423,490`; optimiser covers it | declare 1.0, build 2.0 (cell dropped from the build line) |
| `--aux-drv-bw 3.0 --aux-drv-bw-band-m 3.0` | the multiplier the BUILT loss applies (`bw_probe`, `aux_drivable_head.py:360`): 4 inside / 1 outside | name it, drop it from the loss |
| **G19** head disjoint from ego and command | autograd of the head output w.r.t. ego and command: None or exactly 0.0 with a non-zero control. HB: `actx` is assigned from the copy's ctx BEFORE the existing ego mutation line, so `REFE_AUX_MUTATE_EGO=1` lands on the copy's output; new `REFE_AUX_MUTATE_EGO_TRUNK=1` adds the ego token to the copy's visual tokens before `scene_proj` | both mutants RED. The chain's `mutate_ego` stage asserts G19 specifically rejects it (`vdac_chain.sh:266-281`). **UNVERIFIED end to end through the trainer's own G19 for the trunk-side mutant** (BUILD4 section 3 last paragraph): proved by gradient on the real tiny model; the first pod preflight exercises the real path. |
| **G20** label alignment (bar 0.99, A4) | label vs P6's rank-0 target points on >= 64 frames; independent of the trunk and of the cell (reads the 0.5 m store) | y-flip and one-frame shift both below 0.99 |
| **G21 (new): copy isolation** | P6 parameters outside `SCORER` have no gradient; every copy LoRA tensor non-zero; with `w_aux = 0` exactly 0.0; step-0 `ctx_hb == ctx` bit for bit; scorer-loss gradient on the copy None; `model.state_dict()` sha256 unchanged across 2 updates | `HB_P6CTX`, `HB_SHARE`, `HB_UNFREEZE_SP`, `HB_DETACH`. **bf16 caveat (UNVERIFIED):** on the pod the copy's forward carries `requires_grad` and may take a different SDPA kernel than P6's no-grad forward: a ~1e-3 `step0_ctx_max_abs_diff` would be a kernel choice. The raw field is recorded; if the bit-exact bar refuses on the pod, the arm is NOT launched and the bar is set by an addendum from the recorded value BEFORE any training (no result exists at that point). |
| G12 tap | HB's row reads the copy's ctx = `outs[2]` (`scene_proj_calls == 3`, `tap_index == 2`) | `REFE_AUX_MUTATE_HB_STALE_TAP` (reads `outs[1]`, P6's) |
| G-LIVE | initial `aux_bce` ~ ln 2 +/- 0.15; head and LoRA gradients non-zero; P6's trunk exactly zero; scorer arms bit-identical to a head-less run (feed off) | |
| G-CKPT | `aux_drv/2`: head, LoRA tensors and BOTH Adam states round-trip; 2 resumed updates bit-identical | three strict-load refusals |
| post-training gate | every head AND LoRA parameter holds Adam state, non-zero first moment (`vdac_postgate.py`, the `probe_zerograd3.py` pattern) | a LoRA group never stepped |
| exclusion audit | H48 rows reaching the loss == 0, checked 3 times (preflight data event, NOFILTER mutant rejected rc 4, run data event) | `REFE_AUXEXCL_MUTATE_NOFILTER=1` |
| **SHUF controls** | HB-SHUF / HC-SHUF: a derangement of all 103,139 keys, 0 fixed points, parsed from the data event (`excl_is`) | the control IS the regression: C-SHUF = the arm's own SHUF S1-P1 must read **< 0.50**. It matters more for HB: a head plus a trainable trunk copy is a bigger memoriser, so C-SHUF < 0.50 is the proof that HB's gain is map information and not a leak through the copy. |
| `--stage0-*` measuring modes | forbidden in any launch argv | present in a launch argv |

### 6.1 Not built; what that blocks (named, section 9 gives the smallest fix)
H5 (micro-batch accumulation of the copy's backward), H10 (in-sample dump at the end of each arm), the `aux_drv/2` loader in `vdac_stage0_forward.load_head` (`:205-206` refuses `/2`), `expected_aux_cfg` passing `cell` (`vdac_stage0_forward.py:179-180`), G-EVAL for HB / HC at S2.

### 6.2 Pre-launch tests that are NOT green in the build agent's environment (ENV, not defects, but UNVERIFIED until run where the data exists)
`refe/test_launch_gate.py` (stops at `FileNotFoundError: D:\Projects\TanitAD\data\refe_consistent\targets_rank1.jsonl`, identical on the pristine tree, so everything after that line is UNVERIFIED), `test_vdac_stage0.py` and `test_vdac_stage0_forward.py` (refuse on `eval/s1_read.py` sha `80a9f410` != the frozen pin; run with `S1_READ_V1`), `test_vdac_hb.py` on the pod once (a test that passes off-pod can fail on the pod). The MM runs these before shipping (FREEZE_CHECKLIST step 6).

## 7. The read: instrument, estimator, bars, controls

### 7.1 Instrument
`eval/s1_read_v2.py` (sha256 `796aea21506e...`) = the frozen S1a reader `s1_read.py` (sha256 `58aa3a77b976...`) IMPORTED UNCHANGED with its sha asserted (`s1_read_v2.py:V1_SHA256`, `load_v1`) + exactly: (i) a manifest-driven arm table (arm names, each trained arm names its own SHUF and its own initial head); (ii) paired blocks X minus REF (HB-s0 minus HA-s0, HC-s0 minus HA-s0) through the SAME log resamples; (iii) HC's 1 m grid; (iv) an optional `in_sample_iou`. On the S1a inputs it reproduces `s1a_read.json` digit for digit (MEASURED on 2026-11-12: Stage 0's `s1f_read_v2.json` equals S1a to the digit: S1-P1 0.7802, S1-Q 0.8332 [0.7951, 0.8691], C-SHUF 0.4468, C-FLIP 0.7056).

### 7.2 Estimator, and the question its interval answers
**Paired log-cluster bootstrap, 10,000 draws, seed 20260927, over the H48 logs (45 of the 48 listed logs hold frames, MEASURED `s1a_read.json`) with the models held fixed** (frames within a log are not independent; the arm-minus-HA-s0 deltas go through the same resamples). It answers **"would another draw of LOGS / EPISODES say this?"** and **NOT** "would another TRAINING RUN say this?" (`H-ESTIM-SEED-1`) and not "would another INFERENCE run say this?" (the head is a deterministic forward with no sampler, so the inference variance is nil by construction; ESTIMATED, not separately measured).
⛔ **A separated one-seed interval is NECESSARY, NOT SUFFICIENT for a lever claim.** The paired bootstrap is structurally blind to training variance. X-LIFT therefore triggers the replicate arm (section 8), never a claim by itself; a lever claim needs a replicate against the rig's own seed floor. Every number printed carries "single head seed" until S1-R.

### 7.3 Bars (unchanged from the card and the draft)
For each new arm X in {HB-s0, HC-s0}, on the same 3,207 H48 frames: **S1-P1 >= 0.80** (point decides, as the card words it; city-reweighted to navtest's mix, x in [0, 40) m, `logit > 0`); **S1-Q >= 0.90** (AUC of the head's d_k on the same proposals); **S1-F**: trained minus its OWN initial head, both 95 % lower bounds > 0; **C-SHUF**: X-SHUF's S1-P1 < 0.50; **C-FLIP**: X's y-flipped S1-Q < 0.80. **X-LIFT** = both paired deltas (S1-P1 and S1-Q, X minus HA-s0) have 95 % lower bound > 0.
HC's S1-P1 is read on the **2 m reference grid** (the four 1 m logits of a 2 m cell averaged in probability space, thresholded at 0.5; a fixed rule, not tuned) so it stays comparable to HA-s0; **HC's S1-Q uses the 1 m mask**; the 1 m IoU against a 1 m label mask is reported, not deciding. The frozen reader is located with `S1_READ_V1=<...>/vdac/s1a_read/s1_read.py`.

### 7.4 Freeze order for the read
The reader's own freeze record (a `READER_FREEZE`-style file: sha256 of `s1_read_v2.py`, its tests, the manifest; the regression identity run on the S1a inputs; its mutation arms `no_reweight`, `frame_resample`, `unpaired_delta` still RED) is written **before the FIRST arm row is read**, not before the launch: nothing trained here reads a number from the reader, and the trainer's own `aux_eval` is a watch-curve that decides nothing. The manifest (arm table) is fixed by section 7.5 BEFORE the pulled files are opened.

### 7.5 Init heads for S1-F (pre-registered so no choice is made after the data)
* **HB:** HB's own `scores/head_pred_step0.npz` (the 1,705 SFT-held-out frames, dumped by the trainer at step 0) for the held-out file; for the 24 extra logs (1,502 frames) the HA-s0 initial-head file `head_init0_b.npz` (sha256 above), valid **because HB's head is the identical seed-0 head and its step-0 ctx is bit-identical to P6's (G21)**. Precondition: HB's `head_pred_step0` equals HA's `head_init0_a` on mask sign at >= 99.9 % of cells; if it does not, S1-F for HB is read on the 1,705-frame subset only and says so.
* **HC:** the head is a different object (cell 1.0), so the 1,705-frame `scores/head_pred_step0.npz` of HC is the init file. The 1,502 extra-log frames need `vdac_stage0_forward.py --mode init` with `expected_aux_cfg` passing `cell` (not built, section 9.4). **Pre-registered fallback: HC's S1-F is read on the 1,705 SFT-held-out frames, stated in the verdict row**, unless the fix lands before the read; the verdict rule is unchanged. UNVERIFIED that `s1_read_v2` accepts an init file covering only the held-out group.

### 7.6 Not deciding, reported when available
In-sample IoU of HB / HC (needs H10 and the `aux_drv/2` loader; else "UNAVAILABLE, the loader is not built"); per city and per band; near-boundary SDF MAE (S1a: 1.63 m, RMS 2.36 m); sigma_b; in-sample minus held-out gap.

## 8. Outcomes, committed before any data (Rule Zero: every row names its successor)

| id | condition (arm X) | next (committed) |
|---|---|---|
| **X-GO** | S1-P1, S1-Q, S1-F hold, controls valid | **S1b for X** (seed 1, ~HB 4.4 to 7.6, HC 4.3 L40-h + chain, the PI's pod time; **not in the 12.8 to 16.7 approval**), then **S2** under the card's table with X's head replacing HA-s0 (VDAC-V0T primary, bars P1 0.4971 / P1r / M / G1-G4 unchanged, K + 8 for the substituted arm set; HA arms are not read on E3). If both are X-GO, **both** go to S2, the one with the higher S1-Q point is named primary before S2, the other is a reported comparison. |
| **X-COMP** | S1-Q holds, S1-P1 fails, LV >= 0.80, a non-LV city < 0.70 | HA-CB (city-balanced draws, the card's registered fallback) on X, its own seed pair |
| **X-BOUND** | S1-P1 holds, S1-Q fails | HBC (1 m cells + boundary weight on the LoRA copy), ~HB per arm + 1.1 L40-h |
| **X-LIFT** | both bars fail, X-LIFT holds | the lever is real and short of the bar: (a) both HB and HC lift: **HBC**; (b) one lifts: that arm at **20,000 updates** (the held-out curve was still rising; the one registered training-length arm), one seed first; PI pod time |
| **X-NULL** | both bars fail, no separated lift on either delta | perception of the DAC from P6's single-trunk features is not moved by this lever. If every arm run is X-NULL: **the DAC-only perception line is closed on these levers**; next is the agent check (D-VDAC-6; label build from NAVSIM's agent annotations, NC 1.12 % + TTC 1.527 points, the larger term by `FEASIBILITY.md` F5), with its zero-GPU ceiling read first |
| **INVALID** | C-SHUF or C-FLIP fails for X, a launch-gate fact contradicts the arm, or the reader's regression identity fails | fix, re-run the affected arm; no row counts |
| **NOT RUN** | an arm is blocked at preflight, gate or timeout (section 10) | the row stands as "NOT RUN: <stage, reason>", **never read as X-NULL**; the blocker's smallest fix (section 9) is the next step; the other arm of the pair is not held up |

**Branch read.** The pair is read together, each arm against this table. If only one arm is X-GO only that arm continues; the other's row stands as written. Reported with every number: the A3 disclosures (H48 in P6's own training set; the 60,000-set bank draw with `--epochs 2`; the SHUF key map), the A4 G20 history, and "single head seed" until S1-R.

### 8.1 How P6's trunk is proved untouched (asserted by the gate, not read off the code)
(1) structure: every P6 parameter `requires_grad False` except the scorer copies (`scorer_finetune.py:1739-1747`), the copy is a different object; (2) gradient: after a real head backward every P6 parameter outside `SCORER` has `.grad is None` or exactly 0.0 while the copy's 3,145,728 LoRA tensors read non-zero, and with `--w-aux-drv 0` they read exactly 0.0; (3) state: the sha256 of `model.state_dict()` before equals after, and `model_final.pt` `968a469d...` is re-hashed by the chain; (4) function: P6's forward reproduces `proposals_p6final.npz` `logits` bit-identically on the verification frames (**the function fact is NOT built into G21**: BUILD4 section 5; it is read at S2 by G0-ID, UNVERIFIED at training time).

## 9. Not built, what it blocks, the smallest fix (named, so no arm stops at a refutation of its own reading)

| # | item | blocks | smallest fix |
|---|---|---|---|
| 9.1 | HB `--batch 8 --accum 1` memory (UNMEASURED: a ViT-L trunk copy with LoRA backward at 4 x 1,920 tokens x 8 beside P6's no-grad forward, three scorer copies and the head; `model.py` has no activation checkpointing, `grep -c 'utils.checkpoint'` on `model.py`, `scorer_finetune.py`, `aux_hb.py` = 0, 0, 0, MEASURED) | **HB-s0, HB-SHUF** if the preflight refuses on memory | see 9.2 |
| 9.2 | the fallback micro-batch split | | `--batch 4 --accum 2` (effective batch 8, BN-free, the head's `aux_backward(lv, div)` already divides: `scorer_finetune.py:2995`) for BOTH HB arms: edit `vdac_argv` (`launch_gate.py:927-945`) and `vdac_pod.sh:40` (the `--batch 8 --accum 1` text), re-run `test_vdac_gate.py` and the ~69 launch_gate-reading tests, re-land, re-ship. **Pre-registered here as an implementation detail, allowed only before any HB step trains** and recorded by a one-line addendum A5a; it changes no lever, data, seed or update count. The chain's run check assumes `ceil(n/8*2/1)` draws (the run-check block of `vdac_chain.sh`) and would need the same line changed. |
| 9.3 | HB-s1 / HC-s1 replicate arms | the S1b of an X-GO arm (not today's pause) | add the two arms to `VDAC_ARMS`, `vdac_pod.sh`, `vdac_chain.sh`, `vdac_pause_p7.sh` (a stage `A5HBs1` / `A5HCs1`) with the seed 1 and the profile tests; planned only on X-GO |
| 9.4 | `expected_aux_cfg` does not pass `cell` (`vdac_stage0_forward.py:179-180`); `load_head` refuses `aux_drv/2` (`:205-206`); no H10 | the **extra-log init head of HC** (so HC's S1-F reads on 1,705 frames, section 7.5) and every in-sample number of HB / HC. **Blocks no arm and no deciding read** | pass `cell=ra.aux_drv_cell` in `expected_aux_cfg`; call `aux_hb.load_v2` from `load_head` for `/2`; both are small and CPU-testable |
| 9.5 | G-EVAL for HB / HC at S2 | S2 of any X-GO arm | the S2 tool builds the identical head and the copy (strict `aux_drv/2`) and reproduces P6's banked logits (G0-ID); built when an arm earns S2 |
| 9.6 | `vdac_pause_p7.sh:62` hard-codes `/workspace/refe-plan-vdac/refe/vdac_chain.sh` | shipping to a DIFFERENT pod directory | `${VDAC_PKG:-/workspace/refe-plan-vdac}` in that one line (a pinned file: re-land). Not needed if the A5 tree is shipped into that directory as a FRESH tree (POD_RUN_PLAN). |
| 9.7 | the chain checks the card, A1, A2 bytes only (the precheck stage of `vdac_chain.sh`), not A3, A4 or this card | nothing (an instrument gap, not a lever) | the MM ships this card's bytes into the data root and re-hashes it (POD_RUN_PLAN step 4) |

## 10. Cost, timeouts, stop rules, K ledger

| stage | arm | L40-h lower / upper (ESTIMATED, draft section 5) | pause timeout |
|---|---|---|---|
| `A5HB` | HB-s0 | 4.4 / 7.6 (lower = P6's own step cost x 80,000 draws; upper = HA 3.20 + a full P6-rate copy fwd+bwd 4.38) | 36,000 s (`vdac_pause_p7.sh:59`) |
| `A5HB` | HB-SHUF | 0.88 / 1.52 (0.2 x HB-s0) | 14,400 s |
| `A5HC` | HC-s0 | 4.3 (HA 3.20 + 1.1, the card's 4x-queries estimate) | 21,600 s |
| `A5HC` | HC-SHUF | 0.86 | 7,200 s |
| chain overhead | 4 arms (+ the one-time export 0.1) | 2.2 | |
| **total after Stage 0** | | **about 12.7 / 16.5** (the PI's approval 12.8 to 16.7 includes Stage 0's 0.12 to 0.25) | |
| conditional, later, NOT approved | HB-s1 / HC-s1 / S2 forward of an HB head | 5.0 to 8.2 / 4.8 / 0.44 | |

The one unmeasured quantity is the step cost and memory of the trunk copy. **The PI approves the UPPER bound as the cap.** The first pod preflight prints `gpu_mem_gb` and `s_per_microbatch`; the MM sends them to the PI before HB-s0 trains if the extrapolated cost exceeds 7.6 L40-h (the optional preflight-only pause of POD_RUN_PLAN step 6 exists for this).
**Stop rules.** (1) Any chain refusal, a missing `ZZVDAC_CHAIN_DONE <arm>`, or a timeout: that stage's remaining arms are NOT run and P7 resumes (`vdac_pause_p7.sh:65`, EXIT trap); the arm's row is NOT RUN. No partial read of a timed-out arm. (2) A hit of `HOLD_P7` keeps P7 held (`vdac_pause_p7.sh:31`); the MM clears it. (3) No lever, seed, update count or bar is changed after any number from these arms is seen; the only pre-registered change is 9.2, before training. (4) A killed arm is archived and restarts at update 0 under a fresh preflight and token (`vdac_chain.sh` (the `trainstart` stage, `refuse` at ~l.297); the trainer has no resume). (5) If both HB-s0 and HC-s0 are NOT RUN the pair is unread and the card's S1 branch is NOT advanced.
**K ledger.** S0 and S1 are navtrain reads and are **not keyed** (`PREREG_VDAC_P6.md` section 6; A2 section 5: "the ledger is written by the reader at the real read only"). These four arms are S1 reads on H48 (navtrain), so **no K entry is made at freeze and none is made by their reads**; `K` is the count of NAVTEST selector reads and feeds the Bonferroni interval, so an unread entry would inflate it. Keys are written at S2 only: one per arm read on E3 (`VDAC-<arm>:<sha256 of the per-token selection scores [12146, 64] float32>:E3`; K + 8 for the substituted arm set). `LEDGER_ENTRY.json` in the freeze package is the entry for the MM to append if the MM overrules this reading.

## 11. The four metric families and the tier (binding, 2026-08-02)

Tier: **T0-like perception** (head IoU and AUC on banked navtrain H48 frames from the head's own forward pass); **never driving performance**, never a T1 claim. The four families apply to the **S2 read** (a PDMS-selection read on navtest), not to S1; at S1 each is stated per family:

| family | at S1 (this card) | at S2 (if an arm earns it) |
|---|---|---|
| LONGITUDINAL | not computable: the head predicts a drivable-area mask; n = 0; reason stated | target-speed accuracy and distance-keeping, per family, paired |
| LATERAL | not computable at S1 for the same reason; the head's near-boundary SDF MAE is a perception proxy reported, not the lateral family | heading, curvature, yaw-rate, cross-track |
| TACTICAL | not computable at S1 | selection quality: random / P6 / best-of-64 / arm; command adherence (`command_adherence_p6final.json` method) |
| STRATEGIC | **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25): not a pass, not omitted, not extended to the other three | PAUSED |
| perception (T0-like) | IoU, near-boundary MAE, AUC, per city and band, S1-P1 / S1-Q / S1-F, the paired delta to HA-s0 | from the same forward pass; never deciding |

Per-family, never pooled; each carries its estimator (section 7.2) and CI on the same windows.

## 12. Information-disjointness (the card's section 6, unchanged)
The head reads images and the camera calibration table only (the same inputs P6's `visual` reads); no ego, command, map, route or situation-classifier output at inference. G19 enforces it, now also through the trunk copy (section 6). Labels are built offline from the map and are admissible there; inference is vision-only (PI 2026-08-03).

## 13. Freeze record

* `frozen_utc: 2026-11-12T21:30:35Z`
* Frozen against the code and data pins of sections 3 and 4 at commit `a01be4a4`; the landed commit that carries this card must hold those `refe/` blobs byte for byte.
* This card's own sha256 is recorded OUTSIDE these bytes, in `PREREG_VDAC_P6_A5.sha256` beside the card (no K-ledger entry: these are unkeyed S1 navtrain reads, section 10; MM decision 2026-11-12). The placeholder `<CARD_SHA256>` in this file marks where it would be, is a pointer, and stays as written.
* Frozen by the Master Mind; PI pod approval 2026-11-12 (D4).
